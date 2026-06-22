"""Pluggable visualization backends: Foxglove (via the dimos LCM bridge) and/or Rerun.

A dimos spatial pipeline normally publishes its messages to LCM topics, and a
separate ``foxglove_bridge`` converts them to a Foxglove websocket. Upstream dimos
has since moved to Rerun as its native viewer, and every message type these scripts
publish (Image, PointCloud2, CameraInfo, TFMessage, ...) now implements ``to_rerun()``.

This module lets a script visualize in Foxglove, Rerun, or both, selected at runtime
with ``--viz``. Rerun logging reuses each message's ``to_rerun()`` and mirrors
``dimos.visualization.rerun.bridge``'s ``world/<topic>`` entity-path convention, so
the two backends render the same scene from the same published messages.

Usage::

    from viz_backend import RerunViz, DualPublisher, backends_for
    lcm_on, rerun_on = backends_for(args.viz)
    rerun = RerunViz(rerun_on, app_id="my_app", save_path=args.rerun_save)
    topic = lambda name, T, **kw: DualPublisher(
        name, T, lcm_factory=(LCMTransport if lcm_on else None), rerun=rerun, **kw)
    points = topic("/points_frame", PointCloud2, to_rerun_kwargs={"mode": "points"})
    ...
    rerun.set_time(ts)
    points.publish(points_msg)   # -> LCM and/or rr.log("world/points_frame", ...)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

FOXGLOVE = "foxglove"
RERUN = "rerun"
BOTH = "both"
VIZ_CHOICES = (FOXGLOVE, RERUN, BOTH)


def backends_for(viz: str) -> tuple[bool, bool]:
    """Return (foxglove/LCM enabled, rerun enabled) for a ``--viz`` choice."""
    return viz in (FOXGLOVE, BOTH), viz in (RERUN, BOTH)


class RerunViz:
    """Thin wrapper around a Rerun recording stream.

    A disabled instance is a no-op, so call sites need no ``if`` guards. Logging
    never raises — visualization must not crash the pipeline.
    """

    def __init__(
        self,
        enabled: bool,
        *,
        app_id: str = "dimos_spatial",
        save_path: str | None = None,
        connect: bool = False,
    ) -> None:
        self.enabled = enabled
        self._rr = None
        if not enabled:
            return
        import rerun as rr

        self._rr = rr
        rr.init(app_id)
        if save_path:
            rr.save(str(save_path))
            sink = f"save -> {save_path}"
        elif connect:
            rr.connect_grpc()
            sink = "connect (existing viewer)"
        else:
            rr.spawn()
            sink = "spawn (new viewer)"
        logger.info("Rerun viz enabled (%s): %s", app_id, sink)

    def set_time(self, ts: float) -> None:
        """Set the active timeline position so a frame's logs share a timestamp."""
        if self.enabled:
            self._rr.set_time("ts", timestamp=ts)

    def log(self, entity_path: str, msg: Any, **to_rerun_kwargs: Any) -> None:
        """Log a dimos message via its ``to_rerun()`` method.

        No-op if disabled or the message type has no ``to_rerun`` (e.g. Foxglove-only
        messages). Handles both single archetypes and ``RerunMulti`` lists of
        ``(entity_path, archetype)`` tuples.
        """
        if not self.enabled:
            return
        to_rerun = getattr(msg, "to_rerun", None)
        if to_rerun is None:
            return
        try:
            data = to_rerun(**to_rerun_kwargs)
        except Exception as e:  # noqa: BLE001 - viz must never crash the pipeline
            logger.debug("to_rerun failed for %s: %s", entity_path, e)
            return
        try:
            if isinstance(data, list):  # RerunMulti: [(path, archetype), ...]
                for path, arch in data:
                    self._rr.log(path, arch)
            elif data is not None:
                self._rr.log(entity_path, data)
        except Exception as e:  # noqa: BLE001
            logger.debug("rr.log failed for %s: %s", entity_path, e)

    def log_pointcloud(self, entity_path: str, pc2: Any, radius: float = 0.01) -> None:
        """Log a dimos PointCloud2 as ``rr.Points3D`` preserving its real RGB.

        Unlike ``PointCloud2.to_rerun()`` (which height-colormaps via class_ids and
        draws fat voxel-sized spheres), this uses the cloud's own per-point colors
        (``as_numpy()`` -> [0,1] RGB) and small points, so the cloud looks like the
        real scene. Falls back to uncolored points if the cloud carries no colors.
        """
        if not self.enabled:
            return
        try:
            points, colors = pc2.as_numpy()
        except Exception as e:  # noqa: BLE001
            logger.debug("pointcloud as_numpy failed for %s: %s", entity_path, e)
            return
        if points is None or len(points) == 0:
            return
        kwargs: dict[str, Any] = {"radii": radius}
        if colors is not None and len(colors) == len(points):
            kwargs["colors"] = (np.clip(np.asarray(colors), 0.0, 1.0) * 255).astype(np.uint8)
        try:
            self._rr.log(entity_path, self._rr.Points3D(points, **kwargs))
        except Exception as e:  # noqa: BLE001
            logger.debug("rr points3d log failed for %s: %s", entity_path, e)

    def log_detections_2d(self, entity_path: str, detections: Any) -> None:
        """Overlay 2D detection boxes on an image entity.

        Rerun analogue of the (upstream-removed) Foxglove ImageAnnotations. Reads
        ``.bbox = (x1, y1, x2, y2)`` and ``.name`` off each detection. Log to the
        SAME entity as the image (e.g. ``world/color_image``) so the boxes overlay
        it in one 2D view. Logs an empty Boxes2D when there are no detections so
        stale boxes from a prior frame are cleared.
        """
        if not self.enabled:
            return
        dets = getattr(detections, "detections", None) or []
        if not dets:
            try:
                self._rr.log(entity_path, self._rr.Boxes2D(mins=[], sizes=[]))
            except Exception as e:  # noqa: BLE001
                logger.debug("rr boxes2d clear failed: %s", e)
            return
        mins, sizes, labels = [], [], []
        for d in dets:
            box = getattr(d, "bbox", None)
            if box is None:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except Exception:  # noqa: BLE001
                continue
            mins.append([x1, y1])
            sizes.append([x2 - x1, y2 - y1])
            labels.append(getattr(d, "name", "") or "")
        if not mins:
            return
        try:
            self._rr.log(entity_path, self._rr.Boxes2D(mins=mins, sizes=sizes, labels=labels))
        except Exception as e:  # noqa: BLE001
            logger.debug("rr boxes2d log failed: %s", e)

    def log_object_boxes(self, entity_path: str, objects: list[Any]) -> None:
        """Rerun analogue of the Foxglove SceneUpdate cubes for tracked objects.

        Reads ``center``/``size``/``name`` off each Object and logs an ``rr.Boxes3D``.
        """
        if not self.enabled or not objects:
            return
        centers, half_sizes, labels = [], [], []
        for o in objects:
            c, s = getattr(o, "center", None), getattr(o, "size", None)
            if c is None or s is None:
                continue
            centers.append([c.x, c.y, c.z])
            half_sizes.append([s.x / 2.0, s.y / 2.0, s.z / 2.0])
            labels.append(getattr(o, "name", "") or "")
        if not centers:
            return
        try:
            self._rr.log(
                entity_path,
                self._rr.Boxes3D(centers=centers, half_sizes=half_sizes, labels=labels),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("rr boxes log failed: %s", e)


class DualPublisher:
    """Publishes a dimos message to LCM (the Foxglove path) and/or Rerun.

    Exposes ``.lcm`` (the underlying transport, or ``None`` when Foxglove is off) so
    existing teardown that calls ``topic.lcm.stop()`` keeps working unchanged.
    """

    def __init__(
        self,
        topic: str,
        msg_type: Any,
        *,
        lcm_factory: Callable[[str, Any], Any] | None = None,
        rerun: RerunViz | None = None,
        entity_path: str | None = None,
        to_rerun_kwargs: dict | None = None,
        rerun_as: str | None = None,
        rerun_radius: float = 0.01,
    ) -> None:
        self.topic = topic
        self.lcm = lcm_factory(topic, msg_type) if lcm_factory is not None else None
        self._rr = rerun
        self._entity = entity_path or ("world" + topic)
        self._kw = to_rerun_kwargs or {}
        self._rerun_as = rerun_as          # "pointcloud" -> log real RGB via log_pointcloud
        self._rerun_radius = rerun_radius

    def publish(self, msg: Any) -> None:
        if self.lcm is not None:
            self.lcm.publish(msg)
        if self._rr is not None:
            if self._rerun_as == "pointcloud":
                self._rr.log_pointcloud(self._entity, msg, radius=self._rerun_radius)
            else:
                self._rr.log(self._entity, msg, **self._kw)
