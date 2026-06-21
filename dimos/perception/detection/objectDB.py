# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.perception.detection.type.detection3d.object import Object

logger = setup_logger()


class ObjectDB:
    """Spatial memory database for 3D object detections.

    Maintains two tiers of objects internally:
    - _pending_objects: Recently detected objects (detection_count < threshold)
    - _objects: Confirmed permanent objects (detection_count >= threshold)

    Deduplication uses two heuristics:
    1. track_id match from YOLOE tracker (recent match)
    2. Center distance within threshold (spatial proximity match)
    """

    def __init__(
        self,
        distance_threshold: float = 0.2,
        min_detections_for_permanent: int = 6,
        pending_ttl_s: float = 5.0,
        track_id_ttl_s: float = 5.0,
        class_aware_matching: bool = True,
        frustum_match_pixel_threshold: float = 60.0,
        enable_decay: bool = True,
        confidence_init: float = 0.5,
        confidence_step_up: float = 0.10,
        confidence_step_down: float = 0.05,
        center_from_accumulated_cloud: bool = True,
    ) -> None:
        self._distance_threshold = distance_threshold
        self._min_detections = min_detections_for_permanent
        self._pending_ttl_s = pending_ttl_s
        self._track_id_ttl_s = track_id_ttl_s
        self._class_aware_matching = class_aware_matching
        self._frustum_match_pixel_threshold = frustum_match_pixel_threshold
        self._enable_decay = enable_decay
        self._confidence_init = confidence_init
        self._confidence_step_up = confidence_step_up
        self._confidence_step_down = confidence_step_down
        # True: object center = accumulated-cloud centroid (stable, for static-scene
        # mapping / spatial memory). False: center = latest detection (tracks motion,
        # for live navigation around moving objects). See Object.update_object.
        self._center_from_accumulated_cloud = center_from_accumulated_cloud

        # Internal storage - keyed by object_id
        self._pending_objects: dict[str, Object] = {}
        self._objects: dict[str, Object] = {}  # Permanent objects

        # track_id -> object_id mapping for fast lookup
        self._track_id_map: dict[int, str] = {}
        self._last_add_stats: dict[str, int] = {}

        # Per-object [0, 1] confidence used for decay-driven removal. Bumped on
        # observation, decayed when the object's center projects into the frustum
        # but no detection of the same class lands near it.
        self._confidence: dict[str, float] = {}

        self._lock = threading.RLock()

    # ─────────────────────────────────────────────────────────────────
    # Public Methods
    # ─────────────────────────────────────────────────────────────────

    def add_objects(
        self,
        objects: list[Object],
        *,
        c2w: np.ndarray | None = None,
        K: np.ndarray | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Object]:
        """Add multiple objects to the database with deduplication.

        Args:
            objects: List of Object instances from object_scene_registration
            c2w, K, image_width, image_height: Optional camera context. When all
                are provided, the matcher will additionally try a 2D-pixel fallback
                — projecting each existing object's center into the current image
                and merging if a same-class new detection's bbox center lands within
                ``frustum_match_pixel_threshold`` pixels. This recovers matches when
                visual odometry drift has shifted the world-frame position by more
                than ``distance_threshold``.

        Returns:
            List of updated/created Object instances
        """
        stats = {
            "input": len(objects),
            "created": 0,
            "updated": 0,
            "promoted": 0,
            "matched_track": 0,
            "matched_distance": 0,
            "matched_pixel": 0,
        }

        results: list[Object] = []
        now = time.time()
        camera_ctx = self._build_camera_context(c2w, K, image_width, image_height)
        with self._lock:
            self._prune_stale_pending(now)
            for obj in objects:
                matched, reason = self._match(obj, now, camera_ctx)
                if matched is None:
                    inserted = self._insert_pending(obj, now)
                    self._confidence[inserted.object_id] = self._confidence_init
                    results.append(inserted)
                    stats["created"] += 1
                    continue

                self._update_existing(matched, obj, now)
                self._bump_confidence(matched.object_id)
                results.append(matched)
                stats["updated"] += 1
                if reason == "track":
                    stats["matched_track"] += 1
                elif reason == "distance":
                    stats["matched_distance"] += 1
                elif reason == "pixel":
                    stats["matched_pixel"] += 1
                if self._check_promotion(matched):
                    stats["promoted"] += 1

        stats["pending"] = len(self._pending_objects)
        stats["permanent"] = len(self._objects)
        self._last_add_stats = stats
        if stats["created"] > 0 or stats["promoted"] > 0:
            logger.info(f"ObjectDB: {stats}")
        return results

    def get_last_add_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_add_stats)

    def get_objects(self) -> list[Object]:
        """Get all permanent objects (detection_count >= threshold)."""
        with self._lock:
            return list(self._objects.values())

    def get_all_objects(self) -> list[Object]:
        """Get all objects (both pending and permanent)."""
        with self._lock:
            return list(self._pending_objects.values()) + list(self._objects.values())

    def promote(self, object_id: str) -> bool:
        """Promote an object from pending to permanent."""
        with self._lock:
            if object_id in self._pending_objects:
                self._objects[object_id] = self._pending_objects.pop(object_id)
                return True
            return object_id in self._objects

    def find_by_name(self, name: str) -> list[Object]:
        """Find all permanent objects with matching name."""
        with self._lock:
            return [obj for obj in self._objects.values() if obj.name == name]

    def find_by_object_id(self, object_id: str) -> Object | None:
        """Find an object by its object_id (searches pending and permanent)."""
        with self._lock:
            if object_id in self._objects:
                return self._objects[object_id]
            if object_id in self._pending_objects:
                return self._pending_objects[object_id]
            return None

    def find_nearest(
        self,
        position: Vector3,
        name: str | None = None,
    ) -> Object | None:
        """Find nearest permanent object to a position, optionally filtered by name.

        Args:
            position: Position to search from
            name: Optional name filter

        Returns:
            Nearest Object or None if no objects found
        """
        with self._lock:
            candidates = [
                obj
                for obj in self._objects.values()
                if obj.center is not None and (name is None or obj.name == name)
            ]

            if not candidates:
                return None

            return min(candidates, key=lambda obj: position.distance(obj.center))

    def clear(self) -> None:
        """Clear all objects from the database."""
        with self._lock:
            # Drop Open3D pointcloud references before clearing to reduce shutdown warnings.
            for obj in list(self._pending_objects.values()) + list(self._objects.values()):
                obj.pointcloud = PointCloud2(
                    pointcloud=o3d.geometry.PointCloud(),
                    frame_id=obj.pointcloud.frame_id,
                    ts=obj.pointcloud.ts,
                )
            self._pending_objects.clear()
            self._objects.clear()
            self._track_id_map.clear()
            logger.info("ObjectDB cleared")

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the database."""
        with self._lock:
            return {
                "pending_count": len(self._pending_objects),
                "permanent_count": len(self._objects),
                "total_count": len(self._pending_objects) + len(self._objects),
            }

    def decay_unobserved(
        self,
        observed_ids: set[str],
        *,
        c2w: np.ndarray,
        K: np.ndarray,
        image_width: int,
        image_height: int,
        decay_step: float | None = None,
    ) -> list[str]:
        """Decay confidence of any object whose center projects inside the
        camera frustum but isn't in ``observed_ids``. Objects whose confidence
        falls to 0 or below are deleted entirely (from both tiers and the
        track_id map).

        Args:
            observed_ids: object_ids that were matched/inserted this frame.
                Anything else that's *visible* counts as a non-observation.
            c2w, K, image_width, image_height: current camera context.
            decay_step: override the configured ``confidence_step_down`` for
                this call (useful for variable framerates).

        Returns:
            List of object_ids that were deleted.
        """
        if not self._enable_decay:
            return []

        step = self._confidence_step_down if decay_step is None else decay_step
        deleted: list[str] = []
        with self._lock:
            all_items = list(self._objects.items()) + list(self._pending_objects.items())
            for obj_id, obj in all_items:
                if obj_id in observed_ids or obj.center is None:
                    continue
                pix = self._project_world_to_pixel(
                    obj.center, c2w, K, image_width, image_height
                )
                if pix is None:
                    continue  # not in frustum; absence is not evidence
                cur = self._confidence.get(obj_id, self._confidence_init)
                new_conf = cur - step
                if new_conf <= 0.0:
                    self._delete_object(obj_id)
                    deleted.append(obj_id)
                else:
                    self._confidence[obj_id] = new_conf
        return deleted

    def to_state(self) -> dict:
        """Snapshot the database for save/load. Object instances are kept as-is
        so the bundle round-trips exactly when re-pickled."""
        with self._lock:
            return {
                "pending": dict(self._pending_objects),
                "permanent": dict(self._objects),
                "track_id_map": dict(self._track_id_map),
                "confidence": dict(self._confidence),
                "config": {
                    "distance_threshold": self._distance_threshold,
                    "min_detections": self._min_detections,
                    "pending_ttl_s": self._pending_ttl_s,
                    "track_id_ttl_s": self._track_id_ttl_s,
                    "class_aware_matching": self._class_aware_matching,
                    "frustum_match_pixel_threshold": self._frustum_match_pixel_threshold,
                    "enable_decay": self._enable_decay,
                    "confidence_init": self._confidence_init,
                    "confidence_step_up": self._confidence_step_up,
                    "confidence_step_down": self._confidence_step_down,
                },
            }

    def load_state(self, state: dict, replace: bool = True) -> None:
        """Repopulate from a dict produced by ``to_state``.

        Configuration knobs (thresholds, decay rates) are *not* restored from
        the bundle — the caller's constructor args win. Only the object data
        and track/confidence maps are loaded.
        """
        with self._lock:
            if replace:
                self._pending_objects.clear()
                self._objects.clear()
                self._track_id_map.clear()
                self._confidence.clear()
            self._pending_objects.update(state.get("pending", {}))
            self._objects.update(state.get("permanent", {}))
            self._track_id_map.update(state.get("track_id_map", {}))
            self._confidence.update(state.get("confidence", {}))

    def _delete_object(self, object_id: str) -> None:
        """Remove an object from both tiers and any track_id mapping."""
        if object_id in self._pending_objects:
            del self._pending_objects[object_id]
        if object_id in self._objects:
            del self._objects[object_id]
        self._confidence.pop(object_id, None)
        for track_id, mapped_id in list(self._track_id_map.items()):
            if mapped_id == object_id:
                del self._track_id_map[track_id]

    # ─────────────────────────────────────────────────────────────────
    # Internal Methods
    # ─────────────────────────────────────────────────────────────────

    def _match(
        self,
        obj: Object,
        now: float,
        camera_ctx: dict[str, Any] | None = None,
    ) -> tuple[Object | None, str | None]:
        if obj.track_id >= 0:
            matched = self._match_by_track_id(obj.track_id, now)
            if matched is not None:
                return matched, "track"

        matched = self._match_by_distance(obj)
        if matched is not None:
            return matched, "distance"

        if camera_ctx is not None:
            matched = self._match_by_pixel(obj, camera_ctx)
            if matched is not None:
                return matched, "pixel"
        return None, None

    def _insert_pending(self, obj: Object, now: float) -> Object:
        if not obj.ts:
            obj.ts = now
        self._pending_objects[obj.object_id] = obj
        if obj.track_id >= 0:
            self._track_id_map[obj.track_id] = obj.object_id
        logger.info(f"Created new pending object {obj.object_id} ({obj.name})")
        return obj

    def _update_existing(self, existing: Object, obj: Object, now: float) -> None:
        existing.update_object(obj, center_from_accumulated_cloud=self._center_from_accumulated_cloud)
        existing.ts = obj.ts or now
        if obj.track_id >= 0:
            self._track_id_map[obj.track_id] = existing.object_id

    def _match_by_track_id(self, track_id: int, now: float) -> Object | None:
        """Find object with matching track_id from YOLOE."""
        if track_id < 0:
            return None

        object_id = self._track_id_map.get(track_id)
        if object_id is None:
            return None

        # Check in permanent objects first
        if object_id in self._objects:
            obj = self._objects[object_id]
        elif object_id in self._pending_objects:
            obj = self._pending_objects[object_id]
        else:
            del self._track_id_map[track_id]
            return None

        last_seen = obj.ts if obj.ts else now
        if now - last_seen > self._track_id_ttl_s:
            del self._track_id_map[track_id]
            return None

        return obj

    def _match_by_distance(self, obj: Object) -> Object | None:
        """Find object within distance threshold (and matching class, if enabled).

        When ``_class_aware_matching`` is disabled this is name-agnostic, which
        is often desirable because YOLO labels are unstable across frames — the
        same physical object may be called "sharpener" one frame and "spray can"
        the next. With a tight distance threshold, two distinct objects at the
        same spot is effectively impossible.
        """
        if obj.center is None:
            return None

        all_objects = list(self._objects.values()) + list(self._pending_objects.values())
        candidates = [
            o
            for o in all_objects
            if o.center is not None
            and obj.center.distance(o.center) < self._distance_threshold
            and (not self._class_aware_matching or o.name == obj.name)
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda o: obj.center.distance(o.center))

    def _match_by_pixel(
        self, obj: Object, camera_ctx: dict[str, Any]
    ) -> Object | None:
        """Match by reprojecting existing object centers into the current image
        and finding a same-class new detection's 2D bbox center within
        ``frustum_match_pixel_threshold`` pixels.

        Recovers matches when VO drift has moved the world-frame center beyond
        ``distance_threshold`` but the object is visually still in the same place.
        """
        try:
            new_cx, new_cy = obj.center_bbox
        except Exception:
            return None

        c2w = camera_ctx["c2w"]
        K = camera_ctx["K"]
        w = camera_ctx["w"]
        h = camera_ctx["h"]
        thresh = self._frustum_match_pixel_threshold

        best: tuple[float, Object] | None = None
        all_objects = list(self._objects.values()) + list(self._pending_objects.values())
        for o in all_objects:
            if o.center is None:
                continue
            if self._class_aware_matching and o.name != obj.name:
                continue
            pix = self._project_world_to_pixel(o.center, c2w, K, w, h)
            if pix is None:
                continue
            du = pix[0] - new_cx
            dv = pix[1] - new_cy
            d = (du * du + dv * dv) ** 0.5
            if d < thresh and (best is None or d < best[0]):
                best = (d, o)
        return best[1] if best is not None else None

    @staticmethod
    def _build_camera_context(
        c2w: np.ndarray | None,
        K: np.ndarray | None,
        image_width: int | None,
        image_height: int | None,
    ) -> dict[str, Any] | None:
        if c2w is None or K is None or image_width is None or image_height is None:
            return None
        return {
            "c2w": np.asarray(c2w, dtype=np.float64),
            "K": np.asarray(K, dtype=np.float64),
            "w": int(image_width),
            "h": int(image_height),
        }

    @staticmethod
    def _project_world_to_pixel(
        center: Vector3, c2w: np.ndarray, K: np.ndarray, w: int, h: int
    ) -> tuple[float, float] | None:
        """Project a world-frame point through w2c = inv(c2w) into pixel coords.

        Returns None if the point is behind or too close to the camera, or if the
        projected pixel falls outside the image rectangle.
        """
        p_world = np.array([center.x, center.y, center.z], dtype=np.float64)
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        p_cam = R.T @ (p_world - t)
        z = float(p_cam[2])
        if z <= 0.05:
            return None
        u = float(K[0, 0]) * float(p_cam[0]) / z + float(K[0, 2])
        v = float(K[1, 1]) * float(p_cam[1]) / z + float(K[1, 2])
        if u < 0.0 or u >= float(w) or v < 0.0 or v >= float(h):
            return None
        return u, v

    def _bump_confidence(self, object_id: str) -> None:
        cur = self._confidence.get(object_id, self._confidence_init)
        self._confidence[object_id] = min(1.0, cur + self._confidence_step_up)

    def _prune_stale_pending(self, now: float) -> None:
        if self._pending_ttl_s <= 0:
            return
        cutoff = now - self._pending_ttl_s
        stale_ids = [
            obj_id for obj_id, obj in self._pending_objects.items() if (obj.ts or now) < cutoff
        ]
        for obj_id in stale_ids:
            del self._pending_objects[obj_id]
            for track_id, mapped_id in list(self._track_id_map.items()):
                if mapped_id == obj_id:
                    del self._track_id_map[track_id]

    def _check_promotion(self, obj: Object) -> bool:
        """Move object from pending to permanent if threshold met."""
        if obj.detections_count >= self._min_detections:
            # Check if it's in pending
            if obj.object_id in self._pending_objects:
                # Promote to permanent
                del self._pending_objects[obj.object_id]
                self._objects[obj.object_id] = obj
                logger.info(
                    f"Promoted object {obj.object_id} ({obj.name}) to permanent "
                    f"with {obj.detections_count} detections"
                )
                return True
        return False

    # ─────────────────────────────────────────────────────────────────
    # Agent encoding
    # ─────────────────────────────────────────────────────────────────

    def agent_encode(self) -> list[dict[str, Any]]:
        """Encode permanent objects for agent consumption."""
        with self._lock:
            return [obj.agent_encode() for obj in self._objects.values()]

    def __len__(self) -> int:
        """Return number of permanent objects."""
        with self._lock:
            return len(self._objects)

    def __repr__(self) -> str:
        with self._lock:
            return f"ObjectDB(permanent={len(self._objects)}, pending={len(self._pending_objects)})"


__all__ = ["ObjectDB"]
