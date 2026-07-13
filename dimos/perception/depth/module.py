# Copyright 2026 Dimensional Inc.
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

"""Monocular depth estimation as a dimos Module.

Subscribes ``color_image`` (+ ``camera_info``), runs a learned depth model
(DA3 / Depth Pro) with latest-wins backpressure — frames arriving while the
model is busy are dropped, never queued — and publishes:

- ``depth_image``: float32 metres, same resolution and frame_id as the input
- ``depth_camera_info``: intrinsics rescaled to the processed resolution
- ``pointcloud``: colored camera-frame cloud via ``PointCloud2.from_rgbd``
  (transform to world downstream with the camera TF, e.g. a mapper module)
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.depth.estimator import (
    DepthEstimator,
    filter_depth_edges,
    make_depth_estimator,
    resolve_da3_defaults,
)
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()


class DepthEstimationConfig(ModuleConfig):
    estimator: str = "da3"  # "da3" | "depthpro"
    da3_model: str = "da3metric-large"
    device: str | None = None  # None = cuda > mps > cpu
    da3_conf_threshold: float | None = None  # None = per-variant default
    da3_process_res: int | None = None  # None = per-variant default
    da3_trust_is_metric: bool = False

    depth_near: float = 0.2  # metres; below → invalid (0)
    depth_far: float = 6.0  # metres; above → invalid (0)
    edge_filter_threshold_m: float = 0.10  # 0 disables the flying-pixel filter
    edge_filter_dilate_px: int = 2

    publish_pointcloud: bool = True
    pointcloud_voxel_size: float = 0.0  # metres; >0 downsamples the cloud

    # Test/DI hook: bypass model loading with a ready-made estimator.
    estimator_factory: Callable[[], DepthEstimator] | None = None


class DepthEstimationModule(Module):
    """color_image + camera_info -> depth_image + depth_camera_info + pointcloud."""

    config: DepthEstimationConfig

    color_image: In[Image]
    camera_info: In[CameraInfo]

    depth_image: Out[Image]
    depth_camera_info: Out[CameraInfo]
    pointcloud: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._estimator: DepthEstimator | None = None
        self._camera_info: CameraInfo | None = None
        self._no_info_warned = False
        self._frames = 0
        self._last_stats_t = 0.0

    @rpc
    def start(self) -> None:
        super().start()

        if self.config.estimator_factory is not None:
            self._estimator = self.config.estimator_factory()
        elif self.config.estimator == "da3":
            conf_thresh, process_res, force_relative = resolve_da3_defaults(
                self.config.da3_model,
                self.config.da3_conf_threshold,
                self.config.da3_process_res,
                self.config.da3_trust_is_metric,
            )
            self._estimator = make_depth_estimator(
                "da3",
                device=self.config.device,
                da3_model=self.config.da3_model,
                da3_conf_threshold=conf_thresh,
                da3_process_res=process_res,
                da3_force_relative=force_relative,
                depth_near=self.config.depth_near,
                depth_far=self.config.depth_far,
            )
        else:
            self._estimator = make_depth_estimator(
                self.config.estimator, device=self.config.device
            )

        if not self._estimator.outputs_metric:
            logger.warning(
                f"{self._estimator.name} is not metric — fused geometry will "
                "have arbitrary/drifting scale. Prefer depthpro or da3metric-*."
            )

        # Inputs may be legitimately unwired at start (e.g. intrinsics arrive
        # from a later-started camera); an In without transport/connection
        # can't subscribe, so degrade to dropping frames instead of crashing.
        try:
            self.camera_info.subscribe(lambda msg: setattr(self, "_camera_info", msg))
        except Exception:  # noqa: BLE001
            logger.warning("camera_info input not wired — all frames will be "
                           "dropped until intrinsics arrive on a reconnect")
        self.register_disposable(
            backpressure(self.color_image.observable()).subscribe(self._on_image)
        )
        logger.info(
            f"DepthEstimationModule started: estimator={self._estimator.name} "
            f"metric={self._estimator.outputs_metric} "
            f"pointcloud={self.config.publish_pointcloud}"
        )

    def _scaled_info(self, width: int, height: int) -> CameraInfo | None:
        """Camera intrinsics rescaled to the processed image resolution."""
        info = self._camera_info
        if info is None:
            return None
        if info.width == width and info.height == height:
            return info
        if info.width <= 0 or info.height <= 0:
            return None
        sx = width / info.width
        sy = height / info.height
        K = info.get_K_matrix()
        return CameraInfo.from_intrinsics(
            fx=K[0, 0] * sx, fy=K[1, 1] * sy,
            cx=K[0, 2] * sx, cy=K[1, 2] * sy,
            width=width, height=height,
            frame_id=info.frame_id,
        )

    def _on_image(self, image: Image) -> None:
        assert self._estimator is not None
        info = self._scaled_info(image.width, image.height)
        if info is None:
            if not self._no_info_warned:
                logger.warning("depth: color_image received but no camera_info yet; "
                               "dropping frames until intrinsics arrive")
                self._no_info_warned = True
            return

        cfg = self.config
        rgb = image.to_rgb().as_numpy()
        fx = float(info.K[0])

        t0 = time.perf_counter()
        try:
            depth_m, _conf = self._estimator.infer(rgb, fx=fx)
        except Exception:
            logger.exception(f"depth inference ({self._estimator.name}) failed")
            return
        infer_dt = time.perf_counter() - t0

        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.where(
            (depth_m >= cfg.depth_near) & (depth_m <= cfg.depth_far), depth_m, 0.0
        ).astype(np.float32)
        if depth_m.shape != (image.height, image.width):
            import cv2

            depth_m = cv2.resize(depth_m, (image.width, image.height),
                                 interpolation=cv2.INTER_NEAREST)
        edge_thresh = (0.0 if self._estimator.skip_edge_filter
                       else cfg.edge_filter_threshold_m)
        depth_m = filter_depth_edges(depth_m, edge_thresh, cfg.edge_filter_dilate_px)

        depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                     frame_id=image.frame_id, ts=image.ts)
        self.depth_image.publish(depth_msg)
        self.depth_camera_info.publish(info.with_ts(image.ts))

        if cfg.publish_pointcloud:
            color_rgb_msg = Image.from_numpy(rgb, format=ImageFormat.RGB,
                                             frame_id=image.frame_id, ts=image.ts)
            cloud = PointCloud2.from_rgbd(
                color_image=color_rgb_msg, depth_image=depth_msg, camera_info=info,
                depth_scale=1.0, depth_trunc=cfg.depth_far,
            )
            if cfg.pointcloud_voxel_size > 0:
                cloud = cloud.voxel_downsample(cfg.pointcloud_voxel_size)
            cloud.frame_id = image.frame_id
            cloud.ts = image.ts
            self.pointcloud.publish(cloud)

        self._frames += 1
        now = time.monotonic()
        if now - self._last_stats_t > 5.0:
            self._last_stats_t = now
            d_pos = depth_m[depth_m > 0]
            drange = (f"[{d_pos.min():.2f},{d_pos.max():.2f}]m" if d_pos.size else "[empty]")
            logger.info(
                f"depth: {self._frames} frames, last infer={infer_dt * 1000:.0f}ms "
                f"d={drange} fx={fx:.1f} {image.width}x{image.height}"
            )

    @rpc
    def stop(self) -> None:
        super().stop()
