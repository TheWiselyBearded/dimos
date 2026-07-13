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

"""Reachy Mini camera as a dimos Module.

Publishes ``color_image`` + ``camera_info`` like any dimos camera, plus the
kinematic head pose as a per-frame ``reachy_base -> camera_optical`` TF —
i.e. the native-graph replacement for the ``reachy_replay_spatial_foxglove``
VideoSource monkey-patch. Compose with ``DepthEstimationModule`` for the
RGB -> depth -> camera-frame cloud path; a mapper can fuse the clouds into the
Z-up ``reachy_base`` frame via the published TF.

Run the demo composition on a recording:

    python -m dimos.robot.reachy.module --recording-dir ~/Downloads/reachy_recordings/reachy_trial4 \\
        [--camera-info reachy_mini_intrinsics.json] [--live]
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import reactivex as rx

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.spec import CameraHardware
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.reachy.camera import ReachyCamera
from dimos.spec import perception
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ReachyCameraModuleConfig(ModuleConfig):
    # Callable so replay vs live is a config choice:
    #   hardware=ReachyCamera (default, live SDK) or
    #   hardware=lambda: ReachyReplayCamera(recording_dir=...)
    hardware: Callable[[], CameraHardware] | CameraHardware = ReachyCamera
    camera_info_interval_s: float = 1.0


class ReachyCameraModule(Module, perception.Camera):
    """color_image + camera_info + reachy_base->camera_optical TF."""

    config: ReachyCameraModuleConfig

    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hardware: CameraHardware | None = None
        self._latest_image: Image | None = None

    @rpc
    def start(self) -> None:
        super().start()

        if callable(self.config.hardware):
            self.hardware = self.config.hardware()
        else:
            self.hardware = self.config.hardware

        def on_image(image: Image) -> None:
            self._latest_image = image
            self.color_image.publish(image)

        def on_pose(tf: Transform) -> None:
            self.tf.publish(tf)

        self.register_disposable(self.hardware.image_stream().subscribe(on_image))
        pose_stream = getattr(self.hardware, "pose_stream", None)
        if pose_stream is not None:
            self.register_disposable(pose_stream().subscribe(on_pose))
        else:
            logger.warning("hardware has no pose_stream; publishing images only")

        self.register_disposable(
            rx.interval(self.config.camera_info_interval_s).subscribe(
                lambda _: self.camera_info.publish(
                    self.hardware.camera_info.with_ts(time.time()))
            )
        )
        logger.info(f"ReachyCameraModule started ({type(self.hardware).__name__})")

    @rpc
    def stop(self) -> None:
        if self.hardware is not None and hasattr(self.hardware, "stop"):
            self.hardware.stop()
        super().stop()


def main() -> None:
    """Demo composition: Reachy camera -> DepthEstimationModule, over transport."""
    import argparse

    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from dimos.core.transport_factory import make_transport
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.perception.depth.module import DepthEstimationModule
    from dimos.robot.reachy.camera import ReachyReplayCamera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", default=None,
                        help="on-robot recording directory (omit with --live)")
    parser.add_argument("--live", action="store_true",
                        help="use the live reachy_mini SDK camera")
    parser.add_argument("--camera-info", default=None,
                        help="reachy_mini_intrinsics.json or CameraInfo YAML")
    parser.add_argument("--depth", default="da3", choices=["da3", "depthpro"])
    parser.add_argument("--da3-model", default="da3metric-large")
    parser.add_argument("--loop", action="store_true", help="loop the recording")
    args = parser.parse_args()

    if not args.live and not args.recording_dir:
        parser.error("--recording-dir is required unless --live")

    coordinator = ModuleCoordinator()
    coordinator.start()

    if args.live:
        camera = coordinator.deploy(
            ReachyCameraModule, camera_info_path=args.camera_info)
    else:
        from functools import partial

        camera = coordinator.deploy(
            ReachyCameraModule,
            hardware=partial(ReachyReplayCamera,
                             recording_dir=args.recording_dir,
                             camera_info_path=args.camera_info,
                             loop=args.loop),
        )
    depth = coordinator.deploy(
        DepthEstimationModule, estimator=args.depth, da3_model=args.da3_model)

    camera.color_image.transport = make_transport("/reachy/color_image", Image)
    camera.camera_info.transport = make_transport("/reachy/camera_info", CameraInfo)
    depth.depth_image.transport = make_transport("/reachy/depth", Image)
    depth.depth_camera_info.transport = make_transport("/reachy/depth_camera_info", CameraInfo)
    depth.pointcloud.transport = make_transport("/reachy/pointcloud", PointCloud2)
    depth.color_image.connect(camera.color_image)
    depth.camera_info.connect(camera.camera_info)

    camera.start()
    depth.start()
    print("publishing /reachy/color_image /reachy/camera_info /reachy/depth "
          "/reachy/pointcloud + reachy_base->camera_optical TF. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.stop()


if __name__ == "__main__":
    main()
