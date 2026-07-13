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

"""Test doubles for the depth package.

Lives in an importable module (not a test file) because ModuleCoordinator
deploys modules into worker processes — classes must be picklable by
reference.
"""

from __future__ import annotations

import time

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.depth.estimator import DepthEstimator

TEST_W, TEST_H = 64, 48
TEST_FX = 60.0
# FakeEstimator encodes the fx it was handed into its output depth so tests
# can verify intrinsics plumbing black-box (the module runs in a worker
# process; in-process spies are unreachable).
FX_TO_DEPTH = 1.0 / 30.0


class FakeEstimator(DepthEstimator):
    name = "fake"
    outputs_metric = True

    def infer(self, color_rgb: np.ndarray, fx: float) -> tuple[np.ndarray, np.ndarray]:
        h, w = color_rgb.shape[:2]
        depth = np.full((h, w), float(fx) * FX_TO_DEPTH, dtype=np.float32)
        return depth, np.ones_like(depth)


class SyntheticCameraModule(Module):
    """Publishes N synthetic color frames + camera_info on demand (via RPC)."""

    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def emit(self, n: int, with_info: bool = True) -> None:
        info = CameraInfo.from_intrinsics(
            fx=TEST_FX, fy=TEST_FX, cx=TEST_W / 2, cy=TEST_H / 2,
            width=TEST_W, height=TEST_H, frame_id="camera_optical",
        )
        rgb = np.zeros((TEST_H, TEST_W, 3), dtype=np.uint8)
        rgb[:, :, 1] = 200
        for _ in range(n):
            ts = time.time()
            if with_info:
                self.camera_info.publish(info.with_ts(ts))
            self.color_image.publish(Image.from_numpy(
                rgb, format=ImageFormat.RGB, frame_id="camera_optical", ts=ts))
            time.sleep(0.02)
