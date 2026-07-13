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

"""Headless tests of DepthEstimationModule with a fake estimator (no model load)."""

import time
from typing import Any

import numpy as np
import pytest

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.depth.module import DepthEstimationModule
from dimos.perception.depth.testing import (
    FX_TO_DEPTH,
    TEST_FX,
    TEST_H,
    TEST_W,
    FakeEstimator,
    SyntheticCameraModule,
)

EXPECTED_DEPTH = TEST_FX * FX_TO_DEPTH  # 2.0 m


@pytest.fixture()
def dimos_cluster():  # noqa: ANN201
    """Function-scoped coordinator (shadows the module-scoped conftest fixture).

    The per-test thread-leak monitor snapshots before and checks after each
    test; a module-scoped cluster's threads span tests and get flagged. A
    function-scoped cluster starts after the snapshot and stops before the
    check, so its threads (LCM loops, zenoh pyo3 closures) are torn down in
    time.
    """
    from dimos.core.coordination.module_coordinator import ModuleCoordinator

    coordinator = ModuleCoordinator()
    coordinator.start()
    try:
        yield coordinator
    finally:
        coordinator.stop()


@pytest.fixture()
def transports():  # noqa: ANN201
    """LCMTransport factory that stops every created transport at teardown.

    LCM (not the platform-default zenoh) matches the other dimos module tests:
    zenoh sessions own non-daemon threads that trip the thread-leak monitor
    and block interpreter shutdown.
    """
    created = []

    def _make(topic, mtype):  # noqa: ANN001, ANN202
        t = LCMTransport(topic, mtype)
        created.append(t)
        return t

    yield _make
    for t in created:
        try:
            t.stop()
        except Exception:  # noqa: BLE001
            pass


def _wait_for(collected: list, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline and not collected:
        time.sleep(0.1)


@pytest.mark.self_hosted
def test_depth_module_pipeline(dimos_cluster: Any, transports: Any) -> None:
    source = dimos_cluster.deploy(SyntheticCameraModule)
    depth_mod = dimos_cluster.deploy(
        DepthEstimationModule,
        estimator_factory=FakeEstimator,
        depth_far=6.0,
        edge_filter_threshold_m=0.0,  # constant depth has no edges; keep exact
        publish_pointcloud=True,
    )
    source.color_image.transport = transports("/test_depth/color", Image)
    source.camera_info.transport = transports("/test_depth/color_info", CameraInfo)
    depth_mod.depth_image.transport = transports("/test_depth/depth", Image)
    depth_mod.depth_camera_info.transport = transports(
        "/test_depth/depth_info", CameraInfo)
    depth_mod.pointcloud.transport = transports("/test_depth/cloud", PointCloud2)
    depth_mod.color_image.connect(source.color_image)
    depth_mod.camera_info.connect(source.camera_info)

    got_depth: list = []
    got_info: list = []
    got_cloud: list = []
    depth_mod.depth_image.subscribe(got_depth.append)
    depth_mod.depth_camera_info.subscribe(got_info.append)
    depth_mod.pointcloud.subscribe(got_cloud.append)

    source.start()
    depth_mod.start()
    source.emit(8)

    _wait_for(got_cloud)

    assert got_depth, "no depth image published"
    assert got_info, "no depth camera_info published"
    assert got_cloud, "no pointcloud published"

    d = got_depth[-1]
    assert d.width == TEST_W and d.height == TEST_H
    d_np = d.as_numpy()
    assert d_np.dtype == np.float32
    # FakeEstimator returns depth = fx/30 — equality proves the real fx from
    # camera_info reached the estimator.
    assert np.allclose(d_np, EXPECTED_DEPTH)
    assert d.frame_id == "camera_optical"

    info = got_info[-1]
    assert info.width == TEST_W and info.height == TEST_H
    assert abs(info.K[0] - TEST_FX) < 1e-6

    # Cloud sanity: constant-depth plane at z=EXPECTED_DEPTH in the camera frame
    pts, _cols = got_cloud[-1].as_numpy()
    assert len(pts) > 0
    assert np.allclose(pts[:, 2], EXPECTED_DEPTH, atol=1e-3)
    assert got_cloud[-1].frame_id == "camera_optical"


@pytest.mark.self_hosted
def test_depth_module_drops_frames_without_camera_info(dimos_cluster: Any, transports: Any) -> None:
    source = dimos_cluster.deploy(SyntheticCameraModule)
    depth_mod = dimos_cluster.deploy(
        DepthEstimationModule, estimator_factory=FakeEstimator)
    source.color_image.transport = transports("/test_noinfo/color", Image)
    depth_mod.depth_image.transport = transports("/test_noinfo/depth", Image)
    depth_mod.color_image.connect(source.color_image)
    # camera_info deliberately NOT connected

    got_depth: list = []
    depth_mod.depth_image.subscribe(got_depth.append)

    source.start()
    depth_mod.start()
    source.emit(3, with_info=False)
    time.sleep(1.0)

    assert not got_depth, "depth published without intrinsics"
