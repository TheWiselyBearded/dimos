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

"""Roundtrip tests for the xr-nav bundle <-> .pc2.lcm converter."""

import pickle
from pathlib import Path

import numpy as np

from dimos.mapping.utils.xrnav import (
    BUNDLE_VERSION,
    bundle_to_pointcloud2,
    convert,
    pointcloud2_to_bundle_state,
)
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2


def _make_bundle(path: Path, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(7)
    centroids = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
    state = {
        "version": BUNDLE_VERSION,
        "saved_at": 0.0,
        "voxel_map": {
            "voxel_size": 0.05,
            "max_range": 8.0,
            "keys": np.floor(centroids / 0.05).astype(np.int64),
            "centroids": centroids,
            "colors": rng.uniform(0, 1, size=(n, 3)).astype(np.float32),
            "confidence": np.ones(n, dtype=np.float32),
            "count": np.concatenate([np.ones(n // 2, dtype=np.int32),
                                     np.full(n - n // 2, 3, dtype=np.int32)]),
            "miss_count": np.zeros(n, dtype=np.int32),
        },
        "object_db": {"pending": {}, "permanent": {}, "track_id_map": {},
                      "confidence": {}, "config": {}},
        "extra": {},
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return centroids


def test_bundle_to_pointcloud2(tmp_path: Path) -> None:
    centroids = _make_bundle(tmp_path / "map.pkl", n=500)
    cloud = bundle_to_pointcloud2(tmp_path / "map.pkl")
    pts, cols = cloud.as_numpy()
    assert len(pts) == 500
    assert np.allclose(np.sort(pts[:, 0]), np.sort(centroids[:, 0]), atol=1e-5)
    assert cols is not None and len(cols) == 500

    # min_observations filter: half the voxels have count 1
    filtered = bundle_to_pointcloud2(tmp_path / "map.pkl", min_observations=2)
    pts_f, _ = filtered.as_numpy()
    assert len(pts_f) == 250


def test_cloud_to_bundle_state() -> None:
    import open3d as o3d

    # Two clusters far apart -> two voxels at 0.5 m voxel size
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02],
                    [3.0, 3.0, 3.0], [3.01, 3.0, 3.0]], dtype=np.float64)
    cols = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    cloud = PointCloud2(pointcloud=pcd, frame_id="map", ts=0.0)

    state = pointcloud2_to_bundle_state(cloud, voxel_size=0.5)
    vox = state["voxel_map"]
    assert len(vox["keys"]) == 2
    assert sorted(vox["count"].tolist()) == [2, 2]
    order = np.argsort(vox["centroids"][:, 0])
    assert np.allclose(vox["centroids"][order[0]], [0.015, 0.015, 0.015], atol=1e-6)
    assert np.allclose(vox["colors"][order[0]], [1, 0, 0], atol=1e-6)
    assert np.allclose(vox["colors"][order[1]], [0, 1, 0], atol=1e-6)


def test_roundtrip_bundle_pc2lcm_bundle(tmp_path: Path) -> None:
    _make_bundle(tmp_path / "a.pkl", n=300)
    convert(tmp_path / "a.pkl", tmp_path / "a.pc2.lcm")
    assert (tmp_path / "a.pc2.lcm").exists()

    # The production RelocalizationModule load path:
    decoded = PointCloud2.lcm_decode((tmp_path / "a.pc2.lcm").read_bytes())
    pts, _ = decoded.as_numpy()
    assert len(pts) == 300

    # ... and back to a bundle the camera pipelines can --load-map
    convert(tmp_path / "a.pc2.lcm", tmp_path / "b.pkl", voxel_size=0.05)
    with open(tmp_path / "b.pkl", "rb") as f:
        state = pickle.load(f)
    assert state["version"] == BUNDLE_VERSION
    assert len(state["voxel_map"]["keys"]) > 0
    # xr-nav's VoxelMap can load this state directly (schema compatibility)
    import sys

    xrnav_src = Path(__file__).resolve().parents[3] / "xr-nav" / "src"
    if xrnav_src.is_dir():
        sys.path.insert(0, str(xrnav_src))
        try:
            from xr_nav.voxel_map import VoxelMap

            vm = VoxelMap()
            vm.load_state(state["voxel_map"])
            assert vm.size == len(state["voxel_map"]["keys"])
        finally:
            sys.path.remove(str(xrnav_src))


def test_ply_to_bundle(tmp_path: Path) -> None:
    import open3d as o3d

    rng = np.random.default_rng(3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.uniform(-1, 1, (200, 3)))
    o3d.io.write_point_cloud(str(tmp_path / "scan.ply"), pcd)
    convert(tmp_path / "scan.ply", tmp_path / "scan.pkl", voxel_size=0.1)
    with open(tmp_path / "scan.pkl", "rb") as f:
        state = pickle.load(f)
    assert len(state["voxel_map"]["keys"]) > 0
