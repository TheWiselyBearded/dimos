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

"""Convert between xr-nav map bundles and dimos ``.pc2.lcm`` prior maps.

The camera pipelines (Reachy/Viture/iPhone spatial scripts) persist maps as
xr-nav pickle bundles (``--save-map foo.pkl``: voxel state + object DB), while
the dimos ``RelocalizationModule`` loads LCM-serialized ``PointCloud2`` files
(``<name>.pc2.lcm``). This module bridges the two so a camera-built map can be
relocalized against by the production module and vice versa.

The bundle's voxel state is a plain numpy dict (see ``xr_nav.voxel_map
.VoxelMap.to_state`` / ``xr_nav.map_io``), so no xr-nav import is needed.

CLI (direction inferred from suffixes):

    python -m dimos.mapping.utils.xrnav map.pkl map.pc2.lcm
    python -m dimos.mapping.utils.xrnav map.pc2.lcm map.pkl --voxel-size 0.05
    python -m dimos.mapping.utils.xrnav map.ply map.pc2.lcm
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

BUNDLE_VERSION = 1  # xr_nav.map_io.MAP_BUNDLE_VERSION
DEFAULT_FRAME = "map"

_EMPTY_OBJECT_DB = {
    "pending": {},
    "permanent": {},
    "track_id_map": {},
    "confidence": {},
    "config": {},
}


def bundle_to_pointcloud2(bundle_path: Path | str, *, min_observations: int = 1,
                          frame_id: str = DEFAULT_FRAME) -> PointCloud2:
    """xr-nav ``.pkl`` bundle -> colored PointCloud2 of voxel centroids."""
    import open3d as o3d

    with open(bundle_path, "rb") as f:
        state = pickle.load(f)
    version = state.get("version")
    if version != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version {version} "
                         f"(expected {BUNDLE_VERSION})")
    vox = state["voxel_map"]
    counts = np.asarray(vox["count"])
    keep = counts >= min_observations
    pts = np.asarray(vox["centroids"], dtype=np.float64)[keep]
    cols = np.clip(np.asarray(vox["colors"], dtype=np.float64)[keep], 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    if len(pts):
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    logger.info(f"bundle {bundle_path}: {len(pts)}/{len(counts)} voxels "
                f"(min_observations={min_observations})")
    return PointCloud2(pointcloud=pcd, frame_id=frame_id, ts=time.time())


def pointcloud2_to_bundle_state(cloud: PointCloud2, *, voxel_size: float = 0.05,
                                max_range: float = 8.0) -> dict:
    """PointCloud2 -> xr-nav bundle dict (voxel state + empty object DB).

    Points are quantized to ``voxel_size``; each voxel's centroid/color is the
    mean of its points and ``count``/``confidence`` is the point count, so
    ``--load-map`` / ``ReferenceMap.from_bundle`` see a normal-looking map.
    """
    pts, cols = cloud.as_numpy()
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if cols is None or len(cols) != n:
        cols = np.full((n, 3), 0.6, dtype=np.float64)
    if n == 0:
        keys = np.zeros((0, 3), dtype=np.int64)
        centroids = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)
        counts = np.zeros((0,), dtype=np.int32)
    else:
        keys_int = np.floor(pts / voxel_size).astype(np.int64)
        # pack for a fast 1-D unique (21 bits/axis, offset keeps it positive)
        off = 1 << 20
        mask = (1 << 21) - 1
        packed = (((keys_int[:, 0] + off) << 42)
                  | ((keys_int[:, 1] + off) << 21)
                  | (keys_int[:, 2] + off))
        uniq, inverse = np.unique(packed, return_inverse=True)
        m = len(uniq)
        keys = np.empty((m, 3), dtype=np.int64)
        keys[:, 0] = (uniq >> 42) - off
        keys[:, 1] = ((uniq >> 21) & mask) - off
        keys[:, 2] = (uniq & mask) - off
        counts = np.bincount(inverse, minlength=m).astype(np.int32)
        centroids = np.empty((m, 3), dtype=np.float32)
        colors = np.empty((m, 3), dtype=np.float32)
        for ax in range(3):
            centroids[:, ax] = (np.bincount(inverse, weights=pts[:, ax],
                                            minlength=m) / counts)
            colors[:, ax] = (np.bincount(inverse, weights=cols[:, ax],
                                         minlength=m) / counts)
    return {
        "version": BUNDLE_VERSION,
        "saved_at": time.time(),
        "voxel_map": {
            "voxel_size": float(voxel_size),
            "max_range": float(max_range),
            "keys": keys,
            "centroids": centroids,
            "colors": colors,
            "confidence": counts.astype(np.float32),
            "count": counts,
            "miss_count": np.zeros(len(counts), dtype=np.int32),
        },
        "object_db": dict(_EMPTY_OBJECT_DB),
        "extra": {"converted_from": "pointcloud2", "n_points": int(n)},
    }


def _read_cloud(path: Path, frame_id: str) -> PointCloud2:
    if path.name.endswith(".pc2.lcm"):
        return PointCloud2.lcm_decode(path.read_bytes())
    if path.suffix.lower() in (".ply", ".pcd"):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        return PointCloud2(pointcloud=pcd, frame_id=frame_id, ts=time.time())
    raise ValueError(f"unsupported cloud input: {path}")


def convert(src: Path | str, dst: Path | str, *, voxel_size: float = 0.05,
            min_observations: int = 1, frame_id: str = DEFAULT_FRAME) -> None:
    """Convert between map formats, direction inferred from suffixes."""
    src, dst = Path(src), Path(dst)
    src_is_bundle = src.suffix.lower() == ".pkl"
    dst_is_bundle = dst.suffix.lower() == ".pkl"

    if src_is_bundle and not dst_is_bundle:
        cloud = bundle_to_pointcloud2(src, min_observations=min_observations,
                                      frame_id=frame_id)
        if dst.name.endswith(".pc2.lcm"):
            dst.write_bytes(cloud.lcm_encode())
        elif dst.suffix.lower() in (".ply", ".pcd"):
            import open3d as o3d

            o3d.io.write_point_cloud(str(dst), cloud.pointcloud,
                                     write_ascii=False, compressed=True)
        else:
            raise ValueError(f"unsupported cloud output: {dst}")
        logger.info(f"wrote {dst} ({len(cloud)} points)")
    elif dst_is_bundle and not src_is_bundle:
        cloud = _read_cloud(src, frame_id)
        state = pointcloud2_to_bundle_state(cloud, voxel_size=voxel_size)
        with open(dst, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"wrote {dst} ({len(state['voxel_map']['keys'])} voxels "
                    f"@ {voxel_size} m from {len(cloud)} points)")
    else:
        raise ValueError(
            "exactly one side must be a .pkl bundle "
            f"(got {src.name} -> {dst.name})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path,
                        help=".pkl bundle, .pc2.lcm, .ply or .pcd")
    parser.add_argument("dst", type=Path,
                        help="output path; direction inferred from suffixes")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="voxel size (m) when building a bundle from a cloud")
    parser.add_argument("--min-observations", type=int, default=1,
                        help="bundle->cloud: only export voxels seen this often")
    parser.add_argument("--frame-id", default=DEFAULT_FRAME)
    args = parser.parse_args()
    convert(args.src, args.dst, voxel_size=args.voxel_size,
            min_observations=args.min_observations, frame_id=args.frame_id)


if __name__ == "__main__":
    main()
