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

"""Reachy Mini head-pose helpers.

The Reachy SDK reports the head pose as a 4x4 SE3 ``reachy_base -> head`` in
the robot body convention (FLU: X-fwd, Y-left, Z-up; yaw about Z). Camera
pipelines back-project depth in the OpenCV optical convention (RDF: X-right,
Y-down, Z-fwd), so the body pose must be right-multiplied by ``OPT_TO_BODY``
before it is used as a camera-to-world transform — otherwise a head yaw is
applied as a roll about the optical view axis and the map fans out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

# Optical (RDF) -> Reachy head/body (FLU) axis permutation (rotation only).
OPT_TO_BODY = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# Full head -> camera extrinsic, mirroring the SDK's ReachyMini.T_head_cam:
# OPT_TO_BODY rotation plus the camera's lever arm — the optical center sits
# 43.7 mm forward / 51.2 mm above the head frame origin. Dropping the
# translation puts the modeled camera at the head pivot, so head/body rotation
# swings the real camera on an arc the pose ignores and nearby geometry smears
# by centimeters across viewpoints.
T_HEAD_CAM = OPT_TO_BODY.copy()
T_HEAD_CAM[:3, 3] = [0.0437, 0.0, 0.0512]


def head_pose_to_camera_pose(head_pose_4x4: NDArray[np.float64]) -> NDArray[np.float64]:
    """``reachy_base -> head`` (FLU body) to ``reachy_base -> camera_optical``."""
    return np.asarray(head_pose_4x4, dtype=np.float64) @ T_HEAD_CAM


class HeadPoseTrack:
    """Time-indexed head poses with slerp/lerp interpolation.

    Poses are stored as ``reachy_base -> camera_optical`` (already converted
    with :data:`OPT_TO_BODY`). ``anchor_first=True`` re-expresses every pose
    relative to the first sample (world = initial camera pose) — the behaviour
    the standalone replay script uses; the dimos module keeps the raw base
    frame so the world stays Z-up and gravity-consistent.
    """

    def __init__(self, ts: list[float], head_poses_4x4: list[NDArray[np.float64]],
                 anchor_first: bool = False):
        mats: list[NDArray[np.float64]] = []
        keep_ts: list[float] = []
        for t, m in zip(ts, head_poses_4x4):
            m = np.asarray(m, dtype=np.float64)
            if m.shape != (4, 4) or not np.all(np.isfinite(m)):
                continue
            keep_ts.append(float(t))
            mats.append(head_pose_to_camera_pose(m))
        if anchor_first and mats:
            inv0 = np.linalg.inv(mats[0])
            mats = [inv0 @ m for m in mats]
        self._ts = np.asarray(keep_ts, dtype=np.float64)
        self._mats = mats
        self._slerp: Slerp | None = None
        if len(mats) >= 2:
            self._slerp = Slerp(self._ts,
                                Rotation.from_matrix(np.stack([m[:3, :3] for m in mats])))
            self._pos = np.stack([m[:3, 3] for m in mats])

    @classmethod
    def from_jsonl(cls, path: Path | str, anchor_first: bool = False) -> HeadPoseTrack:
        """Load from the on-robot recorder's ``head_pose.jsonl``
        (lines of ``{"ts": unix_seconds, "value": [[4x4]]}``)."""
        ts: list[float] = []
        poses: list[NDArray[np.float64]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts.append(float(ev["ts"]))
                    poses.append(np.asarray(ev["value"], dtype=np.float64))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return cls(ts, poses, anchor_first=anchor_first)

    def __len__(self) -> int:
        return len(self._mats)

    def c2w_at(self, query_ts: float) -> NDArray[np.float64] | None:
        """Interpolated ``reachy_base -> camera_optical`` at ``query_ts``."""
        n = len(self._mats)
        if n == 0:
            return None
        if n == 1 or query_ts <= self._ts[0]:
            return self._mats[0].copy()
        if query_ts >= self._ts[-1]:
            return self._mats[-1].copy()
        assert self._slerp is not None
        hi = int(np.searchsorted(self._ts, query_ts, side="right"))
        lo = hi - 1
        t0, t1 = self._ts[lo], self._ts[hi]
        u = 0.0 if t1 - t0 < 1e-9 else (query_ts - t0) / (t1 - t0)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self._slerp([query_ts]).as_matrix()[0]
        out[:3, 3] = (1.0 - u) * self._pos[lo] + u * self._pos[hi]
        return out
