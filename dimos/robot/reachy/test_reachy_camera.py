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

"""Tests for the Reachy replay camera, head-pose track, and camera module."""

from functools import partial
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pytest

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.reachy.camera import ReachyReplayCamera, load_reachy_camera_info
from dimos.robot.reachy.head_pose import OPT_TO_BODY, T_HEAD_CAM, HeadPoseTrack
from dimos.robot.reachy.module import ReachyCameraModule

W, H, N_FRAMES = 128, 96, 10
T0 = 1_700_000_000.0
FPS = 10.0


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


@pytest.fixture(scope="module", autouse=True)
def warm_shared_scheduler():  # noqa: ANN201
    """Spawn all shared-threadpool workers before the per-test thread snapshot.

    ``backpressure()`` schedules on the process-global ThreadPoolScheduler;
    its executor threads spawn lazily on first use and live for the process
    lifetime, so without pre-warming the first in-process test to use it gets
    flagged as leaking them.
    """
    import threading

    from dimos.utils.threadpool import get_max_workers, get_scheduler

    n = get_max_workers()
    barrier = threading.Barrier(n + 1)
    scheduler = get_scheduler()
    for _ in range(n):
        scheduler.schedule(lambda *_a, **_k: barrier.wait(timeout=10))
    barrier.wait(timeout=10)
    yield


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


def _yaw(theta: float) -> np.ndarray:
    m = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return m


def make_recording(root: Path) -> Path:
    rec = root / "rec"
    rec.mkdir()
    out = cv2.VideoWriter(str(rec / "camera.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), i * 20 % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    with open(rec / "camera_timestamps.jsonl", "w") as f:
        for i in range(N_FRAMES):
            f.write(json.dumps({"ts": T0 + i / FPS, "value": {"frame": i}}) + "\n")
    with open(rec / "head_pose.jsonl", "w") as f:
        for i in range(N_FRAMES):
            pose = _yaw(np.deg2rad(5.0 * i))  # head yawing 5 deg per frame
            f.write(json.dumps({"ts": T0 + i / FPS, "value": pose.tolist()}) + "\n")
    return rec


# ---------------------------------------------------------------------------
# In-process unit tests (no coordinator)
# ---------------------------------------------------------------------------


def test_head_pose_track_interpolation(tmp_path: Path) -> None:
    rec = make_recording(tmp_path)
    track = HeadPoseTrack.from_jsonl(rec / "head_pose.jsonl")
    assert len(track) == N_FRAMES

    # At a sample: pose = yaw @ T_HEAD_CAM exactly (optical rotation + lever arm)
    c2w = track.c2w_at(T0)
    assert np.allclose(c2w, _yaw(0.0) @ T_HEAD_CAM, atol=1e-9)
    c2w5 = track.c2w_at(T0 + 5 / FPS)
    assert np.allclose(c2w5, _yaw(np.deg2rad(25.0)) @ T_HEAD_CAM, atol=1e-9)

    # Midway between samples: rotation slerps to half the yaw step exactly.
    mid = track.c2w_at(T0 + 0.5 / FPS)
    assert np.allclose(mid[:3, :3], (_yaw(np.deg2rad(2.5)) @ T_HEAD_CAM)[:3, :3], atol=1e-6)
    # Position is LERP'd (not slerp'd), so the rotating lever arm's center is
    # the linear midpoint of the two sample centers — very close to, but not
    # exactly, the rotated lever arm at the mid angle (sub-0.1 mm at this step).
    p_lo = (_yaw(0.0) @ T_HEAD_CAM)[:3, 3]
    p_hi = (_yaw(np.deg2rad(5.0)) @ T_HEAD_CAM)[:3, 3]
    assert np.allclose(mid[:3, 3], 0.5 * (p_lo + p_hi), atol=1e-9)

    # Rotation is unchanged by the lever arm: camera forward (+Z optical) still
    # maps to body +X at yaw 0.
    fwd = (track.c2w_at(T0)[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    assert np.allclose(fwd, [1.0, 0.0, 0.0], atol=1e-9)

    # Lever arm present: camera origin sits forward+up of the head pivot.
    assert np.allclose(track.c2w_at(T0)[:3, 3], T_HEAD_CAM[:3, 3], atol=1e-9)

    # Out of range clamps
    assert np.allclose(track.c2w_at(T0 - 10), track.c2w_at(T0))
    assert np.allclose(track.c2w_at(T0 + 10), track.c2w_at(T0 + (N_FRAMES - 1) / FPS))


def test_replay_camera_emits_frames_and_poses(tmp_path: Path) -> None:
    rec = make_recording(tmp_path)
    cam = ReachyReplayCamera(recording_dir=str(rec), realtime=False, fps=0.0)
    assert cam.camera_info.width == W and cam.camera_info.height == H

    images: list = []
    poses: list = []
    sub_p = cam.pose_stream().subscribe(poses.append)
    sub_i = cam.image_stream().subscribe(images.append)
    deadline = time.time() + 10.0
    while time.time() < deadline and len(images) < 2:
        time.sleep(0.05)
    sub_i.dispose()
    sub_p.dispose()

    assert len(images) >= 2, "replay produced no frames"
    img = images[0]
    assert img.width == W and img.height == H
    assert img.frame_id == "camera_optical"
    assert abs(img.ts - T0) < 1.0  # recorded ts, not wall clock
    assert poses, "no head poses emitted"
    tf = poses[0]
    assert tf.frame_id == "reachy_base"
    assert tf.child_frame_id == "camera_optical"


def test_load_reachy_camera_info_json(tmp_path: Path) -> None:
    K = [[930.0, 0, 655.0], [0, 927.0, 352.0], [0, 0, 1.0]]
    cal = {
        "class_name": "PinholeCameraParameters",
        "intrinsic": {"width": 1280, "height": 720,
                      "intrinsic_matrix": np.asarray(K).flatten().tolist()},
        "distortion": {"model": "plumb_bob",
                       "coeffs": [0.03, -0.08, 0.0003, -0.0005, 0.02]},
    }
    p = tmp_path / "reachy_mini_intrinsics.json"
    p.write_text(json.dumps(cal))
    info = load_reachy_camera_info(p, 1280, 720)
    assert info.width == 1280 and info.height == 720
    assert abs(info.K[0] - 930.0) < 1e-9 and abs(info.K[4] - 927.0) < 1e-9
    assert info.distortion_model == "plumb_bob"
    assert abs(info.D[1] - (-0.08)) < 1e-12

    # HFOV fallback
    fallback = load_reachy_camera_info(None, 1280, 720, hfov_deg=70.0)
    assert abs(fallback.K[0] - (640.0 / np.tan(np.deg2rad(35.0)))) < 1e-6


# ---------------------------------------------------------------------------
# Module test (coordinator + worker process)
# ---------------------------------------------------------------------------


@pytest.mark.self_hosted
def test_reachy_camera_module_publishes(dimos_cluster: Any, transports: Any,
                                        tmp_path: Path) -> None:
    rec = make_recording(tmp_path)
    camera = dimos_cluster.deploy(
        ReachyCameraModule,
        hardware=partial(ReachyReplayCamera, recording_dir=str(rec),
                         realtime=False, fps=30.0, loop=True),
        camera_info_interval_s=0.2,
    )
    camera.color_image.transport = transports("/test_reachy/color", Image)
    camera.camera_info.transport = transports("/test_reachy/info", CameraInfo)

    images: list = []
    infos: list = []
    camera.color_image.subscribe(images.append)
    camera.camera_info.subscribe(infos.append)

    camera.start()
    deadline = time.time() + 15.0
    while time.time() < deadline and (not images or not infos):
        time.sleep(0.1)

    assert images, "module published no images"
    assert infos, "module published no camera_info"
    assert images[0].frame_id == "camera_optical"
    assert infos[0].width == W and infos[0].height == H
