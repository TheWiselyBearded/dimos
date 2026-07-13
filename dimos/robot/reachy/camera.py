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

"""Reachy Mini camera hardware: live SDK stream and recording replay.

Both classes implement the ``CameraHardware`` spec (``image_stream()`` +
``camera_info``) and additionally expose ``pose_stream()``: per-frame
``reachy_base -> camera_optical`` transforms (recorded kinematic head pose for
replay; polled SDK head pose for live). ``ReachyCameraModule`` publishes those
as TF so downstream consumers can fuse camera-frame clouds into the Z-up
``reachy_base`` world.
"""

from __future__ import annotations

from functools import cache
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from reactivex import create
from reactivex.observable import Observable

from dimos.hardware.sensors.camera.spec import CameraConfig, CameraHardware
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.reachy.head_pose import HeadPoseTrack, head_pose_to_camera_pose
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()

BASE_FRAME = "reachy_base"
OPTICAL_FRAME = "camera_optical"
DEFAULT_HFOV_DEG = 70.0  # Reachy Mini wireless, used only without calibration


def load_reachy_camera_info(path: Path | str | None, width: int, height: int,
                            hfov_deg: float = DEFAULT_HFOV_DEG,
                            frame_id: str = OPTICAL_FRAME) -> CameraInfo:
    """CameraInfo from a calibration file, or an HFOV pinhole fallback.

    Accepts the ``PinholeCameraParameters`` JSON written by
    ``xr-nav/scripts/calibrate_reachy_solve.py`` (``reachy_mini_intrinsics.json``)
    or a ROS CameraInfo YAML. Without a file, synthesizes a distortion-free
    pinhole from ``hfov_deg`` — fine for visualization, not for metric mapping.
    """
    if path is None:
        info = CameraInfo.from_fov(hfov_deg, width, height, axis="horizontal",
                                   frame_id=frame_id)
        return info
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        info = CameraInfo.from_yaml(str(path))
        info.frame_id = frame_id
        return info
    with open(path) as f:
        data = json.load(f)
    intr = data["intrinsic"]
    K = np.asarray(intr["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
    # calibrate_reachy_solve.py flattens row-major; genuine Open3D
    # PinholeCameraParameters JSONs are column-major.
    if not np.allclose(K[2], [0, 0, 1]) and np.allclose(K[:2, 2], 0):
        K = K.T
    D = list(data.get("distortion", {}).get("coeffs", [0.0] * 5))
    model = data.get("distortion", {}).get("model", "plumb_bob")
    P = np.zeros((3, 4), dtype=np.float64)
    P[:3, :3] = K
    return CameraInfo(
        height=int(intr["height"]), width=int(intr["width"]),
        distortion_model=model, D=D,
        K=K.flatten().tolist(),
        R=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
        P=P.flatten().tolist(),
        frame_id=frame_id,
    )


def camera_pose_to_transform(c2w_4x4: np.ndarray, ts: float) -> Transform:
    """4x4 ``reachy_base -> camera_optical`` to a dimos TF Transform."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(c2w_4x4[:3, :3]).as_quat()
    t = c2w_4x4[:3, 3]
    return Transform(
        translation=Vector3(float(t[0]), float(t[1]), float(t[2])),
        rotation=Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        frame_id=BASE_FRAME,
        child_frame_id=OPTICAL_FRAME,
        ts=ts,
    )


class _PoseStreamMixin:
    """Shared per-frame pose observable fed by the capture loop."""

    def __init__(self) -> None:
        self._pose_observer = None

    @cache
    def pose_stream(self) -> Observable[Transform]:
        """``reachy_base -> camera_optical`` transforms, one per frame."""

        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._pose_observer = observer

            def dispose() -> None:
                self._pose_observer = None

            return dispose

        return backpressure(create(subscribe))

    def _emit_pose(self, c2w_4x4: np.ndarray | None, ts: float) -> None:
        if c2w_4x4 is None or self._pose_observer is None:
            return
        try:
            self._pose_observer.on_next(camera_pose_to_transform(c2w_4x4, ts))
        except Exception:
            logger.exception("pose emit failed")


class ReachyReplayCameraConfig(CameraConfig):
    recording_dir: str = ""
    loop: bool = False
    realtime: bool = True  # pace frames to the recorded timestamps
    fps: float = 15.0  # only used as a cap when realtime is False
    width: int = 0  # discovered from the recording
    height: int = 0
    frame_id_prefix: str | None = None
    camera_info_path: str | None = None
    hfov_deg: float = DEFAULT_HFOV_DEG


class ReachyReplayCamera(CameraHardware, _PoseStreamMixin):
    """Plays back an on-robot recording (camera.mp4 + camera_timestamps.jsonl
    + head_pose.jsonl) as a CameraHardware with a pose stream."""

    config: ReachyReplayCameraConfig

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        _PoseStreamMixin.__init__(self)
        rec = Path(self.config.recording_dir).expanduser()
        if not (rec / "camera.mp4").exists():
            raise FileNotFoundError(f"missing camera.mp4 in {rec}")
        self._rec = rec
        self._timestamps = self._load_timestamps(rec / "camera_timestamps.jsonl")
        pose_path = rec / "head_pose.jsonl"
        self._poses = (HeadPoseTrack.from_jsonl(pose_path)
                       if pose_path.exists() else HeadPoseTrack([], []))
        cap = cv2.VideoCapture(str(rec / "camera.mp4"))
        self._W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self._camera_info = load_reachy_camera_info(
            self.config.camera_info_path, self._W, self._H, self.config.hfov_deg)
        self._observer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        logger.info(f"reachy replay: {rec.name} {self._W}x{self._H} "
                    f"{len(self._timestamps)} frames, {len(self._poses)} head poses")

    @staticmethod
    def _load_timestamps(path: Path) -> list[float]:
        if not path.exists():
            return []
        out: list[float] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(float(json.loads(line)["ts"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        return out

    @property
    def camera_info(self) -> CameraInfo:
        return self._camera_info

    @cache
    def image_stream(self) -> Observable[Image]:
        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._observer = observer
            try:
                self.start()
            except Exception as e:  # noqa: BLE001
                observer.on_error(e)
                return lambda: None

            def dispose() -> None:
                self._observer = None
                self.stop()

            return dispose

        return backpressure(create(subscribe))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True,
                                        name="reachy-replay-camera")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _replay_loop(self) -> None:
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(str(self._rec / "camera.mp4"))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n_pair = (min(n_frames, len(self._timestamps))
                      if self._timestamps else n_frames)
            wall_start = time.time()
            ts0 = self._timestamps[0] if self._timestamps else wall_start
            min_period = 0.0 if self.config.fps <= 0 else 1.0 / self.config.fps
            last_emit = 0.0
            for idx in range(n_pair):
                if self._stop_event.is_set():
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                ts = self._timestamps[idx] if self._timestamps else time.time()
                if self.config.realtime and self._timestamps:
                    lag = (wall_start + (ts - ts0)) - time.time()
                    if lag > 0 and self._stop_event.wait(min(lag, 1.0)):
                        break
                elif min_period > 0 and last_emit:
                    dt = time.time() - last_emit
                    if dt < min_period and self._stop_event.wait(min_period - dt):
                        break
                last_emit = time.time()
                if bgr.ndim == 2:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.from_numpy(rgb, format=ImageFormat.RGB,
                                         frame_id=OPTICAL_FRAME, ts=ts)
                self._emit_pose(self._poses.c2w_at(ts), ts)
                if self._observer is not None and not self._stop_event.is_set():
                    self._observer.on_next(image)
            cap.release()
            if not self.config.loop:
                break
        if self._observer is not None:
            try:
                self._observer.on_completed()
            except Exception:  # noqa: BLE001
                pass


class ReachyCameraConfig(CameraConfig):
    width: int = 1280
    height: int = 720
    fps: float = 15.0
    frame_id_prefix: str | None = None
    camera_info_path: str | None = None
    hfov_deg: float = DEFAULT_HFOV_DEG
    media_backend: str = "default"
    connection_mode: str | None = None  # None = SDK auto-detect (USB/localhost/network)


class ReachyCamera(CameraHardware, _PoseStreamMixin):
    """Live Reachy Mini camera via the reachy_mini SDK, with polled head pose."""

    config: ReachyCameraConfig

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        _PoseStreamMixin.__init__(self)
        self._mini = None
        self._observer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera_info: CameraInfo | None = None

    @property
    def camera_info(self) -> CameraInfo:
        if self._camera_info is None:
            self._camera_info = load_reachy_camera_info(
                self.config.camera_info_path, self.config.width,
                self.config.height, self.config.hfov_deg)
        return self._camera_info

    @cache
    def image_stream(self) -> Observable[Image]:
        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._observer = observer
            try:
                self.start()
            except Exception as e:  # noqa: BLE001
                observer.on_error(e)
                return lambda: None

            def dispose() -> None:
                self._observer = None
                self.stop()

            return dispose

        return backpressure(create(subscribe))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Imported lazily so the module imports without the SDK installed.
        from reachy_mini import ReachyMini  # type: ignore[import-not-found]

        kwargs = {"media_backend": self.config.media_backend}
        if self.config.connection_mode is not None:
            kwargs["connection_mode"] = self.config.connection_mode
        self._mini = ReachyMini(**kwargs)
        self._mini.__enter__()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name="reachy-camera")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._mini = None

    def _head_pose(self) -> np.ndarray | None:
        """Current ``reachy_base -> camera_optical`` from the SDK kinematics."""
        try:
            pose = self._mini.get_current_head_pose()
        except Exception:  # noqa: BLE001
            return None
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            return None
        return head_pose_to_camera_pose(pose)

    def _capture_loop(self) -> None:
        period = 0.0 if self.config.fps <= 0 else 1.0 / self.config.fps
        next_t = time.time()
        while not self._stop_event.is_set():
            ts = time.time()
            try:
                bgr = self._mini.media.get_frame()
            except Exception:  # noqa: BLE001
                bgr = None
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.from_numpy(rgb, format=ImageFormat.RGB,
                                         frame_id=OPTICAL_FRAME, ts=ts)
                self._emit_pose(self._head_pose(), ts)
                if self._observer is not None and not self._stop_event.is_set():
                    self._observer.on_next(image)
            if period <= 0:
                continue
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                if self._stop_event.wait(timeout=sleep):
                    break
            else:
                next_t = time.time()
