"""Replay a recorded Reachy Mini session through the dimos spatial-memory pipeline.

Mirrors ``integrations/dimos/server.py`` in ReachyBrain, but instead of pulling
frames from a live robot over WebSocket, this script plays back a directory of
files produced by the on-robot recorder:

    reachy_trial4/
      camera.mp4                 monocular RGB at ~15 fps
      camera_timestamps.jsonl    {"ts": unix_seconds, "value": {"frame": N}}
      imu.jsonl                  accel + gyro + quaternion + temp
      joints.jsonl               7 head joints + 2 antennas
      head_pose.jsonl            4x4 SE3 body -> head_optical
      doa.jsonl                  direction-of-arrival + speech bool
      audio.wav                  16 kHz stereo PCM (RMS levels are published)
      metadata.json              counts + start/end timestamps

How:

  1. Load ``mac_iphone_spatial_foxglove.py`` as an importable module
     (``dimos_iphone``), monkey-patch its ``VideoSource`` with a class that reads
     frames out of ``camera.mp4`` and stamps each one with the matching ts from
     ``camera_timestamps.jsonl``. The rest of the dimos pipeline graph (DepthPro
     / DA3 -> VO -> ObjectDB -> SpatialMemory -> Foxglove via LCM) is unchanged.

  2. Start a sidecar Foxglove server (port 8766 by default — same one used by
     the live ``ImuFoxgloveServer``) that publishes the recorder's other streams
     paced against the camera playback clock so every signal stays time-aligned
     with the depth + VO output. Topics published:

         /reachy/imu              foxglove.Imu
         /reachy/imu/temperature  scalar
         /reachy/head_pose        foxglove.FrameTransform (reachy_base -> reachy_head)
         /reachy/joints           json {head: [7], antennas: [2]}
         /reachy/doa              json {angle_rad, angle_deg, speech}
         /reachy/audio/level      scalar RMS amplitude (computed offline)
         /reachy/playback         scalar playback ts (sanity track)

  3. Spawn ``dimos.main()``. The main loop iterates ``ReachyReplayVideoSource``,
     which sets a shared playback clock as it yields each frame; the sidecar
     thread drains its JSONL streams up to that clock, publishing in order.

Pair with the dimos foxglove bridge (LCM -> ws://localhost:8765) in another
terminal — that one carries /color_image, /depth, /map, /points_frame, /tf,
etc., exactly as ``run_camera_pipeline.py`` does it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger("reachy_replay")

DIMOS_DIR = Path(__file__).resolve().parent
DEFAULT_DIMOS_SCRIPT = "mac_iphone_spatial_foxglove.py"

# Optical(RDF: X-right, Y-down, Z-fwd) -> Reachy head/body (FLU: X-fwd, Y-left,
# Z-up). head_pose.jsonl is recorded straight from get_current_head_pose(),
# which is in the body convention (create_head_pose builds rotation as euler
# xyz [roll,pitch,yaw] => yaw about Z). The dimos pipeline back-projects depth
# in the OpenCV optical convention, so the head pose must be right-multiplied
# by this constant before being used as c2w — otherwise a head yaw is applied
# as a roll about the optical view axis and the map fans out.
_OPT_TO_BODY = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


# ----------------------------------------------------------------------------
# Playback clock — shared between the VideoSource and the sidecar publisher.
# ----------------------------------------------------------------------------


class PlaybackClock:
    """Monotonically advanced by the VideoSource as it yields frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._t: float = 0.0
        self._eof = False

    def set(self, t: float) -> None:
        with self._cond:
            if t > self._t:
                self._t = t
                self._cond.notify_all()

    def mark_eof(self) -> None:
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    @property
    def value(self) -> float:
        with self._lock:
            return self._t

    @property
    def eof(self) -> bool:
        with self._lock:
            return self._eof

    def wait_until(self, t: float, stop_event: threading.Event, poll: float = 0.05) -> bool:
        """Block until playback time reaches ``t``. Returns False if we hit EOF
        first or the caller's stop_event fires."""
        with self._cond:
            while not self._eof and self._t < t and not stop_event.is_set():
                self._cond.wait(timeout=poll)
            return self._t >= t


# ----------------------------------------------------------------------------
# Replay VideoSource — drop-in for dimos's VideoSource.
# ----------------------------------------------------------------------------


def _load_camera_timestamps(path: Path) -> list[float]:
    out: list[float] = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            out.append(float(rec["ts"]))
    return out


def make_replay_video_source(
    recording_dir: Path,
    clock: PlaybackClock,
    source_frame_cls,  # dimos_iphone.SourceFrame
    pose_interp: Optional["HeadPoseInterpolator"] = None,
):
    """Return a class that exposes the same interface as
    ``mac_iphone_spatial_foxglove.VideoSource`` but plays back a recording."""

    video_path = recording_dir / "camera.mp4"
    ts_path = recording_dir / "camera_timestamps.jsonl"
    if not video_path.exists():
        sys.exit(f"missing camera.mp4: {video_path}")
    if not ts_path.exists():
        sys.exit(f"missing camera_timestamps.jsonl: {ts_path}")

    timestamps = _load_camera_timestamps(ts_path)

    class ReachyReplayVideoSource:
        def __init__(
            self,
            video_path=None,  # ignored — dimos passes --video but we use recording_dir
            fps_cap: float = 10.0,
            loop: bool = True,
        ):
            mp4 = recording_dir / "camera.mp4"
            self._cap = cv2.VideoCapture(str(mp4))
            if not self._cap.isOpened():
                sys.exit(f"cv2 could not open {mp4}")
            self._W = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._H = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._loop = loop
            self._fps_cap = max(fps_cap, 0.1)
            self._timestamps = timestamps
            n_ts = len(timestamps)
            if n_ts != self._total:
                print(
                    f"[replay:source] WARN: {n_ts} timestamps vs {self._total} mp4 frames "
                    "— pairing by index, extras dropped"
                )
            print(
                f"[replay:source] {video_path.name} {self._W}x{self._H} "
                f"frames={self._total} ts_count={n_ts} loop={loop}"
            )

        @property
        def frame_size(self) -> tuple[int, int]:
            return self._W, self._H

        def __iter__(self) -> Iterator:
            n_ts = len(self._timestamps)
            n_total = self._total
            n_pair = min(n_ts, n_total)
            loops = 0
            while True:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                wall_start = time.time()
                ts_first = self._timestamps[0] if self._timestamps else 0.0
                last_yield_wall = 0.0
                min_period = 1.0 / self._fps_cap
                for idx in range(n_pair):
                    ok, frame_bgr = self._cap.read()
                    if not ok:
                        break
                    if frame_bgr.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
                    ts = self._timestamps[idx]
                    # Wall-clock pace: hold the original timing.
                    wall_target = wall_start + (ts - ts_first)
                    now = time.time()
                    sleep_for = wall_target - now
                    if sleep_for > 0:
                        time.sleep(min(sleep_for, 1.0))
                    # Also rate-limit by --max-fps so dimos doesn't get overrun.
                    if last_yield_wall:
                        dt = time.time() - last_yield_wall
                        if dt < min_period:
                            time.sleep(min_period - dt)
                    last_yield_wall = time.time()
                    clock.set(ts)
                    c2w = pose_interp.c2w_at(ts) if pose_interp is not None else None
                    yield source_frame_cls(color_bgr=frame_bgr, ts=ts, frame_idx=idx, c2w=c2w)
                if not self._loop:
                    clock.mark_eof()
                    return
                loops += 1
                print(f"[replay:source] --- loop {loops} ---")

        def close(self) -> None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass

    return ReachyReplayVideoSource


# ----------------------------------------------------------------------------
# Sidecar Foxglove server: IMU, joints, head_pose, DOA, audio RMS, playback ts.
# ----------------------------------------------------------------------------


_IMU_SCHEMA = {
    "title": "foxglove.Imu",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "orientation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}, "w": {"type": "number"}}},
        "angular_velocity": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
        "linear_acceleration": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
    },
}

_SCALAR_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "value": {"type": "number"},
    },
}

_FRAME_TRANSFORM_SCHEMA = {
    "title": "foxglove.FrameTransform",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
        "rotation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "z": {"type": "number"}, "w": {"type": "number"}}},
    },
}

_JOINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "head": {"type": "array", "items": {"type": "number"}},
        "antennas": {"type": "array", "items": {"type": "number"}},
    },
}

_DOA_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {
            "sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "angle_rad": {"type": "number"},
        "angle_deg": {"type": "number"},
        "speech": {"type": "boolean"},
    },
}


def _ts_to_sec_nsec(ts: float) -> tuple[int, int]:
    sec = int(ts)
    nsec = int(round((ts - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return sec, nsec


def _ts_ns(ts: float) -> int:
    return int(ts * 1e9)


def _matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> (x, y, z, w). Avoids scipy import on this hot path."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


@dataclass
class JsonlEvent:
    ts: float
    value: object


def _read_jsonl(path: Path) -> list[JsonlEvent]:
    if not path.exists():
        return []
    out: list[JsonlEvent] = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            out.append(JsonlEvent(ts=float(rec["ts"]), value=rec["value"]))
    return out


class HeadPoseInterpolator:
    """SLERP+lerp camera-to-world from recorded reachy_base->head_optical matrices.

    Each c2w is anchored to the first usable head_pose so the dimos world frame
    starts at identity (matching the VO convention). Query times outside the
    recording's pose range are clamped to the nearest end.
    """

    def __init__(self, events: list[JsonlEvent]):
        ts: list[float] = []
        mats: list[np.ndarray] = []
        for ev in events:
            try:
                m = np.asarray(ev.value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if m.shape != (4, 4) or not np.all(np.isfinite(m)):
                continue
            ts.append(float(ev.ts))
            mats.append(m)
        self._ts = np.asarray(ts, dtype=np.float64)
        if mats:
            # Convert each body-frame head pose into the optical convention the
            # pipeline expects, then anchor to the first sample (world=identity).
            mats = [m @ _OPT_TO_BODY for m in mats]
            inv0 = np.linalg.inv(mats[0])
            self._c2w = [inv0 @ m for m in mats]
        else:
            self._c2w = []

    def __len__(self) -> int:
        return len(self._c2w)

    def c2w_at(self, query_ts: float) -> Optional[np.ndarray]:
        n = len(self._c2w)
        if n == 0:
            return None
        if n == 1 or query_ts <= self._ts[0]:
            return self._c2w[0].copy()
        if query_ts >= self._ts[-1]:
            return self._c2w[-1].copy()
        hi = int(np.searchsorted(self._ts, query_ts, side="right"))
        lo = hi - 1
        t0 = self._ts[lo]
        t1 = self._ts[hi]
        dt = t1 - t0
        if dt < 1e-9:
            return self._c2w[lo].copy()
        u = (query_ts - t0) / dt
        c0 = self._c2w[lo]
        c1 = self._c2w[hi]
        pos = (1.0 - u) * c0[:3, 3] + u * c1[:3, 3]
        rot = _slerp_3x3(c0[:3, :3], c1[:3, :3], u)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rot
        out[:3, 3] = pos
        return out


def _slerp_3x3(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two 3x3 rotation matrices. Avoids a scipy dep."""
    q0 = _rot_to_quat(R0)
    q1 = _rot_to_quat(R1)
    # Take shortest path
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q = q / np.linalg.norm(q)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
        s1 = np.sin(t * theta_0) / sin_theta_0
        q = s0 * q0 + s1 * q1
    return _quat_to_rot(q)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z) unit quaternion."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _audio_rms_events(wav_path: Path, t0: float, chunk_ms: int = 100) -> list[JsonlEvent]:
    """Read a WAV upfront and emit one RMS sample per ``chunk_ms`` window.

    ``t0`` is the recording's start ts; the event timestamps are absolute (so
    the sidecar can compare against the camera clock the same way it does for
    the other streams)."""
    if not wav_path.exists():
        return []
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        logger.warning("audio sampwidth=%d (expected 2) — skipping audio RMS", sampwidth)
        return []
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0
    chunk = max(1, int(framerate * chunk_ms / 1000))
    out: list[JsonlEvent] = []
    for i in range(0, len(samples), chunk):
        seg = samples[i: i + chunk]
        if len(seg) == 0:
            break
        rms = float(np.sqrt(np.mean(seg * seg)))
        ts = t0 + (i / framerate)
        out.append(JsonlEvent(ts=ts, value={"rms": rms}))
    return out


class SidecarFoxglove:
    """JSON Foxglove server for the reachy-specific streams."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8766,
        imu_frame: str = "reachy_imu",
    ) -> None:
        self.host = host
        self.port = port
        self.imu_frame = imu_frame
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._channels: dict[str, int] = {}
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._available = True

    def start(self) -> bool:
        try:
            import foxglove_websocket  # noqa: F401
        except ImportError:
            logger.warning("foxglove-websocket not installed — sidecar disabled")
            self._available = False
            return False
        self._thread = threading.Thread(target=self._run, daemon=True, name="reachy-sidecar-fox")
        self._thread.start()
        self._ready.wait(timeout=3.0)
        return self._available

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError as e:
            if "Event loop stopped" not in str(e):
                logger.exception("sidecar foxglove crashed: %s", e)
        except Exception:  # noqa: BLE001
            logger.exception("sidecar foxglove crashed")
        finally:
            self._ready.set()

    async def _serve(self) -> None:
        from foxglove_websocket.server import FoxgloveServer

        channels = [
            ("/reachy/imu", "foxglove.Imu", _IMU_SCHEMA),
            ("/reachy/imu/temperature", "ImuTemperature", _SCALAR_SCHEMA),
            ("/reachy/head_pose", "foxglove.FrameTransform", _FRAME_TRANSFORM_SCHEMA),
            ("/reachy/joints", "ReachyJoints", _JOINTS_SCHEMA),
            ("/reachy/doa", "ReachyDoa", _DOA_SCHEMA),
            ("/reachy/audio/level", "AudioRms", _SCALAR_SCHEMA),
            ("/reachy/playback", "PlaybackTs", _SCALAR_SCHEMA),
        ]
        async with FoxgloveServer(
            self.host, self.port, "reachy-replay-sidecar",
            capabilities=[], supported_encodings=["json"],
        ) as server:
            self._server = server
            for topic, schema_name, schema in channels:
                cid = await server.add_channel({
                    "topic": topic, "encoding": "json",
                    "schemaName": schema_name, "schema": json.dumps(schema),
                })
                self._channels[topic] = cid
            logger.info(
                "sidecar foxglove listening on ws://%s:%d (%d topics)",
                self.host, self.port, len(channels),
            )
            self._ready.set()
            await asyncio.Event().wait()

    def _send(self, topic: str, ts: float, payload: dict) -> None:
        if not self._available or self._server is None or self._loop is None or self._loop.is_closed():
            return
        cid = self._channels.get(topic)
        if cid is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._server.send_message(cid, _ts_ns(ts), json.dumps(payload).encode("utf-8")),
            self._loop,
        )

    # -- per-stream publishers ------------------------------------------------

    def publish_imu(self, ts: float, value: dict) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        ax, ay, az = value.get("accelerometer") or [0.0, 0.0, 0.0]
        gx, gy, gz = value.get("gyroscope") or [0.0, 0.0, 0.0]
        q = value.get("quaternion") or [1.0, 0.0, 0.0, 0.0]
        # Recorder stores quaternion as wxyz; Foxglove wants xyzw fields.
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        self._send("/reachy/imu", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "frame_id": self.imu_frame,
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "angular_velocity": {"x": gx, "y": gy, "z": gz},
            "linear_acceleration": {"x": ax, "y": ay, "z": az},
        })
        if "temperature" in value:
            self._send("/reachy/imu/temperature", ts, {
                "timestamp": {"sec": sec, "nsec": nsec},
                "value": float(value["temperature"]),
            })

    def publish_head_pose(self, ts: float, matrix: list) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        m = np.array(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            return
        tx, ty, tz = float(m[0, 3]), float(m[1, 3]), float(m[2, 3])
        qx, qy, qz, qw = _matrix_to_quat(m[:3, :3])
        self._send("/reachy/head_pose", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "parent_frame_id": "reachy_base",
            "child_frame_id": "reachy_head",
            "translation": {"x": tx, "y": ty, "z": tz},
            "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
        })

    def publish_joints(self, ts: float, value: dict) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        self._send("/reachy/joints", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "head": list(value.get("head") or []),
            "antennas": list(value.get("antennas") or []),
        })

    def publish_doa(self, ts: float, value: dict) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        self._send("/reachy/doa", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "angle_rad": float(value.get("angle_rad", 0.0)),
            "angle_deg": float(value.get("angle_deg", 0.0)),
            "speech": bool(value.get("speech", False)),
        })

    def publish_audio_rms(self, ts: float, value: dict) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        self._send("/reachy/audio/level", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "value": float(value.get("rms", 0.0)),
        })

    def publish_playback_ts(self, ts: float) -> None:
        sec, nsec = _ts_to_sec_nsec(ts)
        self._send("/reachy/playback", ts, {
            "timestamp": {"sec": sec, "nsec": nsec},
            "value": float(ts),
        })


# ----------------------------------------------------------------------------
# Replay driver thread — drains JSONL streams against the playback clock.
# ----------------------------------------------------------------------------


@dataclass
class StreamSpec:
    name: str
    events: list[JsonlEvent]
    publish: Callable[[float, object], None]


def _drain_loop(
    streams: list[StreamSpec],
    clock: PlaybackClock,
    stop_event: threading.Event,
    sidecar: SidecarFoxglove,
) -> None:
    """Publish each stream's events in order, gated on the playback clock."""
    indices = {s.name: 0 for s in streams}
    last_playback_pub = 0.0
    while not stop_event.is_set():
        t = clock.value
        any_progress = False
        for s in streams:
            i = indices[s.name]
            while i < len(s.events) and s.events[i].ts <= t:
                ev = s.events[i]
                try:
                    s.publish(ev.ts, ev.value)  # type: ignore[arg-type]
                except Exception as e:  # noqa: BLE001
                    logger.warning("publish %s ts=%.3f failed: %s", s.name, ev.ts, e)
                i += 1
                any_progress = True
            indices[s.name] = i
        # Heartbeat scalar so Foxglove always shows a moving cursor.
        if t > 0 and t - last_playback_pub > 0.25:
            sidecar.publish_playback_ts(t)
            last_playback_pub = t
        all_done = all(indices[s.name] >= len(s.events) for s in streams)
        if all_done and clock.eof:
            return
        if not any_progress:
            time.sleep(0.02)


# ----------------------------------------------------------------------------
# Wiring: load dimos module, swap VideoSource, build sidecar, run main().
# ----------------------------------------------------------------------------


def _load_dimos_module(dimos_dir: Path, script_name: str):
    script = dimos_dir / script_name
    if not script.exists():
        sys.exit(f"dimos script not found: {script}")
    if str(dimos_dir) not in sys.path:
        sys.path.insert(0, str(dimos_dir))
    os.chdir(dimos_dir)
    spec = importlib.util.spec_from_file_location("dimos_iphone", script)
    if spec is None or spec.loader is None:
        sys.exit(f"could not build import spec for {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dimos_iphone"] = mod
    spec.loader.exec_module(mod)
    return mod, script


def _resolve_recording_t0(recording_dir: Path) -> float:
    """Best-effort start ts: prefer metadata.json, fall back to first camera ts."""
    meta = recording_dir / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            started = data.get("started_at")
            if started:
                # ISO 8601 — let datetime parse it.
                from datetime import datetime, timezone
                # Accept the SDK's "+00:00" suffix as UTC.
                ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts.timestamp()
        except Exception as e:  # noqa: BLE001
            logger.warning("metadata.json parse failed (%s) — falling back to camera ts0", e)
    ts_path = recording_dir / "camera_timestamps.jsonl"
    if ts_path.exists():
        with ts_path.open() as fh:
            first = json.loads(fh.readline())
            return float(first["ts"])
    return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recording-dir", type=Path, required=True,
        help="directory produced by the on-robot recorder (camera.mp4 + *.jsonl)",
    )

    # Dimos-pipeline flags — same shape as integrations/dimos/server.py.
    parser.add_argument("--depth", choices=["depthpro", "da3"], default="da3")
    parser.add_argument(
        "--da3-model", default="da3metric-large",
        choices=["da3-small", "da3-base", "da3-large",
                 "da3-giant", "da3metric-large", "da3nested-giant-large"],
        help="DA3 variant. da3metric-large (default) returns true metric depth "
             "for cross-frame consistency. da3-small/base/large are scale-ambiguous.",
    )
    parser.add_argument("--pose", choices=["vo", "identity", "external"], default=None,
                        help="default: 'external' (use recorded head_pose.jsonl) when "
                             "head_pose.jsonl is present, else 'vo'. Pass explicitly to "
                             "override.")
    parser.add_argument("--display-width", type=int, default=768)
    parser.add_argument("--max-fps", type=float, default=2.0)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--hfov-deg", type=float, default=70.0,
                        help="Reachy Mini camera HFOV in degrees (override if your sensor differs)")
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--no-loop", action="store_true",
                        help="exit after one pass through the recording (default: loop)")

    # Visualization backend (forwarded to the dimos spatial pipeline)
    parser.add_argument(
        "--viz", choices=["foxglove", "rerun", "both"], default="foxglove",
        help="foxglove = dimos LCM bridge (default). rerun = Rerun viewer via "
             "to_rerun(). both = both at once. (forwarded to the dimos script)",
    )
    parser.add_argument("--rerun-save", default=None,
                        help="with --viz rerun/both: write the Rerun stream to this .rrd "
                             "file instead of spawning a viewer (headless)")
    parser.add_argument("--rerun-connect", action="store_true",
                        help="with --viz rerun/both: connect to an already-running Rerun viewer")
    parser.add_argument("--rerun-point-radius", type=float, default=None,
                        help="Rerun point-cloud radius in meters (smaller = cleaner). "
                             "Forwarded to the dimos pipeline; default 0.008 there.")

    # Sidecar
    parser.add_argument(
        "--sidecar-port", type=int,
        default=int(os.environ.get("REACHYBRAIN_IMU_FOXGLOVE_PORT", 8766)),
        help="Foxglove WS port for the reachy-side topics (0 to disable)",
    )
    parser.add_argument(
        "--dimos-script", default=os.environ.get("DIMOS_SCRIPT", DEFAULT_DIMOS_SCRIPT),
        help="dimos script to load + swap VideoSource into",
    )
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER, default=[],
        help="pass-through args appended to the dimos argv after `--`",
    )

    # Map I/O + keyframe capture flags (forwarded to the dimos script verbatim).
    sys.path.insert(0, str(DIMOS_DIR / "xr-nav" / "src"))
    from xr_nav.cli_args import add_map_io_args, add_keyframe_args
    add_map_io_args(parser)
    add_keyframe_args(parser)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    recording = args.recording_dir.expanduser().resolve()
    if not recording.is_dir():
        sys.exit(f"--recording-dir does not exist: {recording}")

    # Pre-load all JSONL streams + audio RMS chunks. This is well under 100 MB
    # for a few-minute session and keeps publishing lock-free.
    t0 = _resolve_recording_t0(recording)
    imu_events = _read_jsonl(recording / "imu.jsonl")
    joints_events = _read_jsonl(recording / "joints.jsonl")
    head_pose_events = _read_jsonl(recording / "head_pose.jsonl")
    doa_events = _read_jsonl(recording / "doa.jsonl")
    audio_events = _audio_rms_events(recording / "audio.wav", t0=t0)
    print(
        f"[replay] loaded streams: imu={len(imu_events)} joints={len(joints_events)} "
        f"head_pose={len(head_pose_events)} doa={len(doa_events)} audio_chunks={len(audio_events)}"
    )

    # Build pose interpolator from the recorded mechanical head_pose, and
    # resolve the --pose default. We prefer the recorded pose over monocular
    # VO because Reachy's joint kinematics don't drift with depth noise.
    pose_interp: Optional[HeadPoseInterpolator] = None
    if head_pose_events:
        pose_interp = HeadPoseInterpolator(head_pose_events)
        if len(pose_interp) == 0:
            pose_interp = None
    if args.pose is None:
        if pose_interp is not None:
            args.pose = "external"
            print(f"[replay] --pose default: external (using {len(pose_interp)} head_pose entries)")
        else:
            args.pose = "vo"
            print("[replay] --pose default: vo (no head_pose.jsonl)")
    elif args.pose == "external" and pose_interp is None:
        print("[replay] WARNING: --pose external requested but head_pose.jsonl is empty/missing; "
              "dimos will fall back to identity per frame.")

    # 1. Sidecar foxglove (separate port, JSON channels).
    sidecar: SidecarFoxglove | None = None
    if args.sidecar_port > 0:
        sidecar = SidecarFoxglove(port=args.sidecar_port)
        if not sidecar.start():
            sidecar = None

    streams: list[StreamSpec] = []
    if sidecar is not None:
        streams = [
            StreamSpec("imu", imu_events, sidecar.publish_imu),
            StreamSpec("joints", joints_events, sidecar.publish_joints),
            StreamSpec("head_pose", head_pose_events,
                       lambda ts, v: sidecar.publish_head_pose(ts, v)),
            StreamSpec("doa", doa_events, sidecar.publish_doa),
            StreamSpec("audio", audio_events, sidecar.publish_audio_rms),
        ]

    # 2. Load dimos module + swap in our VideoSource. Keep clock external so
    # the sidecar thread can read playback time as the iterator yields frames.
    clock = PlaybackClock()
    mod, script_path = _load_dimos_module(DIMOS_DIR, args.dimos_script)
    mod.VideoSource = make_replay_video_source(
        recording, clock, mod.SourceFrame, pose_interp=pose_interp,
    )

    # 3. Stream drainer (sidecar publishing).
    stop_event = threading.Event()
    drain_thread: threading.Thread | None = None
    if sidecar is not None and streams:
        drain_thread = threading.Thread(
            target=_drain_loop,
            args=(streams, clock, stop_event, sidecar),
            daemon=True, name="reachy-stream-drain",
        )
        drain_thread.start()

    # 4. Build dimos argv. Mirror integrations/dimos/server.py.
    dimos_argv = [
        str(script_path),
        # Our VideoSource ignores --video and reads recording_dir/camera.mp4
        # directly, but we still pass a real path so dimos's argparse(type=Path)
        # check and any incidental cv2 probing don't choke.
        "--video", str(recording / "camera.mp4"),
        "--depth", args.depth,
        "--pose", args.pose,
        "--display-width", str(args.display_width),
        "--max-fps", str(args.max_fps),
        "--device", args.device,
        "--hfov-deg", str(args.hfov_deg),
    ]
    if args.depth == "da3":
        dimos_argv += ["--da3-model", args.da3_model]
    if args.no_detect:
        dimos_argv.append("--no-detect")
    if args.no_loop:
        dimos_argv.append("--no-loop")

    # Forward visualization backend selection to the dimos pipeline
    dimos_argv += ["--viz", args.viz]
    if args.rerun_save is not None:
        dimos_argv += ["--rerun-save", str(args.rerun_save)]
    if args.rerun_connect:
        dimos_argv.append("--rerun-connect")
    if args.rerun_point_radius is not None:
        dimos_argv += ["--rerun-point-radius", str(args.rerun_point_radius)]

    # Forward map-I/O flags
    if args.save_map is not None:
        dimos_argv += ["--save-map", str(args.save_map)]
    if args.load_map is not None:
        dimos_argv += ["--load-map", str(args.load_map)]
    if args.save_map_every_n:
        dimos_argv += ["--save-map-every-n", str(args.save_map_every_n)]
    if args.save_ply is not None:
        dimos_argv += ["--save-ply", str(args.save_ply)]
    if args.save_pcd is not None:
        dimos_argv += ["--save-pcd", str(args.save_pcd)]
    if args.save_cloud_with_map:
        dimos_argv.append("--save-cloud-with-map")
    if args.cloud_min_observations is not None:
        dimos_argv += ["--cloud-min-observations", str(args.cloud_min_observations)]

    # Forward keyframe-capture flags
    if args.save_keyframes is not None:
        dimos_argv += ["--save-keyframes", str(args.save_keyframes)]
    if args.save_keyframes_every_n != 1:
        dimos_argv += ["--save-keyframes-every-n", str(args.save_keyframes_every_n)]
    if args.keyframe_rgb_format != "png":
        dimos_argv += ["--keyframe-rgb-format", args.keyframe_rgb_format]

    dimos_argv += args.extra

    sys.argv = dimos_argv
    print(f"[replay] dimos argv: {' '.join(dimos_argv)}\n")
    print("[replay] visualization:")
    if args.viz in ("foxglove", "both"):
        print("   spatial scene (Foxglove via dimos LCM bridge):  ws://localhost:8765")
    if args.viz in ("rerun", "both"):
        if args.rerun_save:
            print(f"   spatial scene (Rerun):  save -> {args.rerun_save}")
        else:
            print("   spatial scene (Rerun):  viewer window")
    if sidecar is not None:
        # Reachy telemetry (IMU/joints/head_pose/DOA) is a JSON Foxglove server and
        # stays on Foxglove regardless of --viz; those streams aren't dimos to_rerun() msgs.
        print(f"   reachy telemetry (Foxglove sidecar, always):    ws://localhost:{args.sidecar_port}")
    print("Ctrl+C to exit.\n")

    # 5. Run dimos pipeline.
    try:
        mod.main()
    except KeyboardInterrupt:
        print("[replay] interrupted")
    finally:
        stop_event.set()
        clock.mark_eof()
        if drain_thread is not None:
            drain_thread.join(timeout=2.0)
        if sidecar is not None:
            sidecar.stop()


if __name__ == "__main__":
    main()
