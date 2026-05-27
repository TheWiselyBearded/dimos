"""Viture color + pose -> aligned 3D spatial memory + tracked 3D objects.

Mirrors mac_unitree_replay_foxglove.py for the wearable Viture XR rig:
  - Pluggable depth (Apple Depth Pro | Depth Anything 3 | stereo SGBM)
  - Pluggable source (offline RecordingLoader | live VitureClient TCP stream)
  - VoxelMap fusion (confidence-weighted centroids + raycast free-space clearing)
    instead of the naive voxel_down_sample accumulator the per-model demos use
  - YOLOE (LRPC, prompt-free) -> ObjectDB pending->permanent 3D object tracking
  - Optional CLIP SpatialMemory for text/image queries against the recorded run

Run (recorded, default):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_spatial_foxglove.py \\
        --source recording --depth depthpro

Run (live):
    /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_spatial_foxglove.py \\
      --source live --depth depthpro

Pair with the foxglove bridge in another terminal:
    /opt/anaconda3/envs/xr-nav/bin/python -m \\
      dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Foxglove (ws://localhost:8765):
  - 3D panel (frame=world): /map, /object_clouds, /scene_update, /tf, /points_frame
  - Image panels: /color_image (info /camera_info)  /depth (info /depth_camera_info)
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

DEFAULT_VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_left.mp4")
DEFAULT_RIGHT_VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_right.mp4")
DEFAULT_RECORDING_DIR = Path(
    "/Users/reza/Documents/Projects/VitureSplat/viture_sdk/data/VITURE_recording_2026-03-29_14-34-22"
)

sys.path.insert(0, str(REPO / "xr-nav" / "src"))
sys.path.insert(0, str(REPO / "xr-nav" / "awesome-depth-anything-3" / "src"))

# -------- Defaults (most are exposed as CLI flags below) --------
HFOV_DEG = 46.0
DISPLAY_W = 1024              # color resampled to this; depth model also runs here

DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0

# Depth-edge filter: erode away pixels where local depth gradient is steep.
# This is the single biggest noise win — kills the "ribbon" streaks that
# appear when a pixel at an object edge gets mapped to a 3D point halfway
# between foreground and background. Threshold is in meters per pixel; raise
# to keep more edge points (more detail, more streaks), lower to be stricter.
DEPTH_EDGE_GRAD_THRESHOLD_M = 0.10
DEPTH_EDGE_DILATE_PX = 2

POINTS_STRIDE = 6             # subsample dense per-frame cloud before publish

VOXEL_M = 0.05
VOXEL_MAX_RANGE_M = 8.0
VOXEL_INSERT_MAX_DRIFT_M = 0.04
VOXEL_RAYCAST_SUBSAMPLE = 8
VOXEL_RAYCAST_MAX_MISSES = 3
# Render voxels only if observed at least this many frames. The single biggest
# noise win on the map side — single-frame artifacts (transient depth glitches,
# pose jitter spikes) never become visible.
VOXEL_MIN_OBSERVATIONS = 2

MAP_PUBLISH_EVERY_N = 5

# ObjectDB
# YOLOE LRPC (prompt-free) doesn't always supply stable track_ids, so promotion
# leans on distance matching — which is noisy when depth jitters frame-to-frame.
# Loosen distance + drop promotion threshold so the user sees boxes quickly.
OBJECTS_DIST_THRESHOLD_M = 0.4
OBJECTS_MIN_DETECTIONS = 2
OBJECTS_PUBLISH_EVERY_N = 1
# Skip the per-detection RGBD reprojection every N frames. ObjectDB still keeps
# the previous detections; it just doesn't get fresh ones every frame. Cheap
# way to reclaim ~50-200ms when many objects are visible.
OBJECTS_PROCESS_EVERY_N = 1

# Stereo (only used when --depth stereo)
STEREO_BASELINE_M = 0.063
STEREO_NUM_DISP = 96
STEREO_BLOCK = 7

# CLIP SpatialMemory
SPATIAL_MIN_DISTANCE_M = 0.10
SPATIAL_MIN_INTERVAL_S = 1.0


# ===========================================================================
#  Pose / TF helpers (lifted from existing mac_viture_*_foxglove.py scripts)
# ===========================================================================

def arkit_c2w_to_opencv(c2w: np.ndarray) -> np.ndarray:
    """Viture pose (ARKit, +Y up, +Z back) -> OpenCV optical (+Y down, +Z fwd)."""
    out = c2w.copy()
    out[:3, 1] *= -1
    out[:3, 2] *= -1
    return out


def c2w_to_translation_quat(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation
    return c2w[:3, 3].astype(np.float64), Rotation.from_matrix(c2w[:3, :3]).as_quat()


def transform_points(c2w: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    rot = c2w[:3, :3].astype(np.float32)
    trans = c2w[:3, 3].astype(np.float32)
    return (rot @ pts_cam.astype(np.float32).T).T + trans


def make_tf_msg(c2w: np.ndarray, ts: float, parent: str = "world",
                child: str = "camera_optical"):
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from dimos_lcm.geometry_msgs.TransformStamped import TransformStamped
    from dimos_lcm.geometry_msgs.Transform import Transform as LCMT
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion as LQ
    from dimos_lcm.std_msgs.Header import Header
    from dimos_lcm.std_msgs.Time import Time

    sec = int(ts); nsec = int((ts - sec) * 1e9)
    t, q = c2w_to_translation_quat(c2w)
    s = TransformStamped()
    s.header = Header(); s.header.stamp = Time()
    s.header.stamp.sec = sec; s.header.stamp.nsec = nsec
    s.header.frame_id = parent; s.child_frame_id = child
    s.transform = LCMT()
    s.transform.translation = LV3()
    s.transform.translation.x = float(t[0])
    s.transform.translation.y = float(t[1])
    s.transform.translation.z = float(t[2])
    s.transform.rotation = LQ()
    s.transform.rotation.x = float(q[0])
    s.transform.rotation.y = float(q[1])
    s.transform.rotation.z = float(q[2])
    s.transform.rotation.w = float(q[3])

    msg = TFMessage()
    msg.transforms = [s]
    msg.transforms_length = 1
    return msg


def make_camera_to_world_transform(c2w_opencv: np.ndarray, ts: float):
    """Build a dimos.msgs.Transform whose frame_id="world", child_frame_id="camera_optical".

    `Object.from_2d_to_list` uses this to push per-object pointclouds into world
    frame (it calls pc.transform(camera_transform); see object.py:251).
    """
    from dimos.msgs.geometry_msgs import Transform, Vector3, Quaternion
    t, q = c2w_to_translation_quat(c2w_opencv)
    return Transform(
        translation=Vector3(float(t[0]), float(t[1]), float(t[2])),
        rotation=Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        frame_id="world",
        child_frame_id="camera_optical",
        ts=ts,
    )


# ===========================================================================
#  Camera intrinsics
# ===========================================================================

def make_camera_info(width: int, height: int, hfov_deg: float):
    """Pinhole intrinsics for an undistorted Viture frame at the given size."""
    from dimos.msgs.sensor_msgs import CameraInfo
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return CameraInfo(
        frame_id="camera_optical", height=height, width=width,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )


def filter_depth_edges(depth_m: np.ndarray, grad_threshold_m: float,
                       dilate_px: int) -> np.ndarray:
    """Zero out depth pixels near steep depth discontinuities.

    These edge pixels are the source of the "ribbon" streaks in the world map:
    each one back-projects to a 3D point stuck halfway between the foreground
    and background object, and once accumulated across frames they smear into
    sheets connecting unrelated surfaces.
    """
    if grad_threshold_m <= 0:
        return depth_m
    valid = depth_m > 0
    gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    bad = (grad_mag > grad_threshold_m) & valid
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        bad = cv2.dilate(bad.astype(np.uint8), k).astype(bool)
    out = depth_m.copy()
    out[bad] = 0.0
    return out


def depth_to_confidence(depth_m: np.ndarray) -> np.ndarray:
    """Per-point confidence ~ 1/depth^2 — far points get less weight when fused.

    DepthPro/DA3 don't expose pixelwise confidence, but depth error grows roughly
    quadratically with range for any monocular model. Down-weighting far points
    in VoxelMap.insert reduces the smearing seen at room boundaries.
    """
    conf = np.where(depth_m > 0, 1.0 / np.maximum(depth_m, 0.1) ** 2, 0.0)
    return conf.astype(np.float32)


def stereo_intrinsics_from_calibration(cal_path: Path,
                                       img_size: tuple[int, int]) -> tuple[float, float, float]:
    """Recover undistorted (fx, cx, cy) for the stereo .mp4s. Mirrors the
    derivation in mac_viture_stereo_foxglove.py:78-101."""
    with open(cal_path) as f:
        cal = json.load(f)
    K = np.array([[cal["fx"], 0.0, cal["cx"]],
                  [0.0, cal["fx"], cal["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([cal.get(k, 0.0) for k in ("k1", "k2", "k3", "k4")], dtype=np.float64)
    balance = cal.get("balance", 0.0)
    fov_scale = cal.get("fov_scale", 100) / 100.0
    if cal.get("distortion_model") in ("equidistant", "fisheye", "kannala_brandt4"):
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, img_size, np.eye(3, dtype=np.float64), balance=balance)
    else:
        Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha=balance)
    Knew[0, 0] /= fov_scale
    Knew[1, 1] /= fov_scale
    return float(Knew[0, 0]), float(Knew[0, 2]), float(Knew[1, 2])


# ===========================================================================
#  Frame source: recording (mp4 + RecordingLoader poses) or live (VitureClient)
# ===========================================================================

@dataclass
class SourceFrame:
    color_bgr: np.ndarray            # [H, W, 3] uint8 BGR (left)
    color_right_bgr: np.ndarray | None  # optional right cam for stereo
    c2w_arkit: np.ndarray | None     # 4x4, ARKit convention (None = use identity)
    ts: float
    frame_idx: int


class FrameSource(ABC):
    @abstractmethod
    def __iter__(self): ...
    def close(self) -> None: ...


class RecordingSource(FrameSource):
    """Plays back a Viture recording: left .mp4 + RecordingLoader for poses.

    Optional right .mp4 enables --depth stereo without falling back to PGM.
    """

    def __init__(self, recording_dir: Path, video_path: Path,
                 right_video_path: Path | None = None, fps_cap: float = 10.0):
        from xr_nav.recording_loader import RecordingLoader
        if not video_path.exists():
            sys.exit(f"missing video: {video_path}")
        if not recording_dir.exists():
            sys.exit(f"missing recording dir: {recording_dir}")
        self._video_path = video_path
        self._right_video_path = right_video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            sys.exit(f"cv2 could not open {video_path}")
        self._cap_right: cv2.VideoCapture | None = None
        if right_video_path is not None and right_video_path.exists():
            self._cap_right = cv2.VideoCapture(str(right_video_path))
            if not self._cap_right.isOpened():
                print(f"[source] right video failed to open: {right_video_path}")
                self._cap_right = None
        self._loader = RecordingLoader(str(recording_dir), step=1)
        self._period = 1.0 / max(fps_cap, 0.1)
        self._W = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._H = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[source:recording] {video_path.name} {self._W}x{self._H} {self._total} frames")

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._W, self._H

    def __iter__(self):
        loop = 0
        while True:
            ok, frame_bgr = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self._cap_right is not None:
                    self._cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop += 1
                print(f"  --- loop {loop} ---")
                continue

            if frame_bgr.ndim == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

            right_bgr = None
            if self._cap_right is not None:
                ok_r, right_bgr = self._cap_right.read()
                if not ok_r:
                    right_bgr = None
                elif right_bgr.ndim == 2:
                    right_bgr = cv2.cvtColor(right_bgr, cv2.COLOR_GRAY2BGR)

            mp4_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            c2w = self._loader.pose_for_frame_index(mp4_idx)
            ts = time.time()
            yield SourceFrame(color_bgr=frame_bgr, color_right_bgr=right_bgr,
                              c2w_arkit=c2w, ts=ts, frame_idx=mp4_idx)

            # Pace the loop (depth model often dominates; this is just a ceiling)
            time.sleep(0)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        if self._cap_right is not None:
            try:
                self._cap_right.release()
            except Exception:
                pass


class LiveSource(FrameSource):
    """Live Viture stream over TCP via xr_nav.viture_client.VitureClient.

    Frames come in as grayscale [H, W] uint8 (see viture_client.py:23-27); we
    upsample to BGR so the rest of the pipeline doesn't have to special-case it.
    """

    def __init__(self):
        from xr_nav.viture_client import VitureClient
        self._client = VitureClient()
        self._client.start()
        print("[source:live] VitureClient started; waiting for frames...")

    def __iter__(self):
        idx = 0
        while True:
            f = self._client.get_frame(timeout=2.0)
            if f is None:
                print("[source:live] timeout waiting for frame, retrying...")
                continue
            left = cv2.cvtColor(f.img_left, cv2.COLOR_GRAY2BGR)
            right = cv2.cvtColor(f.img_right, cv2.COLOR_GRAY2BGR) if f.img_right is not None else None
            yield SourceFrame(color_bgr=left, color_right_bgr=right,
                              c2w_arkit=f.pose_4x4, ts=f.timestamp, frame_idx=idx)
            idx += 1

    def close(self) -> None:
        try:
            self._client.stop()
        except Exception:
            pass


# ===========================================================================
#  Depth estimators (all return (depth_m float32 [H,W], conf float32 [H,W]))
# ===========================================================================

class DepthEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def infer(self, color_rgb: np.ndarray, color_right_rgb: np.ndarray | None,
              fx: float) -> tuple[np.ndarray, np.ndarray]: ...

    def warmup(self) -> None:
        pass


class DepthProEstimator(DepthEstimator):
    name = "depthpro"

    def __init__(self, device: str = "mps"):
        try:
            import depth_pro
        except ImportError:
            sys.exit(
                "depth_pro not installed. Install with:\n"
                "  /opt/anaconda3/envs/xr-nav/bin/pip install "
                "git+https://github.com/apple/ml-depth-pro.git\n"
                "Then download checkpoints (~2GB) per the depth_pro README."
            )
        import torch
        self._torch = torch
        print(f"[depth] loading depth-pro on {device}...")
        t0 = time.monotonic()
        model, transform = depth_pro.create_model_and_transforms()
        model.eval()
        self._model = model.to(torch.device(device))
        self._transform = transform
        self._device = torch.device(device)
        print(f"[depth] depth-pro ready in {time.monotonic() - t0:.1f}s")

    def infer(self, color_rgb, color_right_rgb, fx):
        with self._torch.no_grad():
            inp = self._transform(color_rgb).to(self._device)
            # MPS does not support float64; force float32 for f_px
            f_px = self._torch.tensor(float(fx), dtype=self._torch.float32, device=self._device)
            pred = self._model.infer(inp, f_px=f_px)
        depth_m = pred["depth"].detach().cpu().numpy().astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m[0]
        conf = np.ones_like(depth_m, dtype=np.float32)  # depth-pro doesn't expose pixelwise conf
        return depth_m, conf


class DA3Estimator(DepthEstimator):
    """Depth Anything 3, with a one-time scale fit so the map ends up metric.

    DA3 is scale-ambiguous unless its bundled metric prior fires (pred.is_metric).
    When non-metric, we fit a single global scale by sniffing the median depth
    of the first frame against a sane prior (1.5m) — crude but stable enough for
    fusion + tracking. For tighter scale, run --depth depthpro.
    """
    name = "da3"

    def __init__(self, model_name: str = "da3metric-large", device: str = "mps",
                 process_res: int = 504, conf_threshold: float = 0.5):
        from depth_anything_3.api import DepthAnything3
        print(f"[depth] loading {model_name} on {device}...")
        self._model = DepthAnything3(model_name=model_name, device=device)
        self._res = process_res
        self._conf_thresh = conf_threshold
        self._scale: float | None = None

    def infer(self, color_rgb, color_right_rgb, fx):
        pred = self._model.inference(image=[color_rgb], process_res=self._res)
        raw = np.nan_to_num(pred.depth[0].astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        is_metric = bool(getattr(pred, "is_metric", 0))
        if is_metric:
            depth_m = raw
        else:
            # Per-frame normalize to [near, far], then apply learned global scale
            d_min, d_max = float(raw.min()), float(raw.max())
            if d_max - d_min < 1e-8:
                depth_norm = np.full_like(raw, 0.5 * (DEPTH_NEAR_M + DEPTH_FAR_M))
            else:
                d_norm = (raw - d_min) / (d_max - d_min)
                depth_norm = DEPTH_NEAR_M + d_norm * (DEPTH_FAR_M - DEPTH_NEAR_M)
            if self._scale is None:
                # First-frame scale fit: hard-target median depth = 1.5m
                med = float(np.median(depth_norm[depth_norm > 0]))
                self._scale = 1.5 / med if med > 1e-6 else 1.0
                print(f"[depth] DA3 first-frame scale fit: {self._scale:.3f}")
            depth_m = (depth_norm * self._scale).astype(np.float32)

        conf_map = np.ones_like(depth_m, dtype=np.float32)
        if pred.conf is not None:
            c = pred.conf[0].astype(np.float32)
            cmax = float(c.max()) if c.size else 1.0
            if cmax > 1.0:
                c = c / cmax
            conf_map = c
            depth_m = np.where(c >= self._conf_thresh, depth_m, 0.0).astype(np.float32)
        return depth_m, conf_map


class StereoEstimator(DepthEstimator):
    name = "stereo"

    def __init__(self, fx: float, baseline_m: float = STEREO_BASELINE_M,
                 num_disp: int = STEREO_NUM_DISP, block: int = STEREO_BLOCK):
        self._fx = fx
        self._baseline = baseline_m
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8 * 3 * block * block, P2=32 * 3 * block * block,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def infer(self, color_rgb, color_right_rgb, fx):
        if color_right_rgb is None:
            raise RuntimeError("--depth stereo needs a right-camera frame; pass --right-video or use --source live")
        gray_l = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(color_right_rgb, cv2.COLOR_RGB2GRAY)
        disp = self._sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
        valid = disp > 0.5
        depth_m = np.zeros_like(disp, dtype=np.float32)
        np.divide(self._fx * self._baseline, disp, out=depth_m, where=valid)
        depth_m = np.where((depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_FAR_M),
                           depth_m, 0.0).astype(np.float32)
        conf = valid.astype(np.float32)
        return depth_m, conf


def make_depth_estimator(kind: str, fx: float, device: str = "mps",
                         da3_model: str = "da3metric-large") -> DepthEstimator:
    if kind == "depthpro":
        return DepthProEstimator(device=device)
    if kind == "da3":
        return DA3Estimator(model_name=da3_model, device=device)
    if kind == "stereo":
        return StereoEstimator(fx=fx)
    raise ValueError(f"unknown depth kind: {kind}")


# ===========================================================================
#  YOLOE detector that auto-downloads weights via Ultralytics
#  (bypasses dimos's LFS-managed `data/models_yoloe/` which isn't pulled here)
# ===========================================================================

YOLOE_WEIGHTS_DIR = REPO / "checkpoints"


class LocalYoloeDetector:
    """Mirror of dimos.perception.detection.detectors.yoloe.Yoloe2DDetector but
    loads weights from a local path (auto-downloaded by Ultralytics on first use)
    instead of dimos's Git-LFS-managed model archive.
    """

    def __init__(self, device: str = "mps", weights_name: str = "yoloe-11s-seg-pf.pt",
                 max_area_ratio: float | None = 0.3):
        from ultralytics import YOLOE
        import threading
        YOLOE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = YOLOE_WEIGHTS_DIR / weights_name
        if not weights_path.exists():
            print(f"[detector] downloading YOLOE weights -> {weights_path}")
        self.model = YOLOE(str(weights_path))  # Ultralytics auto-downloads if missing
        self.device = device
        self.max_area_ratio = max_area_ratio
        self._lock = threading.Lock()

    def process_image(self, image):
        from dimos.perception.detection.type import ImageDetections2D
        with self._lock:
            results = self.model.track(
                source=image.to_opencv(), device=self.device,
                conf=0.6, iou=0.6, persist=True, verbose=False,
            )
        detections = ImageDetections2D.from_ultralytics_result(image, results)
        if self.max_area_ratio is None:
            return detections
        image_area = image.width * image.height
        if image_area <= 0:
            return detections
        kept = [d for d in detections.detections
                if d.bbox_2d_volume() / image_area <= self.max_area_ratio]
        return ImageDetections2D(image, kept)

    def stop(self) -> None:
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            predictor = self.model.predictor
            if hasattr(predictor, "trackers") and predictor.trackers:
                for tracker in predictor.trackers:
                    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "gmc"):
                        gmc = tracker.tracker.gmc
                        if hasattr(gmc, "executor") and gmc.executor is not None:
                            gmc.executor.shutdown(wait=True)
            self.model.predictor = None


# ===========================================================================
#  Foxglove SceneUpdate from a list of permanent ObjectDB Object instances
# ===========================================================================

def _stable_color_from_id(object_id: str, alpha: float = 0.25) -> tuple[float, float, float, float]:
    """Deterministic RGB from object_id hash so a tracked object keeps its color."""
    try:
        seed = int(object_id, 16)
    except (ValueError, TypeError):
        seed = abs(hash(object_id))
    rng = np.random.default_rng(seed % (2**32 - 1))
    r, g, b = rng.uniform(0.3, 1.0, size=3)
    return float(r), float(g), float(b), alpha


def build_scene_update_for_objects(objects: list[Any], ts: float):
    """Manual SceneUpdate builder for ObjectDB Object instances.

    Mirrors Detection3DPC.to_foxglove_scene_entity (pointcloud.py:110) but uses
    a stable per-object color so tracked objects don't flash colors across frames.
    """
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from lcm_msgs.builtin_interfaces import Duration
    from lcm_msgs.foxglove_msgs import CubePrimitive, SceneEntity, TextPrimitive
    from lcm_msgs.geometry_msgs import Point, Pose, Quaternion as LCMQ, Vector3 as LCMV3
    from dimos.msgs.foxglove_msgs.Color import Color
    from dimos.types.timestamped import to_ros_stamp

    update = SceneUpdate()
    update.deletions_length = 0
    update.deletions = []
    entities: list[Any] = []

    for obj in objects:
        try:
            aabb = obj.pointcloud.axis_aligned_bounding_box
        except Exception:
            continue
        center = aabb.get_center()
        extent = aabb.get_extent()
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
            continue

        cube = CubePrimitive()
        cube.pose = Pose()
        cube.pose.position = Point()
        cube.pose.position.x = float(center[0])
        cube.pose.position.y = float(center[1])
        cube.pose.position.z = float(center[2])
        cube.pose.orientation = LCMQ()
        cube.pose.orientation.x = 0.0
        cube.pose.orientation.y = 0.0
        cube.pose.orientation.z = 0.0
        cube.pose.orientation.w = 1.0
        cube.size = LCMV3()
        cube.size.x = float(extent[0])
        cube.size.y = float(extent[1])
        cube.size.z = float(extent[2])
        r, g, b, a = _stable_color_from_id(obj.object_id)
        cube_color = Color()
        cube_color.r = r; cube_color.g = g; cube_color.b = b; cube_color.a = a
        cube.color = cube_color

        text = TextPrimitive()
        text.pose = Pose()
        text.pose.position = Point()
        text.pose.position.x = float(center[0])
        text.pose.position.y = float(center[1])
        text.pose.position.z = float(center[2]) + float(extent[2]) / 2.0 + 0.1
        text.pose.orientation = LCMQ()
        text.pose.orientation.x = 0.0
        text.pose.orientation.y = 0.0
        text.pose.orientation.z = 0.0
        text.pose.orientation.w = 1.0
        text.billboard = True
        text.font_size = 18.0
        text.scale_invariant = True
        tcolor = Color()
        tcolor.r = 1.0; tcolor.g = 1.0; tcolor.b = 1.0; tcolor.a = 1.0
        text.color = tcolor
        text.text = f"{obj.name} #{obj.object_id} ({obj.detections_count})"

        entity = SceneEntity()
        entity.timestamp = to_ros_stamp(ts)
        entity.frame_id = "world"
        entity.id = obj.object_id
        entity.lifetime = Duration()
        entity.lifetime.sec = 0
        entity.lifetime.nanosec = 0
        entity.frame_locked = False
        entity.metadata_length = 0; entity.metadata = []
        entity.arrows_length = 0; entity.arrows = []
        entity.cubes_length = 1; entity.cubes = [cube]
        entity.spheres_length = 0; entity.spheres = []
        entity.cylinders_length = 0; entity.cylinders = []
        entity.lines_length = 0; entity.lines = []
        entity.triangles_length = 0; entity.triangles = []
        entity.texts_length = 1; entity.texts = [text]
        entity.models_length = 0; entity.models = []
        entities.append(entity)

    update.entities = entities
    update.entities_length = len(entities)
    return update


# ===========================================================================
#  Optional CLIP SpatialMemory adapter (direct calls, no dimos Module wiring)
# ===========================================================================

class SpatialMemoryAdapter:
    """Replicates SpatialMemory._process_frame gating (spatial_perception.py:220)
    without dragging the full Module/TF graph into the demo."""

    def __init__(self, db_path: Path):
        try:
            import chromadb
            from chromadb.config import Settings
            from dimos.agents_deprecated.memory.image_embedding import ImageEmbeddingProvider
            from dimos.agents_deprecated.memory.spatial_vector_db import SpatialVectorDB
            from dimos.agents_deprecated.memory.visual_memory import VisualMemory
        except Exception as e:
            sys.exit(f"--enable-clip-memory requested but imports failed: {e}")
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False))
        self._embed = ImageEmbeddingProvider(model_name="clip", dimensions=512)
        self._vmem = VisualMemory(output_dir=str(db_path))
        self._db = SpatialVectorDB(
            collection_name="spatial_memory", chroma_client=client,
            visual_memory=self._vmem, embedding_provider=self._embed)
        self._last_pos: np.ndarray | None = None
        self._last_t: float | None = None
        self._stored = 0
        print(f"[spatial-mem] enabled (db at {db_path})")

    def maybe_store(self, color_bgr: np.ndarray, c2w_world: np.ndarray, ts: float) -> None:
        pos = c2w_world[:3, 3]
        if self._last_pos is not None and np.linalg.norm(pos - self._last_pos) < SPATIAL_MIN_DISTANCE_M:
            return
        if self._last_t is not None and (ts - self._last_t) < SPATIAL_MIN_INTERVAL_S:
            return
        try:
            emb = self._embed.get_embedding(color_bgr)
        except Exception as e:
            print(f"[spatial-mem] embedding failed: {e}")
            return
        from datetime import datetime
        import uuid
        frame_id = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_matrix(c2w_world[:3, :3]).as_euler("xyz")
        meta = {
            "pos_x": float(pos[0]), "pos_y": float(pos[1]), "pos_z": float(pos[2]),
            "rot_x": float(euler[0]), "rot_y": float(euler[1]), "rot_z": float(euler[2]),
            "timestamp": ts, "frame_id": frame_id,
        }
        self._db.add_image_vector(vector_id=frame_id, image=color_bgr,
                                  embedding=emb, metadata=meta)
        self._last_pos = pos
        self._last_t = ts
        self._stored += 1
        if self._stored % 10 == 0:
            print(f"[spatial-mem] stored {self._stored} frames")


# ===========================================================================
#  Main loop
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["recording", "live"], default="recording")
    parser.add_argument("--depth", choices=["depthpro", "da3", "stereo"], default="depthpro")
    parser.add_argument("--da3-model", default="da3metric-large",
                        choices=["da3-small", "da3-base", "da3-large",
                                 "da3-giant", "da3metric-large",
                                 "da3nested-giant-large"],
                        help="DA3 variant when --depth da3. Default da3metric-large "
                             "returns true metric depth (consistent across frames). "
                             "Use da3-small/base/large for faster but scale-ambiguous depth.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="left undistorted .mp4 (recording mode)")
    parser.add_argument("--right-video", type=Path, default=DEFAULT_RIGHT_VIDEO, help="right .mp4 for --depth stereo")
    parser.add_argument("--recording-dir", type=Path, default=DEFAULT_RECORDING_DIR)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--enable-clip-memory", action="store_true",
                        help="record CLIP-embedded frames to ChromaDB for text/image queries")
    parser.add_argument("--clip-db", type=Path,
                        default=Path.home() / ".dimos" / "spatial_memory" / "viture",
                        help="persistence dir for CLIP SpatialMemory (with --enable-clip-memory)")
    parser.add_argument("--display-width", type=int, default=DISPLAY_W,
                        help="resample input to this width before depth inference. "
                             "DROP THIS FIRST for latency (1024->512 ~= 4x faster depth)")
    parser.add_argument("--hfov-deg", type=float, default=HFOV_DEG)
    parser.add_argument("--max-fps", type=float, default=10.0,
                        help="upper-bound publish rate; depth model often sets the real ceiling")

    # ---- Latency tunables ----
    perf = parser.add_argument_group("latency tunables (raise N to skip work)")
    perf.add_argument("--objects-process-every-n", type=int, default=OBJECTS_PROCESS_EVERY_N,
                      help="run Object.from_2d_to_list every N frames (per-detection RGBD reproj is "
                           "the second-biggest cost after depth)")
    perf.add_argument("--map-publish-every-n", type=int, default=MAP_PUBLISH_EVERY_N,
                      help="republish /map every N frames (large pointcloud serialize is ~50ms)")
    perf.add_argument("--raycast-every-n", type=int, default=1,
                      help="run VoxelMap.raycast_clear every N frames (raycast is ~30-80ms)")
    perf.add_argument("--points-stride", type=int, default=POINTS_STRIDE,
                      help="subsample factor for the per-frame cloud before VoxelMap insert "
                           "(higher = sparser = faster + less noisy)")

    # ---- Noise tunables ----
    nz = parser.add_argument_group("noise tunables (raise to suppress ghosts/streaks)")
    nz.add_argument("--depth-edge-threshold", type=float, default=DEPTH_EDGE_GRAD_THRESHOLD_M,
                    help="zero out depth pixels with |grad|>this (m/pixel). 0 to disable. "
                         "BIGGEST single noise win — kills the ribbon streaks at object edges")
    nz.add_argument("--depth-edge-dilate", type=int, default=DEPTH_EDGE_DILATE_PX,
                    help="dilate the edge-rejection mask by this many pixels (0 to disable)")
    nz.add_argument("--voxel-size", type=float, default=VOXEL_M,
                    help="voxel grid size (m). Larger = smoother map, less detail")
    nz.add_argument("--voxel-min-observations", type=int, default=VOXEL_MIN_OBSERVATIONS,
                    help="only render voxels seen >= N times. Suppresses single-frame artifacts")
    nz.add_argument("--voxel-max-drift", type=float, default=VOXEL_INSERT_MAX_DRIFT_M,
                    help="reject points within this distance of an existing voxel "
                         "(higher = more aggressive drift suppression)")
    nz.add_argument("--use-depth-confidence", action="store_true",
                    help="weight VoxelMap inserts by 1/depth^2 — down-weights noisy far points")

    from xr_nav.cli_args import add_map_io_args, add_keyframe_args
    add_map_io_args(parser)
    add_keyframe_args(parser)

    args = parser.parse_args()

    # ---- LCM transports & dimos message types ----
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image, ImageFormat, PointCloud2, CameraInfo
    from dimos.perception.detection.module2D import Detection2DModule
    from dimos.perception.detection.objectDB import ObjectDB
    from dimos.perception.detection.type.detection3d.object import Object, aggregate_pointclouds
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from xr_nav.voxel_map import VoxelMap, RaycastConfig
    from xr_nav.map_io import MapArtifactWriter, load_map_bundle
    from xr_nav.keyframe_recorder import KeyframeRecorder
    from xr_nav.cli_args import resolve_cloud_min_observations

    # ---- Source ----
    if args.source == "recording":
        right = args.right_video if args.depth == "stereo" else None
        source: FrameSource = RecordingSource(
            recording_dir=args.recording_dir,
            video_path=args.video,
            right_video_path=right,
            fps_cap=args.max_fps,
        )
    else:
        source = LiveSource()

    # ---- Camera intrinsics + frame-resampled dims ----
    # Pull a probe frame to learn input size, then settle DW/DH.
    iterator = iter(source)
    probe = next(iterator)
    H, W = probe.color_bgr.shape[:2]
    scale = args.display_width / W
    DW = args.display_width
    DH = int(H * scale)
    cam_info = make_camera_info(DW, DH, args.hfov_deg)
    fx = cam_info.K[0]
    K = np.array([[fx, 0, DW / 2.0],
                  [0, fx, DH / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    print(f"[main] color {W}x{H} -> resampled {DW}x{DH}  HFOV={args.hfov_deg}°  fx={fx:.1f}")

    # ---- Depth ----
    depth_estimator = make_depth_estimator(args.depth, fx=fx, device=args.device,
                                           da3_model=args.da3_model)

    # ---- Detection ----
    print("[main] warming YOLOE detector (may download weights on first run)...")
    det2d = Detection2DModule(detector=lambda: LocalYoloeDetector(device=args.device))
    object_db = ObjectDB(
        distance_threshold=OBJECTS_DIST_THRESHOLD_M,
        min_detections_for_permanent=OBJECTS_MIN_DETECTIONS,
    )

    # ---- Voxel map (replaces MapAccumulator) ----
    voxel_map = VoxelMap(voxel_size=args.voxel_size, max_range=VOXEL_MAX_RANGE_M)
    raycast_cfg = RaycastConfig(
        subsample=VOXEL_RAYCAST_SUBSAMPLE, max_misses=VOXEL_RAYCAST_MAX_MISSES)

    if args.load_map is not None:
        bundle = load_map_bundle(args.load_map)
        voxel_map.load_state(bundle["voxel_map"])
        object_db.load_state(bundle["object_db"])

    map_writer = MapArtifactWriter(
        save_map=args.save_map,
        save_ply=args.save_ply,
        save_pcd=args.save_pcd,
        save_cloud_with_map=args.save_cloud_with_map,
        cloud_min_observations=resolve_cloud_min_observations(args),
    )
    keyframe_recorder = (
        KeyframeRecorder(args.save_keyframes, rgb_format=args.keyframe_rgb_format)
        if args.save_keyframes is not None else None
    )

    # ---- Optional CLIP SpatialMemory ----
    spatial_mem: SpatialMemoryAdapter | None = None
    if args.enable_clip_memory:
        spatial_mem = SpatialMemoryAdapter(db_path=args.clip_db)

    # ---- Foxglove transports ----
    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)
    depth_cam_info_topic = LCMTransport("/depth_camera_info", CameraInfo)
    depth_topic = LCMTransport("/depth", Image)
    points_topic = LCMTransport("/points_frame", PointCloud2)
    map_topic = LCMTransport("/map", PointCloud2)
    obj_cloud_topic = LCMTransport("/object_clouds", PointCloud2)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    print("\n[main] publishing. Foxglove: ws://localhost:8765")
    print("  3D panel (frame=world): /map  /object_clouds  /scene_update  /tf  /points_frame")
    print("  Image panels: /color_image  /depth")
    print("Ctrl+C to exit.\n")

    n = 0
    period = 1.0 / max(args.max_fps, 0.1)

    # Re-prepend the probe frame so we don't drop it
    def frame_stream():
        yield probe
        yield from iterator

    try:
        for sf in frame_stream():
            t_start = time.perf_counter()
            ts = sf.ts

            # ---- Resample color ----
            small_bgr = cv2.resize(sf.color_bgr, (DW, DH))
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            color_msg = Image.from_numpy(small_bgr, format=ImageFormat.BGR,
                                         frame_id="camera_optical", ts=ts)

            small_right_rgb = None
            if sf.color_right_bgr is not None:
                small_right_bgr = cv2.resize(sf.color_right_bgr, (DW, DH))
                small_right_rgb = cv2.cvtColor(small_right_bgr, cv2.COLOR_BGR2RGB)

            # ---- Depth inference ----
            t_depth = time.perf_counter()
            try:
                depth_m, conf = depth_estimator.infer(small_rgb, small_right_rgb, fx=fx)
            except Exception as e:
                print(f"  frame {n}: depth ({depth_estimator.name}) failed: {e}")
                n += 1
                continue
            t_depth = time.perf_counter() - t_depth

            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_m = np.where((depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_FAR_M),
                               depth_m, 0.0).astype(np.float32)
            if depth_m.shape != (DH, DW):
                depth_m = cv2.resize(depth_m, (DW, DH), interpolation=cv2.INTER_NEAREST)
                conf = cv2.resize(conf, (DW, DH), interpolation=cv2.INTER_NEAREST)
            depth_m = filter_depth_edges(
                depth_m, args.depth_edge_threshold, args.depth_edge_dilate)

            # ---- Pose: ARKit -> OpenCV optical ----
            c2w_arkit = sf.c2w_arkit if sf.c2w_arkit is not None else np.eye(4)
            c2w = arkit_c2w_to_opencv(c2w_arkit)

            if (keyframe_recorder is not None
                    and n % max(1, args.save_keyframes_every_n) == 0):
                keyframe_recorder.record(
                    idx=n, rgb=small_rgb, pose_4x4=c2w, intrinsics=K, timestamp=ts,
                )

            # ---- Build per-frame colored cloud, push into VoxelMap ----
            depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                         frame_id="camera_optical", ts=ts)
            color_rgb_msg = Image.from_numpy(small_rgb, format=ImageFormat.RGB,
                                             frame_id="camera_optical", ts=ts)
            cam_pcd = PointCloud2.from_rgbd(
                color_image=color_rgb_msg, depth_image=depth_msg,
                camera_info=cam_info, depth_scale=1.0, depth_trunc=DEPTH_FAR_M,
            )
            cam_pts, cam_cols = cam_pcd.as_numpy()

            # Stride sub-sample for the per-frame world cloud (debug topic) and for fusion
            stride = max(1, args.points_stride)
            if stride > 1 and len(cam_pts) > 0:
                cam_pts_s = cam_pts[::stride]
                cam_cols_s = cam_cols[::stride] if cam_cols is not None else None
            else:
                cam_pts_s = cam_pts
                cam_cols_s = cam_cols

            if len(cam_pts_s) > 0:
                world_pts = transform_points(c2w, cam_pts_s)
                # Confidence-weighted insertion + (optionally) raycast clearing
                if args.use_depth_confidence:
                    # Recompute per-(strided)-pixel confidence from the depth map
                    conf_full = depth_to_confidence(depth_m).reshape(-1)
                    # cam_pts came from from_rgbd which only emits pixels with depth>0,
                    # so we don't have a direct pixel<->point mapping. Instead, derive
                    # confidence from the world-frame Z distance from the camera.
                    rel = world_pts - c2w[:3, 3].astype(np.float32)
                    dist = np.linalg.norm(rel, axis=1)
                    conf_flat = (1.0 / np.maximum(dist, 0.1) ** 2).astype(np.float32)
                else:
                    conf_flat = np.ones(len(world_pts), dtype=np.float32)
                voxel_map.insert(world_pts, confidences=conf_flat,
                                 max_drift=args.voxel_max_drift, colors=cam_cols_s)
                if args.raycast_every_n > 0 and (n % args.raycast_every_n == 0):
                    voxel_map.raycast_clear(origin=c2w[:3, 3], points=world_pts, config=raycast_cfg)
                voxel_map.prune(float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3]))

                import open3d as o3d
                pf = o3d.geometry.PointCloud()
                pf.points = o3d.utility.Vector3dVector(world_pts.astype(np.float64))
                if cam_cols_s is not None:
                    pf.colors = o3d.utility.Vector3dVector(cam_cols_s.astype(np.float64))
                points_msg = PointCloud2(pointcloud=pf, frame_id="world", ts=ts)
            else:
                import open3d as o3d
                points_msg = PointCloud2(pointcloud=o3d.geometry.PointCloud(),
                                         frame_id="world", ts=ts)

            # ---- 2D detection (YOLOE supplies track_id) ----
            dets2d = det2d.process_image_frame(color_msg)

            # ---- Lift to per-object 3D + push into ObjectDB ----
            new_objects: list[Any] = []
            run_objects = (args.objects_process_every_n > 0
                           and n % args.objects_process_every_n == 0)
            if run_objects and len(dets2d.detections) > 0:
                camera_tf = make_camera_to_world_transform(c2w, ts)
                try:
                    new_objects = Object.from_2d_to_list(
                        detections_2d=dets2d, color_image=color_rgb_msg, depth_image=depth_msg,
                        camera_info=cam_info, camera_transform=camera_tf,
                        depth_scale=1.0, depth_trunc=DEPTH_FAR_M,
                    )
                except Exception as e:
                    if n < 3:
                        print(f"  Object.from_2d_to_list failed: {e}")
                if new_objects:
                    object_db.add_objects(new_objects)

            # ---- Optional CLIP SpatialMemory ----
            if spatial_mem is not None:
                spatial_mem.maybe_store(small_bgr, c2w_arkit, ts)

            # ---- Publish ----
            img_topic.publish(color_msg)
            cam_info_topic.publish(cam_info)
            depth_cam_info_topic.publish(cam_info)
            ann_topic.publish(dets2d.to_foxglove_annotations())
            depth_topic.publish(depth_msg)
            points_topic.publish(points_msg)
            tf_topic.publish(make_tf_msg(c2w, ts))

            if n % args.map_publish_every_n == 0:
                pts, cols = voxel_map.to_points_colored(
                    min_observations=args.voxel_min_observations)
                if len(pts) > 0:
                    import open3d as o3d
                    mp = o3d.geometry.PointCloud()
                    mp.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    mp.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                    map_topic.publish(PointCloud2(pointcloud=mp, frame_id="world", ts=ts))

            if n % OBJECTS_PUBLISH_EVERY_N == 0:
                # Publish ALL tracked objects (pending + permanent) so boxes show
                # up immediately. Permanent are higher-confidence (matched across
                # >= OBJECTS_MIN_DETECTIONS frames) but pending are still useful
                # to render — we just won't trust them for downstream queries.
                all_objs = object_db.get_all_objects()
                if all_objs:
                    obj_cloud_topic.publish(aggregate_pointclouds(all_objs))
                    scene_topic.publish(build_scene_update_for_objects(all_objs, ts))

            n += 1
            if n == 1 or n % max(1, int(args.max_fps)) == 0:
                stats = object_db.get_stats()
                names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                d_pos = depth_m[depth_m > 0]
                drange = f"[{d_pos.min():.2f},{d_pos.max():.2f}]m" if d_pos.size else "[empty]"
                elapsed = time.perf_counter() - t_start
                print(f"  f{n} (src#{sf.frame_idx}): "
                      f"depth={t_depth*1000:.0f}ms total={elapsed*1000:.0f}ms "
                      f"d={drange} voxels={voxel_map.size} "
                      f"objs(perm/pend)={stats['permanent_count']}/{stats['pending_count']} "
                      f"2d={len(dets2d.detections)} [{names}]")

            if (map_writer.enabled and args.save_map_every_n > 0
                    and n > 0 and n % args.save_map_every_n == 0):
                map_writer.write(voxel_map, object_db, extra={"frames": n})

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        if map_writer.enabled:
            map_writer.write(voxel_map, object_db, extra={"frames": n})
        if keyframe_recorder is not None:
            keyframe_recorder.close()
        try:
            source.close()
        except Exception:
            pass
        try:
            det2d.detector.stop()
        except Exception:
            pass
        for t in (img_topic, ann_topic, cam_info_topic, depth_cam_info_topic,
                  depth_topic, points_topic, map_topic, obj_cloud_topic,
                  scene_topic, tf_topic):
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
