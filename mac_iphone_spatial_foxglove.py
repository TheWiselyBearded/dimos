"""iPhone (or any monocular video) -> depth -> 3D spatial memory + tracked objects.

Mirrors mac_viture_spatial_foxglove.py for plain phone video that has no pose track.
Pose options:
  - identity: camera assumed stationary; map fuses into a single viewpoint (debug)
  - vo: ORB feature + depth-PnP monocular visual odometry, so the map accumulates
        as the camera moves. Uses the per-frame depth map (which we compute anyway)
        to back-project keypoints, then solvePnPRansac against the next frame.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      /opt/anaconda3/envs/xr-nav/bin/python -u mac_iphone_spatial_foxglove.py \\
        --depth depthpro --pose vo

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
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_VIDEO = REPO / "datasets" / "iphone" / "phxLivingRoom.MOV"

sys.path.insert(0, str(REPO / "xr-nav" / "src"))
# DA3 source: macOS uses the awesome-depth-anything-3 fork;
# Linux/Windows use the official ByteDance Depth-Anything-3.
_da3_dir = "awesome-depth-anything-3" if sys.platform == "darwin" else "Depth-Anything-3"
sys.path.insert(0, str(REPO / "xr-nav" / _da3_dir / "src"))

# iPhone wide camera HFOV is roughly 60-65deg; user can override with --hfov-deg.
HFOV_DEG = 62.0
DISPLAY_W = 1024

DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0
DEPTH_EDGE_GRAD_THRESHOLD_M = 0.10
DEPTH_EDGE_DILATE_PX = 2

POINTS_STRIDE = 6

VOXEL_M = 0.05
VOXEL_MAX_RANGE_M = 8.0
VOXEL_INSERT_MAX_DRIFT_M = 0.04
VOXEL_RAYCAST_SUBSAMPLE = 8
VOXEL_RAYCAST_MAX_MISSES = 3
VOXEL_MIN_OBSERVATIONS = 2

MAP_PUBLISH_EVERY_N = 5

OBJECTS_DIST_THRESHOLD_M = 0.4
OBJECTS_MIN_DETECTIONS = 2
OBJECTS_PUBLISH_EVERY_N = 1
OBJECTS_PROCESS_EVERY_N = 1
# Keep only the nearest depth band (m) within each segmentation mask so mask spill
# onto far background doesn't leak into the object cloud and inflate its box.
OBJECTS_FG_BAND_M = 0.8
# Scene entity lifetime (s): finite so decayed/removed objects expire from Foxglove
# instead of lingering forever (sec=0 = permanent), but > the publish interval.
OBJECTS_SCENE_TTL_S = 3.0

SPATIAL_MIN_DISTANCE_M = 0.10
SPATIAL_MIN_INTERVAL_S = 1.0

# Raw accumulated-cloud defaults. The voxel map (/map) carves + can be pruned;
# this is a separate "retain everything ever seen" cloud (/accumulated_cloud)
# deduped only enough to bound memory + keep the published message transmittable.
ACCUMULATE_CLOUD_VOXEL_M = 0.02
ACCUMULATE_CLOUD_MAX_POINTS = 1_200_000

# Optional sink for the live object list. When set to a callable
# ``sink(objects: list[dict], ts: float)`` the main loop pushes the tracked
# objects (name, id, distance-from-camera, detection count, world position)
# each publish cycle. Left None for standalone runs (no behaviour change); the
# ReachyBrain server sets it to forward the list to the robot's web app.
OBJECT_LIST_SINK = None


# ===========================================================================
#  Pose / TF helpers
# ===========================================================================

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
    from dimos.msgs.geometry_msgs import Transform, Vector3, Quaternion
    t, q = c2w_to_translation_quat(c2w_opencv)
    return Transform(
        translation=Vector3(float(t[0]), float(t[1]), float(t[2])),
        rotation=Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        frame_id="world",
        child_frame_id="camera_optical",
        ts=ts,
    )


def make_camera_info(width: int, height: int, hfov_deg: float):
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
    conf = np.where(depth_m > 0, 1.0 / np.maximum(depth_m, 0.1) ** 2, 0.0)
    return conf.astype(np.float32)


# ===========================================================================
#  Frame source: any video file (no pose)
# ===========================================================================

@dataclass
class SourceFrame:
    color_bgr: np.ndarray
    ts: float
    frame_idx: int
    # Optional external camera-to-world (OpenCV optical: X right, Y down, Z fwd).
    # When set and --pose external is selected, the main loop uses this instead
    # of running monocular VO. Replay scripts that have a recorded pose stream
    # (e.g. Reachy head_pose.jsonl) can attach it here.
    c2w: Optional[np.ndarray] = None


class VideoSource:
    def __init__(self, video_path: Path, fps_cap: float = 10.0, loop: bool = True):
        if not video_path.exists():
            sys.exit(f"missing video: {video_path}")
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            sys.exit(f"cv2 could not open {video_path}")
        self._loop = loop
        self._W = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._H = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Decimate frames so depth model isn't overrun. e.g. native 30fps + cap 10 -> step 3
        self._step = max(1, int(round(self._native_fps / max(fps_cap, 0.1))))
        print(f"[source:video] {video_path.name} {self._W}x{self._H} "
              f"native={self._native_fps:.1f}fps total={self._total} "
              f"step={self._step} (effective {self._native_fps/self._step:.1f}fps)")

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._W, self._H

    def __iter__(self):
        loops = 0
        idx = 0
        while True:
            # Skip step-1 frames cheaply
            for _ in range(self._step - 1):
                self._cap.grab()
            ok, frame_bgr = self._cap.read()
            if not ok:
                if not self._loop:
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loops += 1
                print(f"  --- loop {loops} ---")
                continue

            if frame_bgr.ndim == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            ts = time.time()
            yield SourceFrame(color_bgr=frame_bgr, ts=ts, frame_idx=idx)
            idx += 1

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


# ===========================================================================
#  Monocular visual odometry: ORB features + depth back-projection + PnP
# ===========================================================================

class MonocularDepthVO:
    """Pose tracker for monocular RGB+D streams.

    Each update:
      1. Detect ORB on the new gray frame.
      2. Match against previous descriptors (BFMatcher, Hamming, crossCheck).
      3. Back-project the *previous* matched keypoints to 3D using the previous
         depth map -> 3D-2D correspondences (prev_3D, cur_2D).
      4. solvePnPRansac yields the extrinsic mapping prev_cam frame -> cur_cam frame
         (i.e., R, t such that p_cur = R @ p_prev + t).
      5. The cur->prev relative pose is its inverse; world c2w accumulates as
         self._c2w := self._c2w @ T_prev_to_cur, where T_prev_to_cur places the
         current camera in the previous camera's frame.

    Output is a 4x4 c2w in the OpenCV optical convention (X right, Y down, Z fwd),
    anchored at identity at the first usable frame.
    """

    def __init__(self, K: np.ndarray, n_features: int = 1500,
                 ransac_reproj_px: float = 3.0, min_inliers: int = 20):
        self._K = K.astype(np.float64)
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ransac_reproj_px = ransac_reproj_px
        self._min_inliers = min_inliers
        self._prev_kp: list | None = None
        self._prev_desc: np.ndarray | None = None
        self._prev_depth: np.ndarray | None = None
        self._c2w = np.eye(4, dtype=np.float64)
        self._fail_streak = 0

    @property
    def c2w(self) -> np.ndarray:
        return self._c2w.copy()

    def update(self, gray: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 8:
            self._stash(kp, desc, depth_m)
            return self._c2w.copy()

        if self._prev_desc is None:
            self._stash(kp, desc, depth_m)
            return self._c2w.copy()

        matches = self._matcher.match(self._prev_desc, desc)
        if len(matches) < 12:
            self._stash(kp, desc, depth_m)
            return self._c2w.copy()

        H, W = self._prev_depth.shape
        fx = self._K[0, 0]; fy = self._K[1, 1]
        cx = self._K[0, 2]; cy = self._K[1, 2]

        prev3d = []
        cur2d = []
        for m in matches:
            u_p, v_p = self._prev_kp[m.queryIdx].pt
            iu, iv = int(round(u_p)), int(round(v_p))
            if iu < 0 or iv < 0 or iu >= W or iv >= H:
                continue
            d = float(self._prev_depth[iv, iu])
            if d <= 0 or not np.isfinite(d):
                continue
            X = (u_p - cx) * d / fx
            Y = (v_p - cy) * d / fy
            prev3d.append((X, Y, d))
            cur2d.append(self._prev_kp[m.queryIdx].pt)  # placeholder, overwritten below
            cur2d[-1] = kp[m.trainIdx].pt

        if len(prev3d) < self._min_inliers:
            self._stash(kp, desc, depth_m)
            self._fail_streak += 1
            return self._c2w.copy()

        prev3d_np = np.asarray(prev3d, dtype=np.float64)
        cur2d_np = np.asarray(cur2d, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            prev3d_np, cur2d_np, self._K, None,
            iterationsCount=150, reprojectionError=self._ransac_reproj_px,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < self._min_inliers:
            self._stash(kp, desc, depth_m)
            self._fail_streak += 1
            return self._c2w.copy()

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        # solvePnPRansac extrinsic: p_cur = R @ p_prev + t. We need the cur camera
        # pose expressed in the prev camera frame: T_prev_cur with columns
        # (R^T, -R^T t).
        T_prev_cur = np.eye(4, dtype=np.float64)
        T_prev_cur[:3, :3] = R.T
        T_prev_cur[:3, 3] = -R.T @ t

        # Sanity: reject implausibly large jumps (depth-PnP occasionally explodes)
        step = float(np.linalg.norm(T_prev_cur[:3, 3]))
        if step > 1.5:  # >1.5m in one decimated step is almost certainly a glitch
            self._stash(kp, desc, depth_m)
            self._fail_streak += 1
            return self._c2w.copy()

        self._c2w = self._c2w @ T_prev_cur
        self._fail_streak = 0
        self._stash(kp, desc, depth_m)
        return self._c2w.copy()

    def _stash(self, kp, desc, depth_m) -> None:
        self._prev_kp = kp
        self._prev_desc = desc
        self._prev_depth = depth_m


# ===========================================================================
#  Depth estimators
# ===========================================================================

class DepthEstimator(ABC):
    name: str = "base"
    # If True, the main loop will skip the depth-edge filter because per-pixel
    # gradients reflect amplified noise rather than real surface discontinuities.
    skip_edge_filter: bool = False

    @abstractmethod
    def infer(self, color_rgb: np.ndarray, fx: float) -> tuple[np.ndarray, np.ndarray]: ...


class DepthProEstimator(DepthEstimator):
    name = "depthpro"

    def __init__(self, device: str = "mps"):
        try:
            import depth_pro
        except ImportError:
            sys.exit(
                "depth_pro not installed. Install with:\n"
                "  /opt/anaconda3/envs/xr-nav/bin/pip install "
                "git+https://github.com/apple/ml-depth-pro.git"
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

    def infer(self, color_rgb, fx):
        with self._torch.no_grad():
            inp = self._transform(color_rgb).to(self._device)
            f_px = self._torch.tensor(float(fx), dtype=self._torch.float32, device=self._device)
            pred = self._model.infer(inp, f_px=f_px)
            depth_m = pred["depth"].detach().cpu().numpy().astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m[0]
        conf = np.ones_like(depth_m, dtype=np.float32)
        # Drop GPU tensors before clearing the cache, otherwise the freed memory
        # is still pinned by the live references and the next frame compounds.
        del inp, f_px, pred
        try:
            if self._device.type == "mps":
                self._torch.mps.empty_cache()
            elif self._device.type == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:
            pass
        return depth_m, conf


class DA3Estimator(DepthEstimator):
    name = "da3"

    def __init__(self, model_name: str = "da3metric-large", device: str = "mps",
                 process_res: int = 504, conf_threshold: float = 0.0,
                 force_relative: bool = False):
        from depth_anything_3.api import DepthAnything3
        print(f"[depth] loading {model_name} on {device}...")
        # The DepthAnything3 constructor only builds the architecture; pretrained
        # weights load via from_pretrained (PyTorchModelHubMixin). A bare constructor
        # leaves the net uninitialized -- its near-constant output gets stretched into
        # blocky garbage -- so on every platform (the macOS fork included) load weights
        # via from_pretrained, then move to device.
        self._model = DepthAnything3.from_pretrained(
            f"depth-anything/{model_name.upper()}").to(device)
        # Prediction.is_metric is unreliable (returns {} -> falsy); trust the model name.
        self._metric_model = "metric" in model_name.lower()
        self._res = process_res
        self._conf_thresh = conf_threshold
        self._force_relative = force_relative
        self._scale: float | None = None
        self._conf_logged = False
        self._raw_logged = False

    def infer(self, color_rgb, fx):
        pred = self._model.inference(image=[color_rgb], process_res=self._res)
        raw = np.nan_to_num(pred.depth[0].astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        is_metric = (self._metric_model or bool(getattr(pred, "is_metric", 0))) and not self._force_relative
        if not self._raw_logged:
            r_pos = raw[raw > 0]
            r_range_val = float(r_pos.max() - r_pos.min()) if r_pos.size else 0.0
            r_range = (f"[{r_pos.min():.3f},{r_pos.max():.3f}] med={float(np.median(r_pos)):.3f}"
                       if r_pos.size else "[empty]")
            print(f"[depth] DA3 raw {r_range} is_metric={is_metric} "
                  f"force_relative={self._force_relative}")
            # In relative mode, narrow raw range means we'll be stretching tiny
            # signal + noise across [0.2, 6.0]m. The edge filter sees that noise
            # as gradients and wipes the map. Auto-disable it and warn loudly.
            if not is_metric and r_range_val < 0.2:
                self.skip_edge_filter = True
                print(f"[depth] WARNING: DA3 raw dynamic range is only {r_range_val:.3f}. "
                      f"Auto-disabling depth-edge filter (output will be noisy). "
                      f"For sharp depth on this content, use --depth depthpro.")
            self._raw_logged = True
        if is_metric:
            depth_m = raw
        else:
            valid = raw > 0
            depth_norm = np.zeros_like(raw, dtype=np.float32)
            if valid.any():
                vals = raw[valid]
                rmin, rmax = float(vals.min()), float(vals.max())
                if rmax - rmin < 1e-8:
                    depth_norm[valid] = 0.5 * (DEPTH_NEAR_M + DEPTH_FAR_M)
                else:
                    d_norm = (vals - rmin) / (rmax - rmin)
                    depth_norm[valid] = (DEPTH_NEAR_M + d_norm * (DEPTH_FAR_M - DEPTH_NEAR_M)).astype(np.float32)
            if self._scale is None and valid.any():
                med = float(np.median(depth_norm[valid]))
                self._scale = 1.5 / med if med > 1e-6 else 1.0
                print(f"[depth] DA3 first-frame scale fit: {self._scale:.3f}")
            scale = self._scale if self._scale is not None else 1.0
            depth_m = (depth_norm * scale).astype(np.float32)

        conf_map = np.ones_like(depth_m, dtype=np.float32)
        if pred.conf is not None:
            c = pred.conf[0].astype(np.float32)
            cmax = float(c.max()) if c.size else 1.0
            cmin = float(c.min()) if c.size else 0.0
            if cmax > 1.0:
                c = c / cmax
            conf_map = c
            if not self._conf_logged:
                kept = float((c >= self._conf_thresh).mean())
                print(f"[depth] DA3 conf range=[{cmin:.3f},{cmax:.3f}] "
                      f"threshold={self._conf_thresh:.2f} kept_frac={kept:.3f}")
                self._conf_logged = True
            if self._conf_thresh > 0.0:
                depth_m = np.where(c >= self._conf_thresh, depth_m, 0.0).astype(np.float32)
        return depth_m, conf_map


def make_depth_estimator(kind: str, device: str = "mps",
                         da3_conf_threshold: float = 0.0,
                         da3_force_relative: bool = False,
                         da3_model: str = "da3metric-large",
                         da3_process_res: int = 504) -> DepthEstimator:
    if kind == "depthpro":
        return DepthProEstimator(device=device)
    if kind == "da3":
        return DA3Estimator(model_name=da3_model, device=device,
                            process_res=da3_process_res,
                            conf_threshold=da3_conf_threshold,
                            force_relative=da3_force_relative)
    raise ValueError(f"unknown depth kind: {kind}")


# ===========================================================================
#  YOLOE detector (auto-downloaded weights, mirrors viture script)
# ===========================================================================

YOLOE_WEIGHTS_DIR = REPO / "checkpoints"


class LocalYoloeDetector:
    def __init__(self, device: str = "mps", weights_name: str = "yoloe-11s-seg-pf.pt",
                 max_area_ratio: float | None = 0.3):
        from ultralytics import YOLOE
        import threading
        YOLOE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = YOLOE_WEIGHTS_DIR / weights_name
        if not weights_path.exists():
            print(f"[detector] downloading YOLOE weights -> {weights_path}")
        self.model = YOLOE(str(weights_path))
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
#  SceneUpdate builder for tracked objects
# ===========================================================================

def _stable_color_from_id(object_id: str, alpha: float = 0.25):
    try:
        seed = int(object_id, 16)
    except (ValueError, TypeError):
        seed = abs(hash(object_id))
    rng = np.random.default_rng(seed % (2**32 - 1))
    r, g, b = rng.uniform(0.3, 1.0, size=3)
    return float(r), float(g), float(b), alpha


def build_scene_update_for_objects(objects: list[Any], ts: float):
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
        # Robust box: a raw min/max AABB is dominated by a handful of smeared/leaked
        # points (depth noise along the view ray), which balloons the box toward the
        # camera. Use a 2-98th percentile extent so a few strays don't inflate it.
        try:
            pts, _ = obj.pointcloud.as_numpy()
        except Exception:
            continue
        if pts is None or len(pts) < 10:
            continue
        lo = np.percentile(pts, 2.0, axis=0)
        hi = np.percentile(pts, 98.0, axis=0)
        center = (lo + hi) / 2.0
        extent = np.maximum(hi - lo, 1e-3)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
            continue

        cube = CubePrimitive()
        cube.pose = Pose(); cube.pose.position = Point()
        cube.pose.position.x = float(center[0])
        cube.pose.position.y = float(center[1])
        cube.pose.position.z = float(center[2])
        cube.pose.orientation = LCMQ()
        cube.pose.orientation.x = 0.0; cube.pose.orientation.y = 0.0
        cube.pose.orientation.z = 0.0; cube.pose.orientation.w = 1.0
        cube.size = LCMV3()
        cube.size.x = float(extent[0]); cube.size.y = float(extent[1]); cube.size.z = float(extent[2])
        r, g, b, a = _stable_color_from_id(obj.object_id)
        cc = Color(); cc.r = r; cc.g = g; cc.b = b; cc.a = a
        cube.color = cc

        text = TextPrimitive()
        text.pose = Pose(); text.pose.position = Point()
        text.pose.position.x = float(center[0])
        text.pose.position.y = float(center[1])
        text.pose.position.z = float(center[2]) + float(extent[2]) / 2.0 + 0.1
        text.pose.orientation = LCMQ()
        text.pose.orientation.x = 0.0; text.pose.orientation.y = 0.0
        text.pose.orientation.z = 0.0; text.pose.orientation.w = 1.0
        text.billboard = True; text.font_size = 18.0; text.scale_invariant = True
        tc = Color(); tc.r = 1.0; tc.g = 1.0; tc.b = 1.0; tc.a = 1.0
        text.color = tc
        text.text = f"{obj.name} #{obj.object_id} ({obj.detections_count})"

        entity = SceneEntity()
        entity.timestamp = to_ros_stamp(ts)
        entity.frame_id = "world"; entity.id = obj.object_id
        entity.lifetime = Duration()
        entity.lifetime.sec = int(OBJECTS_SCENE_TTL_S)
        entity.lifetime.nanosec = int((OBJECTS_SCENE_TTL_S % 1.0) * 1e9)
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
#  Optional CLIP SpatialMemory adapter
# ===========================================================================

class SpatialMemoryAdapter:
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
#  Raw accumulated cloud (retain-everything scan)
# ===========================================================================

class AccumulatedCloud:
    """Persistent world-frame point cloud that only grows.

    Unlike the VoxelMap (which raycast-carves free space and can be distance-
    pruned), this keeps every region the camera has ever seen — including
    surfaces glimpsed only once. To bound memory and keep the published
    PointCloud2 transmittable it voxel-downsamples on each consolidate and
    random-subsamples as a hard backstop.
    """

    def __init__(self, voxel: float = ACCUMULATE_CLOUD_VOXEL_M,
                 max_points: int = ACCUMULATE_CLOUD_MAX_POINTS) -> None:
        self._pts = np.empty((0, 3), np.float32)
        self._cols = np.empty((0, 3), np.float32)
        self._pending_pts: list[np.ndarray] = []
        self._pending_cols: list[np.ndarray] = []
        self._voxel = float(voxel)
        self._max_points = int(max_points)

    def add(self, world_pts: np.ndarray, colors: np.ndarray | None = None) -> None:
        if world_pts is None or len(world_pts) == 0:
            return
        wp = np.asarray(world_pts, np.float32)
        if colors is not None and len(colors) == len(wp):
            cc = np.asarray(colors, np.float32)
        else:
            cc = np.full((len(wp), 3), 0.6, np.float32)
        self._pending_pts.append(wp)
        self._pending_cols.append(cc)

    def _consolidate(self) -> None:
        if not self._pending_pts:
            return
        pts = np.vstack([self._pts, *self._pending_pts])
        cols = np.vstack([self._cols, *self._pending_cols])
        self._pending_pts.clear()
        self._pending_cols.clear()
        if self._voxel > 0 and len(pts):
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            pcd = pcd.voxel_down_sample(self._voxel)
            pts = np.asarray(pcd.points, np.float32)
            cols = np.asarray(pcd.colors, np.float32)
        if len(pts) > self._max_points:
            idx = np.random.choice(len(pts), self._max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
        self._pts, self._cols = pts, cols

    @property
    def size(self) -> int:
        return len(self._pts) + sum(len(p) for p in self._pending_pts)

    def to_msg(self, ts: float):
        self._consolidate()
        import open3d as o3d
        from dimos.msgs.sensor_msgs import PointCloud2
        pcd = o3d.geometry.PointCloud()
        if len(self._pts):
            pcd.points = o3d.utility.Vector3dVector(self._pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(self._cols.astype(np.float64))
        return PointCloud2(pointcloud=pcd, frame_id="world", ts=ts)


# ===========================================================================
#  Main loop
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help="Path to the .mp4/.MOV/etc. to play back")
    parser.add_argument("--depth", choices=["depthpro", "da3"], default="depthpro")
    parser.add_argument("--da3-model", default="da3metric-large",
                        choices=["da3-small", "da3-base", "da3-large",
                                 "da3-giant", "da3metric-large",
                                 "da3nested-giant-large"],
                        help="DA3 variant. da3metric-large (default) returns true metric "
                             "depth so per-frame scale is consistent across frames. "
                             "da3-small/base/large are scale-ambiguous and require the "
                             "first-frame scale fit (drifts on long sessions). "
                             "da3nested-giant-large is highest quality but slowest.")
    parser.add_argument("--pose", choices=["identity", "vo", "external"], default="vo",
                        help="identity = camera fixed at origin (debug). "
                             "vo = ORB+depth-PnP visual odometry (default). "
                             "external = read SourceFrame.c2w supplied by the frame source "
                             "(e.g. Reachy head_pose.jsonl via the replay script). Falls back "
                             "to identity for any frame missing c2w.")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--enable-clip-memory", action="store_true",
                        help="record CLIP-embedded frames to ChromaDB for text/image queries")
    parser.add_argument("--clip-db", type=Path,
                        default=Path.home() / ".dimos" / "spatial_memory" / "iphone")
    parser.add_argument("--display-width", type=int, default=DISPLAY_W,
                        help="resample input to this width before depth inference")
    parser.add_argument("--hfov-deg", type=float, default=HFOV_DEG,
                        help="horizontal FOV (deg) of the source camera. "
                             "iPhone wide ~62, ultrawide ~106, 2x telephoto ~30")
    parser.add_argument("--max-fps", type=float, default=10.0,
                        help="upper-bound publish rate; depth model usually sets the real ceiling")
    parser.add_argument("--no-loop", action="store_true",
                        help="exit after the video ends instead of looping")
    parser.add_argument("--no-detect", action="store_true",
                        help="disable YOLOE 2D detection + ObjectDB tracking (on by default)")

    # xr_nav path must be on sys.path before importing cli_args (matches lines above).
    from xr_nav.cli_args import add_map_io_args, add_keyframe_args
    add_map_io_args(parser)
    add_keyframe_args(parser)

    perf = parser.add_argument_group("latency tunables")
    perf.add_argument("--objects-process-every-n", type=int, default=OBJECTS_PROCESS_EVERY_N)
    perf.add_argument("--map-publish-every-n", type=int, default=MAP_PUBLISH_EVERY_N)
    perf.add_argument("--raycast-every-n", type=int, default=1)
    perf.add_argument("--points-stride", type=int, default=POINTS_STRIDE)

    nz = parser.add_argument_group("noise tunables")
    nz.add_argument("--depth-edge-threshold", type=float, default=DEPTH_EDGE_GRAD_THRESHOLD_M)
    nz.add_argument("--depth-edge-dilate", type=int, default=DEPTH_EDGE_DILATE_PX)
    nz.add_argument("--voxel-size", type=float, default=VOXEL_M)
    nz.add_argument("--voxel-min-observations", type=int, default=VOXEL_MIN_OBSERVATIONS)
    nz.add_argument("--voxel-max-drift", type=float, default=VOXEL_INSERT_MAX_DRIFT_M)
    nz.add_argument("--no-prune", action="store_true",
                    help="Don't distance-prune the voxel map each frame. For a "
                         "fixed-base scanner (e.g. Reachy panning its head) the map "
                         "should accumulate; pruning relative to a drifting VO position "
                         "erases good geometry. Combine with --raycast-every-n 0 for a "
                         "pure accumulate-only map.")
    nz.add_argument("--use-depth-confidence", action="store_true")
    nz.add_argument("--accumulate-cloud", action="store_true",
                    help="Publish a separate /accumulated_cloud that retains every "
                         "region ever seen (not carved/pruned like /map). Deduped to "
                         "--accumulate-cloud-voxel and capped at "
                         "--accumulate-cloud-max-points to bound memory + transport.")
    nz.add_argument("--accumulate-cloud-voxel", type=float, default=ACCUMULATE_CLOUD_VOXEL_M,
                    help="Voxel size (m) the accumulated cloud is deduped to on each "
                         "publish. Smaller = denser scan but larger messages.")
    nz.add_argument("--accumulate-cloud-max-points", type=int, default=ACCUMULATE_CLOUD_MAX_POINTS,
                    help="Hard cap on accumulated-cloud point count (random-subsampled "
                         "above this) to keep the published PointCloud2 transmittable.")
    nz.add_argument("--da3-conf-threshold", type=float, default=None,
                    help="DA3-only: zero out pixels with conf < this. "
                         "Default 0.0 for relative variants (use all DA3 output); "
                         "default 0.5 for da3metric-* variants where the conf channel is calibrated. "
                         "Raise to filter low-confidence regions; check the "
                         "'[depth] DA3 conf range=...' print to pick a value")
    nz.add_argument("--da3-process-res", type=int, default=None,
                    help="DA3-only: input resolution fed to the model. "
                         "Default 504 for small/base/large; 700 for da3metric-* and da3nested-* "
                         "(large variants benefit from higher input res on Reachy-sized scenes)")
    nz.add_argument("--da3-trust-is-metric", action="store_true",
                    help="DA3-only: trust the model's is_metric flag and use raw output when it's set. "
                         "By default we ignore the flag and always run normalize+scale-fit, which is "
                         "more robust on general phone video where is_metric can be unreliable")

    odb = parser.add_argument_group("object tracking (ObjectDB)")
    odb.add_argument("--objects-distance-threshold", type=float, default=OBJECTS_DIST_THRESHOLD_M,
                     help="3D center distance (m) below which two detections merge")
    odb.add_argument("--objects-class-aware", action="store_true",
                     help="require matching class name to merge. Default OFF: YOLOE prompt-free "
                          "emits flickering vocab names for the same object, so name-gating blocks "
                          "every merge and each frame spawns a new '(1)' box. Default merges on "
                          "spatial proximity alone")
    odb.add_argument("--objects-pixel-threshold", type=float, default=60.0,
                     help="2D-pixel fallback radius. After 3D match fails, project the existing "
                          "object's center into the current image; merge if a same-class new "
                          "detection's bbox center lands within this many pixels. Recovers "
                          "matches under VO drift")
    odb.add_argument("--objects-disable-decay", action="store_true",
                     help="don't decay/delete objects that go unobserved while in-frustum")
    odb.add_argument("--objects-confidence-init", type=float, default=0.5)
    odb.add_argument("--objects-confidence-up", type=float, default=0.10,
                     help="confidence bump per matched detection")
    odb.add_argument("--objects-confidence-down", type=float, default=0.05,
                     help="confidence decay per frame the object is in-frustum but undetected. "
                          "Object is deleted when confidence reaches 0")

    args = parser.parse_args()

    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image, ImageFormat, PointCloud2, CameraInfo
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from xr_nav.voxel_map import VoxelMap, RaycastConfig
    from xr_nav.map_io import MapArtifactWriter, NullObjectDB, load_map_bundle
    from xr_nav.keyframe_recorder import KeyframeRecorder
    from xr_nav.cli_args import resolve_cloud_min_observations

    source = VideoSource(video_path=args.video, fps_cap=args.max_fps,
                         loop=not args.no_loop)

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

    # da3metric-* and da3nested-*-large variants return true metric depth; trust
    # their is_metric flag automatically. Other DA3 variants need the explicit
    # --da3-trust-is-metric opt-in (default off, since is_metric is unreliable
    # on those).
    _model_is_metric = "metric" in args.da3_model
    _force_relative = not (args.da3_trust_is_metric or _model_is_metric)
    # Conf threshold default: metric models have a calibrated conf channel that
    # is worth filtering on; relative models don't, so leave it open.
    if args.da3_conf_threshold is None:
        _da3_conf_threshold = 0.5 if _model_is_metric else 0.0
    else:
        _da3_conf_threshold = args.da3_conf_threshold
    # Process-res default: large variants are visibly noisier at 504 on
    # Reachy-sized scenes; bump to 700 for metric/nested-large.
    if args.da3_process_res is None:
        _da3_process_res = 700 if (_model_is_metric or "nested" in args.da3_model) else 504
    else:
        _da3_process_res = args.da3_process_res
    depth_estimator = make_depth_estimator(args.depth, device=args.device,
                                           da3_conf_threshold=_da3_conf_threshold,
                                           da3_force_relative=_force_relative,
                                           da3_model=args.da3_model,
                                           da3_process_res=_da3_process_res)

    det2d = None
    object_db = None
    if not args.no_detect:
        from dimos.perception.detection.module2D import Detection2DModule
        from dimos.perception.detection.objectDB import ObjectDB
        print("[main] warming YOLOE detector (may download weights on first run)...")
        det2d = Detection2DModule(detector=lambda: LocalYoloeDetector(device=args.device))
        object_db = ObjectDB(
            distance_threshold=args.objects_distance_threshold,
            min_detections_for_permanent=OBJECTS_MIN_DETECTIONS,
            class_aware_matching=args.objects_class_aware,
            frustum_match_pixel_threshold=args.objects_pixel_threshold,
            enable_decay=not args.objects_disable_decay,
            confidence_init=args.objects_confidence_init,
            confidence_step_up=args.objects_confidence_up,
            confidence_step_down=args.objects_confidence_down,
        )
        print(f"\n[objdb] === MERGE FIXES ACTIVE === class_aware={args.objects_class_aware} "
              f"dist_thresh={args.objects_distance_threshold}m px_thresh={args.objects_pixel_threshold} "
              f"fg_band={OBJECTS_FG_BAND_M}m scene_ttl={OBJECTS_SCENE_TTL_S}s\n"
              f"[objdb] if you DON'T see this line on startup, you're running stale code "
              f"— kill the process and relaunch\n")

    voxel_map = VoxelMap(voxel_size=args.voxel_size, max_range=VOXEL_MAX_RANGE_M)
    raycast_cfg = RaycastConfig(
        subsample=VOXEL_RAYCAST_SUBSAMPLE, max_misses=VOXEL_RAYCAST_MAX_MISSES)
    acc_cloud = (AccumulatedCloud(voxel=args.accumulate_cloud_voxel,
                                  max_points=args.accumulate_cloud_max_points)
                 if args.accumulate_cloud else None)

    if args.load_map is not None:
        bundle = load_map_bundle(args.load_map)
        voxel_map.load_state(bundle["voxel_map"])
        if object_db is not None:
            object_db.load_state(bundle["object_db"])
        elif bundle["object_db"]["permanent"] or bundle["object_db"]["pending"]:
            print("[load] WARNING: bundle contains objects but --no-detect was passed; "
                  "objects will not be loaded")

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

    spatial_mem = SpatialMemoryAdapter(db_path=args.clip_db) if args.enable_clip_memory else None

    vo = MonocularDepthVO(K=K) if args.pose == "vo" else None
    _external_pose_warned = False
    _last_external_c2w: np.ndarray | None = None

    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)
    depth_cam_info_topic = LCMTransport("/depth_camera_info", CameraInfo)
    depth_topic = LCMTransport("/depth", Image)
    points_topic = LCMTransport("/points_frame", PointCloud2)
    map_topic = LCMTransport("/map", PointCloud2)
    acc_cloud_topic = LCMTransport("/accumulated_cloud", PointCloud2) if args.accumulate_cloud else None
    obj_cloud_topic = LCMTransport("/object_clouds", PointCloud2)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    print("\n[main] publishing. Foxglove: ws://localhost:8765")
    print("  3D panel (frame=world): /map  /object_clouds  /scene_update  /tf  /points_frame")
    if acc_cloud_topic is not None:
        print("  accumulate-cloud ON: /accumulated_cloud "
              f"(voxel={args.accumulate_cloud_voxel}m cap={args.accumulate_cloud_max_points})")
    if args.no_prune:
        print("  no-prune ON: voxel map accumulates (not distance-pruned)")
    print("  Image panels: /color_image  /depth")
    if args.pose == "identity":
        print("  pose=identity — every frame fuses at the origin; map will collapse.")
    elif args.pose == "external":
        print("  pose=external — using SourceFrame.c2w supplied by the frame source.")
    print("Ctrl+C to exit.\n")

    n = 0
    period = 1.0 / max(args.max_fps, 0.1)

    def frame_stream():
        yield probe
        yield from iterator

    try:
        for sf in frame_stream():
            t_start = time.perf_counter()
            ts = sf.ts

            small_bgr = cv2.resize(sf.color_bgr, (DW, DH))
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            color_msg = Image.from_numpy(small_bgr, format=ImageFormat.BGR,
                                         frame_id="camera_optical", ts=ts)

            t_depth = time.perf_counter()
            try:
                depth_m, conf = depth_estimator.infer(small_rgb, fx=fx)
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
            edge_thresh = 0.0 if depth_estimator.skip_edge_filter else args.depth_edge_threshold
            depth_m = filter_depth_edges(
                depth_m, edge_thresh, args.depth_edge_dilate)

            t_pose = time.perf_counter()
            if vo is not None:
                gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
                c2w = vo.update(gray, depth_m)
            elif args.pose == "external":
                ext = getattr(sf, "c2w", None)
                if ext is not None:
                    c2w = np.asarray(ext, dtype=np.float64)
                    _last_external_c2w = c2w
                elif _last_external_c2w is not None:
                    # Brief pose dropout — hold the last known pose instead of
                    # snapping to the origin, which would dump this frame's points
                    # into the map at the wrong place and smear it.
                    c2w = _last_external_c2w
                else:
                    if not _external_pose_warned:
                        print("[main] --pose external but no pose received yet; "
                              "using identity until the first pose arrives (warn once)")
                        _external_pose_warned = True
                    c2w = np.eye(4, dtype=np.float64)
            else:
                c2w = np.eye(4, dtype=np.float64)
            t_pose = time.perf_counter() - t_pose

            if (keyframe_recorder is not None
                    and n % max(1, args.save_keyframes_every_n) == 0):
                keyframe_recorder.record(
                    idx=n, rgb=small_rgb, pose_4x4=c2w, intrinsics=K, timestamp=ts,
                )

            depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                         frame_id="camera_optical", ts=ts)
            color_rgb_msg = Image.from_numpy(small_rgb, format=ImageFormat.RGB,
                                             frame_id="camera_optical", ts=ts)
            cam_pcd = PointCloud2.from_rgbd(
                color_image=color_rgb_msg, depth_image=depth_msg,
                camera_info=cam_info, depth_scale=1.0, depth_trunc=DEPTH_FAR_M,
            )
            cam_pts, cam_cols = cam_pcd.as_numpy()

            stride = max(1, args.points_stride)
            if stride > 1 and len(cam_pts) > 0:
                cam_pts_s = cam_pts[::stride]
                cam_cols_s = cam_cols[::stride] if cam_cols is not None else None
            else:
                cam_pts_s = cam_pts
                cam_cols_s = cam_cols

            if len(cam_pts_s) > 0:
                world_pts = transform_points(c2w, cam_pts_s)
                if args.use_depth_confidence:
                    rel = world_pts - c2w[:3, 3].astype(np.float32)
                    dist = np.linalg.norm(rel, axis=1)
                    conf_flat = (1.0 / np.maximum(dist, 0.1) ** 2).astype(np.float32)
                else:
                    conf_flat = np.ones(len(world_pts), dtype=np.float32)
                voxel_map.insert(world_pts, confidences=conf_flat,
                                 max_drift=args.voxel_max_drift, colors=cam_cols_s)
                if args.raycast_every_n > 0 and (n % args.raycast_every_n == 0):
                    voxel_map.raycast_clear(origin=c2w[:3, 3], points=world_pts, config=raycast_cfg)
                if not args.no_prune:
                    voxel_map.prune(float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3]))
                if acc_cloud is not None:
                    acc_cloud.add(world_pts, cam_cols_s)

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

            dets2d = None
            new_objects: list[Any] = []
            decay_deleted: list[str] = []
            if det2d is not None:
                dets2d = det2d.process_image_frame(color_msg)
                run_objects = (args.objects_process_every_n > 0
                               and n % args.objects_process_every_n == 0)
                if run_objects:
                    if len(dets2d.detections) > 0:
                        camera_tf = make_camera_to_world_transform(c2w, ts)
                        try:
                            from dimos.perception.detection.type.detection3d.object import Object
                            new_objects = Object.from_2d_to_list(
                                detections_2d=dets2d, color_image=color_rgb_msg, depth_image=depth_msg,
                                camera_info=cam_info, camera_transform=camera_tf,
                                depth_scale=1.0, depth_trunc=DEPTH_FAR_M,
                                depth_fg_band_m=OBJECTS_FG_BAND_M,
                            )
                        except Exception as e:
                            if n < 3:
                                print(f"  Object.from_2d_to_list failed: {e}")

                    K_arr = np.array([[fx, 0.0, cam_info.K[2]],
                                      [0.0, fx, cam_info.K[5]],
                                      [0.0, 0.0, 1.0]], dtype=np.float64)
                    observed_ids: set[str] = set()
                    if new_objects:
                        returned = object_db.add_objects(
                            new_objects, c2w=c2w, K=K_arr,
                            image_width=DW, image_height=DH,
                        )
                        observed_ids = {o.object_id for o in returned}
                    decay_deleted = object_db.decay_unobserved(
                        observed_ids, c2w=c2w, K=K_arr,
                        image_width=DW, image_height=DH,
                    )

            if spatial_mem is not None:
                spatial_mem.maybe_store(small_bgr, c2w, ts)

            img_topic.publish(color_msg)
            cam_info_topic.publish(cam_info)
            depth_cam_info_topic.publish(cam_info)
            if dets2d is not None:
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
                if acc_cloud_topic is not None and acc_cloud is not None:
                    acc_cloud_topic.publish(acc_cloud.to_msg(ts))

            if object_db is not None and n % OBJECTS_PUBLISH_EVERY_N == 0:
                from dimos.perception.detection.type.detection3d.object import aggregate_pointclouds
                all_objs = object_db.get_all_objects()
                if all_objs:
                    obj_cloud_topic.publish(aggregate_pointclouds(all_objs))
                    scene_topic.publish(build_scene_update_for_objects(all_objs, ts))
                if OBJECT_LIST_SINK is not None:
                    cam_pos = c2w[:3, 3].astype(np.float64)
                    listing = []
                    for obj in all_objs:
                        ctr = obj.center
                        cxyz = np.array([ctr.x, ctr.y, ctr.z], dtype=np.float64)
                        dist = float(np.linalg.norm(cxyz - cam_pos))
                        listing.append({
                            "id": obj.object_id,
                            "name": obj.name,
                            "distance_m": round(dist, 2),
                            "detections": int(obj.detections_count),
                            "x": round(float(ctr.x), 3),
                            "y": round(float(ctr.y), 3),
                            "z": round(float(ctr.z), 3),
                        })
                    listing.sort(key=lambda o: o["distance_m"])
                    try:
                        OBJECT_LIST_SINK(listing, ts)
                    except Exception as e:  # noqa: BLE001 — never let the sink kill the pipeline
                        if n < 3:
                            print(f"  OBJECT_LIST_SINK failed: {e}")

            n += 1
            if n == 1 or n % max(1, int(args.max_fps)) == 0:
                d_pos = depth_m[depth_m > 0]
                total_px = depth_m.size
                drange = (f"[{d_pos.min():.2f},{d_pos.max():.2f}]m {len(d_pos)}/{total_px} "
                          f"({100*len(d_pos)/total_px:.0f}%)" if d_pos.size else "[empty]")
                pos = c2w[:3, 3]
                elapsed = time.perf_counter() - t_start
                det_info = ""
                if dets2d is not None:
                    names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                    stats = object_db.get_stats()
                    add_stats = object_db.get_last_add_stats() or {}
                    det_info = (f" objs(perm/pend)={stats['permanent_count']}/{stats['pending_count']}"
                                f" 2d={len(dets2d.detections)} [{names}]"
                                f" created={add_stats.get('created', 0)}"
                                f" match(t/d/p)={add_stats.get('matched_track', 0)}"
                                f"/{add_stats.get('matched_distance', 0)}"
                                f"/{add_stats.get('matched_pixel', 0)}"
                                f" decayed={len(decay_deleted)}")
                acc_info = f" acc={acc_cloud.size}" if acc_cloud is not None else ""
                print(f"  f{n} (src#{sf.frame_idx}): "
                      f"depth={t_depth*1000:.0f}ms pose={t_pose*1000:.0f}ms total={elapsed*1000:.0f}ms "
                      f"d={drange} pts={len(cam_pts_s)} pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                      f"voxels={voxel_map.size}{acc_info}{det_info}")

            if (map_writer.enabled and args.save_map_every_n > 0
                    and n > 0 and n % args.save_map_every_n == 0):
                map_writer.write(voxel_map,
                                 object_db if object_db is not None else NullObjectDB(),
                                 extra={"frames": n})

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        if map_writer.enabled:
            map_writer.write(voxel_map,
                             object_db if object_db is not None else NullObjectDB(),
                             extra={"frames": n})
        if keyframe_recorder is not None:
            keyframe_recorder.close()
        try:
            source.close()
        except Exception:
            pass
        if det2d is not None:
            try:
                det2d.detector.stop()
            except Exception:
                pass
        for t in (img_topic, ann_topic, cam_info_topic, depth_cam_info_topic,
                  depth_topic, points_topic, map_topic, acc_cloud_topic,
                  obj_cloud_topic, scene_topic, tf_topic):
            if t is None:
                continue
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
