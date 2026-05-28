"""Real-time DA3 depth + Viture pose -> Foxglove.

Per frame from the undistorted Viture .mp4:
  - Look up the device pose (Viture IMU/SLAM, SLERP-interpolated to camera ts)
  - Run Depth Anything 3 to estimate depth
  - Build a colored point cloud, transform into world via the pose
  - Voxel-accumulate into a persistent map
  - Publish color, depth, per-frame cloud, accumulated map, TF, intrinsics, YOLO

Publishes:
  /color_image    -> RGB video frames (display-resolution)
  /annotations    -> 2D YOLO bbox overlay
  /camera_info    -> camera intrinsics (display-resolution)
  /depth          -> 32FC1 DA3 depth in meters (DA3-resolution)
  /points_frame   -> XYZRGB cloud for the current frame in world frame
  /map            -> XYZRGB voxel-accumulated cloud (republished periodically)
  /tf             -> world -> camera_optical, driven by Viture pose

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
        /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_da3_foxglove.py \\
        --da3-model da3metric-large
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_left.mp4")
RECORDING_DIR = Path(
    "/Users/reza/Documents/Projects/VitureSplat/viture_sdk/data/VITURE_recording_2026-03-29_14-34-22"
)

# Make xr_nav + DA3 importable (these live next to this script)
sys.path.insert(0, str(REPO / "xr-nav" / "src"))
sys.path.insert(0, str(REPO / "xr-nav" / "awesome-depth-anything-3" / "src"))

PUBLISH_FPS = 10
HFOV_DEG = 46.0

DA3_MODEL = "da3metric-large"  # default; override with --da3-model
DA3_DEVICE = "mps"             # default; override with --da3-device
DA3_RES = 504                  # default; override with --da3-res

DEPTH_NEAR_M = 0.3
DEPTH_FAR_M = 6.0

CONF_THRESHOLD = 0.5        # DA3 confidence: drop pixels below this from depth/cloud
POINTS_STRIDE = 8           # subsample DA3-resolution cloud before world-transform
MAP_VOXEL_M = 0.08          # voxel size for the accumulated map
MAP_PUBLISH_EVERY_N = 30    # republish /map at this period (every 3s @ 10fps)
MAX_MAP_POINTS = 600_000    # safety cap on accumulator


def normalize_depth(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """DA3 relative depth (higher = farther) -> pseudo-metric [near_m, far_m].

    Adapted from xr-nav/scripts/run_da3_video.py:128. The 5% floor in the
    original is replaced by an explicit `near_m` so depth never collapses to 0.
    """
    d = depth.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-8:
        return np.full_like(d, 0.5 * (near_m + far_m))
    d_norm = (d - d_min) / (d_max - d_min)
    return near_m + d_norm * (far_m - near_m)


def c2w_to_translation_quat(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 4x4 camera-to-world matrix into (translation, quat[x,y,z,w])."""
    from scipy.spatial.transform import Rotation
    t = c2w[:3, 3].astype(np.float64)
    q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [x, y, z, w]
    return t, q


def make_tf_from_c2w(c2w: np.ndarray, ts: float, parent: str = "world",
                     child: str = "camera_optical"):
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from dimos_lcm.geometry_msgs.TransformStamped import TransformStamped
    from dimos_lcm.geometry_msgs.Transform import Transform as LCMT
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion as LQ
    from dimos_lcm.std_msgs.Header import Header
    from dimos_lcm.std_msgs.Time import Time

    sec = int(ts)
    nsec = int((ts - sec) * 1e9)
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


class PoseProvider:
    """Wraps RecordingLoader for per-mp4-frame pose lookup. No PGM iteration."""

    def __init__(self, recording_dir: Path):
        self.loader = None
        self.last_c2w: np.ndarray | None = None
        self.world_from_first_da3: np.ndarray | None = None  # for DA3 fallback
        if not recording_dir.exists():
            print(f"[pose] recording dir missing: {recording_dir} -- DA3 fallback will be used")
            return
        try:
            from xr_nav.recording_loader import RecordingLoader
            self.loader = RecordingLoader(str(recording_dir), step=1)
        except Exception as e:
            print(f"[pose] failed to load recording: {e} -- DA3 fallback will be used")

    @property
    def num_frames(self) -> int:
        if self.loader is None:
            return 0
        return len(self.loader._frame_index)  # noqa: SLF001 — public-equivalent

    def for_frame(self, idx: int) -> np.ndarray | None:
        if self.loader is None:
            return None
        c2w = self.loader.pose_for_frame_index(idx)
        if c2w is not None:
            self.last_c2w = c2w
        return c2w

    def fallback_from_da3(self, da3_extrinsic: np.ndarray | None) -> np.ndarray | None:
        """When Viture pose is missing, treat first DA3 extrinsic as world origin."""
        if da3_extrinsic is None:
            return None
        ext = da3_extrinsic.astype(np.float64)
        if self.world_from_first_da3 is None:
            self.world_from_first_da3 = np.linalg.inv(ext)
        return self.world_from_first_da3 @ ext


class MapAccumulator:
    """Voxel-downsampled colored point cloud accumulator.

    Holds a single Open3D PointCloud; on each insert appends new points/colors
    and re-runs voxel_down_sample so memory stays bounded. If the cap is
    exceeded after downsampling we tighten the voxel temporarily.
    """

    def __init__(self, voxel: float, cap: int):
        import open3d as o3d
        self._o3d = o3d
        self.voxel = voxel
        self.cap = cap
        self.pcd = o3d.geometry.PointCloud()

    def add(self, world_pts: np.ndarray, colors: np.ndarray | None) -> None:
        if world_pts.size == 0:
            return
        new_pcd = self._o3d.geometry.PointCloud()
        new_pcd.points = self._o3d.utility.Vector3dVector(world_pts.astype(np.float64))
        if colors is not None and len(colors) == len(world_pts):
            new_pcd.colors = self._o3d.utility.Vector3dVector(colors.astype(np.float64))
        self.pcd += new_pcd
        self.pcd = self.pcd.voxel_down_sample(self.voxel)
        if len(self.pcd.points) > self.cap:
            # Last-resort tighten: re-downsample at 1.5x voxel
            self.pcd = self.pcd.voxel_down_sample(self.voxel * 1.5)

    def num_points(self) -> int:
        return len(self.pcd.points)

    def to_pointcloud2(self, frame_id: str, ts: float):
        from dimos.msgs.sensor_msgs import PointCloud2
        return PointCloud2(pointcloud=self.pcd, frame_id=frame_id, ts=ts)


def build_da3_camera_info(K_da3: np.ndarray | None, w: int, h: int, hfov_deg: float):
    """CameraInfo for the DA3-resolution depth used by from_rgbd.

    Prefer DA3-estimated intrinsics when they look sane (focal in 0.3*w..5*w).
    """
    from dimos.msgs.sensor_msgs import CameraInfo

    if K_da3 is not None:
        fx, fy = float(K_da3[0, 0]), float(K_da3[1, 1])
        cx, cy = float(K_da3[0, 2]), float(K_da3[1, 2])
        if 0.3 * w < fx < 5 * w and 0.3 * h < fy < 5 * h:
            return CameraInfo(
                frame_id="camera_optical", height=h, width=w,
                distortion_model="plumb_bob", D=[0.0] * 5,
                K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
                R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
                binning_x=0, binning_y=0,
            )
    # HFOV fallback
    fx = fy = (w / 2) / np.tan(np.deg2rad(hfov_deg / 2))
    cx, cy = w / 2, h / 2
    return CameraInfo(
        frame_id="camera_optical", height=h, width=w,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )


def arkit_c2w_to_opencv(c2w: np.ndarray) -> np.ndarray:
    """Viture poses are ARKit convention (X-right, Y-up, Z-back).
    `from_rgbd` unprojects depth in OpenCV optical (X-right, Y-down, Z-forward).
    To use the pose with optical-frame points, flip the Y and Z basis columns.
    Mirrors xr-nav/src/xr_nav/pipeline.py:232-234.
    """
    out = c2w.copy()
    out[:3, 1] *= -1
    out[:3, 2] *= -1
    return out


def transform_points(c2w_opencv: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    rot = c2w_opencv[:3, :3].astype(np.float32)
    trans = c2w_opencv[:3, 3].astype(np.float32)
    return (rot @ pts_cam.astype(np.float32).T).T + trans


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--da3-model", default=DA3_MODEL,
                        choices=["da3-small", "da3-base", "da3-large",
                                 "da3-giant", "da3metric-large",
                                 "da3nested-giant-large"],
                        help="DA3 variant. da3metric-large (default) returns true "
                             "metric depth so per-frame scale stays consistent. "
                             "da3-small/base/large are scale-ambiguous.")
    parser.add_argument("--da3-device", default=DA3_DEVICE,
                        choices=["mps", "cuda", "cpu"])
    parser.add_argument("--da3-res", type=int, default=DA3_RES,
                        help="DA3 processing resolution (default: 504)")
    args = parser.parse_args()
    da3_model = args.da3_model
    da3_device = args.da3_device
    da3_res = args.da3_res

    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    import open3d as o3d
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image, ImageFormat, PointCloud2, CameraInfo
    from dimos.msgs.geometry_msgs import Transform, Vector3, Quaternion
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector
    from dimos.perception.detection.module2D import Detection2DModule
    from dimos.perception.detection.module3D import Detection3DModule
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage

    # ---- Video ----
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"cv2 could not open {VIDEO}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or PUBLISH_FPS
    print(f"video: {VIDEO.name}  {W}x{H}  {total_frames} frames  {native_fps:.1f} fps")

    # ---- Pose ----
    pose_provider = PoseProvider(RECORDING_DIR)
    if pose_provider.num_frames:
        print(f"pose: {pose_provider.num_frames} frame timestamps in camera_index.csv "
              f"(mp4 has {total_frames})")
        if pose_provider.num_frames < total_frames:
            print(f"  WARN: fewer pose timestamps than mp4 frames -- tail will fall back to DA3")

    # ---- Display intrinsics (1280-wide, for /color_image) ----
    DISPLAY_W = 1280
    scale = DISPLAY_W / W
    DW, DH = DISPLAY_W, int(H * scale)
    DFX = (W / 2) / np.tan(np.deg2rad(HFOV_DEG / 2)) * scale
    DFY = DFX
    DCX, DCY = DW / 2, DH / 2
    display_cam_info = CameraInfo(
        frame_id="camera_optical", height=DH, width=DW,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[DFX, 0, DCX, 0, DFY, DCY, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[DFX, 0, DCX, 0, 0, DFY, DCY, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )
    print(f"display: {DW}x{DH} (scale={scale:.2f})  HFOV={HFOV_DEG}°")

    # ---- DA3 ----
    print(f"loading DA3 model '{da3_model}' on {da3_device}...")
    from depth_anything_3.api import DepthAnything3
    t0 = time.monotonic()
    # Load pretrained weights via from_pretrained: the bare constructor only builds
    # the architecture and leaves the net uninitialized, giving a flat near-constant
    # depth that gets stretched into blocky garbage.
    da3 = DepthAnything3.from_pretrained(
        f"depth-anything/{da3_model.upper()}").to(da3_device)
    # Prediction.is_metric is unreliable (returns {} -> falsy); trust the model name.
    da3_is_metric = "metric" in da3_model.lower()
    print(f"  DA3 ready in {time.monotonic() - t0:.1f}s on {da3.device}")

    # ---- Detection (2D + 3D, like mac_unitree_replay_foxglove.py) ----
    print("warming detectors (yolo cpu)...")
    det2d = Detection2DModule(detector=lambda: Yolo2DDetector(device="cpu"))
    det3d = Detection3DModule(camera_info=display_cam_info)
    # 3D module projects the cloud through display intrinsics; identity transform
    # because our DA3 cloud is already expressed in camera_optical.
    identity_optical_tf = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="camera_optical",
        child_frame_id="camera_optical",
    )

    # ---- Transports ----
    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)
    depth_cam_info_topic = LCMTransport("/depth_camera_info", CameraInfo)
    depth_topic = LCMTransport("/depth", Image)
    points_topic = LCMTransport("/points_frame", PointCloud2)
    map_topic = LCMTransport("/map", PointCloud2)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    map_acc = MapAccumulator(voxel=MAP_VOXEL_M, cap=MAX_MAP_POINTS)

    period = 1.0 / PUBLISH_FPS
    n = 0
    loop_num = 0
    print(f"\npublishing at {PUBLISH_FPS} fps (looping). Open Foxglove (ws://localhost:8765)")
    print("  3D panel (frame=world): /points_frame, /map, /scene_update, /tf")
    print("  Image panels: /color_image  (Camera info -> /camera_info)")
    print("                /depth         (Camera info -> /depth_camera_info)")
    print("Ctrl+C to exit.\n")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_num += 1
                map_acc = MapAccumulator(voxel=MAP_VOXEL_M, cap=MAX_MAP_POINTS)
                pose_provider.world_from_first_da3 = None
                print(f"  --- loop {loop_num} (map reset) ---")
                continue

            mp4_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t_start = time.perf_counter()
            ts = time.time()

            if frame_bgr.ndim == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

            small_bgr = cv2.resize(frame_bgr, (DW, DH))
            color_msg = Image.from_numpy(small_bgr, format=ImageFormat.BGR,
                                         frame_id="camera_optical", ts=ts)

            # YOLO 2D detections via Detection2DModule (matches unitree script)
            dets2d = det2d.process_image_frame(color_msg)

            # DA3 inference on RGB at display resolution
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            t_da3 = time.perf_counter()
            try:
                pred = da3.inference(image=[small_rgb], process_res=da3_res)
            except Exception as e:
                print(f"  frame {n}: DA3 failed: {e}")
                n += 1
                continue
            t_da3 = time.perf_counter() - t_da3

            raw_depth = np.nan_to_num(pred.depth[0].astype(np.float32),
                                      nan=0.0, posinf=0.0, neginf=0.0)
            is_metric = da3_is_metric or bool(getattr(pred, "is_metric", 0))
            depth_m = (raw_depth if is_metric
                       else normalize_depth(raw_depth, DEPTH_NEAR_M, DEPTH_FAR_M))
            # DA3 confidence: zero out low-confidence pixels so from_rgbd skips them
            if pred.conf is not None:
                conf = pred.conf[0].astype(np.float32)
                cmax = float(conf.max()) if conf.size else 1.0
                if cmax > 1.0:
                    conf = conf / cmax
                depth_m = np.where(conf >= CONF_THRESHOLD, depth_m, 0.0).astype(np.float32)
            if not np.isfinite(depth_m).any():
                n += 1
                continue
            dh, dw = depth_m.shape

            # Per-frame DA3 intrinsics for from_rgbd
            K_da3 = pred.intrinsics[0] if pred.intrinsics is not None else None
            da3_cam_info = build_da3_camera_info(K_da3, dw, dh, HFOV_DEG)

            # Pose: Viture primary, DA3 fallback. Convert to OpenCV optical convention
            # so it composes with from_rgbd's optical-frame points.
            c2w_raw = pose_provider.for_frame(mp4_idx)
            pose_src = "viture"
            if c2w_raw is None:
                ext = (pred.extrinsics[0] if pred.extrinsics is not None else None)
                c2w_raw = pose_provider.fallback_from_da3(ext)
                pose_src = "da3"
            if c2w_raw is None:
                c2w_raw = np.eye(4)
                pose_src = "identity"
            # Viture poses are ARKit; DA3 extrinsics are already OpenCV-style.
            c2w = arkit_c2w_to_opencv(c2w_raw) if pose_src == "viture" else c2w_raw

            # Build colored cloud in camera frame via from_rgbd
            depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                         frame_id="camera_optical", ts=ts)
            color_for_cloud = cv2.resize(small_rgb, (dw, dh))
            color_da3 = Image.from_numpy(color_for_cloud, format=ImageFormat.RGB,
                                         frame_id="camera_optical", ts=ts)
            cam_pcd = PointCloud2.from_rgbd(
                color_image=color_da3,
                depth_image=depth_msg,
                camera_info=da3_cam_info,
                depth_scale=1.0,
                depth_trunc=DEPTH_FAR_M,
            )
            cam_pts, cam_cols = cam_pcd.as_numpy()

            if POINTS_STRIDE > 1 and len(cam_pts) > 0:
                cam_pts = cam_pts[::POINTS_STRIDE]
                if cam_cols is not None:
                    cam_cols = cam_cols[::POINTS_STRIDE]

            # Transform camera -> world via c2w
            if len(cam_pts) > 0:
                world_pts = transform_points(c2w, cam_pts)
                # World-frame colored cloud
                world_pcd = o3d.geometry.PointCloud()
                world_pcd.points = o3d.utility.Vector3dVector(world_pts.astype(np.float64))
                if cam_cols is not None:
                    world_pcd.colors = o3d.utility.Vector3dVector(cam_cols.astype(np.float64))
                points_msg = PointCloud2(pointcloud=world_pcd, frame_id="world", ts=ts)

                map_acc.add(world_pts, cam_cols)
            else:
                points_msg = PointCloud2(pointcloud=o3d.geometry.PointCloud(),
                                         frame_id="world", ts=ts)

            # 3D detections: project the per-frame DA3 cloud through display
            # intrinsics, gate by 2D bboxes, build cubes. cam_pcd is in
            # camera_optical, so identity transform suffices.
            dets3d = None
            try:
                dets3d = det3d.process_frame(dets2d, cam_pcd, identity_optical_tf)
            except Exception as e:
                if n == 0:
                    print(f"  3D det failed (will retry): {e}")

            # ---- Publish ----
            img_topic.publish(color_msg)
            cam_info_topic.publish(display_cam_info)
            depth_cam_info_topic.publish(da3_cam_info)
            ann_topic.publish(dets2d.to_foxglove_annotations())
            depth_topic.publish(depth_msg)
            points_topic.publish(points_msg)
            tf_topic.publish(make_tf_from_c2w(c2w, ts))

            if dets3d is not None and len(dets3d.detections):
                scene_topic.publish(dets3d.to_foxglove_scene_update())

            if n % MAP_PUBLISH_EVERY_N == 0 and map_acc.num_points() > 0:
                map_topic.publish(map_acc.to_pointcloud2(frame_id="world", ts=ts))

            n += 1
            if n == 1 or n % PUBLISH_FPS == 0:
                elapsed = time.perf_counter() - t_start
                names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                n3d = len(dets3d.detections) if dets3d is not None else 0
                print(f"  frame {n} (mp4#{mp4_idx}): "
                      f"depth={t_da3*1000:.0f}ms total={elapsed*1000:.0f}ms "
                      f"frame_pts={len(cam_pts)} map_pts={map_acc.num_points()} "
                      f"pose={pose_src} 2d={len(dets2d.detections)} 3d={n3d} [{names}]")

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        cap.release()
        try:
            det2d.detector.stop()
        except Exception:
            pass
        for t in (img_topic, ann_topic, depth_topic, points_topic,
                  map_topic, tf_topic, cam_info_topic,
                  depth_cam_info_topic, scene_topic):
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
