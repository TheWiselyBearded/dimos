"""Real-time Viture stereo depth + Viture pose -> Foxglove.

Uses the device's two fisheye cameras (already undistorted to rectilinear via
xr-nav/scripts/make_stereo_videos.py) for stereo block matching. Disparity ->
metric depth via Z = fx * baseline / d. No depth-estimation network, no scale
ambiguity, runs realtime on CPU.

The undistorted .mp4 pair is NOT explicitly stereo-rectified (each eye is
independently fisheye-undistorted with R=I), but Viture's cameras are
mechanically parallel, so corresponding rows roughly match and SGBM copes.

Per frame:
  - Read aligned frame from undistorted_left.mp4 + undistorted_right.mp4
  - StereoSGBM disparity, validity-mask
  - disparity -> metric depth, build colored cloud via from_rgbd
  - Viture pose (ARKit -> OpenCV) transforms cloud to world
  - YOLO 2D + Detection3DModule for /scene_update cubes
  - Voxel-accumulate into /map

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
        /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_stereo_foxglove.py

Pair with bridge:
    /opt/anaconda3/envs/xr-nav/bin/python -m \\
        dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Foxglove (ws://localhost:8765):
  - 3D panel (frame=world): /points_frame, /map, /scene_update, /tf
  - Image panel: /color_image (Camera info -> /camera_info) + /annotations
                 /depth (Camera info -> /camera_info)
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
LEFT_VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_left.mp4")
RIGHT_VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_right.mp4")
RECORDING_DIR = Path(
    "/Users/reza/Documents/Projects/VitureSplat/viture_sdk/data/VITURE_recording_2026-03-29_14-34-22"
)
FISHEYE_CAL = RECORDING_DIR / "fisheye_calibration.json"

sys.path.insert(0, str(REPO / "xr-nav" / "src"))

PUBLISH_FPS = 10

# Stereo baseline (interpupillary). Viture XR ~63mm. If the accumulated map's
# scale looks consistently off (e.g. room appears 1.5x too big), this is the
# knob to tune.
BASELINE_M = 0.063

DEPTH_NEAR_M = 0.3
DEPTH_FAR_M = 6.0

POINTS_STRIDE = 4
MAP_VOXEL_M = 0.05
MAP_PUBLISH_EVERY_N = 30
MAX_MAP_POINTS = 600_000

# StereoSGBM params (numDisparities must be multiple of 16)
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 96
SGBM_BLOCK = 7


def compute_undistorted_fx(cal_path: Path, img_size: tuple[int, int]) -> tuple[float, float, float]:
    """Mirror xr-nav/scripts/make_stereo_videos.py to recover the fx of the
    rectilinear .mp4 frames. Returns (fx, cx, cy).
    """
    with open(cal_path) as f:
        cal = json.load(f)
    fx_fish = cal["fx"]
    cx_fish = cal["cx"]
    cy_fish = cal["cy"]
    K = np.array([[fx_fish, 0.0, cx_fish],
                  [0.0, fx_fish, cy_fish],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([cal.get("k1", 0.0), cal.get("k2", 0.0),
                  cal.get("k3", 0.0), cal.get("k4", 0.0)], dtype=np.float64)
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


def c2w_to_translation_quat(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation
    return c2w[:3, 3].astype(np.float64), Rotation.from_matrix(c2w[:3, :3]).as_quat()


def make_tf_from_c2w(c2w: np.ndarray, ts: float, parent: str = "world",
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


def arkit_c2w_to_opencv(c2w: np.ndarray) -> np.ndarray:
    out = c2w.copy()
    out[:3, 1] *= -1
    out[:3, 2] *= -1
    return out


def transform_points(c2w: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    rot = c2w[:3, :3].astype(np.float32)
    trans = c2w[:3, 3].astype(np.float32)
    return (rot @ pts_cam.astype(np.float32).T).T + trans


class PoseProvider:
    def __init__(self, recording_dir: Path):
        self.loader = None
        if not recording_dir.exists():
            print(f"[pose] missing dir: {recording_dir}")
            return
        try:
            from xr_nav.recording_loader import RecordingLoader
            self.loader = RecordingLoader(str(recording_dir), step=1)
        except Exception as e:
            print(f"[pose] failed: {e}")

    @property
    def num_frames(self) -> int:
        return 0 if self.loader is None else len(self.loader._frame_index)  # noqa: SLF001

    def for_frame(self, idx: int) -> np.ndarray | None:
        if self.loader is None:
            return None
        return self.loader.pose_for_frame_index(idx)


class MapAccumulator:
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
            self.pcd = self.pcd.voxel_down_sample(self.voxel * 1.5)

    def num_points(self) -> int:
        return len(self.pcd.points)

    def to_pointcloud2(self, frame_id: str, ts: float):
        from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
        return PointCloud2(pointcloud=self.pcd, frame_id=frame_id, ts=ts)


def main():
    if not LEFT_VIDEO.exists():
        sys.exit(f"missing left video: {LEFT_VIDEO}")
    if not RIGHT_VIDEO.exists():
        sys.exit(f"missing right video: {RIGHT_VIDEO}")
    if not FISHEYE_CAL.exists():
        sys.exit(f"missing fisheye calibration: {FISHEYE_CAL}")

    import open3d as o3d
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.Image import ImageFormat
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
    from dimos.msgs.geometry_msgs.Transform import Transform
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector
    from dimos.perception.detection.module2D import Detection2DModule
    from dimos.perception.detection.module3D import Detection3DModule
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage

    # ---- Videos ----
    cap_l = cv2.VideoCapture(str(LEFT_VIDEO))
    cap_r = cv2.VideoCapture(str(RIGHT_VIDEO))
    if not cap_l.isOpened() or not cap_r.isOpened():
        sys.exit("cv2 could not open one of the stereo videos")
    W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"stereo: {W}x{H}  {total_frames} frames")

    # ---- Pose ----
    pose_provider = PoseProvider(RECORDING_DIR)
    print(f"pose: {pose_provider.num_frames} frame timestamps")

    # ---- Intrinsics: fx of the rectilinear .mp4, recovered from fisheye cal ----
    fx, cx, cy = compute_undistorted_fx(FISHEYE_CAL, (W, H))
    fy = fx
    cam_info = CameraInfo(
        frame_id="camera_optical", height=H, width=W,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )
    print(f"intrinsics: fx={fx:.1f}  cx={cx:.1f} cy={cy:.1f}  baseline={BASELINE_M*1000:.0f}mm")
    print(f"depth Z @ d=1px: {fx*BASELINE_M:.2f}m  -- valid disp range will be {fx*BASELINE_M/DEPTH_FAR_M:.1f}..{fx*BASELINE_M/DEPTH_NEAR_M:.1f}px")

    # ---- StereoSGBM ----
    sgbm = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK,
        P1=8 * 3 * SGBM_BLOCK ** 2,
        P2=32 * 3 * SGBM_BLOCK ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # ---- Detection ----
    print("warming detectors (yolo cpu)...")
    det2d = Detection2DModule(detector=lambda: Yolo2DDetector(device="cpu"))
    det3d = Detection3DModule(camera_info=cam_info)
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
    depth_topic = LCMTransport("/depth", Image)
    points_topic = LCMTransport("/points_frame", PointCloud2)
    map_topic = LCMTransport("/map", PointCloud2)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    map_acc = MapAccumulator(voxel=MAP_VOXEL_M, cap=MAX_MAP_POINTS)

    period = 1.0 / PUBLISH_FPS
    n = 0
    loop_num = 0
    print(f"\npublishing at {PUBLISH_FPS} fps. Foxglove: ws://localhost:8765")
    print("  3D panel (frame=world): /points_frame, /map, /scene_update, /tf")
    print("  Image: /color_image (cam info /camera_info)  /depth (cam info /camera_info)")
    print("Ctrl+C to exit.\n")

    try:
        while True:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            if not ok_l or not ok_r:
                cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_num += 1
                map_acc = MapAccumulator(voxel=MAP_VOXEL_M, cap=MAX_MAP_POINTS)
                print(f"  --- loop {loop_num} (map reset) ---")
                continue

            mp4_idx = int(cap_l.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t_start = time.perf_counter()
            ts = time.time()

            # Both videos are grayscale-encoded; ensure 3-channel BGR for color msg
            if frame_l.ndim == 2:
                gray_l = frame_l
                gray_r = frame_r if frame_r.ndim == 2 else cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
                frame_l = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
                frame_r = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)
            else:
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            color_msg = Image.from_numpy(frame_l, format=ImageFormat.BGR,
                                         frame_id="camera_optical", ts=ts)

            dets2d = det2d.process_image_frame(color_msg)

            # ---- Stereo block matching ----
            t_d = time.perf_counter()
            disp_raw = sgbm.compute(gray_l, gray_r)  # int16, scaled by 16
            disp = disp_raw.astype(np.float32) / 16.0
            t_d = time.perf_counter() - t_d

            # disparity -> depth in meters, with validity mask
            valid = disp > 0.5  # SGBM marks invalid as < min_disp; require >= 0.5 to avoid div-by-zero noise
            depth_m = np.zeros_like(disp, dtype=np.float32)
            np.divide(fx * BASELINE_M, disp, out=depth_m, where=valid)
            depth_m = np.where((depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_FAR_M),
                               depth_m, 0.0).astype(np.float32)

            # Pose
            c2w_arkit = pose_provider.for_frame(mp4_idx)
            if c2w_arkit is None:
                c2w_arkit = np.eye(4)
            c2w = arkit_c2w_to_opencv(c2w_arkit)

            # Build colored cloud in camera_optical via from_rgbd
            depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                         frame_id="camera_optical", ts=ts)
            rgb = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
            color_msg_rgb = Image.from_numpy(rgb, format=ImageFormat.RGB,
                                             frame_id="camera_optical", ts=ts)
            cam_pcd = PointCloud2.from_rgbd(
                color_image=color_msg_rgb,
                depth_image=depth_msg,
                camera_info=cam_info,
                depth_scale=1.0,
                depth_trunc=DEPTH_FAR_M,
            )
            cam_pts, cam_cols = cam_pcd.as_numpy()
            if POINTS_STRIDE > 1 and len(cam_pts) > 0:
                cam_pts = cam_pts[::POINTS_STRIDE]
                if cam_cols is not None:
                    cam_cols = cam_cols[::POINTS_STRIDE]

            if len(cam_pts) > 0:
                world_pts = transform_points(c2w, cam_pts)
                world_pcd = o3d.geometry.PointCloud()
                world_pcd.points = o3d.utility.Vector3dVector(world_pts.astype(np.float64))
                if cam_cols is not None:
                    world_pcd.colors = o3d.utility.Vector3dVector(cam_cols.astype(np.float64))
                points_msg = PointCloud2(pointcloud=world_pcd, frame_id="world", ts=ts)
                map_acc.add(world_pts, cam_cols)
            else:
                points_msg = PointCloud2(pointcloud=o3d.geometry.PointCloud(),
                                         frame_id="world", ts=ts)

            # 3D detections
            dets3d = None
            try:
                dets3d = det3d.process_frame(dets2d, cam_pcd, identity_optical_tf)
            except Exception as e:
                if n == 0:
                    print(f"  3D det failed: {e}")

            # ---- Publish ----
            img_topic.publish(color_msg)
            cam_info_topic.publish(cam_info)
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
                d_valid_pct = 100.0 * float(valid.mean())
                print(f"  frame {n} (mp4#{mp4_idx}): "
                      f"sgbm={t_d*1000:.0f}ms total={elapsed*1000:.0f}ms "
                      f"valid_disp={d_valid_pct:.0f}% "
                      f"frame_pts={len(cam_pts)} map_pts={map_acc.num_points()} "
                      f"2d={len(dets2d.detections)} 3d={n3d} [{names}]")

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        cap_l.release()
        cap_r.release()
        try:
            det2d.detector.stop()
        except Exception:
            pass
        for t in (img_topic, ann_topic, cam_info_topic, depth_topic,
                  points_topic, map_topic, scene_topic, tf_topic):
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
