"""Real-time Apple Depth Pro + Viture pose -> Foxglove.

Mirrors mac_viture_da3_foxglove.py but swaps DA3 (relative depth, normalized by
heuristic) for Apple's Depth Pro (metric depth, no scale ambiguity). The
accumulated map is therefore in real meters and naturally aligned to the
Viture pose's coordinate system. No PLY required.

Per frame:
  - Read .mp4 frame, look up Viture pose (SLERP-interpolated)
  - Run Depth Pro -> metric depth (in meters)
  - Build colored cloud via from_rgbd, transform to world via pose
  - YOLO 2D + Detection3DModule -> /scene_update cubes anchored to the cloud
  - Voxel-accumulate into /map

Install (one-time, in the xr-nav conda env):
    /opt/anaconda3/envs/xr-nav/bin/pip install \\
        git+https://github.com/apple/ml-depth-pro.git
    cd /tmp && git clone https://github.com/apple/ml-depth-pro.git ml-depth-pro-ckpt && \\
        cd ml-depth-pro-ckpt && bash get_pretrained_models.sh
    # Then move checkpoints to wherever depth_pro expects (default: ./checkpoints/)
    # Or set the DEPTHPRO_CHECKPOINT env var if depth_pro supports it.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
        /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_depthpro_foxglove.py

Pair with bridge in another terminal:
    /opt/anaconda3/envs/xr-nav/bin/python -m \\
        dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Foxglove (ws://localhost:8765):
  - 3D panel (frame=world): /points_frame, /map, /scene_update, /tf
  - Image panels: /color_image (Camera info -> /camera_info)
                  /depth (Camera info -> /depth_camera_info)
"""

from __future__ import annotations

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

sys.path.insert(0, str(REPO / "xr-nav" / "src"))

# Depth Pro is heavier than DA3-small (~300ms/frame on M-series MPS at 768
# input), so we publish slower than the DA3 script. Bump if your machine keeps
# up; drop if you can't keep up with the live pose track.
PUBLISH_FPS = 3
HFOV_DEG = 46.0
DISPLAY_W = 1024              # also the depth-pro inference width

DEPTHPRO_DEVICE = "mps"       # mps | cuda | cpu
DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0             # truncate metric depth past this for the cloud

POINTS_STRIDE = 6             # subsample dense per-frame cloud
MAP_VOXEL_M = 0.05            # voxel size for accumulated map
MAP_PUBLISH_EVERY_N = 10      # republish /map every ~3s @ 3fps
MAX_MAP_POINTS = 600_000


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
    """Viture pose (ARKit, +Y up, +Z back) -> OpenCV optical (+Y down, +Z fwd)."""
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
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    try:
        import depth_pro
    except ImportError:
        sys.exit("depth_pro not installed. Install with:\n"
                 "  /opt/anaconda3/envs/xr-nav/bin/pip install "
                 "git+https://github.com/apple/ml-depth-pro.git\n"
                 "Then download checkpoints (~2GB) per the depth_pro README.")
    import torch
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

    # ---- Video ----
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"cv2 could not open {VIDEO}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {VIDEO.name}  {W}x{H}  {total_frames} frames")

    # ---- Pose ----
    pose_provider = PoseProvider(RECORDING_DIR)
    print(f"pose: {pose_provider.num_frames} frame timestamps")

    # ---- Display intrinsics: depth-pro outputs depth at the input resolution,
    # so the same camera_info works for /color_image, /depth and from_rgbd. ----
    scale = DISPLAY_W / W
    DW, DH = DISPLAY_W, int(H * scale)
    DFX = (W / 2) / np.tan(np.deg2rad(HFOV_DEG / 2)) * scale
    DFY = DFX
    DCX, DCY = DW / 2, DH / 2
    cam_info = CameraInfo(
        frame_id="camera_optical", height=DH, width=DW,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[DFX, 0, DCX, 0, DFY, DCY, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[DFX, 0, DCX, 0, 0, DFY, DCY, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )
    print(f"display: {DW}x{DH}  HFOV={HFOV_DEG}°  fx={DFX:.1f}")

    # ---- Depth Pro ----
    print(f"loading depth-pro on {DEPTHPRO_DEVICE}...")
    t0 = time.monotonic()
    dp_model, dp_transform = depth_pro.create_model_and_transforms()
    dp_model.eval()
    dp_device = torch.device(DEPTHPRO_DEVICE)
    dp_model = dp_model.to(dp_device)
    print(f"  depth-pro ready in {time.monotonic() - t0:.1f}s")

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
    print(f"\npublishing at {PUBLISH_FPS} fps (looping). Foxglove: ws://localhost:8765")
    print("  3D panel (frame=world): /points_frame, /map, /scene_update, /tf")
    print("  Image: /color_image (cam info /camera_info)  /depth (cam info /depth_camera_info)")
    print("Ctrl+C to exit.\n")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_num += 1
                map_acc = MapAccumulator(voxel=MAP_VOXEL_M, cap=MAX_MAP_POINTS)
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
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

            # YOLO 2D
            dets2d = det2d.process_image_frame(color_msg)

            # Depth Pro inference (metric meters, same resolution as input)
            t_depth = time.perf_counter()
            try:
                with torch.no_grad():
                    dp_input = dp_transform(small_rgb).to(dp_device)
                    pred = dp_model.infer(dp_input, f_px=torch.tensor(DFX).to(dp_device))
                depth_t = pred["depth"]
                depth_m = depth_t.detach().cpu().numpy().astype(np.float32)
                if depth_m.ndim == 3:  # squeeze any batch dim
                    depth_m = depth_m[0]
            except Exception as e:
                print(f"  frame {n}: depth-pro failed: {e}")
                n += 1
                continue
            t_depth = time.perf_counter() - t_depth

            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            # Hard clip the near/far range — depth-pro can output sky as ~inf
            depth_m = np.where((depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_FAR_M),
                               depth_m, 0.0).astype(np.float32)

            # Match depth dims to color dims (depth-pro outputs at input res, but
            # be defensive in case of off-by-one rounding).
            dh, dw = depth_m.shape
            if (dh, dw) != (DH, DW):
                depth_m = cv2.resize(depth_m, (DW, DH), interpolation=cv2.INTER_NEAREST)

            # Pose: Viture (ARKit) -> OpenCV
            c2w_arkit = pose_provider.for_frame(mp4_idx)
            if c2w_arkit is None:
                c2w_arkit = pose_provider.for_frame(min(mp4_idx, max(0, pose_provider.num_frames - 1)))
            if c2w_arkit is None:
                c2w_arkit = np.eye(4)
            c2w = arkit_c2w_to_opencv(c2w_arkit)

            # Build colored cloud in camera frame via from_rgbd
            depth_msg = Image.from_numpy(depth_m, format=ImageFormat.DEPTH,
                                         frame_id="camera_optical", ts=ts)
            color_for_cloud = small_rgb  # already at depth resolution
            color_msg_rgb = Image.from_numpy(color_for_cloud, format=ImageFormat.RGB,
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

            # 3D detections off the per-frame cloud (camera_optical frame)
            dets3d = None
            try:
                dets3d = det3d.process_frame(dets2d, cam_pcd, identity_optical_tf)
            except Exception as e:
                if n == 0:
                    print(f"  3D det failed: {e}")

            # ---- Publish ----
            img_topic.publish(color_msg)
            cam_info_topic.publish(cam_info)
            depth_cam_info_topic.publish(cam_info)
            ann_topic.publish(dets2d.to_foxglove_annotations())
            depth_topic.publish(depth_msg)
            points_topic.publish(points_msg)
            tf_topic.publish(make_tf_from_c2w(c2w, ts))

            if dets3d is not None and len(dets3d.detections):
                scene_topic.publish(dets3d.to_foxglove_scene_update())

            if n % MAP_PUBLISH_EVERY_N == 0 and map_acc.num_points() > 0:
                map_topic.publish(map_acc.to_pointcloud2(frame_id="world", ts=ts))

            n += 1
            if n == 1 or n % max(1, PUBLISH_FPS) == 0:
                elapsed = time.perf_counter() - t_start
                names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                n3d = len(dets3d.detections) if dets3d is not None else 0
                print(f"  frame {n} (mp4#{mp4_idx}): "
                      f"depth={t_depth*1000:.0f}ms total={elapsed*1000:.0f}ms "
                      f"depth_range=[{depth_m[depth_m>0].min() if (depth_m>0).any() else 0:.2f},"
                      f"{depth_m.max():.2f}]m "
                      f"frame_pts={len(cam_pts)} map_pts={map_acc.num_points()} "
                      f"2d={len(dets2d.detections)} 3d={n3d} [{names}]")

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        cap.release()
        try:
            det2d.detector.stop()
        except Exception:
            pass
        for t in (img_topic, ann_topic, cam_info_topic, depth_cam_info_topic,
                  depth_topic, points_topic, map_topic, scene_topic, tf_topic):
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
