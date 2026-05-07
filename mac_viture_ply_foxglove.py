"""Viture color/pose + an existing PLY map -> Foxglove with 3D segmentation.

No depth-estimation network. The PLY is the spatial map; Viture device pose
positions the camera in that map's frame. YOLO 2D detections are lifted to 3D
by projecting the PLY through the camera's display intrinsics — Detection3D
filters cloud points that fall inside each 2D bbox and fits a cube. Result:
clean 3D bounding boxes aligned to the real scan, instead of being attached
to noisy DA3-derived clouds.

Companion to mac_viture_da3_foxglove.py for the static-map workflow.

Publishes:
  /color_image    -> RGB video frames (display-resolution)
  /annotations    -> 2D YOLO bbox overlay
  /camera_info    -> camera intrinsics (display-resolution)
  /lidar          -> the PLY map (frame=world, periodic republish)
  /scene_update   -> 3D detection cubes pulled from the PLY
  /tf             -> world -> camera_optical, driven by Viture pose

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Run:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
        /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_ply_foxglove.py

Foxglove (ws://localhost:8765):
  - 3D panel (frame=world): enable /lidar, /scene_update, /tf
  - Image panel: /color_image with /annotations overlay
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
PLY = Path("/Users/reza/Documents/Tools/room-map/artifacts/a332f615c793/map.ply")

# Make xr_nav importable for the pose loader
sys.path.insert(0, str(REPO / "xr-nav" / "src"))

PUBLISH_FPS = 10
HFOV_DEG = 46.0
DISPLAY_W = 1280

MAP_MAX_POINTS = 200_000      # downsample PLY to this on load
MAP_PUBLISH_EVERY_N = 30      # republish /lidar every 3s @ 10fps
DEPTH_CAP_M = 8.0             # detection cube range (passed via filter no-op)

# If the PLY's world frame is rotated/translated relative to the Viture pose's
# world frame, set this 4x4 to align them. Identity assumes both share origin.
# To tune: open Foxglove, see how /lidar lines up with the colored frustum
# (camera_optical TF) and /scene_update; adjust here, re-run.
WORLD_FROM_PLY = np.eye(4, dtype=np.float64)


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
    """Viture poses are ARKit (X-right, Y-up, Z-back). Flip Y/Z basis columns
    to match the OpenCV optical convention used by camera intrinsics."""
    out = c2w.copy()
    out[:3, 1] *= -1
    out[:3, 2] *= -1
    return out


def transform_to_lcm(T: np.ndarray, frame_id: str, child_frame_id: str):
    """4x4 numpy -> dimos.msgs.geometry_msgs.Transform."""
    from dimos.msgs.geometry_msgs import Transform, Vector3, Quaternion
    t, q = c2w_to_translation_quat(T)
    return Transform(
        translation=Vector3(float(t[0]), float(t[1]), float(t[2])),
        rotation=Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        frame_id=frame_id,
        child_frame_id=child_frame_id,
    )


def load_ply_as_pointcloud2(ply_path: Path, frame_id: str, max_points: int):
    """Load PLY, optionally downsample, and return a colored PointCloud2 in `frame_id`."""
    import open3d as o3d
    from dimos.msgs.sensor_msgs import PointCloud2

    print(f"loading {ply_path.name} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n0 = len(pcd.points)
    if n0 > max_points:
        # Random downsample is faster than voxel for "just trim" purposes
        ratio = max_points / n0
        pcd = pcd.random_down_sample(ratio)
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"  {n0} -> {len(pts)} points  AABB min={pts.min(0)}  max={pts.max(0)}")

    # Optional WORLD_FROM_PLY alignment
    if not np.allclose(WORLD_FROM_PLY, np.eye(4)):
        ones = np.ones((len(pts), 1))
        homo = np.hstack([pts, ones])
        pts = (WORLD_FROM_PLY @ homo.T).T[:, :3]
        pcd.points = o3d.utility.Vector3dVector(pts)

    return PointCloud2(pointcloud=pcd, frame_id=frame_id, ts=time.time())


def main():
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")
    if not PLY.exists():
        sys.exit(f"missing PLY: {PLY}")

    from xr_nav.recording_loader import RecordingLoader
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image, ImageFormat, PointCloud2, CameraInfo
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

    # ---- Pose loader ----
    if not RECORDING_DIR.exists():
        sys.exit(f"missing recording dir: {RECORDING_DIR}")
    loader = RecordingLoader(str(RECORDING_DIR), step=1)
    if loader.pose_for_frame_index(0) is None:
        sys.exit("recording has no usable pose data")
    print(f"pose: {len(loader._frame_index)} frame timestamps")  # noqa: SLF001

    # ---- Display intrinsics ----
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
    print(f"display: {DW}x{DH}  HFOV={HFOV_DEG}°")

    # ---- Map ----
    map_pcd = load_ply_as_pointcloud2(PLY, frame_id="world", max_points=MAP_MAX_POINTS)

    # ---- Detection ----
    print("warming detectors (yolo cpu)...")
    det2d = Detection2DModule(detector=lambda: Yolo2DDetector(device="cpu"))
    det3d = Detection3DModule(camera_info=cam_info)

    # ---- Transports ----
    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)
    lidar_topic = LCMTransport("/lidar", PointCloud2)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    period = 1.0 / PUBLISH_FPS
    n = 0
    loop_num = 0
    print(f"\npublishing at {PUBLISH_FPS} fps (looping). Open Foxglove (ws://localhost:8765)")
    print("  3D panel (frame=world): /lidar, /scene_update, /tf")
    print("  Image panel: /color_image (+ /annotations)")
    print("Ctrl+C to exit.\n")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_num += 1
                print(f"  --- loop {loop_num} ---")
                continue

            mp4_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t_start = time.perf_counter()
            ts = time.time()

            if frame_bgr.ndim == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            small_bgr = cv2.resize(frame_bgr, (DW, DH))
            color_msg = Image.from_numpy(small_bgr, format=ImageFormat.BGR,
                                         frame_id="camera_optical", ts=ts)

            # YOLO 2D
            dets2d = det2d.process_image_frame(color_msg)

            # Pose
            c2w_arkit = loader.pose_for_frame_index(mp4_idx)
            if c2w_arkit is None:
                # off the end of pose data — hold at the last published frame
                c2w_arkit = loader.pose_for_frame_index(len(loader._frame_index) - 1)  # noqa: SLF001
                if c2w_arkit is None:
                    c2w_arkit = np.eye(4)
            c2w = arkit_c2w_to_opencv(c2w_arkit)

            # 3D detections: project the static PLY through the camera
            world_to_optical_4x4 = np.linalg.inv(c2w)
            world_to_optical = transform_to_lcm(
                world_to_optical_4x4,
                frame_id="world",
                child_frame_id="camera_optical",
            )
            dets3d = None
            try:
                dets3d = det3d.process_frame(dets2d, map_pcd, world_to_optical)
            except Exception as e:
                if n == 0:
                    print(f"  3D det failed: {e}")

            # ---- Publish ----
            img_topic.publish(color_msg)
            cam_info_topic.publish(cam_info)
            ann_topic.publish(dets2d.to_foxglove_annotations())
            tf_topic.publish(make_tf_from_c2w(c2w, ts))

            if dets3d is not None and len(dets3d.detections):
                scene_topic.publish(dets3d.to_foxglove_scene_update())

            if n % MAP_PUBLISH_EVERY_N == 0:
                map_pcd.ts = ts
                lidar_topic.publish(map_pcd)

            n += 1
            if n == 1 or n % PUBLISH_FPS == 0:
                elapsed = time.perf_counter() - t_start
                names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                n3d = len(dets3d.detections) if dets3d is not None else 0
                print(f"  frame {n} (mp4#{mp4_idx}): "
                      f"total={elapsed*1000:.0f}ms 2d={len(dets2d.detections)} 3d={n3d} [{names}]")

            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        cap.release()
        try:
            det2d.detector.stop()
        except Exception:
            pass
        for t in (img_topic, ann_topic, cam_info_topic, lidar_topic, scene_topic, tf_topic):
            try:
                t.lcm.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
