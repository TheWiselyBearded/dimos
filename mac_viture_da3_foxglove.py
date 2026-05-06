"""Visualize Viture XR captures in Foxglove.

Streams the undistorted left-eye video alongside the DA3-built PLY map.

Publishes:
  /color_image    -> RGB video frames (looping)
  /annotations    -> 2D YOLO bbox overlay
  /lidar          -> the .ply map (static, republished each frame)
  /camera_info    -> camera intrinsics
  /tf             -> world -> camera_optical (identity)

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Then in Foxglove (ws://localhost:8765):
  - 3D panel: display frame=world; enable /lidar + /tf
  - Image panel on /color_image with /annotations overlay
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
MAP_PLY = REPO / "xr-nav" / "maps" / "map.ply"
VIDEO = Path("/Users/reza/Downloads/VITURE_recording_2026-03-29_14-34-22_undistorted_left.mp4")

PUBLISH_FPS = 10        # video + YOLO frame rate
LIDAR_EVERY_N = 30     # publish the static map much less often (every 3s at 10fps)
HFOV_DEG = 46.0
DEPTH_SCALE = 0.001   # kept for optional depth frame use (mm → m)
DEPTH_TRUNC_M = 8.0


def load_map_pcd(max_points=200_000):
    import open3d as o3d
    print(f"loading {MAP_PLY.name} ...")
    p = o3d.io.read_point_cloud(str(MAP_PLY))
    pts = np.asarray(p.points, dtype=np.float32)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        print(f"  downsampled to {len(pts)} points (from original)")
    else:
        print(f"  {len(pts)} points")
    print(f"  AABB min={pts.min(0)} max={pts.max(0)}")
    return pts


def make_static_tf(ts):
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from dimos_lcm.geometry_msgs.TransformStamped import TransformStamped
    from dimos_lcm.geometry_msgs.Transform import Transform as LCMT
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion as LQ
    from dimos_lcm.std_msgs.Header import Header
    from dimos_lcm.std_msgs.Time import Time

    sec = int(ts); nsec = int((ts - sec) * 1e9)

    def stamped(parent, child):
        s = TransformStamped()
        s.header = Header(); s.header.stamp = Time()
        s.header.stamp.sec = sec; s.header.stamp.nsec = nsec
        s.header.frame_id = parent; s.child_frame_id = child
        s.transform = LCMT()
        s.transform.translation = LV3()
        s.transform.translation.x = 0; s.transform.translation.y = 0; s.transform.translation.z = 0
        s.transform.rotation = LQ()
        s.transform.rotation.x = 0; s.transform.rotation.y = 0
        s.transform.rotation.z = 0; s.transform.rotation.w = 1
        return s

    msg = TFMessage()
    msg.transforms = [stamped("world", "camera_optical")]
    msg.transforms_length = 1
    return msg


def main():
    if not MAP_PLY.exists():
        sys.exit(f"missing map: {MAP_PLY}")
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image, PointCloud2, CameraInfo
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage

    # read video dimensions before computing intrinsics
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"cv2 could not open {VIDEO}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or PUBLISH_FPS
    print(f"video: {VIDEO.name}  {W}x{H}  {total_frames} frames  {native_fps:.1f} fps")

    FX = FY = (W / 2) / np.tan(np.deg2rad(HFOV_DEG / 2))
    CX, CY = W / 2, H / 2
    print(f"intrinsics: fx={FX:.1f}  cx={CX:.1f}  cy={CY:.1f}  (HFOV={HFOV_DEG}°)")

    cam_info = CameraInfo(
        frame_id="camera_optical", height=H, width=W,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[FX, 0, CX, 0, FY, CY, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[FX, 0, CX, 0, 0, FY, CY, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )

    map_arr = load_map_pcd()

    print("warming YOLO (cpu)...")
    detector = Yolo2DDetector(device="cpu")

    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    lidar_topic = LCMTransport("/lidar", PointCloud2)
    tf_topic = LCMTransport("/tf", TFMessage)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)

    # Downsample for display + YOLO — keeps annotations aligned and CPU manageable
    DISPLAY_W = 1280
    scale = DISPLAY_W / W
    DW, DH = DISPLAY_W, int(H * scale)
    DFX = FX * scale; DFY = FY * scale
    DCX = CX * scale; DCY = CY * scale
    display_cam_info = CameraInfo(
        frame_id="camera_optical", height=DH, width=DW,
        distortion_model="plumb_bob", D=[0.0] * 5,
        K=[DFX, 0, DCX, 0, DFY, DCY, 0, 0, 1],
        R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        P=[DFX, 0, DCX, 0, 0, DFY, DCY, 0, 0, 0, 1, 0],
        binning_x=0, binning_y=0,
    )
    print(f"display size: {DW}x{DH} (scale={scale:.2f})")

    # build the map PointCloud2 once — just update .ts each frame
    map_pcd = PointCloud2.from_numpy(map_arr, frame_id="world", timestamp=time.time())

    period = 1.0 / PUBLISH_FPS
    n = 0
    loop_num = 0

    print(f"\npublishing at {PUBLISH_FPS} fps (looping). Open Foxglove (ws://localhost:8765)")
    print("  3D panel → Topics tab → enable /lidar")
    print("  Image panel: /color_image with /annotations overlay")
    print("Ctrl+C to exit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_num += 1
                print(f"  --- loop {loop_num} ---")
                continue

            t0 = time.perf_counter()
            ts = time.time()

            # ensure BGR (video may return grayscale single-channel)
            if frame_bgr.ndim == 2:
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            elif frame_bgr.shape[2] == 1:
                frame_bgr = cv2.cvtColor(frame_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)

            # resize once — same frame for publish and YOLO so annotations align
            small = cv2.resize(frame_bgr, (DW, DH))
            img = Image.from_numpy(small, frame_id="camera_optical")
            img.ts = ts

            dets = detector.process_image(img)

            img_topic.publish(img)
            cam_info_topic.publish(display_cam_info)
            ann_topic.publish(dets.to_foxglove_annotations())
            tf_topic.publish(make_static_tf(ts))

            # static map is large — no need to spam it every frame
            if n % LIDAR_EVERY_N == 0:
                map_pcd.ts = ts
                lidar_topic.publish(map_pcd)

            n += 1
            if n == 1 or n % PUBLISH_FPS == 0:
                names = ", ".join(sorted({d.name for d in dets.detections})) or "(none)"
                elapsed = time.perf_counter() - t0
                print(f"  frame {n}: {len(dets.detections)} dets [{names}]  ({elapsed*1000:.0f}ms/frame)")

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")
    finally:
        cap.release()
        detector.stop()
        for t in (img_topic, ann_topic, lidar_topic, tf_topic, cam_info_topic):
            t.lcm.stop()



if __name__ == "__main__":
    main()
