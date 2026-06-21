"""Replay the unitree_go2_lidar_corrected dataset into Foxglove.

Matches the README's perception/spatial-memory screenshot:
  /color_image        -> RGB camera frame
  /annotations        -> 2D YOLO bbox overlay on the image
  /lidar              -> per-frame point cloud, colorized from the RGB image
  /accumulated_cloud  -> world-frame cloud that retains every scan ever seen
  /scene_update       -> 3D detection cubes (lidar-projected from 2D dets)
  /tf                 -> world -> base_link -> camera_link -> camera_optical chain

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Then in Foxglove (ws://localhost:8765):
  - Image panel on /color_image (with /annotations overlay)
  - 3D panel: display frame = world; enable /lidar + /accumulated_cloud
    + /scene_update + /tf. For /lidar and /accumulated_cloud, set the panel's
    Color mode = "BGRA (packed)" (or "BGR (packed)") with Color field = "rgb"
    to see the camera image painted onto the lidar surfaces. The encoder is
    monkey-patched to set alpha=0xFF so the points actually render.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATASET = "unitree_go2_lidar_corrected"
PUBLISH_FPS = 5  # how fast to step through dataset


def _patch_pointcloud2_rgb_alpha():
    """Force alpha=0xFF in the packed-RGB encoding.

    dimos's PointCloud2.lcm_encode packs colors as 0x00RRGGBB (alpha byte = 0).
    Foxglove's BGR/BGRA (packed) color modes read byte 3 as alpha — alpha=0
    makes every point fully transparent, so Foxglove falls back to a flat gray
    cloud regardless of the per-point RGB. Re-pack with alpha=0xFF.
    """
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2 as _PC2
    from dimos_lcm.sensor_msgs.PointCloud2 import PointCloud2 as _LCMPC2
    from dimos_lcm.std_msgs.Header import Header as _Header

    if getattr(_PC2, "_rgb_alpha_patched", False):
        return
    _orig = _PC2.lcm_encode

    def lcm_encode(self, frame_id=None):
        self._ensure_tensor_initialized()
        if "colors" not in self._pcd_tensor.point:
            return _orig(self, frame_id)
        points, _ = self.as_numpy()
        if len(points) == 0:
            return _orig(self, frame_id)
        colors = self._pcd_tensor.point["colors"].numpy()
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        rgb_uint32 = (
            (np.uint32(0xFF) << 24)  # alpha = 255 so Foxglove BGRA(packed) renders opaque
            | (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )
        rgb_packed = rgb_uint32.view(np.float32)
        msg = _LCMPC2()
        msg.header = _Header()
        msg.header.seq = 0
        msg.header.frame_id = frame_id or self.frame_id
        msg.header.stamp.sec = int(self.ts)
        msg.header.stamp.nsec = int((self.ts - int(self.ts)) * 1e9)
        msg.height = 1
        msg.width = len(points)
        msg.fields = self._create_xyzrgb_fields()
        msg.fields_length = 4
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        point_data = np.column_stack([points, rgb_packed]).astype(np.float32)
        data_bytes = point_data.tobytes()
        msg.data = data_bytes
        msg.data_length = len(data_bytes)
        msg.is_dense = True
        msg.is_bigendian = False
        return msg.lcm_encode()

    _PC2.lcm_encode = lcm_encode
    _PC2._rgb_alpha_patched = True

# Persistent accumulated-cloud bounds. Voxel size dedupes; max_points is a
# hard backstop so the PointCloud2 stays transmittable.
ACC_CLOUD_VOXEL_M = 0.05
ACC_CLOUD_MAX_POINTS = 400_000


def colorize_lidar(lidar_pts, img_rgb, T_cam_lidar, K, max_range=30.0):
    """Sample per-point RGB by projecting lidar into the camera image.

    Points behind the camera, outside the image, or beyond max_range get a
    neutral gray. Returns Nx3 float in [0, 1].
    """
    pts = np.asarray(lidar_pts, np.float32)
    if len(pts) == 0:
        return np.empty((0, 3), np.float32)
    M = np.asarray(T_cam_lidar.to_matrix(), np.float32)
    cam = (pts @ M[:3, :3].T) + M[:3, 3]
    z = cam[:, 2]
    in_front = (z > 0.05) & (z < max_range)
    safe_z = np.where(in_front, z, 1.0)
    u = (K[0] * cam[:, 0] + K[2] * safe_z) / safe_z
    v = (K[4] * cam[:, 1] + K[5] * safe_z) / safe_z
    H, W = img_rgb.shape[:2]
    ui = u.astype(np.int32)
    vi = v.astype(np.int32)
    in_img = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    colors = np.full((len(pts), 3), 0.4, np.float32)
    if in_img.any():
        colors[in_img] = img_rgb[vi[in_img], ui[in_img]].astype(np.float32) / 255.0
    return colors


def to_tfmessage(transforms, ts):
    """Convert a list of dimos.msgs.geometry_msgs.Transform -> tf2_msgs.TFMessage."""
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from dimos_lcm.geometry_msgs.TransformStamped import TransformStamped
    from dimos_lcm.geometry_msgs.Transform import Transform as LCMTransform
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion as LQ
    from dimos_lcm.std_msgs.Header import Header
    from dimos_lcm.std_msgs.Time import Time

    def stamp_for(t):
        sec = int(t); nsec = int((t - sec) * 1e9)
        out = Time(); out.sec = sec; out.nsec = nsec
        return out

    out_transforms = []
    for t in transforms:
        ts_msg = TransformStamped()
        ts_msg.header = Header(); ts_msg.header.stamp = stamp_for(t.ts or ts)
        ts_msg.header.frame_id = t.frame_id
        ts_msg.child_frame_id = t.child_frame_id
        tx = LCMTransform()
        tx.translation = LV3()
        tx.translation.x = t.translation.x; tx.translation.y = t.translation.y; tx.translation.z = t.translation.z
        tx.rotation = LQ()
        tx.rotation.x = t.rotation.x; tx.rotation.y = t.rotation.y
        tx.rotation.z = t.rotation.z; tx.rotation.w = t.rotation.w
        ts_msg.transform = tx
        out_transforms.append(ts_msg)
    msg = TFMessage()
    msg.transforms = out_transforms; msg.transforms_length = len(out_transforms)
    return msg


def main():
    _patch_pointcloud2_rgb_alpha()
    print(f"loading dataset '{DATASET}'...")
    from dimos.utils.testing.replay import TimedSensorReplay
    from dimos.robot.unitree.type.odometry import Odometry
    from dimos.protocol.tf.tf import TF
    from dimos.msgs.geometry_msgs.Transform import Transform
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo

    # Inlined from dimos.robot.unitree.go2.connection — that module pulls in
    # unitree_webrtc_connect which isn't needed for replay.
    def _camera_info_static():
        fx, fy, cx, cy = (819.553492, 820.646595, 625.284099, 336.808987)
        return CameraInfo(
            frame_id="camera_optical", height=720, width=1280,
            distortion_model="plumb_bob", D=[0.0]*5,
            K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
            R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
            binning_x=0, binning_y=0,
        )

    def _odom_to_tf(odom):
        return [
            Transform.from_pose("base_link", odom),
            Transform(translation=Vector3(0.3, 0.0, 0.0),
                      rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
                      frame_id="base_link", child_frame_id="camera_link", ts=odom.ts),
            Transform(translation=Vector3(0.0, 0.0, 0.0),
                      rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
                      frame_id="camera_link", child_frame_id="camera_optical", ts=odom.ts),
        ]
    from dimos.perception.detection.module2D import Detection2DModule
    from dimos.perception.detection.module3D import Detection3DModule
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage

    # data stores
    lidar_store = TimedSensorReplay(f"{DATASET}/lidar")
    video_store = TimedSensorReplay(f"{DATASET}/video")
    odom_store = TimedSensorReplay(f"{DATASET}/odom", autocast=Odometry.from_msg)

    # camera intrinsics + tf chain
    camera_info = _camera_info_static()
    tf = TF()

    # detectors
    print("warming detectors...")
    det2d = Detection2DModule(detector=lambda: Yolo2DDetector(device="cpu"))
    det3d = Detection3DModule(camera_info=camera_info)

    # LCM topics
    img_topic = LCMTransport("/color_image", Image)
    lidar_topic = LCMTransport("/lidar", PointCloud2)
    acc_cloud_topic = LCMTransport("/accumulated_cloud", PointCloud2)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)

    # Reuse the iphone spatial pipeline's accumulator (voxel-deduped, point-capped).
    sys.path.insert(0, str(REPO))
    from mac_iphone_spatial_foxglove import AccumulatedCloud
    import open3d as o3d
    acc_cloud = AccumulatedCloud(voxel=ACC_CLOUD_VOXEL_M, max_points=ACC_CLOUD_MAX_POINTS)

    print(f"streaming dataset on loop at {PUBLISH_FPS} fps — Ctrl+C to stop")
    period = 1.0 / PUBLISH_FPS
    loop_num = 0

    try:
        while True:
            n_pub = 0
            n_3d = 0
            base_ts = None
            loop_num += 1
            print(f"\n--- loop {loop_num} ---")

            for lidar_ts, lidar_frame in lidar_store.iterate_ts():
                loop_t0 = time.perf_counter()

                video_frame = video_store.find_closest(lidar_ts)
                odom_frame = odom_store.find_closest(lidar_ts)
                if video_frame is None or odom_frame is None:
                    continue

                video_frame.frame_id = "camera_optical"

                transforms = _odom_to_tf(odom_frame)
                tf.receive_transform(*transforms)

                dets2d = det2d.process_image_frame(video_frame)

                camera_transform = tf.get("camera_optical", lidar_frame.frame_id)
                dets3d = None
                if camera_transform is not None:
                    try:
                        dets3d = det3d.process_frame(dets2d, lidar_frame, camera_transform)
                    except Exception as e:
                        print(f"  [3D failed: {e}]")

                # Colorize the lidar scan from the RGB frame, and aggregate the
                # world-frame points into the persistent /accumulated_cloud.
                published_acc = False
                if camera_transform is not None:
                    lidar_pts = np.asarray(lidar_frame.pointcloud.points, np.float32)
                    if len(lidar_pts):
                        img_rgb = video_frame.as_numpy()
                        colors = colorize_lidar(lidar_pts, img_rgb, camera_transform,
                                                camera_info.K)
                        pcd_c = o3d.geometry.PointCloud()
                        pcd_c.points = o3d.utility.Vector3dVector(lidar_pts.astype(np.float64))
                        pcd_c.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                        lidar_frame = PointCloud2(pointcloud=pcd_c,
                                                  frame_id=lidar_frame.frame_id,
                                                  ts=lidar_ts)
                        world_pts = None
                        if lidar_frame.frame_id == "world":
                            world_pts = lidar_pts
                        else:
                            T_world_lidar = tf.get("world", lidar_frame.frame_id)
                            if T_world_lidar is not None:
                                M = np.asarray(T_world_lidar.to_matrix(), np.float32)
                                world_pts = (lidar_pts @ M[:3, :3].T) + M[:3, 3]
                        if world_pts is not None:
                            acc_cloud.add(world_pts, colors=colors)
                            acc_cloud_topic.publish(acc_cloud.to_msg(lidar_ts))
                            published_acc = True

                img_topic.publish(video_frame)
                cam_info_topic.publish(camera_info)
                lidar_topic.publish(lidar_frame)
                ann_topic.publish(dets2d.to_foxglove_annotations())
                if dets3d is not None and len(dets3d.detections):
                    scene_topic.publish(dets3d.to_foxglove_scene_update())
                    n_3d += 1
                tf_topic.publish(to_tfmessage(transforms, ts=lidar_ts))

                n_pub += 1
                if base_ts is None:
                    base_ts = lidar_ts
                if n_pub == 1 or n_pub % 5 == 0:
                    names = ", ".join(sorted({d.name for d in dets2d.detections})) or "(none)"
                    n3d = len(dets3d.detections) if dets3d is not None else 0
                    acc_tag = f" acc={acc_cloud.size}" if published_acc else ""
                    print(f"  [{n_pub}] dt={lidar_ts-base_ts:5.1f}s  "
                          f"2d={len(dets2d.detections)} ({names})  3d={n3d}  "
                          f"pcd={len(lidar_frame)} pts{acc_tag}")

                time.sleep(max(0.0, period - (time.perf_counter() - loop_t0)))

            print(f"  loop {loop_num} done: {n_pub} frames, {n_3d} with 3D detections")
    except KeyboardInterrupt:
        pass
    finally:
        det2d.detector.stop()
        for t in (img_topic, lidar_topic, acc_cloud_topic, ann_topic, scene_topic,
                  tf_topic, cam_info_topic):
            t.lcm.stop()
        tf.stop()


if __name__ == "__main__":
    main()
