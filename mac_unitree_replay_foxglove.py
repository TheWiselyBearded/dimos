"""Replay the unitree_go2_lidar_corrected dataset into Foxglove.

Matches the README's perception/spatial-memory screenshot:
  /color_image    -> RGB camera frame
  /annotations    -> 2D YOLO bbox overlay on the image
  /lidar          -> raw point cloud
  /scene_update   -> 3D detection cubes (lidar-projected from 2D dets)
  /tf             -> world -> base_link -> camera_link -> camera_optical chain

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Then in Foxglove (ws://localhost:8765):
  - Image panel on /color_image (with /annotations overlay)
  - 3D panel: display frame = world; enable /lidar + /scene_update + /tf
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATASET = "unitree_go2_lidar_corrected"
PUBLISH_FPS = 5  # how fast to step through dataset


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
    print(f"loading dataset '{DATASET}'...")
    from dimos.utils.testing import TimedSensorReplay
    from dimos.robot.unitree.type.odometry import Odometry
    from dimos.protocol.tf import TF
    from dimos.msgs.geometry_msgs import Transform, Vector3, Quaternion
    from dimos.msgs.sensor_msgs import CameraInfo

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
    from dimos.msgs.sensor_msgs import Image, PointCloud2
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
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)
    cam_info_topic = LCMTransport("/camera_info", CameraInfo)

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
                    print(f"  [{n_pub}] dt={lidar_ts-base_ts:5.1f}s  "
                          f"2d={len(dets2d.detections)} ({names})  3d={n3d}  pcd={len(lidar_frame)} pts")

                time.sleep(max(0.0, period - (time.perf_counter() - loop_t0)))

            print(f"  loop {loop_num} done: {n_pub} frames, {n_3d} with 3D detections")
    except KeyboardInterrupt:
        pass
    finally:
        det2d.detector.stop()
        for t in (img_topic, lidar_topic, ann_topic, scene_topic, tf_topic, cam_info_topic):
            t.lcm.stop()
        tf.stop()


if __name__ == "__main__":
    main()
