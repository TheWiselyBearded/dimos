"""Visualize dimos spatial memory in Foxglove.

Publishes:
  /color_image     -> current video frame (Image)
  /scene_update    -> trajectory of stored frames as spheres+labels (SceneUpdate)
  /tf              -> camera_optical at the current robot pose (FrameTransforms)
  /query_overlay   -> top-K text-query hits, in red, after streaming finishes

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Then in Foxglove:
  - Image panel on /color_image
  - 3D panel: enable /scene_update + /query_overlay; "world" frame as display frame
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

REPO = Path(__file__).resolve().parent
VIDEO = REPO / "xr-nav" / "awesome-depth-anything-3" / "assets" / "examples" / "robot_unitree.mp4"

NUM_FRAMES = 40
PUBLISH_FPS = 5  # how fast to push to LCM (lets user watch the trajectory grow)


def make_sphere_entity(eid, x, y, z, label, *, color, radius=0.25, ts=None):
    """Build a SceneEntity = one sphere + one text label, anchored in 'world'."""
    from dimos_lcm.foxglove_msgs.SceneEntity import SceneEntity
    from dimos_lcm.foxglove_msgs.SpherePrimitive import SpherePrimitive
    from dimos_lcm.foxglove_msgs.TextPrimitive import TextPrimitive
    from dimos_lcm.foxglove_msgs.Color import Color
    from dimos_lcm.geometry_msgs.Pose import Pose
    from dimos_lcm.geometry_msgs.Point import Point
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.builtin_interfaces.Duration import Duration
    from dimos.types.timestamped import to_ros_stamp

    def make_pose(px, py, pz):
        p = Pose()
        p.position = Point(); p.position.x = px; p.position.y = py; p.position.z = pz
        p.orientation = Quaternion(); p.orientation.x = 0; p.orientation.y = 0
        p.orientation.z = 0; p.orientation.w = 1
        return p

    sph = SpherePrimitive()
    sph.pose = make_pose(x, y, z)
    sph.size = LV3(); sph.size.x = sph.size.y = sph.size.z = radius
    sph.color = Color(); sph.color.r, sph.color.g, sph.color.b, sph.color.a = color

    txt = TextPrimitive()
    txt.pose = make_pose(x, y, z + radius + 0.1)
    txt.billboard = True
    txt.font_size = 14.0
    txt.scale_invariant = True
    txt.color = Color(); txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
    txt.text = label

    e = SceneEntity()
    e.timestamp = to_ros_stamp(ts if ts is not None else time.time())
    e.frame_id = "world"
    e.id = eid
    e.lifetime = Duration(); e.lifetime.sec = 0; e.lifetime.nanosec = 0
    e.frame_locked = False
    for arr in ("metadata", "arrows", "cubes", "spheres", "cylinders",
                "lines", "triangles", "texts", "models"):
        setattr(e, arr, [])
        setattr(e, arr + "_length", 0)
    e.spheres = [sph]; e.spheres_length = 1
    e.texts = [txt]; e.texts_length = 1
    return e


def make_scene_update(entities):
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    su = SceneUpdate()
    su.deletions = []; su.deletions_length = 0
    su.entities = list(entities); su.entities_length = len(su.entities)
    return su


def make_world_tf(pose_x=0.0, pose_y=0.0, pose_z=0.0):
    """tf2_msgs.TFMessage is in the bridge's HARDCODED_SCHEMAS,
    so Foxglove can resolve it. foxglove_msgs.FrameTransforms is not."""
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage
    from dimos_lcm.geometry_msgs.TransformStamped import TransformStamped
    from dimos_lcm.geometry_msgs.Transform import Transform
    from dimos_lcm.geometry_msgs.Vector3 import Vector3 as LV3
    from dimos_lcm.geometry_msgs.Quaternion import Quaternion
    from dimos_lcm.std_msgs.Header import Header
    from dimos_lcm.std_msgs.Time import Time

    now = time.time()
    sec = int(now); nsec = int((now - sec) * 1e9)

    def make_ts(parent, child, x, y, z):
        ts = TransformStamped()
        ts.header = Header()
        ts.header.stamp = Time(); ts.header.stamp.sec = sec; ts.header.stamp.nsec = nsec
        ts.header.frame_id = parent
        ts.child_frame_id = child
        ts.transform = Transform()
        ts.transform.translation = LV3()
        ts.transform.translation.x = x; ts.transform.translation.y = y; ts.transform.translation.z = z
        ts.transform.rotation = Quaternion()
        ts.transform.rotation.x = 0; ts.transform.rotation.y = 0
        ts.transform.rotation.z = 0; ts.transform.rotation.w = 1
        return ts

    msg = TFMessage()
    msg.transforms = [
        make_ts("world", "camera_optical", pose_x, pose_y, pose_z),
    ]
    msg.transforms_length = 1
    return msg


def main():
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    import tempfile
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs import Image
    from dimos.perception.spatial_perception import SpatialMemory
    from dimos_lcm.foxglove_msgs.SceneUpdate import SceneUpdate
    from dimos_lcm.tf2_msgs.TFMessage import TFMessage

    workdir = Path(tempfile.mkdtemp(prefix="dimos_spatmem_fox_"))
    print(f"workdir: {workdir}")

    print("loading SpatialMemory (CLIP/CoreML)... this takes ~7s the first time")
    memory = SpatialMemory(
        collection_name="mac_demo_fox",
        embedding_model="clip",
        new_memory=True,
        db_path=str(workdir / "chroma_db"),
        visual_memory_path=str(workdir / "visual_memory.pkl"),
        output_dir=str(workdir / "images"),
        min_distance_threshold=0.01,
        min_time_threshold=0.01,
    )

    img_topic = LCMTransport("/color_image", Image)
    scene_topic = LCMTransport("/scene_update", SceneUpdate)
    overlay_topic = LCMTransport("/query_overlay", SceneUpdate)
    tf_topic = LCMTransport("/tf", TFMessage)

    cap = cv2.VideoCapture(str(VIDEO))
    period = 1.0 / PUBLISH_FPS
    stored = []  # list of (frame_id, x, y, z, frame)

    print(f"streaming {NUM_FRAMES} frames at {PUBLISH_FPS} fps -> Foxglove")
    for n in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = cap.read()
        t0 = time.perf_counter()

        x = n * 0.5; y = n * 0.3; z = 0.0
        pos = SimpleNamespace(x=x, y=y, z=z)
        rot = SimpleNamespace(x=0.0, y=0.0, z=0.0)

        # embed + store via the same pipeline the offline demo used
        emb = memory.embedding_provider.get_embedding(frame)
        from datetime import datetime
        import uuid
        frame_id = f"frame_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        memory.vector_db.add_image_vector(
            vector_id=frame_id, image=frame, embedding=emb,
            metadata={"pos_x": x, "pos_y": y, "pos_z": z,
                      "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
                      "timestamp": time.time(), "frame_id": frame_id},
        )
        stored.append((frame_id, x, y, z, frame))

        # publish image + growing trajectory
        img = Image.from_numpy(frame, frame_id="camera_optical"); img.ts = time.time()
        img_topic.publish(img)

        entities = [
            make_sphere_entity(fid, fx, fy, fz, fid[-6:],
                               color=(0.2, 0.8, 1.0, 0.9), radius=0.20)
            for (fid, fx, fy, fz, _) in stored
        ]
        # current frame in green, larger
        entities.append(
            make_sphere_entity("current", x, y, z, "now",
                               color=(0.2, 1.0, 0.2, 1.0), radius=0.35)
        )
        scene_topic.publish(make_scene_update(entities))
        tf_topic.publish(make_world_tf(pose_x=x, pose_y=y, pose_z=z))

        print(f"  {n+1}/{NUM_FRAMES}  pos=({x:.1f},{y:.1f})  id={frame_id}")
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # query phase: highlight top hits per text query
    print("\nrunning text queries; top hits will appear as RED spheres in /query_overlay")
    queries = ["a humanoid robot", "a kitchen", "a person walking"]
    pos_lookup = {fid: (fx, fy, fz) for (fid, fx, fy, fz, _) in stored}

    for q in queries:
        results = memory.query_by_text(q, limit=5)
        # chroma may nest these
        ids = results[0].get("id") if results and isinstance(results[0].get("id"), list) else [r.get("id") for r in results]
        ids = ids if isinstance(ids, list) else [ids]
        ids = [i for i in ids if i in pos_lookup][:5]
        if not ids:
            print(f"  '{q}' -> no hits"); continue

        entities = []
        for rank, fid in enumerate(ids):
            fx, fy, fz = pos_lookup[fid]
            entities.append(make_sphere_entity(
                f"hit_{q}_{rank}", fx, fy, fz + 0.6,
                f"#{rank+1} {q}", color=(1.0, 0.1, 0.1, 0.9), radius=0.30))
        overlay_topic.publish(make_scene_update(entities))
        print(f"  '{q}' -> top {len(ids)} hits highlighted in /query_overlay")
        time.sleep(2.5)  # leave each overlay up so you can read it

    print("\nDone. Ctrl+C to exit (data will keep being visible in Foxglove until bridge stops).")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); memory.stop()
        for t in (img_topic, scene_topic, overlay_topic, tf_topic):
            t.lcm.stop()


if __name__ == "__main__":
    main()
