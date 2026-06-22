"""Stream video + YOLO detections into LCM so Foxglove Studio can render them.

Pair with the bridge in another terminal:
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

Then open Foxglove Studio, connect to ws://localhost:8765,
and add an Image panel on /color_image with annotations overlay /annotations.
"""

import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent
VIDEO = REPO / "xr-nav" / "awesome-depth-anything-3" / "assets" / "examples" / "robot_unitree.mp4"
FPS = 10


def main():
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    from dimos_lcm.foxglove_msgs.ImageAnnotations import ImageAnnotations
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector

    print("loading detector...")
    detector = Yolo2DDetector(device="mps")  # change to "cpu" if MPS unavailable

    img_topic = LCMTransport("/color_image", Image)
    ann_topic = LCMTransport("/annotations", ImageAnnotations)
    print("LCM topics: /color_image (sensor_msgs/Image), /annotations (foxglove_msgs/ImageAnnotations)")
    print(f"streaming {VIDEO.name} at {FPS} fps — Ctrl+C to stop")

    cap = cv2.VideoCapture(str(VIDEO))
    period = 1.0 / FPS
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                continue
            t0 = time.perf_counter()

            img = Image.from_numpy(frame, frame_id="camera_optical")
            img.ts = time.time()
            dets = detector.process_image(img)

            img_topic.publish(img)
            if hasattr(dets, "to_foxglove_annotations"):  # Foxglove annotations removed upstream (PR #2122)
                ann_topic.publish(dets.to_foxglove_annotations())

            n += 1
            if n % FPS == 0:
                names = ", ".join(d.name for d in dets.detections[:5]) or "(none)"
                print(f"  frame {n}: {len(dets.detections)} detections [{names}]")

            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        cap.release()
        detector.stop()
        img_topic.lcm.stop()
        ann_topic.lcm.stop()


if __name__ == "__main__":
    main()
