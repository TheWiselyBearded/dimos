"""Mac perception smoke-test for dimos.

Runs Yolo2DDetector on a single image using:
  1. Default device selection (should fall back to CPU on Mac)
  2. Forced MPS (Apple Silicon GPU) via direct device override

Reports detection count, top-5 detections, and per-device latency.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
SCREENCAP = REPO / "xr-nav" / "ScreenCapture_2026-04-11-23-28-16.png"
ROBOT_MP4 = REPO / "xr-nav" / "awesome-depth-anything-3" / "assets" / "examples" / "robot_unitree.mp4"


def load_image(path: Path):
    if path.suffix.lower() in {".mp4", ".mov", ".avi"}:
        cap = cv2.VideoCapture(str(path))
        # seek mid-video for a more interesting frame
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, img = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"failed to read frame from {path}")
    else:
        img = cv2.imread(str(path))
        if img is None:
            sys.exit(f"failed to load image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    print(f"source: {path.name}  shape={img.shape}")
    return img


def run(detector_label: str, device: str | None, image_np):
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector

    print(f"\n=== {detector_label} (device={device or 'auto'}) ===")
    t0 = time.perf_counter()
    detector = Yolo2DDetector(device=device)
    t_init = time.perf_counter() - t0
    print(f"init: {t_init*1000:.1f} ms  (detector.device={detector.device})")

    image = Image.from_numpy(image_np)

    # warmup
    detector.process_image(image)

    runs = 3
    t0 = time.perf_counter()
    for _ in range(runs):
        result = detector.process_image(image)
    elapsed = (time.perf_counter() - t0) / runs
    print(f"infer: {elapsed*1000:.1f} ms/frame avg over {runs} runs")
    print(f"detections: {len(result.detections)}")
    for d in result.detections[:5]:
        x1, y1, x2, y2 = d.bbox
        print(
            f"  {d.name:>15s}  conf={d.confidence:.2f}  "
            f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  track_id={d.track_id}"
        )
    detector.stop()
    return result


def annotate(img, detections, out_path: Path):
    annotated = img.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{d.name} {d.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), annotated)
    print(f"  annotated -> {out_path}")


def main():
    print(f"python: {sys.version.split()[0]}")
    import torch
    print(f"torch: {torch.__version__}  mps_available={torch.backends.mps.is_available()}")

    sources = [SCREENCAP]
    if ROBOT_MP4.exists():
        sources.append(ROBOT_MP4)

    for src in sources:
        print(f"\n##### SOURCE: {src.name} #####")
        img = load_image(src)
        cpu_result = run("CPU run", "cpu", img)
        if torch.backends.mps.is_available():
            run("MPS (Apple GPU) run", "mps", img)
        annotate(img, cpu_result.detections, REPO / f"mac_perception_demo_{src.stem}.png")


if __name__ == "__main__":
    main()
