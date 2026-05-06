"""Mac spatial-memory smoke-test for dimos.

- Streams frames from a video.
- Feeds each frame + a synthetic 3D position to dimos.SpatialMemory.
- Embeds with CLIP via ONNX Runtime (uses CoreML provider on macOS).
- Queries the resulting vector DB by text, reports similarity scores.
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from reactivex import operators as ops
from reactivex.scheduler import ThreadPoolScheduler

REPO = Path(__file__).resolve().parent
VIDEO = REPO / "xr-nav" / "awesome-depth-anything-3" / "assets" / "examples" / "robot_unitree.mp4"


def main():
    print(f"python: {sys.version.split()[0]}")
    print(f"video: {VIDEO}")
    if not VIDEO.exists():
        sys.exit(f"missing video: {VIDEO}")

    from dimos.perception.spatial_perception import SpatialMemory
    from dimos.stream.video_provider import VideoProvider

    workdir = Path(tempfile.mkdtemp(prefix="dimos_spatmem_"))
    print(f"workdir: {workdir}")

    t_init0 = time.perf_counter()
    memory = SpatialMemory(
        collection_name="mac_demo",
        embedding_model="clip",
        new_memory=True,
        db_path=str(workdir / "chroma_db"),
        visual_memory_path=str(workdir / "visual_memory.pkl"),
        output_dir=str(workdir / "images"),
        min_distance_threshold=0.01,
        min_time_threshold=0.01,
    )
    t_init = time.perf_counter() - t_init0
    print(f"SpatialMemory init: {t_init*1000:.0f} ms")
    print(f"ONNX providers: {memory.embedding_provider.model.get_providers()}")

    pool = ThreadPoolScheduler(max_workers=4)
    provider = VideoProvider(dev_name="demo", video_source=str(VIDEO), pool_scheduler=pool)
    stream = provider.capture_video_as_observable(realtime=False, fps=10)

    frame_counter = {"n": 0}

    def to_payload(frame):
        n = frame_counter["n"]
        # synthetic trajectory (0.5 m steps along diagonal, no rotation change)
        pos = SimpleNamespace(x=n * 0.5, y=n * 0.5, z=0.0)
        rot = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        frame_counter["n"] += 1
        return {"frame": frame, "position": pos, "rotation": rot}

    spatial_stream = memory.process_stream(stream.pipe(ops.map(to_payload)))

    target = 30  # process up to 30 frames
    processed = []
    embed_times = []

    last_emb_t0 = {"t": time.perf_counter()}

    def on_next(result):
        if not result:
            return
        now = time.perf_counter()
        embed_times.append(now - last_emb_t0["t"])
        last_emb_t0["t"] = now
        processed.append(result)
        if len(processed) >= target:
            sub.dispose()

    def on_error(e):
        print(f"!! stream error: {e}")

    print(f"\nstreaming up to {target} frames...")
    t0 = time.perf_counter()
    sub = spatial_stream.subscribe(on_next=on_next, on_error=on_error)

    timeout = 60.0
    while len(processed) < target and time.perf_counter() - t0 < timeout:
        time.sleep(0.25)
    sub.dispose()
    elapsed = time.perf_counter() - t0
    print(f"processed {len(processed)} frames in {elapsed:.1f}s")
    if embed_times:
        # drop first (warmup)
        warm = embed_times[1:] or embed_times
        print(f"embed time/frame (excl. warmup): mean={1000*np.mean(warm):.0f} ms  "
              f"min={1000*np.min(warm):.0f}  max={1000*np.max(warm):.0f}")

    def fmt_results(results):
        # chroma returns nested lists; flatten metadata + distance pairs
        out = []
        for r in results:
            md = r.get("metadata", {})
            dist = r.get("distance", 1.0)
            md_list = md if isinstance(md, list) else [md]
            dist_list = dist if isinstance(dist, list) else [dist]
            for m, d in zip(md_list, dist_list):
                if isinstance(m, dict):
                    out.append((1.0 - d, m))
        return out

    # text queries
    queries = ["a humanoid robot", "a kitchen", "a dog", "a car on a road"]
    print("\n=== text queries (cosine similarity, higher = better match) ===")
    for q in queries:
        results = memory.query_by_text(q, limit=5)
        flat = fmt_results(results)
        print(f"\n  '{q}'  -> {len(flat)} hits")
        for sim, md in flat[:5]:
            print(f"    sim={sim:+.3f}  pos=({md.get('pos_x', 0):.1f},{md.get('pos_y', 0):.1f})  "
                  f"id={md.get('frame_id', '?')[:30]}")

    # image query: query with the first stored frame's image
    if processed:
        cap = cv2.VideoCapture(str(VIDEO))
        ok, qimg = cap.read()
        cap.release()
        if ok:
            print("\n=== image query (first video frame) ===")
            results = memory.query_by_image(qimg, limit=5)
            for sim, md in fmt_results(results)[:5]:
                print(f"  sim={sim:+.3f}  pos=({md.get('pos_x', 0):.1f},{md.get('pos_y', 0):.1f})  "
                      f"id={md.get('frame_id', '?')[:30]}")

    provider.dispose_all()
    pool.executor.shutdown(wait=True)
    memory.stop()
    print(f"\nDONE.  artifacts in {workdir}")


if __name__ == "__main__":
    main()
