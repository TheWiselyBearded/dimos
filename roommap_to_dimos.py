"""Bridge: room-map cloud (Lambda + lingbot-map) -> dimos VoxelMap.pkl + PLY.

Cloud is an opt-in alternative to the local Mac depth pipelines, not a
replacement. Use when you want the best-quality fused point cloud and have
already booted the GPU with `roommap up`.

Flow:
  1. Resolve a video path (--recording-dir/camera.mp4 or --video).
  2. Shell out to `roommap process` (room-map CLI from your room-map venv).
  3. Locate the resulting artifacts/<job_id>/ (map.ply + predictions.npz).
  4. Voxelize map.ply at --voxel-size into a dimos VoxelMap state.
  5. Wrap as a load_map-compatible bundle and pickle alongside the recording.

Result: a .pkl you can drop into any spatial_foxglove script with
``--load-map cloud_fuse.pkl --no-detect`` to view in Foxglove (or to seed an
incremental local mapping session). The raw map.ply is left untouched in the
room-map artifacts dir for direct viewing in MeshLab / Open3D.

Typical use:

  # One-shot: upload + fuse + convert + suggest next command
  python roommap_to_dimos.py --recording-dir /path/to/reachy_trial4

  # Already ran `roommap process` — just convert an existing job
  python roommap_to_dimos.py --skip-upload --job-id 46865af2daf2 \\
      --out /path/to/reachy_trial4/cloud_fuse.pkl

  # View the result with the iPhone script (works for any recording)
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
      /opt/anaconda3/envs/xr-nav/bin/python -u mac_iphone_spatial_foxglove.py \\
      --video /path/to/camera.mp4 --load-map /path/to/cloud_fuse.pkl \\
      --no-detect --pose identity
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "xr-nav" / "src"))

DEFAULT_ROOMMAP_DIR = Path("/Users/reza/Documents/Tools/room-map")
DEFAULT_ARTIFACTS_DIR = DEFAULT_ROOMMAP_DIR / "artifacts"


def resolve_video(args: argparse.Namespace) -> Path:
    if args.video is not None:
        v = args.video.expanduser().resolve()
        if not v.is_file():
            sys.exit(f"--video does not exist: {v}")
        return v
    rec = args.recording_dir.expanduser().resolve()
    if not rec.is_dir():
        sys.exit(f"--recording-dir does not exist: {rec}")
    for name in ("camera.mp4", "scan.mp4", "video.mp4"):
        candidate = rec / name
        if candidate.is_file():
            return candidate
    sys.exit(f"could not find camera.mp4/scan.mp4/video.mp4 in {rec}")


def submit_to_roommap(video: Path, args: argparse.Namespace) -> str:
    """Shell out to `roommap process`. Returns the job_id parsed from stdout."""
    cmd = [
        args.roommap_bin, "process", str(video),
        "--model", args.model,
        "--fps", str(args.fps),
        "--out", str(args.artifacts_dir),
        "--types", "ply,npz",
    ]
    if args.mode:
        cmd += ["--mode", args.mode]
    print(f"[cloud] $ {' '.join(cmd)}\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    job_id: str | None = None
    job_re = re.compile(r"\bJob\s+([0-9a-f]{8,})\b", re.IGNORECASE)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if job_id is None:
            m = job_re.search(line)
            if m:
                job_id = m.group(1)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[cloud] roommap process exited with {proc.returncode}")
    if job_id is None:
        sys.exit("[cloud] could not parse job_id from roommap output")
    return job_id


def load_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (points float32 [N,3] in world, colors float32 [N,3] in [0,1])."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if pts.size == 0:
        sys.exit(f"[convert] PLY is empty: {path}")
    if cols.size == 0:
        print(f"[convert] WARN: PLY has no colors; defaulting to gray")
        cols = np.full_like(pts, 0.5, dtype=np.float32)
    return pts, np.clip(cols, 0.0, 1.0)


def voxelize(
    pts: np.ndarray,
    cols: np.ndarray,
    voxel_size: float,
    max_range: float,
    origin: np.ndarray | None = None,
) -> dict:
    """Bucket points into a voxel grid, returning a VoxelMap.to_state()-shaped dict.

    Range filter applied if ``origin`` is given (otherwise no filtering).
    """
    if origin is not None:
        d = np.linalg.norm(pts - origin[None, :], axis=1)
        keep = d < max_range
        pts = pts[keep]
        cols = cols[keep]

    keys = np.floor(pts / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    n = len(unique_keys)
    centroids = np.zeros((n, 3), dtype=np.float64)
    colors_sum = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(centroids, inverse, pts.astype(np.float64))
    np.add.at(colors_sum, inverse, cols.astype(np.float64))
    np.add.at(counts, inverse, 1)
    centroids /= counts[:, None]
    colors_avg = colors_sum / counts[:, None]

    return {
        "voxel_size": float(voxel_size),
        "max_range": float(max_range),
        "keys": unique_keys.astype(np.int64),
        "centroids": centroids.astype(np.float32),
        "colors": colors_avg.astype(np.float32),
        "confidence": np.ones(n, dtype=np.float32),
        "count": counts.astype(np.int32),
        "miss_count": np.zeros(n, dtype=np.int32),
    }


def empty_object_db_state() -> dict:
    from xr_nav.map_io import NullObjectDB
    return NullObjectDB().to_state()


def write_bundle(out_path: Path, voxel_state: dict, source_info: dict) -> None:
    from xr_nav.map_io import MAP_BUNDLE_VERSION

    bundle = {
        "version": MAP_BUNDLE_VERSION,
        "saved_at": time.time(),
        "voxel_map": voxel_state,
        "object_db": empty_object_db_state(),
        "extra": {"source": "roommap_to_dimos", **source_info},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_vox = len(voxel_state["keys"])
    print(f"[save] wrote {out_path} — {n_vox} voxels")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--recording-dir", type=Path,
                     help="recording directory containing camera.mp4 (Reachy/Viture/iPhone)")
    src.add_argument("--video", type=Path,
                     help="bare .mp4 path (alternative to --recording-dir)")

    parser.add_argument("--out", type=Path, default=None,
                        help="output .pkl path. Default: <recording-dir>/cloud_fuse.pkl "
                             "or ./cloud_fuse_<job_id>.pkl")
    parser.add_argument("--voxel-size", type=float, default=0.10,
                        help="voxel size (m) for pooling cloud points into dimos VoxelMap")
    parser.add_argument("--max-range", type=float, default=6.0,
                        help="point distance cap from origin (m). 0 disables range filtering")
    parser.add_argument("--scale", type=float, default=None,
                        help="multiplier applied to point coords before voxelization. "
                             "lingbot-map outputs scale-arbitrary world coords (often "
                             "sub-cm); set this so the cloud's bbox is ~your room size. "
                             "Default: auto — fit bbox diagonal to --auto-scale-target")
    parser.add_argument("--auto-scale-target", type=float, default=5.0,
                        help="with --scale auto, target bbox diagonal in meters (default 5)")

    cloud = parser.add_argument_group("cloud submission (roommap)")
    cloud.add_argument("--model", default="lingbot",
                       help="roommap model: lingbot|da3-large|da3-giant|da3nested-giant-large")
    cloud.add_argument("--fps", type=int, default=10,
                       help="frame extraction fps for the cloud pipeline")
    cloud.add_argument("--mode", choices=["streaming", "windowed"], default=None,
                       help="lingbot inference mode (default: cloud decides — streaming for short)")
    cloud.add_argument("--roommap-bin", default="roommap",
                       help="roommap CLI executable. Default: 'roommap' on PATH. "
                            "If you have a venv, point at /path/to/room-map/.venv/bin/roommap")
    cloud.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR,
                       help=f"where roommap writes artifacts/<job_id>/ "
                            f"(default: {DEFAULT_ARTIFACTS_DIR})")

    skip = parser.add_argument_group("offline conversion (skip cloud)")
    skip.add_argument("--skip-upload", action="store_true",
                      help="don't call roommap; just convert an existing artifacts/<job-id>/")
    skip.add_argument("--job-id", default=None,
                      help="with --skip-upload: which job_id to convert")

    args = parser.parse_args()

    if not args.skip_upload and args.recording_dir is None and args.video is None:
        sys.exit("provide --recording-dir or --video (or --skip-upload --job-id)")
    if args.skip_upload and not args.job_id:
        sys.exit("--skip-upload requires --job-id")

    if args.skip_upload:
        job_id = args.job_id
    else:
        video = resolve_video(args)
        print(f"[cloud] video: {video}  size={video.stat().st_size / 1e6:.1f} MB")
        job_id = submit_to_roommap(video, args)

    artifact_dir = args.artifacts_dir / job_id
    ply_path = artifact_dir / "map.ply"
    if not ply_path.is_file():
        sys.exit(f"[convert] missing map.ply: {ply_path}")

    print(f"[convert] reading {ply_path} ...")
    pts, cols = load_colored_ply(ply_path)
    bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    print(f"[convert] {len(pts):,} points  bbox={bbox_min.round(4).tolist()} -> "
          f"{bbox_max.round(4).tolist()}  diag={diag:.4g}")

    if args.scale is None:
        if diag < 1e-9:
            sys.exit("[convert] degenerate cloud (diag=0) — cannot auto-scale")
        scale = float(args.auto_scale_target / diag)
        print(f"[convert] auto-scale: {scale:.3g} (target diag {args.auto_scale_target} m)")
    else:
        scale = float(args.scale)
        print(f"[convert] using --scale {scale:.3g}")

    if abs(scale - 1.0) > 1e-9:
        pts = (pts * scale).astype(np.float32)
        bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
        print(f"[convert] post-scale bbox={bbox_min.round(3).tolist()} -> "
              f"{bbox_max.round(3).tolist()}  diag={float(np.linalg.norm(bbox_max-bbox_min)):.3f} m")

    print(f"[convert] voxelizing at {args.voxel_size} m")
    origin = np.zeros(3, dtype=np.float32) if args.max_range > 0 else None
    voxel_state = voxelize(pts, cols, args.voxel_size,
                           args.max_range, origin=origin)

    if args.out is not None:
        out_path = args.out.expanduser().resolve()
    elif args.recording_dir is not None:
        out_path = args.recording_dir.expanduser().resolve() / "cloud_fuse.pkl"
    else:
        out_path = Path.cwd() / f"cloud_fuse_{job_id}.pkl"

    write_bundle(out_path, voxel_state, source_info={
        "job_id": job_id,
        "model": args.model,
        "fps": args.fps,
        "voxel_size": args.voxel_size,
        "scale_applied": scale,
        "ply": str(ply_path),
    })

    # Suggest a launch command. Picks the iPhone script as the generic viewer.
    if args.recording_dir is not None:
        candidate_video = args.recording_dir.expanduser().resolve() / "camera.mp4"
    elif args.video is not None:
        candidate_video = args.video.expanduser().resolve()
    else:
        candidate_video = Path("/path/to/your/camera.mp4")

    print(
        "\n[next] view in Foxglove (start the LCM->ws bridge first, then):\n"
        "  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\\n"
        f"    /opt/anaconda3/envs/xr-nav/bin/python -u mac_iphone_spatial_foxglove.py \\\n"
        f"      --video {candidate_video} \\\n"
        f"      --load-map {out_path} \\\n"
        "      --no-detect --pose identity"
    )


if __name__ == "__main__":
    main()
