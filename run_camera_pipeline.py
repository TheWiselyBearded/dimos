"""Unified entry for the spatial camera pipelines.

Reads a TOML configuration file (default: configs/camera_pipeline.toml), applies
any CLI overrides on top, then dispatches to the appropriate mac_*_foxglove.py
script with the correct flags. Use this in place of remembering which script
takes which flags for which source.

Modes
-----
  video             -> mac_iphone_spatial_foxglove.py
  viture-recording  -> mac_viture_spatial_foxglove.py --source recording
  viture-live       -> mac_viture_spatial_foxglove.py --source live

Examples
--------
  # Use default config (video, depthpro, vo, no save)
  python run_camera_pipeline.py

  # Save the map under the configured output_dir with a timestamped filename
  python run_camera_pipeline.py --save

  # Override mode + clip
  python run_camera_pipeline.py --mode video --video datasets/iphone/foo.mp4

  # Resume from a saved bundle and continue saving on top of it
  python run_camera_pipeline.py --load ~/.dimos/sessions/session_20260506_193045.pkl --save

  # Show the underlying command without running
  python run_camera_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs" / "camera_pipeline.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _expand(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(p).expanduser()


def compute_save_path(cfg: dict[str, Any]) -> Path:
    output_dir = _expand(cfg["map"]["output_dir"]) or Path.home() / ".dimos" / "sessions"
    template = cfg["map"].get("save_filename", "session_{ts}.pkl")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = template.format(ts=ts, session=ts)
    return output_dir / name


def build_command(cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Translate a config dict into (argv, env) for the underlying script."""
    mode = cfg.get("mode", "video")
    python = cfg.get("runtime", {}).get("python") or sys.executable

    if mode == "video":
        script = REPO / "mac_iphone_spatial_foxglove.py"
        argv = [python, "-u", str(script)]
        v = cfg.get("video", {})
        if v.get("path"):
            video_path = _expand(v["path"])
            if not video_path.is_absolute():
                video_path = REPO / video_path
            argv += ["--video", str(video_path)]
        if v.get("hfov_deg") is not None:
            argv += ["--hfov-deg", str(v["hfov_deg"])]
        if v.get("no_loop"):
            argv.append("--no-loop")
        argv += ["--pose", cfg["pose"].get("mode", "vo")]

    elif mode in ("viture-recording", "viture-live"):
        script = REPO / "mac_viture_spatial_foxglove.py"
        source = "recording" if mode == "viture-recording" else "live"
        argv = [python, "-u", str(script), "--source", source]
        vit = cfg.get("viture", {})
        if vit.get("video"):
            argv += ["--video", str(_expand(vit["video"]))]
        if vit.get("right_video"):
            argv += ["--right-video", str(_expand(vit["right_video"]))]
        if vit.get("recording_dir"):
            argv += ["--recording-dir", str(_expand(vit["recording_dir"]))]
        # Note: pose.mode is implicit on the viture script (always uses ARKit
        # poses when present, identity otherwise). The flag isn't exposed there.

    elif mode == "unitree-replay":
        # The unitree replay script takes no CLI args — it just plays back the
        # bundled `unitree_go2_lidar_corrected` dataset at a hardcoded rate.
        # Skip all the depth/pose/detection/map flag construction below.
        script = REPO / "mac_unitree_replay_foxglove.py"
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        return [python, "-u", str(script)], env

    else:
        sys.exit(f"unknown mode: {mode!r}; expected one of: "
                 "video | viture-recording | viture-live | unitree-replay")

    # Depth
    d = cfg.get("depth", {})
    argv += ["--depth", d.get("provider", "depthpro")]
    if d.get("device"):
        argv += ["--device", d["device"]]
    if d.get("provider") == "da3":
        argv += ["--da3-model", d.get("da3_model", "da3metric-large")]
        if mode == "video" and d.get("da3_trust_is_metric"):
            argv.append("--da3-trust-is-metric")

    # Runtime
    rt = cfg.get("runtime", {})
    if rt.get("display_width") is not None:
        argv += ["--display-width", str(rt["display_width"])]
    if rt.get("max_fps") is not None:
        argv += ["--max-fps", str(rt["max_fps"])]

    # Detection (flags below are only valid on the iPhone script today; the
    # viture script doesn't yet expose these knobs as CLI args.)
    det = cfg.get("detection", {})
    if mode == "video":
        if not det.get("enabled", True):
            argv.append("--no-detect")
        if not det.get("class_aware", True):
            argv.append("--objects-disable-class-aware")
        if not det.get("decay", True):
            argv.append("--objects-disable-decay")
        if det.get("distance_threshold") is not None:
            argv += ["--objects-distance-threshold", str(det["distance_threshold"])]
        if det.get("pixel_threshold") is not None:
            argv += ["--objects-pixel-threshold", str(det["pixel_threshold"])]

    # Save / load
    m = cfg.get("map", {})
    load_path = _expand(m.get("load_path"))
    if load_path:
        if mode != "video":
            print("[run_camera_pipeline] WARNING: --load-map is only wired into the iphone script today; "
                  "ignoring for viture modes")
        else:
            argv += ["--load-map", str(load_path)]
    if m.get("save"):
        if mode != "video":
            print("[run_camera_pipeline] WARNING: --save-map is only wired into the iphone script today; "
                  "ignoring for viture modes")
        else:
            save_path = compute_save_path(cfg)
            argv += ["--save-map", str(save_path)]
            if int(m.get("save_every_n_frames", 0)) > 0:
                argv += ["--save-map-every-n", str(int(m["save_every_n_frames"]))]

    # Verbatim passthrough — for flags the entry doesn't first-class. Useful
    # for preset TOMLs that exercise the underlying script's full CLI surface
    # (e.g. Viture noise tunables --objects-process-every-n, --map-publish-every-n,
    # --raycast-every-n, --depth-edge-threshold, --depth-edge-dilate,
    # --voxel-min-observations, --use-depth-confidence, --points-stride).
    #
    # Look for extra_args at the top level OR in any [section] of the config
    # — TOML's lexical scoping makes it easy to accidentally drop it under the
    # wrong header, and we want the natural placement to keep working.
    import shlex
    def _collect_extra(node: Any) -> list[str]:
        out: list[str] = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "extra_args":
                    if isinstance(v, list):
                        out += [str(x) for x in v]
                    elif isinstance(v, str) and v.strip():
                        out += shlex.split(v)
                elif isinstance(v, dict):
                    out += _collect_extra(v)
        return out
    argv += _collect_extra(cfg)

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    return argv, env


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Layer a few common overrides from the CLI onto the loaded config."""
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.video is not None:
        cfg.setdefault("video", {})["path"] = str(args.video)
    if args.depth is not None:
        cfg.setdefault("depth", {})["provider"] = args.depth
    if args.da3_model is not None:
        cfg.setdefault("depth", {})["da3_model"] = args.da3_model
    if args.pose is not None:
        cfg.setdefault("pose", {})["mode"] = args.pose
    if args.display_width is not None:
        cfg.setdefault("runtime", {})["display_width"] = args.display_width
    if args.max_fps is not None:
        cfg.setdefault("runtime", {})["max_fps"] = args.max_fps
    if args.save:
        cfg.setdefault("map", {})["save"] = True
    if args.load is not None:
        cfg.setdefault("map", {})["load_path"] = str(args.load)
    if args.output_dir is not None:
        cfg.setdefault("map", {})["output_dir"] = str(args.output_dir)
    if args.no_detect:
        cfg.setdefault("detection", {})["enabled"] = False
    if args.no_loop:
        cfg.setdefault("video", {})["no_loop"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"Path to TOML config (default: {DEFAULT_CONFIG.relative_to(REPO)})")
    parser.add_argument("--mode",
                        choices=["video", "viture-recording", "viture-live", "unitree-replay"],
                        help="Override [mode]")
    parser.add_argument("--video", type=Path, help="Override [video.path]")
    parser.add_argument("--depth", choices=["depthpro", "da3"], help="Override [depth.provider]")
    parser.add_argument("--da3-model",
                        choices=["da3-small", "da3-base", "da3-large",
                                 "da3-giant", "da3metric-large",
                                 "da3nested-giant-large"],
                        help="Override [depth.da3_model]. Recommend da3metric-large "
                             "for consistent (true metric) depth.")
    parser.add_argument("--pose", choices=["vo", "identity"], help="Override [pose.mode]")
    parser.add_argument("--display-width", type=int, help="Override [runtime.display_width]")
    parser.add_argument("--max-fps", type=float, help="Override [runtime.max_fps]")
    parser.add_argument("--save", action="store_true",
                        help="Override [map.save] = true (filename auto-stamped)")
    parser.add_argument("--load", type=Path, help="Override [map.load_path]")
    parser.add_argument("--output-dir", type=Path, help="Override [map.output_dir]")
    parser.add_argument("--no-detect", action="store_true", help="Disable detection")
    parser.add_argument("--no-loop", action="store_true", help="Exit at end-of-video")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config + argv without running")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the resolved config (after CLI overrides) as TOML and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.print_config:
        # Tiny TOML emitter — enough to re-load with tomllib later
        import json
        print(json.dumps(cfg, indent=2, default=str))
        return

    argv, env = build_command(cfg)

    print(f"[run_camera_pipeline] config: {args.config}")
    print(f"[run_camera_pipeline] mode:   {cfg.get('mode')}")
    if cfg.get("map", {}).get("save"):
        print(f"[run_camera_pipeline] save:   {compute_save_path(cfg)}")
    print(f"[run_camera_pipeline] argv:   {' '.join(argv)}")

    if args.dry_run:
        return

    try:
        result = subprocess.run(argv, env=env, cwd=str(REPO))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
