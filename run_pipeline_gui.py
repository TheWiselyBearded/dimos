"""Tkinter GUI launcher for the spatial camera pipelines.

Reads defaults from configs/camera_pipeline.toml on startup, exposes them as
form fields, and on Launch shells out to ./run_pipeline.sh with the matching
CLI args.

Quick presets (configs/presets/*.toml) populate the form with one click and
can be launched directly without form interaction.

Most form sections are collapsible — only Mode + Source are expanded by
default to keep the launcher uncluttered.

Run:
    /opt/anaconda3/envs/xr-nav/bin/python run_pipeline_gui.py

Or via run_pipeline.sh (no args) — that script invokes this one.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from tkinter import (
    BooleanVar, DoubleVar, IntVar, StringVar, Tk,
    filedialog, messagebox, ttk,
)
from typing import Any, Callable

REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs" / "camera_pipeline.toml"
PRESETS_DIR = REPO / "configs" / "presets"
SHELL_SCRIPT = REPO / "run_pipeline.sh"
DATASETS_DIR = REPO / "datasets"
DEFAULT_MAPS_DIR = Path.home() / ".dimos" / "sessions"


@dataclass
class PipelineConfig:
    mode: str = "video"
    video_path: str = ""
    video_hfov_deg: float = 62.0
    video_no_loop: bool = False
    viture_video: str = ""
    viture_right_video: str = ""
    viture_recording_dir: str = ""
    pose_mode: str = "vo"
    depth_provider: str = "depthpro"
    depth_device: str = "mps"
    da3_model: str = "da3-small"
    da3_trust_is_metric: bool = False
    det_enabled: bool = True
    det_class_aware: bool = True
    det_decay: bool = True
    save_map: bool = False
    save_output_dir: str = "~/.dimos/sessions"
    save_filename_template: str = "session_{ts}.pkl"
    load_map_path: str = ""
    save_every_n: int = 0
    display_width: int = 768
    max_fps: float = 5.0
    extra_args: list[str] | None = None  # passthrough; only set by presets

    @classmethod
    def from_toml(cls, path: Path) -> "PipelineConfig":
        if not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        c = cls()
        c.mode = data.get("mode", c.mode)
        rt = data.get("runtime", {})
        if rt.get("display_width") is not None:
            c.display_width = int(rt["display_width"])
        if rt.get("max_fps") is not None:
            c.max_fps = float(rt["max_fps"])
        c.pose_mode = data.get("pose", {}).get("mode", c.pose_mode)
        d = data.get("depth", {})
        c.depth_provider = d.get("provider", c.depth_provider)
        c.depth_device = d.get("device", c.depth_device)
        c.da3_model = d.get("da3_model", c.da3_model)
        c.da3_trust_is_metric = bool(d.get("da3_trust_is_metric", c.da3_trust_is_metric))
        det = data.get("detection", {})
        c.det_enabled = bool(det.get("enabled", c.det_enabled))
        c.det_class_aware = bool(det.get("class_aware", c.det_class_aware))
        c.det_decay = bool(det.get("decay", c.det_decay))
        m = data.get("map", {})
        c.save_map = bool(m.get("save", c.save_map))
        c.save_output_dir = m.get("output_dir", c.save_output_dir)
        c.save_filename_template = m.get("save_filename", c.save_filename_template)
        c.load_map_path = m.get("load_path", c.load_map_path)
        c.save_every_n = int(m.get("save_every_n_frames", c.save_every_n))
        v = data.get("video", {})
        c.video_path = v.get("path", c.video_path)
        c.video_hfov_deg = float(v.get("hfov_deg", c.video_hfov_deg))
        c.video_no_loop = bool(v.get("no_loop", c.video_no_loop))
        vit = data.get("viture", {})
        c.viture_video = vit.get("video", c.viture_video)
        c.viture_right_video = vit.get("right_video", c.viture_right_video)
        c.viture_recording_dir = vit.get("recording_dir", c.viture_recording_dir)
        extra = data.get("extra_args")
        if isinstance(extra, list):
            c.extra_args = [str(x) for x in extra]
        return c

    def to_cli_args(self) -> list[str]:
        """Build run_camera_pipeline.py CLI args from this config."""
        args: list[str] = ["--mode", self.mode]
        if self.mode == "unitree-replay":
            return args  # script takes no other flags
        if self.mode == "video" and self.video_path:
            args += ["--video", self.video_path]
        args += ["--depth", self.depth_provider]
        if self.depth_provider == "da3":
            args += ["--da3-model", self.da3_model]
        if self.mode == "video":
            args += ["--pose", self.pose_mode]
        args += ["--display-width", str(self.display_width)]
        args += ["--max-fps", str(self.max_fps)]
        if self.save_map:
            args.append("--save")
            if self.save_output_dir:
                args += ["--output-dir", self.save_output_dir]
        if self.load_map_path:
            args += ["--load", self.load_map_path]
        if not self.det_enabled:
            args.append("--no-detect")
        if self.video_no_loop and self.mode == "video":
            args.append("--no-loop")
        # extra_args isn't representable as CLI to run_camera_pipeline.py without
        # adding a new flag — presets that need extra_args are launched via
        # --config <preset.toml> instead. See PipelineGUI._on_launch_preset.
        return args


def _bridge_is_up(port: int = 8765) -> bool:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=2,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _list_presets() -> list[Path]:
    if not PRESETS_DIR.exists():
        return []
    return sorted(PRESETS_DIR.glob("*.toml"))


# ─────────────────────────────────────────────────────────────────────────
#  CollapsibleSection — header label that toggles content visibility on click
# ─────────────────────────────────────────────────────────────────────────

class CollapsibleSection(ttk.Frame):
    """A LabelFrame-like container with a clickable header that hides/shows
    its contents. Add children to ``self.content`` rather than to the section
    itself.
    """

    def __init__(self, parent: Any, title: str, *, expanded: bool = True,
                 padding: int = 8) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        # Header is a thin frame with a label that looks like a disclosure toggle.
        self._header = ttk.Frame(self)
        self._header.pack(fill="x")
        self._toggle_lbl = ttk.Label(
            self._header,
            text=self._header_text(),
            font=("TkDefaultFont", 12, "bold"),
            cursor="hand2",
        )
        self._toggle_lbl.pack(side="left", padx=4, pady=2)
        self._toggle_lbl.bind("<Button-1>", self._toggle)
        # Whole header row is clickable.
        self._header.bind("<Button-1>", self._toggle)

        self.content = ttk.Frame(self, padding=padding,
                                 relief="groove", borderwidth=1)
        if expanded:
            self.content.pack(fill="x")

    def _header_text(self) -> str:
        marker = "▼" if self._expanded else "▶"
        return f"{marker}  {self._title}"

    def _toggle(self, _event: Any = None) -> None:
        self._expanded = not self._expanded
        self._toggle_lbl.config(text=self._header_text())
        if self._expanded:
            self.content.pack(fill="x")
        else:
            self.content.pack_forget()

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._toggle()


# ─────────────────────────────────────────────────────────────────────────
#  Main GUI
# ─────────────────────────────────────────────────────────────────────────

class PipelineGUI:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Dimos — Camera Pipeline Launcher")
        self.root.geometry("760x820")
        try:
            self.cfg = PipelineConfig.from_toml(DEFAULT_CONFIG)
        except Exception as e:
            messagebox.showwarning("Config", f"Could not read default config:\n{e}\n\nUsing built-in defaults.")
            self.cfg = PipelineConfig()

        self._init_vars()
        self._build_ui()
        self._update_mode_visibility()
        self._update_depth_visibility()
        self._poll_bridge_status()

    # ─────────────────────────────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────────────────────────────

    def _init_vars(self) -> None:
        c = self.cfg
        self.var_mode = StringVar(value=c.mode)

        self.var_video = StringVar(value=c.video_path)
        self.var_hfov = DoubleVar(value=c.video_hfov_deg)
        self.var_no_loop = BooleanVar(value=c.video_no_loop)

        self.var_viture_video = StringVar(value=c.viture_video)
        self.var_viture_right = StringVar(value=c.viture_right_video)
        self.var_viture_dir = StringVar(value=c.viture_recording_dir)

        self.var_pose = StringVar(value=c.pose_mode)

        self.var_depth = StringVar(value=c.depth_provider)
        self.var_device = StringVar(value=c.depth_device)
        self.var_da3 = StringVar(value=c.da3_model)
        self.var_da3_trust = BooleanVar(value=c.da3_trust_is_metric)

        self.var_det = BooleanVar(value=c.det_enabled)
        self.var_class = BooleanVar(value=c.det_class_aware)
        self.var_decay = BooleanVar(value=c.det_decay)

        self.var_save = BooleanVar(value=c.save_map)
        self.var_save_dir = StringVar(value=c.save_output_dir)
        self.var_save_name = StringVar(value=c.save_filename_template)
        self.var_load = StringVar(value=c.load_map_path)
        self.var_save_every = IntVar(value=c.save_every_n)

        self.var_dw = IntVar(value=c.display_width)
        self.var_fps = DoubleVar(value=c.max_fps)

        # Presets
        self._presets = _list_presets()
        preset_labels = [p.stem for p in self._presets]
        self.var_preset = StringVar(value=preset_labels[0] if preset_labels else "")

        self.var_status = StringVar(value="bridge: unknown · pipeline: not launched")
        self.var_preview = StringVar(value="")

    # ─────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.outer = ttk.Frame(self.root, padding=12)
        self.outer.pack(fill="both", expand=True)
        outer = self.outer

        # ----- Quick presets (always-visible at top) -----
        self._build_presets_panel(outer)

        # ----- Mode (always-visible) -----
        self.mode_section = CollapsibleSection(outer, "Mode", expanded=True)
        self.mode_section.pack(fill="x", pady=(8, 6))
        for value, label in [
            ("video", "Video file (any mp4 / MOV — uses VO for pose)"),
            ("viture-recording", "Viture recording (mp4 + ARKit poses)"),
            ("viture-live", "Viture live (TCP stream, real-time)"),
            ("unitree-replay", "Unitree Go2 dataset replay (lidar + camera)"),
        ]:
            ttk.Radiobutton(
                self.mode_section.content, text=label,
                variable=self.var_mode, value=value,
                command=self._update_mode_visibility,
            ).pack(anchor="w")

        # ----- Source (mode-dependent; expanded; visibility via _update_mode_visibility) -----
        self.src_video = CollapsibleSection(outer, "Source — video file", expanded=True)
        self._build_src_video(self.src_video.content)

        self.src_viture = CollapsibleSection(outer, "Source — Viture recording", expanded=True)
        self._build_src_viture(self.src_viture.content)

        # ----- Depth (collapsible; packed by _update_mode_visibility) -----
        self.depth_section = CollapsibleSection(outer, "Depth model", expanded=False)
        self._build_depth(self.depth_section.content)

        # ----- Pose (mode-dependent) -----
        self.pose_section = CollapsibleSection(outer, "Pose (video mode)", expanded=False)
        ttk.Radiobutton(self.pose_section.content,
                        text="Visual odometry (ORB + depth-PnP)",
                        variable=self.var_pose, value="vo").pack(anchor="w")
        ttk.Radiobutton(self.pose_section.content,
                        text="Identity (camera at origin — debug only)",
                        variable=self.var_pose, value="identity").pack(anchor="w")

        # ----- Object tracking (collapsible; packed by _update_mode_visibility) -----
        self.det_section = CollapsibleSection(outer, "Object tracking", expanded=False)
        ttk.Checkbutton(self.det_section.content,
                        text="Enable YOLOE 2D detection + ObjectDB",
                        variable=self.var_det).pack(anchor="w")
        ttk.Checkbutton(self.det_section.content,
                        text="Class-aware matching (require name to match)",
                        variable=self.var_class).pack(anchor="w")
        ttk.Checkbutton(self.det_section.content,
                        text="Confidence decay (delete in-frustum no-shows)",
                        variable=self.var_decay).pack(anchor="w")

        # ----- Map persistence (collapsible; packed by _update_mode_visibility) -----
        self.map_section = CollapsibleSection(outer, "Map persistence", expanded=False)
        self._build_map(self.map_section.content)

        # ----- Runtime (collapsible; packed by _update_mode_visibility) -----
        self.rt_section = CollapsibleSection(outer, "Runtime", expanded=False)
        self._build_runtime(self.rt_section.content)

        # ----- Status + buttons (always visible at bottom) -----
        self._build_status_and_buttons(outer)

    def _build_presets_panel(self, parent: Any) -> None:
        section = ttk.LabelFrame(parent, text="Quick presets", padding=8)
        section.pack(fill="x")
        ttk.Label(section, text="Sample command:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        labels = [p.stem for p in self._presets] or ["(no presets in configs/presets/)"]
        cb = ttk.Combobox(section, textvariable=self.var_preset, values=labels,
                          state="readonly" if self._presets else "disabled", width=40)
        cb.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_changed())
        ttk.Button(section, text="Load to form",
                   command=self._on_load_preset).grid(row=0, column=2, padx=4)
        ttk.Button(section, text="Launch preset",
                   command=self._on_launch_preset).grid(row=0, column=3, padx=4)
        ttk.Label(section, text="(Launch preset uses the TOML directly — bypasses form fields)",
                  foreground="#666").grid(row=1, column=1, columnspan=3, sticky="w", padx=4)

    def _build_src_video(self, parent: Any) -> None:
        self._row_file(parent, "Video:", self.var_video,
                       lambda: self._pick_file(
                           self.var_video, "Select video file",
                           [("Video", "*.mp4 *.MOV *.mov *.avi *.mkv"), ("All files", "*.*")],
                           initialdir=DATASETS_DIR if DATASETS_DIR.exists() else REPO))
        ttk.Label(parent, text="Tip: drop new clips into datasets/iphone/ — gitignored",
                  foreground="#666").grid(row=1, column=1, sticky="w", padx=4)
        self._row_number(parent, "HFOV (deg):", self.var_hfov, row=2, width=8,
                         hint="iPhone wide ~62, ultrawide ~106, 2x telephoto ~30")
        ttk.Checkbutton(parent, text="Exit at end of video (don't loop)",
                        variable=self.var_no_loop).grid(row=3, column=1, sticky="w", padx=4)

    def _build_src_viture(self, parent: Any) -> None:
        self._row_file(parent, "Left video:", self.var_viture_video,
                       lambda: self._pick_file(self.var_viture_video,
                                              "Select left video", [("Video", "*.mp4")]))
        self._row_file(parent, "Right video (optional):", self.var_viture_right,
                       lambda: self._pick_file(self.var_viture_right,
                                              "Select right video", [("Video", "*.mp4")]),
                       row=1)
        self._row_dir(parent, "Recording dir:", self.var_viture_dir,
                      lambda: self._pick_dir(self.var_viture_dir, "Select recording directory"),
                      row=2)
        ttk.Label(parent, text="Empty = use script defaults",
                  foreground="#666").grid(row=3, column=1, sticky="w", padx=4)

    def _build_depth(self, parent: Any) -> None:
        ttk.Label(parent, text="Provider:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        depth_cb = ttk.Combobox(parent, textvariable=self.var_depth,
                                values=["depthpro", "da3"], state="readonly", width=14)
        depth_cb.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        depth_cb.bind("<<ComboboxSelected>>", lambda _e: self._update_depth_visibility())
        ttk.Label(parent, text="Device:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        ttk.Combobox(parent, textvariable=self.var_device, values=["mps", "cuda", "cpu"],
                     state="readonly", width=8).grid(row=0, column=3, sticky="w", padx=4, pady=2)
        self.da3_row = ttk.Frame(parent)
        self.da3_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Label(self.da3_row, text="DA3 size:").pack(side="left", padx=4)
        ttk.Combobox(self.da3_row, textvariable=self.var_da3,
                     values=["da3-small", "da3-base", "da3-large"], state="readonly",
                     width=12).pack(side="left", padx=4)
        ttk.Checkbutton(self.da3_row, text="Trust is_metric (rare; usually leave off)",
                        variable=self.var_da3_trust).pack(side="left", padx=8)

    def _build_map(self, parent: Any) -> None:
        ttk.Checkbutton(parent, text="Save map on exit",
                        variable=self.var_save).grid(row=0, column=0, columnspan=2,
                                                     sticky="w", padx=4)
        self._row_dir(parent, "Output dir:", self.var_save_dir,
                      lambda: self._pick_dir(self.var_save_dir, "Select output directory",
                                             initialdir=DEFAULT_MAPS_DIR.parent),
                      row=1)
        ttk.Label(parent, text="Filename template:").grid(row=2, column=0, sticky="e",
                                                          padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.var_save_name,
                  width=40).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(parent, text="{ts} expands to YYYYMMDD_HHMMSS",
                  foreground="#666").grid(row=2, column=2, sticky="w", padx=4)
        self._row_file(parent, "Load existing map:", self.var_load,
                       lambda: self._pick_file(self.var_load, "Select map bundle (.pkl)",
                                              [("Map bundle", "*.pkl"), ("All files", "*.*")],
                                              initialdir=DEFAULT_MAPS_DIR
                                              if DEFAULT_MAPS_DIR.exists() else REPO),
                       row=3)
        ttk.Button(parent, text="Clear", width=6,
                   command=lambda: self.var_load.set("")).grid(row=3, column=3, sticky="w", padx=4)
        ttk.Label(parent, text="Periodic save every N frames (0 = on-exit only):").grid(
            row=4, column=0, sticky="e", padx=4, pady=2)
        ttk.Spinbox(parent, from_=0, to=10000, increment=10, textvariable=self.var_save_every,
                    width=8).grid(row=4, column=1, sticky="w", padx=4, pady=2)

    def _build_runtime(self, parent: Any) -> None:
        ttk.Label(parent, text="Display width (px):").grid(row=0, column=0, sticky="e",
                                                            padx=4, pady=2)
        ttk.Spinbox(parent, from_=256, to=2048, increment=64, textvariable=self.var_dw,
                    width=8).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(parent, text="Max FPS:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        ttk.Spinbox(parent, from_=0.5, to=30.0, increment=0.5, textvariable=self.var_fps,
                    width=8).grid(row=0, column=3, sticky="w", padx=4, pady=2)
        ttk.Label(parent, text="Lower display width if depth model is the bottleneck",
                  foreground="#666").grid(row=1, column=0, columnspan=4, sticky="w", padx=4)

    def _build_status_and_buttons(self, parent: Any) -> None:
        # Pack from the bottom up so these stay anchored at the bottom even
        # when the dynamic mode-dependent sections above re-pack themselves.
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(btn_frame, text="Preview command",
                   command=self._on_preview).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset to defaults",
                   command=self._on_reset).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Quit", command=self.root.destroy).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Launch (form)",
                   command=self._on_launch).pack(side="right", padx=4)

        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(ctrl_frame, textvariable=self.var_status,
                  foreground="#0a0").pack(anchor="w")
        ttk.Label(ctrl_frame, textvariable=self.var_preview, foreground="#444",
                  wraplength=720, justify="left").pack(anchor="w", pady=(4, 0))

    # ─────────────────────────────────────────────────────────────────────
    # Row helpers
    # ─────────────────────────────────────────────────────────────────────

    def _row_file(self, parent: Any, label: str, var: StringVar,
                  cmd: Callable[[], None], row: int = 0) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=60).grid(row=row, column=1,
                                                           sticky="ew", padx=4, pady=2)
        ttk.Button(parent, text="Browse…", command=cmd, width=10).grid(row=row, column=2,
                                                                       padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

    def _row_dir(self, parent: Any, label: str, var: StringVar,
                 cmd: Callable[[], None], row: int = 0) -> None:
        self._row_file(parent, label, var, cmd, row=row)

    def _row_number(self, parent: Any, label: str, var: Any, row: int = 0,
                    width: int = 8, hint: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1,
                                                              sticky="w", padx=4, pady=2)
        if hint:
            ttk.Label(parent, text=hint, foreground="#666").grid(row=row, column=2,
                                                                 sticky="w", padx=4)

    def _pick_file(self, var: StringVar, title: str, types: list[tuple[str, str]],
                   initialdir: Path | None = None) -> None:
        path = filedialog.askopenfilename(title=title, filetypes=types,
                                          initialdir=str(initialdir or REPO))
        if path:
            var.set(path)

    def _pick_dir(self, var: StringVar, title: str,
                  initialdir: Path | None = None) -> None:
        path = filedialog.askdirectory(title=title, initialdir=str(initialdir or REPO))
        if path:
            var.set(path)

    # ─────────────────────────────────────────────────────────────────────
    # Visibility
    # ─────────────────────────────────────────────────────────────────────

    def _update_mode_visibility(self) -> None:
        """Re-pack the sections that vary by mode.

        Status + button rows are packed with side="bottom" elsewhere, so they
        stay anchored regardless of how often this re-runs. Top-packed sections
        flow above them in the order we pack them here.
        """
        mode = self.var_mode.get()
        for fr in (self.src_video, self.src_viture, self.pose_section,
                   self.depth_section, self.det_section, self.map_section, self.rt_section):
            fr.pack_forget()

        order: list[CollapsibleSection]
        if mode == "video":
            order = [self.src_video, self.pose_section, self.depth_section,
                     self.det_section, self.map_section, self.rt_section]
        elif mode == "viture-recording":
            order = [self.src_viture, self.depth_section, self.det_section,
                     self.map_section, self.rt_section]
        elif mode == "viture-live":
            order = [self.depth_section, self.det_section, self.map_section, self.rt_section]
        elif mode == "unitree-replay":
            order = []  # script takes no flags — no config sections needed
        else:
            order = []

        for fr in order:
            fr.pack(fill="x", pady=(0, 6))

    def _update_depth_visibility(self) -> None:
        if self.var_depth.get() == "da3":
            self.da3_row.grid()
        else:
            self.da3_row.grid_remove()

    # ─────────────────────────────────────────────────────────────────────
    # Status polling
    # ─────────────────────────────────────────────────────────────────────

    def _poll_bridge_status(self) -> None:
        bridge = "up :8765" if _bridge_is_up() else "down"
        cur = self.var_status.get()
        suffix = " · pipeline: launched" if "launched" in cur and "not launched" not in cur else " · pipeline: not launched"
        self.var_status.set(f"bridge: {bridge}{suffix}")
        self.root.after(2000, self._poll_bridge_status)

    # ─────────────────────────────────────────────────────────────────────
    # Actions: form
    # ─────────────────────────────────────────────────────────────────────

    def _gather_config(self) -> PipelineConfig:
        c = PipelineConfig()
        c.mode = self.var_mode.get()
        c.video_path = self.var_video.get().strip()
        c.video_hfov_deg = float(self.var_hfov.get())
        c.video_no_loop = bool(self.var_no_loop.get())
        c.viture_video = self.var_viture_video.get().strip()
        c.viture_right_video = self.var_viture_right.get().strip()
        c.viture_recording_dir = self.var_viture_dir.get().strip()
        c.pose_mode = self.var_pose.get()
        c.depth_provider = self.var_depth.get()
        c.depth_device = self.var_device.get()
        c.da3_model = self.var_da3.get()
        c.da3_trust_is_metric = bool(self.var_da3_trust.get())
        c.det_enabled = bool(self.var_det.get())
        c.det_class_aware = bool(self.var_class.get())
        c.det_decay = bool(self.var_decay.get())
        c.save_map = bool(self.var_save.get())
        c.save_output_dir = self.var_save_dir.get().strip()
        c.save_filename_template = self.var_save_name.get().strip()
        c.load_map_path = self.var_load.get().strip()
        c.save_every_n = int(self.var_save_every.get())
        c.display_width = int(self.var_dw.get())
        c.max_fps = float(self.var_fps.get())
        return c

    def _on_preview(self) -> None:
        cfg = self._gather_config()
        argv = [str(SHELL_SCRIPT), "--headless", *cfg.to_cli_args()]
        self.var_preview.set(" ".join(argv))

    def _on_reset(self) -> None:
        self.cfg = PipelineConfig.from_toml(DEFAULT_CONFIG)
        self._init_vars()
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()
        self._update_mode_visibility()
        self._update_depth_visibility()

    def _on_launch(self) -> None:
        cfg = self._gather_config()
        argv = [str(SHELL_SCRIPT), "--headless", *cfg.to_cli_args()]
        self._spawn_launch(argv)

    # ─────────────────────────────────────────────────────────────────────
    # Actions: presets
    # ─────────────────────────────────────────────────────────────────────

    def _selected_preset_path(self) -> Path | None:
        name = self.var_preset.get()
        for p in self._presets:
            if p.stem == name:
                return p
        return None

    def _on_preset_changed(self) -> None:
        # Show the resolved argv preview when the user picks a preset.
        path = self._selected_preset_path()
        if path is None:
            return
        argv = [str(SHELL_SCRIPT), "--headless", "--config", str(path)]
        self.var_preview.set(" ".join(argv) + "    (preset)")

    def _on_load_preset(self) -> None:
        path = self._selected_preset_path()
        if path is None:
            messagebox.showinfo("Preset", "No preset selected.")
            return
        try:
            self.cfg = PipelineConfig.from_toml(path)
        except Exception as e:
            messagebox.showerror("Preset", f"Failed to load {path.name}:\n{e}")
            return
        self._init_vars()
        for child in self.root.winfo_children():
            child.destroy()
        self._build_ui()
        self._update_mode_visibility()
        self._update_depth_visibility()
        if self.cfg.extra_args:
            messagebox.showinfo(
                "Preset loaded",
                f"{path.name} carries extra_args (passthrough flags) that aren't shown in the form. "
                f"Use 'Launch preset' instead of 'Launch (form)' to include them.",
            )

    def _on_launch_preset(self) -> None:
        """Launch the selected preset directly from its TOML — bypasses form
        fields and includes any extra_args verbatim."""
        path = self._selected_preset_path()
        if path is None:
            messagebox.showinfo("Preset", "No preset selected.")
            return
        argv = [str(SHELL_SCRIPT), "--headless", "--config", str(path)]
        self._spawn_launch(argv)

    # ─────────────────────────────────────────────────────────────────────
    # Common spawn
    # ─────────────────────────────────────────────────────────────────────

    def _spawn_launch(self, argv: list[str]) -> None:
        self.var_preview.set(" ".join(argv))
        if not SHELL_SCRIPT.exists():
            messagebox.showerror("Launch", f"run_pipeline.sh not found at {SHELL_SCRIPT}")
            return
        try:
            subprocess.Popen(argv, cwd=str(REPO))
        except Exception as e:
            messagebox.showerror("Launch", f"Failed to launch:\n{e}")
            return
        bridge = "up :8765" if _bridge_is_up() else "starting..."
        self.var_status.set(f"bridge: {bridge} · pipeline: launched (see Terminal windows)")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PipelineGUI().run()


if __name__ == "__main__":
    main()
