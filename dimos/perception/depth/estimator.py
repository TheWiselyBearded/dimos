# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monocular depth estimator wrappers: Depth Anything 3 and Apple Depth Pro.

Shared by the dimos ``DepthEstimationModule`` and the standalone spatial
scripts (``mac_iphone_spatial_foxglove.py``, ``mac_viture_*``). Each estimator
implements ``infer(color_rgb, fx) -> (depth_m, conf)``:

- ``depth_m``: float32 [H, W] metres (0 = invalid). Metric models
  (``outputs_metric``) return true scale; relative DA3 variants are normalized
  into [near, far] with a one-shot median scale fit — callers that fuse across
  frames should re-fit per frame (see ``xr_nav.scale_align``).
- ``conf``: float32 [H, W] in [0, 1] (all-ones when the model has none).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 6.0

# DA3 model-name → HF repo casing handled in DA3Estimator. The vendored DA3
# source lives in the repo (not pip-installed); macOS uses the
# awesome-depth-anything-3 fork, Linux/Windows the official ByteDance tree.
_DA3_DIRNAME = "awesome-depth-anything-3" if sys.platform == "darwin" else "Depth-Anything-3"


def _ensure_da3_importable() -> None:
    """Put the vendored Depth-Anything-3 source on sys.path if needed."""
    try:
        import depth_anything_3  # noqa: F401

        return
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "xr-nav" / _DA3_DIRNAME / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        return
    raise ImportError(
        "depth_anything_3 is not importable and the vendored copy was not "
        f"found at {candidate}. Install DA3 or check out xr-nav/{_DA3_DIRNAME}."
    )


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def filter_depth_edges(depth_m: np.ndarray, grad_threshold_m: float,
                       dilate_px: int) -> np.ndarray:
    """Zero out depth pixels on strong depth gradients (flying-pixel removal)."""
    if grad_threshold_m <= 0:
        return depth_m
    valid = depth_m > 0
    gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    bad = (grad_mag > grad_threshold_m) & valid
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        bad = cv2.dilate(bad.astype(np.uint8), k).astype(bool)
    out = depth_m.copy()
    out[bad] = 0.0
    return out


class DepthEstimator(ABC):
    name: str = "base"
    # True when depth_m is true metric scale (no per-frame re-fit needed).
    outputs_metric: bool = False
    # If True, callers should skip the depth-edge filter because per-pixel
    # gradients reflect amplified noise rather than real surface discontinuities.
    skip_edge_filter: bool = False

    @abstractmethod
    def infer(self, color_rgb: np.ndarray, fx: float) -> tuple[np.ndarray, np.ndarray]: ...


class DepthProEstimator(DepthEstimator):
    name = "depthpro"
    outputs_metric = True

    def __init__(self, device: str | None = None):
        try:
            import depth_pro
        except ImportError as e:
            raise ImportError(
                "depth_pro not installed. Install with:\n"
                "  pip install git+https://github.com/apple/ml-depth-pro.git"
            ) from e
        import torch
        self._torch = torch
        device = device or default_device()
        logger.info(f"loading depth-pro on {device}...")
        t0 = time.monotonic()
        model, transform = depth_pro.create_model_and_transforms()
        model.eval()
        self._model = model.to(torch.device(device))
        self._transform = transform
        self._device = torch.device(device)
        logger.info(f"depth-pro ready in {time.monotonic() - t0:.1f}s")

    def infer(self, color_rgb: np.ndarray, fx: float) -> tuple[np.ndarray, np.ndarray]:
        with self._torch.no_grad():
            inp = self._transform(color_rgb).to(self._device)
            f_px = self._torch.tensor(float(fx), dtype=self._torch.float32,
                                      device=self._device)
            pred = self._model.infer(inp, f_px=f_px)
            depth_m = pred["depth"].detach().cpu().numpy().astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m[0]
        conf = np.ones_like(depth_m, dtype=np.float32)
        # Drop GPU tensors before clearing the cache, otherwise the freed memory
        # is still pinned by the live references and the next frame compounds.
        del inp, f_px, pred
        try:
            if self._device.type == "mps":
                self._torch.mps.empty_cache()
            elif self._device.type == "cuda":
                self._torch.cuda.empty_cache()
        except Exception:
            pass
        return depth_m, conf


class DA3Estimator(DepthEstimator):
    name = "da3"

    def __init__(self, model_name: str = "da3metric-large", device: str | None = None,
                 process_res: int = 504, conf_threshold: float = 0.0,
                 force_relative: bool = False,
                 depth_near: float = DEPTH_NEAR_M, depth_far: float = DEPTH_FAR_M):
        _ensure_da3_importable()
        from depth_anything_3.api import DepthAnything3
        device = device or default_device()
        logger.info(f"loading {model_name} on {device}...")
        # The DepthAnything3 constructor only builds the architecture; pretrained
        # weights load via from_pretrained (PyTorchModelHubMixin). A bare constructor
        # leaves the net uninitialized — its near-constant output gets stretched into
        # blocky garbage — so on every platform load weights via from_pretrained,
        # then move to device.
        self._model = DepthAnything3.from_pretrained(
            f"depth-anything/{model_name.upper()}").to(device)
        # Prediction.is_metric is unreliable (returns {} -> falsy); trust the model name.
        self._metric_model = "metric" in model_name.lower()
        self.outputs_metric = self._metric_model and not force_relative
        self._res = process_res
        self._conf_thresh = conf_threshold
        self._force_relative = force_relative
        self._near = float(depth_near)
        self._far = float(depth_far)
        self._scale: float | None = None
        self._conf_logged = False
        self._raw_logged = False
        # DA3 also estimates per-frame intrinsics/extrinsics; kept from the
        # last infer() for callers that want them (e.g. DA3-pose fallback).
        self.last_intrinsics: np.ndarray | None = None
        self.last_extrinsics: np.ndarray | None = None

    def infer(self, color_rgb: np.ndarray, fx: float) -> tuple[np.ndarray, np.ndarray]:
        pred = self._model.inference(image=[color_rgb], process_res=self._res)
        raw = np.nan_to_num(pred.depth[0].astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        self.last_intrinsics = (np.asarray(pred.intrinsics[0])
                                if getattr(pred, "intrinsics", None) is not None else None)
        self.last_extrinsics = (np.asarray(pred.extrinsics[0])
                                if getattr(pred, "extrinsics", None) is not None else None)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        is_metric = ((self._metric_model or bool(getattr(pred, "is_metric", 0)))
                     and not self._force_relative)
        if not self._raw_logged:
            r_pos = raw[raw > 0]
            r_range_val = float(r_pos.max() - r_pos.min()) if r_pos.size else 0.0
            r_range = (f"[{r_pos.min():.3f},{r_pos.max():.3f}] med={float(np.median(r_pos)):.3f}"
                       if r_pos.size else "[empty]")
            logger.info(f"DA3 raw {r_range} is_metric={is_metric} "
                        f"force_relative={self._force_relative}")
            # In relative mode, narrow raw range means we'll be stretching tiny
            # signal + noise across [near, far]. The edge filter sees that noise
            # as gradients and wipes the map. Auto-disable it and warn loudly.
            if not is_metric and r_range_val < 0.2:
                self.skip_edge_filter = True
                logger.warning(
                    f"DA3 raw dynamic range is only {r_range_val:.3f}. "
                    f"Auto-disabling depth-edge filter (output will be noisy). "
                    f"For sharp depth on this content, use depthpro.")
            self._raw_logged = True
        if is_metric:
            depth_m = raw
        else:
            valid = raw > 0
            depth_norm = np.zeros_like(raw, dtype=np.float32)
            if valid.any():
                vals = raw[valid]
                rmin, rmax = float(vals.min()), float(vals.max())
                if rmax - rmin < 1e-8:
                    depth_norm[valid] = 0.5 * (self._near + self._far)
                else:
                    d_norm = (vals - rmin) / (rmax - rmin)
                    depth_norm[valid] = (self._near + d_norm
                                         * (self._far - self._near)).astype(np.float32)
            if self._scale is None and valid.any():
                med = float(np.median(depth_norm[valid]))
                self._scale = 1.5 / med if med > 1e-6 else 1.0
                logger.info(f"DA3 first-frame scale fit: {self._scale:.3f}")
            scale = self._scale if self._scale is not None else 1.0
            depth_m = (depth_norm * scale).astype(np.float32)

        conf_map = np.ones_like(depth_m, dtype=np.float32)
        if pred.conf is not None:
            c = pred.conf[0].astype(np.float32)
            cmax = float(c.max()) if c.size else 1.0
            cmin = float(c.min()) if c.size else 0.0
            if cmax > 1.0:
                c = c / cmax
            conf_map = c
            if not self._conf_logged:
                kept = float((c >= self._conf_thresh).mean())
                logger.info(f"DA3 conf range=[{cmin:.3f},{cmax:.3f}] "
                            f"threshold={self._conf_thresh:.2f} kept_frac={kept:.3f}")
                self._conf_logged = True
            if self._conf_thresh > 0.0:
                depth_m = np.where(c >= self._conf_thresh, depth_m, 0.0).astype(np.float32)
        return depth_m, conf_map

    def infer_multi(self, images_rgb: list[np.ndarray],
                    c2w_list: list[np.ndarray] | None = None,
                    K_list: list[np.ndarray] | None = None) -> list[dict]:
        """Multi-view DA3 inference over a window of frames (Phase C).

        DA3's multi-view pass makes depth mutually consistent across the window
        — the one thing single-frame inference can't do — which removes the
        per-frame metric wobble that fuzzes nearby objects. When kinematic
        camera poses are provided they are passed as priors (inverted to the
        world-to-camera convention DA3 uses); with ``align_to_input_ext_scale``
        the model adopts those poses and rescales its depth to their metric
        scale via Umeyama, so every window lands in the same gravity-correct,
        metric, world-locked frame — no cross-window Sim3 stitcher needed.

        Args:
            images_rgb: N RGB frames (H, W, 3), uint8 or float.
            c2w_list:   Optional N kinematic camera-to-world 4x4 priors. When
                        given, unprojection uses these directly (DA3 adopts them).
            K_list:     Optional N intrinsics (3x3) at the image resolution.

        Returns:
            List of N dicts: {"depth" (H,W) metres, "conf" (H,W), "K" (3,3),
            "c2w" (4,4)}. depth/conf are at DA3's processed resolution; K is the
            matching intrinsics for unprojection at that resolution.
        """
        ext = None  # world-to-camera priors (DA3 convention)
        if c2w_list is not None:
            ext = np.stack([
                np.linalg.inv(np.asarray(c, dtype=np.float64)) for c in c2w_list
            ]).astype(np.float64)
        intr = None
        if K_list is not None:
            intr = np.stack([np.asarray(k, dtype=np.float64) for k in K_list])

        pred = self._model.inference(
            image=list(images_rgb), extrinsics=ext, intrinsics=intr,
            align_to_input_ext_scale=True, process_res=self._res,
        )
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        n = len(images_rgb)
        out: list[dict] = []
        for i in range(n):
            depth = np.nan_to_num(pred.depth[i].astype(np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
            if pred.conf is not None:
                cf = pred.conf[i].astype(np.float32)
                cmax = float(cf.max()) if cf.size else 1.0
                if cmax > 1.0:
                    cf = cf / cmax
            else:
                cf = np.ones_like(depth, dtype=np.float32)
            Ki = (np.asarray(pred.intrinsics[i], dtype=np.float64)
                  if getattr(pred, "intrinsics", None) is not None
                  else (intr[i] if intr is not None else None))
            if c2w_list is not None:
                # DA3 adopted our priors (returns them as a 3x4 slice); use the
                # exact 4x4 we passed rather than re-inflating the slice.
                c2w = np.asarray(c2w_list[i], dtype=np.float64)
            elif getattr(pred, "extrinsics", None) is not None:
                w2c = np.eye(4, dtype=np.float64)
                w2c[:3, :] = np.asarray(pred.extrinsics[i], dtype=np.float64)[:3, :]
                c2w = np.linalg.inv(w2c)
            else:
                c2w = np.eye(4, dtype=np.float64)
            out.append({"depth": depth, "conf": cf, "K": Ki, "c2w": c2w})
        return out


def resolve_da3_defaults(model_name: str, conf_threshold: float | None,
                         process_res: int | None,
                         trust_is_metric: bool = False) -> tuple[float, int, bool]:
    """Per-variant DA3 defaults shared by the module and the fork scripts.

    Returns (conf_threshold, process_res, force_relative):
    - metric variants have a calibrated conf channel worth gating on (0.5) and
      benefit from higher input res (700); relative variants keep everything
      (0.0) at 504.
    - force_relative is True unless the variant is metric or the caller opted
      into trusting the model's (unreliable) is_metric flag.
    """
    model_is_metric = "metric" in model_name.lower()
    if conf_threshold is None:
        conf_threshold = 0.5 if model_is_metric else 0.0
    if process_res is None:
        process_res = 700 if (model_is_metric or "nested" in model_name.lower()) else 504
    force_relative = not (trust_is_metric or model_is_metric)
    return conf_threshold, process_res, force_relative


def make_depth_estimator(kind: str, device: str | None = None,
                         da3_conf_threshold: float = 0.0,
                         da3_force_relative: bool = False,
                         da3_model: str = "da3metric-large",
                         da3_process_res: int = 504,
                         depth_near: float = DEPTH_NEAR_M,
                         depth_far: float = DEPTH_FAR_M) -> DepthEstimator:
    if kind == "depthpro":
        return DepthProEstimator(device=device)
    if kind == "da3":
        return DA3Estimator(model_name=da3_model, device=device,
                            process_res=da3_process_res,
                            conf_threshold=da3_conf_threshold,
                            force_relative=da3_force_relative,
                            depth_near=depth_near, depth_far=depth_far)
    raise ValueError(f"unknown depth kind: {kind}")
