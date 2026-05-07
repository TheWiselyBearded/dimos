# Spatial Camera Pipelines (Viture XR + iPhone video)

This document describes the two `mac_*_foxglove.py` demo scripts that turn an
RGB(+pose) stream into:

- a fused, persistent **3D voxel map** in a world frame,
- a **tracked-object database** with class-aware matching, drift-tolerant
  re-association, and observation-decay culling,
- (optionally) a **CLIP-embedded spatial memory** for text/image queries,

all published live to [Foxglove](https://foxglove.dev) on `ws://localhost:8765`.

It is the orientation doc for an agent or engineer landing in this part of the
repo. Read it before touching:

- [`mac_viture_spatial_foxglove.py`](../../mac_viture_spatial_foxglove.py) — Viture XR1 wearable (head-mounted stereo + ARKit pose)
- [`mac_iphone_spatial_foxglove.py`](../../mac_iphone_spatial_foxglove.py) — generic monocular phone video (no pose; VO derives one)
- [`dimos/perception/detection/objectDB.py`](../../dimos/perception/detection/objectDB.py)
- [`xr-nav/src/xr_nav/voxel_map.py`](../../xr-nav/src/xr_nav/voxel_map.py)

> The two scripts are intentionally **standalone** (no shared library between
> them). They duplicate ~60% of their helpers. That's deliberate while the
> design is still moving — promote shared bits to `xr-nav` only after a third
> pipeline lands.

---

## 1. Architecture at a glance

```
┌──────────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  FrameSource │──▶│ Depth  │──▶│  Pose    │──▶│  VoxelMap│──▶│ Foxglove │
│              │   │  est.  │   │  est.    │   │  fusion  │   │ publish  │
│ Recording /  │   │ Depth- │   │ ARKit /  │   │          │   │ /map     │
│ Live (TCP) / │   │ Pro    │   │ VO+depth │   │ + raycast│   │ /tf      │
│ Video        │   │ DA3    │   │  PnP     │   │ clearing │   │ /points… │
│              │   │ Stereo │   │ Identity │   │          │   │          │
└──────────────┘   └────────┘   └──────────┘   └────┬─────┘   └──────────┘
                                                    │
                                  YOLOE 2D ◀────────┤
                                  detection         │
                                       │            │
                                       ▼            ▼
                                  ┌──────────────────┐    ┌────────────┐
                                  │     ObjectDB     │───▶│ /scene_    │
                                  │ - track-id match │    │   update   │
                                  │ - distance match │    │ /object_   │
                                  │ - pixel match    │    │   clouds   │
                                  │ - confidence dec.│    └────────────┘
                                  └──────────────────┘
                                              │
                                              ▼
                              ┌────────────────────────────┐
                              │  CLIP SpatialMemory (opt.) │
                              │  -> ChromaDB persistence   │
                              └────────────────────────────┘
```

Per-frame inner loop (both scripts, [mac_iphone_spatial_foxglove.py:825+](../../mac_iphone_spatial_foxglove.py)):

1. Pull `SourceFrame` from the active source.
2. Resample color → `display_width × DH`, derive intrinsics from `--hfov-deg`.
3. Run depth estimator → `(depth_m, conf)` at `DH × DW`.
4. Resolve pose (ARKit→OpenCV optical, VO, or identity).
5. Build per-frame colored cloud → push into `VoxelMap` with confidence
   weighting + raycast clear.
6. Run YOLOE 2D detection → lift to 3D `Object` instances (`Object.from_2d_to_list`).
7. Insert into `ObjectDB` with class-aware match cascade; decay any in-frustum
   non-observations.
8. Publish color, depth, points, map, scene-update, TF, and (optionally) CLIP
   embeddings.

---

## 2. The two scripts at a glance

| Aspect              | Viture script                                                                 | iPhone script                                                                  |
|---------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| File                | [`mac_viture_spatial_foxglove.py`](../../mac_viture_spatial_foxglove.py)      | [`mac_iphone_spatial_foxglove.py`](../../mac_iphone_spatial_foxglove.py)        |
| Source modes        | `recording` (mp4 + RecordingLoader poses) / `live` (`VitureClient` over TCP)  | `video` (any mp4/MOV)                                                           |
| Pose                | ARKit 4×4 from sensor                                                         | identity OR ORB+depth-PnP visual odometry                                       |
| Right-cam available | Yes (stereo possible)                                                         | No                                                                              |
| Default depth       | `depthpro`                                                                    | `depthpro`                                                                      |
| Default detection   | On                                                                            | On                                                                              |
| Save/load           | Not yet                                                                       | `--save-map` / `--load-map`                                                     |
| Default video path  | `~/Downloads/VITURE_recording_…_undistorted_left.mp4`                         | `dimos/datasets/iphone/phxLivingRoom.MOV` (gitignored — see [.gitignore](../../.gitignore)) |
| Default HFOV        | 46° (Viture sensor, undistorted)                                              | 62° (iPhone wide camera)                                                        |

---

## 3. The FrameSource layer

All sources yield a `SourceFrame` dataclass:

```python
@dataclass
class SourceFrame:
    color_bgr: np.ndarray                 # [H, W, 3] uint8 BGR
    color_right_bgr: np.ndarray | None    # Viture only — right cam for stereo
    c2w_arkit: np.ndarray | None          # 4×4 ARKit pose; None means "no pose"
    ts: float                             # wall-clock seconds
    frame_idx: int
```

Three concrete sources exist:

- **`RecordingSource`** ([viture script:278+](../../mac_viture_spatial_foxglove.py#L278)) — Plays a Viture recording: undistorted `.mp4` + `RecordingLoader` yielding the per-frame ARKit pose. Loops the video by default.
- **`LiveSource`** ([viture script:357+](../../mac_viture_spatial_foxglove.py#L357)) — Wraps `xr_nav.viture_client.VitureClient` (TCP). Frames arrive grayscale; we promote to BGR. Pose comes from the device.
- **`VideoSource`** ([iphone script:189+](../../mac_iphone_spatial_foxglove.py#L189)) — Any mp4/MOV. Decimates frames by integer step so a 30 fps file feeds a 10 fps depth model cleanly. Always emits `c2w_arkit=None`.

When you add a new source (e.g. RealSense, Android), follow `VideoSource`'s
shape and decide whether you have a usable pose stream. If not, the `vo`
pose mode in the iPhone script is your reusable pattern.

---

## 4. Pose estimation

The world frame is **OpenCV optical** (X right, Y down, Z forward). Whatever
the source delivers, it's converted to that frame before fusion.

### 4.1 ARKit conversion (Viture)

`arkit_c2w_to_opencv` ([viture script:112+](../../mac_viture_spatial_foxglove.py#L112)) flips the Y and Z axes. ARKit is +Y up, +Z back; OpenCV optical is +Y down, +Z forward. The flip is its own line — do not "fix" by composing into a more elaborate transform.

### 4.2 Monocular Visual Odometry (iPhone)

`MonocularDepthVO` ([iphone script:228+](../../mac_iphone_spatial_foxglove.py#L228)) implements an ORB feature + depth-PnP tracker. Each update:

1. Detect ORB on the new gray frame.
2. Match to the previous frame's descriptors (BFMatcher, Hamming, crossCheck).
3. Back-project the **previous** matched keypoints to 3D using the **previous** depth map → 3D-2D correspondences (`prev_3D, cur_2D`).
4. `cv2.solvePnPRansac` returns the extrinsic mapping `prev_cam → cur_cam`. The cur-in-prev relative pose is its inverse.
5. Compose onto the running `c2w` accumulator. Reject any single step whose translation exceeds 1.5 m as a glitch.

Why this approach:

- We compute depth every frame anyway, so adding 3D back-projection costs ~5 ms of ORB.
- Pure 2D-2D essential matrix is scale-ambiguous; using depth makes it metric.
- Drift accumulates linearly with frames — fine for short clips, noticeable on long ones. Add IMU integration or loop closure if it becomes a problem.

Failure modes (in practice):

- Low-texture walls / featureless rooms → fewer than `min_inliers=20` correspondences → no pose update that frame.
- Fast rotation → too few matches → no pose update.
- DA3 in relative mode produces noisy depth → wrong 3D back-projections → VO drifts. Pair VO with **DepthPro** for best results.

### 4.3 Identity (debug)

The iPhone script supports `--pose identity`. Every frame fuses at the origin
— useful only for confirming the depth → fusion path is healthy on its own.

---

## 5. Depth estimation

Three estimators are wired in. All return `(depth_m, conf)` in metres at the
display resolution.

| Estimator        | Class                                                                    | Strengths                               | Costs                                         |
|------------------|--------------------------------------------------------------------------|-----------------------------------------|-----------------------------------------------|
| Apple DepthPro   | `DepthProEstimator`                                                      | Sharp metric depth, robust to content   | Heavy on M-series Macs; ~10 s first frame, ~1–3 s/frame steady |
| Depth Anything 3 | `DA3Estimator` (sizes: `da3-small` / `da3-base` / `da3-large`)           | Fast (~140 ms/frame on M4)              | Often non-metric; small model can produce near-uniform output  |
| Stereo SGBM      | `StereoEstimator` (Viture only, needs right cam)                         | Cheap, no deep model                    | Brittle on low-texture surfaces                |

### 5.1 DepthPro

- Loads from `apple/ml-depth-pro` (auto-downloaded on first use; ~2 GB checkpoints).
- Calls `torch.mps.empty_cache()` between frames in the iPhone script ([iphone script:402+](../../mac_iphone_spatial_foxglove.py#L402)). **Do not remove this** — without it, MPS working-set grows until macOS swaps and the second frame appears to hang.
- The depth-pro `f_px` parameter is forced to `float32` because MPS doesn't support `float64`.

### 5.2 DA3 quirks (read this before debugging)

DA3 reports an `is_metric` flag that is **unreliable** on arbitrary phone footage. The iPhone script defaults to **ignoring `is_metric`** and always running the normalize-then-scale-fit path. Override with `--da3-trust-is-metric` only after verifying.

**The empty-map failure mode**: when DA3-small is shown low-information content (soft furnishings, uniform lighting), the raw output dynamic range collapses to a tiny window (e.g. `[0.987, 1.036]`). The relative-mode normalization stretches that into `[0.2, 6.0]m`, which amplifies per-pixel noise into apparent gradients ≥ 0.10 m/pixel. The depth-edge filter (`--depth-edge-threshold 0.10`) then zeros essentially every pixel and the map stays empty.

The script auto-detects this ([iphone script:457+](../../mac_iphone_spatial_foxglove.py#L457)): if raw range is < 0.2 in relative mode it sets `skip_edge_filter = True` on the estimator and prints a warning recommending DepthPro. **The signal-to-noise problem itself is upstream — the right fix is a bigger model** (`--da3-model da3-base` or `da3-large`) or switching to DepthPro.

### 5.3 The depth-edge filter

`filter_depth_edges` ([iphone script:158+](../../mac_iphone_spatial_foxglove.py#L158)) zeroes pixels where the local depth gradient exceeds `--depth-edge-threshold` m/pixel. Without this, every object boundary back-projects to a "ribbon" of points stuck halfway between foreground and background, smearing across the world map after fusion.

Tunable via `--depth-edge-threshold` (default 0.10 m/pixel) and `--depth-edge-dilate` (default 2 px). Setting the threshold to 0 disables the filter — only useful when you've confirmed the noise floor is genuinely small.

---

## 6. VoxelMap fusion

[`xr-nav/src/xr_nav/voxel_map.py`](../../xr-nav/src/xr_nav/voxel_map.py)

A hash-map voxel grid (`dict[(int,int,int), _Voxel]`) with:

- **Confidence-weighted centroid averaging** in `insert()`. Each new point's confidence (`1/depth²` if `--use-depth-confidence`, else 1.0) updates the running mean.
- **Drift-duplicate suppression** via `max_drift`. When a point lands in a *new* voxel cell, the 26-neighbour cube is checked; if a pre-existing centroid is within `max_drift` metres the point is treated as drift noise and skipped.
- **Raycast-based free-space clearing** in `raycast_clear()`. Each frame's points are rays from the camera origin; existing voxels in their path get a `miss_count` bump. After `max_misses` consecutive misses they're erased — that's how we know "the bottle was here last minute, but you're now looking through where it was, so delete it."
- **Distance pruning** (`prune()`) drops voxels beyond `max_range` of the current camera.
- **`min_observations` filter** in `to_points*()` hides voxels seen fewer than N times. The single biggest noise win on the render side — single-frame artifacts (transient depth glitches, pose jitter spikes) never become visible.

Key knobs (script CLI):

| Flag                            | Meaning                                                                              | Default |
|---------------------------------|--------------------------------------------------------------------------------------|---------|
| `--voxel-size`                  | Grid resolution                                                                      | 0.05 m  |
| `--voxel-min-observations`      | Render gate; raise to suppress single-frame ghosts                                   | 2       |
| `--voxel-max-drift`             | Drift duplicate threshold inside `insert()`                                          | 0.04 m  |
| `--use-depth-confidence`        | Down-weight far points in fusion (1/depth²)                                          | off     |
| `--raycast-every-n`             | Run raycast clear every N frames (raycast is the second-most expensive step)         | 1       |
| `--map-publish-every-n`         | Republish `/map` every N frames (large pointcloud serialize is ~50 ms)               | 5       |

---

## 7. Object tracking (ObjectDB)

[`dimos/perception/detection/objectDB.py`](../../dimos/perception/detection/objectDB.py)

Two-tier database (pending vs. permanent) with a multi-stage match cascade
and observation-decay culling. **All stages are class-aware by default.**

### 7.1 The match cascade (`_match`, [objectDB.py:227+](../../dimos/perception/detection/objectDB.py))

For each incoming `Object`:

1. **`_match_by_track_id`** — YOLOE supplies a `track_id` while it can hold a track. TTL is `track_id_ttl_s = 5 s`. Lost track → fall through.
2. **`_match_by_distance`** — nearest existing object within `distance_threshold` (default 0.4 m). When `class_aware_matching=True` (default), candidates are filtered to the same `name` first.
3. **`_match_by_pixel`** (added 2026-05) — *only runs if camera context is supplied to `add_objects`*. Projects each existing object's center through the world→camera transform to pixel coords; merges if a same-class new detection's `bbox` 2D center lands within `frustum_match_pixel_threshold` pixels (default 60 px). **This is what recovers matches when VO drift has shifted the world-frame center beyond the 3D distance threshold but the object is visually still in the same place.**

If all three fail → insert as a new pending object.

### 7.2 Pending → permanent promotion

Pending objects are upgraded to permanent once `detections_count ≥ min_detections_for_permanent` (default 2 — kept low so the user sees boxes immediately on Viture content where YOLOE LRPC track IDs are flaky).

Pending objects also expire after `pending_ttl_s = 5 s` of no observations (`_prune_stale_pending`).

### 7.3 Confidence decay (added 2026-05)

Every object holds a `[0, 1]` confidence (`_confidence` dict, keyed by object_id):

- **Insert**: confidence ← `confidence_init` (default 0.5).
- **Match**: confidence ← min(1.0, conf + `confidence_step_up`) (default +0.10).
- **In-frustum no-show**: `decay_unobserved()` reduces confidence by `confidence_step_down` (default 0.05) for every object whose center projects inside the current camera frustum but isn't in the `observed_ids` set passed by the caller.
- **Confidence ≤ 0**: object is deleted entirely (`_delete_object`) — from both tiers and from `_track_id_map`.

The caller decides what counts as "observed this frame" — typically the object IDs returned by `add_objects()`. **Decay only fires for objects projected inside the image rectangle and in front of the camera**, so walking away from a room does not delete its contents — only looking at where they were and not seeing them does.

The `decay_unobserved` step is currently called only by the iPhone script. Adding it to the Viture script is a one-liner — it just needs the same `K_arr` + `c2w` + `DW`/`DH` plumbing.

### 7.4 Knobs (CLI, iPhone script)

```
--objects-distance-threshold       0.4    # 3D match radius (m)
--objects-pixel-threshold          60     # 2D fallback radius (px)
--objects-disable-class-aware             # don't require name match
--objects-disable-decay                   # never delete due to non-observation
--objects-confidence-init          0.5
--objects-confidence-up            0.10
--objects-confidence-down          0.05
```

Stats line shows `match(t/d/p)=X/Y/Z decayed=N` so you can see which matcher is doing the work.

### 7.5 YOLOE weights

`LocalYoloeDetector` ([iphone script:539+](../../mac_iphone_spatial_foxglove.py#L539)) bypasses dimos's LFS-managed model archive and lets Ultralytics auto-download the `yoloe-11s-seg-pf.pt` checkpoint into `dimos/checkpoints/` on first use. `*.pt` is gitignored — never commit weights.

---

## 8. Save / load

[`mac_iphone_spatial_foxglove.py:save_map_bundle`](../../mac_iphone_spatial_foxglove.py)

A "map bundle" is a single pickled dict:

```python
{
    "version": 1,
    "saved_at": float,            # unix seconds
    "voxel_map": {                # from VoxelMap.to_state()
        "voxel_size": float, "max_range": float,
        "keys": int64[N,3], "centroids": float32[N,3],
        "colors": float32[N,3], "confidence": float32[N],
        "count": int32[N], "miss_count": int32[N],
    },
    "object_db": {                # from ObjectDB.to_state()
        "pending": dict[str, Object],
        "permanent": dict[str, Object],
        "track_id_map": dict[int, str],
        "confidence": dict[str, float],
        "config": {...},          # restored on load is OPTIONAL — caller wins
    },
    "extra": {...},               # session-specific notes
}
```

CLI:

- `--save-map PATH` — writes the bundle on graceful exit (Ctrl+C or end-of-video).
- `--save-map-every-n N` — periodic save in addition to on-exit; useful for crashes.
- `--load-map PATH` — preloads voxels + objects before the main loop.

Pickle is used (not npz) because `Object` instances carry nested `PointCloud2` / `Vector3` / open3d objects. The `to_state()` shape on `VoxelMap` is **already npz-friendly** (fixed-shape arrays), so an alternative npz-only voxel-only saver is straightforward to add if needed for portability.

**Frame-of-reference caveat**: a loaded map is in the **prior session's world frame** (anchored at that session's first VO step). Loading and continuing only aligns when the new VO session starts at the same physical pose — for example, replaying the same clip from the start. Cross-session re-localization (ICP against the loaded voxels) is not implemented yet; see §11.

---

## 9. Foxglove output

LCM transports created in `main()`. Pair the script with the foxglove bridge in another terminal:

```
/opt/anaconda3/envs/xr-nav/bin/python -m \
  dimos.utils.cli.foxglove_bridge.run_foxglove_bridge
```

| Topic                  | Type            | Frame             | Purpose                                                  |
|------------------------|-----------------|-------------------|----------------------------------------------------------|
| `/color_image`         | `Image`         | `camera_optical`  | Resampled color, BGR                                     |
| `/camera_info`         | `CameraInfo`    | `camera_optical`  | Pinhole intrinsics for the resampled color               |
| `/depth`               | `Image`         | `camera_optical`  | Filtered metric depth (float32, m)                       |
| `/depth_camera_info`   | `CameraInfo`    | `camera_optical`  | Same intrinsics; required so Foxglove can colorize depth |
| `/annotations`         | `ImageAnnotations` | (image-frame)  | YOLOE bounding boxes overlaid on `/color_image`          |
| `/points_frame`        | `PointCloud2`   | `world`           | Per-frame strided cloud (debug)                          |
| `/map`                 | `PointCloud2`   | `world`           | Fused VoxelMap (republished every `--map-publish-every-n`)|
| `/object_clouds`       | `PointCloud2`   | `world`           | Aggregated point clouds of all tracked objects           |
| `/scene_update`        | `SceneUpdate`   | `world`           | Per-object cube + label primitives                       |
| `/tf`                  | `TFMessage`     | `world → camera_optical` | Camera pose                                       |

In the Foxglove 3D panel, pin **Fixed frame = world** and Display frame = world. Add image panels for `/color_image` (with annotations enabled, info `/camera_info`) and `/depth` (info `/depth_camera_info`).

---

## 10. Common run commands

The Viture script always needs `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` because PyTorch + OpenMP libs collide on macOS.

**Viture, recorded, DepthPro** (default):
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_spatial_foxglove.py
```

**Viture, live**:
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  /opt/anaconda3/envs/xr-nav/bin/python -u mac_viture_spatial_foxglove.py \
    --source live
```

**iPhone video, DepthPro + VO** (recommended default):
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  /opt/anaconda3/envs/xr-nav/bin/python -u mac_iphone_spatial_foxglove.py \
    --depth depthpro --pose vo --display-width 768 --max-fps 2
```

**iPhone, DA3** (fast but content-limited):
```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  /opt/anaconda3/envs/xr-nav/bin/python -u mac_iphone_spatial_foxglove.py \
    --depth da3 --da3-model da3-large --pose vo --display-width 768
```

**Save then resume**:
```
# build the map
... mac_iphone_spatial_foxglove.py --no-loop \
    --save-map ~/.dimos/maps/phx_living_room.pkl

# resume next session — must replay from the same start to align
... mac_iphone_spatial_foxglove.py \
    --load-map ~/.dimos/maps/phx_living_room.pkl \
    --save-map ~/.dimos/maps/phx_living_room.pkl
```

---

## 11. Known issues / gotchas

- **DA3-small produces empty maps on low-information content.** Watch for the auto-printed `[depth] WARNING: DA3 raw dynamic range is only X` line on first frame. Switch to `--da3-model da3-large` or `--depth depthpro`. See §5.2.
- **Depth-pro on M-series can swap if MPS cache is not freed.** `empty_cache()` is wired in [`DepthProEstimator.infer`](../../mac_iphone_spatial_foxglove.py); do not remove.
- **VO drifts on low-texture rooms.** If `pos=(+0.00, +0.00, +0.00)` after several frames despite camera motion, ORB isn't getting enough inliers. Try a different clip, raise `--display-width`, or pair with DepthPro instead of DA3.
- **iPhone defaults assume the wide camera (~62° HFOV).** Override with `--hfov-deg` for ultrawide (~106°) or 2× telephoto (~30°). Wrong HFOV will scale depth → world incorrectly.
- **Loaded maps don't relocalize.** A loaded bundle is in the prior session's world frame; the new VO starts at identity. Replaying the same clip from the start aligns naturally; arbitrary re-localization is future work.
- **Permanent objects from a saved bundle stay permanent on load.** They keep their last confidence value, so they begin decaying immediately if the new session looks at where they were and doesn't see them. That's usually what you want; if not, raise `--objects-confidence-init` or pass `--objects-disable-decay`.
- **Object class names from YOLOE LRPC are noisy.** "couch" vs. "sofa" vs. "loveseat" can flicker. Class-aware matching (default) prevents merging across labels but causes occasional duplicates. The pixel fallback recovers some of these because the bbox stays in the same place even when the label flips.

---

## 12. Future work hooks

When you extend this, here's where to plug in:

| Want to…                                      | Add it here                                                                                                                                |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Support a new camera (RealSense, Android, etc.)| New `FrameSource` subclass, mirror `VideoSource` shape; wire into the source factory in `main()`                                            |
| Replace the depth model                        | New `DepthEstimator` subclass; register in `make_depth_estimator`. Set `skip_edge_filter = True` if your output has stretched dynamic range |
| Loop closure / cross-session relocalization    | Run a startup ICP between the new VO's first-frame point cloud and the loaded VoxelMap; adjust `MonocularDepthVO._c2w` accordingly          |
| IMU-aided VO                                   | Extend `MonocularDepthVO.update` to take an optional gyro delta; preintegrate between frames; use as the PnP initial guess                  |
| Save bundles in npz form                       | `VoxelMap.to_state()` already returns npz-friendly arrays. Need a JSON sidecar for object metadata + per-object PLYs in a subdirectory      |
| Promote shared helpers to a library            | Move `arkit_c2w_to_opencv`, `make_camera_info`, `filter_depth_edges`, `_stable_color_from_id`, `build_scene_update_for_objects` into `xr_nav` and have both scripts import. Wait until at least a third script needs them |
| Add per-object visual descriptors              | Add an embedding field on `Object`; in `ObjectDB._match` add a fourth stage that compares CLIP/DINO embeddings of bbox crops within a wider distance threshold |
| Decay-aware Viture script                      | Build `K_arr` from the existing Viture `cam_info` and call `object_db.decay_unobserved` after `add_objects` — same shape as the iPhone wiring |

---

## 13. Glossary

- **c2w** — 4×4 camera-to-world transform. The camera origin's pose in world coords.
- **OpenCV optical frame** — X right, Y down, Z forward. The convention used everywhere downstream of the source-specific conversions.
- **Pending vs. permanent objects** — pending have been seen fewer than `min_detections_for_permanent` times; permanent are the published, agent-visible set.
- **Drift** — accumulated VO error. The same physical surface gets fused into slightly different voxel cells across frames.
- **Map bundle** — the pickled dict produced by `--save-map`; structure in §8.
