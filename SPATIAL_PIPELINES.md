# Running the Spatial Pipelines (Foxglove / Rerun)

How to run the fork's `mac_*` / `reachy_replay` spatial pipelines, and how to
visualize them in **Foxglove**, **Rerun**, or **both**.

All commands assume the `xr-nav` conda env (where `dimos` is installed editable):

```bash
conda activate xr-nav
# or prefix any command with: /opt/anaconda3/envs/xr-nav/bin/python
```

The shell wrappers and examples below set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`
(needed on macOS to avoid an OpenMP duplicate-runtime abort).

---

## 1. The `--viz` flag

Three scripts take a `--viz` backend selector (default `foxglove`, so existing
behavior is unchanged):

| Script | Role | `--viz`? |
|---|---|---|
| `reachy_replay_spatial_foxglove.py` | **Reachy replay** (recorded session) | ✅ |
| `mac_iphone_spatial_foxglove.py` | iPhone / generic video pipeline (also the engine the reachy script drives) | ✅ |
| `mac_viture_spatial_foxglove.py` | **Viture XR glasses**, live or recorded | ✅ |
| `mac_viture_{da3,depthpro,stereo,ply}_foxglove.py` | specialized Viture variants | ❌ Foxglove-only (extendable) |

```
--viz foxglove   # publish LCM topics for the dimos Foxglove bridge (DEFAULT)
--viz rerun      # log to a Rerun viewer via each message's to_rerun()
--viz both       # do both at once

--rerun-save out.rrd   # (rerun/both) write a .rrd recording headless instead of a viewer
--rerun-connect        # (rerun/both) attach to an already-running `rerun` viewer
```

**What you need running per backend:**

- **Foxglove** → start the bridge in a second terminal and open the Foxglove app at
  `ws://localhost:8765`:
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge
  ```
- **Rerun** → nothing extra; `--viz rerun` spawns a viewer automatically. To reuse a
  window across runs, launch `rerun` once and pass `--rerun-connect`. For headless,
  use `--rerun-save out.rrd` and later open it with `rerun out.rrd`.

---

## 2. Reachy replay (recorded session) — playback

### Easiest: shell wrapper (auto-opens the Foxglove bridge + replay in two terminals)

```bash
./run_reachy_replay.sh /Users/reza/Downloads/reachy_recordings/reachy_trial4
./run_reachy_replay.sh <recording_dir> --viz rerun          # args pass through
./run_reachy_replay_depthpro.sh <recording_dir>             # same, forces DepthPro
```

### Direct invocation

```bash
PY=/opt/anaconda3/envs/xr-nav/bin/python
ENV="KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1"

# Foxglove (default) — also run the bridge (section 1) in another terminal
env $ENV $PY -u reachy_replay_spatial_foxglove.py \
    --recording-dir <recording_dir>

# Rerun — spawns a Rerun viewer, no bridge needed
env $ENV $PY -u reachy_replay_spatial_foxglove.py \
    --recording-dir <recording_dir> --viz rerun

# Both at once (compare viewers side by side)
env $ENV $PY -u reachy_replay_spatial_foxglove.py \
    --recording-dir <recording_dir> --viz both

# Headless Rerun capture to a file
env $ENV $PY -u reachy_replay_spatial_foxglove.py \
    --recording-dir <recording_dir> --viz rerun --rerun-save reachy_trial4.rrd
```

### Useful replay flags (all combine with `--viz`)

```bash
--depth depthpro                 # DepthPro (most accurate); default is da3
--depth da3 --da3-model da3metric-large   # metric DA3 (cross-frame-consistent)
--pose external                  # use recorded head_pose.jsonl (vs vo / identity)
--max-fps 1.0                    # cap publish rate (depth model is usually the real ceiling)
--display-width 1024             # resample width before depth inference
--no-detect                      # skip 3D object detection (faster)
--no-loop                        # one pass through the recording, then exit
--save-map session.pkl           # on exit, save voxel map + tracked objects
--load-map session.pkl           # resume from a saved map
--save-ply session.ply           # export the fused cloud as PLY
```

The reachy **telemetry sidecar** (IMU / joints / head_pose / DOA, port 8766) always
publishes to Foxglove regardless of `--viz` — those streams aren't dimos
`to_rerun()` messages. Disable it with `--sidecar-port 0`.

---

## 3. Viture XR glasses — live and recorded

`mac_viture_spatial_foxglove.py` takes `--source {live,recording}`.

```bash
PY=/opt/anaconda3/envs/xr-nav/bin/python
ENV="KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1"

# LIVE from the glasses → Rerun
env $ENV $PY -u mac_viture_spatial_foxglove.py \
    --source live --depth depthpro --viz rerun

# LIVE → Foxglove (run the bridge too) / or both
env $ENV $PY -u mac_viture_spatial_foxglove.py --source live --depth depthpro            # foxglove
env $ENV $PY -u mac_viture_spatial_foxglove.py --source live --depth depthpro --viz both

# RECORDING playback → Rerun
env $ENV $PY -u mac_viture_spatial_foxglove.py \
    --source recording --video left.mp4 --depth depthpro --viz rerun

# Stereo depth from the two fisheye cameras (recording)
env $ENV $PY -u mac_viture_spatial_foxglove.py \
    --source recording --depth stereo --video left.mp4 --right-video right.mp4 --viz both
```

Same map-I/O and tuning flags as section 2 (`--save-map`, `--load-map`, `--save-ply`,
`--max-fps`, `--no-detect`, `--enable-clip-memory`, …).

> The specialized Viture variants (`mac_viture_da3_foxglove.py`,
> `…depthpro…`, `…stereo…`, `…ply…`) are currently **Foxglove-only** — run the
> bridge and omit `--viz`. They can be given the same `--viz` flag on request
> (they reuse `viz_backend.py`).

---

## 4. iPhone / generic video pipeline

```bash
PY=/opt/anaconda3/envs/xr-nav/bin/python
ENV="KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1"

# Play back a phone video → Rerun
env $ENV $PY -u mac_iphone_spatial_foxglove.py \
    --video clip.mov --depth da3 --da3-model da3metric-large --viz rerun

# → Foxglove (default; run the bridge) / both
env $ENV $PY -u mac_iphone_spatial_foxglove.py --video clip.mov              # foxglove
env $ENV $PY -u mac_iphone_spatial_foxglove.py --video clip.mov --viz both

# Accumulate a persistent fused cloud and save it
env $ENV $PY -u mac_iphone_spatial_foxglove.py \
    --video clip.mov --viz rerun --accumulate-cloud --save-ply room.ply
```

---

## 5. Common flags reference

**Depth**
```
--depth depthpro | da3 | stereo        # stereo only on the Viture script
--da3-model da3metric-large            # metric (consistent scale); also da3-small/base/large/giant, da3nested-giant-large
--device mps | cuda | cpu
--hfov-deg 70                          # source camera horizontal FOV
```

**Pose** (iphone / reachy)
```
--pose vo         # ORB + depth-PnP visual odometry (default)
--pose external   # use a supplied per-frame pose (Reachy head_pose.jsonl)
--pose identity   # camera fixed at origin (debug)
```

**Map I/O & export**
```
--save-map FILE.pkl   --load-map FILE.pkl   --save-map-every-n N
--save-ply FILE.ply   --save-pcd FILE.pcd   --save-cloud-with-map
--cloud-min-observations N
```

**Object tracking** (iphone pipeline)
```
--objects-distance-threshold M    --objects-class-aware
--objects-confidence-{init,up,down} X    --objects-disable-decay
--no-detect                       # skip 3D detection entirely
```

**Object center source** — set in code via `ObjectDB(center_from_accumulated_cloud=…)`:
- `True` (default) — center = accumulated-cloud centroid; **stable, for static-scene
  mapping / spatial memory** (Reachy replay).
- `False` — center = latest detection; tracks motion, for **live nav around moving
  objects**.

**Performance**
```
--max-fps F   --map-publish-every-n N   --points-stride N   --raycast-every-n N
```

---

## 6. Quick reference

| Goal | Command sketch |
|---|---|
| Reachy replay, Foxglove | `./run_reachy_replay.sh <dir>` (bridge auto-launched) |
| Reachy replay, Rerun | `… reachy_replay_spatial_foxglove.py --recording-dir <dir> --viz rerun` |
| Reachy replay, both | `… --recording-dir <dir> --viz both` |
| Viture live, Rerun | `… mac_viture_spatial_foxglove.py --source live --depth depthpro --viz rerun` |
| iPhone video, Rerun | `… mac_iphone_spatial_foxglove.py --video clip.mov --viz rerun` |
| Headless capture | add `--viz rerun --rerun-save out.rrd`, view later with `rerun out.rrd` |
| Foxglove bridge | `python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge` |

**Backends are independent:** `--viz foxglove` needs the bridge + Foxglove app;
`--viz rerun` needs nothing extra (viewer auto-spawns); `--viz both` runs the two
side by side from the same published messages.
