# Windows (WSL2) Setup

Native Windows **cannot** build dimos: `pip install -e .` pulls Pinocchio (`pin`), whose
`cmeel-boost` dependency compiles Boost from source and fails on Windows
(`'.\b2' is not recognized`, MSBuild error `MSB8066` exit `9009`). On Linux those install as
prebuilt `manylinux` wheels with no compilation, so **run dimos under WSL2 (Ubuntu)**.

This guide goes from a clean Windows machine to a working spatial pipeline + Foxglove.

> Tested on: Windows 11, WSL2 Ubuntu, NVIDIA RTX 4090, conda env `xr-nav` (Python 3.12, CUDA PyTorch).

---

## 1. Initial WSL2 setup (run on the Windows side)

1. Install WSL2 + Ubuntu in an **admin** PowerShell, then reboot if prompted and launch
   "Ubuntu" once to create your Linux user:
   ```powershell
   wsl --install -d Ubuntu
   ```

2. **GPU passthrough:** install the latest NVIDIA *Windows* driver (it ships the WSL CUDA
   stub — you do **not** install a CUDA toolkit inside WSL; conda provides the runtime).
   Verify inside WSL:
   ```bash
   nvidia-smi      # should list your GPU
   ```

3. **Work in the Linux filesystem** (`~`, i.e. `/home/<you>`), **not** under `/mnt/c`.
   `/mnt/c` is slow across the VM boundary and re-triggers the Pinocchio source build.
   Everything below lives in `~`.

## 2. GitHub auth in WSL (for the private `xr-nav` submodule)

dimos vendors the private `xr-nav` repo as a git submodule, so git needs credentials. The
simplest option is a Personal Access Token (PAT) with `repo` scope:

```bash
git config --global credential.helper store
# create a PAT at https://github.com/settings/tokens, then:
printf 'https://x-access-token:YOUR_TOKEN_HERE@github.com\n' > ~/.git-credentials
chmod 600 ~/.git-credentials
```
(SSH keys or the GitHub CLI work too.)

## 3. Clone the repo + submodule

```bash
cd ~
git clone --recurse-submodules https://github.com/TheWiselyBearded/dimos.git
cd dimos
# already cloned without submodules?
git submodule update --init --recursive
```

## 4. Install Miniforge (conda)

```bash
curl -fsSL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec bash      # reload the shell so `conda` is on PATH
```

## 5. Create the conda env

```bash
cd ~/dimos/xr-nav
conda env create -f environment.yml   # env `xr-nav`: Python 3.12 + CUDA PyTorch + xr-nav (editable)
conda activate xr-nav
```

### Required: libstdc++ activation hook
Some pip wheels (PyTorch's `optree`, etc.) need a newer `libstdc++` than Ubuntu ships, so make
conda's copy load first. Set it once in the env's activation hook:
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' \
  > "$CONDA_PREFIX/etc/conda/activate.d/zz_ldpath.sh"
conda deactivate && conda activate xr-nav   # apply now
```
Skipping this causes `ImportError: ... version 'CXXABI_1.3.15' not found`.

## 6. Install dimos

```bash
cd ~/dimos
pip install -e .
```
`pin` (Pinocchio) and its `cmeel-*` / `coal` / `eigenpy` deps install as prebuilt Linux wheels —
no Boost compilation. Sanity check:
```bash
python -c "import dimos, xr_nav, pinocchio, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 7. Depth models

The spatial pipeline needs a depth backend. Install one or both.

### DepthPro (Apple) — sharp metric depth, recommended
```bash
pip install --no-deps "git+https://github.com/apple/ml-depth-pro.git"
pip install timm pillow_heif          # note: pulls a newer pip PyTorch; the §5 hook handles its libstdc++
mkdir -p ~/dimos/checkpoints
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('apple/DepthPro','depth_pro.pt', local_dir='$HOME/dimos/checkpoints')"
```

### Depth Anything 3 (ByteDance) — Linux/Windows path
The spatial scripts load DA3 from `xr-nav/awesome-depth-anything-3` on macOS (a fork) and from
`xr-nav/Depth-Anything-3` (the official repo) on Linux/Windows. Clone the official repo there:
```bash
cd ~/dimos/xr-nav
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
# runtime deps only -- do NOT `pip install` the package itself (it pins numpy<2 and breaks the env):
pip install omegaconf einops imageio trimesh plyfile "moviepy==1.0.3" pycolmap evo
```
Weights download automatically from HuggingFace on first use.
> dimos loads DA3 via `DepthAnything3.from_pretrained(...)` — the bare `DepthAnything3(model_name=...)`
> constructor only builds the architecture (no weights) and yields flat/garbage depth. Use a dimos
> revision that includes the `from_pretrained` fix.

## 8. Object detection (optional — 2D/3D boxes + segmentation)

```bash
pip install langchain_core ultralytics lapx
```
YOLOE weights (`yoloe-11s-seg-pf.pt`) auto-download to `dimos/checkpoints/` on first run.
Skip this and pass `--no-detect` if you don't need detection.

## 9. Run the spatial pipeline

The `run_*.sh` launchers are macOS-only (they spawn Terminal windows via `osascript`). On WSL,
start the two processes directly, in two shells:

```bash
conda activate xr-nav                 # sets LD_LIBRARY_PATH via the activate hook
cd ~/dimos
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# Shell 1 -- dimos -> Foxglove LCM bridge (:8765)
python -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge

# Shell 2 -- Reachy replay (DepthPro shown). Recording is on the Windows side via /mnt/c.
python -u reachy_replay_spatial_foxglove.py \
  --recording-dir /mnt/c/Users/<you>/Downloads/reachy_trial4 \
  --sidecar-port 8766 --device cuda \
  --depth depthpro --display-width 1024 --extra --points-stride 2
#   DA3 metric instead:  --depth da3 --da3-model da3metric-large
#   skip detection:      add --no-detect   (must come before --extra)
```
Flags worth knowing:
- `--device cuda` (the scripts default to `mps` for Mac).
- `--display-width 1024 --points-stride 2` give a denser cloud / more detail than the defaults
  (`--extra` forwards `--points-stride` to the inner pipeline and must be the **last** argument).

## 10. Connect Foxglove Studio

Add **two** WebSocket connections:
- `ws://localhost:8765` — `/color_image` `/depth` `/map` `/points_frame` `/tf`
- `ws://localhost:8766` — Reachy sidecar (`/reachy/imu` `/reachy/joints` `/reachy/head_pose` `/reachy/doa` `/reachy/audio/level`)

To see detections: enable `/annotations` under an Image panel's *Image annotations*, and toggle
`/scene_update` + `/object_clouds` in the 3D panel.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'.\b2' is not recognized` / Boost build fails | You're on native Windows — use WSL (this guide). |
| `ImportError: ... 'CXXABI_1.3.15' not found` | Add the `LD_LIBRARY_PATH` activation hook (§5). |
| `ModuleNotFoundError: No module named 'xr_nav.cli_args'` / `map_io` | Update the `xr-nav` submodule to its `main` (`git -C xr-nav checkout main && git -C xr-nav pull`). |
| `ModuleNotFoundError: No module named 'depth_anything_3'` | Clone the DA3 repo into `xr-nav/Depth-Anything-3` (§7). |
| `ModuleNotFoundError: No module named 'langchain_core'` | Install detection deps (§8), or run with `--no-detect`. |
| DA3 depth looks flat/blocky | Use a dimos revision with the `from_pretrained` weight-load fix (§7). |
| Foxglove "UDP receive buffer is very small" | `sudo sysctl -w net.core.rmem_max=33554432` (otherwise non-fatal). |
| Open3D/Tkinter GUI windows don't open | `sudo apt install -y libgl1 libglib2.0-0` (WSLg usually covers this). |