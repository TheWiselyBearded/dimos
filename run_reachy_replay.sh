#!/usr/bin/env bash
# Replay a recorded Reachy Mini session through the dimos spatial-memory
# pipeline + Foxglove.
#
# Spawns two Terminal windows:
#   1. dimos foxglove LCM bridge (port 8765)  -> /color_image /depth /map /tf ...
#   2. reachy_replay_spatial_foxglove.py       -> loads dimos pipeline + sidecar
#                                                 foxglove on :8766 with
#                                                 /reachy/imu /reachy/joints
#                                                 /reachy/head_pose /reachy/doa
#                                                 /reachy/audio/level
#
# Connect Foxglove Studio to both ws://localhost:8765 and ws://localhost:8766
# (add a second WebSocket connection in Studio).
#
# Usage:
#   ./run_reachy_replay.sh                            # default recording
#   ./run_reachy_replay.sh /path/to/another/trial     # other recording
#   ./run_reachy_replay.sh --depth depthpro           # passes through to dimos
#
# Env overrides:
#   PYTHON=/path/to/python   interpreter (default: xr-nav conda env)
#   BRIDGE_PORT=8765         dimos LCM->foxglove bridge port
#   SIDECAR_PORT=8766        reachy sidecar foxglove port
#   RECORDING_DIR=/path      default recording directory

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/opt/anaconda3/envs/xr-nav/bin/python}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
SIDECAR_PORT="${SIDECAR_PORT:-8766}"
DEFAULT_RECORDING="${RECORDING_DIR:-/Users/reza/Downloads/reachy_recordings/reachy_trial4}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found or not executable: $PY" >&2
    echo "Set PYTHON=/path/to/python or edit run_reachy_replay.sh." >&2
    exit 1
fi

# Allow the first positional arg to be a recording directory; anything else
# (or all args, if the first one starts with '-') is passed through to the
# replay script as dimos-pipeline knobs.
RECORDING="$DEFAULT_RECORDING"
if [[ $# -gt 0 && "${1:-}" != -* && -d "$1" ]]; then
    RECORDING="$1"
    shift
fi

if [[ ! -d "$RECORDING" ]]; then
    echo "ERROR: recording directory not found: $RECORDING" >&2
    echo "Pass a path as the first arg, set RECORDING_DIR, or edit this script." >&2
    exit 1
fi

EXTRA_ARGS=("$@")

bridge_already_up() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$BRIDGE_PORT" -sTCP:LISTEN >/dev/null 2>&1
    else
        return 1
    fi
}

open_terminal_window() {
    local cmdline="$1"
    osascript <<EOF
tell application "Terminal"
    activate
    do script "${cmdline//\"/\\\"}"
end tell
EOF
}

if bridge_already_up; then
    echo "[bridge] already listening on :$BRIDGE_PORT — reusing"
else
    echo "[bridge] launching dimos foxglove bridge on :$BRIDGE_PORT ..."
    BRIDGE_CMD="cd '$REPO' && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 '$PY' -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge"
    open_terminal_window "$BRIDGE_CMD"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 0.5
        bridge_already_up && break
    done
fi

ARGS_QUOTED=""
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    ARGS_QUOTED+=" $(printf '%q' "$arg")"
done

echo "[replay] launching reachy_replay_spatial_foxglove.py ..."
echo "         recording: $RECORDING"
echo "         sidecar  : ws://localhost:$SIDECAR_PORT"
REPLAY_CMD="cd '$REPO' && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 '$PY' -u reachy_replay_spatial_foxglove.py --recording-dir $(printf '%q' "$RECORDING") --sidecar-port $SIDECAR_PORT$ARGS_QUOTED"
open_terminal_window "$REPLAY_CMD"

cat <<EOF

==[ reachy replay pipeline ]==========================================
Foxglove Studio: https://app.foxglove.dev (or local app)
  primary : ws://localhost:$BRIDGE_PORT   (dimos LCM bridge)
            /color_image /depth /points_frame /map /object_clouds
            /scene_update /tf /annotations
  sidecar : ws://localhost:$SIDECAR_PORT   (reachy streams)
            /reachy/imu /reachy/imu/temperature /reachy/head_pose
            /reachy/joints /reachy/doa /reachy/audio/level /reachy/playback

Both processes run in their own Terminal windows; close them or Ctrl+C
in each to stop.
======================================================================
EOF
