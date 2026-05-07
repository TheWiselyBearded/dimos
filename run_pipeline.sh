#!/usr/bin/env bash
# Orchestrator. Two modes:
#
#   GUI (default — run with no args):
#     ./run_pipeline.sh
#       Opens the Tkinter launcher (run_pipeline_gui.py). Pick mode, dataset,
#       depth model, save/load options via buttons; click Launch and the GUI
#       re-invokes this script with --headless + the chosen args.
#
#   Headless (used by the GUI's Launch button, also usable directly):
#     ./run_pipeline.sh --headless [--save] [--mode video] ...
#       Spawns the foxglove bridge (or reuses one already on :8765) and the
#       camera pipeline in two macOS Terminal.app windows. All args after
#       --headless are passed to run_camera_pipeline.py.
#
# Env var overrides:
#   PYTHON=/path/to/python    interpreter (default: xr-nav conda env)
#   BRIDGE_PORT=8765          foxglove bridge port

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/opt/anaconda3/envs/xr-nav/bin/python}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found or not executable: $PY" >&2
    echo "Set PYTHON=/path/to/python in the environment, or edit run_pipeline.sh." >&2
    exit 1
fi

# ─── GUI mode ─────────────────────────────────────────────────────────────
# When the first arg isn't --headless we open the Tk launcher and exit. The
# GUI's Launch button calls back into this script with --headless + args.
if [[ $# -eq 0 || "${1:-}" != "--headless" ]]; then
    if [[ ! -f "$REPO/run_pipeline_gui.py" ]]; then
        echo "ERROR: run_pipeline_gui.py not found. To run headless, pass --headless." >&2
        exit 1
    fi
    exec "$PY" "$REPO/run_pipeline_gui.py" "$@"
fi

# ─── Headless mode ────────────────────────────────────────────────────────
shift  # drop --headless

# Compose the args string for AppleScript. Quote each arg so spaces survive.
ARGS_QUOTED=""
for arg in "$@"; do
    ARGS_QUOTED+=" $(printf '%q' "$arg")"
done

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
    echo "[bridge] launching foxglove bridge in new Terminal window..."
    BRIDGE_CMD="cd '$REPO' && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 '$PY' -m dimos.utils.cli.foxglove_bridge.run_foxglove_bridge"
    open_terminal_window "$BRIDGE_CMD"
    # Give the bridge a moment to bind the socket before the pipeline starts
    # publishing on it.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 0.5
        bridge_already_up && break
    done
fi

echo "[pipeline] launching run_camera_pipeline.py in new Terminal window..."
PIPELINE_CMD="cd '$REPO' && '$PY' -u run_camera_pipeline.py$ARGS_QUOTED"
open_terminal_window "$PIPELINE_CMD"

echo
echo "Foxglove: open https://app.foxglove.dev and connect to ws://localhost:$BRIDGE_PORT"
echo "Both processes run in their own Terminal windows; close them or Ctrl+C to stop."
