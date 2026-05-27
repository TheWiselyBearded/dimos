#!/usr/bin/env bash
# Same as run_reachy_replay.sh, but forces --depth depthpro instead of the
# default --depth da3 --da3-model da3-small. DepthPro is the more accurate
# (and slower) monocular depth model — ~2-4x slower than DA3-small per frame
# but produces metric depth with sharper edges.
#
# Usage mirrors run_reachy_replay.sh:
#   ./run_reachy_replay_depthpro.sh                                  # default recording
#   ./run_reachy_replay_depthpro.sh /path/to/other_trial             # other recording
#   ./run_reachy_replay_depthpro.sh --display-width 1024             # tune knobs
#   ./run_reachy_replay_depthpro.sh /path/to/trial --max-fps 1.0 --no-detect

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build args. If the first positional looks like a recording directory, keep
# it first so the parent script's "is $1 a dir?" heuristic still grabs it as
# the recording path (not as a value for some flag). Then inject the depthpro
# flag before user args so a user-supplied --depth wins (argparse keeps last).
ARGS=()
if [[ $# -gt 0 && "${1:-}" != -* && -d "$1" ]]; then
    ARGS+=("$1")
    shift
fi
ARGS+=(--depth depthpro)
ARGS+=("$@")

exec "$REPO/run_reachy_replay.sh" "${ARGS[@]}"
