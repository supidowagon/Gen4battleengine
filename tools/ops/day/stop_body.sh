#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: stop_body.sh <run>
# Prints the exec-pod body of DAYBOX.md, Housekeeping, "Stop a run": it writes
# tools/ops/day/pod/stop.sh, as committed at this checkout's HEAD, to /work/_podcheck/ and runs
# stop.sh <run> there (so it also stops runs packed before stop.sh existed). Last line of the
# output: STOP_OK <run> | STOP_LEFT <pids> | STOP_REFUSED ... (pod/stop.sh). Sends nothing.
# Refuses like pack.sh (common.sh pod_main_check): run it from a checkout at main.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
[ $# -eq 1 ] || { echo "usage: stop_body.sh <run>"; exit 2; }
check_run_name "$1"
TOP=$(git -C "$HERE" rev-parse --show-toplevel); pod_main_check "$TOP"
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
S=$(git -C "$TOP" -c core.autocrlf=false show HEAD:tools/ops/day/pod/stop.sh | tr -d '\r' | base64 -w0); W=/work/_podcheck
echo "== exec-pod body (stop.sh of HEAD $(git -C "$TOP" rev-parse --short HEAD)):"
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p $W && echo $S | base64 -d > $W/stop.sh && bash $W/stop.sh $1\"],\"timeoutSeconds\":25}"
