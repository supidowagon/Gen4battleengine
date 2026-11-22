#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: stop_body.sh <run>
# Prints the exec-pod body of DAYBOX.md, Housekeeping, "Stop a run": it receives
# tools/ops/day/pod/stop.sh, as committed at this checkout's HEAD, into a fresh
# /work/_podcheck/stop/ and runs stop.sh <run> there (so it also stops runs packed before stop.sh
# existed). Last line of the output: STOP_OK <run> | STOP_LEFT <pids> | STOP_REFUSED ...
# (pod/stop.sh). stop.sh travels as stop.tar by a detached `runpodctl send` ($RENT/podcheck/, as
# podcheck_body.sh); the body checks its sha256 before it runs (send_body.sh tool_body).
# Refuses like pack.sh (common.sh pod_main_check): run it from a checkout at main.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"; . "$HERE/send_body.sh"
[ $# -eq 1 ] || { echo "usage: stop_body.sh <run>"; exit 2; }
check_run_name "$1"
TOP=$(git -C "$HERE" rev-parse --show-toplevel); pod_main_check "$TOP"
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
mkdir -p "$RENT/podcheck"
for d in "$RENT"/podcheck/*/; do grep -q '^SEND_RC=' "$d/send.log" 2>/dev/null && rm -rf "$d"; done
T="$RENT/podcheck/$(date +%s)-$$"; mkdir "$T"; trap '[ -n "${CODE:-}" ] || rm -rf "$T"' EXIT
git -C "$TOP" -c core.autocrlf=false show HEAD:tools/ops/day/pod/stop.sh | tr -d '\r' > "$T/stop.sh"
tar -cf "$T/stop.tar" -C "$T" stop.sh
PSUM=$(sha256sum "$T/stop.tar" | cut -c1-64)
send_start "$T" stop.tar || { CODE=""; exit 4; }
BODY=$(body_json "$(tool_body stop stop.tar "$PSUM" "$CODE" "bash stop.sh $1")") || { kill "$SPID" 2>/dev/null || true; echo "$BODY"; exit 3; }
echo "== exec-pod body (stop.sh of HEAD $(git -C "$TOP" rev-parse --short HEAD), ${#BODY} characters):"
echo "$BODY"
