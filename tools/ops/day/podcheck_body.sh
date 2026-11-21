#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: podcheck_body.sh
# Prints the exec-pod body of DAYBOX.md "Send a run" step 3: it writes tools/ops/day/pod/gate.sh
# and podcheck.sh, as committed at this checkout's HEAD, to /work/_podcheck/ on the pod and runs
# podcheck.sh /work <POD_SCRIPTS of HEAD> (common.sh pod_bin and pod_sum, the same files and sum
# pack.sh writes into meta.env). Its last line is PODCHECK OK or PODCHECK MIXED old=... other=...
# (pod/podcheck.sh). Sends nothing; the scripts travel base64-encoded inside the body.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
TOP=$(git -C "$HERE" rev-parse --show-toplevel)
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
mkdir -p "$RENT"; T=$(mktemp -d "$RENT/podcheck.XXXXXX"); trap 'rm -rf "$T"' EXIT
pod_bin "$TOP" HEAD "$T/bin"
SUM=$(pod_sum "$T/bin")
G=$(base64 -w0 "$T/bin/gate.sh"); C=$(base64 -w0 "$T/bin/podcheck.sh"); W=/work/_podcheck
echo "== exec-pod body (pod scripts of HEAD $(git -C "$TOP" rev-parse --short HEAD), POD_SCRIPTS ${SUM:0:12}):"
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p $W && echo $G | base64 -d > $W/gate.sh && echo $C | base64 -d > $W/podcheck.sh && bash $W/podcheck.sh /work $SUM\"],\"timeoutSeconds\":25}"
