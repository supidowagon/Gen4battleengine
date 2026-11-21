#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: podcheck_body.sh
# Prints the exec-pod body of DAYBOX.md "Send a run" step 3: it writes tools/ops/day/pod/gate.sh
# and podcheck.sh, as committed at this checkout's HEAD, to /work/_podcheck/ on the pod and runs
# podcheck.sh /work <POD_SCRIPTS> <POD_WEIGHTS> of HEAD (common.sh pod_bin, pod_sum, pod_wsum: the
# same files and sums pack.sh writes into meta.env). Its last line is PODCHECK OK or PODCHECK
# MIXED old=... other=... (pod/podcheck.sh). Sends nothing; the scripts travel base64-encoded
# inside the body, in argv items of at most 4000 characters (common.sh exec_body: exec-pod's
# limits). Refuses like pack.sh (common.sh pod_main_check): run it from a checkout at main.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
TOP=$(git -C "$HERE" rev-parse --show-toplevel); pod_main_check "$TOP"
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
mkdir -p "$RENT"; T=$(mktemp -d "$RENT/podcheck.XXXXXX"); trap 'rm -rf "$T"' EXIT
pod_bin "$TOP" HEAD "$T/bin"
SUM=$(pod_sum "$T/bin"); WSUM=$(pod_wsum "$T/bin")
BODY=$(exec_body "bash \$W/podcheck.sh /work $SUM $WSUM" "gate.sh=$T/bin/gate.sh" "podcheck.sh=$T/bin/podcheck.sh") || { echo "$BODY"; exit 3; }
echo "== exec-pod body (pod scripts of HEAD $(git -C "$TOP" rev-parse --short HEAD), POD_SCRIPTS ${SUM:0:12}):"
echo "$BODY"
