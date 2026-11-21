#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: podcheck_body.sh
# Prints the exec-pod body of DAYBOX.md "Send a run" step 3: it writes tools/ops/day/pod/gate.sh
# and podcheck.sh, as committed at this checkout's HEAD, to /work/_podcheck/ on the pod and runs
# podcheck.sh /work <POD_SCRIPTS> <POD_WEIGHTS> of HEAD (common.sh pod_bin, pod_sum, pod_wsum: the
# same files and sums pack.sh writes into meta.env). Its last line is PODCHECK OK or PODCHECK
# MIXED old=... other=... (pod/podcheck.sh), or PODCHECK REFUSED vm.max_map_count=<value> (see
# MAPCHECK below). Sends nothing; the scripts travel base64-encoded inside the body, in argv items of at most 4000 characters (common.sh exec_body: exec-pod's
# limits). Refuses like pack.sh (common.sh pod_main_check): run it from a checkout at main.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
TOP=$(git -C "$HERE" rev-parse --show-toplevel); pod_main_check "$TOP"
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
mkdir -p "$RENT"; T=$(mktemp -d "$RENT/podcheck.XXXXXX"); trap 'rm -rf "$T"' EXIT
pod_bin "$TOP" HEAD "$T/bin"
SUM=$(pod_sum "$T/bin"); WSUM=$(pod_wsum "$T/bin")
# A pod whose vm.max_map_count is under 1048576 (65530 aborts JAX units, rc 134) or unreadable is
# refused before podcheck.sh runs: PODCHECK REFUSED vm.max_map_count=<value>, exit 1. Only its digits
# are read (tr -dc 0-9), so ${m:-0} is one word without quotes (exec_body refuses double quotes;
# an empty value tests as 0).
MAPCHECK="m=\$(tr -dc 0-9 < /proc/sys/vm/max_map_count); [ \${m:-0} -ge 1048576 ] || { echo PODCHECK REFUSED vm.max_map_count=\$m, need 1048576 or more; exit 1; }"
BODY=$(exec_body "$MAPCHECK; bash \$W/podcheck.sh /work $SUM $WSUM" "gate.sh=$T/bin/gate.sh" "podcheck.sh=$T/bin/podcheck.sh") || { echo "$BODY"; exit 3; }
echo "== exec-pod body (pod scripts of HEAD $(git -C "$TOP" rev-parse --short HEAD), POD_SCRIPTS ${SUM:0:12}):"
echo "$BODY"
