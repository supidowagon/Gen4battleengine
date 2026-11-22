#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: podcheck_body.sh
# Prints the exec-pod body of DAYBOX.md "Send a run" step 3: it receives tools/ops/day/pod/gate.sh
# and podcheck.sh, as committed at this checkout's HEAD, into its own /work/_podcheck/check-<code word>/ on the
# pod and runs podcheck.sh /work <POD_SCRIPTS> <POD_WEIGHTS> <quota GB> <warn GB> of HEAD (common.sh pod_bin, pod_sum, pod_wsum: the
# same files and sums pack.sh writes into meta.env). Its last line is PODCHECK OK or PODCHECK
# MIXED old=... other=... (pod/podcheck.sh), or PODCHECK REFUSED vm.max_map_count=<value> (see
# MAPCHECK below). The two scripts travel as podcheck.tar by a detached `runpodctl send`
# ($RENT/podcheck/<epoch>-<pid>/, send.log ends SEND_RC=<n>; folders of finished sends are removed
# by the next call); the body receives it, checks its sha256 and runs podcheck.sh only on a match
# (send_body.sh tool_body: RECEIVE_FAILED, PAYLOAD_SHA_MISMATCH, UNPACK_FAILED, PENDING).
# Refuses like pack.sh (common.sh pod_main_check): run it from a checkout at main.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"; . "$HERE/send_body.sh"
TOP=$(git -C "$HERE" rev-parse --show-toplevel); pod_main_check "$TOP"
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || { echo "REFUSED: uncommitted changes in tools/ops/day/pod; commit them first"; echo "$DIRTY"; exit 3; }
mkdir -p "$RENT/podcheck"
for d in "$RENT"/podcheck/*/; do grep -q '^SEND_RC=' "$d/send.log" 2>/dev/null && rm -rf "$d"; done
T="$RENT/podcheck/$(date +%s)-$$"; mkdir "$T"; trap '[ -n "${CODE:-}" ] || rm -rf "$T"' EXIT
pod_bin "$TOP" HEAD "$T/bin"
SUM=$(pod_sum "$T/bin"); WSUM=$(pod_wsum "$T/bin")
# A pod whose vm.max_map_count is under 1048576 (65530 aborts JAX units, rc 134) or unreadable is
# refused before podcheck.sh runs: PODCHECK REFUSED vm.max_map_count=<value>,
# exit 1. Only its digits are read (tr -dc 0-9), so ${m:-0} is one word (an empty value tests as 0).
MAPCHECK="m=\$(tr -dc 0-9 < /proc/sys/vm/max_map_count); [ \${m:-0} -ge 1048576 ] || { echo PODCHECK REFUSED vm.max_map_count=\$m, need 1048576 or more; exit 1; }"
# The pod check's volume line (GB = 10^9 bytes): the quota CACHE_VOL_QUOTA_GB on the PC, default 30
# (the volume's size, DAYBOX.md "Network volume" step 1; the pod cannot read it), and the warning
# threshold CACHE_FREE_WARN_GB, default 3 (the same knob as cache_sync.sh's prune).
QGB="${CACHE_VOL_QUOTA_GB:-30}"; [[ "$QGB" =~ ^[1-9][0-9]*$ ]] || { echo "REFUSED: CACHE_VOL_QUOTA_GB '$QGB' is not a whole number of GB"; exit 3; }
FWG="${CACHE_FREE_WARN_GB:-3}"; [[ "$FWG" =~ ^[0-9]+$ ]] || { echo "REFUSED: CACHE_FREE_WARN_GB '$FWG' is not a whole number of GB"; exit 3; }
# check.sh (in the tar, so under its sha256, not in the body): MAPCHECK, then podcheck.sh with HEAD's sums.
printf '%s
' "$MAPCHECK" "exec bash podcheck.sh /work $SUM $WSUM $QGB $FWG" > "$T/bin/check.sh"
tar -cf "$T/podcheck.tar" -C "$T/bin" gate.sh podcheck.sh check.sh
PSUM=$(sha256sum "$T/podcheck.tar" | cut -c1-64)
send_start "$T" podcheck.tar 600 || { CODE=""; exit 4; }
BODY=$(body_json "$(tool_body check podcheck.tar "$PSUM" "$CODE" "bash check.sh")") || { kill "$SPID" 2>/dev/null || true; echo "$BODY"; exit 3; }
echo "== exec-pod body (pod scripts of HEAD $(git -C "$TOP" rev-parse --short HEAD), POD_SCRIPTS ${SUM:0:12}, ${#BODY} characters):"
echo "$BODY"
