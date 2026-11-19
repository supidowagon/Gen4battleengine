#!/usr/bin/env bash
# PC SIDE (Git Bash), once per new pod. Sends tools/ops/day/bootstrap.sh (LF) with a detached
# `runpodctl send` and prints the exec-pod body that installs it into /work/_bin and starts it.
# Per-run scripts (tools/ops/day/pod/) travel inside each payload (pack.sh), not here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO="${NUZLINE_REPO:-$(cd "$HERE/../../.." && pwd)}"
RENT="${NUZLINE_SCRATCH:-$REPO/_scratch}/rent"
RPC="${RUNPODCTL:-$RENT/bin/runpodctl-windows-amd64.exe}"
B="$RENT/bringup"; rm -rf "$B"; mkdir -p "$B"; cd "$B"
tr -d '\r' < "$HERE/bootstrap.sh" > bootstrap.sh
HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send bootstrap.sh > send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r')
[ -n "$CODE" ] || { echo "FAILED: see $B/send.log"; exit 4; }
echo "== exec-pod body:"
echo "{\"command\":[\"sh\",\"-c\",\"cd /work/_bin && ./runpodctl receive $CODE && nohup bash bootstrap.sh > /work/bootstrap.log 2>&1 < /dev/null & sleep 15; tail -2 /work/bootstrap.log\"],\"timeoutSeconds\":25}"
