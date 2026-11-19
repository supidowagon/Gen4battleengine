#!/usr/bin/env bash
# PC SIDE (Git Bash), once per new pod. Sends tools/ops/day/bootstrap.sh (LF) with a detached
# `runpodctl send` and prints the exec-pod body that receives it into a fresh /work/_bin/new,
# runs it there (log /work/_bin/new/bootstrap.log) and, only after BOOTSTRAP_OK, moves the script
# to /work/_bin/bootstrap.sh and the log to /work/bootstrap.log and removes /work/_bin/new. A
# failed bootstrap leaves the earlier install and log untouched, so the step is safe to repeat on
# a pod that is already up. Per-run scripts (tools/ops/day/pod/) travel inside each payload
# (pack.sh), not here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
B="$RENT/bringup"; rm -rf "$B"; mkdir -p "$B"; cd "$B"
tr -d '\r' < "$HERE/bootstrap.sh" > bootstrap.sh
HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send bootstrap.sh > send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true)
[ -n "$CODE" ] || { echo "FAILED: see $B/send.log"; exit 4; }
echo "== exec-pod body:"
N=/work/_bin/new
echo "{\"command\":[\"sh\",\"-c\",\"rm -rf $N && mkdir -p $N && cd $N && /work/_bin/runpodctl receive $CODE && nohup sh -c 'bash bootstrap.sh > bootstrap.log 2>&1; tail -1 bootstrap.log | grep -qx BOOTSTRAP_OK && mv -f bootstrap.sh /work/_bin/bootstrap.sh && mv -f bootstrap.log /work/bootstrap.log && cd /work/_bin && rm -rf $N' > /dev/null 2>&1 < /dev/null & sleep 15; tail -2 $N/bootstrap.log 2>/dev/null || tail -2 /work/bootstrap.log\"],\"timeoutSeconds\":25}"
