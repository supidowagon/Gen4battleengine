#!/usr/bin/env bash
# PC SIDE (Git Bash), morning bring-up only. Packs tools/ops/day/pod/*.sh and starts a detached
# `runpodctl send`; prints the exec-pod command that installs them and runs bootstrap.sh.
set -euo pipefail
: "${NUZLINE_SCRATCH:?set NUZLINE_SCRATCH (git-bash style path of the scratch folder)}"
: "${NUZLINE_REPO:?set NUZLINE_REPO}"
RENT="$NUZLINE_SCRATCH/rent"; DAY="$RENT/day"
cd "$DAY"; rm -f bin.tar bin_send.log
tar -cf bin.tar -C "$NUZLINE_REPO/tools/ops/day/pod" .
HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RENT/bin/runpodctl-windows-amd64.exe" send bin.tar > bin_send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" bin_send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" bin_send.log | head -1 | sed 's/code is: //' | tr -d '\r')
[ -n "$CODE" ] || { echo "FAILED: see $DAY/bin_send.log"; exit 4; }
echo "exec-pod on $(cat POD_ID):"
echo "  {\"command\": [\"sh\", \"-c\", \"cd /work/_bin && ./runpodctl receive $CODE && tar --no-same-owner -xf bin.tar && rm bin.tar && chmod +x *.sh && (nohup bash bootstrap.sh > /work/bootstrap.log 2>&1 &)\"], \"timeoutSeconds\": 20}"
