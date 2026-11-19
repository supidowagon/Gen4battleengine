#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: collect.sh <run> <code printed by the pod's fetch.sh>
# Receives results.tgz into $NUZLINE_SCRATCH/rent/out/<run>/ and unpacks it into results/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO="${NUZLINE_REPO:-$(cd "$HERE/../../.." && pwd)}"
RENT="${NUZLINE_SCRATCH:-$REPO/_scratch}/rent"
RPC="${RUNPODCTL:-$RENT/bin/runpodctl-windows-amd64.exe}"
P="$RENT/out/${1:?usage: collect.sh <run> <code>}"; mkdir -p "$P"; cd "$P"; rm -f results.tgz
HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive "${2:?code}" > recv.log 2>&1
rm -rf results; mkdir results; tar -xzf results.tgz -C results
cat results/agg.txt
