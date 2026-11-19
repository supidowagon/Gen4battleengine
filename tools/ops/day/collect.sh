#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: collect.sh <run> <code printed by the pod's fetch.sh>
# Receives results.tgz into $RENT/out/<run>/ and unpacks it into results/.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/../common.sh"
RUN="${1:?usage: collect.sh <run> <code>}"; check_run_name "$RUN"
P="$RENT/out/$RUN"; mkdir -p "$P"; cd "$P"; rm -f results.tgz
HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive "${2:?code}" > recv.log 2>&1
rm -rf results; mkdir results; tar -xzf results.tgz -C results
cat results/agg.txt
