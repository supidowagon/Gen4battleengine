#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: collect.sh <run> <code printed by the pod's fetch.sh | --local>
# Receives results.tgz into $RENT/out/<run>/ and unpacks it into results/. --local skips the
# receive and unpacks the results.tgz already there (wait.sh puts it there).
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/../common.sh"
RUN="${1:?usage: collect.sh <run> <code|--local>}"; check_run_name "$RUN"
P="$RENT/out/$RUN"; mkdir -p "$P"; cd "$P"
if [ "${2:?code}" != --local ]; then rm -f results.tgz
  HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive "$2" > recv.log 2>&1; fi
rm -rf results; mkdir results; tar -xzf results.tgz -C results
cat results/agg.txt
