#!/usr/bin/env bash
# POD SIDE. Usage: bash /work/_bin/start_run.sh <run name> <runpodctl transfer code>
# Creates /work/<run> and starts inner_run.sh detached (receive, unpack, sync env,
# memory guard, pytest). Progress: /work/<run>/launch.log; poll with status.sh.
set -u
RUN="${1:?run name}"; CODE="${2:?transfer code}"
case "$RUN" in *[!a-zA-Z0-9_.-]*|_*) echo "REFUSED: bad run name"; exit 2;; esac
D="/work/$RUN"
[ -e "$D" ] && { echo "REFUSED: $D already exists (pick a new run name)"; exit 3; }
mkdir -p "$D"
echo LAUNCHING > "$D/STATE"
nohup bash /work/_bin/inner_run.sh "$RUN" "$CODE" > "$D/launch.log" 2>&1 < /dev/null &
echo "launched $RUN; poll: bash /work/_bin/status.sh $RUN"
