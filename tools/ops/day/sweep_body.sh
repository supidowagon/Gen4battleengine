#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: sweep_body.sh <run> <jobs file>
# Prints the exec-pod body that starts the nightly soundness sweep (tools/ops/nightly_brief.md,
# "Soundness sweep"; DAYBOX.md "Soundness sweep"): bin/sweep.sh of the already set-up run
# /work/<run> (pod/sweep.sh travels in the run's bin/ when the run was packed from a checkout at
# main that has it), detached (setsid, nohup), SWEEP_P and SWEEP_W from the PC's environment
# passed on, SWEEP_CAP only when set (unset: sweep.sh takes the cap from the pod's memory limit).
# The jobs file (tools/ops/sweep/seeds.py output, one `<node id> <short name> <seed>` per line) is
# too long for a body, so it travels in sweep.tar with start.sh by a detached `runpodctl send`
# ($RENT/podcheck/, as stop_body.sh) and the body checks the tar's sha256 before it unpacks
# (send_body.sh tool_body: RECEIVE_FAILED, PAYLOAD_SHA_MISMATCH, UNPACK_FAILED, PENDING). start.sh
# copies the jobs to /work/<run>/sweep_jobs.txt, starts the sweep and returns at once. Its last
# line: SWEEP_STARTED <run> | SWEEP_REFUSED <reason> (no run folder, no bin/sweep.sh, no repo,
# a sweep of this run still running).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"; . "$HERE/send_body.sh"
[ $# -eq 2 ] || { echo "usage: sweep_body.sh <run> <jobs file>"; exit 2; }
check_run_name "$1"
[ -f "$2" ] || { echo "REFUSED: no jobs file $2"; exit 2; }
[ "$(tr -d '\r' < "$2" | grep -cv '^[[:space:]]*\(#\|$\)')" -gt 0 ] || { echo "REFUSED: $2 has no job line"; exit 2; }
tr -d '\r' < "$2" | grep -v '^[[:space:]]*\(#\|$\)' | grep -qvE '^[^ ]+ [^ ]+ [0-9]+$' && { echo "REFUSED: $2 has a line that is not '<node id> <short name> <seed>'"; exit 2; }
for v in SWEEP_P SWEEP_W SWEEP_CAP; do eval "x=\${$v:-}"; [[ -z "$x" || "$x" =~ ^[0-9]+$ ]] || { echo "REFUSED: $v '$x' is not a whole number"; exit 3; }; done
mkdir -p "$RENT/podcheck"
for d in "$RENT"/podcheck/*/; do grep -q '^SEND_RC=' "$d/send.log" 2>/dev/null && rm -rf "$d"; done
T="$RENT/podcheck/$(date +%s)-$$"; mkdir "$T"; trap '[ -n "${CODE:-}" ] || rm -rf "$T"' EXIT
tr -d '\r' < "$2" | grep -v '^[[:space:]]*\(#\|$\)' > "$T/sweep_jobs.txt"
{ echo "# POD SIDE: start.sh of sweep_body.sh"
  echo "R=/work/$1; export SWEEP_P=${SWEEP_P:-5} SWEEP_W=${SWEEP_W:-13}${SWEEP_CAP:+ SWEEP_CAP=$SWEEP_CAP}"
  cat <<'START'
[ -d "$R" ] || { echo "SWEEP_REFUSED no folder $R"; exit 1; }
[ -f "$R/bin/sweep.sh" ] || { echo "SWEEP_REFUSED no $R/bin/sweep.sh (the run was packed before sweep.sh)"; exit 1; }
[ -x "$R/repo/.venv/bin/python" ] || { echo "SWEEP_REFUSED no $R/repo/.venv (setup.sh not finished?)"; exit 1; }
p=$(cat "$R/sweep.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null && [ ! -f "$R/sweep_out/SWEEP_DONE" ]; then echo "SWEEP_REFUSED a sweep of this run is running (pid $p)"; exit 1; fi
cp sweep_jobs.txt "$R/sweep_jobs.txt" && cd "$R" || exit 1
setsid nohup bash bin/sweep.sh sweep_jobs.txt > sweep.log 2>&1 < /dev/null &
echo $! > sweep.pid
echo "SWEEP_STARTED ${R##*/} jobs=$(grep -c . sweep_jobs.txt)"
START
} > "$T/start.sh"
tar -cf "$T/sweep.tar" -C "$T" start.sh sweep_jobs.txt
PSUM=$(sha256sum "$T/sweep.tar" | cut -c1-64)
send_start "$T" sweep.tar 600 || { CODE=""; exit 4; }
BODY=$(body_json "$(tool_body sweep sweep.tar "$PSUM" "$CODE" "bash start.sh")") || { kill "$SPID" 2>/dev/null || true; echo "$BODY"; exit 3; }
echo "== exec-pod body (sweep of run $1, $(grep -c . "$T/sweep_jobs.txt") jobs, ${#BODY} characters):"
echo "$BODY"
