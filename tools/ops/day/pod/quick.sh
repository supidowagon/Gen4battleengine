#!/usr/bin/env bash
# POD SIDE. The quick lane (DAYBOX.md "Quick lane"): run from /work/<run> after payload.tar of a
# `pack.sh --delta` run is unpacked there (delta.tar, deletions.txt, [tables.tar.xz], meta.env,
# ids.txt, bin/). Moves the stream's lane repo from the commit OLD to NEW by the delta, then
# runs the at most 5 ids in ids.txt as one unit (no runner, no queue) and writes the result.
# Steps: (a) find or create the lane repo /work/_lane/<stream>/repo; (b) check the marker and the
# repo's tree against OLD and OLD_TREE; (c) apply deletions, then delta.tar, then tables.tar.xz;
# (d) check the tree against NEW_TREE, commit the snapshot, write the marker; (e) the start gate
# of runner.sh (gate.sh admit through choose, under RUNNER_LOCK, with the runner's knobs and
# defaults), then pytest as the unit q of this run: out/run/q (the pid of its setsid process,
# its process group) and out/q.w (its weight) count in every runner's podgate and claim-fit
# while it runs, and out/run/q goes when it ends; (f) result.
# Output (appended to quick.log, and shown by the exec-pod body): QUICK_START, the waits, then
# QUICK_PASS or QUICK_FAIL with pytest's summary line and the first failure block, then
# QUICK_DONE rc=<n>; or FALLBACK_FULL <reason> (rc 3: nothing was run; send a normal 1-chunk run
# of the same ids), QUICK_DONE rc=3. rc: 0 pass; 3 fallback; 4 gate wait over QUICK_WAIT_S;
# 5 broken pod state; 6 pytest passed no test (all skipped or deselected); 11-15 pytest's exit
# code 1-5 plus 10 (11: a test failed); 20 any other pytest exit (a signal, uv failing);
# 124 pytest ran over QUICK_MAX_S (its process group was killed). quick.rc holds the rc, agg.txt
# "RESULT: PASS|FAIL (...)" plus the log (wait.sh and collect.sh read agg.txt; bin/notify.sh sees
# quick.rc and offers results.tgz like the end of a normal run). Writes gate.env (podcheck.sh).
# One quick run per stream at a time: the stream's flock (QUICK_LOCK_S, default 600 s, else
# FALLBACK_FULL lane busy) is held from before step (a) to the end, pytest included.
# Env: QUICK_MAX_S (pytest time limit, default 1800), QUICK_WAIT_S (gate wait, default 900),
# QUICK_POLL_S (10), QUICK_LOCK_S, LANE_ROOT (/work/_lane), WORK (/work), CGROOT, and the runner's
# knobs with its defaults: GATE_GB, KILL_GB, MAXP, IDLE_WAIT_S, DEFAULT_WEIGHT_GB, RUNNER_LOCK,
# CLAIMS, CLAIM_S, CLAIM_MAX_S, RESERVE_TOP, RESERVE_FULL_GB, RESERVE_PCT; NUZLINE_TEST_COMPILE_CACHE
# (as unit.sh).
set -u
# The caller's PATH first (a test's stubs); the pod's uv and runpodctl folders after it.
export PATH="$PATH:/root/.local/bin:/work/_bin" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
export NUZLINE_TEST_COMPILE_CACHE="${NUZLINE_TEST_COMPILE_CACHE:-/work/_jaxcache/jax-tests}"
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D" || exit 1
exec >> "$D/quick.log" 2>&1
. ./meta.env; unset WATCH_F WATCH_D   # the beacon codes never reach the log
printf 'GATE_VERSION=%s\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\n' "${GATE_VERSION:-}" "${POD_SCRIPTS:-}" "${POD_WEIGHTS:-}" > "$D/gate.env"   # podcheck.sh reads it
. "$D/bin/gate.sh"; . "$D/bin/lane.sh"
WORK="${WORK:-/work}"; O="$D/out"
# runner.sh's knobs and defaults: the quick unit passes the same gate as a runner's unit.
GATE_GB="${GATE_GB:-88}"; MAXP="${MAXP:-10}"; KILL_GB="${KILL_GB:-118}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"
IDLE_WAIT_S="${IDLE_WAIT_S:-300}"; LOCK="${RUNNER_LOCK:-/work/.runner.lock}"; CLAIMS="${CLAIMS:-/work/.claims}"
CLAIM_S="${CLAIM_S:-120}"; CLAIM_MAX_S="${CLAIM_MAX_S:-3600}"
RESERVE_TOP="${RESERVE_TOP:-2}"; RESERVE_FULL_GB="${RESERVE_FULL_GB:-40}"; RESERVE_PCT="${RESERVE_PCT:-75}"
RUN=$(basename "$D"); ENDED=0; LR=""; L=""

# end <rc> <result text>: the last lines, quick.rc, agg.txt; exits.
end() { local rc="$1" res="$2"; ENDED=1
  rm -f "$O/pending.txt"
  echo "QUICK_DONE rc=$rc"
  { echo "RESULT: $res"; cat "$D/quick.log"; } > "$D/agg.txt.new" && mv -f "$D/agg.txt.new" "$D/agg.txt"
  echo "$rc" > "$D/quick.rc.new" && mv -f "$D/quick.rc.new" "$D/quick.rc"; exit "$rc"; }
trap 'rc=$?; [ "$ENDED" = 1 ] || { echo "QUICK_FAIL unexpected exit rc=$rc"; end 5 "FAIL (unexpected exit rc=$rc)"; }' EXIT
# fallback <reason>: nothing ran; the stream sends a normal run of the same ids.
fallback() { cd "$D" || true; echo "FALLBACK_FULL $1"; end 3 "FAIL (FALLBACK_FULL $1)"; }
# broken <reason>: the lane repo is wrong after the apply: removed and flagged, then a fallback.
broken() { cd "$D" || true; rm -rf "${LR:?}"; mkdir -p "${L:?}"; echo "$1" > "$L/BROKEN"; fallback "$1"; }
# failq <rc> <reason>: a pod-side failure (no fallback: sending the same run again would hit it too).
failq() { echo "QUICK_FAIL $2"; end "$1" "FAIL ($2)"; }

STREAM=$(lane_stream "$RUN") || fallback "run name $RUN has no stream"
[ "${MODE:-}" = delta ] && [ -f delta.tar ] && [ -f deletions.txt ] && [ -f ids.txt ] || fallback "payload is not a delta payload"
mapfile -t IDS < <(tr -d '\r' < ids.txt | sed -E '/^[[:space:]]*(#|$)/d')
[ "${#IDS[@]}" -ge 1 ] && [ "${#IDS[@]}" -le 5 ] || fallback "ids.txt holds ${#IDS[@]} ids (1 to 5 allowed)"
reserve_knobs || failq 5 "bad RESERVE_ knob"
echo "QUICK_START $(date +%s) run $RUN stream $STREAM ${OLD:0:8}..${NEW:0:8} ids ${#IDS[@]}"

L="$LANE_ROOT/$STREAM"; LR="$L/repo"; MK="$LANE_ROOT/$STREAM.sha"
mkdir -p "$LANE_ROOT" || fallback "cannot create the lane folder"
exec 9> "$LANE_ROOT/$STREAM.lock"
flock -w "${QUICK_LOCK_S:-600}" 9 || fallback "lane busy: another quick run of stream $STREAM holds the lock"

# (a) the lane repo: the marker's REPO is the truth. A run repo (a normal run of the stream wrote
# the marker) gets cloned into the lane; the lane repo itself is used as it is.
[ -f "$MK" ] || fallback "no marker"
M_COMMIT=$(lane_get "$MK" COMMIT); M_TREE=$(lane_get "$MK" TREE); M_KEY=$(lane_get "$MK" TABLES_KEY); M_REPO=$(lane_get "$MK" REPO)
case "$M_REPO" in *..*) fallback "marker REPO holds '..'";; "$WORK"/*) ;; *) fallback "marker REPO is outside the work folder";; esac
# (b) the marker names the commit the repo holds: OLD, and the tree OLD_TREE.
[ -n "$M_COMMIT" ] && [ "$M_COMMIT" = "$OLD" ] || fallback "marker commit ${M_COMMIT:0:8} is not OLD ${OLD:0:8}"
[ "$M_TREE" = "$OLD_TREE" ] || fallback "marker tree is not OLD_TREE"
# lane_copy <run repo> <lane repo>: the git-ignored built tables (data/) and node packages of the
# run repo into the lane repo; fails when any part of the pipe fails.
lane_copy() ( set -o pipefail
  git -C "$1" ls-files -o -i --exclude-standard -z -- data tools/smogon_calc/node_modules \
    | tar -C "$1" --null -T - -cf - | tar -C "$2" -xf - )
if [ "$M_REPO" = "$LR" ]; then
  [ -d "$LR/.git" ] && [ ! -e "$L/BROKEN" ] || fallback "lane repo missing or broken"
else
  [ -d "$M_REPO/.git" ] || fallback "marker repo (run folder) is gone"
  echo "QUICK_LANE creating the lane repo from the run repo of the marker"
  rm -rf "${LR:?}" "${L:?}/BROKEN"; mkdir -p "$L"
  # The clone holds the committed snapshot; the built tables and node packages are copied from the
  # run repo; the venv comes from uv sync.
  if ! { git clone -q --local "$M_REPO" "$LR" \
      && git -C "$LR" config core.autocrlf false \
      && lane_copy "$M_REPO" "$LR" \
      && [ -f "$LR/data/gen4_platinum_kaizo/manifest.json" ] \
      && { cd "$LR" && uv sync --frozen -q --group rom; }; }; then cd "$D" || true; rm -rf "${LR:?}"; fallback "lane repo creation failed"; fi
  cd "$D" || exit 1
fi
# The decomp: the worktree of this run's DECOMP commit, as setup.sh links it.
[ -d "$WORK/_decomp/$DECOMP" ] || fallback "decomp ${DECOMP:0:8} is not on the pod"
mkdir -p "$LR/external" && ln -sfn "$WORK/_decomp/$DECOMP" "$LR/external/pokeplatinum" || fallback "cannot link the decomp"
# The tables: a delta without tables.tar.xz needs the marker's key to be OLD_TABLES_KEY (both known).
if [ ! -f tables.tar.xz ] && { [ -z "${OLD_TABLES_KEY:-}" ] || [ "$M_KEY" != "$OLD_TABLES_KEY" ]; }; then
  fallback "tables key of the lane (${M_KEY:-unknown}) is not the delta's OLD key (${OLD_TABLES_KEY:-unknown}) and no tables travel"; fi
T=$(lane_tree "$LR") || fallback "git failed in the lane repo"
[ "$T" = "$OLD_TREE" ] || fallback "lane tree ${T:0:8} is not OLD_TREE ${OLD_TREE:0:8}"

# (c) apply. Every path is checked first, so a refused delta changes nothing in the lane repo.
# The listing: names literal (delta.names), types from the long listing in the same order;
# delta.files: the members that are not directories, force-added in (d).
tar --quoting-style=literal -tf delta.tar > "$D/delta.names" && tar --quoting-style=literal -tvf delta.tar > "$D/delta.long" \
  || fallback "delta.tar does not list"
[ "$(wc -l < "$D/delta.names")" = "$(wc -l < "$D/delta.long")" ] || fallback "unsafe path in delta.tar: the listing does not pair names and types"
grep -v '/$' "$D/delta.names" > "$D/delta.files" || true
REAL=$(cd "$LR" && pwd -P)
safe() { case "$1" in ''|/*|./*|*..*|.git|.git/*) return 1;; esac
  case "$(realpath -m -- "$LR/$1")" in "$REAL"/*|"$LR"/*) return 0;; *) return 1;; esac; }
while IFS= read -r p; do [ -z "$p" ] || safe "$p" || fallback "unsafe path in deletions.txt: $p"; done < deletions.txt
while IFS= read -r p; do safe "$p" || fallback "unsafe path in delta.tar: $p"; done < "$D/delta.names"
if grep -E '^[lh]' "$D/delta.long" | grep -qE -- '(->|link to) (/|.*\.\.)'; then fallback "delta.tar holds a link that leaves the repo"; fi
# A member under a symlink that the same tar creates would be written through that link.
UL=$(awk 'NR == FNR { t[FNR] = substr($0, 1, 1); next }
  { n = $0; sub(/\/$/, "", n); N[FNR] = n; if (t[FNR] == "l") S[n] = 1 }
  END { for (i = 1; i in N; i++) { m = split(N[i], a, "/"); p = ""
          for (j = 1; j < m; j++) { p = (j == 1) ? a[1] : p "/" a[j]; if (p in S) { print N[i]; exit } } } }' "$D/delta.long" "$D/delta.names")
[ -z "$UL" ] || fallback "unsafe path in delta.tar: $UL lies under a link of the same tar"
# Deletions first, so a file that becomes a directory (or the reverse) extracts; empty directories go.
while IFS= read -r p; do [ -z "$p" ] && continue
  rm -f -- "${LR:?}/${p:?}"; d=$(dirname "$p")
  while [ "$d" != . ] && rmdir "$LR/$d" 2> /dev/null; do d=$(dirname "$d"); done; done < deletions.txt
tar --no-same-owner -xf delta.tar -C "$LR" || broken "delta.tar did not extract"
if [ -f tables.tar.xz ]; then
  git -C "$LR" clean -fdXq -- data   # the old built tables (git-ignored) go
  tar --no-same-owner -xJf tables.tar.xz -C "$LR" || broken "tables.tar.xz did not extract"
  [ -f "$LR/data/gen4_platinum_kaizo/manifest.json" ] || broken "tables.tar.xz has no Kaizo manifest"; fi
# (d) the tree must be NEW_TREE (the delta's files force-added: a file of the commit under an
# ignored path counts); then the snapshot is committed (the next OLD_TREE check) and the marker written.
T=$(lane_tree "$LR" "$D/delta.files") || broken "git failed in the lane repo"
[ "$T" = "$NEW_TREE" ] || broken "tree mismatch"
{ lane_stage "$LR" "$D/delta.files" \
  && git -C "$LR" -c user.name=ci -c user.email=ci@localhost commit -q --allow-empty -m "quick $NEW"; } > /dev/null || broken "commit failed"
lane_marker_put "$STREAM" "$NEW" "$NEW_TREE" "$TABLES_KEY" "$LR" || broken "marker not written"
echo "QUICK_APPLIED $(grep -c . deletions.txt) deleted, $(grep -c . "$D/delta.files") files, tables $([ -f tables.tar.xz ] && echo sent || echo kept)"
# New dependencies: uv run syncs the venv itself; node packages only when their lock changed.
if grep -qx 'tools/smogon_calc/package-lock.json' "$D/delta.files" && command -v npm > /dev/null; then
  ( cd "$LR" && npm ci -s --prefix tools/smogon_calc ) || echo "WARNING: npm ci failed"; fi

# (e) the gate: the heaviest id's weight (weights.py over weights.txt, as setup.sh gives a unit
# its weight; 20 for an id no line matches), then runner.sh's start decision (gate.sh choose and
# admit over podgate's live units and reserve, claim_state's oldest claim), each check under
# RUNNER_LOCK, every QUICK_POLL_S s, at most QUICK_WAIT_S s.
W=$(cd "$LR" && uv run --frozen python -c '
import pathlib, sys
sys.path.insert(0, sys.argv[1]); import weights
ids = [x.strip() for x in open(sys.argv[2], encoding="utf-8") if x.strip() and not x.lstrip().startswith("#")]
print(weights.unit_weight(ids, weights.load(pathlib.Path(sys.argv[1]) / "weights.txt")))' "$D/bin" "$D/ids.txt")
case "$W" in ''|*[!0-9]*) failq 5 "weight of the ids not computed";; esac
cgmem current > /dev/null || failq 5 "no cgroup memory file"
mkdir -p "$O/run" || failq 5 "cannot create out/run"
echo "$W" > "$O/q.w"; echo "q quick" > "$O/pending.txt"   # choose reads the head of pending.txt and its .w
exec 8>> "$LOCK" || failq 5 "cannot open the gate lock $LOCK"
START=$(date +%s); idle0=""
while :; do
  [ $(( $(date +%s) - START )) -lt "${QUICK_WAIT_S:-900}" ] || failq 4 "gate wait over ${QUICK_WAIT_S:-900} s"
  flock -w 120 8 || { echo "QUICK_WAIT $(date +%s) gate lock $LOCK still held after 120 s"; continue; }
  now=$(date +%s); B=$(cgmem current) || { flock -u 8; failq 5 "no cgroup memory file"; }; mem=$(( B / 1000000000 ))
  read -r plive pres <<< "$(podgate "$WORK"/*/out/run/* "$WORK"/*/*/out/run/*)"
  case "$plive$pres" in ''|*[!0-9]*) flock -u 8; failq 5 "the gate reserve is not a number";; esac
  [ "$plive" -eq 0 ] || idle0=""; [ "$plive" -eq 0 ] && [ -z "$idle0" ] && idle0=$now
  cs=none; cu=""; cw=0
  if [ "$mem" -lt "$GATE_GB" ]; then read -r cs cu cw <<< "$(claim_state "$RUN")"
    read -r d _ a _ <<< "$(cd "$O" && choose "$plive" "$pres" "$mem" $(( now - ${idle0:-$now} )) 0 "$cs" "${cw:-0}")"
  else d=none; a=hold-mem; idle0=""; fi
  if [ "$d" = start ]; then
    rm -f "$O/pending.txt"
    # The unit: its own process group (setsid, as runner.sh starts unit.sh), both locks closed.
    setsid bash -c 'cd "$1" || exit 1; shift; exec uv run --frozen pytest -q -rfEs -p no:cacheprovider "$@"' \
      quick-unit "$LR" "${IDS[@]}" < /dev/null > "$D/quick.pytest.log" 2>&1 8>&- 9>&- &
    QPG=$!; echo "$QPG" > "$O/run/.q.tmp" && mv -f "$O/run/.q.tmp" "$O/run/q"   # visible to every podgate from here
    flock -u 8; break
  fi
  flock -u 8
  echo "QUICK_WAIT $now $a: mem ${mem} + reserve ${pres} + weight ${W} GB; GATE_GB ${GATE_GB} KILL_GB ${KILL_GB}, ${plive} units live$([ "$cs" = other ] && echo ", claim $cu is older")"
  sleep "${QUICK_POLL_S:-10}"
done
echo "QUICK_RUN $now ($a) mem ${mem} GB, reserve ${pres} GB, weight ${W} GB, pod units $(( plive + 1 ))"
# Time limit: past QUICK_MAX_S the watcher writes out/q.timeout and kills the unit's process group
# (TERM, 5 s, KILL), as runner.sh kills a unit over UNIT_MAX_S.
rm -f "$O/q.timeout"
( e=$(( $(date +%s) + ${QUICK_MAX_S:-1800} ))
  while [ "$(date +%s)" -lt "$e" ]; do sleep 1; kill -0 "$QPG" 2> /dev/null || exit 0; done
  date +%s > "$O/q.timeout"; kill -TERM -- "-$QPG" 2> /dev/null; sleep 5; kill -KILL -- "-$QPG" 2> /dev/null; exit 0 ) 8>&- 9>&- &
WPID=$!
wait "$QPG"; PRC=$?
if [ -f "$O/q.timeout" ]; then wait "$WPID"; else kill "$WPID" 2> /dev/null; wait "$WPID" 2> /dev/null; fi
rm -f "$O/run/q"
# The tests' leftovers (non-ignored files, edits) go, so the next delta's OLD_TREE check holds.
( cd "$LR" && git reset -q --hard HEAD && git clean -fdq ) > /dev/null 2>&1
# (f) result.
SUM=$(grep -v '^[[:space:]]*$' "$D/quick.pytest.log" | tail -1)
# The first failure block: from pytest's first "____ test_a ____" header (or "____ ERROR at setup of
# test_a ____"), else its first FAILED/ERROR summary line. No interval expressions: the pod's awk
# is mawk, which reads _{3,} as a literal.
BLK=$(awk '/^___+ |^(FAILED|ERROR) /{f=1} f{print; if(++n>=30) exit}' "$D/quick.pytest.log")
SKP=$(grep -E '^SKIPPED ' "$D/quick.pytest.log" | head -10)
if [ -f "$O/q.timeout" ]; then echo "QUICK_FAIL timeout after ${QUICK_MAX_S:-1800} s (pytest rc $PRC) $SUM"; [ -z "$BLK" ] || printf '%s\n' "$BLK"
  end 124 "FAIL (timeout after ${QUICK_MAX_S:-1800} s)"; fi
if [ "$PRC" = 0 ]; then
  if printf '%s\n' "$SUM" | grep -qE '(^|[^a-z])[0-9]+ passed'; then echo "QUICK_PASS $SUM"; [ -z "$SKP" ] || printf '%s\n' "$SKP"; end 0 "PASS ($SUM)"; fi
  echo "QUICK_FAIL no test passed: $SUM"; [ -z "$SKP" ] || printf '%s\n' "$SKP"; end 6 "FAIL (no test passed: $SUM)"
fi
case "$PRC" in [1-5]) RC=$(( 10 + PRC ));; *) RC=20;; esac
echo "QUICK_FAIL rc=$RC (pytest rc $PRC) $SUM"
[ -z "$BLK" ] || printf '%s\n' "$BLK"
end "$RC" "FAIL (pytest rc $PRC: $SUM)"
