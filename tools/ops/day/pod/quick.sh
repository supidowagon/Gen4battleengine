#!/usr/bin/env bash
# POD SIDE. The quick lane (DAYBOX.md "Quick lane"): run from /work/<run> after payload.tar of a
# `pack.sh --delta` run is unpacked there (delta.tar, deletions.txt, [tables.tar.xz], meta.env,
# ids.txt, bin/). Moves the stream's lane repo from the commit OLD to NEW by the delta, then
# runs the at most 5 ids in ids.txt directly (no runner, no queue) and writes the result.
# Steps: (a) find or create the lane repo /work/_lane/<stream>/repo; (b) check the marker and the
# repo's tree against OLD and OLD_TREE; (c) apply deletions, then delta.tar, then tables.tar.xz;
# (d) check the tree against NEW_TREE, commit the snapshot, write the marker; (e) wait for
# memory (weight of the heaviest id + the gate's reserve), then pytest; (f) result.
# Output (appended to quick.log, and shown by the exec-pod body): QUICK_START, the waits, then
# QUICK_PASS or QUICK_FAIL with pytest's summary line and the first failure block, then
# QUICK_DONE rc=<n>; or FALLBACK_FULL <reason> (rc 3: nothing was run, the lane repo is as it was
# or, for a failed apply, removed and flagged; send a normal 1-chunk run of the same ids),
# QUICK_DONE rc=3. quick.rc holds the rc, agg.txt "RESULT: PASS|FAIL (...)" plus the log (wait.sh
# and collect.sh read agg.txt; bin/notify.sh sees quick.rc and offers results.tgz like the end of
# a normal run).
# One quick run per stream at a time: the stream's flock (QUICK_LOCK_S, default 600 s, else
# FALLBACK_FULL lane busy) is held from before step (a) to the end, pytest included.
# Env: QUICK_MAX_S (pytest timeout, default 1800), QUICK_WAIT_S (memory wait, default 900),
# QUICK_POLL_S (10), QUICK_LOCK_S, LANE_ROOT (/work/_lane), WORK (/work), CGROOT, DEFAULT_WEIGHT_GB,
# RESERVE_TOP, RESERVE_FULL_GB, RESERVE_PCT, NUZLINE_TEST_COMPILE_CACHE (as unit.sh).
set -u
export PATH="/root/.local/bin:/work/_bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
export NUZLINE_TEST_COMPILE_CACHE="${NUZLINE_TEST_COMPILE_CACHE:-/work/_jaxcache/jax-tests}"
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D" || exit 1
exec >> "$D/quick.log" 2>&1
. ./meta.env; unset WATCH_F WATCH_D   # the beacon codes never reach the log
. "$D/bin/gate.sh"; . "$D/bin/lane.sh"
WORK="${WORK:-/work}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"; MAXP="${MAXP:-10}"; export NO_W_FILE_GB
RUN=$(basename "$D"); ENDED=0; LR=""; L=""

# end <rc> <result text>: the last lines, quick.rc, agg.txt; exits.
end() { local rc="$1" res="$2"; ENDED=1
  echo "QUICK_DONE rc=$rc"
  { echo "RESULT: $res"; cat "$D/quick.log"; } > "$D/agg.txt.new" && mv -f "$D/agg.txt.new" "$D/agg.txt"
  echo "$rc" > "$D/quick.rc.new" && mv -f "$D/quick.rc.new" "$D/quick.rc"; exit "$rc"; }
trap 'rc=$?; [ "$ENDED" = 1 ] || { echo "QUICK_FAIL unexpected exit rc=$rc"; end 5 "FAIL (unexpected exit rc=$rc)"; }' EXIT
# fallback <reason>: nothing ran; the stream sends a normal run of the same ids.
fallback() { cd "$D" || true; echo "FALLBACK_FULL $1"; end 3 "FAIL (FALLBACK_FULL $1)"; }
# broken <reason>: the lane repo is wrong after the apply: removed and flagged, then a fallback.
broken() { cd "$D" || true; rm -rf "${LR:?}"; mkdir -p "${L:?}"; echo "$1" > "$L/BROKEN"; fallback "$1"; }

STREAM=$(lane_stream "$RUN") || fallback "run name $RUN has no stream"
[ "${MODE:-}" = delta ] && [ -f delta.tar ] && [ -f deletions.txt ] && [ -f ids.txt ] || fallback "payload is not a delta payload"
mapfile -t IDS < <(tr -d '\r' < ids.txt | sed -E '/^[[:space:]]*(#|$)/d')
[ "${#IDS[@]}" -ge 1 ] && [ "${#IDS[@]}" -le 5 ] || fallback "ids.txt holds ${#IDS[@]} ids (1 to 5 allowed)"
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
if [ "$M_REPO" = "$LR" ]; then
  [ -d "$LR/.git" ] && [ ! -e "$L/BROKEN" ] || fallback "lane repo missing or broken"
else
  [ -d "$M_REPO/.git" ] || fallback "marker repo (run folder) is gone"
  echo "QUICK_LANE creating the lane repo from the run repo of the marker"
  rm -rf "${LR:?}" "${L:?}/BROKEN"; mkdir -p "$L"
  # The clone holds the committed snapshot; the git-ignored built tables (data/) and node packages
  # are copied from the run repo; the venv comes from uv sync.
  if ! { git clone -q --local "$M_REPO" "$LR" \
      && git -C "$LR" config core.autocrlf false \
      && { git -C "$M_REPO" ls-files -o -i --exclude-standard -z -- data tools/smogon_calc/node_modules \
           | tar -C "$M_REPO" --null -T - -cf - | tar -C "$LR" -xf -; } \
      && { cd "$LR" && uv sync --frozen -q; }; }; then cd "$D" || true; rm -rf "${LR:?}"; fallback "lane repo creation failed"; fi
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

# (c) apply. Every path is checked first, so a refused delta changes nothing.
REAL=$(cd "$LR" && pwd -P)
safe() { case "$1" in ''|/*|./*|*..*|.git|.git/*) return 1;; esac
  case "$(realpath -m -- "$LR/$1")" in "$REAL"/*|"$LR"/*) return 0;; *) return 1;; esac; }
while IFS= read -r p; do [ -z "$p" ] || safe "$p" || fallback "unsafe path in deletions.txt: $p"; done < deletions.txt
while IFS= read -r p; do safe "$p" || fallback "unsafe path in delta.tar: $p"; done < <(tar -tf delta.tar)
if tar -tvf delta.tar | grep -E '^[lh]' | grep -qE -- '(->|link to) (/|.*\.\.)'; then fallback "delta.tar holds a link that leaves the repo"; fi
# Deletions first, so a file that becomes a directory (or the reverse) extracts; empty directories go.
while IFS= read -r p; do [ -z "$p" ] && continue
  rm -f -- "${LR:?}/${p:?}"; d=$(dirname "$p")
  while [ "$d" != . ] && rmdir "$LR/$d" 2> /dev/null; do d=$(dirname "$d"); done; done < deletions.txt
tar --no-same-owner -xf delta.tar -C "$LR" || broken "delta.tar did not extract"
if [ -f tables.tar.xz ]; then
  git -C "$LR" clean -fdXq -- data   # the old built tables (git-ignored) go
  tar --no-same-owner -xJf tables.tar.xz -C "$LR" || broken "tables.tar.xz did not extract"
  [ -f "$LR/data/gen4_platinum_kaizo/manifest.json" ] || broken "tables.tar.xz has no Kaizo manifest"; fi
# (d) the tree must be NEW_TREE; then the snapshot is committed (the next OLD_TREE check) and the marker written.
T=$(lane_tree "$LR") || broken "git failed in the lane repo"
[ "$T" = "$NEW_TREE" ] || broken "tree mismatch"
{ git -C "$LR" -c core.autocrlf=false add -A \
  && git -C "$LR" -c user.name=ci -c user.email=ci@localhost commit -q --allow-empty -m "quick $NEW"; } > /dev/null || broken "commit failed"
lane_marker_put "$STREAM" "$NEW" "$NEW_TREE" "$TABLES_KEY" "$LR" || broken "marker not written"
echo "QUICK_APPLIED $(grep -c . deletions.txt) deleted, $(tar -tf delta.tar | grep -vc '/$') files, tables $([ -f tables.tar.xz ] && echo sent || echo kept)"
# New dependencies: uv run syncs the venv itself; node packages only when their lock changed.
if tar -tf delta.tar | grep -qx 'tools/smogon_calc/package-lock.json' && command -v npm > /dev/null; then
  ( cd "$LR" && npm ci -s --prefix tools/smogon_calc ) || echo "WARNING: npm ci failed"; fi

# (e) memory: wait until the container's free memory exceeds the heaviest id's weight (weights.py
# over weights.txt, as setup.sh gives a unit its weight; 20 for an id no line matches) plus the
# gate's reserve (podgate over every run's live units, gate.sh).
W=$(cd "$LR" && uv run --frozen python -c '
import pathlib, sys
sys.path.insert(0, sys.argv[1]); import weights
ids = [x.strip() for x in open(sys.argv[2], encoding="utf-8") if x.strip() and not x.lstrip().startswith("#")]
print(weights.unit_weight(ids, weights.load(pathlib.Path(sys.argv[1]) / "weights.txt")))' "$D/bin" "$D/ids.txt")
case "$W" in ''|*[!0-9]*) echo "QUICK_FAIL weight of the ids not computed"; end 5 "FAIL (weight not computed)";; esac
LIM=$(cgmem max) || { echo "QUICK_FAIL no cgroup memory file"; end 5 "FAIL (no cgroup memory file)"; }
case "$LIM" in ''|*[!0-9]*) LIM=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ));; esac
[ "${#LIM}" -lt 16 ] || LIM=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))   # a v1 "no limit" is near 2^63
START=$(date +%s)
while :; do
  CUR=$(cgmem current) || { echo "QUICK_FAIL no cgroup memory file"; end 5 "FAIL (no cgroup memory file)"; }
  read -r _ RES <<< "$(podgate "$WORK"/*/out/run/* "$WORK"/*/*/out/run/*)"
  case "$RES" in ''|*[!0-9]*) echo "QUICK_FAIL the gate reserve is not a number"; end 5 "FAIL (gate reserve)";; esac
  FREE=$(( (LIM - CUR) / 1000000000 )); NEED=$(( W + RES ))
  [ "$FREE" -gt "$NEED" ] && break
  [ $(( $(date +%s) - START )) -lt "${QUICK_WAIT_S:-900}" ] || { echo "QUICK_FAIL memory wait over ${QUICK_WAIT_S:-900} s"; end 4 "FAIL (memory wait timed out)"; }
  echo "QUICK_WAIT $(date +%s) free ${FREE} GB, need more than ${NEED} GB (weight ${W} + reserve ${RES})"
  sleep "${QUICK_POLL_S:-10}"
done
echo "QUICK_RUN free ${FREE} GB, weight ${W} GB, reserve ${RES} GB"
cd "$LR" || exit 1
timeout -k 30 "${QUICK_MAX_S:-1800}" uv run --frozen pytest -q -rfE -p no:cacheprovider "${IDS[@]}" 9>&- > "$D/quick.pytest.log" 2>&1
PRC=$?
# The tests' leftovers (non-ignored files, edits) go, so the next delta's OLD_TREE check holds.
git reset -q --hard HEAD > /dev/null 2>&1; git clean -fdq > /dev/null 2>&1; cd "$D" || exit 1
# (f) result.
SUM=$(grep -v '^[[:space:]]*$' "$D/quick.pytest.log" | tail -1)
BLK=$(awk '/^_{3,} |^(FAILED|ERROR) /{f=1} f{print; if(++n>=30) exit}' "$D/quick.pytest.log")
if [ "$PRC" = 0 ]; then echo "QUICK_PASS $SUM"; end 0 "PASS ($SUM)"; fi
if [ "$PRC" = 124 ] || [ "$PRC" = 137 ]; then echo "QUICK_FAIL timeout after ${QUICK_MAX_S:-1800} s (rc $PRC) $SUM"; else echo "QUICK_FAIL rc=$PRC $SUM"; fi
[ -z "$BLK" ] || printf '%s\n' "$BLK"
end "$PRC" "FAIL (pytest rc $PRC: $SUM)"
