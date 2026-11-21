#!/usr/bin/env bash
# PC SIDE (Git Bash). Snapshot capture and compare on the test pod (tools/snap_compare.py;
# rules: docs/process.md section 5; steps: tools/ops/day/DAYBOX.md "Snapshot compare").
# Usage:
#   snap.sh capture <commit> <tables.tar.xz> [<name>] [--tool <commit>] [--dry-run]
#   snap.sh compare <commit> <tables.tar.xz> <capture dir on the pod> <name> [--allow-new a,b] [--dry-run]
#   snap.sh fetch <name> <code>
# capture: pod folder /work/<name> (default snapmain-<7-char short>); payload = git archive of
#   <commit> + the tables; run.sh captures seeds 0-199 into /work/<name>/snap with the checkout's
#   own tools/snap_compare.py, or with --tool <commit>'s copy (re-capture for a candidate whose
#   snap_compare writes another format; name it snap2-<short tip>).
# compare: pod folder /work/<name>; payload = git archive of the candidate + its tables; run.sh
#   compares <capture dir> (e.g. /work/snapmain-08a5700/snap) with the candidate's own
#   tools/snap_compare.py; --allow-new is left out when empty.
# Both write $RENT/out/<name>/{code.tar,tables.tar.xz,payload.tar,run.sh,body.json,send.log},
#   refuse forbidden files (FORBID and TABLE_DENY in the code archive; table entries with '..', a
#   leading '/' or './', outside TABLE_ALLOW or in TABLE_DENY), start a detached send whose
#   send.log ends with SEND_RC=<n>, and print the exec-pod body. --dry-run runs every check and
#   writes run.sh and body.json (receive code "<code>") but sends nothing.
#   run.sh first writes /work/<name>/reads (the capture dir a compare reads; empty for a
#   capture), then /work/<name>/times.txt: START, SYNC_RC=<n> (a failed uv sync stops here,
#   without ALL_DONE), CAPTURE_START or COMPARE_START, CAPTURE_RC=<n> <epoch> or
#   COMPARE_RC=<n> <epoch>, ALL_DONE. Before the job run.sh checks /proc/sys/vm/max_map_count: under
#   1048576 (a pod with 65530 aborts JAX processes, rc 134) or unreadable ends times.txt with
#   MAP_COUNT_LOW <value|none> <epoch>, without ALL_DONE.
#   Env: SNAP_GATE_GB (default 88): run.sh starts the capture or compare only while container
#   memory is under it; pass the shape's SNAP_GATE_GB (176 on a 256 GB pod). Memory is read by pod/gate.sh's
#   cgmem (cgroup v2, else v1), copied into run.sh; with neither readable times.txt ends
#   GATE_NO_CGROUP_MEMORY <epoch>, without ALL_DONE. The same SNAP_GATE_GB is snap_compare's
#   --gate-gb: a part starts only while container memory + the live parts' reserve + its own
#   expected peak stays at or under it. SNAP_WORKERS (default 16): snap_compare's --workers, the
#   parts run at once (1: one after another). Both are passed only when the snap_compare that
#   runs (the commit's, or --tool's) has --workers; an older one runs its parts one by one.
#   Capture store on the network volume (DAYBOX.md "Network volume"; pod/cache_sync.sh, which
#   travels in the payload as committed at HEAD of this checkout): a capture without --tool is
#   keyed "<full sha>-<key16>" (key16: table_key16 below, from the tables archive). On a pod whose
#   /vol is a mountpoint, run.sh first fetches that key from the volume (store.log; times.txt
#   then holds CAPTURE_FROM_VOLUME <key> <epoch>, CAPTURE_RC=0, ALL_DONE, and no capture runs);
#   otherwise it captures and, after CAPTURE_RC=0, stores the capture (STORE_RC=<n> <epoch>; a
#   stored key is never overwritten). A compare of /work/snapmain-<short>/snap whose capture.json
#   is not on the pod fetches it from the volume first (FETCH_RC=<n> <epoch>), keyed by <short>'s
#   full sha and its key16 from $RENT/tables. Without a key nothing is stored or fetched (NOTE
#   line); on a pod without /vol run.sh does what it did before the volume.
# fetch: receives what the pod sends (DAYBOX.md "Snapshot compare", Fetch) into
#   $RENT/out/<name>/fetch-<epoch>/ and unpacks every .tgz received.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
. "$HERE/pod/gate.sh"   # cgmem (cgroup v2, else v1), copied into run.sh
# table_key16 <full sha> [<tables archive>]: the first 16 hex of a build's table key
# (tools/ops/build_tables.sh, tools/ops/table_key.py): the <archive minus .tar.xz>.key file beside
# the archive, else a tables_k<key16>.tar.xz archive name, else the last "<sha> <key>" line of
# $RENT/tables/index.txt for <full sha>; nothing when none is found. (Kept here, not in
# common.sh: common.sh is an input of the table key.)
table_key16() { local k=""
  if [ -n "${2:-}" ]; then
    [ -f "${2%.tar.xz}.key" ] && k=$(tr -dc 0-9a-f < "${2%.tar.xz}.key")
    [ -n "$k" ] || k=$(basename "$2" | sed -n 's/^tables_k\([0-9a-f]\{16\}\)\.tar\.xz$/\1/p')
  fi
  [ -n "$k" ] || k=$(awk -v s="$1" '$1 == s {k = $2} END {print k}' "$RENT/tables/index.txt" 2>/dev/null | tr -dc 0-9a-f)
  k=${k:0:16}; if [[ "$k" =~ ^[0-9a-f]{16}$ ]]; then echo "$k"; fi; return 0; }
usage() { sed -n '4,7p' "$0" | sed 's/^# *//'; exit 2; }
refuse() { echo "REFUSED: $1"; [ -z "${2:-}" ] || echo "$2"; exit 3; }
[ $# -ge 1 ] || usage
MODE="$1"; shift

if [ "$MODE" = fetch ]; then
  [ $# -eq 2 ] || usage
  NAME="$1"; CODE="$2"; check_run_name "$NAME"
  O="$RENT/out/$NAME/fetch-$(date +%s)"; mkdir -p "$O"; cd "$O"
  HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive "$CODE"
  for f in *.tgz; do [ -f "$f" ] || continue
    L=$(tar -tzf "$f") || refuse "cannot list $f"; echo "== $f"; printf '%s\n' "$L"
    printf '%s\n' "$L" | grep -qE '(^|/)\.\.(/|$)|^/' && refuse "$f holds a '..' or absolute path"
    tar -xzf "$f"; done
  echo "FETCHED into $O:"; ls -R "$O" | head -40
  [ -f times.txt ] && tail -3 times.txt
  exit 0
fi

DRY=0; TOOL=""; ALLOW=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --tool) [ $# -ge 2 ] || usage; TOOL="$2"; shift 2 ;;
    --allow-new) [ $# -ge 2 ] || usage; ALLOW="$2"; shift 2 ;;
    -*) echo "unknown option $1"; usage ;;
    *) POS+=("$1"); shift ;;
  esac
done
case "$MODE" in
  capture) [ ${#POS[@]} -ge 2 ] && [ ${#POS[@]} -le 3 ] && [ -z "$ALLOW" ] || usage ;;
  compare) [ ${#POS[@]} -eq 4 ] && [ -z "$TOOL" ] || usage ;;
  *) usage ;;
esac
SHA=$(git -C "$REPO" rev-parse --verify "${POS[0]}^{commit}") || refuse "unknown commit ${POS[0]}"
SHORT=${SHA:0:7}; TB="${POS[1]}"; [ -f "$TB" ] || refuse "no such tables archive" "$TB"
if [ "$MODE" = capture ]; then NAME="${POS[2]:-snapmain-$SHORT}"; CAP=""
else CAP="${POS[2]}"; NAME="${POS[3]}"
  echo "$CAP" | grep -qE '^/work/[a-z0-9][a-z0-9_-]*/[A-Za-z0-9_-]+$' || refuse "capture dir must be /work/<folder>/<dir>" "$CAP"
fi
check_run_name "$NAME"
[ -z "$ALLOW" ] || echo "$ALLOW" | grep -qE '^[A-Za-z0-9_][A-Za-z0-9_.-]*(,[A-Za-z0-9_][A-Za-z0-9_.-]*)*$' || refuse "--allow-new must be name,name,..." "$ALLOW"
TSHA=""; [ -z "$TOOL" ] || TSHA=$(git -C "$REPO" rev-parse --verify "$TOOL^{commit}") || refuse "unknown --tool commit $TOOL"
DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD) || refuse "no decomp checkout at $REPO/external/pokeplatinum"
GATE="${SNAP_GATE_GB:-88}"; echo "$GATE" | grep -qE '^[1-9][0-9]*$' || refuse "SNAP_GATE_GB must be a whole number" "$GATE"
WORKERS="${SNAP_WORKERS:-16}"; echo "$WORKERS" | grep -qE '^[1-9][0-9]*$' || refuse "SNAP_WORKERS must be a whole number of at least 1" "$WORKERS"
O="$RENT/out/$NAME"; [ -e "$O" ] && refuse "$O exists; names are used once"

FORBID='\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml'
BADPATH='\.\.|^/|^\./'
mkdir -p "$O"
trap 'rc=$?; [ "$rc" = 0 ] || { cd /; rm -rf "$O" || echo "WARNING: could not remove $O"; }' EXIT
# deny <reason> <text> <grep args>: refuse when grep selects a line; a grep error (rc > 1) refuses.
deny() { local why="$1" txt="$2" out rc=0; shift 2
  out=$(printf '%s\n' "$txt" | sed '/^$/d' | grep "$@") || rc=$?
  [ "$rc" -le 1 ] || refuse "grep error rc $rc: grep $*"
  [ -z "$out" ] || refuse "$why" "$out"; }

git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$O/code.tar" "$SHA"
FILES=(code.tar tables.tar.xz)
if [ -n "$TSHA" ]; then
  git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$O/tool.tar" "$TSHA" tools/snap_compare.py
  FILES+=(tool.tar)
fi
cp "$TB" "$O/tables.tar.xz"
TOP=$(git -C "$HERE" rev-parse --show-toplevel)
git -C "$TOP" show HEAD:tools/ops/day/pod/cache_sync.sh | tr -d '\r' > "$O/cache_sync.sh" || refuse "no tools/ops/day/pod/cache_sync.sh at HEAD of $TOP"
FILES+=(cache_sync.sh)
CT=$(tar -tf "$O/code.tar"; [ -z "$TSHA" ] || tar -tf "$O/tool.tar")
TALL=$(tar -tJf "$O/tables.tar.xz"); T=$(printf '%s\n' "$TALL" | grep -v '/$' || true)
deny "tables entry with '..', a leading '/' or './'" "$TALL" -E "$BADPATH"
deny "forbidden files" "$(printf '%s\n%s\n' "$CT" "$TALL")" -iE "$FORBID"
deny "calc or docs-sheet extraction in code.tar" "$CT" -E "$TABLE_DENY"
deny "calc or docs-sheet extraction in tables" "$TALL" -E "$TABLE_DENY"
deny "tables entry outside the allowed data/ paths" "$T" -vE "$TABLE_ALLOW"
[ -n "$T" ] || refuse "tables archive holds no files" "$TB"
tar -cf "$O/payload.tar" -C "$O" "${FILES[@]}"

W="/work/$NAME"
# --workers/--gate-gb only for a snap_compare that has them (older commits run their parts serially).
SRC="$SHA"; [ -z "$TSHA" ] || SRC="$TSHA"; PAR=""
SC=$(git -C "$REPO" show "$SRC:tools/snap_compare.py") || refuse "no tools/snap_compare.py at $SRC"
if [[ "$SC" == *'add_argument("--workers"'* ]]; then PAR=" --workers $WORKERS --gate-gb $GATE"
else echo "NOTE: tools/snap_compare.py at ${SRC:0:8} has no --workers: its parts run one after another"; fi
VOLQ="mountpoint -q /vol 2>/dev/null"; PRE=":"; POST=":"
if [ "$MODE" = capture ]; then
  SCRIPT="tools/snap_compare.py"; [ -z "$TSHA" ] || SCRIPT="$W/tool/tools/snap_compare.py"
  JOB="CAPTURE"; CMD="uv run --frozen python \"$SCRIPT\" capture \"\$W/snap\" 0-199 --repo \"\$W/repo\" --commit $SHA$PAR -v > \"\$W/capture.log\" 2>&1"
  K=""; [ -n "$TSHA" ] || K=$(table_key16 "$SHA" "$TB")
  if [ -n "$TSHA" ]; then echo "NOTE: a --tool re-capture is not stored on the network volume"
  elif [ -z "$K" ]; then echo "NOTE: no table key for $TB (no .key file beside it, no tables_k<key16> name, no $RENT/tables/index.txt line for $SHA): the capture is not stored on the network volume"
  else SKEY="$SHA-$K"; echo "capture store key $SKEY (fetched instead of captured when the pod's volume has it)"
    PRE="if $VOLQ && bash \"\$W/cache_sync.sh\" snap-fetch $SKEY \"\$W\" > \"\$W/store.log\" 2>&1; then echo \"CAPTURE_FROM_VOLUME $SKEY \$(date +%s)\" >> \"\$W/times.txt\"; echo \"CAPTURE_RC=0 \$(date +%s)\" >> \"\$W/times.txt\"; echo ALL_DONE >> \"\$W/times.txt\"; exit 0; fi"
    POST="if [ \"\$rc\" = 0 ] && $VOLQ; then bash \"\$W/cache_sync.sh\" snap-publish \"\$W\" $SKEY >> \"\$W/store.log\" 2>&1; echo \"STORE_RC=\$? \$(date +%s)\" >> \"\$W/times.txt\"; fi"
  fi
else
  JOB="COMPARE"; AN=""; [ -z "$ALLOW" ] || AN=" --allow-new $ALLOW"
  CMD="uv run --frozen python tools/snap_compare.py compare \"$CAP\" 0-199 --repo \"\$W/repo\" --commit $SHA$AN$PAR -v > \"\$W/compare.log\" 2>&1"
  CS=""; K=""
  if [[ "$CAP" =~ ^/work/snapmain-([0-9a-f]{7})/snap$ ]]; then
    CS=$(git -C "$REPO" rev-parse --verify -q "${BASH_REMATCH[1]}^{commit}" 2>/dev/null || true)
    [ -z "$CS" ] || K=$(table_key16 "$CS" "$RENT/tables_${CS:0:8}.tar.xz"); fi
  if [ -z "$K" ]; then echo "NOTE: no store key for $CAP (only /work/snapmain-<short>/snap of a commit with a table key in $RENT/tables): a capture missing on the pod is not fetched from the network volume"
  else CKEY="$CS-$K"; echo "capture store key $CKEY (fetched when $CAP is not on the pod)"
    PRE="if [ ! -f \"$CAP/capture.json\" ] && $VOLQ; then bash \"\$W/cache_sync.sh\" snap-fetch $CKEY \"${CAP%/snap}\" > \"\$W/store.log\" 2>&1; echo \"FETCH_RC=\$? \$(date +%s)\" >> \"\$W/times.txt\"; fi"
  fi
fi
cat > "$O/run.sh" <<EOS
#!/bin/bash
# $MODE of $SHA; written by tools/ops/day/snap.sh
export PATH=/root/.local/bin:/work/_bin:\$PATH UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
W="$W"; printf '%s\n' "$CAP" > "\$W/reads"; cd "\$W/repo"
$(declare -f cgmem)
gate() { local b; while b=\$(cgmem current) || { echo "GATE_NO_CGROUP_MEMORY \$(date +%s)" >> "\$W/times.txt"; exit 1; }; [ \$(( b / 1000000000 )) -ge $GATE ]; do echo "gate wait \$(date +%T)"; sleep 30; done; }
echo "START \$(date +%s)" > "\$W/times.txt"
m=\$(tr -dc 0-9 < /proc/sys/vm/max_map_count 2>/dev/null); [ "\${m:-0}" -ge 1048576 ] 2>/dev/null || { echo "MAP_COUNT_LOW \${m:-none} \$(date +%s)" >> "\$W/times.txt"; exit 1; }
(uv sync --frozen -q) > "\$W/sync.log" 2>&1; rc=\$?; echo "SYNC_RC=\$rc \$(date +%s)" >> "\$W/times.txt"; [ "\$rc" = 0 ] || exit 1
export CACHE_PY="\$W/repo/.venv/bin/python"
$PRE
gate; echo "${JOB}_START \$(date +%s)" >> "\$W/times.txt"
$CMD; rc=\$?; echo "${JOB}_RC=\$rc \$(date +%s)" >> "\$W/times.txt"
$POST
echo ALL_DONE >> "\$W/times.txt"
EOS
tr -d '\r' < "$O/run.sh" > "$O/run.tmp" && mv "$O/run.tmp" "$O/run.sh"
bash -n "$O/run.sh" || refuse "generated run.sh does not parse" "$O/run.sh"

echo "== payload.tar"; tar -tvf "$O/payload.tar"
echo "== code.tar $(printf '%s\n' "$CT" | wc -l) entries; tables $(printf '%s\n' "$T" | wc -l) files"
echo "== run.sh"; cat "$O/run.sh"

if [ "$DRY" = 1 ]; then CODE="<code>"; SPID=""
else
  cd "$O"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup bash -c '"$0" send payload.tar & c=$!; trap "kill $c" TERM; wait "$c"; echo "SEND_RC=$?"' "$RPC" > send.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
  CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true)
  [ -n "$CODE" ] || { echo "no code; send.log:"; cat send.log; kill "$SPID" 2>/dev/null || true; exit 4; }
fi

RB=$(base64 -w0 "$O/run.sh")
UNTOOL=""; [ -z "$TSHA" ] || UNTOOL=" && mkdir tool && tar --no-same-owner -xf tool.tar -C tool"
DM="/work/_decomp/$DECOMP"
PREP="/work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; [ -d $DM ] || { echo \"DECOMP_MISSING: $DM is not on this pod; send any pack.sh run first (its setup.sh creates it), then send this body again under a new name\"; exit 1; }; { tar --no-same-owner -xf payload.tar && mkdir repo && tar --no-same-owner -xf code.tar -C repo && tar --no-same-owner -xJf tables.tar.xz -C repo$UNTOOL && mkdir -p repo/external && ln -sfn $DM repo/external/pokeplatinum; } || { echo UNPACK_FAILED; exit 1; }; echo PREP_OK; setsid nohup bash $W/run.sh > run.out 2>&1 < /dev/null &"
SHC="mkdir -p $W && cd $W && echo $RB | base64 -d > run.sh && chmod +x run.sh && nohup sh -c '$PREP' > prep.log 2>&1 < /dev/null & sleep 20; tail -c 300 $W/prep.log"
python_json() { local s="$1"; s=${s//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; }
printf '{"command": ["sh", "-c", %s], "timeoutSeconds": 25}\n' "$(python_json "$SHC")" > "$O/body.json"
echo "== exec-pod body ($O/body.json):"; cat "$O/body.json"
[ "$DRY" = 1 ] && echo "DRY RUN: nothing sent; delete $O before reusing the name" && exit 0
echo "== finished when exec 'tail -3 $W/times.txt' shows ${JOB}_RC=<n> then ALL_DONE"
