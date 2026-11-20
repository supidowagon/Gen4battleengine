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
#   COMPARE_RC=<n> <epoch>, ALL_DONE.
#   Env: SNAP_GATE_GB (default 88): run.sh starts the capture or compare only while container
#   memory is under it; pass the pod's GATE_GB (176 on a 256 GB pod).
# fetch: receives what the pod sends (DAYBOX.md "Snapshot compare", Fetch) into
#   $RENT/out/<name>/fetch-<epoch>/ and unpacks every .tgz received.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
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
if [ "$MODE" = capture ]; then
  SCRIPT="tools/snap_compare.py"; [ -z "$TSHA" ] || SCRIPT="$W/tool/tools/snap_compare.py"
  JOB="CAPTURE"; CMD="uv run --frozen python \"$SCRIPT\" capture \"\$W/snap\" 0-199 --repo \"\$W/repo\" --commit $SHA -v > \"\$W/capture.log\" 2>&1"
else
  JOB="COMPARE"; AN=""; [ -z "$ALLOW" ] || AN=" --allow-new $ALLOW"
  CMD="uv run --frozen python tools/snap_compare.py compare \"$CAP\" 0-199 --repo \"\$W/repo\" --commit $SHA$AN -v > \"\$W/compare.log\" 2>&1"
fi
cat > "$O/run.sh" <<EOS
#!/bin/bash
# $MODE of $SHA; written by tools/ops/day/snap.sh
export PATH=/root/.local/bin:/work/_bin:\$PATH UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
W="$W"; printf '%s\n' "$CAP" > "\$W/reads"; cd "\$W/repo"
gate() { while [ \$(( \$(cat /sys/fs/cgroup/memory.current) / 1000000000 )) -ge $GATE ]; do echo "gate wait \$(date +%T)"; sleep 30; done; }
echo "START \$(date +%s)" > "\$W/times.txt"
(uv sync --frozen -q) > "\$W/sync.log" 2>&1; rc=\$?; echo "SYNC_RC=\$rc \$(date +%s)" >> "\$W/times.txt"; [ "\$rc" = 0 ] || exit 1
gate; echo "${JOB}_START \$(date +%s)" >> "\$W/times.txt"
$CMD; echo "${JOB}_RC=\$? \$(date +%s)" >> "\$W/times.txt"
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
