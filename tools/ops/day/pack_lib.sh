#!/usr/bin/env bash
# PC SIDE, sourced by pack.sh (full pack and --delta). Functions only; the caller (pack.sh) has
# sourced tools/ops/common.sh and sets HERE (tools/ops/day), P, RENT, RPC and the patterns FORBID,
# BADPATH, LINE, COMMENT. Not in common.sh: common.sh is an input of the table key.
. "$HERE/table_key16.sh"

refuse() { echo "REFUSED: $1"; echo "$2"; exit 3; }
# pick <grep args>: grep stdin, blank lines dropped. rc 0 or 1 pass; any other rc refuses.
pick() { local rc=0; sed '/^$/d' | grep "$@" || rc=$?
  [ "$rc" -le 1 ] || { echo "REFUSED: grep error rc $rc: grep $*" >&2; exit 3; }; }
# deny <reason> <text> <grep args>: refuse when grep selects any line of <text>.
deny() { local why="$1" txt="$2" out; shift 2
  out=$(printf '%s\n' "$txt" | pick "$@") || exit 3; [ -z "$out" ] || refuse "$why" "$out"; }

# pack_check_patterns: every pattern must compile: on empty input grep exits 1 for a valid pattern, 2 for a bad one.
pack_check_patterns() { local pat rc
  for pat in "$FORBID" "$BADPATH" "$LINE" "$COMMENT" "$TABLE_ALLOW" "$TABLE_DENY" "$@"; do
    rc=0; grep -E "$pat" < /dev/null || rc=$?; [ "$rc" = 1 ] || refuse "grep pattern does not compile (rc $rc)" "$pat"; done; }

# pack_pod_scripts: the pod scripts of this checkout's committed state only. Sets TOP and OPS.
pack_pod_scripts() { local dirty
  TOP=$(git -C "$HERE" rev-parse --show-toplevel)
  dirty=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
  [ -z "$dirty" ] || refuse "uncommitted changes in tools/ops/day/pod; commit them first" "$dirty"
  OPS=$(git -C "$TOP" rev-parse HEAD)
  pod_main_check "$TOP"
  [ "${PACK_OFF_MAIN_POD:-}" = 1 ] && echo "WARNING: PACK_OFF_MAIN_POD=1: the pod scripts come from HEAD $(git -C "$TOP" rev-parse --short HEAD), not checked against main; a road test of an ops branch only, stated in its report"
  return 0; }

# pack_draw_beacons: sets WF and WD. Two fresh random codephrases per run, WATCH_F (first failure)
# and WATCH_D (results), for bin/notify.sh (runpodctl send --code) and tools/ops/day/wait.sh.
# runpodctl v3.1.1's croc fork (cmd/transfer/croc.go:551,722) puts a transfer in the relay room
# code[:3] and its data rooms in sha256(code[:5]) (:1293), so each code starts with 3 random letters
# (never a digit, as runpodctl's own codes do) and the two codes differ there. Never printed here.
pack_draw_beacons() { local a=abcdefghijklmnopqrstuvwxyz w
  draw() { local l="" b
    for b in $(od -An -N3 -tu1 /dev/urandom); do l="$l${a:$((b % 26)):1}"; done
    echo "$l$(od -An -N15 -tx1 /dev/urandom | tr -d ' \n')"; }
  WF=$(draw); WD=$(draw)
  while [ "${WD:0:3}" = "${WF:0:3}" ]; do WD=$(draw); done
  for w in "$WF" "$WD"; do echo "$w" | grep -qE '^[a-z]{3}[0-9a-f]{30}$' || refuse "could not draw a random beacon code" "${#w} characters"; done; }

# pack_send: in $P, starts the detached send of payload.tar and sets CODE (exit 4 without one).
# SEND_RC=<exit code of the send> ends send.log: wait.sh reads it.
pack_send() {
  cd "$P"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup bash -c '"$0" send payload.tar & c=$!; trap "kill $c" TERM; wait "$c"; echo "SEND_RC=$?"' "$RPC"   > send.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
  CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true)
  [ -n "$CODE" ] || { echo "no code; send.log:"; cat send.log; kill "$SPID" 2>/dev/null || true; exit 4; }; }

# pack_delta_build <repo> <old> <new> <outdir>: writes <outdir>/delta.tar (every file git diff
# --no-renames --name-status <old> <new> lists as A, M or T, taken from <new>; core.autocrlf off,
# literal pathspecs) and <outdir>/deletions.txt (the D paths, one per line). A rename comes out
# as D + A. Refuses: another status, a path with a newline, '..', a leading '/' or './',
# a forbidden file, more than DELTA_MAX_FILES (default 1000) changed files (a delta that large is a
# full run). Sets DELTA_ADD and DELTA_DEL (counts) and DELTA_PATHS (the A/M/T paths, one per line).
pack_delta_build() {
  local repo=$1 old=$2 new=$3 out=$4 st f paths=() dels=() max="${DELTA_MAX_FILES:-1000}"
  git -C "$repo" -c core.quotepath=off diff --no-renames --name-status -z "$old" "$new" > "$out/changes.z" \
    || refuse "git diff $old $new failed" ""
  while IFS= read -r -d '' st && IFS= read -r -d '' f; do
    case "$f" in *$'\n'*) refuse "a changed path holds a newline" "$f";; esac
    case "$st" in A|M|T) paths+=("$f");; D) dels+=("$f");; *) refuse "change status $st is not supported in a delta (only A, M, T, D)" "$f";; esac
  done < "$out/changes.z"
  rm -f "$out/changes.z"
  [ $(( ${#paths[@]} + ${#dels[@]} )) -le "$max" ] || refuse "delta has $(( ${#paths[@]} + ${#dels[@]} )) changed files, over $max: send a normal run" ""
  DELTA_PATHS=$(printf '%s\n' "${paths[@]+"${paths[@]}"}" | sed '/^$/d')
  deny "changed path with '..', a leading '/' or './'" "$DELTA_PATHS
$(printf '%s\n' "${dels[@]+"${dels[@]}"}")" -E "$BADPATH"
  deny "forbidden files in the delta" "$DELTA_PATHS" -iE "$FORBID"
  deny "calc or docs-sheet extraction in the delta" "$DELTA_PATHS" -E "$TABLE_DENY"
  if [ "${#paths[@]}" -gt 0 ]; then
    git -C "$repo" -c core.autocrlf=false --literal-pathspecs archive --format=tar -o "$out/delta.tar" "$new" -- "${paths[@]}" \
      || refuse "git archive of the changed files failed" ""
  else tar -cf "$out/delta.tar" -T /dev/null; fi   # no path: git archive would pack the whole tree
  printf '%s\n' "${dels[@]+"${dels[@]}"}" | sed '/^$/d' > "$out/deletions.txt"
  DELTA_ADD=${#paths[@]}; DELTA_DEL=${#dels[@]}; }
