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
# pack_delta_batch <repo> <new> <outdir> <path>...: git archive of those paths of <new>, appended
# to <outdir>/delta.tar.
pack_delta_batch() { local repo=$1 new=$2 out=$3; shift 3
  git -C "$repo" -c core.autocrlf=false --literal-pathspecs archive --format=tar -o "$out/delta.part.tar" "$new" -- "$@" \
    || refuse "git archive of the changed files failed" ""
  tar -A -f "$out/delta.tar" "$out/delta.part.tar" || refuse "tar --concatenate of the delta failed" ""
  rm -f "$out/delta.part.tar"; }
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
  # git archive per batch of at most DELTA_ARG_CHARS (default 12000) characters of paths, so no
  # command line nears Git Bash's 32 KB limit; the batches are joined with tar --concatenate.
  tar -cf "$out/delta.tar" -T /dev/null   # no path: an empty tar (git archive would pack the whole tree)
  local batch=() len=0 lim="${DELTA_ARG_CHARS:-12000}"
  for f in "${paths[@]+"${paths[@]}"}"; do
    if [ "${#batch[@]}" -gt 0 ] && [ $(( len + ${#f} + 1 )) -gt "$lim" ]; then
      pack_delta_batch "$repo" "$new" "$out" "${batch[@]}"; batch=(); len=0; fi
    batch+=("$f"); len=$(( len + ${#f} + 1 )); done
  [ "${#batch[@]}" = 0 ] || pack_delta_batch "$repo" "$new" "$out" "${batch[@]}"
  printf '%s\n' "${dels[@]+"${dels[@]}"}" | sed '/^$/d' > "$out/deletions.txt"
  DELTA_ADD=${#paths[@]}; DELTA_DEL=${#dels[@]}; }

# pack_code_tar <repo> <commit> <out tar>: a full send's code.tar, the git archive of <commit>
# (git objects only: no untracked or ignored file of the working tree can get in).
pack_code_tar() { git -C "$1" -c core.autocrlf=false archive --format=tar -o "$3" "$2"; }

# pack_sent_key <full sha>: the table key16 an earlier send of that commit recorded: TABLES_KEY of
# the newest $RENT/out/<run>/meta.env whose COMMIT is <sha> (normal and delta sends write both;
# NEW is a delta's COMMIT). The quick lane's marker on the pod holds that same key. Prints
# nothing when no kept meta.env has one.
pack_sent_key() { local f k
  while IFS= read -r f; do
    grep -qx "COMMIT=$1" "$f" || continue
    k=$(sed -n 's/^TABLES_KEY=//p' "$f" | tail -1 | tr -d '\r')
    echo "$k" | grep -qE '^[0-9a-f]{16}$' && { echo "$k"; return 0; };
  done < <(ls -t "$RENT"/out/*/meta.env 2>/dev/null); return 0; }

# pack_sent_chash <full sha>: as pack_sent_key, for TABLES_CHASH (64 hex): the content hash an
# earlier send of that commit recorded, only from a meta.env whose TABLES_CHASH_V is TABLE_CHASH_V
# (table_key16.sh): a hash of another definition, or of none, is unknown and the key decides.
pack_sent_chash() { local f k v
  while IFS= read -r f; do
    grep -qx "COMMIT=$1" "$f" || continue
    v=$(sed -n 's/^TABLES_CHASH_V=//p' "$f" | tail -1 | tr -d '')
    [ -n "$v" ] && [ "$v" = "${TABLE_CHASH_V:-}" ] || continue
    k=$(sed -n 's/^TABLES_CHASH=//p' "$f" | tail -1 | tr -d '\r')
    echo "$k" | grep -qE '^[0-9a-f]{64}$' && { echo "$k"; return 0; };
  done < <(ls -t "$RENT"/out/*/meta.env 2>/dev/null); return 0; }

# pack_body_env: the exec-pod body's env field (',"env":{...}') holding the knobs set on the PC for
# pack.sh, only those set (PACK_ENV_NAMES): GATE_GB, KILL_GB, MAXP (the shape's values,
# docs/process.md section 5), SPACING (the runner's pause between unit starts) and the sweep's
# SWEEP_P, SWEEP_W, SWEEP_CAP, SWEEP_NICE, SWEEP_JOB_S, SWEEP_SPACING, SWEEP_POLL, SWEEP_REPLAY_S
# (tools/ops/day/pod/sweep.sh; setup.sh writes them to run.env, sweep_body.sh's start.sh reads them
# from there when the PC's environment of sweep_body.sh does not set them). Nothing when none is
# set. Refuses a value that is not a whole number (SWEEP_SPACING and SWEEP_POLL: a decimal, as
# sweep_body.sh); warns on stderr when it sets any.
PACK_ENV_NAMES="GATE_GB KILL_GB MAXP SPACING SWEEP_P SWEEP_W SWEEP_CAP SWEEP_NICE SWEEP_JOB_S SWEEP_SPACING SWEEP_POLL SWEEP_REPLAY_S"
pack_body_env() { local n v out=""
  for n in $PACK_ENV_NAMES; do v=${!n-}; [ -n "$v" ] || continue
    case "$n" in
      SWEEP_SPACING|SWEEP_POLL)   # sleep lengths: decimals as sweep_body.sh accepts them
        [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]] || refuse "$n '$v' is not a number" "";;
      *) case "$v" in *[!0-9]*) refuse "$n '$v' is not a whole number" "";; esac;;
    esac
    out="$out${out:+,}\"$n\":\"$v\""; done
  [ -z "$out" ] && return 0
  # A knob exported in the shell reaches every body, normal runs included: say so on stderr.
  echo "WARNING pack.sh: ${out//\"/} from the environment go into this body's env" >&2
  printf ',"env":{%s}' "$out"; }

# pack_sweep_tools <checkout> <commit> <bin dir>: tools/ops/sweep/*.py (extract.py, replay.py,
# collect.py, seeds.py, nodes.txt) as committed at <commit> of <checkout>, LF, into <bin dir>/sweep/.
# sweep.sh takes them from there when the run's repo has no tools/ops/sweep (a sweep of a commit
# older than the sweep tools). Same source rule as the pod scripts: HEAD of the checkout that runs pack.sh.
pack_sweep_tools() { local s f; s=$(mktemp -d "$RENT/sweepsrc.XXXXXX") || return 1; mkdir -p "$3/sweep"
  git -C "$1" -c core.autocrlf=false archive --format=tar "$2" tools/ops/sweep 2>/dev/null | tar -xf - -C "$s" || { rm -rf "$s"; return 1; }
  for f in "$s"/tools/ops/sweep/*.py "$s"/tools/ops/sweep/nodes.txt; do
    [ -f "$f" ] || continue; tr -d '\r' < "$f" > "$3/sweep/$(basename "$f")" || { rm -rf "$s"; return 1; }; done
  rm -rf "$s"; [ -f "$3/sweep/replay.py" ] && [ -f "$3/sweep/extract.py" ]; }
