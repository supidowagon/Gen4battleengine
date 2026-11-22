#!/usr/bin/env bash
# POD SIDE, sourced by setup.sh and quick.sh (DAYBOX.md "Quick lane"). Functions only.
# The lane state of a stream lives in LANE_ROOT (default /work/_lane):
#   <stream>.sha        marker: KEY=VALUE lines COMMIT (full sha last synced), TREE (its git tree
#                       hash), TABLES_KEY (key16 of its built tables, empty when unknown),
#                       TABLES_CHASH (64-hex content hash of those tables, tools/ops/content_hash.py,
#                       empty when unknown), REPO (the repo that holds that state) and TIME (epoch).
#                       Written atomically.
#   <stream>.lock       the flock one quick run of the stream holds from its apply to its end
#   <stream>/repo       the lane repo: a clone of a run's repo that quick.sh moves from commit to commit
#   <stream>/BROKEN     the lane repo failed its tree check and was removed; quick.sh treats it as missing
# <stream> is the first '-' field of the run name, after an optional "q-": e4-a0m-d1ab9b7 and
# q-e4-d1ab9b7 are both stream e4.
LANE_ROOT="${LANE_ROOT:-/work/_lane}"

# lane_stream <run name>: the stream, or return 1 when it is not [a-z0-9_]+.
lane_stream() { local r="${1#q-}"; r="${r%%-*}"
  case "$r" in ''|*[!a-z0-9_]*) return 1;; esac; echo "$r"; }

# lane_get <file> <KEY>: the value of the last KEY= line of a marker file (CR dropped).
lane_get() { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1 | tr -d '\r'; }

# lane_stage <repo> [<paths file>]: stages the repo's working tree into the index git uses
# (GIT_INDEX_FILE when set): `git add -A`, then `git add -f` of every path in <paths file> (one
# per line, relative to the repo, literal), so a file of the commit that sits under an ignored
# path (tracked but ignored) is staged too. Without a paths file only `git add -A`.
lane_stage() { git -C "$1" -c core.autocrlf=false add -A > /dev/null 2>&1 || return 1
  [ -z "${2:-}" ] || [ ! -s "$2" ] || git -C "$1" -c core.autocrlf=false --literal-pathspecs add -f --pathspec-from-file="$2" > /dev/null 2>&1; }

# lane_snapshot <repo> <commit>: setup.sh, right after code.tar is unpacked into the empty <repo>:
# git init and one commit of every unpacked file, ignored ones too (`git add -A -f`: the folder
# holds code.tar's files only, so a tracked-but-ignored file of the commit is in the snapshot).
lane_snapshot() { ( cd "$1" && git init -q && git -c core.autocrlf=false add -A -f \
  && git -c user.name=ci -c user.email=ci@localhost commit -qm "snapshot $2" ); }

# lane_tree <repo> [<paths file>]: the git tree hash of the repo's working tree as lane_stage
# gives it (tracked files stay in it even when ignored; untracked git-ignored files, the built
# tables and the venv among them, are not), computed in a copy of the index so the repo's own
# index and files stay as they are. Prints nothing and returns 1 when git fails.
lane_tree() { local r="$1" idx t
  idx="$r/.git/lane-index.$$"; rm -f "$idx"; [ ! -f "$r/.git/index" ] || cp "$r/.git/index" "$idx" || return 1
  t=$(export GIT_INDEX_FILE="$idx"; lane_stage "$r" "${2:-}" && git -C "$r" write-tree 2>/dev/null) \
    || { rm -f "$idx"; return 1; }
  rm -f "$idx"; echo "$t"; }

# lane_marker_put <stream> <commit> <tree> <tables key> <repo> [<tables content hash>]: writes the
# marker through a temp file and mv. The caller holds the stream's lock (quick.sh) or goes through
# lane_write_marker.
lane_marker_put() { local t="$LANE_ROOT/.$1.sha.$$"
  mkdir -p "$LANE_ROOT" && printf 'COMMIT=%s\nTREE=%s\nTABLES_KEY=%s\nTABLES_CHASH=%s\nREPO=%s\nTIME=%s\n' "$2" "$3" "$4" "${6:-}" "$5" "$(date +%s)" > "$t" \
    && mv -f "$t" "$LANE_ROOT/$1.sha"; }

# lane_write_marker <stream> <commit> <repo> <tables key> <tree> [<tables content hash>]: setup.sh, at the end of a setup
# (just before SETUP_END). <tree> is meta.env's TREE (the commit's tree on the PC). The marker is
# written only when the repo's HEAD tree is <tree> (the snapshot is exactly the commit) and the
# working tree's is HEAD's (the built tables are git-ignored there); else a later delta's tree
# check could never match: prints LANE_MARKER_SKIPPED <reason> and returns 1. Waits for the
# stream's lock at most LANE_LOCK_S (default 5) s: a quick run of the stream holds it through its
# pytest; on a timeout the old marker stays and names another commit, so the next quick check of
# that tip falls back to a full run.
lane_write_marker() { local s="$1" c="$2" r="$3" k="${4:-}" want="${5:-}" ch="${6:-}" head work
  [ -n "$want" ] || { echo "LANE_MARKER_SKIPPED no TREE in meta.env"; return 1; }
  head=$(git -C "$r" rev-parse 'HEAD^{tree}') || return 1
  [ "$head" = "$want" ] || { echo "LANE_MARKER_SKIPPED the snapshot tree ${head:0:8} is not the commit's tree ${want:0:8}"; return 1; }
  work=$(lane_tree "$r") || return 1
  [ "$head" = "$work" ] || { echo "LANE_MARKER_SKIPPED the working tree differs from HEAD (tables not git-ignored?)"; return 1; }
  mkdir -p "$LANE_ROOT" || return 1
  ( flock -w "${LANE_LOCK_S:-5}" 9 || { echo "LANE_MARKER_SKIPPED lock of stream $s busy"; exit 1; }
    rm -f "$LANE_ROOT/$s/BROKEN"
    lane_marker_put "$s" "$c" "$head" "$k" "$r" "$ch" ) 9> "$LANE_ROOT/$s.lock"; }
