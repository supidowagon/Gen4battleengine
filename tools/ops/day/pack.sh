#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>
# Bundles the git archive of <commit> + built tables + the test list + the pod scripts of
# tools/ops/day/pod (from git at this checkout's HEAD, LF) into $RENT/out/<run>/payload.tar,
# lists it, refuses forbidden files, starts a detached send and prints the exec-pod body that
# receives the payload into /work/<run>, starts bin/notify.sh, then bin/setup.sh there; a failed
# receive or unpack ends setup.log with RECEIVE_FAILED or UNPACK_FAILED. <run>: [a-z0-9_-], used once.
# Refuses: uncommitted changes under tools/ops/day/pod; a files.txt line outside the list format
# (tools/ops/day/pod/setup.sh); forbidden files; table entries (directories included) with '..'
# or a leading '/' or './'; table entries outside TABLE_ALLOW or in TABLE_DENY; a grep error in
# any check (the checks fail closed). Any failure after $RENT/out/<run> is created removes it.
# Env: MARK (pytest -m for lines without A; default "not slow"), runner knobs go in the exec body.
# meta.env also gets WATCH_F and WATCH_D (fresh, random beacon codes) for bin/notify.sh and
# tools/ops/day/wait.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
[ $# -eq 4 ] || { echo "usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>"; exit 2; }
C="$1"; TB="$2"; RUN="$3"; LIST="$4"; check_run_name "$RUN"
P="$RENT/out/$RUN"; [ -e "$P" ] && { echo "REFUSED: $P exists; run names are used once"; exit 3; }
refuse() { echo "REFUSED: $1"; echo "$2"; exit 3; }

FORBID='\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml'
BADPATH='\.\.|^/|^\./'
LINE='^tests[^[:space:]]* [1-9][0-9]*( A)?$'
COMMENT='^[[:space:]]*(#|$)'
# Every pattern must compile: on empty input grep exits 1 for a valid pattern, 2 for a bad one.
for pat in "$FORBID" "$BADPATH" "$LINE" "$COMMENT" "$TABLE_ALLOW" "$TABLE_DENY"; do
  rc=0; grep -E "$pat" < /dev/null || rc=$?; [ "$rc" = 1 ] || refuse "grep pattern does not compile (rc $rc)" "$pat"; done
# pick <grep args>: grep stdin, blank lines dropped. rc 0 or 1 pass; any other rc refuses.
pick() { local rc=0; sed '/^$/d' | grep "$@" || rc=$?
  [ "$rc" -le 1 ] || { echo "REFUSED: grep error rc $rc: grep $*" >&2; exit 3; }; }
# deny <reason> <text> <grep args>: refuse when grep selects any line of <text>.
deny() { local why="$1" txt="$2" out; shift 2
  out=$(printf '%s\n' "$txt" | pick "$@") || exit 3; [ -z "$out" ] || refuse "$why" "$out"; }

# Pod scripts: committed state of this checkout only.
TOP=$(git -C "$HERE" rev-parse --show-toplevel)
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || refuse "uncommitted changes in tools/ops/day/pod; commit them first" "$DIRTY"
OPS=$(git -C "$TOP" rev-parse HEAD)
[ -f "$LIST" ] || refuse "no such test list" "$LIST"

mkdir -p "$P/bin"
# From here every failure removes the payload folder, so the run name stays usable.
trap 'rc=$?; [ "$rc" = 0 ] || { cd /; rm -rf "$P" || echo "WARNING: could not remove $P"; }' EXIT

# Test list: "<test file, folder or node id> <ids per chunk> [A]". The uploaded copy drops comment
# and blank lines, and that copy is what gets checked.
L=$(tr -d '\r' < "$LIST" | pick -vE "$COMMENT") || exit 3
printf '%s\n' "$L" | sed '/^$/d' > "$P/files.txt"
[ -s "$P/files.txt" ] || refuse "test list has no entries" "$LIST"
deny "files.txt lines outside the format '<tests path or id> <ids per chunk> [A]'" "$(cat "$P/files.txt")" -vE "$LINE"

COMMIT=$(git -C "$REPO" rev-parse --verify "$C^{commit}")
DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD)
git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$P/code.tar" "$COMMIT"
cp "$TB" "$P/tables.tar.xz"
mkdir "$P/podsrc"; git -C "$TOP" -c core.autocrlf=false archive --format=tar "$OPS" tools/ops/day/pod | tar -xf - -C "$P/podsrc"
for f in "$P"/podsrc/tools/ops/day/pod/*.sh; do tr -d '\r' < "$f" > "$P/bin/$(basename "$f")"; done
rm -rf "$P/podsrc"
# Beacon codes: two fresh random codephrases per run, WATCH_F (first failure) and WATCH_D (results),
# for bin/notify.sh (runpodctl send --code) and tools/ops/day/wait.sh. runpodctl v3.1.1's croc fork
# (cmd/transfer/croc.go:551,722) puts a transfer in the relay room code[:3] and its data rooms in
# sha256(code[:5]) (:1293), so each code starts with 3 random letters (never a digit, as runpodctl's own
# codes do) and the two codes differ there. Never printed here.
draw() { local a=abcdefghijklmnopqrstuvwxyz l="" b
  for b in $(od -An -N3 -tu1 /dev/urandom); do l="$l${a:$((b % 26)):1}"; done
  echo "$l$(od -An -N15 -tx1 /dev/urandom | tr -d ' \n')"; }
WF=$(draw); WD=$(draw)
while [ "${WD:0:3}" = "${WF:0:3}" ]; do WD=$(draw); done
for w in "$WF" "$WD"; do echo "$w" | grep -qE '^[a-z]{3}[0-9a-f]{30}$' || refuse "could not draw a random beacon code" "${#w} characters"; done
printf 'COMMIT=%s\nDECOMP=%s\nOPS=%s\nMARK="%s"\nWATCH_F=%s\nWATCH_D=%s\n' "$COMMIT" "$DECOMP" "$OPS" "${MARK:-not slow}" "$WF" "$WD" > "$P/meta.env"
CT=$(tar -tf "$P/code.tar"); TALL=$(tar -tJf "$P/tables.tar.xz")   # TALL: directories included
T=$(printf '%s\n' "$TALL" | pick -v '/$') || exit 3                  # T: files only
deny "tables entry with '..', a leading '/' or './'" "$TALL" -E "$BADPATH"
deny "forbidden files" "$(printf '%s\n%s\n' "$CT" "$TALL")" -iE "$FORBID"
deny "calc or docs-sheet extraction in code.tar" "$CT" -E "$TABLE_DENY"
deny "calc or docs-sheet extraction in tables" "$TALL" -E "$TABLE_DENY"
deny "tables entry outside the allowed data/ paths" "$T" -vE "$TABLE_ALLOW"
tar -cf "$P/payload.tar" -C "$P" code.tar tables.tar.xz meta.env files.txt bin
echo "== payload.tar"; tar -tvf "$P/payload.tar"
echo "== code.tar $(echo "$CT" | wc -l) entries; tables $(echo "$T" | wc -l) files:"
echo "$T" | sed 's/^/  /'; grep -v '^WATCH_' "$P/meta.env"
cd "$P"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send payload.tar > send.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true)
[ -n "$CODE" ] || { echo "no code; send.log:"; cat send.log; kill "$SPID" 2>/dev/null || true; exit 4; }
echo "== exec-pod body:"
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > setup.pid; /work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; tar --no-same-owner -xf payload.tar || { echo UNPACK_FAILED; exit 1; }; setsid nohup bash bin/notify.sh > notify.log 2>&1 < /dev/null & bash bin/setup.sh' > setup.log 2>&1 < /dev/null & sleep 15; tail -3 /work/$RUN/setup.log\"],\"timeoutSeconds\":25}"
echo "== then, as one background Bash call: bash tools/ops/day/wait.sh $RUN [--until done|first-failure] [--timeout <hours>]"
