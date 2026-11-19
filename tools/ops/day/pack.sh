#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>
# Bundles the git archive of <commit> + built tables + the test list + the pod scripts of
# tools/ops/day/pod (from git at this checkout's HEAD, LF) into $RENT/out/<run>/payload.tar,
# lists it, refuses forbidden files, starts a detached send and prints the exec-pod body that
# receives the payload into /work/<run> and starts bin/setup.sh there. <run>: [a-z0-9_-], used once.
# Refuses: uncommitted changes under tools/ops/day/pod; a files.txt line outside the list format
# (tools/ops/day/pod/setup.sh); forbidden files; table entries outside TABLE_ALLOW or in TABLE_DENY.
# Env: MARK (pytest -m for lines without A; default "not slow"), runner knobs go in the exec body.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
[ $# -eq 4 ] || { echo "usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>"; exit 2; }
C="$1"; TB="$2"; RUN="$3"; LIST="$4"; check_run_name "$RUN"
P="$RENT/out/$RUN"; [ -e "$P" ] && { echo "REFUSED: $P exists; run names are used once"; exit 3; }
refuse() { echo "REFUSED: $1"; echo "$2"; rm -rf "$P"; exit 3; }

# Pod scripts: committed state of this checkout only.
TOP=$(git -C "$HERE" rev-parse --show-toplevel)
DIRTY=$(git -C "$TOP" status --porcelain -- tools/ops/day/pod)
[ -z "$DIRTY" ] || refuse "uncommitted changes in tools/ops/day/pod; commit them first" "$DIRTY"
OPS=$(git -C "$TOP" rev-parse HEAD)

# Test list: "<test file, folder or node id> <ids per chunk> [A]"; comments and blank lines allowed.
BAD=$(tr -d '\r' < "$LIST" | grep -vE '^[[:space:]]*(#|$)' | grep -vE '^tests[^[:space:]]* [1-9][0-9]*( A)?$' || true)
[ -z "$BAD" ] || refuse "files.txt lines outside the format '<tests path or id> <ids per chunk> [A]'" "$BAD"

mkdir -p "$P/bin"
COMMIT=$(git -C "$REPO" rev-parse --verify "$C^{commit}")
DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD)
git -C "$REPO" archive --format=tar -o "$P/code.tar" "$COMMIT"
cp "$TB" "$P/tables.tar.xz"; tr -d '\r' < "$LIST" > "$P/files.txt"
mkdir "$P/podsrc"; git -C "$TOP" archive --format=tar "$OPS" tools/ops/day/pod | tar -xf - -C "$P/podsrc"
for f in "$P"/podsrc/tools/ops/day/pod/*.sh; do tr -d '\r' < "$f" > "$P/bin/$(basename "$f")"; done
rm -rf "$P/podsrc"
printf 'COMMIT=%s\nDECOMP=%s\nOPS=%s\nMARK="%s"\n' "$COMMIT" "$DECOMP" "$OPS" "${MARK:-not slow}" > "$P/meta.env"
CT=$(tar -tf "$P/code.tar"); T=$(tar -tJf "$P/tables.tar.xz" | grep -v '/$')
BAD=$(printf '%s\n%s\n' "$CT" "$T" | grep -iE '\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml' || true)
[ -z "$BAD" ] || refuse "forbidden files" "$BAD"
BAD=$(echo "$CT" | grep -E "$TABLE_DENY" || true); [ -z "$BAD" ] || refuse "calc or docs-sheet extraction in code.tar" "$BAD"
BAD=$(echo "$T" | grep -E '\.\.|^/|^\./' || true); [ -z "$BAD" ] || refuse "tables entry with '..', '/' or './'" "$BAD"
BAD=$(echo "$T" | grep -E "$TABLE_DENY" || true); [ -z "$BAD" ] || refuse "calc or docs-sheet extraction in tables" "$BAD"
BAD=$(echo "$T" | grep -vE "$TABLE_ALLOW" || true); [ -z "$BAD" ] || refuse "tables entry outside the allowed data/ paths" "$BAD"
tar -cf "$P/payload.tar" -C "$P" code.tar tables.tar.xz meta.env files.txt bin
echo "== payload.tar"; tar -tvf "$P/payload.tar"
echo "== code.tar $(echo "$CT" | wc -l) entries; tables $(echo "$T" | wc -l) files:"
echo "$T" | sed 's/^/  /'; cat "$P/meta.env"
cd "$P"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send payload.tar > send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true); [ -n "$CODE" ] || { echo "no code; see $P/send.log"; exit 4; }
echo "== exec-pod body:"
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > setup.pid; /work/_bin/runpodctl receive $CODE && tar --no-same-owner -xf payload.tar && bash bin/setup.sh' > setup.log 2>&1 < /dev/null & sleep 15; tail -3 /work/$RUN/setup.log\"],\"timeoutSeconds\":25}"
