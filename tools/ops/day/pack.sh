#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>
# Bundles the git archive of <commit> + built tables (.json/.npz only) + the test list + the
# pod scripts of tools/ops/day/pod (LF) into $NUZLINE_SCRATCH/rent/out/<run>/payload.tar, lists
# it, refuses forbidden files, starts a detached send and prints the exec-pod body that receives
# the payload into /work/<run> and starts bin/setup.sh there.
# <run>: [a-zA-Z0-9_.-] segments joined by "/" (e2/m26 -> /work/e2/m26); used once.
# Env: MARK (pytest -m for lines without A; default "not slow"), RUNPODCTL (PC binary).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO="${NUZLINE_REPO:-$(cd "$HERE/../../.." && pwd)}"
RENT="${NUZLINE_SCRATCH:-$REPO/_scratch}/rent"
RPC="${RUNPODCTL:-$RENT/bin/runpodctl-windows-amd64.exe}"
[ $# -eq 4 ] || { echo "usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>"; exit 2; }
C="$1"; TB="$2"; RUN="$3"; LIST="$4"
echo "$RUN" | grep -qE '^[a-zA-Z0-9][a-zA-Z0-9_.-]*(/[a-zA-Z0-9][a-zA-Z0-9_.-]*)*$' && ! echo "$RUN" | grep -q '\.\.' \
  || { echo "REFUSED: bad run name"; exit 2; }
P="$RENT/out/$RUN"; [ -e "$P" ] && { echo "REFUSED: $P exists; run names are used once"; exit 3; }
mkdir -p "$P/bin"
COMMIT=$(git -C "$REPO" rev-parse --verify "$C^{commit}")
DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD)
git -C "$REPO" archive --format=tar -o "$P/code.tar" "$COMMIT"
cp "$TB" "$P/tables.tar.xz"; tr -d '\r' < "$LIST" > "$P/files.txt"
for f in "$HERE"/pod/*.sh; do tr -d '\r' < "$f" > "$P/bin/$(basename "$f")"; done
printf 'COMMIT=%s\nDECOMP=%s\nMARK="%s"\n' "$COMMIT" "$DECOMP" "${MARK:-not slow}" > "$P/meta.env"
BAD=$( { tar -tf "$P/code.tar"; tar -tJf "$P/tables.tar.xz"; } | grep -iE '\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml' || true)
[ -z "$BAD" ] || { echo "FORBIDDEN:"; echo "$BAD"; rm -rf "$P"; exit 3; }
NT=$(tar -tJf "$P/tables.tar.xz" | grep -vcE '\.(json|npz)$' || true); [ "$NT" = 0 ] || { echo "REFUSED: non-table file in tables"; rm -rf "$P"; exit 3; }
tar -cf "$P/payload.tar" -C "$P" code.tar tables.tar.xz meta.env files.txt bin
echo "== payload.tar"; tar -tvf "$P/payload.tar"
echo "== code.tar $(tar -tf "$P/code.tar" | wc -l) entries; tables $(tar -tJf "$P/tables.tar.xz" | wc -l) files (.json/.npz):"
tar -tJf "$P/tables.tar.xz" | sed 's/^/  /'; cat "$P/meta.env"
cd "$P"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send payload.tar > send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r'); [ -n "$CODE" ] || { echo "no code; see $P/send.log"; exit 4; }
echo "== exec-pod body:"
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c '/work/_bin/runpodctl receive $CODE && tar --no-same-owner -xf payload.tar && bash bin/setup.sh' > setup.log 2>&1 < /dev/null & sleep 15; tail -3 /work/$RUN/setup.log\"],\"timeoutSeconds\":25}"
