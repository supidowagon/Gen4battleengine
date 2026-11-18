#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage:
#   bash "$NUZLINE_REPO/tools/ops/day/daybox.sh" <worktree> <commit> <run name> -- <pytest args...>
# Packs <commit> (git archive) + the worktree's BUILT tables (git-ignored or modified
# data/**/*.json|*.npz) into one payload.tar, starts `runpodctl send` detached and prints
# the exec-pod command to run on the pod. Uploads nothing else.
set -euo pipefail
: "${NUZLINE_SCRATCH:?set NUZLINE_SCRATCH (git-bash style path of the scratch folder)}"
RENT="$NUZLINE_SCRATCH/rent"
DAY="$RENT/day"   # holds POD_ID (one line: the current pod id, never committed) and out/
RPC="$RENT/bin/runpodctl-windows-amd64.exe"
[ $# -ge 4 ] && [ "$4" = "--" ] || { echo "usage: daybox.sh <worktree> <commit> <run name> -- <pytest args...>"; exit 2; }
WT="$1"; COMMIT_IN="$2"; RUN="$3"; shift 4
[ $# -ge 1 ] || { echo "REFUSED: no pytest args"; exit 2; }
case "$RUN" in *[!a-zA-Z0-9_.-]*|_*|"") echo "REFUSED: run name must be [a-zA-Z0-9_.-], not starting with _"; exit 2;; esac
cd "$WT"
COMMIT=$(git rev-parse --verify "$COMMIT_IN^{commit}")
[ -f data/gen4_platinum_kaizo/manifest.json ] || { echo "REFUSED: data/gen4_platinum_kaizo/manifest.json missing in $WT - rebuild the vanilla and Kaizo tables first"; exit 3; }
[ -f data/gen4_platinum/tables.npz ] || { echo "REFUSED: data/gen4_platinum/tables.npz missing in $WT - rebuild tables first"; exit 3; }
DECOMP=$(git -C external/pokeplatinum rev-parse HEAD 2>/dev/null || echo 579d2d5c65e736a1a4268fb7acbfd70712177c6c)
OUT="$DAY/out/$RUN"
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists - run names are one-shot, pick a new one"; exit 3; }
mkdir -p "$OUT"
git archive --format=tar -o "$OUT/code.tar" "$COMMIT"
# built tables only: ignored or modified-vs-commit files under data/ ending .json/.npz; never a backup folder
{ git ls-files -o -i --exclude-standard -- data; git diff --name-only "$COMMIT" -- 'data/gen4_platinum*/'; } \
  | grep -E '\.(json|npz)$' | grep -viE '(^|/)backups?/' | sort -u \
  | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$OUT/tables.list"
grep -q '^data/gen4_platinum_kaizo/manifest.json$' "$OUT/tables.list" || { echo "REFUSED: Kaizo manifest not in the table list"; exit 3; }
tar -cJf "$OUT/tables.tar.xz" -T "$OUT/tables.list"
printf 'COMMIT=%s\nDECOMP=%s\n' "$COMMIT" "$DECOMP" > "$OUT/meta.env"
printf '%s\n' "$@" > "$OUT/args.txt"
tar -cf "$OUT/payload.tar" -C "$OUT" code.tar tables.tar.xz meta.env args.txt
rm "$OUT/code.tar" "$OUT/tables.tar.xz"
echo "payload: $(du -k "$OUT/payload.tar" | cut -f1) KB, $(wc -l < "$OUT/tables.list") table files, commit $COMMIT"
# send detached; the send stays alive until the pod receives (or ~ the code expires)
cd "$OUT"
HOME="$RENT/home" USERPROFILE="$RENT/home" nohup "$RPC" send payload.tar > send.log 2>&1 < /dev/null &
for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r')
[ -n "$CODE" ] || { echo "FAILED: no transfer code; see $OUT/send.log"; exit 4; }
POD=$(cat "$DAY/POD_ID")
cat <<EOF
transfer code: $CODE   (send log: $OUT/send.log)
NEXT - Runpod MCP exec-pod, id $POD:
  {"command": ["bash", "/work/_bin/start_run.sh", "$RUN", "$CODE"], "timeoutSeconds": 10}
POLL (same tool):   {"command": ["bash", "/work/_bin/status.sh", "$RUN", "30"]}
ALL RUNS:           {"command": ["bash", "/work/_bin/status.sh"]}
FULL LOG (optional): {"command": ["bash", "/work/_bin/fetch.sh", "$RUN"]}  then on the PC:
  cd "$OUT" && HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive <code>
EOF
