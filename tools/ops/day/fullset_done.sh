#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: fullset_done.sh <commit> <files.txt>. The skip check of the merge-
# candidate run (DAYBOX.md "Merge candidate"): prints "FULLSET DONE <run>" and exits 0 when a
# collected run under $RENT/out/ has results/meta.env with COMMIT=<commit> and MARK="fullset", its
# results/files.txt holds every entry of <files.txt> (field 1 of each line; comment and blank
# lines dropped; the chunk size and the A flag ignored, so a list re-sent with other chunk sizes
# still matches), and its results/agg.txt ends with
# RESULT: PASS. Otherwise prints "FULLSET NOT DONE <commit>" and exits 1 (run it). Exit 2: usage,
# a commit that does not resolve, or a list that does not exist or has no entries.
set -uo pipefail
. "$(cd "$(dirname "$0")" && pwd)/../common.sh"
[ $# -eq 2 ] || { echo "usage: fullset_done.sh <commit> <files.txt>"; exit 2; }
C=$(git -C "$REPO" rev-parse --verify -q "$1^{commit}") || { echo "ERROR: $1 is not a commit"; exit 2; }
[ -f "$2" ] || { echo "ERROR: no such list $2"; exit 2; }
norm() { tr -d '\r' < "$1" | grep -vE '^[[:space:]]*(#|$)' | awk '{print $1}'; }   # the entry only: chunk size and A ignored
WANT=$(norm "$2"); [ -n "$WANT" ] || { echo "ERROR: list $2 has no entries"; exit 2; }
for m in "$RENT"/out/*/results/meta.env; do
  [ -f "$m" ] || continue; r=$(dirname "$m")
  tr -d '\r' < "$m" | grep -qx "COMMIT=$C" || continue
  tr -d '\r' < "$m" | grep -qx 'MARK="fullset"' || continue
  [ -f "$r/files.txt" ] || continue
  [ -z "$(printf '%s\n' "$WANT" | grep -vFxf <(norm "$r/files.txt"))" ] || continue
  [ "$(grep '^RESULT:' "$r/agg.txt" 2>/dev/null | tail -1 | tr -d '\r')" = "RESULT: PASS" ] || continue
  echo "FULLSET DONE $(basename "$(dirname "$r")")"; exit 0
done
echo "FULLSET NOT DONE $C"; exit 1
