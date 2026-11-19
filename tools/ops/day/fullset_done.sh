#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: fullset_done.sh <commit>. The skip check of the merge-candidate run
# (DAYBOX.md "Merge candidate"): prints "FULLSET DONE <run>" and exits 0 when a collected run under
# $RENT/out/ has results/meta.env with COMMIT=<commit> and MARK="fullset" and its results/agg.txt
# ends with RESULT: PASS; otherwise prints "FULLSET NOT DONE <commit>" and exits 1 (run it).
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/../common.sh"
[ $# -eq 1 ] || { echo "usage: fullset_done.sh <commit>"; exit 2; }
C=$(git -C "$REPO" rev-parse --verify "$1^{commit}")
for m in "$RENT"/out/*/results/meta.env; do
  [ -f "$m" ] || continue; r=$(dirname "$m")
  tr -d '\r' < "$m" | grep -qx "COMMIT=$C" || continue
  tr -d '\r' < "$m" | grep -qx 'MARK="fullset"' || continue
  [ "$(grep '^RESULT:' "$r/agg.txt" 2>/dev/null | tail -1 | tr -d '\r')" = "RESULT: PASS" ] || continue
  echo "FULLSET DONE $(basename "$(dirname "$r")")"; exit 0
done
echo "FULLSET NOT DONE $C"; exit 1
