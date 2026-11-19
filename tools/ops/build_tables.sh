#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: build_tables.sh <commit>
# Detached worktree $NUZLINE_SCRATCH/rent/wt_<short> at <commit>, decomp link to the main
# checkout's external/pokeplatinum, vanilla + Kaizo build, then packs the built data/**.json|npz
# into $NUZLINE_SCRATCH/rent/tables_<short>.tar.xz and removes the worktree.
# The ROM, PKCalc zip and docs sheet are read locally only; only built tables are packed.
# Python: $NUZLINE_PY (an existing venv's python, run with PYTHONPATH) or else `uv run --group rom python`.
set -euo pipefail
REPO="${NUZLINE_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
RENT="${NUZLINE_SCRATCH:-$REPO/_scratch}/rent"; mkdir -p "$RENT"
C=$(git -C "$REPO" rev-parse --short "${1:?usage: build_tables.sh <commit>}^{commit}")
WT="$RENT/wt_$C"; LINK="$WT/external/pokeplatinum"
unlink_decomp() { [ -e "$LINK" ] || return 0
  if command -v cygpath >/dev/null; then cmd //c rmdir "$(cygpath -w "$LINK")"; else rm "$LINK"; fi; }
trap unlink_decomp EXIT
git -C "$REPO" worktree add --detach "$WT" "$C"
mkdir -p "$WT/external"
if command -v cygpath >/dev/null; then cmd //c mklink /J "$(cygpath -w "$LINK")" "$(cygpath -w "$REPO/external/pokeplatinum")"
else ln -s "$REPO/external/pokeplatinum" "$LINK"; fi
cd "$WT"; export NUZLINE_CACHE_DIR="$RENT/wt-cache_$C"
if [ -n "${NUZLINE_PY:-}" ]; then
  W="$(pwd -W 2>/dev/null || pwd)"; S=":"; command -v cygpath >/dev/null && S=";"
  export PYTHONPATH="$W$S$W/src"; PY=("$NUZLINE_PY")
else uv sync --group rom -q; PY=(uv run --group rom python); fi
"${PY[@]}" -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate
"${PY[@]}" -m tools.build_tables
"${PY[@]}" -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom
"${PY[@]}" -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
"${PY[@]}" -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
"${PY[@]}" -m tools.kaizo.overrides check
"${PY[@]}" -m tools.build_tables --game kaizo
{ git ls-files -o -i --exclude-standard -- data; git diff --name-only -- 'data/gen4_platinum*/'; } \
  | grep -E '\.(json|npz)$' | grep -viE '(^|/)backups?/' | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$RENT/tables_$C.list"
grep -q '^data/gen4_platinum_kaizo/manifest.json$' "$RENT/tables_$C.list"
tar -cJf "$RENT/tables_$C.tar.xz" -T "$RENT/tables_$C.list"
cd "$RENT"; unlink_decomp
git -C "$REPO" worktree remove --force "$WT"; rm -rf "$RENT/wt-cache_$C"
echo "BUILD_OK $RENT/tables_$C.tar.xz ($(wc -l < "$RENT/tables_$C.list") files)"
