#!/usr/bin/env bash
# PC SIDE (Git Bash). Builds the vanilla and Kaizo tables:
#   build_tables.sh --here     in place, in the checkout or worktree you are in (git toplevel of $PWD)
#   build_tables.sh <commit>   in a temporary worktree $RENT/wt_<short>, then packs the uploadable
#                              tables (common.sh TABLE_ALLOW) into $RENT/tables/tables_k<key16>.tar.xz
#                              and removes the worktree (also on failure)
# Reuse: <key> = tools/ops/table_key.py (this script, common.sh, the worktree's .gitignore and git
# excludes, every module the table tools import, the tracked data, the interpreter and its packages,
# node, the decomp, the symbols map, the local inputs by content). When
# $RENT/tables/tables_k<key16>.tar.xz exists the build is skipped; a new one is made read-only.
# Either way $RENT/tables_<short>.tar.xz is a copy of it, $RENT/tables_<short>.key holds <key>,
# and $RENT/tables/index.txt gets the line "<full sha> <key>".
# Inputs from the main checkout ($REPO): data/local.toml (paths to the ROM, patch, calc zip,
# docs sheet; copied into the build checkout when it has none), .cache/decomp/<commit>/main.nef.xMAP
# (symbols) and external/pokeplatinum (linked when the build checkout has none or an empty folder there;
# any other folder without src/ stops the build). These stay on the PC.
# Python: $NUZLINE_PY (an existing venv's python, run with PYTHONPATH) or else `uv run --group rom python`.
# RAM: the build steps run as the child of `tools/ops/ramgate.py --build` (docs/process.md section 5):
# they start once free RAM > 6 GB + the build weight and the machine lock is free, and hold the lock
# until they end. A reuse needs neither. A wait that times out exits 14.
set -euo pipefail
OPS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$OPS_DIR/common.sh"
LINKED=""
unlink_decomp() {  # removes only a link this script made; never deletes through it
  [ -n "$LINKED" ] && [ -e "$LINKED" ] || return 0
  if command -v cygpath >/dev/null; then powershell.exe -NoProfile -Command "[System.IO.Directory]::Delete($(psq "$LINKED"))"
  else rm "$LINKED"; fi
  [ ! -e "$LINKED" ] || { echo "ERROR: could not remove the decomp link $LINKED; remove it by hand before deleting the worktree"; return 1; }
  LINKED=""
}
link_decomp() {  # $1 = checkout without external/pokeplatinum
  local l="$1/external/pokeplatinum"
  [ -d "$REPO/external/pokeplatinum/src" ] || { echo "ERROR: $REPO/external/pokeplatinum/src missing; not linking"; exit 2; }
  mkdir -p "$1/external"
  LINKED="$l"   # set first: whatever the link step leaves behind is removed on exit
  if command -v cygpath >/dev/null; then
    powershell.exe -NoProfile -Command "New-Item -ItemType Junction -Path $(psq "$l") -Target $(psq "$REPO/external/pokeplatinum") | Out-Null"
  else ln -s "$REPO/external/pokeplatinum" "$l"; fi
  [ -d "$l/src" ] || { echo "ERROR: decomp link $l failed"; exit 2; }
}
decomp_state() {  # $1 = path; prints none, link (a symlink or junction), decomp (a folder with src/),
  # empty (an empty folder, no link) or other
  if [ -L "$1" ]; then echo link
  elif [ ! -e "$1" ]; then echo none
  elif [ -d "$1/src" ]; then echo decomp
  elif [ -d "$1" ] && [ -z "$(ls -A "$1")" ]; then echo empty
  else echo other; fi
}
ensure_decomp() {  # $1 = build checkout: keeps its own decomp, else links $REPO's
  local l="$1/external/pokeplatinum"
  case "$(decomp_state "$l")" in
    none) link_decomp "$1" ;;
    empty) rmdir "$l"; link_decomp "$1" ;;   # an empty folder is no decomp; rmdir never removes content
    decomp) ;;
    link) [ -d "$l/src" ] || { echo "ERROR: $l is a link with no src/ (not made by this run); remove it by hand"; exit 2; } ;;
    *) echo "ERROR: $l is not empty and has no src/: not a decomp; move it away by hand"; exit 2 ;;
  esac
}
# Sourced (tests/tools/test_build_tables_decomp.py): the functions only.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0

ARG="${1:?usage: build_tables.sh --here | <commit>}"
[ -f "$REPO/data/local.toml" ] || { echo "ERROR: $REPO/data/local.toml missing (see data/local.example.toml)"; exit 2; }
[ -d "$REPO/external/pokeplatinum" ] || { echo "ERROR: $REPO/external/pokeplatinum missing"; exit 2; }

if [ "$ARG" = --here ]; then
  B=$(git rev-parse --show-toplevel); MODE=here
  trap unlink_decomp EXIT
else
  C=$(git -C "$REPO" rev-parse --short "$ARG^{commit}"); MODE=commit
  B="$RENT/wt_$C"; JC="$RENT/wt-cache_$C"; mkdir -p "$RENT"
  cleanup() {  # unlink first; delete the worktree only once no link is left in it
    [ -n "${KA:-}" ] && rm -f "$KA.tmp.$$"   # a failed or interrupted tar leaves no partial archive
    cd "$RENT"; unlink_decomp || return 1
    [ -e "$B/external/pokeplatinum" ] && [ -z "$LINKED" ] && { echo "ERROR: $B/external/pokeplatinum still present; not deleting $B"; return 1; }
    if [ -e "$B" ]; then git -C "$REPO" worktree remove --force "$B" 2>/dev/null || rm -rf "$B"; fi
    git -C "$REPO" worktree prune; rm -rf "$JC"
  }
  if [ -e "$B" ]; then  # leftover of an earlier run: its link may still be in it
    case "$(decomp_state "$B/external/pokeplatinum")" in
      link) LINKED="$B/external/pokeplatinum" ;;
      empty) rmdir "$B/external/pokeplatinum" ;;
    esac
    cleanup
  fi
  trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM HUP   # a signal runs cleanup too
  git -C "$REPO" worktree add --detach "$B" "$C"
  export NUZLINE_CACHE_DIR="$JC"
fi
cd "$B"
ensure_decomp "$B"
[ -f data/local.toml ] || cp "$REPO/data/local.toml" data/local.toml
if [ -n "${NUZLINE_PY:-}" ]; then
  W="$(pwd -W 2>/dev/null || pwd)"; S=":"; command -v cygpath >/dev/null && S=";"
  export PYTHONPATH="$W$S$W/src"; PY=("$NUZLINE_PY")
else uv sync --group rom -q; PY=(uv run --group rom python); fi
DC=$("${PY[@]}" -c "from tools.rom.symbols import DECOMP_COMMIT; print(DECOMP_COMMIT)" | tr -d '\r')
SYM="$B/.cache/decomp/$DC/main.nef.xMAP"; [ -f "$SYM" ] || SYM="$REPO/.cache/decomp/$DC/main.nef.xMAP"
[ -f "$SYM" ] || { echo "ERROR: symbols map missing: $SYM (docs/rom_pipeline.md, Decomp build)"; exit 2; }

wp() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else echo "$1"; fi; }
if [ "$MODE" = commit ]; then
  KEY=$("${PY[@]}" "$(wp "$OPS_DIR/table_key.py")" --root "$(wp "$B")" --ops "$(wp "$OPS_DIR")" --decomp "$(wp "$B/external/pokeplatinum")" \
    --symbols "$(wp "$SYM")" --config "$(wp "$B/data/local.toml")" | tr -d '\r')
  echo "$KEY" | grep -qE '^[0-9a-f]{64}$' || { echo "ERROR: table_key.py printed no key: $KEY"; exit 2; }
  FULL=$(git -C "$REPO" rev-parse "$C^{commit}"); TD="$RENT/tables"; KA="$TD/tables_k${KEY:0:16}.tar.xz"; mkdir -p "$TD"
  record() {  # tables_<short>.tar.xz is the key's archive; the key and the sha -> key line are kept
    rm -f "$RENT/tables_$C.tar.xz"; cp "$KA" "$RENT/tables_$C.tar.xz"; chmod u+w "$RENT/tables_$C.tar.xz"
    echo "$KEY" > "$RENT/tables_$C.key"; echo "$FULL $KEY" >> "$TD/index.txt"; }
  if [ -f "$KA" ]; then record; echo "BUILD_OK $RENT/tables_$C.tar.xz (reused $KA, key ${KEY:0:16})"; exit 0; fi
fi

# The steps, as one bash child of the RAM gate: $1 = symbols map, the rest = the python command.
"${PY[@]}" "$(wp "$OPS_DIR/ramgate.py")" --build -- "$(wp "$BASH")" -c 'set -euo pipefail; SYM=$1; shift
"$@" -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate
"$@" -m tools.build_tables
"$@" -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom --symbols "$SYM"
"$@" -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
"$@" -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
"$@" -m tools.kaizo.overrides check
"$@" -m tools.build_tables --game kaizo' build-steps "$SYM" "${PY[@]}"
[ "$MODE" = here ] && { echo "BUILD_OK in place: $B"; exit 0; }

L="$RENT/tables_$C.list"
{ git ls-files -o -i --exclude-standard -- data; git diff --name-only -- data/gen4_platinum data/gen4_platinum_kaizo; } | sort -u \
  | grep -E "$TABLE_ALLOW" | grep -vE "$TABLE_DENY" | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$L"
grep -q '^data/gen4_platinum_kaizo/manifest.json$' "$L"
tar -cJf "$KA.tmp.$$" -T "$L"; mv -f "$KA.tmp.$$" "$KA"; chmod a-w "$KA"; record
cd "$RENT"
echo "BUILD_OK $RENT/tables_$C.tar.xz ($(wc -l < "$L") files, key ${KEY:0:16})"
