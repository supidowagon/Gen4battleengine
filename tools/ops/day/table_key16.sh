#!/usr/bin/env bash
# PC SIDE, sourced by snap.sh and pack_lib.sh (needs $RENT from common.sh). Function only.
# table_key16 capture|compare <full sha> <tables archive>: the first 16 hex of a build's table
# key (tools/ops/build_tables.sh, tools/ops/table_key.py), or nothing.
#   capture (the key a capture is stored under): only from the archive itself, the
#     <archive minus .tar.xz>.key file beside it, else its tables_k<key16>.tar.xz name; and when
#     $RENT/tables/tables_k<key16>.tar.xz exists, the archive must be byte for byte that build
#     (cmp), else nothing (a rebuilt or edited archive beside a stale .key). Never from
#     index.txt: a capture made with other tables would be stored under the official key and
#     fetched by every later compare.
#   compare (the key of the main capture to fetch): as capture, else the last "<sha> <key>" line
#     of $RENT/tables/index.txt for <full sha>. The fallback only names which stored capture to
#     fetch; a stored capture got its key from its own archive, so it fetches only a capture made
#     with that key's tables.
# (Kept out of common.sh: common.sh is an input of the table key.)
table_key16() { local k="" a="${3:-}" ka
  if [ -n "$a" ]; then
    [ -f "${a%.tar.xz}.key" ] && k=$(tr -dc 0-9a-f < "${a%.tar.xz}.key")
    [ -n "$k" ] || k=$(basename "$a" | sed -n 's/^tables_k\([0-9a-f]\{16\}\)\.tar\.xz$/\1/p')
    k=${k:0:16}
    if [[ "$k" =~ ^[0-9a-f]{16}$ ]]; then ka="$RENT/tables/tables_k$k.tar.xz"
      if [ -f "$a" ] && [ -f "$ka" ] && ! cmp -s "$a" "$ka"; then
        echo "NOTE: $a differs from $ka, the build of key $k its .key names: no key from it" >&2; k=""; fi
    else k=""; fi
  fi
  [ -n "$k" ] || [ "$1" != compare ] || k=$(awk -v s="$2" '$1 == s {k = $2} END {print k}' "$RENT/tables/index.txt" 2>/dev/null | tr -dc 0-9a-f)
  k=${k:0:16}; if [[ "$k" =~ ^[0-9a-f]{16}$ ]]; then echo "$k"; fi; return 0; }
