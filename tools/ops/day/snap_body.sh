# Sourced by tools/ops/day/snap.sh (PC side): the exec-pod body that carries run.sh. exec-pod
# refuses an argv item over 4096 characters and an argv over 256 items, so run.sh travels
# base64-encoded in items of at most BODY_LIM (4000) characters after the script, as common.sh
# exec_body does: ["sh","-c",<script>,"_",<chunk>,...]. (Not in common.sh: common.sh feeds the
# pod scripts' sums.)
BODY_LIM=4000; BODY_MAX_ITEMS=256
# body_chunks <file>: CH = <file>'s base64 in items of at most BODY_LIM characters, RSUM = its
# sha256. Prints REFUSED and returns 3 when the body (sh, -c, script, _, chunks) would have more
# than BODY_MAX_ITEMS argv items.
body_chunks() { local rb n
  rb=$(base64 -w0 "$1") || { echo "REFUSED: cannot read $1"; return 3; }
  RSUM=$(sha256sum "$1" | cut -c1-64); CH=()
  while [ -n "$rb" ]; do CH+=("${rb:0:$BODY_LIM}"); rb=${rb:$BODY_LIM}; done
  n=$(( ${#CH[@]} + 4 ))
  [ "$n" -le "$BODY_MAX_ITEMS" ] || { echo "REFUSED: exec body would have $n argv items, over $BODY_MAX_ITEMS (run.sh too large)"; return 3; }; }
# body_json <script>: the body (one line), the script JSON-escaped, then CH. Prints REFUSED and
# returns 3 when the script is over BODY_LIM characters.
body_json() { local s="$1" items="" c
  [ "${#s}" -le "$BODY_LIM" ] || { echo "REFUSED: exec body script is ${#s} characters, over $BODY_LIM"; return 3; }
  s=${s//\\/\\\\}; s=${s//\"/\\\"}
  for c in "${CH[@]}"; do items="$items, \"$c\""; done
  printf '{"command": ["sh", "-c", "%s", "_"%s], "timeoutSeconds": 25}\n' "$s" "$items"; }
