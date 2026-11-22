#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>
#                      or: pack.sh --delta [--tables <tables.tar.xz>] <old-sha> <new-sha> <run> <ids.txt>
# --delta is the quick lane (DAYBOX.md "Quick lane"): the files that changed between the stream's
# last synced commit <old-sha> and <new-sha> (delta.tar, deletions.txt), at most 5 test ids, run name
# q-<stream>-<label>; bin/quick.sh applies it to the stream's lane repo on the pod and runs the ids.
# tables.tar.xz travels only when the tables key of <new-sha> differs from <old-sha>'s (key read
# as snap.sh's table_key16 does: <archive>.key beside it, the tables_k<key16> name, index.txt for
# the old one); then --tables is required. Without --tables the new key is read from
# $RENT/tables_<short new>.tar.xz. Everything below holds for both modes except where it says delta.
# Bundles the git archive of <commit> + built tables + the test list + the pod scripts of
# tools/ops/day/pod (*.sh, weights.py, weights.txt; from git at HEAD of the checkout that runs
# pack.sh, LF, not from <commit>) into $RENT/out/<run>/payload.tar,
# lists it, refuses forbidden files, starts a detached send and prints the exec-pod body that
# receives the payload into /work/<run>, starts bin/notify.sh, then bin/setup.sh there; a failed
# receive or unpack ends setup.log with RECEIVE_FAILED or UNPACK_FAILED. <run>: [a-z0-9_-], used once.
# Refuses: uncommitted changes under tools/ops/day/pod; pod scripts at HEAD that differ from local
# main's (common.sh pod_main_check: run pack.sh from a checkout at main, the main checkout;
# PACK_OFF_MAIN_POD=1 overrides, for a road test of an ops branch only); a files.txt line outside the list format
# (tools/ops/day/pod/setup.sh); forbidden files; table entries (directories included) with '..'
# or a leading '/' or './'; table entries outside TABLE_ALLOW or in TABLE_DENY; a grep error in
# any check (the checks fail closed). Any failure after $RENT/out/<run> is created removes it.
# Env: MARK (pytest -m for lines without A; default "not slow and not fullset": the per-tip run;
# MARK=fullset for the merge-candidate run, DAYBOX.md "Merge candidate"), runner knobs go in the exec body.
# meta.env also gets GATE_VERSION (2: weights, reserves, gate lock, claims), POD_SCRIPTS (common.sh
# pod_sum: the packed *.sh and weights.py) and POD_WEIGHTS (pod_wsum: weights.txt); setup.sh copies
# them to gate.env for podcheck.sh, which podcheck_body.sh runs before every pack,
# and WATCH_F and WATCH_D (fresh, random beacon codes) for bin/notify.sh and
# tools/ops/day/wait.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"; . "$HERE/pack_lib.sh"
DELTA=0; TBS=""
if [ "${1:-}" = --delta ]; then DELTA=1; shift; ARGS=()
  while [ $# -gt 0 ]; do case "$1" in
    --tables) [ $# -ge 2 ] || { echo "usage: pack.sh --delta [--tables <tables.tar.xz>] <old-sha> <new-sha> <run> <ids.txt>"; exit 2; }; TBS="$2"; shift 2;;
    *) ARGS+=("$1"); shift;; esac; done
  [ ${#ARGS[@]} -eq 4 ] || { echo "usage: pack.sh --delta [--tables <tables.tar.xz>] <old-sha> <new-sha> <run> <ids.txt>"; exit 2; }
  OLDA="${ARGS[0]}"; C="${ARGS[1]}"; RUN="${ARGS[2]}"; LIST="${ARGS[3]}"; TB=""
else
  [ $# -eq 4 ] || { echo "usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>"; exit 2; }
  C="$1"; TB="$2"; RUN="$3"; LIST="$4"
fi
check_run_name "$RUN"
P="$RENT/out/$RUN"; [ -e "$P" ] && { echo "REFUSED: $P exists; run names are used once"; exit 3; }

FORBID='\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml'
BADPATH='\.\.|^/|^\./'
LINE='^tests[^[:space:]]* [1-9][0-9]*( A)?( [1-9][0-9]*G)?$'
COMMENT='^[[:space:]]*(#|$)'
IDLINE='^tests[^[:space:]]*$'   # --delta: one pytest node id or test file per line
pack_check_patterns   # refuse, pick, deny: pack_lib.sh
[ "$DELTA" = 0 ] || pack_check_patterns "$IDLINE"

if [ "$DELTA" = 1 ]; then
  # Cheap argument checks first: they refuse before anything of the pod scripts is read.
  case "$RUN" in q-[a-z0-9_]*-*) ;; *) refuse "a delta run is named q-<stream>-<label>" "$RUN";; esac
  [ -f "$LIST" ] || refuse "no such ids file" "$LIST"
  IDS=$(tr -d '\r' < "$LIST" | pick -vE "$COMMENT") || exit 3
  NIDS=$(printf '%s\n' "$IDS" | sed '/^$/d' | wc -l)
  [ "$NIDS" -ge 1 ] || refuse "ids file has no entries" "$LIST"
  [ "$NIDS" -le 5 ] || refuse "the quick lane runs at most 5 ids, the list has $NIDS: send a normal run" ""
  deny "ids outside the format '<tests path or node id>' (no spaces, one per line)" "$IDS" -vE "$IDLINE"
  OLD=$(git -C "$REPO" rev-parse --verify "$OLDA^{commit}") || refuse "no such commit" "$OLDA"
  NEW=$(git -C "$REPO" rev-parse --verify "$C^{commit}") || refuse "no such commit" "$C"
  OLD_TREE=$(git -C "$REPO" rev-parse "$OLD^{tree}"); NEW_TREE=$(git -C "$REPO" rev-parse "$NEW^{tree}")
  # Tables key: the old one from its archive or index.txt, the new one from --tables or the standard archive.
  OKEY=$(table_key16 compare "$OLD" "$RENT/tables_${OLD:0:8}.tar.xz")
  if [ -n "$TBS" ]; then [ -f "$TBS" ] || refuse "no such tables archive" "$TBS"
    NKEY=$(table_key16 capture "$NEW" "$TBS"); [ -n "$NKEY" ] || refuse "no table key for the archive (no .key file beside it, no tables_k<key16> name, or it differs from that build)" "$TBS"
  else NKEY=$(table_key16 compare "$NEW" "$RENT/tables_${NEW:0:8}.tar.xz")
    [ -n "$NKEY" ] || refuse "no table key for ${NEW:0:8} (no $RENT/tables_${NEW:0:8}.tar.xz or index.txt line); pass --tables <archive>" ""; fi
  SHIP=0; if [ -z "$OKEY" ] || [ "$OKEY" != "$NKEY" ]; then SHIP=1
    [ -n "$TBS" ] || refuse "the tables key changed (old ${OKEY:-unknown}, new $NKEY) and no --tables was given: the delta must carry the new tables" ""; fi
fi

# Pod scripts: committed state of this checkout only (pod_main_check: common.sh).
pack_pod_scripts
[ -f "$LIST" ] || refuse "no such test list" "$LIST"

mkdir -p "$P/bin"
# From here every failure removes the payload folder, so the run name stays usable.
trap 'rc=$?; [ "$rc" = 0 ] || { cd /; rm -rf "$P" || echo "WARNING: could not remove $P"; }' EXIT

DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD)
if [ "$DELTA" = 0 ]; then
  # Test list: "<test file, folder or node id> <ids per chunk> [A] [<GB>G]". The uploaded copy drops comment
  # and blank lines, and that copy is what gets checked.
  L=$(tr -d '\r' < "$LIST" | pick -vE "$COMMENT") || exit 3
  printf '%s\n' "$L" | sed '/^$/d' > "$P/files.txt"
  [ -s "$P/files.txt" ] || refuse "test list has no entries" "$LIST"
  deny "files.txt lines outside the format '<tests path or id> <ids per chunk> [A] [<GB>G]'" "$(cat "$P/files.txt")" -vE "$LINE"
  COMMIT=$(git -C "$REPO" rev-parse --verify "$C^{commit}")
  TREE=$(git -C "$REPO" rev-parse "$COMMIT^{tree}"); TKEY=$(table_key16 capture "$COMMIT" "$TB")   # TKEY: empty without a key
  git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$P/code.tar" "$COMMIT"
  cp "$TB" "$P/tables.tar.xz"
else
  printf '%s\n' "$IDS" | sed '/^$/d' > "$P/ids.txt"   # at most 5, checked above
  COMMIT="$NEW"
  pack_delta_build "$REPO" "$OLD" "$NEW" "$P"
  [ "$SHIP" = 0 ] || cp "$TBS" "$P/tables.tar.xz"
fi
pod_bin "$TOP" "$OPS" "$P/bin" || refuse "could not extract tools/ops/day/pod at $OPS" ""
pack_draw_beacons   # WF, WD
PODSUM=$(pod_sum "$P/bin"); PODW=$(pod_wsum "$P/bin")
if [ "$DELTA" = 0 ]; then
  printf 'COMMIT=%s\nDECOMP=%s\nOPS=%s\nMARK="%s"\nGATE_VERSION=2\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\nTREE=%s\nTABLES_KEY=%s\nWATCH_F=%s\nWATCH_D=%s\n' "$COMMIT" "$DECOMP" "$OPS" "${MARK:-not slow and not fullset}" "$PODSUM" "$PODW" "$TREE" "$TKEY" "$WF" "$WD" > "$P/meta.env"
  CT=$(tar -tf "$P/code.tar")
else
  printf 'MODE=delta\nOLD=%s\nNEW=%s\nOLD_TREE=%s\nNEW_TREE=%s\nOLD_TABLES_KEY=%s\nTABLES_KEY=%s\nCOMMIT=%s\nDECOMP=%s\nOPS=%s\nGATE_VERSION=2\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\nWATCH_F=%s\nWATCH_D=%s\n' "$OLD" "$NEW" "$OLD_TREE" "$NEW_TREE" "$OKEY" "$NKEY" "$COMMIT" "$DECOMP" "$OPS" "$PODSUM" "$PODW" "$WF" "$WD" > "$P/meta.env"
  CT=$(tar -tf "$P/delta.tar")
fi
TALL=""; T=""
if [ -f "$P/tables.tar.xz" ]; then TALL=$(tar -tJf "$P/tables.tar.xz")   # TALL: directories included
  T=$(printf '%s\n' "$TALL" | pick -v '/$') || exit 3; fi                  # T: files only
deny "tables entry with '..', a leading '/' or './'" "$TALL" -E "$BADPATH"
deny "forbidden files" "$(printf '%s\n%s\n' "$CT" "$TALL")" -iE "$FORBID"
deny "calc or docs-sheet extraction in code.tar" "$CT" -E "$TABLE_DENY"
deny "calc or docs-sheet extraction in tables" "$TALL" -E "$TABLE_DENY"
deny "tables entry outside the allowed data/ paths" "$T" -vE "$TABLE_ALLOW"
if [ "$DELTA" = 0 ]; then tar -cf "$P/payload.tar" -C "$P" code.tar tables.tar.xz meta.env files.txt bin
else tar -cf "$P/payload.tar" -C "$P" delta.tar deletions.txt $([ "$SHIP" = 1 ] && echo tables.tar.xz) meta.env ids.txt bin; fi
echo "== payload.tar"; tar -tvf "$P/payload.tar"
if [ "$DELTA" = 0 ]; then echo "== code.tar $(echo "$CT" | wc -l) entries; tables $(echo "$T" | wc -l) files:"
else echo "== delta ${OLD:0:8}..${NEW:0:8}: $DELTA_ADD files changed, $DELTA_DEL deleted; tables $([ "$SHIP" = 1 ] && echo "sent ($(echo "$T" | wc -l) files, key $NKEY)" || echo "kept (key $NKEY)"):"; fi
echo "$T" | sed 's/^/  /'; grep -v '^WATCH_' "$P/meta.env"
pack_send   # CODE
echo "== exec-pod body:"
if [ "$DELTA" = 0 ]; then
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > setup.pid; /work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; tar --no-same-owner -xf payload.tar || { echo UNPACK_FAILED; exit 1; }; setsid nohup bash bin/notify.sh > notify.log 2>&1 < /dev/null & bash bin/setup.sh' > setup.log 2>&1 < /dev/null & sleep 15; tail -3 /work/$RUN/setup.log\"],\"timeoutSeconds\":25}"
else
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > quick.pid; /work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; tar --no-same-owner -xf payload.tar || { echo UNPACK_FAILED; exit 1; }; setsid nohup bash bin/notify.sh > notify.log 2>&1 < /dev/null & bash bin/quick.sh' > quick.log 2>&1 < /dev/null & sleep 10; tail -3 /work/$RUN/quick.log\"],\"timeoutSeconds\":25}"
fi
echo "== then, as one background Bash call: bash tools/ops/day/wait.sh $RUN [--until done|first-failure] [--timeout <hours>]"
