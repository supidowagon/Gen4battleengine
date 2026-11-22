#!/usr/bin/env bash
# PC SIDE (Git Bash). Usage: pack.sh <commit> <tables.tar.xz> <run> <files.txt>
#                      or: pack.sh --delta [--tables <tables.tar.xz>] <old-sha> <new-sha> <run> <ids.txt>
# --delta is the quick lane (DAYBOX.md "Quick lane"): the files that changed between the stream's
# last synced commit <old-sha> and <new-sha> (delta.tar, deletions.txt), at most 5 test ids, run name
# q-<stream>-<label>; bin/quick.sh applies it to the stream's lane repo on the pod and runs the ids.
# tables.tar.xz travels only when the tables content hash of <new-sha> differs from <old-sha>'s
# (CONTENT_HASH of the build's meta file, tools/ops/content_hash.py; table_key16.sh table_chash, or
# an earlier send's kept meta.env TABLES_CHASH); when either content hash is unknown it travels only
# when the tables key differs (key read as snap.sh's table_key16 does: <archive>.key beside it, the
# tables_k<key16> name, index.txt for the old one). A tables send needs --tables. Without --tables
# the new key and hash are read from $RENT/tables_<short new>.tar.xz and its meta. A key no archive
# or index.txt line gives comes from an earlier send of that commit (pack_lib.sh pack_sent_key: its
# kept meta.env), so the old commit's key is known whenever the marker's commit was sent from this
# PC. The key stays the cache identity (meta.env TABLES_KEY, the lane marker); it is still needed
# for the new commit. Everything below holds for both modes except where it says delta.
# Bundles the git archive of <commit> + built tables + the test list + the pod scripts of
# tools/ops/day/pod (*.sh, weights.py, weights.txt; from git at HEAD of the checkout that runs
# pack.sh, LF, not from <commit>) into $RENT/out/<run>/payload.tar,
# lists it, refuses forbidden files, starts a detached send and prints the exec-pod body that
# receives the payload into /work/<run>, starts bin/notify.sh, then bin/setup.sh there; a failed
# receive or unpack ends setup.log (--delta: receive.log, the receive's own log; quick.log starts at
# QUICK_START) with RECEIVE_FAILED or UNPACK_FAILED. <run>: [a-z0-9_-], used once.
# Refuses: uncommitted changes under tools/ops/day/pod; pod scripts at HEAD that differ from local
# main's (common.sh pod_main_check: run pack.sh from a checkout at main, the main checkout;
# PACK_OFF_MAIN_POD=1 overrides, for a road test of an ops branch only); a files.txt line outside the list format
# (tools/ops/day/pod/setup.sh); forbidden files; table entries (directories included) with '..'
# or a leading '/' or './'; table entries outside TABLE_ALLOW or in TABLE_DENY; a grep error in
# any check (the checks fail closed). Any failure after $RENT/out/<run> is created removes it.
# Env: MARK (pytest -m for lines without A; default "not slow and not fullset": the per-tip run;
# MARK=fullset for the merge-candidate run, DAYBOX.md "Merge candidate"); GATE_GB, KILL_GB, MAXP
# (the pod shape's values), SPACING and the sweep knobs SWEEP_P, SWEEP_W, SWEEP_CAP, SWEEP_NICE,
# SWEEP_JOB_S, SWEEP_SPACING, SWEEP_POLL, SWEEP_REPLAY_S (pack_lib.sh PACK_ENV_NAMES): each one set
# goes into the printed body's env (pack_body_env), in both modes; unset, a normal run's runner
# uses its defaults and quick.sh finds them on the pod. Other runner knobs go in the exec body by hand.
# The payload's bin/ also holds sweep/ (tools/ops/sweep/*.py from git at HEAD of the checkout that
# runs pack.sh, pack_lib.sh pack_sweep_tools): sweep.sh uses it for a repo without tools/ops/sweep.
# meta.env also gets NIGHT (yyyymmdd: SWEEP_NIGHT, else the packing date; the sweep's collect.py takes
# the tip from COMMIT and the night from it when the run name has no date), TABLES_CHASH (and a delta's OLD_TABLES_CHASH) and GATE_VERSION (2: weights, reserves, gate lock, claims), POD_SCRIPTS (common.sh
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
BENV=$(pack_body_env) || { echo "$BENV"; exit 3; }   # GATE_GB, KILL_GB, MAXP set on the PC, for the body's env
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
  OKEY=$(table_key16 compare "$OLD" "$RENT/tables_${OLD:0:8}.tar.xz"); OSRC="archive or index.txt"
  [ -n "$OKEY" ] || { OKEY=$(pack_sent_key "$OLD"); OSRC="an earlier send's meta.env"; }; [ -n "$OKEY" ] || OSRC="unknown"
  if [ -n "$TBS" ]; then [ -f "$TBS" ] || refuse "no such tables archive" "$TBS"
    NSRC="--tables"; NKEY=$(table_key16 capture "$NEW" "$TBS"); [ -n "$NKEY" ] || refuse "no table key for the archive (no .key file beside it, no tables_k<key16> name, or it differs from that build)" "$TBS"
  else NKEY=$(table_key16 compare "$NEW" "$RENT/tables_${NEW:0:8}.tar.xz"); NSRC="archive or index.txt"
    [ -n "$NKEY" ] || { NKEY=$(pack_sent_key "$NEW"); NSRC="an earlier send's meta.env"; }
    [ -n "$NKEY" ] || refuse "no table key for ${NEW:0:8} (no $RENT/tables_${NEW:0:8}.tar.xz, index.txt line or earlier send); pass --tables <archive>" ""; fi
  # Content hash: the identity of "the same tables" (build_tables.sh meta, tools/ops/content_hash.py).
  OCH=$(table_chash compare "$OLD" "$RENT/tables_${OLD:0:8}.tar.xz"); [ -n "$OCH" ] || OCH=$(pack_sent_chash "$OLD")
  if [ -n "$TBS" ]; then NCH=$(table_chash capture "$NEW" "$TBS")
  else NCH=$(table_chash compare "$NEW" "$RENT/tables_${NEW:0:8}.tar.xz"); [ -n "$NCH" ] || NCH=$(pack_sent_chash "$NEW"); fi
  SHIP=0; if [ -n "$OCH" ] && [ -n "$NCH" ]; then   # both known: the content decides, the key does not
    if [ "$OCH" != "$NCH" ]; then SHIP=1
      [ -n "$TBS" ] || refuse "the tables content changed (old ${OCH:0:16}, new ${NCH:0:16}) and no --tables was given: the delta must carry the new tables" ""; fi
  elif [ -z "$OKEY" ] || [ "$OKEY" != "$NKEY" ]; then SHIP=1
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
  TCH=$(table_chash capture "$COMMIT" "$TB")   # TCH: the content hash, empty without a meta file
  pack_code_tar "$REPO" "$COMMIT" "$P/code.tar"
  cp "$TB" "$P/tables.tar.xz"
else
  printf '%s\n' "$IDS" | sed '/^$/d' > "$P/ids.txt"   # at most 5, checked above
  COMMIT="$NEW"
  pack_delta_build "$REPO" "$OLD" "$NEW" "$P"
  [ "$SHIP" = 0 ] || cp "$TBS" "$P/tables.tar.xz"
fi
pod_bin "$TOP" "$OPS" "$P/bin" || refuse "could not extract tools/ops/day/pod at $OPS" ""
pack_sweep_tools "$TOP" "$OPS" "$P/bin" || refuse "could not extract tools/ops/sweep (extract.py, replay.py) at $OPS" ""
pack_draw_beacons   # WF, WD
NIGHT="${SWEEP_NIGHT:-$(date +%Y%m%d)}"   # meta.env NIGHT: tools/ops/sweep/collect.py falls back to it when the run name has no date
echo "$NIGHT" | grep -qE '^[0-9]{8}$' || refuse "SWEEP_NIGHT is not yyyymmdd" "$NIGHT"
PODSUM=$(pod_sum "$P/bin"); PODW=$(pod_wsum "$P/bin")
if [ "$DELTA" = 0 ]; then
  printf 'COMMIT=%s\nDECOMP=%s\nOPS=%s\nMARK="%s"\nGATE_VERSION=2\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\nTREE=%s\nTABLES_KEY=%s\nTABLES_CHASH=%s\nNIGHT=%s\nWATCH_F=%s\nWATCH_D=%s\n' "$COMMIT" "$DECOMP" "$OPS" "${MARK:-not slow and not fullset}" "$PODSUM" "$PODW" "$TREE" "$TKEY" "$TCH" "$NIGHT" "$WF" "$WD" > "$P/meta.env"
  CT=$(tar -tf "$P/code.tar")
else
  printf 'MODE=delta\nOLD=%s\nNEW=%s\nOLD_TREE=%s\nNEW_TREE=%s\nOLD_TABLES_KEY=%s\nTABLES_KEY=%s\nOLD_TABLES_CHASH=%s\nTABLES_CHASH=%s\nCOMMIT=%s\nDECOMP=%s\nOPS=%s\nGATE_VERSION=2\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\nNIGHT=%s\nWATCH_F=%s\nWATCH_D=%s\n' "$OLD" "$NEW" "$OLD_TREE" "$NEW_TREE" "$OKEY" "$NKEY" "$OCH" "$NCH" "$COMMIT" "$DECOMP" "$OPS" "$PODSUM" "$PODW" "$NIGHT" "$WF" "$WD" > "$P/meta.env"
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
else echo "== delta ${OLD:0:8}..${NEW:0:8}: $DELTA_ADD files changed, $DELTA_DEL deleted; tables $([ "$SHIP" = 1 ] && echo "sent ($(echo "$T" | wc -l) files, key $NKEY, content ${NCH:-unknown})" || echo "kept (key $NKEY; old key from $OSRC, new from $NSRC; content ${NCH:-unknown, the key decided})"):"; fi
echo "$T" | sed 's/^/  /'; grep -v '^WATCH_' "$P/meta.env"
pack_send   # CODE
echo "== exec-pod body:"
if [ "$DELTA" = 0 ]; then
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > setup.pid; /work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; tar --no-same-owner -xf payload.tar || { echo UNPACK_FAILED; exit 1; }; setsid nohup bash bin/notify.sh > notify.log 2>&1 < /dev/null & bash bin/setup.sh' > setup.log 2>&1 < /dev/null & sleep 15; tail -3 /work/$RUN/setup.log\"]$BENV,\"timeoutSeconds\":25}"
else
echo "{\"command\":[\"sh\",\"-c\",\"mkdir -p /work/$RUN && cd /work/$RUN && nohup sh -c 'echo \$\$ > quick.pid; /work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; tar --no-same-owner -xf payload.tar || { echo UNPACK_FAILED; exit 1; }; setsid nohup bash bin/notify.sh > notify.log 2>&1 < /dev/null & bash bin/quick.sh' > receive.log 2>&1 < /dev/null & sleep 10; tail -3 /work/$RUN/quick.log 2>/dev/null || tail -3 /work/$RUN/receive.log\"]$BENV,\"timeoutSeconds\":25}"
fi
echo "== then, as one background Bash call: bash tools/ops/day/wait.sh $RUN [--until done|first-failure] [--timeout <hours>]"
