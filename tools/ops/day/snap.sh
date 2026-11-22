#!/usr/bin/env bash
# PC SIDE (Git Bash). Snapshot capture and compare on the test pod (tools/snap_compare.py;
# rules: docs/process.md section 5; steps: tools/ops/day/DAYBOX.md "Snapshot compare").
# Usage:
#   snap.sh capture <commit> <tables.tar.xz> [<name>] [--tool <commit>] [--dry-run]
#   snap.sh compare <commit> <tables.tar.xz> <capture dir on the pod> <name> [--allow-new a,b] [--dry-run]
#   snap.sh fetch <name> <code>
# capture: pod folder /work/<name> (default snapmain-<7-char short>); payload = git archive of
#   <commit> + the tables; run.sh captures seeds 0-199 into /work/<name>/snap with the checkout's
#   own tools/snap_compare.py, or with --tool <commit>'s copy (re-capture for a candidate whose
#   snap_compare writes another format; name it snap2-<short tip>).
# compare: pod folder /work/<name>; payload = git archive of the candidate + its tables; run.sh
#   compares <capture dir> (e.g. /work/snapmain-08a5700/snap) with the candidate's own
#   tools/snap_compare.py; --allow-new is left out when empty.
# Both write $RENT/out/<name>/{code.tar,tables.tar.xz,payload.tar,run.sh,body.json,send.log},
#   refuse forbidden files (FORBID and TABLE_DENY in the code archive; table entries with '..', a
#   leading '/' or './', outside TABLE_ALLOW or in TABLE_DENY), start a detached send whose
#   send.log ends with SEND_RC=<n>, and print the exec-pod body (run.sh in argv chunks of at most
#   4000 characters, checked by sha256 on the pod before it runs: see "The body" below; a body
#   over exec-pod's limits, 4096 characters an item and 256 items, is refused). --dry-run runs every check and
#   writes run.sh and body.json (receive code "<code>") but sends nothing.
#   run.sh first writes /work/<name>/reads (the capture dir a compare reads; empty for a
#   capture), then /work/<name>/times.txt: START, SYNC_RC=<n> (a failed uv sync stops here,
#   without ALL_DONE; it runs under timeout -k 30 1800: rc 124 or 137 once it hangs),
#   SNAP_LOCK <epoch>, CAPTURE_START or COMPARE_START, CAPTURE_RC=<n> <epoch> or
#   COMPARE_RC=<n> <epoch>, ALL_DONE. Before the job run.sh checks /proc/sys/vm/max_map_count: under
#   1048576 (a pod with 65530 aborts JAX processes, rc 134) or unreadable ends times.txt with
#   MAP_COUNT_LOW <value|none> <epoch>, without ALL_DONE.
#   Pod lock: one snapshot job at a time per pod. After uv sync run.sh takes /work/.snaplock
#   (snaplock below; SNAP_LOCK in times.txt); while another job holds it (snaplive), run.out
#   gets "lock wait <time>: <holder folder> holds /work/.snaplock" once a minute; a stale lock is
#   removed (SNAP_LOCK_STALE <pid|none> <epoch> in times.txt). run.sh releases it on every exit.
#   After the job ends run.sh stops what is left of it (snapstop; JOB_LEFT <pids> <epoch> in
#   times.txt when anything was) before COMPARE_RC/CAPTURE_RC is followed by ALL_DONE. A TERM
#   writes TERM <epoch> to times.txt, stops the job the same way and exits 143, without ALL_DONE.
#   A job process still live (not a zombie) after the KILL (SNAPSTOP_MAX_S, default 40 s; a
#   process stuck in the kernel, e.g. on the network volume) ends times.txt with JOB_STUCK <pids>
#   <epoch>: run.sh exits 1 (143 after a TERM) without the store and ALL_DONE and keeps the pod
#   lock, which the next job judges stale once those processes are gone (snaplive).
#   Env: SNAP_GATE_GB (default 88): run.sh starts the capture or compare only while container
#   memory without reclaimable page cache (cgused) is under it, or once no pack unit has been
#   live on the pod for IDLE_WAIT_S s (default 300; GATE_IDLE_START <epoch> in times.txt);
#   pass the shape's SNAP_GATE_GB (176 on a 256 GB pod). With no cgroup memory file times.txt
#   ends GATE_NO_CGROUP_MEMORY <epoch>, without ALL_DONE. The same SNAP_GATE_GB is snap_compare's
#   --gate-gb: a part starts only while container memory + the live parts' reserve + its own
#   expected peak stays at or under it. SNAP_WORKERS (default 16): snap_compare's --workers, the
#   parts run at once (1: one after another). Both are passed only when the snap_compare that
#   runs (the commit's, or --tool's) has --workers; an older one runs its parts one by one.
#   Capture store on the network volume (DAYBOX.md "Network volume"; pod/cache_sync.sh, which
#   travels in the payload as committed at HEAD of this checkout): a capture without --tool is
#   keyed "<full sha>-<key16>" (key16: table_key16 below, from the tables archive itself, never
#   from index.txt). On a pod whose /vol is a mountpoint, run.sh first fetches that key from the
#   volume (store.log; times.txt then holds CAPTURE_FROM_VOLUME <key> <epoch>, CAPTURE_RC=0,
#   ALL_DONE, and no capture runs); otherwise it captures and, after CAPTURE_RC=0, stores the
#   capture (STORE_RC=<n> <epoch>; a stored key is never overwritten). A compare of
#   /work/snapmain-<short>/snap whose capture.json is not on the pod fetches it from the volume
#   first (FETCH_RC=<n> <epoch>), keyed by <short>'s full sha and its key16 from $RENT; when
#   capture.json is still missing after it, times.txt ends NO_CAPTURE <dir> <epoch> and run.sh
#   exits 1 (no compare, no ALL_DONE). Without a key nothing is stored or fetched (NOTE line); on
#   a pod without /vol run.sh does what it did before the volume. Every volume call runs under
#   timeout -k 30 <s>: CACHE_FETCH_TIMEOUT for a fetch, CACHE_STORE_TIMEOUT for a store (PC env,
#   default 900 each, whole seconds, else refused): one that times out (rc 124 or 137) writes VOLUME_TIMEOUT fetch|store
#   <epoch> to times.txt and a WARNING to run.out; a capture's timed-out fetch removes what it
#   may have left (snap/, .snapfetch.*) and captures as without the volume; a compare's leaves
#   only a complete, checked snap/ (renamed whole) or none, so NO_CAPTURE; a timed-out store
#   leaves only a temp name on the volume (STORE_RC=124|137).
#   Order in run.sh: uv sync, pod lock (SNAP_LOCK), the volume fetch, memory gate, job, snapstop,
#   the volume store, ALL_DONE, lock released on exit. The fetch runs under the pod lock, so no
#   other job on this pod writes the same capture folder meanwhile (another pod's store of the
#   same key is a temp copy renamed whole, so a fetch sees the key whole or not at all); it is
#   a file copy, so it does not wait at the memory gate, and a capture found on the volume
#   ends the job before the gate. The store runs after CAPTURE_RC=0 and after snapstop (no job
#   process still writes snap/), still under the lock.
# fetch: receives what the pod sends (DAYBOX.md "Snapshot compare", Fetch) into
#   $RENT/out/<name>/fetch-<epoch>/ and unpacks every .tgz received.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../common.sh"
. "$HERE/pod/gate.sh"   # podgate and reserve_knobs, copied into run.sh
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
# (Kept here, not in common.sh: common.sh is an input of the table key.)
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
usage() { sed -n '4,7p' "$0" | sed 's/^# *//'; exit 2; }
# The functions below are copied into run.sh (declare -f); run.sh runs on the pod, not here.
# cgused: container memory in bytes without reclaimable page cache: cgroup v2 memory.current
# minus inactive_file of memory.stat, else v1 memory/memory.usage_in_bytes minus
# total_inactive_file of memory/memory.stat, under CGROOT (default /sys/fs/cgroup); a missing
# memory.stat or line subtracts nothing. The same reading as tools/snap_compare.py
# container_bytes. Prints nothing and returns 1 when neither usage file holds a number.
cgused() { local r="${CGROOT:-/sys/fs/cgroup}" b="" i s k
  b=$(cat "$r/memory.current" 2>/dev/null); s="$r/memory.stat"; k=inactive_file
  case "$b" in ''|*[!0-9]*) b=$(cat "$r/memory/memory.usage_in_bytes" 2>/dev/null)
    s="$r/memory/memory.stat"; k=total_inactive_file;; esac
  case "$b" in ''|*[!0-9]*) return 1;; esac
  i=$(awk -v k="$k" '$1 == k && $2 ~ /^[0-9]+$/ {print $2; exit}' "$s" 2>/dev/null)
  case "$i" in ''|*[!0-9]*) i=0;; esac
  [ "$i" -le "$b" ] || i=$b
  echo $((b - i)); }
# procscan: PS_ST, PS_PP and PS_PG (arrays indexed by pid): every process's state, parent pid and
# process group from /proc/<pid>/stat, read with shell builtins only, so no process of the scan
# itself shows up in it (a ps pipeline run from run.sh would be a member of run.sh's own group).
procscan() { local f s pid st pp pg
  PS_ST=(); PS_PP=(); PS_PG=()
  for f in /proc/[0-9]*/stat; do
    { read -r s < "$f"; } 2>/dev/null || continue
    pid=${s%% *}; read -r st pp pg _ <<< "${s##*) }"
    case "$pid$pp$pg" in ''|*[!0-9]*) continue;; esac
    PS_ST[pid]=$st; PS_PP[pid]=$pp; PS_PG[pid]=$pg; done; }
# snapmembers <groups> <except pid> [<except group>]: SM = the pids of procscan's processes that
# are not zombies, are not <except pid> and belong to one of the process groups <groups> (ids 0
# and 1 never count, nor <except group>).
snapmembers() { local p g gs=""; SM=""
  for g in $1; do case "$g" in ''|*[!0-9]*|0|1|"${3:-x}") ;; *) gs="$gs $g";; esac; done
  for p in "${!PS_ST[@]}"; do
    if [ "${PS_ST[p]}" = Z ] || [ "$p" = "$2" ]; then continue; fi
    case "$gs " in *" ${PS_PG[p]} "*) SM="$SM $p";; esac; done; }
# Pod-wide snapshot lock: one capture or compare at a time per pod, since each job gates only on
# its own parts. snaplive <pid> <lock dir>: the holder is live while <pid> runs a .../run.sh (not
# a zombie; a reused pid's cmdline names another program) or any process that is not a zombie
# belongs to process group <pid> (run.sh leads its own group under setsid) or to a group in
# <lock dir>/groups (the job's groups, snapwatch): so a KILLed run.sh whose driver or parts
# still run keeps the lock, as podgate counts a unit live while its group has a live member.
# The waiting shell's own group ($$; run.sh leads its group) never counts: a holder group id
# reused by this waiter would otherwise make it wait on itself.
snaplive() { case "$1" in ''|*[!0-9]*|0|1) return 1;; esac
  procscan
  if [ -n "${PS_ST[$1]}" ] && [ "${PS_ST[$1]}" != Z ] &&
     tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qE '/run\.sh( |$)'; then return 0; fi
  snapmembers "$1 $(tr -cs '0-9' ' ' < "$2/groups" 2>/dev/null)" "" "$$"; [ -n "$SM" ]; }
# snaplock <dir> <times.txt>: take the lock (mkdir <dir>; <dir>/pid = this shell's pid, <dir>/name
# = $W). Held by a live holder (snaplive): wait, printing "lock wait <time>: <holder> holds <dir>"
# once a minute. Stale (its holder not live; its pid is this shell's own, a pid reused since the
# holder died, as this shell does not hold the lock while it waits; or no pid file for 10 s: the
# holder died between mkdir and the pid write): removed, logged "SNAP_LOCK_STALE <pid|none>
# <epoch>" in <times.txt>, then 1 s pause and the counts start again. Taken: checked again after
# 1 s (a waiter that judged the old pid stale may have removed it meanwhile).
snaplock() { local d="$1" t="$2" p q n=0 e=0
  while :; do
    if mkdir "$d" 2>/dev/null; then
      echo "$W" > "$d/name"; echo "$$" > "$d/pid"; sleep 1
      [ "$(cat "$d/pid" 2>/dev/null)" = "$$" ] && return 0
      continue
    fi
    p=$(cat "$d/pid" 2>/dev/null)
    if [ -z "$p" ]; then e=$((e + 1)); else e=0; fi
    if { [ -n "$p" ] && { [ "$p" = "$$" ] || ! snaplive "$p" "$d"; }; } || [ "$e" -ge 10 ]; then
      q=$(cat "$d/pid" 2>/dev/null)
      if [ "$q" = "$p" ] && [ -d "$d" ]; then
        rm -rf "$d"; echo "SNAP_LOCK_STALE ${p:-none} $(date +%s)" >> "$t"; fi
      e=0; n=0; sleep 1; continue
    fi
    [ $((n % 60)) -eq 0 ] && echo "lock wait $(date +%T): $(cat "$d/name" 2>/dev/null) holds $d"
    n=$((n + 1)); sleep 1
  done; }
# snapunlock <dir>: release the lock when this shell holds it.
snapunlock() { [ "$(cat "$1/pid" 2>/dev/null)" = "$$" ] && rm -rf "$1"; return 0; }
# sgprune: drop from SG every group with no member left in procscan's scan (no process that is
# not a zombie has it as its group), so a group id the system reuses later for an unrelated
# process is not stopped (snapstop) or counted (snaplive) as the job's.
sgprune() { local p g lg="" n=""
  for p in "${!PS_ST[@]}"; do [ "${PS_ST[p]}" = Z ] || lg="$lg ${PS_PG[p]}"; done
  for g in $SG; do case "$lg " in *" $g "*) n="$n $g";; esac; done
  SG=${n# }; }
# snapwatch <pid> <lock dir>: wait for the job <pid> (a child of this shell) and return its exit
# code. Every 2 s it drops the groups with no member left from SG (sgprune), adds the process
# group of each descendant of this shell to SG (set by run.sh to its own pid) and, when SG
# changed while this shell holds the lock, writes it to <lock dir>/groups: a job process in a
# group of its own (should uv or the driver start one) still holds the lock (snaplive) and is
# still stopped (snapstop).
snapwatch() { local p q o
  while procscan; [ -n "${PS_ST[$1]}" ] && [ "${PS_ST[$1]}" != Z ]; do o=$SG; sgprune
    for p in "${!PS_PP[@]}"; do q=${PS_PP[p]}
      while [ -n "$q" ] && [ "$q" -gt 1 ] && [ "$q" != "$$" ]; do q=${PS_PP[q]}; done
      if [ "$q" = "$$" ]; then case " $SG " in *" ${PS_PG[p]} "*) ;; *) SG="$SG ${PS_PG[p]}";; esac; fi
    done
    [ "$SG" = "$o" ] || { [ "$(cat "$2/pid" 2>/dev/null)" = "$$" ] && echo "$SG" > "$2/groups"; }
    sleep 2 & wait $!
  done
  wait "$1"; }
# snapstop <times.txt>: stop what is left of the job: every process, other than this shell, that
# is not a zombie and belongs to a group in SG (pruned on every scan, sgprune). Logs "JOB_LEFT
# <pids> <epoch>" in <times.txt> when any is left, TERMs them, KILLs what is left after
# SNAPSTOP_KILL_S s (default 30) and returns 0 once none is left. One still live after
# SNAPSTOP_MAX_S s (default 40; at least the KILL time): logs "JOB_STUCK <pids> <epoch>" and
# returns 1, so run.sh keeps the pod lock rather than reporting the job stopped.
snapstop() { local i k="${SNAPSTOP_KILL_S:-30}" m="${SNAPSTOP_MAX_S:-40}"
  case "$k" in ''|*[!0-9]*) k=30;; esac; case "$m" in ''|*[!0-9]*) m=40;; esac; [ "$m" -ge "$k" ] || m=$k
  for ((i = 0; i <= m; i++)); do procscan; sgprune; snapmembers "$SG" "$$"
    [ -n "$SM" ] || return 0
    if [ "$i" = 0 ]; then echo "JOB_LEFT$SM $(date +%s)" >> "$1"; kill -TERM $SM 2>/dev/null; fi
    [ "$i" -ge "$k" ] && kill -KILL $SM 2>/dev/null
    [ "$i" = "$m" ] || sleep 1
  done
  echo "JOB_STUCK$SM $(date +%s)" >> "$1"; return 1; }
# snapgate <gate GB> <times.txt>: return once container memory without page cache (cgused) is
# under <gate GB>. While it is not, run.out gets "gate wait <time>: memory <GB> GB without page
# cache, gate <GB> GB, <n> pack units live" every 30 s. Once no pack unit has been live on the
# pod (podgate over /work/*/out/run/* and /work/*/*/out/run/*, as runner.sh counts them) for
# IDLE_WAIT_S s (default 300) it starts anyway, logging "GATE_IDLE_START <epoch>" in <times.txt>:
# memory no unit holds has not fallen (pod/gate.sh admit start-idle-wait). No memory reading:
# "GATE_NO_CGROUP_MEMORY <epoch>" in <times.txt>, exit 1.
snapgate() { local b live t0="" iw="${IDLE_WAIT_S:-300}"
  case "$iw" in ''|*[!0-9]*) iw=300;; esac
  while :; do
    b=$(cgused) || { echo "GATE_NO_CGROUP_MEMORY $(date +%s)" >> "$2"; exit 1; }
    [ $(( b / 1000000000 )) -ge "$1" ] || return 0
    read -r live _ <<< "$(NO_W_FILE_GB=0 RESERVE_TOP='' RESERVE_FULL_GB='' RESERVE_PCT='' podgate /work/*/out/run/* /work/*/*/out/run/*)"
    case "$live" in ''|*[!0-9]*) live=0;; esac
    if [ "$live" = 0 ]; then [ -n "$t0" ] || t0=$(date +%s)
      if [ $(( $(date +%s) - t0 )) -ge "$iw" ]; then echo "GATE_IDLE_START $(date +%s)" >> "$2"; return 0; fi
    else t0=""; fi
    echo "gate wait $(date +%T): memory $(( b / 1000000000 )) GB without page cache, gate $1 GB, $live pack units live"
    if [ "$iw" -gt 0 ] && [ "$iw" -lt 30 ]; then sleep "$iw" & else sleep 30 & fi
    wait $!
  done; }
refuse() { echo "REFUSED: $1"; [ -z "${2:-}" ] || echo "$2"; exit 3; }
[ $# -ge 1 ] || usage
MODE="$1"; shift

if [ "$MODE" = fetch ]; then
  [ $# -eq 2 ] || usage
  NAME="$1"; CODE="$2"; check_run_name "$NAME"
  O="$RENT/out/$NAME/fetch-$(date +%s)"; mkdir -p "$O"; cd "$O"
  HOME="$RENT/home" USERPROFILE="$RENT/home" "$RPC" receive "$CODE"
  for f in *.tgz; do [ -f "$f" ] || continue
    L=$(tar -tzf "$f") || refuse "cannot list $f"; echo "== $f"; printf '%s\n' "$L"
    printf '%s\n' "$L" | grep -qE '(^|/)\.\.(/|$)|^/' && refuse "$f holds a '..' or absolute path"
    tar -xzf "$f"; done
  echo "FETCHED into $O:"; ls -R "$O" | head -40
  [ -f times.txt ] && tail -3 times.txt
  exit 0
fi

DRY=0; TOOL=""; ALLOW=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --tool) [ $# -ge 2 ] || usage; TOOL="$2"; shift 2 ;;
    --allow-new) [ $# -ge 2 ] || usage; ALLOW="$2"; shift 2 ;;
    -*) echo "unknown option $1"; usage ;;
    *) POS+=("$1"); shift ;;
  esac
done
case "$MODE" in
  capture) [ ${#POS[@]} -ge 2 ] && [ ${#POS[@]} -le 3 ] && [ -z "$ALLOW" ] || usage ;;
  compare) [ ${#POS[@]} -eq 4 ] && [ -z "$TOOL" ] || usage ;;
  *) usage ;;
esac
SHA=$(git -C "$REPO" rev-parse --verify "${POS[0]}^{commit}") || refuse "unknown commit ${POS[0]}"
SHORT=${SHA:0:7}; TB="${POS[1]}"; [ -f "$TB" ] || refuse "no such tables archive" "$TB"
if [ "$MODE" = capture ]; then NAME="${POS[2]:-snapmain-$SHORT}"; CAP=""
else CAP="${POS[2]}"; NAME="${POS[3]}"
  echo "$CAP" | grep -qE '^/work/[a-z0-9][a-z0-9_-]*/[A-Za-z0-9_-]+$' || refuse "capture dir must be /work/<folder>/<dir>" "$CAP"
fi
check_run_name "$NAME"
[ -z "$ALLOW" ] || echo "$ALLOW" | grep -qE '^[A-Za-z0-9_][A-Za-z0-9_.-]*(,[A-Za-z0-9_][A-Za-z0-9_.-]*)*$' || refuse "--allow-new must be name,name,..." "$ALLOW"
TSHA=""; [ -z "$TOOL" ] || TSHA=$(git -C "$REPO" rev-parse --verify "$TOOL^{commit}") || refuse "unknown --tool commit $TOOL"
DECOMP=$(git -C "$REPO/external/pokeplatinum" rev-parse HEAD) || refuse "no decomp checkout at $REPO/external/pokeplatinum"
GATE="${SNAP_GATE_GB:-88}"; echo "$GATE" | grep -qE '^[1-9][0-9]*$' || refuse "SNAP_GATE_GB must be a whole number" "$GATE"
WORKERS="${SNAP_WORKERS:-16}"; echo "$WORKERS" | grep -qE '^[1-9][0-9]*$' || refuse "SNAP_WORKERS must be a whole number of at least 1" "$WORKERS"
O="$RENT/out/$NAME"; [ -e "$O" ] && refuse "$O exists; names are used once"

FORBID='\.(nds|sav|dsv|ds[0-9]|zip|xlsx|ods)$|(^|/)\.cache/|(^|/)backups?/|desktop|local\.toml'
BADPATH='\.\.|^/|^\./'
mkdir -p "$O"
trap 'rc=$?; [ "$rc" = 0 ] || { cd /; rm -rf "$O" || echo "WARNING: could not remove $O"; }' EXIT
# deny <reason> <text> <grep args>: refuse when grep selects a line; a grep error (rc > 1) refuses.
deny() { local why="$1" txt="$2" out rc=0; shift 2
  out=$(printf '%s\n' "$txt" | sed '/^$/d' | grep "$@") || rc=$?
  [ "$rc" -le 1 ] || refuse "grep error rc $rc: grep $*"
  [ -z "$out" ] || refuse "$why" "$out"; }

git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$O/code.tar" "$SHA"
FILES=(code.tar tables.tar.xz)
if [ -n "$TSHA" ]; then
  git -C "$REPO" -c core.autocrlf=false archive --format=tar -o "$O/tool.tar" "$TSHA" tools/snap_compare.py
  FILES+=(tool.tar)
fi
cp "$TB" "$O/tables.tar.xz"
TOP=$(git -C "$HERE" rev-parse --show-toplevel)
git -C "$TOP" show HEAD:tools/ops/day/pod/cache_sync.sh | tr -d '\r' > "$O/cache_sync.sh" || refuse "no tools/ops/day/pod/cache_sync.sh at HEAD of $TOP"
FILES+=(cache_sync.sh)
CT=$(tar -tf "$O/code.tar"; [ -z "$TSHA" ] || tar -tf "$O/tool.tar")
TALL=$(tar -tJf "$O/tables.tar.xz"); T=$(printf '%s\n' "$TALL" | grep -v '/$' || true)
deny "tables entry with '..', a leading '/' or './'" "$TALL" -E "$BADPATH"
deny "forbidden files" "$(printf '%s\n%s\n' "$CT" "$TALL")" -iE "$FORBID"
deny "calc or docs-sheet extraction in code.tar" "$CT" -E "$TABLE_DENY"
deny "calc or docs-sheet extraction in tables" "$TALL" -E "$TABLE_DENY"
deny "tables entry outside the allowed data/ paths" "$T" -vE "$TABLE_ALLOW"
[ -n "$T" ] || refuse "tables archive holds no files" "$TB"
tar -cf "$O/payload.tar" -C "$O" "${FILES[@]}"

W="/work/$NAME"
# --workers/--gate-gb only for a snap_compare that has them (older commits run their parts serially).
SRC="$SHA"; [ -z "$TSHA" ] || SRC="$TSHA"; PAR=""
SC=$(git -C "$REPO" show "$SRC:tools/snap_compare.py") || refuse "no tools/snap_compare.py at $SRC"
if [[ "$SC" == *'"--workers"'* ]]; then PAR=" --workers $WORKERS --gate-gb $GATE"
else echo "NOTE: tools/snap_compare.py at ${SRC:0:8} has no --workers: its parts run one after another"; fi
VOLQ="mountpoint -q /vol 2>/dev/null"; PRE=":"; POST=":"
# timeout -k 30 of the volume calls (PC env, written into run.sh): a fetch, a store
VTF="${CACHE_FETCH_TIMEOUT:-900}"; VTS="${CACHE_STORE_TIMEOUT:-900}"
[[ "$VTF" =~ ^[1-9][0-9]*$ ]] || refuse "CACHE_FETCH_TIMEOUT '$VTF' is not a whole number of seconds"
[[ "$VTS" =~ ^[1-9][0-9]*$ ]] || refuse "CACHE_STORE_TIMEOUT '$VTS' is not a whole number of seconds"
if [ "$MODE" = capture ]; then
  SCRIPT="tools/snap_compare.py"; [ -z "$TSHA" ] || SCRIPT="$W/tool/tools/snap_compare.py"
  JOB="CAPTURE"; CMD="uv run --frozen python \"$SCRIPT\" capture \"\$W/snap\" 0-199 --repo \"\$W/repo\" --commit $SHA$PAR -v > \"\$W/capture.log\" 2>&1"
  K=""; [ -n "$TSHA" ] || K=$(table_key16 capture "$SHA" "$TB")
  if [ -n "$TSHA" ]; then echo "NOTE: a --tool re-capture is not stored on the network volume"
  elif [ -z "$K" ]; then echo "NOTE: no table key for $TB (no .key file beside it and no tables_k<key16> name, or it differs from the build its key names; index.txt is not used for a capture): the capture is neither fetched from nor stored on the network volume"
  else SKEY="$SHA-$K"; echo "capture store key $SKEY (fetched instead of captured when the pod's volume has it)"
    PRE="if $VOLQ; then vr=0; timeout -k 30 $VTF bash \"\$W/cache_sync.sh\" snap-fetch $SKEY \"\$W\" own > \"\$W/store.log\" 2>&1 || vr=\$?
  if [ \"\$vr\" = 0 ]; then echo \"CAPTURE_FROM_VOLUME $SKEY \$(date +%s)\" >> \"\$W/times.txt\"; echo \"CAPTURE_RC=0 \$(date +%s)\" >> \"\$W/times.txt\"; echo ALL_DONE >> \"\$W/times.txt\"; exit 0; fi
  if [ \"\$vr\" = 124 ] || [ \"\$vr\" = 137 ]; then echo \"VOLUME_TIMEOUT fetch \$(date +%s)\" >> \"\$W/times.txt\"; rm -rf \"\$W/snap\" \"\$W\"/.snapfetch.*
    echo \"WARNING: the volume fetch of $SKEY timed out after $VTF s (rc \$vr): captured instead\"; fi
fi"
    POST="if [ \"\$rc\" = 0 ] && $VOLQ; then sr=0; timeout -k 30 $VTS bash \"\$W/cache_sync.sh\" snap-publish \"\$W\" $SKEY >> \"\$W/store.log\" 2>&1 || sr=\$?
  if [ \"\$sr\" = 124 ] || [ \"\$sr\" = 137 ]; then echo \"VOLUME_TIMEOUT store \$(date +%s)\" >> \"\$W/times.txt\"; echo \"WARNING: the volume store of $SKEY timed out after $VTS s (rc \$sr): not stored\"; fi
  echo \"STORE_RC=\$sr \$(date +%s)\" >> \"\$W/times.txt\"
fi"
  fi
else
  JOB="COMPARE"; AN=""; [ -z "$ALLOW" ] || AN=" --allow-new $ALLOW"
  CMD="uv run --frozen python tools/snap_compare.py compare \"$CAP\" 0-199 --repo \"\$W/repo\" --commit $SHA$AN$PAR -v > \"\$W/compare.log\" 2>&1"
  CS=""; K=""
  if [[ "$CAP" =~ ^/work/snapmain-([0-9a-f]{7})/snap$ ]]; then
    CS=$(git -C "$REPO" rev-parse --verify -q "${BASH_REMATCH[1]}^{commit}" 2>/dev/null || true)
    [ -z "$CS" ] || K=$(table_key16 compare "$CS" "$RENT/tables_${CS:0:8}.tar.xz"); fi
  if [ -z "$K" ]; then echo "NOTE: no store key for $CAP (only /work/snapmain-<short>/snap of a commit with a table key in $RENT/tables): a capture missing on the pod is not fetched from the network volume"
  else CKEY="$CS-$K"; echo "capture store key $CKEY (fetched when $CAP is not on the pod)"
    PRE="if [ ! -f \"$CAP/capture.json\" ] && $VOLQ; then vr=0; timeout -k 30 $VTF bash \"\$W/cache_sync.sh\" snap-fetch $CKEY \"${CAP%/snap}\" > \"\$W/store.log\" 2>&1 || vr=\$?
  echo \"FETCH_RC=\$vr \$(date +%s)\" >> \"\$W/times.txt\"
  if [ \"\$vr\" = 124 ] || [ \"\$vr\" = 137 ]; then echo \"VOLUME_TIMEOUT fetch \$(date +%s)\" >> \"\$W/times.txt\"; rm -rf \"${CAP%/snap}\"/.snapfetch.*
    echo \"WARNING: the volume fetch of $CKEY timed out after $VTF s (rc \$vr)\"; fi
  if [ ! -f \"$CAP/capture.json\" ]; then echo \"NO_CAPTURE $CAP \$(date +%s)\" >> \"\$W/times.txt\"; exit 1; fi
fi"
  fi
fi
cat > "$O/run.sh" <<EOS
#!/bin/bash
# $MODE of $SHA; written by tools/ops/day/snap.sh
export PATH=/root/.local/bin:/work/_bin:\$PATH UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
W="$W"; printf '%s\n' "$CAP" > "\$W/reads"; cd "\$W/repo"
$(declare -f cgused procscan snapmembers snaplive snaplock snapunlock sgprune snapwatch snapstop snapgate reserve_knobs podgate)
L=/work/.snaplock; SG=\$\$
# Exit (TERM included) releases the pod lock; a TERM first writes TERM <epoch> to times.txt and
# stops the job (snapstop: run.sh's process group, which it leads under setsid, and every group
# snapwatch saw a descendant in), then exits 143. A job process snapstop could not stop
# (JOB_STUCK) keeps the lock: the EXIT trap is dropped, so the next job waits until it is gone.
trap 'snapunlock "\$L"' EXIT
trap 'trap "" TERM; echo "TERM \$(date +%s)" >> "\$W/times.txt"; snapstop "\$W/times.txt" || trap - EXIT; exit 143' TERM
echo "START \$(date +%s)" > "\$W/times.txt"
m=\$(tr -dc 0-9 < /proc/sys/vm/max_map_count 2>/dev/null); [ "\${m:-0}" -ge 1048576 ] 2>/dev/null || { echo "MAP_COUNT_LOW \${m:-none} \$(date +%s)" >> "\$W/times.txt"; exit 1; }
(timeout -k 30 1800 uv sync --frozen -q) > "\$W/sync.log" 2>&1; rc=\$?; echo "SYNC_RC=\$rc \$(date +%s)" >> "\$W/times.txt"; [ "\$rc" = 0 ] || exit 1
export CACHE_PY="\$W/repo/.venv/bin/python"
snaplock "\$L" "\$W/times.txt"; echo "SNAP_LOCK \$(date +%s)" >> "\$W/times.txt"
$PRE
snapgate $GATE "\$W/times.txt"; echo "${JOB}_START \$(date +%s)" >> "\$W/times.txt"
$CMD &
snapwatch \$! "\$L"; rc=\$?; echo "${JOB}_RC=\$rc \$(date +%s)" >> "\$W/times.txt"
snapstop "\$W/times.txt" || { trap - EXIT; echo "ERROR: job processes still live after the KILL (JOB_STUCK in times.txt): the pod lock stays held until they are gone"; exit 1; }
$POST
echo ALL_DONE >> "\$W/times.txt"
EOS
tr -d '\r' < "$O/run.sh" > "$O/run.tmp" && mv "$O/run.tmp" "$O/run.sh"
bash -n "$O/run.sh" || refuse "generated run.sh does not parse" "$O/run.sh"
# The body: exec-pod refuses an argv item over 4096 characters and an argv over 256 items, so
# run.sh travels base64-encoded in chunks of at most 4000 characters after the script, as
# common.sh exec_body does: ["sh","-c",<script>,"_",<chunk>,...]. The script joins them
# (printf %s "$@"), decodes run.sh and runs it only when its sha256 is run.sh's on the PC
# (else RUNSH_SHA_MISMATCH <sha>, exit 1, nothing started). Refuses (exit 3) a body that would
# break either limit.
LIM=4000; RB=$(base64 -w0 "$O/run.sh"); RSUM=$(sha256sum "$O/run.sh" | cut -c1-64); CH=()
while [ -n "$RB" ]; do CH+=("${RB:0:$LIM}"); RB=${RB:$LIM}; done
[ $(( ${#CH[@]} + 4 )) -le 256 ] || refuse "exec body would have $(( ${#CH[@]} + 4 )) argv items, over 256 (run.sh too large)"

echo "== payload.tar"; tar -tvf "$O/payload.tar"
echo "== code.tar $(printf '%s\n' "$CT" | wc -l) entries; tables $(printf '%s\n' "$T" | wc -l) files"
echo "== run.sh"; cat "$O/run.sh"

if [ "$DRY" = 1 ]; then CODE="<code>"; SPID=""
else
  cd "$O"; HOME="$RENT/home" USERPROFILE="$RENT/home" nohup bash -c '"$0" send payload.tar & c=$!; trap "kill $c" TERM; wait "$c"; echo "SEND_RC=$?"' "$RPC" > send.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 60); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
  CODE=$(grep -o "code is: [^ ]*" send.log | head -1 | sed 's/code is: //' | tr -d '\r' || true)
  [ -n "$CODE" ] || { echo "no code; send.log:"; cat send.log; kill "$SPID" 2>/dev/null || true; exit 4; }
fi

UNTOOL=""; [ -z "$TSHA" ] || UNTOOL=" && mkdir tool && tar --no-same-owner -xf tool.tar -C tool"
DM="/work/_decomp/$DECOMP"
PREP="/work/_bin/runpodctl receive $CODE || { echo RECEIVE_FAILED; exit 1; }; [ -d $DM ] || { echo \"DECOMP_MISSING: $DM is not on this pod; send any pack.sh run first (its setup.sh creates it), then send this body again under a new name\"; exit 1; }; { tar --no-same-owner -xf payload.tar && mkdir repo && tar --no-same-owner -xf code.tar -C repo && tar --no-same-owner -xJf tables.tar.xz -C repo$UNTOOL && mkdir -p repo/external && ln -sfn $DM repo/external/pokeplatinum; } || { echo UNPACK_FAILED; exit 1; }; echo PREP_OK; setsid nohup bash $W/run.sh > run.out 2>&1 < /dev/null &"
SHC="mkdir -p $W && cd $W || exit 1; printf %s \"\$@\" | base64 -d > run.sh || { echo RUNSH_DECODE_FAILED; exit 1; }; s=\$(sha256sum run.sh | cut -c1-64); [ \"\$s\" = $RSUM ] || { echo RUNSH_SHA_MISMATCH \$s; rm -f run.sh; exit 1; }; chmod +x run.sh && nohup sh -c '$PREP' > prep.log 2>&1 < /dev/null & sleep 20; tail -c 300 $W/prep.log"
python_json() { local s="$1"; s=${s//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; }
[ "${#SHC}" -le "$LIM" ] || { [ -z "$SPID" ] || kill "$SPID" 2>/dev/null; refuse "exec body script is ${#SHC} characters, over $LIM"; }
ITEMS=""; for c in "${CH[@]}"; do ITEMS="$ITEMS, \"$c\""; done
printf '{"command": ["sh", "-c", %s, "_"%s], "timeoutSeconds": 25}\n' "$(python_json "$SHC")" "$ITEMS" > "$O/body.json"
echo "== exec-pod body ($O/body.json):"; cat "$O/body.json"
[ "$DRY" = 1 ] && echo "DRY RUN: nothing sent; delete $O before reusing the name" && exit 0
echo "== finished when exec 'tail -3 $W/times.txt' shows ${JOB}_RC=<n> then ALL_DONE"
