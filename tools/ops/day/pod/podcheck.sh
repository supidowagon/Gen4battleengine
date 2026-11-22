#!/usr/bin/env bash
# POD SIDE. Usage: podcheck.sh [<root> [<POD_SCRIPTS> [<POD_WEIGHTS> [<quota GB> [<warn GB>]]]]]   (root default /work). The PC's
# tools/ops/day/podcheck_body.sh prints the exec-pod body that writes this script and gate.sh to
# /work/_podcheck/ and runs it with the POD_SCRIPTS and POD_WEIGHTS of the checkout's HEAD
# (DAYBOX.md "Send a run" step 3). Lists every active run under <root>/*/ and <root>/*/*/ (folders starting with _
# skipped; active: a setup.pid, runner.pid or quick.pid (a quick lane run, bin/quick.sh) process alive
# and not a zombie):
#   run <name> gate=<GATE_VERSION|old> scripts=<first 12 of POD_SCRIPTS|old> runner=<alive|setup|quick> live=<units> pending=<units>
# GATE_VERSION, POD_SCRIPTS (the *.sh and weights.py) and POD_WEIGHTS (weights.txt) come from the
# run's gate.env (setup.sh writes it from pack.sh's meta.env); a run without one was packed by an
# old script: it ignores reserves, the gate lock and claims. Then, when <POD_WEIGHTS> is given and
# an active new-script run has another weights.txt (other expected peaks, same gate rules):
#   weights other=<names,>
# then:
#   memory <GB> GB limit <GB GB|max|none>   gate.sh cgmem (cgroup v2, else v1); "none" when no file
#                                           is readable; limit "max": no limit (v2 "max", or v1 at 2^60 or more)
#   volume /vol used <GB|?> GB of <quota> GB entries <n> captures <m> | volume none
#        the network volume when /vol (CACHE_VOL) is a mountpoint (cache_sync.sh). GB here is
#        10^9 bytes, the unit of the volume's size: used = du -sk in bytes, shown rounded up; "?"
#        when du fails or outlasts PODCHECK_DU_TIMEOUT s (default 10, at most 15: the exec-pod
#        call has 25 s), so a slow mount cannot hang the check (df of the mount shows the whole
#        storage cluster, not the volume); quota: 4th argument, else CACHE_VOL_QUOTA_GB, default 30
#        (the volume's size, which the pod cannot read); compile cache entries (jax/*/*-cache) and
#        stored captures (snap/*); none: no volume. Then, when quota minus used is under <warn GB>
#        (5th argument, else CACHE_FREE_WARN_GB, default 3, the knob of cache_sync.sh's prune):
#   WARNING volume /vol has <GB> GB left of <quota> GB (under <warn> GB): tell the owner
#        (<GB> left rounded down)
#   claim <run>.<unit> age <s> s weight <GB> GB runner <alive|not-runner|dead>   for each file in
#        <root>/.claims (not-runner: the pid is alive but runs no runner*.sh, a reused pid)
# and the last line: PODCHECK OK (exit 0: no active old-script run and, when <POD_SCRIPTS> is
# given, no active run with other scripts; other weights alone never fail it), or PODCHECK MIXED old=<names,|-> other=<names,|->
# (exit 1). Prints nothing from meta.env (beacon codes).
R="${1:-/work}"; WANT="${2:-}"; WANTW="${3:-}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"
Q="${4:-${CACHE_VOL_QUOTA_GB:-30}}"; [[ "$Q" =~ ^[1-9][0-9]*$ ]] || Q=30
FW="${5:-${CACHE_FREE_WARN_GB:-3}}"; [[ "$FW" =~ ^[0-9]+$ ]] || FW=3
DUT="${PODCHECK_DU_TIMEOUT:-10}"; [[ "$DUT" =~ ^[1-9][0-9]*$ ]] || DUT=10; [ "$DUT" -le 15 ] || DUT=15
. "$(cd "$(dirname "$0")" && pwd)/gate.sh"   # cgmem, pidlive, podgate
old=""; other=""; wother=""
for d in "$R"/*/ "$R"/*/*/; do d=${d%/}; n=${d#"$R"/}
  case "$n" in _*) continue;; esac
  [ -d "$d/out" ] || [ -f "$d/setup.pid" ] || [ -f "$d/quick.pid" ] || continue
  rs=""; pidlive "$(cat "$d/runner.pid" 2>/dev/null)" && rs=alive
  [ -z "$rs" ] && pidlive "$(cat "$d/setup.pid" 2>/dev/null)" && rs=setup
  [ -z "$rs" ] && pidlive "$(cat "$d/quick.pid" 2>/dev/null)" && rs=quick
  [ -n "$rs" ] || continue
  gv=$(sed -n 's/^GATE_VERSION=//p' "$d/gate.env" 2>/dev/null); ps_=$(sed -n 's/^POD_SCRIPTS=//p' "$d/gate.env" 2>/dev/null)
  if [ -z "$gv" ]; then gv=old; ps_=old; old="$old $n"
  else [ -n "$WANT" ] && [ "$ps_" != "$WANT" ] && other="$other $n"
    [ -n "$WANTW" ] && [ "$(sed -n 's/^POD_WEIGHTS=//p' "$d/gate.env" 2>/dev/null)" != "$WANTW" ] && wother="$wother $n"; fi
  read -r live _ <<< "$(podgate "$d"/out/run/*)"
  pend=$(grep -c . "$d/out/pending.txt" 2>/dev/null || true)
  echo "run $n gate=$gv scripts=${ps_:0:12} runner=$rs live=$live pending=${pend:-0}"
done
wother=${wother# }; [ -n "$wother" ] && echo "weights other=${wother// /,}"
cur=none; b=$(cgmem current) && cur="$(( b / 1000000000 )) GB"
lim=none; b=$(cgmem max) && case "$b" in max) lim=max;; ''|*[!0-9]*) lim="?";;
  *) if [ "${#b}" -ge 19 ] || [ "$b" -ge 1152921504606846976 ]; then lim=max; else lim="$(( b / 1000000000 )) GB"; fi;; esac
echo "memory $cur limit $lim"
V="${CACHE_VOL:-/vol}"   # the network volume (cache_sync.sh)
if [ -d "$V" ] && { [ "${CACHE_VOL_MOUNTCHECK:-1}" = 0 ] || mountpoint -q "$V" 2>/dev/null; }; then
  k=$(timeout -k 2 "$DUT" du -sk "$V" 2>/dev/null | awk 'NR == 1 {print $1}'); used="?"; G=1000000000
  [[ "$k" =~ ^[0-9]+$ ]] && ub=$(( k * 1024 )) && used=$(( (ub + G - 1) / G ))
  ne=$(find "$V/jax" -mindepth 2 -maxdepth 2 -type f -name '*-cache' ! -name '.*' 2>/dev/null | wc -l)
  nc=$(find "$V/snap" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | wc -l)
  echo "volume $V used $used GB of $Q GB entries $ne captures $nc"
  if [ "$used" != "?" ] && [ $(( Q * G - ub )) -lt $(( FW * G )) ]; then
    echo "WARNING volume $V has $(( (Q * G - ub) / G )) GB left of $Q GB (under $FW GB): tell the owner"; fi
else echo "volume none"; fi
now=$(date +%s)
for c in "$R"/.claims/*; do [ -f "$c" ] || continue; read -r e w p < "$c"
  case "$e" in ''|*[!0-9]*) echo "claim ${c##*/} malformed"; continue;; esac
  echo "claim ${c##*/} age $(( now - e )) s weight ${w:-?} GB runner $(if ! pidlive "$p"; then echo dead; elif runnerpid "$p"; then echo alive; else echo not-runner; fi)"
done
old=${old# }; old=${old// /,}; other=${other# }; other=${other// /,}
if [ -n "$old$other" ]; then echo "PODCHECK MIXED old=${old:--} other=${other:--}"; exit 1; fi
echo "PODCHECK OK"
