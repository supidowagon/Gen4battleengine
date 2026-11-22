#!/usr/bin/env bash
# POD SIDE. Usage: podcheck.sh [<root> [<POD_SCRIPTS> [<POD_WEIGHTS>]]]   (root default /work). The PC's
# tools/ops/day/podcheck_body.sh prints the exec-pod body that writes this script and gate.sh to
# /work/_podcheck/ and runs it with the POD_SCRIPTS and POD_WEIGHTS of the checkout's HEAD
# (DAYBOX.md "Send a run" step 3). Lists every active run under <root>/*/ and <root>/*/*/ (folders starting with _
# skipped; active: a setup.pid or runner.pid process alive and not a zombie):
#   run <name> gate=<GATE_VERSION|old> scripts=<first 12 of POD_SCRIPTS|old> runner=<alive|setup> live=<units> pending=<units>
# GATE_VERSION, POD_SCRIPTS (the *.sh and weights.py) and POD_WEIGHTS (weights.txt) come from the
# run's gate.env (setup.sh writes it from pack.sh's meta.env); a run without one was packed by an
# old script: it ignores reserves, the gate lock and claims. Then, when <POD_WEIGHTS> is given and
# an active new-script run has another weights.txt (other expected peaks, same gate rules):
#   weights other=<names,>
# then:
#   memory <GB> GB limit <GB GB|max|none>   gate.sh cgmem (cgroup v2, else v1); "none" when no file
#                                           is readable; limit "max": no limit (v2 "max", or v1 at 2^60 or more)
#   volume /vol free <GB> GB entries <n> captures <m> | volume none
#        the network volume when /vol (CACHE_VOL) is a mountpoint (cache_sync.sh): free space,
#        compile cache entries (jax/*/*-cache) and stored captures (snap/*); none: no volume
#   claim <run>.<unit> age <s> s weight <GB> GB runner <alive|not-runner|dead>   for each file in
#        <root>/.claims (not-runner: the pid is alive but runs no runner*.sh, a reused pid)
# and the last line: PODCHECK OK (exit 0: no active old-script run and, when <POD_SCRIPTS> is
# given, no active run with other scripts; other weights alone never fail it), or PODCHECK MIXED old=<names,|-> other=<names,|->
# (exit 1). Prints nothing from meta.env (beacon codes).
R="${1:-/work}"; WANT="${2:-}"; WANTW="${3:-}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"
. "$(cd "$(dirname "$0")" && pwd)/gate.sh"   # cgmem, pidlive, podgate
old=""; other=""; wother=""
for d in "$R"/*/ "$R"/*/*/; do d=${d%/}; n=${d#"$R"/}
  case "$n" in _*) continue;; esac
  [ -d "$d/out" ] || [ -f "$d/setup.pid" ] || continue
  rs=""; pidlive "$(cat "$d/runner.pid" 2>/dev/null)" && rs=alive
  [ -z "$rs" ] && pidlive "$(cat "$d/setup.pid" 2>/dev/null)" && rs=setup
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
  fr=$(df -Pk "$V" 2>/dev/null | awk 'NR == 2 {printf "%d", $4 / 1048576}')
  ne=$(find "$V/jax" -mindepth 2 -maxdepth 2 -type f -name '*-cache' ! -name '.*' 2>/dev/null | wc -l)
  nc=$(find "$V/snap" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | wc -l)
  echo "volume $V free ${fr:-?} GB entries $ne captures $nc"
else echo "volume none"; fi
now=$(date +%s)
for c in "$R"/.claims/*; do [ -f "$c" ] || continue; read -r e w p < "$c"
  case "$e" in ''|*[!0-9]*) echo "claim ${c##*/} malformed"; continue;; esac
  echo "claim ${c##*/} age $(( now - e )) s weight ${w:-?} GB runner $(if ! pidlive "$p"; then echo dead; elif runnerpid "$p"; then echo alive; else echo not-runner; fi)"
done
old=${old# }; old=${old// /,}; other=${other# }; other=${other// /,}
if [ -n "$old$other" ]; then echo "PODCHECK MIXED old=${old:--} other=${other:--}"; exit 1; fi
echo "PODCHECK OK"
