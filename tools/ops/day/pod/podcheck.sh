#!/usr/bin/env bash
# POD SIDE. Usage: podcheck.sh [<root> [<POD_SCRIPTS>]]   (root default /work). The PC's
# tools/ops/day/podcheck_body.sh prints the exec-pod body that writes this script and gate.sh to
# /work/_podcheck/ and runs it with the POD_SCRIPTS of the checkout's HEAD (DAYBOX.md "Send a run"
# step 3). Lists every active run under <root>/*/ and <root>/*/*/ (folders starting with _
# skipped; active: a setup.pid or runner.pid process alive and not a zombie):
#   run <name> gate=<GATE_VERSION|old> scripts=<first 12 of POD_SCRIPTS|old> runner=<alive|setup> live=<units> pending=<units>
# GATE_VERSION and POD_SCRIPTS come from the run's gate.env (setup.sh writes it from pack.sh's
# meta.env); a run without one was packed by an old script: it ignores reserves, the gate lock and
# claims. Then:
#   memory <GB> GB limit <GB GB|max|none>   gate.sh cgmem (cgroup v2, else v1); "none" when no file
#                                           is readable; limit "max": no limit (v2 "max", or v1 at 2^60 or more)
#   claim <run>.<unit> age <s> s weight <GB> GB runner <alive|dead>   for each file in <root>/.claims
# and the last line: PODCHECK OK (exit 0: no active old-script run and, when <POD_SCRIPTS> is
# given, no active run with other scripts), or PODCHECK MIXED old=<names,|-> other=<names,|->
# (exit 1). Prints nothing from meta.env (beacon codes).
R="${1:-/work}"; WANT="${2:-}"; NO_W_FILE_GB="${DEFAULT_WEIGHT_GB:-12}"
. "$(cd "$(dirname "$0")" && pwd)/gate.sh"   # cgmem, pidlive, podgate
old=""; other=""
for d in "$R"/*/ "$R"/*/*/; do d=${d%/}; n=${d#"$R"/}
  case "$n" in _*) continue;; esac
  [ -d "$d/out" ] || [ -f "$d/setup.pid" ] || continue
  rs=""; pidlive "$(cat "$d/runner.pid" 2>/dev/null)" && rs=alive
  [ -z "$rs" ] && pidlive "$(cat "$d/setup.pid" 2>/dev/null)" && rs=setup
  [ -n "$rs" ] || continue
  gv=$(sed -n 's/^GATE_VERSION=//p' "$d/gate.env" 2>/dev/null); ps_=$(sed -n 's/^POD_SCRIPTS=//p' "$d/gate.env" 2>/dev/null)
  if [ -z "$gv" ]; then gv=old; ps_=old; old="$old $n"
  elif [ -n "$WANT" ] && [ "$ps_" != "$WANT" ]; then other="$other $n"; fi
  read -r live _ <<< "$(podgate "$d"/out/run/*)"
  pend=$(grep -c . "$d/out/pending.txt" 2>/dev/null || true)
  echo "run $n gate=$gv scripts=${ps_:0:12} runner=$rs live=$live pending=${pend:-0}"
done
cur=none; b=$(cgmem current) && cur="$(( b / 1000000000 )) GB"
lim=none; b=$(cgmem max) && case "$b" in max) lim=max;; ''|*[!0-9]*) lim="?";;
  *) if [ "${#b}" -ge 19 ] || [ "$b" -ge 1152921504606846976 ]; then lim=max; else lim="$(( b / 1000000000 )) GB"; fi;; esac
echo "memory $cur limit $lim"
now=$(date +%s)
for c in "$R"/.claims/*; do [ -f "$c" ] || continue; read -r e w p < "$c"
  case "$e" in ''|*[!0-9]*) echo "claim ${c##*/} malformed"; continue;; esac
  echo "claim ${c##*/} age $(( now - e )) s weight ${w:-?} GB runner $(pidlive "$p" && echo alive || echo dead)"
done
old=${old# }; old=${old// /,}; other=${other# }; other=${other// /,}
if [ -n "$old$other" ]; then echo "PODCHECK MIXED old=${old:--} other=${other:--}"; exit 1; fi
echo "PODCHECK OK"
