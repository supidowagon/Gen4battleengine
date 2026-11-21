#!/usr/bin/env bash
# POD SIDE. Per test file: collected passed failed skipped errors xfailed xpassed not_run max_rss_MB
# wall_sum_s complete (passed + failed + skipped + errors + xfailed + xpassed = collected, the rule
# in docs/process.md section 5); then collection failures, crashed units (rc 97), units killed a
# third time (rc 98), units with any other rc but 0, 1 or 5 (a failure even with complete counts),
# wall time, container peak while this run ran (max of mem_mb.txt) and the container's lifetime peak, CPU-s (user + sys) of the units and of the warm-up (out/warmup/),
# per unit its compile-cache hits and writes and the highest peak of a test with a hit and of one
# with cold compiles only (tests/conftest.py's summary line; the memtable's hit-peak column),
# largest process, per unit its peak RSS and wall time (finished units), the 15 slowest tests, the count over 60 s, every FAILED/ERROR and SKIPPED line,
# and a last line RESULT: PASS or RESULT: FAIL (<reasons>).
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; mkdir -p "$O"; cd "$O"; . "$D/bin/gate.sh"   # cgmem: cgroup v2 or v1
cnt() { v=$(echo "$1" | grep -oE "[0-9]+ $2" | head -1 | cut -d' ' -f1); echo "${v:-0}"; }
TAB=$(while read -r u f; do n=$(grep -c . $u.ids)
  if [ -f $u.rc ]; then l=$(grep -E ' in [0-9.]+s' $u.log 2>/dev/null | tail -1)
    rss=$(awk '/Maximum resident/ {print int($6/1024)}' $u.time 2>/dev/null); w=$(( $(cat $u.end) - $(cat $u.start 2>/dev/null || cat $u.end) ))
    echo "$f $n $(cnt "$l" passed) $(cnt "$l" failed) $(cnt "$l" skipped) $(cnt "$l" 'errors?') $(cnt "$l" xfailed) $(cnt "$l" xpassed) 0 ${rss:-0} $w"
  else echo "$f $n 0 0 0 0 0 0 $n 0 0"; fi
done < queue.txt | awk '{k=$1; if(!(k in c)) o[++m]=k; c[k]+=$2; for(i=3;i<=9;i++) a[k,i]+=$i; if($10>r[k]) r[k]=$10; w[k]+=$11}
  END {print "file collected passed failed skipped errors xfailed xpassed not_run max_rss_MB wall_sum_s complete";
       for(j=1;j<=m;j++){k=o[j]; s=k" "c[k]; t=0; for(i=3;i<=9;i++) s=s" "a[k,i]; for(i=3;i<=8;i++) t+=a[k,i];
         print s, r[k]+0, w[k]+0, (t==c[k] ? "yes" : "NO")}}')
echo "$TAB"
NF=$(echo "$TAB" | awk 'NR>1 {f+=$4+$6} END {print f+0}'); NI=$(echo "$TAB" | awk 'NR>1 && $12=="NO"' | wc -l)
echo "--- collection failures (collect_rc not 0 or 5)"
CF=$(grep -vE 'collect_rc=(0|5)$' "$D/collected.txt" 2>/dev/null); [ -n "$CF" ] && echo "$CF"
grep -hE '^ERROR ' "$D"/col/*.out 2>/dev/null | cut -c1-260
NC=$( [ -n "$CF" ] && echo "$CF" | wc -l || echo 0)
NK=$(cat crashed.txt 2>/dev/null | wc -l); [ "$NK" -gt 0 ] && echo "--- crashed units (rc 97): $(tr '\n' ' ' < crashed.txt)"
NO=$(cat overkilled.txt 2>/dev/null | wc -l); [ "$NO" -gt 0 ] && echo "--- units killed a third time at the memory limit, or left without a result after two requeues (those are in lost.txt) (rc 98): $(tr '\n' ' ' < overkilled.txt)"
BR=$(for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.rc ] || continue; c=$(cat $u.rc)
  case $c in 0|1|5|97|98) ;; *) echo "$u rc=$c $(grep "^$u " queue.txt | cut -d' ' -f2)";; esac; done)
NB=$( [ -n "$BR" ] && echo "$BR" | wc -l || echo 0); [ "$NB" -gt 0 ] && { echo "--- units with an rc other than 0, 1 or 5"; echo "$BR"; }
PK=n/a; b=$(cgmem peak) && PK="$(( b / 1048576 )) MB"
RP=$(awk '$2 ~ /^[0-9]+$/ && $2 + 0 > m {m = $2 + 0} END {print (m > 0) ? m " MB" : "n/a"}' mem_mb.txt 2>/dev/null); RP=${RP:-n/a}
echo "--- wall $(( $(cat ALL.end 2>/dev/null || date +%s) - $(cat ALL.start 2>/dev/null || cat ALL.end 2>/dev/null || date +%s) )) s; container peak while this run ran (max of its mem_mb.txt, 15 s samples, incl. other runs) $RP; container lifetime peak (since the pod started, incl. earlier and other runs) $PK"
cpus() { awk '/User time \(seconds\)|System time \(seconds\)/ {s += $NF} END {printf "%.0f", s}' "$@" /dev/null; }
echo "--- CPU-s (user+sys) units $(cpus $(cut -d' ' -f1 queue.txt | sed 's/$/.time/' | while read -r t; do [ -f "$t" ] && echo "$t"; done))"
[ -f warmup/warmup.rc ] && echo "--- warm-up rc $(cat warmup/warmup.rc) wall $(( $(cat warmup/warmup.end 2>/dev/null || date +%s) - $(cat warmup/warmup.start 2>/dev/null || date +%s) )) s CPU-s $(cpus warmup/warmup.time 2>/dev/null) max_rss_MB $(awk '/Maximum resident/ {print int($6/1024)}' warmup/warmup.time 2>/dev/null)"
for u in $(cut -d' ' -f1 queue.txt); do awk -v u=$u '/Maximum resident/ {print int($6/1024), u}' $u.time 2>/dev/null; done | sort -n | tail -1 | while read -r m u; do echo "largest process ${m} MB: $u $(head -1 $u.ids)"; done
echo "--- per unit: unit ids max_rss_MB wall_s rc first_id"
while read -r u f; do [ -f $u.rc ] || continue
  m=$(awk '/Maximum resident/ {print int($6/1024)}' $u.time 2>/dev/null)
  echo "$u $(grep -c . $u.ids) ${m:-0} $(( $(cat $u.end) - $(cat $u.start 2>/dev/null || cat $u.end) )) $(cat $u.rc) $(head -1 $u.ids | cut -c1-160)"
done < queue.txt
echo "--- compile cache per unit: unit hits writes hit_peak_GB cold_peak_GB first_id"
while read -r u f; do [ -f $u.log ] || continue
  grep -h '^compile cache: ' $u.log | tail -1 | sed -E 's/^compile cache: ([0-9]+) hits, ([0-9]+) writes; .* with a hit ([0-9.]+) GB .* cold compiles only ([0-9.]+) GB.*/\1 \2 \3 \4/' | while read -r h w hp cp; do echo "$u $h $w $hp $cp $(head -1 $u.ids | cut -c1-160)"; done
done < queue.txt
DUR=$(for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -hE '^[0-9.]+s (call|setup|teardown) ' $u.log; done | sort -rn)
echo "--- slowest 15 (of $(echo "$DUR" | awk '$1+0 > 60' | wc -l) over 60 s)"; echo "$DUR" | head -15 | cut -c1-200
echo "--- failures"; for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -E '^(FAILED|ERROR) ' $u.log; done | cut -c1-260
echo "--- skips"; for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -E '^SKIPPED ' $u.log; done | cut -c1-260
WHY=""; [ "$NF" -gt 0 ] && WHY="$WHY $NF failed/errors;"; [ "$NC" -gt 0 ] && WHY="$WHY $NC collection failures;"
[ "$NK" -gt 0 ] && WHY="$WHY $NK crashed units;"; [ "$NO" -gt 0 ] && WHY="$WHY $NO units killed three times;"
[ "$NB" -gt 0 ] && WHY="$WHY $NB units with a bad rc;";[ "$NI" -gt 0 ] && WHY="$WHY $NI incomplete files;"
[ -f DONE ] || WHY="$WHY run not finished;"
[ -z "$WHY" ] && echo "RESULT: PASS" || echo "RESULT: FAIL ($WHY )"
