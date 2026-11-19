#!/usr/bin/env bash
# POD SIDE. Per test file: collected passed failed skipped errors xfailed xpassed not_run max_rss_MB wall_sum_s;
# then wall time, container peak, largest process, the 15 slowest tests, the count over 60 s,
# and every FAILED/ERROR and SKIPPED line.
D=$(cd "$(dirname "$0")/.." && pwd); O="$D/out"; cd "$O"
cnt() { v=$(echo "$1" | grep -oE "[0-9]+ $2" | head -1 | cut -d' ' -f1); echo "${v:-0}"; }
while read -r u f; do n=$(grep -c . $u.ids)
  if [ -f $u.rc ]; then l=$(grep -E ' in [0-9.]+s' $u.log | tail -1)
    rss=$(awk '/Maximum resident/ {print int($6/1024)}' $u.time 2>/dev/null); w=$(( $(cat $u.end) - $(cat $u.start) ))
    echo "$f $n $(cnt "$l" passed) $(cnt "$l" failed) $(cnt "$l" skipped) $(cnt "$l" 'errors?') $(cnt "$l" xfailed) $(cnt "$l" xpassed) 0 ${rss:-0} $w"
  else echo "$f $n 0 0 0 0 0 0 $n 0 0"; fi
done < queue.txt | awk '{k=$1; if(!(k in c)) o[++m]=k; c[k]+=$2; for(i=3;i<=9;i++) a[k,i]+=$i; if($10>r[k]) r[k]=$10; w[k]+=$11}
  END {print "file collected passed failed skipped errors xfailed xpassed not_run max_rss_MB wall_sum_s";
       for(j=1;j<=m;j++){k=o[j]; s=k" "c[k]; for(i=3;i<=9;i++) s=s" "a[k,i]; print s, r[k], w[k]}}'
echo "--- wall $(( $(cat ALL.end 2>/dev/null || date +%s) - $(cat ALL.start) )) s; container peak (incl. other runs) $(( $(cat /sys/fs/cgroup/memory.peak) / 1048576 )) MB"
for u in $(cut -d' ' -f1 queue.txt); do awk -v u=$u '/Maximum resident/ {print int($6/1024), u}' $u.time 2>/dev/null; done | sort -n | tail -1 | while read -r m u; do echo "largest process ${m} MB: $u $(head -1 $u.ids)"; done
DUR=$(for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -hE '^[0-9.]+s (call|setup|teardown) ' $u.log; done | sort -rn)
echo "--- slowest 15 (of $(echo "$DUR" | awk '$1+0 > 60' | wc -l) over 60 s)"; echo "$DUR" | head -15 | cut -c1-200
echo "--- failures"; for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -E '^(FAILED|ERROR) ' $u.log; done | cut -c1-260
echo "--- skips"; for u in $(cut -d' ' -f1 queue.txt); do [ -f $u.log ] && grep -E '^SKIPPED ' $u.log; done | cut -c1-260
