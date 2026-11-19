#!/usr/bin/env bash
# POD SIDE. Usage: fetch.sh [--pack]. Writes agg.txt, packs this run's logs (not the repo) into
# results.tgz, starts a detached send and prints the transfer code. On the PC:
# tools/ops/day/collect.sh <run> <code>. --pack stops after results.tgz (notify.sh sends it).
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
bash bin/agg.sh > agg.txt 2>&1
tar -czf results.tgz agg.txt out col collected.txt setup.log runner.log notify.log meta.env files.txt 2>/dev/null
[ "${1:-}" = --pack ] && exit 0
rm -f fetch_send.log
nohup /work/_bin/runpodctl send results.tgz > fetch_send.log 2>&1 < /dev/null &
for i in $(seq 1 40); do grep -q "code is" fetch_send.log && break; sleep 0.5; done
grep -o "code is: [^ ]*" fetch_send.log | head -1
