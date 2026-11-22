#!/usr/bin/env bash
# POD SIDE. Usage: fetch.sh [--pack]. Writes agg.txt, packs this run's logs (not the repo) into
# results.tgz (meta.env without the beacon codes; notify/ without its open send stderr), starts a
# detached send and prints the transfer code. On the PC:
# tools/ops/day/collect.sh <run> <code>. --pack stops after results.tgz (notify.sh sends it);
# without it, notify/taken_d and notify/stop are written as well.
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"
[ -f quick.rc ] || bash bin/agg.sh > agg.txt 2>&1   # a quick lane run wrote its own agg.txt
# meta.env goes in without the beacon codes.
grep -v '^WATCH_' meta.env > meta.nocodes.env
tar -czf results.tgz --exclude='notify/*.err' --transform 's,^meta\.nocodes\.env$,meta.env,' agg.txt out col collected.txt setup.log runner.log quick.log quick.pytest.log ids.txt notify.log notify meta.nocodes.env files.txt 2>/dev/null
[ "${1:-}" = --pack ] && exit 0
# Collected by hand: notify.sh stops offering and status.sh stops listing this run as (beacon).
mkdir -p notify && touch notify/taken_d notify/stop
rm -f fetch_send.log
nohup /work/_bin/runpodctl send results.tgz > fetch_send.log 2>&1 < /dev/null &
for i in $(seq 1 40); do grep -q "code is" fetch_send.log && break; sleep 0.5; done
grep -o "code is: [^ ]*" fetch_send.log | head -1
