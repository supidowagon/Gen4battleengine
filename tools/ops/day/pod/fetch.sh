#!/usr/bin/env bash
# POD SIDE. Usage: bash /work/_bin/fetch.sh <run name>
# Packs the run's logs (not the repo) and starts `runpodctl send` detached; prints the code.
# On the PC: (cd <dest> && HOME=.. USERPROFILE=.. runpodctl receive <code>)
set -u
D="/work/${1:?run name}"; [ -d "$D" ] || { echo "no run $1"; exit 1; }
cd "$D" && tar -czf "result_$1.tgz" STATE launch.log pytest.log time.txt rc start end args.txt meta.env 2>/dev/null
rm -f send.log
nohup /work/_bin/runpodctl send "result_$1.tgz" > send.log 2>&1 < /dev/null &
for i in $(seq 1 20); do grep -q "code is" send.log 2>/dev/null && break; sleep 0.5; done
grep -o "code is: .*" send.log || { echo "no code yet; cat $D/send.log"; exit 1; }
