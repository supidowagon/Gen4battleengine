#!/usr/bin/env bash
# POD SIDE, started by start_run.sh. Do not call directly.
set -euo pipefail
RUN="$1"; CODE="$2"; D="/work/$RUN"
export PATH="/root/.local/bin:/work/_bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
state() { echo "$1" > "$D/STATE"; echo "[$(date -u +%H:%M:%S)] $1"; }
fail() { state "FAILED: $1"; echo 99 > "$D/rc"; exit 1; }
trap 'fail "launch step failed (see launch.log)"' ERR
cd "$D"
state RECEIVING
runpodctl receive "$CODE"
[ -f payload.tar ] || fail "payload.tar not received"
tar --no-same-owner -xf payload.tar
. ./meta.env   # COMMIT, DECOMP
mkdir repo
tar --no-same-owner -xf code.tar -C repo
( cd repo && git init -q && git add -A && git -c user.name=ci -c user.email=ci@localhost commit -qm "snapshot $COMMIT" )
tar --no-same-owner -xJf tables.tar.xz -C repo
[ -f repo/data/gen4_platinum_kaizo/manifest.json ] || fail "Kaizo manifest missing after unpack"
# decomp (public pret repo) at the worktree's commit, shared read-only between runs
state DECOMP
exec 8>/work/_decomp.lock; flock 8
if [ ! -d "/work/_decomp/$DECOMP" ]; then
  [ -d /work/_decomp/repo ] || git clone -q https://github.com/pret/pokeplatinum /work/_decomp/repo
  git -C /work/_decomp/repo fetch -q origin || true
  git -C /work/_decomp/repo worktree add -q --detach "/work/_decomp/$DECOMP" "$DECOMP"
fi
flock -u 8
mkdir -p repo/external
ln -s "/work/_decomp/$DECOMP" repo/external/pokeplatinum
state SYNCING
( cd repo && uv sync --frozen -q )
# memory guard: queue until >= 12 GB of the container's limit is free (anon use, not page cache)
state "WAITING: another run is starting (90 s guard hold)"
exec 9>/work/_guard.lock; flock 9
while :; do
  FREE=$(bash /work/_bin/freemem.sh)
  [ "$FREE" -ge 12 ] && break
  state "QUEUED: ${FREE} GB free < 12 GB, waiting"
  sleep 30
done
state "RUNNING (started with ${FREE} GB free)"
mapfile -t ARGS < args.txt
date +%s > start
cd repo
set +e
/usr/bin/time -v -o "$D/time.txt" uv run --frozen pytest "${ARGS[@]}" > "$D/pytest.log" 2>&1 &
PID=$!
trap - ERR
# hold the guard 90 s so a run starting right after this one sees its memory growth
sleep 90; flock -u 9
wait $PID; RC=$?
echo $RC > "$D/rc"; date +%s > "$D/end"
state "DONE rc=$RC"
