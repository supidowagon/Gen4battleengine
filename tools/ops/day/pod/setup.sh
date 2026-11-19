#!/usr/bin/env bash
# POD SIDE. Run from /work/<run> after payload.tar is unpacked there (code.tar, tables.tar.xz,
# meta.env, files.txt, bin/). Builds <run>/repo from the commit, unpacks the built tables, links
# the decomp, syncs the env, collects every entry of files.txt, cuts units (one test file per
# unit, at most <ids per chunk> ids each; an id already queued by an earlier line is skipped)
# and starts bin/runner.sh detached. Writes its PID to setup.pid and the runner's to runner.pid
# (status.sh counts a run as active only while one of them is alive); on any failure the log
# ends with a SETUP_FAILED line. Collection runs at most 4 jobs at once, each started only
# while container memory is under GATE_GB (default 88, the runner's gate).
# files.txt lines: "<test file, folder or node id> <ids per chunk> [A]"; A = no -m filter.
set -euxo pipefail
export PATH="/root/.local/bin:/work/_bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"; . ./meta.env; MARK="${MARK:-not slow}"
echo $$ > "$D/setup.pid"; SETUP_DONE=0
trap 'rc=$?; [ "$SETUP_DONE" = 1 ] || echo "SETUP_FAILED rc=$rc $(date +%s)"' EXIT
echo "SETUP_START $(date +%s)"
# The decomp worktree is shared by every run on the pod: one setup at a time creates it.
( flock 9
  [ -d "/work/_decomp/$DECOMP" ] || { git -C /work/_decomp/repo fetch -q origin || true; git -C /work/_decomp/repo worktree add -q --detach "/work/_decomp/$DECOMP" "$DECOMP"; }
) 9> /work/_decomp/.lock
R="$D/repo"; rm -rf "$R"; mkdir "$R"; tar --no-same-owner -xf code.tar -C "$R"
( cd "$R" && git init -q && git add -A && git -c user.name=ci -c user.email=ci@localhost commit -qm "snapshot $COMMIT" )
tar --no-same-owner -xJf tables.tar.xz -C "$R"; [ -f "$R/data/gen4_platinum_kaizo/manifest.json" ]
mkdir -p "$R/external"; ln -sfn "/work/_decomp/$DECOMP" "$R/external/pokeplatinum"
( cd "$R" && uv sync --frozen -q )
# The damage oracle needs node + @smogon/calc; node is installed once per pod (DAYBOX.md).
if command -v npm >/dev/null && [ -f "$R/tools/smogon_calc/package-lock.json" ]; then ( cd "$R/tools/smogon_calc" && npm ci -s ); fi
rm -rf col out; mkdir -p col out
grep -vE '^[[:space:]]*(#|$)' files.txt > col/list.txt   # the one list both steps below number
awk '{print NR "|" $1 "|" ($3 == "A" ? "A" : "N")}' col/list.txt > col/jobs.txt
collect_one() { IFS='|' read -r n f mk <<< "$1"; cd "$R"; ma=(-m "$MARK"); [ "$mk" = A ] && ma=()
  while [ $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) -ge "$GATE_GB" ]; do sleep 10; done
  uv run --frozen pytest --collect-only -q "${ma[@]}" -p no:cacheprovider "$f" > "$D/col/$n.out" 2>&1
  echo $? > "$D/col/$n.rc"; grep '::' "$D/col/$n.out" > "$D/col/$n.ids" || true; }
export -f collect_one; export D R MARK GATE_GB="${GATE_GB:-88}"
xargs -d '\n' -P 4-I{} bash -c 'collect_one "$@"' _ {} < col/jobs.txt
(cd "$R" && uv run --frozen python - "$D") <<'PY'
import sys, pathlib, collections
D = pathlib.Path(sys.argv[1]); q = []; summ = []; n = 0; seen = set()
for k, line in enumerate((D / "col" / "list.txt").read_text().splitlines(), 1):
    p = line.split(); f, chunk = p[0], int(p[1]); mk = "A" if len(p) > 2 and p[2] == "A" else "N"
    rc = (D / "col" / f"{k}.rc").read_text().strip()
    by = collections.OrderedDict()
    for i in (D / "col" / f"{k}.ids").read_text().splitlines():
        if i.strip() and i not in seen: seen.add(i); by.setdefault(i.split("::")[0], []).append(i)
    if not by: summ.append(f"{f} 0 collect_rc={rc}")
    for tf, ids in by.items():
        summ.append(f"{tf} {len(ids)} collect_rc={rc}")
        for s in range(0, len(ids), chunk):
            n += 1; u = f"u_{n:03d}"
            (D / "out" / f"{u}.ids").write_text("\n".join(ids[s:s + chunk]) + "\n")
            (D / "out" / f"{u}.m").write_text(mk + "\n")
            q.append(f"{u} {tf}")
(D / "out" / "queue.txt").write_text("\n".join(q) + "\n")
(D / "collected.txt").write_text("\n".join(summ) + "\n")
print("units", n, "ids", len(seen))
PY
cat collected.txt
echo "SETUP_END $(date +%s)"
nohup bash "$D/bin/runner.sh" > "$D/runner.log" 2>&1 < /dev/null &
echo $! > "$D/runner.pid"; SETUP_DONE=1
echo RUNNER_STARTED
