#!/usr/bin/env bash
# POD SIDE. Run from /work/<run> after payload.tar is unpacked there (code.tar, tables.tar.xz,
# meta.env, files.txt, bin/). Builds <run>/repo from the commit, unpacks the built tables, links
# the decomp, syncs the env, collects every entry of files.txt, cuts units (one test file per
# unit, at most <ids per chunk> ids each; an id already queued by an earlier line is skipped)
# and starts bin/runner.sh detached. Writes its PID to setup.pid and the runner's to runner.pid
# (status.sh counts a run as active only while one of them is alive). A failure after the trap
# below is set ends the log with SETUP_FAILED; one before it (meta.env, a kill) leaves none, so
# status.sh's setup=gone with runner=none is the failure test. bin/notify.sh is started by the
# exec-pod body before this script. Collection runs at most COLLECTP (default 4) jobs at once, each
# started only while container memory is under GATE_GB (default 88, the runner's gate).
# files.txt lines: "<test file, folder or node id> <ids per chunk> [A] [<GB>G]"; A = no -m filter;
# <GB>G = a floor on the expected peak of every unit cut from that line. Each unit's expected peak
# goes to out/<unit>.w: max(<GB>, bin/weights.py over bin/weights.txt); runner.sh gates on it.
# Writes gate.env (GATE_VERSION, POD_SCRIPTS, POD_WEIGHTS for podcheck.sh) and run.env (the
# knobs the exec-pod body's env set, for a runner restart) first.
set -euxo pipefail
export PATH="/root/.local/bin:/work/_bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
# meta.env holds the beacon codes: sourced with the trace off, so setup.log never shows them.
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"; set +x; . ./meta.env; unset WATCH_F WATCH_D; set -x
printf 'GATE_VERSION=%s\nPOD_SCRIPTS=%s\nPOD_WEIGHTS=%s\n' "${GATE_VERSION:-}" "${POD_SCRIPTS:-}" "${POD_WEIGHTS:-}" > "$D/gate.env"   # podcheck.sh reads it
# run.env: every knob the exec-pod body's env set (RUN_ENV_NAMES; never the beacon codes, which
# live in meta.env only), as "export NAME=<value>" lines, so a runner restart repeats them
# (DAYBOX.md, Housekeeping, "Restart a runner"). A knob left unset is not written. Values are in
# POSIX single quotes (a ' as '\''), so sh (dash, the restart body's shell) and bash read them alike.
RUN_ENV_NAMES="GATE_GB MAXP RUNMAXP SPACING KILL_GB DEFAULT_WEIGHT_GB IDLE_WAIT_S RUNNER_LOCK CLAIM_S CLAIMS CLAIM_MAX_S UNIT_MAX_S RESERVE_TOP RESERVE_FULL_GB WORK DUR COLLECTP WATCH_HOURS WARMUP WARMUP_S NUZLINE_TEST_COMPILE_CACHE NUZLINE_HYPOTHESIS_PROFILE NUZLINE_REQUIRE_NODE CGROOT"
for n in $RUN_ENV_NAMES; do if [ -n "${!n+x}" ]; then
  printf "export %s='%s'\n" "$n" "$(printf '%s' "${!n}" | sed "s/'/'\\\\''/g")"; fi; done > "$D/run.env"
. "$D/bin/gate.sh"   # cgmem: container memory, cgroup v2 or v1
MARK="${MARK:-not slow and not fullset}"   # pack.sh writes MARK into meta.env; this default is the same
echo $$ > "$D/setup.pid"; SETUP_DONE=0
trap 'rc=$?; [ "$SETUP_DONE" = 1 ] || echo "SETUP_FAILED rc=$rc $(date +%s)"' EXIT
echo "SETUP_START $(date +%s)"
cgmem current > /dev/null || { echo "ERROR: no cgroup memory file (v2 memory.current or v1 memory/memory.usage_in_bytes)"; exit 1; }
# The decomp worktree is shared by every run on the pod: one setup at a time creates it.
( flock -w 600 9 || { echo "ERROR: decomp lock /work/_decomp/.lock still held after 600 s"; exit 1; }
  if [ ! -d "/work/_decomp/$DECOMP" ]; then
    timeout 300 git -C /work/_decomp/repo fetch -q origin \
      || echo "WARNING: decomp fetch failed or passed 300 s (rc $?); trying the local clone"
    git -C /work/_decomp/repo worktree add -q --detach "/work/_decomp/$DECOMP" "$DECOMP" \
      || { echo "ERROR: decomp commit $DECOMP not in the local clone"; exit 1; }
  fi
) 9> /work/_decomp/.lock
R="$D/repo"; rm -rf "$R"; mkdir "$R"; tar --no-same-owner -xf code.tar -C "$R"
( cd "$R" && git init -q && git add -A && git -c user.name=ci -c user.email=ci@localhost commit -qm "snapshot $COMMIT" )
tar --no-same-owner -xJf tables.tar.xz -C "$R"; [ -f "$R/data/gen4_platinum_kaizo/manifest.json" ]
mkdir -p "$R/external"; ln -sfn "/work/_decomp/$DECOMP" "$R/external/pokeplatinum"
( cd "$R" && uv sync --frozen -q )
# The damage oracle needs node + @smogon/calc; node is installed once per pod (DAYBOX.md bring-up
# step 6). With NUZLINE_REQUIRE_NODE=1 (the nightly) a pod without npm fails the setup.
if command -v npm >/dev/null && [ -f "$R/tools/smogon_calc/package-lock.json" ]; then ( cd "$R" && npm ci -s --prefix tools/smogon_calc )
elif [ "${NUZLINE_REQUIRE_NODE:-}" = 1 ]; then echo "ERROR: npm or tools/smogon_calc/package-lock.json missing"; exit 1; fi
rm -rf col out; mkdir -p col out
grep -vE '^[[:space:]]*(#|$)' files.txt > col/list.txt   # the one list both steps below number
awk '{print NR "|" $1 "|" ($3 == "A" ? "A" : "N")}' col/list.txt > col/jobs.txt
collect_one() { IFS='|' read -r n f mk <<< "$1"; cd "$R"; ma=(-m "$MARK"); [ "$mk" = A ] && ma=()
  while b=$(cgmem current) || return 1; [ $(( b / 1000000000 )) -ge "$GATE_GB" ]; do sleep 10; done
  uv run --frozen pytest --collect-only -q "${ma[@]}" -p no:cacheprovider "$f" > "$D/col/$n.out" 2>&1
  echo $? > "$D/col/$n.rc"; grep '::' "$D/col/$n.out" > "$D/col/$n.ids" || true; }
export -f collect_one cgmem; export D R MARK GATE_GB="${GATE_GB:-88}"
xargs -d '\n' -P "${COLLECTP:-4}" -I{} bash -c 'collect_one "$@"' _ {} < col/jobs.txt
(cd "$R" && uv run --frozen python - "$D") <<'PY'
import sys, pathlib, collections
D = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(D / "bin")); import weights
table = weights.load(D / "bin" / "weights.txt"); q = []; summ = []; n = 0; seen = set()
for k, line in enumerate((D / "col" / "list.txt").read_text().splitlines(), 1):
    f, chunk, mk, line_gb = weights.parse_line(line)
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
            (D / "out" / f"{u}.w").write_text(f"{weights.unit_weight(ids[s:s + chunk], table, line_gb)}\n")
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
