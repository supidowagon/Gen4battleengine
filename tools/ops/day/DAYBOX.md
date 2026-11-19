# DAYBOX: chunked pytest runs on the test pod

Pod shape, daily limit, when to keep or delete it, chunk sizes and memory start gate: `docs/process.md` section 5. This file holds the steps.

Paths: scripts take `NUZLINE_REPO` (default: this checkout) and `NUZLINE_SCRATCH` (default: `<repo>/_scratch`). PC-side state lives in `$NUZLINE_SCRATCH/rent/`: `bin/runpodctl-windows-amd64.exe` (override with `RUNPODCTL`), `home/` (runpodctl config), `POD_ID` (the current pod id; never commit it), `tables_<short>.tar.xz`, `out/<run>/`. Pod side: `/work/_bin` (runpodctl, once per pod), `/work/<run>` (one folder per run, scripts in `/work/<run>/bin`).

## Exec-pod rules

- An exec-pod call returns after at most 25 s. Start everything longer with `nohup ... &` and poll; never wait inside a call for more than about 20 s.
- Never signal PID 1 (`kill 1`, `kill -- -1`, `pkill` without a pattern): it ends the container and every run on it.
- A pod whose container has not started within 5 minutes, or whose exec-pod calls fail for 5 minutes, is a dead host: delete it, confirm with `list-pods`, create a new one.

## Bring-up (once per pod, about 3 minutes)

1. MCP `list-cpu-types` (include availability, product POD, vcpuCount 32): `cpu3g` at about $1.28/h with a data center in stock.
2. MCP `create-pod`: `{"name":"ci-day-big","cloud":"SECURE","cpu":{"id":"cpu3g","vcpuCount":32},"dataCenterIds":["<in stock>"],"image":"ubuntu:24.04","args":"{\"entrypoint\":[\"/bin/bash\",\"-c\"],\"cmd\":[\"sleep infinity\"]}","disk":60}`. Write the id into `$NUZLINE_SCRATCH/rent/POD_ID`.
3. exec-pod, retrying on 409 while the container starts: `{"command":["sh","-c","mkdir -p /work/_bin && cd /work/_bin && nohup sh -c 'apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && curl -LsSf -o runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 && chmod +x runpodctl && echo RPC_OK' > /work/rpc.log 2>&1 < /dev/null & sleep 20; tail -1 /work/rpc.log"],"timeoutSeconds":25}`. Poll `tail -1 /work/rpc.log` until `RPC_OK`.
4. PC: `bash tools/ops/day/push_bin.sh`, then send its exec-pod body. `bootstrap.sh` installs the apt tools, uv, Python 3.12, checks the runpodctl checksum and clones the decomp.
5. Ready when `tail -1 /work/bootstrap.log` prints `BOOTSTRAP_OK`.
6. Only if a list holds `tests/test_damage_oracle.py`: exec-pod `{"command":["sh","-c","DEBIAN_FRONTEND=noninteractive nohup apt-get install -y -qq nodejs npm > /work/node_apt.log 2>&1 < /dev/null &"],"timeoutSeconds":25}`. `setup.sh` then runs `npm ci` in `tools/smogon_calc`.

## Send a run (PC, Git Bash)

1. Tables. If `git diff --stat <commit with a bundle> <commit> -- tools data` is empty, reuse that bundle. Otherwise `bash tools/ops/build_tables.sh <commit>` writes `$NUZLINE_SCRATCH/rent/tables_<short>.tar.xz`.
2. Test list, one line per entry, slowest first: `<test file, folder or node id> <ids per chunk> [A]`. `A` drops the `-m` filter (slow tests included). Chunk sizes: `docs/process.md` section 5. A folder is cut per test file; an id already queued by an earlier line is skipped, so a last line `tests 40` queues the rest. Nightly list: `tools/ops/day/nightly_files.txt`.
3. `bash tools/ops/day/pack.sh <commit> <tables.tar.xz> <run> <files.txt>`. It lists the payload, refuses forbidden files, starts the send detached and prints the exec-pod body. Run names: a stream prefix and a label (`e2/m26`, `nightly-235f2f8`), each used once.
4. Send the printed body with exec-pod at once (the PC send waits for this receive). It receives into `/work/<run>`, runs `bin/setup.sh` (checkout, tables, env, collect, cut units) and starts `bin/runner.sh`.

Runner knobs, as `"env":{...}` in that exec-pod body: `GATE_GB` (start gate, default 88), `MAXP` (this run's units at once, default 10), `SPACING` (s between starts, default 15), `KILL_GB` (default 118: kill this run's newest unit and requeue it at the front). Several runs on one pod share one memory counter; each run's gate reads the whole container.

## Status (a later runner)

- `{"command":["bash","/work/<run>/bin/status.sh","20"],"timeoutSeconds":25}`: setup state, units done/running/pending/killed, memory, running units with F/E marks, every failing unit (`BAD u_NNN`) with its FAILED/ERROR lines.
- All runs: `{"command":["sh","-c","ls /work; cat /sys/fs/cgroup/memory.current"],"timeoutSeconds":25}`. Read memory from `/sys/fs/cgroup/memory.current`, never from `free` (it shows the host).

## Collect

1. `{"command":["bash","/work/<run>/bin/agg.sh"],"timeoutSeconds":25}`: per file collected, passed, failed, skipped, errors, not run, largest process, summed wall; then run wall, container peak, the 15 slowest tests, the count over 60 s, every failure and skip. A file whose passed + failed + skipped + errors differs from collected has not run.
2. `{"command":["bash","/work/<run>/bin/fetch.sh"],"timeoutSeconds":25}` prints a code. PC: `bash tools/ops/day/collect.sh <run> <code>` unpacks into `$NUZLINE_SCRATCH/rent/out/<run>/results/`.
3. Tests that need the ROM, saves, `.cache`, the PKCalc zip or the docs sheet skip on the pod. A skip there is not a pass: run those tests locally, one process at a time (`CLAUDE.md`, "Where tests run").

## Delete

When nothing is queued: collect every run, MCP `delete-pod`, confirm with `list-pods`, clear `POD_ID`, log the hours and cost. Never stop the pod instead: a stopped CPU pod loses `/work` and saves nothing.

## Never upload

Only the `git archive` of a commit and built `.json`/`.npz` tables travel. Never the ROM (`*.nds`), saves (`*.sav`, `*.dsv`), `.cache/`, the PKCalc zip, the docs sheet, `data/local.toml`, any file from the Desktop, anything in a `backup` folder. Never add files to a payload by hand. Never push from the pod.
