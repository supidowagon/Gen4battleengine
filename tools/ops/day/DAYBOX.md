# DAYBOX: chunked pytest runs on the test pod

Rules (pod shape, limits, what may leave the PC, chunk sizes, memory limits, completeness, exec-pod limits, dead pods, never signalling PID 1): `docs/process.md` section 5. This file holds the steps.

Each run, day or nightly, is handled start to finish by one fresh Opus agent: it uploads, starts the run, polls `status.sh`, reports each failure as it appears, collects and reports the final result.

Paths: the PC-side scripts run from any checkout or worktree; `tools/ops/common.sh` resolves the main checkout from `git rev-parse --git-common-dir` (override with `NUZLINE_REPO`) and the scratch folder `<main checkout>/_scratch` (override with `NUZLINE_SCRATCH`). PC-side state lives in `_scratch/rent/`: `bin/runpodctl-windows-amd64.exe` (override with `RUNPODCTL`; download it by hand from the runpodctl GitHub releases page, the same version as the pod's, v3.1.1, into `$NUZLINE_SCRATCH/rent/bin/`), `home/` (runpodctl config), `POD_ID` (the current pod id; never commit it), `tables_<short>.tar.xz`, `out/<run>/`. Pod side: `/work/_bin` (runpodctl, bootstrap), `/work/<run>` (one folder per run, its scripts in `/work/<run>/bin`). Run names: `[a-z0-9_-]`, a stream prefix plus a label (`e2-m26`, `nightly-20261118-235f2f8`), each used once.

## Bring-up (once per pod, about 3 minutes)

1. MCP `list-cpu-types` (include availability, product POD, vcpuCount 32): `cpu3g` at about $1.28/h with a data center in stock.
2. MCP `create-pod` with `<name>` = `ci-day-big` (test pod) or `ci-nightly` (own nightly pod): `{"name":"<name>","cloud":"SECURE","cpu":{"id":"cpu3g","vcpuCount":32},"dataCenterIds":["<in stock>"],"image":"ubuntu:24.04","args":"{\"entrypoint\":[\"/bin/bash\",\"-c\"],\"cmd\":[\"sleep infinity\"]}","disk":60}`. Write the id into `_scratch/rent/POD_ID`.
3. exec-pod, retrying on 409 while the container starts: `{"command":["sh","-c","mkdir -p /work/_bin && cd /work/_bin && nohup sh -c 'apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && curl -LsSf -o runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 && echo \"ae0a2e5ff723c8fceb609e7120dfcaa4d681bf3015adc74141812a21f2bb9486  runpodctl\" | sha256sum -c - && chmod +x runpodctl && echo RPC_OK' > /work/rpc.log 2>&1 < /dev/null & sleep 20; tail -1 /work/rpc.log"],"timeoutSeconds":25}`. Poll `tail -1 /work/rpc.log` until `RPC_OK`; a checksum mismatch stops here.
4. PC: `bash tools/ops/day/push_bin.sh`, then send its exec-pod body. `bootstrap.sh` installs the apt tools, uv and Python 3.12, checks the runpodctl checksum again and clones the decomp.
5. Ready when exec `{"command":["tail","-1","/work/bootstrap.log"],"timeoutSeconds":25}` prints `BOOTSTRAP_OK`.
6. Only if a list holds `tests/test_damage_oracle.py`: exec-pod `{"command":["sh","-c","DEBIAN_FRONTEND=noninteractive nohup sh -c 'apt-get install -y -qq nodejs npm && echo NODE_OK' > /work/node_apt.log 2>&1 < /dev/null &"],"timeoutSeconds":25}`. Poll `tail -1 /work/node_apt.log` until `NODE_OK` before sending a run that needs node; `setup.sh` runs `npm ci` in `tools/smogon_calc` only when `npm` is present.

## Send a run (PC, Git Bash)

1. Tables. Reuse `_scratch/rent/tables_<A>.tar.xz` for commit `<B>` only when `git diff --stat <A> <B> -- src tools data pyproject.toml uv.lock` prints nothing. Otherwise `bash tools/ops/build_tables.sh <B>`.
2. Test list, one line per entry, slowest first: `<test file, folder or node id> <ids per chunk> [A]`. `A` drops the `-m` filter (slow tests included). A folder is cut per test file; an id already queued by an earlier line is skipped, so a last line `tests 40` queues the rest. Nightly list: `tools/ops/day/nightly_files.txt`.
3. `bash tools/ops/day/pack.sh <commit> <tables.tar.xz> <run> <files.txt>`. It takes the pod scripts from git at the checkout's HEAD (refused while `tools/ops/day/pod` has uncommitted changes), checks every list line against the format of step 2, lists the payload, refuses forbidden files and any table entry outside `TABLE_ALLOW` or inside `TABLE_DENY` (`tools/ops/common.sh`), starts the send detached and prints the exec-pod body.
4. Send the printed body with exec-pod at once (the PC send waits for this receive). It receives into `/work/<run>`, runs `bin/setup.sh` (checkout, tables, env, collect, cut units) and starts `bin/runner.sh`. A failed setup ends `setup.log` with `SETUP_FAILED`.

Runner knobs, as `"env":{...}` in that exec-pod body: `GATE_GB` (default 88), `MAXP` (chunks at once across all runs on the pod, default 10), `SPACING` (s between starts, default 15), `KILL_GB` (default 118). Leave them at their defaults unless the managing session says otherwise.

## Status

`{"command":["bash","/work/<run>/bin/status.sh","20"],"timeoutSeconds":25}`: setup state, collection failures, other runs still active on the pod, units done/running/pending/killed/crashed, memory, running units with F/E marks, every failing unit (`BAD u_NNN`) with its FAILED/ERROR lines. The run is finished when it prints `DONE=y`; `runner=gone` without `DONE=y` means the runner died. Another run counts as active only while its setup or runner process is alive. A chunk killed at the memory limit is requeued at most twice; a third kill gives rc 98 and counts as a failure.

## Collect

1. `{"command":["bash","/work/<run>/bin/agg.sh"],"timeoutSeconds":25}`: per-file totals with the completeness column, collection failures, crashed chunks, wall time, container peak, largest process, the 15 slowest tests, the count over 60 s, every failure and skip, and `RESULT: PASS` or `RESULT: FAIL (<reasons>)`.
2. `{"command":["bash","/work/<run>/bin/fetch.sh"],"timeoutSeconds":25}` prints a code. PC: `bash tools/ops/day/collect.sh <run> <code>` unpacks into `_scratch/rent/out/<run>/results/`.
3. Tests that need the ROM, saves, `.cache`, the PKCalc zip, the docs sheet or their extractions skip on the pod. A skip there is not a pass: run those tests locally, one process at a time (`CLAUDE.md`, "Where tests run").

## Delete

When nothing is queued and `status.sh` of every run shows no other active run: collect every run, MCP `delete-pod`, confirm with `list-pods`, clear `POD_ID`, log the hours and cost. Never stop the pod instead: a stopped CPU pod loses `/work` and saves nothing.

## Never upload

Only the `git archive` of a commit and the tables `TABLE_ALLOW` lists travel. Never the ROM (`*.nds`), saves (`*.sav`, `*.dsv`), `.cache/`, the PKCalc zip or its extraction (`data/intermediate_kaizo/pkcalc/`), the docs sheet or its extraction (`data/intermediate_kaizo/docs/`), `data/local.toml`, any file from the Desktop, anything in a `backup` folder. Never add files to a payload by hand. Never push from the pod.
