# DAYBOX: remote pytest runs on the rented pod

A daytime CPU pod (name `ci-day`, ubuntu:24.04, 16 vCPU / 64 GB, about $0.64/h) runs merge test runs while the PC does other work. The pod id is kept only in `$NUZLINE_SCRATCH/rent/day/POD_ID` (never committed). Cap: at most $10 per day; the manager creates and deletes the pod.

Variables: `NUZLINE_REPO` (repo checkout), `NUZLINE_SCRATCH` (scratch folder; the helpers use `$NUZLINE_SCRATCH/rent/` for `bin/runpodctl-windows-amd64.exe`, `home/` and `day/`), `NUZLINE_REPO_URL` (git URL, only for runs that need a clone).

## One remote run (Git Bash, then Runpod MCP `exec-pod`)
1. Rebuild the vanilla + Kaizo tables in your worktree (as for a local run). The helper refuses if `data/gen4_platinum_kaizo/manifest.json` or `data/gen4_platinum/tables.npz` is missing.
2. PC: `bash "$NUZLINE_REPO/tools/ops/day/daybox.sh" "<worktree>" <commit> <run-name> -- <pytest args...>`
   e.g. `... "<worktree>" <commit> e1c-merge1 -- tests/test_cli.py -m "not slow" -q`
   Run names are one-shot and must be unique per stream (prefix with your stream, e.g. `e1-...`, `e2-...`).
3. It prints a transfer code and the exact exec-pod body. Call exec-pod (pod id from `POD_ID`) with
   `{"command": ["bash", "/work/_bin/start_run.sh", "<run-name>", "<code>"], "timeoutSeconds": 10}`
   Do this within a few minutes: the PC-side `runpodctl send` waits for this receive.
4. Poll: `{"command": ["bash", "/work/_bin/status.sh", "<run-name>", "30"]}` (state, wall s, rc, peak RSS, log tail).
   All runs + free memory: `{"command": ["bash", "/work/_bin/status.sh"]}`. Use `sleep 20;` inside `sh -c` to wait in one call; never idle-poll.
5. Optional full log: exec-pod `{"command": ["bash", "/work/_bin/fetch.sh", "<run-name>"]}` prints a code; on the PC the helper's printed `receive` line fetches `result_<run>.tgz` into `day/out/<run>/`.

## Reading results
`/work/<run>/STATE`: LAUNCHING, RECEIVING, DECOMP, SYNCING, WAITING (another run starting), QUEUED (memory guard), RUNNING, `DONE rc=N`, or `FAILED: ...` (rc 99 = launch failed; read `launch.log`).
`rc` = pytest exit code; `time.txt` = GNU `time -v` (Maximum resident set size = largest single process, KB); `pytest.log` = full output.
Tests needing the ROM, saves, `.cache`, the PKCalc zip or the docs sheet skip on the pod: expected, not a pass of those tests.

## Memory guard
Before pytest starts, `inner_run.sh` holds a lock, reads free container memory (`freemem.sh`: cgroup max minus non-cache use; `free` shows the host's memory, ignore it) and waits (state QUEUED) until >= 12 GB is free. It holds the lock 90 s after starting so the next run sees this run's growth. Runs queue, they are not killed.

## Never upload
Only the commit (`git archive`) and built tables (git-ignored or modified `data/**.json|npz`) travel. Never the ROM (`*.nds`), `.cache/`, saves, the PKCalc zip, the docs sheet, anything from the Desktop, anything in a `backup` folder. Do not add files to the payload by hand. Nothing on the pod is pushed anywhere.

## Morning bring-up (about 3 min of pod time)
1. MCP `list-cpu-types` (include AVAILABILITY, product POD, vcpuCount 16): cpu3g at 0.04/vCPU = $0.64/h with a US data center in stock. Above $0.70/h: pick a smaller shape.
2. MCP `create-pod` body: `{"name":"ci-day","cloud":"SECURE","cpu":{"id":"cpu3g","vcpuCount":16},"dataCenterIds":["US-CA-2"],"image":"ubuntu:24.04","args":"{\"entrypoint\":[\"/bin/bash\",\"-c\"],\"cmd\":[\"sleep infinity\"]}","disk":60}`. Write the new id into `$NUZLINE_SCRATCH/rent/day/POD_ID`.
3. exec-pod (retry on 409 while starting): `{"command":["sh","-c","mkdir -p /work/_bin && apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && curl -LsSf -o /work/_bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 && chmod +x /work/_bin/runpodctl"],"timeoutSeconds":25}` (if it times out, re-run; it continues in the background).
4. PC: `bash "$NUZLINE_REPO/tools/ops/day/push_bin.sh"`, then run the exec-pod body it prints (installs the scripts and starts `bootstrap.sh`: apt tools, uv, Python 3.12, runpodctl checksum, decomp clone).
5. Wait for `BOOTSTRAP_OK` at the end of `/work/bootstrap.log` (~2 min). Ready for runs.

## Stop overnight instead of delete?
No gain. The container disk is cleared when a pod stops and CPU pods cannot have a persistent disk, so `/work` (env, decomp, runs) is lost on stop; a stopped pod bills nothing. Delete it at the end of the day and re-bring-up (~3 min).
