# DAYBOX: remote pytest runs on the daytime pod

Pod: name `ci-day`, ubuntu:24.04, 16 vCPU / 64 GB, about $0.64/h. Cap: $10 per day. Only the managing session creates and deletes it. Keep the pod id only in `$NUZLINE_SCRATCH/rent/day/POD_ID`; never commit it.

Variables: `NUZLINE_REPO` (repo checkout), `NUZLINE_SCRATCH` (scratch folder; the helpers use `$NUZLINE_SCRATCH/rent/` for `bin/runpodctl-windows-amd64.exe`, `home/` and `day/`), `NUZLINE_REPO_URL` (git URL, only for runs that need a clone).

## Start one run (Git Bash, then Runpod MCP `exec-pod`)

1. Rebuild the vanilla and Kaizo tables in the worktree. The helper refuses when `data/gen4_platinum_kaizo/manifest.json` or `data/gen4_platinum/tables.npz` is missing.
2. PC: `bash "$NUZLINE_REPO/tools/ops/day/daybox.sh" "<worktree>" <commit> <run-name> -- <pytest args...>`, for example `... "<worktree>" <commit> e1c-merge1 -- tests/test_cli.py -m "not slow" -q`. A run name is used once and starts with the stream's prefix (`e1-...`, `e2-...`).
3. The helper prints a transfer code and the exec-pod body. At once (the PC-side `runpodctl send` waits for this receive), call exec-pod with the id from `POD_ID`: `{"command": ["bash", "/work/_bin/start_run.sh", "<run-name>", "<code>"], "timeoutSeconds": 10}`.
4. Return. Do not wait for the run.

## Check a run (a later runner)

- One run: exec-pod `{"command": ["bash", "/work/_bin/status.sh", "<run-name>", "30"]}` prints state, wall seconds, rc, peak RSS and the log tail.
- All runs and free memory: `{"command": ["bash", "/work/_bin/status.sh"]}`.
- Full log: exec-pod `{"command": ["bash", "/work/_bin/fetch.sh", "<run-name>"]}` prints a code; on the PC run the printed `receive` line, which writes `result_<run>.tgz` into `day/out/<run>/`.

`/work/<run>/STATE` is one of: LAUNCHING, RECEIVING, DECOMP, SYNCING, WAITING (another run is starting), QUEUED (memory guard), RUNNING, `DONE rc=N`, `FAILED: ...` (rc 99 = launch failed; read `launch.log`). `rc` is the pytest exit code. `time.txt` is GNU `time -v` output ("Maximum resident set size" = largest single process, KB). `pytest.log` is the full output. Tests that need the ROM, saves, `.cache`, the PKCalc zip or the docs sheet skip on the pod; a skip there is not a pass.

## Memory guard

Before pytest starts, `inner_run.sh` takes a lock, reads free container memory (`freemem.sh`: cgroup max minus non-cache use; ignore `free`, which shows the host) and waits in state QUEUED until at least 12 GB is free. It keeps the lock 90 s after starting so the next run sees this run's growth. Runs queue; none is killed.

## Never upload

Only the commit (`git archive`) and the built tables (git-ignored or modified `data/**.json|npz`) travel. Never the ROM (`*.nds`), `.cache/`, saves, the PKCalc zip, the docs sheet, anything from the Desktop, anything in a `backup` folder. Never add files to the payload by hand. Never push anything from the pod.

## Morning bring-up (about 3 minutes of pod time)

1. MCP `list-cpu-types` (include AVAILABILITY, product POD, vcpuCount 16): `cpu3g` at $0.04/vCPU = $0.64/h with a US data center in stock. Above $0.70/h, pick a smaller shape.
2. MCP `create-pod` body: `{"name":"ci-day","cloud":"SECURE","cpu":{"id":"cpu3g","vcpuCount":16},"dataCenterIds":["US-CA-2"],"image":"ubuntu:24.04","args":"{\"entrypoint\":[\"/bin/bash\",\"-c\"],\"cmd\":[\"sleep infinity\"]}","disk":60}`. Write the new id into `$NUZLINE_SCRATCH/rent/day/POD_ID`.
3. exec-pod, retrying on 409 while the pod starts: `{"command":["sh","-c","mkdir -p /work/_bin && apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && curl -LsSf -o /work/_bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64 && chmod +x /work/_bin/runpodctl"],"timeoutSeconds":25}`. On a timeout, re-run it; the first call continues in the background.
4. PC: `bash "$NUZLINE_REPO/tools/ops/day/push_bin.sh"`, then run the exec-pod body it prints. It installs the scripts and starts `bootstrap.sh` (apt tools, uv, Python 3.12, runpodctl checksum, decomp clone).
5. The pod is ready when `/work/bootstrap.log` ends with `BOOTSTRAP_OK` (about 2 minutes).

## End of day

Delete the pod; never stop it. A stopped CPU pod loses `/work` (env, decomp, runs) and has no persistent disk, so stopping saves nothing over a delete and a 3-minute bring-up.
