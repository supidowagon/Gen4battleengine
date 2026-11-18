# tools/ops

Operations scripts for the multi-stream workflow and the nightly fast tier on a rented CPU pod.

- `watchdog.ps1`: reports stream commit age, big Python processes, free RAM and the fast-tier lock; `-FixLock` removes a dead lock.
- `watch_flags.ps1`: loops the watchdog and exits only when something needs the manager.
- `nightly_brief.md`: the brief for the agent that runs the nightly on the pod.
- `build_tables.sh`: builds the vanilla and Kaizo tables in the scratch worktree.
- `split_parts.py`: splits the fast-tier tests into N time-balanced parts.
- `analyze.py`: merges per-part pytest outcomes and lists failures and skips.
- `pod/pod_setup_and_run.sh`: sets up the repo and tables on the pod and starts the parallel run.
- `pod/run_parts.sh`: runs the test parts in parallel and records time and memory.
- `pod/progress.sh`: prints per-part progress and memory on the pod.

Environment variables:

- `NUZLINE_REPO`: path of the repo checkout (PowerShell default: two folders above the script).
- `NUZLINE_SCRATCH`: path of the scratch folder (PowerShell default: `_scratch` next to the repo).
- `NUZLINE_REPO_URL`: git URL the pod clones (used by `pod/pod_setup_and_run.sh`).

See docs/process.md for how these are used.

Daytime test pod (`day/`): a rented CPU pod that runs merge test runs while the PC keeps working; at most $10 per day, and only source code at a commit plus built tables leave the PC.

- `day/DAYBOX.md`: the recipe, rules and caps.
- `day/daybox.sh`: PC side, packs a commit and built tables and prints the pod commands.
- `day/push_bin.sh`: PC side, morning bring-up, sends the pod scripts.
- `day/pod/bootstrap.sh`, `start_run.sh`, `inner_run.sh`, `status.sh`, `freemem.sh`, `fetch.sh`: pod side, installs tools, starts a run behind a memory guard, reports state, returns logs.
