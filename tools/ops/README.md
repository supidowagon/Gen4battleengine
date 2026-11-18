# tools/ops

Operations scripts for the work streams and the rented test pods. How they are used: `docs/process.md`.

- `watchdog.ps1`: per stream, last-commit age and uncommitted files; big Python processes; free RAM; the fast-tier lock. `-FixLock` removes a dead lock.
- `watch_flags.ps1`: runs the watchdog every `-EveryMin` minutes (default 10) and exits only on a flag for the managing session; `-HoldUntil` silences the commit-age flags until that time.
- `nightly_brief.md`: the brief for the nightly pod run.
- `build_tables.sh`: builds the vanilla and Kaizo tables in `$NUZLINE_SCRATCH/rent/wt`.
- `split_parts.py`: splits the fast-tier tests into N time-balanced parts.
- `analyze.py`: merges per-part pytest outcomes and lists failures and skips.
- `pod/pod_setup_and_run.sh`: sets up the repo and tables on the nightly pod and starts the parallel run.
- `pod/run_parts.sh`: runs the test parts in parallel and records time and memory.
- `pod/progress.sh`: prints per-part progress and memory on the pod.

Daytime pod (`day/`, recipe and caps in `day/DAYBOX.md`):

- `day/daybox.sh`: PC side; packs a commit and its built tables and prints the pod commands.
- `day/push_bin.sh`: PC side; morning bring-up, sends the pod scripts.
- `day/pod/bootstrap.sh`, `start_run.sh`, `inner_run.sh`, `status.sh`, `freemem.sh`, `fetch.sh`: pod side; install tools, start a run behind the memory guard, report state, return logs.

Environment variables:

- `NUZLINE_REPO`: the repo checkout (PowerShell default: two folders above the script).
- `NUZLINE_SCRATCH`: the scratch folder (PowerShell default: `_scratch` beside the repo).
- `NUZLINE_REPO_URL`: the git URL the pod clones (`pod/pod_setup_and_run.sh`).
