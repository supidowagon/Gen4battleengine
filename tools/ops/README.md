# tools/ops

Operations scripts for the work streams and the rented test pods. How they are used: `docs/process.md`.

Environment variables (optional): `NUZLINE_REPO` (default: the main checkout, found from `git rev-parse --git-common-dir`, so the scripts run from any worktree) and `NUZLINE_SCRATCH` (default: `<main checkout>/_scratch`, git-ignored).

## Watching the streams (PowerShell)

- `watchdog.ps1`: per stream, last-commit age and uncommitted files; big Python processes; free RAM; the fast-tier lock (`$NUZLINE_SCRATCH/fast_tier.lock`). Flags: a stream without a commit for 120 min, a process over 12 GB or 150 min, under 3 GB free RAM, a dead or stale lock, and no process plus no commit for 90 min on every stream. `-FixLock` removes a lock whose owner is dead. `-Streams <path>` overrides the stream list.
- `watch_flags.ps1`: runs the watchdog every `-EveryMin` minutes (default 10) and exits only on a flag for the managing session, or when the lock goes away. Flags that a progressing fast tier explains (lock held and an `ft_*.log` written in the last 150 min) are ignored. `-HoldUntil <time>` silences "has not committed" and "may have stopped" until then.

Stream list: `$NUZLINE_SCRATCH/manager/streams.json`, edited by hand at every stream branch or worktree change. `worktree` and `logs` are relative to the repo; `logs` (optional) is the folder that receives the stream's local `ft_*.log` files.

```json
[
  { "name": "E1", "branch": "claude/e1-example", "worktree": "_scratch/e1/wt", "logs": "_scratch/e1/logs" },
  { "name": "E2", "branch": "e2-example", "worktree": ".claude/worktrees/e2-example" },
  { "name": "E3", "branch": "claude/e3-example", "worktree": ".claude/worktrees/e3-example" }
]
```

## Test pod and nightly (Git Bash on the PC, bash on the pod)

Steps: `day/DAYBOX.md`. Pod shape, limits and chunk sizes: `docs/process.md` section 5.

- `common.sh`: sourced by the PC-side scripts; main checkout and scratch paths, the table upload rules (`TABLE_ALLOW`, `TABLE_DENY`), run-name check.
- `build_tables.sh --here | <commit>`: builds the vanilla and Kaizo tables in place, or for a commit in a temporary worktree packed into `_scratch/rent/tables_<short>.tar.xz` (only `TABLE_ALLOW` files). Local inputs come from the main checkout. `NUZLINE_PY` selects an existing venv's python; default `uv run --group rom python`.
- `nightly_brief.md`: the brief for the nightly run.
- `day/push_bin.sh`: once per pod (safe to repeat); sends `day/bootstrap.sh` and prints the exec-pod body that runs it in `/work/_bin/new`.
- `day/bootstrap.sh`: pod side, once per pod; apt tools, uv, Python 3.12, runpodctl checksum, decomp clone; ends with `BOOTSTRAP_OK`.
- `day/pack.sh <commit> <tables> <run> <files.txt>`: bundles the `git archive`, the tables, the test list and `day/pod/*.sh`; refuses forbidden files and tables outside `TABLE_ALLOW`; sends; prints the exec-pod body.
- `day/collect.sh <run> <code>`: receives a run's `results.tgz` and unpacks it.
- `day/nightly_files.txt`: the nightly test list.
- `day/pod/setup.sh`: checkout, tables, env, collection, chunked units; starts the runner.
- `day/pod/runner.sh`: starts units under the pod-wide memory and process gate and the per-run cap `RUNMAXP`; kills and requeues the newest unit near the limit; records crashed units.
- `day/pod/unit.sh`: one pytest process over one unit's ids.
- `day/pod/status.sh`: progress, memory, collection failures, failing units, other active runs.
- `day/pod/notify.sh`: started by `setup.sh`; offers the first-failure beacon and the results under the run's `WATCH_CODE`.
- `day/wait.sh`: PC side; blocks until the run is over (or its first failure), then collects the results.
- `day/pod/agg.sh`: per-file totals and completeness, collection failures, crashes, peaks, slowest tests, failures, skips, RESULT line.
- `day/pod/fetch.sh`: packs a run's logs and prints the transfer code.
