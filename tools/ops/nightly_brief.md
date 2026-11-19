# Nightly fast tier on a rented Runpod CPU pod (brief for an Opus general-purpose agent)

Approval: run the nightly on Runpod, at most $6 per run on its own pod, uploading only the source at the tip and the built game tables. Nothing else is approved.

Project: `<repo>`. Run the fast tier (`-m "not slow"`) on the tip of local `main`. The tip must equal `origin/main`; if not, stop and report. Steps, scripts and the never-upload list: `tools/ops/day/DAYBOX.md`. Test list: `tools/ops/day/nightly_files.txt` (whole files, slowest first, one test file per process, cut into chunks; the last line queues the rest).

## Hard rules

- Never upload the ROM, saves, the PKCalc zip, the docs sheet, `.cache`, any file from the user's Desktop or anything in a `backup` folder. Read `pack.sh`'s listing before sending the exec-pod body.
- Never write to any folder named `backup`. Write scratch only under `<scratch>\rent\`. Do not modify the main checkout or any branch; never push.
- No personal details in pod names, env vars or requests to outside services. Transfer files only with `runpodctl send` / `receive`. Load the skills `runpod:runpod-mcp` and `runpod:runpod-pod-deploy` first.
- **Pod.** If `list-pods` shows the test pod `ci-day-big` running with `BOOTSTRAP_OK`, run on it as run `nightly-<short tip>`: create no pod and delete nothing; the managing session owns that pod. Otherwise bring up one pod named `ci-nightly` per DAYBOX.md: `cpu3g`, 32 vCPU / 128 GB, a data center with HIGH stock. Re-read price and stock with `list-cpu-types` before creating. If `cpu3g` costs more than $1.28/h, use nothing dearer than $1.76/h (`cpu3m`) and shorten the limit so hours x rate stays under $6. Delete `ci-nightly` no later than 4.5 hours after creation (at $1.28/h), collecting logs first; delete it at once on an unrecoverable failure. Stop or delete only the pod this run created. Confirm every deletion with a fresh `list-pods` read.
- Memory: the runner's defaults (DAYBOX.md, runner knobs) hold the pod under its limit; do not raise `MAXP` or `GATE_GB`.

## Phase 1: launch (the scheduled runner)

Build or reuse the tables for the tip, pick or create the pod, `pack.sh <tip> <tables> nightly-<short tip> tools/ops/day/nightly_files.txt`, send the exec-pod body, confirm with `status.sh` that units are running, then return: the pod id, whether this run created it, creation time, hourly rate and (for `ci-nightly`) the deadline for deletion. Do not wait for the run to finish.

## Phase 2: collect (a later runner, dispatched by the managing session)

Poll `status.sh`. When the runner prints DONE, or at the deadline: `agg.sh`, `fetch.sh` and `collect.sh`. Delete `ci-nightly` if this run created it and confirm with `list-pods`. Then, locally, one process at a time and only with more than 8 GB of free RAM, run in the main checkout with `-m "not slow"` the tests that skipped on the pod for missing inputs (mainly `tests/tools/rom`, `tests/kaizo/test_extract_*`, `tests/saves`, `tests/test_user_save.py`).

## Report (short)

Tip hash; pod name, spec, data center, created and deleted times, hours x rate cost (own pod only); setup time vs test time; totals passed/failed/skipped (pod plus local part) and any file whose totals differ from its collected count; every failure with its one-line error and whether it also fails locally; units killed and requeued; peak memory and largest process; the 15 slowest tests; the count of tests over 60 s; problems hit; anything not done or not verified.
