# Nightly fast tier on a rented Runpod CPU pod (brief for an Opus general-purpose agent)

Approval: run the nightly on Runpod, at most $6 per run, uploading only the built game tables. Nothing else is approved.

Project: `<repo>`. Run the fast tier (`pytest -m "not slow" -q -p no:cacheprovider --durations=40 -rfEs`) on the tip of local `main`. The tip must equal `origin/main`; if not, stop and report. Tools are in `<scratch>\rent\`: `pod\pod_setup_and_run.sh`, `pod\run_parts.sh`, `pod\progress.sh`, `build_tables.sh`, `split_parts.py`, `analyze.py`, and `bin\runpodctl-windows-amd64.exe` (set HOME and USERPROFILE to `<scratch>\rent\home`). Sources of the scripts: `tools/ops/`. The round-robin split into NPARTS parts is untested: read the scripts and smoke-test them before relying on them.

## Hard rules

- Never upload the ROM (`.nds`), any save (`.sav`, `.dsv`), the PKCalc zip, the docs sheet, or any file from the user's Desktop. List the bundle's contents before sending and confirm it holds only built `.json`/`.npz` tables.
- Never write to any folder named `backup`. Write scratch only under `<scratch>\rent\`. Do not modify the main checkout or any branch; never push. Build the tables for the tip in a detached worktree at `<scratch>\rent\wt` (rebuild the Kaizo intermediates too: `extract_rom`, then `build_tables`; a ROM copy may exist only in that worktree's git-ignored `.cache`). Remove the worktree at the end.
- No personal details in pod names, env vars or requests to outside services. Name the pod `ci-nightly`. Transfer files only with `runpodctl send` / `receive`.
- One pod. Load the skills `runpod:runpod-mcp` and `runpod:runpod-pod-deploy` first. CPU flavor `cpu3g` (General Purpose), 32 vCPU / 128 GB, a US data center with HIGH stock. Re-read price and stock with `list-cpu-types` before creating. If `cpu3g` costs more than $1.28/h, use nothing dearer than $1.76/h (`cpu3m`) and shorten the limit so hours x rate stays under $6. Delete the pod no later than 4.5 hours after creation (at $1.28/h), collecting logs first. Delete it at once on an unrecoverable failure. Stop or delete only the pod this run created. Confirm every deletion with a fresh `list-pods` read.
- Memory on the pod: at most 10 parts on 128 GB (the last run peaked at 83 GB, one process at 17.3 GB). Watch the container's own memory counter, not `free`.

## Phase 1: launch (the scheduled runner)

Build the tables, create the pod, upload, start the parts, confirm they are running, then return: the pod id, creation time, hourly rate and the deadline for deletion. Do not wait for the run to finish.

## Phase 2: collect (a later runner, dispatched by the managing session)

Read progress (`pod\progress.sh`). When every part has finished, or at the deadline: fetch the logs, delete the pod, confirm with `list-pods`, merge the outcomes with `analyze.py`. Then, locally, one process at a time and only with more than 8 GB of free RAM, run in the main checkout with `-m "not slow"` the tests that skipped on the pod for missing inputs (mainly `tests/tools/rom`, `tests/kaizo/test_extract_*`, `tests/saves`, `tests/test_user_save.py`).

## Report (short)

Tip hash; pod spec, data center, created and deleted times, hours x rate cost; setup time vs test time; totals passed/failed/skipped (pod plus local part); every failure with its one-line error and whether it also fails locally; wall-clock per part; peak memory; the 15 slowest tests merged across parts; tests over 60 s; problems hit; anything not done or not verified.
