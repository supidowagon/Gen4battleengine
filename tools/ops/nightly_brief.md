# Nightly fast tier on a rented Runpod CPU pod (brief for an Opus general-purpose agent)

Standing approval from the user (2026-11-17): run the nightly on Runpod, at most $6 per run, uploading only the BUILT game tables. Nothing else is approved.

Project: `<repo>`. Run the fast tier (`pytest -m "not slow" -q -p no:cacheprovider --durations=40 -rfEs`) on the current tip of local `main` (must equal `origin/main`; if not, stop and report). Everything from the first run is in `<scratch>ent\` (`pod\pod_setup_and_run.sh`, `pod\run_parts.sh`, `build_tables.sh`, `split_parts.py`, `analyze.py`, `bin\runpodctl-windows-amd64.exe` with HOME/USERPROFILE pointed at `_scratch\rent\home`). The scripts were updated after the first run to deal tests round-robin into NPARTS parts and are untested: read them and smoke-test before relying on them.

HARD RULES
- Never upload the ROM (`.nds`), any save (`.sav`, `.dsv`), the PKCalc zip, the docs sheet, or anything from the user's Desktop. List the bundle's contents before sending and confirm it holds only built `.json`/`.npz` tables.
- Never write to any folder named `backup`. Scratch only under `_scratch\rent\`. Do not modify the main checkout or any branch; never push. Build tables for the tip in a detached worktree under `_scratch\rent\wt` (rebuild the Kaizo intermediates too: extract_rom, then build_tables; a ROM copy may exist only in that worktree's git-ignored `.cache`); remove the worktree at the end.
- No personal details in pod names, env vars or requests to outside services; name the pod `ci-nightly`. Transfer only with runpodctl send/receive.
- Money: one pod. Load skills `runpod:runpod-mcp` and `runpod:runpod-pod-deploy` first. CPU pod flavor `cpu3g` (General Purpose), 32 vCPU / 128 GB, US data center with HIGH stock; re-read price and stock with `list-cpu-types` before creating; if the price is above $1.28/h use nothing dearer than $1.76/h (`cpu3m`) and shorten the limit so hours x rate stays under $6. Delete the pod no later than 4.5 hours after creation (at $1.28/h), collecting logs first; delete immediately on unrecoverable failure. Only stop/delete the pod you created. Confirm deletion with a fresh `list-pods` read.
- Memory on the pod: last run peaked at 83 GB with 10 parts (one process up to 17.3 GB). With 128 GB use at most 10 parts and watch the container's own memory counter (not `free`).

AFTER THE POD RUN (locally, one process at a time, only if free RAM is above 8 GB): run the tests that were skipped on the pod for missing inputs (ROM, saves, PKCalc zip, docs sheet; mainly `tests/tools/rom`, `tests/kaizo/test_extract_*`, `tests/saves`, `tests/test_user_save.py`) in the main checkout with `-m "not slow"`.

REPORT (short): tip hash; pod spec, data center, created/deleted times, hours x rate cost; setup vs test time; totals passed/failed/skipped (pod + local part); every failure with its one-line error and whether it also fails locally; wall-clock per part; peak memory; the slowest-40 list merged across parts (top 15 only); tests over 60 s; problems hit; anything not done or not verified.
