# How this project is run

Standing rules every session obeys: `CLAUDE.md`. Design and roadmap: `docs/design.md`.
Scripts named here: `tools/ops/` (`tools/ops/README.md`).

## 1. Sessions and roles

- **Managing session.** The only session the user talks to. It does not build. It gates
  merges into `main`, pushes `main`, watches the streams, rents, uses and deletes test pods
  and tracks usage. It reports to the user only decisions, finished milestones and real
  problems.
- **Work streams.** Opus sessions, each in its own git worktree and branch:
  - E1: abilities, held items, battle conditions, AI flags.
  - E2: the doubles/tag/multi format.
  - E3: singles move-effect families, mechanic plus trainer AI.
  A stream never pushes, never creates billable resources and never writes to `main`.
- **Subagents.** Each batch runs the pipeline in section 2.

## 2. A batch, start to merge

1. The stream sends its family list (CLAUDE.md, Batches) to the managing session, which
   checks ownership against the other streams.
2. An Opus rule writer reads the decomp, both the C (`src/battle`) and the battle scripts
   (`res/battle/scripts`), and writes one rule sheet per mechanic group, every rule cited by
   file and line. Each sheet states what the effect does to each of these, or "none" with the
   cite: hit chance (accuracy and evasion, including same-turn changes by a faster battler),
   damage inputs, crit inputs, held items read later in the turn, and the trainer AI's view.
3. Sonnet builders implement each sheet in the JAX engine and the NumPy reference, with the
   trainer AI's handling, the coverage check and tests. Rulings go in the stream's ledger
   (`docs/kaizo/e1-mechanics-ledger.md`, `docs/kaizo/e2-mechanics-ledger.md`,
   `docs/kaizo/e3-ledger.md`).
4. The batch's tests run on the test pod while it is in work (CLAUDE.md, Where tests run;
   section 5).
5. One Opus review agent checks the batch; one fix round fixes its findings. Commits added
   after a review get their own review before the merge.
   A new test that asserts over a growing set or a coverage-dependent refusal carries the
   `fullset` marker.
6. The stream merges `main` into its branch, does the after-merge steps (CLAUDE.md), then
   the targeted run on the merged tip (CLAUDE.md, Merges) on the pod. The `fullset`
   tests are their own pod run (DAYBOX.md: one mark per run). With the tip it sends for
   these runs the stream sends its `--allow-new` list and the divergences it expects; then
   the managing session runs the tip's snapshot compare (section 5).
7. The stream sends one message: the final tip, the batch contents with citations, the
   review outcome, the targeted run's results with time and peak memory per test file, and
   its ledger's explanation of every diverged part of the snapshot compare.
8. The managing session dispatches a fresh Opus review agent as the merge gate. Give it the
   diff range and the criteria, never the stream's conclusions (the ledger's explanation of a diverged snapshot compare is
   inside the diff range and is checked against the compare log). Verdict: MERGE or FIX. When
   only tests or docs changed since an already-reviewed commit, a diff stat of `src` and
   `tools` replaces the review.
9. On MERGE: `git merge --no-ff` into `main`, `git push origin main`, send the stream the
   new tip and the non-blocking notes, and tell the user in at most three lines.

**Merge train.** When two or more batches are expected to be merge-ready within one gate run
of each other (about 3 hours), the managing session names a train order.
- The first stream merges `main`; each later stream merges the tip of the stream before it
  (not `main`) and does the after-merge steps (CLAUDE.md) on that. In a train only the last
  tip gets the pod runs of step 6 and the after-merge `fullset` run; earlier members do the
  table rebuild and the duplicate scan only.
- Each batch keeps its own Opus merge-gate review (step 8) on its own diff range.
- The pod work runs once, on the last tip: one targeted run with the union of the lists
  and the default mark, one `MARK='fullset'` run (DAYBOX.md, Merge candidate), one snapshot
  compare against the `main` capture with the union `--allow-new` list (section 5).
- On green and MERGE from every review, the managing session merges the last tip
  `--no-ff` (it contains the others) and pushes once.
- A failure goes to the stream that owns the failing test's area. If it cannot be
  attributed and fixed in one fix round, the train is dropped and the batches merge one by
  one.
- A batch that is ready alone and has no partner within one gate run merges alone.

## 3. Tests

- **Per batch and per merge:** CLAUDE.md, Batches and Merges.
- **Nightly:** the fast tier on `main`'s tip on the nightly pod (section 5). Tests that need
  the ROM, saves, the calc archive or the docs sheet skip there; run them locally afterwards.
- **Slow tier:** after each merge wave (the merges of one day), at most once a day, the
  managing session runs `-m slow` on `main`'s tip on the test pod: one run agent,
  `MARK='slow' bash tools/ops/day/pack.sh <tip> <tables> slow-<yyyymmdd>-<short tip> <list>`
  with the one-line list `tests 8` (DAYBOX.md, Send a run). A red slow tier is fixed before
  new batches start, like a red nightly. Slow tests run in no other gate except
  `tests/test_differential.py`, which every merge run carries whole (list line
  `tests/test_differential.py 1 A`), the nightly's `tests/test_damage_oracle.py 1000 A`
  line, and the files a batch lists with the `A` flag.
- **Split close:** one full suite, one final review, a replay of the user's recorded fight
  for that split, a cross-check against the reference scoring tool, and a speed comparison.
- **Local runs:** CLAUDE.md, Where tests run.

## 4. Progress measure

`tools/kaizo/backlog.py <out.md>` (needs built Kaizo tables) counts, per split, the battles
that are fully playable and the mechanic groups still missing, for the opponent side and for
the player's side. At `aaf2ef7`: 98 of 387 battles playable; 158 missing groups on the
opponent side plus 35 player-only. Re-run it after each merge. The drop in missing groups
per day is the rate used for estimates.

## 5. Rented test pods (Runpod CPU)

- **What may leave the PC:** source code at a commit (`git archive`) and the built tables
  listed by `TABLE_ALLOW` in `tools/ops/common.sh`. Never the ROM, save files, the calc
  archive or its extraction, the docs sheet or its extraction, or anything personal.
- **Nightly:** brief `tools/ops/nightly_brief.md`. Run it on `ci-day-big` when that pod is
  up; when its run ends the nightly
  agent deletes `ci-day-big` if no other run is active on it. Otherwise it rents `ci-nightly`:
  at most $6 per run; the nightly agent itself deletes it within 4.5 hours of creation.
  A tip that equals the last nightly's tip does not run again.
- **Test pod (`ci-day-big`):** 32 vCPU CPU, no daily limit, for every stream test run, in one
  of two shapes, each with its limits passed in the exec-pod body's `env`:
  - 128 GB (`cpu3g`): `GATE_GB` 88, `MAXP` 10, `KILL_GB` 118 (the runner's defaults).
  - 256 GB (the memory-optimized 32 vCPU type): `GATE_GB` 176, `MAXP` 18 (more than about
    18 units on one pod stalls it), `KILL_GB` 236, `SPACING` 4, and `SNAP_GATE_GB` 176 for
    `snap.sh`.
  Several test pods may be up at once (approval: `docs/decisions.md`, section 5). Pod N's id
  is in `_scratch/rent/day<N>/POD_ID_<N>`; pod 1's is in `_scratch/rent/POD_ID`.
  Read the shape before sending a body: exec `cat /sys/fs/cgroup/memory.max` on the pod
  (bytes; `max` or a value under 200 GB means the 128 GB limits). Never pass the 256 GB
  limits to a 128 GB pod.
  Never rent a 16 vCPU / 64 GB pod. Keep it up while streams work; delete it when
  nothing is queued. Delete and re-create a pod whose container has not started within
  5 minutes. Bring-up and run scripts: `tools/ops/day/DAYBOX.md`.
- **Pod runs.** Streams have no pod tool: a stream sends the managing session its commit
  and test-file list; the managing session dispatches one fresh agent per run (Sonnet for a
  day run; `tools/ops/nightly_brief.md` for the nightly) that packs and starts the run with
  the scripts (`pack.sh`, then its exec-pod body), makes one blocking `wait.sh` call in the
  background, and reports once: the final result with counts, time and peak memory per
  file, or the first failure when the brief asked for `--until first-failure`. No polling
  turns: the agent does not call `status.sh` while `wait.sh` runs. Only committed code runs
  on the pod.
- **Snapshot capture of `main`.** After every push of `main` the managing session starts
  one capture of the new tip on a test pod, in `/work/snapmain-<short sha>` (seeds 0-199,
  about 90 minutes; commands: `tools/ops/day/snap.sh`, DAYBOX.md, "Snapshot compare"). Keep the newest two captures;
  delete an older one only when no compare reads it. A capture is never sent back to a pod:
  one lost with a pod is re-made on the next pod.
- **Snapshot compare of a merge candidate.** Compare only (about 85 minutes), against the
  capture of the `main` tip its branch contains, with `--allow-new <the stream's list>`
  (left out when the list is empty). Never compare a merge candidate against an older base
  once `main` was merged into the branch.
- **Snapshot pass rule.** `compare: ok` passes. A diverged compare passes only when the
  stream's ledger explains every diverged part (in a tests/docs-only commit) and the
  merge-gate review agent has checked that explanation against the compare log. Anything
  unexplained blocks the merge.
- **Rebuilding a diverged state.** The snapshot generator has drawn different states on the
  PC (Windows) than on the pod (Linux) from the same seed and content hash. A stream rebuilds
  a diverged part only from the capture's `<part>.inputs.pkl.gz`: it asks the managing
  session, which copies those files from the pod with `snap.sh fetch` (DAYBOX.md) into `<scratch>/rent/out/<snapshot folder>/fetch-<epoch>/`.
- **Snapshot jobs and the runner.** A capture or compare runs outside the runner's `MAXP`,
  `GATE_GB` and `KILL_GB` accounting. While one runs, the managing session sends new pack
  runs with `MAXP` lowered by 2.
- **Chunking.** At most one test file per process. Cut large files into chunks of test ids:
  about 57 ids for `tests/ai/test_kaizo_policy.py`, about 8 ids for memory-heavy engine
  files, about 40 for other files. Start a chunk only while container memory
  (`/sys/fs/cgroup/memory.current`) is under `GATE_GB`, and run at most `MAXP` chunks at
  once across all runs on the pod (Test pod above: 88 GB and 10, or 176 GB and 18): a chunk
  grows to 8-12 GB after it starts. Near `KILL_GB` (118 or 236 GB) kill
  this run's newest chunk and queue it again, at most twice per chunk; a third kill counts
  as a failure. Never signal PID 1.
- **Completeness.** Per test file, passed + failed + skipped + errors + xfailed + xpassed
  equals its collected count. A file whose totals differ, a file that failed to collect and
  a chunk that crashed count as failures.
- Only the managing session rents, uses and deletes pods. Use generic names. Verify every
  delete with a pod listing. Each spending cap is the user's explicit approval; anything
  larger or longer needs a new approval.
- Transfer files with `runpodctl send` / `receive`; run commands on the pod through the
  Runpod tools' pod exec. A pod exec call ends after 25 seconds: start work with `nohup`.
  The pod reports back without pod exec: its `notify.sh` offers a first-failure beacon and
  the results, each under its own random `runpodctl send --code`, and the PC's `wait.sh`
  takes them (`tools/ops/day/DAYBOX.md`, "Wait").

## 6. Watching the streams

- `tools/ops/watchdog.ps1` prints, per stream, the age of its last commit and its
  uncommitted files, the Python test processes with memory and run time, free RAM and the
  fast-tier lock state, and lists flags.
- `tools/ops/watch_flags.ps1` runs the watchdog every 10 minutes in the background and exits
  (waking the managing session) only on a flag that a running test tier does not explain.
  `-HoldUntil` silences "has not committed" and "may have stopped" while streams wait on tests.
- A fallback check runs twice a day (section 7).
- A stream ends a turn only with a merge tip plus results, a real blocker, or a family list
  or plan. It never sends a "still waiting" note.

## 7. Scheduled prompts in the managing session

They live only in the session and expire after 7 days. Re-create them after every restart
or clear. Times are local cron expressions; the scheduler adds jitter.

| Cron | What |
|---|---|
| `47 9,21 * * *` | Fallback check: watchdog plus usage; silent unless something is wrong |
| `7 22 * * *` | Dispatch the nightly agent (`tools/ops/nightly_brief.md`): it runs the fast tier on `ci-day-big` when that is up, else on `ci-nightly`, and deletes `ci-nightly` itself |
| `57 22 * * *` | Delete the test pod if nothing is queued and no nightly runs on it, verify with a pod listing, log the cost |

## 8. Usage rules

The weekly usage limit caps how much work lands in a week. The user manages the limit; the
managing session lowers cost per unit of work and never throttles the streams.

- No idle turns and stream clears: CLAUDE.md, Usage. A stream restarts from its ledger.
- The managing session reads every session's context use at each stream commit point and clears a stream past 30% there. It compacts itself before 35%, and only right after a user message. Automatic compaction is set to 50% as the backstop.
- Batch size (CLAUDE.md, Batches) spreads the fixed cost per batch: review, merge check,
  merge of `main`, table rebuild, targeted run.
- Models: CLAUDE.md, Usage.
- The managing session wakes only for a user message, a watch-loop exit, a stream's merge
  tip or blocker, or a scheduled prompt. It never reads logs or agent transcripts whole.
- Sessions load only the connectors they use; mail, drive and browser connectors are off
  for this project. Subagents run at default effort.

## 9. State and memory

- **Standing rule:** run the project as if a successor with no access to any session takes
  over at any moment, from the project folder alone. Write every decision, standing approval,
  open problem and next step to project files when it happens, not at handoff time. Write for
  a reader with zero context: full paths, branch names and tips, exact commands, what is
  done, in work or blocked and why.
- **Process and rules** are in the repo: `CLAUDE.md`, this file, `tools/ops/`.
- **Standing decisions** (approvals, preferences, lessons) are in `docs/decisions.md`. Add a
  decision there on the day it is made.
- **Live state** is in `_scratch/handoff/` (git-ignored): one state file per role, rewritten
  in full, sections as in section 12, "State files".
- Each stream keeps its ledger in `docs/kaizo/` and restarts from it after a clear.
- A session's private memory folder is a copy. Nothing a successor needs exists only there,
  in a session, or in the app's side-session notes.

## 10. Rebuilding the tables

`tools/ops/build_tables.sh` runs the full sequence: `tools.extract_decomp`,
`tools.build_tables`, then for Kaizo `tools.extract_rom`, `tools.extract_pkcalc`,
`tools.extract_kaizo_docs`, `tools.kaizo.overrides check` and `tools.build_tables --game kaizo`.
It takes the local inputs (`data/local.toml`, the symbols map in `.cache/decomp/`, the decomp)
from the main checkout.

- In a stream's worktree, after every merge of `main` (CLAUDE.md): `bash tools/ops/build_tables.sh --here`.
- For a pod run: `bash tools/ops/build_tables.sh <commit>` writes `_scratch/rent/tables_<short>.tar.xz`.

## 11. Tidiness

- Every project file lives under the project folder. Scratch files, worktrees, scripts and
  logs go in the git-ignored `_scratch/`. Nothing goes on C: or beside the project.
- `.md` files state exact orders: no background, no hedges, one home per rule. Delete stale
  text; never annotate it.
- The managing session runs the cleanup at every split close and once a week: move stray
  files into the project, delete merged worktrees and old logs, tighten the `.md` files.
- The managing session fixes a doc in the same turn as the process change that made it stale.

## 12. Taking over

A successor has the project folder and nothing else. This section and the files it names are
enough to run the project.

**Read in this order:** `CLAUDE.md`; this file; `docs/decisions.md`;
`_scratch/handoff/MANAGER.md` (a stream: its `E<n>.md`); `tools/ops/day/DAYBOX.md` before any
pod work. This section is the source of the handoff procedure; a copy in
`_scratch/handoff/README.md` follows it.

### What travels with the project folder

- The repo, all worktrees, `_scratch/` (tables, lists, pod results, `_scratch/handoff/`).
- `CLAUDE.md`, `docs/`, `tools/ops/`.
- `.claude/agents/review.md`: the fresh-context review agent of every review and merge gate.
  This project copy is the source; a copy in the user profile's `.claude` folder follows it.
  The `/goal` planning command is an optional owner-level command, not needed to run the
  project.
- runpodctl's key: `_scratch/rent/home/.runpod/config.toml`. The scripts run runpodctl with
  `HOME` set to `_scratch/rent/home` (DAYBOX.md, Paths).
- The live watch scripts: `_scratch/manager/watch_flags.ps1` and `_scratch/manager/watchdog.ps1`
  (command line in `_scratch/handoff/MANAGER.md`). The copies under `tools/ops/` are older
  until synced.

### What does not travel

- Session ids, cross-session message addresses, linked-session lists, scheduled prompts,
  background agents and shells. A new session recreates what it needs ("Picking up").
- The app's side-session notes. Durable state never lives there.
- A session's private memory folder (`~/.claude/projects/<project>/memory/`). It loads only
  for a session opened in the main checkout under the same OS user, and it is a copy
  (section 9).
- Tool sign-ins: `gh`, the Runpod plugin's MCP server.

### State files (`_scratch/handoff/`, git-ignored)

- `MANAGER.md`: the managing session. `E1.md`, `E2.md`, `E3.md`: one per work stream.
  `SETUP.md`: this machine's setup inventory with its real paths.
- Every state file has these sections: Role, Tips (branch, worktree, commit), In flight, Held,
  Owed, Next steps in order, Open questions, Pointers. Its first line under the title is
  `Updated:` with date and time.
- A state file states facts as of its `Updated:` line. Verify tips with
  `git log -1 <branch>` and pods with one pod exec call each before acting.
- Personal details, pod ids and codes go only here, never in tracked files.

### Writing a handoff (the session that stops)

Trigger: the owner types `handoff`, the session is told to pause, or it reaches a commit
point with usage nearly spent.
1. Finish or abandon the current edit and commit. A `WIP:` commit is allowed on a stream
   branch, never on `main`. Worktree clean.
2. Stop own agents and local processes. Pod runs may continue; list them by run name.
3. Rewrite its state file completely, never append: everything a stranger needs, nothing that
   git already says.
4. Reply to the owner in one line: file written, tip, what is in flight.

The managing session also confirms that each stream's file is current, then writes
`MANAGER.md` last.

### Picking up (the fresh session)

The owner opens a session in the main checkout folder and pastes one line:
- Managing session: `Read docs/process.md section 12 and _scratch/handoff/MANAGER.md, then take over as the managing session.`
  On a new account or machine add: `First do the setup in docs/process.md section 12.`
- Stream: `Read docs/process.md section 12 and _scratch/handoff/E<n>.md, then take over as stream E<n>.`
  The managing session normally starts the stream sessions itself and gives them this line.

The session then:
1. Reads `CLAUDE.md`, this file and `docs/decisions.md`.
2. Verifies tips and worktrees (`git worktree list`, `git log -1 <branch>`).
3. The managing session only, and only when no pause is in force: recreates the watch loop
   (section 6), the scheduled prompts (section 7) and one Sonnet collector agent per pod for
   unfinished pod runs (DAYBOX.md, Wait and Collect). A stream session recreates none of
   these.
4. Reports to the owner in five lines what it found and what it does first.
5. Waits for nothing unless its state file says an owner decision is open.

### Rules

- One account or person works on a branch or worktree at a time. Never run the same role
  twice at once.
- Old session ids in a state file are history only. Never message them.

### What you must supply yourself

None of these is in the repo. The setup assumes Windows with Git Bash and PowerShell.
- The inputs: the vanilla Platinum (US) ROM, the Kaizo patch, the PKCalc zip (the official
  calc repo), the Kaizo docs spreadsheet and the save files. Their paths go in
  `data/local.toml` (git-ignored; template `data/local.example.toml`; inputs:
  `docs/kaizo/pipeline.md`). The owner writes it; sessions never edit it.
- The decomp at `external/pokeplatinum` (git-ignored): `bash tools/clone_decomp.sh` clones
  it at the pinned commit (`docs/decomp_paths.md`).
- `.cache/` (git-ignored, never uploaded): `.cache/decomp/<commit>/main.nef.xMAP` is built
  by hand once in WSL (`docs/rom_pipeline.md`, Decomp build); `.cache/rom/` (the patched ROM
  copy) is written by `tools.extract_rom`; the JAX caches are written by the engine
  (`src/nuzline/jaxcache.py`: `<repo>/.cache` unless `NUZLINE_CACHE_DIR` is set).
- The owner's recordings folder (path in `_scratch/handoff/`), read-only.
- A Runpod account, signed in through the Runpod plugin's MCP server, and the runpodctl
  binary at `_scratch/rent/bin/runpodctl-windows-amd64.exe` (used by `tools/ops/common.sh`;
  not on PATH; version and source: DAYBOX.md, Paths) with its key in
  `_scratch/rent/home/.runpod/config.toml`.
- Claude Code with the plugins `superpowers@claude-plugins-official` and `runpod@runpod` at
  user scope. User settings `env`: `CLAUDE_CODE_TMPDIR`, `TEMP` and `TMP` on a folder on the
  project's drive (never C:), and `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50`.
- git, `gh` (signed in; it pushes `main`), uv (it brings the project's Python 3.12), node.

### Setup on a new account or machine (once, by the managing session)

1. Open the session in the main checkout folder. The managing session runs on Fable.
2. Runpod: make one read-only call (list pods). If it asks for authorisation, the owner signs
   in; the session never types credentials.
3. Desktop app preferences are per account: permission mode as on the previous setup,
   keep-awake on while streams work, notifications on. The owner sets them, or approves the
   change. Do not change security settings.
4. Add the owner's recordings folder as an additional directory only when the Elite Four
   replay check needs it.
5. Connectors are not needed (section 8).
6. Start three stream sessions (E1, E2, E3): folder the main checkout, model Opus, each with
   its pick-up line. Record the new session ids in `MANAGER.md`.
7. Recreate the watch loop (section 6) and the scheduled prompts (section 7), only when no
   pause is in force.

Checks (Git Bash, from the main checkout):
```
git fetch -q && git status -sb && git log -1 --format=%h          # main matches origin/main
for t in git gh uv python node; do $t --version | head -1; done
gh auth status
ls _scratch/rent/bin/runpodctl-windows-amd64.exe _scratch/rent/home/.runpod/config.toml
ls data/local.toml external/pokeplatinum/src .cache/decomp >/dev/null && echo INPUTS_OK
python -c "import json,os;print(json.load(open(os.path.expanduser('~/.claude/settings.json')))['enabledPlugins'])"
powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"   # free GB; local pytest needs more than 6
```
Then confirm in the session that the skills `superpowers:*` and `runpod:*` are listed and
that the Runpod tools answer. Report to the owner in five lines: what matched, what was
missing, whether Runpod answered, free RAM, and whether a pause is in force.
