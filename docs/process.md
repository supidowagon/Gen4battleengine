# How this project is run

Standing rules every session obeys: `CLAUDE.md`. Design and roadmap: `docs/design.md`.
Scripts named here: `tools/ops/` (`tools/ops/README.md`).

## 1. Sessions and roles

- **Managing session.** The only session the user talks to. It does not build. It gates
  merges into `main`, pushes `main`, watches the streams, creates and deletes test pods and
  tracks usage. It reports to the user only decisions, finished milestones and real problems.
- **Work streams.** Opus sessions, each in its own git worktree and branch:
  - E1: abilities, held items, battle conditions, AI flags, test memory.
  - E2: move-effect families (mechanic plus trainer AI), then the doubles/tag/multi format.
  A stream never pushes, never creates billable resources and never writes to `main`.
- **Subagents.** A stream builds each batch with one builder agent, gets one fresh review
  agent, and fixes what the review finds. Model choice: section 8.

## 2. A batch, start to merge

1. The stream sends its family list (CLAUDE.md, Batches) to the managing session, which
   checks ownership against the other stream.
2. One builder agent implements each rule in both engines (JAX and the NumPy reference),
   with the trainer AI's handling and the coverage check, each rule cited to the decomp by
   file and line. Rulings go in the stream's ledger (`docs/kaizo/e1-mechanics-ledger.md`,
   `docs/kaizo/e2-mechanics-ledger.md`).
3. One fresh review agent checks the batch; the stream fixes the blockers.
4. The stream merges `main` into its branch, does the after-merge steps (CLAUDE.md), then
   the targeted run on the merged tip (CLAUDE.md, Merges).
5. The stream sends one message: the tip, the batch contents with citations, the review
   outcome, and the targeted run's results with time and peak memory per test.
6. The managing session dispatches a fresh Opus review agent as the merge gate. Give it the
   diff range and the criteria, never the stream's conclusions. Verdict: MERGE or FIX. When
   only tests or docs changed since an already-reviewed commit, a diff stat of `src` and
   `tools` replaces the review.
7. On MERGE: `git merge --no-ff` into `main`, `git push origin main`, send the stream the
   new tip and the non-blocking notes, and tell the user in at most three lines.

## 3. Tests

- **Per batch and per merge:** CLAUDE.md, Batches and Merges.
- **Nightly:** the fast tier on `main`'s tip on the nightly pod (section 5). Tests that need
  the ROM, saves, the calc archive or the docs sheet skip there; run them locally afterwards.
- **Split close:** one full suite, one final review, a replay of the user's recorded fight
  for that split, a cross-check against the reference scoring tool, and a speed comparison.
- **Local limits:** heavy search tests compile the whole engine (about 9 GB). Run them one
  process at a time, behind a memory guard, with a free-memory check before each start.

## 4. Progress measure

`tools/kaizo/backlog.py <out.md>` (needs built Kaizo tables) counts, per split, the battles
that are fully playable and the mechanic groups still missing, for the opponent side and for
the player's side. At `aaf2ef7`: 98 of 387 battles playable; 158 missing groups on the
opponent side plus 35 player-only. Re-run it after each merge. The drop in missing groups
per day is the rate used for estimates.

## 5. Rented test pods (Runpod CPU)

- **What may leave the PC:** source code at a commit and the built tables (`.json`/`.npz`).
  Never the ROM, save files, the calc archive, the docs sheet, or anything personal.
- **Agents never wait on a pod.** A runner agent launches the pod run and returns. The
  managing session checks the result later with a second runner that reads the status,
  fetches the logs and reports.
- **Nightly pod (`ci-nightly`):** at most $6 per run. Brief: `tools/ops/nightly_brief.md`.
  Delete the pod within 4.5 hours of creation.
- **Daytime pod (`ci-day`):** at most $10 per day, for the streams' targeted merge runs.
  Recipe: `tools/ops/day/DAYBOX.md`. Streams have no pod tool: a stream sends the managing
  session one line (commit, pytest paths or expression, memory guard); the managing session
  dispatches a Sonnet runner that uploads that commit and its built tables and starts the
  run, then a later runner reports counts, time and peak memory. Only committed code runs
  on the pod. A stopped CPU pod keeps no disk, so the pod is deleted each night (section 7)
  and re-created (about 3 minutes) when first needed the next day.
- Only the managing session creates and deletes pods. Use generic names. Verify every
  delete with a pod listing. Each spending cap is the user's explicit approval; anything
  larger or longer needs a new approval.
- Transfer files with `runpodctl send` / `receive`; run commands on the pod through the
  Runpod tools' pod exec.

## 6. Watching the streams

- `tools/ops/watchdog.ps1` prints, per stream, the age of its last commit and its
  uncommitted files, the Python test processes with memory and run time, free RAM and the
  fast-tier lock state, and lists flags.
- `tools/ops/watch_flags.ps1` runs the watchdog every 10 minutes in the background and exits
  (waking the managing session) only on a flag that a running test tier does not explain.
  `-HoldUntil` silences "has not committed" while streams wait on tests.
- A fallback check runs twice a day (section 7).
- A stream ends a turn only with a merge tip plus results, a real blocker, or a family list
  or plan. It never sends a "still waiting" note.

## 7. Scheduled prompts in the managing session

They live only in the session and expire after 7 days. Re-create them after every restart
or clear. Times are local cron expressions; the scheduler adds jitter.

| Cron | What |
|---|---|
| `47 9,21 * * *` | Fallback check: watchdog plus usage; silent unless something is wrong |
| `7 22 * * *` | Launch the nightly fast tier on the nightly pod |
| `57 22 * * *` | Delete the daytime pod, verify with a pod listing, log the cost |

## 8. Usage rules

The weekly usage limit caps how much work lands in a week. The user manages the limit; the
managing session lowers cost per unit of work and never throttles the streams.

- No idle turns and stream clears: CLAUDE.md, Usage. A stream restarts from its ledger.
- The managing session compacts at about 100k tokens, and only right after a user message.
- Batch size (CLAUDE.md, Batches) spreads the fixed cost per batch: review, merge check,
  merge of `main`, table rebuild, targeted run.
- Models: Sonnet for every mechanical subagent job; Opus for decomp work, reviews and
  merges.
- Trial: on one E1 batch Sonnet builds and Opus reviews. It becomes the rule if it costs at
  most one extra fix round and lets no wrong rule reach the merge gate.
- The managing session wakes only for a user message, a watch-loop exit, a stream's merge
  tip or blocker, or a scheduled prompt. It never reads logs or agent transcripts whole.
- Sessions load only the connectors they use; mail, drive and browser connectors are off
  for this project. Subagents run at default effort.

## 9. State and memory

- Each stream keeps its ledger in `docs/kaizo/` and restarts from it after a clear.
- The managing session keeps, in its private memory folder outside the repo, a handoff note
  (current tips, pending work, schedules to re-create) and one note per standing decision or
  approval. After a clear it reads the handoff first.

## 10. Rebuilding the tables

`tools/ops/build_tables.sh` runs the full sequence: `tools.extract_decomp`,
`tools.build_tables`, then for Kaizo `tools.extract_rom`, `tools.extract_pkcalc`,
`tools.extract_kaizo_docs`, `tools.kaizo.overrides check` and `tools.build_tables --game kaizo`.
Rebuild after every merge of `main` (CLAUDE.md).

## 11. Tidiness

- Every project file lives under the project folder. Scratch files, worktrees, scripts and
  logs go in the git-ignored `_scratch/`. Nothing goes on C: or beside the project.
- `.md` files state exact orders: no background, no hedges, one home per rule. Delete stale
  text; never annotate it.
- The managing session runs the cleanup at every split close and once a week: move stray
  files into the project, delete merged worktrees and old logs, tighten the `.md` files.
- The managing session fixes a doc in the same turn as the process change that made it stale.
