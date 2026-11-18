# How this project is run

This is the working pipeline as of 2026-11-17. `CLAUDE.md` holds the short standing rules;
this file explains the whole setup so it can be rebuilt from nothing. The scripts it names
are in `tools/ops/`. Design and roadmap are in `docs/design.md`.

## 1. Sessions and roles

- **Managing session.** The only session the user talks to. It does not build. It gates
  merges into `main`, pushes `main`, watches the streams, rents and deletes test machines,
  tracks usage, and reports decisions, finished milestones and real problems (no routine
  progress relays).
- **Work streams.** Long-running sessions on Opus, each in its own git worktree and branch:
  - E1: abilities, held items, battle conditions, AI flags, and the test-memory work.
  - E2: move-effect families (mechanic plus trainer AI), then the doubles/tag/multi format.
  A stream never pushes, never creates billable resources, and never writes to `main`.
- **Subagents.** Streams build each batch in one builder agent, get one fresh review agent,
  and fix what it finds. Mechanical jobs (running and reading test runs, table rebuilds,
  ledger and backlog edits) go to Sonnet; merges, review fixes and decomp transcription
  to Opus. (Trial from 17 Nov: Sonnet as builder with an Opus review; see section 8.)

## 2. A batch, start to merge

1. The stream sends its family list (15-20 mechanic groups) to the managing session, which
   checks ownership against the other stream.
2. One builder agent implements each rule in both engines (JAX and the NumPy reference),
   with the trainer AI's handling and the coverage check, each rule cited to the decomp by
   file and line. Rulings go in the stream's ledger (`docs/kaizo/e1-mechanics-ledger.md`,
   `docs/kaizo/e2-mechanics-ledger.md`).
3. One fresh review agent checks the batch; the stream fixes blockers.
4. The stream merges `main` into its branch, rebuilds the vanilla and Kaizo tables, scans
   for anything defined twice, runs the tests that assert something is unsupported, then the
   targeted run on the merged tip: the four guards, `tests/test_differential.py`, and the
   tests of every area touched.
5. The stream sends one message: the tip, what the batch contains with citations, the
   review outcome, and the targeted run's results with time and peak memory per test.
6. The managing session dispatches a fresh Opus review agent as the merge gate. It is given
   the diff range and the criteria, never the stream's conclusions. Verdict MERGE or FIX.
   If only tests or docs changed since an already-reviewed commit, a diff stat of `src` and
   `tools` replaces the review.
7. On MERGE: `git merge --no-ff` into `main`, `git push origin main`, tell the stream the new
   tip, pass on non-blocking notes, and tell the user in a few lines.

No stream starts a third batch while two of its batches are unmerged. Several ready batches
may be stacked into one merge.

## 3. Tests

- **Per merge:** the targeted run above, under an hour.
- **Nightly:** the fast tier (`-m "not slow"`) on `main`'s tip, on a rented machine (section
  5). The tests that need the ROM, saves, the calc archive or the docs sheet are skipped
  there and run locally afterwards. A red nightly is fixed before new batches start.
- **Split close:** one full suite, a final review, a replay of the user's recorded fight
  for that split, a cross-check against the reference scoring tool, and a speed comparison.
- **Limits:** a fast-tier test stays under 60 s and 4 GB or is marked slow. Heavy search
  tests compile the whole engine (about 9 GB); they run one process at a time locally, with
  a memory guard and a free-memory check before each start.
- **Why tests matter here:** the differential test (both engines must agree), the guards
  and late runs have caught real rule errors after review (for example hit thresholds
  after Haze, Psych Up and Acupressure in E2's batch 12).

## 4. Progress measure

`tools/kaizo/backlog.py <out.md>` (needs built Kaizo tables) counts, per split, the battles
that are fully playable and the mechanic groups still missing, for the opponent side and for
the player's side. On 17 Nov at `cc7613c`: 79 of 387 battles playable, 205 missing groups
on the opponent side plus 39 player-only. The missing-group count per day is the rate used
for estimates; it is re-run after each merge.

## 5. Rented test machines (Runpod CPU pods)

The local PC has 32 GB of RAM, so only one heavy test runs at a time; a local nightly took
13 h 27 min, the same run on a rented 32 vCPU / 256 GB pod took 2 h 29 min for about $4.50.

- **What may leave the PC:** source code at a commit and the built tables (`.json`/`.npz`).
  Never the ROM, save files, the calc archive, the docs sheet, or anything personal.
- **Nightly pod (`ci-nightly`):** at most $6 per run. A scheduled prompt in the managing
  session dispatches an Opus agent with `tools/ops/nightly_brief.md`: rebuild the tables in
  a detached worktree, upload, split the fast tier into parts dealt round-robin, run, fetch
  logs, delete the pod within 4.5 hours, then run the skipped tests locally and report.
- **Daytime pod (`ci-day`):** at most $10 per day, for the streams' targeted merge runs so
  they stop queueing on the PC. Its recipe is `tools/ops/day/DAYBOX.md`. The streams have
  no tool to run commands on the pod, so a stream sends the managing session one line
  (commit, the pytest paths or expression, the memory guard) and the managing session
  dispatches a Sonnet runner that uploads that commit and its built tables, runs the tests,
  fetches the logs and reports counts, time and peak memory. Only committed code runs
  there. A CPU pod keeps no disk when stopped, so a scheduled prompt deletes it each night
  and logs the cost, and it is re-created (about 3 minutes) when first needed the next day.
- Only the managing session creates and deletes pods, with generic names, and verifies with
  a pod listing after every delete. Each spending limit is the user's explicit approval;
  anything larger or longer needs a new one.
- File transfer uses `runpodctl send` / `receive`; commands on the pod go through the
  Runpod tools' pod exec.

## 6. Watching the streams

- `tools/ops/watchdog.ps1` prints, for each stream, the age of its last commit and
  uncommitted files, the Python test processes with memory and run time, free RAM, and the
  state of the fast-tier lock; it lists flags.
- `tools/ops/watch_flags.ps1` runs the watchdog every 10 minutes in the background and
  exits (which wakes the managing session) only on a flag that a running test tier does not
  explain. `-HoldUntil` silences "has not committed" while streams are known to be waiting
  on tests.
- A fallback check runs twice a day. Streams end a turn only with a merge tip plus results,
  a real blocker, or a family list or plan; "still waiting" notes are not sent.

## 7. Scheduled prompts in the managing session

These live only in the session and expire after 7 days, so they are re-created after any
restart or clear:

| When | What |
|---|---|
| 09:47 and 21:47 | Fallback check: watchdog plus usage, silent unless something is wrong |
| 22:07 | Nightly fast tier on the rented pod |
| 22:57 | Delete the daytime pod, verify, log the cost |

## 8. Usage rules

The weekly usage limit, not test time, caps how much work lands in a week.

- No idle turns: a test runs inside the subagent that needs its result, so a stream wakes
  once with the result.
- Streams are cleared at about 100k tokens, at commit points, and restart from their
  ledgers. The managing session compacts at about 100k, only with the user present.
- Batches of 15-20 groups, so the fixed cost per batch (review, merge check, merge of
  `main`, table rebuild, targeted run) is spread wider.
- Sonnet for every mechanical subagent job; Opus for decomp work, reviews and merges.
- The managing session wakes only for a user message, a watch-loop exit, a stream's merge
  tip or blocker, or a scheduled prompt; it never reads logs or agent transcripts whole.
- Sessions load only the connectors they use: mail, drive and browser connectors are off
  for this project (each costs context on every turn). Subagents run at default effort.
- Usage limits are the user's side; the managing session optimises cost per unit of work
  and does not throttle the streams over the weekly pace.
- Trial (17 Nov): on one E1 batch Sonnet builds and Opus reviews. It becomes the rule if it
  costs at most one extra fix round and lets no wrong rule reach the merge gate.

## 9. State and memory

- Each stream keeps a ledger in `docs/kaizo/` and restarts from it after a clear.
- The managing session keeps a handoff note (current tips, what is pending, schedules to
  re-create) and one note per standing decision or approval in its private memory folder,
  outside the repo. After a clear it reads the handoff first.
- Standing rules that every session must follow are in `CLAUDE.md`.

## 10. Rebuilding the tables

`tools/ops/build_tables.sh` is the full sequence: `tools.extract_decomp`, `tools.build_tables`,
then for Kaizo `tools.extract_rom`, `tools.extract_pkcalc`, `tools.extract_kaizo_docs`,
`tools.kaizo.overrides check` and `tools.build_tables --game kaizo`. Stale Kaizo intermediates
have caused false test failures, so rebuild after every merge of `main`.
