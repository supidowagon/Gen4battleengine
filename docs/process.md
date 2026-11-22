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
  - E4: the doubles side: the doubles trainer AI, the observation at four battlers, the
    two-trainer and multi fight data, search with targets, prep with two actives. E4 does not
    touch E2's mechanic batches.
  A stream never pushes, never creates billable resources and never writes to `main`.
- **Pods in working hours:** one test pod per active stream plus one for captures and
  compares (section 5). Outside working hours delete every pod with nothing running or
  queued.
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
   after a review get their own review before the merge, except as in step 8.
   Every review brief (this step and step 8) demands the battle-script read for any finding
   that a check is missing because "the C command has no check": the reviewer reads the
   battle scripts (`res/battle/scripts`) that run the command and cites them, or the finding
   is dropped. (21 Nov 2026: a Griseous Orb finding was wrong because only the C was read.)
   A new test that asserts over a growing set or a coverage-dependent refusal carries the
   `fullset` marker.
6. The stream merges `main` into its branch, does the after-merge steps (CLAUDE.md), then
   the targeted run on the merged tip (CLAUDE.md, Merges) on the pod. The `fullset`
   tests are their own pod run (DAYBOX.md: one mark per run). With the tip it sends for
   these runs the stream sends its `--allow-new` list, which first passes the local
   pre-check (section 5, Allow-new list; the tip message says it passed), and the
   divergences it expects; then the managing session runs the tip's snapshot compare
   (section 5).
7. The stream sends one message: the final tip, the batch contents with citations, the
   review outcome, the targeted run's results with time and peak memory per test file, and
   its ledger's explanation of every diverged part of the snapshot compare.
8. The managing session dispatches a fresh Opus review agent as the merge gate. Give it the
   diff range and the criteria, never the stream's conclusions (the ledger's explanation of a diverged snapshot compare is
   inside the diff range and is checked against the compare log). Verdict: MERGE or FIX. When
   only tests or docs changed since an already-reviewed commit, a diff stat of `src` and
   `tools` replaces the review.
9. On MERGE: `git merge --no-ff` into `main`, `git push origin main`, then
   `bash tools/ops/offsite.sh` (section 12, Off-site copy), send the stream the new tip and the
   non-blocking notes, and tell the user in at most three lines.

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
  A tip that equals the last nightly's tip does not run again. A `ci-nightly` rented in the
  network volume's data center mounts the volume (Network volume below).
- **Scheduled tasks, not session timers.** The nightly dispatch (22:07) and the 30-minute
  stall check run as scheduled tasks, which survive a restart or clear of the managing
  session; they are not session timers (owner, 21 Nov 2026, `docs/decisions.md`). Until they
  are set up, at the next restart of the managing session, the session prompts of section 7
  stand in for them and are re-created after every restart.
- **Test pod (`ci-day-big`):** 32 vCPU CPU, no daily limit, for every stream test run, in one
  of two shapes, each with its limits passed in the exec-pod body's `env`:
  - 128 GB (`cpu3g`): `GATE_GB` 88, `MAXP` 10, `KILL_GB` 118 (the runner's defaults).
  - 256 GB (the memory-optimized 32 vCPU type): `GATE_GB` 200, `MAXP` 18 (more than about
    18 units on one pod stalls it), `KILL_GB` 236, and `SNAP_GATE_GB` 176 for `snap.sh`.
  - `RESERVE_TOP` 2, `RESERVE_FULL_GB` 40 and `RESERVE_PCT` 75 on both shapes: the runner's
    defaults; pass none of them.
  - Every run gets the `MAXP` of the live runs on its pod: the shape's value, or the lowered
    one while the pod is throttled (MAXP rule, Chunking below).
  - `SPACING` (s between two chunk starts of one runner) is 4 on both shapes. It is the
    runner's default; pass `"SPACING":"4"` in every exec-pod body's `env` anyway, so a run
    packed with older pod scripts (another default) gets it too. This is the one SPACING
    rule; DAYBOX.md and `tools/ops/nightly_brief.md` point here.
  Several test pods may be up at once (approval: `docs/decisions.md`, section 5). In working
  hours: one pod per active stream plus one capture pod (section 1). Pod N's id
  is in `_scratch/rent/day<N>/POD_ID_<N>`; pod 1's is in `_scratch/rent/POD_ID`. Each pod
  gets its own pod check before every pack (DAYBOX.md, Send a run step 3).
  The capture pod is pod 1 (id in `_scratch/rent/POD_ID`), rented like any test pod
  (`ci-day-big`, Bring-up in DAYBOX.md) in the 256 GB shape, so `snap.sh` runs with
  `SNAP_GATE_GB` 176 (on a 128 GB pod: 88). When pod 1 is down, the managing session names
  another up pod the capture pod and writes it in `_scratch/handoff/MANAGER.md`. For the
  no-idle-pods rule a pending compare counts as queued: a pod holding the capture a sent or
  expected merge candidate compares against is not idle (`docs/decisions.md`, section 5).
  Read the shape before sending a body: exec
  `{"command":["sh","-c","cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"],"timeoutSeconds":25}`
  on the pod (cgroup v2's `memory.max`, else cgroup v1's `memory.limit_in_bytes`: some rented
  hosts are v1), or the `limit` of the pod check (DAYBOX.md, Send a run step 3). Bytes; `max`,
  a value of 2^60 bytes or more (v1 without a limit: 9223372036854771712) or a value under
  200 GB means the 128 GB limits. Never pass the 256 GB limits to a 128 GB pod.
  Never rent a 16 vCPU / 64 GB pod. Check every new pod before its first use: exec
  `{"command":["sh","-c","cat /proc/sys/vm/max_map_count"],"timeoutSeconds":25}` must print
  1048576 or more. A pod with less (65530 aborts JAX units with rc 134) is deleted and another
  rented. The pod check (DAYBOX.md, Send a run step 3) refuses such a pod
  (`PODCHECK REFUSED vm.max_map_count=<value>`) and a snapshot job stops at once
  (`MAP_COUNT_LOW` in `times.txt`). Keep it up while streams work; delete it when
  nothing is queued. Delete and re-create a pod whose container has not started within
  5 minutes. Bring-up and run scripts: `tools/ops/day/DAYBOX.md`.
- **Network volume.** One 30 GB STANDARD network volume (owner, 21 Nov 2026,
  `docs/decisions.md`; a standing charge of about $2.10 a month, billed until it is deleted,
  attached or not) holds two things only: XLA programs compiled on the pods (the compile
  cache) and the complete snapshot captures of `main` tips. Nothing else goes on it: no
  tables, no ROM, saves, calc or docs-sheet data, nothing from the PC (only pods write to it).
  Its id is in `_scratch/rent/VOLUME_ID` (never committed); its data center is fixed at
  creation and every test pod and the capture pod is created there with the volume mounted at
  `/vol` (DAYBOX.md, Bring-up step 2); a volume cannot be attached to a running pod. With no
  stock there a pod is created elsewhere without it and runs as before the volume (cold compile
  cache, no capture store). JAX keeps writing the pod's own cache; `tools/ops/day/pod/cache_sync.sh`
  seeds it from the volume at a run's setup and publishes new entries after the run
  (temp name then rename, each entry checked to decompress first), and is run once more before
  a pod is deleted. The cache on the volume is kept per jax version and per CPU (a host tag of
  the CPU's vendor, family and model numbers and its flags, not its marketing name: the entries
  are compiled CPU programs, JAX's cache key does not hold the CPU, and XLA compiles for the
  host's target features and LLVM's host CPU name, which LLVM derives from those fields; the
  LLVM side is not re-checked in this repo, `tools/ops/day/pod/cache_sync.sh` header), so a pod
  seeds only programs compiled on a CPU of its tag. The low-space warning compares `du` of the
  volume with its size (30 GB; `df` of the mount shows the whole storage cluster). Every volume call
  runs under a timeout (900 s until the road test measures them; then at least three times the
  measured wall, DAYBOX.md "Network volume"), so a volume that does not answer never holds a run
  or a snapshot job; a seed or publish stopped by it keeps what it renamed. Pruning on every publish: unused jax versions after 7 days, the cache
  capped at 20 GiB, the 10 newest captures kept. The volume is never deleted, shrunk or grown
  without the owner. Steps: DAYBOX.md, "Network volume".
- **Pod runs.** Streams have no pod tool: a stream sends the run dispatcher session (below)
  its commit and test-file list; the dispatcher (or, for the nightly and the managing
  session's own merged-tip runs, the managing session) dispatches one fresh agent per run (Sonnet for a
  day run; `tools/ops/nightly_brief.md` for the nightly) that packs and starts the run with
  the scripts (`pack.sh`, then its exec-pod body), makes one blocking `wait.sh` call in the
  background, and reports once: the final result with counts, time and peak memory per
  file, or the first failure when the brief asked for `--until first-failure`. No polling
  turns: the agent does not call `status.sh` while `wait.sh` runs. Only committed code runs
  on the pod.
- **Run dispatcher session.** A standing Sonnet session (name `dispatch`, state file
  `_scratch/handoff/DISPATCH.md`; owner, 21 Nov 2026) handles the streams' run requests in
  place of the managing session: it takes each request (commit and test list), checks the
  list against the run rules of this section (one in-work run per stream, the short and full
  lists, chunking, marks), builds the tables, sends the run (pod check, `pack.sh`, exec-pod
  body), waits (`wait.sh`), collects, and reports to the requesting stream; it copies
  merge-candidate and `fullset` results to the managing session. It never creates pods, never
  pushes and never merges. Between 22:00 and 08:00 it may delete an idle test pod: one of ours,
  with nothing running or queued on it (checked on the pod: DAYBOX.md, Delete, the volume
  publish included), every run on it collected, and never the nightly's pod before or during
  the nightly; it verifies the delete with a pod listing, renames the pod's id file to
  `.deleted`, logs it in `DISPATCH.md` and sends the managing session one line. In working
  hours it only reports idle pods. Creating pods stays with the managing session.
- **Script step on a pod.** Besides pack runs, the dispatcher may run one committed script on
  a pod when a request names it: the script as committed at the sha under test (from that
  sha's checkout on the pod, never a file sent from the PC or typed into the exec body); it
  reads only that checkout and the built tables; it runs after the run's list has finished;
  its output is text written into the run's results folder and collected with the results.
  The exact command, sha and pod go in `_scratch/handoff/DISPATCH.md` before it runs.
- **Local pytest with four streams.** With four streams working, the pod is the default place
  for in-work tests. A local pytest (CLAUDE.md, Where tests run: one process, one file or
  chunk) runs only when more than 6 GB of RAM is free, checked right before it starts
  (PowerShell: `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB`), not earlier
  in the session.
- **One in-work run per stream.** A stream has at most one in-work pod run at a time. It
  sends no new tip while that run is going, unless the run has already failed (a first-failure
  report counts). The managing session holds any other request until the run is back.
- **In-work list (short).** An in-work tip's list holds, in this order:
  1. the test files new or changed since the previous run's tip:
     `git diff --name-only --diff-filter=d <previous tip> <tip> -- ':(glob)tests/**/test_*.py'`
     (the `:(glob)` form also matches `tests/test_*.py`; the plain `'tests/**/test_*.py'`
     does not), and every file the previous run counted as a failure by the Completeness
     rule below (failed, errored, crashed, killed three times, collect failed, incomplete);
  2. the tests of the functions the tip changed (node ids or the files that call them);
  3. the four guards (`CLAUDE.md`, Batches).
  Size it for about 30 minutes on a pod: at most about 40 units. Never
  `tests/test_differential.py` and never a whole-suite or whole-folder line (`tests 40`,
  `tests/engine 8`). Each file keeps the ids per chunk, `A` flag and weight of its line in
  the stream's previous list (a file with no line there: the ids per chunk of Chunking
  below). When a non-test file under `tests/` changed (`git diff --name-only <previous tip>
  <tip> -- tests`: helpers, `conftest.py`, golden data), add the previous-list files that
  import or read it, within the 40 units; the rest waits for the merge-candidate list.
- **Merge-candidate list (full delta).** Only a merge candidate gets the full delta list: the
  test files of every area the change touches, the four guards and `tests/test_differential.py
  1 A`, sized for about one hour on a pod alone (about 120 units). The nightly covers the
  rest. The `fullset` run is unchanged (DAYBOX.md, Merge candidate).
- **Snapshot capture of `main`.** After every push of `main` the managing session starts
  one capture of the new tip on the capture pod, in `/work/snapmain-<short sha>` (seeds 0-199;
  commands: `tools/ops/day/snap.sh`, DAYBOX.md, "Snapshot compare"). Keep the newest two captures;
  delete an older one only when no compare reads it. A capture is never sent back to a pod
  from the PC. On pods with the network volume a passed capture of a `main` tip is stored on
  the volume, keyed by its full sha and the table key of its tables, and is never overwritten;
  a capture of a key the volume holds, and a compare whose capture is not on its pod, fetch it
  from there instead of re-capturing (owner, 21 Nov 2026). Only a capture lost with a pod
  without the volume, or one the volume does not hold, is re-made on the next pod.
- **Capture pod.** Snapshot captures and compares run on the capture pod only. No pack run
  is sent to it while a capture or compare is going there. One snapshot job at a time per
  pod: each job gates only on its own parts, so two at once could overrun memory together.
  `run.sh` enforces it with a pod-wide lock (`/work/.snaplock`, taken after `uv sync`,
  released on every exit, a TERM included; a lock whose pid no longer runs a `run.sh` is
  removed); a second job sent meanwhile waits for it, printing one `lock wait` line a minute
  (DAYBOX.md, Snapshot compare). A job runs its parts in parallel
  (`SNAP_WORKERS`, default 16, each part its own process; `tools/snap_compare.py`, Workers):
  about 30 minutes for a capture or a compare (estimate: the longest part, `roark.rank`, takes
  15-22 minutes alone; replace it with the first timed parallel run). Serially
  (`SNAP_WORKERS=1`, or a commit whose `snap_compare.py` has no `--workers`) a job takes
  85-115 minutes.
- **Snapshot compare of a merge candidate.** Compare only, against the
  capture of the `main` tip its branch contains, with `--allow-new <the stream's list>`
  (left out when the list is empty). Never compare a merge candidate against an older base
  once `main` was merged into the branch.
- **Allow-new list.** The stream derives the list from the code, never from a compare log:
  every `BattleState` and `Outcome` field the tip has that the capture's commit lacks and no
  `UPGRADES` row of the tip's `tools/snap_compare.py` writes
  (`git diff <capture commit> <tip> -- src/nuzline/engine/state.py src/nuzline/engine/outcome.py`),
  plus any other output field new since that commit. It pre-checks the list locally before
  sending (one process, more than 6 GB free RAM, worktrees of both commits with their own
  tables as DAYBOX.md, Collect step 3): `snap_compare.py capture <dir> 0 --no-roark --parts
  step,jitstep,sample,rest,ai --repo <capture commit worktree> --commit <capture commit>`,
  then `snap_compare.py compare <dir> 0 --no-roark --parts step,jitstep,sample,rest,ai
  --repo <tip worktree> --commit <tip> --allow-new <list>`. The list passes when no part
  prints `FAIL only in this checkout` or `FAIL dropped field without an upgrade row`; a
  divergence in this local run does not count (the PC draws other states than the pod). The
  tip message names the list and says it passed the pre-check.
- **Snapshot pass rule.** `compare: ok` passes. A diverged compare passes only when the
  stream's ledger explains every diverged part (in a tests/docs-only commit) and the
  merge-gate review agent has checked that explanation against the compare log. Anything
  unexplained blocks the merge.
- **Rebuilding a diverged state.** The snapshot generator has drawn different states on the
  PC (Windows) than on the pod (Linux) from the same seed and content hash. A stream rebuilds
  a diverged part only from the capture's `<part>.inputs.pkl.gz`: it asks the managing
  session, which copies those files from the pod with `snap.sh fetch` (DAYBOX.md) into `<scratch>/rent/out/<snapshot folder>/fetch-<epoch>/`.
- **Snapshot jobs and the runner.** A capture or compare is outside the runner's chunk count
  and reserves; the runner sees it only as container memory. Pack runs stay off the capture
  pod while a job runs there (Capture pod above); a pack run already live there when a job
  starts keeps the pod's `MAXP`, and the runners' `GATE_GB` and `KILL_GB` read the job's
  memory. `SNAP_GATE_GB` is the job's own gate, on container memory without reclaimable page
  cache (`memory.current` minus `inactive_file`; cgroup v1: `total_inactive_file`): the job
  starts while that is under it, or once no pack unit has been live on the pod for
  `IDLE_WAIT_S` (300 s; `GATE_IDLE_START` in `times.txt`), and each part starts only while
  it + the live parts' reserve
  (expected peak minus RSS) + its own expected peak stays at or under it (`snap_compare.py`
  `PART_GB`: 13 GB for a step, jitstep or sample part, 8 rest, 2 ai, 1 gen, 11 `roark.trace`,
  27 `roark.rank`; the largest peaks of 19 pod logs times about 1.2). With no part of the job
  live the next part starts whatever memory reads (memory the job does not hold need not
  fall); a part held by the gate prints `gate wait` about once a minute.
- **Chunking.** At most one test file per process. Cut large files into chunks of test ids:
  about 57 ids for `tests/ai/test_kaizo_policy.py`, about 8 ids for memory-heavy engine
  files, about 40 for other files. Measured chunk peaks (7183 chunks of 120 pod runs, 20 Nov):
  median 7 GB, 90th percentile 12 GB, 99th 24 GB, largest 92 GB. Every chunk has an expected peak: its heaviest id in
  `tools/ops/day/pod/weights.txt` (every file any run measured, at its largest chunk and
  at least 4 GB, plus hand-kept per-test lines, each citing its run; `UNMEASURED_FILE_GB` 20 GB for a
  file never measured; a test that passed alone in a 1-id chunk at least 4 GB under its
  file's line has its own line; a multi-id chunk weighs the max over its ids), raised to the list line's `<GB>G` field when that is larger. A
  live chunk of a run packed before weights (no `out/<unit>.w`) counts `DEFAULT_WEIGHT_GB`
  12 GB: the 90th percentile of 7183 measured chunks, as those runs are mostly fast-tier
  chunks of measured files. Start a chunk only while container memory (`/sys/fs/cgroup/memory.current`, on cgroup v1
  `/sys/fs/cgroup/memory/memory.usage_in_bytes`) plus the reserve plus its own expected
  peak stays under `GATE_GB`, and run at most `MAXP` chunks at once across all runs on the
  pod (Test pod above: 88 GB and 10, or 200 GB and 18). A live chunk's gap is its expected
  peak minus its current RSS, where positive; the reserve is every gap of a chunk weighing
  `RESERVE_FULL_GB` (40 GB) or more, plus the light reserve, over every live chunk on the
  pod (all runs): the larger of the `RESERVE_TOP` (2) largest gaps of the lighter chunks and
  `RESERVE_PCT` (75) % of their weight sum minus their RSS sum. The top gaps cover a few light
  chunks peaking at once; the floor covers many growing together (chunks of one file grow in
  step; 18 mixed chunks peaked at 0.6-0.73 times their weight sum). `KILL_GB` is the last line. With no other live chunk on
  the pod a chunk starts when memory is under `GATE_GB` and memory plus its expected peak is
  under `KILL_GB`; one whose expected peak alone reaches `KILL_GB` starts alone at once, and
  one kept out only by memory no chunk holds starts after `IDLE_WAIT_S` (300 s). A head
  chunk that does not fit lets the first later chunk of its run that fits start; after
  `CLAIM_S` (120 s) it claims the pod: until it has started, its own run starts nothing else
  and another run starts only a chunk that fits beside it (memory + reserve + the claimed
  chunk's expected peak + its own under `GATE_GB`).
  **MAXP rule.** Every runner compares its own `MAXP` with the pod-wide count of live chunks
  (`tools/ops/day/pod/gate.sh` `admit`, `hold-maxp`; `RUNMAXP` caps this run's own chunks,
  `runner.sh` main loop). A run sent with a lower `MAXP` than another live run: its head
  chunk holds, after `CLAIM_S` it claims the pod, every runner then starts nothing until the
  count is below the low `MAXP`, the low run starts at least one chunk and drops the claim, and the
  others refill to their own `MAXP`; this repeats for every head chunk. A lower `MAXP`
  reserves no slots: it stalls the other runs once per chunk. So every run gets the `MAXP` of
  the live runs on its pod: the shape's value (Test pod above), or the lowered one while the
  pod is throttled. To throttle a pod, restart every live runner with the lower value
  (DAYBOX.md Housekeeping step 4); to end the throttle, restart every live runner with the
  shape's value. To give one run a smaller share, send it with `RUNMAXP` below `MAXP`: a
  runner at its `RUNMAXP` live chunks skips the gate (`runner.sh` main loop, the
  `nrun -lt RUNMAXP` test), so it writes no claim and stalls no other run.
  Heavy tests need no hand placement: list them with their weights and the runner keeps them
  apart. A claim lasts while its runner lives: a dead runner's claim is dropped at once, a
  claim whose pid was reused by another process after `CLAIM_MAX_S` (3600 s); a live
  runner's claim never expires by time. A chunk running longer than `UNIT_MAX_S` (21600 s;
  the longest measured chunk, 10080 s, ran on an uncontended pod) is killed and counts as a
  failure. Old-script runners ignore claims: never send a run to a pod they keep full. Every
  pack is preceded by the pod check (DAYBOX.md, Send a run step 3), and `pack.sh` runs from a
  checkout at `main` (the pod scripts come from that checkout, not from the commit under
  test), so runs of old and new scripts share a pod only when the run's report states it.
  Near `KILL_GB` (118 or 236 GB) kill this run's newest chunk and queue it again, at most
  twice per chunk; a third kill counts as a failure. Never signal PID 1.
- **Completeness.** Per test file, passed + failed + skipped + errors + xfailed + xpassed
  equals its collected count. A file whose totals differ, a file that failed to collect, a
  chunk that crashed and a chunk killed three times count as failures.
- Only the managing session rents pods, and only it creates or deletes the network volume
  (with the owner). Pods are used by the managing session and the run dispatcher's run
  agents, and deleted by the managing session, the nightly agent (`ci-nightly`, and
  `ci-day-big` as above) and the run dispatcher (idle pods between 22:00 and 08:00, as
  above). Use generic names. Verify every
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
  fast-tier lock state, and lists flags. Its paths come from its own checkout (or
  `NUZLINE_REPO` / `NUZLINE_SCRATCH`) and its parameters: `-Streams` (default
  `_scratch/manager/streams.json`), `-Lock` (the fast-tier lock, default
  `_scratch/fast_tier.lock`), `-MinFreeGB` (default 1), `-FixLock` (removes a lock whose owner
  is dead).
- The stream list `_scratch/manager/streams.json` (git-ignored) is edited by hand at every
  stream branch or worktree change: a JSON array, one object per stream, `name`, `branch`,
  `worktree` (relative to the main checkout) and an optional `logs` (relative folder of the
  stream's local `ft_*.log` files):
  `[{"name": "E1", "branch": "claude/e1-example", "worktree": "_scratch/e1/wt", "logs": "_scratch/e1/logs"}]`.
- `tools/ops/watch_flags.ps1` (same parameters, passed through, plus `-EveryMin` and
  `-HoldUntil`) runs the watchdog every 10 minutes in the background and exits
  (waking the managing session) only on a flag that a running test tier does not explain.
  `-HoldUntil` silences "has not committed" and "may have stopped" while streams wait on tests.
- A fallback check runs twice a day (section 7).
- A stream ends a turn only with a merge tip plus results, a real blocker, or a family list
  or plan. It never sends a "still waiting" note.

## 7. Scheduled prompts in the managing session

They live only in the session and expire after 7 days. Re-create them after every restart
or clear. Times are local cron expressions; the scheduler adds jitter. The nightly row and
the 30-minute stall check move to scheduled tasks that survive a restart (section 5,
"Scheduled tasks, not session timers"); once those exist, the session keeps only the other
rows.

| Cron | What |
|---|---|
| `47 9,21 * * *` | Fallback check: watchdog plus usage; silent unless something is wrong |
| `7 22 * * *` | Dispatch the nightly agent (`tools/ops/nightly_brief.md`): it runs the fast tier on `ci-day-big` when that is up, else on `ci-nightly`, and deletes `ci-nightly` itself |
| `57 22 * * *` | Delete the test pod if nothing is queued and no nightly runs on it, verify with a pod listing, log the cost |
| `37 23 * * *` | `bash tools/ops/offsite.sh` from the main checkout (section 12, Off-site copy); report only a non-zero exit (a refusal, a rejected branch, an abort or a failed push) |

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
  It keys the archive on the build's inputs (`tools/ops/table_key.py`) and reuses an archive
  with the same key instead of building; `_scratch/rent/tables_<short>.key` and
  `_scratch/rent/tables/index.txt` record commit and key. Run it for every pod-run commit;
  never reuse another commit's archive by hand.

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
- The watch scripts: `tools/ops/watch_flags.ps1` and `tools/ops/watchdog.ps1` with the stream
  list `_scratch/manager/streams.json` (section 6; command line in `_scratch/handoff/MANAGER.md`).

### Off-site copy

The private GitHub repo added as git remote `offsite` in the main checkout. Never push it to
`origin`, never make it public. It never receives ROMs, saves, save states, keys, tokens,
table archives or run outputs.
- `bash tools/ops/offsite.sh` (Git Bash, from the main checkout's root) pushes every local
  branch (`git push offsite --all`: no force, no delete; rejected branches are reported) and
  mirrors the state files into the clone `_scratch/offsite/state/` as commits on the branch
  `scratch-state`, which has no common history with the code. Deletions are mirrored inside
  that clone only; git history keeps old versions. `--dry-run` prints what would be pushed and
  copied and changes nothing. Temp files go to `_scratch/offsite/tmp/`.
- Refusals (exit 2), checked before any push: a push URL of `offsite` (every one, after
  `insteadOf`/`pushInsteadOf`) that names the same repo as any URL of `origin` (compared as
  lower-case `host/owner/repo`, so a trailing `/`, a missing `.git`, letter case and `git@` vs
  `https://` do not matter); a repo that `gh repo view <host/owner/repo> --json isPrivate` does
  not report private (also when `gh` is missing or not logged in); `remote.offsite.mirror` or
  `remote.offsite.push` set; a local branch named `scratch-state`; a leftover
  `_scratch/offsite/state/.git/index.lock`. The state clone gets the checked push URLs, `core.autocrlf=false`,
  `core.longpaths=true`, and the remote `offsite` when it is missing. Other exits: 1 error,
  3 offender in the state tree (nothing committed), 4 a branch rejected or a push failed.
  `NUZLINE_OFFSITE_TEST_LOCAL=1` lets the tests use local-path remotes; it never skips the
  check for a network URL.
- Runs after every push of `main` (section 2 step 9) and nightly (section 7).
- State files, paths relative to `_scratch/`: `handoff/`; the text files of `manager/` and of
  each stream folder `e<n>/` (rule sheets, family lists, plans, podlists, notes, small
  scripts and logs); in `rent/` the `*.md` notes, the pod id files (`POD_ID*`, `*POD*.txt`),
  `lists/*.txt` and the top-level helper scripts. Folder names are lower case: a top-level
  folder such as `E5/` or `Handoff/` is reported (`dir:not-lower-case`) and skipped. Folders
  named `src`, `tests`, `tools`, `docs`, `data`, `out`, `bin`, `home` (and the cache and
  emulator folders) are skipped at any depth, inside stream folders too.
- Never copied (the script enforces it and re-checks the whole state tree before every
  commit; an offender aborts the commit): worktrees and nested repos, `.cache/`, `.venv/`,
  `external/`, cache, code-copy and table-copy folders, ROM, save and archive files
  (`*.nds`, `*.sav`, `*.dsv`, `*.ds<n>`, `*.zip`, `*.7z`, `*.tar*`, `*.npz`, `*.npy`,
  `*.pkl`, `*.bin`, ...; extensions in any letter case), keys and credentials
  (`_scratch/rent/home/.runpod/`, `*.key`, `*.pem`, `*.ppk`, `.env*`, `*.env`, `local.toml`, `*.b64`),
  files with a NUL byte whatever their extension, files whose content looks like a key or
  token (private keys, Runpod/GitHub/OpenAI/AWS/Slack/Hugging Face/Google keys, JWTs,
  `user:password@` in URLs, `Bearer` tokens, `api_key=`/`apiKey:`/`PASSWORD=`-style
  assignments in any letter case), `rent/out/` and other run outputs, logs over 1 MB, files
  over 2 MB, anything inside a folder named `backup*`, copied `.gitignore`, `.gitattributes`
  and `.gitmodules` files. A content scan that cannot run (grep error) refuses too.
- Only the state files are scanned. The branches go as git has them: `git push offsite --all`
  does not scan branch contents, so what keeps keys, ROMs and saves out of a branch is the
  repo's own rules (`.gitignore`, never committing them), not this script.
- Diverged `scratch-state` (the push of the state clone is rejected, exit 4, because the
  off-site branch has commits the clone lacks, e.g. after a restore elsewhere):
  `git -C _scratch/offsite/state fetch offsite scratch-state`, then
  `git -C _scratch/offsite/state rebase FETCH_HEAD`, then run `bash tools/ops/offsite.sh`
  again. Never force-push it.
- Restore on a new machine: `git clone -o offsite <off-site repo> <project folder>` gives
  every branch (`git checkout <branch>`), then in it `git remote add origin <public repo>`.
  Then `git clone -c core.autocrlf=false -o offsite -b scratch-state <off-site repo>
  _scratch/offsite/state` and copy its files (not `.git`) into `_scratch/`. Worktrees, tables,
  `.cache/`, the runpodctl key and the inputs are rebuilt or supplied as in "What you must
  supply yourself".

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
