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
  compares (section 5). During a merge wave up to six; back to three or four when the merge
  line clears. Outside working hours delete every pod with nothing running or queued.
- **Run dispatcher.** A Sonnet session that sends, waits on and collects every pod run of the
  streams and the managing session's gate runs (section 5, Run dispatcher session).
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
9. On MERGE: `git merge --ff-only <tip>` into `main` (the tip already contains `main`; a
   train branch already holds the `--no-ff` merges), `git push origin main`, confirm with
   `git ls-remote origin main`, then
   `bash tools/ops/offsite.sh` (section 12, Off-site copy), send the stream the new tip and the
   non-blocking notes, and tell the user in at most three lines.

**Gate rules (every batch and every merge).**
- A known gap that departs from the decomp is a must-fix (`docs/decisions.md` section 3),
  also when the stream calls it unobservable.
- The merge-candidate message names the batch's code review file
  (`_scratch/<stream>/<name>_review_<n>.md`; a merge-gate review is
  `_scratch/manager/review_<name>_<sha8>.md`). No file, no gate run.
  (22 Nov 2026: a batch's code went unreviewed; only its sheets had been reviewed.)
- On every "ready" report the managing session runs `git merge-tree --write-tree main <tip>`
  and `git merge-base --is-ancestor main <tip>`. Conflicts surface before the review. A green
  gate on an old base is not a merge gate: when `main` gains engine changes, the stream merges
  `main` and reruns its gate.
- A stream sends no tip that has not passed one quick-lane run of
  `tests/data/test_game_constants.py` (it loads both games; at most 5 ids; section 5, Quick
  lane).
- A merge-gate review that runs no tests never replaces the pod run.
- Every merge gate runs every test that skips on the pod for missing inputs, locally by exact
  node ids (`_scratch/e3/rom_only_tests.md`), not only `tests/test_cli.py` (section 5,
  ROM-only run of a merge gate; exception: the skip rule there).
- An extract output change and its ROM reader change land together, with the ROM-only tests
  run before the candidate is sent.
- No flat fact list that depends on fight order or live count (heal, leave, pc gaps) beside
  `live_next`.
- A closing snapshot compare names generator drift whenever the batch touches a function that
  `tests/strategies.py` calls (drawn inputs include engine output).

**Outcome-law changes (`enumerate.py`, `outcome.py`).**
- A soundness fix widens a law mask only where its condition holds. Rows that turn
  SamplingOnly where the base was exact are a must-fix. The branch reports the exact rows lost
  at its tip.
- Before a laws merge candidate: the deep soundness sweep (`test_mask_sound` and
  `test_mask_sound_with_a_trainer_item`, explore profile, 15 seeds each: 2 nodes x 15 seeds = 30
  processes; DAYBOX.md, Soundness sweep). The nightly runs it on `main`.
- Every merge candidate that touches `enumerate.py` or `outcome.py` lists
  `tests/engine/test_enumerate.py::test_mask_sound 1 A` and
  `::test_mask_sound_with_a_trainer_item 1 A`.
- A candidate blocked by open classes is split: the part strictly sounder than `main` lands,
  with the open classes listed as must-fix in the ledger; the rest follows as its own
  candidate (candidates A and B, 22 Nov 2026).

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
- On green and MERGE from every review, the managing session moves `main` to the last tip
  (it contains the others) and pushes once.
- A failure goes to the stream that owns the failing test's area. If it cannot be
  attributed and fixed in one fix round, the train is dropped and the batches merge one by
  one.
- A batch that is ready alone and has no partner within one gate run merges alone.
- The managing session builds the train in its own worktree: `git checkout -B train/<name>
  <first tip>`, then `git merge --no-ff` of each later member. Never rebase. A train branch
  already copied off-site is never rebuilt under its old name: use a new name
  (`train/<name>2`), or `offsite.sh` rejects it.
- Ops, tools and docs branches gate on `main` directly, outside the train: review agent, a
  small gate (the four guards, `tests/test_differential.py 1 A` and the own tool tests of the
  branch), `git merge --ff-only`.
  When two such branches touch the same files, the second to land merges `main` and reruns its
  small gate.

## 3. Tests

- **Per batch and per merge:** CLAUDE.md, Batches and Merges.
- **Nightly:** the fast tier on `main`'s tip on a day volume pod (section 5), then the
  soundness sweep (DAYBOX.md, Soundness sweep). Tests that need the ROM, saves, the calc
  archive or the docs sheet skip there; run them locally afterwards (ROM-only ids only).
- **`slow` marks** only for a warm call-phase time over 60 s, never for a cold-compile wall.
- **User-facing CLI commands** must fit the owner's 32 GB PC. Check the measured peak of the
  `tests/test_cli.py` prep and solve tests after any change that moves laws or search memory
  (22 Nov 2026: a small prep needed 31.9 GiB).
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
the player's side. At `d2ec0fc1` (22 Nov 2026, `docs/kaizo/mechanics_backlog.md`): 234 of
387 battles playable; 47 missing groups on the opponent side plus 1 player-only. Re-run it
after each merge that changes the engine or tables. The drop in missing groups per day is the
rate used for estimates.

## 5. Rented test pods (Runpod CPU)

- **No engine tests on the PC.** The root `conftest.py` (`tools/ops/ramgate.py`) enforces it
  after collection. A local pytest run is allowed only when every selected test is ROM-only (its
  fixture closure holds a local-input fixture, `ROM_FIXTURES` in `ramgate.py`) or when
  `nuzline.engine` was not imported by the collected modules (a tool test). Any other run exits
  at once, naming the first disallowed ids: send them to the quick lane below. The tests listed
  in `tools/ops/day/pod_only.txt` (JAX compiles without the engine: `tests/test_spike_step.py`,
  `tests/test_jaxcache.py`; manager ruling 22 Nov 2026) are refused on the PC the same way,
  whatever they import. `-n` / `--dist`
  (pytest-xdist) is refused on the PC at configure time. A pytest on the PC that never loaded
  the gate (`--noconftest`, `--confcutdir tests`, also through `PYTEST_ADDOPTS`) cannot import
  the engine: `src/nuzline/engine/__init__.py` raises ImportError when `_pytest` is loaded on
  Windows and the gate module is not. Its four subprocess tests in `tests/tools/test_ramgate.py`
  (`test_a_pytest_without_the_gate_cannot_import_the_engine`) are win32-only, so no pod run
  covers them: run `tests/tools/test_ramgate.py` locally once per change to
  `src/nuzline/engine/__init__.py` or the root `conftest.py`. An allowed run
  waits for free RAM > 6 GB + weight (`tools/ops/day/pod/weights.txt`, longest key; no key =
  6 GB in an engine-free run, 20 GB when the engine loads), a message every 60 s, then takes the machine lock
  `<main checkout>/_scratch/ramgate.lock`: an OS lock on that file (`msvcrt.locking` on
  Windows, `fcntl.flock` elsewhere) held until the run ends, so one local run goes at a time.
  The kernel frees it the moment its holder ends, however it ends (a killed tool, the watchdog's
  hard exit, a power cut): there is no stale lock. The file stays; its line `<pid> <kind>
  <start time>` only names the holder, and a child process of the holder (pid and start time
  of the ancestor both matching, every parent link start-time checked, so a reused pid never
  counts) runs under its hold. Waiting on the lock (polled every 10 s, the message names the
  holder pid and the path) counts toward the same `NUZLINE_RAMGATE_TIMEOUT` minutes (default
  30). To clear a stuck lock, end the named process if it is no pytest or build of yours;
  deleting the file frees nothing.
  `--collect-only` is allowed without a wait or the lock. A ROM-only item weighs its figure in
  `tools/ops/day/rom_weights.txt` (measured locally, same format as `weights.txt`; a file marked `pod-measure needed`
  weighs 20), else 6 GB (the pod weight when the item has a higher key), so an unmeasured
  ROM-only item waits for more than 12 GB free (the PC idles at 16-18 GB free; engine ids that stay ROM-only keep an explicit 14 line, tag `engine-test-pod`). Local measuring pass:
  `python tools/ops/ramgate_measure.py IDS_FILE OUT_JSON [--skip SLOW_IDS_FILE]` (its header
  has the details): it holds the machine lock as kind `measure` for the whole pass, runs one
  file's ROM-only ids per process, strictly serial, each started when free RAM is above 12 GB;
  it polls every 0.5 s and kills that pytest tree when free RAM drops below 4 GB (the file is
  then marked `pod-measure needed`, weight 20); figure = ceil(max(1.25 m, m + 2)) GB of the
  peak summed RSS m of the tree, tag `local-measure-<sha7>-<yyyymmdd>`. Every gated run (and
  every `--build` command) has a watchdog thread too: below 4 GB free it kills the run's child
  processes and stops the run (KeyboardInterrupt, then a hard exit after 30 s), exit 15.
  The merge gate's ROM-only run passes exact ROM-only node ids, one file's ids per process,
  never a file path (a path makes the run mixed: the blocker goes on and the file's ROM-only
  engine tests fail). `python tools/ops/ramgate.py --rom-ids [file ...]` collects without a
  wait and prints one ready command line per file. `NUZLINE_RAMGATE_ROM_GATE=1` (or
  `--rom-gate`) concerns ROM-only ids only: it lifts the POD cut-off (weight 8 or more) of the
  `ramgate.py <ids>` check; the pytest gate ignores it, so it lets no refused run through.
  A refused run exits 13, a wait that times out 14, an unreadable free-RAM figure 16
  (apart from pytest's own codes 0-5); a `NUZLINE_RAMGATE_TIMEOUT` that is not a finite number
  of 0 or more counts as 30. `NUZLINE_RAMGATE_NO_WAIT=1` turns the RAM wait off (the
  refusal stays) only while a live ancestor holds the machine lock as the measuring harness
  (kind `measure`), which checks free RAM and runs the watchdog itself; otherwise it is ignored
  with a note. A pytest started by a test of a gated run is part of that run: the gate sets
  `NUZLINE_RAMGATE_DONE=<its pid>` and skips a run whose live ancestor (or itself, links
  start-time checked) has that pid; any other value is ignored. The pod sets
  `NUZLINE_ON_POD=1` (`unit.sh`, `setup.sh`, `quick.sh`, `sweep.sh`;
  `tests/tools/test_ramgate.py` fails for any script under `tools/ops/day/pod/` that starts
  pytest, or python importing the engine, without it in effect), which skips the gate on
  Linux only (on the Windows PC it is ignored). Local jobs that are not
  pytest run as the gate's child: `python tools/ops/ramgate.py --build [--weight GB] -- <command>`
  waits for RAM (default weight: the `tools/build_tables.py` line of `weights.txt`, 3, else 6, so
  more than 9 or 12 GB free) and the machine lock, runs the command under the lock and exits with
  its code; it never prints POD. On the pod (`NUZLINE_ON_POD=1`) or off Windows it runs the
  command directly: no wait, no lock, no watchdog. `tools/ops/build_tables.sh` wraps its build
  steps in it itself.
  A `tools.kaizo.*` run (it needs the ROM or the docs sheet, so it cannot go to the pod):
  `python tools/ops/ramgate.py --build [--weight GB] -- python -m tools.kaizo.<module> ...`.
  An agent's brief makes it wait on the gate (or `--build`), never print free RAM and start
  anyway. Measured local weights: `tools/ops/day/rom_weights.txt` (most ROM-only files 3 GB,
  unmeasured ones 6). A table build was measured at about 0.6 GB peak; `--build` weighs it
  3 by the `tools/build_tables.py` line of `weights.txt`. Engine work outside pytest on the
  PC (`tools.differential`, `snap_compare`, benchmarks, ad hoc scripts) has no gate: it goes
  to the pod.
- **ROM-only run of a merge gate (PC).** Skip it only when the diff touches no ROM-reading
  code or test (`tools/ops`, `tests/tools` outside `tests/tools/rom`, docs).
  1. Worktree at the tip: the managing session's `_scratch/manager/wt_train`,
     `git -C <wt> checkout --detach <tip>`. When it is missing:
     `git worktree add --detach _scratch/manager/wt_train <tip>`, copy `data/local.toml` in,
     decomp link as DAYBOX.md, Collect step 3.4. Then `bash tools/ops/build_tables.sh --here`
     there.
  2. Copy the template folder `_scratch/manager/romonly_batchi/` (`gen.py`, `driver.sh`,
     `run.ps1`, `waitram.sh`) to `_scratch/manager/romonly_<name>/`, replace `romonly_batchi`
     in the files, run `python gen.py` (exact node ids from `_scratch/e3/rom_only_tests.md`,
     one file's ids per process, slow ids left out), then `bash driver.sh` in the background;
     one Sonnet agent waits on it.
  These helpers are git-ignored local files on the owner's PC (local helpers below).
  3. Read every `*.log` last line: all `passed`; a skip only for an input the PC lacks
     (`kaizo_save_probe`). Write the result to `_scratch/manager/romonly_<name>.md`.
- **ROM + engine tests** (the five Roark/scoreboard ids tagged `engine-test-pod` in `rom_weights.txt`, listed in `pod_only.txt`): refused on the PC and never in a merge-gate ROM-only run; running them on the PC at a split close is against the owner's no-engine-tests order as it stands, so the managing session puts that question to the owner at the first split close (manager ruling, 22 Nov 2026); there is no standing slot.
- **Agents kill only processes they started**, by recorded pid, never by name pattern
  (22 Nov 2026: a stream's agent killed stray `pytest.main` processes that may have been
  another session's run).
- **Quick lane (engine tests, a few ids):** send the managing session a request of at most 5
  test ids at your tip; run name `q-<stream>-<sha7>`, ids file one per line. It starts at once
  on the pod with the most memory room, ahead of in-work lists; it waits only on the memory
  gate (up to 900 s, DAYBOX.md, Quick lane). The result you get back is pass or fail plus
  the first failure lines. Larger lists go on the normal pod list. A quick pass is never a
  substitute for a gate run (DAYBOX.md, Quick lane). While streams work, one pod is kept for
  the lane.
- **What may leave the PC:** source code at a commit (`git archive`) and the built tables
  listed by `TABLE_ALLOW` in `tools/ops/common.sh`. Never the ROM, save files, the calc
  archive or its extraction, the docs sheet or its extraction, or anything personal.
- **Nightly:** brief `tools/ops/nightly_brief.md`. Run it on a free day volume pod
  (`ci-day-big` with the network volume) that holds no push gate; when its run ends the
  nightly agent deletes that pod if no other run is active on it. Only when none is free it
  rents `ci-nightly`: at most $6 per run; the nightly agent itself deletes it within 4.5 hours
  of creation. The fast tier no longer fits a 128 GB pod in 4.5 h (22 Nov 2026: 208 of 317
  units). A tip that equals the last nightly's tip does not run again. A `ci-nightly` rented
  in the network volume's data center mounts the volume (Network volume below).
- **Scheduled tasks, not session timers.** The nightly dispatch (22:07) and the 30-minute
  stall check run as scheduled tasks, which survive a restart or clear of the managing
  session; they are not session timers (owner, 21 Nov 2026, `docs/decisions.md`). A new
  managing session first checks `list_scheduled_tasks` and creates what is missing: the
  nightly task daily at 22:07 local, whose prompt names the project folder, the brief, the
  skip rule, the day volume pods and the state file `_scratch/handoff/MANAGER.md` for its
  one-line result. Where no scheduled-task tool exists, the session prompts of section 7
  stand in for them and are re-created after every restart (list them first, no duplicate).
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
  hours: one pod per active stream plus one capture pod (section 1); during a merge wave up to
  six, back to three or four when the merge line clears. Pod N's id is in
  `_scratch/rent/day<N>/POD_ID_<N>` (pod 1's in `_scratch/rent/POD_ID`); a new pod takes the
  next unused N; a deleted pod's id file is renamed with the suffix `.deleted`. Each pod
  gets its own pod check before every pack (DAYBOX.md, Send a run step 3).
  The capture pod is a test pod with the network volume (`ci-day-big`, Bring-up in DAYBOX.md)
  in the 256 GB shape, so `snap.sh` runs with `SNAP_GATE_GB` 176 (on a 128 GB pod: 88). The
  managing session names it and writes it in `_scratch/handoff/MANAGER.md`. For the
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
  (`MAP_COUNT_LOW` in `times.txt`). Read the ssh user suffix in the create-pod answer: known
  faulty EU-RO-1 hosts (`vm.max_map_count` 65530) are listed in the managing session's state
  file (`_scratch/handoff/MANAGER_START.md`, pod section), not here; a pod on one is deleted
  at once. Placeholder trick (needs a free slot under the six-pod ceiling): create a
  placeholder that takes the faulty host, create the real pod, delete the placeholder
  (DAYBOX.md, Bring-up step 3). Keep it up while streams work; delete it when
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
  seeds only programs compiled on a CPU of its tag: the CPU model splits the cache, so only
  pods of one host tag share a warm cache (22 Nov 2026: the differential took 1832 s on a
  warm pod against 4112 s cold). Gates on volume pods use the shared cache; a pod of a new
  tag fills its own folder on its first gates. The low-space warning compares `du` of the
  volume in bytes with its size (30 GB of 10^9 bytes; `df` of the mount shows the whole storage
  cluster) and fires under 3 GB left. Every volume call
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
  on the pod: `pack.sh` / `snap.sh` payloads of a commit, or a committed script step at the
  sha under test (Script step below). Nothing ad hoc, nothing typed into an exec body, nothing
  sent from the PC by hand, also no read-only tally script (ruling 22 Nov 2026).
- **List size.** The number after a list line is ids per chunk, not units. Size a list in
  units with `python _scratch/manager/unitcount.py <list> [default|fullset]` (a local helper,
  below). A merge-candidate list targets about 120 units; the dispatcher refuses one over
  150. Read every run's unit count when the run is announced.
- **Push gate.** A run that gates a push of `main` gets a pod to itself or the first claim
  (the memory reserve lets a low-priority run block it). It is split across the pods with room
  from the start: parts `<run>`, `<run>-2`, `<run>-3`, ..., `tests/test_differential.py 1 A` as
  its own part. The gate is green when the union of its parts covers every unit with no
  failure.
- **Shell-level ops tests** (`tests/tools` tests that run pod shell scripts) run under WSL in
  a non-reaping PID namespace (`_scratch/offsite/wsl_run_ops_noreap.sh`, a local helper,
  below) before the pod send, as on the pod (PID 1 never reaps). A zombie counts as gone.
- **Local helpers (git-ignored, on the owner's PC only).** `tools/ops/offsite.sh` copies
  `_scratch/manager`, `_scratch/e<n>`, `_scratch/handoff` and `_scratch/rent` to the private
  off-site remote; `_scratch/offsite` is not copied. Rewrite a lost one from its line:
  - `_scratch/manager/unitcount.py <list> [default|fullset]`: estimates a pod list's units
    from the `collected.txt` counts of past runs in `_scratch/rent/out/*/results/` for that
    mark; prints units, ids and source per line, then the total.
  - `_scratch/manager/romonly_batchi/`: `gen.py` reads the "Per test" ids of
    `rom_only_tests.md`, leaves out slow ids and writes one `.args` file per test file plus
    `files.txt`; `driver.sh` runs them one at a time with the worktree's venv python, each
    after `waitram.sh` (waits for more than 10 GB free and no python over 2 GB); `run.ps1`
    starts one pytest, logs its peak working set and kills it under 4 GB free.
  - `_scratch/e3/rom_only_tests.md` (owner: E3): the ROM-only tests, a "Per file" list and a
    "Per test" list of `- \`<node id>\`: <why ROM-only>` lines; one file's ids are also
    printed by `python tools/ops/ramgate.py --rom-ids <file>`.
  - `_scratch/offsite/wsl_run_ops_noreap.sh <checkout> <test file>...`: runs `wsl_run_ops.sh`
    inside a new user, PID and mount namespace whose PID 1 is `sleep infinity`, lists the
    zombies left after the run, then kills the namespace.
  Committing them is owed (`docs/decisions.md` section 2).
- **Exec bodies stay short.** No exec-pod body that an agent must reproduce carries more than
  a short script (at most 1000 characters, `tools/ops/day/send_body.sh`; its only long token a
  64-hex sha256): payloads, scripts included, travel by file transfer (`runpodctl send` on the
  PC, `runpodctl receive` under a timeout on the pod), and the body starts them only after
  their sha256 matches the PC's. Long base64 in a body arrives changed (21 Nov 2026: three
  snapshot bodies in a row refused with `RUNSH_SHA_MISMATCH`).
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
  the nightly; it verifies the delete with `get-pod` (404), renames the pod's id file to
  `.deleted`, logs it in `DISPATCH.md` and sends the managing session one line. In working
  hours it only reports idle pods. Creating pods stays with the managing session.
- **Script step on a pod.** Besides pack runs, the dispatcher may run one committed script on
  a pod when a request names it: the script as committed at the sha under test (from that
  sha's checkout on the pod, never a file sent from the PC or typed into the exec body); it
  reads only that checkout and the built tables; it runs after the run's list has finished;
  its output is text written into the run's results folder and collected with the results.
  The exact command, sha and pod go in `_scratch/handoff/DISPATCH.md` before it runs.
- **Local pytest with four streams.** The pod is the place for every engine test; the PC runs
  only ROM-only and engine-free tests, under the gate above (one process; it waits for free RAM
  > 6 GB + weight itself, no manual check). Engine tests go to the quick lane.
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
- **Snapshot capture of `main`.** After every push of `main` that changes the engine or the
  tables the managing session starts one capture of the new tip on the capture pod, in
  `/work/snapmain-<short sha>` (seeds 0-199; commands: `tools/ops/day/snap.sh`, DAYBOX.md,
  "Snapshot compare"). Keep the newest two captures; delete an older one only when no
  compare reads it. A capture is never sent back to a pod from the PC. On pods with the
  network volume a passed capture of a `main` tip is stored on the volume, keyed by its full
  sha and the table key of its tables, and is never overwritten; a capture of a key the
  volume holds, and a compare whose capture is not on its pod, fetch it from there instead
  of re-capturing (owner, 21 Nov 2026). Only a capture lost with a pod without the volume, or
  one the volume does not hold, is re-made on the next pod.
- **Capture pod.** Snapshot captures and compares run on the capture pod only. No pack run
  is sent to it while a capture or compare is going there. One snapshot job at a time per
  pod: each job gates only on its own parts, so two at once could overrun memory together.
  `run.sh` enforces it with a pod-wide lock (`/work/.snaplock`, taken after `uv sync`,
  released on every exit, a TERM included; a lock whose pid no longer runs a `run.sh` is
  removed); a second job sent meanwhile waits for it, printing one `lock wait` line a minute
  (DAYBOX.md, Snapshot compare). A job runs its parts in parallel
  (`SNAP_WORKERS`, default 16, each part its own process; `tools/snap_compare.py`, Workers):
  a capture takes 17-24 minutes (measured), a compare about 30 minutes (estimate; the longest
  part, `roark.rank`, takes 15-22 minutes alone). Serially
  (`SNAP_WORKERS=1`, or a commit whose `snap_compare.py` has no `--workers`) a job takes
  85-115 minutes.
- **Snapshot compare of a merge candidate.** Compare only, against the
  capture of the `main` tip its branch contains, with `--allow-new <the stream's list>`
  (left out when the list is empty). Never compare a merge candidate against an older base
  once `main` was merged into the branch.
- **Capture identity.** The capture a candidate compares against has the candidate's base
  engine and packed tables. Check that by sha and packed-table content, never by table key:
  the key hashes tool code and tree state, so it differs across machines and after tool-only
  changes. After every landing that changes the engine or tables the new `main` capture
  becomes the standing capture (stored on the volume).
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
- **Gen draws and source literals.** `draw_inputs` draws with Hypothesis's local-constants pool
  empty (`snap_compare.no_local_constants`), so a new literal of the project's own imported
  modules (Hypothesis `ConstantVisitor`'s filter, `hypothesis/internal/constants_ast.py:113-145`:
  integers with |v| >= 100, non-zero non-infinite floats (NaN too), strings of up to 20 characters, bytes up to 50; a module
  under a `test` / `tests` folder or a `test_*` / `*_test` file is skipped, `constants_ast.py:260-283`)
  does not shift the `gen` part's draws. `tools/differential.py` draws with the pool on, so
  its inputs at a seed differ from the snapshot's. This change altered every gen draw once: a main capture
  made before it needs a fresh capture of main with this code before the first compare.
- **Snapshot seeds in tests.** No test may redraw snapshot seeds through the live generator; a
  test that needs a compare state pins the state.
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
  files, about 40 for other files. An `A` run of `tests/ai/test_kaizo_policy.py` takes two
  lines, `tests/ai/test_kaizo_policy.py 57` and then `tests/ai/test_kaizo_policy.py 2 A`: the
  first cuts the fast ids, the second only the slow and `fullset` ids left (an id an earlier
  line queued is skipped). Its slow ids (117 of 1247, 21 Nov) cost 60-900 s each on the pod
  (each new layout traces and compiles for about 40 s), so a 57-id `A` chunk ran over 2 h
  (run train-e3-82e5446, u_098: 25 ids in 2.2 h) while its fast ids cost 0.3-0.5 s after the
  first of each test function. Measured chunk peaks (7183 chunks of 120 pod runs, 20 Nov):
  median 7 GB, 90th percentile 12 GB, 99th 24 GB, largest 92 GB. Every chunk has an expected peak: its heaviest id in
  `tools/ops/day/pod/weights.txt` (every file any run measured, at its largest chunk and
  at least 4 GB, plus hand-kept per-test lines, each citing its run; `UNMEASURED_FILE_GB` 20 GB for a
  file never measured; a test that passed alone in a 1-id chunk at least 4 GB under its
  file's line has its own line; a multi-id chunk weighs the max over its ids), raised to the list line's `<GB>G` field when that is larger.
  Refresh `weights.txt` (DAYBOX.md, Send a run step 2) after every merge wave and after any
  change that moves a test's memory by more than 20%. A
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
  above). Use generic names. Verify every delete with `get-pod` on its id (404; a full pod
  listing is too long). Each spending cap is the user's explicit approval; anything larger or
  longer needs a new approval.
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
rows. Before creating a row, list the session's prompts: no duplicate. Night pod deletes are
the dispatcher's (section 5, Run dispatcher session); the former 22:57 pod-delete row is
gone. Stall check: update `_scratch/manager/streams.json` first, then start
`tools/ops/watch_flags.ps1 -EveryMin 30 -Streams _scratch/manager/streams.json` in the
background (section 6).

| Cron | What |
|---|---|
| `47 9,21 * * *` | Fallback check: watchdog plus usage; silent unless something is wrong |
| `7 22 * * *` | Only where no scheduled task holds it: dispatch the nightly agent (`tools/ops/nightly_brief.md`): it runs the fast tier and the soundness sweep on a free day volume pod (`ci-day-big`) holding no push gate, else on `ci-nightly`, and deletes `ci-nightly` itself |
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
- Heredoc appends to state files are quoted (`<<'EOF'`), so no backtick or `$` runs. Every
  timestamp written comes from `date` (PC) or `date +%s` (pod).
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

**Read in this order** (a new managing session: follow "Taking over as the managing session"
below instead of this list): `CLAUDE.md`; this file; `docs/decisions.md`;
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

- `MANAGER.md`: the managing session (a "Current state" block at the top, then dated log
  entries, newest last). `MANAGER_START.md`: the entry brief for a new managing session
  (hard rules, standing approvals, state, routine jobs, first actions). `E1.md`, `E2.md`,
  `E3.md`, `E4.md`: one per work stream. `DISPATCH.md`: the run dispatcher (standing orders
  at the top, runs in flight and results at the end; large: read its top and tail only).
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

### Taking over as the managing session

Read order (first 15 minutes):
1. `CLAUDE.md`.
2. `_scratch/handoff/MANAGER_START.md` (when present), then `_scratch/handoff/MANAGER.md`:
   its "Current state" block, then every dated entry newer than the brief. Newer entries
   supersede the brief.
3. `_scratch/handoff/DISPATCH.md`: its standing orders at the top and its last 30 lines only.
4. `_scratch/handoff/E1.md` to `E4.md`: the current section at the top of each, or the tail.
5. This file's sections 2, 5 and 12 when a job sends you there; `tools/ops/day/DAYBOX.md`
   only before pod work; `docs/decisions.md` before answering any approval or rule question.
Never read whole: the old managing-session log, agent transcripts, run logs.

First actions, in order:
1. Verify git against the state files: `date`; `git log --oneline -10 main`;
   `git rev-parse origin/main`; `git status -sb`; `git branch -v` for every branch the state
   files name. Where git and a file disagree, git wins: note it in `MANAGER.md`.
2. Sessions: list them with the desktop app's session tools (`list_sessions` with linked
   sessions); check that each stream and the dispatcher is alive; read each one's context use
   (`get_usage` per session) and clear a stream past 30% at its next commit point. Old
   session ids in a file are history: never message a retired managing session.
3. Pods: one `get-pod` per id file in `_scratch/rent/`; read each with one pod-check exec
   before use (DAYBOX.md, Send a run step 3); outside working hours delete any pod with
   nothing running or queued (DAYBOX.md, Delete); in working hours keep one per active
   stream plus the capture pod (section 5, Test pod); count the pods up and their cost per
   hour.
4. Recreate what died with the old session: the nightly trigger (section 5, Scheduled tasks;
   check `list_scheduled_tasks` first), the session prompts of section 7, the stall check
   (section 6) after updating `_scratch/manager/streams.json`.
5. Collect runs whose beacons failed by hand (DAYBOX.md, Collect); land whichever queued
   candidate has a MERGE review and a green gate (section 2 step 9).
6. First message to the owner, five lines: `main`'s tip and what landed since the handoff;
   pods up and cost per hour; what lands next; anything that needs the owner; that a new
   managing session has taken over.
Then wake only for owner messages, stream results, the stall check and scheduled tasks.

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
6. Start the four stream sessions (E1-E4, model Opus) and the dispatcher (`dispatch`, model
   Sonnet): folder the main checkout, each with its pick-up line. Record the new session ids
   in `MANAGER.md`.
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
powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"   # free GB; the ramgate waits for > 6 + weight
```
Then confirm in the session that the skills `superpowers:*` and `runpod:*` are listed and
that the Runpod tools answer. Report to the owner in five lines: what matched, what was
missing, whether Runpod answered, free RAM, and whether a pause is in force.
