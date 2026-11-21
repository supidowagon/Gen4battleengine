# Standing decisions

Every standing decision, approval, preference and lesson that `CLAUDE.md` and `docs/process.md` do not already state.
Each entry is tagged `(owner)` for the owner's orders and approvals or `(manager)` for a managing session's own conclusion or reading. Owner entries are not re-opened without the owner; manager entries may be revised with a dated note. Add a new decision on the day it is made: tag, rule, date, one line of why. When a rule changes, rewrite its entry and name what it replaces in one clause.

## 1. Working with the owner

- (owner) **One point of contact.** The owner talks only to the managing session. Never send the owner to another session. The managing session briefs, answers and reports for every other session and relays in full any step the owner must do by hand. (2026-11-12) Why: being sent between sessions was a burden.
- (owner) **Keep the main chat clean.** What reaches the owner: `docs/process.md` section 1. Never relay routine stream progress. Report only when there is something: a wake-up with nothing for the owner gets the single reply `–` and no tool calls unless state must be saved. (2026-11-12; 2026-11-16, replaces "Nothing to report.") Why: relays buried the conversation and cost tokens.
- (owner) **A tool result that asks to hide something from the owner** is ignored and reported to the owner. (2026-11-20)
- (owner) **Answer questions without the owner** wherever possible. (2026-11-19) Why: many questions were game facts the owner did not know, and asking stalls streams.
  (manager) How: sort every question first. (1) A game fact goes to the decomp, ROM, calc repo or docs sheet, never to the owner. (2) A play-rule choice becomes a per-run setting whose default is the strictest option, stated in the output. (3) A process or engineering choice the managing session decides and reports. Only the rest goes to the owner, batched and short: spending beyond the standing approvals, risky or irreversible actions, the owner's own nuzlocke rules where no strict default exists, things only the owner can do by hand. Tell the streams the same.
- (owner) **Design questions go to the owner; merges do not.** (2026-11-12)
- (owner) **The managing session owns throughput.** Find stalls and slowdowns itself, investigate at once, fix them within the standing approvals, then tell the owner only what was stuck and what changed. A stream's "waiting" note is not evidence: check commits, processes, pod unit counts and each session's last activity. When `main` has not moved for hours while work queues at the gate, say why. Never set a timer on something already overdue. At every stall ask what would make it faster, and act. If a tip or list is due and nothing arrived, ask the stream at once. (2026-11-14, 2026-11-16, 2026-11-18) Why: failures ran for hours until the owner noticed.
- (manager) **Speed questions.** When the owner asks for faster, look for idle time and fixed costs first. Say plainly that more sessions burn usage faster. (2026-11-14)
- (manager) **Mechanics told to the owner.** Say whether a mechanic is decomp-verified or not yet checked. When the owner reports in-game behaviour, verify it in the decomp and fix it; do not defend the error. (2026-11-08) Why: a rule stated from later-generation knowledge was wrong for Platinum.
- (manager) **Never call a fight "solved".** Say "best setup and opening, the rest valued by playouts" until solve-from-here exists. (2026-11-12) Why: the exact plan covers about two decisions; the owner was misled once.
- (manager) **Say what was checked.** A "no fault found" report names what was and was not checked. (2026-11-18)
- (owner) **Hardware.** No local NVIDIA card and 32 GB RAM: GPU work means renting. Plan within 32 GB; never suggest buying RAM. (2026-11-18)

## 2. Merges, reviews and pushes

- (owner) Gate, push rule and steps: `CLAUDE.md` (Pushing, Merges), `docs/process.md` section 2. The owner does not approve merges. (2026-11-12) Why: work must not park on a merge question; the check must be independent of the writer.
- (owner) **Every merge keeps its fresh review agent**, also for a tiny test- or doc-only change. Do not propose skipping it unless the owner raises it. (2026-11-18) Why: the owner values the independent check over its tokens. The re-review exemption in `docs/process.md` section 2 step 8 is unchanged.
- (manager) **Review agent brief.** Give it the exact commit range, covering every non-merge commit, and the criteria. Review every fix commit that touches `src` or `tools`; a tests/docs-only fix commit follows `docs/process.md` section 2 step 8. Give the brief a tool-call budget. (2026-11-18) Why: a "remove this" fix note once removed four needed refusals.
- (owner) **Review agents run on Opus**: pass the model on the Agent call. (2026-11-16)
- (owner) **Lean reviews.** One reviewer per task and one single-reviewer final review per branch. No multi-reviewer or adversarial review workflows. (2026-11-12) Why: review ceremony was most of the token cost.
- (owner) **Pushes.** Standing yes since 2026-11-15 (`CLAUDE.md`, Pushing); replaces "ask before each push" (2026-11-12). Why: the push is the off-PC backup.
  (manager) Scope: `main` only, no force push, no other branch; anything else needs the owner's go.
- (manager) **Before any public announcement** add a licence file and README credits: the pret decomp, the Kaizo hack's author, PKCalc, the Kaizo docs spreadsheet. Owed. (2026-11-12) (owner) The owner wants the tool public eventually.
- (owner) **Repo split.** One repo until the engine is complete. Training code (`src/nuzline/learn`) uses the engine only through a small public interface. When the engine is complete, freeze that interface and move training to its own repo with history. The UI is its own repo from its start, reads only engine output, and is last on the roadmap. Never create repos or move files between repos without the owner's go. (2026-11-13) Why: the battle state still changes every milestone.

## 3. Streams and batches

- (owner) **Streams.** E3 approved 2026-11-13; more streams only if merges stay clean (2026-11-13). E2 owns the doubles/tag/multi format (2026-11-17). Ownership per stream: `docs/process.md` section 1.
- (owner) **Fourth stream E4** for the doubles side: the doubles trainer AI, the observation at four battlers, the two-trainer and multi fight data, search with targets, prep with two actives; work that does not touch E2's mechanic batches. E2 proposed the split, the managing session checks overlap; E4 starts with rule sheets and builds once E2's D6-D7 is in `main`. (2026-11-21)
- (manager) **Family lists.** Before approving one, have the stream rerun the backlog on the newest unmerged tips of the other streams. Approve only groups that cannot overlap; hold the rest. (2026-11-19) Why: two streams nearly built the same groups.
- (manager) **A "known gap" that departs from the decomp is a must-fix**, also when the stream calls it unobservable. Rule sheets and reviews decide observability, not the builder. (2026-11-19)
- (owner) **In-work tip message**: three lines: branch and sha; table inputs changed yes/no; path of `_scratch/<stream>/podlist.txt`. Builders run the guard scans, the dispatch check and the duplicate-name check before every tip. (2026-11-18)
- (owner) **One in-work pod run per stream.** No new tip is sent while that stream's run is going, unless the run has already failed. (2026-11-18; roadmap item 4, 2026-11-20; 2026-11-21, replaces "unless it fixes a reported failure; send new list lines only")
- (owner) **Run lists.** An in-work tip gets a short list: new and changed test files first, the tests of the touched functions, the four guards; about 30 minutes on a pod, at most about 40 units; no `tests/test_differential.py`, no whole-suite files. The full delta list runs only on a merge candidate: the areas the change touches, the four guards and `tests/test_differential.py 1 A`, sized for about one hour on a pod alone (about 120 units); the nightly covers the rest; the `fullset` list as before. Exact orders: `docs/process.md` section 5. (2026-11-21; replaces the manager's "Delta runs" entry of 2026-11-19, in-work lists under about 150 units)
- (manager) **Ten-minute rule.** A local test run is one file, with more than 6 GB free, under 10 minutes. Past that, the stream commits and sends the tip to the pod. (2026-11-19) Why: agents waiting on local runs stalled a merge train.
- (manager) **Skipped is not passed.** Tests that skip on the pod for missing inputs run locally on the gate tip; the tip message says so. (2026-11-19)
- (manager) **Fixed-seed failure on unchanged code**: triage first with a read-only Opus agent from logs, code and decomp. (2026-11-19) Why: both such failures on 19 Nov were test bugs.
- (manager) **Snapshot compares.** A result with almost every part diverged is a tool or input fault until proven otherwise; ask the stream what the tool does before guessing. A captured field missing in the checkout needs an UPGRADES row in `tools/snap_compare.py` (reshapes too); `--allow-new` is for fields new in the checkout only. Run an early per-stream compare against the `main` capture before a train's shared compare. One capture serves several compares: bisect with compare-only runs against it. (2026-11-19)
- (owner) **Allow-new lists** for a snapshot compare are derived from the code and pre-checked locally before the compare is sent (`docs/process.md` section 5, Allow-new list). (2026-11-21) Why: a compare of batch H failed on an incomplete list and had to run again.
- (manager) **Ledgers** keep a RESUME section so a cut-off loses nothing. (2026-11-14)

## 4. Tests

- (owner) Cadence: `CLAUDE.md` (Test tiers), `docs/process.md` section 3. Do not bloat the work with tests. (2026-11-13) Why: full suites almost never caught bugs; reviews and citations did.
  (manager) Write no hand-written test where the two-engine differential test already covers the rule. Keep per-batch review, decomp citations and mutation checks.
- (owner) **Real-game checks** are the replays of the owner's two recordings: the recorded Roark fight at the Roark split close, and the Elite Four + Champion notes at the last split close. No other replay until the engine is complete. (2026-11-13)
- (owner) **Vanilla coverage sweep** after the last Kaizo split: run the coverage check over all vanilla trainers and model what it refuses. Do not raise it before then. (2026-11-17)
- (owner) **Compute during pauses.** Before a usage pause, start the long CPU-only jobs so results wait at the reset. Never run two full suites at once on the PC. (2026-11-12)

## 5. Pods and spending

Rules: `docs/process.md` section 5. Steps, including stopping a run, runner spacing and pod disk: `tools/ops/day/DAYBOX.md` (Housekeeping).

- (owner) **Standing approvals.**
  - Nightly: on Runpod, at most $6 per run on its own pod. (2026-11-17)
  - Test pods: no daily limit (2026-11-18 evening; replaces $10/day of 2026-11-17 and $15/day of 2026-11-18). The 32 vCPU / 256 GB memory-optimized shape at $1.76/h is approved (2026-11-19). Up to six such pods at once (2026-11-20; replaces two pods). Delete a pod when nothing is queued on it. Report the day's pod cost to the owner.
  - Upload scope: the `TABLE_ALLOW` list in `tools/ops/common.sh`. (2026-11-18)
  - A split-close full run on Runpod: agreed in principle; quote the cost to the owner first. (2026-11-17)
  - More pods, an always-on pod or a GPU: a new yes each time.
- (owner) **No idle pods.** Manage the pod count actively: delete a pod once nothing runs or is queued on it, and report it in the morning summary. Never delete the last pod holding a finished capture while a compare of it is pending. (2026-11-20) Why: no wasted Runpod money.
- (owner) **One pod per stream.** In working hours: one test pod per active stream plus one capture pod for captures and compares (about four pods at $1.76/h each); the six-pod ceiling stands. Nights and quiet hours: delete every pod with nothing running or queued. (2026-11-21)
- (owner) **Capture pod.** Snapshot captures and compares run on the capture pod; pack runs stay off it while one is going. Their parts run in parallel (`docs/process.md` section 5, Capture pod). (2026-11-21) Why: a serial job took 85-115 minutes while most cores idled.
- (owner) **New pods: `vm.max_map_count`.** Check a new pod before its first use: 1048576 or more; a pod with 65530 is deleted and another rented. The pod check refuses such a pod. (2026-11-21) Why: on a pod with 65530 JAX aborted units with rc 134 when it wrote a compiled program, which looked like a memory fault. (manager) Why 1048576 and not a lower floor: the failing host had 65530 and every working pod had 1048576 (failing host: 65530, four rc 134 aborts on pod 2; working pods 3 and 4: 1048576, none on pod 3's 30-odd runs; 21 Nov). No pod with a value in between has been tested, so a pod under 1048576 is refused until one is tested and passes.
- (manager) **Load per pod.** On a 256 GB pod at most four test runs. At most one snapshot job per pod, enforced by `run.sh`'s pod lock (`docs/process.md` section 5, Capture pod); a second job waits for the first. Before adding runs read free memory (`status.sh`, last line); under 15 GB free add nothing. A stream has one in-work run at a time and sends no newer tip while it runs (`docs/process.md` section 5, One in-work run per stream), so no older run of a stream is left to cancel; a compare is sent only for a merge candidate. (2026-11-18, 2026-11-20; one snapshot job 2026-11-21) Why: `MAXP` limits chunks, not memory; an overloaded pod killed units without a failure line; each snapshot job gates only on its own parts, so two at once could overrun memory together.
- (manager) **Pod stock.** The stock listing can be wrong for hours. Retry every 30-60 minutes; try without naming a data center. (2026-11-18)
- (manager) **PC memory.** Tables are built locally, so low PC memory blocks pod runs. Reuse a tip's tables only when the diff cannot change them (`tools/ops`, docs, tests, docstrings); engine changes need a real build. Check free RAM at session start and ask the owner for a reboot early. (2026-11-20)
- (manager) **Never `Start-Process bash` from PowerShell**: it starts WSL and takes about 2 GB. Run table builds through Git Bash. (2026-11-20)
- (manager) **Timestamps** come from the pod (`date +%s`) or the PC, never from a running estimate. (2026-11-20)
- (manager) **Pod procedures are scripts.** Road-test a new procedure with an agent told to report every step that does not work as written. (2026-11-19)
- (manager) **Start the pod run as soon as a tip arrives**, in parallel with the merge check. (2026-11-18)

## 6. Usage, context and models

Rules: `CLAUDE.md` (Usage), `docs/process.md` section 8.

- (owner) **The owner handles the usage limits.** Never report usage percentages, run-out dates or pace unless asked. Never slow or pause streams to save usage. Report efficiency, not consumption. (2026-11-14, 2026-11-17) (manager) A stream actually blocked by a limit is one line.
- (owner) **Context.** Limits: `CLAUDE.md`, Usage (2026-11-18; replaces about 200k tokens of 2026-11-16 and about 100k of 2026-11-17). Backstop: `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50` in the user settings' `env` (`docs/process.md` section 12). (manager) Read each session's context with the session usage tool at every stream commit point.
- (owner) **Self-clear only with the owner present.** The managing session clears or compacts itself only right after an owner message, after rewriting its state file, and asks the owner to send one word afterwards to restart it. Never while the owner is away. (2026-11-15) Why: a clear kills the session's schedules and nothing wakes it. (manager) After the restart, re-create the watch loop first.
- (owner) **Lean session context** (all sessions). Waiter and collector agents block in one background command until their run ends and end no turn before that: no timers, no "still waiting" notes. One report per finished run; a first-failure note is the only interim. Answer a no-news ping in a few words. Streams apply the same to their agents and send results, not narrative. (2026-11-20) Why: keep every session's context free of bloat.
- (owner) **The managing session must not become a usage burn.** (2026-11-16) (manager) How: a routine stream note gets at most one tool round. Never read large files, logs or agent transcripts whole; take tails and greps. Replies stay short; no repeated status tables unless asked.
- (owner) **Models.** Well-scoped work (clear brief, named files, a test that says done): Opus. Open-ended or broad work (design, whole-project review, diagnosis with no known cause): Fable. Mechanical jobs checked on the spot: Sonnet (`CLAUDE.md`, Usage). Managing the model is the managing session's job: set it explicitly on every session and Agent call (an agent otherwise inherits the caller's model) and re-check it at every phase change (design finished, building from a plan: switch to Opus). (2026-11-12, 2026-11-13, 2026-11-16) Why: Fable is the costliest model.
- (owner) **The managing session runs on Fable**; the owner chose not to switch it to Opus yet. The Fable usage bar is on half the scale: Fable's share of the week = (Fable bar % / 2) / all-models bar %. (2026-11-17) (manager) If that share is over about a quarter, propose switching the managing session to Opus.

## 7. Roadmap and direction

- (owner) **The end goal is a trained player.** The JAX engine exists so a self-taught best player can be trained on it. Never propose dropping or indefinitely deferring the learned player. (2026-11-12) (manager) Keep the engine batchable and trainable. Strength is win rate against the fixed, known trainer AI, scored on the player scoreboard (same fights, same seeds) and against exact answers where a fight is small enough.
- (owner) **Engine first.** No training runs and no GPU spend until the engine is complete and the owner says to start training. Do not propose training alongside. (2026-11-13; replaces the two-track plan of 2026-11-12) The training proof of concept is parked on branch `claude/training-t1-c74e08`.
- (owner) **Split by split, Kaizo first.** A split is one stretch of progression, gym leader to gym leader, with its trainer list from that game's own data. A split covers every trainer in it, including double, tag and multi battles. Start each split with a coverage gap report; run the vanilla gap report for the same split alongside. (2026-11-09)
- (manager) **The backlog counts battles**, not trainer rows: progress is the number of real battles that are fully playable (`docs/process.md` section 4). (2026-11-21; the owner said to go on the managing session's recommendation and may overrule)
- (owner) **Line quality.** Order of work: `docs/design.md` section 8, item 4a. (2026-11-12)
- (owner) **Prep starting state** (Pokemon pre-damaged, pre-slept or pre-poisoned before a fight) is a prep task, not engine work. (2026-11-14)
- (owner) **Prep ranking.** With a resource the owner has allowed, rank by win chance, then expected deaths; resource cost is the last tie-break and is still listed. A resource the owner has not allowed is never spent. (2026-11-03) Why: a cost-first tie rule once picked a 98.9 % team over 99.2-99.9 % ones. (manager) No tie rule, halving key or leader choice may let cost beat a higher win chance.
- (owner) **Open ideas, not approved** (agenda items; raise when the roadmap reaches them): plan for the whole run with a per-Pokemon weight for deaths; two tiers, a full solver for bosses and a labelled threat sheet for route trainers; whether the tool serves only the owner's run or other players too; exact solving of small fights by grouping damage rolls by break points. (2026-11-12)

## 8. Data sources and owner rulings

Rules: `CLAUDE.md` (first two bullets), `docs/kaizo/pipeline.md`.

- (owner) **Record the data-source decisions so the same mistakes are not repeated.** (2026-11-14) (manager) Before calling data missing, or accepting a stream's "unknown", check that the ROM, the calc repo and the docs sheet were all consulted, and say which. A ruling that excludes data ("sheet only", "non-blocking") is to be re-verified, not settled. Why: partner teams and 60 real battles were dropped this way.
- (owner) **Partner teams** for tag and multi battles are in the calc repo. Look there before calling one unknown. (2026-11-14)
- (manager) **Plan and survey statements of a mechanic are hypotheses** until checked against the decomp code. (2026-11-08)
- (owner) **Kaizo data rulings** (2026-11-05): the Galactic-split level cap is 100. A bare "(SUN)" sheet header means permanent sun; keep that reading in one easily edited place.
- (manager) **`data/local.toml`** holds the input paths. Sessions never edit it.

## 9. Gauntlet rules (owner's play rules)

- (owner) **The owner sets the rules per gauntlet at planning time.** The engine models them as independent options per gauntlet: healing between fights (none, bag items, full heal), team change (none, reorder and held items, swap members), more as they come. What the ROM forces is a constraint or default, never the whole rule, and never rejects a rule the owner states. Do not ask the owner to fix one rule per group in the data. (2026-11-18)
- (manager) **Walk-out, healer and order facts** come from the decomp and ROM, never from questions to the owner. (2026-11-19)
- (owner) **Elite Four + Champion.** Normal Elite Four, except the player may swap in one Pokemon, once; the swap is permanent. Bag healing after each member and TM/move changes between fights are allowed. Bag healing is limited only by what the player carries. The choices are entered per run; there is no saved default. (2026-11-18, 2026-11-19) (manager) So team change has a counted form: a bench of N extra Pokemon and a swap budget for the gauntlet (here bench 1, budget 1). Prep plans 7 Pokemon and when, if ever, to swap.
- (owner) **When a group's rules are not set**, use the strictest settings and say so. (2026-11-19)
- (manager) **Gauntlet positions: strict default.** No healing and no team change between the fights of a gauntlet position until the owner gives that gauntlet's rules; the owner's real rules are per-run settings. (2026-11-21; the owner said to go on the managing session's recommendation and may overrule; the owner confirmed the strict default the same day)
- (manager) **A fainted multi-battle partner counts as a death** by default. It is a per-run setting; the owner may switch it off. (2026-11-21; the owner said to go on the managing session's recommendation and may overrule)
- (owner) **Rare Candies and evolution stones with healing off**: not allowed for now. (2026-11-19)
- (owner) **Move changes between fights and the deaths allowance** (whole gauntlet or per fight): the owner decides per run. (2026-11-19)
- (owner) **Level cap inside a gauntlet**: a Pokemon that passes the next fight's level cap finishes the current fight and takes part in no later fight. (2026-11-19)
- (owner) **Opponent order**: a per-gauntlet setting, default fixed. (2026-11-18)
- (owner) **Level-up and evolution choices** (learn, skip or replace a move at level-up, also mid-fight; evolve or cancel) are decisions of the line, in single fights too. Never treat them as declined by assumption. (2026-11-18)

## 10. Owner's recordings

- (owner) **Elite Four + Champion notes**: five text files (one per opponent) and the gauntlet save, in the owner's recordings folder (path in `_scratch/handoff/`). Read the notes in place. The save is never committed and is copied only into the git-ignored `.cache/` (`CLAUDE.md`). Each file lists a pre-fight setup and one line per turn; no HP numbers or damage rolls. (2026-11-13)
  (manager) Uses: the replay check for the last split (every AI move seen must be one the model allows), known winning lines as benchmarks for the line player and the trained player, the save as the endgame box for prep search. Usable only once that split's mechanics exist.
- (owner) **Pre-Roark Kaizo save** (all bag items available): path in `data/local.toml`. Saves follow `CLAUDE.md`. (2026-11-05)

## 11. References

- (owner) **Jaxcalibur** (`https://jaxcalibur.github.io/`): the project's original inspiration. A pure-JAX masked battle engine plus a transformer trained by self-play PPO; search added little over the network. (2026-11-12)
  (manager) Lessons: the network is most of the strength, so do not over-invest in exact search; sample chance instead of enumerating it and bin HP; feed damage tables and the trainer AI's exact move distribution as inputs; Nuzline's case is single-agent (no hidden information, fixed known opponent).
- (owner) **pkmn.ai** (`https://pkmn.ai/`): a directory of Pokemon battle AI projects and papers, to consult when training starts. (2026-11-17)
  (manager) Read the MissingNo.Zero and Metamon papers and the `/research/` list first. Most entries are player-vs-player with hidden information; the network design, value training and search-with-chance parts transfer best.

## 12. Follow-ups the owner asked to keep

(owner) Keep this list; review it when the owner asks for status or the roadmap; remove an item when it is done or the owner drops it. (2026-11-16)

- Each morning: check that the nightly ran and what it returned.
- At each merge note and in each nightly's durations list: check that new fast-tier tests keep the cap in `CLAUDE.md` (Test tiers) or are marked `slow`.
- Nightly fast tier under 3 hours: slowest tests to the slow set, shared compiled programs, JAX on-disk compile cache (measure first).
- Split the nightly into 2-3 parallel parts once memory peaks are known.
- Shrink the nightly to cross-area tests, with the full run only at split close (owner agreed 2026-11-17).
- Sonnet for mechanical subagent jobs: check merge notes that it is used.
- The Kaizo table rebuild writes a ROM copy to `.cache/rom`: make sure merge-check agents remove their worktrees so copies do not pile up.
- Roadmap items approved 2026-11-20: a memory-aware pod runner (per-test weight), table reuse keyed on a hash of the files the build reads, one in-work pod run per stream. Built on branch `ops/runner-mem-tables`; the `tools/ops/build_tables.sh` pathspec bug (a changed tracked vanilla manifest is never packed) is fixed on the same branch. Its merge needs the review agent.
- Deferred by the owner (2026-11-18, do not raise unless the situation changes): a queue script in place of collector agents; a separate cheaper session that owns pod runs; stopping no-action wake-ups of streams waiting on builders.
- A daily local ROM tier (one process, more than 6 GB free) for tests that skip on the pod: started with E3 (2026-11-20).
