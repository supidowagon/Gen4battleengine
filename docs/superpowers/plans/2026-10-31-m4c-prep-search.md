# Milestone 4c: Prep Search Pipeline, CLI and Validation, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `nuzline prep --save S --trainer roark` returns the setup to bring (team, lead, items, moves, entry HP and status, with a costed prep list), the odds ladder for 0-3 deaths with the death distribution, and the Stage B plan for the requested rung, within 15 minutes cold for Roark, resumable from a checkpoint.

**Architecture:** A new funnel in `nuzline.prep` over 4a's candidate pool: member variants (relearner and TM moves, held items, entry-HP regions, declared prestatus) with a conservative dominance prune; a team phase (every team of 1-6 and lead, base variants, screened in one batch by Stage A playouts that score every rung at once, then successive halving); variant rounds (every one-change neighbour of the survivors, screened and halved the same way); Stage B (4b's budget-aware `solve`) on the finalists of the requested rung. Turn-0 states are built in one batched program (`prep/states.py`). The run is anytime: a time budget, a best-so-far answer at every point, and a JSON checkpoint after every unit of work that `--resume` continues bit-identically. A report module prints the setup card, the ladder and the plan and writes a run record. Validation: a short PP-stall fight the search must solve, a long one it reports on, an exhaustive check of the funnel on Roark (overnight benchmark plus a CI-size test) and a cold timing benchmark.

**Tech Stack:** Python 3.12 (`tomllib`), JAX (CPU, sharded host devices), numpy (float64 on the host), pytest, hypothesis, ruff; `uv run pytest`, `uv run ruff check src tests benchmarks`, `uv run ruff format`.

**Spec:** `docs/superpowers/specs/2026-10-31-prep-search-design.md` §6 (funnel, entry HP and prestatus, prune, screen, halving, Stage B finalists, anytime, checkpoint), §6.1 and §10 item 2 (PP stall), §7 (output, CLI, `config/level_caps.toml`), §8 (exhaustive Roark check, timing target), §3-§4 (rules, objective, tie-breaks); parent `docs/design.md` §4.4.

**Plan family:** Milestone 4 is three plans. Plan 4a (`2026-10-31-m4a-coverage.md`: coverage, TM data, level-to-cap, the candidate pool) and plan 4b (`2026-10-31-m4b-faint-budget.md`: faint budget, Set-mode replacements, death distribution) are independent of each other. **This plan, 4c, depends on both**: start it only once 4a and 4b are merged. It consumes their interfaces by the exact names below. Line numbers cited for files 4a/4b change (`search/playout.py`, `search/leaves.py`, `search/stage_b.py`, `cli.py`, `engine/state.py`) are those at `b36b1d0`; find the code by name after 4a/4b.

## Global Constraints

- Anti-goal, verbatim (spec §2): "A candidate that touches a mechanic the engine does not implement is excluded and named; it is never silently approximated". The run never fails because one boxed Pokemon is unsupported (spec §5.2).
- Objective (spec §4): every rung "maximises P(win and deaths ≤ k), and a loss is never traded for fewer deaths"; protected Pokemon: "any line in which it faints counts as a failure on every rung" (§10.1).
- Ties (spec §4): "Ties (overlapping intervals) break on fewer expected deaths, then resource cost (user-weighted), then expected HP and PP left." Superseded 2026-11-04 (user decision of 2026-11-03 and 2026-11-04, spec §4): win chance first; inside a win-chance tie fewer expected deaths, then cost; HP and PP are reported, not ranked (ruling 7).
- "Hiding cost. Every resource a setup spends is listed" (spec §2). Reserved resources are excluded from the search, never merely penalised (design.md §4.4).
- Rules (spec §3): Set mode; level caps; held items only (no bag items in battle); any faint is a death. Level-to-cap is on by default and its cost is reported as Rare Candies. Each fight is planned alone.
- Ladder semantics (spec §4): the screen and halving score all rungs from one batch; Stage B runs on the finalists of the requested rung (`--deaths K`, default 0); the other rungs' numbers are labelled "screen".
- Target (spec §6): "Roark from the user's save, end to end, within 15 minutes cold on the dev machine".
- Reproducible: same save, settings, seed, tables and code give the same result whatever the host device count; with `--budget` only how far the run got may differ. A resumed run equals the uninterrupted run.
- The user's save is never committed; tests and benchmarks read it only through `NUZLINE_USER_SAVE` and skip without it. Never create, edit or delete anything in a folder named `backup` (any case, any depth).
- No new runtime dependencies. `nuzline.cli` imports no jax at module import (`tests/test_cli.py::test_cli_module_does_not_import_jax`). CLI output is ASCII only; exit codes 0 ok / 1 error / 2 refused.
- Every batched program has one fixed padded chunk shape and goes through `jaxcache.cached_export` (Stage B plan constraints); import `nuzline.search.stage_a` for the `jax.export` registrations, never register `Tables`/`BattleState`/`PlayoutResult` again.
- Default-suite test files stay under ~60 s on one device; anything longer is gated by `NUZLINE_SLOW_TESTS=1` (new, documented in README next to `NUZLINE_ORACLE_DIR`).
- ruff `line-length = 100`; match the surrounding docstring density and decomp citation style.

## Rulings (controller's calls; the spec leaves them open)

1. **Funnel shape.** The full product team x lead x variants is too large to screen (the user's pool alone has 7018 (team, lead) pairs; with one Heart Scale, TM10 and 2-3 HP regions per member it exceeds 10^7). So: a **team phase** screens every (team, lead) of base members (`base_member`: current moves at the cap plus the moves levelling fills into empty slots, ruling 4; own item, full HP, no status) in one batch, then halves; then **variant expansions** (`variant_rounds`, default 2, counts expansions): each expansion screens every one-change neighbour (ruling 4) of the current rung-`k` survivors in one batch, then halves over those neighbours **plus the parents**; then **finalists**. Each phase's first round is one `evaluate_leaves` batch over every setup it has not screened yet.
2. **Successive halving, with a survivor set per rung.** Every screened setup keeps its accumulated `ScreenStats` for the whole run. The halving round index `r` restarts at 0 in the team phase and in every variant expansion. Round `r` tops every setup in the round's set up to `T_r = screen_n * (2**(r + 1) - 1)` playouts (`screen_n` default 8; a setup already at or above `T_r` gets none, so round 0 of an expansion screens only the new neighbours and the parents keep their stats; round `r >= 1` gives a team-phase setup `screen_n * 2**r` more). A phase keeps one survivor set `S_j` per rung `j` in `0..3`; at the phase's start each is the phase's whole pool (an expansion's pool for rung `j` is `S_j` plus the new neighbours, and for rung `k` the parents `S_k` plus the new neighbours). The round's set is the union of the `S_j`. After the round, `S_k` keeps its top `max(keep, ceil(|S_k| / 2))` by rung-`k` point estimate (`keep`: `team_keep` 32, `variant_keep` 8) and each other `S_j` its top `max(ladder_keep, ceil(|S_j| / 2))` by rung-`j` point estimate (`ladder_keep` 4); the halving order is `(-score, tie_deaths, cost, tie rank)` (amended 2026-11-04; the tie rank is a seeded blake2b rank of the setup's turn-0 row, rebuilt on restore and never checkpointed; whole-party HP/PP and the raw index no longer rank). Point estimates of setups with different counts are compared as they are (a parent's tighter estimate is not discounted). The phase ends once `|S_k| <= keep`; only `S_k` is expanded. `SearchStats.team_rounds` counts the team phase's halving rounds, `variant_rounds` the finished expansions and `variant_halvings` the halving rounds summed over them. **Amended 2026-11-04 (funnel fix, docs/superpowers/plans/2026-11-04-prep-funnel-fix.md):** in a variant expansion the new neighbours halve alone; then a **merged equal-n stage** (rung `k` only; keys salted by `MERGED = 1 << 20`; starting round `ladder_round(screen_n, max parent n)`) tops the surviving neighbours and the parents up together to the first halving count at or above the parents' count and cuts to the survivor floor there. **Tie retention**: at a cut that reaches a survivor floor, setups tied exactly (score and deaths) with the last survivor are kept and screened further, doubling each round, while `T_r < tie_playouts` (`tie_playouts = 1016`). Each merged and tie-retention round is its own anytime unit.
3. **Finalists.** The top `finalists` (default 3) of the final `S_k` by the tie rule (ruling 7) each get a Stage B solve at depth 1 (precision target 1.5 pp); the leader among them by ruling 7 on the Stage B values is then solved again at `max_depth` (default 2) with the time left, and that result replaces its depth-1 one. `max_depth = 0` means screen only. **Amended 2026-11-04:** the finalists are the top `finalists` of the final `S_k` by the new `prep_order` (ruling 7), and they are saved in the finalists position (`pos["picks"]`).
4. **Variants and neighbours.** A member's changes: a move change, which **fills the first empty move slot if there is one, else replaces one of the four slots** (the game gives no choice when a slot is free); a slot changed once is not changed again. Move sources: **learned by levelling** (`LEVEL_UP`, cost 0): a move whose level-up level is above the candidate's save level (`spec.level - candies`) and at most the cap, learned while the Rare Candies are used; the game fills an empty slot with it without asking, so `base_member` already holds those (in learn-level order), and one that meets a full set is an optional single change (the game offers the replacement); **Move Relearner** (`RELEARNER`, cost 1 Heart Scale): any other relearner option; **TM/HM** (cost that TM; HMs reusable). a held item from the bag (implemented effects only); an entry-HP region (ruling 5); a declared prestatus (ruling 6). A neighbour applies exactly one change to one member, or makes another member the lead, and must fit the unreserved resources. A member's own held item stays unless a bag item replaces it.
5. **Entry-HP regions.** A member's entry HP matters in two ways. **Survival**: more HP never hurts it, so the damage values the opponent can deal need no split of their own (within a region the top HP survives everything a lower HP does). **Behaviour**: the engine's checks of the player's HP cut the range into regions: (a) the trainer AI's defender percent check (`ai/policy.py:279`, `IfHPPercentGreaterThan(_D, 70)`); (b) the AI's kill check (`_eval_attack`, `ai/policy.py:233`, `if_current_move_kills(ctx, USE_MAX_DAMAGE)`, `ai/commands.py:454-464`): a move scores +4 when the player's HP is at most its `ai_damage` at `SIM_ROLL_MAX`, so each opponent party member's damaging move adds the cut `d` (HP `<= d` turns the bonus on), with `d` taken from the turn-0 matchup of the member against that opponent (`make_state(..., leads=(0, j))`, its switch-in hooks applied); (c) the pinch cut `max_hp // 3` for Blaze, Torrent and Overgrow. Each region is represented by its top HP. The cuts are the turn-0 values: later in the fight stat stages move the AI's damage estimate and the member's HP has changed, so a region is not a guarantee of identical play; the screen judges the regions. The player's playout policy reads the player's HP only through smooth KO odds (`search/playout.py:107-111`) and adds no cut. Thresholds that favour higher HP (Oran/Sitrus at 1/2) add no region. Regions with a top of 0 are dropped (Shedinja). A future HP-dependent mechanic (Reversal, Flail, Focus Sash, a new AI check) adds its cut in `hp_cuts`; a source-scan test pins every AI function that reads `hp[`, so a new AI reader fails it until `AI_HP_READERS` (and `hp_cuts`, if it reads the player's HP) is updated. design.md §4.4 is corrected to say this.
6. **Prestatus.** Off unless declared: `--prestatus STATUS` with poison, burn or paralysis (sleep and freeze change outside battle; toxic becomes poison outside battle). A status is not offered to a member that cannot have it: by type (Poison and Steel: poison; Fire: burn) or by ability (`STATUS_BLOCKING_ABILITIES`: Limber: paralysis; Immunity: poison; Water Veil: burn; resolved by name with `tables.ability`). The engine does not implement these three abilities today, so such a candidate is already left out by `usable_pool`; the rule keeps `prestatus_options` right when one is added.
7. **Tie rule** (prep_order; superseded 2026-11-04 by the user decision of 2026-11-03, see spec section 4). Rows follow the battle tie rule (plan 4b ruling 5, `stage_a.tie_ordered`) at every position: among the remaining rows overlapping the top-scoring remaining row, the fewest tie_deaths, then higher score, lower weighted cost, tie rank, index (user decision 2026-11-04). DEATHS_TOL is removed. HP and PP are reported, not ranked. Default weights unchanged (candy 0, heart_scale 3, tm 10, hm 5, held 1), now the last tie-break only.
   ~~**Tie rule** (`prep_order`). `O` = the setups whose rung-`k` interval overlaps the top-scoring one's. The leader is chosen from `O`: keep those with expected deaths at most `min + DEATHS_TOL` (0.05), then those with the lowest weighted cost, then the highest mean HP left, then the highest mean PP left, then the higher score, then the lower index. Everyone else follows by `(-score, e_deaths, weighted cost, index)`. Default weights: candy 0, heart_scale 3, tm 10, hm 5, held 1; `--weight KIND=W` overrides.~~
8. **Conservative prune** (`prune`), a heuristic, not a proof. Among one candidate's single-change variants, `v` is dropped when another variant `w` of the same candidate has the same item, HP and status, a cost no larger in every component, and slot by slot each of `w`'s moves equals `v`'s or covers it (both `EffectId.PLAIN`, same type, same category, same priority, power and accuracy and PP no lower, and `w`'s move makes contact only if `v`'s does). More damage is not always better: it changes the AI's kill and highest-damage scoring, the opponent's HP-threshold triggers (berries, the AI's 70 % checks) and what Counter, Mirror Coat or Bide return, so a dropped variant can be strictly better. The prune is kept because those cases are rare for same-type PLAIN moves; Task 12 values every setup the prune dropped from the exhaustive space (`pruned_setups`) and fails the check if one beats the pick, and the run record lists every dropped variant. The design.md heuristics (type matchups, speed, OHKO checks) are not used: they are not dominance (a fragile Pokemon can be the right switch-in, and the AI reads the player's moves), so the screen does that work.
9. **Party order.** The lead is party slot 0; the other members follow in pool order (party, then boxes). Set mode makes the rest of the order irrelevant to play; the prep list states it.
10. **Screen scoring.** A setup's screen numbers come from `evaluate_leaves` playouts from its turn-0 state with its own `faint_budget = k` and protected slots (so the playout policy plays the requested rung), both halves pooled: the 8 bucket counts, expected deaths, mean HP and PP left. Rung `r`'s value is `DeathDistribution.rung(r)` with a Wilson interval (`stage_a.wilson`). A setup with any refused playout is left out and named.
11. **Budget.** `--budget SECONDS` (default none) is checked before every unit after the team phase's first round, which always completes (as Stage B's depth 1 does). Finalist solves get `budget_seconds` = the time left. On expiry or Ctrl-C the best-so-far answer is printed (a screen-only answer says "screen only, no plan"), exit 0. `--budget` limits this invocation; the checkpoint keeps the total.
12. **Checkpoint.** A JSON file (`CHECKPOINT_VERSION = 3`; 1 was the first shape, 2 added a variant position's parents, 3 is the 2026-11-04 funnel fix), written atomically (temp file then `os.replace`) after every unit: each screen or halving round, each finalist solve. It holds a fingerprint (problem hash, settings without `budget_seconds`, table version, code version, seed), every screened setup and its accumulated stats, the phase, expansion, round and each rung's survivor set, and the finished Stage B results (`stage_b.result_to_dict`). Keys are derived from the seed, phase and round only, so resuming reproduces the uninterrupted run. Default path `runs/prep_<time>.ckpt.json`; `--resume FILE` continues and rewrites that file and refuses any fingerprint mismatch. **Amended 2026-11-04:** a variants position may carry `"stage": "merged"` (and then no `"parents"`). A finalists position carries `"picks"`. Older checkpoints are refused with the existing one-line message.
13. **Level caps.** `config/level_caps.toml` holds only the cap the user stated (Roark, trainer 246: 16); a trainer without an entry needs `--cap`. `coverage --save` uses the config cap when there is one, else 4a's highest-party-level default.
14. **Exhaustive check cost.** The spec's space (pool of 6 party + 2 boxed, current moves or one relearner variant each, no items, full HP, teams 1-6 with every lead) is at most 5112 setups for one Heart Scale (5112 when all 8 have a designated relearner variant; the benchmark prints the exact count), plus the pruned setups of Task 12 (`pruned_setups`: restricted-shape setups holding one member on a relearner variant that `prune` dropped from the `relearner="all"` variants; possibly none on this save, which the result then states); Stage B depth 1 at the precision target is about 40 s each, ~110 h for both rungs. The benchmark therefore runs in two phases on the same value function: phase 1 is Stage B depth 1 with `budget_playouts = 2000` for every setup; phase 2 re-solves at the precision target every setup whose phase-1 interval overlaps the phase-1 best's, plus the funnel's pick. It checkpoints after every setup and resumes across nights; it prints a measured ETA after 20 setups. The two boxed Pokemon are the first Venonat (box 1 slot 1) and Geodude (box 1 slot 4); a candidate's **designated relearner variant** is its Move Relearner option (source `RELEARNER`, so not a move learned by levelling, which is in the base or free) with the highest table power (ties: learned later), placed by ruling 4 (an empty slot, else the slot holding the lowest-power move, status moves counting 0, ties the lower slot).
15. **PP-stall fights** are synthetic (no save): Roark's Cranidos alone, bag empty, knowing Rock Throw (its only damaging move, `stall_pp` PP left) and Leer (30 PP, so it never Struggles); a tank that cannot damage (Geodude knowing only Defense Curl, stats overridden so it survives every Rock Throw at -6 Defense); an attacker that Rock Throw KOs when it hits and that 2HKOs Cranidos (Starly, stats overridden); a second such attacker (Shinx) as a decoy. The only deathless lines stall Rock Throw's PP with the tank, then bring in an attacker. Short stall: `stall_pp = 5`, required. Long stall: 20 and 30, reported.

## Review Focus

1. **Two Pokemon of one species** (the user's two Venonat) named by `--protect Venonat` or `--exclude Venonat`: an error listing both locations and the `box:1:1` syntax, exit 1; `--exclude box:1:2` works. Test: Task 3 `test_pick_ambiguous_and_locations`, Task 10 `test_cli_prep_bad_flags`.
2. **`--resume` with a checkpoint from another save, trainer, settings, seed, table or code version, or a truncated file**: one stderr line naming what differs, exit 1, the file unchanged. Test: Task 8 `test_resume_refuses_mismatch`.
3. **`--budget` shorter than the first screen round**: that round completes, the answer is its leader labelled "screen only, no plan", exit 0. Test: Task 7 `test_budget_stops_after_first_round`.
4. **No deathless line exists** (every setup's rung 0 is 0): the card says so, the ladder still lists rungs 1-3, the plan is for the requested rung, nothing divides by zero. Test: Task 9 `test_report_without_a_deathless_line`.
5. **Nothing usable left** (every candidate unsupported, or all excluded): "no usable Pokemon" with the reasons, exit 1. Test: Task 3 `test_build_problem_errors`, Task 10 `test_cli_prep_bad_flags`.

---

### Task 1: HP and PP left, and both halves' buckets, from every playout

**Files:**
- Modify: `src/nuzline/search/playout.py` (`PlayoutResult` :78-82, `_run` :205-230), `src/nuzline/search/leaves.py` (`LeafValue` :58-89, `_forced_one` :127-135, `_run` :142-172)
- Test: `tests/search/test_leaves.py`, `tests/search/test_playout.py`

**Interfaces:**
- Consumes: 4b `PlayoutResult.deaths/ended`, `LeafValue.deaths_a/deaths_b/buckets_b/merge`, `deaths.bucket`.
- Produces:
  - `PlayoutResult` gains, after 4b's fields, `hp_left: float32[]` = `sum(hp[0]) / sum(max_hp[0])` and `pp_left: float32[]` = `sum(pp[0]) / max(1, sum(tables.move_pp[moves[0]]))` over the player's party of the final state (refused: 0). `_forced_one` copies `rest`'s.
  - `LeafValue` gains `buckets_a: tuple[int, ...] = ()` (8 counts over half A, as 4b's `buckets_b`), `hp_left: float = 0.0`, `pp_left: float = 0.0` (sums over both halves); `merge` adds them (an empty tuple counts as zeros).

- [ ] **Step 1: Write the failing tests**
  - `test_playout.py::test_hp_and_pp_left`: on `roark_real_state`, 8 keys: each playout's `hp_left`/`pp_left` equal the formula on the final state of a replay with `_turn_jit` (as 4b's `test_result_records_deaths_and_end` replays), within 1e-6; a refused playout (the `switch_refusal_root` fixture's refused key) has 0.
  - `test_leaves.py::test_both_halves_have_buckets`: `evaluate_leaves` on two states, 64 playouts each: `sum(buckets_a) == n_a`, `sum(buckets_b) == n_b`, each equals `np.bincount(deaths.bucket(...), minlength=8)` over that half's `batched_playout` results on `leaf_keys`; `hp_left` equals the sum of both halves' `hp_left`.
  - `test_leaves.py::test_merge_sums_new_fields`.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_playout.py tests/search/test_leaves.py -q`; expected: fail.
- [ ] **Step 3: Implement** the fields (append only; existing fields and keys unchanged).
- [ ] **Step 4: Run** `uv run pytest tests/search -q` (4b's `test_k0_regression.py` included) and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): HP/PP left and both halves' death buckets in leaf values`.

### Task 2: Batched turn-0 states (`prep/states.py`)

**Files:**
- Modify: `src/nuzline/engine/state.py` (`_fill_slot` :225-257 and `make_state` :260-307 split: the numpy part becomes `initial_arrays`)
- Create: `src/nuzline/prep/states.py`
- Test: `tests/prep/test_states.py`

**Interfaces:**
- Consumes: 4b `BattleState.with_budget`, `MAX_DEATH_BUDGET`, fields `faint_budget`, `protected`; `hooks.on_switch_in`; `jaxcache.cached_export`.
- Produces:
  - `state.initial_arrays(tables, player, opp, start_condition=Weather.NONE, opp_items=(), leads=(0, 0)) -> dict[str, np.ndarray]`: every `LAYOUT` field before the switch-in hooks, with `make_state`'s validation; `make_state` becomes `initial_arrays` plus the two `on_switch_in` calls (output unchanged).
  - `STATE_CHUNK = 512`; `make_states(tables, teams: Sequence[Sequence[PokemonSpec]], opp: Sequence[PokemonSpec], start_condition: int = Weather.NONE, opp_items: Sequence[int] = (), budget: int = 0, protected: Sequence[Sequence[int]] | None = None) -> BattleState`: a numpy batch (leading axis `len(teams)`), row `i` byte-equal to `make_state(tables, list(teams[i]), list(opp), start_condition, opp_items, leads=(0, 0)).with_budget(budget, protected[i])`. Host numpy per row, then one exported program `"make_states"` (both sides' `on_switch_in`, vmapped) per padded chunk (padding: copies of row 0, dropped). `ValueError` as `with_budget` for a bad budget or protected slot.

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_states.py`)
  - `test_make_states_matches_make_state`: 40 hypothesis teams (1-6 members drawn from the sample party with random leads permuted to slot 0, HP between 1 and max, statuses from {none, poison, burn, paralysis}) against Roark's real party and bag, `budget` 0-3, protected subsets of living slots: every row byte-equal (`rows.row_bytes`) to the `make_state(...).with_budget(...)` row.
  - `test_make_states_chunks`: 513 teams give 513 rows; `cached_export` (monkeypatched to count) is called with `name="make_states"` twice and one shape.
  - `test_make_states_rejects_bad_budget`: `budget=4` and a protected slot holding a fainted Pokemon raise `ValueError`.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_states.py tests/engine/test_state.py -q`; expected: fail.
- [ ] **Step 3: Implement** `initial_arrays` and `make_states`.
- [ ] **Step 4: Run** `uv run pytest tests/engine tests/prep -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): batched turn-0 state builder`.

### Task 3: The prep problem: usable pool, resources, selectors, level caps

**Files:**
- Create: `src/nuzline/prep/problem.py`, `src/nuzline/prep/caps.py`, `config/level_caps.toml`, `tests/prep/fixtures.py`
- Modify: `src/nuzline/engine/coverage.py` (`check_opponent` split out of `check_coverage` :213-236), `src/nuzline/data/learnsets.py` (4a; `is_hm`)
- Test: `tests/prep/test_problem.py`, `tests/prep/test_caps.py`, `tests/engine/test_coverage.py`

**Interfaces:**
- Consumes: 4a `Candidate`, `pool_candidates`, `Learnsets` (`level_up`), `player_move_refusal`, `player_ability_refusal`; `IMPLEMENTED_EFFECTS`, `IMPLEMENTED_ABILITIES`, `IMPLEMENTED_ITEMS`; 4b `MAX_DEATH_BUDGET`.
- Produces:
  - `coverage.check_opponent(tables, names, trainer, opp, start_condition) -> CoverageReport`: the trainer's party, bag, AI flags and weather lines of `check_coverage` (which now calls it; its output is unchanged). It runs no `ai_refusals`: those depend on the player's party, so prep finds them per setup through the screen's `refused` flag (ruling 10) and `check_coverage` keeps running them on its own state.
  - `Learnsets.is_hm(item: int) -> bool` (TM index >= 92).
  - `caps.CAPS = <repo>/config/level_caps.toml`; `load_level_caps(path: Path = CAPS) -> dict[int, tuple[str, int]]` (trainer id -> (name, cap)); `default_cap(trainer: Trainer, path: Path = CAPS) -> int`: `ValueError` "no level cap for Roark (id 246) in config/level_caps.toml; pass --cap" when absent, and when the file's name for that id differs from `trainer.name` (case-insensitive). The file: a comment line and `[caps]` with `246 = { name = "Roark", cap = 16 }`.
  - `@dataclass(frozen=True) Resources: heart_scales: int, tms: tuple[tuple[int, int], ...]` ((TM item, quantity), item order), `hms: tuple[int, ...]`, `held: tuple[tuple[int, int], ...]` ((item, quantity), effect in `IMPLEMENTED_ITEMS` and not `NONE`).
  - `bag_resources(bag: Mapping[str, list[tuple[int, int]]], tables, names, learnsets, reserved: Collection[int]) -> Resources`: Heart Scales from every pocket; TMs/HMs from `tms_hms`; held items from `items`, `berries`, `medicine`, `battle_items`; every reserved item id removed entirely.
  - `usable_pool(tables, names, candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[str]]`: a candidate whose ability, held item or a current move is unsupported (`IMPLEMENTED_*`) or refused by the AI (`player_ability_refusal`, `player_move_refusal`) is left out with `"<label>: left out: <what> (<reason>)"`; an unsupported or refused relearner or TM option is removed from `relearner`/`tms` with `"<label>: <move> not offered (<reason>)"`.
  - `pick(candidates: Sequence[Candidate], names, value: str) -> int`: `party:N` (N as in the label), `box:B:S` (B 1-based, S as in the label), else a case-insensitive match of the species name or the nickname in `label`; `ValueError` for no match, and for several naming each label and the location syntax.
  - `@dataclass(frozen=True) PrepProblem: candidates: tuple[Candidate, ...], trainer: Trainer, opp: tuple[PokemonSpec, ...], start_condition: int, resources: Resources, deaths: int, protected: frozenset[int]` (candidate indices), `prestatus: tuple[int, ...]` (`ids.Status`), `cap: int`, `source: str` (`"save:<sha256>"` or a test name), `levelled: tuple[tuple[tuple[int, int], ...], ...]` (per candidate, the `(level, move)` pairs of `learnsets.level_up(species)` with `spec.level - candies < level <= cap` whose move the spec does not know and that `usable_pool` kept, ascending, first occurrence per move; ruling 4).
  - `PRESTATUS = {"poison": Status.POISON, "burn": Status.BURN, "paralysis": Status.PARALYSIS}`.
  - `build_problem(tables, names, save: SaveFile, trainer, opp, start_condition, learnsets, *, cap: int, deaths: int = 0, protect: Sequence[str] = (), exclude: Sequence[str] = (), reserve: Sequence[str] = (), prestatus: Sequence[str] = ()) -> tuple[PrepProblem, list[str]]`: 4a's `pool_candidates`, then exclusions, then `usable_pool`; warnings are both lists' lines plus "<item>: reserved but not in the bag". `ValueError` for: `deaths` outside `0..MAX_DEATH_BUDGET`, an unknown item or prestatus name, a value both protected and excluded, a protected candidate left out as unsupported, an empty pool ("no usable Pokemon: " + the reasons).
  - `tests/prep/fixtures.py`: `candidate(names, spec: PokemonSpec, slot: int, relearner=(), tms=(), candies=0) -> Candidate` (location `("party", 0, slot)`, label `"party slot <slot> <SPECIES>"`); `problem(tables, names, candidates, *, opp=None, resources=None, deaths=0, protected=(), prestatus=(), cap=16, levelled=None) -> PrepProblem` (default `opp`: Roark's real party; trainer `roark_trainer()`; resources empty; `levelled=None` computes it as `build_problem` does, from `load_learnsets()` and each candidate's `candies`).

- [ ] **Step 1: Write the failing tests**
  - `test_caps.py::test_level_caps_file`: `load_level_caps() == {246: ("Roark", 16)}`; `default_cap(roark_trainer()) == 16`; trainer 858 raises with `--cap` in the message; a temporary TOML naming 246 "Gardenia" raises.
  - `test_problem.py::test_resources_from_bag`: bag `{"items": [(Heart Scale, 1), (Potion, 3)], "tms_hms": [(TM10, 1), (HM01, 1)], "berries": [(Oran Berry, 2)]}` gives `Resources(1, ((TM10, 1),), (HM01,), ((Oran Berry, 2),))`; reserving Heart Scale gives `heart_scales == 0`; reserving TM10 drops it.
  - `test_usable_pool_excludes_and_trims`: a Spinarak with Metronome as a current move is left out naming Metronome; a Furret with relearner `(Metronome, Fury Swipes)` keeps Fury Swipes only, with a "not offered" line; a supported candidate passes unchanged.
  - `test_pick_ambiguous_and_locations` (Review Focus 1): two Venonat candidates (`box 1 slot 1`, `box 1 slot 2`): `pick(.., "venonat")` raises with both labels and `box:B:S`; `pick(.., "box:1:2")` is the second; a nickname matches; `party:9` raises.
  - `test_build_problem_errors` (Review Focus 5), parametrized on a synthetic save (`tests/saves/synthetic.py`, 4a's `box_record`): `deaths=4`; `reserve=["Potionx"]`; `prestatus=["sleep"]`; protect and exclude the same Pokemon; excluding every candidate ("no usable Pokemon"); each a `ValueError` naming the value.
  - `test_coverage.py::test_check_opponent_matches_check_coverage`: for the sample party vs Roark and vs trainer 858, `check_opponent`'s missing lines equal `check_coverage`'s missing lines that neither start with `"Player "` nor end with `" (trainer AI UNSUPPORTED)"` (the `ai_refusals` lines), in order; and `check_opponent` makes no `make_state` call (monkeypatched to raise).
  - `test_problem.py::test_levelled_moves`: a synthetic save with a boxed Kricketot L5 (Growl, Bide) and a party Furret L16, cap 16: Kricketot's `levelled` entry is `((16, Bug Bite),)`, Furret's is `()`; with cap 15 Kricketot's is `()`.
- [ ] **Step 2: Run** `uv run pytest tests/prep tests/engine/test_coverage.py -q`; expected: fail.
- [ ] **Step 3: Implement**; README gains `config/level_caps.toml` in the file list.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/engine tests/data tests/prep tests/test_cli.py -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): prep problem, usable pool, bag resources and level caps`.

### Task 4: Member variants, costs, HP regions and the prune (`prep/variants.py`)

**Files:**
- Create: `src/nuzline/prep/variants.py`
- Modify: `docs/design.md` §4.4 (HP brackets: ruling 5, survival vs behaviour cuts, the AI's kill check included)
- Test: `tests/prep/test_variants.py`

**Interfaces:**
- Consumes: Task 3 (`PrepProblem`, `Resources`, `levelled`), `state.make_state` (`leads`), `ai.commands.ai_damage`, `SIM_ROLL_MAX`, `is_damaging`, 4a `calc_stats(..., species=)`.
- Produces:
  - `RELEARNER = 0`, `LEVEL_UP = -1`; `@dataclass(frozen=True) MoveChange: slot: int, move: int, source: int` (`RELEARNER`, `LEVEL_UP`, else the TM/HM item).
  - `base_member(problem, candidate: int) -> Member`: the moves of `problem.levelled[candidate]` filled into the empty slots in learn-level order as `LEVEL_UP` changes (ruling 4); `Member(candidate)` when there are none.
  - `@dataclass(frozen=True) Member: candidate: int, moves: tuple[MoveChange, ...] = (), item: int = 0` (0 = keep its own), `hp: int | None = None` (None = full), `status: int = 0`.
  - `@dataclass(frozen=True) Cost: candies: int = 0, heart_scales: int = 0, tms: tuple[int, ...] = (), hm_moves: int = 0, held: tuple[int, ...] = ()` (sorted); `__add__`; `fits(resources: Resources) -> bool` (Heart Scales, each TM's and held item's count; HMs always fit when in `hms`); `weighted(weights: Mapping[str, float]) -> float` over keys `candy`, `heart_scale`, `tm`, `hm`, `held`. `DEFAULT_WEIGHTS` per ruling 7.
  - `member_spec(problem, member) -> PokemonSpec`: the candidate's spec with each change's move in its slot at that move's table PP, `item`, `current_hp`, `status` (counter 0).
  - `member_cost(problem, member) -> Cost` (includes the candidate's `candies`; per change: `LEVEL_UP` 0, `RELEARNER` 1 Heart Scale, a TM its item, an HM `hm_moves` 1).
  - `AI_DEFENDER_HP_CHECKS = (("greater", 70),)`; `AI_HP_READERS: dict[str, str]` = `{"_hp_percent": "percent", "if_current_move_kills": "kill", "count_alive_party_battlers": "alive", "_use_item": "opponent", "_post_ko": "opponent", "_layout": "opponent"}` (every function in `src/nuzline/ai/*.py` that indexes an `hp` array at `b36b1d0`, and how `hp_cuts` covers it: `percent` by `AI_DEFENDER_HP_CHECKS`, `kill` by `kill_damage`; `alive` (HP > 0 only) and `opponent` (the trainer's side) need no cut).
  - `ai_kill_damage(tables, spec, opp: Sequence[PokemonSpec], start_condition: int = Weather.NONE, opp_items: Sequence[int] = ()) -> tuple[int, ...]`: for each opponent slot `j` and each move `m` of it with `is_damaging`, `ai_damage(tables, make_state(tables, [spec], list(opp), start_condition, opp_items, leads=(0, j)), m, SIM_ROLL_MAX)`; distinct, ascending, zeros dropped (ruling 5 b).
  - `hp_cuts(tables, spec, max_hp, kill_damage: Sequence[int] = ()) -> tuple[int, ...]` (region tops below `max_hp`: `((p + 1) * max_hp - 1) // 100` for a "greater" check, `(p * max_hp - 1) // 100` for a "less" check, each `d` in `kill_damage` with `1 <= d < max_hp`, `max_hp // 3` for a Blaze/Torrent/Overgrow holder); `hp_regions(tables, spec, max_hp, kill_damage: Sequence[int] = ()) -> tuple[int, ...]` (`max_hp` then the cuts, descending, distinct, all >= 1). `single_changes` passes `ai_kill_damage` of the member's spec (with its item) against `problem.opp`, `problem.start_condition` and `problem.trainer.items`.
  - `STATUS_BLOCKING_ABILITIES = {"Limber": Status.PARALYSIS, "Immunity": Status.POISON, "Water Veil": Status.BURN}`; `prestatus_options(tables, spec, declared: Sequence[int]) -> tuple[int, ...]` (ruling 6: by type and by ability).
  - `move_changes(spec, move: int, source: int) -> tuple[MoveChange, ...]` (ruling 4; none when `spec` already knows `move`).
  - `designated_change(tables, problem, candidate: int) -> MoveChange | None` (ruling 14; over the candidate's `RELEARNER` options only, placed on `base_member`'s moves).
  - `single_changes(tables, problem, candidate: int, *, relearner: str = "all", tms: bool = True, items: bool = True, entry_hp: bool = True) -> list[Member]`: every one-change variant of `base_member` (`relearner`: `"all"`, `"designated"` or `"none"`, applied to `RELEARNER` options; the `LEVEL_UP` moves not in the base are always offered), only changes whose cost fits `problem.resources`.
  - `prune(tables, problem, members: Sequence[Member]) -> tuple[list[Member], list[tuple[Member, Member]]]` (ruling 8): kept, and (dropped, dominated by).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_variants.py`)
  - `test_hp_regions`: Monferno (Blaze) max 48 -> `(48, 34, 16)`; Furret (Keen Eye) max 57 -> `(57, 40)`; Monferno max 3 -> `(3, 2, 1)`; Shedinja max 1 -> `(1,)`.
  - `test_hp_regions_include_ai_kill_checks`: Monferno at the cap (fixture spec) vs Roark's real party: `d = ai_damage(tables, make_state(tables, [spec], roark_party, leads=(0, 2)), Rock Throw, SIM_ROLL_MAX)` (Cranidos is slot 2); the test asserts `1 <= d < max_hp`, then `d in ai_kill_damage(tables, spec, roark_party)` and `d in hp_regions(tables, spec, max_hp, ai_kill_damage(...))`; with Cranidos active and the player's HP set to `d`, `if_current_move_kills(ctx, USE_MAX_DAMAGE)` on Rock Throw jumps, and with `d + 1` it does not (`commands.new_context(tables, state, sample_ai_outcome(PRNGKey(0), tables, state))._replace(move=Rock Throw)`, reading `.jump`; `USE_MAX_DAMAGE` ignores the outcome).
  - `test_ai_hp_checks_pinned`: (a) every `if_hp_percent_(greater|less)_than(ctx, _D, N)` call in `src/nuzline/ai/*.py` (regex over the source) is in `AI_DEFENDER_HP_CHECKS` and vice versa; (b) the names of the functions in `src/nuzline/ai/*.py` whose body indexes an `hp` attribute (`ast` scan: a `Subscript` whose value is an `Attribute` named `hp`) equal `AI_HP_READERS`'s keys; (c) every call of `if_current_move_kills` passes `USE_MAX_DAMAGE` (a `ROLL_FOR_DAMAGE` call would need a cut per roll).
  - `test_move_changes`: Kricketot `(Growl, Bide, 0, 0)` + Bug Bite -> one change, slot 2; Furret with four moves + Hidden Power -> slots 0-3; a known move -> `()`.
  - `test_member_spec_and_cost`: Furret with TM10 in slot 2: moves `(Scratch, Foresight, Hidden Power, Quick Attack)`, PP of slot 2 = table PP of Hidden Power, the others unchanged; `hp=40` and `status=POISON` carried; cost `Cost(tms=(TM10,))`; a relearner change costs `heart_scales=1`; an HM costs `hm_moves=1`; a boxed candidate's `candies` are included; a boxed Kricketot L5 (Growl, Bide, 0, 0; candies 11; `levelled ((16, Bug Bite),)`): `base_member(...).moves == (MoveChange(2, Bug Bite, LEVEL_UP),)` and its `member_cost == Cost(candies=11)` (0 Heart Scales); a Kricketot with a full set and the same `levelled` gets Bug Bite only as single changes (slots 0-3, `LEVEL_UP`), each costing `Cost(candies=11)`; `weighted(DEFAULT_WEIGHTS)` of TM10 + Heart Scale is 13.
  - `test_fits_respects_quantities`: two TM10 changes do not fit one TM10; HM changes always fit when the HM is in the bag.
  - `test_prestatus_options`: Spinarak (Bug/Poison) with `(POISON, BURN)` -> `(BURN,)`; Monferno -> `(POISON,)`; Hitmonlee (ability slot 0, Limber) with `(POISON, PARALYSIS)` -> `(POISON,)`; Snorlax (ability slot 0, Immunity) with `(POISON, PARALYSIS)` -> `(PARALYSIS,)`.
  - `test_designated_change` (fixture Geodude with `candies=0`, so every option is `RELEARNER`): Geodude with moves `(Tackle, Defense Curl, Mud Sport, 0)` and relearner `(Rock Polish, Rock Throw, Magnitude)` (table powers 0, 50, 1) -> `MoveChange(3, Rock Throw, 0)`; with moves `(Tackle, Defense Curl, Mud Sport, Rock Polish)` and relearner `(Rock Throw, Magnitude)` -> `MoveChange(1, Rock Throw, 0)` (status moves count 0; Defense Curl is the lowest slot among them); no relearner -> `None`.
  - `test_single_changes_respect_resources`: no Heart Scale -> no `RELEARNER` changes, but a `LEVEL_UP` move meeting a full set is still offered; `relearner="designated"` -> at most one move change, the designated one; `entry_hp=False` -> no HP variants.
  - `test_prune_drops_only_strict_dominance`: relearner options Tackle and Scratch into slot 0 of a full moveset: Tackle's variant is dropped, dominated by Scratch's; Ember's variant is never dropped by a PLAIN move; a Mach Punch variant (priority 1) is not dropped by a Scratch variant; a TM variant is not dropped by a relearner variant (different cost components).
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_variants.py -q`; expected: fail.
- [ ] **Step 3: Implement**; design.md §4.4's HP-bracket paragraph is replaced by ruling 5's: damage values matter for survival (no split: a region's top HP is its most favourable) and through the AI's kill check (a cut at each opponent move's `ai_damage` at `SIM_ROLL_MAX` on the turn-0 matchup), plus the AI's percent checks and the pinch cut; these are turn-0 values, so a region is a heuristic the screen judges, not a guarantee of identical play.
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): member variants, costs, entry-HP regions and the conservative prune`.

### Task 5: Setups: teams, neighbours and their states (`prep/setups.py`)

**Files:**
- Create: `src/nuzline/prep/setups.py`
- Test: `tests/prep/test_setups.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  - `@dataclass(frozen=True) Setup: members: tuple[Member, ...]` (`members[0]` leads; distinct candidates; the others in pool order; ruling 9).
  - `team_setups(problem, max_team: int = 6) -> list[Setup]`: every team of `1..min(max_team, n)` candidates (`itertools.combinations` in pool order, by size) with every lead, `base_member`s.
  - `setup_specs(problem, setup) -> list[PokemonSpec]`; `protected_slots(problem, setup) -> tuple[int, ...]`; `setup_cost(problem, setup) -> Cost`.
  - `neighbours(tables, problem, setup, *, pruned: Collection[Member] = (), relearner="all", tms=True, items=True, entry_hp=True) -> list[Setup]` (ruling 4): one-change variants of one member (a changed slot is not changed again; a move change from the member's current spec), each other member as lead; only fitting setups; none whose single-change member is in `pruned`; no duplicates, `setup` itself excluded.
  - `setup_states(tables, problem, setups: Sequence[Setup]) -> BattleState`: `make_states` with `budget=problem.deaths` and each setup's protected slots.
  - `dedupe(states, setups, costs: Sequence[float]) -> list[int]`: indices kept when several setups give byte-equal states (the cheapest, then the lowest index).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_setups.py`, candidates from `tests/prep/fixtures.py`)
  - `test_team_setups_count`: pools of 3 (max 2), 4 and 11 give 9, 32 and 7018 setups (`sum(comb(n, k) * k)`), all distinct, each lead appearing once per team, the non-leads in pool order.
  - `test_neighbours_one_change`: with one Heart Scale and TM10, each neighbour of a base 3-member setup differs from it in exactly one member field or the lead; after a relearner change no neighbour adds another; a reserved TM never appears; neighbours of a setup with a changed slot never change that slot again.
  - `test_setup_states_and_protected`: a setup with the protected candidate in slot 2 gives `protected == [0, 0, 1, 0, 0, 0]` and `faint_budget == problem.deaths`; rows equal Task 2's `make_state` rows.
  - `test_dedupe_keeps_the_cheapest`: two setups whose only difference is an HP region equal to max HP collapse to the cheaper one.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_setups.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): team setups, one-change neighbours and batched setup states`.

### Task 6: The screen and the tie rule (`prep/screen.py`)

**Files:**
- Create: `src/nuzline/prep/screen.py`
- Test: `tests/prep/test_screen.py`

**Interfaces:**
- Consumes: Task 1 (`LeafValue` fields), 4b `DeathDistribution`, `BUCKETS`; `leaves.evaluate_leaves`; `stage_a.wilson`.
- Produces:
  - `@dataclass(frozen=True) ScreenStats: buckets: tuple[int, ...]` (8), `deaths: int`, `hp_left: float`, `pp_left: float`, `refused: bool`; properties `n`, `distribution` (`DeathDistribution.from_counts`), `e_deaths`, `mean_hp`, `mean_pp`; `rung(k) -> float`, `interval(k) -> tuple[float, float]` (Wilson on the rung's count), `merge(other) -> ScreenStats`; `EMPTY` (all zero).
  - `screen(tables, states: BattleState, n: int, key, max_turns: int = 100) -> list[ScreenStats]`: `evaluate_leaves` with `n` each (even), both halves pooled (ruling 10).
  - `@dataclass(frozen=True) Scored: score: float, ci_low: float, ci_high: float, e_deaths: float, cost: float, hp_left: float, pp_left: float`; `scored(stats: ScreenStats, k: int, cost: float) -> Scored`.
  - `DEATHS_TOL = 0.05`; `prep_order(rows: Sequence[Scored]) -> list[int]` (ruling 7).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_screen.py`)
  - `test_screen_pools_both_halves`: two states, `n=64`: `stats.n == 64`, `buckets == buckets_a + buckets_b` of a direct `evaluate_leaves`, `rung(3) >= rung(0)`, `interval(0) == wilson(count, 64)`.
  - `test_screen_flags_refused`: `tests.search.fixtures.switch_refusal_root` with `n=256` (the AI's refused switch has mass ~0.56 on the first turn) gives `refused`; `roark_real_state` does not.
  - `test_prep_order`: rows (score, ci_low, ci_high, e_deaths, cost, hp, pp) = A (0.90, 0.85, 0.95, 0.30, 0, 0.5, 0.5), B (0.88, 0.83, 0.93, 0.10, 3, 0.5, 0.5), C (0.87, 0.82, 0.92, 0.12, 0, 0.4, 0.5), D (0.60, 0.55, 0.65, 0.0, 0, 1.0, 1.0): order `[C, A, B, D]` (B and C are within `DEATHS_TOL` of the fewest deaths in the overlap set, C is cheaper; the rest by score); with C's cost 3: `[B, A, C, D]` (equal cost, B has more HP left).
  - `test_rung_k_score`: `scored(stats, 1, 0).score == stats.rung(1)`.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_screen.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): all-rung Stage A screen and the prep tie rule`.

### Task 7: The funnel and the anytime budget (`prep/search.py`)

**Files:**
- Create: `src/nuzline/prep/search.py`
- Test: `tests/prep/test_search.py`

**Interfaces:**
- Consumes: Tasks 3-6; 4b `solve` with `with_budget` states, `ActionValue.e_deaths/distribution`; `HiddenLegality`, `UnsupportedAi`.
- Produces:
  - `@dataclass(frozen=True) PrepSettings: screen_n: int = 8, team_keep: int = 32, variant_rounds: int = 2, variant_keep: int = 8, ladder_keep: int = 4, finalists: int = 3, max_depth: int = 2, precision: float | None = 0.015, finalist_playouts: int | None = None` (a fixed `budget_playouts` instead of the precision target), `max_team: int = 6, relearner: str = "all", tms: bool = True, items: bool = True, entry_hp: bool = True, max_turns: int = 100, budget_seconds: float | None = None, weights: tuple[tuple[str, float], ...] = tuple(DEFAULT_WEIGHTS.items())`. `ValueError` for an odd or non-positive `screen_n`, any `*_keep < 1`, `variant_rounds < 0`, `finalists < 1`, `max_depth < 0`.
  - `@dataclass(frozen=True) Finalist: setup: Setup, result: SolveResult`; `@dataclass(frozen=True) Rung: k: int, setup: Setup, p: float, ci_low: float, ci_high: float, source: str` (`"plan"` or `"screen"`).
  - `@dataclass SearchStats: setups_screened: int, playouts: int, team_rounds: int, variant_rounds: int, variant_halvings: int, phase: str` (last finished: `"teams"`, `"variants"`, `"finalists"`), `stopped: str | None` (`None`, `"budget"`, `"interrupted"`), `seconds: float = field(compare=False)`, `phase_seconds: tuple[tuple[str, float], ...] = field(default=(), compare=False)`.
  - `@dataclass PrepResult: chosen: Setup, screen: ScreenStats, plan: SolveResult | None, ladder: tuple[Rung, ...]` (k = 0..3), `distribution: DeathDistribution` (the plan leader's, else the screen's), `finalists: tuple[Finalist, ...], excluded: tuple[str, ...], pruned: tuple[tuple[Member, Member], ...], stats: SearchStats`.
  - `@dataclass(frozen=True) Progress: phase: str, round: int, alive: int, elapsed: float, leader: Setup` (`phase` is `"teams"`, `"variants <v>"` with `v` 1-based, or `"finalists"`; `round` the halving round `r`, or the finalist index; `alive` is `|S_k|`).
  - `phase_id(phase: str, v: int = 0) -> int`: `"finalists"` 0, `"teams"` 1, `"deep"` 2, `"variants"` `3 + v` (`v` 0-based); disjoint for every `v >= 0`.
  - `run_prep(tables, names, problem, settings, key, *, checkpoint: Path | None = None, resume: Path | None = None, progress: Callable[[Progress], None] | None = None, clock: Callable[[], float] = time.perf_counter) -> PrepResult`: rulings 1-3, 10, 11; a halving round's key is `fold_in(fold_in(key, phase_id(phase, v)), r)`, finalist `i`'s `fold_in(fold_in(key, phase_id("finalists")), i)`, the deep solve's `fold_in(fold_in(key, phase_id("deep")), 0)`. `progress` is called after every unit (Task 8 writes the checkpoint first); a `KeyboardInterrupt` raised during a unit or by `progress` returns the best-so-far result with `stopped="interrupted"`. Ladder: the requested rung `k` from the plan's leader (`source="plan"`), or, without a plan, the `prep_order` leader of `S_k` on rung `k` (`source="screen"`); each other rung `j` is the `prep_order` leader on rung `j` of `S_j` at the last round reached (ruling 2; `source="screen"`), so a setup halved out of `S_k` early can still head rung `j`. Raises `ValueError` when every setup is refused ("every setup refused: ..." with the first reasons).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_search.py`; pool of 3 sample-party candidates vs Roark's real party, `PrepSettings(screen_n=4, team_keep=2, variant_rounds=1, variant_keep=2, finalists=2, max_depth=1, finalist_playouts=200, max_team=2)`, one Heart Scale, unless stated)
  - `test_team_phase_halving` (`ladder_keep=1`): `screen` monkeypatched to record its calls and to give each setup buckets that rank the setups the same on every rung: `(9 setups, 4 playouts)`, `(5, 8)`, `(3, 16)`, then 2 survivors; `stats.team_rounds == 3`; every survivor's accumulated `n == 28` (`T_2`).
  - `test_variant_round_screens_every_neighbour`: the expansion's first call holds exactly the new neighbours of `S_k` (minus pruned and already-screened setups), deduplicated, in one batch, and no parent (each parent's `n` is already at least `T_0`); round 1 tops neighbours and parents alike up to `T_1` (a parent above it gets none); `stats.variant_rounds == 1` and `stats.variant_halvings` equals the number of the expansion's halving calls.
  - `test_parent_can_win_a_variant_round` (`max_depth=0`): `screen` patched so one team-phase survivor beats every one of its neighbours on rung 0: it stays in `S_k` through the expansion and is `chosen`.
  - `test_ladder_keeps_other_rungs_best` (`deaths=0`, `ladder_keep=1`, `max_depth=0`): `screen` patched so setup X has the lowest rung 0 and the highest rung 1 of the 9 in every round: X leaves `S_0` after round 0, is screened in every later team round (it stays in `S_1`), and `result.ladder[1].setup == X` with `source == "screen"`, while `chosen != X`.
  - `test_phase_keys_are_disjoint`: `len({phase_id("teams"), phase_id("deep"), phase_id("finalists"), *(phase_id("variants", v) for v in range(64))}) == 67`; `PrepSettings(variant_rounds=-1)` and `PrepSettings(ladder_keep=0)` raise `ValueError`.
  - `test_finalists_then_deepening`: `solve` monkeypatched to record `settings.max_depth`: `[1, 1]` with `max_depth=1`; `[1, 1, 2]` with `max_depth=2`, the depth-2 solve on the leader of the depth-1 values; `result.plan` is the deep result.
  - `test_budget_stops_after_first_round` (Review Focus 3): `budget_seconds=1` with a `clock` that advances 10 s per call: one `screen` call, `stats.stopped == "budget"`, `plan is None`, `chosen` is that round's `prep_order` leader, every ladder rung `source == "screen"`.
  - `test_interrupt_returns_best_so_far`: `progress` raising `KeyboardInterrupt` at the second unit: `stats.stopped == "interrupted"`, `stats.team_rounds == 2`, `stats.variant_rounds == 0`.
  - `test_refused_setups_are_left_out`: `screen` patched to flag one setup refused: it is never chosen and `excluded` names it; all refused -> `ValueError`.
  - `test_states_carry_budget_and_protection`: `deaths=1`, candidate 0 protected: every screened state has `faint_budget == 1` and `protected` set exactly at candidate 0's slot.
  - `test_reproducible`: two full runs (real playouts, settings above) with one key give equal `PrepResult`s.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_search.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/prep -q` and ruff; expected: pass, file under ~60 s.
- [ ] **Step 5: Commit** `feat(prep): screening funnel, successive halving, Stage B finalists and the time budget`.

### Task 8: Checkpoint and resume (`prep/checkpoint.py`, Stage B result serialization)

**Files:**
- Create: `src/nuzline/prep/checkpoint.py`
- Modify: `src/nuzline/search/stage_b.py` (append `result_to_dict`, `result_from_dict`), `src/nuzline/prep/search.py` (write after every unit; resume)
- Test: `tests/search/test_stage_b.py`, `tests/prep/test_checkpoint.py`

**Interfaces:**
- Consumes: Task 7; 4b `SolveResult`, `ActionValue`, `PlanNode`, `DeathDistribution`.
- Produces:
  - `stage_b.result_to_dict(result: SolveResult) -> dict` and `result_from_dict(d: dict) -> SolveResult`: every dataclass field recursively (obs bytes as hex, int dict keys as strings, `DeathDistribution` by `as_dict`), JSON-safe; `result_from_dict(result_to_dict(r)) == r`.
  - `CHECKPOINT_VERSION = 1`; `fingerprint(problem, settings, seed: int, table_version: str, code_version: str) -> dict[str, str]` (keys `problem`, `settings`, `seed`, `tables`, `code`; `problem` is the sha256 of a canonical JSON of the candidates' specs, options, `levelled` moves and locations, the opponent specs, trainer id, start condition, resources, deaths, protected, prestatus, cap and source; `settings` omits `budget_seconds`).
  - `write_checkpoint(path: Path, doc: dict) -> None` (temp file in the same directory, then `os.replace`); `read_checkpoint(path: Path, expected: dict[str, str]) -> dict`: `ValueError` naming the first differing fingerprint key (`"checkpoint is for another save or pool"`, `"... settings"`, `"... seed"`, `"... tables"`, `"... code version"`), a wrong version, or unreadable JSON ("checkpoint FILE is damaged: ..."); never writes.
  - `setup_to_json(setup) -> list` / `setup_from_json(data) -> Setup`; `stats_to_json` / `stats_from_json` for `ScreenStats`.
  - `run_prep(..., resume=path)` continues at the saved phase, expansion, round and per-rung survivor sets with the saved stats and finalists; elapsed time adds up across sessions in `stats.seconds`.

- [ ] **Step 1: Write the failing tests**
  - `test_stage_b.py::test_result_dict_round_trip`: the depth-2 `pixel_endgame` solve (zero leaf evaluator) and 4b's `sacrifice_endgame` `k=1` solve: `result_from_dict(json.loads(json.dumps(result_to_dict(r)))) == r`.
  - `test_checkpoint.py::test_resume_equals_uninterrupted`: Task 7's small problem; run A uninterrupted; run B with `checkpoint=p` interrupted by `progress` after unit 3; run C `resume=p`: C equals A (`PrepResult` equality; `seconds` excluded by `compare=False`).
  - `test_resume_refuses_mismatch` (Review Focus 2), parametrized: another seed, `deaths=1`, another cap, one candidate's moves changed, another `tables`/`code` string, a file cut in half, `CHECKPOINT_VERSION + 1`: `ValueError` naming it; the file's bytes are unchanged.
  - `test_checkpoint_write_is_atomic`: `os.replace` monkeypatched to raise `OSError` on the second write: the file still holds the first write's JSON.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_checkpoint.py tests/search/test_stage_b.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/prep tests/search -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): checkpoint after every unit and bit-identical resume`.

### Task 9: Setup card, prep steps, ladder and run record (`prep/report.py`)

**Files:**
- Create: `src/nuzline/prep/report.py`
- Test: `tests/prep/test_report.py`

**Interfaces:**
- Consumes: Tasks 3-8; `plan_tree.render`, `plan_tree.to_json` (4b's headers and labels), 4b `all_zero_footer`.
- Produces:
  - `prep_steps(problem, names, setup) -> list[str]` (unnumbered, in this order): `"Deposit <NICK> (party slot N)."` for each party member not brought; `"Withdraw <NICK> (box B slot S)."` for each boxed member; `"Rare Candy x<c> on <NICK> (Lv<a> -> <cap>), then heal at a Pokemon Center."`; per `LEVEL_UP` change `"<NICK> learns <new> at Lv<n> while levelling."` (an empty slot) or `"<NICK> learns <new> at Lv<n> while levelling: forget <old>."` (no Heart Scale); `"Move Relearner (1 Heart Scale): <NICK> forgets <old>, learns <new>."` (`"learns <new> into an empty slot"` when filling); `"TM<nn> (<move>) on <NICK>: forgets <old>."`; `"HM<nn> ..."` plus `"(an HM move needs the Move Deleter to remove)"`; `"Give <NICK> the <item>."`; `"Bring <NICK> to <hp>/<max> HP (<why>)."` with `<why>` from the region (`"at most 70 % keeps the trainer AI's high-HP check off"`, `"at most 1/3: Blaze active"`); `"<Status> <NICK> before the fight."`; `"Party order: <LEAD> first, then <others>."`.
  - `cost_line(problem, names, setup) -> str`: `"Cost: 10 Rare Candies, 1 Heart Scale, TM10"`, or `"Cost: none"`.
  - `setup_card(problem, names, setup) -> list[str]`: one line per member (lead marked): nickname, species, level, ability, item, HP, status, the four moves.
  - `ladder_lines(result: PrepResult, problem, names) -> list[str]`: header `Odds ladder (rung K planned):`; per `k` 0..3 `"  deaths <= k: 85.0 % [83.5, 86.4]  plan"` or `"  screen"`, plus `"  (best for this rung: <lead>, <members>)"` when that rung's setup differs from `chosen`; then `"Deaths (<plan|screen>): 0: a %  1: b %  2: c %  3+: d %  whiteout: e %  unfinished: f %  protected lost: g %"`.
  - `render(result, problem, names, show_depth: int = 1) -> list[str]`: trainer line with cap, deaths allowed and protected names; `"Setup:"` + card; `"Prep steps:"` numbered; cost line; ladder; `"No deathless line found: every setup's 0-death chance is 0 %."` when rung 0 of every screened setup and of the plan is 0; `"Plan:"` + `plan_tree.render(plan, names, show_depth, root=<chosen state>, trainer=...)` or `"screen only, no plan (<stopped reason>)"`; `"Search: ..."` (setups screened, rounds, playouts, finalists, seconds, stopped); `"Left out:"` lines. ASCII only.
  - `record(result, problem, names, settings, seed: int, table_version: str) -> dict`: `trainer_id`, `trainer`, `cap`, `deaths`, `protected` (labels), `reserved`, `settings`, `seed`, `fingerprint`, `setup` (members with labels, moves, item, hp, status), `prep_steps`, `cost`, `ladder`, `distribution` (`as_dict`), `plan` (`plan_tree.to_json` or `None`), `finalists` (label and value per finalist), `stats`, `excluded`, `pruned` (counts).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_report.py`, fabricated `PrepResult`s over fixture candidates)
  - `test_prep_steps_text`: a setup leading a boxed Kricketot (Lv5, cap 16, its `base_member`: Bug Bite learned at 16 filling slot 2) with party Furret (TM10 over Defense Curl, Move Relearner Fury Swipes over Foresight, entry HP 40/57) and Monferno (16/48 HP), Zubat deposited: the exact list, in the order above, holding `"KRICKETOT learns Bug Bite at Lv16 while levelling."` (no Move Relearner step for Kricketot) and `"Move Relearner (1 Heart Scale): FURRET forgets Foresight, learns Fury Swipes."`, ending `"Party order: KRICKETOT first, then FURRET, MONFERNO."`.
  - `test_cost_line`: that setup gives `"Cost: 11 Rare Candies, 1 Heart Scale, TM10"` (the Heart Scale is Furret's; Kricketot's Bug Bite costs none); the same setup without Furret's Fury Swipes: `"Cost: 11 Rare Candies, TM10"`; a base setup of party members at the cap: `"Cost: none"`.
  - `test_ladder_lines`: rung 0 planned 0.85 [0.835, 0.864], rungs 1-3 screen with rung 2's best setup different: the four lines and the `best for this rung` line under rung 2 only; the deaths line folds 3 and 4+ into `3+`.
  - `test_report_without_a_deathless_line` (Review Focus 4): every rung-0 value 0, `deaths=1`: the "No deathless line found" line, rungs 1-3 printed, no exception, all ASCII.
  - `test_screen_only_report`: `plan=None`, `stopped="budget"`: `"screen only, no plan (budget)"`.
  - `test_record_is_json`: `json.loads(json.dumps(record(...)))` round-trips; keys as listed.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_report.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): setup card, costed prep steps, odds ladder and run record`.

### Task 10: `nuzline prep` and the cap default for `coverage --save`

**Files:**
- Modify: `src/nuzline/cli.py` (module docstring, `_parser` :93-159, a new `_prep`, `_write_run_record` :358-371 gains `prefix: str = "solve"`, `main` :410-428: `prep` configures devices; `_coverage`: cap default per ruling 13), `README.md` (prep usage, `config/level_caps.toml`, `NUZLINE_SLOW_TESTS`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3-9; 4a `load_learnsets`, `pool_candidates`; `check_opponent`.
- Produces:
  - `nuzline prep --save S [--allow-corrupt] (--trainer N | --trainer-id I) [--cap N] [--deaths K] [--protect NAME]... [--exclude NAME]... [--reserve ITEM...] [--prestatus STATUS]... [--no-entry-hp] [--weight KIND=W]... [--budget SECONDS] [--checkpoint FILE] [--resume FILE] [--max-depth 2] [--finalists 3] [--screen-n 8] [--seed 0] [--devices N] [--tree FILE.json] [--show-depth 1]`. `--save` is required (no `--party`). `--reserve` uses `action="extend", nargs="+"`. `--resume` with `--checkpoint` is an error. Order: compile cache, trainer, `default_cap` unless `--cap`, `build_problem` (warnings to stderr), `check_opponent` (missing -> print the report, exit 2), `run_prep` (progress lines to stderr: `"[prep] <phase> round <r>: <alive> setups, <s> s"`, `<phase>` as `Progress.phase`, e.g. `variants 1`), `report.render` to stdout, the plan tree JSON with `--tree`, `record` to `runs/prep_<time>.json` (`save_sha256` included) and `"run record: <path>"`, `"checkpoint: <path>"`. Exit 1 on `ValueError`/`OSError` (bad flags, mismatched or damaged checkpoint, no usable Pokemon), 2 when the opponent is unsupported or every setup is refused, else 0 (interrupted and budget-stopped runs included).

- [ ] **Step 1: Write the failing tests** (`tests/test_cli.py`; a synthetic save with party Monferno, Prinplup, Grotle (L14), box 1 slot 0 Starly L10, bag Heart Scale x1 and TM10; `RUNS` monkeypatched to `tmp_path`)
  - `test_cli_prep_small`: `prep --save S --trainer roark --cap 14 --max-depth 1 --finalists 1 --screen-n 4 --devices 1`: exit 0; stdout has `Setup:`, `Prep steps:`, `Cost:`, `Odds ladder (rung 0 planned):`, `Plan:`; the run record exists with `save_sha256`, `"deaths": 0`, `"cap": 14`; the checkpoint file exists; all ASCII.
  - `test_cli_prep_needs_a_cap`: `--trainer-id 858` without `--cap` -> exit 1, stderr mentions `config/level_caps.toml`.
  - `test_cli_prep_bad_flags` (Review Focus 1 and 5), parametrized: `--deaths 4`; `--protect Starly --exclude Starly`; `--exclude` of every member; `--reserve Potionx`; `--prestatus sleep`; `--weight tm=x`; `--protect Monferno` against a save with two Monferno (stderr lists both locations); each exit 1 with the value in stderr.
  - `test_cli_prep_resume`: a run with `--checkpoint F`, then `--resume F` with the same flags: exit 0, the same `Setup:` block; `--resume F --deaths 1` -> exit 1 naming `settings`.
  - `test_cli_prep_unsupported_opponent`: `--trainer-id 858 --cap 60` -> exit 2, `Missing (` in stdout.
  - `test_coverage_save_uses_the_config_cap`: `coverage --save S --trainer roark` prints `Pool at level cap 16`.
  - `test_cli_module_does_not_import_jax` stays green.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q`; expected: the new tests fail.
- [ ] **Step 3: Implement**; README documents the command, its flags, the output sections and resuming.
- [ ] **Step 4: Run** `uv run pytest -q`, `uv run ruff check src tests benchmarks`, `uv run ruff format --check src tests benchmarks`; expected: pass. By hand: `uv run nuzline prep --save <synthetic save> --trainer roark --cap 14 --max-depth 1` prints the full report.
- [ ] **Step 5: Commit** `feat(cli): nuzline prep, with resume, budget, reservations and protection`.

### Task 11: PP-stall test fights

**Files:**
- Modify: `tests/prep/fixtures.py` (`pp_stall_problem`)
- Create: `tests/prep/test_pp_stall.py`, `benchmarks/pp_stall.py`
- Modify: `benchmarks/README.md` (a "Milestone 4: PP stall" section)

**Interfaces:**
- Consumes: Tasks 3-7, `tests.engine.helpers.roark_party`, `roark_trainer`, `damage_table`, `hit_threshold`.
- Produces:
  - `pp_stall_problem(tables, names, stall_pp: int) -> PrepProblem` (ruling 15): opponent `(Cranidos with moves (Rock Throw, Leer, 0, 0), pp (stall_pp, 30, 0, 0))`, trainer `replace(roark_trainer(), items=())`; candidates in this order: tank Geodude (moves `(Defense Curl,)`, `stats` overridden), attacker Starly, decoy Shinx (both with `stats` overridden), no resources, cap = their level. The fixture asserts, from `damage_table` on the turn-0 states (crits included): the max crit Rock Throw at the tank's Defense -6 times `stall_pp` is below the tank's max HP; Rock Throw's min non-crit roll KOs each attacker; each attacker's best move's min roll is at least half Cranidos's max HP and its max non-crit roll below it; Rock Throw's hit threshold is that of 90 % accuracy.
  - `benchmarks/pp_stall.py [--stall 5 20 30] [--devices N]`: runs `run_prep` with default settings on each `pp_stall_problem`, writes `benchmarks/results/pp_stall_<time>.json` (per stall length: chosen team, lead, rung-0 plan estimate and interval, whether the chosen team holds the tank and an attacker, the best tank-less screen value, seconds) and prints one line per length.

- [ ] **Step 1: Write the test** (`tests/prep/test_pp_stall.py`, `PrepSettings(screen_n=16, team_keep=4, variant_rounds=0, finalists=2, max_depth=1, finalist_playouts=2000, max_team=3)`, `PRNGKey(0)`)
  - `test_short_stall_ranked_first`: `stall_pp=5`: the chosen setup holds the tank and an attacker; its plan estimate `>= 0.9`; the plan's first action is not a switch to an attacker while Rock Throw has PP (`ranking[0].label` is `Defense Curl` with the tank leading, `Switch -> Geodude` otherwise); every screened setup without the tank has an `interval(0)` upper bound below the chosen setup's ladder rung-0 `ci_low`.
  - `test_tank_alone_does_not_win`: the `(tank)` setup's screen rung 0 is 0 (Cranidos never faints: the battle hits the turn cap, bucket `unfinished`).
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_pp_stall.py -q`.
  Expected: pass. If `test_short_stall_ranked_first` fails, do not tune the fixture to the search: report which phase drops the stall line (screen numbers of the tank setups, the finalists' Stage B values) to the controller before changing the playout policy.
- [ ] **Step 3: Run** `uv run python -m benchmarks.pp_stall --stall 5 20 30` and record in `benchmarks/README.md` a table: stall length, found (yes/no), rung-0 estimate with interval, seconds; with one sentence on how far the long stalls fall short (spec §10.2: reported, not blocking).
- [ ] **Step 4: Run** ruff on `tests benchmarks`; expected: pass.
- [ ] **Step 5: Commit** `test(prep): PP-stall fights (short stall required, long stall reported)`.

### Task 12: Exhaustive check of the funnel on Roark

**Files:**
- Create: `src/nuzline/prep/exhaustive.py`, `benchmarks/exhaustive_roark.py`, `tests/prep/test_exhaustive.py`, `benchmarks/exhaustive_roark.json` (the overnight result, committed when the run finishes)
- Modify: `benchmarks/README.md` ("Milestone 4: exhaustive check")

**Interfaces:**
- Consumes: Tasks 2-8 (`designated_change`, `single_changes`, `prune`, `base_member`, `setup_states`, `run_prep`, `write_checkpoint`/`read_checkpoint`), `solve`.
- Produces:
  - `restricted_setups(tables, problem, max_team: int = 6) -> list[Setup]`: every team of `1..max_team` and lead, each member its `base_member`, plus (when `problem.resources.heart_scales >= 1`) each with exactly one member on its `designated_change`. Count: `sum(comb(n, k) * k * (1 + d_k))` where `d_k` is the team's number of members with a designated change.
  - `pruned_setups(tables, problem, max_team: int = 6) -> tuple[list[Setup], list[tuple[Member, Member]]]`: `prune` over each candidate's `single_changes(relearner="all", tms=False, items=False, entry_hp=False)`; for every dropped variant, every team of `1..max_team` holding that candidate, with every lead, that member on the dropped variant and the others base (the restricted shape); with the (dropped, dominated by) pairs. Disjoint from `restricted_setups` unless the dropped variant is a candidate's designated one (then it is left out here).
  - `@dataclass(frozen=True) Value: estimate: float, ci_low: float, ci_high: float, e_deaths: float`.
  - `exhaustive_values(tables, names, problem, setups, key, *, depth: int, n_screen: int = 400, budget_playouts: int | None = None, precision: float | None = None, done: Mapping[int, Value] | None = None, on_value: Callable[[int, Value], None] | None = None) -> dict[int, Value]`: depth 0 values every setup by `screen` (`n_screen` playouts, rung `problem.deaths`, Wilson interval); depth 1 by `solve(..., SolveSettings(max_depth=1, budget_playouts=budget_playouts, precision=precision))` on its state (the best ranked action's estimate, interval and `e_deaths`; no ranking -> all 0); setups in `done` are skipped; `on_value` after each (the benchmark checkpoints there); keys `fold_in(key, setup index)`.
  - `@dataclass(frozen=True) Verdict: best: int, pick: int, pick_ties_best: bool, beaten_by_pruned: tuple[int, ...], passed: bool`; `verdict(values, pick: int, pruned: Collection[int]) -> Verdict`: `best` = highest estimate (ties: lower index); `pick_ties_best` = the pick's interval overlaps the best's; `beaten_by_pruned` = pruned setups whose `ci_low` exceeds the pick's `ci_high`; `passed` = tie and none beaten. `values` covers `restricted_setups + pruned_setups` (the pruned ones indexed after the restricted ones); `pruned` is those indices; `best` is taken over the restricted ones.
  - `benchmarks/exhaustive_roark.py --save S [--resume FILE] [--devices N]`: the user's save (`NUZLINE_USER_SAVE` if `--save` is omitted), pool per ruling 14, `--cap` from config, no items, full HP; the valued setups are `restricted_setups` plus `pruned_setups` (the prune run with `relearner="all"`; when it drops nothing the result says "prune dropped nothing on this save: the pruned-setup criterion is vacuous here" and the check rests on `test_pruned_setups_are_valued`); for `k` in (0, 1): the funnel's pick (`run_prep` with `relearner="designated"`, `tms=False`, `items=False`, `entry_hp=False`, `max_depth=1`), phase 1 at `budget_playouts=2000`, phase 2 at precision 0.015 on the overlapping setups plus the pick plus every pruned setup whose phase-1 interval reaches above the pick's phase-1 `ci_high`, `verdict` on phase 2; the ETA line after 20 setups; checkpoint (Task 8's atomic writer) after every setup; writes `benchmarks/exhaustive_roark.json`: save sha256, pool labels, setup count, per rung the best and pick (labels, values), `passed`, pruned setups beating the pick, phase seconds, machine (`benchmarks.rank_timing.machine`).

- [ ] **Step 1: Write the failing tests** (`tests/prep/test_exhaustive.py`)
  - `test_restricted_setups_count`: fixture pool of 4 with designated changes for 3: the formula's count; every setup has at most one member with a move change, and it is designated.
  - `test_verdict`: hand-made values: an overlapping pick passes; a disjoint pick fails; a pruned setup above the pick fails.
  - `test_pruned_setups_are_valued`: a fixture pool of 3 (Monferno, Prinplup, Starly at 14) where Starly (full set Growl, Quick Attack, Wing Attack, Double Team, `candies=0`) has relearner options Tackle and Scratch and there is one Heart Scale: `pruned_setups(..., max_team=2)` is non-empty, every setup in it has Starly on its Tackle variant, the pairs name Scratch's variant as the dominator; `exhaustive_values(depth=0, n_screen=64)` over `restricted_setups + pruned_setups` returns a value for every index, and `verdict` with those indices as `pruned` reports a pruned setup whose `ci_low` is forced (by an injected value) above the pick's `ci_high` in `beaten_by_pruned`.
  - `test_funnel_matches_exhaustive_small` (default suite): pool Monferno, Prinplup, Starly from the sample party at level 14 vs Roark's real party, `max_team=2`, `k` in (0, 1): `exhaustive_values(depth=0, n_screen=400)` over the 9 setups, the funnel (`max_depth=0`, `screen_n=16`, `team_keep=2`) picks a setup whose interval overlaps the best's.
  - `test_funnel_matches_exhaustive_ci` (default suite; spec §8's CI version): pool Monferno, Prinplup, Grotle, Starly at 14 with designated relearner variants and one Heart Scale, `max_team=2`, `k=0`: depth 1 with `budget_playouts=100` for every restricted setup (44 when all 4 have a designated variant: `4 * 2 + 6 * 2 * 3`) plus `pruned_setups`; the funnel with `max_depth=1`, `finalist_playouts=100`, `screen_n=8`, `team_keep=4`, `variant_keep=2`, `finalists=2`, `relearner="designated"`, `tms=False`, `items=False`, `entry_hp=False`; `verdict(...).passed`. Its budget: the whole file under ~60 s on one device (Step 3 measures it).
  - `test_funnel_matches_exhaustive_ci_full` (`NUZLINE_SLOW_TESTS=1` only): the same pool with `max_team=3`, `k` in (0, 1), `budget_playouts=500`, `finalist_playouts=500`; `verdict(...).passed` on both rungs.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_exhaustive.py -q`; expected: fail; then implement `prep/exhaustive.py` and `benchmarks/exhaustive_roark.py`.
- [ ] **Step 3: Run** `uv run pytest tests/prep/test_exhaustive.py -q --durations=0` and `NUZLINE_SLOW_TESTS=1 uv run pytest tests/prep/test_exhaustive.py -q`; expected: pass; note both runtimes in `benchmarks/README.md`. If the default-suite file is over ~60 s, lower `budget_playouts`/`finalist_playouts` to 50 and re-measure; if it is still over, report the measured time to the controller instead of gating the test.
- [ ] **Step 4: Start the one-off job** (unattended): `uv run python -m benchmarks.exhaustive_roark --save "$NUZLINE_USER_SAVE"`; resume with `--resume` after any stop. When it finishes, commit `benchmarks/exhaustive_roark.json` and a README section (setups, rungs, pass or fail per rung, wall time). A failed verdict is reported to the controller as a finding, not hidden.
- [ ] **Step 5: Commit** `test(prep): exhaustive check of the funnel on Roark (CI size and overnight benchmark)` (and later `bench(prep): exhaustive Roark result`).

### Task 13: Cold timing on the user's save

**Files:**
- Create: `benchmarks/prep_timing.py`, `tests/test_prep_timing.py`
- Modify: `tests/test_user_save.py` (4a), `benchmarks/README.md` ("Milestone 4: `nuzline prep` wall time")

**Interfaces:**
- Consumes: Task 10 (CLI, run record `stats`), `benchmarks.rank_timing.machine`, `RESULTS`.
- Produces:
  - `TARGET_SECONDS = 900.0`; `prep_command(save: Path, devices: int | None) -> list[str]` (`nuzline prep --save S --trainer roark`, defaults otherwise); `parse_record(stdout: str) -> dict` (reads the `run record:` path's JSON: seconds, phase seconds, setups screened, playouts, chosen setup, rung values); `run_prep_timed(save, devices, cache_dir) -> dict` (subprocess, wall time); `main(argv)`: cold (fresh temporary `NUZLINE_CACHE_DIR`) then warm, writes `benchmarks/results/prep_<time>.json`, exits non-zero when the cold run has no `Setup:` block.

- [ ] **Step 1: Write the tests**
  - `tests/test_prep_timing.py::test_parse_record`: a fake stdout with a `run record:` line pointing at a temporary record gives the expected dict; a stdout without one raises `ValueError`.
  - `tests/test_user_save.py::test_prep_on_the_user_save` (skipped without `NUZLINE_USER_SAVE`): `main(["prep", "--save", path, "--trainer", "roark", "--max-depth", "1", "--finalists", "1", "--budget", "300", "--devices", "1"])` returns 0; output holds `Setup:`, `Odds ladder (rung 0 planned):` and no `left out` line for any of the 11 Pokemon.
- [ ] **Step 2: Run** `uv run pytest tests/test_prep_timing.py -q` and, with the save, `NUZLINE_USER_SAVE=... uv run pytest tests/test_user_save.py -q`; expected: pass.
- [ ] **Step 3: Measure** `uv run python -m benchmarks.prep_timing --save "$NUZLINE_USER_SAVE"`; record in `benchmarks/README.md`: cold and warm wall time against 900 s, phase seconds, setups screened, playouts, the chosen setup and its rung-0 value. If the cold run is over 900 s, record the miss first, then lower the defaults in this order and re-measure after each: `team_keep` 32 -> 16, `finalists` 3 -> 2, `variant_rounds` 2 -> 1; record every measurement and change `PrepSettings` only to the first setting that meets the target.
- [ ] **Step 4: Run** `uv run pytest -q`, `uv run ruff check src tests benchmarks`; expected: pass.
- [ ] **Step 5: Commit** `bench(prep): nuzline prep cold wall time on the user's save`.

---

## Self-review notes

- Spec coverage: §6.1 candidate pool -> Task 3 (4a's pool, exclusions, eggs via `pool_candidates`); move, item, entry options -> Task 4; entry HP brackets and the prestatus toggle -> Task 4 (rulings 5, 6), CLI Task 10; conservative prune -> Task 4 (ruling 8); Stage A screen of every survivor in one batch, all rungs at once -> Tasks 1, 6, 7 (rulings 1, 10); successive halving -> Task 7 (ruling 2); Stage B on finalists of the requested rung at the precision target -> Task 7 (ruling 3); anytime budget -> Task 7 (ruling 11); checkpoint and resume -> Task 8 (ruling 12); batched `make_state` -> Task 2. §6.1/§10.2 PP stall -> Task 11. §7 setup card, ladder, plan tree with replacement labels (4b's `render`), run record, CLI, `config/level_caps.toml` -> Tasks 3, 9, 10. §8 exhaustive (overnight + CI) -> Task 12 (ruling 14); timing target -> Task 13. §4 tie-breaks -> Task 6 (ruling 7); §10.1 protected and "leave it out" -> Tasks 3, 5, 7, 10. Out of scope here: the golden trace with the user (§8, §9 item 5), evolution and tutors (§2 non-goals).
- Review round 2 dispositions: kill-check HP cuts (ruling 5, Task 4); levelled moves free and in the base (ruling 4, Tasks 3, 4, 9); variant expansions and per-rung survivor sets defined (ruling 2, Task 7); ladder rungs from their own survivor sets (Task 7); pruned setups valued in the exhaustive check (ruling 14, Task 12); the prune called a heuristic (ruling 8); ability-blocked prestatus (ruling 6); disjoint phase keys (Task 7); `check_opponent` without AI refusals (Task 3); the 4-Pokemon depth-1 CI test in the default suite (Task 12). No 4a/4b interface changed: `levelled` is computed in 4c from 4a's `Learnsets.level_up` and `Candidate.candies`.
- Consumed names: 4a `Candidate`, `pool_candidates`, `check_pool`, `Learnsets` (`level_up`), `load_learnsets`, `level_to_cap`, `player_move_refusal`, `player_ability_refusal`, `calc_stats(..., species=)`; 4b `with_budget`, `MAX_DEATH_BUDGET`, `deaths.BUCKETS`, `DeathDistribution` (`rung`, `p_win`, `as_dict`, `from_counts`), `LeafValue.deaths_a/deaths_b/buckets_b/merge`, `PlayoutResult.deaths/ended`, `ActionValue.e_deaths/distribution`, `SolveResult.deaths_allowed/protected`, `plan_tree.render`/`to_json`, `all_zero_footer`.
