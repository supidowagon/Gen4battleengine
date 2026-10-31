# Stage B Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `nuzline solve`: an exact, depth-limited expectimax over information sets for Roark's fight, with Stage A playouts at the leaves, that ranks the first move and exports a plan tree branching on what the player can see.

**Architecture:** Three JAX building blocks, all fixed-shape and batched: per-field outcome laws (`engine/enumerate.py`), an observation vector (`engine/observe.py`) and a turn expander that steps, replaces and observes padded chunks of (state, actions, outcome) rows (`search/expand.py`). A host-side Python search (`search/stage_b.py`) builds the belief tree level by level, merges children by bytes, groups them by observation, evaluates leaves with chunked Stage A playouts (`search/leaves.py`), and backs values up. `search/plan_tree.py` renders and exports the result; `cli.py` adds `solve`.

**Tech Stack:** Python 3.12, JAX (CPU, sharded over host devices as in Stage A), numpy (float64 on the host), pytest, hypothesis, ruff.

**Spec:** `docs/superpowers/specs/2026-10-30-stage-b-solver-design.md` (parent `docs/design.md` §4.2, §4.3). Code facts with file:line, verified at commit 6e817e1: `docs/superpowers/notes/stage-b/{outcomes,observe,integration,critic}.md`. Read the notes sections a task names before starting it.

## Global Constraints

- No new runtime dependencies; numpy, jax and the standard library only.
- The decomp (`external/pokeplatinum`) is authoritative for every mechanic and for the HP bar; cite file:line in docstrings as the existing code does.
- Never approximate silently: anything not enumerated exactly is refused by name or reported (sampled mass, refused mass, unexplored mass).
- The player's plan never conditions on hidden information (spec *Anti-goals*). Leaf playouts may (spec §3, controller's call) and the output says so.
- Probabilities on the host are float64; battle-outcome laws are computed on the host in float64 from the integer thresholds and odds JAX returns, never from float32 device values. AI rows stay float32 (exact up to float32 rounding); tests use 1e-12 for battle-outcome sums and 1e-6 once AI weights enter.
- Reproducible: same seed, settings, tables and code give bit-identical output whatever the host device count (with `budget_seconds` set, only the depth reached may differ; ruling 9).
- Existing tests are not weakened; `tests/engine/test_coverage.py:92-104` (no mechanic graded `SAMPLING_ONLY`) stays (sampling-only arises at runtime from the width cap).
- CLI output is ASCII only; `nuzline.cli` does not import jax at module import (`tests/test_cli.py:201-211`); exit codes stay 0 ok / 1 error / 2 refused.
- Do not register `Tables`, `BattleState` or `PlayoutResult` for `jax.export` a second time (it raises); import `nuzline.search.stage_a`, which registers them. Register new pytrees once, in the module that exports them.
- Every jitted program whose batch size varies goes through one fixed padded chunk shape and `jaxcache.cached_export`, so a solve compiles each program once per device count. Padded size is `-(-CHUNK // devices) * devices` (padding = copies of row 0, dropped after the call), as `stage_a.rank_actions` does; plain `CHUNK` does not divide by 20 devices.
- Test runtime: no default-suite test runs a depth-2 solve except the 1v1 endgames with a zero leaf evaluator; each new test file stays under ~60 s on one device (each new program shape costs ~20 s to compile cold, notes/critic.md B.3).
- ruff (`line-length = 100`); match the surrounding code's docstring density and citation style.

## Rulings (controller's calls; the spec is silent)

1. **Observation route**: computed in JAX from `(before, actions, outcome, after)`, `after` = the state after `send_replacements`. Which side's action ran is read from PP deltas, switches and items (notes/critic.md B.2). Every outcome-derived flag is **effect-gated**: it is set only when the draw had a visible effect, judged with the battler's status at the moment it acts (notes/observe.md §2.2-2.3), never from `before` or `after` status alone. A draw with no effect never changes the observation.
2. **Merge key**: `(state bytes, observation bytes)`; equal states with different observations stay separate children (notes/outcomes.md §4).
3. **Rolls are not pre-merged**; all 16 are stepped and merged afterwards. Width cap **131072** pre-merge combinations per (state, action pair); above it the pair is sampling-only. Relevance may over-approximate (extra rows are harmless because `step` merges them and `observe` is effect-gated), except `speed_tie` and `end_speed_tie`, which may over-approximate only as {0,1} with equal mass. Multi-hit moves will be sampling-only (none in Roark's fight).
4. **Refusals below the root** (AI refused/NaN, replacement refused, refused leaf playout): valued **0** in both estimate and lower bound; reach mass reported as `refused`. A refused AI at the root refuses the solve (exit 2, as `nuzline rank`). A root player action whose own turn is sampling-only (for any opponent action with mass) is dropped from the ranking and named with the reason.
5. **Max bias**: every evaluation has two halves with independent key streams. Half A picks the action at each inner belief node; half B gives every reported value and interval. The root ranking uses half-B values and `tied_with_best`.
6. **Interval**: for root action `a` under the half-A policy, let `W_j` be the total absolute reach of every tree leaf (and pruned or sampled node) that maps to evaluated state `j` (transpositions share one evaluation, so their weights add before squaring). `var(a) = sum_j W_j^2 * var_j`, `var_j = p~(1-p~)/n_b` with `p~ = (k_b+1)/(n_b+2)`; interval `estimate +- 1.96*sqrt(var(a))`, clipped to [0, 1]. Terminal and refused children have variance 0. Implementation: carry `{eval_id: W}` per node during backup, merge with weights at chance and observation nodes, take the chosen child's map at decision nodes, square only at the root.
7. **Depth 0 is Stage A**: `max_depth=0` returns `rank_actions(n=n_depth0)`'s numbers (Wilson intervals) unchanged. It is never run otherwise.
8. **Leaf allocation**: a depth-`d` pass has leaf budget `B_d`; evaluated state `j` with chance weight `c_j` (sum over its tree leaves of the product of chance probabilities on the path; player choices excluded) gets `n_j = 2 * max(f, round(B_d * c_j / sum c / 2))`, `f = leaf_floor = 2` per half. Pruned and sampled nodes get exactly `2f`. The floor may push a pass above `B_d`; `SolveResult.playouts` counts every playout run.
9. **Budgets**: `budget_playouts` is `B_d` for every pass `d >= 1`; with `None`, `B_d = n_depth0 * n_root_actions`. Iterative deepening runs passes `d = 1..max_depth`; a pass is never cut short; `budget_seconds`, when set, only decides whether the next pass starts (it does not start once elapsed time exceeds the budget). The deepest completed pass is the result; depth 1 always runs.
10. **Absolute reach** (`prob` on plan nodes, `c_j`, the epsilon test) is the product of chance probabilities along the path: belief weight x opponent-action probability x outcome probability; player choices excluded.

## Review Focus

1. **A root action whose turn is sampling-only** while others are fine: the others are ranked; the dropped action is named with its reason. Test in Task 5.
2. **A time budget that runs out after depth 1**: `solve` returns the completed depth-1 result, not an error or an empty ranking. Test in Task 5.
3. **An opponent replacement refused below the root**: the mass appears as `refused`, valued 0, never a win, no exception. Test in Tasks 3 and 5.
4. **Only one legal player action**: solve and plan tree work with one action. Test in Task 6.
5. **Every child of the root terminal**: no leaf playouts, exact values, zero-width intervals. Test in Task 5.

## Shared test fixtures (`tests/search/fixtures.py`, created in Task 3, extended in Task 5)

- `shinx_vs_cranidos(tables, names, slower_opp=False)`: `roark_state(tables, names, player_species="Shinx", opp_species="Cranidos")` (Spark connects on pure Rock, 30 % paralysis; Cranidos knows Tackle). With `slower_opp`, set the opponent's Spe stat one below Shinx's so Spark lands before Tackle and can paralyze it.
- `spike_cannon_root(tables, names)`: `roark_state(..., player_moves=["Spike Cannon", ...], opp_moves=["Spike Cannon", ...])` (id 131, multi-hit both sides; exceeds the width cap).
- `switch_refusal_root(tables, names)`: `tests.ai.fixtures.trace_f(tables, names).replace_hp(1, 0, 1)`: Cranidos active (move_hit = Wing Attack), Geodude (slot 0) at 1 HP vs Starly; the AI voluntarily switches to Geodude (verified p ~ 0.556) and Starly's Tackle KOs it, so the replacement is refused. Assert `ai_rows` puts mass > 0 on that switch.
- `all_terminal_root(tables, names)` (Task 5): `roark_state(tables, names, player_count=1, opp_count=1, player_moves=["Scratch", "Ember", "Mach Punch", "Quick Attack"])`, Roark's bag emptied (`trainer_items[1] = 0`, `trainer_item_count[1] = 0`), Geodude at 1 HP. Assert the AI row has no item or switch mass and `legal_actions` holds only the 4 moves.
- `pixel_endgame(tables, names)` (Task 5): 1v1 `roark_state` with the bag emptied, opponent `max_hp` set to 999 (pixels ~20 HP wide) and HP `H`, player moves = one sure weak move (accuracy 100) and one stronger move with accuracy < 100; `H` and moves chosen so that turn-1 rolls leave the opponent at several HP values inside one pixel straddling the weak move's KO threshold, making the weak move best for the low-HP states and the strong move for the high-HP ones on turn 2; player HP and opponent damage chosen so the fight ends within a 2-turn cap. The fixture asserts `max_hp[1, 0] > 48` and that two reachable turn-1 children share one observation.
- `spore_endgame` and `taunt/confuse` variants (Task 2, 5): 1v1 `roark_state` with Spore (147) / Confuse Ray (109) / a Taunt or Bind move from the tables, as the tests need.

---

### Task 1: Outcome laws and exact enumeration (`engine/enumerate.py`)

Read first: notes/outcomes.md §1-3 and §7, notes/critic.md C2, C10; rulings 3.

**Files:**
- Create: `src/nuzline/engine/enumerate.py`
- Test: `tests/engine/test_enumerate.py`

**Interfaces:**
- Consumes: `outcome.hit_threshold`, `outcome.crit_odds`, `outcome.empty_outcome`, `Outcome`, `order.selected_move`, `effective_speed`, tables fields (notes/critic.md C10).
- Produces:
  - `class OutcomeLaws(NamedTuple)`: fixed-shape, vmappable integer law parameters (`hit_threshold int32[2]`, `crit_odds int32[2]`, crit blocked by Lucky Chant, secondary chance, duration kind per side) and `relevant`, a `NamedTuple` of `bool` arrays shaped like each `Outcome` field.
  - `outcome_laws(tables, state, actions) -> OutcomeLaws` (JAX).
  - `class SamplingOnly(Exception)` carrying the reason; `WIDTH_CAP = 131072`.
  - `enumerate_outcomes(laws: OutcomeLaws) -> tuple[Outcome, np.ndarray]` (host; `laws` one numpy row; returns numpy `Outcome` with leading axis K and float64 probs[K]): the product over relevant fields only; irrelevant fields take the `empty_outcome` value. Raises `SamplingOnly` above `WIDTH_CAP`.

- [ ] **Step 1: Write the failing tests** in `tests/engine/test_enumerate.py`:
  - `test_probs_sum_to_one`: `roark_real_state` and every legal pair with AI mass > 0: `abs(probs.sum() - 1) < 1e-12`, all probs > 0, and `K <= 16384`.
  - `test_switches_only_fixed`: on `roark_real_state`, every switch-vs-switch pair: each row equals `empty_outcome()` on every field except `speed_tie` and `end_speed_tie`; those two take values in {0,1} with equal mass when they vary; stepping every row and merging by state bytes gives exactly one child with probability 1.
  - `test_speed_tie_enumerated`: a state with equal effective speeds and same-kind actions: `speed_tie` rows 0 and 1 with mass 1/2 each.
  - `test_field_marginals_match_decomp_laws`: hypothesis states (extend `tests/strategies.py` as needed so Lucky Chant, paralysis, freeze, confusion and moves with TAUNT, TRAP, MULTI_HIT, sleep, 0 % and non-zero secondary chance occur). The test transcribes the laws itself, citing outcome.py:143-193: hit `clip(t,0,100)/100` (t from `hit_threshold`), crit `1/odds` or 0 under Lucky Chant on the defender, secondary `chance/100` or 1 when chance is 0, `n_hits` 3/8, 3/8, 1/8, 1/8, duration Taunt {3,4,5}, Trap {3..6}, else {2..5} uniform, para 1/4, confusion self-hit 1/2, thaw 1/5, contact 3/10, ties 1/2. For every field marked relevant, its marginal under `enumerate_outcomes` equals the transcription exactly (`fractions.Fraction` or float64 within 1e-15). Skip examples above `WIDTH_CAP`.
  - `test_mask_sound`: hypothesis (about 200 examples): a supported state, a legal pair, an arbitrary outcome. For every field marked irrelevant, copy the arbitrary outcome's value into each enumerated row; `step` gives byte-identical states to the unmodified rows. This pins "never under-approximate".
  - `test_matches_sampled_frequencies`: pairs Ember vs Rock Throw and a switch (to Prinplup) vs Rock Throw on `roark_real_state`; Spark vs Tackle on `shinx_vs_cranidos` and on `shinx_vs_cranidos(slower_opp=True)` (assert some enumerated row leaves the opponent paralyzed). Sample `N = 20000` outcomes (`jax.random.split(jax.random.key(0), N)`; do not change the seed to make the test pass), step, bucket by state bytes; every sampled bucket exists in the enumeration; buckets with `N*p >= 5` satisfy `|freq - p| <= 5*sqrt(p(1-p)/N) + 1e-9`; buckets with `N*p < 5` are pooled into one bucket checked by the same rule.
  - `test_matches_sampled_frequencies_random`: the same check (N = 5000) for 15 hypothesis examples with fixed seeds.
  - `test_reference_child_set`: for 10 generated cases, stepping each enumerated row through `ref_step` (via `to_ref`/`from_ref`, as `tests/test_differential.py` does) gives the same merged child states and probabilities as `step`.
  - `test_width_cap`: `spike_cannon_root` (Spike Cannon vs Spike Cannon) raises `SamplingOnly` naming the cap.
  - `test_sleep_duration_enumerated`: Spore on a non-sleeping opponent covers every sleep duration with its probability.
- [ ] **Step 2: Run** `~/bin/uv.exe run pytest tests/engine/test_enumerate.py -q`; expected: fail (module missing).
- [ ] **Step 3: Implement** `outcome_laws` (relevance per notes/outcomes.md §7; ruling 3) and `enumerate_outcomes` (host product over the relevant fields' supports, float64 probabilities from the integer parameters).
- [ ] **Step 4: Run** the tests and `~/bin/uv.exe run ruff check src tests`; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): exact per-field outcome laws and enumeration`.

### Task 2: Observations (`engine/observe.py`)

Read first: notes/observe.md (all), notes/critic.md B.1, B.2, B.6, C1, C5; ruling 1.

**Files:**
- Create: `src/nuzline/engine/observe.py`
- Test: `tests/engine/test_observe.py`

**Interfaces:**
- Consumes: `BattleState`, `Outcome`, `selected_move`, tables; Task 1 for the grouping tests.
- Produces:
  - `hp_pixels(hp, max_hp) -> int32` = `0 if hp == 0 else max(1, hp * 48 // max_hp)` (healthbox.c:1559-1590), vectorized.
  - `OBS_SIZE: int`; `OBS_FIELDS: tuple[tuple[str, slice], ...]` (name and slice of each field, for Task 6's labels).
  - `observe(tables, before, actions, outcome, after) -> int32[OBS_SIZE]` (JAX, vmappable).

The vector holds, for this turn: the action each side ran (PP deltas, switches, items); turn order as seen (who acted first, when both ran); per side, effect-gated (ruling 1): missed, crit (the hit landed and dealt damage), blocked by full paralysis, flinched, stayed asleep or frozen, thawed, woke, hurt itself in confusion; and the visible projection of `after`: both actives, species, statuses, the non-zero-ness of `confusion`, `taunt`, `trapped`, `leech_seed`, `substitute_hp`; stat stages; hazards, screens, weather; items held and consumed; revealed abilities; the player's exact HP per party slot; the opponent's `hp_pixels` per party slot; the player's PP; `over`, `winner`. Never: the opponent's exact HP, `status_counter`, counter values, `ai_*` fields, any outcome field without a visible effect.

- [ ] **Step 1: Write the failing tests**:
  - `test_hp_pixels_matches_decomp`: cases (0 -> 0; 1/999 -> 1; 999/999 -> 48; 24/48 -> 24; 47/48 -> 47) and, for every `max_hp` in 1..999 and every hp, equality with the decomp's two-branch formula transcribed in the test.
  - `test_opponent_hp_within_pixel_equal`: opponent `max_hp` set to 999: two `after` states differing only in opponent HP inside one pixel observe equal; one pixel apart, different.
  - `test_player_exact_hp_splits`: player HP differing by 1 observes different.
  - `test_hidden_counters_grouped` (parametrized: sleep duration via Spore; confusion duration via Confuse Ray; Taunt duration; Bind/trap duration): outcomes differing only in the counter observe equal; counter 0 versus non-zero in `after` observes different.
  - `test_visible_message_classes` (parametrized, one row per class, pairs from one parent that differ only in that class, observations differ): stat stage (Leer hit vs miss), weather, Stealth Rock set, screen, item consumed/activated (berry or Potion), ability reveal, thaw, full paralysis, flinch, confusion self-hit, switch target, replacement, turn order.
  - `test_ineffective_draws_invisible`: pairs from one state differing only in the named field observe equal: `full_para=1` on a side that is not paralyzed; `shinx_vs_cranidos` where Spark paralyzes the opponent after it already acted, varying the opponent's `full_para`; `thaw` on a side that is not frozen; `confusion_hit` on a side that is not confused; `crit` on a missed hit, on a status move, and on hit indices after the KO; `speed_tie` when speeds differ or only one side ran. Plus a hypothesis companion over Task 1's enumeration (Roark root, `spore` and `shinx_vs_cranidos` states): perturbing any field whose effect gate is false (same `after` bytes) leaves `observe` unchanged.
  - `test_crit_and_miss_visible`: a crit and a non-crit reaching the same `after` HP, and a miss vs a hit, observe different when the move ran.
  - `test_unrun_choice_invisible`: the opponent faints before acting; its different chosen moves observe equal (notes/critic.md B.1).
  - `test_legality_constant_on_equal_obs`: for `roark_real_state` and hypothesis states, every legal pair: enumerate (Task 1), step, `send_replacements`, observe, group children by observation bytes; `legal_actions(tables, child, 0)` is equal within each group. Deterministic cases must produce a group of at least 2 (Spore across durations; a 999-max-HP opponent hit by a rolled move), and one case must split on `taunt[0] > 0` / `trapped[0] > 0` so the test fails if those are left out of the vector.
- [ ] **Step 2: Run** the tests; expected: fail.
- [ ] **Step 3: Implement** `hp_pixels`, `observe`, `OBS_SIZE`, `OBS_FIELDS`.
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): player observations and the decomp HP bar`.

### Task 3: Turn expansion (`search/expand.py`)

Read first: notes/integration.md §1-4 and §13, notes/critic.md B.3, B.4, C2, C4.

**Files:**
- Create: `src/nuzline/search/expand.py`, `tests/search/fixtures.py` (the shared fixtures above, except those marked Task 5)
- Test: `tests/search/test_expand.py`

**Interfaces:**
- Consumes: Task 1 (`outcome_laws`, `enumerate_outcomes`, `SamplingOnly`), Task 2 (`observe`), `step`, `playout.send_replacements` and the `_play` recipe (playout.py:179-199: `ai_switched = is_switch(actions[1])`, Pursuit `power_mul`), policy `_distribution`, `cached_export`.
- Produces:
  - `CHUNK = 16384` (rows per program call before device padding).
  - `ai_rows(tables, states: BattleState (numpy, N rows)) -> np.ndarray float32[N, 14]` (NaN row = refused).
  - `laws_rows(tables, states, actions int32[N, 2]) -> OutcomeLaws` (numpy).
  - `step_rows(tables, states, actions, outcomes) -> tuple[BattleState, np.ndarray bool[N], np.ndarray int32[N, OBS_SIZE]]` (after-replacement states, replacement refused, observations; one cached program).
  - `@dataclass Child: state_row: int, obs: bytes, prob: float, refused: bool`.
  - `expand(tables, states: BattleState (numpy, N rows), pairs: list[tuple[int, int, int]]) -> tuple[BattleState, dict[int, list[Child] | SamplingOnly]]`: request `r = (state index, player action, opponent action)`; the dict key is `r`'s position in `pairs`; the returned store holds only this call's merged child states and `Child.state_row` indexes it (rows are local to the call). Children are merged by ruling 2 with float64 probabilities summing to 1. `expand` **streams**: it builds, steps and merges one `CHUNK` of rows at a time, so peak memory is O(CHUNK + merged children).

- [ ] **Step 1: Write the failing tests**:
  - `test_children_sum_to_one`: every request on `roark_real_state`'s legal pairs.
  - `test_merged_child_equals_each_member`: re-step every member outcome individually (`step` + `send_replacements` + `observe`): identical state bytes and observation.
  - `test_matches_playout_turn`: 200 keys on `roark_real_state`, one player action per key: `k_ai, k_turn = jax.random.split(key)`; `opp = ai_policy(tables, state, sample_ai_outcome(k_ai, tables, state))` (skip when `opp < 0`); `played, _, ai_switched, power_mul = playout._play(tables, state, a, key)`; `after, refused = send_replacements(tables, played, ai_switched, power_mul)`; `obs = observe(tables, state, [a, opp], sample_outcome(k_turn, tables, state, [a, opp]), after)`; exactly one child of request `(state, a, opp)` has those state bytes, that `obs` and `refused`.
  - `test_device_count_invariant`: in subprocesses (pattern: `tests/search/test_stage_a.py`), `expand` output (store rows and probabilities) is identical with 1 and 3 host devices (3 divides neither chunk size).
  - `test_expand_streams`: monkeypatch `step_rows` to assert each call gets at most `CHUNK` rows, on a request set spanning several chunks.
  - `test_sampling_only_reported`: `spike_cannon_root`: that request maps to `SamplingOnly`; other requests in the same call still expand.
  - `test_replacement_refusal_flagged`: `switch_refusal_root`, assert switch mass > 0 in `ai_rows`, request (Tackle, switch to Geodude): some child has `refused=True`.
- [ ] **Step 2: Run** the tests; expected: fail.
- [ ] **Step 3: Implement**: pad each chunk to the device-divisible size, shard like `rank_actions`, run through `cached_export` (register `Outcome` and `OutcomeLaws` for export here); host merging uses dicts keyed by `state.tobytes() + obs.tobytes()` per request (not `np.unique(axis=0)`, notes/integration.md §13).
- [ ] **Step 4: Run** tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): batched exact turn expansion with merging`.

### Task 4: Leaf and forced-turn evaluation (`search/leaves.py`)

Read first: notes/integration.md §1 and §5, notes/critic.md B.3, C9; rulings 5, 6, 8.

**Files:**
- Create: `src/nuzline/search/leaves.py`
- Test: `tests/search/test_leaves.py`

**Interfaces:**
- Consumes: `playout.batched_playout`, `playout`, `PlayoutResult`, `REFUSED_WON`, `step`, `sample_outcome`, `send_replacements`, `pursuit_fires`, `cached_export`.
- Produces:
  - `LEAF_CHUNK = 4096` playouts per program call (before device padding).
  - `@dataclass LeafValue: wins_a: int, n_a: int, wins_b: int, n_b: int, refused: bool` with `p_a`, `p_b` (`k/n`) and `var_b` (ruling 6) properties.
  - `leaf_keys(key, i: int, n: int) -> tuple[keys_a, keys_b]`: `keys_h = jax.random.split(jax.random.fold_in(jax.random.fold_in(key, i), h), n // 2)` for `h` in (0, 1).
  - `allocate(weights: np.ndarray, budget: int, floor: int = 2) -> np.ndarray` (ruling 8; even, `>= 2*floor`).
  - `evaluate_leaves(tables, states, counts, key, max_turns=100) -> list[LeafValue]`: `batched_playout` from each (non-terminal) state; leaf `i` uses `leaf_keys(key, i, counts[i])`; any refused playout marks the leaf refused (never counted as a loss); a state with `over != 0` raises `ValueError`.
  - `evaluate_forced(tables, states, pairs int32[N, 2], counts, key, max_turns=100) -> list[LeafValue]`: for a fixed (player, opponent) action pair: one turn with `sample_outcome`, the `_play` recipe's `ai_switched`/Pursuit `power_mul`, `send_replacements` (refusal marks refused), then `playout` with `max_turns - 1` (a win if already over). Used for sampled pairs below the root and pruned opponent actions.
- [ ] **Step 1: Write the failing tests**:
  - `test_halves_disjoint_and_reproducible`: identical inputs give identical `LeafValue`s; no row of `keys_a` appears in `keys_b`.
  - `test_matches_batched_playout`: totals equal a direct `batched_playout` over the same keys.
  - `test_forced_matches_playout_from_action`: on a state whose AI row is one-hot, `evaluate_forced` totals equal `playout_from_action` over the same keys; for a constructed pair, the first-turn child distribution matches Task 3's `expand` children within the Task 1 tolerance rule.
  - `test_allocate`: sums near the budget, respects the floor, proportional to weights, even.
  - `test_device_count_invariant`: 1 vs 3 host devices give identical `LeafValue`s (subprocess).
  - `test_terminal_state_rejected`: `evaluate_leaves` on an `over != 0` state raises `ValueError`.
- [ ] **Step 2: Run** the tests; expected: fail.
- [ ] **Step 3: Implement** (both evaluators padded and cached like Task 3).
- [ ] **Step 4: Run** tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): chunked leaf and forced-turn playouts with split halves`.

### Task 5: The solver (`search/stage_b.py`)

Read first: the spec §3-4, rulings 4-10, notes/critic.md C6-C9.

**Files:**
- Create: `src/nuzline/search/stage_b.py`, `tests/search/exhaustive.py`
- Modify: `tests/search/fixtures.py` (add `all_terminal_root`, `pixel_endgame`, `spore_endgame`)
- Test: `tests/search/test_stage_b.py`

**Interfaces:**
- Consumes: Tasks 1-4, `stage_a.rank_actions`, `Ranking`, `stage_a._label`.
- Produces:
  - `@dataclass(frozen=True) SolveSettings: max_depth: int = 2, budget_playouts: int | None = None, budget_seconds: float | None = None, epsilon: float = 1e-4, leaf_floor: int = 2, max_turns: int = 100, n_depth0: int = 2000`.
  - `@dataclass(frozen=True) ActionValue: action: int, label: str, estimate: float, ci_low: float, ci_high: float, lower_bound: float, unexplored: float, sampled: float, refused: float, tied_with_best: bool`.
  - `@dataclass PlanNode: obs: bytes | None, prob: float, action: int | None, values: dict[int, float], children: dict[int, list[PlanNode]], kind: str` (`"decision" | "leaf" | "terminal" | "pruned" | "refused" | "sampled"`). `prob` is absolute reach (ruling 10). At the root `children` has one entry per ranked root action; at inner decision nodes only the half-A-chosen `action`; other kinds have none.
  - `@dataclass SolveResult: ranking: list[ActionValue], dropped: dict[int, str], depth: int, plan: PlanNode, playouts: int, seconds: float, settings: SolveSettings`.
  - `solve(tables, names, state, key, settings: SolveSettings, leaf_fn=None) -> SolveResult`. `leaf_fn(states, weights) -> list[LeafValue]` replaces both evaluators (tests pass a zero evaluator).
  - `ActionValue.unexplored`, `sampled`, `refused` for root action `a` = the sum of `prob` over `pruned`, `sampled`, `refused` nodes in `plan.children[a]`, following only the chosen action at inner decision nodes.
- Algorithm (what the interfaces leave open): passes `d = 1..max_depth` per ruling 9, each rebuilding from the root with an expansion cache keyed by (state bytes, pair). At a belief node, the value of player action `a` is the belief-weighted sum over its states of `sum_o q(o) sum_children p * value(group)`, children grouped by observation across the node's states. Opponent actions and chance children with absolute reach `< epsilon` are `pruned`: opponent actions valued by `evaluate_forced`, outcome children by `evaluate_leaves`, each with `2f` playouts, bound 0. Sampling-only pairs below the root are `sampled` (`evaluate_forced`, budget share by ruling 8, bound 0). Backup per rulings 4-6.

- [ ] **Step 1: Write the failing tests** (every test except the two endgame tests uses `SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200)` or smaller):
  - `test_depth0_is_stage_a`: `max_depth=0, n_depth0=200` equals `rank_actions(n=200, key)` (`p_win0`, Wilson bounds).
  - `test_matches_exhaustive_on_endgames`: `tests/search/exhaustive.py` is independent of `expand.py` (no imports from it, no shared merge code): per (state, pair) it pads Task 1's rows to a fixed P (8192; assert K <= P), runs one module-level `jax.jit(jax.vmap(turn_fn))` (the `_play` recipe, `send_replacements`, `observe`), merges on the host by bytes, groups by observation, and recurses to a turn cap; states not over at the cap count 0. Interface: `exhaustive(tables, state, cap, group: bool = True) -> tuple[dict[int, float], Callable[[tuple[bytes, ...]], int]]` (root values and the chosen action at a belief node given its observation path; `group=False` gives each state its own optimum). Endgames (1-2 moves per side, cap 2): `pixel_endgame`, `spore_endgame`, and one plain low-HP 1v1. With `max_depth = cap`, `epsilon = 0`, zero `leaf_fn`: every root estimate equals the exhaustive value within 1e-6; `unexplored == sampled == refused == 0`; and at least one root action has a value strictly between 0 and 1 (so the comparison is not trivial). Budget: under ~60 s; much longer means something is wrong.
  - `test_no_peeking`: on `pixel_endgame`, the turn-2 belief node holds at least 2 states with one observation; `exhaustive(group=False)` exceeds `exhaustive(group=True)` by at least 1e-3 at the root; Stage B's turn-2 action is the grouped exhaustive choice.
  - `test_interval_aggregates_transpositions`: a `leaf_fn` returning fixed known `LeafValue`s on a fixture where two observation children of one root action share a state: the interval equals ruling 6's formula with weights summed before squaring.
  - `test_pruning_bound`: `epsilon=0.05` on `roark_real_state`: each action's `unexplored` equals the summed `prob` of its pruned nodes (1e-9), and `lower_bound <= estimate`.
  - Review Focus 1, `test_dropped_root_action`: `spike_cannon_root`: Spike Cannon is in `dropped` with the reason; the other actions are ranked.
  - Review Focus 2, `test_time_budget_stops_deepening`: `max_depth=2, budget_seconds=1e-9`: `depth == 1`, non-empty ranking.
  - Review Focus 3, `test_refused_replacement_mass`: `switch_refusal_root` with `epsilon=0`: the Tackle action's `refused` equals the AI's switch-to-Geodude mass times the probability the newcomer faints (1e-6), it adds nothing to `estimate`, and no exception is raised.
  - Review Focus 5, `test_all_terminal_root`: `playouts == 0`; every root action `estimate == 1.0`, `ci_low == ci_high == estimate`, `unexplored == sampled == refused == 0`.
  - `test_reproducible`: two solves with the same key give equal `SolveResult`s.
- [ ] **Step 2: Run** the tests; expected: fail.
- [ ] **Step 3: Implement** `stage_b.py`, `exhaustive.py` and the new fixtures.
- [ ] **Step 4: Run** `~/bin/uv.exe run pytest tests/search -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): Stage B expectimax over information sets`.

### Task 6: Plan tree output and `nuzline solve`

Read first: notes/integration.md §7, §11, §12; the spec §5.

**Files:**
- Create: `src/nuzline/search/plan_tree.py`
- Modify: `src/nuzline/cli.py` (a `solve` subcommand next to `rank`)
- Test: `tests/search/test_plan_tree.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 5 (`solve`, `SolveSettings`, `SolveResult`, `PlanNode`), Task 2 (`OBS_FIELDS`), `Names`.
- Produces:
  - `label(obs: bytes, names) -> str` (ASCII, e.g. `"Geodude used Rock Throw, missed"`, `"Geodude used Rock Throw, crit, you 12/58 HP"`, `"Roark sent out Onix"`; the opponent's HP as a bar percentage from pixels/48).
  - `to_json(result, names) -> dict`: per node the decoded observation fields and `label`, `prob`, `kind`, `action` and its label, `values` by action label, `children` by action label; top level: settings, depth, ranking, dropped, playouts, seconds, `table_version`, `code_version`.
  - `render(result, names, show_depth: int) -> list[str]` (the spec §5 layout), ending with a footer stating the leaf budget and that leaf estimates come from playouts that see the full state.
  - CLI: `nuzline solve --party P (--trainer NAME | --trainer-id N) [--lead N] [--opp-active N] [--max-depth D=2] [--budget SECONDS] [--playouts N] [--epsilon E=1e-4] [--seed S=0] [--devices N] [--tree FILE.json] [--show-depth K=1]`; `--playouts` is `budget_playouts` (ruling 9); refusals print `refused: ...` and exit 2; a run record `runs/solve_<YYYYMMDD_HHMMSS>.json` holds the party file's sha256, trainer id, settings, seed and `to_json` output (`runs/` is already gitignored).
- [ ] **Step 1: Write the failing tests**:
  - `test_labels_ascii_and_distinct`: every child label under the Roark root is ASCII and sibling labels are distinct.
  - `test_json_roundtrip`: `json.loads(json.dumps(to_json(...)))` keeps every node; each decision's children sum to the parent's `prob` (1e-9).
  - Review Focus 4, `test_single_action`: a root with one legal player action (one move, `player_count=1`) solves and renders one ranked row.
  - `test_cli_solve_small` (`tests/test_cli.py`): `main(["solve", "--party", SAMPLE, "--trainer", "roark", "--max-depth", "1", "--playouts", "2000", "--tree", tmp/"t.json", "--devices", "1"])` returns 0, prints the ranking header and at least one `if ...` line, writes a valid JSON tree and a run record under a monkeypatched `runs/`.
  - `test_cli_solve_refused`: the existing `rank` refusal fixture with `solve` prints `refused:` and returns 2.
  - `test_cli_imports_no_jax` stays green.
- [ ] **Step 2: Run** the tests; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(cli): nuzline solve with plan tree and run records`.

### Task 7: Timing benchmark and docs

**Files:**
- Create: `benchmarks/solve_timing.py` (the `rank_timing.py` pattern: subprocesses, fresh `NUZLINE_CACHE_DIR`, cold then warm)
- Modify: `benchmarks/README.md` (a Stage B section), `docs/design.md` §4.3 (a paragraph pointing to the Stage B design for information sets and rulings), `README.md` (a `nuzline solve` usage line)

- [ ] **Step 1:** Write `solve_timing.py`: runs `nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth D --playouts B` for D = 1 and 2, cold and warm; records wall time and its phases (expansion, leaf playouts), peak RSS, completed depth, the leader's interval half-width and total playouts in `benchmarks/results/solve_<timestamp>.json`; exits non-zero if the output lacks the ranking header.
- [ ] **Step 2:** Find `B_1` and `B_2` by measurement: start at `B_d = 2 * K_d * 4300` (K_1 ~ 9, K_2 ~ 81 player-action combinations: about 80 000 and 700 000; the 4300 effective playouts for a 1.5 pp half-width is notes/critic.md C7, the factor 2 is the ruling-5 half split) and double until the leader's half-width is at most 1.5 percentage points; time at that budget.
- [ ] **Step 3:** Record the numbers in `benchmarks/README.md` against the targets (depth 1 within 2 minutes, depth 2 within 10 minutes, cold, at that precision). If a target is missed, say so plainly with the phase breakdown and list the spec §7 levers in order; do not change the targets.
- [ ] **Step 4:** Update the docs named above.
- [ ] **Step 5: Commit** `bench: Stage B solve timing; docs`.

---

## Self-review notes

- Spec coverage: §1 -> Task 1 (+ cap, ruling 3); §2 -> Task 2; §3 -> Tasks 3-5 (pruning, leaves, deepening, confidence; leaf peeking disclosed in Task 6's footer); §4 -> the CLI reports the primary objective only; §5 -> Task 6; §6 -> tests in Tasks 1, 2, 5 (reference-engine agreement in Task 1); §7 -> Task 7; milestone order already in `docs/design.md`.
- Plan review (adversarial, 3 lenses, 27 verified findings) is folded in: fixtures that existing helpers cannot build now have recipes; budgets, depth 0, padding, streaming, interval aggregation, key streams and effect-gated observations are pinned by rulings and tests.
