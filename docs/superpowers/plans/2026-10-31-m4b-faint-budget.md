# Milestone 4b: Faint Budget and Set-Mode Replacements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Battles play on past player faints until a win or a whiteout, every search layer counts who died, and `nuzline rank`/`solve` optimise P(win with at most `k` deaths, no protected Pokemon lost) and report the death distribution.

**Architecture:** The engine stops treating a player faint as terminal (`step._faint_check`), records deaths per party slot and settles the decomp's simultaneous-faint order; `k` and the protected set are traced `BattleState` fields, never compile-time constants. The playout gains Set-mode player replacements (a heuristic) and a budget-aware death cost; Stage A scores every rung 0-3 from one batch of playouts; Stage B gains replacement decision nodes (expand, observe, plan tree) and "over budget" nodes, and backs up expected deaths and the distribution. The CLI adds `--deaths K` and `--protect NAME` to `rank` and `solve`.

**Tech Stack:** Python 3.12, JAX (CPU, sharded host devices), numpy (float64 on the host), pytest, hypothesis, ruff; `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format`.

**Spec:** `docs/superpowers/specs/2026-10-31-prep-search-design.md` §4 (objective, ladder semantics, what `k` does, tie-breaks), §5.1 and §10 item 1 (protected Pokemon); parent `docs/design.md` §3, §4.2, §10. Stage B background: `docs/superpowers/specs/2026-10-30-stage-b-solver-design.md` and `docs/superpowers/plans/2026-10-30-stage-b-solver.md` (rulings 1-10 still hold).

**Milestone 4 plans:** 4a (coverage for the user's Roark pool and TM data, spec §5.2-5.3) and 4b (this plan) are independent and can be built in parallel. 4c (prep search pipeline and CLI, spec §6-7) depends on both. 4a and 4b both add `BattleState` fields (4a after `move_hit`, 4b at the end) and edit `LAYOUT`, `engine/STATE.md`, `reference/state.py` and `tests/strategies.py`: whichever merges second rebases its fields onto the other's (field order: 4a's where 4a puts them, 4b's last), regenerates its `tests/strategies.py` draws and reruns `tests/test_differential.py`. Line numbers cited here are those at `b36b1d0`; after 4a lands, find the code by name.

## Global Constraints

- Objective (spec §4): "maximise P(win with deaths ≤ k), default `k = 0`"; "every rung maximises P(win and deaths ≤ k), and a loss is never traded for fewer deaths".
- "A battle always continues until the player wins or whites out; deaths are counted, not terminal. `k` only scores the result" (spec §4).
- Protected (spec §10.1): "any line in which it faints counts as a failure on every rung, whatever `k` is".
- "Every result shows the distribution P(0), P(1), P(2), P(≥3), P(whiteout)" (spec §4).
- Ties "break on fewer expected deaths" (spec §4); resource cost and HP/PP tie-breakers belong to plan 4c.
- Set mode: the player gets no switch when the opponent's Pokemon faints (spec §5.1). Any faint is a death (spec §3).
- `k` and the protected mask are traced data: changing them never recompiles a program (spec §5.1).
- "`k = 0` ... the regression requirement is that P(win, 0 deaths) is statistically unchanged, not byte-identical" (spec §4); Task 1 records today's numbers before anything changes.
- The decomp (`external/pokeplatinum`, pinned commit in `docs/decomp_paths.md`) is authoritative for every mechanic; docstrings cite file:line as the existing code does.
- No new runtime dependencies. CLI output ASCII only; `nuzline.cli` imports no jax at module import (`tests/test_cli.py`); exit codes 0 ok / 1 error / 2 refused.
- Register new pytrees for `jax.export` once, in the module that exports them; every batched program keeps one fixed padded chunk shape through `jaxcache.cached_export` (Stage B plan constraints).
- Each new test file stays under ~60 s on one device; no default-suite test runs a depth-2 solve except the small endgames with a zero leaf evaluator.
- The user's real save is used only through `NUZLINE_USER_SAVE`; tests skip without it. Never touch any folder named `backup`.
- ruff `line-length = 100`; match the surrounding docstring density and citation style.

## Rulings (controller's calls; the spec leaves them open)

1. **Where `k` lives.** The engine never reads `k`: a battle ends at a whiteout or when the opponent has no Pokemon left. `k` (`faint_budget`), the protected slots (`protected`) and the per-slot death record (`deaths`) are `BattleState` fields, so they are traced through every program and are part of the merge bytes (constant within a solve). `MAX_DEATH_BUDGET = 3`: `k` is 0..3, the ladder's range.
2. **Simultaneous faints (decomp).** Both sides out at one faint check is `BATTLE_RESULT_WIN | BATTLE_RESULT_LOSE` = `BATTLE_RESULT_DRAW` (`include/constants/battle.h:78`; `BattleControllerPlayer_CheckBattleOver`, battle_controller_player.c:4181-4262), which `HandleResult` plays as a loss (:4015-4016). Within a turn the faint checks run where they do today, so the side whose check runs first (end-of-turn order) loses first. When both actives are down, `ReplaceFainted` (:4070-4155) flags both, `BtlCmd_ShowParty` (battle_script.c:2378-2414) opens both menus before anyone is sent out, so the AI picks (`Task_TrainerShowPartyMenu`, battle_display.c:4543-4556) against the player's *fainted* battler; `GoToIfAnySwitches` (battle_script.c:2537) sends the lower battler (the player) in first, each entry followed by hazards and a faint check (`subscript_show_party_list.s`), and `TurnEnd` (:1844-1848) re-enters `ReplaceFainted` until nobody needs one. `powerMul` persists until `BattleContext_Init` (:1855), so the turn's value is kept in `replace_mul` while a player replacement is pending.
3. **Over budget ("failed").** A state with `player_faints > faint_budget` or a protected slot in `deaths` is failed: its objective value is exactly 0 (variance 0). Stage B does not expand it; it is evaluated by `2 * leaf_floor` playouts (own key stream, no top-ups) only for the distribution and expected deaths, and the plan says "over the death budget: play on for the win, not planned". A failed root is a `ValueError`.
4. **Depth counts turns.** A player replacement is a Stage B decision node expanded in the same level as the turn that caused it (again after an entry faint); it is never a leaf. A root may be a replacement decision (`decision_actions`); `exhaustive`'s cap counts turns the same way.
5. **Tie rule.** Root rankings (Stage A and B) use `stage_a.tie_ordered`: among rows whose interval overlaps the top-scoring row's, the leader has the fewest expected deaths (then the higher score, then the lower action); the other rows follow by `(-score, expected deaths, action)`; `tied_with_best` marks overlap with the leader. Inner Stage B nodes pick the highest half-A value; values equal to 12 decimals break on fewer half-A expected deaths, then the lower action.
6. **Death cost in the playout policy.** `death_cost(state, slot)`: 1 if `slot` is the player's only living Pokemon; else 1/2 if the state is failed; else 1 if `slot` is protected or no budget is left (`faint_budget - player_faints <= 0`); else `1 / (r + 1)` with `r` the deaths left. A pair is valued `(1 - cost * risk) * (1 + removed)` with today's `risk = ko_opp * (1 - first * ko_player)`, so `k = 0` reproduces today's values.
7. **Replacement heuristic.** `replacement_policy` scores each legal slot `c` as the best over `c`'s usable moves of the mean over the opponent active's legal moves of `_pair_value` with `c` active (stages reset) against the opponent's active, or against the slot the AI will send in when the opponent's active is down; ties to the lowest slot. No key, no AI distribution.
8. **Distribution buckets** (`deaths.BUCKETS`): wins with 0, 1, 2, 3, 4+ deaths and no protected loss; wins with a protected loss; whiteouts (any finished non-win); unfinished (turn cap). Rung `k` = buckets 0..k. Displays fold 3 and 4+ into "3+".
9. **No early stop at the first faint.** Every playout runs to the end because every result shows the distribution; a `k = 0` speed mode is left to plan 4c if its timing benchmark needs it.
10. **`--protect`** takes a species name (case-insensitive) or a 1-based party slot number (the position in the party as `nuzline` lists it, eggs already dropped: `--protect 2` is party index 1), repeatable; an unknown name, an ambiguous name (two party members of the species), a number outside `1..len(party)` or a fainted Pokemon is an error (exit 1).
11. **Plan labels.** Spec §7's "Monferno faints here by design" becomes "Monferno fainted (death 1 of 1 allowed)" on the observation and "send in Staravia" as the decision: the plan cannot tell a planned sacrifice from bad luck, so it states the budget instead.

## Review Focus

1. **A protected Pokemon dies in a battle the player still wins**: a failure on every rung, counted as `win_protected_lost`. Tests: Task 6 `test_protected_death_fails_every_rung`, Task 9 `test_protected_line_matches_exhaustive`.
2. **Both last Pokemon faint at one faint check** (Struggle recoil KO): a loss, never a win. Test: Task 7 `test_double_ko_of_last_pokemon_is_a_loss`.
3. **A replacement faints on entry to Stealth Rock**: a second death, a second replacement decision, in the engine, the playout and the plan. Tests: Task 7 `test_entry_faint_needs_a_second_replacement`, Task 9 `test_second_replacement_decision_in_plan`.
4. **Bad budget flags** (`--deaths 4`, `--deaths -1`, `--protect Pikachu`, an ambiguous species, slot 9, a fainted Pokemon): one stderr line naming the problem, exit 1. Test: Task 10 `test_cli_bad_budget_flags`.
5. **A one-Pokemon party with `--deaths 2`**: the ladder is flat (no replacement exists), nothing breaks. Test: Task 7 `test_single_pokemon_ladder_is_flat`.

---

### Task 1: Record the `k = 0` baseline

Do this first, on the unchanged code.

**Files:**
- Create: `tests/search/k0_baseline.py`, `tests/fixtures/k0_baseline.json` (generated), `tests/search/test_k0_regression.py`

**Interfaces:**
- Consumes: `stage_a.rank_actions`, `stage_b.solve`, `SolveSettings`, `tests.engine.helpers.roark_real_state`.
- Produces:
  - `BASELINE = Path("tests/fixtures/k0_baseline.json")` (absolute, from the repo root); `STAGE_A_N = 2000`; `STAGE_B_SETTINGS = SolveSettings(max_depth=1, budget_playouts=4000, n_depth0=200)`; seed `jax.random.PRNGKey(0)` for both.
  - `baseline(tables, names) -> dict`: `{"commit": str, "stage_a": {"n": 2000, "actions": {str(action): {"wins0": int, "p_win": float}}}, "stage_b": {"actions": {str(action): {"estimate": float, "ci_low": float, "ci_high": float}}}}` (`wins0 = round(p_win0 * n)`). Running the module (`uv run python -m tests.search.k0_baseline`) writes it.

- [ ] **Step 1: Write `tests/search/test_k0_regression.py`**:
  - `test_stage_a_k0_unchanged`: `rank_actions(tables, names, roark_real_state(...), 2000, PRNGKey(0))`; the same action set as the JSON; per action, with `p` the pooled `(wins_old + wins_new) / 4000`: `abs(wins_new - wins_old) / 2000 <= 4 * sqrt(2 * p * (1 - p) / 2000) + 1 / 2000`.
  - `test_stage_b_k0_unchanged`: `solve(..., STAGE_B_SETTINGS)`; the same ranked actions; per action `abs(est_new - est_old) <= half_old + half_new + 1e-9` (`half = (ci_high - ci_low) / 2`).
  Both skip with a message when the JSON is missing.
- [ ] **Step 2: Implement `tests/search/k0_baseline.py`**, run `uv run python -m tests.search.k0_baseline`; check the JSON has every legal root action of `roark_real_state`.
- [ ] **Step 3: Run** `uv run pytest tests/search/test_k0_regression.py -q`; expected: 2 passed (same code, same keys); note the runtime (target under ~90 s).
- [ ] **Step 4: Commit** `test(search): record the k=0 Stage A/B baseline before the faint budget`.

### Task 2: Death record, budget fields and host death bookkeeping

**Files:**
- Modify: `src/nuzline/engine/state.py` (`BattleState` :60-136, `LAYOUT` :156-221; append fields after `ai_ability_known`), `src/nuzline/engine/step.py:607-640` (`_faint_check`: record deaths only, terminal rule unchanged), `src/nuzline/engine/actions.py`, `src/nuzline/reference/state.py` (`RefState`), `src/nuzline/reference/engine.py:418-447`, `src/nuzline/engine/STATE.md` (rows `BattleContext.powerMul` :347 -> `mapped`, `replace_mul`; a mapping note for the search fields), `tests/strategies.py:245`
- Create: `src/nuzline/search/deaths.py`, `tests/engine/test_faints.py`, `tests/search/test_deaths.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `BattleState` fields (in this order, at the end): `deaths: int8[6]` (1 where the player's party slot fainted during this battle), `faint_budget: int8[]` (`k`), `protected: int8[6]`, `replace_mul: int8[]` (the turn's `powerMul` while a player replacement is pending and the opponent's active is down, else 0; ruling 2). `make_state` leaves all zero. Invariant: `player_faints == deaths.sum()`.
  - `MAX_DEATH_BUDGET = 3` (state.py); `BattleState.with_budget(self, k: int = 0, protected: Sequence[int] = ()) -> BattleState`: `ValueError` for `k` outside `0..MAX_DEATH_BUDGET` or a protected slot outside 0..5, empty (`species == 0`) or fainted (`hp == 0`).
  - `actions.decision_actions(tables, state) -> jnp.ndarray` (`bool[NUM_ACTIONS]`): at a player replacement (the player's active has 0 HP, a living bench Pokemon exists, `over == 0`; `step.needs_replacement(state)[0]` without importing step) only the switch actions to `legal_replacements(state, 0)` (trapping ignored); otherwise `legal_actions(tables, state, 0)`.
  - `search/deaths.py` (host numpy unless stated):
    - `BUCKETS = ("win_0", "win_1", "win_2", "win_3", "win_4plus", "win_protected_lost", "whiteout", "unfinished")` (ruling 8).
    - `over_budget(state) -> jnp.ndarray` (bool; one or a batch of states, numpy or jax): `(player_faints > faint_budget) | any(protected & deaths, axis=-1)`.
    - `success(won, faints, deaths, k, protected) -> np.ndarray` (bool, broadcasting over a leading batch axis; `deaths`, `protected` `[..., 6]`): `won & (faints <= k) & ~any(deaths & protected)`.
    - `bucket(won, ended, faints, deaths, protected) -> np.ndarray` (int index into `BUCKETS`).
    - `@dataclass(frozen=True) DeathDistribution: probs: tuple[float, ...]` (8, `BUCKETS` order), `n: int | None`; `rung(k: int) -> float` (sum of `probs[: k + 1]`, `ValueError` for `k > MAX_DEATH_BUDGET`); property `p_win` (first 6); `as_dict() -> dict` (`BUCKETS` names plus `"n"`); `classmethod from_counts(counts) -> DeathDistribution` (`ValueError` on zero total).

- [ ] **Step 1: Write the failing tests**:
  - `tests/engine/test_faints.py::test_new_fields_default_to_zero` (renamed test_new_fields_start_values, D2-38): `make_state` gives all-zero new fields; `LAYOUT` has `deaths ((6,), int8)`, `faint_budget ((), int8)`, `protected ((6,), int8)`, `replace_mul ((), int8)`.
  - `test_with_budget`: `roark_state(...).with_budget(k=2, protected=[1, 3])` gives `faint_budget == 2`, `protected == [0, 1, 0, 1, 0, 0]`; `ValueError` for `k=4`, `k=-1`, slot 6, an empty slot (`player_count=2`, slot 3) and a fainted slot (`replace_hp(0, 2, 0)`).
  - `test_faint_records_the_slot`: `roark_state(player_count=1).replace_hp(0, 0, 1)` stepped with the opponent's Tackle (`empty_outcome()`, hits): `deaths == [1, 0, 0, 0, 0, 0]`, `player_faints == 1`; an opponent faint (`replace_hp(1, 0, 1)`) leaves `deaths` zero; `ref_step` (via `to_ref`/`from_ref`) gives the same state.
  - `test_decision_actions`: equals `legal_actions(tables, s, 0)` on `roark_real_state`; on that state with `hp[0, 0] = 0` (over 0) and `trapped[0] = 3`: exactly the switches to the living slots 1-5.
  - `tests/search/test_deaths.py::test_success_and_bucket`: a table of rows (won, ended, faints, deaths, k, protected) -> (success, bucket name), including: a win with 2 deaths at `k=1` (False, `win_2`), a win with a protected death at `k=3` (False, `win_protected_lost`), a finished loss (False, `whiteout`), an unfinished playout (False, `unfinished`), a win with 5 deaths (`win_4plus`).
  - `test_over_budget`: a numpy batch of 3 states (within budget, faints over, a protected death) -> `[False, True, True]`.
  - `test_distribution`: `from_counts([5, 3, 1, 1, 0, 2, 7, 1])`: `rung(0) == 0.25`, `rung(1) == 0.4`, `rung(3) == 0.5`, `p_win == 0.6`, `rung(4)` raises, `as_dict()` keys are `BUCKETS + ("n",)`; `from_counts([0] * 8)` raises.
  - `tests/test_differential.py::test_jax_matches_reference` stays green once `tests/strategies.py` fills the new fields (`deaths` = the fainted player slots, consistent with `player_faints`; `faint_budget` in 0..3; `protected` a random subset of living player slots; `replace_mul` 0).
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_faints.py tests/search/test_deaths.py -q`; expected: fail (fields and module missing).
- [ ] **Step 3: Implement** the fields, `with_budget`, the death record in both engines, `decision_actions`, `deaths.py`, the strategy and STATE.md updates.
- [ ] **Step 4: Run** `uv run pytest -q` and `uv run ruff check src tests`; expected: pass (the whole suite: merge bytes and the reference conversion see the new fields).
- [ ] **Step 5: Commit** `feat(engine): per-slot death record and traced death budget fields`.

### Task 3: Set-mode player replacements in the playout

Pending states are built by hand here (the terminal rule still ends the battle at a player faint until Task 7).

**Files:**
- Modify: `src/nuzline/search/playout.py` (`PlayoutResult` :78-82, `send_replacements` :155-174, `_turn` :189-194, `_run` :205-230, `replacement_refusal` :255-272, `refusal_reason` :280-306; module docstring), `src/nuzline/engine/step.py:885-892` (`needs_replacement` docstring), `src/nuzline/search/leaves.py:127-135` (`_forced_one` fills the new result fields), `tests/search/fixtures.py`
- Create: `tests/search/test_replacements.py`

**Interfaces:**
- Consumes: Task 2 fields; `step.replace`, `step.NORMAL_POWER_MUL`, `ai.replacement.post_ko_switch_in`, `actions.legal_replacements`.
- Produces:
  - `PlayoutResult` gains `deaths: int8[6]` (the final state's) and `ended: int8[]` (1 when the battle is over), after `refused`.
  - `send_replacements(tables, state, ai_switched, power_mul) -> tuple[BattleState, jnp.ndarray]` (jitted, signature unchanged): as today while only the opponent needs a replacement; at a pass where the player's active is down it stops before anyone enters, setting `replace_mul = power_mul` when the opponent's active is down too (else 0). `refused` as today, also when it stops (ai_switched or a refused `post_ko_switch_in` for a needed opponent replacement).
  - `replacement_pass(tables, state, slot) -> tuple[BattleState, jnp.ndarray]` (jitted): at a player replacement: the opponent's slot is `post_ko_switch_in(tables, state, replace_mul or NORMAL_POWER_MUL)` on this state when its active is down; `replace(tables, ., 0, slot)`, then `replace(tables, ., 1, opp_slot)`; `replace_mul = 0`; then `send_replacements(tables, ., False, that power_mul)`. Returns `(state, refused)`; `slot` must be a `legal_replacements(state, 0)` slot.
  - `replacement_policy(tables, state) -> jnp.ndarray` (jitted, int32 party slot; ruling 7).
  - `finish_turn(tables, state, ai_switched, power_mul) -> tuple[BattleState, jnp.ndarray]` (jitted): `send_replacements`, then while the player needs a replacement (a `fori_loop` of `SLOTS - 1`) `replacement_pass` with `replacement_policy`. `_turn` uses it.
  - `playout(tables, state, key, max_turns)` from a pending state first resolves it with `replacement_policy` and `replacement_pass` (no key used, no turn counted); `playout_from_action` at a pending state takes `action = MOVE_ACTIONS + slot` as that replacement and the policy chooses from the first turn on.
  - `replacement_refusal(tables, state, ai_switched, power_mul) -> str | None` mirrors `finish_turn`.
  - `tests/search/fixtures.py`: `at_replacement(state, opp_down: bool = False) -> BattleState`: the player's active (and with `opp_down` the opponent's) at 0 HP, `deaths`/`player_faints` updated, `over = 0`.

- [ ] **Step 1: Write the failing tests** in `tests/search/test_replacements.py`:
  - `test_send_replacements_stops_at_a_player_replacement`: `at_replacement(roark_state(...), opp_down=True)`, `send_replacements(..., False, 10)`: both actives unchanged, `replace_mul == 10`, not refused; with `power_mul=20`, `replace_mul == 20`.
  - `test_ai_chooses_against_the_fainted_battler`: a pending state where `post_ko_switch_in(tables, s, 10)` differs from `post_ko_switch_in` on `s` with the player's chosen newcomer active (the test asserts they differ); after `replacement_pass(tables, s, slot)` the opponent's active is the former.
  - `test_player_enters_first`: with Stealth Rock on both sides, `replacement_pass(tables, s, slot)[0]` equals `replace(tables, replace(tables, s, 0, slot), 1, o)` field by field (`o` from the pending state), with `replace_mul` 0.
  - `test_replace_mul_reaches_the_ai`: the same with `replace_mul = 20` set: `o` is `post_ko_switch_in(tables, s, 20)`.
  - `test_refused_opponent_replacement_at_a_player_replacement`: `send_replacements(..., ai_switched=True, 10)` on `at_replacement(..., opp_down=True)` is refused.
  - `test_replacement_policy_picks_the_counter`: `roark_state(player_count=4, opp_count=1)` with Grotle's slot emptied, `at_replacement`: Prinplup (slot 1) is picked over Starly (slot 3).
  - `test_replacement_policy_is_legal`: 30 hypothesis states (`tests/strategies.supported_state`) with the player's active set to 0 HP and a living bench: the slot is in `legal_replacements(s, 0)`.
  - `test_playout_from_a_pending_state`: `playout(tables, s, key)` equals `playout(tables, replacement_pass(tables, s, replacement_policy(tables, s))[0], key)` field by field; `playout_from_action(tables, s, MOVE_ACTIONS + 3, key)` equals `playout(tables, replacement_pass(tables, s, 3)[0], key)`.
  - `test_result_records_deaths_and_end`: a finished playout's `deaths` equals its final state's (replay with `_turn_jit`), `ended == 1`; `max_turns=1` on a long fight gives `ended == 0`.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_replacements.py -q`; expected: fail.
- [ ] **Step 3: Implement** the functions above; update `refusal_reason` to replay the same flow.
- [ ] **Step 4: Run** `uv run pytest tests/search -q` and ruff; expected: pass (existing playout, leaves, expand and Stage B tests unchanged).
- [ ] **Step 5: Commit** `feat(search): Set-mode player replacements in playouts`.

### Task 4: A budget-aware playout policy

**Files:**
- Modify: `src/nuzline/search/playout.py` (`_pair_value` :115-125, module docstring)
- Test: `tests/search/test_playout.py`

**Interfaces:**
- Consumes: Task 2 (`deaths.over_budget`, fields), Task 3 (`replacement_policy` reads `_pair_value`).
- Produces: `death_cost(state, slot) -> jnp.ndarray` (float32; ruling 6). `_pair_value(tables, state, a, b)` uses the cost of the slot that takes the opponent's move (the incoming slot for a player switch, else the active). `playout_policy` and `replacement_policy` follow.

- [ ] **Step 1: Write the failing tests**:
  - `test_death_cost` (parametrized on `roark_state` variants): k=0, no deaths -> 1.0; k=1, no deaths -> 0.5; k=2 -> 1/3; k=1 with one death -> 1.0; k=1, protected slot -> 1.0; `player_count=1`, k=3 -> 1.0 (last Pokemon); failed (k=0, one death, bench alive) -> 0.5; failed and last -> 1.0.
  - `test_k0_pair_values_unchanged`: on `roark_real_state` and 20 hypothesis states with `faint_budget = 0` and nothing failed, every `(a, b)` value equals today's formula transcribed in the test (`(1 - ko_opp * (1 - first * ko_player)) * (1 + removed)`, from `_move_odds` and the tie average) within 1e-6.
  - `test_budget_makes_the_policy_accept_ko_risk`: `roark_state(opp_count=1, opp_faster=True)`, Monferno at an HP Rock Throw KOs with probability >= 0.8 (assert from `damage_table`), Geodude at an HP Mach Punch cannot KO (assert): `playout_policy` picks a switch at `k=0` and Mach Punch at `with_budget(k=2)` (seed 0).
  - `test_replacement_policy_spares_a_protected_pokemon`: the Task 3 counter state with Prinplup at 1 HP, Geodude faster (`opp_faster=True`) and Geodude's Stealth Rock and Leer PP zeroed (only Tackle and Rock Throw, so every Geodude move threatens the KO): `with_budget(k=1, protected=[1])` picks Starly. (No unprotected pick is asserted: with Geodude faster, Prinplup's value is low at every `k`; `test_death_cost` pins the cost itself.)
  - `test_budget_change_does_not_recompile`: `playout._cache_size()` after playouts at `k=0` and `k=2` (same shapes) grows by at most one entry in total.
- [ ] **Step 2: Run** them; expected: fail.
- [ ] **Step 3: Implement** `death_cost` and the cost in `_pair_value`, written as `(1.0 - cost * risk) * (1.0 + removed)`.
- [ ] **Step 4: Run** `uv run pytest tests/search -q`, the Task 1 regression and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): playout policy values deaths against the budget`.

### Task 5: Leaves count successes, deaths and buckets

**Files:**
- Modify: `src/nuzline/search/leaves.py` (`LeafValue` :58-89, `_forced_turn` :114-124, `_run` :142-172, `_check` :175-185), `src/nuzline/search/stage_b.py` (`_Store.add` :240-243 terminal success, `_top_up` :534-554 uses `LeafValue.merge`), `tests/search/exhaustive.py:122-123` (terminal value)
- Test: `tests/search/test_leaves.py`, `tests/search/test_stage_b.py`

**Interfaces:**
- Consumes: Task 2 (`deaths.success`, `deaths.bucket`), Task 3 (`PlayoutResult.deaths/ended`, `finish_turn`).
- Produces:
  - `LeafValue` gains `deaths_a: int = 0`, `deaths_b: int = 0` (summed `player_faints` per half), `buckets_b: tuple[int, ...] = ()` (8 counts over half B); properties `e_deaths_a`, `e_deaths_b`; `merge(other) -> LeafValue` (adds every count, ors `refused`). `wins_a`/`wins_b` now count successes: `deaths.success` with the evaluated state's own `faint_budget` and `protected`.
  - `evaluate_forced`'s first turn ends with `finish_turn`; `evaluate_leaves` accepts a pending state.
  - Terminal values count successes, before the engine flip (Task 7) can produce a win after a death: `stage_b._Store.won[i]` is `over & deaths.success(winner == WINNER_PLAYER, player_faints, deaths, faint_budget, protected)` of the stored state (its own `k` and protected set); `tests/search/exhaustive.py` values a terminal state the same way (`total += reach * success`), so the oracle and Stage B change together.

- [ ] **Step 1: Write the failing tests**:
  - `test_wins_count_successes`: `roark_state` with Monferno (slot 0) dead earlier (`hp 0`, `deaths[0] = 1`, `player_faints = 1`) and Prinplup active: at `k=0` both halves have 0 wins although `batched_playout` over the same keys has wins; at `with_budget(k=1)` the wins equal the won playouts with `player_faints <= 1`.
  - `test_protected_death_is_a_failure`: the same state with `protected[0] = 1` set directly (`_replace`; `with_budget` refuses a fainted slot) and `k=3`: 0 wins.
  - `test_deaths_and_buckets`: `deaths_a`/`deaths_b` equal the halves' summed `player_faints`; `buckets_b` equals `np.bincount(deaths.bucket(...), minlength=8)` over half B.
  - `test_merge`: field-wise sums, `refused` or-ed.
  - `test_pending_leaf`: `evaluate_leaves` on `at_replacement(...)` equals `batched_playout` over the same keys.
  - `test_forced_matches_playout_from_action` and the existing tests stay green.
  - `tests/search/test_stage_b.py::test_store_terminal_is_success`: a `_Store` given a batch of three over states, each `winner = WINNER_PLAYER` with `deaths[0] = 1`, `player_faints = 1` (set with `_replace`): `faint_budget` 0, `faint_budget` 1, and `faint_budget` 3 with `protected[0] = 1`: `won == [False, True, False]`; a lost state at `faint_budget` 3: `False`.
  - `test_stage_b.py::test_terminal_win_after_a_death_scores_zero_at_k0`: the Task 5 dead-Monferno state (Prinplup active, knowing only Bubble, against Geodude at full HP, Roark's bag emptied with `empty_bag`; one Bubble KOs, assert with `damage_table`), zero `leaf_fn`, `max_depth=1`: at `faint_budget = 0` every root estimate is 0 and equals `exhaustive(cap=1)`; at `with_budget(k=1)` Bubble's estimate is 1.0 (1e-9) and equals `exhaustive`. (Task 9 makes the `k = 0` root a failed root: that case then asserts the `ValueError` instead; the depth-1 turn after a death made inside the search is Task 7's `test_win_after_a_new_death_scores_zero_at_k0`.)
- [ ] **Step 2: Run** `uv run pytest tests/search/test_leaves.py tests/search/test_stage_b.py -q`; expected: the new tests fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/search -q`, the Task 1 regression and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): leaf values count successes, deaths and the distribution`.

### Task 6: Stage A rungs, ladder, distribution and the tie rule

**Files:**
- Modify: `src/nuzline/search/stage_a.py` (`ActionResult` :40-50, `Ranking` :59-75, `rank_actions` :116-176, module docstring), `src/nuzline/search/stage_b.py:702-740` (`_rank` and `_stage_a` use the score and the tie rule)
- Test: `tests/search/test_stage_a.py`

**Interfaces:**
- Consumes: Task 2 (`decision_actions`, `deaths`), Task 3 (`PlayoutResult`), Task 4.
- Produces:
  - `ActionResult` gains `p_success: float | None = None` (P(win, deaths <= k, no protected death) for the state's `k`; `ci_low`/`ci_high` are its Wilson interval), `ladder: tuple[float, ...] = ()` (rungs 0..3 with the protected set), `distribution: DeathDistribution | None = None`; property `score` (`p_success`, or `p_win0` when `None`). `p_win0 == ladder[0]`.
  - `tie_ordered(rows: Sequence[R], score: Callable[[R], float], deaths: Callable[[R], float]) -> list[R]` (ruling 5; rows need `action`, `ci_low`, `ci_high`, `tied_with_best`). `Ranking.of` uses it with `score` and `e_faints`; `all_zero` when every `score` is 0.
  - `rank_actions` reads `faint_budget`/`protected` from `state`; a pending root ranks the replacements (`decision_actions`).

- [ ] **Step 1: Write the failing tests**:
  - Replace `test_sort_key_is_strict_and_ties_with_the_leader_are_flagged` with `test_tie_rule_prefers_fewer_expected_deaths`: the same five rows give order `[1, 2, 4, 3, 0]` and `tied_with_best == [False, True, True, True, False]`.
  - `test_rank_reports_ladder_and_distribution`: `roark_real_state(...).with_budget(k=2)`, `n=200`: every row's `ladder` is non-decreasing, `ladder[0] == p_win0`, `p_success == ladder[2]`, `sum(distribution.probs) == 1` (1e-12), `distribution.n == 200`.
  - `test_protected_death_fails_every_rung` (Review Focus 1): the Task 5 dead-Monferno state with `protected[0] = 1`, `faint_budget = 3`: every rung and `p_success` 0; `distribution.probs[5] == p_win`.
  - `test_rank_at_a_pending_root`: `at_replacement(roark_state(...))`: the ranked actions are the switches to the living bench slots, labelled `Switch -> X`.
  - `test_budget_is_traced`: `cached_export` monkeypatched to record `(name, static, leaf shapes and dtypes)`: equal for `k=0` and `k=2`.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_stage_a.py -q`; expected: fail.
- [ ] **Step 3: Implement**; `stage_b._stage_a` maps `score`, `e_faints` and `distribution` into `ActionValue` (Task 9 adds those fields; until then only `estimate = score`).
- [ ] **Step 4: Run** `uv run pytest tests/search tests/test_cli.py -q`, the Task 1 regression and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): Stage A ranks by the death budget and reports the ladder`.

### Task 7: Player faints are not terminal

The engine flip. Stage B stays correct for every `k` after it, at any depth: since Task 5 a terminal counts only when `deaths.success` holds for it (a win after a death at `k = 0` is 0) and every playout counts successes, so everything below a failed state scores 0. It still wastes work on failed and pending children until Task 9.

**Files:**
- Modify: `src/nuzline/engine/step.py` (module docstring points 5-6, `_faint_check` :607-640, `needs_replacement` :885-892, `replace` :895-914 docstring), `src/nuzline/reference/engine.py:418-447` and `ref_replace` :1092-1106, `src/nuzline/engine/state.py` (`winner` encoding note), `src/nuzline/engine/STATE.md:81-82`, `docs/design.md` (§3 *Terminal states* row, §4.2 replacement paragraph, §10 simultaneous-faint item, §6 playout termination line)
- Modify tests: `tests/engine/test_step_basic.py:37-41`, `tests/engine/test_field_effects.py:39-44, 236-261, 293-302`, `tests/engine/test_abilities_items.py:187-199`, `tests/search/test_playout.py:122-126, 144-162`
- Test: `tests/engine/test_faints.py`, `tests/test_differential.py`, `tests/search/test_stage_a.py`, `tests/search/test_stage_b.py`

**Interfaces:**
- Consumes: Tasks 2-6.
- Produces: `_faint_check`: the battle ends when the player has no Pokemon with HP (`winner = 1`, a draw included, ruling 2) or the opponent has none (`winner = 2`); a player faint otherwise only records the death. `needs_replacement` serves both sides. The reference engine matches.

- [ ] **Step 1: Write the failing tests** in `tests/engine/test_faints.py` (and the updates):
  - `test_player_faint_counts_a_death_and_the_battle_goes_on`: `roark_state(...).replace_hp(0, 0, 1)` stepped with the opponent's Tackle: `over == 0`, `deaths[0] == 1`, `needs_replacement == [1, 0]`. (Replaces `test_player_faint_is_terminal`; its `player_count=1` version asserts the whiteout.)
  - `test_double_ko_of_last_pokemon_is_a_loss` (Review Focus 2): `roark_state(player_count=1, opp_count=1)` with no PP (Struggle), player HP at most the recoil, Geodude at 1 HP, player faster: both faint in one action; `over == 1`, `winner == 1` (cite ruling 2).
  - `test_last_opponent_with_a_non_last_player_faint_is_a_win`: the same with `player_count=2`: `winner == 2`, `deaths[0] == 1`.
  - `test_entry_faint_needs_a_second_replacement` (Review Focus 3): after a player faint, Stealth Rock on the player's side and Starly at 1 HP: `replace(tables, s, 0, 3)` leaves `deaths[3] == 1`, `player_faints == 2`, `needs_replacement[0] == 1`; `finish_turn` resolves both (`playout`'s policy picks twice).
  - `test_pursuit_ko_still_completes_the_switch`: Budew (1 HP) switching to slot 1 into Pursuit: Budew fainted, `active[0] == 1` (subscript_switch_pokemon.s:10-12 reaches `SwitchAndUpdateMon` at `_026` whatever the HP), `over == 0`.
  - `test_player_replacement_matches_reference`: 200 hypothesis examples: a supported state and legal pair, stepped; when `needs_replacement(after)[0]`, `replace(tables, after, 0, slot)` equals `ref_replace` for every legal slot.
  - `tests/search/test_stage_b.py::test_win_after_a_new_death_scores_zero_at_k0`: the `test_last_opponent_with_a_non_last_player_faint_is_a_win` state (the lead's Struggle KOs Geodude, the recoil KOs the lead; a living bench Pokemon), zero `leaf_fn`, `max_depth=1`: at `k = 0` the Struggle action's estimate is 0 and equals `exhaustive(cap=1)`; at `with_budget(k=1)` it is 1.0 (1e-9) and equals `exhaustive`. (A win reached through a death made inside the search; Task 5's test covers a death before the root.)
  - `test_single_pokemon_ladder_is_flat` (Review Focus 5): `rank_actions` on `roark_state(player_count=1).with_budget(k=2)`, `n=200`: every row's ladder entries equal; no exception.
  - Updates: `test_stealth_rock_can_faint_incoming_and_end_battle` becomes `..._and_needs_a_replacement` (`over == 0`, `deaths[3] == 1`); the two end-of-turn order tests use `player_count=1` (the order decides a whiteout); the Natural Cure/Pursuit and Struggle/Static tests drop `over == 1`; `test_player_faint_ends_playout_as_loss` stays (one Pokemon); `test_playout_from_action_plays_that_action_first` asserts only what the first action fixes, since the playout now goes on with the bench (`opp_count=1`, so it may well win): with `max_turns=1`, the Leer line has `deaths[0] == 1` and `player_faints == 1`, and the switch line has `deaths[0] == 0`, `player_faints == 0` and `refused == 0` (the switch kept the lead alive through turn 1); with `max_turns=100`, the Leer line still has `deaths[0] == 1` (a death is permanent) and the switch line `refused == 0`. No assertion on `won` or `turns`.
- [ ] **Step 2: Run** `uv run pytest tests/engine -q`; expected: the new tests fail.
- [ ] **Step 3: Implement** both engines and the docs (design.md: deaths are counted, `k` scores the result, a whiteout ends the battle; the §10 item closed with ruling 2's citations).
- [ ] **Step 4: Run** `uv run pytest -q` (whole suite, including `tests/search/test_k0_regression.py`) and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): a player faint is a death, not the end of the battle`.

### Task 8: Replacement observations and expansion

**Files:**
- Modify: `src/nuzline/engine/observe.py` (module docstring, new function), `src/nuzline/search/expand.py` (module docstring, new functions)
- Test: `tests/engine/test_observe.py`, `tests/search/test_expand.py`

**Interfaces:**
- Consumes: Task 3 (`send_replacements`, `replacement_pass`), Task 7.
- Produces:
  - `observe.replacement_observation(after: BattleState) -> jnp.ndarray` (`int32[OBS_SIZE]`): no action ran (`act_kind` `ACT_NONE`, flags 0, `order` -1, `pursuit` 0), the visible projection of `after`.
  - `expand.replace_rows(tables, states, actions) -> tuple[BattleState, np.ndarray, np.ndarray]`: `replacement_pass(tables, s, a - MOVE_ACTIONS)` and its `replacement_observation`, `(after, refused bool[N], obs int32[N, OBS_SIZE])`, one cached program `"expand_replace"` through `_run`.
  - `expand.expand_replacements(tables, states, requests: list[tuple[int, int]]) -> tuple[BattleState, dict[int, Child]]`: request `(state index, player action)`; one `Child` each, `prob = 1.0`; the store deduplicated by state bytes as `expand` does.
  - A turn's child at a player replacement is the decision-point state (`send_replacements` stopped; nobody has entered) observed as today.

- [ ] **Step 1: Write the failing tests**:
  - `test_replacement_observation_shows_both_newcomers`: after `replacement_pass` on `at_replacement(..., opp_down=True)`: the `active` field equals both new slots, `act_kind == [0, 0]`, `order == -1`.
  - `test_expand_replacements_match_replacement_pass`: for each legal slot of a pending state: the child's state bytes, observation and `refused` equal a direct call; two requests reaching one state share a store row.
  - `test_turn_child_stops_at_the_player_replacement`: `roark_state(opp_faster=True)` with the lead at 1 HP, request (Leer, Tackle): a child with the lead at 0 HP, `over == 0`, `active` unchanged, its observation's `player_hp` 0 for that slot; both actives down in one turn: `roark_state(opp_count=2, opp_faster=True)` with the lead at 1 HP and Geodude at 1 HP with all PP zeroed (`with_pp(1, 0, [0, 0, 0, 0])`; action 0 is Struggle, assert it is in `legal_actions(tables, s, 1)`), request (Leer, 0): Geodude's Struggle (it never misses) KOs the lead and its recoil (max HP / 4) KOs Geodude; every child of the request has `hp[0, 0] == 0`, `hp[1, 0] == 0`, `over == 0`, `active == [0, 0]` (nobody replaced), `replace_mul == NORMAL_POWER_MUL`, `deaths[0] == 1`.
  - `test_decision_actions_constant_on_equal_obs`: every child of the `roark_real_state` root (all legal pairs) grouped by observation has one `decision_actions` row per group.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_observe.py tests/search/test_expand.py -q`; expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): expand and observe player replacements`.

### Task 9: Stage B with replacement decisions, failed nodes and the death distribution

**Files:**
- Modify: `src/nuzline/search/stage_b.py` (module docstring; `ActionValue` :126-144, `PlanNode` :147-168, `SolveResult` :171-192, `_Node` :203-215, `_Store` :218-252, `_level` :366-400, `_children` :402-436, `_evaluate` :440-458, `_backup`/`_branch` :462-515, `run` :556-689, `_leader` :696-699, `_rank` :702-709, `solve` :743-791), `tests/search/exhaustive.py`, `tests/search/fixtures.py`, `tests/search/test_stage_b.py` (Task 5's `k = 0` case becomes a failed root)
- Create: `tests/search/test_stage_b_budget.py`

**Interfaces:**
- Consumes: Tasks 2-8 (`decision_actions`, `deaths.*`, `LeafValue` fields and `merge`, `tie_ordered`, `expand_replacements`).
- Produces:
  - `ActionValue` gains `e_deaths: float = 0.0`, `failed: float = 0.0` (summed reach of failed nodes in the plan), `distribution: DeathDistribution | None = None` (half-B absolute bucket sums, `n=None`; they sum to 1 minus the refused mass). `estimate` is P(win, deaths <= k, no protected death).
  - `PlanNode` gains `replacement: bool = False` (a decision among replacements); new kind `"failed"`.
  - `SolveResult` gains `deaths_allowed: int = 0`, `protected: tuple[int, ...] = ()` (from the root state).
  - `solve(...)` signature unchanged; a pending root is a replacement decision; a failed root raises `ValueError`. `leaf_fn(states, weights, pairs=None)` is also called once per pass for the failed states (`weights` all 0, `pairs` None).
  - `stage_b.Backup(NamedTuple)`: `a: float`, `b: float`, `bound: float`, `weights: dict`, `plan: list[PlanNode]`, `deaths_a: float`, `deaths_b: float`, `buckets_b: np.ndarray` (float64[8], absolute: reach-weighted bucket probabilities), in that order. `_backup` and `_branch` return it (today's first five fields keep their meaning and order); the `per_action` dicts of `_backup` and the precision check hold `Backup` values, and `_leader(per_action: dict[int, Backup]) -> int` and every other reader use the field names, never positions.
  - `tests/search/exhaustive.py::exhaustive(tables, state, cap, group=True)` (signature unchanged): terminal value = `deaths.success` of the state (since Task 5); a failed state is 0 and not expanded; a pending group is a decision over `decision_actions` whose children come from `playout.replacement_pass` and `replacement_observation` (grouped by observation), not counting a turn; `choose(path)` paths include replacement observations.
  - `tests/search/fixtures.py` (numbers at `b36b1d0`: Spe Monferno 30, Prinplup 21, Starly 19, Geodude 10 without `opp_faster`; HP Geodude 32, Prinplup 44, Starly 30; one Bubble does 48-60 to Geodude, Starly's Wing Attack 3-4, Geodude's Tackle 4-5 to Prinplup and 7-9 to Starly; Tackle is 95 % accurate, Bubble and Wing Attack never miss):
    - `faint_pending_root(tables, names) -> BattleState`: `roark_state(player_count=4, opp_count=1)` (no `opp_faster`), Roark's bag emptied (`empty_bag`), Grotle's slot (2) emptied (`species = 0`, `hp = 0`), Geodude at full HP knowing only Tackle (its other slots' PP zeroed, as `endgame` does), Monferno at 0 HP (`deaths[0] = 1`, `player_faints = 1`), Prinplup knowing only Bubble (other PP zeroed), Starly unchanged; `with_budget(k=1)`. The fixture asserts with `damage_table` and the stats: Bubble's smallest non-critical roll >= Geodude's HP (one Bubble always KOs), the largest critical roll of each of Starly's moves < Geodude's HP (Starly cannot KO in one hit), and `Spe(Geodude) < Spe(Prinplup)`.
    - `sacrifice_endgame(tables, names) -> BattleState`: the same team before the faint: no deaths, `faint_budget` 0, Monferno alive at 1 HP knowing only Leer (other PP zeroed) with its Spe stat set explicitly to Geodude's minus 1 (not `opp_faster`, which would make Geodude faster than Prinplup too), Prinplup at 4 HP knowing only Bubble, Geodude at full HP knowing only Tackle. The fixture asserts `Spe(Monferno) < Spe(Geodude) < Spe(Prinplup)`, that Tackle's smallest non-critical roll on Prinplup is >= 4 (a Tackle that hits KOs it: P = 0.95) and that one Bubble KOs Geodude. So a direct switch to Prinplup eats a Tackle (the `k=0` best is about 0.05, the miss), while after Leer sacrifices Monferno, Prinplup enters free, moves first and wins (`k=1`: Leer about 0.95). It also asserts (with `exhaustive`, cap 2) that the `k=1` best value exceeds the `k=0` best by at least 0.3 and that some root value lies strictly between 0 and 1. Callers set `k` with `with_budget`.
    - `certain_zero_leaf(states, weights, pairs=None) -> list[LeafValue]`: `LeafValue(0, 10**6, 0, 10**6, False)` per state: a leaf worth 0 with a negligible interval. The Task 9 and Task 10 solves use it instead of `test_plan_tree.zero_leaf` (`n = 1`, whose interval half-width is about 0.9 per unit of leaf weight): under ruling 5 a wide zero leaf would overlap a valued line's interval, and with fewer expected deaths (a zero leaf reports none) `Switch -> Starly` would lead `sacrifice_endgame` at `k=1` instead of Leer.
- Algorithm (what the interfaces leave open): child groups are classified in this order: refused, terminal (`over`), failed (`deaths.over_budget`, ruling 3), pruned (reach below epsilon), replacement decision (pending; ruling 4), leaf (`k + 1 == depth`), decision. After a turn level, the level's replacement decisions are expanded with `expand_replacements` (reach `w` per state, children grouped by `(obs, refused)` and classified with the same `k + 1`), repeating while new replacement decisions appear; a pending root does this before the first turn level. `_level(self, frontier, k, depth, dropped, at_root: bool)`: the root-only logic of today's `_level` (stage_b.py:371-373 raises `UnsupportedAi` on `self.ai[0] is None`; :384-388 expands every pair with mass whatever its reach; :389-397 fills `dropped` from `self.expansions[(0, a, o)]`; :398-399 skips dropped actions) keys on `at_root`, not on `k == 0`, and applies to the frontier that holds the outcomes of the root decision. For an ordinary root that is the root itself, as today (`at_root=True` on the first turn level). For a pending root the root decision is the replacement, and the first turn level's frontier is its children (after any chained entry-faint replacement), each mapped back to the root replacement action that leads to it (`_Node.root_action: int | None`, `None` off this level): every pair with mass is expanded, and a child whose AI distribution is refused (`self.ai[s] is None`) or that has a sampling-only pair drops that root replacement action, with the reason in `dropped` (`"<label> is dropped: the AI's action distribution after it is refused"`, or today's sampling-only text); `UnsupportedAi` is raised only when every root replacement is dropped for a refused AI. The pending root's own AI distribution is not required (its row is never requested; its active has 0 HP). Failed states are evaluated once per pass by `evaluate_leaves` with `2 * leaf_floor` playouts on `fold_in(key, 3)`, outside the ruling-8 weights, the `W` maps and the top-ups. The backup (`Backup`) carries, besides the half-A/half-B values, bound and `W`: half-A and half-B expected deaths, absolute like `a`/`b` (terminal: `prob * player_faints`; leaves, pruned, forced and failed: `prob * e_deaths_*` of their evaluation; refused: 0), and the half-B bucket vector (terminal: `prob` in its own bucket; evaluated nodes: `prob * buckets_b / n_b`, where a `LeafValue` with `buckets_b == ()`, as every zero `leaf_fn` in the suite returns, counts as zeros, the rule 4c's `merge` also uses; refused: zeros). A decision node's `Backup` is its chosen action's. `_leader` applies ruling 5's inner rule; `_rank` is `tie_ordered` with `estimate` and `e_deaths`.

- [ ] **Step 1: Write the failing tests** in `tests/search/test_stage_b_budget.py` (`leaf_fn=certain_zero_leaf`, `epsilon=0` unless stated):
  - `test_pending_root_matches_exhaustive`: `faint_pending_root`, `max_depth=1`: every root estimate equals `exhaustive(cap=1)` within 1e-6; the root actions are exactly `MOVE_ACTIONS + 1` and `MOVE_ACTIONS + 3`; Prinplup's estimate is 1.0 (1e-9; Bubble goes first, never misses, always KOs) and Starly's 0.0, so the ranking's first, `Switch -> Prinplup`, is decided by value, not by the tie rule.
  - `test_pending_root_drops_only_an_ai_refused_replacement`: `faint_pending_root`, `max_depth=1`, `stage_b.ai_rows` monkeypatched to the real `ai_rows` with the rows of every state whose player active is slot 3 (Starly) set to NaN: no `UnsupportedAi`; the ranking holds only `Switch -> Prinplup` (estimate 1.0); `SolveResult.dropped` has exactly one entry, for `MOVE_ACTIONS + 3`, whose text contains `refused`. With every row NaN, `solve` raises `UnsupportedAi`. With the real `ai_rows` wrapped in a recording spy, no recorded batch holds the pending root's state.
  - `test_sacrifice_matches_exhaustive`: `sacrifice_endgame` at `k=0` and `k=1`, `max_depth=2`: estimates equal `exhaustive(cap=2)` within 1e-6; at `k=1` the best root action is Leer, and its plan holds a decision node with `replacement=True` whose action is `MOVE_ACTIONS + 1` (Prinplup), the exhaustive choice on the same path.
  - `test_failed_nodes_value_zero`: `sacrifice_endgame` at `k=0`: under Leer, `kind == "failed"` nodes exist, `ActionValue.failed` equals their summed `prob` (1e-9), they add nothing to `estimate` or the interval; a recording `leaf_fn` sees them in a call with all-zero weights.
  - `test_distribution_sums`: `sacrifice_endgame` at `k=1` with the default evaluators (`budget_playouts=2000`, `max_depth=1`): every ranked action's `sum(distribution.probs) == 1 - refused` (1e-9) and `distribution.rung(1) == estimate` (1e-9).
  - `test_protected_line_matches_exhaustive` (Review Focus 1): `faint_pending_root` changed so that a Prinplup death is within budget unless it is protected: Geodude at 3 HP (Starly's Wing Attack KOs it, assert) with its Spe set to 25 (faster than Prinplup and Starly, assert), Prinplup at 4 HP (a Tackle that hits KOs it, assert), `max_depth=2`, unprotected `with_budget(k=2)` and protected `with_budget(k=2, protected=[1])`: estimates equal `exhaustive(cap=2)` within 1e-6 in both; unprotected, `Switch -> Prinplup` is 1.0 (1e-9: a missed Tackle lets Bubble win; a hit is death 2, then Starly survives a Tackle and Wing Attack wins); protected it is 0.05 (1e-9: only the missed Tackle), lower by more than 0.5. (The protected line is failed whatever `k` is: ruling 3.)
  - `test_second_replacement_decision_in_plan` (Review Focus 3): `faint_pending_root` with Stealth Rock on the player's side and Prinplup at an HP the rocks KO, `k=2`: under `Switch -> Prinplup` the plan holds a second `replacement=True` decision; its value matches `exhaustive`.
  - `test_inner_tie_prefers_fewer_deaths`: `stage_b._leader({4: Backup(a=0.5, b=0.5, bound=0.0, weights={}, plan=[], deaths_a=1.0, deaths_b=1.0, buckets_b=np.zeros(8)), 7: <the same with deaths_a=0.2>})` is 7; with `a` 0.5 and 0.5 + 1e-9 the higher `a` wins whatever the deaths; with equal `a` and equal `deaths_a` the lower action wins; `_rank` on synthetic `ActionValue`s follows `tie_ordered`.
  - `tests/search/test_stage_b.py::test_terminal_win_after_a_death_scores_zero_at_k0` (Task 5): its `faint_budget = 0` case now asserts that `solve` raises `ValueError` (a failed root) and that `exhaustive` values every root action 0; the `k=1` case is unchanged.
  - `test_budget_does_not_change_k0`: `tests/search/test_k0_regression.py` stays green; on `roark_real_state` at `k=0`, `max_depth=1`, no node is `replacement=True` and every failed node's value is 0.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_stage_b_budget.py tests/search/test_stage_b.py -q`; expected: the new tests fail.
- [ ] **Step 3: Implement** `stage_b.py`, `exhaustive.py` and the fixtures.
- [ ] **Step 4: Run** `uv run pytest tests/search -q` (Stage B, plan tree and regression included) and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(search): Stage B replacement decisions, death budget and distribution`.

### Task 10: Plan tree labels, `--deaths`, `--protect` and docs

**Files:**
- Modify: `src/nuzline/search/plan_tree.py` (`label` :133-190, `_KIND_TEXT` :76-83, `_action_label` :241-256, `_Walk` :282-324, `_node_json` :332-355, `to_json` :358-389, `_outcome_text` :408-411, `_lines` :414-419, `render` :422-467), `src/nuzline/cli.py` (module docstring, `_parser` :93-159, `_fight` :177-206, `_rank` :224-269, `_solve` :272-347), `README.md` (rank/solve usage)
- Test: `tests/search/test_plan_tree.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 6 (`ActionResult` fields), Task 9 (`ActionValue`, `PlanNode.replacement`, `SolveResult.deaths_allowed/protected`), `DeathDistribution`.
- Produces:
  - `label(obs, names, *, prev_opp=None, prev_mine=None, trainer="Opponent", max_hp=None, budget: tuple[int, int] | None = None) -> str`: adds `"<species> fainted"` when the player's active shows 0 HP, followed with `budget = (k, fainted slots at the root)` by `"(death d of k allowed)"` or `"(death d, over the k allowed)"`; `"you sent in <species>"` when `prev_mine` differs from the active and the player did not switch by action.
  - `render`: the header column is `P(win, 0 faints) [95% CI]` at `k = 0` and `P(win, <=K deaths) [95% CI]` otherwise, plus an `over budget` column; a replacement decision's outcome is `send in <species>`; a failed node's is `(over the death budget: play on for the win, not planned)`; after the ranking one line `deaths under the leader: 0: a %  1: b %  2: c %  3+: d %  whiteout: e %  unfinished: f %  protected lost: g %`; with protected slots a line `protected: <names> (a line where one faints fails)`.
  - `to_json`: nodes carry `"replacement"`; ranking rows carry `e_deaths`, `failed` and `distribution` (`DeathDistribution.as_dict()` or `None`); top level `deaths_allowed`, `protected`.
  - CLI: `rank` and `solve` take `--deaths K` (int, default 0; outside 0..3 is a `ValueError`, exit 1) and `--protect NAME` (`action="append"`, ruling 10). `cli.protected_slots(player: Sequence[PokemonSpec], names: Names, values: Sequence[str]) -> tuple[int, ...]` (0-based party indices, sorted, deduplicated; a value of digits is a 1-based slot number, ruling 10). The state is `make_state(...).with_budget(args.deaths, slots)`. `rank` prints `P(win, <=K deaths)` (the k=0 header unchanged), `P(win)`, `E[deaths]`, ladder columns `<=0 <=1 <=2 <=3 (screen)` (`-` for a row without a ladder) and the leader's distribution line. `all_zero_footer(k: int) -> str` (k=0: today's `ALL_ZERO_FOOTER`; else `"all lines 0 %: no line within K deaths found at this depth"`). The solve run record adds `"deaths"` and `"protected"` (slots).

- [ ] **Step 1: Write the failing tests**:
  - `test_plan_tree.py::test_label_names_the_faint_and_the_budget`: a `sacrifice_endgame` child observation where Monferno fainted, `budget=(1, 0)`: the label contains `Monferno fainted (death 1 of 1 allowed)`; with `budget=(0, 0)`: `(death 1, over the 0 allowed)`.
  - `test_render_replacement_and_failed_lines`: `render` of the Task 9 `sacrifice_endgame` solves (`max_depth=2`, `leaf_fn=certain_zero_leaf`, `show_depth=2`). At `k=1` (leader Leer): a line ending `-> send in Prinplup` (Monferno fainted under Leer), the `P(win, <=1 deaths)` header and the `deaths under the leader:` line. At `k=0` (leader `Switch -> Prinplup`): a line ending `(over the death budget: play on for the win, not planned)` (the Tackle that KOs Prinplup) and today's `P(win, 0 faints) [95% CI]` header. All lines ASCII. (`render` shows the plan under the leader only, so each line is asserted on the solve whose leader holds it.)
  - `test_json_has_replacement_and_distribution`: `json.loads(json.dumps(to_json(...)))` keeps `replacement`, every ranking row's `distribution` keys are `BUCKETS + ("n",)`, `deaths_allowed == 1`.
  - `test_cli.py::test_cli_rank_deaths_and_protect`: `main(["rank", "--party", SAMPLE, "--trainer", "roark", "--deaths", "1", "--protect", "Prinplup", "--n", "50", "--devices", "1"])` returns 0 and prints `P(win, <=1 deaths)`, the ladder header, `protected: Prinplup` and `deaths under the leader:`.
  - `test_cli_solve_deaths`: `solve --max-depth 1 --playouts 2000 --deaths 1 --devices 1` returns 0; the run record has `"deaths": 1`, `"protected": []`.
  - `test_cli.py::test_protected_slots`: on the sample party, `protected_slots(party, names, ["2"]) == (1,)` (1-based: Prinplup), `["prinplup", "2"] == (1,)`, `["1", "Starly"] == (0, 3)`; `["0"]` and `["7"]` raise `ValueError`.
  - `test_cli_bad_budget_flags` (Review Focus 4), parametrized: `--deaths 4`, `--deaths -1`, `--protect Pikachu`, `--protect 9`, `--protect 0`, `--protect Budew` with `sample_party_fainted.toml`, and `--protect Starly` with a temporary party TOML holding two Starly: exit 1, stderr names the flag and the value.
  - `test_cli_user_save_rank`: skips without `NUZLINE_USER_SAVE`; `rank --save $NUZLINE_USER_SAVE --trainer roark --deaths 1 --protect Furret --n 50 --devices 1`: when the output contains `Missing (` (coverage refuses until plan 4a lands) skip with that reason, else exit 0 and the ladder header.
  - `test_cli_imports_no_jax` stays green.
- [ ] **Step 2: Run** `uv run pytest tests/search/test_plan_tree.py tests/test_cli.py -q`; expected: fail.
- [ ] **Step 3: Implement**; README gains the two flags in the `rank` and `solve` usage lines.
- [ ] **Step 4: Run** `uv run pytest -q`, `uv run ruff check src tests` and `uv run ruff format --check src tests`; expected: pass. Then by hand: `uv run nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth 1 --deaths 1 --devices 1` prints the ladder header, the distribution line and a plan.
- [ ] **Step 5: Commit** `feat(cli): --deaths and --protect for rank and solve; plan labels for deaths`.

---

## Self-review notes

- Spec coverage: §4 objective, default `k = 0`, distribution -> Tasks 2, 5, 6, 9, 10; ladder semantics (rungs scored at once from one batch, Stage B on the requested rung, other rungs labelled "screen") -> Tasks 6 and 10 (the per-rung best setup is prep, plan 4c); what `k` does (battles continue, `k` only scores, play on after the budget) -> Tasks 4, 7, 9 (ruling 3); tie-breaks -> Tasks 6 and 9 (ruling 5); §5.1 budget as traced data -> Tasks 2, 4, 6; replacement decision in Stage B (expand, observe, plan tree) and playout heuristic -> Tasks 3, 8, 9, 10; Set mode -> Task 3; simultaneous-faint order -> ruling 2, Tasks 3 and 7; death distribution and the `k = 0` regression -> Tasks 1, 5, 6, 9; §10.1 protected -> Tasks 2, 4, 5, 6, 9, 10. The "leave it out of the team" exclusion of §10.1 is candidate filtering, plan 4c.
- Ordering: the engine flip (Task 7) comes after every consumer can read deaths and successes (playouts and leaves in Tasks 3-5, Stage B's and `exhaustive`'s terminal value in Task 5), so the suite and the `k = 0` regression stay green at every commit and no commit values a win after a death as a `k = 0` success; Stage B's proper handling of the new nodes follows in Task 9.
