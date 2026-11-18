# Milestone 4a: Coverage for the User's Roark Pool, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every move and ability the user's Roark pool can bring (party and boxes at the level cap, current moves, Move Relearner moves and the bag's TM10) is implemented exactly from the decomp, and `nuzline coverage --save <user save> --trainer roark` reports full support for the party fight and for the pool.

**Architecture:** Each new mechanic follows the milestone-1 pattern: an engine id (`engine/ids.py`), a decomp mapping (`tools/effect_map.py`, rebuilt tables), the JAX implementation (`engine/effects.py`, `damage.py`, `step.py`, `outcome.py`, `hooks.py`), the plain-Python mirror (`reference/engine.py`), its exact chance law for Stage B (`engine/enumerate.py`), what the player sees (`engine/observe.py`), and the trainer AI's view where Roark's scripts read it (`ai/`). Learnsets and TM/HM compatibility are packed into `manifest.json`; a new `nuzline.prep` package holds level-to-cap and the candidate pool, which `nuzline coverage --save` reports.

**Tech Stack:** Python 3.12, JAX (CPU), numpy, pytest, hypothesis, ruff.

**Spec:** `docs/superpowers/specs/2026-10-31-prep-search-design.md` §5.2 and §5.3 (parent `docs/design.md` §4.4). Milestone-1 precedent: `docs/superpowers/plans/2026-10-28-m1a-data-and-engine.md` Tasks 8a-8e.

**Plan family:** Milestone 4 is three plans. Plan 4a (this one: coverage and data) and plan 4b (faint budget and Set-mode player replacements, spec §5.1) are independent and can run in parallel; plan 4c (prep search pipeline, CLI, validation; spec §6-§8) depends on both. 4a and 4b both edit `engine/state.py`, `engine/step.py` and `reference/`; whichever merges second resolves those conflicts mechanically (new fields and new checks only, no shared logic).

## Global Constraints

- The decomp (`external/pokeplatinum`, pinned commit `579d2d5c65e736a1a4268fb7acbfd70712177c6c`) is authoritative for every mechanic; cite file:line in docstrings as the existing code does. Line numbers below hold at that pin.
- "Each is implemented exactly from the decomp, with the same tests as milestone 1 (decomp-derived unit tests, differential against the reference engine)" (spec §5.2): every mechanic gets decomp-derived unit tests and is mirrored in `reference/engine.py`; `tests/test_differential.py` (500 hypothesis examples) passes with the new moves in `tests/strategies.py`.
- Never approximate silently: anything not implemented stays `UNSUPPORTED` / refused by name (spec §2 anti-goal).
- Stage B stays exact: every new random roll is an `Outcome` field with an exact law in `enumerate.py` and a decomp-faithful draw in `sample_outcome`; every visible state change is in `observe`; hidden counters (`disable_turns`, `bide_damage`) are not.
- `sample_outcome` keeps its existing `jax.random.split(key, 14)`; new fields draw from `jax.random.fold_in(key, 14 + i)`, so every existing field's samples, and every seeded Stage A number, are unchanged.
- No new runtime dependencies. `nuzline.cli` does not import jax at module import (`tests/test_cli.py:202`). CLI output is ASCII only; exit codes stay 0 ok / 1 error / 2 refused.
- Existing tests are not weakened. Tests that used a now-supported mechanic as a refusal example switch to one that stays refused (Miracle Eye for a player move, opponent-side Hidden Power).
- Tables: after editing `tools/effect_map.py`, `tools/extract_decomp.py` or `tools/build_tables.py`, run `uv run python -m tools.extract_decomp` then `uv run python -m tools.build_tables`; commit `data/gen4_platinum/manifest.json` (tables.npz is gitignored).
- Level-to-cap never evolves a Pokemon (spec §2 non-goal: evolution toggle): every evolution is assumed cancelled.
- The user's save is never committed; tests read it only through `NUZLINE_USER_SAVE` and skip without it. Never create, edit or delete anything in a folder named `backup` (any case, any depth).
- Commands: `uv run pytest`, `uv run ruff check src tests tools`, `uv run ruff format`. ruff `line-length = 100`.

## The move list (computed from the data)

Level-up moves at or below level 16 of the save's 10 species (11 Pokemon: two Venonat) whose decomp effect maps to `UNSUPPORTED` today, plus TM10:

| Move | Decomp effect (id) | Species | Task |
|---|---|---|---|
| Foresight | `BATTLE_EFFECT_FORESIGHT` (113) | Furret L1, Venonat L1 | 4 |
| Night Shade | `BATTLE_EFFECT_LEVEL_DAMAGE_FLAT` (87) | Spinarak L15 | 5 |
| Hidden Power (TM10) | `BATTLE_EFFECT_RANDOM_POWER_BASED_ON_IVS` (135) | every species but Kricketot | 5 |
| Magnitude | `BATTLE_EFFECT_PSYWAVE` (126; the decomp's name, script 0126 is Magnitude's) | Geodude L15 | 5 |
| Confusion | `BATTLE_EFFECT_CONFUSE_HIT` (76) | Venonat L11 | 6 |
| Mud Sport | `BATTLE_EFFECT_HALVE_ELECTRIC_DAMAGE` (201) | Geodude L4 | 6 |
| Bug Bite | `BATTLE_EFFECT_EAT_BERRY` (224) | Kricketot L16 | 6 |
| Disable | `BATTLE_EFFECT_DISABLE` (86) | Psyduck L14, Venonat L1 | 7 |
| Bide | `BATTLE_EFFECT_BIDE` (26) | Kricketot L1 | 8 |
| Rollout | `BATTLE_EFFECT_DOUBLE_POWER_EACH_TURN_LOCK_INTO` (117) | Bidoof L13 | 8 |

Moves sharing those effects become supported with them: Odor Sleuth (113), Seismic Toss (87), Psybeam, Dizzy Punch, DynamicPunch, Signal Beam, Water Pulse, Rock Climb (76), Pluck (224), Ice Ball (117). Abilities: Insomnia (Spinarak), Cloud Nine (Psyduck), Compoundeyes (Venonat), Shed Skin (Kricketot) in Task 9.

## Rulings (controller's calls; the spec is silent)

1. **Level-to-cap is a Pokemon Center visit after the candies:** level `max(level, cap)`, `stats=None` (recomputed), full HP, status cleared, PP = each move's max PP with its PP Ups (`move_pp * (5 + pp_ups) // 5`, equal to the table PP without PP Ups), moves unchanged. Candies = `max(0, cap - level)`. Entry HP and status are plan 4c's entry-state dimension.
2. **Relearner options** are a species' distinct level-up moves at or below the cap (spec §5.3), in learn order, minus the moves it already knows. **TM options** are the TM/HM items in the bag's `tms_hms` pocket that the species can learn, minus moves already known.
3. **TM data lives in `manifest.json`** (version 7): host-only lookups, not device tables.
4. **Mud Sport is per battler** (`MOVE_EFFECT_MUD_SPORT` in `moveEffectsMask`, cleared on switch-out), replacing today's field-wide `mud_sport`; damage halves Electric power while any active battler has it (`BattleSystem_AnyBattlersWithMoveEffect`, battle_lib.c:6812).
5. **`bide_damage` is `storedDamage` only while Biding** (`bide_turns > 0`), else 0: the decomp's value outside a Bide is never read (subscript_bide_start.s zeroes it), and keeping it 0 lets equal states merge.
6. **Cloud Nine has no engine behaviour**: every decomp branch on it is a weather branch, and coverage refuses every weather but none. Task 9 pins this with a decomp-scan test.
7. **`nuzline coverage --save`** also prints a "Pool" section (Task 10); the exit code still follows the party fight alone, since an unsupported boxed Pokemon only excludes that candidate (spec §5.2). `--cap` defaults to the highest party level until plan 4c adds `config/level_caps.toml`.
8. **Defense Curl** stays `STAT_SELF` (Def +1) and sets `curled` through `move_param[m, 2] = 1` (Rollout doubles its power when curled).

## Review Focus

1. **A boxed Pokemon whose relearner set holds an unsupported move:** the pool report names that Pokemon and move, the party fight is still judged on its own, exit code 0. Test in Task 10.
2. **A player locked into Bide or Rollout at a decision:** the only legal action is the locked move, no switch; the observation shows the lock, so Stage B never raises `HiddenLegality`. Test in Task 8.
3. **Disable on the opponent's only usable move:** the AI's distribution puts all mass on Struggle, exactly as the decomp's `STRUGGLING_ALL`. Test in Task 7.
4. **A Pokemon already above the cap:** level unchanged, 0 candies, stats recomputed at its own level equal its stored stats. Test in Task 2.
5. **Hidden Power as the player's last move into the AI (`moveHit`):** the AI's switch passes see its IV type, not Normal, and the fight is not refused. Test in Task 5.

---

### Task 1: TM/HM compatibility and learnsets (`data/learnsets.py`)

Read first: decomp `tools/dataproc/src/speciesproc.c:310-328` (`ProcTMLearnset`), `src/pokemon.c:4703-4738` (`CanPokemonFormLearnTM`), `src/item.c:296-324`, `res/items/data/tm10.json` (`teachesMove`), docs/decomp_paths.md "TM/HM compatibility" row.

**Files:**
- Modify: `tools/extract_decomp.py` (`extract_tm_learnsets`, `extract_items`, `extract_static`), `tools/build_tables.py` (`MANIFEST_VERSION = 7`, manifest keys), `data/gen4_platinum/manifest.json` (regenerated), `docs/decomp_paths.md` (TM row: where the manifest keeps it)
- Create: `src/nuzline/data/learnsets.py`
- Test: `tests/tools/test_extract_decomp.py`, `tests/data/test_learnsets.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `extract_tm_learnsets(decomp: Path) -> dict[str, list[int]]`: species id (string key, as `learnsets.json`) -> sorted TM/HM indices, TM01-TM92 = 0-91, HM01-HM08 = 92-99 (`ProcTMLearnset` bit numbering); written to `tm_learnsets.json`.
  - `items.json` entries gain `"teaches_move": int | None` (decomp move id of `teachesMove`).
  - `manifest.json` keys: `"level_learnsets"` (per species id: `[[level, move], ...]` in learn order), `"tm_learnsets"` (per species id: TM indices), `"tm_moves"` (100 move ids, index = TM index), `"first_tm_item"` (328, `ITEM_TM01`).
  - `class Learnsets` with:
    - `level_up(species: int) -> tuple[tuple[int, int], ...]`
    - `relearner_moves(species: int, level: int) -> tuple[int, ...]`: distinct moves learned at a level <= `level`, first-occurrence order.
    - `tm_move(item: int) -> int`: the move a TM/HM item teaches, 0 for any other item.
    - `can_learn_tm(species: int, item: int) -> bool`: False for a non-TM item; form 0's data (forms with their own TM data are refused by `SaveMon.to_spec`).
  - `load_learnsets(dir: Path = tables.DEFAULT) -> Learnsets`

- [ ] **Step 1: Write the failing tests**
  - `tests/tools/test_extract_decomp.py::test_tm_learnsets_match_by_tm`: for Furret, Kricketot, Shedinja the indices equal those parsed from `res/pokemon/<s>/data.json` `learnset.by_tm`; Kricketot's list is `[]`; `items.json` gives TM01 -> Focus Punch, TM10 -> Hidden Power, HM08 -> Rock Climb, Potion -> None.
  - `tests/data/test_learnsets.py`:
    - `test_relearner_moves_at_16` (names compared through `names.move_names`): Furret `("Scratch", "Foresight", "Defense Curl", "Quick Attack", "Fury Swipes")`; Kricketot `("Growl", "Bide", "Bug Bite")`; Venonat `("Tackle", "Disable", "Foresight", "Supersonic", "Confusion", "PoisonPowder")`; Bidoof `("Tackle", "Growl", "Defense Curl", "Rollout")`; Geodude `("Tackle", "Defense Curl", "Mud Sport", "Rock Polish", "Rock Throw", "Magnitude")`; Kricketot at level 15 has no Bug Bite.
    - `test_tm_moves`: `tm_move(names.item("TM10")) == names.move("Hidden Power")`, `tm_move(names.item("HM01")) == names.move("Cut")`, `tm_move(names.item("Potion")) == 0`.
    - `test_tm_compatibility`: TM10 learnable by Furret, Staravia, Monferno, Zubat, Spinarak, Psyduck, Bidoof, Venonat, Geodude; not by Kricketot or Magikarp; `can_learn_tm(Furret, Potion)` is False.
- [ ] **Step 2: Run** `uv run pytest tests/tools/test_extract_decomp.py tests/data/test_learnsets.py -q`; expected: fail (missing function/module).
- [ ] **Step 3: Implement** the extractor, the manifest keys and `Learnsets`; regenerate the tables (Global Constraints).
- [ ] **Step 4: Run** the tests, the full `uv run pytest tests/data tests/tools -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(data): TM/HM compatibility and level-up learnsets in the manifest`.

### Task 2: Level to cap and the Shedinja HP rule (`prep/leveling.py`)

Read first: decomp `src/pokemon.c:564-640` (`Pokemon_CalcStats`; Shedinja at :598 and :634), `MoveTable_CalcMaxPP`; ruling 1.

**Files:**
- Modify: `src/nuzline/engine/stats.py`, `src/nuzline/engine/state.py` (`_fill_slot` passes the species), `src/nuzline/saves/pkm.py` (`_check_stats` uses `calc_stats(..., species=)`, `_SHEDINJA` replaced by `SPECIES_SHEDINJA`)
- Create: `src/nuzline/prep/__init__.py`, `src/nuzline/prep/leveling.py`, `tests/prep/__init__.py`
- Test: `tests/engine/test_stats.py`, `tests/prep/test_leveling.py`

**Interfaces:**
- Consumes: `SaveMon` (`saves/pkm.py`), `SaveMon.to_spec`.
- Produces:
  - `stats.SPECIES_SHEDINJA = 292`
  - `calc_stats(base, ivs, evs, level, nature_mods, species: int = 0) -> np.ndarray`: HP is 1 when `species == SPECIES_SHEDINJA`.
  - `candies_to_cap(level: int, cap: int) -> int` = `max(0, cap - level)`
  - `level_to_cap(mon: SaveMon, tables: Tables, cap: int) -> PokemonSpec` (ruling 1; raises what `to_spec` raises).

- [ ] **Step 1: Write the failing tests**
  - `test_shedinja_hp_is_one`: Shedinja at levels 1, 16, 100 with any IVs/EVs has HP 1; its other stats equal the formula; a non-Shedinja species is unchanged by the new argument.
  - `tests/prep/test_leveling.py` (saves built with `tests/saves/synthetic.py`):
    - `test_level_to_cap_rebuilds`: a level-10 party mon at 5/30 HP, poisoned, a move at 3/35 PP with 1 PP Up: the spec has level 16, `stats is None`, `current_hp is None`, `status == 0`, `status_counter == 0`, that move's PP 42 and table PP for the others; `make_state` on it gives `hp == max_hp == calc_stats(...)[0]` at level 16.
    - `test_boxed_mon_to_cap`: a boxed level-4 mon: level 16, `candies_to_cap(4, 16) == 12`.
    - `test_above_cap_is_kept` (Review Focus 4): a level-20 party mon with cap 16: level 20, 0 candies, recomputed stats equal its stored stats.
    - `test_shedinja_to_cap`: a Shedinja leveled to the cap enters battle with `max_hp == 1`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_stats.py tests/prep -q`; expected: fail.
- [ ] **Step 3: Implement** the species argument, `candies_to_cap` and `level_to_cap`.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/saves tests/engine/test_state.py -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep): level-to-cap spec rebuild and Shedinja's fixed HP`.

### Task 3: State and outcome fields for the new mechanics

Read first: `src/nuzline/engine/STATE.md`; decomp `include/battle/battle_mon.h` (`statusVolatile`, `moveEffectsMask`, `MoveEffectsData`), `include/constants/battle/condition.h:40-82`, `src/battle/battle_lib.c:1834-1840` (`Battler_UnlockMoveChoice`), `:1975-2011` (`BattleContext_Init`), `battle_script.c:5820-5852` (Magnitude roll), `battle_lib.c:3567-3575` (Shed Skin roll).

**Files:**
- Modify: `src/nuzline/engine/state.py` (fields, `LAYOUT`, docstring), `src/nuzline/engine/STATE.md`, `src/nuzline/engine/step.py` (`_VOLATILE`), `src/nuzline/engine/damage.py:129` (Mud Sport reads every side), `src/nuzline/engine/outcome.py`, `src/nuzline/engine/enumerate.py` (laws, relevance all False for now), `src/nuzline/engine/observe.py`, `src/nuzline/search/plan_tree.py` (`_COND_TEXT`, field loop), `src/nuzline/reference/state.py`, `src/nuzline/reference/engine.py` (reset lists, Mud Sport read), `tests/strategies.py` (`any_outcome`, per-side `mud_sport`), `tests/engine/helpers.py`
- Test: `tests/engine/test_state.py`, `tests/engine/test_outcome.py`, `tests/engine/test_observe.py`, `tests/engine/test_damage.py:249`, `tests/reference/test_reference_smoke.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `BattleState` per-side fields of the active battler, after `move_hit`: `foresight int8[2]` (`VOLATILE_CONDITION_FORESIGHT`), `disabled_move int16[2]`, `disable_turns int8[2]` (`MoveEffectsData.disabledMove/disabledTurns`), `locked_move int16[2]` (`moveLockedInto` while `VOLATILE_CONDITION_MOVE_LOCKED`; 0 = not locked), `bide_turns int8[2]` (`VOLATILE_CONDITION_BIDE`, 0-2), `bide_damage int16[2]` (`storedDamage`, ruling 5), `curled int8[2]` (`VOLATILE_CONDITION_DEFENSE_CURL`), `mud_sport int8[2]` (moved from the field scalars; ruling 4). All eight are in `step._VOLATILE` (reset on switch-in and faint). `RefSide` gains the same fields; `RefState.mud_sport` goes.
  - `Outcome.magnitude int8[2]`: the Magnitude level 4-10 of each side's move (`empty_outcome`: 4); `Outcome.shed_skin int8[2]`: Shed Skin's `RandNext() % 10 < 3` (empty 0); both appended after `end_speed_tie`; drawn per the Global Constraints (magnitude: `rand % 100` mapped <5 -> 4, <15 -> 5, <35 -> 6, <65 -> 7, <85 -> 8, <95 -> 9, else 10).
  - `enumerate._LAWS["magnitude"]`: values 4-10, weights `(5, 10, 20, 30, 20, 10, 5)`, denominator 100; `_LAWS["shed_skin"]`: `[0, 1]`, `[7, 3]`, 10.
  - Observation: `_NONZERO_SIDE` adds `foresight`, `curled`, `bide_turns`, `mud_sport`; `_LAYOUT` adds `("disabled_move", _SIDE)` and `("locked_move", _SIDE)`; `_NONZERO_FIELD` drops `mud_sport`. `disable_turns` and `bide_damage` are not observed.
  - `tests/engine/helpers.py::assert_matches_reference(tables, state: BattleState, actions, outcome: Outcome) -> BattleState`: steps both engines (`jax.jit(step)` and `ref_step` via `to_ref`/`from_ref`), asserts every field equal with the field name in the message, returns the JAX state. Every later task's scenario tests use it.

- [ ] **Step 1: Write the failing tests**
  - `test_new_volatiles_reset_on_switch_and_faint`: on `roark_state`, set all eight fields non-zero on side 0; a switch action zeroes them; a KO of the active (faint path) zeroes them; side 1 is untouched by side 0's switch.
  - `test_state_md_accounts_for_every_struct_field` passes with rows mapping `BattleContext.moveLockedInto` -> `locked_move`, `BattleContext.storedDamage` -> `bide_damage`, `MoveEffectsData.disabledMove`/`disabledTurns`, and the `statusVolatile` / `moveEffectsMask` rows naming the new fields.
  - `test_ref_roundtrip_new_fields`: `from_ref(to_ref(s)) == s` field by field with every new field non-zero.
  - `test_magnitude_and_shed_skin_draws`: `empty_outcome()` gives magnitude `[4, 4]`, shed_skin `[0, 0]`; 20000 `sample_outcome` draws (key 0) give magnitude frequencies within `5*sqrt(p(1-p)/N)` of `(.05, .10, .20, .30, .20, .10, .05)` and shed_skin within the same bound of 0.3; `test_existing_draws_unchanged`: the pre-existing fields of `sample_outcome(jax.random.key(0), ...)` on `roark_real_state` with Ember vs Rock Throw equal literals recorded in the test from the code before this task (write the test and record them in Step 1, before changing `outcome.py`).
  - `test_hidden_counters_grouped` gains `disable_turns` and `bide_damage` (equal observations); `test_visible_message_classes` gains `disabled_move`, `locked_move`, `foresight`, `curled`, `bide_turns`, per-side `mud_sport` (different observations).
  - `tests/engine/test_damage.py:249`: `mud_sport` set on side 1 halves the player's Electric damage; on side 0 too.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_state.py tests/engine/test_outcome.py tests/engine/test_observe.py tests/engine/test_damage.py tests/reference -q`; expected: fail.
- [ ] **Step 3: Implement** the fields, resets, outcome fields and laws, observation entries, plan-tree text ("identified", "curled up", "storing energy", "Mud Sport", "<move> disabled", "locked into <move>") and the helper.
- [ ] **Step 4: Run** the full suite `uv run pytest -q` (differential and Stage B tests included) and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): state and outcome fields for Foresight, Disable, move locks, Bide, Magnitude, Shed Skin`.

### Task 4: Foresight and Odor Sleuth, engine and AI

Read first: decomp `effect_script_0113.s`, `subscript_foresight.s`, `battle_lib.c:2600-2622` (0xFE rows skipped for an identified defender), `battle_controller_player.c:2888-2890` (positive evasion ignored), `include/battle/common.h:48`, `battle_lib.c:2681-2730` (`CalcEffectiveness` does not read Foresight), `trainer_ai.c` callers of `BattleSystem_ApplyTypeChart`; `tools/extract_decomp.py` `foresight_immunities`.

**Files:**
- Modify: `src/nuzline/engine/ids.py`, `tools/effect_map.py`, `tools/build_tables.py`, `src/nuzline/data/tables.py`, `src/nuzline/engine/damage.py`, `src/nuzline/engine/effects.py`, `src/nuzline/engine/outcome.py`, `src/nuzline/ai/switching.py`, `src/nuzline/ai/commands.py`, `src/nuzline/ai/policy.py`, `src/nuzline/ai/replacement.py` (docstring), `src/nuzline/reference/engine.py`, `tests/strategies.py` (`SHARED_MOVES += "Foresight"`, draw `foresight`)
- Create: `tests/engine/test_foresight.py`
- Test: `tests/ai/test_switching.py`, `tests/ai/test_policy.py`

**Interfaces:**
- Consumes: Task 3 (`foresight`, `assert_matches_reference`).
- Produces:
  - `EffectId.FORESIGHT = 20` (decomp 113: on-hit indirect effect, sets `foresight[target] = 1` whenever the move lands on a living target; no failure check).
  - `Tables.type_chart_foresight: int8[T, T]`: 1 for the `[attacker, defender]` rows after the 0xFE separator.
  - `damage.effectiveness(tables, move_type, def_types, identified=False) -> int32`: those rows count as 10 when `identified`. Every engine caller passes the defender's `foresight`.
  - `switching.chart_multipliers(tables, move, def_types, gravity, identified=False) -> tuple[int32, int32]`, `switching.type_chart_flags(tables, move, def_types, gravity, identified=False) -> int32`; `apply_type_chart` reads `state.foresight[1 - atk_side]`; `calc_effectiveness_flags` unchanged.
  - Foresight and Odor Sleuth leave `policy._STATE_MOVE_IDS` and `switching._MOVES`.

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_foresight.py`, player Furret with Foresight/Scratch/Mach Punch/Tackle, opponent Sableye (Dark/Ghost, Keen Eye) unless named):
  - `test_foresight_sets_flag`: Foresight lands -> `foresight[1] == 1`; against a target that fainted earlier in the turn nothing is set; a second use leaves it 1.
  - `test_identified_ghost_is_hit`: Scratch deals 0 with `foresight[1] == 0` and the damage of a neutral hit with it set; Mach Punch with it set equals the x2 Fighting-on-Dark damage.
  - `test_identified_ignores_positive_evasion` (Tackle, accuracy 95; `evaStages = 6 - boost`, battle_controller_player.c:2873-2920, no cap at 100): eva +2 identified: `hit_threshold == 95`; eva -2 identified: 157 (`95 * 166 // 100`, unchanged by Foresight); eva -1 identified: 126 (`95 * 133 // 100`); eva +2 unidentified: 71 (`95 * 75 // 100`).
  - `test_foresight_matches_reference`: the cases above through `assert_matches_reference`.
  - `tests/ai/test_policy.py::test_ai_reads_player_foresight`: player active Sableye vs Cranidos: `effectiveness_class(Headbutt)` is `TypeMulti.IMMUNE`; with `foresight[0] = 1` it equals the class against a pure-Dark defender.
  - `tests/ai/test_policy.py::test_player_foresight_is_not_refused`: `roark_state` with the player's slot 3 knowing Foresight: `ai_refusals == []`.
  - `tests/ai/test_switching.py::test_unimplemented_modifiers_are_refused`: the `("player_move", "Foresight")` case becomes `("player_move", "Miracle Eye")`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_foresight.py tests/ai -q`; expected: fail.
- [ ] **Step 3: Implement** the effect, the table, the identified type chart and accuracy (the target switching in this turn is not identified), the AI reads and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/test_differential.py tests/engine tests/ai -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine,ai): Foresight and Odor Sleuth`.

### Task 5: Night Shade, Hidden Power and Magnitude (power and type from the battler)

Read first: decomp `effect_script_0087.s` (`SYSCTL_IGNORE_TYPE_CHECKS`, damage = level), `battle_lib.c:7544-7556` (immunity bit still set), `effect_script_0135.s`, `battle_script.c:6007-6033` (`CalcHiddenPowerParams`), `trainer_ai.c:3208-3220` and `:3767` (`TrainerAI_MoveType` for `moveHit`), `effect_script_0126.s`, `battle_script.c:5820-5852`, `subscript_magnitude.s`.

**Files:**
- Modify: `src/nuzline/engine/ids.py`, `tools/effect_map.py`, `src/nuzline/engine/damage.py`, `src/nuzline/engine/effects.py`, `src/nuzline/engine/step.py` (Magnitude event), `src/nuzline/engine/enumerate.py`, `src/nuzline/engine/observe.py`, `src/nuzline/ai/switching.py`, `src/nuzline/reference/engine.py`, `tests/strategies.py` (`SHARED_MOVES += ("Night Shade", "Seismic Toss", "Hidden Power", "Magnitude")`, `LAW_MOVES += ("Magnitude",)`)
- Create: `tests/engine/test_power_moves.py`
- Test: `tests/engine/test_enumerate.py`, `tests/engine/test_observe.py`, `tests/ai/test_switching.py`

**Interfaces:**
- Consumes: Task 3 (`Outcome.magnitude`, `assert_matches_reference`), Task 4 (`effectiveness(..., identified)`).
- Produces:
  - `EffectId.LEVEL_DAMAGE = 21`, `HIDDEN_POWER = 22`, `MAGNITUDE = 23`.
  - `damage.hidden_power(ivs) -> tuple[int32, int32]`: (type, power) from `ids.Stat`-order IVs (decomp bit order HP, Atk, Def, Spe, SpA, SpD).
  - `damage.move_type_of(tables, state, side, move) -> int32`: Hidden Power's IV type for `side`'s active, else `tables.move_type[move]`; used for every type read of a used move (STAB, chart, sports, pinch abilities, the Fire thaw).
  - `damage.MAGNITUDE_POWER`: int32[7] = `(10, 30, 50, 70, 90, 110, 150)` for levels 4-10.
  - `damage_for_roll(tables, state, atk_side, move, crit, roll, power_mul=10, magnitude=4) -> int32`.
  - `damage.fixed_damage(tables, state, atk_side, move) -> tuple[bool_, int32]`: `LEVEL_DAMAGE` deals the attacker's level, no crit, roll, STAB or multiplier, 0 on a type immunity (Task 8 adds Bide).
  - Event `magnitude` per side (the level shown, 0 when Magnitude did not run) in `step_events` and the observation.
  - `switching.calc_effectiveness_flags(tables, move, def_types, gravity, move_type=None) -> int32`; the passes use `move_type_of(tables, state, 0, move_hit[1])`. Hidden Power stays refused on side 1 only.
  - enumerate: `magnitude[s]` relevant when `s` uses a `MAGNITUDE` move; `crit`/`roll` not relevant for `LEVEL_DAMAGE`.

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_power_moves.py`)
  - `test_hidden_power_params`: IVs all 31 -> (Dark, 70); all 30 -> (Fighting, 70); all 0 -> (Fighting, 30); `(30, 30, 30, 30, 30, 31)` (Spe odd) -> Flying; `(30, 30, 30, 31, 30, 30)` (SpA odd) -> Ground; and for all 64 parity patterns of each bit, equality with the decomp formula transcribed in the test.
  - `test_hidden_power_damage`: Monferno with all-31 IVs vs Geodude: every (crit, roll) damage equals `damage_for_roll` on tables whose Hidden Power row is set to (Dark, 70).
  - `test_night_shade_deals_level`: Spinarak L16 vs Geodude: HP drops by exactly 16 for every crit and roll; vs Bidoof (Normal): 0; Seismic Toss vs Sableye: 0, and 16 with `foresight[1] = 1`.
  - `test_magnitude_power_and_message`: for each level 4-10 the damage equals the tables with Magnitude's power set to `MAGNITUDE_POWER[level - 4]`; the observation's `magnitude` is that level; vs Staravia: 0.
  - `test_power_moves_match_reference`: the cases above through `assert_matches_reference`.
  - `tests/engine/test_enumerate.py`: Magnitude's marginal is `(5, 10, 20, 30, 20, 10, 5)/100` exactly; Night Shade has no crit or roll factor.
  - `tests/ai/test_switching.py::test_hidden_power_move_hit_type` (Review Focus 5): player IVs `(30, 31, 31, 30, 31, 30)` (Water Hidden Power) with `move_hit[1] = Hidden Power`: `switching_refusals == []` and the switch layout equals the one with `move_hit[1] = Water Gun`, not Tackle's.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_power_moves.py tests/engine/test_enumerate.py tests/ai/test_switching.py -q`; expected: fail.
- [ ] **Step 3: Implement** the three effects, the type/power helpers, the event, the laws, the AI type read and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/test_differential.py tests/engine tests/ai -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): Night Shade, Hidden Power and Magnitude`.

### Task 6: Confusion (secondary confusion), Mud Sport and Bug Bite

Read first: decomp `effect_script_0076.s`, `subscript_confuse.s`, `battle_lib.c:1519-1575` (`TriggerSecondaryEffect`: an `ON_HIT` effect has no HP check), `effect_script_0201.s` (fails when the user's flag is set), `battle_lib.c:6811-6816`, `effect_script_0224.s`, `subscript_pluck.s`, `battle_script.c:8116-8130` (`TryPluck`), `battle_lib.c:5539-5575` (`BattleSystem_PluckBerry`: Oran `PLUCK_EFFECT_HP_RESTORE`, Sitrus `PLUCK_EFFECT_HP_PCT_RESTORE`, only when the user is not at full HP), `battle_script.c:8972` (`RemoveItem`).

**Files:**
- Modify: `src/nuzline/engine/ids.py`, `tools/effect_map.py`, `src/nuzline/engine/effects.py`, `src/nuzline/engine/hooks.py`, `src/nuzline/engine/outcome.py` (duration kind), `src/nuzline/engine/enumerate.py`, `src/nuzline/reference/engine.py`, `tests/strategies.py` (`SHARED_MOVES += ("Confusion", "Mud Sport", "Bug Bite")`, `LAW_MOVES += ("Confusion",)`)
- Create: `tests/engine/test_side_effects.py`
- Test: `tests/engine/test_enumerate.py`

**Interfaces:**
- Consumes: Task 3 (`mud_sport[2]`, helper).
- Produces:
  - `EffectId.SECONDARY_CONFUSE = 24` (76: `secondary` roll at the move's chance, then `subscript_confuse.s`: fails on a confused or fainted target, else `confusion = outcome.duration` 2-5), in `enumerate._PROC_EFFECTS`; the foe's `confusion_hit` relevance includes it.
  - `EffectId.MUD_SPORT = 25` (201: `mud_sport[user side] = 1`, fails when set).
  - `EffectId.EAT_BERRY = 26` (224) and `hooks.pluck_berry(tables, state, side, cond) -> BattleState`: `side` eats the foe's Oran/Sitrus Berry (heal per `PluckBerry` with the item's param, capped at max HP, nothing at full HP); the foe's `item` becomes 0 and its `item_consumed` the berry; applies after a landed, effective hit even when the foe fainted.

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_side_effects.py`)
  - `test_confusion_confuses_on_proc`: Venonat's Confusion vs Geodude with `secondary=1, duration=3` -> `confusion[1] == 3`; `secondary=0` -> 0; an already confused target keeps its counter; a KO'd target gets none.
  - `test_mud_sport_halves_electric_until_switch`: Geodude's Mud Sport sets `mud_sport[0] = 1`; a second use leaves the state unchanged except PP; Shinx's Spark `damage_table` against Monferno halves while either side has the flag; after the user switches out the flag is gone and Spark is full.
  - `test_bug_bite_eats_berry`: Kricketot vs Geodude holding Oran Berry at 10 HP below max: user +10 (capped), target item 0, `item_consumed[1, 0] == Oran Berry`; Sitrus heals `max_hp * 25 // 100`; at full HP nothing is healed and the berry is still gone; a KO'ing Bug Bite still eats it; a target with no item is untouched.
  - `test_side_effects_match_reference`: the cases above through `assert_matches_reference`.
  - `tests/engine/test_enumerate.py`: Confusion's secondary marginal is 10/100 and its duration 2-5 uniform.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_side_effects.py tests/engine/test_enumerate.py -q`; expected: fail.
- [ ] **Step 3: Implement** the three effects, the pluck hook, the laws and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/test_differential.py tests/engine -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): Confusion-family secondary confusion, Mud Sport and Bug Bite`.

### Task 7: Disable

Read first: decomp `effect_script_0086.s`, `subscript_disable_start.s`, `battle_script.c:4581-4600` (`TryDisable`: target's `movePrevByBattler` known, PP left, none disabled yet; `RandNext() % 4 + 3`), `battle_lib.c:2255-2275` (`CHECK_INVALID_DISABLED`), `battle_controller_player.c:2543-2555` (disabled at move time; before the PP decrement), `:1565-1585` (end-of-turn countdown: decrement while non-zero, clear at 0; cleared at once when the move is no longer known), `subscript_move_is_disabled.s`.

**Files:**
- Modify: `src/nuzline/engine/ids.py`, `tools/effect_map.py`, `src/nuzline/engine/effects.py`, `src/nuzline/engine/actions.py`, `src/nuzline/engine/step.py`, `src/nuzline/engine/outcome.py` (duration 3-6), `src/nuzline/engine/enumerate.py` (`DURATION_TRAP` for Disable), `src/nuzline/engine/observe.py` (`ACT_DISABLED`), `src/nuzline/search/plan_tree.py` (text), `src/nuzline/reference/engine.py`, `tests/strategies.py` (`SHARED_MOVES += ("Disable",)`, `LAW_MOVES += ("Disable",)`, draw `disabled_move` among the active's moves or 0 and `disable_turns` 0-6)
- Create: `tests/engine/test_disable.py`
- Test: `tests/engine/test_enumerate.py`, `tests/engine/test_observe.py`, `tests/ai/test_policy.py`

**Interfaces:**
- Consumes: Task 3 (`disabled_move`, `disable_turns`, `last_move`, helper).
- Produces:
  - `EffectId.DISABLE = 27`.
  - `actions.usable_moves` also excludes the slot holding `disabled_move` (so the AI's base scores and Struggle follow, `commands.py:295`, `policy.py:554`).
  - `step.ACT_DISABLED = 5`: the chosen move was disabled before it ran (no PP spent).
  - `step._disable_check(tables, state, side) -> BattleState` in `END_OF_TURN_ORDER` between `bind` and `taunt` (`MON_COND_CHECK_STATE_DISABLE`).

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_disable.py`, Psyduck with Disable vs Geodude whose `last_move[1]` is Rock Throw unless named)
  - `test_disable_disables_last_move`: `duration=4` -> `disabled_move[1] == Rock Throw`, `disable_turns[1] == 4`; Rock Throw's slot is not in `legal_opponent_actions`.
  - `test_disable_fails` (parametrized): no last move; last move Struggle; last move at 0 PP; already disabled; the move missed (`hit=0`).
  - `test_disable_countdown`: with duration 3, `disable_turns[1]` goes 3 -> 2 at the end of the Disable turn itself, then 2 -> 1 and 1 -> 0 at the end of the 1st and 2nd turns after it, with `disabled_move[1]` still set; at the end of the 3rd turn after it (the `duration + 1`-th end of turn counting the Disable turn) `disabled_move[1] == 0` and the observation shows the end; a disabled move the target no longer knows is cleared at the next end of turn (its counter is zeroed, then the move cleared in the same check).
  - `test_chosen_move_disabled_mid_turn`: the faster Psyduck disables the move Geodude chose this turn: Geodude's action is `ACT_DISABLED`, its PP unchanged, the observation shows it.
  - `test_ai_avoids_disabled_slot`: `ai_action_distribution` gives the disabled slot mass 0.
  - `test_ai_struggles_when_all_disabled` (Review Focus 3): Geodude knowing only Rock Throw, disabled: the AI's whole mass is on action 0, `selected_move` is Struggle, and the stepped turn uses Struggle.
  - `test_disable_matches_reference`: the cases above through `assert_matches_reference`.
  - `tests/engine/test_enumerate.py`: Disable's duration marginal is uniform on 3-6.
  - `tests/engine/test_observe.py::test_legality_constant_on_equal_obs` gains a Disable case that splits on `disabled_move` and groups across `disable_turns`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_disable.py tests/engine/test_enumerate.py tests/engine/test_observe.py tests/ai/test_policy.py -q`; expected: fail.
- [ ] **Step 3: Implement** the effect, legality, the move-time check, the countdown, the event and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/test_differential.py tests/engine tests/ai tests/search -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine,ai): Disable`.

### Task 8: Move locks: Bide, Rollout and Ice Ball, Defense Curl

Read first: decomp `effect_script_0026.s`, `subscript_bide_start.s`, `subscript_bide_end.s`, `subscript_bide_no_target.s`, `battle_controller_player.c:2694-2723` (`CHECK_STATUS_STATE_BIDE`: unleash `2 * storedDamage` at the last hitter, else a random opponent, else no target), `:3407` (accumulation), `:2350-2358` (`RANGE_USER` skips the type chart), `:335` (a locked battler picks no command and spends no PP), `:1885-1888` (it uses `moveLockedInto`), `battle_lib.c:1828-1840` (lock/unlock; unlock clears Bide and `rolloutCount`), `:2240-2243` (sleep unlocks at turn start), `:6224-6235`, `effect_script_0117.s`, `battle_script.c:5661-5690` (`CalcRolloutPower`), `effect_script_0156.s`; the scripts that unlock: `subscript_{missed, flinched, freeze, fully_paralyzed, hurt_self_in_confusion, move_fail_taunted, move_is_disabled, no_target, fall_asleep}.s`.

**Files:**
- Modify: `src/nuzline/engine/ids.py` (EffectId docstring: `STAT_SELF` param 2), `tools/effect_map.py` (26, 117; 156 -> `(DEF, 1, 1)`), `src/nuzline/engine/actions.py`, `src/nuzline/engine/order.py`, `src/nuzline/engine/damage.py`, `src/nuzline/engine/effects.py`, `src/nuzline/engine/step.py`, `src/nuzline/engine/enumerate.py`, `src/nuzline/reference/engine.py`, `tests/strategies.py` (`SHARED_MOVES += ("Bide", "Rollout", "Ice Ball", "Defense Curl")`; draw `curled`)
- Create: `tests/engine/test_move_locks.py`
- Test: `tests/engine/test_observe.py`, `tests/search/test_stage_b.py`

**Interfaces:**
- Consumes: Task 3 (`locked_move`, `bide_turns`, `bide_damage`, `curled`, `rollout_count`, helper), Task 5 (`fixed_damage`).
- Produces:
  - `EffectId.BIDE = 28`, `EffectId.ROLLOUT = 29`; `STAT_SELF` with `move_param[m, 2] == 1` also sets `curled` (Defense Curl).
  - `actions.locked_slot(state, side) -> int32`: the slot holding `locked_move[side]`, -1 when not locked. `legal_actions` of a locked side is that slot alone (no switches).
  - `order.selected_move` returns `(locked_move, False)` for a locked side.
  - `damage.fixed_damage` also returns Bide's unleash (`2 * bide_damage`, no type chart, no crit or roll).
  - `step._unlock(state, side, cond) -> BattleState`: `locked_move`, `bide_turns`, `bide_damage`, `rollout_count` to 0 (`Battler_UnlockMoveChoice`), called on every unlocking path listed above.
  - Rollout power: `30 * 2 ** (uses - 1)` for uses 1-5, x2 when `curled`; the lock ends after the fifth use or on any unlocking path.

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_move_locks.py`)
  - `test_bide_stores_and_unleashes`: Kricketot's Bide (priority +1, so Kricketot moves first every turn) vs Geodude's Tackle: turn 1 `bide_turns == 2`, locked, PP -1, `bide_damage` equals the HP turn 1's Tackle took (it lands after Bide started); turn 2 `bide_damage` equals the total HP Kricketot lost since Bide started (both Tackles); turn 3 Geodude loses exactly `2 * bide_damage` (Rock/Ground does not resist it), the lock and Bide clear, PP unchanged since turn 1.
  - `test_bide_fails_without_damage`: no damage taken -> the unleash fails, no HP change, lock clear.
  - `test_bide_hits_the_replacement`: the foe switches on turn 3: the incoming Pokemon takes the unleash.
  - `test_locked_side_actions` (Review Focus 2): while locked, `legal_actions(tables, s, 0)` is one-hot on the locked slot and has no switch.
  - `test_disruption_unlocks` (parametrized: full paralysis, flinch, confusion self-hit, falling asleep, a Rollout miss): the lock, `bide_turns` and `rollout_count` are 0 afterwards.
  - `test_rollout_power_and_curl`: Bidoof's five consecutive Rollout hits deal the damage of power 30, 60, 120, 240, 480 (tables with Rollout's power set to each); after Defense Curl (`curled[0] == 1`, Def +1) they are 60-960; the lock ends after the fifth; PP is spent once.
  - `test_locks_match_reference`: the cases above through `assert_matches_reference`.
  - `tests/engine/test_observe.py::test_legality_constant_on_equal_obs` gains a locked case; `tests/search/test_stage_b.py::test_locked_player_solves`: a 1v1 depth-1 solve from a state where the player is locked into Bide returns one ranked action and no `HiddenLegality`.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_move_locks.py tests/engine/test_observe.py tests/search/test_stage_b.py -q`; expected: fail.
- [ ] **Step 3: Implement** the lock, Bide, Rollout, Defense Curl's flag and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/test_differential.py tests/engine tests/search -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): move locks, Bide and Rollout`.

### Task 9: Insomnia, Cloud Nine, Compoundeyes and Shed Skin

Read first: decomp `subscript_fall_asleep.s:7-27` and `_202`-`_221` (a direct sleep move: ignorable Insomnia, message tags the ability), `battle_lib.c:4378-4450` (`RecoverStatusByAbility`), its callers `battle_controller_player.c:3965-3967` (move end, attacker then defender) and `battle_lib.c:4135-4145` (switch-in), `subscript_ability_forbids_status.s:6`; `include/battle/common.h:63-66` and every `ABILITY_CLOUD_NINE` site (`battle_controller.c:2072`, `battle_script.c:9200`); `battle_controller_player.c:2922-2924` (Compoundeyes x130/100 after the stage scaling); `battle_lib.c:3567-3590` (Shed Skin), `battle_controller_player.c:1355` (`MON_COND_CHECK_STATE_ABILITY`, first of the end-of-turn checks), `subscript_ability_restore_status.s:8`.

**Files:**
- Modify: `src/nuzline/engine/ids.py`, `tools/effect_map.py`, `src/nuzline/engine/effects.py` (`_inflict`, docstring), `src/nuzline/engine/hooks.py`, `src/nuzline/engine/outcome.py`, `src/nuzline/engine/step.py`, `src/nuzline/engine/enumerate.py`, `src/nuzline/reference/engine.py`, `tests/strategies.py` (`M1_SPECIES += ("Furret", "Staravia", "Spinarak", "Psyduck", "Venonat", "Kricketot")`, ability slots drawn only where `tables.species_abilities[species, slot] != UNSUPPORTED`; `LAW_ABILITIES += (SHED_SKIN, COMPOUNDEYES)`)
- Create: `tests/engine/test_m4_abilities.py`
- Test: `tests/engine/test_enumerate.py`

**Interfaces:**
- Consumes: Task 3 (`Outcome.shed_skin`, helper).
- Produces:
  - `AbilityId.INSOMNIA = 19`, `CLOUD_NINE = 20`, `COMPOUNDEYES = 21`, `SHED_SKIN = 22` (decomp names "Insomnia", "Cloud Nine", "Compoundeyes", "Shed Skin").
  - `hooks.SLEEP_BLOCKED, hooks.STATUS_CURE = "sleep_blocked", "status_cure"`, new phase constants beside `SWITCH_IN` and `CONTACT_STATUS`.
  - `hooks.REVEALS_ON: dict[int, frozenset[str]]`: each ability maps to the set of phases at which it reveals (existing entries become `frozenset({SWITCH_IN})` / `frozenset({CONTACT_STATUS})`); `reveal_ability(state, side, slot, phase, cond=True)` keeps its signature and selects the abilities with `phase in phases`. New entries: `INSOMNIA: frozenset({SLEEP_BLOCKED, STATUS_CURE})`, `SHED_SKIN: frozenset({STATUS_CURE})`.
  - Insomnia blocks sleep in `_inflict` through `target_ignorable` (Mold Breaker pierces it), and `_inflict` calls `reveal_ability(..., SLEEP_BLOCKED, cond=blocked)` for the target.
  - `hooks.recover_status_by_ability(state, side, cond=True) -> BattleState`: an asleep Insomnia holder with HP left wakes (counter 0) and calls `reveal_ability(..., STATUS_CURE, cond=woke)`; run in `on_switch_in` and at each action's end for the user, then the target.
  - `outcome` hit rate: `rate * 130 // 100` for a Compoundeyes user, right after the stage scaling.
  - `step._turn_end_ability(tables, state, side) -> BattleState`, first in `END_OF_TURN_ORDER`: Shed Skin with a status, HP left and `outcome.shed_skin[side]` cures it (counter 0) and calls `reveal_ability(..., STATUS_CURE, cond=cured)`.
  - enumerate: `shed_skin[d]` relevant when side `d`'s active or incoming Pokemon has Shed Skin (`_holds`).

- [ ] **Step 1: Write the failing tests** (`tests/engine/test_m4_abilities.py`)
  - `test_insomnia_blocks_sleep`: Hypnosis or Spore on Spinarak (Insomnia): no sleep and `ai_ability_known` set for it; from a Mold Breaker attacker (Cranidos): asleep.
  - `test_insomnia_cures_on_entry_and_move_end`: an asleep Insomnia Pokemon switched in wakes; an asleep Insomnia active wakes at the end of its own action.
  - `test_cloud_nine_is_inert`: hypothesis (100 examples, weather none): stepping with the active's ability set to `CLOUD_NINE` equals stepping with `NONE` field by field.
  - `test_cloud_nine_sites_are_weather_only`: scanning the decomp's `src/battle` for `ABILITY_CLOUD_NINE`, `NO_CLOUD_NINE` and `NO_WEATHER` finds only sites listed in the test, each one a weather condition or the move animation (`battle_controller.c:2072`).
  - `test_compoundeyes_accuracy`: Venonat's Tackle at neutral stages: threshold 123; at accuracy -1: 92.
  - `test_shed_skin_cures_before_poison`: poisoned Kricketot with `shed_skin=1`: status none, no poison damage that turn, ability revealed; with 0: poison damage as before; a fainted holder is not cured.
  - `test_abilities_match_reference`: the cases above through `assert_matches_reference`.
  - `tests/engine/test_enumerate.py`: Shed Skin's marginal is 3/10.
- [ ] **Step 2: Run** `uv run pytest tests/engine/test_m4_abilities.py tests/engine/test_enumerate.py -q`; expected: fail.
- [ ] **Step 3: Implement** the four abilities, the hooks, the law and the reference mirror; regenerate the tables.
- [ ] **Step 4: Run** the full `uv run pytest -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(engine): Insomnia, Cloud Nine, Compoundeyes and Shed Skin`.

### Task 10: The candidate pool and `nuzline coverage --save` (`prep/pool.py`)

Read first: `src/nuzline/engine/coverage.py`, `src/nuzline/cli.py` (`_fight`, `_coverage`), `src/nuzline/saves/save.py` (`SaveFile.boxes`, `bag["tms_hms"]`); rulings 2 and 7.

**Files:**
- Create: `src/nuzline/prep/pool.py`
- Modify: `src/nuzline/ai/policy.py`, `src/nuzline/ai/switching.py` (side-0 refusal map exported), `src/nuzline/cli.py` (module docstring, `--cap`, pool section), `tests/saves/synthetic.py` (`party_record` gains `ability_slot`; new `box_record`)
- Test: `tests/prep/test_pool.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 (`Learnsets`), Task 2 (`level_to_cap`, `candies_to_cap`), `coverage.CoverageReport`, `IMPLEMENTED_EFFECTS`, `IMPLEMENTED_ABILITIES`, `IMPLEMENTED_ITEMS`.
- Produces:
  - `tests/saves/synthetic.py`: `party_record(tables, names, species, level, moves, *, hp=None, status=0, egg=False, ability_slot=0, **kw)` writes `ability=tables.species_abilities_decomp[sp][ability_slot]` (today it is hard-wired to slot 0 and `**kw` cannot override it, since `ability` is an explicit `mon_record` argument); `box_record(tables, names, species, level, moves, *, ability_slot=0, **kw)` builds the same record without the party block. 4c's `test_build_problem_errors` uses `box_record`.
  - `ai.policy.player_move_refusal(move: int) -> str | None` and `ai.policy.player_ability_refusal(ability: int) -> str | None` (decomp ids): the reason the AI refuses a player Pokemon knowing that move or having that ability, from `_STATE_MOVE_IDS`, `_DEFENDER_ABILITIES` and switching's side-0 lists.
  - `@dataclass(frozen=True) class Candidate`: `location: tuple[str, int, int]`, `label: str` ("party slot 4 SPINARAK", "box 1 slot 3 KRICKETOT"; boxes 1-based as `nuzline save show`), `spec: PokemonSpec` (at the cap), `candies: int`, `relearner: tuple[int, ...]`, `tms: tuple[tuple[int, int], ...]` ((item, move)).
  - `pool_candidates(save: SaveFile, tables: Tables, names: Names, learnsets: Learnsets, cap: int) -> tuple[list[Candidate], list[str]]`: party then boxes in order; eggs and records `to_spec` refuses are left out with one line each in the second list.
  - `check_pool(tables: Tables, names: Names, candidates: Sequence[Candidate]) -> CoverageReport`: per candidate its ability, held item, current moves and every option move (effect support and `player_move_refusal`), each missing line as `"<label>: <move> (<current|relearner|TM10>, move effect UNSUPPORTED)"` or the AI's reason.
  - CLI: `nuzline coverage --save S (--trainer N | --trainer-id I) [--cap N]` prints today's report, then `Pool at level cap <cap> (<n> Pokemon):` and `_print_report(check_pool(...))`, then one line per left-out record. `--cap` without `--save` is an error (exit 1); `--cap` defaults to the highest party level. Exit code follows the party fight (ruling 7).

- [ ] **Step 1: Write the failing tests**
  - `tests/prep/test_pool.py` (synthetic saves: party Furret L16 with `ability_slot=1` (Keen Eye; slot 0 is Run Away, `AbilityId.UNSUPPORTED`) and Geodude L16 (slot 0, Rock Head), boxes Kricketot L5 (slot 0, Shed Skin) and Bidoof L4 (slot 0, Simple), bag `tms_hms` = TM10):
    - `test_candidates`: 4 candidates in party-then-box order, all level 16, candies `(0, 0, 11, 12)`; Kricketot's `relearner` is Bug Bite (it already knows Growl and Bide) and `tms == ()`; Furret's `tms == ((TM10, Hidden Power),)`; an egg in the party is left out with a line.
    - `test_check_pool_full`: with every mechanic of this plan in place the report has `level == Support.EXACT` and no missing lines.
    - `test_check_pool_names_unsupported` (Review Focus 1): boxes Geodude L6 (Tackle) and Monferno L14 (Metronome, Scratch), cap 20: the missing lines include `box 1 slot 0 GEODUDE: Selfdestruct (relearner, move effect UNSUPPORTED)` and `box 1 slot 1 MONFERNO: Metronome (current, move effect UNSUPPORTED)`; `level == Support.NONE`.
    - `test_player_move_refusal`: Miracle Eye -> a reason; Foresight and Hidden Power -> None.
  - `tests/test_cli.py`:
    - `test_coverage_save_reports_pool`: synthetic save: exit 0, output contains `Pool at level cap 16 (4 Pokemon):` and `Support: EXACT` twice.
    - `test_coverage_pool_does_not_change_exit_code`: a box Pokemon with an unsupported move: the pool section names it, exit code 0.
    - `test_cap_needs_save`: `--party P --cap 16` -> exit 1.
- [ ] **Step 2: Run** `uv run pytest tests/prep tests/test_cli.py -q`; expected: fail.
- [ ] **Step 3: Implement** the refusal lookups, `pool_candidates`, `check_pool` and the CLI section.
- [ ] **Step 4: Run** the tests, `uv run pytest tests/ai tests/test_cli.py tests/prep -q` and ruff; expected: pass.
- [ ] **Step 5: Commit** `feat(prep,cli): candidate pool coverage for nuzline coverage --save`.

### Task 11: Proof on the user's save

**Files:**
- Create: `tests/test_user_save.py`
- Modify: `README.md` (the `NUZLINE_USER_SAVE` variable next to `NUZLINE_ORACLE_DIR`)

**Interfaces:**
- Consumes: Task 10 (`pool_candidates`, `check_pool`, the CLI), Task 1 (`load_learnsets`).
- Produces: nothing.

- [ ] **Step 1: Write the tests** (module skipped unless `NUZLINE_USER_SAVE` is set: `pytestmark = pytest.mark.skipif(not os.environ.get("NUZLINE_USER_SAVE"), reason="NUZLINE_USER_SAVE not set")`)
  - `test_roark_coverage_is_full(capsys)`: `main(["coverage", "--save", path, "--trainer", "roark"])` returns 0; the output contains `Support: EXACT` twice, `Pool at level cap 16 (11 Pokemon):`, `Missing: none` twice, and no `UNSUPPORTED`.
  - `test_user_pool`: 11 candidates, all level 16; candies party 0 x6, Bidoof 12, Venonat 6 and 6, Kricketot 11, Geodude 10; the union of current moves (`spec.moves`) contains Foresight, Night Shade, Disable, Bide and Mud Sport (all already known, so ruling 2 keeps them out of the options); the union of option moves (`relearner` and the TM moves) contains Confusion, Rollout, Magnitude, Bug Bite and Hidden Power; every candidate but Kricketot has the TM10 option; `check_pool(...).missing == []`.
- [ ] **Step 2: Run** `NUZLINE_USER_SAVE="<user save>" uv run pytest tests/test_user_save.py -v`; expected: 2 passed. Without the variable: 2 skipped.
- [ ] **Step 3: Run** `uv run nuzline coverage --save "$NUZLINE_USER_SAVE" --trainer roark`; expected: exit 0, both sections `Support: EXACT`, `Missing: none`.
- [ ] **Step 4: Run** the full `uv run pytest -q` and `uv run ruff check src tests tools`; expected: pass.
- [ ] **Step 5: Commit** `test: full Roark coverage on the user's save (NUZLINE_USER_SAVE)`.
