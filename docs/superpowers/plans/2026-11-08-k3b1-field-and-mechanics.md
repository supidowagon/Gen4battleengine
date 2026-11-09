# K3b1: Kaizo field pass and the draw-free mechanics, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, exactly and in both engines, the end-of-turn field-condition pass,
permanent sandstorm, screens, obedience, and every Kaizo-Roark/pool move, ability and held item
that needs no new random draw. Also list holdable bag items in the pool check.

**Architecture:** Each mechanic follows the existing path. Steps:
1. an `ids.py` enum member;
2. a `tools/effect_map.py` mapping;
3. the JAX code (`engine/*.py`);
4. the same rule in `reference/engine.py`;
5. the coverage registry, which picks up the new enum member automatically;
6. tests: a decomp-derived unit test, a fixed-state JAX-vs-reference diff, and the Hypothesis
   differential test (`tests/test_differential.py`) once the strategies can draw the mechanic.

The engine survey in `docs/kaizo/k3b-survey.md` (committed with this plan) is the codebase map:
every file:line and every decomp fact below comes from it. Implementers read the sections the
task names.

**Tech Stack:** Python 3.12 (uv), JAX, pytest, Hypothesis, node (the damage oracle).

**Spec:** `docs/superpowers/specs/2026-11-07-kaizo-k3-design.md` §4, §8 (K3b). This plan is
K3b's first half. K3b2 covers the draw mechanics: Thunder Fang, Protect, Present, Attract and Cute
Charm, Quick Claw, Camouflage, Roar, and Nature Power with terrain. It is planned when K3b1 lands.

## Global Constraints

- Behaviour comes from the decomp, cited `file:line`; Kaizo data comes from the Kaizo tables.
  Never approximate. A case the code cannot model is refused by coverage with a named line.
- Every mechanic lands in BOTH engines in the same task, with a fixed-state
  `tests/engine/helpers.py::assert_matches_reference` test. Mechanics the strategies can draw are
  also exercised by `tests/test_differential.py`.
- Adding an enum member marks it EXACT in coverage at once (coverage.py:103-114). Add the member
  in the same commit as its code.
- New `Outcome` fields are out of scope for K3b1. If a task finds it needs one, stop and report.
- Vanilla exact results are unchanged: vanilla `coverage --party tests/fixtures/sample_party.toml
  --trainer roark` output stays byte-identical, and `tests/engine/test_outcome.py::test_existing_draws_unchanged`
  stays green. Any vanilla table change bumps `MANIFEST_VERSION` (`src/nuzline/data/tables.py:26`)
  and rebuilds both games (`uv run --group rom python -m tools.build_tables` and
  `... --game kaizo`).
- Hard-coded decomp ids are registered in `src/nuzline/data/game_constants.py` (survey §2.1
  item 8).
- `engine/STATE.md` rows are updated when state fields are added
  (`tests/engine/test_state.py::test_state_md_accounts_for_every_struct_field`).
- Never commit ROM bytes, saves or traces. Never touch a `backup` folder. Every commit ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push or merge.
- Run focused tests per task. The controller runs the full suite (about 55 minutes), one run at
  a time.

## Review Focus

1. Two Pokémon fainting to sandstorm in the same field pass: both chips apply first and the faint
   check runs once, so the last two Pokémon down give a draw, which is a loss (survey §4.3).
   Task 3.
2. Cloud Nine on the field suppresses sandstorm chip, the Rock Special Defense boost, Sand Veil
   evasion and Swift Swim, with vanilla no-weather results unchanged. Task 4.
3. Focus Sash at full HP survives each of: a crit, every hit of a multi-hit move (a full-HP hit sets the per-action
   `focusItemActivated` flag, so later lethal hits are also capped), a confusion self-hit, and Self-Destruct targeting it. It is consumed exactly once. Task 11.
4. A pool or bag item the engine does not model is a named pool row, and prep warns about it
   instead of dropping it silently. Task 2.
5. Self-Destruct faints its user even on a miss or into a Ghost type. It fails only when there is
   no target. Task 8.

---

### Task 1: Obedience is modelled

**Files:** Test: `tests/engine/test_obedience.py` (new). Modify: `data/kaizo/rom_diff.toml`
(the `BattleControllerPlayer_CheckObedience` entry), `docs/kaizo/rom_diff.md` (regenerated),
`tests/kaizo/test_rom_classification.py` (if it pins the unknown count).

- [ ] **Step 1: Write the test** `test_player_mon_always_obeys`. Build a fight with a level-16
  player Pokémon, no badge or trainer-ID state, and the action "use move slot 0" under
  `empty_outcome()`. Assert the move executes: the target's HP drops by the oracle damage for
  roll index 0. Also assert `BattleState` has no field that names obedience, badges or the
  original trainer (inspect `BattleState._fields`).
- [ ] **Step 2: Run it.** It passes, because the engine has no obedience check. This is a
  pinning test; record that in the report.
- [ ] **Step 3: Classify.** Flip the entry to `modelled`. Its note is `decoded: ... ; the engine
  has no obedience check, pinned by tests/engine/test_obedience.py:<line>`. Regenerate the
  report (`uv run --group rom python -m tools.rom.diff ...`, as in K3a) and rebuild the Kaizo
  tables.
- [ ] **Step 4: Run** `uv run --group rom pytest tests/engine/test_obedience.py tests/kaizo -q`:
  PASS. Coverage for Roark now prints `Kaizo ROM: 1 code changes unclassified`.
- [ ] **Step 5: Commit** `feat(kaizo): obedience modelled (always obeys)`.

### Task 2: The raw hold effect, and bag items in the pool

**Files:** Modify: `tools/build_tables.py` (new array `item_hold_effect`, int16, the decomp hold
effect per item id), `src/nuzline/data/tables.py` (`Tables.item_hold_effect`; `MANIFEST_VERSION`
bump), `src/nuzline/prep/pool.py` (`check_pool` gains bag rows), `src/nuzline/prep/problem.py`
(`bag_resources` warns instead of silently dropping), `src/nuzline/cli.py` (`_print_pool` passes
the save's bag). Test: `tests/prep/test_pool.py`, `tests/prep/test_problem.py`,
`tests/data/test_tables.py` (or the existing tables test).

**Interfaces:** `check_pool(tables, names, candidates, trainer=None, bag=())`, where `bag` is
the save's `[(item, qty)]` across `problem.HELD_POCKETS`. It adds one row per distinct item with
`item_hold_effect[item] != 0`, labelled `f"bag: {item name}"`, kind `"held item"`, graded by
`IMPLEMENTED_ITEMS`. Items with no hold effect produce no row.

- [ ] **Step 1: Write the tests.**
  - `test_item_hold_effect_matches_decomp`: Focus Sash 275 → 103, Quick Claw 217 → 52, Chilan
    200 → 35, Honey 94 → 0. Both games; survey §1.5.
  - `test_pool_lists_holdable_bag_items`: a bag with White Herb, Pecha Berry, Oran Berry and
    Honey gives exactly the White Herb and Pecha rows as missing, Oran graded EXACT, and no
    Honey row.
  - `test_bag_resources_warns_on_unsupported_item`: the dropped item appears in the returned
    warnings.
- [ ] **Step 2: Run them; see them fail.**
- [ ] **Step 3: Implement.** Rebuild both games' tables.
- [ ] **Step 4: Run** `uv run pytest tests/prep tests/data -q` and the vanilla coverage text
  check: PASS, no diff. Vanilla `coverage --save` output gains bag rows; that is intended (see
  the ruling at the end).
- [ ] **Step 5: Commit** `feat(prep): holdable bag items are pool rows; no silent drop`.

### Task 3: The field-condition pass

**Files:** Modify: `src/nuzline/engine/step.py` (new `field_conditions(tables, state) ->
(state, events)`, called between `_execute_actions` and `end_of_turn`; the module docstring at
:47-52), `src/nuzline/reference/engine.py` (the same pass in `ref_step`), `engine/STATE.md`.
Test: `tests/engine/test_field_pass.py` (new).

Decomp order (bcp.c:932-1254; survey §4.2):
1. Reflect, then Light Screen, side 0 then side 1. Decrement; at 0 the screen ends.
2. Mist, Safeguard, Tailwind and Lucky Chant, for each one that has a state field; check
   `state.py` LAYOUT.
3. Weather. Permanent weather (`weather_timer == 0`) never ends; temporary weather decrements
   and ends at 0.
   - Sandstorm chip: `Divide(maxHP, 16)`, truncating, at least 1. It applies to each battler
     in speed order that is alive, not Rock, Ground or Steel, not Sand Veil, with no
     Cloud Nine/Air Lock alive. Survey §4.3 (script.c:5573-5583).
   - Hail chip: the same, except Ice types are immune and Snow Cloak replaces Sand Veil
     (script.c, the hail branch). Rock, Ground and Steel are not immune to hail; only Ice types and
     Snow Cloak are, and Ice Body heals 1/16 instead (script.c:5598-5610). Coverage still refuses hail.
4. Gravity, if a state field exists.
5. Trick Room is not counted here (bcp.c:1823-1826, turn end).

Then ONE faint check after the whole weather step (the Review Focus 1 rule), before the
per-Pokémon `END_OF_TURN_ORDER`.

- [ ] **Step 1: Write the tests.**
  - `test_screens_count_down_and_end`: Light Screen 1 → 0 after one turn, 5 → 4.
  - `test_permanent_sand_never_ends`.
  - `test_temporary_sand_ends_at_zero`.
  - `test_sand_chip_values`: maxHP 46 → 2, 17 → 1, 15 → 1.
  - `test_sand_immunities`: Rock, Ground, Steel and Sand Veil take no chip.
  - `test_sand_double_faint_is_a_loss`: the last player and last opponent Pokémon both at 1 HP,
    neither immune. The battle ends with `winner` = draw, scored as a loss, as in
    `_faint_check` step.py:696-736.
  - `test_field_pass_matches_reference`: fixed states with every weather and timer.
- [ ] **Step 2: Run them; see them fail.**
- [ ] **Step 3: Implement** both engines.
- [ ] **Step 4: Run** `uv run pytest tests/engine/test_field_pass.py
  tests/engine/test_field_effects.py tests/test_differential.py -q`: PASS. The strategies already
  draw weather and timers (survey §2.3), so the differential test covers the pass.
- [ ] **Step 5: Commit** `feat(engine): end-of-turn field-condition pass (screens, weather)`.

### Task 4: Sandstorm is EXACT; Cloud Nine gating; Sand Veil, Snow Cloak, Swift Swim

**Files:** Modify: `src/nuzline/engine/damage.py` (the Rock Special Defense boost at :268-271
and rain/sun power at :298-305, gated on no alive Cloud Nine/Air Lock), `engine/outcome.py`
(`hit_threshold`: Sand Veil ×80/100 in sand and Snow Cloak in hail, after Compoundeyes and
before fog, all gated; fog's gating too; bcp.c:2925-2935), `engine/order.py` (Swift Swim ×2
speed in rain, gated), `reference/engine.py` (the same), `engine/ids.py` (`AbilityId.SAND_VEIL`,
`SNOW_CLOAK`, `SWIFT_SWIM`), `tools/effect_map.py` (by name), `engine/coverage.py`
(`IMPLEMENTED_WEATHER[SANDSTORM] = EXACT`; docstring :26-34), `engine/hooks.py:48-51`
(Cloud Nine docstring), `tests/test_damage_oracle.py` docstring :9-12. Test:
`tests/engine/test_weather.py` (new), `tests/engine/test_coverage.py::test_only_no_weather_is_supported`
(rename to `test_only_no_weather_and_sandstorm_are_supported`).

- [ ] **Step 1: Write the tests.**
  - `test_rock_spdef_boost_in_sand`.
  - `test_cloud_nine_suppresses_sand_boost_and_chip`.
  - `test_sand_veil_accuracy_80_percent`: a 100-accuracy move gives a threshold of 80.
  - `test_snow_cloak_in_hail`.
  - `test_swift_swim_doubles_speed_in_rain_unless_cloud_nine`.
  - `test_weather_abilities_match_reference`.
  - Damage oracle: add sandstorm matchups where `tools/smogon_calc` models them.
- [ ] **Step 2: Fail.** **Step 3: Implement.** Rebuild the tables if the effect map changed.
- [ ] **Step 4: Run** the new tests, `tests/test_damage_oracle.py`,
  `tests/engine/test_m4_abilities.py`, `tests/test_differential.py`, `tests/engine/test_coverage.py`,
  and the vanilla coverage text check: PASS, no vanilla diff.
- [ ] **Step 5: Commit** `feat(engine): sandstorm supported; Cloud Nine gates weather; Sand Veil, Snow Cloak, Swift Swim`.

### Task 5: Light Screen and Brick Break

**Files:** `ids.py` (`EffectId.LIGHT_SCREEN`, `EffectId.BRICK_BREAK`), `effect_map.py` (35 →
LIGHT_SCREEN, 186 → BRICK_BREAK; Reflect's effect id too if it is a one-line twin, with its own
test), `effects.py`, `damage.py:293-296` (the screen halving skips moves whose effect is
REMOVE_SCREENS; lib.c:7023-7032, :6982-6991), `reference/engine.py`. Test:
`tests/engine/test_screens.py` (new).

Rules (survey §4.4):
- Light Screen fails if already up, else sets 5 turns. Light Clay is not modelled; coverage
  refuses it if held.
- Brick Break ignores the screen in its own damage. It clears both screens on the defender's
  side before the damage is applied (bcp.c:3295-3338, lib.c:1495), even when the target is
  immune, but not on a miss.

- [ ] **Step 1: Write the tests.**
  - `test_light_screen_sets_five_turns_and_fails_when_up`.
  - `test_light_screen_halves_special_noncrit`.
  - `test_brick_break_ignores_and_breaks_screens`.
  - `test_brick_break_breaks_on_immune_not_on_miss`.
  - `test_screens_match_reference`.
- [ ] **Steps 2-4:** fail, implement, and run focused tests plus the differential test.
- [ ] **Step 5: Commit** `feat(engine): Light Screen and Brick Break`.

### Task 6: Mapping-only moves

**Files:** `tools/effect_map.py`, `tests/tools/test_effect_map.py`, and the tables rebuild.
`effects.py` and the reference change only where a test shows a gap.

Mappings, from survey §1.3 and §6:
- 33 → `STATUS_FOE(Status.TOXIC, type_checked=0)` (Toxic). Only power moves plus Thunder Wave go
  through the type chart (bcp.c:2350-2357); Toxic's immunities come from its subscript.
- 261 → `TRAP` (Sand Tomb; Whirlpool's dive double is unreachable, because Dive is unsupported).
- 17 → `PLAIN` (Swift, Shock Wave and Kaizo Aurora Beam; never-miss comes from accuracy 0).
- 34 → `PLAIN` (Pay Day; the money is irrelevant to battle).
- 149 → `PLAIN` (Gust). The note says the ×2 against Fly/Bounce is unreachable because neither is
  supported. Add a coverage refusal line if either ever becomes supported:
  `test_gust_note_holds` asserts both stay `UNSUPPORTED`.

- [ ] **Step 1:** `test_mapping_only_moves` asserts each move's Kaizo `move_effect`/`move_param`
  in the tables, and that `check_pool`'s rows for these moves are EXACT.
- [ ] **Steps 2-4:** fail, map, rebuild both games, then run `tests/tools/test_effect_map.py`,
  `tests/test_differential.py` and the vanilla coverage text check.
- [ ] **Step 5: Commit** `feat(engine): Toxic, Sand Tomb, Swift family, Pay Day, Gust mapped`.

### Task 7: Small new move families

**Files:** `ids.py` (`EffectId.RECOVER`, `FAKE_OUT`, `RAISE_ALL_STATS_HIT`, `CURSE`; DoubleSlap
reuses `MULTI_HIT` with `move_param[m, 0] = 2` meaning a fixed hit count, 0 meaning the 2-5
law), `effect_map.py` (32, 158, 140, 109, 44), `effects.py`, `enumerate.py` (n_hits relevance
ignores fixed counts), `reference/engine.py`, `engine/coverage.py` (a per-Pokémon refusal for a
Ghost-type Curse user: "Curse (Ghost user) UNSUPPORTED"). Test: `tests/engine/test_small_moves.py`
(new).

Rules (survey §6):
- Recover heals `Divide(maxHP, 2)` and fails at full HP.
- Fake Out fails unless it is the user's first turn out (`fake_out_turn`, step.py:224-232). It
  flinches 100% through the normal flinch rules (Inner Focus, target has not moved).
- AncientPower raises Atk, Def, Spe, SpA and SpD by 1 on the existing `secondary` roll at the
  move's chance.
- Curse, for a non-Ghost user: Spe -1, then Atk +1, then Def +1. It fails only when every stage
  is capped.
- DoubleSlap hits exactly twice, with one accuracy check.

- [ ] **Step 1: Write the tests.**
  - `test_recover_half_and_fails_full`.
  - `test_fake_out_only_first_turn`.
  - `test_ancientpower_all_stats_on_roll`.
  - `test_curse_non_ghost_order_and_caps`.
  - `test_curse_ghost_user_refused`.
  - `test_doubleslap_two_hits_one_accuracy`.
  - `test_small_moves_match_reference`.
- [ ] **Steps 2-4:** fail, implement, then run the focused tests, `tests/engine/test_enumerate.py`
  and `tests/test_differential.py`.
- [ ] **Step 5: Commit** `feat(engine): Recover, Fake Out, AncientPower, Curse, DoubleSlap`.

### Task 8: Self-Destruct

**Files:** `ids.py` (`EffectId.SELF_DESTRUCT`), `effect_map.py` (7), `effects.py` /
`damage.py` (the target's Defense is halved in the formula, lib.c:6943-6945), `step.py` (the
user's HP is set to 0 in the script, before accuracy and the type chart), `reference/engine.py`.
Damp is unsupported: coverage refuses a fight where Damp and Self-Destruct meet. Test:
`tests/engine/test_self_destruct.py`.

- [ ] **Step 1: Write the tests.**
  - `test_self_destruct_user_faints_on_hit_miss_and_ghost`.
  - `test_self_destruct_halves_defense`.
  - `test_self_destruct_both_last_is_draw`.
  - `test_self_destruct_vs_focus_sash` (enable once Task 11 lands; mark it xfail until then,
    with a reason).
  - `test_self_destruct_matches_reference`.
- [ ] **Steps 2-5:** fail, implement, run, then commit `feat(engine): Self-Destruct`.

### Task 9: Flail

**Files:** `ids.py` (`EffectId.FLAIL`), `effect_map.py` (99), `damage.py` (band power from
`cur*64//max`, at least 1 when cur > 0; bands ≤1:200, ≤5:150, ≤12:100, ≤21:80, ≤42:40, else 20;
script.c:4941-4970; NOT on the AI path: the AI computes no damage at all for moves with table power <= 1
(trainer_ai.c:1036, :1554, :1605, :2828); the engine is right because `ai_damage` is gated by `is_damaging`), `prep/variants.py` (`hp_cuts` adds the band boundaries for a
Flail user), `reference/engine.py`. Test: `tests/engine/test_power_moves.py` (extend).

- [ ] **Step 1: Write the tests.**
  - `test_flail_bands` (boundaries for maxHP 42).
  - `test_flail_ai_damage_uses_power_one`.
  - `test_flail_hp_cuts`.
  - `test_flail_matches_reference`.
  - An oracle row if `tools/smogon_calc` agrees (Gen 4 Flail uses the same 64ths); otherwise
    document why it is excluded.
- [ ] **Steps 2-5:** fail, implement, run, then commit `feat(engine): Flail`.

### Task 10: Abilities — Rough Skin, Flame Body, Limber, Adaptability, Solid Rock

**Files:** `ids.py`, `effect_map.py` (by name), `effects.py` (contact procs at :468-490: Rough
Skin with no roll, `Divide(maxHP, 8)` to the attacker, not ignorable; Flame Body burns on the
existing `contact_proc` 3/10 roll, so extend its relevance in `enumerate.py`), `damage.py`
(Adaptability STAB ×2, lib.c:2592-2597; Solid Rock `Divide(damage*3, 4)` on super-effective
hits after the type chart, ignorable through Mold Breaker, lib.c:2652-2656), `hooks.py` (Limber
cures paralysis at switch-in and move end, like Insomnia, hooks.py:109-134), `effects.py`
(Limber blocks paralysis, ignorable), `prep/problem.py:268-275` (prestatus blocker),
`reference/engine.py`. Test: `tests/engine/test_k3b_abilities.py`; oracle rows for Adaptability
and Solid Rock (check the oracle's float 0.75 against `Divide`; survey §6).

- [ ] **Step 1:** One test per ability rule above, plus `test_k3b_abilities_match_reference` and
  the oracle rows. Add the abilities to `tests/strategies.py` (`LAW_ABILITIES` or the ability
  draw) so the differential test exercises them.
- [ ] **Steps 2-5:** fail, implement, run (focused, enumerate, differential, oracle), then commit
  `feat(engine): Rough Skin, Flame Body, Limber, Adaptability, Solid Rock`.

### Task 11: Items — Focus Sash, Big Root, Berry Juice, status berries, White Herb, Lagging Tail

**Files:** `ids.py` (`ItemId.FOCUS_SASH`, `BIG_ROOT`, `BERRY_JUICE`, `CHESTO_BERRY`,
`PECHA_BERRY`, `LUM_BERRY`, `WHITE_HERB`, `LAGGING_TAIL`), `effect_map.py`, `step.py` /
`effects.py` / `hooks.py` / `order.py`, `prep/variants.py` (`hp_cuts`: full HP for a Focus Sash
holder), `reference/engine.py`. Test: `tests/engine/test_k3b_items.py`.

Rules (survey §6):
- Focus Sash: at full HP a hit that would KO leaves the holder at 1 HP. It applies at the move
  damage site, the confusion self-hit and Pursuit. It is consumed (`item_consumed`). A full-HP
  hit sets the per-action `focusItemActivated` flag (bcp.c:3386-3404, cleared at action end,
  bcp.c:3998), so every later lethal hit of the same move is also capped to 1 HP and multi-hit
  moves do not break the Sash. Pursuit and the confusion self-hit use the plain full-HP check
  (`BtlCmd_CheckHoldOnWith1HP`).
- Big Root: drain and Aqua Ring heals ×(100+30)/100 after the base amount, with no minimum.
  Drain is supported today; Aqua Ring lands in Task 12.
- Berry Juice: Oran's code path with param 20. Pluck and Bug Bite do not eat it: refuse
  `EAT_BERRY` against a Berry Juice holder in coverage, or model the decomp's non-eating, which
  is preferred if it is a one-line check (lib.c:5833).
- Chesto, Pecha and Lum: the same call sites as Oran/Sitrus (lib.c:4576-4700, :4967-5100). Pecha
  also cures toxic. Lum cures every status and confusion.
- White Herb: restores lowered stat stages (lib.c:4875, :5088) and is consumed.
- Lagging Tail: within equal priority, the holder moves last, but Quick Claw would beat it
  (K3b2). It also applies to the end-of-turn order (lib.c:1401-1440).

- [ ] **Step 1:** One test per rule, including:
  - `test_focus_sash_crit`.
  - `test_focus_sash_multi_hit_flag_lasts_the_move`.
  - `test_focus_sash_confusion_self_hit`.
  - `test_lum_cures_confusion`.
  - `test_pecha_cures_toxic`.
  - `test_lagging_tail_end_of_turn_order`.
  - `test_k3b_items_match_reference`.

  Add the items to the strategies' item draw. Enable Task 8's Focus Sash test.
- [ ] **Steps 2-5:** fail, implement, run (focused, enumerate, differential), then commit
  `feat(engine): Focus Sash, Big Root, Berry Juice, status berries, White Herb, Lagging Tail`.

### Task 12: Aqua Ring

**Files:** `ids.py` (`EffectId.AQUA_RING`), `effect_map.py` (251), `state.py` (a volatile flag,
cleared on switch; STATE.md), `step.py` (end of turn, second in the Pokémon-condition order,
bcp.c:1331-1351: heals `Divide(maxHP, 16)` when not at full HP and alive; ×1.3 with Big Root),
`reference/engine.py`. Test: `tests/engine/test_aqua_ring.py`.

- [ ] **Step 1: Write the tests.**
  - `test_aqua_ring_fails_if_set`.
  - `test_aqua_ring_heal_and_order`.
  - `test_aqua_ring_big_root`.
  - `test_aqua_ring_cleared_on_switch`.
  - `test_aqua_ring_matches_reference`.
- [ ] **Steps 2-5:** fail, implement, run, then commit `feat(engine): Aqua Ring`.

### Task 13: Resist berries — Rindo, Chilan

**Files:** `ids.py` (`ItemId.RESIST_BERRY`; `item_param` holds the resisted type, with a
sentinel for Chilan's "any Normal move"), `effect_map.py` (hold effects 19-35 by name; only the
ones whose code is tested become supported), `step.py` (`subscript_type_resist_berry.s`: halves
the final HP change after Focus Sash's cap and removes the berry; Rindo only on super-effective
Grass hits, Chilan on any Normal hit), `observe.py` (the berry message), `reference/engine.py`.
Test: `tests/engine/test_resist_berries.py`.

- [ ] **Step 1: Write the tests.**
  - `test_rindo_halves_se_grass_and_is_consumed`.
  - `test_rindo_ignores_neutral_grass`.
  - `test_chilan_any_normal`.
  - `test_resist_after_focus_sash`.
  - `test_resist_berries_match_reference`.
- [ ] **Steps 2-5:** fail, implement, run, then commit `feat(engine): Rindo and Chilan berries`.

### Task 14: Close K3b1

- [ ] **Step 1:** Rebuild both games' tables and run the full suite (the controller runs it).
- [ ] **Step 2:** Run `uv run nuzline coverage --game kaizo --save "G:\Farry\Downloads\Platinum
  Kaizo Base.dsv" --trainer roark --cap 16` and record every remaining line. Expected: the K3c AI
  lines, `Kaizo ROM: 1 code changes unclassified`, and only K3b2 mechanics: Thunder Fang,
  Protect, Present, Attract, Cute Charm, Quick Claw, Roar, Nature Power, Camouflage, and the two
  terrain ROM conditions. Anything else is a gap to fix here.
- [ ] **Step 3:** Update `docs/kaizo/pipeline.md` ("What is refused") and spec §4 with one line
  per landed group. Write the K3b2 handover in
  `C:\Users\Farry\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_a4231f03-556b-4b7f-bc82-38b27d410c76\kaizo-k3a\k3b1-handover.md`.
  Commit `docs(kaizo): K3b1 done`.

---

**Plan-level rulings (controller):**
- K3b is split into K3b1 (this plan, no new `Outcome` fields) and K3b2 (the draw mechanics,
  Camouflage, Roar, Nature Power and terrain), because K3b is twice K3a's size and the draw
  fields change the enumeration width; survey §3.3.
- Vanilla `coverage --save` output gains bag-item rows (Task 2). This is reporting, not a change
  to battle results: the spec's anti-goal forbids silent drops, and vanilla's exact results are
  unchanged.
