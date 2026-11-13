# K3b planning survey (read-only research, 2026-11-08, main at 083669a)

> Note: statements here are hypotheses from a read-only survey. The per-rule audit against the
> decomp is in `docs/kaizo/k3b-rule-audit.md`; where they disagree the audit wins. User-reported
> in-game quirks are in `docs/kaizo/user-reported-quirks.md`.

Purpose: everything a plan writer needs to cut K3b into TDD tasks without re-reading the code.
Spec: `docs/superpowers/specs/2026-11-07-kaizo-k3-design.md` §4 (mechanics), §8 (K3b = "the field
and the mechanics"). K3a handover: `handover.md` in this folder; `overlay16.md` §2 has the obedience
decode.

Path conventions: paths are relative to the repository root; `bcp.c` =
`external/pokeplatinum/src/battle/battle_controller_player.c`, `lib.c` =
`.../src/battle/battle_lib.c`, `script.c` = `.../src/battle/battle_script.c`; `.s` files are under
`external/pokeplatinum/res/battle/scripts/`. "K" = data/intermediate_kaizo, "V" = data/intermediate.

---

## 1. The exact missing list today

### 1.1 Commands run

`<pre-Roark save>` and `<probe save>` are the `kaizo_save_pre_roark` and `kaizo_save_probe`
saves of `data/local.toml`.

```
uv run nuzline coverage --game kaizo --save <pre-Roark save> --trainer roark --cap 16
   -> exit 2; Fight: Support NONE, Missing (54); Pool (12 Pokemon): Missing (40)
uv run nuzline coverage --game kaizo --save <probe save> --trainer roark --cap 16
   -> exit 2; Fight: Missing (59); Pool: Missing (40)
```

The two saves hold the same 12 Pokemon; they differ only in which six are in the party (the
fight check uses the party, the pool check uses party + box 1). Probe-save party (what the user
brought to Roark): Prinplup L16 Quick Claw [Aqua Cutter, Peck, BubbleBeam, Metal Claw],
Wormadam L16 form 2 (Trash) Snow Cloak [Leech Life, Nature Power, Camouflage], Rhyhorn L16 Rock
Head, Delibird L16 Adaptability [Present, Powder Snow, Gust], Sandshrew L16 Rough Skin, Luvdisc
L16 Swift Swim [Aqua Ring, Charm, Aqua Jet, Water Pulse]. Base-save party: Sandshrew, Flaaffy
(Static), Shinx L8, Vulpix (Flame Body), Goldeen (Swift Swim), Meowth (Limber); box 1 holds the
probe party's six.

The probe-save fight adds two ROM lines the base save does not have, because Wormadam (Nature
Power/Camouflage) is in the party there:
`Kaizo ROM: arm9 MapHeader_GetBattleBG is not modelled and this fight reaches it` and the same for
`MapHeader_GetMapMatrixID` (data/kaizo/rom_diff.toml:115-120 and :35-40, `when = { move_effect =
[173, 197, 213] }`). In the base save they appear only on the pool row of box 1 slot 1 Wormadam.

### 1.2 Weather / field (fight)

| line | where it comes from | K3b? |
|---|---|---|
| `Battle start: Sandstorm (weather UNSUPPORTED)` | `coverage._check_weather`, coverage.py:416-424, `IMPLEMENTED_WEATHER[SANDSTORM] = NONE` coverage.py:132-139; Roark `start_condition = "sandstorm"`, `area_condition.permanent = true` (K trainers.json id 246) | yes |
| `Kaizo ROM: arm9 MapHeader_GetBattleBG ... reaches it` / `..._GetMapMatrixID ...` (only with Wormadam in the party; pool row otherwise) | rom_diff `not_reached` on move effects 173/197/213 | yes (terrain, §5.3) |
| `Kaizo ROM: 2 code changes unclassified` | `_check_kaizo_fields`, coverage.py:407-413, manifest `unclassified_rom_changes` | K3b removes one (CheckObedience); the other (overlay14 0x022249CC Dry Skin word) is K3c |

No screens appear as lines (Light Screen is refused as a move effect).

### 1.3 Move effects (fight + pool), with Kaizo ROM data

Data from K moves.json (power, accuracy 0 = never misses, chance, priority, effect = decomp
`BATTLE_EFFECT_*` id, line N+1 of `external/pokeplatinum/generated/move_battle_effects.txt`).
"Kaizo vs vanilla" compares K vs V moves.json (same index).

Roark's side (trainer 246):

| move (id) | effect id / name | pow | acc | chance | prio | contact | Kaizo vs vanilla |
|---|---|---|---|---|---|---|---|
| Selfdestruct (120) | 7 HALVE_DEFENSE | 200 | 100 | 0 | 0 | no | unchanged; range RANGE_ALL_ADJACENT |
| Brick Break (280) | 186 REMOVE_SCREENS | 75 | 100 | 0 | 0 | yes | unchanged |
| AncientPower (246) | 140 RAISE_ALL_STATS_HIT | 80 | 100 | 10 | 0 | no | power 60->80, pp 5->10 |
| Recover (105) | 32 RESTORE_HALF_HP | 0 | 0 | 0 | 0 | no | unchanged |
| Thunder Fang (422) | 275 FLINCH_PARALYZE_HIT | 90 | 100 | 10 | 0 | yes | power 65->90, acc 95->100 |
| Light Screen (113) | 35 SET_LIGHT_SCREEN | 0 | 0 | 0 | 0 | no | pp 30->2 |
| Toxic (92) | 33 STATUS_BADLY_POISON | 0 | 100 | 0 | 0 | no | acc 85->100 |
| Curse (174) | 109 CURSE | 0 | 0 | 0 | 0 | no | pp 10->3 |
| Sand Tomb (328) | 261 WHIRLPOOL | 90 | 100 | 0 | 0 | no | power 15->90, acc 70->100, effect 42 (BIND_HIT) -> 261 |

Player pool (cap 16):

| move (id) | effect id / name | pow | acc | chance | prio | contact | Kaizo vs vanilla | who |
|---|---|---|---|---|---|---|---|---|
| Toxic (92) | 33 | 0 | 100 | 0 | 0 | no | acc 85->100 | TM06, all 12 |
| Swift (129) | 17 BYPASS_ACCURACY | 60 | 0 | 0 | 0 | no | unchanged; RANGE_ADJACENT_OPPONENTS | Flaaffy, Shinx (current) |
| Shock Wave (351) | 17 | 60 | 0 | 0 | 0 | no | pp 20->15 | Flaaffy (relearner) |
| Roar (46) | 28 FORCE_SWITCH | 0 | 100 | 0 | -6 | no | unchanged | Vulpix (current) |
| Pay Day (6) | 34 INCREASE_PRIZE_MONEY | 40 | 100 | 0 | 0 | no | unchanged | Meowth (relearner) |
| Aqua Ring (392) | 251 RESTORE_HP_EVERY_TURN | 0 | 0 | 0 | 0 | no | unchanged | Luvdisc (current) |
| Flail (175) | 99 INCREASE_POWER_WITH_LESS_HP | 1 | 100 | 0 | 0 | yes | unchanged | Luvdisc (relearner) |
| Nature Power (267) | 173 NATURE_POWER | 0 | 0 | 0 | 0 | no | unchanged; RANGE_SINGLE_TARGET_SPECIAL | Wormadam (current) |
| Camouflage (293) | 213 CAMOUFLAGE | 0 | 0 | 0 | 0 | no | unchanged; RANGE_USER | Wormadam (current) |
| Protect (182) | 111 PROTECT | 0 | 0 | 0 | +3 | no | unchanged | Wormadam (relearner) |
| Fake Out (252) | 158 ALWAYS_FLINCH_FIRST_TURN_ONLY | 40 | 100 | 100 | **+3** | yes | prio 1->3, pp 10->5 | Skitty (current) |
| Present (217) | 122 RANDOM_POWER_MAYBE_HEAL | 1 | 100 | 0 | 0 | no | acc 90->100 | Delibird (current), Skitty (relearner) |
| Attract (213) | 120 INFATUATE | 0 | 100 | 0 | 0 | no | range SINGLE_TARGET -> ADJACENT_OPPONENTS | Skitty (relearner) |
| DoubleSlap (3) | **44 HIT_TWICE** | **30** | **90** | 0 | 0 | yes | power 15->30, acc 85->90, effect 29 (MULTI_HIT) -> 44 | Skitty (relearner) |
| Aurora Beam (62) | **17 BYPASS_ACCURACY** | 60 | **0** | 0 | 0 | no | power 65->60, acc 100->0, effect 68 (LOWER_ATTACK_HIT 10%) -> 17 | Prinplup (relearner) |
| Gust (16) | 149 DOUBLE_DAMAGE_FLY_OR_BOUNCE | 40 | 100 | 0 | 0 | no | unchanged | Delibird (current) |

Nature Power calls Tri Attack in Roark's gym (§5.3): Tri Attack (161) K = effect 36 TRI_ATTACK,
power 95 (V 80), acc 100, chance 30 (V 20). Camouflage there makes the user Normal.

Already-supported moves of Roark (no line today, but Kaizo-changed data the engine already reads
from the K tables): Stealth Rock (446, +1 prio in Kaizo per spec), Accelerock (205, vanilla
Rollout's slot: effect 0, 40 BP, +1), Sludge Bomb 95 BP, Giga Drain 75 BP (Big Root applies),
DragonBreath 70, Flamethrower, Earth Power, Water Pulse (acc 0 = never misses), Head Smash (269
recoil half, acc 100; Cranidos has Rock Head), Zen Headbutt acc 100, Crunch 95, Rock Polish pp 1.
Kaizo Dig (91) is a plain hit (effect 0, 60 BP): no semi-invulnerable turn exists in this pool.

### 1.4 Abilities

| ability (decomp id) | side | holder(s) | notes |
|---|---|---|---|
| Solid Rock (116) | Roark | Lileep, Corsola, Shuckle (slot 0) | V species abilities differ: Lileep [21,0], Corsola [55,30], Shuckle [5,82] -> K [116,0], [116,116], [116,116] |
| Sand Veil (8) | Roark | Gible slot 0 (K abilities [8, 24]; V [8, 0]) | |
| Rough Skin (24) | player | Sandshrew | |
| Flame Body (49) | player | Vulpix | |
| Swift Swim (33) | player | Goldeen, Luvdisc | |
| Limber (7) | player | Meowth | |
| Snow Cloak (81) | player | Wormadam form 2 | |
| Cute Charm (56) | player | Skitty | |
| Adaptability (91) | player | Delibird | |

K abilities.json and V abilities.json have identical names at all 124 ids (checked); abilities.json
is names only, so behaviour is code, and Kaizo changed no battle ability code (overlay16 changed
only the 4 routines of `overlay16.md` §2). Bonsly and Cranidos lead with Rock Head (slot 0).

### 1.5 Held items (K items.json; every record below equals V's)

| item (id) | hold effect (generated/item_hold_effects.txt) | param | who |
|---|---|---|---|
| Focus Sash (275) | 103 HOLD_EFFECT_ENDURE | 0 | Roark Bonsly |
| Big Root (296) | 124 HOLD_EFFECT_LEECH_BOOST | 30 | Roark Lileep |
| Chesto Berry (150) | 6 HOLD_EFFECT_SLP_RESTORE | 0 | Roark Gible |
| Rindo Berry (187) | 22 HOLD_EFFECT_WEAKEN_SE_GRASS | 0 | Roark Corsola |
| Berry Juice (43) | 1 HOLD_EFFECT_HP_RESTORE | 20 | Roark Shuckle |
| Lum Berry (157) | 12 HOLD_EFFECT_STATUS_RESTORE | 0 | Roark Cranidos |
| Quick Claw (217) | 52 HOLD_EFFECT_SOMETIMES_PRIORITY | 20 | Prinplup (held) |
| White Herb (214) | 49 HOLD_EFFECT_STATDOWN_RESTORE | 0 | bag |
| Lagging Tail (279) | 107 HOLD_EFFECT_PRIORITY_DOWN | 0 | bag |
| Pecha Berry (151) | 7 HOLD_EFFECT_PSN_RESTORE | 0 | bag |
| Chilan Berry (200) | 35 HOLD_EFFECT_WEAKEN_NORMAL | 0 | bag |
| Oran Berry (155) | 1 HP_RESTORE | 10 | bag, already supported |

Both saves' bags are identical (items: Honey, Stardust x10, White Herb, Lagging Tail, Yellow
Shard, Heart Scale, Pearl x20, Moon Stone; berries: Pecha, Oran x3, Chilan; medicine: Rare
Candy x30, Potion x84, Antidote, Parlyz Heal, Full Heal, Fresh Water; TMs: TM06 Toxic, TM10,
HM06). Every other bag item has hold_effect 0.

**The pool check lists no bag items today.** `check_pool` (src/nuzline/prep/pool.py:99-162) only
checks each candidate's current item. `bag_resources` (src/nuzline/prep/problem.py:50-80) silently
keeps only bag items whose `ItemId` is implemented (`HELD_POCKETS`, problem.py:34), i.e. today it
drops White Herb, Lagging Tail, Pecha and Chilan without a warning (spec §2 anti-goal). Spec §4.4
asks for one pool row per bag item with a battle hold effect. The tables do not carry the raw hold
effect (`item_effect` is the mapped engine id; build_tables.py:463-477), so "has a hold effect" needs
either a new `item_hold_effect` array (like `move_ai_effect`, build_tables.py:458) or Names data.

### 1.6 AI refusal lines (K3c, listed for completeness; K3b must not try to remove them)

Fight (both saves): `Leader Roark: no transcribed AI policy for this game (spec 5.10)`;
`Leader Roark: AI_FLAG_RISKY` (flags 23 = BASIC|EVAL_ATTACK|EXPERT|RISKY); 24 `ai scoring: opponent
slot N: the script paths of move X are not transcribed (docs/ai/roark.md 5)` lines: slot 0
Selfdestruct, Brick Break, Accelerock; slot 1 Sludge Bomb, Giga Drain, AncientPower, Recover; slot
2 DragonBreath, Flamethrower, Thunder Fang, Earth Power; slot 3 Recover, Water Pulse, Light
Screen, Accelerock; slot 4 Toxic, Accelerock, Curse, Sand Tomb; slot 5 Head Smash, Zen Headbutt,
Crunch, Rock Polish. Plus the remaining `Kaizo ROM: 1 code changes unclassified` after K3b.

Pool: no AI lines today, and none expected after K3b: `check_pool` asks
`player_move_refusal`/`player_ability_refusal` only once the engine supports the mechanic
(pool.py:128, :139); none of the pool's moves is in `switching.PLAYER_MOVE_REFUSALS`
(src/nuzline/ai/switching.py:130-151) or the type-chart state moves (policy.py:325-327), and no
pool ability is in `_DEFENDER_ABILITIES` (policy.py:324: Levitate, Wonder Guard, Filter, Solid
Rock) or `PLAYER_ABILITY_REFUSALS` (switching.py:118-146). The fight-level `_refused`
(policy.py:364-398) checks side 0 against defender abilities and side 1 (Roark) against attacker
abilities (Normalize, Adaptability, Scrappy, Tinted Lens), so Roark's Solid Rock and the player's
Adaptability do not trip it.

### 1.7 What "done" looks like for K3b

After K3b the fight should list exactly: the policy line, the RISKY line, the 24 script-path
lines, and `Kaizo ROM: 1 code changes unclassified`. The pool should be empty, with four new bag
rows supported (White Herb, Lagging Tail, Pecha, Chilan). The spec's §4.5 save test ("pool check on
the save is empty", skipped without the save) belongs here: fixture `kaizo_save_pre_roark_path`
(tests/conftest.py:126-129).

---

## 2. How a mechanic is added today (end to end)

### 2.1 The layers

1. **Enum** (`src/nuzline/engine/ids.py`): `EffectId` :12-67 (29 members incl. UNSUPPORTED=0),
   `AbilityId` :70-93 (22), `ItemId` :96-103 (NONE, ORAN, SITRUS), `TrainerItemId` :106-113,
   `Status`, `Stat`, `MoveFlag` (CONTACT, PROTECT, SOUND, PUNCH :144-150), `Weather` :171-179.
   Param conventions are in the `EffectId` docstring :13-36.
   **Pitfall:** the coverage registries are comprehensions over the enums (coverage.py:103-114),
   so adding a member marks it EXACT immediately. Add the member in the same task as the code
   (or in a commit where nothing maps to it yet).
2. **Effect map** (`tools/effect_map.py`): `_EFFECTS` decomp effect id -> (constant name,
   EffectId, params) :60-148; `CERTAIN_SECONDARY_EFFECTS` :161; `DECOMP_ABILITY_TO_ID` :174-197 (by
   name); `DECOMP_ITEM_TO_ID` :199-203 (by name); trainer items :206-209.
   `tools/build_tables.py` packs it: `_move_effect` :150-173 (SECONDARY rules, chance-0
   handling), `_item_maps` :213-238 (Kaizo: an item maps only when the slot keeps vanilla's
   name and hold effect), `_kaizo_rom_changes` :241-268 (a changed be_seq member forces
   UNSUPPORTED unless `modelled`), arrays :458-477 (`move_effect`, `move_param`,
   `move_ai_effect` = raw decomp effect, `item_effect`, `item_param` = hold param). Rebuild:
   `uv run python -m tools.build_tables` and `... --game kaizo`. Only
   `data/gen4_platinum/manifest.json` is tracked in git; the `.npz` tables are local build
   output, so every table-shape change needs both rebuilds before the suite (tests load them via
   `tests/conftest.py:36-59, 207-230` and skip Kaizo tests when Kaizo tables are absent).
3. **JAX engine**: `damage.py` (power/stat/field/type-chart modifiers; AI path
   `ai_damage_for_roll` :408-425 shares `_pre_random`/`_finish`), `effects.py`
   (`apply_move_effect` :519-642 dispatches per hit; effect functions :184-500), `hooks.py`
   (switch-in, berries, ability reveals: `REVEALS_ON` :89-95, `after_damage_items` :137-162,
   `pluck_berry` :165-195, `on_switch_in` :214-227), `step.py` (status checks and the move :336-566,
   actions :768-820, end of turn :944-998), `order.py` (speed and order :60-88), `actions.py`
   (legality :56-84), `outcome.py` (draws), `enumerate.py` (laws + relevance), `observe.py`
   (player observation; events from `step_events`, step.py:148-168, :1026-1039), `state.py`
   (`BattleState` :70-150, `LAYOUT` :204-279, `make_state` :366-383).
4. **Reference engine** `src/nuzline/reference/engine.py` (imports only `ids` from the engine,
   :1-25): damage `_calc_move_damage` :278-396 (sand :352, screens :378, weather :382-387),
   `_held_item_hp` :542, `_turn_end_ability` :592, `_move_effect` :714, `_contact_ability` :826,
   `_hit` :869, `_status_disrupted` :932, `_fight` :1016, `_end_of_turn` :1153-1175,
   `ref_step` :1284-1371. State mirror `reference/state.py` (`RefPokemon`/`RefSide`/`RefState`,
   field tuples at the end) and `reference/convert.py` (`to_ref`/`from_ref`).
5. **Coverage** `src/nuzline/engine/coverage.py`: registries :103-139, `_check_party` :222-252,
   ROM conditions :279-325, `_check_kaizo_fields` :382-413, `_check_weather` :416-424,
   `check_coverage` :449-473. Pool: `prep/pool.py:99-162`; prep refusals `prep/problem.py:83-127`.
6. **AI player-side refusals**: `ai/policy.py:330-350` (`player_move_refusal`,
   `player_ability_refusal`), lists in `ai/switching.py:93-151`.
7. **Prep HP cuts**: `prep/variants.py:280-294` (`hp_cuts`: AI percent checks, kill damages, pinch
   1/3); `AI_HP_READERS` :73-80 is pinned by a source-scan test; design.md:169 says new HP
   mechanics (Flail bands, Focus Sash) add their cut here.
8. **Per-game constants**: any decomp id hard-coded in code must be registered in
   `src/nuzline/data/game_constants.py` (`CONSTANTS`, e.g. :170-190; item constants can pin the
   hold effect, :184-187) and is checked against each game at load; `tests/data/test_game_constants.py`
   scans for unclassified id tables. K3b will hard-code at least Tri Attack (161), Protect /
   Detect / Endure (182/197/203, the Protect chain), and the terrain move/type tables.
9. **STATE.md** (`src/nuzline/engine/STATE.md`) maps every decomp struct field; rows K3b touches:
   `MoveEffectsData.quickClaw` :221 (later), `BattleContext.moveProtect` :393 (later),
   `speedRand` :413 ("not needed": outcome input), Attract / Aqua Ring "later (M6)" :51, :58,
   `type1` -> `types` :150 (party-slot level, see Camouflage), screens :244-246, weather :263.
   `tests/engine/test_state.py::test_state_md_accounts_for_every_struct_field` checks it.

### 2.2 Concrete recent examples

**Move effect: Magnitude** (commits 565bfa2 state/outcome fields, 4c732e1 the mechanic):
`EffectId.MAGNITUDE = 23` (ids.py:61); `126: ("BATTLE_EFFECT_PSYWAVE", _E.MAGNITUDE, ...)`
(effect_map.py:125-126; the decomp name is wrong, the script is Magnitude's); power table
`MAGNITUDE_POWER` damage.py:48-50 used in `_pre_random` :192-206; Outcome field `magnitude`
outcome.py:61, drawn from a `fold_in` key :213-218 so earlier fields keep their samples; law
`_magnitude` enumerate.py:347-349, relevance :272 and docstring :44-48; step event "magnitude"
step.py:549-554 and observation field observe.py:39-41/:141; reference `MAGNITUDE_POWER`
reference/engine.py:138, :311-312, :894. Tests: `tests/engine/test_power_moves.py`
(`test_magnitude_power_and_message` :260, reference diff `test_power_moves_match_reference`
:308), `tests/engine/test_enumerate.py::test_magnitude_law_exact` :237,
`tests/engine/test_outcome.py::test_magnitude_and_shed_skin_draws` :150, oracle override in
`tests/test_damage_oracle.py`.

**Ability: Shed Skin** (565bfa2 + 2fa645b): `AbilityId.SHED_SKIN = 22` (ids.py:93); map by
name effect_map.py:196; `Outcome.shed_skin` outcome.py:62, :219; law enumerate.py:352-353, :376,
relevance :273 (over-approximate, any holder); end-of-turn check `_turn_end_ability`
step.py:918-939 registered first in `END_OF_TURN_ORDER` :944-967; ability reveal
`REVEALS_ON[SHED_SKIN]` hooks.py:94; reference `_turn_end_ability` reference/engine.py:592-607.
Tests: `tests/engine/test_m4_abilities.py::test_shed_skin_cures_before_poison` :215,
`::test_abilities_match_reference` :240 (fixed-state JAX vs reference), strategies
`LAW_ABILITIES` tests/strategies.py:381-388.

**Held item: Oran / Sitrus** (960b230): `ItemId.ORAN_BERRY/SITRUS_BERRY` ids.py:102-103; by name
effect_map.py:199-203; `item_param[i, 0]` = hold param (build_tables.py:470-477);
`hooks.after_damage_items` :137-162 run after every hit (step.py:495-503), at switch-in
(hooks.py:226-227) and twice at end of turn (step.py:949, :966); Pluck/Bug Bite
`hooks.pluck_berry` :165-195; reference `_held_item_hp` reference/engine.py:542-566 and
`_pluck_berry` :694. Tests: `tests/engine/test_abilities_items.py::test_oran_berry_triggers_and_is_consumed`
:36, `::test_berry_triggers_at_end_of_turn_after_poison` :69,
`::test_sitrus_berry_triggers_after_stealth_rock_on_switch_in` :81.

### 2.3 Test kinds (one representative each)

| kind | file | representative |
|---|---|---|
| Hypothesis differential, random states (vanilla tables) | `tests/test_differential.py` | `test_jax_matches_reference` :35 (500 examples; `supported_state`, `legal_action_pair`, `any_outcome` from `tests/strategies.py`) |
| fixed-state differential | `tests/engine/helpers.py::assert_matches_reference` :289 | `tests/engine/test_power_moves.py::test_power_moves_match_reference` :308 |
| damage array vs @smogon/calc | `tests/test_damage_oracle.py` (node, `tools/smogon_calc/damage.js`) | `test_damage_table_matches_smogon_calc` :234 (300 matchups, all 16 rolls) |
| decomp-derived (tables / source scans) | `tests/tools/test_effect_map.py` | `test_stat_effects_match_decomp_scripts` :51; `tests/engine/test_m4_abilities.py::test_cloud_nine_sites_are_weather_only` :165 |
| outcome draws | `tests/engine/test_outcome.py` | `test_existing_draws_unchanged` :142 (pins seeded draws: new fields must use `fold_in`) |
| enumeration laws | `tests/engine/test_enumerate.py` | `test_matches_sampled_frequencies` :222, `test_width_cap` :181 |
| unit mechanics | `tests/engine/test_effects.py`, `test_field_effects.py`, `test_abilities_items.py` | `test_field_effects.py::test_end_of_turn_order_lists_the_implemented_checks_in_decomp_order` :226 |
| coverage | `tests/engine/test_coverage.py` | `test_only_no_weather_is_supported` :58 (must change for sand) |
| pool | `tests/prep/test_pool.py` | `test_check_pool_names_unsupported` :91 |
| ROM classification | `tests/kaizo/test_rom_classification.py` | `test_unknown_entries_name_their_milestone` :198 |

Strategies: `supported_state` (strategies.py:235-306) draws `weather` from every `Weather` value
and random `reflect`/`light_screen`/`tailwind`/`trick_room` timers (:274-276, :291-292), so the
moment either engine gets a field pass, the differential test exercises it for every weather and
timer: both engines must implement the same pass, including the parts coverage still refuses.
Move pools are vanilla level-up learnsets with supported effects (`m1_move_pools` :114-130);
items Oran/Sitrus/none; abilities the species' own (supported only). New mechanics get exercised
only if the strategies' species/moves/items/abilities include them (vanilla learnsets do include
e.g. Toxic, Protect, Swift, Curse once mapped; held items must be added to the item draw).

---

## 3. The Outcome layout

### 3.1 Fields today (`src/nuzline/engine/outcome.py:44-62`; laws enumerate.py:295-377)

| field | shape | values / law | read by |
|---|---|---|---|
| speed_tie | [] | 0/1, 1/2 | order.py:80 |
| hit | [2,5] | 0/1, threshold/100 (`hit_threshold` :106-142); only index 0 read | step.py:460 |
| crit | [2,5] | 0/1, 1 in 16/8/4/3/2 (:41); Lucky Chant -> 0 | effects.py:561 |
| roll | [2,5] | 0-15 uniform | damage `85 + roll` |
| secondary | [2,5] | 0/1, chance/100; chance 0 = certain | effects.py:604 |
| contact_proc | [2,5] by defending side | 0/1, 3/10 | effects.py:468-490 (Static, Poison Point) |
| n_hits | [2] | 2,3,4,5 at 3/8,3/8,1/8,1/8 | effects.py:159-163 |
| full_para | [2] | 1/4 | step.py:419 |
| confusion_hit | [2] | 1/2 | step.py:406 |
| duration | [2] by inflicting side | 2-5 / 3-6 (Bind, Disable) / 3-5 (Taunt) | effects.py |
| thaw | [2] | 1/5 | step.py:382 |
| flinch_item | [2] | **reserved**, always 0 (King's Rock / Razor Fang) | nothing |
| quick_claw | [2] | **reserved**, always 0 | nothing |
| end_speed_tie | [] | 1/2 | step.py:970-980 |
| magnitude | [2] | 4-10 at 5/10/20/30/20/10/5 % | damage |
| shed_skin | [2] | 3/10 | step.py:918-939 |

Sampling uses `jax.random.split(key, 14)` slots 0-13 (outcome.py:165) and `fold_in(key, 14)`,
`fold_in(key, 15)` for the two later fields (:213-219). New fields should take `fold_in(key, 16...)`
so `test_existing_draws_unchanged` (and vanilla Stage A numbers) stay put. The project treats
`RandNext() % n` as exactly uniform (e.g. `_hit` :299-301), so Quick Claw's `speedRand % 5` is 1/5.

### 3.2 What K3b needs in Outcome

| mechanic | draw (decomp) | proposal | status today |
|---|---|---|---|
| Thunder Fang | `subscript_paralyze_or_flinch.s`: two `CheckEffectActivation` (script.c:8363-8383: `RandNext() % 100 < chance`, target alive), paralysis then flinch; the subscript is ON_HIT so there is no outer roll (`effect_script_0275.s`) | `secondary` = paralysis roll; **new** `secondary2[2,5]` (flinch roll), law chance/100 | new field |
| AncientPower | one roll at 10 % (indirect, TO_ATTACKER) | existing `secondary` | ok |
| Tri Attack via Nature Power | `effect_script_0036.s`: `Random 2, 3` -> subscript 3/4/5 = burn/freeze/paralyze (include/data/move_side_effect_subscripts.h:8-12), then the 30 % roll | `secondary` + **new** `tri_attack[2]` 0-2 uniform 1/3 (drawn in the script, before accuracy) | new |
| Present | `BtlCmd_Present` script.c:5789-5806: `rnd = RandNext() & 0xFF`; <102 -> 40, <178 -> 80, <204 -> 120, else heal maxHP/4 (Divide) | **new** `present[2]` 4 values, law (102, 76, 26, 52)/256; on heal no crit/roll is read | new; design.md:117 names it |
| Roar | `BtlCmd_TryWhirlwind` script.c:5241-5320: if user level < target level, `BattleSystem_CanWhirlwind` lib.c:3072-3086 draws `(((RandNext() & 0xFF) * (Latk+Ldef)) >> 8) + 1 > Ldef/4`; then `RandNext() % partyCount` rejection loop until an eligible (species, not egg, HP>0) slot other than the active -> uniform over eligible bench | **new** `roar_target[2]` (party slot 0-5, law uniform over the target side's eligible bench, state-dependent support like `hit`) and **new** `roar_level[2]` (0/1 with law k/256 from the formula) or fold the level test into one law | new |
| Protect | `BtlCmd_TryProtection` script.c:5149-5190: success if `sProtectSuccessRate[n] >= RandNext()` (:5128-5133: 0xFFFF, 0x7FFF, 0x3FFF, 0x1FFF = 1, 1/2, 1/4, 1/8 of 65536) **and** the user is not the last to move (`waitingBattlers != 1`) | **new** `protect[2]`, law 2^-n by `protect_count` | new |
| Attract / Cute Charm infatuation | `CHECK_STATUS_STATE_ATTRACT` bcp.c:2669-2690: `RandNext() & 1` true -> acts, false -> immobilized | **new** `infatuation[2]`, 1/2 | new |
| Quick Claw | `speedRand[b] = RandNext()` per turn (bcp.c:852-857, also lib.c:2024 at battle start); activated when `speedRand % (100 / param) == 0` (lib.c:1278-1286, :1344-1352), param 20 -> 1/5 | **reserved** `quick_claw[2]`, law (4,1)/5 | reserved, unused |
| contact procs | `BattleSystem_TriggerAbilityOnHit` lib.c:4200-4374: Static :4214, Flame Body (burn), Cute Charm :4339-4355 each `RandNext() % 10 < 3`; Rough Skin :4259-4271 has **no** roll (maxHP/8 Divide) | **reserved/used** `contact_proc[2,5]`; extend relevance to Flame Body and Cute Charm holders | in use |
| Sand Veil | accuracy x80/100 in sand (bcp.c:2925-2928, ignorable ability) | changes `hit_threshold` only | law only |
| Sand Tomb | `subscript_bind_start.s` `Random 3, 3` | existing `duration` (DURATION_TRAP) | ok |
| Focus Sash, Rough Skin, Chilan/Rindo, Lagging Tail, Big Root, berries, sandstorm chip, Light Screen, Toxic, Curse, Recover, Self-Destruct, Brick Break, Fake Out, Flail, DoubleSlap, Gust, Pay Day, Swift family, Camouflage, Aqua Ring | no random draw | none | - |

Not needed: `flinch_item` (no King's Rock/Razor Fang in the pool or on Roark). Focus Band
(`HOLD_EFFECT_MAYBE_ENDURE`, bcp.c:3387) would need a draw but is not present.

### 3.3 Enumeration width and the cap

- Battle outcomes: `enumerate_outcomes` (enumerate.py:398-430) takes the product of every
  *relevant* element's support (values of probability 0 dropped, enumerate.py:385-395); above
  `WIDTH_CAP = 131072` (enumerate.py:83) it raises `SamplingOnly`, and Stage B values that pair as
  "sampled" (stage_b.py:31-33) instead of exactly. There is no single width: it is per (state,
  action pair). Each new field multiplies K by its support only where its relevance mask is true
  (a never-relevant field keeps its `empty_outcome` value and costs nothing).
- Kaizo Roark today: almost every move has accuracy 100, but accuracy 100 still rolls
  (bcp.c:2976; only accuracy 0 skips it) and Sand Veil in sand makes it 80, so `hit` is not
  single-valued where stages or Sand Veil apply. A typical pair is crit 2 x roll 16 x secondary 2 per side = 4096, x2 for each tie flag.
- After K3b, worst cases on this fight: Skitty DoubleSlap (acc 90: hit 2, two hits of crit 2 x
  roll 16 = 2048) vs Gible Thunder Fang into Cute Charm (crit 2 x roll 16 x secondary 2 x
  secondary2 2 x contact_proc 2 = 256) = 524 288 > cap, before Quick Claw/infatuation/ties. Present
  (4) x Thunder Fang (128) x Quick Claw (2) x contact (2) = 131 072 = exactly the cap. So
  relevance masks must be tight (e.g. Present heal reads no crit/roll; flinch roll only if the
  target has not moved and lacks Inner Focus; paralysis roll only if the target can be
  paralyzed; Thunder Fang into a Limber or already-statused target), or more pairs fall to
  "sampled". This is not a coverage refusal; it is a K4 exactness/quality concern to measure.
- AI outcomes are separate: `AiOutcome` `ENUMERATION_WIDTH = 2 x 2^4 x 12 x 10 = 3840 <=
  MAX_ENUMERATED 4096` (src/nuzline/ai/outcome.py:88, :124-130; test tests/ai/test_ai_outcome.py:179).
  K3b does not touch it (K3c builds the per-game layout).

---

## 4. Turn structure and the missing field-condition pass

### 4.1 Engine today

`step_events` (step.py:1026-1039): `_record_last_move` -> `action_order` -> `_execute_actions`
(each action + faint check, :768-820) -> `end_of_turn` (:983-998: for each side in
`end_of_turn_order` (:970-980), every check of `END_OF_TURN_ORDER` (:944-967) followed by a faint
check) -> `_unlock_sleepers` -> `turn + 1`. The module docstring says the field pass is not
implemented (step.py:47-52). Reference: `ref_step` reference/engine.py:1284-1371, `_end_of_turn`
:1153-1175 (same list).

### 4.2 Decomp order

1. After the last action `BattleControllerPlayer_BranchActions` re-sorts `monSpeedOrder`
   (`BattleSystem_SortMonSpeedOrder`, bcp.c:887) and goes to `BATTLE_CONTROL_CHECK_FIELD_CONDITIONS`
   (:889-891).
2. `BattleControllerPlayer_CheckFieldConditions` bcp.c:932-1254, states in
   `enum FieldCondCheckState` :897-915, each step preceded by AnyFainted / ExpPayout /
   CheckBattleOver (:939-944): REFLECT (sides 0 then 1, `--reflectTurns == 0` ends it, :947-967),
   LIGHT_SCREEN (:969-989), MIST, SAFEGUARD, TAILWIND (:1039-1060), LUCKY_CHANT (:1062-1083), WISH
   (:1085-1111, heals maxHP/2), RAINING (:1113-1136), SANDSTORM (:1138-1161), SUNNY, HAILING, DEEP_FOG
   (:1213-1225), GRAVITY (:1227-1238). Permanent weather (`*_PERM`) never counts down; temporary
   weather decrements `weatherTurns` and ends at 0. Trick Room counts down elsewhere (bcp.c:1823-1826,
   in the turn end).
3. Then `BattleControllerPlayer_CheckMonConditions` bcp.c:1289 over `monSpeedOrder` with
   `enum MonCondCheckState` :1256-1287: INGRAIN, **AQUA_RING** (:1331-1351), ABILITY, USE_ITEM,
   LEFTOVERS, LEECH_SEED, POISON, TOXIC, BURN, NIGHTMARE, CURSE, BIND, ..., DISABLE, ..., TAUNT,
   ..., ITEM_CONDITION. The engine's `END_OF_TURN_ORDER` is this list's implemented subset.

### 4.3 Sandstorm in detail

- The weather step loads `subscript_weather_continues.s`: for each battler by speed order
  (`GetMonBySpeedOrder`, `GoToIfValidMon`) `EndOfTurnWeatherEffect` then, if damage, message +
  `BATTLE_SUBSCRIPT_UPDATE_HP` + `BATTLE_SUBSCRIPT_FAINT_MON` inline; Magic Guard skips (script
  line `CheckAbility ... ABILITY_MAGIC_GUARD`).
- `BtlCmd_EndOfTurnWeatherEffect` script.c:5561-5647, sand branch :5573-5583: inside `NO_CLOUD_NINE`
  (include/battle/common.h:63-64: no **alive** battler with Cloud Nine or Air Lock), chip =
  `BattleSystem_Divide(maxHP * -1, 16)` (lib.c:3599: truncation toward 0, at least 1) when neither
  type is Rock, Steel or Ground, curHP > 0, ability != Sand Veil (`Battler_Ability`, not
  ignorable) and no `MOVE_EFFECT_NO_WEATHER_DAMAGE` (Dig/Dive semi-invulnerable; unreachable: Kaizo
  Dig is a plain hit and no charge move is supported).
- **Faint timing (load-bearing):** all battlers take their chip inside one subscript; the battle
  end is checked only when the field-check loop resumes (bcp.c:939-944). So if both last Pokemon
  fall to sand it is a draw (= loss, `_faint_check` step.py:696-736) whichever order they fainted
  in. The engine must apply both chips, then one faint check, not a faint check per battler.
- Damage: Rock defender SpD x1.5 already in damage.py:268-271 and reference :352, **but neither is
  gated on Cloud Nine**; the decomp gates it (lib.c:6921-6929, `NO_CLOUD_NINE`). Must fix
  together with turning sand on. Same for rain/sun power (damage.py:298-305, lib.c:7046) and fog
  accuracy (outcome.py:130 vs bcp.c:2925-2935): those stay refused, but the shared code should be
  right.
- Sand Veil accuracy x80/100 (bcp.c:2925-2928), applied after Compoundeyes, before fog
  (outcome.py:128-131). Snow Cloak is the hail twin (:2930-2932); Swift Swim doubles speed in rain
  (lib.c:1228-1238, `NO_CLOUD_NINE`). With only NONE/SANDSTORM reachable these are inert but
  should be modelled in both engines (cheap) so the abilities are EXACT, not waved through.
- Who chips in this fight: all six of Roark's are Rock or Ground types (immune). Player: Prinplup,
  Delibird, Luvdisc, Flaaffy, Shinx, Vulpix, Skitty, Meowth, Goldeen chip; Sandshrew and Rhyhorn
  (Ground), Wormadam-Trash (Steel) are immune, **unless Camouflage made Wormadam Normal**.
- Start condition: Roark `start_condition = "sandstorm"` (K trainers.json 246; sources say "map
  header weather through the trainer -> map join"); Kaizo map header 47 (Oreburgh Gym) weather
  byte = 10 = `OVERWORLD_WEATHER_SANDSTORM` (read from the patched ROM, d_test.narc member 47).
  `cli.py:515, :555, :821` -> `coverage.start_weather` (coverage.py:182-194, `START_CONDITIONS`
  :151-158) -> `make_state(..., start_condition)` -> `initial_arrays` sets `weather` with
  `weather_timer = 0` = permanent (state.py:20, :336-337, :362). Prep threads
  `problem.start_condition` the same way (prep/problem.py:288, states.py:45-61, variants.py:258-266).
  Decomp equivalent: `BattleSystem_TriggerEffectOnSwitch` FIELD_WEATHER, case 10 ->
  `subscript_overworld_sand` (K3a: `modelled`, overlay16.md §2).
- Coverage: set `IMPLEMENTED_WEATHER[SANDSTORM] = EXACT` (coverage.py:132-139); update
  `tests/engine/test_coverage.py::test_only_no_weather_is_supported` :58, the coverage docstring
  :26-34, hooks.py:48-51 (Cloud Nine "has no behaviour"), and the oracle docstring's Cloud Nine
  exclusion (tests/test_damage_oracle.py:9-12); `test_cloud_nine_is_inert`
  (test_m4_abilities.py:121) forces weather NONE so it stays valid.

### 4.4 Light Screen, Reflect timer, Brick Break

- `BtlCmd_TryLightScreen` script.c:4214-...: fails if the side already has it, else
  `lightScreenTurns = NUM_SCREEN_TURNS` (5) + Light Clay's param (unsupported item -> coverage
  refuses). State field `light_screen` (state.py:123) = turns left; `reflect` likewise.
- Damage: Light Screen halves special damage when `criticalMul == 1` and the move's effect is not
  REMOVE_SCREENS (lib.c:7023-7032; Reflect :6982-6991); damage.py:293-296 and reference :378 lack
  the REMOVE_SCREENS exception.
- Countdown: field pass REFLECT then LIGHT_SCREEN, side 0 then 1, before the weather step.
- Brick Break (`effect_script_0186.s`): damage is computed in the script with the screen ignored
  (the exception above); the break is a *direct* effect (`MOVE_SIDE_EFFECT_BREAK_SCREENS`) applied
  in `BattleSystem_TriggerPrimaryEffect` lib.c:1481-1510, which still fires on an immune target
  (`MOVE_STATUS_DID_NOT_AFFECT`, :1497-1499) but not on a miss; `BtlCmd_TryBreakScreens`
  script.c:6550-6567 clears Reflect and Light Screen on the defender's side.
- **Script-before-accuracy order (general, load-bearing):** the move's effect script, including
  `CalcCrit`/`CalcDamage`, Present's roll, Tri Attack's `Random` and Self-Destruct's self-KO,
  runs before `BattleControllerPlayer_TryMove` (accuracy, hit overrides, type chart;
  bcp.c:3206-3214, :3237-3296); primary (direct) effects run after it (:3298-3310), then
  `UseMove`/`UpdateHP`.

---

## 5. Kaizo data facts

### 5.1 Roark (trainer 246, K trainers.json)

ai_flags 23, items [] (no bag), format single, start sandstorm (permanent), friendship 255 each.
IVs from difficulty: `difficulty * 31 // 255` (src/nuzline/data/trainer.py:277-278) = 30 for
251-254, so all IVs 30. EVs 0.

| slot | species (types) | Lv | nature | gender | ability (slot 0) | item | moves |
|---|---|---|---|---|---|---|---|
| 0 | Bonsly 438 (Rock) | 15 | Impish | M | Rock Head | Focus Sash | Stealth Rock, Selfdestruct, Brick Break, Accelerock |
| 1 | Lileep 345 (Rock/Grass) | 13 | Naive | M | Solid Rock | Big Root | Sludge Bomb, Giga Drain, AncientPower, Recover |
| 2 | Gible 443 (Dragon/Ground) | 14 | Gentle | M | Sand Veil | Chesto Berry | DragonBreath, Flamethrower, Thunder Fang, Earth Power |
| 3 | Corsola 222 (Water/Rock) | 14 | Quirky | F | Solid Rock | Rindo Berry | Recover, Water Pulse, Light Screen, Accelerock |
| 4 | Shuckle 213 (Bug/Rock) | 13 | Naive | M | Solid Rock | Berry Juice | Toxic, Accelerock, Curse, Sand Tomb |
| 5 | Cranidos 408 (Rock) | 16 | Adamant | M | Rock Head | Lum Berry | Head Smash, Zen Headbutt, Crunch, Rock Polish |

Kaizo base-stat changes vs vanilla: Corsola 55/55/85/65/85/35 -> 65/95/105/65/105/35; Shuckle HP
20 -> 50; Cranidos 67/125/40/30/30/58 -> 67/125/50/10/40/58 (others unchanged). Ability changes as
in §1.4.

### 5.2 Kaizo vs vanilla for every K3b move/item/ability

Moves: see the "Kaizo vs vanilla" columns in §1.3. Notable semantic changes: Sand Tomb is
Whirlpool's effect 261 (bind + double vs Dive; Dive unsupported, so it is Bind with
power_mul 10), DoubleSlap is HIT_TWICE (fixed 2 hits) not 2-5, Aurora Beam is a never-missing
plain hit, Fake Out has +3 priority, Swift-family moves (accuracy 0) never miss; accuracy-100 moves such as Toxic still roll (bcp.c:2976; Sand Veil in sand: 80). None of their
`be_seq` scripts changed (changed members are only 48, 88, 104, 141, 182; data/kaizo/rom_diff.toml
:590-622). Items: every K3b item record equals vanilla (§1.5; only King's Rock's param changed,
10 -> 11, and it is not in play). Abilities: names identical; code unchanged.

### 5.3 Terrain (Nature Power / Camouflage)

- Decomp: `SetBackgroundAndTerrain` src/field_battle_data_transfer.c:496-505 reads
  `MapHeader_GetBattleBG(mapHeaderID)`; `CalcTerrain` :467-494 checks the tile behaviour under
  the player (`TerrainCollisionManager_GetTileBehavior`, low 8 bits of the land-data permission):
  ice, tall grass, sand, snow, mud, cave floor, surfable; otherwise
  `sTerrainForBackground[background]` :441-466.
- Battle side: `BtlCmd_GetTerrainMove` script.c:8036-8048 (`terrain > TERRAIN_SPECIAL` ->
  TERRAIN_SPECIAL = TERRAIN_AARON, include/constants/battle/terrain.h), table
  include/data/terrain/to_move.h; `BtlCmd_TryCamouflage` script.c:7997-8022 (fails on Multitype or
  if the user already has the type; sets type1 = type2 = type), table to_type.h; Secret Power
  uses to_secondary_effect.h.
- **Roark's gym, read from the patched ROM** (cache `.cache/rom/kaizo-b787e743b5fa78e1.nds`):
  map header 47 = `MAP_HEADER_OREBURGH_CITY_GYM` (generated/map_headers.txt line 48) in
  `debug/cb_edit/d_test.narc` member 47: matrix 113, weather 10 (vanilla: CLEAR), mapType 4,
  battleBG 6 = `BACKGROUND_INDOORS_1` (vanilla header include/data/map_headers.h:1078-1099 has the
  same BG and matrix). Matrix 113 is one map, land data 229 (unchanged in both ROMs); its 1024
  tiles carry behaviours {0x00 NONE: 1013, 0x30 BLOCK_EASTWARD: 5, 0x31 BLOCK_WESTWARD: 5, 0x65: 1},
  none of which `CalcTerrain` matches, so **terrain = TERRAIN_BUILDING on every tile** ->
  Nature Power = Tri Attack (161; Kaizo 95 BP, 30 % burn/freeze/para), Camouflage = Normal.
  `CalcTerrain`, `SetBackgroundAndTerrain` and the terrain tables are not in rom_diff (unchanged).
  The plan should turn this one-off reading into a tested ROM reader (map header battleBG + every
  tile behaviour of the map's land data -> one terrain per map, or "mixed").
- What K3b must do about the two `not_reached` entries (rom_diff.toml:35-40, :115-120,
  `when = { move_effect = [173, 197, 213] }`): conditions are OR-ed terms (coverage.py:311-324;
  `when` keys: format, area_condition, move_effect, ability, held_item, map_weather, map_header,
  trainer_items, trainers), so there is no way to say "these effects except at Roark". Options:
  (a) add per-trainer terrain data (`Trainer.terrain`, from the trainer -> map join that already
  feeds `start_condition`), model terrain in the engine (a battle constant passed like
  `start_condition`, or a scalar state field), flip both entries to `modelled`, and make coverage
  refuse Nature Power / Camouflage / Secret Power only where a trainer's terrain is unknown or
  mixed; or (b) leave them `not_reached` and leave Wormadam's two moves refused (fails Goal 1).
  Option (a) is required for "nothing missing". It also needs vanilla terrain (vanilla header 47
  is the same) so vanilla Nature Power is consistent.
- Nature Power mechanics: `effect_script_0173.s` prints, `GetTerrainMove`, then
  `GoToMoveScript FALSE` (script.c:2111-2134): `moveCur = Tri Attack`, defender re-picked for Tri
  Attack's range (Nature Power's own range is RANGE_SINGLE_TARGET_SPECIAL, which
  `_SELF_DEFENDER` step.py:139-146 treats as self), Tri Attack's script runs, PP is Nature
  Power's. Because `moveCur` changes, `movePrevByBattler`/`moveHit` record Tri Attack (check in
  `UpdateMoveBuffers`), which matters for Disable and for the AI's seen-moves list (the player's
  slot holds Nature Power, so `_record_last_move` step.py:241-259 would not match it).

---

## 6. Mechanic notes for the plan (decomp facts not covered above)

- **Obedience** (rom_diff.toml:550-554, `unknown -> K3b`): Kaizo always obeys, no RNG. The engine
  has no obedience check and no OT/badge state (`BattleState` state.py:70-150). K3b: a test that
  pins this (e.g. no state field for OT or badges and `step` reads none; a non-OT L16 player mon
  with 0 badges uses its chosen move under an all-zero outcome), flip the entry to `modelled` with
  that test cited, rebuild Kaizo tables (unclassified count 2 -> 1).
- **Self-Destruct** (`effect_script_0007.s`): Damp check (unsupported ability), user HP set to 0 in
  the script, i.e. before accuracy/Protect/type chart, so the user faints even on a miss or into a
  Ghost; a fight with no target fails earlier (HasNoTarget) and the user survives. Defense halved
  in the damage formula (lib.c:6943-6945). User and target faint in the same action: both-last ->
  draw (loss).
- **Recover**: `subscript_recover_half_max_hp.s` maxHP/2 (Divide), `subscript_recover_hp.s` fails
  at full HP.
- **Curse** (`effect_script_0109.s`): non-Ghost user: Speed -1, then Attack +1, Defense +1
  (`subscript_curse_normal.s`); a Ghost-type user takes the Ghost branch (HP cost, curse status),
  so coverage must refuse a Ghost-type Curse user (a per-mon check, not just the effect registry)
  or model it. Needs a multi-stat effect family (params hold only one stat today, ids.py:15-17).
- **AncientPower** (`subscript_boost_all_stats.s`): +1 Atk, Def, Spe, SpA, SpD on the user at the
  10 % roll (indirect, TO_ATTACKER).
- **Toxic**: map 33 -> `STATUS_FOE (Status.TOXIC)`; immunities in `subscript_badly_poison.s` match
  `can_have_status` (effects.py:208-220); counter and damage exist (step.py:860-868).
- **Sand Tomb / Whirlpool** (261): `SYSCTL_HIT_DURING_DIVE`, power x2 vs underwater (unreachable),
  then `MOVE_SUBSCRIPT_PTR_BIND_TARGET` with CHECK_HP_AND_SUBSTITUTE: identical to `EffectId.TRAP`
  (effects.py:355-365).
- **Swift / Shock Wave / Aurora Beam** (17): a bare hit; never-miss comes from accuracy 0. Map to
  PLAIN. **Pay Day** (34): damage + money only (`subscript_pay_day.s`): PLAIN. **Gust** (149):
  power x2 only vs Fly/Bounce (unsupported): PLAIN with a note.
- **DoubleSlap** (44): `SetMultiHit 2`; reuse the multi-hit loop with a fixed count
  (effects.py:159-163); accuracy once.
- **Fake Out** (158): fails unless `fakeOutTurnNumber == totalTurns` (`fake_out_turn` already in
  state, step.py:224-232); 100 % flinch through the normal flinch subscript (Inner Focus,
  target-not-moved rules).
- **Flail** (99): `BtlCmd_CalcFlailPower` script.c:4958-4970, `App_PixelCount(cur, max, 64)`
  (src/unk_0208C098.c:41-49: `cur*64/max`, at least 1 if cur>0), table script.c:4941-4948
  (<=1:200, <=5:150, <=12:100, <=21:80, <=42:40, else 20). The AI computes no damage at all for
  moves with table power <= 1 (trainer_ai.c:1036, :1554, :1605, :2828), so Flail is never
  scored by damage; the engine matches because `ai_damage` is gated by `is_damaging`. HP cuts: one per band boundary.
- **Present**: heal path loads `subscript_present_heal.s` (fails on a miss; heals the target by
  maxHP/4, `SYSCTL_IGNORE_IMMUNITIES`, so even a Ghost is healed); damage path power 40/80/120.
- **Protect**: also needs `moveProtect[battler]` (last executed move; STATE.md:85-87, :393) to reset
  the chain when the last move was not Protect/Detect/Endure, a per-turn "protecting" flag, and
  blocking of moves with `MoveFlag.PROTECT` in `CheckMoveHitOverrides` (bcp.c:3001-). Fake Out
  (+3) vs Protect (+3) ties on speed.
- **Attract / Cute Charm**: `BtlCmd_TryAttract` script.c:5727-5742 (fails if same gender, either
  genderless, or already infatuated); Oblivious blocks (unsupported). The infatuation is tied to
  the source battler and is cleared when the source leaves (lib.c:2090-2091 switch, :2175-2176
  faint). Status-check order: ... CONFUSION, PARALYSIS, ATTRACT, BIDE (bcp.c:2378-2392). Cute Charm
  needs both Pokemon alive (lib.c:4339-4355) and the attacker not already infatuated.
- **Roar**: `subscript_force_target_to_switch_or_flee.s`: fails vs Suction Cups / Ingrain
  (unsupported) and vs Soundproof (Roar is in `sSoundMoves`, lib.c:3473-3475, 3523-3527); `TryWhirlwind`; Natural Cure on the outgoing (`TryRestoreStatusOnSwitch`
  script.c:9139); the drawn slot comes in, "dragged out", hazards check
  (`BATTLE_SUBSCRIPT_HAZARDS_CHECK`), then the switch-in effects at move end. Fails if the target
  side has no other eligible Pokemon. Roar is priority -6, so it acts last.
- **Camouflage**: changes the battler's types (BattleMon type1/2); `BattleState.types` is per
  party slot [2,6,2] and is the persistent party data (state.py:78), so the change needs a
  per-side volatile override that `_reset_battler`/switch clears (`_VOLATILE` step.py:111-136), and
  every type reader (damage STAB/chart, status immunities, sand chip, Stealth Rock, AI) must read
  the effective types.
- **Aqua Ring**: a volatile flag (`MOVE_EFFECT_AQUA_RING`, cleared on switch), fails if already set
  (`effect_script_0251.s`); end-of-turn heal maxHP/16 (Divide) when not full and alive, second in
  the mon-condition order (bcp.c:1331-1351), Big Root x(100+30)/100 (`subscript_aqua_ring_heal.s`).
- **Rough Skin** lib.c:4259-4271: attacker alive, contact, damage taken, maxHP/8 (Divide); not
  ignorable via Mold Breaker (uses `Battler_Ability` of the defender). **Flame Body**: burn at 30 %.
  **Limber**: paralysis immunity in `subscript_paralyze.s:5` (ignorable: Mold Breaker pierces it
  for moves) and a cure by `BattleSystem_RecoverStatusByAbility` lib.c:4378-4400 (switch-in and move
  end, like Insomnia; hooks.py:109-134). Also a prep prestatus blocker (problem.py:268-275).
- **Adaptability**: STAB x2 (lib.c:2592-2597). **Solid Rock**: SE damage `Divide(damage*3, 4)` after
  the chart (lib.c:2652-2656, ignorable). Both feed the AI's damage automatically (it shares
  `BattleSystem_CalcMoveDamage` + `ApplyTypeChart`, trainer_ai.c:3081-3109; engine shares
  `_pre_random`/`_finish`). smogon gen4 models both (node_modules/@smogon/calc/dist/mechanics/gen4.js
  :186-197) with a float 0.75; compare rounding against `Divide`.
- **Focus Sash** (`HOLD_EFFECT_ENDURE`): `BattleControllerPlayer_UpdateHP` bcp.c:3386-3404 (full HP ->
  survive at 1), also `BtlCmd_CheckHoldOnWith1HP` script.c:9098-9124 used by the confusion self-hit,
  Pursuit and Future Sight subscripts; consumed in `subscript_move_followup_message.s:12-19`
  (`RemoveItem`, so `item_consumed`). A full-HP hit sets the per-action `focusItemActivated` flag (cleared at action end, bcp.c:3998), so every later lethal hit of the same move is endured; multi-hit moves do not break the Sash. Pursuit and the confusion self-hit use the plain full-HP check.
- **Quick Claw / Lagging Tail**: in `BattleSystem_CompareBattlerSpeed` lib.c:1188-1445; they apply
  among equal priority only (:1401-1440), Quick Claw beats Lagging Tail beats Stall beats Trick
  Room/speed; both also apply to `SortMonSpeedOrder` (ignoreQuickClaw=TRUE still evaluates the
  flag; only the moveEffectsData write is skipped), i.e. to the end-of-turn order
  (`end_of_turn_order` step.py:970-980). Each side's `speedRand` is one draw per turn.
- **Status berries** (Chesto, Pecha incl. toxic, Lum incl. confusion): `BattleSystem_TriggerHeldItem`
  lib.c:4576-4700 (SLP :4608, PSN :4615, STATUS :4662) and `..._OnStatus` lib.c:4967-5100; same
  call sites as Oran/Sitrus. **White Herb**: STATDOWN_RESTORE lib.c:4875, :5088.
- **Resist berries** (Rindo SE Grass, Chilan any Normal-type hit): `subscript_type_resist_berry.s`,
  called from `subscript_update_hp.s` when the sprite-blink flag is clear and battlers are waiting:
  halves the **final** HP change (after Focus Sash's cap) with `DivideVarByValue` and removes the
  berry. Hold effects 19-34 map to types, 35 = Chilan (any Normal move; a Normalize user makes
  every move Normal).
- **Big Root**: drain, Leech Seed, Aqua Ring, Ingrain heal x(100 + 30)/100 after the base amount
  (`subscript_drain_half_damage_dealt.s`: half via Divide, then the boost, no min).
- **Berry Juice**: HOLD_EFFECT_HP_RESTORE (same branch as Oran, param 20) but Pluck/Bug Bite read
  the item's *pluck effect* (`BattleSystem_PluckBerry` lib.c:5539-...; non-berries are not eaten,
  :5833). items.json has no pluck-effect column, so supporting new berries next to the supported
  `EAT_BERRY` effect needs that data or an explicit refusal of Pluck vs those items.

---

## 7. Risks and sizes

S = a task or two, M = several tasks with state/outcome/observe changes, L = needs design inside
the plan.

| item | size | why |
|---|---|---|
| Field-condition pass (screens, tailwind, lucky chant, mist, safeguard, wish, gravity counters, weather incl. hail/rain/sun steps) in both engines | M | strategies already draw weather and timers, so both engines must agree on all of it; sand faint timing (§4.3) |
| Sandstorm EXACT (chip, immunities, Sand Veil, Cloud Nine gating in damage/accuracy/speed, coverage) | M | Cloud Nine gating touches existing damage code; vanilla must stay unchanged |
| Light Screen + Brick Break | S | timers exist; REMOVE_SCREENS exception |
| Obedience test + rom_diff flip | S | |
| Toxic, Sand Tomb (TRAP), Swift family, Pay Day, Gust, Recover, DoubleSlap, Fake Out, AncientPower | S each | mappings or small families |
| Curse | S (+ Ghost refusal) | multi-stat family; per-mon coverage check |
| Self-Destruct | S-M | self-KO before accuracy; double-faint ordering |
| Thunder Fang | M | new Outcome field + laws + relevance + strategies |
| Flail | S-M | power by pixel bands, no AI damage for power <= 1, HP cuts, oracle check |
| Present | M | new field, heal branch (no crit/roll), AI skips damage for power <= 1 |
| Protect | M | new field, `moveProtect` chain state, last-to-move rule, blocking via MoveFlag.PROTECT, Fake Out interplay |
| Aqua Ring | S-M | volatile flag + end-of-turn step + Big Root |
| Attract + Cute Charm infatuation | M-L | source-tied volatile, cleared on source switch/faint, new 1/2 draw, status-check order, observe |
| Roar | L | state-dependent target law over the opponent's bench, level draw, forced switch-in with hazards/Intimidate/berries mid-turn, observation of the dragged-out Pokemon, interaction with the AI's knowledge and replacement code |
| Nature Power (+ terrain pipeline) | L | ROM terrain reader per trainer, coverage/rom_diff redesign, move calling (moveCur swap, defender re-pick, last-move/AI-seen semantics), Tri Attack 3-way draw |
| Camouflage | M | effective-type override touching every type reader |
| Abilities: Rough Skin, Flame Body, Limber, Swift Swim, Snow Cloak, Adaptability, Solid Rock, Sand Veil | S each | Limber also prestatus/recover hooks |
| Quick Claw | M | reserved field, order + end-of-turn order + enumerate relevance (ties) |
| Lagging Tail | S-M | order + end order |
| Focus Sash | S-M | three damage sites (move, confusion self-hit, Pursuit), consumption, HP cut |
| Status berries (Chesto, Pecha, Lum), White Herb, Big Root, Berry Juice | S each | Pluck data gap (§6) |
| Resist berries (Rindo, Chilan) | M | applied on the final HP change after Focus Sash, consumption, observation |
| Pool bag-item rows + no silent drop in `bag_resources` | S-M | needs raw hold effect in tables |
| Outcome width growth (§3.3) | M (measure) | more "sampled" pairs; tighten relevance |
| Vanilla invariance | cross-cutting | vanilla Roark has none of these mechanics; keep seeded draws via `fold_in`; Cloud Nine fix must not change vanilla no-weather results |

Suggested order: (1) obedience; (2) field pass + Light Screen/Brick Break + sandstorm/Sand Veil/
Cloud Nine; (3) data-only and small move families; (4) abilities; (5) items (non-random, then
Quick Claw/Lagging Tail) + pool bag rows; (6) Outcome additions one at a time (Thunder Fang,
Protect, Present, infatuation); (7) Camouflage, Roar, Nature Power + terrain last.
