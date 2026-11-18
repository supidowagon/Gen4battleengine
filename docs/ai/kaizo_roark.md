# Kaizo Roark's trainer AI, transcribed from the decomp

This is the specification for the Kaizo policy of the trainer AI in `src/nuzline/ai`
(milestone K3c). It is built the way `docs/ai/roark.md` is, and it states only what
differs from that vanilla transcription; every rule not restated here is roark.md's.
Paths are relative to `external/pokeplatinum`. `script.s` means
`src/battle/trainer_ai/script.s` and `trainer_ai.c` means
`src/battle/trainer_ai/trainer_ai.c`. Kaizo data comes from the Kaizo tables
(`data/intermediate_kaizo/*.json`, built into `data/gen4_platinum_kaizo`).

K3c wrote it task by task: sections 1-2 come from Task 2, draws and the outcome layout (8)
from Task 3, the commands (7) from Task 4, the per-flag paths (5) from Tasks 5-6, switching
(11) from Task 7, and the rest (3, 4, 6, 8.7, 9, 10, 12, 14) from Task 8, which closed K3c.
The sections keep roark.md's numbers, so a rule can be compared section by section.

## 1. Flags

- `data/intermediate_kaizo/trainers.json`, trainer 246 (Leader Roark): `ai_flags = 23`,
  `0b10111`.
- Line n+2 of `generated/ai_flags.txt` is bit n (roark.md 1), so 23 = BASIC (bit 0) +
  EVAL_ATTACK (bit 1) + EXPERT (bit 2) + RISKY (bit 4, `AI_FLAG_RISKY`, line 6).
- `FlagTable` (`script.s:18-23`) maps bit 4 to `Risky_Main` (`script.s:6521`).
- `TrainerAI_MainSingles` runs the flags in bit order (`trainer_ai.c:297-310`): BASIC,
  EVAL_ATTACK, EXPERT, then RISKY.
- **Per-trainer flag sets (E1 Ruling R2; replaces plan K3c Ruling 10).** `TrainerAI_Init`
  copies the trainer's `aiMask` (`trainer_ai.c:249`), and `TrainerAI_MainSingles` runs a
  flag's script only when its bit is set, low bit first, each over all four slots
  (`:297-308`). The Kaizo policy has one `AiLayout` per flag set (`layout.kaizo_layout`;
  `AiLayout.flags`, a static jit argument, so a fight traces only its own flags' scripts):
  `policy._flags` runs the set bits in bit order. The fight's trainer reaches the layout
  through `Tables.ai_flags`, which the fight's entry points set (`layout.fight_tables`:
  `cli`, and `check_coverage` for its `ai_refusals`); `layout_of` refuses Kaizo tables whose
  flags are unset. The search's program caches key on `AiLayout.key` (policy and flag set:
  `expand.ai_rows`, `leaves`, `stage_a`). Implemented sets: every set of BASIC, EVAL_ATTACK,
  EXPERT, SETUP_FIRST_TURN, RISKY, PRIORITIZE_EXTREMES, CHECK_HP and WEATHER
  (`layout.KAIZO_FLAG_BITS`; SETUP_FIRST_TURN and WEATHER since E1 batch 2b, 5.5-5.6,
  PRIORITIZE_EXTREMES and CHECK_HP since 2c, 5.7-5.8), among them 0x1, 0x5, 0x7, Roark's
  0x17, 0xd, 0xf, 0x1f, 0x27, 0x10f, 0x207 and 0x20f: every singles set of the Roark split
  and all Gardenia singles but Cheryl's 0x58b (TAG_STRATEGY, HARRASSMENT).
  `KAIZO` is Roark's set and stays bit-identical. The scoring's cost follows the fight's read
  bound (8.3, E1 Ruling R6): `2^slot_reads` scorings per decision, 2^7 for Roark.
- Coverage (`engine/coverage.py`) registers the flags per policy: `IMPLEMENTED_AI_FLAGS` is
  `{0, 1, 2}` for vanilla and `{0, 1, 2, 3, 4, 5, 8, 9}` for Kaizo. A Kaizo trainer is
  refused exactly by its unimplemented flags, one "AI_FLAG_..." line each; every singles set of
  the Roark and Gardenia splits gives none (`tests/engine/test_coverage.py
  test_kaizo_flag_set_has_no_flag_line`). Vanilla keeps its one set: a vanilla trainer
  without all of BASIC, EVAL_ATTACK and EXPERT is refused (`POLICY_AI_FLAGS`), and one with
  RISKY by that flag. Kaizo Roark's flag lines are all EXACT.

## 2. Differences from vanilla

**Where they live.** The game's `Tables` carry `ai_policy` (0 vanilla, 1 kaizo; built
from `game_constants.POLICY`, whose Kaizo entry is `"kaizo"`). `nuzline.ai.layout.layout_of`
reads it on concrete tables and gives the game's `AiLayout`. The AI's traced functions
take the layout as a static argument. The public functions of `ai.policy` resolve it
themselves. Under a trace without a layout they take `VANILLA` and refuse any other
policy's tables (`_Refused.policy`).

**The AI is the vanilla script plus one word.** overlay14, which carries `gTrainerAITable`,
is the same size in both ROMs. Its one changed script range is the word at 0x022249CC.
Every other change in overlay14 is a relocated call into overlay16 (K3a, overlay14
summary sections 1 and 3; `docs/kaizo/rom_diff.md`). So no script command, damage
estimate, switch rule or item rule of the AI's C code differs. Kaizo's different
behaviour comes from that word and from Kaizo's *data*: its moves carry their own power,
type and battle effect, read through the same code.

**2.1 The Dry Skin word.**
- Vanilla: `Basic_CheckForImmunity`'s last compare (`script.s:78`) tests
  `ABILITY_LEVITATE` (26) a second time, after `:77` already took Levitate to
  `Basic_CheckGroundAbsorption`. It never matches (the decomp marks it as a bug).
- Kaizo tests `ABILITY_DRY_SKIN` (87) there (`data/kaizo/rom_diff.toml`,
  `overlay14 0x022249CC-0x022249CD`, now `modelled`).
- On a match `Basic_CheckWaterAbsorption2` (`script.s:106-109`) runs. If
  `LoadTypeFrom LOAD_MOVE_TYPE` is Water, the move scores -12 (`ScoreMinus12`).
- The ability compared is the one `LoadBattlerAbility AI_BATTLER_DEFENDER`
  (`script.s:71`) loaded. That is `AICmd_LoadBattlerAbility` (`trainer_ai.c:1170-1198`): the
  known ability, else a coin-flip guess between the species' two abilities (roark.md 9.2).
  It is not `AICmd_LoadAbility` (`trainer_ai.c:2654-2661`, the true ability), as the K3a
  overlay14 notes and the old `rom_diff.toml` note said.
- Code: `AiLayout.water2_ability` (26 vanilla, 87 Kaizo), read by `policy._basic`. Test:
  `tests/ai/test_layout.py::test_dry_skin_compare_per_policy`.

**2.2 Transcribed moves by battle effect.**
- Vanilla keeps its refusal set by move id (`commands.MOVE_EFFECTS`).
- The Kaizo policy reads a move's script paths by its battle effect,
  `MOVE_DATA(move).effect` (`tables.move_ai_effect`; `IfCurrentMoveEffectEqualTo`,
  `trainer_ai.c:1441`). `commands.move_effect(tables, move)` reads that array in both
  policies. Kaizo refuses an opponent move whose effect is not in `KAIZO.transcribed`.
- That set started as the effects of the vanilla-transcribed moves: HIT (0), DEF_DOWN
  (19), FLINCH_HIT (31), DEF_DOWN_2 (59), HIT_BEFORE_SWITCH (128) and STEALTH_ROCK (266).
  Task 5 added the effects of all of Kaizo Roark's moves (section 5) with their BASIC and
  EVAL_ATTACK paths, and Task 6 their EXPERT and RISKY paths (5.3, 5.4). No move of Kaizo
  Roark's party is refused since.
- The script-roll sites are keyed by battle effect too (section 8, K3c Task 3), so a move is
  transcribed by its effect alone. Example: Accelerock (205, effect HIT) takes Rock Throw's
  paths and its EvalAttack quad roll (`script.s:6383`).
- RISKY (`script.s:6521-6531`) scores only the effects of `Risky_RiskyEffects`
  (`script.s:6536-6561`). Of Roark's effects it lists Selfdestruct (HALVE_DEFENSE, 7,
  `:6538`) and AncientPower (RAISE_ALL_STATS_HIT, 140, `:6550`) (5.4).

**2.3 No damage estimate for the no-calc effects.** These hold in both policies; only
Kaizo data reaches them on a transcribed path.
- `TrainerAI_CalcAllDamage` (`trainer_ai.c:2827-2828`), `FlagMoveDamageScore`
  (`:1035-1036`) and `IfCurrentMoveKills` (`:1553-1554`) estimate damage only for a move
  that meets two tests: it has power greater than 1, and its effect is not in
  `sNoDamageCalcMoveEffects` (`trainer_ai.c:31-46`).
- A move that fails them estimates 0. It gets `AI_NO_COMPARISON_MADE` and never counts as
  the highest damage, and it never "kills".
- Kaizo Selfdestruct (120, HALVE_DEFENSE, power 200) and Head Smash (457, RECOIL_HALF,
  power 150) are in that table. So Kaizo Cranidos's highest-damage move is Crunch or Zen
  Headbutt, not Head Smash (plan K3c Review Focus 1).
- Code: `commands.is_damaging` and `commands.NO_DAMAGE_CALC_EFFECTS`. Test:
  `test_kaizo_highest_damage_ignores_head_smash`.
- No vanilla-transcribed move has such an effect, so vanilla is unchanged
  (`tests/ai/test_policy.py::test_vanilla_distribution_unchanged`).

**2.4 The AI damage's move-id cases, and the refusals left (E1 batch 13).** A move whose
effect is in `sAltPowerMoveEffects` (`trainer_ai.c:48-61`; `commands.ALT_POWER_EFFECTS`) is
damage-estimated whatever its power (`:1035`, `:1553`, `:2827`; `commands.is_damaging`).
`TrainerAI_CalcDamage` then takes a move-id case (`:2885-3071`), whatever the move's effect,
or the default, the table's power and type (`:3074-3078`; `BattleSystem_CalcMoveDamage`
uses the table's power for 0 and the table's type for TYPE_NORMAL, `battle_lib.c:6673-6684`):
- modelled (`engine.damage.ai_case_power`, `ai_case_fixed`): Return and Frustration by the
  attacker's friendship (`:3018-3026`; Kaizo's have effect 0 and still take them), Low Kick
  and Grass Knot by the defender's Pokedex weight (`:3055-3071`, `weight_to_power.h`), and
  the fixed Dragon Rage 40, Sonic Boom 20, Seismic Toss and Night Shade the level
  (`:3005-3012`, `:3051-3053`), which skip `CalcMoveDamage` and take the chart with
  `SYSCTL_IGNORE_TYPE_CHECKS` (no STAB or multiplier, an immunity 0), then the variance;
  every other alt-power move takes the default (Kaizo Fury Cutter, 210, with effect 88);
- Hidden Power (E1 batch 14): its IV power and type (`:2973-2993`), and its AI type
  everywhere the AI types a move (`TrainerAI_MoveType`, `:3208-3220`;
  `engine.damage.ai_move_type`: the effectiveness class, the switch's chart, `moveHit`), a
  bench member's from its own IVs (`Move_CalcVariableType`, `battle_lib.c:8162-8175`;
  `ai.switching.party_move_types`: the switch's bench check and both stages of the
  post-KO replacement, `:7998`, `:8051`);
- Psywave and Magnitude (E1 Roark close batch, G-39): a `RandNext` in the estimate
  (`:3014-3016`, `:3028-3048`), each damage command a read of its own (8.7). Kaizo Triple Axel
  sits on MOVE_PSYWAVE's id and Bulldoze on MOVE_MAGNITUDE's;
- refused (`commands.CALC_DAMAGE_MOVES`): Natural Gift and Judgment (item data), Gyro Ball
  (`monSpeedValues`, left by the last speed compare, `battle_lib.c:1372`);
- a move id that `TrainerAI_MoveType` takes by case (`trainer_ai.c:3132-3222`;
  `commands.MOVE_TYPE_MOVES`): Natural Gift, Judgment, Weather Ball.

In vanilla every such move is already refused as untranscribed, so the vanilla refused set
is unchanged.

**2.5 Ids.** The script's and `trainer_ai.c`'s ids are the same numbers in both ROMs (one
script word changed; the C code only relocated).
- So the AI's id lists keep their decomp ids in Kaizo. `game_constants` checks each one
  against the Kaizo names (scope `"kaizo_policy"`):
  - Fissure 90, Horn Drill 32, U-turn 369, Roar 46, Whirlwind 18;
  - the 11 Soundproof ids (`script.s:121-131`);
  - Gyro Ball 360 and Trick Room 433 (`script.s:3472-3473`);
  - the switching move ids;
  - the flat layout's site table's ids;
  - Dry Skin 87.
- A Kaizo-renamed slot on one of these lists fails `load_game`.
- The id cases of 2.4 are compared by id whatever a game names the slot. They are checked
  against vanilla names only (Kaizo's 149 is Triple Axel, and it still takes Psywave's
  case).

## 3. One decision, start to finish

The steps are roark.md 2's, with RISKY added and the Kaizo party and bag. At each turn's
command selection:

1. `TrainerAI_PickCommand` (`trainer_ai.c:3989`):
   1. `TrainerAI_ShouldSwitch` (`:3894`). If it returns TRUE the AI switches (section 11),
      and the decision ends.
   2. Otherwise `TrainerAI_ShouldUseItem` (`:4056`). Kaizo Roark's bag is empty, so it never
      returns TRUE (section 12).
   3. Otherwise the command is FIGHT.
2. For FIGHT, `TrainerAI_Main` (`trainer_ai.c:258`):
   1. The defender is the player's active battler, `attacker ^ 1` in singles, with no draw
      (`battle_lib.c:4193-4195`).
   2. `TrainerAI_Init` (`:211`): every slot starts at 100 (`:223-230`), a slot that
      `BattleSystem_CheckInvalidMoves` flags at 0 (`:234-238`), and the four simulated damage
      rolls are drawn (`:240`; section 8).
   3. `TrainerAI_MainSingles` (`:286`) records the player's last move (`:295`; section 9),
      runs BASIC, EVAL_ATTACK, EXPERT and RISKY in bit order (`:297-310`; section 1), each
      over the four slots (`TrainerAI_EvalMoves`, `:483-525`), and picks among the top scores
      (`:315-338`; section 10).

**Scores.** `moveScore` is `s8` (`include/battle/ai_context.h:14`). `AddToMoveScore` raises a
negative result to 0 and has no upper clamp (`trainer_ai.c:576-586`). On Kaizo Roark's paths
BASIC only subtracts (5.1), EVAL_ATTACK adds at most +4 (the kill bonus, `:6401` -> `:6411`;
the +6 of `BATTLE_EFFECT_PRIORITY_1`, `:6400`, is no Roark move's, 5.2), EXPERT at most +4
(Expert_Curse with Gyro Ball or Trick Room known, 5.3; Shuckle knows neither, so +3) and RISKY
+2 (`:6530`). So a score is at most 110, below 127, and int32 arithmetic gives the same values.

The result is encoded as in roark.md 2: a move slot 0..3, or a switch `4 + party slot`.

## 4. Kaizo Roark's Pokemon

From `data/intermediate_kaizo/trainers.json`, trainer 246 (the ROM's trainer party, decoded
by the patched `TrainerData_BuildParty`, `docs/kaizo/rom_diff.md`). Every member has ability
slot 0. Types, abilities and items are the Kaizo tables' (`species.json`, `abilities.json`,
`items.json`):

| Party slot | Pokemon | Level | Types | Ability (slot 0) | Held item | Moves (slot order) |
|---|---|---|---|---|---|---|
| 0 | Bonsly | 15 | Rock | Rock Head | Focus Sash | Stealth Rock, Selfdestruct, Brick Break, Accelerock |
| 1 | Lileep | 13 | Rock/Grass | Solid Rock | Big Root | Sludge Bomb, Giga Drain, AncientPower, Recover |
| 2 | Gible | 14 | Dragon/Ground | Sand Veil | Chesto Berry | DragonBreath, Flamethrower, Thunder Fang, Earth Power |
| 3 | Corsola | 14 | Water/Rock | Solid Rock | Rindo Berry | Recover, Water Pulse, Light Screen, Accelerock |
| 4 | Shuckle | 13 | Bug/Rock | Solid Rock | Berry Juice | Toxic, Accelerock, Curse, Sand Tomb |
| 5 | Cranidos | 16 | Rock | Rock Head | Lum Berry | Head Smash, Zen Headbutt, Crunch, Rock Polish |

- The bag is empty (`items: []`). `BattleControllerPlayer_InitAI` copies only the trainer's
  non-empty item slots into `trainerItems` (`battle_controller_player.c:4796-4812`), so
  `trainerItems` stays zero and `trainerItemCounts` 0.
- The battle starts in a permanent sandstorm (`start_condition: sandstorm`,
  `area_condition.permanent`), on BUILDING terrain (9: line 10 of
  `generated/battle_terrains.txt`).
- The moves' battle effects, which the scripts dispatch on, are in section 5.

## 5. What each flag's script does with Kaizo Roark's moves

Kaizo Roark's moves, by Kaizo id, with the battle effect the scripts read
(`data/intermediate_kaizo/moves.json` `effect`; `IfCurrentMoveEffectEqualTo`,
`trainer_ai.c:1441`; `BATTLE_EFFECT_*` is line n + 1 of `generated/move_battle_effects.txt`):

| Pokemon | Moves (Kaizo id: effect) |
|---|---|
| Bonsly | Stealth Rock 446: STEALTH_ROCK 266; Selfdestruct 120: HALVE_DEFENSE 7; Brick Break 280: REMOVE_SCREENS 186; Accelerock 205: HIT 0 |
| Lileep | Sludge Bomb 188: POISON_HIT 2; Giga Drain 202: RECOVER_HALF_DAMAGE_DEALT 3; AncientPower 246: RAISE_ALL_STATS_HIT 140; Recover 105: RESTORE_HALF_HP 32 |
| Gible | DragonBreath 225: PARALYZE_HIT 6; Flamethrower 53: BURN_HIT 4; Thunder Fang 422: FLINCH_PARALYZE_HIT 275; Earth Power 414: LOWER_SP_DEF_HIT 72 |
| Corsola | Recover 105: 32; Water Pulse 352: CONFUSE_HIT 76; Light Screen 113: SET_LIGHT_SCREEN 35; Accelerock 205: 0 |
| Shuckle | Toxic 92: STATUS_BADLY_POISON 33; Accelerock 205: 0; Curse 174: CURSE 109; Sand Tomb 328: WHIRLPOOL 261 |
| Cranidos | Head Smash 457: RECOIL_HALF 269; Zen Headbutt 428: FLINCH_HIT 31; Crunch 242: LOWER_DEFENSE_HIT 69; Rock Polish 397: SPEED_UP_2 52 |

All of these effects are in `KAIZO.transcribed` from K3c Task 5. Their BASIC and
EVAL_ATTACK paths (Task 5) are 5.1-5.2; the EXPERT entries of RECOVER_HALF_DAMAGE_DEALT,
HALVE_DEFENSE, RESTORE_HALF_HP, STATUS_BADLY_POISON, SET_LIGHT_SCREEN, SPEED_UP_2, CURSE,
REMOVE_SCREENS and RECOIL_HALF (`script.s:1629-1806`) and RISKY (Task 6) are 5.3-5.4.

**The Expert guard (coordinator ruling, Task 5).** A missing Expert path is refused, never
scored. `outcome.KAIZO_EXPERT_LABELS` maps each of those nine effects to its label and the
line of its entry in `Expert_Main`'s dispatch table (Giga Drain `Expert_DrainMove` `:1629`,
Selfdestruct `Expert_Explosion` `:1630`, Recover `Expert_Recovery` `:1652`, Toxic
`Expert_ToxicLeechSeed` `:1653`, Light Screen `Expert_LightScreen` `:1654`, Rock Polish
`Expert_StatusSpeedUp` `:1665`, Curse `Expert_Curse` `:1700`, Brick Break `Expert_BrickBreak`
`:1739`, Head Smash `Expert_RecoilMove` `:1806`). A per-slot layout refuses an opponent move
with one of these effects unless the effect is in its `expert_transcribed` set
(`policy._refused`), naming the label. `KAIZO.expert_transcribed` is
`outcome.KAIZO_EXPERT_TRANSCRIBED`, a literal of the nine effects `policy._expert` dispatches
(Task 6 review I1: it is not derived from the table). So the guard refuses nothing of Kaizo
Roark's. The workflow for a later effect with an Expert_Main entry:
1. Add it to the table (and its site row) first. It is then *pending*: in the table, not
   branched on by `_expert`, not in the literal, and every move with it is refused. The
   tests allow this state.
2. Transcribe its label (a `policy._EXPERT_LABELS` row) and add the effect to the literal by
   hand.

`test_expert_transcribed_is_the_dispatch` records the branches a run of `_expert` takes and
checks that the literal is the table's effects among them (and only branched effects). It
also refuses a synthetic table entry outside the literal, and checks that adding an entry to
`outcome.py`'s table leaves the literal unchanged.
`test_expert_guard_refuses_an_untranscribed_label` does the same through coverage, with a
smaller layout. BASIC's table (`policy._BASIC_LABELS`) is pinned the same way against
`Basic_ScoreMoveEffect` (`script.s:133-286`; `test_basic_label_table_is_the_dispatch`). A
transcribed effect with a BASIC entry needs a row there.
`D` is the player's active Pokemon, `A` Roark's. A decomp stat stage is the engine stage plus
6. `HP%` is `curHP * 100 / maxHP`, integer division (`trainer_ai.c:626`, `:641`).

### 5.1 BASIC (`Basic_Main`, `script.s:52-1565`)

Steps 1-5 are roark.md 5.1's. Two facts of Kaizo data decide which moves take step 4:
- **No-calc moves skip `Basic_CheckForImmunity`.** Selfdestruct (HALVE_DEFENSE) and Head Smash
  (RECOIL_HALF) are in `sNoDamageCalcMoveEffects` (`trainer_ai.c:31-46`), so
  `FlagMoveDamageScore` gives `AI_NO_COMPARISON_MADE` (`trainer_ai.c:1035-1036`) and
  `script.s:62-63` jumps to `Basic_CheckSoundproof`, as for the power-0 moves.
- **The `:78` compare is Dry Skin** (2.1), so a Water move (Water Pulse) into a Dry Skin guess
  scores -12.

Step 6, `Basic_ScoreMoveEffect` (`:133`), has an entry for six of Roark's effects (each
effect appears once in the table, so the order of the entries does not matter):

- **HALVE_DEFENSE** (`:135`) -> `Basic_CheckCannotExplode` (`:297-319`):
  1. `IfMoveEffectivenessEquals TYPE_MULTI_IMMUNE, ScoreMinus10` (`:299`): **-10** and stop
     (Selfdestruct, Normal, into a Ghost type).
  2. `LoadBattlerAbility ATTACKER`; Mold Breaker jumps to `Basic_CheckLastMon` (`:302-303`).
  3. Otherwise `LoadBattlerAbility DEFENDER` (an ability guess, roark.md 9.2) and Damp gives
     **-10** and stop (`:304-305`). No Kaizo species has Damp (`species.json`), so in Kaizo
     data this read never changes a score.
  4. `Basic_CheckLastMon` (`:307`): `CountAlivePartyBattlers ATTACKER` (`trainer_ai.c:1126-1156`:
     the party's other members with HP above 0). Not 0 jumps to `Basic_Explode_Terminate`
     (`:318`, `PopOrEnd`): no change. Else `CountAlivePartyBattlers DEFENDER`: not 0 is
     **-10** (`:313`), 0 is **-1** (`GoTo ScoreMinus1`, `:316`).
- **RESTORE_HALF_HP** (`:154`) -> `Basic_CheckCanRecoverHP` (`:511-517`):
  `IfHPPercentNotEqualTo ATTACKER, 100` jumps to `PopOrEnd`; at 100% the move gets **-8**.
- **STATUS_BADLY_POISON** (`:155`) -> `Basic_CheckCannotPoison` (`:519-547`). Each test
  below jumps to `ScoreMinus10` (**-10** and stop):
  1. `D`'s true type 1, then type 2, is Steel or Poison (`LoadTypeFrom`, `trainer_ai.c:945-959`).
  2. `LoadBattlerAbility DEFENDER` (a guess) is Immunity, Magic Guard or Poison Heal.
  3. It is Leaf Guard and `LoadCurrentWeather` is `AI_WEATHER_SUNNY` (`:534-536`; section 7.2:
     the raw weather, no Cloud Nine check).
  4. `Basic_CheckCannotPoison_Hydration` (`:538`): a **second** `LoadBattlerAbility DEFENDER`
     (its own coin flip in the decomp) is Hydration and the weather is `AI_WEATHER_RAINING`.
  5. `Basic_CheckCannotPoison_StatusOrSafeguard` (`:544`): `D` has any status
     (`IfStatus ... MON_CONDITION_ANY`), or `D`'s side has Safeguard.
- **SET_LIGHT_SCREEN** (`:156`) -> `Basic_CheckAlreadyUnderLightScreen` (`:549-552`):
  Light Screen on `A`'s own side (`IfSideCondition AI_BATTLER_ATTACKER`) is **-8**.
- **SPEED_UP_2** (`:166`) -> `Basic_CheckHighStatStage_Speed` (`:363-372`): Trick Room is
  **-10** (`:364`). Then `A`'s true ability: with Simple, Speed above decomp 8 (engine +3
  or more) is **-10** (`:367`); without, Speed at decomp 12 (engine +6) is **-10** (`:371`).
- **CURSE** (`:196`) -> `Basic_CheckCurse` (`:679-698`): a Ghost-type `A` (true type 1 or 2)
  jumps to `Basic_CheckCurse_GhostType` (`:700-705`): the player already cursed, or with
  Magic Guard (its ability, guessed or known), is **-10** (transcribed in the E1 trial batch,
  5.25; the Ghost Curse refusal is gone). Kaizo Roark's Curse user, Shuckle, is Bug/Rock.
  With Simple, Attack or Defense above decomp 8 (engine +3 or more) is **-10**
  (`:690-691`). Without, Attack at 12 is **-10** (`:696`), else Defense at 12 is **-8**
  (`:697`).
- Stealth Rock keeps roark.md's `Basic_CheckStealthRock`. Every other effect of Roark's moves
  has no entry and reaches `PopOrEnd` (`:286`) unchanged.

No label above reads `IfRandomLessThan`.

### 5.2 EVAL_ATTACK (`EvalAttack_Main`, `script.s:6353-6415`)

As roark.md 5.2, with these effect tests now on transcribed paths:
1. `IfCurrentMoveKills USE_MAX_DAMAGE` (`:6358`). A no-calc move never kills: the handler
   computes damage only when the power is above 1 and the effect is not in
   `sNoDamageCalcMoveEffects` (`trainer_ai.c:1539-1554`). So Selfdestruct and Head Smash never
   take the kill bonus, however low `D`'s HP.
2. `FlagMoveDamageScore` (`:6361-6362`): `AI_NOT_HIGHEST_DAMAGE` is **-1** and stop. The
   no-calc and power-0 moves get `AI_NO_COMPARISON_MADE`, never -1. Since no-calc moves
   estimate 0, the highest damage of Cranidos is Crunch's or Zen Headbutt's (plan Review
   Focus 1).
3. HALVE_DEFENSE jumps to `EvalAttack_MaybeDeprioritize` (`:6365` -> `:6372`):
   `IfRandomLessThan 51` (`:6374`). If the draw is below 51 it jumps on; otherwise the move
   gets **-2** (205/256). Both paths go on to the quad check. The Focus Punch and Sucker
   Punch effects (`:6366-6367`) are no Roark move's.
4. `EvalAttack_CheckQuadEffective` (`:6377-6385`): QUADRUPLE runs `IfRandomLessThan 80`
   (`:6383`), +2 when the draw is not below 80 (176/256). Every move that gets past items
   1-2 reaches it (status and no-calc moves always do, since they never kill and never take
   the -1): Rock Polish (Rock) into a Fire/Flying `D` is QUADRUPLE. A move that kills jumps
   to item 5 (`:6358`) and one that is out-damaged stops at **-1** (`:6361-6362`,
   `ScoreMinus1` `:1567-1569`); neither reaches `:6383`.
5. `EvalAttack_ApplyKillBonuses` (`:6387-6415`), for a move that kills:
   - HALVE_DEFENSE ends at `EvalAttack_Terminate` (`:6389`). It is unreachable for
     Selfdestruct, which never kills (1).
   - The Focus Punch, Sucker Punch and Future Sight effects (`:6394-6396`) are no Roark move's.
   - `BATTLE_EFFECT_PRIORITY_1` (`:6400`) gets **+2** then **+4**. It keys on the effect, not
     the move's priority (the decomp's comment, `:6398-6399`). Kaizo Accelerock has priority
     +1 but effect HIT, so its kill bonus is **+4** (plan K3c Ruling 8).
   - Anything else: **+4** (`:6401` -> `:6411`), with no draw.

### 5.3 EXPERT (`Expert_Main`, `script.s:1623-1811`)

`Expert_Main`'s dispatch (`:1628-1808`) sends nine of Roark's effects to a label added in
K3c Task 6 (`policy._expert`); Stealth Rock keeps roark.md's `Expert_StealthRock`. The jump
tables of BASIC and EXPERT (`policy._BASIC_LABELS`, `_EXPERT_LABELS`) are restricted at trace
time to the layout's transcribed effects (`policy._dispatch`), so the vanilla program traces
none of the Kaizo labels (Task 6 review m1; `test_dispatch_is_static_per_layout`). Every
other effect of Roark's moves has no entry and reaches `PopOrEnd` (`:1811`). That includes
Sand Tomb's WHIRLPOOL (261): `Expert_BindingMove` is dispatched on BIND_HIT only (`:1659`,
Ruling 5). "N roll" means `IfRandomLessThan N`: it jumps with probability N/256, and the
score change is on the path that does not jump. The speed compare is
`IfSpeedCompareEqualTo COMPARE_SPEED_SLOWER` (section 7.1): Roark's Pokemon acts after the
player (a lone activated Quick Claw of the player's counts, `battle_lib.c:1408-1411`; Trick
Room reverses, `:1432-1439`; equal speed is never SLOWER, Ruling 3).

- **RECOVER_HALF_DAMAGE_DEALT** (`:1629`) -> `Expert_DrainMove` (`:1827-1839`): IMMUNE,
  HALF_DAMAGE or QUARTER_DAMAGE goes to the 50 roll (`:1835`), **-3**. Otherwise no change.
- **HALVE_DEFENSE** (`:1630`) -> `Expert_Explosion` (`:1841-1884`):
  1. `D`'s evasion at decomp 7 (+1) or more: **-1** (`:1856-1857`). At 10 (+4) or more,
     another **-1** after the 128 roll (`:1858-1860`).
  2. `Expert_Explosion_CheckUserHighHP` (`:1862`): `A`'s HP% at least 80 and `A` not slower:
     the 50 roll (`:1865`), else `GoTo ScoreMinus3`, **-3**.
  3. Otherwise `Expert_Explosion_CheckUserMediumHP` (`:1868`): HP% above 50 goes to
     `Expert_Explosion_TryScoreMinus1` (`:1879`), the 50 roll (`:1880`), **-1**.
  4. HP% 50 or less: the 128 roll (`:1870`), **+1**; then `Expert_Explosion_CheckUserLowHP`
     (`:1873`): HP% 30 or less adds the 50 roll (`:1875`), **+1**. Both can apply.
- **RESTORE_HALF_HP** (`:1652`) -> `Expert_Recovery` (`:2640-2678`):
  1. HP% 100: `Expert_Recovery_ScoreMinus3AndEnd` (`:2659`), **-3**.
  2. Not slower: **-8** (`:2651`).
  3. Slower: `Expert_Recovery_CheckHP` (`:2663`). HP% at least 70: the 30 roll (`:2665`),
     **-3**; its jump goes to 4.
  4. `Expert_Recovery_CheckForSnatch` (`:2669`): if the AI has seen the player use a
     STEAL_STATUS_MOVE (195) move, the 100 roll (`:2671`) can end it. No Kaizo move has that
     effect, so `:2671` is transcribed but never reached in Kaizo data. Then the 20 roll
     (`:2674`), **+2**.
  - `Expert_Recovery_Unused` (`:2654-2657`) follows the `GoTo` at `:2652`, and no jump targets
    it: it is not reached and not transcribed.
- **STATUS_BADLY_POISON** (`:1653`) -> `Expert_ToxicLeechSeed` (`:2680-2708`):
  1. `IfAttackerHasNoDamagingMoves` (`:2688`; power 0 for every move, `trainer_ai.c:913-929`)
     skips to 3.
  2. `A`'s HP% 50 or less: the 50 roll (`:2690`), **-3**. `D`'s HP% 50 or less: the 50 roll
     (`:2695`), **-3**.
     The `:2694` check of `D`'s HP is a prep HP cut of the Kaizo policy only
     (`prep/variants.py` `AI_DEFENDER_HP_CHECKS`, keyed by policy; vanilla keeps 70).
  3. `A` knows a move with effect SP_DEF_UP (14) or PROTECT (111) (`IfMoveEffectKnown
     ATTACKER`, its own true moveset, Ruling 9): the 60 roll (`:2704`), **+2**. No Kaizo move
     has SP_DEF_UP; Kaizo Roark's Pokemon know no PROTECT move.
- **SET_LIGHT_SCREEN** (`:1654`) -> `Expert_LightScreen` (`:2710-2732`): HP% below 50 is
  **-2** (`:2716` -> `:2728`). HP% at least 90 adds the 128 roll (`:2718`), **+1**. Then
  `LoadDefenderLastUsedMoveClass` (section 7.4): `CLASS_SPECIAL` (1) adds the 64 roll
  (`:2724`), **+1**. After Nature Power the class is Nature Power's (status), so no +1.
- **SPEED_UP_2** (`:1665`) -> `Expert_StatusSpeedUp` (`:2066-2079`): not slower is **-3**;
  slower is the 70 roll (`:2075`), **+3**.
- **CURSE** (`:1700`) -> `Expert_Curse` (`:3452-3500`):
  1. A Ghost-type `A` goes to `Expert_Curse_GhostCheckHP` (`:3467-3470` -> `:3495`): above
     80 % HP it ends, else **-1** (`:3496-3497`); no roll (E1 trial batch, 5.25).
  2. `A`'s Defense above decomp 9 (+4 or more) ends it (`:3471`). (The decomp's comment says
     "+3 or higher"; the code tests > 9.)
  3. `A` knows Gyro Ball (360) or Trick Room (433) (`IfMoveKnown ATTACKER`, `:3472-3473`):
     the 32 roll (`:3477`), **+1**; its jump skips to 5.
  4. The 128 roll (`:3481`), **+1**.
  5. `Expert_Curse_CheckDefenseStage` (`:3484`): Defense above 7 (+2 or more) ends it; else
     the 128 roll (`:3486`), **+1**.
  6. Defense above 6 (+1 or more) ends it (`:3490`); else the 128 roll (`:3491`), **+1**.
- **REMOVE_SCREENS** (`:1739`) -> `Expert_BrickBreak` (`:4559-4568`): Reflect or Light Screen
  on `D`'s side (`IfSideCondition AI_BATTLER_DEFENDER`) is **+1**. No roll.
- **RECOIL_HALF** (`:1806`) -> `Expert_RecoilMove` (`:6289-6305`): IMMUNE, HALF_DAMAGE or
  QUARTER_DAMAGE ends it; otherwise `A`'s true ability Rock Head (69) or Magic Guard (98) is
  **+1**. Kaizo Cranidos has Rock Head in ability slot 0 and Mold Breaker in slot 1. No roll.

### 5.4 RISKY (`Risky_Main`, `script.s:6521-6534`)

`LoadCurrentMoveEffect` (`trainer_ai.c:1164`), then `IfLoadedNotInTable Risky_RiskyEffects`
(`:6527`; the table is `:6536-6562`, `commands.RISKY_EFFECTS`). A listed effect takes the 128
roll (`:6530`), **+2**. Of Roark's effects the table lists HALVE_DEFENSE (`:6538`,
Selfdestruct) and RAISE_ALL_STATS_HIT (`:6550`, AncientPower). RISKY runs after EXPERT, in
bit order (`FlagTable`, `script.s:18-30`; `TrainerAI_MainSingles`, `trainer_ai.c:297-310`).

RISKY is part of the Kaizo policy itself (`policy._flags` runs `_risky` when the layout's
flag set has bit 4, E1 Ruling R2), not only of coverage. So a caller of `ai_refusals` and
`ai_scores` alone never scores Selfdestruct or AncientPower without the roll under Roark's set
(Task 5 review m2; `test_risky_is_part_of_the_policy`), and never adds it under a set without
RISKY (`test_flag_set_runs_its_scripts`).

### 5.5 SETUP_FIRST_TURN (`SetupFirstTurn_Main`, `script.s:6417-6433`; E1 batch 2b)

Not one of Roark's flags (bit 3; sets 0xd, 0xf, 0x1f, 0x10f, 0x20f of the Roark and Gardenia
splits). `LoadTurnCount` (`trainer_ai.c:931-935`, `totalTurns`, `BattleState.turn`) not 0
ends it (`:6421-6422`): it scores only on the battle's first turn. Then `LoadCurrentMoveEffect`
and `IfLoadedNotInTable SetupFirstTurn_SetupEffects` (`:6425-6426`; the table is
`:6435-6497`, 61 entries and 59 effects, SP_DEF_UP_2 and CAMOUFLAGE listed twice:
`commands.SETUP_FIRST_TURN_EFFECTS`). A listed effect takes the 80 roll (`:6429`), **+2** when
it does not jump (probability 176/256). Of the transcribed effects the table lists DEF_DOWN,
SET_LIGHT_SCREEN, SPEED_UP_2, DEF_DOWN_2, CURSE and WHIRLPOOL. The roll is one more read on
the slot's path (8.2); the read bound (8.3) holds under every implemented set
(`test_slot_reads_fit_every_transcribed_effect` runs them all at once, on the first turn and
after it).

### 5.6 WEATHER (`Weather_Main`, `script.s:7979-8019`; E1 batch 2b)

Bit 9 (sets 0x207, 0x20f). It too ends unless `LoadTurnCount` is 0 (`:7983-7984`). The four
`IfCurrentMoveEffectEqualTo` (`:7987-7990`: WEATHER_SUN, _RAIN, _SANDSTORM, _HAIL) send a
weather move to its weather's label, and **no `GoTo` follows them: every other move falls
through into `Weather_Sun`** (`:7992`; E1 Ruling R3, transcribed as the game has it). Each
label loads `LoadCurrentWeather` (`trainer_ai.c:1414-1439`) and ends when it is the label's
own weather (`:7993-7994` and so on), else goes to `Weather_ScorePlus5` (`:8012-8016`):
**+5** when the attacker's `LoadIsFirstTurnInBattle` is TRUE (`:2480`; on the first turn the
lead's `fakeOutTurnNumber` is 0, so always). So on the first turn every non-weather move gets
+5 unless the weather is sun, and a weather move gets +5 unless its weather is already up.
No roll. The +5 is uniform over the non-weather moves but not a no-op: a later flag's
`AddToMoveScore` clamps at 0 (`trainer_ai.c:576`). The weather moves themselves are
unmodelled engine effects (refused by move effect), and a runnable fight's weather is none or
sandstorm (`coverage.IMPLEMENTED_WEATHER`); the transcription covers all four labels anyway.

### 5.7 PRIORITIZE_EXTREMES (`PrioritizeExtremes_Main`, `script.s:6499-6519`; E1 batch 2c)

Bit 5 (Barry's 0x27). `FlagMoveDamageScore USE_MAX_DAMAGE` then `IfLoadedNotEqualTo
AI_NO_COMPARISON_MADE` (`:6511-6512`): only a move the damage compare skips continues, that is
one not in `sAltPowerMoveEffects` with power at most 1 or a `sNoDamageCalcMoveEffects` effect
(`trainer_ai.c:1012-1063`). Despite the comment at `:6503-6509`, alt-power moves are compared,
so they never reach the roll (`commands.is_damaging`). The 100 roll
(`:6515`) gives **+2** when it does not jump (156/256). Of the transcribed effects it reaches
every status move (Stealth Rock, Toxic, Recover, Light Screen, Rock Polish, Curse, Leer,
Screech) and the no-calc Selfdestruct and Head Smash.

### 5.8 CHECK_HP (`CheckHP_Main`, `script.s:7696-7746`; E1 batch 2c)

Bit 8 (Sebastian's 0x10f). Its `IfTargetIsPartner` goes to `TagStrategy_Partner` (`:7697`),
never taken in singles. Two rounds, each with one roll:

1. The attacker's HP percent (`IfHPPercentGreaterThan` 70, then 30, `:7700-7701`; integer
   `curHP * 100 / maxHP`, `trainer_ai.c:603`) picks `CheckHP_DiscourageAtHighHP` (> 70),
   `..AtMediumHP` (31-70) or `..AtLowHP` (30 and below). `LoadCurrentMoveEffect` and the new
   command `IfLoadedInTable` (`trainer_ai.c:858-874`) jump to `CheckHP_TryScoreMinus2` for a
   listed effect, which gives **-2** unless the 50 roll (`:7718`) jumps; every path then goes on
   to the target's round.
2. The same with the player's true HP (`:7724-7725`) and the `CheckHP_Target_Discourage*`
   tables, roll `:7742`. `CheckHP_Target_DiscourageAtHighHP` is empty (`:7866-7867`).

The six tables are `commands.CHECK_HP_TABLES` (213 entries, by label, the
repeated HALVE_HP of `:7935-7936` included). The player-HP reads
> 70 and > 30 are prep cuts for a CHECK_HP fight (`prep/variants.AI_FLAG_DEFENDER_HP_CHECKS`,
added to the policy's checks by `ai_defender_hp_checks` when the set has bit 8).

### 5.9 Effects with no Basic or Expert label (E1 batch 3)

The 26 effects of trainers' moves (the backlog's) that have no entry in
`Basic_ScoreMoveEffect` (`script.s:133-286`) or `Expert_Main` (`:1628-1808`); about 20 more
effects of Kaizo moves have none either and stay refused until a trainer uses them: FREEZE_HIT, MULTI_HIT, TRI_ATTACK, HIT_TWICE, CRASH_ON_MISS,
LOWER_ACCURACY_HIT, CALL_RANDOM_MOVE, LEAVE_WITH_1_HP, PRIORITY_1, THAW_AND_BURN_HIT,
RAISE_DEF_HIT, RAISE_ATTACK_HIT, DOUBLE_DAMAGE_DIG, DOUBLE_DAMAGE_FLY_OR_BOUNCE,
FLINCH_MINIMIZE_DOUBLE_HIT, THUNDER, UPROAR, MAKE_GLOBAL_TARGET, USE_RANDOM_ALLY_MOVE,
SECRET_POWER, BADLY_POISON_HIT, DOUBLE_DAMAGE_DIVE, LOWER_SP_DEF_2_HIT, FLINCH_BURN_HIT,
FLINCH_FREEZE_HIT and RAISE_SP_ATK_HIT. Every transcribed flag treats them generically, so
they only join `outcome.KAIZO_TRANSCRIBED`:

- The C code reads a move's effect only against `sNoDamageCalcMoveEffects` and
  `sAltPowerMoveEffects` (`FlagMoveDamageScore`, `trainer_ai.c:1012-1063`; `IfCurrentMoveKills`
  and `IfCurrentMoveDoesNotKill`, `:1525-1610`; `CheckIfHighestDamageWithPartner`, `:2382-2388`;
  the damage estimates, `:2800-2830`), and none of the 26 is in either. `TrainerAI_CalcDamage`
  has no case for their moves, and the Dig/Fly/Dive/Minimize doublings are a move script's
  power multiplier, reset each turn (`battle_controller_player.c:1855`), so the AI never sees
  them.
- Their other mentions in `script.s` are generic tables or other moves' paths: PRIORITY_1 in
  EvalAttack's kill bonus (`:6400`, 5.2); CALL_RANDOM_MOVE in `Risky_RiskyEffects` (`:6543`);
  MAKE_GLOBAL_TARGET in `Expert_Encore_EncouragedMoveEffects` (`:3249`), which Encore reads for
  the player's last move; SECRET_POWER in HARRASSMENT's table (`:8062`; not a transcribed
  flag); TAG_STRATEGY names several of them (`:6691`, `:6736`, `:6746`, `:7270`, `:7280`,
  `:7495`), to check when that flag is transcribed (it runs on 15 singles fights). Uproar is in
  Basic's Soundproof list by move id (`:127`, `commands.SOUND_MOVES`). THUNDER has no `Expert_Thunder` path: `Expert_Main` tests
  SKIP_CHARGE_TURN_IN_SUN for it (`:1718`, the decomp's own BUG comment), transcribed as the
  game has it.
- `test_expert_label_table_is_the_dispatch` and `test_basic_label_table_is_the_dispatch` fail
  for any transcribed effect with an entry whose label is not transcribed.

Many of these moves are still refused by the engine's move-effect coverage (Metronome, Follow
Me, Assist, Uproar, Secret Power's terrain, ...); the AI side no longer adds a line for them.

### 5.10 Stat raises (E1 batch 4)

The stat raises Kaizo moves carry: ATK_UP, DEF_UP, SP_ATK_UP, ATK_UP_2, DEF_UP_2, SP_ATK_UP_2
and SP_DEF_UP_2 (SPEED_UP_2 was Roark's). No Kaizo move has SPEED_UP, SP_DEF_UP, ACC_UP,
ACC_UP_2 or DEF_UP_DOUBLE_ROLLOUT_POWER, so their paths (the accuracy labels among them) are
not transcribed. The evasion raises waited for a per-fight read bound (`Expert_StatusEvasionUp`
alone can make 7 random reads on one path, 8.3): Minimize (EVA_UP_2_MINIMIZE) came in prep P1
(5.21) and EVA_UP (Double Team) in the E1 trial batch. EVA_UP_2 (56) has no Kaizo move and
stays untranscribed.

- BASIC: `Basic_CheckHighStatStage_<stat>` (`script.s:343-392`; dispatch `:137-141`,
  `:164-168`): -10 when the attacker is already at the top, above +2 under Simple
  (`IfStatStageGreaterThan` 8) or at +6 otherwise (`IfStatStageEqualTo` 12).
- EXPERT (`:1633-1637`, `:1663-1667`):
  - `Expert_StatusAttackUp` / `_SpAttackUp` (`:1978-2007`, `:2081-2110`): at +3 or more, -1
    unless the 100 roll jumps (`:1989`, `:2092`); otherwise at exactly 100 % HP, +2 unless
    the 128 roll jumps (`:1995`, `:2098`). Then above 70 % HP nothing more, below 40 % -2,
    otherwise -2 unless the 40 (Attack, `:2001`) or 70 (Sp. Atk, `:2104`) roll jumps.
  - `Expert_StatusDefenseUp` / `_SpDefenseUp` (`:2009-2052`, `:2112-2155`): the same first two
    steps (`:2023`, `:2029`; `:2126`, `:2132`); then at 70 % HP or more nothing more if the 200
    roll jumps (`:2034`, `:2137`), else on. Below 40 % -2. Otherwise the player's previous move
    (`LoadBattlerPreviousMove`, `trainer_ai.c:1068-1076`, and the new
    `LoadPowerOfLoadedMove`, `:2526-2530`): power 0 goes to the second 60 roll; a special
    move (Defense) or a physical one (Sp. Def; `LoadDefenderLastUsedMoveClass`) to -2; any
    other to the first 60 roll, which falls into the second when it does not jump (`:2043`,
    `:2046`; `:2146`, `:2149`). -2 unless a 60 roll jumps.
  - The `PreSplitPhysicalTypes` tables after the defense labels (`:2054`, `:2157`) are not
    read.

### 5.11 Status moves (E1 batch 6)

STATUS_SLEEP, STATUS_CONFUSE, STATUS_PARALYZE, STATUS_LEECH_SEED, ATK_UP_2_STATUS_CONFUSION
(Swagger), INFATUATE (Attract) and STATUS_BURN (Will-O-Wisp).

- BASIC (dispatch `:134`, `:163`, `:180`, `:183`, `:201`, `:202`, `:231`): -10 when the move
  cannot work. `Basic_CheckCannotSleep` (`:288-295`): any status, Safeguard, Insomnia or Vital
  Spirit. `Basic_CheckCannotConfuse` (`:597-605`, Swagger too): already confused -5, then Own
  Tempo or Safeguard -10. `Basic_CheckCannotParalyze` (`:612-631`): immune by type, Limber or
  Magic Guard; unless the attacker has Mold Breaker, Thunder Wave also against Motor Drive or
  Volt Absorb; then any status or Safeguard. `Basic_CheckCannotLeechSeed` (`:639-648`): seeded
  already, a Grass type, Magic Guard. `Basic_CheckCannotAttract` (`:731-752`): attracted
  already, Oblivious, or not of opposite genders (genderless on either side included).
  `Basic_CheckCannotBurn` (`:879-890`): Water Veil, Magic Guard, any status, a Fire type,
  Safeguard. Every ability is the defender's guessed or known one (section 9).
- EXPERT (`:1628`, `:1662`, `:1679`, `:1680`, `:1686`): `Expert_StatusSleep` (`:1813-1825`):
  +1 unless the 128 roll (`:1821`) jumps, when the attacker knows a Dream Eater or Nightmare
  effect. `Expert_StatusConfuse` (`:2861-2877`): nothing above 70 % of the player's HP; else -1
  unless the 128 roll (`:2864`) jumps, -1 at 50 % or less, -1 more at 30 % or less (prep cuts
  the player's HP at 70, 50 and 30, `prep/variants.AI_DEFENDER_HP_CHECKS`).
  `Expert_StatusParalyze` (`:2940-2954`): slower than the player, +3 unless the 20 roll
  (`:2950`) jumps; else -1 when the attacker's own HP is 70 % or less (`:2945`). `Expert_Swagger` (`:2845-2891`): with Psych Up
  known, -5 when the player's Attack stage is above decomp 3, else +3, +2 more on the first
  turn; without it, `Expert_Flatter`'s +1 unless the 128 roll (`:2858`) jumps, then
  `Expert_StatusConfuse`. Leech Seed takes Roark's `Expert_ToxicLeechSeed` (5.3); Attract and
  Will-O-Wisp have no `Expert_Main` entry.

### 5.12 Hits scored only by an Expert label (E1 batch 7)

None of these has a `Basic_ScoreMoveEffect` entry.
- `Expert_BypassAccuracyMove` (`:1640`; `:2256-2275`): the player's evasion above decomp 10 or
  the attacker's accuracy below 2, +1 and +1 more unless the 100 roll (`:2271`) jumps; above 8
  or below 4, only the roll's +1.
- `Expert_HighCritical` (`:1660`; `:2827-2843`): resisted, nothing; super effective, +1 unless
  the 128 roll (`:2839`) jumps; neutral, a first 128 roll (`:2836`) ends it when it jumps,
  else the same. RISKY's table holds HIGH_CRITICAL too (`:6541`; its roll `:6530`).
- `Expert_SpeedDownOnHit` (`:1681`; `:2338-2350`): resisted, nothing; Icy Wind, Rock Tomb and
  Mud Shot go to `Expert_StatusSpeedDown` (`:2353-2366`: not slower -3; slower +2 unless the 70
  roll, `:2362`, jumps); any other LOWER_SPEED_HIT move ends.
- `Expert_Facade` (`:1729`; `:4120-4127`): +1 when the *player* is poisoned, burned or
  paralyzed (the decomp tests the defender: `IfNotStatus AI_BATTLER_DEFENDER`).
- `Expert_SmellingSalts` (`:1731`; `:4165-4174`): +1 when the player is paralyzed.
- `Expert_Brine` (`:1765`; `:4873-4890`): resisted -1; else at the player's 50 % HP or less +1,
  and +1 more unless the 128 roll (`:4882`) jumps.
- RECOIL_QUARTER, RECOIL_THIRD and RECOIL_PARALYZE_HIT (`:1661`, `:1747`, `:1802`) share
  Head Smash's `Expert_RecoilMove` (5.3).
- U-turn (`Expert_UTurn`) waits: it reads two party-damage commands. Payback came in batch 11
  (5.15).

### 5.13 Setup moves, hazards and Destiny Bond (E1 batch 9)

- `Basic_CheckBulkUp` / `_CalmMind` / `_DragonDance` (`:965-1019`; dispatch `:248`, `:250`,
  `:251`): with Simple, -10 when either stat is above decomp 8; without, -10 when the first
  (Attack, Sp. Atk) is at +6 and -8 when the second (Defense, Sp. Def, Speed) is; Dragon Dance
  first gives -10 under Trick Room. Bulk Up and Calm Mind then take `Expert_StatusDefenseUp`
  and `Expert_StatusSpDefenseUp` (5.10; `:1753`, `:1756`). `Expert_DragonDance`
  (`:1757`; `:4765-4780`): slower than the player, +1 unless the 128 roll (`:4776`) jumps;
  else at 50 % HP or less -1 unless the 70 roll (`:4771`) jumps.
- `Basic_CheckSpikes` / `_ToxicSpikes` (`:707-713`, `:1445-1453`; `:197`, `:277`): -10 at 3
  Spikes / 2 Toxic Spikes layers on the player's side (the new command `LoadSpikesLayers`,
  `trainer_ai.c:1948-1966`) or with no other living Pokemon there. `Expert_Spikes` /
  `_ToxicSpikes` (`:1702`, `:1792`; `:3591-3606`, `:6012-6029`): +1 unless the 128 roll
  (`:3595`, `:6018`) jumps, and with Roar or Whirlwind known +1 more unless the 64 roll
  (`:3602`, `:6025`) jumps.
- `Expert_DestinyBond` (`:1694`; `:3335-3360`): -1; slower than the player, nothing more;
  otherwise at 70 % HP or less +1 unless `:3346` jumps, at 50 % or less +1 more unless `:3351`
  jumps, at 30 % or less +2 more unless the 100 roll (`:3356`) jumps.
- Roost's labels exist (Recover's); the AI's type chart reads the roosting state since E1 prep
  batch P2 (5.22). Tailwind came in batch 11 (5.15).

### 5.14 Stat drops and Reflect (E1 batch 10)

ATK_DOWN, SPEED_DOWN, EVA_DOWN, ATK_DOWN_2, SPEED_DOWN_2, SP_DEF_DOWN_2 (DEF_DOWN and
DEF_DOWN_2 were Roark's) and SET_REFLECT. ACC_DOWN came with E2's Ingrain and Curse state
(5.21).
- BASIC (`:144`, `:146`, `:150`, `:171`, `:173`, `:175`): `Basic_CheckLowStatStage_<stat>`
  (`:428-467`), -10 at -6 already, Attack also against Hyper Cutter, Speed also under Trick
  Room or against Speed Boost as the new command `CheckBattlerAbility` sees it (`:441`; for an
  unknown player of two abilities, one of them Speed Boost, it answers unknown, so no -10),
  evasion also against No Guard on either side; then `Basic_CheckClearBodyEffect`
  (`:469-473`), -10 against Clear Body or White Smoke. `Basic_CheckAlreadyUnderReflect`
  (`:607-610`): -8 under the AI's own Reflect.
- EXPERT: `Expert_StatusAttackDown` (`:2277-2307`): with the player's Attack not at +0, -1,
  -1 more at 90 % HP or less, and -2 unless the 50 roll (`:2293`) jumps when it is at decomp
  3 or less; then -2 at the player's 70 % HP or less; then -2 unless the 128 roll (`:2303`)
  jumps when its last move was special. `Expert_StatusSpeedDown` (`:2353-2366`, shared with
  5.12). `Expert_StatusSpDefenseDown` / `_EvasionDown` (`:2411-2429`, `:2500-2518`): Roark's
  `Expert_StatusDefenseDown` shape with rolls `:2421`, `:2510`. `Expert_Reflect`
  (`:2893-2915`): below 50 % HP -2; at 90 % or more +1 unless the 128 roll (`:2901`) jumps;
  then +1 unless the 64 roll (`:2907`) jumps when the player's last move was physical.

### 5.15 Tailwind and Payback (E1 batch 11, Ruling R8)

Both Expert labels compare for `COMPARE_SPEED_FASTER`, so the equal-speed tie decides them
(E1 Ruling R8, 7.1).
- `Basic_CheckTailwind` (`:1128-1133`; dispatch `:259`): -10 under Trick Room or with the AI's
  side already under Tailwind (`SideCondition.TAILWIND`, `include/constants/battle/condition.h:96-104`,
  `BattleState.tailwind`).
- `Expert_Tailwind` (`:1768`; `:4982-5006`): nothing when the 64 roll (`:4992`) jumps; then
  faster than the player, or below 31 % HP, -1; above 75 % +1; between, +1 unless the 64 roll
  (`:4996`) jumps.
- `Expert_Payback` (`:1773`; `:5169-5187`): resisted -1; else faster than the player or below
  30 % HP, nothing; else +1 unless the 64 roll (`:5179`) jumps.

### 5.16 Sleep moves, Fake Out, Pluck, counters and weather moves (E1 batch 12)

USE_RANDOM_LEARNED_MOVE_SLEEP (97), DAMAGE_WHILE_ASLEEP (92), ALWAYS_FLINCH_FIRST_TURN_ONLY
(158), EAT_BERRY (224), HALVE_HP (40), INCREASE_POWER_WITH_LESS_HP (99), COUNTER (89),
MIRROR_COAT (144), LOWER_OWN_ATK_AND_DEF (182), WEATHER_RAIN (136), WEATHER_HAIL (164).
- BASIC: `Basic_CheckAttackerAsleep` (`:660-663`; dispatch `:189`, `:191`): the attacker not
  asleep -8. `Basic_CheckFirstTurnInBattle` (`:829-833`; `:224`): not its first turn -10.
  `Basic_CheckNonStandardDamageOrChargeTurn` (`:572-585`; `:159`, `:187`, `:192`, `:218`,
  `:237`): immune -10; against Wonder Guard (a guessing read) unless Mold Breaker or super
  effective -10. `Basic_CheckRainDance` (`:786-801`; `:214`): without Swift Swim or Hydration,
  against a statused Hydration player -8; in rain -8. `Basic_CheckHail` (`:853-871`; `:228`):
  in hail -8; against Ice Body -8, +8 back with the AI's own Ice Body.
- EXPERT: `Expert_SleepTalk` (`:3327-3333`): asleep +10, else -5. `Expert_Nightmare` (Snore,
  `:3314-3316`) and `Expert_FakeOut` (`:4074-4076`): +2. `Expert_SuperFang` (`:2800-2806`):
  the player at 50 % HP or less -1. `Expert_Reversal` (`:3362-3395`): HP bands by the SLOWER
  compare, then +1 unless the 100 roll (`:3388`) jumps. `Expert_Counter` / `_MirrorCoat`
  (`:3095-3163`, `:3852-3920`, one shape, `policy._counter_like`): seven rolls each (10 and
  100 at low HP, 100 per try, 50 before the +4 try); the type tables are
  `commands.COUNTER_PHYSICAL_TYPES` / `MIRROR_COAT_SPECIAL_TYPES` (`:3165-3175`, `:3922-3931`);
  the new command `IfTargetIsNotTaunted` (`trainer_ai.c:2620-2628`). `Expert_Superpower`
  (`:4467-4488`), `Expert_RainDance` (`:3737-3771`, a FASTER compare, Ruling R8),
  `Expert_Hail` (`:4089-4119`), `Expert_Pluck` (`:4957-4980`, rolls `:4968`, `:4972`).
- PROTECT (111): `Expert_Protect` (`:3502-3589`; dispatch `:1701`; no Basic entry) reads
  Curse, Perish Song, Yawn and Lock-On from the per-battler fields of da1d60f (engine/STATE.md;
  nothing sets them until E2 models those moves), the chain through the new command
  `LoadProtectChain` (`BattleState.protect_count`), and the battle type through
  `LoadBattleType` (`Tables.battle_type`, per fight from `Trainer.format`,
  `encounter.c:724-740`). Rolls `:3541` (128), `:3564` (85), `:3571` (128), `:3578` (128).
- Waiting: Metal Burst (`Basic_CheckMetalBurst`, `:1160-1182`) reads
  the AI's record of the player's held item (`aiContext.battlerHeldItems`,
  `battle_script.c:12121`), which `BattleState` does not hold either. Both wait for the
  managing session's decision. The alternative-power moves' AI damage came in batch 13
  (2.4).

### 5.17 Alt-power effects, Hidden Power and OHKO (E1 batch 14)

- 40_DAMAGE_FLAT (41), LEVEL_DAMAGE_FLAT (87), RANDOM_DAMAGE_1_TO_150_LEVEL (88),
  POWER_BASED_ON_FRIENDSHIP (121), RANDOM_POWER_MAYBE_HEAL (122), POWER_BASED_ON_LOW_FRIENDSHIP
  (123), 20_DAMAGE_FLAT (130), RANDOM_POWER_BASED_ON_IVS (135), INCREASE_POWER_WITH_WEIGHT
  (196): `Basic_CheckNonStandardDamageOrChargeTurn` only (`:160`-`:244`, 5.16), no Expert_Main
  entry; their AI damage is 2.4's. Psywave's id (Kaizo Triple Axel) draws in its estimate (8.7).
- ONE_HIT_KO (38): `Basic_CheckOHKOWouldFail` (`:554-564`; dispatch `:157`): immune -10,
  against Sturdy (a guessing read) -10 unless Mold Breaker, the attacker's level below the
  player's -10 (the new command `IfLevel`, `trainer_ai.c:2579-2608`). `Expert_OHKOMove`
  (`:1656`; `:2792-2798`): +1 unless the 192 roll (`:2794`) jumps.
- ACC_DOWN waits: `Expert_StatusAccuracyDown` reads the AI's Ingrain (`:2472`,
  `MOVE_EFFECT_INGRAIN`), which `BattleState` does not hold.

### 5.18 U-turn (E1 batch 15)

SWITCH_HIT (228): `Expert_UTurn` (`:1771`; `:5085-5145`; no Basic entry): resisted -1; with
no other living Pokemon, nothing; a super-effective move (the new command
`IfHasSuperEffectiveMove`, `trainer_ai.c:2176-2184`, `:3560-3622`) -2 unless the 64 roll
(`:5114`) jumps; unless a party member out-damages the active (the new command
`IfPartyMemberDealsMoreDamage`, `:2106-2174`, which calculates the member's moves **as the
active battler**, with only the member's IVs), -2 unless `:5119` jumps and end; then the
player's HP bands (`:5126`, `:5130`, `:5134`) and the FASTER compare (its tie a read, Ruling
R8) with `:5139`.

### 5.19 Gardenia's first groups (E1 batch 16)

- PREVENT_STATUS (124): `Basic_CheckAlreadyUnderSafeguard` (`:754-757`; `:206`), -8 under the
  AI's own Safeguard; no Expert entry.
- WEATHER_SUN (137): `Basic_CheckSunnyDay` (`:803-820`; `:215`), with Flower Gift, Leaf Guard
  or Solar Power straight to the sun check, else against a statused Hydration player -10,
  then in sun -8; `Expert_SunnyDay` (`:1711`; `:3773-3801`), Rain Dance's shape (5.16).
- MAX_ATK_LOSE_HALF_MAX_HP (142): `Basic_CheckBellyDrum` (`:335-337`; `:216`), at 50 % HP or
  less -10, then falling into `Basic_CheckHighStatStage_Attack` (`:343`); `Expert_BellyDrum`
  (`:1712`; `:3803-3812`), below 90 % HP -2.
- SET_HP_EQUAL_TO_USER (189) and AVERAGE_HP (91): `Expert_Endeavor` (`:1741`; `:4583-4608`)
  and `Expert_PainSplit` (`:1690`; `:3287-3312`), one shape (`policy._hp_trade`); Endeavor
  also takes the non-standard Basic label (`:241`).
- HIT_LAST_WHIFF_IF_HIT (170, Focus Punch): the non-standard Basic label (`:233`),
  `Expert_FocusPunch` (`:1730`; `:4129-4163`; +5 behind its own Substitute, the new
  `VolatileCondition.SUBSTITUTE` read from `substitute_hp`), and EVAL_ATTACK's deprioritize
  test (`:6366`, the 51 roll at `:6374`). Its kill bonus (`:6394`) is unreachable: a no-calc
  move never kills.
- BLIZZARD (260, Kaizo Blizzard, Ice Ball and Sheer Cold): `Expert_Blizzard` (`:1801`; `:6216-6234`),
  resisted -3 unless the 50 roll (`:6229`) jumps, else in hail +1.

### 5.20 Roar and Whirlwind, Substitute, Dream Eater (E1 batch 17)

- FORCE_SWITCH (28): `Basic_CheckCanForceSwitch` (`:499-509`; `:153`), the player with no
  other living Pokemon -10, else unless Mold Breaker against Suction Cups (a guessing read)
  -10; `Expert_ForceSwitch` (`:1650`; `:2574-2611`): the player in more than 3 turns (the
  new command `LoadBattlerTurnCount`, `totalTurns - fakeOutTurnNumber`) +2 unless `:2603`
  jumps then +2 unless `:2607` jumps; hazards on its side (`IfSideCondition` now reads
  Spikes and Toxic Spikes from their layer counts) or a stat above +2, the second try only;
  else -3.
- SET_SUBSTITUTE (79): `Basic_CheckCannotSubstitute` (`:633-637`; `:181`);
  `Expert_Substitute` (`:1684`; `:2976-3042`), with the new command `LoadEffectOfLoadedMove`
  on the player's last move and `IfNotVolatileStatus`.
- RECOVER_DAMAGE_SLEEP (8): `Basic_CheckDreamEater` (`:329-333`; `:136`); `Expert_DreamEater`
  (`:1631`; `:1886-1905`), the player asleep +3 unless the 51 roll (`:1897`) jumps.
- Knock Off, Imprison, Mean Look and Minimize waited on E2's state; they came in E1 prep
  batch P1 (5.21).

### 5.21 Metal Burst, Future Sight, ACC_DOWN, Minimize, Mean Look, Knock Off, Imprison (E1 prep batch P1)

On E2's state (`ai_item_known`, `ingrain`, `mean_look`, `imprison` / `imprisoned`,
`future_sight`, `curse`).
- New commands: `IfHeldItemEqualTo` (trainer_ai.c; the AI side's true item, the player's
  side the AI's record of it, `ai_item_known`), `LoadHeldItem` (the true item),
  `IfNotMoveEffect`.
- METAL_BURST (227): `Basic_CheckMetalBurst` (`:1160-1182`, FASTER's tie a read, Ruling R8;
  the Shiny Stone compare is the decomp's Lagging Tail quirk); `Expert_MetalBurst`
  (`:5029-5083`).
- HIT_IN_3_TURNS (148): `Basic_CheckFutureSight` (`:823-827`), Future Sight pending on
  either side -12; `EvalAttack_ApplyKillBonuses` `:6396`, a killing Future Sight +4 unless
  the 170 roll (`:6405`) jumps.
- ACC_DOWN (23) and EVA_UP_2_MINIMIZE (108): `Basic_CheckLowStatStage_Accuracy` /
  `Basic_CheckHighStatStage_Evasion`; `Expert_StatusAccuracyDown` (`:2431-2498`) and
  `Expert_StatusEvasionUp` (`:2184-2253`), the player below 40 % (a per-effect HP cut).
  EVA_UP (16, Double Team, E1 trial batch) shares both labels (`:143`, `:1639`) and the
  40 % cut; it is also in the SetupFirstTurn and CheckHP tables, so its read bounds add the
  CheckHP reads that Minimize's lack.
- PREVENT_ESCAPE (106): `Basic_CheckMeanLook` (`:674-677`); `Expert_BindingMove`
  (`:2808-2825`).
- REMOVE_HELD_ITEM (188): `Basic_CheckCanRemoveItem` (`:898-904`); `Expert_KnockOff`
  (`:4571-4581`), the player below 30 %.
- MAKE_SHARED_MOVES_UNUSEABLE (192): `Basic_CheckCanImprison` (`:917-921`);
  `Expert_Imprison` (`:4634-4642`).

### 5.22 Copycat, Ingrain, the chart's Ingrain and Roost, Baton Pass (E1 prep batches P2-P3)

- USE_LAST_USED_MOVE (242): `Basic_CheckCopycat` (`:1383-1390`), on the battle's first turn
  faster (FASTER's tie a read) -10; `Expert_Copycat` (`:5552-5580`) with its
  `EncouragedMoves` table (`:5582-5630`, `commands.COPYCAT_ENCOURAGED_MOVES`) and the new
  command `IfBattlerDealsMoreDamage` (trainer_ai.c:2187-2240): the AI's best
  `TrainerAI_CalcAllDamage` against `TrainerAI_CalcDamage` of the player's last move with
  the player as attacker **and** defender (`AI_CONTEXT.defender`) and the AI attacker's IVs,
  both at variance 100, strictly greater. There is no damaging-move test: a status move or
  `MOVE_NONE` deals `CalcMoveDamage`'s `+ 2` through the chart (`engine.damage._finish`
  computes it on the AI path; `ai_damage_for_roll` takes `def_side` and `ivs`). The AI's
  estimate never sees Stomp's x2 (`effect_script_0150.s`'s `BTLVAR_POWER_MULTI`; `powerMul`
  is reset at the turn's end, battle_controller_player.c:1855). A player move with a refused
  `TrainerAI_CalcDamage` id case (Gyro Ball, Natural Gift, Judgment) is refused in a fight
  whose AI knows Copycat; Psywave / Triple Axel and Magnitude / Bulldoze draw (8.7).
- GROUND_TRAP_USER_CONTINUOUS_HEAL (181): `Basic_CheckAlreadyIngrained` (`:906-909`);
  `Expert_Ingrain` (`:4463-4465`, no change).
- The AI's `ApplyTypeChart` (`switching.chart_multipliers`, `defender_chart_flags`) reads the
  defender's Ingrain (the Flying immunity lifted) and `turnFlags.roosting` (every Flying row
  skipped), `BasicTypeMulApplies` (battle_lib.c:2534-2543): `IfMoveEffectivenessEquals`,
  `IfHasSuperEffectiveMove`, the switch check's keep draws and the replacement's stage 2.
  `CalcEffectiveness` has neither (`NoImmunityOverrides`, :2759). Roosting is cleared before
  command selection (battle_lib.c:2229-2231), so only the replacement after a faint reads it.
  Roost and Ingrain are no longer refused on either side; Miracle Eye and Magnet Rise are.
- PASS_STATS_AND_STATUS (127): `Basic_CheckBatonPass` (`:780-784`); `Expert_BatonPass`
  (`:3651-3704`, rolls `:3680`).
- AI_FLAG_BATON_PASS (bit 6): `BatonPass_Main` (`:6564-6635`), rolls `:6577` and `:6591`.
  As written, the `+3` of `BatonPass_EvalMove` (`:6592`) falls into `_SetupAtHighHP`. The
  read bounds gain its column; the widest Kaizo fight with bit 6 (Danny, 725) bounds at 14,
  under `WIDTH_CAP`.
- The type-chart bonuses (Filter, Solid Rock, Expert Belt, Tinted Lens) need the move's
  table power (`ApplyTypeMultiplier`'s `update`, battle_lib.c:2627, :7558-7568; :2656, :2667):
  the status move or `MOVE_NONE` Copycat's test computes takes none. Every Kaizo trainer's
  fight fits `WIDTH_CAP` (tests/ai/test_layout.py; the widest, Jason 82 and Danny 725, at 14).

### 5.23 Helping Hand and SolarBeam (E1 prep batch P4)

- BOOST_ALLY_POWER_BY_50_PERCENT (176): `Basic_CheckHelpingHand` (`:892-896`), a singles
  battle -10 (`LoadBattleType`, the new `IfLoadedNotMask`); no Expert entry.
- SKIP_CHARGE_TURN_IN_SUN (151): `Expert_ChargeTurnNoInvuln` (`:1716`; `:3933-3972`): resisted
  or immune -2; in sun +2; else Power Herb (its true item) +2; else the player known to have
  a Protect effect -2, or the attacker at 38 % HP or less -1. Expert_Main's second entry for
  this effect (`:1719`, Expert_Thunder, the decomp's Thunder bug) is never reached. Its other
  effect, CHARGE_TURN_DEF_UP (`:1715`, Skull Bash), is transcribed since E2 batch 10 to the
  same label (`policy.py`, `_expert_charge_turn_no_invuln`).
- Encore waits on an encore state (`IfBattlerUnderEffect CHECK_ENCORE`, `:657`).

### 5.24 AI_FLAG_HARRASSMENT (E1 Roark close batch, G-37)

- `Harrassment_Main` (`:8021-8034`), FlagTable bit 10 (`:29`): `IfTargetIsPartner` (never in
  singles), `LoadCurrentMoveEffect`, `IfLoadedNotInTable Harrassment_Effects` (`:8036-8071`,
  34 effects, `commands.HARRASSMENT_EFFECTS`), then `IfRandomLessThan 128` (`:8030`): +2 when it
  does not jump. One read per path for a table effect, none for any other (`read_bounds`
  column 10). `policy._harrassment`; `layout.KAIZO_FLAG_BITS` and coverage's
  `IMPLEMENTED_AI_FLAGS["kaizo"]` carry bit 10.

### 5.25 E1 trial batch: Perish Song (G-64), Nature Power, Ghost Curse, Endure, Magnet Rise, Gravity

- ALL_FAINT_3_TURNS (114): `Basic_CheckPerishSong` (`:720-723`), the player already perishing
  -10; no Expert entry. Switching step 3, `AI_PerishSongKO` (`trainer_ai.c:3263-3272`): the AI's
  active with the effect at `perishSongTurns == 0` switches before any draw
  (`switching._Layout.perish`; the pick is `BattleAI_PostKOSwitchIn`'s).

- NATURE_POWER (173): no Basic or Expert entry, only `Harrassment_Effects` (`:8058`). It is
  scored as its own status move (power 0): the called move is chosen after the AI picks.

- Ghost Curse (CURSE, 109): `Basic_CheckCurse_GhostType` (`:700-705`) and
  `Expert_Curse_GhostCheckHP` (`:3495-3500`) are transcribed on the Ghost-type jumps
  (`policy._basic_check_curse`, `_expert_curse`); `policy._refused` no longer refuses a Ghost
  Curse user and `opponent_move_refusals` has no "ghost curse" kind. The engine side was
  modelled in E2 batch 8 (B8-5).

- SURVIVE_WITH_1_HP (116, Endure): no Basic entry; `Expert_Endure` (`:3633-3649`, dispatch
  `:1704`): the AI below 4 % HP **-1**; below 35 % **+1** unless the 70 roll (`:3645`) jumps; at
  35 % or more the fall-through reaches `Expert_Endure_ScoreMinus1`, **-1**. The engine move is
  G-65.

- GIVE_GROUND_IMMUNITY (252, Magnet Rise, G-66): `Basic_CheckMagnetRise` (`:1461-1474`), -10 for
  the attacker with the bit, Levitate or a Flying type; `Expert_MagnetRise` (`:6090-6119`, dispatch
  `:1795`): the AI below 50 % HP ends; a player known to have Earthquake, Earth Power or Fissure
  **+1**; a Ground-type player **+1**, else **+1** unless the 128 roll (`:6113`) jumps. The
  player's Magnet Rise now reaches the AI's chart (`switching.defender_chart_flags`): a Ground move
  gets `MOVE_STATUS_MAGNET_RISE` (immune) and no walk.

- GRAVITY (215, G-67): `Basic_CheckGravityActive` (`:1026-1029`), -10 with Gravity up
  (`commands.FieldCondition.GRAVITY` = the three counter bits, read as `state.gravity != 0`);
  `Expert_Gravity` (`:4782-4803`, dispatch `:1759`): a Levitate (guessed or known), Magnet Rise
  (the bit) or Flying player goes to `TryScorePlus1` (`:4798`, **+1** unless the 64 roll `:4799`
  jumps); otherwise the AI below 60 % HP ends and the 128 roll (`:4795`) either goes there (jump)
  or ends. The AI's own move list under Gravity follows `actions.usable_moves` (the banned moves
  are invalid, `trainer_ai.c:234`).

### 5.26 E1 trial batch: Mimic (G-68), Transform (G-69), Chatter (G-70)

- COPY_MOVE_FOR_BATTLE (82, Mimic, G-68): no Basic or Expert entry and no other `script.s` mention;
  the generic paths score a status move of power 0. The AI's own copied move is scored by the
  copied move's effect label (`layout.fight_effects`): a fight with a Mimic or Transform user
  holds the effects of the foe's possible moves (`layout.fight_tables(foe_moves=...)`).
- The seen-move record is move ids now (`BattleState.ai_seen_moves`, int16[6, 4], first-free order,
  `TrainerAI_RecordLastMove` `trainer_ai.c:2705-2716`; `state.record_seen_move`): a player that
  Mimics shows Mimic and then the copy, and its own Struggle is recorded like any move.

- TRANSFORM (57, G-69): no Basic or Expert entry and no script.s mention; `trainer_ai.c` never tests
  `VOLATILE_CONDITION_TRANSFORM`, so the transformed BattleMon is read like any other. The AI's
  recorded ability is the decomp id (`BattleState.ai_ability_known`, `battlerAbilities`): a reveal
  before the Transform stays the old ability, an unrevealed one is guessed from the copied species.

## 6. Where the AI reads the true battle state

roark.md 6 holds: the damage estimate, the kill check and the effectiveness class read the
real state (roark.md 7; in a sandstorm the estimate includes the Rock-type Special Defense
boost behind the Cloud Nine gate, 7.2). Only the player's moves and ability go through the
AI's knowledge (section 9). Kaizo Roark's labels add these reads of the real state, each by
the command's handler:

| Read | Handler | On Roark's paths |
|---|---|---|
| `A`'s and `D`'s HP% (`curHP * 100 / maxHP`) | `IfHPPercent*`, `trainer_ai.c:588-646` | `A`: Recover `script.s:513`, `:2649`, `:2664`; Explosion `:1863`, `:1869`, `:1874`; Toxic `:2689`; Light Screen `:2716-2717`. `D`: Toxic `:2694` |
| stat stages | `IfStatStage*`, `trainer_ai.c:1465-1523` | `D`'s evasion `:1856`, `:1858`; `A`'s Speed `:367`, `:371`; `A`'s Attack and Defense `:690-691`, `:696-697`, `:3471`, `:3485`, `:3490` |
| side conditions | `IfSideCondition`, `trainer_ai.c:732-745` (the side's `sideConditionsMask`) | Light Screen on `A`'s side `:551`; Reflect and Light Screen on `D`'s `:4561-4562`; Safeguard on `D`'s `:546`; Stealth Rock on `D`'s (roark.md 5.1) |
| `D`'s status | `IfStatus`, `trainer_ai.c:648-659` | `:545`, `MON_CONDITION_ANY` |
| Trick Room | `IfFieldConditionsMask`, `trainer_ai.c:1936-1945` | `:364` |
| the weather, raw | `LoadCurrentWeather`, `trainer_ai.c:1414-1439` | `:535`, `:541` (7.2) |
| true types | `LoadTypeFrom`, `trainer_ai.c:937-959` | `D`'s `:521-526`; `A`'s (Ghost Curse) `:681-684`, `:3467-3470` |
| `A`'s own ability | `LoadBattlerAbility ATTACKER`, `trainer_ai.c:1208` | Mold Breaker `:69-70`, `:302-303`; Simple `:365-366`, `:688-689`; Rock Head and Magic Guard `:6296-6298` |
| `A`'s own moves | `IfMoveKnown`, `IfMoveEffectKnown ATTACKER` (`trainer_ai.c:1627`; `:1750-1760`, the ATTACKER case), `IfAttackerHasNoDamagingMoves` (`:913-929`, power only) | Gyro Ball and Trick Room `:3472-3473`; SP_DEF_UP and PROTECT `:2699-2700`; `:2688` |
| the living party members of each side | `CountAlivePartyBattlers`, `trainer_ai.c:1126-1156` | `:310`, `:312` (Basic_CheckLastMon); Stealth Rock (roark.md 5.1) |
| the speed order | `IfSpeedCompareEqualTo`, `trainer_ai.c:1102-1112` | `:1864`, `:2070`, `:2650`: true speeds with stages, items and weather, the previous turn's Quick Claw roll, Trick Room (7.1) |
| the player's last used move | `LoadDefenderLastUsedMoveClass`, `trainer_ai.c:2050-2054` (`movePrevByBattler`) | `:2722` (7.4) |
| the move's own effect | `LoadCurrentMoveEffect`, `trainer_ai.c:1164-1168` | `Risky_Main` (5.4) |

**The K3b2 mechanics, as the AI sees them** (the K3b2 handover's list, settled):
- Quick Claw: the speed compare reads the previous turn's roll (7.1).
- Infatuation: no label on Roark's paths reads it. Kaizo Sand Tomb is WHIRLPOOL, which has no
  `Expert_BindingMove` entry (plan K3c Ruling 5; `script.s:1659`).
- Protect: only `A`'s own moves are read (`:2699-2700`, Ruling 9). `LoadProtectChain` and the
  player's seen Protect are on no Roark path.
- Roar: the player's Roar is recorded as a seen move like any other (section 9). No label of
  Roark's reads seen moves except Recover's Snatch test (section 9).
- Camouflage: the types the AI reads are the battler's current ones, which the engine writes
  through (`LoadTypeFrom` reads `BATTLEMON_TYPE_1/2`).
- Nature Power: the class read is Nature Power's own (7.4).
- `AICmd_LoadBattlerSpeedRank` (`trainer_ai.c:2056-2093`) is reached only from `TagStrategy`
  (flag 7), which Roark lacks (Ruling 4).

## 7. The commands Kaizo Roark's labels add (K3c Task 4)

The new script commands are rows of `docs/ai/script_commands.md` (after `IfMoveNotKnown`),
each a function of `nuzline.ai.commands` with a fixed-state test in
`tests/ai/test_kaizo_commands.py`. Tasks 5 and 6 put them on Roark's paths. This section
records the rules that need more than one table row.

**7.1 The AI's speed compare** (`IfSpeedCompareEqualTo`, `trainer_ai.c:1102-1112`).
- It is `BattleSystem_CompareBattlerSpeed(attacker = the AI, defender = the player, TRUE)`.
  `order.compare_speed(tables, state, battler1, tie_roll, qc)` transcribes its
  equal-priority result (FASTER 0, SLOWER 1, TIE 2, `include/constants/battle.h:163-165`;
  `battle_lib.c:1401-1447`) for either side as `battler1`. `order.battler_slower` is "not
  FASTER", and the turn order's `player_second` is its `battler1 = 0` case, unchanged.
- **Quick Claw.** `ignoreQuickClaw` TRUE still reads `speedRand` for both battlers
  (`battle_lib.c:1278-1286`, `:1344-1352`). The flag only skips the `moveEffectsData`
  write there and the priority read (`:1375-1399`), so both priorities are 0. A lone
  activated Quick Claw is FASTER (`:1408-1411`). The AI decides before the turn's
  `speedRand` redraw (`battle_controller_player.c:853-857`), so the compare reads
  `BattleState.qc_roll`, the previous turn's draw, as `action_order` does. So the player's
  Quick Claw proc from the previous turn makes Roark SLOWER (plan Review Focus 2).
- **The tie draw (plan K3c Ruling 3).** At equal speed each branch that compares speeds
  (both Quick Claws `:1402-1407`, both Lagging Tails `:1412-1417`, both Stall `:1422-1427`,
  Trick Room `:1432-1439`, plain `:1440-1447`) gives TIE when `RandNext() & 1` is 1, else
  FASTER, never SLOWER. A SLOWER compare (`script.s:1864`, `:2070`, `:2650`) therefore never
  depends on the draw: the exact command passes `tie_roll = 0`
  (`battler_slower(..., 1, 0, qc_roll)`) with no read, and the decomp still consumes the draw
  (8.7). **E1 Ruling R8:** a FASTER compare (Tailwind `:4993`, Payback `:5177`) does depend on
  it, so on the per-slot layout it is a slot read with p = 1/2 (TIE when the draw's low bit
  is set), counted by the read-bound walker (8.3) and the loose oracle's `SPEED_TIE` bit; the
  flat layout refuses FASTER at trace time, and `COMPARE_SPEED_TIE` is not a value any
  script tests. The both-Stall branch is never on a graded path: Stall is not in coverage's
  implemented abilities, so a state with it is refused (`engine/order.py`, "Custap and Stall
  are unsupported").
- **Loose.** The tie is its own draw (`commands.ReadKind.SPEED_TIE`, probability 1/2). It is
  on the path exactly where the decomp draws it: equal speed in one of those branches,
  which is where `compare_speed` gives TIE for a 1. The loose command compares the
  three-valued result with SLOWER, so it does not reuse the gate.
  `test_speed_tie_never_slower` checks every branch with both bits.

**7.2 Weather.** The AI reads weather in four places:
1. `TrainerAI_CalcDamage` passes `fieldConditionsMask` to `CalcMoveDamage`
   (`trainer_ai.c:3081-3091`). The Rock-type Special Defense boost in sand is inside
   `NO_CLOUD_NINE` (`battle_lib.c:6921`, `:6926-6929`), which counts Cloud Nine and Air Lock
   on every living battler (`include/battle/common.h:63-64`; `COUNT_ALIVE_BATTLERS`,
   `battle_lib.c:2952-2959`). The engine's `ai_damage_for_roll` applies it through
   `_pre_random(..., ai=True)` behind `damage.weather_suppressed`, the same gate
   (`test_ai_damage_sand_spdef_gated`).
2. `LoadCurrentWeather` (`trainer_ai.c:1414-1439`) reads the raw mask through `WEATHER_IS_*`
   (`include/battle/common.h:68-72`), with **no** Cloud Nine check. A permanent sandstorm
   with a Cloud Nine holder still loads `AI_WEATHER_SANDSTORM`
   (`test_load_current_weather_ignores_cloud_nine`). On Roark's paths it is reached only
   from `Basic_CheckCannotPoison` (`script.s:535`, `:541`), for Leaf Guard and Hydration.
3. `IfFieldConditionsMask FIELD_CONDITION_TRICK_ROOM` (`script.s:364`).
4. The weather speed doubling inside `CompareBattlerSpeed` (`battle_lib.c:1228-1238`, through
   `order.effective_speed`).

**7.3 Statuses and side conditions.** `IfStatus` takes a mask made of whole
`MON_CONDITION_*` groups, each one engine `Status`; Roark's path passes
`MON_CONDITION_ANY` (`script.s:545`). `IfSideCondition` takes the decomp's mask bits
(`commands.SideCondition`), each read from its `BattleState` timer.

**7.4 The player's last used move** (`LoadDefenderLastUsedMoveClass`,
`trainer_ai.c:2050-2054`). `movePrevByBattler` is `moveTemp` when the move succeeded, else
`MOVE_NONE` (`battle_controller_player.c:3944-3949`). `moveTemp` is set at command
selection (`battle_controller_player.c:1861-1898`). For Nature Power it stays Nature Power:
- Its script (`effect_script_0173.s`) writes `msgMoveTemp` (`GetTerrainMove`,
  `battle_script.c:8045`) and `moveCur` (`GoToMoveScript`, `:2118`), never `moveTemp`.
- No battle script writes `BTLVAR_MOVE_TEMP` (`battle_script.c:9585-9586`).
  `effect_script_0092.s:6` only compares it. `subscript_pursuit.s:81` and `:117` use
  `UpdateVarFromVar OPCODE_GET`, which copies `moveTemp` into `moveCur`
  (`battle_script.c:3278-3279`).
- So the class read is Nature Power's (status), not the called move's. The engine's
  `last_move` already holds Nature Power (`step._use_move`).
  `test_last_used_move_class_nature_power` checks it, and also the `MOVE_NONE` (physical)
  after a failed move.

## 8. Random draws: the per-slot layout

The draws themselves are roark.md 8's: `BattleSystem_RandNext` is a uniform `u16`, every
`IfRandomLessThan N` jumps with probability N/256 (`trainer_ai.c:528-538`), every
`LoadBattlerAbility DEFENDER` guess is a coin flip (`trainer_ai.c:1191-1198`), and the
tie-break is `RandNext % n` (`trainer_ai.c:338`). What differs is how `AiOutcome` lays them
out (`nuzline.ai.outcome`; `AiLayout.per_slot_sites`).

**8.1 Fields.**
- `sim_roll`: fixed at 100, as roark.md 8.1. Every damage read on the transcribed paths
  passes `USE_MAX_DAMAGE`.
- `script_rolls: uint8[MOVES, KAIZO.slot_reads]` (E1 Ruling R4, 8.2). Entry k of slot m is the
  k-th random read on the slot's scoring path (flag order, `trainer_ai.c:297-310`, then line
  order), as `RandNext() % 256`: an `IfRandomLessThan N` jumps when it is below N, a guessing
  `LoadBattlerAbility DEFENDER` takes ability 1 when its low bit is set (`:1195`), else
  ability 2.
- `ability_guess`: 0 and unread (the flat layout's scalar guess; every per-slot guess is a
  read of `script_rolls`).
- `tiebreak`: `RandNext % 12`, as roark.md 8.4.
- `switch_stop`: as roark.md 8.5, over `KAIZO.switch_sites` = 60 sites, every switch draw
  of a party of six (section 11; 44 until E1 ruling G-33); 60 means no site stops.

**8.2 Reads keyed by order on the path (E1 Ruling R4; replaces K3c's per-effect site table and
batch 1's guess columns).** Every random read of a slot's scoring is one
`BattleSystem_RandNext`:
- `IfRandomLessThan N` (`AICmd_IfRandomLessThan`, `trainer_ai.c:528-538`): `RandNext() % 256 <
  N`, probability N/256;
- a guessing `LoadBattlerAbility DEFENDER` (8.4): `RandNext() & 1` (`:1195`), probability 1/2.

The k-th read on slot m's path takes `script_rolls[m, k]` (`commands._slot_read`, counting in
`AiContext.reads`; a read off the path, or of an empty slot, draws nothing), and its
probability comes from the read itself (`AiContext.weight` multiplies it in). This is exact:
each read is a fresh independent draw, which read is the k-th is decided by the draws before
it on the same path, and `TrainerAI_EvalMoves` scores slot by slot (`:483-525`), so no read is
shared between slots. The reads are named by their `script.s` line (`outcome.SITE_N`, checked
against the script by `test_site_lines_are_the_scripts_rolls`, and at every call of
`commands.if_random_less_than`), but no table says which effect reaches which line: the
script itself does. So a forgotten site cannot be read as never jumping, and a flag set's
reads cost nothing to a move that does not make them.

- `KAIZO.transcribed` is `outcome.KAIZO_TRANSCRIBED` (Kaizo Roark's 24 effects); its comments
  name each effect's Expert label and rolls (5.3).
- **The bound (E1 Ruling R6: per fight).** `AiLayout.slot_reads` is the most random reads
  one slot's path can make in the fight: over the opponent party's moves, the most of the
  sum, over the fight's flag set, of each flag script's reads for that move
  (`layout.fight_slot_reads`). The per-move counts are `ai.read_bounds.READ_BOUNDS`,
  generated by `tools/ai_read_bounds.py`: a walk of every path of `script.s` from each
  transcribed flag's entry, keyed by the move's battle effect, whether it is damaging and
  whether its type can be quadruple effective. The walk decides what the move fixes (the
  effect compares and the effect tables, `FlagMoveDamageScore`'s comparison,
  `IfCurrentMoveKills` for a non-damaging move, the quad roll's effectiveness test,
  `IfMoveEqualTo` against the named move's effect, `IfTargetIsPartner` in singles) and takes
  every other branch both ways, so it is an upper bound: no path makes more, and none is
  refused for it. `tests/ai/test_read_bounds.py` regenerates the table from the decomp.
  `fight_tables` stores the party's rows (`Tables.ai_move_reads`) with its flags, and the
  program caches key on the bound (`AiLayout.key`). Roark's 0x17 is 7 (Selfdestruct: `:117`
  and `:304` guesses, `:6374`, `:1859`, `:1870`, `:1875`, Risky's `:6530`), as before. A
  bound whose scoring width (`switch_sites + 1 + 4 x 2^slot_reads`) passes `WIDTH_CAP` is
  refused (`layout.kaizo_layout`; a coverage line). Replaces Ruling R5's per-set bound, a base
  measured by a random sweep, which every fight would have paid for the heaviest effect
  (the walker gives the evasion raises 11 reads under Roark's flags and 14-15 under every
  flag). The walk assumes singles (`IfTargetIsPartner` never jumps; 24 rows, one party) and
  that an AI Adaptability Pokemon stays refused (with Adaptability one super-effective type
  makes a quadruple class, `battle_lib.c:2592-2594`); both change with their milestones. A path that makes more than
  the bound is still a path refusal (8.3), a layout bound, never a reused draw.
- **Enumeration.** `policy` enumerates the reads as bit patterns: bit 0 / 1 of read k stands
  for the draw `outcome.READ_DRAWS` = (254, 1), which split every read (254 is below no
  `SITE_N` N and even, 1 below every one and odd). A pattern whose bits past a slot's reads
  are not all 0 repeats a shorter pattern and gets weight 0 (`policy._canonical`), so each
  path is counted once.

**8.3 The exact distribution, factored per slot** (`policy._slot_move_distribution`).
- `TrainerAI_EvalMoves` scores the move slots one by one (`trainer_ai.c:483-525`).
  Slot m's score reads only its own draws. `FlagMoveDamageScore` compares the damage
  estimates of all four moves, but those are deterministic (`trainer_ai.c:1012-1062`), not
  draws.
- So the scoring runs once per local bit pattern p (`2^slot_reads` patterns, 2^7 = 128 under
  Roark's set; the same p in every
  slot), and slot m's score and path weight with its reads p are read off. Each slot's
  histogram over score values 0-255 adds the path weight of p, the product of the
  probabilities of the reads the path made (0 for a non-canonical pattern, 8.2).
- The pick (`trainer_ai.c:315-338`): slot m with value v is picked when every other
  candidate ties at v or is below it. For each subset T of the other candidates that ties,
  m's share is `P(RandNext % (|T| + 1) == m's place in the tied list)`.
- The switch comes first and the item next, as `_distribution` does (roark.md 2).
  Struggle and the locked slot are unchanged.
- **Path refusals (K3c final review; E1).** Two things the static refusals (`_refused`) cannot
  see are refused on the scoring path, so the per-slot layout never ranks a score it
  cannot vouch for:
  - `moveScore` is `s8` (`include/battle/ai_context.h:14`), and `AddToMoveScore` adds then
    clamps a negative result to 0 (`trainer_ai.c:581-585`), so a sum past 127 wraps
    negative and ends at 0. `commands.add_to_move_score` models the wrap; a score outside
    0-127 makes the distribution NaN. Kaizo Roark's largest score is about 108.
  - A read past the slot's `slot_reads` draws (8.2) is recorded by `commands._slot_read`
    (`AiContext.missing_read`), and `_scores` gives -1, so the distribution is NaN and
    `_policy` refuses.
  - On a concrete state `ai_scores`, `ai_policy` and `ai_action_distribution` raise
    `UnsupportedAi` for either. `policy.ai_path_refusals` names the slot and move past the
    bound (else `policy._PATH_REFUSAL`), and Stage A's `refusal_reason` and Stage B's root
    refusal use it after `ai_refusals`. A caller's `SWITCH_REFUSED` stop still gives
    `REFUSED_ACTION`, not this error. Tests: `tests/ai/test_kaizo_commands.py
    test_add_to_move_score_wraps_the_s8`, `test_a_score_past_the_s8_is_refused`;
    `tests/ai/test_kaizo_outcome.py test_a_read_past_the_bound_is_refused`.

**8.4 The ability guesses (E1 Ruling R1; replaces plan K3c Ruling 6).**
`AICmd_LoadBattlerAbility` (`trainer_ai.c:1170-1209`) draws a fresh coin,
`BattleSystem_RandNext(battleSys) & 1` (`:1194-1198`), at every `LoadBattlerAbility DEFENDER`
read of a decision when the ability is not suppressed (`:1177`), the AI does not know it
(`:1181`), it is not Shadow Tag, Magnet Pull or Arena Trap (`:1185-1188`), and both species
abilities are nonzero (`:1191-1194`). Whether a read guesses is therefore the same for every
read of the decision (`commands.ability_guessed`); which reads are made depends on the path.
- **The rule.** Each guessing read is the slot's next random read (8.2): a fair coin,
  independent of every other, ability 1 (the species' slot 0) when the draw's low bit is set.
- **The reads.** On the transcribed paths they are all in BASIC: `script.s:71`
  (`Basic_CheckForImmunity`), `:117` (`Basic_CheckSoundproof`), `:470`
  (`Basic_CheckClearBodyEffect`, DEF_DOWN and DEF_DOWN_2 only: `:145`, `:172`), `:304`
  (`Basic_CheckCannotExplode`, Damp), and `:529` and `:539` (`Basic_CheckCannotPoison`:
  Immunity, Magic Guard, Poison Heal, Leaf Guard in sun; Hydration in rain). EVAL_ATTACK,
  EXPERT and RISKY load no defender ability on them (`Expert_RecoilMove` loads the
  attacker's, `:6296`).
- **Examples one scalar guess got wrong.** Leer and Screech in two slots against Tentacool
  (Kaizo abilities 44 / Clear Body): each `:470` read changes its slot's score, and the slots
  guess apart (5/24 each, where one scalar gave 1/6; `test_each_slot_guesses_on_its_own`).
  Screech against a species with Soundproof and Clear Body (made up): the `:117` and `:470`
  reads of one slot are two coins, and Screech keeps 100 one time in four
  (`test_two_guesses_in_one_slot`). Both equal the loose oracle (8.6).
- The flat (vanilla) layout keeps its one scalar (roark.md 8.2), bit-identical.
- Coverage has no ability-guess gate: the rule is the decomp's for every trainer.

**8.5 Width** (plan K3c Ruling 12; E1 Rulings R1, R4). `policy.ai_width` counts the
combinations the factored distribution merges: the switch stops plus each slot's distinct
scoring paths (`policy._path_counts`: the canonical patterns), which
`tests/ai/test_kaizo_outcome.py test_kaizo_width_under_cap` and the Roark audit check against
the loose oracle's own count of paths. It is at most `45 + 4 x 2^slot_reads` for any state:
557 under Roark's set, 2093 under 0x10f, 8237 under every transcribed flag at once (checked at
import for the widest set, `policy._check_width`), against `WIDTH_CAP` 131072; the program
always evaluates `2^slot_reads` scorings. `policy.ai_flat_width` reports the flat product, `prod_slots (paths) x 12 x
45`, which is not used. Section 14 has the measured numbers.

**8.6 The loose oracle** (tests only; `policy.ai_action_distribution_loose`). Every
`RandNext` read of the scoring is its own independent draw, as in the decomp: each
`IfRandomLessThan` with its own N (the literal at its call), each `LoadBattlerAbility
DEFENDER` its own coin, the tie-break `RandNext % n` over a `u16`. Its reads are indexed in the
static order of the straight-line scripts, not by order on the path, and are not bounded. It
finds every read on every path by enumerating the reads found so far until no new one
appears, then enumerates the slots jointly when they have at most 18 reads in all, and per
slot otherwise (exact in loose, where no two slots share a draw). The gates of the exact
layout are each checked against it (`tests/ai/test_kaizo_outcome.py` docstring):
- per-slot factoring and the tie subsets, against a joint enumeration of every slot's reads
  and the 12 tiebreaks (`test_factored_equals_joint_bruteforce`);
- the read order with each read's own probability (8.2, 8.4): `test_exact_equals_loose` on
  random states, `test_each_slot_guesses_on_its_own`, `test_two_guesses_in_one_slot`, the
  traced states of section 13, two Pokemon of other Roark-split trainers, and the Roark audit
  (section 14); and the count of distinct paths per slot (8.5);
- the read bound (8.2): `test_slot_reads_fit_every_transcribed_effect`.

The fourth gate is the dropped speed-tie draw (plan K3c Ruling 3; 7.1). At equal speed
`BattleSystem_CompareBattlerSpeed` returns FASTER or TIE, never SLOWER
(`battle_lib.c:1401-1447`), and every compare on Roark's paths tests SLOWER.
`IfSpeedCompareEqualTo` gives the tie its own loose bit (`ReadKind.SPEED_TIE`), and
`tests/ai/test_kaizo_commands.py test_speed_tie_never_slower` checks the gate in every
branch. Since Task 6 the compare is on Roark's paths (`:1864`, `:2070`, `:2650`), and
`tests/ai/test_kaizo_policy.py test_exact_equals_loose_at_equal_speed` checks exact against
loose at equal speed on each of the three, with the loose tie read on the path, and on the
player's Quick Claw (no tie read).
- `LOOSE_READS` is 48: every read of the straight-line scripts counts in every slot, on the
  path or not, and Task 6 brings them to 35.

**What the loose oracle does not check: the switch.** `ai_action_distribution_loose` builds
its switch part from the exact path's own `switch_sites`, `switch_stop_probs` and
`switch_action`, combined as `_distribution` does. So an exact==loose comparison through it
(the Roark audit, `tests/kaizo/test_roark_ai_audit.py`, section 14) checks only the scoring
and the move pick; the switch categorical is compared with itself. The switch has its own
independent oracle and tests instead (11.1, "The loose check"): `_loose_switch`, which walks
`trainer_ai.c:3906-3987` with every read its own draw on states with at most 12 reads, and
`test_six_bench_layout_fits`, which reaches the 44 keep and pass sites against a closed form
(`test_kaizo_switch_sites_reach_sixty` reaches all 60 with Wonder Guard, ruling G-33).

**8.7 RNG read order (for K3d's trace replay).** Every read is one `BattleSystem_RandNext`
(`battle_system.c:1308`), made in this order in one decision. The reads marked *merged* have
no `AiOutcome` field because they cannot change the action, but they still advance the RNG,
so a replay that tracks the RNG must count them.

0. **Precondition: which steps run at all.** The AI's command is picked by
   `TrainerAI_PickCommand` from `Task_TrainerSetCommandSelection`
   (`battle_display.c:3411`; steps 1-2), and its move by `TrainerAI_Main`, called only from
   the move-select-menu task (`battle_display.c:3590`; steps 3-8). The controller skips
   them in three cases:
   - When `Battler_CanPickCommand` is FALSE (recharging, thrash, uproar or move-locked,
     `battle_lib.c:6224-6236`), command selection is skipped and FIGHT is forced
     (`battle_controller_player.c:334-338`): **no AI read at all** that turn.
   - Under STRUGGLING_ALL (`BattleSystem_CheckInvalidMoves`,
     `battle_controller_player.c:433-442`) the battler Struggles, and under Encore
     (`:443-455`) the Encored slot is set directly. Neither reaches
     `COMMAND_SELECTION_MOVE_SELECT_INIT` (`:456`, the only state that shows the move menu,
     `:502-503`), so **only steps 1-2 draw**; steps 3-8 do not happen. Struggle is not an
     AI choice.
   This step is about the RNG count only: the model's action for Struggle and for a
   move-locked battler is already the game's (`policy._pick` gives action 0 with no usable
   move; `policy._policy` overrides a locked battler with `actions.locked_slot`).
1. **`TrainerAI_ShouldSwitch`** (section 11), as reached. None (`trainer_ai.c:3906-3913`)
   if `A` is trapped (`VOLATILE_CONDITION_TRAPPED`, bind or Mean Look), has Ingrain
   (refused, 11.3), the player's side has Shadow Tag or Arena Trap, or another battler has
   Magnet Pull while `A` is Steel-type; none either if no other Roark Pokemon lives
   (`:3915-3940`; the checks after that are under `if (alivePartyMons)`, `:3940`).
   Otherwise:
   1. *merged:* `AI_HasAbsorbAbilityInParty`'s `RandNext() % 3` (`:3651`), when `A` has a
      super-effective move (`AI_HasSuperEffectiveMove(..., TRUE)` itself draws nothing,
      `:3586-3587`). Its later draw (`:3706`) needs a bench Flash Fire, Water Absorb or Volt
      Absorb, which Kaizo Roark lacks.
   2. The keep draws, `RandNext() % 10` (`:3588`), one per super-effective slot of `A` in
      slot order, until one is nonzero (then no further read).
   3. Unless `A` has 4 or more positive stages (`:3969`): pass 1 (`0x8`, `% 2`, `:3975`),
      then pass 2 (`0x4`, `% 3`, `:3981`). A pass reads nothing unless `moveHit[A]` is set,
      its user `moveHitBattler[A]` is set (`:3737-3739`) and the move has nonzero power
      (`:3742-3744`; a status move such as Toxic or Growl sets `moveHit` too,
      `battle_controller_player.c:4498-4502`, so it blocks the pass). It then reads
      `RandNext() % rand` (`:3796`) per (bench Pokemon, move) in party then slot order, only
      for a bench Pokemon that `moveHit` meets with the pass's bit
      (`effectiveness & checkEffectiveness`, `:3767-3779`), and only for its moves that are
      super-effective against `moveHitBattler` (`:3785-3796`; the read is short-circuited
      behind that test), until one is zero (a switch, and no further read of the decision).
      Section 11.1 has the full rule.

   The draws of `AI_CannotDamageWonderGuard` (`:3339`), `AI_OnlyIneffectiveMoves`
   (`:3459`, `:3477`, `:3518`, `:3536`) and `AI_IsAsleepWithNaturalCure` (`:3828`, `:3835`,
   `:3853`) are on refused states only (`switching_refusals`, 11.3).
2. `TrainerAI_ShouldUseItem` makes no draw (`trainer_ai.c:4056-4199`). If the decision ended
   with a switch, nothing below happens.
3. **`TrainerAI_Init`**: *merged:* four damage rolls, `RandNext() % 16` (`:240`), one per slot,
   empty slots included. No transcribed path passes anything but `USE_MAX_DAMAGE` (8.1).
4. **BASIC**, slots 0-3 (`TrainerAI_EvalMoves`, `:483-525`; an empty or 0-PP slot runs no
   command, `:486-503`). In a slot, each `LoadBattlerAbility DEFENDER` that guesses draws
   `RandNext() & 1` (`:1191-1198`): the player's ability is not suppressed, not known, not
   Shadow Tag, Magnet Pull or Arena Trap, and both species abilities are nonzero (**even when
   they are the same ability**, `:1194`). In line order of the path taken:
   - `script.s:71` (`Basic_CheckForImmunity`): a damaging move (`:62-63`) that is not IMMUNE
     (`:68`), when `A` is not Mold Breaker (`:69-70`);
   - `:117` (`Basic_CheckSoundproof`): unless the slot already ended (`ScoreMinus10` and
     `ScoreMinus12` end it with `PopOrEnd`, `:1591-1597`);
   - the effect's label (`ScoreMinus10` ends the slot's BASIC script with `PopOrEnd`,
     `:1591-1593`; the later flags still run on the slot):
     - `:304` (Selfdestruct), unless `IfMoveEffectivenessEquals TYPE_MULTI_IMMUNE` at `:299`
       ended the slot (Selfdestruct is Normal, so a Ghost-type `D`), and not when `A` is Mold
       Breaker (`:302-303`);
     - `:529` (Toxic), unless the type tests `:521-526` ended the slot;
     - `:539` (Toxic), unless the type tests, the `:529` guess being Immunity, Magic Guard or
       Poison Heal (`:530-532`), or Leaf Guard in sun (`:534-536`) ended the slot. Whether
       `:539` is drawn therefore depends on the `:529` coin.
   The exact layout gives each guessing read its own draw, slot by slot in path order
   (8.2), so the RNG count of these steps is the number of guessing reads reached.
5. **EVAL_ATTACK**, slots 0-3: `:6374` (Selfdestruct, 51), then `:6383` (80) only for a
   QUADRUPLE move that neither kills (`:6358`) nor is out-damaged (`:6361-6362`), as
   reached (5.2). Selfdestruct never kills and never takes the -1 (5.2 items 1-2), so it
   always reaches `:6374`.
6. **EXPERT**, slots 0-3: the label's `IfRandomLessThan` reads in line order of the path taken
   (5.3, 8.2), and, where the label's `IfSpeedCompareEqualTo` (`:1864`,
   `:2070`, `:2650`) meets equal speeds in a branch that draws, *merged:* the tie's
   `RandNext() & 1` (`battle_lib.c:1405`, `:1415`, `:1425`, `:1437`, `:1445`; Ruling 3, 7.1).
   The compare comes before the label's next roll: Explosion `:1864` before `:1865`,
   Rock Polish `:2070` before `:2075`, Recover `:2650` before `:2665`.
7. **RISKY**, slots 0-3: `:6530` (128) for Selfdestruct and AncientPower.
8. **The tie-break**, `RandNext() % n` (`trainer_ai.c:338`), always, even with n = 1.

A post-KO replacement (`BattleAI_PostKOSwitchIn`, `battle_lib.c:7923-8089`) draws nothing.

### 8.6 Two shapes: the fight's own layout and `ALLIN` (E1 batch AI-1, 2026-11-15)

The user chose one transcription in two program shapes (`ai.layout` module docstring).

- **The exact distribution stays per fight.** The search and the exact == loose checks use the fight's own layout: its flags, own effects (R7) and read bound (R6), all static.
  - Its `2**slot_reads` scorings stay under `WIDTH_CAP`.
  - The exact entry points refuse `ALLIN` with a `ValueError`.
- **The sampled AI can run `ALLIN`, one program for every Kaizo fight.** `ALLIN` is used by `sample_ai_outcome` and `ai_policy`.
  - It carries every transcribed flag bit and effect, and the bound `ALLIN_READS`: the largest row sum of `read_bounds.READ_BOUNDS`, which no fight's path under any of its flags passes.
  - The trainer's flags are read at run time: `layout.thinking_mask`, the trainer's `tables.ai_flags` plus TAG_STRATEGY in a doubles battle (`trainer_ai.c:245-255`).
  - A loop takes `ALLIN` only once its player and its opponent take separate layouts. Today's `playout`, `GREEDY` and `LINE` players read the exact distribution, so they stay per fight; a fixed or learned player does not. `_run_flags` runs each flag script and keeps its result only when the bit is set (`trainer_ai.c:297-308` runs a set bit's script only).
  - Otherwise it restores every context field the script could write, `reads`, `weight` and `missing_read` among them. So a masked script consumes no random read, and the next script's reads take the same `script_rolls[slot, k]` as in the fight's own layout (8.2).
- **Draws do not depend on the read bound.** `sample_ai_outcome` draws read column `i` from `fold_in(key, i)`, so one key gives both shapes the same reads.
- **Refusals under `ALLIN`.**
  - Unset flags, or flags with a bit the policy does not transcribe, are refused as a policy refusal.
  - The Expert-label refusal follows the fight's own EXPERT bit.
- **Tests:** `tests/ai/test_allin.py`.
  - The same action as the fight's own layout for every key, over a cover of the real flag sets in the fast tier and over all of them in the slow tier.
  - The same read counts, path weights and scores.
  - The masked scripts would have read.

### 8.7 The damage estimate's draws (E1 Roark close batch, G-39)

- **The draws.** `TrainerAI_CalcDamage` switches on the move id (`trainer_ai.c:2885`):
  MOVE_PSYWAVE (149, Kaizo Triple Axel) deals `level * (RandNext() % 11 + 5) / 10` as fixed
  damage (`:3014-3016`; a 0 at level 1 falls to `CalcMoveDamage` at the table power, `:3073`),
  MOVE_MAGNITUDE (222, Kaizo Bulldoze) takes the power of `RandNext() % 100` bucketed below 5,
  15, 35, 65, 85, 95 into 10-150 (`:3028-3048`). Their laws over a u16 are
  `outcome.CALC_DRAW_P` (11 classes of 5958 or 5957 values; 7 buckets of 3280, 6560, 13120,
  19651, 13100, 6550, 3275).
- **Who draws.** Every estimate of such a move draws afresh: `FlagMoveDamageScore` (`:1041`, the
  attacker's four moves), `IfCurrentMoveKills` (`:1560`), `IfPartyMemberDealsMoreDamage`
  (`:2130`, `:2156`), `IfBattlerDealsMoreDamage` (`:2210`, `:2228`, the battler's last move with
  no damaging test, at its own level) and `CheckIfHighestDamageWithPartner` (`:2402`, TAG_STRATEGY
  only, not transcribed).
- **One read per command.** Each command's result is one yes/no and its draws reach nothing
  else, so the per-slot layout makes one read per command whose estimate draws
  (`commands._calc_read`), with the exact probability of yes over the draws' joint law: kill
  `sum_c P(c) [hp <= d_c]`; highest `sum_c P(D_slot = d_c) prod_{i != slot} P(D_i <= d_c)`; a
  party member out-damaging `1 - sum_v P(A = v) prod_j P(M_j <= v)`; the battler out-damaging
  `sum_c P(B = b_c) P(A < b_c)`. In the enumeration the read's bit is the result
  (`READ_DRAWS[1]` yes); a sampled outcome carries the raw draws, `AiOutcome.calc_draws`
  (`uint16[MOVES, slot_reads, CALC_DRAWS]`), and the command computes its result from them.
  Basic_Main's and PrioritizeExtremes' and BatonPass_Main's `FlagMoveDamageScore`
  (script.s:62, :112, :6511, :6572) are compared only with `AI_NO_COMPARISON_MADE`: their draws
  are no read (`compared=False`).
- **Bounds.** `tools/ai_read_bounds.py` counts these reads (`read_bounds.DRAWN_READ_BOUNDS` for
  a fight with a drawing move, keyed also by whether the scored move draws; Copycat's
  `IfBattlerDealsMoreDamage` always, the player's move being unknown to the layout). Seventeen
  Kaizo fights' bounds grow by 1 or 2; the widest drawing fight, Barry #6 (479-481), reads 11 (8253 scorings; the widest overall,
  Jason 82 and Danny 725, read 14),
  all within `WIDTH_CAP`.
- **The loose oracle** draws every `RandNext` of every estimate as its own digit
  (`ReadKind.CALC_PSYWAVE`, `CALC_MAGNITUDE`), including the dead ones, and computes each command
  as the decomp does; its patterns are mixed-radix (`policy._loose_patterns`).
- **Tests:** `tests/engine/test_e1_roark_close.py` (family 3).

## 9. What the AI knows about the player

The records are roark.md 9's, kept the same way by the engine (`BattleState.ai_seen_moves`,
`ai_ability_known`): seen moves from `TrainerAI_RecordLastMove` (`trainer_ai.c:2705`, run at
`:295`), the ability from `BattleAI_SetAbility` (`battle_script.c:12106`), both cleared when
the player's battler switches in or faints (`battle_lib.c:2151-2153`, `:2223-2225`). What
Kaizo Roark's paths read from them:

- **Seen moves.** One read: `Expert_Recovery_CheckForSnatch`'s `IfMoveEffectNotKnown
  AI_BATTLER_DEFENDER, BATTLE_EFFECT_STEAL_STATUS_MOVE` (`script.s:2670`), which scans
  `battlerMoves` (`trainer_ai.c:1781-1820`, the DEFENDER case). No Kaizo move has effect 195,
  so the test always jumps and the `:2671` roll is never reached (5.3). The vanilla U-turn read
  (`Expert_Pursuit`, roark.md 9.1) is on no Kaizo Roark path: none of his moves is
  HIT_BEFORE_SWITCH.
- **Ability.** Every `LoadBattlerAbility DEFENDER` resolves as roark.md 9.2: suppressed, then
  known (the true ability), then Shadow Tag, Magnet Pull or Arena Trap (true), then the guess
  (`trainer_ai.c:1170-1209`). The reads on Roark's paths and their draws are 8.4.
- **Not knowledge.** `IfMoveEffectKnown AI_BATTLER_ATTACKER` and `IfMoveKnown ATTACKER` read
  the AI's own true moves (`trainer_ai.c:1750-1760`), and `LoadDefenderLastUsedMoveClass`
  reads the battle's `movePrevByBattler`, not the AI's records (section 6, 7.4).
- The player's held items are recorded (roark.md 9.3) and never read on Roark's paths.

## 10. Picking among the top scores

Unchanged: roark.md 10 (`trainer_ai.c:315-338`). Slot 0 is always a candidate; slots 1-3 when
`A` has a move there; the pick is `list[RandNext() % n]`, encoded by `tiebreak = RandNext %
12` (roark.md 8.4). The per-slot layout computes the same pick from the slot score
histograms (8.3).

## 11. Switching with six Pokemon (K3c Task 7)

Kaizo runs `TrainerAI_ShouldSwitch` (`trainer_ai.c:3894-3987`) and `BattleAI_PostKOSwitchIn`
(`battle_lib.c:7923-8089`) unchanged. `docs/kaizo/rom_diff.md` decodes every change to them
and their helpers as relocated calls: `TrainerAI_ShouldSwitch` (:356), `AI_OnlyIneffectiveMoves`,
`AI_HasSuperEffectiveMove`, `AI_HasAbsorbAbilityInParty` and
`AI_HasPartyMemberWithSuperEffectiveMove` (:351-354), `BattleAI_PostKOSwitchIn` (:373),
`BattleContext_Init` (:377), `BattleSystem_ApplyTypeChart` (:597) and
`BattleSystem_CalcEffectiveness` (:601). The rules are roark.md 11.1 and 11.2. What changes is the
party: Kaizo Roark has six Pokemon, so five can be on the bench.

### 11.1 The switch sites (`KAIZO.switch_sites` = 60)

The draws that can change the result are those of roark.md 11.1:

- **Keep draws.** `AI_HasSuperEffectiveMove(..., FALSE)` (called at `:3964`) loops over the
  active's four move slots (`:3576-3592`). Each slot whose move is super effective against the
  player's active (`BattleSystem_ApplyTypeChart` flags, `:3582`) reads `RandNext() % 10 != 0`
  (`:3588`); a nonzero residue stops the check with "no switch". At most 4.
- **Party passes.** `AI_HasPartyMemberWithSuperEffectiveMove` (`:3727-3807`) is called with
  `MOVE_STATUS_INEFFECTIVE, 2` (`:3975`) and then `MOVE_STATUS_NOT_VERY_EFFECTIVE, 3`
  (`:3981`). Each loops over the party in order (`:3756-3804`). For each bench Pokemon that
  `moveHit` meets with the pass's bit, each of its four moves that is super effective against
  the player (`BattleSystem_CalcEffectiveness`, any power) reads `RandNext() % rand == 0`
  (`:3796`); a zero residue stops the check with "switch to this Pokemon" (`:3797`). At most
  5 bench Pokemon x 4 moves per pass.

So the keep and pass draws number at most `4 + 2 x 5 x 4 = 44`. Against a player's Wonder Guard,
`AI_CannotDamageWonderGuard` (`:3286-3348`, E1 ruling G-23) adds one draw per (bench Pokemon,
super-effective move), at most 5 x 4, but only when the active has no super-effective move, so
never beside a keep draw (E1 ruling G-33): `KAIZO.switch_sites = max(4, 5 x 4) + 2 x 5 x 4 = 60`
(`switching.KAIZO_SWITCH_SITES`; 44 before G-33), and vanilla
keeps `SWITCH_SITES` = 9 (roark.md 11.1). One Pokemon can be in both passes: the two bits
are not exclusive. `UpateMoveStatusForTypeMul` (`battle_lib.c:2781`) sets INEFFECTIVE on an
immune chart row and then NOT_VERY_EFFECTIVE on a later resisted row, so Ground against
Flying/Bug has both. `tests/ai/test_kaizo_switching.py test_six_bench_layout_fits` builds that
case: the active and all five bench Pokemon with four super-effective Rock moves, Earthquake
as `moveHit`, the bench Flying/Bug. It has exactly 44 sites, no refusal, and the stop
probabilities `prod_{h<i}(1 - p_h) * p_i`, with p counted over the u16 residues.

**The bench is every living non-active slot.** The pass skips the active
(`selectedPartySlot`) and `aiSwitchedPartySlot` (`:3762-3765`). `aiSwitchedPartySlot` is 6
whenever `TrainerAI_ShouldSwitch` runs:
- `BattleContext_Init` sets it to 6 (`battle_lib.c:2015`);
- `BattleContext_Init` runs at every turn's end (`battle_controller_player.c:1855`, after
  `ReplaceFainted`) and after every action (`:3998`);
- inside the check only a branch that returns TRUE writes it (`:3797`, and the post-KO
  branches of the refused checks).

So 6 never matches a party index 0-5, and the layout's candidate list (`switching._CANDIDATES`:
4 keep draws, then each pass's 6 slots x 4 moves, the active's slot never a candidate) needed no
change. The only vanilla limit was the 9 sites `AiOutcome.switch_stop` encodes.

**Implementation.** The site count is the layout's: `switch_sites(tables, state, sites)`,
`switch_action(..., sites)` and `switching_refusals(..., sites)` take it (default 9, the
vanilla count, so vanilla is unchanged). `sample_ai_outcome`, `enumerate_ai_outcomes`
(per-slot), `policy.maybe_switch`, `_policy`, `_distribution`, `ai_refusals` and the loose
oracle pass `layout.switch_sites`. `switch_stop = 60` means no site stops under Kaizo
(`SWITCH_NONE`, 9, stays the vanilla value). The "more draws than sites" refusal cannot fire
under Kaizo, since 60 is every candidate a party of six has.

**Merged draws.** `AI_HasAbsorbAbilityInParty`'s `RandNext() % 3` (`:3651`) is read whenever
the active has a super-effective move (`AI_HasSuperEffectiveMove(..., TRUE)` reads nothing,
`:3586-3587`). It cannot change the result without a bench Flash Fire, Water Absorb or Volt
Absorb (`:3655-3713`), which `switching` refuses on the AI's side, so it is not a site. It still
consumes RNG (8.7, K3d). Kaizo Roark's six have no switching refusal
(`tests/ai/test_kaizo_switching.py`, and the pool check below).

**The loose check.** `test_roark_switch_distribution_matches_loose` walks the decomp's control
flow (`:3906-3987`) with every `RandNext` read its own draw, the `:3651` read included, and
enumerates every combination of the reads (at most 12). It reads no site layout. It equals the
switch part of the exact distribution on random states of Roark's six: any active, fainted
bench members, player and bench types, `moveHit`, Gravity, Foresight, trapping and boosts. Four
constructed small layouts reach every bench slot and the last site.
`test_loose_counts_the_absorb_read` shows that the `:3651` read is a loose read with no exact
site.

### 11.2 Replacement

`BattleAI_PostKOSwitchIn` is refused in one case only: stage 2 when the first candidate's move
slot 0 is empty or has power 1. The slot is skipped (`if (move && MOVE_DATA(move).power != 1)`,
`battle_lib.c:8053`), but the compare `maxScore < score` (`:8080`) still runs on the stale
stage-1 score. Kaizo Roark's six never reach it. Each has a slot-0 move whose power is not 1:
Stealth Rock 0, Sludge Bomb 95, DragonBreath 70, Recover 0, Toxic 0, Head Smash 150
(`test_roark_moves_never_reach_the_stale_stage_2_compare`).
`test_roark_replacement_each_faint_order` runs every state a faint order reaches (the
fainted active and every nonempty set of living others, 6 x 31) against four players and
both `powerMul` values: it never refuses and always picks a living non-active slot. Against
Ghost/Dark and Fighting/Psychic none of the six has a super-effective move, so every one of
those states runs stage 2. (The decomp's compare is at `:8080`; roark.md 11.2 and
`replacement.py` cited `:8076`, corrected in Task 7.)

### 11.3 The real pools

`switching_refusals` refuses Levitate, Wonder Guard, Scrappy and Normalize on either side,
Flash Fire, Water Absorb, Volt Absorb and Natural Cure on the AI's, Iron Ball, the state moves
(Magnet Rise, Roost, Miracle Eye, Ingrain, Perish Song), the variable-type moves, and
`AI_OnlyIneffectiveMoves`. Checked 2026-11-11 with each candidate of the
`kaizo_save_pre_roark` and `kaizo_save_probe` pools (12 each, party and box, at cap 16) as the
player's lead against each of Kaizo Roark's six as the active: no line. Roark's six against
his own Gible: no line either. The pools are Sandshrew, Flaaffy, Shinx, Vulpix, Goldeen,
Meowth, Luvdisc, Wormadam, Rhyhorn, Skitty, Prinplup and Delibird. None has a refused ability
(Rough Skin, Static, Rivalry, Flame Body, Swift Swim, Limber, Snow Cloak, Rock Head, Cute
Charm, Torrent, Adaptability), Iron Ball or a refused move.

## 12. Item use

`TrainerAI_ShouldUseItem` (`trainer_ai.c:4056-4199`) runs when `ShouldSwitch` returned FALSE.
Its loop takes each bag slot's item and skips `ITEM_NONE` (`:4089-4096`). Kaizo Roark's bag is
empty (section 4), so every slot is `ITEM_NONE`, the result stays FALSE, and the command is
FIGHT. The engine agrees: `make_state` gives him `trainer_items` all 0 and
`trainer_item_count` 0, and `policy._use_item` only uses a nonzero item, so `maybe_use_item`
never uses one, and `ai_refusals` has no item line.

## 13. Hand-traced decisions (K3c Task 5 and Task 6 fixtures)

`tests/ai/test_kaizo_policy.py`. Common setup:
- Kaizo Roark's party as `build_trainer_party` gives it, the named Pokemon active with the
  moves listed and the bench listed (default: Cranidos). The player's party is one or two
  copies of Roark's own Gible (Dragon/Ground, Sand Veil / Rough Skin, neither of which any
  label tests), at 400/400 HP unless stated: no transcribed move kills it. A "player X" is
  that species put on the player's active slot (its types and ability slot; stats unchanged).
- The scores of K1-K11 (Task 5) are those after BASIC and EVAL_ATTACK; those of E1-E9
  (Task 6) after all four flags, BASIC, EVAL_ATTACK, EXPERT and RISKY. Both start from 100
  (`TrainerAI_Init`, `trainer_ai.c:223`), with refusals not applied; an empty slot scores 0
  (`trainer_ai.c:499-501`).
- Rolls are named by line: "`:6374` jumps" makes that read's draw a jump; every other roll
  does not jump. `g` is the ability guess (0 or 1), taken by every guessing read. The tests
  place each at its order on the path (8.2) with the loose oracle's static read list
  (`tests/ai/test_kaizo_policy.py _outcome`).
- Each test first asserts its inputs: the effectiveness class (`IfMoveEffectivenessEquals`,
  `trainer_ai.c:1311-1340`; type chart `battle_lib.c:2399-2515`, a second type counted only
  when it differs, `:2634-2635`), which slot's damage estimate is highest, and who dies.

**Trace K1: Bonsly [Stealth Rock, Selfdestruct, Brick Break, Accelerock] vs Gible.**
Brick Break (Fighting, neutral) out-damages Accelerock (Rock, resisted by Ground,
`battle_lib.c:2486`); Selfdestruct estimates 0 (no-calc). Walk:
- Stealth Rock: no comparison (`:62-63`), not Soundproof, `Basic_CheckStealthRock`
  (`:1540-1547`): no rocks on the player's side, and the player's other Pokemon alive is 1
  (two Gibles) or 0 (one: **-10**). EVAL: not QUADRUPLE.
- Selfdestruct: no comparison, then `Basic_CheckCannotExplode` (`:297-319`) as 5.1. EVAL:
  `:6365` -> `:6374`; -2 unless it jumps; not QUADRUPLE.
- Brick Break: immunity checks pass, REMOVE_SCREENS has no BASIC entry. EVAL: highest, not
  QUADRUPLE: 100.
- Accelerock: as Brick Break, but not the highest: **-1** (`:6362`).

| Variant | Bench | Player | `:6374` | Scores |
|---|---|---|---|---|
| K1a | Cranidos | 2 | jumps | **[100, 100, 100, 99]** (`:311` -> `:318`) |
| K1b | Cranidos | 2 | no | **[100, 98, 100, 99]** |
| K1c | none | 2 | jumps | **[100, 90, 100, 99]** (`:313`) |
| K1d | none | 2 | no | **[100, 88, 100, 99]** |
| K1e | none | 1 | jumps | **[90, 99, 100, 99]** (`:316`; Stealth Rock `:1546-1547`) |
| K1f | none | 1 | no | **[90, 97, 100, 99]** |

**Trace K2: the Damp guess.** As K1a against player Wooper (Water/Ground) with the synthetic
abilities [Damp, Water Absorb] (vanilla Wooper's; no Kaizo species has Damp). `:6374` jumps.
- `g = 0`, Damp: Selfdestruct **-10** (`:305`): **[100, 90, 100, 99]**.
- `g = 1`, Water Absorb: Selfdestruct goes on to `Basic_CheckLastMon` (no change); Brick Break
  and Accelerock take `Basic_CheckWaterAbsorption` (`:74` -> `:86-89`), not Water: no change:
  **[100, 100, 100, 99]**.
- `ai_ability_known` set (Wooper's slot 0, Damp), any `g`: **[100, 90, 100, 99]**.

**Trace K3: Mold Breaker.** Cranidos in ability slot 1 (Mold Breaker) [Selfdestruct, Crunch],
bench Bonsly, against K2's Wooper: `:302-303` jumps past the Damp read for either `g`;
Crunch skips the immunity abilities (`:69-70`). `:6374` jumps: **[100, 100, 0, 0]**; no
jump: **[98, 100, 0, 0]**.

**Trace K4: Selfdestruct into a Ghost.** Bonsly [Selfdestruct] against player Gastly:
IMMUNE, **-10** and stop (`:299`). `:6374` jumps: **[90, 0, 0, 0]**.

**Trace K5: Recover.** Lileep [Sludge Bomb, Giga Drain, AncientPower, Recover]. Giga Drain
(Grass, STAB; Grass x2 on Ground, x0.5 on Dragon, `battle_lib.c:2426`, `:2430`) out-damages
Sludge Bomb and AncientPower, which get **-1**; none is QUADRUPLE.
- Lileep at full HP: Recover **-8** (`:513-514`): **[99, 100, 99, 92]**.
- One HP lower (HP% 99): **[99, 100, 99, 100]**.

**Trace K6: Toxic.** Shuckle [Toxic], bench Bonsly. Toxic is never QUADRUPLE here.

| Variant | Player / state | `g` | Script | Scores |
|---|---|---|---|---|
| K6a | Gible | any | to `:547` | **[100, 0, 0, 0]** |
| K6b | Mawile (Steel/Steel) | any | `:521-522` | **[90, 0, 0, 0]** |
| K6c | Scizor (Bug/Steel) | any | `:524-525` | **[90, 0, 0, 0]** |
| K6d | Tentacool (Water/Poison) | any | `:524`, `:526` | **[90, 0, 0, 0]** |
| K6e | Gible paralysed | any | `:545` | **[90, 0, 0, 0]** |
| K6f | Gible, Safeguard on the player's side | any | `:546` | **[90, 0, 0, 0]** |
| K6g | Gible, sun | any | `:534` and `:540` jump | **[100, 0, 0, 0]** |
| K6h | Zangoose [Hyper Cutter, Immunity] | 1 / 0 | `:530` / none | **[90, 0, 0, 0]** / **[100, ...]** |
| K6i | Kadabra [Magic Guard, Inner Focus] | 0 / 1 | `:531` / none | **[90, ...]** / **[100, ...]** |
| K6j | Lickilicky [Poison Heal, Guts] | 0 / 1 | `:532` / none | **[90, ...]** / **[100, ...]** |
| K6k | Grotle [Leaf Guard, Shell Armor], sun | 0 / 1 | `:534-536` / none | **[90, ...]** / **[100, ...]** |
| K6l | Grotle, no weather | 0 | `:536` falls through | **[100, ...]** |
| K6m | Whiscash [Hydration, Swift Swim], rain | 0 / 1 | `:539-542` / none | **[90, ...]** / **[100, ...]** |
| K6n | Whiscash, no weather | 0 | `:542` falls through | **[100, ...]** |

**Trace K7: Light Screen.** Corsola [Recover, Light Screen] at full HP, bench Bonsly. Recover
**-8** always (`:513-514`).
- Light Screen up on Corsola's side: **-8** (`:551`): **[92, 92, 0, 0]**.
- No screen: **[92, 100, 0, 0]**. A screen on the player's side only: **[92, 100, 0, 0]**.

**Trace K8: Rock Polish.** Cranidos (Rock Head) [Head Smash, Zen Headbutt, Crunch, Rock
Polish], bench Bonsly. Head Smash estimates 0 (no-calc): no immunity checks, no -1, no kill
bonus. Crunch out-damages Zen Headbutt (**-1**). Not Simple: `:366` jumps to `_NoSimple`.

| Variant | Speed stage | Trick Room | Scores |
|---|---|---|---|
| K8a | 0 | no | **[100, 99, 100, 100]** |
| K8b | +5 | no | **[100, 99, 100, 100]** |
| K8c | +6 | no | **[100, 99, 100, 90]** (`:371`) |
| K8d | 0 | yes | **[100, 99, 100, 90]** (`:364`) |
| K8e | Cranidos [Rock Polish] with synthetic abilities [Simple, Mold Breaker], +2 | no | **[100, 0, 0, 0]** |
| K8f | the same, +3 | no | **[90, 0, 0, 0]** (`:367`) |

**Trace K9: Curse.** Shuckle (Bug/Rock, Solid Rock) [Toxic, Accelerock, Curse, Sand Tomb],
bench Bonsly. Sand Tomb out-damages Accelerock (**-1**). Not Ghost (`:681-684` fall
through), not Simple (`:689` jumps).

| Variant | Attack, Defense | Scores |
|---|---|---|
| K9a | 0, 0 | **[100, 99, 100, 100]** |
| K9b | +6, 0 | **[100, 99, 90, 100]** (`:696`) |
| K9c | 0, +6 | **[100, 99, 92, 100]** (`:697`) |
| K9d | +6, +6 | **[100, 99, 90, 100]** (`:696` first) |
| K9e | +2, +2 | **[100, 99, 100, 100]** |
| K9f | Shuckle [Curse], synthetic [Simple, Simple]: +2, +2 / +3, 0 / 0, +3 | **[100, ...]** / **[90, ...]** / **[90, ...]** (`:690-691`) |

A Ghost-type Shuckle with Curse is refused (5.1).

**Trace K10: kill bonuses.** The player at 1/1 HP.
- Bonsly [Accelerock]: kills, effect HIT: **+4** (`:6401` -> `:6411`): **[104, 0, 0, 0]**
  (Ruling 8: not +6, though its priority is +1).
- The same with Accelerock given effect PRIORITY_1 (synthetic, refused; scores only):
  `:6400` -> `:6408`, **+6**: **[106, 0, 0, 0]**.
- Bonsly [Selfdestruct, Accelerock]: Selfdestruct's real damage kills, but
  `IfCurrentMoveKills` computes none for it (`trainer_ai.c:1539-1554`). `:6374` jumps:
  **[100, 104, 0, 0]**; no jump: **[98, 104, 0, 0]**.
- Cranidos [Head Smash, Crunch]: **[100, 104, 0, 0]**.

**Trace K11: the quad roll on a status move.** Cranidos [Rock Polish] against player Moltres
(Fire/Flying: Rock x2 on each, `battle_lib.c:2483`, `:2487`): QUADRUPLE. `:6383` does not
jump: **+2**, **[102, 0, 0, 0]**; it jumps: **[100, 0, 0, 0]**.

### Task 6 traces (all four flags)

Speeds are the actives' Speed stats, with no modifier: the player's Gible 20, Bonsly 12,
Lileep 15, Corsola 19, Shuckle 11, Cranidos 28. "Player Speed s" sets the player's stat;
each test asserts the AI's compare (`CompareBattlerSpeed(AI, player, TRUE)`, tie bit 0).
The BASIC and EVAL_ATTACK parts of these scores are the K traces'.

**Trace E1: Explosion.** Bonsly [Selfdestruct, Brick Break], bench Cranidos, two Gibles
(Basic_CheckLastMon ends with no change, `:311` -> `:318`). Brick Break: the only damage
estimate, Expert_BrickBreak without screens (`:4561-4563`): **100** throughout. For
Selfdestruct, `:6374` and `:6530` jump (EVAL's -2 and RISKY's +2 not taken), so its score
moves by Expert_Explosion (5.3) alone:

| Variant | Bonsly HP | Player | Rolls that jump | Path | Selfdestruct |
|---|---|---|---|---|---|
| E1a | 44/44 (100%) | Speed 20 (AI slower) | none | `:1864` -> `:1868` -> `:1879`, -1 | **99** |
| E1b | 100% | Speed 20 | `:1880` | | **100** |
| E1c | 100% | Speed 10 (AI faster) | none | `:1865` -> ScoreMinus3 | **97** |
| E1d | 100% | Speed 10 | `:1865` | | **100** |
| E1e | 100% | Speed 10, Quick Claw activated | none | lone claw: AI SLOWER (`battle_lib.c:1408-1411`), -1 | **99** |
| E1f | 31/44 (70%) | Speed 20 | none | `:1863` -> `:1868` -> -1 | **99** |
| E1g | 22/44 (50%) | Speed 20 | none | `:1870` +1, `:1874` ends | **101** |
| E1h | 50% | Speed 20 | `:1870` | | **100** |
| E1i | 13/44 (29%) | Speed 20 | none | +1 (`:1871`), +1 (`:1876`) | **102** |
| E1j | 29% | Speed 20 | `:1870` | +1 (`:1876`) | **101** |
| E1k | 29% | Speed 20 | `:1870`, `:1875` | | **100** |
| E1l | 100% | evasion +1 (decomp 7) | none | -1 (`:1857`), then as E1a | **98** |
| E1m | 100% | evasion +4 (decomp 10) | none | -1, -1 (`:1860`), then as E1a | **97** |
| E1n | 100% | evasion +4 | `:1859` | -1, then as E1a | **98** |

With no roll jumping, Selfdestruct is 100 - 2 - 1 + 2 = **99** (E1a with EVAL and RISKY);
with `:6530` jumping, **97**: **[99, 100, 0, 0]** and **[97, 100, 0, 0]**.

**Trace E2: Recover.** Lileep [Sludge Bomb, Giga Drain, AncientPower, Recover] (K5). Giga
Drain (Grass, STAB) is not resisted by Gible: `Expert_DrainMove` ends (`:1832`), **100**.
Sludge Bomb **99**. AncientPower 99 + 2 (RISKY, `:6530`) = **101**.

| Variant | Lileep HP | Player | Rolls that jump | Recover | Scores |
|---|---|---|---|---|---|
| E2a | 44/44 | Speed 20 | none | -8 (BASIC), -3 (`:2649` -> `:2659`) | **[99, 100, 101, 89]** |
| E2b | 44/44 | Speed 20 | `:6530` (slot 2) | | **[99, 100, 99, 89]** |
| E2c | 43/44 (97%) | Speed 20 | none | slower, not below 70%: -3 (`:2666`) | **[99, 100, 101, 97]** |
| E2d | 97% | Speed 20 | `:2665` | -> `:2669`, no Snatch effect seen -> +2 (`:2675`) | **[99, 100, 101, 102]** |
| E2e | 97% | Speed 20 | `:2665`, `:2674` | | **[99, 100, 101, 100]** |
| E2f | 22/44 (50%) | Speed 20 | none | below 70% -> `:2669` -> +2 (Recover when slower) | **[99, 100, 101, 102]** |
| E2g | 97% | Speed 10 | none | not slower: -8 (`:2651`) | **[99, 100, 101, 92]** |

**Trace E3: Giga Drain resisted.** Lileep [Giga Drain] against player Zubat (Poison/Flying:
each halves Grass): resisted, -3 (`:1835-1836`): **[97, 0, 0, 0]**; `:1835` jumps:
**[100, 0, 0, 0]**.

**Trace E4: Toxic.** Shuckle [Toxic, Accelerock, Curse, Sand Tomb], bench Bonsly (K6, K9).
Accelerock **99** (-1); Curse **103** (E8a); Sand Tomb **100**.

| Variant | Shuckle HP | Player HP | Rolls that jump | Toxic | Scores |
|---|---|---|---|---|---|
| E4a | 39/39 | 400/400 | none | both above 50%, no SP_DEF_UP/PROTECT move | **[100, 99, 103, 100]** |
| E4b | 19/39 (48%) | 400/400 | none | -3 (`:2690-2691`) | **[97, 99, 103, 100]** |
| E4c | 48% | 400/400 | `:2690` | | **[100, 99, 103, 100]** |
| E4d | 39/39 | 200/400 (50%) | none | -3 (`:2695-2696`) | **[97, 99, 103, 100]** |
| E4e | 48% | 50% | none | -3, -3 | **[94, 99, 103, 100]** |
| E4f | Shuckle [Toxic, Curse]: 48% | 50% | none | no damaging move: `:2688` skips both | **[100, 103, 0, 0]** |
| E4g | Shuckle [Toxic, Protect, Curse, Sand Tomb] (synthetic) | 400/400 | none / `:2704` | PROTECT known (`:2700`): +2 | Toxic **102** / **100** |

**Trace E5: Light Screen.** Corsola [Recover, Light Screen], bench Bonsly; Corsola is slower.

| Variant | Corsola HP | Player's last move | Rolls that jump | Scores |
|---|---|---|---|---|
| E5a | 46/46 | none (physical) | none | **[89, 101]** (Recover -8, -3; Light Screen +1, `:2718-2719`) |
| E5b | 46/46 | none | `:2718` | **[89, 100]** |
| E5c | 46/46 | Flamethrower (special) | none | **[89, 102]** (+1, `:2724-2725`) |
| E5d | 46/46 | Flamethrower | `:2724` | **[89, 101]** |
| E5e | 46/46 | Nature Power (status, 7.4) | none | **[89, 101]** |
| E5f | 40/46 (86%) | Flamethrower | none | **[97, 101]** (below 90%: no `:2718`; Recover -3, `:2666`) |
| E5g | 22/46 (47%) | Flamethrower | none | **[102, 98]** (Light Screen -2, `:2728-2729`; Recover +2) |

**Trace E6: Head Smash and Rock Polish.** Cranidos [Head Smash, Zen Headbutt, Crunch, Rock
Polish], bench Bonsly (K8). Zen Headbutt **99**, Crunch **100**.

| Variant | Player | Cranidos ability | Speed / field | Rolls that jump | Scores |
|---|---|---|---|---|---|
| E6a | Gible | Rock Head | AI faster (28/20) | none | **[100, 99, 100, 97]** (Head Smash resisted: `:6294`; Rock Polish -3, `:2071`) |
| E6b | Rattata (Normal) | Rock Head | faster | none | **[101, 99, 100, 97]** (not resisted, Rock Head +1, `:6297`) |
| E6c | Rattata | Mold Breaker (slot 1) | faster | none | **[100, 99, 100, 97]** |
| E6d | Gible | Rock Head | player Speed 30: slower | none | **[100, 99, 100, 103]** (+3, `:2075-2076`) |
| E6e | Gible | Rock Head | slower | `:2075` | **[100, 99, 100, 100]** |
| E6f | Gible | Rock Head | Trick Room: slower (`battle_lib.c:1432-1439`) | none | **[100, 99, 100, 93]** (BASIC -10, `:364`; +3) |

**Trace E7: Brick Break.** Bonsly [Brick Break, Accelerock], bench Cranidos. Accelerock
**99**. Reflect or Light Screen on the player's side: Brick Break **101** (`:4561-4566`);
none, or Reflect on Bonsly's own side: **100**.

**Trace E8: Curse.** Shuckle as E4a, with Defense stage d (Attack 0); Curse's score:

| Variant | d (decomp) | Path | Curse |
|---|---|---|---|
| E8a | 0 (6) | +1 (`:3482`), +1 (`:3487`), +1 (`:3492`) | **103** |
| E8b | +1 (7) | `:3490` ends after two | **102** |
| E8c | +2 (8) | `:3485` ends after the coin (Curse at +2 Defense) | **101** |
| E8d | +3 (9) | `:3471` tests > 9: as E8c | **101** |
| E8e | +4 (10) | `:3471` ends | **100** |
| E8f | +6 (12) | BASIC -8 (`:697`), `:3471` ends | **92** |
| E8g | 0, Shuckle knowing Trick Room or Gyro Ball (synthetic, slot 1) | +1 (`:3478`), then as E8a | **104** |
| E8h | as E8g, `:3477` jumps | skips the coin: +1, +1 | **102** |

The full scores of E8a-E8f are **[100, 99, Curse, 100]**.

**Trace E9: RISKY.** Against the scores after the first three flags, RISKY adds **+2** to
Selfdestruct (Bonsly [Selfdestruct, Brick Break]) and to AncientPower (Lileep's set) when
`:6530` does not jump, and nothing to any other slot or when it jumps. The same through the
public `ai_scores` (no coverage): Bonsly **[99, 100, 0, 0]** (`:6530` jumping: Selfdestruct
97) and Lileep **[99, 100, 101, 89]** (AncientPower 99).

**Exact against loose.** The exact distribution equals the loose oracle, the slot score
distributions too, on the real `KAIZO` layout (`test_new_paths_exact_equals_loose`, slow): on
K1c, K5, K6h with Shuckle's full moveset, K9b on that state, K6m with the full moveset, K7
with Corsola's full moveset, K8c and the Selfdestruct kill state, and on E1i, E1m, E4e, E5c
with Corsola's full moveset, E2c, E8b, Lileep's set against Zubat, E6b, and Bonsly's full set
with Reflect on the player's side. At equal speed (`test_exact_equals_loose_at_equal_speed`):
E1a with the player at Speed 12 (`:1864`), E6a at 28 (`:2070`) and E2c at 15 (`:2650`), each
with the loose tie read on the slot's path; and E1e (Quick Claw), with no tie read.

## 14. Width and cost, measured (K3c Tasks 3, 6, 7 and 8; E1 batches 1 and 2a)

### E1 batch 2a: reads keyed by order on the path (8.2)

- `ai_width` is 45 + each slot's distinct scoring paths (8.5), checked equal to the loose
  oracle's count on all 144 audit decisions; the bound for any state is 45 + 4 x 2^7 = 557.
- Same state and script as batch 1 (`_scratch/e1/b2a/measure.py`): `_policy` 17448 -> 17755
  jaxpr equations, compile 5.9 -> 5.7 s, one call 0.23 -> 0.18 ms; `_distribution` 19030 ->
  19528 equations, compile 6.6 -> 6.4 s, one call 1.80 -> 0.37 ms (2^7 = 128 scorings instead
  of 2^11 = 2048). Single runs on a shared machine (about +-0.5 s compile).

### E1 batch 1: per-slot guess columns (8.4)

- `ai_width` is now 45 + `sum_slots 2^(sites + guess columns)`: per Roark active, 45 +
  8 x S when the AI guesses the candidate's ability and 45 + S when it does not, with S =
  `sum_slots 2^(sites)` = Bonsly 264, Lileep 26, Gible 8, Corsola 28, Shuckle 52, Cranidos
  10 (`tests/kaizo/test_roark_ai_audit.py SITE_SUMS`). The largest is Bonsly's 2157; the
  bound for any state is 8237 (8.5), against `WIDTH_CAP` 131072. The audit still finds
  exact == loose on all 144 decisions of both pools.
- One Kaizo Roark state (Gible against the six), warm, 20-CPU machine, measured before and
  after (`measure.py` in the batch scratch): `_policy` 16976 -> 17448 jaxpr equations
  (+2.8%), compile 5.2 -> 5.9 s, one call 0.22 -> 0.23 ms; `_distribution` 18531 -> 19030
  equations (+2.7%), compile 5.7 -> 6.6 s, one call 0.58 -> 1.80 ms (2^11 = 2048 scorings
  instead of 2 x 2^8 = 512).

### Final, at K3c close (Task 8)

Measured 2026-11-12 on the final code, warm, a fresh cache per run, on the same 20-CPU
machine (Task 7's `measure7.py`, reusing Task 3's `measure.py` helpers; Task 8 report).

**Widths against both saves' pools** (`tests/kaizo/test_roark_ai_audit.py`: every pool
candidate at cap 16, its own moves and item, leading the save's party, against each of
Roark's six as the active at turn 0; 72 decisions per save, 12 candidates x 6 actives):

| Roark's active | `ai_width` | `ai_flat_width` (not used) | loose reads |
|---|---|---|---|
| Bonsly | 573 | 4423680 | 11-12 |
| Lileep | 97 | 552960 | 7-9 |
| Gible | 61 | 17280 | 6-9 |
| Corsola | 101 | 552960 | 7 |
| Shuckle | 149 | 2211840 | 9-11 |
| Cranidos | 65 | 34560 | 6-9 |

- The values are the same for every candidate and for both saves (`kaizo_save_pre_roark`,
  `kaizo_save_probe`); the test pins them (`WIDTHS`).
- No decision is refused. The maximum is 573; the bound for any state was 2093 (8.5 before E1);
  `WIDTH_CAP` is 131072.
- Every decision has at most 12 loose reads, so all 144 are checked exact against the
  joint loose enumeration (at most 2^18), each slot's score distribution too. That check
  covers the scoring and the move pick; the switch part of the loose oracle reuses the
  exact layout (8.6), and the switch is checked by its own oracle (11.1).

**Cost** (1000 states of the `kaizo_save_pre_roark` pool, HP varied; bucket 1024):

| per state | Kaizo, 1 device | Kaizo, 20 devices | vanilla, 1 device | vanilla, 20 devices |
|---|---|---|---|---|
| `ai_rows` | **215 µs** | **31.2 µs** | 13.9 µs | 13.1 µs |
| vmapped `policy._distribution` | 265 µs | 271 µs | 10.8 µs | 11.2 µs |
| one state (bucket 256) | 0.052 s | 0.015 s | 0.005 s | 0.010 s |

**AI share of a depth-1 expansion** of one Kaizo root (27 pairs; the root's distribution:
Stealth Rock 0.687, Selfdestruct 0.096, Brick Break 0.217):

| | 1 device | 20 devices |
|---|---|---|
| Kaizo | 19.1% (0.051 s of 0.266 s) | 9.6% (0.016 s of 0.168 s) |
| vanilla | 2.5% | 7.7% |

The spec 6 stop condition (above 50%) is not met. These shares are warm. A first solve also
pays the cold compile and export of each `ai_rows` bucket it uses (up to 7 per device count,
about 2.4-5 s each, the tables below; Task 3 review m5); later runs load them from the cache
(`jaxcache`).

The rest of this section is the history, measured as the tasks landed.

### Task 3

Measured 2026-11-11 on one CPU device (20 logical CPUs), warm, with Kaizo Roark's AI still
refused (Tasks 5-7): a refused row costs what an accepted one does, since the program
computes everything and then masks it. Scripts: the Task 3 report.

**Width.**
- `ai_width` (the combinations merged, 8.5) for Kaizo Roark's six against the
  `kaizo_save_pre_roark` pool: 18 to 26. Its bound for any state is 2058; `WIDTH_CAP` is 131072.
- `ai_flat_width` (reported, not used): 240 to 1920.
- Vanilla: `ai_width` 42, flat 3840 (`ENUMERATION_WIDTH`).

**Cost per state.** `policy._distribution`, vmapped over 1000 states, no padding:
- vanilla: 10.2 µs;
- Kaizo: 162 µs. The Kaizo layout runs `_scores` 2 x 2^8 = 512 times per state, vanilla 32.

**`ai_rows` buckets.** `expand.ai_rows` pads each call to a power-of-two bucket of its rows
(256 minimum, `CHUNK` 16384 maximum; `expand.bucket_rows`). Before that every call paid for
16384 rows: 2.39 s Kaizo and 0.24 s vanilla, for one state as for a thousand.

| bucket | vanilla warm | Kaizo warm | cold (compile + export) |
|---|---|---|---|
| 256 | 0.005 s | 0.039 s | not separated (compiled by an earlier step of the run) |
| 512 | 0.008 s | 0.077 s | 3.5 s each |
| 1024 | 0.013 s | 0.176 s | not separated |
| 16384 | 0.248 s | 2.34 s | 3.8 s vanilla, 5.1 s Kaizo |

The padding rows are copies of the call's row 0 and are dropped. A row's result does not
depend on the bucket: `tests/search/test_expand.py test_ai_rows_do_not_depend_on_the_bucket`
and `tests/ai/test_kaizo_outcome.py test_kaizo_ai_rows_do_not_depend_on_the_bucket` compare
the bytes.

**AI share of an expansion.** This is `ai_rows` over (`ai_rows` + `expand`) for one level.
The Kaizo AI is refused, so its legal move actions stand in for the actions with mass.

| | states | pairs | ai_rows | expand | AI share |
|---|---|---|---|---|---|
| Kaizo, depth 1 (one root) | 1 | 36 | 0.040 s | 0.218 s | 15.5% |
| Kaizo, depth 1, all 81 legal pairs | 1 | 81 | 0.039 s | 0.237 s | 14.0% |
| Kaizo, depth 2 (the root's decision children) | 399 | 14244 | 0.079 s | 13.6 s | 0.6% |
| vanilla, depth 1 | 1 | 18 | 0.005 s | 0.193 s | 2.5% |
| vanilla, depth 2 | 146 | 1867 | 0.005 s | 1.12 s | 0.5% |
| vanilla `nuzline solve`, depth 2, 2000 playouts (`stage_b.solve`) | 179 in 2 calls | | 0.012 s | 1.87 s | 0.6% (of a 58 s solve) |

**Review rounds (I2, I2-R1).**
- Kaizo `ai_rows` now maps each device's rows one at a time: `lax.map` inside `shard_map`
  over the call's device axis (`expand._ai`). Under `vmap`, some Kaizo rows were observed
  to differ by 1 ulp between call sizes.
- A row's float32 result is now the same at every call size and device count
  (`tests/ai/test_kaizo_outcome.py`, the bucket test and the device-count test).
- `lax.map` alone, without `shard_map`, ran every row on one device, which was slower at
  the default device count.

Warm `ai_rows` after the change (cold compile and export in brackets):

| bucket | 1 device: vanilla | 1 device: Kaizo | 20 devices (the CLI default here): vanilla | 20 devices: Kaizo |
|---|---|---|---|---|
| 256 | 0.005 s | 0.025 s | 0.011 s | 0.013 s |
| 512 | 0.008 s (3.5 s) | 0.046 s (2.8 s) | 0.012 s (3.3 s) | 0.016 s (2.4 s) |
| 1024 | 0.013 s | 0.096 s | 0.013 s | 0.020 s |
| 16384 | 0.243 s (3.9 s) | 1.49 s (4.2 s) | 0.044 s (3.4 s) | 0.177 s (2.5 s) |

The AI share of a depth-1 expansion (one root):

| | 1 device | 20 devices |
|---|---|---|
| Kaizo | 10% | 8% |
| vanilla | 2.5% | 7.5% |

The spec 6 stop condition (an AI share above 50%) is not met. Before bucketing the Kaizo
depth-1 share was 91% and the vanilla one 56%, both from the 16384-row padding.

**Re-measured after K3c Task 6** (2026-11-11, warm, the same machine; scripts
`measure6.py`, reusing Task 3's `measure.py` helpers; Task 6 report). Kaizo Roark's AI is now
accepted (no refused row), so these are the costs of real distributions.

- `ai_width` with each of Kaizo Roark's six as the active (against his Gible): Bonsly 538,
  Lileep 62, Gible 26, Corsola 66, Shuckle 114, Cranidos 30. The maximum is **538**
  (Selfdestruct's 8 sites: 10 + 2 x (4 + 256 + 2 + 2)), against `WIDTH_CAP` 131072 and the
  bound 2058 of 8.5. The same six values over the `kaizo_save_pre_roark` pool.
- `ai_flat_width` (reported, not used): 3840 to 983040.
- Cost, after the Task 6 review fix (m1: each policy's jump tables trace only its own
  labels, `policy._dispatch`). In the commit before that fix, the vanilla program also traced
  the Kaizo labels, masked off: vmapped vanilla `_distribution` cost 24 µs per state and
  `ai_rows` 29 µs per state on 1 device, with a 4.0% depth-1 share.

| per state, 1000 states | Kaizo, 1 device | Kaizo, 20 devices | vanilla, 1 device | vanilla, 20 devices |
|---|---|---|---|---|
| `ai_rows` (bucket 1024) | **206 µs** | **32.5 µs** | 14.5 µs | 12.2 µs |
| vmapped `policy._distribution` | 269 µs | 272 µs | **10.6 µs** (Task 3: 10.2) | 10.6 µs |
| one state (bucket 256) | 0.050 s | 0.016 s | 0.005 s | 0.010 s |

  20 devices is the CLI default here (`os.cpu_count()`).
- The AI share of a depth-1 expansion of one Kaizo root (27 pairs, the root's real
  distribution: Stealth Rock 0.687, Selfdestruct 0.096, Brick Break 0.217):

| | 1 device | 20 devices |
|---|---|---|
| Kaizo | 19.4% (0.051 s of 0.262 s) | 9.5% (0.016 s of 0.166 s) |
| vanilla | 2.4% | 7.9% |

The spec 6 stop condition (above 50%) is not met.

**Re-measured after K3c Task 7** (2026-11-11, warm, the same machine; script `measure7.py`,
reusing Task 3's `measure.py` helpers; Task 7 report). `KAIZO.switch_sites` is 44, so every
Kaizo `ai_width` gains 35 switch stops (45 instead of 10).

- `ai_width` with each of Kaizo Roark's six as the active (against his Gible): Bonsly 573,
  Lileep 97, Gible 61, Corsola 101, Shuckle 149, Cranidos 65. The maximum is **573**. Over
  every state of the `kaizo_save_pre_roark` pool (12 candidates x 6 actives) the widths are
  the same six values. The bound for any state is 45 + 2 x 4 x 2^8 = **2093** (8.5). Both are
  under `WIDTH_CAP` 131072 and under the 16384-row `ai_rows` bucket maximum.
- `ai_flat_width` (reported, not used): 17280 to 4423680.
- No row refused.

| per state, 1000 states | Kaizo, 1 device | Kaizo, 20 devices | vanilla, 1 device | vanilla, 20 devices |
|---|---|---|---|---|
| `ai_rows` (bucket 1024) | **207 µs** | **33.2 µs** | 13.7 µs | 12.8 µs |
| vmapped `policy._distribution` | 262 µs | 267 µs | 11.1 µs | 10.5 µs |
| one state (bucket 256) | 0.051 s | 0.017 s | 0.005 s | 0.010 s |

The 45 switch stops do not show in the cost: the switch part maps 45 stops through
`switch_action` instead of 10, against 512 scorings.

- The AI share of a depth-1 expansion of one Kaizo root (27 pairs, the same root distribution
  as Task 6: Stealth Rock 0.687, Selfdestruct 0.096, Brick Break 0.217):

| | 1 device | 20 devices |
|---|---|---|
| Kaizo | 20.6% (0.051 s of 0.250 s) | 10.2% (0.016 s of 0.161 s) |
| vanilla | 2.5% | 7.6% |

The spec 6 stop condition (above 50%) is not met.
