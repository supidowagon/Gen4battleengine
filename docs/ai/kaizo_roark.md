# Kaizo Roark's trainer AI, transcribed from the decomp

This is the specification for the Kaizo policy of the trainer AI in `src/nuzline/ai`
(milestone K3c). It is built the way `docs/ai/roark.md` is, and it states only what
differs from that vanilla transcription; every rule not restated here is roark.md's.
Paths are relative to `external/pokeplatinum`. `script.s` means
`src/battle/trainer_ai/script.s` and `trainer_ai.c` means
`src/battle/trainer_ai/trainer_ai.c`. Kaizo data comes from the Kaizo tables
(`data/intermediate_kaizo/*.json`, built into `data/gen4_platinum_kaizo`).

K3c is written task by task: sections 1-2 come from Task 2, the later sections from the
tasks that own them (draws and the outcome layout: Task 3; commands: Task 4; per-flag
paths: Tasks 5-6; switching: Task 7; the rest: Task 8).

## 1. Flags

- `data/intermediate_kaizo/trainers.json`, trainer 246 (Leader Roark): `ai_flags = 23`,
  `0b10111`.
- Line n+2 of `generated/ai_flags.txt` is bit n (roark.md 1), so 23 = BASIC (bit 0) +
  EVAL_ATTACK (bit 1) + EXPERT (bit 2) + RISKY (bit 4, `AI_FLAG_RISKY`, line 6).
- `FlagTable` (`script.s:18-22`) maps bit 4 to `Risky_Main` (`script.s:6521`).
- `TrainerAI_MainSingles` runs the flags in bit order (`trainer_ai.c:297-310`): BASIC,
  EVAL_ATTACK, EXPERT, then RISKY.
- The Kaizo policy runs exactly these four scripts (plan K3c Ruling 10). Until K3c Task 6
  registers RISKY, coverage refuses the flag (`engine/coverage.py`
  `IMPLEMENTED_AI_FLAGS`), so no Kaizo Roark fight is graded yet.

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
  EVAL_ATTACK paths; Task 6 adds their EXPERT and RISKY paths. Until then coverage refuses
  Kaizo Roark by his RISKY flag (section 1), and the AI itself refuses every move whose
  effect goes to an Expert label not transcribed yet (section 5).
- The script-roll sites are keyed by battle effect too (section 8, K3c Task 3), so a move is
  transcribed by its effect alone. Example: Accelerock (205, effect HIT) takes Rock Throw's
  paths and its EvalAttack quad roll (`script.s:6383`).
- RISKY (`script.s:6521-6531`) scores only the effects of `Risky_RiskyEffects`
  (`script.s:6536-6561`). Of Roark's effects it lists Selfdestruct (HALVE_DEFENSE, 7,
  `:6538`) and AncientPower (RAISE_ALL_STATS_HIT, 140, `:6550`); Task 6 transcribes it.

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

**2.4 Refusals that only Kaizo data reaches.** Both policies refuse an opponent move that
the AI's damage estimate or move type does not take from the move table:
- an effect in `sAltPowerMoveEffects` (`trainer_ai.c:48-61`; `commands.ALT_POWER_EFFECTS`).
  Such a move is damage-estimated whatever its power (`:1035`, `:1553`, `:2827`). Kaizo
  Triple Axel (149) and Fury Cutter (210) carry RANDOM_DAMAGE_1_TO_150_LEVEL (88);
- a move id that `TrainerAI_CalcDamage` takes by case (`trainer_ai.c:2885-3056`;
  `commands.CALC_DAMAGE_MOVES`). The case applies whatever the move's effect. Kaizo
  Triple Axel sits on MOVE_PSYWAVE's id (`:3014`, which also draws `RandNext`). Kaizo
  Return (216) and Frustration (218) have effect 0 but still take their friendship
  cases (`:3018-3026`);
- a move id that `TrainerAI_MoveType` takes by case (`trainer_ai.c:3132-3222`;
  `commands.MOVE_TYPE_MOVES`): Natural Gift, Judgment, Hidden Power, Weather Ball.

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
  - the site table's ids;
  - Dry Skin 87.
- A Kaizo-renamed slot on one of these lists fails `load_game`.
- The id cases of 2.4 are compared by id whatever a game names the slot. They are checked
  against vanilla names only (Kaizo's 149 is Triple Axel, and it still takes Psywave's
  case).

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

All of these effects are in `KAIZO.transcribed` from K3c Task 5. The paths below are the
BASIC and EVAL_ATTACK ones (Task 5); the EXPERT entries of RECOVER_HALF_DAMAGE_DEALT,
HALVE_DEFENSE, RESTORE_HALF_HP, STATUS_BADLY_POISON, SET_LIGHT_SCREEN, SPEED_UP_2, CURSE,
REMOVE_SCREENS and RECOIL_HALF (`script.s:1629-1806`) and RISKY are Task 6 (sections 5.3-5.4).

**The Expert guard (coordinator ruling, Task 5).** A missing Expert path is refused, never
scored. `outcome.KAIZO_EXPERT_LABELS` maps each of those nine effects to its label and the
line of its entry in `Expert_Main`'s dispatch table (Giga Drain `Expert_DrainMove` `:1629`,
Selfdestruct `Expert_Explosion` `:1630`, Recover `Expert_Recovery` `:1652`, Toxic
`Expert_ToxicLeechSeed` `:1653`, Light Screen `Expert_LightScreen` `:1654`, Rock Polish
`Expert_StatusSpeedUp` `:1665`, Curse `Expert_Curse` `:1700`, Brick Break `Expert_BrickBreak`
`:1739`, Head Smash `Expert_RecoilMove` `:1806`). The Kaizo policy always runs EXPERT (Ruling
10), so it refuses an opponent move with one of these effects unless the effect is in
`KAIZO.expert_transcribed` (`policy._refused`), naming the label. That set is empty after
Task 5; Task 6 fills it. So no Kaizo trainer, `0b111` or not, is graded with a missing
Expert path (`test_expert_guard_refuses_a_0b111_kaizo_trainer`).
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
  jumps to `Basic_CheckCurse_GhostType` (`:700`), which is not transcribed: the AI refuses a
  Ghost-type Curse user (`policy._refused`). Kaizo Roark's Curse user, Shuckle, is Bug/Rock.
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
   (`:6383`), +2 when the draw is not below 80 (176/256). Every move reaches it, status
   moves too: Rock Polish (Rock) into a Fire/Flying `D` is QUADRUPLE.
5. `EvalAttack_ApplyKillBonuses` (`:6387-6415`), for a move that kills:
   - HALVE_DEFENSE ends at `EvalAttack_Terminate` (`:6389`). It is unreachable for
     Selfdestruct, which never kills (1).
   - The Focus Punch, Sucker Punch and Future Sight effects (`:6394-6396`) are no Roark move's.
   - `BATTLE_EFFECT_PRIORITY_1` (`:6400`) gets **+2** then **+4**. It keys on the effect, not
     the move's priority (the decomp's comment, `:6398-6399`). Kaizo Accelerock has priority
     +1 but effect HIT, so its kill bonus is **+4** (plan K3c Ruling 8).
   - Anything else: **+4** (`:6401` -> `:6411`), with no draw.

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
  (both Quick Claws `:1402-1407`, both Lagging Tails `:1412-1417`, Trick Room `:1432-1439`,
  plain `:1440-1447`) gives TIE when `RandNext() & 1` is 1, else FASTER, never SLOWER. Every
  compare on Roark's paths tests SLOWER (`script.s:1864`, `:2070`, `:2650`). So the exact
  command passes `tie_roll = 0` (`battler_slower(..., 1, 0, qc_roll)`), has no `AiOutcome`
  field, and asserts at trace time that its value is SLOWER. The decomp still consumes the
  draw (8.7).
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
- `ability_guess`: one scalar for the whole decision (8.4).
- `script_rolls: int8[MOVES, KAIZO.slot_sites]`. Column k of slot m is the k-th
  `IfRandomLessThan` site of the slot's move, in script order (flag order BASIC,
  EVAL_ATTACK, EXPERT, then RISKY, `trainer_ai.c:297-310`; then line order). 1 means the
  jump is taken.
- `tiebreak`: `RandNext % 12`, as roark.md 8.4.
- `switch_stop`: as roark.md 8.5, over `KAIZO.switch_sites` (K3c Task 7 sets them for the
  six-Pokemon party).

**8.2 The site table, keyed by battle effect.** A move reaches a site through
`IfCurrentMoveEffectEqualTo` (`trainer_ai.c:1441`), so the table is keyed by the move's
effect (`outcome.EFFECT_SITE_LINES`, carried as `KAIZO.effect_sites`), not its id. Each
site is named by its `script.s` line, and `outcome.SITE_N` gives its N:

| Effect | Sites (line: N) |
|---|---|
| HIT (0) | `:6383`: 80 |
| HALVE_DEFENSE (7) | `:6374`: 51, `:6383`: 80 (Task 6 appends Expert_Explosion's and Risky's) |
| DEF_DOWN (19) | `:6383`: 80, `:2328`: 50 |
| FLINCH_HIT (31) | `:6383`: 80 |
| DEF_DOWN_2 (59) | `:6383`: 80, `:2328`: 50 |
| HIT_BEFORE_SWITCH (128) | `:6383`: 80, `:3726`: 128, `:3731`: 128 |
| STEALTH_ROCK (266) | `:6383`: 80, `:6274`: 128 |
| the other effects of section 5 (2, 3, 4, 6, 32, 33, 35, 52, 69, 72, 76, 109, 140, 186, 261, 269, 275) | `:6383`: 80 (Task 6 appends the Expert and Risky sites of those with entries) |

- `:6383` is `EvalAttack_TryScorePlus2`. It is reached by any move whose
  `IfMoveEffectivenessEquals` is QUADRUPLE (`script.s:6378`), whatever its effect, so every
  effect has it. The vanilla table left it out of the Normal-type moves, but a table keyed
  by effect cannot know the type. An extra site that is never reached only costs width.
- The Expert sites follow `Expert_Main`'s dispatch (`script.s:1623-1811`): DEF_DOWN `:1642`
  and DEF_DOWN_2 `:1671` go to `Expert_StatusDefenseDown` (`:2318`, roll `:2328`);
  HIT_BEFORE_SWITCH `:1706` to `Expert_Pursuit` (`:3706`, rolls `:3726`, `:3731`);
  STEALTH_ROCK `:1805` to `Expert_StealthRock` (`:6268`, roll `:6274`). HIT and FLINCH_HIT
  have no entry and end at `:1811`.
- `KAIZO.transcribed` is exactly the table's effects, so the effect check that refuses an
  untranscribed move also guarantees the move has its row.
- `KAIZO.slot_sites` is 8, the most sites one Kaizo Roark move reaches: Selfdestruct
  (HALVE_DEFENSE), with EvalAttack `:6374` and `:6383`, Expert_Explosion `:1859`, `:1865`,
  `:1870`, `:1875` and `:1880`, and Risky `:6530`. Tasks 5 and 6 add Roark's effects to the
  table.

**8.3 The exact distribution, factored per slot** (`policy._slot_move_distribution`).
- `TrainerAI_EvalMoves` scores the move slots one by one (`trainer_ai.c:483-525`).
  Slot m's score reads only its own script rolls and the guess. `FlagMoveDamageScore`
  compares the damage estimates of all four moves, but those are deterministic
  (`trainer_ai.c:1012-1062`), not draws.
- So, for each guess: `_scores` runs once per local bit pattern p (2^8 = 256 patterns,
  the same p in every slot's columns), and slot m's score with its bits p is read off.
  Each slot's histogram over score values 0-255 weights p by its probability in that slot,
  `prod_k` of N_k/256 where bit k is 1, else 1 - N_k/256.
- The pick (`trainer_ai.c:315-338`): slot m with value v is picked when every other
  candidate ties at v or is below it. For each subset T of the other candidates that ties,
  m's share is `P(RandNext % (|T| + 1) == m's place in the tied list)`.
- The two guesses are averaged. The switch comes first and the item next, as
  `_distribution` does (roark.md 2). Struggle and the locked slot are unchanged.

**8.4 The one-scalar guess (plan K3c Ruling 6).** The decomp draws a fresh coin at every
`LoadBattlerAbility DEFENDER` (`trainer_ai.c:1191-1198`). One scalar is exact when at most
one such read in a decision can change a score. On the transcribed paths the reads are
`script.s:71` (`Basic_CheckForImmunity`), `:117` (`Basic_CheckSoundproof`), `:470`
(`Basic_CheckClearBodyEffect`), and from Task 5 `:304` (`Basic_CheckCannotExplode`, Damp) and
`:529` and `:539` (`Basic_CheckCannotPoison`: Immunity, Magic Guard, Poison Heal, Leaf Guard in
sun; Hydration in rain). The two Toxic reads are two coin flips in the decomp and one guess
here. Both can change Toxic's score only for a species with Hydration and one of the `:529`
abilities, and no Kaizo species has that pair (`species.json`;
`tests/ai/test_kaizo_policy.py test_no_species_splits_the_toxic_reads`, which checks it in
any weather, where the data test below uses the state's). `tests/ai/test_kaizo_outcome.py
test_one_scalar_guess_is_exact_for_roark` checks this as data: for every species with two
abilities, against each of Roark's Pokemon (Kaizo's six, vanilla's three), the loose
oracle (8.6) finds at most one read whose coin changes a score. It re-runs with every
read a later task adds.
- This holds for Roark's movesets, not for every moveset. A Pokemon with both Leer and
  Screech against Tentacool (Kaizo abilities 44 / Clear Body) has two `:470` reads that
  each change a score; the scalar moves them together and the decomp does not. Such a
  moveset is not on Roark's team. A later trainer's must pass the same data test.

**8.5 Width** (plan K3c Ruling 12). `policy.ai_width` counts the combinations the factored
distribution merges: the switch stops plus, per guess, `sum_slots 2^(sites of the slot)`.
For Kaizo Roark's six it is at most 10 + 2 x 4 x 2^8 = 2058 for any state (checked at
import, `policy._check_width`), against `WIDTH_CAP` 131072. `policy.ai_flat_width` reports
the flat product, `2 x 2^(sum of sites) x 12 x 10`, which is not used. Section 14 has the
measured numbers.

**8.6 The loose oracle** (tests only; `policy.ai_action_distribution_loose`). Every
`RandNext` read of the scoring is its own independent draw, as in the decomp: each
`IfRandomLessThan` with its own N (the literal at its call), each `LoadBattlerAbility
DEFENDER` its own coin, the tie-break `RandNext % n` over a `u16`. It reads neither the site
table nor the scalar guess. It finds every read on every path by enumerating the reads
found so far until no new one appears, then enumerates the slots jointly when they have at
most 18 reads in all, and per slot otherwise (exact in loose, where no two slots share a
draw). The gates of the exact layout are each checked against it, with a test that fails
when the gate is removed (`tests/ai/test_kaizo_outcome.py` docstring):
- per-slot factoring and the tie subsets, against a joint enumeration of every slot's bits
  and the 12 tiebreaks;
- the path-only sites (a move has only its effect's sites);
- the one-scalar guess (8.4).

The fourth gate is the dropped speed-tie draw (plan K3c Ruling 3; 7.1). At equal speed
`BattleSystem_CompareBattlerSpeed` returns FASTER or TIE, never SLOWER
(`battle_lib.c:1401-1447`), and every compare on Roark's paths tests SLOWER.
`IfSpeedCompareEqualTo` gives the tie its own loose bit (`ReadKind.SPEED_TIE`), and
`tests/ai/test_kaizo_commands.py test_speed_tie_never_slower` checks the gate in every
branch. Once Tasks 5-6 put the compare on Roark's paths, `test_exact_equals_loose` covers it
too.

**8.7 RNG read order.** As roark.md 8.7, with RISKY after EXPERT (slots 0-3: its roll as
reached), and the speed compare's tie draw where an `IfSpeedCompareEqualTo` meets equal
speeds (no `AiOutcome` field, Ruling 3; K3d's trace replay needs the count).

## 13. Hand-traced decisions (K3c Task 5 fixtures)

`tests/ai/test_kaizo_policy.py`. Common setup:
- Kaizo Roark's party as `build_trainer_party` gives it, the named Pokemon active with the
  moves listed and the bench listed (default: Cranidos). The player's party is one or two
  copies of Roark's own Gible (Dragon/Ground, Sand Veil / Rough Skin, neither of which any
  label tests), at 400/400 HP unless stated: no transcribed move kills it. A "player X" is
  that species put on the player's active slot (its types and ability slot; stats unchanged).
- The scores are those after BASIC and EVAL_ATTACK, from 100 (`TrainerAI_Init`,
  `trainer_ai.c:223`), with refusals not applied; an empty slot scores 0
  (`trainer_ai.c:499-501`). EXPERT and RISKY are Task 6.
- Rolls are named by line: "`:6374` jumps" sets that site's bit; every other bit is 0 (the
  jump is not taken). `g` is the ability guess (0 or 1).
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

With every Expert label counted as transcribed (`KAIZO_ALL_EXPERT`; both sides share the
script transcription, so the gates are unaffected), the exact distribution equals the loose oracle on K1c, K5, K6h with Shuckle's full moveset,
K9b on that state, K6m with the full moveset, K7 with Corsola's full moveset, K8c and the
Selfdestruct kill state (`test_new_paths_exact_equals_loose`, slow).

## 14. Width and cost, measured (K3c Task 3)

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
