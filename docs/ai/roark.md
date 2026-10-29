# Roark's trainer AI, transcribed from the decomp

This is the specification for Roark's AI in `src/nuzline/ai` (plan 1b, Tasks 2, 2b
and 3). It covers what the decomp does when Roark (`TRAINER_LEADER_ROARK`, id 246)
picks an action in his gym fight. Paths are relative to `external/pokeplatinum`, at the
commit pinned in `docs/decomp_paths.md`. `script.s` means
`src/battle/trainer_ai/script.s` and `trainer_ai.c` means
`src/battle/trainer_ai/trainer_ai.c`. The script commands, with their JAX names,
are in `docs/ai/script_commands.md`.

The AI never uses heuristics of its own. Every rule below is the decomp's. Where the
decomp has a bug, the bug is the rule, and it is marked **(quirk)**.

## 1. Flags

- `res/trainers/data/leader_roark.json`: `ai_flags` = `AI_FLAG_BASIC`,
  `AI_FLAG_EVAL_ATTACK`, `AI_FLAG_EXPERT`. In `data/intermediate/trainers.json` this
  is `ai_flags = 7`.
- The flags form a bitmask. Line n+2 of `generated/ai_flags.txt` is bit n, so
  BASIC = bit 0 (1), EVAL_ATTACK = bit 1 (2), EXPERT = bit 2 (4), and 1 + 2 + 4 = 7.
- `FlagTable` (`script.s:18-50`) maps bit n to its entry label:
  bit 0 -> `Basic_Main` (`script.s:52`), bit 1 -> `EvalAttack_Main` (`:6353`),
  bit 2 -> `Expert_Main` (`:1623`).
- `TrainerAI_MainSingles` (`trainer_ai.c:286`) runs the flags in bit order: BASIC,
  then EVAL_ATTACK, then EXPERT (`:297-310`). Each flag scores every move slot,
  0 to 3, before the next flag starts (`TrainerAI_EvalMoves`, `:483-525`). The
  order matters only for the order of RNG reads (section 8). The scores just add up.
- Roark's fight is a single battle (`double_battle: false`), so
  `AI_FLAG_TAG_STRATEGY` is not forced in (`trainer_ai.c:252-255`), and
  `IfTargetIsPartner` never jumps.

## 2. One decision, start to finish

At each turn's command selection, the AI runs these steps in this order. It never
sees the player's choice for the current turn.

1. `Task_TrainerSetCommandSelection` (`src/battle/battle_display.c:3408`) calls
   `TrainerAI_PickCommand` (`trainer_ai.c:3989`).
   1. `TrainerAI_ShouldSwitch` (`:3894`). If it returns TRUE, the AI switches
      (section 11) and the turn's decision ends.
   2. Otherwise `TrainerAI_ShouldUseItem` (`:4056`). If it returns TRUE, the AI uses a
      bag item (section 12) and the turn's decision ends.
   3. Otherwise the command is FIGHT.
2. For FIGHT, `Task_TrainerShowMoveSelectMenu` (`battle_display.c:3579`) calls
   `TrainerAI_Main` (`trainer_ai.c:258`):
   1. `AI_CONTEXT.defender` = the player's active battler. In singles,
      `BattleSystem_RandomOpponent` is `attacker ^ 1`, with no RNG
      (`src/battle/battle_lib.c:4193-4195`).
   2. `TrainerAI_Init` (`trainer_ai.c:211`) sets the base scores and draws the four
      simulated damage rolls (section 3).
   3. `TrainerAI_MainSingles` records the player's last move (section 9), runs the
      three flag scripts, and picks the highest score. Ties are broken at random
      (section 10).

The result, in plan 1a's opponent action encoding: a move slot `0..3`; a switch
`4 + party slot`; or (Task 2b) a bag item `10 + k`.

## 3. Move scores

- **Base score.** `TrainerAI_Init` with `AI_INIT_SCORE_ALL_MOVES` sets all four
  `moveScore` to **100** (`trainer_ai.c:223-230`).
- **Invalid moves start at 0.** Any slot that `BattleSystem_CheckInvalidMoves(...,
  CHECK_INVALID_ALL)` flags starts at 0 (`:234-238`). That covers an empty slot, no
  PP, Disable, Torment, Taunt on a power-0 move, Imprison, Gravity, Heal Block,
  Encore and a Choice lock.
- **Empty and 0-PP slots skip the scripts.** For these slots `TrainerAI_EvalMoves`
  sets the move to `MOVE_NONE`, writes score 0 and runs no script command (`:486-503`).
  Every other slot runs the scripts, even one already at 0. For example, a Taunted
  Stealth Rock starts at 0 and can still gain +1.
- **Score type and clamp.** `moveScore` is `s8` (`include/battle/ai_context.h:14`).
  `AddToMoveScore` adds, then raises any negative result to 0 (`trainer_ai.c:576-586`).
  It has no upper clamp. Roark's largest possible score is 106 (Pursuit: +4 for a KO,
  then +1 and +1 in EXPERT), far below 127, so int32 arithmetic in JAX gives the same
  values.
- **Damage rolls.** `TrainerAI_Init` draws `moveDamageRolls[i] = 100 -
  (RandNext % 16)` for all four slots, empty ones included (`:240`). Each roll is in
  85..100. None of Roark's three scripts reads these rolls (section 8.1).

## 4. Roark's Pokemon and the move effects the scripts see

| Party slot | Pokemon | Ability (slot 0) | Moves (slot order) |
|---|---|---|---|
| 0 | Geodude L12 | Rock Head | Stealth Rock, Rock Throw, -, - |
| 1 | Onix L12 | Rock Head | Stealth Rock, Rock Throw, Screech, - |
| 2 | Cranidos L14 | Mold Breaker | Headbutt, Pursuit, Leer, - |

None of them holds an item. The bag holds `ITEM_POTION` x 2 (decomp ids [17, 17]).

| Move | Effect (`res/moves/*/data.json`) | Type | Power |
|---|---|---|---|
| Stealth Rock | `BATTLE_EFFECT_STEALTH_ROCK` | Rock | 0 |
| Rock Throw | `BATTLE_EFFECT_HIT` | Rock | 50 |
| Screech | `BATTLE_EFFECT_DEF_DOWN_2` | Normal | 0 |
| Headbutt | `BATTLE_EFFECT_FLINCH_HIT` | Normal | 70 |
| Pursuit | `BATTLE_EFFECT_HIT_BEFORE_SWITCH` | Dark | 40 |
| Leer | `BATTLE_EFFECT_DEF_DOWN` | Normal | 0 |

None of these effects is in `sNoDamageCalcMoveEffects` or `sAltPowerMoveEffects`
(`trainer_ai.c:30-62`). So a move is "damaging" for the AI exactly when its power is
greater than 1.

## 5. What each flag's script does with Roark's moves

The path for each of Roark's six moves, step by step. `D` is the player's active
Pokemon. `A` is Roark's active Pokemon. A decomp stat stage equals the engine stage
plus 6 (decomp 6 = neutral = engine 0). `HP%` is `curHP * 100 / maxHP`, integer
division (`trainer_ai.c:596`).

### 5.1 BASIC (`Basic_Main`, `script.s:52-1565`)

The opening steps are the same for every move:

1. `IfTargetIsPartner Terminate`: never jumps.
2. `IfMoveEqualTo MOVE_FISSURE` / `MOVE_HORN_DRILL`: never jumps for Roark.
3. `FlagMoveDamageScore USE_MAX_DAMAGE`, then `IfLoadedEqualTo AI_NO_COMPARISON_MADE,
   Basic_CheckSoundproof`. The result is NO_COMPARISON (0) for the power-0 moves
   (Stealth Rock, Screech, Leer), which jump straight to step 5. Damaging moves get
   HIGHEST (2) or NOT_HIGHEST (1) and fall through to step 4.
4. `Basic_CheckForImmunity` (`:65`), damaging moves only:
   - `IfMoveEffectivenessEquals TYPE_MULTI_IMMUNE, ScoreMinus10`: **-10** and stop
     if `A`'s move cannot affect `D` (section 7.2). For Roark this means Headbutt
     against a Ghost type, or any move against a true Wonder Guard that it does not
     hit super effectively.
   - `LoadBattlerAbility ATTACKER`. `A`'s own ability needs no RNG. If it is Mold
     Breaker (Cranidos), go to `Basic_NoImmunityAbility`.
   - Otherwise `LoadBattlerAbility DEFENDER` resolves `D`'s ability, possibly with
     an ability guess (section 9.2). Each match jumps to a check:
     - Volt Absorb or Motor Drive: `LoadTypeFrom LOAD_MOVE_TYPE` +
       `IfTempEqualTo TYPE_ELECTRIC` (-12).
     - Water Absorb: the same check against `TYPE_WATER` (-12).
     - Flash Fire: the same check against `TYPE_FIRE` (-12).
     - Wonder Guard: `Basic_CheckWonderGuard` (`:96`). DOUBLE or QUADRUPLE
       effectiveness passes, anything else is -12.
     - Levitate: the same check against `TYPE_GROUND` (-12).
     - **(quirk)** A second Levitate test (`:78`) should be Dry Skin. It can never
       match.

     Roark's damaging moves are Rock, Normal or Dark type, so no absorb check can
     fire. Wonder Guard is decided before any guess, because only Shedinja has it and
     Shedinja has one ability. So this read never changes Roark's scores.
   - `Basic_NoImmunityAbility` (`:111`): `FlagMoveDamageScore` again, then
     `IfLoadedEqualTo NO_COMPARISON`. Both branches lead to step 5.
5. `Basic_CheckSoundproof` (`:115`): `LoadBattlerAbility DEFENDER` (a possible ability
   guess).
   - If the result is not Soundproof, go to step 6.
   - If it is Soundproof: `LoadBattlerAbility ATTACKER`, and Mold Breaker goes to
     step 6. Otherwise the `IfMoveEqualTo` list (`:121-131`) gives a sound move
     **-10** and stops. Screech is the only sound move Roark has. Other moves fall
     through into step 6.
6. `Basic_ScoreMoveEffect` (`:133`) jumps on the move's effect:
   - `BATTLE_EFFECT_DEF_DOWN` (Leer, `:145`) and `DEF_DOWN_2` (Screech, `:172`) go
     to `Basic_CheckLowStatStage_Defense` (`:434`):
     - `IfStatStageEqualTo DEFENDER, DEFENSE, 0`: **-10** if `D`'s Defense is at
       decomp stage 0 (engine -6).
     - Otherwise `Basic_CheckClearBodyEffect` (`:469`): `LoadBattlerAbility
       DEFENDER` (a possible ability guess), and Clear Body or White Smoke gives
       **-10**.
     - Mold Breaker is not consulted here, so Cranidos's Leer also gets the -10.
   - `BATTLE_EFFECT_STEALTH_ROCK` (`:284`) goes to `Basic_CheckStealthRock`
     (`:1540`):
     - **-10** if the player's side already has Stealth Rock (`IfSideCondition`).
     - Otherwise `CountAlivePartyBattlers DEFENDER`: **-10** if the player has no
       living Pokemon besides the active one.
   - `HIT`, `FLINCH_HIT` and `HIT_BEFORE_SWITCH` match nothing. They reach
     `PopOrEnd` (`:286`) with no change.

### 5.2 EVAL_ATTACK (`EvalAttack_Main`, `script.s:6353-6415`)

1. `IfTargetIsPartner`: never jumps.
2. `IfCurrentMoveKills USE_MAX_DAMAGE, EvalAttack_ApplyKillBonuses`. This jumps when
   the move is damaging and `D.curHP <= ai_damage(move)` (section 7.1).
   - If it jumps, `EvalAttack_ApplyKillBonuses` (`:6387`) checks the move's effect:
     - Explosion-type effects: no bonus.
     - Focus Punch, Sucker Punch and Future Sight effects: +4 at 86/256
       (the 170 roll).
     - `BATTLE_EFFECT_PRIORITY_1`: +2 and then +4.
     - Anything else: **+4**. All of Roark's damaging moves take this branch, with
       no RNG. Then stop.
3. `FlagMoveDamageScore USE_MAX_DAMAGE`. If the result is NOT_HIGHEST, the move gets
   **-1** (`ScoreMinus1`) and stops. NOT_HIGHEST means some slot of `A` has a
   strictly greater `ai_damage` (section 7.1). Every slot is compared, including
   0-PP and status slots, which estimate 0. Power-0 moves get NO_COMPARISON and are
   never -1.
4. Explosion, Focus Punch and Sucker Punch effects have their own path (`:6364-6366`).
   None of them is Roark's.
5. `EvalAttack_CheckQuadEffective` (`:6377`): `IfMoveEffectivenessEquals
   TYPE_MULTI_QUADRUPLE_DAMAGE`. On QUADRUPLE, `EvalAttack_TryScorePlus2` runs
   `IfRandomLessThan 80` (`:6383`). If the draw is below 80 nothing happens.
   Otherwise the move gets **+2** (176/256).
   - This applies to status moves too. Stealth Rock against a target 4x weak to Rock
     can gain +2 (section 7.2).
   - Normal-type moves (Headbutt, Screech, Leer) are never QUADRUPLE.

### 5.3 EXPERT (`Expert_Main`, `script.s:1623-1811`)

1. `IfTargetIsPartner`: never jumps.
2. The jump table on the move's effect. `HIT` (Rock Throw) and `FLINCH_HIT` (Headbutt)
   have no entry and reach `PopOrEnd` (`:1811`) with no change.
3. `BATTLE_EFFECT_STEALTH_ROCK` goes to `Expert_StealthRock` (`:6268`):
   - `IfRandomLessThan 128` (`:6274`). If the draw is below 128, stop. Otherwise
     **+1**.
   - `IfMoveKnown ATTACKER, MOVE_ROAR` / `MOVE_WHIRLWIND`. Roark's Pokemon know
     neither, so the extra 64-roll (`:6281`) is never reached.
4. `DEF_DOWN` (Leer) and `DEF_DOWN_2` (Screech) go to `Expert_StatusDefenseDown`
   (`:2318`):
   - If `IfHPPercentLessThan ATTACKER, 70` (A's HP% < 70), go to the roll.
   - Otherwise, `IfStatStageGreaterThan DEFENDER, DEFENSE, 3` (decomp stage > 3,
     engine > -3) skips the roll and goes to the target-HP check.
   - Anything else falls into the roll.
   - The roll (`Expert_StatusDefenseDown_TryScoreMinus2`): `IfRandomLessThan 50`
     (`:2328`). If the draw is below 50 nothing happens. Otherwise **-2**
     (206/256).
   - The target-HP check (`Expert_StatusDefenseDown_CheckTargetHP`):
     `IfHPPercentGreaterThan DEFENDER, 70` ends. Otherwise **-2** (D's HP% <= 70).
5. `HIT_BEFORE_SWITCH` (Pursuit) goes to `Expert_Pursuit` (`:3706`):
   - `LoadIsFirstTurnInBattle ATTACKER`. If the result is TRUE, go to
     `Expert_Pursuit_TryScorePlus1`.
   - Otherwise `LoadTypeFrom LOAD_DEFENDER_TYPE_1`, then `LOAD_DEFENDER_TYPE_2`.
     Ghost or Psychic (D's true types) also goes to `TryScorePlus1`.
   - Otherwise skip to `CheckUturn`.
   - `TryScorePlus1`: `IfRandomLessThan 128` (`:3726`). If the draw is below 128 go
     to `CheckUturn`. Otherwise **+1**.
   - `CheckUturn`: `IfMoveNotKnown DEFENDER, MOVE_U_TURN` ends. This is the AI's
     seen-move list (section 9.1). If it has seen U-turn, `IfRandomLessThan 128`
     (`:3731`), then **+1** when the draw is not below 128.
   - `LoadIsFirstTurnInBattle` (`trainer_ai.c:2480`) is TRUE when
     `A.fakeOutTurnNumber >= totalTurns`.
     - A switch-in sets `fakeOutTurnNumber = totalTurns + 1`
       (`battle_lib.c:2117`). That includes a faint replacement, which runs in
       `TurnEnd` before `totalTurns++` (`battle_controller_player.c:1844-1852`).
     - The leads start at 0: `BattleContext` is zeroed at
       `battle_controller_player.c:180-182`, and the battle-start send-out does not
       call `UpdateAfterSwitch`.
     - The result is TRUE on exactly the first decision after `A` entered: turn 1
       for a lead, and the next turn for a mon that entered mid-battle.
     - The engine state carries it as `BattleState.fake_out_turn` (Task 3a), so the
       check is `fake_out_turn[1] >= turn`.

### 5.4 Ordered list of script commands each flag uses (for Roark's moves)

In order of first execution along Roark's paths:

- BASIC: `IfTargetIsPartner`, `IfMoveEqualTo`, `FlagMoveDamageScore`,
  `IfLoadedEqualTo`, `IfMoveEffectivenessEquals`, `LoadBattlerAbility`,
  `LoadTypeFrom`, `IfTempEqualTo`, `GoTo`, `IfLoadedNotEqualTo`,
  `IfCurrentMoveEffectEqualTo`, `IfStatStageEqualTo`, `IfSideCondition`,
  `CountAlivePartyBattlers`, `AddToMoveScore`, `PopOrEnd`.
- EVAL_ATTACK: `IfTargetIsPartner`, `IfCurrentMoveKills`,
  `IfCurrentMoveEffectEqualTo`, `GoTo`, `AddToMoveScore`, `PopOrEnd`,
  `FlagMoveDamageScore`, `IfLoadedEqualTo`, `IfMoveEffectivenessEquals`,
  `IfRandomLessThan`.
- EXPERT: `IfTargetIsPartner`, `IfCurrentMoveEffectEqualTo`, `PopOrEnd`,
  `IfRandomLessThan`, `AddToMoveScore`, `IfMoveKnown`, `GoTo`, `IfHPPercentLessThan`,
  `IfStatStageGreaterThan`, `IfHPPercentGreaterThan`, `LoadIsFirstTurnInBattle`,
  `IfLoadedNotEqualTo`, `LoadTypeFrom`, `IfLoadedEqualTo`, `IfMoveNotKnown`.

The union is 24 commands, one row each in `docs/ai/script_commands.md`.

**How the list was checked.** I parsed `script.s` and walked it from the three entry
labels. The walk follows a conditional jump only where it can happen for Roark:
`IfCurrentMoveEffectEqualTo` jumps only for Roark's six effects, `IfMoveEqualTo` only
for his six moves, `IfMoveKnown ATTACKER` only for moves Roark's Pokemon know, and
`IfTargetIsPartner` never. Every other branch counts both ways. The walk reaches
exactly the labels named in 5.1-5.3 (plus `ScoreMinus1/10/12`) and exactly these 24
commands.

## 6. Where the AI reads the true battle state

These reads use the real state, not the AI's knowledge:

- the damage estimate and the kill check: `D`'s true stats, stages, HP, types,
  ability and held item;
- type effectiveness: true types, true Levitate and Wonder Guard;
- `IfStatStage*`, `IfHPPercent*`, `IfSideCondition`, `CountAlivePartyBattlers` and
  `LoadTypeFrom DEFENDER_TYPE_*`.

Only the player's moves (section 9.1) and ability (section 9.2) go through the AI's
knowledge.

## 7. The AI's damage estimate and effectiveness class

### 7.1 `ai_damage(move)`: `TrainerAI_CalcDamage` (`trainer_ai.c:2868`) with variance 100

All of Roark's moves take the `default` case (`:3074-3078`), with power 0 and type 0,
so the move's own data is used. The function then calls:

1. `BattleSystem_CalcMoveDamage(..., criticalMul = 1)` (`battle_lib.c:6601`), with
   the player side's side conditions and the field conditions;
2. `BattleSystem_ApplyTypeChart` (`battle_lib.c:2560`) for STAB and each defender
   type;
3. 0 if the result is flagged `MOVE_STATUS_IMMUNE`, otherwise
   `BattleSystem_Divide(damage * 100, 100)`, which leaves it unchanged.

Every one of Roark's script calls passes `USE_MAX_DAMAGE` (0), so the variance is
always 100 (`:1050` with `:2830-2833`, and `:1531-1537`).

This equals plan 1a's `damage_for_roll(tables, state, 1, move, crit=False, roll=15)`
(roll 15 = 85 + 15 = 100 %). The engine applies the variance before STAB and the
chart, but at 100 % that step leaves the damage unchanged.

The real move pipeline also applies Life Orb and Metronome
(`BattleScript_CalcMoveDamage`). The AI estimate skips them. Roark's Pokemon hold no
items, so the two agree for him.

`battleCtx->powerMul` is reset to 10 by `BattleContext_Init`
(`battle_lib.c:1980`) before every turn's decision.

### 7.2 Effectiveness class: `IfMoveEffectivenessEquals` (`trainer_ai.c:1311`)

1. Start from `TYPE_MULTI_BASE_DAMAGE` = 40.
2. Apply `BattleSystem_ApplyTypeChart`: STAB x15/10 when `A` shares the move's type
   (40 -> 60), then `Divide(x * m, 10)` for each defender type in chart-row order,
   with m in {0, 5, 20}.
3. Map the result: 120 -> 80 (DOUBLE), 240 -> 160 (QUADRUPLE), 30 -> 20 (HALF),
   15 -> 10 (QUARTER).
4. If the status mask has any of `MOVE_STATUS_INEFFECTIVE | WONDER_GUARD | LEVITATED
   | MAGNET_RISE` (= `MOVE_STATUS_IMMUNE`, `include/constants/battle/moves.h:87`), the
   result is 0 (IMMUNE).

Without STAB the raw values already equal the class constants: 80, 160, 20, 10 and
40 for neutral. With STAB a neutral hit stays 60, which matches no class. The
power-0 moves go through the same chart, so Stealth Rock against a Fire/Flying
target is QUADRUPLE. Levitate and Wonder Guard come from `D`'s true ability. Under
Wonder Guard, every non-super-effective damaging move is IMMUNE.

## 8. Random draws

`BattleSystem_RandNext` (`src/battle/battle_system.c:1308`) returns a `u16`, uniform
over 0..65535. `x % 256`, `x % 16`, `x % 2` and `x & 1` are therefore exactly
uniform. `x % 3` and `x % 10` are not: 65536 = 3·21845 + 1 = 10·6553 + 6, so residue
0 is slightly more likely. All draws are treated as independent.

### 8.1 Simulated damage rolls (`AiOutcome.sim_roll`)

- `moveDamageRolls[i] = 100 - RandNext % 16`, for i = 0..3 (`trainer_ai.c:240`).
  Each roll is uniform on 85..100.
- These four draws are made at every FIGHT decision, but Roark's scripts never read
  them. Every `FlagMoveDamageScore` and `IfCurrentMoveKills` in BASIC, EVAL_ATTACK and
  EXPERT passes `USE_MAX_DAMAGE`.
- **For Roark, `sim_roll` has the single value 100 (roll index 15).** Enumerating
  16^4 values would multiply the outcome count by 65536 for nothing. It would also
  push Task 2's enumerator over its 4096 limit.

### 8.2 Ability guess (`AiOutcome.ability_guess`)

`LoadBattlerAbility DEFENDER` (`trainer_ai.c:1170`) draws `RandNext & 1` only when all
of these hold:

- `D`'s ability is not suppressed;
- the AI has not learned it (`battlerAbilities[D] == 0`);
- `D`'s true ability is not Shadow Tag, Magnet Pull or Arena Trap;
- both species abilities are nonzero.

A draw of 1 gives `SPECIES_DATA_ABILITY_1` and 0 gives `ABILITY_2`, each with
probability 1/2.

- Encoding: `ability_guess = 0` means ability slot 0 (ability 1), and
  `ability_guess = 1` means slot 1 (ability 2), each 1/2.
- The decomp draws afresh at every `LoadBattlerAbility DEFENDER`. A decision makes
  up to 2 such reads per move slot.
- **One scalar is exact for Roark.** At most one read per decision can change a
  score:
  - Onix's Screech: the `Basic_CheckSoundproof` read matters for species with
    Soundproof plus a second ability (Voltorb, Electrode, Mr. Mime, Mime Jr.). The
    `Basic_CheckClearBodyEffect` read matters for Clear Body or White Smoke plus a
    second ability (Tentacool, Tentacruel). No species has both, and a Soundproof
    guess ends the slot before the Clear Body read.
  - Cranidos's Leer: only the Clear Body read matters.
  - The immunity read for Geodude's and Onix's damaging moves never changes a score
    (section 5.1, step 4).

  I checked this against the `abilities` of every `res/pokemon/*/data.json`.
- `CheckBattlerAbility` (`trainer_ai.c:1212`) has a different, RNG-free guess rule.
  Roark's scripts never reach it.

### 8.3 Script rolls (`AiOutcome.script_rolls`, `MAX_SCRIPT_ROLLS = 4`)

- Every script read is `IfRandomLessThan N`: it jumps when `RandNext % 256 < N`, with
  probability exactly N/256.
- A raw draw (0..255) does not fit `int8`. **Encoding: `script_rolls[k] = 1` means
  the draw was below N (the jump is taken), with probability N/256. 0 means it was
  not.**
- The sites are fixed per Roark Pokemon and listed in script execution order. Sites
  run flag by flag (BASIC, EVAL_ATTACK, EXPERT), and within a flag slot by slot.
- A site that is not reached draws nothing, and its entry is ignored.

| Active | k | Flag, slot (move) | Site | N | Effect when the draw is NOT below N | Reached when |
|---|---|---|---|---|---|---|
| Geodude | 0 | EVAL_ATTACK, 0 (Stealth Rock) | `script.s:6383` | 80 | +2 | Stealth Rock is QUADRUPLE vs `D` |
| Geodude | 1 | EVAL_ATTACK, 1 (Rock Throw) | `script.s:6383` | 80 | +2 | it does not KO, is HIGHEST and is QUADRUPLE |
| Geodude | 2 | EXPERT, 0 (Stealth Rock) | `script.s:6274` | 128 | +1 | Stealth Rock has PP |
| Geodude | 3 | (unused) | | | | |
| Onix | 0 | EVAL_ATTACK, 0 (Stealth Rock) | `script.s:6383` | 80 | +2 | as Geodude k0 |
| Onix | 1 | EVAL_ATTACK, 1 (Rock Throw) | `script.s:6383` | 80 | +2 | as Geodude k1 |
| Onix | 2 | EXPERT, 0 (Stealth Rock) | `script.s:6274` | 128 | +1 | Stealth Rock has PP |
| Onix | 3 | EXPERT, 2 (Screech) | `script.s:2328` | 50 | -2 | A's HP% < 70, or D's Def stage <= -3 (engine) |
| Cranidos | 0 | EVAL_ATTACK, 1 (Pursuit) | `script.s:6383` | 80 | +2 | it does not KO, is HIGHEST and is QUADRUPLE (needs a Ghost/Psychic dual type; no species has one) |
| Cranidos | 1 | EXPERT, 1 (Pursuit) | `script.s:3726` | 128 | +1 | first turn after entering, or `D` is Ghost or Psychic type |
| Cranidos | 2 | EXPERT, 1 (Pursuit) | `script.s:3731` | 128 | +1 | the AI has seen U-turn from `D` |
| Cranidos | 3 | EXPERT, 2 (Leer) | `script.s:2328` | 50 | -2 | A's HP% < 70, or D's Def stage <= -3 (engine) |

The site layout depends on which Pokemon is active, through its moveset. Headbutt,
Screech and Leer have no EVAL_ATTACK site because a Normal-type move is never
QUADRUPLE.

### 8.4 Tie-break (`AiOutcome.tiebreak`)

- One draw, `RandNext % n`, where n = 1..3 is the number of tied top-scoring slots
  (`trainer_ai.c:338`). The draw is made even when n = 1.
- **Exact encoding: `tiebreak = RandNext % 12`**, in 0..11.
  - Values 0-3 have probability 5462/65536 each, and values 4-11 have 5461/65536
    each.
  - Then `tiebreak % n == RandNext % n` for every n that divides 12, which covers
    n = 1, 2, 3 and 4.
  - With these weights, n = 2 gives exactly 1/2 each, and n = 3 gives exactly
    21846/21845/21845 over 65536.
- An index drawn uniformly over 0..n-1 would be off by about 1.5e-5 for n = 3, so it
  is not exact.

### 8.5 Switching draws (`AiOutcome.switch_stop`)

The draws made by `TrainerAI_ShouldSwitch` are listed in section 11.1. For Roark,
only two kinds change the result:

- the 90 % "keep" draw for each super-effective damaging move of `A`;
- the first-success draws in the two party-member passes.

Their exact probabilities are in section 11.1. Task 3a encodes them as one field,
`AiOutcome.switch_stop` (the first stopping site, 9 = none; section 11.1,
"Encoding"). `sample_ai_outcome` and `enumerate_ai_outcomes` take the site layout as
`switch_sites`, one `SwitchSite` kind (`KEEP`, `PASS_INEFFECTIVE`,
`PASS_NOT_VERY_EFFECTIVE`, or `NONE` for unused entries) per draw, because working out
which sites exist is the switching rule itself (`maybe_switch`, Task 3).

### 8.6 Item use and post-KO replacement

Neither draws from the RNG for Roark. In singles, `BattleAI_PostKOSwitchIn` picks
its defender with `attacker ^ 1`, so it makes no draw either.

### 8.7 RNG read order (for trace replay, Task 9)

1. In `ShouldSwitch`, as reached (section 11.1).
2. In `TrainerAI_Init`, 4 damage rolls.
3. BASIC slots 0-3: each ability guess as reached.
4. EVAL_ATTACK slots 0-3: script rolls as reached.
5. EXPERT slots 0-3: script rolls as reached.
6. The tie-break.

`ShouldUseItem` makes no draws. If the decision ends with a switch or an item, steps
2-6 do not happen.

## 9. What the AI knows about the player

### 9.1 Seen moves (`aiContext.battlerMoves` -> `BattleState.ai_seen_moves`)

- `TrainerAI_RecordLastMove` (`trainer_ai.c:2705`) runs once per FIGHT decision, at
  the start of `TrainerAI_MainSingles` (`:295`). It copies
  `movePrevByBattler[D]` into the first empty slot of the AI's list, unless the move
  is already there.
- `movePrevByBattler[D]` is written in `UpdateMoveBuffers`
  (`battle_controller_player.c:3945-3949`). It is the player's last move if that move
  started executing, meaning `SYSCTL_MOVE_SUCCEEDED` was set at `:3212`: past sleep,
  freeze, flinch, full paralysis and the confusion self-hit. A miss still counts.
  Otherwise it is `MOVE_NONE`.
- **(quirk)** On a turn where Roark switches or uses an item, `TrainerAI_Main` does
  not run, so nothing is recorded. If the player then uses a different move before
  Roark's next FIGHT decision, the earlier move is never recorded.
- **Cleared on switch.** When the player's battler switches in,
  `BattleSystem_UpdateAfterSwitch` clears that battler's seen moves, known ability
  and known item (`battle_lib.c:2151-2153`) and resets `movePrevByBattler`
  (`:2124`). A faint does the same (`BattleSystem_CleanupFaintedMon`,
  `:2223-2225`). Knowledge is per battler position, so the engine clears the
  entering Pokemon's row of `ai_seen_moves` and `ai_ability_known`.
- Roark's scripts read the seen moves in exactly one place: `IfMoveNotKnown
  DEFENDER, MOVE_U_TURN` in `Expert_Pursuit`.

### 9.2 Ability (`aiContext.battlerAbilities` -> `BattleState.ai_ability_known`)

- Learned through `BattleAI_SetAbility` (`src/battle/battle_script.c:12106`). It is
  called when a battle message tags the battler's ability (`:11806`,
  `BattleMessage_AbilityTag`) and when a script writes `BATTLEMON_ABILITY`
  (`:3074`, `:3403`). It stores the true ability, so "known" means the AI uses the
  true ability.
- Resolution order in `LoadBattlerAbility DEFENDER` (`trainer_ai.c:1170-1209`):
  1. suppressed -> `ABILITY_NONE`;
  2. known -> the true ability;
  3. the true ability is Shadow Tag, Magnet Pull or Arena Trap -> the true ability;
  4. two nonzero species abilities -> the guess (section 8.2);
  5. otherwise -> `ABILITY_1`, or `ABILITY_2` if `ABILITY_1` is 0.
- `LoadBattlerAbility ATTACKER` always gives Roark's own true ability.
- In decomp ids (Task 3a): the true ability is
  `tables.species_abilities_decomp[species, BattleState.ability_slot]` and the two
  species abilities are `tables.species_abilities_decomp[species]`, also for
  abilities the engine maps to `UNSUPPORTED`.

### 9.3 Items

The AI records the player's held items (`BattleAI_SetHeldItem`,
`battle_script.c:12119`, when a message names the item). None of Roark's scripts read
that record. The damage estimate uses the player's true held item (section 6).

## 10. Picking among the top scores

From `TrainerAI_MainSingles` (`trainer_ai.c:315-338`):

- The candidate list starts as {slot 0}, with slot 0's score.
- For i = 1..3, but only if `A` has a move in slot i:
  - an equal score appends i;
  - a higher score resets the list to {i}.
- The result is every slot with the top score, in ascending order. Slot 0 is always a
  candidate, even when it is empty.
- The pick is `list[RandNext % n]` (section 8.4).

The result is the move slot index, which is plan 1a's opponent action 0..3. When no
move is usable the battle uses Struggle whatever the AI picked; the engine treats
action 0 as Struggle in that case.

## 11. Switching

### 11.1 Voluntary switch: `TrainerAI_ShouldSwitch` (`trainer_ai.c:3894`) for Roark

Checks run in this order:

1. **No switch** if any of these hold (`:3906-3913`):
   - `A` is trapped: `VOLATILE_CONDITION_TRAPPED` = Bind family | Mean Look, the
     engine's `trapped > 0`;
   - `A` has Ingrain;
   - a player battler's true ability is Shadow Tag or Arena Trap;
   - Magnet Pull while `A` is Steel type (never for Roark).
2. **No switch** if Roark has no other living party member (`:3915-3940`). The count
   excludes the active slot and `aiSwitchedPartySlot`, which is reset to 6 every
   turn (`battle_lib.c:2015`).
3. `AI_PerishSongKO`: Perish Song is not in M1 (coverage refuses it), so FALSE.
4. `AI_CannotDamageWonderGuard`: FALSE for Roark, with no draws. Only Shedinja has
   Wonder Guard, and each of Roark's Pokemon has a move super effective against
   Bug/Ghost: Rock Throw, or Pursuit for Cranidos. That returns before the party loop.
5. `AI_OnlyIneffectiveMoves`: FALSE for Roark, with no draws.
   - The check needs at least 2 damaging moves, all `MOVE_STATUS_INEFFECTIVE`.
   - Geodude and Onix have one damaging move each.
   - Cranidos's Pursuit is never ineffective: no type is immune to Dark, and Wonder
     Guard sets a different flag.
6. `AI_HasAbsorbAbilityInParty`: FALSE for Roark, because no Roark Pokemon has Flash
   Fire, Water Absorb or Volt Absorb. It can still consume one draw:
   - if `A` has a super-effective damaging move against `D`, it draws `RandNext % 3`
     (`:3651`);
   - the result does not matter, since every later branch returns FALSE.
7. `AI_IsAsleepWithNaturalCure`: FALSE, since no Roark Pokemon has Natural Cure.
8. **Keep draws.** `AI_HasSuperEffectiveMove(..., FALSE)` (`:3560`):
   - For each of `A`'s move slots in order whose move is super effective against `D`,
     draw `RandNext % 10`. A nonzero result (58982/65536) means **no switch**, and
     the check stops.
   - "Super effective" uses `ApplyTypeChart` flags, which are set only for moves with
     power, so status moves never qualify.
   - For Roark there is at most one such move: Rock Throw for Geodude and Onix, or
     Pursuit for Cranidos.
9. **No switch** if `A`'s positive stat stages sum to 4 or more
   (`AI_IsHeavilyStatBoosted`, `:3872`).
10. **Party pass 1:** `AI_HasPartyMemberWithSuperEffectiveMove(..., 0x8 =
    MOVE_STATUS_INEFFECTIVE, 2)` (`:3727`).
    - It needs `moveHit[A]`: the last move that hit `A` since `A`'s own last action.
      That move must be valid and have power > 0, and its user `moveHitBattler[A]`
      must be valid.
    - For each living bench slot j in party order (not active, not
      `aiSwitchedPartySlot`), compute `BattleSystem_CalcEffectiveness` of
      `moveHit[A]` against bench mon j. This uses j's types and ability and the
      attacker's true ability.
    - If the result has the INEFFECTIVE bit, then for each of j's moves in slot order
      that is super effective against the attacker (`CalcEffectiveness`, which
      ignores power, so Stealth Rock counts), draw `RandNext % 2`. A result of 0
      (1/2) means **switch to j**.
11. **Party pass 2:** the same with `0x4 = MOVE_STATUS_NOT_VERY_EFFECTIVE` and
    `RandNext % 3 == 0` (21846/65536) per qualifying move.
12. Otherwise **no switch**.

The code comments at `:3973-3980` describe passes 1 and 2 as 33 % and 25 % and use
the wrong masks. The rule above is what the code does.

**Encoding.** The result is the first site that "stops":

- a keep site stops with "no switch";
- a party-pass site stops with "switch to j".

If no site stops, the result is "no switch". The probability that site i is the
first to stop is `prod_{h<i}(1 - p_h) * p_i`. So a single categorical draw over the
first stopping site, at most 9 sites for Roark (1 keep + 4 + 4), is exact.

Before Roark's party-pass sites can exist, `moveHit[A]` must be set. That happens when
the player's damaging move hit `A` after `A` had already acted that turn, or when `A`
did not act. `moveHit[A]` is written at `battle_controller_player.c:4500-4507` and
cleared for the attacker at its move end (`:4389`, via `:3986`). The engine state
carries it as `BattleState.move_hit[1]` (Task 3a; `src/nuzline/engine/STATE.md` lists
every write and reset).

Example: Starly's Wing Attack or Tackle hits Cranidos. Geodude and Onix resist both
types, and their Rock moves are super effective against Starly, so pass 2 fires.
Section 13, trace F, works this through.

### 11.2 Replacement after a faint: `BattleAI_PostKOSwitchIn` (`battle_lib.c:7923`)

It is called from `Task_TrainerShowPartyMenu` (`battle_display.c:4543-4559`) when
`aiSwitchedPartySlot == 6`. Candidates are the living, non-active party slots.

1. **Stage 1** (`:7959-8031`):
   - Each candidate's score is `u8`:
     `TypeMatchupMultiplier(t1, D.t1, D.t2) + TypeMatchupMultiplier(t2, D.t1, D.t2)`.
     A mono-type mon counts its type twice.
   - `BattleSystem_TypeMatchupMultiplier` (`battle_lib.c:3009`) starts at 40 and
     multiplies by each matching chart row with `* m / 10`. It includes the
     Foresight-skippable Ghost rows.
   - **(quirk)** The `u8` sum wraps modulo 256 (`docs/bugs_and_glitches.md`, "Post-KO
     Switch-In AI Scoring Overflow"). For example, Cranidos (Rock/Rock) against
     Fire/Flying scores 160 + 160 = 320 -> 64.
   - Take the highest score, with ties going to the lower slot. A score of 0 is
     never picked.
   - If that mon has a move super effective against `D` (`CalcEffectiveness`, any
     power), pick it. Otherwise discard it and repeat.
2. **Stage 2** (`:8034-8088`), when no candidate passes stage 1:
   - For every candidate move with power != 1, the score is
     `CalcMoveDamage(move, attacker = the fainted battler, defender = D, criticalMul =
     1)` passed through `ApplyTypeChart` with the fainted battler as attacker, and 0
     if immune.
   - The fainted battler's stats, level, types and ability are used, not the
     candidate's. **(quirk)**
   - A status move scores the `+2` from `CalcMoveDamage` times the chart.
   - The `u8` score wraps.
   - Pick the highest, with ties going to the lower slot.
3. If both stages return 6: pick the first living non-active slot
   (`battle_display.c:4548-4556`).

This differs from Task 6's "first legal replacement slot" (see section 14).

## 12. Item use: `TrainerAI_ShouldUseItem` (`trainer_ai.c:4056`)

The bag is loaded at battle start: `trainerItems[0] = [17, 17, 0, 0]` and
`trainerItemCounts[0] = 2` (`battle_controller_player.c:4796-4812`).
`trainerItemCounts` is never decremented.

The rule is reached only when `ShouldSwitch` returned FALSE. It has no randomness.

1. Nothing if `A` is under Embargo.
2. `aliveMons` = Roark's living party members, the active one included.
3. Slot i of the bag is considered when `i == 0`, or when
   `aliveMons <= trainerItemCounts - i + 1` (`:4092`). For Roark:
   - slot 0 is always considered;
   - slot 1 only when at most 2 of his Pokemon are alive;
   - slots 2 and 3 are empty.
4. For a considered Potion (`hpRestore` flag set, `hpRestored = 20`,
   `res/items/data/potion.json`):
   - use it when `A.curHP > 0` and either `A.curHP < A.maxHP / 4` (integer) or
     `A.maxHP - A.curHP > 20` (`:4105-4116`);
   - empty slots are skipped (`continue`).
5. **(quirk)** The loop never breaks (`:4191-4194`). Once `result` is TRUE, every
   later considered non-empty slot is also written to `usedItem` and zeroed. With 2
   or fewer Pokemon alive and both Potions left, one use empties both slots but heals
   only once.
6. The heal runs in `BattleControllerPlayer_ItemCommand`
   (`battle_controller_player.c:1905-1919`, `subscript_use_potion`). Task 2b
   confirms the amount and the turn order.

In plan 1a + 2b encoding, the action is `10 + k`, where k is the first slot used.
Since Task 3a both engines also zero every later slot the decomp zeroes, from
`BattleState.trainer_item_count` (`trainerItemCounts`) and the living party count at
command selection. They apply slot k's item: see `docs/discrepancies.md` for a bag
that mixes items.

## 13. Hand-traced decisions (Task 3 fixtures)

Common setup for all traces:

- The state is Roark's real party (section 4), as built by
  `build_trainer_party(roark)` (IVs 6, natures from the PID, ability slot 0). The
  opponent's active slot is the Pokemon named.
- The player's party is `tests/fixtures/sample_party.toml`, with the named Pokemon as
  the active one.
- Unless a trace says otherwise:
  - it is the decision for turn 1 (`turn = 0`, both leads, so "first turn" is TRUE);
  - all stages are neutral and nobody has a status;
  - no hazards are up;
  - the AI knows nothing (no seen moves, `ai_ability_known = 0`);
  - `moveHit` is none;
  - the bag is `[17, 17, 0, 0]` and all 6 + 3 Pokemon are alive.
- Damage values are `damage_for_roll(..., crit=False, roll=15)` from the M1a engine.
- `script_rolls` follows the table in 8.3. Unreached entries are given as 0.

**Trace A: Geodude (32/32) vs Monferno (44/44).**

- Rock Throw `ai_damage` = 12, with no KO. Rock vs Fire/Fighting is 2 x 0.5, so
  NEUTRAL (60 with STAB), not QUADRUPLE. Stealth Rock is not QUADRUPLE either.
- Monferno's species ability is Blaze only, so no guess is made.
- `ShouldSwitch`: Geodude has no super-effective move, so there are no keep draws,
  and `moveHit` is none. No switch.
- `ShouldUseItem`: full HP, so no item.

| Variant | Outcome | BASIC | EVAL_ATTACK | EXPERT | Scores | Action |
|---|---|---|---|---|---|---|
| A1 | `script_rolls = [0,0,0,0]` | [100,100,0,0] | +0 | SR +1 | **[101, 100, 0, 0]** | 0 |
| A2 | `script_rolls = [0,0,1,0]`, `tiebreak = 0` | | | | **[100, 100, 0, 0]** | 0 |
| A3 | `script_rolls = [0,0,1,0]`, `tiebreak = 1` | | | | **[100, 100, 0, 0]** | 1 |
| A4 | player side `stealth_rock = 1`, `script_rolls = [0,0,0,0]` | SR -10 | | SR +1 | **[91, 100, 0, 0]** | 1 |

Distribution in the base state: P(action 0) = 1/2 + 1/2 · 1/2 = **3/4**,
P(action 1) = **1/4**.

**Trace B: Onix (31/31) vs Starly (30/30).**

- Rock Throw = 24, with no KO. Rock vs Normal/Flying is DOUBLE, not QUADRUPLE.
- Screech:
  - BASIC: Defense stage neutral, Starly's ability is Keen Eye only (no Clear Body),
    so no change.
  - EXPERT: Onix's HP% is 100 and Starly's Defense stage is above -3, so it goes to
    the target-HP check. Starly's HP% is 100 > 70, so it ends with no draw (site 3
    not reached).
- `ShouldSwitch`:
  - Rock Throw is super effective, so `HasAbsorbAbilityInParty` draws `% 3`. The
    result does not matter.
  - The keep draw: `% 10 != 0` means no switch.
  - If the keep draw is 0, the passes find `moveHit` none, so still no switch.
  - Never switches.

| Variant | Outcome | Scores | Action |
|---|---|---|---|
| B1 | `script_rolls = [0,0,0,0]` | **[101, 100, 100, 0]** | 0 |
| B2 | `script_rolls = [0,0,1,0]`, `tiebreak = 0` | **[100, 100, 100, 0]** | 0 |
| B3 | `script_rolls = [0,0,1,0]`, `tiebreak = 1` | **[100, 100, 100, 0]** | 1 |
| B4 | `script_rolls = [0,0,1,0]`, `tiebreak = 5` | **[100, 100, 100, 0]** | 2 |

Distribution: P(0) = 1/2 + 1/2 · 21846/65536, and P(1) = P(2) = 1/2 · 21845/65536.

**Trace C: Cranidos (43/43) vs Starly (30/30).**

- Headbutt = 32 >= 30, so it KOs: EVAL_ATTACK +4.
- Pursuit = 19, with no KO and not the highest: EVAL_ATTACK -1 and stop, so site 0
  is not reached.
- Expert Pursuit: first turn, so site 1 is reached. U-turn has not been seen, so
  site 2 is not reached.
- Leer: no BASIC change. EXPERT ends at the target-HP check (site 3 not reached).
- Cranidos has Mold Breaker, so its damaging moves skip the immunity ability read.
- `ShouldSwitch`: Headbutt and Pursuit are neutral against Normal/Flying, and
  `moveHit` is none. No switch.

| Variant | Outcome | Scores | Action |
|---|---|---|---|
| C1 | `script_rolls = [0,0,0,0]` | **[104, 100, 100, 0]** | 0 |
| C2 | `script_rolls = [0,1,0,0]` | **[104, 99, 100, 0]** | 0 |

Always action 0. This is the KO case behind Task 3's
`test_ko_move_preferred_when_available`.

**Trace D: Cranidos (30/43) vs Monferno (30/44).**

- HP%: Cranidos 30·100/43 = 69 < 70, so the Leer roll (site 3) is reached.
  Monferno: 30·100/44 = 68, not > 70, so the target-HP -2 applies.
- Headbutt = 19 (no KO, HIGHEST), and Pursuit = 6 (NOT_HIGHEST, -1).
- Items: 43 - 30 = 13 is not > 20, and 30 is not < 43/4 = 10. No Potion.
- `ShouldSwitch`: Headbutt is neutral and Pursuit is not very effective against
  Fire/Fighting, and `moveHit` is none. No switch.

| Variant | Outcome | Scores | Action |
|---|---|---|---|
| D1 | `script_rolls = [0,0,0,0]`, `tiebreak = 0` | **[100, 100, 96, 0]** | 0 |
| D2 | `script_rolls = [0,0,0,0]`, `tiebreak = 1` | **[100, 100, 96, 0]** | 1 |
| D3 | `script_rolls = [0,1,0,1]` | **[100, 99, 98, 0]** | 0 |
| D4 | `script_rolls = [0,1,0,0]` | **[100, 99, 96, 0]** | 0 |

**Trace G: Onix (31/31) vs Voltorb (L14 default moves, true ability Static, unrevealed).**

- Voltorb's species abilities are (Soundproof, Static), so the AI guesses.
- Rock Throw = 9, with no KO, and is neutral against Electric.
- Screech:
  - With guess Soundproof (`ability_guess = 0`): BASIC -10.
  - With guess Static: no change, and the Clear Body read cannot match.
- `ShouldSwitch`: no super-effective move, and `moveHit` is none. No switch.

| Variant | Outcome | Scores | Action |
|---|---|---|---|
| G1 | `ability_guess = 0`, `script_rolls = [0,0,0,0]` | **[101, 100, 90, 0]** | 0 |
| G2 | `ability_guess = 0`, `script_rolls = [0,0,1,0]`, `tiebreak = 1` | **[100, 100, 90, 0]** | 1 |
| G3 | `ability_guess = 1`, `script_rolls = [0,0,1,0]`, `tiebreak = 2` | **[100, 100, 100, 0]** | 2 |
| G4 | `ai_ability_known[player slot] = 1`, `ability_guess = 0`, `script_rolls = [0,0,0,0]` | **[101, 100, 100, 0]** | 0 |

Soundproof is `AbilityId.UNSUPPORTED` in the engine. Trace G needs the decomp ability
ids of the species (see section 14).

**Trace E: item use.** Geodude (11/32) vs Monferno (32/44) on turn 2.

- `moveHit` is none: Monferno (Spe 30) moved first, then Geodude moved, which clears
  `moveHit[Geodude]`.
- No switch, as in Trace A.

| Variant | State change | Result |
|---|---|---|
| E1 | (as above) | 32 - 11 = 21 > 20 -> Potion from slot 0. Action **10**, bag after `[0, 17, 0, 0]` |
| E2 | Onix fainted (Roark has 2 alive) | Slots 0 and 1 are both considered and both zeroed. Action **10**, bag after `[0, 0, 0, 0]`, HP 11 -> 31 |
| E3 | Geodude 12/32 | 20 is not > 20, and 12 is not < 8. No item, so the move scoring of Trace A applies |
| E4 | Geodude 7/32 | 7 < 32/4 = 8 -> Potion, action **10** |

**Trace F: voluntary switch.** Cranidos (40/43, active from the start as the
opponent's lead, party slot 2) vs Starly (30/30), decision for turn 2 (`turn = 1`).
This is not Cranidos's first decision: `fakeOutTurnNumber` 0 < `totalTurns` 1.

- Geodude (slot 0) and Onix (slot 1) are alive.
- `moveHit[Cranidos]` = Wing Attack, by Starly: Starly (Spe 19) moved after Cranidos
  (Spe 24) on turn 1.
- The AI records Wing Attack as seen, but only if this decision reaches FIGHT (section 9.1).
- Cranidos has no super-effective damaging move against Normal/Flying, so there are
  no keep draws.
- Pass 1: Flying is not ineffective against Rock/Ground, so it finds nothing.
- Pass 2: Flying is not very effective against Geodude and Onix. Their Stealth Rock
  and Rock Throw are super effective against Starly, which gives 4 sites, each
  stopping with p = 21846/65536:
  - Geodude Stealth Rock -> switch to slot 0;
  - Geodude Rock Throw -> switch to slot 0;
  - Onix Stealth Rock -> switch to slot 1;
  - Onix Rock Throw -> switch to slot 1.

With q = 43690/65536:

- P(action 4) = 1 - q^2 ≈ 0.555569;
- P(action 5) = q^2 (1 - q^2) ≈ 0.246912;
- P(no switch) = q^4 ≈ 0.197519.

With no switch, there is no item (3 HP missing, 40 >= 10), and the moves score:

- Headbutt 104 (KO, 32 >= 30);
- Pursuit 99 (not the first turn, Starly is not Ghost/Psychic, U-turn not seen, so
  no draw);
- Leer 100.

That gives **[104, 99, 100, 0]** and action 0.

## 14. Inputs the later tasks need that M1a does not have

**Status (Task 3a).** The "State fields", "Decomp ability ids", "`AiOutcome` width"
and "Item consumption" items below are done, in both engines where they touch the
state: `BattleState.move_hit`, `BattleState.fake_out_turn`,
`Tables.species_abilities_decomp` with `BattleState.ability_slot`,
`AiOutcome.switch_stop` (enumeration width 3840), and `BattleState.trainer_item_count`
with the double consumption (section 12). The post-KO replacement and the emulator
checks remain open.

- **State fields.** `BattleState` lacks the fields for:
  - `moveHit` / `moveHitBattler` (section 11.1): the last damaging move that hit the
    opponent's active since its own last action, and who used it. STATE.md lists
    them as "later (M6)".
  - `fakeOutTurnNumber` (section 5.3, Pursuit's first-turn check). STATE.md lists it
    as "later (M6)". The check can be derived from when the active entered: it is
    TRUE exactly on the first decision since the active entered (turn 1 for a
    lead).

  Task 3 (or Task 4, which adds the knowledge updates) has to add them to both
  engines.
- **Decomp ability ids for the guess.** `Tables.species_abilities` is mapped to
  `AbilityId`, where Soundproof, Clear Body, White Smoke, Wonder Guard, Volt Absorb,
  Water Absorb, Motor Drive, Flash Fire and Levitate are all `UNSUPPORTED`. The guess
  needs the decomp ids, or an AI-specific mapping of these abilities. This holds even
  when the player's true ability is supported (Voltorb with Static, Trace G).
- **`AiOutcome` width.**
  - With `sim_roll` fixed at 100, the remaining fields give 2 (`ability_guess`) x
    2^4 (`script_rolls`) x 12 (`tiebreak`) = 384 combinations.
  - The switching draws (section 11.1) are not in Task 2's interface. One extra
    categorical field (first stopping site, up to 10 values) keeps the product at or
    below 3840 <= 4096.
  - Separate Bernoulli fields per site would give 2^9 x 384 and overflow the limit.
- **Replacement after a faint** follows `BattleAI_PostKOSwitchIn` (section 11.2), not
  "first legal replacement slot" as Task 6's brief says. The two can pick different
  Pokemon. For example, after Geodude faints against a pure Fighting type, neither
  Onix nor Cranidos has a super-effective move, so stage 2 decides. Rock Throw is
  resisted and Headbutt is not, so the AI sends Cranidos (slot 2), while "first legal
  slot" would send Onix (slot 1).
- **Item consumption** (section 12, quirk 5) can zero two bag slots in one action.
  Task 2b's "sets the slot to 0" needs to cover this.

**To confirm with an emulator trace (Task 9):**

- **(1)** The double Potion consumption (section 12).
- **(2)** Missed records of seen moves on switch or item turns (section 9.1).
- **(3)** A faint replacement in the same turn the AI switched.
  `aiSwitchedPartySlot` still holds the slot that just switched in, so
  `Task_TrainerShowPartyMenu` would return that fainted slot (`battle_display.c:4543`).
  The AI code does not show what happens next. Until a trace shows it, the engine
  should refuse this path rather than guess.
- **(4)** The leads' `fakeOutTurnNumber` of 0.

## 15. JAX names for the C routines

Script command functions are listed in `docs/ai/script_commands.md`. The C-level
routines map as follows:

| Decomp routine | JAX name (module) | Notes |
|---|---|---|
| `TrainerAI_MainSingles` scoring loop | `ai_scores` (`policy.py`) | BASIC, EVAL_ATTACK, EXPERT over slots 0-3 from base 100 |
| `TrainerAI_PickCommand` + `TrainerAI_Main` | `ai_policy` (`policy.py`) | switch -> item -> moves, sections 2, 10-12 |
| `TrainerAI_ShouldSwitch` | `maybe_switch` (`policy.py`) | section 11.1 |
| `TrainerAI_ShouldUseItem` | `maybe_use_item` (`policy.py`) | section 12 |
| `BattleAI_PostKOSwitchIn` | `post_ko_switch_in` (`policy.py`) | section 11.2 |
| `TrainerAI_CalcDamage` (variance 100) | `ai_damage` (`commands.py`) | `damage_for_roll(..., crit=False, roll=15)`, section 7.1 |
| `IfMoveEffectivenessEquals` class | `effectiveness_class` (`commands.py`) | section 7.2 |
| `LoadBattlerAbility DEFENDER` resolution | `resolve_player_ability` (Task 4) | section 9.2 |
| `TrainerAI_RecordLastMove` | knowledge update in `step` (Task 4) | section 9.1 |
