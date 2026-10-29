# AI script commands used by Roark's flags

This file has one row for each script command that Roark's three flag scripts
(`AI_FLAG_BASIC`, `AI_FLAG_EVAL_ATTACK`, `AI_FLAG_EXPERT`) can execute for his
moves. `docs/ai/roark.md` section 5.4 explains how the set was derived. Rows are in
order of first execution: BASIC first, then EVAL_ATTACK, then EXPERT.

- **Sources.** Macros come from `asm/macros/aicmd.inc`, and the handler `AICmd_<name>`
  comes from `src/battle/trainer_ai/trainer_ai.c` (line given). Both paths are
  relative to `external/pokeplatinum`.
- **Battler argument.** `D` is the player's active Pokemon (`AI_BATTLER_DEFENDER`)
  and `A` is Roark's active Pokemon (`AI_BATTLER_ATTACKER`).
- **Jumps.** `jump` is a label. In JAX the scripts are straight-line: every command
  runs, masked by the conditions of the branches that lead to it. A jump command
  therefore yields its condition, and the masks follow it.
- **JAX functions.** Each one is in `src/nuzline/ai/commands.py`, with signature
  `(ctx: AiContext, ...) -> AiContext`.
- **`calcTemp`.** This is the AI's scratch register. The `Load*` and `Flag*` commands
  write it, and the `IfLoaded*` and `IfTemp*` commands read it.

| command | arguments | semantics | RNG? | JAX function name |
|---|---|---|---|---|
| `IfTargetIsPartner` | `jump` | Jumps when `A` and `D` are on the same side (`(attacker & 1) == (defender & 1)`, `:2630`). This is never true in a single battle, so the command is a no-op for Roark. | no | `if_target_is_partner` |
| `IfMoveEqualTo` | `move`, `jump` | Jumps when the move being scored equals `move` (`:834`). | no | `if_move_equal_to` |
| `FlagMoveDamageScore` | `bool` (`USE_MAX_DAMAGE` = 0 / `ROLL_FOR_DAMAGE` = 1) | Sets `calcTemp` (`:1012`). It is `AI_NO_COMPARISON_MADE` (0) when the move is not damaging (power <= 1 and no alternate-power effect, or a no-calc effect). Otherwise it computes `ai_damage` for all four of `A`'s slots and gives `AI_MOVE_IS_HIGHEST_DAMAGE` (2) when no slot's damage is strictly greater than this slot's, and `AI_NOT_HIGHEST_DAMAGE` (1) otherwise. With `USE_MAX_DAMAGE` the variance is 100. With `ROLL_FOR_DAMAGE` slot i would use `moveDamageRolls[i]`, but Roark's scripts never pass it. | no (reads `sim_roll` only for `ROLL_FOR_DAMAGE`) | `flag_move_damage_score` |
| `IfLoadedEqualTo` | `value`, `jump` | Jumps when `calcTemp == value` (`:786`). | no | `if_loaded_equal_to` |
| `IfMoveEffectivenessEquals` | `TYPE_MULTI_*` class, `jump` | Jumps when the effectiveness class of the move being scored against `D` equals the given class (`:1311`). The class is IMMUNE 0, QUARTER 10, HALF 20, BASE 40, STAB 60, DOUBLE 80 or QUADRUPLE 160, computed from 40 through `ApplyTypeChart` (STAB included) and then mapped. A true Levitate, Wonder Guard or Magnet Rise gives IMMUNE. See `roark.md` section 7.2. | no | `if_move_effectiveness_equals` |
| `LoadBattlerAbility` | battler | Sets `calcTemp` to an ability (`:1170`). For `A` it is the true ability. For `D` it is `ABILITY_NONE` if suppressed, otherwise the known ability, otherwise the true ability if that is Shadow Tag, Magnet Pull or Arena Trap. Failing those, a species with two nonzero abilities gives a coin flip (`RandNext & 1`: 1 gives ability 1, 0 gives ability 2), and a species with one gives that one (`roark.md` 9.2). | yes: `ability_guess` (1/2 each), only when `D`'s ability is unknown and the species has two | `load_battler_ability` |
| `LoadTypeFrom` | `LOAD_*` target | Sets `calcTemp` to a type (`:937`). Roark uses three targets: `LOAD_MOVE_TYPE` (the type of the move being scored), and `LOAD_DEFENDER_TYPE_1` and `LOAD_DEFENDER_TYPE_2` (`D`'s true types; a mono-type has type 2 = type 1). | no | `load_type_from` |
| `IfTempEqualTo` | `value`, `jump` | Jumps when `calcTemp == value` (`:1078`). Same as `IfLoadedEqualTo`, with the operands in the other order. | no | `if_temp_equal_to` |
| `GoTo` | `jump` | Unconditional jump (`:2561`). | no | `go_to` |
| `IfLoadedNotEqualTo` | `value`, `jump` | Jumps when `calcTemp != value` (`:798`). | no | `if_loaded_not_equal_to` |
| `IfCurrentMoveEffectEqualTo` | `BATTLE_EFFECT_*`, `jump` | Jumps when the effect of the move being scored equals the argument (`:1441`). | no | `if_current_move_effect_equal_to` |
| `IfStatStageEqualTo` | battler, `BATTLE_STAT_*`, `value`, `jump` | Jumps when the battler's true stat stage equals `value` (`:1495`). Decomp stages run 0..12 with 6 neutral, so decomp = engine + 6. | no | `if_stat_stage_equal_to` |
| `IfSideCondition` | battler, `SIDE_CONDITION_*` mask, `jump` | Jumps when the battler's side has any condition in the mask (`:732`). Roark uses it only for `SIDE_CONDITION_STEALTH_ROCK` on the player's side (`stealth_rock[0]`). | no | `if_side_condition` |
| `CountAlivePartyBattlers` | battler | Sets `calcTemp` to the number of that side's party members that are alive and not active (`:1126`). For `D`: player party slots with species > 0 and HP > 0, other than the active one. | no | `count_alive_party_battlers` |
| `AddToMoveScore` | `value` (signed) | Adds `value` to the current slot's score, then raises a negative result to 0 (`:576`). It is also the body of every `ScoreMinusN` and `ScorePlusN` label (`script.s:1567-1621`). | no | `add_to_move_score` |
| `PopOrEnd` | none | Ends the current slot for this flag (`:2568`). Roark's scripts never push, so the stack is always empty. | no | `pop_or_end` |
| `IfCurrentMoveKills` | `bool` (`USE_MAX_DAMAGE` / `ROLL_FOR_DAMAGE`), `jump` | Jumps when the move being scored is damaging and `D.curHP <= ai_damage(move)` (`:1525`). `D.curHP` is the true HP. With `USE_MAX_DAMAGE` the variance is 100; with `ROLL_FOR_DAMAGE` it would be `moveDamageRolls[slot]`. | no (reads `sim_roll` only for `ROLL_FOR_DAMAGE`) | `if_current_move_kills` |
| `IfRandomLessThan` | `N`, `jump` | Jumps when `RandNext % 256 < N` (`:528`), with probability exactly N/256. In `AiOutcome` this is `script_rolls[k] = 1` for the site's fixed index k (`roark.md` 8.3). | yes: `script_rolls[k]`, Bernoulli(N/256) | `if_random_less_than` |
| `IfMoveKnown` | battler, `move`, `jump` | Jumps when the battler knows `move` (`:1627`). For `A` (and its partner) this checks the true moveset. For `D` it checks the AI's seen-move list (`ai_seen_moves`). | no | `if_move_known` |
| `IfHPPercentLessThan` | battler, `percent`, `jump` | Jumps when `curHP * 100 / maxHP < percent`, with integer division (`:588`). | no | `if_hp_percent_less_than` |
| `IfStatStageGreaterThan` | battler, `BATTLE_STAT_*`, `value`, `jump` | Jumps when the battler's true stat stage, in decomp units, is greater than `value` (`:1480`). | no | `if_stat_stage_greater_than` |
| `IfHPPercentGreaterThan` | battler, `percent`, `jump` | Jumps when `curHP * 100 / maxHP > percent`, with integer division (`:603`). | no | `if_hp_percent_greater_than` |
| `LoadIsFirstTurnInBattle` | battler | Sets `calcTemp` to TRUE when `fakeOutTurnNumber >= totalTurns`, otherwise FALSE (`:2480`). This is TRUE exactly on the battler's first decision after it entered (`roark.md` 5.3). | no | `load_is_first_turn_in_battle` |
| `IfMoveNotKnown` | battler, `move`, `jump` | The negation of `IfMoveKnown` (`:1683`). For `D` it checks the AI's seen-move list. | no | `if_move_not_known` |

**Count check.** `grep -c '^|' docs/ai/script_commands.md` gives 26: 24 command rows
plus the header and separator lines.
