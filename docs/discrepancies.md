# Engine discrepancies

Places where two implementations disagreed (the JAX engine `src/nuzline/engine`, the
plain-Python reference `src/nuzline/reference`, and later the Smogon damage oracle), and
how the decomp (`pret/pokeplatinum`, pinned in `docs/decomp_paths.md`) settled each one.
The implementation that matches the decomp is right; the other one was fixed. Paths are
relative to `external/pokeplatinum`.

## Reference engine vs JAX engine (Task 10)

### 1. A fainted battler keeps its stages and volatiles until it is replaced

- **JAX before:** `_faint_check` only ended the Bind the fainted battler had put on the
  foe. Its stat stages, confusion, Taunt, Leech Seed, Water Sport and the other
  per-battler fields stayed until `replace`; the end-of-turn Taunt check even counted
  down the fainted battler's Taunt.
- **Decomp:** `subscript_faint_mon.s` -> `BtlCmd_PlayFaintAnimation`
  (`src/battle/battle_script.c:1795`) calls `BattleSystem_CleanupFaintedMon`
  (`src/battle/battle_lib.c:2156`), which resets `statBoosts`, clears `statusVolatile`,
  `moveEffectsMask` and `moveEffectsData`, and resets `movePrevByBattler` at the faint.
- **Fix:** JAX `_faint_check` resets the fainted active battler like a switch-in does
  (`step._reset_battler`). Test: `test_a_fainted_battler_loses_its_stages_and_volatiles`.

### 2. Struggle's recoil came after the defender's contact ability

- **JAX before:** the recoil was applied after `apply_move_effect`, so Static or Poison
  Point could paralyze or poison a Struggle user that the recoil then fainted.
- **Decomp:** Struggle's effect script (`res/battle/scripts/effects/effect_script_0254.s`)
  sets `MOVE_SIDE_EFFECT_ON_HIT | subscript_struggle`, a secondary effect.
  `BattleControllerPlayer_AfterMoveMessage` (`src/battle/battle_controller_player.c:3467`)
  runs `ONE_HIT_TRIGGER_SECONDARY` before `ONE_HIT_TRIGGER_ABILITY`, and Static and Poison
  Point need a living attacker (`BattleSystem_TriggerAbilityOnHit`,
  `src/battle/battle_lib.c:4215`).
- **Fix:** JAX applies the recoil inside `apply_move_effect`, after the move's effect and
  before the contact ability (`effects._struggle_recoil`). Test:
  `test_struggle_recoil_comes_before_static`.

### 3. Stealth Rock failed when the foe's active had fainted earlier in the turn

- **JAX before:** the no-target check passed only for moves aimed at the user, so a
  Stealth Rock used after the foe fainted (to its own recoil, say) failed.
- **Decomp:** `BattleSystem_Defender` (`src/battle/battle_lib.c:1663`) gives a
  `RANGE_OPPONENT_SIDE` move the foe's battler slot whatever its HP
  (`BattleSystem_RandomOpponent`, `:4194`), so `BattleControllerPlayer_HasNoTarget`
  (`src/battle/battle_controller_player.c:2306`) does not fail it.
- **Fix:** JAX `effects.defender_always` (the user-aimed ranges plus
  `RANGE_OPPONENT_SIDE`) decides the no-target check and whether the move lands. Test:
  `test_stealth_rock_lands_after_the_foe_fainted_from_its_own_recoil`.

### 4. Natural Cure came after a Pursuit that ended the battle

- **JAX before:** Natural Cure ran in the switch-out, after the Pursuit hit. When
  Pursuit fainted the player's switcher the battle ended first, so the fainted Pokemon
  kept its status.
- **Decomp:** `res/battle/scripts/subscripts/subscript_switch_pokemon.s:5-6` restores
  the status (`BtlCmd_TryRestoreStatusOnSwitch`, `src/battle/battle_script.c:9139`)
  before the Pursuit call at `:11`.
- **Fix:** JAX `execute_actions` applies `step._natural_cure` to a switching first actor
  before the Pursuit hit. Test:
  `test_natural_cure_comes_before_a_pursuit_that_faints_the_switcher`.

## Reference engine vs JAX engine: Hypothesis differential test (Task 11)

`tests/test_differential.py` runs 500 generated states (`tests/strategies.py`: 1-6 M1
Pokemon per side, random HP, status, PP, stages, volatiles, hazards, screens and
weather), legal action pairs and uniformly drawn outcomes through both engines. It
found no disagreement beyond the four above; there was nothing new to settle.

## Damage oracle: @smogon/calc vs JAX engine (Task 11)

`tests/test_damage_oracle.py` compares `damage_table` with @smogon/calc 0.12.0's Gen 4
calculator (`tools/smogon_calc/damage.js`) on 300 generated matchups, all 16 rolls
exact. The stats and move data agreed everywhere. The damage differed in two cases, and
in both the decomp sides with the JAX engine, so the engine is unchanged and the test
does not generate those matchups (`_smogon_departs`).

### 5. Sandstorm's SpD boost comes before the stat stage

- **@smogon/calc:** `calculateDefenseDPP` applies the SpD stage, then multiplies by 1.5
  for a Rock-type defender in a sandstorm. With a non-zero SpD stage the rounding
  differs (Zubat's Air Cutter on Onix at SpD -3 in sand: 5-6 against 6-7).
- **Decomp:** `BattleSystem_CalcMoveDamage` multiplies `spDefenseStat` by 15 / 10 at
  `src/battle/battle_lib.c:6925-6928`, before the stage scales it into `stageDivisor`
  (`:7008-7018`).
- **Ruling:** the engine (`damage._pre_random` boosts `sp_defense` before `apply_stage`) is
  right. Test: `test_sandstorm_spd_boost_comes_before_the_stage`.

### 6. A crit applies its stage rule after Simple doubles the stage

- **@smogon/calc:** with Simple, `calculateAttackDPP` / `calculateDefenseDPP` apply the
  doubled stage whatever the crit, so a critical hit from a Simple attacker at -3
  Attack is still cut to -6, and one on a Simple defender at +2 Defense still meets +4.
- **Decomp:** `BattleSystem_CalcMoveDamage` doubles the stages for Simple first
  (`src/battle/battle_lib.c:6850-6884`), then a crit ignores an attack stage at or
  below neutral and a defense stage at or above neutral (`:6947-6973` physical,
  `:6992-7018` special).
- **Ruling:** the engine (`damage._pre_random`: Simple, then the crit rule) is right.
  Test: `test_crit_stage_rule_applies_after_simple_doubles`.

## Deliberate differences from the decomp (both engines agree)

- **`turn` on the last turn.** `BattleControllerPlayer_TurnEnd`
  (`src/battle/battle_controller_player.c:1844`) does not increment `totalTurns` on the
  turn the battle ends; both engines count every played turn (`turn + 1`). A finished
  state is never stepped again, so nothing reads the difference.
- **Illegal actions.** Both engines assume legal actions (`engine.actions.legal_actions`).
  Choosing a move slot with 0 PP while another move is usable is outside the contract:
  the decomp fails the move (`MOVE_STATUS_NO_PP`, `BattleControllerPlayer_DecrementPP`,
  `src/battle/battle_controller_player.c:2235`) and so does the reference, while the JAX
  engine does not check.
