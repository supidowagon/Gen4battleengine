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
