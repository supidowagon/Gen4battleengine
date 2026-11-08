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
exact. The stats and move data agreed everywhere. The damage differed in three cases, and
in each the decomp sides with the JAX engine, so the engine is unchanged and the test
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

### 7. Selfdestruct and Explosion halve Defense before the stat stage

- **@smogon/calc:** `calculateDefenseDPP` applies the Defense stage, then halves the
  result for Explosion and Self-Destruct. With a non-zero Defense stage the rounding
  differs (Defense 51 at +1: 76 / 2 = 38 against 25 x 3 / 2 = 37).
- **Decomp:** `BattleSystem_CalcMoveDamage` halves `defenseStat` for
  `BATTLE_EFFECT_HALVE_DEFENSE` at `src/battle/battle_lib.c:6943-6945`, before the stage
  scales it into `stageDivisor` (`:6963-6975`).
- **Ruling:** the engine (`damage._pre_random` halves `defense` before `apply_stage`) is
  right. Test: `tests/engine/test_self_destruct.py::test_self_destruct_halves_defense`.

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
- **End-of-turn order with a fainted battler.** The reference sorts the end-of-turn
  order with `_player_second` (`BattleSystem_CompareBattlerSpeed`, which puts a fainted
  battler after a living one, `src/battle/battle_lib.c:1205-1211`), while the JAX
  engine's `step.end_of_turn_order` compares speed only (`ignoreQuickClaw = TRUE`,
  `battle_lib.c:3364`) and ignores HP. When one active battler has fainted the two
  orders can differ, but the results are identical: every implemented end-of-turn check
  on a fainted active battler is a no-op (held item, Leech Seed, poison, toxic, burn
  and Bind require the battler to be alive; its Taunt counter and Leech Seed were
  already cleared when it fainted), so running it first or second changes nothing.

- **A player's Struggle is not recorded as a seen move.** Both engines follow the
  decomp's timing: `last_move` is `movePrevByBattler`, written at `UpdateMoveBuffers`
  (`src/battle/battle_controller_player.c:3945-3949`; a pre-switch Pursuit at
  `src/battle/battle_script.c:6930`), and `step` commits it to `ai_seen_moves` at the
  start of a turn on which the AI fights, as `TrainerAI_RecordLastMove`
  (`src/battle/trainer_ai/trainer_ai.c:2705-2717`, called at `:295`) does; the AI's
  scores read the pending move too (`ai.commands.new_context`). `ai_seen_moves` holds
  the AI's list as slots of the player's active Pokemon, and Struggle (`MOVE_STRUGGLE`,
  165) is in no slot, so it is never recorded. The decomp adds it to `battlerMoves` like
  any move. Roark's scripts never ask for Struggle (`IfMoveNotKnown DEFENDER,
  MOVE_U_TURN` is the only read), so the one observable difference is a full list: with
  Struggle taking one of its four entries, the decomp cannot record the player's fourth
  distinct move, while the engines can. That needs a player Pokemon that Struggles (no
  PP left, or Taunted with only status moves) and later uses all four of its moves.
- **A locked battler's move is fixed at the start of the turn.** `FightCommand`
  (`src/battle/battle_controller_player.c:1885-1888`) re-reads `Battler_CanPickCommand` when
  the battler acts, and `ppDecremented` is set at command selection (`:335`). The engines
  take both from the turn's starting state (`step._execute_actions`: `choices[2]`). They differ
  only when the foe unlocks the battler earlier in the same turn (puts it to sleep or freezes
  it) and the battler still acts (a freeze it then thaws out of): the decomp would then use
  `moves[moveSlot]` and spend PP, the engines use the locked move and spend none.
- **Bide's unleash ignores the foe's chart and records `moveHit`.** The unleash script
  prints no attack message of its own; the engines write `moveHit[foe] = Bide` for a
  successful unleash (the foe is the defender `CHECK_STATUS_STATE_BIDE` picks, `:2704-2713`),
  assuming the controller's attack-message flag is set as for any damaging move. A Bide
  that failed (nothing stored) or found no target leaves `moveHit` untouched.
- **A fainted player Pokemon's AI knowledge.** `BattleSystem_CleanupFaintedMon`
  (`src/battle/battle_lib.c:2223-2225`) clears the AI's knowledge of a fainted battler;
  the engines clear a player Pokemon's `ai_seen_moves` / `ai_ability_known` row only
  when it enters (`step._forget`). A player faint ends the battle (spec §3), so nothing
  reads the fainted Pokemon's row.

## Not implemented in either engine (M6; M1b's coverage registry must refuse them)

- **The item a multi-slot bag use applies.** `TrainerAI_ShouldUseItem`
  (`src/battle/trainer_ai/trainer_ai.c:4056`) never breaks out of its bag loop: after a
  use, every later slot it considers (`aliveMons <= trainerItemCounts - i + 1`,
  `:4092`) and finds non-empty is zeroed too, and also becomes `usedItem` (`:4191-4194`),
  so the item applied is the last one zeroed. Both engines zero the same slots (item
  action `10 + k`, `trainer_item_count`) but apply slot `k`'s item. The two agree when
  the zeroed slots hold the same item, as every M1 bag does (Roark: two Potions); a bag
  mixing items (for example a Potion before a Super Potion) would heal by the wrong
  item. M1b's coverage registry must refuse a trainer bag that mixes item ids, or that
  holds an item without a supported `TrainerItemId`.

- **Field-condition end-of-turn pass: Wish and Trick Room.** Both engines run
  `BattleControllerPlayer_CheckFieldConditions` (`src/battle/battle_controller_player.c:932-1254`;
  `step.field_conditions`, reference `_field_conditions`, K3b1 Task 3): the side-condition
  and Gravity countdowns, the weather countdown and the sandstorm / hail chip. Still
  missing in both: Wish's heal (`FIELD_COND_CHECK_STATE_WISH`, `:1085-1111`; no supported
  move sets `wish_timer`) and the Trick Room countdown of the turn end's side-condition
  pass (`:1823-1826`), so a nonzero `trick_room` (which `tests/strategies.py` draws) never
  counts down; coverage refuses every move and area condition that sets Trick Room. The
  weather effects of Dry Skin, Solar Power, Rain Dish and Hydration are not modelled
  (those abilities map to `AbilityId.UNSUPPORTED`, which coverage refuses).
