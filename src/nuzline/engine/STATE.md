# BattleState checklist against the decomp battle structs

Every field of the three decomp battle-state structs (`docs/decomp_paths.md`,
"Battle-state struct" rows; pinned `pret/pokeplatinum` commit) has one row here,
saying where it lives in `BattleState` (`state.py`) or why it does not.
`MoveEffectsData` (nested in `BattleMon`) and `FieldConditions` (embedded in
`BattleContext`) are expanded too, because most per-battler and field mechanics
live in them.

Status is one of:

- `mapped`: the named `BattleState` field(s) hold this state (the mechanic may
  still be implemented in a later task or milestone).
- `later (<milestone>)`: no field yet; the milestone (spec §8) adds one when the
  mechanic is implemented. M1b is Roark's AI, M3 the save parser, M6 coverage
  expansion beyond Roark.
- `not needed because ...`: the reason.

`tests/engine/test_state.py::test_state_md_accounts_for_every_struct_field`
checks every row's status, that every `mapped` row names real `BattleState`
fields, and (when `external/pokeplatinum` is cloned) that every field of the
three structs has a row.

## Mapping notes

- **Search fields.** `deaths` (per player party slot), `faint_budget` (`k`) and
  `protected` (per player slot) are not decomp state: the death record comes from
  the faint check (`step._faint_check`) and the other two are the search's inputs,
  traced data the engine never reads. `replace_mul` is the decomp's `powerMul` kept
  for a pending player replacement (it persists until `BattleContext_Init`,
  battle_controller_player.c:1855), 0 otherwise.
- **Battlers vs. party slots.** The decomp keeps a `BattleMon` per battler
  (`BattleContext.battleMons[4]`, built from the party mon by
  `BattleSystem_InitBattleMon`, `src/battle/battle_lib.c:70`) and writes HP, PP,
  status and item back to the party. `BattleState` instead keeps every party
  slot's persistent data in `[2,6]` arrays and the active Pokemon's volatile
  data in `[2]` arrays indexed by side; `active[side]` is the battler's party slot.
- **Stats.** `attack`/`defense`/`speed`/`spAttack`/`spDefense` become
  `stats[..., 0..4]` in Atk, Def, SpA, SpD, Spe order (`ids.Stat` minus one);
  HP is `max_hp`. IVs become `ivs` in `ids.Stat` order (HP, Atk, Def, SpA, SpD, Spe).
- **Stages.** `statBoosts[8]` is HP, Atk, Def, Spe, SpA, SpD, Acc, Eva with 6
  neutral; `stages[side, ids.Stat]` is HP, Atk, Def, SpA, SpD, Spe, Acc, Eva with
  0 neutral (index 0 unused).
- **Status.** The decomp packs sleep turns and the toxic counter into `status`
  bits (`include/constants/battle/condition.h`); `BattleState` splits them into
  `status` (`ids.Status`) and `status_counter`.
- **statusVolatile** bits mapped: confusion turns -> `confusion`, flinch ->
  `flinched`, bind turns -> `trapped`, recharging -> `recharge`, Foresight ->
  `foresight`, Bide turns -> `bide_turns`, Defense Curl -> `curled`,
  `VOLATILE_CONDITION_MOVE_LOCKED` -> `locked_move` (with `moveLockedInto`), the
  ATTRACT bits -> `infatuated` (below). The
  other bits (Uproar, Thrash, Focus Energy, Transform, Rage, Substitute
  flag, Destiny Bond, Mean Look, Nightmare, Curse, Torment) are later (M6); the Substitute flag is implied by `substitute_hp > 0`.
- **moveEffectsMask** bits mapped: Leech Seed -> `leech_seed`, Water Sport ->
  `water_sport`, Mud Sport -> `mud_sport`, Aqua Ring -> `aqua_ring` (per battler, cleared on switch-out; the
  Electric power halving reads every active battler, `battle_lib.c:6812`). The rest
  (Perish Song, semi-invulnerable turns, Minimize,
  Charge, Ingrain, Imprison, Grudge, Gastro Acid, Miracle
  Eye, Power Trick, Heal Block, Embargo, Magnet Rise, Lock-On) are
  later (M6).
- **fieldConditionsMask**: the weather bits (temporary vs. permanent rain, sand,
  sun, hail, plus deep fog) -> `weather` with `weather_timer == 0` meaning
  permanent; Trick Room turns -> `trick_room`; Gravity turns -> `gravity`. The
  Uproar battler mask is later (M6).
- **sideConditionsMask**: Stealth Rock -> `stealth_rock`, Tailwind turns ->
  `tailwind`, Lucky Chant turns -> `lucky_chant`. Its Reflect, Light Screen,
  Spikes, Safeguard, Mist, Toxic Spikes, Future Sight and Wish bits are implied by
  the nonzero turn/layer counts in the mapped fields.
- **The field-condition pass** (`step.field_conditions`,
  `BattleControllerPlayer_CheckFieldConditions`, battle_controller_player.c:932-1254)
  counts down `reflect`, `light_screen`, `mist`, `safeguard`, `tailwind`, `lucky_chant`,
  a temporary weather's `weather_timer` (the weather ends at 0) and `gravity`; a turn
  count of 0 is the condition's mask bit clear. `wish_timer` (Wish's heal) and
  `trick_room` (the turn end's side-condition pass, :1823-1826) are not counted down yet.
- **moveLockedInto -> `locked_move`**, not `charge_turn`: it holds the move id a
  battler is locked into for its next turn (Bide, Rollout; charge-up moves are
  later, M6), and `charge_turn` is an `int8` flag that cannot hold a move id.
  `locked_move` is an `int16`, 0 = not locked (set only while
  `VOLATILE_CONDITION_MOVE_LOCKED`); `charge_turn` stays reserved.
- **storedDamage -> `bide_damage`** only while Biding (`bide_turns > 0`), else 0:
  the decomp's value outside a Bide is never read (`subscript_bide_start.s`
  zeroes it), and 0 lets equal states merge.
- **Audit-relabelled rows (Task 6 ledger).** Eight "not needed" judgements were
  refuted against the decomp and are `later (M6)`: the switch-in ability checks
  are re-run after every move, gated by `intimidateAnnounced`,
  `weatherAbilityAnnounced`, `traceAnnounced` and `downloadAnnounced`
  (`BattleSystem_TriggerEffectOnSwitch`, `src/battle/battle_lib.c:3649-4168`,
  called from `src/battle/battle_controller_player.c:3673` and `:3973`), so a
  mid-battle ability gain (Trace, Skill Swap, Role Play) announces again;
  `MoveEffectsData.quickClaw` is read by `BtlCmd_TryStealItem`
  (`src/battle/battle_script.c:5114`: Thief and Covet fail against a battler
  whose Quick Claw activated this turn); `monSpeedValues`, the
  speeds computed when the turn was ordered, by `BtlCmd_CalcGyroBallPower`
  (`:7127`); and `moveSketched` by `BtlCmd_TrySketch` (`:4848`). M1 is
  unaffected: no M1 mechanic gains an ability mid-battle, Quick Claw's
  activation is `qc_roll` (below; the per-turn flag only shows its animation, a
  `step` event), and none of Thief, Covet, Gyro Ball or Sketch is in M1. M6's
  coverage work must add these fields before the mechanics that read them.
- **moveProtect -> `protect_count`** (K3b2 Task 6; docs/kaizo/k3b2-rules.md §2(a),
  §12): `BattleContext.moveProtect`, the move whose attack message the battler last
  showed (`UpdateMoveBuffers`, `src/battle/battle_controller_player.c:3936-3942`),
  is read only by `BtlCmd_TryProtection` (`src/battle/battle_script.c:5154-5158`,
  which zeroes `protectSuccessTurns` unless it is Protect, Detect or Endure) and by
  `AICmd_LoadProtectChain` (`src/battle/trainer_ai/trainer_ai.c:2546-2552`, which
  reads 0 in that case). The reset is folded into `protect_count`: at the end of
  each move action (`step._use_move`) the count goes to 0 unless the move that showed
  its attack message is one of `ids.PROTECT_CHAIN_MOVES`. A switch and a faint clear
  both (`battle_lib.c:2099-2102`, `:2120`, `:2179-2182`, `:2198`); a bag item and
  Pursuit's pre-switch hit (`BtlCmd_TryPursuit` ends the action without
  `UpdateMoveBuffers`, `battle_script.c:6931`) leave both alone.
- **statusVolatile ATTRACT -> `infatuated`** (K3b2 Task 7; docs/kaizo/k3b2-rules.md §4(c),
  §12): `VOLATILE_CONDITION_ATTRACT_0..3` hold `FlagIndex(source)` of the battler the
  holder is in love with (`BtlCmd_TryAttract`, `src/battle/battle_script.c:5738`). In
  singles the source is always the other side's active battler, so `infatuated[side]` is
  0/1. It is cleared with the holder's other volatiles (`step._VOLATILE`) and when the
  source leaves, by a switch or a faint (`BattleSystem_UpdateAfterSwitch`,
  `src/battle/battle_lib.c:2088-2092`; `BattleSystem_CleanupFaintedMon`, `:2168-2177`;
  `step._source_leaves`, which Roar's drag reuses). `SELF_TURN_FLAG_INFATUATED` (a per-action
  turn flag read only by Destiny Knot, an unsupported item) stays a local.
- **speedRand -> `qc_roll`** (K3b2 Task 8; docs/kaizo/k3b2-rules.md §5, §12):
  `speedRand[battler]` is one `RandNext()` per battler slot, drawn at battle start
  (`BattleContext_InitCounters`, `src/battle/battle_lib.c:2019-2025`) and in each turn's
  pre-move step, after the turn order is fixed (`src/battle/battle_controller_player.c:853-857`,
  after `CalcTurnOrder` :707-790). It is read only as `speedRand % (100 / param) == 0` for a
  `HOLD_EFFECT_SOMETIMES_PRIORITY` holder (`BattleSystem_CompareBattlerSpeed`,
  `battle_lib.c:1278-1286`, :1344-1352; Quick Claw, param 20, the only such item), so
  `qc_roll[side]` holds that bit. The turn's draw is `Outcome.quick_claw`: the end-of-turn
  order reads it the same turn (`SortMonSpeedOrder`, bcp:888), and `step` stores it for the
  next turn's order (and the AI's `IfSpeedCompare*`, trainer_ai.c:1102-1124, and
  `LoadBattlerSpeedRank`, trainer_ai.c:2081) only while a
  reader can see a holder before the redraw (`step.qc_kept`), else 0 (exact: the value
  is never read).
- **BattleSystem.terrain -> `terrain`** (K3b2 Task 9; docs/kaizo/k3b2-rules.md §9, §12):
  not one of the three structs above (`include/struct_defs/battle_system.h:112`), copied
  from the field's `FieldBattleDTO.terrain` once (`src/battle/battle_main.c:1052`) and
  never written in battle. `BattleSystem_GetTerrain` (`src/battle/battle_system.c:375-382`)
  is read by Nature Power, Secret Power and Camouflage (`src/battle/battle_script.c:8007`,
  `:8040`, `:8067`); its other readers are graphics and wild catches. `make_state` takes
  it from `Trainer.terrain` (`tools/terrain.py`); `TERRAIN_UNKNOWN` (-1) when the trainer's
  maps give several terrains or none, and coverage then refuses those three effects.
- **battleEndFlag -> `over`, `winner`**: `winner` is 0 none, 1 player lost (no
  player Pokemon has HP left; a double KO of both last Pokemon is a draw, played as a
  loss), 2 player won. A player faint alone only records a death.
- **recycleItem -> `item_consumed`**: the last item consumed, per party slot.
- **aiContext**: `battlerMoves` (moves the AI has seen) -> `ai_seen_moves`,
  `battlerAbilities` (revealed abilities) -> `ai_ability_known`, `trainerItems`
  -> `trainer_items`, `trainerItemCounts` (the bag's size at battle start, never
  decremented, read by `TrainerAI_ShouldUseItem`'s gate, trainer_ai.c:4092) ->
  `trainer_item_count`. Its scoring scratch is rebuilt every decision.
- **Engine inputs of the AI** (docs/ai/roark.md sections 5.3, 8.2, 11.1, 12):
  - `fakeOutTurnNumber` -> `fake_out_turn`: 0 for the leads (the battle-start
    send-out does not call `BattleSystem_UpdateAfterSwitch`), `totalTurns + 1` after
    every switch-in (battle_lib.c:2117: switch action, Pursuit-interrupted switch,
    faint replacement) and faint (`BattleSystem_CleanupFaintedMon`, :2195).
    `LoadIsFirstTurnInBattle` (trainer_ai.c:2487) is `fake_out_turn >= turn`; Fake Out
    (`effect_script_0158.s`) fails unless `fake_out_turn == turn` (`step._use_move`). `turn`
    is `totalTurns` during a turn, so `step` writes `turn + 1`; `step.replace` runs
    after `step` counted the turn but before the decomp's `totalTurns++`
    (battle_controller_player.c:1844-1852), so it writes `turn`.
  - `moveHit` -> `move_hit`: `UpdateFlagsWhenHit` (battle_controller_player.c:4499-4507)
    writes the move to its defender when the attack message was shown, `MOVE_NONE`
    when the move was disrupted before it (sleep, freeze, flinch, Taunt, full
    paralysis; a confusion self-hit makes the user the defender, :2629). It runs after
    a KO'd defender's faint (:3820-3837), so a fainted defender keeps it. The battler's
    own move end clears it (`ClearFlags`, :4389, via :3986; not when the battle just
    ended, nor for a pursuer, whose action `TryPursuit` ends, battle_script.c:6931),
    and so do a switch-in (battle_lib.c:2121) and a faint (:2199).
  - `ability_slot`: the species ability slot (from the personality, resolved into
    `PokemonSpec.ability_slot`); `tables.species_abilities_decomp[row, ability_slot]`,
    with `row = tables.species_form_row[species, form]`, is the decomp ability id, also
    for abilities the engine does not implement (`ability` is then `UNSUPPORTED`).
- **Not from a struct field.** `pursuit_target` and `wish_hp` are
  `BattleState`'s own bookkeeping: the decomp derives Pursuit's pre-switch hit
  from per-turn scratch (the opposing battler's selected move when a switch is
  chosen) and Wish's heal from the battlers' HP when it lands.

## BattleMon (`include/battle/battle_mon.h`)

| Struct.field | pytree field | status |
|---|---|---|
| `BattleMon.species` | `species` | mapped |
| `BattleMon.attack` | `stats` | mapped |
| `BattleMon.defense` | `stats` | mapped |
| `BattleMon.speed` | `stats` | mapped |
| `BattleMon.spAttack` | `stats` | mapped |
| `BattleMon.spDefense` | `stats` | mapped |
| `BattleMon.moves` | `moves` | mapped |
| `BattleMon.hpIV` | `ivs` | mapped |
| `BattleMon.attackIV` | `ivs` | mapped |
| `BattleMon.defenseIV` | `ivs` | mapped |
| `BattleMon.speedIV` | `ivs` | mapped |
| `BattleMon.spAttackIV` | `ivs` | mapped |
| `BattleMon.spDefenseIV` | `ivs` | mapped |
| `BattleMon.isEgg` | — | not needed because eggs never battle |
| `BattleMon.hasNickname` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.statBoosts` | `stages` | mapped |
| `BattleMon.weight` | — | not needed because it is constant per species in Gen 4; read from `tables.species_weight` |
| `BattleMon.type1` | `types` | mapped |
| `BattleMon.type2` | `types` | mapped |
| `BattleMon.formNum` | `form` | mapped |
| `BattleMon.isShiny` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.padding0026_06` | — | not needed because it is padding |
| `BattleMon.ability` | `ability`, `ability_slot` | mapped |
| `BattleMon.weatherAbilityAnnounced` | — | later (M6) |
| `BattleMon.intimidateAnnounced` | — | later (M6) |
| `BattleMon.traceAnnounced` | — | later (M6) |
| `BattleMon.downloadAnnounced` | — | later (M6) |
| `BattleMon.anticipationAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.forewarnAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.slowStartAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.slowStartFinished` | — | later (M6) |
| `BattleMon.friskAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.moldBreakerAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.pressureAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.padding0028_0B` | — | not needed because it is padding |
| `BattleMon.ppCur` | `pp` | mapped |
| `BattleMon.ppUps` | — | later (M3) |
| `BattleMon.level` | `level` | mapped |
| `BattleMon.friendship` | `friendship` | mapped |
| `BattleMon.nickname` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.curHP` | `hp` | mapped |
| `BattleMon.maxHP` | `max_hp` | mapped |
| `BattleMon.OTName` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.exp` | — | not needed because experience and level-ups do not change the fight being searched |
| `BattleMon.personality` | — | not needed because its battle uses (nature, ability slot, gender) are resolved into `PokemonSpec` before `make_state` |
| `BattleMon.status` | `status`, `status_counter` | mapped |
| `BattleMon.statusVolatile` | `confusion`, `flinched`, `trapped`, `recharge`, `foresight`, `bide_turns`, `curled`, `locked_move`, `infatuated` | mapped |
| `BattleMon.OTId` | — | not needed because only obedience and the traded-Pokemon exp bonus read it; the tool assumes an obedient party |
| `BattleMon.heldItem` | `item` | mapped |
| `BattleMon.padding007A` | — | not needed because it is padding |
| `BattleMon.timesDamaged` | — | not needed because no battle or AI logic reads it (only the generic BattleMon getter and setter) |
| `BattleMon.trainerMessageFlags` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.gender` | `gender` | mapped |
| `BattleMon.OTGender` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.capturedBall` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.moveEffectsMask` | `leech_seed`, `water_sport`, `mud_sport`, `aqua_ring` | mapped |
| `BattleMon.moveEffectsTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleMon.moveEffectsData` | `taunt`, `protect_count`, `rollout_count`, `recharge`, `substitute_hp`, `trap_move`, `choice_lock` | mapped |

## MoveEffectsData (nested in `BattleMon.moveEffectsData`)

| Struct.field | pytree field | status |
|---|---|---|
| `MoveEffectsData.disabledTurns` | `disable_turns` | mapped |
| `MoveEffectsData.encoredTurns` | — | later (M6) |
| `MoveEffectsData.chargedTurns` | — | later (M6) |
| `MoveEffectsData.tauntedTurns` | `taunt` | mapped |
| `MoveEffectsData.protectSuccessTurns` | `protect_count` | mapped |
| `MoveEffectsData.perishSongTurns` | — | later (M6) |
| `MoveEffectsData.rolloutCount` | `rollout_count` | mapped |
| `MoveEffectsData.furyCutterCount` | — | later (M6) |
| `MoveEffectsData.stockpileCount` | — | later (M6) |
| `MoveEffectsData.stockpileDefBoosts` | — | later (M6) |
| `MoveEffectsData.stockpileSpDefBoosts` | — | later (M6) |
| `MoveEffectsData.truant` | — | later (M6) |
| `MoveEffectsData.flashFire` | — | later (M6) |
| `MoveEffectsData.lockOnTarget` | — | later (M6) |
| `MoveEffectsData.mimickedMoveSlot` | — | later (M6) |
| `MoveEffectsData.bindTarget` | — | not needed because in singles the binder is always the opposing active Pokemon |
| `MoveEffectsData.meanLookTarget` | — | later (M6) |
| `MoveEffectsData.lastResortCount` | — | later (M6) |
| `MoveEffectsData.magnetRiseTurns` | — | later (M6) |
| `MoveEffectsData.healBlockTurns` | — | later (M6) |
| `MoveEffectsData.embargoTurns` | — | later (M6) |
| `MoveEffectsData.canUnburden` | — | later (M6) |
| `MoveEffectsData.metronomeTurns` | — | later (M6) |
| `MoveEffectsData.micleBerry` | — | later (M6) |
| `MoveEffectsData.custapBerry` | — | later (M6) |
| `MoveEffectsData.quickClaw` | — | later (M6) |
| `MoveEffectsData.meFirst` | — | later (M6) |
| `MoveEffectsData.padding0004_1F` | — | not needed because it is padding |
| `MoveEffectsData.rechargeTurnNumber` | `recharge` | mapped |
| `MoveEffectsData.fakeOutTurnNumber` | `fake_out_turn` | mapped |
| `MoveEffectsData.slowStartTurnNumber` | — | later (M6) |
| `MoveEffectsData.meFirstTurnNumber` | — | later (M6) |
| `MoveEffectsData.substituteHP` | `substitute_hp` | mapped |
| `MoveEffectsData.transformedPID` | — | later (M6) |
| `MoveEffectsData.disabledMove` | `disabled_move` | mapped |
| `MoveEffectsData.bindingMove` | `trap_move` | mapped |
| `MoveEffectsData.encoredMove` | — | later (M6) |
| `MoveEffectsData.encoredMoveSlot` | — | later (M6) |
| `MoveEffectsData.lastResortMoves` | — | later (M6) |
| `MoveEffectsData.choiceLockedMove` | `choice_lock` | mapped |
| `MoveEffectsData.transformedGender` | — | later (M6) |
| `MoveEffectsData.itemHPRecovery` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |

## SideConditions (`include/battle/battle_context.h`)

| Struct.field | pytree field | status |
|---|---|---|
| `SideConditions.reflectUser` | — | not needed because in singles the user is always that side's active Pokemon (the field names the battler for messages and doubles) |
| `SideConditions.reflectTurns` | `reflect` | mapped |
| `SideConditions.lightScreenUser` | — | not needed because in singles the user is always that side's active Pokemon (the field names the battler for messages and doubles) |
| `SideConditions.lightScreenTurns` | `light_screen` | mapped |
| `SideConditions.mistUser` | — | not needed because in singles the user is always that side's active Pokemon (the field names the battler for messages and doubles) |
| `SideConditions.mistTurns` | `mist` | mapped |
| `SideConditions.safeguardUser` | — | not needed because in singles the user is always that side's active Pokemon (the field names the battler for messages and doubles) |
| `SideConditions.safeguardTurns` | `safeguard` | mapped |
| `SideConditions.followMe` | — | not needed because Follow Me only redirects moves in doubles |
| `SideConditions.followMeUser` | — | not needed because Follow Me only redirects moves in doubles |
| `SideConditions.knockedOffItemsMask` | — | later (M6) |
| `SideConditions.padding00_1D` | — | not needed because it is padding |
| `SideConditions.spikesLayers` | `spikes` | mapped |
| `SideConditions.toxicSpikesLayers` | `toxic_spikes` | mapped |
| `SideConditions.padding04_04` | — | not needed because it is padding |

## FieldConditions (embedded in `BattleContext.fieldConditions`)

| Struct.field | pytree field | status |
|---|---|---|
| `FieldConditions.weatherTurns` | `weather_timer` | mapped |
| `FieldConditions.futureSightTurns` | `future_timer` | mapped |
| `FieldConditions.wishTurns` | `wish_timer` | mapped |
| `FieldConditions.futureSightMove` | — | later (M6) |
| `FieldConditions.futureSightAttacker` | — | later (M6) |
| `FieldConditions.futureSightDamage` | `future_damage` | mapped |
| `FieldConditions.wishTarget` | — | not needed because in singles Wish always lands on the user's side's active slot |

## BattleContext (`include/battle/battle_context.h`)

| Struct.field | pytree field | status |
|---|---|---|
| `BattleContext.curCommandState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.nextCommandState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.command` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.commandNext` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.fieldConditionCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.fieldConditionCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.monConditionCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.monConditionCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.sideConditionCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.sideConditionCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.turnStartCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.turnStartCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveHitCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveHitCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveMessageType` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveMessageState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveEffectState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.afterMoveEffectTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.beforeMoveCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.tryMoveCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.statusCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.abilityCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.switchInCheckState` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.vanishedCheckTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.padding0060` | — | not needed because it is padding |
| `BattleContext.attacker` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.attackerTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.defender` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.defenderTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.faintedMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.switchedMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.switchedMonTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.abilityMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.magicCoatMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectType` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectParam` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.lastBattlerId` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.gainedExp` | — | not needed because experience and level-ups do not change the fight being searched |
| `BattleContext.sharedExp` | — | not needed because experience and level-ups do not change the fight being searched |
| `BattleContext.sideGetExpMask` | — | not needed because experience and level-ups do not change the fight being searched |
| `BattleContext.scriptNarc` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptFile` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackPointer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackNarc` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackFile` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.turnOrderCounter` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.waitCounter` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgBuffer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgBattlerTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgAttacker` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgDefender` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgMoveTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgItemTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgAbilityTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.msgTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.calcTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battlerStatusFlags` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.payDayCount` | — | not needed because prize money does not affect the battle |
| `BattleContext.totalTurns` | `turn` | mapped |
| `BattleContext.totalFainted` | `player_faints` | mapped |
| `BattleContext.totalDamage` | — | not needed because only the post-battle records read it |
| `BattleContext.meFirstTurnOrder` | — | later (M6) |
| `BattleContext.taskData` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.tmpData` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.fieldConditionsMask` | `weather`, `trick_room`, `gravity` | mapped |
| `BattleContext.fieldConditions` | `weather_timer`, `future_timer`, `future_damage`, `wish_timer` | mapped |
| `BattleContext.sideConditionsMask` | `stealth_rock`, `tailwind`, `lucky_chant` | mapped |
| `BattleContext.sideConditions` | `reflect`, `light_screen`, `mist`, `safeguard`, `spikes`, `toxic_spikes` | mapped |
| `BattleContext.turnFlags` | — | not needed because it is reset every turn and `step` resolves a whole turn (`protecting`: the first actor's `protect_ok` event, read by the second actor in `step._execute_actions`) |
| `BattleContext.selfTurnFlags` | — | not needed because it is reset at every action's end (`BattleContext_Init`, bcp.c:3998) and `step` resolves a whole action; `focusItemActivated` and `SELF_TURN_FLAG_PLUCK_BERRY` live in `_use_move`'s hit loop (`effects.MoveHit`) |
| `BattleContext.moveFailFlags` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.aiContext` | `ai_seen_moves`, `ai_ability_known`, `trainer_items`, `trainer_item_count` | mapped |
| `BattleContext.aiScriptTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.aiScriptCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleStatusMask` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.battleStatusMask2` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.damage` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.hitDamage` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.criticalBoosts` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.criticalMul` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.movePower` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.powerMul` | `replace_mul` | mapped |
| `BattleContext.hpCalcTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.moveType` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.moveEffectChance` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.prizeMoneyMul` | — | not needed because prize money does not affect the battle |
| `BattleContext.moveStatusFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectDirectFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectIndirectFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectAbilityFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.multiHitCounter` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.multiHitNumHits` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.battlerCounter` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.beatUpCounter` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.multiHitLoop` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.multiHitCheckFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.multiHitAccuracyCheck` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.clearVolatileStatus` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.selectedPartySlot` | — | not needed because the switch target comes from the action and is written to `active` inside `step` |
| `BattleContext.switchedPartySlot` | — | not needed because the switch target comes from the action and is written to `active` inside `step` |
| `BattleContext.aiSwitchedPartySlot` | — | not needed because the switch target comes from the action and is written to `active` inside `step` |
| `BattleContext.battlerActions` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.battlerActionOrder` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.monSpeedOrder` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.monSpeedValues` | — | later (M6) |
| `BattleContext.ioQueue` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.ioBuffer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleScript` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleMons` | `active` | mapped |
| `BattleContext.moveTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.moveCur` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.movePrev` | — | later (M6) |
| `BattleContext.moveLockedInto` | `locked_move` | mapped |
| `BattleContext.moveProtect` | `protect_count` | mapped |
| `BattleContext.moveHit` | `move_hit` | mapped |
| `BattleContext.moveHitBattler` | — | not needed because in singles it is the opposing battler whenever `move_hit` is set: both are written together (battle_controller_player.c:4500-4507), and `ClearFlags` (:4389) clears only `moveHit` |
| `BattleContext.moveHitType` | — | later (M6) |
| `BattleContext.movePrevByBattler` | `last_move` | mapped |
| `BattleContext.moveCopied` | — | later (M6) |
| `BattleContext.moveCopiedHit` | — | later (M6) |
| `BattleContext.moveSketched` | — | later (M6) |
| `BattleContext.moveSelected` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.moveSlot` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.conversion2Move` | — | later (M6) |
| `BattleContext.conversion2Battler` | — | later (M6) |
| `BattleContext.conversion2Type` | — | later (M6) |
| `BattleContext.metronomeMove` | — | later (M6) |
| `BattleContext.storedDamage` | `bide_damage` | mapped |
| `BattleContext.lastHitByBattler` | — | later (M6) |
| `BattleContext.battlerSpeedTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.battlersSwitchingMask` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.levelUpMons` | — | not needed because experience and level-ups do not change the fight being searched |
| `BattleContext.padding310A` | — | not needed because it is padding |
| `BattleContext.speedRand` | `qc_roll` | mapped |
| `BattleContext.flingTemp` | — | later (M6) |
| `BattleContext.flingScript` | — | later (M6) |
| `BattleContext.safariCatchStage` | — | not needed because trainer battles have no Safari Zone |
| `BattleContext.safariEscapeCount` | — | not needed because trainer battles have no Safari Zone |
| `BattleContext.runAttempts` | — | not needed because the player cannot flee a trainer battle |
| `BattleContext.battleEndFlag` | `over`, `winner` | mapped |
| `BattleContext.magnitude` | — | later (M6) |
| `BattleContext.fieldWeatherChecked` | — | not needed because `make_state` applies the start weather directly |
| `BattleContext.hpTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.recycleItem` | `item_consumed` | mapped |
| `BattleContext.partyOrder` | — | not needed because BattleState keeps party slots in a fixed order and switching only changes `active` |
| `BattleContext.expJinglePlayed` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleContext.linkBattleTimeout` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.recordedCommandFlags` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.waitingBattlers` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleProgressFlag` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.padding3154_01` | — | not needed because it is padding |

## BattleSystem (`include/struct_defs/battle_system.h`, the fields the engine reads)

| Struct.field | pytree field | status |
|---|---|---|
| `BattleSystem.terrain` | `terrain` | mapped |
