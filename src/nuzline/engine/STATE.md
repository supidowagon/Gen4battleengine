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
  slot's persistent data in `[P,6]` arrays and the active Pokemon's volatile
  data in `[B]` arrays indexed by battler; `active[b]` is the battler's party slot.
  `B` is the battler count and `P` the party count of the fight's `Format`
  (`format.py`: singles 2 and 2; doubles 4 and 2; two trainers 4 and 3; multi 4 and 4);
  battler `b` has side `b & 1`, partner `b ^ 2` and party `party(fmt, b)`. Four shape
  keys (`state._MON` `[P,6]`, `_BATTLER` `[B]`, `_SIDE` `[2]`, `_PARTY` `[P]`) place every
  field, and `state.layout(fmt)` gives the shapes. Per side `[2]`: `spikes`, `toxic_spikes`,
  `stealth_rock`, `reflect`, `light_screen`, `safeguard`, `mist`, `lucky_chant`, `tailwind`,
  `future_sight` (`SideConditions`). Per party `[P]`: `ts_alive`. Every other `[2]` field of
  the singles state is per battler `[B]`, `trainer_items` and `trainer_item_count` included
  (the bag `trainerItems[battler >> 1]` sits on the opposing battler, ruling D1-5), `stages`,
  `ts_taken` and `entry_pending` included. The player-only fields (`ai_*`, the exp block,
  `deaths`, `protected`) stay on party 0. A field that names another battler stores
  `1 + battler` (docs/doubles-state-design.md section 5); `infatuated` is still 0/1 (D5).
  Singles shapes are unchanged.
- **Stats.** `attack`/`defense`/`speed`/`spAttack`/`spDefense` become
  `stats[..., 0..4]` in Atk, Def, SpA, SpD, Spe order (`ids.Stat` minus one);
  HP is `max_hp`. IVs become `ivs` in `ids.Stat` order (HP, Atk, Def, SpA, SpD, Spe).
- **Stages.** `statBoosts[8]` is HP, Atk, Def, Spe, SpA, SpD, Acc, Eva with 6
  neutral; `stages[side, ids.Stat]` is HP, Atk, Def, SpA, SpD, Spe, Acc, Eva with
  0 neutral (index 0 unused).
- **Status.** The decomp packs sleep turns and the toxic counter into `status`
  bits (`include/constants/battle/condition.h`); `BattleState` splits them into
  `status` (`ids.Status`) and `status_counter`.
- **Pending confusion.** `confusion == ids.CONFUSION_PENDING` (6): a disliked flavor berry
  confused the battler (subscript_confuse.s:34's `Random 3, 2` not drawn yet); its next
  confusion check draws the counter (`Outcome.berry_confusion`), ruling G-5 in
  docs/kaizo/e1-mechanics-ledger.md. Every other read takes it as confused.
- **statusVolatile** bits mapped: confusion turns -> `confusion`, flinch ->
  `flinched`, bind turns -> `trapped`, recharging -> `recharge`, Foresight ->
  `foresight`, Bide turns -> `bide_turns`, Defense Curl -> `curled`,
  `VOLATILE_CONDITION_MOVE_LOCKED` -> `locked_move` (with `moveLockedInto`), the
  ATTRACT bits -> `infatuated` (below), Destiny Bond -> `destiny_bond` (E2 batch 4:
  per battler, set by subscript_destiny_bond.s:9, cleared at the holder's next
  `CHECK_STATUS_START`, bcp.c:2425, and with the volatiles), the Uproar count ->
  `uproar` (E2 batch 5, per battler: `Random 3, 3` turns from subscript_uproar.s:6-8, counted
  down by the end-of-turn check, bcp.c:1505-1541, cleared with the volatiles), Mean Look ->
  `mean_look` (E2 batch 7, per battler, with `MoveEffectsData.meanLookTarget` ->
  `mean_look_target` as `1 + raw`, the attacker ORed in by subscript_mean_look.s:10 and never
  cleared but with the trapped battler's volatiles, so a stale value stays (B7-15); `mean_look`
  ends when the trapper leaves by a switch that is not a Baton Pass, battle_lib.c:2055-2061, or
  faints, :2170-2172; Baton Pass keeps both, :2078, :2111), Curse ->
  `curse` (below), Focus Energy -> `focus_energy` (E2 batch 12, B12-4: per battler, set by
  subscript_focus_energy.s:6, ORed in by Psych Up, battle_script.c:6050, cleared by Haze, :3544,
  and with the volatiles; Baton Pass keeps it, condition.h:72). The other bits (Thrash,
  Transform, Rage, Substitute flag) are later (M6); the Substitute flag is
  implied by `substitute_hp > 0`. Torment -> `torment` (E3 batch 1, E3-2-1: per battler `[B]`,
  `VOLATILE_CONDITION_TORMENT` bit 31, set by subscript_torment_start.s:8 with no count, cleared
  with the volatiles; Baton Pass drops it, condition.h:72-76, battle_lib.c:2075; it only blocks
  the slot holding `last_move` at the next move selection, `actions.usable_moves`).
  Nightmare -> `nightmare` (E1 batch G, G-31..G-33: per battler `[B]`,
  `VOLATILE_CONDITION_NIGHTMARE` bit 27, condition.h:59, set on a sleeping defender by
  subscript_nightmare_start.s:9; the turn end, after the burn and before the curse, costs a
  sleeper `Divide(maxHP, 4)` unless Magic Guard and clears it on a battler not asleep,
  bcp.c:1436-1448; every wake-up, Chesto / Lum, Shed Skin, Hydration, Insomnia / Vital Spirit,
  Heal Bell / Aromatherapy and a bag sleep cure clear it (`effects.clear_nightmare`); cleared
  with the volatiles, not Baton-passed, condition.h:72-76, not copied by Transform).
- **moveEffectsMask** bits mapped: Leech Seed -> `leech_seed`, Leech Seed recipient -> `leech_seed_recipient` (`1 + battler`, the battler that used the move, MOVE_EFFECT_LEECH_SEED_RECIPIENT, moves.h:8-10; D3-11b), Water Sport ->
  `water_sport`, Mud Sport -> `mud_sport`, Aqua Ring -> `aqua_ring` (per battler, cleared on switch-out; the
  Electric power halving reads every active battler, `battle_lib.c:6812`), Ingrain -> `ingrain`,
  Imprison -> `imprison` (its user) and Imprisoned -> `imprisoned` (its foes) (E2 batch 7, per
  battler, cleared with the volatiles; Baton Pass keeps Ingrain only, moves.h:46-48), Perish
  Song, Yawn and Lock-On (below). Camouflage -> `camouflage` (E1 batch G, G-20: per battler, set
  by a Camouflage that changed the types, effect_script_0213.s:6; not copied by Transform, which
  copies the BattleMon up to `ability`, battle_script.c:5346-5348; cleared by a switch or a faint,
  battle_lib.c:2071, :2166, and not Baton-passed, moves.h:46-48).
  Roar's Ingrain test compares the whole mask (`effects.ingrain_anchors`, the kept bits in
  `effects.MOVE_EFFECT_MASK_FIELDS`). The semi-invulnerable bits (AIRBORNE, UNDERGROUND,
  UNDERWATER, SHADOW_FORCE) -> `semi_invulnerable`, one `ids.SemiKind` (E2 batch 10, B10-12).
  Power Trick -> `power_trick` (E2 batch 12, B12-10: per battler, toggled by
  subscript_user_swap_atk_and_def.s:5; the BattleMon Attack and Defense it swaps are read
  through `damage.battle_stats`; Baton Pass keeps it, moves.h:47, and swaps the newcomer's,
  battle_lib.c:2137-2141); a level-up's BattleSystem_ReloadPokemon, battle_lib.c:169-174, undoes
  the swap and keeps the bit: `power_trick_desync`). Magnet Rise -> `magnet_rise` (the
  `MOVE_EFFECT_MAGNET_RISE` bit) and `magnet_rise_turns` (`MoveEffectsData.magnetRiseTurns`) (E1
  trial batch, G-66: set by effect_script_0252.s, counted down after Taunt, kept by Baton Pass; the
  bit stays when Gravity zeroes the turns). Mimic -> `mimic_slots`, `shadow_moves`, `shadow_pp` (E1
  trial batch, G-68: the BattleMon / party split of the move and PP arrays, below). Embargo ->
  `embargo` (E2 batch 13, B13-29: embargoTurns with its bit, set to 5 by
  subscript_embargo_start.s:13-14, counted down at the turn end, battle_controller_player.c:1671-1677,
  kept by Baton Pass, moves.h:46-48, battle_lib.c:2112; `state.held_item` reads it). Gastro
  Acid's ABILITY_SUPPRESSED (moves.h:29) -> `ability_suppressed` (E3-2-6a: per battler `[B]`,
  0/1, cleared on switch-out, battle_lib.c:2071, kept by Baton Pass, moves.h:46-48, :2076; read
  through `state.battler_ability` / `damage.battler_decomp_ability`, `Battler_Ability`,
  battle_lib.c:3088-3093, and by the AI's LoadBattlerAbility / CheckBattlerAbility,
  trainer_ai.c:1177-1178, :1221-1222). Gastro Acid (E3-2-6, `effects._gastro_acid`) sets it
  (subscript_suppress_target_ability.s:10); it is in `effects.MOVE_EFFECT_MASK_FIELDS`, so a
  suppressed Ingrain user no longer anchors against Roar (the whole-mask compare, :7). Charge ->
  `charged_turns` (E3-B2-8: `MoveEffectsData.chargedTurns`, 2 bits, with its `MOVE_EFFECT_CHARGE`
  bit, set to 2 by subscript_charge.s:5-6, counted down at the turn end, bcp.c:1626-1630; cleared
  with the volatiles, not Baton-passed, moves.h:46-48; in `effects.MOVE_EFFECT_MASK_FIELDS`;
  doubles the user's Electric power, battle_lib.c:6690). Grudge -> `grudge` (E3-B2-9: the
  `MOVE_EFFECT_GRUDGE` bit, set by effect_script_0194.s:6, cleared at the user's next move attempt,
  bcp.c:2427, and with the volatiles; not Baton-passed, moves.h:46-48; in
  `effects.MOVE_EFFECT_MASK_FIELDS`; `effects.try_grudge` zeroes a KO-ing attacker's selected slot's
  PP, battle_script.c:6743-6763). Miracle Eye -> `miracle_eye` (E3-B2-10: the `MOVE_EFFECT_MIRACLE_EYE`
  bit, moves.h:30, set on the defender by subscript_miracle_eye.s:5 with no failure and no
  Substitute test; cleared with the volatiles, not Baton-passed, moves.h:46-48; in
  `effects.MOVE_EFFECT_MASK_FIELDS`; read as `MON_IS_IDENTIFIED`, common.h:48-49: a positive evasion
  stage counts as 0, bcp.c:2888-2890, and `BasicTypeMulApplies` skips its Dark immunity rows,
  battle_lib.c:2551-2555). Last Resort -> `last_resort_count` (3 bits, battle_mon.h:27) and
  `last_resort_moves` (`[B, 4]` move ids, battle_mon.h:50) (E3-B2-11: both per battler,
  written by `step._use_move` at BeforeMove's tail through `effects.last_resort_update`, battle_lib.c:
  3443-3459 / bcp.c:3218; zeroed with the volatiles at a switch or faint, battle_lib.c:2100-2104,
  and not Baton-passed, :2107-2115; zeroed by Transform, battle_script.c:5340, and a Mimic that
  copies Last Resort, :4514-4516; `effects.last_resort_fails` is BtlCmd_TryLastResort,
  battle_script.c:7567-7579; the Sketch zeroing, :4876-4878, waits for Sketch). The rest (Heal
  Block) is later (M6).
- **fieldConditionsMask -> `field_mask`** (int32, `_FIELD`, one per battle: BattleContext state,
  doubles-state-design section 5 convention 2): `BattleContext.fieldConditionsMask`
  (battle_context.h:210) bit for bit, bits condition.h:110-149 (`ids.FieldCondition`): the
  weather bits 0-7 (TEMP / PERM of rain, sand, sun, hail) and deep fog 15, the Uproar bits 8-11
  (bit `8 + b` of battler `b`), the Gravity counter 12-14 and the Trick Room counter 16-18.
  **`weather_turns`** (int32, `_FIELD`): `FieldConditions.weatherTurns` (battle_context.h:29);
  stale outside a TEMP-only weather (only the weather moves, effect_script_0115/0136/0137/0164.s
  :10-14, and the TEMP-only countdown, bcp.c:1121, :1146, :1171, :1196, write it). FCM (FCM.md
  section 4, steps 1-3): every writer and reader goes through `engine.field` (the reference the
  same), so a mask with several groups or a borrowed counter (FCM 3.1, Acid Rain) reads as the
  decomp reads it (FCM 3.2). `uproar[b]` is the battler's VOLATILE_CONDITION_UPROAR count only
  (the lock, its end check, Metronome): the field's Uproar readers read bits 8-11
  (`field.uproar_any`). The old `weather`, `weather_timer`, `gravity` and `trick_room` fields
  are gone (step 3). Defog clears the deep fog bit 15 alone (subscript_defog.s:64-65,
  `effects._defog`, E3-B2-14).
- **sideConditionsMask**: Stealth Rock -> `stealth_rock`, Tailwind turns ->
  `tailwind`, Lucky Chant turns -> `lucky_chant` (the 3-bit counter at bits 12-14, set to 5 by
  `effects._lucky_chant`, subscript_lucky_chant_start.s:5-6, E3-B2-7), Future Sight -> `future_sight` (E2 batch 5:
  per side; set by `BtlCmd_TryFutureSight`, battle_script.c:6070-6072, cleared only when the hit
  fires on a battler with HP, bcp.c:1767-1770, so it outlives a count that ran out on a fainted
  battler; the AI reads it). Its Reflect, Light Screen, Spikes, Safeguard, Mist, Toxic Spikes
  and Wish bits are implied by the nonzero turn/layer counts in the mapped fields. Defog clears
  seven of them on its defender's side (`effects._defog`, subscript_defog.s:21-61, E3-B2-14).
- **FieldConditions.futureSight\*** (E2 batch 5): per battler (`[MAX_BATTLERS]` arrays, the
  battler the move aims at; in singles the side's slot): `future_timer`, `future_damage` (the
  HP the hit takes: the decomp stores it negative), `future_move` and `future_attacker` (the
  attacker's battler as `1 + battler`, 0 = none; convention 3). Field state: they stay when the
  battler leaves, and the hit lands on whatever battler holds the slot (`step._future_sight`).
- **The field-condition pass** (`step.field_conditions`,
  `BattleControllerPlayer_CheckFieldConditions`, battle_controller_player.c:932-1254)
  counts down `reflect`, `light_screen`, `mist`, `safeguard`, `tailwind`, `lucky_chant`,
  `weather_turns` once per TEMP-only weather group (the group's TEMP bit clears at 0,
  `field.weather_state`) and the Gravity bits (`field.gravity_tick`); a turn count of 0 is the
  condition's mask bit clear. `wish_timer` (Wish's heal) is not counted down yet. The Trick
  Room bits are counted down by the turn end's side-condition pass (`step._trick_room_check`,
  `field.trick_room_tick`, bcp.c:1822-1830, E3-2-4; Kaizo re-arms it, `Tables.trick_room_rearm`).
- **movePrev -> `move_prev`** (E2 batch 5): battle-wide, the last move used in the battle:
  `UpdateMoveBuffers` / `UpdateFlagsWhenHit` (bcp.c:3936-3942, :4499-4508) write moveTemp when
  the attack message showed, else 0, on every acting path (`step._use_move`); Copycat reads it
  (`order.copycat_call`). A field condition, not a battler's: it does not leave with a battler.
- **moveCopied / moveCopiedHit -> `move_copied` / `move_copied_hit`** (E3-B2-2): per battler `[B]`
  (the holder) and `[B,B]` (holder, then attacking battler), int16 move ids, 0 = none. Mirror Move
  (`order.mirror_call`) reads `move_copied[b]`, else in a doubles battle a random nonzero
  `move_copied_hit[b]` entry (battle_script.c:3483-3503). `UpdateFlagsWhenHit` writes both for the
  defender when moveTemp has `MoveFlag.MIRROR_MOVE` and the attack message showed
  (battle_controller_player.c:4480-4486; `step._use_move` computes the write, `step._move_end`
  applies it after the faint check like `move_hit`). A battler's switch-in or faint clears its own
  entry and row (`VOLATILE_FIELDS`) and, through `state.clear_copied`, every foe's `move_copied`
  and every holder's column of it (battle_lib.c:2125-2149, :2203-2220; `step._reset_battler`).
  Nothing counts them down; Baton Pass clears them too.
- **turnFlags.magicCoat -> `magic_coat`** (E3-B2-6): per battler `[B]`, int8, `1 + moveSlot` of the
  Magic Coat user (0 = down). `TrySetMagicCoat` (battle_script.c:6459-6478) sets it
  (`effects._magic_coat`; it fails when the user acts last, `waitingBattlers == 1`); the first
  reflectable move aimed at the holder clears it (bcp.c:3071, `step._use_move`) and runs as the
  holder's own move on the mover in a second pass (`step._execute_actions`' `_REFLECT` phase,
  `MODE_REFLECT`). It is a turn flag: cleared at the holder's faint (`VOLATILE_FIELDS`) and at the
  end of the turn's actions (`SetupNextTurn`, battle_lib.c:2231); kept at a pivot's pause.
- **conversion2Move / conversion2Battler / conversion2Type -> `conv2_move` / `conv2_battler` /
  `conv2_type`** (E3-B2-3): per battler `[B]` (the defender), int16 move id (0 = none), int8
  `1 + attacking battler` (0 = none) and int8 type id. `UpdateFlagsWhenHit` writes them for the
  defender of every move (battle_controller_player.c:4509-4533): moveCur (the called move), the
  attacker and `CalcCurrentMoveType` where the move succeeded without `MOVE_STATUS_NO_EFFECTS` and
  its range is not user / user side / field / opponent side / ally / user-or-ally, else clears all
  three (`step._use_move` computes the write, `step._move_end` applies it after the faint check like
  `move_hit`). `ClearFlags` clears the actor's own `conv2_move` at its MoveEnd (:4390). A battler's
  switch-in or faint clears its own three (battle_lib.c:2131-2133, :2209-2211: `VOLATILE_FIELDS`).
  Conversion 2 (`effects._conversion_2`) reads them.
- **moveLockedInto -> `locked_move`**, not `charge_turn`: it holds the move id a
  battler is locked into for its next turn (Bide, Rollout, Uproar, and since E2 batch 10 the
  charge-up moves: subscript_vanish_on_charge_turn.s:10 locks into moveCur, B10-12), and
  `charge_turn` is an `int8` flag that cannot hold a move id.
  `locked_move` is an `int16`, 0 = not locked: it is set while `Battler_CanPickCommand` is
  FALSE (battle_lib.c:6224-6235) through `VOLATILE_CONDITION_MOVE_LOCKED` or an uproar (E2
  batch 5: `Battler_UnlockMoveChoice` clears only the first, battle_lib.c:1834-1841, so an
  uproaring battler stays locked, ruling B5-5); `charge_turn` stays reserved. A lock on a
  move the battler does not know (a call's) is still a lock (`actions.locked_slot`).
- **storedDamage -> `bide_damage`** only while Biding (`bide_turns > 0`), else 0:
  the decomp's value outside a Bide is never read (`subscript_bide_start.s`
  zeroes it), and 0 lets equal states merge.
- **Audit-relabelled rows (Task 6 ledger).** Eight "not needed" judgements were
  refuted against the decomp and are `later (M6)`: the switch-in ability checks
  are re-run after every move, gated by `intimidateAnnounced`,
  `weatherAbilityAnnounced`, `traceAnnounced` and `downloadAnnounced`
  (`BattleSystem_TriggerEffectOnSwitch`, `src/battle/battle_lib.c:3649-4168`,
  called from `src/battle/battle_controller_player.c:3673` and `:3973`), so a
  mid-battle ability gain (Trace, Skill Swap, Role Play) announces again (these
  four are mapped since E3-2-5, `hooks.move_end_switch_in`);
  `MoveEffectsData.quickClaw` is read by `BtlCmd_TryStealItem`
  (`src/battle/battle_script.c:5114`: Thief and Covet fail against a battler
  whose Quick Claw activated this turn); `monSpeedValues` (mapped since E3-B2-15 to
  `speed_rec`, below), by `BtlCmd_CalcGyroBallPower`
  (`:7127`); and `moveSketched` by `BtlCmd_TrySketch` (`:4848`). M1 is
  unaffected: no M1 mechanic gains an ability mid-battle, Quick Claw's
  activation is `qc_roll` (below; the flag `claw_flag` carries its animation and the
  Knock Off / Thief / Covet failures into the next turn when no BeforeMove cleared it,
  SF2 (k)), and none of Thief, Covet, Gyro Ball or Sketch is in M1. M6's
  coverage work must add these fields before the mechanics that read them.
- **Trace, the weather abilities and Truant (E2 batch 3).** `BattleMon.ability` is per
  battler: Trace (`hooks.trace`, battle_lib.c:3718-3737) writes the active battler's
  `ability`, `step._restore_ability` puts the party's back when it leaves (as
  `BattleSystem_InitBattleMon` re-reads it, battle_lib.c:124), and `damage.decomp_abilities`
  gives a Trace holder that traced the copied ability's decomp id (`ids.ABILITY_DECOMP_IDS`). Role Play (E3-B2-5) writes the user's `ability` = the target's the same way (and its `truant`; no announced flag, so the move-end switch-in pass acts for the copied entry ability); Worry Seed (E3-2-2) writes `ability` = Insomnia the same way; `decomp_abilities` now reads the written ability whenever `ability` differs from the personal row's.
  `traceAnnounced`, `weatherAbilityAnnounced`, `intimidateAnnounced` and `downloadAnnounced` are
  the per-battler fields `trace_announced`, `weather_announced`, `intimidate_announced` and
  `download_announced` since E3-2-5 (Skill Swap gives a battler an ability mid-battle): cleared
  by a switch (`VOLATILE_FIELDS`, `BattleSystem_InitBattleMon`, battle_lib.c:103-106) and by
  Transform (battle_script.c:5350-5362), set by the hook that acts on a living holder
  (`hooks.trace`, `weather_ability`, `intimidate`, `download`), and read by
  `hooks.move_end_switch_in`, MoveEnd's `BattleSystem_TriggerEffectOnSwitch` pass over every
  battler (battle_controller_player.c:3973), which fires an entry ability whose flag is still
  clear. The entry hooks (`on_switch_in`, `show_battle_mon`) keep their explicit conditions
  (newcomer, traced battler) but read `trace_announced` for a Trace holder. The one newcomer that
  meets no living foe (the player's at a double KO) still has Intimidate and Download fired by
  `search.playout.replacement_pass` once the opposing newcomer is in (G-22). Anticipation,
  Forewarn and Slow Start have no ability id in the engine (coverage refuses their holders), so
  they get no flag, and Skill Swap cannot hand a battler Slow Start (E3-2-5). `MoveEffectsData.truant` is
  `(totalTurns + 1) & 1` at the same writes as `fakeOutTurnNumber` (battle_lib.c:2117-2119,
  :2195-2197; 0 for a lead) and `totalTurns & 1` after a Transform (battle_script.c:5359): the
  field `truant` (`order.truant_loafs`; E1 trial batch, G-69).
  **The Trace invariant** (e2-tracelaws): ShowBattleMon runs SWITCH_IN_CHECK_STATE_TRACE before
  every turn's commands (bcp.c:256, from TurnEnd :1858 and TrainerMessage :240-249; the battle
  start runs it too, `hooks.battle_start`), so in every turn-start state no battler with
  `Battler_Ability` Trace, HP, no Griseous Orb and `trace_announced` 0 faces an opposing battler
  with HP whose raw `ability` ChooseTraceTarget accepts (not Forecast, Trace or Multitype,
  battle_lib.c:7750-7781): it would have traced (:3711-3737). `outcome.trace_sources` /
  `trace_copies` (the laws and the sampler) rely on it: an armed holder copies only what enters
  or changes this turn (at B = 2: an opposing newcomer, or the Insomnia of its own Worry Seed;
  at B = 4 also any battler's changed raw ability and a drag's bench); the opposing actives as
  they stand count only for a battler that gains Trace mid-turn, a Trace newcomer (a traceable
  foe may switch out after it), in a state paused mid-turn (`learn_site`, `pivot_battler`:
  no turn start since the last firing point), and for a battler waiting for ShowBattleMon
  (`entry_pending`: a B = 4 start or `replace_round` state, never a B = 2 turn start) and every
  holder opposing one. `step` and the reference do not rely on it (they
  fire the hook where the decomp does). Generated states keep it: `tests/strategies.settle_traces` sets `trace_announced` 1 on
  a drawn holder that breaks it (no draw; a raw Trace with the flag set is reachable, by a later
  Skill Swap of Trace back, which leaves the flag). A hand-built state that breaks it is not a
  game state; the laws may under-read it.
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
- **Curse, Perish Song, Yawn, Lock-On -> `curse`, `perish_song`, `perish_song_turns`, `yawn`,
  `lock_on`, `lock_on_target`** (E1 batch 12): per battler, defined with the decomp's meaning
  so the trainer AI reads them (`Expert_Protect`, `script.s:3544-3586`), set by E2 batch 8's
  Curse (Ghost user), Perish Song, Yawn and Lock-On / Mind Reader (B8-1..B8-5). `curse` is `VOLATILE_CONDITION_CURSE`
  (`include/constants/battle/condition.h:60`), 0/1. `perish_song` is
  `MOVE_EFFECT_PERISH_SONG` (`include/constants/battle/moves.h:13`), 0/1, and
  `perish_song_turns` is `MoveEffectsData.perishSongTurns` (`include/battle/battle_mon.h:14`),
  a counter still live at 0 (`battle_controller_player.c:1801`). `yawn` is the two-bit
  `MOVE_EFFECT_YAWN` counter (`moves.h:19-20`, `:42`, `:53`; decremented at
  `battle_controller_player.c:1684-1687`), 0 = none. `lock_on` is the two-bit
  `MOVE_EFFECT_LOCK_ON` counter (`moves.h:11-12`, `:41`), 0 = none, on the battler locked
  onto (Lock-On sets it on its target, `subscript_lock_on.s`), and `lock_on_target` is that
  battler's `MoveEffectsData.lockOnTarget` (`battle_mon.h:23`), the battler that locked on,
  as `1 + battler`, 0 = none (doubles convention 3). A switch-out or faint clears the
  battler's own (volatiles); Baton Pass keeps Curse (`condition.h:72-76`), Lock-On and Perish
  Song (`moves.h:46-48`, `battle_lib.c:2109-2110`); the Lock-On of another battler that the
  leaving one locked onto is cleared by its switch, restarted by its Baton Pass
  (`battle_lib.c:2063-2085`) and kept through its faint until the slot's replacement comes in,
  whose UpdateAfterSwitch (`battle_script.c:2522`) clears it (B8-2, B8-7). Their turn-end checks:
  Curse (bcp.c:1451, with Burn), Lock-On (:1618, with Uproar / Disable), Yawn (:1684, with
  Taunt), Perish Song in the side-condition pass (:1791).
- **monSpeedOrder -> `war_swaps`** (E1 Gardenia batch, ruling G-26): inside a turn `step`
  resolves the order itself; between turns the switch-in states run over the order
  `BattleSystem_SortMonSpeedOrder` gave at `TrainerMessage` (battle_lib.c:3364-3382,
  battle_controller_player.c:251), and only the weather state's result depends on it (two
  different switch-in weather abilities entering together: the slower one's stays). Where
  that order hangs on a draw still to come (the battle-start `speedRand`, a speed tie's
  `RandNext() & 1`), `war_swaps` holds the swap bit of each draw (`hooks.order_swaps`) until
  `step.resolve_war` makes it; else 0. A B = 2 field: 0 and unread at B = 4 (D3-13: the B = 4 entry
  points take the sort's draws as inputs, `switch_in.show_battle_mon4(ties)`).
- **The announced flags while a list newcomer waits -> `entry_pending`** (E1 Gardenia review,
  ruling G-32): `traceAnnounced`, `weatherAbilityAnnounced`, `intimidateAnnounced` and
  `downloadAnnounced` are fields since E3-2-5, but one span needs their
  "all still clear" state: a newcomer a replacement's party list sent in
  (subscript_show_party_list.s:22-24) while the player's newcomer of the same list fainted on
  its hazards, so `TurnEnd` re-enters `ReplaceFainted` (battle_controller_player.c:1846-1858)
  before `ShowBattleMon` (:256) runs its switch-in states. `entry_pending[battler]` is 1 then
  (`step.replace(..., effects=False)`), until `search.playout.replacement_pass` runs both
  newcomers' states (`hooks.show_battle_mon`); else 0.
  At B = 4 `step.replace_round` (D3-12) sets it for every living newcomer of the round, and
  `initial_arrays` for every living lead of a four-battler start state, until
  `switch_in.show_battle_mon4` (D3-13; `step.sample_root` runs it at the start) clears it.
- **No empty-slot field** (battlersSwitchingMask, bcp.c:330-333, :4109-4111): `absent_now` (active
  at 0 HP and no `replacement_bench`) equals the mask because no slot is refilled in battle.
  Premise, a design rule: the player uses no bag item in battle (docs/design.md:19); the only
  in-battle party-HP raise is a bag revive (src/battle/battle_system.c:665-710). Trainers never
  revive: the AI's item tests need its own active's curHP (trainer_ai.c:4100-4101, :4111-4113),
  the AI partner uses none (trainer_ai.c:4069-4073), and no vanilla or Kaizo trainer bag holds
  item 28, 29 or 37. If design.md:19 changes, the engine needs the decomp's mask as a field:
  `absent` int8[B] `_BATTLER`, written by ReplaceFainted's no-replacement branch and at battle
  start, cleared by the send-in (script.c:2511-2512), plus an "active none" value
  (selectedPartySlot 6, bcp.c:4111) so a revived slot's party HP is not read as the masked
  battler's.
- **Learn prompts at B = 4** (D3-12, finding 8): no pin; the prompt pauses the turn (E3-B2-21,
  `learn_site`) and a pending FIGHT re-reads its slot on the resume (bcp.c:1889-1895), not one
  that is MOVE_END (acted, pursued, dragged; `step._execute_actions`' `reread`). A level-up of
  battler 2's active reloads battler 2 (`expBattler = 2`, battle_script.c:9716-9720, :9860-9864;
  `levelup._reloaders`).
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
- **Tie bits are never stored** (D3-3): `Outcome.order_tie` (`int8[TIE_PAIRS[B]]`, one `RandNext() & 1` per
  position pair of `CalcTurnOrder`, `battle_controller_player.c:773-786`) and `Outcome.sort_tie`
  (`int8[B + 1, TIE_PAIRS[B]]`, the later sorts' bits) are read only where the decomp's compare draws them
  (`order.action_order_reads`); no `BattleState` field or scratch keeps one (a paused action keeps its
  sort's result in `ts_order` / `ts_speed_order`, not the bits). `Outcome.resort_tie` (`int8[B, TIE_PAIRS[B]]`, D3-18) is
  the same for the re-sort a Trick Room's MoveEnd runs (row = the Trick Room's battler). All have size 0 at B = 2, where
  `speed_tie` / `end_speed_tie` are the bits.
- **monSpeedValues -> `speed_rec`** (E3-B2-15): `BattleContext.monSpeedValues[battler]` is the
  speed `BattleSystem_CompareBattlerSpeed` last computed for the battler
  (`src/battle/battle_lib.c:1372-1373`, `order.effective_speed`), read only by
  `BtlCmd_CalcGyroBallPower` (`battle_script.c:7123-7131`) and `TrainerAI_CalcDamage`'s Gyro Ball
  case (`trainer_ai.c:2995-3003`). Per battler (`[B]`), not a volatile: a replacement's slot keeps
  its predecessor's value, and nothing writes a battler without HP or when it has no live
  opponent (`:1204-1211`). Written by `order.record_speeds` at the sorts that can be read before
  the next one: after the battle start's switch-in effects (`battle_controller_player.c:263`,
  `hooks.battle_start`; a weather war the start leaves open is settled by `step.resolve_war`,
  which writes again at turn 0), before each action and after the last one of a turn
  (`BranchActions`, `:888`, `step._execute_actions`). The turn-order sort's own compares
  (`:773-786`, the Trick Room sort `:3990-3992`) are followed by one of those sorts before any
  reader and add no state. The trainer AI's compares (`IfSpeedCompareEqualTo`,
  `trainer_ai.c:1109`) write it inside one decision, which no later decision reads (the next
  turn's `BranchActions` comes first): `ai.commands.AiContext.written`, not state.
- **Starf Berry draw -> `starf_roll`** (E1 batch F, F-10): `BattleSystem_TriggerHeldItem`
  (`battle_lib.c:4864-4866`) and `BattleSystem_PluckBerry` (:5805-5810) draw `RandNext() % 5` until
  the stat is below +6. `Outcome.starf[side, k]` holds a draw as one value in [0, 60) (the stat
  is rank `value % eligible` among Attack, Defense, Speed, Sp. Atk, Sp. Def below +6: uniform,
  the project's `% n` convention): k = 0 the turn's, which `step._action_phase` copies into
  `starf_roll` so `hooks.held_item_check` and `hooks.pluck_berry` read it; k = 1 a faint
  replacement's, which `step_events` keeps in `starf_roll` at the turn's end where a replacement
  can send in a living Starf holder (`step.starf_kept`; `step.replace` reads it, then clears it),
  else 0, and 0 at a pause (states merge). Per battler, hidden. The battle start's checks run with
  `starf_roll` = `STARF_UNDRAWN`: a lead's berry is eaten and its stat waits in **`starf_open`**
  (per battler, no decomp field: the draw the decomp makes there, `battle_lib.c:5273-5294`) for
  the root's draw (`step.root_belief`, `step.sample_root`, `hooks.resolve_starf`).
  F-O3 (E1 batch G, G-27): each eat draws its own `RandNext() % 5` (lib:4864-4866, :5286-5288,
  Pluck :5808), so `starf_roll` is per battler `int8[B, 3]`, the turn's queue
  `Outcome.starf[:, ids.STARF_QUEUE]` (columns 0, 2, 3: first, second, third eat), and
  **`starf_taken`** (per battler, `int8[B]`, no decomp field: the count of draws the battler's
  eats made this turn) picks the entry an eat reads, `min(starf_taken, 2)` (`hooks.starf_draw`;
  `hooks._pinch_effects` adds one per draw, an open battle-start berry takes entry 0). Both
  0 between turns except a kept replacement draw in entry 0; `step.replace` sets `starf_taken` 0.
- **BattleSystem.terrain -> `terrain`** (K3b2 Task 9; docs/kaizo/k3b2-rules.md §9, §12):
  not one of the three structs above (`include/struct_defs/battle_system.h:112`), copied
  from the field's `FieldBattleDTO.terrain` once (`src/battle/battle_main.c:1052`) and
  never written in battle. `BattleSystem_GetTerrain` (`src/battle/battle_system.c:375-382`)
  is read by Nature Power, Secret Power and Camouflage (`src/battle/battle_script.c:8007`,
  `:8040`, `:8067`); its other readers are graphics and wild catches. `make_state` takes
  it from `Trainer.terrain` (`tools/terrain.py`), or for a trainer whose start tiles give
  several, from the run's `--terrain` (`cli.fight_terrain`, ledger G-18); `TERRAIN_UNKNOWN`
  (-1) while none is named or no map places the trainer, and coverage then refuses those
  three effects (G-21).
  Nature Power (`order.called_move`), Secret Power (`order.move_effects`) and Camouflage
  (`effects._camouflage`) read it through `ids.TERRAIN_MOVES` / `SECRET_POWER_EFFECTS` /
  `TERRAIN_TYPES` (`order.terrain_index`, the decomp's clamp to `TERRAIN_SPECIAL`).
- **BattleMon.type1/type2 -> `types`**: per party slot, from the personal row; written for
  the active battler by Camouflage (`BtlCmd_TryCamouflage`, `battle_script.c:7997-8022`)
  and restored from the personal row when the battler leaves (a switch, a faint,
  `step._restore_types`), as `BattleSystem_InitBattleMon` re-reads the party's types on the
  next switch-in (`battle_lib.c:114-115`). The Camouflage bit of `moveEffectsMask` is
  `camouflage` (G-20; read by the AI's `Basic_CheckCamouflage`, `script.s:1023`, and Roar's
  Ingrain test).
- **moveCur / msgMoveTemp** (Nature Power's call) stay per-action scratch: `step._use_move`
  recomputes the called move (`order.called_move`) from the chosen move and the terrain.
- **battleEndFlag -> `over`, `winner`**: `winner` is 0 none, 1 player lost (no
  player Pokemon has HP left; a double KO of both last Pokemon is a draw, played as a
  loss), 2 player won. A player faint alone only records a death.
- **recycleItem -> `recycle_item`** (per battler, as the decomp keeps it: `BtlCmd_RemoveItem`
  writes it, `BtlCmd_TryRecycle` reads and clears it, battle_script.c:8977, :9005-9007; a switch
  keeps it, B13-19); `item_consumed` is the engine's per-party-slot record of the same removals,
  read by the observation only.
- **Transform** (G-69): `VOLATILE_CONDITION_TRANSFORM` is `transformed`. `BtlCmd_Transform` copies
  the defender's BattleMon up to and including `ability` into the user's (species, the five stats
  as the BattleMon holds them, the moves, the IVs, the stat stages, both types, the form, the
  ability; battle_script.c:5346-5348), each slot's PP `min(5, table PP)` (:5364-5370); the party
  Pokemon's own species, form, stats, IVs, ability slot / form, moves and PP wait in the `shadow_*`
  fields (`shadow_moves` / `shadow_pp` with `mimic_slots` all four) until the battler leaves
  (`step._restore_transform`, `_restore_moves`) or the player's battle ends; a fainted one keeps the
  copy until its replacement enters (the exp payout reads its copied species, :2216). A level-up
  reloads the battler's own stats, level and HP, not its moves (battle_lib.c:169-194):
  `levelup.party_identity`. `ai_ability_known` is `battlerAbilities`: the decomp id the AI
  recorded, 0 none, -1 known but not named by the engine (settled to its id by a Transform).
- **formNum -> `form_num`** (H-5, per battler, `BattleMon.formNum`, battle_mon.h:79): the party's form at the
  send-out (`BattleSystem_InitBattleMon`, battle_lib.c:132: `initial_arrays` for the leads, `step._switch_in`
  after the Transform restore), copied by Transform with the BattleMon bytes (battle_script.c:5346-5348),
  written by `hooks.form_change` (`BattleSystem_TriggerFormChange`, battle_lib.c:6305-6451: Castform 0-3,
  Cherrim Overcast 0 / Sunshine 1, Arceus its plate's type, a Transformed Giratina-Origin back to Altered 0)
  and by Shaymin's revert (`subscript_check_shaymin_form.s:10`). Not cleared by a switch (the send-out
  overwrites it), not Baton-passed. Read only by that function's Cherrim, Arceus and Giratina tests; `form`
  (the party's, per slot) and its readers stay (Beat Up, personal rows): the two differ only for a Castform,
  Cherrim or Arceus (their personal rows ignore the form) and the Giratina rebuild writes both.
- **chatotCries -> `chatot_cry`** (E1 trial batch, G-70): `BattleSystem.chatotCries[battler]` as
  Chatter's parameter 0 / 1 / 2 (`ids.CHATTER_CHANCES`, sound_chatot.c:193-213), a battle constant
  like `terrain`: every enemy's is 0 (`ChatotCry_New` is invalid, field_battle_data_transfer.c:97,
  chatot_cry.c:13-25), the player's is the save's (`make_state(chatot_cry=...)`, party 0 (`[P]`, battle_system.c:915-924)).
- **BattleMon vs. party moves** (G-68): `moves` / `pp` of the active slot are the BattleMon's;
  Mimic writes only those (battle_script.c:4502-4516), flags the slot in `mimic_slots` and keeps the
  party's move and PP in `shadow_moves` / `shadow_pp`; `BattleMon_CopyToParty` skips flagged slots
  (its `BtlIOCmd_UpdatePartyMon`, battle_io_command.c:581-587), a level-up's learn writes the shadow, and
  leaving (`step._restore_moves`) puts the party's back (`InitBattleMon`, battle_lib.c:80-85).
  Spite (E3-B2-4, `effects._spite`) takes `min(4, ppCur)` from the BattleMon's slot of the target's
  `last_move` (BtlCmd_TrySpite, battle_script.c:4990-5016): it writes `pp` only, so a Mimic slot's
  `shadow_pp` stays (`BattleMon_CopyToParty` skips it).
- **aiContext**: `battlerMoves` (moves the AI has seen, as move ids in first-free order,
  `TrainerAI_RecordLastMove`, trainer_ai.c:2705-2716) -> `ai_seen_moves`,
  `battlerHeldItems` (the player's held items a message named, E2 batch 7) -> `ai_item_known`
  (per player party slot, an item id; set by `effects.reveal_item` where an item tag of
  `BTLSCR_ATTACKER` / `DEFENDER` / `MSG_BATTLER_TEMP` names it, battle_script.c:11726-11745:
  Leftovers and Black Sludge at the turn end, Sticky Barb's damage, Shell Bell's heal; cleared by
  `effects.forget_item` when the item goes, battle_lib.c:1819-1823, and on entry, :2153),
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
    with `row = tables.species_form_row[species, ability_form]`, is the decomp ability id,
    also for abilities the engine does not implement (`ability` is then `UNSUPPORTED`).
  - `ability_form` (G-41): the form whose personal row `ability_slot` indexes
    (`PokemonSpec.ability_from`). A trainer Pokemon's `MON_DATA_ABILITY` is set from the
    base species before the party builder sets the form (pokemon.c:473-485,
    src/trainer_data.c:216-218), so it is 0 there; the player's is the form's row. It is
    the party Pokemon's stored ability that `BattleSystem_InitBattleMon` re-reads
    (battle_lib.c:124), so `step._restore_ability` reads it.
  - `party_friendship` (F-15): the party Pokemon's u8 friendship (`include/struct_defs/pokemon.h:23`),
    which `Pokemon_IncreaseValue` wraps past 255 (pokemon.c:1859-1874) while `friendship` is
    `BattleMon.friendship`, clamped (battle_lib.c:1032-1043). A bag item's friendship band reads and
    writes both (`engine.bag.use_bag_item`); every switch-in (`step._switch_in`, battle_lib.c:136),
    a level-up and Shaymin-Sky's revert (`BattleSystem_ReloadPokemon`, :179) copy it into
    `friendship`. It is the value a save returns for the player's Pokemon.
  - `party_ability` (F-O2, per party Pokemon like `party_friendship`): the ability a script wrote into
    the party Pokemon in battle (MON_DATA_ABILITY written in battle, `BtlIOCmd_UpdatePartyMon`,
    battle_io_command.c:601-604); 0 (`UNSUPPORTED`) = none, the personal row's. Only `hooks.shaymin_land`
    on a Transformed battler writes it (Land's ability: the revert's `UpdateMonDataFromVar` sends the
    BattleMon's ability to the Transform user's OWN party Pokemon, battle_script.c:3401-3407,
    battle_controller.c:1496-1499); `step._restore_ability` puts it back into `ability` when that
    Pokemon leaves (`BattleSystem_InitBattleMon`, battle_lib.c:124), so every later read of the bench
    slot (`damage.decomp_abilities`, the AI's, Natural Cure, Heal Bell) sees it through `ability`. The
    same write sets the party's form 0 and recalculated stats: `shadow_form` / `shadow_stats` /
    `max_hp` / `hp` (`hooks.own_stats`). A state built with `party_ability` set must set `ability`
    to match (a run planner carrying the override forward does both).
  - `form` (F-16): a frozen Shaymin-Sky's form becomes 0 after a hit (`hooks.shaymin_land`;
    `subscript_check_shaymin_form.s`), with `ability`, `ability_slot`, `ability_form`, `types`,
    `stats`, `max_hp` / `hp` and `friendship` rewritten as `BattleSystem_ReloadPokemon` reloads them;
    the player's Pokemon keeps form 0 after the battle (`BtlIOCmd_UpdatePartyMon`,
    battle_io_command.c:597-604).
- **Experience (K3d-exp, design §2.1).** One block between `ai_ability_known` and
  `deaths`. Experience changes the fight: the payout runs mid-turn and a level-up rewrites
  the level, stats, max HP, HP and moves of a live battler (design §0, §1.4).
  - `exp` (`int32[6]`, the one exception to `int8`/`int16` storage, ruling D1: level-100
    values exceed `int16`) is the player party's `MON_DATA_EXPERIENCE`, which
    `BattleSystem_ReloadPokemon` copies into `BattleMon.exp` (battle_lib.c:192). Only the
    player's Pokemon gain exp (`BtlCmd_CalcExpGain` pays for a `SOLO_ENEMY` faint,
    battle_script.c:2196), so the opponent's is not kept. `make_state` takes the spec's
    `exp`, else the base of the level, and refuses an `exp` that is not of the level
    (`SpeciesData_GetLevelAt`, pokemon.c:2379-2392).
  - `evs` (`ids.Stat` order, like `ivs`) and `nature`, both sides: party data (no battle
    struct holds them) that the level-up `Pokemon_CalcStats` reads.
  - `exp_boost` (100 / 150 / 170, 0 = unresolved), `pokerus` and `friend_bonus`: the
    player's `PokemonSpec` fields (`BattleScript_GetExpTask`, battle_script.c:9757-9763;
    pokemon.c:4184-4199; pokemon.c:2664-2666). The host resolves the OT and language test
    into `exp_boost`.
  - `exp_mask` is `sideGetExpMask` (row below), a bitmask over the player's party slots
    (`u32[2]`, battle_context.h:176). The index is the enemy position `(b >> 1) & 1` of an
    enemy battler `b` (battle_lib.c:1458, :1470; entry 0 battler 1, entry 1 battler 3), neither
    the battler nor the side (doubles-state-design section 5 rule 2). Shape key `"E"`: an int8
    scalar at B = 2, int8[2] at B = 4. `1 << lead` at the start (B = 4: the player battlers'
    leads, battler 2's unless MULTI, battle_lib.c:1475-1477), then written by every send-out
    (`step._switch_in`; battle_lib.c:1456-1479, battle_script.c:543-635): a player send-out
    flags both entries, an enemy send-out clears and flags its own. The payout of enemy
    battler `b` reads and clears entry `(b >> 1) & 1` only (battle_script.c:2205, :9700,
    :9709, :9741, :10191).
  - `knocked_off` is per SIDE, int8[S, 6]: bit k of `SideConditions.knockedOffItemsMask`
    (battle_context.h:49), set by a Knock Off on any battler of the side whose
    `selectedPartySlot` is k (battle_script.c:6627). Every party of the side shares it (MULTI:
    a Knock Off on battler 2's slot k covers party 0's slot k; TWO_TRAINERS: parties 1 and 2).
    Readers take the battler's own slot (`state.knocked_off_at`): InitBattleMon zeroes the
    battle item of a newcomer under the bit (battle_lib.c:161-163, `step._switch_in`), the
    Sticky Barb transfers skip such an attacker (:5390, :7380), UpdatePartyMon skips the item
    (battle_io_command.c:590-592; the side's mask, battle_controller.c:1472). D3 fix B; at
    B = 2 the side is the party, so the values are unchanged.
  - `knocked_item` (D9) is the party Pokemon's held item while its side's `knocked_off` bit is
    set: its item when the bit was set (for the knocked Pokemon the item Knock Off removed;
    for another party's slot k its item then, which it keeps on the field until it is loaded
    again). The party copy is never written under the bit (battle_io_command.c:590-592), and
    the payout reads the party item (bs.c:2210, :9706, :10226; `state.party_held_item`), so a
    knocked-off Lucky Egg or Exp. Share still pays. Every other item change reaches the party
    at once (`BattleMon_CopyToParty`, battle_lib.c:1819-1823). 0 when no bit covers the slot.
  - `learn_level` / `learn_cursor` are the deferred learnset scan: the level whose scan is
    next and the index into it (the decomp restarts `GET_EXP_LEARNSET_INDEX` at 0 for each
    level, battle_script.c:9882). Both are 0 until the payout and the learn resolve set them
    (the selected-slot pin and the window (c) refusal code are removed: review b2 #6, below).
  - **Mid-turn learn prompts (E3-B2-21).** The payout runs the whole `GetExpTask`, every level's
    learnset scan and prompt included, inside itself, so the prompt comes before any later
    action or draw of the turn: at `MoveEnd` (`AnyExpPayout`, battle_controller_player.c:3968,
    before the tail `TriggerEffectOnSwitch` at :3973), in `CheckFieldConditions` (:941, the
    weather faints), in `CheckMonConditions` (:1296, a battler's end-of-turn block) and at the
    `TurnEnd` payout (:1846, before `ReplaceFainted`, :1848). `step` pauses at the prompt:
    `learn_site` (per field, 0 none, 1 `LEARN_MOVE_END`, 2 `LEARN_FIELD`, 3 `LEARN_MON`) and
    the turn scratch (`ts_ran`: the MoveEnd `clear`; `ts_move_cur`: each battler's `moveCur`;
    `ts_eot`: the turn end's order, fixed at the last `BranchActions`, :888-891; `ts_qc`: the
    turn's speedRand draw, :853-857, which every stop, a party list's included, keeps for the
    resume's re-sorts and turn end, `step._turn_outcome`; a battler keeps its bit only with a living Quick
    Claw holder in its party, a holder on its field, a Quick Claw in its `recycle_item`, or an
    item-transfer move known and a living holder in any other party, else 0, `step.qc_readable`,
    E3-B2-31; a partner's move learnt at the pause is not counted, a doubles item, E3-B2-31(b)(3)) hold the turn, and the answered state resumes it through the
    same `step` (`levelup.learn_prompt`, `levelup.learn`). A pending action re-reads its move
    slot (`FightCommand`, :1889-1895), so the learner's replaced slot runs the new move. The
    answer is a request `(state, option)` of `search.expand.expand`; a playout gives the
    move up at the pause (`resolve_learns`, ruling D3). A payout that ends the battle leaves
    its prompts (ruling D8: `over`).
  - The search masks only the HM slots and, when the offered move's effect is unsupported,
    every replacement (ruling D10; coverage then refuses it only where it could auto-learn
    into an empty slot, `coverage.learnable_moves`). The D2 masks (Disable, Encore) and the
    selected-slot pin are gone: the prompt is answered at the payout, before the turn end
    that ends them and before the slot is read again. They are legal game choices the search
    skips, so plan values at learn nodes are achievable lower bounds (`levelup.learn_options`).
- **Not from a struct field.** `pursuit_target` and `wish_hp` are
  `BattleState`'s own bookkeeping: the decomp derives Pursuit's pre-switch hit
  from per-turn scratch (the opposing battler's selected move when a switch is
  chosen) and Wish's heal from the battlers' HP when it lands.

- The mid-turn switch decision (E2 batch 6, U-turn and Baton Pass): `pivot_battler` (per field,
  `1 + battler` owing the party pick at subscript_attack_then_switch_out.s:71 /
  subscript_baton_pass.s:17, else 0) and the turn scratch `ts_*` (per field: the
  `BattleContext` values a resumed turn reads: `turnOrderCounter`, `battlerActionOrder`, the
  command-selection picks (`ts_move`, `ts_struggling`, `ts_locked`, `ts_item`), the per-battler
  `moveEffectsData.custapBerry` (`ts_custap`) and `.quickClaw` (`ts_claw`, B10-6; between turns `custap_flag` / `claw_flag`, SF2 (k)), the per-battler
  `turnFlags` later actors read
  (`ts_protect_ok`, `ts_endure_ok` (`turnFlags.enduring`, Endure, G-65), `ts_taken`,
  `ts_follow_me` (per side, `SideConditions.followMe` / `followMeUser`: `1 + followMeUser`, 0 when `followMe`
  is clear; `BattleSystem_SetupNextTurn` clears it, battle_lib.c:2251-2252, D2-11; written in both
  formats, ruling D2-33: at B = 2 the other side's non-Ghost Curse reads it, lib.c:1714-1717),
  `ts_defender` / `ts_speed_order` (the paused action's `battleCtx->defender` as `1 + defender` and `monSpeedOrder`, written at B = 4 only, D3-10), `ts_assurance`: `assuranceDamageMask`, which the action-phase HP updates write in the state
  itself and Assurance reads, B13-17; `ts_helping_hand`: `turnFlags.helpingHand` per battler,
  set on the partner by a Helping Hand that worked (battle_script.c:6291-6314, B = 4 only),
  read by CalcMoveDamage (battle_lib.c:6694-6696), Future Sight's setup (battle_script.c:
  6092-6094) and Beat Up (:6214-6216), cleared by `CleanupFaintedMon` (battle_lib.c:2190-2193)
  and the turn's end (D3 fix B); `ts_sub_hit`: the paused action's `selfTurnFlags.statusFlags & SELF_TURN_FLAG_SUBSTITUTE_HIT` per
  battler (`hooks.intimidate` reads it, F-4), and `ts_loved`: the paused action's `selfTurnFlags.statusFlags &
  SELF_TURN_FLAG_INFATUATED` per battler slot (H-4: set by a landed Cute Charm, subscript_infatuate.s:29). The
  attacker's slot is cleared before the list (subscript_attack_then_switch_out.s:70, battle_script.c:9623-9624), so
  the resumed tail sees only a pursuer flagged by the U-turn user's Cute Charm during the Pursuit (:54,
  subscript_pursuit.s:40); a user felled by the Pursuit ends the script at :55 and its MODE_PURSUIT tail reads its
  own flag. Both feed the tail's Destiny Knot step (battle_lib.c:4546-4566). Value 1: flagged and the
  sideEffectMon at the list; 2 (B = 4 only, H-M1): flagged, but a later pursuer's Cute Charm trigger made that
  pursuer the sideEffectMon (battle_lib.c:4350), so the resumed tail reads the 1 slot (or none) as sideEffectMon. The two self turn flags kept across a
  U-turn's list (`BattleContext_Init` clears the rest at the action's end, bcp.c:3998); `ts_grudge`: per battler, the battler fainted in the paused
  action holding `MOVE_EFFECT_GRUDGE`: its `battleStatusMask` SYSCTL_MON_FAINTED bit and `moveEffectsMask` survive the
  U-turn's list (battle_script.c:1770-1771, :1799; battle_lib.c:2166), so the resumed action's LOOP_FAINTED runs
  TryGrudge on the newcomer (review b2 #5); `focusItemActivated` and
  `SELF_TURN_FLAG_PLUCK_BERRY` live in `_use_move`'s hit loop (`effects.MoveHit`);
  `lightningRodActivated` / `stormDrainActivated` are `_use_move`'s local `redirected`; `turnFlags.roosting` is the public `roosting`, kept at a pause
  and while a faint replacement is pending, `engine.state`; the other turn flags are reset every
  turn and `step` resolves a whole turn: `protecting` is the first actor's `protect_ok` event,
  read by the second actor in `step._execute_actions`; a Feint that hits clears it,
  subscript_feint.s:20, E3-B2-13: `step._execute_actions` zeroes the defender's `protect_ok` read), the per-battler `moveFailFlags`
  (`ts_move_failed`), the paused action's `moveCur` and its defender's damage taken (`ts_used`,
  `ts_damaged`, `ts_dealt`), its `moveTemp` (`ts_move_temp`: the move FightCommand ran, the
  Encore's or the lock's included, bcp.c:1861-1898; its tail's Choice lock, `movePrev` and
  `movePrevByBattler` read it, bcp.c:3917-3946, E2-ML-4), `moveStatusFlags & MOVE_STATUS_SUPER_EFFECTIVE` (`ts_se`: the paused or
  chased action's only; the U-turn's, or the last Pursuit's, lib.c:3359, script.c:6964-6973; the tail's
  Enigma Berry reads it with `ts_sub_hit`, battle_lib.c:5378, :5421-5428), the per-side bitmask of the faints handled (`ts_alive`), `powerMul`
  at the list (`ts_pick_mul`) and the search's turn-start reads (`ts_power_mul`,
  `ts_switched`; E3-B2-21: `ts_ran`, `ts_move_cur`, `ts_eot`, `ts_qc`, with `learn_site` the
  learn pause above). All are 0 between turns (`engine.state` docstring): `turnOrderCounter`,
  `battlerActionOrder` and `battlerActions` (the actions as chosen, and the MOVE_END writes of a
  Pursuit, `ts_pursued` (bitmask, bit b = battler b's PICK_COMMAND set MOVE_END by TryPursuit (script.c:6931), written `|= 1 << pursuer` by both engines; BtlCmd_TryPursuit's loop, battle_script.c:6884-6935 with subscript_pursuit.s:82, lets several battlers chase one switch), and of UpdateAfterSwitch for a Pokemon Roar drags in, `ts_dragged`, which
  skip their actions in `step._execute_actions`) are kept only at such a pause.
- The chosen target (D2-1): `chosen_target` (per battler, `battlerActions[b][1]`, a battler index
  0..B-1, not 1 + battler: it has no "none" value; initial `b ^ 1`). Unlike the other
  `battlerActions` values it persists across turns and switches (no code clears it,
  battle_controller_player.c:180-182) and is written at selection (`actions.chosen_targets`, in
  `step._action_phase`) only by a battler that reaches the target menu (bcp.c:507-531); D2-9 and
  D2-13 overwrite it later (GoToMoveScript stores the called move's defender, script.c:2123). Not in
  `VOLATILE_FIELDS` or `BATON_PASSED_FIELDS`. Written in both formats (ruling D2-33, which
  supersedes D2-28): at B = 2 a locked battler reads it (bcp.c:1885-1888, lib.c:1711), the user after
  a RANGE_MY_SIDE or non-Ghost Curse selection. At the end of a turn at B = 2 it is canonical
  (ruling D2-35, `step.canonical_target`, reference `_end_turn`): `b ^ 1` unless the battler is
  locked (`locked_move`) or encored (`encore_move`), the only states in which it can be read
  before the next selection overwrites it; a paused turn keeps it for the actions still to run.
- The last hitter (D2-14): `last_hit_by` (per battler, `1 + BattleContext.lastHitByBattler[b]`, 0 =
  no hit yet), written by every damaging `UpdateHP` on `b` (a Substitute hit and a partner's hit
  included, bcp.c:3347, :3357) and never cleared: not by a switch or a faint (lib.c:2110-2225), so
  it is not in `VOLATILE_FIELDS` or `BATON_PASSED_FIELDS`. Bide's release reads it. Written at
  B = 4 only (ruling D2-24).
- The move slot (E2-ML-5): `move_slot` (per battler, `BattleContext.moveSlot[b]`, a slot 0..3, initial 0).
  The command selection writes it for a FIGHT that picks a move: the Encore slot under an Encore
  (bcp.c:440-446), else the chosen slot (:526); a Struggling battler (:433-434), a locked one (it
  picks no command, :333-337), an item, a switch and an absent battler keep it. The only other
  writer is the disobedience check (ATTACKER_MOVE_SLOT = a random slot, bcp.c:2183), which never
  runs under the obedient-party assumption (the `BattleMon.OTId` row). Nothing clears it
  (BattleContext_Init, battle_lib.c:1974-2017; zeroed only by BattleContext_New, bcp.c:182; not a
  switch or a faint), so it is not in `VOLATILE_FIELDS`. An Encore takeover changes moveTemp, not
  moveSlot (bcp.c:1869-1883). Readers: CalcTurnOrder's priority, `moves[moveSlot]` as it stands
  now (battle_lib.c:1375-1397; `order.order_moves`), and ATTACKER_MOVE_SLOT (common.h:8) in
  TryGrudge (battle_script.c:6753), Trump Card's power (:7237) and BattleSystem_DecPPForPressure
  (battle_lib.c:7390-7395): Counter, Mirror Coat and Metal Burst (battle_script.c:4646, :4699,
  :7180) and the Magic Coat reflection's drain of the holder's slot (bcp.c:3083; the slot is kept
  in `magic_coat` as 1 + moveSlot when the Magic Coat lands). The engines read `move_slot` at every
  one of them (`chosen_slot` in `step._use_move`, `_Turn.chosen_slot` in the reference). Written
  in `step._action_phase` (`order.selection_slots`) and the reference's `_ref_step`.
- The selected move's carried value (E2-ML-8): `selected_temp` (per battler,
  `battlerActions[b][BATTLE_ACTION_TEMP_VALUE]`, initial 0). The selection writes 1 + the slot for
  a FIGHT that picks a move (bcp.c:525), 0 under an Encore (:447), the party slot 0..5 for a switch
  (:595) and the BattleItemUse data for a bag item (:565); a battler that picks no command (a
  locked one, :333-337; Battler_CanPickCommand, battle_lib.c:6224-6232, also counts RECHARGING,
  which no modelled move sets: `recharge` is reserved), a Struggling one (:433-434) and an
  absent one keep it, and nothing resets it (BattleContext_Init, battle_lib.c:1974-2017).
  `Battler_SelectedMove` (battle_lib.c:2858-2868) reads `moves[TEMP_VALUE - 1]` under a FIGHT
  command: 1..4 read the move slots; 5 reads `moves[4]`, the u16 after them, the low 16 bits of the
  IV bitfield (battle_mon.h:64-72): `hpIV | attackIV << 5 | defenseIV << 10 | (speedIV & 1) << 15`
  (`order.past_moves_word`, the reference's `_past_moves_word`). Stated assumptions: the bitfield
  is allocated from bit 0 (the ARM ABI); a `moves[4]` id past the move table (>= MAX_MOVES = 468,
  ai_context.h:45) reads as MOVE_NONE (MOVE_DATA would read past `moveTable`); a bag item's
  BattleItemUse (message_defs.h:15-19: item id, category, target) is kept as 0 (MOVE_NONE): its
  `moves[TEMP_VALUE - 1]` lies far past the BattleMon. Both are reachable only through a
  Struggling or locked battler whose last selection was the switch or the item. Readers: Pursuit
  (battle_script.c:6909) and Me First (:7296), through `order.battler_selected_move`; a locked
  battler reads the value of its last selection, not its lock's move. Written by
  `order.with_selection` (`step._action_phase`, `step.replacement_power_mul`, the laws' entry
  points) and the reference's `_ref_step`. Not in `VOLATILE_FIELDS`.
- The Trace fires (D2-16, ruling D2-5): `ts_trace_fires` (per battler, turn scratch, 0 between turns)
  counts the Traces of the slot's holder in the running step, the index into `Outcome.trace_pick`;
  written at B = 4 only.

- **Mapped rows with extra rules (kept out of the status column).**
  `BattleContext.defender`: the paused action's only, `1 + defender`, 0 = BATTLER_NONE, written at B = 4: the resumed tail reads the defender FightCommand fixed, bcp.c:1900, kept through the party list and a Pursuit, subscript_pursuit.s:5, :80, :116; inside the turn `step` recomputes it; at B = 2 the re-resolve gives the same battler, D3-10.
  `BattleContext.turnOrderCounter`: reset to 0 by a Trick Room's MoveEnd re-sort at B = 4, bcp.c:3993; the battlers that acted are `ts_acted`, not the cursor's side of the order.
  `BattleContext.battlerActionOrder`: written from `order.action_order`'s `int8[B]` order; the position loop's tie bits are `Outcome.order_tie`, never stored; re-sorted at a Trick Room's MoveEnd at B = 4 (`order.resort_action_order`, bits `Outcome.resort_tie[b]`, b the Trick Room's battler, never stored); at a pause `ts_order` / `ts_cursor`.
  `BattleContext.monSpeedOrder`: the running action's: computed before each action at B = 4 (`order.sort_mon_speed`, `Outcome.sort_tie[b]`, b the acting battler), not stored; a paused action's: `ts_speed_order`, D3-10.

## BattleMon (`include/battle/battle_mon.h`)

| Struct.field | pytree field | status |
|---|---|---|
| `BattleMon.species` | `species`, `shadow_species` | mapped |
| `BattleMon.attack` | `stats`, `power_trick`, `power_trick_desync` | mapped |
| `BattleMon.defense` | `stats`, `power_trick`, `power_trick_desync` | mapped |
| `BattleMon.speed` | `stats`, `shadow_stats` | mapped |
| `BattleMon.spAttack` | `stats` | mapped |
| `BattleMon.spDefense` | `stats` | mapped |
| `BattleMon.moves` | `moves`, `shadow_moves`, `mimic_slots` | mapped |
| `BattleMon.hpIV` | `ivs`, `shadow_ivs` | mapped |
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
| `BattleMon.formNum` | `form`, `shadow_form`, `form_num` | mapped |
| `BattleMon.isShiny` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.padding0026_06` | — | not needed because it is padding |
| `BattleMon.ability` | `ability`, `ability_slot`, `ability_form`, `party_ability` | mapped |
| `BattleMon.weatherAbilityAnnounced` | `weather_announced` | mapped |
| `BattleMon.intimidateAnnounced` | `intimidate_announced` | mapped |
| `BattleMon.traceAnnounced` | `trace_announced` | mapped |
| `BattleMon.downloadAnnounced` | `download_announced` | mapped |
| `BattleMon.anticipationAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.forewarnAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.slowStartAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.slowStartFinished` | — | later (M6) |
| `BattleMon.friskAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.moldBreakerAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in |
| `BattleMon.pressureAnnounced` | — | not needed because it only stops a switch-in message repeating; switch-in hooks run once per switch-in (the message's AI reveal, ruling G-10, is `ai_ability_known`) |
| `BattleMon.padding0028_0B` | — | not needed because it is padding |
| `BattleMon.ppCur` | `pp`, `shadow_pp`, `mimic_slots` | mapped |
| `BattleMon.ppUps` | `pp_ups` | mapped |
| `BattleMon.level` | `level` | mapped |
| `BattleMon.friendship` | `friendship`, `party_friendship` | mapped |
| `BattleMon.nickname` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.curHP` | `hp` | mapped |
| `BattleMon.maxHP` | `max_hp` | mapped |
| `BattleMon.OTName` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.exp` | `exp` | mapped |
| `BattleMon.personality` | — | not needed because its battle uses (nature, ability slot, gender) are resolved into `PokemonSpec` before `make_state` |
| `BattleMon.status` | `status`, `status_counter` | mapped |
| `BattleMon.statusVolatile` | `confusion`, `flinched`, `trapped`, `recharge`, `foresight`, `bide_turns`, `curled`, `locked_move`, `infatuated`, `destiny_bond`, `uproar`, `curse`, `nightmare`, `mean_look`, `focus_energy`, `torment`, `transformed` | mapped |
| `BattleMon.OTId` | — | not needed because only obedience and the traded-Pokemon exp bonus read it; the tool assumes an obedient party and resolves the exp bonus into `exp_boost` before `make_state` |
| `BattleMon.heldItem` | `item` | mapped |
| `BattleMon.padding007A` | — | not needed because it is padding |
| `BattleMon.timesDamaged` | — | not needed because no battle or AI logic reads it (only the generic BattleMon getter and setter) |
| `BattleMon.trainerMessageFlags` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.gender` | `gender` | mapped |
| `BattleMon.OTGender` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.capturedBall` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.moveEffectsMask` | `leech_seed`, `leech_seed_recipient`, `water_sport`, `mud_sport`, `aqua_ring`, `perish_song`, `yawn`, `lock_on`, `ingrain`, `imprison`, `imprisoned`, `minimized`, `semi_invulnerable`, `power_trick`, `power_trick_desync`, `magnet_rise`, `ability_suppressed`, `charged_turns`, `grudge`, `miracle_eye`, `camouflage` | mapped |
| `BattleMon.moveEffectsTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleMon.moveEffectsData` | `taunt`, `protect_count`, `rollout_count`, `recharge`, `substitute_hp`, `trap_move`, `choice_lock` | mapped |

## MoveEffectsData (nested in `BattleMon.moveEffectsData`)

| Struct.field | pytree field | status |
|---|---|---|
| `MoveEffectsData.disabledTurns` | `disable_turns` | mapped |
| `MoveEffectsData.encoredTurns` | `encore_turns` | mapped |
| `MoveEffectsData.chargedTurns` | `charged_turns` | mapped |
| `MoveEffectsData.tauntedTurns` | `taunt` | mapped |
| `MoveEffectsData.protectSuccessTurns` | `protect_count` | mapped |
| `MoveEffectsData.perishSongTurns` | `perish_song_turns` | mapped |
| `MoveEffectsData.rolloutCount` | `rollout_count` | mapped |
| `MoveEffectsData.furyCutterCount` | — | later (M6) |
| `MoveEffectsData.stockpileCount` | — | later (M6) |
| `MoveEffectsData.stockpileDefBoosts` | — | later (M6) |
| `MoveEffectsData.stockpileSpDefBoosts` | — | later (M6) |
| `MoveEffectsData.truant` | `truant`, `fake_out_turn` | mapped |
| `MoveEffectsData.flashFire` | `flash_fire` | mapped |
| `MoveEffectsData.lockOnTarget` | `lock_on_target` | mapped |
| `MoveEffectsData.mimickedMoveSlot` | `mimic_slots` | mapped |
| `MoveEffectsData.bindTarget` | — | not needed because in singles the binder is always the opposing active Pokemon |
| `MoveEffectsData.meanLookTarget` | `mean_look_target` | mapped |
| `MoveEffectsData.lastResortCount` | `last_resort_count` | mapped |
| `MoveEffectsData.magnetRiseTurns` | — | later (M6) |
| `MoveEffectsData.healBlockTurns` | — | later (M6) |
| `MoveEffectsData.embargoTurns` | `embargo` | mapped |
| `MoveEffectsData.canUnburden` | `can_unburden` | mapped |
| `MoveEffectsData.metronomeTurns` | `metronome_turns` | mapped |
| `MoveEffectsData.micleBerry` | `micle` | mapped |
| `MoveEffectsData.custapBerry` | `custap_flag`, `ts_custap` | mapped |
| `MoveEffectsData.quickClaw` | `claw_flag`, `ts_claw` | mapped |
| `MoveEffectsData.meFirst` | `me_first` | mapped |
| `MoveEffectsData.padding0004_1F` | — | not needed because it is padding |
| `MoveEffectsData.rechargeTurnNumber` | `recharge` | mapped |
| `MoveEffectsData.fakeOutTurnNumber` | `fake_out_turn` | mapped |
| `MoveEffectsData.slowStartTurnNumber` | — | later (M6) |
| `MoveEffectsData.meFirstTurnNumber` | `me_first_turn` | mapped |
| `MoveEffectsData.substituteHP` | `substitute_hp` | mapped |
| `MoveEffectsData.transformedPID` | — | not needed because it is cosmetic (the transformed sprite and name, messages): Transform copies no personality for any battle rule |
| `MoveEffectsData.disabledMove` | `disabled_move` | mapped |
| `MoveEffectsData.bindingMove` | `trap_move` | mapped |
| `MoveEffectsData.encoredMove` | `encore_move` | mapped |
| `MoveEffectsData.encoredMoveSlot` | `encore_slot` | mapped |
| `MoveEffectsData.lastResortMoves` | `last_resort_moves` | mapped |
| `MoveEffectsData.choiceLockedMove` | `choice_lock` | mapped |
| `MoveEffectsData.transformedGender` | — | not needed because it is cosmetic (sprites and messages): Attract, Captivate and Rivalry keep the user's own `gender` (battle_script.c:5337-5338) |
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
| `SideConditions.followMe` | `ts_follow_me` | mapped |
| `SideConditions.followMeUser` | `ts_follow_me` | mapped |
| `SideConditions.knockedOffItemsMask` | `knocked_off`, `knocked_item` | mapped |
| `SideConditions.padding00_1D` | — | not needed because it is padding |
| `SideConditions.spikesLayers` | `spikes` | mapped |
| `SideConditions.toxicSpikesLayers` | `toxic_spikes` | mapped |
| `SideConditions.padding04_04` | — | not needed because it is padding |

## FieldConditions (embedded in `BattleContext.fieldConditions`)

| Struct.field | pytree field | status |
|---|---|---|
| `FieldConditions.weatherTurns` | `weather_turns` | mapped |
| `FieldConditions.futureSightTurns` | `future_timer` | mapped |
| `FieldConditions.wishTurns` | `wish_timer` | mapped |
| `FieldConditions.futureSightMove` | `future_move` | mapped |
| `FieldConditions.futureSightAttacker` | `future_attacker` | mapped |
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
| `BattleContext.defender` | `ts_defender` | mapped |
| `BattleContext.defenderTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.faintedMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.switchedMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.switchedMonTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.abilityMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.magicCoatMon` | — | not needed because it is per-action scratch: `step._execute_actions`' reflect phase (E3-B2-6) knows the reflected move's user as the actor whose pass it resumes |
| `BattleContext.sideEffectType` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectParam` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectFlags` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.sideEffectMon` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.lastBattlerId` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.gainedExp` | — | not needed because it is per-payout scratch (`BtlCmd_CalcExpGain`, battle_script.c:2220-2236) that the payout recomputes from `exp_mask` and the party |
| `BattleContext.sharedExp` | — | not needed because it is per-payout scratch (`BtlCmd_CalcExpGain`, battle_script.c:2226-2230) that the payout recomputes from the party's held items |
| `BattleContext.sideGetExpMask` | `exp_mask` | mapped |
| `BattleContext.scriptNarc` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptFile` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackPointer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackNarc` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackFile` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.scriptStackCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.turnOrderCounter` | `ts_cursor` | mapped |
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
| `BattleContext.meFirstTurnOrder` | — | not needed because it is `turn` + 1 (lib.c:2028 starts it at 1, bcp.c:1853 adds 1 at each TurnEnd with `totalTurns`): `damage.me_first_active` reads it so |
| `BattleContext.taskData` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.tmpData` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.fieldConditionsMask` | `field_mask` | mapped |
| `BattleContext.fieldConditions` | `weather_turns`, `future_timer`, `future_damage`, `wish_timer` | mapped |
| `BattleContext.sideConditionsMask` | `stealth_rock`, `tailwind`, `lucky_chant`, `future_sight` | mapped |
| `BattleContext.sideConditions` | `reflect`, `light_screen`, `mist`, `safeguard`, `spikes`, `toxic_spikes` | mapped |
| `BattleContext.turnFlags` | `roosting`, `ts_assurance`, `magic_coat`, `ts_helping_hand` | mapped |
| `BattleContext.selfTurnFlags` | `ts_sub_hit`, `ts_loved` | mapped |
| `BattleContext.moveFailFlags` | — | not needed because it is reset every turn and `step` resolves a whole turn (its only reader, the Uproar check, reads the turn's `move_failed` event) |
| `BattleContext.aiContext` | `ai_seen_moves`, `ai_item_known`, `ai_ability_known`, `trainer_items`, `trainer_item_count` | mapped |
| `BattleContext.aiScriptTemp` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.aiScriptCursor` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleStatusMask` | `ts_grudge` | mapped |
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
| `BattleContext.moveStatusFlags` | `ts_se` | mapped |
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
| `BattleContext.switchedPartySlot` | — | not needed because the switch target comes from the action (or, for Roar's drag, from `Outcome.roar_target` inside `step._use_move`) and is written to `active` inside `step` |
| `BattleContext.aiSwitchedPartySlot` | — | not needed because the switch target comes from the action and is written to `active` inside `step` |
| `BattleContext.battlerActions` | `ts_actions`, `ts_pursued`, `ts_dragged`, `chosen_target`, `selected_temp` | mapped |
| `BattleContext.battlerActionOrder` | `ts_order` | mapped |
| `BattleContext.monSpeedOrder` | `war_swaps`, `ts_speed_order` | mapped |
| `BattleContext.monSpeedValues` | `speed_rec` | mapped |
| `BattleContext.ioQueue` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.ioBuffer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleScript` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleMons` | `active` | mapped |
| `BattleContext.moveTemp` | `ts_move_temp` | mapped: per-action scratch `step` recomputes inside the turn, kept only for an action paused at a party list (the paused action's moveTemp, which its resumed tail's UpdateMoveBuffers reads, bcp.c:3917-3946; E2-ML-4) |
| `BattleContext.moveCur` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.movePrev` | `move_prev` | mapped |
| `BattleContext.moveLockedInto` | `locked_move` | mapped |
| `BattleContext.moveProtect` | `protect_count` | mapped |
| `BattleContext.moveHit` | `move_hit` | mapped |
| `BattleContext.moveHitBattler` | — | not needed because in singles it is the opposing battler whenever `move_hit` is set: both are written together (battle_controller_player.c:4500-4507), and `ClearFlags` (:4389) clears only `moveHit` |
| `BattleContext.moveHitType` | — | later (M6) |
| `BattleContext.movePrevByBattler` | `last_move` | mapped |
| `BattleContext.moveCopied` | `move_copied` | mapped |
| `BattleContext.moveCopiedHit` | `move_copied_hit` | mapped |
| `BattleContext.moveSketched` | — | later (M6) |
| `BattleContext.moveSelected` | — | not needed because its one reader, FightCommand's branch 1.5 (bcp.c:1889), runs only for a battler that picked a move this turn, which wrote it first (:446, :527): a Struggling, encored or locked battler takes an earlier branch (:1866-1888). It is not reset every turn (zeroed only by BattleContext_New, bcp.c:182; BattleContext_Init, battle_lib.c:1974-2017, leaves it), but a stale value is never read; the learn prompt that replaces the slot is `reselected` (E3-B2-21) |
| `BattleContext.moveSlot` | `move_slot` | mapped: written only at a FIGHT selection that picks a move (bcp.c:445 the Encore slot, :526 the chosen slot), never reset (BattleContext_Init, battle_lib.c:1974-2017; zeroed only by BattleContext_New, bcp.c:182), so a locked battler keeps its last one: CalcTurnOrder's priority (battle_lib.c:1375-1397) and TryGrudge (common.h:8) read it (E2-ML-5) |
| `BattleContext.conversion2Move` | `conv2_move` | mapped |
| `BattleContext.conversion2Battler` | `conv2_battler` | mapped |
| `BattleContext.conversion2Type` | `conv2_type` | mapped |
| `BattleContext.metronomeMove` | `metronome_move` | mapped |
| `BattleContext.storedDamage` | `bide_damage` | mapped |
| `BattleContext.lastHitByBattler` | `last_hit_by` | mapped |
| `BattleContext.battlerSpeedTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.battlersSwitchingMask` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.levelUpMons` | — | not needed because only the battle's end reads it (`BattleMain_CopyBattleSysToDTOAndFree`, battle_main.c:723, for evolution), after the fight being searched |
| `BattleContext.padding310A` | — | not needed because it is padding |
| `BattleContext.speedRand` | `qc_roll` | mapped |
| `BattleContext.flingTemp` | — | not needed because Fling's flung item (`_Hit.fling_item`) and its script run inside the one hit (`effects._fling`, `hooks.fling_effect`, E3-B2-1) and nothing reads it after |
| `BattleContext.flingScript` | — | not needed because the item's script is chosen from `tables.item_fling_effect` inside the hit (`effects._fling`, E3-B2-1) and nothing reads it after |
| `BattleContext.safariCatchStage` | — | not needed because trainer battles have no Safari Zone |
| `BattleContext.safariEscapeCount` | — | not needed because trainer battles have no Safari Zone |
| `BattleContext.runAttempts` | — | not needed because the player cannot flee a trainer battle |
| `BattleContext.battleEndFlag` | `over`, `winner` | mapped |
| `BattleContext.magnitude` | — | later (M6) |
| `BattleContext.fieldWeatherChecked` | — | not needed because `make_state` applies the start weather directly |
| `BattleContext.hpTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.recycleItem` | `recycle_item`, `item_consumed` | mapped |
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
