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
  `acid_rain` (int8, after `replace_mul`, outside the experience block; sticky to the
  battle's end) is not decomp state either: the operand of Pursuit's
  `fieldConditionsMask -=` (subscript_pursuit.s:100, :115; OPCODE_SUB_TO_ZERO,
  battle_script.c:3282-3287), 1..3, 0 = none; set only where the subtraction happens with a
  nonzero operand while a field condition is up. The engine cannot hold the result: a state
  with `acid_rain != 0` is refused (`coverage.check_acid_rain`; the search's `exp_refused`
  readers). Owed: FCM.md (the field conditions as the decomp mask).
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
  Transform, Rage, Substitute flag, Nightmare) are later (M6); the Substitute flag is
  implied by `substitute_hp > 0`. Torment -> `torment` (E3 batch 1, E3-2-1: per battler `[B]`,
  `VOLATILE_CONDITION_TORMENT` bit 31, set by subscript_torment_start.s:8 with no count, cleared
  with the volatiles; Baton Pass drops it, condition.h:72-76, battle_lib.c:2075; it only blocks
  the slot holding `last_move` at the next move selection, `actions.usable_moves`).
- **moveEffectsMask** bits mapped: Leech Seed -> `leech_seed`, Water Sport ->
  `water_sport`, Mud Sport -> `mud_sport`, Aqua Ring -> `aqua_ring` (per battler, cleared on switch-out; the
  Electric power halving reads every active battler, `battle_lib.c:6812`), Ingrain -> `ingrain`,
  Imprison -> `imprison` (its user) and Imprisoned -> `imprisoned` (its foes) (E2 batch 7, per
  battler, cleared with the volatiles; Baton Pass keeps Ingrain only, moves.h:46-48), Perish
  Song, Yawn and Lock-On (below). Camouflage's bit is read off `types` (`effects.camouflaged`).
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
  suppressed Ingrain user no longer anchors against Roar (the whole-mask compare, :7). The rest
  (Charge, Grudge, Miracle Eye, Heal Block) are later (M6).
- **fieldConditionsMask**: the weather bits (temporary vs. permanent rain, sand,
  sun, hail, plus deep fog) -> `weather` with `weather_timer == 0` meaning
  permanent; Trick Room turns -> `trick_room`; Gravity turns -> `gravity`. The
  Uproar battler mask (`FIELD_CONDITION_UPROAR`, one bit per battler) is `uproar != 0`: the
  bit and the battler's Uproar count are set together (subscript_uproar.s:6-11) and cleared
  together (bcp.c:1525-1535; `UpdateAfterSwitch`, battle_lib.c:2135, and
  `CleanupFaintedMon`, :2213, with the volatiles).
- **sideConditionsMask**: Stealth Rock -> `stealth_rock`, Tailwind turns ->
  `tailwind`, Lucky Chant turns -> `lucky_chant`, Future Sight -> `future_sight` (E2 batch 5:
  per side; set by `BtlCmd_TryFutureSight`, battle_script.c:6070-6072, cleared only when the hit
  fires on a battler with HP, bcp.c:1767-1770, so it outlives a count that ran out on a fainted
  battler; the AI reads it). Its Reflect, Light Screen, Spikes, Safeguard, Mist, Toxic Spikes
  and Wish bits are implied by the nonzero turn/layer counts in the mapped fields.
- **FieldConditions.futureSight\*** (E2 batch 5): per battler (`[MAX_BATTLERS]` arrays, the
  battler the move aims at; in singles the side's slot): `future_timer`, `future_damage` (the
  HP the hit takes: the decomp stores it negative), `future_move` and `future_attacker` (the
  attacker's battler as `1 + battler`, 0 = none; convention 3). Field state: they stay when the
  battler leaves, and the hit lands on whatever battler holds the slot (`step._future_sight`).
- **The field-condition pass** (`step.field_conditions`,
  `BattleControllerPlayer_CheckFieldConditions`, battle_controller_player.c:932-1254)
  counts down `reflect`, `light_screen`, `mist`, `safeguard`, `tailwind`, `lucky_chant`,
  a temporary weather's `weather_timer` (the weather ends at 0) and `gravity`; a turn
  count of 0 is the condition's mask bit clear. `wish_timer` (Wish's heal) is not counted
  down yet. `trick_room` is counted down by the turn end's side-condition pass
  (`step._trick_room_check`, bcp.c:1822-1830, E3-2-4; Kaizo re-arms it, `Tables.trick_room_rearm`).
- **movePrev -> `move_prev`** (E2 batch 5): battle-wide, the last move used in the battle:
  `UpdateMoveBuffers` / `UpdateFlagsWhenHit` (bcp.c:3936-3942, :4499-4508) write moveTemp when
  the attack message showed, else 0, on every acting path (`step._use_move`); Copycat reads it
  (`order.copycat_call`). A field condition, not a battler's: it does not leave with a battler.
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
  whose Quick Claw activated this turn); `monSpeedValues`, the
  speeds computed when the turn was ordered, by `BtlCmd_CalcGyroBallPower`
  (`:7127`); and `moveSketched` by `BtlCmd_TrySketch` (`:4848`). M1 is
  unaffected: no M1 mechanic gains an ability mid-battle, Quick Claw's
  activation is `qc_roll` (below; the per-turn flag only shows its animation, a
  `step` event), and none of Thief, Covet, Gyro Ball or Sketch is in M1. M6's
  coverage work must add these fields before the mechanics that read them.
- **Trace, the weather abilities and Truant (E2 batch 3).** `BattleMon.ability` is per
  battler: Trace (`hooks.trace`, battle_lib.c:3718-3737) writes the active battler's
  `ability`, `step._restore_ability` puts the party's back when it leaves (as
  `BattleSystem_InitBattleMon` re-reads it, battle_lib.c:124), and `damage.decomp_abilities`
  gives a Trace holder that traced the copied ability's decomp id (`ids.ABILITY_DECOMP_IDS`). Worry Seed (E3-2-2) writes `ability` = Insomnia the same way; `decomp_abilities` now reads the written ability whenever `ability` differs from the personal row's.
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
  `step.resolve_war` makes it; else 0.
- **The announced flags while a list newcomer waits -> `entry_pending`** (E1 Gardenia review,
  ruling G-32): `traceAnnounced`, `weatherAbilityAnnounced`, `intimidateAnnounced` and
  `downloadAnnounced` are fields since E3-2-5, but one span needs their
  "all still clear" state: a newcomer a replacement's party list sent in
  (subscript_show_party_list.s:22-24) while the player's newcomer of the same list fainted on
  its hazards, so `TurnEnd` re-enters `ReplaceFainted` (battle_controller_player.c:1846-1858)
  before `ShowBattleMon` (:256) runs its switch-in states. `entry_pending[battler]` is 1 then
  (`step.replace(..., effects=False)`), until `search.playout.replacement_pass` runs both
  newcomers' states (`hooks.show_battle_mon`); else 0.
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
  Nature Power (`order.called_move`) and Camouflage (`effects._camouflage`) read it through
  `ids.TERRAIN_MOVES` / `TERRAIN_TYPES` (K3b2 Task 10); an unknown terrain indexes no table
  (`order.terrain_index`: those readers then do nothing).
- **BattleMon.type1/type2 -> `types`**: per party slot, from the personal row; written for
  the active battler by Camouflage (`BtlCmd_TryCamouflage`, `battle_script.c:7997-8022`)
  and restored from the personal row when the battler leaves (a switch, a faint,
  `step._restore_types`), as `BattleSystem_InitBattleMon` re-reads the party's types on the
  next switch-in (`battle_lib.c:114-115`). The Camouflage bit of `moveEffectsMask` stays
  later: only the AI's own Camouflage scoring reads it (`script.s:1023`).
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
  ability; battle_script.c:5342-5348), each slot's PP `min(5, table PP)` (:5364-5370); the party
  Pokemon's own species, form, stats, IVs, ability slot / form, moves and PP wait in the `shadow_*`
  fields (`shadow_moves` / `shadow_pp` with `mimic_slots` all four) until the battler leaves
  (`step._restore_transform`, `_restore_moves`) or the player's battle ends; a fainted one keeps the
  copy until its replacement enters (the exp payout reads its copied species, :2216). A level-up
  reloads the battler's own stats, level and HP, not its moves (battle_lib.c:169-194):
  `levelup.party_identity`. `ai_ability_known` is `battlerAbilities`: the decomp id the AI
  recorded, 0 none, -1 known but not named by the engine (settled to its id by a Transform).
- **chatotCries -> `chatot_cry`** (E1 trial batch, G-70): `BattleSystem.chatotCries[battler]` as
  Chatter's parameter 0 / 1 / 2 (`ids.CHATTER_CHANCES`, sound_chatot.c:193-213), a battle constant
  like `terrain`: every enemy's is 0 (`ChatotCry_New` is invalid, field_battle_data_transfer.c:97,
  chatot_cry.c:13-25), the player's is the save's (`make_state(chatot_cry=...)`, party 0 (`[P]`, battle_system.c:915-924)).
- **BattleMon vs. party moves** (G-68): `moves` / `pp` of the active slot are the BattleMon's;
  Mimic writes only those (battle_script.c:4502-4516), flags the slot in `mimic_slots` and keeps the
  party's move and PP in `shadow_moves` / `shadow_pp`; `BattleMon_CopyToParty` skips flagged slots
  (its `BtlIOCmd_UpdatePartyMon`, battle_io_command.c:581-587), a level-up's learn writes the shadow, and
  leaving (`step._restore_moves`) puts the party's back (`InitBattleMon`, battle_lib.c:80-85).
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
  - `exp_mask` is `sideGetExpMask[0]` (row below), a bitmask over the player's party
    slots: `1 << lead` at the start, then written by every send-out (`step._switch_in`;
    battle_lib.c:1456-1479, battle_script.c:543-635).
  - `learn_level` / `learn_cursor` are the deferred learnset scan: the level whose scan is
    next and the index into it (the decomp restarts `GET_EXP_LEARNSET_INDEX` at 0 for each
    level, battle_script.c:9882). `learn_pin` (the move slot the player selected, when a
    payout reached its active before its move ran) and `exp_refused` (a sticky refusal
    code) are the engine's own bookkeeping. All four are 0 until the payout and the learn
    resolve set them.
  - The search masks some learn-move options (ruling D2, and `learn_pin`, ruling U1):
    replacing the active learner's disabled or encored move, or the slot pinned by
    `learn_pin` (the decomp ends Disable and Encore, and re-reads the pinned slot, after a
    mid-turn learn; the locked and choice-locked moves are read by id and stay offered;
    `levelup` module docstring), and every replacement when the offered move's effect is
    unsupported (ruling D10; coverage then refuses it only where it could auto-learn into an
    empty slot, `coverage.learnable_moves`). They are legal game choices the search skips, so plan
    values at learn nodes are achievable lower bounds (`levelup.learn_options`; the playout
    always gives the move up, ruling D3). The scan, auto-learns and prompts run after `step`,
    before any faint replacement (`levelup.advance_learns` / `learn`,
    `search.learn.resolve_learns`); prompts pending when the battle ends stay (ruling D8).
  - Known limitation (T8 review M1): Stage B's learn node comes after the whole turn's
    outcome (`expand._turn`), but the decomp can prompt mid-turn, at a payout in `MoveEnd`
    (battle_controller_player.c:3968), `CheckFieldConditions` (:941) or `CheckMonConditions`
    (:1296), before a later draw of the same turn such as Shed Skin's
    (`BattleSystem_RandNext(battleSys) % 10 < 3`, battle_lib.c:3567-3570, run from
    `MON_COND_CHECK_STATE_ABILITY`, battle_controller_player.c:1354-1355). The search then
    picks the answer knowing that draw, which the player cannot, so in that case the learn
    node's value is possibly optimistic: it may exceed what the player can achieve. It
    needs a chance event after a mid-turn payout in the same turn; the replayed state
    stays exact, only the choice is better informed than the player's.
- **Not from a struct field.** `pursuit_target` and `wish_hp` are
  `BattleState`'s own bookkeeping: the decomp derives Pursuit's pre-switch hit
  from per-turn scratch (the opposing battler's selected move when a switch is
  chosen) and Wish's heal from the battlers' HP when it lands.

- The mid-turn switch decision (E2 batch 6, U-turn and Baton Pass): `pivot_battler` (per field,
  `1 + battler` owing the party pick at subscript_attack_then_switch_out.s:71 /
  subscript_baton_pass.s:17, else 0) and the turn scratch `ts_*` (per field: the
  `BattleContext` values a resumed turn reads: `turnOrderCounter`, `battlerActionOrder`, the
  command-selection picks (`ts_move`, `ts_struggling`, `ts_locked`, `ts_item`), the per-battler
  `moveEffectsData.custapBerry` (`ts_custap`) and `.quickClaw` (`ts_claw`, B10-6), the per-battler
  `turnFlags` later actors read
  (`ts_protect_ok`, `ts_endure_ok` (`turnFlags.enduring`, Endure, G-65), `ts_taken`,
  `ts_follow_me` (per side, `SideConditions.followMe` / `followMeUser`: `1 + followMeUser`, 0 when `followMe`
  is clear; `BattleSystem_SetupNextTurn` clears it, battle_lib.c:2251-2252, D2-11; written in both
  formats, ruling D2-33: at B = 2 the other side's non-Ghost Curse reads it, lib.c:1714-1717),
  `ts_assurance`: `assuranceDamageMask`, which the action-phase HP updates write in the state
  itself and Assurance reads, B13-17; `turnFlags.roosting` is the public `roosting`, kept at a pause
  and while a faint replacement is pending, `engine.state`; the other turn flags are reset every
  turn and `step` resolves a whole turn: `protecting` is the first actor's `protect_ok` event,
  read by the second actor in `step._execute_actions`), the per-battler `moveFailFlags`
  (`ts_move_failed`), the paused action's `moveCur` and its defender's damage taken (`ts_used`,
  `ts_damaged`, `ts_dealt`), the per-side bitmask of the faints handled (`ts_alive`), `powerMul`
  at the list (`ts_pick_mul`) and the search's turn-start reads (`ts_power_mul`,
  `ts_switched`). All are 0 between turns (`engine.state` docstring): `turnOrderCounter`,
  `battlerActionOrder` and `battlerActions` (the actions as chosen, and the MOVE_END writes of a
  Pursuit, `ts_pursued` (a bitmask of the battlers whose action a Pursuit ended, bit b = battler b; at B = 2 written as 1 + pursuer, the same value), and of UpdateAfterSwitch for a Pokemon Roar drags in, `ts_dragged`, which
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
- The Trace fires (D2-16, ruling D2-5): `ts_trace_fires` (per battler, turn scratch, 0 between turns)
  counts the Traces of the slot's holder in the running step, the index into `Outcome.trace_pick`;
  written at B = 4 only.

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
| `BattleMon.formNum` | `form`, `shadow_form` | mapped |
| `BattleMon.isShiny` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.padding0026_06` | — | not needed because it is padding |
| `BattleMon.ability` | `ability`, `ability_slot`, `ability_form` | mapped |
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
| `BattleMon.friendship` | `friendship` | mapped |
| `BattleMon.nickname` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.curHP` | `hp` | mapped |
| `BattleMon.maxHP` | `max_hp` | mapped |
| `BattleMon.OTName` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.exp` | `exp` | mapped |
| `BattleMon.personality` | — | not needed because its battle uses (nature, ability slot, gender) are resolved into `PokemonSpec` before `make_state` |
| `BattleMon.status` | `status`, `status_counter` | mapped |
| `BattleMon.statusVolatile` | `confusion`, `flinched`, `trapped`, `recharge`, `foresight`, `bide_turns`, `curled`, `locked_move`, `infatuated`, `destiny_bond`, `uproar`, `curse`, `mean_look`, `focus_energy`, `torment`, `transformed` | mapped |
| `BattleMon.OTId` | — | not needed because only obedience and the traded-Pokemon exp bonus read it; the tool assumes an obedient party and resolves the exp bonus into `exp_boost` before `make_state` |
| `BattleMon.heldItem` | `item` | mapped |
| `BattleMon.padding007A` | — | not needed because it is padding |
| `BattleMon.timesDamaged` | — | not needed because no battle or AI logic reads it (only the generic BattleMon getter and setter) |
| `BattleMon.trainerMessageFlags` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.gender` | `gender` | mapped |
| `BattleMon.OTGender` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.capturedBall` | — | not needed because it is cosmetic (names, messages, sprites, animation) |
| `BattleMon.moveEffectsMask` | `leech_seed`, `water_sport`, `mud_sport`, `aqua_ring`, `perish_song`, `yawn`, `lock_on`, `ingrain`, `imprison`, `imprisoned`, `minimized`, `semi_invulnerable`, `power_trick`, `power_trick_desync`, `magnet_rise`, `ability_suppressed` | mapped |
| `BattleMon.moveEffectsTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleMon.moveEffectsData` | `taunt`, `protect_count`, `rollout_count`, `recharge`, `substitute_hp`, `trap_move`, `choice_lock` | mapped |

## MoveEffectsData (nested in `BattleMon.moveEffectsData`)

| Struct.field | pytree field | status |
|---|---|---|
| `MoveEffectsData.disabledTurns` | `disable_turns` | mapped |
| `MoveEffectsData.encoredTurns` | `encore_turns` | mapped |
| `MoveEffectsData.chargedTurns` | — | later (M6) |
| `MoveEffectsData.tauntedTurns` | `taunt` | mapped |
| `MoveEffectsData.protectSuccessTurns` | `protect_count` | mapped |
| `MoveEffectsData.perishSongTurns` | `perish_song_turns` | mapped |
| `MoveEffectsData.rolloutCount` | `rollout_count` | mapped |
| `MoveEffectsData.furyCutterCount` | — | later (M6) |
| `MoveEffectsData.stockpileCount` | — | later (M6) |
| `MoveEffectsData.stockpileDefBoosts` | — | later (M6) |
| `MoveEffectsData.stockpileSpDefBoosts` | — | later (M6) |
| `MoveEffectsData.truant` | `truant`, `fake_out_turn` | mapped |
| `MoveEffectsData.flashFire` | — | later (M6) |
| `MoveEffectsData.lockOnTarget` | `lock_on_target` | mapped |
| `MoveEffectsData.mimickedMoveSlot` | `mimic_slots` | mapped |
| `MoveEffectsData.bindTarget` | — | not needed because in singles the binder is always the opposing active Pokemon |
| `MoveEffectsData.meanLookTarget` | `mean_look_target` | mapped |
| `MoveEffectsData.lastResortCount` | — | later (M6) |
| `MoveEffectsData.magnetRiseTurns` | — | later (M6) |
| `MoveEffectsData.healBlockTurns` | — | later (M6) |
| `MoveEffectsData.embargoTurns` | `embargo` | mapped |
| `MoveEffectsData.canUnburden` | `can_unburden` | mapped |
| `MoveEffectsData.metronomeTurns` | `metronome_turns` | mapped |
| `MoveEffectsData.micleBerry` | `micle` | mapped |
| `MoveEffectsData.custapBerry` | — | later (M6) |
| `MoveEffectsData.quickClaw` | `ts_claw` | mapped |
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
| `MoveEffectsData.lastResortMoves` | — | later (M6) |
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
| `SideConditions.knockedOffItemsMask` | `knocked_off` | mapped |
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
| `BattleContext.fieldConditionsMask` | `weather`, `trick_room`, `gravity`, `uproar`, `acid_rain` | mapped |
| `BattleContext.fieldConditions` | `weather_timer`, `future_timer`, `future_damage`, `wish_timer` | mapped |
| `BattleContext.sideConditionsMask` | `stealth_rock`, `tailwind`, `lucky_chant`, `future_sight` | mapped |
| `BattleContext.sideConditions` | `reflect`, `light_screen`, `mist`, `safeguard`, `spikes`, `toxic_spikes` | mapped |
| `BattleContext.turnFlags` | `roosting`, `ts_assurance` | mapped |
| `BattleContext.selfTurnFlags` | — | not needed because it is reset at every action's end (`BattleContext_Init`, bcp.c:3998) and `step` resolves a whole action; `focusItemActivated` and `SELF_TURN_FLAG_PLUCK_BERRY` live in `_use_move`'s hit loop (`effects.MoveHit`); `lightningRodActivated` / `stormDrainActivated` are `_use_move`'s local `redirected` |
| `BattleContext.moveFailFlags` | — | not needed because it is reset every turn and `step` resolves a whole turn (its only reader, the Uproar check, reads the turn's `move_failed` event) |
| `BattleContext.aiContext` | `ai_seen_moves`, `ai_item_known`, `ai_ability_known`, `trainer_items`, `trainer_item_count` | mapped |
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
| `BattleContext.switchedPartySlot` | — | not needed because the switch target comes from the action (or, for Roar's drag, from `Outcome.roar_target` inside `step._use_move`) and is written to `active` inside `step` |
| `BattleContext.aiSwitchedPartySlot` | — | not needed because the switch target comes from the action and is written to `active` inside `step` |
| `BattleContext.battlerActions` | `ts_actions`, `ts_pursued`, `ts_dragged`, `chosen_target` | mapped |
| `BattleContext.battlerActionOrder` | `ts_order` | mapped |
| `BattleContext.monSpeedOrder` | `war_swaps` | mapped |
| `BattleContext.monSpeedValues` | — | later (M6) |
| `BattleContext.ioQueue` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.ioBuffer` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleScript` | — | not needed because it is script-interpreter, IO or UI state; `step` is one pure function per turn |
| `BattleContext.battleMons` | `active` | mapped |
| `BattleContext.moveTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.moveCur` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.movePrev` | `move_prev` | mapped |
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
| `BattleContext.metronomeMove` | `metronome_move` | mapped |
| `BattleContext.storedDamage` | `bide_damage` | mapped |
| `BattleContext.lastHitByBattler` | `last_hit_by` | mapped |
| `BattleContext.battlerSpeedTemp` | — | not needed because it is per-action scratch that `step` recomputes inside the turn |
| `BattleContext.battlersSwitchingMask` | — | not needed because it is reset every turn and `step` resolves a whole turn |
| `BattleContext.levelUpMons` | — | not needed because only the battle's end reads it (`BattleMain_CopyBattleSysToDTOAndFree`, battle_main.c:723, for evolution), after the fight being searched |
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
