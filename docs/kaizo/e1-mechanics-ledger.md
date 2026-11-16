# E1 mechanics ledger

Rulings for the engine mechanics stream E1 builds (both engines, decomp-cited). Each bullet
names the decomp sites, what the engine does, and where. E2's rulings are in
`e2-mechanics-ledger.md`.

## Gardenia batch (G-n rulings)

- G-1 Focus Band (`HOLD_EFFECT_MAYBE_ENDURE`, `ItemId.FOCUS_BAND`, param 10 in both games:
  `ids.FOCUS_BAND_PARAM`, checked by `game_constants`). The roll is
  `BattleSystem_RandNext() % 100 < itemPower` at `BattleControllerPlayer_UpdateHP`
  (battle_controller_player.c:3386-3389) and at `BtlCmd_CheckHoldOnWith1HP`
  (battle_script.c:9098-9124, the only other site; its scripts are
  subscript_hurt_self_in_confusion.s:16, subscript_pursuit.s:35 and
  subscript_future_sight_damage.s:28). The `&&` tests the item first and then draws, before
  any HP test, so the roll is drawn whenever a holder reaches the site, lethal or not. At
  UpdateHP that is every nonzero damage not taken by a substitute (:3346, the substitute
  returns at :3359-3378), after False Swipe's cap (:3381-3384), while `enduring` is 0 (Endure,
  the move, is unsupported, so always). The band has no full-HP test (unlike the sash at
  :3391), it sets the same `focusItemActivated` / `endure`, and it is never removed
  (subscript_move_followup_message.s:17 removes only `HOLD_EFFECT_ENDURE`); the "hung on"
  message tags the defender's item (`BattleMessage_ItemTag`, battle_script.c:11736-11744), so
  the AI learns a player's band (`effects.reveal_item`). Engine: `Outcome.focus_band[s, i]` is
  the defender's roll at hit `i` of `s`'s move, `Outcome.future_band[d]` the Future Sight hit's;
  each is a Bernoulli of 10/100 (`enumerate._focus_band`). `effects.play_hit`,
  `step._use_move` (self-hit), `step._future_sight`, and the reference engine's `_hit`,
  `_status_disrupted`, `_pursuit`, `_future_sight`.
- G-2 Focus Band on a multi-hit move: one roll per hit (UpdateHP runs per hit). A passed roll
  sets `focusItemActivated`, a self-turn flag that lasts until `BattleContext_Init` at the
  action's end (bcp.c:3998, battle_lib.c:2014), so a roll that passed on a non-lethal earlier
  hit also caps every later hit (as the sash's flag does). An OHKO move sets its damage to
  `-maxHP` before these checks (bcp.c:3342-3344), so the band's roll can save from it too.
- G-3 The `CheckHoldOnWith1HP` sites draw their own roll: the confusion self-hit reads
  `focus_band[s, 0]` (the user's own band; its move does not run after a self-hit), Pursuit's
  pre-switch hit `focus_band[pursuer, 0]` (its action ends there; `effects.play_hit` with hit
  index 0), and the Future Sight hit `future_band[battler]`. The Future Sight subscript's
  message runs under `MOVE_STATUS_ENDURED_ITEM`, which lasts the whole pass, and tags
  `msgBattlerTemp`'s item (bcp.c:1776; subscript_future_sight_damage.s:39-44): the engine now
  reveals that item to the AI there too, before a sash is removed.
- G-4 The flavor berries Figy, Wiki, Mago, Aguav, Iapapa (`HOLD_EFFECT_HP_RESTORE_SPICY` ..
  `_SOUR`, hold effects 14-18; one `ItemId.FLAVOR_BERRY` with `item_param[i, 1]` its `Flavor`,
  `effect_map.FLAVOR_HOLD_EFFECTS`; param 8 in both games). `BattleSystem_TriggerHeldItem`
  (battle_lib.c:4698-4771) and `..._OnStatus` (:5121-5194): at `curHP <= maxHP / 2` (no
  Gluttony here: the pinch berries' halving is not in these cases) the holder heals
  `BattleSystem_Divide(maxHP, itemPower)`; it then dislikes the flavor when
  `Pokemon_GetFlavorAffinityOf(personality, flavor) == -1` (src/pokemon.c:4123-4127,
  `sNatureFlavorAffinities` :61-237, transcribed as `ids.NATURE_DISLIKED_FLAVOR` by nature and
  read back from the decomp by a test). Pluck / Bug Bite (`PLUCK_EFFECT_HP_RESTORE_*`,
  :5678-5751) heal the eater by the same amount whenever it is below full HP, with its own
  nature's dislike. A param below 1 would divide by zero: `DIVISOR_PARAM_MIN` 1.
  `hooks._hp_restore`, `hooks.held_item_check`, `hooks.pluck_berry`, `order.dislikes_berry`;
  the reference's `_held_item`, `_pluck_berry`, `_dislikes`.
- G-5 A disliked flavor berry's confusion: subscript_held_item_dislike_flavor.s runs the heal
  (its `PLUCK_CHECK` removes the berry, subscript_held_item_hp_restore.s:13), then
  subscript_confuse.s with `SIDE_EFFECT_TYPE_HELD_ITEM` (:10-13): the eater's own Own Tempo
  (`CheckAbility`, :6; `ITEM_ABILITY_IDS["Own Tempo"]`, the decomp ability, so it is exact
  whether or not Own Tempo is otherwise built), a confusion already (:7) or Safeguard on its
  side (:8) end the script with no draw; else `Random 3, 2` (:34) sets the counter 2-5. The
  engine stores `ids.CONFUSION_PENDING` (6) and draws the counter at the battler's next
  confusion check (`Outcome.berry_confusion[side]`, `step._use_move`; one check per side per
  turn). This late draw has the decomp's law: the draw is independent of every other, and
  nothing reads more than the confusion bit before the check (the AI's
  `VOLATILE_CONDITION_CONFUSION` reads `> 0`, the observation non-zero; Baton Pass copies the
  counter, a switch-out clears it). It covers the berry checks outside `step` too (the battle
  start's, a faint replacement's switch-in). The enumeration reads the draw where the check
  can see a pending counter (`berry_confusion[s]`: pending at the start, or a foe moving first
  can feed an unconfused holder its disliked berry), and the Tangled Feet threshold variant
  covers a target confused by its berry before the check (`outcome.hit_thresholds`).
  `hooks.berry_confuse`; the reference's `_berry_confuse` and its confusion check.
- G-6 Pluck / Bug Bite on a flavor berry: subscript_pluck.s sets `msgBattlerTemp` to the
  attacker, so the dislike confusion (G-5) is the eater's; at full HP the berry is eaten with
  no effect (`nextSeq == 0`) and `SELF_TURN_FLAG_PLUCK_BERRY` stays set (battle_lib.c:5848).
- G-7 AI: the only trainer AI reads of this family are the flavor-berry tables of
  `Expert_Trick` (script.s:4336-4340, :4373-4377), behind `LoadHeldItemEffect`, which is not
  transcribed, in Trick's branch (Trick's effect is unsupported, so coverage refuses a fight
  with it before the branch can run). Nothing in the AI reads `HOLD_EFFECT_MAYBE_ENDURE`. The
  AI's knowledge of a player's Focus Band comes from G-1's reveal.
- G-8 Pressure's PP cost: `BattleControllerPlayer_DecrementPP` (battle_controller_player.c:
  2235-2296) takes `ppCost` from the chosen slot (at most what is left, :2276-2280; the move still
  runs) for `moveTemp`, the move chosen (an invoker's own range, not the called move's), against
  the defender the fight command picked (bcp.c:1900, `BattleSystem_Defender`). With no defender
  (BATTLER_NONE, :2238) the cost is 1. Imprison (by move id, :2239-2240) and the ranges
  `RANGE_ALL_ADJACENT` / `RANGE_FIELD` (COUNT_ALIVE_BATTLERS_EXCEPT_ME) and
  `RANGE_ADJACENT_OPPONENTS` / `RANGE_OPPONENT_SIDE` (COUNT_ALIVE_BATTLERS_THEIR_SIDE,
  battle_lib.c:2905-2967) add one per living Pressure holder other than the user (singles: the
  foe if alive; the user's own Pressure never counts). `RANGE_USER_SIDE`, `RANGE_USER`,
  `RANGE_USER_OR_ALLY`, `RANGE_ALLY` add nothing. Every other range (single target, the special
  single target of Counter / the invokers / Nature Power, random opponent, Me First) adds one
  when the defender is not the user and its `Battler_Ability` is Pressure (:2261-2266, no HP
  test): so Sunny Day, Spikes, Growl, Earthquake and Imprison cost 2 against a Pressure foe,
  Swords Dance, Reflect, a non-Ghost Curse, Counter and Sleep Talk 1. `skipPressureCheck` is set
  only by script commands (battle_script.c:3521, :4648, :4701, :5395, :7599) that run after the
  decrement, and `BattleContext_Init` clears it before the next action (battle_lib.c:2014), so it
  never changes a cost here. Struggle and a battler locked at command selection spend nothing
  (unchanged). `effects.pressure_pp_cost` / `pressure_holder`, `step._use_move`; the reference's
  `_pp_cost`.
- G-9 Pressure's other PP sites: Pursuit's hit before a switch takes one more PP while any is
  left when the switching battler has Pressure (`BtlCmd_TryPursuit`, battle_script.c:6915-6922;
  a U-turn user's switch goes through the same command), `step._pursuit_hit`, the reference's
  `_pursuit`. A Counter / Mirror Coat / Metal Burst whose command passes runs
  `BattleSystem_DecPPForPressure` (battle_lib.c:7390-7397) on its new defender, the last
  attacker (battle_script.c:4646, :4699, :7180): one more PP from `moveSlot`, the slot chosen
  (Sleep Talk's or Copycat's when it called the move), while that slot has PP left; with the
  special range's DecrementPP cost of 1 a passing Counter costs 2 against a Pressure foe.
  Magic Coat and Snatch (bcp.c:3083, :3107) and Spite are unsupported moves (refused by
  coverage), so their Pressure branches are not built.
- G-10 Pressure's switch-in announcement: `SWITCH_IN_CHECK_STATE_PRESSURE` (battle_lib.c:
  4096-4110) shows subscript_pressure.s for a living holder whose `pressureAnnounced` is clear
  (cleared at entry, :113, and by Transform, battle_script.c:5358, unsupported); its message tags
  `BTLSCR_MSG_BATTLER_TEMP`'s ability (`TAG_NICKNAME_ABILITY`), so `BattleMessage_AbilityTag`
  runs `BattleAI_SetAbility` (battle_script.c:11798-11809): the AI learns a player's Pressure
  (`ai_ability_known`) at the battle start and at its switch-in, as Mold Breaker's
  (`hooks.REVEALS_ON`, the reference's `_switch_in_effects`). The flag itself only stops a
  repeat (STATE.md row); a traced Pressure is known already through Trace's message.
- G-11 AI: the only trainer AI read of Pressure is `Expert_TrumpCard` (script.s:5331-5334,
  `LoadBattlerAbility AI_BATTLER_DEFENDER`), which is not transcribed (Trump Card's effect is
  unsupported, so coverage refuses a fight with it before the branch could run). The AI's
  other PP readers (`LoadCurrentMovePP`, the usable-move mask) read the state, so they see
  Pressure's cost as it happens.
- G-12 Synchronize's pass (`AbilityId.SYNCHRONIZE`, decomp 28; `BattleSystem_SynchronizeStatus`,
  battle_lib.c:4493-4534). With `SYSCTL_TRY_SYNCHRONIZE_STATUS` set (G-13): a defender whose
  `Battler_Ability` is Synchronize and which is the `sideEffectMon` passes to the attacker
  (:4499-4504); else a Synchronize attacker that is the `sideEffectMon` (a Static-like ability of
  the defender statused it) passes to the defender (:4506-4511); either way the receiver becomes
  the `sideEffectMon`, even when nothing passes. The passer's status picks the script
  (:4514-4520): any poison runs subscript_poison.s, so a bad poison passes as a plain poison
  (counter 0); a burn subscript_burn.s; a paralysis subscript_paralyze.s; a sleep or a freeze
  nothing. As `SIDE_EFFECT_TYPE_ABILITY` (:4524) the receiver's own Immunity / Water Veil /
  Limber (the ignorable tests read the receiver's own ability: neither battler of a pass holds
  Mold Breaker in place of the tested one), its Leaf Guard in sun (G-14), any status, a Poison /
  Steel (poison) or Fire (burn) type end it silently (poison :13-16, :32-38, its ABILITY ends
  :72-73, :89-90, :98-100, :107-110; burn :20-23, :39-42, :113-114; paralyze :5-8, :24-25,
  :79-80); it skips CheckSubstitute (poison :28, burn :35, paralyze :20) and Safeguard (the
  ABILITY jumps at poison :39, burn :43, paralyze :26 pass the tests). No script line tests HP: the
  engine sets the status only on a living receiver, since a fainted one's is cleared at its
  faint (`step._faint_check`, battle_controller.c:1482-1486). The message (poison :57, burn :68,
  paralyze :44) tags `BTLSCR_MSG_BATTLER_TEMP`'s ability, the passer: the AI learns a player's
  Synchronize (`hooks.REVEALS_ON`, phase `SYNCHRONIZED`); a blocked pass is silent. A landed
  pass toggles the flag off. `effects.synchronize`; the reference's `_synchronize`.
- G-13 The Synchronize flag and `sideEffectMon`. Every poison, bad poison, burn and paralysis
  subscript that lands toggles `SYSCTL_TRY_SYNCHRONIZE_STATUS` (set when clear, cleared when
  set: subscript_poison.s:63-69, _badly_poison.s:83-88, _burn.s:79-84, _paralyze.s:50-56);
  sleep and freeze never touch it. `BattleContext_Init` clears it at every action's end
  (battle_lib.c:2009), U-turn before its party list (subscript_attack_then_switch_out.s:69).
  `BattleSystem_SynchronizeStatus` is called only from `AFTER_MOVE_EFFECT_SYNCHRONIZE_STATUS`
  (bcp.c:3665-3670), after every hit of a move whose AfterMoveEffects run, before the switch-in
  effects (:3672) and the item checks. Within a hit the `sideEffectMon` is the defender for the
  secondary effect (TRIGGER_SECONDARY, bcp.c:3484) or a status move, the attacker for the contact
  ability (`BattleSystem_TriggerAbilityOnHit`, battle_lib.c:4225, :4305, :4323), then the defender
  again when `CheckExtraFlinch`'s own tests pass (bcp.c:4636-4643: the attacker's King's Rock,
  its roll, the move's flag, a damaged defender with HP; before subscript_flinch_mon.s's tests).
  So a secondary status on the defender and a contact status on the attacker in one hit toggle
  the flag back: nothing passes. Paths: every move status (direct, secondary, Tri Attack, the
  fangs, Secret Power), the contact abilities, Roar's drag into Toxic Spikes (the hazards run
  in the move script; `BtlCmd_CheckToxicSpikes` makes the newcomer the `sideEffectMon`,
  battle_script.c:7640-7641; the pass to the Roar user comes before the switch-in effects, so a
  Pecha Berry cures the newcomer after it), and the resumed U-turn / Baton Pass tail (the list's
  Toxic Spikes poison sets the flag; a Baton Pass's defender is the user, so its pass aims at the
  already poisoned newcomer and ends). Pursuit's hit runs no AfterMoveEffects
  (subscript_pursuit.s), and a turn-end orb status or a switch action's Toxic Spikes is cleared by
  the next `BattleContext_Init` before any check: no pass. A passer that fainted in between
  (the U-turn user that falls to its foe's Pursuit, a newcomer that faints to Spikes / Stealth
  Rock after Toxic Spikes poisoned it) is G-24.
  `step._use_move` (the hit loop's `sync_flag` / `sync_sem` carry from `effects.sync_landed`, the
  step before the switch-in effects for a drag or a tail), `step._act` (`entry_poisoned`); the
  reference's `_status_landed` (in `_inflict`, `_contact_ability`, `_entry_hazards`), the King's
  Rock sites of `_hit`, `_u_turn`, `_resume`, and the calls in `_fight`, `_roar`, `_resume`.
- G-14 Leaf Guard (`AbilityId.LEAF_GUARD`, decomp 102). Every status subscript but
  subscript_freeze.s tests `CheckIgnoreWeather` (a living Cloud Nine / Air Lock holder skips the
  test, battle_script.c:9195-9206), then `FIELD_CONDITION_SUNNY`, then the side-effect battler's
  Leaf Guard, right after the status's own ability test: subscript_fall_asleep.s:25-27,
  subscript_poison.s:14-16, subscript_badly_poison.s:32-34, subscript_burn.s:21-23,
  subscript_paralyze.s:6-8 (`CheckIgnorableAbility`: an attacker's Mold Breaker pierces it). A
  direct effect (a status move) prints "prevents ..." / "stayed awake because of its {1}"
  (`TAG_NICKNAME_ABILITY`: poison :_177/_201, badly_poison :_249/_266, burn :_264, paralyze
  :_170, fall_asleep :_202/_221), which the AI learns (phase `STATUS_BLOCKED`); a secondary
  effect (INDIRECT) and an ABILITY side effect end silently. `effects._inflict` (before Shield
  Dust), `effects._contact_ability` (the attacker's own Leaf Guard against Static, Poison Point,
  Flame Body), `effects.synchronize`; the reference's `_inflict`, `_contact_ability`,
  `_synchronize`. The reads E2 built by decomp id stay as they are: Toxic Spikes (B4-22,
  `CheckAbility`, :5-10), the status orbs (the HELD_ITEM paths, subscript_badly_poison.s:13-17
  and subscript_burn.s:5-9, end silently), Yawn's drowsiness (subscript_yawn.s:7-9) and its
  later sleep (subscript_fall_asleep.s:9-11, the MOVE_EFFECT path, `CheckAbility`). A blocked
  direct status ends with `MOVE_STATUS_NO_MORE_WORK`, which skips AfterMoveEffects; as for
  Limber's and Insomnia's blocks the engine does not mark it: nothing landed, so the Synchronize
  step has no flag, and no item check there finds anything a status move could have changed.
- G-15 Rest: subscript_rest.s:7-11 tests Insomnia, Vital Spirit, a sleep and Soundproof /
  an uproar only: Leaf Guard never stops Rest. Rest is an unsupported move (coverage refuses it),
  so nothing is built.
- G-16 AI: the trainer AI reads `ABILITY_LEAF_GUARD` at `Basic_CheckCannotPoison`
  (script.s:534-536, the defender's, with `LoadCurrentWeather` sunny), `Basic_CheckSunnyDay`
  (:808, the attacker's) and `Expert_SunnyDay` (:3789-3790, the attacker's with `IfStatus ANY`:
  the decomp's inverted test is kept), all transcribed (`policy._basic_check_cannot_poison`,
  `_basic_check_sunny_day`, `_expert_sunny_day`) through `commands.load_battler_ability`'s
  decomp ids; nothing refuses. `TagStrategy` (:6797, :6824) is doubles only. No script or
  trainer_ai.c code reads `ABILITY_SYNCHRONIZE`. The AI's knowledge of a player's Leaf Guard or
  Synchronize comes from their messages (G-12, G-14) and from the guess otherwise.
- G-17 Enumeration (`enumerate.py`): Synchronize can paralyze mid-turn, so `full_para` is read
  for a side whose faster foe's Synchronize holder made a contact hit on the side's Static
  (`para_inflict`), and the end-of-turn speed grid frees the paralysis of a side that paralyzes
  a foe holder or whose Static meets a foe holder's contact hit (`para_free`). The damage classes
  get the synchronize clauses: a foe holder's contact hit on the side's Static / Poison Point /
  Flame Body before the side's hit, and the side's later hits once its move can status a foe
  holder, where the passed status matters (a burn on a physical non-Guts hit, any under Guts).
  Leaf Guard is input-neutral (it only stops statuses).
- G-18 Wonder Guard (`AbilityId.WONDER_GUARD`, decomp 25: Shedinja; Kaizo also Unown). Battle:
  `BattleSystem_ApplyTypeChart` (battle_lib.c:2649-2654), run by `CheckTypeChart` (bcp.c:
  2350-2370: a move with power that is not RANGE_USER / RANGE_USER_SIDE, not under
  SYSCTL_IGNORE_IMMUNITIES (Future Sight), hit, miss or Protect alike, bcp.c:3271-3277), sets
  `MOVE_STATUS_WONDER_GUARD` when `Battler_IgnorableAbility` finds Wonder Guard (an attacker's
  Mold Breaker hides it), `MoveIsOnDamagingTurn` holds, the walk left
  `MOVE_STATUS_SUPER_EFFECTIVE` clear (the two effectiveness bits never stand together, so the
  BASIC_EFFECTIVENESS clause never applies) and the move has table power; Struggle returns
  first (:2573). So neutral, resisted and immune hits are guarded, the fixed-damage moves
  (power 1: Night Shade, Seismic Toss, Dragon Rage, SonicBoom, Super Fang, Endeavor, the OHKO
  moves) and the Counter family too, status moves (Thunder Wave included: power 0) never, Bide
  never (RANGE_USER: no chart), Future Sight never (no chart at the use, and its later hit runs
  no chart). The flag is in `MOVE_STATUS_DID_NOT_HIT` / `_NO_EFFECTS` / `_DID_NOT_AFFECT` /
  `_IMMUNE` (moves.h:81-88): the move ends in subscript_missed.s (bcp.c:3324-3327), so no damage
  and no secondary effect, as an immunity (Brick Break's screens still break: DID_NOT_AFFECT,
  battle_lib.c:1499-1500); it is not `MOVE_STATUS_INEFFECTIVE`, so `moveFailFlags.noEffect`
  stays clear (bcp.c:2368). `MoveIsOnDamagingTurn` (battle_lib.c:7588-7604) lists Bide, the
  charge moves and BATTLE_EFFECT_FLINCH_BURN_HIT (Fire Fang, the decomp's bug note :7600):
  they are on their damaging turn only under SYSCTL_LAST_OF_MULTI_TURN, which only
  subscript_charge_move_cleanup.s:7 and subscript_bide_end.s:12 set and BattleContext_Init
  clears (SYSCTL_INIT_P1, system_control.h:31-34): Fire Fang is never guarded. Ice Fang (274)
  and Thunder Fang (275) are not listed. Engine: `damage.wonder_guarded` (by the decomp effect,
  `tables.move_ai_effect`; `on_damaging_turn`, `super_effective_flag`, `chart_walk`), zeroing
  the hit in `_finish`, `fixed_damage`, the Counter family in `effects.play_hit`; the reference's
  `_wonder_guarded`, `_move_damage`, `_fixed_damage`, the counter branch.
- G-19 What the guarded move still does: subscript_missed.s unlocks the attacker
  (`Battler_UnlockMoveChoice`; a Rollout / Uproar lock ends as on an immunity, through
  `_Hit.effective`, subscript_missed.s:97) and runs subscript_crash_on_miss.s (:98): a guarded Jump Kick crashes for
  min(Divide(battleCtx->damage, 2), Divide(defender maxHP, 2)), where `battleCtx->damage` is
  CalcDamage's through the chart walk, which Wonder Guard does not zero (only an immune entry
  does): `step._use_move` (`guarded`, `damage_for_roll(..., wonder_guard=False)`), the
  reference's crash branch. CheckMoveFailure's DID_NOT_HIT path skips AfterMoveEffects
  (`items_full`; the reference's `continue`). PP, `movePrevByBattler`, `moveHit` and the Choice
  lock are written as for any move that showed its attack message (UpdateMoveBuffers,
  UpdateFlagsWhenHit). The enumeration keys a crash move's damage classes on the unzeroed table
  (`enumerate.outcome_laws`, `hit_damage_table(..., wonder_guard=False)` for `EffectId.CRASH`).
  Wonder Guard is input-neutral (a fixed factor through the chart, whose inputs have clauses);
  Download is a switch-in clause ability.
- G-20 The AI learns the holder's ability: subscript_missed.s:53-57 prints "{0} avoided
  damage by using {1}!" (`TAG_NICKNAME_ABILITY`, BTLSCR_DEFENDER: `BattleMessage_AbilityTag`,
  battle_script.c:11798-11806, `BattleAI_SetAbility`) unless an earlier flag prints first:
  lost focus (:8-10), protected (:16-32), failed (:47-51). A plain miss reaches it (the miss
  message is :78-84, after it), and an immune Tackle on Shedinja too (WONDER_GUARD is tested at :54,
  before INEFFECTIVE at :60). `hooks.DAMAGE_AVOIDED` (`step._use_move`, `step._pursuit_hit`: Pursuit's
  hit on a switching holder goes to subscript_missed.s through subscript_pursuit.s:17, :75-77);
  the reference's `guarded` reveal.
- G-21 Download (`AbilityId.DOWNLOAD`, decomp 88: Porygon, Porygon2, Porygon-Z).
  `SWITCH_IN_CHECK_STATE_DOWNLOAD` (battle_lib.c:3814-3851), after the Intimidate state: a
  living holder (`Battler_Ability`) whose `downloadAnnounced` is clear sums over the opposing
  battlers with HP and no `VOLATILE_CONDITION_SUBSTITUTE` their Defense and Sp. Def, each
  `stat * sStatStageBoosts[stage]` (the raw stage, no Simple); the flag is set whatever the
  sums (:3836); a nonzero total raises Sp. Attack when Defense >= Sp. Def, else Attack, by one
  through subscript_update_stat_stage.s as an ABILITY side effect (`BtlCmd_ChangeStatStage`,
  battle_script.c:2841-2873: at +6 it jumps silently). The "raised its" message is C-built
  (`SetupNicknameAbilityStatMsg`, :2769-2776, :2854): no AI reveal. `hooks.download` from
  `switch_in_effects` (the newcomer, and a battler that just traced Download) and
  `battle_start` (its own loop over the battlers, after Intimidate's); the reference's
  `_download`. No `downloadAnnounced` field (STATE.md): the hook runs exactly when it is clear.
- G-22 Download at a double KO: as Intimidate (playout.replacement_pass), the player's newcomer
  meets a fainted foe at its entry (no sum: nothing rises; the decomp's flag is only set at
  the later ShowBattleMon check, when both newcomers are in), and fires once the opponent's
  side is settled, reading the opposing newcomer at stage 0 (`download(..., staged=False)`:
  a fresh battler, whose only possible Defense / Sp. Def change by then is its own pinch
  berry at the held-item state, which comes after the Download state). The opposing
  newcomers' own Downloads read the player's newcomer with its Defense / Sp. Def stages set
  aside to 0 for their entries and put back after (nothing an opposing switch-in does touches
  them).
- G-23 AI (`ai/switching.py`, `ai/commands.py`, `ai/replacement.py`; the refusals
  "AI switching: ability Wonder Guard" and "AI scoring: ability Wonder Guard changes
  IfMoveEffectivenessEquals" are lifted). Every AI `ApplyTypeChart` call sees the true
  ability (Battler_IgnorableAbility of battleMons; the AI's knowledge is not read) and runs at
  command selection or at the replacement, after BattleContext_Init: Fire Fang and the listed
  effects are never on their damaging turn there. `switching.wonder_guard_bit` adds the bit to
  `MOVE_STATUS_IMMUNE`: `IfMoveEffectivenessEquals` (trainer_ai.c:1339; script.s:68
  Basic_CheckForImmunity's -10 and Basic_CheckWonderGuard's -12 at :76/:96-99, both already
  transcribed, then take their decomp paths; :577 Basic_CheckNonStandardDamageOrChargeTurn
  likewise), `TrainerAI_CalcDamage` (:3106, `damage._finish` / `ai_damage_for_roll`'s fixed
  cases) and `BattleAI_PostKOSwitchIn` stage 2 (battle_lib.c:8075). `TrainerAI_ShouldSwitch`
  step 4, `AI_CannotDamageWonderGuard` (trainer_ai.c:3286-3348, called at :3943): singles, the
  player's `battleMons[].ability` Wonder Guard (no Mold Breaker test), no move of the AI's active
  super effective through ApplyTypeChart; then each bench Pokemon's moves in party order through
  CalcEffectiveness, and each super-effective one draws `RandNext() % 3 < 2` (43691 / 65536),
  the first success switching to it: `SwitchSite.WONDER_GUARD` sites before the keep draws
  (`outcome._SWITCH_*`: residues 0-1 of 3), mapped to the switch by `switch_action`. Wonder
  Guard never clears the super-effective bit, so the keep draws and the passes' second calls
  are unchanged; the passes' first call has the AI's bench Pokemon as the defender
  (trainer_ai.c:3769-3777), where a bench Wonder Guard makes a non-super-effective moveHit
  INEFFECTIVE (battle_lib.c:2734-2739) unless the player's active has Mold Breaker. The sites
  stay within `KAIZO_SWITCH_SITES` (module docstring). `Expert_ChangeUserAbility`'s table
  (script.s:4441) belongs to Role Play / Skill Swap, unsupported moves (refused with their
  user).
- G-24 Synchronize from a fainted passer. A faint zeroes only the party copy of the status
  (`BattleMon_CopyToParty`, battle_controller.c:1481-1486); `BattleMon.status` stays, and
  `BattleSystem_SynchronizeStatus` reads it (battle_lib.c:4514-4520) with no HP test of the
  passer. (1) U-turn's Pursuit: a user that falls to it ends subscript_attack_then_switch_out.s
  at :55, before the flag clear at :69, so its AfterMoveEffects' Synchronize step
  (bcp.c:3665-3670) reads the head's flag and `sideEffectMon` as the Pursuit's contact ability
  left them (subscript_pursuit.s:40 / :86, TriggerAbilityOnHit: the pursuer becomes the
  `sideEffectMon`), after CheckExtraFlinch made the defender it (bcp.c:3524, :4636-4643: the
  fainted user's King's Rock, its roll, the move's flag, a damaged defender with HP). So a
  Synchronize user that Static paralyzed and Pursuit fainted passes the paralysis to the
  pursuer. (2) A U-turn / Baton Pass newcomer that Toxic Spikes poisoned and Spikes / Stealth
  Rock fainted in the list (subscript_show_party_list.s:22-24) passes the poison in the resumed
  tail; a Roar's drag already did (no faint check between its hazards and the step).
  `effects.synchronize`'s `held` (each battler's `BattleMon.status`); `step._use_move`
  (`pursuit_sync`: the head's flag and sideEffectMon through `ctl`, `_execute_actions`' loop
  carry and the Pursuit's toggle; `entry_status`: the newcomer's status before the list's faint
  check, from `_act`); the reference's `_Turn.battle_status` (written by `_faint_check`, read by
  `_synchronize` for a fainted passer) and `_u_turn` (the flag goes on through the Pursuit).
- G-25 Trace in both parties (lifts B3-14's refusal `coverage.trace_standoff_line`). A Trace
  holder facing Trace stays armed: `ChooseTraceTarget` (battle_lib.c:7750-7775) skips a Trace
  target, so `traceAnnounced` stays clear (:3726-3733) and every later
  `SWITCH_IN_CHECK_STATE_TRACE` (:3718) tries again. In singles neither can copy the other, so
  there is no order dependence and no draw; the holder copies the next battler its side faces,
  at that battler's switch-in checks: the foe's switch action, U-turn / Baton Pass pick, Roar
  drag or replacement (`hooks.switch_in_effects` already runs Trace for the other battler, in
  both engines). New here: the turn's chance laws read the copy. `outcome.pivot_view` (every law
  and the sampler read through it) gives the staying side's active Pokemon the copy
  (`outcome._armed_trace`) where its foe enters this turn (a switch action, or the paused
  pivot's pick) and `damage.trace_copy` passes on the newcomer (its own ability, not Forecast /
  Trace / Multitype; it has HP after its entry hazards, the check reads the target's `curHP`,
  :3730; `trace_copy`'s new `target` argument), so its crit odds, accuracy, secondary chance,
  armor and speed reads are the copy's. Not where its Pursuit fires on that switch
  (`step.pursuit_fires`: the hit comes before the switch, subscript_switch_pokemon.s:11, and
  ends its action, so those reads see Trace). A Roar drag comes after the holder's own move
  (the end-of-turn masks keep a Trace holder's reads, as before). `coverage.trace_standoff_line`,
  its `check_coverage` call and prep's per-candidate uses (`prep.problem.usable_pool`,
  `prep.pool.check_pool`) are deleted.
- G-26 Two switch-in weather abilities at war (lifts B3-15's refusal
  `coverage.weather_war_line`). `SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES` loops over
  `monSpeedOrder` (battle_lib.c:3740-3790): two battlers that run it together with Drizzle /
  Sand Stream / Drought / Snow Warning setting different weathers each set their own (the
  other's permanent weather is not theirs), so the second one's stays and both scripts run.
  Only two newcomers meet in one weather state: the leads (`ShowBattleMon`,
  battle_controller_player.c:256) and a double KO's replacements (the list runs only the
  hazards, subscript_show_party_list.s:22-24; `TurnEnd` then goes to `TrainerMessage` and
  `ShowBattleMon`, :1858, :251-263); an active battler announced its own at its entry. The
  order is `BattleSystem_SortMonSpeedOrder` at `TrainerMessage` (:251; battle_lib.c:3364-3382):
  `CompareBattlerSpeed(0, 1, TRUE)` puts the player second (`order.player_second`):
  `ignoreQuickClaw` keeps the priorities 0, but a Quick Claw's `speedRand` bit still activates it
  (:1278-1286) and an exact tie swaps on `RandNext() & 1` (:1445); a fainted battler goes last
  (:1205-1211). The sort reads the battlers as sent out: at the battle start no weather yet (the
  start condition is set inside the switch-in state machine, subscript_overworld_rain.s etc.)
  and the `speedRand` of `BattleContext_InitCounters` (:2024, the root's `qc_roll`); at the
  replacements the turn's `speedRand` (`qc_roll`; `speedRand` is redrawn only at the next turn's
  pre-move step, bcp.c:853-857) and each newcomer's status and item as it entered, Speed stage
  0, the turn end's weather (`playout._sort_view`). The engine runs the player's weather ability
  first; `hooks.order_swaps` gives the sort's swap for each (tie, qc) draw, `hooks.hold_war`
  settles a war the draws to come cannot change (`hooks.swap_war`: the player's weather,
  permanent, its script revealing the ability to the AI) and otherwise stores the bits in the
  new `BattleState.war_swaps` (STATE.md `monSpeedOrder`). `step.resolve_war` settles it on the
  state's `qc_roll` and a tie bit: at the root chance node (`step.sample_root` draws the tie from
  `fold_in(key, 1)`; `step.root_belief` splits the tie 1/2-1/2: those states differ in the
  visible weather but stay one root belief node, so the root action is chosen over them, a lower
  bound), in the tree's replacement children (`expand.expand_replacements`: two children at 1/2
  where the tie decides; `stage_b` answers hold a list of children) and in the playout's next
  turn before its decisions (`playout.settle_war` on `fold_in(k_turn, WAR_FOLD)`, in
  `playout._run`, `player._player_run` and their replays). `hooks.battle_start`,
  `playout.replacement_pass`. `coverage.weather_war_line`, its `check_coverage` call and prep's
  per-candidate uses are deleted. The reference engine has no battle start or replacement pass
  (only `ref_step` / `ref_replace`), so nothing mirrors there. Not modelled (an engine gap, not
  a refusal): when the player's newcomer faints on its entry hazards at a double KO, the decomp
  runs the next player newcomer's and the opponent's newcomer's switch-in states together after
  the next pass, while the engine has run the opponent's at its entry (its weather, and its
  Intimidate against the fainted battler), so that pair is not ordered.

## The merge with E2's charge turn (B10-12, B10-13)

- G-27 Wonder Guard and the charge moves (amends G-18's "no supported move sets it"). The
  charge turn sets SYSCTL_FIRST_OF_MULTI_TURN (effect_script_0039.s:9 and the other charge
  scripts), so `CheckTypeChart` does not run (bcp.c:2356): no Wonder Guard flag and no reveal on
  that turn. Every path of a charge script that does not charge (the locked second turn, the
  Power Herb's skip, SolarBeam in sun) ends in subscript_charge_move_cleanup.s, which sets
  SYSCTL_LAST_OF_MULTI_TURN (:7), and the move script runs before TryMove (bcp.c:3214-3216,
  :3272), so `MoveIsOnDamagingTurn` (battle_lib.c:7588-7604) holds at the strike: a strike that
  is not super effective is guarded (no damage, subscript_missed.s, the reveal). Bide's
  subscript_bide_end.s:12 sets the flag too, but Bide is RANGE_USER (no chart); Fire Fang is never
  on it. The trainer AI reads after `BattleContext_Init` (battle_lib.c:2009, bcp.c:1855, :3998),
  so in every AI read the charge moves stay off their damaging turn (G-23 unchanged). Engine:
  `damage.charge_strike` (MoveFlag.CHARGE), `wonder_guarded(..., last_of_multi_turn)`, passed by
  `_finish` for a battle hit (not for the AI's `variance_last` estimate) and by
  `step._use_move`'s `guard_tested` (with `~charging`); the reference's `_wonder_guarded` (a
  charge move's chart only runs at its strike: `_use_move` returns before it on the charge turn).
- G-28 A semi-invulnerable miss on a Wonder Guard holder: the chart still runs after
  `CheckMoveHitOverrides` (bcp.c:3263-3277), but subscript_missed.s:12-14 sends
  MOVE_STATUS_SEMI_INVULNERABLE straight to the miss message (:78-84), past the Wonder Guard
  message (:53-57): no reveal (`step._use_move`'s `reveal_ability(..., ~semi_miss)`, the
  reference's `guarded` reveal). The miss still crashes a Jump Kick (subscript_missed.s:96-98:
  `CRASH_ON_MISS` on every path), as E2's B10-12 has it; the merged reference keeps one
  `blocked` (miss, semi-invulnerable miss, Protect) for the crash and the return.
- G-29 Pressure and the charge moves: the charge turn's (or the Power Herb turn's)
  `BattleControllerPlayer_DecrementPP` takes 1 plus the Pressure foe's (bcp.c:2235-2280); the
  locked second turn cannot pick a command (`Battler_CanPickCommand`, battle_lib.c:6224-6235:
  VOLATILE_CONDITION_MOVE_LOCKED), so `ppDecremented` is set (bcp.c:334-335) and it spends
  nothing (the engines' `locked`). No code change; a hand test pins it. Focus Band, Synchronize,
  Leaf Guard and Download need nothing: the charge turn and a semi-invulnerable miss deal no
  damage and inflict no status (no UpdateHP roll, nothing to pass or block), and Download's sums
  (battle_lib.c:3814-3851) read HP and the substitute only, not MOVE_EFFECT_SEMI_INVULNERABLE.
- Merge note: `ItemId.POWER_HERB` (E2) took 51; `FOCUS_BAND` / `FLAVOR_BERRY` are now 52 / 53
  (the tables map by name; rebuilt). E1 prep P4's `Expert_ChargeTurnNoInvuln` (SolarBeam only),
  its second `IfHeldItemEqualTo` and the duplicate `SKIP_CHARGE_TURN_IN_SUN` /
  `ITEM_POWER_HERB` / outcome-table entries gave way to E2's transcription of every entry.

## The Gardenia review fixes (2026-11-15)

- G-30 Present's heal path and Wonder Guard. effect_script_0122.s `_004` sets
  SYSCTL_IGNORE_IMMUNITIES (:13) in the move script (bcp.c:3214), before TryMove's
  `CheckTypeChart` (:3272), which then skips the chart (bcp.c:2350-2356): no
  MOVE_STATUS_WONDER_GUARD, no subscript_missed.s, no reveal, and AfterMoveEffects run (Synchronize,
  the item checks); likewise no `moveFailFlags.noEffect` (a heal of a Ghost). The damage path is a
  Normal hit, never super effective: guarded as before (G-18). The script runs hit or miss, so a
  miss on a Wonder Guard holder reveals it only on the damage path. Engine: `step._use_move`'s
  `heal_path` (out of `guard_tested` and `no_effect`), the reference's `_present_heals`; the exact
  enumeration reads the Present draw on a miss too against a side with a Wonder Guard Pokemon
  (`enumerate.Reads.present_guard`).
- G-31 Synchronize to a fainted receiver. The receiver's status test is the status subscript's
  `CompareMonDataToValue ... BATTLEMON_STATUS` (subscript_paralyze.s:41-42; poison, burn alike) on
  `BattleMon.status`, which a faint keeps (only the party copy is zeroed, battle_controller.c:
  1481-1486). So a statused receiver that fainted (the AI's burned Static U-turn user KOed by the
  player's Synchronize Pursuit user, subscript_pursuit.s:86) takes nothing and nothing is
  revealed; an unstatused fainted receiver still "takes" it (the message tags Synchronize, the
  status is not kept on the fainted Pokemon, as before). `effects.synchronize`'s `held` serves the
  receiver too (`can_have_status(..., current)`); the reference's `_synchronize` reads
  `_Turn.battle_status` for a fainted receiver.
- G-32 A double KO whose player newcomer faints on its hazards (closes G-26's "not modelled"
  note). The list runs only the hazards and a faint check (subscript_show_party_list.s:22-24);
  `TurnEnd` re-enters `ReplaceFainted` for the player (bcp.c:1846-1858) before `TrainerMessage`
  and `ShowBattleMon` (:251-263), so the opposing newcomer's switch-in states (weather,
  Intimidate, Download, Trace, the announcements, its held-item check) all wait and run with the
  player's next newcomer's, state by state over `monSpeedOrder` (battle_lib.c:3649-4168).
  Engine: `step.replace(..., effects)` (the list alone, new `BattleState.entry_pending[battler]`,
  STATE.md note; the reference's `ref_replace(..., effects)` mirrors it); `hooks.show_battle_mon`
  (the state machine `battle_start` now shares, with the weather war's order from the sort at
  TrainerMessage, `hold_war` on the turn's `qc_roll`); `search.playout.replacement_pass` enters
  the opposing newcomer without its states when the player's newcomer fell, and at the next
  pass enters the player's newcomer with the list alone and runs both through
  `show_battle_mon`; a newcomer that falls again leaves it waiting. The Intimidate and weather of
  the waiting newcomer now meet the player's second newcomer.
- G-33 The Kaizo switch-site bound. `AI_CannotDamageWonderGuard`'s draws (trainer_ai.c:3338,
  one per bench (Pokemon, super-effective move): 20) exist only when the active has no
  super-effective move (:3300-3311), so never beside a keep draw (4); a bench Pokemon can be in
  both passes (an immune row then a resisted row, battle_lib.c:2762-2781, or Wonder Guard's
  INEFFECTIVE, :2734-2739, which never clears NOT_VERY_EFFECTIVE): 40. So
  `KAIZO_SWITCH_SITES = max(4, 20) + 40 = 60` (was 44, which a player's Wonder Guard against
  Flying/Bug bench Pokemon exceeded: a "count > sites" refusal). The switching docstring's "each
  bench Pokemon is in at most one pass" was wrong and is corrected;
  `test_kaizo_switch_sites_reach_sixty` builds the 60. No new refusal; 61 switch stops stay far
  inside `WIDTH_CAP`.
- G-34 The battle-start weather war's tie is a chance node the player sees (amends G-26's root
  note). The tie is drawn at TrainerMessage's sort (battle_controller_player.c:251,
  battle_lib.c:1445) and the weather shows before command selection, so Stage B's root groups
  `step.root_belief`'s states by what the player sees (`observe.replacement_observation`): one
  group is the belief node as before (the Quick Claw draw stays hidden); several make the root a
  chance node, each group solved as its own belief node (`stage_b._chance_root`,
  `SolveResult.outcomes`, one ranking row `ROOT_CHANCE` with the weighted best and a `chance`
  plan node; `plan_tree.render` / `to_json` show each outcome). Stage A (depth 0) still samples
  the root per playout and ranks actions over it, its usual heuristic.
- Doubles-ready cleanup (review item 5): `damage.wonder_guarded` and `switching.wonder_guard_bit`
  take the defender from their callers (`step._use_move` passes `target_side`, `_pursuit_hit`
  the switcher, whose HP `before_hp` now reads too).

## Roark close batch

- G-35 Battle terrain is fight data from the ROM (`Trainer.terrain`, `tools/terrain.py`): the
  player's tile under the battle's map header, `CalcTerrain` (field_battle_data_transfer.c:
  467-505) from `SetBackgroundAndTerrain` (:496-505), narrowed per trainer to the tiles its
  battle can start on (`narrowed_terrains`, E2 batch 4, d4e3d99: sight tiles,
  trainer_encounter.c:165-343; talk tiles, unk_0203C954.c:62-97; a coord event's rectangle,
  :208-225). The Roark split's terrain refusals came from Kaizo intermediates built before that
  narrowing; re-extracted from the ROM (`tools.extract_rom --game kaizo`, `tools.build_tables
  --game kaizo`; only `terrain` / `terrains` change, 136 / 185 rows), Barry #2 (247-249) is
  TERRAIN_PLAIN (Route203_CoordEvent_Rival's 1x4 rectangle at x 196, z 757, no player movement
  before StartTrainerBattle), Lass Kaitlin (323) TERRAIN_GRASS and Camper Zackary (377)
  TERRAIN_MOUNTAIN. Secret Power (`BtlCmd_GetTerrainSecondaryEffect`, battle_script.c:8063-8075),
  Nature Power and Camouflage were already built on a known terrain (K3b2 Tasks 9-10, E2 batch
  4), so these fights no longer refuse. Tests: `tests/engine/test_e1_roark_close.py`.
- G-36 Clowns Ronald (252) and Moe (254) stay mixed (TERRAIN_PLAIN, TERRAIN_GRASS,
  TERRAIN_BUILDING); they read the terrain only through Metronome's calls. ROM: the join places
  them on Jubilife City (header 3; whole-header terrain {BUILDING}) and Route 202 (header 343;
  {PLAIN, GRASS}), because Kaizo's Route 202 script file (scr_seq member 428, used by that
  header alone) carries their battle entries 14 and 15 (whole-file walk, maps.join_trainer_maps)
  though no Route 202 event runs them. A battle with them under header 343 would need their
  Jubilife objects, which `sub_0206184C` deletes on the zone change (map_object.c:171-198:
  no matching local id in Route 202's list, not persistent). The narrowing cannot drop header
  343 by its rules: its C-startable entry 4 (C_STARTED_SCRIPTS: the start menu's retire,
  start_menu.c:1443-1460) starts a StartTagBattle whose partner operand is a variable
  (scrcmd_trainer.c:146-161: the first operand is the partner), so it matches every trainer.
  Calc repo: no location or terrain field. Sheet: both under "Jubilife City" (Roark Split:37).
  Not guessed; resolving it needs the narrowing to read a battle command's partner operand as
  the partner and to drop entries a header cannot run.
- G-37 AI_FLAG_HARRASSMENT (bit 10, FlagTable script.s:29): `Harrassment_Main` (:8021-8034)
  is `IfTargetIsPartner` (never in singles, trainer_ai.c:2630), `LoadCurrentMoveEffect`
  (:1164), `IfLoadedNotInTable Harrassment_Effects` (:8036-8071, 34 effects, trainer_ai.c:
  876-891), `IfRandomLessThan 128` (:8030, trainer_ai.c:528-538), +2 when it does not jump.
  `policy._harrassment` runs it through `policy._flags` / `_run_flags` (ALLIN: masked by the
  fight's `tables.ai_flags`); the roll is the slot's next `commands._slot_read` (outcome
  `SITE_N[8030] = 128`); `tools/ai_read_bounds.py` bounds it (column 10 of
  `read_bounds.READ_BOUNDS`: one read for a table effect); `layout.KAIZO_FLAG_BITS` and
  coverage's `IMPLEMENTED_AI_FLAGS["kaizo"]` carry bit 10; `commands.HARRASSMENT_EFFECTS`
  (game_constants: decomp effect ids). The eight Gardenia-split fights with the flag are all
  multi battles (201-203, 395, 435-436, 461-462; Cheryl 608 is a partner row), so they stay
  refused by format, and 201, 395 and 608 also by TAG_STRATEGY. Tests:
  `tests/engine/test_e1_roark_close.py`; docs/ai/kaizo_roark.md 5.24.
- G-38 Clowns Ronald (252) and Moe (254) are TERRAIN_BUILDING (closes G-36). `tools/terrain.py`
  `narrowed_terrains` reads two more rules. (1) A battle command's partner operand is the
  partner: `ScrCmd_StartTagBattle` reads partnerTrainer, then the two enemies
  (scrcmd_trainer.c:146-161; `StartTrainerBattle` :123-144 and `StartFirstBattle`, scrcmd.c:
  4038-4048, read enemies only). A variable partner operand resolves to the constants its paths
  assign (`partner_operands`, as the trainer -> map join resolves operands,
  `maps.script_battles`); one no constant reaches on some path is `UNKNOWN_PARTNER`, which
  matches only a trainer some script makes a partner (`ScriptFacts.partners`), not every
  trainer. (2) A script entry the header cannot run starts no battle there: a header runs its
  bg, object and coord events' and init scripts' local ids, the C-started ids
  (`C_STARTED_SCRIPTS`, tools/rom/readers/maps.py:1158-1166) and, when its file can start an entry by id, any
  entry; when every entry naming the trainer is one it cannot run and no object of the header
  is the trainer, the header gives it no terrain. Route 202 (header 343, scr_seq 428) carries
  the clowns' battles as entries 14 and 15, which no Route 202 event runs, and its C-started
  entry 4 is `StartTagBattle VAR_0x8004, 414, 415` (VAR_0x8004 set to 613-615 on its paths), so
  Route 202 drops out and Jubilife City's whole header (header 3, whose objects run the
  entries) gives BUILDING. Kaizo intermediates and tables rebuilt in the worktree
  (`tools.extract_rom --game kaizo`, `tools.build_tables --game kaizo`; local data, only
  `terrain`/`terrains` change, 25 rows). Every change shrinks a set: fights Youngster Tristan
  (1) and Bug Catcher Logan (2) PLAIN (Roark split), Leader Byron (250) BUILDING, the clowns
  BUILDING, Fisherman Andrew (22) {PLAIN, WATER} and Marley (610) {GRASS, MOUNTAIN, CAVE}
  narrower but still mixed; 658 and 765-767 resolve to PLAIN; 16 rows in no split (255, 487,
  494-503, 921-927) are now placed nowhere (`[]`). No row's set grew. Vanilla: unchanged
  (tests/tools/rom/test_terrain_narrowing.py). Tests: that file and
  `tests/engine/test_e1_roark_close.py`.
- G-39 TrainerAI_CalcDamage's draws (trainer_ai.c:2868-3110). MOVE_PSYWAVE (149, Kaizo Triple
  Axel): `level * (RandNext() % 11 + 5) / 10` as fixed damage (:3014-3016; a 0, level 1 with a
  draw below 5, falls to CalcMoveDamage at the table power, :3073); MOVE_MAGNITUDE (222, Kaizo
  Bulldoze): the power of `RandNext() % 100` bucketed below 5/15/35/65/85/95 into
  10/30/50/70/90/110/150 (:3028-3048). `engine.damage.ai_case_fixed` / `ai_case_power` take the
  draw's class (`ai_damage_for_roll(draw=)`); `ai.outcome.CALC_DRAW_P` holds the laws, counted
  over a u16 (Psywave 9 x 5958 + 2 x 5957; Magnitude 3280, 6560, 13120, 19651, 13100, 6550,
  3275), `calc_draw_class` the class of a raw draw. Every estimate draws afresh:
  FlagMoveDamageScore (:1041), IfCurrentMoveKills (:1560), IfPartyMemberDealsMoreDamage
  (:2130, :2156), IfBattlerDealsMoreDamage (:2210, :2228; the battler's move with no damaging
  test, at its own level), CheckIfHighestDamageWithPartner (:2402, TAG_STRATEGY, not
  transcribed). Each command's result is one yes/no whose draws reach nothing else, so the
  per-slot layout makes one read per command that draws (`commands._calc_read`), with the
  exact probability of yes over the draws' joint law (kill: `sum_c P(c) [hp <= d_c]`;
  highest: `sum_c P(D_slot = d_c) prod_{i != slot} P(D_i <= d_c)`; a member out-damaging:
  `1 - sum_v P(A = v) prod_j P(M_j <= v)`; the battler: `sum_c P(B = b_c) P(A < b_c)`): the
  enumeration's pattern bit is the result (`READ_DRAWS[1]` yes), a sampled outcome carries
  the raw draws (`AiOutcome.calc_draws`, uint16[MOVES, slot_reads, 24]) and the command
  computes its result from them as the decomp does. So an 11- or 7-way draw costs one binary
  read, not 11 or 7 patterns per draw. FlagMoveDamageScore at script.s:62, :112, :6511 and
  :6572 is compared only with AI_NO_COMPARISON_MADE (decided by the move, :1035-1036) and
  calcTemp is reloaded (:67, :117) or the script ends: its draws are no read
  (`compared=False`), as the SLOWER tie (K3c Ruling 3). The loose oracle draws every
  `RandNext` of every estimate as a digit of its own (`ReadKind.CALC_PSYWAVE` /
  `CALC_MAGNITUDE`, the dead ones included) and computes each command from them; its patterns
  are mixed-radix and run in chunks (`policy._loose_patterns`, `LOOSE_CHUNK`). Read bounds:
  `tools/ai_read_bounds.py` counts IfBattlerDealsMoreDamage always (the layout does not know
  the player's moves; Copycat's EXPERT column 1 -> 3), and in a fight with a drawing damaging
  move (`read_bounds.DRAWN_READ_BOUNDS`, keyed also by whether the scored move draws)
  IfPartyMemberDealsMoreDamage, IfCurrentMoveKills on a drawing move, and FlagMoveDamageScore
  where a later compare reads its result. 17 Kaizo fights' bounds grow (by 1 or 2): Cyclists
  James (26) and Nicole (30), Jogger Scott (130), Ninja Boy Matthew (146), Pokefan Sheen (177),
  Collector Tobias (227), Worker Hugh (239), Galactic Terra #1 (297), Leader Gardenia (315,
  5 -> 6), Galactic Laomedeia #1 (392), Cyrus #3 (404), Barry #6 (479-481, 9-10 -> 11, 8253
  scorings), Neptune #3 (581), Barry @ Spear Pillar (607, 620); none passes WIDTH_CAP (131072).
  `commands.CALC_DAMAGE_MOVES` keeps Natural Gift, Judgment and Gyro Ball only;
  `prep.variants.ai_kill_damage` lists every class's damage. Tests:
  `tests/engine/test_e1_roark_close.py` (family 3), `tests/ai/test_read_bounds.py`;
  docs/ai/kaizo_roark.md 8.7.

## Batch A

- G-41 Trainer Pokemon in a data form (Wormadam, Deoxys, Rotom, Giratina-Origin, Shaymin-Sky
  personal rows 496-507, `Pokemon_GetFormNarcIndex`, pokemon.c:4929-4960) are built as the
  game builds them, and the coverage refusal "trainer Pokemon in a data form" is gone.
  `TrainerData_BuildParty` (src/trainer_data.c:216-218, :246-248, :272-274, :303-305) sets
  `MON_DATA_FORM` after `Pokemon_InitWith` has read the base species' friendship
  (pokemon.c:435-436), ability (:473-485, ability 1 when ability 2 is `ABILITY_NONE`), gender
  ratio (`Pokemon_GetGenderOf`, :2707-2715) and level-up learnset (`BoxPokemon_SetDefaultMoves`
  reads `MON_DATA_FORM`, still 0, :3819-3829); setting the form only stores it (:1608-1609),
  and the types follow the form (`MON_DATA_TYPE_1/2`, :1219-1225; `BattleSystem_InitBattleMon`
  copies them, battle/battle_lib.c:114-115). Vanilla keeps the base row's stats (no recalc);
  Kaizo's patched builder calls `Pokemon_CalcStats` after the form (rom_diff.md
  "TrainerData_BuildParty, decoded", step 11), which loads the form's row (pokemon.c:595) and
  keeps a full-HP Pokemon at full HP (:633-641). Engine: `PokemonSpec.ability_form` /
  `stats_form` (None = `form`); `build_trainer_party` sets `ability_form=0` and, for the vanilla
  PID routine, `stats_form=0`; the new state field `ability_form` (int8[2,6], STATE.md) is the
  row `ability_slot` indexes in `_fill_slot`, `step._restore_ability`,
  `damage.decomp_abilities`, the reference engine and coverage. Rotom's form rows equal its
  base row; Giratina-Origin (Cyrus 780) now has Pressure and Shaymin-Sky (Cynthia 870) Natural
  Cure, both still refused by Ruling 8 (`_check_form`). Tests: `tests/engine/test_e1_batch_a.py`
  (family 1), `tests/data/test_trainer.py`, `tests/engine/test_coverage.py`.
- G-42 Local script ids 3 and 4 run on a header only where game code can start them
  (`tools/terrain.py` `C_STARTED_SCRIPTS`, `c_started` in `narrowed_terrains`): the Pal Park
  encounter task's last ball (`FieldTask_PalParkEncounter`, encounter.c:591-645, :636-637,
  created only by `Encounter_NewVsPalParkTransfer`, :647-650) and the start menu's Retire,
  which runs 4 outside the Safari Game (start_menu.c:1452-1457) and is shown only in the Safari
  and Pal Park option sets (:279-287, :333-347; the normal, Battle Tower, Union Room and
  Colosseum sets hide it, :385, :402, :407, :412). `FLAG_IN_PAL_PARK` is set and cleared only
  by `ScrCmd_SetClearInCatchingShowFlag` (scrcmd_catching_show.c:24-38; system_flags.c:
  418-431), so 3 and 4 run that way only on a header whose own file reaches
  `SCRCMD_SETCLEARINCATCHINGSHOWFLAG` (Pal Park in both ROMs). Overlay 9 also starts 2, 3, 4,
  6, 7 and 8 (ov9_02249960.c:99, :106, :8765, :11050-12709), kept behind the Distortion World's
  map features as before. 5 (Union Room) is unchanged. Kaizo: 11 rows lose Route 202's or a
  sibling header's tiles (Venus 414, Luna 415, Lucas/Dawn partner rows 613-618 -> BUILDING; Io
  442 -> MOUNTAIN; Scarlet 892, Celeste 893 -> PLAIN); vanilla: none. Tests:
  `tests/tools/rom/test_terrain_narrowing.py`, `tests/tools/rom/test_terrain.py` (Kaizo mixed
  219 -> 208).
- G-40 Clone objects and past-table local ids in the terrain narrowing (`tools/terrain.py`
  `clone_scripts`, `past_table_starts`, `HeaderSites.extra_starts` / `runs_any`). A clone
  object (script `NO_SCRIPT`) keeps the old header's map object with its local id when the
  clone's flag names that header (`sub_0206184C` / `sub_0206244C`, map_object.c:171-198,
  :772-801), so the header runs that object's script id; a clone whose flag header has no such
  object keeps nothing and is built fresh from the clone event (`MapObjectMan_AddMapObjectFrom
  Header` -> `sub_020624CC`'s search, :236-241, :803-815, then :244-255), whose script 65535
  starts a fixed file, no local id (script_manager.c:32-33, :228-254): it adds no id (Kaizo
  headers 336 and 371). A local id past its file's entry table runs
  `ScriptContext_JumpToOffsetID` (script_manager.c:412-416) at `4 * id` plus the word read
  there: inside the file it is an extra start node explored like a bg/object entry (vanilla
  Route 207's clone id 15 -> offset 70: SetVarFromValue, End, no battle); outside the file it
  runs no entry of it (Kaizo's five, vanilla Oreburgh City's 45); a table word past the file's
  end is read from heap memory the decomp does not fix, so that header may run any entry of its
  file (the conservative ruling; vanilla Route 223, header 468, `scripts_empty`). The decomp
  narrowing (vanilla intermediates) has no byte offsets, so a past-table id runs no entry there;
  `test_vanilla_rom_narrows_as_the_decomp` pins that the ROM reader's exact answer gives the same
  rows. Moves no row in either game. Tests: `tests/tools/rom/test_terrain_narrowing.py`.
