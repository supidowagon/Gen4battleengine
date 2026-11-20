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
  returns at :3359-3378), after False Swipe's cap (:3381-3384), while `enduring` is 0 (the
  defender has not used Endure this turn; G-65). The band has no full-HP test (unlike the sash at
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
  `stats_form` (None = `form`); `build_trainer_party` sets `ability_form=0` and, by game,
  `stats_form=0` for vanilla's builder (`IntermediateData.patched_builder` is False: no
  `kaizo.json`, which only Kaizo's resolve step writes); the new state field `ability_form`
  (int8[2,6], STATE.md) is the
  row `ability_slot` indexes in `_fill_slot`, `step._restore_ability`,
  `damage.decomp_abilities`, the reference engine and coverage. Rotom's form rows' abilities
  equal its base row's; Giratina-Origin (Cyrus 780) now has Pressure and Shaymin-Sky (Cynthia 870) Natural
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
  :772-801), so the header runs that object's script id (no chain: a full load's clone path
  `sub_02062714`, :962-970, sets the kept object's script to the clone's 65535 and marks it
  `MAP_OBJ_STATUS_25`, `sub_02062E94`/`sub_02062C18`, :1630-1634, :1861-1868, so only the flag
  header's own non-clone objects carry a script; nine clone-of-clone pairs in each ROM, e.g.
  Route 206/207 local 8, add nothing more); a clone whose flag header has no such
  object keeps nothing and is built fresh from the clone event (`MapObjectMan_AddMapObjectFrom
  Header` -> `sub_020624CC`'s search, :236-241, :803-815, then :244-255), whose script 65535
  starts a fixed file, no local id (script_manager.c:32-33, :228-254): it adds no id (Kaizo
  headers 336 and 371). A local id past its file's entry table runs
  `ScriptContext_JumpToOffsetID` (script_manager.c:412-416) at `4 * id` plus the word read
  there: inside the file it is an extra start node explored like a bg/object entry (vanilla
  Route 207's clone id 15 -> offset 70: SetVarFromValue, End, no battle). A target outside the
  file (Kaizo's, vanilla Oreburgh City's 45) runs heap memory past the loaded file, and a table
  word past the file's end (vanilla Route 223, header 468, `scripts_empty`) is itself read from
  heap memory; the decomp fixes neither, so in both cases the header may run any entry of its
  file (the conservative ruling; review fix: an outside target first ran no entry). Headers
  that may run any entry: vanilla 45, 344, 349, 350, 362, 365, 400, 407, 468; Kaizo 2, 45, 349,
  353, 362, 365, 400, 407, 418, 439. The decomp narrowing (vanilla intermediates) has no byte
  offsets, so there any past-table id may run any entry; `test_vanilla_rom_narrows_as_the_decomp`
  pins that the ROM reader's exact answer gives the same rows. Rows that move: vanilla rival on
  Route 203 (247-249) PLAIN -> mixed (PLAIN, GRASS, WATER), its coord battle no longer
  narrowed because Route 203's past-table id 22 may run any entry; Kaizo Byron (250) BUILDING
  -> mixed. Tests: `tests/tools/rom/test_terrain_narrowing.py`, `tests/tools/rom/test_terrain.py`.
  G-38 still holds under G-40 (checked 2026-11-17 on the Kaizo ROM): Route 202 (header 343,
  file 428, 31 entries) has no clone object (no object script `0xffff`,
  `ObjectEvent_HasNoScript`, map_object.c:2322-2331) and no local id past its table (its ids
  are 1, 29, 30, 31 plus single-battle, visible- and hidden-item ids 3000-3002, 7027, 8161,
  include/script_manager.h:93-97; `ScriptContext_JumpToOffsetID`,
  script_manager.c:412-416, is reached only past entry 31), so neither `extra_starts` nor
  `runs_any` reaches entries 14 and 15, and the clowns stay BUILDING from Jubilife City
  (header 3, objects running 14 and 15). A table giving them -1 / {PLAIN, GRASS, BUILDING} is
  an intermediate built before G-38's narrowing; rebuilding at dc46b16 gives 9 / {9}.
- G-43 A trainer's bag items act by their party params, generic over every item (no
  Potion or Poke Ball case). Data: `Item_Get` (src/item.c:153-198) of each party param the AI
  and `BattleSystem_UseBagItem` read, as the new table `item_party_param[item, ids.ItemParam]`
  (`tools/rom/readers/items.py` `party_params` from the ROM's `pl_item_data.narc`: `partyUse`
  byte 12, the union from byte 14, bit fields LSB first per include/item.h:8-54; the decomp
  extractor packs the JSON as itemproc.c `ProcItem`/`ProcParams` do, :131-279, including the
  `dummy` index of balls, rods and mail; the two agree on every vanilla item,
  `tests/tools/rom/test_moves_items.py`). With `partyUse` FALSE every param reads `dummy`
  (item.c:186-191): a Poke Ball reads 3 everywhere. Kaizo's X Defend, X Accuracy and X
  Special read 0 everywhere (the AI never uses them; no effect). AI (`engine.bag.
  bag_item_category`, `ai.policy._use_item`): `TrainerAI_ShouldUseItem`'s first true branch
  (trainer_ai.c:4099-4189): Full Restore by id below 1/4 HP; an HP item with a nonzero u8
  `hpRestored` below 1/4 HP or when the whole amount fits; each status cure only for that
  status (so Full Heal, whose first param is the sleep cure, is used only on a sleeping
  battler); confusion; then, while `fakeOutTurnNumber - totalTurns >= 0`, a stage param or
  Guard Spec. Use (`engine.bag.use_bag_item`, `step._use_trainer_item`, reference
  `_use_item`): the subscript of the item's branch (`bag_item_branch`; the AI chose the item
  there, and nothing before an item action changes what that choice read), i.e. the
  category's (battle_controller_player.c:1905-1944;
  subscript_use_full_restore / _use_potion / _use_status_recovery / _use_stat_booster) around
  `UseBagItem` (battle_system.c:435-758): status cures, confusion, infatuation, +1 per stage
  param below +6, PP_RESTORE to move slot 0 and PP_RESTORE_ALL to every slot (capped at the
  base PP, trainer Pokemon having no PP Ups), HP (0xFF/0xFE/0xFD fractions, at least 1; the
  potion subscript adds it), the friendship band's amount (x1.5 for a FRIENDSHIP_UP item; no
  Luxury Ball or egg bonus for a trainer Pokemon). Not built and refused by name in coverage
  and the AI (`bag.refused_bag_params`), for an item the AI can use (a branch: Full Restore's
  id, HP_RESTORE, a status or confusion cure, a stage param or Guard Spec,
  `bag.AI_BRANCH_PARAMS`; Dire Hit, Ether, HP Up, Rare Candy and EV berries are never used and
  never refused): an item whose params set Mist (Guard Spec,
  :563-573) or Focus Energy (crit stages, :617-626; both modelled since G-62), and one whose
  friendship amount can pass
  255 within its band, where `Pokemon_IncreaseValue` wraps the party's u8 friendship
  (pokemon.c:1859-1874) apart from the battler's clamped value (battle_lib.c:1032-1044). So
  School Kid Christine (345, Poke Ball x4) stays refused, now for Mist and Focus Energy
  (and the friendship wrap: the ball's high band adds 3 at 255). `TrainerItemId`,
  `trainer_item_effect`, `POTION_HP_RESTORED` and `DECOMP_TRAINER_ITEM_TO_ID` are gone;
  `ids.ITEM_FULL_RESTORE` is a checked game constant. Tests: `tests/engine/test_e1_batch_a.py`
  (family 3), `tests/engine/test_trainer_items.py`, `tests/engine/test_coverage.py`.

## Batch C

- G-44 Deep fog (`Weather.FOG`, `coverage.IMPLEMENTED_WEATHER` now EXACT). Every decomp reader:
  CheckMoveHitAccuracy x6/10 inside `NO_CLOUD_NINE` (battle_controller_player.c:2934-2936;
  `outcome._rate`, reference `_hit_rate`, after Sand Veil / Snow Cloak, before Tangled Feet);
  `FIELD_CONDITION_SOLAR_DOWN` (include/constants/battle/condition.h:136-139) halves SolarBeam
  (battle_lib.c:7058; `damage.py`, B10-13); `FIELD_COND_CHECK_STATE_DEEP_FOG` (bcp.c:1213-1225)
  only prints "The fog is deep..." through subscript_weather_continues.s, whose
  `EndOfTurnWeatherEffect` (battle_script.c:5561-5647) matches no fog branch: no countdown, no
  HP change (`step.field_conditions`, `_TIMED_WEATHERS`). Fog starts only from the map
  (subscript_overworld_fog.s:11, battle_lib.c:3684-3687) and is one bit of
  `FIELD_CONDITION_WEATHER` (condition.h:131-135), which every weather move
  (effect_script_0115.s:8, _0136, _0137, _0164) and weather ability (subscript_drizzle.s:11 and
  its siblings) clears before it sets its own: the engine's single `BattleState.weather` is
  replaced the same way, and the abilities' "already up" test reads only their own `*_PERM` bit
  (battle_lib.c:3746-3778). The AI reads it as `AI_WEATHER_DEEP_FOG` (trainer_ai.c:1436-1437;
  `commands.load_current_weather`). The other readers belong to refused moves or abilities:
  Weather Ball (`FIELD_CONDITION_WEATHER`, battle_script.c:6848, battle_lib.c:7896, :8178,
  trainer_ai.c:3223), Moonlight / Synthesis / Morning Sun (`NO_WEATHER`,
  battle_script.c:5970-5976: fog is "any other weather", 1/4), Defog (subscript_defog.s:12, :64),
  Forecast. The enumeration needs no clause: fog never starts mid-battle, so a move's accuracy
  check reads the start weather or a weather move's replacement (`outcome.hit_thresholds`).
  Random states already drew it (`tests/strategies.py`). Tests: `tests/engine/test_e1_batch_c.py`,
  `tests/engine/test_coverage.py`.
- G-45 Ice Body (`AbilityId.ICE_BODY`, decomp 115). `BtlCmd_EndOfTurnWeatherEffect`'s hail
  branch (battle_script.c:5597-5610): a holder (`Battler_Ability`) with HP left and no
  `MOVE_EFFECT_NO_WEATHER_DAMAGE` gains `Divide(maxHP, 16)` below full HP and takes no chip
  either way (the chip is the `else if`). Both engines now read it from `state.ability`
  (`step._weather_effect`, reference `_weather_effect`; it left `ids.WEATHER_ABILITY_IDS`).
  subscript_weather_continues.s:_076 prints "restored HP using its {1}" with
  `TAG_NICKNAME_ABILITY` for every heal, so the AI learns the holder's ability
  (`hooks.REVEALS_ON` `WEATHER_HP`; the reference now reveals on every heal, Rain Dish's and Dry
  Skin's as before). AI: Basic_CheckHail (script.s:853-871), Expert_Hail (:4110) and the tag
  strategy (:6864, :6873) read it by decomp id (`policy.py`). Tests: `test_e1_batch_c.py`,
  `tests/engine/test_field_pass.py`.
- G-46 White Smoke (`AbilityId.WHITE_SMOKE`, decomp 73). `BtlCmd_ChangeStatStage` tests it in
  Clear Body's branch (battle_script.c:2883-2884: a drop by another battler, after Mist,
  `Battler_IgnorableAbility`), so it is Clear Body's twin everywhere: a move's drop
  (`effects._foe_stage`, reference `_drop_blocked`) and Intimidate (`hooks.intimidate`, the
  reference switch-in). Its messages are built in C (`SetupNicknameAbilityNicknameAbilityMsg`,
  "prevents stat loss" with the ability id in `msgBuffer`, :2886-2895): no AI reveal. AI:
  script.s:472, :766, :940, :1516 by decomp id. Tests: `test_e1_batch_c.py`.
- G-47 Vital Spirit (`AbilityId.VITAL_SPIRIT`, decomp 72). Every decomp site names it beside
  Insomnia with the same branch: subscript_fall_asleep.s:7-8 (`CheckAbility`, an ability side
  effect) and :23-24 (`CheckIgnorableAbility`), subscript_yawn.s:5-6, subscript_rest.s:7-8,
  `BattleSystem_RecoverStatusByAbility`'s shared case (battle_lib.c:4404-4405, :4471-4472) and
  the disobedience nap (bcp.c:2202-2203; the engine has no obedience check, both games). The
  engines test `ids.sleepless` / `SLEEPLESS_ABILITIES` at every Insomnia site: the sleep block and
  its "stayed awake" reveal (`effects._inflict`), Uproar's "kept awake" (`step._use_move`), the
  status cure (`hooks.recover_status_by_ability`, reveal `STATUS_CURE`), the enumeration's status
  clause; Yawn already read both by decomp id (`YAWN_ABILITY_IDS`). Worry Seed writes Insomnia
  only (subscript_give_target_insomnia.s:11). AI: Basic_CheckCannotSleep (script.s:288-295) and
  :1432-1433 by decomp id. Tests: `test_e1_batch_c.py`.
- G-48 Speed Boost (`AbilityId.SPEED_BOOST`, decomp 3). `BattleSystem_TriggerTurnEndAbility`
  (battle_lib.c:3555-3564), run by `MON_COND_CHECK_STATE_ABILITY` (bcp.c:1355) per battler in
  `monSpeedOrder`: a holder with HP left, Speed below `MAX_STAT_STAGE` (+6) and
  `fakeOutTurnNumber != totalTurns + 1` (not entered this turn: a switch, a U-turn / Baton Pass
  newcomer or Roar's victim; a lead's 0 and a faint replacement's previous-turn value pass)
  gets `SPEED_UP_1_STAGE` through subscript_update_stat_stage.s as an ability side effect
  (`step._turn_end_ability`, reference `_turn_end_ability`; the switch on the one ability keeps
  it apart from Shed Skin). Its "raised its" message is built in C
  (`SetupNicknameAbilityStatMsg`, battle_script.c:2854): no reveal. No new draw. The
  enumeration needs no clause: it acts after every hit, and the turn-end order was sorted
  before it (`INPUT_NEUTRAL_ABILITIES`). AI: Basic_CheckLowStatStage_Speed (script.s:438-443)
  and Expert_ChangeUserAbility's table (:4436). Tests: `test_e1_batch_c.py`.
- G-49 Iron Fist (`AbilityId.IRON_FIST`, decomp 89). `BattleSystem_CalcMoveDamage`
  (battle_lib.c:6913-6918): `movePower * 12 / 10` for a move of `sPunchingMoves` when
  `attackerParams.ability` is Iron Fist, after Rivalry and before the weather block
  (`damage._pre_random`, reference `_calc_move_damage`). The list is the ROM's
  (`MoveFlag.PUNCH` from `move_lists.json`), so it covers Kaizo's moves at those ids; the
  confusion self-hit passes MOVE_STRUGGLE (no punch). The AI's damage (`TrainerAI_CalcDamage`,
  trainer_ai.c:3082) runs the same formula. Tests: `test_e1_batch_c.py`.
- G-50 The Kaizo overlay16 accuracy change (data/kaizo/rom_diff.toml: both entries now
  `modelled`). Decoded again with `python -m tools.rom.decode code overlay16 <symbol>`:
  CheckMoveHitAccuracy's sun compare `cmp r0, #0x98; bne` became `cmp r0, #0x98; beq; cmp r0,
  #0x8d; bne` (rom:overlay16@0x0224F068, bcp.c:2915) and CheckMoveHitOverrides' rain compare
  likewise (rom:overlay16@0x0224F384, bcp.c:3025); Blizzard's compare (260) is unchanged. So a
  Kaizo move with effect 141 has Thunder's weather accuracy: 50 before the stage scaling in sun,
  a sure hit in rain, both inside `NO_CLOUD_NINE`. The engines read Thunder's by
  `MoveFlag.THUNDER`; `tools/effect_map.py` `KAIZO_THUNDER_EFFECTS = {141}` gives it to effect-141
  moves in the Kaizo build (`tools/build_tables.py` `_move_flags(..., kaizo)`), so Kaizo's
  Hurricane (move 140) carries it and vanilla's tables are unchanged. Calc repo and sheet: no
  accuracy rule for it. Hurricane stays refused by its own unmodelled be_seq member 141 (and the
  AI's untranscribed effect 141); this lifts only the two overlay16 refusals. Tests:
  `test_e1_batch_c.py`.
- G-51 `IntermediateData.patched_builder` keys on the game id (`IntermediateData(dir, game)`;
  `game` defaults to the game whose `intermediate_dir` is `dir`, `games.game_of_intermediate`,
  ValueError otherwise), no longer on whether `kaizo.json` is in the directory. The CLI, the
  backlog and trace conversion pass the id. Tests: `test_e1_batch_c.py`.
- G-52 Persistent map objects and the terrain narrowing (checked 2026-11-17, no change).
  `sub_0206184C` keeps an object flagged `MAP_OBJ_STATUS_PERSISTENT` across a header change
  (map_object.c:171-198, :182), with its script id. The flag is set by
  `MapObject_SetFlagIsPersistent` (:1843-1850) from `ScrCmd_SetObjectFlagIsPersistent`
  (scrcmd.c:2501-2508), from overlay 9 (ov9_02249960.c:6034, :6059, :7262) and for the player
  (player_avatar.c:79, :181). Walking every header script file and scripts_common in both ROMs
  (`tools.rom.readers.maps._walk`): the only `SetObjectFlagIsPersistent ..., TRUE` commands are
  scripts_eterna_forest's (scr_seq 219, local 0, Cheryl, header 203) and scripts_route_201's
  (427, local 2, the rival, header 342), the same in Kaizo. Those objects' scripts are 9702 and
  9700, global ids (`SCRIPT_ID_OFFSET_FOLLOWER_PARTNERS`, include/script_manager.h:108;
  script_manager.c:44) that run scripts_follower_partners under any header, a file that reaches
  no battle command and no common script in either ROM. Overlay 9's objects are the Distortion
  World's moving platforms (script 0: scripts_unk_0402, script_manager.c:248-251, no battle) and
  its own object events (ov9_02249960.c:12840-13300: scripts 0, 2, 3, 5, 6, 8 and 10002,
  scripts_field_moves, no battle), whose local ids 2-8 are already runnable on every header with
  the Distortion World's features (`C_STARTED_SCRIPTS`, G-42); a warp out reloads the objects.
  The player's object runs no script. So no persistent object can start a trainer battle in a
  header the narrowing does not credit, and `tools/terrain.py` stays as it is (its
  `decomp_check_trainer_objects` and `maps.check_trainer_objects` still raise if a header's own
  script makes one of its trainer objects persistent).

## Batch D

The Fantina split's held items. Item data: vanilla `res/items/data/*.json`; Kaizo from the ROM
(`data/intermediate_kaizo/rom/items.json`): every item keeps vanilla's name and hold effect, with
these params changed: Ganlon and Apicot 0 (vanilla 4), Jaboca and Rowap 6 (vanilla 8). The calc
repo and the sheet add no rule for any of them. New state: `pp_ups` (per move slot), `micle`,
`metronome_turns`, `metronome_move` (per side, volatiles: a switch or faint zeroes
`moveEffectsData` and `metronomeMove`, battle_lib.c:2100-2104, :2134, :2212; Baton Pass keeps none
of them, :2107-2115).

- G-53 A pinch berry's trigger `curHP <= maxHP / itemPower` (battle_lib.c:4778 etc., :5201 etc.;
  Custap :1294) is a plain C `/`. In the Kaizo ROM it calls the ARM9 runtime's `_u32_div_f`
  (rom:arm9@0x020E2178; from overlay16@0x02257DA2 for Ganlon, @0x02253202 for Custap), whose first
  instructions `cmp r1, #0; bxeq lr` return the dividend for a zero divisor (the signed
  `_s32_div_f`, @0x020E1F6C, also returns the dividend). Decoded with `python -m tools.rom.decode
  at <addr> <n> --module <m>` (Kaizo ROM): arm9 `0x020e2178 cmp r1, #0` / `0x020e217c bxeq lr`
  (r0, the dividend, is returned); arm9 `0x020e1f6c eor ip, r0, r1` ... `0x020e1f80 cmp r1, #0`,
  `0x020e1f84 rsblt r1, r1, #0`, `0x020e1f88 beq #0x20e2164`, then `0x020e2164 ands r3, ip,
  #0x80000000` / `0x020e2168 rsbne r0, r0, #0` ... `0x020e2174 bx lr` (|dividend| with the
  dividend's sign: the dividend); overlay16 `0x02257d9e ldr r0, [r1, r0]` (maxHP), `0x02257da0
  adds r1, r6, #0` (itemPower), `0x02257da2 blx #0x20e2178`, `0x02257daa cmp r1, r0` (curHP).
  So a param 0, or 1 under Gluttony, gives
  `maxHP`: Kaizo's Ganlon and Apicot raise Defense / Sp. Def at the first held-item check whatever
  the HP. `order.pinch_limit` (the reference's `_pinch_limit`) at every pinch site; the pinch
  berries and Custap leave `effect_map.DIVISOR_PARAM_MIN` (Sticky Barb, Shell Bell and the flavor
  berries stay: they go through `BattleSystem_Divide`). Pluck's stat raise has no HP test.
- G-54 Jaboca / Rowap Berry (`HOLD_EFFECT_RECOIL_PHYSICAL` / `_SPECIAL`, `ItemId.JABOCA_BERRY` /
  `ROWAP_BERRY`): `BattleSystem_TriggerHeldItemOnHit` (battle_lib.c:5400-5419), one switch on the
  defender's hold effect after the defender's item check (`AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT`,
  bcp.c:3698-3706). The defender's `physicalDamageTaken` / `specialDamageTaken` (set by the move's
  class at UpdateHP, bcp.c:3411-3424, this action), a living attacker without Magic Guard, Jaboca
  also not under `SYSCTL_UTURN_ACTIVE`; no HP test on the holder. The substitute check (:5378)
  returns first. subscript_held_item_recoil_when_hit.s: the attacker loses `Divide(maxHP,
  itemPower)` through UPDATE_HP with SKIP_SPRITE_BLINK (no resist berry; it may faint), then
  `RemoveItem BTLSCR_DEFENDER`. `AFTER_MOVE_EFFECT_HELD_ITEM_STATUS` (bcp.c:3730) then re-checks
  the attacker's item. Pursuit's pre-switch hit: subscript_pursuit.s:32 writes the damage into
  `physicalDamageTaken` whatever the class, then OnHit (:44, :90). U-turn:
  `TriggerHeldItemOnPivotMove`'s Jaboca case (:7368-7375) is the third of four overwriting `if`s
  (Shell Bell, Life Orb, Jaboca, Sticky Barb). Pluck / Bug Bite eat both with no effect
  (`PLUCK_EFFECT_NONE` berries). `effects.on_hit_berry`, `effects.pivot_item`, `step._use_move` /
  `_pursuit_hit`; reference `_on_hit_berry`, `_u_turn`, `_pursuit`.
- G-55 Enigma Berry (`HOLD_EFFECT_HP_RESTORE_SE`, `ItemId.ENIGMA_BERRY`; param 4 in both games):
  the same OnHit switch (:5421-5428): a living defender and `MOVE_STATUS_SUPER_EFFECTIVE`, which
  CheckTypeChart sets for a move with power that is not RANGE_USER (bcp.c:2350-2358) and
  ApplyTypeChart's end clears under `SYSCTL_IGNORE_TYPE_CHECKS` / `_IGNORE_IMMUNITIES`
  (battle_lib.c:2649-2678: Night Shade, Seismic Toss, Super Fang, Counter, Endeavor, Future
  Sight, Present's heal). subscript_held_item_hp_restore.s heals `Divide(maxHP, itemPower)`,
  capped; its PLUCK_CHECK removes it. `MoveHit.super_effective`, `effects.on_hit_berry`;
  reference `_super_effective`, `_on_hit_berry`.
- G-56 Iron Ball (`HOLD_EFFECT_SPEED_DOWN_GROUNDED`, `ItemId.IRON_BALL`): (a) Speed halved by
  `sSpeedHalvingItemEffects` (battle_lib.c:1160-1169, :1240-1246) through
  `BattleSystem_GetItemData` on the held item (no Klutz / Embargo), after Swift Swim and before
  Choice Scarf; the list also holds Macho Brace and the Power items, so `order.effective_speed`
  halves by `ids.SPEED_HALVING_HOLD_EFFECTS` on the raw hold effect (those items stay refused for
  their other effects). (b) `BasicTypeMulApplies` (:2534-2537) lifts an immune Flying row
  (`damage.identified`'s `CHART_GROUNDED`, as Ingrain). (c) ApplyTypeChart's Levitate
  (:2600-2603) skips a holder (`damage.levitated`). (d) The entry hazards
  (subscript_hazards_check.s:7) and Arena Trap (`BattlerIsGrounded`, :5500-5507, in
  `Battler_IsTrapped`) ground the holder (`damage.grounded`, `actions.trapped`). (e) AI:
  `switching.defender_chart_flags` grounds a holder for every ApplyTypeChart walk;
  `calc_effectiveness_flags(..., iron)` passes each CalcEffectiveness call's `defenderItemEffect`
  (`NoImmunityOverrides`, :2759-2764): the bench member's party item (trainer_ai.c:3774), the
  player's active battler's (:3790, battle_lib.c:8006); speed compares go through
  `order.effective_speed`. The AI's Iron Ball refusals and `switching.ITEM_IRON_BALL` are gone.
  Fling (power 130) is unsupported.
- G-57 Grip Claw (`HOLD_EFFECT_EXTEND_TRAPPING`, `ItemId.GRIP_CLAW`): subscript_bind_start.s:9-12
  draws `Random 3, 3`, then for an attacker holding the claw sets 6 (`ids.GRIP_CLAW_TURNS`). The
  duration draw stays in the outcome with its law; a holder does not read it. `effects._trap`.
- G-58 Leppa Berry (`HOLD_EFFECT_PP_RESTORE`, `ItemId.LEPPA_BERRY`; param 10 in both games):
  held (battle_lib.c:4636-4653 / :5026-5043): the first slot with a move at 0 PP regains
  `itemPower` through `BattleMon_AddVal`, capped at `MoveTable_CalcMaxPP(move, ppUps)`
  (:1009-1019; move_table.c:25-34: `pp + pp * 20 * min(ppUps, 3) / 100` in a u8); no such slot,
  no trigger. Pluck / Bug Bite (`PLUCK_EFFECT_PP_RESTORE`, :5609-5633): the eater's first slot with
  the largest `maxPP - ppCur` above 0 regains it, capped. With every move full the decomp's `slot`
  is uninitialized; in the Kaizo ROM it is PluckBerry's `[sp, #8]` (written @0x02258E3C, read
  @0x02258E4A), last written by `Battler_Ability`'s `push {r4, r5}` (overlay16@0x02255A54) under
  `Battler_IgnorableAbility` (@0x02255ABC), called by `BtlCmd_TryPluck` (@0x02247B5C) with r4 = 60
  (`ABILITY_STICKY_HOLD`), a call made whenever the defender holds an item. So `slot` is 60:
  `AddVal(BATTLEMON_CUR_PP_1 + 60)` is `BATTLEMON_TRANSFORMED_PERSONALITY`
  (generated/battle_mon_params.txt), AddVal's default, `ErrorHandling_AssertFail` (compiled in,
  @0x02252E8A), which does nothing outside a link (error_handling.c:11-16). Decoded (overlay16,
  Kaizo ROM): TryPluck `0x02247b28 push {r3, r4, r5, r6, r7, lr}` (24 bytes: sp_T), `0x02247b5a
  movs r3, #0x3c`, `0x02247b5c bl #0x2255abc`, then `0x02247b8c bl #0x2258cd0`;
  IgnorableAbility `0x02255abc push {r3, r4, r5, r6, r7, lr}` (sp_T - 24), `0x02255ac4 adds r4,
  r3, #0` (r4 = 60), `0x02255ac8 bl #0x2255a54`; Battler_Ability `0x02255a54 push {r4, r5}`
  (r4 stored at sp_T - 32); PluckBerry `0x02258cd0 push {r4, r5, r6, r7, lr}`, `0x02258cd2 sub sp,
  #0x14` (sp_P = sp_T - 40, so `[sp, #8]` is sp_T - 32), its only write `0x02258e3c str r5, [sp,
  #8]` (when `0x02258e36 cmp r1, r0` / `ble` finds a larger gap), its read `0x02258e4a ldr r1, [sp,
  #8]`, `0x02258e52 adds r1, #0x1f` (BATTLEMON_CUR_PP_1 = 31), `0x02258e54 bl #0x2252a34`; AddVal
  `0x02252a3a cmp r1, #0x62` and its jump table sends 91 (31 + 60) to `0x02252e8a bl #0x2022974`
  (ErrorHandling_AssertFail), `0x02252e8e pop {r4, r5, r6, pc}`. Battler_Ability makes no call
  (a leaf), nothing between it and PluckBerry's frame writes sp_T - 32, so the slot holds 60. No PP changes, and
  the restore subscript still runs (the pluck flag is cleared). `ppUps` is `BattleState.pp_ups`
  from the party (battle_lib.c:85); a learned move replacing a slot zeroes it
  (battle_script.c:10169-10170). `hooks._leppa`, `hooks.max_pp`; reference `_held_item`,
  `_pluck_berry`, `_max_pp`.
- G-59 Micle Berry (`HOLD_EFFECT_PINCH_ACC_UP`, `ItemId.MICLE_BERRY`; param 4 in both games): at
  the pinch (`pinch_limit`, Gluttony halves; :4897-4906, :5110-5119)
  subscript_held_item_temp_acc_up.s:10 sets `moveEffectsData.micleBerry`; its PLUCK_CHECK removes
  the berry; Pluck sets the eater's flag with no HP test (:5827-5830). The flag gives x120/100
  after the held items and before Gravity in `CheckMoveHitAccuracy`, which clears it
  (bcp.c:2967-2970) once past its early returns (accuracy 0, `SYSCTL_FIRST_OF_MULTI_TURN`,
  `SYSCTL_NONSTANDARD_ACC_CHECK`: Future Sight's use, an OHKO move), at TryMove and at the Future
  Sight hit (battle_script.c:6115, the stored attacker slot's battler). No new draw. The accuracy
  roll's classes (`outcome.hit_thresholds`): a Micle holder's flag can be set before its check,
  and the variants share the grid's Zoom Lens half (one item holds one of them; the target
  having moved changes only Zoom Lens), so `HIT_VARIANTS` is unchanged. The Future Sight hit's
  grid (`future_thresholds`) reads the active battler's flag of now and keeps the roll whole
  (`loose`) where it can change (a flag set, or a Micle Berry on the attacker's field).
- G-60 The Metronome item (`HOLD_EFFECT_BOOST_REPEATED`, `ItemId.METRONOME_ITEM`): CalcDamage
  multiplies by `(10 + metronomeTurns) / 10` right after Life Orb (battle_script.c:1350-1352; the
  param is not read). `BattleSystem_UpdateMetronomeCount` (battle_lib.c:6160-6178) runs at the end
  of BeforeMove (bcp.c:3221: past the status checks and HasNoTarget; DecrementPP never returns
  TRUE): a holder not uproaring nor move-locked (Thrash is unsupported; `SYSCTL_LAST_OF_MULTI_TURN`
  is clear at BeforeMove) counts `moveTemp` (the chosen or encored move, Struggle, an invoker):
  the same move as `metronomeMove` +1 up to 10, another one 0 and it becomes `metronomeMove`; an
  attacker holding no Metronome has its own count reset to 0 (:6176-6177; Verify likewise,
  :6195-6196). The update follows BeforeMove's `MOVE_STATUS_NO_EFFECTS` branch (bcp.c:3209-3221),
  so a move DecrementPP flagged `MOVE_STATUS_NO_PP` (bcp.c:2285) counts too; only Spite empties a
  chosen move's PP before its use, and Spite is unsupported (the encored move with no PP, B10-9,
  takes the same path). `BattleSystem_VerifyMetronomeCount` at UpdateMoveBuffers
  (bcp.c:3957; :6181-6198) takes one back for `MOVE_STATUS_NO_EFFECTS` (CheckMoveFailure did not
  reach USE_MOVE, bcp.c:3312-3331: a miss, Protect, a semi-invulnerable foe, a failure, an
  immunity, no PP), `metronomeMove == moveTemp`, a count above 0, `repeatedMoveCount` 0 (Rollout's
  count before its script, battle_script.c:5665), not uproaring, not
  `SYSCTL_LAST_OF_MULTI_TURN` (a charge move's strike, subscript_charge_move_cleanup.s:7, and
  Bide's end) and not move-locked; a status-check disruption or no target sets none of those
  bits. Multi-hit moves loop to TRY_MOVE (bcp.c:3790): one update per action. TrainerAI_CalcDamage
  has no item factor. The turn-start damage classes use the count the side's own BeforeMove leaves
  (`enumerate.outcome_laws`). `damage.metronome_boost`, `effects.metronome_update` /
  `metronome_verify`, `step._use_move`; reference `_move_damage`, `_metronome_update`,
  `_metronome_verify`.
- G-61 (a reference fix found by the Batch D differential) The invokers (Metronome, Sleep Talk,
  Copycat, Assist) have `RANGE_SINGLE_TARGET_SPECIAL` (res/moves/*/data.json), whose
  `BattleSystem_Defender` is the user (battle_lib.c:1665-1669), so BeforeMove's HasNoTarget never
  fails them: `SYSCTL_MOVE_SUCCEEDED` is set and `movePrevByBattler` keeps the invoker even when the
  foe fainted earlier in the turn (its call then finds no target, battle_script.c:2124-2126).
  `step._use_move`'s `succeeded` had it; the reference's no-target check now passes them too.

## Batch E

- G-62 Guard Spec and the crit-stages param are modelled, lifting their G-43 refusal
  (`bag.refused_bag_params` now names only the friendship wrap). `BattleSystem_UseBagItem`
  (battle_system.c:563-573): Guard Spec reads the user side's `mistTurns`; when 0 it sets
  `SIDE_CONDITION_MIST` and 5 turns (:569) and counts as applied: `BattleState.mist[side]`,
  E2's Mist timer (B12-23; mask and turns are one field there). The crit-stages param
  (:617-626, gated like the stage params by `selectedSlot == partySlot || targetSlot ==
  partySlot`, the user's own slot in singles, :443-454) sets `VOLATILE_CONDITION_FOCUS_ENERGY`
  when clear and counts as applied: `BattleState.focus_energy[side]` (B12-4), +2 crit stages
  on its hits. Order: Guard Spec after infatuation, before the stage params; crit after
  them, before PP. The subscript of the GUARD_SPEC category, subscript_use_guard_spec.s, runs
  `UseBagItem` and a message only. AI: `TrainerAI_ShouldUseItem`'s Guard Spec branch
  (trainer_ai.c:4181-4182) reads `sideConditionsMask[1]`, the AI's own side, which is the side
  the item sets: a used Guard Spec blocks its own reuse while the Mist lasts. Dire Hit (crit
  stages only) still has no AI branch, so a trainer never uses it (Kaizo's reads 0 everywhere);
  a Poke Ball (every param 3) and Kaizo's Guard Spec. (every param reads 128, as a
  `partyUse` FALSE item's dummy) stay refused for the friendship wrap.
  `engine.bag.use_bag_item`, reference `_use_item`. Tests: `tests/engine/test_e1_batch_e.py`.

## Trial batch

- G-63 Plus and Minus (`AbilityId.PLUS` / `MINUS`, decomp 57 / 58). `BattleSystem_CalcMoveDamage`
  (battle_lib.c:6802-6809, after the Marvel Scale defense line :6797-6800 and before the Mud Sport
  / Water Sport lines :6811 on; attacker ability `attackerParams.ability`, :6653): a Plus attacker
  with `BattleSystem_CountAbility(.., COUNT_ALIVE_BATTLERS_OUR_SIDE, attacker, ABILITY_MINUS)`
  nonzero (and Minus with Plus) gets `spAttackStat * 150 / 100`. The count (battle_lib.c:2886-2893)
  is every battler of the attacker's side, the attacker included, with `curHP != 0` and
  `Battler_Ability == ability`; only battlers on the field, never the bench. In singles the
  attacker is its side's only battler and has one ability, so the count is 0 and the boost never
  applies; a doubles Plus with a living Minus ally (or the reverse) gets x1.5, Plus with Plus does
  not. Built as `damage.side_ability_count` over `damage.side_active_slots` (the one function a
  doubles layout extends; reference `_side_ability_count`), read in `damage._pre_random` and the
  reference `_calc_move_damage` at the decomp's position. The AI scripts never read either ability.
  Tests: `test_e1_trial.py`.
- G-64 Perish Song, the AI's switching step 3. `TrainerAI_ShouldSwitch` (trainer_ai.c:3894-3987), after
  the trap gate (:3906-3913) and the bench test (:3941), runs `AI_PerishSongKO` (:3263-3272) before
  the Wonder Guard step (:3945) and without a draw: the AI's active battler has
  `MOVE_EFFECT_PERISH_SONG` and `perishSongTurns == 0` -> `aiSwitchedPartySlot = 6`, TRUE. The
  comment above it (:3253-3262, "does nothing") is wrong; the code is followed. Reachability:
  `BtlCmd_TryPerishSong` sets 3 (battle_script.c:5480-5481), the turn-end pass shows and decrements
  3 -> 2 -> 1 -> 0 and faints at 0 (battle_controller_player.c:1799-1809), so at the selection of
  the turn on whose end it faints the AI sees 0. `TrainerAI_PickCommand` (:4001-4026) then sends
  `BattleAI_PostKOSwitchIn`'s pick (battle_lib.c:7923), else the first living slot that is not the
  active (:4014-4022). Built as `switching._Layout.perish`: no switch site exists then
  (`exists & ~perish`), and `switch_action` returns the `replacement` pick for every stop; the
  scoring side is `Basic_CheckPerishSong` (script.s:720-723, -10 against a perishing player),
  ALL_FAINT_3_TURNS (114) transcribed with no Expert entry. Tests: `test_switching.py`,
  `test_kaizo_policy.py`.
- G-65 Endure (`EffectId.ENDURE`, BATTLE_EFFECT_SURVIVE_WITH_1_HP 116). effect_script_0116.s:3 is
  Protect's script (effect_script_0111.s:3): `BtlCmd_TryProtection` (battle_script.c:5149-5190), so
  the chain reset (:5154-5158, Endure counts), the roll (:5167-5168) and the last-actor test
  (:5160-5165) are shared and Endure and Protect continue each other's chain; a success sets
  `turnFlags.enduring` (:5174-5177) instead of `protecting`. `enduring` is read at one site,
  `BattleControllerPlayer_UpdateHP` (battle_controller_player.c:3386-3404), for a damaging hit not
  taken by a substitute, after False Swipe's cap: `if (enduring == 0)` wraps both the Focus Band roll
  (:3387-3389, not drawn) and the Focus Sash test (:3391-3393, not armed); then
  `enduring || focusItemActivated` with `curHP + damage <= 0` leaves 1 HP (:3396-3404), as
  `MOVE_STATUS_ENDURED` (a message only; no item is named or removed) when enduring, else
  `MOVE_STATUS_ENDURED_ITEM`. An earlier hit's `focusItemActivated` still applies (an OR), so an
  enduring hit after it is `ENDURED` too. `BtlCmd_CheckHoldOnWith1HP` (:9098-9124: the confusion
  self-hit, Pursuit's pre-switch hit, Future Sight) and every non-UpdateHP damage (poison, weather,
  recoil, Curse) ignore Endure. Built as the `endure_ok` event and `ts_endure_ok` paused-turn field
  (beside `protect_ok`), read as `foe_enduring` by `effects.play_hit`; the reference `_Turn.enduring`.
  Pursuit's hit does not pass it (subscript_pursuit.s:35 is CheckHoldOnWith1HP). AI: `Expert_Endure`
  (script.s:3633-3649, dispatch :1704): below 4 % HP -1; below 35 % +1 unless the 70 roll (:3645)
  jumps; else -1 (the fall-through at :3638 reaches :3640); no Basic entry. Tests:
  `test_e1_trial.py`, `test_kaizo_policy.py`.
- G-66 Magnet Rise (`EffectId.MAGNET_RISE`, BATTLE_EFFECT_GIVE_GROUND_IMMUNITY 252). The move
  (effect_script_0252.s) fails (:16, MOVE_STATUS_FAILED) for a user with `magnetRiseTurns` (:5, the
  turns, not the bit), Levitate (:6, `Battler_Ability`) or `MOVE_EFFECT_INGRAIN` (:7), else sets
  `MOVE_EFFECT_MAGNET_RISE` (:8) and `magnetRiseTurns = 5` (:9). `BattleSystem_ApplyTypeChart`
  (battle_lib.c:2605-2609, after the Levitate test) gives a Ground move against a defender with
  turns, no Ingrain and no Iron Ball `MOVE_STATUS_MAGNET_RISE` (moves.h:77, part of
  `MOVE_STATUS_IMMUNE`, :87): the walk is skipped (only STAB stays in the value); Mold Breaker does
  not lift it; `CalcEffectiveness` has no such test. Hazards read the mask BIT
  (subscript_hazards_check.s:11, after Gravity :6 and Iron Ball :7); Arena Trap and
  `BattlerIsGrounded` the TURNS (battle_lib.c:3215-3218, :5500-5507). `MON_COND_CHECK_STATE_MAGNET_RISE`
  (bcp.c:1647-1656, after TAUNT) counts the turns down and `subscript_magnet_rise_end.s` clears the
  bit at 0; turns zeroed by Gravity never reach it (the bit stays). A switch clears both
  (battle_lib.c:2100-2104); Baton Pass keeps the bit (`MOVE_EFFECT_BATON_PASSED`, moves.h:42-44)
  and the turns (:2112). Built as `BattleState.magnet_rise` / `magnet_rise_turns`,
  `damage.magnet_risen` / `ground_immune` at every `ApplyTypeChart` immunity site (five in the
  engine, six in the reference), `damage.grounded`'s `magnet` argument, `actions.trapped`, the
  countdown folded into `step._taunt_check` (reference `_taunt_check`), and in the AI
  `switching.defender_chart_flags`' `magnet` with the `MAGNET_RISE` bit counted as immune by
  `commands.effectiveness_class` and `replacement._stage_2_score`. AI scoring:
  `Basic_CheckMagnetRise` (script.s:1461-1474, -10 with the bit, Levitate or a Flying type) and
  `Expert_MagnetRise` (:6090-6119, dispatch :1795, roll :6113); the Magnet Rise refusals are gone.
  Tests: `test_e1_trial.py`, `test_kaizo_policy.py`, `test_switching.py`.
- G-67 Gravity (`EffectId.GRAVITY`, BATTLE_EFFECT_GRAVITY 215), in both games. The move
  (effect_script_0215.s) fails with `FIELD_CONDITION_GRAVITY` up (:3, :9-10), else the indirect
  `subscript_gravity_start.s`: `FIELD_CONDITION_GRAVITY_INIT` (5 turns, condition.h:143), then per
  battler with HP (:13-85): `magnetRiseTurns != 0` -> 0 (the `MOVE_EFFECT_MAGNET_RISE` bit stays),
  else `MOVE_EFFECT_AIRBORNE` (a Fly / Bounce charge turn) -> `Battler_UnlockMoveChoice`
  (battle_lib.c:1834-1841) and ToggleVanish; the Gastro Acid, Ingrain, Levitate and Flying branches
  only print (`effects.gravity_ground`). The countdown (`FIELD_COND_CHECK_STATE_GRAVITY`,
  battle_controller_player.c:1227-1238) runs `subscript_gravity_end` at 0: vanilla prints; Kaizo's
  replacement (sub_seq 238, data/kaizo/rom_diff.toml) sets the counter to 5 again and runs the
  grounding loop without messages, so Gravity never ends there (`Tables.gravity_rearm`, 1 in Kaizo,
  built by tools/build_tables.py). Accuracy x10/6 (bcp.c:2972-2974) and the hazards / Arena Trap
  grounding were already in. New: `BasicTypeMulApplies` (battle_lib.c:2545-2549) and
  `NoImmunityOverrides` (:2766-2770) skip the Ground-on-Flying immunity (`damage.identified`'s
  `CHART_GROUNDED`; the AI's `chart_multipliers` already read `state.gravity`); `Battler_Ability`
  reads Levitate as none (:3095-3098: `damage.ignorable`, `effects.magnet_rise_fails`, the reference
  `_ignorable`). `Move_FailsInHighGravity` (:3394-3407) bans Fly, Bounce, Jump Kick, Hi Jump Kick,
  Splash and Magnet Rise (`ids.GRAVITY_MOVES`, decomp ids; Kaizo's id 150 is "Hidden Power Water",
  banned by id) at selection (CHECK_INVALID_GRAVITY, :2292-2295: `actions.usable_moves`, all banned
  -> Struggle; the AI's list too, trainer_ai.c:234), at the status check right after Imprison
  (CHECK_STATUS_STATE_GRAVITY, bcp.c:2586-2595, `moveCur`: Sleep Talk itself, not its call;
  `moveFailFlags.gravity`, a message only: no unlock), from Sleep Talk's set (CHECK_INVALID_GRAVITY in
  `BtlCmd_TrySleepTalk`'s `~CHECK_INVALID_NO_PP` mask, battle_script.c:4927: `order.sleep_talk_slots`) and
  for Metronome, Assist and Copycat calls (`Move_CanBeMetronomed`, :7198-7201; Assist's list,
  battle_script.c:6430). Those sets are read when the call runs, so a faster foe's Gravity earlier in the
  same turn shrinks them: `enumerate.call_laws` adds a Gravity variant (`CallLaws.can_gravity`, where
  `order.gravity_possible`: Gravity down at the turn start, a Sleep Talk / Metronome / Assist user, a foe
  move that can run Gravity) whose law is read under Gravity (`order.gravity_state`), its values offset by
  `ids.CALL_GRAVITY` (`invoked_move` reads `call % CALL_GRAVITY`); `call_consistent` keeps a row of either
  law only where the step's `gravity_at_move` event agrees, and `sample_outcome` redraws the call and the
  called move's draws from the Gravity set where it landed. Copycat needs no variant (no draw; a banned
  movePrev fails and reads nothing). Area Gravity (Kaizo,
  sub_seq 297: the animation then `subscript_gravity_start` byte for byte, run at battle start on a
  map with weather 12): the trainer's area condition starts the fight with Gravity at 5
  (`coverage.start_gravity`, `state.initial_arrays(gravity=)`; the loop is a no-op at battle start); the
  coverage refusal is gone and rom_diff members 238 and 297 are `modelled`. Trick Room's twin
  (sub_seq 251) is not part of this group. AI: `Basic_CheckGravityActive`
  (script.s:1026-1029), `Expert_Gravity` (:4782-4803, dispatch :1759, rolls :4795 and :4799;
  `commands.FieldCondition.GRAVITY` over `state.gravity`). Tests: `test_e1_trial.py`,
  `test_invokers.py`, `test_kaizo_policy.py`, `test_coverage.py`.
- G-68 Mimic (`EffectId.MIMIC`, BATTLE_EFFECT_COPY_MOVE_FOR_BATTLE 82), in both games; the party / battler
  split of the move and PP arrays. effect_script_0082.s:3 runs `subscript_mimic.s` on a hit that
  connected (battle_lib.c:1485-1490). `BtlCmd_TryMimic` (battle_script.c:4476-4524) fails
  (subscript_mimic.s:16, MOVE_STATUS_FAILED) when `movePrevByBattler[defender]` is MOVE_NONE or not
  Mimic-able (`Move_CanBeMimicked`, battle_lib.c:7152-7158, :7182-7192: Metronome, Struggle, Sketch,
  Mimic, Chatter, `ids.MIMIC_FORBIDDEN`), the user is Transformed (:4482, G-69), the defender has a
  Substitute (:4483) or the user already knows the move (:4490-4492, :4519-4521). Success: slot `j`
  is the user's first Mimic slot (:4494-4499; `j` always exists, the script runs only from the user's
  own slot, as itself or through Sleep Talk: Metronome, Copycat and Assist cannot call it,
  `Move_CanBeMetronomed` :7196-7218), `moves[j]` = the move, `ppCur[j]` = min(5, its table PP) (:4502-
  4513), `mimickedMoveSlot |= 1 << j` (:4515). The decomp keeps the party Pokemon and the BattleMon
  apart and Mimic writes the BattleMon only: PP use and every `BattleMon_CopyToParty` skip flagged
  slots (battle_lib.c:1819-1825; battle_io_command.c:574-587), so the party keeps Mimic with its PP
  after Mimic's own use; a level-up's learn writes the party (battle_script.c:10168-10176) and
  `BattleSystem_ReloadPokemon` reloads only unflagged slots (battle_lib.c:183-191); leaving the field
  clears `moveEffectsData` (:2100-2104; Baton Pass keeps no `mimickedMoveSlot`, :2107-2114) and the next
  `InitBattleMon` loads moves and PP from the party (:80-85). The engine's per-slot `moves` / `pp` are
  both, so a per-battler shadow (`mimic_slots`, `shadow_moves`, `shadow_pp`) holds the party's value of each
  flagged slot: `effects._mimic` writes it, `step._restore_moves` (beside `_restore_types` /
  `_restore_ability`) puts it back on a switch, a drag, the replacement of a fainted battler and, for
  the player's party, at the battle's end (`_faint_check`); `levelup` learns into the shadow of a flagged
  slot. The AI record `ai_seen_moves` is move ids in first-free order (`TrainerAI_RecordLastMove`,
  trainer_ai.c:2705-2716; `state.record_seen_move`), so a Mimic copy, a replaced slot and the player's
  Struggle are recorded as the decomp does (docs/discrepancies.md's Struggle entry is gone). The
  AI side: no Basic or Expert entry and no other script.s mention (`KAIZO_TRANSCRIBED`); an AI Mimic's copy
  is scored by the copied move's label, so a fight with a Mimic or Transform user adds the effects of the
  foe's possible moves to its layout (`layout.fight_tables(foe_moves=...)`). Coverage no longer refuses a
  Mimic user that levels up (82 left `RELOAD_STATE_EFFECTS`). Tests: `test_e1_trial.py`,
  `test_knowledge.py`, `test_learn.py`.
- G-69 Transform (`EffectId.TRANSFORM`, BATTLE_EFFECT_TRANSFORM 57), in both games (Kaizo's move has
  priority 1, vanilla 0; no CAN_PROTECT in either). effect_script_0057.s:3 is a direct `CHECK_HP | TO_DEFENDER`
  `subscript_transform_into_target.s`: it runs on a defender with HP, also after a general miss
  (battle_lib.c:1492-1504), and fails (:23-24, MOVE_STATUS_FAILED) on a semi-invulnerable (:5) or already
  Transformed (:6) defender; a Substitute is not tested. `BtlCmd_Transform` (battle_script.c:5330-5373)
  sets `VOLATILE_CONDITION_TRANSFORM`, ends Disable (:5335-5336), clears `mimickedMoveSlot` (:5339: the
  mimicked slot is NOT restored, the copy overwrites it) and Last Resort's count, and copies every byte of
  the defender's BattleMon up to and including `ability` (:5346-5348; battle_mon.h:57-84): species, the
  five stats as the BattleMon holds them (a Power-Tricked defender's swapped ones), the four moves, the IVs,
  all eight stat stages, both types, the form and the ability; not HP, max HP, level, friendship, PP Ups,
  status, volatiles, `moveEffectsMask`, the item, the gender (`transformedPID` / `transformedGender` are
  sprites and messages only, :5337-5338) or exp. It clears the switch-in announce flags (:5350-5362) and
  sets `truant = totalTurns & 1`, then each slot's PP is `min(5, table PP)` (:5364-5370). The switch-in
  checks run again after the move (AFTER_MOVE_EFFECT_TRIGGER_SWITCH_IN_EFFECTS, bcp.c:3672-3676 ->
  `BattleSystem_TriggerEffectOnSwitch`, battle_lib.c:3650-4170): the copied ability acts as on a switch-in
  (Trace, a weather ability, Intimidate, Download, the reveals), through the engine's own hooks
  (`step._use_move` calls `hooks.switch_in_effects` for the user). A copied ability is always one of the
  fight's own Pokemon's, which coverage grades already, so an ability the engine does not model (Slow Start)
  refuses the fight through the Pokemon that has it, never through Transform. While Transformed the party never sees moves or PP (battle_io_command.c:581-587);
  the engine's per-slot arrays are both, so the Mimic shadow grows to the whole copied set
  (`transformed`, `shadow_species`, `shadow_form`, `shadow_stats`, `shadow_ivs`, `shadow_ability_slot`,
  `shadow_ability_form`, with `shadow_moves` / `shadow_pp` of all four slots and `mimic_slots` = 0b1111;
  the party's ability is derived from the restored species, form and ability slot, as Trace's
  `_restore_ability` does): `effects._transform` writes the copy and the shadow once (a second Transform
  keeps the first's), `step._restore_transform` (before `_restore_types` / `_restore_ability` /
  `_restore_moves`) restores it on a switch, a drag, a replacement and, for the player, at the battle's end
  (which now also restores the player's Camouflage types and Trace ability); a fainted battler keeps the
  copy until its replacement enters, because the exp payout reads the copied species and its yield
  (battle_script.c:2216) and EVs. A level-up's `ReloadPokemon` reloads the user's OWN stats, level and HP
  but not its moves, PP, PP Ups or exp (battle_lib.c:169-194): `levelup.party_identity` reads the party
  species, form and IVs (the exp table, `Pokemon_CalcStats`, the learnset, the Shedinja rule) and the new
  stats go to the battler and to `shadow_stats`; a learned move goes to the party's shadow. Species-keyed
  rules read the copied species by themselves (Metal Powder's Ditto, Light Ball, Thick Club, the weight);
  the Griseous Orb boost excludes a Transformed user (:6770, `damage.py`); Mimic (:4482) and Chatter's
  confusion (:8404) fail / skip while Transformed. `MoveEffectsData.truant` was `fake_out_turn`'s parity
  and is now a field (`BattleState.truant`: `(totalTurns + 1) & 1` at an entry or faint, `totalTurns & 1` by
  Transform; `order.truant_loafs`). AI: `BATTLE_EFFECT_TRANSFORM` has no Basic or Expert entry and no script.s
  mention (`KAIZO_TRANSCRIBED`); `trainer_ai.c` never tests the volatile bit, so the AI reads the
  transformed BattleMon like any other; an AI Transform user's copied moves are scored by their labels
  (`layout.fight_tables`, G-68). `ai_ability_known` is `battlerAbilities`: the decomp id a message
  named (`BattleAI_SetAbility`, battle_script.c:12106-12109; hooks / effects record
  `damage.decomp_ability_id`), 0 none, -1 an ability the engine does not name (a species ability the
  tables carry only as a decomp id: read as the current one, settled to its id when its holder Transforms); `LoadBattlerAbility DEFENDER` (trainer_ai.c:1170-1205) returns it
  when set, else Shadow Tag / Magnet Pull / Arena Trap of the BattleMon, else a coin between the BattleMon
  SPECIES' two abilities (the copied species after a Transform): a pre-Transform reveal stays the old
  ability. Coverage no longer refuses a Transform, Mimic or Power Trick user that levels up
  (`RELOAD_STATE_EFFECTS` and its refusal are gone). Tests: `test_e1_trial.py`, `test_knowledge.py`. Found by the differential test with Transform's draws: the Metronome item's
  count (`effects.metronome_update`, G-60) ran after the second turn's MOVE_LOCKED cleanup in `step._use_move`, so a locked
  Fly user counted; `BattleSystem_UpdateMetronomeCount` runs at the end of BeforeMove (bcp.c:3221), before the move's script,
  and still sees the lock (the reference already did): the call now precedes the cleanup.
- G-70 Chatter (`EffectId.CHATTER`, BATTLE_EFFECT_CHATTER 267), in both games. effect_script_0267.s:3-5 is a
  plain hit (CalcCrit, CalcDamage) with an indirect `ON_HIT | TO_DEFENDER` `subscript_chatter.s`, which runs on a
  hit that did not fail (`MOVE_STATUS_NO_EFFECTS` clear, battle_lib.c:1524-1529). `CheckChatterActivation`
  (battle_script.c:8397-8436) jumps (no effect) unless the attacker's `BattleMon.species` is Chatot, the defender's
  `curHP != 0` and the attacker is not Transformed (:8403-8405); then `chatter =
  Sound_GetChatterActivationParameter(BattleSystem_GetChatotCry(battler))` (:8407; recorded battles read a stored
  value, never ours) gives `effectChance` 0 / 10 / 30 for chatter 0 / 1 / 2 (:8413-8426) and `RandNext() % 100 >
  effectChance` jumps (:8428-8430): the confusion comes on `rand % 100 <= chance`, 1 %, 11 % or 31 %
  (`Outcome.chatter`, law `[99 - chance, chance + 1]` of 100, `enumerate._chatter`). The confusion is
  `subscript_confuse.s` with `SIDE_EFFECT_TYPE_INDIRECT` (effect_script_0267.s:5, battle_lib.c:1520-1522):
  `effects._chatter` (target alive; the defender's Shield Dust as the attacker sees it stops it, :14-15
  `CheckIgnorableAbility`, so Mold Breaker pierces it, after the activation roll is drawn; not confused :24, no
  Substitute :23, no Safeguard :26; the counter is `Outcome.duration`, drawn only when the roll passes:
  `enumerate._chatter_reads`). Kaizo's move row changed (power 120, pp 5,
  RANGE_ALL_ADJACENT, effect_chance 50 against vanilla 60 / pp 20 / 0): the chance is never read (the ON_HIT
  branch reads no `effectChance`, battle_lib.c:1524-1529; only the PROBABILISTIC and bare-flags branches,
  :1549-1585, and `BtlCmd_CheckEffectActivation`, battle_script.c:8366-8381, do), so the tables keep the ROM's
  row and the engine reads it through `outcome.read_chance`, 0 for an `ids.ON_HIT_EFFECTS` script (Chatter),
  else the engine's `secondary_chance` would roll a phantom 50 %. The cry parameter is per battler, a battle constant (`BattleState.chatot_cry`, like `terrain`):
  `BattleSystem_GetChatotCry` (battle_system.c:915-924) reads `chatotCries[battler]`; `FieldBattleDTO` creates
  every cry with `ChatotCry_New` (`valid = FALSE`, field_battle_data_transfer.c:97, chatot_cry.c:13-25) and only
  `BATTLER_PLAYER_1` gets the save's (:260, :319, :385), and an invalid cry gives 0 (sound_chatot.c:193-200): an
  enemy Chatot always confuses 1 % of the time (every Kaizo Chatter user is an enemy Chatot). The player's is the
  save's (`SaveData_GetChatotCry`, chatot_cry.c:27-30; `BOOL valid; s8 audioData[1000]`): 0 if not valid, else
  `audioData[15]` in [-128, -30) -> 1, [30, 128) -> 2, else 0 (sound_chatot.c:193-213);
  `initial_arrays` / `make_state` take it as `chatot_cry` (side 0; 0 by default). Data order (pipeline.md): the
  ROM has no cry (save data), the calc repo states the 1 % no-recording case (pkcalc.json:8696), the sheet has
  nothing; the save does: `saves.load_save` reads the SAVE_TABLE_ENTRY_CHATOT page (save_table.c:71) at
  `layout.CHATOT_CRY` = 0x64EC of the normal block, the sum of the 22 pages before it (`layout.NORMAL_PAGES`,
  each `sizeof` rounded as `s += 4 - s % 4; s += 4`, savedata.c:813-823; the 37 pages sum to the block, and
  the per-page CRC16s of the pages after it, savedata.c:1349-1362, match in the real saves), and
  `SaveFile.chatot_cry` is its parameter (`saves.chatot_parameter`). The cli, `nuzline line` and the prep
  problem (`PrepProblem.chatot_cry`) pass it to coverage and `make_state`; a party file gives it as a
  top-level `chatot_cry` (`data.party.load_chatot_cry`), and without one (unknown: None) coverage refuses a
  player Chatot that knows or can learn Chatter: "Chatter's confusion chance needs the save's Chatot cry"
  (data, not a mechanic); an enemy Chatter is never refused. AI:
  `BATTLE_EFFECT_CHATTER` has no Basic or Expert entry; its only script.s mention is `Basic_CheckSoundproof`
  `:131` by move id (`commands.SOUND_MOVES`, already transcribed): `KAIZO_TRANSCRIBED`; the damage estimate is
  the table power and type. Tests: `tests/engine/test_chatter.py`.
- Batch 13 merge check (B13-39, branch `claude/e1-trial-on-b13`). Chatter's roll moves to fold_in 43:
  42 is batch 13's `outcome.SIXTH_HIT_FOLD`. The reference `_magnet_risen` reads the Iron Ball through
  the gated accessor (`_iron_ball(tables, s, mon)`): ApplyTypeChart's `defenderItemEffect` is
  `Battler_HeldItemEffect` (battle_lib.c:2579, :2605-2608). Checked and unchanged: every other trial
  item read goes through `held_item` / `holds` / `iron_ball` (Focus Band and Focus Sash under Endure:
  bcp.c:3348-3349, :3386-3404, and the Sash removal's CheckItemHoldEffect, battle_script.c:7919-7922);
  per-hit arrays use `MAX_HITS` (6); the Metronome item count runs once, before the move script
  (B13-34); Pluck and the Custap gate are batch 13's (battle_lib.c:5839-5844,
  subscript_check_quick_claw.s:26-27).
- Trial accuracy classes: Gravity and Transform in `outcome.hit_thresholds` (branch `claude/e1-trial-on-b13`).
  A faster foe's move can change the slower move's threshold before its check: Gravity sets
  FIELD_CONDITION_GRAVITY, read as `hitRate * 10 / 6` (battle_controller_player.c:2972-2974); Transform
  (`BtlCmd_Transform`, battle_script.c:5330-5373) copies the BattleMon up to `ability`, so `statBoosts`
  (battle_mon.h:75) and `ability` (:84): the target's evasion becomes the side's own (as Psych Up's), and
  the target holds the side's ability (Sand Veil / Snow Cloak bcp.c:2927-2933, Tangled Feet bcp.c:2944-2947,
  Simple / Unaware bcp.c:2876-2887, No Guard bcp.c:3016-3021, Cloud Nine); the cleared announce flags
  (battle_script.c:5350-5358) let its switch-in checks rerun (bcp.c:3672-3673, :3973), so a copied weather ability sets its weather (battle_lib.c:3740-3790). Built in
  `_threshold_grid`: the eva copy joins Psych Up's, and the second weather variant is now the foe's field
  move (weather move, Gravity, or Transform with its copied ability and weather); one move has one effect.
  `future_thresholds` keeps the Future Sight roll whole (`loose`) when a Gravity or Transform is used this
  turn or Gravity is at 1 (FIELD_COND_CHECK_STATE_GRAVITY, bcp.c:1227-1238, ends it before the hit
  :1758-1783). The reference engine's live check (`_hit_rate`, `_transform`) already read both; no change.
  Tests: `test_accuracy_classes.py` (registry, `test_gravity_and_transform_are_seen_by_hit_thresholds`).
- Trial outcome laws: Transform, Mimic and Chatter in the damage registries and the laws (branch
  `claude/e1-trial-on-b13`). `BtlCmd_Transform` (battle_script.c:5330-5373) copies the BattleMon up to
  `ability` (battle_mon.h:57-84: stats, `statBoosts`, types, ability; not the gender), so a faster foe's
  Transform changes reads the turn-start laws took as fixed. Built in `enumerate`: the transform clause of
  `damage_inputs_can_change` (`INPUT_CLAUSE_EFFECTS`; stats, stages, types, ability, a rerun Intimidate)
  and the White Herb's any-stage term; `contact_proc` for a copied Static / Poison Point / Flame Body or
  Cute Charm (the user's own gender kept); `present_guard` for a copied Wonder Guard; `end_speed_tie`
  whenever a Transform is used (the copied Spe, stage and speed ability); `shed_skin` for a copied Shed
  Skin. `outcome.armored` (laws `no_crit`, `sample_outcome`) drops a Battle Armor / Shell Armor the
  Transform replaces with the side's own ability (battle_lib.c:7135-7136). Mimic (`BtlCmd_TryMimic`,
  :4476-4524) writes only its user's moves, PP, `mimickedMoveSlot` and `lastResortCount`: input-neutral,
  non-table. Chatter (effect_script_0267.s:5-7: CalcCrit, CalcDamage, then the ON_HIT confusion):
  input-neutral, table damage. `UNSUPPORTED` stays outside both registries (the tests add it). The
  reference engine reads the live state at each check; no change. Tests: `test_transform_laws.py`,
  `test_damage_classes.py` (Chatter's exact case, Transform's compression case).

## Batch F (Kaizo abilities, held items, other)

Rule sheets: `_scratch/e1/b-next/sheets/` (17 groups). Carried review notes: `_scratch/e1/b-next/carry.md`.

### Open items
- **F-O1 Weather Ball's AI type outside the typed weathers.** `TrainerAI_MoveType` (src/battle/trainer_ai/trainer_ai.c:
  3222-3240) assigns a type only in rain, sand, sun and hail (each under `NO_CLOUD_NINE`); with no weather, fog, or
  Cloud Nine / Air Lock out it returns an uninitialised local (`int result;`, :3129; no `default` for the move). Ruling
  (manager, 2026-11-18), built: the four typed weathers are modelled (`damage.ai_move_type` via `weather_ball_params`,
  `ai_weather_ball_typed`; the switch AI's and the replacement's reads share it); the AI refusal for the other states
  stays (`policy._refused` `move_type`, `switching.weather_ball_refused`), and the tests of that refusal carry `fullset`
  (`test_batch13_refusals_lifted`, `test_weather_ball_move_type_is_untyped_outside_the_four_weathers`,
  `test_party_variable_types_follow_move_calc_variable_type`). The docstring cites now give :3222-3240 (they gave the
  pre-re-index :3209-3227). Blocks 4 Kaizo battles (opponent parties with Weather Ball, `mechanics_backlog.md`); each
  can reach the untyped state (Cloud Nine / Air Lock on the player's side reaches it under any weather). A player's own
  Weather Ball reaches it through the switch AI (`ai/replacement.py:142`) in any battle. Trigger: a real-game check
  (memory dump per caller) when the user next does real-game checks.
- **F-O2 A Transform user facing a Sky Shaymin (F-26) stays refused; first group of E1's next batch (closed: G-1).** Ruling
  (manager, merge gate of batch F): kept as ordinary backlog in this batch, built as the next batch's first group. The
  decomp path: a frozen Transformed copy passes `subscript_check_shaymin_form.s:5-9` (species 492, form not 0, no
  Substitute, Freeze); :10 sets the form to 0; :12-19 load Land's ability by the copy's own personality bit 0; :22-23
  write it under `SYSCTL_RECALC_MON_STATS`, so `BtlCmd_UpdateMonDataFromVar`'s `BattleMon_CopyToParty`
  (battle_script.c:3401-3407) has `BtlIOCmd_UpdatePartyMon` set `MON_DATA_ABILITY` on the Transform user's own party
  Pokemon and recalculate its stats (battle_io_command.c:601-604); :24-27 write Land's types; :30 `RefreshMonData`
  reloads the stats, level, friendship and HP from that party Pokemon (`BattleSystem_ReloadPokemon`, battle_lib.c:
  169-193; the moves stay while transformed, :183); every later switch-in reads the written ability
  (`BattleSystem_InitBattleMon`, battle_lib.c:124). To build: a per-party ability override (per party slot, both
  sides), written by `hooks.shaymin_land` (and the reference's) for a Transformed defender together with the party
  stats recomputed on its own species, and read wherever the engine takes a party Pokemon's ability
  (`damage.decomp_abilities` at switch-in / InitBattleMon, the Transform end's `_restore_ability`, the AI's
  `MON_DATA_ABILITY` reads); then drop `coverage.transform_sky_refusals` and the backlog's F-26 rows. Kaizo: one
  battle, Champion Cynthia (trainer 870, Elite Four Split, a level-100 Sky Shaymin), refused only when your side can
  run Transform; `check_opponent` without your party counts none, so `tools/kaizo/backlog.py` adds the row with your
  Transform user assumed (`mechanics_backlog.md`, "form change").
- **F-O3 Two Starf Berry eats on one side in one turn share one draw.** The game draws `RandNext() % 5` per eat
  (battle_lib.c:4864-4866); the engine has one turn draw per side (`Outcome.starf[side, 0]` -> `starf_roll`,
  `step._action_phase`), so a holder that eats at its pinch and then U-turns / Baton Passes into a second holder that
  eats on entry (the same `BattleSystem_TriggerHeldItem` path, :5273-5294) repeats the first rank, as F-10's Recycled
  berry eaten twice. Needs a third draw column (`starf[side, 2]`) for the pivot newcomer's check with its own law in
  `enumerate._starf_reads`. Not built in this fix round. (Modelled in batch G: G-27..G-30.)
- **F-O4 The AI's absorb switch under its shared gate draw, and Water / Volt Absorb on its bench.**
  `AI_HasAbsorbAbilityInParty` (trainer_ai.c:3640) draws one `RandNext() % 3` when the AI has a super-effective move
  and returns FALSE unless it is 0 (:3651), and only then the bench's `RandNext() & 1` draws (:3706): one sequence the
  independent switch sites cannot express, refused (`switching._Layout.absorb_gate`, raises). The same function's
  Water (:3668-3669) and Electric (:3670-3671) cases are refused outright (`switching._AI_SIDE_ABILITIES`). Both
  raise; to build: the gate as a conditioning draw of the absorb sites, and `checkAbility` by `moveHit`'s type for the
  bench test. (Closed in batch G: G-36..G-38.)
- **F-O5 Hydration's Nightmare clear waits for Nightmare.** `subscript_ability_restore_status.s:6` clears
  `VOLATILE_CONDITION_NIGHTMARE` with the status (F-5); Nightmare is not modelled, so this is part of the Nightmare
  group: when it is built, Hydration's (and every `subscript_ability_restore_status` caller's) cure clears it.
  (Built in batch G: G-31..G-34.)

### Rulings
- F-1 Heatproof (`AbilityId.HEATPROOF`, decomp 85). `BattleSystem_CalcMoveDamage` halves a Fire move's power
  (`movePower /= 2`) for a defender whose `Battler_IgnorableAbility` is Heatproof (battle_lib.c:6841-6844): Mold Breaker
  ignores it (`damage._pre_random`, `d_ability_eff`; reference `_calc_move_damage`); the AI estimate shares the code.
  Burn damage: `subscript_burn_damage.s:6-9` takes `Divide(maxHP, 8)` and, for a Heatproof holder (`CheckAbility`, so
  `Battler_Ability`: Mold Breaker never matters), `Divide(loss, 2)`; `BattleSystem_Divide` floors a positive dividend
  at 1 (battle_lib.c:3599-3618), so the loss is max(1, max(1, maxHP/8) / 2) (`step._burn_check`, reference
  `_burn_check`). Will-O-Wisp is not blocked (no other Heatproof read: `grep ABILITY_HEATPROOF`: lib:6842 and
  burn_damage.s:8 only).
- F-2 Flash Fire (`AbilityId.FLASH_FIRE`, decomp 18; new per-battler `flash_fire` = `MoveEffectsData.flashFire`,
  battle_mon.h:21). Absorb: `BattleSystem_TriggerImmunityAbility` (battle_lib.c:3515-3521): `Battler_IgnorableAbility`
  Flash Fire, moveType Fire, defender not frozen, not `SYSCTL_FIRST_OF_MULTI_TURN`, and move power != 0 or
  Will-O-Wisp (`ids.MOVE_WILL_O_WISP`); no attacker != defender test. Runs where Dry Skin / Water / Volt Absorb do
  (`step._use_move`, reference `_use_move` beside them; bcp.c:3281-3286, :2789-2797: only a move that hit; no
  damage, no effect); no heal: `subscript_absorb_and_boost_fire_type_moves.s:13` sets the flag, the holder's ability
  is revealed (`hooks.REVEALS_ON` ABSORB). Boost: `BattleSystem_CalcMoveDamage` damage x15/10 for the attacker's flag
  and a Fire move, after the weather block and before `+ 2` (:7074-7076; `damage._pre_random`, reference
  `_calc_move_damage`): it reads the flag, so it survives a lost ability and Mold Breaker. The flag clears on a
  switch and a faint (lib:2101-2104, :2185-2188: `VOLATILE_FIELDS`) and Baton Pass does not keep it (lib:2107-2115).
  Enumerate: the absorb law removes a sure flinch; the foe's Fire move on a Flash Fire side changes its own later
  Fire hit's input (`foe_move` clause). AI: `Basic_CheckForImmunity` already read decomp ability 18 (policy.py);
  the AI estimate's boost is its attacker's flag; `AI_HasAbsorbAbilityInParty` (trainer_ai.c:3640-3713), Flash
  Fire's case: a bench Flash Fire holder after a Fire `moveHit` with power (table type, :3668) draws `RandNext() & 1`
  in party order, A's own Flash Fire holds the switch (:3678) (`switching.SwitchSite.ABSORB`). With a super-
  effective move of A the function first reads `RandNext() % 3` (:3651), one draw shared by the bench draws:
  independent sites cannot express it, so that state stays refused (`_Layout.absorb_gate`); Water Absorb / Volt
  Absorb on the AI's bench stay refused wholesale (not in this group). Doubles-only AI parts (TagStrategy,
  IfActivatedFlashFire) and Expert_ChangeUserAbility (Role Play / Skill Swap unmodelled) are not built.
- F-3 Multitype (`AbilityId.MULTITYPE`, decomp 121). `Battler_MonType` (battle_lib.c:7618-7703): an Arceus whose raw
  `BattleMon.ability` (not `Battler_Ability`) is Multitype has both types of its raw `heldItem`'s
  `HOLD_EFFECT_ARCEUS_*` (:7631-7696), Normal for any other item or none (:7697-7699); the stored types and the
  form are ignored, Klutz and Embargo do not hide the item. The party side agrees (`MON_DATA_TYPE_1/2`,
  pokemon.c:1219-1226). `damage.multitype_plate` / `personal_types` (reference `_multitype_plate`,
  `_personal_types`) are used where the engine builds or restores a battler's types: `make_state`,
  `step._restore_types`, `effects.camouflaged`, and Transform (`effects._transform`: copying species Arceus and
  Multitype, the user's own plate decides, not the target's types, :7629-7699; the reverse, an Arceus copying
  another ability, keeps the copied types). The plate's power is the item's table value (Kaizo 50: `tables`,
  nothing hardcoded; `judgment_type`). The protections already in place: Camouflage, Knock Off, Embargo, Trace's
  exclusion; Gastro Acid, Worry Seed, Role Play, Skill Swap, Trick, Thief / Covet, Fling and Conversion are not
  modelled moves, so their Multitype tests are not built (the sheet's rule stands for when they are). The save
  parser's Arceus-form refusal (`saves/pkm.py`) is gone: the form's row is the species' and the types follow the
  plate. AI: type loads read `state.types` (`BattleMon_Get`); the Fling / Gastro Acid / Worry Seed Multitype
  scores belong to those moves.
- F-4 Intimidate and a substitute the action hit. `subscript_intimidate.s:13` `CheckSubstitute` skips a foe with
  the substitute bit or `selfTurnFlags[foe] & SELF_TURN_FLAG_SUBSTITUTE_HIT` (`BtlCmd_CheckSubstitute`,
  battle_script.c:9172-9184), which every hit that lands on a substitute sets, the breaking one included
  (bcp.c:3359-3371), until `BattleContext_Init` (battle_lib.c:2013-2015; every action's end, bcp.c:3998, and the
  turn end, :1855). A U-turn's newcomer enters inside that action, so its Intimidate skips a foe whose substitute
  the U-turn hit even with `substitute_hp` 0. New per-battler scratch `ts_sub_hit` (written at the pause beside
  `ts_damaged`, read at the resumed `switch_in_effects`, `hooks.intimidate(sub_hit=)`); reference
  `_Turn.sub_hit_mon`. An end-of-turn replacement or a turn-start switch finds the flag clear (unchanged). The other
  `CheckSubstitute` users reachable from a switch-in: none (grep: `CheckSubstitute` in `res/battle/scripts`
  switch-in subscripts is Intimidate's alone; the move subscripts that call it are not switch-ins).
- F-5 Hydration (`AbilityId.HYDRATION`, decomp 93). `BtlCmd_EndOfTurnWeatherEffect` (battle_script.c:5626-5639):
  in rain, a battler with HP and any status (`MON_CONDITION_ANY`) whose `Battler_Ability` is Hydration sets
  `scriptTemp = WEATHER_EFFECT_HYDRATION`; `subscript_weather_continues.s` then runs
  `subscript_ability_restore_status.s`: status := `MON_CONDITION_NONE` (sleep and toxic counters live in the status
  bits, so `status_counter` goes with it), Nightmare off (:6; Nightmare is not modelled), message tags the holder
  (`hooks.REVEALS_ON` STATUS_CURE). No HP change; no `MOVE_EFFECT_NO_WEATHER_DAMAGE` test in the rain block, so a
  Diving holder is cured. The whole body sits inside `NO_CLOUD_NINE`, the weather step's own gate
  (`step.field_conditions`: `weather_suppressed`), and runs only when the rain continues (the turn it runs out
  nothing is cured). It sits in `step._weather_effect` (reference `_weather_effect`), read by decomp id from
  `damage.decomp_abilities` (carry 9); the weather step precedes every mon-condition step (bcp.c:900-912, :1252), so
  the cure comes before the turn's poison/burn damage and the held-item status checks. Not a
  `BattleSystem_RecoverStatusByAbility` case (:3966-3968). The AI side was modelled before (Basic_CheckCannotPoison,
  Basic_CheckRainDance / SunnyDay, Expert_RainDance: `ai/policy.py`); coverage no longer refuses Hydration.
- F-6 Forecast (`AbilityId.FORECAST`, decomp 59; `ids.SPECIES_CASTFORM` 351). `BattleSystem_TriggerFormChange`'s
  Castform block (battle_lib.c:6313-6364): for a battler of species Castform with HP and `Battler_Ability` Forecast,
  with no alive Cloud Nine / Air Lock: no rain, sun or hail (`FIELD_CONDITION_CASTFORM`, condition.h:140) and
  neither type Normal -> Normal/Normal; else sun and neither type Fire -> Fire/Fire; else rain and neither Water ->
  Water/Water; else hail and neither Ice -> Ice/Ice; with one alive: neither type Normal -> Normal/Normal
  (:6355-6363). Sandstorm and fog count as none. Only `type1/type2` are written (`hooks.forecast`, reference
  `_forecast`); `formNum` is read by the display only (grep `formNum` in src/battle: battle_controller.c, battle_display.c)
  and the party's form is not updated (no `SYSCTL_FORM_CHANGE`, only Giratina sets it, :6446), so `state.form` stays
  and the switch-in's re-read of the party types (:113-114) restores the personal types. The state is not advanced
  while a change happened (:4115-4121); each result is stable under its own test and no battler's change feeds
  another's, so one pass per call reaches the same fixpoint in any order. Called where
  `BattleSystem_TriggerEffectOnSwitch` runs: `hooks.switch_in_effects` and `hooks.show_battle_mon` after Pressure's
  announcement (FORM_CHANGE follows PRESSURE, :3629-3640), `step._move_end` (every move; bcp.c:3672-3680, :3973),
  and `step._turn_start_items` before its item check (the next turn's ShowBattleMon, bcp.c:256, :1844-1858: a
  weather that ended reverts Castform before commands). Reference: `_switch_in_effects`, `_finish_action`,
  `_end_turn`. Skill Swap / Role Play / Worry Seed keep the current types (no revert code); Transform copies species
  and ability so a Castform copy follows the rule (the species test reads the battler's species). Trace cannot copy
  Forecast (`UNTRACEABLE_ABILITIES`, already). AI: no `script.s` read; type loads read `state.types`
  (`BattleMon_Get`), so the AI sees the changed types.

Held items, hold effects 50, 36, 41, 42, 92, 0, 1, 12 (sheets `item-*.md`). Kaizo's renamed and re-purposed slots take
the id of the hold effect they now have through `effect_map.KAIZO_ITEM_IDS` (keyed by the slot's `spelling()` name and
Kaizo hold effect, used by `build_tables._item_maps` where the slot is not vanilla's); a non-berry never takes a
berry's id, since Pluck / Bug Bite read the item's own pluck effect and `Item_IsBerry` (`hooks.PLUCKABLE`).
- F-7 Macho Brace and the Power items (hold 50 `EVS_UP_SPEED_DOWN`; `ItemId.MACHO_BRACE`, an id with no hook). Battle
  Speed: `BattleSystem_CompareBattlerSpeed` halves a holder of `sSpeedHalvingItemEffects` (50, 106, 117-122,
  battle_lib.c:1160-1169, :1240-1246, :1306-1312) read through `BattleSystem_GetItemData`, so Klutz and Embargo do not
  stop it (`order.effective_speed`, reference `_speed`: both already read the raw hold effect). EVs:
  `BattleScript_CalcEffortValues` doubles every yield for hold 50 (battle_script.c:10289-10291) and adds the item param
  for 117-122 (:10243-10283); Kaizo's Power items 289-294 are hold 50 with param 0 (vanilla 117-122 with 4), so they act
  as Macho Braces (`levelup`, which reads the party mon's raw hold effect and param; Knock Off keeps the party copy,
  io:590-591). `effect_map.HOLD_EFFECT_TO_ID` maps 50 and 117-122 to the id (vanilla's Power items were UNSUPPORTED
  for want of one) and `KAIZO_ITEM_IDS` the six Kaizo Power items (same name, other hold effect). Enumerate: neutral (a
  speed factor of the order fixed at the turn's start, like Iron Ball). AI: `AICmd_IfSpeedCompare*` call
  `CompareBattlerSpeed` (ai.c:1109, :1121), so the halving is seen; `Expert_Trick` and Fling are unsupported.
- F-8 Berserk Gene (Kaizo item 70, hold 36 `PINCH_ATK_UP`, param 0; `ItemId.BERSERK_GENE`). The Liechi path:
  `BattleSystem_TriggerHeldItem` / `_OnStatus` (battle_lib.c:4773-4784, :5196-5207): with param 0, `maxHP / 0` is
  `maxHP` (ruling G-53), so it fires at any HP once Attack is below +6; +1 Attack by the stage routine
  (`subscript_held_item_raise_stat.s:11`), then `PLUCK_CHECK` removes it. It is its own id because Pluck / Bug Bite
  leave it (PLUCK_EFFECT_NONE, not a berry: lib:5832-5835): `hooks._raise_stat` and reference
  `_held_item` take it with `PINCH_BERRY`, `PLUCKABLE` does not. `item_param[i, 1]` is Attack from
  `PINCH_STAT_HOLD_EFFECTS[36]`; `PINCH_DIVISOR_ITEMS` takes any param. Enumerate: the pinch berry's item clause.
- F-9 Lansat Berry (hold 41, param 4; `ItemId.LANSAT_BERRY`). Held: at `curHP <= maxHP / param` (Gluttony halves the
  param) and no `VOLATILE_CONDITION_FOCUS_ENERGY`, the holder gets it (`subscript_held_item_raise_crit.s:11`) and the
  berry is eaten (lib:4838-4848, :5261-5271); already focused: no trigger, the berry stays. Pluck / Bug Bite: the
  eater gains it unless it has it, at any HP, the berry eaten either way (:5819-5825; Klutz / Embargo eater:
  no effect, :5839-5844). `hooks._pinch_effects` (called by `held_item_check` and `pluck_berry`), reference
  `_pinch_effect`. Its crit effect is the existing `focus_energy` (+2, `outcome.side_crit_odds`). The pinch can arrive
  from a foe's hit before the holder's own CalcCrit, so `outcome.crit_variants` has one more variant bit
  (`_lansat_pinch`: a Lansat holder has Focus Energy, used by any non-status move of another battler and by Pain
  Split, the one modelled status move that lowers its target's HP (`effects._pain_split`,
  subscript_pain_split.s:12-22; the defender's item check follows every move, bcp.c:3691-3696); over-
  approximate): `CRIT_VARIANTS` is 16. AI: `Basic_CheckAlreadyPumpedUp` already reads the volatile (ai.s:592-594);
  the Natural Gift and Recycle tables already list the berry (items 149-212).
- F-10 Starf Berry (hold 42, param 4; `ItemId.STARF_BERRY`). At the same pinch (lib:4850-4872, :5273-5294) it raises
  one of Attack, Defense, Speed, Sp. Atk, Sp. Def (`ids.STARF_STATS`: `BATTLE_STAT_ATTACK + i`, i < 5) by 2, capped at
  +6, by a direct `UpdateMonData` write (`subscript_held_item_sharply_raise_stat.s:13-17`: no Mist, no Clear Body;
  `hooks._pinch_effects` uses `change_stage`); with all five at +6 it does not trigger and is kept. The stat:
  `RandNext() % 5` until the drawn stat is below +6 (lib:4864-4866; Pluck :5798-5817, eaten even with all five at +6).
  Ruling: by the project's `% n` convention (`RandNext() % n` uniform, as `Outcome.acupressure`, B12-9) it is uniform
  over the eligible stats, not the exact 13108 / 13107 split the sheet notes (Attack's extra 1 in 65536): the draw is
  `Outcome.starf[side, k]`, one value in [0, `STARF_DRAWS` = 60), and the stat is the eligible one of rank `value %
  eligible` (`hooks.starf_stat`). The draw reaches the item checks, which take no outcome, through
  `BattleState.starf_roll` (STATE.md). Each check outside the turn's actions draws its own (lib:5273-5294 via
  `BattleSystem_TriggerHeldItem`, review F item 5): k = 0 the turn's (`step._action_phase` loads it); k = 1 a faint
  replacement's switch-in check (ShowBattleMon :4151 after `subscript_show_party_list.s:22-24`), kept by
  `step_events` in `starf_roll` (modulo 5: a fresh battler's count) where `step.starf_kept` (the side awaits a
  replacement and a living bench holder is at its pinch or Spikes / Stealth Rock lie on its side) and read by
  `step.replace` (so `search.playout.replacement_pass` and the tree's replacement children), then cleared; hidden,
  like `qc_roll`. The battle start's checks (`hooks.battle_start`) run with `starf_roll` = `STARF_UNDRAWN`: a lead's
  berry is eaten there and its stat waits in `starf_open` for the root's chance node (`step.root_belief`: one state
  per eligible stat at 1 / count, visible, so separate root outcomes; `step.sample_root` for playouts;
  `hooks.resolve_starf`); a `step` given a state still open draws it first from `starf[:, 0]`. Reference: `ref_step`
  and `ref_replace` the same (`_resolve_starf`, `_starf_kept`). The turn-end `_turn_start_items` check (the next
  ShowBattleMon) still reads the turn's draw, and a Recycled berry eaten twice in a turn repeats the pick
  (outcome-variable limit). Enumerate (`_starf_reads`): `starf[d, 0]` only where a holder can eat this turn (the
  active or switched-in one, or any living holder of the party when its U-turn / Baton Pass or a foe's Roar can
  bring it in, at its pinch or able to lose HP: a move, a residual damage source, hazards; a Pluck / Bug Bite of a
  foe holder; a Recycle of one; an open berry); its law the five values while the count stays 5 (the five stages'
  highest plus this turn's bounded gains below +6, none of Belly Drum, Psych Up, the stage swaps, Transform, Baton
  Pass, Recycle), else all 60. `starf[d, 1]`: five values where a holder can enter at its pinch after the turn. An item
  clause on any of the five stats (own and foe's), a Speed +2 grid point for a Roar's drag. AI: the Recycle and
  Natural Gift tables list it.
- F-11 Leek (Kaizo item 259, vanilla Stick; hold 92 `FARFETCHD_CRITRATE_UP`, param 0). Identical bytes in both ROMs
  except the name: `KAIZO_ITEM_IDS` maps the slot to `ItemId.CRIT_ITEM` (+2 crit stages for a Farfetch'd,
  `BattleSystem_CalcCriticalMulti`, battle_lib.c:7128, `outcome.side_crit_odds`).
- F-12 Lava Cookie and Old Gateau (Kaizo hold 12 `STATUS_RESTORE`, param 0, non-berries; `ItemId.LAVA_COOKIE`). Lum's
  trigger (lib:4662-4697, :5052-5087: any status or confusion, no HP test, no Gluttony), cured status and confusion
  as `_restore` / `_cure` do for Lum, and the item removed; Pluck / Bug Bite leave it (PLUCK_EFFECT_NONE, no berry:
  :5832-5835). Enumerate: a status cure (the burn and paralysis clauses of Lum).
- F-13 Big Nugget (Kaizo item 92, vanilla Nugget; hold 0). Only the name and Fling power 250 differ: `KAIZO_ITEM_IDS`
  maps it to `ItemId.NONE` (no hold effect, not a berry: `Item_IsBerry`, src/item.c:357-365; Pluck leaves it,
  lib:5832-5835). Fling is unsupported.
- F-14 RageCandyBar (Kaizo item 71, vanilla Shoal Shell; hold 1 `HP_RESTORE`, param 100, no berry): Berry Juice's
  effect with 100 HP, `KAIZO_ITEM_IDS` maps it to `ItemId.BERRY_JUICE` (not pluckable: lib:5832-5835; trigger
  `curHP <= maxHP / 2`, lib:4585-4591). Natural Gift power 250, type 31: `BtlCmd_CalcNaturalGiftParams`
  (battle_script.c:8088-8102) sets `moveType` 31, a value that is no type id, so `BattleSystem_ApplyTypeChart` finds no
  STAB (`MON_HAS_TYPE`, lib:2592) and no chart row (:2611-2645) and no type-keyed item, ability or weather case matches
  it: `damage.move_type_of` and reference `_move_type` give a Natural Gift type above `Type.DARK` the value
  `Type.MYSTERY` (no row, and no battler's `state.types` holds it: roost only reads Flying as Mystery in the defender's
  chart), as ruling B13-26 does for the AI's estimate; the table reads stay in range.
- F-15 Party friendship wraps apart from the battler's (trainer bag items; the Kaizo Poke Ball of #345 Christine).
  `BattleSystem_UseBagItem` bands by `Pokemon_GetValue(mon, MON_DATA_FRIENDSHIP)` of the **party** Pokemon (src/battle/
  battle_system.c:716-733) and, when the item's amount is nonzero, adds it with `Pokemon_IncreaseValue` (:751) to the
  party's u8 and with `Battler_AddVal` (:752-754) to the battler's. `BoxPokemon_IncreaseDataInternal` sets `newValue`
  to 255 when `friendship + value > 255`, then adds `value` unless the sum is below 0 (src/pokemon.c:1859-1874): the u8
  stores `(255 + value) & 0xFF`, so any gain that crosses 255 stores `value - 1` (254 + 3 -> 2), and a loss below 0
  stores 0. `BattleMon_AddVal` clamps to 0-255 (battle_lib.c:1032-1043). The battler re-reads the party value in
  `BattleSystem_InitBattleMon` (:136) and `BattleSystem_ReloadPokemon` (:179). New per-slot `party_friendship` beside
  `friendship` (STATE.md): `bag.use_bag_item` bands by it and writes both; `step._switch_in` / reference `_enter` copy it
  into `friendship`; a level-up reads it and writes both (`levelup.pay_exp_leveled`; the closed form needs no change,
  a u8 never exceeds 255); `make_state` starts both equal. The refusal is removed (`bag.refused_bag_params`,
  `refused_bag_items`, `AI_BRANCH_PARAMS`, `coverage` "trainer item", `policy._Refused.bag`, `_item_refusals`). Not
  built: the Luxury Ball and egg-location +1 (:737-743), neither true of a trainer Pokemon (Poke Ball, egg location 0,
  pokemon.c:441-442), and a player-side bag item; the save export of the player's party value is `party_friendship`
  (no export path reads it yet). Kaizo's only wrapping trainer item is the Poke Ball of #345 Christine (Lv1 Sandshrew,
  friendship 255): party 2, battler 255; the AI reads the battler value (Return / Frustration, trainer_ai.c:3018-3026).
- F-16 Shaymin-Sky reverts to Land Forme when frozen (Kaizo #870 Cynthia; any side). `BattleControllerPlayer_
  AfterMoveMessage` loads `subscript_check_shaymin_form` after the secondary effect of every hit and before Rage and
  the contact ability (src/battle/battle_controller_player.c:3497-3505 one hit, :3557-3565 per multi-hit hit). The
  script (res/battle/scripts/subscripts/subscript_check_shaymin_form.s:5-30) needs a defender whose species is 492,
  form number nonzero, no Substitute and status exactly `MON_CONDITION_FREEZE`; it sets the form to 0, the ability to
  Land's (personality bit 0 clear: ABILITY_1, else ABILITY_2 when nonzero, else ABILITY_1; `LoadArchivedMonData`
  writes `calcTemp`, battle_script.c:9414-9425, which the following `UpdateMonDataFromVar` reads), raises
  `SYSCTL_RECALC_MON_STATS`, sets the two types from Land's row and runs `RefreshMonData`. `BattleMon_CopyToParty`
  sends `updateForm` / `updateStats` (battle_controller.c:1496-1499): the party form 0, ability, and
  `Pokemon_CalcLevelAndStats` (battle_io_command.c:597-604); `BattleSystem_ReloadPokemon` then reloads the battler's
  stats, HP, level and friendship (battle_lib.c:169-193). `hooks.shaymin_land` (JAX, called from `effects.play_hit`
  right before `_contact_ability`, not for Pursuit's pre-switch hit, which loads no `AfterMoveMessage`) and reference
  `_shaymin_land` (called from `_hit` at the same point): form 0, `ability_slot` kept when Land's row has an ability
  there else 0 (Land's second slot is 0, so Natural Cure), `ability_form` 0, types from the Land row, stats and max HP
  from `calc_stats_jnp` / `_calc_stats` on Land's row with the slot's IVs, EVs, nature and level (HP moves by the max
  HP's change; both rows have base HP 100), `friendship` = `party_friendship`, `power_trick_desync` = `power_trick`
  (the Atk / Def reload is unswapped, as a level-up's, B12-10), and the AI learns the ability (`BattleAI_SetAbility`,
  battle_script.c:3402-3403: `hooks._know`). Stages, status, volatiles and the thaw stay: the freeze goes at the
  later thaw checks. `landed` stands for "the hit reached `AfterMoveMessage`" (a miss or no target does not). For a
  U-turn the decomp runs the form change after the party list; the engine runs it before the pivot (nothing between
  reads form-dependent state, F-4's skip aside). Enumerate: a foe's freezing move on a Sky Shaymin side and a hit on
  a Sky Shaymin frozen at the turn start are input clauses (`enumerate.py`, `sky`, `foe_freezes`, `sky_frozen`).
  Coverage: the "Sky Forme reverts when frozen" refusal is removed; `SPECIES_SHAYMIN` moved to `engine.ids`. A
  Transformed copy of a Sky Shaymin is refused (F-26); `shaymin_land` skips a Transformed battler. The end-of-turn
  speed grid takes the reverted Speed (F-27).
- F-17 A faster foe's Knock Off takes the side's Wide Lens before its accuracy check. `BtlCmd_TryKnockOff`
  (battle_script.c:6608-6632) sets `heldItem = ITEM_NONE` (:6622-6624); the check reads the attacker's
  `HOLD_EFFECT_ACCURACY_UP` (bcp.c:2955-2957). `outcome._threshold_grid` files Knock Off with Embargo (`foe_embargo`):
  the grid's second half, the threshold without the lens, is possible for a Wide Lens holder whose foe can Knock Off
  first (a Zoom Lens or Micle holder's lost item is the first half's, as for Embargo). `future_thresholds` keeps the
  Future Sight roll whole (`loose`) when a Knock Off is used this turn, as for Embargo (the turn-end hit reads the
  items then). Tests: `test_accuracy_classes.py::test_taken_items_and_copied_or_lost_abilities_are_seen_by_hit_
  thresholds` (slow), `test_batch8.py::test_future_sight_roll_is_kept_whole_around_gravity_and_transform`.
- F-18 `future_thresholds`' stage sum follows Haze, Psych Up and Acupressure (B12-31's writers): Haze
  (battle_script.c:3534-3549) zeroes both stages, Psych Up (:6042-6053) copies them, Acupressure (:8932-8956) can
  raise evasion, before the turn end's hit (bcp.c:1758-1783); such a move used this turn frees the sum (`free`), as a
  stat move on accuracy or evasion does. Nothing else in enumerate marked them loose. Gravity at 1 keeps the roll
  whole only where it ends (FIELD_COND_CHECK_STATE_GRAVITY, bcp.c:1227-1238): not under `Tables.gravity_rearm`
  (Kaizo's sub_seq 238 sets it again, G-67), where it holds at the hit. Tests: `test_batch8.py::
  test_future_sight_law_follows_a_haze_this_turn`, `::test_future_sight_roll_is_kept_whole_around_gravity_and_transform`.
- F-19 Learn masks lift once the battle is over. `MoveEnd` runs `AnyExpPayout` before `CheckBattleOver`
  (bcp.c:3965-3969): the payout that ends the battle runs no later turn end (the Disable / Encore checks,
  bcp.c:1565-1576, :1591-1602) and no move (the pinned slot's re-read, :1891-1895). `levelup.learn_options` and the
  reference's `ref_learn_options` mask the disabled, encored and pinned slots only while `state.over == 0`; the
  converter's `_learn_refusal` follows (it asks `learn_options`), and its unreachable D10 message is gone (an HM is
  never fitted, D10 is refused first by `_check_learned`). The end-of-turn Disable and Encore checks end on a move no
  longer known in both engines (`step._disable_check`, `_encore_check`). Spite, Grudge and Last Resort have no engine
  reader (their moves are unsupported); Mimic's `_masked` slot takes a learned move into the shadow (`levelup.learn`).
  Test: `test_learn.py::test_no_mask_once_the_battle_is_over`.
- F-20 A stale Choice lock is cleared at the command selection. For every battler whose command is FIGHT and that
  picks one (`Battler_CanPickCommand`, bcp.c:335; battle_lib.c:6224-6235), `BattleSystem_CheckInvalidMoves`
  (bcp.c:433) clears `choiceLockedMove` when the Choice holder no longer knows it (battle_lib.c:2307-2310).
  `step._clear_stale_choice_locks` (called from `_action_phase` on the fresh, unpaused state) and the reference's
  `_ref_step` prologue do the same, so a move forgotten and learned again later does not revive the lock. Test:
  `test_damage_items.py::test_a_choice_lock_the_holder_no_longer_knows_is_cleared_at_selection`.
- F-21 Ether / Elixir on a Transform or Mimic copy. `BattleSystem_UseBagItem` tests and restores the party Pokemon's
  PP (`Pokemon_GetValue` / `Pokemon_IncreaseValue`, battle_system.c:631-632, :646-648) and adds to the battler's PP
  only for a slot that is not mimicked and a battler not transformed (:634-639, :651-654). `bag.use_bag_item` and the
  reference's `_use_item` read a copied slot (`mimic_slots`, all four while transformed) from the shadow
  (`shadow_moves`, `shadow_pp`) and restore `shadow_pp`; the battler's PP stays. Test:
  `test_e1_batch_a.py::test_elixir_restores_the_party_pp_of_a_copied_slot`.
- F-22 `gravity_rearm` is read from the ROM diff, not the game id. `tools/build_tables._gravity_rearm`: 1 when
  sub_seq 238 (subscript_gravity_end) is among the members the ROM diff changed (`rom/changed_scripts.json`, written
  by `tools.extract_rom` from both ROMs) and `data/kaizo/rom_diff.toml` classifies it `modelled` (its note decodes
  the re-arm); a changed member classified otherwise stops the build; 0 when unchanged (vanilla). Values unchanged
  (0 vanilla, 1 Kaizo). The build's one game branch (`tools/build_tables.py:730`, `_gravity_rearm(inter) if kaizo
  else 0`) is vanilla being the diff's baseline: it has no `rom/` diff to read. Test: `tests/tools/test_build_tables_game.py::
  test_gravity_rearm_reads_sub_seq_238_from_the_rom_diff`.
- F-23 ROM confirmations from the repo's ROM diff (`tools/rom/diff.py`, read from both ROMs by `tools.extract_rom`;
  `data/intermediate_kaizo/rom/changed_code.json`, `data/kaizo/rom_diff.toml`): (1) `sMovesAffectedByGravity`
  (battle_lib.c:3385-3392, overlay16) is unpatched: the changed overlay16 keys are only
  BattleControllerPlayer_CheckObedience, _CheckMoveHitAccuracy, _CheckMoveHitOverrides,
  BattleSystem_TriggerEffectOnSwitch and sTrainerClassPrizeMul (rom_diff.toml:549-578), so the ban stays by id
  (Kaizo move 150 Hidden Power Water is not on it; `ids.GRAVITY_MOVES`, G-67). (2) The Kaizo AI scripts match
  vanilla's but one word: `tr_ai_seq.narc` is unchanged (rom_diff.toml:583) and the overlay14 script data
  `gTrainerAITable` differs only at 0x022249CC (script.s:78, Dry Skin for Levitate, rom_diff.toml:543-546); none of
  the script.s lines the E1 trial cites (G-63..G-70: :29, :62, :68, :112, :288-295, :438-443, :472, :534-536,
  :720-723, :766, :853-871, :940, :1026-1029, :1461-1474, :1516, :3633-3649, :4336-4340, :4373-4377, :4441,
  :5331-5334, :6511) is :78.
- F-24 Cite fixes (reviews of bb24193 + 9a9b63e, a968143): Transform's copy loop is battle_script.c:5346-5348
  (5342-5345 is pointer setup), the switch-in checks rerun at bcp.c:3672-3673 and :3973 (the announce flags are
  cleared at battle_script.c:5350-5358), Sand Veil / Snow Cloak bcp.c:2927-2933, the Gravity countdown
  bcp.c:1227-1238, `BtlCmd_TryMimic` ends at :4524; fixed in outcome.py (`armored`, `_threshold_grid`), enumerate.py,
  effects.py, ids.py, state.py, STATE.md, tools/effect_map.py (comment only) and this ledger. Tests owed by 9a9b63e,
  in F-17's test: copied Drizzle (Thunder in rain, bcp.c:3025-3027), copied Unaware (bcp.c:2882-2884), the foe's own
  No Guard (bcp.c:3016-3021) and Cloud Nine lost by its Transform; the future Gravity / Transform `loose` flag in
  F-18's.
- F-25 A foe that loses its own Cloud Nine / Air Lock by its Transform (found by F-24's test, silently wrong before):
  `NO_CLOUD_NINE` (bcp.c:2915, :3023) counts the battlers' abilities as `damage.decomp_abilities` reads them, from the
  species row and ability slot, which `BtlCmd_Transform` copies (battle_script.c:5346-5348, battle_mon.h:57-84). The
  grid's Transform variant (`_threshold_grid`'s `point`) copied only `ability`, so the transformed foe still
  suppressed the weather: in sun the player's Thunder kept 70 where the live check reads 50 (bcp.c:2915-2917), and the
  rolls 51-70 were one class with the hits. The variant now also copies the side's `species`, `ability_form` and
  `ability_slot` onto the foe. Test: the sun case in `test_taken_items_and_copied_or_lost_abilities_are_seen_by_hit_
  thresholds` (law {1: 50, 51: 20, 0: 30}).
- F-26 A Transform user facing a Sky Shaymin is refused (review F item 4). `subscript_check_shaymin_form.s:6-9` tests
  only the defender's `BATTLEMON_SPECIES` 492, a nonzero `BATTLEMON_FORM_NUM`, no Substitute and Freeze, all of which
  a Transform copy holds (`BtlCmd_Transform` copies the BattleMon up to `ability`, battle_script.c:5346-5348,
  battle_mon.h:57-84). For such a copy the script then: sets the battler's form to 0 (:10); takes Land's ability by
  `BATTLEMON_PERSONALITY` bit 0, the user's own personality (not in the copied range); writes it with
  `UpdateMonDataFromVar` under `SYSCTL_RECALC_MON_STATS` (:22-23), whose `BattleMon_CopyToParty`
  (battle_script.c:3401-3407) sends `updateStats` (battle_controller.c:1496-1499), so `BtlIOCmd_UpdatePartyMon`
  writes that ability into the Transform user's **own party Pokemon** and recalculates its stats on its own species
  (battle_io_command.c:597-604); writes Land's types (:24-27); and `RefreshMonData` (:30, battle_script.c:9438-9447)
  reloads Attack, Defense, Speed, Sp. Atk, Sp. Def, level, friendship and HP from that party Pokemon
  (`BattleSystem_ReloadPokemon`, battle_lib.c:169-193: the copied stats are lost, the moves kept while
  transformed). The party Pokemon keeps the written ability for every later switch-in (`BattleSystem_InitBattleMon`
  reads `MON_DATA_ABILITY`, battle_lib.c:124). The engine has no party ability apart from the species row and
  ability slot (`damage.decomp_abilities`), so modelling it needs a new per-party mechanic: refused instead.
  `coverage.transform_sky_refusals` (called by `check_opponent`): one side can run Transform (its moves, a player's
  learnable moves, Metronome's calls, or Mimic / Copycat while the fight has Transform) and the other has a Shaymin
  with a nonzero form. Test: `test_e1_batch_f.py::test_a_transform_user_facing_a_sky_shaymin_is_refused`
  (`fullset`). Open item F-O2: the next batch's first group.
- F-27 The end-of-turn speed grid takes Shaymin's Land Speed (review F item 6). The revert recalculates the stats on
  Land's row (base Speed 127 -> 100; F-16) mid-turn, before `SortMonSpeedOrder` at the turn's end, so
  `enumerate._end_speeds` takes every grid point also with a Sky Shaymin's Land stats (`hooks.land_stats`, shared
  with `shaymin_land`) where `land_free`: a living Sky Shaymin of the side and a foe's move this turn, with the
  Shaymin frozen at the turn start or the foe's move able to freeze. Test: `test_the_shaymin_sky_clauses_equal_the_
  bruteforce`.
- F-28 Review F fixes: F-4's flag is read by the resumed action only (`resumed & ts_sub_hit`) and cleared at that
  action's end (`BattleContext_Init`, battle_lib.c:2013-2015, bcp.c:3998), so a later drag's Intimidate in the same
  turn meets a clear flag, as the reference's `turn.sub_hit` (test: `test_a_later_drag_in_the_same_turn_finds_the_
  substitute_flag_clear`). F-2's AI absorb switch takes residue 1 of `RandNext() & 1` (trainer_ai.c:3706:
  `_SWITCH_LO` / `_SWITCH_HI` 1, 2). Brute-force checks of the batch's laws: `test_the_starf_laws_equal_the_
  bruteforce`, `test_the_lansat_crit_variant_equals_the_bruteforce`, `test_the_flash_fire_foe_move_clause_equals_
  the_bruteforce`, `test_the_shaymin_sky_clauses_equal_the_bruteforce`.
- F-29 Merge-gate fixes. JAX healed a Flash Fire holder below full HP by a quarter on an absorbed Fire move (the heal's
  gate took every absorb); the heal is now Volt / Water Absorb's only (`hpCalcTemp`, battle_lib.c:3503, :3510; Flash
  Fire's branch :3515-3521 sets none, `subscript_absorb_and_boost_fire_type_moves.s` changes no HP), as the
  reference. Test: `test_a_holder_below_full_hp_is_not_healed`. F-26's test carries `fullset`. Open items F-O2..F-O5.
  F-16's `landed` gate checked against bcp.c:3497-3505 / :3557-3565: `AfterMoveMessage` runs after `UpdateHP` for
  every connected hit, damaging or not, and after a substitute hit (:3376, :3440, :3443; a substitute broken by the hit
  no longer blocks the check, :3362, which `substitute_hp` 0 gives), never after a miss or `NO_MORE_WORK` (:3313-3328,
  `subscript_missed` -> `LOOP_FAINTED`). One fix: the script reads `BTLVAR_DEFENDER`, which for a self-targeted move
  (RANGE_USER, USER_SIDE, FIELD, and ALLY / USER_OR_ALLY in singles; `targets_user`) is the user (battle_lib.c:
  1665-1684), so `effects.play_hit` and the reference's `_hit` no longer revert a frozen foe Shaymin on a Swords Dance.
  Test: `test_a_foes_self_targeted_move_leaves_a_frozen_shaymin_sky`.
- Open (not modelled; cites where stopped):
  - Sleep Talk's sealed set after a faster foe that already imprisons Transforms (copies every move) or Mimics (one
    slot): the call reads the foe's moves when it runs (CHECK_INVALID_IMPRISONED, battle_lib.c:2287-2290), but
    `enumerate.call_laws` offers `CALL_SEALED` only when no foe imprisons at the turn start (`can_seal`,
    enumerate.py:3701-3706) and `_split_events` sees only an Imprison landing (`sealed_at_move`, step.py:903). Needs a
    hypothetical state of the foe's moves after the copy and a step event of the sealed set at the action: not a
    bounded change, left for the next batch.
  - Effect Spore's contact roll (battle_lib.c:4214-4335) and Suction Cups against Roar: both map to
    `AbilityId.UNSUPPORTED`, so coverage refuses every holder (`IMPLEMENTED_ABILITIES`, coverage.py:194, :743).
    Vanilla holders: 4 personal rows with Effect Spore, 3 with Suction Cups (`species_abilities_decomp` 27, 21); no
    Kaizo species has either (`data/intermediate_kaizo/rom/species.json`), so the Kaizo backlog does not undercount.
  - `test_transform_laws.py`: `assert_equals_bruteforce` in its four tests, a copied Cute Charm with the kept-gender
    rule (battle_script.c:5732-5735), `present_guard` with a copied Wonder Guard, Transform through an invoker: not
    written in this batch.
- For E2 (doubles; not changed here):
  - `chatot_cry` is per party (state.py:747); `BattleSystem_GetChatotCry` (battle_system.c:915-924) reads it per
    battler in 2vs2 and for a tag battle's enemy battlers, per side (`battler & 1`) in other doubles. Check that the
    party index gives that in every format.
  - `outcome._threshold_grid`'s "one move has one effect" (one foe move between the turn start and the side's check)
    holds with one foe only; with two foes both moves can land before the check (the D-steps).
## Batch G

- G-1 A Transform user's Sky Shaymin revert (F-O2, closes F-26): modelled, nothing is refused. The copy of a frozen
  Sky Shaymin passes `subscript_check_shaymin_form.s:5-9` (Transform copies the BattleMon up to `ability`,
  battle_script.c:5346-5348). The revert writes Land's ability into the Transform user's OWN party Pokemon:
  :23's `UpdateMonDataFromVar` runs `BattleMon_CopyToParty` (battle_script.c:3401-3407) under
  `SYSCTL_RECALC_MON_STATS`, which sends `updateStats` and `updateForm` (battle_controller.c:1496-1499), and
  `BtlIOCmd_UpdatePartyMon` sets `MON_DATA_FORM` (form 0 from :10) and `MON_DATA_ABILITY` and recalculates the
  stats (battle_io_command.c:597-604). New state `party_ability[P, 6]` (0 = no override); `hooks.shaymin_land`
  and the reference's `_shaymin_land` drop the Transformed term. For a Transformed battler: the copy takes
  form 0, Land's ability, ability slot / form and types; `party_ability` := Land's ability; `shadow_form` := 0;
  the party's own stats (its own species at form 0, its IVs from `shadow_ivs`, EVs, level, nature:
  `hooks.own_stats`) go to `shadow_stats`, `max_hp` / `hp` (by the max-HP delta) and, through `RefreshMonData`
  (:30, `BattleSystem_ReloadPokemon`, battle_lib.c:169-193), to the copy's Atk..Spe: the copy has the user's own
  stats, not Land's. Coverage: `coverage.transform_sky_refusals`, its call in `check_opponent` and the backlog's
  F-26 rows are deleted; `enumerate.damage_inputs_can_change`'s `sky` loses its `transformed == 0` term; the end
  speed grid's Land point is `hooks.reverted_stats` (the user's own Speed for a Transformed battler).
- G-2 The Land ability slot is the battler's own personality bit (:12-19: `BATTLEMON_PERSONALITY` is not in the
  copied range, battle_mon.h:105-106): `shadow_ability_slot` while Transformed (the own `ability_slot` saved by
  Transform), the battler's `ability_slot` otherwise; slot 1 falls back to slot 0 when Land's ability 2 is empty
  (:18-19). Kaizo's Land Shaymin is [Natural Cure, -]: always Natural Cure.
- G-3 Later reads of the written ability go through `ability`, not a second lookup. `step._restore_ability`
  (battle_lib.c:124, `InitBattleMon` reads `MON_DATA_ABILITY`) puts `party_ability` into `ability` when the
  Transform user leaves (the shadow is restored first, so species and slot are its own), so the bench slot
  reads as written (`damage.decomp_abilities`, whose written-ability branch needs no new term; its readers: the
  AI's bench reads, Natural Cure at switch-out, Heal Bell's Soundproof) and a new Transform copies the target's
  ability over it again (:5346-5348). The AI learns Land's ability at the write (`_know`,
  battle_script.c:3401-3403). `AI_IsAsleepWithNaturalCure` is still refused (`switching._AI_SIDE_ABILITIES`): an
  AI Ditto with the override now reaches that refusal, which names it. A state built with `party_ability` set
  must also set `ability` (the planner that carries a player's override forward does both).
  Also fixed: the empty-slot fallback tested `AbilityId` 0, but an empty slot maps to `AbilityId.NONE` (1); it tests the decomp id now (:18-19). Tests: `tests/engine/test_e1_batch_g.py`.
- G-4 FCM step 1 (FCM.md 4.4; fcm-verify corrections (a)-(d)): `field_mask` (int32, `_FIELD`, `BattleContext.
  fieldConditionsMask`, battle_context.h:210, bits condition.h:110-149 as `ids.FieldCondition`) and
  `weather_turns` (int32, `FieldConditions.weatherTurns`, battle_context.h:29) beside the old fields; readers
  unchanged. Writers (`engine.field`, the reference the same): start condition -> PERM bit or DEEP_FOG
  (subscript_overworld_*.s:11, lib.c:3664-3696), Kaizo area Gravity / Trick Room counters (sub_seq 297;
  subscript_overworld_trick_room.s:14) (`ids.start_field_mask`); weather moves WEATHER off + TEMP bit +
  weatherTurns (effect_script_0115/0136/0137/0164.s:8-14); weather abilities WEATHER off + PERM bit, skipped
  while their own PERM bit is set (lib.c:3749-3776; subscript_drizzle.s etc. :11-12), at the switch-in, the
  move-end recheck and the weather war's swap (`hooks.swap_war`); the TEMP-only countdown in bcp.c order
  (:1113-1211, `--weatherTurns == 0` at :1121, :1146, :1171, :1196; the end scripts clear the TEMP bit, :5);
  Gravity `|= GRAVITY_INIT` (subscript_gravity_start.s:9), `-= 1 << 12` (bcp.c:1229), Kaizo re-arm at 0
  (sub_seq 238, G-67); Trick Room toggle (effect_script_0259.s:5-12), `-= 1 << 16` (bcp.c:1824) and Kaizo's
  re-arm at 0 (sub_seq 251 = the move's script, :6; manager ruling 8: E3's `_trick_room_check` stays the one
  implementation and writes the bits too); Uproar bit 8 + b on (subscript_uproar.s:9-11), off at its end
  (bcp.c:1528, :1534), switch (lib.c:2135) and faint (lib.c:2213, `step._reset_battler`). The views that
  write the old fields write the bits too: `order.gravity_state`, `outcome`'s foe-Gravity variant,
  `outcome.pivot_view` (the pivot's Uproar bit), `playout._sort_view`, the battle-start sort's no-weather view.
  Acid Rain (subscript_pursuit.s:115) is not written yet: FCM 4.4 places it after the readers move (its diff is
  the stated one). Tests: `tests/engine/test_field_mask.py` (clean mask = old fields on each writer path,
  engine = reference). Owed for step 2: `tests/strategies.py` and hand-built test states write the old
  fields only; they must set `field_mask` / `weather_turns` before the readers move.
- G-5 Motor Drive also tests ``attacker != defender`` (battle_lib.c:3532-3536), as Volt Absorb does
  (:3500-3502); Water Absorb, Dry Skin and Flash Fire do not. Whoever models Motor Drive gates it like
  Volt Absorb (`volt & ~self_target`, step.py `absorbed`). Not modelled here (batch F gate note).
- G-6 FCM step 2, weather family (FCM.md 4.4 (2), 4.7 weather; fcm-verify (b)): every reader of the weather
  bits (0-7, deep fog 15) and `weatherTurns` reads `field_mask` / `weather_turns` through `engine.field`
  (`weather_bits`, `rain` .. `fog` = WEATHER_IS_*, include/battle/common.h:68-72; `any_weather`, `solar_down`,
  condition.h:131-139), the reference the same; `weather` / `weather_timer` are written only (step 3 drops them).
  Each reader tests its own group, so several set groups read as the decomp reads them (FCM 3.2): damage
  (lib.c:6921-6940, :7046-7070: rain's step then sun's, SOLAR_DOWN); Weather Ball power on any WEATHER bit, type
  of the last set group rain < sand < sun < hail (battle_script.c:6847-6865, lib.c:7896-7911, trainer_ai.c:
  3223-3238; `field.weather_ball_type`); the AI's `LoadCurrentWeather` last writer rain < sand < sun < hail < fog
  (trainer_ai.c:1414-1438); accuracy: Thunder-flag sun 50 then rain sure, Sand Veil, Snow Cloak, fog (bcp.c:
  2915-2935, :3025-3029), where Kaizo's effect 141 (Hurricane) already carries `MoveFlag.THUNDER` (B13-3), so
  fcm-verify (b) (rom_diff.md:366-367) needs no new branch; Swift Swim / Chlorophyll (lib.c:1229-1235); the
  weather moves' fail tests (effect_script_0115/0136/0137.s:5 the group, 0164.s:5 HAILING_TEMP only:
  `field.weather_move_fails`); the weather abilities skip on their own PERM bit (lib.c:3749-3776; the AI
  reveal follows); WeatherHPRecovery (script.c:5970-5976); Leaf Guard / freeze / SolarBeam charge (any SUNNY
  bit, script.c:9195-9206, bcp.c:2320-2323); Forecast (lib.c:6318-6352, if-else). The turn end runs one weather
  state per set group in bcp.c order (:1113-1225, `field.weather_state`, a `lax.scan`): a TEMP-only group counts
  the shared `weather_turns` down (two such groups: twice; past 0 it never ends, the u32 wrap), and each group
  that goes on runs EndOfTurnWeatherEffect on the whole mask, last writer of hpCalcTemp sand < sun < hail < rain
  (script.c:5573-5625; Ice Body at full HP writes nothing), Hydration's scriptTemp 1 skips the HP step
  (subscript_weather_continues.s:13), a chip under any SUNNY bit tags the ability (:17); the faint check, the
  payout and the battle end run after each state (the loop head, bcp.c:938-944). The outcome / enumerate what-if
  weathers are weather bits (`field.ability_bits`, `field.move_bits`); the observation's weather is
  `field.weather_seen` (the one group's `Weather` id, or 8 + a bit per group), printed by `field.weather_text`.
  A clean mask reads as before (4.4). Prerequisite commit: `tests.strategies` and the hand-built test states set
  a clean mask (`ids.clean_field_mask`, `field.with_weather`), the G-4 owed item; the snap capture for this
  step is taken there. Owed for the Acid Rain commit (corrupted masks become reachable there): the Future Sight
  threshold grid's weathers (start, none, sand, hail) and `_threshold_grid`'s (start, the foe's move's) cover
  clean masks; a corrupted start can end the turn as itself minus an ended TEMP group (0x8044 -> 0x8040). Not
  built: Forecast under sun + rain bits re-triggers forever (lib.c:6327-6343 alternate Fire / Water); Forecast
  stays refused (FCM 4.5). Gravity, Uproar and Trick Room readers unchanged (their commits). Tests:
  `tests/engine/test_field_mask.py` (readers on 0x7F, 0x5, 0x1, 0x3F, fog; turn end on 0x7F stale 3; 0x5 twice
  and the wrap; Snow Warning cleaning 0x7F; Rain Dance failing under 0x7F, Hail under 0x3F; engine = reference),
  `tests/ai/test_kaizo_commands.py::test_load_current_weather_takes_the_last_set_group`.
- G-7 Defog (#258, fog clear and side-condition clears) is E3's (claude/e3-batch2 400c898; manager ruling
  19 Nov). Its fog clear is one more weather writer for the field mask: it must clear the fog bit in
  `field_mask` (engine.field writers, G-4) and Defog's readers read the mask (G-6). Whichever of batch G and
  E3 batch 2 reaches main second converts it on its merge of main.
- G-8 FCM step 2, Gravity family (FCM.md 4.4 (2), 4.7 Gravity): every Gravity reader reads the mask's bits
  12-14 through `engine.field` (`gravity_on`: any `FIELD_CONDITION_GRAVITY` bit; `gravity_count`: the counter),
  the reference the same (`_gravity_on`); `gravity` is written only (step 3 drops it). Readers: the move ban
  `Move_FailsInHighGravity` (battle_lib.c:3398; selection, the status check, the invokers' calls), Levitate as
  none (`Battler_Ability`, :3095), the chart's grounded flag (:2545, :2701, :2766), `BattlerIsGrounded` (:5506:
  `damage.grounded`, Arena Trap's `actions.trapped`, :3215), accuracy x10/6 (bcp.c:2972), the Gravity move's fail
  test (effect_script_0215.s:5), Magnet Rise's Levitate test, the hazards check (subscript_hazards_check.s:6),
  the AI's chart flags (trainer_ai type-chart, `replacement.post_ko_switch_in`, `switching`) and
  `IfFieldConditionsMask` (trainer_ai.c:1936-1946, script.s:1028, :6920); the views: the `gravity_at_move`
  event and `enumerate.gravity_events`, `order.gravity_possible`, outcome's Future Sight `ends` (a count of 1),
  the observation (`observe._FIELD_VIEWS`). The countdown (bcp.c:1227-1238) re-arms and runs the grounding
  loop where the tick takes the bits from 1 to 0 (`field.gravity_tick`; Kaizo sub_seq 238, G-67), so a borrowed
  counter of 7 (FCM 3.1) runs 7 turns and then never ends in Kaizo. `field.with_gravity` writes the bits and
  the old field (the views `order.gravity_state`, outcome's foe Gravity; hand-built test states,
  prerequisite commit a397bce, the snap base). Clean masks read as before (4.4). Tests:
  `tests/engine/test_field_mask.py` (0x7FFF, 0x5000 - 3, 0xFFF readers; 0x7000 to the Kaizo re-arm; engine =
  reference), `tests/ai/test_kaizo_commands.py::test_if_field_conditions_mask_gravity_reads_the_mask`.
- G-9 FCM step 2, Uproar family (FCM.md 4.4 (2), 4.7 Uproar): the field's Uproar readers read the mask's bits
  8-11 (`field.uproar_any`: any `FIELD_CONDITION_UPROAR` bit; `effects.uproar_on` is gone), the reference the
  same (`_uproar_on`): a sleeper wakes at its move (bcp.c:2432, subscript_wake_up.s:6), sleep does not land
  (subscript_fall_asleep.s:16, :45, `_inflict`, the sleep move's `kept_awake`), Yawn's sleep and block
  (subscript_yawn.s:16), Rest fails (subscript_rest.s:11). `uproar[b]` stays the battler's
  VOLATILE_CONDITION_UPROAR count and is still read where the decomp reads the volatile: Uproar's own start
  test (effect_script_0159.s:5), the lock (B5-5), the end check (bcp.c:1504-1535) and Metronome's count
  (battle_lib.c:6160-6198, `statusVolatile & VOLATILE_CONDITION_UPROAR`). So a spurious bit (FCM 3.1, e.g.
  0xFFF) keeps every battler awake until its battler switches or faints (battle_lib.c:2135, :2213). Not
  built, no engine site: CheckObedience's nap (bcp.c:2204; Kaizo always obeys, rom_diff.md:365) and
  DecrementPP's second-call `MON_IS_UPROARING` (bcp.c:2287-2293; the engine models the first call only).
  No hand-built test state sets `uproar` (tests.strategies derives the bits), so no prerequisite commit: the
  snap base is the Gravity commit. Tests: `tests/engine/test_field_mask.py::
  test_spurious_uproar_bits_keep_everyone_awake_until_their_battler_leaves` (0xFFF; engine = reference).
- G-10 FCM step 2, Trick Room family (FCM.md 4.4 (2), 4.7 Trick Room; fcm-verify (a); manager ruling 8): the
  Trick Room bits are a real model (E3's Kaizo re-arm, sub_seq 251), not "carried, refused". Every reader tests
  any `FIELD_CONDITION_TRICK_ROOM` bit (bits 16-18, condition.h:123-125, :144) through `field.trick_room_on`,
  the reference the same: `CompareBattlerSpeed`'s reversal (battle_lib.c:1432; `order.compare_speed`, which
  the AI's speed compare also takes), the move's toggle (effect_script_0259.s:5, `field.trick_room_toggle`),
  the countdown (bcp.c:1822-1827, `field.trick_room_tick` inside E3's one `step._trick_room_check`, which
  stays the one implementation), `IfFieldConditionsMask` (trainer_ai.c:1936-1946; script.s:364, :439,
  :1004, :1131), now `field_mask & operand` for both supported operands, and the observation
  (`observe._FIELD_VIEWS`). `trick_room` is written only (step 3 drops it). A borrowed counter t (FCM 3.1)
  runs t turns; t = 1 ends silently in vanilla and re-arms in Kaizo. `field.with_trick_room` writes the bits
  and the old field for hand-built test states (prerequisite commit ff665e3, the snap base). Clean masks read
  as before (4.4). Tests: `tests/engine/test_field_mask.py` (0x40000 - 3 reverses the order, 0xFFFF does not;
  the toggle under a borrow; 0x1FFFF ends / re-arms; engine = reference),
  `tests/ai/test_kaizo_commands.py::test_if_field_conditions_mask_trick_room_reads_the_mask`. Acid Rain
  (subscript_pursuit.s:115) is not part of step 2 and is not built here.
- G-11 FCM step 3 (FCM.md 4.2, 4.4 (3)): `weather`, `weather_timer`, `gravity` and `trick_room` are dropped
  from `BattleState`, `RefState`, `layout` and STATE.md; `field_mask` / `weather_turns` are the only
  field-condition state (`fieldConditionsMask`, battle_context.h:210; `weatherTurns`, :29). `uproar[b]` stays
  the battler's VOLATILE_CONDITION_UPROAR count (G-9: read where the decomp reads the volatile); no field-wide
  Uproar read of it was left. `make_state` keeps its `start_condition`, `gravity`, `trick_room` inputs and
  writes only `ids.start_field_mask` of them. `field.with_weather` / `with_gravity` / `with_trick_room` write
  the mask only. The golden-trace converter (`tools/trace_to_state.py`) compares and carries the traced
  `fieldConditionsMask` bit for bit as `field_mask` (was: one decoded `weather`; `decode_weather` is gone).
  Test reads of the dropped fields go through `tests.engine.helpers.weather_of`, `weather_timer_of` (FCM
  4.4's `where(temp_only, weather_turns, 0)`), `gravity_of`, `trick_room_of`; `tests.strategies` draws the
  field into the mask (`_set_weather`, `_drawn_weather`, `ids.clean_field_mask`). Snap diff: the four
  dropped fields only.
- G-12 FCM step 4, Acid Rain (FCM.md 1, 3, 4.3; fcm-verify (c), (d)): modelled in every format, nothing refused.
  At a Pursuit KO of the switcher (subscript_pursuit.s:39 `_166`) :100 copies `faintedMon` to `scriptTemp` and
  :115, after the payout loop (:102-113), runs `fieldConditionsMask = sub_to_zero(mask, scriptTemp)` (OPCODE_
  SUB_TO_ZERO on the signed int, battle_script.c:3282-3287; `field.sub_to_zero`, `field.acid_rain`). Engine:
  `step._acid_rain`, called at the one Pursuit site of `_execute_actions` after the pass's faint check and
  `pay_exp` (the switch's chase and U-turn's, subscript_attack_then_switch_out.s:54); reference: `_pursuit`
  returns the operand, `_acid_rain` applies it after `_faint_check` / `_pay_exp` in `_switch` and the U-turn
  site. `faintedMon` (FCM 1.3): the pursuer when it is at 0 HP after the chain (Rough Skin, Aftermath, Jaboca
  Berry, Life Orb: UPDATE_HP's TryFaintMon, battle_script.c:1767-1770) and the switcher held no Destiny Bond
  (the game's order: Destiny Bond at :85 before every recoil, the pursuer always alive there; it restores
  `faintedMon`, subscript_faint_check_destiny_bond.s:10, :25), else the switcher. The mask read at :115 (FCM 1.4,
  fcm-verify (c)): the switcher's Uproar bit is gone (its FAINT_MON, :36) and, after Destiny Bond, the
  pursuer's (:16 PlayFaintAnimation -> battle_script.c:1803 -> battle_lib.c:2213); a recoil-fainted pursuer's
  bit is still set (TryFaintMon only, battle_script.c:1769-1772) and cleared after the subtraction by its own
  cleanup (`kept`). fcm-verify (d): `step.first_pursuer` (reference `_pursuit`) takes the first ready battler
  of the other side in index order (battle_script.c:6899, `break` :6934; not acted, :6900, the caller's test: both singles sites run before the
  other side acts),
  not the fastest; `pursuit_fires` reads it; in singles it is the foe as before. Battlers are passed in
  (doubles-state-design section 5): the site's pursuer is `first_pursuer`'s for the switch and for U-turn's chase
  (no `_foe(side)` left there; in singles the foe either way, review G 11). E2's B = 4 Pursuit site (G-13, not on
  this branch) must call `first_pursuer` with its own not-acted test (:6900) and `_acid_rain` with the pursuer it
  found; `_use_move`'s `chases` still tests U-turn's target, the foe in singles (review G 8). G-6's owed item: `outcome.future_thresholds`' weathers are none,
  sandstorm, hail, each with and without the start's deep fog bit (`_rate` reads sand for Sand Veil, hail for
  Snow Cloak and fog; a defender holds one ability): every mask at the hit, a corrupted start that loses a TEMP
  group included (0x8044 -> 0x8040), since weather moves and abilities clear fog (effect_script_0115.s etc. :8,
  subscript_drizzle.s etc. :11). Where an Acid Rain write can happen this turn (`outcome.acid_rain_reachable`:
  BtlCmd_TryPursuit runs only from a switch, subscript_switch_pokemon.s:11, and U-turn,
  subscript_attack_then_switch_out.s:54, so a battler switches (not in a turn resumed after a party list, whose
  switches have all run and whose pick reaches no TryPursuit) or uses U-turn while one of the other
  side uses Pursuit, battle_script.c:6899-6916; and the mask at :115 can be nonzero: nonzero now, or a
  third battler acts before the chase, never in singles) the Future Sight roll is kept whole (`loose`:
  Acid Rain can turn Gravity or fog on or off) and the end-speed weather points are freed
  (`enumerate`'s `weather_free`: it can set rain and sun bits). (Fix of a0a09a9, which took any
  Pursuit on a nonzero mask, a chase or not: the snap compare's 12 diverged parts.) `outcome._threshold_grid` needs no change: its
  weathers are the start mask through `field.ability_bits` / `move_bits`, exact on any mask, and in singles an
  Acid Rain write ends both actions (the pursuer's and the switcher's), so no accuracy check follows it.
  Coverage had no Acid Rain refusal on this branch (none in src); its docstring describes the mask (FCM 4.5).
  Tests: `tests/engine/test_field_mask.py` (3.1's rows; the player's and the foe's Pursuit KO, a Life Orb
  recoil faint, Destiny Bond against a recoil with a spurious Uproar bit, U-turn's chase under Gravity; the
  first pursuer by index at B = 4; the Future Sight law on 0x8044). tests/engine/test_d3_pursuit.py does not
  exist on this branch.
- G-13 Manager ruling 7: E2's interim Acid Rain guard (SF2 (f) at B = 2, D3-9 at B = 4; the `acid_rain` field,
  its refusal and its `fullset` test) is not on main, so batch G touches no guard. When E2 merges main after
  G-12 lands, E2 drops the guard (the field, the refusal line, the `exp_refused`-style consumers, the backlog
  kind and tests/engine/test_acid_rain_guard.py) and its B = 4 Pursuit site calls `step.first_pursuer` and
  `step._acid_rain` instead (G-12).
  Done in batch G's merge of main 08a5700 (SF2): the B = 2 guard is gone (`BattleState.acid_rain`,
  `RefState.acid_rain`, `coverage.acid_rain_reason` / `check_acid_rain` and their host entry calls, the
  search's `acid_rain != 0` refusals, `CoverageReport.runtime` with its CLI print and the backlog's
  "Playable, lines hitting the glitch refused" section, tests/engine/test_acid_rain_guard.py and the two
  runtime-line tests); SF2's Pursuit sites call `step._acid_rain` (engine) and apply `_pursuit`'s
  `(faintedMon, kept)` with `_acid_rain` (reference: `_pursuit` returns `(pursuer, took, acid)`, the
  pursuer from `_pursuer_ready` in index order). The guard cases test_field_mask.py lacked are model tests
  there now (Aftermath recoil, the player's Life Orb pursuer, Destiny Bond before Rough Skin). D3-9's
  B = 4 guard is not on main yet: E2 drops it as above.
- G-14 Air Lock (ability 76) counts exactly as Cloud Nine: `AbilityId.AIR_LOCK` = 96 (`ABILITY_DECOMP_IDS` 76,
  `tools/effect_map.py` name map, `enumerate.INPUT_NEUTRAL_ABILITIES` beside Cloud Nine, the
  `ids.ABILITY_DECOMP_IDS` table in `data/game_constants.py`). The rule was already modelled: every decomp read of the
  ability pairs it with Cloud Nine (`ABILITY_AIR_LOCK`: include/battle/common.h:64, :67, battle_controller.c:2073,
  battle_script.c:9201; no other C, script or AI-script read), and both engines read the lock as a count of living
  battlers with decomp ability 13 or 76 (`damage.weather_suppressed`, the reference `_weather_suppressed`,
  `ids.WEATHER_ABILITY_IDS`), so the missing `AbilityId` was the only gap (coverage refused it,
  `IMPLEMENTED_ABILITIES`). No hook, reveal or switch-in message (battle_lib.c:3649-4170 has no case). The weather
  countdown, the weather moves' fail tests, the weather abilities' PERM test and the AI's `LoadCurrentWeather` /
  `IfFieldConditionsMask` stay ungated (the mask read raw, G-6). MANIFEST_VERSION 36 (table inputs changed: the
  ability id numbering). Backlog: the Roark Split's "ability: Air Lock" row is gone (regenerated). Trainer 745's
  Rayquaza still waits on its other groups. Tests: `tests/engine/test_e1_batch_g.py` (rain / sun damage equals Cloud
  Nine's and the no-weather value; no sand chip).
- G-15 MapHeader_GetMapType (rom_diff.md:277; sheet rom-map-type): `modelled`, its `when` dropped. The patched getter
  changes no fight: every header a fight reaches keeps vanilla's mapType (d_test.narc u16 at 22, bits 0-6: 363
  ROUTE_210_NORTH outdoors, 449 CELESTIC_TOWN_CAVE indoors, 247 VICTORY_ROAD_1F_ROOM_2 cave; 154-156, 284 the same).
  The field rule it feeds is vanilla: a warp into a non-cave map clears FLAG_FLASH_ACTIVE / FLAG_DEFOG_ACTIVE
  (field_map_change_flags.c:80-85; a walking zone change keeps them, :36-53); every map change sets the overworld
  weather to the header's, clear for 14 with the Defog flag or 16 with the Flash flag (field_map_change.c:273-280);
  field Defog sets the flag and the weather clear at once (scripts_field_moves.s:385-387, scrcmd.c:3667-3674),
  offered only under weather 14 with the Relic Badge (field_move_tasks.c:285-288, :504); the battle starts in the
  overworld weather (field_battle_data_transfer.c:267). Weather 16 has no battle effect (the field-weather switch's
  default, battle_lib.c:3700; Kaizo keeps it, rom_diff.md:368): the 24 dark-flash rows (Sunyshore Gym, Wayward Cave)
  leave the refusal on that source (manager ruling 4). Weather 14: `AreaCondition.fog_headers` (sorted header ids;
  `fog_clearable`) lists a fog fight's headers of weather 14 (`OVERWORLD_WEATHER_FOG`, data/trainer.py); Kaizo:
  `tools/kaizo/resolve.py` `fog_headers` over `rom_trainer_maps.json` and `rom_map_weather.json` (`Sources.
  map_weather`), into `area_condition`; vanilla: `tools/extract_decomp.py` writes `fog_headers` beside the start
  condition (header ids from `tools.terrain.header_names`). Kaizo: Byron Split's Route 210 North rows (66-69, 71-76,
  78, 79, 81, 441, 442, 489, 490), 913 Cyrus #1 (449) and the unplaced-split 636, 637, 659, 660, 748, 774, 776;
  Wake 695 / 696 (start none) and Candice 610 (rain, overrides) are on 363 / 247 without a fog start; deep fog (15:
  Hearthome Gym, Coronet) is never cleared (fmc:275 tests 14 only). MANIFEST_VERSION 37 (table inputs changed:
  trainers.json). Backlog: the GetMapType group is gone (Byron Split ready rows 24 -> 34; regenerated).
- G-16 The fog start is a run input (manager ruling 4, the Route 213 input kind: default from the map, fog).
  `coverage.start_weather(trainer, field_defog)` gives no weather for a `fog_clearable` fight and refuses any other.
  `cli.fight_start` (called by `_fight` for coverage / rank / solve, `_prep`, `line_setup` for both line commands):
  a save standing on one of the fight's fog headers gives the start from its overworld weather (`SaveFile.map_header`
  = the player Location's mapHeaderID, `overworld_weather` = FieldOverworldState.weather, `saves.layout.MAP_HEADER`
  0x1280 / `OVERWORLD_WEATHER` 0x12E6, field_overworld_state.h:13-20, location.h:18-24; Continue keeps it,
  field_map_change.c:521-525): 0 clear, 14 fog, any other value refused; a `--field-defog` the save's fog (14)
  contradicts is refused, one rule with `--script-weather` (G-24: the save decides when it shows the state on
  the fight's map; review G 1). The warp acts on both the same way and opposite to the plan: it brings the fog
  back (prep keeps a Defog holder) and ends the script rain for good (prep plans clear behind a warp step;
  G-24, gate review G MUST 1). In prep (no team yet) a save's clear on the header keeps a Defog holder
  (`field_move` = Defog): any PC visit is a warp, which clears FLAG_DEFOG_ACTIVE (field_map_change_flags.c:80-85)
  and rewrites the weather from the header, fog again (field_map_change.c:273-280), so Defog is used again
  before the fight (review G 2). Otherwise fog unless `--field-defog`,
  which needs the save's Relic Badge (bit 4, generated/badges.txt:5, trainer_info.c:107-112) and a team member that
  knows Defog (a fixed team: checked; prep: `PrepProblem.field_move` = Defog). FLAG_DEFOG_ACTIVE itself is not read:
  on the header the weather already carries it (fmc:275-280), and off it the way in may warp (fmcf:80-85). Prep: the
  user's flag is the confirmation the sheet asks for, so prep runs the chosen start only (not both); the party at the
  fight is the party that used Defog (no PC after the last warp), so `setups.knows_field_move` keeps only team setups
  whose members still know it after their changes (`team_setups`, `neighbours`' `add`, `exhaustive.restricted_setups`
  / `pruned_setups`); `build_problem` refuses a pool with no Defog knower (the team phase starts from the candidates'
  own moves: an HM05 Defog taught in prep counts only beside a base knower); `report.prep_steps` adds the field step;
  the checkpoint digest carries `field_move` when set. Engine and AI unchanged: the start is `ids.start_field_mask`
  (fog bit 15 or none, FCM.md 4.3), read by accuracy, Weather Ball, SolarBeam, recovery and the AI through the mask
  (G-6); doubles 68, 75, 79, 441, 442 share the one `_FIELD` mask. Tests: tests/tools/rom/test_kaizo_rom.py
  (headers), tests/kaizo/test_resolve.py and tests/data/test_trainer.py (`fog_headers`), tests/saves/test_save.py
  (layout), tests/prep/test_setups.py (Defog teams, start mask), tests/engine/test_e1_batch_g.py (`fight_start`).
- G-17 Rulings by group: ruling 4 (overworld Defog before a fog fight; the 24 weather-16 rows) is applied here
  (G-15, G-16). Ruling 3 (Route 213 script rain as a start input) and ruling 5 (835 / 836 start in rain, scr_seq
  329 + PKCalc) are the rom-field-script-commands group's (gFieldScriptCommands, sheet steps 1-2); its Route 213
  check of a save on header 373 can reuse `SaveFile.map_header` / `overworld_weather`. Ruling 1 (per-fight terrain)
  is terrain-per-map's.
- G-18 Battle terrain per fight (sheet terrain-per-map; manager ruling 1): a run input, refused until set. A trainer
  battle's terrain is the tile under the player at the start: Encounter_NewVsTrainer -> FieldBattleDTO_Init
  (encounter.c:744) -> SetBackgroundAndTerrain (field_battle_data_transfer.c:496-505: background from the map header,
  BACKGROUND_WATER while surfing) -> CalcTerrain (:467-494: the tile behaviour at location x, z,
  terrain_collision_manager.c:229-240; else the background's row, :441-465), copied to BattleSystem.terrain
  (battle_main.c:1052) and never written in battle. `Trainer.terrains` (tools/terrain.py trainer_terrains, the
  start tiles) with one value is `Trainer.terrain`: no input. With several the tile is the run's route; sources
  checked: ROM several, calc repo none (pkcalc raw/MOVES.json Nature Power / Secret Power texts only), docs sheet
  none (docs.json has no terrain field). The row records them: Kaizo `sources.terrain` of each such trainer (tools/
  kaizo/resolve.py: "several: pkcalc none (move texts only), sheet none (no terrain field): a run input
  (--terrain)"); the coverage line (`coverage.terrain_refusal`) names the start-tile terrains and the three sources
  and asks for `--terrain`. Input: `--terrain NAME` beside `--field-defog` (`cli.run_inputs`) on coverage, rank,
  solve, prep and both line commands; `cli.fight_terrain(trainer, terrain)` (called by `_fight`, `_prep`,
  `line_setup`) returns the trainer with `terrain` = the named value, which must be one of its known `terrains`
  (`terrain_id` reads `TERRAIN_SNOW`, `snow`, `great marsh`); any other value, and any value for a trainer no map
  places, is a `ValueError`. Every reader downstream already takes `trainer.terrain`: coverage, `make_state`
  (`_FIELD` int8, one per battle, doubles convention 2), prep (`PrepProblem.trainer`). Prep: `report.prep_steps`
  adds "Start the fight standing on a tile whose battle terrain is X"; the checkpoint digest carries the terrain
  when the run named one (`data.trainer.run_terrain`), so a checkpoint of another terrain does not resume. Kaizo:
  160 split rows have several start-tile terrains (none unread, none unplaced); vanilla none. MANIFEST_VERSION 38
  (table inputs changed: trainers.json sources). Backlog regenerated: the 9 battle-terrain groups stay as run-input
  rows with the new text. Tests: tests/engine/test_e1_batch_g.py (Laura 134 GRASS / SNOW: refusal, input, Nature
  Power calls Blizzard / Seed Bomb, bad values, a single-terrain row), tests/prep/test_setups.py (step, digest),
  tests/engine/test_coverage.py (line text).
- G-19 Not built here, by ruling or by group: the sheet's per-terrain prep axis (evaluate every terrain, plan on the
  worst) is replaced by ruling 1's refusal until set. A save does not give the terrain: its Location (header, x, z)
  would, but CalcTerrain needs the ROM land data at run time, which no table carries; the user names the terrain
  instead. `order.terrain_known` and the reference `_terrain_row` None paths stay: a mixed fight without a terrain
  move still builds with TERRAIN_UNKNOWN (nothing reads it); dropping them is terrain-moves.md "Missing 1" (the
  next group), which reads the state's `terrain` as set here.
- G-20 Camouflage's `MOVE_EFFECT_CAMOUFLAGE` (moves.h:36, bit 28) is a battler field, `camouflage` (int8 per
  battler, doubles convention 2; the decomp's `BattleMon.moveEffectsMask`, battle_mon.h:127), replacing
  `effects.camouflaged`, which inferred it from types != personal row. Set only on success:
  effect_script_0213.s:5-6 (`TryCamouflage _016` jumps to :16, MOVE_STATUS_FAILED, past the
  `UpdateMonData ... MOVE_EFFECT_CAMOUFLAGE` on Multitype or an existing type, battle_script.c:7997-8022).
  Cleared with the mask on a switch (battle_lib.c:2071) and a faint (:2166); Baton Pass keeps only
  MOVE_EFFECT_BATON_PASSED (:2076, moves.h:46-48, which omits it): `VOLATILE_FIELDS`, not
  `BATON_PASSED_FIELDS`; reference `_reset_battler` likewise. Transform does not copy it (BtlCmd_Transform
  copies the BattleMon bytes up to `ability`, battle_script.c:5346-5348; the mask lies after), so a
  Transformed copy of a Camouflaged foe has its types and no bit: the old inference read such a battler as
  camouflaged. Readers: Roar's Ingrain test (`effects.ingrain_anchors` / reference `_ingrain_anchors`, the
  bit joins `MOVE_EFFECT_MASK_FIELDS`, subscript_force_target_to_switch_or_flee.s:7) and the AI. AI (the
  approved list's "#213", backlog row "AI move effect #213 CAMOUFLAGE not transcribed"): Basic_ScoreMoveEffect
  dispatches BATTLE_EFFECT_CAMOUFLAGE at script.s:252 to Basic_CheckCamouflage (:1021-1024): `IfMoveEffect
  AI_BATTLER_ATTACKER, MOVE_EFFECT_CAMOUFLAGE` (trainer_ai.c:704-714) -> -10. `MoveEffectMask.CAMOUFLAGE`
  (ai/commands.py) reads `camouflage`; `policy._basic_check_camouflage` at :252 in the Basic table; 213 joins
  `outcome.KAIZO_TRANSCRIBED` (no Expert_Main entry, :1623-1811; its other mentions are tables the generic
  code already reads: Encore :3264, SetupFirstTurn :6477 / :6490, Harrassment :8065). Kaizo: overlay14
  changes only script.s:78 (rom_diff.md:359). Observation: unchanged (the types it shows are the written
  ones; no message-only state). Random states draw the bit (tests/strategies.py `_draw_batch12`).
- G-21 The `terrain_known` guards are gone (terrain-moves.md "Missing 1"): `order.called_move`,
  `order.move_effects` (its PLAIN fallback) and `effects._camouflage` read `TERRAIN_MOVES` /
  `SECRET_POWER_EFFECTS` / `TERRAIN_TYPES` at `order.terrain_index` = min(terrain, TERRAIN_SPECIAL)
  (battle_script.c:8007-8010, :8040-8043, :8067-8070); the reference `_terrain_row` returns a row, never
  None, and both bound -1 at row 0 only to keep the index in the tables. What stays refused, and why:
  `coverage.terrain_refusal` for a trainer whose terrain is `TERRAIN_UNKNOWN` (mixed start tiles the run
  has not named, or no map places it) when a terrain move is in the fight. That is manager ruling 1
  ("refused until set", a user input like the gauntlet rules), not a coverage gap, so it is kept;
  `tools/trace_to_state.py` keeps its own TraceError for the same case. Dropped: the branch that refused
  Nature Power on a named terrain whose call has an unsupported effect: all 13 calls (Earthquake, Seed
  Bomb, Rock Slide, Blizzard, Hydro Pump, Ice Beam, Tri Attack, Mud Bomb, Air Slash) are modelled in both
  games (tests/engine/test_terrain_moves.py). Random states draw only named terrains when a reader is
  known (tests/strategies.py `TERRAINS`).
- G-22 Secret Power's gates (terrain-moves.md "Missing 4"), confirmed, no change: effect_script_0197.s:7
  sets `sideEffectIndirectFlags = sTerrainSideEffect[row]` with TO_DEFENDER only (battle_script.c:
  8063-8075), the plain-flags branch of TriggerSecondaryEffect (battle_lib.c:1570-1587: chance x2 for
  Serene Grace, roll, then HP, no substitute hit, no NO_EFFECTS), which is the branch of Rock Slide's,
  Flamethrower's and Crunch's own scripts (effect_script_0031.s, 0004.s, 0069.s: TO_DEFENDER | PTR). So
  the swapped-in effect (`order.move_effects`) runs through the same handlers as those moves' table
  effects (SECONDARY_STATUS, FLINCH, SECONDARY_STAT_FOE: the substitute gate, the target-HP gate) with
  Secret Power's own chance (`move_effect_chance`, 30). Tests: SNOW freeze, none through a substitute;
  BRIDGE evasion -1 (engine == reference).

- G-23 Kaizo script weather before a battle (rom_diff.md:292 gFieldScriptCommands; sheet
  rom-field-script-commands; manager ruling 5). Command 0x9C (scrcmd.h:164, ScrCmd_Unused_09C) runs the
  injected code at 0x023C8345: VAR_0x4015 != 0 -> FieldOverworldState_SetWeather(VAR_0x4016); the battle
  starts in the overworld weather (field_battle_data_transfer.c:267; Kaizo switch, battle_lib.c:3665-3669:
  4 THUNDERSTORM -> rain); ScrCmd_0C3 / 0C4 set clear (scrcmd.c:3657-3674), a warp rewrites it
  (field_map_change.c:272-280). `maps._walk` carries the weather command a path last ran
  (`ScriptWeatherWalk`; resets: 0C3, 0C4, Warp) and `maps.script_weathers` records the opponents of a
  battle reached after it: scr_seq 329 (header 293) 0x01df -> StartTagBattle 609, 835, 836 at 0x0202, so
  835 Triton #2 and 836 Neptune #2 start in rain (the partner 609 is not an opponent and keeps none).
  `trainer_start_conditions` replaces the header's condition for that placement;
  `rom/rom_script_weather.json` ("battles") and the trainer's sources cite it. PKCalc's rain list agrees, so
  the resolutions.toml ROM-vs-PKCalc row for 835/836 and its "User calls" line are gone. scr_seq 182 sets 18
  (Trick Room) before 666 (0x00d5 -> 0x0101, 0x01a3 -> 0x01cf): field effects are left to overrides
  area.666 (sheet Elite Four Split:206), which now cites the script; 666 stays refused as every Trick Room
  fight. scr_seq 1056 sets 16 (no battle effect, battle_lib.c:3700) and starts no battle; 1117 runs only the
  wild branch. No other Kaizo trainer's ROM start condition changes (835, 836 only).
- G-24 Route 213 script rain is a start-condition input (manager ruling 3). scr_seq 459 entry 12 runs from
  coord event VAR 0x40FA == 7 (vanilla VAR_UNUSED_0x40FA; zone_event 359 coord #3; trigger test
  src/unk_0203C954.c:218), sets overworld weather 4 at 0x0264, then VAR 0x40FA := 8 (0x0283) and ends; the
  rain lasts until 0C4 after the Chelle win (0x02b7, 0x02ff) or a map change. `script_weathers` gives the
  window to the header's object trainers (103-108, 110, 111, 696) and to the battles of entries an object
  runs or a coord event at VAR 0x40FA >= 7 (491 Chelle, entries 13 and 14); the coord battles at 3, 4, 5
  (692-695, 466; entries 9-11) run before it and get none. Data: `AreaCondition.script_weather` ("rain")
  and `script_window` (373, 4, 0x40FA, 8). The start: `cli.script_weather_state` -- the run input
  `--script-weather` / `--no-script-weather` (coverage, rank, solve, prep, both line commands, beside
  `--field-defog` and `--terrain`, `run_inputs`); a save with VAR 0x40FA >= 8 decides (off header 373:
  none, the map change cleared it and nothing sets it again; on it: its overworld weather, 4 rain, 0
  clear; any other refused), and an input it contradicts is refused; before that the route decides, so the
  default is the start condition (the user's rain, overrides area.103-111/491, 2026-11-07; none for 104
  and 696). In prep (no team yet, `warp`) a save on header 373 in the rain gives False (gate review G MUST 1;
  amends review G 2's "the input decides", which planned every changed team in rain): a team change is a PC
  visit, a warp that rewrites the weather from the header, clear (field_map_change.c:273-280;
  rom_map_weather.json[373] = 0), and VAR 0x40FA >= 8 keeps scr_seq 459 entry 12 from running again, so the
  rain never returns. The save's own party (no PC) would keep it; rather than a second start weather per
  setup (prep has one `start_condition`), prep lists a warp onto the map before the fight
  (`cli.script_rain_on_save` -> `PrepProblem.script_weather_warp` -> `report.prep_steps`), so every setup
  starts clear as planned; `--script-weather` there is refused ("prep plans a warp onto the map first"). A
  fixed team (coverage, rank, solve, line) keeps the save's rain. A save's False binds in prep too.
  `SaveFile.vars` /
  `var` read VarsFlags.vars (vars_flags.c:65-69, NUM_VARS 288).
  `coverage.start_weather(script_weather=)` turns the state into the weather.
- G-25 Status 0x3F (manager ruling 2): a real battle of a split can start under Newmoon's 0x7B. scr_seq 363
  entry 4 (header 321, an init-script entry) writes party[0] := 0x7B (sleep 3 | poison | burn | freeze |
  paralysis, condition.h:7-13) when it has no status, then StartFirstBattle 674 (Lady Emi, no split) and
  warps to header 415. Header 321 is reached mid-story: Kaizo's Pokemon Center 2F bed script (scr_seq 213
  entry 20, common script 9019) runs once trainer flag 882 is set (scr_seq 278 0x011d, the Floaroma Meadow
  house), and with trainer flag 823 set always (0x0199 -> 0x0b18: GetRandom 100 <= 100, GoToIf 3 is
  "<=", scrcmd.c:744-752, :1069 -> 0x09d6), else on some maps by a GetRandom roll, `Warp 320` (0x09dc); scr_seq 362 0x00f8 warps 320 -> 321. (Vanilla's way in needs
  FLAG_GAME_COMPLETED, scripts_canalave_city_harbor_inn.s:16-17, kept by Kaizo's scr_seq 47.) The battle
  keeps every bit (wake: `status &= ~MON_CONDITION_SLEEP`, battle_controller_player.c:2449; thaw clears
  only freeze, subscript_thaw_out.s; each reader tests its own bit, e.g. battle_lib.c:4479), so the word
  reaches the next fight unless healed. NOT modelled here: the engine's status is one condition plus a
  counter (`ids.Status`, STATE.md "Status"; saves/status.py decode_status kept only the first), and the full
  word is a state change through every status reader (engine, reference, AI). What is done: a party
  Pokemon whose word holds several conditions is refused at the save (`SaveMon.to_spec` RecordError,
  `status.status_conditions` names them) instead of being silently cut to one; prep heals (`level_to_cap`,
  `healed=True`). The full-word model is left for its own batch. Floaroma's (scr_seq 278) and the beds'
  0x3F write 1 (sleep 1), one condition, read from the save as any status.
- G-26 gFieldScriptCommands in rom_diff.toml / rom_diff.md: the weather by G-23 / G-24 and the status and exp
  (0x1CE) through the save's party are modelled; the multi-condition word is the `not_reached` remainder with
  `when = { map_header = [321] }` (G-39; amended from `modelled` by review G 6).
- G-27 Every Starf Berry eat on one battler in one turn draws its own (F-O3, F-10's Recycle repeat): each
  site runs its own `RandNext() % 5` loop: `BattleSystem_TriggerHeldItem` (battle_lib.c:4576, Starf
  :4850-4872, draw :4865), `BattleSystem_TriggerHeldItemOnStatus` (:4967, Starf :5273-5294, draw :5288; the
  F-O3 / F-10 text names it `TriggerHeldItem`: same code, its own function and draw), `BattleSystem_PluckBerry`
  on the attacker (:5798-5817, draw :5808). `Outcome.starf` is `int8[B, 4]`: `[b, 0]` the battler's first eat,
  `[b, 1]` the faint replacement's (unchanged), `[b, 2]` its second, `[b, 3]` its third. State: `starf_roll`
  `int8[B, 3]` is the turn's queue (`Outcome.starf[:, ids.STARF_QUEUE]`, columns 0, 2, 3) and `starf_taken`
  `int8[B]` counts the draws its eats made (STATE.md); `hooks.starf_draw` reads entry `min(starf_taken, 2)`,
  `hooks._pinch_effects` (every held-item check and Pluck) adds one where a draw is made (a stat below +6),
  an open battle-start berry takes entry 0 (`step._action_phase`: `starf_taken` 1). `step_events` and
  `step.replace` leave `starf_taken` 0 and a kept replacement draw in entry 0, so a second replacement
  re-reads it as before. The turn-start check (`_turn_start_items`) reads the queue, not the stale first
  draw. Reference: `_pinch_effect`, `ref_step`, `ref_replace` the same.
- G-28 Three draws without Fling (amends the sheet's bound "without Fling ... two"): battler b's Pluck / Bug
  Bite of a foe's Starf (draw :5808; its `subscript_held_item_sharply_raise_stat.s:21` PLUCK_CHECK clears
  SELF_TURN_FLAG_PLUCK_BERRY), its own Starf at AFTER_MOVE_EFFECT_ATTACKER_ITEM
  (battle_controller_player.c:3684-3689), then a slower foe's Roar (priority -6) drags in a bench holder at
  its pinch, whose switch-in check eats (lib:4151-4165 -> :5288). And one berry eaten twice: a Pluck whose
  berry did nothing leaves the flag set (lib:5848, `subscript_pluck_check.s:5-9` skips RemoveItem), so the
  attacker's own Starf is kept at :3684 and eaten again at _HELD_ITEM_STATUS (:3730-3761), two draws
  (`step` `info.pluck_flag`). A fourth draw needs a foe's Fling of a Starf (lib:6110-6127, draw :6121) on
  top; Fling (effect 233), Thief / Covet (105) and Trick / Switcheroo (177) are unsupported
  (`coverage.ITEM_MOVING_EFFECTS`), so an eat past the third reads the queue's last entry; Fling's builder
  adds its column. A U-turn / Baton Pass newcomer is no same-call source: the turn pauses at the party
  pick (`pivot_battler`, subscript_attack_then_switch_out.s:71, subscript_baton_pass.s:17) and the resumed
  call loads its own queue, so its eat takes the resumed outcome's `[b, 0]`.
  With Fling (gate review G MUST 2; E3's e5490ccb, E3-B2-1, E3-B2-22: a flung Starf takes the defender's next
  entry): three draws per battler per call still bound singles. Every extra source is one action or one item:
  b's action is Pluck / Bug Bite, Recycle or neither; the foe's is Fling (TryFling draws at :6121 before
  TryMove, effect_script_0233.s:7, :18, so a miss, Protect or Substitute still takes the entry) or Roar; Fling
  and Pluck share the foe's one item (Fling removes it; a Pluck of a foe without a berry sets no flag,
  lib:5832-5835, :5848). A held berry is eaten once, twice only under the acting attacker's set
  SELF_TURN_FLAG_PLUCK_BERRY (subscript_pluck_check.s:5-9), and flags last one action (BattleContext_Init at
  MoveEnd, bcp.c:3998; lib:2013-2015). Fling's own flag (lib:6150-6152) stays set only when
  subscript_fling.s:5-7 skips (b at 0 HP or behind a Substitute): b then loses no HP and no stage in that
  action, and any earlier pinch was eaten at the check after it (bcp.c:3684-3761, :1362-1368, :1699-1708; the
  turn-start switch-in check, lib:4151-4165), so no own Starf of b is eaten under it. Singles worst cases, three
  each: own + flung + Recycled own (an open battle-start berry may stand for the own eat); Pluck draw + own +
  Roar newcomer; own twice under b's Pluck flag + Roar newcomer. A fourth needs two foe actions or items:
  doubles only (G-40).
- G-29 The laws (`enumerate._starf_reads`): `[b, 0]` and `[b, 1]` unchanged. Battler b's eat sources this
  call are counted: its own (the field holder at its pinch or able to lose HP; once more with a Pluck /
  Bug Bite of a foe item (G-28); a Recycle of a Starf, recycled before or eaten first; a Pluck / Bug Bite of
  a foe holder; an open battle-start berry) plus a Roar newcomer holder. `[b, k]` (k = 2, 3) is read where
  the count reaches k. Law: the k-th eat is the same battler's while it has k own sources: the `[b, 0]` rule
  with +2 per earlier eat (Recycle stays a stage jump: all 60); else the Roar newcomer's on fresh stages
  (`InitBattleMon`, battle_lib.c:97-101; a dragged battler does not act): five values unless the foe's
  bounded gains + 2 reach +6 or a stage jump is used. Brute force (`loose`): `[b, 2]`, `[b, 3]` all 60 where
  a holder or berry source meets a second source of any kind (a drag, Recycle, Pluck / Bug Bite, open berry).
  `starf_loose` is `bool[B, 4]`. Doubles: partners' actions add sources (multi-battler milestone).
- G-30 No coverage refusal held this case (F-O3 was a known limit, none to remove); no AI read (the AI reads
  the stages next turn; Recycle / Natural Gift tables already list Starf, F-10); no table input changed (no
  MANIFEST bump). Tests: tests/engine/test_e1_batch_g.py (engine equals reference per case, the laws).
- G-31 Nightmare (F-O5; manager ruling 6: kept, vanilla coverage). Effect 107 BATTLE_EFFECT_STATUS_NIGHTMARE
  (generated/move_battle_effects.txt:108; res/moves/nightmare/data.json: Ghost, status, power 0, accuracy 100,
  CAN_PROTECT) maps to `EffectId.NIGHTMARE` (tools/effect_map.py; MANIFEST 40). No Kaizo move has effect 107
  (Kaizo move 171 is Hidden Power Ghost, effect 0), so it is vanilla only; Snore (effect 92, whose AI label is
  named Expert_Nightmare) is unchanged. effect_script_0107.s fails (MOVE_STATUS_FAILED, :11-12) on a defender
  behind a Substitute (:5), with the bit (:6) or not asleep (:7): `effects.nightmare_fails(state,
  target_side)`, read with the script beside Dream Eater's in `step._use_move` and the reference's
  `_script_fails(..., target)`; a landed move runs subscript_nightmare_start.s (re-tests :5-7, sets the bit
  :9): `effects._nightmare`, the reference's `_move_effect`. Power 0: no type chart (bcp.c:2350-2357), a
  Normal type is hit. New state `nightmare: int8[B]` (statusVolatile bit 27, condition.h:59): in
  `VOLATILE_FIELDS`, not in `BATON_PASSED_FIELDS` (condition.h:72-76; battle_lib.c:2070, :2075, faint :2165),
  not copied by Transform (statusVolatile, battle_mon.h:115, lies past `ability`, battle_script.c:5346-5348);
  STATE.md row. Doubles: the defender is passed in (`target_side` / `turn.target_side`), never `1 - side`.
  Coverage: the effect is EXACT once mapped (coverage.py:203); no backlog row (no Kaizo move).
- G-32 What clears the bit (every decomp writer of VOLATILE_CONDITION_NIGHTMARE, by grep; `effects.clear_nightmare`
  and the reference's `nightmare = 0`): every wake-up, subscript_wake_up.s:20 (the counter at the move,
  bcp.c:2443-2470; an uproar at the move, :2432-2439, and at the turn end, :1505-1516); Chesto / Lum through
  subscript_held_item_slp_restore.s:13 (a sleep cure: battle_lib.c:4608-4611, :4667-4671, Pluck :5577-5580,
  :5648-5651) and _multi_restore.s:13 (Lum on a status and confusion, :4689-4692, :5669-5672); Lum's other
  restore subscripts (prz/psn/brn/frz/cnf) keep it (`hooks._restore`, reference `_cure`); Shed Skin
  (battle_lib.c:3567-3586) and Hydration in rain (battle_script.c:5626-5639 -> subscript_weather_continues.s:54)
  through subscript_ability_restore_status.s:6, with any status (closes F-O5); Insomnia / Vital Spirit
  (RecoverStatusByAbility, battle_lib.c:4404-4408, at the switch-in :4139 and the move end bcp.c:3965-3967)
  through subscript_ability_forbids_status.s:16-18 (msgTemp MSGCOND_SLEEP = 0 only; Limber's branch keeps it);
  Heal Bell / Aromatherapy (BtlCmd_TryPartyStatusRefresh, battle_script.c:5039-5070: the user unless its
  Soundproof spares it, and the doubles partner): each battler of the user's party whose active slot the
  refresh cures (the partner's battlersSwitchingMask test, :5049, :5068, is a doubles case for the multi-battler
  milestone); a bag sleep cure on the active battler (battle_system.c:459-475, `bag.use_bag_item`). Full
  Restore's subscript_use_full_restore.s clears the status first and copies it to the party
  (battle_script.c:3077-3078), so its UseBagItem finds no sleep and keeps the bit (the engines already read
  the status after it). Not cleared: Wake-Up Slap (subscript_heal_target_sleep.s:6), Psycho Shift
  (subscript_give_target_own_status.s:30), Refresh, Natural Cure (the switch clears it anyway); the bit then
  waits for the turn end (G-33), or costs a quarter if the battler is asleep again by then (a Rest after a
  Wake-Up Slap). Fling's Chesto / Lum (battle_lib.c:5886, :5948, :5969) has no engine path: Fling is refused
  (`coverage.ITEM_MOVING_EFFECTS`).
- G-33 The turn end: MON_COND_CHECK_STATE_NIGHTMARE (bcp.c:1436-1448) runs after BURN and before CURSE
  (enum bcp.c:1256-1283): a battler with the bit and curHP that is asleep runs subscript_nightmare_effect.s
  (Magic Guard: CheckAbility :5 skips it; else Divide(maxHP, 4), :8-10, battle_lib.c:3599-3618, floor 1;
  UPDATE_HP :16, it may faint); not asleep, the bit is cleared with no message (:1444). JAX
  `step._nightmare_check` inside `_burn_then_curse` (one faint check still serves the three: each needs
  curHP); reference `_nightmare_check` in `_end_of_turn`'s list. The loss is before ITEM_CONDITION (:1699),
  so a pinch berry can eat after it, and after Shed Skin (:1355) and USE_ITEM (:1362), so those cures come
  first. `enumerate._starf_reads` counts the bit as a residual HP source. Observation: `nightmare` beside
  `torment` (observe.py). Tests: tests/engine/test_e1_batch_g.py (engine equals reference per case).
- G-34 AI: Basic_CheckNightmare (script.s:194 -> :321-327) is `policy._basic_check_nightmare`: the defender's
  VOLATILE_CONDITION_NIGHTMARE -10 (`commands.VolatileCondition.NIGHTMARE`, read from `nightmare`), not asleep
  -8, a loaded Magic Guard (LoadBattlerAbility: known or guessed, one read) -10. 107 joins
  `outcome.KAIZO_TRANSCRIBED` (no Expert_Main entry: Expert_StatusSleep's IfMoveEffectKnown :1817 and
  Expert_Encore's table :3227 name it for other moves, both already transcribed), so
  `opponent_move_refusals` no longer refuses it; `read_bounds.py` regenerated (tools/ai_read_bounds.py) with
  its rows; the run also adds the rows of 213 (Camouflage), which G-20 put in `KAIZO_TRANSCRIBED` without
  regenerating (tests/ai/test_read_bounds.py pins the key set to it). Test: tests/ai/test_kaizo_policy.py
  test_nightmare_basic (a synthetic effect-107 move).
- G-35 Backlog row (manager order 19 Nov, G-25 context): `tools/kaizo/backlog.py` `SCRIPT_START_CONDITIONS`
  writes "Start conditions set by ROM scripts" into docs/kaizo/mechanics_backlog.md: Newmoon's status 0x7B
  on party[0] is "playable after heal; multi-status start refused" (`SaveMon.to_spec`), citing scr_seq 213
  entry 20, scr_seq 363 0x00f2 and trainer flags 882 + 823. It blocks no split battle (Lady Emi 674 has no
  split), so it counts in no total.
- G-36 `AI_HasAbsorbAbilityInParty` (F-O4) is modelled whole: Water Absorb and Volt Absorb on the AI's bench are
  no longer refused (`switching._AI_SIDE_ABILITIES` keeps Natural Cure only). `checkAbility` comes from `moveHit`'s
  table type (`MOVE_DATA().type`, trainer_ai.c:3665-3674: Fire -> Flash Fire, Water -> Water Absorb, Electric ->
  Volt Absorb, else `return ABILITY_NONE` = FALSE), after a `moveHit` with power (:3656-3663); the active's own
  `Battler_Ability` (`damage.battler_decomp_ability`, so a suppressed one is none) equal to it holds the switch
  (:3677-3679); each living bench Pokemon whose `MON_DATA_ABILITY` is it draws `RandNext() & 1` (:3703-3708) in
  party order, the first success switching to it (`SwitchSite.ABSORB`). Bench: `aiSlot2` is the battler itself in
  singles (`BattleSystem_GetPartner`, battle_system.c:400-402), so the passes' bench. A Weather Ball `moveHit` is
  read as Normal here (table type), unlike the passes (`TrainerAI_MoveType`, :3767).
- G-37 The gate. With a super-effective move of the active (`AI_HasSuperEffectiveMove(..., TRUE)`, no draw inside,
  :3586-3588) the function first reads `RandNext() % 3` and returns FALSE unless it is 0 (:3651-3653): 21846 of
  65536. Its bench sites are `SwitchSite.ABSORB_GATED` (same `& 1` draw); the gate is no site: `outcome.
  switch_stop_probs` is `g * law(p) + (1 - g) * law(p, ABSORB_GATED at 0)`, exact since the gate only filters
  those sites and every later read is its own draw; `_first_stop` masks them unless the sampler's gate u16 (its
  own key, `fold_in(k[3], sites)`, so the site draws are unchanged) is 0 mod 3. The gate is read before the
  `moveHit` tests, so also when no bench site exists; then it changes nothing and the law is unchanged. The gate
  rides in the site kinds, so every caller of `switch_stop_probs(switch_sites(...))` (outcome enumerator,
  `policy` x3) gets it with no API change (the sheet's `gated` flag / companion function is that bit).
- G-38 Site count: absorb sites now exist beside the keep draws too (`_layout` drops `& ~any(keep)` and the
  `absorb_gate` refusal): at most 5 + 4 + 40 = 49 <= 60; with the Wonder Guard draws (no keep) 20 + 5 + 40 can
  pass 60 and the count refusal stays. Tests: tests/ai/test_switching.py (`_brute_stop_law` over residue classes,
  the sampler, Water / Volt / Ice / mismatched holder, own ability, Weather Ball in rain); the Kaizo loose oracle
  (tests/ai/test_kaizo_switching.py `_loose_reads`) walks the gate with a skip of the bench reads, and its
  hypothesis states now draw absorb holders.
- G-39 Manager ruling 2026-11-19 (supersedes ruling 2 for batch G; review G 5): the Newmoon multi-condition status
  word (G-25) is its own batch after G, family "status word"; batch G keeps the interim named refusal at the save
  (`SaveMon.to_spec`, `status.status_conditions`) and the backlog row (G-35). rom_diff.toml's gFieldScriptCommands
  is `not_reached` with `when = { map_header = [321] }`: file 363 (scr_seq 363 entry 4, an init script of header
  321) writes 0x7B to party[0] after any save and before StartFirstBattle 674, so the fight on that header is
  refused at the build's rom-change condition; a later fight gets the word only through a save, refused there.
  The other 0x3F writes (scr_seq 278 entry 1, common script 9019) are one condition (sleep 1), read from the save
  as any status, so the earlier 20-header `when` is not restored. MANIFEST_VERSION 41. Lady Emi 674's split is
  open (CLAUDE.md: every real battle belongs to a split): the ROM has no split field; PKCalc lists "Lady Emi" in
  `FLAGS_PK.json`'s hidden list beside "Maid Yoki", and `PARTY_ORDER_PK.json` orders her after Aroma Lady Taylor
  (Gardenia Split:79) and before Galactic Mercury #1 (Gardenia Split:91), an order, not a split; no split sheet
  of the docs lists her. The ROM gate: flags 882 and 823 come from scr_seq 278 (header 257, Floaroma Meadow house:
  entry 1 sets 882 at 0x011d, entry 3 StartTrainerBattle 823 at 0x0155, no badge check); the bed script (scr_seq
  213 entry 20) needs 882 (0x0171) and with 823 always warps (0x0199 -> 0x0b18 -> Warp 320 at 0x09dc), without it
  checks badge 2 (0x01a4, Cobble, generated/badges.txt:3). The window this points to (from the meadow on) is the
  manager's call; until then 674 counts in no split total.
- G-40 For E2's doubles work (multi-battler milestone; gate review G MUST 2): the Starf queue's three draws per
  battler per call (`starf_roll` int8[B, 3], `Outcome.starf` int8[B, 4], `ids.STARF_QUEUE`) is a singles bound
  (G-28). With B = 4 a partner's and both foes' actions add sources: own eat + the partner's flung Starf + a foe's
  flung Starf + its Recycle is four, and `hooks.starf_draw`'s `min(starf_taken, 2)` re-reads the last entry. The
  doubles batch widens the queue to its bound (state, Outcome, `STARF_QUEUE`, `enumerate._starf_reads`, the
  reference, tests/strategies.py, a tools/snap_compare.py UPGRADES row).
- G-41 Pursuit targets, doubles-ready (gate review G record a): `step._pursuit_hit`'s Sticky Barb and on-hit berry
  take `switcher` (was `_foe(pursuer)`); the reference's U-turn site takes `_pursuit`'s returned pursuer for
  `turn.pursued` and `_owed_pursuer` (`not bonded`: the U-turn user has no Destiny Bond, bcp.c:2425) in place of
  `foe` / `turn.done[foe]`. Same values in singles (one foe; `_pursuer_ready` needs a living pursuer).
- G-42 Gate review G record b (`upgrade_entries` keyed by kind) is not built here: the snap_compare upgrade
  rewrite on its own branch carries it (manager, 19 Nov).
- G-43 Batch G's interim refusals (gate review G record c): the multi-condition status word at the save
  (`SaveMon.to_spec`, G-25) closes with the status-word batch (G-39; the backlog row of G-35 says so). The 24
  weather-16 rows (Volkner Split 11 on headers 154-156, Galactic Split 13 on 284 / 380) are no interim refusal and
  do not move there: they start `none` (data/kaizo/start_conditions.json; weather 16 has no battle effect,
  battle_lib.c:3700, manager ruling 4) and hold no weather refusal; any refusal of theirs is their own other
  source's.
- G-44 Snapshot sampler drift (pod compare of 56cd9f4 vs main 08a57000): `sample_outcome` drew `starf` as
  `randint(fold_in(key, 48), (B, 2))` (08a57000 outcome.py:2076) and now `(B, 4)` from the same key
  (outcome.py:2139), which changes every entry, columns 0-1 too: all four columns of a fresh draw are sampler drift
  (`snap_compare.SAMPLED`); a saved `in_outcome` keeps columns 0-1 strict. No other draw's key, shape or bounds
  changed in `sample_outcome` (body diff: the starf line only); its state reads moved to `field_mask` (FCM).
- G-45 Camouflage on `TERRAIN_UNKNOWN` (snapshot step / jitstep first diff `.camouflage[0]` 0 -> 1): batch G
  dropped `terrain_known` from `effects._camouflage` (08a57000 effects.py:618-633), so terrain -1 reads row 0
  (`order.terrain_index` clip) and succeeds where main wrote nothing; the reference (`_terrain_row`) does the
  same. In a fight -1 never reaches it: `coverage.terrain_refusal` runs over the known moves plus
  `fight_calls` (Metronome's calls) and the invokers' sources are known moves. The drawn states reach it:
  `tests.strategies._draw_terrain` draws a terrain only when a Pokemon knows a reader, not when a Metronome (or a
  drawn last move a caller copies) can call one (E2 owns the strategies). The game never holds an unknown:
  `CalcTerrain` (field_battle_data_transfer.c:470-504) always gives a terrain and `BattleSystem_GetTerrain`
  (battle_system.c:375-381) maps an out-of-range value to TERRAIN_MAX, read as `min(terrain, TERRAIN_SPECIAL)`
  (battle_script.c:8007-8010). No engine change.
- G-46 `enumerate._starf_reads` counted a plucked Starf Berry and the Pluck flag's second own eat as two
  sources (`twice` & `plucked`): with the attacker's own Starf that read `starf[d, 3]` with no Roar newcomer, a
  third eat G-28 shows unreachable. A plucked Starf draws while a stat of the five is below +6 and its
  `subscript_held_item_sharply_raise_stat.s:21` PLUCK_CHECK clears the flag; with all five at +6 neither berry
  draws (lib:5798-5817, :4850-4872); under Klutz / Embargo (lib:5838-5841) no own eat. `twice` now excludes
  `plucked`. The engine's step and the reference are unchanged (they already draw twice there:
  `test_bug_bite_of_a_starf_berry_then_its_own_then_a_roar_newcomer[Tackle]`).
- G-47 The widened snapshot "rest" states (seed 165 turn 3: 160 -> 16000 rows, 2^2 * 5^2; seeds 22 / 64 / 120: the
  width-cap text; seed 107 turn 3: SamplingOnly where main enumerated) come from batch G's later Starf reads
  (G-29): factor 5 per read eat. Not reproduced locally (RAM gate); G-46's over-read is the only unreachable
  read found in `_starf_reads`. Batch G's other enumerator changes: `weather_free` also on `acid`
  (`outcome.acid_rain_reachable`, G-12, singles only with a nonzero mask), Shaymin and band reads narrowed.
  The pod checks `test_snapshot_states_read_a_third_starf_eat_only_with_a_roar_newcomer`.

## Stream state (restart point, 2026-11-19)

Read this section first after a clear, then `docs/process.md` section 2. Branches, worktrees and
pod state are kept outside the repo (the stream's `_scratch/e1/` notes).

- Batch F (section above) holds the Kaizo abilities, held items, Weather Ball's AI type, the Poke Ball
  friendship wrap, Sky Forme's freeze revert, the U-turn Intimidate case and the owed items
  (F-17..F-25). Effect Spore and Suction Cups are dropped (no Kaizo species has them).
- Still open from the owed list: F (Sleep Talk's sealed set after a faster Transform/Mimic) and G
  (the `test_transform_laws.py` additions); see Batch F's open items.
