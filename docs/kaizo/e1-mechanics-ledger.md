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
  the next `BattleContext_Init` before any check: no pass. Not modelled (an engine gap, not a
  refusal): the U-turn user that falls to its foe's Pursuit (the `MODE_PURSUIT` tail) and a
  newcomer that faints to Spikes / Stealth Rock after Toxic Spikes poisoned it: the decomp's
  step still reads the fainted battler's `BattleMon.status` (only the party copy is zeroed,
  battle_controller.c:1481-1486), which the engine's faint check has cleared, so the engine
  passes nothing there (the reference's `_u_turn` clears the flag after the Pursuit to match).
  `step._use_move` (the hit loop's `sync_flag` / `sync_sem` carry from `effects.sync_landed`, the
  step before the switch-in effects for a drag or a tail), `step._act` (`entry_poisoned`); the
  reference's `_status_landed` (in `_inflict`, `_contact_ability`, `_entry_hazards`), the King's
  Rock sites of `_hit`, `_u_turn`, `_resume`, and the calls in `_fight`, `_roar`, `_resume`.
- G-14 Leaf Guard (`AbilityId.LEAF_GUARD`, decomp 102). Every status subscript but
  subscript_freeze.s tests `CheckIgnoreWeather` (a living Cloud Nine / Air Lock holder skips the
  test, battle_script.c:9195-9206), then `FIELD_CONDITION_SUNNY`, then the side-effect battler's
  Leaf Guard, right after the status's own ability test: subscript_fall_asleep.s:23-25,
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
