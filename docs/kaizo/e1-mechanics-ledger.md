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
  `_Hit.effective`) and runs subscript_crash_on_miss.s (:98): a guarded Jump Kick crashes for
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
- G-20 The AI learns the holder's ability: subscript_missed.s:107-110 prints "{0} avoided
  damage by using {1}!" (`TAG_NICKNAME_ABILITY`, BTLSCR_DEFENDER: `BattleMessage_AbilityTag`,
  battle_script.c:11798-11806, `BattleAI_SetAbility`) unless an earlier flag prints first:
  lost focus (:5), protected (:18), failed (:97). A plain miss reaches it (the miss message is
  :157, after it), and an immune Tackle on Shedinja too (WONDER_GUARD is tested before
  INEFFECTIVE, :119). `hooks.DAMAGE_AVOIDED` (`step._use_move`, `step._pursuit_hit`: Pursuit's
  hit on a switching holder goes to subscript_missed.s through subscript_pursuit.s:17, :154);
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
