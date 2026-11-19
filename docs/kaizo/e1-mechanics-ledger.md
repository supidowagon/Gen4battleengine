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
  connected (battle_lib.c:1485-1490). `BtlCmd_TryMimic` (battle_script.c:4476-4523) fails
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
  the defender's BattleMon up to and including `ability` (:5342-5348; battle_mon.h:57-84): species, the
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
  the target holds the side's ability (Sand Veil / Snow Cloak :2927-2933, Tangled Feet :2944-2947, Simple /
  Unaware :2876-2887, No Guard bcp.c:3016-3021, Cloud Nine); the cleared announce flags (:5350-5358) rerun
  its switch-in checks, so a copied weather ability sets its weather (battle_lib.c:3740-3790). Built in
  `_threshold_grid`: the eva copy joins Psych Up's, and the second weather variant is now the foe's field
  move (weather move, Gravity, or Transform with its copied ability and weather); one move has one effect.
  `future_thresholds` keeps the Future Sight roll whole (`loose`) when a Gravity or Transform is used this
  turn or Gravity is at 1 (FIELD_COND_CHECK_STATE_GRAVITY, bcp.c:1227-1235, ends it before the hit
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
  :4476-4523) writes only its user's moves, PP, `mimickedMoveSlot` and `lastResortCount`: input-neutral,
  non-table. Chatter (effect_script_0267.s:5-7: CalcCrit, CalcDamage, then the ON_HIT confusion):
  input-neutral, table damage. `UNSUPPORTED` stays outside both registries (the tests add it). The
  reference engine reads the live state at each check; no change. Tests: `test_transform_laws.py`,
  `test_damage_classes.py` (Chatter's exact case, Transform's compression case).

## Stream state (restart point, 2026-11-18)

Read this section first after a clear, then `docs/process.md` section 2.

### Branches and worktrees
- `claude/e1-test-speed-p1` (worktree `_scratch/e1/sp1wt`): speed pass part 1, test-side only,
  rebased on main 47a62e2. Tip carries this section. Last code commit 7c62f18: slow marks on
  `search/test_learn.py::test_window_c_refuses_finish_turn` (pod 190 s / 5.38 GB) and
  `search/test_stage_b_learn.py::test_an_ai_switch_with_a_needed_replacement_refuses_at_the_learn_stop`
  (172 s / 6.00 GB), both kept `fullset`; `test_sheet_unmapped::test_ally_reasons_name_the_partner_rows_starter`
  skips without `docs/docs.json`. Not run locally (under 6 GB free).
- `claude/e1-test-speed` (worktree `_scratch/e1/speedwt`): WIP measurements only, never merge.
  80cfb4b = change 1 (fixed-size batches for keep and turn_events), f2114f3 = change 2
  (`_reset_battler` side mask). Source for part 2.
- `claude/e1-fullset` (worktree `_scratch/e1/fswt`): merged into main (ad6504b). Remove the worktree.

### Waiting on the manager
- Pod run of the whole fast tier at part 1 (first at 2845c4d; rerun at the new tip). Fix any
  failure, then one fresh review of part 1, then hand the tip over for merge.

### Next steps, in order
1. Part 1 to merge (above). After the merge remove `sp1wt` and `fswt` (unlink the `external`
   and `.cache` junctions with `cmd /c rmdir` before `git worktree remove`).
2. Small branch off main: a Water move into Croagunk's Dry Skin gives 6-8 in the damage table,
   the calc gives 0. Decide from the decomp; fix, or record a cited departure in the
   damage-oracle test's list.
3. Part 2 (changes 1 and 2 from `claude/e1-test-speed`) only after E2's D1 group A merges (it
   rewrites `engine/enumerate.py` keep step and `engine/step.py` `_reset_battler`). Change 1:
   time a real prep search on the pod before and after; several bucket sizes if small turns
   slow by more than 5%. Change 2: `test_differential` with slow tests, own review. Keep E3's
   "announced" flags in `_reset_battler`.
4. Next mechanics batch: send the family list to the manager first. Kaizo abilities (5 groups,
   13 battles; Hydration covers 8), held items (8 groups, 6 battles), Weather Ball AI typeless
   refusal, Poke Ball friendship wrapping past 255, Sky Forme reverting when frozen, Effect Spore
   and Suction Cups (check whether `tools/kaizo/backlog.py` undercounts). Owed items below first.

### Frozen files (E2 D1 group B, until the manager says merged)
`engine/hooks.py`, `engine/bag.py`, `ai/commands.py`, `ai/switching.py`, `ai/policy.py`,
`ai/replacement.py`, `ai/outcome.py`.

### After every merge of main into an E1 branch
Rebuild tables, duplicate scan, the unsupported-asserting tests, `pytest -m "fullset and not slow"`
locally; the pod list carries `-m fullset`. Locally one pytest process per stream, one file or
chunk, only with more than 6 GB free; everything else on the pod (send tip plus file list).

### Owed items (into the next mechanics batch)
- `chatot_cry` is per side; the decomp keeps it per battler in tag/2vs2
  (battle_system.c:915-924). Tell E2 for doubles.
- Confirm the Kaizo ROM leaves `sMovesAffectedByGravity` unpatched (Kaizo move 150 is Hidden
  Power Water); confirm the Kaizo AI scripts match the vanilla script.s lines cited.
- Bag Ether/Elixir on a Transformed/Mimic slot (battle_system.c:636-653).
- `gravity_rearm`: read from ROM sub_seq 238, not the game id.
- Learn masks (levelup.py:462-469, reference/engine.py:2079-2082) and `_learn_refusal` ignore
  battle end: add `& (state.over == 0)` in both engines (learn_pin, step.py:2833-2834).
- Stale Choice lock: the decomp clears `choiceLockedMove` at the next selection
  (battle_lib.c:2307-2310); the engine reads it lazily (step.py:31, 436-456). Model the clear.
- Nits: unreachable D10 fallback in `_learn_refusal`; STATE.md:349-352 "mid-turn learn"; end-of-turn
  Disable/Encore on an unknown move; Spite, Grudge, Last Resort, Mimic `_masked` readers.
- A. `hit_thresholds`: a faster foe's Knock Off removes Wide Lens before the hit
  (battle_script.c:6608-6632); same-turn Knock Off vs Future Sight's items.
- B. `future_thresholds` `free` (outcome.py:522) lacks Haze, Psych Up, Acupressure.
- C. Tests for copied weather, copied No Guard / Unaware, losing the foe's own Cloud Nine /
  No Guard, the Gravity/Transform loose flag.
- D. outcome.py:595-599 Gravity "ends a count of 1" is wrong for Kaizo (`gravity_rearm`); cite
  nits (Sand Veil/Snow Cloak are bcp.c lines; Gravity countdown to :1238; Mimic ends at :4524).
- E. Doubles: `_threshold_grid`'s "one move has one effect" holds only with one foe. Tell E2.
- F. Sleep Talk's call law (enumerate.py:3550-3555; order.py:316) uses the foe's turn-start moves
  while it imprisons; a faster Transform/Mimic changes the sealed set (battle_lib.c:2287-2290).
- G. Tests: `assert_equals_bruteforce` in test_transform_laws.py; copied Cute Charm with the kept
  gender (battle_script.c:5732-5735); copied Wonder Guard; Transform through an invoker.
- H. Cites: copy loop battle_script.c:5346-5348; switch-in re-trigger bcp.c:3672-3673 and 3973.
- I. Effect Spore's contact roll (battle_lib.c:4214-4335) and Suction Cups vs Roar: confirm
  coverage refuses their holders, then build them.
