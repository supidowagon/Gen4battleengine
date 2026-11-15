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
