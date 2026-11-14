# E2 mechanics ledger (Kaizo Roark split)

## RESUME (updated 2026-11-14)

- Closed: batch 1 and batch 2 (reviewed; follow-ups in c61c768 and the merge-check commit),
  committed on this branch. The merged fast tier passed 2715 / 8 skipped (log
  `_scratch/e2/b2/fast_tier_merge_377037d.log` on G:).
- Batch 3 (abilities, and sun / rain / hail as one family): committed, awaiting its review.
  Checks: differential (jax_matches_reference, payouts) green; focused engine, data and tools
  tests green; cold `jit(step)` 45.3 s and `jit(vmap(step))` 35.0 s against 39.5 / 32.2 s for
  a0033f9 measured in the same session (+15 % / +9 %; this machine gave 20.7 / 15.6 s for batch
  2 earlier, so compare the pairs); Roark split 104 -> 81 distinct groups, its move effect
  + held item + ability + terrain entries 47 -> 24. The AI's own refusals of the new abilities
  (Levitate, Shield Dust, ...) and the weather moves (#136, #164) stay for E1.
- Standing: the doubles conventions below apply to all new code; one review per batch.

Session E2 builds the battle mechanics the Roark split's trainers need (held items,
abilities, move effects, battle terrain), batch by family. The AI side belongs to E1.
Goal: no Roark-split fight blocked by a "move effect", "held item effect", "ability" or
"battle terrain" entry of `docs/kaizo/mechanics_backlog.md`.

Baseline (main e139cd1): 79 such entries (32 held item, 25 move effect, 21 ability,
1 terrain). Cold compile of `jit(step)` on vanilla Roark: 18.7 s (`jit(vmap(step))`, batch
64: 14.6 s), measured with `_scratch/e2/compile_time.py` on G:.

## Conventions for new code (user-approved 2026-11-13)

From section 5 of the doubles design (`docs/doubles-state-design.md` in the main checkout), so
singles mechanics need no rework when four-battler fights arrive. New code only; existing code,
the `step` turn skeleton and `enumerate.py`'s branch structure stay as they are.

1. Pass the other Pokemon in; never compute `1 - side` inside a mechanic (`hooks.pluck_berry`'s
   `foe = 1 - side` is the pattern not to copy; `effects._Hit` carries side and target).
2. Keep per-battler state (`BattleMon`, `turnFlags`, `[MAX_BATTLERS]` arrays) and per-side state
   (`SideConditions`) apart, as the decomp struct has it; say which in `STATE.md`; index side
   fields by the battler's side.
3. A field that names another Pokemon stores the battler (`1 + battler`), not a yes/no
   (Lock-On, Mean Look, Counter, Mirror Coat).
4. "Every Pokemon on the field" is a loop over battlers; "an ally has X" is a count over the
   battler's side.
5. State turn-order facts about a named battler ("the target has moved", "the target is
   protecting"), never "I am the second actor" (Payback, Avalanche, Sucker Punch).
6. Read a move's range only through `targets_user`, `self_defender`, `defender_always`; new
   chance fields are indexed by the acting battler, never a scalar meaning player vs opponent.

## Batches

1. Attacker damage items and resist berries: the type-boosting table, the species items
   of `BattleSystem_CalcMoveDamage`, Muscle Band / Wise Glasses, the Choice items, Expert
   Belt, Life Orb, every resist berry, the hold-effect-0 items.
2. Status orbs and the other berries and held items.
3. Abilities.
4. Move effects and Secret Power's terrain.

## Owed

Rules left out because another mechanic is missing; each closes with the mechanic named.

- King's Rock's flinch vs Substitute (effects.py `_kings_rock`; subscript_flinch_mon.s:6):
  Substitute. (Its Shield Dust test, :9, is modelled by E2 batch 3, B3-9.)
- The Sticky Barb transfer vs Knock Off, the knocked-off mask, U-turn and the substitute test
  (effects.py `sticky_barb_transfer`; battle_lib.c:5378, :5390-5393): those moves.
- Shell Bell and Life Orb vs `SYSCTL_UTURN_ACTIVE` (bcp.c:4739, :4761): U-turn.
- The Trace refusal (`coverage.trace_ai_line`, B3-21): the AI must read the battler's ability
  (`state.ability`, the traced copy, through `damage.decomp_abilities`) instead of
  `tables.species_abilities_decomp` by species in ai/commands.py (~558-580), ai/policy.py (~950)
  and ai/switching.py (~301), as `LoadBattlerAbility` does (trainer_ai.c:1170-1198); the AI owner's
  fix lifts it.
- Fire Fang (`BATTLE_EFFECT_FLINCH_BURN_HIT`) always bypasses Wonder Guard (the decomp's bug note,
  battle_lib.c:7600): Fire Fang / Wonder Guard.
- Gravity makes the moves that fail in high gravity (Jump Kick, Hi Jump Kick, ...) invalid:
  `Move_FailsInHighGravity` under `CHECK_INVALID_GRAVITY` in `BattleSystem_CheckInvalidMoves`
  (battle_lib.c:2292, :2348), which the AI also reads (trainer_ai.c:234); `actions.usable_moves`
  ignores Gravity: Gravity.
- The status orbs vs Safeguard (subscript_badly_poison.s, subscript_burn.s): Safeguard.
- `outcome.kings_rock_chance` reads the turn-start item (outcome.py), which breaks once a move
  can move items mid-turn: Trick, Thief, Fling.
- Kaizo's Ganlon and Apicot Berries (param 0) are refused by the divisor guard
  (`effect_map.DIVISOR_PARAM_MIN`, B2-3). Finding: `curHP <= maxHP / itemPower`
  (battle_lib.c:4778, :1294) is a C `/` on ints, which the compiler emits as a call to the
  Metrowerks / NitroSDK runtime's signed divide helper; that helper is linked from the SDK
  libraries (subprojects/NitroSDK, not in the decomp's own sources, and not fetched here), so
  the decomp does not say what a zero divisor returns. Closing it needs the helper's code read
  from the ROM's arm9 (its divide-by-zero branch), then the berries modelled with that result
  instead of refused.

## Rulings

(One line each, with the decomp citation.)

- B1-1 Item order: the held-item block runs after powerMul, Charge, Helping Hand, Technician,
  Huge Power / Pure Power and Slow Start (battle_lib.c:6688-6714), before every ability
  modifier from :6785 on (Thick Fat, Hustle, Guts, ...), and in the decomp's
  order (type item, Choice Band/Specs, Soul Dew, DeepSeaTooth/Scale, Light Ball, Metal Powder,
  Thick Club, orbs, Muscle Band/Wise Glasses), so e.g. Thick Club x2 truncates before Guts x1.5;
  the u16 stores never truncate in reach (battle_lib.c:6716-6783 before :6793).
- B1-2 Light Ball doubles `movePower` for any move class, physical and special alike
  (battle_lib.c:6748-6751).
- B1-3 The species items read `BATTLEMON_SPECIES`; Soul Dew's Battle Frontier test never holds in
  a trainer battle and the Griseous Orb's Transform test never holds while Transform is
  unsupported (battle_lib.c:6730-6739, :6768-6772).
- B1-4 The confusion self-hit gets the holder's own `BattleSystem_CalcMoveDamage` items (Choice
  Band, a Normal-type item for Struggle's type, Muscle Band, Light Ball, its own Metal Powder /
  DeepSeaScale / Soul Dew as defender), never Life Orb or Expert Belt (`CALC_SELF_HIT`,
  include/battle/common.h:35; battle_controller_player.c:2633).
- B1-5 Life Orb's x1.3 is the move script's (`BattleScript_CalcMoveDamage`, battle_script.c:1346,
  after `criticalMul`): not in the AI's `TrainerAI_CalcDamage` estimate (trainer_ai.c:3081) and not
  on a fixed-damage move (Night Shade / Seismic Toss: effect_script_0087.s has no CalcDamage; Bide);
  Pursuit's switch hit gets it (subscript_pursuit.s CalcDamage).
- B1-6 Life Orb recoil is `Divide(maxHP, 10)` (not the param), once per move after all hits,
  when a hit's `UpdateHP` ran with damage != 0 (`SYSCTL_MOVE_HIT`, battle_controller_player.c:3340,
  :3441), `moveCur` is not a status move and the holder lives: fixed-damage hits recoil too, a hit
  capped to 0 by a 1-HP Focus Sash still counts, the confusion self-hit never does
  (battle_controller_player.c:4758-4772).
- B1-7 Life Orb recoil also follows Pursuit's pre-switch hit unless it did not hit
  (subscript_pursuit.s:17, :48-49, :94-95 -> `BattleSystem_TriggerHeldItemOnPivotMove`,
  battle_lib.c:7357-7366).
- B1-8 Life Orb recoil runs inside the action, before its faint check: a holder that KOs the
  last foe and faints to its recoil ends the battle with both sides out, a loss (ruling 2;
  `TRIGGER_AFTER_HIT_EFFECTS` before `MOVE_END`, battle_controller_player.c:3897-3915).
- B1-9 Choice lock: set at `UpdateMoveBuffers` only when the attack message showed, to `moveTemp`
  (Nature Power itself, not its call), never to Struggle; a non-holder's lock is cleared there; a
  disrupted move (sleep, freeze, flinch, Disable, Taunt, confusion self-hit, paralysis, love)
  changes nothing (battle_controller_player.c:3921-3934).
- B1-10 The defender's lock is cleared in `UpdateFlagsWhenHit` unless it holds a Choice item, when
  the attack message showed (battle_controller_player.c:4489-4496); `BtlCmd_TryPursuit` locks a
  Choice holder into Pursuit (battle_script.c:6943-6950).
- B1-11 A Choice-locked battler whose locked move has no PP (or is disabled) has no usable move and
  Struggles; a lock the battler does not know is cleared, read as no lock
  (battle_lib.c:2307-2315; `STRUGGLING_ALL`, battle_controller_player.c:433).
- B1-12 A lock set on a battler that fainted during its move is never read and is cleared at the
  next switch-in (battle_lib.c:2100-2103); the engine clears it at the faint check.
- B1-13 Choice Scarf's x1.5 Speed comes after Swift Swim and before paralysis (battle_lib.c:1228,
  :1248, :1260).
- B1-14 Expert Belt multiplies a `MOVE_STATUS_SUPER_EFFECTIVE` hit after Solid Rock / Filter, inside
  `BattleSystem_ApplyTypeChart`, so the AI's estimate has it too (battle_lib.c:2657-2665;
  trainer_ai.c:3096).
- B1-15 Items with no hold effect do nothing, except that a berry among them (`Item_IsBerry`,
  src/item.c:357-365) is eaten by Pluck / Bug Bite with no effect, leaving the pluck flag set
  (battle_lib.c:5832-5835, :5848); the prep never offers such a berry (it can only help the foe).
- B2-1 Cheri, Rawst, Aspear and Persim Berries cure paralysis, burn, freeze and confusion at every
  held-item site (after each hit, on switch-in, `USE_ITEM` and `ITEM_CONDITION` at the turn's end),
  and a Pluck / Bug Bite that eats one cures the eater the same way (battle_lib.c:4601-4660,
  :4991-5050, :5569-5641).
- B2-2 A stat pinch berry (Liechi, Ganlon, Salac, Petaya, Apicot) raises its stat by one stage at
  `curHP <= maxHP / param` (C division; the holder's own Gluttony halves the param), only below
  +6, at the same sites; one eaten by Pluck / Bug Bite raises the eater's stat with no HP test
  (battle_lib.c:4773-4836, :5196-5259, :5753-5796).
- B2-3 A pinch berry or Custap Berry whose param is below 2 is unsupported: its trigger divides by
  zero (with Gluttony's halving), which the decomp leaves to the ROM's division routine
  (battle_lib.c:4778, :1294); Kaizo's Ganlon and Apicot Berries have param 0.
- B2-4 Custap Berry in its pinch sets the Quick Claw flag in every `CompareBattlerSpeed`: the turn
  order, the end-of-turn order and the AI's speed compares (battle_lib.c:1288-1300); its
  `custapBerry` flag is written by `CalcTurnOrder`'s FIGHT-against-FIGHT compare
  (bcp.c:772-786) and by `BattleSystem_SortMonActionOrder` (battle_lib.c:7276-7300, the same
  compare with `ignoreQuickClaw = FALSE`, :1296, :1362) after a Trick Room activation
  (bcp.c:3990-3993; Trick Room is unsupported: adding it must re-set the flag there), and the holder eats the berry when its move starts, before the status
  checks, first or last (subscript_check_quick_claw.s:26-28, bcp.c:3140-3143).
- B2-5 While that flag is up, a foe's Pluck / Bug Bite does nothing to the holder: no berry eaten,
  no pluck flag (battle_script.c:8125-8127).
- B2-6 Leftovers and Black Sludge act at `MON_COND_CHECK_STATE_LEFTOVERS`, right after `USE_ITEM`
  and before Leech Seed (bcp.c:1369): `Divide(maxHP, 16)` below full HP, for Black Sludge only on a
  Poison-type battler (its current types); any other Black Sludge holder loses `Divide(maxHP, 8)`
  unless Magic Guard; neither item is used up (battle_lib.c:4922-4965).
- B2-7 Toxic Orb, Flame Orb and Sticky Barb act at `MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT`, the
  last check (bcp.c:1710): an orb's held-item side effect skips its holder's own Immunity / Water
  Veil, Leaf Guard in sun without a weather lock, any status and a Poison or Steel / Fire type, else
  it badly poisons (counter 0) or burns (subscript_badly_poison.s:13-28, subscript_burn.s:5-17);
  Safeguard comes from an unsupported move and is not read.
- B2-8 Sticky Barb's holder loses `Divide(maxHP, param)` (param 8) at that check unless Magic Guard;
  it may faint (battle_lib.c:5330-5336).
- B2-9 Sticky Barb moves to the attacker after a hit of a contact move once the defender took damage
  this action, when the attacker lives and holds nothing (an eaten berry leaves nothing); the
  defender may have fainted, and the move is not a `RemoveItem` (no recycle item)
  (`AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT`, battle_lib.c:5386-5398,
  subscript_transfer_sticky_barb.s); Pursuit's pre-switch hit moves it too (subscript_pursuit.s:44,
  :90).
- B2-10 After every hit the order is the attacker's item, the defender's item, the Sticky Barb, the
  defender's thaw, then the held-item-status pass (bcp.c:3684-3730): a frozen defender's Aspear or
  Lum Berry cures the freeze before a Fire hit thaws it, and is used up (this also settles Lum,
  which the engines thawed first before batch 2).
- B2-11 Accuracy items come after the abilities and deep fog, before Gravity: the defender's
  BrightPowder / Lax Incense `* (100 - param) / 100`, then the attacker's Wide Lens and, against a
  target that has moved, Zoom Lens, each `* (100 + param) / 100` (bcp.c:2949-2965); "moved" is
  `Battler_MovedThisTurn` (battle_lib.c:5365-5368): in singles, the attacker acts second (a foe
  that switched in this turn has moved).
- B2-12 King's Rock / Razor Fang roll per hit after the contact ability (`*_EXTRA_FLINCH`) when the
  hit's `UpdateHP` ran, `RandNext() % 100 < param`, the move has `MOVE_FLAG_TRIGGERS_KINGS_ROCK` and
  the defender lives; the flinch needs a target that has not moved and no Inner Focus as the
  attacker sees it (bcp.c:4630-4653, subscript_flinch_mon.s:5-9): `Outcome.flinch_item[2, 5]`.
- B2-13 Shell Bell heals `Divide(dealt, param)` after the move, before Life Orb, where `dealt` sums each
  hit's `battleCtx->damage` after the endure cap, capped at the target's HP and not halved by a
  resist berry (bcp.c:3427-3431, :4736-4752); after Pursuit's hit `dealt` is the script's damage
  before the Focus Sash cap, capped at the HP (subscript_pursuit.s:21-29, battle_lib.c:7345-7355).
- B2-14 Light Clay makes its holder's Reflect / Light Screen last 5 + param (3) turns
  (battle_script.c:4229, :4272).
- B2-15 Gluttony is supported: the pinch berries and Custap are its only decomp readers
  (battle_lib.c:1289, :1355, :4774-4851, :5197-5275).
- B2-16 Kaizo changes item params only, read from each game's table: BrightPowder, Lax Incense and
  King's Rock have param 11 (vanilla 10); its Berserk Gene is a renamed Shoal Salt slot with
  hold effect 36 and param 0, so it stays unsupported (renamed slot, and B2-3).
- B3-1 Technician (`attackerParams.ability`) raises `movePower <= 60` by x15/10 after powerMul (Charge
  and Helping Hand, between them, are unsupported) and before the held items, never for Struggle
  (so not the confusion self-hit); it reads the input power (Hidden Power's, Magnitude's, Flail's,
  Rollout's) and is in the AI's estimate (battle_lib.c:6688-6702; trainer_ai.c:3078).
- B3-2 Reckless is only the recoil scripts' `BTLVAR_POWER_MULTI = 12` (effect_script_0048.s,
  _0269.s: the supported RECOIL moves; 0045, 0198, 0253, 0262 are unsupported effects): no AI
  estimate, no self-hit, and Pursuit is no recoil move (`damage.script_power_mul`).
- B3-3 Thick Fat (ignorable) halves Fire / Ice power right after the items (battle_lib.c:6785-6788);
  Dry Skin (ignorable) raises Fire power x125/100 after Overgrow / Blaze / Torrent (:6845-6848);
  Solar Power raises Sp. Atk x15/10 in sun inside `NO_CLOUD_NINE` (:6922-6924; the self-hit passes
  no field conditions).
- B3-4 In `BattleSystem_ApplyTypeChart`: Levitate (ignorable) sets `MOVE_STATUS_LEVITATED` for a
  Ground move (:2600-2603), part of `MOVE_STATUS_IMMUNE` and `DID_NOT_HIT`, so no damage, no
  AfterMoveEffects and a zero in the AI estimate (trainer_ai.c:3106); Scrappy breaks the walk at
  0xFE like Foresight (:2611-2615) for every chart reader (damage, Thunder Wave, the resist-berry
  flag, Night Shade's immunity); Tinted Lens doubles a not-very-effective hit after Expert Belt
  (:2668-2672). Status moves other than Thunder Wave skip the chart (bcp.c:2350-2358).
- B3-5 Battle Armor and Shell Armor (the same ignorable test, battle_lib.c:7135-7136) stop every
  crit; Super Luck adds 1 to the crit stage from the battler's own ability (:7126). Both engines
  gate the drawn crit with the armor at the hit; the laws draw none (`outcome.armored`).
- B3-6 No Guard of either battler (`Battler_Ability`, not ignorable) clears `MOVE_STATUS_MISSED` after
  the roll (bcp.c:3016-3021): threshold 100. Tangled Feet (ignorable) halves the rate against a
  confused target after fog and before the items (bcp.c:2944-2947).
- B3-7 Quick Feet with any status (`MON_CONDITION_ANY`) raises Speed x15/10 instead of the paralysis
  quartering (battle_lib.c:1257-1261, :1323-1327).
- B3-8 Serene Grace doubles `effectChance` (a u16, then `% 100 <`) in
  `BattleSystem_TriggerSecondaryEffect` and `BtlCmd_CheckEffectActivation` (battle_lib.c:1552-1556,
  :1571-1575; battle_script.c:8369-8373): every probabilistic secondary, Fake Out's flinch and both
  Thunder Fang rolls; not the contact abilities' or King's Rock's rolls.
- B3-9 Shield Dust (ignorable) silently stops every `SIDE_EFFECT_TYPE_INDIRECT` effect on its holder:
  a secondary status, confusion, flinch (King's Rock's too) or stat drop (battle_script.c:2918-2921;
  subscript_paralyze.s:13, _burn.s:27, _freeze.s:11, _poison.s:20, _badly_poison.s:38,
  _fall_asleep.s:31, _confuse.s:15, _flinch_mon.s:9), after Insomnia / Limber / Inner Focus; Bind's
  start (no check) and the contact abilities are not stopped.
- B3-10 Truant: `Battler_CheckTruant` loafs when `truant != (totalTurns & 1)`, `truant` being
  `(totalTurns + 1) & 1` from the entry (battle_lib.c:2119, :2197, :3304-3313): `fake_out_turn`'s
  parity, 0 for a lead, whatever the battler did. Checked after sleep and freeze, before the flinch
  (bcp.c:2502-2510); subscript_loafing_around.s unlocks and tags no ability; a loafing pursuer does
  not hit a switcher (battle_script.c:6903).
- B3-11 Dry Skin takes a damaging Water move that hit (Hidden Power's type too;
  `TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES`, bcp.c:3281-3286, :2782-2810, battle_lib.c:3538-3544):
  `Divide(maxHP, 4)` below full HP, its ability tagged either way, then `MOVE_STATUS_NO_MORE_WORK`
  (no hit, no secondary, no AfterMoveEffects; the attack message showed). At the turn's end sun
  takes `Divide(maxHP, 8)` from Dry Skin and Solar Power (Magic Guard skips it) and rain gives it to
  Dry Skin below full HP (battle_script.c:5585-5625), each tagging the ability.
- B3-12 The weather moves fail on their own weather, temporary or permanent, except Hail, which
  fails only on a temporary hail (effect_script_0164.s checks `FIELD_CONDITION_HAILING_TEMP`): it
  turns a permanent hail into a 5-turn one. Success clears every weather bit and sets 5 turns plus
  the user's matching rock param (Damp / Heat / Smooth / Icy Rock); a failure is
  `MOVE_STATUS_FAILED` (no AfterMoveEffects). They are `RANGE_FIELD`, accuracy 0.
- B3-13 Drizzle, Sand Stream, Drought, Snow Warning set their weather permanently at their holder's
  switch-in check when it has HP (after Stealth Rock), whatever weather was up; the script (and
  its ability message) is skipped only under that same permanent weather (battle_lib.c:3740-3790);
  Cloud Nine does not stop it.
- B3-14 Trace (battle_lib.c:3718-3737, `ChooseTraceTarget` :7750-7775): a holder with HP and no
  Griseous Orb copies a living foe's `BattleMon.ability` unless it is Forecast, Trace or Multitype;
  only the battler's (no form change: no party write), so it ends when it leaves; the AI learns the
  tracer's and the target's abilities (battle_script.c:3403, :11806). A Trace facing Trace stays
  armed and copies the next newcomer mid-turn, which the turn-start chance laws do not follow:
  Trace in both parties is refused (`coverage.trace_standoff_line`). A player Trace newcomer that
  entered against a fainted foe traces when the foe's replacement enters (`hooks.on_switch_in`
  runs Trace for the other battler too; `playout.replacement_pass` defers only the newcomer's own
  Intimidate).
- B3-15 Two weather abilities that set different weathers, one in each party, are refused
  (`coverage.weather_war_line`): entering together (battle start, a double KO's replacements) the
  `monSpeedOrder` loop (battle_lib.c:3740-3790, sorted by `CompareBattlerSpeed` at
  bcp.c:251, :263) leaves the slower one's weather, with an exact tie a `RandNext() & 1` the engine
  does not draw there.
- B3-16 Lightningrod redirects only to a battler other than the attacker and defender
  (battle_lib.c:1756-1772), which singles never has; Honey Gather acts only after the battle
  (battle_script.c:7825): both supported with no in-battle effect.
- B3-17 Kaizo changes none of these: its accuracy changes add effect 141 (Hurricane, unsupported) to
  Thunder's sun and rain branches (docs/kaizo/rom_diff.md, CheckMoveHitAccuracy /
  CheckMoveHitOverrides), and TriggerEffectOnSwitch only its field-weather switch.
- B3-18 The leads' switch-in checks run as one `BattleSystem_TriggerEffectOnSwitch` pass over both
  battlers, state by state (battle_lib.c:3649-4168; Trace :3718-3737 for every lead before
  Intimidate :3795-3808, each Intimidate announced once): a lead that traced Intimidate lowers the
  foe once, like any Intimidate lead (`hooks.battle_start`, from `make_state` and prep's
  `make_states`).
- B3-19 The turn-start laws read a switching-in target's entry ability, the copy when it traces
  (`damage.entry_ability`): its switch-in checks run before the foe moves, so a Trace newcomer
  facing a Sand Veil attacker in sand is hit at x80/100 (bcp.c:2927-2928), one facing a Static
  contact attacker rolls Static, and one that copies Battle Armor / Shell Armor is never crit
  (`outcome._rate`, `outcome.armored`, `enumerate._holds`, Cute Charm's `charm_later`).
- B3-20 Prep applies the party-pair refusals per candidate against the trainer's party
  (`weather_war_line`, `trace_standoff_line`): `prep.problem.usable_pool` leaves the candidate
  out and `prep.pool.check_pool` refuses it.
- B3-21 A Trace holder on either side (and in the pool) is refused as an AI gap
  (`coverage.trace_ai_line`): the AI reads `species_abilities_decomp` by species, where the
  decomp's `LoadBattlerAbility` (trainer_ai.c:1170-1198) returns the copy `subscript_trace.s`
  wrote through `BattleAI_SetAbility` (battle_script.c:3401-3403).
- B3-22 Stealth Rock's damage has one formula (`damage.stealth_rock_loss`), read by the switch-in
  (`effects.stealth_rock_damage`) and by the turn-start weather of a newcomer
  (`damage.entry_weather`).
