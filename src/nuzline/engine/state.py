"""The battle state pytree and its construction from two parties.

``BattleState`` is unbatched: the party axis (0 the player, 1 the opponent) and the battler axis
(singles: the same two) lead the arrays, then the slot axis in party order. Every field is ``int8``/``int16`` storage (spec §4.2), except ``exp``:
``int32``, because the level-100 experience of most growth rates exceeds ``int16`` (K3d-exp
ruling D1). The field list is ticked off against the decomp's battle structs in
``STATE.md``.

Encodings:

* ``species``/``moves``/``item``/``item_consumed``/``trainer_items``/``trap_move``/
  ``choice_lock``/``last_move``/``move_hit`` hold decomp ids (0 = none); ``ability``
  holds an ``ids.AbilityId`` (from ``tables.species_abilities``) and ``ability_slot``
  the species slot (0/1) it came from, so ``tables.species_abilities_decomp[row,
  ability_slot]`` is the decomp ability id, ``row = tables.species_form_row[species,
  ability_form]``; ``ability_form`` is the form whose personal row holds the ability
  (``PokemonSpec.ability_from``: 0 for a trainer Pokemon, whose ability comes from the base
  species, pokemon.c:473-485; ``form`` for the player's). ``species_form_row[species, form]``
  is the personal row of the Pokemon's form (types are read by it); ``status`` an
  ``ids.Status``.
* ``stats[..., 0..4]`` are Atk, Def, SpA, SpD, Spe (``ids.Stat`` minus one); HP is
  ``max_hp``. ``ivs`` are in ``ids.Stat`` order.
* ``stages[side, ids.Stat]`` are the active Pokemon's stat stages, 0 = neutral
  (the decomp stores 6 = neutral); index 0 (HP) is unused.
* ``weather`` is an ``ids.Weather``; ``weather_timer == 0`` means permanent.
* ``winner``: 0 none, 1 player lost (no player Pokemon with HP left, a
  double KO included), 2 player won.
* ``turn`` is the decomp's ``totalTurns`` while a turn is played and at each decision;
  ``step`` adds 1 at the end, so a faint replacement (``step.replace``, run after
  ``step``) sees ``turn`` one ahead of the decomp's, which increments after it
  (battle_controller_player.c:1844-1852).
* ``fake_out_turn`` is ``fakeOutTurnNumber``: 0 for the leads, ``totalTurns + 1`` after
  every switch-in and faint; the AI's ``LoadIsFirstTurnInBattle`` (trainer_ai.c:2487)
  is ``fake_out_turn >= turn``; Fake Out fails unless ``fake_out_turn == turn``.
* ``move_hit`` is ``moveHit``: the last move aimed at the side's active battler since its
  own last action ended (see ``step``).
* ``foresight``, ``disabled_move``/``disable_turns``, ``locked_move`` (0 = not locked),
  ``bide_turns``/``bide_damage`` (``bide_damage`` is ``storedDamage`` only while
  ``bide_turns > 0``, else 0), ``curled``, ``mud_sport`` and ``aqua_ring`` are the active battler's
  matching volatile / move-effect state; all are reset like the other volatiles
  (``step._VOLATILE``).
* ``infatuated`` is the active battler's ``VOLATILE_CONDITION_ATTRACT`` bits: 1 when it is in
  love with the other side's active battler (in singles the only possible source,
  ``BtlCmd_TryAttract``, battle_script.c:5738). It is a volatile (cleared when the battler
  leaves) and is also cleared when its source leaves (``step._source_leaves``).
* ``destiny_bond`` is the active battler's ``VOLATILE_CONDITION_DESTINY_BOND``
  (subscript_destiny_bond.s:9): cleared when its next move attempt starts
  (``CHECK_STATUS_START``, bcp.c:2425) and with the other volatiles (``step._VOLATILE``).
* ``uproar`` is the active battler's ``VOLATILE_CONDITION_UPROAR`` count (subscript_uproar.s:
  ``Random 3, 3``, 3-6; condition.h:66, :81), decremented by the end-of-turn check
  (``step._uproar_check``, bcp.c:1505-1541) and cleared with the other volatiles. The battler's
  ``FIELD_CONDITION_UPROAR`` bit (condition.h:118, :147) is set and cleared with it
  (subscript_uproar.s:9-11, bcp.c:1527-1534, battle_lib.c:2135, :2213), so it is
  ``uproar != 0``. While it is up, ``locked_move`` holds the move (``moveLockedInto``,
  subscript_uproar.s:5): ``Battler_CanPickCommand`` is FALSE (battle_lib.c:6224-6235).
* Future Sight / Doom Desire (E2 batch 5): ``future_timer``, ``future_damage``, ``future_move``
  and ``future_attacker`` are ``FieldConditions.futureSight*[battler]`` of the battler the move
  aims at (battle_script.c:6070-6090): the turns left, the HP the hit will take (the decomp keeps
  it negative, ``* -1``), the move and the attacker's battler as ``1 + battler`` (convention 3),
  0 when none; they stay when the battler leaves (field state). ``future_sight`` is the side's
  ``SIDE_CONDITION_FUTURE_SIGHT`` bit: set with them, cleared only when the hit fires on a
  battler with HP (bcp.c:1767-1770), so it can outlive the timer (the AI reads it).
* ``curse``, ``perish_song``/``perish_song_turns``, ``yawn`` and ``lock_on``/
  ``lock_on_target`` are the active battler's decomp state, which the trainer AI reads
  (``Expert_Protect``, script.s:3544-3586) and E2 batch 8 sets with Curse by a Ghost user,
  Perish Song, Yawn and Lock-On / Mind Reader (rulings B8-1..B8-5); all are volatiles, Baton
  Pass keeping ``curse``, the Perish Song pair and the Lock-On pair. ``curse`` is ``VOLATILE_CONDITION_CURSE`` (condition.h:60), 0/1;
  ``perish_song`` is ``MOVE_EFFECT_PERISH_SONG`` (moves.h:13), 0/1, and
  ``perish_song_turns`` is ``MoveEffectsData.perishSongTurns`` (battle_mon.h:14), the
  counter that is still live at 0 (the battler faints at the turn end that finds it 0,
  battle_controller_player.c:1801); ``yawn`` is the two-bit ``MOVE_EFFECT_YAWN`` counter in
  ``moveEffectsMask`` (moves.h:19-20, :42, :53), 0 = none; ``lock_on`` is the two-bit
  ``MOVE_EFFECT_LOCK_ON`` counter (moves.h:11-12, :41), 0 = none, on the battler **locked
  onto** (Lock-On sets it on its target, subscript_lock_on.s), and ``lock_on_target`` is that
  battler's ``MoveEffectsData.lockOnTarget`` (battle_mon.h:23), the battler that locked on,
  stored as ``1 + battler``, 0 = none
  (doubles convention 3, docs/doubles-state-design.md section 5).
* ``ai_item_known`` is ``aiContext.battlerHeldItems`` of the player's party slot: the held item
  a battle message named (``BattleAI_SetHeldItem``, battle_script.c:11726-11745, :12119-12122;
  ``effects.reveal_item``), 0 when none. It is cleared when the Pokemon loses its item (every
  removal runs ``BattleMon_CopyToParty``, battle_lib.c:1819-1823; ``effects.forget_item``) and
  when it enters (``BattleSystem_UpdateAfterSwitch``, :2153; ``step._forget``). The AI reads it
  for the other side's battlers only (trainer_ai.c:1901-1934), so the opponent's is not kept.
* E2 batch 7, the state the trainer AI reads (rulings B7-n). ``ingrain`` is the active battler's
  ``MOVE_EFFECT_INGRAIN`` (moves.h:18), 0/1. ``mean_look`` is its ``VOLATILE_CONDITION_MEAN_LOOK``
  (condition.h:58), 0/1, and ``mean_look_target`` its ``MoveEffectsData.meanLookTarget``
  (battle_mon.h:26) as ``1 + raw`` (0 for a zeroed field): subscript_mean_look.s:10 ORs the
  attacker's battler into it and nothing clears it but the trapped battler's own switch-in or faint
  (its moveEffectsData zeroed; Baton Pass keeps it, battle_lib.c:2111), so a stale value outlives
  the Mean Look and enters the next OR (B7-15); the leave tests read it with the bit
  (:2058-2059). ``mean_look`` ends when the trapper leaves by a switch that is not a Baton Pass
  (battle_lib.c:2055-2061) or faints (:2168-2173).
  ``imprison`` is ``MOVE_EFFECT_IMPRISON`` (moves.h:21) on Imprison's user, ``imprisoned``
  ``MOVE_EFFECT_IMPRISONED`` (:38) on its foes, set whenever the user's Imprison script runs,
  success or not (battle_script.c:6686-6690), and only cleared when that battler leaves. All five
  are volatiles; Baton Pass keeps ``ingrain``, ``mean_look`` and ``mean_look_target``.
  ``roosting`` is ``turnFlags.roosting`` (battle_context.h:64): a Roost that healed sets it for the
  rest of the turn (subscript_roost.s:11-12); a faint clears it (battle_lib.c:2189-2192) and so
  does ``BattleSystem_SetupNextTurn`` (:2229-2231), which runs after the faint replacements
  (bcp.c:1843-1856). It is kept where it can still be read: at a pause at a party list and while
  a replacement is pending (the AI's ``BattleAI_PostKOSwitchIn`` walks the type chart, whose
  ``BasicTypeMulApplies`` reads it, battle_lib.c:2540); else 0.
* ``can_unburden`` is the active battler's ``MoveEffectsData.canUnburden`` (battle_mon.h:31,
  E2 batch 9): ``BattleSystem_RecoverStatusByAbility`` sets it for an Unburden holder
  (``Battler_Ability``) holding an item (battle_lib.c:4433-4437), at every move end for the
  attacker and the defender (bcp.c:3965-3967) and in the switch-in checks for every battler
  (battle_lib.c:4135-4145); ``BattleSystem_InitBattleMon`` sets it for any holder of an item
  (:161-166), which matters only at the battle start (a switch's UpdateAfterSwitch zeroes the
  whole MoveEffectsData after it, :2100-2104, Baton Pass included): ``hooks.battle_start`` sets
  it for every lead holding an item, whatever its ability (a lead that gains Unburden later
  keeps it). Nothing clears it but a switch (a volatile).
  ``order.effective_speed`` doubles an Unburden holder with it and no item (:1268-1272).
* ``recycle_item`` is ``BattleContext.recycleItem[battler]`` (battle_context.h:307; E2 batch 13,
  B13-19): ``BtlCmd_RemoveItem`` writes the item it removes (battle_script.c:8977; every
  ``RemoveItem``: ``effects.consume_item``, ``hooks.held_item_check``, ``hooks.pluck_berry``) and
  ``BtlCmd_TryRecycle`` reads and clears it (:9005-9007). It belongs to the battler position, not
  the Pokemon: nothing else writes it, so a switch keeps it for the newcomer. The party slot's
  ``item_consumed`` is the engine's own record of the same removals (observation only).
* ``minimized`` is the active battler's ``MOVE_EFFECT_MINIMIZE`` (moves.h:16; E2 batch 9): set by
  Minimize (subscript_minimize.s:5), read by Stomp (effect_script_0150.s:7); a volatile Baton Pass
  does not keep (moves.h:46-48).
* ``semi_invulnerable`` is the active battler's MOVE_EFFECT_SEMI_INVULNERABLE bit as an
  ``ids.SemiKind`` (moves.h:14-15, :26, :37, :44; E2 batch 10, B10-12): the charge turn of Fly /
  Bounce (AIRBORNE), Dig, Dive and Shadow Force sets it (effect_script_0155.s:7 and the others),
  their attack turn clears it before the hit (:24), and so does every Battler_UnlockMoveChoice
  (battle_lib.c:1838: a disrupted or missed move, a sleeper at the next turn's setup). A
  volatile Baton Pass does not keep (moves.h:46-48). A foe's move aimed at it misses unless its
  script hits that kind (bcp.c:3034-3040; ``MoveFlag.HIT_*``); the charge's lock is
  ``locked_move``.
* E1 Batch D. ``pp_ups`` is each move slot's ``ppUps`` (``BattleMon.ppUps``, from the party
  Pokemon, battle_lib.c:85): with the move it gives ``MoveTable_CalcMaxPP`` (move_table.c:25-34),
  which caps the Leppa Berry's restore (``BattleMon_AddVal``, battle_lib.c:1009-1019) and picks
  the slot Pluck's Leppa restores (:5611-5621); a level-up move that replaces a slot sets it to 0
  (``Pokemon_ResetMoveSlot``, pokemon.c:3909-3918). ``micle`` is the active battler's
  ``MoveEffectsData.micleBerry`` (subscript_held_item_temp_acc_up.s:10), cleared by its next
  ordinary accuracy check (bcp.c:2967-2970). ``metronome_turns`` / ``metronome_move`` are its
  ``MoveEffectsData.metronomeTurns`` and ``BattleContext.metronomeMove[battler]`` (the Metronome
  item's count and the move it counts, battle_lib.c:6160-6198). All three are volatiles: a switch
  or a faint zeroes ``moveEffectsData`` and ``metronomeMove`` (:2100-2104, :2134; :2212), Baton
  Pass too (it keeps none of them, :2107-2115).
* E2 batch 12 (rulings B12-n). ``focus_energy`` is the active battler's
  VOLATILE_CONDITION_FOCUS_ENERGY (condition.h:52), 0/1: Focus Energy sets it
  (subscript_focus_energy.s:6), Psych Up ORs the defender's in (battle_script.c:6050), Haze
  clears every battler's (:3544); +2 crit stages (battle_lib.c:7123). ``power_trick`` is its
  MOVE_EFFECT_POWER_TRICK (moves.h:31), 0/1, toggled by Power Trick
  (subscript_user_swap_atk_and_def.s:5); while it is set the BattleMon Attack and Defense are
  the party's swapped (:6-9, and battle_lib.c:2137-2141 for a Baton Pass newcomer), which
  ``damage.battle_stats`` reads. Both are volatiles Baton Pass keeps (condition.h:72,
  moves.h:47). ``power_trick_desync`` (0/1) is whether the BattleMon's Attack / Defense swap
  disagrees with that bit: BattleSystem_ReloadPokemon (battle_lib.c:169-174) reloads them
  unswapped from the party on a level-up of the active battler (battle_script.c:9860-9865,
  :10022-10027, :10173-10177) and leaves the bit, so the swap is ``power_trick XOR
  power_trick_desync``; Power Trick toggles both (unchanged), any switch clears it (a Baton Pass
  newcomer is swapped by the bit alone, battle_lib.c:2137-2141), B12-10.
* E1 trial batch (G-68, Mimic). The decomp keeps the party ``Pokemon`` and the active
  ``BattleMon`` apart; the engine's per-slot ``moves`` / ``pp`` are both. Where the battler's
  differ from the party's, ``mimic_slots`` (``MoveEffectsData.mimickedMoveSlot``) flags the move
  slots and ``shadow_moves`` / ``shadow_pp`` hold the party's value of each (battle_script.c:
  4502-4516). Every ``BattleMon_CopyToParty`` skips a flagged slot (its ``BtlIOCmd_UpdatePartyMon``,
  battle_io_command.c:581-587), a
  level-up's learn writes the party only (:10168-10176) and ``ReloadPokemon`` reloads the other
  slots (:183-191), and the next ``InitBattleMon`` brings the party's back (``step._restore_moves``
  on leaving, ``moveEffectsData`` is cleared, :2100-2104). ``ai_seen_moves`` holds
  ``battlerMoves`` as move ids (``TrainerAI_RecordLastMove``, trainer_ai.c:2705-2716): a slot's
  move can change under Mimic, so the record cannot be flags.
* E1 trial batch (G-69, Transform). ``transformed`` is VOLATILE_CONDITION_TRANSFORM: the active slot's
  species, form, stats, IVs, ability slot / form, types, ability, moves and PP are then the copied
  target's (``BtlCmd_Transform``, battle_script.c:5342-5348, :5364-5370), and the ``shadow_*`` fields hold
  the party Pokemon's own (``shadow_moves`` / ``shadow_pp`` of every slot, ``mimic_slots`` all four).
  They are restored on leaving like the Mimic shadow (``step._restore_transform``), not volatile: a
  fainted battler keeps the copy until its replacement enters (the exp payout reads it). ``truant`` is
  ``MoveEffectsData.truant``. ``ai_ability_known`` is ``battlerAbilities`` as the decomp ability id.
* ``trainer_items`` / ``trainer_item_count`` are ``aiContext.trainerItems[battler >> 1]`` /
  ``trainerItemCounts[battler >> 1]`` (the bag's size at battle start, never decremented), held
  on the opposing battler ``b`` whose ``b >> 1`` it is (1 -> bag 0, 3 -> bag 1): one bag per
  opposing trainer slot, the player's battlers' rows empty (ruling D1-5).
* ``terrain`` is ``BattleSystem.terrain``: the battle terrain, a decomp ``TERRAIN_*`` id
  (``nuzline.data.trainer.TERRAIN_NAMES``) fixed when the battle starts
  (``SetBackgroundAndTerrain``, field_battle_data_transfer.c:496-505) and never written in
  battle; ``TERRAIN_UNKNOWN`` (-1) when the trainer's maps give several terrains or none
  (``Trainer.terrain``, tools/terrain.py). Only Nature Power, Secret Power and Camouflage
  read it (battle_script.c:8007, :8040, :8067), and coverage refuses them unless it is known.
* ``qc_roll`` is ``BattleContext.speedRand[side] % (100 / QUICK_CLAW_PARAM) == 0`` (1 = a
  Quick Claw would activate), the per-battler draw the next turn's order reads
  (``order.action_order``). ``speedRand`` belongs to the battler slot, not to a Pokemon: it
  is redrawn every turn after the order is fixed (battle_controller_player.c:853-857), and
  ``step`` stores the turn's draw (``Outcome.quick_claw``) only while something can read it
  before the next redraw (``step.qc_kept``: a living Quick Claw holder is active, or the
  side's replacement can send one in); else 0. Hidden from the player. At a root it is a
  chance node (``step.root_belief``, ``step.sample_root``), as the battle-start draw
  (``BattleContext_InitCounters``, battle_lib.c:2024) is.
* ``war_swaps`` (E1 Gardenia batch, ruling G-26): two newcomers that run the weather state
  together (the leads at the battle start, a double KO's replacements) with switch-in weather
  abilities setting different weathers leave the slower one's weather in ``monSpeedOrder``
  (battle_lib.c:3740-3790), the order ``BattleSystem_SortMonSpeedOrder`` (:3364-3382) gave them
  before (battle_controller_player.c:251), which reads each Quick Claw's ``speedRand`` and a
  speed tie's ``RandNext() & 1``. The engine runs the player's first; where that order is
  still open (a draw not yet made: the battle-start ``speedRand`` and a tie, or a tie at a
  replacement), ``war_swaps`` holds its bits: bit ``tie + 2 * qc[0] + 4 * qc[1]`` set where the
  sort puts the player second, so its weather stays (``hooks.order_swaps``). The draw that
  settles it (``step.resolve_war``: the root chance node, a playout's next turn, the tree's
  replacement children) then clears it; 0 otherwise, so states merge.
* ``entry_pending`` (E1 Gardenia review, ruling G-32): per battler, 1 while a battler that a
  replacement's party list sent in (subscript_show_party_list.s:22-24: the hazards and a faint
  check only) still waits for ``ShowBattleMon``'s switch-in states (battle_controller_player.c:
  256, ``BattleSystem_TriggerEffectOnSwitch``: its BattleMon ``traceAnnounced``,
  ``weatherAbilityAnnounced``, ``intimidateAnnounced``, ``downloadAnnounced`` all clear) while
  the player sends in another Pokemon first: at a double KO whose player newcomer fainted on the
  entry hazards (``search.playout.replacement_pass``). Else 0, so states merge.
* Experience (K3d-exp; ``STATE.md`` "Experience"): ``exp`` is the player party's
  ``MON_DATA_EXPERIENCE``; ``evs`` (``ids.Stat`` order, like ``ivs``) and ``nature`` feed the
  level-up stat recompute; ``exp_boost``, ``pokerus`` and ``friend_bonus`` are the
  ``PokemonSpec`` fields of the player's slots. ``exp_mask`` is ``sideGetExpMask[0]`` as a
  bitmask over the player's party slots: ``1 << lead`` at the start, then written by every
  send-out (``step._switch_in``). ``learn_level`` / ``learn_cursor`` (the pending learnset
  scan), ``learn_pin`` and ``exp_refused`` are 0 until the payout and the learn resolve set
  them (later K3d-exp tasks).
* E2 batch 13 (B13-29, B13-30). ``embargo`` is the active battler's
  ``MoveEffectsData.embargoTurns`` (battle_mon.h:30, 3 bits), 0 = none. Its
  ``MOVE_EFFECT_EMBARGO`` bit (moves.h:34) is set and cleared with it (subscript_embargo_start.s:
  13-14, subscript_embargo_end.s:9 right after the count reaches 0, battle_controller_player.c:
  1671-1677) and both are volatiles Baton Pass keeps (moves.h:46-48, battle_lib.c:2112), so one
  field holds both. ``held_item`` is ``Battler_HeldItem`` (battle_lib.c:5352-5363): the held item
  as every gated read sees it, none under Klutz or Embargo.
* The mid-turn switch decision (E2 batch 6, U-turn and Baton Pass; docs/kaizo/
  e2-mechanics-ledger.md "Batch 6 design"): a turn can pause at the party list of
  subscript_attack_then_switch_out.s:71 / subscript_baton_pass.s:17, inside one action.
  ``pivot_battler`` is ``1 + battler`` of the battler owing the pick (one at a time: each list is
  inside one action and its flag clears at the switch, battle_script.c:2511, or the faint,
  battle_lib.c:2222), else 0. While it is set, the ``ts_*`` fields hold the turn's scratch
  (``BattleContext``: the order and its counter, the picks of command selection, the turn flags
  later actors read, the paused action's own inputs, and the search's ``powerMul`` reads), which
  ``step`` resumes from (``step.step``'s docstring); otherwise every ``ts_*`` is 0, so states
  between turns merge. ``qc_roll`` is 0 at a pause: the order read it, and the next read is the
  next turn's, from the resumed half's draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.data.trainer import TERRAIN_NAMES, TERRAIN_UNKNOWN
from nuzline.engine.format import SINGLES, Format, format_of, party, side_of
from nuzline.engine.ids import CHATTER_CHANCES, GRAVITY_TURNS, AbilityId, Stat, Weather
from nuzline.engine.stats import calc_stats

SIDES, SLOTS, MOVES = 2, 6, 4
MAX_DEATH_BUDGET = 3  # the ladder's range: k = 0..3 deaths allowed (spec §4)
BAG_SLOTS = 4  # MAX_TRAINER_ITEMS: trainer_items[battler] holds the AI's bag (D1-5)
STAGE_MIN, STAGE_MAX = -6, 6
_STAGE_KEYWORDS = {
    "atk": Stat.ATK,
    "def_": Stat.DEF,
    "spa": Stat.SPA,
    "spd": Stat.SPD,
    "spe": Stat.SPE,
    "acc": Stat.ACC,
    "eva": Stat.EVA,
}


class BattleState(NamedTuple):
    # per party x slot [P,6]
    species: jnp.ndarray  # int16
    form: jnp.ndarray  # int8
    level: jnp.ndarray  # int8
    hp: jnp.ndarray  # int16
    max_hp: jnp.ndarray  # int16
    stats: jnp.ndarray  # int16[P,6,5]: Atk, Def, SpA, SpD, Spe
    types: jnp.ndarray  # int8[P,6,2]
    ability: jnp.ndarray  # int16, ids.AbilityId
    ability_slot: jnp.ndarray  # int8, species ability slot 0/1
    ability_form: jnp.ndarray  # int8, the form whose personal row ability_slot indexes
    item: jnp.ndarray  # int16
    item_consumed: jnp.ndarray  # int16, id of the last item consumed
    # int8, the slot's bit of its side's knockedOffItemsMask (battle_script.c:6627), B10-6
    knocked_off: jnp.ndarray
    moves: jnp.ndarray  # int16[P,6,4]
    pp: jnp.ndarray  # int8[P,6,4]
    pp_ups: jnp.ndarray  # int8[P,6,4], BattleMon.ppUps (E1 Batch D)
    status: jnp.ndarray  # int8, ids.Status
    status_counter: jnp.ndarray  # int8, sleep turns / toxic counter
    gender: jnp.ndarray  # int8, 0 male, 1 female, 2 none
    friendship: jnp.ndarray  # int16
    ivs: jnp.ndarray  # int8[P,6,6], ids.Stat order
    # per battler [B]: the active Pokemon
    active: jnp.ndarray  # int8, party slot
    stages: jnp.ndarray  # int8[B,8], index ids.Stat, 0 = neutral
    confusion: jnp.ndarray  # int8, turns left
    leech_seed: jnp.ndarray  # int8
    substitute_hp: jnp.ndarray  # int16
    taunt: jnp.ndarray  # int8, turns left
    trapped: jnp.ndarray  # int8, turns left
    trap_move: jnp.ndarray  # int16
    flinched: jnp.ndarray  # int8
    choice_lock: jnp.ndarray  # int16
    last_move: jnp.ndarray  # int16
    protect_count: jnp.ndarray  # int8
    rollout_count: jnp.ndarray  # int8
    charge_turn: jnp.ndarray  # int8
    recharge: jnp.ndarray  # int8
    pursuit_target: jnp.ndarray  # int8, reserved: no mechanic sets or reads it yet (only cleared)
    water_sport: jnp.ndarray  # int8
    fake_out_turn: jnp.ndarray  # int16, fakeOutTurnNumber
    move_hit: jnp.ndarray  # int16, moveHit: decomp move id, 0 = none
    foresight: jnp.ndarray  # int8, VOLATILE_CONDITION_FORESIGHT
    disabled_move: jnp.ndarray  # int16, MoveEffectsData.disabledMove: move id, 0 = none
    disable_turns: jnp.ndarray  # int8, MoveEffectsData.disabledTurns (hidden)
    # E2 batch 10 (B10-9): MoveEffectsData.encoredMove / encoredMoveSlot / encoredTurns.
    encore_move: jnp.ndarray  # int16, 0 = not encored
    encore_slot: jnp.ndarray  # int8
    encore_turns: jnp.ndarray  # int8 (hidden)
    locked_move: jnp.ndarray  # int16, moveLockedInto while VOLATILE_CONDITION_MOVE_LOCKED
    bide_turns: jnp.ndarray  # int8, VOLATILE_CONDITION_BIDE turns, 0-2
    bide_damage: jnp.ndarray  # int16, storedDamage while Biding (hidden), else 0
    curled: jnp.ndarray  # int8, VOLATILE_CONDITION_DEFENSE_CURL
    mud_sport: jnp.ndarray  # int8, MOVE_EFFECT_MUD_SPORT in moveEffectsMask
    aqua_ring: jnp.ndarray  # int8, MOVE_EFFECT_AQUA_RING in moveEffectsMask
    infatuated: jnp.ndarray  # int8, VOLATILE_CONDITION_ATTRACT (0/1 in D1; 1 + battler in D5)
    destiny_bond: jnp.ndarray  # int8, VOLATILE_CONDITION_DESTINY_BOND
    uproar: jnp.ndarray  # int8, VOLATILE_CONDITION_UPROAR turns left (its FIELD_CONDITION bit)
    curse: jnp.ndarray  # int8, VOLATILE_CONDITION_CURSE
    perish_song: jnp.ndarray  # int8, MOVE_EFFECT_PERISH_SONG in moveEffectsMask
    perish_song_turns: jnp.ndarray  # int8, MoveEffectsData.perishSongTurns (live at 0)
    yawn: jnp.ndarray  # int8, MOVE_EFFECT_YAWN counter in moveEffectsMask, 0 = none
    lock_on: jnp.ndarray  # int8, MOVE_EFFECT_LOCK_ON counter in moveEffectsMask, 0 = none
    lock_on_target: jnp.ndarray  # int8, MoveEffectsData.lockOnTarget as 1 + battler, 0 = none
    ingrain: jnp.ndarray  # int8, MOVE_EFFECT_INGRAIN in moveEffectsMask
    mean_look: jnp.ndarray  # int8, VOLATILE_CONDITION_MEAN_LOOK
    mean_look_target: jnp.ndarray  # int8, MoveEffectsData.meanLookTarget as 1 + battler, 0 = none
    imprison: jnp.ndarray  # int8, MOVE_EFFECT_IMPRISON (the user of Imprison)
    imprisoned: jnp.ndarray  # int8, MOVE_EFFECT_IMPRISONED (a foe of an Imprison user)
    roosting: jnp.ndarray  # int8, turnFlags.roosting, kept only while it can be read
    can_unburden: jnp.ndarray  # int8, MoveEffectsData.canUnburden
    recycle_item: jnp.ndarray  # int16[B], BattleContext.recycleItem per battler (B13-19)
    minimized: jnp.ndarray  # int8, MOVE_EFFECT_MINIMIZE in moveEffectsMask
    semi_invulnerable: jnp.ndarray  # int8, ids.SemiKind: MOVE_EFFECT_SEMI_INVULNERABLE's bit
    micle: jnp.ndarray  # int8, MoveEffectsData.micleBerry (E1 Batch D)
    metronome_turns: jnp.ndarray  # int8, MoveEffectsData.metronomeTurns, 0-10
    metronome_move: jnp.ndarray  # int16, BattleContext.metronomeMove[battler], 0 = none
    focus_energy: jnp.ndarray  # int8, VOLATILE_CONDITION_FOCUS_ENERGY (B12-4)
    power_trick: jnp.ndarray  # int8, MOVE_EFFECT_POWER_TRICK (B12-10)
    power_trick_desync: jnp.ndarray  # int8, the Attack / Defense swap XOR power_trick (B12-10)
    magnet_rise: jnp.ndarray  # int8, MOVE_EFFECT_MAGNET_RISE in moveEffectsMask (G-66)
    magnet_rise_turns: jnp.ndarray  # int8, MoveEffectsData.magnetRiseTurns, 0-5 (G-66)
    # The party Pokemon's moves and PP under the BattleMon's (G-68): ``moves`` / ``pp`` of the
    # active slot are the BattleMon's, and these hold the party's values of the slots the battler
    # changed without the party seeing it. ``mimic_slots`` is ``mimickedMoveSlot`` (bit k = move slot
    # k holds a Mimic copy), ``shadow_moves`` / ``shadow_pp`` the party's move and PP of each such
    # slot (0 elsewhere).
    mimic_slots: jnp.ndarray  # int8, MoveEffectsData.mimickedMoveSlot bitmask (G-68)
    shadow_moves: jnp.ndarray  # int16[B,4], the party's move of each masked slot (G-68)
    shadow_pp: jnp.ndarray  # int8[B,4], the party's PP of each masked slot (G-68)
    # Transform (G-69): ``transformed`` is VOLATILE_CONDITION_TRANSFORM; while it is set the active
    # slot's species, form, stats, ivs, ability slot / form, types, ability, moves and PP are the
    # copied target's (``BtlCmd_Transform``, battle_script.c:5342-5348, :5364-5370), and the
    # ``shadow_*`` hold the party Pokemon's own (``shadow_moves`` / ``shadow_pp`` of every slot,
    # ``mimic_slots`` all four). Restored by ``step._restore_transform`` / ``_restore_moves``.
    transformed: jnp.ndarray  # int8, VOLATILE_CONDITION_TRANSFORM, kept until restored (G-69)
    shadow_species: jnp.ndarray  # int16, the party Pokemon's species while transformed (G-69)
    shadow_form: jnp.ndarray  # int8 (G-69)
    shadow_stats: jnp.ndarray  # int16[B,5], the party's Atk, Def, SpA, SpD, Spe (G-69)
    shadow_ivs: jnp.ndarray  # int8[B,6] (G-69)
    shadow_ability_slot: jnp.ndarray  # int8 (G-69)
    shadow_ability_form: jnp.ndarray  # int8 (G-69)
    # int8, MoveEffectsData.truant: the parity Truant compares with ``totalTurns & 1``
    # (``order.truant_loafs``): set to ``(totalTurns + 1) & 1`` at an entry (battle_lib.c:2119,
    # :2197), to ``totalTurns & 1`` by Transform (battle_script.c:5359), G-69.
    truant: jnp.ndarray
    # int8 per party [P]: party 0 the save's, every other 0 (ruling D1-2). BattleSystem.chatotCries
    # as Chatter's parameter 0 / 1 / 2 (G-70), a battle constant like ``terrain``, set by
    # ``initial_arrays``.
    chatot_cry: jnp.ndarray
    embargo: jnp.ndarray  # int8, MoveEffectsData.embargoTurns, 0 = none (B13-29)
    # side conditions [2], then per-battler [B] and per-party [P] fields (``_KEYS`` has each one)
    spikes: jnp.ndarray  # int8, layers
    toxic_spikes: jnp.ndarray  # int8, layers
    stealth_rock: jnp.ndarray  # int8
    reflect: jnp.ndarray  # int8, turns left
    light_screen: jnp.ndarray  # int8
    safeguard: jnp.ndarray  # int8
    mist: jnp.ndarray  # int8, mistTurns left (per side, SideConditions; B12-23)
    lucky_chant: jnp.ndarray  # int8
    tailwind: jnp.ndarray  # int8
    wish_timer: jnp.ndarray  # int8
    wish_hp: jnp.ndarray  # int16
    future_timer: jnp.ndarray  # int8, futureSightTurns (per battler)
    future_damage: jnp.ndarray  # int16, -futureSightDamage: the HP the hit takes (per battler)
    future_move: jnp.ndarray  # int16, futureSightMove (per battler)
    future_attacker: jnp.ndarray  # int8, 1 + futureSightAttacker (per battler), 0 = none
    future_sight: jnp.ndarray  # int8, SIDE_CONDITION_FUTURE_SIGHT (per side)
    trainer_items: jnp.ndarray  # int16[B,4], bag item ids (aiContext.trainerItems, ruling D1-5)
    trainer_item_count: jnp.ndarray  # int8[B], bag size at battle start (trainerItemCounts)
    qc_roll: jnp.ndarray  # int8, speedRand % 5 == 0 (Quick Claw), hidden, see the docstring
    # field (scalars)
    weather: jnp.ndarray  # int8, ids.Weather
    weather_timer: jnp.ndarray  # int8, 0 = permanent
    trick_room: jnp.ndarray  # int8
    gravity: jnp.ndarray  # int8
    war_swaps: jnp.ndarray  # int16, a weather war's open monSpeedOrder (docstring), 0 = none
    entry_pending: jnp.ndarray  # int8[B], 1: a list newcomer's switch-in states wait (docstring)
    move_prev: jnp.ndarray  # int16, BattleContext.movePrev: the last move used in the battle
    terrain: jnp.ndarray  # int8, decomp TERRAIN_* id, -1 unknown (constant per battle)
    turn: jnp.ndarray  # int16
    over: jnp.ndarray  # int8
    winner: jnp.ndarray  # int8
    player_faints: jnp.ndarray  # int8
    # AI knowledge of the player
    # int16[6,4], battlerMoves of the player's slot: move ids in first-free order (G-68)
    ai_seen_moves: jnp.ndarray
    ai_item_known: jnp.ndarray  # int16[6], battlerHeldItems: the item id a message named, 0 none
    # int16[6], battlerAbilities: the decomp ability id a message named, 0 = none (G-69)
    ai_ability_known: jnp.ndarray
    # experience (K3d-exp, design §2.1; see the module docstring)
    exp: jnp.ndarray  # int32[6], player party MON_DATA_EXPERIENCE
    evs: jnp.ndarray  # int16[P,6,6], ids.Stat order
    nature: jnp.ndarray  # int8[P,6]
    exp_mask: jnp.ndarray  # int8, sideGetExpMask[0]: bit = player party slot
    exp_boost: jnp.ndarray  # int16[6], 100 / 150 / 170 percent, 0 = unresolved
    pokerus: jnp.ndarray  # int8[6], 1 = the Pokerus byte is nonzero
    friend_bonus: jnp.ndarray  # int8[6], 1 = caught in a Luxury Ball
    learn_level: jnp.ndarray  # int8[6], 0 = no learnset work, else the level to scan
    learn_cursor: jnp.ndarray  # int8[6], index into that level's learnset scan
    learn_pin: jnp.ndarray  # int8, 0 none, else 1 + the move slot the player selected
    exp_refused: jnp.ndarray  # int8, sticky refusal code (0 none, 2 = window (c))
    # faint budget (search data; the engine only records deaths and never reads k)
    deaths: jnp.ndarray  # int8[6], 1 where the player's party slot fainted this battle
    faint_budget: jnp.ndarray  # int8, k: the deaths the search allows
    protected: jnp.ndarray  # int8[6], 1 where the player's slot must not faint
    replace_mul: jnp.ndarray  # int8, the turn's powerMul while a player replacement is pending
    # E2 batch 6: the mid-turn switch decision (U-turn, Baton Pass; see the docstring).
    pivot_battler: jnp.ndarray  # int8, 1 + the battler owing the party pick, 0 = none
    ts_cursor: jnp.ndarray  # int8, turnOrderCounter: the paused action's position
    ts_acted: jnp.ndarray  # int8[B], the battler's action is over (Battler_MovedThisTurn)
    # int8[B], battlerActionOrder (written from order.action_order's [2] until D6-D7)
    ts_order: jnp.ndarray
    ts_actions: jnp.ndarray  # int8[B], the turn's actions as chosen
    ts_move: jnp.ndarray  # int16[B], each battler's move fixed at selection (selected_move)
    ts_struggling: jnp.ndarray  # int8[B]
    ts_locked: jnp.ndarray  # int8[B], locked at selection (no PP)
    ts_item: jnp.ndarray  # int16[B], the bag item picked
    ts_custap: jnp.ndarray  # int8[B], moveEffectsData.custapBerry from the turn order
    ts_claw: jnp.ndarray  # int8[B], moveEffectsData.quickClaw from the turn order (B10-6)
    ts_pursued: jnp.ndarray  # int8, 1 + the battler whose action a Pursuit ended, 0 = none
    ts_protect_ok: jnp.ndarray  # int8[B], turnFlags.protecting
    ts_endure_ok: jnp.ndarray  # int8[B], turnFlags.enduring (Endure, G-65)
    ts_taken: jnp.ndarray  # int16[B,6], turnFlags damage taken and attackers (step._use_move)
    ts_dragged: jnp.ndarray  # int8[B], 1 + the slot a Roar dragged in this turn
    ts_move_failed: jnp.ndarray  # int8[B], moveFailFlags (the turn end's Uproar check)
    ts_used: jnp.ndarray  # int16, the paused action's moveCur
    ts_damaged: jnp.ndarray  # int8, its defender's physical / specialDamageTaken != 0
    ts_dealt: jnp.ndarray  # int16, its hit's HP drop (the thaw)
    ts_alive: jnp.ndarray  # int8[P], per party its bitmask of HP > 0 at the last faint check
    ts_power_mul: jnp.ndarray  # int8, the search's turn powerMul (replacement_power_mul)
    ts_switched: jnp.ndarray  # int8[B], the battler chose a switch this turn
    ts_pick_mul: jnp.ndarray  # int8, powerMul at the party list (20 after U-turn's Pursuit)
    ts_assurance: jnp.ndarray  # int8[B], turnFlags.assuranceDamageMask != 0 (per battler, B13-17)

    def replace_hp(self, side: int, slot: int, hp: int) -> BattleState:
        return self._replace(hp=self.hp.at[party(format_of(self), side), slot].set(hp))

    def with_stages(self, side: int, **stages: int) -> BattleState:
        """Set (not add) the active Pokemon's stages; keywords atk, def_, spa, spd,
        spe, acc, eva."""
        new = self.stages
        for key, value in stages.items():
            if key not in _STAGE_KEYWORDS:
                raise TypeError(f"unknown stage {key!r}; use one of {sorted(_STAGE_KEYWORDS)}")
            if not STAGE_MIN <= value <= STAGE_MAX:
                raise ValueError(f"stage {key}={value} outside [{STAGE_MIN}, {STAGE_MAX}]")
            new = new.at[side, _STAGE_KEYWORDS[key]].set(value)
        return self._replace(stages=new)

    def with_pp(self, side: int, slot: int, pp_list: Sequence[int]) -> BattleState:
        row = party(format_of(self), side)
        return self._replace(pp=self.pp.at[row, slot].set(jnp.asarray(pp_list, jnp.int8)))

    def with_status(self, side: int, slot: int, status: int, counter: int = 0) -> BattleState:
        row = party(format_of(self), side)
        return self._replace(
            status=self.status.at[row, slot].set(status),
            status_counter=self.status_counter.at[row, slot].set(counter),
        )

    def with_budget(self, k: int = 0, protected: Sequence[int] = ()) -> BattleState:
        """Set the faint budget ``k`` (0..``MAX_DEATH_BUDGET``) and the protected player
        slots: each must hold a living Pokemon (not empty, not fainted)."""
        mask = protected_mask(self.species[0], self.hp[0], k, protected)
        return self._replace(
            faint_budget=jnp.asarray(k, jnp.int8), protected=jnp.asarray(mask, jnp.int8)
        )


def record_seen_move(row, move):
    """``TrainerAI_RecordLastMove``'s list update (trainer_ai.c:2705-2716) on one battler's
    ``battlerMoves`` ``row`` (``ai_seen_moves[slot]``, move ids in first-free order): ``move``
    goes into the first empty entry unless the list already holds it or is full; MOVE_NONE
    matches the first empty entry and records nothing (G-68)."""
    row = jnp.asarray(row)
    move = jnp.asarray(move, row.dtype)
    free = row == 0
    add = ~jnp.any(row == move) & jnp.any(free) & (move != 0)
    first = jnp.arange(row.shape[0]) == jnp.argmax(free)
    return jnp.where(add & first, move, row)


def protected_mask(species, hp, k: int, protected: Sequence[int]) -> np.ndarray:
    """``int8[SLOTS]`` mask of the protected player slots, after ``with_budget``'s checks
    on the budget ``k`` and on ``species`` / ``hp`` (the player's per-slot arrays)."""
    if not 0 <= k <= MAX_DEATH_BUDGET:
        raise ValueError(f"faint budget {k} outside [0, {MAX_DEATH_BUDGET}]")
    mask = np.zeros(SLOTS, np.int8)
    for slot in protected:
        if not 0 <= slot < SLOTS:
            raise ValueError(f"protected slot {slot} outside [0, {SLOTS - 1}]")
        if int(species[slot]) == 0:
            raise ValueError(f"protected slot {slot} is empty")
        if int(hp[slot]) == 0:
            raise ValueError(f"protected slot {slot} has already fainted")
        mask[slot] = 1
    return mask


# Per-side fields of the active Pokemon that leave with it: the incoming battler is
# rebuilt from its party mon by ``BattleSystem_InitBattleMon`` (battle_lib.c:70), stat
# stages reset to neutral (:97-101; a Baton Pass keeps them and ``BATON_PASSED_FIELDS``). Hazards and screens are per side
# and stay. ``move_hit`` is cleared by ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:
# 2121) and ``BattleSystem_CleanupFaintedMon`` (:2199); ``fake_out_turn`` is set there
# instead (``step._entered``).
VOLATILE_FIELDS = (
    "confusion",
    "leech_seed",
    "substitute_hp",
    "taunt",
    "trapped",
    "trap_move",
    "flinched",
    "choice_lock",
    "last_move",
    "protect_count",
    "rollout_count",
    "charge_turn",
    "recharge",
    "pursuit_target",
    "water_sport",
    "move_hit",
    "foresight",
    "disabled_move",
    "disable_turns",
    "encore_move",
    "encore_slot",
    "encore_turns",
    "locked_move",
    "bide_turns",
    "bide_damage",
    "curled",
    "mud_sport",
    "aqua_ring",
    "infatuated",
    "destiny_bond",
    "uproar",
    "ingrain",
    "mean_look",
    "mean_look_target",
    "imprison",
    "imprisoned",
    "lock_on",
    "lock_on_target",
    "perish_song",
    "perish_song_turns",
    "yawn",
    "curse",
    "can_unburden",
    "minimized",
    "semi_invulnerable",
    "micle",
    "metronome_turns",
    "metronome_move",
    "focus_energy",
    "power_trick",
    "power_trick_desync",
    "magnet_rise",
    "magnet_rise_turns",
    "embargo",
)
# ``mimic_slots`` and the shadow (``shadow_moves`` / ``shadow_pp``) are not in ``VOLATILE_FIELDS``: a
# faint clears ``moveEffectsData`` too, but the fainted battler keeps its copy until its replacement
# enters (``step._restore_moves``, as ``_restore_ability``), which clears them.
# Baton Pass keeps these of ``VOLATILE_FIELDS`` (and the stat stages): BattleSystem_UpdateAfterSwitch
# keeps statusVolatile & VOLATILE_CONDITION_BATON_PASSED (battle_lib.c:2078; condition.h:72-76:
# confusion, Mean Look, Curse, Substitute's bit (``substitute_hp != 0``), Focus Energy (B12-4)),
# moveEffectsMask & MOVE_EFFECT_BATON_PASSED (:2079; moves.h:46-48: the Leech Seed recipient, Mud
# and Water Sport, Aqua Ring, Ingrain, Power Trick (B12-10), Magnet Rise (G-66); the rest from
# unsupported moves), and magnetRiseTurns (:2112), the
# substitute HP and the
# meanLookTarget (:2107, :2111); BattleSystem_InitBattleMon keeps the stat stages (:97-101).
# Rulings B6-3, B7-n.
BATON_PASSED_FIELDS = (
    "confusion",
    "leech_seed",
    "substitute_hp",
    "water_sport",
    "mud_sport",
    "aqua_ring",
    "ingrain",
    "mean_look",
    "mean_look_target",
    "lock_on",
    "lock_on_target",
    "perish_song",
    "perish_song_turns",
    "curse",
    "focus_energy",
    "power_trick",
    "magnet_rise",
    "magnet_rise_turns",
    "embargo",
)


# Shape key and storage dtype of every field, in BattleState order: per party slot [P, 6]
# (``_MON``), per battler [B], per side [2], per party [P], or a scalar (docs/kaizo/
# e2-mechanics-ledger.md "D1: change plan", field decisions).
# A shape key names its leading axes: "P" parties, "B" battlers, "S" sides; ``layout(fmt)``
# fills them in.
_MON, _BATTLER, _PARTY, _SIDE, _FIELD = ("P", SLOTS), ("B",), ("P",), ("S",), ()
_KEYS: dict[str, tuple[tuple[int | str, ...], type]] = {
    "species": (_MON, np.int16),
    "form": (_MON, np.int8),
    "level": (_MON, np.int8),
    "hp": (_MON, np.int16),
    "max_hp": (_MON, np.int16),
    "stats": ((*_MON, 5), np.int16),
    "types": ((*_MON, 2), np.int8),
    "ability": (_MON, np.int16),
    "ability_slot": (_MON, np.int8),
    "ability_form": (_MON, np.int8),
    "item": (_MON, np.int16),
    "item_consumed": (_MON, np.int16),
    "knocked_off": (_MON, np.int8),
    "moves": ((*_MON, MOVES), np.int16),
    "pp": ((*_MON, MOVES), np.int8),
    "pp_ups": ((*_MON, MOVES), np.int8),
    "status": (_MON, np.int8),
    "status_counter": (_MON, np.int8),
    "gender": (_MON, np.int8),
    "friendship": (_MON, np.int16),
    "ivs": ((*_MON, 6), np.int8),
    "active": (_BATTLER, np.int8),
    "stages": ((*_BATTLER, 8), np.int8),
    "confusion": (_BATTLER, np.int8),
    "leech_seed": (_BATTLER, np.int8),
    "substitute_hp": (_BATTLER, np.int16),
    "taunt": (_BATTLER, np.int8),
    "trapped": (_BATTLER, np.int8),
    "trap_move": (_BATTLER, np.int16),
    "flinched": (_BATTLER, np.int8),
    "choice_lock": (_BATTLER, np.int16),
    "last_move": (_BATTLER, np.int16),
    "protect_count": (_BATTLER, np.int8),
    "rollout_count": (_BATTLER, np.int8),
    "charge_turn": (_BATTLER, np.int8),
    "recharge": (_BATTLER, np.int8),
    "pursuit_target": (_BATTLER, np.int8),
    "water_sport": (_BATTLER, np.int8),
    "fake_out_turn": (_BATTLER, np.int16),
    "move_hit": (_BATTLER, np.int16),
    "foresight": (_BATTLER, np.int8),
    "disabled_move": (_BATTLER, np.int16),
    "disable_turns": (_BATTLER, np.int8),
    "encore_move": (_BATTLER, np.int16),
    "encore_slot": (_BATTLER, np.int8),
    "encore_turns": (_BATTLER, np.int8),
    "locked_move": (_BATTLER, np.int16),
    "bide_turns": (_BATTLER, np.int8),
    "bide_damage": (_BATTLER, np.int16),
    "curled": (_BATTLER, np.int8),
    "mud_sport": (_BATTLER, np.int8),
    "aqua_ring": (_BATTLER, np.int8),
    "infatuated": (_BATTLER, np.int8),
    "destiny_bond": (_BATTLER, np.int8),
    "uproar": (_BATTLER, np.int8),
    "curse": (_BATTLER, np.int8),
    "perish_song": (_BATTLER, np.int8),
    "perish_song_turns": (_BATTLER, np.int8),
    "yawn": (_BATTLER, np.int8),
    "lock_on": (_BATTLER, np.int8),
    "lock_on_target": (_BATTLER, np.int8),
    "ingrain": (_BATTLER, np.int8),
    "mean_look": (_BATTLER, np.int8),
    "mean_look_target": (_BATTLER, np.int8),
    "imprison": (_BATTLER, np.int8),
    "imprisoned": (_BATTLER, np.int8),
    "roosting": (_BATTLER, np.int8),
    "can_unburden": (_BATTLER, np.int8),
    "recycle_item": (_BATTLER, np.int16),
    "minimized": (_BATTLER, np.int8),
    "semi_invulnerable": (_BATTLER, np.int8),
    "micle": (_BATTLER, np.int8),
    "metronome_turns": (_BATTLER, np.int8),
    "metronome_move": (_BATTLER, np.int16),
    "focus_energy": (_BATTLER, np.int8),
    "power_trick": (_BATTLER, np.int8),
    "power_trick_desync": (_BATTLER, np.int8),
    "magnet_rise": (_BATTLER, np.int8),
    "magnet_rise_turns": (_BATTLER, np.int8),
    "mimic_slots": (_BATTLER, np.int8),
    "shadow_moves": ((*_BATTLER, MOVES), np.int16),
    "shadow_pp": ((*_BATTLER, MOVES), np.int8),
    "transformed": (_BATTLER, np.int8),
    "shadow_species": (_BATTLER, np.int16),
    "shadow_form": (_BATTLER, np.int8),
    "shadow_stats": ((*_BATTLER, 5), np.int16),
    "shadow_ivs": ((*_BATTLER, 6), np.int8),
    "shadow_ability_slot": (_BATTLER, np.int8),
    "shadow_ability_form": (_BATTLER, np.int8),
    "truant": (_BATTLER, np.int8),
    "chatot_cry": (_PARTY, np.int8),
    "embargo": (_BATTLER, np.int8),
    "spikes": (_SIDE, np.int8),
    "toxic_spikes": (_SIDE, np.int8),
    "stealth_rock": (_SIDE, np.int8),
    "reflect": (_SIDE, np.int8),
    "light_screen": (_SIDE, np.int8),
    "safeguard": (_SIDE, np.int8),
    "mist": (_SIDE, np.int8),
    "lucky_chant": (_SIDE, np.int8),
    "tailwind": (_SIDE, np.int8),
    "wish_timer": (_BATTLER, np.int8),
    "wish_hp": (_BATTLER, np.int16),
    "future_timer": (_BATTLER, np.int8),
    "future_damage": (_BATTLER, np.int16),
    "future_move": (_BATTLER, np.int16),
    "future_attacker": (_BATTLER, np.int8),
    "future_sight": (_SIDE, np.int8),
    "trainer_items": ((*_BATTLER, BAG_SLOTS), np.int16),
    "trainer_item_count": (_BATTLER, np.int8),
    "qc_roll": (_BATTLER, np.int8),
    "weather": (_FIELD, np.int8),
    "weather_timer": (_FIELD, np.int8),
    "trick_room": (_FIELD, np.int8),
    "gravity": (_FIELD, np.int8),
    "war_swaps": (_FIELD, np.int16),
    "entry_pending": (_BATTLER, np.int8),
    "move_prev": (_FIELD, np.int16),
    "terrain": (_FIELD, np.int8),
    "turn": (_FIELD, np.int16),
    "over": (_FIELD, np.int8),
    "winner": (_FIELD, np.int8),
    "player_faints": (_FIELD, np.int8),
    "ai_seen_moves": ((SLOTS, MOVES), np.int16),
    "ai_item_known": ((SLOTS,), np.int16),
    "ai_ability_known": ((SLOTS,), np.int16),
    "exp": ((SLOTS,), np.int32),
    "evs": ((*_MON, 6), np.int16),
    "nature": (_MON, np.int8),
    "exp_mask": (_FIELD, np.int8),
    "exp_boost": ((SLOTS,), np.int16),
    "pokerus": ((SLOTS,), np.int8),
    "friend_bonus": ((SLOTS,), np.int8),
    "learn_level": ((SLOTS,), np.int8),
    "learn_cursor": ((SLOTS,), np.int8),
    "learn_pin": (_FIELD, np.int8),
    "exp_refused": (_FIELD, np.int8),
    "deaths": ((SLOTS,), np.int8),
    "faint_budget": (_FIELD, np.int8),
    "protected": ((SLOTS,), np.int8),
    "replace_mul": (_FIELD, np.int8),
    "pivot_battler": (_FIELD, np.int8),
    "ts_cursor": (_FIELD, np.int8),
    "ts_acted": (_BATTLER, np.int8),
    "ts_order": (_BATTLER, np.int8),
    "ts_actions": (_BATTLER, np.int8),
    "ts_move": (_BATTLER, np.int16),
    "ts_struggling": (_BATTLER, np.int8),
    "ts_locked": (_BATTLER, np.int8),
    "ts_item": (_BATTLER, np.int16),
    "ts_custap": (_BATTLER, np.int8),
    "ts_claw": (_BATTLER, np.int8),
    "ts_pursued": (_FIELD, np.int8),
    "ts_protect_ok": (_BATTLER, np.int8),
    "ts_endure_ok": (_BATTLER, np.int8),
    "ts_taken": ((*_BATTLER, 6), np.int16),
    "ts_dragged": (_BATTLER, np.int8),
    "ts_move_failed": (_BATTLER, np.int8),
    "ts_used": (_FIELD, np.int16),
    "ts_damaged": (_FIELD, np.int8),
    "ts_dealt": (_FIELD, np.int16),
    "ts_alive": (_PARTY, np.int8),
    "ts_power_mul": (_FIELD, np.int8),
    "ts_switched": (_BATTLER, np.int8),
    "ts_pick_mul": (_FIELD, np.int8),
    "ts_assurance": (_BATTLER, np.int8),
}


def layout(fmt: Format) -> dict[str, tuple[tuple[int, ...], type]]:
    """Shape and storage dtype of every field of a ``fmt`` fight, in ``BattleState`` order."""
    dims = {"P": fmt.P, "B": fmt.B, "S": SIDES}
    return {
        name: (tuple(dims.get(d, d) for d in shape), dtype)  # type: ignore[arg-type]
        for name, (shape, dtype) in _KEYS.items()
    }


LAYOUT = layout(SINGLES)  # the singles shapes: what every caller outside the format work reads

# The turn scratch (``ts_*``, design rev. 2 section 1): 0 everywhere unless ``pivot_battler``.
TURN_SCRATCH = tuple(name for name in LAYOUT if name.startswith("ts_"))


def held_item(state: BattleState, side, slot=None) -> jnp.ndarray:
    """``int32``: ``Battler_HeldItem`` (battle_lib.c:5352-5363) of ``side``'s active battler, or
    of its party ``slot``: ``ITEM_NONE`` when its ``Battler_Ability`` is Klutz (:5354-5356; no
    supported mechanic makes ``Battler_Ability`` differ from the held ability for Klutz:
    Gastro Acid is unsupported, :3088-3106) or its ``embargoTurns`` is nonzero (:5358-5360),
    else its ``heldItem``. Every ``Battler_HeldItemEffect`` / ``Battler_HeldItemPower(...,
    ITEM_POWER_CHECK_ALL)`` / ``Battler_NaturalGift*`` read goes through it (:5438-5481); the
    reads of ``battleMons[b].heldItem`` itself (the speed-halving items, :1240-1246, Unburden's
    test, :1268-1272, Knock Off, Pluck, RemoveItem, Recycle, Trace's and Embargo's Griseous Orb,
    the party data of the exp payout) read ``state.item`` directly (ruling B13-29). Embargo is
    the active battler's: a bench ``slot`` (a newcomer read before it is in) gets none, which
    is what a plain switch gives it (``VOLATILE_FIELDS``); a Baton Pass newcomer is read after
    the pass, as the active battler."""
    side = jnp.asarray(side, jnp.int32)
    row = party(format_of(state), side)
    active = state.active[side]
    slot = active if slot is None else jnp.asarray(slot, jnp.int32)
    hidden = (state.ability[row, slot] == AbilityId.KLUTZ) | (
        (state.embargo[side] != 0) & (slot == active)
    )
    return jnp.where(hidden, 0, state.item[row, slot].astype(jnp.int32))


def held_items(state: BattleState) -> jnp.ndarray:
    """``int32[B]``: ``held_item`` of each battler."""
    return jnp.stack([held_item(state, side) for side in range(format_of(state).B)])


def field_index(state: BattleState, field: str, index):
    """``index`` (a battler, or ``(battler, slot, ...)``) re-indexed to the leading axis of the
    named field: the party of the battler for a ``[P, ...]`` field, its side for a ``[2]`` field,
    the battler itself for a ``[B, ...]`` field (docs/doubles-state-design.md section 5.2). For
    the generic field writers that take the field name as data. ``index`` leads with a battler,
    never an already mapped party or side: the map is not idempotent (TWO_TRAINERS: battler 3
    is party 2, and party 2 read as a battler is party 0)."""
    lead = _KEYS[field][0][:1]
    parts = index if isinstance(index, tuple) else (index,)
    if lead == ("P",):
        parts = (party(format_of(state), parts[0]), *parts[1:])
    elif lead == _SIDE:
        parts = (side_of(parts[0]), *parts[1:])
    return parts if isinstance(index, tuple) else parts[0]


if tuple(LAYOUT) != BattleState._fields:
    raise RuntimeError("LAYOUT must list every BattleState field, in order")


def spec_stats(tables: Tables, mon: PokemonSpec) -> np.ndarray:
    """``mon``'s six stats as ``make_state`` builds them: its stored ``stats``, else
    ``calc_stats`` of the ``stats_from`` form's base stats (``PokemonSpec`` docstring;
    ValueError: no such form, or ``stats`` not six values)."""
    personal_row(tables, mon.species, mon.form)  # ValueError: no such form
    row = personal_row(tables, mon.species, mon.stats_from)
    if mon.stats is not None:
        if len(mon.stats) != 6:
            raise ValueError(f"species {mon.species}: stats needs 6 values, got {len(mon.stats)}")
        return np.asarray(mon.stats)
    return calc_stats(
        np.asarray(tables.species_base_stats[row]),
        np.asarray(mon.ivs),
        np.asarray(mon.evs),
        mon.level,
        np.asarray(tables.nature_mods[mon.nature]),
        species=mon.species,
    )


def spec_exp(tables: Tables, mon: PokemonSpec) -> int:
    """``mon``'s experience as ``make_state`` stores it: its ``exp``, else the base of its
    level (``exp_table[rate][level]``, the rate by species as ``Pokemon_GetSpeciesBaseExpAt``
    reads it, pokemon.c:2327-2330). ``ValueError`` when ``exp`` is not of ``mon``'s level:
    the level of ``x`` is the last ``L`` in 1..100 with ``table[L] <= x``
    (``SpeciesData_GetLevelAt``, pokemon.c:2379-2392), so ``exp`` must lie in
    ``[table[level], table[level + 1])``, or be at least ``table[100]`` at level 100."""
    rate = int(np.asarray(tables.species_exp_rate)[mon.species])
    table = np.asarray(tables.exp_table)[rate]
    if mon.exp is None:
        return int(table[mon.level])
    top = len(table) - 1
    high = None if mon.level >= top else int(table[mon.level + 1])
    if mon.exp < table[mon.level] or (high is not None and mon.exp >= high):
        span = f"[{int(table[mon.level])}, {high})" if high is not None else f">= {int(table[top])}"
        raise ValueError(
            f"species {mon.species}: exp {mon.exp} is not of level {mon.level} (needs {span})"
        )
    return int(mon.exp)


def _fill_slot(a: dict[str, np.ndarray], tables: Tables, side: int, slot: int, mon: PokemonSpec):
    """Write ``mon`` into party ``side``'s slot ``slot`` (the leading axis of every ``[P, 6]`` field)."""
    species = mon.species
    row = personal_row(tables, species, mon.form)  # ValueError: no such form
    stats = spec_stats(tables, mon)
    max_hp = int(stats[Stat.HP])
    hp = max_hp if mon.current_hp is None else mon.current_hp
    if not 0 <= hp <= max_hp:
        raise ValueError(f"species {species}: current_hp {hp} outside [0, {max_hp}]")
    move_pp = np.asarray(tables.move_pp)
    a["species"][side, slot] = species
    a["form"][side, slot] = mon.form
    a["level"][side, slot] = mon.level
    a["hp"][side, slot] = hp
    a["max_hp"][side, slot] = max_hp
    a["stats"][side, slot] = stats[Stat.ATK : Stat.SPE + 1]
    a["types"][side, slot] = np.asarray(tables.species_types[row])
    ability_row = personal_row(tables, species, mon.ability_from)
    a["ability"][side, slot] = int(tables.species_abilities[ability_row, mon.ability_slot])
    a["ability_slot"][side, slot] = mon.ability_slot
    a["ability_form"][side, slot] = mon.ability_from
    a["item"][side, slot] = mon.item
    a["moves"][side, slot] = mon.moves
    a["pp"][side, slot] = [move_pp[m] if m else 0 for m in mon.moves] if mon.pp is None else mon.pp
    a["pp_ups"][side, slot] = mon.pp_ups
    a["status"][side, slot] = mon.status
    a["status_counter"][side, slot] = mon.status_counter
    a["gender"][side, slot] = mon.gender
    a["friendship"][side, slot] = mon.friendship
    a["ivs"][side, slot] = mon.ivs
    a["evs"][side, slot] = mon.evs
    a["nature"][side, slot] = mon.nature
    exp = spec_exp(tables, mon)  # ValueError: exp not of the level
    if side == 0:  # the opponent's exp is never read (its Pokemon gain none)
        a["exp"][slot] = exp
        a["exp_boost"][slot] = mon.exp_boost
        a["pokerus"][slot] = mon.pokerus
        a["friend_bonus"][slot] = mon.friend_bonus


def initial_arrays(
    fmt: Format,
    tables: Tables,
    parties: Sequence[list[PokemonSpec]],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    leads: Sequence[int] | None = None,
    terrain: int = TERRAIN_UNKNOWN,
    gravity: int = 0,
    chatot_cry: int = 0,
    opp2_items: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Every ``layout(fmt)`` field of the turn-0 state, before the switch-in hooks, as numpy
    (``make_state`` is this plus ``hooks.battle_start`` for singles): ``parties`` holds the
    ``fmt.P`` parties (party 0 the player's, 1 the opponent's), stats via ``calc_stats``, empty
    slots all zero (``species = 0, hp = 0``), battler ``b``'s lead in party slot ``leads[b]``
    (default: the battlers of one party take slots 0, 1, ... in battler order),
    no volatiles, hazards or screens, and
    ``weather = start_condition`` with a permanent timer (0), and ``terrain`` (a decomp
    ``TERRAIN_*`` id, ``Trainer.terrain``; ``TERRAIN_UNKNOWN`` by default) and ``gravity``
    (``ids.GRAVITY_TURNS`` for a trainer whose area condition is Gravity, Kaizo's sub_seq 297, 0
    otherwise; the grounding loop it runs at battle start is a no-op, G-67). The leads'
    switch-in effects then run (``hooks.battle_start``: each switch-in state over both
    leads, as ``BattleSystem_TriggerEffectOnSwitch`` runs them).

    ``opp_items`` is the trainer's bag (``Trainer.items``, decomp item ids). As
    ``BattleControllerPlayer_InitAI`` (battle_controller_player.c:4796-4811) does, its
    non-empty items fill ``trainer_items[1]`` from slot 0 in order and their number is
    ``trainer_item_count[1]``; the player's battlers' bags stay empty (only ``battler &
    BATTLER_THEM`` is filled, :4803). With four battlers, battler 3's bag is ``opp2_items``,
    the second opposing trainer's (``trainers[3]``; ``TWO_TRAINERS`` / ``MULTI`` must give
    it), by default ``opp_items`` again: a plain double loads the one trainer into both
    enemy slots (encounter.c:736, :751-752; trainer_data.c:31-37), so each holds a copy
    (ruling D1-5). The leads'
    ``fake_out_turn`` is 0: the battle-start send-out does not run
    ``BattleSystem_UpdateAfterSwitch``, and ``BattleContext`` starts zeroed
    (battle_controller_player.c:180-184).

    Experience: ``exp`` is ``spec_exp`` (``ValueError`` when a spec's ``exp`` is not of its
    level), the player's ``exp_boost`` / ``pokerus`` / ``friend_bonus`` come from its specs,
    ``evs`` and ``nature`` from both sides', and ``exp_mask = 1 << leads[0]``: the
    battle-start ``PokemonSlideIn`` of the enemy (clear, then flag) and of the player (flag)
    (subscript_start_encounter.s:84, :91; ``BtlCmd_PokemonSlideIn``, battle_script.c:465-466,
    :474-475) leave the player lead's bit (``BattleSystem_FlagBattlerExpGain``,
    battle_lib.c:1461-1479). The learn fields and ``exp_refused`` start at 0."""
    if len(parties) != fmt.P:
        raise ValueError(f"a {fmt.B}-battler fight has {fmt.P} parties, got {len(parties)}")
    if leads is None:
        leads = tuple(
            sum(1 for c in range(b) if fmt.party_of[c] == fmt.party_of[b]) for b in range(fmt.B)
        )
    a = {name: np.zeros(shape, dtype) for name, (shape, dtype) in layout(fmt).items()}
    for side, members in enumerate(parties):
        if not 1 <= len(members) <= SLOTS:
            raise ValueError(f"side {side}: a party has 1-{SLOTS} Pokemon, got {len(members)}")
        for slot, mon in enumerate(members):
            _fill_slot(a, tables, side, slot, mon)
        own = [b for b in range(fmt.B) if fmt.party_of[b] == side]
        for b in own:
            if not 0 <= leads[b] < len(members):
                raise ValueError(
                    f"side {side}: lead slot {leads[b]} is not in a party of {len(members)}"
                )
            a["active"][b] = leads[b]
        if len({leads[b] for b in own}) != len(own):
            raise ValueError(
                f"side {side}: two battlers lead from one slot ({[leads[b] for b in own]})"
            )
    a["exp_mask"][...] = 1 << leads[0]
    bags = {1: opp_items}
    if fmt.B == 4:
        if opp2_items is None and fmt.P > 2:
            raise ValueError("a fight of two opposing trainers needs the second one's bag")
        bags[3] = opp_items if opp2_items is None else opp2_items
    for b, items in bags.items():
        bag = [item for item in items if item]
        if len(bag) > BAG_SLOTS:
            raise ValueError(f"a trainer's bag holds at most {BAG_SLOTS} items, got {len(bag)}")
        a["trainer_items"][b, : len(bag)] = bag
        a["trainer_item_count"][b] = len(bag)
    a["weather"][...] = Weather(start_condition)
    if not TERRAIN_UNKNOWN <= terrain < len(TERRAIN_NAMES):
        raise ValueError(f"terrain {terrain} is neither a TERRAIN_* id nor TERRAIN_UNKNOWN")
    a["terrain"][...] = terrain
    if not 0 <= gravity <= GRAVITY_TURNS:
        raise ValueError(f"gravity {gravity} is not 0-{GRAVITY_TURNS} turns")
    a["gravity"][...] = gravity
    # BattleSystem.chatotCries (battle_system.c:915-924): every enemy cry is invalid and gives 0
    # (field_battle_data_transfer.c:97, chatot_cry.c:13-25, sound_chatot.c:193-200); the player's
    # is the save's (:260, :319, :385), ``chatot_cry``: 0, 1 or 2 (``saves.chatot_parameter``).
    if not 0 <= chatot_cry < len(CHATTER_CHANCES):
        raise ValueError(f"chatot_cry {chatot_cry} is not a Chatter parameter 0-2")
    a["chatot_cry"][0] = chatot_cry
    return a


def make_state(
    tables: Tables,
    player: list[PokemonSpec],
    opp: list[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    leads: tuple[int, int] = (0, 0),
    terrain: int = TERRAIN_UNKNOWN,
    gravity: int = 0,
    chatot_cry: int = 0,
) -> BattleState:
    """Build the turn-0 state (``initial_arrays``), then run the leads' switch-in
    effects (``hooks.battle_start``). Singles: ``initial_arrays`` with ``SINGLES``."""
    from nuzline.engine.hooks import battle_start  # hooks imports this module

    a = initial_arrays(
        SINGLES,
        tables,
        [player, opp],
        start_condition,
        opp_items,
        leads,
        terrain,
        gravity,
        chatot_cry,
    )
    state = BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})
    return battle_start(tables, state)
