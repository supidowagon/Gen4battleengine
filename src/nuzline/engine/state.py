"""The battle state pytree and its construction from two parties.

``BattleState`` is unbatched: side axis 0 is the player, 1 the opponent; slot axis
is party order. Every field is ``int8``/``int16`` storage (spec §4.2), except ``exp``:
``int32``, because the level-100 experience of most growth rates exceeds ``int16`` (K3d-exp
ruling D1). The field list is ticked off against the decomp's battle structs in
``STATE.md``.

Encodings:

* ``species``/``moves``/``item``/``item_consumed``/``trainer_items``/``trap_move``/
  ``choice_lock``/``last_move``/``move_hit`` hold decomp ids (0 = none); ``ability``
  holds an ``ids.AbilityId`` (from ``tables.species_abilities``) and ``ability_slot``
  the species slot (0/1) it came from, so ``tables.species_abilities_decomp[row,
  ability_slot]`` is the decomp ability id, ``row = tables.species_form_row[species,
  form]`` the personal row of the Pokemon's form (base stats, types and abilities are
  read by row); ``status`` an ``ids.Status``.
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
* ``trainer_item_count`` is ``aiContext.trainerItemCounts``: the bag's size at battle
  start, never decremented.
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
from nuzline.engine.ids import Stat, Weather
from nuzline.engine.stats import calc_stats

SIDES, SLOTS, MOVES = 2, 6, 4
MAX_DEATH_BUDGET = 3  # the ladder's range: k = 0..3 deaths allowed (spec §4)
BAG_SLOTS = 4  # MAX_TRAINER_ITEMS: trainer_items[side] holds the AI's bag
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
    # per side x slot [2,6]
    species: jnp.ndarray  # int16
    form: jnp.ndarray  # int8
    level: jnp.ndarray  # int8
    hp: jnp.ndarray  # int16
    max_hp: jnp.ndarray  # int16
    stats: jnp.ndarray  # int16[2,6,5]: Atk, Def, SpA, SpD, Spe
    types: jnp.ndarray  # int8[2,6,2]
    ability: jnp.ndarray  # int16, ids.AbilityId
    ability_slot: jnp.ndarray  # int8, species ability slot 0/1
    item: jnp.ndarray  # int16
    item_consumed: jnp.ndarray  # int16, id of the last item consumed
    # int8, the slot's bit of its side's knockedOffItemsMask (battle_script.c:6627), B10-6
    knocked_off: jnp.ndarray
    moves: jnp.ndarray  # int16[2,6,4]
    pp: jnp.ndarray  # int8[2,6,4]
    status: jnp.ndarray  # int8, ids.Status
    status_counter: jnp.ndarray  # int8, sleep turns / toxic counter
    gender: jnp.ndarray  # int8, 0 male, 1 female, 2 none
    friendship: jnp.ndarray  # int16
    ivs: jnp.ndarray  # int8[2,6,6], ids.Stat order
    # per side, active Pokemon [2]
    active: jnp.ndarray  # int8, party slot
    stages: jnp.ndarray  # int8[2,8], index ids.Stat, 0 = neutral
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
    infatuated: jnp.ndarray  # int8, VOLATILE_CONDITION_ATTRACT (source: the foe's active)
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
    minimized: jnp.ndarray  # int8, MOVE_EFFECT_MINIMIZE in moveEffectsMask
    semi_invulnerable: jnp.ndarray  # int8, ids.SemiKind: MOVE_EFFECT_SEMI_INVULNERABLE's bit
    focus_energy: jnp.ndarray  # int8, VOLATILE_CONDITION_FOCUS_ENERGY (B12-4)
    power_trick: jnp.ndarray  # int8, MOVE_EFFECT_POWER_TRICK (B12-10)
    power_trick_desync: jnp.ndarray  # int8, the Attack / Defense swap XOR power_trick (B12-10)
    # per side [2]
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
    trainer_items: jnp.ndarray  # int16[2,4], bag item ids (aiContext.trainerItems)
    trainer_item_count: jnp.ndarray  # int8, bag size at battle start (trainerItemCounts)
    qc_roll: jnp.ndarray  # int8, speedRand % 5 == 0 (Quick Claw), hidden, see the docstring
    # field (scalars)
    weather: jnp.ndarray  # int8, ids.Weather
    weather_timer: jnp.ndarray  # int8, 0 = permanent
    trick_room: jnp.ndarray  # int8
    gravity: jnp.ndarray  # int8
    war_swaps: jnp.ndarray  # int16, a weather war's open monSpeedOrder (docstring), 0 = none
    entry_pending: jnp.ndarray  # int8[2], 1: a list newcomer's switch-in states wait (docstring)
    move_prev: jnp.ndarray  # int16, BattleContext.movePrev: the last move used in the battle
    terrain: jnp.ndarray  # int8, decomp TERRAIN_* id, -1 unknown (constant per battle)
    turn: jnp.ndarray  # int16
    over: jnp.ndarray  # int8
    winner: jnp.ndarray  # int8
    player_faints: jnp.ndarray  # int8
    # AI knowledge of the player
    ai_seen_moves: jnp.ndarray  # int8[6,4], player slot x move slot
    ai_item_known: jnp.ndarray  # int16[6], battlerHeldItems: the item id a message named, 0 none
    ai_ability_known: jnp.ndarray  # int8[6]
    # experience (K3d-exp, design §2.1; see the module docstring)
    exp: jnp.ndarray  # int32[6], player party MON_DATA_EXPERIENCE
    evs: jnp.ndarray  # int16[2,6,6], ids.Stat order
    nature: jnp.ndarray  # int8[2,6]
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
    ts_acted: jnp.ndarray  # int8[2], the battler's action is over (Battler_MovedThisTurn)
    ts_order: jnp.ndarray  # int8[2], battlerActionOrder
    ts_actions: jnp.ndarray  # int8[2], the turn's actions as chosen
    ts_move: jnp.ndarray  # int16[2], each side's move fixed at selection (selected_move)
    ts_struggling: jnp.ndarray  # int8[2]
    ts_locked: jnp.ndarray  # int8[2], locked at selection (no PP)
    ts_item: jnp.ndarray  # int16[2], the bag item picked
    ts_custap: jnp.ndarray  # int8[2], moveEffectsData.custapBerry from the turn order
    ts_claw: jnp.ndarray  # int8[2], moveEffectsData.quickClaw from the turn order (B10-6)
    ts_pursued: jnp.ndarray  # int8, 1 + the battler whose action a Pursuit ended, 0 = none
    ts_protect_ok: jnp.ndarray  # int8[2], turnFlags.protecting
    ts_taken: jnp.ndarray  # int16[2,6], turnFlags damage taken and attackers (step._use_move)
    ts_dragged: jnp.ndarray  # int8[2], 1 + the slot a Roar dragged in this turn
    ts_move_failed: jnp.ndarray  # int8[2], moveFailFlags (the turn end's Uproar check)
    ts_used: jnp.ndarray  # int16, the paused action's moveCur
    ts_damaged: jnp.ndarray  # int8, its defender's physical / specialDamageTaken != 0
    ts_dealt: jnp.ndarray  # int16, its hit's HP drop (the thaw)
    ts_alive: jnp.ndarray  # int8[2], per side the party bitmask of HP > 0 at the last faint check
    ts_power_mul: jnp.ndarray  # int8, the search's turn powerMul (replacement_power_mul)
    ts_switched: jnp.ndarray  # int8[2], the battler chose a switch this turn
    ts_pick_mul: jnp.ndarray  # int8, powerMul at the party list (20 after U-turn's Pursuit)
    ts_assurance: jnp.ndarray  # int8[2], turnFlags.assuranceDamageMask != 0 (per battler, B13-17)

    def replace_hp(self, side: int, slot: int, hp: int) -> BattleState:
        return self._replace(hp=self.hp.at[side, slot].set(hp))

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
        return self._replace(pp=self.pp.at[side, slot].set(jnp.asarray(pp_list, jnp.int8)))

    def with_status(self, side: int, slot: int, status: int, counter: int = 0) -> BattleState:
        return self._replace(
            status=self.status.at[side, slot].set(status),
            status_counter=self.status_counter.at[side, slot].set(counter),
        )

    def with_budget(self, k: int = 0, protected: Sequence[int] = ()) -> BattleState:
        """Set the faint budget ``k`` (0..``MAX_DEATH_BUDGET``) and the protected player
        slots: each must hold a living Pokemon (not empty, not fainted)."""
        mask = protected_mask(self.species[0], self.hp[0], k, protected)
        return self._replace(
            faint_budget=jnp.asarray(k, jnp.int8), protected=jnp.asarray(mask, jnp.int8)
        )


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
    "focus_energy",
    "power_trick",
    "power_trick_desync",
)
# Baton Pass keeps these of ``VOLATILE_FIELDS`` (and the stat stages): BattleSystem_UpdateAfterSwitch
# keeps statusVolatile & VOLATILE_CONDITION_BATON_PASSED (battle_lib.c:2078; condition.h:72-76:
# confusion, Mean Look, Curse, Substitute's bit (``substitute_hp != 0``), Focus Energy (B12-4)),
# moveEffectsMask & MOVE_EFFECT_BATON_PASSED (:2079; moves.h:46-48: the Leech Seed recipient, Mud
# and Water Sport, Aqua Ring, Ingrain, Power Trick (B12-10); the rest from unsupported moves), the
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
)


# Shape and storage dtype of every field, in BattleState order.
_SLOT, _SIDE, _FIELD = (SIDES, SLOTS), (SIDES,), ()
LAYOUT: dict[str, tuple[tuple[int, ...], type]] = {
    "species": (_SLOT, np.int16),
    "form": (_SLOT, np.int8),
    "level": (_SLOT, np.int8),
    "hp": (_SLOT, np.int16),
    "max_hp": (_SLOT, np.int16),
    "stats": ((*_SLOT, 5), np.int16),
    "types": ((*_SLOT, 2), np.int8),
    "ability": (_SLOT, np.int16),
    "ability_slot": (_SLOT, np.int8),
    "item": (_SLOT, np.int16),
    "item_consumed": (_SLOT, np.int16),
    "knocked_off": (_SLOT, np.int8),
    "moves": ((*_SLOT, MOVES), np.int16),
    "pp": ((*_SLOT, MOVES), np.int8),
    "status": (_SLOT, np.int8),
    "status_counter": (_SLOT, np.int8),
    "gender": (_SLOT, np.int8),
    "friendship": (_SLOT, np.int16),
    "ivs": ((*_SLOT, 6), np.int8),
    "active": (_SIDE, np.int8),
    "stages": ((SIDES, 8), np.int8),
    "confusion": (_SIDE, np.int8),
    "leech_seed": (_SIDE, np.int8),
    "substitute_hp": (_SIDE, np.int16),
    "taunt": (_SIDE, np.int8),
    "trapped": (_SIDE, np.int8),
    "trap_move": (_SIDE, np.int16),
    "flinched": (_SIDE, np.int8),
    "choice_lock": (_SIDE, np.int16),
    "last_move": (_SIDE, np.int16),
    "protect_count": (_SIDE, np.int8),
    "rollout_count": (_SIDE, np.int8),
    "charge_turn": (_SIDE, np.int8),
    "recharge": (_SIDE, np.int8),
    "pursuit_target": (_SIDE, np.int8),
    "water_sport": (_SIDE, np.int8),
    "fake_out_turn": (_SIDE, np.int16),
    "move_hit": (_SIDE, np.int16),
    "foresight": (_SIDE, np.int8),
    "disabled_move": (_SIDE, np.int16),
    "disable_turns": (_SIDE, np.int8),
    "encore_move": (_SIDE, np.int16),
    "encore_slot": (_SIDE, np.int8),
    "encore_turns": (_SIDE, np.int8),
    "locked_move": (_SIDE, np.int16),
    "bide_turns": (_SIDE, np.int8),
    "bide_damage": (_SIDE, np.int16),
    "curled": (_SIDE, np.int8),
    "mud_sport": (_SIDE, np.int8),
    "aqua_ring": (_SIDE, np.int8),
    "infatuated": (_SIDE, np.int8),
    "destiny_bond": (_SIDE, np.int8),
    "uproar": (_SIDE, np.int8),
    "curse": (_SIDE, np.int8),
    "perish_song": (_SIDE, np.int8),
    "perish_song_turns": (_SIDE, np.int8),
    "yawn": (_SIDE, np.int8),
    "lock_on": (_SIDE, np.int8),
    "lock_on_target": (_SIDE, np.int8),
    "ingrain": (_SIDE, np.int8),
    "mean_look": (_SIDE, np.int8),
    "mean_look_target": (_SIDE, np.int8),
    "imprison": (_SIDE, np.int8),
    "imprisoned": (_SIDE, np.int8),
    "roosting": (_SIDE, np.int8),
    "can_unburden": (_SIDE, np.int8),
    "minimized": (_SIDE, np.int8),
    "semi_invulnerable": (_SIDE, np.int8),
    "focus_energy": (_SIDE, np.int8),
    "power_trick": (_SIDE, np.int8),
    "power_trick_desync": (_SIDE, np.int8),
    "spikes": (_SIDE, np.int8),
    "toxic_spikes": (_SIDE, np.int8),
    "stealth_rock": (_SIDE, np.int8),
    "reflect": (_SIDE, np.int8),
    "light_screen": (_SIDE, np.int8),
    "safeguard": (_SIDE, np.int8),
    "mist": (_SIDE, np.int8),
    "lucky_chant": (_SIDE, np.int8),
    "tailwind": (_SIDE, np.int8),
    "wish_timer": (_SIDE, np.int8),
    "wish_hp": (_SIDE, np.int16),
    "future_timer": (_SIDE, np.int8),
    "future_damage": (_SIDE, np.int16),
    "future_move": (_SIDE, np.int16),
    "future_attacker": (_SIDE, np.int8),
    "future_sight": (_SIDE, np.int8),
    "trainer_items": ((SIDES, BAG_SLOTS), np.int16),
    "trainer_item_count": (_SIDE, np.int8),
    "qc_roll": (_SIDE, np.int8),
    "weather": (_FIELD, np.int8),
    "weather_timer": (_FIELD, np.int8),
    "trick_room": (_FIELD, np.int8),
    "gravity": (_FIELD, np.int8),
    "war_swaps": (_FIELD, np.int16),
    "entry_pending": (_SIDE, np.int8),
    "move_prev": (_FIELD, np.int16),
    "terrain": (_FIELD, np.int8),
    "turn": (_FIELD, np.int16),
    "over": (_FIELD, np.int8),
    "winner": (_FIELD, np.int8),
    "player_faints": (_FIELD, np.int8),
    "ai_seen_moves": ((SLOTS, MOVES), np.int8),
    "ai_item_known": ((SLOTS,), np.int16),
    "ai_ability_known": ((SLOTS,), np.int8),
    "exp": ((SLOTS,), np.int32),
    "evs": ((*_SLOT, 6), np.int16),
    "nature": (_SLOT, np.int8),
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
    "ts_acted": (_SIDE, np.int8),
    "ts_order": (_SIDE, np.int8),
    "ts_actions": (_SIDE, np.int8),
    "ts_move": (_SIDE, np.int16),
    "ts_struggling": (_SIDE, np.int8),
    "ts_locked": (_SIDE, np.int8),
    "ts_item": (_SIDE, np.int16),
    "ts_custap": (_SIDE, np.int8),
    "ts_claw": (_SIDE, np.int8),
    "ts_pursued": (_FIELD, np.int8),
    "ts_protect_ok": (_SIDE, np.int8),
    "ts_taken": ((SIDES, 6), np.int16),
    "ts_dragged": (_SIDE, np.int8),
    "ts_move_failed": (_SIDE, np.int8),
    "ts_used": (_FIELD, np.int16),
    "ts_damaged": (_FIELD, np.int8),
    "ts_dealt": (_FIELD, np.int16),
    "ts_alive": (_SIDE, np.int8),
    "ts_power_mul": (_FIELD, np.int8),
    "ts_switched": (_SIDE, np.int8),
    "ts_pick_mul": (_FIELD, np.int8),
    "ts_assurance": (_SIDE, np.int8),
}
# The turn scratch (``ts_*``, design rev. 2 section 1): 0 everywhere unless ``pivot_battler``.
TURN_SCRATCH = tuple(name for name in LAYOUT if name.startswith("ts_"))
if tuple(LAYOUT) != BattleState._fields:
    raise RuntimeError("LAYOUT must list every BattleState field, in order")


def spec_stats(tables: Tables, mon: PokemonSpec) -> np.ndarray:
    """``mon``'s six stats as ``make_state`` builds them: its stored ``stats``, else
    ``calc_stats`` (ValueError: no such form, or ``stats`` not six values)."""
    row = personal_row(tables, mon.species, mon.form)  # ValueError: no such form
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
    a["ability"][side, slot] = int(tables.species_abilities[row, mon.ability_slot])
    a["ability_slot"][side, slot] = mon.ability_slot
    a["item"][side, slot] = mon.item
    a["moves"][side, slot] = mon.moves
    a["pp"][side, slot] = [move_pp[m] if m else 0 for m in mon.moves] if mon.pp is None else mon.pp
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
    tables: Tables,
    player: list[PokemonSpec],
    opp: list[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    leads: tuple[int, int] = (0, 0),
    terrain: int = TERRAIN_UNKNOWN,
) -> dict[str, np.ndarray]:
    """Every ``LAYOUT`` field of the turn-0 state, before the switch-in hooks, as numpy
    (``make_state`` is this plus ``hooks.battle_start``): stats via ``calc_stats``, empty slots all zero
    (``species = 0, hp = 0``), the leads in party slots ``leads`` (player, opponent),
    no volatiles, hazards or screens, and
    ``weather = start_condition`` with a permanent timer (0), and ``terrain`` (a decomp
    ``TERRAIN_*`` id, ``Trainer.terrain``; ``TERRAIN_UNKNOWN`` by default). The leads'
    switch-in effects then run (``hooks.battle_start``: each switch-in state over both
    leads, as ``BattleSystem_TriggerEffectOnSwitch`` runs them).

    ``opp_items`` is the trainer's bag (``Trainer.items``, decomp item ids). As
    ``BattleControllerPlayer_InitAI`` (battle_controller_player.c:4796) does, its
    non-empty items fill ``trainer_items[1]`` from slot 0 in order and their number is
    ``trainer_item_count[1]``; the player's bag stays empty. The leads'
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
    a = {name: np.zeros(shape, dtype) for name, (shape, dtype) in LAYOUT.items()}
    for side, party in enumerate((player, opp)):
        if not 1 <= len(party) <= SLOTS:
            raise ValueError(f"side {side}: a party has 1-{SLOTS} Pokemon, got {len(party)}")
        for slot, mon in enumerate(party):
            _fill_slot(a, tables, side, slot, mon)
        lead = leads[side]
        if not 0 <= lead < len(party):
            raise ValueError(f"side {side}: lead slot {lead} is not in a party of {len(party)}")
        a["active"][side] = lead
    a["exp_mask"][...] = 1 << leads[0]
    bag = [item for item in opp_items if item]
    if len(bag) > BAG_SLOTS:
        raise ValueError(f"a trainer's bag holds at most {BAG_SLOTS} items, got {len(bag)}")
    a["trainer_items"][1, : len(bag)] = bag
    a["trainer_item_count"][1] = len(bag)
    a["weather"][...] = Weather(start_condition)
    if not TERRAIN_UNKNOWN <= terrain < len(TERRAIN_NAMES):
        raise ValueError(f"terrain {terrain} is neither a TERRAIN_* id nor TERRAIN_UNKNOWN")
    a["terrain"][...] = terrain
    return a


def make_state(
    tables: Tables,
    player: list[PokemonSpec],
    opp: list[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    leads: tuple[int, int] = (0, 0),
    terrain: int = TERRAIN_UNKNOWN,
) -> BattleState:
    """Build the turn-0 state (``initial_arrays``), then run the leads' switch-in
    effects (``hooks.battle_start``)."""
    from nuzline.engine.hooks import battle_start  # hooks imports this module

    a = initial_arrays(tables, player, opp, start_condition, opp_items, leads, terrain)
    state = BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})
    return battle_start(tables, state)
