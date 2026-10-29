"""Ability and held-item hooks at fixed phases (pure ``jnp``, jit/vmap-able, masked).

* ``on_switch_in``: what follows a Pokemon entering the field. The switch script
  (``subscript_switch_pokemon.s``) runs the entry hazards
  (``subscript_hazards_check.s``; Stealth Rock is the only M1 hazard), and the
  controller then runs ``BattleSystem_TriggerEffectOnSwitch`` (battle_lib.c:3649,
  from ``BattleControllerPlayer_MoveEnd``, battle_controller_player.c:3973; at the
  start of the battle from ``ShowBattleMon``, :256). Of its states M1 reaches
  ``SWITCH_IN_CHECK_STATE_INTIMIDATE`` (:3795) and
  ``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS`` (:4151).
* ``after_damage_items``: the HP branch of ``BattleSystem_TriggerHeldItem``
  (battle_lib.c:4576) and ``BattleSystem_TriggerHeldItemOnStatus`` (:4967), which
  are the same for Oran and Sitrus Berries. ``step`` runs it after every hit of a
  move (``AFTER_MOVE_EFFECT_ATTACKER_ITEM`` / ``_DEFENDER_ITEM``,
  battle_controller_player.c:3684, :3691; the after-move effects run per hit before
  ``BATTLE_CONTROL_LOOP_MULTI_HIT``, :3769), after a switch-in, and at the end of
  the turn (``MON_COND_CHECK_STATE_USE_ITEM``, :1362, and
  ``MON_COND_CHECK_STATE_ITEM_CONDITION``, :1699).

Intimidate (``subscript_intimidate.s``): every living foe that is not behind a
Substitute gets ``ATTACK_DOWN_1_STAGE`` as an ability side effect, through
``BtlCmd_ChangeStatStage``; in M1 nothing blocks it (Clear Body, Hyper Cutter,
White Smoke and Mist are unsupported), and Simple is applied where the stage is
used (``damage.simple_stage``). The decomp runs the switch-in checks over
``monSpeedOrder``; in singles each Intimidate only lowers the other side's Attack,
so the order does not change the result and ``make_state`` runs the player's lead
first.

Oran Berry (``HOLD_EFFECT_HP_RESTORE``) heals ``itemPower`` (10) and Sitrus Berry
(``HOLD_EFFECT_HP_PCT_RESTORE``) ``BattleSystem_Divide(maxHP * itemPower, 100)``
(itemPower 25), both when ``0 < curHP <= maxHP / 2``.
``subscript_held_item_hp_restore.s`` heals through ``BATTLE_SUBSCRIPT_UPDATE_HP``
(capped at max HP) and ``BATTLE_SUBSCRIPT_PLUCK_CHECK`` removes the item
(``BtlCmd_RemoveItem``, battle_script.c:8972, which keeps it in ``recycleItem``:
``item_consumed`` here). Klutz, Embargo and Gluttony are unsupported.

What the AI learns (``REVEALS_ON``, ``reveal_ability``): the AI knows the player's
ability (``BattleState.ai_ability_known``) once ``BattleAI_SetAbility``
(battle_script.c:12106) has run for it. That happens when a script message tags the
battler's ability (``BattleMessage_AbilityTag``, :11798, from ``PrintMessage`` /
``BufferMessage``) or a script writes ``BATTLEMON_ABILITY`` (:3074, :3403; no M1 move
does). A message the C code builds in ``msgBuffer`` itself carries the ability id and
reveals nothing.

Abilities handled elsewhere: Blaze, Torrent, Overgrow, Rivalry, Guts, Unaware,
Simple and Mold Breaker in ``damage.py``; Keen Eye, Inner Focus, Rock Head,
Static and Poison Point in ``effects.py``; Steadfast and Natural Cure in
``step.py``. Sturdy only blocks OHKO moves in Gen 4, and none is in M1, so it has
no hook.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.effects import change_stage, stealth_rock_damage
from nuzline.engine.ids import AbilityId, ItemId, Stat
from nuzline.engine.state import BattleState

_I32 = jnp.int32

# The phases at which a Pokemon's ability can reveal itself to the AI.
SWITCH_IN, CONTACT_STATUS = "switch_in", "contact_status"
# Ability -> the phase at which it shows a message tagging it, for the implemented
# abilities that ever do:
# * Mold Breaker: SWITCH_IN_CHECK_STATE_MOLD_BREAKER (battle_lib.c:4075), a living
#   battler that has not announced it: subscript_mold_breaker.s prints
#   ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP, BTLSCR_MSG_BATTLER_TEMP``.
# * Static / Poison Point: when the contact ability inflicts its status
#   (``effects._contact_ability``): subscript_paralyze.s:44 / subscript_poison.s:57 tag
#   ``BTLSCR_MSG_BATTLER_TEMP``, the holder. A target that cannot take the status gets
#   no message.
# The others never reveal: Intimidate's "cuts" message and Keen Eye's "prevents"
# message are built in C (``BtlCmd_ChangeStatStage``, battle_script.c:2956, :2900);
# Inner Focus prints only against Fake Out or a direct flinch effect
# (subscript_flinch_mon.s:26-36), and every implemented flinch is a secondary effect;
# Blaze, Torrent, Overgrow, Rivalry, Guts, Simple, Unaware, Rock Head, Natural Cure,
# Steadfast and Sturdy show no ability message in M1.
REVEALS_ON = {
    AbilityId.MOLD_BREAKER: SWITCH_IN,
    AbilityId.STATIC: CONTACT_STATUS,
    AbilityId.POISON_POINT: CONTACT_STATUS,
}


def reveal_ability(state: BattleState, side, slot, phase: str, cond=True) -> BattleState:
    """``ai_ability_known[slot] = 1`` where ``cond``, ``side`` is the player (the AI only
    tracks the player's Pokemon) and party ``slot``'s ability reveals itself at
    ``phase`` (``REVEALS_ON``)."""
    abilities = jnp.asarray([a for a, p in REVEALS_ON.items() if p == phase], state.ability.dtype)
    side = jnp.asarray(side, _I32)
    on = cond & (side == 0) & jnp.isin(state.ability[side, slot], abilities)
    known = jnp.where(on, 1, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
    return state._replace(ai_ability_known=state.ai_ability_known.at[slot].set(known))


def after_damage_items(tables: Tables, state: BattleState, side, cond=True) -> BattleState:
    """``side``'s active Pokemon eats its Oran or Sitrus Berry if it has HP left and at
    most half its max HP (``curHP <= maxHP / 2``, i.e. ``hp * 2 <= max_hp``), where
    ``cond``: it heals (capped at max HP), loses the item, and ``item_consumed``
    records it."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    item = state.item[side, mon]
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    hp = state.hp[side, mon].astype(_I32)
    max_hp = state.max_hp[side, mon].astype(_I32)
    heal = jnp.where(
        effect == ItemId.ORAN_BERRY,
        power,
        jnp.where(effect == ItemId.SITRUS_BERRY, jnp.maximum(max_hp * power // 100, 1), 0),
    )
    on = cond & (heal > 0) & (hp > 0) & (hp * 2 <= max_hp)
    new_hp = jnp.where(on, jnp.minimum(hp + heal, max_hp), hp)
    return state._replace(
        hp=state.hp.at[side, mon].set(new_hp.astype(state.hp.dtype)),
        item=state.item.at[side, mon].set(jnp.where(on, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[side, mon].set(
            jnp.where(on, item, state.item_consumed[side, mon]).astype(state.item_consumed.dtype)
        ),
    )


def _intimidate(state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon, alive with Intimidate, lowers the living foe's Attack
    by 1 (clamped at -6)."""
    foe = 1 - side
    mon, target = state.active[side], state.active[foe]
    on = (
        (state.ability[side, mon] == AbilityId.INTIMIDATE)
        & (state.hp[side, mon] > 0)
        & (state.hp[foe, target] > 0)
    )
    return change_stage(state, foe, Stat.ATK, -1, on)


def on_switch_in(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon has just entered: Stealth Rock, Intimidate, a living
    newcomer's switch-in announcement (``reveal_ability``, ``SWITCH_IN``), then the
    held-item check of both active Pokemon (``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS``
    loops over every battler)."""
    side = jnp.asarray(side, _I32)
    state = stealth_rock_damage(tables, state, side)
    state = _intimidate(state, side)
    mon = state.active[side]
    state = reveal_ability(state, side, mon, SWITCH_IN, state.hp[side, mon] > 0)
    state = after_damage_items(tables, state, side)
    return after_damage_items(tables, state, 1 - side)
