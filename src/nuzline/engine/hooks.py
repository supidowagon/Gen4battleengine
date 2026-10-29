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
    """``side``'s active Pokemon has just entered: Stealth Rock, Intimidate, then the
    held-item check of both active Pokemon (``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS``
    loops over every battler)."""
    side = jnp.asarray(side, _I32)
    state = stealth_rock_damage(tables, state, side)
    state = _intimidate(state, side)
    state = after_damage_items(tables, state, side)
    return after_damage_items(tables, state, 1 - side)
