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
Static and Poison Point in ``effects.py``; Steadfast, Natural Cure and Shed Skin in
``step.py``, Compoundeyes in ``outcome.py``. Cloud Nine has no behaviour: every decomp
branch on it is a weather branch (``NO_CLOUD_NINE``, include/battle/common.h:63) and
coverage refuses every weather but none. Sturdy only blocks OHKO moves in Gen 4, and
none is in M1, so it has no hook.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.effects import change_stage, stealth_rock_damage
from nuzline.engine.ids import AbilityId, ItemId, Stat, Status
from nuzline.engine.state import BattleState

_I32 = jnp.int32

# The phases at which a Pokemon's ability can reveal itself to the AI.
SWITCH_IN, CONTACT_STATUS = "switch_in", "contact_status"
SLEEP_BLOCKED, STATUS_CURE = "sleep_blocked", "status_cure"
# Ability -> the phases at which it shows a message tagging it, for the implemented
# abilities that ever do:
# * Mold Breaker: SWITCH_IN_CHECK_STATE_MOLD_BREAKER (battle_lib.c:4075), a living
#   battler that has not announced it: subscript_mold_breaker.s prints
#   ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP, BTLSCR_MSG_BATTLER_TEMP``.
# * Static / Poison Point: when the contact ability inflicts its status
#   (``effects._contact_ability``): subscript_paralyze.s:44 / subscript_poison.s:57 tag
#   ``BTLSCR_MSG_BATTLER_TEMP``, the holder. A target that cannot take the status gets
#   no message.
# * Insomnia: a sleep move it blocks (subscript_fall_asleep.s:_221 prints "stayed awake
#   because of its {1}" with ``TAG_NICKNAME_ABILITY``), and the cure of a sleep status
#   (``RecoverStatusByAbility``, subscript_ability_forbids_status.s:6 tags
#   ``BTLSCR_MSG_TEMP``).
# * Shed Skin: subscript_ability_restore_status.s:8 tags the holder
#   (``TAG_NICKNAME_ABILITY_STATUS``).
# The others never reveal: Intimidate's "cuts" message and Keen Eye's "prevents"
# message are built in C (``BtlCmd_ChangeStatStage``, battle_script.c:2956, :2900);
# Inner Focus prints only against Fake Out or a direct flinch effect
# (subscript_flinch_mon.s:26-36), and every implemented flinch is a secondary effect;
# Blaze, Torrent, Overgrow, Rivalry, Guts, Simple, Unaware, Rock Head, Natural Cure,
# Steadfast, Sturdy, Cloud Nine and Compoundeyes show no ability message in M1.
REVEALS_ON: dict[int, frozenset[str]] = {
    AbilityId.MOLD_BREAKER: frozenset({SWITCH_IN}),
    AbilityId.STATIC: frozenset({CONTACT_STATUS}),
    AbilityId.POISON_POINT: frozenset({CONTACT_STATUS}),
    AbilityId.INSOMNIA: frozenset({SLEEP_BLOCKED, STATUS_CURE}),
    AbilityId.SHED_SKIN: frozenset({STATUS_CURE}),
}


def reveal_ability(state: BattleState, side, slot, phase: str, cond=True) -> BattleState:
    """``ai_ability_known[slot] = 1`` where ``cond``, ``side`` is the player (the AI only
    tracks the player's Pokemon) and party ``slot``'s ability reveals itself at
    ``phase`` (``REVEALS_ON``)."""
    abilities = jnp.asarray([a for a, p in REVEALS_ON.items() if phase in p], state.ability.dtype)
    side = jnp.asarray(side, _I32)
    on = cond & (side == 0) & jnp.isin(state.ability[side, slot], abilities)
    known = jnp.where(on, 1, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
    return state._replace(ai_ability_known=state.ai_ability_known.at[slot].set(known))


def recover_status_by_ability(state: BattleState, side, cond=True) -> BattleState:
    """``BattleSystem_RecoverStatusByAbility`` (battle_lib.c:4378) for ``side``'s active
    Pokemon, where ``cond``: an Insomnia holder (the only ability of the family in M1)
    that is asleep and has HP left wakes (counter 0) and shows
    ``subscript_ability_forbids_status.s``, whose message tags its ability. The decomp
    runs it for the attacker then the defender at each move end
    (``BattleControllerPlayer_MoveEnd``, battle_controller_player.c:3965-3967) and for
    every battler on a switch-in (``SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS``,
    battle_lib.c:4135-4145)."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    woke = (
        cond
        & (state.ability[side, mon] == AbilityId.INSOMNIA)
        & (state.status[side, mon] == Status.SLEEP)
        & (state.hp[side, mon] > 0)
    )
    state = state._replace(
        status=state.status.at[side, mon].set(
            jnp.where(woke, Status.NONE, state.status[side, mon]).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[side, mon].set(
            jnp.where(woke, 0, state.status_counter[side, mon]).astype(state.status_counter.dtype)
        ),
    )
    return reveal_ability(state, side, mon, STATUS_CURE, woke)


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


def pluck_berry(tables: Tables, state: BattleState, side, cond=True) -> BattleState:
    """``side``'s active Pokemon eats the foe's Oran or Sitrus Berry where ``cond``
    (``subscript_pluck.s``): ``BtlCmd_TryPluck`` (battle_script.c:8116) ->
    ``BattleSystem_PluckBerry`` (battle_lib.c:5539), which heals only when the eater is
    not at full HP (Oran ``PLUCK_EFFECT_HP_RESTORE`` the item's param, Sitrus
    ``PLUCK_EFFECT_HP_PCT_RESTORE`` ``maxHP * param / 100``; capped at max HP). The berry
    is removed either way (``BtlCmd_RemoveItem``, battle_script.c:8972, keeps it in
    ``item_consumed``), even if the foe fainted. Items without a pluck effect stay;
    Sticky Hold and Custap Berry are unsupported."""
    side = jnp.asarray(side, _I32)
    foe = 1 - side
    user, target = state.active[side], state.active[foe]
    item = state.item[foe, target]
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    hp = state.hp[side, user].astype(_I32)
    max_hp = state.max_hp[side, user].astype(_I32)
    heal = jnp.where(
        effect == ItemId.ORAN_BERRY,
        power,
        jnp.where(effect == ItemId.SITRUS_BERRY, jnp.maximum(max_hp * power // 100, 1), 0),
    )
    on = cond & (heal > 0)
    new_hp = jnp.where(on & (hp != max_hp), jnp.minimum(hp + heal, max_hp), hp)
    return state._replace(
        hp=state.hp.at[side, user].set(new_hp.astype(state.hp.dtype)),
        item=state.item.at[foe, target].set(jnp.where(on, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[foe, target].set(
            jnp.where(on, item, state.item_consumed[foe, target]).astype(state.item_consumed.dtype)
        ),
    )


def intimidate(state: BattleState, side, cond=True) -> BattleState:
    """Where ``cond``, ``side``'s active Pokemon, alive with Intimidate, lowers the living
    foe's Attack by 1 (clamped at -6). ``on_switch_in`` runs it for a newcomer;
    ``playout.replacement_pass`` runs it again for a player newcomer that entered while the
    opponent's active was down (its Intimidate is still unannounced, battle_lib.c:3795)."""
    foe = 1 - side
    mon, target = state.active[side], state.active[foe]
    on = (
        jnp.asarray(cond)
        & (state.ability[side, mon] == AbilityId.INTIMIDATE)
        & (state.hp[side, mon] > 0)
        & (state.hp[foe, target] > 0)
    )
    return change_stage(state, foe, Stat.ATK, -1, on)


def on_switch_in(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon has just entered: Stealth Rock, Intimidate, a living
    newcomer's switch-in announcement (``reveal_ability``, ``SWITCH_IN``), the
    forbidden-status check (``recover_status_by_ability``, battle_lib.c:4135), then the
    held-item check of both active Pokemon (``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS``
    loops over every battler)."""
    side = jnp.asarray(side, _I32)
    state = stealth_rock_damage(tables, state, side)
    state = intimidate(state, side)
    mon = state.active[side]
    state = reveal_ability(state, side, mon, SWITCH_IN, state.hp[side, mon] > 0)
    state = recover_status_by_ability(state, side)
    state = after_damage_items(tables, state, side)
    return after_damage_items(tables, state, 1 - side)
