"""Ability and held-item hooks at fixed phases (pure ``jnp``, jit/vmap-able, masked).

* ``on_switch_in``: what follows a Pokemon entering the field. The switch script
  (``subscript_switch_pokemon.s``) runs the entry hazards
  (``subscript_hazards_check.s``; Stealth Rock is the only M1 hazard), and the
  controller then runs ``BattleSystem_TriggerEffectOnSwitch`` (battle_lib.c:3649,
  from ``BattleControllerPlayer_MoveEnd``, battle_controller_player.c:3973; at the
  start of the battle from ``ShowBattleMon``, :256). Of its states M1 reaches
  ``SWITCH_IN_CHECK_STATE_INTIMIDATE`` (:3795) and
  ``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS`` (:4151).
* ``held_item_check`` / ``after_damage_items``: ``BattleSystem_TriggerHeldItem``
  (battle_lib.c:4576) and ``BattleSystem_TriggerHeldItemOnStatus`` (:4967), which
  are the same for every implemented item (HP berries, Berry Juice, Chesto, Pecha,
  Lum, White Herb). ``step`` runs it after every hit of a
  move (``AFTER_MOVE_EFFECT_ATTACKER_ITEM`` / ``_DEFENDER_ITEM``,
  battle_controller_player.c:3684, :3691, then ``_HELD_ITEM_STATUS`` for both, :3730;
  the after-move effects run per hit before
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

Oran Berry and Berry Juice (``HOLD_EFFECT_HP_RESTORE``) heal ``itemPower`` (10, 20) and
Sitrus Berry (``HOLD_EFFECT_HP_PCT_RESTORE``) ``BattleSystem_Divide(maxHP * itemPower, 100)``
(itemPower 25), all when ``0 < curHP <= maxHP / 2``.
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
Simple, Mold Breaker, Adaptability and Solid Rock in ``damage.py``; Keen Eye, Inner
Focus, Rock Head, Static, Poison Point, Flame Body, Rough Skin and Limber in
``effects.py`` (Limber's cure here); Steadfast, Natural Cure and Shed Skin in ``step.py``,
Compoundeyes, Sand Veil and Snow Cloak (accuracy) in ``outcome.py``, Swift
Swim in ``order.py``; Sand Veil and Snow Cloak also in the end-of-turn weather chip
(``step.py``). Cloud Nine: every decomp branch on it is a weather branch
(``NO_CLOUD_NINE``, include/battle/common.h:63); ``damage.weather_suppressed`` (read by
decomp ability id, with Air Lock) gates the weather modifiers of the damage formula, the
accuracy check, Swift Swim and the end-of-turn chip. Sturdy only blocks OHKO moves in Gen
4, and none is in M1, so it has no hook.
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
FLINCH_BLOCKED = "flinch_blocked"
CONTACT_DAMAGE, PARALYSIS_BLOCKED = "contact_damage", "paralysis_blocked"
# Ability -> the phases at which it shows a message tagging it, for the implemented
# abilities that ever do:
# * Mold Breaker: SWITCH_IN_CHECK_STATE_MOLD_BREAKER (battle_lib.c:4075), a living
#   battler that has not announced it: subscript_mold_breaker.s prints
#   ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP, BTLSCR_MSG_BATTLER_TEMP``.
# * Static / Poison Point / Flame Body: when the contact ability inflicts its status
#   (``effects._contact_ability``): subscript_paralyze.s:44 / subscript_poison.s:57 /
#   subscript_burn.s:_177 tag ``BTLSCR_MSG_BATTLER_TEMP``, the holder. A target that
#   cannot take the status (a Limber attacker against Static included, :_170) gets no
#   message.
# * Rough Skin: whenever it hurts the attacker: subscript_rough_skin.s prints
#   ``TAG_NICKNAME_ABILITY_NICKNAME, BTLSCR_DEFENDER, BTLSCR_DEFENDER, BTLSCR_ATTACKER``.
# * Limber: a direct paralysis (Thunder Wave) it blocks (subscript_paralyze.s:_170, "prevents
#   paralysis", ``TAG_NICKNAME_ABILITY``; an indirect or ability side effect ends silently),
#   and the cure of a paralysis (``RecoverStatusByAbility``, as Insomnia).
# * Insomnia: a sleep move it blocks (subscript_fall_asleep.s:_221 prints "stayed awake
#   because of its {1}" with ``TAG_NICKNAME_ABILITY``), and the cure of a sleep status
#   (``RecoverStatusByAbility``, subscript_ability_forbids_status.s:6 tags
#   ``BTLSCR_MSG_TEMP``).
# * Shed Skin: subscript_ability_restore_status.s:8 tags the holder
#   (``TAG_NICKNAME_ABILITY_STATUS``).
# * Inner Focus: a Fake Out flinch it blocks (``effects._flinch``): subscript_flinch_mon.s:18-23
#   prints "won't flinch because of its {1}" with ``TAG_NICKNAME_ABILITY`` for
#   ``MOVE_FAKE_OUT`` (or a direct flinch effect, which no supported move has).
# The others never reveal: Intimidate's "cuts" message and Keen Eye's "prevents"
# message are built in C (``BtlCmd_ChangeStatStage``, battle_script.c:2956, :2900);
# Blaze, Torrent, Overgrow, Rivalry, Guts, Simple, Unaware, Rock Head, Natural Cure,
# Steadfast, Sturdy, Cloud Nine, Compoundeyes, Sand Veil, Snow Cloak, Swift Swim,
# Adaptability and Solid Rock show no ability message in M1 (``ApplyTypeChart`` prints
# nothing).
REVEALS_ON: dict[int, frozenset[str]] = {
    AbilityId.MOLD_BREAKER: frozenset({SWITCH_IN}),
    AbilityId.STATIC: frozenset({CONTACT_STATUS}),
    AbilityId.POISON_POINT: frozenset({CONTACT_STATUS}),
    AbilityId.FLAME_BODY: frozenset({CONTACT_STATUS}),
    AbilityId.ROUGH_SKIN: frozenset({CONTACT_DAMAGE}),
    AbilityId.INSOMNIA: frozenset({SLEEP_BLOCKED, STATUS_CURE}),
    AbilityId.LIMBER: frozenset({PARALYSIS_BLOCKED, STATUS_CURE}),
    AbilityId.SHED_SKIN: frozenset({STATUS_CURE}),
    AbilityId.INNER_FOCUS: frozenset({FLINCH_BLOCKED}),
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
    Pokemon, where ``cond``: an asleep Insomnia holder (:4404) or a paralyzed Limber holder
    (:4397), the abilities of the family in M1, with HP left is cured (counter 0) and
    shows ``subscript_ability_forbids_status.s``, whose message tags its ability. The
    ability is the holder's own (``Battler_Ability``: Mold Breaker does not matter). The
    decomp runs it for the attacker then the defender at each move end
    (``BattleControllerPlayer_MoveEnd``, battle_controller_player.c:3965-3967) and for
    every battler on a switch-in (``SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS``,
    battle_lib.c:4135-4145)."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    ability, status = state.ability[side, mon], state.status[side, mon]
    woke = (
        cond
        & (
            ((ability == AbilityId.INSOMNIA) & (status == Status.SLEEP))
            | ((ability == AbilityId.LIMBER) & (status == Status.PARALYSIS))
        )
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


def _restore(state: BattleState, side, mon, effect, cond):
    """``(state, triggered)``: the cure half of a status or stat item ``effect`` on
    ``side``'s party ``mon`` (its active battler), where ``cond`` and it has something to
    cure. Chesto: sleep (``subscript_held_item_slp_restore.s``); Pecha: poison or toxic
    (``MON_CONDITION_ANY_POISON``, ``..._psn_restore.s``); Lum: any status and confusion
    (``..._prz/slp/psn/brn/frz/cnf/multi_restore.s``: the status, then the confusion);
    White Herb: every stage below neutral back to neutral (battle_lib.c:4875-4887, done in
    C before ``..._statdown_restore.s``). A cured status takes its counter with it (sleep
    turns and the toxic counter are bits of ``status``)."""
    status = state.status[side, mon]
    asleep = status == Status.SLEEP
    poisoned = (status == Status.POISON) | (status == Status.TOXIC)
    any_status = status != Status.NONE
    confused = state.confusion[side] > 0
    lowered = state.stages[side] < 0
    lum = effect == ItemId.LUM_BERRY
    cure_status = cond & (
        ((effect == ItemId.CHESTO_BERRY) & asleep)
        | ((effect == ItemId.PECHA_BERRY) & poisoned)
        | (lum & any_status)
    )
    cure_confusion = cond & lum & confused
    herb = cond & (effect == ItemId.WHITE_HERB) & jnp.any(lowered)
    state = state._replace(
        status=state.status.at[side, mon].set(
            jnp.where(cure_status, Status.NONE, status).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[side, mon].set(
            jnp.where(cure_status, 0, state.status_counter[side, mon]).astype(
                state.status_counter.dtype
            )
        ),
        confusion=state.confusion.at[side].set(
            jnp.where(cure_confusion, 0, state.confusion[side]).astype(state.confusion.dtype)
        ),
        stages=state.stages.at[side].set(
            jnp.where(herb & lowered, 0, state.stages[side]).astype(state.stages.dtype)
        ),
    )
    return state, cure_status | cure_confusion | herb


def _hp_restore(effect, power, max_hp) -> jnp.ndarray:
    """HP an HP berry restores: Oran Berry and Berry Juice (``HOLD_EFFECT_HP_RESTORE``) its
    param, Sitrus Berry (``HOLD_EFFECT_HP_PCT_RESTORE``) ``BattleSystem_Divide(maxHP *
    param, 100)``; else 0."""
    flat = (effect == ItemId.ORAN_BERRY) | (effect == ItemId.BERRY_JUICE)
    pct = jnp.maximum(max_hp * power // 100, 1)
    return jnp.where(flat, power, jnp.where(effect == ItemId.SITRUS_BERRY, pct, 0))


def held_item_check(
    tables: Tables, state: BattleState, side, cond=True, keep=False
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, triggered)``: ``BattleSystem_TriggerHeldItem`` (battle_lib.c:4576) /
    ``BattleSystem_TriggerHeldItemOnStatus`` (:4967), the same for every implemented item,
    on ``side``'s active Pokemon where ``cond`` and it has HP left: an HP berry
    (``_hp_restore``) at ``curHP <= maxHP / 2`` (``hp * 2 <= max_hp``) heals, capped
    (``subscript_held_item_hp_restore.s``); a status berry or White Herb cures
    (``_restore``). Every one of those subscripts ends in ``BATTLE_SUBSCRIPT_PLUCK_CHECK``,
    which removes the item (``BtlCmd_RemoveItem``, battle_script.c:8972, which keeps it in
    ``item_consumed``) unless the attacker's ``SELF_TURN_FLAG_PLUCK_BERRY`` is set
    (``keep``, ``effects.MoveHit.pluck_flag``), and clears that flag."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    item = state.item[side, mon]
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    hp = state.hp[side, mon].astype(_I32)
    max_hp = state.max_hp[side, mon].astype(_I32)
    live = jnp.asarray(cond) & (hp > 0)
    heal = _hp_restore(effect, power, max_hp)
    healed = live & (heal > 0) & (hp * 2 <= max_hp)
    new_hp = jnp.where(healed, jnp.minimum(hp + heal, max_hp), hp)
    state = state._replace(hp=state.hp.at[side, mon].set(new_hp.astype(state.hp.dtype)))
    state, cured = _restore(state, side, mon, effect, live)
    on = healed | cured
    spent = on & ~jnp.asarray(keep)
    state = state._replace(
        item=state.item.at[side, mon].set(jnp.where(spent, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[side, mon].set(
            jnp.where(spent, item, state.item_consumed[side, mon]).astype(state.item_consumed.dtype)
        ),
    )
    return state, on


def after_damage_items(tables: Tables, state: BattleState, side, cond=True) -> BattleState:
    """``held_item_check`` without the pluck flag (every call site but the hits of a
    move, ``step._use_move``)."""
    return held_item_check(tables, state, side, cond)[0]


# Item effects with a pluck effect (res/items/data/*.json ``pluckEffect``): Oran
# PLUCK_EFFECT_HP_RESTORE, Sitrus _HP_PCT_RESTORE, Chesto _SLP_RESTORE, Pecha _PSN_RESTORE,
# Lum _ALL_RESTORE. Berry Juice, Focus Sash, White Herb, Big Root and Lagging Tail have
# PLUCK_EFFECT_NONE and are no berries (Item_IsBerry, battle_lib.c:5833): not eaten.
PLUCKABLE = (
    ItemId.ORAN_BERRY,
    ItemId.SITRUS_BERRY,
    ItemId.CHESTO_BERRY,
    ItemId.PECHA_BERRY,
    ItemId.LUM_BERRY,
    # Rindo, Chilan: PLUCK_EFFECT_NONE berries, eaten with no effect (Item_IsBerry, :5833).
    ItemId.RESIST_BERRY,
)


def pluck_berry(
    tables: Tables, state: BattleState, side, cond=True
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, pluck_flag)``: ``side``'s active Pokemon eats the foe's berry where
    ``cond`` (``subscript_pluck.s``): ``BtlCmd_TryPluck`` (battle_script.c:8116) ->
    ``BattleSystem_PluckBerry`` (battle_lib.c:5539). A ``PLUCKABLE`` berry is removed
    (``RemoveItem BTLSCR_DEFENDER``, kept in ``item_consumed``) even if the foe fainted,
    and its pluck effect acts on the eater: Oran heals its param, Sitrus
    ``Divide(maxHP * param, 100)``, each only below full HP (capped); Chesto, Pecha and Lum
    cure as the held berry would (``_restore``: the same restore subscripts); Rindo and
    Chilan (``PLUCK_EFFECT_NONE`` berries) are eaten with no effect. Anything else stays
    (:5833). ``SELF_TURN_FLAG_PLUCK_BERRY`` is set for every eaten berry
    (:5848) and cleared by the restore subscript's ``PLUCK_CHECK``, so it is left set
    (``pluck_flag``) when the berry did nothing for the eater (``nextSeq == 0``). The
    eater's own item is never touched here. Sticky Hold, Custap Berry, Klutz and Embargo
    are unsupported."""
    side = jnp.asarray(side, _I32)
    foe = 1 - side
    user, target = state.active[side], state.active[foe]
    item = state.item[foe, target]
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    hp = state.hp[side, user].astype(_I32)
    max_hp = state.max_hp[side, user].astype(_I32)
    on = jnp.asarray(cond) & jnp.isin(effect, jnp.asarray(PLUCKABLE, effect.dtype))
    heal = _hp_restore(effect, power, max_hp)
    healed = on & (heal > 0) & (hp != max_hp)
    new_hp = jnp.where(healed, jnp.minimum(hp + heal, max_hp), hp)
    state = state._replace(hp=state.hp.at[side, user].set(new_hp.astype(state.hp.dtype)))
    state, cured = _restore(state, side, user, effect, on)
    state = state._replace(
        item=state.item.at[foe, target].set(jnp.where(on, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[foe, target].set(
            jnp.where(on, item, state.item_consumed[foe, target]).astype(state.item_consumed.dtype)
        ),
    )
    return state, on & ~healed & ~cured


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
