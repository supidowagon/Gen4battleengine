"""Ability and held-item hooks at fixed phases (pure ``jnp``, jit/vmap-able, masked).

* ``on_switch_in``: what follows a Pokemon entering the field. The switch script
  (``subscript_switch_pokemon.s``) runs the entry hazards
  (``subscript_hazards_check.s``; Stealth Rock is the only M1 hazard), and the
  controller then runs ``BattleSystem_TriggerEffectOnSwitch`` (battle_lib.c:3649,
  from ``BattleControllerPlayer_MoveEnd``, battle_controller_player.c:3973; at the
  start of the battle from ``ShowBattleMon``, :256). Of its states the engine reaches
  ``SWITCH_IN_CHECK_STATE_TRACE`` (:3718, ``trace``), ``_WEATHER_ABILITIES`` (:3740,
  ``weather_ability``), ``_INTIMIDATE`` (:3795), ``_DOWNLOAD`` (:3814, ``download``),
  ``_FORBIDDEN_STATUS`` (:4135) and ``_HELD_ITEM_STATUS`` (:4151).
* ``held_item_check`` / ``after_damage_items``: ``BattleSystem_TriggerHeldItem``
  (battle_lib.c:4576) and ``BattleSystem_TriggerHeldItemOnStatus`` (:4967), which
  are the same for every implemented item (HP berries, Berry Juice, the status berries,
  Lum, White Herb, the stat pinch berries). ``step`` runs it after every hit of a
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
``item_consumed`` here). Klutz and Embargo are unsupported; Gluttony halves only the pinch
berries' param (``order.pinch_divisor``).

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
accuracy check, Swift Swim, the end-of-turn chip and the sun's freeze immunity
(``effects.can_have_status``: ``CheckIgnoreWeather``, subscript_freeze.s:6). Sturdy only blocks
OHKO moves in Gen 4 (``damage.fixed_damage``; its reveal is ``step._use_move``'s). E2 batch 3: Technician, Thick Fat, Dry Skin, Solar
Power, Reckless, Tinted Lens, Levitate and Scrappy in ``damage.py``; Shield Dust and the
armors in ``effects.py``; Super Luck, Serene Grace, No Guard and Tangled Feet in
``outcome.py``; Quick Feet and Truant in ``order.py`` / ``step.py``; Dry Skin's absorb and the
weather abilities' turn-end HP in ``step.py``; Trace and the weather abilities here.
Lightningrod redirects only to a battler other than the defender (battle_lib.c:1756-1772),
which singles never has, and Honey Gather acts after the battle (battle_script.c:7825):
neither has a hook.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import (
    ability_weather,
    apply_stage,
    decomp_abilities,
    divide,
    trace_copy,
)
from nuzline.engine.effects import change_stage, entry_hazards, forget_item
from nuzline.engine.ids import (
    CONFUSION_PENDING,
    ITEM_ABILITY_IDS,
    AbilityId,
    ItemId,
    Stat,
    Status,
    Weather,
)
from nuzline.engine.order import dislikes_berry, pinch_divisor
from nuzline.engine.state import STAGE_MAX, BattleState

_I32 = jnp.int32

# The phases at which a Pokemon's ability can reveal itself to the AI.
SWITCH_IN, CONTACT_STATUS = "switch_in", "contact_status"
SLEEP_BLOCKED, STATUS_CURE = "sleep_blocked", "status_cure"
FLINCH_BLOCKED = "flinch_blocked"
CONTACT_DAMAGE, PARALYSIS_BLOCKED = "contact_damage", "paralysis_blocked"
WEATHER_SET, ABSORB, WEATHER_HP = "weather_set", "absorb", "weather_hp"
OHKO_BLOCKED = "ohko_blocked"
STATUS_HP, TORMENT = "status_hp", "torment"
STATUS_BLOCKED, SYNCHRONIZED = "status_blocked", "synchronized"
DAMAGE_AVOIDED = "damage_avoided"
# Ability -> the phases at which it shows a message tagging it, for the implemented
# abilities that ever do:
# * Mold Breaker: SWITCH_IN_CHECK_STATE_MOLD_BREAKER (battle_lib.c:4075), a living
#   battler that has not announced it: subscript_mold_breaker.s prints
#   ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP, BTLSCR_MSG_BATTLER_TEMP``.
# * Pressure: SWITCH_IN_CHECK_STATE_PRESSURE (battle_lib.c:4096-4110), likewise for a living
#   battler whose ``pressureAnnounced`` is clear (cleared on entry, :113): subscript_pressure.s
#   prints "{0} is exerting its {1}!" with ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP,
#   BTLSCR_MSG_BATTLER_TEMP`` (ruling G-10).
# * Static / Poison Point / Flame Body: when the contact ability inflicts its status
#   (``effects._contact_ability``): subscript_paralyze.s:44 / subscript_poison.s:57 /
#   subscript_burn.s:_177 tag ``BTLSCR_MSG_BATTLER_TEMP``, the holder. A target that
#   cannot take the status (a Limber attacker against Static included, :_170) gets no
#   message.
# * Cute Charm: when it infatuates the attacker (``effects._contact_ability``):
#   subscript_infatuate.s:17 prints "{0}'s {1} infatuated {2}!" with
#   ``TAG_NICKNAME_ABILITY_NICKNAME, BTLSCR_MSG_TEMP, BTLSCR_MSG_BATTLER_TEMP`` (the ability
#   tag is the holder's, BattleMessage_AbilityTag, battle_script.c:11798-11809); a failed
#   TryAttract ends silently (:60-65).
# * Rough Skin: whenever it hurts the attacker: subscript_rough_skin.s prints
#   ``TAG_NICKNAME_ABILITY_NICKNAME, BTLSCR_DEFENDER, BTLSCR_DEFENDER, BTLSCR_ATTACKER``.
# * Limber: a direct paralysis (Thunder Wave) it blocks (subscript_paralyze.s:_170, "prevents
#   paralysis", ``TAG_NICKNAME_ABILITY``; an indirect or ability side effect ends silently),
#   and the cure of a paralysis (``RecoverStatusByAbility``, as Insomnia).
# * Insomnia: a sleep move it blocks (subscript_fall_asleep.s:_221 prints "stayed awake
#   because of its {1}" with ``TAG_NICKNAME_ABILITY``), and the cure of a sleep status
#   (``RecoverStatusByAbility``, subscript_ability_forbids_status.s:6 tags
#   ``BTLSCR_MSG_TEMP``).
# * Leaf Guard: a direct status (a status move) it blocks in sun (``effects._inflict``; ruling
#   G-14): subscript_poison.s:_201, _badly_poison.s:_266, _burn.s:_264, _paralyze.s:_170 and
#   _fall_asleep.s:_221 print their "prevents" / "stayed awake" message with
#   ``TAG_NICKNAME_ABILITY``; an indirect or ability side effect ends silently. (Toxic Spikes and
#   Yawn reveal it through their own code, by decomp id.)
# * Synchronize: a status it passes on (``effects.synchronize``; ruling G-13): subscript_poison.s
#   :57 / _burn.s:68 / _paralyze.s:44 tag ``BTLSCR_MSG_BATTLER_TEMP``, the holder; a pass the
#   receiver blocks is silent.
# * Wonder Guard: a move it guards (MOVE_STATUS_WONDER_GUARD, ``damage.wonder_guarded``; ruling
#   G-20): subscript_missed.s:107-110 prints "{0} avoided damage by using {1}!" with
#   ``TAG_NICKNAME_ABILITY``, BTLSCR_DEFENDER, unless an earlier flag of subscript_missed.s prints
#   first: lost focus (:5), protected (:18), failed (:97) (``step._use_move``). A miss still
#   reaches it (the plain-miss message is :157, after it).
# * Shed Skin: subscript_ability_restore_status.s:8 tags the holder
#   (``TAG_NICKNAME_ABILITY_STATUS``).
# * Inner Focus: a Fake Out flinch it blocks (``effects._flinch``): subscript_flinch_mon.s:18-23
#   prints "won't flinch because of its {1}" with ``TAG_NICKNAME_ABILITY`` for
#   ``MOVE_FAKE_OUT`` (or a direct flinch effect, which no supported move has).
# * Drizzle, Sand Stream, Drought, Snow Warning: when the switch-in sets the weather
#   (``weather_ability``): subscript_drizzle.s etc. print ``TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP,
#   BTLSCR_MSG_BATTLER_TEMP`` (the holder); the decomp skips the script when its permanent
#   weather is up already, so nothing is shown then.
# * Dry Skin: a Water move it absorbs (subscript_ability_restores_hp.s tags BTLSCR_DEFENDER
#   either way, ``step._use_move``) and its turn-end HP change in sun or rain
#   (subscript_weather_continues.s :_045, :_076 tag ``BTLSCR_MSG_BATTLER_TEMP``, the battler);
#   Solar Power's turn-end loss in sun (:_057).
# * Trace's own message names the copied ability through the target (``trace``).
# * Sturdy: an OHKO move it blocks (BtlCmd_TryOHKOMove sets MOVE_STATUS_STURDY,
#   battle_script.c:4340-4341; subscript_missed.s:67-69 prints "{0} was protected by {1}!"
#   with ``TAG_NICKNAME_ABILITY``, BTLSCR_DEFENDER) unless an earlier flag of subscript_missed.s
#   prints first: protected (:18), Levitate (:34), ineffective (:59) (``step._use_move``).
# The others never reveal: Intimidate's "cuts" message, Download's "raised its" message
# (``SetupNicknameAbilityStatMsg``, :2854) and Keen Eye's "prevents"
# message are built in C (``BtlCmd_ChangeStatStage``, battle_script.c:2956, :2900);
# Shield Dust blocks silently (an indirect side effect); Truant's "loafing around"
# (subscript_loafing_around.s) tags only the nickname; the damage, accuracy, crit, speed and
# chance abilities print nothing;
# Blaze, Torrent, Overgrow, Rivalry, Guts, Simple, Unaware, Rock Head, Natural Cure,
# Steadfast, Cloud Nine, Compoundeyes, Sand Veil, Snow Cloak, Swift Swim,
# Adaptability and Solid Rock show no ability message in M1 (``ApplyTypeChart`` prints
# nothing).
REVEALS_ON: dict[int, frozenset[str]] = {
    AbilityId.MOLD_BREAKER: frozenset({SWITCH_IN}),
    AbilityId.PRESSURE: frozenset({SWITCH_IN}),
    AbilityId.STATIC: frozenset({CONTACT_STATUS}),
    AbilityId.POISON_POINT: frozenset({CONTACT_STATUS}),
    AbilityId.FLAME_BODY: frozenset({CONTACT_STATUS}),
    AbilityId.CUTE_CHARM: frozenset({CONTACT_STATUS}),
    AbilityId.ROUGH_SKIN: frozenset({CONTACT_DAMAGE}),
    AbilityId.INSOMNIA: frozenset({SLEEP_BLOCKED, STATUS_CURE}),
    AbilityId.LIMBER: frozenset({PARALYSIS_BLOCKED, STATUS_CURE}),
    AbilityId.SHED_SKIN: frozenset({STATUS_CURE}),
    AbilityId.INNER_FOCUS: frozenset({FLINCH_BLOCKED}),
    AbilityId.DRIZZLE: frozenset({WEATHER_SET}),
    AbilityId.SAND_STREAM: frozenset({WEATHER_SET}),
    AbilityId.DROUGHT: frozenset({WEATHER_SET}),
    AbilityId.SNOW_WARNING: frozenset({WEATHER_SET}),
    AbilityId.DRY_SKIN: frozenset({ABSORB, WEATHER_HP}),
    AbilityId.SOLAR_POWER: frozenset({WEATHER_HP}),
    # B9-6: subscript_ability_restores_hp.s and subscript_weather_continues.s:_076, as Dry Skin's.
    AbilityId.VOLT_ABSORB: frozenset({ABSORB}),
    AbilityId.WATER_ABSORB: frozenset({ABSORB}),
    AbilityId.RAIN_DISH: frozenset({WEATHER_HP}),
    AbilityId.AFTERMATH: frozenset({CONTACT_DAMAGE}),
    # subscript_poison_damage.s:10 (TAG_NICKNAME_ABILITY, the holder); subscript_bad_dreams.s:6
    # (TAG_NICKNAME_ABILITY_NICKNAME: BTLSCR_MSG_ATTACKER, the holder).
    AbilityId.POISON_HEAL: frozenset({STATUS_HP}),
    AbilityId.BAD_DREAMS: frozenset({TORMENT}),
    AbilityId.STURDY: frozenset({OHKO_BLOCKED}),
    AbilityId.LEAF_GUARD: frozenset({STATUS_BLOCKED}),
    AbilityId.SYNCHRONIZE: frozenset({SYNCHRONIZED}),
    AbilityId.WONDER_GUARD: frozenset({DAMAGE_AVOIDED}),
}


def download(state: BattleState, side, foes, cond=True, staged=True) -> BattleState:
    """``SWITCH_IN_CHECK_STATE_DOWNLOAD`` (battle_lib.c:3814-3851) for ``side``'s active battler
    where ``cond`` (its ``downloadAnnounced`` is clear: it entered, or it just traced Download),
    ruling G-21: with HP left and Download (``Battler_Ability``), it sums over the opposing
    battlers ``foes`` (sides; the singles foe) that have HP and are not behind a substitute
    (``VOLATILE_CONDITION_SUBSTITUTE``: its bit) their Defense and their Sp. Def, each through
    its stage (``stats * sStatStageBoosts[stage]``, ``damage.apply_stage``; the raw stage, no
    Simple). ``downloadAnnounced`` is set whatever the sums (:3836); when they are not both 0
    the larger-or-equal Defense raises Sp. Attack, a larger Sp. Def raises Attack, by 1 through
    ``subscript_update_stat_stage.s`` as an ability side effect (``BtlCmd_ChangeStatStage``,
    battle_script.c:2841-2873: at +6 it ends silently). Its "raised its" message is built in C
    (``SetupNicknameAbilityStatMsg``, :2854): the AI learns nothing. ``staged=False`` reads every
    foe's stages as 0 (``search.playout.replacement_pass``, a fresh newcomer)."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    sum_def, sum_spd = jnp.int32(0), jnp.int32(0)
    for foe in foes:  # the opposing battlers (doubles: both)
        target = state.active[foe]
        counts = (state.hp[foe, target] > 0) & (state.substitute_hp[foe] == 0)
        stats = state.stats[foe, target].astype(_I32)
        stages = state.stages[foe].astype(_I32) * int(bool(staged))
        defense = apply_stage(stats[Stat.DEF - 1], stages[Stat.DEF])
        sp_def = apply_stage(stats[Stat.SPD - 1], stages[Stat.SPD])
        sum_def = sum_def + jnp.where(counts, defense, 0)
        sum_spd = sum_spd + jnp.where(counts, sp_def, 0)
    on = (
        jnp.asarray(cond)
        & (state.ability[side, mon] == AbilityId.DOWNLOAD)
        & (state.hp[side, mon] > 0)
        & (sum_def + sum_spd != 0)
    )
    state = change_stage(state, side, Stat.SPA, 1, on & (sum_def >= sum_spd))
    return change_stage(state, side, Stat.ATK, 1, on & (sum_def < sum_spd))


def reveal_ability(state: BattleState, side, slot, phase: str, cond=True) -> BattleState:
    """``ai_ability_known[slot] = 1`` where ``cond``, ``side`` is the player (the AI only
    tracks the player's Pokemon) and party ``slot``'s ability reveals itself at
    ``phase`` (``REVEALS_ON``)."""
    abilities = jnp.asarray([a for a, p in REVEALS_ON.items() if phase in p], state.ability.dtype)
    side = jnp.asarray(side, _I32)
    on = cond & (side == 0) & jnp.isin(state.ability[side, slot], abilities)
    known = jnp.where(on, 1, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
    return state._replace(ai_ability_known=state.ai_ability_known.at[slot].set(known))


def _know(state: BattleState, side, slot, cond) -> BattleState:
    """``ai_ability_known[slot] = 1`` where ``cond`` and ``side`` is the player: a script
    wrote or tagged the battler's ability (``BattleAI_SetAbility``, battle_script.c:3074,
    :3403, :11806), whatever the ability."""
    on = jnp.asarray(cond) & (jnp.asarray(side, _I32) == 0)
    known = jnp.where(on, 1, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
    return state._replace(ai_ability_known=state.ai_ability_known.at[slot].set(known))


def trace(
    tables: Tables, state: BattleState, side, foe, cond=True
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, traced)``: ``SWITCH_IN_CHECK_STATE_TRACE`` (battle_lib.c:3718-3737) for
    ``side``'s active battler against ``foe``'s, where ``cond`` and ``damage.trace_copy``
    passes: ``subscript_trace.s`` sets its ``BATTLEMON_ABILITY`` to the foe's (the battler's,
    not the party's: ``BattleMon_CopyToParty`` writes an ability only for a form change,
    battle_controller.c:1495-1500, so it ends when the battler leaves,
    ``step._restore_ability``). The write is ``BattleAI_SetAbility`` of the tracer
    (battle_script.c:3403) and the message tags the target's ability (``BTLSCR_MSG_DEFENDER``,
    :11806): the AI knows both (``_know``). Abilities the engine does not model are refused by
    coverage before they can be copied (the target holds them)."""
    mon, target = state.active[side], state.active[foe]
    traced = jnp.asarray(cond) & trace_copy(tables, state, side, foe)
    new = jnp.where(traced, state.ability[foe, target], state.ability[side, mon])
    state = state._replace(ability=state.ability.at[side, mon].set(new.astype(state.ability.dtype)))
    state = _know(state, side, mon, traced)
    return _know(state, foe, target, traced), traced


def weather_ability(state: BattleState, side, cond=True) -> BattleState:
    """``SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES`` (battle_lib.c:3740-3790) for ``side``'s
    active battler where ``cond`` (it has not announced: it entered, or it just traced): with
    HP left and Drizzle, Sand Stream, Drought or Snow Warning (``Battler_Ability``), the weather
    becomes that one, permanently (``weather_timer`` 0; subscript_drizzle.s etc.:
    ``FIELD_CONDITION_WEATHER`` off, then the ``*_PERM`` bit), unless that permanent weather is
    up already (no script then, the field is the same). Cloud Nine does not stop it."""
    mon = state.active[side]
    weather = ability_weather(state.ability[side, mon])
    on = jnp.asarray(cond) & (weather != Weather.NONE) & (state.hp[side, mon] > 0)
    already = (state.weather == weather) & (state.weather_timer == 0)
    state = reveal_ability(state, side, mon, WEATHER_SET, on & ~already)
    return state._replace(
        weather=jnp.where(on, weather, state.weather).astype(state.weather.dtype),
        weather_timer=jnp.where(on, 0, state.weather_timer).astype(state.weather_timer.dtype),
    )


def arm_unburden(state: BattleState, side, cond=True) -> BattleState:
    """``BattleSystem_RecoverStatusByAbility``'s ``ABILITY_UNBURDEN`` case (battle_lib.c:
    4433-4437) for ``side``'s active battler where ``cond``: an Unburden holder
    (``Battler_Ability``) holding an item gets ``canUnburden`` (no HP test, no result)."""
    side = jnp.asarray(side, _I32)
    mon = state.active[side]
    on = (
        jnp.asarray(cond)
        & (state.ability[side, mon] == AbilityId.UNBURDEN)
        & (state.item[side, mon] != 0)
    )
    value = jnp.where(on, 1, state.can_unburden[side]).astype(state.can_unburden.dtype)
    return state._replace(can_unburden=state.can_unburden.at[side].set(value))


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
    state = arm_unburden(state, side, cond)
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


# The status berries: the statuses each one cures (``BattleSystem_TriggerHeldItem``,
# battle_lib.c:4601-4634; ``BattleSystem_PluckBerry``, :5569-5607). Pecha also cures toxic
# (``MON_CONDITION_ANY_POISON``, :4616).
_STATUS_BERRIES = (
    (ItemId.CHERI_BERRY, (Status.PARALYSIS,)),  # HOLD_EFFECT_PRZ_RESTORE, :4601
    (ItemId.CHESTO_BERRY, (Status.SLEEP,)),  # _SLP_RESTORE, :4608
    (ItemId.PECHA_BERRY, (Status.POISON, Status.TOXIC)),  # _PSN_RESTORE, :4615
    (ItemId.RAWST_BERRY, (Status.BURN,)),  # _BRN_RESTORE, :4622
    (ItemId.ASPEAR_BERRY, (Status.FREEZE,)),  # _FRZ_RESTORE, :4629
)


def _restore(state: BattleState, side, mon, effect, cond):
    """``(state, triggered)``: the cure half of a status or stat item ``effect`` on
    ``side``'s party ``mon`` (its active battler), where ``cond`` and it has something to
    cure. Cheri: paralysis (``subscript_held_item_prz_restore.s``); Chesto: sleep
    (``..._slp_restore.s``); Pecha: poison or toxic (``MON_CONDITION_ANY_POISON``,
    ``..._psn_restore.s``); Rawst: burn (``..._brn_restore.s``); Aspear: freeze
    (``..._frz_restore.s``); Persim: confusion (``..._cnf_restore.s``, battle_lib.c:4655-4660);
    Lum: any status and confusion (``..._prz/slp/psn/brn/frz/cnf/multi_restore.s``: the
    status, then the confusion); White Herb: every stage below neutral back to neutral
    (battle_lib.c:4875-4887, done in C before ``..._statdown_restore.s``). A cured status
    takes its counter with it (sleep turns and the toxic counter are bits of ``status``)."""
    status = state.status[side, mon]
    any_status = status != Status.NONE
    confused = state.confusion[side] > 0
    lowered = state.stages[side] < 0
    lum = effect == ItemId.LUM_BERRY
    berry = jnp.zeros((), bool)
    for item, statuses in _STATUS_BERRIES:
        cures = jnp.zeros((), bool)
        for cured in statuses:
            cures = cures | (status == cured)
        berry = berry | ((effect == item) & cures)
    cure_status = cond & (berry | (lum & any_status))
    cure_confusion = cond & (lum | (effect == ItemId.PERSIM_BERRY)) & confused
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


def _raise_stat(tables: Tables, state: BattleState, side, item, cond):
    """``(state, raised)``: a ``PINCH_BERRY`` ``item`` raises ``side``'s stat
    ``item_param[item, 1]`` by 1 where ``cond`` and that stage is below +6
    (``statBoosts[stat] < MAX_STAT_STAGE``, battle_lib.c:4779 and :5754):
    ``subscript_held_item_raise_stat.s`` runs ``BtlCmd_ChangeStatStage`` with
    ``SIDE_EFFECT_TYPE_HELD_ITEM`` (battle_script.c:2841-2873; Simple applies where the stage
    is read, ``damage.simple_stage``)."""
    stat = tables.item_param[item, 1].astype(_I32)
    on = jnp.asarray(cond) & (tables.item_effect[item] == ItemId.PINCH_BERRY)
    on = on & (state.stages[side, stat] < STAGE_MAX)
    return change_stage(state, side, stat, 1, on), on


def _hp_restore(effect, power, max_hp) -> jnp.ndarray:
    """HP an HP berry restores: Oran Berry and Berry Juice (``HOLD_EFFECT_HP_RESTORE``) its
    param, Sitrus Berry (``HOLD_EFFECT_HP_PCT_RESTORE``) ``BattleSystem_Divide(maxHP *
    param, 100)``, a flavor berry (``HOLD_EFFECT_HP_RESTORE_SPICY`` .. ``_SOUR``)
    ``BattleSystem_Divide(maxHP, param)`` (battle_lib.c:4700 etc., :5680 etc.; build_tables
    keeps its param >= 1, ruling G-4); else 0."""
    flat = (effect == ItemId.ORAN_BERRY) | (effect == ItemId.BERRY_JUICE)
    pct = jnp.maximum(max_hp * power // 100, 1)
    flavor = divide(max_hp, jnp.maximum(power, 1))
    heal = jnp.where(effect == ItemId.FLAVOR_BERRY, flavor, 0)
    return jnp.where(flat, power, jnp.where(effect == ItemId.SITRUS_BERRY, pct, heal))


def berry_confuse(tables: Tables, state: BattleState, side, mon, cond) -> BattleState:
    """A flavor berry's dislike (subscript_held_item_dislike_flavor.s:10-13): after the heal,
    ``side``'s Pokemon in party slot ``mon`` (its active battler) goes through
    subscript_confuse.s with ``SIDE_EFFECT_TYPE_HELD_ITEM`` where ``cond``: its own Own Tempo
    (``CheckAbility``, :6), a confusion already (:7) or Safeguard on its side (:8) end the
    script before any draw; else ``Random 3, 2`` (:34) sets the counter. The counter is
    ``CONFUSION_PENDING`` until the battler's confusion check reads it, which draws it
    (``Outcome.berry_confusion``, ``step._use_move``): nothing reads more than the confusion
    bit before that, so the late draw has the decomp's law, in and out of ``step`` (the battle
    start, a faint replacement), ruling G-5."""
    own_tempo = decomp_abilities(tables, state)[side, mon] == ITEM_ABILITY_IDS["Own Tempo"]
    on = (
        jnp.asarray(cond) & ~own_tempo & (state.confusion[side] == 0) & (state.safeguard[side] == 0)
    )
    new = jnp.where(on, CONFUSION_PENDING, state.confusion[side]).astype(state.confusion.dtype)
    return state._replace(confusion=state.confusion.at[side].set(new))


def held_item_check(
    tables: Tables, state: BattleState, side, cond=True, keep=False
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, triggered)``: ``BattleSystem_TriggerHeldItem`` (battle_lib.c:4576) /
    ``BattleSystem_TriggerHeldItemOnStatus`` (:4967), the same for every implemented item,
    on ``side``'s active Pokemon where ``cond`` and it has HP left: an HP berry
    (``_hp_restore``) at ``curHP <= maxHP / 2`` (``hp * 2 <= max_hp``) heals, capped
    (``subscript_held_item_hp_restore.s``); a status berry or White Herb cures
    (``_restore``); a stat pinch berry at ``curHP <= maxHP / pinch_divisor`` raises its stat
    (``_raise_stat``, battle_lib.c:4773-4836; ``order.pinch_divisor``); a flavor berry heals
    like the HP berries and then confuses a holder that dislikes its flavor (``berry_confuse``,
    after the item is removed: the heal subscript's ``PLUCK_CHECK`` runs first,
    subscript_held_item_hp_restore.s:13; Gluttony does not touch it, battle_lib.c:4698-4771).
    Every one of those subscripts ends in
    ``BATTLE_SUBSCRIPT_PLUCK_CHECK``, which removes the item (``BtlCmd_RemoveItem``,
    battle_script.c:8972, which keeps it in ``item_consumed``) unless the attacker's
    ``SELF_TURN_FLAG_PLUCK_BERRY`` is set (``keep``, ``effects.MoveHit.pluck_flag``), and
    clears that flag."""
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
    pinch = live & (hp <= max_hp // pinch_divisor(tables, state, side, mon, item))
    state, raised = _raise_stat(tables, state, side, item, pinch)
    on = healed | cured | raised
    spent = on & ~jnp.asarray(keep)
    state = forget_item(state, side, mon, spent)  # RemoveItem's CopyToParty
    state = state._replace(
        item=state.item.at[side, mon].set(jnp.where(spent, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[side, mon].set(
            jnp.where(spent, item, state.item_consumed[side, mon]).astype(state.item_consumed.dtype)
        ),
    )
    disliked = healed & dislikes_berry(tables, state, side, mon, item)
    return berry_confuse(tables, state, side, mon, disliked), on


def after_damage_items(tables: Tables, state: BattleState, side, cond=True) -> BattleState:
    """``held_item_check`` without the pluck flag (every call site but the hits of a
    move, ``step._use_move``)."""
    return held_item_check(tables, state, side, cond)[0]


# Item effects with a pluck effect (res/items/data/*.json ``pluckEffect``): Oran
# PLUCK_EFFECT_HP_RESTORE, Sitrus _HP_PCT_RESTORE, Cheri _PRZ_RESTORE, Chesto _SLP_RESTORE,
# Pecha _PSN_RESTORE, Rawst _BRN_RESTORE, Aspear _FRZ_RESTORE, Persim _CNF_RESTORE, Lum
# _ALL_RESTORE, the stat pinch berries _ATK_UP .. _SPDEF_UP. Every other implemented item has
# PLUCK_EFFECT_NONE; of those only the berries (Item_IsBerry, battle_lib.c:5833) are eaten,
# with no effect. Berry Juice and every non-berry item (``NONE`` included: Nugget, a Poke
# Ball) are not eaten.
PLUCKABLE = (
    ItemId.ORAN_BERRY,
    ItemId.SITRUS_BERRY,
    ItemId.CHERI_BERRY,
    ItemId.CHESTO_BERRY,
    ItemId.PECHA_BERRY,
    ItemId.RAWST_BERRY,
    ItemId.ASPEAR_BERRY,
    ItemId.PERSIM_BERRY,
    ItemId.LUM_BERRY,
    ItemId.PINCH_BERRY,
    ItemId.FLAVOR_BERRY,  # PLUCK_EFFECT_HP_RESTORE_SPICY .. _SOUR (:5678-5751), G-4
    # The resist berries, Custap Berry and the berries with no hold effect (Razz, Nanab,
    # ...): PLUCK_EFFECT_NONE berries, eaten with no effect (Item_IsBerry, :5833).
    ItemId.RESIST_BERRY,
    ItemId.CUSTAP_BERRY,
    ItemId.NO_EFFECT_BERRY,
)


def pluck_berry(
    tables: Tables, state: BattleState, side, foe, cond=True, foe_custap=False
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, pluck_flag)``: ``side``'s active Pokemon eats the berry of ``foe``'s (the
    defender's side) active Pokemon where
    ``cond`` (``subscript_pluck.s``): ``BtlCmd_TryPluck`` (battle_script.c:8116) ->
    ``BattleSystem_PluckBerry`` (battle_lib.c:5539). ``foe_custap``: the foe's
    ``moveEffectsData.custapBerry`` is set (``order.custap_flags``: the turn order activated
    its Custap Berry, eaten when its move starts), and then ``TryPluck`` does nothing while the
    foe holds an item (battle_script.c:8125-8127). A ``PLUCKABLE`` berry is removed
    (``RemoveItem BTLSCR_DEFENDER``, kept in ``item_consumed``) even if the foe fainted,
    and its pluck effect acts on the eater: Oran heals its param, Sitrus
    ``Divide(maxHP * param, 100)``, a flavor berry ``Divide(maxHP, param)`` (and confuses an
    eater whose nature dislikes it, ``berry_confuse``: subscript_pluck.s sets
    ``msgBattlerTemp`` to the attacker, battle_lib.c:5678-5751), each only below full HP
    (capped); Cheri, Chesto, Pecha,
    Rawst, Aspear, Persim and Lum cure as the held berry would (``_restore``: the same restore
    subscripts); a stat pinch berry raises the eater's stat by 1 below +6 whatever its HP
    (``_raise_stat``, :5753-5796: ``subscript_pluck.s`` sets ``msgBattlerTemp`` to the
    attacker); the resist berries, Custap and the berries with no hold effect
    (``PLUCK_EFFECT_NONE`` berries) are eaten with no effect. Anything else stays
    (:5833). ``SELF_TURN_FLAG_PLUCK_BERRY`` is set for every eaten berry
    (:5848) and cleared by the restore subscript's ``PLUCK_CHECK``, so it is left set
    (``pluck_flag``) when the berry did nothing for the eater (``nextSeq == 0``). The
    eater's own item is never touched here. Sticky Hold, Klutz and Embargo are
    unsupported."""
    side = jnp.asarray(side, _I32)
    foe = jnp.asarray(foe, _I32)
    user, target = state.active[side], state.active[foe]
    item = state.item[foe, target]
    effect = tables.item_effect[item]
    power = tables.item_param[item, 0].astype(_I32)
    hp = state.hp[side, user].astype(_I32)
    max_hp = state.max_hp[side, user].astype(_I32)
    blocked = jnp.asarray(foe_custap) & (item != 0)
    on = jnp.asarray(cond) & ~blocked & jnp.isin(effect, jnp.asarray(PLUCKABLE, effect.dtype))
    heal = _hp_restore(effect, power, max_hp)
    healed = on & (heal > 0) & (hp != max_hp)
    new_hp = jnp.where(healed, jnp.minimum(hp + heal, max_hp), hp)
    state = state._replace(hp=state.hp.at[side, user].set(new_hp.astype(state.hp.dtype)))
    state, cured = _restore(state, side, user, effect, on)
    state, raised = _raise_stat(tables, state, side, item, on)
    # subscript_pluck.s:7 names the berry (BTLSCR_DEFENDER), then RemoveItem BTLSCR_DEFENDER
    # (:10) clears the record again (forget_item).
    state = forget_item(state, foe, target, on)
    state = state._replace(
        item=state.item.at[foe, target].set(jnp.where(on, 0, item).astype(state.item.dtype)),
        item_consumed=state.item_consumed.at[foe, target].set(
            jnp.where(on, item, state.item_consumed[foe, target]).astype(state.item_consumed.dtype)
        ),
    )
    disliked = healed & dislikes_berry(tables, state, side, user, item)
    state = berry_confuse(tables, state, side, user, disliked)
    return state, on & ~healed & ~cured & ~raised


def intimidate(state: BattleState, side, cond=True, foe=None) -> BattleState:
    """Where ``cond``, ``side``'s active Pokemon, alive with Intimidate, lowers the living
    ``foe``'s Attack by 1 (clamped at -6; by default the other side). ``on_switch_in`` runs it
    for a newcomer and for a battler that just traced Intimidate;
    ``playout.replacement_pass`` runs it again for a player newcomer that entered while the
    opponent's active was down (its Intimidate is still unannounced, battle_lib.c:3795)."""
    foe = 1 - side if foe is None else foe
    mon, target = state.active[side], state.active[foe]
    # BtlCmd_ChangeStatStage (battle_script.c:2884-2902): the foe's Clear Body or Hyper Cutter
    # (Battler_IgnorableAbility with the Intimidate holder as attacker, never Mold Breaker)
    # blocks the drop; its "suppressed" message is built in C (no AI reveal), B9-6.
    blocks = (state.ability[foe, target] == AbilityId.CLEAR_BODY) | (
        state.ability[foe, target] == AbilityId.HYPER_CUTTER
    )
    on = (
        jnp.asarray(cond)
        & (state.ability[side, mon] == AbilityId.INTIMIDATE)
        & (state.hp[side, mon] > 0)
        & (state.hp[foe, target] > 0)
        # subscript_intimidate.s:13: a foe behind a substitute is skipped (B10-4); outside an
        # action no flag is set, so its bit.
        & (state.substitute_hp[foe] == 0)
        & ~blocks
    )
    return change_stage(state, foe, Stat.ATK, -1, on)


def on_switch_in(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon has just entered: the entry hazards
    (``effects.entry_hazards``: Toxic Spikes, Spikes, Stealth Rock), then the switch-in checks
    in their state order (``BattleSystem_TriggerEffectOnSwitch``): Trace (``trace``) for the
    newcomer and for the other battler, whose Trace stayed armed while its foe was down (a
    double KO's replacements: the player's newcomer enters first, ``playout``); the weather
    abilities (``weather_ability``), Intimidate and Download (``download``) of the newcomer and
    of a battler that just traced one; a living newcomer's switch-in announcement (``reveal_ability``,
    ``SWITCH_IN``: Mold Breaker, Pressure) and a traced one's; the forbidden-status check
    (``recover_status_by_ability``, battle_lib.c:4135); then the held-item check of both
    active Pokemon (``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS`` loops over every battler). Each
    state loops over ``monSpeedOrder``; in singles the order changes nothing here: two Trace
    holders cannot copy each other, a traced ability is the other battler's own, and coverage
    refuses two different weather abilities on the two sides (``coverage.weather_war``)."""
    return switch_in_effects(tables, entry_hazards(tables, state, jnp.asarray(side, _I32)), side)


def switch_in_effects(tables: Tables, state: BattleState, side) -> BattleState:
    """``on_switch_in`` after the entry hazards: ``BattleSystem_TriggerEffectOnSwitch``'s states
    for ``side``'s newcomer. A U-turn or Baton Pass newcomer meets them apart from its hazards: the
    party list runs only the hazards and a faint check (subscript_show_party_list.s:22-24), and the
    rest comes at ``AfterMoveEffects``' ``TRIGGER_SWITCH_IN_EFFECTS`` (bcp.c:3672-3682;
    ``step._use_move``)."""
    side = jnp.asarray(side, _I32)
    foe = 1 - side  # the entry point: the battler facing the newcomer in singles
    state, _ = trace(tables, state, side, foe)
    state, traced = trace(tables, state, foe, side)
    state = weather_ability(state, side)
    state = weather_ability(state, foe, traced)
    state = intimidate(state, side, foe=foe)
    state = intimidate(state, foe, traced, foe=side)
    state = download(state, side, (foe,))
    state = download(state, foe, (side,), traced)
    mon, other = state.active[side], state.active[foe]
    state = reveal_ability(state, side, mon, SWITCH_IN, state.hp[side, mon] > 0)
    state = reveal_ability(state, foe, other, SWITCH_IN, traced)
    state = recover_status_by_ability(state, side)
    state = recover_status_by_ability(state, foe, traced)
    # SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS runs for every battler (battle_lib.c:4135-4145):
    # the foe's Unburden case too, traced or not (its status cure cannot newly apply untraced).
    state = arm_unburden(state, foe)
    state = after_damage_items(tables, state, side)
    return after_damage_items(tables, state, foe)


def battle_start(tables: Tables, state: BattleState) -> BattleState:
    """The leads' switch-in checks at the start of the battle: ``ShowBattleMon``
    (battle_controller_player.c:256) runs ``BattleSystem_TriggerEffectOnSwitch`` once, a state
    machine whose every state loops over all battlers (``monSpeedOrder``) before the next
    state runs (battle_lib.c:3649-4168): Trace (:3718-3737) for each lead, then the weather
    abilities (:3740-3790), then Intimidate (:3795-3808, once per battler: its
    ``intimidateAnnounced``), then Download (:3814-3851, its ``downloadAnnounced``; it reads the
    foe's Defense and Sp. Def, which no earlier state changes), then the switch-in announcements, the forbidden-status check
    (:4135) and the held-item check (:4151). So a lead that traced Intimidate fires it once,
    in the Intimidate state, like any Intimidate lead (ruling B3-18). No entry hazard is up at
    the start. A loop over battlers per state (the doubles conventions); in singles the order
    within a state changes nothing (``on_switch_in``'s docstring)."""
    sides = (0, 1)
    # Before ShowBattleMon, BattleSystem_InitBattleMon (bcp.c:227) arms canUnburden for every
    # lead holding an item, whatever its ability (battle_lib.c:161-166; no item is knocked off
    # yet): a lead that gains Unburden later (Trace) still has it.
    for side in sides:
        held = state.item[side, state.active[side]] != 0
        value = jnp.where(held, 1, state.can_unburden[side]).astype(state.can_unburden.dtype)
        state = state._replace(can_unburden=state.can_unburden.at[side].set(value))
    for side in sides:
        state, _ = trace(tables, state, side, 1 - side)
    for side in sides:
        state = weather_ability(state, side)
    for side in sides:
        state = intimidate(state, side, foe=1 - side)
    for side in sides:  # SWITCH_IN_CHECK_STATE_DOWNLOAD (:3814-3851), after every Intimidate
        state = download(state, side, (1 - side,))
    for side in sides:
        mon = state.active[side]
        state = reveal_ability(state, side, mon, SWITCH_IN, state.hp[side, mon] > 0)
    for side in sides:
        state = recover_status_by_ability(state, side)
    for side in sides:
        state = after_damage_items(tables, state, side)
    return state
