"""Chance outcomes of one turn (spec §4.2 "Explicit chance outcomes").

``step`` draws no randomness: every random event of a turn is a field of
``Outcome``, which ``sample_outcome`` draws (Monte Carlo) and later enumeration
lists. Per-hit fields are ``[side, hit_index]`` with up to 5 hits; per-action
fields are ``[side]``. ``secondary`` is indexed by the attacking side (its own
move's secondary effect; Thunder Fang's paralysis roll); ``fang_flinch`` by the
attacking side too (Thunder Fang's second roll, its flinch), ``present`` (Present's
power or heal), ``protect`` (the user's Protect / Detect roll), ``infatuation`` (the
moving side's attract check, 1 = immobilized by love), ``tri_attack`` (Tri Attack's
status, ``ids.TRI_ATTACK_STATUSES``), ``roar_level`` (Roar's level check passes) and
``roar_target`` (the rank of the Pokemon Roar drags in among the target side's eligible
bench, ``roar_slot``); ``contact_proc`` by the
defending side (its contact ability: Static, Poison Point, Flame Body, Cute Charm);
``duration`` by the acting (inflicting) side, and it is the counter value the decomp's
``Random`` command stores for that side's move
(``sample_outcome``).

``sample_outcome`` reproduces each decomp roll (citations per field below). The
crit draw is read with the odds at the hit (``crit_lands``). Rolls whose effect
depends on state that can change mid-turn (status, the defender's ability) are drawn
unconditionally and gated by ``step``, which keeps their probabilities exact.

The accuracy roll is the exception: its threshold can change between the start of the
turn and the move's own check (a faster Sand-Attack or Double Team), and the decomp
computes it at the check (``BattleControllerPlayer_CheckMoveHitAccuracy``,
battle_controller_player.c:2865-2979, with the stages of that moment). So
``hit[side, 0]`` holds the roll ``u = RandNext() % 100 + 1`` (:2976) reduced to its
*interval class*: ``hit_thresholds`` lists every threshold the move can have at its
check this turn, and two rolls are in one class when they hit or miss alike at each of
them. The stored value is the smallest roll of the class, or 0 when the class misses
at every one of them (``hit_class``). ``step`` compares it with the live threshold
(``live_hit_threshold``): the move hits when ``0 < hit <= threshold``. With a single
threshold ``t`` below 100 this is the old boolean (1 = hits, 0 = misses); 0 in a
hand-built outcome forces a miss.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.damage import (
    PRESENT_HEAL,
    ability_weather,
    entry_ability,
    entry_weather,
    hazards_loss,
    trace_copy,
    weather_suppressed,
)
from nuzline.engine.format import SINGLES, Format, foes, format_of, party, side_of
from nuzline.engine.format import foe as _foe
from nuzline.engine.ids import (
    CHATTER_CHANCES,
    FOCUS_BAND_PARAM,
    GRAVITY_TURNS,
    ON_HIT_EFFECTS,
    QUICK_CLAW_MODULUS,
    SLEEP_TALK_VALUES,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    Stat,
    Weather,
)
from nuzline.engine.order import (
    CALL_ASSIST,
    CALL_METRONOME,
    CALL_NONE,
    CALL_SLEEP_TALK,
    MOVE_ACTIONS,
    action_order,
    assist_candidates,
    call_kind,
    disabled_state,
    dislikes_berry,
    encored_state,
    gravity_possible,
    gravity_state,
    is_item,
    is_switch,
    metronome_weights,
    move_effects,
    sealed_state,
    selected_move,
    takeover_possible,
    used_move,
    used_moves,
)
from nuzline.engine.state import (
    BATON_PASSED_FIELDS,
    STAGE_MAX,
    STAGE_MIN,
    VOLATILE_FIELDS,
    BattleState,
    held_item,
    held_items,
)

# Beat Up hits once per eligible party member, up to the six of a full party
# (BtlCmd_BeatUp, battle_script.c:6173-6247; E2 batch 13, B13-9).
MAX_HITS = 6
_I8, _I32 = jnp.int8, jnp.int32

# HitRateByStage (include/data/hit_rate_stages.h), indexed by the clamped stage sum 0..12.
_ACC_NUM = jnp.array([33, 36, 43, 50, 60, 75, 1, 133, 166, 2, 233, 133, 3], _I32)
_ACC_DEN = jnp.array([100, 100, 100, 100, 100, 100, 1, 100, 100, 1, 100, 50, 1], _I32)
# sCriticalStageRates (battle_lib.c:7097): a crit is 1 in N at stage 0..4.
_CRIT_ODDS = jnp.array([16, 8, 4, 3, 2], _I32)


class Outcome(NamedTuple):
    speed_tie: jnp.ndarray  # int8[], 0 = player first on equal speed
    # int8[B,6], index 0: the accuracy roll's class (module docstring); index k > 0: Triple Kick's
    # hit k's own roll, its class over the thresholds without the overrides (B13-6)
    hit: jnp.ndarray
    crit: jnp.ndarray  # int8[B,6], the crit draw's class (CRIT_DRAWS): 0 never, 1 always crits
    roll: jnp.ndarray  # int8[B,6], 0-15 (85-100 %)
    secondary: jnp.ndarray  # int8[B,6], the attacker's own secondary effect
    contact_proc: jnp.ndarray  # int8[B,6], the defender's contact ability
    n_hits: jnp.ndarray  # int8[B], 2-5
    full_para: jnp.ndarray  # int8[B]
    confusion_hit: jnp.ndarray  # int8[B]
    duration: (
        jnp.ndarray
    )  # int8[B], sleep/confusion/Bind/Taunt/Disable/Uproar counter rolled at infliction
    thaw: jnp.ndarray  # int8[B]
    flinch_item: jnp.ndarray  # int8[B,6], King's Rock / Razor Fang's flinch roll per hit
    quick_claw: jnp.ndarray  # int8[B], this turn's speedRand % 5 == 0 (Quick Claw)
    end_speed_tie: jnp.ndarray  # int8[], 0 = player first in the end-of-turn order on equal speed
    magnitude: jnp.ndarray  # int8[B], Magnitude's level 4-10 of each side's move
    shed_skin: jnp.ndarray  # int8[B], Shed Skin's RandNext() % 10 < 3
    fang_flinch: jnp.ndarray  # int8[B], Thunder Fang's flinch roll (secondary is its paralysis)
    present: jnp.ndarray  # int8[B], Present: 0/1/2 = power 40/80/120, PRESENT_HEAL = the heal
    protect: jnp.ndarray  # int8[B], 1 = the Protect / Detect / Endure roll passes (PROTECT_RATES)
    infatuation: jnp.ndarray  # int8[B], 1 = immobilized by love (RandNext() & 1 == 0)
    tri_attack: jnp.ndarray  # int8[B], Tri Attack's status 0/1/2 (ids.TRI_ATTACK_STATUSES)
    roar_level: jnp.ndarray  # int8[B], 1 = Roar's level check passes (BattleSystem_CanWhirlwind)
    roar_target: jnp.ndarray  # int8[B], Roar's pick: a rank in the eligible bench (roar_slot)
    # int16[B], an invoker's draw (order.invoked_move): Sleep Talk's 0..SLEEP_TALK_VALUES - 1,
    # Metronome's candidate rank; Copycat draws nothing.
    call: jnp.ndarray
    # int8[B], by the battler Future Sight hits at this turn's end: its CheckMoveHit roll's class
    # (``future_thresholds``; the raw roll 1-100 where ``future_loose``), read like ``hit``.
    future_hit: jnp.ndarray
    # int8[B], by the battler a Yawn puts to sleep at this turn's end: subscript_fall_asleep.s's
    # ``Random 3, 2`` (2-5 turns), a draw of its own (bcp.c:1684-1697).
    yawn_sleep: jnp.ndarray
    # int16[B], Acupressure's RandNext() (BtlCmd_BoostRandomStatBy2, battle_script.c:8950) as a
    # value in [0, ACUPRESSURE_DRAWS): the stat is rank ``value % validStats`` among the stats
    # below +6 at the move (``effects.acupressure_stat``), B12-9.
    acupressure: jnp.ndarray
    # int8[B,6], 1 = a Focus Band roll passes (``RandNext() % 100 < FOCUS_BAND_PARAM``,
    # bcp.c:3387, battle_script.c:9110): ``[s, i]`` the defender's roll at hit ``i`` of ``s``'s
    # move (``BattleControllerPlayer_UpdateHP``; Pursuit's ``CheckHoldOnWith1HP`` at ``[pursuer,
    # 0]``), ``[s, 0]`` also ``s``'s own roll at its confusion self-hit (its move does not run
    # then), ruling G-1.
    focus_band: jnp.ndarray
    # int8[B], by the battler Future Sight hits at this turn's end: its Focus Band roll
    # (subscript_future_sight_damage.s:28, ``CheckHoldOnWith1HP``).
    future_band: jnp.ndarray
    # int8[B]: the confusion counter (2-5) of subscript_confuse.s's ``Random 3, 2`` for a flavor
    # berry the side's battler disliked (subscript_held_item_dislike_flavor.s:13), drawn when
    # its confusion check first reads the pending counter (``ids.CONFUSION_PENDING``; one check
    # per side per turn), ruling G-5.
    berry_confusion: jnp.ndarray
    # int8[B], 1 = Chatter's roll passes (``RandNext() % 100 <= effectChance``, battle_script.c:
    # 8428-8430; the chance is ``ids.CHATTER_CHANCES`` of the user's ``chatot_cry``), G-70.
    chatter: jnp.ndarray


def empty_outcome(fmt: Format = SINGLES) -> Outcome:
    """Every hit lands, no crits, roll 15 (100 %), no procs, shortest durations."""
    hits, sides = jnp.zeros((fmt.B, MAX_HITS), _I8), jnp.zeros((fmt.B,), _I8)
    return Outcome(
        speed_tie=jnp.zeros((), _I8),
        hit=hits + 1,
        crit=hits,
        roll=hits + 15,
        secondary=hits,
        contact_proc=hits,
        n_hits=sides + 2,
        full_para=sides,
        confusion_hit=sides,
        duration=sides + 2,
        thaw=sides,
        flinch_item=hits,
        quick_claw=sides,
        end_speed_tie=jnp.zeros((), _I8),
        magnitude=sides + 4,
        shed_skin=sides,
        fang_flinch=sides,
        present=sides,
        protect=sides + 1,
        infatuation=sides,
        tri_attack=sides,
        roar_level=sides,
        roar_target=sides,
        call=jnp.zeros((fmt.B,), jnp.int16),
        future_hit=sides + 1,
        yawn_sleep=sides + 2,
        acupressure=jnp.zeros((fmt.B,), jnp.int16),
        focus_band=hits,
        future_band=sides,
        berry_confusion=sides + 2,
        chatter=sides,
    )


def outcome_to_py(outcome: Outcome) -> Outcome:
    """The same ``Outcome`` with every field Python ints / nested lists (for the
    plain-Python reference engine, ``nuzline.reference``)."""
    return Outcome(*(np.asarray(field).tolist() for field in outcome))


def _target_mon(state: BattleState, actions, side):
    """The party slot ``side``'s move will hit: the other side's active Pokemon, or
    the one it switches in this turn (switches go first)."""
    target = _foe(side)
    switching = is_switch(actions[target])
    return (
        target,
        switching,
        jnp.where(switching, actions[target] - MOVE_ACTIONS, state.active[target]),
    )


def _rate(
    tables: Tables,
    state: BattleState,
    side,
    move,
    target,
    target_mon,
    target_switches,
    acc,
    eva_stage,
    target_moved=False,
    weather=None,
    target_confused=None,
    total=None,
    overrides=True,
    micle=None,
    later_micle=None,
) -> jnp.ndarray:
    """``hitRate`` of ``side``'s ``move`` against side ``target``'s ``target_mon`` with the
    attacker's accuracy stage ``acc`` and the target's evasion stage ``eva_stage``
    (``target_switches``: it comes in this turn, fresh); ``target_moved``: the target's action
    is over (``Battler_MovedThisTurn``, battle_lib.c:5365-5368: in singles, the attacker acts
    second), for Zoom Lens; ``weather`` (default ``state.weather``) and ``target_confused``
    (default: the target's confusion counter is nonzero, a newcomer never) as at the check.
    100 when No Guard makes the move hit (``CheckMoveHitOverrides``, bcp.c:3016-3021). A
    target that switches in holds its entry ability (``damage.entry_ability``: the copy when it
    traces), since its switch-in checks run before the move (ruling B3-19). ``total``: the
    clamped stage sum to use instead of the one ``acc`` and ``eva_stage`` give (0-12,
    ``future_thresholds``' grid); with it the weather is taken as given, unsuppressed.
    ``overrides`` (static): ``CheckMoveHitOverrides`` runs after the roll (No Guard, Lock-On,
    Thunder in rain, Blizzard in hail give 100); False for a Triple Kick hit after the first,
    whose SYSCTL_SKIP_ACCURACY_OVERRIDES skips it (bcp.c:3264-3268, system_control.h:81-86),
    B13-6; ``"both"``: the two as a trailing axis ``[with, without]``, one computation.
    ``micle``: the attacker's ``moveEffectsData.micleBerry`` at the check (default
    ``state.micle[side]``), x120/100 after the held items (bcp.c:2967-2970, G-59);
    ``later_micle``: with ``"both"``, the flag for the column without the overrides (a later
    Triple Kick hit's: the first hit's check cleared the flag, :2968, so only a berry eaten
    since sets it; default ``micle``)."""
    fmt = format_of(state)
    weather = state.weather if weather is None else weather
    # A pivot's newcomer (``pivot_view``) is not fresh: its view holds its own stages / volatiles.
    fresh = jnp.asarray(target_switches, bool) & (state.pivot_battler != target + 1)
    if target_confused is None:
        target_confused = ~fresh & (state.confusion[target] > 0)
    accuracy = tables.move_accuracy[move].astype(_I32)
    a_ability = state.ability[party(fmt, side), state.active[side]]
    own = jnp.where(
        jnp.asarray(target_switches, bool),
        entry_ability(tables, state, target, target_mon, side),
        state.ability[party(fmt, target), target_mon],
    )
    d_ability = jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, own)

    # BattleControllerPlayer_CheckMoveHitAccuracy (battle_controller_player.c:2865).
    acc = jnp.asarray(acc, _I32)
    eva = -jnp.asarray(eva_stage, _I32)
    acc = jnp.where(a_ability == AbilityId.SIMPLE, acc * 2, acc)
    eva = jnp.where(d_ability == AbilityId.SIMPLE, eva * 2, eva)
    acc = jnp.where(d_ability == AbilityId.UNAWARE, 0, acc)
    eva = jnp.where(a_ability == AbilityId.UNAWARE, 0, eva)
    # MON_IS_IDENTIFIED(defender) && evaStages < 0 (:2888): a Foresight-identified
    # defender's positive evasion is ignored (evaStages = 6 - boost, so eva = -boost).
    # A defender that switches in this turn is the fresh Pokemon, not identified.
    identified = ~fresh & (state.foresight[target] != 0)
    eva = jnp.where(identified & (eva < 0), 0, eva)
    grid = total is not None  # future_thresholds' grid point
    total = (
        jnp.clip(6 + eva + acc, 0, 12) if not grid else jnp.clip(jnp.asarray(total, _I32), 0, 12)
    )
    # NO_CLOUD_NINE (:2915, :2926, :3023), with the battlers on the field when the move runs.
    on_field = state.active.at[target].set(target_mon.astype(state.active.dtype))
    # A Selfdestruct / Explosion user's HP is already 0 here (effect_script_0007.s runs
    # before the check): its own Cloud Nine / Air Lock no longer counts.
    user = state.active[side]
    exploded = tables.move_effect[move] == EffectId.SELF_DESTRUCT
    hp = state.hp.at[party(fmt, side), user].set(
        jnp.where(exploded, 0, state.hp[party(fmt, side), user])
    )
    suppressed = weather_suppressed(tables, state._replace(hp=hp), on_field)
    if grid:  # the grid's weather is the one at the check (none for a lock)
        suppressed = jnp.asarray(False)
    weather = jnp.where(suppressed, Weather.NONE, weather)
    flags = tables.move_flags[move]
    thunder = (flags & MoveFlag.THUNDER) != 0
    # :2915-2917 Thunder in sun: hitRate = 50, before the stage scaling (B9-1).
    base = jnp.where(thunder & (weather == Weather.SUN), 50, accuracy)
    rate = base * _ACC_NUM[total] // _ACC_DEN[total]
    # Compoundeyes (battle_controller_player.c:2922-2924), right after the stage scaling.
    rate = jnp.where(a_ability == AbilityId.COMPOUNDEYES, rate * 130 // 100, rate)
    # Sand Veil in sand and Snow Cloak in hail (Battler_IgnorableAbility: Mold Breaker pierces
    # them, :2927, :2931) x80/100, then deep fog x6/10 (:2934).
    veiled = ((weather == Weather.SANDSTORM) & (d_ability == AbilityId.SAND_VEIL)) | (
        (weather == Weather.HAIL) & (d_ability == AbilityId.SNOW_CLOAK)
    )
    rate = jnp.where(veiled, rate * 80 // 100, rate)
    rate = jnp.where(weather == Weather.FOG, rate * 6 // 10, rate)  # deep fog
    # :2944-2947 Tangled Feet (Battler_IgnorableAbility) on a confused target: x50/100 (Hustle,
    # just before it, is unsupported).
    tangled = (d_ability == AbilityId.TANGLED_FEET) & jnp.asarray(target_confused, bool)
    rate = jnp.where(tangled, rate * 50 // 100, rate)
    # The held items (:2949-2965), after Hustle (an unsupported ability) and Tangled Feet and
    # before the Micle Berry flag and Gravity: the defender's
    # HOLD_EFFECT_ACC_REDUCE (BrightPowder, Lax Incense) * (100 - param) / 100, then the
    # attacker's HOLD_EFFECT_ACCURACY_UP (Wide Lens) and, against a target that has moved,
    # HOLD_EFFECT_ACCURACY_UP_SLOWER (Zoom Lens), each * (100 + param) / 100
    # (Battler_HeldItemEffect / Battler_HeldItemPower: ``state.held_item``, B13-29).
    d_item = held_item(state, target, target_mon)
    d_power = tables.item_param[d_item, 0].astype(_I32)
    reduce = tables.item_effect[d_item] == ItemId.BRIGHT_POWDER
    rate = jnp.where(reduce, rate * (100 - d_power) // 100, rate)
    a_item = held_item(state, side)
    a_effect = tables.item_effect[a_item]
    a_power = tables.item_param[a_item, 0].astype(_I32)
    lens = (a_effect == ItemId.WIDE_LENS) | (
        (a_effect == ItemId.ZOOM_LENS) & jnp.asarray(target_moved, bool)
    )
    rate = jnp.where(lens, rate * (100 + a_power) // 100, rate)
    # bcp.c:2967-2970: the Micle Berry's flag, after the items and before Gravity (G-59).
    micle = state.micle[side] != 0 if micle is None else jnp.asarray(micle, bool)

    def finish(rate, micle):
        rate = jnp.where(micle, rate * 120 // 100, rate)
        return jnp.where(state.gravity != 0, rate * 10 // 6, rate)

    before_micle = rate
    rate = finish(before_micle, micle)
    # CheckMoveHitOverrides (bcp.c:3016-3021), after the roll: the attacker's or the target's
    # No Guard (Battler_Ability, the target's own: not ignorable) clears MOVE_STATUS_MISSED.
    no_guard = (a_ability == AbilityId.NO_GUARD) | (own == AbilityId.NO_GUARD)
    # MON_IS_LOCKED_ONTO(attacker, defender) sits beside No Guard there (bcp.c:3016-3018), and
    # in TryOHKOMove's bypass (battle_script.c:4343-4361); a switch action's newcomer carries no
    # lock (UpdateAfterSwitch clears its moveEffectsMask, battle_lib.c:2070), a pivot's view keeps
    # what Baton Pass keeps (B8-2).
    locked = (
        ~fresh
        & (state.lock_on[target] != 0)
        & (state.lock_on_target[target] == 1 + jnp.asarray(side, _I32))
    )
    no_guard = no_guard | locked
    # :3023-3031 (after No Guard / Lock-On, inside NO_CLOUD_NINE): Thunder in rain, Blizzard in
    # hail clear MOVE_STATUS_MISSED (B9-1).
    weather_sure = (thunder & (weather == Weather.RAIN)) | (
        ((flags & MoveFlag.BLIZZARD) != 0) & (weather == Weather.HAIL)
    )
    no_guard = no_guard | weather_sure
    # An OHKO move (BtlCmd_TryOHKOMove, battle_script.c:4334-4386) sets
    # SYSCTL_NONSTANDARD_ACC_CHECK (:4338), so none of the above applies (bcp.c:2911-2913, :3016):
    # it hits when RandNext() % 100 < accuracy + (attacker level - defender level) and the
    # attacker's level is not lower (:4347-4353); with No Guard on either side it always hits
    # at a level that is not lower (:4357-4363). Sturdy is its damage's (damage.fixed_damage).
    user_level = state.level[party(fmt, side), state.active[side]].astype(_I32)
    target_level = state.level[party(fmt, target), target_mon].astype(_I32)
    ohko_rate = jnp.where(
        user_level >= target_level,
        jnp.where(no_guard, 100, accuracy + user_level - target_level),
        0,
    )
    ohko = tables.move_effect[move] == EffectId.OHKO
    # Accuracy 0 never misses (the check returns before rolling).
    full = jnp.where(ohko, ohko_rate, jnp.where((accuracy == 0) | no_guard, 100, rate))
    # SYSCTL_SKIP_ACCURACY_OVERRIDES (B13-6): the roll alone decides (a Triple Kick hit; no
    # OHKO move has one).
    raw = jnp.where(accuracy == 0, 100, rate)
    if overrides == "both":
        if later_micle is not None:
            later = finish(before_micle, jnp.asarray(later_micle, bool))
            raw = jnp.where(accuracy == 0, 100, later)
        return jnp.stack([full, raw], axis=-1)
    return full if overrides else raw


def live_hit_threshold(
    tables: Tables, state: BattleState, side, move, target_moved=False, overrides=True
) -> jnp.ndarray:
    """``int32``: the threshold of ``side``'s ``move`` at its accuracy check, read from
    ``state`` as it is at that moment (``step``): the stages, abilities, items, weather and
    HP of the two active battlers then; ``target_moved``: the target has acted this turn.
    ``overrides``: as ``_rate``'s (False: a later Triple Kick hit, B13-6)."""
    target_mon = state.active[_foe(side)]
    eva = state.stages[_foe(side), Stat.EVA]
    acc = state.stages[side, Stat.ACC]
    return _rate(
        tables,
        state,
        side,
        move,
        _foe(side),
        target_mon,
        jnp.asarray(False),
        acc,
        eva,
        target_moved,
        overrides=overrides,
    )


def stage_total(tables: Tables, state: BattleState, side, target, target_mon, acc, eva_stage):
    """``int32``: the clamped stage sum ``6 + evaStages + accStages`` of
    ``CheckMoveHitAccuracy`` (bcp.c:2873-2898) for ``side``'s active battler against side
    ``target``'s ``target_mon`` on the field (``_rate``'s, with ``acc`` / ``eva_stage``)."""
    fmt = format_of(state)
    a_ability = state.ability[party(fmt, side), state.active[side]]
    own = state.ability[party(fmt, target), target_mon]
    d_ability = jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, own)
    acc = jnp.asarray(acc, _I32)
    eva = -jnp.asarray(eva_stage, _I32)
    acc = jnp.where(a_ability == AbilityId.SIMPLE, acc * 2, acc)
    eva = jnp.where(d_ability == AbilityId.SIMPLE, eva * 2, eva)
    acc = jnp.where(d_ability == AbilityId.UNAWARE, 0, acc)
    eva = jnp.where(a_ability == AbilityId.UNAWARE, 0, eva)
    identified = (state.foresight[target] != 0) & (state.active[target] == target_mon)
    eva = jnp.where(identified & (eva < 0), 0, eva)
    return jnp.clip(6 + eva + acc, 0, 12)


def future_threshold(tables: Tables, state: BattleState, defender) -> jnp.ndarray:
    """``int32``: the threshold of the Future Sight / Doom Desire hit on side ``defender``'s
    battler at this moment: ``BtlCmd_CheckMoveHit`` (battle_script.c:6115-6135) runs
    ``BattleControllerPlayer_CheckMoveHit`` (bcp.c:217-221) for the stored attacker's battler slot
    (``future_attacker``, whatever battler is in it now) against the defender with the stored move,
    as an ordinary accuracy check (no SYSCTL flag is set at the turn's end: the last MoveEnd's
    BattleContext_Init cleared them, bcp.c:3998): its stages, abilities, items, weather, and Zoom
    Lens's ``Battler_MovedThisTurn`` true for every battler after the actions
    (battle_lib.c:5365-5368). 100 under either battler's No Guard (bcp.c:3016-3021)."""
    fmt = format_of(state)
    defender = jnp.asarray(defender, _I32)
    attacker = jnp.clip(state.future_attacker[defender].astype(_I32) - 1, 0, fmt.B - 1)
    move = state.future_move[defender].astype(_I32)
    return _rate(
        tables,
        state,
        attacker,
        move,
        defender,
        state.active[defender],
        jnp.asarray(False),
        state.stages[attacker, Stat.ACC],
        state.stages[defender, Stat.EVA],
        True,
    )


# future_thresholds' grid per defender: attacker slot (SLOTS) x defender slot (SLOTS) x stage
# total (13) x weather (4: the turn start's, none, sandstorm, hail) x the target confused or not.
_FUTURE_TOTALS = 13
_FUTURE_WEATHERS = 4


def future_thresholds(
    tables: Tables, state: BattleState, actions, moves=None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(int32[B, K], bool[B, K], bool[B])``: per defender side ``d``, every threshold its
    Future Sight hit (``future_threshold``) can have at this turn's end, which are possible, and
    ``loose``: a Trace holder is among the battlers, so the grid does not follow the abilities and
    the roll is kept whole (a class per roll). Sound, not tight: an extra threshold only splits a
    class.

    The hit reads the battlers on the field at the turn's end. The attacker's slot holds its
    active Pokemon, the one it switches in, or (a foe's Roar) any living party member; likewise
    the defender's. Their abilities and items do not change within a turn but by Trace or Transform
    (``loose``; a Gravity that is set or ends this turn is ``loose`` too).
    The stage sum takes every value 0-12 when a stage it reads can change this turn (a move with a
    stat effect on accuracy or evasion, Foresight, a White Herb); else the start stages of a battler
    that stays and 0 for a newcomer or a fainted attacker (CleanupFaintedMon). The weather can be
    the start's, none (it ended, or a lock), or a weather move's or ability's sandstorm or hail
    (the only ones the rate reads besides fog, which never starts); the target confused or not;
    and the defender locked onto by the attacker's slot or not (B8-6: the lock as it stands after
    the turn end's countdown). The attacker's Micle Berry flag is the active battler's of now, the
    roll kept whole where it can change (G-59), and so is it where an Embargo is used or up
    (B13-31)."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in range(fmt.B)])
    live = state.over == 0
    uses = live & ~is_switch(actions) & ~is_item(actions)
    effect, params = move_effects(tables, state, moves)
    slots = jnp.arange(state.hp.shape[1])
    rows = party(fmt, jnp.arange(fmt.B))  # each battler's party
    living = (state.species[rows] != 0) & (state.hp[rows] > 0)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.hp.shape[1] - 1)
    roar = uses & (effect == EffectId.ROAR)
    on_field = jnp.stack(
        [
            (slots == state.active[x])
            | (is_switch(actions[x]) & (slots == incoming[x]))
            | (roar[_foe(x)] & living[x])
            for x in range(fmt.B)
        ]
    )
    accuracy_moves = (
        (effect == EffectId.STAT_FOE)
        | (effect == EffectId.SECONDARY_STAT_FOE)
        | (effect == EffectId.STAT_SELF)
        | (effect == EffectId.SECONDARY_STAT_SELF)
    ) & ((params[:, 0] == Stat.ACC) | (params[:, 0] == Stat.EVA))
    herbs = jnp.any(on_field & (tables.item_effect[state.item[rows]] == ItemId.WHITE_HERB) & living)
    free = jnp.any(uses & (accuracy_moves | (effect == EffectId.FORESIGHT))) | herbs
    tracer = jnp.any(on_field & (state.ability[rows] == AbilityId.TRACE))
    weathers = jnp.stack(
        [state.weather.astype(_I32), Weather.NONE, Weather.SANDSTORM, Weather.HAIL]
    ).astype(_I32)
    a_mon, d_mon, total, wi, conf, lk = (
        g.reshape(-1)
        for g in jnp.meshgrid(
            slots,
            slots,
            jnp.arange(_FUTURE_TOTALS),
            jnp.arange(_FUTURE_WEATHERS),
            jnp.arange(2),
            jnp.arange(2),
            indexing="ij",
        )
    )
    # Lock-On at the hit (B8-6): MON_COND_CHECK_STATE_LOCK_ON drops the count in the turn end's
    # per-battler checks (bcp.c:1618-1624), before the side-condition pass hits (:1758-1783), so
    # a lock on the defender holds there when its count is 2 or more now, or when this turn's
    # Lock-On sets it (2, subscript_lock_on.s); a switch of the locker clears it
    # (battle_lib.c:2063-2067) and a newcomer defender starts without one. The grid's lock axis:
    # 0 no lock (always possible), 1 locked by the stored attacker's slot (possible as above).
    lock_move = jnp.any(uses & (effect == EffectId.LOCK_ON))
    thresholds, possible = [], []
    for d in range(fmt.B):
        a = jnp.clip(state.future_attacker[d].astype(_I32) - 1, 0, fmt.B - 1)
        move = state.future_move[d].astype(_I32)
        foe = _foe(a)  # the defender is the other side's battler (singles)

        def rate(am, dm, t, w, c, k, a=a, move=move, d=d):
            s = state._replace(
                active=state.active.at[a].set(am.astype(state.active.dtype)),
                lock_on=state.lock_on.at[d].set(k.astype(state.lock_on.dtype)),
                lock_on_target=state.lock_on_target.at[d].set(
                    jnp.where(k == 1, 1 + a, 0).astype(state.lock_on_target.dtype)
                ),
            )
            return _rate(
                tables, s, a, move, d, dm, jnp.asarray(False), 0, 0, True,
                weathers[w], c == 1, t, micle=(am == state.active[a]) & (state.micle[a] != 0),
            )  # fmt: skip

        thresholds.append(jax.vmap(rate)(a_mon, d_mon, total, wi, conf, lk))
        lock_ok = (lk == 0) | lock_move | (state.lock_on[d] >= 2)

        def actual(am, dm, keep_acc, a=a, d=d):
            s = state._replace(active=state.active.at[a].set(am.astype(state.active.dtype)))
            acc = jnp.where(keep_acc & (am == state.active[a]), state.stages[a, Stat.ACC], 0)
            eva = jnp.where(dm == state.active[d], state.stages[d, Stat.EVA], 0)
            return stage_total(tables, s, a, d, dm, acc, eva)

        kept = jax.vmap(actual)(a_mon, d_mon, jnp.ones_like(a_mon, bool))
        reset = jax.vmap(actual)(a_mon, d_mon, jnp.zeros_like(a_mon, bool))
        stage_ok = free | (total == kept) | (total == reset)
        possible.append(on_field[a][a_mon] & on_field[d][d_mon] & stage_ok & (foe == d) & lock_ok)
    # The Micle Berry's flag (G-59) at the hit: the grid reads the active battler's flag of now
    # (a newcomer has none). It can differ at the turn's end when the attacker's side has a flag
    # set (its own accuracy check clears it) or a Micle Berry on the field (its pinch sets it):
    # then that defender's roll is kept whole (``loose``), rare and sound.
    micle_field = jnp.stack(
        [
            (state.micle[x] != 0)
            | jnp.any(
                on_field[x] & (tables.item_effect[state.item[party(fmt, x)]] == ItemId.MICLE_BERRY)
            )
            for x in range(fmt.B)
        ]
    )
    attackers = jnp.clip(state.future_attacker.astype(_I32) - 1, 0, fmt.B - 1)
    # Embargo (B13-31): an Embargo used this turn, or a count that the turn end's
    # MON_COND_CHECK_STATE_EMBARGO (bcp.c:1671-1677) can end before the side-condition pass's
    # hit (:1758-1783), changes which items the hit reads (state.held_item): the roll is kept
    # whole, rare and sound.
    embargo_field = jnp.any(uses & (effect == EffectId.EMBARGO)) | jnp.any(state.embargo != 0)
    # Gravity at the hit (x10/6, bcp.c:2972-2974) is the turn end's: a Gravity used this turn
    # sets it, and FIELD_COND_CHECK_STATE_GRAVITY (bcp.c:1227-1235) ends a count of 1 before the
    # side-condition pass's hit (:1758-1783). A Transform used this turn copies a battler's
    # ability and stages (BtlCmd_Transform, battle_script.c:5330-5373). Either way the roll is
    # kept whole, rare and sound.
    field_change = jnp.any(
        uses & ((effect == EffectId.GRAVITY) | (effect == EffectId.TRANSFORM))
    ) | (state.gravity == 1)
    loose = (
        jnp.broadcast_to(tracer | embargo_field | field_change, (fmt.B,)) | micle_field[attackers]
    )
    return jnp.stack(thresholds).astype(_I32), jnp.stack(possible), loose


def _hit_threshold(
    tables: Tables,
    state: BattleState,
    actions,
    side,
    acc=None,
    eva=None,
    target_moved=False,
    weather=None,
    target_confused=None,
    move=None,
    overrides=True,
    micle=None,
    later_micle=None,
):
    """The turn-start threshold of ``side``'s move, or with these stages (and, for Zoom Lens,
    a target that has moved; the weather and the target's confusion at the check) instead.
    The default weather is the one after the target's switch-in, which goes first
    (``damage.entry_weather``). Nature Power's accuracy check is its called move's
    (``order.called_move``). ``move``: the move the side runs (default ``used_move`` without a
    call; an invoker's called move comes from ``hit_thresholds``' ``moves``). ``overrides``:
    as ``_rate``'s."""
    if move is None:
        move = used_move(tables, state, side, actions[side])
    target, target_switches, target_mon = _target_mon(state, actions, side)
    acc = state.stages[side, Stat.ACC] if acc is None else acc
    if eva is None:
        eva = jnp.where(fresh_switch(state, actions, target), 0, state.stages[target, Stat.EVA])
    if weather is None:
        weather = jnp.where(
            target_switches, entry_weather(tables, state, target, target_mon, side), state.weather
        )
    rate = _rate(
        tables,
        state,
        side,
        move,
        target,
        target_mon,
        target_switches,
        acc,
        eva,
        target_moved,
        weather,
        target_confused,
        overrides=overrides,
        micle=micle,
        later_micle=later_micle,
    )
    # A switch or an item use runs no move, so it has no roll.
    no_move = is_switch(actions[side]) | is_item(actions[side])
    return jnp.where(no_move, 100, rate)


def hit_threshold(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32[B]``: each side's move hits when ``RandNext() % 100 + 1 <= threshold``
    (battle_controller_player.c:2976) if nothing changes it before the move's check;
    100 or more never misses. ``hit_thresholds`` adds the changes."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    return jnp.stack([_hit_threshold(tables, state, actions, side) for side in range(fmt.B)])


# Per stage: the start value, the one change an earlier action can make, the White Herb's
# reset to 0, and the change after that reset; hit_thresholds' second axis is the
# accuracy x evasion grid, once with the target not moved yet and once moved (Zoom Lens),
# each with the weather at the start or after the foe's weather move, and the target confused
# as at the start or not (Tangled Feet).
_STAGE_VARIANTS = 4
_FIELD_VARIANTS = 4  # weather (start, the foe's move's) x target confusion (start, cleared)
HIT_VARIANTS = 2 * _STAGE_VARIANTS**2 * _FIELD_VARIANTS


def _can_go_first(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[B]``: the side acts first in some order the turn can take
    (``order.action_order`` under either ``speed_tie``)."""
    fmt = format_of(state)
    empty = empty_outcome(fmt)
    firsts = [
        action_order(tables, state, actions, empty._replace(speed_tie=jnp.int8(tie)))[0]
        for tie in (0, 1)
    ]
    return jnp.stack([(firsts[0] == side) | (firsts[1] == side) for side in range(fmt.B)])


def _stage_variants(start, changed, from_zero, changes, herb) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[4], bool[4])``: the stage ``start``; ``changed``, the stage after the foe's
    move (``hit_thresholds``), where ``changes``; 0 where a White Herb can restore a lowered
    stage (``hooks.held_item_check``); and ``from_zero``, the move's stage when the herb
    restored a lowered start before it."""
    start = start.astype(_I32)
    changed, from_zero = changed.astype(_I32), from_zero.astype(_I32)
    lowered = (start < 0) | (changes & (changed < 0))
    stages = jnp.stack([start, changed, jnp.zeros((), _I32), from_zero])
    return stages, jnp.stack(
        [jnp.asarray(True), changes, herb & lowered, herb & changes & (start < 0)]
    )


def hit_thresholds(
    tables: Tables, state: BattleState, actions, moves=None, overrides=True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``_threshold_grid`` with or without the overrides (its docstring)."""
    return _threshold_grid(tables, state, actions, moves, overrides)


def hit_threshold_pair(
    tables: Tables, state: BattleState, actions, moves=None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(thresholds, possible, later)``: ``hit_thresholds`` and, on the same grid in the same
    pass, the thresholds without the overrides (a later Triple Kick hit's, B13-6)."""
    both, possible = _threshold_grid(tables, state, actions, moves, "both")
    return both[..., 0], possible, both[..., 1]


def _threshold_grid(
    tables: Tables, state: BattleState, actions, moves, overrides
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[B, HIT_VARIANTS], bool[B, HIT_VARIANTS])``: every threshold each side's
    move can have at its accuracy check this turn, and which of them are possible.
    Column 0 is the turn-start threshold (``hit_threshold``), always possible.

    The check reads the stages of that moment (battle_controller_player.c:2873-2874).
    Between the start of the turn and a move's check, only the other side's action can
    run, and only when it can go first (``order.action_order`` under either
    ``speed_tie``). Of the implemented mechanics, it changes the threshold through:

    * the attacker's accuracy stage: a ``STAT_FOE`` / ``SECONDARY_STAT_FOE`` move on
      ``Stat.ACC`` (Sand-Attack, Mud-Slap; effects.py ``_foe_stage``);
    * the target's evasion stage: a ``STAT_SELF`` / ``SECONDARY_STAT_SELF`` move on
      ``Stat.EVA`` (Double Team; ``_stat_self``);
    * Haze (``effects._haze``, BtlCmd_ResetAllStatChanges battle_script.c:3534-3549): both
      stages to 0; Psych Up (``effects.copy_stat_stages``, BtlCmd_CopyStatStages :6042-6053):
      the target's evasion becomes the side's own; Acupressure (``effects._acupressure``,
      BtlCmd_BoostRandomStatBy2 :8932-8956, evasion among the drawn stats): the target's
      evasion +2 (B12-31); Transform (``effects._transform``, BtlCmd_Transform :5330-5373, the
      BattleMon copy up to ``ability`` includes ``statBoosts``, battle_mon.h:75, :84): the target's
      evasion becomes the side's own, as Psych Up's;
    * a White Herb on either battler, which resets a lowered stage to 0;
    * the order: a Zoom Lens (``HOLD_EFFECT_ACCURACY_UP_SLOWER``) raises the rate against a
      target that has moved, i.e. when the side acts second (the second half of the grid,
      possible for a holder that can act second; the first half stays possible, so column 0,
      the turn-start threshold, always is);
    * the weather (Sand Veil, Snow Cloak): the foe's weather move (``EffectId.WEATHER``,
      ``effects._weather``) sets its weather first; a switch-in weather ability is in the
      turn-start threshold (``damage.entry_weather``);
    * the foe's field move, the second weather variant (one move has one effect): its weather
      move's weather; its Gravity (``effects._gravity``), FIELD_CONDITION_GRAVITY's x10/6
      (battle_controller_player.c:2972-2974); its Transform into the side's battler, which then
      holds the side's ability (Sand Veil, Snow Cloak, Tangled Feet, Simple, Unaware, No Guard,
      Cloud Nine as the target, :2876-2947, bcp.c:3016-3021) and runs its switch-in checks again
      (step._use_move, the announce flags cleared :5350-5358), so a copied weather ability sets
      its weather (``hooks.weather_ability``, battle_lib.c:3740-3790);
    * the target's confusion (Tangled Feet): a confused foe moving first can snap out of it
      or be cured by its Persim / Lum Berry (over-approximate: whenever it is confused and can
      go first); an unconfused one can eat a flavor berry it dislikes before the check (its own
      move's after-effects, its switch-in, or a Pluck of the side's berry: ``hooks``,
      ruling G-5; over-approximate: whenever it or the side holds such a berry).

    * the attacker's Micle Berry flag (G-59): its berry can trigger before the check (the
      second half of the grid, Zoom Lens's, possible for a holder: one item holds one of them).

    Each applies once: a move has one effect, and the only repeated hits are
    ``MULTI_HIT``'s and Twineedle's (``MoveFlag.HIT_TWICE``, a poison), which change no stage
    (a future multi-hit stat effect must add its repeats here).

    Switches (which go first) are in the turn-start threshold already, with the
    newcomer's switch-in checks: the weather its ability sets and the ability it traces
    (``damage.entry_weather``, ``damage.entry_ability``: a Trace newcomer facing a Sand Veil
    attacker in sand holds Sand Veil at the check). No implemented move but Gravity and Transform
    (above) changes Gravity, an ability or Foresight's flag on the target before the target's foe
    moves (an armed Trace
    holder copies the newcomer the foe's switch brings, before either moves: the view holds the
    copy, ``pivot_view``), and a faint of either battler leaves no check. The
    variants over-approximate (Keen Eye, misses and the chance of a secondary only remove
    changes): an extra threshold splits a class, never merges two. ``moves`` (``int32[B]``): the moves the sides run (an invoker's call,
    ``order.used_moves``), default ``used_move`` without a call.

    ``overrides=False``: the thresholds of a Triple Kick hit after the first (B13-6), its
    accuracy check without ``CheckMoveHitOverrides``. Between the first hit and a later one only
    the move's own hits run; of what they can change, the target's confusion (a disliked flavor
    berry eaten after a hit, ruling G-5) is the second confusion variant already, and a White
    Herb the herb variants; a stat pinch berry raises no accuracy or evasion. So the same grid
    covers them."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in range(fmt.B)])
    effect, params = move_effects(tables, state, moves)  # Secret Power's terrain effect
    stat = params[:, 0]
    delta = params[:, 1]
    changes = uses_move & _can_go_first(tables, state, actions)
    on_foe_acc = (
        changes
        & ((effect == EffectId.STAT_FOE) | (effect == EffectId.SECONDARY_STAT_FOE))
        & (stat == Stat.ACC)
    )
    on_own_eva = (
        changes
        & ((effect == EffectId.STAT_SELF) | (effect == EffectId.SECONDARY_STAT_SELF))
        & (stat == Stat.EVA)
    )
    held = tables.item_effect[held_items(state)]
    herb = held == ItemId.WHITE_HERB
    firsts = [
        action_order(tables, state, actions, empty_outcome(fmt)._replace(speed_tie=jnp.int8(t)))[0]
        for t in (0, 1)
    ]
    # Battler_MovedThisTurn(defender) (battle_lib.c:5365-5368): the side's target can have
    # moved before the side's check in some order (in singles, when the target acts first).
    target_may_have_moved = jnp.stack(
        [(firsts[0] != side) | (firsts[1] != side) for side in range(fmt.B)]
    )
    zoom = uses_move & (held == ItemId.ZOOM_LENS) & target_may_have_moved
    weather_move = changes & (effect == EffectId.WEATHER)
    # The Micle Berry's flag set before the side's check (G-59): only the side's own berry can
    # set it then (hooks.held_item_check at a pinch an earlier hit brings; a Pluck sets the
    # eater's, after the eater's own check). Over-approximate: the side's active holds one.
    # Only the side's own check clears it.
    micle_free = uses_move & (held == ItemId.MICLE_BERRY)
    # The foe's Embargo before the side's check (B13-31; effects._embargo, in singles only when
    # the foe acts first) hides the side's Wide Lens (battle_lib.c:5358, bcp.c:2955-2957): the
    # second half, which a Wide Lens holder (no Zoom Lens, no Micle Berry) never used, is its
    # threshold without the lens. A Zoom Lens or Micle holder's hidden item is the first half's
    # (no Zoom bonus, no berry); a hidden Bright Powder needs the defender embargoed before
    # the side's own move: only the side itself could do it.
    wide = held == ItemId.WIDE_LENS
    foe_embargo = foes(changes & (effect == EffectId.EMBARGO))
    thresholds, possible = [], []
    n_stage = _STAGE_VARIANTS**2
    for side in range(fmt.B):
        foe = _foe(side)
        foe_switches = is_switch(actions[foe])
        foe_fresh = fresh_switch(state, actions, foe)
        acc0 = state.stages[side, Stat.ACC].astype(_I32)
        eva0 = jnp.where(foe_fresh, 0, state.stages[foe, Stat.EVA]).astype(_I32)
        # The foe's move's one stage write each (B12-31): a stat move's delta, Haze's 0,
        # Psych Up's copy of the side's own evasion (the side has not moved: its turn-start
        # stage), Acupressure's +2 (its defender is its user in singles).
        haze, acup = effect[foe] == EffectId.HAZE, effect[foe] == EffectId.ACUPRESSURE
        # Psych Up and Transform copy the side's stages onto the foe (BtlCmd_CopyStatStages;
        # BtlCmd_Transform's BattleMon copy, statBoosts at battle_mon.h:75).
        xform = effect[foe] == EffectId.TRANSFORM
        psych = (effect[foe] == EffectId.PSYCH_UP) | xform
        acc_delta = jnp.clip(delta[foe], STAGE_MIN, STAGE_MAX).astype(_I32)
        accs, acc_ok = _stage_variants(
            acc0,
            jnp.where(haze, 0, jnp.clip(acc0 + delta[foe], STAGE_MIN, STAGE_MAX)),
            jnp.where(haze, 0, acc_delta),
            on_foe_acc[foe] | (changes[foe] & haze),
            herb[side],
        )
        eva_delta = jnp.where(acup, 2, delta[foe]).astype(_I32)
        copied = state.stages[side, Stat.EVA].astype(_I32)
        eva_changed = jnp.where(
            haze, 0, jnp.where(psych, copied, jnp.clip(eva0 + eva_delta, STAGE_MIN, STAGE_MAX))
        )
        eva_from_zero = jnp.where(
            haze, 0, jnp.where(psych, copied, jnp.clip(eva_delta, STAGE_MIN, STAGE_MAX))
        )
        evas, eva_ok = _stage_variants(
            eva0,
            eva_changed,
            eva_from_zero,
            on_own_eva[foe] | (changes[foe] & (haze | psych | acup)),
            herb[foe] & ~foe_fresh,
        )
        _, _, target_mon = _target_mon(state, actions, side)
        weather0 = jnp.where(
            foe_switches, entry_weather(tables, state, foe, target_mon, side), state.weather
        )
        # The foe's field move (the second variant): a weather move's weather (another move's
        # param 0 is no weather); a Transform's copied weather ability, which its switch-in checks
        # set (hooks.weather_ability); Gravity; the side's ability on the transformed foe.
        side_ability = state.ability[party(fmt, side), state.active[side]]
        copied_weather = ability_weather(side_ability)
        moved_weather = jnp.where(
            effect[foe] == EffectId.WEATHER,
            stat[foe],
            jnp.where(xform & (copied_weather != Weather.NONE), copied_weather, weather0),
        )
        weathers = jnp.stack([weather0, moved_weather.astype(weather0.dtype)])
        gravity_move = effect[foe] == EffectId.GRAVITY
        field_move = weather_move[foe] | (changes[foe] & (gravity_move | xform))
        weather_ok = jnp.stack([jnp.asarray(True), field_move])
        field_grid = jnp.tile(jnp.repeat(jnp.array([False, True]), 2 * n_stage), 2)
        confused0 = ~foe_fresh & (state.confusion[foe] > 0)
        # The second variant is the other confusion value: snapping out (confused now), or a
        # disliked flavor berry eaten before the check (not confused now), ruling G-5.
        berry = dislikes_berry(
            tables, state, foe, target_mon, state.item[party(fmt, foe), target_mon]
        ) | dislikes_berry(
            tables, state, foe, target_mon, state.item[party(fmt, side), state.active[side]]
        )
        confuseds = jnp.stack([confused0, ~confused0])
        confused_ok = jnp.stack(
            [jnp.asarray(True), jnp.where(confused0, changes[foe], live & berry)]
        )
        # Grid order: moved (2) x weather (2) x confusion (2) x accuracy (4) x evasion (4).
        acc_grid = jnp.tile(jnp.repeat(accs, _STAGE_VARIANTS), 8)
        eva_grid = jnp.tile(evas, 8 * _STAGE_VARIANTS)
        conf_grid = jnp.tile(jnp.repeat(confuseds, n_stage), 4)
        weather_grid = jnp.tile(jnp.repeat(weathers, 2 * n_stage), 2)
        moved_grid = jnp.repeat(jnp.array([False, True]), 4 * n_stage)
        # The Micle flag (G-59) shares the "moved" half with Zoom Lens: one held item holds one
        # of them, and the target having moved changes nothing but Zoom Lens. The first half
        # reads the flag as at the start; the second half, the target moved, also the flag a
        # Micle holder's berry can set before the check.
        holder = held[side] == ItemId.MICLE_BERRY
        micle_grid = (state.micle[side] != 0) | (moved_grid & holder)
        # A later Triple Kick hit ("both"'s second column): the first hit's check cleared the
        # flag (bcp.c:2968), so only the holder's berry eaten since sets it (B13-6, G-59).
        later_grid = moved_grid & holder if overrides == "both" else None
        hide_grid = moved_grid & wide[side]

        def point(
            a,
            e,
            m,
            w,
            c,
            k,
            lk,
            hide,
            f,
            side=side,
            foe=foe,
            xf=xform,
            grav=gravity_move,
            copied=side_ability,
        ):
            embargo = jnp.where(hide, 5, state.embargo[side]).astype(state.embargo.dtype)
            foe_mon = state.active[foe]
            ability = jnp.where(f & xf, copied, state.ability[party(fmt, foe), foe_mon])
            gravity = jnp.where(f & grav, GRAVITY_TURNS, state.gravity)
            hidden = state._replace(
                embargo=state.embargo.at[side].set(embargo),
                ability=state.ability.at[party(fmt, foe), foe_mon].set(
                    ability.astype(state.ability.dtype)
                ),
                gravity=gravity.astype(state.gravity.dtype),
            )
            return _hit_threshold(
                tables, hidden, actions, side, a, e, m, w, c, moves[side], overrides, k, lk
            )

        thresholds.append(
            jax.vmap(point)(
                acc_grid,
                eva_grid,
                moved_grid,
                weather_grid,
                conf_grid,
                micle_grid,
                later_grid,
                hide_grid,
                field_grid,
            )
        )
        stage_ok = jnp.repeat(acc_ok, _STAGE_VARIANTS) & jnp.tile(eva_ok, _STAGE_VARIANTS)
        field_ok = jnp.repeat(weather_ok, 2) & jnp.tile(confused_ok, 2)
        grid_ok = (field_ok[:, None] & stage_ok[None, :]).reshape(-1)
        second = zoom[side] | micle_free[side] | (uses_move[side] & wide[side] & foe_embargo[side])
        possible.append(jnp.concatenate([grid_ok, grid_ok & second]))
    return jnp.stack(thresholds).astype(_I32), jnp.stack(possible)


def hit_class(u, thresholds, possible) -> jnp.ndarray:
    """The class value of accuracy roll ``u`` (1-100) among the ``possible`` entries of
    ``thresholds`` (last axis; ``u`` broadcasts against the others): the smallest roll
    that hits or misses alike at each of them, or 0 when ``u`` misses at all of them
    (module docstring). Thresholds are clipped to 0-100: above 100 hits like 100."""
    u = jnp.asarray(u, _I32)[..., None]
    t = jnp.clip(thresholds, 0, 100)
    below = possible & (t < u)
    low = jnp.max(jnp.where(below, t, 0), axis=-1) + 1
    misses_all = jnp.all(below | ~possible, axis=-1)
    return jnp.where(misses_all, 0, low)


def side_crit_odds(tables: Tables, state: BattleState, side, move) -> jnp.ndarray:
    """``int32``: ``side``'s ``move`` crits 1 in N with ``state`` as it is when the move runs
    (``BattleSystem_CalcCriticalMulti``, battle_lib.c:7105-7133, reads the attacker's held item,
    species, statusVolatile and ability at its ``CalcCrit``): ``HIGH_CRIT`` (+1, the script's
    criticalStage), Focus Energy (+2, :7123), the crit items (``ItemId.CRIT_ITEM``: +1, or +2 on
    their species, :7124-7128), Super Luck (+1, :7126), capped at 4 (:7130)."""
    fmt = format_of(state)
    stage = (tables.move_effect[move] == EffectId.HIGH_CRIT).astype(_I32)
    # Razor Wind, Sky Attack: BTLVAR_CRITICAL_BOOSTS + 1 before CalcCrit (effect_script_0039.s:17,
    # _0075.s:16; MoveFlag.CRIT_UP), B10-12; Blaze Kick, Poison Tail, Cross Poison (_0200.s:5,
    # _0209.s:5), B13-2.
    stage = stage + ((tables.move_flags[move].astype(_I32) & MoveFlag.CRIT_UP) != 0).astype(_I32)
    mon = state.active[side]
    # battle_lib.c:7126: + (attackerAbility == ABILITY_SUPER_LUCK), the battler's own ability.
    stage = stage + (state.ability[party(fmt, side), mon] == AbilityId.SUPER_LUCK).astype(_I32)
    # :7123: + 2 for the attacker's VOLATILE_CONDITION_FOCUS_ENERGY (B12-4).
    stage = stage + 2 * (state.focus_energy[side] != 0).astype(_I32)
    # :7124-7128: + 1 for HOLD_EFFECT_CRITRATE_UP (Scope Lens, Razor Claw), + 2 for Lucky Punch on
    # Chansey and Stick on Farfetch'd (Battler_HeldItem: ``state.held_item``, B13-29;
    # ids.ItemId.CRIT_ITEM).
    item = held_item(state, side, mon)
    species = state.species[party(fmt, side), mon].astype(_I32)
    needs = tables.item_param[item, 1].astype(_I32)
    held = tables.item_effect[item] == ItemId.CRIT_ITEM
    stage = (
        stage + jnp.where(held & (needs == 0), 1, 0) + jnp.where(held & (needs == species), 2, 0)
    )
    return _CRIT_ODDS[jnp.minimum(stage, 4)]


def crit_odds(tables: Tables, state: BattleState, actions, moves=None) -> jnp.ndarray:
    """``int32[B]``: each side's move crits 1 in N with ``state`` (``side_crit_odds``), the
    turn-start odds. Lucky Chant on the defending side and the target's Battle Armor / Shell
    Armor (``armored``) are applied by ``sample_outcome``. Nature Power's are its called move's
    (``CalcCrit`` runs in the called move's script); ``moves`` as ``hit_thresholds``'."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in range(fmt.B)])
    return jnp.stack([side_crit_odds(tables, state, side, moves[side]) for side in range(fmt.B)])


# The crit draw ``RandNext() % sCriticalStageRates[stage]`` (battle_lib.c:7132) is taken when the
# move runs, with the odds of that moment (B12-4 review). ``Outcome.crit`` holds one draw r in
# [0, CRIT_DRAWS) = lcm(16, 8, 4, 3, 2), which serves every rate (uniform by the % n convention),
# as r + 1 reduced to its class over the odds the hit can meet (``crit_class``); 0 never crits.
# The hit crits when ``(value - 1) % odds == 0`` with the odds at the hit (``crit_lands``): a
# hand-built 1 always crits.
CRIT_DRAWS = 48


def _clear_focus_energy(state: BattleState) -> BattleState:
    """Haze's write to the crit inputs: BtlCmd_ResetAllStatChanges clears every battler's
    VOLATILE_CONDITION_FOCUS_ENERGY (battle_script.c:3534-3549; ``effects._haze``)."""
    return state._replace(focus_energy=jnp.zeros_like(state.focus_energy))


def _remove_held_item(state: BattleState) -> BattleState:
    """Knock Off's write to the crit inputs: BtlCmd_TryKnockOff sets the defender's heldItem to
    ITEM_NONE (battle_script.c:6608-6632; ``effects._knock_off``), here every battler's own."""
    fmt = format_of(state)
    battlers = jnp.arange(state.active.shape[0])
    return state._replace(item=state.item.at[party(fmt, battlers), state.active].set(0))


def _embargo_all(state: BattleState) -> BattleState:
    """Embargo's write to the crit inputs: subscript_embargo_start.s:14 sets the defender's
    embargoTurns (5), which hides its held item from CalcCrit's Battler_HeldItem (battle_lib.c:
    7117; ``effects._embargo``, B13-31), here every battler's own."""
    return state._replace(embargo=jnp.full_like(state.embargo, 5))


# The move effects that can change another battler's crit inputs before its CalcCrit
# (BattleSystem_CalcCriticalMulti, battle_lib.c:7105-7132, reads the attacker's Focus Energy,
# held item, species and ability), each with its write to those inputs. Every battler's odds are
# read with every subset of these writes applied (``crit_variants``): a new effect that sets or
# clears another battler's Focus Energy, or changes its held item, species or ability, joins
# them with its write (tests/engine/test_batch12.py derives them from the effect handlers).
CRIT_ODDS_EFFECTS = {
    EffectId.HAZE: _clear_focus_energy,
    EffectId.KNOCK_OFF: _remove_held_item,
    EffectId.EMBARGO: _embargo_all,
}
# The odds variants of ``crit_variants``: variant v applies the writes of the effects whose bit
# is set in v (in ``CRIT_ODDS_EFFECTS`` order); variant 0 is the turn start.
CRIT_VARIANTS = 2 ** len(CRIT_ODDS_EFFECTS)


def crit_variants(
    tables: Tables, state: BattleState, actions, moves=None, before=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[B, CRIT_VARIANTS], bool[B, CRIT_VARIANTS])``: the crit odds each battler's move
    can have at its CalcCrit this turn, and which can occur. Variant v has the writes of the
    ``CRIT_ODDS_EFFECTS`` entries in v applied; it can occur when, for each of them, a battler
    that can act before this one's hit (``before[b, o]``: battler ``o`` may act before ``b``'s
    hit; default every other battler) uses that effect and the write alone changes ``b``'s odds.
    Over-approximate: that battler may move later, miss, or aim elsewhere."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in range(fmt.B)])
    n = state.active.shape[0]
    if before is None:
        before = ~jnp.eye(n, dtype=bool)
    before = jnp.asarray(before, bool)
    writes = tuple(CRIT_ODDS_EFFECTS.values())
    odds, singles = [], []
    for v in range(CRIT_VARIANTS):
        s = state
        for e, write in enumerate(writes):
            if v >> e & 1:
                s = write(s)
        odds.append(crit_odds(tables, s, actions, moves))
    base = odds[0]
    uses_move = (state.over == 0) & ~is_switch(actions) & ~is_item(actions)
    effect = tables.move_effect[moves]
    for e, eff in enumerate(CRIT_ODDS_EFFECTS):
        used = uses_move & (effect == eff)  # [B], by the user
        reached = jnp.any(before & used[None, :], axis=1)  # [B], by the battler hit after
        singles.append(reached & (odds[1 << e] != base))
    possible = []
    for v in range(CRIT_VARIANTS):
        ok = jnp.ones(base.shape, bool)
        for e in range(len(writes)):
            if v >> e & 1:
                ok = ok & singles[e]
        possible.append(ok)
    return jnp.stack(odds, axis=1), jnp.stack(possible, axis=1)


def crit_class(value, odds, possible) -> jnp.ndarray:
    """The ``Outcome.crit`` value of draw ``value`` (r in [0, CRIT_DRAWS)) among the ``possible``
    entries of ``odds`` (last axis; ``value`` broadcasts against the others): the smallest
    r' + 1 that crits or not alike at each of them, or 0 when r crits at none of them."""
    r = jnp.asarray(value, _I32)[..., None]
    odds = jnp.maximum(jnp.asarray(odds, _I32), 1)
    possible = jnp.asarray(possible, bool)
    key = (r % odds == 0) & possible  # [..., V]
    rs = jnp.arange(CRIT_DRAWS, dtype=_I32)
    keys = ((rs[:, None] % odds[..., None, :]) == 0) & possible[..., None, :]  # [..., R, V]
    same = jnp.all(keys == key[..., None, :], axis=-1)
    return jnp.where(jnp.any(key, axis=-1), jnp.argmax(same, axis=-1) + 1, 0).astype(_I32)


def crit_lands(tables: Tables, state: BattleState, side, move, value) -> jnp.ndarray:
    """``bool``: ``Outcome.crit`` value ``value`` crits ``side``'s ``move`` at its CalcCrit with
    ``state`` (``side_crit_odds`` of that moment, battle_lib.c:7132): ``(value - 1) % odds == 0``;
    0 never. Lucky Chant and Battle Armor / Shell Armor are the caller's."""
    value = jnp.asarray(value, _I32)
    odds = side_crit_odds(tables, state, side, move)
    return (value > 0) & ((value - 1) % odds == 0)


def armored(tables: Tables, state: BattleState, actions, moves=None) -> jnp.ndarray:
    """``bool[B]``: each side's move meets a Battle Armor or Shell Armor holder as it sees it
    (``Battler_IgnorableAbility``, battle_lib.c:7135-7136): the foe's active Pokemon or the one
    it switches in, which goes first. No crit then (``BattleSystem_CalcCriticalMulti``).

    A foe that uses Transform this turn (``moves``: the move each action runs, as
    ``crit_odds``') may hold the side's own ability at the side's CalcCrit: ``BtlCmd_Transform``
    copies the BattleMon up to ``ability`` (battle_script.c:5342-5345, battle_mon.h:84). It is
    armored then only when the side's own ability is an armor too (over-approximate: the
    Transform may move later or fail, so either ability can be the one read)."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in range(fmt.B)])
    uses = (state.over == 0) & ~is_switch(actions) & ~is_item(actions)
    xform = uses & (tables.move_effect[jnp.asarray(moves, _I32)] == EffectId.TRANSFORM)
    out = []
    for side in range(fmt.B):
        target, switches, target_mon = _target_mon(state, actions, side)
        own_ability = state.ability[party(fmt, side), state.active[side]]
        mold = own_ability == AbilityId.MOLD_BREAKER
        # A newcomer holds its entry ability, the traced copy included (ruling B3-19).
        own = jnp.where(
            switches,
            entry_ability(tables, state, target, target_mon, side),
            state.ability[party(fmt, target), target_mon],
        )
        armor = (own == AbilityId.SHELL_ARMOR) | (own == AbilityId.BATTLE_ARMOR)
        copied = (own_ability == AbilityId.SHELL_ARMOR) | (own_ability == AbilityId.BATTLE_ARMOR)
        out.append(~mold & armor & (~xform[_foe(side)] | copied))
    return jnp.stack(out)


def read_chance(tables: Tables, moves) -> jnp.ndarray:
    """``int32`` per move of ``moves``: the ``effectChance`` the battle reads,
    ``tables.move_effect_chance`` (the ROM's move row), 0 for an ``ids.ON_HIT_EFFECTS`` move,
    whose script never reads it (battle_lib.c:1524-1529)."""
    moves = jnp.asarray(moves, _I32)
    on_hit = jnp.isin(tables.move_effect[moves], jnp.asarray(ON_HIT_EFFECTS, _I32))
    return jnp.where(on_hit, 0, tables.move_effect_chance[moves].astype(_I32))


def secondary_chance(tables: Tables, state: BattleState, moves) -> jnp.ndarray:
    """``int32[B]``: each side's ``effectChance`` (``tables.move_effect_chance``), doubled for
    a Serene Grace attacker (``Battler_Ability``): ``BattleSystem_TriggerSecondaryEffect``
    (battle_lib.c:1552-1556, :1571-1575) and ``BtlCmd_CheckEffectActivation`` (battle_script.c:
    8369-8373); 0 (an ``ON_HIT`` effect, ``read_chance``) stays 0. A roll
    ``RandNext() % 100 < chance``."""
    fmt = format_of(state)
    chance = read_chance(tables, moves)
    grace = state.ability[party(fmt, jnp.arange(fmt.B)), state.active] == AbilityId.SERENE_GRACE
    return jnp.where(grace, chance * 2, chance)


def _magnitude_level(rand) -> jnp.ndarray:
    """Magnitude's level 4-10 from ``rand % 100``: <5 -> 4, <15 -> 5, <35 -> 6, <65 -> 7,
    <85 -> 8, <95 -> 9, else 10 (battle_script.c:5820-5852)."""
    return 4 + jnp.searchsorted(jnp.array([5, 15, 35, 65, 85, 95], _I32), rand, side="right")


# BtlCmd_Present (battle_script.c:5789-5806): rnd = RandNext() & 0xFF (:5793), then
# rnd < 255 * 40 / 100 -> power 40, < 255 * 70 / 100 -> 80, < 255 * 80 / 100 -> 120, else
# the heal path (:5795-5803). Integer C arithmetic: the cuts are 102, 178 and 204, so the
# law is (102, 76, 26, 52) / 256, exact (a power-of-two mask, not a modulo).
PRESENT_CUTS = (255 * 40 // 100, 255 * 70 // 100, 255 * 80 // 100)
assert len(PRESENT_CUTS) == PRESENT_HEAL  # values 0-2 are powers, PRESENT_HEAL the heal


def present_kind(rand) -> jnp.ndarray:
    """``Outcome.present`` from ``rnd = RandNext() & 0xFF``: 0, 1, 2 (power 40, 80, 120) or
    ``PRESENT_HEAL`` (``PRESENT_CUTS``)."""
    return jnp.searchsorted(jnp.array(PRESENT_CUTS, _I32), rand, side="right")


# sProtectSuccessRate (battle_script.c:5128-5133), indexed by protectSuccessTurns (0-3):
# BtlCmd_TryProtection succeeds when ``rate >= RandNext()`` (:5167), a u16, so with
# probability (rate + 1) / 65536 = 1, 1/2, 1/4, 1/8 exactly (a threshold, not a modulo).
PROTECT_RATES = (0xFFFF, 0x7FFF, 0x3FFF, 0x1FFF)


# BattleSystem_CanWhirlwind (battle_lib.c:3072-3086): a user of a lower level than its target
# passes when ``(((RandNext() & 0xFF) * (La + Ld)) >> 8) + 1 > Ld / 4``: one of 256 equally
# likely values (a mask, not a modulo: exact).
ROAR_LEVEL_VALUES = 256


def roar_level_passes(rand, attacker_level, defender_level) -> jnp.ndarray:
    """``bool``: the level check passes for ``rand = RandNext() & 0xFF`` (C integer division
    for ``Ld / 4``; ``BattleMon.level`` is a u8)."""
    la = jnp.asarray(attacker_level, _I32)
    ld = jnp.asarray(defender_level, _I32)
    return ((jnp.asarray(rand, _I32) * (la + ld)) >> 8) + 1 > ld // 4


def roar_level_weight(attacker_level, defender_level) -> jnp.ndarray:
    """``int32``: how many of the 256 values of ``RandNext() & 0xFF`` pass the level check
    (``roar_level_passes``), whatever the levels (the check runs only when ``La < Ld``)."""
    rand = jnp.arange(ROAR_LEVEL_VALUES, dtype=_I32)
    la = jnp.asarray(attacker_level, _I32)[..., None]
    ld = jnp.asarray(defender_level, _I32)[..., None]
    return jnp.sum(roar_level_passes(rand, la, ld), axis=-1).astype(_I32)


def roar_bench(state: BattleState, side, mon) -> jnp.ndarray:
    """``bool[SLOTS]``: the slots of ``side``'s party Roar can drag in while ``mon`` is
    active: ``BtlCmd_TryWhirlwind`` (battle_script.c:5285-5311) takes a slot with a species,
    not an egg (no egg battles here) and HP left, other than the active one."""
    fmt = format_of(state)
    slots = jnp.arange(state.hp.shape[1])
    return (
        (state.species[party(fmt, side)] != 0) & (state.hp[party(fmt, side)] > 0) & (slots != mon)
    )


def roar_slot(eligible, rank) -> jnp.ndarray:
    """``int32``: the party slot of rank ``rank`` (clipped to the last) among the ``eligible``
    slots in party order (``Outcome.roar_target``)."""
    count = jnp.sum(eligible.astype(_I32))
    rank = jnp.clip(jnp.asarray(rank, _I32), 0, jnp.maximum(count - 1, 0))
    ranks = jnp.cumsum(eligible.astype(_I32)) - 1
    return jnp.argmax(eligible & (ranks == rank)).astype(_I32)


def roar_levels(state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[B], int32[B])``: each side's level and its Roar target's."""
    fmt = format_of(state)
    la, ld = [], []
    for side in range(fmt.B):
        target, _, target_mon = _target_mon(state, actions, side)
        la.append(state.level[party(fmt, side), state.active[side]].astype(_I32))
        ld.append(state.level[party(fmt, target), target_mon].astype(_I32))
    return jnp.stack(la), jnp.stack(ld)


def roar_inputs(state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[B], int32[B])``: per side, as the side's Roar would meet them, the level
    check's passing weight out of ``ROAR_LEVEL_VALUES`` (``roar_level_weight`` of the user
    and its target: the foe's active Pokemon or the one it switches in, which goes first)
    and the number of the target side's eligible bench Pokemon (``roar_bench``). A bench
    Pokemon cannot lose HP during the turn and a switch only swaps the active one with one
    of them, so the count at the turn start is the count when Roar runs."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    la, ld = roar_levels(state, actions)
    counts = []
    for side in range(fmt.B):
        target, _, target_mon = _target_mon(state, actions, side)
        counts.append(jnp.sum(roar_bench(state, target, target_mon).astype(_I32)))
    return roar_level_weight(la, ld), jnp.stack(counts)


def pivot_view(tables: Tables, state: BattleState, actions) -> tuple[BattleState, jnp.ndarray]:
    """``(state, actions)`` the laws and the sampler read (E2 batch 6, rulings B6-n). A state
    paused at a U-turn's or Baton Pass's party list (``pivot_battler``) resumes with
    ``actions[pivot]`` the switch to the picked slot: its draws are those of a turn in which the
    pivoting side switches to it first, with the switch already applied to the battler (the
    newcomer on the field, its stat stages and ``VOLATILE_FIELDS`` reset as UpdateAfterSwitch and
    InitBattleMon do, a Baton Pass keeping the stages and ``BATON_PASSED_FIELDS``, battle_lib.c:
    97-101, :2078-2079, :2107; ruling B6-10), the battle's movePrev the paused move's moveTemp
    (the tail's UpdateMoveBuffers writes it before a later actor's Copycat reads it, bcp.c:
    3936-3942) and the other side's action as chosen (``ts_actions``) while it has not acted
    (``ts_acted``), else a switch to its own active Pokemon (no move). Its switch-in effects
    (weather, Trace, Intimidate) are still a switch's: ``fresh_switch`` tells the laws that its
    stages and volatiles are the view's, not a fresh battler's. The paused action's own King's
    Rock read is ``enumerate.outcome_laws``'. Any other state: unchanged, but for an armed
    Trace holder whose foe enters this turn, which holds the copy (``_armed_trace``, G-25)."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    paused = state.pivot_battler != 0
    pivot = jnp.clip(state.pivot_battler.astype(_I32) - 1, 0, fmt.B - 1)
    sides = jnp.arange(fmt.B)
    acted = state.ts_acted != 0
    other = jnp.where(
        acted, MOVE_ACTIONS + state.active.astype(_I32), state.ts_actions.astype(_I32)
    )
    viewed = jnp.where(sides == pivot, actions, other)
    slot = jnp.clip(actions[pivot] - MOVE_ACTIONS, 0, state.hp.shape[1] - 1)
    baton = tables.move_effect[state.ts_used] == EffectId.BATON_PASS

    def reset(name):
        arr = getattr(state, name)
        clear = ~baton if name in (*BATON_PASSED_FIELDS, "stages") else True
        return arr.at[pivot].set(jnp.where(clear, jnp.zeros_like(arr[pivot]), arr[pivot]))

    seen = state._replace(
        active=state.active.at[pivot].set(slot.astype(state.active.dtype)),
        move_prev=state.ts_move[pivot].astype(state.move_prev.dtype),
        **{name: reset(name) for name in (*VOLATILE_FIELDS, "stages")},
    )
    view = jax.tree.map(lambda a, b: jnp.where(paused, a, b), seen, state)
    viewed = jnp.where(paused, viewed, actions)
    return _armed_trace(tables, view, viewed, paused, pivot), viewed


def _armed_trace(tables: Tables, state: BattleState, actions, paused, pivot) -> BattleState:
    """The laws' view of an armed Trace holder (ruling G-25). A Trace holder facing a foe whose
    ability ``ChooseTraceTarget`` skips (Trace, battle_lib.c:7750-7775) keeps its
    ``traceAnnounced`` clear (:3726-3733), so ``SWITCH_IN_CHECK_STATE_TRACE`` copies the next
    battler its side faces at that newcomer's switch-in checks (``hooks.switch_in_effects``):
    mid-turn, before the holder's own move. So where side ``d``'s foe enters this turn (its
    switch action, or the paused pivot's pick) and ``d`` stays, ``d``'s active Pokemon holds
    the copy here when ``damage.trace_copy`` passes on the newcomer once its entry hazards left
    it HP (the check reads the target's ``curHP``, :3730; a switch action's newcomer has no
    Ingrain, a Baton Pass's keeps it). The newcomer's own Trace cannot copy the holder's (Trace),
    and no other switch-in state of the newcomer changes an ability, so this is its ability.
    Not where ``d``'s Pursuit fires on the switch: that hit comes before the switch
    (``step.pursuit_fires``, subscript_switch_pokemon.s:11) and ends ``d``'s action, so its
    reads see Trace; a paused pivot's Pursuit has already run."""
    fmt = format_of(state)
    from nuzline.engine.step import pursuit_fires  # step imports this module

    sides = jnp.arange(fmt.B)
    enters = jnp.where(paused, sides == pivot, is_switch(actions))
    picks = [selected_move(tables, state, side, actions[side]) for side in range(fmt.B)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    ability = state.ability
    for side in range(fmt.B):
        foe = _foe(side)
        slot = jnp.clip(actions[foe] - MOVE_ACTIONS, 0, state.hp.shape[1] - 1)
        ingrained = jnp.where(paused, state.ingrain[foe] != 0, False)
        magnet = jnp.where(paused, state.magnet_rise[foe] != 0, False)  # Baton Pass keeps it too
        loss = hazards_loss(tables, state, foe, slot, ingrained, magnet)
        alive = state.hp[party(fmt, foe), slot].astype(_I32) - loss > 0
        stays = ~jnp.where(paused, sides[side] == pivot, is_switch(actions[side]))
        pursued = ~paused & pursuit_fires(tables, state, foe, actions, choices)
        copies = (
            enters[foe]
            & stays
            & ~pursued
            & alive
            & trace_copy(tables, state, side, foe, target=slot)
        )
        mon = state.active[side]
        new = jnp.where(
            copies, state.ability[party(fmt, foe), slot], ability[party(fmt, side), mon]
        )
        ability = ability.at[party(fmt, side), mon].set(new.astype(ability.dtype))
    return state._replace(ability=ability)


def fresh_switch(state: BattleState, actions, side) -> jnp.ndarray:
    """``bool``: ``side`` switches this turn to a fresh battler (stat stages, evasion,
    confusion and Foresight reset at its switch-in). A paused pivot's switch (``pivot_view``)
    already holds the newcomer's state, a Baton Pass's carried stages included (ruling B6-10)."""
    return is_switch(jnp.asarray(actions)[side]) & (state.pivot_battler != side + 1)


def kings_rock_chance(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[B]``: the hold param of each side's active King's Rock / Razor Fang
    (``HOLD_EFFECT_SOMETIMES_FLINCH``, ``Battler_HeldItemPower(attacker, 0)``,
    battle_controller_player.c:4633), else 0. The attacker's item cannot become one during the
    turn (no implemented path gives a King's Rock: Recycle gives back only an item a RemoveItem
    took, never a King's Rock while Fling is unsupported, and only to its user, whose move has
    run)."""
    item = held_items(state)  # Battler_HeldItemEffect (B13-29)
    holder = tables.item_effect[item] == ItemId.KINGS_ROCK
    return jnp.where(holder, tables.item_param[item, 0].astype(_I32), 0)


def _draw_calls(key, tables: Tables, state: BattleState, chosen) -> jnp.ndarray:
    """``int32[B]``: each side's invoker draw for its move ``chosen`` (``int32[B]``) read with
    ``state`` (order.invoked_move), from fold_in keys so every other field keeps its samples:
    Sleep Talk's value uniform on 0..SLEEP_TALK_VALUES - 1 (BtlCmd_TrySleepTalk's RandNext() % 4
    redraw, battle_script.c:4931-4933), Metronome's candidate by order.metronome_weights
    (BtlCmd_Metronome's RandNext() % 467 + 1 redraw, :4538-4560) as its rank among the candidates,
    Assist's RandNext() % numMoves over its list (order.assist_candidates, B10-11), fixed for the
    turn (the party's moves; the user is its active battler throughout); 0 for any other move."""
    fmt = format_of(state)
    talk = jax.random.randint(
        jax.random.fold_in(key, 25), (fmt.B,), 0, SLEEP_TALK_VALUES, dtype=_I32
    )
    ranks, assist = [], []
    for side in range(fmt.B):
        weights = metronome_weights(tables, state, side)
        cands = weights > 0
        pick = jax.random.choice(
            jax.random.fold_in(key, 32 + side),
            weights.shape[0],
            p=weights / jnp.maximum(jnp.sum(weights), 1),
        )
        ranks.append(jnp.sum(cands & (jnp.arange(weights.shape[0]) < pick)))
        count = jnp.sum(assist_candidates(tables, state, side)[1].astype(_I32))
        draw_ = jax.random.randint(jax.random.fold_in(key, 34 + side), (), 0, jnp.maximum(count, 1))
        assist.append(draw_.astype(_I32))
    kinds = call_kind(tables, chosen)
    return jnp.where(
        kinds == CALL_SLEEP_TALK,
        talk,
        jnp.where(
            kinds == CALL_METRONOME,
            jnp.stack(ranks),
            jnp.where(kinds == CALL_ASSIST, jnp.stack(assist), 0),
        ),
    ).astype(_I32)


# The fields a battler's move decides the law of (ruling B11-1): redrawn for a replaced move.
MOVE_FIELDS = ("hit", "crit", "secondary", "fang_flinch", "duration", "future_hit", "call")


# The sixth hit's column of every per-hit draw (Beat Up's sixth member, B13-9) comes from
# ``fold_in(key, SIXTH_HIT_FOLD)``, split per field (``SIXTH_HIT_FIELDS``): columns 0-4 keep the
# samples they had while MAX_HITS was 5.
SIXTH_HIT_FOLD = 42
SIXTH_HIT_FIELDS = ("hit", "crit", "crit_class", "secondary", "roll", "contact_proc",
                    "flinch_item", "focus_band")  # fmt: skip


def _per_hit(fmt, key, field_key, field: str, low, high) -> jnp.ndarray:
    """``int32[B, MAX_HITS]``: ``randint(field_key)`` for hits 0-4, as before the sixth hit
    existed, and the sixth from ``fold_in(key, SIXTH_HIT_FOLD)``'s ``field`` key."""
    first = jax.random.randint(field_key, (fmt.B, MAX_HITS - 1), low, high, dtype=_I32)
    sixth_key = jax.random.split(jax.random.fold_in(key, SIXTH_HIT_FOLD), len(SIXTH_HIT_FIELDS))
    sixth = jax.random.randint(
        sixth_key[SIXTH_HIT_FIELDS.index(field)], (fmt.B, 1), low, high, dtype=_I32
    )
    return jnp.concatenate([first, sixth], axis=1)


def _move_draws(key, tables: Tables, state: BattleState, actions, moves) -> dict:
    """The move-dependent fields (``MOVE_FIELDS`` but ``call``) for the moves the sides run
    (``moves``, ``int32[B]``), drawn from ``key`` (``sample_outcome``'s split indices 0, 1, 2, 10,
    13 and fold_ins 16, 28, 41: Fang flinch, the Future Sight hit, the crit-class draw)."""
    fmt = format_of(state)
    k = jax.random.split(key, 14)
    per_side = (fmt.B,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    future_thr, future_ok, future_loose = future_thresholds(tables, state, actions, moves)
    future_roll = jax.random.randint(jax.random.fold_in(key, 28), (fmt.B,), 1, 101, dtype=_I32)
    # The roll u = RandNext() % 100 + 1 (battle_controller_player.c:2976), as its class.
    # A Triple Kick hit after the first rolls its own check without the overrides (B13-6):
    # hit[s, k > 0] is its class over those thresholds (no other move reads it).
    thresholds, possible, later = hit_threshold_pair(tables, state, actions, moves)
    rolls = _per_hit(fmt, key, k[0], "hit", 0, 100) + 1
    hit = hit_class(rolls, thresholds[:, None, :], possible[:, None, :])
    later_hit = hit_class(rolls, later[:, None, :], possible[:, None, :])
    hit = jnp.where(jnp.arange(MAX_HITS)[None, :] == 0, hit, later_hit)
    # Lucky Chant on the defending side blocks crits (battle_lib.c:7137), as does the target's
    # Battle Armor / Shell Armor (:7135-7136).
    no_crit = (
        (state.lucky_chant[side_of(_foe(jnp.arange(fmt.B)))] != 0)
        | armored(tables, state, actions, moves)
    )[:, None]
    # The crit draw, read by step with the odds at the hit (crit_lands). Odds that cannot change
    # before the hit: RandNext() % odds == 0 as 1 / 0 (the turn-start draw). Odds that can
    # (crit_variants): one r in [0, CRIT_DRAWS) per hit, as its class over the odds the hit can
    # meet (crit_class), from a fold_in key (every other field keeps its samples).
    odds, odds_ok = crit_variants(tables, state, actions, moves)
    moving = jnp.any(odds_ok & (odds != odds[:, :1]), axis=1)[:, None]
    fixed = (_per_hit(fmt, key, k[1], "crit", 0, odds[:, :1]) == 0).astype(_I32)
    r = _per_hit(fmt, key, jax.random.fold_in(key, 41), "crit_class", 0, CRIT_DRAWS)
    crit = jnp.where(moving, crit_class(r, odds[:, None, :], odds_ok[:, None, :]), fixed)
    crit = jnp.where(no_crit, 0, crit)
    # BtlCmd_CheckEffectActivation (battle_script.c:8377): RandNext() % 100 < chance (Serene
    # Grace doubled); chance 0 means the effect applies on every hit (ids.EffectId).
    chance = secondary_chance(tables, state, moves)[:, None]
    secondary = (_per_hit(fmt, key, k[2], "secondary", 0, 100) < chance) | (chance == 0)
    # BtlCmd_Random (battle_script.c:3207) ``Random r, o`` is rand % (r + 1) + o, by the
    # acting side's move: Random 3, 2 (subscript_fall_asleep.s, subscript_confuse.s),
    # Random 3, 3 (subscript_bind_start.s, subscript_uproar.s:6), Random 2, 3
    # (subscript_taunt_start.s); TryDisable (battle_script.c:4590) stores RandNext() % 4 + 3, the
    # same 3-6 as Bind.
    effect = tables.move_effect[moves]
    three = (effect == EffectId.TRAP) | (effect == EffectId.DISABLE) | (effect == EffectId.UPROAR)
    # BtlCmd_TryEncore (battle_script.c:4742): RandNext() % 5 + 3, 3-7 (B10-9).
    duration = jnp.where(
        effect == EffectId.TAUNT,
        draw(13, per_side, 3) + 3,
        jnp.where(
            effect == EffectId.ENCORE,
            draw(10, per_side, 5) + 3,
            draw(10, per_side, 4) + jnp.where(three, 3, 2),
        ),
    )
    return {
        "hit": hit,
        "crit": crit,
        "secondary": secondary,
        # subscript_paralyze_or_flinch.s:9: the second CheckEffectActivation (battle_script.c:
        # 8377), RandNext() % 100 < chance, independent of the first (``secondary``). A fang
        # never has chance 0 (tools/build_tables.py refuses it: the decomp asserts it).
        "fang_flinch": jax.random.randint(jax.random.fold_in(key, 16), per_side, 0, 100, dtype=_I32)
        < chance[:, 0],
        "duration": duration,
        # The Future Sight hit's CheckMoveHit roll at the turn's end (bcp.c:2976, RandNext() % 100
        # + 1), as its class over future_thresholds (the raw roll where loose).
        "future_hit": jnp.where(
            future_loose, future_roll, hit_class(future_roll, future_thr, future_ok)
        ),
    }


def replacement_possible(tables: Tables, state: BattleState, actions, side, foe) -> jnp.ndarray:
    """``bool``: ``side``'s move can be replaced within the turn by ``foe``'s action (ruling
    B11-1): a Sleep Talk user whose foe uses a move (a faster Disable or Imprison can change its
    set: rulings B5-13, B7-13), an ``order.takeover_possible`` side, or an
    ``order.gravity_possible`` one (a faster Gravity changes its call set, G-67). ``foe`` is named
    by the caller (doubles rule 1; B11-11). Over-approximate."""
    actions = jnp.asarray(actions, _I32)
    uses_move = (state.over == 0) & ~is_switch(actions) & ~is_item(actions)
    chosen = selected_move(tables, state, side, actions[side])[0]
    talk = uses_move[side] & (call_kind(tables, chosen) == CALL_SLEEP_TALK) & uses_move[foe]
    return (
        talk
        | takeover_possible(tables, state, actions, side, foe)
        | gravity_possible(tables, state, actions, side, foe)
    )


def replacement_known(tables: Tables, state: BattleState) -> bool:
    """Host side, on a concrete ``state`` (any leading batch axes): whether some Pokemon of
    either party knows, or can still learn by level-up (``tables.learnset_*`` at or above its
    level: ``levelup.advance_learns`` in a playout), a move that can make
    ``replacement_possible`` true: Encore (``order.takeover_possible``'s foe move) or an invoker
    (``order.call_kind`` other than CALL_NONE: Sleep Talk, ``replacement_possible``'s own talk
    and a ``can_call_encore`` caller; Metronome, Assist and Copycat, ``can_call_encore``'s
    other callers). False means ``replacement_possible`` is false at ``state`` and at every
    state a turn or a playout reaches from it, so ``sample_outcome``'s static ``replacement``
    may be False there. A move that changes a battler's set mid-battle (Mimic, Sketch,
    Transform: unsupported, ``coverage.RELOAD_STATE_EFFECTS``) joins Encore when supported.
    The guard: tests/engine/test_batch11.py, on generated states."""
    species = np.asarray(state.species).astype(np.int64)
    learnt = np.asarray(tables.learnset_move)[species]
    reachable = np.asarray(tables.learnset_level)[species] >= np.asarray(state.level)[..., None]
    known = np.concatenate(
        [np.asarray(state.moves).ravel(), np.where(reachable, learnt, 0).ravel()]
    )
    known = np.unique(known[known != 0]).astype(np.int32)
    if known.size == 0:
        return False
    effect = np.asarray(tables.move_effect)[known]
    kinds = np.asarray(call_kind(tables, jnp.asarray(known)))
    return bool(np.any((effect == EffectId.ENCORE) | (kinds != CALL_NONE)))


def sample_outcome(key, tables: Tables, state: BattleState, actions, replacement=True) -> Outcome:
    """Draw every field from its Gen 4 probability for this state and these actions (a paused
    state: ``pivot_view``'s).

    A move replaced within the turn (ruling B11-1, the ledger's "Design: a chosen move replaced
    mid-turn"): outcome A is drawn for the turn-start prediction (``used_moves``); the turn's
    events are played on A (``step.turn_events``, only where ``replacement_possible``) and a side
    whose event fired (``enumerate._split_events``, ``enumerate.taken_events``) takes outcome B's
    move-dependent fields (``MOVE_FIELDS``), redrawn for the move it then runs: from the Sleep
    Talk set a faster Disable or Imprison left (``order.disabled_state``, ``order.sealed_state``),
    the encored move a faster Encore forced (``order.encored_state``) or, for a Sleep Talk,
    Metronome or Assist user, the set a faster Gravity left (``order.gravity_state``, its call
    redrawn from that set's law, G-67). Exact: the event is
    fixed before that side's action starts, and the redrawn fields are read only after it.
    The events and outcome B are computed only where ``replacement_possible`` holds (a
    ``lax.cond``; A otherwise, which B left unchanged there). ``replacement`` (a Python bool,
    static under ``jit``): False leaves both out of the program, for a ``state`` whose
    ``replacement_known`` is False (where the predicate is always false)."""
    fmt = format_of(state)
    # importers of this module
    from nuzline.engine.enumerate import _split_events, gravity_events, taken_events
    from nuzline.engine.step import turn_events

    state_in, actions_in = state, jnp.asarray(actions, _I32)
    state, actions = pivot_view(tables, state, actions)
    k = jax.random.split(key, 14)
    per_side = (fmt.B,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    roar_counts = roar_inputs(state, actions)[1]
    roar_la, roar_ld = roar_levels(state, actions)
    speed_tie = draw(5, (), 2)  # battle_lib.c:1445, RandNext() & 1
    chosen = jnp.stack(
        [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
    )
    call = _draw_calls(key, tables, state, chosen)
    # Nature Power and the invokers draw for their called move (order.called_move, used_moves):
    # its accuracy, crit odds, effect chance and secondary effect are the called move's. Copycat's
    # call follows this outcome's order (the foe's move before it, or the turn-start movePrev).
    tie_outcome = empty_outcome(fmt)._replace(speed_tie=speed_tie)
    first = action_order(tables, state, actions, tie_outcome)[0]
    fight = ~is_switch(actions) & ~is_item(actions)
    after_foe = jnp.stack([(first == _foe(side)) & fight[_foe(side)] for side in range(fmt.B)])
    moves = used_moves(tables, state, actions, call, after_foe)
    drawn = _move_draws(key, tables, state, actions, moves)
    # BtlCmd_SetMultiHit (battle_script.c:2645): & 3 below 2 gives 2-3, else (& 3) + 2.
    first, second = draw(3, per_side, 4), draw(4, per_side, 4)
    n_hits = jnp.where(first < 2, first + 2, second + 2)
    outcome = Outcome(
        speed_tie=speed_tie,
        hit=drawn["hit"],
        crit=drawn["crit"],
        roll=_per_hit(fmt, key, k[6], "roll", 0, 16),  # BattleSystem_CalcDamageVariance, rand % 16
        secondary=drawn["secondary"],
        contact_proc=_per_hit(fmt, key, k[7], "contact_proc", 0, 10) < 3,  # battle_lib.c:4222
        n_hits=n_hits,
        full_para=draw(8, per_side, 4) == 0,  # battle_controller_player.c:2655
        confusion_hit=draw(9, per_side, 2) == 1,  # battle_controller_player.c:2621
        duration=drawn["duration"],
        thaw=draw(11, per_side, 5) == 0,  # battle_controller_player.c:2478, % 5 != 0 stays
        # BattleControllerPlayer_CheckExtraFlinch (battle_controller_player.c:4639):
        # RandNext() % 100 < the attacker's King's Rock / Razor Fang param, per hit. From a
        # fold_in key: every field above keeps its samples.
        flinch_item=_per_hit(fmt, key, jax.random.fold_in(key, 24), "flinch_item", 0, 100)
        < kings_rock_chance(tables, state)[:, None],
        # The turn's speedRand (bcp.c:853-857, after the order is fixed), Quick Claw's
        # % (100 / param) == 0 (battle_lib.c:1278-1286): uniform 1/5 by the project's % n
        # convention (exact 13108/65536). Read by the end-of-turn order, stored for the next
        # turn (step.qc_kept). From a fold_in key: every field above keeps its samples.
        quick_claw=jax.random.randint(
            jax.random.fold_in(key, 20), per_side, 0, QUICK_CLAW_MODULUS, dtype=_I32
        )
        == 0,
        # BattleSystem_SortMonSpeedOrder (battle_lib.c:3364) on equal speed: RandNext() & 1.
        end_speed_tie=draw(12, (), 2),
        # BtlCmd_CalcMagnitude (battle_script.c:5820-5852): rand % 100 mapped to the
        # level; ShedSkin (battle_lib.c:3567-3575): RandNext() % 10 < 3. Drawn from
        # fold_in keys so every field above keeps its samples.
        magnitude=_magnitude_level(
            jax.random.randint(jax.random.fold_in(key, 14), per_side, 0, 100, dtype=_I32)
        ),
        shed_skin=jax.random.randint(jax.random.fold_in(key, 15), per_side, 0, 10, dtype=_I32) < 3,
        fang_flinch=drawn["fang_flinch"],
        present=present_kind(
            jax.random.randint(jax.random.fold_in(key, 17), per_side, 0, 256, dtype=_I32)
        ),
        # BtlCmd_TryProtection (battle_script.c:5167; Protect, Detect and Endure share it, G-65):
        # RandNext() <= the rate of the user's
        # turn-start chain (the chain cannot change before its own action; the reset of a
        # chain broken by another move is folded into protect_count, step._use_move).
        protect=jax.random.randint(jax.random.fold_in(key, 18), per_side, 0, 65536, dtype=_I32)
        <= jnp.asarray(PROTECT_RATES, _I32)[jnp.clip(state.protect_count.astype(_I32), 0, 3)],
        # CHECK_STATUS_STATE_ATTRACT (battle_controller_player.c:2669-2692): RandNext() & 1
        # nonzero moves, zero is immobilized (exact: a mask, not a modulo).
        infatuation=(
            jax.random.randint(jax.random.fold_in(key, 19), per_side, 0, 65536, dtype=_I32) & 1
        )
        == 0,
        # effect_script_0036.s ``Random 2, 3`` (BtlCmd_Random, battle_script.c:3200-3210):
        # RandNext() % 3 picks burn, freeze or paralysis (ids.TRI_ATTACK_STATUSES), uniform by
        # the project's % n convention (exact 21846/21845/21845 of 65536).
        tri_attack=jax.random.randint(jax.random.fold_in(key, 21), per_side, 0, 3, dtype=_I32),
        # BattleSystem_CanWhirlwind (battle_lib.c:3072-3086): RandNext() & 0xFF against the
        # levels (roar_level_passes; exact, a mask), read only when the user's level is lower.
        roar_level=roar_level_passes(
            jax.random.randint(jax.random.fold_in(key, 22), per_side, 0, 256, dtype=_I32),
            roar_la,
            roar_ld,
        ),
        # BtlCmd_TryWhirlwind (battle_script.c:5303-5311) repeats RandNext() % partyCount
        # until it picks an eligible bench slot: uniform over them by the project's % n
        # convention (exact: each slot i weighs #{u16 r : r % partyCount == i}, e.g. 10923
        # for slots 0-3 and 10922 for 4-5 of a party of 6). Drawn as the rank (roar_slot).
        roar_target=jax.random.randint(
            jax.random.fold_in(key, 23), per_side, 0, jnp.maximum(roar_counts, 1), dtype=_I32
        ),
        call=call,
        future_hit=drawn["future_hit"],
        # Yawn's sleep at the turn end: subscript_fall_asleep.s ``Random 3, 2`` (BtlCmd_Random,
        # battle_script.c:3200-3210: rand % 4 + 2). From a fold_in key.
        yawn_sleep=jax.random.randint(jax.random.fold_in(key, 26), per_side, 0, 4, dtype=_I32) + 2,
        # BtlCmd_BoostRandomStatBy2 (battle_script.c:8950): RandNext() % validStats, the count
        # read at the move (1-7): one draw in [0, 420) (lcm(1..7)) serves every count, uniform
        # by the % n convention (B12-9).
        acupressure=jax.random.randint(
            jax.random.fold_in(key, 36), per_side, 0, ACUPRESSURE_DRAWS, dtype=_I32
        ),
        # Focus Band (bcp.c:3387, battle_script.c:9110): RandNext() % 100 < param, uniform by the
        # % n convention, per hit and for the Future Sight hit. From fold_in keys (G-1).
        focus_band=_per_hit(fmt, key, jax.random.fold_in(key, 29), "focus_band", 0, 100)
        < FOCUS_BAND_PARAM,
        future_band=jax.random.randint(jax.random.fold_in(key, 30), per_side, 0, 100, dtype=_I32)
        < FOCUS_BAND_PARAM,
        # A flavor berry's confusion: subscript_confuse.s:34 ``Random 3, 2`` (rand % 4 + 2). From a
        # fold_in key (G-5).
        berry_confusion=jax.random.randint(jax.random.fold_in(key, 31), per_side, 0, 4, dtype=_I32)
        + 2,
        # Chatter's confusion (battle_script.c:8428-8430): RandNext() % 100 <= the user's chance,
        # uniform by the % n convention (G-70). fold_in 43: 42 is SIXTH_HIT_FOLD.
        chatter=jax.random.randint(jax.random.fold_in(key, 43), per_side, 0, 100, dtype=_I32)
        <= jnp.asarray(CHATTER_CHANCES, _I32)[
            jnp.clip(state.chatot_cry[party(fmt, jnp.arange(fmt.B))].astype(_I32), 0, 2)
        ],
    )
    outcome = _cast(outcome)

    if not replacement:
        return outcome

    # B11-1: which replacement each side's action met (the events on A), then outcome B: one
    # alternative per side, the one that fired (at most one in singles: the foe runs one move
    # before it), its move from a hypothetical state where only that side's set changed (Copycat
    # reads the foe's turn-start moveTemp). A side whose event did not fire keeps A's fields.
    def fired(outcome):
        events = turn_events(tables, state_in, actions_in, outcome)
        disabled, sealed = _split_events(state_in, events)
        kinds = call_kind(tables, chosen)
        listing = (kinds == CALL_SLEEP_TALK) | (kinds == CALL_METRONOME) | (kinds == CALL_ASSIST)
        landed = gravity_events(state_in, events) & listing
        which = jnp.stack([disabled, sealed, taken_events(state_in, events), landed])
        sides = jnp.arange(fmt.B)
        taken = which[2]
        # The moveTemp each side runs: the encored move where taken. The invokers' draws do not
        # read Disable, Imprison or Encore (Sleep Talk's value is uniform; Metronome's and
        # Assist's lists ignore them), so one draw serves those alternatives; Gravity shrinks
        # Metronome's and Assist's lists, so its alternative draws from the set it leaves.
        own = jnp.where(taken, state.last_move.astype(_I32), chosen)
        alt_key = jax.random.fold_in(key, 40)
        heavy = gravity_state(state)
        alt_call = jnp.where(
            landed,
            _draw_calls(alt_key, tables, heavy, own),
            _draw_calls(alt_key, tables, state, own),
        )
        alt_moves = []
        for side in range(fmt.B):
            mine = sides == side
            hyps = (
                disabled_state(state, mine),
                sealed_state(state, mine),
                encored_state(tables, state, mine),
                heavy,
            )
            options = [used_moves(tables, h, actions, alt_call, after_foe, own)[side] for h in hyps]
            alt_moves.append(
                jnp.where(
                    which[3, side],
                    options[3],
                    jnp.where(
                        which[2, side],
                        options[2],
                        jnp.where(which[1, side], options[1], options[0]),
                    ),
                )
            )
        fields = _move_draws(alt_key, tables, state, actions, jnp.stack(alt_moves))
        fields["call"] = alt_call
        replaced = jnp.any(which, axis=0)
        for name, value in fields.items():
            old = getattr(outcome, name)
            mask = replaced.reshape((fmt.B,) + (1,) * (old.ndim - 1))
            outcome = outcome._replace(
                **{name: jnp.where(mask, jnp.asarray(value).astype(old.dtype), old)}
            )
        return outcome

    possible = jnp.stack(
        [replacement_possible(tables, state, actions, sd, _foe(sd)) for sd in range(fmt.B)]
    )
    return jax.lax.cond(jnp.any(possible), fired, lambda o: o, outcome)


# The Outcome fields stored as int16 (the rest are int8).
INT16_FIELDS = ("call", "acupressure")
# Acupressure's draw values: lcm(1..7), every count of stats below +6 dividing it (B12-9).
ACUPRESSURE_DRAWS = 420


def _cast(outcome: Outcome) -> Outcome:
    """``outcome`` with every field at its stored dtype (``INT16_FIELDS`` int16, the rest
    int8)."""
    return Outcome(
        *(
            jnp.asarray(f).astype(jnp.int16 if name in INT16_FIELDS else _I8)
            for name, f in zip(Outcome._fields, outcome, strict=True)
        )
    )
