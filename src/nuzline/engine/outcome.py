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
crit odds are read from the state at the start of the turn. Rolls whose effect
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
from nuzline.engine.damage import PRESENT_HEAL, entry_ability, entry_weather, weather_suppressed
from nuzline.engine.ids import (
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
    CALL_SLEEP_TALK,
    MOVE_ACTIONS,
    action_order,
    assist_candidates,
    call_kind,
    is_item,
    is_switch,
    metronome_weights,
    move_effects,
    selected_move,
    used_move,
    used_moves,
)
from nuzline.engine.state import (
    BATON_PASSED_FIELDS,
    STAGE_MAX,
    STAGE_MIN,
    VOLATILE_FIELDS,
    BattleState,
)

MAX_HITS = 5
_I8, _I32 = jnp.int8, jnp.int32

# HitRateByStage (include/data/hit_rate_stages.h), indexed by the clamped stage sum 0..12.
_ACC_NUM = jnp.array([33, 36, 43, 50, 60, 75, 1, 133, 166, 2, 233, 133, 3], _I32)
_ACC_DEN = jnp.array([100, 100, 100, 100, 100, 100, 1, 100, 100, 1, 100, 50, 1], _I32)
# sCriticalStageRates (battle_lib.c:7097): a crit is 1 in N at stage 0..4.
_CRIT_ODDS = jnp.array([16, 8, 4, 3, 2], _I32)


class Outcome(NamedTuple):
    speed_tie: jnp.ndarray  # int8[], 0 = player first on equal speed
    hit: jnp.ndarray  # int8[2,5], index 0: the accuracy roll's class (module docstring)
    crit: jnp.ndarray  # int8[2,5]
    roll: jnp.ndarray  # int8[2,5], 0-15 (85-100 %)
    secondary: jnp.ndarray  # int8[2,5], the attacker's own secondary effect
    contact_proc: jnp.ndarray  # int8[2,5], the defender's contact ability
    n_hits: jnp.ndarray  # int8[2], 2-5
    full_para: jnp.ndarray  # int8[2]
    confusion_hit: jnp.ndarray  # int8[2]
    duration: (
        jnp.ndarray
    )  # int8[2], sleep/confusion/Bind/Taunt/Disable/Uproar counter rolled at infliction
    thaw: jnp.ndarray  # int8[2]
    flinch_item: jnp.ndarray  # int8[2,5], King's Rock / Razor Fang's flinch roll per hit
    quick_claw: jnp.ndarray  # int8[2], this turn's speedRand % 5 == 0 (Quick Claw)
    end_speed_tie: jnp.ndarray  # int8[], 0 = player first in the end-of-turn order on equal speed
    magnitude: jnp.ndarray  # int8[2], Magnitude's level 4-10 of each side's move
    shed_skin: jnp.ndarray  # int8[2], Shed Skin's RandNext() % 10 < 3
    fang_flinch: jnp.ndarray  # int8[2], Thunder Fang's flinch roll (secondary is its paralysis)
    present: jnp.ndarray  # int8[2], Present: 0/1/2 = power 40/80/120, PRESENT_HEAL = the heal
    protect: jnp.ndarray  # int8[2], 1 = the Protect / Detect roll passes (PROTECT_RATES)
    infatuation: jnp.ndarray  # int8[2], 1 = immobilized by love (RandNext() & 1 == 0)
    tri_attack: jnp.ndarray  # int8[2], Tri Attack's status 0/1/2 (ids.TRI_ATTACK_STATUSES)
    roar_level: jnp.ndarray  # int8[2], 1 = Roar's level check passes (BattleSystem_CanWhirlwind)
    roar_target: jnp.ndarray  # int8[2], Roar's pick: a rank in the eligible bench (roar_slot)
    # int16[2], an invoker's draw (order.invoked_move): Sleep Talk's 0..SLEEP_TALK_VALUES - 1,
    # Metronome's candidate rank; Copycat draws nothing.
    call: jnp.ndarray
    # int8[2], by the battler Future Sight hits at this turn's end: its CheckMoveHit roll's class
    # (``future_thresholds``; the raw roll 1-100 where ``future_loose``), read like ``hit``.
    future_hit: jnp.ndarray
    # int8[2], by the battler a Yawn puts to sleep at this turn's end: subscript_fall_asleep.s's
    # ``Random 3, 2`` (2-5 turns), a draw of its own (bcp.c:1684-1697).
    yawn_sleep: jnp.ndarray


def empty_outcome() -> Outcome:
    """Every hit lands, no crits, roll 15 (100 %), no procs, shortest durations."""
    hits, sides = jnp.zeros((2, MAX_HITS), _I8), jnp.zeros((2,), _I8)
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
        call=jnp.zeros((2,), jnp.int16),
        future_hit=sides + 1,
        yawn_sleep=sides + 2,
    )


def outcome_to_py(outcome: Outcome) -> Outcome:
    """The same ``Outcome`` with every field Python ints / nested lists (for the
    plain-Python reference engine, ``nuzline.reference``)."""
    return Outcome(*(np.asarray(field).tolist() for field in outcome))


def _target_mon(state: BattleState, actions, side):
    """The party slot ``side``'s move will hit: the other side's active Pokemon, or
    the one it switches in this turn (switches go first)."""
    target = 1 - side
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
    ``future_thresholds``' grid); with it the weather is taken as given, unsuppressed."""
    weather = state.weather if weather is None else weather
    # A pivot's newcomer (``pivot_view``) is not fresh: its view holds its own stages / volatiles.
    fresh = jnp.asarray(target_switches, bool) & (state.pivot_battler != target + 1)
    if target_confused is None:
        target_confused = ~fresh & (state.confusion[target] > 0)
    accuracy = tables.move_accuracy[move].astype(_I32)
    a_ability = state.ability[side, state.active[side]]
    own = jnp.where(
        jnp.asarray(target_switches, bool),
        entry_ability(tables, state, target, target_mon, side),
        state.ability[target, target_mon],
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
    hp = state.hp.at[side, user].set(jnp.where(exploded, 0, state.hp[side, user]))
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
    # before the Micle Berry flag (an unsupported item) and Gravity: the defender's
    # HOLD_EFFECT_ACC_REDUCE (BrightPowder, Lax Incense) * (100 - param) / 100, then the
    # attacker's HOLD_EFFECT_ACCURACY_UP (Wide Lens) and, against a target that has moved,
    # HOLD_EFFECT_ACCURACY_UP_SLOWER (Zoom Lens), each * (100 + param) / 100
    # (Battler_HeldItemEffect / Battler_HeldItemPower: Klutz and Embargo are unsupported).
    d_item = state.item[target, target_mon]
    d_power = tables.item_param[d_item, 0].astype(_I32)
    reduce = tables.item_effect[d_item] == ItemId.BRIGHT_POWDER
    rate = jnp.where(reduce, rate * (100 - d_power) // 100, rate)
    a_item = state.item[side, state.active[side]]
    a_effect = tables.item_effect[a_item]
    a_power = tables.item_param[a_item, 0].astype(_I32)
    lens = (a_effect == ItemId.WIDE_LENS) | (
        (a_effect == ItemId.ZOOM_LENS) & jnp.asarray(target_moved, bool)
    )
    rate = jnp.where(lens, rate * (100 + a_power) // 100, rate)
    rate = jnp.where(state.gravity != 0, rate * 10 // 6, rate)
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
    user_level = state.level[side, state.active[side]].astype(_I32)
    target_level = state.level[target, target_mon].astype(_I32)
    ohko_rate = jnp.where(
        user_level >= target_level,
        jnp.where(no_guard, 100, accuracy + user_level - target_level),
        0,
    )
    ohko = tables.move_effect[move] == EffectId.OHKO
    # Accuracy 0 never misses (the check returns before rolling).
    return jnp.where(ohko, ohko_rate, jnp.where((accuracy == 0) | no_guard, 100, rate))


def live_hit_threshold(
    tables: Tables, state: BattleState, side, move, target_moved=False
) -> jnp.ndarray:
    """``int32``: the threshold of ``side``'s ``move`` at its accuracy check, read from
    ``state`` as it is at that moment (``step``): the stages, abilities, items, weather and
    HP of the two active battlers then; ``target_moved``: the target has acted this turn."""
    target_mon = state.active[1 - side]
    eva = state.stages[1 - side, Stat.EVA]
    acc = state.stages[side, Stat.ACC]
    return _rate(
        tables, state, side, move, 1 - side, target_mon, jnp.asarray(False), acc, eva, target_moved
    )


def stage_total(tables: Tables, state: BattleState, side, target, target_mon, acc, eva_stage):
    """``int32``: the clamped stage sum ``6 + evaStages + accStages`` of
    ``CheckMoveHitAccuracy`` (bcp.c:2873-2898) for ``side``'s active battler against side
    ``target``'s ``target_mon`` on the field (``_rate``'s, with ``acc`` / ``eva_stage``)."""
    a_ability = state.ability[side, state.active[side]]
    own = state.ability[target, target_mon]
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
    defender = jnp.asarray(defender, _I32)
    attacker = jnp.clip(state.future_attacker[defender].astype(_I32) - 1, 0, 1)
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
    """``(int32[2, K], bool[2, K], bool[2])``: per defender side ``d``, every threshold its
    Future Sight hit (``future_threshold``) can have at this turn's end, which are possible, and
    ``loose``: a Trace holder is among the battlers, so the grid does not follow the abilities and
    the roll is kept whole (a class per roll). Sound, not tight: an extra threshold only splits a
    class.

    The hit reads the battlers on the field at the turn's end. The attacker's slot holds its
    active Pokemon, the one it switches in, or (a foe's Roar) any living party member; likewise
    the defender's. Their abilities and items do not change within a turn but by Trace (``loose``).
    The stage sum takes every value 0-12 when a stage it reads can change this turn (a move with a
    stat effect on accuracy or evasion, Foresight, a White Herb); else the start stages of a battler
    that stays and 0 for a newcomer or a fainted attacker (CleanupFaintedMon). The weather can be
    the start's, none (it ended, or a lock), or a weather move's or ability's sandstorm or hail
    (the only ones the rate reads besides fog, which never starts); the target confused or not;
    and the defender locked onto by the attacker's slot or not (B8-6: the lock as it stands after
    the turn end's countdown)."""
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
    live = state.over == 0
    uses = live & ~is_switch(actions) & ~is_item(actions)
    effect, params = move_effects(tables, state, moves)
    slots = jnp.arange(state.hp.shape[1])
    living = (state.species != 0) & (state.hp > 0)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.hp.shape[1] - 1)
    roar = uses & (effect == EffectId.ROAR)
    on_field = jnp.stack(
        [
            (slots == state.active[x])
            | (is_switch(actions[x]) & (slots == incoming[x]))
            | (roar[1 - x] & living[x])
            for x in (0, 1)
        ]
    )
    accuracy_moves = (
        (effect == EffectId.STAT_FOE)
        | (effect == EffectId.SECONDARY_STAT_FOE)
        | (effect == EffectId.STAT_SELF)
        | (effect == EffectId.SECONDARY_STAT_SELF)
    ) & ((params[:, 0] == Stat.ACC) | (params[:, 0] == Stat.EVA))
    herbs = jnp.any(on_field & (tables.item_effect[state.item] == ItemId.WHITE_HERB) & living)
    free = jnp.any(uses & (accuracy_moves | (effect == EffectId.FORESIGHT))) | herbs
    tracer = jnp.any(on_field & (state.ability == AbilityId.TRACE))
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
    for d in (0, 1):
        a = jnp.clip(state.future_attacker[d].astype(_I32) - 1, 0, 1)
        move = state.future_move[d].astype(_I32)
        foe = 1 - a  # the defender is the other side's battler (singles)

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
                weathers[w], c == 1, t,
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
    loose = jnp.broadcast_to(tracer, (2,))
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
):
    """The turn-start threshold of ``side``'s move, or with these stages (and, for Zoom Lens,
    a target that has moved; the weather and the target's confusion at the check) instead.
    The default weather is the one after the target's switch-in, which goes first
    (``damage.entry_weather``). Nature Power's accuracy check is its called move's
    (``order.called_move``). ``move``: the move the side runs (default ``used_move`` without a
    call; an invoker's called move comes from ``hit_thresholds``' ``moves``)."""
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
    )
    # A switch or an item use runs no move, so it has no roll.
    no_move = is_switch(actions[side]) | is_item(actions[side])
    return jnp.where(no_move, 100, rate)


def hit_threshold(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32[2]``: each side's move hits when ``RandNext() % 100 + 1 <= threshold``
    (battle_controller_player.c:2976) if nothing changes it before the move's check;
    100 or more never misses. ``hit_thresholds`` adds the changes."""
    actions = jnp.asarray(actions, _I32)
    return jnp.stack([_hit_threshold(tables, state, actions, side) for side in (0, 1)])


# Per stage: the start value, the one change an earlier action can make, the White Herb's
# reset to 0, and the change after that reset; hit_thresholds' second axis is the
# accuracy x evasion grid, once with the target not moved yet and once moved (Zoom Lens),
# each with the weather at the start or after the foe's weather move, and the target confused
# as at the start or not (Tangled Feet).
_STAGE_VARIANTS = 4
_FIELD_VARIANTS = 4  # weather (start, the foe's move's) x target confusion (start, cleared)
HIT_VARIANTS = 2 * _STAGE_VARIANTS**2 * _FIELD_VARIANTS


def _can_go_first(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: the side acts first in some order the turn can take
    (``order.action_order`` under either ``speed_tie``)."""
    firsts = [
        action_order(tables, state, actions, empty_outcome()._replace(speed_tie=jnp.int8(tie)))[0]
        for tie in (0, 1)
    ]
    return jnp.stack([(firsts[0] == side) | (firsts[1] == side) for side in (0, 1)])


def _stage_variants(start, delta, changes, herb) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[4], bool[4])``: the stage ``start``; ``start + delta`` (clamped to +-6, as
    ``effects.change_stage``) where ``changes``; 0 where a White Herb can restore a
    lowered stage (``hooks.held_item_check``); and ``delta`` where the herb restored a
    lowered start before the change."""
    start = start.astype(_I32)
    changed = jnp.clip(start + delta, STAGE_MIN, STAGE_MAX)
    lowered = (start < 0) | (changes & (changed < 0))
    stages = jnp.stack([start, changed, jnp.zeros((), _I32), jnp.clip(delta, STAGE_MIN, STAGE_MAX)])
    return stages, jnp.stack(
        [jnp.asarray(True), changes, herb & lowered, herb & changes & (start < 0)]
    )


def hit_thresholds(
    tables: Tables, state: BattleState, actions, moves=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[2, HIT_VARIANTS], bool[2, HIT_VARIANTS])``: every threshold each side's
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
    * a White Herb on either battler, which resets a lowered stage to 0;
    * the order: a Zoom Lens (``HOLD_EFFECT_ACCURACY_UP_SLOWER``) raises the rate against a
      target that has moved, i.e. when the side acts second (the second half of the grid,
      possible for a holder that can act second; the first half stays possible, so column 0,
      the turn-start threshold, always is);
    * the weather (Sand Veil, Snow Cloak): the foe's weather move (``EffectId.WEATHER``,
      ``effects._weather``) sets its weather first; a switch-in weather ability is in the
      turn-start threshold (``damage.entry_weather``);
    * the target's confusion (Tangled Feet): a confused foe moving first can snap out of it
      or be cured by its Persim / Lum Berry (over-approximate: whenever it is confused and can
      go first).

    Each applies once: a move has one effect, and the only repeated hits are
    ``MULTI_HIT``'s, which change no stage (a future multi-hit stat effect must add
    its repeats here).

    Switches (which go first) are in the turn-start threshold already, with the
    newcomer's switch-in checks: the weather its ability sets and the ability it traces
    (``damage.entry_weather``, ``damage.entry_ability``: a Trace newcomer facing a Sand Veil
    attacker in sand holds Sand Veil at the check). No implemented move changes Gravity, an
    ability or Foresight's flag on the target before the target's foe moves (Trace copies
    mid-turn only facing another Trace holder, which coverage refuses), and a faint of either
    battler leaves no check. The
    variants over-approximate (Keen Eye, misses and the chance of a secondary only remove
    changes): an extra threshold splits a class, never merges two. ``moves`` (``int32[2]``): the moves the sides run (an invoker's call,
    ``order.used_moves``), default ``used_move`` without a call."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
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
    held = tables.item_effect[state.item[jnp.arange(2), state.active]]
    herb = held == ItemId.WHITE_HERB
    firsts = [
        action_order(tables, state, actions, empty_outcome()._replace(speed_tie=jnp.int8(t)))[0]
        for t in (0, 1)
    ]
    # Battler_MovedThisTurn(defender) (battle_lib.c:5365-5368): the side's target can have
    # moved before the side's check in some order (in singles, when the target acts first).
    target_may_have_moved = jnp.stack(
        [(firsts[0] != side) | (firsts[1] != side) for side in (0, 1)]
    )
    zoom = uses_move & (held == ItemId.ZOOM_LENS) & target_may_have_moved
    weather_move = changes & (effect == EffectId.WEATHER)
    thresholds, possible = [], []
    n_stage = _STAGE_VARIANTS**2
    for side in (0, 1):
        foe = 1 - side
        foe_switches = is_switch(actions[foe])
        foe_fresh = fresh_switch(state, actions, foe)
        acc0 = state.stages[side, Stat.ACC]
        eva0 = jnp.where(foe_fresh, 0, state.stages[foe, Stat.EVA])
        accs, acc_ok = _stage_variants(acc0, delta[foe], on_foe_acc[foe], herb[side])
        evas, eva_ok = _stage_variants(eva0, delta[foe], on_own_eva[foe], herb[foe] & ~foe_fresh)
        _, _, target_mon = _target_mon(state, actions, side)
        weather0 = jnp.where(
            foe_switches, entry_weather(tables, state, foe, target_mon, side), state.weather
        )
        # The foe's move's weather, only for a weather move (another move's param 0 is no
        # weather): else the start weather again, which adds no threshold.
        moved_weather = jnp.where(effect[foe] == EffectId.WEATHER, stat[foe], weather0)
        weathers = jnp.stack([weather0, moved_weather.astype(weather0.dtype)])
        weather_ok = jnp.stack([jnp.asarray(True), weather_move[foe]])
        confused0 = ~foe_fresh & (state.confusion[foe] > 0)
        confuseds = jnp.stack([confused0, jnp.asarray(False)])
        confused_ok = jnp.stack([jnp.asarray(True), confused0 & changes[foe]])
        # Grid order: moved (2) x weather (2) x confusion (2) x accuracy (4) x evasion (4).
        acc_grid = jnp.tile(jnp.repeat(accs, _STAGE_VARIANTS), 8)
        eva_grid = jnp.tile(evas, 8 * _STAGE_VARIANTS)
        conf_grid = jnp.tile(jnp.repeat(confuseds, n_stage), 4)
        weather_grid = jnp.tile(jnp.repeat(weathers, 2 * n_stage), 2)
        moved_grid = jnp.repeat(jnp.array([False, True]), 4 * n_stage)
        thresholds.append(
            jax.vmap(
                lambda a, e, m, w, c, side=side: _hit_threshold(
                    tables, state, actions, side, a, e, m, w, c, moves[side]
                )
            )(acc_grid, eva_grid, moved_grid, weather_grid, conf_grid)
        )
        stage_ok = jnp.repeat(acc_ok, _STAGE_VARIANTS) & jnp.tile(eva_ok, _STAGE_VARIANTS)
        field_ok = jnp.repeat(weather_ok, 2) & jnp.tile(confused_ok, 2)
        grid_ok = (field_ok[:, None] & stage_ok[None, :]).reshape(-1)
        possible.append(jnp.concatenate([grid_ok, grid_ok & zoom[side]]))
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


def crit_odds(tables: Tables, state: BattleState, actions, moves=None) -> jnp.ndarray:
    """``int32[2]``: each side's move crits 1 in N (``BattleSystem_CalcCriticalMulti``,
    battle_lib.c:7105). ``HIGH_CRIT`` (+1) and the attacker's Super Luck (+1, :7126) raise
    the stage, and so do the crit items (``ItemId.CRIT_ITEM``); Focus Energy is not implemented. Lucky
    Chant on the defending side and the target's Battle Armor / Shell Armor (``armored``)
    are applied by ``sample_outcome``. Nature Power's are its called move's
    (``CalcCrit`` runs in the called move's script); ``moves`` as ``hit_thresholds``'."""
    actions = jnp.asarray(actions, _I32)
    if moves is None:
        moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
    stage = (tables.move_effect[moves] == EffectId.HIGH_CRIT).astype(_I32)
    # battle_lib.c:7126: + (attackerAbility == ABILITY_SUPER_LUCK), the battler's own ability.
    super_luck = state.ability[jnp.arange(2), state.active] == AbilityId.SUPER_LUCK
    stage = stage + super_luck.astype(_I32)
    # :7123-7128: + 1 for HOLD_EFFECT_CRITRATE_UP (Scope Lens, Razor Claw), + 2 for Lucky Punch on
    # Chansey and Stick on Farfetch'd (Battler_HeldItem; ids.ItemId.CRIT_ITEM).
    item = state.item[jnp.arange(2), state.active]
    species = state.species[jnp.arange(2), state.active].astype(_I32)
    needs = tables.item_param[item, 1].astype(_I32)
    held = tables.item_effect[item] == ItemId.CRIT_ITEM
    stage = (
        stage + jnp.where(held & (needs == 0), 1, 0) + jnp.where(held & (needs == species), 2, 0)
    )
    return _CRIT_ODDS[jnp.minimum(stage, 4)]


def armored(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: each side's move meets a Battle Armor or Shell Armor holder as it sees it
    (``Battler_IgnorableAbility``, battle_lib.c:7135-7136): the foe's active Pokemon or the one
    it switches in, which goes first. No crit then (``BattleSystem_CalcCriticalMulti``)."""
    actions = jnp.asarray(actions, _I32)
    out = []
    for side in (0, 1):
        target, switches, target_mon = _target_mon(state, actions, side)
        mold = state.ability[side, state.active[side]] == AbilityId.MOLD_BREAKER
        # A newcomer holds its entry ability, the traced copy included (ruling B3-19).
        own = jnp.where(
            switches,
            entry_ability(tables, state, target, target_mon, side),
            state.ability[target, target_mon],
        )
        out.append(~mold & ((own == AbilityId.SHELL_ARMOR) | (own == AbilityId.BATTLE_ARMOR)))
    return jnp.stack(out)


def secondary_chance(tables: Tables, state: BattleState, moves) -> jnp.ndarray:
    """``int32[2]``: each side's ``effectChance`` (``tables.move_effect_chance``), doubled for
    a Serene Grace attacker (``Battler_Ability``): ``BattleSystem_TriggerSecondaryEffect``
    (battle_lib.c:1552-1556, :1571-1575) and ``BtlCmd_CheckEffectActivation`` (battle_script.c:
    8369-8373); 0 (an ``ON_HIT`` effect) stays 0. A roll ``RandNext() % 100 < chance``."""
    chance = tables.move_effect_chance[moves].astype(_I32)
    grace = state.ability[jnp.arange(2), state.active] == AbilityId.SERENE_GRACE
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
    slots = jnp.arange(state.hp.shape[1])
    return (state.species[side] != 0) & (state.hp[side] > 0) & (slots != mon)


def roar_slot(eligible, rank) -> jnp.ndarray:
    """``int32``: the party slot of rank ``rank`` (clipped to the last) among the ``eligible``
    slots in party order (``Outcome.roar_target``)."""
    count = jnp.sum(eligible.astype(_I32))
    rank = jnp.clip(jnp.asarray(rank, _I32), 0, jnp.maximum(count - 1, 0))
    ranks = jnp.cumsum(eligible.astype(_I32)) - 1
    return jnp.argmax(eligible & (ranks == rank)).astype(_I32)


def roar_levels(state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[2], int32[2])``: each side's level and its Roar target's."""
    la, ld = [], []
    for side in (0, 1):
        target, _, target_mon = _target_mon(state, actions, side)
        la.append(state.level[side, state.active[side]].astype(_I32))
        ld.append(state.level[target, target_mon].astype(_I32))
    return jnp.stack(la), jnp.stack(ld)


def roar_inputs(state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(int32[2], int32[2])``: per side, as the side's Roar would meet them, the level
    check's passing weight out of ``ROAR_LEVEL_VALUES`` (``roar_level_weight`` of the user
    and its target: the foe's active Pokemon or the one it switches in, which goes first)
    and the number of the target side's eligible bench Pokemon (``roar_bench``). A bench
    Pokemon cannot lose HP during the turn and a switch only swaps the active one with one
    of them, so the count at the turn start is the count when Roar runs."""
    actions = jnp.asarray(actions, _I32)
    la, ld = roar_levels(state, actions)
    counts = []
    for side in (0, 1):
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
    Rock read is ``enumerate.outcome_laws``'. Any other state: unchanged."""
    actions = jnp.asarray(actions, _I32)
    paused = state.pivot_battler != 0
    pivot = jnp.clip(state.pivot_battler.astype(_I32) - 1, 0, 1)
    sides = jnp.arange(2)
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
    return view, jnp.where(paused, viewed, actions)


def fresh_switch(state: BattleState, actions, side) -> jnp.ndarray:
    """``bool``: ``side`` switches this turn to a fresh battler (stat stages, evasion,
    confusion and Foresight reset at its switch-in). A paused pivot's switch (``pivot_view``)
    already holds the newcomer's state, a Baton Pass's carried stages included (ruling B6-10)."""
    return is_switch(jnp.asarray(actions)[side]) & (state.pivot_battler != side + 1)


def kings_rock_chance(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[2]``: the hold param of each side's active King's Rock / Razor Fang
    (``HOLD_EFFECT_SOMETIMES_FLINCH``, ``Battler_HeldItemPower(attacker, 0)``,
    battle_controller_player.c:4633), else 0. The attacker's item cannot become one during the
    turn (no implemented path gives a King's Rock)."""
    item = state.item[jnp.arange(2), state.active]
    holder = tables.item_effect[item] == ItemId.KINGS_ROCK
    return jnp.where(holder, tables.item_param[item, 0].astype(_I32), 0)


def sample_outcome(key, tables: Tables, state: BattleState, actions) -> Outcome:
    """Draw every field from its Gen 4 probability for this state and these actions (a paused
    state: ``pivot_view``'s)."""
    state, actions = pivot_view(tables, state, actions)
    k = jax.random.split(key, 14)
    per_hit, per_side = (2, MAX_HITS), (2,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    roar_counts = roar_inputs(state, actions)[1]
    roar_la, roar_ld = roar_levels(state, actions)
    speed_tie = draw(5, (), 2)  # battle_lib.c:1445, RandNext() & 1
    # The invokers' draws (order.invoked_move), from fold_in keys so every other field keeps its
    # samples: Sleep Talk's value uniform on 0..SLEEP_TALK_VALUES - 1 (BtlCmd_TrySleepTalk's
    # RandNext() % 4 redraw, battle_script.c:4931-4933), Metronome's candidate by
    # order.metronome_weights (BtlCmd_Metronome's RandNext() % 467 + 1 redraw, :4538-4560) as its
    # rank among the candidates.
    talk = jax.random.randint(jax.random.fold_in(key, 25), (2,), 0, SLEEP_TALK_VALUES, dtype=_I32)
    ranks = []
    for side in (0, 1):
        weights = metronome_weights(tables, state, side)
        cands = weights > 0
        pick = jax.random.choice(
            jax.random.fold_in(key, 26 + side),
            weights.shape[0],
            p=weights / jnp.maximum(jnp.sum(weights), 1),
        )
        ranks.append(jnp.sum(cands & (jnp.arange(weights.shape[0]) < pick)))
    chosen = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    kinds = call_kind(tables, chosen)
    # Assist (B10-11): RandNext() % numMoves over its list (order.assist_candidates), fixed for
    # the turn (the party's moves; the user is its active battler throughout).
    assist = []
    for side in (0, 1):
        count = jnp.sum(assist_candidates(tables, state, side)[1].astype(_I32))
        draw_ = jax.random.randint(jax.random.fold_in(key, 30 + side), (), 0, jnp.maximum(count, 1))
        assist.append(draw_.astype(_I32))
    call = jnp.where(
        kinds == CALL_SLEEP_TALK,
        talk,
        jnp.where(
            kinds == CALL_METRONOME,
            jnp.stack(ranks),
            jnp.where(kinds == CALL_ASSIST, jnp.stack(assist), 0),
        ),
    )
    # Nature Power and the invokers draw for their called move (order.called_move, used_moves):
    # its accuracy, crit odds, effect chance and secondary effect are the called move's. Copycat's
    # call follows this outcome's order (the foe's move before it, or the turn-start movePrev).
    first = action_order(tables, state, actions, empty_outcome()._replace(speed_tie=speed_tie))[0]
    fight = ~is_switch(actions) & ~is_item(actions)
    after_foe = jnp.stack([(first == 1 - side) & fight[1 - side] for side in (0, 1)])
    moves = used_moves(tables, state, actions, call, after_foe)
    future_thr, future_ok, future_loose = future_thresholds(tables, state, actions, moves)
    future_roll = jax.random.randint(jax.random.fold_in(key, 28), (2,), 1, 101, dtype=_I32)
    # The roll u = RandNext() % 100 + 1 (battle_controller_player.c:2976), as its class.
    thresholds, possible = hit_thresholds(tables, state, actions, moves)
    hit = hit_class(draw(0, per_hit, 100) + 1, thresholds[:, None, :], possible[:, None, :])
    # Lucky Chant on the defending side blocks crits (battle_lib.c:7137), as does the target's
    # Battle Armor / Shell Armor (:7135-7136).
    no_crit = ((state.lucky_chant[::-1] != 0) | armored(tables, state, actions))[:, None]
    crit = (draw(1, per_hit, crit_odds(tables, state, actions, moves)[:, None]) == 0) & ~no_crit
    # BtlCmd_CheckEffectActivation (battle_script.c:8377): RandNext() % 100 < chance (Serene
    # Grace doubled); chance 0 means the effect applies on every hit (ids.EffectId).
    chance = secondary_chance(tables, state, moves)[:, None]
    secondary = (draw(2, per_hit, 100) < chance) | (chance == 0)
    # BtlCmd_SetMultiHit (battle_script.c:2645): & 3 below 2 gives 2-3, else (& 3) + 2.
    first, second = draw(3, per_side, 4), draw(4, per_side, 4)
    n_hits = jnp.where(first < 2, first + 2, second + 2)
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
    outcome = Outcome(
        speed_tie=speed_tie,
        hit=hit,
        crit=crit,
        roll=draw(6, per_hit, 16),  # BattleSystem_CalcDamageVariance, rand % 16
        secondary=secondary,
        contact_proc=draw(7, per_hit, 10) < 3,  # Static etc., battle_lib.c:4222
        n_hits=n_hits,
        full_para=draw(8, per_side, 4) == 0,  # battle_controller_player.c:2655
        confusion_hit=draw(9, per_side, 2) == 1,  # battle_controller_player.c:2621
        duration=duration,
        thaw=draw(11, per_side, 5) == 0,  # battle_controller_player.c:2478, % 5 != 0 stays
        # BattleControllerPlayer_CheckExtraFlinch (battle_controller_player.c:4639):
        # RandNext() % 100 < the attacker's King's Rock / Razor Fang param, per hit. From a
        # fold_in key: every field above keeps its samples.
        flinch_item=jax.random.randint(jax.random.fold_in(key, 24), per_hit, 0, 100, dtype=_I32)
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
        # subscript_paralyze_or_flinch.s:9: the second CheckEffectActivation (battle_script.c:
        # 8377), RandNext() % 100 < chance, independent of the first (``secondary``). A fang
        # never has chance 0 (tools/build_tables.py refuses it: the decomp asserts it).
        fang_flinch=jax.random.randint(jax.random.fold_in(key, 16), per_side, 0, 100, dtype=_I32)
        < chance[:, 0],
        present=present_kind(
            jax.random.randint(jax.random.fold_in(key, 17), per_side, 0, 256, dtype=_I32)
        ),
        # BtlCmd_TryProtection (battle_script.c:5167): RandNext() <= the rate of the user's
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
        # The Future Sight hit's CheckMoveHit roll at the turn's end (bcp.c:2976, RandNext() % 100
        # + 1), as its class over future_thresholds (the raw roll where loose). From a fold_in key.
        future_hit=jnp.where(
            future_loose,
            future_roll,
            hit_class(future_roll, future_thr, future_ok),
        ),
        # Yawn's sleep at the turn end: subscript_fall_asleep.s ``Random 3, 2`` (BtlCmd_Random,
        # battle_script.c:3200-3210: rand % 4 + 2). From a fold_in key.
        yawn_sleep=jax.random.randint(jax.random.fold_in(key, 26), per_side, 0, 4, dtype=_I32) + 2,
    )
    return Outcome(
        *(
            jnp.asarray(f).astype(jnp.int16 if name == "call" else _I8)
            for name, f in zip(Outcome._fields, outcome, strict=True)
        )
    )
