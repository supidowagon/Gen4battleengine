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
from nuzline.engine.damage import PRESENT_HEAL, entry_weather, weather_suppressed
from nuzline.engine.ids import QUICK_CLAW_MODULUS, AbilityId, EffectId, ItemId, Stat, Weather
from nuzline.engine.order import MOVE_ACTIONS, action_order, is_item, is_switch, used_move
from nuzline.engine.state import STAGE_MAX, STAGE_MIN, BattleState

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
    )  # int8[2], sleep/confusion/Bind/Taunt/Disable counter rolled at infliction
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
    target_mon,
    target_switches,
    acc,
    eva_stage,
    target_moved=False,
    weather=None,
    target_confused=None,
) -> jnp.ndarray:
    """``hitRate`` of ``side``'s ``move`` against the other side's ``target_mon`` with the
    attacker's accuracy stage ``acc`` and the target's evasion stage ``eva_stage``
    (``target_switches``: it comes in this turn, fresh); ``target_moved``: the target's action
    is over (``Battler_MovedThisTurn``, battle_lib.c:5365-5368: in singles, the attacker acts
    second), for Zoom Lens; ``weather`` (default ``state.weather``) and ``target_confused``
    (default: the target's confusion counter is nonzero, a newcomer never) as at the check.
    100 when No Guard makes the move hit (``CheckMoveHitOverrides``, bcp.c:3016-3021)."""
    weather = state.weather if weather is None else weather
    if target_confused is None:
        target_confused = ~target_switches & (state.confusion[1 - side] > 0)
    accuracy = tables.move_accuracy[move].astype(_I32)
    target = 1 - side
    a_ability = state.ability[side, state.active[side]]
    d_ability = state.ability[target, target_mon]
    d_ability = jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, d_ability)

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
    identified = ~target_switches & (state.foresight[target] != 0)
    eva = jnp.where(identified & (eva < 0), 0, eva)
    total = jnp.clip(6 + eva + acc, 0, 12)
    rate = accuracy * _ACC_NUM[total] // _ACC_DEN[total]
    # Compoundeyes (battle_controller_player.c:2922-2924), right after the stage scaling.
    rate = jnp.where(a_ability == AbilityId.COMPOUNDEYES, rate * 130 // 100, rate)
    # NO_CLOUD_NINE (:2926), with the battlers on the field when the move runs: Sand Veil
    # in sand and Snow Cloak in hail (Battler_IgnorableAbility: Mold Breaker pierces them,
    # :2927, :2931) x80/100, then deep fog x6/10 (:2934).
    on_field = state.active.at[target].set(target_mon.astype(state.active.dtype))
    # A Selfdestruct / Explosion user's HP is already 0 here (effect_script_0007.s runs
    # before the check): its own Cloud Nine / Air Lock no longer counts.
    user = state.active[side]
    exploded = tables.move_effect[move] == EffectId.SELF_DESTRUCT
    hp = state.hp.at[side, user].set(jnp.where(exploded, 0, state.hp[side, user]))
    suppressed = weather_suppressed(tables, state._replace(hp=hp), on_field)
    weather = jnp.where(suppressed, Weather.NONE, weather)
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
    no_guard = (a_ability == AbilityId.NO_GUARD) | (
        state.ability[target, target_mon] == AbilityId.NO_GUARD
    )
    # Accuracy 0 never misses (the check returns before rolling).
    return jnp.where((accuracy == 0) | no_guard, 100, rate)


def live_hit_threshold(
    tables: Tables, state: BattleState, side, move, target_moved=False
) -> jnp.ndarray:
    """``int32``: the threshold of ``side``'s ``move`` at its accuracy check, read from
    ``state`` as it is at that moment (``step``): the stages, abilities, items, weather and
    HP of the two active battlers then; ``target_moved``: the target has acted this turn."""
    target_mon = state.active[1 - side]
    eva = state.stages[1 - side, Stat.EVA]
    acc = state.stages[side, Stat.ACC]
    return _rate(tables, state, side, move, target_mon, jnp.asarray(False), acc, eva, target_moved)


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
):
    """The turn-start threshold of ``side``'s move, or with these stages (and, for Zoom Lens,
    a target that has moved; the weather and the target's confusion at the check) instead.
    The default weather is the one after the target's switch-in, which goes first
    (``damage.entry_weather``). Nature Power's accuracy check is its called move's
    (``order.called_move``)."""
    move = used_move(tables, state, side, actions[side])
    target, target_switches, target_mon = _target_mon(state, actions, side)
    acc = state.stages[side, Stat.ACC] if acc is None else acc
    if eva is None:
        eva = jnp.where(target_switches, 0, state.stages[target, Stat.EVA])
    if weather is None:
        weather = jnp.where(
            target_switches, entry_weather(tables, state, target, target_mon, side), state.weather
        )
    rate = _rate(
        tables,
        state,
        side,
        move,
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


def hit_thresholds(tables: Tables, state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    Switches (which go first) are in the turn-start threshold already. No implemented
    move changes Gravity, an ability or Foresight's flag on the target before the target's
    foe moves (Trace copies mid-turn only facing another Trace holder, which coverage
    refuses), and a faint of either battler leaves no check. The
    variants over-approximate (Keen Eye, misses and the chance of a secondary only remove
    changes): an extra threshold splits a class, never merges two."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
    effect = tables.move_effect[moves]
    stat = tables.move_param[moves, 0]
    delta = tables.move_param[moves, 1].astype(_I32)
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
        acc0 = state.stages[side, Stat.ACC]
        eva0 = jnp.where(foe_switches, 0, state.stages[foe, Stat.EVA])
        accs, acc_ok = _stage_variants(acc0, delta[foe], on_foe_acc[foe], herb[side])
        evas, eva_ok = _stage_variants(eva0, delta[foe], on_own_eva[foe], herb[foe] & ~foe_switches)
        _, _, target_mon = _target_mon(state, actions, side)
        weather0 = jnp.where(
            foe_switches, entry_weather(tables, state, foe, target_mon, side), state.weather
        )
        # The foe's move's weather, only for a weather move (another move's param 0 is no
        # weather): else the start weather again, which adds no threshold.
        moved_weather = jnp.where(effect[foe] == EffectId.WEATHER, stat[foe], weather0)
        weathers = jnp.stack([weather0, moved_weather.astype(weather0.dtype)])
        weather_ok = jnp.stack([jnp.asarray(True), weather_move[foe]])
        confused0 = ~foe_switches & (state.confusion[foe] > 0)
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
                    tables, state, actions, side, a, e, m, w, c
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


def crit_odds(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32[2]``: each side's move crits 1 in N (``BattleSystem_CalcCriticalMulti``,
    battle_lib.c:7105). ``HIGH_CRIT`` (+1) and the attacker's Super Luck (+1, :7126) raise
    the stage: Focus Energy, Scope Lens and the species items are not implemented. Lucky
    Chant on the defending side and the target's Battle Armor / Shell Armor (``armored``)
    are applied by ``sample_outcome``. Nature Power's are its called move's
    (``CalcCrit`` runs in the called move's script)."""
    actions = jnp.asarray(actions, _I32)
    moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
    stage = (tables.move_effect[moves] == EffectId.HIGH_CRIT).astype(_I32)
    # battle_lib.c:7126: + (attackerAbility == ABILITY_SUPER_LUCK), the battler's own ability.
    super_luck = state.ability[jnp.arange(2), state.active] == AbilityId.SUPER_LUCK
    stage = stage + super_luck.astype(_I32)
    return _CRIT_ODDS[jnp.minimum(stage, 4)]


def armored(state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: each side's move meets a Battle Armor or Shell Armor holder as it sees it
    (``Battler_IgnorableAbility``, battle_lib.c:7135-7136): the foe's active Pokemon or the one
    it switches in, which goes first. No crit then (``BattleSystem_CalcCriticalMulti``)."""
    actions = jnp.asarray(actions, _I32)
    out = []
    for side in (0, 1):
        target, _, target_mon = _target_mon(state, actions, side)
        mold = state.ability[side, state.active[side]] == AbilityId.MOLD_BREAKER
        own = state.ability[target, target_mon]
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


def kings_rock_chance(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[2]``: the hold param of each side's active King's Rock / Razor Fang
    (``HOLD_EFFECT_SOMETIMES_FLINCH``, ``Battler_HeldItemPower(attacker, 0)``,
    battle_controller_player.c:4633), else 0. The attacker's item cannot become one during the
    turn (no implemented path gives a King's Rock)."""
    item = state.item[jnp.arange(2), state.active]
    holder = tables.item_effect[item] == ItemId.KINGS_ROCK
    return jnp.where(holder, tables.item_param[item, 0].astype(_I32), 0)


def sample_outcome(key, tables: Tables, state: BattleState, actions) -> Outcome:
    """Draw every field from its Gen 4 probability for this state and these actions."""
    actions = jnp.asarray(actions, _I32)
    k = jax.random.split(key, 14)
    per_hit, per_side = (2, MAX_HITS), (2,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    roar_counts = roar_inputs(state, actions)[1]
    roar_la, roar_ld = roar_levels(state, actions)
    # Nature Power draws for its called move (order.called_move): its accuracy, crit odds,
    # effect chance and secondary effect are the called move's.
    moves = jnp.stack([used_move(tables, state, side, actions[side]) for side in (0, 1)])
    # The roll u = RandNext() % 100 + 1 (battle_controller_player.c:2976), as its class.
    thresholds, possible = hit_thresholds(tables, state, actions)
    hit = hit_class(draw(0, per_hit, 100) + 1, thresholds[:, None, :], possible[:, None, :])
    # Lucky Chant on the defending side blocks crits (battle_lib.c:7137), as does the target's
    # Battle Armor / Shell Armor (:7135-7136).
    no_crit = ((state.lucky_chant[::-1] != 0) | armored(state, actions))[:, None]
    crit = (draw(1, per_hit, crit_odds(tables, state, actions)[:, None]) == 0) & ~no_crit
    # BtlCmd_CheckEffectActivation (battle_script.c:8377): RandNext() % 100 < chance (Serene
    # Grace doubled); chance 0 means the effect applies on every hit (ids.EffectId).
    chance = secondary_chance(tables, state, moves)[:, None]
    secondary = (draw(2, per_hit, 100) < chance) | (chance == 0)
    # BtlCmd_SetMultiHit (battle_script.c:2645): & 3 below 2 gives 2-3, else (& 3) + 2.
    first, second = draw(3, per_side, 4), draw(4, per_side, 4)
    n_hits = jnp.where(first < 2, first + 2, second + 2)
    # BtlCmd_Random (battle_script.c:3207) ``Random r, o`` is rand % (r + 1) + o, by the
    # acting side's move: Random 3, 2 (subscript_fall_asleep.s, subscript_confuse.s),
    # Random 3, 3 (subscript_bind_start.s), Random 2, 3 (subscript_taunt_start.s);
    # TryDisable (battle_script.c:4590) stores RandNext() % 4 + 3, the same 3-6 as Bind.
    effect = tables.move_effect[moves]
    duration = jnp.where(
        effect == EffectId.TAUNT,
        draw(13, per_side, 3) + 3,
        draw(10, per_side, 4)
        + jnp.where((effect == EffectId.TRAP) | (effect == EffectId.DISABLE), 3, 2),
    )
    outcome = Outcome(
        speed_tie=draw(5, (), 2),  # battle_lib.c:1445, RandNext() & 1
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
    )
    return Outcome(*(jnp.asarray(f).astype(_I8) for f in outcome))
