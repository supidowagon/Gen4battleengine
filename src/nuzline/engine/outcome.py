"""Chance outcomes of one turn (spec §4.2 "Explicit chance outcomes").

``step`` draws no randomness: every random event of a turn is a field of
``Outcome``, which ``sample_outcome`` draws (Monte Carlo) and later enumeration
lists. Per-hit fields are ``[side, hit_index]`` with up to 5 hits; per-action
fields are ``[side]``. ``secondary`` is indexed by the attacking side (its own
move's secondary effect; Thunder Fang's paralysis roll); ``fang_flinch`` by the
attacking side too (Thunder Fang's second roll, its flinch); ``contact_proc`` by the
defending side (its contact ability); ``duration`` by the acting (inflicting) side, and it
is the counter value the decomp's ``Random`` command stores for that side's move
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
from nuzline.engine.damage import weather_suppressed
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat, Weather
from nuzline.engine.order import MOVE_ACTIONS, action_order, is_item, is_switch, selected_move
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
    flinch_item: jnp.ndarray  # int8[2], reserved: King's Rock / Razor Fang (unsupported items)
    quick_claw: jnp.ndarray  # int8[2], reserved: Quick Claw (unsupported item)
    end_speed_tie: jnp.ndarray  # int8[], 0 = player first in the end-of-turn order on equal speed
    magnitude: jnp.ndarray  # int8[2], Magnitude's level 4-10 of each side's move
    shed_skin: jnp.ndarray  # int8[2], Shed Skin's RandNext() % 10 < 3
    fang_flinch: jnp.ndarray  # int8[2], Thunder Fang's flinch roll (secondary is its paralysis)


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
        flinch_item=sides,
        quick_claw=sides,
        end_speed_tie=jnp.zeros((), _I8),
        magnitude=sides + 4,
        shed_skin=sides,
        fang_flinch=sides,
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
    tables: Tables, state: BattleState, side, move, target_mon, target_switches, acc, eva_stage
) -> jnp.ndarray:
    """``hitRate`` of ``side``'s ``move`` against the other side's ``target_mon`` with the
    attacker's accuracy stage ``acc`` and the target's evasion stage ``eva_stage``
    (``target_switches``: it comes in this turn, fresh)."""
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
    weather = jnp.where(suppressed, Weather.NONE, state.weather)
    veiled = ((weather == Weather.SANDSTORM) & (d_ability == AbilityId.SAND_VEIL)) | (
        (weather == Weather.HAIL) & (d_ability == AbilityId.SNOW_CLOAK)
    )
    rate = jnp.where(veiled, rate * 80 // 100, rate)
    rate = jnp.where(weather == Weather.FOG, rate * 6 // 10, rate)  # deep fog
    rate = jnp.where(state.gravity != 0, rate * 10 // 6, rate)
    # Accuracy 0 never misses (the check returns before rolling).
    return jnp.where(accuracy == 0, 100, rate)


def live_hit_threshold(tables: Tables, state: BattleState, side, move) -> jnp.ndarray:
    """``int32``: the threshold of ``side``'s ``move`` at its accuracy check, read from
    ``state`` as it is at that moment (``step``): the stages, abilities, weather and HP
    of the two active battlers then."""
    target_mon = state.active[1 - side]
    eva = state.stages[1 - side, Stat.EVA]
    acc = state.stages[side, Stat.ACC]
    return _rate(tables, state, side, move, target_mon, jnp.asarray(False), acc, eva)


def _hit_threshold(tables: Tables, state: BattleState, actions, side, acc=None, eva=None):
    """The turn-start threshold of ``side``'s move, or with these stages instead."""
    move, _ = selected_move(tables, state, side, actions[side])
    target, target_switches, target_mon = _target_mon(state, actions, side)
    acc = state.stages[side, Stat.ACC] if acc is None else acc
    if eva is None:
        eva = jnp.where(target_switches, 0, state.stages[target, Stat.EVA])
    rate = _rate(tables, state, side, move, target_mon, target_switches, acc, eva)
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
# accuracy x evasion grid.
_STAGE_VARIANTS = 4
HIT_VARIANTS = _STAGE_VARIANTS**2


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
    * a White Herb on either battler, which resets a lowered stage to 0.

    Each applies once: a move has one effect, and the only repeated hits are
    ``MULTI_HIT``'s, which change no stage (a future multi-hit stat effect must add
    its repeats here).

    Switches (which go first) are in the turn-start threshold already. No implemented
    move changes the weather, Gravity, an ability or Foresight's flag on the target
    before the target's foe moves, and a faint of either battler leaves no check. The
    variants over-approximate (Keen Eye, misses and the chance of a secondary only remove
    changes): an extra threshold splits a class, never merges two."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
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
    herb = tables.item_effect[state.item[jnp.arange(2), state.active]] == ItemId.WHITE_HERB
    thresholds, possible = [], []
    for side in (0, 1):
        foe = 1 - side
        foe_switches = is_switch(actions[foe])
        acc0 = state.stages[side, Stat.ACC]
        eva0 = jnp.where(foe_switches, 0, state.stages[foe, Stat.EVA])
        accs, acc_ok = _stage_variants(acc0, delta[foe], on_foe_acc[foe], herb[side])
        evas, eva_ok = _stage_variants(eva0, delta[foe], on_own_eva[foe], herb[foe] & ~foe_switches)
        acc_grid = jnp.repeat(accs, _STAGE_VARIANTS)
        eva_grid = jnp.tile(evas, _STAGE_VARIANTS)
        thresholds.append(
            jax.vmap(lambda a, e, side=side: _hit_threshold(tables, state, actions, side, a, e))(
                acc_grid, eva_grid
            )
        )
        possible.append(jnp.repeat(acc_ok, _STAGE_VARIANTS) & jnp.tile(eva_ok, _STAGE_VARIANTS))
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
    battle_lib.c:7105). Only ``HIGH_CRIT`` (+1) raises the stage: Focus Energy, Super
    Luck, Scope Lens and the species items are not implemented. Lucky Chant on the
    defending side is applied by ``sample_outcome``."""
    actions = jnp.asarray(actions, _I32)
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    stage = (tables.move_effect[moves] == EffectId.HIGH_CRIT).astype(_I32)
    return _CRIT_ODDS[jnp.minimum(stage, 4)]


def _magnitude_level(rand) -> jnp.ndarray:
    """Magnitude's level 4-10 from ``rand % 100``: <5 -> 4, <15 -> 5, <35 -> 6, <65 -> 7,
    <85 -> 8, <95 -> 9, else 10 (battle_script.c:5820-5852)."""
    return 4 + jnp.searchsorted(jnp.array([5, 15, 35, 65, 85, 95], _I32), rand, side="right")


def sample_outcome(key, tables: Tables, state: BattleState, actions) -> Outcome:
    """Draw every field from its Gen 4 probability for this state and these actions."""
    actions = jnp.asarray(actions, _I32)
    k = jax.random.split(key, 14)
    per_hit, per_side = (2, MAX_HITS), (2,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    # The roll u = RandNext() % 100 + 1 (battle_controller_player.c:2976), as its class.
    thresholds, possible = hit_thresholds(tables, state, actions)
    hit = hit_class(draw(0, per_hit, 100) + 1, thresholds[:, None, :], possible[:, None, :])
    # Lucky Chant on the defending side blocks crits (battle_lib.c:7137).
    no_crit = (state.lucky_chant[::-1] != 0)[:, None]
    crit = (draw(1, per_hit, crit_odds(tables, state, actions)[:, None]) == 0) & ~no_crit
    # BtlCmd_CheckEffectActivation (battle_script.c:8377): RandNext() % 100 < chance;
    # chance 0 means the effect applies on every hit (ids.EffectId).
    chance = tables.move_effect_chance[moves].astype(_I32)[:, None]
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
        # King's Rock / Razor Fang and Quick Claw are not implemented items (ItemId).
        flinch_item=jnp.zeros(per_side, _I32),
        quick_claw=jnp.zeros(per_side, _I32),
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
    )
    return Outcome(*(jnp.asarray(f).astype(_I8) for f in outcome))
