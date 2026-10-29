"""Chance outcomes of one turn (spec §4.2 "Explicit chance outcomes").

``step`` draws no randomness: every random event of a turn is a field of
``Outcome``, which ``sample_outcome`` draws (Monte Carlo) and later enumeration
lists. Per-hit fields are ``[side, hit_index]`` with up to 5 hits; per-action
fields are ``[side]``. ``secondary`` is indexed by the attacking side (its own
move's secondary effect); ``contact_proc`` by the defending side (its contact
ability); ``duration`` by the acting (inflicting) side.

``sample_outcome`` reproduces each decomp roll (citations per field below). The
hit and crit odds are read from the state at the start of the turn, with one
correction known in advance: a target that switches out this turn is replaced by
the incoming Pokemon (neutral evasion stage, its own ability). Stage changes made
earlier in the same turn are not seen; no M1 move changes accuracy or evasion.
Rolls whose effect depends on state that can change mid-turn (status, the
defender's ability) are drawn unconditionally and gated by ``step``, which keeps
their probabilities exact.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import AbilityId, EffectId, Stat, Weather
from nuzline.engine.order import MOVE_ACTIONS, is_switch, selected_move
from nuzline.engine.state import BattleState

MAX_HITS = 5
_I8, _I32 = jnp.int8, jnp.int32

# HitRateByStage (include/data/hit_rate_stages.h), indexed by the clamped stage sum 0..12.
_ACC_NUM = jnp.array([33, 36, 43, 50, 60, 75, 1, 133, 166, 2, 233, 133, 3], _I32)
_ACC_DEN = jnp.array([100, 100, 100, 100, 100, 100, 1, 100, 100, 1, 100, 50, 1], _I32)
# sCriticalStageRates (battle_lib.c:7097): a crit is 1 in N at stage 0..4.
_CRIT_ODDS = jnp.array([16, 8, 4, 3, 2], _I32)


class Outcome(NamedTuple):
    speed_tie: jnp.ndarray  # int8[], 0 = player first on equal speed
    hit: jnp.ndarray  # int8[2,5]
    crit: jnp.ndarray  # int8[2,5]
    roll: jnp.ndarray  # int8[2,5], 0-15 (85-100 %)
    secondary: jnp.ndarray  # int8[2,5], the attacker's own secondary effect
    contact_proc: jnp.ndarray  # int8[2,5], the defender's contact ability
    n_hits: jnp.ndarray  # int8[2], 2-5
    full_para: jnp.ndarray  # int8[2]
    confusion_hit: jnp.ndarray  # int8[2]
    duration: jnp.ndarray  # int8[2], sleep/confusion turns rolled at infliction
    thaw: jnp.ndarray  # int8[2]
    flinch_item: jnp.ndarray  # int8[2]
    quick_claw: jnp.ndarray  # int8[2]


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
    )


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


def _hit_threshold(tables: Tables, state: BattleState, actions, side) -> jnp.ndarray:
    move, _ = selected_move(state, side, actions[side])
    accuracy = tables.move_accuracy[move].astype(_I32)
    target, target_switches, target_mon = _target_mon(state, actions, side)
    a_ability = state.ability[side, state.active[side]]
    d_ability = state.ability[target, target_mon]
    d_ability = jnp.where(a_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, d_ability)

    # BattleControllerPlayer_CheckMoveHitAccuracy (battle_controller_player.c:2865).
    acc = state.stages[side, Stat.ACC].astype(_I32)
    eva = -jnp.where(target_switches, 0, state.stages[target, Stat.EVA].astype(_I32))
    acc = jnp.where(a_ability == AbilityId.SIMPLE, acc * 2, acc)
    eva = jnp.where(d_ability == AbilityId.SIMPLE, eva * 2, eva)
    acc = jnp.where(d_ability == AbilityId.UNAWARE, 0, acc)
    eva = jnp.where(a_ability == AbilityId.UNAWARE, 0, eva)
    total = jnp.clip(6 + eva + acc, 0, 12)
    rate = accuracy * _ACC_NUM[total] // _ACC_DEN[total]
    rate = jnp.where(state.weather == Weather.FOG, rate * 6 // 10, rate)  # deep fog
    rate = jnp.where(state.gravity != 0, rate * 10 // 6, rate)
    # Accuracy 0 never misses (the check returns before rolling); a switch has no roll.
    return jnp.where(is_switch(actions[side]) | (accuracy == 0), 100, rate)


def hit_threshold(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32[2]``: each side's move hits when ``RandNext() % 100 + 1 <= threshold``
    (battle_controller_player.c:2976); 100 or more never misses."""
    actions = jnp.asarray(actions, _I32)
    return jnp.stack([_hit_threshold(tables, state, actions, side) for side in (0, 1)])


def crit_odds(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32[2]``: each side's move crits 1 in N (``BattleSystem_CalcCriticalMulti``,
    battle_lib.c:7105). Only ``HIGH_CRIT`` (+1) raises the stage: Focus Energy, Super
    Luck, Scope Lens and the species items are not implemented. Lucky Chant on the
    defending side is applied by ``sample_outcome``."""
    actions = jnp.asarray(actions, _I32)
    moves = jnp.stack([selected_move(state, side, actions[side])[0] for side in (0, 1)])
    stage = (tables.move_effect[moves] == EffectId.HIGH_CRIT).astype(_I32)
    return _CRIT_ODDS[jnp.minimum(stage, 4)]


def sample_outcome(key, tables: Tables, state: BattleState, actions) -> Outcome:
    """Draw every field from its Gen 4 probability for this state and these actions."""
    actions = jnp.asarray(actions, _I32)
    k = jax.random.split(key, 12)
    per_hit, per_side = (2, MAX_HITS), (2,)

    def draw(i, shape, n):
        """``RandNext() % n`` for each element."""
        return jax.random.randint(k[i], shape, 0, n, dtype=_I32)

    moves = jnp.stack([selected_move(state, side, actions[side])[0] for side in (0, 1)])
    threshold = hit_threshold(tables, state, actions)[:, None]
    hit = draw(0, per_hit, 100) + 1 <= threshold
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
    outcome = Outcome(
        speed_tie=draw(5, (), 2),  # battle_lib.c:1445, RandNext() & 1
        hit=hit,
        crit=crit,
        roll=draw(6, per_hit, 16),  # BattleSystem_CalcDamageVariance, rand % 16
        secondary=secondary,
        contact_proc=draw(7, per_hit, 10) < 3,  # Static / Poison Point, battle_lib.c:4222
        n_hits=n_hits,
        full_para=draw(8, per_side, 4) == 0,  # battle_controller_player.c:2655
        confusion_hit=draw(9, per_side, 2) == 1,  # battle_controller_player.c:2621
        # Random 3, 2 (subscript_fall_asleep.s, subscript_confuse.s): rand % 4 + 2.
        duration=draw(10, per_side, 4) + 2,
        thaw=draw(11, per_side, 5) == 0,  # battle_controller_player.c:2478, % 5 != 0 stays
        # King's Rock / Razor Fang and Quick Claw are not implemented items (ItemId).
        flinch_item=jnp.zeros(per_side, _I32),
        quick_claw=jnp.zeros(per_side, _I32),
    )
    return Outcome(*(jnp.asarray(f).astype(_I8) for f in outcome))
