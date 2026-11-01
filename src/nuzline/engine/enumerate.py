"""Exact chance nodes: every ``Outcome`` field's law for one state and action pair,
and their joint enumeration (Stage B spec §1; plan ruling 3).

``outcome_laws`` (``jnp``, fixed shapes, jit/vmap-able) returns the integer
parameters of each field's law, exactly as ``outcome.sample_outcome`` draws it
(outcome.py:143-193), and ``relevant``: the elements ``step`` can read this turn.
``enumerate_outcomes`` (host, numpy) lists the product of the relevant elements'
supports with float64 probabilities computed from those integers; every other
element keeps its ``empty_outcome`` value. Values of probability 0 are left out.

Relevance (notes/stage-b/outcomes.md §3, §7) may over-approximate: a row that turns
out not to matter steps to the same state as its neighbours and is merged after
``step``, so the masks only need to be sound, never tight. They are judged on the
turn-start state; an element is relevant when it can be read under some earlier event
of the turn:

* ``hit[s, 0]``: ``s`` uses a move (not a switch or item) with accuracy != 0
  (step.py:320, :329). Only hit index 0 is read (a multi-hit move checks accuracy on
  its first hit, effects.py docstring). A threshold of 100 or more gives the single
  value 1.
* ``crit[s, i]``, ``roll[s, i]``: a damaging move (status moves and power 0 deal 0,
  damage.py:254), index 0, or 0-4 for ``MULTI_HIT`` (effects.py:123-127). ``roll[s, 0]``
  also when a confusion self-hit is possible (step.py:296). The Pursuit pre-switch hit
  reads ``[pursuer, 0]`` (step.py:464-474), a move action too.
* ``secondary[s, i]``: as ``crit``, for a move with chance != 0 whose effect reads
  ``proc`` (effects.py:468: the two secondary stat effects, secondary status, flinch,
  Bind).
* ``contact_proc[d, i]``: the attacker's damaging contact move against a Static or
  Poison Point holder on side ``d``: its active Pokemon (Pursuit hits the outgoing
  one) or the one it switches in (effects.py:343-365).
* ``n_hits[s]``: ``MULTI_HIT``.
* ``full_para[s]`` / ``thaw[s]``: ``s`` moves and is paralyzed / frozen at the start,
  or the foe uses a move that inflicts it (step.py:277-278, :305).
* ``confusion_hit[s]``: ``s`` moves and its confusion counter is >= 2, or the foe uses
  a ``CONFUSE_FOE`` move (counter 2-5; step.py:292-294).
* ``duration[s]``: ``s``'s move is Taunt, Bind, ``CONFUSE_FOE`` or inflicts sleep
  (effects.py:189, :286, :297, :307).
* ``speed_tie``: two actions of one kind, equal priority and equal effective speeds
  at the start (order.py:67-85), exactly as ``action_order`` decides.
* ``end_speed_tie``: the effective speeds after the actions can be equal
  (step.py:663-673): each side's possible end speeds cover every Speed stage when a
  stage change is possible this turn, paralysis when it can be inflicted, and the
  neutral speed of a fainted battler (``_faint_check`` resets stages and status).
* ``magnitude[s]``: ``s`` uses a ``MAGNITUDE`` move (``BtlCmd_CalcMagnitudePower``).
  ``crit``/``roll`` are not relevant for ``LEVEL_DAMAGE`` (Night Shade deals its user's
  level with no crit or roll, ``damage.fixed_damage``).
* ``flinch_item``, ``quick_claw``: never read. ``shed_skin``: not read yet (no
  ability uses it), so never relevant.
* Nothing is relevant once the battle is over (``step`` returns the state, step.py:700).
"""

from __future__ import annotations

import math
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.damage import STATUS
from nuzline.engine.ids import AbilityId, EffectId, MoveFlag, Stat, Status
from nuzline.engine.order import (
    MOVE_ACTIONS,
    effective_speed,
    is_item,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome, crit_odds, empty_outcome, hit_threshold
from nuzline.engine.state import STAGE_MAX, STAGE_MIN, BattleState

_I32 = jnp.int32
_E = EffectId

# Pre-merge outcome combinations per (state, action pair); above it the pair is
# sampling-only (plan ruling 3).
WIDTH_CAP = 131072

# ``OutcomeLaws.duration_kind``: the acting side's ``Random`` command (outcome.py:166-174).
DURATION_DEFAULT = 0  # Random 3, 2: 2-5 (sleep, confusion and every other move)
DURATION_TRAP = 1  # Random 3, 3: 3-6 (Bind)
DURATION_TAUNT = 2  # Random 2, 3: 3-5

# Effects whose hits read ``outcome.secondary`` through ``proc`` (effects.py:468).
_PROC_EFFECTS = (
    _E.SECONDARY_STAT_SELF,
    _E.SECONDARY_STAT_FOE,
    _E.SECONDARY_STATUS,
    _E.FLINCH,
    _E.TRAP,
)


class OutcomeLaws(NamedTuple):
    """Integer law parameters of one (state, action pair), per side, and the relevance
    mask. Fixed shapes; a batch adds a leading axis to every field."""

    hit_threshold: jnp.ndarray  # int32[2], outcome.hit_threshold
    crit_odds: jnp.ndarray  # int32[2], outcome.crit_odds (1 in N)
    no_crit: jnp.ndarray  # bool[2], Lucky Chant on the defending side (outcome.py:157)
    secondary_chance: jnp.ndarray  # int32[2], tables.move_effect_chance of the move
    duration_kind: jnp.ndarray  # int32[2], DURATION_*
    relevant: Outcome  # bool, each field shaped like the ``Outcome`` field


class SamplingOnly(Exception):
    """The action pair cannot be enumerated exactly; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _any_of(x, values) -> jnp.ndarray:
    out = jnp.zeros(jnp.shape(x), bool)
    for v in values:
        out = out | (x == v)
    return out


def _inflicts(tables: Tables, moves, status) -> jnp.ndarray:
    """The move inflicts ``status`` on the foe (``STATUS_FOE`` / ``SECONDARY_STATUS``
    with ``move_param[m, 0]``, effects.py:186-208)."""
    effect = tables.move_effect[moves]
    status_effect = (effect == _E.STATUS_FOE) | (effect == _E.SECONDARY_STATUS)
    return status_effect & (tables.move_param[moves, 0] == status)


def _holds(state: BattleState, actions, abilities) -> jnp.ndarray:
    """``bool[2]``: side ``d``'s active Pokemon, or the one it switches in this turn,
    has one of ``abilities``."""
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.ability.shape[1] - 1)
    sides = jnp.arange(2)
    now = _any_of(state.ability[sides, state.active], abilities)
    later = _any_of(state.ability[sides, incoming], abilities) & is_switch(actions)
    return now | later


def _end_speeds(state: BattleState, actions, side, stage_free, para_free):
    """``(int32[27], bool[27])``: the effective speeds ``side`` can have after the
    actions (``end_of_turn_order`` on the post-action state, step.py:663-673) and which
    of them are possible. Grid: Speed stage -6..6 x paralyzed or not of the Pokemon
    active after the actions, plus its neutral speed (a faint resets stages and
    status, step.py:477-510). A stage other than the start one needs ``stage_free``,
    a paralysis state other than the start one ``para_free``."""
    switching = is_switch(actions[side])
    mon = jnp.where(switching, actions[side] - MOVE_ACTIONS, state.active[side]).astype(
        state.active.dtype
    )
    stage0 = jnp.where(switching, 0, state.stages[side, Stat.SPE].astype(_I32))
    para0 = state.status[side, mon] == Status.PARALYSIS
    stages = jnp.repeat(jnp.arange(STAGE_MIN, STAGE_MAX + 1, dtype=_I32), 2)
    paras = jnp.tile(jnp.array([False, True]), STAGE_MAX - STAGE_MIN + 1)
    stages = jnp.concatenate([stages, jnp.zeros((1,), _I32)])
    paras = jnp.concatenate([paras, jnp.zeros((1,), bool)])

    def speed(stage, para):
        s = state._replace(
            active=state.active.at[side].set(mon),
            stages=state.stages.at[side, Stat.SPE].set(stage.astype(state.stages.dtype)),
            status=state.status.at[side, mon].set(
                jnp.where(para, Status.PARALYSIS, Status.NONE).astype(state.status.dtype)
            ),
        )
        return effective_speed(s, side)

    speeds = jax.vmap(speed)(stages, paras)
    possible = (stage_free | (stages == stage0)) & (para_free | (paras == para0))
    fainted = jnp.arange(stages.shape[0]) == stages.shape[0] - 1
    return speeds, possible | fainted


def outcome_laws(tables: Tables, state: BattleState, actions) -> OutcomeLaws:
    """The law parameters and relevance mask of ``step(tables, state, actions, .)``
    (module docstring)."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    sides = jnp.arange(2)
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    switch, item = is_switch(actions), is_item(actions)
    uses_move = live & ~switch & ~item  # bool[2]
    foe_moves = uses_move[::-1]  # the other side uses a move
    effect = tables.move_effect[moves]
    foe_move = moves[::-1]
    chance = tables.move_effect_chance[moves].astype(_I32)
    damaging = (tables.move_category[moves] != STATUS) & (tables.move_power[moves] != 0)
    rolled = damaging & (effect != _E.LEVEL_DAMAGE)  # LEVEL_DAMAGE: no crit, no roll
    contact = (tables.move_flags[moves] & MoveFlag.CONTACT) != 0
    multi = effect == _E.MULTI_HIT
    status = state.status[sides, state.active]

    hit_index = jnp.arange(MAX_HITS)[None, :]
    lands = uses_move[:, None] & ((hit_index == 0) | multi[:, None])  # [2,5]
    hit = uses_move[:, None] & (tables.move_accuracy[moves] != 0)[:, None] & (hit_index == 0)
    crit = lands & rolled[:, None]
    confusable = uses_move & (
        (state.confusion >= 2) | (foe_moves & (tables.move_effect[foe_move] == _E.CONFUSE_FOE))
    )
    roll = crit | (confusable[:, None] & (hit_index == 0))
    secondary = lands & ((chance != 0) & _any_of(effect, _PROC_EFFECTS))[:, None]
    holders = _holds(state, actions, (AbilityId.STATIC, AbilityId.POISON_POINT))
    contact_proc = (lands & (damaging & contact & holders[::-1])[:, None])[::-1]
    full_para = uses_move & (
        (status == Status.PARALYSIS) | (foe_moves & _inflicts(tables, foe_move, Status.PARALYSIS))
    )
    thaw = uses_move & (
        (status == Status.FREEZE) | (foe_moves & _inflicts(tables, foe_move, Status.FREEZE))
    )
    duration = uses_move & (
        _any_of(effect, (_E.TAUNT, _E.TRAP, _E.CONFUSE_FOE))
        | _inflicts(tables, moves, Status.SLEEP)
    )

    # order.action_order: priority only between two moves; a tie only between two
    # actions of the same kind.
    priority = jnp.where(uses_move[0] & uses_move[1], tables.move_priority[moves].astype(_I32), 0)
    same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
    speed_tie = (
        live
        & same_kind
        & (priority[0] == priority[1])
        & (effective_speed(state, 0) == effective_speed(state, 1))
    )

    param0 = tables.move_param[moves, 0]
    own_speed = _any_of(effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF)) & (param0 == Stat.SPE)
    foe_speed = _any_of(effect, (_E.STAT_FOE, _E.SECONDARY_STAT_FOE)) & (param0 == Stat.SPE)
    # subscript_flinched.s:10-14: a flinched Steadfast Pokemon gains +1 Speed.
    steadfast = _holds(state, actions, (AbilityId.STEADFAST,)) & foe_moves
    steadfast = steadfast & (tables.move_effect[foe_move] == _E.FLINCH)
    stage_free = (uses_move & own_speed) | (foe_moves & foe_speed[::-1]) | steadfast
    static = _holds(state, actions, (AbilityId.STATIC,))
    para_free = (foe_moves & _inflicts(tables, foe_move, Status.PARALYSIS)) | (
        uses_move & damaging & contact & static[::-1]
    )
    speeds = [
        _end_speeds(state, actions, side, stage_free[side], para_free[side]) for side in (0, 1)
    ]
    (speed0, ok0), (speed1, ok1) = speeds
    end_speed_tie = live & jnp.any(
        ok0[:, None] & ok1[None, :] & (speed0[:, None] == speed1[None, :])
    )

    never = jnp.zeros((2,), bool)
    relevant = Outcome(
        speed_tie=speed_tie,
        hit=hit,
        crit=crit,
        roll=roll,
        secondary=secondary,
        contact_proc=contact_proc,
        n_hits=uses_move & multi,
        full_para=full_para,
        confusion_hit=confusable,
        duration=duration,
        thaw=thaw,
        flinch_item=never,
        quick_claw=never,
        end_speed_tie=end_speed_tie,
        magnitude=uses_move & (effect == _E.MAGNITUDE),
        shed_skin=never,  # read by Shed Skin (a later task)
    )
    kind = jnp.where(
        effect == _E.TAUNT,
        DURATION_TAUNT,
        jnp.where(effect == _E.TRAP, DURATION_TRAP, DURATION_DEFAULT),
    )
    return OutcomeLaws(
        hit_threshold=hit_threshold(tables, state, actions),
        crit_odds=crit_odds(tables, state, actions),
        no_crit=state.lucky_chant[::-1] != 0,
        secondary_chance=chance,
        duration_kind=kind.astype(_I32),
        relevant=Outcome(*(jnp.asarray(f, bool) for f in relevant)),
    )


# Host laws: ``(values, numerators, denominator)`` of one element, all integers, the
# same draws as ``sample_outcome`` (outcome.py:143-193).
_Law = tuple[list[int], list[int], int]


def _coin(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [1, 1], 2  # RandNext() & 1, battle_lib.c:1445 and :3364; % 2, :2621


def _hit(laws: OutcomeLaws, side: int) -> _Law:
    t = min(max(int(laws.hit_threshold[side]), 0), 100)  # rand % 100 + 1 <= threshold
    return [0, 1], [100 - t, t], 100


def _crit(laws: OutcomeLaws, side: int) -> _Law:
    if bool(laws.no_crit[side]):
        return [0], [1], 1
    odds = int(laws.crit_odds[side])  # rand % odds == 0
    return [0, 1], [odds - 1, 1], odds


def _roll(laws: OutcomeLaws, side: int) -> _Law:
    return list(range(16)), [1] * 16, 16  # rand % 16


def _secondary(laws: OutcomeLaws, side: int) -> _Law:
    chance = int(laws.secondary_chance[side])
    if chance == 0:  # applies on every hit
        return [1], [1], 1
    chance = min(max(chance, 0), 100)  # rand % 100 < chance
    return [0, 1], [100 - chance, chance], 100


def _contact(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [7, 3], 10  # rand % 10 < 3


def _n_hits(laws: OutcomeLaws, side: int) -> _Law:
    return [2, 3, 4, 5], [3, 3, 1, 1], 8  # BtlCmd_SetMultiHit: 2-3 at 3/8, 4-5 at 1/8


def _full_para(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [3, 1], 4  # rand % 4 == 0


def _duration(laws: OutcomeLaws, side: int) -> _Law:
    kind = int(laws.duration_kind[side])
    if kind == DURATION_TAUNT:
        return [3, 4, 5], [1, 1, 1], 3
    low = 3 if kind == DURATION_TRAP else 2
    return list(range(low, low + 4)), [1] * 4, 4


def _thaw(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [4, 1], 5  # rand % 5 == 0


def _magnitude(laws: OutcomeLaws, side: int) -> _Law:
    # BtlCmd_CalcMagnitude (battle_script.c:5820-5852): rand % 100 -> level 4-10.
    return list(range(4, 11)), [5, 10, 20, 30, 20, 10, 5], 100


def _shed_skin(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [7, 3], 10  # RandNext() % 10 < 3, battle_lib.c:3567


def _zero(laws: OutcomeLaws, side: int) -> _Law:
    return [0], [1], 1


_LAWS = {
    "speed_tie": _coin,
    "hit": _hit,
    "crit": _crit,
    "roll": _roll,
    "secondary": _secondary,
    "contact_proc": _contact,
    "n_hits": _n_hits,
    "full_para": _full_para,
    "confusion_hit": _coin,
    "duration": _duration,
    "thaw": _thaw,
    "flinch_item": _zero,
    "quick_claw": _zero,
    "end_speed_tie": _coin,
    "magnitude": _magnitude,
    "shed_skin": _shed_skin,
}


@cache
def _empty() -> Outcome:
    return Outcome(*(np.asarray(f) for f in empty_outcome()))


def _factors(laws: OutcomeLaws) -> list[tuple[str, tuple[int, ...], _Law]]:
    """Each relevant element with its law, values of probability 0 dropped."""
    out = []
    for name in Outcome._fields:
        mask = np.asarray(getattr(laws.relevant, name), bool)
        for index in np.argwhere(mask) if mask.ndim else [()] * int(mask):
            index = tuple(int(i) for i in index)
            values, weights, den = _LAWS[name](laws, index[0] if index else 0)
            kept = [(v, w) for v, w in zip(values, weights) if w > 0]
            out.append((name, index, ([v for v, _ in kept], [w for _, w in kept], den)))
    return out


def enumerate_outcomes(laws: OutcomeLaws) -> tuple[Outcome, np.ndarray]:
    """Every combination of the relevant elements' values: an ``Outcome`` of numpy
    ``int8`` arrays with a leading axis ``K`` and ``float64[K]`` probabilities summing
    to 1. ``laws`` is one (unbatched) row of ``outcome_laws``. Each probability is the
    product of the elements' integer weights over the product of their denominators,
    rounded once when that denominator is below 2**53 (else a float64 product of
    the factors). Raises ``SamplingOnly`` when ``K`` exceeds ``WIDTH_CAP``."""
    laws = jax.tree.map(np.asarray, laws)
    factors = _factors(laws)
    sizes = [len(law[0]) for _, _, law in factors]
    k = math.prod(sizes)
    if k > WIDTH_CAP:
        raise SamplingOnly(
            f"{k} outcome combinations exceed the width cap {WIDTH_CAP} "
            f"({len(factors)} random elements)"
        )
    rows = np.arange(k)
    fields = {name: np.repeat(f[None], k, axis=0) for name, f in zip(Outcome._fields, _empty())}
    exact = math.prod(law[2] for _, _, law in factors) <= 2**53
    num = np.ones(k, np.int64)
    probs = np.ones(k, np.float64)
    stride = k
    for (name, index, (values, weights, den)), size in zip(factors, sizes):
        stride //= size
        pick = (rows // stride) % size
        fields[name][(slice(None), *index)] = np.asarray(values, np.int8)[pick]
        if exact:
            num *= np.asarray(weights, np.int64)[pick]
        else:
            probs *= np.asarray(weights, np.float64)[pick] / den
    if exact:
        probs = num.astype(np.float64) / float(math.prod(law[2] for _, _, law in factors))
    return Outcome(**fields), probs
