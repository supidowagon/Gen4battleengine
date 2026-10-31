"""Leaf and forced-turn evaluation for the Stage B tree (plan rulings 5, 6 and 8): Stage A
playouts from the tree's leaves, in chunks of one fixed padded shape.

``evaluate_leaves`` plays ``playout.batched_playout`` from each leaf state (a player
decision point after the opponent's replacement, so the policy chooses from the first
turn; ``max_turns`` counts from the leaf). ``evaluate_forced`` evaluates a fixed (player,
opponent) action pair of a state, as the solver needs for a pair it cannot enumerate
(sampling-only) and for pruned opponent actions: one turn with the pair, its outcome
drawn by ``sample_outcome``, then ``send_replacements`` with the playout's ``ai_switched``
and ``replacement_power_mul`` (``playout._play``, playout.py:177-186), then ``playout``
with ``max_turns - 1`` turns (a battle already over is a win iff the player won it).

Every leaf has two halves with independent key streams (``leaf_keys``): half A picks the
action at the inner belief nodes, half B gives every reported value and interval (ruling
5, the max-bias fix). A refused playout (``PlayoutResult.refused``) has no result: any
refused playout marks its leaf ``refused``, and the solver values it 0 (ruling 4); it is
never counted as a loss.

Programs run in chunks of ``LEAF_CHUNK`` rows padded to ``-(-LEAF_CHUNK // devices) *
devices`` (padding = copies of the chunk's row 0, dropped after the call), sharded over
the host devices and exported through ``cached_export``, as ``expand`` does, so a solve
compiles each program once per device count and the results do not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState/PlayoutResult
from nuzline.data.tables import Tables
from nuzline.engine.order import is_switch
from nuzline.engine.outcome import sample_outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import replacement_power_mul, step
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.playout import REFUSED_WON, PlayoutResult, playout, send_replacements

__all__ = [
    "LEAF_CHUNK",
    "LeafValue",
    "allocate",
    "evaluate_forced",
    "evaluate_leaves",
    "leaf_keys",
]

LEAF_CHUNK = 4096  # playouts per program call, before padding to the device count

_I32 = jnp.int32


@dataclass(frozen=True)
class LeafValue:
    """Wins and playouts of the two halves of one evaluated state. ``refused``: some
    playout met a refused mechanic (the leaf has no value; ruling 4)."""

    wins_a: int
    n_a: int
    wins_b: int
    n_b: int
    refused: bool

    @property
    def p_a(self) -> float:
        return self.wins_a / self.n_a

    @property
    def p_b(self) -> float:
        return self.wins_b / self.n_b

    @property
    def var_a(self) -> float:
        """``var_b`` on half A: ``p~ (1 - p~) / n_a`` with ``p~ = (k_a + 1) / (n_a + 2)``.
        Only the solver's stopping rule reads it (precision target; ruling 5 keeps half B
        out of every decision)."""
        p = (self.wins_a + 1) / (self.n_a + 2)
        return p * (1 - p) / self.n_a

    @property
    def var_b(self) -> float:
        """``p~ (1 - p~) / n_b`` with ``p~ = (k_b + 1) / (n_b + 2)`` (ruling 6)."""
        p = (self.wins_b + 1) / (self.n_b + 2)
        return p * (1 - p) / self.n_b


def leaf_keys(key, i: int, n: int):
    """``(keys_a, keys_b)``, ``n // 2`` keys each, of leaf ``i``: half ``h`` splits
    ``fold_in(fold_in(key, i), h)``."""
    base = jax.random.fold_in(key, i)
    return tuple(jax.random.split(jax.random.fold_in(base, h), n // 2) for h in (0, 1))


def allocate(weights: np.ndarray, budget: int, floor: int = 1) -> np.ndarray:
    """Playouts per evaluated state (ruling 8): ``2 * max(floor, round(budget * w / sum(w)
    / 2))`` for chance weights ``w`` (all ``floor`` when they sum to 0). Even, ``>= 2 *
    floor``; the floor may push the total above ``budget``. ``floor=0`` gives a state
    whose rounded share is 0 no playouts (the solver's top-up rounds)."""
    w = np.asarray(weights, np.float64)
    total = w.sum()
    share = budget * w / total / 2 if total > 0 else np.zeros_like(w)
    return (2 * np.maximum(floor, np.rint(share))).astype(np.int64)


def _leaf_program(tables: Tables, states: BattleState, keys, max_turns: int) -> PlayoutResult:
    return jax.vmap(lambda s, k: playout(tables, s, k, max_turns))(states, keys)


def _forced_turn(tables: Tables, state: BattleState, pair, key):
    """The forced first turn: ``playout_from_action``'s key rule (playout._run) -- the
    turn's draws come from the second split of ``k_turn``. Returns the state after
    ``send_replacements``, whether that was refused, and the key for the rest."""
    key, _, k_turn = jax.random.split(key, 3)
    actions = pair.astype(_I32)
    power_mul = replacement_power_mul(tables, state, actions)
    outcome = sample_outcome(jax.random.split(k_turn)[1], tables, state, actions)
    stepped = step(tables, state, actions, outcome)
    after, refused = send_replacements(tables, stepped, is_switch(actions[1]), power_mul)
    return after, refused, key


def _forced_one(tables: Tables, state: BattleState, pair, key, max_turns: int) -> PlayoutResult:
    after, refused, key = _forced_turn(tables, state, pair, key)
    rest = playout(tables, after, key, max_turns - 1)
    return PlayoutResult(
        won=jnp.where(refused, REFUSED_WON, rest.won).astype(jnp.int8),
        player_faints=rest.player_faints,
        turns=rest.turns + 1,
        refused=(refused | (rest.refused == 1)).astype(jnp.int8),
    )


def _forced_program(tables: Tables, states: BattleState, pairs, keys, max_turns: int):
    return jax.vmap(lambda s, p, k: _forced_one(tables, s, p, k, max_turns))(states, pairs, keys)


def _run(fn, name: str, max_turns: int, tables: Tables, rows: tuple, row_of, keys, counts):
    """``fn`` over ``sum(counts)`` playout rows: row ``r`` plays ``rows[...][row_of[r]]``
    with ``keys[r]``. Chunked and padded as the module docstring says; returns the
    per-leaf ``LeafValue``s."""
    n = len(row_of)
    devices = jax.device_count()
    size = -(-LEAF_CHUNK // devices) * devices
    sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
    program = partial(fn, max_turns=max_turns)
    won, refused = [], []
    for start in range(0, n, LEAF_CHUNK):
        idx = np.arange(start, min(start + LEAF_CHUNK, n))
        pad = np.concatenate([idx, np.full(size - len(idx), idx[0])])
        chunk = (*jax.tree.map(lambda f, p=pad: np.asarray(f)[row_of[p]], rows), keys[pad])
        chunk = jax.device_put(chunk, sharding) if devices > 1 else jax.device_put(chunk)
        args = (tables, *chunk)
        exported = cached_export(
            program, args, name=name, static={"max_turns": max_turns}, version=code_version()
        )
        out = jax.device_get(exported(*args))
        won.append(np.asarray(out.won)[: len(idx)])
        refused.append(np.asarray(out.refused)[: len(idx)])
    won, refused = np.concatenate(won), np.concatenate(refused).astype(bool)
    starts = np.concatenate([[0], np.cumsum(counts)])
    values = []
    for i, c in enumerate(counts):
        h = int(c) // 2
        w = won[starts[i] : starts[i + 1]] == 1
        r = refused[starts[i] : starts[i + 1]]
        values.append(LeafValue(int(w[:h].sum()), h, int(w[h:].sum()), h, bool(r.any())))
    return values


def _check(name: str, states: BattleState, counts) -> np.ndarray:
    """``counts`` as ``int64[N]`` for the ``N`` numpy ``states``; ``ValueError`` naming
    ``name`` on a length mismatch or a finished battle (it has no playout: value it
    exactly)."""
    counts = np.asarray(counts, np.int64)
    n = len(states.hp)
    if counts.shape != (n,):
        raise ValueError(f"{name}: counts has shape {counts.shape}, want ({n},)")
    if np.any(states.over != 0):
        raise ValueError(f"{name}: a terminal state has no playout (value it exactly)")
    return counts


def _prepare(counts, key):
    counts = np.asarray(counts, np.int64)
    if np.any(counts < 2) or np.any(counts % 2):
        raise ValueError("playout counts must be even and at least 2")
    parts = [
        np.concatenate([np.asarray(k) for k in leaf_keys(key, i, int(c))])
        for i, c in enumerate(counts)
    ]
    return counts, np.concatenate(parts), np.repeat(np.arange(len(counts)), counts)


def evaluate_leaves(
    tables: Tables, states: BattleState, counts, key, max_turns: int = 100
) -> list[LeafValue]:
    """``counts[i]`` playouts (even) from each of the numpy ``states`` (``N`` rows), leaf
    ``i`` on ``leaf_keys(key, i, counts[i])``. ``ValueError`` on a finished battle or
    ``counts`` not of length ``N``."""
    states = jax.tree.map(np.asarray, states)
    counts, keys, row_of = _prepare(_check("evaluate_leaves", states, counts), key)
    return _run(_leaf_program, "leaf_playouts", max_turns, tables, (states,), row_of, keys, counts)


def evaluate_forced(
    tables: Tables, states: BattleState, pairs, counts, key, max_turns: int = 100
) -> list[LeafValue]:
    """``evaluate_leaves`` with the first turn forced to the (player, opponent) action
    pair ``pairs[i]`` (``int32[N, 2]``; module docstring). ``ValueError`` as
    ``evaluate_leaves`` and on ``pairs`` not of shape ``(N, 2)``."""
    states = jax.tree.map(np.asarray, states)
    counts = _check("evaluate_forced", states, counts)
    pairs = np.asarray(pairs, np.int32)
    if pairs.shape != (len(counts), 2):
        raise ValueError(f"evaluate_forced: pairs has shape {pairs.shape}, want ({len(counts)}, 2)")
    counts, keys, row_of = _prepare(counts, key)
    return _run(
        _forced_program, "leaf_forced", max_turns, tables, (states, pairs), row_of, keys, counts
    )
