"""Batched exact turn expansion (Stage B spec §1-2; plan rulings 2-4): every outcome of
one (state, player action, opponent action) request, stepped, replaced and observed on
the device, merged on the host.

A turn is played exactly as a playout plays it (``playout._play`` and ``_turn``,
playout.py:179-199): ``step``, then the opponent's forced replacement
``send_replacements`` with ``ai_switched = is_switch(opponent action)`` and
``power_mul`` = ``PURSUIT_POWER_MUL`` when the player's switch met Pursuit
(``pursuit_fires`` on the turn-start state), else 10. A refused replacement keeps the
state ``send_replacements`` left and marks the child ``refused`` (ruling 4 values it 0).
The observation is ``engine.observe``'s, projected from ``step``'s own events
(``step.step_events``, ``observe.observation``) of the state after the replacement.

Every batched program (``ai_rows``, ``laws_rows``, ``step_rows``) runs in chunks of
``CHUNK`` rows padded to one fixed shape, ``-(-CHUNK // devices) * devices`` rows
(padding = copies of the chunk's row 0, dropped after the call), sharded over the host
devices as ``stage_a.rank_actions`` shards and exported through ``cached_export``, so
a solve compiles each program once per device count and the results do not depend on
it.

``expand`` enumerates each request's outcomes with ``enumerate_outcomes`` (float64
probabilities from the integer laws, ruling 3: rolls are not pre-merged), streams the
rows through ``step_rows`` one ``CHUNK`` at a time and merges each request's children
in a dict keyed by the child's state bytes and observation bytes (ruling 2; a bytes
dict, not ``np.unique(axis=0)``, which is ~150x slower: notes/stage-b/integration.md
§13). The refusal flag is part of the key too; it is a function of the state in
practice, so this never splits a child in two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState for jax.export
from nuzline.ai.policy import _distribution
from nuzline.data.tables import Tables
from nuzline.engine.enumerate import OutcomeLaws, SamplingOnly, enumerate_outcomes, outcome_laws
from nuzline.engine.observe import observation
from nuzline.engine.order import is_switch, selected_move
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import PURSUIT_POWER_MUL, pursuit_fires, step_events
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.playout import send_replacements

__all__ = ["CHUNK", "Child", "ai_rows", "expand", "laws_rows", "step_rows"]

CHUNK = 16384  # rows per program call, before padding to the device count
_NORMAL_POWER_MUL = 10

# The pytrees of the programs' arguments and results, for ``jax.export`` serialization
# (``Tables`` and ``BattleState`` are registered by ``stage_a``).
for _cls in (Outcome, OutcomeLaws):
    jax.export.register_namedtuple_serialization(_cls, serialized_name=f"nuzline.{_cls.__name__}")


@dataclass(frozen=True)
class Child:
    """One merged child of a request: ``state_row`` indexes the store ``expand``
    returns; ``obs`` is the ``int32[OBS_SIZE]`` observation's bytes; ``prob`` the
    float64 total of its outcomes; ``refused`` the opponent's replacement was refused."""

    state_row: int
    obs: bytes
    prob: float
    refused: bool


def _take(tree, idx):
    return jax.tree.map(lambda f: np.asarray(f)[idx], tree)


def _concat(parts):
    return jax.tree.map(lambda *fs: np.concatenate(fs), *parts)


def _row_bytes(tree, n: int) -> np.ndarray:
    """``uint8[n, bytes]``: each row of a stacked numpy pytree, flattened."""
    leaves = jax.tree.leaves(tree)
    return np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in leaves], axis=1
    )


def _run(fn: Callable, name: str, tables: Tables, rows: tuple, n: int):
    """``fn(tables, *rows)`` (a ``vmap`` over the rows' leading axis) over ``n`` rows,
    in ``CHUNK``-row calls padded and sharded as the module docstring says; numpy out."""
    if n == 0:
        specs = jax.tree.map(lambda a: jax.ShapeDtypeStruct((1, *a.shape[1:]), a.dtype), rows)
        shape = jax.eval_shape(fn, tables, *specs)
        return jax.tree.map(lambda s: np.zeros((0, *s.shape[1:]), s.dtype), shape)
    devices = jax.device_count()
    size = -(-CHUNK // devices) * devices
    sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
    outs = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(size - len(idx), idx[0])])
        chunk = _take(rows, pad)
        chunk = jax.device_put(chunk, sharding) if devices > 1 else jax.device_put(chunk)
        args = (tables, *chunk)
        program = cached_export(fn, args, name=name, static={}, version=code_version())
        outs.append(_take(jax.device_get(program(*args)), slice(0, len(idx))))
    return _concat(outs)


def _ai(tables: Tables, states: BattleState) -> jnp.ndarray:
    return jax.vmap(_distribution, in_axes=(None, 0))(tables, states)


def ai_rows(tables: Tables, states: BattleState) -> np.ndarray:
    """``float32[N, NUM_OPP_ACTIONS]``: the AI's exact action distribution of each of
    ``N`` states (``policy._distribution``, the batched form of
    ``ai_action_distribution``); a NaN row is a refused state."""
    return _run(_ai, "expand_ai", tables, (states,), len(states.hp))


def _laws(tables: Tables, states: BattleState, actions) -> OutcomeLaws:
    return jax.vmap(outcome_laws, in_axes=(None, 0, 0))(tables, states, actions)


def laws_rows(tables: Tables, states: BattleState, actions) -> OutcomeLaws:
    """``outcome_laws`` of each of ``N`` (state, ``int32[2]`` actions) rows, numpy."""
    actions = np.asarray(actions, np.int32)
    return _run(_laws, "expand_laws", tables, (states, actions), len(actions))


def _turn(tables: Tables, before: BattleState, actions, outcome: Outcome):
    """One outcome of a turn, as the playout plays it (module docstring): ``(after,
    refused, observation)``."""
    picks = [selected_move(tables, before, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    pursued = pursuit_fires(tables, before, 0, actions, choices)
    power_mul = jnp.where(pursued, PURSUIT_POWER_MUL, _NORMAL_POWER_MUL)
    stepped, events = step_events(tables, before, actions, outcome)
    after, refused = send_replacements(tables, stepped, is_switch(actions[1]), power_mul)
    return after, refused, observation(events, after)


def _turns(tables: Tables, states: BattleState, actions, outcomes: Outcome):
    return jax.vmap(_turn, in_axes=(None, 0, 0, 0))(tables, states, actions, outcomes)


def step_rows(
    tables: Tables, states: BattleState, actions, outcomes: Outcome
) -> tuple[BattleState, np.ndarray, np.ndarray]:
    """``(after, refused bool[N], observation int32[N, OBS_SIZE])`` of ``N`` rows of
    (state, ``int32[2]`` actions, outcome): ``step``, the opponent's replacement and the
    observation, in one cached program (module docstring)."""
    actions = np.asarray(actions, np.int32)
    after, refused, obs = _run(
        _turns, "expand_step", tables, (states, actions, outcomes), len(actions)
    )
    return after, np.asarray(refused, bool), np.asarray(obs, np.int32)


class _Merger:
    """Host merging of stepped rows: per request, children keyed by (state bytes,
    observation bytes, refused); one store row per distinct child state of the call."""

    def __init__(self, tables: Tables, requests: int):
        self.tables = tables
        self.children: list[dict[bytes, list]] = [{} for _ in range(requests)]
        self.rows: dict[bytes, int] = {}  # state bytes -> store row
        self.parts: list[BattleState] = []  # new store rows, per chunk

    def add(self, request, states, actions, outcomes, probs) -> None:
        after, refused, obs = step_rows(self.tables, states, actions, outcomes)
        n = len(request)
        state_bytes, obs_bytes = _row_bytes(after, n), _row_bytes(obs, n)
        new = []
        for i in range(n):
            sk, ok = state_bytes[i].tobytes(), obs_bytes[i].tobytes()
            row = self.rows.get(sk)
            if row is None:
                row = self.rows[sk] = len(self.rows)
                new.append(i)
            key = sk + ok + (b"\x01" if refused[i] else b"\x00")
            entry = self.children[request[i]].get(key)
            if entry is None:
                self.children[request[i]][key] = [row, ok, float(probs[i]), bool(refused[i])]
            else:
                entry[2] += float(probs[i])
        if new:
            self.parts.append(_take(after, np.asarray(new)))


def expand(
    tables: Tables, states: BattleState, pairs: list[tuple[int, int, int]]
) -> tuple[BattleState, dict[int, list[Child] | SamplingOnly]]:
    """The merged children of each request ``(state index, player action, opponent
    action)`` of ``pairs`` on the numpy ``states`` (``N`` rows).

    Returns ``(store, children)``: ``store`` holds this call's distinct child states
    (numpy, one row each; ``Child.state_row`` indexes it), ``children[r]`` the merged
    children of ``pairs[r]`` in first-seen order, with float64 probabilities summing to
    1, or the ``SamplingOnly`` that ``enumerate_outcomes`` raised for it (above the
    width cap). Rows stream through ``step_rows`` one ``CHUNK`` at a time, so memory is
    one chunk, one request's enumeration and the merged children."""
    states = jax.tree.map(np.asarray, states)
    index = np.asarray([p[0] for p in pairs], np.int64).reshape(-1)
    actions = np.asarray([p[1:] for p in pairs], np.int32).reshape(-1, 2)
    laws = laws_rows(tables, _take(states, index), actions)
    merger = _Merger(tables, len(pairs))
    out: dict[int, list[Child] | SamplingOnly] = {}
    buffer: list[tuple] = []  # (request ids, outcomes, probs) segments of the next chunk
    filled = 0

    def flush():
        nonlocal buffer, filled
        request = np.concatenate([b[0] for b in buffer])
        outcomes = _concat([b[1] for b in buffer])
        probs = np.concatenate([b[2] for b in buffer])
        merger.add(request, _take(states, index[request]), actions[request], outcomes, probs)
        buffer, filled = [], 0

    for r in range(len(pairs)):
        try:
            outcomes, probs = enumerate_outcomes(_take(laws, r))
        except SamplingOnly as e:
            out[r] = e
            continue
        k, start = len(probs), 0
        while start < k:
            stop = min(k, start + CHUNK - filled)
            seg = slice(start, stop)
            buffer.append((np.full(stop - start, r), _take(outcomes, seg), probs[seg]))
            filled += stop - start
            start = stop
            if filled == CHUNK:
                flush()
    if buffer:
        flush()
    for r in range(len(pairs)):
        if r not in out:
            out[r] = [Child(*entry) for entry in merger.children[r].values()]
    store = _concat(merger.parts) if merger.parts else _take(states, slice(0, 0))
    return store, dict(sorted(out.items()))
