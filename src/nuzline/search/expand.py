"""Batched exact turn expansion (Stage B spec §1-2; plan rulings 2-4): every outcome of
one (state, player action, opponent action) request, stepped, replaced and observed on
the device, merged on the host.

A turn is played exactly as a playout plays it (``playout._play`` and ``_turn``,
playout.py:177-194): ``step``, then the opponent's forced replacement
``send_replacements`` with ``ai_switched = is_switch(opponent action)`` and
``power_mul`` = ``step.replacement_power_mul`` (Pursuit's 20 when the player's switch
met Pursuit, else 10; the helper both paths share). A refused replacement keeps the
state ``send_replacements`` left and marks the child ``refused`` (ruling 4 values it 0).
The observation is ``engine.observe``'s, projected from ``step``'s own events
(``step.step_events``, ``observe.observation``) of the state after the replacement.

Every batched program (``ai_rows``, ``laws_rows``, ``step_rows``, ``replace_rows``) runs
in calls of a fixed number of rows: ``CHUNK``, except ``ai_rows``, whose calls have a
power-of-two bucket of its rows (``bucket_rows``, 256 to ``CHUNK``), so a small frontier
does not pay for ``CHUNK`` rows. A call's rows are rounded up to a multiple of the device
count (padding = copies of the call's row 0, dropped after the call), sharded over the
host devices as ``stage_a.rank_actions`` shards and exported through ``cached_export``, so
a solve compiles each program once per call shape (``ai_rows``: at most 7 buckets) and
device count, and a row's result depends on neither (``_ai``).

``expand`` enumerates each request's outcomes with ``enumerate_outcomes`` (float64
probabilities from the integer laws, ruling 3: rolls are not pre-merged), streams the
rows through ``step_rows`` one ``CHUNK`` at a time and merges each request's children
in a dict keyed by the child's state bytes and observation bytes (ruling 2; a bytes
dict, not ``np.unique(axis=0)``, which is ~150x slower: notes/stage-b/integration.md
§13). The refusal flag is part of the key too; it is a function of the state in
practice, so this never splits a child in two.

**Player replacements** (faint-budget spec §4): a turn's child that stops at a pending
player replacement is the decision-point state (``send_replacements`` stopped before
anyone entered), observed as any turn child. The replacement itself is a request
``(state index, player action)`` of ``expand_replacements``: ``replace_rows`` runs
``playout.replacement_pass`` for the chosen slot (``action - MOVE_ACTIONS``) and observes
the result with ``replacement_observation`` (no action ran). It has no outcome draw, so
the one child has probability 1; the store is deduplicated by state bytes as ``expand``'s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState for jax.export
from nuzline.ai.layout import AiLayout, layout_of
from nuzline.ai.policy import _distribution
from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import (
    OutcomeLaws,
    Reads,
    SamplingOnly,
    enumerate_outcomes,
    outcome_laws,
)
from nuzline.engine.observe import observation, replacement_observation
from nuzline.engine.order import is_switch
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import replacement_power_mul, step_events
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.playout import replacement_pass, send_replacements
from nuzline.search.rows import concat, row_bytes, take

__all__ = [
    "CHUNK",
    "Child",
    "ai_rows",
    "bucket_rows",
    "expand",
    "expand_replacements",
    "laws_rows",
    "replace_rows",
    "step_rows",
]

CHUNK = 16384  # rows per program call, before padding to the device count
AI_MIN_ROWS = 256  # ai_rows' smallest call (bucket_rows)

# The pytrees of the programs' arguments and results, for ``jax.export`` serialization
# (``Tables`` and ``BattleState`` are registered by ``stage_a``).
for _cls in (Outcome, OutcomeLaws, Reads):
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


def bucket_rows(n: int) -> int:
    """The rows of one ``ai_rows`` program call for ``n`` rows: the smallest power of two
    at least ``n``, between ``AI_MIN_ROWS`` and ``CHUNK`` (above ``CHUNK``, ``CHUNK``-row
    calls). So at most 7 shapes (256 .. 16384) are compiled per device count."""
    return min(CHUNK, max(AI_MIN_ROWS, 1 << max(n - 1, 0).bit_length()))


def _run(fn: Callable, name: str, tables: Tables, rows: tuple, n: int, static=None, per_call=CHUNK):
    """``fn(tables, *rows)`` (a ``vmap`` over the rows' leading axis) over ``n`` rows,
    in ``per_call``-row calls (default ``CHUNK``) padded and sharded as the module
    docstring says; numpy out. ``static`` names what else ``fn`` closes over
    (``cached_export``). Padding rows are copies of the call's row 0 and are dropped. A
    row's result does not depend on ``per_call`` when ``fn``'s per-row program does not
    change with the batch shape (``_ai`` says how ``ai_rows`` ensures it)."""
    if n == 0:
        specs = jax.tree.map(lambda a: jax.ShapeDtypeStruct((1, *a.shape[1:]), a.dtype), rows)
        shape = jax.eval_shape(fn, tables, *specs)
        return jax.tree.map(lambda s: np.zeros((0, *s.shape[1:]), s.dtype), shape)
    devices = jax.device_count()
    size = -(-per_call // devices) * devices
    sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
    outs = []
    for start in range(0, n, per_call):
        idx = np.arange(start, min(start + per_call, n))
        pad = np.concatenate([idx, np.full(size - len(idx), idx[0])])
        chunk = take(rows, pad)
        chunk = jax.device_put(chunk, sharding) if devices > 1 else jax.device_put(chunk)
        args = (tables, *chunk)
        program = cached_export(fn, args, name=name, static=static or {}, version=code_version())
        outs.append(take(jax.device_get(program(*args)), slice(0, len(idx))))
    return concat(outs)


def _ai(tables: Tables, states: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``_distribution`` of every row. A per-slot layout maps the rows one at a time
    (``lax.map``): the per-row program is then the same at every call size, so a row's
    float32 result does not depend on the bucket (under ``vmap`` XLA fuses the combine
    per batch shape, which moved some Kaizo rows by 1 ulp). Vanilla keeps ``vmap``,
    whose rows are stable (tests/search/test_expand.py)."""
    one = partial(_distribution, tables, layout=layout)
    if layout.per_slot_sites:
        return jax.lax.map(one, states)
    return jax.vmap(one)(states)


def ai_rows(tables: Tables, states: BattleState, per_call: int | None = None) -> np.ndarray:
    """``float32[N, NUM_OPP_ACTIONS]``: the AI's exact action distribution of each of
    ``N`` states (``policy._distribution``, the batched form of
    ``ai_action_distribution``) under the concrete ``tables``' AI layout (``layout_of``,
    a static key of the cached program); a NaN row is a refused state. The calls have
    ``bucket_rows(N)`` rows (``per_call`` overrides it, for tests): a lone state does not
    pay for ``CHUNK`` rows."""
    layout = layout_of(tables)
    fn = partial(_ai, layout=layout)
    n = len(states.hp)
    per_call = bucket_rows(n) if per_call is None else per_call
    return _run(fn, "expand_ai", tables, (states,), n, {"layout": layout.name}, per_call)


def _laws(tables: Tables, states: BattleState, actions) -> OutcomeLaws:
    return jax.vmap(outcome_laws, in_axes=(None, 0, 0))(tables, states, actions)


def laws_rows(tables: Tables, states: BattleState, actions) -> OutcomeLaws:
    """``outcome_laws`` of each of ``N`` (state, ``int32[2]`` actions) rows, numpy."""
    actions = np.asarray(actions, np.int32)
    return _run(_laws, "expand_laws", tables, (states, actions), len(actions))


def _turn(tables: Tables, before: BattleState, actions, outcome: Outcome):
    """One outcome of a turn, as the playout plays it (module docstring): ``(after,
    refused, observation)``."""
    power_mul = replacement_power_mul(tables, before, actions)
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
        state_bytes, obs_bytes = row_bytes(after, n), row_bytes(obs, n)
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
            self.parts.append(take(after, np.asarray(new)))


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
    laws = laws_rows(tables, take(states, index), actions)
    merger = _Merger(tables, len(pairs))
    out: dict[int, list[Child] | SamplingOnly] = {}
    buffer: list[tuple] = []  # (request ids, outcomes, probs) segments of the next chunk
    filled = 0

    def flush():
        nonlocal buffer, filled
        request = np.concatenate([b[0] for b in buffer])
        outcomes = concat([b[1] for b in buffer])
        probs = np.concatenate([b[2] for b in buffer])
        merger.add(request, take(states, index[request]), actions[request], outcomes, probs)
        buffer, filled = [], 0

    for r in range(len(pairs)):
        try:
            outcomes, probs = enumerate_outcomes(take(laws, r))
        except SamplingOnly as e:
            out[r] = e
            continue
        k, start = len(probs), 0
        while start < k:
            stop = min(k, start + CHUNK - filled)
            seg = slice(start, stop)
            buffer.append((np.full(stop - start, r), take(outcomes, seg), probs[seg]))
            filled += stop - start
            start = stop
            if filled == CHUNK:
                flush()
    if buffer:
        flush()
    for r in range(len(pairs)):
        if r not in out:
            out[r] = [Child(*entry) for entry in merger.children[r].values()]
    store = concat(merger.parts) if merger.parts else take(states, slice(0, 0))
    return store, dict(sorted(out.items()))


def _replace(tables: Tables, state: BattleState, action):
    after, refused = replacement_pass(tables, state, action - MOVE_ACTIONS)
    return after, refused, replacement_observation(after)


def _replaces(tables: Tables, states: BattleState, actions):
    return jax.vmap(_replace, in_axes=(None, 0, 0))(tables, states, actions)


def replace_rows(
    tables: Tables, states: BattleState, actions
) -> tuple[BattleState, np.ndarray, np.ndarray]:
    """``(after, refused bool[N], observation int32[N, OBS_SIZE])`` of ``N`` rows of
    (state at a player replacement, player action ``MOVE_ACTIONS + slot``):
    ``replacement_pass`` and ``replacement_observation``, in one cached program."""
    actions = np.asarray(actions, np.int32)
    after, refused, obs = _run(_replaces, "expand_replace", tables, (states, actions), len(actions))
    return after, np.asarray(refused, bool), np.asarray(obs, np.int32)


def expand_replacements(
    tables: Tables, states: BattleState, requests: list[tuple[int, int]]
) -> tuple[BattleState, dict[int, Child]]:
    """The child of each request ``(state index, player action)`` on the numpy
    ``states``: ``(store, children)`` as ``expand`` returns them, one ``Child`` per
    request with ``prob`` 1.0, the store deduplicated by state bytes."""
    states = jax.tree.map(np.asarray, states)
    index = np.asarray([r[0] for r in requests], np.int64).reshape(-1)
    actions = np.asarray([r[1] for r in requests], np.int32)
    after, refused, obs = replace_rows(tables, take(states, index), actions)
    n = len(requests)
    state_bytes, obs_bytes = row_bytes(after, n), row_bytes(obs, n)
    rows: dict[bytes, int] = {}
    new: list[int] = []
    children: dict[int, Child] = {}
    for i in range(n):
        key = state_bytes[i].tobytes()
        row = rows.get(key)
        if row is None:
            row = rows[key] = len(rows)
            new.append(i)
        children[i] = Child(row, obs_bytes[i].tobytes(), 1.0, bool(refused[i]))
    store = take(after, np.asarray(new)) if new else take(states, slice(0, 0))
    return store, children
