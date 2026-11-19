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
the result with ``replacement_observation`` (no action ran). Its one draw is an open weather
war's tie (ruling G-26, ``step.resolve_war``): two children at 1/2 where it decides the
weather, else the one child has probability 1; the store is deduplicated by state bytes as
``expand``'s.

**Learn-move prompts** (K3d-exp design §3.5): ``TurnEnd`` pays exp and runs its learn-move
prompts (battle_controller_player.c:1846) before ``ReplaceFainted`` (:1848). So a turn's
child first runs the learnset scans (``levelup.advance_learns``); when a prompt is left and
the battle goes on, the child **stops** at that learn decision before the opponent's
replacement, holding the turn's ``power_mul`` in ``replace_mul`` when the opponent's active
is down (as ``send_replacements`` holds it at a player replacement), and refused when the
opponent needs a replacement after an AI switch (``ai_switched`` is decided now; roark.md
14, item 3). Learn work left at a battle-ending KO is no prompt (ruling D8): the child is
terminal. The answer is a request ``(state index, option)`` of ``expand_learns``:
``learn_rows`` runs ``levelup.learn`` with the option (``learn_options``' indexing), then,
once no prompt is left, ``send_replacements`` with the carried ``powerMul``
(``playout._carried_mul``), which may stop at a player replacement and hold it again;
``replace_mul`` is cleared when nothing is pending. It is observed with
``replacement_observation`` (no action ran). Every row kind (turn, learn, replacement) also
marks a child refused when ``exp_refused`` is set (design §3.3, window (c)).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState for jax.export
from nuzline.ai.layout import AiLayout, layout_of
from nuzline.ai.policy import _distribution
from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import (
    CallLaws,
    OutcomeLaws,
    Reads,
    SamplingOnly,
    call_consistent,
    call_laws,
    call_variants,
    combine_variants,
    enumerate_outcomes,
    outcome_laws,
)
from nuzline.engine.levelup import advance_learns, learn, needs_learn
from nuzline.engine.observe import observation, replacement_observation
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import needs_replacement, resolve_war, step_events, war_split
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.playout import (
    _carried_mul,
    replacement_pass,
    send_replacements,
    turn_inputs,
)
from nuzline.search.rows import concat, row_bytes, take

__all__ = [
    "CHUNK",
    "Child",
    "ai_rows",
    "bucket_rows",
    "expand",
    "expand_learns",
    "expand_replacements",
    "laws_rows",
    "learn_rows",
    "replace_rows",
    "step_rows",
]

CHUNK = 16384  # rows per program call, before padding to the device count
AI_MIN_ROWS = 256  # ai_rows' smallest call (bucket_rows)

# The pytrees of the programs' arguments and results, for ``jax.export`` serialization
# (``Tables`` and ``BattleState`` are registered by ``stage_a``).
for _cls in (Outcome, OutcomeLaws, Reads, CallLaws):
    jax.export.register_namedtuple_serialization(_cls, serialized_name=f"nuzline.{_cls.__name__}")


@dataclass(frozen=True)
class Child:
    """One merged child of a request: ``state_row`` indexes the store ``expand``
    returns; ``obs`` is the ``int32[OBS_SIZE]`` observation's bytes; ``prob`` the
    float64 total of its outcomes; ``refused`` the opponent's replacement was refused (or
    ``exp_refused`` is set)."""

    state_row: int
    obs: bytes
    prob: float
    refused: bool


def bucket_rows(n: int) -> int:
    """The rows of one ``ai_rows`` program call for ``n`` rows: the smallest power of two
    at least ``n``, between ``AI_MIN_ROWS`` and ``CHUNK`` (above ``CHUNK``, ``CHUNK``-row
    calls). So at most 7 shapes (256 .. 16384) are compiled per device count."""
    return min(CHUNK, max(AI_MIN_ROWS, 1 << max(n - 1, 0).bit_length()))


def _run(fn: Callable, name: str, tables: Tables, rows: tuple, n: int, static=None, per_call=None):
    """``fn(tables, *rows)`` (a ``vmap`` over the rows' leading axis) over ``n`` rows,
    in ``per_call``-row calls (default ``CHUNK``, read at the call) padded and sharded as the module
    docstring says; numpy out. ``static`` names what else ``fn`` closes over
    (``cached_export``). Padding rows are copies of the call's row 0 and are dropped. A
    row's result does not depend on ``per_call`` when ``fn``'s per-row program does not
    change with the batch shape (``_ai`` says how ``ai_rows`` ensures it)."""
    if n == 0:
        # One row per device: a shard_map'd fn (_ai's per-slot layout) needs the leading
        # axis to divide by the device count, even under eval_shape.
        lead = jax.device_count()
        specs = jax.tree.map(lambda a: jax.ShapeDtypeStruct((lead, *a.shape[1:]), a.dtype), rows)
        shape = jax.eval_shape(fn, tables, *specs)
        return jax.tree.map(lambda s: np.zeros((0, *s.shape[1:]), s.dtype), shape)
    per_call = CHUNK if per_call is None else per_call
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
    """``_distribution`` of every row. A per-slot layout maps each device's rows one at a
    time (``lax.map`` inside ``shard_map`` over the call's device axis): the per-row
    program is then the same at every call size and device count, so a row's float32
    result depends on neither. Under ``vmap`` some Kaizo rows were observed to differ by
    1 ulp between call sizes. ``shard_map`` keeps the rows on their devices; ``lax.map``
    alone over the sharded axis would run them on one. Vanilla keeps ``vmap``, whose rows
    were observed stable (tests/search/test_expand.py)."""
    one = partial(_distribution, tables, layout=layout)
    if not layout.per_slot_sites:
        return jax.vmap(one)(states)
    mesh = Mesh(jax.devices(), ("d",))
    return shard_map(
        lambda rows: jax.lax.map(one, rows),
        mesh=mesh,
        in_specs=PartitionSpec("d"),
        out_specs=PartitionSpec("d"),
        check_vma=False,
    )(states)


def ai_rows(tables: Tables, states: BattleState, per_call: int | None = None) -> np.ndarray:
    """``float32[N, NUM_OPP_ACTIONS]``: the AI's exact action distribution of each of
    ``N`` states (``policy._distribution``, the batched form of
    ``ai_action_distribution``) under the concrete ``tables``' AI layout (``layout_of``,
    whose ``key``, the policy and the trainer's flag set, keys the cached program); a NaN row is a refused state. The calls have
    ``bucket_rows(N)`` rows (``per_call`` overrides it, for tests): a lone state does not
    pay for ``CHUNK`` rows."""
    layout = layout_of(tables)
    fn = partial(_ai, layout=layout)
    n = len(states.hp)
    per_call = bucket_rows(n) if per_call is None else per_call
    return _run(fn, "expand_ai", tables, (states,), n, {"layout": layout.key}, per_call)


def _laws(tables: Tables, states: BattleState, actions, calls, takens) -> OutcomeLaws:
    return jax.vmap(outcome_laws, in_axes=(None, 0, 0, 0, 0))(
        tables, states, actions, calls, takens
    )


def laws_rows(tables: Tables, states: BattleState, actions, calls, takens=None) -> OutcomeLaws:
    """``outcome_laws`` of each of ``N`` (state, ``int32[2]`` actions, ``int32[2]`` call,
    ``bool[2]`` takeover tag) rows, numpy (``enumerate.call_variants`` gives the calls and tags;
    no tag: none taken)."""
    actions = np.asarray(actions, np.int32)
    calls = np.asarray(calls, np.int32)
    takens = np.zeros(calls.shape, bool) if takens is None else np.asarray(takens, bool)
    return _run(_laws, "expand_laws", tables, (states, actions, calls, takens), len(actions))


def _calls(tables: Tables, states: BattleState, actions) -> CallLaws:
    return jax.vmap(call_laws, in_axes=(None, 0, 0))(tables, states, actions)


def calls_rows(tables: Tables, states: BattleState, actions) -> CallLaws:
    """``call_laws`` of each of ``N`` (state, ``int32[2]`` actions) rows, numpy."""
    actions = np.asarray(actions, np.int32)
    return _run(_calls, "expand_calls", tables, (states, actions), len(actions))


@jax.jit
def _picks(tables: Tables, states: BattleState, actions) -> jnp.ndarray:
    return jax.vmap(lambda s, a: turn_inputs(tables, s, a)[0])(states, actions)


def _refused_state(state: BattleState) -> jnp.ndarray:
    """A state the search must not play on: learn work refused (``exp_refused``) or a Pursuit
    Acid Rain (``acid_rain``)."""
    return (state.exp_refused != 0) | (state.acid_rain != 0)


def _select(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def _turn(tables: Tables, before: BattleState, actions, outcome: Outcome, taken):
    """One outcome of a turn, as the playout plays it (module docstring): ``(after,
    refused, observation)``. The learnset scans run first (``advance_learns``: the
    ``TurnEnd`` payout's prompts come before ``ReplaceFainted``, bcp.c:1846-1848); a prompt
    in a battle that goes on stops the child at the learn decision (module docstring),
    else the opponent's replacement follows as in the playout (``finish_turn``). The fourth
    result is ``enumerate.call_consistent``: a row of a split Sleep Talk variant (ruling B5-13)
    or of a takeover variant (``taken``, B11-1) whose turn did not meet the variant's assumption
    is dropped."""
    # E2 batch 6: a state paused at a party list resumes. Its opponent's pick at its own list is
    # made once, host-side in ``expand`` (the laws read it): ``actions`` carry it, and only the
    # AI's refusal of it is read here (``playout.pivot_pick``, refused as a replacement's); the
    # turn's powerMul and AI switch are the turn start's, kept in the scratch (the resume's
    # actions are switch-shaped). B11 review item: one pick, so dropping it is caught
    # (tests/search/test_pivot_search.py test_the_opponents_list_is_resolved_by_its_ai).
    _, pick_refused, power_mul, ai_switched = turn_inputs(tables, before, actions)
    stepped, events = step_events(tables, before, actions, outcome)
    # A turn that pauses stops here: the child is the decision at the list (stage_b's "pivot").
    halted = stepped.pivot_battler != 0
    paused_child = stepped
    stepped = advance_learns(tables, stepped)
    prompt = needs_learn(stepped) & (stepped.over == 0)  # D8: no prompt once it is over
    # The learn stop: the turn's powerMul waits in replace_mul for the opponent's
    # replacement after the answer (as at send_replacements' player stop), and an opponent
    # replacement after an AI switch is refused now (send_replacements' rule).
    opp_down = stepped.hp[1, stepped.active[1]] == 0
    held = stepped._replace(
        replace_mul=jnp.where(opp_down, power_mul, 0).astype(stepped.replace_mul.dtype)
    )
    held_refused = ai_switched & (needs_replacement(stepped)[1] == 1)
    sent, sent_refused = send_replacements(tables, stepped, ai_switched, power_mul)
    after = _select(prompt, held, sent)
    refused = jnp.where(prompt, held_refused, sent_refused) | _refused_state(after)
    after = _select(halted, paused_child, after)
    refused = jnp.where(halted, False, refused) | pick_refused
    keep = call_consistent(tables, before, actions, outcome, events, taken)
    return after, refused, observation(events, after), keep


def _turns(tables: Tables, states: BattleState, actions, outcomes: Outcome, takens):
    return jax.vmap(_turn, in_axes=(None, 0, 0, 0, 0))(tables, states, actions, outcomes, takens)


def step_rows(
    tables: Tables, states: BattleState, actions, outcomes: Outcome, takens=None
) -> tuple[BattleState, np.ndarray, np.ndarray, np.ndarray]:
    """``(after, refused bool[N], observation int32[N, OBS_SIZE], keep bool[N])`` of ``N``
    rows of (state, ``int32[2]`` actions, outcome, ``bool[2]`` takeover tag of its call variant;
    none: no tag): ``step``, the opponent's replacement, the observation and
    ``enumerate.call_consistent``, in one cached program (module docstring)."""
    actions = np.asarray(actions, np.int32)
    takens = np.zeros(actions.shape, bool) if takens is None else np.asarray(takens, bool)
    after, refused, obs, keep = _run(
        _turns, "expand_step", tables, (states, actions, outcomes, takens), len(actions)
    )
    return after, np.asarray(refused, bool), np.asarray(obs, np.int32), np.asarray(keep, bool)


class _Merger:
    """Host merging of stepped rows: per request, children keyed by (state bytes,
    observation bytes, refused); one store row per distinct child state of the call."""

    def __init__(self, tables: Tables, requests: int):
        self.tables = tables
        self.children: list[dict[bytes, list]] = [{} for _ in range(requests)]
        self.rows: dict[bytes, int] = {}  # state bytes -> store row
        self.parts: list[BattleState] = []  # new store rows, per chunk

    def add(self, request, states, actions, outcomes, probs, takens) -> None:
        after, refused, obs, keep = step_rows(self.tables, states, actions, outcomes, takens)
        n = len(request)
        state_bytes, obs_bytes = row_bytes(after, n), row_bytes(obs, n)
        new = []
        for i in range(n):
            if not keep[i]:
                continue
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
    # E2 batch 6: at the opponent's own party list its pick is the resume's switch, which the
    # laws read (``outcome.pivot_view``): made here only (``turn_inputs``' pick); the row program
    # steps these actions and reads only the pick's refusal.
    opp_list = np.asarray(states.pivot_battler)[index] == 2
    if opp_list.any():
        rows = np.flatnonzero(opp_list)
        picked = _picks(tables, take(states, index[rows]), jnp.asarray(actions[rows]))
        actions[rows] = np.asarray(picked, np.int32)
    # The invokers' calls (enumerate.call_variants): one laws row per (request, call variant).
    cl = calls_rows(tables, take(states, index), actions)
    variants = [call_variants(take(cl, r)) for r in range(len(pairs))]
    v_req = np.asarray([r for r, vs in enumerate(variants) for _ in vs], np.int64)
    v_call = np.asarray([v.call for vs in variants for v in vs], np.int32).reshape(-1, 2)
    v_taken = np.asarray([v.taken for vs in variants for v in vs], bool).reshape(-1, 2)
    laws = laws_rows(tables, take(states, index[v_req]), actions[v_req], v_call, v_taken)
    first_row = np.concatenate([[0], np.cumsum([len(vs) for vs in variants])])
    merger = _Merger(tables, len(pairs))
    out: dict[int, list[Child] | SamplingOnly] = {}
    buffer: list[tuple] = []  # (request ids, outcomes, probs, tags) segments of the next chunk
    filled = 0

    def flush():
        nonlocal buffer, filled
        request = np.concatenate([b[0] for b in buffer])
        outcomes = concat([b[1] for b in buffer])
        probs = np.concatenate([b[2] for b in buffer])
        tags = np.concatenate([b[3] for b in buffer])
        merger.add(request, take(states, index[request]), actions[request], outcomes, probs, tags)
        buffer, filled = [], 0

    for r in range(len(pairs)):
        try:
            parts, tags = [], []
            for i, v in enumerate(variants[r]):
                outcomes, probs = enumerate_outcomes(take(laws, int(first_row[r]) + i))
                parts.append((outcomes, probs, v.num, v.den))
                tags.append(np.broadcast_to(v.taken, (len(probs), 2)))
            outcomes, probs = combine_variants(parts)
            takens = np.concatenate(tags)
        except SamplingOnly as e:
            out[r] = e
            continue
        k, start = len(probs), 0
        while start < k:
            stop = min(k, start + CHUNK - filled)
            seg = slice(start, stop)
            buffer.append((np.full(stop - start, r), take(outcomes, seg), probs[seg], takens[seg]))
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
    refused = refused | _refused_state(after)
    # An open weather war (ruling G-26): its tie bit is a chance node here, before the
    # player's next decision (step.resolve_war), else both rows are the one child.
    ties = [resolve_war(after, tie) for tie in (0, 1)]
    obs = jnp.stack([replacement_observation(t) for t in ties])
    both = jax.tree.map(lambda a, b: jnp.stack([a, b]), *ties)
    return both, war_split(after), refused, obs


def _replaces(tables: Tables, states: BattleState, actions):
    return jax.vmap(_replace, in_axes=(None, 0, 0))(tables, states, actions)


def replace_rows(
    tables: Tables, states: BattleState, actions
) -> tuple[BattleState, np.ndarray, np.ndarray, np.ndarray]:
    """``(after, split bool[N], refused bool[N], observation int32[N, 2, OBS_SIZE])`` of ``N``
    rows of (state at a player replacement, player action ``MOVE_ACTIONS + slot``):
    ``replacement_pass`` and ``replacement_observation``, in one cached program. ``after``'s
    fields are ``[N, 2, ...]``: the state with an open weather war settled on each tie bit
    (``step.resolve_war``); ``split`` where the two differ (else both are the one child)."""
    actions = np.asarray(actions, np.int32)
    after, split, refused, obs = _run(
        _replaces, "expand_replace", tables, (states, actions), len(actions)
    )
    return after, np.asarray(split, bool), np.asarray(refused, bool), np.asarray(obs, np.int32)


def _one_child_rows(rows_fn, tables: Tables, states: BattleState, requests):
    """``(store, children)`` of one-child requests ``(state index, action)`` through
    ``rows_fn`` (``replace_rows``, ``learn_rows``): one ``Child`` per request with ``prob``
    1.0, the store deduplicated by state bytes."""
    states = jax.tree.map(np.asarray, states)
    index = np.asarray([r[0] for r in requests], np.int64).reshape(-1)
    actions = np.asarray([r[1] for r in requests], np.int32)
    after, refused, obs = rows_fn(tables, take(states, index), actions)
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


def expand_replacements(
    tables: Tables, states: BattleState, requests: list[tuple[int, int]]
) -> tuple[BattleState, dict[int, list[Child]]]:
    """The children of each request ``(state index, player action)`` on the numpy
    ``states``: ``(store, children)`` as ``expand`` returns them, the store deduplicated by
    state bytes. One ``Child`` with ``prob`` 1.0, or, where an open weather war's tie decides
    the weather (``replace_rows``' ``split``, ruling G-26), one per tie bit at 0.5."""
    states = jax.tree.map(np.asarray, states)
    index = np.asarray([r[0] for r in requests], np.int64).reshape(-1)
    actions = np.asarray([r[1] for r in requests], np.int32)
    after, split, refused, obs = replace_rows(tables, take(states, index), actions)
    n = len(requests)
    flat = jax.tree.map(lambda f: np.asarray(f).reshape((2 * n, *np.shape(f)[2:])), after)
    obs = obs.reshape((2 * n, -1))
    state_bytes, obs_bytes = row_bytes(flat, 2 * n), row_bytes(obs, 2 * n)
    rows: dict[bytes, int] = {}
    new: list[int] = []
    children: dict[int, list[Child]] = {}
    for i in range(n):
        ties = (0, 1) if split[i] else (0,)
        kids = []
        for tie in ties:
            j = 2 * i + tie
            key = state_bytes[j].tobytes()
            row = rows.get(key)
            if row is None:
                row = rows[key] = len(rows)
                new.append(j)
            kids.append(Child(row, obs_bytes[j].tobytes(), 1.0 / len(ties), bool(refused[i])))
        children[i] = kids
    store = take(flat, np.asarray(new)) if new else take(states, slice(0, 0))
    return store, children


def _learn(tables: Tables, state: BattleState, option):
    """The answer ``option`` to ``state``'s learn-move prompt (module docstring):
    ``(after, refused, observation)``. Another prompt left stops there; else the
    opponent's pending replacement runs with the carried ``powerMul`` (``ai_switched`` was
    settled at the stop), and ``replace_mul`` is cleared unless the player's replacement is
    pending (``send_replacements`` then holds it again)."""
    answered = learn(tables, state, option)
    prompt = needs_learn(answered) & (answered.over == 0)
    sent, sent_refused = send_replacements(tables, answered, False, _carried_mul(answered))
    pending = needs_replacement(sent)[0] == 1
    sent = sent._replace(
        replace_mul=jnp.where(pending, sent.replace_mul, 0).astype(sent.replace_mul.dtype)
    )
    after = _select(prompt, answered, sent)
    refused = (~prompt & sent_refused) | _refused_state(after)
    return after, refused, replacement_observation(after)


def _learns(tables: Tables, states: BattleState, options):
    return jax.vmap(_learn, in_axes=(None, 0, 0))(tables, states, options)


def learn_rows(
    tables: Tables, states: BattleState, options
) -> tuple[BattleState, np.ndarray, np.ndarray]:
    """``(after, refused bool[N], observation int32[N, OBS_SIZE])`` of ``N`` rows of
    (state at a learn-move prompt, ``learn_options`` option): ``_learn`` in one cached
    program."""
    options = np.asarray(options, np.int32)
    after, refused, obs = _run(_learns, "expand_learn", tables, (states, options), len(options))
    return after, np.asarray(refused, bool), np.asarray(obs, np.int32)


def expand_learns(
    tables: Tables, states: BattleState, requests: list[tuple[int, int]]
) -> tuple[BattleState, dict[int, Child]]:
    """The child of each request ``(state index, learn option)`` on the numpy ``states``
    (each at a learn-move prompt), as ``expand_replacements`` returns them."""
    return _one_child_rows(learn_rows, tables, states, requests)
