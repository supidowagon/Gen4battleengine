"""An exhaustive exact solver for small endgames (Stage B spec §6): every opponent action
and every battle outcome of every turn to a turn cap, no leaves, no pruning, written
plainly over ``step`` and ``enumerate_outcomes``. The reference ``stage_b.solve`` must
match on the endgames.

Independent of ``search/expand.py`` (nothing imported from it, no shared merge code):
each (state, action pair) has its enumerated outcome rows padded to ``P`` and played by
one module-level ``jax.jit(jax.vmap(_turn))`` (the playout's turn recipe,
``playout._play``/``_turn``: ``step``, the opponent's forced replacement
``send_replacements`` with ``ai_switched = is_switch(opponent action)`` and the
replacement's ``power_mul``, then ``observe``), merged on the host by (state bytes,
observation bytes, refused) and grouped by observation into belief nodes. A refused
child is valued 0 (plan ruling 4); a state not over at the cap counts 0.

A finished battle is worth ``deaths.success`` of its state; a failed state (over the death
budget, plan ruling 3) is worth 0 and not expanded. A pending state (the player's active
Pokemon fainted, a healthy one left) is a decision over ``decision_actions`` whose children
come from ``playout.replacement_pass`` and ``replacement_observation`` (one outcome each,
grouped by observation); a replacement does not count a turn, so a cap of ``cap`` counts
turns only.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai.policy import ai_action_distribution
from nuzline.engine.actions import MOVE_ACTIONS, decision_actions
from nuzline.engine.enumerate import enumerate_outcomes, outcome_laws
from nuzline.engine.observe import OBS_SIZE, observe, replacement_observation
from nuzline.engine.order import is_switch
from nuzline.engine.state import BattleState
from nuzline.engine.step import (
    WINNER_PLAYER,
    needs_replacement,
    replacement_power_mul,
    resolve_war,
    step,
    war_split,
)
from nuzline.search import deaths
from nuzline.search.playout import replacement_pass, send_replacements

P = 8192  # outcome rows per program call (padded)
_OBS_BYTES = OBS_SIZE * 4  # int32 observation


def _turn(tables, before, actions, outcome):
    power_mul = replacement_power_mul(tables, before, actions)
    stepped = step(tables, before, actions, outcome)
    after, refused = send_replacements(tables, stepped, is_switch(actions[1]), power_mul)
    return after, refused, observe(tables, before, actions, outcome, after)


_turns = jax.jit(jax.vmap(_turn, in_axes=(None, None, None, 0)))
_laws = jax.jit(outcome_laws)
_legal = jax.jit(decision_actions)


def _replace(tables, state, action, tie):
    after, refused = replacement_pass(tables, state, action - MOVE_ACTIONS)
    split = war_split(after)
    after = resolve_war(after, tie)
    return after, refused, replacement_observation(after), split


_replaced = jax.jit(_replace)
_pending = jax.jit(lambda state: needs_replacement(state)[0] == 1)


def state_bytes(state: BattleState) -> bytes:
    """The bytes of one (unbatched) state."""
    return b"".join(np.ascontiguousarray(np.asarray(f)).tobytes() for f in state)


def _row(tree, i: int):
    return jax.tree.map(lambda f: np.asarray(f)[i], tree)


_CHILDREN: dict[tuple, list] = {}


def children(
    tables, state: BattleState, a: int, o: int
) -> list[tuple[BattleState, bool, bytes, float]]:
    """``(after, refused, obs bytes, prob)`` of each merged child of the turn
    ``(a, o)`` at ``state`` (module docstring); probabilities float64."""
    key = (id(tables), state_bytes(state), a, o)
    if key in _CHILDREN:
        return _CHILDREN[key]
    actions = jnp.asarray([a, o], jnp.int32)
    outcomes, probs = enumerate_outcomes(jax.device_get(_laws(tables, state, actions)))
    k = len(probs)
    assert k <= P, k
    pad = np.concatenate([np.arange(k), np.zeros(P - k, np.int64)])
    padded = jax.tree.map(lambda f: np.asarray(f)[pad], outcomes)
    after, refused, obs = jax.device_get(_turns(tables, state, actions, padded))
    # Each row's state bytes at once: the fields' bytes side by side, in ``state_bytes``'
    # order, so row i of ``rows`` is ``state_bytes(_row(after, i))``.
    rows = np.concatenate(
        [np.ascontiguousarray(np.asarray(f)[:k]).reshape(k, -1).view(np.uint8) for f in after],
        axis=1,
    )
    merged: dict[bytes, list] = {}
    for i in range(k):
        ob = np.asarray(obs[i], np.int32).tobytes()
        mkey = rows[i].tobytes() + ob + bytes([bool(refused[i])])
        if mkey in merged:
            merged[mkey][3] += float(probs[i])
        else:
            merged[mkey] = [_row(after, i), bool(refused[i]), ob, float(probs[i])]
    out = [tuple(m) for m in merged.values()]
    _CHILDREN[key] = out
    return out


_REPLACED: dict[tuple, list] = {}


def replacement_children(
    tables, state: BattleState, a: int
) -> list[tuple[BattleState, bool, bytes, float]]:
    """``children`` of the replacement ``a`` (``MOVE_ACTIONS + slot``) at a pending
    ``state``: one child, probability 1, or one per tie bit (1/2) of an open weather war
    (``step.resolve_war``, ruling G-26)."""
    key = (id(tables), state_bytes(state), a)
    if key not in _REPLACED:
        kids = []
        for tie in (0, 1):
            after, refused, obs, split = jax.device_get(
                _replaced(tables, state, jnp.int32(a), jnp.int32(tie))
            )
            kids.append(
                (
                    jax.tree.map(np.asarray, after),
                    bool(refused),
                    np.asarray(obs, np.int32).tobytes(),
                    0.5 if bool(split) else 1.0,
                )
            )
            if not bool(split):
                break
        _REPLACED[key] = kids
    return _REPLACED[key]


def _ai(tables, state: BattleState) -> np.ndarray:
    return np.asarray(ai_action_distribution(tables, state), np.float64)


def exhaustive(
    tables,
    state: BattleState,
    cap: int,
    group: bool = True,
    belief: list[tuple[BattleState, float]] | None = None,
) -> tuple[dict[int, float], Callable[[tuple[bytes, ...]], int]]:
    """The exact value of every legal player action at ``state`` over ``cap`` turns, and
    ``choose(path)``: the optimal action at the belief node reached by the observation
    ``path`` (a tuple of observation bytes, one per turn). ``group=False`` lets each
    state choose its own optimum (the peeking player); ``choose`` raises ``KeyError`` on
    a path that names no node or two nodes with different choices. ``belief`` replaces the
    root ``[(state, 1.0)]`` with weighted states the player cannot tell apart (a root chance
    node, ``step.root_belief``)."""
    choices: dict[tuple[bytes, ...], int | None] = {}

    def value(belief: list[tuple[BattleState, float]], left: int, path) -> dict[int, float]:
        pending = bool(_pending(belief[0][0]))
        legal = np.flatnonzero(np.asarray(_legal(tables, belief[0][0])))
        qs = [None if pending else _ai(tables, s) for s, _ in belief]
        values = {}
        for a in legal.tolist():
            total = 0.0
            groups: dict[bytes, dict[bytes, list]] = {}
            # (state, reach, child) for every outcome of ``a``: a replacement has one
            # outcome and no opponent action; a turn is ``left - 1`` turns from here.
            outcomes = []
            for (s, w), q in zip(belief, qs, strict=True):
                if pending:
                    outcomes.extend((w * c[3], c) for c in replacement_children(tables, s, a))
                    continue
                for o in np.flatnonzero(q > 0).tolist():
                    outcomes.extend((w * q[o] * c[3], c) for c in children(tables, s, a, o))
            after_left = left if pending else left - 1
            for reach, (after, refused, ob, _) in outcomes:
                if refused:
                    continue
                if int(after.over) != 0:
                    total += reach * bool(
                        deaths.success(
                            int(after.winner) == WINNER_PLAYER,
                            int(after.player_faints),
                            np.asarray(after.deaths),
                            int(after.faint_budget),
                            np.asarray(after.protected),
                        )
                    )
                    continue
                if bool(deaths.over_budget(after)):
                    continue
                if after_left == 0 and not bool(_pending(after)):
                    continue
                sb = state_bytes(after)
                gkey = ob if group else ob + sb
                member = groups.setdefault(gkey, {}).setdefault(sb, [after, 0.0])
                member[1] += reach
            for gkey, members in groups.items():
                belief_next = [(m[0], m[1]) for m in members.values()]
                sub = value(belief_next, after_left, (*path, gkey[:_OBS_BYTES]))
                total += max(sub.values())
            values[a] = total
        best = max(values, key=lambda a: (values[a], -a))
        if path in choices and choices[path] != best:
            choices[path] = None
        else:
            choices.setdefault(path, best)
        return values

    start = [(state, 1.0)] if belief is None else belief
    root = value([(jax.tree.map(np.asarray, s), w) for s, w in start], cap, ())

    def choose(path: tuple[bytes, ...]) -> int:
        best = choices[tuple(path)]
        if best is None:
            raise KeyError(f"two belief nodes share the observation path {path!r}")
        return best

    return root, choose
