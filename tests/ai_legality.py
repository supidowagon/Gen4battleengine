"""The opponent-legality invariant (test side only, no engine cost; ledger "Opponent-action
legality"): every action the AI policy picks, or gives a positive probability, is one
``legal_opponent_actions`` allows in that state; every replacement it picks is a
``legal_replacements`` slot. ``step`` trusts the AI for this (it checks no opponent action), so
the tests check it on the concrete outputs of every host-side call, never inside a traced
program (a traced call passes through unchecked).

``install`` wraps ``ai_policy``, ``ai_action_distribution``, ``expand.ai_rows`` and
``post_ko_switch_in`` wherever a loaded module bound them (the policy, the search, the test
modules), once per session (tests/conftest.py).

Exceptions the decomp itself makes are listed in ``EXCEPTED`` with their citation; none so far.
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from nuzline.engine.actions import legal_opponent_actions, legal_replacements
from nuzline.engine.state import BattleState

# Decomp-sanctioned escapes from ``legal_opponent_actions`` (none known).
EXCEPTED: tuple[str, ...] = ()


class IllegalAiAction(AssertionError):
    """The AI chose (or weighted) an action the opponent may not take."""


def _concrete(*xs) -> bool:
    leaves = jax.tree.leaves(xs)
    return not any(isinstance(x, jax.core.Tracer) for x in leaves)


def _turn_state(state: BattleState) -> bool:
    """A turn decision: no pause at a party list and no replacement pending."""
    from nuzline.engine.step import needs_replacement

    return int(state.pivot_battler) == 0 and not np.any(np.asarray(needs_replacement(state)))


def _fail(what: str, state: BattleState, action, legal) -> None:
    raise IllegalAiAction(
        f"the AI {what} {action}, which legal_opponent_actions forbids in this state "
        f"(legal: {np.flatnonzero(legal).tolist()})\nstate: {state}"
    )


def check_action(tables, state: BattleState, action) -> None:
    """A concrete ``ai_policy`` pick: legal at a turn decision (``REFUSED_ACTION`` aside)."""
    from nuzline.ai.switching import REFUSED_ACTION

    if not _concrete(state, action) or not _turn_state(state):
        return
    a = int(action)
    if a == REFUSED_ACTION:
        return
    legal = np.asarray(legal_opponent_actions(tables, state))
    if not bool(legal[a]):
        _fail("chose action", state, a, legal)


def check_distribution(tables, state: BattleState, probs) -> None:
    """A concrete distribution: every action with a positive weight legal (a NaN row is a
    refused state)."""
    if not _concrete(state, probs) or not _turn_state(state):
        return
    p = np.asarray(probs)
    if np.any(np.isnan(p)):
        return
    legal = np.asarray(legal_opponent_actions(tables, state))
    bad = np.flatnonzero((p > 0) & ~legal)
    if bad.size:
        _fail("gave a positive weight to action(s)", state, bad.tolist(), legal)


def check_rows(tables, states: BattleState, rows) -> None:
    """``ai_rows``: each row of a turn state."""
    rows = np.asarray(rows)
    for i in range(rows.shape[0]):
        one = jax.tree.map(lambda x, i=i: np.asarray(x)[i], states)
        check_distribution(tables, BattleState(*one), rows[i])


def check_replacement(tables, state: BattleState, slot) -> None:
    """A concrete ``post_ko_switch_in`` pick: a ``legal_replacements`` slot (``REFUSED_SLOT``
    aside), where one exists."""
    from nuzline.ai.replacement import REFUSED_SLOT

    if not _concrete(state, slot):
        return
    s = int(slot)
    legal = np.asarray(legal_replacements(state, 1))
    if s != REFUSED_SLOT and legal.any() and not legal[s]:
        raise IllegalAiAction(f"the AI replaced with slot {s}, not a legal one\nstate: {state}")


def _wrap_policy(fn):
    def wrapped(tables, state, *args, **kwargs):
        out = fn(tables, state, *args, **kwargs)
        check_action(tables, state, out)
        return out

    return wrapped


def _wrap_distribution(fn):
    def wrapped(tables, state, *args, **kwargs):
        out = fn(tables, state, *args, **kwargs)
        check_distribution(tables, state, out)
        return out

    return wrapped


def _wrap_rows(fn):
    def wrapped(tables, states, *args, **kwargs):
        out = fn(tables, states, *args, **kwargs)
        check_rows(tables, states, out)
        return out

    return wrapped


def _wrap_replacement(fn):
    def wrapped(tables, state, *args, **kwargs):
        out = fn(tables, state, *args, **kwargs)
        check_replacement(tables, state, out)
        return out

    return wrapped


def install() -> list[tuple[object, str, object]]:
    """Wrap the four entry points in every loaded module that bound them; returns the
    ``(module, name, original)`` list ``uninstall`` restores."""
    from nuzline.ai import policy, replacement
    from nuzline.search import expand

    targets = {
        id(policy.ai_policy): (policy.ai_policy, _wrap_policy),
        id(policy.ai_action_distribution): (policy.ai_action_distribution, _wrap_distribution),
        id(expand.ai_rows): (expand.ai_rows, _wrap_rows),
        id(replacement.post_ko_switch_in): (replacement.post_ko_switch_in, _wrap_replacement),
    }
    wrapped = {key: wrap(fn) for key, (fn, wrap) in targets.items()}
    patched = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith(("nuzline", "tests")):
            continue
        for attr, value in list(vars(module).items()):
            if id(value) in wrapped and value is targets[id(value)][0]:
                setattr(module, attr, wrapped[id(value)])
                patched.append((module, attr, value))
    return patched


def uninstall(patched) -> None:
    for module, attr, value in patched:
        setattr(module, attr, value)
