"""The line player (docs/superpowers/specs/2026-11-12-l1-line-player-design.md §3): a
simulated player that follows one rule per opposing Pokemon. These are search heuristics,
player choices only, not battle rules: the trainer AI, ``step`` and the chance model are
``playout``'s, and every choice is checked against ``legal_actions`` /
``legal_replacements``, which carry their own decomp citations.

A ``Line`` holds six ``int32[6]`` arrays indexed by the opposing party slot (-1 = unset;
``wait`` 0/1), so a batch of lines is data and one compiled program plays any of them. A
turn's choice, with ``foe`` the opponent's active slot and ``me`` the player's
(``line_action``):

1. No rule for ``foe``, or its ``use`` has fainted: the fallback (``playout_policy``)
   decides (``MODE_FALLBACK``).
2. ``me == use``: ``first`` while it is set, legal and the foe's ``status`` is 0, else
   ``move``.
3. ``me != use`` and ``wait`` set: the fallback decides among the active Pokemon's moves
   only (``MODE_MOVES_ONLY``); the answer enters on the free switch after a faint.
4. ``me != use``, ``via`` set and living, ``me != via`` and (``via_from`` unset or
   ``me == via_from``): switch to ``via``.
5. Otherwise switch to ``use``.

A rule action that ``legal_actions`` forbids this turn (trapped, locked, no PP) is left
to the fallback, so a rule never makes a playout illegal.

A player replacement (``line_replacement``) answers the opponent's active Pokemon, or,
when it is down too, the slot ``post_ko_switch_in`` will send (as ``replacement_policy``
reads it): that rule's ``use`` enters when it may (``legal_replacements``), else
``replacement_policy`` decides. ``via`` and ``wait`` do not apply.

``LINE`` is these two choices as a ``Player`` (``nuzline.search.player``; its ``params``
are a ``Line``), and ``line_playout`` is ``player_playout`` with it, so with
``empty_line()`` it returns what ``playout`` returns from the same root and key.
``PlayerResult.fallback_turns`` counts the turns played in ``MODE_FALLBACK`` or
``MODE_MOVES_ONLY``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai.layout import VANILLA, AiLayout
from nuzline.ai.replacement import post_ko_switch_in
from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, legal_actions, legal_replacements
from nuzline.engine.state import SLOTS, BattleState
from nuzline.search.player import Player, PlayerResult, player_playout
from nuzline.search.playout import _carried_mul, _policy, replacement_policy

__all__ = [
    "LINE",
    "MODE_FALLBACK",
    "MODE_MOVES_ONLY",
    "MODE_RULE",
    "Line",
    "empty_line",
    "line_action",
    "line_playout",
    "line_replacement",
]

_I32 = jnp.int32
MODE_FALLBACK, MODE_RULE, MODE_MOVES_ONLY = 0, 1, 2


class Line(NamedTuple):  # each int32[6], indexed by opposing party slot; -1 = unset
    """Values are not checked here (traced): the boundary (the line file) validates them;
    ``use``/``via``/``via_from`` must be party slots 0-5 and ``move``/``first`` 0-3."""

    use: jnp.ndarray  # player's party slot that answers the foe
    move: jnp.ndarray  # its move slot 0-3
    first: jnp.ndarray  # move slot played while the foe's status == 0
    via: jnp.ndarray  # party slot to switch through
    via_from: jnp.ndarray  # via applies only when this slot is active
    wait: jnp.ndarray  # 1 = never hard-switch the answer in


def empty_line() -> Line:
    """A line with no rule (one fresh numpy array per field, all -1, ``wait`` 0): the
    fallback plays."""
    return Line(
        *(np.full(SLOTS, -1, np.int32) for _ in Line._fields[:-1]), np.zeros(SLOTS, np.int32)
    )


def _rule(line: Line, foe) -> Line:
    return Line(*(jnp.asarray(f, _I32)[foe] for f in line))


def _at(mask, slot):
    """``mask[slot]`` for a slot that may be -1 (then ``False``)."""
    return (slot >= 0) & mask[jnp.maximum(slot, 0)]


def line_action(tables: Tables, state: BattleState, line: Line) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(action, mode)`` (``int32``) of the turn rule (module docstring). ``action`` is
    the rule's legal action in ``MODE_RULE`` and -1 in the fallback modes."""
    foe = state.active[1].astype(_I32)
    me = state.active[0].astype(_I32)
    r = _rule(line, foe)
    living = (state.species[0] > 0) & (state.hp[0] > 0)
    legal = legal_actions(tables, state, 0)
    opener = _at(legal[:MOVE_ACTIONS], r.first) & (state.status[1, foe] == 0)
    move = jnp.where(opener, r.first, r.move)
    pivot = _at(living, r.via) & (me != r.via) & ((r.via_from < 0) | (me == r.via_from))
    switch = MOVE_ACTIONS + jnp.where(pivot, r.via, r.use)
    action = jnp.where(me == r.use, move, switch)
    legal_rule = _at(legal, jnp.where(action < NUM_ACTIONS, action, -1))
    mode = jnp.where(
        (me != r.use) & (r.wait == 1),
        MODE_MOVES_ONLY,
        jnp.where(legal_rule, MODE_RULE, MODE_FALLBACK),
    )
    mode = jnp.where(_at(living, r.use), mode, MODE_FALLBACK).astype(_I32)
    return jnp.where(mode == MODE_RULE, action, -1).astype(_I32), mode


def line_replacement(tables: Tables, state: BattleState, line: Line) -> jnp.ndarray:
    """``int32``: the party slot the player sends in at a replacement (module docstring)."""
    opp_down = state.hp[1, state.active[1]] == 0
    incoming = jnp.maximum(post_ko_switch_in(tables, state, _carried_mul(state)), 0)
    foe = jnp.where(opp_down, incoming, state.active[1]).astype(_I32)
    use = _rule(line, foe).use
    ok = _at(legal_replacements(state, 0), use)
    return jnp.where(ok, use, replacement_policy(tables, state)).astype(_I32)


def _line_act(tables: Tables, state: BattleState, line: Line, key, layout: AiLayout):
    """``LINE.act``: the rule's action in ``MODE_RULE``, else the fallback's (masked to
    moves in ``MODE_MOVES_ONLY``). A refused distribution refuses the turn only when the
    fallback's action is used."""
    rule, mode = line_action(tables, state, line)
    moves_only = jnp.arange(NUM_ACTIONS) < MOVE_ACTIONS
    allowed = jnp.where(mode == MODE_MOVES_ONLY, moves_only, True)
    chosen, q_refused = _policy(tables, state, key, layout, allowed)
    by_rule = mode == MODE_RULE
    action = jnp.where(by_rule, rule, chosen).astype(_I32)
    return action, q_refused & ~by_rule, ~by_rule


def _line_replace(tables: Tables, state: BattleState, line: Line, key, layout: AiLayout):
    """``LINE.replace``: ``line_replacement``'s slot; ``key`` unused, never refused."""
    return line_replacement(tables, state, line), jnp.bool_(False)


LINE = Player("line", _line_act, _line_replace)


@partial(jax.jit, static_argnames=("max_turns", "layout"))
def line_playout(
    tables: Tables,
    state: BattleState,
    line: Line,
    key,
    max_turns: int = 100,
    layout: AiLayout = VANILLA,
) -> PlayerResult:
    """``player_playout`` with ``LINE`` and ``line`` (module docstring); ``line`` may be
    traced, ``max_turns`` and ``layout`` are static."""
    return player_playout(tables, state, LINE, line, key, max_turns, layout)
