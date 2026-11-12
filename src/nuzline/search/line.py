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

``line_playout`` is ``playout._run``'s loop with these two choices: the root's hidden draws
first (``root_state``), the same key splits per turn, so with ``empty_line()`` it returns
what ``playout`` returns from that root and key. ``LineResult.fallback_turns`` counts the
turns played in ``MODE_FALLBACK`` or ``MODE_MOVES_ONLY``.
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
from nuzline.engine.step import WINNER_PLAYER, needs_replacement
from nuzline.search.playout import (
    REFUSED_WON,
    _carried_mul,
    _play,
    _policy,
    _resources_left,
    _select,
    replacement_pass,
    replacement_policy,
    root_state,
    send_replacements,
)

__all__ = [
    "MODE_FALLBACK",
    "MODE_MOVES_ONLY",
    "MODE_RULE",
    "Line",
    "LineResult",
    "empty_line",
    "line_action",
    "line_playout",
    "line_replacement",
]

_I32, _F32 = jnp.int32, jnp.float32
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


class LineResult(NamedTuple):
    won: jnp.ndarray  # int8[], 1 = the player won the battle, REFUSED_WON = refused
    player_faints: jnp.ndarray  # int8[]
    turns: jnp.ndarray  # int16[], turns played
    refused: jnp.ndarray  # int8[], 1 = stopped at a refused mechanic (no result)
    deaths: jnp.ndarray  # int8[6], the final state's per-slot death record
    ended: jnp.ndarray  # int8[], 1 = the battle is over (not cut off by max_turns)
    hp_left: jnp.ndarray  # float32[], player party HP / max HP at the end (refused: 0)
    pp_left: jnp.ndarray  # float32[], player party PP / full PP of its moves (refused: 0)
    fallback_turns: jnp.ndarray  # int16[], turns decided by the fallback (modes 0 and 2)


def _replace_while_needed(tables: Tables, state: BattleState, line: Line, refused, passes):
    """``(state, refused)``: while the player needs a replacement (``passes`` at most)
    ``replacement_pass`` with ``line_replacement``'s slot."""

    def cond(carry):
        state, _, n = carry
        return (needs_replacement(state)[0] == 1) & (n < passes)

    def body(carry):
        state, refused, n = carry
        passed, bad = replacement_pass(tables, state, line_replacement(tables, state, line))
        return passed, refused | bad, n + 1

    state, refused, _ = jax.lax.while_loop(cond, body, (state, refused, jnp.int32(0)))
    return state, refused


def _line_run(
    tables: Tables,
    state: BattleState,
    line: Line,
    key,
    max_turns: int,
    layout: AiLayout = VANILLA,
) -> tuple[LineResult, BattleState]:
    """``(result, final state)`` of ``line_playout``."""
    line = Line(*(jnp.asarray(f, _I32) for f in line))
    state = root_state(tables, state, key)
    # A pending player replacement first (no key, no turn), as playout._resolve_pending.
    state, refused0 = _replace_while_needed(tables, state, line, jnp.bool_(False), SLOTS - 1)
    moves_only = jnp.arange(NUM_ACTIONS) < MOVE_ACTIONS

    def cond(carry):
        s, _, turns, _, refused = carry
        return (s.over == 0) & (turns < max_turns) & ~refused

    def body(carry):
        s, key, turns, fallback, _ = carry
        key, k_policy, k_turn = jax.random.split(key, 3)
        rule, mode = line_action(tables, s, line)
        allowed = jnp.where(mode == MODE_MOVES_ONLY, moves_only, True)
        chosen, q_refused = _policy(tables, s, k_policy, layout, allowed)
        by_rule = mode == MODE_RULE
        # A refused distribution refuses the turn only when the fallback's action is used.
        q_refused = q_refused & ~by_rule
        action = jnp.where(by_rule, rule, chosen).astype(_I32)
        played, refused, ai_switched, power_mul = _play(tables, s, action, k_turn, layout)
        played, sent_refused = send_replacements(tables, played, ai_switched, power_mul)
        played, pass_refused = _replace_while_needed(tables, played, line, sent_refused, SLOTS - 1)
        refused = refused | pass_refused | q_refused
        s = _select(refused, s, played)
        counted = jnp.where(refused, 0, 1)
        return s, key, turns + counted, fallback + jnp.where(by_rule, 0, counted), refused

    init = (state, key, jnp.int32(0), jnp.int32(0), refused0)
    s, _, turns, fallback, refused = jax.lax.while_loop(cond, body, init)
    won = (s.over != 0) & (s.winner == WINNER_PLAYER)
    hp_left, pp_left = _resources_left(tables, s)
    result = LineResult(
        won=jnp.where(refused, REFUSED_WON, won).astype(jnp.int8),
        player_faints=s.player_faints.astype(jnp.int8),
        turns=turns.astype(jnp.int16),
        refused=refused.astype(jnp.int8),
        deaths=s.deaths.astype(jnp.int8),
        ended=(s.over != 0).astype(jnp.int8),
        hp_left=jnp.where(refused, 0.0, hp_left).astype(_F32),
        pp_left=jnp.where(refused, 0.0, pp_left).astype(_F32),
        fallback_turns=fallback.astype(jnp.int16),
    )
    return result, s


@partial(jax.jit, static_argnames=("max_turns", "layout"))
def line_playout(
    tables: Tables,
    state: BattleState,
    line: Line,
    key,
    max_turns: int = 100,
    layout: AiLayout = VANILLA,
) -> LineResult:
    """Play from ``state`` (its root draws made first, ``root_state``) with ``line`` for
    the player (module docstring); ``line`` may be traced, ``max_turns`` and ``layout``
    are static."""
    return _line_run(tables, state, line, key, max_turns, layout)[0]
