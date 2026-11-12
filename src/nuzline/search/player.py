"""Pluggable players (docs/superpowers/specs/2026-11-12-l1-line-player-design.md §7a): the
playout loop knows only a ``Player``, so a new player (search-based, learned) drops in
without touching it. Players are search heuristics, player choices only: the trainer AI,
``step`` and the chance model are ``playout``'s.

A ``Player`` is static (a name and two module-level functions, hashable, so it is a ``jit``
static argument); what varies between runs is its ``params``, any pytree of arrays
(traced, so a batch of them can be ``vmap``ped):

- ``act(tables, state, params, key, layout) -> (action, refused, fallback)``: the
  player's action (``int32``, a legal action); ``refused`` (``bool``) when this choice
  cannot be made (the opponent's distribution it needs is refused), which refuses the turn;
  ``fallback`` (``bool``) when the default policy (``playout_policy``) decided the turn.
- ``replace(tables, state, params) -> slot``: the party slot (``int32``, a
  ``legal_replacements(state, 0)`` slot) the player sends in at a replacement.

``player_playout`` is ``playout._run``'s loop with these two choices: the root's hidden
draws first (``root_state``), a pending player replacement resolved first (no key, no
turn), the same key splits per turn (``act`` gets the policy key). ``GREEDY``
(``playout_policy`` and ``replacement_policy``, ``params`` ignored) so returns what
``playout`` returns from that root and key. ``PlayerResult.fallback_turns`` counts the
turns ``act`` marked ``fallback``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.layout import VANILLA, AiLayout
from nuzline.data.tables import Tables
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import WINNER_PLAYER, needs_replacement
from nuzline.search.playout import (
    REFUSED_WON,
    _play,
    _policy,
    _resources_left,
    _select,
    replacement_pass,
    replacement_policy,
    root_state,
    send_replacements,
)

__all__ = ["GREEDY", "Player", "PlayerResult", "player_playout"]

_I32, _F32 = jnp.int32, jnp.float32


class Player(NamedTuple):  # static and hashable: passed to jit as a static argument
    name: str
    act: Callable  # (tables, state, params, key, layout) -> (action, refused, fallback)
    replace: Callable  # (tables, state, params) -> party slot int32


class PlayerResult(NamedTuple):
    won: jnp.ndarray  # int8[], 1 = the player won the battle, REFUSED_WON = refused
    player_faints: jnp.ndarray  # int8[]
    turns: jnp.ndarray  # int16[], turns played
    refused: jnp.ndarray  # int8[], 1 = stopped at a refused mechanic (no result)
    deaths: jnp.ndarray  # int8[6], the final state's per-slot death record
    ended: jnp.ndarray  # int8[], 1 = the battle is over (not cut off by max_turns)
    hp_left: jnp.ndarray  # float32[], player party HP / max HP at the end (refused: 0)
    pp_left: jnp.ndarray  # float32[], player party PP / full PP of its moves (refused: 0)
    fallback_turns: jnp.ndarray  # int16[], turns decided by the default policy


def _greedy_act(tables: Tables, state: BattleState, params, key, layout: AiLayout):
    action, refused = _policy(tables, state, key, layout)
    return action, refused, jnp.bool_(True)


def _greedy_replace(tables: Tables, state: BattleState, params) -> jnp.ndarray:
    return replacement_policy(tables, state)


GREEDY = Player("greedy", _greedy_act, _greedy_replace)


def _replace_while_needed(
    tables: Tables, state: BattleState, player: Player, params, refused, passes
):
    """``(state, refused)``: while the player needs a replacement (``passes`` at most)
    ``replacement_pass`` with ``player.replace``'s slot."""

    def cond(carry):
        state, _, n = carry
        return (needs_replacement(state)[0] == 1) & (n < passes)

    def body(carry):
        state, refused, n = carry
        slot = player.replace(tables, state, params)
        passed, bad = replacement_pass(tables, state, slot)
        return passed, refused | bad, n + 1

    state, refused, _ = jax.lax.while_loop(cond, body, (state, refused, jnp.int32(0)))
    return state, refused


def _player_run(
    tables: Tables,
    state: BattleState,
    player: Player,
    params,
    key,
    max_turns: int,
    layout: AiLayout = VANILLA,
) -> tuple[PlayerResult, BattleState]:
    """``(result, final state)`` of ``player_playout``."""
    state = root_state(tables, state, key)
    # A pending player replacement first (no key, no turn), as playout._resolve_pending.
    state, refused0 = _replace_while_needed(
        tables, state, player, params, jnp.bool_(False), SLOTS - 1
    )

    def cond(carry):
        s, _, turns, _, refused = carry
        return (s.over == 0) & (turns < max_turns) & ~refused

    def body(carry):
        s, key, turns, fallback, _ = carry
        key, k_policy, k_turn = jax.random.split(key, 3)
        action, act_refused, by_fallback = player.act(tables, s, params, k_policy, layout)
        played, refused, ai_switched, power_mul = _play(
            tables, s, jnp.asarray(action, _I32), k_turn, layout
        )
        played, sent_refused = send_replacements(tables, played, ai_switched, power_mul)
        played, pass_refused = _replace_while_needed(
            tables, played, player, params, sent_refused, SLOTS - 1
        )
        refused = refused | pass_refused | act_refused
        s = _select(refused, s, played)
        counted = jnp.where(refused, 0, 1)
        return s, key, turns + counted, fallback + jnp.where(by_fallback, counted, 0), refused

    init = (state, key, jnp.int32(0), jnp.int32(0), refused0)
    s, _, turns, fallback, refused = jax.lax.while_loop(cond, body, init)
    won = (s.over != 0) & (s.winner == WINNER_PLAYER)
    hp_left, pp_left = _resources_left(tables, s)
    result = PlayerResult(
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


@partial(jax.jit, static_argnames=("player", "max_turns", "layout"))
def player_playout(
    tables: Tables,
    state: BattleState,
    player: Player,
    params,
    key,
    max_turns: int = 100,
    layout: AiLayout = VANILLA,
) -> PlayerResult:
    """Play from ``state`` (its root draws made first, ``root_state``) with ``player``
    and its ``params`` (module docstring); ``params`` may be traced, ``player``,
    ``max_turns`` and ``layout`` are static."""
    return _player_run(tables, state, player, params, key, max_turns, layout)[0]
