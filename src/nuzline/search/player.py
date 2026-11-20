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
- ``replace(tables, state, params, key, layout) -> (slot, refused)``: the party slot
  (``int32``, a ``legal_replacements(state, 0)`` slot) the player sends in at a
  replacement; ``refused`` (``bool``) when this choice cannot be made, which refuses the
  playout. ``key`` lets a player sample there; it is folded from the playout's keys
  (``fold_in(key, REPLACE_FOLD + i)`` for pass ``i`` of a pending replacement before the
  first turn, ``fold_in(k_policy, 1 + i)`` for pass ``i`` after a turn), so it does not
  change the key stream.

``player_playout`` is ``playout._run``'s loop with these two choices: the root's hidden
draws first (``root_state``), a pending learn-move prompt and opponent replacement settled
(``learn.settle_learns``) and a pending player replacement resolved first (no turn), the
same key splits per turn (``act`` gets the policy key ``k_policy``), and each turn's learns
resolved before its replacements (``learn.resolve_learns``, as ``playout.finish_turn``).
Learn prompts are answered by ``learn.learn_policy`` (give the move up), not by the player:
a level-up only fills an empty move slot, so a line's move slots stay valid. ``GREEDY`` is
``playout_policy`` and ``replacement_policy`` (``params`` and the replacement key
ignored), so it returns what ``playout`` returns from that root and key.
``PlayerResult.fallback_turns`` counts the turns ``act`` marked ``fallback``.

A refused playout stops with ``refused = 1`` (``playout``'s rule); ``refusal_reason``
replays one ``player_playout`` outside the loop, with the same key splits, and names the
refused mechanic, as ``playout.refusal_reason`` does for ``playout_from_action``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.layout import VANILLA, AiLayout, layout_of
from nuzline.ai.policy import ai_path_refusals, ai_refusals
from nuzline.data.tables import Tables
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import WINNER_PLAYER, needs_replacement
from nuzline.search.learn import learn_reason, resolve_learns, settle_learns, settle_reason
from nuzline.search.playout import (
    REFUSED_WON,
    _carried_mul,
    _pass_reason,
    _play,
    _play_jit,
    _policy,
    _resources_left,
    _select,
    _send_reason,
    replacement_pass,
    replacement_policy,
    root_state,
    send_replacements,
    settle_war,
)

__all__ = ["GREEDY", "Player", "PlayerResult", "player_playout", "refusal_reason"]

_I32, _F32 = jnp.int32, jnp.float32
REPLACE_FOLD = 0x5EED  # fold_in base of a pending replacement's keys (module docstring)


class Player(NamedTuple):  # static and hashable: passed to jit as a static argument
    name: str
    act: Callable  # (tables, state, params, key, layout) -> (action, refused, fallback)
    replace: Callable  # (tables, state, params, key, layout) -> (slot int32, refused bool)


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


def _greedy_replace(tables: Tables, state: BattleState, params, key, layout: AiLayout):
    return replacement_policy(tables, state), jnp.bool_(False)


GREEDY = Player("greedy", _greedy_act, _greedy_replace)


def _replace_while_needed(
    tables: Tables,
    state: BattleState,
    player: Player,
    params,
    refused,
    passes,
    key,
    fold,
    layout: AiLayout,
):
    """``(state, refused)``: while the player needs a replacement (``passes`` at most)
    ``replacement_pass`` with ``player.replace``'s slot, pass ``i`` given
    ``fold_in(key, fold + i)``; a refused choice refuses."""

    def cond(carry):
        state, _, n = carry
        return (needs_replacement(state)[0] == 1) & (n < passes)

    def body(carry):
        state, refused, n = carry
        k = jax.random.fold_in(key, fold + n)
        slot, choice_refused = player.replace(tables, state, params, k, layout)
        passed, bad = replacement_pass(tables, state, jnp.asarray(slot, _I32))
        return passed, refused | bad | choice_refused, n + 1

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
    # A pending learn-move prompt and opponent replacement are settled first, then a pending
    # player replacement (no turn), as playout._resolve_pending.
    state, learn_refused = settle_learns(tables, state, send_replacements, _carried_mul)
    state, refused0 = _replace_while_needed(
        tables, state, player, params, learn_refused, SLOTS - 1, key, REPLACE_FOLD, layout
    )

    def cond(carry):
        s, _, turns, _, refused = carry
        return (s.over == 0) & (turns < max_turns) & ~refused

    def body(carry):
        s, key, turns, fallback, _ = carry
        key, k_policy, k_turn = jax.random.split(key, 3)
        s = settle_war(s, k_turn, tables)  # an open weather war, before the decisions (G-26)
        action, act_refused, by_fallback = player.act(tables, s, params, k_policy, layout)
        played, refused, ai_switched, power_mul = _play(
            tables, s, jnp.asarray(action, _I32), k_turn, layout
        )
        # The turn's learn-move prompts before any replacement, as playout.finish_turn.
        played = resolve_learns(tables, played)
        played, sent_refused = send_replacements(tables, played, ai_switched, power_mul)
        played, pass_refused = _replace_while_needed(
            tables, played, player, params, sent_refused, SLOTS - 1, k_policy, 1, layout
        )
        # A newcomer's hazard faint inside the replacement loop leaves its prompt (E3-B2-21), but
        # not at a finished battle (ruling D8).
        played = _select(played.over != 0, played, resolve_learns(tables, played))
        refused = refused | pass_refused | act_refused | (played.exp_refused != 0)
        s = _select(refused, s, played)
        counted = jnp.where(refused, 0, 1)
        return s, key, turns + counted, fallback + jnp.where(by_fallback, counted, 0), refused

    init = (state, key, jnp.int32(0), jnp.int32(0), refused0)
    # At ``max_turns`` 0 ``cond`` is false at the start: no turn in the program (playout._run).
    if max_turns > 0:
        s, _, turns, fallback, refused = jax.lax.while_loop(cond, body, init)
    else:
        s, _, turns, fallback, refused = init
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


@partial(jax.jit, static_argnames=("player", "layout"))
def _act_jit(tables: Tables, state: BattleState, player: Player, params, key, layout: AiLayout):
    return player.act(tables, state, params, key, layout)


@partial(jax.jit, static_argnames=("player", "layout"))
def _replace_jit(tables: Tables, state: BattleState, player: Player, params, key, layout):
    return player.replace(tables, state, params, key, layout)


_resolve_jit = jax.jit(resolve_learns)


def _ai_reason(tables: Tables, state: BattleState, layout: AiLayout) -> str | None:
    """The AI refusals of the concrete ``state`` (``ai_refusals``, then the scoring path's
    ``ai_path_refusals``), joined; ``None`` when there are none."""
    reasons = ai_refusals(tables, state, layout=layout) or ai_path_refusals(
        tables, state, layout=layout
    )
    return "; ".join(reasons) if reasons else None


def _passes_reason(
    tables: Tables, state: BattleState, player: Player, params, key, fold, layout: AiLayout
):
    """``(reason, state)``: ``_replace_while_needed`` outside the loop, stopping at the
    first refused pass (``reason``; ``state`` before it) or after the last (``None``)."""
    for i in range(SLOTS - 1):
        if int(needs_replacement(state)[0]) != 1:
            break
        k = jax.random.fold_in(key, fold + i)
        slot, choice_refused = _replace_jit(tables, state, player, params, k, layout)
        passed, bad = replacement_pass(tables, state, jnp.asarray(slot, _I32))
        if bool(choice_refused):
            reason = _ai_reason(tables, state, layout)
            return reason or "the player's replacement choice is refused", state
        if bool(bad):
            return _pass_reason(tables, state, int(slot))[0] or "a replacement is refused", state
        state = passed
    return None, state


def _replay(
    tables: Tables, state: BattleState, player: Player, params, key, max_turns: int, layout
) -> tuple[str | None, BattleState, int]:
    """``(reason, state, turns)`` of ``_player_run`` replayed outside the loop: ``reason``
    names the refused mechanic (``None``: not refused), ``state`` and ``turns`` are the
    run's final state and turn count. At a refused turn the state is the one before it,
    as in the run; at a refused pending replacement it is the one before that pass (the
    run's passes go on)."""
    state = root_state(tables, state, key)
    reason, state = settle_reason(tables, state, _send_reason, _carried_mul)
    if reason:
        return reason, state, 0
    reason, state = _passes_reason(tables, state, player, params, key, REPLACE_FOLD, layout)
    if reason:
        return reason, state, 0
    for turns in range(max_turns):
        if int(state.over) != 0:
            return None, state, turns
        key, k_policy, k_turn = jax.random.split(key, 3)
        state = settle_war(state, k_turn, tables)
        action, act_refused, _ = _act_jit(tables, state, player, params, k_policy, layout)
        played, ai_refused, ai_switched, power_mul = _play_jit(
            tables, state, jnp.asarray(action, _I32), k_turn, layout
        )
        if bool(act_refused) or bool(ai_refused):
            reason = _ai_reason(tables, state, layout)
            if reason is None and bool(act_refused):
                reason = "the opponent's action distribution is refused"
            return reason or "the opponent's AI is refused", state, turns
        played = _resolve_jit(tables, played)
        sent, sent_refused = send_replacements(tables, played, ai_switched, power_mul)
        # Refused learn work first, as playout.refusal_reason (a window (c) set by the send).
        reason = learn_reason(sent)
        if reason:
            return reason, state, turns
        if bool(sent_refused):
            reason = _send_reason(tables, played, ai_switched, power_mul)[0]
            return reason or "the opponent's replacement is refused", state, turns
        reason, after = _passes_reason(tables, sent, player, params, k_policy, 1, layout)
        reason = reason or learn_reason(after)
        if reason:
            return reason, state, turns
        state = _resolve_jit(tables, after)  # the loop's hazard-faint prompts (E3-B2-21)
    return None, state, max_turns


def refusal_reason(
    tables: Tables,
    state: BattleState,
    player: Player,
    params,
    key,
    max_turns: int = 100,
    layout: AiLayout | None = None,
) -> str | None:
    """Why ``player_playout(tables, state, player, params, key, max_turns, layout)`` is
    refused, or ``None`` when it is not: the playout is replayed outside ``jit`` with the
    same key splits (the root draws, the pending replacement's passes, per turn
    ``split(key, 3)``, the passes after a turn), ``player.act`` and ``player.replace``
    jitted per call, and the refused step is asked on its concrete state: the AI
    (``ai_refusals``, then ``ai_path_refusals``) for a refused turn or player choice, the
    replacement mirrors (``playout._send_reason``, ``_pass_reason``) for a refused
    replacement, ``learn.settle_reason`` / ``learn_reason`` for refused learn-move work. ``layout`` defaults to the concrete ``tables``' own (``layout_of``)."""
    layout = layout_of(tables) if layout is None else layout
    return _replay(tables, state, player, params, key, max_turns, layout)[0]
