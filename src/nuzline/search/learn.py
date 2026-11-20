"""Learn-move prompts in the playout (K3d-exp design §3.5): the player's learn policy and
the resolve after a turn, before any faint replacement.

The decomp's ``TurnEnd`` pays exp and runs its learn-move prompts (battle_controller_player.c:
1846) before ``ReplaceFainted`` (:1848), so ``finish_turn`` resolves the learns first and the
replacements then read the moves after them. The engine side (``advance_learns``,
``needs_learn``, ``learn``, ``learn_options``) is ``nuzline.engine.levelup``.

This module never imports ``nuzline.search.playout``: ``settle_learns`` takes
``send_replacements`` and ``_carried_mul`` from its caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.coverage import acid_rain_reason
from nuzline.engine.levelup import GIVE_UP, advance_learns, learn, needs_learn
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import EXP_REFUSED_WINDOW_C, needs_replacement

__all__ = ["learn_policy", "learn_reason", "resolve_learns", "settle_learns", "settle_reason"]

_I32 = jnp.int32


def learn_policy(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: the playout's answer to a learn-move prompt: give the move up (ruling D3;
    a better learn policy belongs to the stronger simulated player)."""
    del tables, state
    return jnp.int32(GIVE_UP)


def resolve_learns(tables: Tables, state: BattleState) -> BattleState:
    """``state`` with every pending learnset scan run to its end (``advance_learns``),
    answering each prompt with ``learn_policy``. A ``lax.while_loop``: a batch with no
    prompt runs no ``learn``; each answer uses up a learnset entry, hence the bound."""
    state = advance_learns(tables, state)
    bound = SLOTS * tables.learnset_level.shape[1]

    def cond(carry):
        s, n = carry
        return needs_learn(s) & (n < bound)

    def body(carry):
        s, n = carry
        return learn(tables, s, learn_policy(tables, s)), n + 1

    state, _ = jax.lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state


def settle_learns(tables: Tables, state: BattleState, send, carried_mul):
    """``(state, refused)`` at a playout's root, before its pending player replacement:
    ``resolve_learns``, then the opponent's replacement where it is pending and the
    player's is not (a learn decision taken before the opponent's ``ReplaceFainted``), with
    the turn's carried ``powerMul`` (``carried_mul(state)``: a Pursuit hit leaves 20),
    and ``replace_mul`` cleared afterwards, as ``replacement_pass`` does. ``send`` is
    ``playout.send_replacements``. ``refused`` (``bool``): that send refused, or
    ``exp_refused`` is set. A one-pass ``while_loop``, so a batch where no row needs the
    send runs none."""
    state = resolve_learns(tables, state)

    def cond(carry):
        s, _, n = carry
        need = needs_replacement(s)
        return (n == 0) & (need[1] == 1) & (need[0] == 0)

    def body(carry):
        s, refused, n = carry
        sent, bad = send(tables, s, False, carried_mul(s))
        sent = sent._replace(replace_mul=jnp.zeros_like(sent.replace_mul))
        return sent, refused | bad, n + 1

    state, refused, _ = jax.lax.while_loop(cond, body, (state, jnp.bool_(False), jnp.int32(0)))
    return state, refused | (state.exp_refused != 0) | (state.acid_rain != 0)


_resolve_jit = jax.jit(resolve_learns)


def settle_reason(tables: Tables, state: BattleState, send_reason, carried_mul):
    """``(reason, state)``: ``settle_learns`` on the concrete ``state``, outside ``jit``
    (``playout.refusal_reason``'s pending root): ``resolve_learns``, then, where the
    opponent needs a replacement and the player does not, ``send_reason``
    (``playout._send_reason``, the concrete ``send_replacements``) with the carried
    ``powerMul`` (``carried_mul``) and ``replace_mul`` cleared after it. ``reason`` is that
    send's refusal, else ``learn_reason``, else ``None``."""
    state = _resolve_jit(tables, state)
    need = needs_replacement(state)
    if int(need[1]) == 1 and int(need[0]) == 0:
        reason, state = send_reason(tables, state, False, int(carried_mul(state)))
        if reason:
            return reason, state
        state = state._replace(replace_mul=jnp.zeros_like(state.replace_mul))
    return learn_reason(state), state


def learn_reason(state: BattleState) -> str | None:
    """The refusal message for a concrete state's ``exp_refused``, or ``None``."""
    code = int(state.exp_refused)
    if code == 0:
        return acid_rain_reason(state)
    if code == EXP_REFUSED_WINDOW_C:
        return (
            "a level-up from an opponent replacement fainting on entry (entry hazards) "
            "leaves learn-move work inside the opponent's replacement loop "
            "(K3d-exp design §3.3, window (c))"
        )
    return f"learn-move work refused (exp_refused = {code})"
