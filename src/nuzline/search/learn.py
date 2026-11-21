"""Learn-move prompts in the playout (K3d-exp design §3.5): the player's learn policy and
the resolve after a turn, before any faint replacement.

The decomp's ``TurnEnd`` pays exp and runs its learn-move prompts (battle_controller_player.c:
1846) before ``ReplaceFainted`` (:1848), so ``finish_turn`` resolves the learns first and the
replacements then read the moves after them. A prompt inside a turn is answered at the pause
(``playout.step_through``, E3-B2-21). The engine side (``advance_learns``,
``needs_learn``, ``learn``, ``learn_options``) is ``nuzline.engine.levelup``.

This module never imports ``nuzline.search.playout``: ``settle_learns`` takes
``send_replacements`` and ``_carried_mul`` from its caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.levelup import GIVE_UP, advance_learns, learn, learn_options, needs_learn
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import needs_replacement

__all__ = ["learn_answer_array", "learn_policy", "resolve_learns", "settle_learns", "settle_reason"]

_I32 = jnp.int32


def learn_answer_array(
    tables: Tables,
    members: Sequence[tuple[int, int, Mapping[tuple[int, int], int | None]] | None],
) -> np.ndarray:
    """``int32[SLOTS, width]``: the growth plan's answers as the playout takes them (E3-B3-16).
    ``members[slot]`` is ``(species, form, answers)`` of party slot ``slot`` (or ``None``: the base
    plan), ``answers`` the plan's ``(level, move) -> slot or None`` (``GrowthPlan.answers``). The
    columns are the entries of the member's learnset row (``Pokemon_LevelUpMove`` offers each
    entry of exactly the current level, src/pokemon.c:3935-3966), so an entry is answered by its
    own ``(level, move)``; an offer without an answer, ``None`` and the base plan give up
    (``GIVE_UP``). The array is a batch input of the playout, not a ``BattleState`` field."""
    width = tables.learnset_level.shape[1]
    out = np.full((SLOTS, width), GIVE_UP, np.int32)
    for slot, member in enumerate(members):
        if member is None:
            continue
        species, form, answers = member
        row = int(tables.species_form_row[species, form])
        if row < 0:
            continue
        levels = np.asarray(tables.learnset_level[row])
        moves = np.asarray(tables.learnset_move[row])
        for (level, move), choice in answers.items():
            if choice is None:
                continue
            if not 0 <= choice < GIVE_UP:
                raise ValueError(f"answer {choice} is not a move slot 0-{GIVE_UP - 1} or None")
            out[slot, (levels == level) & (moves == move)] = choice
    return out


def learn_policy(
    tables: Tables, state: BattleState, answers: jnp.ndarray | None = None
) -> jnp.ndarray:
    """``int32``: the playout's answer to the pending learn-move prompt, from the growth plan's
    ``answers`` (:func:`learn_answer_array`, indexed by the learner's party slot and the learnset
    column of the offered move). The answer does not depend on which ``learn_site`` paused the
    battle. No plan (``answers`` is ``None``) or an offer the plan does not answer: give the move
    up (``GIVE_UP``; an empty slot learns with no prompt, ``advance_learns``). An answer the
    prompt does not allow (an HM slot, ``learn_options``) is given up as well."""
    if answers is None:
        return jnp.int32(GIVE_UP)
    slot = jnp.argmax(state.learn_level > 0)  # the lowest slot with learn work (needs_learn)
    col = jnp.minimum(state.learn_cursor[slot].astype(_I32), answers.shape[1] - 1)
    option = jnp.clip(jnp.asarray(answers)[slot, col].astype(_I32), 0, GIVE_UP)
    return jnp.where(learn_options(tables, state)[option], option, jnp.int32(GIVE_UP))


def resolve_learns(
    tables: Tables, state: BattleState, answers: jnp.ndarray | None = None
) -> BattleState:
    """``state`` with every pending learnset scan run to its end (``advance_learns``),
    answering each prompt with ``learn_policy(…, answers)``. A ``lax.while_loop``: a batch with no
    prompt runs no ``learn``; each answer uses up a learnset entry, hence the bound."""
    state = advance_learns(tables, state)
    bound = SLOTS * tables.learnset_level.shape[1]

    def cond(carry):
        s, n = carry
        return needs_learn(s) & (n < bound)

    def body(carry):
        s, n = carry
        return learn(tables, s, learn_policy(tables, s, answers)), n + 1

    state, _ = jax.lax.while_loop(cond, body, (state, jnp.int32(0)))
    return state


def settle_learns(tables: Tables, state: BattleState, send, carried_mul, answers=None):
    """``(state, refused)`` at a playout's root, before its pending player replacement:
    ``resolve_learns``, then the opponent's replacement where it is pending and the
    player's is not (a learn decision taken before the opponent's ``ReplaceFainted``), with
    the turn's carried ``powerMul`` (``carried_mul(state)``: a Pursuit hit leaves 20),
    and ``replace_mul`` cleared afterwards, as ``replacement_pass`` does. ``send`` is
    ``playout.send_replacements``. ``refused`` (``bool``): that send refused (learn work never
    refuses since E3-B2-21: the prompt pauses the turn). A one-pass ``while_loop``, so a batch
    where no row needs the send runs none. A newcomer the send's hazards faint leaves learn work
    (``step.replace`` marks it, no scan), as the root's player passes after it can: the caller
    resolves both after those passes, before the first turn (``player._player_run``,
    ``playout._resolve_pending``; a finished battle keeps it, ruling D8)."""
    state = resolve_learns(tables, state, answers)

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
    return state, refused


_resolve_jit = jax.jit(resolve_learns)


def settle_reason(tables: Tables, state: BattleState, send_reason, carried_mul, answers=None):
    """``(reason, state)``: ``settle_learns`` on the concrete ``state``, outside ``jit``
    (``playout.refusal_reason``'s pending root): ``resolve_learns``, then, where the
    opponent needs a replacement and the player does not, ``send_reason``
    (``playout._send_reason``, the concrete ``send_replacements``) with the carried
    ``powerMul`` (``carried_mul``) and ``replace_mul`` cleared after it. ``reason`` is that
    send's refusal, else ``None``. The send's learn work is left, as ``settle_learns`` leaves
    it."""
    state = _resolve_jit(tables, state, answers)
    need = needs_replacement(state)
    if int(need[1]) == 1 and int(need[0]) == 0:
        reason, state = send_reason(tables, state, False, int(carried_mul(state)))
        if reason:
            return reason, state
        state = state._replace(replace_mul=jnp.zeros_like(state.replace_mul))
    return None, state
