"""Stage A playouts (docs/design.md §4.3): play a battle on to its end from a state.

Each turn the player picks ``playout_policy``'s action, the opponent ``ai_policy``'s
under a sampled ``AiOutcome``, the turn's chance events come from ``sample_outcome``
and ``step`` plays it; a fainted opponent is then replaced (``send_replacements``). A
playout ends at the first player faint or the end of the battle (``BattleState.over``),
after ``max_turns`` turns, or when the AI meets a refused mechanic.

``playout_policy`` is the player's default policy, a search heuristic (it is not the
trainer AI, which follows the decomp): a one-ply expectimax over the player's legal
actions against the opponent's exact action distribution (``ai_action_distribution``).
Each pair of actions is valued ``P(no player faint this turn) * (1 + fraction of the
target's current HP removed)`` from the exact non-critical damage tables (16 rolls) and
the accuracy of both moves: the player is safe unless the opponent's move KOs, and the
opponent's move does not run when the player acts first (``action_order``, a speed tie
counted as 1/2) and KOs. A switch is valued against the incoming Pokemon (stages reset);
crits, multi-hit counts, secondary effects, switch-in effects and end-of-turn damage are
left out. ``key`` breaks exact ties only.

The opponent's replacement follows ``BattleAI_PostKOSwitchIn`` (``nuzline.ai.replacement``,
docs/ai/roark.md 11.2), with ``powerMul`` 20 on a turn where its Pursuit hit the
player's switch. A replacement in a turn the AI switched is refused (roark.md 14, item
(3): ``Task_TrainerShowPartyMenu`` reads the stale ``aiSwitchedPartySlot``).

Refusals cannot raise under ``jit``: a playout that meets one stops with
``PlayoutResult.refused = 1`` and ``won = REFUSED_WON`` (-1; the turn is not played), so
summing ``won`` over refused playouts cannot pass for losses. Callers must exclude
refused playouts or refuse loudly; ``refusal_reason`` replays one playout outside
``jit`` and names the refused mechanic for the message.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.outcome import sample_ai_outcome
from nuzline.ai.policy import UnsupportedAi, ai_action_distribution, ai_policy, ai_refusals
from nuzline.ai.replacement import post_ko_switch_in
from nuzline.data.tables import Tables
from nuzline.engine.actions import (
    MOVE_ACTIONS,
    NUM_ACTIONS,
    NUM_OPP_ACTIONS,
    legal_actions,
)
from nuzline.engine.damage import damage_table
from nuzline.engine.order import action_order, is_switch, selected_move
from nuzline.engine.outcome import empty_outcome, hit_threshold, sample_outcome
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import (
    WINNER_PLAYER,
    needs_replacement,
    replace,
    replacement_power_mul,
    step,
)

__all__ = [
    "REFUSED_WON",
    "PlayoutResult",
    "batched_playout",
    "playout",
    "playout_from_action",
    "playout_policy",
    "refusal_reason",
    "replacement_refusal",
    "send_replacements",
]

_I32, _F32 = jnp.int32, jnp.float32
REFUSED_WON = -1  # PlayoutResult.won of a refused playout


class PlayoutResult(NamedTuple):
    won: jnp.ndarray  # int8[], 1 = the player won the battle, REFUSED_WON = refused
    player_faints: jnp.ndarray  # int8[]
    turns: jnp.ndarray  # int16[], turns played
    refused: jnp.ndarray  # int8[], 1 = stopped at a refused mechanic (no result)


def _select(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def _facing(state: BattleState, side, action) -> BattleState:
    """``state`` with ``side``'s switch ``action`` made (the incoming Pokemon active,
    stages reset), as the other side's move will meet it; unchanged otherwise."""
    switching = is_switch(action)
    slot = jnp.where(switching, action - MOVE_ACTIONS, state.active[side])
    stages = jnp.where(switching, 0, state.stages[side])
    return state._replace(
        active=state.active.at[side].set(slot.astype(state.active.dtype)),
        stages=state.stages.at[side].set(stages.astype(state.stages.dtype)),
    )


def _move_odds(tables: Tables, state: BattleState, side, actions, hit):
    """``(P(KO), expected fraction of the target's HP removed)`` of ``side``'s action."""
    action = actions[side]
    target = _facing(state, 1 - side, actions[1 - side])
    move, _ = selected_move(tables, state, side, action)
    damage = damage_table(tables, target, side, move, False)  # [16] non-critical rolls
    hp = target.hp[1 - side, target.active[1 - side]].astype(_I32)
    hp_ = jnp.maximum(hp, 1).astype(_F32)
    p = jnp.where(action < MOVE_ACTIONS, hit, 0.0)
    ko = p * jnp.mean((damage >= hp).astype(_F32))
    removed = p * jnp.mean(jnp.minimum(damage, hp).astype(_F32)) / hp_
    return ko, removed


def _pair_value(tables: Tables, state: BattleState, a, b) -> jnp.ndarray:
    actions = jnp.stack([a, b]).astype(_I32)
    hit = jnp.clip(hit_threshold(tables, state, actions), 0, 100).astype(_F32) / 100
    ko_player, removed = _move_odds(tables, state, 0, actions, hit[0])
    ko_opp, _ = _move_odds(tables, state, 1, actions, hit[1])
    ties = [empty_outcome()._replace(speed_tie=jnp.int8(tie)) for tie in (0, 1)]
    player_first = [action_order(tables, state, actions, o)[0] == 0 for o in ties]
    first = jnp.mean(jnp.stack(player_first).astype(_F32))
    safe = 1.0 - ko_opp * (1.0 - first * ko_player)
    return safe * (1.0 + removed)


@jax.jit
def _choose(tables: Tables, state: BattleState, q, key) -> jnp.ndarray:
    legal = legal_actions(tables, state, 0)
    mine, theirs = jnp.arange(NUM_ACTIONS), jnp.arange(NUM_OPP_ACTIONS)
    row = jax.vmap(lambda a, b: _pair_value(tables, state, a, b), in_axes=(None, 0))
    values = jax.vmap(lambda a: row(a, theirs))(mine)  # [NUM_ACTIONS, NUM_OPP_ACTIONS]
    score = jnp.where(legal, values @ q, -jnp.inf)
    tied = score == jnp.max(score)
    return jax.random.choice(key, NUM_ACTIONS, p=tied / jnp.sum(tied)).astype(_I32)


def _policy(tables: Tables, state: BattleState, key) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(action, refused)``: ``playout_policy``'s action, and whether the opponent's
    distribution it was chosen against is refused (traced, a refused distribution is
    NaN; the action is then meaningless and the caller must refuse the playout)."""
    q = ai_action_distribution(tables, state)
    refused = jnp.any(jnp.isnan(q))
    return _choose(tables, state, jnp.where(refused, 0.0, q), key), refused


def playout_policy(tables: Tables, state: BattleState, key) -> jnp.ndarray:
    """``int32``: the player's action with the highest expected value (module docstring),
    a uniformly random one among exact ties; illegal actions score -inf. Raises
    ``UnsupportedAi`` on a concrete state whose AI is refused; traced, the action on a
    refused distribution is meaningless (``_run`` refuses the playout instead)."""
    return _policy(tables, state, key)[0]


@jax.jit
def send_replacements(
    tables: Tables, state: BattleState, ai_switched, power_mul
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, refused)``: the opponent's fainted active Pokemon replaced by
    ``post_ko_switch_in`` (``power_mul`` is the turn's ``battleCtx->powerMul``), again
    while each newcomer faints on entry (Stealth Rock), as ``TurnEnd`` re-enters
    ``ReplaceFainted``. ``refused`` (``bool``) when a replacement is needed after the AI
    switched this turn (``ai_switched``) or ``post_ko_switch_in`` refuses; the state is
    then left as it was at that point."""

    def once(_, carry):  # a loop, not unrolled: one copy of the replacement rule
        state, refused = carry
        need = needs_replacement(state)[1] == 1
        slot = post_ko_switch_in(tables, state, power_mul)
        bad = need & (jnp.asarray(ai_switched) | (slot < 0))
        replaced = replace(tables, state, 1, jnp.maximum(slot, 0))
        return _select(need & ~bad, replaced, state), refused | bad

    return jax.lax.fori_loop(0, SLOTS - 1, once, (state, jnp.bool_(False)))


def _play(tables: Tables, state: BattleState, player_action, key):
    """One turn after the player's choice, before the opponent's replacement:
    ``(state after step, ai_refused, ai_switched, power_mul)``."""
    k_ai, k_turn = jax.random.split(key)
    opp_action = ai_policy(tables, state, sample_ai_outcome(k_ai, tables, state))
    ai_refused = opp_action < 0  # switching.REFUSED_ACTION
    actions = jnp.stack([player_action, jnp.maximum(opp_action, 0)]).astype(_I32)
    power_mul = replacement_power_mul(tables, state, actions)
    played = step(tables, state, actions, sample_outcome(k_turn, tables, state, actions))
    return played, ai_refused, is_switch(actions[1]), power_mul


def _turn(tables: Tables, state: BattleState, player_action, key):
    """One turn after the player's choice: ``(state, refused)``."""
    played, refused, ai_switched, power_mul = _play(tables, state, player_action, key)
    played, replace_refused = send_replacements(tables, played, ai_switched, power_mul)
    refused = refused | replace_refused
    return _select(refused, state, played), refused


def _fixed(turns, first_action):
    return (turns == 0) & (first_action >= 0)


def _first_action(turns, first_action, chosen):
    return jnp.where(_fixed(turns, first_action), first_action, chosen)


def _run(tables: Tables, state: BattleState, key, max_turns: int, first_action):
    def cond(carry):
        s, _, turns, refused = carry
        return (s.over == 0) & (turns < max_turns) & ~refused

    def body(carry):
        s, key, turns, _ = carry
        key, k_policy, k_turn = jax.random.split(key, 3)
        chosen, q_refused = _policy(tables, s, k_policy)
        # A refused distribution refuses the turn whenever the player's choice uses it.
        q_refused = q_refused & ~_fixed(turns, first_action)
        action = _first_action(turns, first_action, chosen)
        played, refused = _turn(tables, s, action.astype(_I32), k_turn)
        refused = refused | q_refused
        s = _select(q_refused, s, played)
        return s, key, turns + jnp.where(refused, 0, 1), refused

    init = (state, key, jnp.int32(0), jnp.bool_(False))
    s, _, turns, refused = jax.lax.while_loop(cond, body, init)
    won = (s.over != 0) & (s.winner == WINNER_PLAYER)
    return PlayoutResult(
        won=jnp.where(refused, REFUSED_WON, won).astype(jnp.int8),
        player_faints=s.player_faints.astype(jnp.int8),
        turns=turns.astype(jnp.int16),
        refused=refused.astype(jnp.int8),
    )


@partial(jax.jit, static_argnames=("max_turns",))
def playout(tables: Tables, state: BattleState, key, max_turns: int = 100) -> PlayoutResult:
    """Play from ``state`` with ``playout_policy`` for the player (module docstring)."""
    return _run(tables, state, key, max_turns, jnp.int32(-1))


@partial(jax.jit, static_argnames=("max_turns",))
def playout_from_action(
    tables: Tables, state: BattleState, action, key, max_turns: int = 100
) -> PlayoutResult:
    """``playout`` with the player's first action fixed to ``action`` (a legal action);
    the same ``key`` gives every first action the same draws (common random numbers)."""
    return _run(tables, state, key, max_turns, jnp.asarray(action, _I32))


@partial(jax.jit, static_argnames=("max_turns",))
def batched_playout(tables: Tables, states: BattleState, keys, max_turns: int = 100):
    """``playout`` over a leading batch axis of ``states`` and ``keys`` (``jax.vmap``,
    ``tables`` shared); ``max_turns`` is static."""
    return jax.vmap(lambda s, k: playout(tables, s, k, max_turns))(states, keys)


def replacement_refusal(tables: Tables, state: BattleState, ai_switched, power_mul) -> str | None:
    """Why ``send_replacements`` refuses the concrete ``state`` (a state after ``step``),
    or ``None`` when it does not; mirrors its loop outside ``jit``."""
    for _ in range(SLOTS - 1):
        if int(needs_replacement(state)[1]) != 1:
            return None
        if bool(ai_switched):
            return (
                "replacement: the opponent's Pokemon fainted in a turn its AI switched; "
                "Task_TrainerShowPartyMenu reads the stale aiSwitchedPartySlot "
                "(docs/ai/roark.md 14, item 3); not implemented"
            )
        try:
            slot = post_ko_switch_in(tables, state, power_mul)
        except UnsupportedAi as e:
            return str(e)
        state = replace(tables, state, 1, slot)
    return None


_play_jit = jax.jit(_play)
_turn_jit = jax.jit(_turn)
_policy_jit = jax.jit(_policy)


def refusal_reason(
    tables: Tables, state: BattleState, action, key, max_turns: int = 100
) -> str | None:
    """Why ``playout_from_action(tables, state, action, key, max_turns)`` is refused, or
    ``None`` when it is not: the playout is replayed turn by turn outside the loop (the
    same key splits) and the refused turn's AI (``ai_refusals``, also when the player's
    choice met a refused distribution) or replacement (``replacement_refusal``) is asked
    on its concrete state."""
    first = jnp.asarray(action, _I32)
    for turns in range(max_turns):
        if int(state.over) != 0:
            return None
        key, k_policy, k_turn = jax.random.split(key, 3)
        chosen, q_refused = _policy_jit(tables, state, k_policy)
        q_refused = bool(q_refused) and not bool(_fixed(turns, first))
        player_action = _first_action(turns, first, chosen).astype(_I32)
        played, refused = _turn_jit(tables, state, player_action, k_turn)
        if q_refused or bool(refused):
            reasons = ai_refusals(tables, state)
            if reasons:
                return "; ".join(reasons)
            if q_refused:
                return "the opponent's action distribution is refused"
            played, _, ai_switched, power_mul = _play_jit(tables, state, player_action, k_turn)
            return replacement_refusal(tables, played, ai_switched, power_mul)
        state = played
    return None
