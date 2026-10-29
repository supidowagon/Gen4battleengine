"""The opponent's replacement after a faint: ``BattleAI_PostKOSwitchIn``
(battle_lib.c:7923) and its fallback in ``Task_TrainerShowPartyMenu``
(battle_display.c:4543-4556). ``docs/ai/roark.md`` 11.2 is the specification.

Candidates are the opponent's living, non-active party slots; the defender ``D`` is the
player's active Pokemon (``BattleSystem_RandomOpponent`` is ``attacker ^ 1`` in
singles, so nothing is drawn).

1. Stage 1: each candidate scores ``TypeMatchupMultiplier(t1) + TypeMatchupMultiplier(t2)``
   of its own types against ``D``'s, in a ``u8`` that wraps (the "Post-KO Switch-In AI
   Scoring Overflow" quirk). In the decomp's loop the highest non-zero score is tried
   first (ties to the lower slot) and kept when one of its moves is super effective
   against ``D`` (``BattleSystem_CalcEffectiveness``), else discarded; that is the
   highest-scoring candidate, lowest slot first, among those with such a move.
2. Stage 2, when stage 1 picks nobody: every candidate move with power other than 1
   scores ``CalcMoveDamage`` (criticalMul 1, ``battleCtx->powerMul``) as used by the
   *fainted* battler against ``D``, kept in the ``u8``, then ``ApplyTypeChart`` (STAB of
   the fainted battler, the chart), again in the ``u8``. The highest non-zero score wins,
   ties to the lower slot.
3. Neither: the first living non-active slot.

``power_mul`` is ``battleCtx->powerMul`` at the replacement: 20 after a Pursuit hit this
turn (``BtlCmd_TryPursuit``, battle_script.c:6928), since only ``BattleContext_Init``
(battle_lib.c:1980, after ``ReplaceFainted``) resets it; 10 otherwise.

Refused (``UnsupportedAi`` on concrete inputs, ``REFUSED_SLOT`` traced): stage 2 reached
with the first candidate's slot 0 empty or of power 1. That slot is skipped but the
comparison after it still runs (battle_lib.c:8076), on the ``score`` left over from stage
1, which this module does not track. The type-chart modifiers (Levitate, Wonder Guard,
Normalize, Scrappy, Adaptability, Tinted Lens, Filter, Solid Rock, Expert Belt, Iron
Ball, the Foresight-style volatiles, the variable-type moves) are not implemented here:
``policy.ai_refusals`` refuses every one of them for the fight, so the precondition is
that the fight passed it (the coverage registry and every ``ai_policy`` call check it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.ai.policy import UnsupportedAi
from nuzline.ai.switching import SUPER_EFFECTIVE, calc_effectiveness_flags, chart_multipliers
from nuzline.data.tables import Tables
from nuzline.engine.damage import calc_move_damage
from nuzline.engine.state import MOVES, SLOTS, BattleState

__all__ = ["REFUSED_SLOT", "post_ko_switch_in"]

REFUSED_SLOT = -1  # post_ko_switch_in of a refused state (traced only)
_I32 = jnp.int32
_U8 = 0xFF
_MATCHUP_BASE = 40  # BattleSystem_TypeMatchupMultiplier (battle_lib.c:3009): mul = 40


def _matchup(tables: Tables, atk_type, def_types) -> jnp.ndarray:
    """``BattleSystem_TypeMatchupMultiplier``: 40 times each matching chart row in tenths
    (the Ghost rows included); type 2 counts only when it differs from type 1. Every
    product of 40 and the factors 0, 5, 10, 20 divides exactly, so the row order is
    irrelevant."""
    t1, t2 = def_types[0].astype(_I32), def_types[1].astype(_I32)
    mul = _MATCHUP_BASE * tables.type_chart[atk_type, t1].astype(_I32) // 10
    e2 = tables.type_chart[atk_type, t2].astype(_I32)
    return jnp.where(t2 != t1, mul * e2 // 10, mul)


def _divide(x, d):
    """``BattleSystem_Divide`` (battle_lib.c:3599) for non-negative ``x``."""
    return jnp.where(x == 0, 0, jnp.maximum(x // d, 1))


def _stage_2_score(tables: Tables, state: BattleState, move, power_mul) -> jnp.ndarray:
    """battle_lib.c:8055-8074: ``CalcMoveDamage`` by the fainted battler (side 1's
    active) against ``D``, then ``ApplyTypeChart`` with the fainted battler as attacker,
    each result stored in the ``u8`` score. A chart immunity leaves 0 through
    ``ApplyTypeMultiplier`` (:7547); the ``MOVE_STATUS_IMMUNE`` check (:8070) only adds the
    refused Levitate and Magnet Rise."""
    score = calc_move_damage(tables, state, 1, move, power_mul) & _U8
    a_types = state.types[1, state.active[1]].astype(_I32)
    d_types = state.types[0, state.active[0]]
    move_type = tables.move_type[move].astype(_I32)
    score = jnp.where(jnp.any(a_types == move_type), score * 15 // 10, score)  # :2592
    first, second = chart_multipliers(tables, move, d_types, state.gravity)  # :2611-2646
    return _divide(_divide(score * first, 10) * second, 10) & _U8


def _post_ko(tables: Tables, state: BattleState, power_mul) -> tuple[jnp.ndarray, jnp.ndarray]:
    slots = jnp.arange(SLOTS)
    candidate = (state.species[1] > 0) & (state.hp[1] > 0) & (slots != state.active[1])
    d_types = state.types[0, state.active[0]]
    moves = state.moves[1].astype(_I32)  # [SLOTS, MOVES]

    # Stage 1 (:7959-8031).
    own_types = state.types[1].astype(_I32)  # [SLOTS, 2]; a mono type counts twice
    score_1 = (
        jax.vmap(lambda t: _matchup(tables, t, d_types))(own_types[:, 0])
        + jax.vmap(lambda t: _matchup(tables, t, d_types))(own_types[:, 1])
    ) & _U8
    flags = jax.vmap(lambda m: calc_effectiveness_flags(tables, m, d_types, state.gravity))(
        moves.reshape(-1)
    ).reshape(SLOTS, MOVES)
    super_effective = jnp.any((moves != 0) & ((flags & SUPER_EFFECTIVE) != 0), axis=1)
    eligible = candidate & super_effective & (score_1 > 0)
    pick_1 = jnp.argmax(jnp.where(eligible, score_1, -1)).astype(_I32)

    # Stage 2 (:8034-8088).
    scored = candidate[:, None] & (moves != 0) & (tables.move_power[moves] != 1)
    score_2 = jax.vmap(lambda m: _stage_2_score(tables, state, m, power_mul))(
        moves.reshape(-1)
    ).reshape(SLOTS, MOVES)
    score_2 = jnp.max(jnp.where(scored, score_2, 0), axis=1)
    pick_2 = jnp.argmax(score_2).astype(_I32)  # first maximum: the lower slot
    first = jnp.argmax(candidate).astype(_I32)

    stage_2 = ~jnp.any(eligible)
    stale = stage_2 & jnp.any(candidate) & ~scored[first, 0]
    pick = jnp.where(
        jnp.any(eligible), pick_1, jnp.where(jnp.max(score_2) > 0, pick_2, first)
    )  # 3: battle_display.c:4548-4556
    return pick, stale


_post_ko_jit = jax.jit(_post_ko)


def post_ko_switch_in(tables: Tables, state: BattleState, power_mul=10) -> jnp.ndarray:
    """``int32``: the party slot the opponent sends in for its fainted active Pokemon
    (module docstring), always a ``legal_replacements(state, 1)`` slot when one exists.
    Raises ``UnsupportedAi`` on a refused concrete state; ``REFUSED_SLOT`` when traced."""
    pick, stale = _post_ko_jit(tables, state, power_mul)
    if not isinstance(stale, jax.core.Tracer) and bool(stale):
        raise UnsupportedAi(
            "replacement: stage 2 of BattleAI_PostKOSwitchIn compares the stale stage-1 "
            "score after the first candidate's empty or power-1 move slot 0 "
            "(battle_lib.c:8076); not implemented"
        )
    return jnp.where(stale, REFUSED_SLOT, pick).astype(_I32)
