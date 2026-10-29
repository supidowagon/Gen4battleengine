"""The trainer AI's random draws as explicit outcomes (spec §4.2 Trainer AI).

``ai_policy`` draws no randomness: every ``RandNext`` the decomp's AI makes during
one FIGHT decision is a field of ``AiOutcome``. ``sample_ai_outcome`` draws it
(Monte Carlo) and ``enumerate_ai_outcomes`` lists every combination with its exact
probability. Ranges and encodings are those of ``docs/ai/roark.md`` section 8:

* ``sim_roll[i]``: the simulated damage roll of opponent move slot i
  (``moveDamageRolls``, ``trainer_ai.c:240``), as an engine roll index 0-15
  (85-100 %). Roark's scripts only pass ``USE_MAX_DAMAGE``, so it is fixed at 15
  (100 %) and never enumerated (8.1).
* ``ability_guess``: 0 = the player's ability slot 0, 1 = slot 1, each 1/2
  (``LoadBattlerAbility DEFENDER``, ``trainer_ai.c:1170``; 8.2). One scalar is exact
  for Roark; it is ignored when no guess is made.
* ``script_rolls[k]``: 1 when the k-th ``IfRandomLessThan N`` site in script order
  jumps (``RandNext % 256 < N``), probability N/256 (8.3). The site layout follows
  the opponent active's moveset: ``script_roll_thresholds``.
* ``tiebreak``: ``RandNext % 12`` (0-11), so ``tiebreak % n`` is exactly the decomp's
  ``RandNext % n`` for n = 1..4 tied slots (``trainer_ai.c:338``; 8.4).

The switching draws (roark.md 11.1) are not fields yet. Item use draws nothing (8.6).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.engine.state import MOVES, BattleState

MAX_SCRIPT_ROLLS = 4  # roark.md 8.3: at most four sites per Roark Pokemon
MAX_ENUMERATED = 4096  # spec §10 "AI outcome width"
SIM_ROLL_MAX = 15  # roll index of 100 %
TIEBREAK_MOD = 12
_I8, _I32, _F32 = jnp.int8, jnp.int32, jnp.float32

# IfRandomLessThan sites per move (decomp move id), transcribed from roark.md 5.2-5.3
# and 8.3: the N of the move's EVAL_ATTACK site (EvalAttack_TryScorePlus2,
# script.s:6383; 0 = none, a Normal-type move is never QUADRUPLE), then the Ns of its
# EXPERT sites in script order. Only these moves' script paths are transcribed; the
# coverage registry must refuse an opponent move outside this table.
_SITES: dict[int, tuple[int, tuple[int, ...]]] = {
    446: (80, (128,)),  # Stealth Rock: Expert_StealthRock, script.s:6274
    88: (80, ()),  # Rock Throw
    228: (80, (128, 128)),  # Pursuit: Expert_Pursuit, script.s:3726 and :3731
    103: (0, (50,)),  # Screech: Expert_StatusDefenseDown, script.s:2328
    43: (0, (50,)),  # Leer: Expert_StatusDefenseDown, script.s:2328
    29: (0, ()),  # Headbutt
    33: (0, ()),  # Tackle (the test fixture's slot 0)
}
AI_SITE_MOVES = frozenset(_SITES)
_MAX_EXPERT = max(len(e) for _, e in _SITES.values())
_SITE_MOVE_IDS = jnp.array(list(_SITES), _I32)
_EVAL_N = jnp.array([n for n, _ in _SITES.values()], _I32)
_EXPERT_N = jnp.array([list(e) + [0] * (_MAX_EXPERT - len(e)) for _, e in _SITES.values()], _I32)

# Values each field takes, in enumeration order (roark.md 14 "AiOutcome width").
_SIM_ROLL_VALUES, _ABILITY_VALUES, _TIEBREAK_VALUES = 1, 2, TIEBREAK_MOD
_WIDTH = math.prod((_SIM_ROLL_VALUES, _ABILITY_VALUES, 2**MAX_SCRIPT_ROLLS, _TIEBREAK_VALUES))
# RandNext % 12 over a uniform u16: 65536 = 12 * 5461 + 4.
_TIEBREAK_P = np.array([5462] * 4 + [5461] * 8, np.float64) / 65536


class AiOutcomeTooWide(ValueError):
    """The product of the AI outcome ranges exceeds ``MAX_ENUMERATED``; the fight is
    sampling-only (spec §10)."""


class AiOutcome(NamedTuple):
    sim_roll: jnp.ndarray  # int8[4], roll index 0-15 per opponent move slot
    ability_guess: jnp.ndarray  # int8[], player ability slot 0/1
    script_rolls: jnp.ndarray  # int8[MAX_SCRIPT_ROLLS], 1 = the site's jump is taken
    tiebreak: jnp.ndarray  # int8[], RandNext % 12


def script_roll_thresholds(state: BattleState) -> jnp.ndarray:
    """``int32[MAX_SCRIPT_ROLLS]``: the N of each ``IfRandomLessThan`` site of the
    opponent's active Pokemon, in script order (EVAL_ATTACK slots 0-3, then EXPERT
    slots 0-3; roark.md 8.3). 0 marks an unused entry, which never rolls 1."""
    moves = state.moves[1, state.active[1]].astype(_I32)
    match = (moves[:, None] == _SITE_MOVE_IDS[None, :]).astype(_I32)  # [4, n_moves]
    eval_n = match @ _EVAL_N  # [4]
    expert_n = (match @ _EXPERT_N).reshape(MOVES * _MAX_EXPERT)  # slot-major
    n = jnp.concatenate([eval_n, expert_n])
    used = n > 0
    position = jnp.cumsum(used.astype(_I32)) - 1
    target = jnp.where(used, position, MAX_SCRIPT_ROLLS)  # out of range: dropped
    return jnp.zeros(MAX_SCRIPT_ROLLS, _I32).at[target].set(n, mode="drop")


def sample_ai_outcome(key, state: BattleState) -> AiOutcome:
    """Draw every field from its decomp probability for this state."""
    k = jax.random.split(key, 3)
    rolls = jax.random.randint(k[1], (MAX_SCRIPT_ROLLS,), 0, 256, dtype=_I32)
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, _I8),
        ability_guess=jax.random.randint(k[0], (), 0, 2, dtype=_I32).astype(_I8),
        script_rolls=(rolls < script_roll_thresholds(state)).astype(_I8),
        tiebreak=(jax.random.randint(k[2], (), 0, 65536, dtype=_I32) % TIEBREAK_MOD).astype(_I8),
    )


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (ability_guess, script_rolls, tiebreak) combination, tiebreak fastest."""
    ability, bits, tiebreak = np.meshgrid(
        np.arange(_ABILITY_VALUES),
        np.arange(2**MAX_SCRIPT_ROLLS),
        np.arange(_TIEBREAK_VALUES),
        indexing="ij",
    )
    rolls = (bits.reshape(-1, 1) >> np.arange(MAX_SCRIPT_ROLLS)) & 1
    return ability.reshape(-1), rolls, tiebreak.reshape(-1)


def enumerate_ai_outcomes(state: BattleState) -> tuple[AiOutcome, jnp.ndarray]:
    """Every outcome combination (a batched ``AiOutcome`` with leading axis K) and its
    exact probability ``float32[K]``. K is fixed (384); combinations with an unused
    script-roll site set to 1 have probability 0. Raises ``AiOutcomeTooWide`` when
    the product of the ranges exceeds ``MAX_ENUMERATED``."""
    if _WIDTH > MAX_ENUMERATED:
        raise AiOutcomeTooWide(f"{_WIDTH} AI outcome combinations > {MAX_ENUMERATED}")
    ability, rolls, tiebreak = _grid()
    p_jump = script_roll_thresholds(state).astype(_F32) / 256  # [MAX_SCRIPT_ROLLS]
    rolls_j = jnp.asarray(rolls, _I32)
    p_rolls = jnp.prod(jnp.where(rolls_j == 1, p_jump, 1 - p_jump), axis=1)
    p_rest = jnp.asarray(_TIEBREAK_P[tiebreak] / _ABILITY_VALUES, _F32)
    outs = AiOutcome(
        sim_roll=jnp.full((_WIDTH, MOVES), SIM_ROLL_MAX, _I8),
        ability_guess=jnp.asarray(ability, _I8),
        script_rolls=jnp.asarray(rolls, _I8),
        tiebreak=jnp.asarray(tiebreak, _I8),
    )
    return outs, p_rolls * p_rest
