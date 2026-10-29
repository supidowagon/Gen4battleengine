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
* ``switch_stop``: the first of the voluntary-switch draws of
  ``TrainerAI_ShouldSwitch`` (``trainer_ai.c:3894``; roark.md 8.5 and 11.1) that
  stops the check, 0..``SWITCH_SITES - 1``, or ``SWITCH_NONE`` (9) when none does. A
  keep draw that stops means "no switch", a party-pass draw that stops means "switch
  to that party member". Site i stops with ``prod_{h<i} (1 - p_h) * p_i``, where p
  depends on the site's ``SwitchSite`` kind, so one categorical is exact.

The site layout depends on the state (which of the AI's moves are super effective,
which bench Pokemon resist ``move_hit``): both functions take ``tables`` and derive it
with ``nuzline.ai.switching.switch_sites``, the one layout function Task 3's
``maybe_switch`` also uses (``switch_action`` maps a stop to its action). Draws that
cannot change the result (``AI_HasAbsorbAbilityInParty``'s ``% 3`` for Roark) are not
sites. A refused layout raises ``UnsupportedSwitching`` on concrete inputs; traced, it
gives ``switch_stop = SWITCH_REFUSED`` and NaN probabilities (``switching``). Item use
draws nothing (8.6).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai.switching import (
    SWITCH_NONE,
    SWITCH_REFUSED,
    SWITCH_SITES,
    SwitchSite,
    switch_sites,
)
from nuzline.data.tables import Tables
from nuzline.engine.state import MOVES, BattleState

__all__ = [
    "AI_SITE_MOVES",
    "MAX_ENUMERATED",
    "MAX_EXPERT_SITES",
    "MAX_SCRIPT_ROLLS",
    "SWITCH_NONE",
    "SWITCH_REFUSED",
    "SWITCH_SITES",
    "AiOutcome",
    "AiOutcomeTooWide",
    "SwitchSite",
    "enumerate_ai_outcomes",
    "sample_ai_outcome",
    "script_roll_count",
    "script_roll_sites",
    "script_roll_thresholds",
    "switch_stop_probs",
]

MAX_SCRIPT_ROLLS = 4  # roark.md 8.3: at most four sites per Roark Pokemon
MAX_ENUMERATED = 4096  # spec §10 "AI outcome width"
SIM_ROLL_MAX = 15  # roll index of 100 %
TIEBREAK_MOD = 12
_I8, _I32, _F32 = jnp.int8, jnp.int32, jnp.float32

# RandNext() % m over a uniform u16 per SwitchSite (NONE, KEEP, PASS_INEFFECTIVE,
# PASS_NOT_VERY_EFFECTIVE, REFUSED): (modulus, stops when the residue is 0) -> stop
# probability, NaN for REFUSED. 65536 = 10 * 6553 + 6 = 3 * 21845 + 1.
_SWITCH_MOD = jnp.array([1, 10, 2, 3, 1], _I32)
_SWITCH_STOPS_ON_ZERO = jnp.array([False, False, True, True, False])
_SWITCH_P = jnp.array([0, 58982, 32768, 21846, jnp.nan], _F32) / 65536

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
_MAX_EXPERT = MAX_EXPERT_SITES = max(len(e) for _, e in _SITES.values())
_SITE_MOVE_IDS = jnp.array(list(_SITES), _I32)
_EVAL_N = jnp.array([n for n, _ in _SITES.values()], _I32)
_EXPERT_N = jnp.array([list(e) + [0] * (_MAX_EXPERT - len(e)) for _, e in _SITES.values()], _I32)

# Values each field takes, in enumeration order (roark.md 14 "AiOutcome width").
_SIM_ROLL_VALUES, _ABILITY_VALUES, _TIEBREAK_VALUES = 1, 2, TIEBREAK_MOD
_SWITCH_VALUES = SWITCH_SITES + 1
_WIDTH = math.prod(
    (_SIM_ROLL_VALUES, _ABILITY_VALUES, 2**MAX_SCRIPT_ROLLS, _SWITCH_VALUES, _TIEBREAK_VALUES)
)
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
    switch_stop: jnp.ndarray  # int8[], first stopping switch draw 0-8, 9 = none, -1 refused


def _site_table(moves) -> tuple[jnp.ndarray, jnp.ndarray]:
    """For the move slots ``moves`` (int[4], decomp ids): the N of every candidate site,
    EVAL_ATTACK slots 0-3 then EXPERT slots 0-3 (slot-major), and its index k in
    ``script_rolls`` (``MAX_SCRIPT_ROLLS`` or more when absent or beyond the last
    entry)."""
    moves = jnp.asarray(moves).astype(_I32)
    match = (moves[:, None] == _SITE_MOVE_IDS[None, :]).astype(_I32)  # [4, n_moves]
    eval_n = match @ _EVAL_N  # [4]
    expert_n = (match @ _EXPERT_N).reshape(MOVES * _MAX_EXPERT)  # slot-major
    n = jnp.concatenate([eval_n, expert_n])
    used = n > 0
    position = jnp.cumsum(used.astype(_I32)) - 1
    return n, jnp.where(used, position, MAX_SCRIPT_ROLLS)


def script_roll_count(moves) -> jnp.ndarray:
    """``int32``: the number of ``IfRandomLessThan`` sites of a moveset (int[4]); more
    than ``MAX_SCRIPT_ROLLS`` cannot be encoded (the AI refuses it)."""
    n, _ = _site_table(moves)
    return jnp.sum((n > 0).astype(_I32))


def script_roll_sites(state: BattleState) -> jnp.ndarray:
    """``int32[MOVES, 1 + MAX_EXPERT_SITES]``: for each move slot of the opponent's
    active Pokemon, the ``script_rolls`` index k of its EVAL_ATTACK site (column 0) and
    of its EXPERT sites in script order (columns 1..), ``MAX_SCRIPT_ROLLS`` or more where
    the move has no such site (roark.md 8.3)."""
    _, k = _site_table(state.moves[1, state.active[1]])
    eval_k = k[:MOVES].reshape(MOVES, 1)
    expert_k = k[MOVES:].reshape(MOVES, _MAX_EXPERT)
    return jnp.concatenate([eval_k, expert_k], axis=1)


def script_roll_thresholds(state: BattleState) -> jnp.ndarray:
    """``int32[MAX_SCRIPT_ROLLS]``: the N of each ``IfRandomLessThan`` site of the
    opponent's active Pokemon, in script order (EVAL_ATTACK slots 0-3, then EXPERT
    slots 0-3; roark.md 8.3). 0 marks an unused entry, which never rolls 1."""
    n, target = _site_table(state.moves[1, state.active[1]])
    return jnp.zeros(MAX_SCRIPT_ROLLS, _I32).at[target].set(n, mode="drop")


def switch_stop_probs(switch_sites) -> jnp.ndarray:
    """``float32[SWITCH_SITES + 1]``: the probability that site i is the first to stop,
    ``prod_{h<i} (1 - p_h) * p_i``, then (last entry) that none stops; all NaN for a
    refused layout."""
    kind = jnp.asarray(switch_sites, _I32)
    p = _SWITCH_P[kind]
    reach = jnp.cumprod(jnp.concatenate([jnp.ones(1, _F32), 1 - p]))  # [SWITCH_SITES + 1]
    probs = jnp.concatenate([reach[:-1] * p, reach[-1:]])
    return jnp.where(jnp.any(kind == SwitchSite.REFUSED), jnp.nan, probs)


def _first_stop(draws, switch_sites) -> jnp.ndarray:
    """The first site whose u16 ``draws[i]`` stops the check, else ``SWITCH_NONE``;
    ``SWITCH_REFUSED`` for a refused layout."""
    kind = jnp.asarray(switch_sites, _I32)
    zero = draws % _SWITCH_MOD[kind] == 0
    stops = (kind != SwitchSite.NONE) & (zero == _SWITCH_STOPS_ON_ZERO[kind])
    first = jnp.where(jnp.any(stops), jnp.argmax(stops), SWITCH_NONE)
    return jnp.where(jnp.any(kind == SwitchSite.REFUSED), SWITCH_REFUSED, first)


def sample_ai_outcome(key, tables: Tables, state: BattleState) -> AiOutcome:
    """Draw every field from its decomp probability for this state (the switch draws
    over ``switch_sites(tables, state)``)."""
    layout = switch_sites(tables, state)
    k = jax.random.split(key, 4)
    rolls = jax.random.randint(k[1], (MAX_SCRIPT_ROLLS,), 0, 256, dtype=_I32)
    draws = jax.random.randint(k[3], (SWITCH_SITES,), 0, 65536, dtype=_I32)
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, _I8),
        ability_guess=jax.random.randint(k[0], (), 0, 2, dtype=_I32).astype(_I8),
        script_rolls=(rolls < script_roll_thresholds(state)).astype(_I8),
        tiebreak=(jax.random.randint(k[2], (), 0, 65536, dtype=_I32) % TIEBREAK_MOD).astype(_I8),
        switch_stop=_first_stop(draws, layout).astype(_I8),
    )


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every (ability_guess, script_rolls, switch_stop, tiebreak) combination,
    tiebreak fastest."""
    ability, bits, switch, tiebreak = np.meshgrid(
        np.arange(_ABILITY_VALUES),
        np.arange(2**MAX_SCRIPT_ROLLS),
        np.arange(_SWITCH_VALUES),
        np.arange(_TIEBREAK_VALUES),
        indexing="ij",
    )
    rolls = (bits.reshape(-1, 1) >> np.arange(MAX_SCRIPT_ROLLS)) & 1
    return ability.reshape(-1), rolls, switch.reshape(-1), tiebreak.reshape(-1)


def enumerate_ai_outcomes(tables: Tables, state: BattleState) -> tuple[AiOutcome, jnp.ndarray]:
    """Every outcome combination (a batched ``AiOutcome`` with leading axis K) and its
    exact probability ``float32[K]``. K is fixed (3840); combinations with an unused
    script-roll site set to 1, or a ``switch_stop`` at a ``SwitchSite.NONE`` entry, have
    probability 0. Raises ``AiOutcomeTooWide`` when the product of the ranges exceeds
    ``MAX_ENUMERATED``. The switch draws are over ``switch_sites(tables, state)``."""
    if _WIDTH > MAX_ENUMERATED:
        raise AiOutcomeTooWide(f"{_WIDTH} AI outcome combinations > {MAX_ENUMERATED}")
    ability, rolls, switch, tiebreak = _grid()
    p_jump = script_roll_thresholds(state).astype(_F32) / 256  # [MAX_SCRIPT_ROLLS]
    rolls_j = jnp.asarray(rolls, _I32)
    p_rolls = jnp.prod(jnp.where(rolls_j == 1, p_jump, 1 - p_jump), axis=1)
    p_switch = switch_stop_probs(switch_sites(tables, state))[jnp.asarray(switch, _I32)]
    p_rest = jnp.asarray(_TIEBREAK_P[tiebreak] / _ABILITY_VALUES, _F32)
    outs = AiOutcome(
        sim_roll=jnp.full((_WIDTH, MOVES), SIM_ROLL_MAX, _I8),
        ability_guess=jnp.asarray(ability, _I8),
        script_rolls=jnp.asarray(rolls, _I8),
        tiebreak=jnp.asarray(tiebreak, _I8),
        switch_stop=jnp.asarray(switch, _I8),
    )
    return outs, p_rolls * p_switch * p_rest
