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

The enumeration width is a constant of this field layout, not of the state: every
field's range is fixed, so ``ENUMERATION_WIDTH`` (3840, roark.md 14) is the same for
every fight and is checked once, at import, against the spec's ``MAX_ENUMERATED``
budget. Roark never needs the spec's sampling-only fallback (spec section 4.2, "an AI
whose enumeration is too wide"): a trainer whose draws do not fit this layout needs a
new layout, and the sampling-only grade is unimplemented end to end (see the
``nuzline.engine.coverage`` docstring).

**Per-slot layouts** (``AiLayout.per_slot_sites``, Kaizo; docs/ai/kaizo_roark.md 8). The
flat layout above is the vanilla one and does not change. A per-slot layout keys the
script rolls by move slot: ``script_rolls`` is ``int8[MOVES, layout.slot_sites]``, and
column k of slot m is the k-th ``IfRandomLessThan`` site of the slot's move in script
order (``slot_site_table``: the battle effect of the move picks its row of
``layout.effect_sites``). Every slot reads only its own columns (``TrainerAI_EvalMoves``
scores slot by slot, trainer_ai.c:483-525), which is what lets ``policy`` factor the
exact distribution per slot. The other fields are as above. A per-slot layout's
enumeration is not a constant: ``enumerate_ai_outcomes`` lists the used sites only, on
concrete inputs (it is a test oracle; ``policy`` never enumerates the flat product).

Sites are named by their ``script.s`` line (``SITE_N``): every ``IfRandomLessThan`` the
transcription reads passes its line to ``commands.if_random_less_than``, which finds the
draw's index in ``AiContext.sites`` (one column per ``SITE_LINES`` entry): the flat
layout's index for vanilla (``line_sites``), the slot's column for a per-slot layout
(``slot_roll_sites``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

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

if TYPE_CHECKING:
    from nuzline.ai.layout import AiLayout

__all__ = [
    "ABILITY_GUESSES",
    "AI_SITE_MOVES",
    "EFFECT_SITE_LINES",
    "ENUMERATION_WIDTH",
    "KAIZO_SLOT_SITES",
    "MAX_ENUMERATED",
    "MAX_EXPERT_SITES",
    "MAX_JOINT_SITES",
    "MAX_SCRIPT_ROLLS",
    "SITE_LINES",
    "SITE_N",
    "SWITCH_NONE",
    "SWITCH_REFUSED",
    "SWITCH_SITES",
    "TIEBREAK_MOD",
    "TIEBREAK_P",
    "AiOutcome",
    "SwitchSite",
    "enumerate_ai_outcomes",
    "line_sites",
    "sample_ai_outcome",
    "script_roll_count",
    "script_roll_probs",
    "script_roll_sites",
    "script_roll_thresholds",
    "slot_roll_sites",
    "slot_roll_thresholds",
    "slot_site_counts",
    "slot_site_table",
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

# Every IfRandomLessThan the transcription reads: its script.s line and its N (the jump is
# taken when RandNext() % 256 < N, trainer_ai.c:528-538). AiContext.sites has one column
# per line, in this order (commands.if_random_less_than).
SITE_N: dict[int, int] = {
    6383: 80,  # EvalAttack_TryScorePlus2 (+2 when it does NOT jump)
    2328: 50,  # Expert_StatusDefenseDown_TryScoreMinus2
    3726: 128,  # Expert_Pursuit_TryScorePlus1
    3731: 128,  # Expert_Pursuit_CheckUturn
    6274: 128,  # Expert_StealthRock
}
SITE_LINES = tuple(SITE_N)
# The flat (vanilla) layout's column of script_roll_sites that each line reads: column 0 is
# the EVAL_ATTACK site, columns 1.. the move's EXPERT sites in script order (_SITES).
_VANILLA_COLUMN = {6383: 0, 2328: 1, 3726: 1, 3731: 2, 6274: 1}
assert set(_VANILLA_COLUMN) == set(SITE_N)

# The per-slot (Kaizo) site table, keyed by decomp battle effect (BATTLE_EFFECT_*, line - 1
# of generated/move_battle_effects.txt): the script.s lines of the sites a move with that
# effect can reach, in script order (flag order BASIC, EVAL_ATTACK, EXPERT, trainer_ai.c
# :297-310, then line order). EVAL_ATTACK's quad roll (:6383) is reached by every move whose
# IfMoveEffectivenessEquals is QUADRUPLE (:6378), whatever its effect, so every effect
# has it. K3c Tasks 5 and 6 add Roark's effects; layout.KAIZO's transcribed set is this
# table's keys.
EFFECT_SITE_LINES: dict[int, tuple[int, ...]] = {
    0: (6383,),  # HIT: no Expert_Main entry (script.s:1623-1811)
    19: (6383, 2328),  # DEF_DOWN: Expert_StatusDefenseDown (:1642 -> :2318)
    31: (6383,),  # FLINCH_HIT: no Expert_Main entry
    59: (6383, 2328),  # DEF_DOWN_2: Expert_StatusDefenseDown (:1671 -> :2318)
    128: (6383, 3726, 3731),  # HIT_BEFORE_SWITCH: Expert_Pursuit (:1706 -> :3706)
    266: (6383, 6274),  # STEALTH_ROCK: Expert_StealthRock (:1805 -> :6268)
}
# The most sites one move slot can reach in the Kaizo transcription, sized for Selfdestruct
# (HALVE_DEFENSE, K3c Task 6): EvalAttack :6374 and :6383, Expert_Explosion :1859, :1865,
# :1870, :1875 and :1880, Risky :6530.
KAIZO_SLOT_SITES = 8

# Values each field takes, in enumeration order (roark.md 14 "AiOutcome width").
# ABILITY_GUESSES: AiOutcome.ability_guess is 0 or 1, 1/2 each.
_SIM_ROLL_VALUES, ABILITY_GUESSES, _TIEBREAK_VALUES = 1, 2, TIEBREAK_MOD
_SWITCH_VALUES = SWITCH_SITES + 1
ENUMERATION_WIDTH = math.prod(
    (_SIM_ROLL_VALUES, ABILITY_GUESSES, 2**MAX_SCRIPT_ROLLS, _SWITCH_VALUES, _TIEBREAK_VALUES)
)
# The layout is fixed, so this is a design-time invariant: widening a field range past
# the budget is a change to this module, not a property of some fight.
if ENUMERATION_WIDTH > MAX_ENUMERATED:
    raise RuntimeError(f"{ENUMERATION_WIDTH} AI outcome combinations > {MAX_ENUMERATED}")
# P(tiebreak = t), t = 0..11: RandNext % 12 over a uniform u16, 65536 = 12 * 5461 + 4.
TIEBREAK_P = np.array([5462] * 4 + [5461] * 8, np.float64) / 65536
# A per-slot layout's joint enumeration (enumerate_ai_outcomes, a test oracle) lists at most
# 2**MAX_JOINT_SITES script-roll patterns.
MAX_JOINT_SITES = 10


class AiOutcome(NamedTuple):
    sim_roll: jnp.ndarray  # int8[4], roll index 0-15 per opponent move slot
    ability_guess: jnp.ndarray  # int8[], player ability slot 0/1
    # int8[MAX_SCRIPT_ROLLS] (flat layout) or int8[MOVES, slot_sites] (per-slot layout):
    # 1 = the site's jump is taken
    script_rolls: jnp.ndarray
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


def line_sites(state: BattleState) -> jnp.ndarray:
    """``int32[MOVES, len(SITE_LINES)]``: ``AiContext.sites`` of the flat layout: for each
    move slot of the opponent's active Pokemon and each ``SITE_LINES`` line, the
    ``script_rolls`` index that ``script_roll_sites`` gives the column the line reads
    (``_VANILLA_COLUMN``)."""
    k = script_roll_sites(state)
    return k[:, jnp.array([_VANILLA_COLUMN[line] for line in SITE_LINES])]


def _effect_rows(layout: AiLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``layout.effect_sites`` as arrays: the effects ``int32[E]``, and per effect the N and
    the line of each column, ``int32[E, slot_sites]`` (0 where the effect has no site)."""
    effects = np.array([e for e, _ in layout.effect_sites] or [-1], np.int32)
    n = np.zeros((len(effects), layout.slot_sites), np.int32)
    lines = np.zeros((len(effects), layout.slot_sites), np.int32)
    for i, (_, sites) in enumerate(layout.effect_sites):
        for k, (line, threshold) in enumerate(sites):
            n[i, k], lines[i, k] = threshold, line
    return effects, n, lines


def slot_site_table(tables: Tables, layout: AiLayout, move) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(n, lines)``, both ``int32[layout.slot_sites]``: the N and the ``script.s`` line of
    each script-roll site of ``move`` (decomp id) in a per-slot layout, the row of
    ``layout.effect_sites`` of the move's battle effect (``tables.move_ai_effect``, as
    ``IfCurrentMoveEffectEqualTo`` reads it, trainer_ai.c:1441); zeros past its last site,
    for ``MOVE_NONE`` and for an effect without a row (an untranscribed move, refused)."""
    move = jnp.asarray(move, _I32)
    effects, n, lines = _effect_rows(layout)
    effect = tables.move_ai_effect[move].astype(_I32)
    match = ((effect == effects) & (move != 0)).astype(_I32)  # [E]
    return match @ jnp.asarray(n), match @ jnp.asarray(lines)


def _slot_tables(tables: Tables, state: BattleState, layout: AiLayout):
    moves = state.moves[1, state.active[1]].astype(_I32)
    return jax.vmap(lambda m: slot_site_table(tables, layout, m))(moves)  # 2 x [MOVES, S]


def slot_roll_thresholds(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``int32[MOVES, layout.slot_sites]``: the N of each site of each move slot of the
    opponent's active Pokemon (``slot_site_table``); 0 marks an unused column, which never
    rolls 1."""
    return _slot_tables(tables, state, layout)[0]


def slot_site_counts(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``int32[MOVES]``: the number of script-roll sites of each move slot of the opponent's
    active Pokemon in a per-slot layout."""
    return jnp.sum((slot_roll_thresholds(tables, state, layout) > 0).astype(_I32), axis=1)


def slot_roll_sites(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``int32[MOVES, len(SITE_LINES)]``: ``AiContext.sites`` of a per-slot layout: for slot
    m and line L, the index ``m * slot_sites + k`` into the flattened ``script_rolls`` when L
    is the slot's k-th site, else ``MOVES * slot_sites`` (out of range: never rolls 1)."""
    n, lines = _slot_tables(tables, state, layout)
    width = layout.slot_sites
    columns = []
    for line in SITE_LINES:
        here = (lines == line) & (n > 0)  # [MOVES, S]
        flat = jnp.arange(MOVES) * width + jnp.argmax(here, axis=1)
        columns.append(jnp.where(jnp.any(here, axis=1), flat, MOVES * width))
    return jnp.stack(columns, axis=1).astype(_I32)


def _roll_weights(state: BattleState, rolls) -> jnp.ndarray:
    """``float32[R]``: the exact probability of each ``script_rolls`` row of ``rolls``
    (int[R, MAX_SCRIPT_ROLLS], 1 = jump) for this state, ``prod_k`` of N_k/256 where
    bit k is 1, else 1 - N_k/256 (``script_roll_thresholds``; roark.md 8.3). The one
    copy of this rule, shared by ``script_roll_probs`` and ``enumerate_ai_outcomes``."""
    p_jump = script_roll_thresholds(state).astype(_F32) / 256  # [MAX_SCRIPT_ROLLS]
    return jnp.prod(jnp.where(jnp.asarray(rolls, _I32) == 1, p_jump, 1 - p_jump), axis=1)


def script_roll_probs(state: BattleState) -> tuple[np.ndarray, jnp.ndarray]:
    """Every ``script_rolls`` bit pattern, ``int8[2**MAX_SCRIPT_ROLLS, MAX_SCRIPT_ROLLS]``
    (pattern b has bit k = ``(b >> k) & 1``), and its exact probability ``float32[...]``
    for this state: ``prod_k`` of N_k/256 where bit k is 1, else 1 - N_k/256
    (``script_roll_thresholds``), the weights ``enumerate_ai_outcomes`` gives them."""
    rolls = (np.arange(2**MAX_SCRIPT_ROLLS)[:, None] >> np.arange(MAX_SCRIPT_ROLLS)) & 1
    return rolls.astype(np.int8), _roll_weights(state, rolls)


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


def sample_ai_outcome(
    key, tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> AiOutcome:
    """Draw every field from its decomp probability for this state (the switch draws
    over ``switch_sites(tables, state)``). ``layout``: a per-slot layout draws
    ``script_rolls`` as ``int8[MOVES, layout.slot_sites]`` (``slot_roll_thresholds``); the
    default, and any flat layout, is the vanilla layout."""
    layout_sites = switch_sites(tables, state)
    k = jax.random.split(key, 4)
    if layout is not None and layout.per_slot_sites:
        shape, thresholds = (MOVES, layout.slot_sites), slot_roll_thresholds(tables, state, layout)
    else:
        shape, thresholds = (MAX_SCRIPT_ROLLS,), script_roll_thresholds(state)
    rolls = jax.random.randint(k[1], shape, 0, 256, dtype=_I32)
    draws = jax.random.randint(k[3], (SWITCH_SITES,), 0, 65536, dtype=_I32)
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, _I8),
        ability_guess=jax.random.randint(k[0], (), 0, 2, dtype=_I32).astype(_I8),
        script_rolls=(rolls < thresholds).astype(_I8),
        tiebreak=(jax.random.randint(k[2], (), 0, 65536, dtype=_I32) % TIEBREAK_MOD).astype(_I8),
        switch_stop=_first_stop(draws, layout_sites).astype(_I8),
    )


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every (ability_guess, script_rolls, switch_stop, tiebreak) combination,
    tiebreak fastest."""
    ability, bits, switch, tiebreak = np.meshgrid(
        np.arange(ABILITY_GUESSES),
        np.arange(2**MAX_SCRIPT_ROLLS),
        np.arange(_SWITCH_VALUES),
        np.arange(_TIEBREAK_VALUES),
        indexing="ij",
    )
    rolls = (bits.reshape(-1, 1) >> np.arange(MAX_SCRIPT_ROLLS)) & 1
    return ability.reshape(-1), rolls, switch.reshape(-1), tiebreak.reshape(-1)


def enumerate_ai_outcomes(
    tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> tuple[AiOutcome, jnp.ndarray]:
    """Every outcome combination (a batched ``AiOutcome`` with leading axis K) and its
    exact probability ``float32[K]``. Flat layout (the default): K is
    ``ENUMERATION_WIDTH`` (3840) for every state; combinations with an unused script-roll
    site set to 1, or a ``switch_stop`` at a ``SwitchSite.NONE`` entry, have probability
    0. The switch draws are over ``switch_sites(tables, state)``.

    A per-slot ``layout`` (a test oracle, concrete inputs only): every combination of the
    used sites of the four slots, ``K = ABILITY_GUESSES * 2**sum(slot_site_counts) *
    (SWITCH_SITES + 1) * TIEBREAK_MOD``; ``ValueError`` when the sites number more than
    ``MAX_JOINT_SITES``."""
    if layout is None or not layout.per_slot_sites:
        return _enumerate_flat(tables, state)
    thresholds = np.asarray(slot_roll_thresholds(tables, state, layout))
    used = np.argwhere(thresholds > 0)  # [U, 2] (slot, column), script order per slot
    if len(used) > MAX_JOINT_SITES:
        raise ValueError(
            f"{len(used)} script-roll sites: more than the {MAX_JOINT_SITES} a joint "
            "enumeration lists"
        )
    ability, bits, switch, tiebreak = (
        a.reshape(-1)
        for a in np.meshgrid(
            np.arange(ABILITY_GUESSES),
            np.arange(2 ** len(used)),
            np.arange(_SWITCH_VALUES),
            np.arange(_TIEBREAK_VALUES),
            indexing="ij",
        )
    )
    on = (bits[:, None] >> np.arange(len(used))) & 1  # [K, U]
    rolls = np.zeros((len(bits), MOVES, layout.slot_sites), np.int8)
    rolls[:, used[:, 0], used[:, 1]] = on
    p_jump = thresholds[used[:, 0], used[:, 1]].astype(np.float64) / 256
    p_rolls = np.prod(np.where(on == 1, p_jump, 1 - p_jump), axis=1)
    p_switch = np.asarray(switch_stop_probs(switch_sites(tables, state)), np.float64)
    probs = p_rolls * p_switch[switch] * TIEBREAK_P[tiebreak] / ABILITY_GUESSES
    outs = AiOutcome(
        sim_roll=jnp.full((len(bits), MOVES), SIM_ROLL_MAX, _I8),
        ability_guess=jnp.asarray(ability, _I8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.asarray(tiebreak, _I8),
        switch_stop=jnp.asarray(switch, _I8),
    )
    return outs, jnp.asarray(probs, _F32)


def _enumerate_flat(tables: Tables, state: BattleState) -> tuple[AiOutcome, jnp.ndarray]:
    ability, rolls, switch, tiebreak = _grid()
    p_rolls = _roll_weights(state, rolls)
    p_switch = switch_stop_probs(switch_sites(tables, state))[jnp.asarray(switch, _I32)]
    p_rest = jnp.asarray(TIEBREAK_P[tiebreak] / ABILITY_GUESSES, _F32)
    outs = AiOutcome(
        sim_roll=jnp.full((ENUMERATION_WIDTH, MOVES), SIM_ROLL_MAX, _I8),
        ability_guess=jnp.asarray(ability, _I8),
        script_rolls=jnp.asarray(rolls, _I8),
        tiebreak=jnp.asarray(tiebreak, _I8),
        switch_stop=jnp.asarray(switch, _I8),
    )
    return outs, p_rolls * p_switch * p_rest
