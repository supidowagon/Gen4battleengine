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
  (``LoadBattlerAbility DEFENDER``, ``trainer_ai.c:1170``; 8.2). The flat layout has one
  scalar for the decision, exact for vanilla Roark (roark.md 8.2); a per-slot layout has
  one bit per guessing read (below). It is ignored when no guess is made.
* ``script_rolls[k]``: 1 when the k-th ``IfRandomLessThan N`` site in script order
  jumps (``RandNext % 256 < N``), probability N/256 (8.3). The site layout follows
  the opponent active's moveset: ``script_roll_thresholds``.
* ``tiebreak``: ``RandNext % 12`` (0-11), so ``tiebreak % n`` is exactly the decomp's
  ``RandNext % n`` for n = 1..4 tied slots (``trainer_ai.c:338``; 8.4).
* ``switch_stop``: the first of the voluntary-switch draws of
  ``TrainerAI_ShouldSwitch`` (``trainer_ai.c:3894``; roark.md 8.5 and 11.1) that
  stops the check, 0..``sites - 1``, or ``sites`` when none does, where ``sites`` is the
  layout's ``switch_sites``: ``SWITCH_SITES`` (9, ``SWITCH_NONE`` = 9 none) for the flat
  layout, 44 for Kaizo's party of six (docs/ai/kaizo_roark.md 11). A
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
exact distribution per slot.

The guess is per read in a per-slot layout too (E1 Ruling R1; docs/ai/kaizo_roark.md 8.4).
``AICmd_LoadBattlerAbility`` (trainer_ai.c:1170-1209) draws a fresh coin,
``BattleSystem_RandNext() & 1`` (:1194-1198), at every ``LoadBattlerAbility DEFENDER`` read
of a decision when the AI does not know the player's ability, it is not Shadow Tag, Magnet
Pull or Arena Trap, and the species has two nonzero abilities. So ``ability_guess`` is
``int8[MOVES, layout.slot_guesses]``: the k-th guessing read on slot m's scoring path, in
path order (flag order, then line order), takes column k of slot m. Each column is a fair
coin, independent of every other, as each read is its own ``RandNext``; no read is shared
between slots, since ``TrainerAI_EvalMoves`` scores slot by slot (:483-525). A path that
reaches more guessing reads than ``slot_guesses`` is refused (``AiContext.missing_guess``),
as a missing site column is. The other fields are as above. A per-slot layout's
enumeration is not a constant: ``enumerate_ai_outcomes`` lists the used sites only, on
concrete inputs (it is a test oracle; ``policy`` never enumerates the flat product).
Its guess columns are the ones the caller says the paths reach (``guess_columns``).

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
    "KAIZO_EXPERT_LABELS",
    "KAIZO_EXPERT_TRANSCRIBED",
    "KAIZO_SLOT_GUESSES",
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
    6374: 51,  # EvalAttack_MaybeDeprioritize (-2 when it does NOT jump)
    # K3c Task 6: the Expert labels of Kaizo Roark's effects and Risky_Main.
    1835: 50,  # Expert_DrainMove_TryScoreMinus3
    1859: 128,  # Expert_Explosion (evasion +4 or more)
    1865: 50,  # Expert_Explosion_CheckUserHighHP
    1870: 128,  # Expert_Explosion_CheckUserMediumHP
    1875: 50,  # Expert_Explosion_CheckUserLowHP
    1880: 50,  # Expert_Explosion_TryScoreMinus1
    2075: 70,  # Expert_StatusSpeedUp_TryScorePlus3
    2665: 30,  # Expert_Recovery_CheckHP
    2671: 100,  # Expert_Recovery_CheckForSnatch (the player seen with a Snatch effect)
    2674: 20,  # Expert_Recovery_TryScorePlus2
    2690: 50,  # Expert_ToxicLeechSeed
    2695: 50,  # Expert_ToxicLeechSeed_CheckTargetHP
    2704: 60,  # Expert_ToxicLeechSeed_TryScorePlus2
    2718: 128,  # Expert_LightScreen
    2724: 64,  # Expert_LightScreen_CheckLastUsedMove
    3477: 32,  # Expert_Curse_HighChanceScorePlus1
    3481: 128,  # Expert_Curse_FlipCoinScorePlus1
    3486: 128,  # Expert_Curse_CheckDefenseStage
    3491: 128,  # Expert_Curse_CheckDefenseStageAnyBoosts
    6530: 128,  # Risky_Main (+2 when it does NOT jump)
}
SITE_LINES = tuple(SITE_N)
# The flat (vanilla) layout's column of script_roll_sites that each line reads: column 0 is
# the EVAL_ATTACK site, columns 1.. the move's EXPERT sites in script order (_SITES). None:
# a line no vanilla-transcribed move reaches (:6374 is HALVE_DEFENSE's only, script.s:6365;
# the K3c Task 6 lines are labels of effects no vanilla-transcribed move has, and RISKY is
# not a vanilla flag); it reads an unused index, which never rolls 1.
_VANILLA_COLUMN = {6383: 0, 2328: 1, 3726: 1, 3731: 2, 6274: 1} | {
    line: None for line in SITE_N if line not in (6383, 2328, 3726, 3731, 6274)
}
assert set(_VANILLA_COLUMN) == set(SITE_N)

# The per-slot (Kaizo) site table, keyed by decomp battle effect (BATTLE_EFFECT_*, line - 1
# of generated/move_battle_effects.txt): the script.s lines of the sites a move with that
# effect can reach, in script order (flag order BASIC, EVAL_ATTACK, EXPERT, trainer_ai.c
# :297-310, then line order). EVAL_ATTACK's quad roll (:6383) is reached by every move whose
# IfMoveEffectivenessEquals is QUADRUPLE (:6378), whatever its effect, so every effect
# has it. No BASIC label of a transcribed effect reads IfRandomLessThan. layout.KAIZO's
# transcribed set is this table's keys.
#
# K3c Task 5 added the effects of Kaizo Roark's moves (data/intermediate_kaizo/moves.json)
# with their EVAL_ATTACK sites; Task 6 appends the sites of their Expert_Main labels
# (:1623-1811) and of Risky_Main (:6530, for the effects of Risky_RiskyEffects,
# :6536-6562: HALVE_DEFENSE :6538 and RAISE_ALL_STATS_HIT :6550; layout.KAIZO checks that
# every transcribed effect of commands.RISKY_EFFECTS has it). A site that is transcribed but
# never reached in Kaizo data only costs width (:2671, below).
EFFECT_SITE_LINES: dict[int, tuple[int, ...]] = {
    0: (6383,),  # HIT: no Expert_Main entry (script.s:1623-1811)
    2: (6383,),  # POISON_HIT (Sludge Bomb): no Expert_Main entry
    # RECOVER_HALF_DAMAGE_DEALT (Giga Drain): Expert_DrainMove (:1629 -> :1827), roll :1835.
    3: (6383, 1835),
    4: (6383,),  # BURN_HIT (Flamethrower): no Expert_Main entry
    6: (6383,),  # PARALYZE_HIT (DragonBreath): no Expert_Main entry
    # HALVE_DEFENSE (Selfdestruct): EvalAttack_MaybeDeprioritize (:6365 -> :6374) before the
    # quad roll; Expert_Explosion (:1630 -> :1841), rolls :1859, :1865, :1870, :1875, :1880;
    # Risky :6530.
    7: (6374, 6383, 1859, 1865, 1870, 1875, 1880, 6530),
    19: (6383, 2328),  # DEF_DOWN: Expert_StatusDefenseDown (:1642 -> :2318)
    31: (6383,),  # FLINCH_HIT (Zen Headbutt): no Expert_Main entry
    # RESTORE_HALF_HP (Recover): Expert_Recovery (:1652 -> :2640), rolls :2665, :2671, :2674.
    # :2671 needs the player seen with a STEAL_STATUS_MOVE move (:2670), which no Kaizo move
    # is; Expert_Recovery_Unused's :2657 follows a GoTo (:2652) and no jump targets it.
    32: (6383, 2665, 2671, 2674),
    # STATUS_BADLY_POISON (Toxic): Expert_ToxicLeechSeed (:1653 -> :2680), rolls :2690,
    # :2695, :2704.
    33: (6383, 2690, 2695, 2704),
    # SET_LIGHT_SCREEN (Light Screen): Expert_LightScreen (:1654 -> :2710), rolls :2718, :2724.
    35: (6383, 2718, 2724),
    52: (6383, 2075),  # SPEED_UP_2 (Rock Polish): Expert_StatusSpeedUp (:1665 -> :2066)
    59: (6383, 2328),  # DEF_DOWN_2: Expert_StatusDefenseDown (:1671 -> :2318)
    69: (6383,),  # LOWER_DEFENSE_HIT (Crunch): no Expert_Main entry
    72: (6383,),  # LOWER_SP_DEF_HIT (Earth Power): no Expert_Main entry
    76: (6383,),  # CONFUSE_HIT (Water Pulse): no Expert_Main entry
    # CURSE (Curse): Expert_Curse (:1700 -> :3452), rolls :3477, :3481, :3486, :3491.
    109: (6383, 3477, 3481, 3486, 3491),
    128: (6383, 3726, 3731),  # HIT_BEFORE_SWITCH: Expert_Pursuit (:1706 -> :3706)
    # RAISE_ALL_STATS_HIT (AncientPower): no Expert_Main entry; Risky :6530.
    140: (6383, 6530),
    186: (6383,),  # REMOVE_SCREENS (Brick Break): Expert_BrickBreak (:1739 -> :4559), no roll
    261: (6383,),  # WHIRLPOOL (Sand Tomb): no Expert_Main entry (plan K3c Ruling 5)
    266: (6383, 6274),  # STEALTH_ROCK: Expert_StealthRock (:1805 -> :6268)
    269: (6383,),  # RECOIL_HALF (Head Smash): Expert_RecoilMove (:1806 -> :6289), no roll
    275: (6383,),  # FLINCH_PARALYZE_HIT (Thunder Fang): no Expert_Main entry
}
# The transcribed effects (EFFECT_SITE_LINES) whose Expert_Main entry goes to a label added
# after the vanilla transcription: effect -> (label, the script.s line of its entry in
# Expert_Main's dispatch table, :1628-1808). The other transcribed effects either have no
# entry (they end at :1811) or go to a vanilla-transcribed label (DEF_DOWN :1642, DEF_DOWN_2
# :1671, HIT_BEFORE_SWITCH :1706, STEALTH_ROCK :1805). A per-slot layout refuses a move whose
# effect is here but not in its expert_transcribed set. The workflow for a future effect with
# an Expert_Main entry: add it here first (with its EFFECT_SITE_LINES row). It is then
# pending: in the table, not dispatched by policy._expert, not in KAIZO_EXPERT_TRANSCRIBED,
# and every move with it is refused; the tests allow this state. Then transcribe its label
# (an _EXPERT_LABELS row) and add the effect to KAIZO_EXPERT_TRANSCRIBED by hand.
KAIZO_EXPERT_LABELS: dict[int, tuple[str, int]] = {
    3: ("Expert_DrainMove", 1629),  # RECOVER_HALF_DAMAGE_DEALT (Giga Drain)
    7: ("Expert_Explosion", 1630),  # HALVE_DEFENSE (Selfdestruct)
    32: ("Expert_Recovery", 1652),  # RESTORE_HALF_HP (Recover)
    33: ("Expert_ToxicLeechSeed", 1653),  # STATUS_BADLY_POISON (Toxic)
    35: ("Expert_LightScreen", 1654),  # SET_LIGHT_SCREEN (Light Screen)
    52: ("Expert_StatusSpeedUp", 1665),  # SPEED_UP_2 (Rock Polish)
    109: ("Expert_Curse", 1700),  # CURSE (Curse)
    186: ("Expert_BrickBreak", 1739),  # REMOVE_SCREENS (Brick Break)
    269: ("Expert_RecoilMove", 1806),  # RECOIL_HALF (Head Smash)
}
# Of KAIZO_EXPERT_LABELS, the effects whose Expert label policy._expert transcribes
# (layout.KAIZO expert_transcribed), written out: it is not derived from the table, so an
# entry added to the table stays refused until its label is transcribed (K3c Task 6 review
# I1). Pinned against policy.EXPERT_DISPATCH, the effects _expert branches on
# (tests/ai/test_kaizo_policy.py test_expert_transcribed_is_the_dispatch).
KAIZO_EXPERT_TRANSCRIBED: frozenset[int] = frozenset(
    {
        3,  # RECOVER_HALF_DAMAGE_DEALT: Expert_DrainMove
        7,  # HALVE_DEFENSE: Expert_Explosion
        32,  # RESTORE_HALF_HP: Expert_Recovery
        33,  # STATUS_BADLY_POISON: Expert_ToxicLeechSeed
        35,  # SET_LIGHT_SCREEN: Expert_LightScreen
        52,  # SPEED_UP_2: Expert_StatusSpeedUp
        109,  # CURSE: Expert_Curse
        186,  # REMOVE_SCREENS: Expert_BrickBreak
        269,  # RECOIL_HALF: Expert_RecoilMove
    }
)
assert set(KAIZO_EXPERT_LABELS) <= set(EFFECT_SITE_LINES)
assert KAIZO_EXPERT_TRANSCRIBED <= set(KAIZO_EXPERT_LABELS)

# The most sites one move slot can reach in the Kaizo transcription: Selfdestruct
# (HALVE_DEFENSE): EvalAttack :6374 and :6383, Expert_Explosion :1859, :1865, :1870, :1875
# and :1880, Risky :6530 (layout.KAIZO asserts every row fits).
KAIZO_SLOT_SITES = 8

# The most guessing LoadBattlerAbility DEFENDER reads (trainer_ai.c:1170-1209) one move
# slot's path reaches in the Kaizo transcription (E1 Ruling R1), all in Basic_Main: Toxic
# (STATUS_BADLY_POISON) is not damaging, so FlagMoveDamageScore's no-comparison jump
# (script.s:62-63) skips Basic_CheckForImmunity's :71; it reads Basic_CheckSoundproof's :117
# and Basic_CheckCannotPoison's :529 and :539. A damaging move reads :71 and :117 and, at
# most, one label read (:470 for a Defense drop); Selfdestruct is a no-calc effect
# (sNoDamageCalcMoveEffects, trainer_ai.c:31-46) and reads :117 and :304. EVAL_ATTACK,
# EXPERT and RISKY load no defender ability on the transcribed paths (Expert_RecoilMove
# loads the attacker's, :6296). tests/ai/test_kaizo_outcome.py
# test_slot_guesses_fit_every_transcribed_effect finds it from the data.
KAIZO_SLOT_GUESSES = 3

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
# 2**MAX_JOINT_SITES patterns of script-roll sites and guess columns.
MAX_JOINT_SITES = 10


class AiOutcome(NamedTuple):
    sim_roll: jnp.ndarray  # int8[4], roll index 0-15 per opponent move slot
    # int8[] (flat layout) or int8[MOVES, slot_guesses] (per-slot layout, column k of slot m
    # = the slot's k-th guessing read): player ability slot 0/1
    ability_guess: jnp.ndarray
    # int8[MAX_SCRIPT_ROLLS] (flat layout) or int8[MOVES, slot_sites] (per-slot layout):
    # 1 = the site's jump is taken
    script_rolls: jnp.ndarray
    tiebreak: jnp.ndarray  # int8[], RandNext % 12
    # int8[]: the first stopping switch draw, 0..sites - 1; sites (layout.switch_sites: 9
    # vanilla, 44 Kaizo) = none; -1 refused
    switch_stop: jnp.ndarray


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
    (``_VANILLA_COLUMN``), or ``MAX_SCRIPT_ROLLS`` (never rolls 1) for a line no vanilla
    move reaches."""
    k = script_roll_sites(state)
    k = jnp.concatenate([k, jnp.full((MOVES, 1), MAX_SCRIPT_ROLLS, k.dtype)], axis=1)
    unused = k.shape[1] - 1
    columns = [
        unused if _VANILLA_COLUMN[line] is None else _VANILLA_COLUMN[line] for line in SITE_LINES
    ]
    return k[:, jnp.array(columns)]


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
    """``float32[sites + 1]``: over the ``switch_sites`` kinds (``int8[sites]``), the
    probability that site i is the first to stop,
    ``prod_{h<i} (1 - p_h) * p_i``, then (last entry) that none stops; all NaN for a
    refused layout."""
    kind = jnp.asarray(switch_sites, _I32)
    p = _SWITCH_P[kind]
    reach = jnp.cumprod(jnp.concatenate([jnp.ones(1, _F32), 1 - p]))  # [sites + 1]
    probs = jnp.concatenate([reach[:-1] * p, reach[-1:]])
    return jnp.where(jnp.any(kind == SwitchSite.REFUSED), jnp.nan, probs)


def _first_stop(draws, switch_sites) -> jnp.ndarray:
    """The first site whose u16 ``draws[i]`` stops the check, else the number of sites
    (``SWITCH_NONE`` for the flat layout); ``SWITCH_REFUSED`` for a refused layout."""
    kind = jnp.asarray(switch_sites, _I32)
    zero = draws % _SWITCH_MOD[kind] == 0
    stops = (kind != SwitchSite.NONE) & (zero == _SWITCH_STOPS_ON_ZERO[kind])
    first = jnp.where(jnp.any(stops), jnp.argmax(stops), kind.shape[0])
    return jnp.where(jnp.any(kind == SwitchSite.REFUSED), SWITCH_REFUSED, first)


def sample_ai_outcome(
    key, tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> AiOutcome:
    """Draw every field from its decomp probability for this state (the switch draws
    over ``switch_sites(tables, state, layout.switch_sites)``). ``layout``: a per-slot layout
    draws ``script_rolls`` as ``int8[MOVES, layout.slot_sites]`` (``slot_roll_thresholds``)
    and ``ability_guess`` as ``int8[MOVES, layout.slot_guesses]``, every column a fair coin;
    the default, and any flat layout, is the vanilla layout."""
    sites = SWITCH_SITES if layout is None else layout.switch_sites
    layout_sites = switch_sites(tables, state, sites)
    k = jax.random.split(key, 4)
    if layout is not None and layout.per_slot_sites:
        shape, thresholds = (MOVES, layout.slot_sites), slot_roll_thresholds(tables, state, layout)
        guess_shape = (MOVES, layout.slot_guesses)
    else:
        shape, thresholds = (MAX_SCRIPT_ROLLS,), script_roll_thresholds(state)
        guess_shape = ()
    rolls = jax.random.randint(k[1], shape, 0, 256, dtype=_I32)
    draws = jax.random.randint(k[3], (sites,), 0, 65536, dtype=_I32)
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, _I8),
        ability_guess=jax.random.randint(k[0], guess_shape, 0, 2, dtype=_I32).astype(_I8),
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
    tables: Tables, state: BattleState, layout: AiLayout | None = None, guess_columns=None
) -> tuple[AiOutcome, jnp.ndarray]:
    """Every outcome combination (a batched ``AiOutcome`` with leading axis K) and its
    exact probability ``float32[K]``. Flat layout (the default): K is
    ``ENUMERATION_WIDTH`` (3840) for every state; combinations with an unused script-roll
    site set to 1, or a ``switch_stop`` at a ``SwitchSite.NONE`` entry, have probability
    0. The switch draws are over ``switch_sites(tables, state)``.

    A per-slot ``layout`` (a test oracle, concrete inputs only): every combination of the
    used sites of the four slots and of the first ``guess_columns[m]`` guess columns of each
    slot m (``int[MOVES]``, required: the guessing reads the slot's paths reach, which
    ``policy.ai_action_distribution_enumerated`` finds; a column no path reads would only
    repeat every outcome), ``K = 2**(sum(slot_site_counts) + sum(guess_columns)) *
    (layout.switch_sites + 1) * TIEBREAK_MOD``; ``ValueError`` when the bits number more
    than ``MAX_JOINT_SITES``."""
    if layout is None or not layout.per_slot_sites:
        return _enumerate_flat(tables, state)
    if guess_columns is None:
        raise ValueError("a per-slot enumeration needs guess_columns (int[MOVES])")
    guess_columns = np.clip(np.asarray(guess_columns, np.int64), 0, layout.slot_guesses)
    thresholds = np.asarray(slot_roll_thresholds(tables, state, layout))
    used = np.argwhere(thresholds > 0)  # [U, 2] (slot, column), script order per slot
    guessed = np.array(
        [(m, k) for m in range(MOVES) for k in range(int(guess_columns[m]))], np.int64
    ).reshape(-1, 2)  # [G, 2] (slot, guess column), path order per slot
    n_bits = len(used) + len(guessed)
    if n_bits > MAX_JOINT_SITES:
        raise ValueError(
            f"{len(used)} script-roll sites and {len(guessed)} guess columns: more than the "
            f"{MAX_JOINT_SITES} bits a joint enumeration lists"
        )
    bits, switch, tiebreak = (
        a.reshape(-1)
        for a in np.meshgrid(
            np.arange(2**n_bits),
            np.arange(layout.switch_sites + 1),
            np.arange(_TIEBREAK_VALUES),
            indexing="ij",
        )
    )
    on = (bits[:, None] >> np.arange(n_bits)) & 1  # [K, U + G]
    rolls = np.zeros((len(bits), MOVES, layout.slot_sites), np.int8)
    rolls[:, used[:, 0], used[:, 1]] = on[:, : len(used)]
    guesses = np.zeros((len(bits), MOVES, layout.slot_guesses), np.int8)
    guesses[:, guessed[:, 0], guessed[:, 1]] = on[:, len(used) :]
    p_jump = thresholds[used[:, 0], used[:, 1]].astype(np.float64) / 256
    p_rolls = np.prod(np.where(on[:, : len(used)] == 1, p_jump, 1 - p_jump), axis=1)
    p_switch = np.asarray(
        switch_stop_probs(switch_sites(tables, state, layout.switch_sites)), np.float64
    )
    probs = p_rolls * p_switch[switch] * TIEBREAK_P[tiebreak] * 0.5 ** len(guessed)
    outs = AiOutcome(
        sim_roll=jnp.full((len(bits), MOVES), SIM_ROLL_MAX, _I8),
        ability_guess=jnp.asarray(guesses),
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
