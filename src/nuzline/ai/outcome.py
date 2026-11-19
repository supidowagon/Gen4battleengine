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
  scalar for the decision, exact for vanilla Roark (roark.md 8.2); a per-slot layout reads
  every guess from ``script_rolls`` (below) and leaves this field 0, unread. It is ignored
  when no guess is made.
* ``script_rolls[k]``: 1 when the k-th ``IfRandomLessThan N`` site in script order
  jumps (``RandNext % 256 < N``), probability N/256 (8.3). The site layout follows
  the opponent active's moveset: ``script_roll_thresholds``.
* ``tiebreak``: ``RandNext % 12`` (0-11), so ``tiebreak % n`` is exactly the decomp's
  ``RandNext % n`` for n = 1..4 tied slots (``trainer_ai.c:338``; 8.4).
* ``switch_stop``: the first of the voluntary-switch draws of
  ``TrainerAI_ShouldSwitch`` (``trainer_ai.c:3894``; roark.md 8.5 and 11.1) that
  stops the check, 0..``sites - 1``, or ``sites`` when none does, where ``sites`` is the
  layout's ``switch_sites``: ``SWITCH_SITES`` (9, ``SWITCH_NONE`` = 9 none) for the flat
  layout, 60 for Kaizo's party of six (docs/ai/kaizo_roark.md 11). A
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

**Per-slot layouts** (``AiLayout.per_slot_sites``, Kaizo; docs/ai/kaizo_roark.md 8; E1
Rulings R1 and R4). The flat layout above is the vanilla one and does not change. A per-slot
layout keys every random read of the scoring by move slot and by its order on the slot's
path: ``script_rolls`` is ``uint8[MOVES, layout.slot_reads]``, and entry k of slot m is the
k-th random read on slot m's scoring path (flag order, trainer_ai.c:297-310, then line
order), as ``RandNext() % 256``, whatever the read is:

* an ``IfRandomLessThan N`` (``AICmd_IfRandomLessThan``, trainer_ai.c:528-538) jumps when the
  draw is below N, probability N/256;
* a guessing ``LoadBattlerAbility DEFENDER`` (``AICmd_LoadBattlerAbility``,
  trainer_ai.c:1170-1209: the AI does not know the player's ability, it is not Shadow Tag,
  Magnet Pull or Arena Trap, and the species has two nonzero abilities) takes ability 1 when
  the draw's low bit is set (``BattleSystem_RandNext() & 1``, :1195), else ability 2,
  probability 1/2 each.

Each read is its own ``RandNext``, so the k-th read on a path is one fresh independent draw
whose probability comes from the read itself, and which read is the k-th is decided by the
earlier draws of the path: keying by read order is exact. No read is shared between slots,
since ``TrainerAI_EvalMoves`` scores slot by slot (:483-525), which is what lets ``policy``
factor the exact distribution per slot. A path that makes more random reads than
``slot_reads`` is refused (``AiContext.missing_read``), never given a reused draw. The read
order replaces K3c's per-effect site columns and batch 1's guess columns: no site table is
needed, so a forgotten site cannot be read as never jumping, and a flag set's reads do not
multiply the width.

``policy`` enumerates a per-slot layout's reads as bit patterns: bit 0 / 1 of read k stands
for the draw ``READ_DRAWS[bit]`` (254 / 1), which split every read in two (254 is below no
``SITE_N`` N, 1 below every one; their low bits differ), and the scoring path records each
read's probability (``AiContext.weight``) and the number of reads it made
(``AiContext.reads``). A pattern whose bits past a slot's reads are not all 0 repeats a
shorter one and is given weight 0.

Sites are named by their ``script.s`` line (``SITE_N``): every ``IfRandomLessThan`` the
transcription reads passes its N and its line to ``commands.if_random_less_than``. The flat
layout finds the draw's index in ``AiContext.sites`` (one column per ``SITE_LINES`` entry,
``line_sites``); the per-slot layout takes the slot's next read.

**The damage estimate's draws** (E1 ruling G-39). ``TrainerAI_CalcDamage`` draws for two move
ids (trainer_ai.c:2868-3110): MOVE_PSYWAVE's damage ``level * (RandNext() % 11 + 5) / 10``
(:3014-3016) and MOVE_MAGNITUDE's power ``RandNext() % 100`` in seven buckets (:3028-3048);
Kaizo's Triple Axel (149) and Bulldoze (222) sit on those ids. Every damage command draws
afresh for each such move it estimates: ``FlagMoveDamageScore`` (:1041, ``CalcAllDamage`` over
the attacker's four moves), ``IfCurrentMoveKills`` (:1560), ``IfPartyMemberDealsMoreDamage``
(:2130, :2156, the attacker's moves then every living party member's), ``IfBattlerDealsMoreDamage``
(:2210, :2228) and ``CheckIfHighestDamageWithPartner`` (:2402, doubles only). Each draw's law
(``CALC_DRAW_P``, by ``calc_draw_kind``): the class ``calc_draw_class`` of a uniform u16, Psywave's
``% 11`` (65536 = 11 * 5957 + 9: classes 0-8 have 5958 values, 9-10 5957) and Magnitude's
bucket of ``% 100`` (65536 = 100 * 655 + 36: residues 0-35 have 656 values, 36-99 655).

Each command's result is one yes/no (the jump, or ``calcTemp``'s HIGHEST / NOT_HIGHEST), and its
draws reach nothing else (local arrays, trainer_ai.c:1016-1063, :1525-1567, :2106-2174,
:2187-2240). So in the per-slot layout a command whose estimate draws is **one read** of the
slot's path, whose result has the exact probability of the draws' joint law
(``commands._calc_read``): a multi-valued draw, or several, collapse to one binary read, so the
width grows by 2 per such command, not by 11 or 7 per draw. ``script_rolls`` keeps its meaning
for the enumeration (the read's bit is its result: ``READ_DRAWS[1]`` yes). A sampled outcome
also carries the raw draws, ``calc_draws``: ``uint16[MOVES, slot_reads, CALC_DRAWS]``, entry
``[m, k, j]`` the j-th ``RandNext()`` of slot m's k-th read when that read is a damage command
(j counting the moves it estimates with a drawing case, in the decomp's order), and the
command's result is computed from them as the decomp does. The enumeration's patterns
(``policy._pattern_outcomes``) take the result from the bit instead: their ``calc_draws`` is
the marker of no raw draws, a last axis of 0 (``no_calc_draws``), and so does every flat
outcome (``uint16[MOVES, 0, 0]``: the flat layout's damage commands never draw,
``commands._calc_read``). Every outcome carries one or the other (``policy._check_outcome``).
Where the result is never read (``FlagMoveDamageScore`` compared only with
``AI_NO_COMPARISON_MADE``, which does not depend on the draws: script.s:62-63, :112-113,
:6511-6512, :6572-6573, then overwritten by ``LoadBattlerAbility`` :67/:117 or ended), the
draws change nothing and are no read, as the SLOWER compare's tie (plan K3c Ruling 3); the
loose oracle still draws every one of them, digit by digit, and checks that.
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
from nuzline.engine.damage import AI_DRAW_CLASSES, AI_MAGNITUDE, AI_PSYWAVE
from nuzline.engine.state import MOVES, SLOTS, BattleState

if TYPE_CHECKING:
    from nuzline.ai.layout import AiLayout

__all__ = [
    "ABILITY_GUESSES",
    "AI_SITE_MOVES",
    "CALC_DRAWS",
    "CALC_DRAW_CLASSES",
    "CALC_DRAW_P",
    "ENUMERATION_WIDTH",
    "KAIZO_EXPERT_LABELS",
    "KAIZO_EXPERT_TRANSCRIBED",
    "KAIZO_SLOT_READS",
    "KAIZO_TRANSCRIBED",
    "MAX_ENUMERATED",
    "MAX_EXPERT_SITES",
    "MAX_SCRIPT_ROLLS",
    "READ_DRAWS",
    "SITE_LINES",
    "SITE_N",
    "SWITCH_NONE",
    "SWITCH_REFUSED",
    "SWITCH_SITES",
    "TIEBREAK_MOD",
    "TIEBREAK_P",
    "AiOutcome",
    "SwitchSite",
    "calc_draw_class",
    "calc_draw_kind",
    "calc_kind_class",
    "enumerate_ai_outcomes",
    "line_sites",
    "sample_ai_outcome",
    "script_roll_count",
    "script_roll_probs",
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
# PASS_NOT_VERY_EFFECTIVE, REFUSED, WONDER_GUARD): (modulus, the residues lo <= r < hi that
# stop) -> stop probability, NaN for REFUSED. 65536 = 10 * 6553 + 6 = 3 * 21845 + 1, so the
# residues 0 and 1 of 3 have 21846 and 21845 values (WONDER_GUARD's % 3 < 2: 43691).
_SWITCH_MOD = jnp.array([1, 10, 2, 3, 1, 3], _I32)
_SWITCH_LO = jnp.array([0, 1, 0, 0, 0, 0], _I32)
_SWITCH_HI = jnp.array([0, 10, 1, 1, 0, 2], _I32)
_SWITCH_P = jnp.array([0, 58982, 32768, 21846, jnp.nan, 43691], _F32) / 65536

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
    # E1 batch 2b: SetupFirstTurn_Main, turn 0 only, an effect of its table.
    6429: 80,  # SetupFirstTurn_Main (+2 when it does NOT jump)
    # E1 batch 2c: PrioritizeExtremes_Main and CheckHP_Main.
    6515: 100,  # PrioritizeExtremes_Main (+2 when it does NOT jump)
    7718: 50,  # CheckHP_TryScoreMinus2 (-2 when it does NOT jump)
    7742: 50,  # CheckHP_Target_TryScoreMinus2 (-2 when it does NOT jump)
    # E1 batch 4: the stat-raise Expert labels (script.s:1978-2155).
    1989: 100,  # Expert_StatusAttackUp (+3 or more: -1 when it does NOT jump)
    1995: 128,  # Expert_StatusAttackUp_CheckUserAtMaxHP (+2 when it does NOT jump)
    2001: 40,  # Expert_StatusAttackUp_CheckUserHPRange (-2 when it does NOT jump)
    2023: 100,  # Expert_StatusDefenseUp
    2029: 128,  # Expert_StatusDefenseUp_CheckUserAtMaxHP
    2034: 200,  # Expert_StatusDefenseUp_CheckUserHighHP (jump: no further change)
    2043: 60,  # Expert_StatusDefenseUp_CheckUserMediumHP
    2046: 60,  # Expert_StatusDefenseUp_UserAtLowHP
    2092: 100,  # Expert_StatusSpAttackUp
    2098: 128,  # Expert_StatusSpAttackUp_CheckUserAtMaxHP
    2104: 70,  # Expert_StatusSpAttackUp_CheckUserHPRange
    2126: 100,  # Expert_StatusSpDefenseUp
    2132: 128,  # Expert_StatusSpDefenseUp_CheckUserAtMaxHP
    2137: 200,  # Expert_StatusSpDefenseUp_CheckUserHighHP
    2146: 60,  # Expert_StatusSpDefenseUp_CheckUserMediumHP
    2149: 60,  # Expert_StatusSpDefenseUp_UserAtLowHP
    # E1 batch 6: the status moves' Expert labels.
    1821: 128,  # Expert_StatusSleep_TryScorePlus1 (+1 when it does NOT jump)
    2858: 128,  # Expert_Flatter (+1 when it does NOT jump)
    2864: 128,  # Expert_StatusConfuse (-1 when it does NOT jump)
    2950: 20,  # Expert_StatusParalyze_TryScorePlus3 (+3 when it does NOT jump)
    # E1 batch 7: the Expert-only hits.
    2271: 100,  # Expert_BypassAccuracyMove_TryScorePlus1 (+1 when it does NOT jump)
    2362: 70,  # Expert_StatusSpeedDown_TryScorePlus2 (+2 when it does NOT jump)
    2836: 128,  # Expert_HighCritical (neutral: on to TryScorePlus1 when it does NOT jump)
    2839: 128,  # Expert_HighCritical_TryScorePlus1 (+1 when it does NOT jump)
    4882: 128,  # Expert_Brine (+1 more when it does NOT jump)
    # E1 batch 9.
    3346: 128,  # Expert_DestinyBond (+1 when it does NOT jump)
    3351: 128,  # Expert_DestinyBond_CheckUserMediumHP (+1 when it does NOT jump)
    3356: 100,  # Expert_DestinyBond_CheckUserLowHP (+2 when it does NOT jump)
    3595: 128,  # Expert_Spikes (+1 when it does NOT jump)
    3602: 64,  # Expert_Spikes_TryScorePlus1 (+1 when it does NOT jump)
    4771: 70,  # Expert_DragonDance (-1 when it does NOT jump)
    4776: 128,  # Expert_DragonDance_TryScorePlus1 (+1 when it does NOT jump)
    6018: 128,  # Expert_ToxicSpikes (+1 when it does NOT jump)
    6025: 64,  # Expert_ToxicSpikes_TryScorePlus1 (+1 when it does NOT jump)
    # E1 batch 10: the stat drops and Reflect.
    2293: 50,  # Expert_StatusAttackDown_CheckTargetStatStage (-2 when it does NOT jump)
    2303: 128,  # Expert_StatusAttackDown_CheckLastUsedMove (-2 when it does NOT jump)
    2421: 50,  # Expert_StatusSpDefenseDown_TryScoreMinus2 (-2 when it does NOT jump)
    2510: 50,  # Expert_StatusEvasionDown_TryScoreMinus2 (-2 when it does NOT jump)
    2901: 128,  # Expert_Reflect (+1 when it does NOT jump)
    2907: 64,  # Expert_Reflect_CheckLastUsedMove (+1 when it does NOT jump)
    # E1 batch 11: Tailwind and Payback (COMPARE_SPEED_FASTER, E1 Ruling R8).
    4992: 64,  # Expert_Tailwind (jump: nothing)
    4996: 64,  # Expert_Tailwind, 31-75 % HP (+1 when it does NOT jump)
    5179: 64,  # Expert_Payback (+1 when it does NOT jump)
    # E1 batch 12.
    3388: 100,  # Expert_Reversal_TryScorePlus1 (+1 when it does NOT jump)
    4968: 64,  # Expert_Pluck, first turn (+1 when it does NOT jump)
    4972: 128,  # Expert_Pluck_TryScorePlus1 (+1 when it does NOT jump)
    # Expert_Counter (:3116-3153) and Expert_MirrorCoat (:3873-3910), in line order
    3116: 10,
    3121: 100,
    3130: 100,
    3136: 100,
    3142: 100,
    3150: 50,
    3153: 100,
    3873: 10,
    3878: 100,
    3887: 100,
    3893: 100,
    3899: 100,
    3907: 50,
    3910: 100,
    # Expert_Protect (:3536-3589)
    3541: 128,  # -2 when it does NOT jump
    3564: 85,  # +2 when it jumps
    3571: 128,  # -1 when it does NOT jump
    3578: 128,  # -1 more when it does NOT jump
    2794: 192,  # Expert_OHKOMove (+1 when it does NOT jump; E1 batch 14)
    # Expert_UTurn (:5085-5145; E1 batch 15)
    5114: 64,
    5119: 64,
    5126: 128,
    5130: 64,
    5134: 128,
    5139: 128,
    # E1 batch 16
    6229: 50,  # Expert_Blizzard (-3 when it does NOT jump)
    4148: 200,  # Expert_FocusPunch (+1 when it does NOT jump)
    4157: 100,  # Expert_FocusPunch_TryScorePlus1 (+1 when it does NOT jump)
    # E2 batch 11 (B11-4)
    3192: 30,  # Expert_Encore_TryScorePlus3 (+3 when it does NOT jump)
    # E1 batch 17
    2603: 64,  # Expert_ForceSwitch_75PercentScorePlus2
    2607: 128,  # Expert_ForceSwitch_50PercentScorePlus2
    2995: 96,  # Expert_Substitute, Focus Punch known
    3002: 100,
    3006: 100,
    3010: 100,
    3038: 100,  # Expert_Substitute_TryScorePlus1
    1897: 51,  # Expert_DreamEater_TryScorePlus3
    6281: 64,  # Expert_StealthRock_TryScorePlus1: with Roar or Whirlwind (since batch 17)
    # E1 prep batch P1 (on E2's state)
    6405: 170,  # EvalAttack_TryScorePlus4: Future Sight's kill bonus
    5053: 10,  # Expert_MetalBurst, 30 % HP or less
    5058: 100,
    5062: 192,
    5070: 100,
    5075: 100,
    2453: 100,  # Expert_StatusAccuracyDown
    2458: 80,
    2463: 70,
    2468: 70,
    2473: 128,
    2479: 128,
    2484: 70,
    2492: 70,
    2204: 100,  # Expert_StatusEvasionUp
    2209: 128,
    2215: 80,
    2218: 50,
    2223: 70,
    2228: 128,
    2234: 128,
    2240: 70,
    2248: 70,
    2821: 128,  # Expert_BindingMove
    4577: 180,  # Expert_KnockOff
    4638: 100,  # Expert_Imprison
    # E1 prep batch P2 (on E2's state)
    5563: 128,  # Expert_Copycat: an encouraged last move
    5568: 32,  # Expert_Copycat_TryScorePlus2
    5576: 80,  # Expert_Copycat_CheckMoveEncouraged
    # E1 prep batch P3: Baton Pass
    3680: 80,  # Expert_BatonPass_HighStatStage_TryScorePlus2
    6577: 80,  # BatonPass_Main: the attacker without a Baton Pass effect
    6591: 20,  # BatonPass_EvalMove: +3
    # E2 batch 12, the stat family (B12-11).
    2541: 50,  # Expert_Haze_TryScoreMinus3
    2555: 50,  # Expert_Haze_CheckToEncourage (-1 when it does NOT jump)
    2560: 50,  # Expert_Haze_TryScorePlus3
    3836: 50,  # Expert_PsychUp_CheckUserStatStages (-2 when it does NOT jump)
    5016: 128,  # Expert_Acupressure (to _TryScorePlus1 when it does NOT jump)
    5019: 64,  # Expert_Acupressure_TryScorePlus1
    5454: 96,  # Expert_PowerTrick_LikelyScorePlus1
    5459: 128,  # Expert_PowerTrick_CoinFlipScorePlus1
    5464: 164,  # Expert_PowerTrick_UnlikelyScorePlus1
    5739: 128,  # Expert_PowerSwap_TryScorePlus5
    5744: 128,  # Expert_PowerSwap_TryScorePlus4
    5749: 128,  # Expert_PowerSwap_TryScorePlus3
    5754: 128,  # Expert_PowerSwap_TryScorePlus2
    5759: 128,  # Expert_PowerSwap_TryScorePlus1
    5873: 128,  # Expert_GuardSwap_TryScorePlus5
    5878: 128,  # Expert_GuardSwap_TryScorePlus4
    5883: 128,  # Expert_GuardSwap_TryScorePlus3
    5888: 128,  # Expert_GuardSwap_TryScorePlus2
    5893: 128,  # Expert_GuardSwap_TryScorePlus1
    6252: 50,  # Expert_Captivate_CheckLowStatStage
    6262: 64,  # Expert_Captivate_CheckOpponentLastMove
    # E2 batch 12, the healing family (B12-18).
    2766: 70,  # Expert_Rest_FasterCheckHP
    2775: 50,  # Expert_Rest_SlowerCheckHP
    2783: 50,  # Expert_Rest_CheckForSnatch
    2786: 10,  # Expert_Rest_TryScorePlus3
    5301: 128,  # Expert_PsychoShift
    # E2 batch 10: the charge-turn moves.
    4058: 80,  # Expert_ChargeTurnWithInvuln_TryScorePlus1 (+1 when it does NOT jump)
    # E1 Roark close batch: AI_FLAG_HARRASSMENT (G-37).
    8030: 128,  # Harrassment_Main (+2 when it does NOT jump)
    # E1 trial batch.
    3645: 70,  # Expert_Endure_TryScorePlus1 (+1 when it does NOT jump)
    6113: 128,  # Expert_MagnetRise_CheckOpponentTyping (+1 when it does NOT jump)
    4795: 128,  # Expert_Gravity (goes to TryScorePlus1 when it jumps)
    4799: 64,  # Expert_Gravity_TryScorePlus1 (+1 when it does NOT jump)
    # E2 batch 13 (B13-23).
    4526: 50,  # Expert_Recycle (+1 when it does NOT jump)
    5204: 128,  # Expert_Assurance (jumps to _TryScorePlus1)
    5212: 128,  # Expert_Assurance_TryScorePlus1
    5333: 30,  # Expert_TrumpCard (the Pressure +1)
    5347: 100,  # Expert_TrumpCard_ScorePlus1
    5225: 128,  # Expert_Embargo (+1 when it does NOT jump), B13-32
    3089: 100,  # Expert_Disable_TryScoreMinus1 (-1 when it does NOT jump), E3-1-3
    5935: 128,  # Expert_Punishment_TryScorePlus4
    5939: 128,  # Expert_Punishment_TryScorePlus3
    5943: 128,  # Expert_Punishment_TryScorePlus2
    5947: 128,  # Expert_Punishment_TryScorePlus1
    # E3 batch 1 round 2.
    5984: 128,  # Expert_WorrySeed_CheckUserHP (to _TryScorePlus1 when it jumps)
    5988: 64,  # Expert_WorrySeed_TryScorePlus1 (+1 when it does NOT jump)
    5481: 64,  # Expert_GastroAcid (to _End when it jumps: 0 for the whole script)
    5484: 128,  # Expert_GastroAcid (to _ContinueHPCheck when it jumps: skips the -1)
    5532: 32,  # Expert_MeFirst_TryScorePlus1 (jumps to _CheckLastUsedMove)
    5538: 128,  # Expert_MeFirst_CheckLastUsedMove (+1 when it does NOT jump)
    5542: 64,  # Expert_MeFirst_TryScorePlus1AndEnd (+1 when it does NOT jump)
    6210: 64,  # Expert_TrickRoom_TryScorePlus3 (+3 when it does NOT jump)
    4429: 50,  # Expert_ChangeUserAbility_TryScorePlus2 (+2 when it does NOT jump)
    5257: 128,  # Expert_Fling (power 30-60: to _End when it jumps, else -1)
    5268: 128,  # Expert_Fling_CheckWeakness (to _TryScorePlus1 when it jumps, else +1 first)
    5276: 64,  # Expert_Fling_TryScorePlus1 (+1 when it does NOT jump)
    1915: 128,  # Expert_MirrorMove (faster, last move in the table: to _End when it jumps, else +2)
    1922: 80,  # Expert_MirrorMove_TryScoreMinus1 (to _End when it jumps, else -1)
    5506: 64,  # Expert_LuckyChant (to _ScorePlus1 when it jumps: +1, else 0), E3-B2-7
    4499: 100,  # Expert_MagicCoat (to _CheckUserFirstTurn when it jumps, else -1), E3-B2-6
    4505: 150,  # Expert_MagicCoat_CheckUserFirstTurn (first turn: to _End when it jumps, else +1)
    4511: 30,  # Expert_MagicCoat_TryScoreMinus1 (to _End when it jumps, else -1)
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

# The battle effects (BATTLE_EFFECT_*, line - 1 of generated/move_battle_effects.txt) whose
# script paths the per-slot (Kaizo) layout transcribes: layout.KAIZO's transcribed set. K3c
# Task 5 added the effects of Kaizo Roark's moves (data/intermediate_kaizo/moves.json) with
# their BASIC and EVAL_ATTACK paths, Task 6 their Expert_Main labels (:1623-1811) and
# Risky_Main (:6521-6534). The random reads on their paths are keyed by read order (module
# docstring), so the effects carry no site table (E1 Ruling R4); the comments name each
# effect's Expert_Main label and its IfRandomLessThan lines. Every effect can reach
# EVAL_ATTACK's quad roll (:6383), behind IfMoveEffectivenessEquals QUADRUPLE (:6378).
KAIZO_TRANSCRIBED: frozenset[int] = frozenset(
    {
        0,  # HIT: no Expert_Main entry (script.s:1623-1811)
        2,  # POISON_HIT (Sludge Bomb): no Expert_Main entry
        3,  # RECOVER_HALF_DAMAGE_DEALT (Giga Drain): Expert_DrainMove (:1629), :1835
        4,  # BURN_HIT (Flamethrower): no Expert_Main entry
        6,  # PARALYZE_HIT (DragonBreath): no Expert_Main entry
        # HALVE_DEFENSE (Selfdestruct): EvalAttack_MaybeDeprioritize (:6365 -> :6374);
        # Expert_Explosion (:1630 -> :1841), :1859, :1865, :1870, :1875, :1880; Risky :6530.
        7,
        19,  # DEF_DOWN: Expert_StatusDefenseDown (:1642 -> :2318), :2328
        31,  # FLINCH_HIT (Zen Headbutt): no Expert_Main entry
        # RESTORE_HALF_HP (Recover): Expert_Recovery (:1652 -> :2640), :2665, :2671, :2674.
        # Expert_Recovery_Unused's :2657 follows a GoTo (:2652) and no jump targets it.
        32,
        # STATUS_BADLY_POISON (Toxic): Expert_ToxicLeechSeed (:1653 -> :2680), :2690, :2695,
        # :2704.
        33,
        35,  # SET_LIGHT_SCREEN: Expert_LightScreen (:1654 -> :2710), :2718, :2724
        52,  # SPEED_UP_2 (Rock Polish): Expert_StatusSpeedUp (:1665 -> :2066), :2075
        59,  # DEF_DOWN_2: Expert_StatusDefenseDown (:1671 -> :2318), :2328
        69,  # LOWER_DEFENSE_HIT (Crunch): no Expert_Main entry
        72,  # LOWER_SP_DEF_HIT (Earth Power): no Expert_Main entry
        76,  # CONFUSE_HIT (Water Pulse): no Expert_Main entry
        109,  # CURSE: Expert_Curse (:1700 -> :3452), :3477, :3481, :3486, :3491
        128,  # HIT_BEFORE_SWITCH (Pursuit): Expert_Pursuit (:1706 -> :3706), :3726, :3731
        140,  # RAISE_ALL_STATS_HIT (AncientPower): no Expert_Main entry; Risky :6530
        186,  # REMOVE_SCREENS (Brick Break): Expert_BrickBreak (:1739 -> :4559), no roll
        261,  # WHIRLPOOL (Sand Tomb): no Expert_Main entry (plan K3c Ruling 5)
        266,  # STEALTH_ROCK: Expert_StealthRock (:1805 -> :6268), :6274, :6281
        269,  # RECOIL_HALF (Head Smash): Expert_RecoilMove (:1806 -> :6289), no roll
        275,  # FLINCH_PARALYZE_HIT (Thunder Fang): no Expert_Main entry
        # E1 batch 3: the 26 trainer-used effects with no Basic_ScoreMoveEffect (:133-286) or Expert_Main
        # (:1628-1808) entry (docs/ai/kaizo_roark.md 5.9). Every flag script treats them
        # generically: FlagMoveDamageScore, IfCurrentMoveKills and the damage estimates read only
        # the no-calc and alt-power lists (trainer_ai.c:1012-1063, :1525-1560, :2800-2830), and
        # the other mentions in script.s are tables the generic code already reads.
        5,  # FREEZE_HIT
        29,  # MULTI_HIT
        36,  # TRI_ATTACK
        44,  # HIT_TWICE
        45,  # CRASH_ON_MISS
        73,  # LOWER_ACCURACY_HIT
        83,  # CALL_RANDOM_MOVE; Risky_RiskyEffects :6543
        101,  # LEAVE_WITH_1_HP
        103,  # PRIORITY_1; EvalAttack_ApplyKillBonuses :6400
        125,  # THAW_AND_BURN_HIT
        138,  # RAISE_DEF_HIT
        139,  # RAISE_ATTACK_HIT
        147,  # DOUBLE_DAMAGE_DIG
        149,  # DOUBLE_DAMAGE_FLY_OR_BOUNCE
        150,  # FLINCH_MINIMIZE_DOUBLE_HIT
        152,  # THUNDER; Expert_Main tests SKIP_CHARGE_TURN_IN_SUN for Expert_Thunder, :1718
        159,  # UPROAR
        172,  # MAKE_GLOBAL_TARGET
        180,  # USE_RANDOM_ALLY_MOVE
        197,  # SECRET_POWER; Harrassment table :8062
        202,  # BADLY_POISON_HIT
        257,  # DOUBLE_DAMAGE_DIVE
        271,  # LOWER_SP_DEF_2_HIT
        273,  # FLINCH_BURN_HIT
        274,  # FLINCH_FREEZE_HIT
        276,  # RAISE_SP_ATK_HIT
        # E1 batch 4: the stat raises (Basic_CheckHighStatStage_*, script.s:343-392;
        # Expert_Status*Up, :1978-2155) of the Kaizo moves; evasion came in
        # prep P1 (Minimize, 108) and the E1 trial batch (EVA_UP, 16; docs/ai/kaizo_roark.md 5.10).
        10,  # ATK_UP: Expert_StatusAttackUp (:1633), :1989, :1995, :2001
        11,  # DEF_UP: Expert_StatusDefenseUp (:1634), :2023, :2029, :2034, :2043, :2046
        13,  # SP_ATK_UP: Expert_StatusSpAttackUp (:1636), :2092, :2098, :2104
        50,  # ATK_UP_2: Expert_StatusAttackUp (:1663)
        51,  # DEF_UP_2: Expert_StatusDefenseUp (:1664)
        53,  # SP_ATK_UP_2: Expert_StatusSpAttackUp (:1666)
        54,  # SP_DEF_UP_2: Expert_StatusSpDefenseUp (:1667)
        # E1 batch 6: the status moves (docs/ai/kaizo_roark.md 5.11).
        1,  # STATUS_SLEEP: Basic_CheckCannotSleep (:134); Expert_StatusSleep (:1628), :1821
        49,  # STATUS_CONFUSE: Basic_CheckCannotConfuse (:163); Expert_StatusConfuse (:1662), :2864
        67,  # STATUS_PARALYZE: Basic_CheckCannotParalyze (:180); Expert_StatusParalyze (:1679), :2950
        84,  # STATUS_LEECH_SEED: Basic_CheckCannotLeechSeed (:183); Expert_ToxicLeechSeed (:1686)
        118,  # ATK_UP_2_STATUS_CONFUSION: Basic_CheckCannotConfuse (:201); Expert_Swagger (:1680)
        120,  # INFATUATE: Basic_CheckCannotAttract (:202); no Expert_Main entry
        167,  # STATUS_BURN: Basic_CheckCannotBurn (:231); no Expert_Main entry
        # E1 batch 7: hits whose only label is in Expert_Main (docs/ai/kaizo_roark.md 5.12).
        17,  # BYPASS_ACCURACY: Expert_BypassAccuracyMove (:1640), :2271
        43,  # HIGH_CRITICAL: Expert_HighCritical (:1660), :2836, :2839; Risky :6530
        48,  # RECOIL_QUARTER: Expert_RecoilMove (:1661), no roll
        70,  # LOWER_SPEED_HIT: Expert_SpeedDownOnHit (:1681) -> Expert_StatusSpeedDown, :2362
        169,  # DOUBLE_POWER_WHEN_STATUSED (Facade): Expert_Facade (:1729), no roll
        171,  # DOUBLE_POWER_AND_CURE_PARALYSIS (SmellingSalt): Expert_SmellingSalts (:1731)
        198,  # RECOIL_THIRD: Expert_RecoilMove (:1747), no roll
        221,  # DOUBLE_POWER_WHEN_BELOW_HALF (Brine): Expert_Brine (:1765), :4882
        262,  # RECOIL_PARALYZE_HIT: Expert_RecoilMove (:1802), no roll
        # E2 batch 10 (B10-1).
        253,  # RECOIL_BURN_HIT (Flare Blitz): Expert_RecoilMove (:1796), no roll
        # E2 batch 10: the charge-turn moves. Basic_CheckNonStandardDamageOrChargeTurn (:158,
        # :219) for the first and third; Expert_ChargeTurnNoInvuln (:3933-3973, no roll),
        # Expert_ChargeTurnWithInvuln (:3992) and Expert_ShadowForce (:4014), :4058.
        39,  # CHARGE_TURN_HIGH_CRIT (Razor Wind): :158; Expert_ChargeTurnNoInvuln (:1657)
        75,  # CHARGE_TURN_HIGH_CRIT_FLINCH (Sky Attack): Expert_ChargeTurnNoInvuln (:1682)
        145,  # CHARGE_TURN_DEF_UP (Skull Bash): :219; Expert_ChargeTurnNoInvuln (:1715)
        151,  # SKIP_CHARGE_TURN_IN_SUN (SolarBeam): Expert_ChargeTurnNoInvuln (:1716)
        155,  # FLY: Expert_ChargeTurnWithInvuln (:1721), :4058
        255,  # DIVE: Expert_ChargeTurnWithInvuln (:1797), :4058
        256,  # DIG: Expert_ChargeTurnWithInvuln (:1798), :4058
        263,  # BOUNCE: Expert_ChargeTurnWithInvuln (:1803), :4058
        272,  # SHADOW_FORCE: Expert_ShadowForce (:1808), :4058
        # E2 batch 10: no script names them (no Basic_Main, Expert_Main or other entry).
        146,  # FLINCH_DOUBLE_DAMAGE_FLY_OR_BOUNCE (Twister)
        207,  # HIT_FLY (Sky Uppercut)
        # E1 batch 9 (docs/ai/kaizo_roark.md 5.13).
        98,  # KO_MON_THAT_DEFEATED_USER: Expert_DestinyBond (:1694), :3346, :3351, :3356
        112,  # SET_SPIKES: Basic_CheckSpikes (:197); Expert_Spikes (:1702), :3595, :3602
        208,  # ATK_DEF_UP: Basic_CheckBulkUp (:248); Expert_StatusDefenseUp (:1753)
        211,  # SP_ATK_SP_DEF_UP: Basic_CheckCalmMind (:250); Expert_StatusSpDefenseUp (:1756)
        212,  # ATK_SPD_UP: Basic_CheckDragonDance (:251); Expert_DragonDance (:1757), :4771, :4776
        249,  # TOXIC_SPIKES: Basic_CheckToxicSpikes (:277); Expert_ToxicSpikes (:1792), :6018, :6025
        # E1 batch 10: the stat drops Kaizo moves carry but accuracy's (docs/ai/kaizo_roark.md
        # 5.14), and Reflect.
        18,  # ATK_DOWN: Basic_CheckLowStatStage_Attack (:144); Expert_StatusAttackDown (:1641)
        20,  # SPEED_DOWN: Basic_CheckLowStatStage_Speed (:146); Expert_StatusSpeedDown (:1643)
        24,  # EVA_DOWN: Basic_CheckLowStatStage_Evasion (:150); Expert_StatusEvasionDown (:1647)
        58,  # ATK_DOWN_2: :171; Expert_StatusAttackDown (:1670)
        60,  # SPEED_DOWN_2: :173; Expert_StatusSpeedDown (:1672)
        62,  # SP_DEF_DOWN_2: Basic_CheckLowStatStage_SpDefense (:175); Expert_StatusSpDefenseDown (:1674)
        65,  # SET_REFLECT: Basic_CheckAlreadyUnderReflect (:178); Expert_Reflect (:1677)
        # E1 batch 11 (docs/ai/kaizo_roark.md 5.15): their Expert labels compare for
        # COMPARE_SPEED_FASTER, whose tie is a read (E1 Ruling R8).
        225,  # DOUBLE_SPEED_3_TURNS: Basic_CheckTailwind (:259); Expert_Tailwind (:1768), :4992, :4996
        230,  # DOUBLE_POWER_IF_MOVING_SECOND: Expert_Payback (:1773), :5179
        # E1 batch 12 (docs/ai/kaizo_roark.md 5.16).
        40,  # HALVE_HP: Basic_CheckNonStandardDamageOrChargeTurn (:159); Expert_SuperFang (:1658)
        89,  # COUNTER: Basic_CheckNonStandard... (:187); Expert_Counter (:1688)
        92,  # DAMAGE_WHILE_ASLEEP: Basic_CheckAttackerAsleep (:189); Expert_Nightmare (:1691)
        97,  # USE_RANDOM_LEARNED_MOVE_SLEEP: Basic_CheckAttackerAsleep (:191); Expert_SleepTalk
        99,  # INCREASE_POWER_WITH_LESS_HP: Basic_CheckNonStandard... (:192); Expert_Reversal
        136,  # WEATHER_RAIN: Basic_CheckRainDance (:214); Expert_RainDance (:1710)
        144,  # MIRROR_COAT: Basic_CheckNonStandard... (:218); Expert_MirrorCoat (:1714)
        158,  # ALWAYS_FLINCH_FIRST_TURN_ONLY: Basic_CheckFirstTurnInBattle (:224); Expert_FakeOut
        164,  # WEATHER_HAIL: Basic_CheckHail (:228); Expert_Hail (:1726)
        182,  # LOWER_OWN_ATK_AND_DEF: Basic_CheckNonStandard... (:237); Expert_Superpower
        224,  # EAT_BERRY: Expert_Pluck (:1767)
        111,  # PROTECT: Expert_Protect (:1701), on the da1d60f state fields
        # E1 batch 14: Basic_CheckNonStandardDamageOrChargeTurn only (no Expert_Main entry);
        # their AI damage: engine.damage.ai_case_power / ai_case_fixed or the default.
        41,  # 40_DAMAGE_FLAT (:160)
        87,  # LEVEL_DAMAGE_FLAT (:185)
        88,  # RANDOM_DAMAGE_1_TO_150_LEVEL (:186; Psywave's id case draws, ruling G-39)
        121,  # POWER_BASED_ON_FRIENDSHIP (:203)
        122,  # RANDOM_POWER_MAYBE_HEAL (:204)
        123,  # POWER_BASED_ON_LOW_FRIENDSHIP (:205)
        130,  # 20_DAMAGE_FLAT (:209)
        135,  # RANDOM_POWER_BASED_ON_IVS (:213): Hidden Power, its AI type ai_move_type
        196,  # INCREASE_POWER_WITH_WEIGHT (:244)
        38,  # ONE_HIT_KO: Basic_CheckOHKOWouldFail (:157); Expert_OHKOMove (:1656), :2794
        228,  # SWITCH_HIT: Expert_UTurn (:1771); no Basic entry (E1 batch 15)
        # E1 batch 16 (docs/ai/kaizo_roark.md 5.19)
        91,  # AVERAGE_HP: Expert_PainSplit (:1690); no Basic entry
        124,  # PREVENT_STATUS: Basic_CheckAlreadyUnderSafeguard (:206); no Expert entry
        137,  # WEATHER_SUN: Basic_CheckSunnyDay (:215); Expert_SunnyDay (:1711)
        142,  # MAX_ATK_LOSE_HALF_MAX_HP: Basic_CheckBellyDrum (:216); Expert_BellyDrum (:1712)
        170,  # HIT_LAST_WHIFF_IF_HIT: NonStandard (:233); Expert_FocusPunch (:1730), :6366
        189,  # SET_HP_EQUAL_TO_USER: NonStandard (:241); Expert_Endeavor (:1741)
        260,  # BLIZZARD: Expert_Blizzard (:1801); no Basic entry
        # E2 batch 11 (B11-4)
        90,  # ENCORE: Basic_CheckCannotEncore (:188); Expert_Encore (:1689), :3192
        # E1 batch 17 (docs/ai/kaizo_roark.md 5.20)
        8,  # RECOVER_DAMAGE_SLEEP: Basic_CheckDreamEater (:136); Expert_DreamEater (:1631)
        28,  # FORCE_SWITCH: Basic_CheckCanForceSwitch (:153); Expert_ForceSwitch (:1650)
        79,  # SET_SUBSTITUTE: Basic_CheckCannotSubstitute (:181); Expert_Substitute (:1684)
        # Roost: Recover's labels. turnFlags.roosting is cleared before command selection
        # (BattleSystem_SetupNextTurn, battle_lib.c:2229-2231), so scoring never sees it set.
        214,  # HEAL_HALF_REMOVE_FLYING_TYPE: Basic_CheckCanRecoverHP (:253); Expert_Recovery
        # E1 prep batch P1, on E2's state (ingrain, mean_look, imprison/ed, ai_item_known,
        # future_sight; docs/ai/kaizo_roark.md 5.21)
        23,  # ACC_DOWN: Basic_CheckLowStatStage_Accuracy (:149); Expert_StatusAccuracyDown
        106,  # PREVENT_ESCAPE: Basic_CheckMeanLook (:193); Expert_BindingMove (:1698)
        108,  # EVA_UP_2_MINIMIZE: Basic_CheckHighStatStage_Evasion (:195); Expert_StatusEvasionUp
        148,  # HIT_IN_3_TURNS: Basic_CheckFutureSight (:220); EvalAttack :6396 kill bonus
        188,  # REMOVE_HELD_ITEM: Basic_CheckCanRemoveItem (:240); Expert_KnockOff (:1740)
        192,  # MAKE_SHARED_MOVES_UNUSEABLE: Basic_CheckCanImprison (:242); Expert_Imprison
        227,  # METAL_BURST: Basic_CheckMetalBurst (:261); Expert_MetalBurst (:1770)
        # E1 prep batch P2, on E2's state (last_move, Copycat's call)
        242,  # USE_LAST_USED_MOVE: Basic_CheckCopycat (:271); Expert_Copycat (:1785)
        181,  # GROUND_TRAP_USER_CONTINUOUS_HEAL: Basic_CheckAlreadyIngrained (:236); Expert_Ingrain
        127,  # PASS_STATS_AND_STATUS: Basic_CheckBatonPass (:208); Expert_BatonPass (:1705)
        # E1 prep batch P4 (Gardenia)
        176,  # BOOST_ALLY_POWER_BY_50_PERCENT: Basic_CheckHelpingHand (:234); no Expert entry
        # E2 batch 12, the stat family (B12-11).
        25,  # RESET_STAT_CHANGES: Basic_CheckStatStageImbalance (:151); Expert_Haze (:1648)
        47,  # CRIT_UP_2: Basic_CheckAlreadyPumpedUp (:162); no Expert entry
        46,  # PREVENT_STAT_REDUCTION: Basic_CheckAlreadyUnderMist (:161); no Expert (B12-23)
        143,  # COPY_STAT_CHANGES: Basic_CheckStatStageImbalance (:217); Expert_PsychUp (:1713)
        166,  # SP_ATK_UP_CAUSE_CONFUSION: Basic_CheckCannotConfuse (:230); Expert_Flatter
        205,  # ATK_DEF_DOWN: Basic_CheckTickle (:246); Expert_StatusDefenseDown (:1751)
        206,  # DEF_SPD_UP: Basic_CheckCosmicPower (:247); Expert_StatusSpDefenseUp (:1752)
        226,  # RANDOM_STAT_UP_2: Basic_CheckAcupressure (:260); Expert_Acupressure (:1769)
        238,  # SWAP_ATK_DEF: Basic_CheckPowerTrick (:268); Expert_PowerTrick (:1781)
        243,  # SWAP_ATK_SP_ATK_STAT_CHANGES: Basic_CheckPowerSwap (:272); Expert_PowerSwap
        244,  # SWAP_DEF_SP_DEF_STAT_CHANGES: Basic_CheckGuardSwap (:273); Expert_GuardSwap
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER: Basic_CheckCaptivate (:283); Expert_Captivate
        # E2 batch 12, the healing family (B12-18).
        37,  # REST: no Basic entry; Expert_Rest (:1655), :2766, :2775, :2783, :2786
        102,  # CURE_PARTY_STATUS: no Basic entry; Expert_HealBell (:1696)
        132,  # HEAL_HALF_MORE_IN_SUN: Basic_CheckCanRecoverHP (:210); Expert_Synthesis (:1707)
        179,  # HEAL_IN_3_TURNS: no Basic_ScoreMoveEffect or Expert_Main entry
        193,  # HEAL_STATUS: Basic_CheckCanRefreshStatus (:243); Expert_Refresh (:1745)
        234,  # TRANSFER_STATUS: Basic_CheckCanPsychoShift (:264); Expert_PsychoShift, :5301
        # E1 trial batch (EVA_UP shares both labels with EVA_UP_2_MINIMIZE, 108).
        16,  # EVA_UP: Basic_CheckHighStatStage_Evasion (:143); Expert_StatusEvasionUp (:1639)
        114,  # ALL_FAINT_3_TURNS: Basic_CheckPerishSong (:199); no Expert entry
        173,  # NATURE_POWER: no Basic or Expert entry; Harrassment_Effects :8058
        116,  # SURVIVE_WITH_1_HP: no Basic entry; Expert_Endure (:1704), :3645
        252,  # GIVE_GROUND_IMMUNITY: Basic_CheckMagnetRise (:280); Expert_MagnetRise (:1795), :6113
        215,  # GRAVITY: Basic_CheckGravityActive (:254); Expert_Gravity (:1759), :4795, :4799
        82,  # COPY_MOVE_FOR_BATTLE (Mimic): no Basic or Expert entry, no other script.s mention
        57,  # TRANSFORM: no Basic or Expert entry, no script.s mention
        267,  # CHATTER: no Basic or Expert entry; Basic_CheckSoundproof :131 by move id
        # E3-B2-3. No Basic_ScoreMoveEffect (:133-286), Expert_Main (:1623-1811) or other flag
        # entry names it: its move takes the generic paths (the tables that list it, Expert_
        # Encore's :3223 and the CheckHP ones :7798, :7847, :7958, are transcribed).
        93,  # CONVERSION2
        # E3-B2-4. Named only by the tables Harrassment_Effects (script.s:8051) and CheckHP_Target_
        # DiscourageAtLowHP (:7960), both transcribed (commands.py): no Basic_ScoreMoveEffect or
        # Expert_Main entry, so its move takes the generic paths.
        100,  # DECREASE_LAST_MOVE_PP (Spite)
        # E3-B2-9. Named only by the tables Expert_Encore_EncouragedMoveEffects (script.s:3259) and
        # CheckHP_DiscourageAtHighHP (:7757), both transcribed (commands.py): no Basic_ScoreMoveEffect
        # or Expert_Main entry, no roll, and no AI command reads the bit.
        194,  # REMOVE_ALL_PP_ON_DEFEAT (Grudge)
        # E2 batch 13 (B13-23). No Basic_ScoreMoveEffect (:133-286), Expert_Main
        # (:1623-1811) or other flag entry names these: their moves take the generic paths.
        77,  # POISON_MULTI_HIT (Twineedle)
        104,  # HIT_THREE_TIMES (Triple Kick; Kaizo Rock Wrecker)
        141,  # UNUSED_141 (Kaizo Hurricane, be_seq 141)
        154,  # BEAT_UP
        203,  # CHANGE_TYPE_WITH_WEATHER (Weather Ball; its AI type, B13-26)
        268,  # JUDGEMENT (its AI damage and type cases, B13-25..26)
        # Expert_Main entries only (CheckHP's attacker table :7853 names 190, Risky's :6556
        # names 219, Expert_LuckyChant :5504-5505 names 200 and 209 as the player's moves).
        190,  # DECREASE_POWER_WITH_LESS_USER_HP: Expert_WaterSpout (:1742), no roll
        200,  # HIGH_CRITICAL_BURN_HIT: Expert_HighCritical (:1748), :2836, :2839
        209,  # HIGH_CRITICAL_POISON_HIT: Expert_HighCritical (:1754), :2836, :2839
        217,  # DOUBLE_POWER_HEAL_SLEEP: Expert_WakeUpSlap (:1761), no roll
        231,  # DOUBLE_POWER_IF_TARGET_HIT: Expert_Assurance (:1774), :5204, :5212
        # Basic and Expert entries.
        184,  # RECYCLE: Basic_CheckCanRecycle (:238); Expert_Recycle (:1737), :4526
        219,  # POWER_BASED_ON_LOW_SPEED: NonStandard (:256); Expert_GyroBall (:1763), no roll
        222,  # NATURAL_GIFT: Basic_CheckNaturalGift (:258); no Expert entry
        235,  # HIGHER_POWER_WHEN_LOW_PP: NonStandard (:265); Expert_TrumpCard (:1778), :5333,
        # :5347
        245,  # INCREASE_POWER_WITH_MORE_STAT_UP: NonStandard (:274); Expert_Punishment (:1788)
        232,  # PREVENT_ITEM_USE: Basic_CheckEmbargo (:262); Expert_Embargo (:1775), :5225 (B13-32)
        # E3 batch 1.
        71,  # LOWER_SP_ATK_HIT (Mist Ball, Mystical Fire): no Basic or Expert entry (E3-1-1)
        85,  # DO_NOTHING (Kaizo Thief): no Basic or Expert entry; Encore table :3220 (E3-1-2)
        86,  # DISABLE: Basic_CheckCannotDisable (:184); Expert_Disable (:1687), :3089 (E3-1-3)
        153,  # FLEE_FROM_WILD_BATTLE (Teleport): Basic :221 ScoreMinus10 (E3-1-4)
        218,  # SPEED_DOWN_HIT (Hammer Arm): Expert_HammerArm (:1762), no roll (E3-1-5)
        # E3 batch 1 round 2.
        165,  # TORMENT: Basic_CheckTorment (:229); no Expert entry; tables :6480, :8055 (E3-2-1)
        241,  # USE_MOVE_FIRST (Me First): no Basic entry; Expert_MeFirst (:1784), :5532, :5538, :5542;
        # Risky :6560 (E3-2-3)
        247,  # SET_ABILITY_TO_INSOMNIA (Worry Seed): Basic_CheckWorrySeed (:276); Expert_WorrySeed
        # (:1790), :5984, :5988 (E3-2-2)
        259,  # TRICK_ROOM: Basic_CheckTrickRoom (:282); Expert_TrickRoom (:1800), :6210 (E3-2-4)
        191,  # SWITCH_ABILITIES (Skill Swap): Expert_ChangeUserAbility (:1743), :4429 (E3-2-5)
        178,  # COPY_ABILITY (Role Play): Expert_ChangeUserAbility (:1733), :4429 (E3-B2-5)
        183,  # APPLY_MAGIC_COAT: no Basic entry; Expert_MagicCoat (:1736), :4499, :4505, :4511
        # (E3-B2-6)
        240,  # PREVENT_CRITS (Lucky Chant): Basic_CheckLuckyChant (:270); Expert_LuckyChant
        # (:1783), :5506 (E3-B2-7)
        233,  # FLING: Basic_CheckFling (:263); Expert_Fling (:1776), :5257, :5268, :5276
        # (E3-B2-1)
        9,  # COPY_MOVE (Mirror Move): no Basic entry; Expert_MirrorMove (:1632), :1915, :1922;
        # Risky :6539 (E3-B2-2)
        239,  # SUPRESS_ABILITY (Gastro Acid): Basic_CheckGastroAcid (:269); Expert_GastroAcid
        # (:1782), :5481, :5484 (E3-2-6)
    }
)
# The transcribed effects (KAIZO_TRANSCRIBED) whose Expert_Main entry goes to a label added
# after the vanilla transcription: effect -> (label, the script.s line of its entry in
# Expert_Main's dispatch table, :1628-1808). The other transcribed effects either have no
# entry (they end at :1811) or go to a vanilla-transcribed label (DEF_DOWN :1642, DEF_DOWN_2
# :1671, HIT_BEFORE_SWITCH :1706, STEALTH_ROCK :1805). A per-slot layout refuses a move whose
# effect is here but not in its expert_transcribed set. The workflow for a future effect with
# an Expert_Main entry: add it here first (and to KAIZO_TRANSCRIBED). It is then
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
    # E1 batch 4: the stat raises (SPEED_UP_2's Expert_StatusSpeedUp is above).
    10: ("Expert_StatusAttackUp", 1633),
    11: ("Expert_StatusDefenseUp", 1634),
    13: ("Expert_StatusSpAttackUp", 1636),
    50: ("Expert_StatusAttackUp", 1663),
    51: ("Expert_StatusDefenseUp", 1664),
    53: ("Expert_StatusSpAttackUp", 1666),
    54: ("Expert_StatusSpDefenseUp", 1667),
    # E1 batch 6: the status moves.
    1: ("Expert_StatusSleep", 1628),
    49: ("Expert_StatusConfuse", 1662),
    67: ("Expert_StatusParalyze", 1679),
    84: ("Expert_ToxicLeechSeed", 1686),
    118: ("Expert_Swagger", 1680),
    # E1 batch 7: the Expert-only hits.
    17: ("Expert_BypassAccuracyMove", 1640),
    43: ("Expert_HighCritical", 1660),
    48: ("Expert_RecoilMove", 1661),
    70: ("Expert_SpeedDownOnHit", 1681),
    169: ("Expert_Facade", 1729),
    171: ("Expert_SmellingSalts", 1731),
    198: ("Expert_RecoilMove", 1747),
    221: ("Expert_Brine", 1765),
    262: ("Expert_RecoilMove", 1802),
    253: ("Expert_RecoilMove", 1796),  # E2 batch 10: RECOIL_BURN_HIT (Flare Blitz)
    # E2 batch 10: the charge-turn moves.
    39: ("Expert_ChargeTurnNoInvuln", 1657),
    75: ("Expert_ChargeTurnNoInvuln", 1682),
    145: ("Expert_ChargeTurnNoInvuln", 1715),
    151: ("Expert_ChargeTurnNoInvuln", 1716),
    155: ("Expert_ChargeTurnWithInvuln", 1721),
    255: ("Expert_ChargeTurnWithInvuln", 1797),
    256: ("Expert_ChargeTurnWithInvuln", 1798),
    263: ("Expert_ChargeTurnWithInvuln", 1803),
    272: ("Expert_ShadowForce", 1808),
    # E1 batch 9.
    98: ("Expert_DestinyBond", 1694),
    112: ("Expert_Spikes", 1702),
    208: ("Expert_StatusDefenseUp", 1753),
    211: ("Expert_StatusSpDefenseUp", 1756),
    212: ("Expert_DragonDance", 1757),
    249: ("Expert_ToxicSpikes", 1792),
    # E1 batch 10.
    18: ("Expert_StatusAttackDown", 1641),
    20: ("Expert_StatusSpeedDown", 1643),
    24: ("Expert_StatusEvasionDown", 1647),
    58: ("Expert_StatusAttackDown", 1670),
    60: ("Expert_StatusSpeedDown", 1672),
    62: ("Expert_StatusSpDefenseDown", 1674),
    65: ("Expert_Reflect", 1677),
    # E1 batch 11.
    225: ("Expert_Tailwind", 1768),
    230: ("Expert_Payback", 1773),
    # E1 batch 12.
    40: ("Expert_SuperFang", 1658),
    89: ("Expert_Counter", 1688),
    92: ("Expert_Nightmare", 1691),
    97: ("Expert_SleepTalk", 1693),
    99: ("Expert_Reversal", 1695),
    136: ("Expert_RainDance", 1710),
    144: ("Expert_MirrorCoat", 1714),
    158: ("Expert_FakeOut", 1723),
    164: ("Expert_Hail", 1726),
    182: ("Expert_Superpower", 1735),
    224: ("Expert_Pluck", 1767),
    111: ("Expert_Protect", 1701),
    38: ("Expert_OHKOMove", 1656),
    228: ("Expert_UTurn", 1771),
    91: ("Expert_PainSplit", 1690),
    137: ("Expert_SunnyDay", 1711),
    142: ("Expert_BellyDrum", 1712),
    170: ("Expert_FocusPunch", 1730),
    189: ("Expert_Endeavor", 1741),
    260: ("Expert_Blizzard", 1801),
    90: ("Expert_Encore", 1689),
    8: ("Expert_DreamEater", 1631),
    28: ("Expert_ForceSwitch", 1650),
    79: ("Expert_Substitute", 1684),
    214: ("Expert_Recovery", 1758),
    23: ("Expert_StatusAccuracyDown", 1646),
    106: ("Expert_BindingMove", 1698),
    108: ("Expert_StatusEvasionUp", 1699),
    188: ("Expert_KnockOff", 1740),
    192: ("Expert_Imprison", 1744),
    227: ("Expert_MetalBurst", 1770),
    242: ("Expert_Copycat", 1785),
    181: ("Expert_Ingrain", 1734),
    127: ("Expert_BatonPass", 1705),
    # E2 batch 12 (B12-11).
    25: ("Expert_Haze", 1648),
    143: ("Expert_PsychUp", 1713),
    166: ("Expert_Flatter", 1727),
    205: ("Expert_StatusDefenseDown", 1751),
    206: ("Expert_StatusSpDefenseUp", 1752),
    226: ("Expert_Acupressure", 1769),
    238: ("Expert_PowerTrick", 1781),
    243: ("Expert_PowerSwap", 1786),
    244: ("Expert_GuardSwap", 1787),
    265: ("Expert_Captivate", 1804),
    37: ("Expert_Rest", 1655),
    102: ("Expert_HealBell", 1696),
    132: ("Expert_Synthesis", 1707),
    193: ("Expert_Refresh", 1745),
    234: ("Expert_PsychoShift", 1777),
    16: ("Expert_StatusEvasionUp", 1639),
    116: ("Expert_Endure", 1704),
    252: ("Expert_MagnetRise", 1795),
    215: ("Expert_Gravity", 1759),
    # E2 batch 13 (B13-23).
    184: ("Expert_Recycle", 1737),
    190: ("Expert_WaterSpout", 1742),
    200: ("Expert_HighCritical", 1748),
    209: ("Expert_HighCritical", 1754),
    217: ("Expert_WakeUpSlap", 1761),
    219: ("Expert_GyroBall", 1763),
    232: ("Expert_Embargo", 1775),  # B13-32
    231: ("Expert_Assurance", 1774),
    235: ("Expert_TrumpCard", 1778),
    245: ("Expert_Punishment", 1788),
    # E3 batch 1.
    86: ("Expert_Disable", 1687),
    218: ("Expert_HammerArm", 1762),
    # E3 batch 1 round 2.
    241: ("Expert_MeFirst", 1784),
    247: ("Expert_WorrySeed", 1790),
    239: ("Expert_GastroAcid", 1782),
    259: ("Expert_TrickRoom", 1800),
    191: ("Expert_ChangeUserAbility", 1743),
    178: ("Expert_ChangeUserAbility", 1733),
    233: ("Expert_Fling", 1776),
    9: ("Expert_MirrorMove", 1632),
    183: ("Expert_MagicCoat", 1736),
    240: ("Expert_LuckyChant", 1783),
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
        10,  # ATK_UP: Expert_StatusAttackUp
        11,  # DEF_UP: Expert_StatusDefenseUp
        13,  # SP_ATK_UP: Expert_StatusSpAttackUp
        50,  # ATK_UP_2: Expert_StatusAttackUp
        51,  # DEF_UP_2: Expert_StatusDefenseUp
        53,  # SP_ATK_UP_2: Expert_StatusSpAttackUp
        54,  # SP_DEF_UP_2: Expert_StatusSpDefenseUp
        1,  # STATUS_SLEEP: Expert_StatusSleep
        49,  # STATUS_CONFUSE: Expert_StatusConfuse
        67,  # STATUS_PARALYZE: Expert_StatusParalyze
        84,  # STATUS_LEECH_SEED: Expert_ToxicLeechSeed
        118,  # ATK_UP_2_STATUS_CONFUSION: Expert_Swagger
        17,  # BYPASS_ACCURACY: Expert_BypassAccuracyMove
        43,  # HIGH_CRITICAL: Expert_HighCritical
        48,  # RECOIL_QUARTER: Expert_RecoilMove
        70,  # LOWER_SPEED_HIT: Expert_SpeedDownOnHit
        169,  # DOUBLE_POWER_WHEN_STATUSED: Expert_Facade
        171,  # DOUBLE_POWER_AND_CURE_PARALYSIS: Expert_SmellingSalts
        198,  # RECOIL_THIRD: Expert_RecoilMove
        221,  # DOUBLE_POWER_WHEN_BELOW_HALF: Expert_Brine
        262,  # RECOIL_PARALYZE_HIT: Expert_RecoilMove
        253,  # RECOIL_BURN_HIT: Expert_RecoilMove (E2 batch 10)
        39,  # CHARGE_TURN_HIGH_CRIT: Expert_ChargeTurnNoInvuln (E2 batch 10)
        75,  # CHARGE_TURN_HIGH_CRIT_FLINCH: Expert_ChargeTurnNoInvuln
        145,  # CHARGE_TURN_DEF_UP: Expert_ChargeTurnNoInvuln
        151,  # SKIP_CHARGE_TURN_IN_SUN: Expert_ChargeTurnNoInvuln
        155,  # FLY: Expert_ChargeTurnWithInvuln
        255,  # DIVE: Expert_ChargeTurnWithInvuln
        256,  # DIG: Expert_ChargeTurnWithInvuln
        263,  # BOUNCE: Expert_ChargeTurnWithInvuln
        272,  # SHADOW_FORCE: Expert_ShadowForce
        98,  # KO_MON_THAT_DEFEATED_USER: Expert_DestinyBond
        112,  # SET_SPIKES: Expert_Spikes
        208,  # ATK_DEF_UP: Expert_StatusDefenseUp
        211,  # SP_ATK_SP_DEF_UP: Expert_StatusSpDefenseUp
        212,  # ATK_SPD_UP: Expert_DragonDance
        249,  # TOXIC_SPIKES: Expert_ToxicSpikes
        18,  # ATK_DOWN: Expert_StatusAttackDown
        20,  # SPEED_DOWN: Expert_StatusSpeedDown
        24,  # EVA_DOWN: Expert_StatusEvasionDown
        58,  # ATK_DOWN_2: Expert_StatusAttackDown
        60,  # SPEED_DOWN_2: Expert_StatusSpeedDown
        62,  # SP_DEF_DOWN_2: Expert_StatusSpDefenseDown
        65,  # SET_REFLECT: Expert_Reflect
        225,  # DOUBLE_SPEED_3_TURNS: Expert_Tailwind
        230,  # DOUBLE_POWER_IF_MOVING_SECOND: Expert_Payback
        40,  # HALVE_HP: Expert_SuperFang
        89,  # COUNTER: Expert_Counter
        92,  # DAMAGE_WHILE_ASLEEP: Expert_Nightmare
        97,  # USE_RANDOM_LEARNED_MOVE_SLEEP: Expert_SleepTalk
        99,  # INCREASE_POWER_WITH_LESS_HP: Expert_Reversal
        136,  # WEATHER_RAIN: Expert_RainDance
        144,  # MIRROR_COAT: Expert_MirrorCoat
        158,  # ALWAYS_FLINCH_FIRST_TURN_ONLY: Expert_FakeOut
        164,  # WEATHER_HAIL: Expert_Hail
        182,  # LOWER_OWN_ATK_AND_DEF: Expert_Superpower
        224,  # EAT_BERRY: Expert_Pluck
        111,  # PROTECT: Expert_Protect
        38,  # ONE_HIT_KO: Expert_OHKOMove
        228,  # SWITCH_HIT: Expert_UTurn
        91,  # AVERAGE_HP: Expert_PainSplit
        137,  # WEATHER_SUN: Expert_SunnyDay
        142,  # MAX_ATK_LOSE_HALF_MAX_HP: Expert_BellyDrum
        170,  # HIT_LAST_WHIFF_IF_HIT: Expert_FocusPunch
        189,  # SET_HP_EQUAL_TO_USER: Expert_Endeavor
        260,  # BLIZZARD: Expert_Blizzard
        90,  # ENCORE: Expert_Encore (E2 batch 11)
        8,  # RECOVER_DAMAGE_SLEEP: Expert_DreamEater
        28,  # FORCE_SWITCH: Expert_ForceSwitch
        79,  # SET_SUBSTITUTE: Expert_Substitute
        214,  # HEAL_HALF_REMOVE_FLYING_TYPE: Expert_Recovery
        23,  # ACC_DOWN: Expert_StatusAccuracyDown
        106,  # PREVENT_ESCAPE: Expert_BindingMove
        108,  # EVA_UP_2_MINIMIZE: Expert_StatusEvasionUp
        188,  # REMOVE_HELD_ITEM: Expert_KnockOff
        192,  # MAKE_SHARED_MOVES_UNUSEABLE: Expert_Imprison
        227,  # METAL_BURST: Expert_MetalBurst
        242,  # USE_LAST_USED_MOVE: Expert_Copycat
        181,  # GROUND_TRAP_USER_CONTINUOUS_HEAL: Expert_Ingrain (no score change)
        127,  # PASS_STATS_AND_STATUS: Expert_BatonPass
        # E2 batch 12 (B12-11).
        25,  # RESET_STAT_CHANGES: Expert_Haze
        143,  # COPY_STAT_CHANGES: Expert_PsychUp
        166,  # SP_ATK_UP_CAUSE_CONFUSION: Expert_Flatter
        205,  # ATK_DEF_DOWN: Expert_StatusDefenseDown
        206,  # DEF_SPD_UP: Expert_StatusSpDefenseUp
        226,  # RANDOM_STAT_UP_2: Expert_Acupressure
        238,  # SWAP_ATK_DEF: Expert_PowerTrick
        243,  # SWAP_ATK_SP_ATK_STAT_CHANGES: Expert_PowerSwap
        244,  # SWAP_DEF_SP_DEF_STAT_CHANGES: Expert_GuardSwap
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER: Expert_Captivate
        37,  # REST: Expert_Rest (B12-18)
        102,  # CURE_PARTY_STATUS: Expert_HealBell
        132,  # HEAL_HALF_MORE_IN_SUN: Expert_Synthesis
        193,  # HEAL_STATUS: Expert_Refresh
        234,  # TRANSFER_STATUS: Expert_PsychoShift
        16,  # EVA_UP: Expert_StatusEvasionUp
        116,  # SURVIVE_WITH_1_HP: Expert_Endure
        252,  # GIVE_GROUND_IMMUNITY: Expert_MagnetRise
        215,  # GRAVITY: Expert_Gravity
        # E2 batch 13 (B13-23).
        184,  # RECYCLE: Expert_Recycle
        190,  # DECREASE_POWER_WITH_LESS_USER_HP: Expert_WaterSpout
        200,  # HIGH_CRITICAL_BURN_HIT: Expert_HighCritical
        209,  # HIGH_CRITICAL_POISON_HIT: Expert_HighCritical
        217,  # DOUBLE_POWER_HEAL_SLEEP: Expert_WakeUpSlap
        219,  # POWER_BASED_ON_LOW_SPEED: Expert_GyroBall (no score change)
        231,  # DOUBLE_POWER_IF_TARGET_HIT: Expert_Assurance
        232,  # PREVENT_ITEM_USE: Expert_Embargo (B13-32)
        235,  # HIGHER_POWER_WHEN_LOW_PP: Expert_TrumpCard
        245,  # INCREASE_POWER_WITH_MORE_STAT_UP: Expert_Punishment
        86,  # DISABLE: Expert_Disable (E3-1-3)
        218,  # SPEED_DOWN_HIT: Expert_HammerArm (E3-1-5)
        241,  # USE_MOVE_FIRST: Expert_MeFirst (E3-2-3)
        247,  # SET_ABILITY_TO_INSOMNIA: Expert_WorrySeed (E3-2-2)
        239,  # SUPRESS_ABILITY: Expert_GastroAcid (E3-2-6)
        259,  # TRICK_ROOM: Expert_TrickRoom (E3-2-4)
        191,  # SWITCH_ABILITIES: Expert_ChangeUserAbility (E3-2-5)
        178,  # COPY_ABILITY: Expert_ChangeUserAbility (E3-B2-5)
        233,  # FLING: Expert_Fling (E3-B2-1)
        9,  # COPY_MOVE: Expert_MirrorMove (E3-B2-2)
        183,  # APPLY_MAGIC_COAT: Expert_MagicCoat (E3-B2-6)
        240,  # PREVENT_CRITS: Expert_LuckyChant (E3-B2-7)
    }
)
assert set(KAIZO_EXPERT_LABELS) <= KAIZO_TRANSCRIBED
assert KAIZO_EXPERT_TRANSCRIBED <= set(KAIZO_EXPERT_LABELS)

# The read bound of Kaizo Roark's layout (layout.KAIZO): the most random reads one slot's
# path makes for his moves under his flags 0x17, by ai.read_bounds (E1 Ruling R6;
# layout.fight_slot_reads, pinned by tests/ai/test_read_bounds.py): Selfdestruct's :117 and
# :304 guesses, :6374, :1859, :1870, :1875 and Risky's :6530.
KAIZO_SLOT_READS = 7

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
# The u8 draw (RandNext() % 256) a per-slot pattern's bit 0 / 1 stands for: 254 is below no
# IfRandomLessThan N (and even: ability 2), 1 below every one (and odd: ability 1), so the two
# split every read (module docstring).
READ_DRAWS = (254, 1)
assert all(READ_DRAWS[1] < n <= READ_DRAWS[0] for n in SITE_N.values())

# TrainerAI_CalcDamage's draws (module docstring, E1 ruling G-39). calc_draw_kind: 0 no draw,
# CALC_PSYWAVE (MOVE_PSYWAVE's case, trainer_ai.c:3014-3016), CALC_MAGNITUDE (:3028-3048).
CALC_NONE, CALC_PSYWAVE, CALC_MAGNITUDE = 0, 1, 2
CALC_DRAW_CLASSES = AI_DRAW_CLASSES  # 11, Psywave's % 11; Magnitude uses classes 0-6
# Magnitude's buckets: RandNext() % 100 below 5, 15, 35, 65, 85, 95 -> powers 10, 30, 50, 70,
# 90, 110, else 150 (engine.damage.AI_MAGNITUDE_POWER).
MAGNITUDE_BUCKETS = (5, 15, 35, 65, 85, 95)
_U16 = np.arange(65536)
# CALC_DRAW_P[kind, class]: the exact law of a draw's class over a uniform u16, counted.
CALC_DRAW_P = np.stack(
    [
        np.eye(CALC_DRAW_CLASSES)[0],
        np.bincount(_U16 % 11, minlength=CALC_DRAW_CLASSES) / 65536,
        np.bincount(
            np.searchsorted(MAGNITUDE_BUCKETS, _U16 % 100, side="right"),
            minlength=CALC_DRAW_CLASSES,
        )
        / 65536,
    ]
)
assert np.allclose(CALC_DRAW_P.sum(axis=1), 1)
# The most draws one command makes: IfPartyMemberDealsMoreDamage estimates the attacker's four
# moves and four of each other party member (trainer_ai.c:2130-2171), a party of SLOTS.
CALC_DRAWS = MOVES * SLOTS


def calc_draw_kind(move) -> jnp.ndarray:
    """``int32``: the draw ``TrainerAI_CalcDamage`` makes for ``move`` (decomp id): it switches
    on the id (trainer_ai.c:2885), so ``CALC_PSYWAVE`` for 149, ``CALC_MAGNITUDE`` for 222,
    else ``CALC_NONE``."""
    move = jnp.asarray(move, _I32)
    return jnp.where(
        move == AI_PSYWAVE, CALC_PSYWAVE, jnp.where(move == AI_MAGNITUDE, CALC_MAGNITUDE, CALC_NONE)
    ).astype(_I32)


def calc_kind_class(kind, draw) -> jnp.ndarray:
    """``int32``: the class of the raw ``RandNext()`` ``draw`` (a u16) in a case of ``kind``
    (``calc_draw_kind``): ``draw % 11`` for Psywave (trainer_ai.c:3015), the bucket of
    ``draw % 100`` for Magnitude (:3030-3046), 0 for ``CALC_NONE``."""
    draw = jnp.asarray(draw, _I32)
    kind = jnp.asarray(kind, _I32)
    bucket = jnp.searchsorted(jnp.asarray(MAGNITUDE_BUCKETS, _I32), draw % 100, side="right")
    return jnp.where(
        kind == CALC_PSYWAVE, draw % 11, jnp.where(kind == CALC_MAGNITUDE, bucket, 0)
    ).astype(_I32)


def calc_draw_class(move, draw) -> jnp.ndarray:
    """``int32``: ``calc_kind_class`` of ``move``'s case (decomp id)."""
    return calc_kind_class(calc_draw_kind(move), draw)


class AiOutcome(NamedTuple):
    sim_roll: jnp.ndarray  # int8[4], roll index 0-15 per opponent move slot
    # int8[]: player ability slot 0/1 (flat layout); 0 and unread in a per-slot layout
    ability_guess: jnp.ndarray
    # int8[MAX_SCRIPT_ROLLS] (flat layout): 1 = the site's jump is taken; or
    # uint8[MOVES, slot_reads] (per-slot layout): entry k of slot m is the k-th random read on
    # slot m's path, as RandNext() % 256 (module docstring)
    script_rolls: jnp.ndarray
    tiebreak: jnp.ndarray  # int8[], RandNext % 12
    # int8[]: the first stopping switch draw, 0..sites - 1; sites (layout.switch_sites: 9
    # vanilla, 60 Kaizo) = none; -1 refused
    switch_stop: jnp.ndarray
    # uint16[MOVES, slot_reads, CALC_DRAWS] (a sampled per-slot outcome): the raw RandNext()
    # draws of the damage commands among the reads (module docstring); uint16[MOVES,
    # slot_reads, 0] in the enumeration's patterns, where a damage read takes its bit, and
    # uint16[MOVES, 0, 0] in a flat outcome (no_calc_draws: no raw draws)
    calc_draws: jnp.ndarray


def no_calc_draws(lead: tuple[int, ...] = (), slot_reads: int = 0) -> jnp.ndarray:
    """``uint16[*lead, MOVES, slot_reads, 0]``: the ``calc_draws`` that carries no raw draws
    (module docstring): a per-slot layout's enumeration patterns (``slot_reads`` its bound),
    each damage read taking its result from its ``script_rolls`` bit, and a flat outcome
    (``slot_reads`` 0), whose damage commands never draw (``commands._calc_read``)."""
    return jnp.zeros((*lead, MOVES, slot_reads, 0), jnp.uint16)


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
    residue = draws % _SWITCH_MOD[kind]
    stops = (residue >= _SWITCH_LO[kind]) & (residue < _SWITCH_HI[kind])
    first = jnp.where(jnp.any(stops), jnp.argmax(stops), kind.shape[0])
    return jnp.where(jnp.any(kind == SwitchSite.REFUSED), SWITCH_REFUSED, first)


def sample_ai_outcome(
    key, tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> AiOutcome:
    """Draw every field from its decomp probability for this state (the switch draws
    over ``switch_sites(tables, state, layout.switch_sites)``). ``layout``: a per-slot layout
    draws ``script_rolls`` as ``uint8[MOVES, layout.slot_reads]``, each entry a uniform
    ``RandNext() % 256`` (each read compares its own draw: module docstring), and leaves
    ``ability_guess`` 0; the default, and any flat layout, is the vanilla layout. Read column
    i is drawn from its own key (``fold_in(key, i)``), so a key gives the same reads whatever
    the layout's bound: the fight's own layout and ``layout.ALLIN`` see the same draws."""
    sites = SWITCH_SITES if layout is None else layout.switch_sites
    layout_sites = switch_sites(tables, state, sites)
    k = jax.random.split(key, 4)
    draws = jax.random.randint(k[3], (sites,), 0, 65536, dtype=_I32)
    calc = no_calc_draws()
    if layout is not None and layout.per_slot_sites:
        rolls = jax.vmap(
            lambda i: jax.random.randint(jax.random.fold_in(k[1], i), (MOVES,), 0, 256, dtype=_I32)
        )(jnp.arange(layout.slot_reads)).T
        script_rolls, guess = rolls.astype(jnp.uint8), jnp.zeros((), _I8)
        # The damage commands' raw draws, read column i from its own key as above (a key of
        # its own beside the four streams, so they are unchanged).
        k_calc = jax.random.fold_in(key, 0x43414C43)
        calc = jax.vmap(
            lambda i: jax.random.randint(
                jax.random.fold_in(k_calc, i), (MOVES, CALC_DRAWS), 0, 65536, dtype=_I32
            )
        )(jnp.arange(layout.slot_reads))
        calc = jnp.transpose(calc, (1, 0, 2)).astype(jnp.uint16)
    else:
        rolls = jax.random.randint(k[1], (MAX_SCRIPT_ROLLS,), 0, 256, dtype=_I32)
        script_rolls = (rolls < script_roll_thresholds(state)).astype(_I8)
        guess = jax.random.randint(k[0], (), 0, 2, dtype=_I32).astype(_I8)
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, _I8),
        ability_guess=guess,
        script_rolls=script_rolls,
        tiebreak=(jax.random.randint(k[2], (), 0, 65536, dtype=_I32) % TIEBREAK_MOD).astype(_I8),
        switch_stop=_first_stop(draws, layout_sites).astype(_I8),
        calc_draws=calc,
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


def enumerate_ai_outcomes(tables: Tables, state: BattleState) -> tuple[AiOutcome, jnp.ndarray]:
    """Every outcome combination of the flat layout (a batched ``AiOutcome`` with leading
    axis K) and its exact probability ``float32[K]``: K is ``ENUMERATION_WIDTH`` (3840) for
    every state; combinations with an unused script-roll site set to 1, or a ``switch_stop``
    at a ``SwitchSite.NONE`` entry, have probability 0. The switch draws are over
    ``switch_sites(tables, state)``. A per-slot layout's reads take their probabilities from
    the path (module docstring), so its joint enumeration runs the scoring
    (``policy.ai_action_distribution_enumerated``)."""
    return _enumerate_flat(tables, state)


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
        calc_draws=no_calc_draws((ENUMERATION_WIDTH,)),
    )
    return outs, p_rolls * p_switch * p_rest
