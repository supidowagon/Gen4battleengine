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
        # Expert_Status*Up, :1978-2155) of the Kaizo moves, but evasion
        # (docs/ai/kaizo_roark.md 5.10).
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
        88,  # RANDOM_DAMAGE_1_TO_150_LEVEL (:186; Psywave's id itself stays refused)
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
        # E1 batch 17 (docs/ai/kaizo_roark.md 5.20)
        8,  # RECOVER_DAMAGE_SLEEP: Basic_CheckDreamEater (:136); Expert_DreamEater (:1631)
        28,  # FORCE_SWITCH: Basic_CheckCanForceSwitch (:153); Expert_ForceSwitch (:1650)
        79,  # SET_SUBSTITUTE: Basic_CheckCannotSubstitute (:181); Expert_Substitute (:1684)
        # Roost: Recover's labels. turnFlags.roosting is cleared before command selection
        # (BattleSystem_SetupNextTurn, battle_lib.c:2229-2231), so scoring never sees it set.
        214,  # HEAL_HALF_REMOVE_FLYING_TYPE: Basic_CheckCanRecoverHP (:253); Expert_Recovery
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
    8: ("Expert_DreamEater", 1631),
    28: ("Expert_ForceSwitch", 1650),
    79: ("Expert_Substitute", 1684),
    214: ("Expert_Recovery", 1758),
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
        8,  # RECOVER_DAMAGE_SLEEP: Expert_DreamEater
        28,  # FORCE_SWITCH: Expert_ForceSwitch
        79,  # SET_SUBSTITUTE: Expert_Substitute
        214,  # HEAL_HALF_REMOVE_FLYING_TYPE: Expert_Recovery
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
    draws ``script_rolls`` as ``uint8[MOVES, layout.slot_reads]``, each entry a uniform
    ``RandNext() % 256`` (each read compares its own draw: module docstring), and leaves
    ``ability_guess`` 0; the default, and any flat layout, is the vanilla layout. Read column
    i is drawn from its own key (``fold_in(key, i)``), so a key gives the same reads whatever
    the layout's bound: the fight's own layout and ``layout.ALLIN`` see the same draws."""
    sites = SWITCH_SITES if layout is None else layout.switch_sites
    layout_sites = switch_sites(tables, state, sites)
    k = jax.random.split(key, 4)
    draws = jax.random.randint(k[3], (sites,), 0, 65536, dtype=_I32)
    if layout is not None and layout.per_slot_sites:
        rolls = jax.vmap(
            lambda i: jax.random.randint(jax.random.fold_in(k[1], i), (MOVES,), 0, 256, dtype=_I32)
        )(jnp.arange(layout.slot_reads)).T
        script_rolls, guess = rolls.astype(jnp.uint8), jnp.zeros((), _I8)
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
    )
    return outs, p_rolls * p_switch * p_rest
