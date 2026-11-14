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
    "KAIZO_BASE_READS",
    "KAIZO_EXPERT_LABELS",
    "KAIZO_EXPERT_TRANSCRIBED",
    "KAIZO_FLAG_READS",
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
    "kaizo_slot_reads",
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
        266,  # STEALTH_ROCK: Expert_StealthRock (:1805 -> :6268), :6274
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
        # E1 batch 4: the stat raises (Basic_CheckHighStatStage_*, script.s:343-418;
        # Expert_Status*Up, :1978-2155) of the Kaizo moves, but evasion
        # (docs/ai/kaizo_roark.md 5.10).
        10,  # ATK_UP: Expert_StatusAttackUp (:1633), :1989, :1995, :2001
        11,  # DEF_UP: Expert_StatusDefenseUp (:1634), :2023, :2029, :2034, :2043, :2046
        13,  # SP_ATK_UP: Expert_StatusSpAttackUp (:1636), :2092, :2098, :2104
        50,  # ATK_UP_2: Expert_StatusAttackUp (:1663)
        51,  # DEF_UP_2: Expert_StatusDefenseUp (:1664)
        53,  # SP_ATK_UP_2: Expert_StatusSpAttackUp (:1666)
        54,  # SP_DEF_UP_2: Expert_StatusSpDefenseUp (:1667)
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
    }
)
assert set(KAIZO_EXPERT_LABELS) <= KAIZO_TRANSCRIBED
assert KAIZO_EXPERT_TRANSCRIBED <= set(KAIZO_EXPERT_LABELS)

# The read bound of a per-slot layout (E1 Ruling R5, per flag set): a slot's path makes the
# reads of each flag script that runs, and no flag's path depends on another flag's (scripts
# read the state, not the scores), so the most reads one path makes under a flag set is at
# most the sum of each flag's most. KAIZO_BASE_READS: the most random reads (IfRandomLessThan
# and guessing LoadBattlerAbility DEFENDER) BASIC, EVAL_ATTACK and EXPERT together make on one
# path, found from the data by tests/ai/test_kaizo_outcome.py
# test_slot_reads_fit_every_transcribed_effect: Selfdestruct (HALVE_DEFENSE, a no-calc effect)
# against a guessed player with evasion +4 and Bonsly at 30 % HP or less reads :117 and :304
# (two guesses), :6374, :1859, :1870 and :1875. Any subset of the three makes no more.
KAIZO_BASE_READS = 6
# The most random reads each other transcribed flag's script makes on one path, from the
# script itself: SETUP_FIRST_TURN :6429; RISKY :6530; PRIORITIZE_EXTREMES :6515; CHECK_HP the
# attacker band's :7718, then the target band's :7742; WEATHER none. By flag bit.
KAIZO_FLAG_READS: dict[int, int] = {3: 1, 4: 1, 5: 1, 8: 2, 9: 0}


def kaizo_slot_reads(flags: int) -> int:
    """The per-slot read bound (``AiLayout.slot_reads``) of the Kaizo flag set ``flags``:
    ``KAIZO_BASE_READS`` plus each set bit's ``KAIZO_FLAG_READS`` (Roark's 0x17: 7). A path
    that makes more is refused (``AiContext.missing_read``)."""
    return KAIZO_BASE_READS + sum(n for bit, n in KAIZO_FLAG_READS.items() if flags >> bit & 1)


KAIZO_SLOT_READS = kaizo_slot_reads(0b10111)  # Kaizo Roark's set (layout.KAIZO)

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
    ``ability_guess`` 0; the default, and any flat layout, is the vanilla layout."""
    sites = SWITCH_SITES if layout is None else layout.switch_sites
    layout_sites = switch_sites(tables, state, sites)
    k = jax.random.split(key, 4)
    draws = jax.random.randint(k[3], (sites,), 0, 65536, dtype=_I32)
    if layout is not None and layout.per_slot_sites:
        rolls = jax.random.randint(k[1], (MOVES, layout.slot_reads), 0, 256, dtype=_I32)
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
