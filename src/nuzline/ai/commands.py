"""The trainer-AI script commands Roark's flags execute (``docs/ai/script_commands.md``).

One JAX function per command row, each ``(ctx: AiContext, ...) -> AiContext``. The
decomp's interpreter (``TrainerAI_EvalMoves``, trainer_ai.c:483) walks the script with
jumps; here the scripts are straight-line (``nuzline.ai.policy``): every command runs,
and ``AiContext.live`` masks the commands of a path the decomp would not be on. A jump
command only stores its condition in ``AiContext.jump``; the script splits the mask on
it. Commands that write ``calcTemp`` do so on every path, which is safe because the
scripts always reload ``calcTemp`` before reading it and every value written is a
function of the state and the outcome alone.

Paths are relative to ``external/pokeplatinum``; ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``. The AI reads the true battle state except for
the player's moves and ability, which go through its knowledge (roark.md 6, 9):
``AiContext.seen_moves`` and ``AiContext.player_ability``.

Only the script paths of the moves in ``MOVE_EFFECTS`` are transcribed (roark.md 5),
keyed by move id in vanilla and by their battle effects in Kaizo (``nuzline.ai.layout``);
``nuzline.ai.policy.ai_refusals`` refuses an opponent that knows any other move, and
the mechanics ``effectiveness_class`` does not implement.

**The loose oracle** (tests only; ``policy.ai_action_distribution_loose``): with
``AiContext.loose`` set, every ``RandNext`` read of the scripts is its own independent draw,
as in the decomp. Each ``IfRandomLessThan`` (trainer_ai.c:528-538), each
``LoadBattlerAbility DEFENDER`` coin flip (:1191-1198) and each ``IfSpeedCompareEqualTo``
tie (battle_lib.c:1401-1447) takes the next bit of its slot's ``LooseDraws.bits``, in the
static order the straight-line scripts make the reads, and records the bit's probability:
the read's own N/256, or 1/2. Its index counts every read of the straight-line scripts, on
the path or not; the exact per-slot layout counts only the reads on the path (E1 Ruling R4).
It reads neither ``AiContext.sites`` nor ``AiOutcome``'s draws, and it compares the speed
compare's three-valued result, so it shares no gate with the exact layout: no read order, no
read bound, no dropped tie.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.outcome import SITE_LINES, SITE_N, AiOutcome, line_sites
from nuzline.ai.switching import INEFFECTIVE, apply_type_chart, type_chart_flags
from nuzline.data.tables import Tables
from nuzline.engine.actions import usable_moves
from nuzline.engine.damage import ai_damage_for_roll, decomp_abilities
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.order import (
    COMPARE_SPEED_SLOWER,
    COMPARE_SPEED_TIE,
    battler_slower,
    compare_speed,
)
from nuzline.engine.state import MOVES, BattleState

_I32 = jnp.int32

# FlagMoveDamageScore / IfCurrentMoveKills argument (include/constants/battle/trainer_ai.h:45).
USE_MAX_DAMAGE, ROLL_FOR_DAMAGE = 0, 1
# CheckBattlerAbility results in calcTemp (trainer_ai.h:39-41).
AI_NOT_HAVE, AI_HAVE, AI_UNKNOWN = 0, 1, 2
# FlagMoveDamageScore results in calcTemp (trainer_ai.h:35-37).
AI_NO_COMPARISON_MADE, AI_NOT_HIGHEST_DAMAGE, AI_MOVE_IS_HIGHEST_DAMAGE = 0, 1, 2
SIM_ROLL_MAX = 15  # engine roll index of variance 100 (USE_MAX_DAMAGE)
STAGE_NEUTRAL = 6  # decomp stat stages run 0..12 with 6 neutral (engine 0)


class Battler(IntEnum):
    """Script battler argument, as the engine side it names (singles)."""

    DEFENDER = 0  # AI_BATTLER_DEFENDER: the player's active Pokemon
    ATTACKER = 1  # AI_BATTLER_ATTACKER: the AI's active Pokemon


class LoadType(IntEnum):
    """``LoadTypeFrom`` targets Roark's scripts use (trainer_ai.c:937), with the decomp's
    values (``LOAD_*``: line n + 1 of generated/ai_load_type_targets.txt is n)."""

    DEFENDER_TYPE_1 = 0  # LOAD_DEFENDER_TYPE_1
    ATTACKER_TYPE_1 = 1  # LOAD_ATTACKER_TYPE_1
    DEFENDER_TYPE_2 = 2  # LOAD_DEFENDER_TYPE_2
    ATTACKER_TYPE_2 = 3  # LOAD_ATTACKER_TYPE_2
    MOVE_TYPE = 4  # LOAD_MOVE_TYPE


class SideCondition(IntEnum):
    """``IfSideCondition`` masks Roark's scripts use (``SIDE_CONDITION_*``,
    include/constants/battle/condition.h:88-95), each read from its ``BattleState`` field
    (``_SIDE_CONDITION_FIELDS``)."""

    REFLECT = 1 << 0  # BattleState.reflect
    LIGHT_SCREEN = 1 << 1  # BattleState.light_screen
    SPIKES = 1 << 2  # :90; BattleState.spikes (layers, LoadSpikesLayers)
    SAFEGUARD = 1 << 3  # BattleState.safeguard
    STEALTH_ROCK = 1 << 7  # BattleState.stealth_rock
    TOXIC_SPIKES = 1 << 10  # :98; BattleState.toxic_spikes (layers, LoadSpikesLayers)


_SIDE_CONDITION_FIELDS = {
    SideCondition.REFLECT: "reflect",
    SideCondition.LIGHT_SCREEN: "light_screen",
    SideCondition.SAFEGUARD: "safeguard",
    SideCondition.STEALTH_ROCK: "stealth_rock",
}


class MonCondition(IntEnum):
    """``IfStatus`` masks: the decomp's ``MON_CONDITION_*`` status bits
    (include/constants/battle/condition.h:7-14, :22, :25), each group one ``ids.Status``."""

    SLEEP = 0b111  # MON_CONDITION_SLEEP: bits 0-2, any of them set is asleep
    POISON = 1 << 3
    BURN = 1 << 4
    FREEZE = 1 << 5
    PARALYSIS = 1 << 6
    TOXIC = 1 << 7  # its counter (bits 8-11) is refused in a mask: _condition_statuses
    ANY = 0xFF  # MON_CONDITION_ANY: every group above
    # MON_CONDITION_FACADE_BOOST (include/constants/battle/condition.h:29): toxic, poison,
    # burn, paralysis
    FACADE_BOOST = (1 << 7) | (1 << 3) | (1 << 4) | (1 << 6)


class FieldCondition(IntEnum):
    """``IfFieldConditionsMask`` masks Roark's scripts use (``FIELD_CONDITION_*``,
    include/constants/battle/condition.h:123-125, :144)."""

    TRICK_ROOM = (1 << 16) | (1 << 17) | (1 << 18)  # BattleState.trick_room


class VolatileCondition(IntEnum):
    """``IfVolatileStatus`` masks the transcribed scripts test (``VOLATILE_CONDITION_*``,
    include/constants/battle/condition.h), each read from its ``BattleState`` field
    (``if_volatile_status``)."""

    CONFUSION = (1 << 0) | (1 << 1) | (1 << 2)  # :32-34, :65; BattleState.confusion
    ATTRACT = (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19)  # :48-51, :70; .infatuated


_VOLATILE_FIELDS = {
    VolatileCondition.CONFUSION: "confusion",
    VolatileCondition.ATTRACT: "infatuated",
}


class MoveEffectMask(IntEnum):
    """``IfMoveEffect`` masks the transcribed scripts test (``MOVE_EFFECT_*``,
    include/constants/battle/moves.h)."""

    LEECH_SEED = 1 << 2  # :10; BattleState.leech_seed (the seeded battler's side)


_MOVE_EFFECT_FIELDS = {MoveEffectMask.LEECH_SEED: "leech_seed"}


class Gender(IntEnum):
    """``LoadGender`` values (line n + 1 of generated/genders.txt is n), as
    ``BattleState.gender`` holds them."""

    MALE = 0
    FEMALE = 1
    NONE = 2


class AiWeather(IntEnum):
    """``LoadCurrentWeather`` values (``AI_WEATHER_*``: line n + 1 of
    generated/ai_weather_types.txt is n)."""

    CLEAR = 0
    SUNNY = 1
    RAINING = 2
    SANDSTORM = 3
    HAILING = 4
    DEEP_FOG = 5


class TypeMulti(IntEnum):
    """``IfMoveEffectivenessEquals`` classes (include/constants/battle.h:144-153)."""

    IMMUNE = 0
    QUARTER = 10
    HALF = 20
    BASE = 40
    STAB = 60
    DOUBLE = 80
    QUADRUPLE = 160


class BattleEffect(IntEnum):
    """Decomp ``BATTLE_EFFECT_*`` ids (line - 1 of generated/move_battle_effects.txt) that
    the transcribed scripts compare (``IfCurrentMoveEffectEqualTo``, trainer_ai.c:1441)."""

    HIT = 0
    RECOVER_HALF_DAMAGE_DEALT = 3  # Giga Drain
    STATUS_SLEEP = 1
    HALVE_DEFENSE = 7  # Selfdestruct, Explosion
    RECOVER_DAMAGE_SLEEP = 8  # Dream Eater (Expert_StatusSleep's IfMoveEffectKnown, :1816)
    ATK_UP = 10
    DEF_UP = 11
    SP_ATK_UP = 13
    SP_DEF_UP = 14  # Expert_ToxicLeechSeed's IfMoveEffectKnown (script.s:2699)
    ATK_DOWN = 18
    DEF_DOWN = 19
    SPEED_DOWN = 20
    EVA_DOWN = 24
    FLINCH_HIT = 31
    RESTORE_HALF_HP = 32  # Recover
    STATUS_BADLY_POISON = 33  # Toxic
    SET_LIGHT_SCREEN = 35  # Light Screen
    BYPASS_ACCURACY = 17
    HIGH_CRITICAL = 43
    RECOIL_QUARTER = 48
    STATUS_CONFUSE = 49
    ATK_UP_2 = 50
    DEF_UP_2 = 51
    SPEED_UP_2 = 52  # Rock Polish
    SP_ATK_UP_2 = 53
    SP_DEF_UP_2 = 54
    ATK_DOWN_2 = 58
    DEF_DOWN_2 = 59
    SPEED_DOWN_2 = 60
    SP_DEF_DOWN_2 = 62
    SET_REFLECT = 65
    STATUS_PARALYZE = 67
    LOWER_SPEED_HIT = 70
    STATUS_LEECH_SEED = 84
    KO_MON_THAT_DEFEATED_USER = 98  # Destiny Bond
    PRIORITY_1 = 103  # EvalAttack_ApplyKillBonuses' +2 (script.s:6400); no Roark move has it
    STATUS_NIGHTMARE = 107  # Nightmare (Expert_StatusSleep's IfMoveEffectKnown, :1817)
    CURSE = 109
    PROTECT = 111  # Expert_ToxicLeechSeed's IfMoveEffectKnown (script.s:2700)
    SET_SPIKES = 112
    WEATHER_SANDSTORM = 115  # Weather_Main (script.s:7989)
    ATK_UP_2_STATUS_CONFUSION = 118  # Swagger
    INFATUATE = 120  # Attract
    HIT_BEFORE_SWITCH = 128
    WEATHER_RAIN = 136  # Weather_Main (script.s:7988)
    WEATHER_SUN = 137  # Weather_Main (script.s:7987)
    RAISE_ALL_STATS_HIT = 140  # AncientPower (Risky_RiskyEffects, script.s:6550)
    WEATHER_HAIL = 164  # Weather_Main (script.s:7990)
    STATUS_BURN = 167  # Will-O-Wisp
    DOUBLE_POWER_WHEN_STATUSED = 169  # Facade
    DOUBLE_POWER_AND_CURE_PARALYSIS = 171  # SmellingSalt
    REMOVE_SCREENS = 186  # Brick Break
    STEAL_STATUS_MOVE = 195  # Snatch (Expert_Recovery_CheckForSnatch, script.s:2670)
    RECOIL_THIRD = 198
    ATK_DEF_UP = 208  # Bulk Up
    SP_ATK_SP_DEF_UP = 211  # Calm Mind
    ATK_SPD_UP = 212  # Dragon Dance
    DOUBLE_POWER_WHEN_BELOW_HALF = 221  # Brine
    TOXIC_SPIKES = 249
    RECOIL_PARALYZE_HIT = 262
    STEALTH_ROCK = 266
    RECOIL_HALF = 269  # Head Smash


# Decomp move id -> battle effect for every move whose script paths are transcribed
# (roark.md 4-5, plus the test fixture's Tackle); the same set as the script-roll site
# table (nuzline.ai.outcome.AI_SITE_MOVES). None of these effects is in
# NO_DAMAGE_CALC_EFFECTS or ALT_POWER_EFFECTS.
MOVE_EFFECTS: dict[int, int] = {
    446: BattleEffect.STEALTH_ROCK,  # Stealth Rock
    88: BattleEffect.HIT,  # Rock Throw
    228: BattleEffect.HIT_BEFORE_SWITCH,  # Pursuit
    103: BattleEffect.DEF_DOWN_2,  # Screech
    43: BattleEffect.DEF_DOWN,  # Leer
    29: BattleEffect.FLINCH_HIT,  # Headbutt
    33: BattleEffect.HIT,  # Tackle
}

# sNoDamageCalcMoveEffects (trainer_ai.c:31-46): FlagMoveDamageScore (:1035-1036),
# IfCurrentMoveKills (:1553-1554) and TrainerAI_CalcAllDamage (:2827-2828) compute no damage
# for a move with one of these effects. BATTLE_EFFECT_* ids, line - 1 of
# generated/move_battle_effects.txt.
NO_DAMAGE_CALC_EFFECTS = (
    7,  # HALVE_DEFENSE (Selfdestruct, Explosion)
    8,  # RECOVER_DAMAGE_SLEEP
    39,  # CHARGE_TURN_HIGH_CRIT
    75,  # CHARGE_TURN_HIGH_CRIT_FLINCH
    80,  # RECHARGE_AFTER
    145,  # CHARGE_TURN_DEF_UP
    151,  # SKIP_CHARGE_TURN_IN_SUN
    161,  # SPIT_UP
    170,  # HIT_LAST_WHIFF_IF_HIT
    182,  # LOWER_OWN_ATK_AND_DEF
    190,  # DECREASE_POWER_WITH_LESS_USER_HP
    248,  # HIT_FIRST_IF_TARGET_ATTACKING
    269,  # RECOIL_HALF (Head Smash)
)
# sAltPowerMoveEffects (trainer_ai.c:48-61): the same three commands compute damage for a
# move with one of these effects whatever its power (:1035, :1553, :2827),
# through TrainerAI_CalcDamage's id cases, which the engine's AI damage does not transcribe;
# policy.ai_refusals refuses such an opponent move.
ALT_POWER_EFFECTS = (
    135,  # RANDOM_POWER_BASED_ON_IVS
    219,  # POWER_BASED_ON_LOW_SPEED
    222,  # NATURAL_GIFT
    268,  # JUDGEMENT
    41,  # 40_DAMAGE_FLAT
    87,  # LEVEL_DAMAGE_FLAT
    88,  # RANDOM_DAMAGE_1_TO_150_LEVEL (Kaizo Triple Axel and Fury Cutter)
    121,  # POWER_BASED_ON_FRIENDSHIP
    123,  # POWER_BASED_ON_LOW_FRIENDSHIP
    130,  # 20_DAMAGE_FLAT
    196,  # INCREASE_POWER_WITH_WEIGHT
)
_NO_DAMAGE_CALC = jnp.array(NO_DAMAGE_CALC_EFFECTS, _I32)
# Risky_RiskyEffects (script.s:6536-6562): the effects Risky_Main scores (IfLoadedNotInTable,
# :6527; trainer_ai.c:876-891). BATTLE_EFFECT_* ids, line - 1 of
# generated/move_battle_effects.txt, in the table's order.
RISKY_EFFECTS = (
    1,  # STATUS_SLEEP
    7,  # HALVE_DEFENSE (Selfdestruct, Explosion)
    9,  # COPY_MOVE
    38,  # ONE_HIT_KO
    43,  # HIGH_CRITICAL
    49,  # STATUS_CONFUSE
    83,  # CALL_RANDOM_MOVE
    88,  # RANDOM_DAMAGE_1_TO_150_LEVEL
    89,  # COUNTER
    98,  # KO_MON_THAT_DEFEATED_USER
    118,  # ATK_UP_2_STATUS_CONFUSION
    120,  # INFATUATE
    122,  # RANDOM_POWER_MAYBE_HEAL
    140,  # RAISE_ALL_STATS_HIT (AncientPower)
    142,  # MAX_ATK_LOSE_HALF_MAX_HP
    144,  # MIRROR_COAT
    170,  # HIT_LAST_WHIFF_IF_HIT
    185,  # DOUBLE_POWER_IF_HIT
    199,  # CONFUSE_ALL
    219,  # POWER_BASED_ON_LOW_SPEED
    226,  # RANDOM_STAT_UP_2
    227,  # METAL_BURST
    230,  # DOUBLE_POWER_IF_MOVING_SECOND
    241,  # USE_MOVE_FIRST
    248,  # HIT_FIRST_IF_TARGET_ATTACKING
)
# SetupFirstTurn_SetupEffects (script.s:6435-6497): the effects SetupFirstTurn_Main scores on
# turn 0 (IfLoadedNotInTable, :6426; trainer_ai.c:876-891). BATTLE_EFFECT_* ids, line - 1 of
# generated/move_battle_effects.txt, in the table's order, its two repeats included (61
# entries, 59 effects).
SETUP_FIRST_TURN_EFFECTS = (
    10,  # ATK_UP (:6436)
    11,  # DEF_UP
    12,  # SPEED_UP
    13,  # SP_ATK_UP
    14,  # SP_DEF_UP
    15,  # ACC_UP
    16,  # EVA_UP
    18,  # ATK_DOWN
    19,  # DEF_DOWN
    20,  # SPEED_DOWN
    21,  # SP_ATK_DOWN
    22,  # SP_DEF_DOWN
    23,  # ACC_DOWN
    24,  # EVA_DOWN
    30,  # CONVERSION
    35,  # SET_LIGHT_SCREEN
    54,  # SP_DEF_UP_2 (:6452, again at :6459)
    47,  # CRIT_UP_2
    49,  # STATUS_CONFUSE
    50,  # ATK_UP_2
    51,  # DEF_UP_2
    52,  # SPEED_UP_2
    53,  # SP_ATK_UP_2
    54,  # SP_DEF_UP_2
    55,  # ACC_UP_2
    56,  # EVA_UP_2
    58,  # ATK_DOWN_2
    59,  # DEF_DOWN_2
    60,  # SPEED_DOWN_2
    61,  # SP_ATK_DOWN_2
    62,  # SP_DEF_DOWN_2
    63,  # EVA_DOWN_2
    64,  # ACC_DOWN_2
    65,  # SET_REFLECT
    66,  # STATUS_POISON
    67,  # STATUS_PARALYZE
    79,  # SET_SUBSTITUTE
    84,  # STATUS_LEECH_SEED
    108,  # EVA_UP_2_MINIMIZE
    109,  # CURSE
    118,  # ATK_UP_2_STATUS_CONFUSION
    213,  # CAMOUFLAGE (:6477, again at :6490)
    187,  # STATUS_SLEEP_NEXT_TURN
    156,  # DEF_UP_DOUBLE_ROLLOUT_POWER
    165,  # TORMENT
    166,  # SP_ATK_UP_CAUSE_CONFUSION
    167,  # STATUS_BURN
    181,  # GROUND_TRAP_USER_CONTINUOUS_HEAL
    192,  # MAKE_SHARED_MOVES_UNUSEABLE
    199,  # CONFUSE_ALL
    205,  # ATK_DEF_DOWN
    206,  # DEF_SPD_UP
    208,  # ATK_DEF_UP
    211,  # SP_ATK_SP_DEF_UP
    213,  # CAMOUFLAGE
    225,  # DOUBLE_SPEED_3_TURNS
    226,  # RANDOM_STAT_UP_2
    240,  # PREVENT_CRITS
    252,  # GIVE_GROUND_IMMUNITY
    258,  # REMOVE_HAZARDS_SCREENS_EVA_DOWN
    261,  # WHIRLPOOL (:6496)
)
# CheckHP_Main's six tables (script.s:7748-7977), by label: the effects it discourages at the
# attacker's (CheckHP_Discourage*) and the target's (CheckHP_Target_Discourage*) high (> 70 %),
# medium (31-70 %) and low HP (IfLoadedInTable, :7703-7737; trainer_ai.c:858-874).
# BATTLE_EFFECT_* ids, line - 1 of generated/move_battle_effects.txt, in each table's order,
# repeats included. CheckHP_Target_DiscourageAtHighHP is empty (:7866-7867).
CHECK_HP_TABLES: dict[str, tuple[int, ...]] = {
    "CheckHP_DiscourageAtHighHP": (  # script.s:7748
        7,  # HALVE_DEFENSE
        32,  # RESTORE_HALF_HP
        37,  # REST
        98,  # KO_MON_THAT_DEFEATED_USER
        99,  # INCREASE_POWER_WITH_LESS_HP
        116,  # SURVIVE_WITH_1_HP
        132,  # HEAL_HALF_MORE_IN_SUN
        168,  # FAINT_AND_ATK_SP_ATK_DOWN_2
        194,  # REMOVE_ALL_PP_ON_DEFEAT
        214,  # HEAL_HALF_REMOVE_FLYING_TYPE
        220,  # FAINT_AND_FULL_HEAL_NEXT_MON
        270,  # FAINT_FULL_RESTORE_NEXT_MON
    ),
    "CheckHP_DiscourageAtMediumHP": (  # script.s:7763
        7,  # HALVE_DEFENSE
        10,  # ATK_UP
        11,  # DEF_UP
        12,  # SPEED_UP
        13,  # SP_ATK_UP
        14,  # SP_DEF_UP
        15,  # ACC_UP
        16,  # EVA_UP
        18,  # ATK_DOWN
        19,  # DEF_DOWN
        20,  # SPEED_DOWN
        21,  # SP_ATK_DOWN
        22,  # SP_DEF_DOWN
        23,  # ACC_DOWN
        24,  # EVA_DOWN
        26,  # BIDE
        30,  # CONVERSION
        35,  # SET_LIGHT_SCREEN
        46,  # PREVENT_STAT_REDUCTION
        47,  # CRIT_UP_2
        50,  # ATK_UP_2
        51,  # DEF_UP_2
        52,  # SPEED_UP_2
        53,  # SP_ATK_UP_2
        54,  # SP_DEF_UP_2
        55,  # ACC_UP_2
        56,  # EVA_UP_2
        58,  # ATK_DOWN_2
        59,  # DEF_DOWN_2
        60,  # SPEED_DOWN_2
        61,  # SP_ATK_DOWN_2
        62,  # SP_DEF_DOWN_2
        63,  # EVA_DOWN_2
        64,  # ACC_DOWN_2
        93,  # CONVERSION2
        124,  # PREVENT_STATUS
        142,  # MAX_ATK_LOSE_HALF_MAX_HP
        205,  # ATK_DEF_DOWN
        206,  # DEF_SPD_UP
        208,  # ATK_DEF_UP
        211,  # SP_ATK_SP_DEF_UP
        212,  # ATK_SPD_UP
        240,  # PREVENT_CRITS
        243,  # SWAP_ATK_SP_ATK_STAT_CHANGES
        244,  # SWAP_DEF_SP_DEF_STAT_CHANGES
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER
    ),
    "CheckHP_DiscourageAtLowHP": (  # script.s:7812
        10,  # ATK_UP
        11,  # DEF_UP
        12,  # SPEED_UP
        13,  # SP_ATK_UP
        14,  # SP_DEF_UP
        15,  # ACC_UP
        16,  # EVA_UP
        18,  # ATK_DOWN
        19,  # DEF_DOWN
        20,  # SPEED_DOWN
        21,  # SP_ATK_DOWN
        22,  # SP_DEF_DOWN
        23,  # ACC_DOWN
        24,  # EVA_DOWN
        26,  # BIDE
        30,  # CONVERSION
        35,  # SET_LIGHT_SCREEN
        46,  # PREVENT_STAT_REDUCTION
        47,  # CRIT_UP_2
        50,  # ATK_UP_2
        51,  # DEF_UP_2
        52,  # SPEED_UP_2
        53,  # SP_ATK_UP_2
        54,  # SP_DEF_UP_2
        55,  # ACC_UP_2
        56,  # EVA_UP_2
        58,  # ATK_DOWN_2
        59,  # DEF_DOWN_2
        60,  # SPEED_DOWN_2
        61,  # SP_ATK_DOWN_2
        62,  # SP_DEF_DOWN_2
        63,  # EVA_DOWN_2
        64,  # ACC_DOWN_2
        81,  # RAISE_ATK_WHEN_HIT
        93,  # CONVERSION2
        94,  # NEXT_ATTACK_ALWAYS_HITS
        124,  # PREVENT_STATUS
        142,  # MAX_ATK_LOSE_HALF_MAX_HP
        143,  # COPY_STAT_CHANGES
        144,  # MIRROR_COAT
        190,  # DECREASE_POWER_WITH_LESS_USER_HP
        205,  # ATK_DEF_DOWN
        206,  # DEF_SPD_UP
        208,  # ATK_DEF_UP
        211,  # SP_ATK_SP_DEF_UP
        212,  # ATK_SPD_UP
        201,  # HALVE_ELECTRIC_DAMAGE
        210,  # HALVE_FIRE_DAMAGE
        226,  # RANDOM_STAT_UP_2
        227,  # METAL_BURST
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER
    ),
    "CheckHP_Target_DiscourageAtHighHP": (  # script.s:7866
    ),
    "CheckHP_Target_DiscourageAtMediumHP": (  # script.s:7869
        10,  # ATK_UP
        11,  # DEF_UP
        12,  # SPEED_UP
        13,  # SP_ATK_UP
        14,  # SP_DEF_UP
        15,  # ACC_UP
        16,  # EVA_UP
        18,  # ATK_DOWN
        19,  # DEF_DOWN
        20,  # SPEED_DOWN
        21,  # SP_ATK_DOWN
        22,  # SP_DEF_DOWN
        23,  # ACC_DOWN
        24,  # EVA_DOWN
        46,  # PREVENT_STAT_REDUCTION
        47,  # CRIT_UP_2
        50,  # ATK_UP_2
        51,  # DEF_UP_2
        52,  # SPEED_UP_2
        53,  # SP_ATK_UP_2
        54,  # SP_DEF_UP_2
        55,  # ACC_UP_2
        56,  # EVA_UP_2
        58,  # ATK_DOWN_2
        59,  # DEF_DOWN_2
        60,  # SPEED_DOWN_2
        61,  # SP_ATK_DOWN_2
        62,  # SP_DEF_DOWN_2
        63,  # EVA_DOWN_2
        64,  # ACC_DOWN_2
        66,  # STATUS_POISON
        91,  # AVERAGE_HP
        114,  # ALL_FAINT_3_TURNS
        124,  # PREVENT_STATUS
        205,  # ATK_DEF_DOWN
        206,  # DEF_SPD_UP
        208,  # ATK_DEF_UP
        211,  # SP_ATK_SP_DEF_UP
        212,  # ATK_SPD_UP
        226,  # RANDOM_STAT_UP_2
        237,  # INCREASE_POWER_WITH_MORE_HP
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER
    ),
    "CheckHP_Target_DiscourageAtLowHP": (  # script.s:7914
        1,  # STATUS_SLEEP
        7,  # HALVE_DEFENSE
        10,  # ATK_UP
        11,  # DEF_UP
        12,  # SPEED_UP
        13,  # SP_ATK_UP
        14,  # SP_DEF_UP
        15,  # ACC_UP
        16,  # EVA_UP
        18,  # ATK_DOWN
        19,  # DEF_DOWN
        20,  # SPEED_DOWN
        21,  # SP_ATK_DOWN
        22,  # SP_DEF_DOWN
        23,  # ACC_DOWN
        24,  # EVA_DOWN
        26,  # BIDE
        30,  # CONVERSION
        33,  # STATUS_BADLY_POISON
        35,  # SET_LIGHT_SCREEN
        38,  # ONE_HIT_KO
        40,  # HALVE_HP
        40,  # HALVE_HP
        46,  # PREVENT_STAT_REDUCTION
        47,  # CRIT_UP_2
        49,  # STATUS_CONFUSE
        50,  # ATK_UP_2
        51,  # DEF_UP_2
        52,  # SPEED_UP_2
        53,  # SP_ATK_UP_2
        54,  # SP_DEF_UP_2
        55,  # ACC_UP_2
        56,  # EVA_UP_2
        58,  # ATK_DOWN_2
        59,  # DEF_DOWN_2
        60,  # SPEED_DOWN_2
        61,  # SP_ATK_DOWN_2
        62,  # SP_DEF_DOWN_2
        63,  # EVA_DOWN_2
        64,  # ACC_DOWN_2
        66,  # STATUS_POISON
        67,  # STATUS_PARALYZE
        91,  # AVERAGE_HP
        93,  # CONVERSION2
        94,  # NEXT_ATTACK_ALWAYS_HITS
        100,  # DECREASE_LAST_MOVE_PP
        114,  # ALL_FAINT_3_TURNS
        118,  # ATK_UP_2_STATUS_CONFUSION
        119,  # DOUBLE_POWER_EACH_TURN
        120,  # INFATUATE
        124,  # PREVENT_STATUS
        143,  # COPY_STAT_CHANGES
        144,  # MIRROR_COAT
        167,  # STATUS_BURN
        205,  # ATK_DEF_DOWN
        206,  # DEF_SPD_UP
        208,  # ATK_DEF_UP
        211,  # SP_ATK_SP_DEF_UP
        212,  # ATK_SPD_UP
        226,  # RANDOM_STAT_UP_2
        237,  # INCREASE_POWER_WITH_MORE_HP
        265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER
    ),
}
# A move's class (MoveTable.class, generated/move_classes.txt: line n + 1 is id n), as
# LoadDefenderLastUsedMoveClass loads it (Expert_LightScreen, script.s:2723).
CLASS_PHYSICAL, CLASS_SPECIAL = 0, 1
# TrainerAI_CalcDamage's move-id cases (trainer_ai.c:2885-3056): power, type or damage of
# these moves come from the case, not the move table, whatever the move's effect (Kaizo's
# Return and Frustration have effect 0 and still take theirs, :3018-3026). Decomp move ids.
CALC_DAMAGE_MOVES = (363, 449, 237, 360, 82, 69, 101, 149, 216, 218, 222, 49, 67, 447)
# TrainerAI_MoveType's move-id cases (trainer_ai.c:3132-3222): the type the AI gives
# these moves is not their table type. Decomp move ids.
MOVE_TYPE_MOVES = (363, 449, 237, 311)

# Decomp ids, checked against the manifest by tests/ai/test_policy.py.
ABILITY_IDS = {
    "Mold Breaker": 104,
    "Volt Absorb": 10,
    "Motor Drive": 78,
    "Water Absorb": 11,
    "Flash Fire": 18,
    "Wonder Guard": 25,
    "Levitate": 26,
    "Soundproof": 43,
    "Clear Body": 29,
    "White Smoke": 73,
    "Shadow Tag": 23,
    "Magnet Pull": 42,
    "Arena Trap": 71,
    "Normalize": 96,
    "Adaptability": 91,
    "Scrappy": 113,
    "Tinted Lens": 110,
    "Filter": 111,
    "Solid Rock": 116,
    # Basic_CheckCannotExplode (script.s:305), Basic_CheckHighStatStage_Speed and
    # Basic_CheckCurse (:366, :689), Basic_CheckCannotPoison (:530-540).
    "Damp": 6,
    "Simple": 86,
    "Immunity": 17,
    "Magic Guard": 98,
    "Poison Heal": 90,
    "Leaf Guard": 102,
    "Hydration": 93,
    # Expert_RecoilMove (script.s:6297-6298; Magic Guard above).
    "Rock Head": 69,
    # E1 batch 6: Basic_CheckCannotSleep / _Confuse / _Paralyze / _Attract / _Burn
    # (script.s:293-294, :603, :616, :625-626 with Volt Absorb above, :735, :882).
    "Insomnia": 15,
    "Vital Spirit": 72,
    "Own Tempo": 20,
    "Limber": 7,
    "Oblivious": 12,
    "Water Veil": 41,
    # E1 batch 10: Basic_CheckLowStatStage_Attack / _Speed / _Evasion (script.s:431, :441,
    # :465-467).
    "Hyper Cutter": 52,
    "Speed Boost": 3,
    "No Guard": 99,
}
MOVE_IDS = {
    "Thunder Wave": 86,  # Basic_CheckCannotParalyze (script.s:620)
    "Psych Up": 244,  # Expert_Swagger (script.s:2852)
    # Expert_SpeedDownOnHit (script.s:2345-2347)
    "Icy Wind": 196,
    "Rock Tomb": 317,
    "Mud Shot": 341,
    "Fissure": 90,
    "Horn Drill": 32,
    "U-turn": 369,
    "Roar": 46,
    "Whirlwind": 18,
    # Expert_Curse's IfMoveKnown checks (script.s:3472-3473), read from K3c Task 6.
    "Gyro Ball": 360,
    "Trick Room": 433,
    # Basic_CheckSoundproof's IfMoveEqualTo list (script.s:121-131).
    "Growl": 45,
    "Sing": 47,
    "Supersonic": 48,
    "Screech": 103,
    "Snore": 173,
    "Uproar": 253,
    "Metal Sound": 319,
    "GrassWhistle": 320,
    "Bug Buzz": 405,
    "Chatter": 448,
}
SOUND_MOVES = tuple(
    MOVE_IDS[n]
    for n in (
        "Growl",
        "Roar",
        "Sing",
        "Supersonic",
        "Screech",
        "Snore",
        "Uproar",
        "Metal Sound",
        "GrassWhistle",
        "Bug Buzz",
        "Chatter",
    )
)
TYPE_IDS = {
    "Poison": 3,
    "Ground": 4,
    "Ghost": 7,
    "Steel": 8,
    "Fire": 10,
    "Grass": 12,  # Basic_CheckCannotLeechSeed (script.s:643-645)
    "Water": 11,
    "Electric": 13,
    "Psychic": 14,
}
ITEM_EXPERT_BELT = 268  # HOLD_EFFECT_POWER_UP_SE
_TRAPPING = jnp.array([ABILITY_IDS[n] for n in ("Shadow Tag", "Magnet Pull", "Arena Trap")], _I32)


class ReadKind(IntEnum):
    """What a loose read draws (``LooseDraws.kind``)."""

    NONE = 0  # no read at this index
    ROLL = 1  # IfRandomLessThan N: P(1) = N/256
    ABILITY = 2  # LoadBattlerAbility DEFENDER's coin flip: P(1) = 1/2
    SPEED_TIE = 3  # IfSpeedCompareEqualTo's tie, RandNext() & 1: P(1) = 1/2


# LooseDraws columns: at least the reads one slot's scripts make. Every read counts, on the
# slot's path or not (the scripts are straight-line): 35 after K3c Task 6, 74 after E1 batch 7
# under every transcribed flag. More raises (policy._loose_reads). Tests only, but each column
# costs compile memory in every loose program, so the headroom is kept modest.
LOOSE_READS = 96


class LooseDraws(NamedTuple):
    """The loose oracle's draws (module docstring). Read i of slot m, counted in the order
    the straight-line scripts make the reads (``count``), takes ``bits[m, i]``."""

    bits: jnp.ndarray  # int8[MOVES, LOOSE_READS]: 1 = the jump / ability slot 1
    count: jnp.ndarray  # int32[MOVES]: reads made so far in each slot
    p: jnp.ndarray  # float32[MOVES, LOOSE_READS]: P(bit = 1) of each read
    kind: jnp.ndarray  # int8[MOVES, LOOSE_READS]: the ReadKind of each read
    live: jnp.ndarray  # bool[MOVES, LOOSE_READS]: the read is on the slot's path
    line: jnp.ndarray  # int32[MOVES, LOOSE_READS]: an IfRandomLessThan's script.s line, else 0


def loose_draws(bits) -> LooseDraws:
    """``LooseDraws`` with ``bits`` (``int[MOVES, LOOSE_READS]``) and no read made yet."""
    shape = (MOVES, LOOSE_READS)
    return LooseDraws(
        bits=jnp.asarray(bits, jnp.int8),
        count=jnp.zeros(MOVES, _I32),
        p=jnp.zeros(shape, jnp.float32),
        kind=jnp.zeros(shape, jnp.int8),
        live=jnp.zeros(shape, bool),
        line=jnp.zeros(shape, _I32),
    )


class AiContext(NamedTuple):
    """The decomp's ``AIContext`` for one decision, plus the masks of the straight-line
    scripts."""

    tables: Tables
    state: BattleState
    outcome: AiOutcome
    scores: jnp.ndarray  # int32[4], moveScore
    slot: int  # moveSlot under evaluation (static: the scripts unroll over slots)
    move: jnp.ndarray  # int32, AI_CONTEXT.move: decomp id, 0 = MOVE_NONE (empty or no PP)
    live: jnp.ndarray  # bool: the decomp's script cursor is on the current path
    jump: jnp.ndarray  # bool: the last jump command's condition
    calc_temp: jnp.ndarray  # int32, calcTemp
    seen_moves: jnp.ndarray  # int32[4]: the player's active's moves the AI has seen, else 0
    player_ability: jnp.ndarray  # int32: LoadBattlerAbility DEFENDER (decomp id)
    damage: jnp.ndarray  # int32[4]: ai_damage of the AI's move slots, USE_MAX_DAMAGE
    # int32[4, len(SITE_LINES)]: the flat layout's script_rolls index of each line; an
    # unused int32[] in a per-slot layout, whose reads go by order (per_slot)
    sites: jnp.ndarray
    # int32[2]: LoadBattlerAbility DEFENDER under guess 0 and guess 1 (defender_ability);
    # bool: its reads draw a coin this decision (ability_guessed). Per-slot layout.
    guess_abilities: jnp.ndarray
    guessing: jnp.ndarray
    # Per-slot layout (E1 Ruling R4), per slot: the random reads made so far on its path (the
    # next one takes script_rolls[slot, reads[slot]]) and the product of their probabilities
    reads: jnp.ndarray  # int32[MOVES]
    weight: jnp.ndarray  # float32[MOVES]
    # int32: 0, or 1 + slot of the first read on a path past the slot's slot_reads draws
    # (_slot_read); the per-slot layout refuses the decision
    missing_read: jnp.ndarray
    loose: LooseDraws | None = None  # the loose oracle's draws (tests only), else None


def ai_damage(tables: Tables, state: BattleState, move, roll) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868) of ``move`` by the AI's
    active Pokemon against the player's, with variance ``85 + roll`` percent applied
    after STAB and the type chart (:3109; the battle applies it before them): the
    engine's ``ai_damage_for_roll`` (roark.md 7.1), 0 for a move that is not damaging
    (``is_damaging``) and for an immunity."""
    move = jnp.asarray(move, _I32)
    damage = ai_damage_for_roll(tables, state, 1, move, roll)
    return jnp.where(is_damaging(tables, move), damage, 0)


def is_damaging(tables: Tables, move) -> jnp.ndarray:
    """``bool``: the AI computes damage for ``move``: it is a move, its power is greater
    than 1 and its battle effect is not in ``NO_DAMAGE_CALC_EFFECTS``
    (``TrainerAI_CalcAllDamage``, trainer_ai.c:2827-2828; ``FlagMoveDamageScore``,
    :1035-1036; ``IfCurrentMoveKills``, :1553-1554). Those tests also pass a move whose
    effect is in ``ALT_POWER_EFFECTS``; ``policy.ai_refusals`` refuses those."""
    move = jnp.asarray(move, _I32)
    no_calc = jnp.isin(move_effect(tables, move), _NO_DAMAGE_CALC)
    return (move != 0) & (tables.move_power[move] > 1) & ~no_calc


def _slot_damage(tables: Tables, state: BattleState, rolls) -> jnp.ndarray:
    """``int32[4]``: ``TrainerAI_CalcAllDamage`` (trainer_ai.c:2799) over the AI's true
    moveset, whatever the PP, with ``rolls[i]`` for slot i."""
    moves = state.moves[1, state.active[1]].astype(_I32)
    return jax.vmap(lambda m, r: ai_damage(tables, state, m, r))(moves, jnp.asarray(rolls, _I32))


def move_effect(tables: Tables, move) -> jnp.ndarray:
    """``int32``: the decomp battle effect of ``move`` as the AI reads it,
    ``MOVE_DATA(move).effect`` (``tables.move_ai_effect``; ``IfCurrentMoveEffectEqualTo``,
    trainer_ai.c:1441)."""
    return tables.move_ai_effect[jnp.asarray(move, _I32)].astype(_I32)


def effectiveness_class(tables: Tables, state: BattleState, move) -> jnp.ndarray:
    """``int32``: the ``TypeMulti`` class of ``move`` by the AI's active Pokemon against
    the player's (``IfMoveEffectivenessEquals``, trainer_ai.c:1311; roark.md 7.2): 40
    through ``BattleSystem_ApplyTypeChart`` (STAB x15/10, then each chart entry with
    ``BattleSystem_Divide``), mapped 120 -> 80, 240 -> 160, 30 -> 20, 15 -> 10, and 0 when
    the status mask has an immunity bit. Normalize, Adaptability, Scrappy, Tinted Lens,
    Expert Belt, Levitate, Wonder Guard, Filter, Solid Rock, Iron Ball,
    Miracle Eye, Roost, Magnet Rise and Ingrain change this value and are refused
    (``policy.ai_refusals``)."""
    move = jnp.asarray(move, _I32)
    d_types = state.types[0, state.active[0]]
    value = apply_type_chart(tables, state, 1, move, TypeMulti.BASE)
    stab_value = int(TypeMulti.STAB)
    mapped = jnp.select(
        [
            value == stab_value * 2,
            value == stab_value * 4,
            value == stab_value // 2,
            value == stab_value // 4,
        ],
        [TypeMulti.DOUBLE, TypeMulti.QUADRUPLE, TypeMulti.HALF, TypeMulti.QUARTER],
        value,
    )
    flags = type_chart_flags(tables, move, d_types, state.gravity, state.foresight[0] != 0)
    immune = (flags & INEFFECTIVE) != 0
    # Struggle: ApplyTypeChart returns 40 unchanged and sets no bit (battle_lib.c:2573).
    return jnp.where(immune, TypeMulti.IMMUNE, mapped).astype(_I32)


def attacker_ability(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: the AI's active Pokemon's current ability (decomp id): ``battleMons[battler]
    .ability`` (trainer_ai.c:1205): ``damage.decomp_abilities``, the engine's current ability
    (a change in battle such as Trace's copy follows it once the engine models the change)."""
    return decomp_abilities(tables, state)[1, state.active[1].astype(_I32)]


def _defender_abilities(tables: Tables, state: BattleState):
    """``(true, known, base)`` of the player's active Pokemon: its true ability, the current
    one (``battleMons[battler].ability``, trainer_ai.c:1184-1186; ``damage.decomp_abilities``:
    the form's, or a copy the engine models, such as Trace's, which the AI also learns,
    ``BattleAI_SetAbility``,
    battle_script.c:3401-3403, :12106-12109),
    whether ``LoadBattlerAbility DEFENDER`` returns it without a guess (learned, or Shadow
    Tag, Magnet Pull or Arena Trap, trainer_ai.c:1181-1188), and the base species' two
    abilities ``int32[2]`` (``defender_ability``)."""
    d = state.active[0].astype(_I32)
    species = state.species[0, d].astype(_I32)
    true = decomp_abilities(tables, state)[0, d]
    known = (state.ai_ability_known[d] > 0) | jnp.isin(true, _TRAPPING)
    base = tables.species_abilities_decomp[species].astype(_I32)
    return true, known, base


def ability_guessed(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool``: a ``LoadBattlerAbility DEFENDER`` read of this decision draws its coin,
    ``BattleSystem_RandNext() & 1`` (trainer_ai.c:1194-1198): the AI does not know the
    player's ability, it is not Shadow Tag, Magnet Pull or Arena Trap (:1181-1188), and the
    base species has two nonzero abilities (:1191-1194; even when they are the same
    ability). It is the same for every read of the decision."""
    _, known, base = _defender_abilities(tables, state)
    return ~known & (base[0] != 0) & (base[1] != 0)


def defender_ability(tables: Tables, state: BattleState, ability_guess) -> jnp.ndarray:
    """``int32``: ``LoadBattlerAbility DEFENDER`` (trainer_ai.c:1170; roark.md 9.2): the
    true ability when the AI has learned it or when it is Shadow Tag, Magnet Pull or
    Arena Trap; else the guessed species slot ``ability_guess`` when the species has two
    abilities; else its only one. The engine has no ability suppression (Gastro Acid),
    so the ``ABILITY_NONE`` case never applies.

    The true ability is the form's (its personal row: ``BoxPokemon_SetGiratinaForm``
    recomputes it, ``src/pokemon.c:4391-4400``); the guess is the base species' two
    abilities, whatever the form, since ``AICmd_LoadBattlerAbility`` reads
    ``SpeciesData_GetSpeciesValue(species, SPECIES_DATA_ABILITY_1/2)``
    (``trainer_ai.c:1191-1192``; ``AICmd_CheckBattlerAbility`` likewise, ``:1236-1237``),
    which loads the personal data of the species id (``src/pokemon.c:2283``)."""
    true, known, base = _defender_abilities(tables, state)
    guess = base[jnp.clip(jnp.asarray(ability_guess, _I32), 0, 1)]
    only = jnp.where(base[0] != 0, base[0], base[1])
    resolved = jnp.where((base[0] != 0) & (base[1] != 0), guess, only)
    return jnp.where(known, true, resolved)


def new_context(
    tables: Tables,
    state: BattleState,
    outcome: AiOutcome,
    sites=None,
    loose: LooseDraws | None = None,
) -> AiContext:
    """``TrainerAI_Init`` (trainer_ai.c:211) with ``AI_INIT_SCORE_ALL_MOVES``: every score
    100, then 0 for the slots ``BattleSystem_CheckInvalidMoves`` flags (the engine's
    ``usable_moves``: empty, no PP, Disabled, Taunt on a power-0 move).

    A per-slot outcome (``script_rolls`` of rank 2, ``per_slot``) keys its reads by order on
    the path (E1 Ruling R4); a flat one reads the ``script_rolls`` index of each ``SITE_LINES``
    line per slot, ``sites`` (default ``outcome.line_sites``). ``loose``: the loose oracle's
    draws (module docstring), else None.

    ``player_ability`` is the flat layout's one resolution for the decision (its scalar
    ``ability_guess``); a per-slot layout resolves each guessing read from its own draw
    (``load_battler_ability``), and ``player_ability`` is -1.

    The seen moves are the stored ``ai_seen_moves`` plus the player's pending
    ``last_move``: ``TrainerAI_MainSingles`` runs ``TrainerAI_RecordLastMove`` (:295)
    before scoring, so the scores already know the move the player used last turn;
    ``step`` commits it to ``ai_seen_moves`` when the AI fights."""
    d = state.active[0]
    moves = state.moves[0, d].astype(_I32)
    pending = state.last_move[0].astype(_I32)
    known = (state.ai_seen_moves[d] > 0) | ((moves == pending) & (pending != 0))
    seen = jnp.where(known, moves, 0).astype(_I32)
    scores = jnp.where(usable_moves(tables, state, 1), 100, 0).astype(_I32)
    false = jnp.asarray(False)
    slotted = jnp.ndim(outcome.script_rolls) == 2
    if slotted:
        sites = jnp.zeros((), _I32)
    elif sites is None:
        sites = line_sites(state)
    return AiContext(
        tables=tables,
        state=state,
        outcome=outcome,
        scores=scores,
        slot=0,
        move=jnp.asarray(0, _I32),
        live=false,
        jump=false,
        calc_temp=jnp.asarray(0, _I32),
        seen_moves=seen,
        player_ability=(
            jnp.asarray(-1, _I32)
            if slotted
            else defender_ability(tables, state, outcome.ability_guess)
        ),
        damage=_slot_damage(tables, state, jnp.full((MOVES,), SIM_ROLL_MAX, _I32)),
        sites=jnp.asarray(sites, _I32),
        guess_abilities=jnp.stack([defender_ability(tables, state, g) for g in (0, 1)]),
        guessing=ability_guessed(tables, state),
        reads=jnp.zeros(MOVES, _I32),
        weight=jnp.ones(MOVES, jnp.float32),
        missing_read=jnp.asarray(0, _I32),
        loose=loose,
    )


def at_slot(ctx: AiContext, slot: int) -> AiContext:
    """``AI_EVAL_STEP_INIT`` (trainer_ai.c:487): the move of ``slot``, ``MOVE_NONE`` when
    it has no PP; a ``MOVE_NONE`` slot runs no command."""
    state = ctx.state
    a = state.active[1]
    move = jnp.where(state.pp[1, a, slot] > 0, state.moves[1, a, slot], 0).astype(_I32)
    return ctx._replace(
        slot=jnp.asarray(slot, _I32), move=move, live=move != 0, jump=jnp.asarray(False)
    )


def _side(battler: Battler) -> int:
    return int(battler)


def _jump(ctx: AiContext, cond) -> AiContext:
    return ctx._replace(jump=jnp.asarray(cond))


def _loose_read(
    ctx: AiContext, p: float, kind: ReadKind, live=None, line: int = 0
) -> tuple[AiContext, jnp.ndarray]:
    """The loose oracle's next draw of the current slot (module docstring): ``(ctx, bit)``,
    with the read's probability ``p`` of a 1, its kind and whether it is on the path
    recorded at its index. ``live`` (default ``ctx.live``) is whether the decomp makes the
    read here, for a read the path reaches only in some states."""
    live = ctx.live if live is None else live
    lo, slot = ctx.loose, ctx.slot
    i = lo.count[slot]
    bit = jnp.take(lo.bits[slot], i, mode="fill", fill_value=0)
    lo = lo._replace(
        count=lo.count.at[slot].add(1),
        p=lo.p.at[slot, i].set(p, mode="drop"),
        kind=lo.kind.at[slot, i].set(int(kind), mode="drop"),
        live=lo.live.at[slot, i].set(live, mode="drop"),
        line=lo.line.at[slot, i].set(line, mode="drop"),
    )
    return ctx._replace(loose=lo), bit


def per_slot(ctx: AiContext) -> bool:
    """The outcome is a per-slot layout's (``script_rolls`` of rank 2): its reads are keyed by
    order on the slot's path (``_slot_read``). Static."""
    return jnp.ndim(ctx.outcome.script_rolls) == 2


def _slot_read(ctx: AiContext, takes) -> tuple[AiContext, jnp.ndarray]:
    """The per-slot layout's next random read on the current slot's path (E1 Ruling R4;
    ``outcome`` docstring): ``(ctx, r)``, ``r`` the draw ``script_rolls[slot, reads[slot]]``
    (``RandNext() % 256``). ``takes(r)`` gives ``(p, drawn)``: the probability of the result
    ``r`` gives, and whether the decomp draws here at all (a read off the path, or of a
    ``MOVE_NONE`` slot, draws nothing). A drawn read counts in ``reads`` and multiplies its
    probability into ``weight``; one past the slot's draws is recorded in ``missing_read``,
    so the decision is refused (``policy._scores``) instead of reusing a draw."""
    draws = ctx.outcome.script_rolls[ctx.slot]  # uint8[slot_reads]
    k = ctx.reads[ctx.slot]
    r = jnp.take(draws, k, mode="fill", fill_value=0).astype(_I32)
    p, drawn = takes(r)
    drawn = drawn & ctx.live & (ctx.move != 0)
    missing = drawn & (k >= draws.shape[0])
    code = jnp.asarray(ctx.slot, _I32) + 1
    factor = jnp.where(drawn, p, 1.0).astype(jnp.float32)
    ctx = ctx._replace(
        reads=ctx.reads.at[ctx.slot].add(drawn.astype(_I32)),
        weight=ctx.weight.at[ctx.slot].multiply(factor),
        missing_read=jnp.where(missing & (ctx.missing_read == 0), code, ctx.missing_read).astype(
            _I32
        ),
    )
    return ctx, r


# --- The command functions, one per row of docs/ai/script_commands.md -------------


def if_target_is_partner(ctx: AiContext) -> AiContext:
    """``IfTargetIsPartner`` (trainer_ai.c:2630): never in a single battle."""
    return _jump(ctx, False)


def if_move_equal_to(ctx: AiContext, move: int) -> AiContext:
    """``IfMoveEqualTo`` (trainer_ai.c:834)."""
    return _jump(ctx, ctx.move == move)


def flag_move_damage_score(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``FlagMoveDamageScore`` (trainer_ai.c:1012): ``AI_NO_COMPARISON_MADE`` for a move
    that is not damaging; else ``AI_MOVE_IS_HIGHEST_DAMAGE`` when no slot of the true
    moveset estimates strictly more damage, ``AI_NOT_HIGHEST_DAMAGE`` otherwise."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        damage = _slot_damage(ctx.tables, ctx.state, ctx.outcome.sim_roll)
    else:
        damage = ctx.damage
    highest = ~jnp.any(damage > damage[ctx.slot])
    flag = jnp.where(highest, AI_MOVE_IS_HIGHEST_DAMAGE, AI_NOT_HIGHEST_DAMAGE)
    value = jnp.where(is_damaging(ctx.tables, ctx.move), flag, AI_NO_COMPARISON_MADE)
    return ctx._replace(calc_temp=value.astype(_I32))


def if_loaded_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedEqualTo`` (trainer_ai.c:786)."""
    return _jump(ctx, ctx.calc_temp == value)


def if_move_effectiveness_equals(ctx: AiContext, cls: TypeMulti) -> AiContext:
    """``IfMoveEffectivenessEquals`` (trainer_ai.c:1311)."""
    return _jump(ctx, effectiveness_class(ctx.tables, ctx.state, ctx.move) == int(cls))


def load_battler_ability(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadBattlerAbility`` (trainer_ai.c:1170): the attacker's true ability, or the
    defender's ability as the AI resolves it (``defender_ability``). Its coin flip,
    ``RandNext() & 1`` (:1194-1198), is fresh at every read that guesses
    (``ability_guessed``):
    - flat layout (scalar ``AiOutcome.ability_guess``): one guess for the decision,
      ``AiContext.player_ability`` (roark.md 8.2);
    - per-slot layout (E1 Rulings R1, R4; docs/ai/kaizo_roark.md 8.4): a guessing read is
      the slot's next random read (``_slot_read``): ability 1 when the draw's low bit is set,
      else ability 2 (:1195-1199), probability 1/2;
    - loose: its own draw."""
    if battler == Battler.ATTACKER:
        value = attacker_ability(ctx.tables, ctx.state)
    elif ctx.loose is not None:  # its own coin flip (trainer_ai.c:1191-1198)
        ctx, bit = _loose_read(ctx, 0.5, ReadKind.ABILITY)
        value = defender_ability(ctx.tables, ctx.state, bit)
    elif not per_slot(ctx):
        value = ctx.player_ability
    else:
        ctx, r = _slot_read(ctx, lambda r: (0.5, ctx.guessing))
        # RandNext() & 1 set: SPECIES_DATA_ABILITY_1, the species' slot 0 (:1191-1197).
        value = ctx.guess_abilities[jnp.where((r & 1) == 1, 0, 1)]
    return ctx._replace(calc_temp=jnp.asarray(value, _I32))


def load_type_from(ctx: AiContext, target: LoadType) -> AiContext:
    """``LoadTypeFrom`` (trainer_ai.c:937): the move's type, or the defender's or the
    attacker's true type 1 or 2 (``BattleMon_Get``, ``BATTLEMON_TYPE_1/2``, :945-959)."""
    state = ctx.state
    if target == LoadType.MOVE_TYPE:
        return ctx._replace(calc_temp=ctx.tables.move_type[ctx.move].astype(_I32))
    target = LoadType(target)
    side = 1 if target in (LoadType.ATTACKER_TYPE_1, LoadType.ATTACKER_TYPE_2) else 0
    types = state.types[side, state.active[side]].astype(_I32)
    index = 0 if target in (LoadType.DEFENDER_TYPE_1, LoadType.ATTACKER_TYPE_1) else 1
    return ctx._replace(calc_temp=types[index])


def if_temp_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfTempEqualTo`` (trainer_ai.c:1078)."""
    return _jump(ctx, ctx.calc_temp == value)


def go_to(ctx: AiContext) -> AiContext:
    """``GoTo`` (trainer_ai.c:2561): an unconditional jump."""
    return _jump(ctx, True)


def if_loaded_not_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedNotEqualTo`` (trainer_ai.c:798)."""
    return _jump(ctx, ctx.calc_temp != value)


def if_current_move_effect_equal_to(ctx: AiContext, effect: BattleEffect) -> AiContext:
    """``IfCurrentMoveEffectEqualTo`` (trainer_ai.c:1441)."""
    return _jump(ctx, move_effect(ctx.tables, ctx.move) == int(effect))


def _stage(ctx: AiContext, battler: Battler, stat: Stat) -> jnp.ndarray:
    """The battler's true stat stage in decomp units (engine + 6)."""
    return ctx.state.stages[_side(battler), stat].astype(_I32) + STAGE_NEUTRAL


def if_stat_stage_equal_to(ctx: AiContext, battler: Battler, stat: Stat, value: int) -> AiContext:
    """``IfStatStageEqualTo`` (trainer_ai.c:1495), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) == value)


def if_side_condition(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfSideCondition`` (trainer_ai.c:732): ``sideConditionsMask[side] & mask``, the
    battler's side having any condition of ``mask``, a union of ``SideCondition`` members
    (static; any other bit is refused at trace time). Each condition is on while its
    ``BattleState`` field is above 0."""
    mask = int(mask)
    known = sum(int(m) for m in SideCondition)
    assert mask and not mask & ~known, f"unsupported side condition mask {mask:#x}"
    side = _side(battler)
    on = jnp.asarray(False)
    for member, field in _SIDE_CONDITION_FIELDS.items():
        if mask & member:
            on = on | (getattr(ctx.state, field)[side] > 0)
    return _jump(ctx, on)


def check_battler_ability(ctx: AiContext, battler: Battler, expected: int) -> AiContext:
    """``CheckBattlerAbility`` (trainer_ai.c:1212-1265): ``AI_HAVE`` when the battler's
    ability as the AI sees it is ``expected``, ``AI_NOT_HAVE`` when it is another, and
    ``AI_UNKNOWN`` when the AI does not know it. The AI's own Pokemon: its current ability.
    The player's: the one the AI knows (learned, or a trapping ability, :1224-1233); else, for
    a species of two abilities, ability 1 when neither is ``expected`` and unknown when one
    is (:1238-1246), for a species of one, that one (:1247-1251). No draw."""
    if battler == Battler.ATTACKER:
        ability = attacker_ability(ctx.tables, ctx.state)
    else:
        true, known, base = _defender_abilities(ctx.tables, ctx.state)
        two = (base[0] != 0) & (base[1] != 0)
        neither = (base[0] != expected) & (base[1] != expected)
        guess = jnp.where(
            two, jnp.where(neither, base[0], 0), jnp.where(base[0] != 0, base[0], base[1])
        )
        ability = jnp.where(known, true, guess)
    value = jnp.where(
        ability == 0, AI_UNKNOWN, jnp.where(ability == expected, AI_HAVE, AI_NOT_HAVE)
    )
    return ctx._replace(calc_temp=jnp.asarray(value, _I32))


def load_spikes_layers(ctx: AiContext, battler: Battler, condition: SideCondition) -> AiContext:
    """``LoadSpikesLayers`` (trainer_ai.c:1948-1966): the battler's side's Spikes or Toxic
    Spikes layers (``sideConditions[side].spikesLayers`` / ``.toxicSpikesLayers``;
    ``BattleState.spikes`` / ``.toxic_spikes``). Any other condition is refused at trace
    time."""
    field = {SideCondition.SPIKES: "spikes", SideCondition.TOXIC_SPIKES: "toxic_spikes"}[
        SideCondition(int(condition))
    ]
    return ctx._replace(calc_temp=getattr(ctx.state, field)[_side(battler)].astype(_I32))


def count_alive_party_battlers(ctx: AiContext, battler: Battler) -> AiContext:
    """``CountAlivePartyBattlers`` (trainer_ai.c:1126): the side's living party members
    other than the active one."""
    state, side = ctx.state, _side(battler)
    alive = (state.species[side] > 0) & (state.hp[side] > 0)
    alive = alive & (jnp.arange(alive.shape[0]) != state.active[side])
    return ctx._replace(calc_temp=jnp.sum(alive.astype(_I32)))


def add_to_move_score(ctx: AiContext, value: int) -> AiContext:
    """``AddToMoveScore`` (trainer_ai.c:576-586): ``moveScore[slot] += val``, then a
    negative score becomes 0. ``moveScore`` is ``s8`` (include/battle/ai_context.h:14), so
    the sum wraps to -128..127 before the clamp: 127 + 2 is -127, then 0. No Roark path
    comes near it (his largest score is about 108); the wrap keeps a later one exact."""
    current = ctx.scores[ctx.slot]
    new = jnp.maximum((current + value + 128) % 256 - 128, 0)
    return ctx._replace(scores=ctx.scores.at[ctx.slot].set(jnp.where(ctx.live, new, current)))


def pop_or_end(ctx: AiContext) -> AiContext:
    """``PopOrEnd`` (trainer_ai.c:2568): Roark's scripts never push, so this ends the
    slot for this flag."""
    return ctx._replace(live=jnp.asarray(False))


def if_current_move_kills(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``IfCurrentMoveKills`` (trainer_ai.c:1525): the move is damaging and the
    defender's true HP is at most its estimated damage."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        roll = ctx.outcome.sim_roll[ctx.slot]
    else:
        roll = SIM_ROLL_MAX
    damage = ai_damage(ctx.tables, ctx.state, ctx.move, roll)
    state = ctx.state
    hp = state.hp[0, state.active[0]].astype(_I32)
    return _jump(ctx, is_damaging(ctx.tables, ctx.move) & (hp <= damage))


def if_random_less_than(ctx: AiContext, n: int, line: int) -> AiContext:
    """``IfRandomLessThan`` (trainer_ai.c:528-538): ``RandNext() % 256 < n``. ``line`` is the
    read's ``script.s`` line (``outcome.SITE_N``, checked).
    - flat layout: ``AiOutcome.script_rolls[site] == 1`` (probability n/256, roark.md 8.3),
      ``AiContext.sites`` giving the line's index (a line no vanilla move reaches reads an
      out-of-range index, which never rolls 1, and is never on a vanilla path);
    - per-slot layout (E1 Ruling R4): the slot's next random read (``_slot_read``) jumps when
      its draw is below ``n``, probability n/256; drawn on the path only;
    - loose (module docstring): the read's own draw, with probability ``n``/256."""
    assert SITE_N.get(int(line)) == int(n), f"IfRandomLessThan {n} at script.s:{line}"
    if ctx.loose is not None:
        ctx, bit = _loose_read(ctx, n / 256, ReadKind.ROLL, line=int(line))
        return _jump(ctx, bit == 1)
    if per_slot(ctx):
        p = n / 256
        ctx, r = _slot_read(ctx, lambda r: (jnp.where(r < n, p, 1 - p), jnp.asarray(True)))
        return _jump(ctx, r < n)
    site = ctx.sites[ctx.slot, SITE_LINES.index(line)]
    roll = jnp.take(ctx.outcome.script_rolls, site, mode="fill", fill_value=0)
    return _jump(ctx, roll == 1)


def _knows(ctx: AiContext, battler: Battler, move: int) -> jnp.ndarray:
    state = ctx.state
    if battler == Battler.ATTACKER:
        moves = state.moves[1, state.active[1]].astype(_I32)
    else:
        moves = ctx.seen_moves  # AI_CONTEXT.battlerMoves (trainer_ai.c:1665)
    return jnp.any(moves == move)


def if_move_known(ctx: AiContext, battler: Battler, move: int) -> AiContext:
    """``IfMoveKnown`` (trainer_ai.c:1627): the attacker's true moveset, or the moves the
    AI has seen from the defender."""
    return _jump(ctx, _knows(ctx, battler, move))


def _hp_percent(ctx: AiContext, battler: Battler) -> jnp.ndarray:
    state, side = ctx.state, _side(battler)
    mon = state.active[side]
    return state.hp[side, mon].astype(_I32) * 100 // state.max_hp[side, mon].astype(_I32)


def if_hp_percent_less_than(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentLessThan`` (trainer_ai.c:588): ``curHP * 100 / maxHP < percent``."""
    return _jump(ctx, _hp_percent(ctx, battler) < percent)


def if_stat_stage_greater_than(
    ctx: AiContext, battler: Battler, stat: Stat, value: int
) -> AiContext:
    """``IfStatStageGreaterThan`` (trainer_ai.c:1480), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) > value)


def if_hp_percent_greater_than(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentGreaterThan`` (trainer_ai.c:603): ``curHP * 100 / maxHP > percent``."""
    return _jump(ctx, _hp_percent(ctx, battler) > percent)


def load_is_first_turn_in_battle(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadIsFirstTurnInBattle`` (trainer_ai.c:2480): TRUE (1) unless
    ``fakeOutTurnNumber < totalTurns`` (``BattleState.fake_out_turn``, ``turn``)."""
    state = ctx.state
    first = state.fake_out_turn[_side(battler)].astype(_I32) >= state.turn.astype(_I32)
    return ctx._replace(calc_temp=first.astype(_I32))


def if_move_not_known(ctx: AiContext, battler: Battler, move: int) -> AiContext:
    """``IfMoveNotKnown`` (trainer_ai.c:1683): the negation of ``IfMoveKnown``."""
    return _jump(ctx, ~_knows(ctx, battler, move))


def if_hp_percent_equal_to(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentEqualTo`` (trainer_ai.c:618): ``curHP * 100 / maxHP == percent``, a
    ``u32`` with integer division (:626)."""
    return _jump(ctx, _hp_percent(ctx, battler) == percent)


def if_hp_percent_not_equal_to(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentNotEqualTo`` (trainer_ai.c:633): ``curHP * 100 / maxHP != percent``
    (:641)."""
    return _jump(ctx, _hp_percent(ctx, battler) != percent)


# The engine status of each MonCondition group (one status at a time, ids.Status).
_CONDITION_GROUPS = (
    (MonCondition.SLEEP, Status.SLEEP),
    (MonCondition.POISON, Status.POISON),
    (MonCondition.BURN, Status.BURN),
    (MonCondition.FREEZE, Status.FREEZE),
    (MonCondition.PARALYSIS, Status.PARALYSIS),
    (MonCondition.TOXIC, Status.TOXIC),
)


def _condition_statuses(mask: int) -> tuple[int, ...]:
    """The ``ids.Status`` values whose decomp status bits meet ``mask``. ``mask`` must be a
    union of whole ``MonCondition`` groups: a part of the sleep bits or a toxic counter
    bit is not a function of the engine's status alone, and is refused at trace time."""
    mask = int(mask)
    assert not mask & ~MonCondition.ANY, f"unsupported status mask {mask:#x}"
    statuses = []
    for group, status in _CONDITION_GROUPS:
        assert mask & group in (0, group), f"status mask {mask:#x} splits {group.name}"
        if mask & group:
            statuses.append(int(status))
    return tuple(statuses)


def if_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfStatus`` (trainer_ai.c:648): the battler's true status meets ``mask``
    (``battleMons[battler].status & mask``, :657), a union of ``MonCondition`` groups. An
    asleep Pokemon meets the whole ``SLEEP`` group: the battle tests sleep as
    ``status & MON_CONDITION_SLEEP`` (e.g. battle_lib.c:1845)."""
    side = _side(battler)
    status = ctx.state.status[side, ctx.state.active[side]].astype(_I32)
    return _jump(ctx, jnp.isin(status, jnp.asarray(_condition_statuses(mask), _I32)))


def if_not_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfNotStatus`` (trainer_ai.c:662-674): the negation of ``IfStatus``."""
    ctx = if_status(ctx, battler, mask)
    return ctx._replace(jump=~ctx.jump)


def if_stat_stage_less_than(ctx: AiContext, battler: Battler, stat: Stat, value: int) -> AiContext:
    """``IfStatStageLessThan`` (trainer_ai.c:1465), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) < value)


def if_field_conditions_mask(ctx: AiContext, mask: int) -> AiContext:
    """``IfFieldConditionsMask`` (trainer_ai.c:1936-1946): ``fieldConditionsMask & mask``.
    Only ``FieldCondition.TRICK_ROOM`` is supported (static; refused otherwise): Trick
    Room is on while ``BattleState.trick_room`` is nonzero."""
    assert int(mask) == FieldCondition.TRICK_ROOM, f"unsupported field mask {int(mask):#x}"
    return _jump(ctx, ctx.state.trick_room != 0)


def load_current_weather(ctx: AiContext) -> AiContext:
    """``LoadCurrentWeather`` (trainer_ai.c:1414-1439): the ``AiWeather`` of the raw
    ``fieldConditionsMask`` through ``WEATHER_IS_*`` (include/battle/common.h:68-72), with
    **no** Cloud Nine or Air Lock check. Each ``WEATHER_IS_*`` takes both the temporary and
    the permanent bit (``FIELD_CONDITION_SANDSTORM`` = ``_TEMP | _PERM``,
    include/constants/battle/condition.h:112-128). The battle holds one weather at a time:
    a weather move clears ``FIELD_CONDITION_WEATHER`` before it sets its own bit
    (effect_script_0115.s:8-9, Sandstorm), so the decomp's last-match order does not matter
    and the engine's single ``BattleState.weather`` gives the value."""
    w = ctx.state.weather
    value = jnp.select(
        [
            w == Weather.RAIN,
            w == Weather.SANDSTORM,
            w == Weather.SUN,
            w == Weather.HAIL,
            w == Weather.FOG,
        ],
        [
            AiWeather.RAINING,
            AiWeather.SANDSTORM,
            AiWeather.SUNNY,
            AiWeather.HAILING,
            AiWeather.DEEP_FOG,
        ],
        AiWeather.CLEAR,
    )
    return ctx._replace(calc_temp=jnp.asarray(value, _I32))


def if_speed_compare_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfSpeedCompareEqualTo`` (trainer_ai.c:1102-1112):
    ``BattleSystem_CompareBattlerSpeed(attacker = the AI, defender = the player, TRUE) ==
    value``, which is ``order.compare_speed`` with ``battler1 = 1``:

    - Quick Claw is read through ``speedRand`` with ``ignoreQuickClaw`` TRUE too
      (battle_lib.c:1278-1286, :1344-1352; the flag only skips the ``moveEffectsData``
      write). The AI decides before the turn's redraw (battle_controller_player.c:853-857),
      so it reads ``BattleState.qc_roll``, the previous turn's draw.
    - ``ignoreQuickClaw`` skips the priority read (:1375-1399): both priorities are 0.
    - The tie's ``RandNext() & 1`` is not drawn (plan K3c Ruling 3). At equal speed every
      branch gives FASTER or TIE, never SLOWER (:1401-1447), so ``== SLOWER`` does not
      depend on it: ``order.battler_slower`` with ``tie_roll = 0``. ``value`` must be
      ``COMPARE_SPEED_SLOWER`` (asserted at trace time), as at every transcribed compare on the
      paths (script.s:1864, :2070, :2357, :2650, :2944). The decomp still consumes the draw (K3d).

    Loose (module docstring): the tie is its own draw (``ReadKind.SPEED_TIE``), on the path
    where the decomp makes it (equal speed in a branch that reads it, which is where
    ``compare_speed`` gives TIE for a 1), and the decomp's three-valued result is compared
    with ``value``."""
    assert int(value) == COMPARE_SPEED_SLOWER, f"unsupported speed compare {value}"
    tables, state = ctx.tables, ctx.state
    if ctx.loose is not None:
        reads = compare_speed(tables, state, 1, 1, state.qc_roll) == COMPARE_SPEED_TIE
        ctx, bit = _loose_read(ctx, 0.5, ReadKind.SPEED_TIE, live=ctx.live & reads)
        return _jump(ctx, compare_speed(tables, state, 1, bit, state.qc_roll) == value)
    return _jump(ctx, battler_slower(tables, state, 1, 0, state.qc_roll))


def _knows_effect(ctx: AiContext, battler: Battler, effect: int) -> jnp.ndarray:
    state = ctx.state
    if battler == Battler.ATTACKER:
        moves = state.moves[1, state.active[1]].astype(_I32)
    else:
        moves = ctx.seen_moves  # AI_CONTEXT.battlerMoves
    return jnp.any((moves != 0) & (move_effect(ctx.tables, moves) == effect))


def if_move_effect_known(ctx: AiContext, battler: Battler, effect: int) -> AiContext:
    """``IfMoveEffectKnown`` (trainer_ai.c:1739): a nonzero move whose effect
    (``move_effect``, the game's own) is ``effect``, in the attacker's true moveset
    (:1752-1753) or among the defender's moves the AI has seen (:1765-1766)."""
    return _jump(ctx, _knows_effect(ctx, battler, effect))


def if_move_effect_not_known(ctx: AiContext, battler: Battler, effect: int) -> AiContext:
    """``IfMoveEffectNotKnown`` (trainer_ai.c:1781): the negation of ``IfMoveEffectKnown``."""
    return _jump(ctx, ~_knows_effect(ctx, battler, effect))


def if_attacker_has_no_damaging_moves(ctx: AiContext) -> AiContext:
    """``IfAttackerHasNoDamagingMoves`` (trainer_ai.c:913-929): no move of the attacker's
    true moveset is nonzero with ``MOVE_DATA(move).power`` nonzero (:920-921). Power 1
    counts, unlike ``is_damaging``'s ``> 1``."""
    state = ctx.state
    moves = state.moves[1, state.active[1]].astype(_I32)
    damaging = (moves != 0) & (ctx.tables.move_power[moves] != 0)
    return _jump(ctx, ~jnp.any(damaging))


def load_battler_previous_move(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadBattlerPreviousMove`` (trainer_ai.c:1068-1076): ``calcTemp =
    movePrevByBattler[battler]`` (``BattleState.last_move`` of the battler's side; the
    selected move on success, else ``MOVE_NONE``, battle_controller_player.c:3944-3949)."""
    return ctx._replace(calc_temp=ctx.state.last_move[_side(battler)].astype(_I32))


def load_power_of_loaded_move(ctx: AiContext) -> AiContext:
    """``LoadPowerOfLoadedMove`` (trainer_ai.c:2526-2530): ``calcTemp =
    MOVE_DATA(calcTemp).power``."""
    return ctx._replace(calc_temp=ctx.tables.move_power[ctx.calc_temp].astype(_I32))


def if_volatile_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfVolatileStatus`` (trainer_ai.c:676-686): ``statusVolatile & mask``. ``mask`` must
    be one ``VolatileCondition`` (refused at trace time otherwise), read from its field."""
    field = _VOLATILE_FIELDS[VolatileCondition(int(mask))]
    return _jump(ctx, getattr(ctx.state, field)[_side(battler)] > 0)


def if_move_effect(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfMoveEffect`` (trainer_ai.c:704-714): ``moveEffectsMask & mask``. ``mask`` must be
    one ``MoveEffectMask`` (refused at trace time otherwise), read from its field."""
    field = _MOVE_EFFECT_FIELDS[MoveEffectMask(int(mask))]
    return _jump(ctx, getattr(ctx.state, field)[_side(battler)] > 0)


def load_gender(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadGender`` (trainer_ai.c:2470-2477): ``battleMons[battler].gender``
    (``BattleState.gender`` of the side's active Pokemon; ``Gender``)."""
    side = _side(battler)
    state = ctx.state
    return ctx._replace(calc_temp=state.gender[side, state.active[side]].astype(_I32))


def load_defender_last_used_move_class(ctx: AiContext) -> AiContext:
    """``LoadDefenderLastUsedMoveClass`` (trainer_ai.c:2050-2054):
    ``MOVE_DATA(movePrevByBattler[defender]).class`` (``BattleState.last_move[0]``; 0
    physical, 1 special, 2 status). ``movePrevByBattler`` is the selected move on success,
    else ``MOVE_NONE`` (battle_controller_player.c:3944-3949), whose class is physical
    (res/moves/none/data.json); for Nature Power it is Nature Power."""
    move = ctx.state.last_move[0].astype(_I32)
    return ctx._replace(calc_temp=ctx.tables.move_category[move].astype(_I32))


def load_turn_count(ctx: AiContext) -> AiContext:
    """``LoadTurnCount`` (trainer_ai.c:931-935): ``calcTemp = totalTurns``
    (``BattleState.turn`` at a decision, engine/state.py; the battle increments it once a
    turn ends, battle_controller_player.c:1852), 0 on the battle's first turn."""
    return ctx._replace(calc_temp=ctx.state.turn.astype(_I32))


def load_current_move_effect(ctx: AiContext) -> AiContext:
    """``LoadCurrentMoveEffect`` (trainer_ai.c:1164): ``MOVE_DATA(move).effect``."""
    return ctx._replace(calc_temp=move_effect(ctx.tables, ctx.move))


def if_loaded_in_table(ctx: AiContext, table: tuple[int, ...]) -> AiContext:
    """``IfLoadedInTable`` (trainer_ai.c:858-874): ``calcTemp`` equals one of ``table``'s
    values (the script's list up to its 0xFFFFFFFF end); an empty table never jumps."""
    if not table:
        return _jump(ctx, False)
    return _jump(ctx, jnp.isin(ctx.calc_temp, jnp.asarray(table, _I32)))


def if_loaded_not_in_table(ctx: AiContext, table: tuple[int, ...]) -> AiContext:
    """``IfLoadedNotInTable`` (trainer_ai.c:876-891): ``calcTemp`` equals none of
    ``table``'s values (the script's list up to its 0xFFFFFFFF end)."""
    if not table:
        return _jump(ctx, True)
    return _jump(ctx, ~jnp.isin(ctx.calc_temp, jnp.asarray(table, _I32)))
