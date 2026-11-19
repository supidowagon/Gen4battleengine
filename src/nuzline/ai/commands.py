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
read bound, no dropped tie. Each ``RandNext`` of ``TrainerAI_CalcDamage`` (Psywave's and
Magnitude's cases, trainer_ai.c:3014-3048; ``outcome`` docstring) is a loose read of its own
too, a digit (``ReadKind.CALC_PSYWAVE``: the draw's ``% 11``, ``CALC_MAGNITUDE``: its bucket)
with its class law (``outcome.CALC_DRAW_P``), one per move each damage command estimates, on
the path when that move draws; the command then computes its result from the digits as the
decomp does. The exact layout instead makes one read per command, with the result's
probability (``_calc_read``), and drops the draws whose result is never read.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.outcome import (
    CALC_DRAW_CLASSES,
    CALC_DRAW_P,
    CALC_DRAWS,
    CALC_MAGNITUDE,
    CALC_NONE,
    CALC_PSYWAVE,
    READ_DRAWS,
    SITE_LINES,
    SITE_N,
    AiOutcome,
    calc_draw_kind,
    calc_kind_class,
    line_sites,
)
from nuzline.ai.switching import (
    INEFFECTIVE,
    MAGNET_RISE,
    SUPER_EFFECTIVE,
    apply_type_chart,
    defender_chart_flags,
    type_chart_flags,
    wonder_guard_bit,
)
from nuzline.data.tables import BATTLE_TYPE_DOUBLES as _BATTLE_TYPE_DOUBLES
from nuzline.data.tables import Tables
from nuzline.engine.actions import usable_moves
from nuzline.engine.damage import (
    AI_MAGNITUDE,
    AI_PSYWAVE,
    ai_damage_for_roll,
    ai_move_type,
    decomp_abilities,
)
from nuzline.engine.format import format_of, party, side_of
from nuzline.engine.ids import AbilityId, Stat, Status, Weather
from nuzline.engine.order import (
    COMPARE_SPEED_FASTER,
    COMPARE_SPEED_SLOWER,
    COMPARE_SPEED_TIE,
    battler_slower,
    compare_speed,
)
from nuzline.engine.state import MOVES, BattleState, held_item, record_seen_move

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
    MIST = 1 << 6  # :94; BattleState.mist (turns left), B12-23
    STEALTH_ROCK = 1 << 7  # BattleState.stealth_rock
    TOXIC_SPIKES = 1 << 10  # :98; BattleState.toxic_spikes (layers, LoadSpikesLayers)
    TAILWIND = (1 << 8) | (1 << 9)  # :96-97, :104; BattleState.tailwind (turns left)
    FUTURE_SIGHT = 1 << 4  # :92; BattleState.future_sight (per side)
    LUCKY_CHANT = (
        (1 << 12) | (1 << 13) | (1 << 14)
    )  # :100-105; BattleState.lucky_chant (turns left)


_SIDE_CONDITION_FIELDS = {
    SideCondition.REFLECT: "reflect",
    SideCondition.LIGHT_SCREEN: "light_screen",
    SideCondition.SAFEGUARD: "safeguard",
    SideCondition.MIST: "mist",
    SideCondition.STEALTH_ROCK: "stealth_rock",
    SideCondition.TAILWIND: "tailwind",
    SideCondition.SPIKES: "spikes",  # layers; on while above 0 (Expert_ForceSwitch, :2591)
    SideCondition.TOXIC_SPIKES: "toxic_spikes",
    SideCondition.FUTURE_SIGHT: "future_sight",
    SideCondition.LUCKY_CHANT: "lucky_chant",
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
    # MON_CONDITION_ANY_POISON (condition.h:24): poison, toxic and the toxic counter, which is
    # set only with the toxic bit, so the group is poison or toxic (E2 batch 12, B12-18).
    ANY_POISON = (1 << 3) | (1 << 7) | (0xF << 8)


class FieldCondition(IntEnum):
    """``IfFieldConditionsMask`` masks Roark's scripts use (``FIELD_CONDITION_*``,
    include/constants/battle/condition.h:123-125, :144)."""

    TRICK_ROOM = (1 << 16) | (1 << 17) | (1 << 18)  # BattleState.trick_room
    GRAVITY = (1 << 12) | (1 << 13) | (1 << 14)  # :119-121, :142-143; BattleState.gravity (G-67)


class VolatileCondition(IntEnum):
    """``IfVolatileStatus`` masks the transcribed scripts test (``VOLATILE_CONDITION_*``,
    include/constants/battle/condition.h), each read from its ``BattleState`` field
    (``if_volatile_status``)."""

    CONFUSION = (1 << 0) | (1 << 1) | (1 << 2)  # :32-34, :65; BattleState.confusion
    ATTRACT = (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19)  # :48-51, :70; .infatuated
    CURSE = 1 << 28  # :60; BattleState.curse
    SUBSTITUTE = 1 << 24  # :56; BattleState.substitute_hp (its HP left, 0 = none)
    MEAN_LOOK = 1 << 26  # :58; BattleState.mean_look
    FOCUS_ENERGY = 1 << 20  # :52; BattleState.focus_energy (B12-4)
    TORMENT = 1 << 31  # :63; BattleState.torment (E3-2-1)


_VOLATILE_FIELDS = {
    VolatileCondition.CONFUSION: "confusion",
    VolatileCondition.ATTRACT: "infatuated",
    VolatileCondition.CURSE: "curse",
    VolatileCondition.SUBSTITUTE: "substitute_hp",
    VolatileCondition.MEAN_LOOK: "mean_look",
    VolatileCondition.FOCUS_ENERGY: "focus_energy",
    VolatileCondition.TORMENT: "torment",
}


class MoveEffectMask(IntEnum):
    """``IfMoveEffect`` masks the transcribed scripts test (``MOVE_EFFECT_*``,
    include/constants/battle/moves.h)."""

    LEECH_SEED = 1 << 2  # :10; BattleState.leech_seed (the seeded battler's side)
    LOCK_ON = (1 << 3) | (1 << 4)  # :11-12, :41; BattleState.lock_on (turns)
    PERISH_SONG = 1 << 5  # :13; BattleState.perish_song
    YAWN = (1 << 11) | (1 << 12)  # :19-20, :42; BattleState.yawn (turns)
    INGRAIN = 1 << 10  # :18; BattleState.ingrain
    IMPRISON = 1 << 13  # :21; BattleState.imprison (the Imprison user)
    AQUA_RING = 1 << 24  # :37; BattleState.aqua_ring
    IMPRISONED = 1 << 30  # :38; BattleState.imprisoned
    POWER_TRICK = 1 << 23  # :31; BattleState.power_trick (B12-10)
    EMBARGO = 1 << 26  # :34; BattleState.embargo (set and cleared with embargoTurns, B13-31)
    MAGNET_RISE = 1 << 27  # :31; BattleState.magnet_rise (the bit, not the turns; G-66)
    ABILITY_SUPPRESSED = 1 << 21  # :29; BattleState.ability_suppressed (Gastro Acid, E3-2-6a)
    MIRACLE_EYE = 1 << 22  # moves.h:30; BattleState.miracle_eye (Basic_CheckMiracleEye, E3-B2-10)


_MOVE_EFFECT_FIELDS = {
    MoveEffectMask.LEECH_SEED: "leech_seed",
    MoveEffectMask.LOCK_ON: "lock_on",
    MoveEffectMask.PERISH_SONG: "perish_song",
    MoveEffectMask.YAWN: "yawn",
    MoveEffectMask.INGRAIN: "ingrain",
    MoveEffectMask.IMPRISON: "imprison",
    MoveEffectMask.AQUA_RING: "aqua_ring",
    MoveEffectMask.IMPRISONED: "imprisoned",
    MoveEffectMask.POWER_TRICK: "power_trick",
    MoveEffectMask.EMBARGO: "embargo",
    MoveEffectMask.MAGNET_RISE: "magnet_rise",
    MoveEffectMask.ABILITY_SUPPRESSED: "ability_suppressed",
    MoveEffectMask.MIRACLE_EYE: "miracle_eye",
}
BATTLE_TYPE_DOUBLES = int(_BATTLE_TYPE_DOUBLES)  # include/constants/battle.h:27
# include/constants/battle.h:33: never set in a trainer battle of the story (Basic_CheckEmbargo).
BATTLE_TYPE_FRONTIER = 1 << 7


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
    EVA_UP = 16  # Double Team (Basic_CheckHighStatStage_Evasion :143; Expert_StatusEvasionUp :1639)
    ATK_DOWN = 18
    DEF_DOWN = 19
    SPEED_DOWN = 20
    EVA_DOWN = 24
    ALL_FAINT_3_TURNS = 114  # Perish Song (Basic_CheckPerishSong :199)
    SURVIVE_WITH_1_HP = 116  # Endure (Expert_Endure :1704)
    GIVE_GROUND_IMMUNITY = 252  # Magnet Rise (Basic_CheckMagnetRise :280; Expert_MagnetRise :1795)
    GRAVITY = 215  # Gravity (Basic_CheckGravityActive :254; Expert_Gravity :1759)
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
    PROTECT = 111  # Expert_ToxicLeechSeed's IfMoveEffectKnown (script.s:2700); Protect, Detect
    DEF_UP_DOUBLE_ROLLOUT_POWER = 156  # Defense Curl (Expert_Protect, script.s:3554)
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
    DOUBLE_SPEED_3_TURNS = 225  # Tailwind
    DOUBLE_POWER_IF_MOVING_SECOND = 230  # Payback, Revenge
    HALVE_HP = 40  # Super Fang
    COUNTER = 89  # Counter
    DAMAGE_WHILE_ASLEEP = 92  # Snore
    USE_RANDOM_LEARNED_MOVE_SLEEP = 97  # Sleep Talk
    INCREASE_POWER_WITH_LESS_HP = 99  # Flail, Reversal
    MIRROR_COAT = 144  # Mirror Coat
    ALWAYS_FLINCH_FIRST_TURN_ONLY = 158  # Fake Out
    LOWER_OWN_ATK_AND_DEF = 182  # Superpower, Close Combat
    EAT_BERRY = 224  # Pluck, Bug Bite
    DAMAGE_FLAT_40 = 41  # 40_DAMAGE_FLAT: Dragon Rage
    LEVEL_DAMAGE_FLAT = 87  # Seismic Toss, Night Shade
    RANDOM_DAMAGE_1_TO_150_LEVEL = 88  # Psywave; Kaizo Triple Axel, Fury Cutter
    POWER_BASED_ON_FRIENDSHIP = 121  # Return
    RANDOM_POWER_MAYBE_HEAL = 122  # Present
    POWER_BASED_ON_LOW_FRIENDSHIP = 123  # Frustration
    DAMAGE_FLAT_20 = 130  # 20_DAMAGE_FLAT: SonicBoom
    INCREASE_POWER_WITH_WEIGHT = 196  # Low Kick, Grass Knot
    RANDOM_POWER_BASED_ON_IVS = 135  # Hidden Power
    ONE_HIT_KO = 38  # Guillotine, Horn Drill, Fissure
    SWITCH_HIT = 228  # U-turn
    AVERAGE_HP = 91  # Pain Split
    ENCORE = 90  # Encore (E2 batch 11, B11-4)
    PREVENT_STATUS = 124  # Safeguard
    MAX_ATK_LOSE_HALF_MAX_HP = 142  # Belly Drum
    HIT_LAST_WHIFF_IF_HIT = 170  # Focus Punch
    SET_HP_EQUAL_TO_USER = 189  # Endeavor
    BLIZZARD = 260  # Kaizo Blizzard, Ice Ball, Sheer Cold
    FORCE_SWITCH = 28  # Roar, Whirlwind
    HEAL_HALF_REMOVE_FLYING_TYPE = 214  # Roost
    METAL_BURST = 227  # Metal Burst, Bide (Kaizo)
    HIT_IN_3_TURNS = 148  # Future Sight, Doom Desire
    ACC_DOWN = 23  # Sand-Attack, SmokeScreen, Flash
    EVA_UP_2_MINIMIZE = 108  # Minimize
    PREVENT_ESCAPE = 106  # Mean Look, Block, Spider Web
    REMOVE_HELD_ITEM = 188  # Knock Off
    MAKE_SHARED_MOVES_UNUSEABLE = 192  # Imprison
    USE_LAST_USED_MOVE = 242  # Copycat
    GROUND_TRAP_USER_CONTINUOUS_HEAL = 181  # Ingrain
    PASS_STATS_AND_STATUS = 127  # Baton Pass
    BOOST_ALLY_POWER_BY_50_PERCENT = 176  # Helping Hand
    # E2 batch 12, the stat family (B12-11).
    RESET_STAT_CHANGES = 25  # Haze
    CRIT_UP_2 = 47  # Focus Energy
    PREVENT_STAT_REDUCTION = 46  # Mist (B12-23)
    COPY_STAT_CHANGES = 143  # Psych Up
    SP_ATK_UP_CAUSE_CONFUSION = 166  # Flatter
    ATK_DEF_DOWN = 205  # Tickle
    DEF_SPD_UP = 206  # Cosmic Power, Defend Order; Kaizo's Stockpile
    RANDOM_STAT_UP_2 = 226  # Acupressure
    SWAP_ATK_DEF = 238  # Power Trick
    SWAP_ATK_SP_ATK_STAT_CHANGES = 243  # Power Swap
    SWAP_DEF_SP_DEF_STAT_CHANGES = 244  # Guard Swap
    SP_ATK_DOWN_2_OPPOSITE_GENDER = 265  # Captivate
    # E2 batch 12, the healing family (B12-18).
    REST = 37  # Rest
    CURE_PARTY_STATUS = 102  # Heal Bell, Aromatherapy
    HEAL_HALF_MORE_IN_SUN = 132  # Morning Sun, Synthesis, Moonlight
    HEAL_IN_3_TURNS = 179  # Wish
    HEAL_STATUS = 193  # Refresh
    TRANSFER_STATUS = 234  # Psycho Shift
    DOUBLE_POWER_IF_HIT = 185  # Avalanche, Revenge (Expert_MetalBurst, script.s:5049)
    PRIORITY_NEG_1_BYPASS_ACCURACY = 78  # Vital Throw (Expert_MetalBurst, script.s:5051)
    SET_SUBSTITUTE = 79  # Substitute
    STATUS_POISON = 66  # Poison Powder (Expert_Substitute's last-move table, script.s:3019)

    TOXIC_SPIKES = 249
    RECOIL_BURN_HIT = 253  # Flare Blitz
    RECOIL_PARALYZE_HIT = 262
    STEALTH_ROCK = 266
    RECOIL_HALF = 269  # Head Smash
    # E2 batch 10: the charge-turn moves (Expert_ChargeTurnNoInvuln, script.s:3933-3973;
    # Expert_ChargeTurnWithInvuln / Expert_ShadowForce, :3992-4072) and the two hits on a
    # flying target, which no script names.
    CHARGE_TURN_HIGH_CRIT = 39  # Razor Wind
    CHARGE_TURN_HIGH_CRIT_FLINCH = 75  # Sky Attack
    NEXT_ATTACK_ALWAYS_HITS = 94  # Lock-On, Mind Reader (Expert_ChargeTurnWithInvuln, :4053)
    CHARGE_TURN_DEF_UP = 145  # Skull Bash
    FLINCH_DOUBLE_DAMAGE_FLY_OR_BOUNCE = 146  # Twister
    SKIP_CHARGE_TURN_IN_SUN = 151  # SolarBeam
    FLY = 155
    HIT_FLY = 207  # Sky Uppercut
    DIVE = 255
    DIG = 256
    BOUNCE = 263
    SHADOW_FORCE = 272
    # E2 batch 13 (B13-23..): the variable-power and secondary hits, Recycle.
    POISON_MULTI_HIT = 77  # Twineedle: no script entry
    HIT_THREE_TIMES = 104  # Triple Kick (Kaizo: Rock Wrecker too): no script entry
    UNUSED_141 = 141  # Kaizo Hurricane (be_seq 141): no script entry
    BEAT_UP = 154  # no script entry
    RECYCLE = 184  # Basic_CheckCanRecycle (script.s:238), Expert_Recycle (:1737)
    DECREASE_POWER_WITH_LESS_USER_HP = 190  # Water Spout: Expert_WaterSpout (:1742)
    HIGH_CRITICAL_BURN_HIT = 200  # Blaze Kick: Expert_HighCritical (:1748)
    CHANGE_TYPE_WITH_WEATHER = 203  # Weather Ball: no script entry
    HIGH_CRITICAL_POISON_HIT = 209  # Poison Tail, Cross Poison: Expert_HighCritical (:1754)
    DOUBLE_POWER_HEAL_SLEEP = 217  # Wake-Up Slap: Expert_WakeUpSlap (:1761)
    POWER_BASED_ON_LOW_SPEED = 219  # Gyro Ball: NonStandard (:256), Expert_GyroBall (:1763)
    NATURAL_GIFT = 222  # Basic_CheckNaturalGift (:258)
    DOUBLE_POWER_IF_TARGET_HIT = 231  # Assurance: Expert_Assurance (:1774)
    PREVENT_ITEM_USE = 232  # Embargo: Basic_CheckEmbargo (:262), Expert_Embargo (:1775), B13-32
    HIGHER_POWER_WHEN_LOW_PP = 235  # Trump Card: NonStandard (:265), Expert_TrumpCard (:1778)
    INCREASE_POWER_WITH_MORE_STAT_UP = 245  # Punishment: NonStandard (:274), Expert (:1788)
    JUDGEMENT = 268  # Judgment: no script entry
    # E3 batch 1 (E3-1-1..E3-1-5).
    LOWER_SP_ATK_HIT = 71  # Mist Ball, Kaizo Mystical Fire: no script entry
    DO_NOTHING = 85  # Splash, Kaizo Thief (power 40): no Basic or Expert entry; Encore table :3220
    DISABLE = 86  # Basic_CheckCannotDisable (:184); Expert_Disable (:1687)
    FLEE_FROM_WILD_BATTLE = 153  # Teleport: Basic :221 ScoreMinus10
    SPEED_DOWN_HIT = 218  # Hammer Arm: Expert_HammerArm (:1762)
    # E3 batch 1 round 2 (E3-2-1 .. E3-2-4).
    TORMENT = 165  # Basic_CheckTorment (:229); no Expert entry
    USE_MOVE_FIRST = 241  # Me First: Expert_MeFirst (:1784); Risky :6560
    SET_ABILITY_TO_INSOMNIA = 247  # Worry Seed: Basic :276; Expert_WorrySeed (:1790)
    TRICK_ROOM = 259  # Basic_CheckTrickRoom (:282); Expert_TrickRoom (:1800)
    SWITCH_ABILITIES = 191  # Skill Swap: Expert_ChangeUserAbility (:1743); no Basic entry
    SUPRESS_ABILITY = 239  # Gastro Acid: Basic_CheckGastroAcid (:269); Expert_GastroAcid (:1782)
    FLING = 233  # Basic_CheckFling (:263); Expert_Fling (:1776)
    # E3 batch 2 (E3-B2-2).
    COPY_MOVE = 9  # Mirror Move: Expert_MirrorMove (:1632); Risky :6539; no Basic entry
    COPY_ABILITY = 178  # Role Play: Expert_ChangeUserAbility (:1733); no Basic entry
    APPLY_MAGIC_COAT = 183  # Magic Coat: Expert_MagicCoat (:1736); no Basic entry (E3-B2-6)
    PREVENT_CRITS = (
        240  # Lucky Chant: Basic_CheckLuckyChant (:270); Expert_LuckyChant (:1783), E3-B2-7
    )
    # Miracle Eye: Basic_CheckMiracleEye (:255); Expert_MiracleEye (:1760), E3-B2-10
    IGNORE_EVATION_REMOVE_DARK_IMMUNE = 216


# BattleEffect members whose decomp name (BATTLE_EFFECT_<name>) starts with a digit, which a
# Python name cannot.
BATTLE_EFFECT_DECOMP_NAMES = {
    BattleEffect.DAMAGE_FLAT_40: "40_DAMAGE_FLAT",
    BattleEffect.DAMAGE_FLAT_20: "20_DAMAGE_FLAT",
}


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
# move with one of these effects whatever its power (:1035, :1553, :2827), through
# TrainerAI_CalcDamage: its move-id case, or, for another move id, the default (the table's
# power and type, :3074-3078).
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
# Expert_Encore_EncouragedMoveEffects (script.s:3202-3285): the last-move effects Expert_Encore
# encourages (IfLoadedNotInTable, :3189; trainer_ai.c:876-891). BATTLE_EFFECT_* ids, line - 1 of
# generated/move_battle_effects.txt, in the table's order, its repeats included (82 entries;
# E2 batch 11, B11-4).
ENCORE_ENCOURAGED_EFFECTS = (
    8,  # RECOVER_DAMAGE_SLEEP
    10,  # ATK_UP
    11,  # DEF_UP
    12,  # SPEED_UP
    13,  # SP_ATK_UP
    25,  # RESET_STAT_CHANGES
    28,  # FORCE_SWITCH
    30,  # CONVERSION
    33,  # STATUS_BADLY_POISON
    35,  # SET_LIGHT_SCREEN
    37,  # REST
    40,  # HALVE_HP
    54,  # SP_DEF_UP_2
    49,  # STATUS_CONFUSE
    66,  # STATUS_POISON
    67,  # STATUS_PARALYZE
    84,  # STATUS_LEECH_SEED
    85,  # DO_NOTHING
    50,  # ATK_UP_2
    90,  # ENCORE
    93,  # CONVERSION2
    94,  # NEXT_ATTACK_ALWAYS_HITS
    102,  # CURE_PARTY_STATUS
    106,  # PREVENT_ESCAPE
    107,  # STATUS_NIGHTMARE
    111,  # PROTECT
    191,  # SWITCH_ABILITIES
    113,  # FORESIGHT
    114,  # ALL_FAINT_3_TURNS
    115,  # WEATHER_SANDSTORM
    116,  # SURVIVE_WITH_1_HP
    118,  # ATK_UP_2_STATUS_CONFUSION
    120,  # INFATUATE
    124,  # PREVENT_STATUS
    136,  # WEATHER_RAIN
    137,  # WEATHER_SUN
    142,  # MAX_ATK_LOSE_HALF_MAX_HP
    143,  # COPY_STAT_CHANGES
    148,  # HIT_IN_3_TURNS
    158,  # ALWAYS_FLINCH_FIRST_TURN_ONLY
    160,  # STOCKPILE
    161,  # SPIT_UP
    162,  # SWALLOW
    164,  # WEATHER_HAIL
    165,  # TORMENT
    167,  # STATUS_BURN
    172,  # MAKE_GLOBAL_TARGET
    174,  # SP_DEF_UP_DOUBLE_ELECTRIC_POWER
    177,  # SWITCH_HELD_ITEMS
    178,  # COPY_ABILITY
    181,  # GROUND_TRAP_USER_CONTINUOUS_HEAL
    184,  # RECYCLE
    188,  # REMOVE_HELD_ITEM
    191,  # SWITCH_ABILITIES
    192,  # MAKE_SHARED_MOVES_UNUSEABLE
    193,  # HEAL_STATUS
    194,  # REMOVE_ALL_PP_ON_DEFEAT
    199,  # CONFUSE_ALL
    201,  # HALVE_ELECTRIC_DAMAGE
    210,  # HALVE_FIRE_DAMAGE
    212,  # ATK_SPD_UP
    213,  # CAMOUFLAGE
    215,  # GRAVITY
    216,  # IGNORE_EVATION_REMOVE_DARK_IMMUNE
    220,  # FAINT_AND_FULL_HEAL_NEXT_MON
    222,  # NATURAL_GIFT
    223,  # REMOVE_PROTECT
    225,  # DOUBLE_SPEED_3_TURNS
    226,  # RANDOM_STAT_UP_2
    233,  # FLING
    234,  # TRANSFER_STATUS
    236,  # PREVENT_HEALING
    238,  # SWAP_ATK_DEF
    239,  # SUPRESS_ABILITY
    240,  # PREVENT_CRITS
    243,  # SWAP_ATK_SP_ATK_STAT_CHANGES
    244,  # SWAP_DEF_SP_DEF_STAT_CHANGES
    247,  # SET_ABILITY_TO_INSOMNIA
    250,  # SWAP_STAT_CHANGES
    251,  # RESTORE_HP_EVERY_TURN
    252,  # GIVE_GROUND_IMMUNITY
    259,  # TRICK_ROOM
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
# Harrassment_Effects (script.s:8036-8071): the effects Harrassment_Main scores
# (IfLoadedNotInTable, :8027; trainer_ai.c:876-891). BATTLE_EFFECT_* ids, line - 1 of
# generated/move_battle_effects.txt, in the table's order (34 entries).
HARRASSMENT_EFFECTS = (
    1,  # STATUS_SLEEP
    18,  # ATK_DOWN
    19,  # DEF_DOWN
    23,  # ACC_DOWN
    24,  # EVA_DOWN
    49,  # STATUS_CONFUSE
    58,  # ATK_DOWN_2
    59,  # DEF_DOWN_2
    60,  # SPEED_DOWN_2
    62,  # SP_DEF_DOWN_2
    66,  # STATUS_POISON
    67,  # STATUS_PARALYZE
    84,  # STATUS_LEECH_SEED
    90,  # ENCORE
    100,  # DECREASE_LAST_MOVE_PP
    112,  # SET_SPIKES
    118,  # ATK_UP_2_STATUS_CONFUSION
    120,  # INFATUATE
    165,  # TORMENT
    166,  # SP_ATK_UP_CAUSE_CONFUSION
    167,  # STATUS_BURN
    173,  # NATURE_POWER
    187,  # STATUS_SLEEP_NEXT_TURN
    188,  # REMOVE_HELD_ITEM
    192,  # MAKE_SHARED_MOVES_UNUSEABLE
    197,  # SECRET_POWER
    199,  # CONFUSE_ALL
    205,  # ATK_DEF_DOWN
    213,  # CAMOUFLAGE
    232,  # PREVENT_ITEM_USE
    234,  # TRANSFER_STATUS
    249,  # TOXIC_SPIKES
    258,  # REMOVE_HAZARDS_SCREENS_EVA_DOWN
    265,  # SP_ATK_DOWN_2_OPPOSITE_GENDER
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
CLASS_PHYSICAL, CLASS_SPECIAL, CLASS_STATUS = 0, 1, 2
# TrainerAI_CalcDamage's move-id cases (trainer_ai.c:2885-3071) the AI damage does not model,
# refused: Gyro Ball (:2995-3003), whose power reads monSpeedValues, the speeds of the last
# CompareBattlerSpeed of any caller (battle_lib.c:1372-1373: the action order's sorts,
# battle_controller_player.c:780, :888, :3992, and the AI's own compares, trainer_ai.c:1109,
# :1121, :2081, in the scripts' order across every move slot), which no state field holds
# (B13-27). Return, Frustration, Low Kick, Grass Knot, Hidden Power and the fixed-damage cases
# are modelled (engine.damage.ai_case_power, ai_case_fixed; E1 batch 13), Psywave and
# Magnitude with their RandNext draws (:3015, :3030; E1 ruling G-39, ``_calc_read``), Natural
# Gift and Judgment by the held item (ai_case_power, ai_case_type; B13-25). Decomp move ids.
CALC_DAMAGE_MOVES = (360,)
# TrainerAI_MoveType's move-id cases (trainer_ai.c:3132-3227) refused in some state: Weather
# Ball outside rain, sand, sun and hail or under Cloud Nine / Air Lock, where the case assigns
# no result (engine.damage.ai_weather_ball_typed; policy._refused). Hidden Power's, Natural
# Gift's, Judgment's and Weather Ball's assigned types are modelled (damage.ai_move_type;
# B13-26). Decomp move ids.
MOVE_TYPE_MOVES = (311,)
# Expert_Copycat_EncouragedMoves (script.s:5582-5630): the player's last moves Copycat is
# encouraged to copy. Decomp move ids (MOVE_* of generated/moves.txt, line N is id N-1), in
# script order; the same numbers in Kaizo, whose names differ (data/game_constants.py).
COPYCAT_ENCOURAGED_MOVES = (
    79, 142, 147, 95, 47, 320, 325, 28, 108, 92, 12, 32, 90, 329, 238, 177, 109, 186, 103,
    178, 184, 313, 319, 86, 137, 77, 247, 223, 63, 245, 168, 343, 213, 207, 259, 260, 271,
    276, 285, 375, 384, 385, 389, 391, 415, 445, 464,
)  # fmt: skip

# Expert_MirrorMove_MoveTable (script.s:1928-1976): the same 47 moves in the same order as
# Expert_Copycat_EncouragedMoves above (compared name by name against the decomp's two tables by
# tests/engine/test_e3_batch2.py, E3-B2-2).
MIRROR_MOVE_TABLE = COPYCAT_ENCOURAGED_MOVES

# Expert_ChangeUserAbility_DesirableAbilities (script.s:4435-4461): ABILITY_SPEED_BOOST, _BATTLE_ARMOR,
# _SAND_VEIL, _STATIC, _FLASH_FIRE, _WONDER_GUARD, _EFFECT_SPORE, _SWIFT_SWIM, _HUGE_POWER,
# _RAIN_DISH, _CUTE_CHARM, _SHED_SKIN, _MARVEL_SCALE, _PURE_POWER, _CHLOROPHYLL, _SHIELD_DUST,
# _ADAPTABILITY, _MAGIC_GUARD, _MOLD_BREAKER, _SUPER_LUCK, _UNAWARE, _TINTED_LENS, _FILTER,
# _SOLID_ROCK, _RECKLESS (decomp ability ids, E3-2-5; checked against the manifest's names by
# tests/engine/test_e3_batch1.py).
CHANGE_USER_ABILITY_DESIRABLE = (
    3, 4, 8, 9, 18, 25, 27, 33, 37, 44, 56, 61, 63, 74, 34, 19, 91, 98, 104, 105, 109, 110, 111,
    116, 120,
)  # fmt: skip

# Basic_FlingItems_Poison / _Burn / _Paralyze and Expert_Fling_DesirableFlingEffects
# (script.s:1288-1299, :5288-5294): decomp hold-effect ids (generated/item_hold_effects.txt, line
# N is id N-1: HOLD_EFFECT_PSN_USER 100, _STRENGTHEN_POISON 80, _BRN_USER 101, _PIKA_SPATK_UP 71,
# _SOMETIMES_FLINCH 56), compared with ``LoadHeldItemEffect`` (Tables.item_hold_effect); checked
# against the decomp's names by tests/engine/test_e3_batch2.py (E3-B2-1).
FLING_ITEMS_POISON = (100, 80)
FLING_ITEMS_BURN = (101,)
FLING_ITEMS_PARALYZE = (71,)
FLING_DESIRABLE_EFFECTS = (56, 80, 100, 101, 71)

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
    # Basic_CheckOHKOWouldFail (script.s:560).
    "Sturdy": 5,
    # Basic_CheckRainDance / Expert_RainDance (script.s:789, :3749, :3758), Basic_CheckHail /
    # Expert_Hail (:860, :868, :4110); generated/abilities.txt.
    "Swift Swim": 33,
    "Rain Dish": 44,
    "Ice Body": 115,
    # Basic_CheckSunnyDay / Expert_SunnyDay (script.s:807-809, :3788).
    "Flower Gift": 122,
    # Basic_CheckCanForceSwitch (script.s:506).
    "Suction Cups": 21,
    # Basic_CheckMetalBurst (script.s:1168, :1175), Basic_CheckCanRemoveItem (:901),
    # Basic_CheckLowStatStage_Accuracy (:458).
    "Stall": 100,
    "Sticky Hold": 60,
    "Keen Eye": 51,
    "Solar Power": 94,
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
    # E2 batch 13: Expert_TrumpCard (script.s:5331-5332), Expert_Assurance (:5200-5201).
    "Pressure": 46,
    "Rough Skin": 24,
    # E3-2-2: Basic_CheckWorrySeed (script.s:1430-1433; Insomnia and Vital Spirit above).
    "Truant": 54,
    "Multitype": 121,
    # E3-2-6: Basic_CheckGastroAcid (script.s:1371-1376; Truant and Multitype above).
    "Slow Start": 112,
    "Stench": 1,
    "Run Away": 50,
    "Pickup": 53,
    "Honey Gather": 118,
    # E3-B2-1: Basic_FlingPoison / _FlingBurn AttackerChecks (script.s:1235-1278; Poison Heal,
    # Immunity, Magic Guard and Water Veil above).
    "Klutz": 103,
    "Guts": 62,
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
    # BatonPass_Main's IfMoveEqualTo and BatonPass_ProtectDetect (script.s:6581-6588,
    # :6615-6618), E1 prep batch P3.
    "Swords Dance": 14,
    "Dragon Dance": 349,
    "Calm Mind": 347,
    "Nasty Plot": 417,
    "Baton Pass": 226,
    "Protect": 182,
    "Detect": 197,
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
    # Expert_Counter / Expert_MirrorCoat (script.s:3125, :3882), Expert_Hail (:4105).
    "Counter": 68,
    "Mirror Coat": 243,
    "Blizzard": 59,
    # Expert_Protect (script.s:3536-3537).
    "Feint": 364,
    # Expert_Substitute (script.s:2994).
    "Focus Punch": 264,
    "Shadow Force": 467,
    # Expert_MagnetRise's IfMoveKnown DEFENDER list (script.s:6100-6102), E1 trial batch.
    "Earthquake": 89,
    "Earth Power": 414,
    # E3-2-2: Basic_CheckWorrySeed (script.s:1436-1437, with Snore above), Expert_WorrySeed (:5979).
    "Sleep Talk": 214,
    "Rest": 156,
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
    # Expert_Counter_PhysicalTypes / Expert_MirrorCoat_SpecialTypes (script.s:3165-3175,
    # :3922-3931); generated/pokemon_types.txt (line n + 1 is n).
    "Normal": 0,
    "Fighting": 1,
    "Flying": 2,
    "Rock": 5,
    "Bug": 6,
    "Ice": 15,
    "Dragon": 16,
    "Dark": 17,
}
COUNTER_PHYSICAL_TYPES = tuple(
    TYPE_IDS[n]
    for n in ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel")
)
MIRROR_COAT_SPECIAL_TYPES = tuple(
    TYPE_IDS[n] for n in ("Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark")
)
ITEM_EXPERT_BELT = 268  # HOLD_EFFECT_POWER_UP_SE
# E2 batch 13. Basic_NaturalGiftBerries (script.s:1061-1126): ITEM_CHERI_BERRY (149) through
# ITEM_ROWAP_BERRY (212), every id between, in id order (generated/items.txt, id = line - 1).
NATURAL_GIFT_BERRIES = tuple(range(149, 213))
# Expert_Recycle_DesirableItems (script.s:4536-4540): Chesto, Lum, Starf Berry.
RECYCLE_DESIRABLE_ITEMS = (150, 157, 207)
# Expert_Assurance_RecoilBerries (script.s:5218-5221): Jaboca and Rowap Berry, compared with a
# hold effect (LoadHeldItemEffect, :5202), whose ids stop at 146 (generated/
# item_hold_effects.txt): the jump never happens (decomp quirk), transcribed as written.
ASSURANCE_RECOIL_BERRIES = (211, 212)
# ITEM_POWER_HERB: line 272 of generated/items.txt (id = line - 1), the item
# Expert_ChargeTurnNoInvuln / _WithInvuln / Expert_ShadowForce test (script.s:3956, :4009, :4020).
ITEM_POWER_HERB = 271
# Expert_ChargeTurnWithInvuln_SandImmuneTypes (script.s:4068-4072).
CHARGE_TURN_SAND_IMMUNE_TYPES = tuple(TYPE_IDS[n] for n in ("Ground", "Rock", "Steel"))
_TRAPPING = jnp.array([ABILITY_IDS[n] for n in ("Shadow Tag", "Magnet Pull", "Arena Trap")], _I32)


class ReadKind(IntEnum):
    """What a loose read draws (``LooseDraws.kind``)."""

    NONE = 0  # no read at this index
    ROLL = 1  # IfRandomLessThan N: P(1) = N/256
    ABILITY = 2  # LoadBattlerAbility DEFENDER's coin flip: P(1) = 1/2
    SPEED_TIE = 3  # a SLOWER compare's tie, RandNext() & 1: P(1) = 1/2; no exact read (R3)
    SPEED_TIE_DRAWN = 4  # a FASTER or TIE compare's tie: a read of the exact layout (E1 R8)
    # TrainerAI_CalcDamage's draw (E1 ruling G-39): a digit, the class of the draw, with law
    # outcome.CALC_DRAW_P[CALC_PSYWAVE] (11 classes) or [CALC_MAGNITUDE] (7)
    CALC_PSYWAVE = 5
    CALC_MAGNITUDE = 6


def read_radix(kind: int) -> int:
    """The number of values a loose read of ``ReadKind`` ``kind`` takes (the loose oracle's
    radix): a damage draw's classes (11 for Psywave's ``% 11``, 7 Magnitude buckets), else 2."""
    if kind == ReadKind.CALC_PSYWAVE:
        return CALC_DRAW_CLASSES
    return 7 if kind == ReadKind.CALC_MAGNITUDE else 2


# LooseDraws columns: at least the reads one slot's scripts make. Every read counts, on the
# slot's path or not (the scripts are straight-line): 35 after K3c Task 6, 74 after E1 batch 7
# under every transcribed flag; only fight-sized layouts (E1 Ruling R7) are run loose, and each
# stays well under this (tests/ai/test_kaizo_outcome.py test_slot_reads_fit_*). More raises (policy._loose_reads). Tests only, but each column
# costs compile memory in every loose program, so the headroom is kept modest. E1 ruling G-39
# adds a column per move each damage command estimates: four per FlagMoveDamageScore, one per
# IfCurrentMoveKills, five per IfBattlerDealsMoreDamage, MOVES * (1 + SLOTS) = 28 per
# IfPartyMemberDealsMoreDamage (the active's four moves, then each party slot's four).
LOOSE_READS = 160


# ``LooseDraws.line`` of a damage draw the loose oracle makes but the exact layout does not:
# FlagMoveDamageScore compared only with AI_NO_COMPARISON_MADE (``flag_move_damage_score``).
UNREAD_DRAW = -1


class LooseDraws(NamedTuple):
    """The loose oracle's draws (module docstring). Read i of slot m, counted in the order
    the straight-line scripts make the reads (``count``), takes ``bits[m, i]``."""

    # int8[MOVES, LOOSE_READS]: 1 = the jump / ability slot 1; a CALC read's digit, its class
    bits: jnp.ndarray
    count: jnp.ndarray  # int32[MOVES]: reads made so far in each slot
    p: jnp.ndarray  # float32[MOVES, LOOSE_READS]: P(bit = 1) of each read
    kind: jnp.ndarray  # int8[MOVES, LOOSE_READS]: the ReadKind of each read
    live: jnp.ndarray  # bool[MOVES, LOOSE_READS]: the read is on the slot's path
    # int32[MOVES, LOOSE_READS]: an IfRandomLessThan's script.s line; UNREAD_DRAW for a damage
    # draw whose result the script never reads (``_calc_read``'s ``read`` False); else 0
    line: jnp.ndarray


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


class CalcEstimate(NamedTuple):
    """``TrainerAI_CalcDamage`` of moves (leading axes), one value per class of the move's
    ``RandNext`` draw (``outcome`` docstring; E1 ruling G-39): ``value[..., c]`` the damage when
    the draw's class is c, ``prob[..., c]`` the class's probability (``outcome.CALC_DRAW_P``; a
    move that does not draw has class 0 only), ``kind`` its ``outcome.calc_draw_kind``
    (``CALC_NONE`` when the call draws nothing)."""

    value: jnp.ndarray  # int32[..., CALC_DRAW_CLASSES]
    prob: jnp.ndarray  # float32[..., CALC_DRAW_CLASSES]
    kind: jnp.ndarray  # int32[...]


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
    seen_moves: (
        jnp.ndarray
    )  # int32[4]: the player's active's move ids the AI has seen (battlerMoves)
    player_ability: jnp.ndarray  # int32: LoadBattlerAbility DEFENDER (decomp id)
    # int32[4, len(SITE_LINES)]: the flat layout's script_rolls index of each line; an
    # unused int32[] in a per-slot layout, whose reads go by order (per_slot)
    sites: jnp.ndarray
    # int32[2]: LoadBattlerAbility DEFENDER under guess 0 and guess 1 (defender_ability);
    # bool: its reads draw a coin this decision (ability_guessed). Per-slot layout.
    guess_abilities: jnp.ndarray
    guessing: jnp.ndarray
    # CalcEstimate [MOVES, C]: TrainerAI_CalcAllDamage of the AI's active's moves at
    # USE_MAX_DAMAGE, every class of each move's draw (``estimate``)
    damage: CalcEstimate
    # Per-slot layout (E1 Ruling R4), per slot: the random reads made so far on its path (the
    # next one takes script_rolls[slot, reads[slot]]) and the product of their probabilities
    reads: jnp.ndarray  # int32[MOVES]
    weight: jnp.ndarray  # float32[MOVES]
    # int32: 0, or 1 + slot of the first read on a path past the slot's slot_reads draws
    # (_slot_read); the per-slot layout refuses the decision
    missing_read: jnp.ndarray
    loose: LooseDraws | None = None  # the loose oracle's draws (tests only), else None


def ai_damage(tables: Tables, state: BattleState, move, roll, draw=0) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868) of ``move`` by the AI's
    active Pokemon against the player's, with variance ``85 + roll`` percent applied
    after STAB and the type chart (:3109; the battle applies it before them): the
    engine's ``ai_damage_for_roll`` (roark.md 7.1), 0 for a move that is not damaging
    (``is_damaging``) and for an immunity. ``draw``: the class of Psywave's or Magnitude's
    ``RandNext`` draw (``estimate`` has every class)."""
    move = jnp.asarray(move, _I32)
    damage = ai_damage_for_roll(tables, state, 1, move, roll, draw=draw)
    return jnp.where(is_damaging(tables, move), damage, 0)


_CALC_P = jnp.asarray(CALC_DRAW_P, jnp.float32)  # [kind, class]
_CLASSES = jnp.arange(CALC_DRAW_CLASSES, dtype=_I32)


def drawn_cases(
    tables: Tables, state: BattleState, roll, *, atk_side=1, def_side=None, ivs=None
) -> jnp.ndarray:
    """``int32[3, C]``: per ``calc_draw_kind`` (row 0 unused), ``ai_damage_for_roll`` of the
    drawing move id (MOVE_PSYWAVE, MOVE_MAGNITUDE) for each class of its draw: the same for any
    move with that id, so ``estimate`` computes it once per attacker and defender instead of
    once per move (its IVs reach Hidden Power's case only, trainer_ai.c:2973-2993)."""

    def table(move):
        return jax.vmap(
            lambda c: ai_damage_for_roll(
                tables, state, atk_side, move, roll, def_side=def_side, ivs=ivs, draw=c
            )
        )(_CLASSES)

    zero = jnp.zeros(CALC_DRAW_CLASSES, _I32)
    return jnp.stack([zero, table(AI_PSYWAVE), table(AI_MAGNITUDE)]).astype(_I32)


def estimate(
    tables: Tables,
    state: BattleState,
    move,
    roll,
    *,
    atk_side=1,
    def_side=None,
    ivs=None,
    damaging_test: bool = True,
    cases=None,
    item=None,
) -> CalcEstimate:
    """``TrainerAI_CalcDamage`` of ``move`` by ``atk_side``'s active Pokemon at variance
    ``85 + roll``, for every class of the move's draw (``CalcEstimate``; trainer_ai.c:
    3014-3016, :3028-3048). ``damaging_test``: as ``TrainerAI_CalcAllDamage`` (:2810-2838) and
    the kill checks (:1539-1557), a move that is not damaging (``is_damaging``) is not
    estimated, 0 and no draw; ``IfBattlerDealsMoreDamage`` estimates the battler's move without
    it (:2228-2237). ``def_side`` / ``ivs`` / ``item``: ``ai_damage_for_roll``'s. ``cases``: the
    ``drawn_cases`` of the same attacker, defender and roll (computed here when not given)."""
    move = jnp.asarray(move, _I32)
    if cases is None:
        cases = drawn_cases(tables, state, roll, atk_side=atk_side, def_side=def_side, ivs=ivs)
    kind = calc_draw_kind(move)
    once = ai_damage_for_roll(
        tables, state, atk_side, move, roll, def_side=def_side, ivs=ivs, item=item
    )
    value = jnp.where(kind == CALC_NONE, jnp.broadcast_to(once, (CALC_DRAW_CLASSES,)), cases[kind])
    if damaging_test:
        damaging = is_damaging(tables, move)
        value = jnp.where(damaging, value, 0)
        kind = jnp.where(damaging, kind, CALC_NONE)
    return CalcEstimate(value.astype(_I32), _CALC_P[kind], kind.astype(_I32))


def _at_most(est: CalcEstimate, x, strict: bool = False) -> jnp.ndarray:
    """``P(D <= x)`` (``strict``: ``P(D < x)``) of each estimate's damage D, ``x`` broadcast
    over the estimates' leading axes."""
    x = jnp.asarray(x)[..., None]
    hit = est.value < x if strict else est.value <= x
    return jnp.sum(jnp.where(hit, est.prob, 0.0), axis=-1)


def _max_at_most(est: CalcEstimate, x, strict: bool = False) -> jnp.ndarray:
    """``P(max_i D_i <= x)`` (``strict``: ``<``) over the independent estimates ``est`` (one
    leading axis i), for each ``x`` (any shape)."""
    x = jnp.asarray(x)
    return jnp.prod(_at_most(est, x[..., None], strict), axis=-1)


def _realized(est: CalcEstimate, classes) -> jnp.ndarray:
    """``int32[n]``: each estimate's damage when its draw has class ``classes[i]``."""
    classes = jnp.clip(jnp.asarray(classes, _I32), 0, CALC_DRAW_CLASSES - 1)
    return jnp.take_along_axis(est.value, classes[:, None], axis=1)[:, 0]


def _drawn_classes(est: CalcEstimate, draws) -> jnp.ndarray:
    """``int32[n]``: the class of each estimate's draw from the raw ``RandNext()`` values
    ``draws`` (``uint16[CALC_DRAWS]``), the estimates that draw taking ``draws[0], draws[1],
    ...`` in their order (the decomp's), the others class 0."""
    drawn = est.kind != CALC_NONE
    index = jnp.clip(jnp.cumsum(drawn.astype(_I32)) - 1, 0, CALC_DRAWS - 1)
    raw = jnp.take(jnp.asarray(draws, _I32), index)
    return jnp.where(drawn, calc_kind_class(est.kind, raw), 0)


def _calc_read(ctx, est: CalcEstimate, gate, p_yes, decide, read: bool = True, ai_own=True):
    """One damage command's draws (E1 ruling G-39; ``outcome`` docstring): ``(ctx, yes)``, the
    command's yes/no from ``est`` (its estimates, one leading axis in the decomp's draw order)
    through ``decide(damages int32[n]) -> bool``. ``gate``: the command estimates at all on
    this path (its damaging test passed); ``p_yes``: the exact probability of yes over the
    draws' joint law. ``ai_own``: every estimate is of a move of the AI's active Pokemon
    (False: ``IfBattlerDealsMoreDamage``'s battler's move, ``IfPartyMemberDealsMoreDamage``'s
    party members').

    - loose: each move that draws is its own read, a digit with its class law, live on the
      path when ``gate`` holds and the move draws, and ``decide`` reads the digits' damages;
    - per-slot exact (``read``): when some move draws, the slot's next read (``_slot_read``)
      with probability ``p_yes``: in the enumeration's patterns (``calc_draws`` with a last
      axis of 0, ``outcome.no_calc_draws``) its bit (``READ_DRAWS[1]``: yes), else
      ``decide`` of the raw draws ``AiOutcome.calc_draws[slot, k]``;
    - ``read`` False (a result the script never reads), or no draw: the result at class 0;
    - the flat layout: the result at class 0, exact by a static invariant (raised otherwise).
      The flat layout is vanilla's (``layout.VANILLA``), whose policy refuses every state with
      an AI party move outside ``MOVE_EFFECTS`` (``policy._refused``, by move id), and whose
      Expert dispatch, restricted to those moves' effects, traces neither Copycat's
      ``IfBattlerDealsMoreDamage`` nor U-turn's ``IfPartyMemberDealsMoreDamage``. So in every
      accepted state its estimates are of ``MOVE_EFFECTS`` moves, and none of those is
      MOVE_PSYWAVE or MOVE_MAGNITUDE, the only ids that draw (trainer_ai.c:3014-3016,
      :3028-3048; ``outcome.calc_draw_kind``): nothing draws. A call that breaks either half
      (``ai_own`` False, or a drawing id in ``MOVE_EFFECTS``) raises."""
    drawn_each = gate & (est.kind != CALC_NONE)
    det = decide(est.value[:, 0])
    if ctx.loose is not None:
        digits = []
        for i in range(est.kind.shape[0]):
            kind = jnp.where(
                est.kind[i] == CALC_PSYWAVE,
                int(ReadKind.CALC_PSYWAVE),
                jnp.where(est.kind[i] == CALC_MAGNITUDE, int(ReadKind.CALC_MAGNITUDE), 0),
            )
            line = 0 if read else UNREAD_DRAW  # a result never read: no read of the exact layout
            ctx, digit = _loose_read(ctx, 0.0, kind, live=ctx.live & drawn_each[i], line=line)
            digits.append(jnp.where(drawn_each[i], digit.astype(_I32), 0))
        return ctx, decide(_realized(est, jnp.stack(digits)))
    if not per_slot(ctx):
        if not ai_own or set(MOVE_EFFECTS) & {AI_PSYWAVE, AI_MAGNITUDE}:
            raise ValueError("the flat layout estimates only MOVE_EFFECTS moves, which never draw")
        return ctx, det
    if not read:
        return ctx, det
    drawn = jnp.any(drawn_each)
    calc = ctx.outcome.calc_draws
    if calc is None:
        raise ValueError("an AiOutcome carries calc_draws (outcome.no_calc_draws)")
    if calc.shape[-1] == 0:  # an enumeration pattern: the read's bit is its result
        ctx, r = _slot_read(ctx, lambda r: (jnp.where(r == READ_DRAWS[1], p_yes, 1 - p_yes), drawn))
        return ctx, jnp.where(drawn, r == READ_DRAWS[1], det)
    k = jnp.clip(ctx.reads[ctx.slot], 0, calc.shape[1] - 1)
    raw = calc[ctx.slot, k]
    yes = decide(
        _realized(
            est, _drawn_classes(est._replace(kind=jnp.where(drawn_each, est.kind, CALC_NONE)), raw)
        )
    )
    ctx, _ = _slot_read(ctx, lambda r: (jnp.where(yes, p_yes, 1 - p_yes), drawn))
    return ctx, jnp.where(drawn, yes, det)


def is_damaging(tables: Tables, move) -> jnp.ndarray:
    """``bool``: the AI computes damage for ``move``: it is a move, its power is greater
    than 1 and its battle effect is not in ``NO_DAMAGE_CALC_EFFECTS``
    (``TrainerAI_CalcAllDamage``, trainer_ai.c:2827-2828; ``FlagMoveDamageScore``,
    :1035-1036; ``IfCurrentMoveKills``, :1553-1554). Those tests also pass a move whose
    effect is in ``ALT_POWER_EFFECTS``, whatever its power (:2823-2828)."""
    move = jnp.asarray(move, _I32)
    effect = move_effect(tables, move)
    no_calc = jnp.isin(effect, _NO_DAMAGE_CALC)
    alt = jnp.isin(effect, jnp.asarray(ALT_POWER_EFFECTS, _I32))
    return alt | ((move != 0) & (tables.move_power[move] > 1) & ~no_calc)


def _slot_damage(tables: Tables, state: BattleState, rolls) -> CalcEstimate:
    """``CalcEstimate [4, C]``: ``TrainerAI_CalcAllDamage`` (trainer_ai.c:2799) over the AI's
    true moveset, whatever the PP, with ``rolls[i]`` for slot i (or one roll for all), every
    class of each draw."""
    moves = state.moves[1, state.active[1]].astype(_I32)
    rolls = jnp.asarray(rolls, _I32)
    if rolls.ndim == 0:  # one roll for every slot (USE_MAX_DAMAGE): the drawing cases once
        cases = drawn_cases(tables, state, rolls)
        return jax.vmap(lambda m: estimate(tables, state, m, rolls, cases=cases))(moves)
    return jax.vmap(lambda m, r: estimate(tables, state, m, r))(moves, rolls)


def if_has_super_effective_move(ctx: AiContext) -> AiContext:
    """``IfHasSuperEffectiveMove`` (trainer_ai.c:2176-2184): ``AI_HasSuperEffectiveMove(
    attacker, TRUE)`` (:3560-3622), which draws nothing with TRUE: a move of the AI's active
    whose ``ApplyTypeChart`` (:3582, ``TrainerAI_MoveType`` :3578) sets
    ``MOVE_STATUS_SUPER_EFFECTIVE`` against the player's active. That battler is never in
    ``battlersSwitchingMask`` while its side has HP left (battle_controller_player.c:
    4108-4110, :4133-4134), and the partner check needs a doubles battle (:3597)."""
    tables, state = ctx.tables, ctx.state
    a, d = _side(Battler.ATTACKER), _side(Battler.DEFENDER)
    moves = state.moves[_row(state, a), state.active[a]].astype(_I32)
    d_types = state.types[_row(state, d), state.active[d]]
    identified, grounded, roosting, magnet, eyed = defender_chart_flags(tables, state, d)
    flags = jax.vmap(
        lambda m: type_chart_flags(
            tables,
            m,
            d_types,
            state.gravity,
            identified,
            ai_move_type(tables, state, a, m),
            grounded,
            roosting,
            magnet,
            eyed,
        )
    )(moves)
    return _jump(ctx, jnp.any((moves != 0) & ((flags & SUPER_EFFECTIVE) != 0)))


def if_party_member_deals_more_damage(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``IfPartyMemberDealsMoreDamage`` (trainer_ai.c:2106-2174): jumps when a living party
    member other than the active one has a move whose ``TrainerAI_CalcAllDamage`` beats the
    active's best. The party member's moves are calculated **as the AI's active battler**
    (``AI_CONTEXT.attacker``: its stats, level, types, ability and item in
    ``CalcMoveDamage``), with the member's IVs (Hidden Power's case) and held item (Natural
    Gift's and Judgment's cases, :2156-2165) passed; its ability parameter reaches only the
    cases' Klutz tests (:2886, :2898), with ``embargoTurns`` FALSE (:2163): a Klutz member's item
    is passed as none, and the active's as ``state.held_item`` (its ``Battler_Ability`` and
    ``embargoTurns``, :2133-2138), B13-25, B13-29. Only
    ``USE_MAX_DAMAGE`` (variance 100) is transcribed (asserted).

    Psywave's and Magnitude's cases draw for every such move estimated (:2130 the active's,
    :2156 each living member's; Psywave at the active's level, ``battleMons[attacker]``): one
    read (``_calc_read``) whose jump has probability ``1 - E[prod_j P(M_j <= A)]``, A the
    active's best and M_j member j's, all independent (the loop's ``break``, :2167, only stops
    draws that change nothing)."""
    assert int(use_damage_roll) == USE_MAX_DAMAGE, use_damage_roll
    tables, state = ctx.tables, ctx.state
    a = _side(Battler.ATTACKER)
    active = state.active[a]

    cases = drawn_cases(tables, state, SIM_ROLL_MAX)  # IVs reach Hidden Power's case only

    def estimates(moves, ivs, item):
        as_active = state._replace(ivs=state.ivs.at[_row(state, a), active].set(ivs))
        return jax.vmap(
            lambda m: estimate(tables, as_active, m, SIM_ROLL_MAX, cases=cases, item=item)
        )(moves.astype(_I32))

    mine = estimates(
        state.moves[_row(state, a), active], state.ivs[_row(state, a), active], held_item(state, a)
    )
    # The bench's own MON_DATA_ABILITY (trainer_ai.c:2162): raw party data.
    member_items = jnp.where(
        state.ability[_row(state, a)] == AbilityId.KLUTZ, 0, state.item[_row(state, a)]
    )
    theirs = jax.vmap(estimates)(
        state.moves[_row(state, a)], state.ivs[_row(state, a)], member_items
    )  # [SLOTS, 4, C]
    slots = jnp.arange(state.species.shape[1])
    alive = (state.species[_row(state, a)] > 0) & (state.hp[_row(state, a)] > 0) & (slots != active)
    # A member that is not estimated draws nothing (:2145-2148).
    theirs = theirs._replace(kind=jnp.where(alive[:, None], theirs.kind, CALC_NONE))
    n = theirs.kind.shape[0]
    # P(no jump) = sum over A's support of P(A = v) prod_j P(M_j <= v) (each value counted at
    # its first occurrence among the support's).
    support = mine.value.reshape(-1)  # [4C]
    valid = mine.prob.reshape(-1) > 0
    first = valid & ~jnp.any(
        jnp.tril(support[:, None] == support[None, :], -1) & valid[None, :], axis=1
    )
    p_a = _max_at_most(mine, support) - _max_at_most(mine, support, strict=True)  # [4C]
    members = jax.vmap(lambda e: _max_at_most(e, support))(theirs)  # [SLOTS, 4C]
    beaten = jnp.prod(jnp.where(alive[:, None], members, 1.0), axis=0)
    p_jump = 1.0 - jnp.sum(jnp.where(first, p_a * beaten, 0.0))
    flat = CalcEstimate(
        jnp.concatenate([mine.value, theirs.value.reshape(-1, CALC_DRAW_CLASSES)]),
        jnp.concatenate([mine.prob, theirs.prob.reshape(-1, CALC_DRAW_CLASSES)]),
        jnp.concatenate([mine.kind, theirs.kind.reshape(-1)]),
    )

    def decide(d):
        best = jnp.max(d[:MOVES])
        member_best = jnp.max(d[MOVES:].reshape(n, MOVES), axis=1)
        return jnp.any(alive & (member_best > best))

    ctx, jump = _calc_read(ctx, flat, ctx.live, p_jump, decide, ai_own=False)
    return _jump(ctx, jump)


def if_battler_deals_more_damage(
    ctx: AiContext, battler: Battler, use_damage_roll: int
) -> AiContext:
    """``IfBattlerDealsMoreDamage`` (trainer_ai.c:2187-2240): jumps when ``battler``'s last
    move (``movePrevByBattler``, ``BattleState.last_move``) out-damages the AI's best move.
    The AI's best is ``TrainerAI_CalcAllDamage`` over its active's moves (:2210-2219,
    ``_slot_damage``). The battler's move goes through ``TrainerAI_CalcDamage`` with no
    damaging-move test (:2229-2237): the battler is the attacker, ``AI_CONTEXT.defender``
    stays the defender (for ``AI_BATTLER_DEFENDER``, the player attacking itself: its own
    Defense, types, ability, screens), and the IVs are the AI attacker's (:2206-2208, Hidden
    Power's case and type). A status move or ``MOVE_NONE`` deals ``CalcMoveDamage``'s ``+ 2``
    through the chart (``engine.damage._finish``). Strictly greater (:2237). Only
    ``USE_MAX_DAMAGE`` (variance 100 on both sides, :2224-2227) is transcribed (asserted),
    and only the defender (asserted): the decomp's one call (script.s:5560, :5573)."""
    assert int(use_damage_roll) == USE_MAX_DAMAGE, use_damage_roll
    assert battler == Battler.DEFENDER, battler
    tables, state = ctx.tables, ctx.state
    a, d = _side(Battler.ATTACKER), _side(battler)
    mine = ctx.damage  # _slot_damage at USE_MAX_DAMAGE (new_context)
    theirs = estimate(
        tables,
        state,
        state.last_move[d].astype(_I32),
        SIM_ROLL_MAX,
        atk_side=d,
        def_side=_side(Battler.DEFENDER),
        ivs=state.ivs[_row(state, a), state.active[a]],
        damaging_test=False,
    )
    # Psywave's and Magnitude's draws (:2210 the AI's moves, :2228 the battler's; Psywave at
    # the battler's level): one read, P(B > A) = sum_c P(B = b_c) P(A < b_c).
    p_jump = jnp.sum(theirs.prob * _max_at_most(mine, theirs.value, strict=True))
    both = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None]]), mine, theirs)
    ctx, jump = _calc_read(
        ctx, both, ctx.live, p_jump, lambda dm: dm[MOVES] > jnp.max(dm[:MOVES]), ai_own=False
    )
    return _jump(ctx, jump)


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
    the status mask has an immunity bit (``MOVE_STATUS_IMMUNE``: ``INEFFECTIVE`` or Wonder
    Guard's bit, ``switching.wonder_guard_bit``, ruling G-23). Normalize, Adaptability,
    Scrappy, Tinted Lens, Expert Belt, Levitate, Filter and Solid Rock change
    this value and are refused (``policy.ai_refusals``); the player's Foresight, Ingrain /
    Iron Ball, roosting, Magnet Rise and Miracle Eye flags are read
    (``switching.defender_chart_flags``)."""
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
    identified, grounded, roosting, magnet, eyed = defender_chart_flags(tables, state, 0)
    flags = type_chart_flags(
        tables,
        move,
        d_types,
        state.gravity,
        identified,
        ai_move_type(tables, state, 1, move),  # TrainerAI_MoveType (:1323)
        grounded,
        roosting,
        magnet,
        eyed,
    )
    # MOVE_STATUS_IMMUNE (moves.h:87): INEFFECTIVE, Wonder Guard's or Magnet Rise's bit.
    immune = ((flags & (INEFFECTIVE | MAGNET_RISE)) != 0) | wonder_guard_bit(
        tables, state, 1, 0, move, flags
    )
    # Struggle: ApplyTypeChart returns 40 unchanged and sets no bit (battle_lib.c:2573).
    return jnp.where(immune, TypeMulti.IMMUNE, mapped).astype(_I32)


def attacker_ability(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: the AI's active Pokemon's current ability (decomp id): ``ABILITY_NONE`` (0)
    while its ``MOVE_EFFECT_ABILITY_SUPPRESSED`` bit is set (trainer_ai.c:1177-1178, :1221-1222:
    the bit alone, no Multitype exception), else the raw ``battleMons[battler].ability``
    (:1208, :1256): ``damage.decomp_abilities``, the engine's current ability (a change in
    battle such as Trace's copy follows it once the engine models the change)."""
    raw = decomp_abilities(tables, state)[1, state.active[1].astype(_I32)]
    return jnp.where(state.ability_suppressed[1] != 0, 0, raw)


def _defender_abilities(tables: Tables, state: BattleState):
    """``(value, known, base)`` of the player's active Pokemon: the ability
    ``LoadBattlerAbility DEFENDER`` returns without a guess, whether it does, and the base
    species' two abilities ``int32[2]`` (``defender_ability``). It returns the id the AI recorded
    (``battlerAbilities``, ``BattleAI_SetAbility``, battle_script.c:3401-3403, :12106-12109; a
    pre-Transform reveal stays the old ability after a Transform, G-69), else the battler's current
    ability when it is Shadow Tag, Magnet Pull or Arena Trap (trainer_ai.c:1181-1188;
    ``damage.decomp_abilities``: the form's, or a copy the engine models, Trace's or Transform's).
    ``base`` follows the BattleMon's species, the copied one after a Transform (:1191-1192).
    A battler with its ``MOVE_EFFECT_ABILITY_SUPPRESSED`` bit reads ``ABILITY_NONE`` (0), known,
    before any of that (:1177-1178, :1221-1222: the bit alone, no Multitype exception; no coin)."""
    d = state.active[0].astype(_I32)
    species = state.species[0, d].astype(_I32)
    # Raw battleMons[].ability (:1185-1188, :1230-1233), not Battler_Ability.
    current = decomp_abilities(tables, state)[0, d]
    recorded = state.ai_ability_known[d].astype(_I32)  # -1: known, not named by the engine
    suppressed = state.ability_suppressed[0] != 0
    true = jnp.where(suppressed, 0, jnp.where(recorded > 0, recorded, current))
    known = suppressed | (recorded != 0) | jnp.isin(current, _TRAPPING)
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
    abilities; else its only one. ``ABILITY_NONE`` for a suppressed battler (Gastro Acid,
    :1177-1178; ``_defender_abilities``).

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
    seen = record_seen_move(state.ai_seen_moves[d], state.last_move[0]).astype(_I32)
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
        damage=_slot_damage(tables, state, SIM_ROLL_MAX),
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


def _row(state: BattleState, battler):
    """The party index of ``battler`` (``engine.format.party``)."""
    return party(format_of(state), battler)


def _jump(ctx: AiContext, cond) -> AiContext:
    return ctx._replace(jump=jnp.asarray(cond))


def _loose_read(
    ctx: AiContext, p: float, kind, live=None, line: int = 0
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
        kind=lo.kind.at[slot, i].set(jnp.asarray(kind).astype(jnp.int8), mode="drop"),
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


def if_target_is_not_taunted(ctx: AiContext) -> AiContext:
    """``IfTargetIsNotTaunted`` (trainer_ai.c:2620-2628): jumps when the defender's
    ``moveEffectsData.tauntedTurns`` is 0 (``BattleState.taunt``, turns left)."""
    return _jump(ctx, ctx.state.taunt[_side(Battler.DEFENDER)] == 0)


def if_target_is_partner(ctx: AiContext) -> AiContext:
    """``IfTargetIsPartner`` (trainer_ai.c:2630): never in a single battle."""
    return _jump(ctx, False)


def if_move_equal_to(ctx: AiContext, move: int) -> AiContext:
    """``IfMoveEqualTo`` (trainer_ai.c:834)."""
    return _jump(ctx, ctx.move == move)


def flag_move_damage_score(
    ctx: AiContext, use_damage_roll: int, compared: bool = True
) -> AiContext:
    """``FlagMoveDamageScore`` (trainer_ai.c:1012): ``AI_NO_COMPARISON_MADE`` for a move
    that is not damaging; else ``AI_MOVE_IS_HIGHEST_DAMAGE`` when no slot of the true
    moveset estimates strictly more damage, ``AI_NOT_HIGHEST_DAMAGE`` otherwise.

    ``TrainerAI_CalcAllDamage`` (:1041) draws for each Psywave or Magnitude move of the moveset
    (``estimate``): one read (``_calc_read``), highest with probability ``sum_c P(D_slot = d_c)
    prod_{i != slot} P(D_i <= d_c)``. ``compared`` False: the script compares the result only
    with ``AI_NO_COMPARISON_MADE``, which does not depend on the draws, and loads ``calcTemp``
    anew before reading it again (script.s:62-67, :112-117, :6511-6512, :6572-6573), so the
    draws are no read (the loose oracle still draws them)."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        damage = _slot_damage(ctx.tables, ctx.state, ctx.outcome.sim_roll)
    else:
        damage = ctx.damage
    mine = jax.tree.map(lambda x: jnp.take(x, ctx.slot, axis=0), damage)
    others = jnp.arange(MOVES) != ctx.slot
    le = _at_most(damage, mine.value[:, None])  # [C, MOVES]: P(D_i <= d_c)
    p_high = jnp.sum(mine.prob * jnp.prod(jnp.where(others[None, :], le, 1.0), axis=1))
    damaging = is_damaging(ctx.tables, ctx.move)

    def decide(d):
        return ~jnp.any(d > jnp.take(d, ctx.slot))

    ctx, highest = _calc_read(ctx, damage, damaging, p_high, decide, read=compared)
    flag = jnp.where(highest, AI_MOVE_IS_HIGHEST_DAMAGE, AI_NOT_HIGHEST_DAMAGE)
    value = jnp.where(damaging, flag, AI_NO_COMPARISON_MADE)
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
    types = state.types[_row(state, side), state.active[side]].astype(_I32)
    index = 0 if target in (LoadType.DEFENDER_TYPE_1, LoadType.ATTACKER_TYPE_1) else 1
    return ctx._replace(calc_temp=types[index])


def if_temp_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfTempEqualTo`` (trainer_ai.c:1078)."""
    return _jump(ctx, ctx.calc_temp == value)


def go_to(ctx: AiContext) -> AiContext:
    """``GoTo`` (trainer_ai.c:2561): an unconditional jump."""
    return _jump(ctx, True)


# IfLevel's operations (include/constants/battle/trainer_ai.h:51-53).
CHECK_HIGHER_THAN_TARGET, CHECK_LOWER_THAN_TARGET, CHECK_EQUAL_TO_TARGET = 0, 1, 2


def if_level(ctx: AiContext, op: int) -> AiContext:
    """``IfLevel`` (trainer_ai.c:2579-2608): the attacker's level compared with the
    defender's (``battleMons[...].level``): higher, lower or equal by ``op``; any other
    ``op`` never jumps."""
    state = ctx.state
    a, d = _side(Battler.ATTACKER), _side(Battler.DEFENDER)
    mine = state.level[_row(state, a), state.active[a]].astype(_I32)
    theirs = state.level[_row(state, d), state.active[d]].astype(_I32)
    jump = {
        CHECK_HIGHER_THAN_TARGET: mine > theirs,
        CHECK_LOWER_THAN_TARGET: mine < theirs,
        CHECK_EQUAL_TO_TARGET: mine == theirs,
    }.get(int(op), jnp.asarray(False))
    return _jump(ctx, jump)


def load_battler_turn_count(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadBattlerTurnCount`` (trainer_ai.c): ``totalTurns - fakeOutTurnNumber`` of the
    battler, its turns since it came in (``BattleState.turn``, ``.fake_out_turn``)."""
    state, side = ctx.state, _side(battler)
    turns = state.turn.astype(_I32) - state.fake_out_turn[side].astype(_I32)
    return ctx._replace(calc_temp=turns)


def load_effect_of_loaded_move(ctx: AiContext) -> AiContext:
    """``LoadEffectOfLoadedMove`` (trainer_ai.c): ``calcTemp = MOVE_DATA(calcTemp).effect``
    (``Tables.move_ai_effect``)."""
    return ctx._replace(calc_temp=ctx.tables.move_ai_effect[ctx.calc_temp].astype(_I32))


ITEM_SHINY_STONE = 107  # Basic_CheckMetalBurst (script.s:1169, :1176; generated/items.txt)


def _held_item(ctx: AiContext, battler: Battler) -> jnp.ndarray:
    """The battler's held item as the AI sees it for ``IfHeldItemEqualTo`` (trainer_ai.c:
    1915-1934): its own side's true ``heldItem``; the other side's
    ``aiContext.battlerHeldItems`` (``BattleState.ai_item_known`` of the active slot, the
    item a message named, 0 none)."""
    state, side = ctx.state, _side(battler)
    if side == _side(Battler.ATTACKER):
        return state.item[_row(state, side), state.active[side]].astype(_I32)
    return state.ai_item_known[state.active[side]].astype(_I32)


def if_held_item_equal_to(ctx: AiContext, battler: Battler, item: int) -> AiContext:
    """``IfHeldItemEqualTo`` (trainer_ai.c:1915-1934): ``_held_item == item``."""
    return _jump(ctx, _held_item(ctx, battler) == item)


def load_fling_power(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadFlingPower`` (trainer_ai.c:2014-2022): ``Battler_ItemFlingPower`` of the battler
    (battle_lib.c:5491-5498): 0 under its Embargo, else the true ``heldItem``'s ``flingPower``
    (``Tables.item_fling_power``), whichever side (E3-B2-1)."""
    state, side = ctx.state, _side(battler)
    item = state.item[_row(state, side), state.active[side]]
    power = jnp.where(state.embargo[side] != 0, 0, ctx.tables.item_fling_power[item])
    return ctx._replace(calc_temp=power.astype(_I32))


def load_held_item(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadHeldItem`` (trainer_ai.c:1891-1899): ``battleMons[battler].heldItem``, the true item of
    either side (unlike ``IfHeldItemEqualTo``)."""
    state, side = ctx.state, _side(battler)
    return ctx._replace(calc_temp=state.item[_row(state, side), state.active[side]].astype(_I32))


def load_held_item_effect(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadHeldItemEffect`` (trainer_ai.c:1901-1913): the hold effect
    (``Tables.item_hold_effect``) of the attacker's true ``heldItem``, or of
    ``battlerHeldItems`` (``_held_item``'s) for another battler."""
    item = _held_item(ctx, battler)
    return ctx._replace(calc_temp=ctx.tables.item_hold_effect[item].astype(_I32))


def load_recycle_item(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadRecycleItem`` (trainer_ai.c:2510-2518): ``battleCtx->recycleItem[battler]``
    (``BattleState.recycle_item``, B13-19)."""
    return ctx._replace(calc_temp=ctx.state.recycle_item[_side(battler)].astype(_I32))


def load_current_move_pp(ctx: AiContext) -> AiContext:
    """``LoadCurrentMovePP`` (trainer_ai.c:2024-2028): the attacker's
    ``ppCur[AI_CONTEXT.moveSlot]``."""
    state, a = ctx.state, _side(Battler.ATTACKER)
    return ctx._replace(calc_temp=state.pp[_row(state, a), state.active[a], ctx.slot].astype(_I32))


def sum_positive_stat_stages(ctx: AiContext, battler: Battler) -> AiContext:
    """``SumPositiveStatStages`` (trainer_ai.c:2243-2257): the sum of ``statBoosts[stat] - 6``
    over the stages above 6, ``BATTLE_STAT_HP`` through accuracy and evasion (the HP entry
    never moves)."""
    stages = ctx.state.stages[_side(battler)].astype(_I32)  # engine units, 0 = neutral
    return ctx._replace(calc_temp=jnp.sum(jnp.maximum(stages, 0)))


def if_not_move_effect(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfNotMoveEffect`` (trainer_ai.c:718-730): the negation of ``IfMoveEffect``,
    ``(moveEffectsMask & mask) == FALSE`` (:727)."""
    ctx = if_move_effect(ctx, battler, mask)
    return ctx._replace(jump=~ctx.jump)


def if_loaded_less_than(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedLessThan`` (trainer_ai.c:762-772): ``calcTemp < value`` (B12-11)."""
    return _jump(ctx, ctx.calc_temp < value)


def if_loaded_greater_than(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedGreaterThan`` (trainer_ai.c:774-784): ``calcTemp > value``."""
    return _jump(ctx, ctx.calc_temp > value)


def if_loaded_mask(ctx: AiContext, mask: int) -> AiContext:
    """``IfLoadedMask`` (trainer_ai.c:810-820): ``calcTemp & mask``."""
    return _jump(ctx, (ctx.calc_temp & mask) != 0)


def if_loaded_not_mask(ctx: AiContext, mask: int) -> AiContext:
    """``IfLoadedNotMask`` (trainer_ai.c:822-832): ``(calcTemp & mask) == 0``."""
    return _jump(ctx, (ctx.calc_temp & mask) == 0)


def load_battle_type(ctx: AiContext) -> AiContext:
    """``LoadBattleType`` (trainer_ai.c:2504-2508): ``battleSys->battleType``
    (``Tables.battle_type``, the fight's, ``ai.layout.fight_tables``)."""
    return ctx._replace(calc_temp=jnp.asarray(ctx.tables.battle_type, _I32))


def load_protect_chain(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadProtectChain`` (trainer_ai.c:2538-2552): ``protectSuccessTurns``, or 0 when the
    battler's ``moveProtect`` is not Protect, Detect or Endure; ``BattleState.protect_count``
    folds that reset in (engine/STATE.md, moveProtect)."""
    return ctx._replace(calc_temp=ctx.state.protect_count[_side(battler)].astype(_I32))


def if_loaded_not_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedNotEqualTo`` (trainer_ai.c:798)."""
    return _jump(ctx, ctx.calc_temp != value)


def if_current_move_effect_equal_to(ctx: AiContext, effect: BattleEffect) -> AiContext:
    """``IfCurrentMoveEffectEqualTo`` (trainer_ai.c:1441)."""
    return _jump(ctx, move_effect(ctx.tables, ctx.move) == int(effect))


def _stage(ctx: AiContext, battler: Battler, stat: Stat) -> jnp.ndarray:
    """The battler's true stat stage in decomp units (engine + 6)."""
    return ctx.state.stages[_side(battler), stat].astype(_I32) + STAGE_NEUTRAL


def diff_stat_stages(ctx: AiContext, battler: Battler, stat: Stat) -> AiContext:
    """``DiffStatStages`` (trainer_ai.c:2259-2268): ``calcTemp = statBoosts[battler][stat] -
    statBoosts[attacker][stat]`` (the decomp's +6 offset cancels), B12-11."""
    diff = _stage(ctx, battler, stat) - _stage(ctx, Battler.ATTACKER, stat)
    return ctx._replace(calc_temp=diff)


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
            on = on | (getattr(ctx.state, field)[side_of(side)] > 0)
    return _jump(ctx, on)


def check_battler_ability(ctx: AiContext, battler: Battler, expected: int) -> AiContext:
    """``CheckBattlerAbility`` (trainer_ai.c:1212-1265): ``AI_HAVE`` when the battler's
    ability as the AI sees it is ``expected``, ``AI_NOT_HAVE`` when it is another, and
    ``AI_UNKNOWN`` when the AI does not know it, or the battler's ability is suppressed
    (:1221-1222: ``ABILITY_NONE``). The AI's own Pokemon: its current ability.
    The player's: the one the AI knows (learned, or a trapping ability, :1224-1233); else, for
    a species of two abilities, ability 1 when neither is ``expected`` and unknown when one
    is (:1238-1246), for a species of one, that one (:1247-1251). No draw."""
    # The decomp branches on the player's side (DEFENDER or DEFENDER_PARTNER, :1223).
    if _side(battler) == _side(Battler.ATTACKER):
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
    return ctx._replace(calc_temp=getattr(ctx.state, field)[side_of(_side(battler))].astype(_I32))


def count_alive_party_battlers(ctx: AiContext, battler: Battler) -> AiContext:
    """``CountAlivePartyBattlers`` (trainer_ai.c:1126): the side's living party members
    other than the active one."""
    state, side = ctx.state, _side(battler)
    alive = (state.species[_row(state, side)] > 0) & (state.hp[_row(state, side)] > 0)
    alive = alive & (jnp.arange(alive.shape[0]) != state.active[side])
    return ctx._replace(calc_temp=jnp.sum(alive.astype(_I32)))


def add_to_move_score(ctx: AiContext, value: int) -> AiContext:
    """``AddToMoveScore`` (trainer_ai.c:576-586): ``moveScore[slot] += val``, then a
    negative score becomes 0. ``moveScore`` is ``s8`` (include/battle/ai_context.h:14), so
    the sum wraps to -128..127 before the clamp: 127 + 2 is -127, then 0."""
    current = ctx.scores[ctx.slot]
    new = jnp.maximum((current + value + 128) % 256 - 128, 0)
    return ctx._replace(scores=ctx.scores.at[ctx.slot].set(jnp.where(ctx.live, new, current)))


def pop_or_end(ctx: AiContext) -> AiContext:
    """``PopOrEnd`` (trainer_ai.c:2568): nothing pushes (``PushAndGoTo``, :2554, appears
    nowhere in script.s), so this ends the slot for this flag."""
    return ctx._replace(live=jnp.asarray(False))


def if_current_move_kills(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``IfCurrentMoveKills`` (trainer_ai.c:1525): the move is damaging and the
    defender's true HP is at most its estimated damage. Psywave's and Magnitude's cases draw
    (:1560): one read (``_calc_read``) that kills with probability ``sum_c P(c) [hp <= d_c]``."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        roll = ctx.outcome.sim_roll[ctx.slot]
    else:
        roll = SIM_ROLL_MAX
    est = estimate(ctx.tables, ctx.state, ctx.move, roll)
    state = ctx.state
    hp = state.hp[0, state.active[0]].astype(_I32)
    p_kill = jnp.sum(jnp.where(hp <= est.value, est.prob, 0.0))
    damaging = is_damaging(ctx.tables, ctx.move)
    one = jax.tree.map(lambda x: x[None], est)
    ctx, kills = _calc_read(ctx, one, damaging, p_kill, lambda d: hp <= d[0])
    return _jump(ctx, damaging & kills)


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
    return (
        state.hp[_row(state, side), mon].astype(_I32)
        * 100
        // state.max_hp[_row(state, side), mon].astype(_I32)
    )


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
TOXIC_COUNTER = 0xF << 8  # MON_CONDITION_TOXIC_COUNTER (condition.h:15-18, the group :23)
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
    # The whole toxic counter with the toxic bit (MON_CONDITION_ANY_POISON) adds nothing: the
    # counter is nonzero only on a badly poisoned Pokemon.
    if mask & TOXIC_COUNTER == TOXIC_COUNTER and mask & MonCondition.TOXIC:
        mask &= ~TOXIC_COUNTER
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
    status = ctx.state.status[_row(ctx.state, side), ctx.state.active[side]].astype(_I32)
    return _jump(ctx, jnp.isin(status, jnp.asarray(_condition_statuses(mask), _I32)))


def if_party_member_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfPartyMemberStatus`` (trainer_ai.c:1348-1379): in singles, a party member other than
    the battler's selected slot, with HP and a species, whose party status meets ``mask``
    (``_condition_statuses``), B12-18."""
    side = _side(battler)
    state = ctx.state
    statuses = jnp.asarray(_condition_statuses(mask), _I32)
    status = state.status[_row(state, side)].astype(_I32)
    slots = jnp.arange(status.shape[0])
    member = (
        (slots != state.active[side])
        & (state.hp[_row(state, side)] != 0)
        & (state.species[_row(state, side)] != 0)
    )
    return _jump(ctx, jnp.any(member & jnp.isin(status, statuses)))


def if_not_volatile_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfNotVolatileStatus`` (trainer_ai.c): the negation of ``IfVolatileStatus``."""
    ctx = if_volatile_status(ctx, battler, mask)
    return ctx._replace(jump=~ctx.jump)


def if_not_status(ctx: AiContext, battler: Battler, mask: int) -> AiContext:
    """``IfNotStatus`` (trainer_ai.c:662-674): the negation of ``IfStatus``."""
    ctx = if_status(ctx, battler, mask)
    return ctx._replace(jump=~ctx.jump)


def if_stat_stage_less_than(ctx: AiContext, battler: Battler, stat: Stat, value: int) -> AiContext:
    """``IfStatStageLessThan`` (trainer_ai.c:1465), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) < value)


def if_field_conditions_mask(ctx: AiContext, mask: int) -> AiContext:
    """``IfFieldConditionsMask`` (trainer_ai.c:1936-1946): ``fieldConditionsMask & mask``.
    ``FieldCondition.TRICK_ROOM`` and ``FieldCondition.GRAVITY`` are supported (static; refused
    otherwise): Trick Room is on while ``BattleState.trick_room`` is nonzero, Gravity while
    ``BattleState.gravity`` is."""
    assert int(mask) in (
        FieldCondition.TRICK_ROOM,
        FieldCondition.GRAVITY,
    ), f"unsupported field mask {int(mask):#x}"
    field = ctx.state.gravity if int(mask) == FieldCondition.GRAVITY else ctx.state.trick_room
    return _jump(ctx, field != 0)


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
    - At equal speed every branch gives FASTER, or TIE on the tie's ``RandNext() & 1``, never
      SLOWER (:1401-1447). So ``== SLOWER`` does not depend on the tie, and the exact layout
      does not draw it (plan K3c Ruling 3; ``order.battler_slower`` with ``tie_roll = 0``); the
      decomp still consumes the draw (K3d), an independent draw that changes nothing here.
      ``== FASTER`` does depend on it (E1 Ruling R8): where the compare would read the tie
      (``compare_speed`` gives TIE for a set bit), it is the slot's next random read
      (``_slot_read``), TIE when the draw's low bit is set (READ_DRAWS[1] = 1), probability
      1/2. ``== TIE`` (Basic_CheckTrickRoom, script.s:1505) reads it the same way.

    Loose (module docstring): the tie is its own draw (``ReadKind.SPEED_TIE`` for SLOWER,
    ``SPEED_TIE_DRAWN`` for the compares that the exact layout reads it for), on the path
    where the decomp makes it (equal speed in a branch that reads it, which is where
    ``compare_speed`` gives TIE for a 1), and the decomp's three-valued result is compared
    with ``value``."""
    value = int(value)
    assert value in (COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER, COMPARE_SPEED_TIE), value
    tables, state = ctx.tables, ctx.state
    a = _side(Battler.ATTACKER)
    slower = value == COMPARE_SPEED_SLOWER
    if ctx.loose is not None:
        reads = compare_speed(tables, state, a, 1, state.qc_roll) == COMPARE_SPEED_TIE
        kind = ReadKind.SPEED_TIE if slower else ReadKind.SPEED_TIE_DRAWN
        ctx, bit = _loose_read(ctx, 0.5, kind, live=ctx.live & reads)
        return _jump(ctx, compare_speed(tables, state, a, bit, state.qc_roll) == value)
    if slower:
        return _jump(ctx, battler_slower(tables, state, a, 0, state.qc_roll))
    if not per_slot(ctx):
        raise AssertionError("a FASTER or TIE compare is transcribed for the per-slot layout only")
    tie = compare_speed(tables, state, a, 1, state.qc_roll) == COMPARE_SPEED_TIE
    ctx, r = _slot_read(ctx, lambda r: (0.5, tie))
    return _jump(ctx, compare_speed(tables, state, a, r & 1, state.qc_roll) == value)


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


# IfBattlerUnderEffect's checks (include/constants/battle/trainer_ai.h:48-49).
CHECK_DISABLE, CHECK_ENCORE = 0, 1


def if_battler_under_effect(ctx: AiContext, battler: Battler, check: int) -> AiContext:
    """``IfBattlerUnderEffect`` (trainer_ai.c:1823-1848): ``CHECK_DISABLE`` jumps when the
    battler's ``moveEffectsData.disabledTurns`` is nonzero (``BattleState.disable_turns``),
    ``CHECK_ENCORE`` when its ``encoredTurns`` is (``BattleState.encore_turns``); any other check
    never jumps (E2 batch 11, B11-4)."""
    check = int(check)
    assert check in (CHECK_DISABLE, CHECK_ENCORE), check
    field = "disable_turns" if check == CHECK_DISABLE else "encore_turns"
    return _jump(ctx, getattr(ctx.state, field)[_side(battler)] != 0)


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
    return ctx._replace(calc_temp=state.gender[_row(state, side), state.active[side]].astype(_I32))


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
