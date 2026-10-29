"""The engine's vocabulary: effect, ability, item, status, stat, move-flag and type ids.

Tables store these ids (``tables.move_effect``, ``species_abilities``,
``item_effect``, ``move_flags``); the decomp's own ids are mapped onto them by
``tools/effect_map.py`` at table-build time. Anything outside the implemented set
is the ``UNSUPPORTED`` member, never an approximation.
"""

from enum import IntEnum, IntFlag


class EffectId(IntEnum):
    """Move effect. Params live in ``tables.move_param[m, 0..2]``:

    * stat effects (``STAT_SELF``, ``STAT_FOE``, ``SECONDARY_STAT_*``): ``(Stat, stages, 0)``
    * status effects (``STATUS_FOE``, ``SECONDARY_STATUS``): ``(Status, 0, 0)``
    * ``DRAIN``, ``RECOIL``: ``(percent of damage dealt, 0, 0)``
    * everything else: zeros.

    The proc chance of a ``SECONDARY_*`` or ``FLINCH`` effect is the move's own
    ``move_effect_chance``; a ``SECONDARY_*`` move with chance 0 applies its
    effect on every hit (the decomp's ``MOVE_SIDE_EFFECT_ON_HIT``, e.g. Hammer Arm).
    """

    UNSUPPORTED = 0
    PLAIN = 1
    STAT_SELF = 2
    STAT_FOE = 3
    SECONDARY_STAT_SELF = 4
    SECONDARY_STAT_FOE = 5
    STATUS_FOE = 6
    SECONDARY_STATUS = 7
    FLINCH = 8
    HIGH_CRIT = 9
    DRAIN = 10
    STEALTH_ROCK = 11
    TRAP = 12
    PURSUIT = 13
    TAUNT = 14
    CONFUSE_FOE = 15
    LEECH_SEED = 16
    MULTI_HIT = 17
    WATER_SPORT = 18
    RECOIL = 19


class AbilityId(IntEnum):
    UNSUPPORTED = 0
    NONE = 1
    STURDY = 2
    ROCK_HEAD = 3
    MOLD_BREAKER = 4
    BLAZE = 5
    TORRENT = 6
    OVERGROW = 7
    INTIMIDATE = 8
    RIVALRY = 9
    KEEN_EYE = 10
    SIMPLE = 11
    UNAWARE = 12
    NATURAL_CURE = 13
    POISON_POINT = 14
    INNER_FOCUS = 15
    STATIC = 16
    GUTS = 17
    STEADFAST = 18


class ItemId(IntEnum):
    """Held-item effect. ``tables.item_param[i, 0]`` is the decomp's hold param
    (Oran Berry: HP restored; Sitrus Berry: percent of max HP); column 1 is 0."""

    UNSUPPORTED = 0
    NONE = 1
    ORAN_BERRY = 2
    SITRUS_BERRY = 3


class Status(IntEnum):
    NONE = 0
    SLEEP = 1
    POISON = 2
    BURN = 3
    FREEZE = 4
    PARALYSIS = 5
    TOXIC = 6


class Stat(IntEnum):
    """Index into every stat vector (base stats, stats, nature mods) and stages."""

    HP = 0
    ATK = 1
    DEF = 2
    SPA = 3
    SPD = 4
    SPE = 5
    ACC = 6
    EVA = 7


class MoveFlag(IntFlag):
    """Bits of ``tables.move_flags``."""

    CONTACT = 1
    PROTECT = 2
    SOUND = 4
    PUNCH = 8


class Weather(IntEnum):
    """``BattleState.weather``; also a battle's start condition (``make_state``)."""

    NONE = 0
    SUN = 1
    RAIN = 2
    SANDSTORM = 3
    HAIL = 4
    FOG = 5


class Type(IntEnum):
    """Decomp type ids (``generated/pokemon_types.txt`` order), as stored in
    ``tables.move_type``, ``tables.species_types`` and the type chart axes."""

    NORMAL = 0
    FIGHTING = 1
    FLYING = 2
    POISON = 3
    GROUND = 4
    ROCK = 5
    BUG = 6
    GHOST = 7
    STEEL = 8
    MYSTERY = 9
    FIRE = 10
    WATER = 11
    GRASS = 12
    ELECTRIC = 13
    PSYCHIC = 14
    ICE = 15
    DRAGON = 16
    DARK = 17


# Decomp move id the engine names directly (``generated/moves.txt`` line 166, 0-based
# from MOVE_NONE): Struggle replaces the chosen move when it has no PP left.
MOVE_STRUGGLE = 165
