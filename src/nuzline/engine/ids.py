"""The engine's vocabulary: effect, ability, item, status, stat, move-flag and type ids.

Tables store these ids (``tables.move_effect``, ``species_abilities``,
``item_effect``, ``trainer_item_effect``, ``move_flags``); the decomp's own ids are mapped onto them by
``tools/effect_map.py`` at table-build time. Anything outside the implemented set
is the ``UNSUPPORTED`` member, never an approximation.
"""

from enum import IntEnum, IntFlag


class EffectId(IntEnum):
    """Move effect. Params live in ``tables.move_param[m, 0..2]``:

    * stat effects (``STAT_SELF``, ``STAT_FOE``, ``SECONDARY_STAT_*``): ``(Stat, stages, 0)``;
      ``STAT_SELF`` with ``move_param[m, 2] == 1`` (Defense Curl) also sets ``curled``
    * status effects (``STATUS_FOE``, ``SECONDARY_STATUS``): ``(Status, type_checked,
      thaws_user)``; ``type_checked`` is 1 for a status move the decomp runs through
      the type chart (Thunder Wave: Ground is immune), else 0; ``thaws_user`` is 1 for
      ``BATTLE_EFFECT_THAW_AND_BURN_HIT`` (Flame Wheel, Sacred Fire), which a frozen
      user can use and which thaws it, else 0
    * ``DRAIN``, ``RECOIL``: ``(percent of damage dealt, 0, 0)``
    * everything else: zeros. ``LEVEL_DAMAGE`` (Night Shade, Seismic Toss) deals the user's
      level; ``HIDDEN_POWER`` and ``MAGNITUDE`` take their type and power from the
      battler and ``Outcome.magnitude`` (``engine/damage.py``).
    * ``SECONDARY_CONFUSE`` (Confusion, Psybeam, ...): zeros; confuses the target at the
      move's effect chance. ``MUD_SPORT``: zeros. ``EAT_BERRY`` (Bug Bite, Pluck): zeros;
      the user eats the target's Oran or Sitrus Berry (``hooks.pluck_berry``).
    ``AQUA_RING``: zeros; sets the user's ``aqua_ring`` (fails if set), healed at turn end
    (``step._aqua_ring_check``).
    ``DISABLE``: zeros; disables the target's last move for 3-6 turns (``effects._disable``).
    ``LIGHT_SCREEN``, ``REFLECT``: zeros; set the user's side screen for 5 turns
    (``effects._screen``). ``BRICK_BREAK``: zeros; ignores the screens in its damage and
    clears both of the defender's side after the hit (``effects._brick_break``).
    ``BIDE``, ``ROLLOUT`` (Rollout, Ice Ball): zeros; they lock the user into the move
    (``BattleState.locked_move``, ``engine/step.py``, ``engine/damage.py``).
    * ``MULTI_HIT``: ``(hits, 0, 0)``: ``hits == 0`` is the 2-5 law (``Outcome.n_hits``,
      ``BATTLE_EFFECT_MULTI_HIT``), else exactly that many hits (``BATTLE_EFFECT_HIT_TWICE``:
      Double Kick, Kaizo's DoubleSlap, ``SetMultiHit 2``).
    * Multi-stat effects: the stat list is the effect's own, so the params stay zeros.
      ``RAISE_ALL_STATS_HIT`` (AncientPower): Atk, Def, Spe, SpA, SpD +1 on the user at the
      move's effect chance; ``CURSE`` (a non-Ghost user; coverage refuses a Ghost-type one):
      Spe -1, Atk +1, Def +1 on the user.
    * ``RECOVER``: zeros; heals ``maxHP / 2``, fails at full HP. ``FAKE_OUT``: zeros; fails
      unless it is the user's first turn out, else flinches at the move's chance.
    * ``SELF_DESTRUCT`` (Selfdestruct, Explosion; Kaizo's Memento): zeros; the move script
      sets the user's HP to 0 before the accuracy check, and a physical one hits the halved
      Defense (``damage.py``). Damp would stop it; Damp is ``AbilityId.UNSUPPORTED``.
    * ``FLAIL`` (Flail, Reversal): zeros; the power comes from the user's HP pixels
      (``damage.flail_power``); the trainer AI scores it at the table power.
    * ``FANG_PARALYZE_FLINCH`` (Thunder Fang, ``BATTLE_EFFECT_FLINCH_PARALYZE_HIT``):
      ``(Status.PARALYSIS, 0, 0)``; two independent rolls at the move's chance, the
      paralysis (``Outcome.secondary``) then the flinch (``Outcome.fang_flinch``)
      (``effects._fang``).
    * ``PRESENT`` (Present, ``BATTLE_EFFECT_RANDOM_POWER_MAYBE_HEAL``): zeros;
      ``Outcome.present`` draws a power of 40/80/120 for the hit (``damage.PRESENT_POWER``)
      or a heal of the target by ``maxHP / 4`` that ignores immunities and reads no crit or
      roll (``effects._present``).
    * ``PROTECT`` (Protect, Detect, ``BATTLE_EFFECT_PROTECT``): zeros; ``BtlCmd_TryProtection``
      succeeds at 2^-n by ``protect_count`` unless the user acts last, and its success blocks
      the foe's ``MoveFlag.PROTECT`` moves for the rest of the turn (``step._use_move``).
    * ``ATTRACT`` (Attract, ``BATTLE_EFFECT_INFATUATE``): zeros; infatuates a living target of
      the opposite gender that is not in love yet (``effects._attract``); an infatuated
      battler is immobilized by ``Outcome.infatuation`` at its status check (``step``).
    * ``CAMOUFLAGE`` (Camouflage, ``BATTLE_EFFECT_CAMOUFLAGE``): zeros; ``BtlCmd_TryCamouflage``
      writes ``TERRAIN_TYPES[terrain]`` to both of the user's ``types`` unless it has Multitype
      or already has that type (``effects._camouflage``); ``step`` restores the party types
      when the battler leaves.
    * ``NATURE_POWER`` (Nature Power, ``BATTLE_EFFECT_NATURE_POWER``): zeros; the move runs the
      terrain's move ``TERRAIN_MOVES[terrain]`` from its accuracy check on
      (``order.called_move``, ``step._use_move``); it stays Nature Power for the status checks,
      the PP, Taunt, Disable, priority and ``last_move``.
    * ``TRI_ATTACK`` (Tri Attack, ``BATTLE_EFFECT_TRI_ATTACK``): zeros; at the move's chance
      (``Outcome.secondary``) a burn, freeze or paralysis (``TRI_ATTACK_STATUSES``, drawn by
      ``Outcome.tri_attack``) as a secondary status (``effects._tri_attack``).
    * ``ROAR`` (Roar, Whirlwind, ``BATTLE_EFFECT_FORCE_SWITCH``): zeros; in a trainer battle
      ``BtlCmd_TryWhirlwind`` drags a random eligible bench Pokemon of the target's side in
      (``Outcome.roar_target``), after the level check of a lower-levelled user
      (``Outcome.roar_level``); it fails with no eligible bench Pokemon (``step._use_move``).

    The proc chance of a ``SECONDARY_*``, ``FLINCH``, ``FAKE_OUT``, ``RAISE_ALL_STATS_HIT`` or
    ``FANG_PARALYZE_FLINCH``
    effect is the move's own ``move_effect_chance``; a ``SECONDARY_*`` move with chance 0 applies its
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
    FORESIGHT = 20
    LEVEL_DAMAGE = 21
    HIDDEN_POWER = 22
    MAGNITUDE = 23
    SECONDARY_CONFUSE = 24
    MUD_SPORT = 25
    EAT_BERRY = 26
    DISABLE = 27
    BIDE = 28
    ROLLOUT = 29
    LIGHT_SCREEN = 30
    REFLECT = 31
    BRICK_BREAK = 32
    RECOVER = 33
    FAKE_OUT = 34
    RAISE_ALL_STATS_HIT = 35
    CURSE = 36
    SELF_DESTRUCT = 37
    FLAIL = 38
    AQUA_RING = 39
    FANG_PARALYZE_FLINCH = 40
    PRESENT = 41
    PROTECT = 42
    ATTRACT = 43
    CAMOUFLAGE = 44
    NATURE_POWER = 45
    TRI_ATTACK = 46
    ROAR = 47


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
    INSOMNIA = 19
    CLOUD_NINE = 20
    COMPOUNDEYES = 21
    SHED_SKIN = 22
    SAND_VEIL = 23
    SNOW_CLOAK = 24
    SWIFT_SWIM = 25
    ROUGH_SKIN = 26
    FLAME_BODY = 27
    LIMBER = 28
    ADAPTABILITY = 29
    SOLID_ROCK = 30
    CUTE_CHARM = 31


class ItemId(IntEnum):
    """Held-item effect. ``tables.item_param[i, 0]`` is the decomp's hold param
    (Oran Berry, Berry Juice: HP restored; Sitrus Berry: percent of max HP; Big Root: the
    percent added to a drain, 30); column 1 is a resist berry's type (below), else 0.

    * ``ORAN_BERRY``, ``BERRY_JUICE`` (both ``HOLD_EFFECT_HP_RESTORE``), ``SITRUS_BERRY``,
      ``CHESTO_BERRY`` (``SLP_RESTORE``), ``PECHA_BERRY`` (``PSN_RESTORE``, toxic too),
      ``LUM_BERRY`` (``STATUS_RESTORE``, confusion too), ``WHITE_HERB``
      (``STATDOWN_RESTORE``): ``hooks.held_item_check``. Berry Juice is its own id because
      its pluck effect is ``PLUCK_EFFECT_NONE`` and it is no berry, so Pluck / Bug Bite
      leave it (battle_lib.c:5833, ``hooks.pluck_berry``).
    * ``FOCUS_SASH`` (``HOLD_EFFECT_ENDURE``): ``effects.play_hit`` and the confusion
      self-hit (``step._use_move``).
    * ``BIG_ROOT`` (``HOLD_EFFECT_LEECH_BOOST``): drain (``effects._drain``) and Leech Seed
      (``step._leech_seed_check``).
    * ``LAGGING_TAIL`` (``HOLD_EFFECT_PRIORITY_DOWN``): ``order.player_second``.
    * ``QUICK_CLAW`` (``HOLD_EFFECT_SOMETIMES_PRIORITY``, param ``QUICK_CLAW_PARAM``):
      ``order.player_second`` with the holder's ``speedRand`` draw (``BattleState.qc_roll``
      for the turn order, ``Outcome.quick_claw`` for the end-of-turn order).
    * ``RESIST_BERRY`` (Rindo ``HOLD_EFFECT_WEAKEN_SE_GRASS``, Chilan ``..._WEAKEN_NORMAL``):
      ``tables.item_param[i, 1]`` is the type the berry resists, ``RESIST_ANY_NORMAL`` for
      Chilan (any Normal-type hit, no super-effective test); ``effects.resist_berry``
      (``subscript_type_resist_berry.s``).
    """

    UNSUPPORTED = 0
    NONE = 1
    ORAN_BERRY = 2
    SITRUS_BERRY = 3
    FOCUS_SASH = 4
    BIG_ROOT = 5
    BERRY_JUICE = 6
    CHESTO_BERRY = 7
    PECHA_BERRY = 8
    LUM_BERRY = 9
    WHITE_HERB = 10
    LAGGING_TAIL = 11
    RESIST_BERRY = 12
    QUICK_CLAW = 13


# ``item_param[i, 1]`` of Chilan Berry (``HOLD_EFFECT_WEAKEN_NORMAL``): the berry halves any
# hit whose type is Normal, super-effective or not (``subscript_type_resist_berry.s:_128``);
# every other resist berry halves a super-effective hit of its type (``_135``-``_240``).
RESIST_ANY_NORMAL = -1


class TrainerItemId(IntEnum):
    """Effect of a trainer's bag item (``tables.trainer_item_effect``), used by the
    opponent's item actions (``actions.FIRST_ITEM_ACTION``). ``POTION`` restores
    ``POTION_HP_RESTORED`` HP to the active Pokemon, capped at its max HP."""

    UNSUPPORTED = 0
    NONE = 1
    POTION = 2


# Potion's ``itemUseParams.hpRestored`` (res/items/data/potion.json), the amount
# ``BattleSystem_UseBagItem`` (src/battle/battle_system.c:435) restores.
POTION_HP_RESTORED = 20

# Quick Claw's hold param (res/items/data/quick_claw.json ``effectParam`` 20, the only
# ``HOLD_EFFECT_SOMETIMES_PRIORITY`` item): ``BattleSystem_CompareBattlerSpeed`` activates it
# when ``speedRand % (100 / param) == 0`` (battle_lib.c:1278-1286, :1344-1352). The engine
# stores that test's result (``BattleState.qc_roll``, ``Outcome.quick_claw``), which is exact
# only while every holder has this param: ``game_constants`` asserts it for every game.
QUICK_CLAW_PARAM = 20
QUICK_CLAW_MODULUS = 100 // QUICK_CLAW_PARAM  # % 5


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


class MoveTarget(IntEnum):
    """``tables.move_target``: the decomp's ``RANGE_*`` (``generated/move_ranges.txt``
    order, without ``RANGE_UNUSED_*`` and ``RANGE_ALL``, which no move uses)."""

    RANGE_SINGLE_TARGET = 0
    RANGE_SINGLE_TARGET_SPECIAL = 1
    RANGE_RANDOM_OPPONENT = 2
    RANGE_ADJACENT_OPPONENTS = 3
    RANGE_ALL_ADJACENT = 4
    RANGE_USER = 5
    RANGE_USER_SIDE = 6
    RANGE_FIELD = 7
    RANGE_OPPONENT_SIDE = 8
    RANGE_ALLY = 9
    RANGE_USER_OR_ALLY = 10
    RANGE_SINGLE_TARGET_ME_FIRST = 11


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
# Fake Out (``generated/moves.txt`` line 253): ``subscript_flinch_mon.s:18`` shows Inner
# Focus's message, which names the ability, only for ``BTLVAR_CURRENT_MOVE == MOVE_FAKE_OUT``
# (or a direct flinch effect, which no supported move has).
MOVE_FAKE_OUT = 252
# Protect, Detect, Endure (``generated/moves.txt`` lines 183, 198, 204): ``BtlCmd_TryProtection``
# (battle_script.c:5154-5158) keeps the chain only when ``moveProtect`` is one of them, and
# ``AICmd_LoadProtectChain`` (trainer_ai.c:2546-2552) reads it the same way. The engine folds
# ``moveProtect`` into ``protect_count`` (``step._use_move``).
MOVE_PROTECT = 182
MOVE_DETECT = 197
MOVE_ENDURE = 203
PROTECT_CHAIN_MOVES = (MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE)

# ABILITY_MULTITYPE (``generated/abilities.txt`` line 122): ``BtlCmd_TryCamouflage``
# (battle_script.c:8001-8004) fails for a Multitype user, read through
# ``tables.species_abilities_decomp`` (``damage.decomp_abilities``). Multitype itself is an
# unsupported ability, so coverage refuses a holder before this test can matter.
ABILITY_MULTITYPE = 121

# The battle terrain tables (include/data/terrain/to_move.h ``sTerrainMove``, to_type.h
# ``sTerrainCamouflageType``), indexed by the decomp's ``TERRAIN_*`` id
# (``nuzline.data.trainer.TERRAIN_NAMES``) up to ``TERRAIN_SPECIAL``: ``BtlCmd_GetTerrainMove``
# (battle_script.c:8036-8048) and ``BtlCmd_TryCamouflage`` (:8007-8012) read
# ``min(terrain, TERRAIN_SPECIAL)`` (include/constants/battle/terrain.h: ``TERRAIN_SPECIAL`` is
# ``TERRAIN_AARON``, 12). The moves are decomp move ids (``generated/moves.txt`` line N is id
# N-1): Earthquake 89, Seed Bomb 402, Rock Slide 157, Blizzard 59, Hydro Pump 56, Ice Beam 58,
# Tri Attack 161, Mud Bomb 426, Air Slash 403. tests/engine/test_terrain_moves.py reads both
# tables from the decomp and, through the ROM readers, from both ROMs.
TERRAIN_SPECIAL = 12
TERRAIN_MOVES = (89, 89, 402, 402, 157, 157, 59, 56, 58, 161, 426, 403, 161)

# Decomp ability ids the weather code names directly (``generated/abilities.txt`` line N is
# id N-1), read through ``tables.species_abilities_decomp`` (``Battler_Ability``) rather than
# an ``AbilityId``, for the abilities no table maps (Air Lock, Magic Guard, Ice Body) and the
# lock they share with Cloud Nine: Cloud Nine / Air Lock suppress weather (``NO_CLOUD_NINE``,
# include/battle/common.h:63-64); the end-of-turn chip (``BtlCmd_EndOfTurnWeatherEffect``,
# battle_script.c:5561-5647, and ``subscript_weather_continues.s``) skips Ice Body (which
# heals instead) and Magic Guard. Sand Veil and Snow Cloak are ``AbilityId`` members.
WEATHER_ABILITY_IDS = {
    "Cloud Nine": 13,
    "Air Lock": 76,
    "Magic Guard": 98,
    "Ice Body": 115,
}

# ``sTerrainCamouflageType`` (include/data/terrain/to_type.h), indexed like ``TERRAIN_MOVES``.
TERRAIN_TYPES = (
    Type.GROUND,  # TERRAIN_PLAIN
    Type.GROUND,  # TERRAIN_SAND
    Type.GRASS,  # TERRAIN_GRASS
    Type.GRASS,  # TERRAIN_PUDDLE
    Type.ROCK,  # TERRAIN_MOUNTAIN
    Type.ROCK,  # TERRAIN_CAVE
    Type.ICE,  # TERRAIN_SNOW
    Type.WATER,  # TERRAIN_WATER
    Type.ICE,  # TERRAIN_ICE
    Type.NORMAL,  # TERRAIN_BUILDING
    Type.GROUND,  # TERRAIN_GREAT_MARSH
    Type.FLYING,  # TERRAIN_BRIDGE
    Type.NORMAL,  # TERRAIN_SPECIAL
)
# effect_script_0036.s: ``Random 2, 3`` (``BtlCmd_Random``, battle_script.c:3200-3210:
# ``RandNext() % 3 + 3``) picks MOVE_SUBSCRIPT_PTR_BURN, _FREEZE or _PARALYZE (3, 4, 5:
# generated/battle_move_subscript_ptrs.txt lines 4-6), ``Outcome.tri_attack`` 0, 1, 2.
TRI_ATTACK_STATUSES = (Status.BURN, Status.FREEZE, Status.PARALYSIS)
