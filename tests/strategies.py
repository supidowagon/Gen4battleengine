"""Hypothesis strategies for engine tests: random M1 Pokemon, battle states, legal
actions and chance outcomes.

Everything stays inside the M1 implemented set: species from ``M1_SPECIES``, moves
from their level-up learnsets with a supported effect (``move_effect !=
UNSUPPORTED``), their own abilities (all implemented), an implemented held item
(``M1_ITEMS`` and ``ITEM_GROUPS``) or none;
the opponent's bag holds Potions or nothing in each slot.
State fields that no M1 mechanic can set are left at 0, except those the engines
read (screens, Tailwind, Trick Room, Mud Sport, weather), which are drawn so the
code reading them is exercised. Active Pokemon have HP >= 1; benched ones may have
fainted (plan ruling).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import strategies as st

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import TERRAIN_NAMES, TERRAIN_UNKNOWN, IntermediateData
from nuzline.engine.actions import (
    MOVE_ACTIONS,
    legal_actions,
    legal_opponent_actions,
    legal_replacements,
)
from nuzline.engine.coverage import halved_defense_line, low_defense
from nuzline.engine.damage import STATUS as STATUS_CATEGORY
from nuzline.engine.ids import (
    CONFUSION_PENDING,
    MOVE_STRUGGLE,
    NATURE_DISLIKED_FLAVOR,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import (
    BAG_SLOTS,
    MAX_DEATH_BUDGET,
    SLOTS,
    STAGE_MAX,
    STAGE_MIN,
    BattleState,
    make_state,
)
from nuzline.engine.stats import calc_stats

INTERMEDIATE = Path(__file__).resolve().parents[1] / "data" / "intermediate"

# The M1 species set: the sample team plus Roark's and the route-encounter species, a
# holder of each weather ability (Swift Swim, Sand Veil, Snow Cloak), and of Rough Skin
# (Carvanha), Flame Body (Magby), Limber (Glameow), Adaptability (Eevee) and Solid Rock
# (Camerupt): their other slot is empty or unimplemented, so these are always drawn.
M1_SPECIES = (
    "Monferno",
    "Prinplup",
    "Grotle",
    "Starly",
    "Shinx",
    "Budew",
    "Geodude",
    "Onix",
    "Cranidos",
    "Zubat",
    "Bidoof",
    "Furret",
    "Staravia",
    "Spinarak",
    "Psyduck",
    "Venonat",
    "Kricketot",
    "Goldeen",
    "Sandshrew",
    "Swinub",
    "Carvanha",
    "Magby",
    "Glameow",
    "Eevee",
    "Camerupt",
    # E2 batch 3: a holder of each batch-3 ability (the other slot empty, unsupported or
    # supported), weather setters and Trace holders included.
    "Meowth",
    "Hitmonlee",
    "Slakoth",
    "Croagunk",
    "Electrike",
    "Machop",
    "Chansey",
    "Groudon",
    "Combee",
    "Bronzor",
    "Teddiursa",
    "Miltank",
    "Krabby",
    "Kabuto",
    "Wurmple",
    "Sunkern",
    "Murkrow",
    "Pidgey",
    "Ralts",
    "Kyogre",
    "Hippopotas",
    "Snover",
)
M1_ITEMS = (
    "",
    "Oran Berry",
    "Sitrus Berry",
    "Focus Sash",
    "Big Root",
    "Berry Juice",
    "Chesto Berry",
    "Pecha Berry",
    "Lum Berry",
    "White Herb",
    "Lagging Tail",
    "Quick Claw",
    "Rindo Berry",
    "Chilan Berry",
)
# E2 batch 1 (held items by hold effect): drawn as groups next to M1_ITEMS
# (``_held_item``), so that the older items keep their share. Every type-boosting hold effect
# (sTypeBoostingItems: the plates and incenses too), the other damage items of
# BattleSystem_CalcMoveDamage, the Choice items, Expert Belt, Life Orb, every resist berry and
# items with no hold effect (berries, which Pluck eats, and others).
TYPE_BOOST_ITEMS = (
    "Silk Scarf",
    "Black Belt",
    "Sharp Beak",
    "Poison Barb",
    "Soft Sand",
    "Hard Stone",
    "SilverPowder",
    "Spell Tag",
    "Metal Coat",
    "Charcoal",
    "Mystic Water",
    "Miracle Seed",
    "Magnet",
    "TwistedSpoon",
    "NeverMeltIce",
    "Dragon Fang",
    "BlackGlasses",
    "Sea Incense",
    "Wave Incense",
    "Rock Incense",
    "Rose Incense",
    "Odd Incense",
    "Flame Plate",
    "Splash Plate",
    "Zap Plate",
    "Meadow Plate",
    "Icicle Plate",
    "Fist Plate",
    "Toxic Plate",
    "Earth Plate",
    "Sky Plate",
    "Mind Plate",
    "Insect Plate",
    "Stone Plate",
    "Spooky Plate",
    "Draco Plate",
    "Dread Plate",
    "Iron Plate",
)
DAMAGE_ITEMS = (
    "Choice Band",
    "Choice Specs",
    "Choice Scarf",
    "Muscle Band",
    "Wise Glasses",
    "Expert Belt",
    "Life Orb",
    "Light Ball",
    "Thick Club",
    "Metal Powder",
    "DeepSeaTooth",
    "DeepSeaScale",
    "Soul Dew",
    "Adamant Orb",
    "Lustrous Orb",
    "Griseous Orb",
)
RESIST_BERRIES = (
    "Occa Berry",
    "Passho Berry",
    "Wacan Berry",
    "Rindo Berry",
    "Yache Berry",
    "Chople Berry",
    "Kebia Berry",
    "Shuca Berry",
    "Coba Berry",
    "Payapa Berry",
    "Tanga Berry",
    "Charti Berry",
    "Kasib Berry",
    "Haban Berry",
    "Colbur Berry",
    "Babiri Berry",
    "Chilan Berry",
)
NO_EFFECT_ITEMS = ("Razz Berry", "Nanab Berry", "Belue Berry", "Nugget", "Blue Scarf", "Bicycle")
# E2 batch 2: the status berries, the stat pinch berries and Custap, the residual items
# (Leftovers, Black Sludge, the status orbs, Sticky Barb), the accuracy items, King's Rock /
# Razor Fang, Shell Bell and Light Clay. ``_draw_batch2_item`` also gives one to an active
# Pokemon on purpose, with what makes it act.
PINCH_BERRIES = (
    "Liechi Berry",
    "Ganlon Berry",
    "Salac Berry",
    "Petaya Berry",
    "Apicot Berry",
    "Custap Berry",
)
BATCH2_ITEMS = (
    "Cheri Berry",
    "Rawst Berry",
    "Aspear Berry",
    "Persim Berry",
    *PINCH_BERRIES,
    "Leftovers",
    "Black Sludge",
    "Toxic Orb",
    "Flame Orb",
    "Sticky Barb",
    "BrightPowder",
    "Lax Incense",
    "Wide Lens",
    "Zoom Lens",
    "King’s Rock",
    "Razor Fang",
    "Shell Bell",
    "Light Clay",
)
# E2 batch 3: the weather rocks (their holder's weather move lasts longer).
WEATHER_ROCKS = ("Damp Rock", "Heat Rock", "Smooth Rock", "Icy Rock")
# E1 Gardenia batch (G-1, G-4): Focus Band and the flavor berries; ``_draw_gardenia_item`` also
# gives one to an active Pokemon on purpose, with what makes it act.
FLAVOR_BERRIES = ("Figy Berry", "Wiki Berry", "Mago Berry", "Aguav Berry", "Iapapa Berry")
GARDENIA_ITEMS = ("Focus Band", *FLAVOR_BERRIES)
ITEM_GROUPS = (
    TYPE_BOOST_ITEMS,
    DAMAGE_ITEMS,
    RESIST_BERRIES,
    NO_EFFECT_ITEMS,
    BATCH2_ITEMS,
    WEATHER_ROCKS,
    GARDENIA_ITEMS,
)
# E2 batch 3: abilities ``_draw_batch3`` puts on an active Pokemon (the battler's ability; a
# switch-out restores its own) and the switch-in ones it puts on a bench Pokemon.
BATCH3_ABILITIES = (
    AbilityId.RECKLESS,
    AbilityId.TECHNICIAN,
    AbilityId.TRUANT,
    AbilityId.DRY_SKIN,
    AbilityId.LIGHTNINGROD,
    AbilityId.NO_GUARD,
    AbilityId.SERENE_GRACE,
    AbilityId.HONEY_GATHER,
    AbilityId.LEVITATE,
    AbilityId.QUICK_FEET,
    AbilityId.SCRAPPY,
    AbilityId.SHELL_ARMOR,
    AbilityId.BATTLE_ARMOR,
    AbilityId.SHIELD_DUST,
    AbilityId.SOLAR_POWER,
    AbilityId.SUPER_LUCK,
    AbilityId.TANGLED_FEET,
    AbilityId.THICK_FAT,
    AbilityId.TINTED_LENS,
)
# E2 batch 4: the moves ``_draw_batch4`` gives the active Pokemon (vanilla effects: SmellingSalt
# 171, Facade 169, Brine 221, Wring Out / Crush Grip 237, Grass Knot / Low Kick 196, Super Fang
# 40, the OHKO moves 38, False Swipe 101, Swagger 118, Dragon Dance 212, Poison Fang 202,
# Earthquake 147), its abilities and the crit items.
BATCH4_MOVES = (
    "SmellingSalt",
    "Facade",
    "Brine",
    "Wring Out",
    "Crush Grip",
    "Grass Knot",
    "Low Kick",
    "Super Fang",
    "Horn Drill",
    "Guillotine",
    "Fissure",
    "False Swipe",
    "Swagger",
    "Dragon Dance",
    "Poison Fang",
    "Earthquake",
    "Snore",
    "Secret Power",
    "Counter",
    "Mirror Coat",
    "Metal Burst",
)
BATCH4_ABILITIES = (AbilityId.FILTER, AbilityId.MAGIC_GUARD, AbilityId.SNIPER)
# E2 batch 5: the invokers, and moves for them to call or copy (a sleep move, Protect, a
# multi-hit move, Disable, a Choice item's lock).
INVOKERS = ("Sleep Talk", "Sleep Talk", "Copycat", "Metronome")
CALLED_MOVES = ("Tackle", "Spore", "Protect", "Fury Swipes", "Disable", "Counter", "Ember", "Bide")
CRIT_ITEMS = ("Scope Lens", "Razor Claw", "Lucky Punch", "Stick")
SWITCH_IN_ABILITIES = (
    AbilityId.TRACE,
    AbilityId.DROUGHT,
    AbilityId.DRIZZLE,
    AbilityId.SAND_STREAM,
    AbilityId.SNOW_WARNING,
    AbilityId.PRESSURE,  # its switch-in announcement (G-8)
)
# E1 Gardenia batch (G-8): moves of each Pressure range class ``_draw_pressure`` gives the foe of
# a Pressure holder (single target, field, opponent side, adjacent opponents, all adjacent, the
# user, Imprison by id, the Counter family and Pursuit's extra PP; a Sleep Talk / Copycat call).
PRESSURE_MOVES = (
    "Tackle",
    "Sunny Day",
    "Spikes",
    "Toxic Spikes",
    "Growl",
    "Earthquake",
    "Swords Dance",
    "Imprison",
    "Counter",
    "Mirror Coat",
    "Metal Burst",
    "Pursuit",
    "Sleep Talk",
)
# The species items' holders (BattleSystem_CalcMoveDamage, battle_lib.c:6730-6774), none of
# them an M1 species: ``_draw_species_item`` makes an active Pokemon one of them, holding the
# item, with the rest of its data (stats, types, ability) kept, which both engines read alike.
# Each orb's holder also gets a move of a type its orb boosts in slot 0.
SPECIES_ITEMS = (
    ("Pikachu", "Light Ball", ()),
    ("Cubone", "Thick Club", ()),
    ("Marowak", "Thick Club", ()),
    ("Ditto", "Metal Powder", ()),
    ("Clamperl", "DeepSeaTooth", ()),
    ("Clamperl", "DeepSeaScale", ()),
    ("Latias", "Soul Dew", ()),
    ("Latios", "Soul Dew", ()),
    ("Dialga", "Adamant Orb", ("Dragon Claw", "Metal Claw")),
    ("Palkia", "Lustrous Orb", ("DragonBreath", "Water Gun")),
    ("Giratina", "Griseous Orb", ("Dragon Claw", "Shadow Ball")),
)
# Items whose effect needs their holder to act (the Choice lock and Scarf, Life Orb's recoil,
# Expert Belt): ``_draw_battle_item`` gives them to an active Pokemon on purpose.
BATTLE_ITEMS = (
    "Life Orb",
    "Choice Band",
    "Choice Specs",
    "Choice Scarf",
    "Expert Belt",
    "Muscle Band",
    "Wise Glasses",
)
M1_BAG_ITEMS = ("", "Potion")  # the opponent's bag (ids.TrainerItemId)
# Items that act on their holder's own damaging move (``legal_action_pair`` favours one).
HITTER_ITEMS = (ItemId.KINGS_ROCK, ItemId.SHELL_BELL, ItemId.ZOOM_LENS, ItemId.WIDE_LENS)

# decomp gender ratios (``tables.species_gender``) with a single possible gender
GENDERLESS, MALE_ONLY, FEMALE_ONLY = 255, 0, 254
_SLEEP_TURNS = (1, 5)  # BtlCmd_Random 3, 2 stores 2-5; a counter of 1 wakes next turn
_TOXIC_MAX = 15  # step.TOXIC_COUNTER_MAX
_CONFUSION_MAX, _TAUNT_MAX, _BIND_MAX = 5, 5, 6  # outcome.sample_outcome's duration ranges
_DURATION = (2, 6)  # outcome.duration over every move: 2-5 sleep/confusion, 3-6 Bind, 3-5 Taunt
_DISABLE_MAX = 6  # Disable stores RandNext() % 4 + 3
_TIMER_MAX = 5  # screens, Tailwind, Trick Room: turns left


@lru_cache(maxsize=1)
def _learnsets() -> dict[int, list[tuple[int, int]]]:
    if not (INTERMEDIATE / "learnsets.json").exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")
    return IntermediateData(INTERMEDIATE).learnsets


# The fixture and effect-test moves, given to every species so the infliction,
# trapping, flinch, multi-hit (2-5 and fixed two), stat-stage, Recover, Fake Out, Curse,
# screen (Reflect, Light Screen, broken by Brick Break), Toxic, mid-turn accuracy
# (Sand-Attack, Double Team) and Thunder Fang (paralysis then flinch) branches run in the
# differential test (no M1 species is Ghost-type: Curse is the non-Ghost one).
SHARED_MOVES = (
    "Stealth Rock",
    "Taunt",
    "Leech Seed",
    "Bind",
    "Pursuit",
    "Supersonic",
    "Headbutt",
    "Rock Polish",
    "Fury Swipes",
    "Thunder Wave",
    "Stun Spore",
    "Foresight",
    "Night Shade",
    "Seismic Toss",
    "Hidden Power",
    "Magnitude",
    "Confusion",
    "Mud Sport",
    "Bug Bite",
    "Disable",
    "Bide",
    "Rollout",
    "Ice Ball",
    "Defense Curl",
    "Recover",
    "Fake Out",
    "AncientPower",
    "Curse",
    "Double Kick",
    "Selfdestruct",
    "Flail",
    "Aqua Ring",
    "Light Screen",
    "Reflect",
    "Brick Break",
    "Toxic",
    "Sand-Attack",
    "Double Team",
    "Thunder Fang",
    # K3b2 Task 10: the terrain moves (status moves: the damage oracle's damaging pools are
    # unchanged); supported_state draws a terrain whenever one is known.
    "Nature Power",
    "Camouflage",
    # K3b2 Task 11: Roar and Whirlwind (status moves: the damaging pools are unchanged).
    "Roar",
    "Whirlwind",
    # E2 batch 3: the weather moves (status moves: the damaging pools are unchanged).
    "Sunny Day",
    "Rain Dance",
    "Sandstorm",
    "Hail",
    # E2 batch 4: status moves (the damaging pools are unchanged).
    "Roost",
    "Tailwind",
    "Spikes",
    "Toxic Spikes",
    "Destiny Bond",
)


def m1_move_pools(tables: Tables, names: Names) -> dict[int, tuple[int, ...]]:
    """Species id -> its level-up learnset moves with a supported effect plus
    ``SHARED_MOVES``, in id order."""
    effect = np.asarray(tables.move_effect)
    learnsets = _learnsets()
    shared = {names.move(m) for m in SHARED_MOVES}
    unsupported = sorted(m for m in shared if effect[m] == EffectId.UNSUPPORTED)
    if unsupported:
        raise ValueError(f"SHARED_MOVES has unsupported moves: {unsupported}")
    pools = {}
    for name in M1_SPECIES:
        species = names.species(name)
        if Type.GHOST in np.asarray(tables.species_types[species]).tolist():
            # Curse is shared: a Ghost user takes the branch coverage refuses.
            raise ValueError(f"M1 species {name} is Ghost-type")
        moves = {m for _, m in learnsets[species] if effect[m] != EffectId.UNSUPPORTED}
        moves |= shared
        pools[species] = tuple(sorted(moves))
    return pools


def _max_hp(tables: Tables, species, ivs, evs, level, nature) -> int:
    stats = calc_stats(
        np.asarray(tables.species_base_stats[species]),
        np.asarray(ivs),
        np.asarray(evs),
        level,
        np.asarray(tables.nature_mods[nature]),
    )
    return int(stats[Stat.HP])


def possible_genders(ratio: int) -> tuple[int, ...]:
    """Genders (0 male, 1 female, 2 genderless) a species with this decomp ratio can have."""
    if ratio == GENDERLESS:
        return (2,)
    if ratio == MALE_ONLY:
        return (0,)
    if ratio == FEMALE_ONLY:
        return (1,)
    return (0, 1)


def _held_item() -> st.SearchStrategy[str]:
    """Half the time an ``M1_ITEMS`` name, else one of an ``ITEM_GROUPS`` group."""
    groups = st.sampled_from(ITEM_GROUPS).flatmap(st.sampled_from)
    return st.one_of(st.sampled_from(M1_ITEMS), groups)


@st.composite
def supported_pokemon(draw, tables: Tables, names: Names) -> PokemonSpec:
    """A random M1 Pokemon: level 5-20, random nature, IVs, EVs (at most 510 in all),
    1-4 supported learnset moves, its own ability, an M1 item or none, any current HP
    including 0 (full HP drawn often: Focus Sash needs it), and a random status (none when
    fainted)."""
    pools = m1_move_pools(tables, names)
    species = draw(st.sampled_from(sorted(pools)))
    level = draw(st.integers(5, 20))
    nature = draw(st.integers(0, len(names.nature_names) - 1))
    ivs = tuple(draw(st.lists(st.integers(0, 31), min_size=6, max_size=6)))
    evs = tuple(draw(st.lists(st.integers(0, 85), min_size=6, max_size=6)))
    moves = draw(st.lists(st.sampled_from(pools[species]), min_size=1, max_size=4, unique=True))
    abilities = names.species_abilities(species)
    # a slot whose ability is not implemented (Run Away, Damp, ...) is never drawn
    supported = np.asarray(tables.species_abilities[species]) != AbilityId.UNSUPPORTED
    ability_slot = draw(st.sampled_from([i for i, a in enumerate(abilities) if a and supported[i]]))
    item = names.item(draw(_held_item()))
    gender = draw(st.sampled_from(possible_genders(int(tables.species_gender[species]))))
    max_hp = _max_hp(tables, species, ivs, evs, level, nature)
    hp = draw(st.one_of(st.just(max_hp), st.integers(0, max_hp)))
    status = draw(st.sampled_from(list(Status))) if hp > 0 else Status.NONE
    return PokemonSpec(
        species=species,
        level=level,
        nature=nature,
        ivs=ivs,
        evs=evs,
        moves=tuple(moves + [0] * (4 - len(moves))),
        item=item,
        ability_slot=ability_slot,
        gender=gender,
        friendship=draw(st.integers(0, 255)),
        current_hp=hp,
        status=int(status),
    )


def _status_counter(draw, status: int) -> int:
    if status == Status.SLEEP:
        return draw(st.integers(*_SLEEP_TURNS))
    if status == Status.TOXIC:
        return draw(st.integers(0, _TOXIC_MAX))
    return 0


@st.composite
def _party(draw, tables: Tables, names: Names) -> tuple[list[PokemonSpec], int]:
    """1-6 Pokemon and the active slot, whose Pokemon has HP >= 1."""
    party = draw(st.lists(supported_pokemon(tables, names), min_size=1, max_size=SLOTS))
    active = draw(st.integers(0, len(party) - 1))
    lead = party[active]
    if lead.current_hp == 0:
        max_hp = _max_hp(tables, lead.species, lead.ivs, lead.evs, lead.level, lead.nature)
        hp = draw(st.integers(1, max_hp))
        party[active] = replace(
            lead, current_hp=hp, status=int(draw(st.sampled_from(list(Status))))
        )
    return party, active


# TERRAIN_UNKNOWN and every TERRAIN_* id (nuzline.data.trainer.TERRAIN_NAMES, 0-24).
TERRAINS = (TERRAIN_UNKNOWN, *range(len(TERRAIN_NAMES)))


def _draw_terrain(draw, tables: Tables, a: dict) -> None:
    """A terrain (``TERRAINS``) when any Pokemon knows a move that reads it (Nature Power,
    Camouflage), so that both engines play those moves on a known terrain too; otherwise the
    state keeps ``make_state``'s ``TERRAIN_UNKNOWN`` (nothing reads it)."""
    effect = np.asarray(tables.move_effect)[a["moves"]]
    readers = (EffectId.NATURE_POWER, EffectId.CAMOUFLAGE, EffectId.SECRET_POWER)
    if np.isin(effect, readers).any():
        a["terrain"][...] = draw(st.sampled_from(TERRAINS))


def _draw_battle_item(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 2) give the active Pokemon one of ``BATTLE_ITEMS``, or a
    ``TYPE_BOOST_ITEMS`` item of the type of one of its moves (so the boost applies)."""
    if draw(st.booleans()):
        return
    # Expert Belt weighs three: it acts only on a super-effective hit.
    choice = draw(st.sampled_from((*BATTLE_ITEMS, "Expert Belt", "Expert Belt", "type")))
    if choice == "type":
        move = int(draw(st.sampled_from([int(m) for m in a["moves"][side, active] if m])))
        kind = int(tables.move_type[move])
        fits = [n for n in TYPE_BOOST_ITEMS if int(tables.item_param[names.item(n), 1]) == kind]
        if not fits:  # a type no item boosts (the ??? type)
            return
        choice = draw(st.sampled_from(fits))
    a["item"][side, active] = names.item(choice)


def _draw_species_item(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 4) make the active Pokemon a ``SPECIES_ITEMS`` species holding its item
    (the species items only work for their species), an orb's holder with a move its orb
    boosts in slot 0 (full PP). Its stats and max HP are the new species' (``calc_stats`` with
    its IVs, EVs, level and nature: an exp payout's level-up recomputes them from the species,
    so a stale max HP could fall), its HP capped at the new max."""
    if draw(st.integers(0, 3)) != 0:
        return
    species, item, moves = draw(st.sampled_from(SPECIES_ITEMS))
    sid = names.species(species)
    a["species"][side, active] = sid
    a["item"][side, active] = names.item(item)
    stats = calc_stats(
        np.asarray(tables.species_base_stats[sid]),
        a["ivs"][side, active],
        a["evs"][side, active],
        int(a["level"][side, active]),
        np.asarray(tables.nature_mods[int(a["nature"][side, active])]),
        sid,
    )
    a["max_hp"][side, active] = stats[Stat.HP]
    a["stats"][side, active] = stats[Stat.ATK :]
    a["hp"][side, active] = min(int(a["hp"][side, active]), int(stats[Stat.HP]))
    if moves:
        move = names.move(draw(st.sampled_from(moves)))
        a["moves"][side, active, 0] = move
        a["pp"][side, active, 0] = int(tables.move_pp[move])


def _draw_pursuit(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 6) put Pursuit (full PP) in a move slot of the active Pokemon, so that
    ``legal_action_pair``'s Pursuit-against-a-switch draw has a pursuer."""
    if draw(st.integers(0, 5)) != 0:
        return
    pursuit = names.move("Pursuit")
    k = draw(st.integers(0, 3))
    a["moves"][side, active, k] = pursuit
    a["pp"][side, active, k] = int(tables.move_pp[pursuit])


def _draw_pivot(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """E2 batch 6: maybe (1 in 2) put U-turn (two times in three) or Baton Pass (full PP) in a
    move slot of the active Pokemon; ``legal_action_pair`` uses it three draws in four, against
    a Pursuit one draw in four when the foe has one. With a stage or a kept volatile drawn
    elsewhere, a Baton Pass carries it."""
    if draw(st.booleans()):
        return
    name = "Baton Pass" if draw(st.integers(0, 2)) == 0 else "U-turn"
    move = names.move(name)
    # Not over a move the battler is locked into (Bide, Rollout, Uproar: its lock reads it).
    free = [k for k in range(4) if a["moves"][side, active, k] != a["locked_move"][side]]
    if not free:
        return
    k = draw(st.sampled_from(free))
    a["moves"][side, active, k] = move
    a["pp"][side, active, k] = int(tables.move_pp[move])
    # Some of the time make the pivot reachable: a living bench (a fainted slot revived to full
    # HP, unless it is a recorded death) and a user that can act (no sleep, freeze, confusion
    # or love).
    if draw(st.booleans()):
        bench = [j for j in range(6) if j != active and a["species"][side, j]]
        alive = [j for j in bench if a["hp"][side, j] > 0]
        dead = [j for j in bench if a["hp"][side, j] == 0 and not (side == 0 and a["deaths"][j])]
        if not alive and dead:
            j = draw(st.sampled_from(dead))
            a["hp"][side, j] = a["max_hp"][side, j]
        if a["status"][side, active] in (Status.SLEEP, Status.FREEZE):
            a["status"][side, active] = Status.NONE
            a["status_counter"][side, active] = 0
        a["confusion"][side] = a["infatuated"][side] = 0


def _draw_batch2_item(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 2) give the active Pokemon a ``BATCH2_ITEMS`` item, picked uniformly
    (``st.randoms``: ``sampled_from`` favours its first entries), with what makes it act, each
    most of the time: a stat pinch berry or Custap Berry at or below its trigger (``maxHP /
    4``, or ``maxHP / 2`` under Gluttony, which it has half the time: ``hooks.held_item_check``,
    ``order.custap_active``), a status berry with the status it cures (Persim: confusion), a
    status orb on a Pokemon with no status, Leftovers, Black Sludge or Shell Bell below full HP,
    Light Clay with Reflect or Light Screen in a slot (full PP; no screen up, no Taunt), Tackle (95 % accurate, contact,
    ``MOVE_FLAG_TRIGGERS_KINGS_ROCK``) in a slot of a King's Rock, Razor Fang, lens or Shell
    Bell holder and of the foe of a Sticky Barb holder (whose item is cleared: the transfer
    needs an attacker holding nothing), and a Poison type for a Black Sludge holder."""
    if not draw(st.booleans()):
        return
    rnd = draw(st.randoms(use_true_random=False))
    name = rnd.choice(BATCH2_ITEMS)
    a["item"][side, active] = names.item(name)
    max_hp = int(a["max_hp"][side, active])
    if name in PINCH_BERRIES:
        if rnd.random() < 0.5:
            a["ability"][side, active] = AbilityId.GLUTTONY
        divisor = 4 // (2 if a["ability"][side, active] == AbilityId.GLUTTONY else 1)
        if rnd.random() < 0.75:
            a["hp"][side, active] = rnd.randint(1, max(max_hp // divisor, 1))
    cures = {
        "Cheri Berry": Status.PARALYSIS,
        "Rawst Berry": Status.BURN,
        "Aspear Berry": Status.FREEZE,
    }
    if name in cures and rnd.random() < 0.75:
        a["status"][side, active] = cures[name]
        a["status_counter"][side, active] = 0
    if name == "Persim Berry" and rnd.random() < 0.75:
        a["confusion"][side] = rnd.randint(1, _CONFUSION_MAX)
    if name in ("Toxic Orb", "Flame Orb") and rnd.random() < 0.75:
        a["status"][side, active] = Status.NONE
        a["status_counter"][side, active] = 0
    if name in ("Leftovers", "Black Sludge", "Shell Bell") and rnd.random() < 0.75 and max_hp > 1:
        a["hp"][side, active] = rnd.randint(1, max_hp - 1)
    if name == "Black Sludge" and rnd.random() < 0.5:
        a["types"][side, active, 1] = Type.POISON

    def give(who: int, mon: int, move_name: str) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][who, mon, k] = move
        a["pp"][who, mon, k] = int(tables.move_pp[move])

    if name == "Light Clay":
        give(side, active, rnd.choice(("Reflect", "Light Screen")))
        a["reflect"][side] = a["light_screen"][side] = a["taunt"][side] = 0
    hitters = ("King’s Rock", "Razor Fang", "Wide Lens", "Zoom Lens", "Shell Bell")
    if name in hitters and rnd.random() < 0.75:
        give(side, active, "Tackle")
    if name == "Sticky Barb" and rnd.random() < 0.75:
        foe = int(a["active"][1 - side])
        a["item"][1 - side, foe] = 0
        give(1 - side, foe, "Tackle")


def _draw_gardenia_item(
    draw, tables: Tables, a: dict, side: int, active: int, names: Names
) -> None:
    """Maybe (1 in 3) give the active Pokemon a ``GARDENIA_ITEMS`` item, picked uniformly, with
    what makes it act most of the time: a flavor berry at or near half HP and (half the time) a
    nature that dislikes it (``ids.NATURE_DISLIKED_FLAVOR``), the foe sometimes with Bug Bite to
    eat it; a Focus Band holder at low HP or confused (its self-hit roll). Sometimes the active
    Pokemon's confusion is a flavor berry's pending counter (``CONFUSION_PENDING``, G-5)."""
    if draw(st.integers(0, 2)) != 0:
        return
    rnd = draw(st.randoms(use_true_random=False))
    if rnd.random() < 0.15:
        a["confusion"][side] = CONFUSION_PENDING
    name = rnd.choice(GARDENIA_ITEMS)
    a["item"][side, active] = names.item(name)
    max_hp = int(a["max_hp"][side, active])
    if rnd.random() < 0.6:
        a["hp"][side, active] = rnd.randint(1, max(max_hp * 3 // 4, 1))
    if name in FLAVOR_BERRIES:
        flavor = FLAVOR_BERRIES.index(name)
        if rnd.random() < 0.5:
            disliking = [k for k, f in enumerate(NATURE_DISLIKED_FLAVOR) if f == flavor]
            a["nature"][side, active] = rnd.choice(disliking)
        if rnd.random() < 0.3:
            foe = 1 - side
            foe_mon = int(a["active"][foe])
            move = names.move(rnd.choice(("Bug Bite", "Pluck")))
            k = rnd.randint(0, 3)
            a["moves"][foe, foe_mon, k] = move
            a["pp"][foe, foe_mon, k] = int(tables.move_pp[move])
    elif rnd.random() < 0.3:
        a["confusion"][side] = rnd.randint(2, _CONFUSION_MAX)


def _draw_batch3(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 2) put a ``BATCH3_ABILITIES`` ability on the active Pokemon, picked uniformly,
    with what makes it act most of the time: a recoil move for Reckless, a low-power hit for
    Technician and Serene Grace's secondary, a Water and a Fire move on the foe for Dry Skin, a
    Fire / Ice move on the foe for Thick Fat, a Ground move on the foe for Levitate, a hit for
    Tinted Lens, a Ghost type on the foe for Scrappy (unless it knows Curse), sun and a special
    move for Solar Power, confusion for Tangled Feet, a secondary-effect move on the foe for
    Shield Dust. And maybe (1 in 3) a ``SWITCH_IN_ABILITIES`` ability on a bench Pokemon."""
    rnd = draw(st.randoms(use_true_random=False))
    foe = 1 - side
    foe_mon = int(a["active"][foe])

    def give(who: int, mon: int, move_name: str) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][who, mon, k] = move
        a["pp"][who, mon, k] = int(tables.move_pp[move])

    if draw(st.booleans()):
        ability = rnd.choice(BATCH3_ABILITIES)
        a["ability"][side, active] = ability
        enable = rnd.random() < 0.75
        if enable and ability == AbilityId.RECKLESS:
            give(side, active, rnd.choice(("Take Down", "Head Smash", "Double-Edge")))
        if enable and ability in (AbilityId.TECHNICIAN, AbilityId.TINTED_LENS):
            give(side, active, rnd.choice(("Tackle", "Water Gun", "Ember")))
        if enable and ability == AbilityId.SERENE_GRACE:
            give(side, active, rnd.choice(("Headbutt", "Ember", "Confusion", "Mud-Slap")))
        if enable and ability == AbilityId.DRY_SKIN:
            give(foe, foe_mon, rnd.choice(("Water Gun", "Ember")))
        if enable and ability == AbilityId.THICK_FAT:
            give(foe, foe_mon, rnd.choice(("Ember", "Ice Ball")))
        if enable and ability == AbilityId.LEVITATE:
            give(foe, foe_mon, rnd.choice(("Mud-Slap", "Magnitude")))
        if enable and ability == AbilityId.SHIELD_DUST:
            give(foe, foe_mon, rnd.choice(("Headbutt", "Ember", "Confusion", "Thunder Fang")))
        if enable and ability == AbilityId.SCRAPPY:
            give(side, active, "Tackle")
            if names.move("Curse") not in a["moves"][foe, foe_mon]:
                a["types"][foe, foe_mon, 0] = Type.GHOST
        if enable and ability == AbilityId.SOLAR_POWER:
            give(side, active, "Ember")
            a["weather"][...] = Weather.SUN
        if enable and ability == AbilityId.TRUANT:  # a loafing turn (order.truant_loafs)
            a["fake_out_turn"][side] = (int(a["turn"][()]) + 1) % 2
        if enable and ability == AbilityId.TANGLED_FEET:
            a["confusion"][side] = rnd.randint(1, _CONFUSION_MAX)
    if draw(st.integers(0, 5)) == 0:  # a weather rock with its own weather move
        rock, move = rnd.choice(
            (
                ("Damp Rock", "Rain Dance"),
                ("Heat Rock", "Sunny Day"),
                ("Smooth Rock", "Sandstorm"),
                ("Icy Rock", "Hail"),
            )
        )
        a["item"][side, active] = names.item(rock)
        give(side, active, move)
    bench = [
        slot
        for slot in range(a["species"].shape[1])
        if slot != active and a["species"][side, slot] and a["hp"][side, slot]
    ]
    if bench and draw(st.integers(0, 2)) == 0:
        a["ability"][side, rnd.choice(bench)] = rnd.choice(SWITCH_IN_ABILITIES)


def _draw_pressure(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 4) give the active Pokemon Pressure (G-8), its foe a ``PRESSURE_MOVES`` move
    (kept only when supported) and, half the time, 1-2 PP left in each of the foe's moves, so a
    Pressure cost can run a move out."""
    if draw(st.integers(0, 3)) != 0:
        return
    rnd = draw(st.randoms(use_true_random=False))
    a["ability"][side, active] = AbilityId.PRESSURE
    foe = 1 - side
    foe_mon = int(a["active"][foe])
    move = names.move(rnd.choice(PRESSURE_MOVES))
    if tables.move_effect[move] != EffectId.UNSUPPORTED:
        k = rnd.randint(0, 3)
        a["moves"][foe, foe_mon, k] = move
        a["pp"][foe, foe_mon, k] = int(tables.move_pp[move])
    if rnd.random() < 0.5:
        for k in range(4):
            if a["moves"][foe, foe_mon, k]:
                a["pp"][foe, foe_mon, k] = rnd.randint(1, 2)


# E2 batch 9: abilities ``_draw_batch9`` puts on the active Pokemon.
BATCH9_ABILITIES = (
    AbilityId.CHLOROPHYLL,
    AbilityId.HUGE_POWER,
    AbilityId.PURE_POWER,
    AbilityId.SWARM,
    AbilityId.UNBURDEN,
    AbilityId.FLOWER_GIFT,
    AbilityId.VOLT_ABSORB,
    AbilityId.WATER_ABSORB,
    AbilityId.RAIN_DISH,
    AbilityId.CLEAR_BODY,
    AbilityId.HYPER_CUTTER,
    AbilityId.AFTERMATH,
    AbilityId.POISON_HEAL,
    AbilityId.BAD_DREAMS,
    AbilityId.ARENA_TRAP,
    AbilityId.SHADOW_TAG,
    AbilityId.MAGNET_PULL,
)


def _draw_batch9(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 2) put a ``BATCH9_ABILITIES`` ability on the active Pokemon with what makes it
    act most of the time: sun for Chlorophyll and Flower Gift, a Bug move at a third of its HP
    for Swarm, a physical hit for Huge Power / Pure Power, for Unburden a lost item (canUnburden,
    no item) or a held berry it can eat (the flag up, as its switch-in check left it)."""
    rnd = draw(st.randoms(use_true_random=False))

    def give(move_name: str) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][side, active, k] = move
        a["pp"][side, active, k] = int(tables.move_pp[move])

    if draw(st.integers(0, 2)) == 0:  # B9-1, B9-2: a weather-accuracy move or a fang
        give(
            rnd.choice(
                (
                    "Thunder",
                    "Blizzard",
                    "Fire Fang",
                    "Ice Fang",
                    "Double-Edge",
                    "Surf",
                    "Follow Me",
                    "Helping Hand",
                    "Dragon Rage",
                    "SonicBoom",
                    "Minimize",
                    "Stomp",
                    "Belly Drum",
                    "Bulk Up",
                    "Dream Eater",
                    "Payback",
                )
            )
        )
        if rnd.random() < 0.5:
            a["weather"][...] = rnd.choice((Weather.RAIN, Weather.SUN, Weather.HAIL))
    a["minimized"][side] = draw(st.sampled_from([0, 0, 1]))
    if not draw(st.booleans()):
        return
    ability = rnd.choice(BATCH9_ABILITIES)
    a["ability"][side, active] = ability
    if rnd.random() >= 0.75:
        return
    if ability in (AbilityId.CHLOROPHYLL, AbilityId.FLOWER_GIFT):
        a["weather"][...] = Weather.SUN
        give("Tackle")
    if ability == AbilityId.SWARM:
        give("Bug Bite")
        a["hp"][side, active] = max(1, int(a["max_hp"][side, active]) // 3)
    if ability in (AbilityId.HUGE_POWER, AbilityId.PURE_POWER):
        give("Tackle")
    foe = 1 - side
    foe_mon = int(a["active"][foe])

    def give_foe(move_name: str) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][foe, foe_mon, k] = move
        a["pp"][foe, foe_mon, k] = int(tables.move_pp[move])

    if ability == AbilityId.VOLT_ABSORB:
        give_foe(rnd.choice(("ThunderShock", "Thunder Wave", "Thunder Fang")))
    if ability == AbilityId.WATER_ABSORB:
        give_foe(rnd.choice(("Water Gun", "Surf")))
    if ability == AbilityId.RAIN_DISH:
        a["weather"][...] = Weather.RAIN
        a["hp"][side, active] = max(1, int(a["max_hp"][side, active]) // 2)
    if ability == AbilityId.POISON_HEAL:
        a["status"][side, active] = rnd.choice((Status.POISON, Status.TOXIC))
        a["status_counter"][side, active] = 0
    if ability == AbilityId.BAD_DREAMS:
        a["status"][foe, foe_mon] = Status.SLEEP
        a["status_counter"][foe, foe_mon] = rnd.randint(2, 5)
    if ability == AbilityId.AFTERMATH:  # a contact hit that can KO it
        give_foe(rnd.choice(("Tackle", "Scratch", "Headbutt")))
        a["hp"][side, active] = rnd.randint(1, 3)
    if ability in (AbilityId.CLEAR_BODY, AbilityId.HYPER_CUTTER):
        give_foe(rnd.choice(("Growl", "Leer", "Sand-Attack", "Aurora Beam")))
    if ability == AbilityId.UNBURDEN:
        if rnd.random() < 0.5:
            a["item"][side, active] = 0
            a["can_unburden"][side] = 1
        else:
            a["item"][side, active] = names.item(
                rnd.choice(("Oran Berry", "Sitrus Berry", "Lum Berry"))
            )
            # A holder with an item has canUnburden set: its switch-in check armed it
            # (battle_lib.c:4433-4437) and nothing clears it while it stays in (B9-13).
            a["can_unburden"][side] = 1


def _draw_batch10(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 3) give the active Pokemon a batch-10 move with what makes it act: Volt
    Tackle / Flare Blitz (recoil and status; a frozen user for Flare Blitz's thaw, Rock Head or
    Reckless), Jump Kick / Hi Jump Kick (a crash on a miss: a Protect or a Ghost foe, Magic
    Guard, a foe at low max HP for the cap), Endeavor / Pain Split (B10-5; a low user), Knock Off
    (B10-6; an item on the foe, a knocked-off user with no item for the barb)."""
    if draw(st.integers(0, 2)) != 0:
        return
    rnd = draw(st.randoms(use_true_random=False))
    foe = 1 - side
    foe_mon = int(a["active"][foe])

    def give(move_name: str, who: int = side, mon: int = active) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][who, mon, k] = move
        a["pp"][who, mon, k] = int(tables.move_pp[move])

    move = rnd.choice(
        (
            "Volt Tackle",
            "Flare Blitz",
            "Jump Kick",
            "Hi Jump Kick",
            "Endeavor",
            "Pain Split",
            "Knock Off",
            "Safeguard",
            "Focus Punch",
        )
    )
    give(move)
    if rnd.random() < 0.15:  # B10-7: a Safeguard up on either side, against the status moves
        a["safeguard"][rnd.choice((side, foe))] = rnd.randint(1, 5)
        give(rnd.choice(("Thunder Wave", "Toxic", "Supersonic", "Swagger", "Ember")), foe, foe_mon)
    if move == "Knock Off":  # B10-6: an item to take, a Quick Claw, a barb; a knocked-off user
        a["item"][foe, foe_mon] = names.item(
            rnd.choice(("Leftovers", "Quick Claw", "Sticky Barb", "Oran Berry", "Life Orb"))
        )
        if rnd.random() < 0.3:
            a["knocked_off"][side, active] = 1
            a["item"][side, active] = 0
    if move in ("Endeavor", "Pain Split") and rnd.random() < 0.6:  # a low user: Endeavor works
        a["hp"][side, active] = rnd.randint(1, max(1, int(a["max_hp"][side, active]) // 3))
    if rnd.random() < 0.3:
        a["ability"][side, active] = rnd.choice(
            (AbilityId.RECKLESS, AbilityId.ROCK_HEAD, AbilityId.MAGIC_GUARD)
        )
    if move == "Flare Blitz" and rnd.random() < 0.4:
        a["status"][side, active] = Status.FREEZE
        a["status_counter"][side, active] = 0
    if move in ("Jump Kick", "Hi Jump Kick"):
        roll = rnd.random()
        if roll < 0.3:
            give(rnd.choice(("Protect", "Detect")), foe, foe_mon)
        elif roll < 0.5:
            a["types"][foe, foe_mon] = (Type.GHOST, Type.GHOST)
        if rnd.random() < 0.3:  # a small foe: the crash's cap
            a["max_hp"][foe, foe_mon] = rnd.randint(10, 40)
            a["hp"][foe, foe_mon] = min(int(a["hp"][foe, foe_mon]), int(a["max_hp"][foe, foe_mon]))
    if rnd.random() < 0.3:  # a low user: the recoil or crash can faint it
        a["hp"][side, active] = max(1, int(a["max_hp"][side, active]) // 8)


# Moves whose decomp path tests a substitute or meets one (B10-4).
SUBSTITUTE_FOE_MOVES = (
    "Tackle",
    "Double Kick",
    "Thunder Wave",
    "Growl",
    "Toxic",
    "Supersonic",
    "Swagger",
    "Leech Seed",
    "Mega Drain",
    "Dream Eater",
    "SmellingSalt",
    "Wrap",
    "Fire Fang",
    "Headbutt",
    "Fake Out",
    "Thunderbolt",
    "Pursuit",
    "Future Sight",
    "Pluck",
    "False Swipe",
    "Super Fang",
    "Take Down",
)


def _draw_substitute(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 3) give the active Pokemon Substitute, or a substitute already up (a small one
    breaks, a big one holds), and its foe a move that meets it (``SUBSTITUTE_FOE_MOVES``); a
    sleeping or paralyzed holder for Dream Eater / SmellingSalt, Static for the contact gate."""
    if draw(st.integers(0, 2)) != 0:
        return
    rnd = draw(st.randoms(use_true_random=False))
    foe = 1 - side
    foe_mon = int(a["active"][foe])

    def give(move_name: str, who: int, mon: int) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][who, mon, k] = move
        a["pp"][who, mon, k] = int(tables.move_pp[move])

    if rnd.random() < 0.5:
        give("Substitute", side, active)
        if rnd.random() < 0.3:  # at or near the quarter: the failure
            quarter = max(int(a["max_hp"][side, active]) // 4, 1)
            a["hp"][side, active] = rnd.randint(1, quarter + 1)
    if rnd.random() < 0.7:
        a["substitute_hp"][side] = rnd.choice((1, rnd.randint(1, 60), 200))
    give(rnd.choice(SUBSTITUTE_FOE_MOVES), foe, foe_mon)
    if rnd.random() < 0.3:
        a["status"][side, active] = rnd.choice((Status.SLEEP, Status.PARALYSIS))
        a["status_counter"][side, active] = 2 if a["status"][side, active] == Status.SLEEP else 0
    if rnd.random() < 0.2:
        a["ability"][side, active] = rnd.choice((AbilityId.STATIC, AbilityId.INTIMIDATE))


def _draw_batch4(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 2) give the active Pokemon a ``BATCH4_MOVES`` move, with what makes it act
    most of the time: a paralyzed foe for SmellingSalt, a status for Facade, a foe at low HP for
    Brine and False Swipe, a confused foe at +6 Attack for Swagger's failure, +6 stages for
    Dragon Dance's, No Guard or a lower foe for the OHKO moves, Sturdy on the foe. Maybe (1 in
    3) a ``BATCH4_ABILITIES`` ability on it, Magic Guard with a residual to skip (a status,
    Leech Seed, Bind, Life Orb, a recoil move, Rough Skin on the foe) or on a bench Pokemon
    facing Stealth Rock; maybe (1 in 4) a crit item (Lucky Punch on a Chansey)."""
    rnd = draw(st.randoms(use_true_random=False))
    foe = 1 - side
    foe_mon = int(a["active"][foe])

    def give(who: int, mon: int, move_name: str) -> None:
        move = names.move(move_name)
        k = rnd.randint(0, 3)
        a["moves"][who, mon, k] = move
        a["pp"][who, mon, k] = int(tables.move_pp[move])

    foe_alive = a["hp"][foe, foe_mon] > 0
    if draw(st.booleans()):
        name = rnd.choice(BATCH4_MOVES)
        give(side, active, name)
        enable = rnd.random() < 0.6
        if enable and name == "SmellingSalt" and foe_alive:
            a["status"][foe, foe_mon] = Status.PARALYSIS
        if enable and name == "Facade" and a["hp"][side, active] > 0:
            a["status"][side, active] = rnd.choice((Status.BURN, Status.POISON, Status.PARALYSIS))
            a["status_counter"][side, active] = 0
        if enable and name in ("Brine", "False Swipe", "Super Fang") and foe_alive:
            a["hp"][foe, foe_mon] = rnd.randint(1, max(1, int(a["max_hp"][foe, foe_mon]) // 2))
        if enable and name == "Swagger":
            a["stages"][foe, Stat.ATK] = STAGE_MAX if rnd.random() < 0.5 else 4
            a["confusion"][foe] = rnd.randint(0, 2)
        if enable and name == "Snore" and a["hp"][side, active] > 0:
            a["status"][side, active] = Status.SLEEP
            a["status_counter"][side, active] = rnd.randint(1, 4)
        if enable and name == "Dragon Dance":
            a["stages"][side, Stat.ATK] = STAGE_MAX
            a["stages"][side, Stat.SPE] = rnd.choice((STAGE_MAX, 5))
        if enable and name in ("Horn Drill", "Guillotine", "Fissure"):
            choice = rnd.randint(0, 2)
            if choice == 0:
                a["ability"][side, active] = AbilityId.NO_GUARD
            elif choice == 1:
                a["level"][foe, foe_mon] = int(a["level"][side, active]) + 1
            else:
                a["ability"][foe, foe_mon] = AbilityId.STURDY
    if draw(st.integers(0, 2)) == 0:
        ability = rnd.choice(BATCH4_ABILITIES)
        a["ability"][side, active] = ability
        if ability == AbilityId.MAGIC_GUARD and rnd.random() < 0.75:
            kind = rnd.randint(0, 5)
            if kind == 0 and a["hp"][side, active] > 0:
                a["status"][side, active] = rnd.choice(
                    (Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)
                )
                a["status_counter"][side, active] = 0
            elif kind == 1:
                a["leech_seed"][side] = 1
            elif kind == 2:
                a["trapped"][side] = rnd.randint(2, _BIND_MAX)
                a["trap_move"][side] = names.move("Bind")
            elif kind == 3:
                a["item"][side, active] = names.item("Life Orb")
            elif kind == 4:
                give(side, active, "Take Down")
            else:
                a["ability"][foe, foe_mon] = AbilityId.ROUGH_SKIN
                give(side, active, "Tackle")
    bench = [
        slot
        for slot in range(a["species"].shape[1])
        if slot != active and a["species"][side, slot] and a["hp"][side, slot]
    ]
    # The side conditions and Destiny Bond (any layers, a bond from an earlier turn), and a
    # Flying-type foe for a Roost user's chart.
    a["spikes"][side] = draw(st.integers(0, 3))
    a["toxic_spikes"][side] = draw(st.integers(0, 2))
    a["destiny_bond"][side] = draw(st.integers(0, 1))
    if draw(st.integers(0, 4)) == 0:
        a["types"][side, active, 1] = Type.FLYING
    if bench and draw(st.integers(0, 3)) == 0:  # a Magic Guard newcomer facing Stealth Rock
        a["ability"][side, rnd.choice(bench)] = AbilityId.MAGIC_GUARD
        a["stealth_rock"][side] = 1
    if draw(st.integers(0, 3)) == 0:
        chansey = int(a["species"][side, active]) == names.species("Chansey")
        item = "Lucky Punch" if chansey and rnd.random() < 0.7 else rnd.choice(CRIT_ITEMS)
        a["item"][side, active] = names.item(item)


def _draw_batch5(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe (1 in 3) give the active Pokemon an invoker (``INVOKERS``) and another move it can
    call (``CALLED_MOVES``), a Sleep Talk user mostly asleep (1-4 turns left, so it can wake);
    the battle's ``move_prev``: none, a move of either active, Struggle, an invoker or a move
    Copycat cannot copy (Protect)."""
    rnd = draw(st.randoms(use_true_random=False))
    if draw(st.integers(0, 2)) == 0:
        name = rnd.choice(INVOKERS)
        for slot, move_name in zip(rnd.sample(range(4), 2), (name, rnd.choice(CALLED_MOVES))):
            move = names.move(move_name)
            a["moves"][side, active, slot] = move
            a["pp"][side, active, slot] = int(tables.move_pp[move])
        if name == "Sleep Talk" and a["hp"][side, active] > 0 and rnd.random() < 0.8:
            a["status"][side, active] = Status.SLEEP
            a["status_counter"][side, active] = rnd.randint(1, 4)
    pool = [int(m) for sd in (0, 1) for m in a["moves"][sd, a["active"][sd]] if m]
    extra = [MOVE_STRUGGLE, names.move("Sleep Talk"), names.move("Protect"), names.move("Uproar")]
    a["move_prev"][...] = draw(st.sampled_from([0, *pool, *extra]))


def _draw_uproar(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """E2 batch 5: maybe (1 in 3) give the active Pokemon Uproar (full PP), and then maybe (1 in
    2) an uproar under way: 1-5 turns left (subscript_uproar.s's 3-6, counted down at every turn
    end), locked into Uproar (``locked_move``; a quarter of the time it does not know the move:
    Copycat's call), the Bide / Rollout lock dropped (an uproarer runs only Uproar), and the
    actives awake (no battler on the field sleeps at a turn start during an uproar: the turn end's
    check woke them, B5-7). Else maybe (1 in 4) the foe asleep (1-4 turns: Uproar wakes it); maybe
    a Ghost-type foe (Uproar is ineffective: noEffect ends the uproar)."""
    rnd = draw(st.randoms(use_true_random=False))
    if draw(st.integers(0, 2)) != 0:
        return
    uproar = names.move("Uproar")
    k = rnd.randint(0, 3)
    a["moves"][side, active, k] = uproar
    a["pp"][side, active, k] = int(tables.move_pp[uproar])
    foe = 1 - side
    foe_mon = int(a["active"][foe])
    under_way = draw(st.booleans())
    if under_way:
        if rnd.random() < 0.25:
            a["moves"][side, active, k] = names.move("Copycat")
        a["uproar"][side] = rnd.randint(1, 5)
        a["locked_move"][side] = uproar
        a["bide_turns"][side] = a["bide_damage"][side] = a["rollout_count"][side] = 0
        for sd in (0, 1):
            mon = int(a["active"][sd])
            if a["status"][sd, mon] == Status.SLEEP:
                a["status"][sd, mon] = Status.NONE
                a["status_counter"][sd, mon] = 0
    if not under_way and a["hp"][foe, foe_mon] > 0 and draw(st.integers(0, 3)) == 0:
        a["status"][foe, foe_mon] = Status.SLEEP
        a["status_counter"][foe, foe_mon] = rnd.randint(1, 4)
    elif draw(st.integers(0, 3)) == 0:
        a["types"][foe, foe_mon, 0] = Type.GHOST


def _draw_future_sight(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """E2 batch 5: maybe (1 in 3) give the active Pokemon Future Sight or Doom Desire (full PP);
    maybe (1 in 3) one pending on this side's battler from the foe's: 1 or 2 turns left (it hits
    at the end of a turn that starts at 1), the move, the attacker (1 + battler) and a damage up
    to twice the max HP, the side's bit up; else maybe (1 in 8) the bit alone (a hit whose
    battler had fainted keeps it). With a hit due, maybe (1 in 3) a full-HP Focus Sash on the
    battler, or No Guard on it or on the foe."""
    rnd = draw(st.randoms(use_true_random=False))
    moves = (names.move("Future Sight"), names.move("Doom Desire"))
    if draw(st.integers(0, 2)) == 0:
        move = rnd.choice(moves)
        k = rnd.randint(0, 3)
        a["moves"][side, active, k] = move
        a["pp"][side, active, k] = int(tables.move_pp[move])
    if draw(st.integers(0, 2)) == 0:
        timer = rnd.choice((1, 1, 2))
        a["future_timer"][side] = timer
        a["future_move"][side] = rnd.choice(moves)
        a["future_attacker"][side] = 1 + (1 - side)
        a["future_damage"][side] = rnd.randint(1, 2 * int(a["max_hp"][side, active]))
        a["future_sight"][side] = 1
        if timer == 1 and draw(st.integers(0, 2)) == 0:
            pick = rnd.randint(0, 2)
            if pick == 0:
                a["item"][side, active] = names.item("Focus Sash")
                a["hp"][side, active] = a["max_hp"][side, active]
            else:
                a["ability"][
                    side if pick == 1 else 1 - side,
                    int(a["active"][side if pick == 1 else 1 - side]),
                ] = AbilityId.NO_GUARD
    elif draw(st.integers(0, 7)) == 0:
        a["future_sight"][side] = 1


BATCH7_MOVES = (
    "Ingrain",
    "Mean Look",
    "Block",
    "Spider Web",
    "Imprison",
    "Roar",
    "Roost",
    # E2 batch 8
    "Lock-On",
    "Mind Reader",
    "Sheer Cold",
    "Perish Song",
    "Yawn",
    "Curse",
)


def _draw_batch7(draw, tables: Tables, a: dict, side: int, active: int, names: Names) -> None:
    """E2 batch 7: maybe (1 in 3) give the active Pokemon one of ``BATCH7_MOVES`` (full PP), an
    Imprison user mostly sharing a move with the foe (one of the foe's moves copied into another
    slot, so ``BtlCmd_TryImprison`` can succeed); maybe the side's battler ingrained (1 in 4),
    trapped by the foe's Mean Look (1 in 4, ``mean_look_target`` the foe's battler), imprisoning
    (1 in 6) or imprisoned (1 in 6). The other draws give Ingrain beside other mask bits (Aqua
    Ring, the sports, Leech Seed) or alone, which Roar's anchoring tells apart."""
    rnd = draw(st.randoms(use_true_random=False))
    foe = 1 - side
    if draw(st.integers(0, 2)) == 0:
        name = rnd.choice(BATCH7_MOVES)
        free = [k for k in range(4) if a["moves"][side, active, k] != a["locked_move"][side]]
        if free:
            k = rnd.choice(free)
            move = names.move(name)
            a["moves"][side, active, k] = move
            a["pp"][side, active, k] = int(tables.move_pp[move])
            foe_moves = [int(m) for m in a["moves"][foe, a["active"][foe]] if m]
            others = [j for j in free if j != k]
            if name == "Curse" and rnd.random() < 0.5:  # a Ghost user (B8-5)
                a["types"][side, active] = [Type.GHOST, Type.GHOST]
            if name == "Imprison" and foe_moves and others and rnd.random() < 0.8:
                shared = rnd.choice(foe_moves)
                j = rnd.choice(others)
                a["moves"][side, active, j] = shared
                a["pp"][side, active, j] = int(tables.move_pp[shared])
    a["ingrain"][side] = int(draw(st.integers(0, 3)) == 0)
    if draw(st.integers(0, 3)) == 0:
        a["mean_look"][side] = 1
        a["mean_look_target"][side] = 1 + foe
    elif draw(st.integers(0, 3)) == 0:  # a stale meanLookTarget (B7-15)
        a["mean_look_target"][side] = 1 + foe
    a["imprison"][side] = int(draw(st.integers(0, 5)) == 0)
    # Batch 8: locked onto by the foe (count 1-2) or a stale lockOnTarget with no count.
    lock = draw(st.integers(0, 5))
    if lock <= 1:
        a["lock_on"][side] = lock + 1
        a["lock_on_target"][side] = 1 + foe
    elif lock == 2:
        a["lock_on_target"][side] = 1 + foe
    a["curse"][side] = int(draw(st.integers(0, 5)) == 0)
    # A Yawn counting (1: it falls asleep at this turn's end).
    a["yawn"][side] = draw(st.sampled_from([0, 0, 0, 1, 2]))
    # Perish Song under way: a count 0-3 (0 faints at this turn's end).
    if draw(st.integers(0, 4)) == 0:
        a["perish_song"][side] = 1
        a["perish_song_turns"][side] = draw(st.integers(0, 3))
    a["imprisoned"][side] = int(draw(st.integers(0, 5)) == 0)


def _share_with_sleep_talk(draw, tables: Tables, a: dict, names: Names) -> None:
    """Maybe (1 in 2) copy a move the other active's Sleep Talk can call into a free slot of an
    active that knows Imprison (``_draw_batch7``, B7-13)."""
    imprison, talk = names.move("Imprison"), names.move("Sleep Talk")
    for side in (0, 1):
        foe = 1 - side
        own = a["moves"][side, a["active"][side]]
        theirs = a["moves"][foe, a["active"][foe]]
        if imprison not in own or talk not in theirs or not draw(st.booleans()):
            continue
        callable_ = [int(m) for m in theirs if m and m != talk and m not in own]
        free = [k for k in range(4) if own[k] not in (imprison, a["locked_move"][side])]
        if callable_ and free:
            k = draw(st.sampled_from(free))
            move = draw(st.sampled_from(callable_))
            a["moves"][side, a["active"][side], k] = move
            a["pp"][side, a["active"][side], k] = int(tables.move_pp[move])


def _draw_choice_lock(draw, a: dict, side: int, active: int) -> None:
    """``choice_lock``: none or one of the active Pokemon's moves, whatever its item (a lock
    without a Choice item is what a lost item would leave; ``UpdateMoveBuffers`` clears it)."""
    moves = [int(m) for m in a["moves"][side, active] if m]
    a["choice_lock"][side] = draw(st.sampled_from([0, 0, *moves]))


def _draw_lock(draw, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe lock the active Pokemon into a Bide or Rollout / Ice Ball it knows (not while
    it sleeps: ``step`` unlocks a sleeper), with the matching hidden counters: Bide's
    turns left (1 or 2: the counter starts at 2 and the unleash turn finds 1; a locked Bide
    never rests at 0) and stored damage, Rollout's count (1-4)."""
    if a["status"][side, active] == Status.SLEEP or not draw(st.booleans()):
        return
    bide, rollout, ice_ball = (names.move(m) for m in ("Bide", "Rollout", "Ice Ball"))
    known = [int(m) for m in a["moves"][side, active] if int(m) in (bide, rollout, ice_ball)]
    if not known:
        return
    move = draw(st.sampled_from(known))
    a["locked_move"][side] = move
    if move == bide:
        a["bide_turns"][side] = draw(st.integers(1, 2))
        a["bide_damage"][side] = draw(st.integers(0, 2 * int(a["max_hp"][side, active])))
    else:
        a["rollout_count"][side] = draw(st.integers(1, 4))


@st.composite
def supported_state(draw, tables: Tables, names: Names, invokers: bool = True) -> BattleState:
    """A mid-battle state: 1-6 Pokemon per side (``supported_pokemon``), any active
    slot, random PP, status counters, stat stages, confusion, Taunt, Bind, Leech Seed, Disable (a move of the active and its turns), Defense Curl, a Bide / Rollout lock (``_draw_lock``),
    Pursuit in a slot (``_draw_pursuit``), a Choice item, Life Orb or other battle item
    (``_draw_battle_item``), a species item's holder (``_draw_species_item``), a batch-2 item
    with what makes it act (``_draw_batch2_item``), a Choice lock
    (``_draw_choice_lock``),
    Water Sport, Foresight, Stealth Rock, screens, Tailwind, Trick Room, Mud Sport, weather,
    the turn, each active's ``fake_out_turn`` (0 to ``turn + 1``), ``move_hit`` (none or
    one of the foe's active's moves) and ``last_move`` (none, one of its own moves or
    Struggle), and the opponent's bag (a size of 0-4 in
    ``trainer_item_count``, ``M1_BAG_ITEMS`` in each slot below it, gaps allowed as
    after a use) and the AI's knowledge of the player (any ``ai_seen_moves`` and
    ``ai_ability_known`` flags, ``ai_item_known`` none or the held item), the death record (``deaths`` = the fainted player slots), a
    faint budget 0-3 and a random set of living player slots as ``protected``, each within
    its legal range, and the player's experience block (``_draw_exp``); the battle is not
    over."""
    parties = [draw(_party(tables, names)) for _ in (0, 1)]
    for side in (0, 1):  # coverage refuses a Selfdestruct facing a Defense below 8
        foe_defense = low_defense(tables, parties[1 - side][0])
        moves = [m for mon in parties[side][0] for m in mon.moves if m]
        assume(not any(halved_defense_line(tables, m, foe_defense) for m in moves))
    s = make_state(tables, parties[0][0], parties[1][0])
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    trap_moves = np.flatnonzero(np.asarray(tables.move_effect) == EffectId.TRAP)
    move_pp = np.asarray(tables.move_pp)
    turn = draw(st.integers(0, 50))
    a["turn"][...] = turn  # read by _draw_batch3 (Truant)
    for side, (party, active) in enumerate(parties):
        a["active"][side] = active
        for slot in range(len(party)):
            for k, move in enumerate(a["moves"][side, slot]):
                if move:
                    a["pp"][side, slot, k] = draw(st.integers(0, int(move_pp[move])))
            a["status_counter"][side, slot] = _status_counter(draw, int(a["status"][side, slot]))
        a["stages"][side, Stat.ATK :] = draw(
            st.lists(st.integers(STAGE_MIN, STAGE_MAX), min_size=7, max_size=7)
        )
        a["confusion"][side] = draw(st.integers(0, _CONFUSION_MAX))
        a["taunt"][side] = draw(st.integers(0, _TAUNT_MAX))
        trapped = draw(st.integers(0, _BIND_MAX))
        a["trapped"][side] = trapped
        a["trap_move"][side] = draw(st.sampled_from(trap_moves.tolist())) if trapped else 0
        a["leech_seed"][side] = draw(st.integers(0, 1))
        a["water_sport"][side] = draw(st.integers(0, 1))
        a["mud_sport"][side] = draw(st.integers(0, 1))
        a["aqua_ring"][side] = draw(st.integers(0, 1))
        a["infatuated"][side] = draw(st.integers(0, 1))
        a["qc_roll"][side] = draw(st.integers(0, 1))
        a["curled"][side] = draw(st.integers(0, 1))
        a["foresight"][side] = draw(st.integers(0, 1))
        a["stealth_rock"][side] = draw(st.integers(0, 1))
        a["reflect"][side] = draw(st.integers(0, _TIMER_MAX))
        a["light_screen"][side] = draw(st.integers(0, _TIMER_MAX))
        a["tailwind"][side] = draw(st.integers(0, _TIMER_MAX))
        a["fake_out_turn"][side] = draw(st.integers(0, turn + 1))
        foe_moves = [int(m) for m in a["moves"][1 - side, parties[1 - side][1]] if m]
        a["move_hit"][side] = draw(st.sampled_from([0, *foe_moves]))
        own_moves = [int(m) for m in a["moves"][side, active] if m]
        a["last_move"][side] = draw(st.sampled_from([0, *own_moves, MOVE_STRUGGLE]))
        disabled = draw(st.sampled_from([0, *own_moves]))
        a["disabled_move"][side] = disabled
        turns = draw(st.integers(0, _DISABLE_MAX))
        a["disable_turns"][side] = turns if disabled else 0
        _draw_pursuit(draw, tables, a, side, active, names)
        _draw_lock(draw, a, side, active, names)
        _draw_battle_item(draw, tables, a, side, active, names)
        _draw_species_item(draw, tables, a, side, active, names)
        _draw_batch2_item(draw, tables, a, side, active, names)
        _draw_gardenia_item(draw, tables, a, side, active, names)
        _draw_batch3(draw, tables, a, side, active, names)
        _draw_batch4(draw, tables, a, side, active, names)
        if invokers:
            _draw_batch5(draw, tables, a, side, active, names)
        _draw_uproar(draw, tables, a, side, active, names)
        _draw_future_sight(draw, tables, a, side, active, names)
        _draw_pivot(draw, tables, a, side, active, names)
        _draw_batch7(draw, tables, a, side, active, names)
        _draw_batch9(draw, tables, a, side, active, names)
        _draw_batch10(draw, tables, a, side, active, names)
        _draw_substitute(draw, tables, a, side, active, names)
        _draw_pressure(draw, tables, a, side, active, names)
        _draw_choice_lock(draw, a, side, active)
    # B7-13: an Imprison user sharing a move with a Sleep Talk user, so a faster Imprison (or one
    # already up) can shrink the Sleep Talk set.
    _share_with_sleep_talk(draw, tables, a, names)
    # No battler on the field sleeps at a turn start during an uproar (B5-7: the turn end woke
    # it): the other side's draws above may have put one to sleep after the uproar was drawn.
    if np.any(a["uproar"] > 0):
        for sd in (0, 1):
            mon = int(a["active"][sd])
            if a["status"][sd, mon] == Status.SLEEP:
                a["status"][sd, mon] = Status.NONE
                a["status_counter"][sd, mon] = 0
    count = draw(st.integers(0, BAG_SLOTS))
    bag = draw(st.lists(st.sampled_from(M1_BAG_ITEMS), min_size=count, max_size=count))
    a["trainer_items"][1] = [names.item(i) for i in bag] + [0] * (BAG_SLOTS - count)
    a["trainer_item_count"][1] = count
    weather = a["weather"][()] if a["weather"][()] else draw(st.sampled_from(list(Weather)))
    a["weather"][...] = weather
    # A temporary weather (a weather move's: 1-8 turns left) or a permanent one (0).
    timed = weather not in (Weather.NONE, Weather.FOG)
    a["weather_timer"][...] = draw(st.integers(0, 8)) if timed else 0
    _draw_terrain(draw, tables, a)
    a["trick_room"][...] = draw(st.integers(0, _TIMER_MAX))
    a["turn"][...] = turn
    for name in ("ai_seen_moves", "ai_ability_known"):
        flags = draw(st.lists(st.integers(0, 1), min_size=a[name].size, max_size=a[name].size))
        a[name][...] = np.reshape(flags, a[name].shape)
    # The AI's record of each player item (batch 7): none, or the item a message named, which
    # the Pokemon still holds (every removal clears the record, effects.forget_item).
    known = draw(st.lists(st.booleans(), min_size=SLOTS, max_size=SLOTS))
    a["ai_item_known"][...] = np.where(known, a["item"][0], 0)
    fainted = (a["species"][0] > 0) & (a["hp"][0] == 0)
    a["player_faints"][...] = int(np.sum(fainted))
    a["deaths"][...] = fainted
    a["faint_budget"][...] = draw(st.integers(0, MAX_DEATH_BUDGET))
    living = (a["species"][0] > 0) & (a["hp"][0] > 0)
    a["protected"][...] = [int(draw(st.booleans())) if alive else 0 for alive in living]
    a["replace_mul"][...] = 0
    _draw_exp(draw, tables, a)
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})


def _draw_exp(draw, tables: Tables, a: dict) -> None:
    """The player's experience block (K3d-exp): each Pokemon's ``exp`` anywhere in its level
    (``[base(L), base(L + 1))``), its ``exp_boost`` (100 / 150 / 170), ``pokerus`` and
    ``friend_bonus``; ``exp_mask`` is the active slot plus any other living slots (the
    participants since the foe's entry); the learn fields and ``exp_refused`` stay 0."""
    rates, table = np.asarray(tables.species_exp_rate), np.asarray(tables.exp_table)
    living = (a["species"][0] > 0) & (a["hp"][0] > 0)
    mask = 1 << int(a["active"][0])
    for slot in np.flatnonzero(a["species"][0]):
        row = table[rates[a["species"][0, slot]]]
        level = int(a["level"][0, slot])
        a["exp"][slot] = draw(st.integers(int(row[level]), int(row[level + 1]) - 1))
        a["exp_boost"][slot] = draw(st.sampled_from((100, 150, 170)))
        a["pokerus"][slot] = draw(st.integers(0, 1))
        a["friend_bonus"][slot] = draw(st.integers(0, 1))
        if living[slot] and draw(st.booleans()):
            mask |= 1 << int(slot)
    a["exp_mask"][...] = mask


@st.composite
def legal_action_pair(draw, tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[2]``: a legal action for each side: the player's from
    ``engine.actions.legal_actions``, the opponent's (moves, switches and bag items) from
    ``legal_opponent_actions``. One draw in four, when a side can use Pursuit and the other
    can switch, picks that pair: the Pursuit hit on a switching Pokemon (``step._pursuit_hit``)
    is rare under uniform draws. Otherwise a side whose batch-2 item acts on its own move
    (``HITTER_ITEMS``, Light Clay) or faces a Sticky Barb uses such a move one draw in two."""
    masks = [
        np.asarray(legal_actions(tables, state, 0)),
        np.asarray(legal_opponent_actions(tables, state)),
    ]
    legal = [np.flatnonzero(m).tolist() for m in masks]
    moves = np.asarray(state.moves)[np.arange(2), np.asarray(state.active)]
    pursuit = np.asarray(tables.move_effect)[moves] == EffectId.PURSUIT
    pairs = [
        (k, j, side)
        for side in (0, 1)
        for k in range(4)
        if pursuit[side, k] and masks[side][k]
        for j in range(4, 10)
        if masks[1 - side][j]
    ]
    # E2 batch 6: U-turn / Baton Pass, and U-turn against a Pursuit.
    pivots = np.isin(np.asarray(tables.move_effect)[moves], (EffectId.U_TURN, EffectId.BATON_PASS))
    chase = [
        (k, j, side)
        for side in (0, 1)
        for k in range(4)
        if pivots[side, k] and masks[side][k]
        for j in range(4)
        if pursuit[1 - side, j] and masks[1 - side][j]
    ]
    if pairs and draw(st.integers(0, 3)) == 0:
        k, j, side = draw(st.sampled_from(pairs))
        acts = [k, j] if side == 0 else [j, k]
    elif chase and draw(st.integers(0, 3)) == 0:
        k, j, side = draw(st.sampled_from(chase))
        acts = [k, j] if side == 0 else [j, k]
    else:
        acts = [draw(st.sampled_from(options)) for options in legal]
        # E2 batch 2: an item that acts on its holder's own move (King's Rock / Razor Fang,
        # Shell Bell, the lenses: a damaging move; Light Clay: Reflect or Light Screen), or a
        # Sticky Barb on the foe (a damaging contact move), is used one draw in two.
        effect = np.asarray(tables.move_effect)[moves]
        damaging = (np.asarray(tables.move_category)[moves] != STATUS_CATEGORY) & (
            np.asarray(tables.move_power)[moves] > 0
        )
        contact = (np.asarray(tables.move_flags)[moves] & MoveFlag.CONTACT) != 0
        screens = np.isin(effect, (EffectId.REFLECT, EffectId.LIGHT_SCREEN))
        held = np.asarray(tables.item_effect)[
            np.asarray(state.item)[np.arange(2), np.asarray(state.active)]
        ]
        weather_moves = effect == EffectId.WEATHER
        # E2 batch 3: a move the foe's ability acts on (Dry Skin: Water or Fire; Thick Fat:
        # Fire or Ice; Levitate: Ground; Shield Dust: a secondary effect), or one the side's own
        # acts on (Serene Grace: a secondary effect; Reckless: a recoil move; Technician: power
        # 60 or less), is used one draw in two, like the items' moves below.
        kinds = np.asarray(tables.move_type)[moves]
        chance = np.asarray(tables.move_effect_chance)[moves]
        abilities = np.asarray(state.ability)[np.arange(2), np.asarray(state.active)]
        aimed = {
            AbilityId.DRY_SKIN: (Type.WATER, Type.FIRE),
            AbilityId.THICK_FAT: (Type.FIRE, Type.ICE),
            AbilityId.LEVITATE: (Type.GROUND,),
        }
        power = np.asarray(tables.move_power)[moves]
        for side in (0, 1):
            foe_ability, own = int(abilities[1 - side]), int(abilities[side])
            if foe_ability in aimed:
                wanted = damaging[side] & np.isin(kinds[side], aimed[foe_ability])
            elif foe_ability == AbilityId.SHIELD_DUST or own == AbilityId.SERENE_GRACE:
                wanted = damaging[side] & (chance[side] > 0)
            elif own == AbilityId.RECKLESS:
                wanted = effect[side] == EffectId.RECOIL
            elif own == AbilityId.TECHNICIAN:
                wanted = damaging[side] & (power[side] <= 60)
            else:
                continue
            slots = [k for k in range(4) if wanted[k] and masks[side][k]]
            if slots and draw(st.booleans()):
                acts[side] = draw(st.sampled_from(slots))
        for side in (0, 1):
            if held[side] == ItemId.WEATHER_ROCK:
                wanted = weather_moves[side]
            elif held[side] == ItemId.LIGHT_CLAY:
                wanted = screens[side]
            elif held[side] in HITTER_ITEMS:
                wanted = damaging[side]
            elif held[1 - side] == ItemId.STICKY_BARB:
                wanted = damaging[side] & contact[side]
            else:
                continue
            slots = [k for k in range(4) if wanted[k] and masks[side][k]]
            if slots and draw(st.booleans()):
                acts[side] = draw(st.sampled_from(slots))
        for side in (0, 1):
            slots = [k for k in range(4) if pivots[side, k] and masks[side][k]]
            if slots and draw(st.integers(0, 3)) != 0:
                acts[side] = draw(st.sampled_from(slots))
    return jnp.asarray(acts, jnp.int32)


@st.composite
def pivot_actions(draw, state: BattleState, actions) -> jnp.ndarray:
    """E2 batch 6: the actions that resume ``state`` paused at a party list: the pivoting side's
    switch to any legal replacement (the engine takes any; the AI's pick is the search's), the
    other side's as before (``step`` reads the scratch's)."""
    side = int(state.pivot_battler) - 1
    slots = np.flatnonzero(np.asarray(legal_replacements(state, side))).tolist()
    acts = np.array(actions)
    acts[side] = MOVE_ACTIONS + draw(st.sampled_from(slots))
    return jnp.asarray(acts, jnp.int32)


def _ints(lo: int, hi: int, shape: tuple[int, ...]) -> st.SearchStrategy[jnp.ndarray]:
    size = int(np.prod(shape, dtype=int))
    return st.lists(st.integers(lo, hi), min_size=size, max_size=size).map(
        lambda v: jnp.asarray(np.asarray(v, np.int8).reshape(shape))
    )


def _flag(shape: tuple[int, ...]) -> st.SearchStrategy[jnp.ndarray]:
    return _ints(0, 1, shape)


def any_outcome() -> st.SearchStrategy[Outcome]:
    """Every ``Outcome`` field uniform over its range (``hit``: either flag-only or any
    class value)."""
    hits, sides = (2, MAX_HITS), (2,)
    return st.builds(
        Outcome,
        speed_tie=_flag(()),
        # 0-1 as a hit / miss flag, or any class value (outcome.hit_class: a roll 1-100).
        hit=st.one_of(_flag(hits), _ints(0, 100, hits)),
        crit=_flag(hits),
        roll=_ints(0, 15, hits),
        secondary=_flag(hits),
        contact_proc=_flag(hits),
        n_hits=_ints(2, 5, sides),
        full_para=_flag(sides),
        confusion_hit=_flag(sides),
        duration=_ints(*_DURATION, sides),
        thaw=_flag(sides),
        flinch_item=_flag(hits),
        quick_claw=_flag(sides),
        end_speed_tie=_flag(()),
        magnitude=_ints(4, 10, sides),
        shed_skin=_flag(sides),
        fang_flinch=_flag(sides),
        present=_ints(0, 3, sides),
        protect=_flag(sides),
        infatuation=_flag(sides),
        tri_attack=_ints(0, 2, sides),
        roar_level=_flag(sides),
        roar_target=_ints(0, 4, sides),
        # An invoker's draw: Sleep Talk's value, Metronome's candidate rank (taken modulo
        # their counts, order.invoked_move).
        call=st.lists(st.integers(0, 500), min_size=2, max_size=2).map(
            lambda v: jnp.asarray(np.asarray(v, np.int16))
        ),
        # The Future Sight hit's roll: a flag or any class value (1-100), like ``hit``.
        future_hit=st.one_of(_flag(sides), _ints(0, 100, sides)),
        yawn_sleep=_ints(2, 5, sides),
        focus_band=_flag(hits),
        future_band=_flag(sides),
        berry_confusion=_ints(2, 5, sides),
    )


# Moves whose outcome laws the enumeration tests need (tests/engine/test_enumerate.py):
# sleep (Spore, Hypnosis), Taunt, Bind (TRAP), multi-hit (Fury Swipes), confusion
# (Supersonic, Confuse Ray), secondary paralysis / freeze / burn with a thawing move
# (Spark, Ice Beam, Flame Wheel), flinch (Headbutt; Thunder Fang, paralysis then flinch), a
# Speed drop (Bubble), a Speed boost (Rock Polish), a high-crit move (Razor Leaf), a 95 %
# plain move (Tackle), accuracy / evasion changes before the foe's check (Sand-Attack,
# Double Team, Mud-Slap), Present (drawn power or heal), Protect (its chain's 2^-n roll,
# with a chain of 0-3 drawn for it), Attract (the foe's attract check), Tri Attack (its
# status draw), Nature Power (its call on the drawn terrain), Camouflage (``_draw_terrain``) and
# Roar (its level check and pick).
LAW_MOVES = (
    "Spore",
    "Hypnosis",
    "Taunt",
    "Bind",
    "Fury Swipes",
    "Supersonic",
    "Confuse Ray",
    "Spark",
    "Ice Beam",
    "Flame Wheel",
    "Headbutt",
    "Thunder Wave",
    "Bubble",
    "Rock Polish",
    "Razor Leaf",
    "Tackle",
    "Magnitude",
    "Confusion",
    "Disable",
    "Sand-Attack",
    "Double Team",
    "Mud-Slap",
    "Thunder Fang",
    "Present",
    "Protect",
    "Attract",
    "Tri Attack",
    "Nature Power",
    "Camouflage",
    "Roar",
    "Rain Dance",
    "Sunny Day",
    # E2 batch 4: Swagger's confusion draw, Dragon Dance's Speed, the OHKO hit test, Super Fang
    # (no crit, no roll), Poison Fang's secondary.
    "Swagger",
    "Dragon Dance",
    "Horn Drill",
    "Super Fang",
    "Poison Fang",
    "Tailwind",
    "Roost",
    "Snore",
    "Secret Power",
    "Spikes",
    "Counter",
    # E2 batch 5: Uproar's duration (Random 3, 3); Future Sight's stored roll.
    "Uproar",
    "Future Sight",
    "Doom Desire",
)
# Abilities the outcome relevance reads (contact procs, Steadfast, Simple's and Swift
# Swim's Speed) or the hit odds read (Compoundeyes, Sand Veil, Snow Cloak), and Rough Skin
# and Limber, which sit next to the contact procs and the paralysis they relate to.
LAW_ABILITIES = (
    AbilityId.STATIC,
    AbilityId.POISON_POINT,
    AbilityId.FLAME_BODY,
    AbilityId.CUTE_CHARM,
    AbilityId.ROUGH_SKIN,
    AbilityId.LIMBER,
    AbilityId.STEADFAST,
    AbilityId.SIMPLE,
    AbilityId.SHED_SKIN,
    AbilityId.COMPOUNDEYES,
    AbilityId.SAND_VEIL,
    AbilityId.SNOW_CLOAK,
    AbilityId.SWIFT_SWIM,
    # E2 batch 3: the chance laws (Serene Grace, Super Luck, the armors), the hit odds (No
    # Guard, Tangled Feet), the end speeds (Quick Feet), the flinch and secondary blocks (Shield
    # Dust, Dry Skin's absorb), Truant's lost turn and Trace.
    AbilityId.SERENE_GRACE,
    AbilityId.SUPER_LUCK,
    AbilityId.SHELL_ARMOR,
    AbilityId.BATTLE_ARMOR,
    AbilityId.NO_GUARD,
    AbilityId.TANGLED_FEET,
    AbilityId.QUICK_FEET,
    AbilityId.SHIELD_DUST,
    AbilityId.DRY_SKIN,
    AbilityId.TRUANT,
    AbilityId.LEVITATE,
    AbilityId.TRACE,
    # E2 batch 4: full paralysis (Magic Guard), the OHKO block (Sturdy).
    AbilityId.MAGIC_GUARD,
    AbilityId.STURDY,
)


@st.composite
def outcome_law_state(draw, tables: Tables, names: Names) -> BattleState:
    """``supported_state`` widened for the outcome-law tests: Lucky Chant on either
    side (0-5 turns), and on each side, independently, maybe one of ``LAW_MOVES`` in
    the active Pokemon's slot 0 (full PP) and maybe one of ``LAW_ABILITIES`` on the
    active Pokemon. Only ``sample_outcome`` reads Lucky Chant."""
    # No invokers: these laws are one call variant's (enumerate.outcome_laws' ``call``); the
    # invokers' variants have their own tests (tests/engine/test_invokers.py).
    s = draw(supported_state(tables, names, invokers=False))
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    move_pp = np.asarray(tables.move_pp)
    effect = np.asarray(tables.move_effect)
    invoker = np.isin(
        effect[a["moves"]], (EffectId.SLEEP_TALK, EffectId.METRONOME, EffectId.COPYCAT)
    )
    a["moves"][invoker] = names.move("Tackle")  # a learnset's own invoker too
    for side in (0, 1):
        mon = int(a["active"][side])
        a["lucky_chant"][side] = draw(st.sampled_from([0, 0, 1, 5]))
        if draw(st.booleans()):
            move = names.move(draw(st.sampled_from(LAW_MOVES)))
            a["moves"][side, mon, 0] = move
            a["pp"][side, mon, 0] = move_pp[move]
            if int(tables.move_effect[move]) == EffectId.PROTECT:
                a["protect_count"][side] = draw(st.integers(0, 3))

        if draw(st.integers(0, 3)) == 0:
            a["ability"][side, mon] = draw(st.sampled_from(LAW_ABILITIES))
    _draw_terrain(draw, tables, a)  # a LAW_MOVES terrain move needs one too
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})
