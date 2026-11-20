"""Evolution conditions (design v5.4 section 6, G8; ledger E3-B3-12).

:func:`evolution_target` transcribes ``Pokemon_GetEvolutionTargetSpecies``
(``src/pokemon.c:3528-3790``): given a party member, the party and the evolution class it says
which species the member evolves into, and by which method. Evolution runs after the battle
(``src/battle/battle_system.c:826-850``) or after a Rare Candy
(``src/applications/party_menu/callbacks.c:816-829``); whether the answer is taken is the
caller's (GT-14), and what the evolution changes is GT-13's.

Inputs the function reads from its arguments, never from a Carry:

- the member's :class:`~nuzline.data.spec.PokemonSpec` (level, friendship, held item, moves,
  gender, stats). The caller passes the spec as the level-up left it: the level-up raised the
  friendship before the check (``bs.c:9857``; the Rare Candy path likewise);
- ``personality`` and ``beauty``, constants of the Pokemon from the save
  (``MON_DATA_PERSONALITY``, ``MON_DATA_BEAUTY``; ``src/pokemon.c:3535-3536``);
- ``party``: party 0's specs (``Party_HasSpecies``, ``src/party.c:128-139``);
- ``evo_param``: the map's evolution method for ``EVO_CLASS_BY_LEVEL`` (the battle passes
  ``dto->mapEvolutionMethod``, ``src/field_battle_data_transfer.c:264``; the Rare Candy path
  reads it for the current header, ``callbacks.c:820``; both from
  ``MapHeader_GetMapEvolutionMethod``, here ``Learnsets.map_evolution_method``), the stone's
  item id for ``EVO_CLASS_BY_ITEM``;
- ``time``: the run's ``time`` setting (``"day"``, ``"night"`` or ``None``). Day and night
  methods are offered only with it set (design section 6, "Time of day"): ``IsNight``
  (``src/rtc.c:134-140``) is read as ``time == "night"``, and a day row needs ``time == "day"``.
"""

from __future__ import annotations

from collections.abc import Sequence

from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.ids import Stat
from nuzline.engine.state import spec_stats

__all__ = [
    "EVOLVE_FRIENDSHIP_THRESHOLD",
    "EVO_CLASS_BY_ITEM",
    "EVO_CLASS_BY_LEVEL",
    "EVO_CLASS_BY_TRADE",
    "EVO_CLASS_UNUSED_02",
    "HOLD_EFFECT_NO_EVOLVE",
    "SPECIES_KADABRA",
    "evolution_target",
]

# enum EvolutionClass (include/pokemon.h:51-56).
EVO_CLASS_BY_LEVEL = 0
EVO_CLASS_BY_TRADE = 1
EVO_CLASS_UNUSED_02 = 2
EVO_CLASS_BY_ITEM = 3

EVOLVE_FRIENDSHIP_THRESHOLD = 220  # include/constants/pokemon.h:25
# HOLD_EFFECT_NO_EVOLVE, Everstone's effect (generated/item_hold_effects.txt line 65: id 64);
# compared with each game's own item data (``tables.item_hold_effect``).
HOLD_EFFECT_NO_EVOLVE = 64
SPECIES_KADABRA = 64  # generated/species.txt line 65

_DAY, _NIGHT = "day", "night"
_MALE, _FEMALE = 0, 1  # PokemonSpec.gender
# Level methods that need the level and Attack against Defense (src/pokemon.c:3592-3618).
_ATK_VS_DEF = {
    "level_atk_gt_def": lambda atk, dfn: atk > dfn,
    "level_atk_eq_def": lambda atk, dfn: atk == dfn,
    "level_atk_lt_def": lambda atk, dfn: atk < dfn,
}
_MAP_METHODS = frozenset({"level_magnetic_field", "level_moss_rock", "level_ice_rock"})


def evolution_target(
    tables: Tables,
    learnsets: Learnsets,
    member: PokemonSpec,
    party: Sequence[PokemonSpec] | None,
    evo_class: int,
    evo_param: int,
    time: str | None,
    *,
    personality: int,
    beauty: int,
) -> tuple[int, int] | None:
    """``(target species, method id)`` of the first evolution row that matches, else ``None``
    (``Pokemon_GetEvolutionTargetSpecies``, ``src/pokemon.c:3528-3790``). The method id is the
    row's for the level class and ``EVO_NONE`` (0) for the item class (``:3748-3782``).

    - Everstone (``HOLD_EFFECT_NO_EVOLVE``): no evolution unless the species is Kadabra or the
      class is ``EVO_CLASS_BY_ITEM`` (``:3541-3547``); the held item is read from ``member``.
    - ``EVO_CLASS_BY_LEVEL`` (``:3557-3726``): each row of the species in table order, the first
      match winning (``if (targetSpecies) break``, ``:3721-3723``). ``EVO_LEVEL_SHEDINJA``
      matches nothing (``:3644-3646``: it only sets the method result; Shedinja is GT-13).
      A map method row matches when the row's method equals ``evo_param`` (``:3699-3720``).
    - ``EVO_CLASS_BY_TRADE``: never offered, a run has no trade (design section 6, G7): ``None``.
    - ``EVO_CLASS_BY_ITEM`` and the unused class 2 (``:3745-3784``): a ``use_item`` row whose
      param is ``evo_param``, or a ``use_item_male`` / ``use_item_female`` row of the member's
      gender.

    ``party`` is party 0's specs, or ``None`` for no party (``:3676``: the species-in-party row
    then never matches). ``time``: see the module docstring. ``ValueError`` for an
    ``evo_class`` the game does not have, or a ``time`` that is not a setting value."""
    if evo_class not in (
        EVO_CLASS_BY_LEVEL,
        EVO_CLASS_BY_TRADE,
        EVO_CLASS_UNUSED_02,
        EVO_CLASS_BY_ITEM,
    ):
        raise ValueError(f"evolution class {evo_class} is not one of the four classes")
    if time not in (None, _DAY, _NIGHT):
        raise ValueError(f"time {time!r} is not 'day', 'night' or unset")
    hold = int(tables.item_hold_effect[member.item])
    blocked = member.species != SPECIES_KADABRA and hold == HOLD_EFFECT_NO_EVOLVE
    if blocked and evo_class != EVO_CLASS_BY_ITEM:
        return None  # src/pokemon.c:3541-3547
    rows = learnsets.evolutions(member.species)
    names = learnsets.evolution_methods
    if evo_class == EVO_CLASS_BY_LEVEL:
        return _by_level(tables, member, party, evo_param, time, personality, beauty, rows, names)
    if evo_class == EVO_CLASS_BY_TRADE:
        return None
    return _by_item(member, evo_param, rows, names)


def _by_level(
    tables, member, party, evo_param, time, personality, beauty, rows, names
) -> tuple[int, int] | None:
    """The ``EVO_CLASS_BY_LEVEL`` case (``src/pokemon.c:3557-3726``)."""
    level = member.level
    day, night = time == _DAY, time == _NIGHT
    happy = member.friendship >= EVOLVE_FRIENDSHIP_THRESHOLD
    for method, param, target in rows:
        name = names[method]
        if name == "level_happiness":
            matched = happy
        elif name == "level_happiness_day":
            matched = day and happy
        elif name == "level_happiness_night":
            matched = night and happy
        elif name in ("level", "level_ninjask"):
            matched = param <= level
        elif name in _ATK_VS_DEF:
            stats = spec_stats(tables, member)
            matched = param <= level and _ATK_VS_DEF[name](
                int(stats[Stat.ATK]), int(stats[Stat.DEF])
            )
        elif name == "level_pid_low":
            matched = param <= level and (personality >> 16) % 10 < 5
        elif name == "level_pid_high":
            matched = param <= level and (personality >> 16) % 10 >= 5
        elif name == "level_beauty":
            matched = param <= beauty
        elif name == "level_with_held_item_day":
            matched = day and param == member.item
        elif name == "level_with_held_item_night":
            matched = night and param == member.item
        elif name == "level_know_move":
            matched = param in member.moves
        elif name == "level_species_in_party":
            matched = party is not None and any(p.species == param for p in party)
        elif name == "level_male":
            matched = member.gender == _MALE and param <= level
        elif name == "level_female":
            matched = member.gender == _FEMALE and param <= level
        elif name in _MAP_METHODS:
            matched = method == evo_param
        else:  # level_shedinja only sets the method (:3644-3646); the rest are other classes
            matched = False
        if matched:
            return target, method
    return None


def _by_item(member, stone, rows, names) -> tuple[int, int] | None:
    """The ``EVO_CLASS_BY_ITEM`` case (``src/pokemon.c:3745-3784``): ``EVO_NONE`` as the method."""
    for method, param, target in rows:
        name = names[method]
        if param != stone:
            continue
        if (
            name == "use_item"
            or (name == "use_item_male" and member.gender == _MALE)
            or (name == "use_item_female" and member.gender == _FEMALE)
        ):
            return target, 0
    return None
