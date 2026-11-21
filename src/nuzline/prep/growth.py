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

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.gauntlet import Settings
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.ids import Stat
from nuzline.engine.state import spec_exp, spec_stats
from nuzline.engine.stats import SPECIES_SHEDINJA

__all__ = [
    "EVOLVE_FRIENDSHIP_THRESHOLD",
    "EVO_CLASS_BY_ITEM",
    "EVO_CLASS_BY_LEVEL",
    "EVO_CLASS_BY_TRADE",
    "EVO_CLASS_UNUSED_02",
    "HOLD_EFFECT_NO_EVOLVE",
    "ITEM_POKE_BALL",
    "MAX_PARTY_SIZE",
    "SPECIES_KADABRA",
    "evolution_target",
    "evolve",
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

ITEM_POKE_BALL = 4  # generated/items.txt line 5 (ids start at ITEM_NONE = 0 on line 1)
MAX_PARTY_SIZE = 6  # include/constants/pokemon.h:9
LEARNED_MOVES_MAX = 4
# GENDER_RATIO_* (generated/gender_ratios.txt lines 1, 7, 8); PokemonSpec.gender ids.
_GENDER_RATIO_MALE_ONLY, _GENDER_RATIO_FEMALE_ONLY, _GENDER_RATIO_NO_GENDER = 0, 254, 255
_GENDER_NONE = 2
# Methods whose Pokemon uses up its held item (src/evolution.c:782-787) and the two that add a
# Shedinja (:718-780), by ``Learnsets.evolution_methods`` name.
_HELD_ITEM_METHODS = frozenset(
    {"trade_with_held_item", "level_with_held_item_day", "level_with_held_item_night"}
)
_SHEDINJA_METHODS = frozenset({"level_ninjask", "level_shedinja"})

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


def _row(tables: Tables, species: int, form: int) -> tuple[int, int]:
    """``(personal row, form used)``: the form's row, else the species' base row with form 0.
    The game maps a form to its own personal row only for the species with data forms
    (``Pokemon_GetFormNarcIndex``, ``src/pokemon.c:4929-4960``), so any other form reads the
    species' base row; the engine cannot index such a form, so it is form 0."""
    try:
        return personal_row(tables, species, form), form
    except ValueError:
        return personal_row(tables, species, 0), 0


def _ability_slot(tables: Tables, row: int, personality: int) -> int:
    """``BoxPokemon_CalcAbility`` (``src/pokemon.c:4746-4763``): the second ability when the
    species has one and personality bit 0 is set, else the first."""
    second = int(np.asarray(tables.species_abilities_decomp)[row][1])
    return personality & 1 if second != 0 else 0


def _gender(tables: Tables, row: int, personality: int) -> int:
    """``SpeciesData_GetGenderOf`` (``src/pokemon.c:2707-2730``): the species' ratio, else
    the personality's low byte against it. ``PokemonSpec.gender``: 0 male, 1 female, 2 none."""
    ratio = int(np.asarray(tables.species_gender)[row])
    if ratio == _GENDER_RATIO_MALE_ONLY:
        return _MALE
    if ratio == _GENDER_RATIO_FEMALE_ONLY:
        return _FEMALE
    if ratio == _GENDER_RATIO_NO_GENDER:
        return _GENDER_NONE
    return _FEMALE if ratio > (personality & 0xFF) else _MALE


def _level_at(tables: Tables, species: int, exp: int) -> int:
    """``SpeciesData_GetLevelAt`` (``src/pokemon.c:2379-2392``): the last level 1..100 whose
    experience is at most ``exp`` on the species' own experience table."""
    rate = int(np.asarray(tables.species_exp_rate)[species])
    table = np.asarray(tables.exp_table)[rate]
    level = 1
    for i in range(1, len(table)):
        if table[i] > exp:
            break
        level = i
    return level


def _recalculated(tables: Tables, mon: PokemonSpec, species: int, personality: int) -> PokemonSpec:
    """``mon`` with ``MON_DATA_SPECIES`` set to ``species``, then ``Pokemon_CalcAbility`` and
    ``Pokemon_CalcLevelAndStats`` (``src/evolution.c:433-437``): the ability slot from the
    personality, the level from the experience on the new species' table
    (``Pokemon_CalcLevelAndStats``, ``src/pokemon.c:633-637``), the stats from the new base
    stats, and a nonzero current HP raised by the max-HP gain, or set to 1 for Shedinja
    (``Pokemon_CalcStats``, ``src/pokemon.c:640-651``). ``current_hp`` ``None`` (full HP)
    stays full: the gain added to a full HP is the new maximum. The stored stats are dropped
    and the ``ability_form`` / ``stats_form`` quirks of a trainer Pokemon end (the game reads
    the Pokemon's own form row)."""
    old_max = int(spec_stats(tables, mon)[Stat.HP])
    exp = spec_exp(tables, mon)
    row, form = _row(tables, species, mon.form)
    moved = replace(
        mon,
        species=species,
        form=form,
        level=_level_at(tables, species, exp),
        exp=exp,
        ability_slot=_ability_slot(tables, row, personality),
        ability_form=None,
        stats_form=None,
        stats=None,
    )
    hp = moved.current_hp
    if hp is not None and hp != 0:
        new_max = int(spec_stats(tables, moved)[Stat.HP])
        hp = 1 if species == SPECIES_SHEDINJA else hp + new_max - old_max
    return replace(moved, current_hp=hp)


def _max_pp(tables: Tables, move: int, ups: int) -> int:
    """``MoveTable_CalcMaxPP`` (``src/move_table.c:25-35``)."""
    pp = int(np.asarray(tables.move_pp)[move])
    return pp + pp * 20 * min(ups, 3) // 100


def _set_slot(tables: Tables, mon: PokemonSpec, move: int, slot: int, ups: int) -> PokemonSpec:
    """``BoxPokemon_SetMoveSlot`` (``src/pokemon.c:3925-3933``) with the slot's PP Ups at
    ``ups``: the move, and its PP at the maximum for those ups. ``pp`` ``None`` (table PP)
    stays ``None``."""
    moves = list(mon.moves)
    moves[slot] = move
    pp_ups = list(mon.pp_ups)
    pp_ups[slot] = ups
    pp = mon.pp
    if pp is not None:
        filled = list(pp)
        filled[slot] = _max_pp(tables, move, ups)
        pp = tuple(filled)
    return replace(mon, moves=tuple(moves), pp_ups=tuple(pp_ups), pp=pp)


def _post_evolution_learn(
    tables: Tables,
    learnsets: Learnsets,
    mon: PokemonSpec,
    learn_answers: Mapping[int, int | None],
) -> PokemonSpec:
    """``EVOLUTION_STATE_CHECK_LEARN_MOVE`` (``src/evolution.c:461-480``): each entry of the
    new species' learnset at exactly the current level, in table order
    (``Pokemon_LevelUpMove``, ``src/pokemon.c:3935-3966``; ``(entry & 0xFE00) == level << 9``,
    ``:3951``), goes through ``Pokemon_AddMove`` (``:3852-3874``): a move already known is
    skipped, the first empty slot takes it (the slot's PP Ups kept, ``:3925-3933``).
    With four moves the game asks (``src/evolution.c:508-526``): ``learn_answers[move]`` is the
    slot (0-3) to forget, which sets the slot's PP Ups to 0 (``:631-633``). No entry or ``None``
    is the "no" path (``MENU_CANCEL`` at ``:525``, ``MENU_YES`` of "stop trying" at ``:590-595``,
    the summary screen's ``selectedMoveSlot == LEARNED_MOVES_MAX`` at ``:570``), the strictest
    answer. A slot holding an HM cannot be chosen (``Item_IsHMMove``,
    ``src/applications/pokemon_summary_screen/main.c:814-820``): ``ValueError``."""
    hm = np.asarray(tables.move_is_hm)
    for lv, move in learnsets.level_up(mon.species, mon.form):
        if lv != mon.level or move in mon.moves:
            continue
        if 0 in mon.moves:
            slot = mon.moves.index(0)
            mon = _set_slot(tables, mon, move, slot, mon.pp_ups[slot])
            continue
        answer = learn_answers.get(move)
        if answer is None:
            continue
        if not 0 <= answer < LEARNED_MOVES_MAX:
            raise ValueError(f"move {move}: answer {answer} is not a slot 0-3")
        if hm[mon.moves[answer]]:
            raise ValueError(
                f"move {move}: slot {answer} holds the HM move {mon.moves[answer]}, "
                "which cannot be forgotten"
            )
        mon = _set_slot(tables, mon, move, answer, 0)
    return mon


def _shedinja(tables: Tables, mon: PokemonSpec, personality: int) -> PokemonSpec:
    """The Shedinja block of ``Evolution_ProcessEvolutionEffects`` (``src/evolution.c:718-780``)
    applied to the evolved ``mon``, field by field:

    - ``:731`` ``Pokemon_Copy``: every field not named below is the evolved Pokemon's (IVs,
      EVs, nature, moves, PP, PP Ups, friendship, experience, Pokerus, egg location, form,
      personality);
    - ``:733-734`` species ``SPECIES_SHEDINJA``;
    - ``:736-737`` ball ``ITEM_POKE_BALL``: no ``PokemonSpec`` field;
    - ``:740`` held item 0;
    - ``:741`` markings 0, ``:743-753`` ribbons 0, ``:755`` nickname reset to the species
      name, ``:756`` ``MON_DATA_HAS_NICKNAME`` 0, ``:759-760`` empty mail, ``:762`` ball
      capsule id 0, ``:764-766`` empty ball capsule: no ``PokemonSpec`` fields;
    - ``:757`` status 0 (and so the sleep counter);
    - ``:767`` ``Pokemon_CalcAbility``; ``:769-770`` ``Pokemon_GetGender`` stored as gender;
    - ``:772`` ``Pokemon_CalcLevelAndStats``: level from experience on Shedinja's table, max
      HP 1, current HP 1 (``src/pokemon.c:640-643``; the evolved Pokemon is alive, a dead
      one never evolves)."""
    copy = _recalculated(tables, mon, SPECIES_SHEDINJA, personality)
    row, _ = _row(tables, SPECIES_SHEDINJA, mon.form)
    return replace(
        copy,
        item=0,
        status=0,
        status_counter=0,
        gender=_gender(tables, row, personality),
        current_hp=None if mon.current_hp is None else 1,
    )


def evolve(
    tables: Tables,
    learnsets: Learnsets,
    member: PokemonSpec,
    target: int,
    method: int,
    bag: Mapping[int, int],
    party: Sequence[PokemonSpec],
    settings: Settings,
    *,
    personality: int,
    learn_answers: Mapping[int, int | None] | None = None,
) -> tuple[PokemonSpec, PokemonSpec | None, dict[int, int]]:
    """Apply an evolution (``target`` and ``method`` as :func:`evolution_target` returns
    them): ``(evolved member, the added Shedinja or None, the bag afterwards)`` (ledger
    E3-B3-13).

    The order is the game's (``src/evolution.c``): species, ability and stats
    (``:433-437``), the learn loop (``:461-480``), then ``Evolution_ProcessEvolutionEffects``
    (``:708-790``).

    - ``bag``: item id -> count, a plain mapping (GT-4's Carry has no bag type yet). The result
      is a new dict; ``bag`` is not changed.
    - ``party``: party 0's members, ``member`` included (``Party_GetCurrentCount``,
      ``:725``). A Shedinja needs fewer than ``MAX_PARTY_SIZE`` of them, a Poke Ball in the
      bag and ``settings["shedinja"] == "allowed"``; ``forbidden`` (the default) adds none, the
      plan does not count one. It goes at the end of the party (``:773``) and one Poke Ball
      is used up (``:779``).
    - ``method``: ``trade_with_held_item``, ``level_with_held_item_day`` and ``_night`` set the
      held item to none (``:782-787``).
    - ``personality``: the member's, a constant of the Pokemon (``MON_DATA_PERSONALITY``).
    - ``learn_answers``: move id -> slot to forget (:func:`_post_evolution_learn`); a move
      without an answer is given up.
    - A dead member (current HP 0) does not evolve: the answer is cancel and nothing changes
      (design section 6, "Decisions")."""
    out_bag = dict(bag)
    if member.current_hp == 0:
        return member, None, out_bag
    name = learnsets.evolution_methods[method]
    mon = _recalculated(tables, member, target, personality)
    mon = _post_evolution_learn(tables, learnsets, mon, learn_answers or {})
    extra = None
    if name in _SHEDINJA_METHODS:
        shedinja = settings["shedinja"].value
        if (
            shedinja == "allowed"
            and out_bag.get(ITEM_POKE_BALL, 0) > 0
            and len(party) < MAX_PARTY_SIZE
        ):
            extra = _shedinja(tables, mon, personality)
            out_bag[ITEM_POKE_BALL] -= 1
    elif name in _HELD_ITEM_METHODS:
        mon = replace(mon, item=0)
    return mon, extra, out_bag
