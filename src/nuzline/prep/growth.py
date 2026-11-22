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

import itertools
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.gauntlet import UNLIMITED, Settings
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.coverage import learn_masked
from nuzline.engine.damage import WEIGHT_BOUNDS
from nuzline.engine.ids import (
    SPECIES_ARCEUS,
    SPECIES_CASTFORM,
    SPECIES_CHANSEY,
    SPECIES_CHATOT,
    SPECIES_CHERRIM,
    SPECIES_CLAMPERL,
    SPECIES_CUBONE,
    SPECIES_DIALGA,
    SPECIES_DITTO,
    SPECIES_FARFETCHD,
    SPECIES_GIRATINA,
    SPECIES_LATIAS,
    SPECIES_LATIOS,
    SPECIES_MAROWAK,
    SPECIES_PALKIA,
    SPECIES_PIKACHU,
    SPECIES_SHAYMIN,
    ItemParam,
    Stat,
)
from nuzline.engine.levelup import (
    BOOST_PERCENT,
    FRIENDSHIP_TIERS,
    HOLD_EFFECT_FRIENDSHIP_UP,
    MAX_FRIENDSHIP,
    MAX_LEVEL,
)
from nuzline.engine.state import spec_exp, spec_stats
from nuzline.engine.stats import SPECIES_SHEDINJA

if TYPE_CHECKING:
    from nuzline.prep.gauntlet import Carry

__all__ = [
    "BASE_PLAN",
    "BATTLE_RESULT_CAPTURED_MON",
    "BATTLE_RESULT_PLAYER_FLED",
    "BATTLE_RESULT_WIN",
    "EVOLVE_FRIENDSHIP_THRESHOLD",
    "EVO_CLASS_BY_ITEM",
    "EVO_CLASS_BY_LEVEL",
    "EVO_CLASS_BY_TRADE",
    "EVO_CLASS_UNUSED_02",
    "EVO_NONE",
    "HOLD_EFFECT_NO_EVOLVE",
    "ITEM_POKE_BALL",
    "MAX_GROWTH_PLANS",
    "MAX_PARTY_SIZE",
    "SPECIES_KADABRA",
    "EvolutionGap",
    "EvolutionOption",
    "GrowthPlan",
    "GrowthPlanSet",
    "MemberOptions",
    "Prompt",
    "evolution_prompt",
    "evolution_target",
    "evolve",
    "evolve_dominates",
    "give_up_only",
    "growth_plans",
    "learn_offers",
    "member_options",
    "member_plan_count",
    "member_plans",
    "post_battle_prompts",
    "rare_candy",
    "resolve_post_battle",
    "use_stone",
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
# The species a battle rule names by id, so evolving into or out of one changes a rule
# (``evolve_dominates``): the species-bound held items of BattleSystem_CalcMoveDamage
# (Latios/Latias, Clamperl, Pikachu, Ditto, Cubone/Marowak, Dialga, Palkia, Giratina;
# battle_lib.c:6730-6774), Quick Powder (Ditto, :1252-1254), the crit items
# (Chansey, Farfetch'd; :7127-7128), the form changes (Castform :6314, Cherrim :6366,
# Arceus :6399 and :7629, Giratina :6412, Shaymin subscript_check_shaymin_form.s:13),
# Chatter (Chatot, battle_script.c:8402) and Burmy's form after the battle
# (battle_system.c:938, :1562).
_SPECIES_BURMY = 412  # generated/species.txt line 413
_SPECIES_KEYED = frozenset(
    {
        SPECIES_PIKACHU,
        SPECIES_FARFETCHD,
        SPECIES_CUBONE,
        SPECIES_MAROWAK,
        SPECIES_CHANSEY,
        SPECIES_DITTO,
        SPECIES_CASTFORM,
        SPECIES_CLAMPERL,
        SPECIES_LATIAS,
        SPECIES_LATIOS,
        _SPECIES_BURMY,
        SPECIES_CHERRIM,
        SPECIES_CHATOT,
        SPECIES_DIALGA,
        SPECIES_PALKIA,
        SPECIES_GIRATINA,
        SPECIES_SHAYMIN,
        SPECIES_ARCEUS,
    }
)

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
    the Pokemon's own form row). The gender is the new species' for the personality: the game
    stores none that survives, every read recomputes it (``BoxPokemon_GetGender``,
    ``src/pokemon.c:2696-2705``; the battle's, ``battle_lib.c:116``)."""
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
        gender=_gender(tables, row, personality),
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


# --- GT-14: evolution prompts, Rare Candy, stones (design v5.4 section 6 G6, G7; E3-B3-14) ------

# include/constants/battle.h:72, :74, :79.
BATTLE_RESULT_WIN = 1 << 0
BATTLE_RESULT_CAPTURED_MON = 1 << 2
BATTLE_RESULT_PLAYER_FLED = BATTLE_RESULT_CAPTURED_MON | BATTLE_RESULT_WIN
# The results after which the game looks for an evolving member
# (``Battle_FindEvolvingPartyMember``, src/battle/battle_system.c:829-833).
_EVOLVING_RESULTS = frozenset(
    {BATTLE_RESULT_WIN, BATTLE_RESULT_CAPTURED_MON, BATTLE_RESULT_PLAYER_FLED}
)
EVO_NONE = 0  # generated/evolution_methods.txt, line 1: the method of an item evolution


class Prompt(NamedTuple):
    """One evolution prompt of the game: carry slot ``slot`` of species ``species`` would evolve
    into ``target`` by ``method`` (:func:`evolution_target`'s answer) in class ``evo_class``.

    ``cancellable``: the class passes the cancel flag 0x1 (``EVO_CLASS_BY_LEVEL`` after a battle,
    ``battle_main.c:283``, and after a Rare Candy, ``src/start_menu.c:1740``); a stone passes
    ``NULL`` (``:1742``) and cannot be cancelled. ``dead``: the member is at 0 HP: the game
    offers it (HP is not checked, ``battle_system.c:826-850``) but the answer is cancel and
    nothing changes (design section 6, "Decisions"; :func:`evolve`)."""

    slot: int
    species: int
    target: int
    method: int
    evo_class: int
    cancellable: bool
    dead: bool


def evolution_prompt(
    tables: Tables,
    learnsets: Learnsets,
    slot: int,
    member: PokemonSpec,
    party: Sequence[PokemonSpec] | None,
    evo_class: int,
    evo_param: int,
    time: str | None,
    *,
    personality: int,
    beauty: int = 0,
) -> Prompt | None:
    """The prompt the game raises for ``member`` (:func:`evolution_target` and its arguments),
    else ``None``."""
    found = evolution_target(
        tables,
        learnsets,
        member,
        party,
        evo_class,
        evo_param,
        time,
        personality=personality,
        beauty=beauty,
    )
    if found is None:
        return None
    target, method = found
    return Prompt(
        slot,
        member.species,
        target,
        method,
        evo_class,
        evo_class == EVO_CLASS_BY_LEVEL,
        member.current_hp == 0,
    )


def _party_specs(carry: Carry) -> dict[int, PokemonSpec]:
    from nuzline.prep.gauntlet import member_spec  # gauntlet imports this module

    return {int(i): member_spec(carry, int(i)) for i in np.flatnonzero(carry.present)}


def _levelled_mask(levelled: Sequence[bool] | np.ndarray) -> np.ndarray:
    mask = np.asarray(levelled, dtype=bool)
    if mask.shape != (MAX_PARTY_SIZE,):
        raise ValueError(f"levelled is a party-0 slot mask of {MAX_PARTY_SIZE}, got {mask.shape}")
    return mask


def post_battle_prompts(
    tables: Tables,
    learnsets: Learnsets,
    carry: Carry,
    levelled: Sequence[bool] | np.ndarray,
    result: int,
    map_method: int,
    time: str | None,
) -> list[Prompt]:
    """The evolution prompts after a battle, as the game raises them from ``carry`` (a single
    Carry after the battle: levels, friendship and moves as the level-ups left them).

    ``result`` is the battle's ``resultMask`` (``BATTLE_RESULT_*``): only a win, a capture or a
    flee looks for an evolving member (``src/battle/battle_system.c:829-833``); anything else,
    a draw or a loss included, gives ``[]``. ``levelled``: the ``leveledUpMonsMask`` as a
    party-0 slot mask (``bool[6]``; ``bs.c:9867``): only those members are checked, each once,
    slot by slot, with class ``EVO_CLASS_BY_LEVEL`` and the battle's map method
    (``battle_system.c:835-849``); ``map_method`` is ``Learnsets.map_evolution_method`` of the
    map. A member is offered whether or not it fainted (``dead`` on the prompt). The prompts are
    those of the Carry as it stands, every answer a cancel; :func:`resolve_post_battle` follows
    the answers (an evolution changes the party the later slots are checked against). A cancelled
    member is offered again only when it levels again (the mask is per battle)."""
    mask = _levelled_mask(levelled)
    if result not in _EVOLVING_RESULTS:
        return []
    members = _party_specs(carry)
    party = list(members.values())
    out = []
    for slot, member in members.items():
        if not mask[slot]:
            continue
        prompt = evolution_prompt(
            tables,
            learnsets,
            slot,
            member,
            party,
            EVO_CLASS_BY_LEVEL,
            map_method,
            time,
            personality=int(carry.personality[slot]),
            beauty=int(carry.beauty[slot]),
        )
        if prompt is not None:
            out.append(prompt)
    return out


def resolve_post_battle(
    tables: Tables,
    learnsets: Learnsets,
    carry: Carry,
    levelled: Sequence[bool] | np.ndarray,
    result: int,
    map_method: int,
    time: str | None,
    settings: Settings,
    decide: Callable[[Prompt], Mapping[int, int | None] | None],
) -> tuple[Carry, list[Prompt]]:
    """:func:`post_battle_prompts` with the answers applied, slot by slot as the game's loop
    (``Battle_FindEvolvingPartyMember``, ``battle_system.c:835-849``): ``decide(prompt)`` is
    ``None`` to cancel, or the post-evolution learn answers (move id -> slot, see
    :func:`evolve`) to evolve. An evolution changes the party the next slots are checked
    against (``Party_HasSpecies``; an added Shedinja is not in the mask). A dead member is
    offered but never evolves. Returns ``(the Carry afterwards, the prompts offered)``."""
    from nuzline.prep.gauntlet import evolve_carry

    mask = _levelled_mask(levelled)
    offered: list[Prompt] = []
    if result not in _EVOLVING_RESULTS:
        return carry, offered
    for slot in range(MAX_PARTY_SIZE):
        if not mask[slot] or not carry.present[slot]:
            continue
        members = _party_specs(carry)
        prompt = evolution_prompt(
            tables,
            learnsets,
            slot,
            members[slot],
            list(members.values()),
            EVO_CLASS_BY_LEVEL,
            map_method,
            time,
            personality=int(carry.personality[slot]),
            beauty=int(carry.beauty[slot]),
        )
        if prompt is None:
            continue
        offered.append(prompt)
        if prompt.dead:
            continue
        answer = decide(prompt)
        if answer is not None:
            carry = evolve_carry(
                tables,
                learnsets,
                carry,
                slot,
                prompt.target,
                prompt.method,
                settings,
                learn_answers=answer,
            )
    return carry, offered


def _item_friendship(tables: Tables, member: PokemonSpec, item: int, location: int | None) -> int:
    """The member's friendship after an item's friendship params: the tier of the current value
    picks ``GIVE_FRIENDSHIP_*`` and ``FRIENDSHIP_*`` (``Pokemon_ApplyItemEffects``,
    ``src/item_use_pokemon.c:363-387``), then ``UpdatePokemonFriendship`` (``:565-600``): no
    change at 255 for a gain or at 0 for a loss (``:567-573``); for a gain, Soothe Bell x150/100
    first (``:575-578``, ``include/constants/items.h:50``), then +1 for a Luxury Ball
    (``:580-582``) and +1 when the egg location equals ``location`` (``:584-586``); clamp
    0..255 (``:591-598``). ``tables.item_party_param`` holds the item's params
    (``ids.ItemParam``). The battle level-up differs: ``Pokemon_UpdateFriendship`` adds the
    bonuses first and multiplies after (``src/pokemon.c:2664-2678``)."""
    params = np.asarray(tables.item_party_param)[item]
    friendship = member.friendship
    low, med = FRIENDSHIP_TIERS
    if friendship < low:
        give, change = ItemParam.GIVE_FRIENDSHIP_LOW, ItemParam.FRIENDSHIP_LOW
    elif friendship < med:
        give, change = ItemParam.GIVE_FRIENDSHIP_MED, ItemParam.FRIENDSHIP_MED
    elif friendship <= MAX_FRIENDSHIP:
        give, change = ItemParam.GIVE_FRIENDSHIP_HIGH, ItemParam.FRIENDSHIP_HIGH
    else:
        return friendship
    if not params[give]:
        return friendship
    delta = int(params[change])
    if (friendship == MAX_FRIENDSHIP and delta > 0) or (friendship == 0 and delta < 0):
        return friendship
    if delta > 0:
        if int(np.asarray(tables.item_hold_effect)[member.item]) == HOLD_EFFECT_FRIENDSHIP_UP:
            delta = delta * BOOST_PERCENT // 100
        if member.friend_bonus:
            delta += 1
        if location is not None and member.egg_location == location:
            delta += 1
    return min(max(friendship + delta, 0), MAX_FRIENDSHIP)


def rare_candy(
    tables: Tables,
    learnsets: Learnsets,
    member: PokemonSpec,
    item: int,
    *,
    location: int | None = None,
    learn_answers: Mapping[int, int | None] | None = None,
) -> PokemonSpec:
    """``member`` after one Rare Candy ``item`` (ledger E3-B3-14; ``Pokemon_ApplyItemEffects``,
    ``src/item_use_pokemon.c:237-391``, then the party menu's level-up states).

    - usable only below level 100 (``:171-175``; ``ValueError`` otherwise);
    - experience to the next level and ``Pokemon_CalcLevelAndStats`` (``:291-293``): the new
      level, stats, and current HP raised by the max-HP gain (``src/pokemon.c:640-651``; 1 for
      Shedinja). A fainted member is revived with HP equal to the gain (``:294-298``,
      ``RestorePokemonHP``, ``:478-495``; 1 when the maximum is 1). Under
      ``revived_dead = stay_dead`` the member stays dead and barred: that flag is the Carry's
      (``dead``), which :func:`~nuzline.prep.gauntlet.set_member` does not touch (GT-5);
    - friendship from the item's params by tier (:func:`_item_friendship`), read from
      ``tables.item_party_param[item]``; ``location`` is the map the item is used on
      (``None``: no egg-location bonus);
    - the learn prompts of the new level (``Pokemon_LevelUpMove``, ``src/pokemon.c:3935-3966``):
      an empty slot learns, with four moves ``learn_answers[move]`` is the slot to forget (its PP
      Ups become 0, ``src/applications/party_menu/callbacks.c:1105-1113``) and a missing answer
      or ``None`` gives the move up, as :func:`evolve`'s learn loop.

    The evolution check that follows (``callbacks.c:816-829``; the current header's map method is
    read at ``:818``) is :func:`evolution_prompt` with ``EVO_CLASS_BY_LEVEL`` on the returned
    member: the caller answers it. Which item ids are Rare Candies (``ITEM_PARAM_LEVEL_UP``) is
    the caller's; the item table holds no such column."""
    if member.level >= MAX_LEVEL:
        raise ValueError(f"level {member.level}: a Rare Candy needs a level below {MAX_LEVEL}")
    rate = int(np.asarray(tables.species_exp_rate)[member.species])
    exp = int(np.asarray(tables.exp_table)[rate][member.level + 1])
    old_max = int(spec_stats(tables, member)[Stat.HP])
    moved = replace(member, exp=exp, level=_level_at(tables, member.species, exp), stats=None)
    new_max = int(spec_stats(tables, moved)[Stat.HP])
    hp = member.current_hp
    if hp is not None:
        if hp != 0:
            hp = 1 if member.species == SPECIES_SHEDINJA else hp + new_max - old_max
        else:
            hp = 1 if new_max == 1 else min(new_max, new_max - old_max)
    moved = replace(moved, current_hp=hp)
    moved = replace(moved, friendship=_item_friendship(tables, moved, item, location))
    return _post_evolution_learn(tables, learnsets, moved, learn_answers or {})


def use_stone(
    tables: Tables,
    learnsets: Learnsets,
    member: PokemonSpec,
    stone: int,
    bag: Mapping[int, int],
    party: Sequence[PokemonSpec],
    settings: Settings,
    *,
    personality: int,
    learn_answers: Mapping[int, int | None] | None = None,
) -> tuple[PokemonSpec, PokemonSpec | None, dict[int, int]]:
    """Use the evolution item ``stone`` on ``member``: ``(evolved member, an added Shedinja or
    None, the bag afterwards)``. The item must evolve the member
    (``Pokemon_GetEvolutionTargetSpecies`` with ``EVO_CLASS_BY_ITEM`` and no party,
    ``src/item_use_pokemon.c:177-181``; ``ValueError`` otherwise) and the bag must hold it. The
    stone is used up when the effect is accepted, before the evolution screen
    (``Bag_TryRemoveItem``, ``src/applications/party_menu/main.c:2699-2701``); the screen passes
    no cancel flag (``src/start_menu.c:1742``), so it cannot be cancelled: the decision is
    whether to call this. The item path applies no friendship (``main.c:2702-2706`` leaves the
    menu straight into the evolution). A dead member is not offered a stone (design section 6):
    ``ValueError``. ``party``, ``settings``, ``personality`` and ``learn_answers`` as
    :func:`evolve`."""
    if member.current_hp == 0:
        raise ValueError("a dead member is not offered an evolution stone")
    if bag.get(stone, 0) < 1:
        raise ValueError(f"the bag holds no item {stone}")
    found = evolution_target(
        tables,
        learnsets,
        member,
        None,
        EVO_CLASS_BY_ITEM,
        stone,
        None,
        personality=personality,
        beauty=0,
    )
    if found is None:
        raise ValueError(f"item {stone} does not evolve species {member.species}")
    target, method = found
    after = dict(bag)
    after[stone] -= 1
    return evolve(
        tables,
        learnsets,
        member,
        target,
        method,
        after,
        party,
        settings,
        personality=personality,
        learn_answers=learn_answers,
    )


# --- GT-15: growth plan, never-better pruning, --growth-cap (design v5.4 section 6; E3-B3-15) ---

MAX_GROWTH_PLANS = 64  # the default of --growth-cap (design section 6, "Bound", rule 3)
GIVE_UP_ANSWER = None  # an answer is a move slot 0-3, or None to give the move up

Offer = tuple[int, int]  # (level, move): one learnset entry the member is offered at that level


class GrowthPlan(NamedTuple):
    """One member's plan, held in every playout (design section 6, "Decisions").

    ``answers``: ``((level, move), slot or None)`` per offered move that has a free answer,
    sorted; a slot 0-3 replaces that move slot, ``None`` gives the move up. An offer without an
    entry is given up. ``evolve_at``: the gap (an index of the chain's gaps, ``EvolutionGap.gap``)
    where the member evolves, or ``None`` for never."""

    answers: tuple[tuple[Offer, int | None], ...] = ()
    evolve_at: int | None = None

    def answer(self, offer: Offer) -> int | None:
        """The answer to ``offer``: its entry, else give up."""
        return dict(self.answers).get(offer)

    def answers_at(self, level: int) -> dict[int, int | None]:
        """The answers to the offers of ``level`` as ``evolve`` and ``rare_candy`` take them
        (move id -> slot or ``None``)."""
        return {move: slot for (lv, move), slot in self.answers if lv == level}


# The base plan (funnel): learn into an empty slot, else give up; never evolve
# (``search/learn.py`` ``learn_policy``, ``GIVE_UP``).
BASE_PLAN = GrowthPlan()


class EvolutionGap(NamedTuple):
    """An eligible evolution gap of one member, from the caller (the chain): the gap's index,
    the member's ``level`` there, and the ``target`` species and ``method`` that
    :func:`evolution_target` gives it there."""

    gap: int
    level: int
    target: int
    method: int


class EvolutionOption(NamedTuple):
    """An eligible gap with the moves offered on that path: ``pre`` the old line's offers up to
    and including the evolution level, ``post`` the new species' offers from that level to the
    last cap (``src/evolution.c:461-480`` offers its entries of exactly the current level)."""

    gap: int
    level: int
    target: int
    method: int
    pre: tuple[Offer, ...]
    post: tuple[Offer, ...]


class MemberOptions(NamedTuple):
    """What the plan of one member chooses among: ``member`` and ``personality`` as it stands at
    the root, ``offers`` the moves offered before the last cap while it never evolves, and one
    :class:`EvolutionOption` per eligible gap (in gap order)."""

    slot: int
    member: PokemonSpec
    personality: int
    offers: tuple[Offer, ...]
    evolutions: tuple[EvolutionOption, ...]


def learn_offers(
    learnsets: Learnsets,
    species: int,
    form: int,
    known: Collection[int],
    level_from: int,
    level_to: int,
) -> tuple[Offer, ...]:
    """The learnset entries of ``species`` at levels above ``level_from`` up to ``level_to``
    whose move is not in ``known`` (``Pokemon_LevelUpMove`` offers each entry of exactly the
    level reached, ``src/pokemon.c:3935-3966``; a known move is skipped, ``:3852-3874``), in
    table order, once each."""
    out: dict[Offer, None] = {}
    for level, move in learnsets.level_up(species, form):
        if level_from < level <= level_to and move not in known:
            out.setdefault((level, move))
    return tuple(out)


def member_options(
    tables: Tables,
    learnsets: Learnsets,
    slot: int,
    member: PokemonSpec,
    personality: int,
    cap: int,
    gaps: Sequence[EvolutionGap],
) -> MemberOptions:
    """The plan's choices for ``member`` up to the last level cap ``cap``, with ``gaps`` its
    eligible evolution gaps. The offers count the moves not known at the root; whether a slot is
    still empty when one is offered is a runtime fact (an offer with an empty slot is answered by
    the game itself and its answer is unused)."""
    known = set(member.moves)
    offers = learn_offers(learnsets, member.species, member.form, known, member.level, cap)
    options = []
    for gap in sorted(gaps, key=lambda g: g.gap):
        _, target_form = _row(tables, gap.target, member.form)
        options.append(
            EvolutionOption(
                gap.gap,
                gap.level,
                gap.target,
                gap.method,
                learn_offers(
                    learnsets, member.species, member.form, known, member.level, gap.level
                ),
                learn_offers(learnsets, gap.target, target_form, known, gap.level - 1, cap),
            )
        )
    return MemberOptions(slot, member, personality, offers, tuple(options))


def _accuracy(value: int) -> int:
    return 101 if value == 0 else value  # accuracy 0 never misses


def give_up_only(tables: Tables, known: Collection[int], move: int) -> bool:
    """Whether the only answer worth enumerating for the offered ``move`` is to give it up
    (never-better pruning, design section 6, "Bound" rule 2): its effect is unsupported
    (``coverage.learn_masked``, ruling D10), or a known move has the same effect id, type and
    category and a power, accuracy (0 = never misses) and PP at least as high. The known move
    must also match priority, target, move params and flags, and have at least the offered
    move's secondary-effect chance: a difference there could make the offered move the better
    one. The offered move is then never better on these columns, but the pruning is not exact:
    a second copy of an equal move can still be the better plan (more PP in total; a second
    copy to use under Torment, Disable or Encore on the first). Pruning that answer only lowers
    the result, never raises it."""
    if learn_masked(tables, move):
        return True
    t = {name: np.asarray(getattr(tables, name)) for name in _MOVE_COLUMNS}
    for k in known:
        if k == 0 or k == move:
            continue
        same = all(
            np.array_equal(t[name][k], t[name][move])
            for name in (
                "move_effect",
                "move_type",
                "move_category",
                "move_priority",
                "move_target",
                "move_param",
                "move_flags",
            )
        )
        if (
            same
            and t["move_power"][k] >= t["move_power"][move]
            and _accuracy(int(t["move_accuracy"][k])) >= _accuracy(int(t["move_accuracy"][move]))
            and t["move_pp"][k] >= t["move_pp"][move]
            and t["move_effect_chance"][k] >= t["move_effect_chance"][move]
        ):
            return True
    return False


_MOVE_COLUMNS = (
    "move_effect",
    "move_type",
    "move_category",
    "move_priority",
    "move_target",
    "move_param",
    "move_flags",
    "move_power",
    "move_accuracy",
    "move_pp",
    "move_effect_chance",
)


def evolve_dominates(
    tables: Tables, learnsets: Learnsets, options: MemberOptions, option: EvolutionOption
) -> bool:
    """Whether evolving at ``option`` never does worse than cancelling (design section 6,
    "Bound" rule 2), claimed only when the evolved form is the same in every battle rule and
    the only difference is extra learn offers (each of which can be given up):

    - the same types, the same ability, and the same species ability pair (the trainer AI
      guesses an unannounced ability from the species' two, ``trainer_ai.c:1191-1192``);
    - the same base stats. A rise is not provably better: a faster battler moves last under
      Trick Room (``battle_lib.c:1432-1435``), Gyro Ball's power falls with the user's speed and
      rises with the target's (``battle_script.c:7127``), the confusion self-hit uses the
      battler's own Attack (``CALC_SELF_HIT``, ``battle_controller_player.c:2631``), and the
      trainer AI scores its moves by damage against the battler's stats
      (``TrainerAI_CalcDamage``, ``trainer_ai.c:2835``);
    - the same gender for the personality (the battle recomputes it from the species,
      ``src/pokemon.c:2696-2705``, ``battle_lib.c:116``);
    - the same experience group (later level-ups and :func:`_recalculated`'s level come from
      the species' own experience table);
    - the same Low Kick / Grass Knot power against it: ``BattleMon.weight`` is the species'
      Pokedex weight (``battle_lib.c:150``), read by ``BtlCmd_CalcWeightBasedPower``
      (``battle_script.c:6808-6826``) and the AI's power estimate (``trainer_ai.c:3059-3066``);
    - neither species is one a battle rule names (``_SPECIES_KEYED``: species-bound held items
      such as Chansey's Lucky Punch, form changes, Chatter);
    - the method neither uses up a held item (``_HELD_ITEM_METHODS``) nor adds a Shedinja
      (``_SHEDINJA_METHODS``);
    - every offer of the cancelled path, ``(level, move)``, is offered at the same level on the
      evolved path (an offer at another level changes which moves the member holds in the
      fights between)."""
    member = options.member
    old_row, _ = _row(tables, member.species, member.form)
    new_row, _ = _row(tables, option.target, member.form)
    if member.species in _SPECIES_KEYED or option.target in _SPECIES_KEYED:
        return False
    name = learnsets.evolution_methods[option.method]
    if name in _HELD_ITEM_METHODS or name in _SHEDINJA_METHODS:
        return False
    for column in ("species_types", "species_abilities_decomp", "species_base_stats"):
        values = np.asarray(getattr(tables, column))
        if not np.array_equal(values[old_row], values[new_row]):
            return False
    abilities = np.asarray(tables.species_abilities_decomp)
    new_slot = _ability_slot(tables, new_row, options.personality)
    if abilities[new_row][new_slot] != abilities[old_row][member.ability_slot]:
        return False
    # Attract, Cute Charm and Rivalry read the gender, recomputed from the species
    # (``BoxPokemon_GetGender``, ``src/pokemon.c:2696-2705``; ``battle_lib.c:116``).
    if _gender(tables, new_row, options.personality) != _gender(
        tables, old_row, options.personality
    ):
        return False
    rates = np.asarray(tables.species_exp_rate)
    if int(rates[member.species]) != int(rates[option.target]):
        return False
    weights = np.asarray(tables.species_weight)
    if _weight_bracket(int(weights[member.species])) != _weight_bracket(
        int(weights[option.target])
    ):
        return False
    kept = set(option.pre + option.post)
    return all(offer in kept for offer in options.offers)


def _weight_bracket(weight: int) -> int:
    """The index of Low Kick's / Grass Knot's power for a defender of ``weight`` hectograms:
    the first bound at least the weight (``sWeightToPower``, ``battle_script.c:6815-6826``)."""
    return sum(weight > bound for bound in WEIGHT_BOUNDS)


def _paths(
    tables: Tables, learnsets: Learnsets, options: MemberOptions
) -> list[tuple[int | None, tuple[Offer, ...]]]:
    """The non-dominated ``(evolve_at, offers on that path)`` of one member: never plus every
    eligible gap, or, when evolving at the earliest gap dominates cancelling, only that gap
    (design section 6, "Bound" rule 2)."""
    paths = [(o.gap, tuple(dict.fromkeys(o.pre + o.post))) for o in options.evolutions]
    if options.evolutions and evolve_dominates(tables, learnsets, options, options.evolutions[0]):
        return paths[:1]
    return [(None, options.offers), *paths]


def _answer_lists(
    tables: Tables, options: MemberOptions, offers: Sequence[Offer]
) -> Iterator[tuple[tuple[Offer, int | None], ...]]:
    """Every answer list of one path's ``offers``, taken in game order (by level; within a
    level the order given: the old species' battle level-up before the evolution's own learn
    loop, ``src/evolution.c:461-480``), sorted for :class:`GrowthPlan`. An offer is
    :func:`give_up_only` (no entry) only against the root moves this plan still surely holds:
    a root move leaves only by an answer that names its slot (an offer with an empty slot is
    learned there, ``Pokemon_LevelUpMove``, ``src/pokemon.c:3935-3966``, and removes nothing),
    so a slot an earlier answer of the plan names no longer counts. An HM slot is never an
    answer (:func:`_answers`), so an HM always counts."""
    order = sorted(offers, key=lambda offer: offer[0])
    roots = tuple(options.member.moves)
    answers = _answers(tables, options)
    pruned: dict[tuple[int, frozenset[int]], bool] = {}

    def only_give_up(i: int, named: frozenset[int]) -> bool:
        if (i, named) not in pruned:
            held = [m for s, m in enumerate(roots) if s not in named]
            pruned[i, named] = give_up_only(tables, held, order[i][1])
        return pruned[i, named]

    def walk(i: int, named: frozenset[int], chosen: tuple) -> Iterator[tuple]:
        if i == len(order):
            yield tuple(sorted(chosen, key=lambda entry: entry[0]))
            return
        if only_give_up(i, named):
            yield from walk(i + 1, named, chosen)
            return
        for answer in answers:
            after = named if answer is GIVE_UP_ANSWER else named | {answer}
            yield from walk(i + 1, after, (*chosen, (order[i], answer)))

    yield from walk(0, frozenset(), ())


def _answer_count(tables: Tables, options: MemberOptions, offers: Sequence[Offer]) -> int:
    """``len(list(_answer_lists(...)))`` without listing them (the same walk, counted)."""
    order = sorted(offers, key=lambda offer: offer[0])
    roots = tuple(options.member.moves)
    answers = _answers(tables, options)
    counts: dict[tuple[int, frozenset[int]], int] = {}

    def count(i: int, named: frozenset[int]) -> int:
        if i == len(order):
            return 1
        if (i, named) not in counts:
            held = [m for s, m in enumerate(roots) if s not in named]
            if give_up_only(tables, held, order[i][1]):
                counts[i, named] = count(i + 1, named)
            else:
                counts[i, named] = sum(
                    count(i + 1, named if a is GIVE_UP_ANSWER else named | {a}) for a in answers
                )
        return counts[i, named]

    return count(0, frozenset())


def _answers(tables: Tables, options: MemberOptions) -> list[int | None]:
    """The answers a free offer can take: each slot not holding an HM (``Item_IsHMMove``;
    ``src/battle_sub_menus/battle_party.c:1724-1735`` in battle, ``pokemon_summary_screen/
    main.c:814-820`` between fights), and give up."""
    hm = np.asarray(tables.move_is_hm)
    slots = [s for s, m in enumerate(options.member.moves) if not (m and hm[m])]
    return [*slots, GIVE_UP_ANSWER]


def member_plan_count(tables: Tables, learnsets: Learnsets, options: MemberOptions) -> int:
    """The number of non-dominated plans of one member: per path, every answer list of its
    offers (5 answers per free offer, fewer with an HM slot; a :func:`give_up_only` offer has
    one), summed over the paths (design section 6, "Bound")."""
    return sum(
        _answer_count(tables, options, offers) for _, offers in _paths(tables, learnsets, options)
    )


def member_plans(
    tables: Tables, learnsets: Learnsets, options: MemberOptions
) -> tuple[GrowthPlan, ...]:
    """Every non-dominated plan of one member (:func:`member_plan_count` of them). A given-up-
    only offer has no entry."""
    return tuple(
        GrowthPlan(answers, evolve_at)
        for evolve_at, offers in _paths(tables, learnsets, options)
        for answers in _answer_lists(tables, options, offers)
    )


def _committed_plans(
    tables: Tables,
    learnsets: Learnsets,
    options: MemberOptions,
    current_gap: int,
    later: Callable[[int, GrowthPlan], GrowthPlan] | None,
) -> tuple[GrowthPlan, ...]:
    """The plans the root enumerates above the cap (design section 6, "Bound" rule 3): do not
    evolve at ``current_gap``, or evolve there with every answer to the learn prompts of the
    evolution itself (the new species' offers at the evolution level); every later prompt is the
    caller's best answer, ``later(slot, plan)`` (the single-fight screen), by default the base
    plan's (give up, never evolve later)."""
    complete = later or (lambda _slot, plan: plan)
    plans = [complete(options.slot, BASE_PLAN)]
    for option in options.evolutions:
        if option.gap != current_gap:
            continue
        at_level = [o for o in option.post if o[0] == option.level]
        for answers in _answer_lists(tables, options, at_level):
            plans.append(complete(options.slot, GrowthPlan(answers, option.gap)))
    return tuple(plans)


class GrowthPlanSet(NamedTuple):
    """The plans one setup screens: ``members`` the plans of each member (a setup takes one of
    each), ``total`` the non-dominated setups before the cap, ``cap`` the limit used
    (``UNLIMITED`` = -1: none), ``capped`` whether the cap removed options, ``screened_later``
    whether the later prompts of a capped set took the caller's single-fight screen answer
    (``later``) rather than the base plan's."""

    members: tuple[tuple[GrowthPlan, ...], ...]
    total: int
    cap: int
    capped: bool
    screened_later: bool = False

    @property
    def kept(self) -> int:
        """The number of setups screened (the product over the members)."""
        return int(np.prod([len(p) for p in self.members], dtype=object))

    def setups(self) -> Iterator[tuple[GrowthPlan, ...]]:
        """Every setup: one plan per member."""
        return itertools.product(*self.members)

    def report_line(self) -> str | None:
        """The plain report line when the cap removed options (GT-11 prints it), else ``None``."""
        if not self.capped:
            return None
        rest = (
            "every later learn and evolution prompt takes the best answer by single-fight screen"
            if self.screened_later
            else "every later learn prompt learns into an empty slot, else gives the move up, "
            "and no later evolution is taken (the base plan)"
        )
        return (
            f"growth options removed by --growth-cap {self.cap}: screened {self.kept} of "
            f"{self.total} growth plans; only this fight's evolution and its learn prompts are "
            f"enumerated, and {rest}. Run with --growth-cap -1 to screen all."
        )


def growth_plans(
    tables: Tables,
    learnsets: Learnsets,
    options: Sequence[MemberOptions],
    *,
    growth_cap: int | None = None,
    current_gap: int = 0,
    later: Callable[[int, GrowthPlan], GrowthPlan] | None = None,
) -> GrowthPlanSet:
    """The growth plans of one setup (design section 6, "Bound"): the product over ``options``
    of each member's non-dominated plans. ``growth_cap`` is ``--growth-cap`` (``None`` =
    ``MAX_GROWTH_PLANS``, ``-1`` = none, ``ValueError`` below -1): when the product is at most
    the cap, all are kept. Above it only the current fight's evolution (gap ``current_gap``) and
    the prompts of that evolution are enumerated, every later prompt takes ``later``'s answer.
    The current fight's own prompts are exact Stage B nodes, never in a plan."""
    cap = MAX_GROWTH_PLANS if growth_cap is None else growth_cap
    if cap < UNLIMITED:
        raise ValueError(f"growth cap {cap} is below -1")
    counts = [member_plan_count(tables, learnsets, o) for o in options]
    total = int(np.prod(counts, dtype=object)) if counts else 1
    if cap == UNLIMITED or total <= cap:
        members = tuple(member_plans(tables, learnsets, o) for o in options)
        return GrowthPlanSet(members, total, cap, False)
    members = tuple(_committed_plans(tables, learnsets, o, current_gap, later) for o in options)
    return GrowthPlanSet(members, total, cap, True, later is not None)
