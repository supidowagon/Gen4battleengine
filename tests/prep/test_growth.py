"""Evolution conditions (E3-B3-12): ``prep.growth.evolution_target`` against the rows of
``Learnsets.evolutions`` (synthetic rows where the table has no case), and the map method table.
Rules: ``Pokemon_GetEvolutionTargetSpecies``, ``src/pokemon.c:3528-3790``."""

from __future__ import annotations

from pathlib import Path

import pytest

from nuzline.data import tables as table_data
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.prep.growth import (
    EVO_CLASS_BY_ITEM,
    EVO_CLASS_BY_LEVEL,
    EVO_CLASS_BY_TRADE,
    EVO_CLASS_UNUSED_02,
    HOLD_EFFECT_NO_EVOLVE,
    SPECIES_KADABRA,
    evolution_target,
)

DECOMP = Path("external/pokeplatinum")
SUBJECT, A, B = 132, 1, 2  # a species whose rows the test replaces, and two target species
EVO_NONE = 0


def method_name(learnsets, method_id):
    return learnsets.evolution_methods[method_id]


@pytest.fixture(scope="module")
def manifest(tables):
    return table_data._read_manifest(table_data.DEFAULT)


@pytest.fixture(scope="module")
def method(manifest):
    """Evolution method name -> id."""
    return {name: i for i, name in enumerate(manifest["evolution_methods"])}


def synthetic(manifest, method, rows, maps=None, species=SUBJECT):
    """A ``Learnsets`` whose ``species`` has ``rows`` ``(method name, param, target)``; ``maps``
    (header -> method name) replaces the map table."""
    evolutions = {**manifest["evolutions"], str(species): [[method[n], p, t] for n, p, t in rows]}
    patched = {**manifest, "evolutions": evolutions}
    if maps is not None:
        patched["map_evolution_methods"] = {str(h): method[n] for h, n in maps.items()}
    return Learnsets(patched)


def spec(
    *, species=SUBJECT, level=30, friendship=0, item=0, moves=(0, 0, 0, 0), gender=0, stats=None
):
    return PokemonSpec(
        species=species,
        level=level,
        nature=0,
        ivs=(0,) * 6,
        evs=(0,) * 6,
        moves=moves,
        item=item,
        ability_slot=0,
        gender=gender,
        friendship=friendship,
        stats=stats,
    )


def target(
    tables,
    learnsets,
    mon,
    *,
    party=None,
    cls=EVO_CLASS_BY_LEVEL,
    param=0,
    time=None,
    pid=0,
    beauty=0,
):
    return evolution_target(
        tables, learnsets, mon, party, cls, param, time, personality=pid, beauty=beauty
    )


def test_level_row_and_first_match_order(tables, manifest, method):
    one = synthetic(manifest, method, [("level", 16, A)])
    assert target(tables, one, spec(level=15)) is None
    assert target(tables, one, spec(level=16)) == (A, method["level"])
    # Two rows: the first that matches wins (src/pokemon.c:3721-3723).
    two = synthetic(manifest, method, [("level", 20, A), ("level", 10, B)])
    assert target(tables, two, spec(level=25)) == (A, method["level"])
    assert target(tables, two, spec(level=15)) == (B, method["level"])
    assert target(tables, two, spec(level=9)) is None


@pytest.mark.parametrize(
    ("friendship", "want"), [(0, False), (219, False), (220, True), (255, True)]
)
def test_friendship_threshold(tables, manifest, method, friendship, want):
    ls = synthetic(manifest, method, [("level_happiness", 0, A)])
    got = target(tables, ls, spec(level=1, friendship=friendship))
    assert got == ((A, method["level_happiness"]) if want else None)


@pytest.mark.parametrize(
    ("name", "time", "want"),
    [
        ("level_happiness_day", None, False),
        ("level_happiness_day", "day", True),
        ("level_happiness_day", "night", False),
        ("level_happiness_night", None, False),
        ("level_happiness_night", "day", False),
        ("level_happiness_night", "night", True),
    ],
)
def test_day_night_happiness_only_with_time_set(tables, manifest, method, name, time, want):
    ls = synthetic(manifest, method, [(name, 0, A)])
    got = target(tables, ls, spec(friendship=220), time=time)
    assert got == ((A, method[name]) if want else None)
    assert target(tables, ls, spec(friendship=219), time=time) is None


@pytest.mark.parametrize(
    ("name", "time", "item_ok", "want"),
    [
        ("level_with_held_item_day", None, True, False),
        ("level_with_held_item_day", "day", True, True),
        ("level_with_held_item_day", "day", False, False),
        ("level_with_held_item_day", "night", True, False),
        ("level_with_held_item_night", None, True, False),
        ("level_with_held_item_night", "night", True, True),
        ("level_with_held_item_night", "night", False, False),
        ("level_with_held_item_night", "day", True, False),
    ],
)
def test_held_item_day_night(tables, names, manifest, method, name, time, item_ok, want):
    razor_claw, other = names.item("Razor Claw"), names.item("Oran Berry")
    ls = synthetic(manifest, method, [(name, razor_claw, A)])
    mon = spec(level=1, item=razor_claw if item_ok else other)
    assert target(tables, ls, mon, time=time) == ((A, method[name]) if want else None)


def test_time_must_be_a_setting_value(tables, manifest, method):
    ls = synthetic(manifest, method, [("level", 1, A)])
    with pytest.raises(ValueError, match="time"):
        target(tables, ls, spec(), time="dusk")


@pytest.mark.parametrize(
    ("name", "atk", "dfn", "want"),
    [
        ("level_atk_gt_def", 51, 50, True),
        ("level_atk_gt_def", 50, 50, False),
        ("level_atk_gt_def", 49, 50, False),
        ("level_atk_eq_def", 50, 50, True),
        ("level_atk_eq_def", 51, 50, False),
        ("level_atk_lt_def", 49, 50, True),
        ("level_atk_lt_def", 50, 50, False),
    ],
)
def test_attack_against_defense(tables, manifest, method, name, atk, dfn, want):
    ls = synthetic(manifest, method, [(name, 20, A)])
    stats = (100, atk, dfn, 40, 40, 40)  # ids.Stat order: HP, Atk, Def, SpA, SpD, Spe
    assert target(tables, ls, spec(level=20, stats=stats)) == ((A, method[name]) if want else None)
    assert target(tables, ls, spec(level=19, stats=stats)) is None  # level gate


@pytest.mark.parametrize(
    ("name", "pid", "want"),
    [
        ("level_pid_low", 4 << 16, True),
        ("level_pid_low", 5 << 16, False),
        ("level_pid_low", (14 << 16) | 0xFFFF, True),  # only the upper half counts
        ("level_pid_high", 5 << 16, True),
        ("level_pid_high", 15 << 16, True),  # 15 % 10 = 5
        ("level_pid_high", 9 << 16, True),
        ("level_pid_high", 10 << 16, False),
        ("level_pid_high", 4 << 16, False),
    ],
)
def test_personality_halves(tables, manifest, method, name, pid, want):
    ls = synthetic(manifest, method, [(name, 7, A)])
    assert target(tables, ls, spec(level=7), pid=pid) == ((A, method[name]) if want else None)
    assert target(tables, ls, spec(level=6), pid=pid) is None


def test_ninjask_and_shedinja(tables, names, learnsets, method):
    nincada = names.species("Nincada")
    rows = learnsets.evolutions(nincada)
    assert [(method_name(learnsets, m), p) for m, p, _ in rows] == [
        ("level_ninjask", 20),
        ("level_shedinja", 20),
    ]
    mon = spec(species=nincada, level=19)
    assert target(tables, learnsets, mon) is None
    # The Ninjask row matches first and breaks the loop (:3721-3723); Shedinja is GT-13's.
    assert target(tables, learnsets, spec(species=nincada, level=20)) == (
        names.species("Ninjask"),
        method["level_ninjask"],
    )


def test_shedinja_row_alone_matches_nothing(tables, manifest, method):
    ls = synthetic(manifest, method, [("level_shedinja", 20, A)])
    assert target(tables, ls, spec(level=50)) is None  # src/pokemon.c:3644-3646


@pytest.mark.parametrize(("beauty", "want"), [(169, False), (170, True), (255, True)])
def test_beauty(tables, manifest, method, beauty, want):
    ls = synthetic(manifest, method, [("level_beauty", 170, A)])
    got = target(tables, ls, spec(level=1), beauty=beauty)
    assert got == ((A, method["level_beauty"]) if want else None)


def test_know_move(tables, names, manifest, method):
    ancient_power = names.move("AncientPower")
    ls = synthetic(manifest, method, [("level_know_move", ancient_power, A)])
    assert target(tables, ls, spec(moves=(1, 2, 3, 4))) is None
    got = target(tables, ls, spec(moves=(1, 2, 3, ancient_power)))
    assert got == (A, method["level_know_move"])


def test_species_in_party(tables, names, manifest, method):
    ls = synthetic(manifest, method, [("level_species_in_party", names.species("Remoraid"), A)])
    mon = spec(level=1)
    assert target(tables, ls, mon, party=None) is None  # party == NULL (:3676)
    assert target(tables, ls, mon, party=[mon, spec(species=names.species("Bidoof"))]) is None
    party = [mon, spec(species=names.species("Remoraid"))]
    assert target(tables, ls, mon, party=party) == (A, method["level_species_in_party"])


@pytest.mark.parametrize(
    ("name", "gender", "level", "want"),
    [
        ("level_male", 0, 30, True),
        ("level_male", 1, 30, False),
        ("level_male", 2, 30, False),
        ("level_male", 0, 29, False),
        ("level_female", 1, 30, True),
        ("level_female", 0, 30, False),
        ("level_female", 2, 30, False),
        ("level_female", 1, 29, False),
    ],
)
def test_gender_level_rows(tables, manifest, method, name, gender, level, want):
    ls = synthetic(manifest, method, [(name, 30, A)])
    got = target(tables, ls, spec(level=level, gender=gender))
    assert got == ((A, method[name]) if want else None)


def test_map_rows_match_only_the_maps_method(tables, manifest, method):
    names3 = ("level_magnetic_field", "level_moss_rock", "level_ice_rock")
    for name in names3:
        ls = synthetic(manifest, method, [(name, 0, A)])
        for other in names3:
            got = target(tables, ls, spec(level=1), param=method[other])
            assert got == ((A, method[name]) if other == name else None)
        assert target(tables, ls, spec(level=1), param=EVO_NONE) is None


def test_map_method_table_lists_only_its_headers(tables, manifest, method):
    ls = synthetic(
        manifest,
        method,
        [("level_magnetic_field", 0, A)],
        maps={10: "level_magnetic_field", 20: "level_moss_rock"},
    )
    assert ls.map_evolution_method(10) == method["level_magnetic_field"]
    assert ls.map_evolution_method(20) == method["level_moss_rock"]
    assert ls.map_evolution_method(11) == EVO_NONE  # MapHeader_GetMapEvolutionMethod: EVO_NONE
    for header, want in ((10, A), (20, None), (11, None)):
        param = ls.map_evolution_method(header)
        got = target(tables, ls, spec(level=1), param=param)
        assert got == ((want, method["level_magnetic_field"]) if want else None)


def test_vanilla_map_table_equals_the_decomp(learnsets, manifest):
    from tools.extract_decomp import _map_header_ids, extract_map_evolution_methods

    if not DECOMP.exists():
        pytest.skip("decomp not cloned")
    want = extract_map_evolution_methods(DECOMP)
    assert manifest["map_evolution_methods"] == want
    assert len(want) == 21  # 0x54 bytes of pairs
    ids = _map_header_ids(DECOMP)
    by_name = manifest["evolution_methods"]
    assert by_name[learnsets.map_evolution_method(ids["MAP_HEADER_ROUTE_217"])] == "level_ice_rock"
    assert by_name[learnsets.map_evolution_method(ids["MAP_HEADER_ETERNA_FOREST"])] == (
        "level_moss_rock"
    )
    assert by_name[learnsets.map_evolution_method(ids["MAP_HEADER_MT_CORONET_1F_SOUTH"])] == (
        "level_magnetic_field"
    )
    assert learnsets.map_evolution_method(ids["MAP_HEADER_EVERYWHERE"]) == EVO_NONE


def test_everstone_blocks_level_not_stone(tables, names, manifest, method):
    everstone, stone = names.item("Everstone"), names.item("Moon Stone")
    assert int(tables.item_hold_effect[everstone]) == HOLD_EFFECT_NO_EVOLVE
    ls = synthetic(manifest, method, [("level", 10, A), ("use_item", stone, B)])
    assert target(tables, ls, spec(level=10)) == (A, method["level"])
    held = spec(level=10, item=everstone)
    assert target(tables, ls, held) is None  # src/pokemon.c:3543-3547
    # Only the item class passes the Everstone (:3545). Class 2 shares the item case
    # (:3749-3750) but the guard before the switch stops it.
    assert target(tables, ls, held, cls=EVO_CLASS_BY_ITEM, param=stone) == (B, EVO_NONE)
    assert target(tables, ls, held, cls=EVO_CLASS_UNUSED_02, param=stone) is None
    bare = spec(level=10)
    assert target(tables, ls, bare, cls=EVO_CLASS_UNUSED_02, param=stone) == (B, EVO_NONE)


def test_everstone_does_not_stop_kadabra(tables, names, manifest, method):
    assert SPECIES_KADABRA == names.species("Kadabra")
    everstone = names.item("Everstone")
    ls = synthetic(manifest, method, [("level", 10, A)], species=SPECIES_KADABRA)
    held = spec(species=SPECIES_KADABRA, level=10, item=everstone)
    assert target(tables, ls, held) == (A, method["level"])
    other = synthetic(manifest, method, [("level", 10, A)])
    assert target(tables, other, spec(level=10, item=everstone)) is None


def test_trade_class_is_never_offered(tables, names, learnsets, manifest, method):
    # Kadabra's only evolution is a trade (generated res/pokemon/kadabra/data.json).
    kadabra = names.species("Kadabra")
    assert [method_name(learnsets, m) for m, _, _ in learnsets.evolutions(kadabra)] == ["trade"]
    mon = spec(species=kadabra, level=50, friendship=255)
    for cls in (EVO_CLASS_BY_TRADE, EVO_CLASS_BY_LEVEL, EVO_CLASS_BY_ITEM):
        assert target(tables, learnsets, mon, cls=cls) is None
    ls = synthetic(manifest, method, [("trade", 0, A), ("trade_with_held_item", 5, B)])
    assert target(tables, ls, spec(), cls=EVO_CLASS_BY_TRADE) is None


def test_evolution_class_must_exist(tables, learnsets):
    with pytest.raises(ValueError, match="evolution class 4"):
        target(tables, learnsets, spec(), cls=4)


def test_item_class_rows(tables, names, manifest, method):
    stone, other = names.item("Dawn Stone"), names.item("Fire Stone")
    ls = synthetic(
        manifest,
        method,
        [
            ("use_item_male", stone, A),
            ("use_item_female", stone, B),
            ("use_item", other, 3),
        ],
    )
    assert target(tables, ls, spec(gender=0), cls=EVO_CLASS_BY_ITEM, param=stone) == (A, EVO_NONE)
    assert target(tables, ls, spec(gender=1), cls=EVO_CLASS_BY_ITEM, param=stone) == (B, EVO_NONE)
    assert target(tables, ls, spec(gender=2), cls=EVO_CLASS_BY_ITEM, param=stone) is None
    assert target(tables, ls, spec(gender=2), cls=EVO_CLASS_BY_ITEM, param=other) == (3, EVO_NONE)
    assert (
        target(tables, ls, spec(gender=0), cls=EVO_CLASS_BY_ITEM, param=names.item("Leaf Stone"))
        is None
    )
    # The level class ignores item rows, and the item class ignores level rows.
    assert target(tables, ls, spec(gender=0), param=stone) is None
    only_level = synthetic(manifest, method, [("level", 1, A)])
    assert target(tables, only_level, spec(), cls=EVO_CLASS_BY_ITEM, param=stone) is None


def test_item_class_first_match_wins(tables, names, manifest, method):
    stone = names.item("Moon Stone")
    ls = synthetic(manifest, method, [("use_item", stone, A), ("use_item", stone, B)])
    assert target(tables, ls, spec(), cls=EVO_CLASS_BY_ITEM, param=stone) == (A, EVO_NONE)


# ---------------------------------------------------------------- the vanilla rows


def test_eevee_rows_in_game_order(tables, names, learnsets, method):
    eevee = names.species("Eevee")
    got = [(method_name(learnsets, m), t) for m, _, t in learnsets.evolutions(eevee)]
    assert got == [
        ("level_moss_rock", names.species("Leafeon")),
        ("level_ice_rock", names.species("Glaceon")),
        ("use_item", names.species("Jolteon")),
        ("use_item", names.species("Vaporeon")),
        ("use_item", names.species("Flareon")),
        ("level_happiness_day", names.species("Espeon")),
        ("level_happiness_night", names.species("Umbreon")),
    ]
    mon = spec(species=eevee, level=20, friendship=220)
    assert target(tables, learnsets, mon, time="day") == (
        names.species("Espeon"),
        method["level_happiness_day"],
    )
    assert target(tables, learnsets, mon, time="night") == (
        names.species("Umbreon"),
        method["level_happiness_night"],
    )
    assert target(tables, learnsets, mon) is None  # time unset: neither row is offered
    # A listed map's method row comes first in the table, so it wins over happiness.
    moss = target(tables, learnsets, mon, time="day", param=method["level_moss_rock"])
    assert moss == (names.species("Leafeon"), method["level_moss_rock"])
    stone = names.item("Thunderstone")
    assert target(tables, learnsets, mon, cls=EVO_CLASS_BY_ITEM, param=stone) == (
        names.species("Jolteon"),
        EVO_NONE,
    )


def test_gender_stone_rows_in_the_vanilla_table(tables, names, learnsets):
    dawn = names.item("Dawn Stone")
    kirlia, snorunt = names.species("Kirlia"), names.species("Snorunt")
    male, female = spec(species=kirlia, level=20, gender=0), spec(species=kirlia, gender=1)
    assert target(tables, learnsets, male, cls=EVO_CLASS_BY_ITEM, param=dawn) == (
        names.species("Gallade"),
        EVO_NONE,
    )
    assert target(tables, learnsets, female, cls=EVO_CLASS_BY_ITEM, param=dawn) is None
    froslass = target(
        tables, learnsets, spec(species=snorunt, gender=1), cls=EVO_CLASS_BY_ITEM, param=dawn
    )
    assert froslass == (names.species("Froslass"), EVO_NONE)
    glalie = target(tables, learnsets, spec(species=snorunt, level=42, gender=1))
    assert glalie == (names.species("Glalie"), learnsets.evolution_methods.index("level"))


def test_wurmple_personality_rows(tables, names, learnsets, method):
    wurmple = names.species("Wurmple")
    mon = spec(species=wurmple, level=7)
    assert target(tables, learnsets, mon, pid=3 << 16) == (
        names.species("Silcoon"),
        method["level_pid_low"],
    )
    assert target(tables, learnsets, mon, pid=8 << 16) == (
        names.species("Cascoon"),
        method["level_pid_high"],
    )


def test_pichu_friendship_row(tables, names, learnsets, method):
    pichu = names.species("Pichu")
    assert target(tables, learnsets, spec(species=pichu, friendship=219)) is None
    assert target(tables, learnsets, spec(species=pichu, friendship=220)) == (
        names.species("Pikachu"),
        method["level_happiness"],
    )


def test_evolution_rows_equal_the_decomp(names, learnsets, manifest):
    if not DECOMP.exists():
        pytest.skip("decomp not cloned")
    from tools.extract_decomp import extract_evolutions

    want = extract_evolutions(DECOMP)
    for species in ("Eevee", "Wurmple", "Nincada", "Kadabra", "Pichu", "Snorunt", "Kirlia"):
        sid = names.species(species)
        assert learnsets.evolutions(sid) == tuple(tuple(r) for r in want[str(sid)]), species
    assert manifest["evolution_methods"][EVO_NONE] == "none"
