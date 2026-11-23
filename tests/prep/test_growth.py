"""Evolution conditions (E3-B3-12): ``prep.growth.evolution_target`` against the rows of
``Learnsets.evolutions`` (synthetic rows where the table has no case), and the map method table.
Rules: ``Pokemon_GetEvolutionTargetSpecies``, ``src/pokemon.c:3528-3790``.

Evolution applied (E3-B3-13): ``prep.growth.evolve`` against the learn loop and
``Evolution_ProcessEvolutionEffects`` of ``src/evolution.c:433-480``, ``:708-790``."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data import tables as table_data
from nuzline.data.gauntlet import Origin, Resolved, Settings, Swap
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.engine.ids import EffectId, ItemParam
from nuzline.engine.state import spec_stats
from nuzline.prep.gauntlet import carry_from_party
from nuzline.prep.growth import (
    BASE_PLAN,
    BATTLE_RESULT_CAPTURED_MON,
    BATTLE_RESULT_PLAYER_FLED,
    BATTLE_RESULT_WIN,
    EVO_CLASS_BY_ITEM,
    EVO_CLASS_BY_LEVEL,
    EVO_CLASS_BY_TRADE,
    EVO_CLASS_UNUSED_02,
    HOLD_EFFECT_NO_EVOLVE,
    ITEM_POKE_BALL,
    MAX_GROWTH_PLANS,
    MAX_PARTY_SIZE,
    SPECIES_KADABRA,
    EvolutionGap,
    GrowthPlan,
    evolution_prompt,
    evolution_target,
    evolve,
    evolve_dominates,
    give_up_only,
    growth_plans,
    learn_offers,
    member_options,
    member_plan_count,
    member_plans,
    post_battle_prompts,
    rare_candy,
    resolve_post_battle,
    use_stone,
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


# --- evolution applied (E3-B3-13) ---------------------------------------------------------

KADABRA, ALAKAZAM, NINCADA, NINJASK, SHEDINJA = 64, 65, 290, 291, 292
TACKLE, GROWL, CONFUSION, MIRACLE_EYE = 33, 45, 93, 357  # Alakazam learns Miracle Eye at 22
HELD_ITEM = 17


def mon(**fields):
    base = spec(species=KADABRA, level=22, moves=(TACKLE, GROWL, CONFUSION, 0))
    return replace(base, **fields)


def shedinja_setting(value):
    return Settings(
        values={"shedinja": Resolved(value, Origin.DEFAULT, "test")},
        gaps={},
        locked=(),
        locked_source="",
    )


def run_evolve(tables, learnsets, member, **kw):
    args = {
        "target": ALAKAZAM,
        "method": 0,
        "bag": {},
        "party": [member],
        "settings": shedinja_setting("forbidden"),
        "personality": 0,
        "learn_answers": None,
    }
    args.update(kw)
    return evolve(
        tables,
        learnsets,
        member,
        args["target"],
        args["method"],
        args["bag"],
        args["party"],
        args["settings"],
        personality=args["personality"],
        learn_answers=args["learn_answers"],
    )


def test_poke_ball_and_party_size_constants_equal_the_decomp():
    if not DECOMP.exists():
        pytest.skip("decomp not cloned")
    items = (DECOMP / "generated/items.txt").read_text(encoding="utf-8").split()
    assert items[ITEM_POKE_BALL] == "ITEM_POKE_BALL"
    header = (DECOMP / "include/constants/pokemon.h").read_text(encoding="utf-8")
    assert f"#define MAX_PARTY_SIZE  {MAX_PARTY_SIZE}" in header


def test_evolving_at_partial_hp_raises_hp_by_the_max_hp_gain(tables, learnsets):
    member = mon(current_hp=30)
    old_max = int(spec_stats(tables, member)[0])
    evolved, extra, _ = run_evolve(tables, learnsets, member)
    new_stats = spec_stats(tables, evolved)
    assert extra is None and evolved.species == ALAKAZAM and evolved.level == 22
    assert evolved.stats is None  # recomputed from Alakazam's base stats
    assert int(new_stats[0]) > old_max  # Alakazam's base HP is above Kadabra's
    assert evolved.current_hp == 30 + int(new_stats[0]) - old_max  # src/pokemon.c:640-651
    fresh = spec_stats(tables, replace(member, species=ALAKAZAM))
    assert list(new_stats) == list(fresh)


def test_stored_stats_are_replaced_by_the_new_species_stats(tables, learnsets):
    member = mon(stats=(100, 11, 12, 13, 14, 15))
    evolved, _, _ = run_evolve(tables, learnsets, member)
    assert list(spec_stats(tables, evolved)) == list(
        spec_stats(tables, replace(member, species=ALAKAZAM, stats=None))
    )


def test_full_hp_stays_full_and_dead_member_does_not_evolve(tables, learnsets):
    full, _, _ = run_evolve(tables, learnsets, mon())
    assert full.current_hp is None and full.species == ALAKAZAM
    dead = mon(current_hp=0)
    out, extra, bag = run_evolve(tables, learnsets, dead, bag={ITEM_POKE_BALL: 2})
    assert out == dead and extra is None and bag == {ITEM_POKE_BALL: 2}


@pytest.mark.parametrize(("pid", "slot"), [(0, 0), (1, 1), (2, 0), (0xFFFF0003, 1)])
def test_ability_slot_is_personality_bit_zero(tables, learnsets, pid, slot):
    row = np.asarray(tables.species_abilities_decomp)[ALAKAZAM]
    assert int(row[1]) != 0  # two abilities: bit 0 chooses (src/pokemon.c:4746-4763)
    evolved, _, _ = run_evolve(tables, learnsets, mon(ability_slot=1 - slot), personality=pid)
    assert evolved.ability_slot == slot


@pytest.mark.parametrize(
    ("ratio", "pid", "gender"), [(254, 0, 1), (0, 0, 0), (255, 0, 2), (127, 200, 0)]
)
def test_the_gender_is_the_new_species_for_the_personality(tables, learnsets, ratio, pid, gender):
    # BoxPokemon_GetGender recomputes it from species and personality (src/pokemon.c:2696-2705)
    t = edited(tables, species_gender={ALAKAZAM: ratio})
    evolved, _, _ = run_evolve(t, learnsets, mon(gender=1 - gender % 2), personality=pid)
    assert evolved.gender == gender


def test_single_ability_species_ignores_the_personality_bit(tables, learnsets):
    assert int(np.asarray(tables.species_abilities_decomp)[SHEDINJA][1]) == 0
    evolved, _, _ = run_evolve(tables, learnsets, mon(), target=SHEDINJA, personality=1)
    assert evolved.ability_slot == 0


def test_learn_into_an_empty_slot_keeps_the_slots_pp_ups(tables, learnsets):
    assert (22, MIRACLE_EYE) in learnsets.level_up(ALAKAZAM)
    member = mon(pp=(1, 2, 3, 4), pp_ups=(0, 0, 0, 2))
    evolved, _, _ = run_evolve(tables, learnsets, member)
    assert evolved.moves == (TACKLE, GROWL, CONFUSION, MIRACLE_EYE)
    base = int(np.asarray(tables.move_pp)[MIRACLE_EYE])
    assert evolved.pp == (1, 2, 3, base + base * 20 * 2 // 100)  # src/pokemon.c:3925-3933
    assert evolved.pp_ups == (0, 0, 0, 2)


def test_a_known_move_is_skipped_and_other_levels_offer_nothing(tables, learnsets):
    known = mon(moves=(MIRACLE_EYE, TACKLE, GROWL, CONFUSION))
    assert run_evolve(tables, learnsets, known)[0].moves == known.moves
    other = mon(level=23)
    assert (23, MIRACLE_EYE) not in learnsets.level_up(ALAKAZAM)
    assert run_evolve(tables, learnsets, other)[0].moves == other.moves


def test_full_slots_replace_with_pp_ups_reset_or_give_up(tables, learnsets):
    member = mon(moves=(TACKLE, GROWL, CONFUSION, 98), pp=(9, 9, 9, 9), pp_ups=(1, 3, 2, 0))
    swapped, _, _ = run_evolve(tables, learnsets, member, learn_answers={MIRACLE_EYE: 2})
    assert swapped.moves == (TACKLE, GROWL, MIRACLE_EYE, 98)
    assert swapped.pp_ups == (1, 3, 0, 0)  # the replaced slot's PP Ups are 0 (:631-633)
    assert swapped.pp == (9, 9, int(np.asarray(tables.move_pp)[MIRACLE_EYE]), 9)
    for answers in (None, {}, {MIRACLE_EYE: None}):
        kept, _, _ = run_evolve(tables, learnsets, member, learn_answers=answers)
        assert kept.moves == member.moves and kept.pp_ups == member.pp_ups
        assert kept.pp == member.pp


def test_an_hm_slot_cannot_be_replaced(tables, learnsets):
    hms = np.flatnonzero(np.asarray(tables.move_is_hm))
    assert len(hms) > 0
    hm = int(hms[0])
    member = mon(moves=(TACKLE, hm, CONFUSION, GROWL))
    with pytest.raises(ValueError, match="HM"):
        run_evolve(tables, learnsets, member, learn_answers={MIRACLE_EYE: 1})
    ok, _, _ = run_evolve(tables, learnsets, member, learn_answers={MIRACLE_EYE: 0})
    assert ok.moves == (MIRACLE_EYE, hm, CONFUSION, GROWL)
    with pytest.raises(ValueError, match="slot"):
        run_evolve(tables, learnsets, member, learn_answers={MIRACLE_EYE: 4})


@pytest.mark.parametrize(
    ("name", "uses_up"),
    [
        ("level_with_held_item_day", True),
        ("level_with_held_item_night", True),
        ("trade_with_held_item", True),
        ("level", False),
        ("use_item", False),
    ],
)
def test_held_item_methods_use_up_the_item(tables, learnsets, method, name, uses_up):
    evolved, extra, _ = run_evolve(tables, learnsets, mon(item=HELD_ITEM), method=method[name])
    assert extra is None
    assert evolved.item == (0 if uses_up else HELD_ITEM)  # src/evolution.c:782-787


def nincada(**fields):
    base = spec(species=NINCADA, level=20, moves=(TACKLE, GROWL, 0, 0), gender=1)
    return replace(base, item=HELD_ITEM, status=4, status_counter=2, **fields)


def ninjask_run(tables, learnsets, method, setting, bag, party_size=3, name="level_ninjask", **kw):
    member = nincada(**kw)
    return member, run_evolve(
        tables,
        learnsets,
        member,
        target=NINJASK,
        method=method[name],
        bag=bag,
        party=[member] * party_size,
        settings=shedinja_setting(setting),
        personality=1,
    )


def test_shedinja_is_added_with_ball_free_slot_and_allowed(tables, learnsets, method):
    bag = {ITEM_POKE_BALL: 3, 5: 1}
    member, (ninjask, extra, out) = ninjask_run(tables, learnsets, method, "allowed", bag)
    assert ninjask.species == NINJASK and ninjask.item == HELD_ITEM  # Ninjask keeps its item
    assert out == {ITEM_POKE_BALL: 2, 5: 1} and bag == {ITEM_POKE_BALL: 3, 5: 1}  # :779
    assert extra is not None and extra.species == SHEDINJA  # :733-734
    assert extra.item == 0  # :740
    assert extra.status == 0 and extra.status_counter == 0  # :757
    assert extra.gender == 2  # genderless, :769-770
    assert extra.ability_slot == 0  # Shedinja has one ability, :767
    assert int(spec_stats(tables, extra)[0]) == 1  # :640-643
    assert extra.moves == ninjask.moves  # :731 copy
    assert (extra.ivs, extra.evs, extra.nature, extra.friendship) == (
        member.ivs,
        member.evs,
        member.nature,
        member.friendship,
    )
    assert extra is not ninjask


def test_shedinja_hp_is_one_for_a_wounded_copy(tables, learnsets, method):
    _, (_, extra, _) = ninjask_run(
        tables, learnsets, method, "allowed", {ITEM_POKE_BALL: 1}, current_hp=7
    )
    assert extra.current_hp == 1


@pytest.mark.parametrize(
    ("setting", "bag", "party_size"),
    [
        ("allowed", {}, 3),
        ("allowed", {ITEM_POKE_BALL: 0}, 3),
        ("allowed", {ITEM_POKE_BALL: 4}, MAX_PARTY_SIZE),
        ("forbidden", {ITEM_POKE_BALL: 4}, 3),
    ],
)
def test_no_shedinja_without_ball_free_slot_or_allowed(
    tables, learnsets, method, setting, bag, party_size
):
    _, (ninjask, extra, out) = ninjask_run(tables, learnsets, method, setting, bag, party_size)
    assert extra is None and out == bag and ninjask.species == NINJASK


@pytest.mark.parametrize(
    ("name", "added"),
    [
        ("level", False),
        ("level_with_held_item_day", False),
        ("use_item", False),
        ("level_ninjask", True),
        ("level_shedinja", True),
    ],
)
def test_only_the_ninjask_and_shedinja_methods_add_a_shedinja(
    tables, learnsets, method, name, added
):
    _, (_, extra, out) = ninjask_run(
        tables, learnsets, method, "allowed", {ITEM_POKE_BALL: 1}, name=name
    )
    assert (extra is not None) == added
    assert out == {ITEM_POKE_BALL: 0 if added else 1}


# --- GT-14 evolution prompts, Rare Candy, stones (E3-B3-14) ------------------------------------

OLD = SUBJECT  # the evolving species of the prompt and plan tests; A is its target
MASK1 = [True] + [False] * 5
STONE = 82  # an item id; the synthetic rows name it, so the table does not matter


def sp(species=OLD, level=30, **fields):
    return replace(spec(species=species, level=level), **fields)


def carry_settings():
    def resolved(value):
        return Resolved(value, Origin.DEFAULT, "test")

    return Settings(
        values={
            "shedinja": resolved("forbidden"),
            "swap": resolved(Swap(1, 2)),
            "bag_uses": resolved(3),
        },
        gaps={},
        locked=(),
        locked_source="",
    )


def carry_of(tables, party, bag=None):
    return carry_from_party(
        tables, party, bag or {}, carry_settings(), personality=[2 * i for i in range(len(party))]
    )


def rows_learnsets(manifest, method, rows_by_species):
    """Learnsets whose species in ``rows_by_species`` have the evolution rows
    ``(method name, param, target)``."""
    evolutions = {
        **manifest["evolutions"],
        **{str(s): [[method[n], p, t] for n, p, t in rows] for s, rows in rows_by_species.items()},
    }
    return Learnsets({**manifest, "evolutions": evolutions})


def level_evolver(manifest, method, level=10):
    """Learnsets where OLD evolves into A at ``level`` by the plain level method."""
    return rows_learnsets(manifest, method, {OLD: [("level", level, A)]})


@pytest.mark.parametrize(
    ("result", "offered"),
    [
        (BATTLE_RESULT_WIN, True),
        (BATTLE_RESULT_CAPTURED_MON, True),
        (BATTLE_RESULT_PLAYER_FLED, True),
        (2, False),  # BATTLE_RESULT_LOSE, include/constants/battle.h:73
        (3, False),  # BATTLE_RESULT_DRAW, :78
        (6, False),  # BATTLE_RESULT_ENEMY_FLED, :80
        (0, False),
    ],
)
def test_prompts_only_on_win_capture_or_flee(tables, manifest, method, result, offered):
    ls = level_evolver(manifest, method)
    carry = carry_of(tables, [sp()])
    got = post_battle_prompts(tables, ls, carry, MASK1, result, 0, None)
    assert [p.slot for p in got] == ([0] if offered else [])


def test_only_levelled_members_are_offered_and_a_fainted_one_too(tables, manifest, method):
    ls = level_evolver(manifest, method)
    carry = carry_of(tables, [sp(), sp(current_hp=0), sp(), sp(level=5)])
    mask = [False, True, True, True, False, False]
    got = post_battle_prompts(tables, ls, carry, mask, BATTLE_RESULT_WIN, 0, None)
    # slot 3 is below the evolution level, slot 0 did not level up
    assert [(p.slot, p.dead) for p in got] == [(1, True), (2, False)]
    assert all(p.species == OLD and p.target == A and p.cancellable for p in got)
    assert all(p.evo_class == EVO_CLASS_BY_LEVEL for p in got)


def test_the_mask_is_a_party_slot_mask(tables, manifest, method):
    ls = level_evolver(manifest, method)
    carry = carry_of(tables, [sp()])
    with pytest.raises(ValueError, match="mask"):
        post_battle_prompts(tables, ls, carry, [True], BATTLE_RESULT_WIN, 0, None)


def test_cancel_changes_nothing_and_offers_again_only_after_the_next_level_up(
    tables, manifest, method
):
    ls = level_evolver(manifest, method)
    carry = carry_of(tables, [sp()])
    asked = []

    def cancel(prompt):
        asked.append(prompt)

    def run(c, mask):
        args = (tables, ls, c, mask, BATTLE_RESULT_WIN, 0, None, carry_settings(), cancel)
        return resolve_post_battle(*args)

    after, offered = run(carry, MASK1)
    assert len(asked) == len(offered) == 1 and int(after.species[0]) == OLD
    assert all(np.array_equal(getattr(after, f), getattr(carry, f)) for f in carry._fields)
    # the next battle with no level-up offers nothing; the one with a level-up offers again
    _, none = run(after, [False] * 6)
    assert none == []
    _, second = run(after, MASK1)
    assert len(second) == 1


def test_an_accepted_prompt_evolves_and_a_dead_member_is_never_asked(tables, manifest, method):
    ls = level_evolver(manifest, method)
    carry = carry_of(tables, [sp(), sp(current_hp=0)])
    mask = [True, True, False, False, False, False]
    seen = []

    def accept(prompt):
        seen.append(prompt.slot)
        return {}

    after, offered = resolve_post_battle(
        tables, ls, carry, mask, BATTLE_RESULT_WIN, 0, None, carry_settings(), accept
    )
    assert [p.slot for p in offered] == [0, 1] and seen == [0]  # the dead one is offered only
    assert int(after.species[0]) == A and int(after.species[1]) == OLD
    assert int(after.hp[1]) == 0


def test_a_later_slot_is_checked_against_the_party_after_an_evolution(tables, manifest, method):
    # slot 0 evolves into A; slot 1 (B) evolves only with an A in the party (src/pokemon.c:3676)
    ls = rows_learnsets(
        manifest,
        method,
        {OLD: [("level", 10, A)], B: [("level_species_in_party", A, 3)]},
    )
    carry = carry_of(tables, [sp(), sp(species=B)])
    mask = [True, True, False, False, False, False]
    _, offered = resolve_post_battle(
        tables, ls, carry, mask, BATTLE_RESULT_WIN, 0, None, carry_settings(), lambda p: None
    )
    assert [p.slot for p in offered] == [0]  # nobody evolved, no A in the party yet
    _, offered = resolve_post_battle(
        tables, ls, carry, mask, BATTLE_RESULT_WIN, 0, None, carry_settings(), lambda p: {}
    )
    assert [p.slot for p in offered] == [0, 1]


def test_a_stone_prompt_cannot_be_cancelled_and_a_level_prompt_can(tables, manifest, method):
    ls = rows_learnsets(manifest, method, {OLD: [("use_item", STONE, A), ("level", 10, B)]})
    member = sp()
    stone = evolution_prompt(
        tables, ls, 0, member, None, EVO_CLASS_BY_ITEM, STONE, None, personality=0
    )
    level = evolution_prompt(
        tables, ls, 0, member, None, EVO_CLASS_BY_LEVEL, 0, None, personality=0
    )
    assert (stone.target, stone.cancellable, stone.method) == (A, False, EVO_NONE)
    assert (level.target, level.cancellable) == (B, True)
    other = evolution_prompt(
        tables, ls, 0, member, None, EVO_CLASS_BY_ITEM, STONE + 1, None, personality=0
    )
    assert other is None


def test_use_stone_consumes_it_and_evolves(tables, manifest, method):
    ls = rows_learnsets(manifest, method, {OLD: [("use_item", STONE, A)]})
    member = sp(moves=(TACKLE, GROWL, 0, 0))
    bag = {STONE: 2, 5: 1}
    evolved, extra, out = use_stone(
        tables, ls, member, STONE, bag, [member], shedinja_setting("forbidden"), personality=0
    )
    assert evolved.species == A and extra is None
    assert out == {STONE: 1, 5: 1} and bag == {STONE: 2, 5: 1}  # main.c:2699-2701


def test_use_stone_refusals(tables, manifest, method):
    ls = rows_learnsets(manifest, method, {OLD: [("use_item", STONE, A)]})
    setting = shedinja_setting("forbidden")

    def go(member, stone=STONE, bag=None):
        have = {STONE: 1} if bag is None else bag
        return use_stone(tables, ls, member, stone, have, [member], setting, personality=0)

    with pytest.raises(ValueError, match="dead"):
        go(sp(current_hp=0))
    with pytest.raises(ValueError, match="holds no item"):
        go(sp(), bag={})
    with pytest.raises(ValueError, match="does not evolve"):
        go(sp(), stone=STONE + 1, bag={STONE + 1: 1})


# Rare Candy -----------------------------------------------------------------------------------


def candy_params(tables, names):
    item = names.item("Rare Candy")
    return item, np.asarray(tables.item_party_param)[item]


def tier_gain(params, friendship):
    """The item's friendship change by the tier of ``friendship`` (the table's params)."""
    if friendship < 100:
        give, change = ItemParam.GIVE_FRIENDSHIP_LOW, ItemParam.FRIENDSHIP_LOW
    elif friendship < 200:
        give, change = ItemParam.GIVE_FRIENDSHIP_MED, ItemParam.FRIENDSHIP_MED
    else:
        give, change = ItemParam.GIVE_FRIENDSHIP_HIGH, ItemParam.FRIENDSHIP_HIGH
    return int(params[change]) if params[give] else 0


@pytest.mark.parametrize("friendship", [0, 99, 100, 150, 199, 200, 250])
def test_rare_candy_friendship_follows_the_params_by_tier(tables, names, learnsets, friendship):
    item, params = candy_params(tables, names)
    out = rare_candy(tables, learnsets, sp(friendship=friendship), item)
    assert out.friendship == min(255, friendship + tier_gain(params, friendship))
    assert tier_gain(params, friendship) > 0  # the vanilla candy gives friendship in every tier


def test_rare_candy_soothe_bell_comes_before_the_plus_ones(tables, names, learnsets):
    item, params = candy_params(tables, names)
    soothe = names.item("Soothe Bell")
    assert int(np.asarray(tables.item_hold_effect)[soothe]) == 53
    base = tier_gain(params, 50)
    member = sp(friendship=50, item=soothe, friend_bonus=1, egg_location=7)
    out = rare_candy(tables, learnsets, member, item, location=7)
    item_path = base * 150 // 100 + 1 + 1  # src/item_use_pokemon.c:575-586
    battle_path = (base + 1 + 1) * 150 // 100  # src/pokemon.c:2664-2678
    assert item_path != battle_path
    assert out.friendship == 50 + item_path
    # no location given: no egg-location bonus
    no_egg = rare_candy(tables, learnsets, member, item)
    assert no_egg.friendship == 50 + base * 150 // 100 + 1


def test_rare_candy_friendship_clamps_and_stops_at_255(tables, names, learnsets):
    item, _ = candy_params(tables, names)
    assert rare_candy(tables, learnsets, sp(friendship=254), item).friendship == 255
    assert rare_candy(tables, learnsets, sp(friendship=255), item).friendship == 255


def test_rare_candy_level_exp_stats_and_hp(tables, names, learnsets):
    item, _ = candy_params(tables, names)
    member = sp(current_hp=20)
    old_max = int(spec_stats(tables, member)[0])
    out = rare_candy(tables, learnsets, member, item)
    rate = int(np.asarray(tables.species_exp_rate)[member.species])
    assert out.level == 31 and out.exp == int(np.asarray(tables.exp_table)[rate][31])
    new_max = int(spec_stats(tables, out)[0])
    assert out.current_hp == 20 + new_max - old_max  # src/pokemon.c:640-651
    full = rare_candy(tables, learnsets, sp(), item)
    assert full.current_hp is None and full.level == 31


def test_rare_candy_revives_a_fainted_member_with_the_hp_gain(tables, names, learnsets):
    item, _ = candy_params(tables, names)
    member = sp(current_hp=0)
    old_max = int(spec_stats(tables, member)[0])
    out = rare_candy(tables, learnsets, member, item)
    new_max = int(spec_stats(tables, out)[0])
    assert out.current_hp == new_max - old_max > 0  # item_use_pokemon.c:294-298
    assert out.level == 31


def test_rare_candy_below_level_100_only(tables, names, learnsets):
    item, _ = candy_params(tables, names)
    with pytest.raises(ValueError, match="below"):
        rare_candy(tables, learnsets, sp(level=100), item)
    assert rare_candy(tables, learnsets, sp(level=99), item).level == 100


def test_rare_candy_learn_prompt_at_the_new_level(tables, names, manifest):
    item, _ = candy_params(tables, names)
    levels = {**manifest["level_learnsets"], str(OLD): [[31, MIRACLE_EYE]]}
    ls = Learnsets({**manifest, "level_learnsets": levels})
    full = sp(moves=(TACKLE, GROWL, CONFUSION, 98), pp_ups=(1, 2, 3, 0))
    gave_up = rare_candy(tables, ls, full, item)
    assert gave_up.moves == full.moves and gave_up.pp_ups == full.pp_ups
    learned = rare_candy(tables, ls, full, item, learn_answers={MIRACLE_EYE: 1})
    assert learned.moves == (TACKLE, MIRACLE_EYE, CONFUSION, 98)
    assert learned.pp_ups == (1, 0, 3, 0)  # callbacks.c:1105-1113
    free = sp(moves=(TACKLE, 0, 0, 0), pp_ups=(0, 2, 0, 0))
    assert rare_candy(tables, ls, free, item).moves == (TACKLE, MIRACLE_EYE, 0, 0)
    assert rare_candy(tables, ls, replace(free, level=31), item).moves == free.moves  # level 32


def test_rare_candy_evolution_is_a_cancellable_level_prompt(tables, names, manifest, method):
    item, _ = candy_params(tables, names)
    ls = level_evolver(manifest, method, level=31)
    out = rare_candy(tables, ls, sp(), item)
    prompt = evolution_prompt(tables, ls, 0, out, None, EVO_CLASS_BY_LEVEL, 0, None, personality=0)
    assert prompt is not None and prompt.cancellable and prompt.target == A
    before = evolution_prompt(tables, ls, 0, sp(), None, EVO_CLASS_BY_LEVEL, 0, None, personality=0)
    assert before is None


# --- GT-15 growth plan (E3-B3-15) ---------------------------------------------------------------

KNOWN = (TACKLE, GROWL, CONFUSION, 10)
M1, M2, M3, M4 = 150, 151, 152, 153  # offered moves; effects set apart below
CAP = 20
MOVE_COLUMNS = (
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


def edited(tables, **columns):
    """``tables`` with ``column[index] = value`` (or a row) for each ``name={index: value}``."""
    out = {}
    for name, edits in columns.items():
        arr = np.array(getattr(tables, name))
        for index, value in edits.items():
            arr[index] = value
        out[name] = jnp.asarray(arr)
    return tables._replace(**out)


def apart(tables, moves):
    """Every move of ``moves`` with its own supported effect, so none dominates another."""
    supported = [int(e) for e in EffectId if e != EffectId.UNSUPPORTED]
    return edited(tables, move_effect={m: supported[i] for i, m in enumerate(moves)})


def with_learnsets(manifest, method, entries, rows=None):
    """Learnsets with the level-up lists of ``entries`` (species -> ``[(level, move)]``) and the
    evolution ``rows`` of OLD."""
    levels = {
        **manifest["level_learnsets"],
        **{str(s): [[lv, mv] for lv, mv in e] for s, e in entries.items()},
    }
    patched = {**manifest, "level_learnsets": levels}
    if rows is not None:
        patched["evolutions"] = {
            **manifest["evolutions"],
            str(OLD): [[method[n], p, t] for n, p, t in rows],
        }
    return Learnsets(patched)


TWIN = 7  # the evolving species of the domination tests: no battle rule names it
CHANSEY, PIKACHU = 113, 25  # species a battle rule names (battle_lib.c:7127, :6749)


def plan_tables(tables, old=TWIN, new=A):
    """A twin of ``old``: ``new`` with ``old``'s types, abilities, base stats, experience group
    and weight, and the offered moves apart from each other and from the known ones."""
    columns = (
        "species_types",
        "species_abilities_decomp",
        "species_base_stats",
        "species_weight",
        "species_gender",
    )
    twin = edited(
        tables,
        **{c: {new: np.asarray(getattr(tables, c))[old]} for c in columns},
        species_exp_rate={new: 0, old: 0},
    )
    return apart(twin, [*KNOWN, M1, M2, M3, M4])


def old_member(**fields):
    return sp(**{"species": OLD, "level": 10, "moves": KNOWN, **fields})


def test_base_plan_gives_up_and_never_evolves():
    assert BASE_PLAN == GrowthPlan((), None)
    assert BASE_PLAN.answer((12, M1)) is None and BASE_PLAN.evolve_at is None
    plan = GrowthPlan((((12, M1), 2), ((14, M2), None)), 0)
    assert plan.answer((12, M1)) == 2 and plan.answer((14, M2)) is None
    assert plan.answers_at(12) == {M1: 2} and plan.answers_at(13) == {}


def test_max_growth_plans_is_64():
    assert MAX_GROWTH_PLANS == 64


def test_learn_offers_skip_known_moves_and_other_levels(manifest, method):
    entries = [(5, M1), (12, M1), (12, TACKLE), (21, M2), (15, M3)]
    ls = with_learnsets(manifest, method, {OLD: entries})
    assert learn_offers(ls, OLD, 0, {TACKLE}, 10, CAP) == ((12, M1), (15, M3))
    assert learn_offers(ls, OLD, 0, set(), 12, 15) == ((15, M3),)


def counted(tables, manifest, method, gap_level=14):
    """OLD offered M1 at 12 and M2 at 15; A offered M3 at 14 and M4 at 18; one evolution gap
    at level ``gap_level``. A is not a twin of OLD, so nothing is dominated."""
    t = apart(tables, [*KNOWN, M1, M2, M3, M4])
    entries = {OLD: [(12, M1), (15, M2)], A: [(14, M3), (18, M4)]}
    ls = with_learnsets(manifest, method, entries, [("level", gap_level, A)])
    gaps = [EvolutionGap(0, gap_level, A, method["level"])]
    return t, ls, member_options(t, ls, 0, old_member(), 0, CAP, gaps)


def test_plan_counts_for_two_offers_and_one_evolution_gap(tables, manifest, method):
    t, ls, opts = counted(tables, manifest, method)
    assert opts.offers == ((12, M1), (15, M2))
    (gap,) = opts.evolutions
    assert gap.pre == ((12, M1),) and gap.post == ((14, M3), (18, M4))
    # never: 5^2; evolve at the gap: the old offer before it and the two of the new species, 5^3
    assert member_plan_count(t, ls, opts) == 5**2 + 5**3
    plans = member_plans(t, ls, opts)
    assert len(plans) == 150 and len(set(plans)) == 150
    assert sum(p.evolve_at is None for p in plans) == 25
    assert all(p.evolve_at in (None, 0) for p in plans)
    # a setup is the product over members
    both = growth_plans(t, ls, [opts, opts], growth_cap=-1)
    assert both.total == 150 * 150 and both.kept == both.total and not both.capped


def test_an_hm_slot_is_never_an_answer(tables, manifest, method):
    t, ls, opts = counted(tables, manifest, method)
    hm = int(np.flatnonzero(np.asarray(t.move_is_hm))[0])
    t = apart(t, [hm, M1, M2, M3, M4])
    opts = opts._replace(member=old_member(moves=(hm, GROWL, CONFUSION, 10)))
    assert member_plan_count(t, ls, opts) == 4**2 + 4**3  # the HM slot is never replaced
    answers = [a for p in member_plans(t, ls, opts) for _, a in p.answers]
    assert 0 not in answers


def test_a_strictly_worse_duplicate_is_pruned_to_give_up(tables):
    t = apart(tables, [TACKLE])
    edits = {}
    for column in MOVE_COLUMNS:
        row = np.asarray(getattr(t, column))[TACKLE]
        edits[column] = {160: row, 161: row}
    t = edited(t, **edits)
    power = int(np.asarray(t.move_power)[TACKLE])
    pp = int(np.asarray(t.move_pp)[TACKLE])
    other_type = 1 - int(np.asarray(t.move_type)[TACKLE])
    t = edited(t, move_power={160: power - 1, 161: power + 1})
    assert give_up_only(t, [TACKLE], 160)  # lower power, the rest equal
    assert not give_up_only(t, [TACKLE], 161)  # more power: may be better
    assert not give_up_only(t, [GROWL], 160)  # a different known move
    more_pp = edited(t, move_pp={160: pp + 1}, move_power={160: power})
    assert not give_up_only(more_pp, [TACKLE], 160)
    another_type = edited(t, move_power={160: power}, move_type={160: other_type})
    assert not give_up_only(another_type, [TACKLE], 160)
    sure_hit = edited(t, move_power={160: power}, move_accuracy={TACKLE: 50, 160: 0})
    assert not give_up_only(sure_hit, [TACKLE], 160)  # never misses against a known 50


def test_an_unsupported_move_is_pruned_and_costs_no_branch(tables, manifest, method):
    t = apart(tables, [*KNOWN, M1, M2])
    t = edited(t, move_effect={M2: int(EffectId.UNSUPPORTED)})
    ls = with_learnsets(manifest, method, {OLD: [(12, M1), (15, M2)]}, [])
    opts = member_options(t, ls, 0, old_member(), 0, CAP, [])
    assert give_up_only(t, KNOWN, M2) and not give_up_only(t, KNOWN, M1)
    assert member_plan_count(t, ls, opts) == 5  # only M1 has a real choice
    assert all(dict(p.answers).keys() == {(12, M1)} for p in member_plans(t, ls, opts))


def twin_case(tables, manifest, method, old=TWIN, new=A, new_entries=((15, M2),)):
    entries = {old: [(12, M1), (15, M2)], new: list(new_entries)}
    ls = with_learnsets(manifest, method, entries)
    t = plan_tables(tables, old, new)
    gap = EvolutionGap(0, 14, new, method["level"])
    return t, ls, member_options(t, ls, 0, old_member(species=old), 0, CAP, [gap])


def test_evolve_dominates_cancel_only_when_every_condition_holds(tables, manifest, method):
    t, ls, opts = twin_case(tables, manifest, method)
    (gap,) = opts.evolutions
    assert evolve_dominates(t, ls, opts, gap)
    # then only the evolving path, at the earliest gap: 5^2 (the old offer at 12, the new at 15)
    assert member_plan_count(t, ls, opts) == 25
    assert {p.evolve_at for p in member_plans(t, ls, opts)} == {0}

    def row(column, index, delta):
        values = np.array(np.asarray(getattr(t, column))[A])
        values[index] = int(values[index]) + delta
        return {A: values}

    second = np.array(np.asarray(t.species_abilities_decomp)[A])
    second[1] = 0 if second[1] else int(second[0]) + 1
    cases = {
        "types change": edited(t, species_types=row("species_types", 0, 1)),
        "a base stat falls": edited(t, species_base_stats=row("species_base_stats", 3, -1)),
        # a faster battler moves last under Trick Room (battle_lib.c:1432-1435)
        "a base stat rises": edited(t, species_base_stats=row("species_base_stats", 3, 1)),
        "the ability changes": edited(
            t, species_abilities_decomp=row("species_abilities_decomp", 0, 1)
        ),
        # the AI guesses an unannounced ability from the species' pair (trainer_ai.c:1191-1192)
        "the species' other ability changes": edited(t, species_abilities_decomp={A: second}),
        "the experience group changes": edited(t, species_exp_rate={A: 5}),
        # personality 0 against ratio 254 is female, against 0 male (src/pokemon.c:2696-2705)
        "the gender changes": edited(t, species_gender={TWIN: 254, A: 0}),
        # Low Kick / Grass Knot: 10.0 kg and 10.1 kg are in different brackets (WEIGHT_BOUNDS)
        "Low Kick's power changes": edited(t, species_weight={TWIN: 100, A: 101}),
    }
    for why, bad in cases.items():
        assert not evolve_dominates(bad, ls, opts, gap), why
        assert member_plan_count(bad, ls, opts) == 25 + 25, why  # never and the gap, both kept
    same_bracket = edited(t, species_weight={TWIN: 101, A: 250})
    assert evolve_dominates(same_bracket, ls, opts, gap)


def test_evolve_does_not_dominate_when_a_move_comes_at_another_level(tables, manifest, method):
    # the old species' M2 at 15 comes at 16 on the evolved path: one fight without it
    t, ls, opts = twin_case(tables, manifest, method, new_entries=((16, M2),))
    assert not evolve_dominates(t, ls, opts, opts.evolutions[0])
    # and earlier is not better either: M2 at 14 replaces a move for the fights at 14
    t, ls, opts = twin_case(tables, manifest, method, new_entries=((14, M2),))
    assert not evolve_dominates(t, ls, opts, opts.evolutions[0])
    # lost outright
    t, ls, opts = twin_case(tables, manifest, method, new_entries=((15, M3),))
    assert not evolve_dominates(t, ls, opts, opts.evolutions[0])


def test_evolve_does_not_dominate_out_of_or_into_a_species_a_rule_names(tables, manifest, method):
    # Chansey's Lucky Punch (battle_lib.c:7127) is lost on its twin; Pikachu gains Light Ball
    for old, new in ((CHANSEY, A), (TWIN, PIKACHU)):
        t, ls, opts = twin_case(tables, manifest, method, old=old, new=new)
        assert not evolve_dominates(t, ls, opts, opts.evolutions[0]), (old, new)


@pytest.mark.parametrize("name", ["level_with_held_item_day", "level_ninjask"])
def test_evolve_does_not_dominate_with_an_item_using_or_shedinja_method(
    tables, manifest, method, name
):
    t, ls, _ = twin_case(tables, manifest, method)
    gaps = [EvolutionGap(0, 14, A, method[name])]
    opts = member_options(t, ls, 0, old_member(species=TWIN), 0, CAP, gaps)
    assert not evolve_dominates(t, ls, opts, opts.evolutions[0])


def test_only_the_earliest_gap_is_kept_when_evolving_dominates(tables, manifest, method):
    t, ls, _ = twin_case(tables, manifest, method)
    gaps = [EvolutionGap(g, 14 + g, A, method["level"]) for g in range(3)]
    opts = member_options(t, ls, 0, old_member(species=TWIN), 0, CAP, gaps)
    assert {p.evolve_at for p in member_plans(t, ls, opts)} == {0}


def test_an_offer_is_pruned_only_while_the_plan_still_holds_the_better_move(
    tables, manifest, method
):
    """Hand-built: TACKLE (slot 0) dominates move 160, offered at 15. A plan that replaced
    slot 0 with M1 at 12 no longer holds TACKLE, so 160 keeps its five answers there."""
    t = apart(tables, [*KNOWN, M1])
    t = edited(t, **{c: {160: np.asarray(getattr(t, c))[TACKLE]} for c in MOVE_COLUMNS})
    t = edited(t, move_power={160: int(np.asarray(t.move_power)[TACKLE]) - 1})
    assert give_up_only(t, KNOWN, 160)
    ls = with_learnsets(
        manifest, method, {OLD: [(15, 160), (12, M1)]}
    )  # table order is not game order
    opts = member_options(t, ls, 0, old_member(), 0, CAP, [])
    plans = member_plans(t, ls, opts)
    # M1 into slot 0: 160 has 5 answers; M1 elsewhere or given up: TACKLE is held, 160 pruned
    assert member_plan_count(t, ls, opts) == len(plans) == 5 + 4
    replaced = [p for p in plans if p.answer((12, M1)) == 0]
    assert sorted(p.answer((15, 160)) for p in replaced if p.answer((15, 160)) is not None) == [
        0,
        1,
        2,
        3,
    ]
    assert len(replaced) == 5 and all(
        dict(p.answers).keys() == {(12, M1), (15, 160)} for p in replaced
    )
    kept = [p for p in plans if p.answer((12, M1)) != 0]
    assert len(kept) == 4 and all((15, 160) not in dict(p.answers) for p in kept)


def many_gaps(tables, manifest, method, n):
    """A member with no offers and ``n`` eligible gaps: 1 + n plans (A has other types)."""
    ls = with_learnsets(manifest, method, {OLD: [], A: []}, [])
    gaps = [EvolutionGap(g, 11, A, method["level"]) for g in range(n)]
    return ls, member_options(tables, ls, 0, old_member(), 0, CAP, gaps)


def test_the_cap_enumerates_only_the_current_fight_and_reports_the_count(tables, manifest, method):
    ls, opts = many_gaps(tables, manifest, method, 64)
    assert member_plan_count(tables, ls, opts) == 65
    over = growth_plans(tables, ls, [opts], growth_cap=64)
    assert over.capped and over.total == 65 and over.kept == 2
    # the plan that evolves at the current fight's gap, and the one that does not
    assert {p.evolve_at for (p,) in over.setups()} == {None, 0}
    line = over.report_line()
    assert "2 of 65" in line and "--growth-cap -1" in line
    # without ``later`` the rest take the base plan, and the line says so (not a screen)
    assert "single-fight screen" not in line and "(the base plan)" in line
    assert "gives the move up" in line and "no later evolution is taken" in line
    # the default cap is MAX_GROWTH_PLANS
    assert growth_plans(tables, ls, [opts]).capped
    at = growth_plans(tables, ls, [opts], growth_cap=65)
    assert not at.capped and at.kept == 65 and at.report_line() is None
    every = growth_plans(tables, ls, [opts], growth_cap=-1)
    assert not every.capped and every.kept == 65 and len(list(every.setups())) == 65
    with pytest.raises(ValueError, match="below -1"):
        growth_plans(tables, ls, [opts], growth_cap=-2)


def test_later_prompts_take_the_screens_best_answer_above_the_cap(tables, manifest, method):
    ls, opts = many_gaps(tables, manifest, method, 64)
    calls = []

    def later(slot, plan):
        calls.append(slot)
        return plan._replace(answers=(*plan.answers, ((20, M1), 1)))

    over = growth_plans(tables, ls, [opts], growth_cap=10, later=later)
    assert calls == [0, 0]
    assert all(p.answer((20, M1)) == 1 for (p,) in over.setups())
    line = over.report_line()
    assert "best answer by single-fight screen" in line and "base plan" not in line
