"""Cross-check rows for moves, species, learnsets, TM/tutor tables, items and abilities
(Task 12). Synthetic three-source fixtures per function, plus real-input checks that skip
when the ROM, PKCalc zip or docs workbook is absent."""

import pytest

from nuzline.engine.ids import Stat
from tools.kaizo.crosscheck import compare
from tools.kaizo.rows_tables import (
    PKCALC_TARGETS,
    ability_rows,
    item_rows,
    learnset_rows,
    load_rom,
    move_rows,
    species_rows,
    table_rows,
    tm_rows,
    tutor_rows,
)
from tools.rom.inputs import KAIZO_DOCS_SHA256, PKCALC_SHA256


def find(rows, table, key, field):
    hits = [r for r in rows if (r.table, r.key, r.field) == (table, key, field)]
    assert len(hits) == 1, (table, key, field, hits)
    return hits[0]


# --- synthetic sources ------------------------------------------------------------------


def _rom_move(name, **kw):
    m = {
        "name": name,
        "power": 40,
        "type": 0,
        "category": 0,
        "accuracy": 100,
        "pp": 35,
        "priority": 0,
        "target": "RANGE_SINGLE_TARGET",
        "effect": 0,
        "ai_effect": 0,
        "effect_chance": 0,
        "flags": [],
    }
    m.update(kw)
    return m


def _pk_move(name, **kw):
    m = {
        "name": name,
        "aliases": [name, name],
        "type": 0,
        "category": 0,
        "power": 40,
        "accuracy": 100,
        "pp": 35,
        "chance": 0,
        "priority": 0,
        "target": "normal",
        "effect_text": "",
        "machine": 0,
        "tutor": None,
        "price": [],
    }
    m.update(kw)
    return m


def _sheet_move(name, **kw):
    m = {
        "name": name,
        "label": "None",
        "effect": 0,
        "category": 0,
        "power": 40,
        "type": 0,
        "accuracy": 100,
        "pp": 35,
        "chance": 0,
        "target": "RANGE_SINGLE_TARGET",
        "priority": 0,
        "flags": [],
        "cite": "Moves:3",
    }
    m.update(kw)
    return m


def rom_moves(n, **kw):
    return [_rom_move("-")] + [_rom_move(f"Move{i}", **kw) for i in range(1, n + 1)]


def pkcalc_moves(n, **kw):
    return {"moves": {str(i): _pk_move(f"Move{i}", **kw) for i in range(1, n + 1)}}


def empty_docs():
    return {"moves": {}}


def _species(stats, types, abilities, **kw):
    s = {
        "name": "X",
        "base_stats": stats,
        "types": types,
        "abilities": abilities,
        "gender_ratio": 31,
        "weight_hg": 1,
        "base_friendship": 70,
        "exp_rate": 3,
        "exp_yield": 64,
        "catch_rate": 45,
    }
    s.update(kw)
    return s


@pytest.fixture
def synthetic_sources():
    """(rom, pkcalc, docs) for two species, one form, three moves, a TM and a tutor."""
    rom = {
        "moves": [_rom_move("-"), _rom_move("Pound"), _rom_move("Ember"), _rom_move("Tackle")],
        "species": [
            _species([0] * 6, [0, 0], [0, 0], name="----------"),
            _species([45, 49, 49, 65, 65, 45], [12, 3], [65, 65], name="BULBASAUR"),
            _species([60, 62, 63, 80, 80, 60], [12, 3], [65, 0], name="IVYSAUR"),
        ],
        "forms": [
            {
                "species": 2,
                "form": 1,
                "row": 3,
                "name": "Mega",
                "base_stats": [60, 62, 63, 80, 80, 91],
                "types": [12, 3],
                "abilities": [26, 0],
                "gender_ratio": 31,
                "base_friendship": 70,
                "exp_rate": 3,
                "exp_yield": 141,
                "weight_hg": 1,
            }
        ],
        "learnsets": {"0": [], "1": [[1, 1], [7, 3]], "2": [[1, 2]], "3": [[1, 1]]},
        "tm_learnsets": {"0": [], "1": [0], "2": [0, 1], "3": []},
        "tutor_learnsets": {"0": [], "1": [0], "2": [], "3": []},
        "evolutions": {"0": [], "1": [[4, 16, 2]], "2": []},
        "evolution_methods": ["none", "level_happiness", "x2", "x3", "level"],
        "items": [
            {"name": "None", "hold_effect": 0, "hold_param": 0, "teaches_move": None},
            {
                "name": "Charcoal",
                "hold_effect": 5,
                "hold_param": 20,
                "teaches_move": None,
                "fling_power": 30,
            },
            {"name": "ITEM_UNUSED_2", "hold_effect": 0, "hold_param": 0, "teaches_move": None},
            {"name": "TM01", "hold_effect": 0, "hold_param": 0, "teaches_move": 3},
            {"name": "HM01", "hold_effect": 0, "hold_param": 0, "teaches_move": 1},
        ],
        "tutor_moves": [
            {
                "move": 2,
                "costs": {"red": 0, "blue": 6, "yellow": 0, "green": 0},
                "location": "route_212",
            },
            {
                "move": 3,
                "costs": {"red": 1, "blue": 0, "yellow": 0, "green": 0},
                "location": "route_212",
            },
        ],
        "abilities": [{"name": "-"}, {"name": "Stench"}],
    }
    pk_species = {
        "num": 1,
        "form": 0,
        "name": "Bulbasaur",
        "base_stats": [45, 49, 49, 65, 65, 45],
        "types": [12, 3],
        "abilities": [65, 0],
        "catch_rate": 45,
        "exp_yield": 64,
        "growth_rate": 3,
        "gender_ratio": 31,
        "evolutions": [],
        "level_up": [[1, 1], [7, 3]],
        "tm": [3],
        "tutor": [2],
    }
    pkcalc = {
        "moves": {
            "1": _pk_move("Pound"),
            "2": _pk_move("Ember"),
            "3": _pk_move("Tackle"),
        },
        "species": {
            "1": pk_species,
            "2": dict(
                pk_species,
                num=2,
                name="Ivysaur",
                base_stats=[60, 62, 63, 80, 80, 60],
                level_up=[[1, 2]],
                tm=[3, 1],
                tutor=[],
                exp_yield=141,
            ),
            "3": dict(
                pk_species,
                num=2,
                form=1,
                name="Ivysaur-Mega",
                base_stats=[60, 62, 63, 80, 80, 91],
                abilities=[26, 0],
                level_up=[[1, 1]],
                tm=[],
                tutor=[],
                exp_yield=141,
            ),
        },
        "items": {
            "1": {"name": "Charcoal", "aliases": ["Charcoal"], "desc": ""},
            "3": {"name": "TM01", "aliases": ["TM01"], "desc": ""},
            "4": {"name": "HM01", "aliases": ["HM01"], "desc": ""},
        },
        "abilities": {"1": {"name": "Stench", "aliases": ["Stench"], "desc": {}}},
        "tm_table": {"1": 3, "93": 1},
        "tutor_table": [
            {
                "move": 2,
                "cost": {"red": 0, "blue": 6, "yellow": 0, "green": 0},
                "location": "route212",
            },
            {
                "move": 3,
                "cost": {"red": 2, "blue": 0, "yellow": 0, "green": 0},
                "location": "route212",
            },
        ],
    }
    docs = {
        "moves": {
            "1": _sheet_move("Pound"),
            "2": _sheet_move("Ember"),
            "3": _sheet_move("Tackle"),
        },
        "species": {
            "1": {
                "name": "BULBASAUR",
                "base_stats": [45, 49, 49, 65, 65, 45],
                "types": [12, 3],
                "catch_rate": 59,
                "exp_yield": 64,
                "gender_ratio": 31,
                "growth_rate": "Medium Slow",
                "abilities": ["Overgrow", "Overgrow"],
            },
            "2": {
                "name": "IVYSAUR",
                "base_stats": [60, 62, 63, 80, 80, 60],
                "types": [12, 3],
                "catch_rate": 45,
                "exp_yield": 141,
                "gender_ratio": 31,
                "growth_rate": "Medium Slow",
                "abilities": ["Overgrow", "-"],
            },
        },
        "evolutions": {
            "1": {
                "name": "BULBASAUR",
                "evolutions": [{"method": "Level Up", "required": 16, "result": "IVYSAUR"}],
            },
        },
        "level_up": {"1": {"name": "BULBASAUR", "moves": [["Pound", 1], ["Tackle", 7]]}},
        "tm": {"1": {"name": "BULBASAUR", "machines": ["TM01"]}},
        "tm_moves": {"TM01": {"move": "Tackle"}, "HM01": {"move": "Ember"}},
        "items": {
            "Charcoal": {"sheet_id": 1, "hold_effect": 5, "hold_param": 50, "fling_power": 30}
        },
    }
    rom["abilities"] += [{"name": f"A{i}"} for i in range(2, 66)]
    rom["abilities"][26] = {"name": "Levitate"}
    rom["abilities"][65] = {"name": "Overgrow"}
    return rom, pkcalc, docs


# --- moves -------------------------------------------------------------------------------


def test_move_fields_and_execution():
    rows = move_rows(rom_moves(1), pkcalc_moves(1), {"moves": {"1": _sheet_move("Move1")}})
    fields = {r.field: r.executed for r in rows}
    assert fields == {
        "power": True,
        "accuracy": True,
        "pp": True,
        "priority": True,
        "type": True,
        "category": True,
        "target": True,
        "chance": True,
        "effect": True,
        "name": False,
    }
    assert {r.key for r in rows} == {"1 Move1"}
    assert find(rows, "moves", "1 Move1", "target").pkcalc == "RANGE_SINGLE_TARGET"
    effect = find(rows, "moves", "1 Move1", "effect")
    assert (effect.rom, effect.pkcalc, effect.sheet) == (0, None, 0)


def test_new_effect_label_is_reported_not_agreed():
    docs = {"moves": {"1": _sheet_move("Move1", effect=None, label="1/2 recoil, fixed AI")}}
    row = find(move_rows(rom_moves(1), pkcalc_moves(1), docs), "moves", "1 Move1", "effect")
    assert row.sheet == "label: 1/2 recoil, fixed AI"
    assert compare([row], []).outcomes[0].kind == "rom_vs_sheet"


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="nowhere"):
        move_rows(rom_moves(1), pkcalc_moves(1, target="nowhere"), empty_docs())


def test_ruling_33_targets_mapped():
    assert PKCALC_TARGETS["adjacentAlly"] == "RANGE_ALLY"
    assert PKCALC_TARGETS["adjacentAllyOrSelf"] == "RANGE_USER_OR_ALLY"
    assert PKCALC_TARGETS["adjacentFoe"] == "RANGE_SINGLE_TARGET_ME_FIRST"


@pytest.mark.parametrize("source", ["rom", "pkcalc", "sheet"])
def test_priority_out_of_range_raises(source):
    rom, pk, docs = rom_moves(1), pkcalc_moves(1), {"moves": {"1": _sheet_move("Move1")}}
    if source == "rom":
        rom[1]["priority"] = 8
    elif source == "pkcalc":
        pk["moves"]["1"]["priority"] = -8
    else:
        docs["moves"]["1"]["priority"] = 9
    with pytest.raises(ValueError, match="priority"):
        move_rows(rom, pk, docs)


# --- species -----------------------------------------------------------------------------


def test_species_fields(synthetic_sources):
    rows = species_rows(*synthetic_sources)
    assert {r.field: r.executed for r in rows} == {
        "base_stats": True,
        "types": True,
        "abilities": True,
        "exp_yield": True,
        "growth_rate": True,
        "gender_ratio": True,
        "catch_rate": False,
        "name": False,
    }
    assert {r.key for r in rows} == {"1 Bulbasaur", "2 Ivysaur", "3 Ivysaur-Mega"}
    # slot 2 = 0 means "slot 1" (src/pokemon.c:472-483): [65, 65] and [65, 0] agree
    ab = find(rows, "species", "1 Bulbasaur", "abilities")
    assert ab.rom == ab.pkcalc == ab.sheet == [65, 65]
    form = find(rows, "species", "3 Ivysaur-Mega", "base_stats")
    assert form.rom == form.pkcalc == [60, 62, 63, 80, 80, 91]
    catch = find(rows, "species", "1 Bulbasaur", "catch_rate")
    assert (catch.rom, catch.pkcalc, catch.sheet) == (45, 45, 59)
    growth = find(rows, "species", "1 Bulbasaur", "growth_rate")
    assert growth.rom == growth.pkcalc == growth.sheet == 3
    name = find(rows, "species", "1 Bulbasaur", "name")
    assert name.rom == name.pkcalc == name.sheet == "BULBASAUR"
    assert all(isinstance(r.rom, list) for r in rows if r.field in ("base_stats", "types"))


# --- learnsets ---------------------------------------------------------------------------


def test_learnset_rows(synthetic_sources):
    rows = learnset_rows(*synthetic_sources)
    assert {(r.field, r.executed) for r in rows} == {
        ("level_up", True),
        ("tm", True),
        ("tutor", True),
        ("evolutions", False),
    }
    lv = find(rows, "learnsets", "1 Bulbasaur", "level_up")
    assert lv.rom == lv.pkcalc == lv.sheet == [[1, 1], [7, 3]]
    tm = find(rows, "learnsets", "2 Ivysaur", "tm")
    assert tm.rom == tm.pkcalc == ["TM01", "HM01"]  # machine labels, machine order
    assert find(rows, "learnsets", "1 Bulbasaur", "tm").sheet == ["TM01"]
    tutor = find(rows, "learnsets", "1 Bulbasaur", "tutor")
    assert (tutor.rom, tutor.pkcalc, tutor.sheet) == ([2], [2], None)
    evo = find(rows, "learnsets", "1 Bulbasaur", "evolutions")
    assert evo.rom == evo.sheet == [["level", 16, 2]] and evo.pkcalc is None
    assert not [r for r in rows if r.field == "evolutions" and r.key.startswith("3 ")]


def test_sheet_move_names(synthetic_sources):
    rom, pk, docs = synthetic_sources
    # case-insensitive ("Solarbeam"); a name no id carries is kept, never guessed
    docs["level_up"]["1"]["moves"] += [["TACKLE", 8], ["Splash", 9]]
    row = find(learnset_rows(rom, pk, docs), "learnsets", "1 Bulbasaur", "level_up")
    assert row.sheet[-2:] == [[8, 3], [9, "name: Splash"]]


def test_pkcalc_tm_move_without_machine_raises(synthetic_sources):
    rom, pk, docs = synthetic_sources
    pk["species"]["1"]["tm"] = [2]
    with pytest.raises(ValueError, match="PKCalc TM move"):
        learnset_rows(rom, pk, docs)


def test_unknown_evolution_label_raises(synthetic_sources):
    rom, pk, docs = synthetic_sources
    docs["evolutions"]["1"]["evolutions"][0]["method"] = "Spin Around"
    with pytest.raises(ValueError, match="Spin Around"):
        learnset_rows(rom, pk, docs)


# --- TM and tutor tables ----------------------------------------------------------------


def test_tm_rows(synthetic_sources):
    rows = tm_rows(*synthetic_sources)
    assert [(r.key, r.rom, r.pkcalc, r.sheet, r.executed) for r in rows] == [
        ("TM01", 3, 3, 3, True),
        ("HM01", 1, 1, 2, True),
    ]


def test_tutor_cost_unexecuted(synthetic_sources):
    rows = tutor_rows(*synthetic_sources)
    assert {(r.field, r.executed) for r in rows} == {
        ("move", True),
        ("cost", False),
        ("location", False),
    }


def test_tutor_rows_keyed_by_move(synthetic_sources):
    rom, pk, docs = synthetic_sources
    pk["tutor_table"][1] = dict(pk["tutor_table"][1], move=1)  # 1 in PKCalc only, 3 in ROM only
    rows = tutor_rows(rom, pk, docs)
    pound = find(rows, "tutor_table", "1 Pound", "move")
    tackle = find(rows, "tutor_table", "3 Tackle", "move")
    assert (pound.rom, pound.pkcalc, pound.sheet) == (False, True, None)
    assert (tackle.rom, tackle.pkcalc, tackle.sheet) == (True, False, None)
    assert not [r for r in rows if r.key == "3 Tackle" and r.field != "move"]
    cost = find(rows, "tutor_table", "2 Ember", "cost")
    assert cost.rom == cost.pkcalc == {"red": 0, "blue": 6, "yellow": 0, "green": 0}
    loc = find(rows, "tutor_table", "2 Ember", "location")
    assert loc.rom == loc.pkcalc == "route212"


def test_tutor_cost_differs(synthetic_sources):
    rows = tutor_rows(*synthetic_sources)
    cost = find(rows, "tutor_table", "3 Tackle", "cost")
    assert cost.rom["red"] == 1 and cost.pkcalc["red"] == 2
    assert compare([cost], []).outcomes[0].kind == "unexecuted"


# --- items and abilities ----------------------------------------------------------------


def test_item_rows(synthetic_sources):
    rows = item_rows(*synthetic_sources)
    hold = find(rows, "items", "1 Charcoal", "hold_param")
    assert (hold.rom, hold.pkcalc, hold.sheet, hold.executed) == (20, None, 50, True)
    fling = find(rows, "items", "1 Charcoal", "fling_power")
    assert (fling.rom, fling.pkcalc, fling.sheet) == (30, None, 30)
    assert not find(rows, "items", "1 Charcoal", "name").executed
    assert not [r for r in rows if r.key.startswith("2 ")]  # ITEM_UNUSED_*


def test_item_sheet_ids_are_packed(synthetic_sources):
    rom, pk, docs = synthetic_sources
    docs["items"]["TM01"] = {"sheet_id": 2, "hold_effect": 0, "hold_param": 0, "fling_power": 9}
    rows = item_rows(rom, pk, docs)
    assert find(rows, "items", "3 TM01", "fling_power").sheet == 9


def test_ability_rows(synthetic_sources):
    rows = ability_rows(*synthetic_sources)
    assert len(rows) == 65 and {r.field for r in rows} == {"name"}
    assert (rows[0].key, rows[0].rom, rows[0].pkcalc, rows[0].executed) == (
        "1 Stench",
        "Stench",
        "Stench",
        False,
    )
    assert (rows[1].key, rows[1].rom, rows[1].pkcalc) == ("2 A2", "A2", None)


# --- real inputs (skip when absent) -----------------------------------------------------


@pytest.fixture(scope="module")
def real_rows(kaizo_rom_dir, pkcalc, docs):
    return table_rows(load_rom(kaizo_rom_dir), pkcalc, docs)


@pytest.fixture(scope="module")
def real_check(real_rows):
    return compare(real_rows, [])


def test_trick_room_priority_row(real_rows):
    row = find(real_rows, "moves", "433 Trick Room", "priority")
    assert (row.rom, row.pkcalc) == (7, 7)


def test_known_sheet_conflicts_listed(real_check):  # spec 5.2: Swords Dance PP sheet 30
    assert any(
        o.row.key.endswith("Swords Dance") and o.row.field == "pp"
        for o in real_check.outcomes
        if o.kind != "agree"
    )


def test_forms_compared(real_rows):  # Rotom-Heat is personal index 503 (Ruling 4)
    row = next(
        r
        for r in real_rows
        if r.table == "species" and r.key.startswith("503 ") and r.field == "base_stats"
    )
    assert row.rom[Stat.SPE] == 91 and row.rom == row.pkcalc


def test_tm07_is_aurora_beam(real_rows):
    row = find(real_rows, "tm_table", "TM07", "move")
    assert (row.rom, row.pkcalc, row.executed) == (62, 62, True)


def test_every_pkcalc_target_maps(pkcalc):  # preflight #33
    targets = {m["target"] for m in pkcalc["moves"].values()}
    assert targets <= set(PKCALC_TARGETS)


def test_real_tutor_table_has_38_moves(real_rows):
    moves = [r for r in real_rows if r.table == "tutor_table" and r.field == "move"]
    assert sum(r.rom for r in moves) == sum(r.pkcalc for r in moves) == 38


def test_real_abilities_all_equal(real_check):
    out = [o for o in real_check.outcomes if o.row.table == "abilities"]
    assert len(out) == 123 and {o.kind for o in out} == {"agree"}


def test_pkcalc_machine_teaching_one_move_twice_raises(synthetic_sources):
    rom, pk, docs = synthetic_sources
    pk["tm_table"]["93"] = 3  # HM01 teaches TM01's move
    with pytest.raises(ValueError, match="taught by both TM01 and HM01"):
        learnset_rows(rom, pk, docs)


def test_sheet_names_differing_only_by_case_raise(synthetic_sources):
    rom, pk, docs = synthetic_sources
    docs["moves"]["2"]["name"] = "POUND"
    with pytest.raises(ValueError, match="sheet move name 'POUND'"):
        learnset_rows(rom, pk, docs)


def test_spec_5_5_fling_power_from_rom(real_check):  # Town Map 0 -> 120 (spec 5.5)
    out = next(
        o
        for o in real_check.outcomes
        if o.row.table == "items" and o.row.key == "442 Town Map" and o.row.field == "fling_power"
    )
    assert (out.row.rom, out.row.sheet, out.kind) == (120, 120, "agree")


def test_spec_5_3_catch_rate_from_rom(real_check):  # sheet catch rate disagrees (spec 5.3)
    out = next(
        o
        for o in real_check.outcomes
        if o.row.table == "species" and o.row.key == "1 Bulbasaur" and o.row.field == "catch_rate"
    )
    assert (out.row.rom, out.row.pkcalc, out.row.sheet, out.kind) == (175, 175, 59, "sheet_only")


def test_real_unresolved_set(real_check, pkcalc, docs):  # Ruling 19: audited inputs only
    if pkcalc["sha256"] != PKCALC_SHA256 or docs["sha256"] != KAIZO_DOCS_SHA256:
        pytest.skip("PKCalc zip or docs workbook is not the audited one")
    got = sorted((r.table, r.key, r.field) for r in real_check.unresolved)
    assert got == [
        ("moves", "270 Helping Hand", "target"),
        ("moves", "367 Acupressure", "target"),
        ("moves", "382 Me First", "target"),
    ]
