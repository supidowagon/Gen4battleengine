"""Committed Kaizo overrides: schema, citation checks and the reviewed draft (Task 10)."""

import json
import tomllib

import pytest

from tools.kaizo.overrides import (
    DEFAULT_PATH,
    AreaException,
    AreaOverride,
    Gauntlet,
    Mappings,
    Overrides,
    OverridesError,
    build_mappings,
    check_citations,
    check_gauntlet_members,
    check_rom_trick_room,
    check_trainer_ids,
    draft_overrides,
    load_overrides,
)
from tools.kaizo.trainer_map import Mapping, MappingError, map_trainers


def write(tmp_path, text):
    p = tmp_path / "overrides.toml"
    p.write_text(text, encoding="utf-8")
    return p


def overrides_with(**kw):
    return Overrides(**kw)


# --- loader ---------------------------------------------------------------------------


def test_entry_without_source_refused(tmp_path):
    with pytest.raises(OverridesError, match="area.246.*source"):
        load_overrides(write(tmp_path, "[area.246]\ntrick_room = true\n"))


def test_unknown_rule_key_refused(tmp_path):
    with pytest.raises(OverridesError, match="'heal'"):
        load_overrides(
            write(
                tmp_path,
                '[[gauntlet]]\nname = "g"\ntrainers = [1, 2]\nrules = {heal = true}\n'
                'source = "user, 2026-11-03"\n',
            )
        )


@pytest.mark.parametrize(
    "source", ["somewhere", "pkcalc:sets.js", "user 2026-11-03", "Roark Split:x", "rom:"]
)
def test_malformed_source_refused(tmp_path, source):
    with pytest.raises(OverridesError, match="area.7: source"):
        load_overrides(write(tmp_path, f'[area.7]\ntrick_room = true\nsource = "{source}"\n'))


def test_unknown_key_refused(tmp_path):
    with pytest.raises(OverridesError, match="area.7.*'trickroom'"):
        load_overrides(write(tmp_path, '[area.7]\ntrickroom = true\nsource = "rom:x"\n'))


def test_unknown_table_refused(tmp_path):
    with pytest.raises(OverridesError, match="'level_caps'"):
        load_overrides(write(tmp_path, "[level_caps]\nGalactic = 100\n"))


def test_area_and_exception_on_one_id_refused(tmp_path):
    text = (
        '[area.7]\ntrick_room = true\nsource = "rom:x"\n'
        '[area_exception.7]\nreason = "r"\nsource = "rom:y"\n'
    )
    with pytest.raises(OverridesError, match="7"):
        load_overrides(write(tmp_path, text))


def test_loads_every_table(tmp_path):
    text = """
[area.403]
trick_room = true
source = "rom:map header 308 weather 18 (Trick Room)"

[area.870]
gravity = true
source = "pkcalc:src/js/shared_controls.js:611"

[area_exception.509]
reason = "two maps"
source = "rom:trainers.json start_condition unknown"

[[gauntlet]]
name = "Oreburgh Mine"
trainers = [196, 195]
rules = {}
source = "Roark Split:229"

[trainer_ids."Gardenia Split:91"]
id = 296
source = "pkcalc:src/js/data/sets.js:988"

[fields.107]
optional = true
source = "Wake Split:195"

[move_lists.punch]
ids = [5, 7]
source = "user, 2026-11-03"
"""
    o = load_overrides(write(tmp_path, text))
    assert o.area == {
        403: AreaOverride(True, False, None, "rom:map header 308 weather 18 (Trick Room)"),
        870: AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611"),
    }
    assert o.area_exceptions == {
        509: AreaException("two maps", "rom:trainers.json start_condition unknown")
    }
    assert o.gauntlets == (Gauntlet("Oreburgh Mine", (196, 195), {}, "Roark Split:229"),)
    assert o.trainer_ids == {"Gardenia Split:91": 296}
    assert o.fields == {107: {"optional": True}}
    assert o.move_lists == {"punch": [5, 7]}
    assert o.sources["trainer_ids.Gardenia Split:91"] == "pkcalc:src/js/data/sets.js:988"


def test_gauntlet_rules_must_be_booleans(tmp_path):
    text = (
        '[[gauntlet]]\nname = "g"\ntrainers = [1, 2]\nrules = {heal_between = 1}\n'
        'source = "user, 2026-11-03"\n'
    )
    with pytest.raises(OverridesError, match="heal_between"):
        load_overrides(write(tmp_path, text))


def test_missing_file_is_empty(tmp_path):
    assert load_overrides(tmp_path / "absent.toml") == Overrides()


# --- citation checks (synthetic docs) ---------------------------------------------------


def header(cite, text, tags=None, party=(("Onix", 12),)):
    return {
        "cite": cite,
        "text": text,
        "name": text.split(" (")[0],
        "tags": tags or {},
        "party": [{"species": s, "level": lv} for s, lv in party],
    }


def section(cite, name, trainers, tags=None):
    return {"cite": cite, "name": name, "text": name, "tags": tags or {}, "trainers": trainers}


def docs_of(*sections):
    return {"splits": {"S": {"sheet": "S Split", "sections": list(sections)}}}


SYN = docs_of(
    section(
        "S Split:1",
        "Warehouse",
        [header("S Split:2", "Grunt A"), header("S Split:3", "Grunt B")],
        {"permanent": "trick_room"},
    ),
    section(
        "S Split:10",
        "Mine",
        [
            header("S Split:11", "Worker A (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:12", "Worker B (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:13", "Hiker C"),
        ],
    ),
    section(
        "S Split:20",
        "League",
        [
            header("S Split:21", "Champion X (Permanent Gravity)", {"permanent": "gravity"}),
            header("S Split:22", "Hooded (PERMANENT TRICK ROOM)", {"permanent": "trick_room"}),
        ],
    ),
)


def test_section_tag_supports_header_citation():
    o = overrides_with(area={5: AreaOverride(True, False, None, "S Split:3")})
    assert check_citations(o, SYN) == []


def test_header_without_claim_is_stale():
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "S Split:13"),
            6: AreaOverride(False, True, None, "S Split:22"),
        }
    )
    assert check_citations(o, SYN) == [
        "area.5: S Split:13 does not say Trick Room",
        "area.6: S Split:22 does not say Gravity",
    ]


def test_citation_to_missing_row():
    o = overrides_with(area={5: AreaOverride(True, False, None, "S Split:99")})
    assert check_citations(o, SYN) == ["area.5: S Split:99 is no split sheet row"]


def test_gauntlet_citation_needs_gauntlet_tag():
    o = overrides_with(
        gauntlets=(
            Gauntlet("ok", (1, 2), {}, "S Split:11"),
            Gauntlet("bad", (3, 4), {}, "S Split:13"),
        )
    )
    assert check_citations(o, SYN) == ["gauntlet 'bad': S Split:13 does not say gauntlet"]


def test_non_sheet_sources_are_not_checked_against_docs():
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "rom:map header 1"),
            6: AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611"),
        }
    )
    assert check_citations(o, SYN) == []


def test_rom_trick_room_needs_override_or_exception():
    effects = [
        {"id": 5, "name": "A", "effect": "trick_room", "map_headers": [305]},
        {"id": 6, "name": "B", "effect": "trick_room", "map_headers": [305]},
        {"id": 7, "name": "C", "effect": "trick_room", "map_headers": [307]},
        {"id": 8, "name": "D", "effect": "trick_room", "map_headers": [307]},
    ]
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "rom:map header 305 weather 18 (Trick Room)"),
            8: AreaOverride(False, True, None, "rom:x"),
        },
        area_exceptions={6: AreaException("two maps", "rom:y")},
    )
    assert check_rom_trick_room(o, effects) == [
        (
            "area.7: ROM Trick Room (C, map headers [307]) has no trick_room override "
            "and no area_exception"
        ),
        (
            "area.8: ROM Trick Room (D, map headers [307]) has no trick_room override "
            "and no area_exception"
        ),
    ]


TR_ROWS = [
    {"id": 5, "name": "Saturn", "effect": "trick_room", "map_headers": [305]},
    {"id": 6, "name": "Io", "effect": "trick_room", "map_headers": [305, 306]},
]


@pytest.mark.parametrize(
    ("tid", "source", "problem"),
    [
        (5, "rom:map header 305 weather 18 (Trick Room)", None),
        (6, "rom:map headers 305, 306 weather 18 (Trick Room)", None),
        (246, "rom:map header 999 weather 18 (Trick Room)", "has no Trick Room row"),
        (5, "rom:map header 306 weather 18 (Trick Room)", "has map headers [305]"),
        (5, "rom:somewhere with Trick Room", "is no ROM Trick Room citation"),
    ],
)
def test_rom_trick_room_citations_checked_back(tid, source, problem):
    rows = [*TR_ROWS, {"id": 7, "name": "X", "effect": "trick_room", "map_headers": [1]}]
    o = overrides_with(
        area={tid: AreaOverride(True, False, None, source)},
        area_exceptions={i: AreaException("r", "rom:y") for i in (5, 6, 7) if i != tid},
    )
    got = check_rom_trick_room(o, rows)
    if problem is None:
        assert got == []
    else:
        assert len(got) == 1 and got[0].startswith(f"area.{tid}: ") and problem in got[0]


@pytest.mark.parametrize(
    ("tid", "source", "problem"),
    [
        (5, "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 305)", None),
        (
            9,
            "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 305)",
            "not 9",
        ),
        (
            5,
            "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 999)",
            "[305]",
        ),
        (5, "rom:only Mars on a Trick Room map (rom_field_effects.json, map header 305)", "not 5"),
        (5, "rom:only Saturn somewhere", "is no ROM Trick Room citation"),
    ],
)
def test_rom_only_trainer_id_citations_checked(tid, source, problem):
    o = overrides_with(
        trainer_ids={"S Split:2": tid},
        sources={"trainer_ids.S Split:2": source},
        area_exceptions={5: AreaException("r", "rom:y"), 6: AreaException("r", "rom:y")},
    )
    got = check_rom_trick_room(o, TR_ROWS)
    if problem is None:
        assert got == []
    else:
        assert len(got) == 1 and got[0].startswith("trainer_ids.S Split:2: ") and problem in got[0]


def test_gauntlet_members_tied_to_cited_group():
    sheet = [
        Mapping("S Split:11", 21, "exact", (21,)),
        Mapping("S Split:12", 22, "exact", (22,)),
        Mapping("S Split:13", 23, "exact", (23,)),
        Mapping("S Split:2", 2, "exact", (2,)),
    ]
    o = overrides_with(
        gauntlets=(
            Gauntlet("ok", (21, 22), {}, "S Split:11"),
            Gauntlet("wrong", (2, 23), {}, "S Split:11"),
        )
    )
    got = check_gauntlet_members(o, SYN, sheet)
    assert [p.split(":")[0] for p in got] == ["gauntlet 'wrong'", "gauntlet 'wrong'"]
    assert "trainer 2 " in got[0] and "trainer 23 " in got[1]  # 23 is outside the chain


def test_trainer_in_two_gauntlets_refused(tmp_path):
    g = '[[gauntlet]]\nname = "{}"\ntrainers = {}\nrules = {{}}\nsource = "user, 2026-11-03"\n'
    text = g.format("a", "[1, 2]") + g.format("b", "[2, 3]")
    with pytest.raises(OverridesError, match="gauntlet 'b': trainer 2 .*'a'"):
        load_overrides(write(tmp_path, text))


def test_ids_must_be_rom_trainers():
    o = overrides_with(
        area={5: AreaOverride(True, False, None, "rom:x")},
        gauntlets=(Gauntlet("g", (5, 9), {}, "S Split:11"),),
    )
    assert check_trainer_ids(o, {5, 6}) == ["gauntlet 'g': 9 is no ROM trainer id"]


# --- draft (synthetic) ----------------------------------------------------------------


def syn_mappings(sheet, field_effects=(), starts=None, pkcalc=()):
    rom = {
        i: {"id": i, "name": f"T{i}", "start_condition": (starts or {}).get(i, "none")}
        for i in range(1, 40)
    }
    return Mappings(
        pkcalc=tuple(pkcalc),
        sheet=tuple(
            Mapping(c, i, "exact" if i else "unmapped", (i,) if i else ()) for c, i in sheet
        ),
        rom_trainers=rom,
        field_effects=tuple(field_effects),
    )


DRAFT_DOCS = docs_of(
    section(
        "S Split:1",
        "Warehouse",
        [
            header("S Split:2", "Grunt A"),
            header("S Split:3", "Grunt B"),
            header("S Split:4", "Grunt C"),
            header("S Split:5", "Grunt D"),
        ],
        {"permanent": "trick_room"},
    ),
    section(
        "S Split:10",
        "Highlands",
        [header("S Split:11", "Grunt B"), header("S Split:12", "Hiker")],
        {"weather": "hail", "assumed_permanent": True},
    ),
    section(
        "S Split:20",
        "Mine",
        [
            header("S Split:21", "W1 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:22", "W2 (Gauntlet)", {"gauntlet": "middle"}),
            header("S Split:23", "W3 (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:24", "W4 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:25", "W5 (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:26", "Galactic M (B2B With Galactic T)", {"b2b_with": "Galactic T"}),
            header("S Split:27", "GalacticfT (B2B With Galactic M)", {"b2b_with": "Galactic M"}),
            header("S Split:28", "U1 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:29", "U2 (Gauntlet End)", {"gauntlet": "end"}),
        ],
    ),
    section(
        "S Split:30",
        "Gym",
        [
            header("S Split:31", "Lass"),
            header("S Split:32", "Artist"),
            header("S Split:33", "Leader G"),
        ],
        {"gym_gauntlet": True},
    ),
    section(
        "S Split:40",
        "Cave",
        [
            header("S Split:41", "Mira (Tag Partner)", {"tag_partner": True}),
            header("S Split:42", "C1 (MULTI, GAUNTLET)", {"gauntlet": "middle"}),
            header("S Split:43", "C2 (MULTI, GAUNTLET)", {"gauntlet": "middle"}),
            header("S Split:44", "C3 (MULTI, GAUNTLET END)", {"gauntlet": "end"}),
            header("S Split:45", "C4 (MULTI, GAUNTLET END)", {"gauntlet": "end"}),
        ],
        {"optional": True, "gauntlet_with": "Mira"},
    ),
    section(
        "S Split:50",
        "League",
        [
            header("S Split:51", "Hooded (PERMANENT TRICK ROOM)", {"permanent": "trick_room"}),
            header("S Split:52", "Champion Cynthia (Permanent Gravity)", {"permanent": "gravity"}),
        ],
    ),
)
DRAFT_SHEET = [
    ("S Split:2", 2), ("S Split:3", 3), ("S Split:4", 4), ("S Split:5", None),
    ("S Split:11", 3), ("S Split:12", 12),
    ("S Split:21", 21), ("S Split:22", 22), ("S Split:23", 23),
    ("S Split:24", 24), ("S Split:25", 25), ("S Split:26", 26), ("S Split:27", 27),
    ("S Split:28", 28), ("S Split:29", None),
    ("S Split:31", 31), ("S Split:32", 32), ("S Split:33", 33),
    ("S Split:41", 34), ("S Split:42", 35), ("S Split:43", 36), ("S Split:44", 37),
    ("S Split:45", 38),
    ("S Split:51", 39), ("S Split:52", 1),
]  # fmt: skip
DRAFT_EFFECTS = [
    {"id": 2, "name": "T2", "effect": "trick_room", "map_headers": [305]},
    {"id": 3, "name": "T3", "effect": "trick_room", "map_headers": [307]},
    {"id": 4, "name": "T4", "effect": "trick_room", "map_headers": [305, 306]},
    {"id": 6, "name": "T6", "effect": "trick_room", "map_headers": [306]},
    {"id": 7, "name": "T7", "effect": "trick_room", "map_headers": [306]},
]


@pytest.fixture
def syn_draft(tmp_path):
    m = syn_mappings(
        DRAFT_SHEET,
        DRAFT_EFFECTS,
        starts={3: "unknown", 7: "unknown"},
        pkcalc=[Mapping("Champion Cynthia", 1, "exact", (1,))],
    )
    text = draft_overrides(DRAFT_DOCS, m)
    return text, load_overrides(write(tmp_path, text))


def test_draft_is_loadable_and_cites_clean(syn_draft):
    _, o = syn_draft
    assert check_citations(o, DRAFT_DOCS) == []


def test_draft_rom_trick_room_from_the_rom(syn_draft):
    _, o = syn_draft
    assert o.area[2] == AreaOverride(
        True, False, None, "rom:map header 305 weather 18 (Trick Room)"
    )
    assert o.area[4].source == "rom:map headers 305, 306 weather 18 (Trick Room)"
    assert o.area[6].trick_room  # ROM only, no sheet row
    assert check_rom_trick_room(o, DRAFT_EFFECTS) == []


def test_draft_sheet_only_trick_room_from_the_sheet(syn_draft):
    text, o = syn_draft
    assert "no map of this trainer has weather 18 and start_condition is 'none'" in text
    assert o.area[39] == AreaOverride(True, False, None, "S Split:51")


def test_draft_conflicts_commented_with_both_citations(syn_draft):
    text, o = syn_draft
    assert 3 not in o.area and 7 not in o.area
    assert set(o.area_exceptions) == {3, 7}
    block = text[text.index("# CONFLICT 3") :].split("\n\n")[0]
    assert "S Split:3" in block and "S Split:11" in block and "unknown" in block
    assert "# [area.3]" in block and "# trick_room = true" in block


def test_draft_unmapped_trick_room_header_listed(syn_draft):
    text, _ = syn_draft
    assert "# UNMAPPED S Split:5" in text


def test_draft_gravity_from_pkcalc(syn_draft):
    _, o = syn_draft
    assert o.area[1] == AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611")


def test_draft_gauntlet_groups(syn_draft):
    text, o = syn_draft
    got = {g.trainers: (g.source, g.rules) for g in o.gauntlets}
    assert got == {
        (21, 22, 23): ("S Split:21", {}),
        (24, 25): ("S Split:24", {}),
        (26, 27): ("S Split:26", {}),
        (31, 32): ("S Split:30", {}),  # leader excluded
        (35, 36, 37, 38): ("S Split:40", {}),  # tag partner excluded
    }
    assert "rules = {}" in text
    # a group with an unmapped member is drafted commented out, citing the row
    block = text[text.index("# UNSETTLED gauntlet") :].split("\n\n")[0]
    assert "S Split:28" in block and "S Split:29" in block and "# [[gauntlet]]" in block


def test_draft_gauntlet_without_end_is_unsettled():
    d = docs_of(
        section(
            "S Split:1",
            "Mine",
            [
                header("S Split:2", "A (Gauntlet Start)", {"gauntlet": "start"}),
                header("S Split:3", "B (Gauntlet)", {"gauntlet": "middle"}),
            ],
        )
    )
    text = draft_overrides(d, syn_mappings([("S Split:2", 2), ("S Split:3", 3)]))
    assert "# UNSETTLED gauntlet" in text and "\n[[gauntlet]]" not in text


def test_build_mappings_checks_override_keys():
    pk = {"Leader X": {"party": []}}
    with pytest.raises(MappingError, match="Typo Split:1"):
        build_mappings(pk, SYN, [], [], [], {"Typo Split:1": 5})


# --- real inputs (each skips without them) ----------------------------------------------


@pytest.fixture(scope="module")
def field_effects(kaizo_rom_dir):
    return json.loads((kaizo_rom_dir / "rom_field_effects.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mappings(pkcalc, docs, kaizo_rom_trainers, species, field_effects):
    return build_mappings(
        pkcalc["trainers"],
        docs,
        kaizo_rom_trainers,
        species,
        field_effects,
        load_overrides().trainer_ids,
    )


def test_stale_citation_detected(docs):  # Roark Split:7's header has no TRICK ROOM tag
    o = overrides_with(area={7: AreaOverride(True, False, None, "Roark Split:7")})
    assert check_citations(o, docs) == ["area.7: Roark Split:7 does not say Trick Room"]


def test_rules_left_unset_by_draft(docs, mappings):
    assert "rules = {}" in draft_overrides(docs, mappings)


def test_real_draft_loads_and_cites_clean(docs, mappings, field_effects, tmp_path):
    o = load_overrides(write(tmp_path, draft_overrides(docs, mappings)))
    assert check_citations(o, docs) == []
    assert check_rom_trick_room(o, field_effects) == []
    assert {509, 559, 649} <= set(o.area_exceptions)  # start condition unknown (Task 8)
    assert not {509, 559, 649} & set(o.area)


def test_committed_file_checks_clean(docs):  # skips without the workbook
    assert check_citations(load_overrides(), docs) == []


def test_committed_file_covers_rom_trick_room(field_effects, kaizo_rom_trainers):
    o = load_overrides()
    assert check_rom_trick_room(o, field_effects) == []
    assert check_trainer_ids(o, {t["id"] for t in kaizo_rom_trainers}) == []


def test_committed_override_keys_known(mappings):
    by_label = {m.label: m for m in (*mappings.pkcalc, *mappings.sheet)}
    for label, tid in load_overrides().trainer_ids.items():
        assert (by_label[label].how, by_label[label].id) == ("override", tid)


def test_committed_gauntlet_members_tied_to_their_cite(docs, mappings):
    assert check_gauntlet_members(load_overrides(), docs, mappings.sheet) == []
    probe = overrides_with(gauntlets=(Gauntlet("wrong", (246, 870), {}, "Roark Split:229"),))
    assert len(check_gauntlet_members(probe, docs, mappings.sheet)) == 2


def test_committed_rom_citations_hold(field_effects):
    o = load_overrides()
    assert check_rom_trick_room(o, field_effects) == []
    probe = overrides_with(
        area={246: AreaOverride(True, False, None, "rom:map header 999 weather 18 (Trick Room)")}
    )
    assert any(p.startswith("area.246: ") for p in check_rom_trick_room(probe, field_effects))


def test_trick_room_and_gravity_present():
    o = load_overrides()
    # K2b Task 15 fix round 1: 35 ROM Trick Room trainers with a second, non-Trick-Room
    # placement became area_exceptions (start condition refused); together >= 51 as before.
    tr = {i for i, a in o.area.items() if a.trick_room}
    assert len(tr) >= 19 and len(tr | set(o.area_exceptions)) >= 51
    assert any(a.gravity for a in o.area.values())


def test_committed_file_leaves_unknown_starts_unsettled():
    o = load_overrides()
    unknown = {318, 325, 343, 509, 559, 610, 649, 676, 746, 747}
    assert not unknown & set(o.area)
    assert all(a.weather is None for a in o.area.values())


def test_committed_rules_unset():
    raw = tomllib.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert raw["gauntlet"] and all(g["rules"] == {} for g in raw["gauntlet"])


def test_real_mapping_with_overrides(pkcalc, kaizo_rom_trainers, species):
    m = map_trainers(pkcalc["trainers"], kaizo_rom_trainers, species, load_overrides().trainer_ids)
    assert sum(x.id is not None for x in m) >= 473


def test_fields_format_only_unknown(tmp_path):  # K2b Task 15: two-structure trainers
    ok = '[fields.556]\nformat = "unknown"\nsource = "rom:zone events 257"\n'
    assert load_overrides(write(tmp_path, ok)).fields == {556: {"format": "unknown"}}
    bad = '[fields.556]\nformat = "double"\nsource = "rom:zone events 257"\n'
    with pytest.raises(OverridesError, match="format may only be set"):
        load_overrides(write(tmp_path, bad))
