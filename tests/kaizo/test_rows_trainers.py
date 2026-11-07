"""Cross-check rows for trainers, AI flags, formats and areas, and the vanilla PID report
(Task 13). Synthetic three-source fixtures, plus real-input checks that skip when the ROM,
PKCalc zip or docs workbook is absent."""

import json
import re

import pytest

from tools.kaizo.crosscheck import compare
from tools.kaizo.overrides import AreaOverride, Mappings, Overrides
from tools.kaizo.rows_trainers import (
    area_section,
    mapping_section,
    pid_section,
    trainer_rows,
)
from tools.kaizo.trainer_map import Mapping


def find(rows, table, key, field):
    hits = [r for r in rows if (r.table, r.key, r.field) == (table, key, field)]
    assert len(hits) == 1, (table, key, field, hits)
    return hits[0]


# --- synthetic sources ------------------------------------------------------------------
# Species 1 (abilities [10, 11], learnset up to level 9), 2 ([20, 0]), 3 ([30, 31]).
# ROM trainers 5 "X" (label "Ace X") and 9 "Y" ("Ace Y"), class 3; X's sheet header is
# "S Split:7" in section "S Split:5".

_ABILITIES = {"10": "Overgrow", "11": "Chlorophyll", "20": "Blaze", "30": "Torrent",
              "31": "Swift Swim", "70": "Drought", "2": "Drizzle"}  # fmt: skip


def _species():
    return {
        "species": [
            {"name": "-", "abilities": [0, 0], "gender_ratio": 255},
            {"name": "ONE", "abilities": [10, 11], "gender_ratio": 31},
            {"name": "TWO", "abilities": [20, 0], "gender_ratio": 127},
            {"name": "THREE", "abilities": [30, 31], "gender_ratio": 0},
        ],
        "learnsets": [[], [[1, 1], [5, 2], [9, 3]], [[1, 2]], [[1, 3]]],
        "items": [{"name": "None"}, {"name": "Charcoal"}],
    }


def _rom_mon(species=1, level=10, **kw):
    m = {
        "species": species,
        "form": 0,
        "level": level,
        "difficulty": 200,
        "ability_override": 0,
        "moves": [1, 2, 0, 0],
        "item": 1,
    }
    m.update(kw)
    return m


def _rom_trainer(tid, name, party, **kw):
    t = {
        "id": tid,
        "name": name,
        "class": "Ace Trainer",
        "class_id": 3,
        "class_gender": 0,
        "ai_flags": 7,
        "items": [],
        "party": party,
        "format": "single",
        "opponent_2": None,
        "partner": [],
        "start_condition": "none",
    }
    t.update(kw)
    return t


def _pk_mon(species=1, level=10, **kw):
    m = {
        "species": species,
        "form": 0,
        "level": level,
        "item": 1,
        "ability": 10,
        "nature": "Hardy",
        "gender": "Male",
        "ivs": [24] * 6,
        "moves": [1, 2],
    }
    m.update(kw)
    return m


def _raw_mon(species=1, level=10, **kw):
    m = {
        "species": species,
        "species_name": "ONE",
        "form": 0,
        "level": level,
        "difficulty": 200,
        "ability_number": 0,
        "item": "Charcoal",
        "moves": ["Move1", "Move2", None, None],
        "ball_seal": 0,
    }
    m.update(kw)
    return m


def _raw(name, party, has_moves=True):
    return {
        "name": name,
        "has_moves": has_moves,
        "has_items": True,
        "class": 3,
        "battle_type_value": 0,
        "battle_type": "Single Battle",
        "items": [None] * 4,
        "ai": [],
        "party": party,
        "cite": "RAW TRAINER DATA:2",
        "party_cite": "RAW TRAINER POKEMON:2",
    }


def _header(text, cite, tags=None, ai=("Basic", "Eval Att", "Expert")):
    return {
        "text": text,
        "name": text,
        "tags": tags or {},
        "party": [{"species": "One", "level": 10}],
        "cite": cite,
        "ai_cells": [],
        "ai_known": ai is not None,
        "ai_flags": list(ai) if ai is not None else None,
        "left_side": False if ai is not None else None,
    }


def synthetic(
    *,
    mappings=None,
    lead_ability=None,
    pkcalc_weather=None,
    rom_condition="none",
    rom_ai=7,
    pkcalc_ai=7,
    rom_format="single",
    rom_opponent_2=None,
    rom_partner=(),
    pkcalc_tag=None,
    pkcalc_double=None,
    rom_difficulty=200,
    sheet_difficulty=200,
    rom_override=0,
    sheet_override=0,
    pk_ability=10,
    rom_moves=(1, 2, 0, 0),
    header_tags=None,
    section_tags=None,
    sheet_ai=("Basic", "Eval Att", "Expert"),
    x_label="Ace X",
    fields=None,
):
    """``(rom, mappings, pkcalc, docs, species, overrides)`` for ``trainer_rows``."""
    rom = [
        _rom_trainer(
            5,
            "X",
            [
                _rom_mon(
                    difficulty=rom_difficulty,
                    ability_override=rom_override,
                    moves=None if rom_moves is None else list(rom_moves),
                )
            ],
            ai_flags=rom_ai,
            format=rom_format,
            opponent_2=rom_opponent_2,
            partner=list(rom_partner) if rom_partner is not None else None,
            start_condition=rom_condition,
        ),
        _rom_trainer(9, "Y", [_rom_mon(species=3, level=12)]),
    ]
    x_mon = _pk_mon(ability=lead_ability_id(lead_ability) if lead_ability else pk_ability)
    pkcalc = {
        "trainers": {
            x_label: {"party": [x_mon]},
            "Ace Y": {"party": [_pk_mon(species=3, level=12, ability=30)]},
        },
        "ai_flags": {x_label: pkcalc_ai, "Ace Y": 7},
        "formats": {"true_double": [], "double": pkcalc_double or [], "tag": pkcalc_tag or []},
        "weather": {"": [], "clear": [], **(pkcalc_weather or {})},
        "hidden": [],
        "flag_labels_without_party": [],
        "abilities": {k: {"name": v} for k, v in _ABILITIES.items()},
    }
    docs = {
        "moves": {"1": {"name": "Move1"}, "2": {"name": "Move2"}, "3": {"name": "Move3"}},
        "items": {"Charcoal": {"sheet_id": 1}},
        "raw_trainers": {
            "5": _raw(
                "X",
                [_raw_mon(difficulty=sheet_difficulty, ability_number=sheet_override)],
            ),
            "9": _raw("Y", [_raw_mon(species=3, level=12, species_name="THREE")]),
        },
        "splits": {
            "S": {
                "sheet": "S Split",
                "sections": [
                    {
                        "name": "Area",
                        "text": "Area",
                        "tags": section_tags or {},
                        "cite": "S Split:5",
                        "trainers": [
                            _header("Ace X", "S Split:7", header_tags, sheet_ai),
                            _header("Ace Y", "S Split:17"),
                        ],
                    }
                ],
            }
        },
    }
    if mappings is None:
        mappings = [Mapping(x_label, 5, "exact", (5,)), Mapping("Ace Y", 9, "exact", (9,))]
    bundle = Mappings(
        pkcalc=tuple(mappings),
        sheet=(Mapping("S Split:7", 5, "exact", (5,)), Mapping("S Split:17", 9, "exact", (9,))),
        rom_trainers={t["id"]: t for t in rom},
        field_effects=(),
    )
    overrides = Overrides(fields=fields or {}, sources={})
    return rom, bundle, pkcalc, docs, _species(), overrides


def lead_ability_id(name):
    return int(next(k for k, v in _ABILITIES.items() if v == name))


# --- brief tests --------------------------------------------------------------------------


def test_ability_weather_not_an_area_claim():  # PKCalc's sun list holds a trainer whose lead has Drought
    rows = trainer_rows(
        *synthetic(lead_ability="Drought", pkcalc_weather={"sun": ["Ace X"]}, rom_condition="none")
    )
    row = find(rows, "trainers", "5 Ace X", "area_weather")
    assert (row.rom, row.pkcalc, row.sheet) == ("none", None, None)


def test_unmapped_label_row_is_executed():
    rows = trainer_rows(*synthetic(mappings=[Mapping("Lady Emi", None, "unmapped", ())]))
    row = find(rows, "trainers", "Lady Emi", "mapped")
    assert row.executed and row.rom is None
    assert [r.key for r in compare(rows, []).unresolved] == ["Lady Emi"]


def test_ai_flag_bits():
    row = find(trainer_rows(*synthetic(rom_ai=23, pkcalc_ai=7)), "trainers", "5 Ace X", "ai_flags")
    assert (row.rom, row.pkcalc, row.executed) == (23, 7, True)


def test_format_rows_from_rom_structure():  # ROM multi with opponent_2 9; PKCalc tag pair ["Ace X", "Ace Y"]
    rows = trainer_rows(
        *synthetic(
            rom_format="multi", rom_opponent_2=9, pkcalc_tag=[["Ace X", "Ace Y", "{RIVAL2_NAME}"]]
        )
    )
    assert find(rows, "trainers", "5 Ace X", "format").pkcalc == "multi"
    assert find(rows, "trainers", "5 Ace X", "opponent_2").pkcalc == 9  # "Ace Y" mapped to 9
    assert find(rows, "trainers", "5 Ace X", "partner_present").pkcalc is True


def test_sheet_difficulty_is_rom_vs_sheet():
    out = compare(trainer_rows(*synthetic(rom_difficulty=200, sheet_difficulty=180)), []).outcomes
    assert next(o for o in out if o.row.field == "difficulty").kind == "rom_vs_sheet"


# --- more synthetic checks ----------------------------------------------------------------


def test_synthetic_trainer_agrees():
    check = compare(trainer_rows(*synthetic()), [])
    assert {o.kind for o in check.outcomes} == {"agree"}
    fields = {o.row.field for o in check.outcomes if o.row.key.startswith("5 Ace X")}
    assert {"party_size", "species", "form", "level", "item", "moves", "ivs", "difficulty",
            "ability_override", "ai_flags", "format", "opponent_2", "partner_present",
            "area_weather"} <= fields  # fmt: skip


def test_mon_rows_keyed_by_slot():
    row = find(trainer_rows(*synthetic()), "trainers", "5 Ace X/0", "ivs")
    assert (row.rom, row.pkcalc, row.executed) == ([24] * 6, [24] * 6, True)


def test_moves_without_rom_moves_are_default_moves():  # level 10: learnset 1, 2, 3
    rows = trainer_rows(*synthetic(rom_moves=None))
    row = find(rows, "trainers", "5 Ace X/0", "moves")
    assert (row.rom, row.pkcalc) == ([1, 2, 3], [1, 2])


def test_ability_override_18_vs_sheet_2_is_rom_vs_sheet():
    out = compare(trainer_rows(*synthetic(rom_override=18, sheet_override=2)), []).outcomes
    o = next(o for o in out if o.row.field == "ability_override")
    assert (o.row.rom, o.row.pkcalc, o.row.sheet, o.kind) == (18, None, 2, "rom_vs_sheet")


@pytest.mark.parametrize(
    ("override", "species_slot2", "want"),
    [(16, 11, 10), (32, 11, 11), (18, 11, 10), (32, 0, 10)],
)
def test_ability_slot_row_from_override(override, species_slot2, want):
    rom, maps, pk, docs, species, o = synthetic(rom_override=override, pk_ability=11)
    species["species"][1]["abilities"] = [10, species_slot2]
    rows = trainer_rows(rom, maps, pk, docs, species, o)
    row = find(rows, "trainers", "5 Ace X/0", "ability_slot")
    assert (row.rom, row.pkcalc, row.executed) == (want, 11, True)


def test_ability_slot_two_on_single_ability_species_is_slot_one():
    rom, maps, pk, docs, species, o = synthetic(rom_override=32, pk_ability=10)
    species["species"][1]["abilities"] = [10, 0]
    row = find(
        trainer_rows(rom, maps, pk, docs, species, o), "trainers", "5 Ace X/0", "ability_slot"
    )
    assert (row.rom, row.pkcalc) == (10, 10)


def test_no_ability_slot_row_without_ability_nibble():
    rows = trainer_rows(*synthetic(rom_override=2))
    assert not [r for r in rows if r.field == "ability_slot"]


def test_unknown_ai_is_none_never_off():  # Task 7: ai_known false
    row = find(trainer_rows(*synthetic(sheet_ai=None)), "trainers", "5 Ace X", "ai_flags")
    assert row.sheet is None


def test_sheet_ai_cells_are_bits():
    row = find(
        trainer_rows(*synthetic(sheet_ai=("Basic", "Risky", "Left Side AI"))),
        "trainers",
        "5 Ace X",
        "ai_flags",
    )
    assert row.sheet == 0b10001


def test_double_pair_opponent_2():
    rows = trainer_rows(
        *synthetic(
            rom_format="double_two_trainer",
            rom_opponent_2=9,
            pkcalc_double=[["Ace X", "Ace Y"]],
        )
    )
    row = find(rows, "trainers", "5 Ace X", "opponent_2")
    assert (row.rom, row.pkcalc) == (9, 9)
    assert find(rows, "trainers", "5 Ace X", "format").pkcalc == "double_two_trainer"
    assert find(rows, "trainers", "5 Ace X", "partner_present").pkcalc is False


def test_null_partner_is_present():  # Task 8: 25 partners the rules cannot evaluate
    row = find(trainer_rows(*synthetic(rom_partner=None)), "trainers", "5 Ace X", "partner_present")
    assert row.rom is True


def test_sheet_tag_battle_tags():
    rows = trainer_rows(
        *synthetic(
            rom_format="multi",
            rom_opponent_2=9,
            pkcalc_tag=[["Ace X", "Ace Y", "{RIVAL2_NAME}"]],
            header_tags={"tag_partner": True, "multi_with": "ACE Y"},
        )
    )
    assert find(rows, "trainers", "5 Ace X", "format").sheet == "multi"
    assert find(rows, "trainers", "5 Ace X", "opponent_2").sheet == 9
    assert find(rows, "trainers", "5 Ace X", "partner_present").sheet is True


def test_sheet_multi_with_unknown_name_is_reported_not_guessed():
    rows = trainer_rows(*synthetic(header_tags={"multi_with": "Ace Zed"}))
    assert find(rows, "trainers", "5 Ace X", "sight_pair").sheet == "name: Ace Zed"
    assert find(rows, "trainers", "5 Ace X", "format").sheet is None


# --- sight pairs (fix round 1, I1) -------------------------------------------------------

_DOUBLE_PAIR = [["Ace X", "Ace Y"]]


def test_sight_pair_sharing_a_header_agrees():  # Cyclist Axel/John on Route 206 (350)
    src = synthetic(pkcalc_double=_DOUBLE_PAIR)
    where = {5: [350, 351], 9: [350]}
    rows = trainer_rows(*src, trainer_maps=where)
    row = find(rows, "trainers", "5 Ace X", "sight_pair")
    assert (row.rom, row.pkcalc, row.executed) == (9, 9, True)
    assert not [r for r in rows if r.key == "5 Ace X" and r.field in ("format", "opponent_2")]
    assert compare(rows, []).unresolved == []
    lines = area_section(src[0], src[1], src[2], src[3], src[5], where)
    assert any(
        ln.startswith("sight_pair: 5 Ace X: ROM 9") and "rom: map header 350, sight pair" in ln
        for ln in lines
    )


def test_sight_pair_without_a_shared_header_is_unresolved():
    rows = trainer_rows(*synthetic(pkcalc_double=_DOUBLE_PAIR), trainer_maps={5: [1], 9: [2]})
    row = find(rows, "trainers", "5 Ace X", "sight_pair")
    assert (row.rom, row.pkcalc) == (None, 9)
    assert [(r.key, r.field) for r in compare(rows, []).unresolved] == [
        ("5 Ace X", "sight_pair"),
        ("9 Ace Y", "sight_pair"),
    ]


def test_rom_pair_against_pkcalc_single_disagrees():
    rows = trainer_rows(*synthetic(rom_format="double_two_trainer", rom_opponent_2=9))
    row = find(rows, "trainers", "5 Ace X", "opponent_2")
    assert (row.rom, row.pkcalc) == (9, "none")
    assert ("5 Ace X", "opponent_2") in [(r.key, r.field) for r in compare(rows, []).unresolved]


def test_no_second_opponent_is_none_not_absent():
    row = find(trainer_rows(*synthetic()), "trainers", "5 Ace X", "opponent_2")
    assert (row.rom, row.pkcalc, row.sheet) == ("none", "none", None)


def test_walk_multi_keeps_its_format_row_beside_the_sight_pair():
    rows = trainer_rows(
        *synthetic(rom_format="multi", pkcalc_tag=[["Ace X", "Ace Y", "Cheryl"]]),
        trainer_maps={5: [7], 9: [7]},
    )
    assert find(rows, "trainers", "5 Ace X", "format").rom == "multi"
    assert find(rows, "trainers", "5 Ace X", "sight_pair").rom == 9


def test_unknown_start_condition_and_format_are_not_rows():
    rows = trainer_rows(*synthetic(rom_condition="unknown", rom_format="unknown"))
    fields = {r.field for r in rows if r.key == "5 Ace X"}
    assert not fields & {"area_weather", "format", "opponent_2", "partner_present"}


def test_sheet_area_weather_from_section():
    rows = trainer_rows(
        *synthetic(rom_condition="sandstorm", section_tags={"permanent": "sandstorm"})
    )
    row = find(rows, "trainers", "5 Ace X", "area_weather")
    assert (row.rom, row.pkcalc, row.sheet) == ("sandstorm", None, "sandstorm")


def test_sheet_weather_header_tag_wins_over_section():
    rows = trainer_rows(
        *synthetic(
            rom_condition="rain",
            section_tags={"permanent": "sandstorm"},
            header_tags={"weather": "rain", "assumed_permanent": True},
        )
    )
    assert find(rows, "trainers", "5 Ace X", "area_weather").sheet == "rain"
    # Ace Y has no tag of its own: its section's tag spreads to it.
    assert find(rows, "trainers", "9 Ace Y", "area_weather").sheet == "sandstorm"


def test_pkcalc_sand_list_is_sandstorm():
    rows = trainer_rows(*synthetic(rom_condition="sandstorm", pkcalc_weather={"sand": ["Ace X"]}))
    assert find(rows, "trainers", "5 Ace X", "area_weather").pkcalc == "sandstorm"


def test_partner_weather_ability_dropped():  # Ace Y is the tag pair's other lead, with Drizzle
    rom, maps, pk, docs, species, o = synthetic(
        rom_format="multi",
        rom_opponent_2=9,
        pkcalc_tag=[["Ace X", "Ace Y", "{RIVAL2_NAME}"]],
        pkcalc_weather={"rain": ["Ace X"]},
    )
    pk["trainers"]["Ace Y"]["party"][0]["ability"] = 2
    row = find(trainer_rows(rom, maps, pk, docs, species, o), "trainers", "5 Ace X", "area_weather")
    assert row.pkcalc is None


def test_player_starter_from_bracket_and_sheet():
    label = "Pokémon Trainer Barry #2 [Piplup]"  # Barry has Piplup: the player chose Chimchar
    rows = trainer_rows(*synthetic(x_label=label, header_tags={"player_starter": 390}))
    row = find(rows, "trainers", f"5 {label}", "player_starter")
    assert (row.rom, row.pkcalc, row.sheet) == (390, 390, 390)


def test_player_starter_disagreement_needs_override():
    label = "Pokémon Trainer Barry #2 [Piplup]"
    src = synthetic(x_label=label, header_tags={"player_starter": 387})
    assert [r.field for r in compare(trainer_rows(*src), []).unresolved] == ["player_starter"]
    fixed = synthetic(
        x_label=label, header_tags={"player_starter": 387}, fields={5: {"player_starter": 390}}
    )
    assert compare(trainer_rows(*fixed), []).unresolved == []


def test_rival2_bracket_names_its_own_line():  # Dawn has Grotle: the player chose Chimchar
    label = "Pokémon Trainer Dawn @ Jubilife City [Grotle]"
    row = find(trainer_rows(*synthetic(x_label=label)), "trainers", f"5 {label}", "player_starter")
    assert row.pkcalc == 390


def test_rom_partner_variant_starter_is_the_rom_value():
    label = "Pokémon Trainer Dawn @ Jubilife City [Grotle]"
    rom, maps, pk, docs, species, o = synthetic(x_label=label)
    rom[1]["partner"] = [{"trainer": 5, "starter": 390, "player_gender": 0}]
    row = find(
        trainer_rows(rom, maps, pk, docs, species, o), "trainers", f"5 {label}", "player_starter"
    )
    assert (row.rom, row.pkcalc) == (390, 390)


def test_unknown_rival_bracket_raises():
    with pytest.raises(ValueError, match="Bulbasaur"):
        trainer_rows(*synthetic(x_label="Pokémon Trainer Barry #2 [Bulbasaur]"))


def test_form_row_reports_form_difference():
    rom, maps, pk, docs, species, o = synthetic()
    rom[0]["party"][0]["form"] = 1
    row = find(trainer_rows(rom, maps, pk, docs, species, o), "trainers", "5 Ace X/0", "form")
    assert (row.rom, row.pkcalc, row.executed) == (1, 0, True)


def test_no_item_is_zero_in_every_source():
    rom, maps, pk, docs, species, o = synthetic()
    rom[0]["party"][0]["item"] = 0
    pk["trainers"]["Ace X"]["party"][0]["item"] = None
    docs["raw_trainers"]["5"]["party"][0]["item"] = None
    row = find(trainer_rows(rom, maps, pk, docs, species, o), "trainers", "5 Ace X/0", "item")
    assert (row.rom, row.pkcalc, row.sheet) == (0, 0, 0)


def test_mapping_section_lists_non_exact_and_unmapped():
    _, maps, *_ = synthetic(
        mappings=[
            Mapping("Ace X", 5, "pid", (5, 6)),
            Mapping("Ace Y", 9, "form", (9,), ((0, 0, 1),)),
            Mapping("Lady Emi", None, "unmapped", (7, 8)),
        ]
    )
    lines = mapping_section(maps)
    assert any("Ace X" in ln and "pid" in ln and "[5, 6]" in ln for ln in lines)
    assert any("Ace Y" in ln and "form" in ln for ln in lines)
    assert any("Lady Emi" in ln and "unmapped" in ln and "[7, 8]" in ln for ln in lines)
    assert not any("S Split:7" in ln for ln in lines)  # exact: not listed


def _pid_sources():
    rom, maps, pk, _docs, species, _o = synthetic()
    return rom, maps, pk, species


def test_pid_section_counts_and_override_lines():
    from nuzline.data.trainer import gender_of, trainer_pid
    from tools.kaizo.trainer_map import NATURES

    rom, maps, pk, species = _pid_sources()
    pid = trainer_pid(200, 10, 1, 5, 3, female_class=False)
    pk["trainers"]["Ace X"]["party"][0].update(
        nature=NATURES[pid % 25],
        gender="Female" if gender_of(31, pid) else "Male",
        ability=11 if pid & 1 else 10,
    )
    rom[1]["party"][0]["ability_override"] = 16
    lines = pid_section(rom, maps, pk, species)
    assert lines[0] == "nature and gender reproduced for 1 of 1 override-0 Pokemon"
    assert lines[1] == "ability reproduced for 1 of 1"
    assert lines[2] == "trainers reproduced completely: 1 of 1"
    assert any(ln.startswith("not checked: override byte 16") and "9 Ace Y/0" in ln for ln in lines)


def test_area_section_lists_refused_unknowns_and_flag_labels():
    rom, maps, pk, docs, _species, o = synthetic(rom_condition="unknown", rom_format="unknown")
    pk["flag_labels_without_party"] = ["Maid Yoki"]
    pk["hidden"] = ["Maid Yoki"]
    lines = area_section(rom, maps, pk, docs, o)
    i = lines.index("refused: unknown (user call)")
    assert any("5 X" in ln and "start_condition" in ln for ln in lines[i:])
    assert any("5 X" in ln and "format" in ln for ln in lines[i:])
    assert any("Maid Yoki" in ln for ln in lines)


def test_area_section_marks_pkcalc_only_starters():
    label = "Pok\u00e9mon Trainer Barry #2 [Piplup]"
    rom, maps, pk, docs, _species, o = synthetic(x_label=label)
    lines = area_section(rom, maps, pk, docs, o)
    line = next(ln for ln in lines if ln.startswith(f"player_starter: 5 {label}"))
    assert line.endswith("from pkcalc bracket (PKCalc only, unconfirmed)")


def test_mapping_section_lists_unmapped_headers():
    _, maps, *_ = synthetic()
    maps = Mappings(maps.pkcalc, (Mapping("S Split:7", None, "unmapped", ()),), {}, ())
    want = "unmapped sheet headers (non-blocking, sheet_only by controller ruling): 1: S Split:7"
    assert want in mapping_section(maps)


def test_area_section_trick_room_sheet_only():  # the Hooded Figure shape: sheet yes, ROM no
    rom, maps, pk, docs, _species, _o = synthetic(header_tags={"permanent": "trick_room"})
    o = Overrides(area={5: AreaOverride(True, False, None, "S Split:7")})
    lines = area_section(rom, maps, pk, docs, o)
    line = next(ln for ln in lines if ln.startswith("trick_room: 5 X"))
    assert "ROM -" in line and "sheet S Split:7" in line and "differs" in line


# --- real inputs (skip when absent) ----------------------------------------------------


@pytest.fixture(scope="module")
def real_sources(kaizo_rom_dir, pkcalc, docs):
    from tools.kaizo.overrides import build_mappings, load_overrides
    from tools.kaizo.rows_tables import load_rom

    rom = json.loads((kaizo_rom_dir / "trainers.json").read_text(encoding="utf-8"))
    effects = json.loads((kaizo_rom_dir / "rom_field_effects.json").read_text(encoding="utf-8"))
    species = load_rom(kaizo_rom_dir)
    overrides = load_overrides()
    maps = build_mappings(
        pkcalc["trainers"], docs, rom, species["species"], effects, overrides.trainer_ids
    )
    where = json.loads((kaizo_rom_dir / "rom_trainer_maps.json").read_text(encoding="utf-8"))
    return rom, maps, pkcalc, docs, species, overrides, {int(k): v for k, v in where.items()}


@pytest.fixture(scope="module")
def real_rows(real_sources):
    return trainer_rows(*real_sources)


@pytest.fixture(scope="module")
def real_sections(real_sources):
    rom, maps, pk, docs, species, overrides, where = real_sources
    return {
        "mapping": mapping_section(maps),
        "pid": pid_section(rom, maps, pk, species),
        "area": area_section(rom, maps, pk, docs, overrides, where),
    }


def test_roark_rows_agree(real_rows):  # preflight #29: through the policy
    assert not [
        o
        for o in compare(real_rows, []).outcomes
        if o.row.key.startswith("246 ") and o.kind in ("resolved", "unresolved")
    ]


def test_pid_section_format(real_sections):
    lines = real_sections["pid"]
    assert "Leader Roark (246): 6/6" in lines
    assert re.fullmatch(r"nature and gender reproduced for \d+ of \d+ override-0 Pokemon", lines[0])
    assert re.fullmatch(r"ability reproduced for \d+ of \d+", lines[1])
    assert re.fullmatch(r"trainers reproduced completely: \d+ of \d+", lines[2])


def test_pid_carry_over_model_leaves_no_nature_miss(real_sections):  # fix round 1, I3
    line = next(
        ln for ln in real_sections["pid"] if ln.startswith("nature with the patched genderMod")
    )
    n, m = map(int, re.fullmatch(r".*: (\d+) of (\d+)", line).groups())
    assert n == m > 2000


def test_real_sight_pairs_cite_shared_headers(real_rows, real_sections):
    row = find(real_rows, "trainers", "25 Cyclist Axel", "sight_pair")
    assert (row.rom, row.pkcalc) == (27, 27)
    assert any(
        ln.startswith("sight_pair: 25 Cyclist Axel: ROM 27 (rom: map header 350, sight pair")
        for ln in real_sections["area"]
    )


def test_real_self_pairs_agree_as_doubles(real_rows):  # fix round 1, I2: Volkner
    assert find(real_rows, "trainers", "320 Leader Volkner", "format").rom == "double"
    row = find(real_rows, "trainers", "320 Leader Volkner", "opponent_2")
    assert (row.rom, row.pkcalc) == ("none", "none")


def test_real_override_18_is_rom_vs_sheet(real_rows):  # Task 8: one mon 18 where the sheet says 2
    out = [
        o
        for o in compare(real_rows, []).outcomes
        if o.row.field == "ability_override" and o.row.rom == 18
    ]
    assert [(o.row.sheet, o.kind) for o in out] == [(2, "rom_vs_sheet")]


def test_real_area_section(real_sections):
    lines = real_sections["area"]
    i = lines.index("refused: unknown (user call)")
    refused = " ".join(lines[i:])
    for tid in (318, 325, 343, 509, 559, 610, 649, 676, 746, 747, 521, 527):
        assert f"{tid} " in refused
    assert any(ln.startswith("trick_room: 666 ") and "ROM -" in ln for ln in lines)
    assert any("Maid Yoki" in ln for ln in lines) and any("Skier Isaiah" in ln for ln in lines)


def test_real_every_pkcalc_label_has_rows(real_rows, real_sources):
    maps = real_sources[1]
    keys = {r.key for r in real_rows if r.field == "party_size"}
    assert keys == {f"{m.id} {m.label}" for m in maps.pkcalc if m.id is not None}
