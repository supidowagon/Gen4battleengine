"""``tools.kaizo.resolve`` (K2b Task 15): the cross-check gate, Ruling 3's names, the
non-ROM value guard and the resolved ``trainers.json``. Synthetic sources, plus checks of
the real resolved intermediates that skip until ``python -m tools.build_tables --game
kaizo`` has run."""

import json
from pathlib import Path

import pytest

from nuzline.data.games import GAMES, KAIZO
from nuzline.data.trainer import GENDER_NONE, AreaCondition
from tools.extract_rom import INPUTS_KEY, MOVE_LISTS_INPUT, kaizo_input_hashes
from tools.kaizo.crosscheck import CrossCheck, CrossCheckError, Outcome, Row
from tools.kaizo.overrides import Overrides
from tools.kaizo.resolve import (
    ResolveError,
    _sha256,
    check_extract_inputs,
    check_used,
    names,
    resolve,
    spelling,
    two_placement_trick_room,
    with_overrides,
)
from tools.rom.inputs import RomInputs

KAIZO_INTER = GAMES[KAIZO].intermediate_dir


# --- synthetic sources ------------------------------------------------------------------


def _move(name, **kw):
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
    return {**m, **kw}


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
    return {**m, **kw}


def _species(name):
    return {
        "name": name,
        "base_stats": [1] * 6,
        "types": [0, 0],
        "abilities": [0, 0],
        "gender_ratio": 255,
        "weight_hg": 1,
        "base_friendship": 70,
        "exp_rate": 0,
        "exp_yield": 1,
        "catch_rate": 1,
    }


def _item(name, **kw):
    it = {"name": name, "hold_effect": 0, "hold_param": 0, "teaches_move": None}
    return {**it, "hp_restored": 0, "fling_power": 0, **kw}


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def fake_inputs():
    """Inputs that point nowhere: the cross-check must stop before it reads the ROM."""
    missing = Path("does-not-exist")
    return RomInputs(missing / "rom.nds", missing / "p.xdelta", None, None, None)


@pytest.fixture
def fake_inter(tmp_path):
    """One move whose ROM power (60) and PKCalc power (40) disagree; nothing else."""
    inter = tmp_path / "inter"
    rom = {
        "moves": [_move("-"), _move("Pound", power=60)],
        "species": [_species("----------")],
        "forms": [],
        "learnsets": {},
        "tm_learnsets": {},
        "tutor_learnsets": {},
        "tutor_moves": [],
        "evolutions": {},
        "evolution_methods": [],
        "items": [_item("None"), _item("TM01", teaches_move=1)],
        "abilities": [{"name": "-"}],
        "trainers": [],
        "rom_field_effects": [],
        "rom_trainer_maps": {},
    }
    for name, data in rom.items():
        _write(inter / "rom" / f"{name}.json", data)
    _write(inter / "rom" / "rom_sources.json", {INPUTS_KEY: kaizo_input_hashes({})})
    pkcalc = {"sha256": "0" * 64, "moves": {"1": _pk_move("Pound")}, "trainers": {}}
    _write(inter / "pkcalc" / "pkcalc.json", pkcalc)
    docs = {"sha256": "1" * 64, "moves": {}, "items": {}, "splits": {}, "raw_trainers": {}}
    _write(inter / "docs" / "docs.json", docs)
    (tmp_path / "overrides.toml").write_text("", encoding="utf-8")
    (tmp_path / "resolutions.toml").write_text("", encoding="utf-8")
    return inter


def _resolve(inputs, tmp_path, inter):
    return resolve(
        inputs,
        inter,
        report=tmp_path / "report.md",
        resolutions=tmp_path / "resolutions.toml",
        overrides=tmp_path / "overrides.toml",
    )


def test_unresolved_stops_the_build(fake_inputs, fake_inter, tmp_path):
    with pytest.raises(CrossCheckError, match="unresolved"):
        _resolve(fake_inputs, tmp_path, fake_inter)
    assert "| unresolved |" in (tmp_path / "report.md").read_text("utf-8")  # written either way
    assert not (fake_inter / "moves.json").exists()  # nothing resolved is written


def test_stale_resolution_stops_the_build(fake_inputs, fake_inter, tmp_path):
    (tmp_path / "resolutions.toml").write_text(
        '[[resolution]]\ntable = "moves"\nkeys = ["1 Pound"]\nfield = "power"\npkcalc = 40\n'
        'use = "rom"\nreason = "r"\nevidence = "e"\n'
        '[[resolution]]\ntable = "moves"\nkeys = ["1 Pound"]\nfield = "pp"\npkcalc = 35\n'
        'use = "rom"\nreason = "r"\nevidence = "e"\n',
        encoding="utf-8",
    )
    with pytest.raises(CrossCheckError, match="stale resolution"):
        _resolve(fake_inputs, tmp_path, fake_inter)


def test_stale_extract_inputs_stop_the_build(fake_inputs, fake_inter, tmp_path):  # M1
    sources = fake_inter / "rom" / "rom_sources.json"
    check_extract_inputs(fake_inter, Overrides())  # recorded = current
    with pytest.raises(ResolveError, match=r"\[move_lists\].*re-run python -m tools.extract_rom"):
        check_extract_inputs(fake_inter, Overrides(move_lists={"sound": [45]}))
    recorded = json.loads(sources.read_text("utf-8"))
    recorded[INPUTS_KEY]["data/kaizo/battle_formats.json"] = "0" * 64  # edited since extract
    _write(sources, recorded)
    with pytest.raises(ResolveError, match="battle_formats.json"):
        _resolve(fake_inputs, tmp_path, fake_inter)
    _write(sources, {})  # extracted before the hashes were recorded
    with pytest.raises(ResolveError, match="no input hashes recorded"):
        check_extract_inputs(fake_inter, Overrides())
    assert MOVE_LISTS_INPUT in kaizo_input_hashes(None)


# --- names (Ruling 3) -------------------------------------------------------------------


def test_spelling_folds_case_punctuation_and_gender_signs():
    assert spelling("NIDORAN♀") == spelling("Nidoran-F")
    assert spelling("King’s Rock") == spelling("King's Rock")
    assert spelling("Rollout") != spelling("Accelerock")


def _named(*rows):
    return [{"name": n} for n in rows]


def _names(rom_moves, vanilla_moves, pk_moves):
    empty = {"species": _named(), "items": _named(), "abilities": _named()}
    pkcalc = {"moves": {str(i): {"name": n} for i, n in pk_moves.items()}}
    return names(
        {**empty, "moves": _named(*rom_moves)}, {**empty, "moves": _named(*vanilla_moves)}, pkcalc
    )


def test_renamed_slot_and_rom_text_alias():
    got = _names(["-", "Water Ball", "ThunderPunch"], ["-", "Comet Punch", "ThunderPunch"],
                 {1: "Weather Ball Water", 2: "Thunder Punch"})  # fmt: skip
    assert got.display["move"] == ["-", "Weather Ball Water", "Thunder Punch"]
    assert got.renamed["move"] == {"comet punch": "Weather Ball Water"}  # spelling is no rename
    assert got.aliases["move"] == {"Water Ball": 1, "ThunderPunch": 2}


def test_vanilla_name_of_a_renamed_slot_is_never_an_alias():
    got = _names(["-", "Rollout"], ["-", "Rollout"], {1: "Accelerock"})
    assert got.renamed["move"] == {"rollout": "Accelerock"}
    assert got.aliases["move"] == {}
    assert any("not an alias" in n for n in got.notes)


def test_alias_naming_another_id_is_an_error():
    with pytest.raises(ResolveError, match="display name of id 1"):
        _names(["-", "Pound", "Pound"], ["-", "Pound", "Tackle"], {2: "Tackle"})


# --- the non-ROM value guard and format corrections ---------------------------------------


def _outcome(field, rom, pkcalc, sheet, kind, used):
    return Outcome(Row("moves", "1 Pound", field, rom, pkcalc, sheet, True), kind, used, None)


def test_non_rom_value_without_a_writer_raises():
    check = CrossCheck([_outcome("power", 60, 40, None, "unexecuted", 40)])
    with pytest.raises(ResolveError, match="power"):
        check_used(check)


def test_names_and_rows_without_a_rom_slot_are_not_written():
    check = CrossCheck(
        [
            _outcome("name", "Water Ball", "Weather Ball Water", None, "unexecuted", "x"),
            _outcome("pp", None, None, 10, "sheet_only", 10),
            _outcome("pp", 35, 35, None, "agree", 35),
        ]
    )
    assert len(check_used(check)) == 2


def test_refusing_overrides_apply_to_the_rom_trainer():
    from tools.kaizo.overrides import AreaException

    rom = [
        {"id": 556, "format": "multi", "opponent_2": 557, "start_condition": "none"},
        {"id": 557, "format": "single", "opponent_2": None, "start_condition": "none"},
        {"id": 627, "format": "single", "opponent_2": None, "start_condition": "none"},
    ]
    ov = Overrides(
        fields={556: {"format": "unknown"}},
        area_exceptions={627: AreaException("two placements", "rom:x")},
    )
    out = with_overrides(rom, ov)
    assert [(t["format"], t["opponent_2"]) for t in out[:2]] == [
        ("unknown", None),
        ("single", None),
    ]
    assert [t["start_condition"] for t in out] == ["none", "none", "unknown"]
    assert rom[0]["format"] == "multi"  # the input is not changed


def test_provenance_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.toml", tmp_path / "crlf.toml"
    lf.write_bytes(b'a = 1\nb = "x"\n')
    crlf.write_bytes(b'a = 1\r\nb = "x"\r\n')
    assert _sha256(lf) == _sha256(crlf)


def test_two_placement_trick_room_rule():
    effects = [
        {"id": 1, "effect": "trick_room", "map_headers": [306]},
        {"id": 2, "effect": "trick_room", "map_headers": [306]},
        {"id": 562, "effect": "trick_room", "map_headers": [306]},  # noisy map join
    ]
    maps = {1: [306, 347], 2: [306], 562: [306, 1, 2, 3]}
    assert two_placement_trick_room(effects, maps) == {1: ([306], [347])}


# --- the real resolved intermediates (skip until built) -----------------------------------


@pytest.fixture(scope="module")
def kaizo_intermediate():
    if not (KAIZO_INTER / "renamed.json").exists():
        pytest.skip("Kaizo intermediates not resolved (python -m tools.build_tables --game kaizo)")
    return KAIZO_INTER


@pytest.fixture(scope="module")
def resolved_trainers(kaizo_intermediate):
    rows = json.loads((kaizo_intermediate / "trainers.json").read_text("utf-8"))
    return {t["id"]: t for t in rows}


def test_renamed_json_in_ruling_3_format(kaizo_intermediate):
    renamed = json.loads((kaizo_intermediate / "renamed.json").read_text("utf-8"))
    assert renamed["move"]["rollout"] == "Accelerock"
    assert set(renamed) == {"species", "move", "item", "ability"}
    assert renamed["item"]["shoal salt"] == "Berserk Gene"
    assert "king’s rock" not in renamed["item"]  # a spelling, not a rename


def test_kaizo_json_aliases_and_provenance(kaizo_intermediate):
    meta = json.loads((kaizo_intermediate / "kaizo.json").read_text("utf-8"))
    assert set(meta) == {"aliases", "provenance"}
    assert meta["aliases"]["move"]["Water Ball"] == 4
    assert meta["aliases"]["species"]["NIDORAN♀"] == 29


def test_every_kaizo_mon_has_explicit_set_data(resolved_trainers):
    for t in resolved_trainers.values():
        assert t["pid_routine"] is None
        for mon in t["party"]:
            assert None not in (mon["nature"], mon["gender"], mon["ability_slot"])
            assert mon["friendship"] == (0 if 218 in (mon["moves"] or []) else 255)


def test_genderless_species_stay_genderless(resolved_trainers, kaizo_intermediate):
    species = json.loads((kaizo_intermediate / "species.json").read_text("utf-8"))
    mons = [m for t in resolved_trainers.values() for m in t["party"]]
    genderless = [m for m in mons if species[m["species"]]["gender_ratio"] == 255]
    assert genderless and all(m["gender"] == GENDER_NONE for m in genderless)


def test_sight_pair_becomes_a_two_trainer_double(resolved_trainers):  # Cyclist Axel / John
    axel = resolved_trainers[25]
    assert (axel["format"], axel["opponent_2"]) == ("double_two_trainer", 27)
    assert axel["sources"]["format"].startswith("rom: map header 350, sight pair")


def test_two_structure_trainers_are_refused(resolved_trainers):
    for tid in (521, 527, 539, 552, 556, 561, 569):
        assert resolved_trainers[tid]["format"] == "unknown", tid


def test_user_call_start_conditions_stay_unknown(resolved_trainers):
    for tid in (318, 325, 343, 509, 559, 610, 649, 676, 746, 747):
        t = resolved_trainers[tid]
        assert t["start_condition"] == "unknown"
        assert t["area_condition"]["weather"] == "unknown"


def test_start_condition_matches_area_condition(resolved_trainers):
    for t in resolved_trainers.values():
        area = AreaCondition(**t["area_condition"])
        want = AreaCondition.from_start_condition(t["start_condition"])
        assert (area.weather, area.fog) == (want.weather, want.fog), t["id"]


def test_rom_trick_room_and_gravity(resolved_trainers):
    assert resolved_trainers[403]["area_condition"]["trick_room"] is True  # Cyrus, header 308
    assert sum(t["area_condition"]["gravity"] for t in resolved_trainers.values()) == 1


def test_split_fields_from_the_sheet_or_unknown(resolved_trainers):
    roark = resolved_trainers[246]
    assert (roark["split"], roark["split_order"]) == ("Roark Split", [0, 272])
    assert roark["sources"]["split"] == "sheet Roark Split:272"
    unplaced = [t for t in resolved_trainers.values() if t["label"] and t["split"] is None]
    assert unplaced and all(t["optional"] is None and t["left_side"] is None for t in unplaced)


def test_json_shapes(resolved_trainers):
    for t in resolved_trainers.values():
        assert t["gauntlet"] is None or len(t["gauntlet"]) == 2
        assert t["split_order"] is None or len(t["split_order"]) == 2
        assert isinstance(t["gauntlet_rules"], dict) and isinstance(t["sources"], dict)
        assert set(t["area_condition"]) == {"weather", "fog", "trick_room", "gravity", "permanent"}


# Derived from rom_trainer_maps.json and rom_field_effects.json (fix round 1): ROM Trick Room
# trainers also placed on a header without Trick Room, the noisy map join excepted.
TWO_PLACEMENT_TRICK_ROOM = {509, 559, 649, *range(539, 553), 556, 561, 569,
                            *range(627, 633), *range(639, 647), 653}  # fmt: skip


def test_two_placement_trick_room_set(kaizo_intermediate):
    rom = kaizo_intermediate / "rom"
    effects = json.loads((rom / "rom_field_effects.json").read_text("utf-8"))
    maps = json.loads((rom / "rom_trainer_maps.json").read_text("utf-8"))
    got = two_placement_trick_room(effects, {int(k): v for k, v in maps.items()})
    assert set(got) == TWO_PLACEMENT_TRICK_ROOM and len(got) == 35


def test_two_placement_trick_room_trainers_are_refused(resolved_trainers):
    for tid in TWO_PLACEMENT_TRICK_ROOM:
        t = resolved_trainers[tid]
        assert t["start_condition"] == "unknown" and not t["area_condition"]["trick_room"], tid


def test_unknown_formats_have_no_opponent_2(resolved_trainers):
    for tid in (*range(539, 553), 556, 561, 569):
        assert resolved_trainers[tid]["opponent_2"] is None, tid


def test_report_lists_user_calls():
    from tools.kaizo.crosscheck import DEFAULT_REPORT

    if not DEFAULT_REPORT.exists():
        pytest.skip("no report")
    text = DEFAULT_REPORT.read_text("utf-8")
    calls = text[text.index("## User calls") :].split("\n## ", 1)[0]
    for needle in ("Trick Room on one placement only: 627", "format unknown: 556",
                   "area weather 107 Tuber Jared", "area weather 291 Idol Catherine #2",
                   "Gravity: 870", "ROM changes unclassified"):  # fmt: skip
        assert needle in calls, needle
