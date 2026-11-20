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
from tools.kaizo.crosscheck import CrossCheck, CrossCheckError, Outcome, Resolution, Row
from tools.kaizo.overrides import AreaOverride, Overrides
from tools.kaizo.resolve import (
    ResolveError,
    _sha256,
    check_extract_inputs,
    check_used,
    names,
    resolve,
    spelling,
    two_placement_trick_room,
    unplaced_ids,
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
        "rom_map_weather": [],
        "rom_script_weather": {"battles": {}, "windows": {}},
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


def test_users_format_answer_applies_before_the_cross_check():
    buck = [{"trainer": 611, "starter": None, "player_gender": None}]
    rom = [
        {"id": 556, "format": "multi", "opponent_2": None, "partner": buck},
        {"id": 521, "format": "unknown", "opponent_2": 527, "partner": None},
        {"id": 202, "format": "multi", "opponent_2": None, "partner": buck},
    ]
    ov = Overrides(
        fields={
            556: {"format": "double_two_trainer", "opponent_2": 557},
            521: {"format": "multi", "opponent_2": 527, "partner": 864},
            202: {"format": "multi"},
        }
    )
    out = with_overrides(rom, ov)
    assert [(t["format"], t["opponent_2"], t["partner"]) for t in out] == [
        ("double_two_trainer", 557, []),  # the ROM's partner goes with its structure
        ("multi", 527, [{"trainer": 864, "starter": None, "player_gender": None}]),
        ("multi", None, buck),  # no partner named: the ROM's stands
    ]
    assert rom[0]["partner"] == buck  # the input is not changed


def test_provenance_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.toml", tmp_path / "crlf.toml"
    lf.write_bytes(b'a = 1\nb = "x"\n')
    crlf.write_bytes(b'a = 1\r\nb = "x"\r\n')
    assert _sha256(lf) == _sha256(crlf)


def test_committed_kaizo_inputs_are_lf_pinned():  # final review M15
    # The provenance and extract-input hashes fold CRLF anyway; the checkout stays LF too.
    import shutil
    import subprocess

    from tools.rom.inputs import REPO_ROOT

    rules = (REPO_ROOT / ".gitattributes").read_text("utf-8").splitlines()
    assert "data/kaizo/*.json text eol=lf" in rules
    if shutil.which("git"):
        files = ["data/kaizo/battle_formats.json", "data/kaizo/start_conditions.json"]
        out = subprocess.run(["git", "check-attr", "eol", *files], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout  # fmt: skip
        assert out.count("eol: lf") == 2


def test_two_placement_trick_room_rule():
    effects = [
        {"id": 1, "effect": "trick_room", "map_headers": [306]},
        {"id": 2, "effect": "trick_room", "map_headers": [306]},
    ]
    maps = {1: [306, 347], 2: [306]}
    assert two_placement_trick_room(effects, maps) == {1: ([306], [347])}


# --- final review fixes: sight pairs, headers, split, unplaced, permanence, paths -------


def _trainer(tid, **kw):
    t = {"id": tid, "name": f"T{tid}", "party": [], "start_condition": "none",
         "format": "single", "opponent_2": None, "class_gender": 0, "class_id": 0}  # fmt: skip
    return {**t, **kw}


def _sight_check(*pairs):
    """A cross-check whose sight_pair rows confirm ``pairs`` (trainer id, partner id)."""
    return CrossCheck(
        [
            Outcome(Row("trainers", f"{a} T{a}", "sight_pair", b, b, None, True), "agree", b, None)
            for a, b in pairs
        ]
    )


def test_one_sided_sight_pair_raises():  # M2
    from tools.kaizo.overrides import Mappings
    from tools.kaizo.resolve import trainer_context

    maps = Mappings((), (), {}, ())
    where = {5: [1], 6: [1]}
    ctx = trainer_context(_sight_check((5, 6), (6, 5)), maps, Overrides(), where, {"splits": {}})
    assert ctx.sight == {5: (6, 1), 6: (5, 1)}
    with pytest.raises(ResolveError, match="confirmed on 5's side only"):
        trainer_context(_sight_check((5, 6)), maps, Overrides(), where, {"splits": {}},
                        {5: "single", 6: "single"})  # fmt: skip
    # A partner whose format is unknown is refused anyway (535-538, 557, 627 today).
    trainer_context(_sight_check((5, 6)), maps, Overrides(), where, {"splits": {}},
                    {5: "single", 6: "unknown"})  # fmt: skip


def _record(cites, fields=None):
    """``trainer_record`` of a partyless trainer 1 with sheet headers ``cites``."""
    from tools.kaizo.resolve import Sources, TrainerContext, trainer_record

    rows = {
        "A Split:10": (0, "A Split", {"tags": {"optional": True}}, {"tags": {}}),
        "A Split:20": (0, "A Split", {"tags": {}}, {"tags": {}, "ai_known": False}),
        "A Split:30": (0, "A Split", {"tags": {}}, {"tags": {"optional": True}}),
    }
    ctx = TrainerContext({}, {1: list(cites)}, {}, {})
    ov = Overrides(fields={1: fields} if fields else {}, sources={"fields.1": "user, 2026-11-07"})
    src = Sources({"learnsets": {}, "species": []}, [], [], {1: [5]}, {}, {})
    return trainer_record(_trainer(1), ctx, ov, src, {}, {}, rows)


def test_headers_that_disagree_give_unknown():  # M3
    agree = _record(["A Split:10", "A Split:30"])
    assert agree["optional"] is True and agree["split_order"] == [0, 10]
    split = _record(["A Split:10", "A Split:20"])
    assert split["optional"] is None
    assert "headers disagree: A Split:10 true, A Split:20 false" in split["sources"]["optional"]
    assert split["split"] == "A Split" and split["split_order"] == [0, 10]


def test_split_override_drops_the_sheet_position():  # M5
    t = _record(["A Split:10"], {"split": "B Split"})
    assert (t["split"], t["split_order"]) == ("B Split", None)
    assert t["sources"]["split_order"].startswith("overrides fields.1 (user, 2026-11-07)")


def test_unplaced_trainers_start_unknown():  # M4
    mon = {"species": 1, "level": 5}
    rom = [
        _trainer(1, party=[mon]),  # placed
        _trainer(2, party=[mon]),  # no join at all
        _trainer(3, party=[mon]),  # a rematch of an unplaced trainer: empty join
        _trainer(4),  # no party: nothing to fight
        _trainer(5, party=[mon]),  # unplaced, settled by the user
    ]
    where = {1: [10], 3: []}
    ov = Overrides(area={5: AreaOverride(False, False, "rain", "user, 2026-11-07")})
    assert unplaced_ids(rom, where) == [2, 3, 5]
    out = with_overrides(rom, ov, where)
    assert [t["start_condition"] for t in out] == ["none", "unknown", "unknown", "none", "none"]
    assert [t["start_condition"] for t in with_overrides(rom, ov)] == ["none"] * 5  # no join


def test_unplaced_trainers_are_one_user_call():  # M4
    from tools.kaizo.resolve import Sources, user_calls

    mon = {"species": 1, "level": 5}
    rom = [_trainer(i, party=[mon]) for i in (1, 2, 3)] + [_trainer(4, start_condition="unknown")]
    src = Sources({}, rom, [], {1: [10]}, {}, {})
    resolved = with_overrides(rom, Overrides(), src.trainer_maps)
    calls = user_calls(resolved, CrossCheck(), Overrides(), src, 0)
    grouped = [c for c in calls if c.startswith("- no map placement found: 2 trainers")]
    assert len(grouped) == 1 and grouped[0].endswith(": 2, 3")
    singles = [c for c in calls if c.startswith("- start condition unknown")]
    want = (
        "- start condition unknown: 4 T4 (data/kaizo/start_conditions.json: its maps start "
        "different conditions)"
    )
    assert singles == [want]


def test_bare_weather_lever_flips_permanence(monkeypatch):  # M16 (Ruling 22)
    from tools.kaizo import resolve as r

    rows = [({"tags": {}}, {"tags": {"weather": "sun", "assumed_permanent": True}})]
    bare = r.bare_tags(rows)
    assert bare == {"sun"}
    assert r._area("sun", None, bare)[1]["permanent"] is True  # user-confirmed 2026-11-05
    monkeypatch.setattr(r, "BARE_WEATHER_PERMANENT", False)
    assert r._area("sun", None, bare)[1]["permanent"] is False
    assert r._area("rain", None, bare)[1]["permanent"] is True  # not the bare tag's weather
    assert r._area("sun", None)[1]["permanent"] is True  # map weather is permanent
    tr = AreaOverride(True, False, None, "user, 2026-11-07")
    room = r.bare_tags([({"tags": {"trick_room": True, "assumed_permanent": True}}, {"tags": {}})])
    assert r._area("none", tr, room)[1]["permanent"] is False


def test_rehash_lines_name_files_not_paths(tmp_path):  # M14
    import hashlib

    from tools.kaizo.resolve import Sources, check_input_hashes

    local = tmp_path / "private" / "PKCalc-main.zip"
    local.parent.mkdir()
    local.write_bytes(b"zip")
    sha = hashlib.sha256(b"zip").hexdigest()
    src = Sources({}, [], [], {}, {"sha256": sha}, {"sha256": "1" * 64})
    lines = check_input_hashes(RomInputs(None, None, local, tmp_path / "absent.xlsx", None), src)
    assert lines[0] == f"PKCalc zip: PKCalc-main.zip re-hashed, matches {sha}"
    assert "absent.xlsx" in lines[1] and not any(str(tmp_path) in line for line in lines)


# --- the real resolved intermediates (skip until built) -----------------------------------


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


# The user's format answers (2026-11-07): the Galactic HQ layout, and Spear Pillar beside Mira.
USER_FORMATS = {
    539: ("double_two_trainer", 540), 540: ("double_two_trainer", 539),
    541: ("double_two_trainer", 542), 542: ("double_two_trainer", 541),
    543: ("double_two_trainer", 544), 544: ("double_two_trainer", 543),
    551: ("double_two_trainer", 552), 552: ("double_two_trainer", 551),
    556: ("double_two_trainer", 557), 557: ("double_two_trainer", 556),
    569: ("double_two_trainer", 627), 627: ("double_two_trainer", 569),
    545: ("double", None), 546: ("double", None), 561: ("double", None),
    547: ("single", None), 548: ("single", None), 549: ("single", None), 550: ("single", None),
    521: ("multi", 527), 527: ("multi", 521),
}  # fmt: skip


def test_two_structure_trainers_take_the_users_format(resolved_trainers):
    for tid, (fmt, opp) in USER_FORMATS.items():
        t = resolved_trainers[tid]
        assert (t["format"], t["opponent_2"]) == (fmt, opp), tid
        assert t["sources"]["format"] == f"overrides fields.{tid} (user, 2026-11-07)", tid
        # Only the Spear Pillar multi has a partner; Buck 611 (Stark Mountain) is dropped.
        want = [{"trainer": 864, "starter": None, "player_gender": None}] if fmt == "multi" else []
        assert t["partner"] == want, tid


def test_user_call_start_conditions(resolved_trainers):
    # 676 Corbin is the Vs. Seeker rematch of 583 Titania (gVsSeekerRematchData), placed on
    # header 211 MT_CORONET_OUTSIDE_SOUTH only (E3 batch 5 shared-script join): hail.
    corbin = resolved_trainers[676]
    assert corbin["start_condition"] == "hail"
    answers = {318: "none", 325: "fog", 343: "none", 610: "rain", 746: "none", 747: "none",
               103: "rain", 491: "rain", 291: "fog", 874: "none", 896: "none"}  # fmt: skip
    for tid, start in answers.items():
        t = resolved_trainers[tid]
        assert t["start_condition"] == start, tid
        assert "overrides area" in t["sources"]["start_condition"], tid
    for tid in (906, 920, 468):  # postgame daily trainers, cut content: still refused
        assert resolved_trainers[tid]["start_condition"] == "unknown", tid
    assert resolved_trainers[870]["area_condition"]["gravity"] is True


def test_start_condition_matches_area_condition(resolved_trainers):
    for t in resolved_trainers.values():
        area = AreaCondition(**t["area_condition"])
        want = AreaCondition.from_start_condition(t["start_condition"])
        assert (area.weather, area.fog) == (want.weather, want.fog), t["id"]


def test_fog_headers_are_the_weather_14_fights(resolved_trainers):
    """``area_condition.fog_headers`` (G-15; rom-map-type sheet): the fog fights on a header of
    d_test.narc weather 14 (ROUTE_210_NORTH 363, CELESTIC_TOWN_CAVE 449). Byron Split's Route
    210 North rows and 913 Cyrus #1; Wake 695 / 696 (start none) and Candice 610 (rain by
    override) are on 363 / 247 but do not start in fog; the deep-fog (15) Hearthome Gym and
    Coronet fights are never clearable (field_map_change.c:275-277)."""
    got = {
        t["id"]: t["area_condition"]["fog_headers"]
        for t in resolved_trainers.values()
        if t["area_condition"]["fog_headers"]
    }
    byron = [66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 78, 79, 81, 441, 442, 489, 490]
    assert {i: got[i] for i in byron} == {i: [363] for i in byron}
    assert got[913] == [449]
    assert set(got) == {*byron, 913, 636, 637, 659, 660, 748, 774, 776}
    for tid in (34, 562, 572, 695, 696, 610):
        assert resolved_trainers[tid]["area_condition"]["fog_headers"] == [], tid


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
        # fog_headers on every trainer (ledger G-15); script_weather and script_window only
        # where a script weather window reaches the fight (G-23, G-24).
        keys = {"weather", "fog", "trick_room", "gravity", "permanent", "fog_headers"}
        assert set(t["area_condition"]) in (keys, keys | {"script_weather", "script_window"})


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


def test_two_placement_trick_room_trainers_take_the_users_answer(resolved_trainers):
    # The user's answer (2026-11-07): the Galactic HQ fight, Trick Room with no weather.
    for tid in TWO_PLACEMENT_TRICK_ROOM:
        t = resolved_trainers[tid]
        assert t["start_condition"] == "none" and t["area_condition"]["trick_room"], tid
        assert "user, 2026-11-07" in t["sources"]["area_condition"], tid


def _unclassified() -> int:
    """The ``unknown`` entries of ``data/kaizo/rom_diff.toml`` (code and members)."""
    import tomllib

    from tools.build_tables import ROM_DIFF

    with ROM_DIFF.open("rb") as f:
        data = tomllib.load(f)
    return sum(e["status"] == "unknown" for t in ("code", "member") for e in data.get(t, []))


def test_report_lists_user_calls():
    from tools.kaizo.crosscheck import DEFAULT_REPORT

    if not DEFAULT_REPORT.exists():
        pytest.skip("no report")
    text = DEFAULT_REPORT.read_text("utf-8")
    calls = text[text.index("## User calls") :].split("\n## ", 1)[0]
    lines = [ln for ln in calls.splitlines() if ln.startswith("- ")]
    assert [ln.split(":", 1)[0] for ln in lines] == [
        "- no map placement found",
        # Iron Island 835/836 left this list (ledger G-23): the ROM's script weather (scr_seq
        # 329 0x01df) starts them in rain, as the sheet and PKCalc say.
        f"- {_unclassified()} ROM changes unclassified in data/kaizo/rom_diff.toml",
    ]
    unplaced = {int(i) for i in lines[0].rsplit(": ", 1)[1].split(", ")}
    assert len(unplaced) == 278 and {906, 920, 468} <= unplaced
    assert not unplaced & {*range(874, 892), 894, 895, 896}


def test_unclassified_line_is_true_at_zero():
    """Task 2 review m7: with no unknown entry left the line no longer says every Kaizo fight
    is refused."""
    from tools.kaizo.resolve import unclassified_line

    assert unclassified_line(0) == (
        "- 0 ROM changes unclassified in data/kaizo/rom_diff.toml: none is left, so no Kaizo "
        "fight is refused for one (Ruling 13)"
    )
    assert unclassified_line(2).endswith(
        "every Kaizo fight is refused until they are classified (Ruling 13)"
    )


def test_user_calls_drop_answered_weather_and_sheet_backed_gravity():
    from tools.kaizo.overrides import Mappings
    from tools.kaizo.resolve import Sources, user_calls
    from tools.kaizo.trainer_map import Mapping

    rom = [_trainer(1, party=[{"species": 1, "level": 5}]), _trainer(2)]
    docs = {"splits": {"S": {"sheet": "S Split", "sections": [
        {"name": "Room", "tags": {}, "trainers": [
            {"cite": "S Split:2", "tags": {"permanent": "gravity"}}]}]}}}  # fmt: skip
    src = Sources({}, rom, [], {1: [10], 2: [10]}, {}, docs)
    row = Row("trainers", "1 T1", "area_weather", "none", "rain", "rain", True)
    res = Resolution("trainers", ("1 T1",), "area_weather", "rain", "rom", "r", "rom: x")
    check = CrossCheck([Outcome(row, "resolved", "none", res)])
    grav = AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611")
    ov = Overrides(area={2: grav})

    def calls(ov, maps=None):
        return [c for c in user_calls(rom, check, ov, src, 0, maps) if "ROM changes" not in c]

    assert [c[:22] for c in calls(ov)] == ["- area weather 1 T1: R", "- Gravity: 2 T2 rests "]
    rain = AreaOverride(False, False, "rain", "user, 2026-11-07")
    maps = Mappings((), (Mapping("S Split:2", 2, "exact", (2,)),), {}, ())
    assert calls(Overrides(area={1: rain, 2: grav}), maps) == []
