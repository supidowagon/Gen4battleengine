"""PKCalc extractor: the hardened evaluation, the zip checks and the normalised data."""

import shutil
import time
import zipfile

import pytest

from tools.extract_decomp import ExtractError
from tools.extract_pkcalc import extract_pkcalc, normalise, run_eval
from tools.rom.inputs import PKCALC_SHA256

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")


def make_zip(tmp_path, members):
    path = tmp_path / "pkcalc.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in members.items():
            z.writestr(name, text)
    return path


@needs_node
def test_eval_has_no_host_globals(tmp_path):
    js = tmp_path / "x.js"
    js.write_text(
        "const X = {a: typeof require, b: typeof process, c: typeof setTimeout,"
        " d: typeof console, e: typeof queueMicrotask, f: typeof fetch};"
    )
    assert set(run_eval(js, "X").values()) == {"undefined"}


@needs_node
def test_let_and_const_become_globals(tmp_path):
    js = tmp_path / "x.js"
    js.write_text("let A = 1;\nconst X = {a: A, b: [1, 'two']};\n")
    assert run_eval(js, "X") == {"a": 1, "b": [1, "two"]}


@needs_node
def test_constructor_escape_throws(tmp_path):
    js = tmp_path / "x.js"
    js.write_text("const X = ({}).constructor.constructor('return process')();")
    with pytest.raises(ExtractError, match="EvalError|Code generation"):
        run_eval(js, "X")


@needs_node
def test_eval_time_limit(tmp_path):
    js = tmp_path / "x.js"
    js.write_text("const X = 1; while (true) {}")
    with pytest.raises(ExtractError, match="timed out"):
        run_eval(js, "X", timeout_ms=300)


@needs_node
def test_serialisation_is_time_limited(tmp_path):
    js = tmp_path / "x.js"
    js.write_text("const X = {toJSON() { while (true) {} }};")
    with pytest.raises(ExtractError, match="timed out"):
        run_eval(js, "X", timeout_ms=300)


@needs_node
@pytest.mark.parametrize(
    "tail",
    [
        "Promise.resolve().then(() => { while (true) {} });",
        "Promise.reject(1).catch(() => { while (true) {} });",
    ],
)
def test_microtasks_are_time_limited(tmp_path, tail):
    js = tmp_path / "x.js"
    js.write_text("const X = 1; " + tail)
    start = time.monotonic()
    with pytest.raises(ExtractError, match="timed out"):
        run_eval(js, "X", timeout_ms=300)
    assert time.monotonic() - start < 30  # the limit, not the outer subprocess timeout


@needs_node
@pytest.mark.parametrize(
    "thrown",
    [
        "{get name() { while (true) {} }}",
        "{get message() { while (true) {} }, toString() { while (true) {} }}",
        (
            "new Proxy({}, {get() { while (true) {} }, "
            "getOwnPropertyDescriptor() { while (true) {} }})"
        ),
        "{get stack() { while (true) {} }}",
        "new Proxy({}, {get() { while (true) {} }})",
        "'a string'",
    ],
)
def test_thrown_value_is_never_run(tmp_path, thrown):
    js = tmp_path / "x.js"
    js.write_text(f"const X = 1; throw {thrown};")
    start = time.monotonic()
    with pytest.raises(ExtractError, match="non-Error value"):
        run_eval(js, "X", timeout_ms=300)
    assert time.monotonic() - start < 30


def test_node_missing_named(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ExtractError, match="node"):
        run_eval(tmp_path / "x.js", "X")


def test_zip_slip_refused(tmp_path):
    with pytest.raises(ExtractError, match=r"\.\./sets\.js"):
        extract_pkcalc(make_zip(tmp_path, {"../sets.js": "var SETDEX_PK = {};"}), tmp_path / "out")


def test_absolute_member_refused(tmp_path):
    zip_path = make_zip(tmp_path, {"/pkcalc/src/js/data/sets.js": "var SETDEX_PK = {};"})
    with pytest.raises(ExtractError, match="absolute"):
        extract_pkcalc(zip_path, tmp_path / "out")


def test_missing_member_named(tmp_path):
    zip_path = make_zip(tmp_path, {"pkcalc/src/js/data/sets.js": "var SETDEX_PK = {};"})
    with pytest.raises(ExtractError, match="src/js/data/dex/species.js"):
        extract_pkcalc(zip_path, tmp_path / "out", cache=tmp_path / "cache")


def test_two_top_folders_refused(tmp_path):
    zip_path = make_zip(tmp_path, {"a/x.js": "", "b/y.js": ""})
    with pytest.raises(ExtractError, match="single top folder"):
        extract_pkcalc(zip_path, tmp_path / "out")


def _raw(**over):
    """A minimal raw PKCalc data set: one species, one move, one trainer."""
    raw = {
        "SPECIES": {
            "pichu": {
                "num": 172,
                "id": "pichu",
                "name": "Pichu",
                "baseStats": {"hp": 20, "atk": 40, "def": 15, "spa": 35, "spd": 35, "spe": 60},
                "types": ["electric"],
                "catchRate": 190,
                "expYield": 42,
                "genderRatio": "50% Male, 50% Female",
                "growthRate": "Medium Fast",
                "abilities": ["static"],
                "evos": [],
                "learnset": [{"move": "pound", "method": "level", "level": 1}],
                "baseSpecies": "",
            }
        },
        "MOVES": {
            "pound": {
                "num": 1,
                "name": "Pound",
                "shortName": "Pound",
                "calcName": "Pound",
                "type": "normal",
                "category": "physical",
                "basePower": 40,
                "accuracy": 100,
                "pp": 35,
                "secondaryChance": 0,
                "priority": 0,
                "target": "normal",
                "effect": "",
                "machine": 0,
                "tutor": "",
                "price": [],
            }
        },
        "ABILITIES": {"static": {"num": 9, "name": "Static", "calcName": "Static", "desc": {}}},
        "ITEMS": {},
        "NATURES": {"hardy": {"num": 0, "name": "Hardy"}},
        "TYPES": {},
        "SETDEX_PK": {
            "Pichu": {
                "Ace X": {
                    "ability": "Static",
                    "gender": "",
                    "ivs": {"hp": 1, "at": 2, "df": 3, "sa": 4, "sd": 5, "sp": 6},
                    "level": 5,
                    "moves": ["Pound"],
                    "nature": "Hardy",
                }
            }
        },
        "PARTY_ORDER_PK": {"Ace X": ["Pichu"], "Rival [{RIVAL_STARTER_1}]": []},
        "TRAINER_AI_FLAGS_PK": {
            "Ace X": {
                "Basic": 1,
                "EvaluateAttack": 0,
                "Expert": 0,
                "Setup": 0,
                "Risky": 0,
                "DamagePriority": 0,
                "BatonPass": 0,
                "TagStrategy": 0,
                "CheckHP": 0,
                "Weather": 0,
                "Harassment": 1,
            }
        },
        "FLAGS_PK": {
            "battleType": {"double": [], "trueDouble": {"Ace X": {}}, "tag": []},
            "hidden": [],
            "weather": {"rain": ["Ace X"]},
        },
    }
    raw.update(over)
    return raw


def test_normalise_synthetic():
    data = normalise(_raw(), "0" * 64)
    assert data["species"]["172"]["types"] == [13, 13]  # TYPE_ELECTRIC, mono-type doubled
    assert data["species"]["172"]["gender_ratio"] == 127
    assert data["species"]["172"]["level_up"] == [[1, 1]]
    mon = data["trainers"]["Ace X"]["party"][0]
    assert (mon["species"], mon["form"], mon["gender"], mon["item"]) == (172, 0, None, None)
    assert mon["ivs"] == [1, 2, 3, 4, 5, 6] and mon["moves"] == [1] and mon["ability"] == 9
    assert data["trainers"]["Rival [{RIVAL_STARTER_1}]"] == {"party": [], "template": True}
    assert data["ai_flags"]["Ace X"] == 1 | 1 << 10
    assert data["formats"]["true_double"] == ["Ace X"] and data["weather"]["rain"] == ["Ace X"]


def test_unknown_form_key_raises():
    species = _raw()["SPECIES"]
    species["pichuspiky"] = dict(species["pichu"], id="pichuspiky", baseSpecies="pichu")
    with pytest.raises(ExtractError, match="pichuspiky"):
        normalise(_raw(SPECIES=species), "0" * 64)


@pytest.mark.parametrize(
    ("kind", "match"),
    [("SPECIES", "Pichu"), ("ITEMS", "num 1"), ("ABILITIES", "Static")],
)
def test_duplicate_entries_raise(kind, match):
    raw = _raw(
        ITEMS={
            "a": {"num": 1, "name": "A", "calcName": "A", "desc": ""},
            "b": {"num": 2, "name": "B", "calcName": "B", "desc": ""},
        }
    )
    entries = raw[kind]
    first = next(iter(entries.values()))
    if kind == "ITEMS":
        entries["b"]["num"] = 1
    elif kind == "SPECIES":  # same display name on another species
        entries["pichu2"] = dict(first, id="pichu2", num=25)
    else:
        entries["static2"] = dict(first, num=31)
    with pytest.raises(ExtractError, match=match):
        normalise(raw, "0" * 64)


def test_unused_set_raises():
    raw = _raw(PARTY_ORDER_PK={"Ace X": []})
    with pytest.raises(ExtractError, match="Ace X"):
        normalise(raw, "0" * 64)


def test_kaizo_roark_set(pkcalc):
    roark = pkcalc["trainers"]["Leader Roark"]["party"]
    assert [m["species"] for m in roark] == [438, 345, 443, 222, 213, 408]
    assert roark[0]["nature"] == "Impish" and roark[0]["ivs"] == [30] * 6
    assert roark[0]["level"] == 15
    assert pkcalc["ai_flags"]["Leader Roark"] == 23


def test_counts(pkcalc):
    if pkcalc["sha256"] != PKCALC_SHA256:
        pytest.skip("not the audited PKCalc zip")
    real = [t for t in pkcalc["trainers"].values() if not t.get("template")]
    assert len(real) == 485 and sum(len(t["party"]) for t in real) == 2112
    assert pkcalc["moves"]["205"]["name"] == "Accelerock"
    assert pkcalc["moves"]["4"]["name"] == "Weather Ball Water"
    assert pkcalc["tm_table"]["7"] == 62  # TM07 Aurora Beam (spec 5.3)


def test_forms(pkcalc):
    rotom = {(s["num"], s["form"]) for s in pkcalc["species"].values() if s["num"] == 479}
    assert rotom == {(479, f) for f in range(6)}
    assert pkcalc["species"]["503"]["name"] == "Rotom-Heat"
    assert (pkcalc["species"]["496"]["num"], pkcalc["species"]["496"]["form"]) == (386, 1)


def test_tutor_table_in_move_order(pkcalc):
    moves = [t["move"] for t in pkcalc["tutor_table"]]
    assert moves == sorted(moves) and 205 in moves


def test_ai_flag_bits_follow_coverage():
    from nuzline.engine.coverage import AI_FLAG_NAMES
    from tools.extract_pkcalc import AI_FLAG_KEYS

    assert list(zip(AI_FLAG_KEYS, AI_FLAG_NAMES, strict=True)) == [
        ("Basic", "BASIC"),
        ("EvaluateAttack", "EVAL_ATTACK"),
        ("Expert", "EXPERT"),
        ("Setup", "SETUP_FIRST_TURN"),
        ("Risky", "RISKY"),
        ("DamagePriority", "PRIORITIZE_EXTREMES"),
        ("BatonPass", "BATON_PASS"),
        ("TagStrategy", "TAG_STRATEGY"),
        ("CheckHP", "CHECK_HP"),
        ("Weather", "WEATHER"),
        ("Harassment", "HARRASSMENT"),
    ]
