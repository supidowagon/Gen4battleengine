"""Kaizo ROM specifics (K2b Task 8): the trainer struct, the map headers in ``d_test.narc``,
the changed battle scripts and their classification.

Every ROM test skips without the local inputs (``kaizo_files``, ``kaizo_rom_dir``)."""

import json
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from tests.tools.rom.diff_helpers import fake
from tools.rom.diff import changed_members, diff_roms, load_member_classification

ROM_DIFF = Path(__file__).resolve().parents[3] / "data" / "kaizo" / "rom_diff.toml"
SCRIPT_NARCS = (
    "battle/skill/be_seq.narc",
    "battle/skill/sub_seq.narc",
    "battle/tr_ai/tr_ai_seq.narc",
)


def write(tmp_path, text, name="rom_diff.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --- trainer struct (Ruling 6) ---------------------------------------------------------


def test_kaizo_roark_struct(kaizo_rom_trainers, docs):
    roark = kaizo_rom_trainers[246]["party"]
    assert [m["species"] for m in roark] == [438, 345, 443, 222, 213, 408]
    assert {m["difficulty"] * 31 // 255 for m in roark} == {30}
    raw = docs["raw_trainers"]["246"]["party"]
    assert [(m["difficulty"], m["ability_override"]) for m in roark] == [
        (r["difficulty"], r["ability_number"]) for r in raw
    ]


def test_override_byte_values(kaizo_rom_trainers):
    # The byte is flags (spec 5.4): 16 forces ability slot 0, 32 slot 1, 2 female. The
    # sheet uses 16, 32 and 2 alone; the ROM also has one 18 (16 | 2).
    mons = [m for t in kaizo_rom_trainers for m in t["party"]]
    values = Counter(m["ability_override"] for m in mons)
    assert set(values) <= {0, 2, 16, 32, 18}
    assert 100 <= sum(n for v, n in values.items() if v) <= 122  # spec 5.4: 111 on the sheet
    assert all(0 <= m["difficulty"] <= 255 for m in mons)


def test_kaizo_trainers_differ_from_vanilla_and_carry_structures(kaizo_rom_trainers):
    assert len(kaizo_rom_trainers) == 928
    assert kaizo_rom_trainers[246]["name"] == "Roark"
    assert [m["species"] for m in kaizo_rom_trainers[246]["party"]] != [74, 95, 408]
    for key in ("format", "opponent_2", "partner", "start_condition"):
        assert all(key in t for t in kaizo_rom_trainers), key
    formats = Counter(t["format"] for t in kaizo_rom_trainers)
    assert formats["single"] > 0 and formats["double"] > 0 and formats["multi"] > 0


def test_kaizo_split_of_iv_scale():
    from tools.rom.readers import ReaderError
    from tools.rom.readers.trainers import kaizo_party_fields

    assert kaizo_party_fields(0x10FB, "w") == {"difficulty": 251, "ability_override": 16}
    assert kaizo_party_fields(0x12FF, "w") == {"difficulty": 255, "ability_override": 18}
    for bad in (0x40FB, 0x30FB, 0x01FB):
        with pytest.raises(ReaderError, match="w: ability override byte"):
            kaizo_party_fields(bad, "w")


# --- map headers from d_test.narc (Ruling 8, pre-flight #1/#12) -------------------------


def test_kaizo_header_rows_vs_vanilla(kaizo_files, vanilla_files, symbols):
    from tools.rom.readers.maps import KAIZO_MAP_HEADERS, map_header_rows

    vrows = map_header_rows(vanilla_files, symbols)
    krows = map_header_rows(kaizo_files, symbols)
    assert (len(vrows), len(krows)) == (593, 618)
    members = kaizo_files.narc(KAIZO_MAP_HEADERS)
    from tools.rom.readers import symbol_bytes

    table = symbol_bytes(vanilla_files, symbols, "sMapHeaders")
    same = sum(members[i] == table[24 * i : 24 * i + 24] for i in range(593))
    weather = sum(members[i][20] != table[24 * i + 20] for i in range(593))
    assert (same, weather, len(members) - 593) == (455, 61, 25)
    assert [r[2] for r in krows] == [m[20] for m in members]


class _Arm9Patched:
    """``rom`` with ARM9 bytes at ``address`` replaced by ``new``."""

    def __init__(self, rom, address, new):
        self._rom, self._address, self._new = rom, address, new

    def __getattr__(self, name):
        return getattr(self._rom, name)

    def arm9_sections(self):
        out = []
        for ram, data in self._rom.arm9_sections():
            if ram <= self._address < ram + len(data):
                off = self._address - ram
                data = data[:off] + self._new + data[off + len(self._new) :]
            out.append((ram, data))
        return out


def test_kaizo_header_accessors_checked(kaizo_files, symbols):
    from tools.rom.readers import ReaderError
    from tools.rom.readers.maps import map_header_rows

    bounds = symbols.find("MapHeader_IDBoundsCheck").address & ~1
    # movs r0, #148 (NARC_INDEX_DEBUG__CB_EDIT__D_TEST) -> movs r0, #149
    bad = _Arm9Patched(kaizo_files, bounds + 4, bytes([149, 0x20]))
    with pytest.raises(ReaderError, match="MapHeader_IDBoundsCheck"):
        map_header_rows(bad, symbols)
    weather = symbols.find("MapHeader_GetWeatherType").address & ~1
    bad = _Arm9Patched(kaizo_files, weather + 16, (21).to_bytes(4, "little"))
    with pytest.raises(ReaderError, match="MapHeader_GetWeatherType"):
        map_header_rows(bad, symbols)
    yearly = symbols.find("FieldSystem_GetWeather").address & ~1
    bad = _Arm9Patched(kaizo_files, yearly + 128, (0).to_bytes(4, "little"))
    with pytest.raises(ReaderError, match="sYearlyWeather"):
        map_header_rows(bad, symbols)


def test_other_game_code_refused(kaizo_files, symbols):
    from tools.rom.readers import ReaderError
    from tools.rom.readers.maps import map_header_rows

    class Other:
        game_code = "XXXX"

    with pytest.raises(ReaderError, match="game code XXXX"):
        map_header_rows(Other(), symbols)


def test_kaizo_headers_narc_id_is_d_test(decomp):
    from tools.rom.readers.maps import KAIZO_HEADERS_NARC_ID

    text = (decomp / "include" / "constants" / "narc.h").read_text("utf-8")
    body = text[text.index("enum NarcID {") :]
    names = [ln.strip().split(" ")[0].rstrip(",") for ln in body.splitlines()[1:]]
    names = [n for n in names if n.startswith("NARC_INDEX_")]
    assert names.index("NARC_INDEX_DEBUG__CB_EDIT__D_TEST") == KAIZO_HEADERS_NARC_ID


def test_weather_switch_decoded(vanilla_files, kaizo_files, symbols, decomp):
    # Vanilla: the decoder reproduces WEATHER_CONDITION for the switch's ids 0-15.
    from tools.extract_decomp import WEATHER_CONDITION
    from tools.rom.readers.maps import _weather_names, weather_switch_conditions

    names, _, _ = _weather_names(decomp)
    got, top = weather_switch_conditions(vanilla_files, symbols, decomp)
    assert top == 15
    assert got == {w: WEATHER_CONDITION.get(names.get(w), "none") for w in range(16)}
    # Kaizo: ids 0-15 as vanilla; 12 runs the new sub_seq 297, 17 sun, 18 Trick Room.
    kaizo, top = weather_switch_conditions(kaizo_files, symbols, decomp)
    assert top == 18
    assert kaizo == {**got, 12: "subscript 297", 16: "none", 17: "sun", 18: "trick_room"}


def test_kaizo_field_effects_and_unknown_starts(kaizo_rom_dir, kaizo_rom_trainers):
    rows = json.loads((kaizo_rom_dir / "rom_field_effects.json").read_text("utf-8"))
    assert {r["effect"] for r in rows} == {"trick_room"}
    by_id = {r["id"]: r for r in rows}
    assert by_id[403]["name"] == "Cyrus" and kaizo_rom_trainers[403]["start_condition"] == "none"
    assert len(rows) == 53
    unknown = [t["id"] for t in kaizo_rom_trainers if t["start_condition"] == "unknown"]
    # 318, 583 and 676 were settled in E3's batches 5-7.
    assert unknown == [325, 343, 509, 559, 610, 649, 746, 747]
    assert [kaizo_rom_trainers[i]["format"] for i in (521, 527)] == ["unknown", "unknown"]
    assert kaizo_rom_trainers[94]["start_condition"] == "rain"  # Route 212 South


def test_vanilla_address_readers_unchanged(vanilla_files, kaizo_files, symbols):
    # sTrainerClassGender and gVsSeekerRematchData are read at vanilla addresses on Kaizo
    # (fix round 1): neither they nor their readers are a code change, and the classified
    # [[code]] entries name none of them.
    from tools.rom.diff import load_classification

    names = {
        "sTrainerClassGender",
        "TrainerClass_Gender",
        "gVsSeekerRematchData",
        "VsSeeker_GetRematchDataIndexForTrainer",
        "VsSeeker_GetCurrentLevelForRematchData",
        "VsSeeker_GetNextLowerRematchLevel",
        "VsSeeker_GetTrainerIDForRematchLevel",
    }
    for n in names:
        symbols.find(n)  # each name is in the linker map
    changed = {c.symbol for c in diff_roms(vanilla_files, kaizo_files, symbols).code}
    assert not names & changed
    assert not names & {symbol for _, symbol in load_classification(ROM_DIFF)}


def test_kaizo_trainer_maps_report_member_136(kaizo_files, decomp, symbols):
    from tools.rom.readers.maps import UnresolvedOperand, trainer_maps

    found = []
    where, weathers = trainer_maps(kaizo_files, decomp, symbols, 928, found)
    assert len(weathers) == 618
    assert UnresolvedOperand(136, 909, 0x8004, 0) in found
    assert where  # trainers are placed


def test_rom_battle_structures_kaizo(kaizo_files, decomp, symbols):
    from tools.rom.readers.maps import rom_battle_structures

    structures = rom_battle_structures(kaizo_files, decomp, symbols, 928)
    assert structures
    assert {s.format for s in structures.values()} >= {"multi"}


# --- changed battle scripts (Ruling 13) -------------------------------------------------


def test_changed_members_synthetic():
    a = fake(files={"x.narc": [b"0", b"1", b"2", b"3"]})
    b = fake(files={"x.narc": [b"0", b"9", b"2", b"8", b"4"], "y.narc": [b"0"]})
    d = diff_roms(a, b, None)
    assert changed_members(d, "x.narc") == [1, 3, 4]
    assert changed_members(d, "z.narc") == []  # unchanged
    with pytest.raises(ValueError, match="y.narc.*added"):
        changed_members(d, "y.narc")


def test_member_classification_rules(tmp_path):
    p = write(
        tmp_path,
        '[[member]]\nnarc = "battle/skill/be_seq.narc"\nindex = 7\nstatus = "modelled"\n'
        'note = "n"\n',
    )
    got = load_member_classification(p, {"battle/skill/be_seq.narc": [7, 9]})
    assert got == {
        ("battle/skill/be_seq.narc", 7): ("modelled", "n"),
        ("battle/skill/be_seq.narc", 9): ("unknown", ""),
    }
    with pytest.raises(ValueError, match="be_seq.narc#7.*stale"):
        load_member_classification(p, {"battle/skill/be_seq.narc": [9]})
    bad = write(
        tmp_path,
        '[[member]]\nnarc = "battle/skill/be_seq.narc"\nindex = 7\nstatus = "maybe"\n',
        "b.toml",
    )
    with pytest.raises(ValueError, match="unknown status 'maybe'"):
        load_member_classification(bad, {"battle/skill/be_seq.narc": [7]})
    dup = write(tmp_path, p.read_text("utf-8") * 2, "c.toml")
    with pytest.raises(ValueError, match="duplicate"):
        load_member_classification(dup, {"battle/skill/be_seq.narc": [7]})


def test_code_loader_ignores_member_tables(tmp_path):
    from tools.rom.diff import load_classification

    p = write(
        tmp_path,
        '[[code]]\nmodule = "arm9"\nsymbol = "Foo"\nstatus = "unknown"\n\n'
        '[[member]]\nnarc = "battle/skill/be_seq.narc"\nindex = 7\nstatus = "unknown"\n',
    )
    assert load_classification(p) == {("arm9", "Foo"): ("unknown", "")}


def test_append_unknown_members(tmp_path):
    from tools.extract_rom import append_unknown_members

    p = write(
        tmp_path,
        '# head\n\n[[member]]\nnarc = "battle/skill/be_seq.narc"\nindex = 7\n'
        'status = "modelled"\nnote = "n"\n',
    )
    added = append_unknown_members(p, {"battle/skill/be_seq.narc": [7, 9], "x.narc": []})
    assert added == [("battle/skill/be_seq.narc", 9)]
    got = load_member_classification(p, {"battle/skill/be_seq.narc": [7, 9]})
    assert got[("battle/skill/be_seq.narc", 9)] == ("unknown", "")
    assert append_unknown_members(p, {"battle/skill/be_seq.narc": [7, 9]}) == []


@pytest.mark.fullset
def test_every_changed_script_classified(kaizo_rom_dir):
    changed = json.loads((kaizo_rom_dir / "changed_scripts.json").read_text("utf-8"))
    assert sorted(changed) == sorted(SCRIPT_NARCS)
    entries = load_member_classification(ROM_DIFF, changed)  # stale entries raise
    want = {(narc, i) for narc, ids in changed.items() for i in ids}
    assert set(entries) == want
    with ROM_DIFF.open("rb") as f:
        listed = {(m["narc"], m["index"]) for m in tomllib.load(f).get("member", [])}
    assert listed == want  # every changed member has an explicit entry
