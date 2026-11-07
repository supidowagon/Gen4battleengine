"""PKCalc label / spreadsheet header -> ROM trainer id mapping (Task 9)."""

import pytest

from nuzline.data.trainer import trainer_pid
from tools.kaizo.trainer_map import (
    NATURES,
    Mapping,
    MappingError,
    check_override_keys,
    map_sheet_headers,
    map_trainers,
)
from tools.rom.inputs import KAIZO_DOCS_SHA256, PKCALC_SHA256

P = ((74, 0, 12), (95, 0, 14))
# Geodude (74) and Onix (95) are both 50 % female; index by species id.
SPECIES = [{"name": f"S{i}", "gender_ratio": 127} for i in range(100)]
SPECIES[74] = {"name": "GEODUDE", "gender_ratio": 127}
SPECIES[95] = {"name": "ONIX", "gender_ratio": 127}
SPECIES[1] = {"name": "BULBASAUR", "gender_ratio": 31}


def rom_trainer(tid, name, sig, difficulty=0, override=0, class_id=10, class_gender=0):
    return {
        "id": tid,
        "name": name,
        "class_id": class_id,
        "class_gender": class_gender,
        "party": [
            {
                "species": s,
                "form": f,
                "level": lv,
                "difficulty": difficulty,
                "ability_override": override,
            }
            for s, f, lv in sig
        ],
    }


def pk(sig, natures=None, genders=None):
    return {
        "party": [
            {
                "species": s,
                "form": f,
                "level": lv,
                "nature": (natures or ["Hardy"] * len(sig))[i],
                "gender": (genders or ["Male"] * len(sig))[i],
            }
            for i, (s, f, lv) in enumerate(sig)
        ]
    }


def pid_traits(tid, sig, difficulty, class_id=10):
    """The natures and genders the vanilla PID gives a trainer's party."""
    from nuzline.data.trainer import gender_of

    natures, genders = [], []
    for s, _f, lv in sig:
        pid = trainer_pid(difficulty, lv, s, tid, class_id, female_class=False)
        natures.append(NATURES[pid % 25])
        genders.append("Female" if gender_of(127, pid) == 1 else "Male")
    return natures, genders


def test_exact():
    m = map_trainers({"Leader X": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("Leader X", 5, "exact", (5,))]


def test_templates_and_partyless_labels_are_not_crashes():
    labels = {"T": {"party": [], "template": True}, "Maid Yoki": {"party": []}, "Z": {}}
    m = map_trainers(labels, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("Maid Yoki", None, "unmapped", ()), Mapping("Z", None, "unmapped", ())]


def test_pid_tiebreak():
    # ids 6 and 7 share P; only id 7's difficulty gives PKCalc's natures and genders.
    n7, g7 = pid_traits(7, P, 200)
    assert (n7, g7) != pid_traits(6, P, 0)
    m = map_trainers(
        {"Hiker Y": pk(P, n7, g7)},
        [rom_trainer(6, "Y", P, 0), rom_trainer(7, "Y", P, 200)],
        SPECIES,
        {},
    )
    assert (m[0].id, m[0].how, m[0].candidates) == (7, "pid", (6, 7))


def test_pid_ignores_override_mons():
    # id 6's mons carry an ability override (high byte), so they cannot score: 7 wins.
    n7, g7 = pid_traits(7, P, 200)
    m = map_trainers(
        {"Hiker Y": pk(P, n7, g7)},
        [rom_trainer(6, "Y", P, 200, override=16), rom_trainer(7, "Y", P, 200)],
        SPECIES,
        {},
    )
    assert (m[0].id, m[0].how) == (7, "pid")


def test_name_breaks_a_pid_tie():
    m = map_trainers(
        {"Ace Trainer Bob": pk(P)},
        [rom_trainer(6, "Alice", P, override=2), rom_trainer(7, "Bob", P, override=2)],
        SPECIES,
        {},
    )
    assert (m[0].id, m[0].how, m[0].candidates) == (7, "name", (6, 7))


def test_tie_without_name_is_unmapped_with_candidates():
    m = map_trainers(
        {"Ace Trainer": pk(P)},
        [rom_trainer(6, "Alice", P, override=2), rom_trainer(7, "Bob", P, override=2)],
        SPECIES,
        {},
    )
    assert m == [Mapping("Ace Trainer", None, "unmapped", (6, 7))]


def test_unmapped_is_reported_not_guessed():
    m = map_trainers({"Lady Emi": pk(((1, 0, 5),))}, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("Lady Emi", None, "unmapped", ())]


def test_two_labels_one_id_raises():
    with pytest.raises(MappingError, match="A.*B"):
        map_trainers({"A": pk(P), "B": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {})


def test_override_breaks_the_tie():
    m = {
        x.label: x
        for x in map_trainers({"A": pk(P), "B": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {"B": 5})
    }
    assert (m["B"].id, m["B"].how) == (5, "override") and m["A"].id is None


def test_override_to_unknown_id_raises():
    with pytest.raises(MappingError, match="999"):
        map_trainers({"A": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {"A": 999})


# --- spreadsheet headers ---------------------------------------------------------------


def header(cite, text, party):
    return {
        "cite": cite,
        "text": text,
        "name": text,
        "tags": {},
        "party": [{"species": s, "level": lv} for s, lv in party],
    }


def docs_of(*headers):
    return {"splits": {"S": {"sheet": "S Split", "sections": [{"trainers": list(headers)}]}}}


MAPPED_HEADERS = 416
SHEET_P = (("Geodude", 12), ("ONIX", 14))


def test_sheet_exact_uses_species_names_and_levels():
    d = docs_of(header("S Split:5", "Leader X", SHEET_P))
    m = map_sheet_headers(d, [rom_trainer(5, "X", P)], SPECIES, {})
    assert m == [Mapping("S Split:5", 5, "exact", (5,))]


def test_sheet_name_tiebreak_never_pid():
    roms = [rom_trainer(6, "Alice", P), rom_trainer(7, "Bob", P)]
    d = docs_of(header("S Split:5", "Ace Bob", SHEET_P), header("S Split:9", "Ace", SHEET_P))
    m = {x.label: x for x in map_sheet_headers(d, roms, SPECIES, {})}
    assert (m["S Split:5"].id, m["S Split:5"].how) == (7, "name")
    assert (m["S Split:9"].id, m["S Split:9"].how, m["S Split:9"].candidates) == (
        None,
        "unmapped",
        (6, 7),
    )


def test_sheet_unknown_species_is_unmapped():
    d = docs_of(header("S Split:5", "Leader X", (("Missingno", 12),)))
    assert map_sheet_headers(d, [rom_trainer(5, "X", P)], SPECIES, {})[0].id is None


def test_sheet_override_by_cite():
    d = docs_of(header("S Split:5", "Mystery", (("Bulbasaur", 3),)))
    m = map_sheet_headers(d, [rom_trainer(5, "X", P)], SPECIES, {"S Split:5": 5})
    assert m == [Mapping("S Split:5", 5, "override", (5,))]


def test_sheet_same_trainer_listed_twice_is_no_conflict():
    d = docs_of(header("S Split:5", "Leader X", SHEET_P), header("S Split:50", "Leader X", SHEET_P))
    m = map_sheet_headers(d, [rom_trainer(5, "X", P)], SPECIES, {})
    assert [(x.label, x.id) for x in m] == [("S Split:5", 5), ("S Split:50", 5)]


def test_sheet_different_headers_one_id_raise():
    d = docs_of(header("S Split:5", "Leader X", SHEET_P), header("S Split:50", "Other X", SHEET_P))
    with pytest.raises(MappingError, match="S Split:5.*S Split:50"):
        map_sheet_headers(d, [rom_trainer(5, "X", P)], SPECIES, {})


def test_override_loser_records_why():
    m = {
        x.label: x
        for x in map_trainers({"A": pk(P), "B": pk(P)}, [rom_trainer(5, "X", P)], SPECIES, {"B": 5})
    }
    assert (m["A"].how, m["A"].candidates, m["A"].lost_to) == ("unmapped", (5,), 5)


def test_name_tiebreak_matches_whole_words_only():
    roms = [rom_trainer(6, "Ann", P, override=2), rom_trainer(7, "Bob", P, override=2)]
    m = map_trainers({"Lady Joanna": pk(P)}, roms, SPECIES, {})
    assert m[0].id is None
    m = map_trainers({"Lady Ann": pk(P)}, roms, SPECIES, {})
    assert m[0].id == 6


def test_form_blind_retry_records_the_form_conflict():
    sig_rom = ((74, 1, 12), (95, 0, 14))
    m = map_trainers({"Hiker Y": pk(P)}, [rom_trainer(5, "Y", sig_rom)], SPECIES, {})
    assert m == [Mapping("Hiker Y", 5, "form", (5,), ((0, 0, 1),))]


def test_form_blind_retry_still_refuses_ties():
    sig_rom = ((74, 1, 12), (95, 0, 14))
    roms = [rom_trainer(5, "A", sig_rom, override=2), rom_trainer(6, "B", sig_rom, override=2)]
    m = map_trainers({"Hiker": pk(P)}, roms, SPECIES, {})
    assert m == [Mapping("Hiker", None, "unmapped", (5, 6))]


def test_strict_signature_beats_the_form_blind_one():
    roms = [rom_trainer(5, "Y", ((74, 1, 12), (95, 0, 14))), rom_trainer(6, "Y", P)]
    m = map_trainers({"Hiker Y": pk(P)}, roms, SPECIES, {})
    assert (m[0].id, m[0].how) == (6, "exact")


def test_check_override_keys_rejects_typos():
    check_override_keys(
        {"Leader Roark": 1, "Roark Split:5": 2}, ["Leader Roark"], ["Roark Split:5"]
    )
    with pytest.raises(MappingError, match="Leader Roak"):
        check_override_keys({"Leader Roak": 246}, ["Leader Roark"], ["Roark Split:5"])


def test_sheet_form_suffix_resolves_by_base_species():
    species = [*SPECIES, {"name": "ROTOM", "gender_ratio": 255}]  # id 100
    rom = rom_trainer(5, "X", ((100, 5, 30), (95, 0, 14)))
    d = docs_of(header("S Split:5", "Leader X", (("Rotom-Mow", 30), ("Onix", 14))))
    assert map_sheet_headers(d, [rom], species, {}) == [Mapping("S Split:5", 5, "exact", (5,))]


def test_sheet_full_name_wins_over_base():
    species = [*SPECIES, {"name": "HO-OH", "gender_ratio": 255}, {"name": "HO", "gender_ratio": 0}]
    rom = rom_trainer(5, "X", ((100, 0, 30),))
    d = docs_of(header("S Split:5", "Leader X", (("Ho-Oh", 30),)))
    assert map_sheet_headers(d, [rom], species, {})[0].id == 5


# --- real inputs (skip without them) -----------------------------------------------------


def test_real_mapping(pkcalc, kaizo_rom_trainers, species):
    m = {x.label: x for x in map_trainers(pkcalc["trainers"], kaizo_rom_trainers, species, {})}
    assert m["Leader Roark"].id == 246
    assert len(m) == 485
    assert sum(x.id is not None for x in m.values()) >= 473
    assert not any(v.get("template") and k in m for k, v in pkcalc["trainers"].items())
    assert all((x.how == "unmapped") == (x.id is None) for x in m.values())
    if pkcalc["sha256"] == PKCALC_SHA256:  # pin the audited zip's result
        assert sorted(x.label for x in m.values() if x.id is None) == []
        forms = {x.label: x.id for x in m.values() if x.how == "form"}
        assert forms["Bug Catcher Sebastian"] == 356 and forms["Idol Zoe"] == 236
        assert len(forms) == 10


def test_real_sheet_headers(docs, kaizo_rom_trainers, species):
    m = {x.label: x for x in map_sheet_headers(docs, kaizo_rom_trainers, species, {})}
    cite = next(
        t["cite"]
        for s in docs["splits"]["Roark"]["sections"]
        for t in s["trainers"]
        if t["text"] == "Leader Roark"
    )
    assert m[cite].id == 246
    if docs["sha256"] == KAIZO_DOCS_SHA256:
        mapped = sum(x.id is not None for x in m.values())
        assert (len(m), mapped) == (482, MAPPED_HEADERS)
        for c, tid in {
            "Gardenia Split:484": 315,
            "Fantina Split:452": 318,
            "Elite Four Split:298": 870,
            "Galactic Split:1229": 780,
        }.items():
            assert m[c].id == tid
