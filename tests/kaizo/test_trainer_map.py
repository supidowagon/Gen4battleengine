"""PKCalc label / spreadsheet header -> ROM trainer id mapping (Task 9)."""

import pytest

from nuzline.data.trainer import trainer_pid
from tools.kaizo.trainer_map import NATURES, Mapping, MappingError, map_sheet_headers, map_trainers

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


# --- real inputs (skip without them) -----------------------------------------------------


def test_real_mapping(pkcalc, kaizo_rom_trainers, species):
    m = {x.label: x for x in map_trainers(pkcalc["trainers"], kaizo_rom_trainers, species, {})}
    assert m["Leader Roark"].id == 246
    assert len(m) == 485
    assert not any(k for k, v in pkcalc["trainers"].items() if v.get("template") and k in m)
    unmapped = [x for x in m.values() if x.id is None]
    assert all(x.how == "unmapped" for x in unmapped)
    assert all(x.how != "unmapped" for x in m.values() if x.id is not None)


def test_real_sheet_headers(docs, kaizo_rom_trainers, species):
    m = {x.label: x for x in map_sheet_headers(docs, kaizo_rom_trainers, species, {})}
    cite = next(
        t["cite"]
        for s in docs["splits"]["Roark"]["sections"]
        for t in s["trainers"]
        if t["text"] == "Leader Roark"
    )
    assert m[cite].id == 246
