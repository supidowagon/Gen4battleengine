import json
from collections import Counter
from pathlib import Path

import pytest

from tools.extract_decomp import (
    FORMS_H,
    ExtractError,
    _form_counts,
    _start_condition,
    _start_condition_overrides,
    _tutor_indices,
    extract_charmap,
    extract_egg_moves,
    extract_evolutions,
    extract_exp_tables,
    extract_forms,
    extract_items,
    extract_move_lists,
    extract_species,
    extract_static,
    extract_tutor_learnsets,
    extract_tutor_moves,
    species_form_rows,
)

DECOMP = Path("external/pokeplatinum")
pytestmark = pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("inter")
    extract_static(DECOMP, d)
    return d


def test_bulbasaur(out):
    sp = json.loads((out / "species.json").read_text())
    assert sp[1]["name"].lower() == "bulbasaur"
    assert sp[1]["base_stats"] == [45, 49, 49, 65, 65, 45]


def test_tackle_is_gen4_tackle(out):
    mv = json.loads((out / "moves.json").read_text())
    tackle = next(m for m in mv if m["name"].lower() == "tackle")
    assert (tackle["power"], tackle["accuracy"], tackle["pp"], tackle["category"]) == (
        35,
        95,
        35,
        0,
    )


def test_type_chart_fire_grass_and_ghost_normal(out):
    t = json.loads((out / "types.json").read_text())
    n = [x.lower() for x in t["names"]]
    assert t["chart"][n.index("fire")][n.index("grass")] == 20
    assert t["chart"][n.index("ghost")][n.index("normal")] == 0


def test_type_chart_order_is_decomp_row_index(out):
    # ApplyTypeChart applies the defender types in sTypeMatchupMultipliers row order
    # (battle_lib.c:2611-2646): Fire->Bug (:2406) comes before Fire->Rock (:2407).
    t = json.loads((out / "types.json").read_text())
    n = [x.lower() for x in t["names"]]
    order = t["chart_order"]
    fire = n.index("fire")
    assert order[n.index("normal")][n.index("rock")] == 0  # the first row (:2400)
    assert 0 <= order[fire][n.index("bug")] < order[fire][n.index("rock")]
    assert order[fire][n.index("normal")] == -1  # unlisted: neutral, order irrelevant
    listed = [o for row in order for o in row if o >= 0]
    assert sorted(listed) == list(range(len(listed)))  # each row index used once


def test_natures_count_and_adamant(out):
    nat = json.loads((out / "natures.json").read_text())
    assert len(nat) == 25
    adamant = next(x for x in nat if x["name"].lower() == "adamant")
    assert (adamant["up"], adamant["down"]) == (1, 3)


def test_learnset_chimchar_has_ember_at_7(out):
    ls = json.loads((out / "learnsets.json").read_text())
    sp = json.loads((out / "species.json").read_text())
    mv = json.loads((out / "moves.json").read_text())
    chimchar = next(i for i, s in enumerate(sp) if s["name"].lower() == "chimchar")
    ember = next(i for i, m in enumerate(mv) if m["name"].lower() == "ember")
    assert [7, ember] in ls[str(chimchar)]


def test_species_exp_rate():
    species = {s["name"].lower(): s for s in extract_species(DECOMP)}
    assert species["abra"]["exp_rate"] == 3  # EXP_RATE_MEDIUM_SLOW


def test_exp_yield(out):
    sp = json.loads((out / "species.json").read_text("utf-8"))
    assert (sp[1]["exp_yield"], sp[487]["exp_yield"]) == (64, 220)  # Bulbasaur, Giratina


def test_ai_effect_equals_effect_in_vanilla(out):
    moves = json.loads((out / "moves.json").read_text("utf-8"))
    assert all(m["ai_effect"] == m["effect"] for m in moves)


def test_exp_tables():
    t = extract_exp_tables(DECOMP)
    assert len(t) == 8 and all(len(r) == 101 for r in t)
    assert t[0][100] == 1_000_000 and t[3][100] == 1_059_860 and t[5][100] == 1_250_000


def test_charmap():
    cm = extract_charmap(DECOMP)
    assert cm[0x0121] == "0" and cm[0x0145] == "a" and cm[0x01AB] == "!"
    assert cm[0x01C1] == "=" and cm[0x01DE] == " "  # trailing space is the value
    assert all(isinstance(k, int) for k in cm)


def test_static_writes_growth_files(out):
    assert len(json.loads((out / "exp_tables.json").read_text(encoding="utf-8"))) == 8
    assert json.loads((out / "charmap.json").read_text(encoding="utf-8"))["0145"] == "a"


def _by_tm_indices(species_dir: str) -> list[int]:
    d = json.loads((DECOMP / "res/pokemon" / species_dir / "data.json").read_text("utf-8"))
    out = []
    for marker in d["learnset"].get("by_tm", []):
        n = int(marker[2:]) - 1
        out.append(n + 92 if marker.startswith("HM") else n)
    return sorted(out)


def test_tm_learnsets_match_by_tm(out):
    tm = json.loads((out / "tm_learnsets.json").read_text())
    sp = json.loads((out / "species.json").read_text())
    ids = {s["name"].lower(): i for i, s in enumerate(sp)}
    for name in ("furret", "kricketot", "shedinja"):
        assert tm[str(ids[name])] == _by_tm_indices(name)
    assert tm[str(ids["kricketot"])] == []
    assert 9 in tm[str(ids["furret"])]  # TM10

    items = json.loads((out / "items.json").read_text())
    mv = json.loads((out / "moves.json").read_text())

    def teaches(item: str):
        i = next(i for i in items if i["name"] == item)["teaches_move"]
        return None if i is None else mv[i]["name"]

    assert teaches("TM01") == "Focus Punch"
    assert teaches("TM10") == "Hidden Power"
    assert teaches("HM08") == "Rock Climb"
    assert teaches("Potion") is None


def test_forms_registry_order_matches_narc_index():
    f = extract_forms(DECOMP)
    assert [(x["species"], x["form"], x["row"]) for x in f][:3] == [
        (386, 1, 496),
        (386, 2, 497),
        (386, 3, 498),
    ]
    assert [x["row"] for x in f] == list(range(496, 508))
    wash = next(x for x in f if (x["species"], x["form"]) == (479, 2))
    assert wash["row"] == 504 and wash["base_stats"] == [50, 65, 107, 105, 107, 86]
    assert wash["name"] == "Wash"
    sp = extract_species(DECOMP)
    assert set(wash) - {"species", "form", "row"} == set(sp[479])
    origin = next(x for x in f if x["row"] == 501)
    assert origin["weight_hg"] == sp[487]["weight_hg"]  # weight from the base species


def test_form_grid():
    g = species_form_rows(DECOMP)
    assert len(g) == 496 and all(len(r) == 28 for r in g)
    assert g[479][:7] == [479, 503, 504, 505, 506, 507, -1] and g[201][27] == 201
    assert g[201][0] == 201
    assert g[493][17] == 493 and g[493][18] == -1 and g[1][1] == -1


def test_form_intermediates(out):
    forms = json.loads((out / "forms.json").read_text("utf-8"))
    assert forms == extract_forms(DECOMP)
    assert json.loads((out / "form_rows.json").read_text("utf-8")) == species_form_rows(DECOMP)
    learn = json.loads((out / "learnsets.json").read_text("utf-8"))
    tm = json.loads((out / "tm_learnsets.json").read_text("utf-8"))
    assert sorted(map(int, learn)) == list(range(508)) == sorted(map(int, tm))
    assert tm["504"] == _by_tm_indices("rotom/forms/wash")


def test_evolutions():
    ev = extract_evolutions(DECOMP)
    assert ev["1"] == [[4, 16, 2]]  # EVO_LEVEL 16 -> Ivysaur
    assert len(ev["133"]) == 7 and ev["133"][0] == [25, 0, 470]  # Eevee: MOSS_ROCK -> Leafeon
    assert ev["3"] == []  # Venusaur


def test_egg_moves():
    assert extract_egg_moves(DECOMP)["1"][:3] == [113, 130, 219]  # Light Screen, Skull Bash, ...


def test_static_writes_evolutions_and_egg_moves(out):
    ev = json.loads((out / "evolutions.json").read_text("utf-8"))
    assert len(ev) == 496 and ev["1"] == [[4, 16, 2]]
    assert "3" not in json.loads((out / "egg_moves.json").read_text("utf-8"))


def test_tutor_moves():
    t = extract_tutor_moves(DECOMP)
    assert len(t) == 38 and t[0] == {
        "move": 291,
        "costs": {"red": 2, "blue": 4, "yellow": 2, "green": 0},
        "location": "route_212",
    }
    assert t[1]["move"] == 189 and t[1]["location"] == "survival_area"  # Mud-Slap
    assert t[-1]["move"] == 253 and t[-1]["location"] == "snowpoint_city"  # Uproar
    assert sorted(Counter(x["location"] for x in t).values()) == [8, 13, 17]


def test_tutor_learnsets_rotom_and_forms():
    tl = extract_tutor_learnsets(DECOMP)
    assert sorted(map(int, tl)) == list(range(508))
    assert 1 in tl["479"] and len(tl["479"]) == 9  # Rotom: Mud-Slap (index 1) among its 9
    assert tl["504"] == tl["479"] and tl["0"] == tl["494"] == tl["495"] == []  # Wash = base list
    assert all(i == sorted(set(i)) for i in tl.values())


def test_unlisted_tutor_move_raises():
    with pytest.raises(ExtractError, match="MOVE_TACKLE") as e:
        _tutor_indices(
            ["MOVE_MUD_SLAP", "MOVE_TACKLE"], ["MOVE_DIVE", "MOVE_MUD_SLAP"], Path("x/data.json")
        )
    assert "x" in str(e.value) and "data.json" in str(e.value)
    assert _tutor_indices(["MOVE_MUD_SLAP"], ["MOVE_DIVE", "MOVE_MUD_SLAP"], Path("x")) == [1]


def test_static_writes_tutor_files(out):
    assert len(json.loads((out / "tutor_moves.json").read_text("utf-8"))) == 38
    assert len(json.loads((out / "tutor_learnsets.json").read_text("utf-8"))) == 508


@pytest.mark.parametrize("value", ["(NUM_TYPES)", "0x12", "2 // east/west", "SHELLOS_FORMS"])
def test_form_count_that_is_not_a_decimal_raises(tmp_path, value):
    # Final review: such a define was skipped and the species defaulted to one form.
    (tmp_path / FORMS_H).parent.mkdir(parents=True)
    text = f"#define UNOWN_FORM_COUNT 28\n#define SHELLOS_FORM_COUNT {value}\n"
    (tmp_path / FORMS_H).write_text(text, "utf-8")
    species = ["SPECIES_NONE", "SPECIES_UNOWN", "SPECIES_SHELLOS"]
    with pytest.raises(ExtractError, match=r"forms\.h: SHELLOS_FORM_COUNT") as e:
        _form_counts(tmp_path, species)
    assert value in str(e.value)
    (tmp_path / FORMS_H).write_text("#define  SHELLOS_FORM_COUNT\t2  \n", "utf-8")
    assert _form_counts(tmp_path, species) == {2: 2}


def test_hp_restored_extracted():
    items = {i["name"]: i for i in extract_items(DECOMP)}
    assert items["Potion"]["hp_restored"] == 20
    # ItemPartyParam.hpRestored is a u8 the data processor fills from an s8 (itemproc.c:168):
    # Max Potion's -1 (restore all) is 255 in the ROM.
    assert items["Max Potion"]["hp_restored"] == 255
    assert items["Expert Belt"]["hp_restored"] == 0  # no itemUseParams


def test_decomp_lists():
    lists = extract_move_lists(DECOMP)
    assert list(lists) == ["sound", "punch"]
    assert 4 in lists["punch"] and len(lists["punch"]) == 15 and len(lists["sound"]) == 12
    assert lists["sound"][:2] == [45, 46]  # Growl, Roar: source order


def _write(tmp_path, text, name="sc.json"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_start_condition_names_checked(tmp_path):
    p = _write(tmp_path, '{"246": {"trainer": "Byron", "condition": "none"}}')
    with pytest.raises(ExtractError, match="id 246 is Roark, not Byron"):
        _start_condition_overrides(p, {246: "Roark"}.__getitem__)
    p = _write(tmp_path, '{"999": {"trainer": "Roark", "condition": "none"}}', "b.json")
    with pytest.raises(ExtractError, match="not a trainer id"):
        _start_condition_overrides(p, {246: "Roark"}.__getitem__)
    with pytest.raises(ExtractError, match="not a trainer id"):
        _start_condition_overrides(p, ["Roark"].__getitem__)  # a list raises IndexError


def test_start_condition_overrides_keyed_by_id(tmp_path):
    # Kaizo ROM names repeat ("Grunt"): an override applies to its id only.
    names = {10: "Grunt", 11: "Grunt"}
    p = _write(tmp_path, '{"comment": ["x"], "10": {"trainer": "Grunt", "condition": "rain"}}')
    fixed = _start_condition_overrides(p, names.__getitem__)
    assert fixed == {10: "rain"}
    assert _start_condition(10, "Grunt", {"rain", "none"}, fixed, p) == "rain"
    assert _start_condition(11, "Grunt", {"none"}, fixed, p) == "none"
    with pytest.raises(ExtractError, match=r"Grunt \(id 11\) can start with any of.*sc\.json"):
        _start_condition(11, "Grunt", {"rain", "none"}, fixed, p)
    with pytest.raises(ExtractError, match=r"sc\.json: Grunt is 'rain' but the maps give 'sun'"):
        _start_condition(10, "Grunt", {"sun"}, fixed, p)


def test_unknown_start_condition_only_where_allowed(tmp_path):
    from tools.extract_decomp import START_CONDITIONS, UNKNOWN_CONDITION

    p = _write(tmp_path, '{"10": {"trainer": "Grunt", "condition": "unknown"}}')
    with pytest.raises(ExtractError, match="unknown start condition 'unknown'"):
        _start_condition_overrides(p, {10: "Grunt"}.__getitem__)
    fixed = _start_condition_overrides(
        p, {10: "Grunt"}.__getitem__, (*START_CONDITIONS, UNKNOWN_CONDITION)
    )
    assert _start_condition(10, "Grunt", {"rain", "hail"}, fixed, p) == "unknown"
    with pytest.raises(ExtractError, match="'unknown' but the maps give 'rain'"):
        _start_condition(10, "Grunt", {"rain"}, fixed, p)
