import json
from pathlib import Path

import pytest

from tools.extract_decomp import (
    extract_charmap,
    extract_exp_tables,
    extract_species,
    extract_static,
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
