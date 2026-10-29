import json
from pathlib import Path

import pytest

from tools.extract_decomp import extract_static

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
