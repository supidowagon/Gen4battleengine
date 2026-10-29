from pathlib import Path

import pytest

from nuzline.data.party import load_party
from nuzline.data.tables import load_names

TABLES = Path("data/gen4_platinum")
FIXTURES = Path("tests/fixtures")

_SHINX = (
    '[[pokemon]]\nspecies = "Shinx"\nlevel = 5\nnature = "{nature}"\n'
    "ivs = [1,2,3,4,5,6]\nevs = [0,0,0,0,0,4]\n"
    'moves = ["Tackle"]\nitem = ""\nability = "{ability}"\ngender = "female"\n'
)


@pytest.fixture(scope="module")
def n():
    return load_names(TABLES)


def test_sample_party_loads(n):
    p = load_party(FIXTURES / "sample_party.toml", n)
    assert len(p) == 6 and p[0].species == n.species("Monferno") and p[0].level == 14
    assert p[3].item == n.item("Oran Berry") and p[2].item == n.item("Sitrus Berry")


def test_sample_party_fields(n):
    p = load_party(FIXTURES / "sample_party.toml", n)
    assert p[0].moves == tuple(n.move(x) for x in ("Scratch", "Leer", "Ember", "Mach Punch"))
    assert p[4].moves == (n.move("Tackle"), n.move("Leer"), n.move("Spark"), 0)
    assert all(m.ivs == (20,) * 6 and m.evs == (0,) * 6 for m in p)
    assert all(m.gender == 0 and m.nature == 0 and m.current_hp is None for m in p)
    # Every sample ability is its species' first: Shinx (Rivalry / Intimidate) and
    # Budew (Natural Cure / Poison Point) included.
    assert [m.ability_slot for m in p] == [0] * 6
    assert p[0].friendship == n.species_base_friendship(p[0].species)


def test_ability_resolves_to_its_slot(n, tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(_SHINX.format(nature="Adamant", ability="Intimidate"), encoding="utf-8")
    (mon,) = load_party(f, n)
    assert mon.ability_slot == 1 and mon.gender == 1
    assert mon.nature == n.nature_names.index("Adamant")
    assert mon.ivs == (1, 2, 3, 4, 5, 6) and mon.evs == (0, 0, 0, 0, 0, 4)
    assert mon.moves == (n.move("Tackle"), 0, 0, 0)


def test_ability_the_species_cannot_have_is_rejected(n, tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(_SHINX.format(nature="Hardy", ability="Blaze"), encoding="utf-8")
    with pytest.raises(ValueError, match="Blaze"):
        load_party(f, n)


def test_fainted_party_member_keeps_hp_zero(n):
    p = load_party(FIXTURES / "sample_party_fainted.toml", n)
    assert p[5].current_hp == 0
