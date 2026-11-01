from pathlib import Path

import pytest

from nuzline.data.party import load_party
from nuzline.engine.ids import Status

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

_SHINX = (
    '[[pokemon]]\nspecies = "Shinx"\nlevel = 5\nnature = "{nature}"\n'
    "ivs = [1,2,3,4,5,6]\nevs = [0,0,0,0,0,4]\n"
    'moves = ["Tackle"]\nitem = ""\nability = "{ability}"\ngender = "female"\n'
)


def test_sample_party_loads(names):
    p = load_party(FIXTURES / "sample_party.toml", names)
    assert len(p) == 6 and p[0].species == names.species("Monferno") and p[0].level == 14
    assert p[3].item == names.item("Oran Berry") and p[2].item == names.item("Sitrus Berry")


def test_sample_party_fields(names):
    p = load_party(FIXTURES / "sample_party.toml", names)
    assert p[0].moves == tuple(names.move(x) for x in ("Scratch", "Leer", "Ember", "Mach Punch"))
    assert p[4].moves == (names.move("Tackle"), names.move("Leer"), names.move("Spark"), 0)
    assert all(m.ivs == (20,) * 6 and m.evs == (0,) * 6 for m in p)
    assert all(m.gender == 0 and m.nature == 0 and m.current_hp is None for m in p)
    # Every sample ability is its species' first: Shinx (Rivalry / Intimidate) and
    # Budew (Natural Cure / Poison Point) included.
    assert [m.ability_slot for m in p] == [0] * 6
    assert p[0].friendship == names.species_base_friendship(p[0].species)


def test_ability_resolves_to_its_slot(names, tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(_SHINX.format(nature="Adamant", ability="Intimidate"), encoding="utf-8")
    (mon,) = load_party(f, names)
    assert mon.ability_slot == 1 and mon.gender == 1
    assert mon.nature == names.nature_names.index("Adamant")
    assert mon.ivs == (1, 2, 3, 4, 5, 6) and mon.evs == (0, 0, 0, 0, 0, 4)
    assert mon.moves == (names.move("Tackle"), 0, 0, 0)


def test_ability_the_species_cannot_have_is_rejected(names, tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(_SHINX.format(nature="Hardy", ability="Blaze"), encoding="utf-8")
    with pytest.raises(ValueError, match="Blaze"):
        load_party(f, names)


def test_fainted_party_member_keeps_hp_zero(names):
    p = load_party(FIXTURES / "sample_party_fainted.toml", names)
    assert p[5].current_hp == 0


def test_party_toml_new_keys(tmp_path, names):
    f = tmp_path / "p.toml"
    f.write_text(
        _SHINX.format(nature="Hardy", ability="Rivalry")
        + 'status = "sleep"\nsleep_turns = 2\npp = [10]\nstats = [30, 11, 12, 13, 14, 15]\n',
        encoding="utf-8",
    )
    (mon,) = load_party(f, names)
    assert mon.status == Status.SLEEP and mon.status_counter == 2
    assert mon.pp == (10, 0, 0, 0) and mon.stats == (30, 11, 12, 13, 14, 15)


def test_party_toml_new_keys_default_to_none(names):
    (mon, *_) = load_party(FIXTURES / "sample_party.toml", names)
    assert mon.pp is None and mon.stats is None
    assert mon.status == Status.NONE and mon.status_counter == 0


@pytest.mark.parametrize(
    "extra, match",
    [
        ("sleep_turns = 2\n", "sleep_turns"),
        ('status = "poison"\nsleep_turns = 2\n', "sleep_turns"),
        ('status = "drowsy"\n', "status"),
        ("pp = [1, 2]\nmoves_extra = 1\n" if False else "pp = [1, 2]\n", "pp"),
        ("pp = [1, 2, 3, 4, 5]\n", "pp"),
        ("stats = [1, 2, 3]\n", "stats"),
        ("stats = [0, 11, 12, 13, 14, 15]\n", "stats must be 1-999"),
        ("stats = [30, 11, 12, 13, 14, 1000]\n", "stats must be 1-999"),
        ('status = "sleep"\nsleep_turns = 0\n', "sleep_turns must be 1-7"),
        ('status = "sleep"\nsleep_turns = 8\n', "sleep_turns must be 1-7"),
        ("pp = [65]\n", "pp values must be 0-64"),
        ("pp = [-1]\n", "pp values must be 0-64"),
    ],
)
def test_party_toml_new_key_errors(tmp_path, names, extra, match):
    f = tmp_path / "p.toml"
    f.write_text(_SHINX.format(nature="Hardy", ability="Rivalry") + extra, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_party(f, names)
