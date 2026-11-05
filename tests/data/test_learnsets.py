import pytest

from nuzline.data.learnsets import load_learnsets
from nuzline.data.tables import load_names


@pytest.fixture(scope="module")
def names():
    return load_names()


@pytest.fixture(scope="module")
def learnsets():
    return load_learnsets()


def _relearn(names, learnsets, species: str, level: int) -> tuple[str, ...]:
    moves = learnsets.relearner_moves(names.species(species), level)
    return tuple(names.move_names[m] for m in moves)


@pytest.mark.parametrize(
    ("species", "expected"),
    [
        ("Furret", ("Scratch", "Foresight", "Defense Curl", "Quick Attack", "Fury Swipes")),
        ("Kricketot", ("Growl", "Bide", "Bug Bite")),
        ("Venonat", ("Tackle", "Disable", "Foresight", "Supersonic", "Confusion", "PoisonPowder")),
        ("Bidoof", ("Tackle", "Growl", "Defense Curl", "Rollout")),
        (
            "Geodude",
            ("Tackle", "Defense Curl", "Mud Sport", "Rock Polish", "Rock Throw", "Magnitude"),
        ),
    ],
)
def test_relearner_moves_at_16(names, learnsets, species, expected):
    assert _relearn(names, learnsets, species, 16) == expected


def test_relearner_moves_respects_level(names, learnsets):
    assert "Bug Bite" not in _relearn(names, learnsets, "Kricketot", 15)


def test_level_up_is_pairs_in_learn_order(names, learnsets):
    pairs = learnsets.level_up(names.species("Kricketot"))
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    assert [lv for lv, _ in pairs] == sorted(lv for lv, _ in pairs)


def test_tm_moves(names, learnsets):
    assert learnsets.tm_move(names.item("TM10")) == names.move("Hidden Power")
    assert learnsets.tm_move(names.item("HM01")) == names.move("Cut")
    assert learnsets.tm_move(names.item("Potion")) == 0
    assert learnsets.tm_move(0) == 0


def test_tm_compatibility(names, learnsets):
    tm10 = names.item("TM10")
    for s in (
        "Furret",
        "Staravia",
        "Monferno",
        "Zubat",
        "Spinarak",
        "Psyduck",
        "Bidoof",
        "Venonat",
        "Geodude",
    ):
        assert learnsets.can_learn_tm(names.species(s), tm10), s
    for s in ("Kricketot", "Magikarp"):
        assert not learnsets.can_learn_tm(names.species(s), tm10), s
    assert not learnsets.can_learn_tm(names.species("Furret"), names.item("Potion"))
