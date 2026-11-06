import json
from pathlib import Path

import pytest

from nuzline.data.learnsets import TutorMove, load_learnsets
from nuzline.data.tables import load_names

DECOMP = Path("external/pokeplatinum")


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


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_form_rows_have_learnsets(learnsets, names):
    wash = json.loads((DECOMP / "res/pokemon/rotom/forms/wash/data.json").read_text("utf-8"))
    consts = (DECOMP / "generated" / "moves.txt").read_text("utf-8").split()
    assert [m for _, m in learnsets.level_up(479, form=2)] == [
        consts.index(n) for _, n in wash["learnset"]["by_level"]
    ]
    assert learnsets.can_learn_tm(479, names.item("TM17"), form=2) == (
        "TM17" in wash["learnset"]["by_tm"]
    )
    assert learnsets.relearner_moves(479, 1, form=2) == learnsets.relearner_moves(479, 1)
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        learnsets.level_up(479, form=6)


def test_learnsets_evolutions_and_egg_moves(learnsets):
    assert learnsets.evolutions(1) == ((4, 16, 2),)
    assert learnsets.evolutions(3) == ()  # Venusaur
    assert learnsets.egg_moves(1)[:3] == (113, 130, 219) and learnsets.egg_moves(3) == ()


@pytest.mark.parametrize("species", [-1, 496, 10_000])
def test_evolutions_and_egg_moves_check_the_species_range(learnsets, species):
    # Final review: evolutions raised a bare KeyError, egg_moves returned () for any id.
    for lookup in (learnsets.evolutions, learnsets.egg_moves):
        with pytest.raises(ValueError, match=f"species {species} out of range"):
            lookup(species)
    assert learnsets.evolutions(0) == () and learnsets.egg_moves(0) == ()  # SPECIES_NONE
    assert learnsets.evolutions(495) == () and learnsets.egg_moves(495) == ()  # Arceus


def test_can_learn_tutor(learnsets):
    assert learnsets.can_learn_tutor(479, 189) is True  # Rotom, Mud-Slap
    assert learnsets.can_learn_tutor(479, 291) is False  # Rotom, Dive
    assert learnsets.can_learn_tutor(479, 189, form=2) is True  # Rotom-Wash
    assert learnsets.tutor_moves()[0] == TutorMove(291, 2, 4, 2, 0, "route_212")
    assert len(learnsets.tutor_moves()) == 38
    assert learnsets.can_learn_tutor(479, 1) is False  # a move no tutor teaches
