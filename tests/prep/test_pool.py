"""The candidate pool and its coverage report (plan 4a Task 10)."""

import pytest

from nuzline.ai.policy import player_ability_refusal, player_move_refusal
from nuzline.data.learnsets import load_learnsets
from nuzline.engine.coverage import Support
from nuzline.prep.pool import check_pool, pool_candidates
from nuzline.saves import load_save
from tests.saves.synthetic import box_record, party_record
from tests.saves.writer import build_image


@pytest.fixture(scope="module")
def learnsets():
    return load_learnsets()


def _save(tmp_path, tables, names, party, boxes, *, tms=("TM10",)):
    path = tmp_path / "pool.sav"
    bag = {"tms_hms": [(names.item(t), 1) for t in tms]}
    path.write_bytes(
        build_image(
            party=[party_record(tables, names, *row, **kw) for row, kw in party],
            boxes={(0, i): box_record(tables, names, *row) for i, row in enumerate(boxes)},
            bag=bag,
        )
    )
    return load_save(path, tables=tables, names=names)


@pytest.fixture
def pool_save(tmp_path, tables, names):
    return _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch", "Defense Curl")), {"ability_slot": 1}),
            (("Geodude", 16, ("Tackle", "Defense Curl")), {}),
        ],
        [
            ("Kricketot", 5, ("Growl", "Bide")),
            ("Bidoof", 4, ("Tackle",)),
        ],
    )


def test_candidates(pool_save, tables, names, learnsets):
    cands, left_out = pool_candidates(pool_save, tables, names, learnsets, 16)
    assert left_out == []
    assert [c.label for c in cands] == [
        "party slot 0 FURRET",
        "party slot 1 GEODUDE",
        "box 1 slot 0 KRICKETOT",
        "box 1 slot 1 BIDOOF",
    ]
    assert [c.spec.level for c in cands] == [16] * 4
    assert [c.candies for c in cands] == [0, 0, 11, 12]
    kricketot = cands[2]
    assert kricketot.relearner == (names.move("Bug Bite"),)
    assert kricketot.tms == ()
    tm10 = names.item("TM10")
    assert cands[0].tms == ((tm10, names.move("Hidden Power")),)


def test_egg_is_left_out(tmp_path, tables, names, learnsets):
    save = _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch",)), {"ability_slot": 1}),
            (("Starly", 5, ("Tackle",)), {"egg": True}),
        ],
        [],
    )
    cands, left_out = pool_candidates(save, tables, names, learnsets, 16)
    assert [c.label for c in cands] == ["party slot 0 FURRET"]
    assert len(left_out) == 1 and "party slot 1" in left_out[0] and "egg" in left_out[0]


def test_check_pool_full(pool_save, tables, names, learnsets):
    cands, _ = pool_candidates(pool_save, tables, names, learnsets, 16)
    report = check_pool(tables, names, cands)
    assert report.missing == []
    assert report.level == Support.EXACT


def test_check_pool_names_unsupported(tmp_path, tables, names, learnsets):
    """Review Focus 1: the boxed Pokemon and the move are named; level is NONE."""
    save = _save(
        tmp_path,
        tables,
        names,
        [(("Furret", 16, ("Scratch",)), {"ability_slot": 1})],
        [("Geodude", 6, ("Tackle",)), ("Monferno", 14, ("Metronome", "Scratch"))],
    )
    cands, _ = pool_candidates(save, tables, names, learnsets, 20)
    report = check_pool(tables, names, cands)
    assert (
        "box 1 slot 0 GEODUDE: Selfdestruct (relearner, move effect UNSUPPORTED)" in report.missing
    )
    assert "box 1 slot 1 MONFERNO: Metronome (current, move effect UNSUPPORTED)" in report.missing
    assert report.level == Support.NONE


def test_player_move_refusal(names):
    assert player_move_refusal(names.move("Miracle Eye"))
    assert player_move_refusal(names.move("Foresight")) is None
    assert player_move_refusal(names.move("Hidden Power")) is None


def test_player_ability_refusal(names):
    assert player_ability_refusal(26)  # Levitate
    assert player_ability_refusal(51) is None  # Keen Eye
