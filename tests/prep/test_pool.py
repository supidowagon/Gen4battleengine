"""The candidate pool and its coverage report (plan 4a Task 10)."""

import numpy as np
import pytest

from nuzline.ai.policy import player_ability_refusal, player_move_refusal
from nuzline.data.learnsets import load_learnsets
from nuzline.engine.coverage import Support
from nuzline.prep.pool import Candidate, check_pool, pool_candidates
from nuzline.saves import load_save
from tests.engine.helpers import kaizo_names_with, roark_trainer, sample_party
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
    # K3d-exp T3: exp at the cap (the base exp of the cap for the boxed, raised ones) and
    # the boost resolved with the save's trainer (the synthetic records are its own).
    rates, table = tables.species_exp_rate, tables.exp_table
    assert [c.exp for c in cands] == [int(table[int(rates[c.spec.species]), 16]) for c in cands]
    assert [c.spec.exp_boost for c in cands] == [100] * 4


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
    report = check_pool(tables, names, cands, ())
    assert report.missing == []
    assert report.level == Support.EXACT


def test_check_pool_names_unsupported(tmp_path, tables, names, learnsets, metronome_refused):
    """Review Focus 1: the boxed Pokemon and the move are named; level is NONE."""
    save = _save(
        tmp_path,
        tables,
        names,
        [(("Furret", 16, ("Scratch",)), {"ability_slot": 1})],
        [("Geodude", 6, ("Tackle",)), ("Monferno", 14, ("Metronome", "Scratch"))],
    )
    cands, _ = pool_candidates(save, tables, names, learnsets, 20)
    report = check_pool(tables, names, cands, ())
    # Geodude's relearner Selfdestruct is supported since Task K3b1-8 (SELF_DESTRUCT).
    assert not [line for line in report.missing if "GEODUDE" in line]
    assert "box 1 slot 1 MONFERNO: Metronome (current, move effect UNSUPPORTED)" in report.missing
    assert report.level == Support.NONE


@pytest.mark.fullset
def test_player_move_refusal(names):
    # Miracle Eye is modelled since E3-B2-10: no move is refused for its state now.
    assert player_move_refusal(names.move("Miracle Eye")) is None
    assert player_move_refusal(names.move("Foresight")) is None
    assert player_move_refusal(names.move("Hidden Power")) is None


@pytest.mark.fullset
def test_player_ability_refusal(names):
    assert player_ability_refusal(26)  # Levitate
    assert player_ability_refusal(51) is None  # Keen Eye


# Gen IV move index of Water Gun (names.move("Water Gun")): an implemented move no sample
# or Roark party member knows, so only a candidate's relearner list carries it.
MOVE_X = 55


def test_pool_candidate_reached_by_relearner_move(tables, names, vanilla_manifest):
    spec = sample_party(names)[0]
    cand = Candidate(("party", 0, 0), "party slot 0 MONFERNO", spec, 0, (MOVE_X,), ())
    other = Candidate(("party", 0, 1), "party slot 1 MONFERNO", spec, 0, (), ())
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_X]}])
    report = check_pool(tables, kaizo, [cand, other], (), trainer=roark_trainer())
    assert report.missing == [
        (
            "party slot 0 MONFERNO: Kaizo ROM: c is not modelled and this fight reaches it "
            "(docs/kaizo/rom_diff.md) (ROM change UNSUPPORTED)"
        )
    ]
    # Without the fight's trainer, or in vanilla, the pool refuses nothing for it.
    assert check_pool(tables, kaizo, [cand, other], ()).missing == []
    assert check_pool(tables, names, [cand, other], (), trainer=roark_trainer()).missing == []


def test_pool_candidate_reached_by_tm_move(tables, names, vanilla_manifest):
    spec = sample_party(names)[0]
    cand = Candidate(
        ("party", 0, 0), "party slot 0 MONFERNO", spec, 0, (), ((names.item("TM10"), MOVE_X),)
    )
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_X]}])
    assert len(check_pool(tables, kaizo, [cand], (), trainer=roark_trainer()).missing) == 1


@pytest.mark.parametrize(
    "cond", [{"trainers": [246]}, {"trainer_items": True}, {"formats": ["single"]}]
)
def test_pool_ignores_fight_level_conditions(tables, names, vanilla_manifest, cond):
    # Roark's id, bag and format hold, but they are the fight's report's lines, not a candidate's.
    spec = sample_party(names)[0]
    cands = [
        Candidate(("party", 0, i), f"party slot {i} MONFERNO", spec, 0, (), ()) for i in range(2)
    ]
    kaizo = kaizo_names_with(vanilla_manifest, [cond])
    assert check_pool(tables, kaizo, cands, (), trainer=roark_trainer()).missing == []


@pytest.mark.fullset
def test_pool_lists_holdable_bag_items(tables, names):
    from nuzline.engine.coverage import Support

    # Two held items still unsupported, picked from the coverage table at run time.
    refused = _refused_items(tables, names, 2)
    bag = [(names.item(n), 1) for n in (*refused, "Oran Berry", "Honey")]
    report = check_pool(tables, names, [], (), bag=bag)
    assert sorted(report.missing) == sorted(f"bag: {n} (held item UNSUPPORTED)" for n in refused)
    assert not any("Honey" in line for line in report.missing)
    assert report.level == Support.NONE
    only_oran = check_pool(tables, names, [], (), bag=[(names.item("Oran Berry"), 3)])
    assert only_oran.level == Support.EXACT and not only_oran.missing


def _refused_items(tables, names, k: int) -> list[str]:
    """The names of the first ``k`` held items the coverage table still maps to
    ``ItemId.UNSUPPORTED`` (picked at run time, not hard-coded)."""
    from nuzline.engine.ids import ItemId

    effects = np.asarray(tables.item_effect)
    ids = [i for i in range(1, len(effects)) if effects[i] == ItemId.UNSUPPORTED]
    return [names.item_names[i] for i in ids[:k]]
