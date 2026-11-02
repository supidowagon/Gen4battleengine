"""The prep problem: resources, usable pool, selectors, levelled moves (plan 4c Task 3)."""

from dataclasses import replace

import pytest

from nuzline.data.learnsets import load_learnsets
from nuzline.prep.pool import pool_candidates
from nuzline.prep.problem import (
    PRESTATUS,
    Resources,
    bag_resources,
    build_problem,
    pick,
    usable_pool,
)
from nuzline.saves import load_save
from tests.engine.helpers import roark_party, roark_trainer, sample_party
from tests.prep.fixtures import candidate, problem
from tests.saves.synthetic import box_record, party_record
from tests.saves.writer import build_image


@pytest.fixture(scope="module")
def learnsets():
    return load_learnsets()


def _save(tmp_path, tables, names, party, boxes, bag=None):
    path = tmp_path / "prep.sav"
    path.write_bytes(
        build_image(
            party=[party_record(tables, names, *row, **kw) for row, kw in party],
            boxes={(0, i): box_record(tables, names, *row) for i, row in enumerate(boxes)},
            bag=bag or {"tms_hms": [(names.item("TM10"), 1)]},
        )
    )
    return load_save(path, tables=tables, names=names)


def _build(tables, names, save, learnsets, **kw):
    kw.setdefault("cap", 16)
    return build_problem(
        tables, names, save, roark_trainer(), roark_party(names), 0, learnsets, **kw
    )


def test_resources_from_bag(tables, names, learnsets):
    scale, potion = names.item("Heart Scale"), names.item("Potion")
    tm10, hm01, oran = names.item("TM10"), names.item("HM01"), names.item("Oran Berry")
    bag = {
        "items": [(scale, 1), (potion, 3)],
        "tms_hms": [(tm10, 1), (hm01, 1)],
        "berries": [(oran, 2)],
    }
    assert bag_resources(bag, tables, names, learnsets, ()) == Resources(
        1, ((tm10, 1),), (hm01,), ((oran, 2),)
    )
    assert bag_resources(bag, tables, names, learnsets, {scale}).heart_scales == 0
    assert bag_resources(bag, tables, names, learnsets, {tm10}).tms == ()
    assert bag_resources(bag, tables, names, learnsets, {oran}).held == ()


def test_usable_pool_excludes_and_trims(tables, names):
    base = sample_party(names)[0]
    metronome, fury = names.move("Metronome"), names.move("Fury Swipes")
    scratch = names.move("Scratch")
    spinarak = candidate(
        names,
        replace(base, species=names.species("Spinarak"), moves=(metronome, scratch, 0, 0)),
        0,
    )
    furret_spec = replace(
        base, species=names.species("Furret"), moves=(scratch, 0, 0, 0), ability_slot=1
    )
    furret = candidate(names, furret_spec, 1, relearner=(metronome, fury))
    plain = candidate(names, furret_spec, 2)
    pool, lines = usable_pool(tables, names, [spinarak, furret, plain])
    assert [c.label for c in pool] == [furret.label, plain.label]
    assert pool[0].relearner == (fury,)
    assert pool[1] == plain
    assert any(line.startswith(spinarak.label + ": left out: Metronome") for line in lines), lines
    assert any(line.startswith(furret.label + ": Metronome not offered (") for line in lines)
    assert not any(plain.label in line for line in lines)


def test_unsupported_item_is_taken_off(tables, names):
    spec = replace(sample_party(names)[0], item=names.item("Leftovers"))
    pool, lines = usable_pool(tables, names, [candidate(names, spec, 0)])
    assert pool[0].spec.item == 0
    assert len(lines) == 1 and "Leftovers taken off" in lines[0]


def test_pick_ambiguous_and_locations(names):
    spec = replace(sample_party(names)[0], species=names.species("Venonat"))
    first = replace(candidate(names, spec, 1), location=("box", 0, 1), label="box 1 slot 1 VENONAT")
    second = replace(
        candidate(names, spec, 2), location=("box", 0, 2), label="box 1 slot 2 VENONAT (Bug)"
    )
    pool = [first, second]
    with pytest.raises(ValueError, match="box:B:S") as e:
        pick(pool, names, "venonat")
    assert first.label in str(e.value) and second.label in str(e.value)
    assert pick(pool, names, "box:1:2") == 1
    assert pick(pool, names, "bug") == 1
    party = candidate(names, spec, 0)
    assert pick([party, *pool], names, "party:0") == 0
    with pytest.raises(ValueError, match="party:9"):
        pick([party, *pool], names, "party:9")
    with pytest.raises(ValueError, match="Geodude"):
        pick(pool, names, "Geodude")


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"deaths": 4}, "4"),
        ({"reserve": ["Potionx"]}, "Potionx"),
        ({"prestatus": ["sleep"]}, "sleep"),
        ({"protect": ["Furret"], "exclude": ["party:0"]}, "both protected and excluded"),
        ({"exclude": ["party:0", "party:1"]}, "no usable Pokemon"),
    ],
)
def test_build_problem_errors(tmp_path, tables, names, learnsets, kw, message):
    save = _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch", "Defense Curl")), {"ability_slot": 1}),
            (("Geodude", 16, ("Tackle", "Defense Curl")), {}),
        ],
        [],
    )
    with pytest.raises(ValueError, match=message):
        _build(tables, names, save, learnsets, **kw)


def test_protecting_a_left_out_pokemon_is_an_error(tmp_path, tables, names, learnsets):
    save = _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch",)), {"ability_slot": 1}),
            (("Spinarak", 16, ("Metronome",)), {}),
        ],
        [],
    )
    with pytest.raises(ValueError, match="Metronome"):
        _build(tables, names, save, learnsets, protect=["party:1"])
    p, warnings = _build(tables, names, save, learnsets, protect=["party:0"])
    assert p.protected == {0} and len(p.candidates) == 1
    assert any("SPINARAK" in w and "left out" in w for w in warnings)


def test_build_problem_fields(tmp_path, tables, names, learnsets):
    scale = names.item("Heart Scale")
    save = _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch",)), {"ability_slot": 1}),
            (("Geodude", 16, ("Tackle",)), {}),
        ],
        [],
        bag={"items": [(scale, 2)]},
    )
    p, warnings = _build(
        tables,
        names,
        save,
        learnsets,
        deaths=1,
        protect=["geodude"],
        reserve=["Leftovers"],
        prestatus=["Poison"],
    )
    assert p.deaths == 1 and p.protected == {1} and p.prestatus == (PRESTATUS["poison"],)
    assert p.source == f"save:{save.sha256}" and p.cap == 16
    assert p.resources.heart_scales == 2
    assert warnings == ["Leftovers: reserved but not in the bag"]
    q, _ = _build(tables, names, save, learnsets, reserve=["Heart Scale"])
    assert q.resources.heart_scales == 0


def test_levelled_moves(tmp_path, tables, names, learnsets):
    save = _save(
        tmp_path,
        tables,
        names,
        [(("Furret", 16, ("Scratch", "Defense Curl")), {"ability_slot": 1})],
        [("Kricketot", 5, ("Growl", "Bide"))],
    )
    p, _ = _build(tables, names, save, learnsets)
    assert [c.label for c in p.candidates] == ["party slot 0 FURRET", "box 1 slot 0 KRICKETOT"]
    assert p.levelled == ((), ((16, names.move("Bug Bite")),))
    q, _ = _build(tables, names, save, learnsets, cap=15)
    assert q.levelled == ((), ())
    cands, _ = pool_candidates(save, tables, names, learnsets, 16)
    assert problem(tables, names, cands).levelled == p.levelled
