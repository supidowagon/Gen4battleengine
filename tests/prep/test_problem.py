"""The prep problem: resources, usable pool, selectors, levelled moves (plan 4c Task 3)."""

from dataclasses import replace

import numpy as np
import pytest

from nuzline.data.forms import personal_row
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
from tests.engine.helpers import kaizo_names_with, roark_party, roark_trainer, sample_party
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
    assert bag_resources(bag, tables, names, learnsets, ()) == (
        Resources(1, ((tm10, 1),), (hm01,), ((oran, 2),)),
        [],
    )
    assert bag_resources(bag, tables, names, learnsets, {scale})[0].heart_scales == 0
    assert bag_resources(bag, tables, names, learnsets, {tm10})[0].tms == ()
    assert bag_resources(bag, tables, names, learnsets, {oran})[0].held == ()


def test_bag_resources_warns_on_unsupported_item(tables, names, learnsets):
    # Razor Claw is unsupported (Quick Claw was, until K3b2 Task 8).
    rock, honey = names.item("Razor Claw"), names.item("Honey")
    bag = {"items": [(rock, 1), (honey, 1)], "berries": [(names.item("Oran Berry"), 1)]}
    resources, warnings = bag_resources(bag, tables, names, learnsets, ())
    assert rock not in dict(resources.held)
    assert warnings == ["Razor Claw: not offered (hold effect not supported)"]


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
    pool, lines = usable_pool(tables, names, [spinarak, furret, plain], ())
    assert [c.label for c in pool] == [furret.label, plain.label]
    assert pool[0].relearner == (fury,)
    assert pool[1] == plain
    assert any(line.startswith(spinarak.label + ": left out: Metronome") for line in lines), lines
    assert any(line.startswith(furret.label + ": Metronome not offered (") for line in lines)
    assert not any(plain.label in line for line in lines)


def test_unsupported_item_is_taken_off(tables, names):
    spec = replace(sample_party(names)[0], item=names.item("Scope Lens"))
    pool, lines = usable_pool(tables, names, [candidate(names, spec, 0)], ())
    assert pool[0].spec.item == 0
    assert len(lines) == 1 and "Scope Lens taken off" in lines[0]


def test_pick_ambiguous_and_locations(names):
    spec = replace(sample_party(names)[0], species=names.species("Venonat"))
    first = replace(candidate(names, spec, 1), location=("box", 0, 1), label="box 1 slot 1 VENONAT")
    second = replace(
        candidate(names, spec, 2),
        location=("box", 0, 2),
        label="box 1 slot 2 VENONAT",
        nickname="Bug",
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
    immunity = (names.ability("Immunity"), int(PRESTATUS["poison"]))
    assert immunity in p.blocking_abilities  # from the game's own Names (final review I4)
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


def test_pick_nickname_on_pool_candidates(tmp_path, tables, names, learnsets):
    inv = {v: k for k, v in names.charmap.items() if len(v) == 1}
    p = tmp_path / "nick.sav"
    p.write_bytes(
        build_image(
            party=[
                party_record(
                    tables,
                    names,
                    "Furret",
                    16,
                    ("Scratch",),
                    nickname_codes=tuple(inv[c] for c in "Zorro"),
                )
            ]
        )
    )
    save = load_save(p, tables=tables, names=names)
    cands, _ = pool_candidates(save, tables, names, learnsets, 16)
    assert cands[0].label == "party slot 0 FURRET" and cands[0].nickname == "Zorro"
    assert pick(cands, names, "zorro") == 0
    assert pick(cands, names, "furret") == 0


def test_left_out_party_lists_excluded_and_unsupported(tmp_path, tables, names, learnsets):
    save = _save(
        tmp_path,
        tables,
        names,
        [
            (("Furret", 16, ("Scratch",)), {"ability_slot": 1}),
            (("Spinarak", 16, ("Metronome",)), {}),
            (("Geodude", 16, ("Tackle",)), {}),
        ],
        [],
    )
    p, _ = _build(tables, names, save, learnsets, exclude=["party:2"])
    assert [c.location[2] for c in p.candidates] == [0]
    assert p.left_out_party == ((1, "SPINARAK"), (2, "GEODUDE"))


# Gen IV move index of Water Gun (names.move("Water Gun")): an implemented move no sample or
# Roark party member knows, so only the TM or relearner lists below carry it.
MOVE_X = 55
ROM_LINE = "Kaizo ROM: c is not modelled and this fight reaches it (docs/kaizo/rom_diff.md)"


def test_kaizo_condition_on_a_tm_move_drops_it(tables, names, vanilla_manifest):
    # Final review I1: a TM or relearner move a not_reached condition lists is not offered;
    # a candidate that already knows it is left out.
    spec = sample_party(names)[0]
    tm10 = names.item("TM10")
    offered = candidate(names, spec, 0, relearner=(MOVE_X,), tms=((tm10, MOVE_X),))
    knows = candidate(names, replace(spec, moves=(MOVE_X, *spec.moves[1:])), 1)
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_X]}])
    pool, lines = usable_pool(tables, kaizo, [offered, knows], (), trainer=roark_trainer())
    assert [c.label for c in pool] == [offered.label]
    assert pool[0].relearner == () and pool[0].tms == ()
    assert lines == [
        f"{offered.label}: Water Gun not offered ({ROM_LINE})",
        f"{offered.label}: Water Gun not offered ({ROM_LINE})",
        f"{knows.label}: left out: Water Gun ({ROM_LINE})",
    ]
    # Vanilla, or no trainer: nothing changes.
    for n, t in ((names, roark_trainer()), (kaizo, None)):
        assert usable_pool(tables, n, [offered, knows], (), trainer=t) == (
            [offered, knows],
            [],
        )


def test_kaizo_condition_on_an_ability_leaves_the_candidate_out(tables, names, vanilla_manifest):
    spec = sample_party(names)[0]
    row = personal_row(tables, spec.species, spec.form)
    ability = int(np.asarray(tables.species_abilities_decomp)[row][spec.ability_slot])
    kaizo = kaizo_names_with(vanilla_manifest, [{"abilities": [ability]}])
    cand = candidate(names, spec, 0)
    pool, lines = usable_pool(tables, kaizo, [cand], (), trainer=roark_trainer())
    assert pool == []
    assert lines == [f"{cand.label}: left out: {names.ability_names[ability]} ({ROM_LINE})"]
    assert usable_pool(tables, names, [cand], (), trainer=roark_trainer()) == ([cand], [])


def test_kaizo_condition_on_a_held_item_takes_it_off(tables, names, vanilla_manifest):
    oran = names.item("Oran Berry")
    spec = replace(sample_party(names)[0], item=oran)
    kaizo = kaizo_names_with(vanilla_manifest, [{"items": [oran]}])
    pool, lines = usable_pool(
        tables, kaizo, [candidate(names, spec, 0)], (), trainer=roark_trainer()
    )
    assert pool[0].spec.item == 0
    assert lines == [f"party slot 0 MONFERNO: Oran Berry taken off ({ROM_LINE})"]


def test_kaizo_condition_drops_levelled_moves_and_bag_items(
    tmp_path, tables, names, learnsets, vanilla_manifest
):
    oran = names.item("Oran Berry")
    save = _save(
        tmp_path,
        tables,
        names,
        [(("Furret", 16, ("Scratch", "Defense Curl")), {"ability_slot": 1})],
        [("Kricketot", 5, ("Growl", "Bide"))],
        bag={"berries": [(oran, 2)]},
    )
    kaizo = kaizo_names_with(
        vanilla_manifest, [{"moves": [names.move("Bug Bite")], "items": [oran]}]
    )
    p, warnings = _build(tables, kaizo, save, learnsets)
    assert p.levelled == ((), ())
    assert p.resources.held == ()
    assert f"Oran Berry: not offered ({ROM_LINE})" in warnings
    q, _ = _build(tables, names, save, learnsets)
    assert q.levelled == ((), ((16, names.move("Bug Bite")),))
    assert q.resources.held == ((oran, 2),)


def test_ghost_curse_is_not_offered_in_prep(tables, names, learnsets):
    """A Ghost-type user's Curse takes the Ghost branch (effect_script_0109.s:5-6), which the
    engine does not model: prep refuses it per Pokemon, as coverage does (Task K3b1-7)."""
    from nuzline.prep.problem import GHOST_CURSE_REFUSAL, levelled_moves

    base = sample_party(names)[0]
    curse, tackle = names.move("Curse"), names.move("Tackle")
    shuppet = names.species("Shuppet")
    slot = names.species_abilities(shuppet).index(names.ability("Insomnia"))
    ghost = replace(base, species=shuppet, form=0, ability_slot=slot, level=16)
    assert (13, curse) in learnsets.level_up(shuppet)
    knows = candidate(names, replace(ghost, moves=(curse, 0, 0, 0)), 0)
    could = candidate(names, replace(ghost, moves=(tackle, 0, 0, 0)), 1, relearner=(curse,),
                      tms=((names.item("TM10"), curse),), candies=4)  # fmt: skip
    furret = replace(base, species=names.species("Furret"), ability_slot=1, level=16)
    plain = candidate(names, replace(furret, moves=(curse, 0, 0, 0)), 2, relearner=(curse,))
    pool, lines = usable_pool(tables, names, [knows, could, plain], ())
    assert [c.label for c in pool] == [could.label, plain.label]
    assert f"{knows.label}: left out: Curse ({GHOST_CURSE_REFUSAL})" in lines
    assert lines.count(f"{could.label}: Curse not offered ({GHOST_CURSE_REFUSAL})") == 2
    assert pool[0].relearner == () and pool[0].tms == ()
    assert pool[1].relearner == (curse,) and not any(plain.label in ln for ln in lines)
    # Levelled moves: Shuppet's level-13 Curse (save level 12 -> cap 16) is dropped.
    levelled = levelled_moves(tables, [pool[0]], learnsets, 16, ())
    assert all(move != curse for _, move in levelled[0])
    ungated = [m for lv, m in learnsets.level_up(shuppet) if 12 < lv <= 16]
    assert curse in ungated


def test_opponent_explosion_leaves_a_low_defense_candidate_out(tmp_path, tables, names, learnsets):
    """An opponent's Explosion against a candidate whose Defense is below
    ``HALVED_DEFENSE_MIN``: the halved Defense at -6 gives stageDivisor 0
    (battle_lib.c:6943-6975), which the engines do not model, so prep leaves the candidate
    out by name as coverage refuses the fight (final review, K3b1). Defense 8 is kept."""
    from nuzline.engine.coverage import defense_stat
    from nuzline.prep.pool import check_pool

    save = _save(
        tmp_path,
        tables,
        names,
        [(("Pichu", 5, ("Thundershock",)), {}), (("Diglett", 5, ("Scratch",)), {})],
        [],
    )
    cands, _ = pool_candidates(save, tables, names, learnsets, 5)
    pichu, diglett = cands
    assert [defense_stat(tables, c.spec) for c in cands] == [7, 8]
    explosion = names.move("Explosion")
    roark = roark_party(names)
    boomer = [replace(roark[0], moves=(explosion, 0, 0, 0)), *roark[1:]]
    p, warnings = build_problem(tables, names, save, roark_trainer(), boomer, 0, learnsets, cap=5)
    line = (
        f"{pichu.label}: left out: Defense 7 against the opponent's Explosion "
        "(zero damage divisor at -6 Defense, not implemented)"
    )
    assert [c.label for c in p.candidates] == [diglett.label]
    assert line in warnings
    assert p.left_out_party == ((0, "PICHU"),)
    missing = check_pool(tables, names, cands, boomer).missing
    assert missing == [
        f"{pichu.label}: Defense 7 against the opponent's Explosion (opponent move effect UNSUPPORTED)"
    ]
    # Without the Explosion both are kept; a fainted Explosion user threatens nobody.
    q, lines = build_problem(tables, names, save, roark_trainer(), roark, 0, learnsets, cap=5)
    assert len(q.candidates) == 2 and not any("Defense" in ln for ln in lines)
    fainted = [replace(boomer[0], current_hp=0), *roark[1:]]
    assert check_pool(tables, names, cands, fainted).missing == []


# --- K3d-exp T7: moves learnable by levelling up in the fight ---------------------------

STARLY_L17 = 3120  # exp_table[3][17]; Roark's Geodude L12 bounds a payout at 188


def _starly_near_17(names, **changes):
    """Starly L16 that Roark's Geodude can level to 17, where it learns Endeavor (an
    unsupported effect) into its empty last slot; ``coverage.learnable_moves``."""
    gap = (names.move("Tackle"), names.move("Growl"), names.move("Quick Attack"), 0)
    spec = replace(
        sample_party(names)[3], level=16, **{"exp": STARLY_L17 - 188, "moves": gap, **changes}
    )
    return candidate(names, spec, 3)


def test_a_learnable_unsupported_move_leaves_the_candidate_out(tables, names):
    geodude = roark_party(names)[:1]
    near = _starly_near_17(names)
    pool, lines = usable_pool(tables, names, [near], geodude)
    assert pool == []
    assert lines == [
        "party slot 3 STARLY: left out: Endeavor (level-up in battle, move effect not implemented)"
    ]
    far = _starly_near_17(names, exp=STARLY_L17 - 189)
    assert usable_pool(tables, names, [far], geodude) == ([far], [])
    # Four moves: Endeavor is only a prompt, masked in the learn options (ruling D10).
    full = _starly_near_17(names, moves=sample_party(names)[3].moves)
    assert usable_pool(tables, names, [full], geodude) == ([full], [])


def test_check_pool_grades_learnable_moves(tables, names):
    from nuzline.engine.coverage import Support
    from nuzline.prep.pool import check_pool

    geodude = roark_party(names)[:1]
    report = check_pool(tables, names, [_starly_near_17(names)], geodude)
    assert report.level == Support.NONE
    assert report.missing == [
        "party slot 3 STARLY: Endeavor (level-up in battle, move effect UNSUPPORTED)"
    ]
    far = _starly_near_17(names, exp=STARLY_L17 - 189)
    assert check_pool(tables, names, [far], geodude).missing == []
    full = _starly_near_17(names, moves=sample_party(names)[3].moves)
    assert check_pool(tables, names, [full], geodude).level == Support.EXACT


def test_usable_pool_applies_the_exp_refusals(tables, names):
    """``coverage.exp_refusals`` per candidate: junk PP Ups in an empty slot (§4.6)."""
    base = sample_party(names)[0]
    junk = candidate(
        names, replace(base, moves=(*base.moves[:3], 0), pp=None, pp_ups=(0, 0, 0, 1)), 0
    )
    pool, lines = usable_pool(tables, names, [junk], roark_party(names))
    assert pool == []
    assert lines == [
        (
            "party slot 0 MONFERNO: left out: empty move slot 3 has 1 PP Ups, which a move learned "
            "there inherits"
        )
    ]
