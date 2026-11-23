"""GT-5 (E3-B3-5a): the battle-start friendship of a Gym Leader, Elite Four or Champion fight
(``prep.friendship``, ledger E3-B3-5.a) and the level-cap sit-out (``prep.gauntlet``, E3-B3-5.b).
The fight start from a Carry (lead, ``slot_of``, ``barred``, format per fight, ``write_carry``,
the re-run root) waits for GT-4 and E2's D3-8b."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nuzline.data.tables import to_numpy
from nuzline.data.trainer import Trainer
from nuzline.engine.damage import damage_table
from nuzline.engine.state import make_state
from nuzline.prep.caps import next_fight_cap
from nuzline.prep.friendship import (
    BEAT_GYM_FRIENDSHIP,
    GYM_E4_CHAMPION_CLASSES,
    battle_start_bounds,
    battle_start_friendship,
    battle_start_party,
    battle_start_spec,
    egg_match_bounds,
    fight_friendship,
    fight_label,
    friendship_readers,
    starts_with_friendship,
)
from nuzline.prep.gauntlet import CapSitOut, cap_sit_out
from nuzline.prep.setups import Setup, base_member, setup_specs
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import roark_party, roark_trainer, sample_party
from tests.prep.fixtures import candidate, problem

# --- the friendship change (src/pokemon.c:2635-2689) --------------------------------------


def _gain(friendship, *, luxury=False, egg_match=False, soothe=False):
    return (
        battle_start_friendship(friendship, luxury=luxury, egg_match=egg_match, soothe=soothe)
        - friendship
    )


def test_the_change_table_is_the_decomps():
    assert BEAT_GYM_FRIENDSHIP == (3, 2, 1)  # sFriendshipChangeTable, src/pokemon.c:2626


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (0, 3),
        (99, 102),  # tier 0 below 100: the value before the change picks the tier
        (100, 102),
        (199, 201),
        (200, 201),
        (254, 255),
        (255, 255),  # clamped
    ],
)
def test_tiers_at_the_limits_and_the_clamp(before, after):
    assert battle_start_friendship(before, luxury=False, egg_match=False, soothe=False) == after


def test_luxury_ball_and_egg_location_add_one_each():
    assert _gain(50, luxury=True) == 4
    assert _gain(50, egg_match=True) == 4
    assert _gain(50, luxury=True, egg_match=True) == 5
    assert _gain(150, luxury=True, egg_match=True) == 4
    assert _gain(250, luxury=True, egg_match=True) == 3


@pytest.mark.parametrize(
    ("before", "kwargs", "gain"),
    [
        (50, {}, 4),  # 3 * 150 / 100 = 4 in integers
        (150, {}, 3),  # 2 -> 3
        (250, {}, 1),  # 1 -> 1
        (50, {"luxury": True}, 6),  # (3 + 1) * 150 / 100
        (50, {"luxury": True, "egg_match": True}, 7),  # 5 -> 7
        (150, {"luxury": True}, 4),  # 3 -> 4
    ],
)
def test_soothe_bell_multiplies_the_change_in_integers(before, kwargs, gain):
    assert _gain(before, soothe=True, **kwargs) == gain


def test_the_soothe_bell_gain_is_clamped():
    assert battle_start_friendship(252, luxury=True, egg_match=True, soothe=True) == 255


# --- the classes (src/battle/battle_main.c:1376-1397) -------------------------------------


def test_the_thirteen_classes():
    assert len(GYM_E4_CHAMPION_CLASSES) == 13
    # Roark's class id (62, TRAINER_CLASS_LEADER_ROARK) in the shipped trainer data
    assert roark_trainer().class_id in GYM_E4_CHAMPION_CLASSES


def test_a_gym_e4_or_champion_class_on_either_side_starts_the_raise():
    assert starts_with_friendship([62])
    assert starts_with_friendship([69, None])  # Champion Cynthia
    assert starts_with_friendship([13, 79])  # enemy 2 is Volkner
    assert starts_with_friendship([65, 68])
    assert not starts_with_friendship([13])  # a regular class
    assert not starts_with_friendship([13, 63, 70, None])
    assert not starts_with_friendship([])


# --- one member (src/pokemon.c:2635-2689) -------------------------------------------------


def _spec(names, **kw):
    return replace(sample_party(names)[0], **kw)


def test_battle_start_spec_applies_every_term(tables, names):
    soothe = names.item("Soothe Bell")
    spec = _spec(names, friendship=50, item=soothe, friend_bonus=1, egg_location=77)
    got = battle_start_spec(tables, spec, (77,))
    assert got.friendship == 50 + 7  # (3 + 1 + 1) * 150 / 100
    assert got == replace(spec, friendship=57)


def test_the_egg_location_is_compared_with_the_label_text_id_not_the_header_id(tables, names):
    header_id, label = 454, 82  # Kaizo header 454's label text id is 82 (ledger E3-B3-5.a)
    spec = _spec(names, friendship=50, item=0, egg_location=label)
    assert battle_start_spec(tables, spec, (label,)).friendship == 54
    assert battle_start_spec(tables, spec, (header_id,)).friendship == 53


def test_one_label_fixes_the_fight_several_or_none_leave_it_open():
    # mapID is the header's label (battle_main.c:1054): headers sharing a label fix it.
    assert fight_label((7,)) == 7 and fight_label((7, 7)) == 7
    assert fight_label((7, 119)) is None  # Kaizo Byron 250, headers 2 and 34 (E3-B3-5.a)
    assert fight_label(()) is None  # no map places the trainer: any header
    assert egg_match_bounds(7, (7,)) == (True, True)
    assert egg_match_bounds(8, (7,)) == (False, False)
    assert egg_match_bounds(7, (7, 119)) == (False, True)
    assert egg_match_bounds(8, (7, 119)) == (False, False)  # no label is the egg location
    assert egg_match_bounds(0, ()) == (False, True)  # label 0 (Mystery Zone) may hold it


def test_an_open_label_carries_both_bounds(tables, names):
    hatched = _spec(names, friendship=50, item=0, egg_location=7)
    assert battle_start_bounds(tables, hatched, (7,)) == (54, 54)
    assert battle_start_bounds(tables, hatched, (7, 119)) == (53, 54)
    assert battle_start_bounds(tables, hatched, (8, 119)) == (53, 53)
    soothe = replace(hatched, item=names.item("Soothe Bell"))
    assert battle_start_bounds(tables, soothe, (7, 119)) == (54, 56)  # 3 * 1.5, 4 * 1.5


def test_each_reader_takes_the_bound_that_hurts_it():
    ret, frus = 121, 123  # effect_script_0121.s (Return), effect_script_0123.s (Frustration)
    # Return's power friendship * 10 / 25 rises: the low bound; Frustration's falls: the high.
    assert fight_friendship(102, 103, {ret}) == 102
    assert fight_friendship(102, 103, {frus}) == 103
    assert fight_friendship(102, 103, set()) == 102  # evolution's bound (pokemon.c:3565-3583)
    # Both: the bound where the stronger is weaker. At 102/103 Frustration (61/60) is stronger.
    assert fight_friendship(102, 103, {ret, frus}) == 103
    assert fight_friendship(200, 201, {ret, frus}) == 200  # Return 80 / 80: a tie keeps low
    assert fight_friendship(101, 102, {frus}) == 101  # 61 at both: a tie keeps low


def test_the_readers_come_from_the_moves(tables, names):
    ret, frus = names.move("Return"), names.move("Frustration")
    tackle, metronome = names.move("Tackle"), names.move("Metronome")
    copycat = names.move("Copycat")
    assert friendship_readers(tables, (ret, tackle), (frus,)) == {121}
    assert friendship_readers(tables, (frus,), ()) == {123}
    assert friendship_readers(tables, (tackle,), (ret, frus)) == frozenset()
    assert friendship_readers(tables, (metronome,), ()) == {121, 123}
    assert friendship_readers(tables, (copycat,), (frus, tackle)) == {123}


def test_an_open_label_gives_frustration_the_high_and_return_the_low_bound(tables, names):
    # 52 + 3 or + 4: 55 or 56, where Frustration's power (255 - f) * 10 / 25 falls, 80 to 79
    hatched = _spec(names, friendship=52, item=0, egg_location=7)
    assert battle_start_spec(tables, hatched, (7, 119), {123}).friendship == 56
    assert battle_start_spec(tables, hatched, (7, 119), {121}).friendship == 55
    assert battle_start_spec(tables, hatched, (7, 119)).friendship == 55  # no reader: low
    # a fixed label has one value whatever reads it
    assert battle_start_spec(tables, hatched, (7,), {121}).friendship == 56
    assert battle_start_spec(tables, hatched, (119,), {123}).friendship == 55


def test_egg_location_0_matches_a_label_0_header(tables, names):
    # MON_DATA_EGG_LOCATION is 0 for a Pokemon not hatched; LocationNames_Text_MysteryZone is
    # text id 0 (res/text/location_names.json), the label of 33 Kaizo headers (ledger E3-B3-5.a).
    plain = _spec(names, friendship=50, item=0, egg_location=0)
    assert battle_start_spec(tables, plain, (0,)).friendship == 54
    assert battle_start_spec(tables, plain, (8,)).friendship == 53


def test_a_party_is_raised_only_against_a_gym_class(tables, names):
    party = [_spec(names, friendship=f, item=0, egg_location=0) for f in (0, 100, 200)]
    foes = roark_party(names)
    raised = battle_start_party(tables, party, (62, None), (8,), foes)
    assert [s.friendship for s in raised] == [3, 102, 201]
    assert [s.friendship for s in party] == [0, 100, 200]  # the input is not changed
    e4 = battle_start_party(tables, party, (13, 66), (15,), foes)
    assert [s.friendship for s in e4] == [3, 102, 201]
    assert battle_start_party(tables, party, (13, None), (8,), foes) == party


def test_a_party_member_takes_the_bound_its_moves_read(tables, names):
    ret, frus = names.move("Return"), names.move("Frustration")
    hatched = _spec(names, friendship=52, item=0, egg_location=7)
    users = [replace(hatched, moves=(m, 0, 0, 0)) for m in (ret, frus)]
    got = battle_start_party(tables, users, (64,), (7, 119), roark_party(names))
    assert [s.friendship for s in got] == [55, 56]
    fixed = battle_start_party(tables, users, (64,), (7,), roark_party(names))
    assert [s.friendship for s in fixed] == [56, 56]


# --- the prep problem's states ------------------------------------------------------------


def _problem(tables, names, party_friendship, trainer):
    party = sample_party(names)
    cands = [
        candidate(names, replace(party[i], friendship=f, item=0), i)
        for i, f in enumerate(party_friendship)
    ]
    return replace(problem(tables, names, cands), trainer=trainer)


def test_setup_specs_carry_the_raise_against_a_leader_only(tables, names):
    prob = _problem(tables, names, (99, 100, 255), roark_trainer())
    setup = Setup(tuple(base_member(prob, i) for i in range(3)))
    assert [s.friendship for s in setup_specs(prob, setup, tables)] == [102, 102, 255]
    assert [s.friendship for s in setup_specs(prob, setup)] == [102, 102, 255]  # default tables
    regular = replace(roark_trainer(), class_id=13)
    prob = replace(prob, trainer=regular)
    assert [s.friendship for s in setup_specs(prob, setup, tables)] == [99, 100, 255]
    prob = replace(prob, enemy_2_class=79)  # the second enemy is a Gym Leader
    assert [s.friendship for s in setup_specs(prob, setup, tables)] == [102, 102, 255]


def test_setup_specs_give_each_reader_its_bound_on_an_open_label(tables, names):
    party = sample_party(names)
    ret, frus = names.move("Return"), names.move("Frustration")
    hatched = replace(party[0], friendship=52, item=0, egg_location=82)
    cands = [
        candidate(names, replace(hatched, moves=(m, 0, 0, 0)), i) for i, m in enumerate((ret, frus))
    ]
    trainer = replace(roark_trainer(), map_labels=(82, 119))
    prob = replace(problem(tables, names, cands), trainer=trainer)
    setup = Setup(tuple(base_member(prob, i) for i in range(2)))
    assert [s.friendship for s in setup_specs(prob, setup, tables)] == [55, 56]
    prob = replace(prob, trainer=replace(trainer, map_labels=(82,)))  # one header fixes it
    assert [s.friendship for s in setup_specs(prob, setup, tables)] == [56, 56]


def test_the_raise_follows_the_setups_held_item(tables, names):
    soothe = names.item("Soothe Bell")
    prob = _problem(tables, names, (50,), roark_trainer())
    plain = Setup((base_member(prob, 0),))
    held = Setup((replace(base_member(prob, 0), item=soothe),))
    assert setup_specs(prob, plain, tables)[0].friendship == 53
    assert setup_specs(prob, held, tables)[0].friendship == 54


def test_the_map_label_reaches_the_egg_location_test(tables, names):
    party = sample_party(names)
    cand = candidate(names, replace(party[0], friendship=50, item=0, egg_location=82), 0)
    trainer = replace(roark_trainer(), map_labels=(82,))
    prob = replace(problem(tables, names, [cand]), trainer=trainer)
    setup = Setup((base_member(prob, 0),))
    assert setup_specs(prob, setup, tables)[0].friendship == 54
    prob = replace(prob, trainer=replace(trainer, map_labels=(82, 119)))  # open, no reader: low
    assert setup_specs(prob, setup, tables)[0].friendship == 53
    prob = replace(prob, trainer=replace(trainer, map_labels=(454,)))  # a header id, not a label
    assert setup_specs(prob, setup, tables)[0].friendship == 53


def _damage_both(tables, state, move):
    """The 16 rolls of ``move`` from side 0, JAX and reference (asserted equal)."""
    got = damage_table(tables, state, 0, move, False).tolist()
    assert got == ref_damage_table(to_numpy(tables), to_ref(state), 0, move, False)
    return got


def test_return_in_the_first_turn_reads_the_raised_friendship(tables, names):
    """Return and Frustration read the party Pokemon's friendship (damage.py: ``friendship *
    10 / 25``), which the raise changes before the battlers are built (battle_main.c:1360-1372)."""
    ret, frustration = names.move("Return"), names.move("Frustration")
    foes = roark_party(names)
    spec = _spec(names, friendship=99, item=0, friend_bonus=0, egg_location=0)
    roark = roark_trainer()
    (raised,) = battle_start_party(tables, [spec], (roark.class_id,), roark.map_labels, foes)
    assert raised.friendship == 102
    start = make_state(tables, [raised], foes)
    assert int(start.friendship[0, 0]) == int(start.party_friendship[0, 0]) == 102
    by_hand = make_state(tables, [replace(spec, friendship=102)], foes)
    unraised = make_state(tables, [spec], foes)
    for move in (ret, frustration):
        assert _damage_both(tables, start, move) == _damage_both(tables, by_hand, move)
    # Return's power is friendship * 10 / 25 (39 at 99, 40 at 102), Frustration's the mirror
    assert _damage_both(tables, start, ret) >= _damage_both(tables, unraised, ret)
    assert _damage_both(tables, start, frustration) <= _damage_both(tables, unraised, frustration)


# --- the level-cap sit-out ----------------------------------------------------------------


def _sit(levels, cap, *, party, room, sat_out=(), overflow="sit_out_after_fight"):
    return cap_sit_out(
        levels,
        cap,
        cap_overflow=overflow,
        cap_sit_out_in_party=party,
        deposit_room=room,
        sat_out=sat_out,
    )


def test_a_member_over_the_next_cap_is_left_out():
    got = _sit([20, 17, None, 16, 12], 16, party="allowed", room=0)
    assert got == CapSitOut(over=(0, 1), deposited=(), barred=(0, 1), lost=False)  # 16 == cap: in


def test_forbidden_deposits_the_over_cap_members_at_an_open_gap():
    open_pc = _sit([20, 17, 12], 16, party="forbidden", room=None)
    assert open_pc == CapSitOut(over=(0, 1), deposited=(0, 1), barred=(), lost=False)
    swaps = _sit([20, 12], 16, party="forbidden", room=1)
    assert swaps == CapSitOut(over=(0,), deposited=(0,), barred=(), lost=False)


def test_forbidden_with_no_pc_or_swap_at_the_gap_is_lost():
    assert _sit([20, 12], 16, party="forbidden", room=0).lost
    assert _sit([20, 18, 12], 16, party="forbidden", room=1).lost  # one swap, two over
    assert not _sit([14, 12], 16, party="forbidden", room=0).lost  # nobody is over


def test_under_cap_overflow_fight_nothing_is_removed():
    got = _sit([40, 30], 16, party="forbidden", room=0, overflow="fight")
    assert got == CapSitOut((), (), (), False)
    assert _sit([40], 16, party="allowed", room=0, overflow="fight").barred == ()


def test_no_next_fight_sits_nobody_out():
    assert _sit([40], None, party="forbidden", room=0) == CapSitOut((), (), (), False)


def test_earlier_sit_outs_stay_barred_and_are_not_counted_again():
    got = _sit([30, 20, 12], 16, party="allowed", room=0, sat_out={0})
    assert got == CapSitOut(over=(1,), deposited=(), barred=(0, 1), lost=False)
    kept = _sit([30, 12], 16, party="forbidden", room=0, sat_out={0})
    assert kept == CapSitOut(over=(), deposited=(), barred=(0,), lost=False)


def test_unknown_setting_values_and_a_negative_room_are_refused():
    with pytest.raises(ValueError, match="cap_overflow"):
        _sit([20], 16, party="allowed", room=0, overflow="never")
    with pytest.raises(ValueError, match="cap_sit_out_in_party"):
        _sit([20], 16, party="sometimes", room=0)
    with pytest.raises(ValueError, match="deposit_room"):
        _sit([20], 16, party="forbidden", room=-1)


def _fight(label, split, order):
    return Trainer(
        id=1,
        name=label,
        trainer_class="x",
        class_id=0,
        class_gender=0,
        ai_flags=0,
        items=(),
        party=(),
        start_condition="none",
        format="single",
        label=label,
        split=split,
        split_order=order,
    )


def test_the_next_cap_comes_from_the_caps_file(tmp_path):
    caps = tmp_path / "caps.toml"
    caps.write_text('source = "test"\n[splits]\nRoark = 16\nGardenia = 28\n', encoding="utf-8")
    roark = _fight("Leader Roark", "Roark", (0, 9))
    gardenia = _fight("Leader Gardenia", "Gardenia", (1, 0))
    kaizo = "platinum_kaizo_nightmare_final"
    cap = next_fight_cap(roark, [roark, gardenia], kaizo, None, caps)
    assert cap == 28
    assert _sit([30, 28, 12], cap, party="allowed", room=0).barred == (0,)  # 28 is not over 28
    assert next_fight_cap(gardenia, [roark, gardenia], kaizo, None, caps) is None
