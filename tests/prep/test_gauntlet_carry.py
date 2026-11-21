"""GT-4a (E3-B3-4a): the Carry type of ``prep.gauntlet`` (design v5.4 sections 5 and 7), its first
build from the save party, the member views in both directions against ``PokemonSpec`` and GT-13's
``evolve`` on a Carry member. ``carry_out`` and everything that reads a terminal battle state are
GT-4b's, with the tests that need them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nuzline.data.gauntlet import COUNTED, OPTIONS, Origin, Resolved, Settings, Swap
from nuzline.data.spec import PokemonSpec
from nuzline.engine.state import spec_exp, spec_stats
from nuzline.prep.gauntlet import (
    Carry,
    carry_batch,
    carry_from_party,
    carry_row,
    carry_stats,
    evolve_carry,
    member_spec,
    set_member,
    stack_carries,
)
from nuzline.prep.growth import ITEM_POKE_BALL
from nuzline.saves.pkm import ITEM_LUXURY_BALL

NINCADA, NINJASK, SHEDINJA = 290, 291, 292
TACKLE, GROWL, SCRATCH = 33, 45, 10
PIDGEY, RATTATA = 16, 19

# The trailing shape of every field of a single Carry (n_items, n_box, n_fights set per build).
PER_SLOT = {
    "present": (6,),
    "species": (6,),
    "form": (6,),
    "ability_slot": (6,),
    "personality": (6,),
    "nature": (6,),
    "gender": (6,),
    "beauty": (6,),
    "ball": (6,),
    "egg_location": (6,),
    "pokerus": (6,),
    "exp_boost": (6,),
    "ivs": (6, 6),
    "level": (6,),
    "exp": (6,),
    "evs": (6, 6),
    "stats": (6, 6),
    "hp": (6,),
    "status": (6,),
    "status_counter": (6,),
    "moves": (6, 4),
    "pp": (6, 4),
    "pp_ups": (6, 4),
    "item": (6,),
    "friendship": (6,),
    "dead": (6,),
    "sat_out": (6,),
    "deaths_used": (),
    "swap_left": (),
    "bag_uses_left": (),
    "partner_set": (),
}


def settings(*, shedinja="forbidden", swap=None, bag_uses=3):
    swap = Swap(1, 2) if swap is None else swap

    def resolved(value):
        return Resolved(value, Origin.DEFAULT, "test")

    return Settings(
        values={
            "shedinja": resolved(shedinja),
            "swap": resolved(swap),
            "bag_uses": resolved(bag_uses),
        },
        gaps={},
        locked=(),
        locked_source="",
    )


def mon(species=PIDGEY, level=20, **fields):
    base = PokemonSpec(
        species=species,
        level=level,
        nature=3,
        ivs=(10, 20, 30, 5, 15, 25),
        evs=(4, 8, 0, 252, 0, 12),
        moves=(TACKLE, GROWL, 0, 0),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )
    return replace(base, **fields)


def party3():
    return [
        mon(PIDGEY, 20, pp_ups=(2, 0, 0, 0), friendship=100, item=5),
        mon(RATTATA, 18, current_hp=7, status=2, gender=1, egg_location=3),
        mon(NINCADA, 20, nature=11, status=4, status_counter=2, gender=1, friendship=200),
    ]


def build(tables, party=None, **kw):
    party = party3() if party is None else party
    args = {
        "personality": [1000 + 7 * i for i in range(len(party))],
        "box_size": 4,
        "bench": (1, 3),
        "n_fights": 5,
        "beaten": (0, 2),
    }
    args.update(kw)
    return carry_from_party(tables, party, {ITEM_POKE_BALL: 3, 20: 2}, settings(), **args)


def same(a, b):
    assert Carry._fields == tuple(a._fields)
    for name in Carry._fields:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


# --- shapes ----------------------------------------------------------------------------------


def test_every_field_has_its_shape_without_batch_axis(tables):
    carry = build(tables)
    assert carry_batch(carry) is None
    shapes = {**PER_SLOT, "bag": (len(tables.item_names),), "bench": (4,), "beaten": (5,)}
    assert set(shapes) == set(Carry._fields)
    for name, shape in shapes.items():
        assert getattr(carry, name).shape == shape, name


def test_every_field_has_the_leading_batch_axis_with_it(tables):
    rows = [build(tables, deaths_used=i) for i in range(3)]
    batch = stack_carries(rows)
    assert carry_batch(batch) == 3
    for name in Carry._fields:
        assert getattr(batch, name).shape == (3, *getattr(rows[0], name).shape), name


def test_field_dtypes(tables):
    carry = build(tables)
    for name in ("present", "dead", "sat_out", "bench", "beaten", "partner_set"):
        assert getattr(carry, name).dtype == bool, name
    assert carry.personality.dtype == np.int64
    assert carry.stats.dtype == np.int32 and carry.hp.dtype == np.int32


def test_the_carry_holds_no_swap_or_growth_plan_yet():
    assert not {f for f in Carry._fields if "plan" in f}


def test_one_amount_left_per_counted_option():
    counted = {o.name for o in OPTIONS.values() if o.kind == COUNTED}
    assert counted == {"swap", "bag_uses"}
    assert {"swap_left", "bag_uses_left"} <= set(Carry._fields)


def test_a_mixed_batch_is_refused(tables):
    single = build(tables)
    mixed = single._replace(hp=np.stack([single.hp, single.hp]))
    with pytest.raises(ValueError, match="disagree"):
        carry_batch(mixed)
    with pytest.raises(ValueError, match="single"):
        stack_carries([stack_carries([single]), single])
    with pytest.raises(ValueError):
        stack_carries([])


# --- the first fight's carry -----------------------------------------------------------------


def test_the_first_carry_holds_the_party_in_order(tables):
    party = party3()
    carry = build(tables)
    assert carry.present.tolist() == [True, True, True, False, False, False]
    assert carry.species[:3].tolist() == [PIDGEY, RATTATA, NINCADA]
    assert carry.personality[:3].tolist() == [1000, 1007, 1014]
    assert carry.level[:3].tolist() == [20, 18, 20]
    assert carry.friendship[:3].tolist() == [100, 70, 200]
    assert carry.item[:3].tolist() == [5, 0, 0]
    assert carry.status[:3].tolist() == [0, 2, 4] and carry.status_counter[2] == 2
    assert carry.gender[:3].tolist() == [0, 1, 1] and carry.egg_location[1] == 3
    assert carry.moves[0].tolist() == [TACKLE, GROWL, 0, 0]
    assert carry.ivs[1].tolist() == list(party[1].ivs) and carry.evs[1].tolist() == [
        4,
        8,
        0,
        252,
        0,
        12,
    ]
    assert not carry.species[3:].any() and not carry.stats[3:].any() and not carry.hp[3:].any()
    assert not carry.moves[3:].any() and not carry.pp[3:].any()


def test_stats_are_computed_and_full_hp_is_the_maximum(tables):
    party = party3()
    carry = build(tables)
    for i, mon_ in enumerate(party):
        # the formula of Pokemon_CalcStats (src/pokemon.c:564-648), through calc_stats
        assert carry.stats[i].tolist() == spec_stats(tables, mon_).tolist()
    assert carry.hp[0] == carry.stats[0, 0] and carry.hp[2] == carry.stats[2, 0]
    assert carry.hp[1] == 7
    assert carry.exp[:3].tolist() == [spec_exp(tables, m) for m in party]


def test_a_stored_stats_value_is_not_read(tables):
    stored = replace(mon(PIDGEY, 20), stats=(1, 2, 3, 4, 5, 6))
    carry = build(tables, [stored], personality=[1])
    assert carry.stats[0].tolist() == spec_stats(tables, mon(PIDGEY, 20)).tolist()


def test_shedinja_max_hp_is_one(tables):
    carry = build(tables, [mon(SHEDINJA, 30)], personality=[2])
    assert carry.stats[0, 0] == 1 and carry.hp[0] == 1


def test_pp_defaults_to_the_maximum_at_the_pp_ups(tables):
    carry = build(tables)
    move_pp = np.asarray(tables.move_pp)
    assert carry.pp[0, 0] == move_pp[TACKLE] + move_pp[TACKLE] * 20 * 2 // 100  # src/pokemon.c:961
    assert carry.pp[0, 1] == move_pp[GROWL] and carry.pp[0, 2] == 0
    explicit = build(tables, [replace(mon(), pp=(3, 4, 0, 0))], personality=[0])
    assert explicit.pp[0].tolist() == [3, 4, 0, 0]


def test_dead_members_and_the_chain_fields(tables):
    party = party3()
    party[0] = replace(party[0], current_hp=0)
    carry = build(tables, party, dead=(2,), sat_out=(1,), deaths_used=1, partner_set=True)
    assert carry.dead.tolist() == [True, False, True, False, False, False]
    assert carry.sat_out.tolist() == [False, True, False, False, False, False]
    assert int(carry.deaths_used) == 1 and bool(carry.partner_set)
    assert int(carry.swap_left) == 2 and int(carry.bag_uses_left) == 3  # the settings' budget
    assert carry.bag[ITEM_POKE_BALL] == 3 and carry.bag[20] == 2 and carry.bag.sum() == 5
    assert carry.bench.tolist() == [False, True, False, True]
    assert carry.beaten.tolist() == [True, False, True, False, False]


def test_the_ball_follows_friend_bonus_unless_given(tables):
    party = [mon(), mon(friend_bonus=1)]
    carry = build(tables, party, personality=[1, 2])
    assert carry.ball[:2].tolist() == [ITEM_POKE_BALL, ITEM_LUXURY_BALL]
    given = build(tables, party, personality=[1, 2], ball=[7, 8], beauty=[9, 10])
    assert given.ball[:2].tolist() == [7, 8] and given.beauty[:2].tolist() == [9, 10]


@pytest.mark.parametrize(
    "kw",
    [
        {"personality": [1]},
        {"bag": {99999: 1}},
        {"bag": {4: -1}},
        {"dead": (3,)},
        {"sat_out": (-1,)},
        {"bench": (9,)},
        {"beaten": (5,)},
        {"beauty": [1]},
    ],
)
def test_bad_inputs_are_refused(tables, kw):
    args = {"personality": [1, 2, 3], "box_size": 4, "n_fights": 5}
    bag = kw.pop("bag", {})
    args.update(kw)
    with pytest.raises(ValueError):
        carry_from_party(tables, party3(), bag, settings(), **args)


def test_a_party_of_zero_or_seven_is_refused(tables):
    for party in ([], [mon()] * 7):
        with pytest.raises(ValueError, match="party holds"):
            carry_from_party(tables, party, {}, settings(), personality=[0] * len(party))


# --- the batch axis --------------------------------------------------------------------------


def test_r_rows_equal_r_single_builds(tables):
    rows = []
    for i in range(3):
        party = party3()[: 3 - i % 2]
        party[0] = replace(party[0], level=15 + 5 * i, evs=(i, 0, 0, 0, 0, 255), nature=2 * i)
        rows.append(build(tables, party, deaths_used=i, dead=(i % 2,)))
    batch = stack_carries(rows)
    for i, row in enumerate(rows):
        same(carry_row(batch, i), row)
    # the vectorised stats of the batch are each row's
    stats = carry_stats(tables, batch)
    assert stats.shape == (3, 6, 6)
    for i, row in enumerate(rows):
        np.testing.assert_array_equal(stats[i], carry_stats(tables, row))
        np.testing.assert_array_equal(stats[i], row.stats)


def test_carry_row_takes_a_batch_only(tables):
    with pytest.raises(ValueError, match="batched"):
        carry_row(build(tables), 0)
    with pytest.raises(ValueError, match="single"):
        member_spec(stack_carries([build(tables)]), 0)


# --- the member views ------------------------------------------------------------------------


def test_the_view_is_the_prep_record(tables):
    party = party3()
    carry = build(tables)
    for i, original in enumerate(party):
        view = member_spec(carry, i)
        assert view.current_hp == (original.current_hp or carry.stats[i, 0])
        # every field the record states survives; the ones it leaves open are filled
        expected = replace(
            original,
            current_hp=view.current_hp,
            pp=view.pp,
            exp=spec_exp(tables, original),
            stats=None,
        )
        assert view == expected


def test_a_round_trip_between_the_record_and_the_carry(tables):
    carry = build(tables)
    # carry -> record -> carry
    again = carry
    for i in range(3):
        again = set_member(tables, again, i, member_spec(carry, i))
    same(again, carry)
    # record -> carry -> record -> carry, by the first build
    specs = [member_spec(carry, i) for i in range(3)]
    rebuilt = build(tables, specs, dead=(), sat_out=(1,), deaths_used=0)
    for i in range(3):
        assert member_spec(rebuilt, i) == specs[i]
    np.testing.assert_array_equal(rebuilt.stats, carry.stats)


def test_a_view_of_an_empty_slot_is_refused(tables):
    with pytest.raises(ValueError, match="empty"):
        member_spec(build(tables), 3)


def test_set_member_recomputes_the_stats_and_keeps_the_constants(tables):
    carry = build(tables, beauty=[11, 12, 13], ball=[7, 8, 9])
    leveled = replace(member_spec(carry, 0), level=30, exp=None, current_hp=None)
    out = set_member(tables, carry, 0, leveled)
    assert out.level[0] == 30 and out.stats[0].tolist() == spec_stats(tables, leveled).tolist()
    assert out.hp[0] == out.stats[0, 0]  # None: full HP
    assert (out.personality[0], out.beauty[0], out.ball[0]) == (1000, 11, 7)
    assert out.exp[0] == spec_exp(tables, leveled)
    # the other slots and the chain fields are not touched
    for name in Carry._fields:
        if name not in ("level", "stats", "hp", "exp"):
            np.testing.assert_array_equal(getattr(out, name), getattr(carry, name), err_msg=name)
    assert out is not carry and carry.level[0] == 20  # the input is not changed


def test_set_member_into_an_empty_slot(tables):
    carry = build(tables)
    with pytest.raises(ValueError, match="personality"):
        set_member(tables, carry, 4, mon())
    out = set_member(tables, carry, 4, mon(friend_bonus=1), personality=77)
    assert out.present.tolist() == [True, True, True, False, True, False]
    assert (out.personality[4], out.beauty[4], out.ball[4]) == (77, 0, ITEM_LUXURY_BALL)
    assert not out.dead[4] and not out.sat_out[4]
    with pytest.raises(ValueError, match="outside"):
        set_member(tables, carry, 6, mon(), personality=1)


# --- evolve on a Carry member ----------------------------------------------------------------


def method_id(learnsets, name):
    return list(learnsets.evolution_methods).index(name)


def nincada_carry(tables, bag=None):
    party = [mon(RATTATA, 18), mon(NINCADA, 20, status=4, status_counter=2, item=5), mon()]
    return carry_from_party(
        tables,
        party,
        {ITEM_POKE_BALL: 3, 20: 2} if bag is None else bag,
        settings(),
        personality=[10, 11, 12],
        beauty=[0, 33, 0],
    )


def test_evolve_on_a_carry_member(tables, learnsets):
    carry = nincada_carry(tables)
    out = evolve_carry(
        tables, learnsets, carry, 1, NINJASK, method_id(learnsets, "level_ninjask"), settings()
    )
    assert out.species[1] == NINJASK and out.present.tolist() == carry.present.tolist()
    assert out.item[1] == 5 and out.status[1] == 4  # Ninjask keeps its item and status
    # the evolved member's fields are what GT-13's evolve makes of the record
    assert member_spec(out, 1).species == NINJASK
    assert out.stats[1].tolist() == carry_stats(tables, out)[1].tolist()
    assert out.stats[1].tolist() != carry.stats[1].tolist()
    assert out.personality[1] == 11 and out.beauty[1] == 33  # constants kept
    assert out.hp[1] == carry.hp[1] + out.stats[1, 0] - carry.stats[1, 0]  # src/pokemon.c:640-651
    for slot in (0, 2):  # the other members are untouched
        assert member_spec(out, slot) == member_spec(carry, slot)
    np.testing.assert_array_equal(out.bag, carry.bag)  # shedinja forbidden: no ball used
    assert carry.species[1] == NINCADA  # the input is not changed


def test_evolve_a_dead_member_changes_nothing(tables, learnsets):
    carry = nincada_carry(tables)
    carry = set_member(tables, carry, 1, replace(member_spec(carry, 1), current_hp=0))
    out = evolve_carry(
        tables, learnsets, carry, 1, NINJASK, method_id(learnsets, "level_ninjask"), settings()
    )
    same(out, carry)


def test_the_shedinja_goes_into_the_first_free_slot(tables, learnsets):
    carry = nincada_carry(tables)
    allowed = settings(shedinja="allowed")
    out = evolve_carry(
        tables, learnsets, carry, 1, NINJASK, method_id(learnsets, "level_ninjask"), allowed
    )
    assert out.present.tolist() == [True, True, True, True, False, False]
    shed = member_spec(out, 3)
    assert (
        shed.species == SHEDINJA and shed.item == 0 and shed.status == 0
    )  # src/evolution.c:733-757
    assert out.stats[3, 0] == 1 and out.hp[3] == 1 and shed.gender == 2
    assert out.ball[3] == ITEM_POKE_BALL  # :736-737
    assert (out.personality[3], out.beauty[3]) == (11, 33)  # Pokemon_Copy, :731
    assert out.moves[3].tolist() == out.moves[1].tolist() and out.friendship[3] == out.friendship[1]
    assert not out.dead[3] and not out.sat_out[3]
    assert out.bag[ITEM_POKE_BALL] == 2 and out.bag[20] == 2  # one Poke Ball used up, :779
    assert carry.bag[ITEM_POKE_BALL] == 3


def test_the_shedinja_takes_the_first_free_slot_not_the_end(tables, learnsets):
    carry = set_member(tables, nincada_carry(tables), 4, mon(), personality=5)  # slot 3 is free
    out = evolve_carry(
        tables,
        learnsets,
        carry,
        1,
        NINJASK,
        method_id(learnsets, "level_shedinja"),
        settings(shedinja="allowed"),
    )
    assert out.present.tolist() == [True, True, True, True, True, False]
    assert out.species[3] == SHEDINJA and out.species[4] == PIDGEY


@pytest.mark.parametrize(
    ("bag", "party_size", "shedinja"),
    [
        ({}, 3, "allowed"),
        ({ITEM_POKE_BALL: 2}, 3, "forbidden"),
        ({ITEM_POKE_BALL: 2}, 6, "allowed"),
    ],
)
def test_no_shedinja_without_ball_setting_or_room(tables, learnsets, bag, party_size, shedinja):
    party = [mon(NINCADA, 20)] + [mon()] * (party_size - 1)
    carry = carry_from_party(tables, party, bag, settings(), personality=list(range(party_size)))
    out = evolve_carry(
        tables,
        learnsets,
        carry,
        0,
        NINJASK,
        method_id(learnsets, "level_ninjask"),
        settings(shedinja=shedinja),
    )
    assert out.species[0] == NINJASK and out.present.sum() == party_size
    np.testing.assert_array_equal(out.bag, carry.bag)
