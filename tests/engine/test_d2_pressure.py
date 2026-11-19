"""D2-15: Pressure's PP cost reads the resolved defender and the counting ranges
(``BattleControllerPlayer_DecrementPP``, bcp.c:2235-2289) at B = 4: ``effects.pressure_pp_cost``
engine against ``reference._pp_cost``. The four-battler step is D3's (ledger D2-17), so the
resolver and the cost are composed here."""

from __future__ import annotations

import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.effects import pressure_pp_cost
from nuzline.engine.ids import AbilityId
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, with_ability, with_hp

NONE = targets.NO_DEFENDER
USER = 0
PRESSURE = AbilityId.PRESSURE


def cost(tables, names, s, move, defender):
    mid = names.move(move)
    got = int(pressure_pp_cost(tables, s, USER, defender, mid))
    want = ref._pp_cost(
        to_numpy(tables), to_ref(s), USER, None if defender == NONE else defender, mid
    )
    assert got == want, (move, defender, got, want)
    return got


def field(tables, names, holders):
    s = four(tables, names)
    for b in holders:
        s = with_ability(s, b, PRESSURE)
    return s


def test_spread_ranges(tables, names):
    # Pressure on the user's partner (2) and one foe (1): Earthquake counts both, the opponent-only
    # spread move counts the foe alone (the partner is not counted).
    s = field(tables, names, [2, 1])
    assert cost(tables, names, s, "Earthquake", 1) == 3
    assert cost(tables, names, s, "Razor Leaf", 1) == 2


def test_single_target(tables, names):
    s = field(tables, names, [2, 1])
    assert cost(tables, names, s, "Tackle", 2) == 2  # a single-target move at a Pressure ally
    s = field(tables, names, [1])
    assert cost(tables, names, s, "Tackle", 3) == 1  # a non-Pressure foe beside a Pressure foe
    assert cost(tables, names, s, "Tackle", 1) == 2
    assert cost(tables, names, s, "Tackle", USER) == 1  # the user's own Pressure never counts
    assert cost(tables, names, s, "Tackle", NONE) == 1  # no defender: bcp.c:2238


def test_fainted_holder(tables, names):
    s = with_hp(field(tables, names, [1, 3]), 3, 0)
    assert cost(tables, names, s, "Earthquake", 1) == 2  # the fainted holder is not counted
    assert cost(tables, names, s, "Tackle", 3) == 2  # the single-target case reads any HP


def test_resolved_defender_by_coin(tables, names):
    # One foe down, the coin picks the Pressure foe: the resolver's defender is the living one.
    s = with_hp(field(tables, names, [1]), 3, 0)
    for bit in (0, 1):
        res = targets.resolve(tables, s, USER, names.move("Tackle"), True, 3, np.arange(4), bit)
        assert int(res.defender) == 1
        assert cost(tables, names, s, "Tackle", int(res.defender)) == 2


def test_imprison_and_own_side(tables, names):
    s = field(tables, names, [1, 3, 2])
    assert cost(tables, names, s, "Imprison", 1) == 3  # both foes, not the partner
    assert cost(tables, names, s, "Swords Dance", USER) == 1
