"""D2-10: no target (``NO_TARGET``, include/battle/common.h:56-61; ``BattleControllerPlayer_
HasNoTarget``, bcp.c:2306-2337): ``targets.multi_turn`` / ``no_target`` / ``charge_keeps_user``,
engine against reference. The four-battler step cases are owed to D3 (ledger D2-17): at B = 4 the
resolver and these functions are composed here."""

from __future__ import annotations

import itertools

import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.ids import ItemId, Weather
from nuzline.engine.state import make_state
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, mon, with_hp
from tests.engine.test_targets import go

NONE = targets.NO_DEFENDER
USER = 0


def _b2(tables, names):
    return make_state(tables, [mon(names)], [mon(names)])


def _locked(s, names, move="Fly"):
    return s._replace(locked_move=s.locked_move.at[USER].set(names.move(move)))


def _holding(s, tables, effect_item):
    item = int((tables.item_effect == effect_item).nonzero()[0][0])
    p, slot = int(targets.party(targets.format_of(s), USER)), int(s.active[USER])
    return s._replace(item=s.item.at[p, slot].set(item))


def _both(tables, s, name, *args):
    """``targets.<name>`` and ``reference.<name>`` on the same arguments, asserted equal."""
    got = bool(getattr(targets, name)(tables, s, *args))
    args_ref = tuple(None if a == NONE else a for a in args)
    want = getattr(ref, name)(to_numpy(tables), to_ref(s), *args_ref)
    assert got == bool(want), (name, args)
    return got


def _no_target(tables, s, move, defender, last=False):
    return _both(tables, s, "no_target", USER, move, defender, last)


def _keeps(tables, s, move, defender):
    return _both(tables, s, "charge_keeps_user", USER, move, defender)


def test_multi_turn(tables, names):
    for move, want in (("Fly", True), ("Bide", True), ("SolarBeam", True), ("Tackle", False)):
        mid = names.move(move)
        assert bool(targets.multi_turn(tables, mid)) == want
        assert ref.multi_turn(to_numpy(tables), mid) == want


@pytest.mark.parametrize("fmt", [2, 4])
def test_no_target_rule(tables, names, fmt):
    s = _b2(tables, names) if fmt == 2 else four(tables, names)
    fly, tackle = names.move("Fly"), names.move("Tackle")
    assert _no_target(tables, s, tackle, NONE)  # single-turn move, no defender
    assert not _no_target(tables, s, fly, NONE)  # a charge turn, unlocked
    assert _no_target(tables, _locked(s, names), fly, NONE)  # locked: the strike
    assert _no_target(tables, s, fly, NONE, last=True)  # SYSCTL_LAST_OF_MULTI_TURN
    assert not _no_target(tables, s, tackle, 1)  # a defender
    assert not _no_target(tables, s, fly, 1, last=True)


@pytest.mark.parametrize("fmt", [2, 4])
def test_charge_keeps_user(tables, names, fmt):
    s = _b2(tables, names) if fmt == 2 else four(tables, names)
    fly, solar, tackle = names.move("Fly"), names.move("SolarBeam"), names.move("Tackle")
    assert _keeps(tables, s, fly, NONE)  # first-turn Fly with no foe charges
    assert not _keeps(tables, s, fly, 1)  # a defender
    assert not _keeps(tables, s, tackle, NONE)  # not multi-turn
    assert not _keeps(tables, _locked(s, names), fly, NONE)
    assert not _keeps(tables, _holding(s, tables, ItemId.POWER_HERB), fly, NONE)
    sun = s._replace(weather=Weather.SUN)
    assert not _keeps(tables, sun, solar, NONE)  # SolarBeam skips its charge in sun
    assert _keeps(tables, s, solar, NONE)
    assert _keeps(tables, sun, fly, NONE)  # sun only skips SolarBeam's charge


def test_b4_both_foes_fainted(tables, names):
    """The resolver and the no-target rule composed on a four-battler state (D2-10 section 4.5)."""
    s = with_hp(with_hp(four(tables, names), 1, 0), 3, 0)

    def defender(state, move):
        return go(tables, names, state, USER, move)[0]

    assert defender(s, "Tackle") == NONE
    assert _no_target(tables, s, names.move("Tackle"), NONE)
    # RANGE_OPPONENT_SIDE (Spikes): a battler of any HP, the move still lands
    spikes = defender(s, "Spikes")
    assert spikes in (1, 3) and not _no_target(tables, s, names.move("Spikes"), spikes)
    # Helping Hand at a fainted partner: the partner of any HP, no no-target
    dead_partner = with_hp(s, 2, 0)
    hh = defender(dead_partner, "Helping Hand")
    assert hh == 2 and not _no_target(tables, dead_partner, names.move("Helping Hand"), hh)
    # Fly: the first turn charges with the user as its defender; with a Power Herb it ends in no
    # target (last of the multi-turn move, a defender left BATTLER_NONE)
    fly = names.move("Fly")
    assert defender(s, "Fly") == NONE
    assert _keeps(tables, s, fly, NONE) and not _no_target(tables, s, fly, NONE)
    herb = _holding(s, tables, ItemId.POWER_HERB)
    assert not _keeps(tables, herb, fly, NONE)
    assert _no_target(tables, herb, fly, NONE, last=True)


@pytest.mark.parametrize("last", [False, True])
def test_engine_matches_reference_over_states(tables, names, last):
    moves = [names.move(m) for m in ("Tackle", "Fly", "Bide", "SolarBeam", "Dig")]
    base = four(tables, names)
    for s, move, locked, defender in itertools.product(
        (base, base._replace(weather=Weather.SUN)),
        moves,
        (False, True),
        (NONE, 1, 3),
    ):
        state = _locked(s, names) if locked else s
        _no_target(tables, state, move, defender, last)
        _keeps(tables, state, move, defender)
