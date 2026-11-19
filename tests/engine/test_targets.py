"""D2-4 .. D2-8: ``targets.resolve`` (``BattleSystem_Defender``, battle_lib.c:1600-1730) on states
of both formats, with the reference's ``_resolve`` held equal on every case."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _resolve
from tests.engine.d2_states import four, mon, with_hp

NONE = targets.NO_DEFENDER


def _states(tables, names):
    b4 = four(tables, names)
    b2 = make_state(tables, [mon(names)], [mon(names)])
    return b2, b4


def follow_me(state, side, user):
    return state._replace(ts_follow_me=state.ts_follow_me.at[side].set(user + 1))


def go(tables, names, s, a, move, chosen=None, bit=0, randomize=False, order=None):
    """``(defender, target_side, counter, drew)`` of the engine, asserted equal to the reference's."""
    count = s.active.shape[0]
    chosen = a ^ 1 if chosen is None else chosen
    order = list(range(count)) if order is None else order
    mid = names.move(move)
    got = targets.resolve(tables, s, a, mid, randomize, chosen, jnp.asarray(order), bit)
    out = (int(got.defender), int(got.target_side), int(got.counter), bool(got.drew))
    ref = _resolve(to_numpy(tables), to_ref(s), a, mid, randomize, chosen, order, bit)
    want = (NONE if ref[0] is None else ref[0], ref[1], ref[2], ref[3])
    assert out == want, (move, a, out, want)
    return out


def test_opponents_table():
    for a, want in ((0, (3, 1)), (1, (2, 0)), (2, (3, 1)), (3, (2, 0))):
        got = targets.opponents(a)
        assert (int(got[0]), int(got[1])) == want


def test_single_target_branch_at_b4(tables, names):
    _, s = _states(tables, names)
    assert go(tables, names, s, 0, "Tackle", chosen=1) == (1, 1, 0, False)
    assert go(tables, names, s, 0, "Tackle", chosen=2) == (2, 2, 0, False)
    # chosen fainted, both foes alive: opponents[bit], one draw
    dead_partner = with_hp(s, 2, 0)
    assert go(tables, names, dead_partner, 0, "Tackle", chosen=2, bit=0) == (3, 3, 0, True)
    assert go(tables, names, dead_partner, 0, "Tackle", chosen=2, bit=1) == (1, 1, 0, True)
    # chosen fainted, one foe alive: that one for both bits
    for bit in (0, 1):
        assert go(tables, names, with_hp(s, 3, 0), 0, "Tackle", chosen=3, bit=bit)[0] == 1
    # all foes fainted: NONE, and target_side is the foe across (dead)
    gone = with_hp(with_hp(s, 1, 0), 3, 0)
    assert go(tables, names, gone, 0, "Tackle", chosen=1) == (NONE, 1, 0, True)


def test_follow_me_overrides_the_choice_at_b4(tables, names):
    _, s = _states(tables, names)
    fm = follow_me(s, 1, 3)
    for chosen in (1, 2, 0):  # a foe, the partner, itself
        assert go(tables, names, fm, 0, "Tackle", chosen=chosen) == (3, 3, 0, False)
    # a single-target move a battler aims at its own partner is drawn to the foe's user too
    assert go(tables, names, fm, 2, "Tackle", chosen=0)[0] == 3
    # the user fainted: rules 4-5 as if the field were clear
    assert go(tables, names, with_hp(fm, 3, 0), 0, "Tackle", chosen=1) == (1, 1, 0, False)
    # only the other side's moves read it
    assert go(tables, names, fm, 1, "Tackle", chosen=0)[0] == 0
    # a spread move and a hazard are not redirected
    assert go(tables, names, fm, 0, "Growl")[0] == 1
    assert go(tables, names, fm, 0, "Spikes", bit=1)[0] == 1


def test_b2_singles_values(tables, names):
    s, _ = _states(tables, names)
    foe_dead = with_hp(s, 1, 0)
    for bit in (0, 1):
        for move in ("Tackle", "Outrage", "Growl", "Earthquake"):
            spread = move in ("Growl", "Earthquake")
            assert go(tables, names, s, 0, move, bit=bit) == (1, 1, 2 if spread else 0, False)
            got = go(tables, names, foe_dead, 0, move, bit=bit)
            assert (got[0], got[1], got[3]) == (NONE, 1, False)
        assert go(tables, names, s, 0, "Tackle", randomize=True, bit=bit)[0] == 1
        assert go(tables, names, foe_dead, 0, "Spikes", bit=bit)[0] == 1  # any HP
        assert go(tables, names, foe_dead, 0, "Me First", randomize=True, bit=bit)[0] == 1
        for move in ("Swords Dance", "Reflect", "Sunny Day", "Counter", "Helping Hand"):
            for rnd in (False, True):
                assert go(tables, names, s, 0, move, randomize=rnd, bit=bit) == (0, 1, 0, False)
        for rnd in (False, True):
            assert go(tables, names, s, 0, "Acupressure", randomize=rnd, bit=bit)[0] == 0
            # a non-Ghost Curse names its user, kept for a randomized call (ledger D2-12)
            assert go(tables, names, s, 0, "Curse", randomize=rnd, bit=bit)[0] == 0
    assert go(tables, names, s, 1, "Tackle", chosen=0)[:2] == (0, 0)


def test_own_side_ranges_at_b4(tables, names):  # D2-5
    _, s = _states(tables, names)
    for move in ("Swords Dance", "Reflect", "Sunny Day", "Counter"):
        for rnd in (False, True):
            assert go(tables, names, s, 0, move, randomize=rnd, bit=1) == (0, 1, 0, False)
    for sd in (s, with_hp(s, 2, 0)):  # the partner, alive or fainted
        assert go(tables, names, sd, 0, "Helping Hand", chosen=2, randomize=True) == (
            2,
            2,
            0,
            False,
        )
    assert go(tables, names, s, 0, "Acupressure", chosen=0)[0] == 0
    assert go(tables, names, s, 0, "Acupressure", chosen=2)[0] == 2
    assert go(tables, names, with_hp(s, 2, 0), 0, "Acupressure", chosen=2)[0] == 0
    assert go(tables, names, s, 0, "Acupressure", randomize=True, bit=0) == (2, 2, 0, True)
    assert go(tables, names, with_hp(s, 2, 0), 0, "Acupressure", randomize=True, bit=0)[0] == 0
    assert go(tables, names, s, 0, "Acupressure", randomize=True, bit=1) == (0, 1, 0, True)


def test_opponent_side_and_me_first_at_b4(tables, names):  # D2-6
    _, s = _states(tables, names)
    for move, rnd in (("Spikes", False), ("Stealth Rock", True), ("Me First", True)):
        got = go(tables, names, s, 0, move, randomize=rnd, bit=0)
        assert (got[0], got[3]) == (3, True)
        got = go(tables, names, s, 0, move, randomize=rnd, bit=1)
        assert (got[0], got[3]) == (1, True)
        assert go(tables, names, with_hp(s, 3, 0), 0, move, randomize=rnd, bit=0)[0] == 1
        both = with_hp(with_hp(s, 1, 0), 3, 0)  # both down: a fainted battler, never NONE
        assert go(tables, names, both, 0, move, randomize=rnd, bit=0)[0] == 1
        assert go(tables, names, both, 0, move, randomize=rnd, bit=1)[0] == 3
        assert go(tables, names, follow_me(s, 1, 1), 0, move, randomize=rnd, bit=0)[0] == 3
    assert go(tables, names, s, 0, "Me First", chosen=1) == (1, 1, 0, False)  # branch 10


def test_random_opponent_branch_at_b4(tables, names):  # D2-7
    _, s = _states(tables, names)
    for move, rnd in (("Outrage", False), ("Tackle", True)):
        assert go(tables, names, s, 0, move, randomize=rnd, chosen=2, bit=0) == (3, 3, 0, True)
        assert go(tables, names, s, 0, move, randomize=rnd, chosen=2, bit=1) == (1, 1, 0, True)
        one = with_hp(s, 3, 0)
        for bit in (0, 1):
            assert go(tables, names, one, 0, move, randomize=rnd, bit=bit) == (1, 1, 0, False)
        assert go(tables, names, with_hp(one, 1, 0), 0, move, randomize=rnd) == (NONE, 1, 0, False)
        fm = follow_me(s, 1, 1)
        for bit in (0, 1):
            assert go(tables, names, fm, 0, move, randomize=rnd, bit=bit) == (1, 1, 0, False)
        dead_fm = with_hp(fm, 1, 0)  # the user fainted: only foe 3 is left
        assert go(tables, names, dead_fm, 0, move, randomize=rnd, bit=0) == (3, 3, 0, False)


def test_spread_first_defender(tables, names):  # D2-8
    b2, s = _states(tables, names)
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 1, 3, 0], [1, 0, 3, 2]):
        for rnd in (False, True):
            adj = next(b for b in order if b % 2 == 1)
            got = go(tables, names, s, 0, "Growl", order=order, randomize=rnd)
            assert got == (adj, adj, order.index(adj) + 1, False)
            near = next(b for b in order if b != 0)
            got = go(tables, names, s, 0, "Earthquake", order=order, randomize=rnd)
            assert got == (near, near, order.index(near) + 1, False)
    d = with_hp(s, 1, 0)
    assert go(tables, names, d, 0, "Growl", order=[1, 2, 3, 0])[:2] == (3, 3)
    assert go(tables, names, s, 0, "Growl", order=[2, 0, 3, 1])[0] == 3  # never the partner
    assert go(tables, names, s, 0, "Earthquake", order=[0, 2, 1, 3])[:3] == (2, 2, 2)
    none = with_hp(with_hp(s, 1, 0), 3, 0)
    assert go(tables, names, none, 0, "Growl") == (NONE, 1, 4, False)
    for order in ([0, 1], [1, 0]):
        assert go(tables, names, b2, 0, "Growl", order=order)[:2] == (1, 1)
        assert go(tables, names, with_hp(b2, 1, 0), 0, "Earthquake", order=order)[0] == NONE


def test_fight_randomizes(tables, names):
    s, _ = _states(tables, names)
    assert not bool(targets.fight_randomizes(s, 0, False))
    assert bool(targets.fight_randomizes(s, 0, True))
    enc = s._replace(encore_move=s.encore_move.at[0].set(names.move("Tackle")))
    assert bool(targets.fight_randomizes(enc, 0, False))


@pytest.mark.parametrize("fmt_b", [2, 4])
def test_target_side_is_never_the_attacker(tables, names, fmt_b):
    b2, b4 = _states(tables, names)
    s = b2 if fmt_b == 2 else b4
    for move in ("Tackle", "Swords Dance", "Helping Hand", "Curse"):
        for a in range(fmt_b):
            assert go(tables, names, s, a, move, chosen=a)[1] != a
