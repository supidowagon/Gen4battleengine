"""D2-14 (Counter family and Bide's release defenders) and D2-13 (GoToMoveScript's called-move
defender) at B = 4: ``targets.counter_defender`` / ``bide_defender`` / ``called_defender`` against
the reference. The four-actor step is D3's (ledger D2-17), so the functions are tested directly."""

from __future__ import annotations

import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, with_hp

NONE = targets.NO_DEFENDER


def _hit_by(s, b, hitter):
    return s._replace(last_hit_by=s.last_hit_by.at[b].set(hitter + 1))


def _follow_me(s, side, user):
    return s._replace(ts_follow_me=s.ts_follow_me.at[side].set(user + 1))


def counter(s, user, last):
    got = int(targets.counter_defender(s, user, last))
    assert got == ref.counter_defender(to_ref(s), user, last)
    return got


def bide(s, b, bit):
    got = int(targets.bide_defender(s, b, bit))
    want = ref.bide_defender(to_ref(s), b, bit)
    assert got == (NONE if want is None else want)
    return got


def test_counter_defender_follow_me(tables, names):
    s = four(tables, names)
    assert counter(s, 0, 3) == 3
    s = _follow_me(s, 1, 1)  # the opposing side's Follow Me user is battler 1
    assert counter(s, 0, 3) == 1
    assert counter(with_hp(s, 1, 0), 0, 3) == 3  # fainted user: the last attacker


def test_bide_defender(tables, names):
    s = four(tables, names)
    s = _hit_by(s, 0, 1)
    assert bide(s, 0, 0) == 1 and bide(s, 0, 1) == 1
    assert bide(_hit_by(s, 0, 2), 0, 0) == 2  # the partner hit last
    dead = with_hp(s, 1, 0)  # last hitter fainted: coin between the opponents
    assert bide(dead, 0, 0) == 3 and bide(dead, 0, 1) == 3  # 1 is down: the other with either coin
    both = with_hp(dead, 3, 0)
    assert bide(both, 0, 0) == NONE and bide(both, 0, 1) == NONE
    fresh = four(tables, names)  # nobody hit it yet: random opponent
    assert bide(fresh, 0, 0) == 3 and bide(fresh, 0, 1) == 1


def test_called_defender_ranges(tables, names):
    s = four(tables, names)
    order = np.asarray([0, 1, 2, 3])
    t, r = to_numpy(tables), to_ref(s)
    for move, bit, want in (
        ("Tackle", 0, 3),  # single target: random opponent, RIGHT slot first
        ("Tackle", 1, 1),
        ("Swords Dance", 0, 0),  # RANGE_USER: the user
        ("Earthquake", 0, 1),  # RANGE_ALL_ADJACENT: first living battler in the order
    ):
        mid = names.move(move)
        got, flag = targets.called_defender(tables, s, 0, mid, order, bit)
        want_ref = ref.called_defender(t, r, 0, mid, list(order), bit)
        assert (int(got), bool(flag)) == (want, False) == (want_ref[0], want_ref[1])
    gone = with_hp(s, 3, 0)
    got, _ = targets.called_defender(tables, gone, 0, names.move("Tackle"), order, 0)
    assert int(got) == 1  # one opponent down: the living one with either bit
    got, _ = targets.called_defender(tables, with_hp(gone, 1, 0), 0, names.move("Tackle"), order, 0)
    assert int(got) == NONE
