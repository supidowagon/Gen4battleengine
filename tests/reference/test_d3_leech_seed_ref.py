"""D3-11b: the reference's Leech Seed recipient (battle_controller_player.c:1378-1391): the
end-of-turn check of ``ref_step``'s pipeline equals the engine's on the brief's cases 2, 3, 4 and
6 (the recipient battler's current occupant receives; a 0-HP occupant: no drain)."""

from __future__ import annotations

import numpy as np

from nuzline.engine.format import DOUBLES
from nuzline.engine.outcome import empty_outcome
from tests.engine.d2_states import with_hp
from tests.engine.test_d3_action_loop import hp
from tests.engine.test_d3_end_of_turn import base, both, ref_end_turn
from tests.engine.test_d3_leech_seed import drain, seed, seeded_3_by_0


def ref_only(tables, state):
    return ref_end_turn(tables, state, empty_outcome(DOUBLES))


def test_case_2_recipient_is_the_named_battler(tables, names):
    s = seeded_3_by_0(tables, names)
    ref = ref_only(tables, s)
    assert hp(ref, 3) == hp(s, 3) - drain(s, 3) and hp(ref, 0) == hp(s, 0) + drain(s, 3)
    assert hp(ref, 2) == hp(s, 2)
    both(tables, s)


def test_case_3_the_partner_receives(tables, names):
    s = with_hp(with_hp(seed(base(tables, names), 2, 0), 0, 10), 2, 60)
    ref = ref_only(tables, s)
    assert hp(ref, 2) == 60 - drain(s, 2) and hp(ref, 0) == 10 + drain(s, 2)
    both(tables, s)


def test_case_4_a_fainted_seeder_drains_nothing(tables, names):
    s = with_hp(seeded_3_by_0(tables, names), 0, 0)
    ref = ref_only(tables, s)
    assert hp(ref, 3) == hp(s, 3)
    assert int(ref.leech_seed[3]) == 1 and int(ref.leech_seed_recipient[3]) == 1
    both(tables, s)


def test_case_6_the_newcomer_of_the_slot_receives(tables, names):
    s = seeded_3_by_0(tables, names)
    s = s._replace(active=s.active.at[0].set(2), hp=s.hp.at[0, 2].set(10))
    ref = ref_only(tables, s)
    assert hp(ref, 3) == hp(s, 3) - drain(s, 3)
    assert int(np.asarray(ref.hp)[0, 2]) == 10 + drain(s, 3)
    both(tables, s)
