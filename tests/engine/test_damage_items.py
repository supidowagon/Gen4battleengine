"""E2 batch 1, the cases random battles reach too rarely to count on (the differential test
draws every item; ``tests/strategies.py``): Expert Belt on a super-effective hit, and Pursuit's
pre-switch hit by a Choice or Life Orb holder.

* Expert Belt (``HOLD_EFFECT_POWER_UP_SE``): ``damage * (100 + 20) / 100`` on a
  ``MOVE_STATUS_SUPER_EFFECTIVE`` hit, after Solid Rock (battle_lib.c:2663-2665).
* ``BtlCmd_TryPursuit`` locks a Choice holder into Pursuit (battle_script.c:6943-6950);
  ``subscript_pursuit.s:48-49`` runs ``TriggerHeldItemOnPivotMove``, Life Orb's recoil
  ``Divide(maxHP, 10)`` (battle_lib.c:7357-7366).
"""

import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.damage import damage_table
from nuzline.engine.outcome import empty_outcome
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state


def _item(s, side, item, names):
    return s._replace(item=s.item.at[side, 0].set(names.item(item)))


def _tables(tables, s, move):
    jax_rolls = [int(x) for x in damage_table(tables, s, 0, move, False)]
    assert jax_rolls == ref_damage_table(to_numpy(tables), to_ref(s), 0, move, False)
    return np.asarray(jax_rolls)


def test_expert_belt_boosts_only_super_effective_hits(tables, names):
    s = roark_state(tables, names, player_species="Grotle", player_moves=["Razor Leaf", "Tackle"])
    for move, boosted in (("Razor Leaf", True), ("Tackle", False)):  # vs Geodude: x4, x0.5
        m = names.move(move)
        plain = _tables(tables, _item(s, 0, "", names), m)
        belt = _tables(tables, _item(s, 0, "Expert Belt", names), m)
        assert (belt == (plain * 120 // 100 if boosted else plain)).all(), move


@pytest.mark.parametrize("item", ["Choice Band", "Life Orb"])
def test_pursuit_on_a_switch_locks_and_recoils(tables, names, item):
    s = _item(roark_state(tables, names, opp_moves=["Pursuit"]), 1, item, names)
    pursuit = names.move("Pursuit")
    out = assert_matches_reference(tables, s, A(5, "Pursuit", s, names), empty_outcome())
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # Pursuit hit the outgoing Pokemon
    if item == "Choice Band":
        assert int(out.choice_lock[1]) == pursuit
    else:
        assert int(out.choice_lock[1]) == 0
        loss = max(int(s.max_hp[1, 0]) // 10, 1)
        assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - loss
