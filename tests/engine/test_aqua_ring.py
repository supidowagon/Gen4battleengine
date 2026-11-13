"""Aqua Ring (K3b1 task 12, ``EffectId.AQUA_RING``).

* ``effect_script_0251.s``: sets ``MOVE_EFFECT_AQUA_RING`` on the user; fails when it is
  already set.
* ``BattleControllerPlayer_CheckMonConditions`` (bcp.c:1331-1351), ``MON_COND_CHECK_STATE_AQUA_RING``:
  second in the mon-condition order (after Ingrain, before the turn-end ability): a ringed
  battler alive and below full HP heals ``BattleSystem_Divide(maxHP, 16)`` through
  ``subscript_aqua_ring_heal.s``, which multiplies by ``(100 + param) / 100`` with a Big
  Root. Only Heal Block (``healBlockTurns``, unsupported) gates the heal.
* ``MOVE_EFFECT_AQUA_RING`` is part of ``MOVE_EFFECT_BATON_PASSED`` (moves.h:44), but Baton
  Pass is unsupported, so the carry-over is unreachable.
"""

import pytest

from nuzline.engine.ids import EffectId, ItemId, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import END_OF_TURN_ORDER
from tests.engine.helpers import A, assert_matches_reference, roark_state

SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot


def _ring(tables, names, **kw):
    return roark_state(tables, names, player_moves=["Aqua Ring", "Leer"], **kw)


def _ringed(s, side=0):
    return s._replace(aqua_ring=s.aqua_ring.at[side].set(1))


def test_aqua_ring_maps_in_both_games(tables, names, kaizo_tables, kaizo_names):
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Aqua Ring")]) == EffectId.AQUA_RING


@pytest.mark.slow
def test_aqua_ring_fails_if_set(tables, names):
    s = _ring(tables, names)
    once = assert_matches_reference(tables, s, A("Aqua Ring", "Leer", s, names), empty_outcome())
    assert int(once.aqua_ring[0]) == 1 and int(once.aqua_ring[1]) == 0
    # A second use fails: effect_script_0251.s jumps to MOVE_STATUS_FAILED. Only PP drops.
    twice = assert_matches_reference(
        tables, once, A("Aqua Ring", "Leer", once, names), empty_outcome()
    )
    for name in ("aqua_ring", "status", "confusion", "item", "mud_sport", "water_sport"):
        assert (getattr(twice, name) == getattr(once, name)).all(), name
    assert int(twice.pp[0, 0, 0]) == int(once.pp[0, 0, 0]) - 1


def test_aqua_ring_heal_and_order(tables, names):
    s = _ringed(_ring(tables, names))
    mx = int(s.max_hp[0, 0])
    s = s.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == 1 + max(mx // 16, 1)
    # Never above max HP.
    near = s.replace_hp(0, 0, mx - 1)
    out = assert_matches_reference(tables, near, A("Leer", "Leer", near, names), empty_outcome())
    assert int(out.hp[0, 0]) == mx
    # No heal at full HP,.
    full = _ringed(_ring(tables, names))
    out = assert_matches_reference(tables, full, A("Leer", "Leer", full, names), empty_outcome())
    assert int(out.hp[0, 0]) == mx
    # Second in the mon-condition order, before the turn-end ability (bcp.c:1331-1355).
    assert [n for n, _ in END_OF_TURN_ORDER][:2] == ["aqua_ring", "ability"]


def test_aqua_ring_heals_a_poisoned_holder_before_the_poison(tables, names):
    # Ring first (bcp.c:1332), poison later (:1393): 1 HP survives the ring, then the
    # poison takes an eighth.
    s = _ringed(_ring(tables, names)).with_status(0, 0, Status.POISON)
    mx = int(s.max_hp[0, 0])
    s = s.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == max(1 + max(mx // 16, 1) - max(mx // 8, 1), 0)


def test_aqua_ring_big_root(tables, names):
    s = _ringed(_ring(tables, names))
    root = s._replace(item=s.item.at[0, 0].set(names.item("Big Root")))
    assert int(tables.item_effect[names.item("Big Root")]) == ItemId.BIG_ROOT
    mx = int(s.max_hp[0, 0])
    base_heal = max(mx // 16, 1)
    low = root.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, low, A("Leer", "Leer", low, names), empty_outcome())
    assert int(out.hp[0, 0]) == 1 + base_heal * 130 // 100


def test_aqua_ring_cleared_on_switch(tables, names):
    s = _ringed(_ring(tables, names))
    acts = A("Leer", "Leer", s, names)
    acts = acts.at[0].set(SWITCH_TO_1)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.aqua_ring[0]) == 0


def test_aqua_ring_matches_reference(tables, names):
    # The foe's ring heals it too; an Aqua Ring used the same turn heals at its end.
    s = _ringed(_ring(tables, names), side=1)
    s = s.replace_hp(1, 0, 5)
    for acts in (A("Aqua Ring", "Leer", s, names), A("Leer", "Leer", s, names)):
        out = assert_matches_reference(tables, s, acts, empty_outcome())
        assert int(out.hp[1, 0]) == 5 + max(int(s.max_hp[1, 0]) // 16, 1)
