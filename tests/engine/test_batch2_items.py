"""E2 batch 2, the cases random battles reach too rarely to count on (the differential test
draws every batch-2 item with what makes it act; ``tests/strategies._draw_batch2_item``):

* King's Rock (``HOLD_EFFECT_SOMETIMES_FLINCH``): ``BattleControllerPlayer_CheckExtraFlinch``
  flinches a target that has not moved (battle_controller_player.c:4630-4653).
* Zoom Lens (``HOLD_EFFECT_ACCURACY_UP_SLOWER``): ``hitRate * (100 + 20) / 100`` against a
  target that has moved (bcp.c:2963-2965), one more accuracy class.
* Light Clay (``HOLD_EFFECT_EXTEND_SCREENS``): Reflect for 5 + 3 turns (battle_script.c:4272).
* Sticky Barb (``HOLD_EFFECT_DMG_USER_CONTACT_XFR``): moves to an item-less contact attacker
  (battle_lib.c:5386-5398; Pursuit's hit, subscript_pursuit.s:44), which then takes
  ``Divide(maxHP, 8)`` at the turn's end (:5330-5336).
* Shell Bell (``HOLD_EFFECT_HP_RESTORE_ON_DMG``) after Pursuit's hit (battle_lib.c:7345-7355).
* Custap Berry: an activated one cannot be Plucked (battle_script.c:8125-8127) and is eaten
  when its holder's move starts (subscript_check_quick_claw.s:26-28).
* A frozen defender's Aspear / Lum Berry cures the freeze at ``AFTER_MOVE_EFFECT_DEFENDER_ITEM``,
  before a Fire hit's thaw (``_THAW_DEFENDER``, bcp.c:3691, :3708), so the berry is used.
"""

import jax
import numpy as np
import pytest

from nuzline.engine.enumerate import hit_class_law, outcome_laws
from nuzline.engine.ids import Status
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import A, assert_matches_reference, roark_state


def _item(s, side, item, names, slot=0):
    return s._replace(item=s.item.at[side, slot].set(names.item(item) if item else 0))


def _hp(s, side, hp, slot=0):
    return s._replace(hp=s.hp.at[side, slot].set(hp))


def _outcome(**cells):
    o = empty_outcome()
    for name, (index, value) in cells.items():
        o = o._replace(**{name: getattr(o, name).at[index].set(value)})
    return o


def test_kings_rock_flinches_a_target_that_has_not_moved(tables, names):
    s = _item(roark_state(tables, names), 0, "King’s Rock", names)  # Monferno is faster
    acts = A("Scratch", "Tackle", s, names)
    tackle = names.move_slot(s, 1, "Tackle")
    calm = assert_matches_reference(tables, s, acts, empty_outcome())
    flinch = assert_matches_reference(tables, s, acts, _outcome(flinch_item=((0, 0), 1)))
    assert int(calm.pp[1, 0, tackle]) == int(s.pp[1, 0, tackle]) - 1
    assert int(flinch.pp[1, 0, tackle]) == int(s.pp[1, 0, tackle])  # flinched: no Tackle
    assert int(flinch.hp[0, 0]) == int(s.hp[0, 0])
    # Moving second, the roll changes nothing (the target has moved).
    slow = _item(roark_state(tables, names, opp_faster=True), 0, "King’s Rock", names)
    acts = A("Scratch", "Tackle", slow, names)
    a = assert_matches_reference(tables, slow, acts, empty_outcome())
    b = assert_matches_reference(tables, slow, acts, _outcome(flinch_item=((0, 0), 1)))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_zoom_lens_raises_accuracy_against_a_target_that_moved(tables, names):
    for opp_faster in (True, False):
        s = roark_state(tables, names, player_moves=["Tackle"], opp_faster=opp_faster)
        s = _item(s, 0, "Zoom Lens", names)
        acts = A("Tackle", "Leer", s, names)
        out = assert_matches_reference(tables, s, acts, _outcome(hit=((0, 0), 96)))
        # Tackle's 95 becomes 114 only when the player moves second.
        assert (int(out.hp[1, 0]) < int(s.hp[1, 0])) == opp_faster
        laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
        law = hit_class_law(laws.hit_thresholds[0][laws.hit_possible[0]])
        assert dict(zip(law[0], law[1], strict=True)) == (
            {1: 95, 96: 5} if opp_faster else {0: 5, 1: 95}
        )


def test_light_clay_extends_reflect(tables, names):
    s = _item(roark_state(tables, names, player_moves=["Reflect"]), 0, "Light Clay", names)
    out = assert_matches_reference(tables, s, A("Reflect", "Leer", s, names), empty_outcome())
    assert int(out.reflect[0]) == 5 + 3 - 1  # counted down once at the turn's end


def test_sticky_barb_moves_to_a_contact_attacker(tables, names):
    s = _item(_item(roark_state(tables, names), 1, "Sticky Barb", names), 0, "", names)
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert int(out.item[0, 0]) == names.item("Sticky Barb") and int(out.item[1, 0]) == 0
    # The new holder takes Divide(maxHP, 8) at the turn's end; Leer dealt nothing.
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) - max(int(s.max_hp[0, 0]) // 8, 1)
    # Ember makes no contact: the barb stays.
    out = assert_matches_reference(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    assert int(out.item[1, 0]) == names.item("Sticky Barb")


def test_pursuit_takes_the_barb_and_rings_the_shell_bell(tables, names):
    s = roark_state(tables, names, opp_moves=["Pursuit"])
    barb = _item(_item(s, 0, "Sticky Barb", names), 1, "", names)
    out = assert_matches_reference(tables, barb, A(5, "Pursuit", barb, names), empty_outcome())
    assert int(out.item[1, 0]) == names.item("Sticky Barb") and int(out.item[0, 0]) == 0
    bell = _hp(_item(s, 1, "Shell Bell", names), 1, 5)
    out = assert_matches_reference(tables, bell, A(5, "Pursuit", bell, names), empty_outcome())
    dealt = int(bell.hp[0, 0]) - int(out.hp[0, 0])
    assert dealt > 0 and int(out.hp[1, 0]) == 5 + max(dealt // 8, 1)


def test_activated_custap_berry_cannot_be_plucked(tables, names):
    s = roark_state(tables, names, player_moves=["Roar"], opp_moves=["Bug Bite"])
    s = _item(_item(s, 0, "Custap Berry", names), 1, "Sitrus Berry", names)
    s = _hp(s, 1, int(s.max_hp[1, 0]) // 2)  # the foe's Sitrus triggers after its Bug Bite
    custap, sitrus = names.item("Custap Berry"), names.item("Sitrus Berry")
    heal = int(s.max_hp[1, 0]) * 25 // 100
    acts = A("Roar", "Bug Bite", s, names)  # Bug Bite (0) goes before Roar (-6)
    for low in (True, False):
        state = _hp(s, 0, int(s.max_hp[0, 0]) // 4 if low else int(s.max_hp[0, 0]))
        out = assert_matches_reference(tables, state, acts, empty_outcome())
        assert int(out.item_consumed[0, 0]) == custap and int(out.item[0, 0]) == 0
        assert int(out.hp[1, 0]) == int(state.hp[1, 0]) + heal  # the Sitrus healed once
        # Activated (a pinch): Bug Bite fails and the Sitrus is used up. Not activated: Bug Bite
        # eats the Custap Berry with no effect, and the pluck flag it leaves keeps the Sitrus
        # (BATTLE_SUBSCRIPT_PLUCK_CHECK), which does not trigger again above half HP.
        assert int(out.item[1, 0]) == (0 if low else sitrus)
        assert int(out.item_consumed[1, 0]) == (sitrus if low else 0)


@pytest.mark.parametrize("berry", ["Aspear Berry", "Lum Berry"])
def test_freeze_berry_acts_before_a_fire_hit_thaws(tables, names, berry):
    s = _item(roark_state(tables, names), 1, berry, names)
    s = s._replace(status=s.status.at[1, 0].set(Status.FREEZE))
    out = assert_matches_reference(tables, s, A("Ember", "Tackle", s, names), empty_outcome())
    assert int(out.status[1, 0]) == Status.NONE
    assert int(out.item_consumed[1, 0]) == names.item(berry) and int(out.item[1, 0]) == 0


def test_shell_bell_heals_after_the_move(tables, names):
    s = _hp(_item(roark_state(tables, names), 0, "Shell Bell", names), 0, 5)
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert dealt > 0 and int(out.hp[0, 0]) == 5 + max(dealt // 8, 1)
