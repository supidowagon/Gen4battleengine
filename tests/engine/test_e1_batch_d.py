"""E1 Batch D (rulings G-53 .. G-60, docs/kaizo/e1-mechanics-ledger.md): the Fantina split's held
items, hand cases checked against the reference engine (``assert_matches_reference``).

Ganlon / Apicot with param 0 (G-53), Jaboca / Rowap (G-54), Enigma (G-55), Iron Ball (G-56), Grip
Claw (G-57), Leppa (G-58), Micle (G-59) and the Metronome item (G-60).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.ai.policy import ai_refusals
from nuzline.engine.coverage import IMPLEMENTED_ITEMS, Support
from nuzline.engine.damage import damage_table
from nuzline.engine.hooks import max_pp
from nuzline.engine.ids import GRIP_CLAW_TURNS, ItemId, Stat
from nuzline.engine.order import effective_speed, pinch_limit
from nuzline.engine.outcome import empty_outcome, hit_threshold
from tests.engine.helpers import A, assert_matches_reference, roark_state

O = empty_outcome()
BATCH_D = {
    "Jaboca Berry": ItemId.JABOCA_BERRY,
    "Rowap Berry": ItemId.ROWAP_BERRY,
    "Enigma Berry": ItemId.ENIGMA_BERRY,
    "Leppa Berry": ItemId.LEPPA_BERRY,
    "Micle Berry": ItemId.MICLE_BERRY,
    "Grip Claw": ItemId.GRIP_CLAW,
    "Metronome": ItemId.METRONOME_ITEM,
    "Iron Ball": ItemId.IRON_BALL,
    "Ganlon Berry": ItemId.PINCH_BERRY,
    "Apicot Berry": ItemId.PINCH_BERRY,
}


def _item(s, side, slot, item):
    return s._replace(item=s.item.at[side, slot].set(item))


def _hp(s, side, slot, hp):
    return s._replace(hp=s.hp.at[side, slot].set(hp))


def test_the_items_are_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    for game_tables, game_names in ((tables, names), (kaizo_tables, kaizo_names)):
        for name, item in BATCH_D.items():
            assert int(game_tables.item_effect[game_names.item(name)]) == item, name
            assert IMPLEMENTED_ITEMS[item] == Support.EXACT


# --- G-53 pinch berries at param 0 ----------------------------------------------------------


def test_pinch_limit_of_a_zero_divisor_is_max_hp():
    """_u32_div_f's ``cmp r1, #0; bxeq lr`` returns the dividend."""
    assert int(pinch_limit(120, 0)) == 120
    assert int(pinch_limit(120, 4)) == 30
    assert int(pinch_limit(121, 1)) == 121


def test_a_param_0_ganlon_berry_triggers_at_full_hp(tables, names):
    ganlon = names.item("Ganlon Berry")
    zero = tables._replace(item_param=tables.item_param.at[ganlon, 0].set(0))
    s = _item(roark_state(zero, names), 0, 0, ganlon)
    assert int(s.hp[0, 0]) == int(s.max_hp[0, 0])
    out = assert_matches_reference(zero, s, A("Leer", "Leer", s, names), O)
    assert int(out.stages[0, Stat.DEF]) == 0  # Leer -1, the berry +1
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == ganlon
    # Vanilla's param 4: no trigger at full HP.
    s4 = _item(roark_state(tables, names), 0, 0, ganlon)
    out = assert_matches_reference(tables, s4, A("Leer", "Leer", s4, names), O)
    assert int(out.stages[0, Stat.DEF]) == -1 and int(out.item[0, 0]) == ganlon


# --- G-54 Jaboca / Rowap, G-55 Enigma -------------------------------------------------------


def _foe_holds(tables, names, item, **kw):
    s = roark_state(tables, names, **kw)
    return _item(s, 1, 0, names.item(item))


def test_jaboca_hurts_a_physical_attacker_and_is_used(tables, names):
    s = _foe_holds(tables, names, "Jaboca Berry")
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), O)
    loss = max(int(s.max_hp[0, 0]) // int(tables.item_param[names.item("Jaboca Berry"), 0]), 1)
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == loss
    assert int(out.item[1, 0]) == 0
    # A special move does not trigger it.
    out = assert_matches_reference(tables, s, A("Ember", "Leer", s, names), O)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.item[1, 0]) != 0


def test_rowap_hurts_a_special_attacker_only(tables, names):
    s = _foe_holds(tables, names, "Rowap Berry")
    out = assert_matches_reference(tables, s, A("Ember", "Leer", s, names), O)
    assert int(out.hp[0, 0]) < int(s.hp[0, 0]) and int(out.item[1, 0]) == 0
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), O)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.item[1, 0]) != 0


def test_enigma_heals_after_a_super_effective_hit(tables, names):
    s = _foe_holds(tables, names, "Enigma Berry")  # Geodude, Rock / Ground
    plain = assert_matches_reference(
        tables, _item(s, 1, 0, 0), A("Mach Punch", "Leer", s, names), O
    )
    out = assert_matches_reference(tables, s, A("Mach Punch", "Leer", s, names), O)
    gain = max(int(s.max_hp[1, 0]) // int(tables.item_param[names.item("Enigma Berry"), 0]), 1)
    assert int(out.hp[1, 0]) == min(int(plain.hp[1, 0]) + gain, int(s.max_hp[1, 0]))
    assert int(out.item[1, 0]) == 0
    # Not super effective (Scratch): no heal.
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), O)
    assert int(out.item[1, 0]) != 0


# --- G-56 Iron Ball -------------------------------------------------------------------------


def test_iron_ball_halves_speed_and_grounds_a_flyer(tables, names):
    s = roark_state(tables, names, player_species="Starly", opp_moves=["Mud-Slap"])
    ball = _item(s, 0, 0, names.item("Iron Ball"))
    assert int(effective_speed(tables, ball, 0)) == int(effective_speed(tables, s, 0)) // 2
    free = assert_matches_reference(tables, s, A("Growl", "Mud-Slap", s, names), O)
    assert int(free.hp[0, 0]) == int(s.hp[0, 0])  # Flying: immune
    held = assert_matches_reference(tables, ball, A("Growl", "Mud-Slap", ball, names), O)
    assert int(held.hp[0, 0]) < int(s.hp[0, 0])
    assert int(damage_table(tables, ball, 1, names.move("Mud-Slap"), False)[0]) > 0


def test_the_ai_no_longer_refuses_a_player_iron_ball(tables, names):
    s = _item(roark_state(tables, names), 0, 0, names.item("Iron Ball"))
    assert not [r for r in ai_refusals(tables, s, names) if "Iron Ball" in r]


# --- G-57 Grip Claw -------------------------------------------------------------------------


def test_grip_claw_binds_for_six_turns(tables, names):
    s = roark_state(tables, names, player_moves=["Wrap"])
    o = O._replace(duration=O.duration.at[0].set(3))
    plain = assert_matches_reference(tables, s, A("Wrap", "Leer", s, names), o)
    claw = _item(s, 0, 0, names.item("Grip Claw"))
    held = assert_matches_reference(tables, claw, A("Wrap", "Leer", claw, names), o)
    assert int(held.trapped[1]) - int(plain.trapped[1]) == GRIP_CLAW_TURNS - 3


# --- G-58 Leppa Berry -----------------------------------------------------------------------


def test_leppa_restores_the_first_empty_slot_capped(tables, names):
    s = _item(roark_state(tables, names), 0, 0, names.item("Leppa Berry"))
    s = s._replace(
        pp=s.pp.at[0, 0, 1].set(0), pp_ups=s.pp_ups.at[0, 0, 1].set(3)
    )  # Leer, 30 * 1.6 = 48
    assert int(max_pp(tables, s, 0, 0)[1]) == 48
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), O)
    power = int(tables.item_param[names.item("Leppa Berry"), 0])
    assert int(out.pp[0, 0, 1]) == min(power, 48) and int(out.item[0, 0]) == 0


def test_plucked_leppa_restores_the_eaters_largest_gap(tables, names):
    s = _item(roark_state(tables, names, player_moves=["Pluck"]), 1, 0, names.item("Leppa Berry"))
    slot = names.move_slot(s, 0, "Pluck")
    out = assert_matches_reference(tables, s, A("Pluck", "Leer", s, names), O)
    assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot])  # the spent PP came back
    assert int(out.item[1, 0]) == 0


# --- G-59 Micle Berry -----------------------------------------------------------------------


def test_micle_sets_its_flag_at_a_pinch_and_the_next_check_uses_it(tables, names):
    s = _item(roark_state(tables, names), 0, 0, names.item("Micle Berry"))
    s = _hp(s, 0, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.micle[0]) == 1 and int(out.item[0, 0]) == 0
    flagged = s._replace(micle=s.micle.at[0].set(1), item=s.item.at[0, 0].set(0))
    acts = A("Leer", "Leer", s, names)
    base = int(hit_threshold(tables, s._replace(item=flagged.item), acts)[0])
    assert int(hit_threshold(tables, flagged, acts)[0]) == base * 120 // 100
    out = assert_matches_reference(tables, flagged, acts, O)
    assert int(out.micle[0]) == 0  # the check cleared it


# --- G-60 the Metronome item ----------------------------------------------------------------


def test_metronome_counts_the_same_move_and_boosts_it(tables, names):
    s = _item(roark_state(tables, names), 0, 0, names.item("Metronome"))
    scratch = names.move("Scratch")
    s = s._replace(
        metronome_move=s.metronome_move.at[0].set(scratch),
        metronome_turns=s.metronome_turns.at[0].set(4),
    )
    acts = A("Scratch", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.metronome_turns[0]) == 5 and int(out.metronome_move[0]) == scratch
    plain = assert_matches_reference(tables, _item(s, 0, 0, 0), acts, O)
    assert int(plain.metronome_turns[0]) == 0
    dealt, base = int(s.hp[1, 0] - out.hp[1, 0]), int(s.hp[1, 0] - plain.hp[1, 0])
    assert dealt > base
    # Another move resets the count and is the new metronomeMove.
    leer = A("Leer", "Leer", s, names)
    out = assert_matches_reference(tables, s, leer, O)
    assert int(out.metronome_turns[0]) == 0 and int(out.metronome_move[0]) == names.move("Leer")


def test_a_missed_move_takes_its_count_back(tables, names):
    s = _item(roark_state(tables, names), 0, 0, names.item("Metronome"))
    leer = names.move("Leer")
    s = s._replace(
        metronome_move=s.metronome_move.at[0].set(leer),
        metronome_turns=s.metronome_turns.at[0].set(4),
    )
    miss = O._replace(hit=O.hit.at[0, 0].set(0))
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), miss)
    assert int(out.metronome_turns[0]) == 4  # +1 at BeforeMove, -1 at UpdateMoveBuffers
    assert np.asarray(out.stages)[1, Stat.DEF] == 0


def test_new_fields_reset_on_a_switch(tables, names):
    s = roark_state(tables, names)
    s = s._replace(
        micle=jnp.asarray([1, 1], s.micle.dtype),
        metronome_turns=jnp.asarray([3, 3], s.metronome_turns.dtype),
        metronome_move=jnp.asarray([10, 10], s.metronome_move.dtype),
    )
    out = assert_matches_reference(tables, s, A(4 + 1, "Leer", s, names), O)
    assert [int(out.micle[0]), int(out.metronome_turns[0]), int(out.metronome_move[0])] == [0, 0, 0]
