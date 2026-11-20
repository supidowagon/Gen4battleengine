"""D3 review fix B: Helping Hand at B = 4 and the MULTI Knock Off side bit.

Every case plays the action phase on the engine against ``ref_step(..., actions_only=True)``
(``play_turn``: every field but the Follow Me seam), so each assertion holds in both engines.

Helping Hand (effect_script_0176.s:5, :12-14): BtlCmd_TryHelpingHand (battle_script.c:6291-6314)
sets the partner's turnFlags.helpingHand when the partner is not switching, not MOVE_END, has HP,
and neither battler's flag is set; CalcMoveDamage then multiplies movePower by 15/10
(battle_lib.c:6694-6696) and Future Sight's setup multiplies the stored damage by 15/10 again
(battle_script.c:6092-6094). The flag is a turn flag (``ts_helping_hand``): it shows at a U-turn's
pause, and is 0 after an action phase that ran to its end.

Knock Off (battle_script.c:6627): the knockedOffItemsMask bit is the defender SIDE's, by its
selectedPartySlot. In MULTI battler 0 (party 0) and battler 2 (party 2) share side 0, so a Knock
Off on battler 2's slot k sets the bit for party 0's slot k: party 0's party item of slot k is no
longer written (battle_io_command.c:590-592, the side's mask from battle_controller.c:1472), and a
party-0 newcomer in slot k holds nothing in battle (InitBattleMon, battle_lib.c:161-163).
"""

from __future__ import annotations

import numpy as np

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import calc_move_damage
from nuzline.engine.format import DOUBLES, MULTI
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import party_held_item
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.test_d3_owed_follow_me import idle, play_turn, resume_turn

HH = "Helping Hand"
# battler 0 aims at its partner (Helping Hand ignores it), battler 1 at battler 0, battler 2 at 1
TARGETS = (2, 0, 1, 0)


def helped(tables, names, user_move=HH):
    """DOUBLES: battler 0 ``user_move``, battler 2 Tackle at battler 1, the others idle."""
    s = with_moves(idle(tables, names), names, 0, [user_move])
    return with_moves(s, names, 2, ["Tackle"])


def flags(state):
    return np.asarray(state.ts_helping_hand).tolist()


def lost(state, b):
    """The HP battler ``b``'s active Pokemon is below its max (every party member starts full)."""
    p, slot = party_of(state, b), int(state.active[b])
    return int(state.max_hp[p, slot]) - int(state.hp[p, slot])


# --- Helping Hand ------------------------------------------------------------------------------


def test_helping_hand_boosts_the_partners_later_hit(tables, names):
    """Order 0, 2: battler 2's Tackle after battler 0's Helping Hand deals more than after a
    Swords Dance (movePower x15/10, battle_lib.c:6694-6696); the flag is 0 after the actions."""
    s = helped(tables, names)
    _, out = play_turn(tables, s, [0, 0, 0, 0], (0, 2, 1, 3), TARGETS)
    plain = helped(tables, names, "Swords Dance")
    _, base = play_turn(tables, plain, [0, 0, 0, 0], (0, 2, 1, 3), TARGETS)
    assert lost(out, 1) > lost(base, 1) > 0
    assert flags(out) == [0, 0, 0, 0]


def test_helping_hand_flag_stands_at_a_pause_and_boosts_after_it(tables, names):
    """Battler 0's Helping Hand, then battler 1's U-turn stops at its party list: battler 2's flag
    is set there (:6305) and its Tackle after the resume deals more than without it."""
    s = with_moves(helped(tables, names), names, 1, ["U-turn"])
    _, paused = play_turn(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), TARGETS)
    assert int(paused.pivot_battler) == 2
    assert flags(paused) == [0, 0, 1, 0]
    out, _ = resume_turn(tables, paused, TARGETS)
    plain = with_moves(helped(tables, names, "Swords Dance"), names, 1, ["U-turn"])
    _, plain_paused = play_turn(tables, plain, [0, 0, 0, 0], (0, 1, 2, 3), TARGETS)
    assert flags(plain_paused) == [0, 0, 0, 0]
    base, _ = resume_turn(tables, plain_paused, TARGETS)
    # battler 1's newcomer (party slot 2, full HP) takes the Tackle
    assert int(out.active[1]) == 2 and int(base.active[1]) == 2
    assert lost(out, 1) > lost(base, 1) > 0
    assert flags(out) == [0, 0, 0, 0]


def test_helping_hand_fails_when_the_partner_has_moved(tables, names):
    """Battler 2 acts first: its PICK_COMMAND is MOVE_END (:6301), so no flag at the pause."""
    s = with_moves(helped(tables, names), names, 1, ["U-turn"])
    _, paused = play_turn(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), TARGETS)
    assert int(paused.pivot_battler) == 2
    assert flags(paused) == [0, 0, 0, 0]


def test_helping_hand_fails_with_the_partner_at_zero_hp(tables, names):
    """The partner's curHP is 0 (:6302): no flag at the pause."""
    s = with_hp(with_moves(helped(tables, names), names, 1, ["U-turn"]), 2, 0)
    _, paused = play_turn(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), TARGETS)
    assert int(paused.pivot_battler) == 2
    assert flags(paused) == [0, 0, 0, 0]


def test_a_helped_user_cannot_help_back(tables, names):
    """Battler 2's Helping Hand flags battler 0; battler 0's own Helping Hand then fails on its own
    flag (ATTACKER_TURN_FLAGS.helpingHand, :6302): the pause shows battler 0's flag only."""
    s = with_moves(helped(tables, names), names, 2, [HH])
    s = with_moves(s, names, 1, ["U-turn"])
    _, paused = play_turn(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), (0, 0, 0, 0))
    assert int(paused.pivot_battler) == 2
    assert flags(paused) == [1, 0, 0, 0]


def test_helped_future_sight_stores_the_damage_boosted_twice(tables, names):
    """BtlCmd_TryFutureSight: CalcMoveDamage with the helpingHand power (lib.c:6694-6696), the
    variance, then x15/10 again (battle_script.c:6092-6094)."""
    s = with_moves(helped(tables, names), names, 2, ["Future Sight"])
    ordered, out = play_turn(tables, s, [0, 0, 0, 0], (0, 2, 1, 3), TARGETS)
    move = names.move("Future Sight")
    flagged = ordered._replace(ts_helping_hand=ordered.ts_helping_hand.at[2].set(1))
    calc = int(calc_move_damage(tables, flagged, 2, move, def_side=1))
    rolled = 85 + int(empty_outcome(DOUBLES).roll[2, 0])
    assert int(out.future_damage[1]) == max(calc * rolled // 100, 1) * 15 // 10


# --- MULTI Knock Off: the side's bit -----------------------------------------------------------


def knock_battler_2(tables, names, slot):
    """MULTI: battler 2 (party 2) is out from ``slot`` holding Leftovers; party 0's ``slot`` holds
    an Exp. Share; battler 1 Knock Offs battler 2 first."""
    s = four(tables, names, MULTI)
    s = s._replace(active=s.active.at[2].set(slot))
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = with_moves(s, names, 1, ["Knock Off"])
    s = s._replace(
        item=s.item.at[2, slot]
        .set(names.item("Leftovers"))
        .at[0, slot]
        .set(names.item("Exp. Share"))
    )
    return play_turn(tables, s, [0, 0, 0, 0], (1, 0, 2, 3), (0, 2, 0, 0))


def test_multi_knock_off_on_battler_2_covers_party_0s_active_slot(tables, names):
    """Slot 0: battler 0's own active slot. The side's bit is set (script.c:6627); battler 2 holds
    nothing; battler 0 keeps its battle item (only InitBattleMon reads the bit, lib.c:161); both
    parties keep their slot-0 party item (battle_io_command.c:590-592)."""
    _, out = knock_battler_2(tables, names, 0)
    assert np.asarray(out.knocked_off).tolist()[0][0] == 1
    assert int(out.item[2, 0]) == 0
    assert int(out.item[0, 0]) == names.item("Exp. Share")
    assert int(party_held_item(out, 2, 0)) == names.item("Leftovers")
    assert int(party_held_item(out, 0, 0)) == names.item("Exp. Share")
    assert int(out.knocked_off[1, 0]) == 0


def test_multi_knock_off_on_battler_2_empties_party_0s_newcomer(tables, names):
    """Slot 1, on party 0's bench: after the Knock Off, battler 0 sends out slot 1, which holds
    nothing in battle (InitBattleMon, lib.c:161-163) while its party keeps the Exp. Share."""
    _, out = knock_battler_2(tables, names, 1)
    assert int(out.knocked_off[0, 1]) == 1 and int(out.knocked_off[0, 0]) == 0
    assert int(out.item[0, 1]) == names.item("Exp. Share")  # not loaded yet
    _, after = play_turn(tables, out, [MOVE_ACTIONS + 1, 0, 0, 0], (0, 1, 2, 3), (0, 0, 0, 0))
    assert int(after.active[0]) == 1
    assert int(after.item[0, 1]) == 0
    assert int(party_held_item(after, 0, 1)) == names.item("Exp. Share")
