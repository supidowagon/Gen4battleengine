"""D3-16 (owes D2-20): no target through the four-actor step. Battler 3 (1 HP) Double-Edges itself
out and battler 2's Tackle takes battler 1 (1 HP) out, both before battler 0 acts (real
CalcTurnOrder, order 3, 2, 0, 1); the foe party has a bench, so the battle goes on.
``BattleControllerPlayer_HasNoTarget`` (bcp.c:2306-2338) after the PP is spent (bcp.c:3181-3187,
:2238); UpdateMoveBuffers writes no move without SYSCTL_MOVE_SUCCEEDED (:3944-3948);
subscript_no_target.s:4-18 unlocks the move; RANGE_OPPONENT_SIDE keeps a defender
(battle_lib.c:1661-1664); Fly's first turn charges with the user as defender (bcp.c:2326-2334).
Engine equal to the reference through ``play``."""

from __future__ import annotations

import numpy as np

from nuzline.engine.ids import ItemId
from tests.engine.d2_states import four, with_hp, with_moves
from tests.engine.test_d3_action_loop import _set_item, hp, play, pp

ORDER = (3, 2, 0, 1)
TARGETS = (1, 0, 1, 0)  # battler 0 and battler 2 choose battler 1; battler 3 chooses battler 0


def foes_fall(tables, names, move, item=None):
    """Battler 0 knows ``move``; both foes are down when it acts."""
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = with_hp(with_hp(s, 1, 1), 3, 1)
    s = with_moves(s, names, 3, ["Double-Edge"])  # its recoil takes it out
    s = with_moves(s, names, 2, ["Tackle"])
    s = with_moves(s, names, 0, [move])
    if item is not None:
        s = _set_item(s, 0, item)
    return s


def run(tables, s, order=ORDER, targets=TARGETS):
    out, shown = play(tables, s, [0, 0, 0, 0], order, targets=targets)
    assert hp(out, 1) == 0 and hp(out, 3) == 0  # the setup: both foes are down
    return out, shown


def test_a_single_target_move_spends_pp_and_leaves_no_move(tables, names):
    tackle = names.move("Tackle")
    s = foes_fall(tables, names, "Tackle", names.item("Choice Band"))
    s = s._replace(last_move=s.last_move.at[0].set(tackle))
    out, shown = run(tables, s)
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1
    assert int(out.last_move[0]) == 0  # no SYSCTL_MOVE_SUCCEEDED
    assert int(out.locked_move[0]) == 0
    assert int(out.choice_lock[0]) == tackle  # the attack message shows: UpdateMoveBuffers locks
    assert int(shown["no_target"][0]) == 1
    assert hp(out, 2) > 0 and hp(out, 0) > 0


def test_spikes_still_land_on_the_foe_side(tables, names):
    s = foes_fall(tables, names, "Spikes")
    out, shown = run(tables, s)
    assert int(out.spikes[1]) == int(s.spikes[1]) + 1
    assert int(shown["no_target"][0]) == 0  # RandomOpponent returns a 0-HP opponent
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1


def test_helping_hand_at_an_absent_partner_reaches_its_script_and_fails(tables, names):
    """Helping Hand's +5 priority puts it first, so battler 2 cannot fall earlier in the turn:
    battler 2 is absent from the start (0 HP, nothing left behind it in party 0). The partner is
    the defender whatever its HP (RANGE_ALLY), so it is not a no-target; BtlCmd_TryHelpingHand
    fails on ``curHP`` 0 (battle_script.c:6297-6303). PP is spent."""
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = with_moves(s, names, 0, ["Helping Hand"])
    s = s._replace(hp=s.hp.at[0, 1:].set(0))  # party 0: only battler 0 has HP
    out, shown = play(tables, s, [0, 0, 0, 0], (0, 1, 3, 2), targets=(2, 0, 0, 0))
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1
    assert int(shown["no_target"][0]) == 0
    assert hp(out, 2) == 0


def test_fly_charges_with_the_user_as_its_defender(tables, names):
    fly = names.move("Fly")
    s = foes_fall(tables, names, "Fly")
    out, shown = run(tables, s)
    assert int(out.locked_move[0]) == fly  # the charge's lock
    assert int(out.semi_invulnerable[0]) != 0
    assert int(shown["no_target"][0]) == 0
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1


def test_fly_with_a_power_herb_strikes_at_no_target(tables, names):
    """The herb skips the charge (``charge_keeps_user`` is false), so the strike turn finds no
    target; Fly is still the turn's move (HasNoTarget's first-of-multi-turn case, bcp.c:2327-2334)."""
    fly = names.move("Fly")
    herb = int(np.nonzero(np.asarray(tables.item_effect) == ItemId.POWER_HERB)[0][0])
    s = foes_fall(tables, names, "Fly", herb)
    out, _ = run(tables, s)
    assert int(out.last_move[0]) == fly
    assert int(out.semi_invulnerable[0]) == 0
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1
