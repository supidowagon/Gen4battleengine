"""D3-16 (owes D2-21, D2-31): Lightning Rod / Storm Drain through the four-actor step. The redirect
(``BattleSystem_CheckRedirectionAbilities``, battle_lib.c:1732-1790) and its message
(``TriggerRedirectionAbilities``, :1792-1817; the AI records the holder's ability) only run without
MOVE_STATUS_NO_EFFECTS (:1796, :1806; moves.h:85 = DID_NOT_HIT | NO_PP | NO_MORE_WORK). The holder
is battler 0 (the AI-record difference at battler 2, D3-17, is not met). Real CalcTurnOrder through
``play`` (engine equal to the reference, field for field)."""

from __future__ import annotations

from nuzline.engine.ids import AbilityId
from tests.engine.d2_states import four, with_ability, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_pause import encore

ORDER = (1, 0, 2, 3)  # battler 1 acts first
TARGETS = (0, 2, 0, 0)  # battler 1's move is chosen at battler 2
LR_ID, SD_ID = 31, 114  # the decomp ability ids ``ai_ability_known`` holds


def state(tables, names, holder_ability, move):
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = with_moves(s, names, 1, [move])
    return with_ability(s, 0, holder_ability)


def test_lightning_rod_takes_the_move_and_is_recorded(tables, names):
    s = state(tables, names, AbilityId.LIGHTNINGROD, "Thunderbolt")
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=TARGETS)
    assert hp(out, 0) < hp(s, 0) and hp(out, 2) == hp(s, 2)
    assert int(out.ai_ability_known[int(out.active[0])]) == LR_ID


def test_storm_drain_takes_a_single_target_water_move_and_is_recorded(tables, names):
    s = state(tables, names, AbilityId.STORM_DRAIN, "Water Gun")
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=TARGETS)
    assert hp(out, 0) < hp(s, 0) and hp(out, 2) == hp(s, 2)
    assert int(out.ai_ability_known[int(out.active[0])]) == SD_ID


def test_a_move_with_no_pp_neither_redirects_nor_records(tables, names):
    """MOVE_STATUS_NO_PP (B10-9): battler 1 is Encored into Thunderbolt, whose slot has 0 PP, and
    the Encore takes over what it chose (FightCommand, bcp.c:1866-1883). The slot is the first of
    two Thunderbolts: Encore's selection (``actions.usable_moves``) leaves a 0-PP encored move no
    legal slot and Struggles, so a 0-PP takeover is reached only while another slot still holds the
    move. The move fails at DecrementPP (bcp.c:2285): MOVE_STATUS_NO_PP is a NO_EFFECTS status
    (moves.h:85), so no redirect (battle_lib.c:1796) and no SYSCTL_MOVE_SUCCEEDED (bcp.c:3208-3212);
    subscript_no_pp.s prints the attack message and "no PP left" (bcp.c:3316-3319): no hit, no
    ability message, no AI record, and movePrevByBattler = MOVE_NONE (bcp.c:3945-3949)."""
    s = state(tables, names, AbilityId.LIGHTNINGROD, "Thunderbolt")
    s = with_moves(s, names, 1, ["Swords Dance", "Thunderbolt", "Thunderbolt"])
    slot1 = int(s.active[1])
    s = s._replace(pp=s.pp.at[1, slot1, 1].set(0))
    s = encore(s, names, 1, 1, "Thunderbolt")
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=TARGETS)
    assert hp(out, 0) == hp(s, 0) and hp(out, 2) == hp(s, 2)
    assert int(out.ai_ability_known[int(out.active[0])]) == int(s.ai_ability_known[0])
    assert int(out.pp[1, slot1, 1]) == 0
    assert int(out.last_move[1]) == 0
