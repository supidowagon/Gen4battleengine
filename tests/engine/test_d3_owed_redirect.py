"""D3-16 (owes D2-21, D2-31): Lightning Rod / Storm Drain through the four-actor step. The redirect
(``BattleSystem_CheckRedirectionAbilities``, battle_lib.c:1732-1790, BeforeMove's REDIRECT_TARGET,
bcp.c:3204-3206) has no MOVE_STATUS_NO_EFFECTS check; its message (``TriggerRedirectionAbilities``,
:1792-1817; the AI records the holder's ability) runs only without it (:1796, :1806; moves.h:85 =
DID_NOT_HIT | NO_PP | NO_MORE_WORK). The holder
is battler 0 (the AI-record difference at battler 2, D3-17, is not met). Real CalcTurnOrder through
``play`` (engine equal to the reference, field for field)."""

from __future__ import annotations

from nuzline.engine.ids import AbilityId, Status, Type
from tests.engine.d2_states import four, with_ability, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_pause import encore
from tests.engine.test_d3_review_targets import with_status

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


def test_a_move_with_no_pp_is_redirected_silently_and_records_at_the_holder(tables, names):
    """MOVE_STATUS_NO_PP (B10-9): battler 1 is Encored into Thunderbolt, whose slot has 0 PP, and
    the Encore takes over what it chose (FightCommand, bcp.c:1866-1883). The slot is the first of
    two Thunderbolts: Encore's selection (``actions.usable_moves``) leaves a 0-PP encored move no
    legal slot and Struggles, so a 0-PP takeover is reached only while another slot still holds the
    move. DecrementPP flags MOVE_STATUS_NO_PP and returns FALSE (bcp.c:2285-2297), so BeforeMove
    still reaches REDIRECT_TARGET (bcp.c:3204-3206), which has no NO_EFFECTS check
    (battle_lib.c:1732-1790): the holder becomes the defender. NO_PP is a NO_EFFECTS status
    (moves.h:85): no SYSCTL_MOVE_SUCCEEDED (bcp.c:3209-3212), no TryMove, so no redirect message
    (battle_lib.c:1796); subscript_no_pp.s prints the attack message and "no PP left"
    (bcp.c:3316-3319): no hit, no AI record, movePrevByBattler = MOVE_NONE (bcp.c:3945-3949), and
    UpdateFlagsWhenHit (bcp.c:4480-4486, SYSCTL_ATTACK_MESSAGE_SHOWN) writes moveCopied at the
    holder, not at the chosen battler 2."""
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
    tbolt = names.move("Thunderbolt")
    assert int(out.move_copied[0]) == tbolt and int(out.move_copied[2]) == 0


def test_the_no_pp_move_updates_the_holders_hit_flags(tables, names):
    """The case above, UpdateFlagsWhenHit's other writes at the holder (bcp.c:4487-4535): with
    SYSCTL_ATTACK_MESSAGE_SHOWN its choiceLockedMove is cleared (no Choice item, :4489-4495) and
    moveHit = moveCur (:4498-4500); NO_PP is a NO_EFFECTS status and sets no SYSCTL_MOVE_SUCCEEDED,
    so conversion2Move / Battler / Type are cleared (:4511-4512 fail, else branch :4531-4534). Drawn
    state: battlers 0 and 2 asleep (no attack message: their own UpdateMoveBuffers leaves their
    lock, bcp.c:3921-3934) with a Choice lock and a conversion2 record, battler 0 first so its
    own ClearFlags (:4389-4390) runs before the redirect. Battler 2 keeps its lock and its
    conversion2 battler and type (its ClearFlags clears only its moveHit and conversion2Move)."""
    s = state(tables, names, AbilityId.LIGHTNINGROD, "Thunderbolt")
    s = with_moves(s, names, 1, ["Swords Dance", "Thunderbolt", "Thunderbolt"])
    slot1 = int(s.active[1])
    s = s._replace(pp=s.pp.at[1, slot1, 1].set(0))
    s = encore(s, names, 1, 1, "Thunderbolt")
    sd, tackle = names.move("Swords Dance"), names.move("Tackle")
    for b in (0, 2):
        s = with_status(s, b, Status.SLEEP, 4)
        s = s._replace(
            choice_lock=s.choice_lock.at[b].set(sd),
            conv2_move=s.conv2_move.at[b].set(tackle),
            conv2_battler=s.conv2_battler.at[b].set(4),  # 1 + battler 3
            conv2_type=s.conv2_type.at[b].set(int(Type.FIRE)),
        )
    out, _ = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=TARGETS)
    tbolt = names.move("Thunderbolt")
    assert int(out.move_copied[0]) == tbolt
    assert int(out.move_hit[0]) == tbolt and int(out.move_hit[2]) == 0
    assert int(out.choice_lock[0]) == 0 and int(out.choice_lock[2]) == sd
    assert [int(out.conv2_move[0]), int(out.conv2_battler[0]), int(out.conv2_type[0])] == [0, 0, 0]
    assert [int(out.conv2_battler[2]), int(out.conv2_type[2])] == [4, int(Type.FIRE)]
