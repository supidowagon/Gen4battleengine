"""E2 batch 10 (rulings B10-n, docs/kaizo/e2-mechanics-ledger.md): the rest of the Gardenia
split's mechanics, hand cases each checked against the reference engine
(``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.ids import Stat, Status, Type
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon, _resume

O = empty_outcome()


def _roll(u: int):
    return O._replace(hit=jnp.full((2, 5), u, jnp.int8))


def test_volt_tackle_recoils_a_third_and_paralyzes(tables, names):
    """B10-1: subscript_recoil_1_3_chance_to_paralyze.s: Divide(hitDamage, 3) recoil, then the
    paralysis under SYSCTL_APPLY_SECONDARY_EFFECT."""
    n = names
    s = make_state(
        tables, [_mon(n, "Pikachu", ["Volt Tackle"], 40)], [_mon(n, "Bidoof", ["Growl"], 40)]
    )
    o = O._replace(secondary=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert dealt > 0
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(dealt // 3, 1)
    assert int(out.status[1, 0]) == Status.PARALYSIS
    quiet = assert_matches_reference(tables, s, (0, 0), O)
    assert int(quiet.status[1, 0]) == Status.NONE


def test_flare_blitz_thaws_its_frozen_user(tables, names):
    """B10-1: RECOIL_BURN_HIT lets a frozen user move and thaws it (bcp.c:2481, :2730)."""
    n = names
    s = make_state(
        tables, [_mon(n, "Charmeleon", ["Flare Blitz"], 40)], [_mon(n, "Bidoof", ["Growl"], 40)]
    )
    s = s.with_status(0, 0, Status.FREEZE)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])


def test_jump_kick_crashes_on_a_miss(tables, names):
    """B10-2: subscript_crash_on_miss.s: min(Divide(damage, 2), Divide(defender maxHP, 2)) of the
    hit 0 damage the script computed; nothing into a Ghost (the chart's 0)."""
    n = names
    s = make_state(
        tables, [_mon(n, "Machop", ["Jump Kick"], 40)], [_mon(n, "Bidoof", ["Growl"], 40)]
    )
    out = assert_matches_reference(tables, s, (0, 0), _roll(0))
    would = int(damage_for_roll(tables, s, 0, n.move("Jump Kick"), False, 15))
    cap = max(int(s.max_hp[1, 0]) // 2, 1)
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == min(max(would // 2, 1), cap)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    ghost = s._replace(types=s.types.at[1, 0].set(jnp.asarray([Type.GHOST, Type.GHOST], jnp.int8)))
    out = assert_matches_reference(tables, ghost, (0, 0), _roll(100))
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])


def test_baton_pass_keeps_mean_look_lock_on_perish_song_and_curse(tables, names):
    """Batch 3-9 merge check: UpdateAfterSwitch keeps, under SYSCTL_BATON_PASS, the substitute
    HP, lockOnTarget and perishSongTurns (battle_lib.c:2107-2115), with the moveEffectsMask bits
    (Lock-On, Perish Song) and the volatile bits (Mean Look, Curse) the switch keeps
    (:2073-2085). The newcomer still carries all four after the turn's end, in both engines."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Ninjask", ["Baton Pass"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Growl"], 20)],
    )
    s = s._replace(
        mean_look=s.mean_look.at[0].set(1),
        mean_look_target=s.mean_look_target.at[0].set(2),
        lock_on=s.lock_on.at[0].set(2),  # the foe locked onto the passer
        lock_on_target=s.lock_on_target.at[0].set(2),
        perish_song=s.perish_song.at[0].set(1),
        perish_song_turns=s.perish_song_turns.at[0].set(3),
        curse=s.curse.at[0].set(1),
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    after = _resume(tables, paused, 1)
    assert int(after.active[0]) == 1
    assert int(after.mean_look[0]) == 1 and int(after.mean_look_target[0]) == 2
    assert int(after.perish_song[0]) == 1 and int(after.perish_song_turns[0]) == 2
    assert int(after.curse[0]) == 1
    assert int(after.hp[0, 1]) < int(after.max_hp[0, 1])  # the curse's turn-end damage
    assert int(after.lock_on[0]) != 0 and int(after.lock_on_target[0]) == 2


def test_any_item_holder_lead_arms_unburden(tables, names):
    """InitBattleMon arms canUnburden for any lead holding an item (battle_lib.c:161-166),
    whatever its ability; a lead with no item has it clear."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Pikachu", ["Growl"], 30, "Oran Berry")],
        [_mon(n, "Bidoof", ["Growl"], 30)],
    )
    assert int(s.can_unburden[0]) == 1 and int(s.can_unburden[1]) == 0


def _subbed(s, side: int, hp: int):
    return s._replace(substitute_hp=s.substitute_hp.at[side].set(hp))


def test_substitute_costs_a_quarter_and_fails_twice_or_when_weak(tables, names):
    """B10-4: BtlCmd_TrySubstitute (battle_script.c:5211-5226): Divide(maxHP, 4) off the user into
    substituteHP; subscript_try_substitute.s:7 fails behind one, :8 at curHP <= the quarter."""
    n = names
    s = make_state(
        tables, [_mon(n, "Pikachu", ["Substitute"], 40)], [_mon(n, "Bidoof", ["Growl"], 40)]
    )
    quarter = max(int(s.max_hp[0, 0]) // 4, 1)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.substitute_hp[0]) == quarter
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == quarter
    again = assert_matches_reference(tables, out, (0, 0), O)
    assert int(again.hp[0, 0]) == int(out.hp[0, 0]) and int(again.substitute_hp[0]) == quarter
    weak = s._replace(hp=s.hp.at[0, 0].set(quarter))
    failed = assert_matches_reference(tables, weak, (0, 0), O)
    assert int(failed.substitute_hp[0]) == 0 and int(failed.hp[0, 0]) == quarter


def test_a_hit_takes_the_substitute_not_the_hp(tables, names):
    """B10-4: bcp.c:3359-3378: the substitute takes the damage (at most its HP) and breaks at 0;
    the defender's HP stays."""
    n = names
    s = make_state(
        tables, [_mon(n, "Machop", ["Karate Chop"], 40)], [_mon(n, "Bidoof", ["Growl"], 40)]
    )
    big = _subbed(s, 1, 200)
    out = assert_matches_reference(tables, big, (0, 0), O)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    assert 0 < int(out.substitute_hp[1]) < 200
    small = _subbed(s, 1, 1)
    broke = assert_matches_reference(tables, small, (0, 0), O)
    assert int(broke.substitute_hp[1]) == 0 and int(broke.hp[1, 0]) == int(s.hp[1, 0])


def test_substitute_blocks_status_drops_and_drains_the_share(tables, names):
    """B10-4: Thunder Wave (subscript_paralyze.s:21) and Growl (battle_script.c:2921-2943) fail on
    a substitute; Giga Drain heals from the substitute's share (hitDamage, bcp.c:3363, :3367)."""
    n = names
    foe = [_mon(n, "Bidoof", ["Growl"], 40)]
    s = _subbed(make_state(tables, [_mon(n, "Pikachu", ["Thunder Wave", "Growl"], 40)], foe), 1, 50)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[1, 0]) == Status.NONE
    out = assert_matches_reference(tables, s, (1, 0), O)
    assert int(out.stages[1, Stat.ATK]) == int(s.stages[1, Stat.ATK])
    s = make_state(tables, [_mon(n, "Oddish", ["Mega Drain"], 40)], foe)
    s = _subbed(s._replace(hp=s.hp.at[0, 0].set(5)), 1, 1)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.substitute_hp[1]) == 0 and int(out.hp[1, 0]) == int(s.hp[1, 0])
    assert int(out.hp[0, 0]) == 6  # max(Divide(1 * 50 / 100), 1)


def test_multi_hit_breaks_the_substitute_then_hits_the_hp_without_static(tables, names):
    """B10-4: the loop goes on after the break (no stop at bcp.c:3780), later hits on real HP;
    SUBSTITUTE_HIT stays for the action, so Static never triggers (battle_lib.c:4209)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Machop", ["Double Kick"], 40)],
        [_mon(n, "Pikachu", ["Growl"], 40)],
    )
    s = _subbed(s, 1, 1)
    o = O._replace(contact_proc=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.substitute_hp[1]) == 0
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # the second kick
    assert int(out.status[0, 0]) == Status.NONE  # no Static


def test_kings_rock_never_flinches_through_a_substitute_hit(tables, names):
    """B10-4 (Owed: King's Rock vs Substitute): subscript_flinch_mon.s:6 CheckSubstitute reads the
    flag a hit of the action set, so Double Kick's second kick on real HP does not flinch the foe:
    it still moves (Growl); without the substitute it flinches."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Machop", ["Double Kick"], 40, "Razor Fang")],
        [_mon(n, "Snorlax", ["Growl"], 40)],
    )
    o = O._replace(flinch_item=jnp.ones((2, 5), jnp.int8))
    plain = assert_matches_reference(tables, s, (0, 0), o)
    assert int(plain.stages[0, Stat.ATK]) == 0  # flinched: no Growl
    out = assert_matches_reference(tables, _subbed(s, 1, 1), (0, 0), o)
    assert int(out.substitute_hp[1]) == 0 and int(out.hp[1, 0]) < int(s.hp[1, 0])
    assert int(out.stages[0, Stat.ATK]) == -1  # the foe moved


def test_pursuit_on_a_switcher_only_breaks_a_substitute(tables, names):
    """B10-4: subscript_pursuit.s:20, :56-73: a substitute the hit does not break keeps its HP and
    the switcher's HP stays; one it breaks goes (the switcher's HP stays too)."""
    from nuzline.engine.actions import MOVE_ACTIONS

    n = names
    s = make_state(
        tables,
        [_mon(n, "Houndour", ["Pursuit"], 40)],
        [_mon(n, "Bidoof", ["Growl"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    switch = (0, MOVE_ACTIONS + 1)
    plain = assert_matches_reference(tables, s, switch, O)
    assert int(plain.hp[1, 0]) < int(s.hp[1, 0])
    for sub in (1, 200):
        out = assert_matches_reference(tables, _subbed(s, 1, sub), switch, O)
        assert int(out.hp[1, 0]) == int(s.hp[1, 0]), sub


def test_ghost_curse_fails_on_a_substitute_without_its_cost(tables, names):
    """B10-4 (Owed: Ghost Curse's substitute test): subscript_curse_ghost.s:6 CheckSubstitute jumps
    to the failure (:22-24) before the curse and the HP cost (:9-15)."""
    n = names
    s = make_state(tables, [_mon(n, "Gastly", ["Curse"], 40)], [_mon(n, "Bidoof", ["Growl"], 30)])
    out = assert_matches_reference(tables, _subbed(s, 1, 30), (0, 0), O)
    assert int(out.curse[1]) == 0 and int(out.hp[0, 0]) == int(s.hp[0, 0])


def test_endeavor_cuts_to_the_users_hp_or_fails(tables, names):
    """B10-5: effect_script_0189.s: damage = the defender's curHP - the attacker's; fails when it
    is <= 0; a Ghost is immune (SYSCTL_IGNORE_TYPE_CHECKS still flags it)."""
    n = names
    s = make_state(
        tables, [_mon(n, "Rattata", ["Endeavor"], 40)], [_mon(n, "Snorlax", ["Growl"], 40)]
    )
    s = s._replace(hp=s.hp.at[0, 0].set(7))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[1, 0]) == 7
    even = s._replace(hp=s.hp.at[1, 0].set(7))
    assert int(assert_matches_reference(tables, even, (0, 0), O).hp[1, 0]) == 7
    ghost = make_state(
        tables, [_mon(n, "Rattata", ["Endeavor"], 40)], [_mon(n, "Gastly", ["Growl"], 40)]
    )
    ghost = ghost._replace(hp=ghost.hp.at[0, 0].set(7))
    assert int(assert_matches_reference(tables, ghost, (0, 0), O).hp[1, 0]) == int(ghost.hp[1, 0])


def test_pain_split_averages_and_fails_on_a_substitute(tables, names):
    """B10-5: subscript_pain_split.s: both HP to (a + d) / 2, capped at max HP; a substitute fails
    it (:5)."""
    n = names
    s = make_state(
        tables, [_mon(n, "Misdreavus", ["Pain Split"], 40)], [_mon(n, "Snorlax", ["Growl"], 40)]
    )
    s = s._replace(hp=s.hp.at[0, 0].set(10))
    out = assert_matches_reference(tables, s, (0, 0), O)
    avg = (10 + int(s.hp[1, 0])) // 2
    assert int(out.hp[0, 0]) == min(avg, int(s.max_hp[0, 0]))
    assert int(out.hp[1, 0]) == avg
    sub = assert_matches_reference(tables, _subbed(s, 1, 30), (0, 0), O)
    assert int(sub.hp[0, 0]) == 10 and int(sub.hp[1, 0]) == int(s.hp[1, 0])


def test_knock_off_takes_the_item_and_marks_the_slot(tables, names):
    """B10-6: BtlCmd_TryKnockOff (battle_script.c:6620-6627): heldItem = ITEM_NONE and the slot's
    knockedOffItemsMask bit; an ignorable Sticky Hold keeps it (:6614); a substitute stops it
    (subscript_knock_off.s:5)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Sneasel", ["Knock Off"], 40)],
        [_mon(n, "Snorlax", ["Growl"], 40, "Leftovers")],
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.item[1, 0]) == 0 and int(out.knocked_off[1, 0]) == 1
    assert int(out.item_consumed[1, 0]) == 0
    sub = assert_matches_reference(tables, _subbed(s, 1, 200), (0, 0), O)
    assert int(sub.item[1, 0]) == n.item("Leftovers") and int(sub.knocked_off[1, 0]) == 0
    shellos = make_state(
        tables,
        [_mon(n, "Sneasel", ["Knock Off"], 40)],
        [_mon(n, "Shellos", ["Growl"], 40, "Leftovers")],
    )
    kept = assert_matches_reference(tables, shellos, (0, 0), O)
    assert int(kept.item[1, 0]) == n.item("Leftovers")


def test_knock_off_misses_a_quick_claw_holder_yet_to_move(tables, names):
    """B10-6: subscript_knock_off.s:8: the defender's moveEffectsData.quickClaw (set by the turn
    order, battle_lib.c:1283; cleared at its own move's start, subscript_check_quick_claw.s:22-23)
    keeps its item: here it activated but moves last (Whirlwind, priority -6)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Sneasel", ["Knock Off"], 40)],
        [_mon(n, "Snorlax", ["Whirlwind"], 40, "Quick Claw")],
    )
    fired = s._replace(qc_roll=s.qc_roll.at[1].set(1))
    out = assert_matches_reference(tables, fired, (0, 0), O)
    assert int(out.item[1, 0]) == n.item("Quick Claw")
    quiet = assert_matches_reference(tables, s._replace(qc_roll=s.qc_roll.at[1].set(0)), (0, 0), O)
    assert int(quiet.item[1, 0]) == 0


def test_a_knocked_off_attacker_never_takes_a_sticky_barb(tables, names):
    """B10-6: BattleSystem_TriggerHeldItemOnHit (battle_lib.c:5390): the knocked-off mask stops the
    transfer to an attacker holding nothing."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Machop", ["Karate Chop"], 30)],
        [_mon(n, "Snorlax", ["Growl"], 40, "Sticky Barb")],
    )
    plain = assert_matches_reference(tables, s, (0, 0), O)
    assert int(plain.item[0, 0]) == n.item("Sticky Barb")
    knocked = s._replace(knocked_off=s.knocked_off.at[0, 0].set(1))
    out = assert_matches_reference(tables, knocked, (0, 0), O)
    assert int(out.item[0, 0]) == 0 and int(out.item[1, 0]) == n.item("Sticky Barb")


def test_safeguard_sets_five_turns_and_blocks_status_and_confusion(tables, names):
    """B10-7: BtlCmd_TrySafeguard (battle_script.c:5754-5772): 5 turns, failing while up; the
    status subscripts (paralyze :28) and subscript_confuse.s:26 stop at it; Swagger still raises
    the Attack."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Chansey", ["Safeguard"], 40)],
        [_mon(n, "Pikachu", ["Thunder Wave", "Swagger"], 10)],
    )
    up = assert_matches_reference(tables, s, (0, 0), O)
    assert int(up.safeguard[0]) == 4  # set to 5, then this turn's count-down
    assert int(up.status[0, 0]) == Status.NONE  # the slower Thunder Wave met it
    again = assert_matches_reference(tables, up, (0, 1), O)
    assert int(again.confusion[0]) == 0 and int(again.stages[0, Stat.ATK]) == 2
    assert int(again.safeguard[0]) == 3  # a second use failed: the count only went down


def test_safeguard_stops_toxic_spikes_and_flame_orb(tables, names):
    """B10-7: subscript_poison.s:41 (Toxic Spikes on a newcomer) and subscript_burn.s:16 (Flame
    Orb at the turn end)."""
    from nuzline.engine.actions import MOVE_ACTIONS

    n = names
    s = make_state(
        tables,
        [_mon(n, "Bidoof", ["Growl"], 30, "Flame Orb"), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Snorlax", ["Growl"], 30)],
    )
    s = s._replace(safeguard=s.safeguard.at[0].set(5), toxic_spikes=s.toxic_spikes.at[0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE
    switched = assert_matches_reference(tables, s, (MOVE_ACTIONS + 1, 0), O)
    assert int(switched.status[0, 1]) == Status.NONE


def test_laws_no_sure_flinch_through_a_substitute(tables, names):
    """B10-4 (law): moving first, Headbutt never flinches a foe behind a substitute
    (subscript_flinch_mon.s:6), so ``Reads.flinch_sure`` must not let its roll drop the foe's
    hit: the exact laws equal brute force. A slow user under Trick Room, the foe at a 1-HP
    substitute."""
    import numpy as np

    from tests.engine.test_damage_classes import assert_equals_bruteforce

    n = names
    s = make_state(
        tables, [_mon(n, "Pikachu", ["Headbutt"], 12)], [_mon(n, "Starly", ["Tackle"], 30)]
    )
    s = _subbed(s, 1, 1)._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    s = s._replace(hp=s.hp.at[0, 0].set(3))
    assert_equals_bruteforce(tables, s, np.zeros(2, np.int32))


def test_focus_punch_loses_its_focus_after_a_hit_but_not_a_substitute_hit(tables, names):
    """B10-8: effect_script_0170.s:5-6: a user hit on its HP this turn (the turnFlags attacker
    masks, bcp.c:3416-3424) loses its focus (MOVE_STATUS_LOST_FOCUS); a hit on its substitute sets
    no mask (bcp.c:3359-3378 returns first), so it punches."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Machop", ["Focus Punch"], 40)],
        [_mon(n, "Snorlax", ["Tackle"], 40)],
    )
    hit = assert_matches_reference(tables, s, (0, 0), O)
    assert int(hit.hp[1, 0]) == int(s.hp[1, 0])  # lost its focus
    sub = assert_matches_reference(tables, _subbed(s, 0, 200), (0, 0), O)
    assert int(sub.hp[1, 0]) < int(s.hp[1, 0])
    calm = make_state(
        tables,
        [_mon(n, "Machop", ["Focus Punch"], 40)],
        [_mon(n, "Snorlax", ["Growl"], 40)],
    )
    out = assert_matches_reference(tables, calm, (0, 0), O)
    assert int(out.hp[1, 0]) < int(calm.hp[1, 0])


def test_encore_locks_the_last_move_and_counts_down(tables, names):
    """B10-9: BtlCmd_TryEncore (battle_script.c:4725-4748) encores the target's last move for
    RandNext() % 5 + 3 turns (outcome.duration); CheckInvalidMoves (battle_lib.c:2302) then leaves
    only that move, which the selection takes (bcp.c:443-446); the count drops at the turn end
    (bcp.c:1591-1612). Encore fails on a target that has not moved."""
    from nuzline.engine.actions import usable_moves

    n = names
    s = make_state(
        tables,
        [_mon(n, "Clefairy", ["Encore"], 40)],
        [_mon(n, "Bidoof", ["Growl", "Tackle"], 30)],
    )
    fresh = assert_matches_reference(tables, s, (0, 1), O)
    assert int(fresh.encore_move[1]) == 0  # Clefairy moved first: no last move yet
    grown = s._replace(last_move=s.last_move.at[1].set(n.move("Growl")))
    o = O._replace(duration=jnp.asarray([5, 2], O.duration.dtype))
    out = assert_matches_reference(tables, grown, (0, 1), o)
    assert int(out.encore_move[1]) == n.move("Growl") and int(out.encore_slot[1]) == 0
    assert int(out.encore_turns[1]) == 4  # 5, then this turn's count-down
    assert [bool(x) for x in usable_moves(tables, out, 1)] == [True, False, False, False]


def test_a_faster_encore_takes_over_this_turns_move(tables, names):
    """B10-9: FightCommand (bcp.c:1869-1873) runs the encored move whatever was chosen: Bidoof
    chose Tackle, a faster Encore on its last Growl makes it Growl this turn, from Growl's PP."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Clefairy", ["Encore"], 40)],
        [_mon(n, "Bidoof", ["Growl", "Tackle"], 30)],
    )
    s = s._replace(last_move=s.last_move.at[1].set(n.move("Growl")))
    out = assert_matches_reference(tables, s, (0, 1), O)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])  # no Tackle
    assert int(out.stages[0, Stat.ATK]) == -1  # a Growl
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1 and int(out.pp[1, 0, 1]) == int(
        s.pp[1, 0, 1]
    )


def test_knock_off_leaves_the_ai_record_until_the_next_copy_to_party(tables, names):
    """B10-6: TryKnockOff runs no CopyToParty, so the AI's record of the player's knocked item
    stays (aiContext.battlerHeldItems) until the battler's next CopyToParty: its move attempt
    (bcp.c:2748) or an HP-bar update (battle_script.c:1720), not the knock-off hit's own update."""
    n = names
    # Ninjask moves first (its record stays); Bidoof moves after the knock-off (cleared).
    for species, kept in (("Ninjask", True), ("Bidoof", False)):
        s = make_state(
            tables,
            [_mon(n, species, ["Growl"], 30, "Leftovers")],
            [_mon(n, "Sneasel", ["Knock Off"], 30)],
        )
        s = s._replace(ai_item_known=s.ai_item_known.at[0].set(n.item("Leftovers")))
        out = assert_matches_reference(tables, s, (0, 0), O)
        assert int(out.item[0, 0]) == 0 and int(out.knocked_off[0, 0]) == 1
        assert (int(out.ai_item_known[0]) == n.item("Leftovers")) == kept, species


def test_assist_calls_a_party_mates_move(tables, names):
    """B10-11: BtlCmd_TryAssist (battle_script.c:6405-6444): the moves of the other party slots
    (a fainted mate's too), invokers and Metronome-banned moves left out, RandNext() % numMoves
    (``Outcome.call``); with no mate it fails."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Skitty", ["Assist"], 30), _mon(n, "Machop", ["Growl", "Tackle"], 30)],
        [_mon(n, "Snorlax", ["Splash"], 30)],
    )
    growl = assert_matches_reference(
        tables, s, (0, 0), O._replace(call=jnp.asarray([0, 0], O.call.dtype))
    )
    assert int(growl.stages[1, Stat.ATK]) == -1
    tackle = assert_matches_reference(
        tables, s, (0, 0), O._replace(call=jnp.asarray([1, 0], O.call.dtype))
    )
    assert int(tackle.hp[1, 0]) < int(s.hp[1, 0]) and int(tackle.stages[1, Stat.ATK]) == 0
    fainted = s._replace(hp=s.hp.at[0, 1].set(0))
    still = assert_matches_reference(tables, fainted, (0, 0), O)
    assert int(still.stages[1, Stat.ATK]) == -1  # the fainted mate's Growl
    alone = make_state(
        tables, [_mon(n, "Skitty", ["Assist"], 30)], [_mon(n, "Snorlax", ["Splash"], 30)]
    )
    out = assert_matches_reference(tables, alone, (0, 0), O)
    assert int(out.stages[1, Stat.ATK]) == 0 and int(out.hp[1, 0]) == int(alone.hp[1, 0])


def test_fly_charges_vanishes_then_hits_and_dodges_tackle(tables, names):
    """B10-12: effect_script_0155.s: the first use charges (no PP on the second turn, the lock
    into Fly, MOVE_EFFECT_AIRBORNE); the foe's Tackle misses the airborne user
    (bcp.c:3034-3040); the second turn hits and clears the bit and the lock."""
    from nuzline.engine.ids import SemiKind

    n = names
    s = make_state(tables, [_mon(n, "Pidgeotto", ["Fly"], 40)], [_mon(n, "Bidoof", ["Tackle"], 20)])
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.semi_invulnerable[0]) == SemiKind.AIRBORNE
    assert int(one.locked_move[0]) == n.move("Fly")
    assert int(one.hp[1, 0]) == int(s.hp[1, 0])
    assert int(one.hp[0, 0]) == int(s.hp[0, 0])  # the slower foe's Tackle missed
    assert int(one.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
    two = assert_matches_reference(tables, one, (0, 0), O)
    assert int(two.semi_invulnerable[0]) == 0 and int(two.locked_move[0]) == 0
    assert int(two.hp[1, 0]) < int(one.hp[1, 0])
    assert int(two.pp[0, 0, 0]) == int(one.pp[0, 0, 0])


def test_gust_doubles_and_earthquake_misses_a_flyer(tables, names):
    """B10-12: effect_script_0149.s: Gust hits an airborne foe at powerMul 20; Earthquake
    (effect_script_0147.s: SYSCTL_HIT_DURING_DIG only) misses it."""
    from nuzline.engine.ids import SemiKind

    n = names
    s = make_state(
        tables,
        [_mon(n, "Pidgey", ["Gust", "Earthquake"], 40)],
        [_mon(n, "Pidgeotto", ["Fly"], 40)],
    )
    up = s._replace(
        semi_invulnerable=s.semi_invulnerable.at[1].set(SemiKind.AIRBORNE),
        locked_move=s.locked_move.at[1].set(n.move("Fly")),
    )
    gust = int(damage_for_roll(tables, s, 0, n.move("Gust"), False, 15))
    doubled = int(damage_for_roll(tables, up, 0, n.move("Gust"), False, 15))
    assert doubled > gust
    quake = assert_matches_reference(tables, up, (1, 0), O)
    # The Earthquake missed; Fly then landed on the player.
    assert int(quake.hp[0, 0]) < int(up.hp[0, 0])
    assert int(quake.semi_invulnerable[1]) == 0
    assert_matches_reference(tables, up, (0, 0), O)


def test_dig_dodges_sandstorm_and_earthquake_doubles(tables, names):
    """B10-12: MOVE_EFFECT_NO_WEATHER_DAMAGE (moves.h:50) skips the sandstorm chip of a digging
    battler (battle_script.c:5580); Earthquake doubles on it (effect_script_0147.s:7-8)."""
    from nuzline.engine.ids import SemiKind, Weather

    n = names
    s = make_state(tables, [_mon(n, "Sandshrew", ["Dig"], 30)], [_mon(n, "Pikachu", ["Growl"], 30)])
    s = s._replace(weather=jnp.asarray(Weather.SANDSTORM, s.weather.dtype))
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.semi_invulnerable[0]) == SemiKind.UNDERGROUND
    assert int(one.hp[1, 0]) < int(s.hp[1, 0])  # the foe takes the chip
    d = make_state(tables, [_mon(n, "Pikachu", ["Dig"], 30)], [_mon(n, "Pikachu", ["Growl"], 30)])
    d = d._replace(weather=jnp.asarray(Weather.SANDSTORM, d.weather.dtype))
    dug = assert_matches_reference(tables, d, (0, 0), O)
    assert int(dug.hp[0, 0]) == int(d.hp[0, 0])  # no chip while underground
    assert int(dug.hp[1, 0]) < int(d.hp[1, 0])


def test_solarbeam_fires_in_sun_and_power_herb_skips_the_charge(tables, names):
    """B10-12: effect_script_0151.s:5-6 hits at once in sun; a Power Herb skips any charge and
    is used up (subscript_item_skip_charge_turn.s:20); SolarBeam is halved in rain
    (battle_lib.c:7058-7060)."""
    from nuzline.engine.ids import Weather

    n = names
    s = make_state(
        tables, [_mon(n, "Bulbasaur", ["SolarBeam"], 40)], [_mon(n, "Geodude", ["Growl"], 30)]
    )
    sun = s._replace(weather=jnp.asarray(Weather.SUN, s.weather.dtype))
    out = assert_matches_reference(tables, sun, (0, 0), O)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.locked_move[0]) == 0
    herb = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["SolarBeam"], 40, item="Power Herb")],
        [_mon(n, "Geodude", ["Growl"], 30)],
    )
    out = assert_matches_reference(tables, herb, (0, 0), O)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.item[0, 0]) == 0
    rain = s._replace(weather=jnp.asarray(Weather.RAIN, s.weather.dtype))
    beam = n.move("SolarBeam")
    assert int(damage_for_roll(tables, rain, 0, beam, False, 15)) < int(
        damage_for_roll(tables, s, 0, beam, False, 15)
    )


def test_skull_bash_raises_defense_and_a_flinch_drops_the_charge(tables, names):
    """B10-12: effect_script_0145.s:8 Defense +1 on the charge turn; on the second turn a faster
    foe's flinch (subscript_flinched.s:9, Battler_UnlockMoveChoice) drops the lock, and a
    full paralysis likewise; the next turn the move charges again."""
    n = names
    s = make_state(
        tables, [_mon(n, "Squirtle", ["Skull Bash"], 30)], [_mon(n, "Starly", ["Headbutt"], 30)]
    )
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.stages[0, Stat.DEF]) == 1 and int(one.locked_move[0]) == n.move("Skull Bash")
    flinch = O._replace(secondary=jnp.ones((2, 5), jnp.int8))
    two = assert_matches_reference(tables, one, (0, 0), flinch)
    assert int(two.locked_move[0]) == 0
    para = one.with_status(0, 0, Status.PARALYSIS)
    full = O._replace(full_para=jnp.ones((2,), jnp.int8))
    three = assert_matches_reference(tables, para, (0, 0), full)
    assert int(three.locked_move[0]) == 0


def test_shadow_force_vanishes_and_surf_doubles_on_a_diver(tables, names):
    """B10-12: effect_script_0272.s sets MOVE_EFFECT_SHADOW_FORCE (Surf misses it); Surf
    (effect_script_0257.s) hits and doubles on a diver (MOVE_EFFECT_UNDERWATER); the diver's
    switch clears its bit (a volatile)."""
    from nuzline.engine.ids import SemiKind

    n = names
    s = make_state(
        tables, [_mon(n, "Gengar", ["Shadow Force"], 40)], [_mon(n, "Psyduck", ["Surf"], 30)]
    )
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.semi_invulnerable[0]) == SemiKind.SHADOW_FORCE
    assert int(one.hp[0, 0]) == int(s.hp[0, 0])
    d = make_state(
        tables,
        [_mon(n, "Psyduck", ["Surf"], 40)],
        [_mon(n, "Psyduck", ["Dive"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    under = d._replace(
        semi_invulnerable=d.semi_invulnerable.at[1].set(SemiKind.UNDERWATER),
        locked_move=d.locked_move.at[1].set(n.move("Dive")),
    )
    surf = n.move("Surf")
    assert int(damage_for_roll(tables, under, 0, surf, False, 15)) > int(
        damage_for_roll(tables, d, 0, surf, False, 15)
    )
    out = assert_matches_reference(tables, under, (0, 0), O)
    assert int(out.hp[1, 0]) < int(d.hp[1, 0])
