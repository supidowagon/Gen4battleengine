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
