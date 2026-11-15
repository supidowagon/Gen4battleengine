"""E2 batch 10 (rulings B10-n, docs/kaizo/e2-mechanics-ledger.md): the rest of the Gardenia
split's mechanics, hand cases each checked against the reference engine
(``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.ids import Status, Type
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
