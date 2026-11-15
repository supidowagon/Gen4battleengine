"""E2 batch 8 (rulings B8-n, docs/kaizo/e2-mechanics-ledger.md): the moves behind the AI's
fields, hand cases each checked against the reference engine (``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()
MISS = O._replace(hit=jnp.full((2, 5), 100, jnp.int8))  # a roll class above any rate below 100


def test_lock_on_makes_the_next_move_hit_and_runs_out(tables, names):
    """B8-1: subscript_lock_on.s sets the defender's count to 2 and lockOnTarget (:7-9); the turn
    end drops it (bcp.c:1618-1624); MON_IS_LOCKED_ONTO skips the miss (bcp.c:3016-3018), Sheer
    Cold's too (TryOHKOMove, battle_script.c:4343-4361)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Glalie", ["Lock-On", "Sheer Cold"], 50)],
        [_mon(n, "Geodude", ["Growl"], 30), _mon(n, "Onix", ["Growl"], 30)],
    )
    aimed = assert_matches_reference(tables, s, (0, 0), O)
    assert int(aimed.lock_on[1]) == 1 and int(aimed.lock_on_target[1]) == 1
    ko = assert_matches_reference(tables, aimed, (1, 0), MISS)
    assert int(ko.hp[1, 0]) == 0
    missed = assert_matches_reference(tables, s, (1, 0), MISS)  # no lock: the roll misses
    assert int(missed.hp[1, 0]) == int(s.hp[1, 0])


def test_lock_on_ends_with_its_lockers_switch_or_replacement(tables, names):
    """B8-2 (as corrected by B8-7): UpdateAfterSwitch clears the lock of a battler the leaving
    one locked onto (battle_lib.c:2063-2067); CleanupFaintedMon has no such loop (:2156-2205), so
    after the locker's faint the lock stays until the replacement's SwitchAndUpdateMon runs
    UpdateAfterSwitch (battle_script.c:2522) for the slot, which clears it the same way."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Glalie", ["Tackle"], 50), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Onix", ["Tackle"], 30)],
    )
    s = s._replace(lock_on=s.lock_on.at[1].set(2), lock_on_target=s.lock_on_target.at[1].set(1))
    switched = assert_matches_reference(tables, s, (MOVE_ACTIONS + 1, 0), O)
    assert int(switched.lock_on[1]) == 0 and int(switched.lock_on_target[1]) == 0
    fainted = assert_matches_reference(tables, s.replace_hp(0, 0, 1), (0, 0), O)
    assert int(fainted.hp[0, 0]) == 0
    assert int(fainted.lock_on[1]) == 1 and int(fainted.lock_on_target[1]) == 1
    from nuzline.engine.step import replace
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace
    from nuzline.data.tables import to_numpy

    sent = replace(tables, fainted, 0, 1)
    assert int(sent.lock_on[1]) == 0 and int(sent.lock_on_target[1]) == 0
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(fainted), 0, 1))
    assert int(ref.lock_on[1]) == 0 and int(ref.lock_on_target[1]) == 0


def test_perish_song_counts_down_and_faints_at_zero(tables, names):
    """B8-3: BtlCmd_TryPerishSong gives every living battler the effect and a count of 3
    (battle_script.c:5465-5487); the side-condition pass drops it at each turn end and faints a
    battler that it finds at 0 (bcp.c:1791-1820). Both last Pokemon at 0 faint in one pass with no
    battle-over check between (AnyFainted, :1749): the draw TurnEnd plays as a loss."""
    n = names
    s = make_state(
        tables, [_mon(n, "Misdreavus", ["Perish Song"], 30)], [_mon(n, "Geodude", ["Growl"], 30)]
    )
    sung = assert_matches_reference(tables, s, (0, 0), O)
    assert [int(x) for x in sung.perish_song] == [1, 1]
    assert [int(x) for x in sung.perish_song_turns] == [2, 2]
    doomed = sung._replace(perish_song_turns=jnp.zeros_like(sung.perish_song_turns))
    end = assert_matches_reference(tables, doomed, (0, 0), O)
    assert [int(x) for x in end.hp[:, 0]] == [0, 0] and int(end.winner) == 1


def test_yawn_sleeps_at_the_second_turn_end_through_safeguard(tables, names):
    """B8-4: subscript_yawn.s sets the count 2 (TryYawn, battle_script.c:6587); the turn end
    drops it (bcp.c:1684-1687) and at 0 runs subscript_fall_asleep.s as a move effect, whose path
    has no Safeguard test (:4-14), with its own Random 3, 2 draw (``yawn_sleep``)."""
    from nuzline.engine.ids import Status

    n = names
    s = make_state(tables, [_mon(n, "Slakoth", ["Yawn"], 30)], [_mon(n, "Geodude", ["Growl"], 30)])
    drowsy = assert_matches_reference(tables, s, (0, 0), O)
    assert int(drowsy.yawn[1]) == 1 and int(drowsy.status[1, 0]) == Status.NONE
    guarded = drowsy._replace(safeguard=drowsy.safeguard.at[1].set(3))
    o = O._replace(yawn_sleep=jnp.asarray([2, 4], jnp.int8))
    out = assert_matches_reference(tables, guarded, (0, 0), o)
    assert int(out.status[1, 0]) == Status.SLEEP and int(out.status_counter[1, 0]) == 4
    assert int(out.yawn[1]) == 0


def test_yawn_meets_insomnia(tables, names):
    """B8-4: an ignorable Insomnia makes Yawn ineffective (subscript_yawn.s:5, :25-30), tagging
    the ability (BattleAI_SetAbility through TAG_NICKNAME_ABILITY)."""
    n = names
    s = make_state(tables, [_mon(n, "Murkrow", ["Growl"], 30)], [_mon(n, "Slakoth", ["Yawn"], 30)])
    s = s._replace(ai_ability_known=jnp.zeros_like(s.ai_ability_known))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.yawn[0]) == 0 and int(out.ai_ability_known[0]) == 1


def test_ghost_curse_halves_the_user_and_hurts_the_foe_each_turn(tables, names):
    """B8-5: a Ghost user's Curse (effect_script_0109.s:5-6, :9-17) curses the foe and costs
    Divide(maxHP, 2) (subscript_curse_ghost.s:6-12); MON_COND_CHECK_STATE_CURSE takes
    Divide(maxHP, 4) at each turn end (bcp.c:1451-1459, subscript_curse_damage.s). A cursed foe
    makes the next Curse fail (:4)."""
    n = names
    s = make_state(tables, [_mon(n, "Gastly", ["Curse"], 30)], [_mon(n, "Geodude", ["Growl"], 30)])
    out = assert_matches_reference(tables, s, (0, 0), O)
    user, foe = int(s.max_hp[0, 0]), int(s.max_hp[1, 0])
    assert int(out.curse[1]) == 1 and int(out.curse[0]) == 0
    assert int(out.hp[0, 0]) == user - max(user // 2, 1)
    assert int(out.hp[1, 0]) == foe - max(foe // 4, 1)
    again = assert_matches_reference(tables, out, (0, 0), O)
    assert int(again.hp[0, 0]) == int(out.hp[0, 0])  # failed: no cost


def test_future_sight_law_reads_the_lock_after_the_countdown(tables, names):
    """B8-6: MON_COND_CHECK_STATE_LOCK_ON drops the lock (bcp.c:1618-1624) before the
    side-condition pass's Future Sight hit (:1758-1783), which reads MON_IS_LOCKED_ONTO
    (:3016-3018). A lock at 1 with the hit due this turn is gone at the hit: the law must hold the
    unlocked threshold, taken here from the live threshold step reads (``future_threshold``) on
    the state as it stands at the hit, not from ``future_thresholds``."""
    import numpy as np

    from nuzline.engine.enumerate import outcome_laws
    from nuzline.engine.outcome import future_threshold

    n = names
    s = make_state(
        tables, [_mon(n, "Abra", ["Future Sight"], 30)], [_mon(n, "Geodude", ["Growl"], 30)]
    )
    move = n.move("Future Sight")
    s = s._replace(
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(5),
        future_move=s.future_move.at[1].set(move),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
        lock_on=s.lock_on.at[1].set(1),
        lock_on_target=s.lock_on_target.at[1].set(1),
    )
    at_hit = s._replace(lock_on=s.lock_on.at[1].set(0), lock_on_target=s.lock_on_target.at[1].set(0))
    live = int(future_threshold(tables, at_hit, 1))
    assert live < 100
    laws = outcome_laws(tables, s, jnp.asarray([0, 0]))
    ts = np.asarray(laws.future_thresholds[1])[np.asarray(laws.future_possible[1], bool)]
    assert live in set(int(t) for t in ts)
    # And the step itself misses on a roll above the live threshold.
    o = O._replace(future_hit=jnp.asarray([1, live + 1], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
