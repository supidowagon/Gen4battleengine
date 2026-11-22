"""Move locks: Bide (``effect_script_0026.s``, ``subscript_bide_{start,end,no_target}.s``,
``CHECK_STATUS_STATE_BIDE`` battle_controller_player.c:2694), Rollout and Ice Ball
(``effect_script_0117.s``, ``BtlCmd_CalcRolloutPower`` battle_script.c:5661) and Defense
Curl (``effect_script_0156.s``).

A battler locked into a move (``VOLATILE_CONDITION_MOVE_LOCKED``) picks no command and
spends no PP (battle_controller_player.c:335); every script that disrupts a move
unlocks it (``Battler_UnlockMoveChoice``, battle_lib.c:1834).

The player is a Kricketot (Bide, priority +1, so it moves first every turn) against a
Geodude, both with 400 HP so nothing faints, unless a test says otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import legal_actions, legal_opponent_actions, locked_slot
from nuzline.engine.ids import Status
from nuzline.engine.order import metronome_candidates
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from tests.engine.helpers import A, assert_matches_reference, jit_step, roark_state

BIG = 400
SPE = 4  # stats[..., 0..4] are Atk..Spe


def big_hp(s, side, slot, hp=BIG):
    return s._replace(
        hp=s.hp.at[side, slot].set(hp),
        max_hp=s.max_hp.at[side, slot].set(max(hp, int(s.max_hp[side, slot]))),
    )


def bide_state(tables, names, *, opp_moves=("Tackle",), player_count=1, opp_count=1):
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_moves=("Bide", "Growl"),
        opp_species="Geodude",
        opp_moves=opp_moves,
        player_count=player_count,
        opp_count=opp_count,
    )
    return big_hp(big_hp(s, 0, 0), 1, 0)


def locked_into_bide(s, names, turns=2, damage=30):
    return s._replace(
        locked_move=s.locked_move.at[0].set(names.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(turns),
        bide_damage=s.bide_damage.at[0].set(damage),
    )


def acts(s, names, opp="Tackle"):
    """The player's locked action (its own slot) against ``opp``."""
    return jnp.asarray([int(locked_slot(s, 0)), names.move_slot(s, 1, opp)], jnp.int32)


def play(tables, s, actions, outcome=None):
    """One turn through both engines (every field compared); the JAX state."""
    return assert_matches_reference(
        tables, s, actions, empty_outcome() if outcome is None else outcome
    )


# --- Bide -----------------------------------------------------------------------------


@pytest.mark.slow
def test_bide_stores_and_unleashes(tables, names):
    bide = names.move("Bide")
    s0 = bide_state(tables, names)
    slot = names.move_slot(s0, 0, "Bide")

    s1 = play(tables, s0, A("Bide", "Tackle", s0, names))
    lost1 = int(s0.hp[0, 0]) - int(s1.hp[0, 0])
    assert lost1 > 0  # Bide has +1 priority: Tackle lands after Bide started
    assert int(s1.locked_move[0]) == bide and int(s1.bide_turns[0]) == 2
    assert int(s1.pp[0, 0, slot]) == int(s0.pp[0, 0, slot]) - 1
    assert int(s1.bide_damage[0]) == lost1

    s2 = play(tables, s1, acts(s1, names))
    assert int(s2.bide_turns[0]) == 1 and int(s2.locked_move[0]) == bide
    assert int(s2.bide_damage[0]) == int(s0.hp[0, 0]) - int(s2.hp[0, 0]) > lost1
    assert int(s2.pp[0, 0, slot]) == int(s1.pp[0, 0, slot])  # a locked battler spends no PP

    s3 = play(tables, s2, acts(s2, names))
    # Rock/Ground does not resist the unleash: no type chart, no crit, no roll.
    assert int(s2.hp[1, 0]) - int(s3.hp[1, 0]) == 2 * int(s2.bide_damage[0])
    assert int(s3.locked_move[0]) == 0 and int(s3.bide_turns[0]) == 0
    assert int(s3.bide_damage[0]) == 0
    assert int(s3.pp[0, 0, slot]) == int(s1.pp[0, 0, slot])


def test_unleash_ignores_roll_and_crit(tables, names):
    s = locked_into_bide(bide_state(tables, names), names, turns=1, damage=37)
    seen = set()
    for roll, crit in ((0, 0), (15, 1)):
        o = empty_outcome()._replace(
            roll=jnp.full((2, MAX_HITS), roll, jnp.int8),
            crit=jnp.full((2, MAX_HITS), crit, jnp.int8),
        )
        out = play(tables, s, acts(s, names), o)
        seen.add(int(s.hp[1, 0]) - int(out.hp[1, 0]))
    assert seen == {74}


def test_bide_fails_without_damage(tables, names):
    s = bide_state(tables, names, opp_moves=("Leer",))
    start = s
    s = play(tables, s, A("Bide", "Leer", s, names))
    for _ in range(2):
        s = play(tables, s, acts(s, names, "Leer"))
    assert int(s.hp[1, 0]) == int(start.hp[1, 0])  # "But it failed!"
    assert int(s.locked_move[0]) == 0 and int(s.bide_turns[0]) == 0
    assert int(s.last_move[0]) == 0  # a failed move is not a successful one


def test_bide_hits_the_replacement(tables, names):
    s = big_hp(bide_state(tables, names, opp_count=2), 1, 1, 900)
    for _ in range(2):
        s = play(tables, s, A("Bide", "Tackle", s, names) if _ == 0 else acts(s, names))
    stored = int(s.bide_damage[0])
    out = play(tables, s, jnp.asarray([int(locked_slot(s, 0)), 5], jnp.int32))  # foe -> slot 1
    assert int(out.active[1]) == 1
    assert int(s.hp[1, 1]) - int(out.hp[1, 1]) == 2 * stored
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_bide_without_target_is_lost(tables, names):
    """The foe fainted before the unleash: ``subscript_bide_no_target.s`` unlocks."""
    s = locked_into_bide(bide_state(tables, names, opp_count=2), names, turns=1, damage=30)
    s = s._replace(hp=s.hp.at[1, 0].set(0))
    out = play(tables, s, jnp.asarray([int(locked_slot(s, 0)), 0], jnp.int32))
    assert int(out.locked_move[0]) == 0 and int(out.bide_damage[0]) == 0


@pytest.mark.parametrize("move", ["SolarBeam", "Rollout"])
def test_bide_release_unlocks_before_the_move(tables, names, move):
    """Review M1 (tracelaws 26ea67d7): on the release turn ``CHECK_STATUS_STATE_BIDE`` runs
    subscript_bide_end.s before the move (bcp.c:2722-2724, commandNext = command), and its :11
    clears VOLATILE_CONDITION_MOVE_LOCKED. A Metronome called Bide, and Encore then landed on
    Metronome (movePrevByBattler = moveTemp, bcp.c:3945-3946): on the release turn FightCommand's
    Encore branch (bcp.c:1869-1873, before the relock at :1884-1887) runs Metronome, which calls
    ``move``. SolarBeam then charges (effect_script_0151.s:9 tests only the lock) and Rollout
    starts at its first power (battle_script.c:5667-5674); the lock either sets stays after it."""
    s = locked_into_bide(bide_state(tables, names), names, turns=1, damage=30)
    metronome = names.move("Metronome")
    s = s._replace(
        moves=s.moves.at[0, 0, 0].set(metronome),
        encore_move=s.encore_move.at[0].set(metronome),
        encore_slot=s.encore_slot.at[0].set(0),
        encore_turns=s.encore_turns.at[0].set(3),
    )
    called = names.move(move)
    rank = int(jnp.sum(metronome_candidates(tables, s, 0)[:called]))
    call = empty_outcome().call
    out = play(
        tables, s, acts(s, names, "Tackle"), empty_outcome()._replace(call=call.at[0].set(rank))
    )
    assert int(out.locked_move[0]) == called
    assert int(out.bide_turns[0]) == 0 and int(out.bide_damage[0]) == 0
    if move == "SolarBeam":
        assert int(out.hp[1, 0]) == int(s.hp[1, 0])  # the charge turn: no damage
    else:
        assert int(out.rollout_count[0]) == 4  # rolloutCount 5, decremented once


def test_locked_side_actions(tables, names):
    """Review Focus 2: a locked side's only legal action is its locked slot."""
    s0 = bide_state(tables, names, player_count=3, opp_count=3)
    s = play(tables, s0, A("Bide", "Tackle", s0, names))
    slot = names.move_slot(s, 0, "Bide")
    assert int(locked_slot(s, 0)) == slot and int(locked_slot(s, 1)) == -1
    legal = np.asarray(legal_actions(tables, s, 0))
    assert legal.tolist() == [i == slot for i in range(10)]  # no switch
    free = np.asarray(legal_actions(tables, s._replace(locked_move=s.locked_move * 0), 0))
    assert free[4:].any()

    # The opponent locked into its own move: neither a switch nor a bag item.
    foe = s._replace(locked_move=s.locked_move.at[1].set(names.move("Tackle")))
    legal = np.asarray(legal_opponent_actions(tables, foe))
    assert np.flatnonzero(legal).tolist() == [names.move_slot(foe, 1, "Tackle")]
    free = np.asarray(legal_opponent_actions(tables, foe._replace(locked_move=foe.locked_move * 0)))
    assert free[10:].any()


def test_locked_with_no_pp_does_not_struggle(tables, names):
    s = locked_into_bide(bide_state(tables, names).with_pp(0, 0, [0, 0, 0, 0]), names)
    assert np.flatnonzero(np.asarray(legal_actions(tables, s, 0))).tolist() == [
        names.move_slot(s, 0, "Bide")
    ]
    out = play(tables, s, acts(s, names))
    assert int(out.last_move[0]) == names.move("Bide")
    assert int(out.pp[0, 0, 0]) == 0


# --- unlocking ------------------------------------------------------------------------


def _paralysis(s, names):
    return (
        s.with_status(0, 0, Status.PARALYSIS),
        empty_outcome()._replace(full_para=jnp.asarray([1, 0], jnp.int8)),
    )


def _flinch(s, names):
    return s._replace(flinched=s.flinched.at[0].set(1)), empty_outcome()


def _confusion(s, names):
    s = s._replace(confusion=s.confusion.at[0].set(3))
    return s, empty_outcome()._replace(confusion_hit=jnp.asarray([1, 0], jnp.int8))


def _disabled(s, names):
    return s._replace(disabled_move=s.disabled_move.at[0].set(names.move("Bide"))), (
        empty_outcome()
    )


DISRUPTIONS = (
    pytest.param(_paralysis, "Tackle", id="full-paralysis"),
    pytest.param(_flinch, "Tackle", id="flinch"),
    pytest.param(_confusion, "Tackle", id="confusion-self-hit"),
    pytest.param(_disabled, "Tackle", id="disabled"),
)


@pytest.mark.parametrize(("setup", "opp"), DISRUPTIONS)
def test_disruption_unlocks(tables, names, setup, opp):
    s = locked_into_bide(bide_state(tables, names), names)
    s, outcome = setup(s, names)
    out = play(tables, s, acts(s, names, opp), outcome)
    assert int(out.locked_move[0]) == 0
    assert int(out.bide_turns[0]) == 0 and int(out.rollout_count[0]) == 0
    assert int(out.bide_damage[0]) == 0


def test_falling_asleep_unlocks(tables, names):
    s = locked_into_bide(bide_state(tables, names, opp_moves=("Hypnosis",)), names)
    out = play(tables, s, acts(s, names, "Hypnosis"))
    assert int(out.status[0, 0]) == Status.SLEEP
    assert int(out.locked_move[0]) == 0 and int(out.bide_turns[0]) == 0


def test_freezing_unlocks(tables, names):
    s = locked_into_bide(bide_state(tables, names, opp_moves=("Ice Beam",)), names)
    o = empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8))
    out = play(tables, s, acts(s, names, "Ice Beam"), o)
    assert int(out.status[0, 0]) == Status.FREEZE
    assert int(out.locked_move[0]) == 0 and int(out.bide_turns[0]) == 0


def test_sleep_unlocks_at_turn_end(tables, names):
    """battle_lib.c:2240-2243: a locked battler found asleep when the next turn is set up."""
    s = locked_into_bide(bide_state(tables, names, opp_moves=("Leer",)), names)
    s = s.with_status(0, 0, Status.SLEEP, 3)
    out = play(tables, s, acts(s, names, "Leer"))
    assert int(out.locked_move[0]) == 0 and int(out.bide_turns[0]) == 0


# --- Rollout and Defense Curl ---------------------------------------------------------


def rollout_state(tables, names, move="Rollout", opp="Bidoof", opp_count=1):
    s = roark_state(
        tables,
        names,
        player_species="Bidoof",
        player_moves=(move, "Defense Curl"),
        opp_species=opp,
        opp_moves=("Leer",),
        player_count=1,
        opp_count=opp_count,
    )
    return big_hp(big_hp(s, 0, 0), 1, 0, 30000)


def rolling(s, names, left=3, move="Rollout"):
    return s._replace(
        locked_move=s.locked_move.at[0].set(names.move(move)),
        rollout_count=s.rollout_count.at[0].set(left),
    )


def single_use_damage(tables, names, s, move, power):
    """What one fresh, uncurled use of ``move`` deals with its table power set to ``power``."""
    mid = names.move(move)
    t = tables._replace(move_power=tables.move_power.at[mid].set(power))
    fresh = s._replace(curled=s.curled * 0, locked_move=s.locked_move * 0)
    out = jit_step(t, fresh, A(move, "Leer", fresh, names), empty_outcome())
    return int(s.hp[1, 0]) - int(out.hp[1, 0])


def roll_five(tables, names, s, move):
    damages, slot = [], names.move_slot(s, 0, move)
    pp = int(s.pp[0, 0, slot])
    for use in range(5):
        before = s
        s = play(tables, s, A(move, "Leer", s, names) if use == 0 else acts(s, names, "Leer"))
        damages.append(int(before.hp[1, 0]) - int(s.hp[1, 0]))
        if use < 4:
            assert int(s.locked_move[0]) == names.move(move)
            assert int(s.rollout_count[0]) == 4 - use
            assert np.flatnonzero(np.asarray(legal_actions(tables, s, 0))).tolist() == [slot]
    assert int(s.locked_move[0]) == 0 and int(s.rollout_count[0]) == 0
    assert int(s.pp[0, 0, slot]) == pp - 1  # PP is spent once
    return damages


@pytest.mark.slow
@pytest.mark.parametrize("move", ["Rollout", "Ice Ball"])
def test_rollout_power_and_curl(tables, names, move):
    s = rollout_state(tables, names, move)
    plain = roll_five(tables, names, s, move)
    assert plain == [single_use_damage(tables, names, s, move, 30 * 2**k) for k in range(5)]

    # Defense Curl: Def +1, `curled`, and every Rollout use doubles (60..960).
    curl = play(tables, s, A("Defense Curl", "Leer", s, names))
    assert int(curl.curled[0]) == 1 and int(curl.stages[0, 2]) == 0  # +1 from Curl, -1 from Leer
    curled = roll_five(tables, names, curl, move)
    assert curled == [single_use_damage(tables, names, s, move, 60 * 2**k) for k in range(5)]
    assert int(s.stages[0, 2]) == 0


def test_defense_curl_sets_the_flag_at_plus_six(tables, names):
    s = rollout_state(tables, names)
    s = s._replace(stages=s.stages.at[0, 2].set(6))
    out = play(tables, s, A("Defense Curl", "Leer", s, names))
    assert int(out.curled[0]) == 1 and int(out.stages[0, 2]) == 5  # Leer -1 after the cap


def test_rollout_miss_unlocks(tables, names):
    s = rolling(rollout_state(tables, names), names)
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = play(tables, s, acts(s, names, "Leer"), miss)
    assert int(out.locked_move[0]) == 0 and int(out.rollout_count[0]) == 0


def test_first_rollout_miss_never_locks(tables, names):
    s = rollout_state(tables, names)
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = play(tables, s, A("Rollout", "Leer", s, names), miss)
    assert int(out.locked_move[0]) == 0 and int(out.rollout_count[0]) == 0
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1


def test_rollout_without_target_unlocks(tables, names):
    s = rolling(rollout_state(tables, names, opp_count=2), names)
    s = s._replace(hp=s.hp.at[1, 0].set(0))
    out = play(tables, s, jnp.asarray([int(locked_slot(s, 0)), 0], jnp.int32))
    assert int(out.locked_move[0]) == 0 and int(out.rollout_count[0]) == 0


# --- the reference engine ----------------------------------------------------------------


def test_locks_match_reference(tables, names):
    s0 = bide_state(tables, names, opp_count=2)
    # Bide through its three turns, with and without damage taken.
    for opp in ("Tackle", "Leer"):
        s = bide_state(tables, names, opp_moves=(opp,), opp_count=2)
        s = play(tables, s, A("Bide", opp, s, names))
        for _ in range(2):
            s = play(tables, s, acts(s, names, opp))
    # Stored damage 0 / 1 / large at the unleash, a dead foe, a Biding foe.
    for stored in (0, 1, 399):
        s = locked_into_bide(s0, names, turns=1, damage=stored)
        play(tables, s, acts(s, names))
    foe = locked_into_bide(s0, names, turns=2, damage=5)
    foe = foe._replace(
        locked_move=foe.locked_move.at[1].set(names.move("Tackle")),
        bide_turns=foe.bide_turns.at[1].set(2),
    )
    play(tables, foe, jnp.asarray([int(locked_slot(foe, 0)), 0], jnp.int32))
    # Rollout / Ice Ball, curled or not, hit or miss, at every count.
    for move in ("Rollout", "Ice Ball"):
        r = rollout_state(tables, names, move)
        for left in (0, 1, 2, 4):
            for curled in (0, 1):
                for hit in (0, 1):
                    s = rolling(r, names, left, move)._replace(curled=r.curled.at[0].set(curled))
                    o = empty_outcome()._replace(hit=jnp.full((2, MAX_HITS), hit, jnp.int8))
                    play(tables, s, acts(s, names, "Leer"), o)
        play(tables, r, A(move, "Leer", r, names))
        play(tables, r, A("Defense Curl", "Leer", r, names))
