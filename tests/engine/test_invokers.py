"""E2 batch 5: the invokers, Sleep Talk, Metronome and Copycat (rulings B5-1..B5-3,
docs/kaizo/e2-mechanics-ledger.md), in both engines, and their call laws in the exact
enumeration (``enumerate.call_laws``, ``call_variants``, ``enumerate_turn``)."""

import math
from collections import Counter
from fractions import Fraction
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import metronome_calls
from nuzline.engine.enumerate import (
    CALL_DISABLED,
    CALL_KEEP,
    CALL_SEALED,
    SamplingOnly,
    _split_events,
    call_laws,
    call_variants,
    enumerate_outcomes,
    enumerate_turn,
    gravity_events,
    outcome_laws,
)
from nuzline.engine.ids import CALL_GRAVITY, METRONOME_HEAVY, EffectId, MoveFlag, Status
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    assist_candidates,
    gravity_state,
    metronome_candidates,
    metronome_weights,
)
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import step, step_events, turn_events
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.helpers import sample_rows, unmodelled
from tests.strategies import legal_action_pair, supported_state


def _mon(names, species: str, moves, level=30) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


_events = jax.jit(step_events)
_loose = jax.jit(partial(outcome_laws, loose=True))


def _both(tables, state, actions, outcome):
    """The JAX turn (checked against the reference engine) and its events."""
    jax_out, events = _events(tables, state, jnp.asarray(actions, jnp.int32), outcome)
    ref = from_ref(ref_step(to_numpy(tables), to_ref(state), actions, outcome_to_py(outcome)))
    for name, a, b in zip(jax_out._fields, jax_out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    return jax_out, events


def _call(player=0, opp=0):
    return empty_outcome()._replace(call=jnp.asarray([player, opp], jnp.int16))


def test_sleep_talk_draws_an_allowed_slot(tables, names):
    """BtlCmd_TrySleepTalk (battle_script.c:4907-4937): the slot ``call % count`` of the allowed
    ones; Bide (Move_IsMultiTurn) and Sleep Talk itself are never drawn; the called move's
    damage lands, the PP and movePrev are Sleep Talk's."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Bide", "Ember"])],
        [_mon(n, "Geodude", ["Growl"])],
    ).with_status(0, 0, Status.SLEEP, 3)
    seen = []
    for u in range(12):
        out, ev = _both(t, s, (0, 0), _call(u))
        seen.append(n.move_names[int(ev["called"][0])])
        assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
        assert int(out.pp[0, 0, 1]) == int(s.pp[0, 0, 1])
        assert int(out.move_prev) == n.move("Growl")  # the foe acted last
    assert seen == ["Tackle", "Ember"] * 6
    # Awake (or waking now): the script fails (effect_script_0097.s:5), nothing is called.
    awake = s.with_status(0, 0, Status.NONE, 0)
    out, ev = _both(t, awake, (0, 0), _call(0))
    assert int(ev["called"][0]) == 0 and int(out.hp[1, 0]) == int(s.hp[1, 0])
    woke = s.with_status(0, 0, Status.SLEEP, 1)
    out, ev = _both(t, woke, (0, 0), _call(0))
    assert int(ev["called"][0]) == 0 and int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_sleep_talk_skips_a_disabled_or_taunt_blocked_slot(tables, names):
    """CheckInvalidMoves without NO_PP (battle_script.c:4929): the disabled move is not drawn; a
    0-PP move is."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Ember"])],
        [_mon(n, "Geodude", ["Growl"])],
    ).with_status(0, 0, Status.SLEEP, 3)
    s = s._replace(disabled_move=s.disabled_move.at[0].set(n.move("Tackle")))
    s = s._replace(disable_turns=s.disable_turns.at[0].set(3))
    s = s.with_pp(0, 0, [10, 30, 0, 0])
    for u in range(3):
        _, ev = _both(t, s, (0, 0), _call(u))
        assert n.move_names[int(ev["called"][0])] == "Ember"


def test_sleep_talk_with_no_living_foe_still_succeeds(tables, names):
    """Ruling B12-30: Sleep Talk's defender is its user (RANGE_SINGLE_TARGET_SPECIAL,
    battle_lib.c:1665-1669), so with the foe fainted to its own confusion self-hit first it
    passes HasNoTarget (bcp.c:3190), uses its PP and sets SYSCTL_MOVE_SUCCEEDED (:3212):
    movePrevByBattler is Sleep Talk (:3945-3946), awake (the script fails,
    effect_script_0097.s:5) or asleep (the called Tackle finds no defender in GoToMoveScript,
    battle_script.c:2121-2126)."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle"])],
        [_mon(n, "Geodude", ["Tackle"], level=50), _mon(n, "Geodude", ["Tackle"])],
    )  # a second foe, so the battle goes on past the faint
    s = s._replace(confusion=s.confusion.at[1].set(3), hp=s.hp.at[1, 0].set(1))
    hurt = _call(0)._replace(confusion_hit=jnp.asarray([0, 1], jnp.int8))
    for asleep in (False, True):
        start = s.with_status(0, 0, Status.SLEEP, 3) if asleep else s
        out, ev = _both(t, start, (0, 0), hurt)
        assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) == int(s.hp[0, 0])
        assert int(out.last_move[0]) == n.move("Sleep Talk"), asleep
        assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
        assert int(out.pp[0, 0, 1]) == int(s.pp[0, 0, 1])


def test_metronome_with_no_living_foe_still_succeeds(tables, names):
    """Ruling B12-30 for Metronome (RANGE_SINGLE_TARGET_SPECIAL as Sleep Talk; the case of the
    manager's diff seed on main c4f0a60): with the foe fainted to its own confusion self-hit
    first, Metronome passes HasNoTarget (bcp.c:3190), uses its PP and is movePrevByBattler
    (:3212, :3945-3946); the called Pound finds no defender (battle_script.c:2121-2126)."""
    t, n = tables, names
    spec = _mon(n, "Clefairy", ["Metronome", "Tackle"])
    s = make_state(
        t,
        [spec],
        [_mon(n, "Geodude", ["Tackle"], level=50), _mon(n, "Geodude", ["Tackle"])],
    )  # a second foe, so the battle goes on past the faint
    s = s._replace(confusion=s.confusion.at[1].set(3), hp=s.hp.at[1, 0].set(1))
    pound = metronome_calls(t, spec).index(n.move("Pound"))
    hurt = _call(pound)._replace(confusion_hit=jnp.asarray([0, 1], jnp.int8))
    out, _ = _both(t, s, (0, 0), hurt)
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert int(out.last_move[0]) == n.move("Metronome")
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1


def test_copycat_copies_the_battles_last_move(tables, names):
    """BtlCmd_TryCopycat (battle_script.c:7328-7341) reads movePrev: the faster foe's move this
    turn, else the turn-start one; MOVE_NONE, a non-Metronome-able move (Protect) fails."""
    t, n = tables, names
    fast_foe = make_state(
        t, [_mon(n, "Snorlax", ["Copycat"])], [_mon(n, "Geodude", ["Tackle"], level=50)]
    )
    out, ev = _both(t, fast_foe, (0, 0), empty_outcome())
    assert n.move_names[int(ev["called"][0])] == "Tackle"
    assert int(out.move_prev) == n.move("Copycat")
    slow_foe = make_state(
        t, [_mon(n, "Snorlax", ["Copycat"], level=50)], [_mon(n, "Snorlax", ["Tackle"], level=25)]
    )
    for prev, want in (("Ember", "Ember"), ("Protect", None), (None, None)):
        s = slow_foe._replace(move_prev=jnp.asarray(n.move(prev) if prev else 0, jnp.int16))
        out, ev = _both(t, s, (0, 0), empty_outcome())
        got = int(ev["called"][0])
        assert (n.move_names[got] if got else None) == want, prev
        assert int(out.move_prev) == n.move("Tackle")


def test_a_disrupted_move_clears_move_prev(tables, names):
    """UpdateMoveBuffers (bcp.c:3936-3942): no attack message (a sleeping foe) leaves MOVE_NONE,
    so a slower Copycat fails."""
    t, n = tables, names
    s = make_state(t, [_mon(n, "Snorlax", ["Copycat"])], [_mon(n, "Geodude", ["Tackle"], level=50)])
    s = s.with_status(1, 0, Status.SLEEP, 3)
    outcome = empty_outcome()
    s = s._replace(move_prev=jnp.asarray(n.move("Ember"), jnp.int16))
    out, ev = _both(t, s, (0, 0), outcome)
    assert int(ev["called"][0]) == 0
    assert int(out.move_prev) == n.move("Copycat")


def test_metronome_calls_by_rank_and_weights(tables, names):
    """BtlCmd_Metronome (battle_script.c:4535-4562): ids 1-467 but the user's moves and
    sCannotMetronomeMoves; ``RandNext() % 467 + 1`` weighs ids 1-156 at 141, the rest 140."""
    t, n = tables, names
    s = make_state(
        t, [_mon(n, "Clefairy", ["Metronome", "Tackle"])], [_mon(n, "Geodude", ["Growl"])]
    )
    spec = _mon(n, "Clefairy", ["Metronome", "Tackle"])
    calls = metronome_calls(t, spec)
    flags = np.asarray(t.move_flags)
    assert n.move("Tackle") not in calls and n.move("Protect") not in calls
    assert all(not flags[m] & MoveFlag.NO_METRONOME for m in calls)
    weights = np.asarray(metronome_weights(t, s, 0))
    assert sorted(np.flatnonzero(weights)) == calls
    assert {int(weights[m]) for m in calls if m <= METRONOME_HEAVY} == {141}
    assert {int(weights[m]) for m in calls if m > METRONOME_HEAVY} == {140}
    for k in (0, 1, 57):
        _, ev = jax.jit(step_events)(t, s, jnp.zeros(2, jnp.int32), _call(k))
        assert int(ev["called"][0]) == calls[k]


def test_call_laws_and_variants(tables, names):
    """Sleep Talk's 12 values group by the move they call; Metronome's candidates each weigh
    their share; Copycat's two orders only where the foe uses a move."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Ember", "Ember"])],
        [_mon(n, "Clefairy", ["Metronome"])],
    ).with_status(0, 0, Status.SLEEP, 3)
    variants = call_variants(call_laws(t, s, jnp.zeros(2, jnp.int32)))
    talk = Counter()
    for v in variants:
        talk[int(v.call[0])] += Fraction(v.num, v.den)
    # Slots Tackle, Ember, Ember: Tackle 1/3 (value 0), Ember 2/3 (value 1).
    assert talk == {0: Fraction(1, 3), 1: Fraction(2, 3)}
    assert math.isclose(sum(v.num / v.den for v in variants), 1.0)
    assert len(variants) == 2 * len(metronome_calls(t, _mon(n, "Clefairy", ["Metronome"])))


CHUNK = 1024  # one compiled step program for every batch in this file
_steps = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_turn_events = jax.jit(jax.vmap(turn_events, in_axes=(None, None, None, 0)))
_order = jax.jit(action_order)


def _raw_brute(tables, state, actions) -> dict[bytes, float]:
    """The children of a turn by brute force over the raw draws, independent of ``call_laws``,
    ``call_variants`` and ``branch_ok``: for each raw ``speed_tie`` value, the order ``step``
    plays (``action_order``) fixes Copycat's movePrev (its foe used a move before it, or the
    turn-start one); each Sleep Talk value weighs 1/12, played under every set assumption
    (``CALL_KEEP``, ``CALL_DISABLED``, ``CALL_SEALED``, ``CALL_GRAVITY``) with each row kept where
    the turn's own ``disabled_at_move`` / ``sealed_at_move`` / ``gravity_at_move`` events agree;
    each Assist list position weighs 1 / its list's length, for the turn-start list and (``+
    CALL_GRAVITY``) the list Gravity leaves, each row kept where a foe's Gravity landed first
    exactly for the latter (G-67); the loose laws with every branch allowed, cut to the rows of
    that ``speed_tie``. Metronome is left to the hand cases."""
    effect = np.asarray(tables.move_effect)
    acts = np.asarray(actions)
    kinds = []
    for sd in (0, 1):
        m = int(np.asarray(state.moves)[sd, int(state.active[sd]), min(int(acts[sd]), 3)])
        kinds.append(int(effect[m]) if int(acts[sd]) < MOVE_ACTIONS else -1)
    zero = jax.tree.map(np.asarray, _loose(tables, state, actions, jnp.zeros(2, jnp.int32)))
    ties = (0, 1) if bool(zero.relevant.speed_tie) else (0,)
    brute: dict[bytes, float] = {}
    for tie in ties:
        tied = empty_outcome()._replace(speed_tie=jnp.int8(tie))
        order = np.asarray(_order(tables, state, actions, tied))
        per_side = []
        for sd in (0, 1):
            if kinds[sd] == EffectId.SLEEP_TALK:
                offsets = (CALL_KEEP, CALL_DISABLED, CALL_SEALED, CALL_GRAVITY)
                per_side.append([(u + off, Fraction(1, 12)) for u in range(12) for off in offsets])
            elif kinds[sd] == EffectId.ASSIST:
                rows = []
                for hyp, off in ((state, 0), (gravity_state(state), CALL_GRAVITY)):
                    n_listed = int(np.sum(np.asarray(assist_candidates(tables, hyp, sd)[1])))
                    rows += [(v + off, Fraction(1, n_listed)) for v in range(n_listed)]
                per_side.append(rows)
            elif kinds[sd] == EffectId.COPYCAT:
                foe = 1 - sd
                first = int(order[0]) == foe and int(acts[foe]) < MOVE_ACTIONS
                per_side.append([(1 if first else 0, Fraction(1))])
            else:
                per_side.append([(0, Fraction(1))])
        for a, wa in per_side[0]:
            for b, wb in per_side[1]:
                laws = _loose(tables, state, actions, jnp.asarray([a, b]))
                laws = jax.tree.map(np.asarray, laws)._replace(branch_ok=np.ones(2, bool))
                o, p = enumerate_outcomes(laws)
                keep = np.ones(len(p), bool)
                if len(ties) == 2:
                    keep &= np.asarray(o.speed_tie) == tie
                events = _turn_events(tables, state, jnp.asarray(actions, jnp.int32), o)
                disabled, sealed = _split_events(state, events)
                disabled, sealed = np.asarray(disabled), np.asarray(sealed)
                landed = np.asarray(gravity_events(state, events))
                for sd, call in ((0, a), (1, b)):
                    if kinds[sd] == EffectId.SLEEP_TALK:
                        want = (
                            3
                            if call >= CALL_GRAVITY
                            else 2
                            if call >= CALL_SEALED
                            else 1
                            if call >= CALL_DISABLED
                            else 0
                        )
                        seen = np.where(
                            landed[:, sd],
                            3,
                            np.where(sealed[:, sd], 2, np.where(disabled[:, sd], 1, 0)),
                        )
                        keep &= seen == want
                    elif kinds[sd] == EffectId.ASSIST:
                        keep &= landed[:, sd] == (call >= CALL_GRAVITY)
                if not keep.any():
                    continue
                rows = jax.tree.map(lambda f, keep=keep: np.asarray(f)[keep], o)
                weight = float(wa * wb)
                for key, q in _children(tables, state, actions, rows, p[keep] * weight).items():
                    brute[key] = brute.get(key, 0.0) + q
    return brute


def _children(tables, state, actions, outcomes, probs):
    """``state bytes -> probability`` of each outcome row's step, merged, in fixed ``CHUNK``
    batches (padding = copies of row 0)."""
    out: dict[bytes, float] = {}
    n = len(probs)
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = jax.tree.map(lambda f, pad=pad: np.asarray(f)[pad], outcomes)
        stepped = _steps(tables, state, jnp.asarray(actions, jnp.int32), rows)
        flat = np.concatenate(
            [np.asarray(f).reshape(CHUNK, -1).astype(np.int64) for f in stepped], axis=1
        )
        for k, row in enumerate(idx):
            key = flat[k].tobytes()
            out[key] = out.get(key, 0.0) + float(probs[row])
    return out


@pytest.mark.parametrize(
    "case",
    [
        "sleep_talk",
        "sleep_talk_disable",
        "sleep_talk_imprison",
        "sleep_talk_gravity",
        "assist_gravity",
        "assist_gravity_slow",
        "copycat_tie",
        "copycat_fast",
        "copycat_switch",
    ],
)
def test_enumeration_equals_the_raw_call_brute_force(tables, names, case):
    """``enumerate_turn`` against the raw brute force (``_raw_brute``): the same merged children.
    ``sleep_talk_disable``: a faster Disable can shrink the Sleep Talk set (ruling B5-13);
    ``copycat_switch``: the foe switches first, so Copycat copies the turn-start movePrev;
    ``*_gravity``: a faster foe's Gravity bans Splash and Jump Kick from the Sleep Talk set and
    the Assist list before the call (``Move_CanBeMetronomed``, battle_lib.c:7198-7201;
    CHECK_INVALID_GRAVITY, :2292-2295; G-67), a slower one does not."""
    t, n = tables, names
    if case == "sleep_talk":
        s = make_state(
            t,
            [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Spore", "Ember"])],
            [_mon(n, "Geodude", ["Tackle"])],
        ).with_status(0, 0, Status.SLEEP, 2)
    elif case == "sleep_talk_disable":
        s = make_state(
            t,
            [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Ember"])],
            [_mon(n, "Gastly", ["Disable"], level=40)],
        ).with_status(0, 0, Status.SLEEP, 2)
        s = s._replace(last_move=s.last_move.at[0].set(n.move("Tackle")))
    elif case == "sleep_talk_gravity":
        s = make_state(
            t,
            [_mon(n, "Snorlax", ["Sleep Talk", "Splash", "Tackle", "Jump Kick"])],
            [_mon(n, "Gastly", ["Gravity"], level=40)],
        ).with_status(0, 0, Status.SLEEP, 2)
    elif case in ("assist_gravity", "assist_gravity_slow"):
        level = 40 if case == "assist_gravity" else 5
        s = make_state(
            t,
            [
                _mon(n, "Snorlax", ["Assist"]),
                _mon(n, "Geodude", ["Splash", "Ember"]),
                _mon(n, "Bulbasaur", ["Jump Kick", "Water Gun"]),
            ],
            [_mon(n, "Gastly", ["Gravity"], level=level)],
        )
    elif case == "sleep_talk_imprison":
        # B7-13: a faster foe's Imprison seals the shared Tackle before the Sleep Talk.
        s = make_state(
            t,
            [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Ember"])],
            [_mon(n, "Gastly", ["Imprison", "Tackle"], level=40)],
        ).with_status(0, 0, Status.SLEEP, 2)
    else:
        level = 30 if case in ("copycat_tie", "copycat_switch") else 50
        foe = [_mon(n, "Snorlax", ["Tackle"], level=level)]
        if case == "copycat_switch":
            foe.append(_mon(n, "Geodude", ["Tackle"]))
        s = make_state(t, [_mon(n, "Snorlax", ["Copycat"])], foe)
        s = s._replace(move_prev=jnp.asarray(n.move("Ember"), jnp.int16))
    acts = jnp.asarray([0, MOVE_ACTIONS + 1 if case == "copycat_switch" else 0], jnp.int32)
    outcomes, probs = enumerate_turn(t, s, acts)
    assert math.isclose(math.fsum(probs), 1.0)
    exact = _children(t, s, acts, outcomes, probs)
    brute = _raw_brute(t, s, acts)
    if case == "sleep_talk_disable":  # both sets are played
        assert np.any(np.asarray(outcomes.call)[:, 0] >= CALL_DISABLED)
    if case == "sleep_talk_imprison":  # the sealed set is played: no Tackle from it
        assert np.any(np.asarray(outcomes.call)[:, 0] >= CALL_SEALED)
    calls = np.asarray(outcomes.call)[:, 0]
    if case == "sleep_talk_gravity":  # the Gravity set is played, the turn-start one dropped
        assert np.any(calls >= CALL_GRAVITY)
        assert not np.any((calls >= CALL_KEEP) & (calls < CALL_DISABLED))
    if case == "assist_gravity":  # only the list Gravity leaves is played
        assert np.all(calls >= CALL_GRAVITY)
    if case == "assist_gravity_slow":  # the foe's Gravity comes after: the turn-start list
        assert np.all(calls < CALL_GRAVITY)
    assert set(exact) == set(brute)
    for key, p in brute.items():
        assert abs(exact[key] - p) < 1e-12


def test_metronome_after_a_faster_gravity_calls_from_the_set_it_leaves(tables, names):
    """G-67: ``Move_CanBeMetronomed`` (battle_lib.c:7198-7201) reads Gravity when Metronome
    runs, so a faster foe's Gravity this turn leaves Metronome the set without ``GRAVITY_MOVES``:
    the Gravity variant's law is ``metronome_weights`` under Gravity (total weight 1 beside the
    turn-start law's), and its value ``v + CALL_GRAVITY`` plays that set's ``v``-th candidate,
    as the reference engine does; a turn whose Gravity did not land first drops it."""
    t, n = tables, names
    s = make_state(
        t, [_mon(n, "Clefairy", ["Metronome"])], [_mon(n, "Gastly", ["Gravity"], level=40)]
    )
    acts = jnp.zeros(2, jnp.int32)
    cl = call_laws(t, s, acts)
    heavy = gravity_state(s)
    assert bool(cl.can_gravity[0]) and not bool(cl.can_gravity[1])
    want = np.asarray(metronome_weights(t, heavy, 0))
    got = np.asarray(cl.gravity_weight[0])
    assert got[got > 0].tolist() == want[want > 0].tolist()
    assert int(np.sum(want > 0)) < int(np.sum(np.asarray(metronome_weights(t, s, 0)) > 0))
    sides = [v for v in call_variants(cl) if v.call[0] >= CALL_GRAVITY]
    assert math.isclose(sum(Fraction(v.num, v.den) for v in sides), 1)
    cands = np.flatnonzero(np.asarray(metronome_candidates(t, heavy, 0)))
    fly = n.move("Fly")
    assert fly not in cands and bool(metronome_candidates(t, s, 0)[fly])
    rank = int(np.searchsorted(cands, n.move("Tackle")))
    _, events = _both(t, s, np.zeros(2, np.int32), _call(rank + CALL_GRAVITY))
    assert int(events["called"][0]) == n.move("Tackle")
    assert bool(gravity_events(s, events)[0])


def test_sampled_calls_match_the_enumeration(tables, names):
    """``sample_outcome``'s call draw (Sleep Talk) against ``enumerate_turn``: the children's
    frequencies within 5 sigma."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Snorlax", ["Sleep Talk", "Tackle", "Spore", "Ember"])],
        [_mon(n, "Geodude", ["Tackle"])],
    ).with_status(0, 0, Status.SLEEP, 3)
    acts = jnp.zeros(2, jnp.int32)
    outcomes, probs = enumerate_turn(t, s, acts)
    expected = _children(t, s, acts, outcomes, probs)
    count = 4000
    keys = jax.random.split(jax.random.key(5), count)
    sampled = sample_rows(keys, t, s, acts)
    got = Counter(_children(t, s, acts, sampled, np.ones(count)))
    assert set(got) <= set(expected)
    for key, p in expected.items():
        k = got.get(key, 0)
        assert abs(k / count - p) <= 5 * math.sqrt(p * (1 - p) / count) + 1e-9


def test_enumerate_refuses_uncalled_laws(tables, names):
    t, n = tables, names
    s = make_state(t, [_mon(n, "Snorlax", ["Copycat"])], [_mon(n, "Geodude", ["Tackle"])])
    laws = outcome_laws(t, s, jnp.zeros(2, jnp.int32))
    with pytest.raises(ValueError, match="enumerate_turn"):
        enumerate_outcomes(laws)


def test_coverage_refuses_unmodelled_metronome(tables, names):
    """B5-2: Metronome while it can call an unmodelled move (one made so, ``unmodelled``, so
    the test holds whatever the engine models); Sleep Talk is supported (B5-13: a faster
    Disable is modelled, not refused)."""
    n = names
    talker = _mon(n, "Snorlax", ["Sleep Talk", "Tackle"])
    spec = _mon(n, "Clefairy", ["Metronome"])
    t = unmodelled(tables, metronome_calls(tables, spec)[0])
    from nuzline.engine.coverage import _check_party, _Collector

    c = _Collector()
    _check_party(c, t, names, "Player", [spec])
    assert any("Metronome" in line for line in c.report.missing)
    c = _Collector()
    _check_party(c, tables, names, "Player", [talker])
    assert not any("Sleep Talk" in line for line in c.report.missing)


@pytest.mark.slow
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_random_invoker_turns_equal_the_raw_call_brute_force(tables, names, data):
    """Random states with Sleep Talk or Copycat (``strategies._draw_batch5``): ``enumerate_turn``
    against the brute force over every raw call value with the loose laws (Metronome's 442
    values are left to the hand cases)."""
    s = data.draw(supported_state(tables, names))
    effect = np.asarray(tables.move_effect)
    # Force an invoker into one active Pokemon's slot 0 (full PP), a Sleep Talk user asleep, and
    # make that slot its action.
    side = data.draw(st.integers(0, 1))
    name = data.draw(st.sampled_from(("Sleep Talk", "Copycat")))
    mon = int(s.active[side])
    assume(int(s.hp[side, mon]) > 0 and int(s.locked_move[side]) == 0)
    move = names.move(name)
    s = s._replace(
        moves=s.moves.at[side, mon, 0].set(move),
        pp=s.pp.at[side, mon, 0].set(int(tables.move_pp[move])),
        taunt=s.taunt.at[side].set(0),
        disabled_move=s.disabled_move.at[side].set(0),
        choice_lock=s.choice_lock.at[side].set(0),
    )
    if name == "Sleep Talk":
        s = s.with_status(side, mon, Status.SLEEP, data.draw(st.integers(1, 3)))
    acts = np.array(data.draw(legal_action_pair(tables, s)))  # a writable copy
    acts[side] = 0
    acts = jnp.asarray(acts, jnp.int32)
    kinds = []
    for sd in (0, 1):
        m = int(np.asarray(s.moves)[sd, int(s.active[sd]), min(int(acts[sd]), 3)])
        kinds.append(int(effect[m]) if int(acts[sd]) < 4 else -1)
    assume(EffectId.METRONOME not in kinds)
    try:
        outcomes, probs = enumerate_turn(tables, s, acts)
        brute = _raw_brute(tables, s, acts)
    except SamplingOnly:
        return
    exact = _children(tables, s, acts, outcomes, probs)
    assert set(exact) == set(brute)
    for key, p in brute.items():
        assert abs(exact[key] - p) < 1e-9
