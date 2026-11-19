"""E2 batch 11 (rulings B11-n, docs/kaizo/e2-mechanics-ledger.md): a chosen move replaced
mid-turn (the ledger's "Design: a chosen move replaced mid-turn"): Encore's takeover in the laws
and the sampler, the Sleep Talk Disable / Imprison sampler fix, Encore into the invokers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.actions import MOVE_ACTIONS, legal_actions
from nuzline.engine.enumerate import (
    _split_events,
    call_laws,
    call_variants,
    enumerate_turn,
    taken_events,
)
from nuzline.engine.ids import FieldCondition, Stat, Status
from nuzline.engine.order import (
    CALL_ASSIST,
    CALL_COPYCAT,
    CALL_METRONOME,
    CALL_SLEEP_TALK,
    call_kind,
    metronome_weights,
    selected_move,
    takeover_possible,
)
from nuzline.engine.outcome import (
    empty_outcome,
    pivot_view,
    replacement_known,
    replacement_possible,
    sample_outcome,
)
from nuzline.engine.state import make_state
from nuzline.engine.step import turn_events
from tests.engine.helpers import BATCH, assert_matches_reference, sample_rows, step_rows
from tests.engine.test_pivot import _mon
from tests.strategies import _teach, any_outcome, legal_action_pair, supported_state

CHUNK = BATCH  # the suite's one padded row count (tests.engine.helpers): shared programs


def _stepped(tables, state, actions, outcomes):
    """``step`` of every row of ``outcomes`` (numpy), in the suite's shared ``BATCH``-row
    program (``step_rows``)."""
    out = step_rows(tables, state, jnp.asarray(actions, jnp.int32), outcomes)
    return jax.tree.map(np.asarray, out)


def _exact(tables, state, actions, pred) -> float:
    """The probability of ``pred`` (on the stepped states) under ``enumerate_turn``."""
    outcomes, probs = enumerate_turn(tables, state, jnp.asarray(actions, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    return float(np.sum(probs[np.asarray(pred(_stepped(tables, state, actions, outcomes)))]))


def _sampled(tables, state, actions, pred, n=8 * CHUNK, seed=11) -> float:
    """The frequency of ``pred`` over ``n`` turns drawn by ``sample_outcome``."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    o = sample_rows(keys, tables, state, jnp.asarray(actions, jnp.int32))
    return int(np.sum(np.asarray(pred(_stepped(tables, state, actions, o))))) / n


# Five standard deviations of a frequency of 8192 draws near p = 0.75 (0.0048 each).
TOL = 0.025


def _sleep_talk_split(tables, names, foe_moves):
    """A sleeping Snorlax whose Sleep Talk calls Ember (accuracy 100) or Thunder (70), its last
    move Ember, against a faster foe using ``foe_moves[0]`` (Disable on Ember, or an Imprison
    sealing Ember)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Snorlax", ["Sleep Talk", "Ember", "Thunder"], 30)],
        [_mon(n, "Pikachu", foe_moves, 50)],
    ).with_status(0, 0, Status.SLEEP, 5)
    return s._replace(last_move=s.last_move.at[0].set(n.move("Ember")))


def _foe_hit(s):
    return lambda out: out.hp[:, 1, 0] < int(s.hp[1, 0])


def test_sleep_talk_after_a_faster_disable_is_sampled_at_the_called_moves_rates(tables, names):
    """B11-2: a faster Disable (accuracy 80) on Ember leaves Sleep Talk only Thunder, so the
    called move hits 0.8 * 0.7 + 0.2 * (0.5 + 0.5 * 0.7) = 0.73 of the time. ``sample_outcome``
    must draw the hit for the move the turn calls (the Disable-shrunk set), not the turn-start
    prediction (which gave 0.85)."""
    s = _sleep_talk_split(tables, names, ["Disable"])
    acts = (0, 0)
    exact = _exact(tables, s, acts, _foe_hit(s))
    assert abs(exact - 0.73) < 1e-9
    assert abs(_sampled(tables, s, acts, _foe_hit(s)) - exact) < TOL


def test_sleep_talk_after_a_faster_imprison_is_sampled_at_the_called_moves_rates(tables, names):
    """B11-2: a faster Imprison seals Ember (the foe knows it), so Sleep Talk calls Thunder:
    0.7, not the 0.85 of the turn-start set."""
    s = _sleep_talk_split(tables, names, ["Imprison", "Ember"])
    acts = (0, 0)
    exact = _exact(tables, s, acts, _foe_hit(s))
    assert abs(exact - 0.7) < 1e-9
    assert abs(_sampled(tables, s, acts, _foe_hit(s)) - exact) < TOL


def _encore_case(tables, names, chosen, last, extra=(), mates=()):
    """A faster Clefairy using Encore on Bidoof, who chose ``chosen`` and last used ``last``
    (Bidoof knows both, plus ``extra``; ``mates``: its party mates' movesets)."""
    n = names
    moves = [last, chosen, *extra]
    party = [_mon(n, "Bidoof", moves, 30)] + [_mon(n, "Geodude", m, 30) for m in mates]
    s = make_state(tables, [_mon(n, "Clefairy", ["Encore"], 40)], party)
    return s._replace(last_move=s.last_move.at[1].set(n.move(last))), (0, 1)


def test_a_faster_encore_takes_over_with_the_encored_moves_laws(tables, names):
    """B11-1, the hand case: Bidoof chose Growl (no draw), a faster Encore forces its last
    Tackle (accuracy 95, crit 1/16, 16 rolls), 3-7 turns. The children are Tackle's: the laws of
    the takeover variant (the encored move's hit, crit and roll), not Growl's single child per
    duration. Expected by hand from ``damage_for_roll``."""
    from nuzline.engine.damage import damage_for_roll

    s, acts = _encore_case(tables, names, "Growl", "Tackle")
    outcomes, probs = enumerate_turn(tables, s, jnp.asarray(acts, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    out = _stepped(tables, s, acts, outcomes)
    got: dict[tuple[int, int], float] = {}
    for hp, turns, p in zip(out.hp[:, 0, 0], out.encore_turns[:, 1], probs, strict=True):
        got[(int(hp), int(turns))] = got.get((int(hp), int(turns)), 0.0) + float(p)
    tackle = names.move("Tackle")
    start = int(s.hp[0, 0])
    want: dict[tuple[int, int], float] = {}
    for turns in range(2, 7):  # 3-7, then this turn's count-down
        want[(start, turns)] = want.get((start, turns), 0.0) + 0.05 / 5
        for crit in (False, True):
            for roll in range(16):
                dealt = int(damage_for_roll(tables, s, 1, tackle, crit, roll))
                key = (max(start - dealt, 0), turns)
                p = 0.95 * (1 / 16 if crit else 15 / 16) / 16 / 5
                want[key] = want.get(key, 0.0) + p
    assert set(got) == set(want)
    for key, p in want.items():
        assert abs(got[key] - p) < 1e-9, key
    assert np.all(out.stages[:, 0, Stat.ATK] == 0)  # never a Growl


def _asleep(side):
    return lambda out: out.status[:, side, 0] == Status.SLEEP


def test_a_faster_encore_into_hypnosis_is_sampled_at_hypnosis_rates(tables, names):
    """B11-1: Bidoof chose Ember (accuracy 100), a faster Encore forces Hypnosis (60): Clefairy
    sleeps 0.6 of the time, in the laws and in ``sample_outcome`` (the turn-start prediction
    gave 1.0: Ember's hit class)."""
    s, acts = _encore_case(tables, names, "Ember", "Hypnosis")
    exact = _exact(tables, s, acts, _asleep(0))
    assert abs(exact - 0.6) < 1e-9
    assert abs(_sampled(tables, s, acts, _asleep(0)) - exact) < TOL


_events_rows = jax.jit(jax.vmap(turn_events, in_axes=(None, None, None, 0)))


def _called(tables, state, actions, outcomes, side) -> np.ndarray:
    """The ``called`` event of ``side`` for every row of ``outcomes`` (``CHUNK`` batches)."""
    n = len(np.asarray(outcomes.speed_tie))
    out = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = jax.tree.map(lambda f, pad=pad: np.asarray(f)[pad], outcomes)
        ev = _events_rows(tables, state, jnp.asarray(actions, jnp.int32), rows)
        out.append(np.asarray(ev["called"])[: len(idx), side])
    return np.concatenate(out)


def _pp_spent(s, out, slot):
    return int(s.pp[1, 0, slot]) - np.asarray(out.pp[:, 1, 0, slot])


def test_encore_into_assist_calls_a_mates_move(tables, names):
    """B11-3: Encore keeps Assist (Move_CanBeEncored, battle_lib.c:7221-7231): a faster Encore on
    Bidoof's last Assist makes it Assist this turn instead of its Ember, calling Geodude's Thunder
    Wave or Rock Throw at 1/2 each (BtlCmd_TryAssist, battle_script.c:6418-6439), from the
    takeover variant's call law; Ember's PP is never spent."""
    s, acts = _encore_case(tables, names, "Ember", "Assist", mates=[["Thunder Wave", "Rock Throw"]])
    cl = call_laws(tables, s, jnp.asarray(acts, jnp.int32))
    assert bool(cl.can_take[1]) and int(cl.taken_kind[1]) == CALL_ASSIST
    outcomes, probs = enumerate_turn(tables, s, jnp.asarray(acts, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    called = _called(tables, s, acts, outcomes, 1)
    for move in ("Thunder Wave", "Rock Throw"):
        assert abs(float(np.sum(probs[called == names.move(move)])) - 0.5) < 1e-9
    out = _stepped(tables, s, acts, outcomes)
    assert np.all(_pp_spent(s, out, 0) == 1) and np.all(_pp_spent(s, out, 1) == 0)
    o = empty_outcome()._replace(call=jnp.asarray([0, 1], jnp.int16))
    assert_matches_reference(tables, s, acts, o)


def test_encore_into_metronome_takes_the_metronome_law(tables, names):
    """B11-3: Encore keeps Metronome: the takeover variants of Bidoof (chose Ember, last used
    Metronome) are Metronome's candidates at their weights (BtlCmd_Metronome, battle_script.c:
    4535-4562; order.metronome_weights), beside the variant where the Encore fails; a call of the
    takeover variant runs in both engines."""
    s, acts = _encore_case(tables, names, "Ember", "Metronome")
    cl = call_laws(tables, s, jnp.asarray(acts, jnp.int32))
    assert bool(cl.can_take[1]) and int(cl.taken_kind[1]) == CALL_METRONOME
    variants = call_variants(cl)
    taken = [v for v in variants if bool(v.taken[1])]
    kept = [v for v in variants if not bool(v.taken[1])]
    assert math.isclose(sum(v.num / v.den for v in taken), 1.0)
    assert math.isclose(sum(v.num / v.den for v in kept), 1.0)
    weights = np.asarray(metronome_weights(tables, s, 1))
    assert len(taken) == int(np.count_nonzero(weights))
    for k in (0, 57):
        o = empty_outcome()._replace(call=jnp.asarray([0, k], jnp.int16))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1  # Metronome's PP
        assert int(out.pp[1, 0, 1]) == int(s.pp[1, 0, 1])  # not Ember's


def test_an_encored_sleep_talk_always_fails(tables, names):
    """B11-3: Encore keeps Sleep Talk, but an encored Sleep Talk has no slot to draw:
    BattleSystem_CheckInvalidMoves (battle_lib.c:2302-2304) flags every move but the encored one,
    and TrySleepTalk (battle_script.c:4916-4930) flags the invoker itself. A sleeping Bidoof that
    chose Ember is encored into Sleep Talk: it spends Sleep Talk's PP and does nothing."""
    s, acts = _encore_case(tables, names, "Ember", "Sleep Talk")
    s = s.with_status(1, 0, Status.SLEEP, 5)
    outcomes, probs = enumerate_turn(tables, s, jnp.asarray(acts, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    out = _stepped(tables, s, acts, outcomes)
    assert np.all(_pp_spent(s, out, 0) == 1) and np.all(_pp_spent(s, out, 1) == 0)
    assert np.all(out.hp[:, 0, 0] == int(s.hp[0, 0]))
    assert np.all(_called(tables, s, acts, outcomes, 1) == 0)
    assert_matches_reference(tables, s, acts, empty_outcome())


def test_encore_into_copycat_copies_the_encore_and_fails(tables, names):
    """B11-3: Encore keeps Copycat: encored into it, Bidoof copies the battle's movePrev, the
    Encore just used (BtlCmd_TryCopycat, battle_script.c:7328-7341), whose target Clefairy last
    used Encore, which Move_CanBeEncored refuses (BtlCmd_TryEncore fails): nobody else is
    encored, and Ember's PP is not spent."""
    s, acts = _encore_case(tables, names, "Ember", "Copycat")
    cl = call_laws(tables, s, jnp.asarray(acts, jnp.int32))
    assert bool(cl.can_take[1]) and int(cl.taken_kind[1]) == CALL_COPYCAT
    outcomes, probs = enumerate_turn(tables, s, jnp.asarray(acts, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    out = _stepped(tables, s, acts, outcomes)
    assert np.all(_pp_spent(s, out, 0) == 1) and np.all(_pp_spent(s, out, 1) == 0)
    assert np.all(_called(tables, s, acts, outcomes, 1) == names.move("Encore"))
    assert np.all(out.encore_move[:, 0] == 0)
    o = empty_outcome()._replace(call=jnp.asarray([0, 1], jnp.int16))
    assert_matches_reference(tables, s, acts, o)


def _cover(tables, state, actions, outcome):
    """``(fired, uncovered)`` ``bool[2]`` each: a replacement event that changes the side's move
    fired (taken over, or a Sleep Talk side's set changed by a faster Disable or Imprison), and
    one fired that the predicates did not allow (``order.takeover_possible`` for a takeover,
    ``outcome.replacement_possible`` for any), each read on the ``pivot_view`` as the laws and
    the sampler read them."""
    events = turn_events(tables, state, actions, outcome)
    disabled, sealed = _split_events(state, events)
    taken = taken_events(state, events)
    view, view_actions = pivot_view(tables, state, actions)
    fired, uncovered = [], []
    for side in (0, 1):
        foe = 1 - side
        chosen = selected_move(tables, view, side, view_actions[side])[0]
        talk = call_kind(tables, chosen) == CALL_SLEEP_TALK
        event = taken[side] | (talk & (disabled[side] | sealed[side]))
        can_take = takeover_possible(tables, view, view_actions, side, foe)
        can_replace = replacement_possible(tables, view, view_actions, side, foe)
        fired.append(event)
        uncovered.append((taken[side] & ~can_take) | (event & ~can_replace))
    return jnp.stack(fired), jnp.stack(uncovered)


_covers = jax.jit(_cover)
_FIRED = [0]  # replacement events met (a coverage aid for runs)


def _steer(data, tables, names, s):
    """Three draws in four, ``s`` and the actions steered at a replacement (the generator alone
    met none in 300 draws): a fast foe (Speed 999) that knows Encore, Disable or Imprison (full
    PP) and uses it; the target's last move one it knows, nobody encored, disabled or
    imprisoning, no Trick Room, no lock, the foe untaunted; for Disable and Imprison the target
    asleep using Sleep Talk, and for Imprison the foe knowing the target's last move. Else
    ``None``."""
    side = data.draw(st.integers(0, 1))
    foe = 1 - side
    mon, foe_mon = int(s.active[side]), int(s.active[foe])
    live = int(s.hp[side, mon]) > 0 and int(s.hp[foe, foe_mon]) > 0
    if data.draw(st.integers(0, 3)) == 0 or not live or int(s.pivot_battler) or int(s.over):
        return None
    a = {name: np.asarray(getattr(s, name)).copy() for name in type(s)._fields}
    kind = data.draw(st.sampled_from(["Encore", "Disable", "Imprison"]))
    k = _teach(tables, a, foe, foe_mon, names.move(kind), data.draw(st.integers(0, 3)))
    talk = names.move("Sleep Talk")
    if kind != "Encore":
        talk = _teach(tables, a, side, mon, talk, data.draw(st.integers(0, 3)))
        a["status"][side, mon] = Status.SLEEP
        a["status_counter"][side, mon] = data.draw(st.integers(2, 4))
    known = [
        int(m) for j, m in enumerate(a["moves"][side, mon]) if m and (kind == "Encore" or j != talk)
    ]
    if not known:
        return None
    last = data.draw(st.sampled_from(known))
    if kind == "Imprison":
        _teach(
            tables,
            a,
            foe,
            foe_mon,
            last,
            data.draw(st.sampled_from([j for j in range(4) if j != k])),
        )
    a["last_move"][side] = last
    a["stats"][foe, foe_mon, Stat.SPE - Stat.ATK] = 999
    a["encore_move"][:] = a["encore_slot"][:] = a["encore_turns"][:] = 0
    a["disabled_move"][side] = a["disable_turns"][side] = 0
    a["imprison"][:] = 0
    a["taunt"][foe] = 0
    a["locked_move"][:] = a["choice_lock"][:] = 0
    a["field_mask"][...] &= ~int(FieldCondition.TRICK_ROOM)  # no Trick Room
    s = type(s)(**{name: jnp.asarray(a[name]) for name in type(s)._fields})
    foe_slot = int(np.flatnonzero(np.asarray(s.moves[foe, foe_mon]) == names.move(kind))[0])
    legal = [np.asarray(legal_actions(tables, s, sd))[:MOVE_ACTIONS] for sd in (0, 1)]
    if kind != "Encore":
        mine = [talk] if legal[side][talk] else []
    else:
        mine = np.flatnonzero(legal[side]).tolist()
    if not legal[foe][foe_slot] or not mine:
        return None
    acts = [0, 0]
    acts[side], acts[foe] = data.draw(st.sampled_from(mine)), foe_slot
    return s, jnp.asarray(acts, jnp.int32)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_the_replacement_predicates_cover_every_event(tables, names, data):
    """B11-10: the sampler redraws a side's move fields only where ``replacement_possible`` holds,
    and the laws split a takeover only where ``takeover_possible`` holds (``call_consistent``):
    both must hold wherever the event fires (``step_events``' ``encored_at_move``, and
    ``disabled_at_move`` / ``sealed_at_move`` for a Sleep Talk side), for any outcome, on
    generated states, three in four steered at a replacement (``_steer``). Also: no generated
    Pokemon knows a move twice (B11-5)."""
    s = data.draw(supported_state(tables, names))
    moves = np.asarray(s.moves)
    for row in moves.reshape(-1, moves.shape[-1]):
        known = row[row != 0]
        assert len(set(known.tolist())) == len(known), row
    steered = _steer(data, tables, names, s)
    if steered is None:
        acts = jnp.asarray(data.draw(legal_action_pair(tables, s)), jnp.int32)
    else:
        s, acts = steered
    fired, uncovered = _covers(tables, s, acts, data.draw(any_outcome()))
    _FIRED[0] += int(np.sum(np.asarray(fired)))
    assert not np.any(np.asarray(uncovered)), (np.asarray(acts), np.asarray(fired))


_possible = jax.jit(
    lambda t, s, a: jnp.any(
        jnp.stack([replacement_possible(t, s, a, side, 1 - side) for side in (0, 1)])
    )
)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_replacement_known_holds_wherever_a_replacement_is_possible(tables, names, data):
    """The sampler's static flag (``outcome.replacement_known``, B11-1's cost cut): wherever
    ``replacement_possible`` holds for a side on a generated state (three in four steered at a
    replacement, ``_steer``), the flag is True, so a False flag never drops a possible
    redraw."""
    s = data.draw(supported_state(tables, names))
    steered = _steer(data, tables, names, s)
    if steered is None:
        acts = jnp.asarray(data.draw(legal_action_pair(tables, s)), jnp.int32)
    else:
        s, acts = steered
    if bool(_possible(tables, s, acts)):
        assert replacement_known(tables, s), np.asarray(s.moves)


def test_replacement_known_counts_a_move_learnt_at_a_later_level(tables, names):
    """``outcome.replacement_known``'s level-up half: Glameow learns Assist (an invoker) at level
    29 (its ``tables.learnset_*`` row), so a level-28 Glameow that does not know it learns it at
    its next level in a playout (``levelup.advance_learns``): the flag is True. At level 30 that
    entry is behind it, and neither party knows or can still learn Encore or an invoker (Bidoof's
    learnset has none): False."""
    n = names
    glameow = n.species("Glameow")
    learnt = zip(
        np.asarray(tables.learnset_move[glameow]).tolist(),
        np.asarray(tables.learnset_level[glameow]).tolist(),
        strict=True,
    )
    assert [level for move, level in learnt if move == n.move("Assist")] == [29]

    def state(level):
        return make_state(
            tables, [_mon(n, "Glameow", ["Scratch"], level)], [_mon(n, "Bidoof", ["Tackle"], 30)]
        )

    assert replacement_known(tables, state(28))
    assert not replacement_known(tables, state(30))


_sample_flag = jax.jit(sample_outcome, static_argnames=("replacement",))


def _through(tables, state, actions, first_key, key, replacement):
    from nuzline.search.playout import step_through

    return step_through(tables, state, actions, first_key, key, -1, replacement)


_through_flag = jax.jit(_through, static_argnames=("replacement",))


def _same(a, b):
    for got, want in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.slow  # 444 s, 7.28 GB: four step_through / sample_outcome compiles
def test_replacement_flag_true_matches_false_where_none_can_fire(tables, names):
    """The sampler's static flag on its True path (``outcome.replacement_known``; B11-1's
    redraw compiled in, as the leaf playouts and Stage A run it): Monferno knows Assist (an
    invoker), so the flag is True; with no party mate Assist calls nothing, and neither side
    knows or can learn Encore or Sleep Talk, so no replacement can fire and the redraw (a
    ``lax.cond`` that leaves outcome A where ``replacement_possible`` is false) changes nothing.
    Over three turns (Assist, Ember, Assist against Tackle), ``sample_outcome`` and
    ``playout.step_through`` give identical outputs with the flag True and False, key for key."""
    from tests.engine.helpers import roark_state

    s = roark_state(tables, names, player_count=1, opp_count=1, player_moves=["Assist"])
    assert replacement_known(tables, s)
    keys = jax.random.split(jax.random.PRNGKey(5), 3 * 4 * 2).reshape(3, 4, 2, -1)
    for turn, move in enumerate(("Assist", "Ember", "Assist")):
        if int(s.over):
            break
        acts = jnp.asarray(
            [names.move_slot(s, 0, move), names.move_slot(s, 1, "Tackle")], jnp.int32
        )
        for first_key, key in keys[turn]:
            on = _sample_flag(first_key, tables, s, acts, replacement=True)
            _same(on, _sample_flag(first_key, tables, s, acts, replacement=False))
            on = _through_flag(tables, s, acts, first_key, key, replacement=True)
            _same(on, _through_flag(tables, s, acts, first_key, key, replacement=False))
        s, refused = on
        assert not bool(refused)
