"""E2 batch 11 (rulings B11-n, docs/kaizo/e2-mechanics-ledger.md): a chosen move replaced
mid-turn (the ledger's "Design: a chosen move replaced mid-turn"): Encore's takeover in the laws
and the sampler, the Sleep Talk Disable / Imprison sampler fix, Encore into the invokers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.engine.enumerate import call_laws, call_variants, enumerate_turn
from nuzline.engine.ids import Stat, Status
from nuzline.engine.order import CALL_ASSIST, CALL_COPYCAT, CALL_METRONOME, metronome_weights
from nuzline.engine.outcome import empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step, turn_events
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

CHUNK = 256  # one compiled program per kind for every batch in this file (small: under 8 GB)
_steps = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_samples = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))


def _stepped(tables, state, actions, outcomes):
    """``step`` of every row of ``outcomes`` (numpy), in ``CHUNK`` batches (padding = row 0)."""
    n = len(np.asarray(outcomes.speed_tie))
    parts = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = jax.tree.map(lambda f, pad=pad: np.asarray(f)[pad], outcomes)
        out = _steps(tables, state, jnp.asarray(actions, jnp.int32), rows)
        k = len(idx)
        parts.append(jax.tree.map(lambda f, k=k: np.asarray(f)[:k], out))
    return jax.tree.map(lambda *f: np.concatenate(f), *parts)


def _exact(tables, state, actions, pred) -> float:
    """The probability of ``pred`` (on the stepped states) under ``enumerate_turn``."""
    outcomes, probs = enumerate_turn(tables, state, jnp.asarray(actions, jnp.int32))
    assert math.isclose(math.fsum(probs), 1.0)
    return float(np.sum(probs[np.asarray(pred(_stepped(tables, state, actions, outcomes)))]))


def _sampled(tables, state, actions, pred, n=8 * CHUNK, seed=11) -> float:
    """The frequency of ``pred`` over ``n`` turns drawn by ``sample_outcome``."""
    hits = 0
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    for start in range(0, n, CHUNK):
        o = _samples(keys[start : start + CHUNK], tables, state, jnp.asarray(actions, jnp.int32))
        hits += int(np.sum(np.asarray(pred(_stepped(tables, state, actions, o)))))
    return hits / n


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
