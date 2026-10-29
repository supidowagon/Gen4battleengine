"""The AI's random draws as explicit outcomes (docs/ai/roark.md section 8)."""

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai.outcome import (
    MAX_SCRIPT_ROLLS,
    AiOutcome,
    enumerate_ai_outcomes,
    sample_ai_outcome,
    script_roll_thresholds,
)
from tests.engine.helpers import roark_state

N = 20_000


def _with_opp_moves(state, names, moves):
    ids = [names.move(m) for m in moves] + [0] * (4 - len(moves))
    return state._replace(moves=state.moves.at[1, 0].set(jnp.asarray(ids, jnp.int16)))


def test_sample_within_ranges(tables, names):
    s = roark_state(tables, names)
    o = sample_ai_outcome(jax.random.PRNGKey(0), s)
    assert o.sim_roll.shape == (4,) and o.script_rolls.shape == (MAX_SCRIPT_ROLLS,)
    assert int(o.ability_guess) in (0, 1)
    assert o.sim_roll.tolist() == [15] * 4  # roll index 15 = 100 %, roark.md 8.1
    assert set(o.script_rolls.tolist()) <= {0, 1} and 0 <= int(o.tiebreak) <= 11
    assert all(getattr(o, f).dtype == jnp.int8 for f in AiOutcome._fields)


def test_enumeration_probabilities_sum_to_one(tables, names):
    s = roark_state(tables, names)
    outs, probs = enumerate_ai_outcomes(s)
    assert abs(float(probs.sum()) - 1.0) < 1e-5 and outs.sim_roll.shape[0] == probs.shape[0]


def test_script_roll_thresholds_match_the_site_table(tables, names):
    """roark.md 8.3: sites in script order, EVAL_ATTACK slots 0-3 then EXPERT."""
    s = roark_state(tables, names)
    geodude = ["Stealth Rock", "Rock Throw"]
    onix = ["Stealth Rock", "Rock Throw", "Screech"]
    cranidos = ["Headbutt", "Pursuit", "Leer"]
    assert script_roll_thresholds(_with_opp_moves(s, names, geodude)).tolist() == [80, 80, 128, 0]
    assert script_roll_thresholds(_with_opp_moves(s, names, onix)).tolist() == [80, 80, 128, 50]
    assert script_roll_thresholds(_with_opp_moves(s, names, cranidos)).tolist() == [
        80,
        128,
        128,
        50,
    ]
    # The test fixture's moveset: Tackle, Rock Throw, Stealth Rock, Leer.
    assert script_roll_thresholds(s).tolist() == [80, 80, 128, 50]


def test_enumeration_probabilities_are_exact(tables, names):
    s = roark_state(tables, names)
    outs, probs = enumerate_ai_outcomes(s)
    probs = np.asarray(probs, np.float64)
    tb = np.asarray(outs.tiebreak)
    # RandNext() % 12 over a u16: residues 0-3 have 5462/65536, 4-11 have 5461/65536.
    for v in range(12):
        assert abs(probs[tb == v].sum() - (5462 if v < 4 else 5461) / 65536) < 1e-6
    rolls = np.asarray(outs.script_rolls)
    for k, n in enumerate((80, 80, 128, 50)):
        assert abs(probs[rolls[:, k] == 1].sum() - n / 256) < 1e-6
    assert abs(probs[np.asarray(outs.ability_guess) == 0].sum() - 0.5) < 1e-6


def test_unused_site_never_rolls(tables, names):
    s = _with_opp_moves(roark_state(tables, names), names, ["Stealth Rock", "Rock Throw"])
    outs, probs = enumerate_ai_outcomes(s)
    assert float(probs[np.asarray(outs.script_rolls)[:, 3] == 1].sum()) == 0.0
    o = jax.vmap(lambda k: sample_ai_outcome(k, s))(jax.random.split(jax.random.PRNGKey(1), 500))
    assert int(o.script_rolls[:, 3].max()) == 0


def test_sampled_frequencies_match_enumeration(tables, names):
    s = roark_state(tables, names)
    keys = jax.random.split(jax.random.PRNGKey(42), N)
    samples = jax.jit(jax.vmap(sample_ai_outcome, in_axes=(0, None)))(keys, s)
    outs, probs = enumerate_ai_outcomes(s)
    probs = np.asarray(probs, np.float64)
    for field in AiOutcome._fields:
        sampled = np.asarray(getattr(samples, field)).reshape(N, -1)
        listed = np.asarray(getattr(outs, field)).reshape(probs.shape[0], -1)
        for i in range(sampled.shape[1]):
            for v in np.union1d(sampled[:, i], listed[:, i]):
                freq = float((sampled[:, i] == v).mean())
                exact = float(probs[listed[:, i] == v].sum())
                assert abs(freq - exact) < 0.02, (field, i, int(v), freq, exact)


def test_sampler_and_enumerator_jit_without_64bit(tables, names):
    s = roark_state(tables, names)
    for fn, args in (
        (sample_ai_outcome, (jax.random.PRNGKey(0), s)),
        (enumerate_ai_outcomes, (s,)),
    ):
        jaxpr = str(jax.make_jaxpr(fn)(*args))
        assert "int64" not in jaxpr and "float64" not in jaxpr
    outs, probs = jax.jit(enumerate_ai_outcomes)(s)
    assert outs.tiebreak.shape == probs.shape
