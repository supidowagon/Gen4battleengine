"""The AI's random draws as explicit outcomes (docs/ai/roark.md section 8)."""

import jax
import jax.numpy as jnp
import numpy as np

import nuzline.ai.outcome as outcome_module
from nuzline.ai.outcome import (
    ENUMERATION_WIDTH,
    MAX_ENUMERATED,
    MAX_SCRIPT_ROLLS,
    SWITCH_NONE,
    SWITCH_SITES,
    AiOutcome,
    SwitchSite,
    enumerate_ai_outcomes,
    sample_ai_outcome,
    script_roll_thresholds,
    switch_stop_probs,
)
from nuzline.ai.switching import switch_sites
from tests.engine.helpers import roark_state

N = 20_000
NO_SITES = [SwitchSite.NONE] * SWITCH_SITES
# The layout of _every_kind's state (roark.md 11.1): one keep draw, then two
# party-pass-1 and two party-pass-2 draws.
LAYOUT = jnp.asarray(
    [SwitchSite.KEEP, SwitchSite.PASS_INEFFECTIVE, SwitchSite.PASS_INEFFECTIVE]
    + [SwitchSite.PASS_NOT_VERY_EFFECTIVE] * 2
    + [SwitchSite.NONE] * 4,
    jnp.int8,
)
# The decomp's stop probability per kind over a uniform u16 (trainer_ai.c:3588, :3796).
P_KEEP, P_PASS_1, P_PASS_2 = 58982 / 65536, 32768 / 65536, 21846 / 65536


def _every_kind(tables, names):
    """Geodude (fixture moves) vs Starly, hit by Tackle, with Cranidos made Ghost:
    Rock Throw's keep draw, Cranidos's pass-1 and Onix's pass-2 draws (LAYOUT)."""
    s = roark_state(tables, names, player_species="Starly")
    ghost = names.type("Ghost")
    s = s._replace(
        move_hit=s.move_hit.at[1].set(names.move("Tackle")),
        types=s.types.at[1, 2].set(jnp.asarray([ghost, ghost], jnp.int8)),
    )
    assert switch_sites(tables, s).tolist() == LAYOUT.tolist()
    return s


def _with_opp_moves(state, names, moves):
    ids = [names.move(m) for m in moves] + [0] * (4 - len(moves))
    return state._replace(moves=state.moves.at[1, 0].set(jnp.asarray(ids, jnp.int16)))


def test_sample_within_ranges(tables, names):
    s = roark_state(tables, names)
    o = sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    assert o.sim_roll.shape == (4,) and o.script_rolls.shape == (MAX_SCRIPT_ROLLS,)
    assert int(o.ability_guess) in (0, 1)
    assert o.sim_roll.tolist() == [15] * 4  # roll index 15 = 100 %, roark.md 8.1
    assert set(o.script_rolls.tolist()) <= {0, 1} and 0 <= int(o.tiebreak) <= 11
    assert all(getattr(o, f).dtype == jnp.int8 for f in AiOutcome._fields if f != "calc_draws")
    assert o.calc_draws.shape == (4, 0, 0)  # a flat outcome draws no raw damage draws


def test_enumeration_probabilities_sum_to_one(tables, names):
    s = _every_kind(tables, names)
    outs, probs = enumerate_ai_outcomes(tables, s)
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
    outs, probs = enumerate_ai_outcomes(tables, s)
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
    outs, probs = enumerate_ai_outcomes(tables, s)
    assert float(probs[np.asarray(outs.script_rolls)[:, 3] == 1].sum()) == 0.0
    o = jax.vmap(lambda k: sample_ai_outcome(k, tables, s))(
        jax.random.split(jax.random.PRNGKey(1), 500)
    )
    assert int(o.script_rolls[:, 3].max()) == 0


def test_sampled_frequencies_match_enumeration(tables, names):
    s = _every_kind(tables, names)
    keys = jax.random.split(jax.random.PRNGKey(42), N)
    samples = jax.jit(jax.vmap(sample_ai_outcome, in_axes=(0, None, None)))(keys, tables, s)
    outs, probs = enumerate_ai_outcomes(tables, s)
    probs = np.asarray(probs, np.float64)
    for field in AiOutcome._fields:
        sampled = np.asarray(getattr(samples, field))
        sampled = sampled.reshape(N, sampled[0].size)  # calc_draws: empty
        listed = np.asarray(getattr(outs, field))
        listed = listed.reshape(probs.shape[0], listed[0].size)
        for i in range(sampled.shape[1]):
            for v in np.union1d(sampled[:, i], listed[:, i]):
                freq = float((sampled[:, i] == v).mean())
                exact = float(probs[listed[:, i] == v].sum())
                assert abs(freq - exact) < 0.02, (field, i, int(v), freq, exact)


def test_sampler_and_enumerator_jit_without_64bit(tables, names):
    s = _every_kind(tables, names)
    for fn, args in (
        (sample_ai_outcome, (jax.random.PRNGKey(0), tables, s)),
        (enumerate_ai_outcomes, (tables, s)),
    ):
        jaxpr = str(jax.make_jaxpr(fn)(*args))
        assert "int64" not in jaxpr and "float64" not in jaxpr
    outs, probs = jax.jit(enumerate_ai_outcomes)(tables, s)
    assert outs.tiebreak.shape == probs.shape


def test_switch_stop_range(tables, names):
    s = _every_kind(tables, names)
    keys = jax.random.split(jax.random.PRNGKey(3), 500)
    sample = jax.jit(jax.vmap(sample_ai_outcome, in_axes=(0, None, None)))
    stops = np.asarray(sample(keys, tables, s).switch_stop)
    assert stops.dtype == np.int8 and stops.min() >= 0 and stops.max() <= SWITCH_NONE == 9
    # Only sites that exist can stop: here 0-4, or none (9).
    assert set(stops.tolist()) <= {0, 1, 2, 3, 4, SWITCH_NONE}
    # No sites (Geodude vs Monferno, no moveHit): no draw can stop.
    s = roark_state(tables, names)
    assert switch_sites(tables, s).tolist() == NO_SITES
    assert set(np.asarray(sample(keys, tables, s).switch_stop).tolist()) == {SWITCH_NONE}
    outs, probs = enumerate_ai_outcomes(tables, s)
    stops = np.asarray(outs.switch_stop)
    assert abs(float(np.asarray(probs)[stops == SWITCH_NONE].sum()) - 1.0) < 1e-6


def test_switch_stop_probabilities_are_exact(tables, names):
    # The first site to stop: prod_{h<i} (1 - p_h) * p_i (roark.md 11.1, "Encoding").
    p = [P_KEEP, P_PASS_1, P_PASS_1, P_PASS_2, P_PASS_2]
    expected, keep_going = [], 1.0
    for p_i in p:
        expected.append(keep_going * p_i)
        keep_going *= 1 - p_i
    expected += [0.0] * (SWITCH_SITES - len(p)) + [keep_going]
    np.testing.assert_allclose(np.asarray(switch_stop_probs(LAYOUT)), expected, atol=1e-7)
    s = _every_kind(tables, names)
    outs, probs = enumerate_ai_outcomes(tables, s)
    probs, stops = np.asarray(probs, np.float64), np.asarray(outs.switch_stop)
    for v in range(SWITCH_NONE + 1):
        assert abs(probs[stops == v].sum() - expected[v]) < 1e-6, v


def test_enumeration_width_is_fixed_and_within_the_limit(tables, names):
    # 2 ability guesses x 2^4 script rolls x 12 tie-breaks x 10 switch stops: the
    # field ranges do not depend on the state, so the width is a module constant and
    # the spec's sampling-only fallback (section 4.2) is never needed for Roark.
    assert ENUMERATION_WIDTH == 2 * 2**4 * 12 * 10 == 3840 <= MAX_ENUMERATED
    outs, probs = enumerate_ai_outcomes(tables, _every_kind(tables, names))
    assert probs.shape == (ENUMERATION_WIDTH,)
    assert outs.switch_stop.shape == (ENUMERATION_WIDTH,)
    assert not hasattr(outcome_module, "AiOutcomeTooWide")  # the runtime check could never fire
