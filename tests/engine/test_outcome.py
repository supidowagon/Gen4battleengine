"""Chance outcomes: the empty outcome, exact hit and crit odds, and the sampler's
distributions (decomp citations in ``outcome.py``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import AbilityId, Weather
from nuzline.engine.outcome import (
    INT16_FIELDS,
    MAX_HITS,
    Outcome,
    crit_odds,
    empty_outcome,
    hit_threshold,
    sample_outcome,
)
from nuzline.engine.step import replace
from tests.engine.helpers import roark_real_state, roark_state, sample_rows

SCRATCH, EMBER, TACKLE = 0, 2, 0  # Monferno's slots 0 and 2; every fixture opponent's slot 0
N = 20_000


def test_empty_outcome_is_all_hits_no_crits_max_roll():
    o = empty_outcome()
    assert o.speed_tie.shape == () and int(o.speed_tie) == 0
    for name in ("hit", "crit", "roll", "secondary", "contact_proc"):
        assert getattr(o, name).shape == (2, MAX_HITS)
    assert o.hit.tolist() == [[1] * MAX_HITS] * 2 and o.crit.tolist() == [[0] * MAX_HITS] * 2
    assert o.roll.tolist() == [[15] * MAX_HITS] * 2
    assert (
        o.secondary.tolist() == [[0] * MAX_HITS] * 2
        and o.contact_proc.tolist() == [[0] * MAX_HITS] * 2
    )
    # Every field int8 but INT16_FIELDS (the invokers' call, E2 batch 5; Acupressure's draw, B12-9).
    assert all(getattr(o, f).dtype == jnp.int8 for f in Outcome._fields if f not in INT16_FIELDS)
    assert o.call.dtype == jnp.int16 and o.call.tolist() == [0, 0]


def acts(*a):
    return jnp.array(a, jnp.int32)


def test_hit_threshold_accuracy_and_stages(tables, names):
    s = roark_state(tables, names)
    assert hit_threshold(tables, s, acts(SCRATCH, TACKLE)).tolist() == [100, 95]
    # HitRateByStage (include/data/hit_rate_stages.h): -1 -> 75/100, -2 -> 60/100.
    assert hit_threshold(tables, s.with_stages(0, acc=-1), acts(SCRATCH, TACKLE)).tolist()[0] == 75
    assert hit_threshold(tables, s.with_stages(1, eva=1), acts(SCRATCH, TACKLE)).tolist()[0] == 75
    simple = s._replace(ability=s.ability.at[0, 0].set(AbilityId.SIMPLE)).with_stages(0, acc=-1)
    assert hit_threshold(tables, simple, acts(SCRATCH, TACKLE)).tolist()[0] == 60
    # A target that switches out comes in with neutral evasion.
    assert hit_threshold(tables, s.with_stages(1, eva=1), acts(SCRATCH, 4 + 1)).tolist()[0] == 100


def test_hit_threshold_fog_gravity_and_never_miss(tables, names):
    s = roark_state(tables, names)
    fog = s._replace(weather=jnp.int8(Weather.FOG))
    assert hit_threshold(tables, fog, acts(SCRATCH, TACKLE)).tolist() == [60, 57]  # x 6 // 10
    grav = s._replace(gravity=jnp.int8(5))
    assert hit_threshold(tables, grav, acts(SCRATCH, TACKLE)).tolist() == [166, 158]  # x 10 // 6
    struggle = s.with_pp(0, 0, [0, 0, 0, 0])  # Struggle has accuracy 0: never misses
    assert (
        hit_threshold(tables, struggle.with_stages(0, acc=-6), acts(0, TACKLE)).tolist()[0] >= 100
    )


def test_crit_odds(tables, names):
    s = roark_state(tables, names)
    assert crit_odds(tables, s, acts(SCRATCH, TACKLE)).tolist() == [16, 16]
    grotle = replace(tables, s, 0, 2)  # Razor Leaf (HIGH_CRIT) is Grotle's slot 3
    assert crit_odds(tables, grotle, acts(3, TACKLE)).tolist() == [8, 16]


@pytest.fixture(scope="module")
def samples(tables, names):
    s = roark_state(tables, names)
    keys = jax.random.split(jax.random.key(0), N)
    sample = sample_rows
    return sample(keys, tables, s, acts(EMBER, TACKLE))


def freq(x, value=1):
    return float(np.mean(np.asarray(x) == value))


def test_sample_shapes_and_dtypes(samples):
    for f in Outcome._fields:
        assert getattr(samples, f).dtype == (jnp.int16 if f in INT16_FIELDS else jnp.int8), f
    assert samples.hit.shape == (N, 2, MAX_HITS) and samples.n_hits.shape == (N, 2)


@pytest.mark.parametrize(
    ("field", "index", "p"),
    [
        ("speed_tie", (), 1 / 2),
        ("hit", (0, 0), 1.0),  # Ember: 100 accuracy
        ("hit", (1, 0), 0.95),  # Tackle: 95
        ("crit", (0, 0), 1 / 16),
        ("secondary", (0, 0), 0.10),  # Ember's 10 % burn chance
        ("contact_proc", (1, 0), 0.30),  # % 10 < 3
        ("full_para", (0,), 0.25),
        ("confusion_hit", (1,), 0.5),
        ("thaw", (0,), 0.20),
        ("flinch_item", (0, 0), 0.0),  # neither lead holds a King's Rock
        ("quick_claw", (1,), 0.2),  # speedRand % 5 == 0 (battle_lib.c:1279), K3b2 Task 8
    ],
)
def test_sample_bernoulli_rates(samples, field, index, p):
    x = np.asarray(getattr(samples, field))[(slice(None), *index)]
    assert abs(freq(x) - p) < 4 * np.sqrt(p * (1 - p) / N) + 1e-9


def test_sample_rolls_durations_and_hit_counts(samples):
    rolls = np.asarray(samples.roll)
    assert rolls.min() == 0 and rolls.max() == 15 and len(np.unique(rolls)) == 16
    durations = np.asarray(samples.duration)
    assert sorted(np.unique(durations).tolist()) == [2, 3, 4, 5]
    hits = np.asarray(samples.n_hits)[:, 0]
    # BtlCmd_SetMultiHit (battle_script.c:2634): 2 and 3 at 3/8 each, 4 and 5 at 1/8 each.
    for n, p in ((2, 3 / 8), (3, 3 / 8), (4, 1 / 8), (5, 1 / 8)):
        assert abs(freq(hits, n) - p) < 4 * np.sqrt(p * (1 - p) / N)


@pytest.mark.slow
def test_lucky_chant_prevents_crits(tables, names):
    s = roark_state(tables, names)
    s = s._replace(lucky_chant=s.lucky_chant.at[1].set(5))
    keys = jax.random.split(jax.random.key(1), 2000)
    o = jax.vmap(sample_outcome, in_axes=(0, None, None, None))(keys, tables, s, acts(SCRATCH, 0))
    assert int(o.crit[:, 0].sum()) == 0 and int(o.crit[:, 1].sum()) > 0


# Recorded from ``sample_outcome(jax.random.key(0), ...)`` on ``roark_real_state`` with
# Ember (player) vs Rock Throw (Roark's Geodude), before Magnitude and Shed Skin existed.
EXISTING_DRAWS = {
    "speed_tie": 0,
    "hit": [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
    "crit": [[0, 0, 0, 0, 0], [0, 1, 0, 0, 1]],
    "roll": [[4, 1, 3, 15, 10], [3, 14, 14, 8, 8]],
    "secondary": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
    "contact_proc": [[1, 0, 0, 0, 1], [0, 0, 0, 0, 1]],
    "n_hits": [2, 2],
    "full_para": [0, 1],
    "confusion_hit": [0, 1],
    "duration": [5, 4],
    "thaw": [0, 0],
    "flinch_item": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    "end_speed_tie": 0,
}
# ``quick_claw`` was reserved (always 0) when these were recorded; K3b2 Task 8 made it a
# draw from its own key ``fold_in(key, 20)``, so it is checked by
# ``test_quick_claw_draw_has_its_own_key`` instead.


def test_existing_draws_unchanged(tables, names):
    s = roark_real_state(tables, names)
    assert names.move_names[int(s.moves[0, int(s.active[0]), EMBER])] == "Ember"
    o = sample_outcome(jax.random.key(0), tables, s, acts(EMBER, 1))
    for name, want in EXISTING_DRAWS.items():
        got = np.asarray(getattr(o, name))
        # B13-9: the sixth hit's column is a draw of its own (outcome.SIXTH_HIT_FOLD); hits 0-4
        # keep their samples.
        got = got[:, : len(want[0])] if got.ndim == 2 else got
        assert got.tolist() == want, name


def test_quick_claw_draw_has_its_own_key(tables, names):
    # Outcome fields from K3b2 on take fold_in(key, 16 + k) in order of landing: quick_claw is
    # the fifth (fang_flinch 16, present 17, protect 18, infatuation 19).
    # Recorded at K3b2 Task 8 (fold_in(key, 20)), like EXISTING_DRAWS: keys 0-3.
    s = roark_real_state(tables, names)
    sample = jax.jit(sample_outcome)  # one program for the four keys (eager runs grow memory)
    got = [
        sample(jax.random.key(k), tables, s, acts(EMBER, 1)).quick_claw.tolist() for k in range(4)
    ]
    assert got == [[0, 1], [0, 0], [0, 0], [0, 1]]


def test_magnitude_and_shed_skin_draws(tables, names):
    e = empty_outcome()
    assert e.magnitude.tolist() == [4, 4] and e.shed_skin.tolist() == [0, 0]
    assert Outcome._fields[14:16] == ("magnitude", "shed_skin")
    s = roark_state(tables, names)
    keys = jax.random.split(jax.random.key(0), N)
    sample = sample_rows
    o = sample(keys, tables, s, acts(SCRATCH, TACKLE))
    assert o.magnitude.dtype == jnp.int8 and o.shed_skin.dtype == jnp.int8
    magnitude = np.asarray(o.magnitude)
    assert magnitude.shape == (N, 2) and set(np.unique(magnitude)) == set(range(4, 11))
    for side in (0, 1):
        for level, p in zip(range(4, 11), (0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05)):
            assert abs(freq(magnitude[:, side], level) - p) < 5 * np.sqrt(p * (1 - p) / N), level
        shed = np.asarray(o.shed_skin)[:, side]
        assert abs(freq(shed) - 0.3) < 5 * np.sqrt(0.3 * 0.7 / N)


def test_target_draw_field(samples):
    """D2-4: ``target_draw`` is int8[B], a fair bit from its own key (fold_in 44), so every other
    field keeps its samples; ``empty_outcome`` zeroes it."""
    x = np.asarray(samples.target_draw)
    assert x.shape == (N, 2) and set(np.unique(x).tolist()) == {0, 1}
    assert abs(freq(x[:, 0]) - 0.5) < 4 * np.sqrt(0.25 / N)
    keys = jax.random.split(jax.random.key(0), N)
    want = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 44), (2,), 0, 2))(keys)
    np.testing.assert_array_equal(x, np.asarray(want))
    assert np.asarray(empty_outcome().target_draw).tolist() == [0, 0]
