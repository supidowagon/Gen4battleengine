"""Chance outcomes: the empty outcome, exact hit and crit odds, and the sampler's
distributions (decomp citations in ``outcome.py``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import AbilityId, Weather
from nuzline.engine.outcome import Outcome, crit_odds, empty_outcome, hit_threshold, sample_outcome
from nuzline.engine.step import replace
from tests.engine.helpers import roark_state

SCRATCH, EMBER, TACKLE = 0, 2, 0  # Monferno's slots 0 and 2; every fixture opponent's slot 0
N = 20_000


def test_empty_outcome_is_all_hits_no_crits_max_roll():
    o = empty_outcome()
    assert o.speed_tie.shape == () and int(o.speed_tie) == 0
    for name in ("hit", "crit", "roll", "secondary", "contact_proc"):
        assert getattr(o, name).shape == (2, 5)
    assert o.hit.tolist() == [[1] * 5] * 2 and o.crit.tolist() == [[0] * 5] * 2
    assert o.roll.tolist() == [[15] * 5] * 2
    assert o.secondary.tolist() == [[0] * 5] * 2 and o.contact_proc.tolist() == [[0] * 5] * 2
    assert all(getattr(o, f).dtype == jnp.int8 for f in Outcome._fields)


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
    sample = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))
    return sample(keys, tables, s, acts(EMBER, TACKLE))


def freq(x, value=1):
    return float(np.mean(np.asarray(x) == value))


def test_sample_shapes_and_dtypes(samples):
    for f in Outcome._fields:
        assert getattr(samples, f).dtype == jnp.int8, f
    assert samples.hit.shape == (N, 2, 5) and samples.n_hits.shape == (N, 2)


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
        ("flinch_item", (0,), 0.0),  # no implemented item flinches
        ("quick_claw", (1,), 0.0),  # Quick Claw is not implemented
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


def test_lucky_chant_prevents_crits(tables, names):
    s = roark_state(tables, names)
    s = s._replace(lucky_chant=s.lucky_chant.at[1].set(5))
    keys = jax.random.split(jax.random.key(1), 2000)
    o = jax.vmap(sample_outcome, in_axes=(0, None, None, None))(keys, tables, s, acts(SCRATCH, 0))
    assert int(o.crit[:, 0].sum()) == 0 and int(o.crit[:, 1].sum()) > 0
