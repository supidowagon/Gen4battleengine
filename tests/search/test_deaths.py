"""Success test, buckets and death distribution (``search/deaths.py``; plan 4b ruling 8)."""

from types import SimpleNamespace

import numpy as np
import pytest

from nuzline.search.deaths import BUCKETS, DeathDistribution, bucket, over_budget, success


def _mask(*slots):
    out = np.zeros(6, np.int8)
    out[list(slots)] = 1
    return out


# won, ended, faints, deaths, k, protected -> (success, bucket)
CASES = [
    (1, 1, 0, _mask(), 0, _mask(), True, "win_0"),
    (1, 1, 1, _mask(2), 0, _mask(), False, "win_1"),
    (1, 1, 1, _mask(2), 1, _mask(), True, "win_1"),
    (1, 1, 2, _mask(1, 2), 1, _mask(), False, "win_2"),
    (1, 1, 3, _mask(1, 2, 3), 3, _mask(), True, "win_3"),
    (1, 1, 5, _mask(0, 1, 2, 3, 4), 3, _mask(), False, "win_4plus"),
    (1, 1, 1, _mask(1), 3, _mask(1), False, "win_protected_lost"),
    (1, 1, 1, _mask(2), 3, _mask(1), True, "win_1"),
    (0, 1, 3, _mask(0, 1, 2), 3, _mask(), False, "whiteout"),
    (0, 0, 0, _mask(), 3, _mask(), False, "unfinished"),
]


@pytest.mark.parametrize("won,ended,faints,deaths,k,protected,ok,name", CASES)
def test_success_and_bucket(won, ended, faints, deaths, k, protected, ok, name):
    assert bool(success(won, faints, deaths, k, protected)) is ok
    assert BUCKETS[int(bucket(won, ended, faints, deaths, protected))] == name


def test_batch_broadcast():
    won = np.array([1, 1, 0])
    faints = np.array([0, 2, 1])
    deaths = np.stack([_mask(), _mask(0, 1), _mask(0)])
    assert success(won, faints, deaths, 1, _mask()).tolist() == [True, False, False]
    got = bucket(won, np.array([1, 1, 1]), faints, deaths, _mask())
    assert [BUCKETS[i] for i in got] == ["win_0", "win_2", "whiteout"]


def test_over_budget():
    state = SimpleNamespace(
        player_faints=np.array([1, 2, 1]),
        faint_budget=np.array([1, 1, 3]),
        deaths=np.stack([_mask(0), _mask(0, 1), _mask(2)]),
        protected=np.stack([_mask(), _mask(), _mask(2)]),
    )
    assert over_budget(state).tolist() == [False, True, True]
    one = SimpleNamespace(
        player_faints=np.int8(0), faint_budget=np.int8(0), deaths=_mask(), protected=_mask()
    )
    assert not bool(over_budget(one))


def test_distribution():
    d = DeathDistribution.from_counts([5, 3, 1, 1, 0, 2, 7, 1])
    assert d.n == 20
    assert d.rung(0) == 0.25 and d.rung(1) == 0.4 and d.rung(3) == 0.5
    assert d.p_win == pytest.approx(0.6)
    with pytest.raises(ValueError):
        d.rung(4)
    assert tuple(d.as_dict()) == BUCKETS + ("n",)
    with pytest.raises(ValueError):
        DeathDistribution.from_counts([0] * 8)
