import numpy as np

from nuzline.engine.stats import calc_stats


def test_level_100_neutral():
    s = calc_stats(np.full(6, 100), np.full(6, 31), np.full(6, 252), 100, np.zeros(6, int))
    assert s.tolist() == [404, 299, 299, 299, 299, 299]


def test_adamant_level_100():
    s = calc_stats(
        np.full(6, 100), np.full(6, 31), np.full(6, 252), 100, np.array([0, 1, 0, -1, 0, 0])
    )
    assert s[1] == 328 and s[3] == 269


def test_low_level_truncation():
    s = calc_stats(
        np.array([44, 58, 44, 58, 44, 61]), np.full(6, 20), np.zeros(6, int), 14, np.zeros(6, int)
    )
    assert s[0] == (2 * 44 + 20) * 14 // 100 + 14 + 10


def test_int32_and_nature_never_touches_hp():
    # Pokemon_GetNatureStatValue (src/pokemon.c) leaves HP alone.
    s = calc_stats(np.full(6, 100), np.full(6, 31), np.full(6, 252), 100, np.ones(6, int))
    assert s.dtype == np.int32
    assert s[0] == 404 and s[1] == 299 * 110 // 100
