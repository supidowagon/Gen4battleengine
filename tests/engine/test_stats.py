import numpy as np

from nuzline.engine.stats import SPECIES_SHEDINJA, calc_stats


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


def test_shedinja_hp_is_one():
    # Pokemon_CalcStats (src/pokemon.c:598, :634) pins Shedinja's max HP at 1.
    base = np.array([1, 90, 45, 30, 30, 40])
    for level in (1, 16, 100):
        for iv, ev in ((0, 0), (31, 252)):
            ivs, evs, mods = np.full(6, iv), np.full(6, ev), np.zeros(6, int)
            s = calc_stats(base, ivs, evs, level, mods, species=SPECIES_SHEDINJA)
            other = calc_stats(base, ivs, evs, level, mods)
            assert s[0] == 1
            assert s[1:].tolist() == other[1:].tolist()
            assert other[0] > 1
    assert SPECIES_SHEDINJA == 292


def test_species_argument_leaves_other_species_unchanged():
    args = (np.full(6, 100), np.full(6, 31), np.full(6, 252), 50, np.ones(6, int))
    assert calc_stats(*args, species=291).tolist() == calc_stats(*args).tolist()
