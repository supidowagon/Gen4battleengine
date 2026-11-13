import dataclasses

import numpy as np

from nuzline.engine.ids import Stat, Status
from nuzline.engine.state import make_state
from nuzline.engine.stats import calc_stats
from nuzline.prep.leveling import candies_to_cap, level_to_cap
from nuzline.saves.pkm import decode_mon
from tests.saves.writer import mon_record, personality_for

IVS = (10, 11, 12, 13, 14, 15)


def _stats(tables, sp, level, nature=3, ivs=IVS, species=0):
    return calc_stats(
        np.asarray(tables.species_base_stats)[sp],
        ivs,
        (0,) * 6,
        level,
        np.asarray(tables.nature_mods)[nature],
        species=species,
    )


def _mon(tables, names, species, level, *, party=True, hp=None, status_raw=0, **kw):
    sp = names.species(species)
    personality = personality_for(3, 0)
    stats = tuple(int(v) for v in _stats(tables, sp, level, species=sp))
    raw = mon_record(
        species=sp,
        personality=personality,
        level=level,
        ivs=IVS,
        ability=int(tables.species_abilities_decomp[sp][0]),
        party={"status": status_raw, "level": level, "hp": hp or stats[0], "stats": stats}
        if party
        else None,
        **kw,
    )
    loc = ("party", 0, 0) if party else ("box", 1, 0)
    return decode_mon(raw, loc, tables, names.charmap)


def test_level_to_cap_rebuilds(tables, names):
    tackle = names.move("Tackle")
    leer = names.move("Leer")
    mon = _mon(
        tables,
        names,
        "Pikachu",
        10,
        hp=5,
        status_raw=0x08,  # poison
        moves=(tackle, leer, 0, 0),
        pp=(3, 30, 0, 0),
        pp_ups=(1, 0, 0, 0),
    )
    mon = dataclasses.replace(mon, pp=(3, 30, 0, 0))
    base_pp = np.asarray(tables.move_pp)
    spec = level_to_cap(mon, tables, 16)
    assert spec.level == 16
    assert spec.stats is None and spec.current_hp is None
    assert spec.status == 0 and spec.status_counter == 0
    assert spec.pp == (
        int(base_pp[tackle]) * 6 // 5,
        int(base_pp[leer]),
        0,
        0,
    )
    state = make_state(tables, [spec], [spec])
    expected = _stats(tables, mon.species, 16, nature=mon.nature, ivs=mon.ivs)[0]
    assert int(state.hp[0, 0]) == int(state.max_hp[0, 0]) == int(expected)


def test_boxed_mon_to_cap(tables, names):
    mon = _mon(tables, names, "Pikachu", 4, party=False, moves=(names.move("Tackle"), 0, 0, 0))
    assert level_to_cap(mon, tables, 16).level == 16
    assert candies_to_cap(4, 16) == 12


def test_above_cap_is_kept(tables, names):
    mon = _mon(tables, names, "Pikachu", 20, moves=(names.move("Tackle"), 0, 0, 0))
    spec = level_to_cap(mon, tables, 16)
    assert spec.level == 20
    assert candies_to_cap(20, 16) == 0
    again = _stats(tables, mon.species, 20, nature=mon.nature, ivs=mon.ivs)
    assert again.tolist() == list(mon.stored_stats)
    state = make_state(tables, [spec], [spec])
    assert int(state.max_hp[0, 0]) == mon.stored_stats[Stat.HP]


def test_shedinja_to_cap(tables, names):
    mon = _mon(tables, names, "Shedinja", 10, party=False, moves=(names.move("Tackle"), 0, 0, 0))
    spec = level_to_cap(mon, tables, 16)
    state = make_state(tables, [spec], [spec])
    assert int(state.max_hp[0, 0]) == 1 and int(state.hp[0, 0]) == 1
    assert Status.NONE == spec.status


def _base_exp(tables, species: int, level: int) -> int:
    return int(tables.exp_table[int(tables.species_exp_rate[species]), level])


def test_a_raised_mon_has_the_caps_base_exp(tables, names):
    # Rare Candy: exp += Pokemon_GetExpToNextLevel (item_use_pokemon.c:291-293), so the
    # result is the next level's base exactly (pokemon.c:2308-2319).
    sp = names.species("Pikachu")
    exp = _base_exp(tables, sp, 12) + 40
    mon = _mon(tables, names, "Pikachu", 12, exp=exp, moves=(names.move("Tackle"), 0, 0, 0))
    assert mon.experience == exp
    assert level_to_cap(mon, tables, 16).exp == _base_exp(tables, sp, 16)


def test_a_mon_at_or_above_the_cap_keeps_its_exp(tables, names):
    sp = names.species("Pikachu")
    for level in (16, 20):
        exp = _base_exp(tables, sp, level) + 3
        mon = _mon(tables, names, "Pikachu", level, exp=exp, moves=(names.move("Tackle"), 0, 0, 0))
        assert level_to_cap(mon, tables, 16).exp == exp


def test_level_to_cap_resolves_the_boost_with_the_trainer(tables, names):
    from nuzline.saves.save import SaveTrainer

    mon = _mon(tables, names, "Pikachu", 12, moves=(names.move("Tackle"), 0, 0, 0))
    me = SaveTrainer(id=0, gender=0, name_codes=(0xFFFF,) * 8, rom_language=2)
    assert level_to_cap(mon, tables, 16, me).exp_boost == 100
    assert level_to_cap(mon, tables, 16).exp_boost == 0
