"""E1 Batch A (rulings G-40 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

Family 1: trainer Pokemon in a data form (G-41).
"""

from __future__ import annotations

import numpy as np

from nuzline.data.trainer import IntermediateData, build_trainer_party, default_moves
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.state import initial_arrays
from nuzline.engine.stats import calc_stats
from tests.conftest import KAIZO_INTERMEDIATE
from tests.engine.helpers import INTERMEDIATE, _inter

# --- Family 1: trainer Pokemon in a data form (G-41) ----------------------------------------


def _slot(tables, opp, slot):
    """The opponent's party slot ``slot`` as ``initial_arrays`` builds it."""
    a = initial_arrays(tables, list(opp), opp)
    return {k: np.asarray(v)[1, slot] for k, v in a.items() if k in _MON}


_MON = ("stats", "max_hp", "types", "ability", "ability_form", "form")


def _calc(tables, spec, row):
    stats = calc_stats(
        np.asarray(tables.species_base_stats[row]),
        np.asarray(spec.ivs),
        np.asarray(spec.evs),
        spec.level,
        np.asarray(tables.nature_mods[spec.nature]),
        species=spec.species,
    )
    return int(stats[Stat.HP]), list(stats[Stat.ATK : Stat.SPE + 1])


def test_kaizo_data_forms_take_the_form_stats_and_the_base_ability(kaizo_tables, kaizo_trainers):
    # Kaizo's builder calls Pokemon_CalcStats after MON_DATA_FORM (rom_diff.md, step 11):
    # Cyrus 913's Deoxys-Speed (form 3, row 498) has row 498's stats. The ability is set at
    # init from the base species (pokemon.c:473-485): Cyrus 780's Giratina-Origin (row 501,
    # Levitate) has Altered's Pressure, with row 501's stats and types.
    inter = IntermediateData(KAIZO_INTERMEDIATE)
    t = next(x for x in kaizo_trainers if x.id == 913)
    opp = build_trainer_party(t, inter)
    k = next(i for i, m in enumerate(opp) if m.species == 386)
    got = _slot(kaizo_tables, opp, k)
    hp, stats = _calc(kaizo_tables, opp[k], 498)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert (int(got["max_hp"]), list(got["stats"])) != _calc(kaizo_tables, opp[k], 386)

    t = next(x for x in kaizo_trainers if x.id == 780)
    opp = build_trainer_party(t, inter)
    k = next(i for i, m in enumerate(opp) if m.species == 487)
    got = _slot(kaizo_tables, opp, k)
    assert (int(got["form"]), int(got["ability_form"])) == (1, 0)
    assert int(got["ability"]) == AbilityId.PRESSURE
    assert np.asarray(kaizo_tables.species_abilities)[501, 0] == AbilityId.LEVITATE
    hp, stats = _calc(kaizo_tables, opp[k], 501)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert list(got["types"]) == list(np.asarray(kaizo_tables.species_types)[501])


def test_vanilla_data_form_keeps_the_base_stats_and_moves(tables):
    # Vanilla's builder never recomputes the stats after the form (src/trainer_data.c:216-218):
    # Devon's (491) Wormadam-Sandy (row 499) has Plant Cloak's stats and level-up default
    # moves, and Sandy's Bug/Ground types (pokemon.c:1219-1225).
    from nuzline.data.trainer import load_trainers

    devon = next(t for t in load_trainers(INTERMEDIATE) if t.id == 491)
    inter = _inter()
    opp = build_trainer_party(devon, inter)
    k = next(i for i, m in enumerate(opp) if m.form == 1)
    got = _slot(tables, opp, k)
    hp, stats = _calc(tables, opp[k], 413)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert (int(got["max_hp"]), list(got["stats"])) != _calc(tables, opp[k], 499)
    assert list(got["types"]) == list(np.asarray(tables.species_types)[499])
    assert opp[k].moves == default_moves(inter.learnsets[413], opp[k].level)
