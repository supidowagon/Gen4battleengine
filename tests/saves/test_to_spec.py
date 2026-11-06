import dataclasses

import numpy as np
import pytest

from nuzline.engine.ids import Status
from nuzline.engine.stats import calc_stats
from nuzline.saves.pkm import RecordError, decode_mon
from tests.saves.writer import mon_record, personality_for


def _calc(tables, sp, ivs, evs, level, nature):
    return calc_stats(
        np.asarray(tables.species_base_stats)[sp],
        ivs,
        evs,
        level,
        np.asarray(tables.nature_mods)[nature],
    )


def _box(tables, names, species="Pikachu", **kw):
    sp = names.species(species)
    kw.setdefault("personality", personality_for(3, 0))
    kw.setdefault("ability", int(tables.species_abilities_decomp[sp][0]))
    raw = mon_record(species=sp, **kw)
    return decode_mon(raw, ("box", 1, 4), tables, names.charmap)


def _party(tables, names, species="Pikachu", level=30, evs=(0,) * 6, status=0, hp=20):
    sp = names.species(species)
    personality = personality_for(3, 0)
    ivs = (10, 11, 12, 13, 14, 15)
    stats = tuple(int(v) for v in _calc(tables, sp, ivs, (0,) * 6, level, personality % 25))
    raw = mon_record(
        species=sp,
        personality=personality,
        level=level,
        ivs=ivs,
        evs=evs,
        ability=int(tables.species_abilities_decomp[sp][0]),
        party={"status": status, "level": level, "hp": hp, "stats": stats},
    )
    return decode_mon(raw, ("party", 0, 1), tables, names.charmap)


def test_box_mon_converts(tables, names):
    mon = _box(tables, names, pp=(30, 0, 0, 0), moves=(33, 0, 0, 0))
    spec = mon.to_spec(tables)
    assert spec.stats is None and spec.current_hp is None
    assert spec.pp == (30, 0, 0, 0)
    assert spec.species == mon.species and spec.level == mon.level
    assert spec.ability_slot == 0


def test_second_ability_slot(tables, names):
    abil = [int(a) for a in tables.species_abilities_decomp[names.species("Pidgey")]]
    assert abil[1] != 0
    mon = dataclasses.replace(_box(tables, names, species="Pidgey"), ability=abil[1])
    assert mon.to_spec(tables).ability_slot == 1


def test_party_mon_with_lagging_stats_converts(tables, names):
    mon = _party(tables, names, evs=(40, 40, 40, 40, 40, 40))
    spec = mon.to_spec(tables)
    assert spec.stats == mon.stored_stats and spec.current_hp == 20


def test_party_stat_outside_band_raises(tables, names):
    mon = _party(tables, names, evs=(0, 8, 0, 0, 0, 0))
    top = [int(v) for v in _calc(tables, mon.species, mon.ivs, mon.evs, mon.level, mon.nature)]
    top[1] += 1
    with pytest.raises(RecordError, match="party slot 1: Atk"):
        dataclasses.replace(mon, stored_stats=tuple(top)).to_spec(tables)
    below = list(mon.stored_stats)
    below[2] -= 1
    with pytest.raises(RecordError, match="party slot 1: Def"):
        dataclasses.replace(mon, stored_stats=tuple(below)).to_spec(tables)


def test_fainted_party_mon_converts(tables, names):
    spec = _party(tables, names, hp=0).to_spec(tables)
    assert spec.current_hp == 0


def test_pp_with_pp_ups_converts_and_above_max_raises(tables, names):
    kw = {"moves": (33, 0, 0, 0), "pp_ups": (3, 0, 0, 0)}  # Tackle, base 35
    assert int(tables.move_pp[33]) == 35
    assert _box(tables, names, pp=(56, 0, 0, 0), **kw).to_spec(tables).pp[0] == 56
    with pytest.raises(RecordError, match="box 2 slot 4"):
        _box(tables, names, pp=(57, 0, 0, 0), **kw).to_spec(tables)


def test_sleep_status_and_counter(tables, names):
    spec = _party(tables, names, status=3).to_spec(tables)
    assert (spec.status, spec.status_counter) == (Status.SLEEP, 3)


def test_toxic_counter_dropped(tables, names):
    spec = _party(tables, names, status=(1 << 7) | (5 << 8)).to_spec(tables)
    assert (spec.status, spec.status_counter) == (Status.TOXIC, 0)


def _row_party(tables, names, species, form, slot=0):
    """A party record of ``species`` ``form`` with stored stats from its personal row, at
    party slot ``slot``. (The brief's ``mon_record(...).to_spec`` and ``load_party_record``:
    ``mon_record`` returns the encrypted bytes, so the record is decoded here.)"""
    sp = names.species(species)
    row = int(tables.species_form_row[sp, form])
    personality = personality_for(3, 0)
    ivs = (10, 11, 12, 13, 14, 15)
    stats = tuple(int(v) for v in _calc(tables, row, ivs, (0,) * 6, 30, personality % 25))
    raw = mon_record(
        species=sp,
        form=form,
        personality=personality,
        level=30,
        ivs=ivs,
        ability=int(tables.species_abilities_decomp[row][0]),
        party={"status": 0, "level": 30, "hp": stats[0], "stats": stats},
    )
    return decode_mon(raw, ("party", 0, slot), tables, names.charmap)


@pytest.mark.parametrize(
    ("species", "form"),
    [("Rotom", 2), ("Rotom", 5), ("Wormadam", 1), ("Giratina", 1), ("Shaymin", 1), ("Deoxys", 3)],
)
def test_data_form_converts(tables, names, species, form):
    spec = _row_party(tables, names, species, form).to_spec(tables)
    assert spec.form == form and spec.species == names.species(species)


def test_data_form_converts_arceus_form_raises(
    tables, names
):  # replaces test_alternate_form_raises
    assert _row_party(tables, names, "Rotom", 2).to_spec(tables).form == 2
    with pytest.raises(RecordError, match="Arceus"):
        _box(tables, names, species="Arceus", form=3).to_spec(tables)
    assert _box(tables, names, species="Arceus", form=0).to_spec(tables).form == 0
    with pytest.raises(RecordError, match="party slot 0: species 479 has no form 6"):
        mon = dataclasses.replace(_row_party(tables, names, "Rotom", 0), form=6)
        mon.to_spec(tables)


def test_form_beyond_the_count_raises(tables, names):
    # Unown has 28 forms (0-27); a cosmetic species' form past its count is no form at all.
    with pytest.raises(RecordError, match="box 2 slot 4: species 201 has no form 28"):
        _box(tables, names, species="Unown", form=28).to_spec(tables)
    with pytest.raises(RecordError, match="box 2 slot 4: species 25 has no form 1"):
        _box(tables, names, species="Pikachu", form=1).to_spec(tables)


def test_form_ability_comes_from_the_row(tables, names):
    # Giratina-Origin stores Levitate (row 501), which Altered Forme cannot have.
    spec = _row_party(tables, names, "Giratina", 1).to_spec(tables)
    assert spec.ability_slot == 0
    altered = dataclasses.replace(_row_party(tables, names, "Giratina", 1), form=0)
    with pytest.raises(RecordError, match="party slot 0: ability"):
        altered.to_spec(tables)


@pytest.mark.parametrize("species", ["Unown", "Burmy", "Shellos", "Gastrodon"])
def test_cosmetic_form_passes(tables, names, species):
    assert _box(tables, names, species=species, form=1).to_spec(tables).form == 1


def test_current_hp_above_max_raises(tables, names):
    mon = _party(tables, names)
    bad = dataclasses.replace(mon, current_hp=mon.stored_stats[0] + 1)
    with pytest.raises(RecordError, match="party slot 1"):
        bad.to_spec(tables)


def test_egg_raises(tables, names):
    with pytest.raises(RecordError, match="box 2 slot 4"):
        _box(tables, names, is_egg=True).to_spec(tables)


def test_foreign_ability_raises(tables, names):
    mon = dataclasses.replace(_box(tables, names), ability=123)
    with pytest.raises(RecordError, match="box 2 slot 4"):
        mon.to_spec(tables)


def test_shedinja_hp_must_be_one(tables, names):
    mon = _party(tables, names, species="Shedinja", level=30, hp=1)
    assert mon.stored_stats[0] > 1  # the generic formula would give more
    ok = dataclasses.replace(mon, stored_stats=(1, *mon.stored_stats[1:]), current_hp=1)
    assert ok.to_spec(tables).stats[0] == 1
    bad = dataclasses.replace(mon, stored_stats=(2, *mon.stored_stats[1:]), current_hp=1)
    with pytest.raises(RecordError, match="party slot 1: Shedinja HP 2 must be 1"):
        bad.to_spec(tables)
