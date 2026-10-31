import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nuzline.saves.crypto import decrypt_record, encrypt_record
from nuzline.saves.pkm import RecordError, decode_mon, level_from_exp
from tests.saves.writer import mon_record, personality_for

SPECIES = [1, 63, 150, 252, 445]


def test_level_from_exp(tables, names):
    abra = names.species("Abra")  # medium slow
    assert level_from_exp(tables, abra, 0) == 1
    assert level_from_exp(tables, abra, int(tables.exp_table[3, 37])) == 37
    assert level_from_exp(tables, abra, int(tables.exp_table[3, 37]) - 1) == 36
    assert level_from_exp(tables, abra, int(tables.exp_table[3, 100]) + 99999) == 100


def test_all_zero_record_is_empty(tables, names):
    assert decode_mon(b"\0" * 136, ("box", 0, 0), tables, names.charmap) is None
    assert decode_mon(b"\0" * 236, ("party", 0, 0), tables, names.charmap) is None


@settings(deadline=None, max_examples=60)
@given(
    species=st.sampled_from(SPECIES),
    nature=st.integers(0, 24),
    shuffle=st.integers(0, 31),
    ivs=st.tuples(*[st.integers(0, 31)] * 6),
    evs=st.tuples(*[st.integers(0, 255)] * 6),
    moves=st.tuples(*[st.integers(0, 467)] * 4),
    pp=st.tuples(*[st.integers(0, 64)] * 4),
    pp_ups=st.tuples(*[st.integers(0, 3)] * 4),
    item=st.integers(0, 467),
    ability=st.integers(0, 123),
    gender=st.integers(0, 2),
    level=st.integers(1, 100),
)
def test_box_round_trip(
    tables,
    names,
    species,
    nature,
    shuffle,
    ivs,
    evs,
    moves,
    pp,
    pp_ups,
    item,
    ability,
    gender,
    level,
):
    personality = personality_for(nature, shuffle)
    raw = mon_record(
        species=species,
        personality=personality,
        ivs=ivs,
        evs=evs,
        moves=moves,
        pp=pp,
        pp_ups=pp_ups,
        item=item,
        ability=ability,
        gender=gender,
        level=level,
    )
    mon = decode_mon(raw, ("box", 2, 7), tables, names.charmap)
    got = (mon.species, mon.ivs, mon.evs, mon.moves, mon.pp, mon.pp_ups, mon.level)
    assert got == (species, ivs, evs, moves, pp, pp_ups, level)
    got = (mon.nature, mon.location, mon.item, mon.ability, mon.gender, mon.personality)
    assert got == (nature, ("box", 2, 7), item, ability, gender, personality)
    assert mon.current_hp is None and mon.status_raw == 0 and mon.stored_stats is None


def test_party_fields(tables, names):
    stats = (40, 31, 22, 53, 44, 35)
    raw = mon_record(
        species=63,
        level=12,
        personality=personality_for(3, 9),
        party={"status": 0x45, "level": 12, "hp": 30, "stats": stats},
        ot_id=0xDEADBEEF,
        friendship=200,
        form=1,
        is_egg=True,
    )
    assert len(raw) == 236
    mon = decode_mon(raw, ("party", 0, 4), tables, names.charmap)
    assert mon.status_raw == 0x45 and mon.current_hp == 30 and mon.level == 12
    assert mon.stored_stats == stats
    assert mon.ot_id == 0xDEADBEEF and mon.friendship == 200
    assert mon.form == 1 and mon.is_egg is True


def test_checksum_mismatch_raises(tables, names):
    raw = bytearray(mon_record(species=63))
    raw[0x20] ^= 1
    with pytest.raises(RecordError, match="box 0 slot 3"):
        decode_mon(bytes(raw), ("box", 0, 3), tables, names.charmap)


def test_party_level_disagreeing_with_exp_raises(tables, names):
    raw = mon_record(
        species=63,
        level=12,
        party={"status": 0, "level": 13, "hp": 1, "stats": (1,) * 6},
    )
    with pytest.raises(RecordError, match="party slot 2"):
        decode_mon(raw, ("party", 0, 2), tables, names.charmap)


def test_nickname_decoded(tables, names):
    inv = {v: k for k, v in names.charmap.items() if len(v) == 1}
    codes = tuple(inv[ch] for ch in "Abra")
    raw = mon_record(species=63, nickname_codes=codes)
    assert decode_mon(raw, ("box", 0, 0), tables, names.charmap).nickname == "Abra"
    # control code dropped, unknown code becomes "?"
    raw = mon_record(species=63, nickname_codes=(*codes, 0xE000, 0x7777))
    assert decode_mon(raw, ("box", 0, 0), tables, names.charmap).nickname == "Abra?"
    for control in (0x25BC, 0x25BD):
        raw = mon_record(species=63, nickname_codes=(*codes, control))
        assert decode_mon(raw, ("box", 0, 0), tables, names.charmap).nickname == "Abra"


def test_species_zero_with_bad_checksum_raises(tables, names):
    plain = bytearray(decrypt_record(mon_record(species=0, exp=0, ot_id=5)))
    plain[6:8] = ((int.from_bytes(plain[6:8], "little") + 1) & 0xFFFF).to_bytes(2, "little")
    raw = encrypt_record(bytes(plain))
    assert any(raw)
    with pytest.raises(RecordError, match="box 1 slot 4"):
        decode_mon(raw, ("box", 1, 4), tables, names.charmap)


def test_unknown_move_id_names_the_slot(tables, names):
    raw = mon_record(species=63, moves=(33, 0, 9999, 0))
    with pytest.raises(RecordError, match=r"box 0 slot 3: move slot 2 has unknown move id 9999"):
        decode_mon(raw, ("box", 0, 3), tables, names.charmap)


def test_unknown_item_id_names_the_slot(tables, names):
    raw = mon_record(species=63, item=9999)
    with pytest.raises(RecordError, match=r"party slot 4: unknown item id 9999"):
        decode_mon(raw, ("party", 0, 4), tables, names.charmap)


def test_out_of_range_species_names_the_slot(tables, names):
    raw = mon_record(species=9999, exp=0)
    with pytest.raises(RecordError, match=r"box 2 slot 5: species 9999 out of range"):
        decode_mon(raw, ("box", 2, 5), tables, names.charmap)
