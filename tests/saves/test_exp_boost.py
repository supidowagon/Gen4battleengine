"""The save's exp fields and the exp boost (K3d-exp T3): ``SaveTrainer``, the OT test of
``BattleSystem_PokemonIsOT`` (battle_lib.c:6286-6303) on raw name codes
(charcode_util.c:87-104), and the language test against the ROM's ``gGameLanguage``
(battle_script.c:9757-9763)."""

from __future__ import annotations

import pytest

from nuzline.data.games import LANGUAGE_ENGLISH
from nuzline.saves import load_save
from nuzline.saves.pkm import ITEM_LUXURY_BALL, decode_text
from nuzline.saves.save import SaveTrainer
from tests.saves.synthetic import SAMPLE, party_record
from tests.saves.writer import build_image

EOS = 0xFFFF
A, B, C, D, E, F, G = range(0x12B, 0x132)  # charmap codes of "A" .. "G"
QUESTION = 0x1AC  # the "?" glyph
UNMAPPED = 0x1E3  # no glyph: decoded as "?" too
LANGUAGE_JAPANESE = 1  # enum Language, include/constants/versions.h:23-26


def _save(tmp_path, tables, names, trainer=None, **mon):
    """A one-Pokemon save; ``trainer`` holds ``build_image``'s trainer keywords."""
    rec = party_record(tables, names, "Monferno", 14, SAMPLE[0][2], **mon)
    path = tmp_path / "g.sav"
    path.write_bytes(build_image(party=[rec], **(trainer or {})))
    return load_save(path, tables=tables, names=names)


def _boost(tmp_path, tables, names, trainer=None, **mon) -> int:
    save = _save(tmp_path, tables, names, trainer, **mon)
    return save.party[0].to_spec(tables, save.trainer).exp_boost


ME = {"trainer_name_codes": (D, A, C, EOS), "tid": 1234, "sid": 5678, "trainer_gender": 1}
MINE = {"ot_name_codes": (D, A, C, EOS), "ot_id": 1234 | (5678 << 16), "ot_gender": 1}


def test_the_save_trainer_is_read(tmp_path, tables, names):
    save = _save(tmp_path, tables, names, ME, **MINE)
    assert save.trainer == SaveTrainer(
        id=1234 | (5678 << 16),
        gender=1,
        name_codes=(D, A, C, EOS, EOS, EOS, EOS, EOS),
        rom_language=LANGUAGE_ENGLISH,
    )


def test_the_original_trainer_gets_100(tmp_path, tables, names):
    assert _boost(tmp_path, tables, names, ME, **MINE) == 100


@pytest.mark.parametrize(
    "change",
    [
        {"ot_id": 1234 | (5679 << 16)},  # the full u32: same TID, other SID
        {"ot_gender": 0},
        {"ot_name_codes": (D, A, D, EOS)},
    ],
    ids=["id", "gender", "name"],
)
def test_another_trainer_gets_150(tmp_path, tables, names, change):
    assert _boost(tmp_path, tables, names, ME, **{**MINE, **change}) == 150


def test_a_foreign_language_traded_pokemon_gets_170(tmp_path, tables, names):
    mon = {**MINE, "ot_id": 1, "language": LANGUAGE_JAPANESE}
    assert _boost(tmp_path, tables, names, ME, **mon) == 170


def test_the_language_test_reads_the_rom_not_the_saves_trainer_language(tmp_path, tables, names):
    # The save's TrainerInfo.language byte equals the Pokemon's language here; the boost
    # compares with gGameLanguage (English), so it is still 170.
    trainer = {**ME, "trainer_language": LANGUAGE_JAPANESE}
    mon = {**MINE, "ot_id": 1, "language": LANGUAGE_JAPANESE}
    assert _boost(tmp_path, tables, names, trainer, **mon) == 170
    english = {**mon, "language": LANGUAGE_ENGLISH}
    assert _boost(tmp_path, tables, names, trainer, **english) == 150


def test_names_that_decode_alike_but_differ_in_raw_codes_are_not_the_ot(tmp_path, tables, names):
    trainer = {**ME, "trainer_name_codes": (D, QUESTION, EOS)}
    mon = {**MINE, "ot_name_codes": (D, UNMAPPED, EOS)}
    save = _save(tmp_path, tables, names, trainer, **mon)
    assert names.charmap.get(QUESTION) == "?" and UNMAPPED not in names.charmap
    raw = bytes(b for code in (D, QUESTION, EOS) for b in code.to_bytes(2, "little"))
    alike = bytes(b for code in (D, UNMAPPED, EOS) for b in code.to_bytes(2, "little"))
    assert decode_text(raw, 0, 8, names.charmap) == decode_text(alike, 0, 8, names.charmap)
    assert save.party[0].to_spec(tables, save.trainer).exp_boost == 150


def test_codes_after_a_shared_eos_are_ignored(tmp_path, tables, names):
    trainer = {**ME, "trainer_name_codes": (D, A, EOS, B, C)}
    mon = {**MINE, "ot_name_codes": (D, A, EOS, F, G)}
    assert _boost(tmp_path, tables, names, trainer, **mon) == 100


def test_the_eighth_ot_code_reads_as_eos(tmp_path, tables, names):
    # MON_DATA_OT_NAME copies 7 codes and writes EOS (pokemon.c:1121-1128): a 7-letter
    # name matches whatever the Pokemon stores in its 8th code.
    trainer = {**ME, "trainer_name_codes": (A, B, C, D, E, F, G, EOS)}
    mon = {**MINE, "ot_name_codes": (A, B, C, D, E, F, G, QUESTION)}
    assert _boost(tmp_path, tables, names, trainer, **mon) == 100
    other = {**mon, "ot_name_codes": (A, B, C, D, E, F, A, EOS)}
    assert _boost(tmp_path, tables, names, trainer, **other) == 150


def test_to_spec_without_the_trainer_leaves_the_boost_unresolved(tmp_path, tables, names):
    save = _save(tmp_path, tables, names, ME, **MINE)
    assert save.party[0].to_spec(tables).exp_boost == 0


def test_exp_pokerus_ball_and_egg_location(tmp_path, tables, names):
    rate = int(tables.species_exp_rate[names.species("Monferno")])
    exp = int(tables.exp_table[rate, 14]) + 5  # level 14, 5 past its base
    save = _save(
        tmp_path,
        tables,
        names,
        ME,
        **MINE,
        exp=exp,
        pokerus=0x21,
        pokeball=ITEM_LUXURY_BALL,
        egg_location_dp=2000,
    )
    mon = save.party[0]
    assert (mon.pokerus, mon.pokeball, mon.egg_location) == (0x21, ITEM_LUXURY_BALL, 2000)
    spec = mon.to_spec(tables, save.trainer)
    assert (spec.exp, spec.pokerus, spec.friend_bonus) == (exp, 1, 1)
    # K3d-exp T7: the egg location and PP Ups reach the spec (coverage inputs).
    assert (spec.egg_location, spec.pp_ups) == (2000, mon.pp_ups)
    plain = _save(tmp_path, tables, names, ME, **MINE).party[0].to_spec(tables)
    assert (plain.pokerus, plain.friend_bonus, plain.egg_location) == (0, 0, 0)


@pytest.mark.parametrize(
    ("dp", "pt", "want"),
    [(3002, 1000, 1000), (3002, 0, 3002), (2000, 1000, 2000)],
)
def test_egg_location_follows_mon_data_egg_location(tmp_path, tables, names, dp, pt, want):
    # pokemon.c:1159-1166: the PtHGSS field only behind the fateful-encounter location.
    mon = {**MINE, "egg_location_dp": dp, "egg_location_pt": pt}
    assert _save(tmp_path, tables, names, ME, **mon).party[0].egg_location == want


def test_the_probe_save_is_the_players_own_one_exp_short_of_17(kaizo_save_probe_path, kaizo_tables):
    # Design §0: every probe party member is the OT's (boost 100) and 1 exp short of 17.
    from nuzline.data.games import KAIZO

    save = load_save(kaizo_save_probe_path, game=KAIZO, tables=kaizo_tables)
    rates, table = kaizo_tables.species_exp_rate, kaizo_tables.exp_table
    for mon in save.party:
        spec = mon.to_spec(kaizo_tables, save.trainer)
        assert spec.exp_boost == 100
        assert spec.exp == int(table[int(rates[mon.species]), 17]) - 1
