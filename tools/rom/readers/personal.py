"""Species data, experience tables and species weights from the ROM."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import STAT_KEYS, _constants, _data_forms, species_form_rows
from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.text import message_bank

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles

PERSONAL_NARC = "poketool/personal/pl_personal.narc"
GROWTBL_NARC = "poketool/personal/pl_growtbl.narc"
GIRA_NARC = "application/zukanlist/zkn_data/zukan_data_gira.narc"
PERSONAL_SIZE = 44  # sizeof(SpeciesData), include/struct_defs/species.h:39-58
SPECIES_COUNT = 496  # lines of generated/species.txt
FORM_ROW_COUNT = 12  # data forms, personal rows 496-507 (Pokemon_GetFormNarcIndex, Ruling 6)
ROW_COUNT = SPECIES_COUNT + FORM_ROW_COUNT  # members of pl_personal.narc and wotbl.narc
DEX_COUNT = 494  # National Dex 0..493; Egg and Bad Egg have no dex entry
WEIGHT_MEMBER = 1  # PackDexData: height is member 0, weight member 1 (speciesproc.c:963-966)
EXP_RATES = 8
EXP_LEVELS = 101  # levels 0..100, tools/scripts/make_pl_growtbl.py
# Offsets in SpeciesBaseStats order (include/struct_defs/species.h:15-22): hp, atk, def, spe, spa, spd.
_STAT_OFFSET = {
    "hp": 0,
    "attack": 1,
    "defense": 2,
    "speed": 3,
    "special_attack": 4,
    "special_defense": 5,
}


def read_exp_tables(rom: RomFiles) -> list[list[int]]:
    """``[rate][level]`` for levels 0..100: member ``rate`` of ``pl_growtbl.narc`` is 101
    little-endian u32 (``tools/scripts/make_pl_growtbl.py``)."""
    members = fixed_members(rom.narc(GROWTBL_NARC), 4 * EXP_LEVELS, GROWTBL_NARC)
    if len(members) != EXP_RATES:
        raise ReaderError(f"{GROWTBL_NARC}: expected {EXP_RATES} members, got {len(members)}")
    return [list(struct.unpack(f"<{EXP_LEVELS}I", m)) for m in members]


def _weights(rom: RomFiles) -> list[int]:
    """Altered-Giratina weights (Ruling 9) for species 0..493, then 0 for 494 and 495."""
    members = rom.narc(GIRA_NARC)
    if len(members) <= WEIGHT_MEMBER:
        raise ReaderError(f"{GIRA_NARC}: {len(members)} members, no weight member")
    data = members[WEIGHT_MEMBER]
    if len(data) != 4 * DEX_COUNT:
        raise ReaderError(
            f"{GIRA_NARC} member {WEIGHT_MEMBER}: expected {4 * DEX_COUNT} bytes, got {len(data)}"
        )
    return list(struct.unpack(f"<{DEX_COUNT}I", data)) + [0] * (SPECIES_COUNT - DEX_COUNT)


def _personal_members(rom: RomFiles) -> list[bytes]:
    """The ``ROW_COUNT`` members of ``pl_personal.narc`` (species, then data forms)."""
    members = fixed_members(rom.narc(PERSONAL_NARC), PERSONAL_SIZE, PERSONAL_NARC)
    if len(members) != ROW_COUNT:
        raise ReaderError(f"{PERSONAL_NARC}: {len(members)} members, expected {ROW_COUNT}")
    return members


def _row_reader(decomp: Path):
    """A function ``(members, row) -> dict`` of member ``row``'s ``SpeciesData`` fields
    (``include/struct_defs/species.h:39-58``): stats 0-5, types 6-7, base exp reward 9,
    gender ratio 16, base friendship 18, exp rate 19, abilities 22-23. No name or weight."""
    n_types = len(_constants(decomp, "pokemon_types", "TYPE_"))
    n_abilities = len(_constants(decomp, "abilities", "ABILITY_"))

    def check(row: int, what: str, value: int, limit: int, off: int) -> int:
        if value >= limit:
            raise ReaderError(
                f"{PERSONAL_NARC} member {row} offset {off}: {what} {value} is out of range"
            )
        return value

    def read(members: list[bytes], row: int) -> dict:
        m = members[row]
        return {
            "base_stats": [m[_STAT_OFFSET[k]] for k in STAT_KEYS],
            "types": [check(row, "type", m[6 + i], n_types, 6 + i) for i in range(2)],
            "abilities": [check(row, "ability", m[22 + i], n_abilities, 22 + i) for i in range(2)],
            # Any byte is a valid ratio threshold (Kaizo gives Vespiquen 207, which no
            # GENDER_RATIO_* names), so it is not checked against the named set.
            "gender_ratio": m[16],
            "base_friendship": m[18],
            "exp_rate": check(row, "exp rate", m[19], EXP_RATES, 19),
            "exp_yield": m[9],  # baseExpReward, include/struct_defs/species.h:43
        }

    return read


def read_species(rom: RomFiles, decomp: Path, charmap: dict[int, str]) -> list[dict]:
    """``species.json`` rows for ids ``0..495`` (``_row_reader``'s fields). Names come from
    ``TEXT_BANK_SPECIES_NAME``."""
    members = _personal_members(rom)
    names = message_bank(rom, decomp, "TEXT_BANK_SPECIES_NAME", charmap)
    if len(names) < SPECIES_COUNT:
        raise ReaderError(f"TEXT_BANK_SPECIES_NAME: {len(names)} entries, need {SPECIES_COUNT}")
    read = _row_reader(decomp)
    weights = _weights(rom)
    out = []
    for sid in range(SPECIES_COUNT):
        row = read(members, sid)
        out.append(
            {
                # The ROM bank calls species 0 "-----" (speciesproc.c:846) but
                # extract_species names SPECIES_NONE "----------"; id 0 is not a real
                # species and the round trip needs equality, so use the extractor's.
                "name": "----------" if sid == 0 else names[sid],
                "base_stats": row["base_stats"],
                "types": row["types"],
                "abilities": row["abilities"],
                "gender_ratio": row["gender_ratio"],
                "weight_hg": weights[sid],
                "base_friendship": row["base_friendship"],
                "exp_rate": row["exp_rate"],
                "exp_yield": row["exp_yield"],
            }
        )
    return out


def read_forms(rom: RomFiles, decomp: Path, charmap: dict[int, str]) -> list[dict]:
    """``forms.json``: the data forms' row fields read from ``pl_personal.narc`` members
    496-507. Each form's species, form number, row and name come from the decomp's form
    registry (Ruling 6; the ROM holds no form names), and its weight is the base species'
    (the dex weight is per species). The decomp grid's data rows must be exactly
    ``496..ROW_COUNT - 1``. ``charmap`` is unused; it keeps ``read_species``' signature."""
    del charmap
    members = _personal_members(rom)
    data_rows = sorted({r for row in species_form_rows(decomp) for r in row if r >= SPECIES_COUNT})
    if data_rows != list(range(SPECIES_COUNT, len(members))):
        raise ReaderError(
            f"decomp form rows {data_rows} are not the data members "
            f"{SPECIES_COUNT}..{len(members) - 1} of {PERSONAL_NARC}"
        )
    read = _row_reader(decomp)
    weights = _weights(rom)
    return [
        {
            "species": sid,
            "form": form,
            "row": row,
            "name": key.title(),
            **read(members, row),
            "weight_hg": weights[sid],
        }
        for sid, form, key, row in _data_forms(decomp)
    ]


TM_MASK_OFFSET = (
    28  # tmLearnsetMasks, include/struct_defs/species.h:57 (u32[4], after abilities etc.)
)
TM_BITS = 100  # NUM_TMS + NUM_HMS; bits 100-127 are unused


def read_tm_learnsets(rom: RomFiles) -> dict[str, list[int]]:
    """``tm_learnsets.json``: personal row -> sorted TM/HM indices set in ``tmLearnsetMasks``
    (bit n = TM index n: TM01-TM92 are 0-91, HM01-HM08 are 92-99, as ``ProcTMLearnset`` in
    ``tools/dataproc/src/speciesproc.c`` sets them) for personal rows ``0..507`` (species,
    then data forms)."""
    members = _personal_members(rom)
    out = {}
    for sid in range(ROW_COUNT):
        masks = struct.unpack_from("<4I", members[sid], TM_MASK_OFFSET)
        mask = sum(w << (32 * i) for i, w in enumerate(masks))
        if mask >> TM_BITS:
            raise ReaderError(
                f"{PERSONAL_NARC} member {sid} offset {TM_MASK_OFFSET}: "
                f"TM mask bits above {TM_BITS - 1} are set"
            )
        out[str(sid)] = [b for b in range(TM_BITS) if mask >> b & 1]
    return out
