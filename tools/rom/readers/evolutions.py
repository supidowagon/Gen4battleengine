"""Evolutions from ``poketool/personal/evo.narc``."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.personal import SPECIES_COUNT

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles

EVO_NARC = "poketool/personal/evo.narc"
MAX_EVOLUTIONS = 7  # include/struct_defs/species.h:8
EVO_SIZE = 6  # SpeciesEvolution: u16 method, param, targetSpecies (species.h:60-64)
MEMBER_SIZE = 44  # SpeciesEvolutionList: data[7] ALIGN_4 (speciesproc.c:388) = 42 + 2 padding
METHOD_COUNT = 27  # lines of generated/evolution_methods.txt


def read_evolutions(rom: RomFiles) -> dict[str, list[list[int]]]:
    """``{species: [[method, param, target], ...]}`` for species ``0..495``.

    ``LoadSpeciesEvolutions`` (``src/pokemon.c:4812``) loads member ``species``; the archive
    holds one member per personal row (the data forms follow the species), and only the
    species' own are read. Each member is 7 slots of ``(method, param, target)`` u16s plus
    two padding bytes (``ProcEvolutions``, ``speciesproc.c:391-469``). Method-0 slots are
    dropped whatever else they hold: the game treats method 0 as no evolution, and Kaizo's
    member 436 (Bronzor) keeps a level-33 param and a target in one (measured), so a method-0
    slot's content is not a layout error. A method or target past its table, or non-zero
    padding, raises."""
    members = fixed_members(rom.narc(EVO_NARC), MEMBER_SIZE, EVO_NARC)
    if len(members) < SPECIES_COUNT:
        raise ReaderError(f"{EVO_NARC}: {len(members)} members, expected at least {SPECIES_COUNT}")
    out = {}
    for sid in range(SPECIES_COUNT):
        m = members[sid]
        where = f"{EVO_NARC} member {sid}"
        if any(m[EVO_SIZE * MAX_EVOLUTIONS :]):
            raise ReaderError(f"{where}: non-zero padding bytes")
        rows = []
        for slot in range(MAX_EVOLUTIONS):
            method, param, target = struct.unpack_from("<3H", m, slot * EVO_SIZE)
            if method == 0:
                continue
            if method >= METHOD_COUNT:
                raise ReaderError(f"{where} slot {slot}: method {method} is out of range")
            if target >= SPECIES_COUNT:
                raise ReaderError(f"{where} slot {slot}: target species {target} is out of range")
            rows.append([method, param, target])
        out[str(sid)] = rows
    return out
