"""Decode one Pokemon record into a :class:`SaveMon`.

Layout: ``include/struct_defs/pokemon.h`` (blocks A-D, 32 bytes each, after the 8-byte
header; ``PartyPokemon`` extension after the 0x88-byte ``BoxPokemon``). Nature is
``personality % 25`` (``Pokemon_GetNatureOf``, ``src/pokemon.c:2410``); the level of a
boxed Pokemon is the highest level whose experience entry is <= its experience
(``SpeciesData_GetLevelAt``, ``src/pokemon.c:2378``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.engine.ids import Stat, Status
from nuzline.engine.stats import calc_stats
from nuzline.saves.crypto import decrypt_record, record_checksum
from nuzline.saves.status import decode_status

__all__ = ["RecordError", "SaveMon", "decode_mon", "decode_text", "level_from_exp"]

# Offsets in the decrypted, A-B-C-D ordered record (header 8 bytes, blocks 32 each).
_A, _B, _C = 8, 40, 72
_PARTY = 136
# Stored order of EV/IV/stat fields is HP, Atk, Def, Spe, SpA, SpD; ids.Stat order is
# HP, Atk, Def, SpA, SpD, Spe.
_STORED_TO_STAT = (0, 1, 2, 4, 5, 3)
# Charmap control codes (not glyphs): newline, carriage return, form feed.
_CONTROL = ("\\n", "\\r", "\\f")


class RecordError(ValueError):
    """A record that cannot be trusted (bad checksum, impossible level, ...)."""


@dataclass(frozen=True)
class SaveMon:
    location: tuple[str, int, int]  # ("party", 0, slot) or ("box", box, slot)
    personality: int
    ot_id: int
    species: int  # decomp ids throughout
    form: int
    nickname: str
    is_egg: bool
    experience: int
    level: int  # stored (party) or computed from experience (box)
    nature: int
    ability: int  # decomp ability id as stored
    gender: int  # 0 male, 1 female, 2 none (PokemonSpec convention)
    friendship: int
    item: int
    ivs: tuple[int, ...]  # ids.Stat order
    evs: tuple[int, ...]  # ids.Stat order
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]
    pp_ups: tuple[int, int, int, int]
    current_hp: int | None  # party only; None in a box
    status_raw: int  # MON_CONDITION_* word; 0 in a box
    stored_stats: tuple[int, ...] | None  # party only, ids.Stat order, max HP first

    def to_spec(self, tables) -> PokemonSpec:
        """The engine's :class:`PokemonSpec`. Raises :class:`RecordError` for an egg, an
        alternate form that changes stats/types/abilities, an ability the species
        cannot have, PP above the move's maximum, or (party) a stored stat outside the
        band the current EVs allow."""
        label = _label(self.location)
        if self.is_egg:
            raise RecordError(f"{label}: is an egg")
        if self.form != 0 and self.species in _REFUSED_FORM_SPECIES:
            raise RecordError(f"{label}: species {self.species} form {self.form} is unsupported")
        abilities = [int(a) for a in _as_np(tables.species_abilities_decomp)[self.species]]
        if self.ability == 0 or self.ability not in abilities:
            raise RecordError(
                f"{label}: ability {self.ability} is not one of the species' {abilities}"
            )
        move_pp = _as_np(tables.move_pp)
        for i, move in enumerate(self.moves):
            if move == 0:
                continue
            limit = int(move_pp[move]) * (5 + self.pp_ups[i]) // 5
            if self.pp[i] > limit:
                raise RecordError(
                    f"{label}: move slot {i} has PP {self.pp[i]} above the maximum {limit}"
                )
        if self.stored_stats is not None:
            self._check_stats(tables, label)
            if self.current_hp is not None and self.current_hp > self.stored_stats[Stat.HP]:
                raise RecordError(
                    f"{label}: current HP {self.current_hp} exceeds max HP "
                    f"{self.stored_stats[Stat.HP]}"
                )
        status, counter = decode_status(self.status_raw)
        return PokemonSpec(
            species=self.species,
            level=self.level,
            nature=self.nature,
            ivs=self.ivs,
            evs=self.evs,
            moves=self.moves,
            item=self.item,
            ability_slot=abilities.index(self.ability),
            gender=self.gender,
            friendship=self.friendship,
            form=self.form,
            current_hp=self.current_hp,
            status=int(status),
            status_counter=0 if status == Status.TOXIC else counter,
            pp=self.pp,
            stats=self.stored_stats,
        )

    def _check_stats(self, tables, label: str) -> None:
        """Each stored stat lies between the stats with 0 EVs and with the current EVs
        (same IVs, nature and level): the game recomputes stats at level-up from an
        EV vector no larger than today's."""
        base = _as_np(tables.species_base_stats)[self.species]
        mods = _as_np(tables.nature_mods)[self.nature]
        low = calc_stats(base, self.ivs, (0,) * 6, self.level, mods)
        high = calc_stats(base, self.ivs, self.evs, self.level, mods)
        for stat in (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE):
            stored = self.stored_stats[stat]
            if self.species == _SHEDINJA and stat == Stat.HP:
                if stored != 1:  # Pokemon_CalcStats (src/pokemon.c:598) pins max HP at 1
                    raise RecordError(f"{label}: Shedinja HP {stored} must be 1")
                continue
            if not low[stat] <= stored <= high[stat]:
                raise RecordError(
                    f"{label}: {stat.name.capitalize()} {stored} is outside "
                    f"[{int(low[stat])}, {int(high[stat])}] for level {self.level}"
                )


_np_cache: dict[int, tuple[object, np.ndarray]] = {}


def _as_np(arr) -> np.ndarray:
    """numpy view of a table, converted once per array (jax indexing is slow)."""
    if isinstance(arr, np.ndarray):
        return arr
    hit = _np_cache.get(id(arr))
    if hit is None or hit[0] is not arr:
        hit = (arr, np.asarray(arr))
        _np_cache[id(arr)] = hit
    return hit[1]


def level_from_exp(tables, species: int, exp: int) -> int:
    """Highest level (1-100) whose experience entry is <= ``exp``."""
    rates = _as_np(tables.species_exp_rate)
    if not 0 <= species < len(rates):
        raise RecordError(f"species {species} out of range")
    row = _as_np(tables.exp_table)[int(rates[species])]
    return max(1, int(np.searchsorted(row[1:], exp, side="right")))


# Species whose non-zero forms change stats, types or abilities (plan Ruling 2), as decomp
# ids (line N of generated/species.txt is id N-1): Deoxys 386, Wormadam 413, Rotom 479,
# Giratina 487, Shaymin 492, Arceus 493. Burmy, Shellos, Gastrodon, Unown and Cherrim
# forms are cosmetic and pass.
_SHEDINJA = 292
_REFUSED_FORM_SPECIES = frozenset({386, 413, 479, 487, 492, 493})


def _label(location: tuple[str, int, int]) -> str:
    kind, box, slot = location
    return f"party slot {slot}" if kind == "party" else f"box {box} slot {slot}"


def _u16(b: bytes, o: int) -> int:
    return int.from_bytes(b[o : o + 2], "little")


def _u32(b: bytes, o: int) -> int:
    return int.from_bytes(b[o : o + 4], "little")


def decode_text(buf: bytes, offset: int, max_chars: int, charmap: dict[int, str]) -> str:
    """Up to ``max_chars`` u16 text codes at ``offset``, stopping at ``0xFFFF``
    (``CHAR_EOS``). Unmapped codes become ``?``; control codes are dropped."""
    out = []
    for i in range(max_chars):
        code = _u16(buf, offset + 2 * i)
        if code == 0xFFFF:
            break
        ch = charmap.get(code, "?")
        out.append("" if ch in _CONTROL else ch)
    return "".join(out)


def decode_mon(
    raw: bytes,
    location: tuple[str, int, int],
    tables,
    charmap: dict[int, str],
) -> SaveMon | None:
    """Decode a 136/236-byte record; ``None`` for an empty slot.

    Empty means an all-zero box part, or a checksum-valid record with species 0.
    """
    label = _label(location)
    if not any(raw[:_PARTY]):
        return None  # all-zero box part: an empty slot, never decrypted
    rec = decrypt_record(raw)
    if record_checksum(rec[8:_PARTY]) != _u16(rec, 6):
        raise RecordError(f"{label}: checksum mismatch")
    species = _u16(rec, _A)
    if species == 0:
        return None

    personality = _u32(rec, 0)
    exp = _u32(rec, _A + 0x08)
    iv_word = _u32(rec, _B + 0x10)
    ivs_stored = [(iv_word >> (5 * i)) & 31 for i in range(6)]
    evs_stored = [rec[_A + 0x10 + i] for i in range(6)]
    form_byte = rec[_B + 0x18]

    try:
        level = level_from_exp(tables, species, exp)
    except RecordError as e:
        raise RecordError(f"{label}: {e}") from None

    moves = tuple(_u16(rec, _B + 2 * i) for i in range(4))
    item = _u16(rec, _A + 0x02)
    for i, move in enumerate(moves):
        if move >= len(tables.move_pp):
            raise RecordError(f"{label}: move slot {i} has unknown move id {move}")
    if item >= len(tables.item_effect):
        raise RecordError(f"{label}: unknown item id {item}")

    current_hp = None
    status_raw = 0
    stored_stats = None
    if len(rec) == 236:
        status_raw = _u32(rec, _PARTY)
        party_level = rec[_PARTY + 4]
        if party_level != level:
            raise RecordError(
                f"{label}: stored level {party_level} disagrees with level {level} "
                f"from experience {exp}"
            )
        current_hp = _u16(rec, _PARTY + 6)
        s = [_u16(rec, _PARTY + 8 + 2 * i) for i in range(6)]
        stored_stats = tuple(s[i] for i in _STORED_TO_STAT)

    return SaveMon(
        location=location,
        personality=personality,
        ot_id=_u32(rec, _A + 0x04),
        species=species,
        form=form_byte >> 3,
        nickname=decode_text(rec, _C, 11, charmap),
        is_egg=bool((iv_word >> 30) & 1),
        experience=exp,
        level=level,
        nature=personality % 25,
        ability=rec[_A + 0x0D],
        gender=(form_byte >> 1) & 3,
        friendship=rec[_A + 0x0C],
        item=item,
        ivs=tuple(ivs_stored[i] for i in _STORED_TO_STAT),
        evs=tuple(evs_stored[i] for i in _STORED_TO_STAT),
        moves=moves,
        pp=tuple(rec[_B + 0x08 + i] for i in range(4)),
        pp_ups=tuple(rec[_B + 0x0C + i] for i in range(4)),
        current_hp=current_hp,
        status_raw=status_raw,
        stored_stats=stored_stats,
    )
