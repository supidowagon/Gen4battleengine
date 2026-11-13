"""Decode one Pokemon record into a :class:`SaveMon`.

Layout: ``include/struct_defs/pokemon.h`` (blocks A-D, 32 bytes each, after the 8-byte
header; ``PartyPokemon`` extension after the 0x88-byte ``BoxPokemon``). Nature is
``personality % 25`` (``Pokemon_GetNatureOf``, ``src/pokemon.c:2410``); the level of a
boxed Pokemon is the highest level whose experience entry is <= its experience
(``SpeciesData_GetLevelAt``, ``src/pokemon.c:2378``).

The exp fields (K3d-exp) come from blocks A, B and D (pokemon.h:26, 69, 89, 99, 102-105):
the origin language, the OT's raw name codes and gender, the Pokerus byte, the ball and
the egg location (``MON_DATA_EGG_LOCATION``'s rule, ``src/pokemon.c:1159-1166``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nuzline.data.forms import form_row
from nuzline.data.spec import PokemonSpec
from nuzline.engine.ids import Stat, Status
from nuzline.engine.stats import SPECIES_SHEDINJA, calc_stats
from nuzline.saves.crypto import decrypt_record, record_checksum
from nuzline.saves.status import decode_status

__all__ = [
    "ITEM_LUXURY_BALL",
    "RecordError",
    "SaveMon",
    "decode_mon",
    "decode_text",
    "level_from_exp",
    "location_label",
]

# Offsets in the decrypted, A-B-C-D ordered record (header 8 bytes, blocks 32 each).
_A, _B, _C, _D = 8, 40, 72, 104
_PARTY = 136
# Stored order of EV/IV/stat fields is HP, Atk, Def, Spe, SpA, SpD; ids.Stat order is
# HP, Atk, Def, SpA, SpD, Spe.
_STORED_TO_STAT = (0, 1, 2, 4, 5, 3)
# Charmap control codes (not glyphs): newline, carriage return, form feed.
_CONTROL = ("\\n", "\\r", "\\f")
CHAR_EOS = 0xFFFF
OT_NAME_LEN = 7  # TRAINER_NAME_LEN, include/constants/string.h:5
# ITEM_LUXURY_BALL (generated/items.txt line 12: id 11); Pokemon_UpdateFriendship's +1
# (src/pokemon.c:2664-2666) tests the ball the Pokemon was caught in.
ITEM_LUXURY_BALL = 11
# FATEFUL_ENCOUNTER_LOCATION (src/pokemon.c:59): the DP egg location that defers to the
# PtHGSS field (MON_DATA_EGG_LOCATION, src/pokemon.c:1159-1166).
_FATEFUL_ENCOUNTER_LOCATION = 3002


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
    language: int  # originLanguage (MON_DATA_LANGUAGE), block A 0x0F
    ot_name_codes: tuple[int, ...]  # otName, 8 raw charcodes, block D 0x00
    ot_gender: int  # otGender, block D 0x1C bit 7
    pokerus: int  # the Pokerus byte, block D 0x1A
    pokeball: int  # the ball's item id, block D 0x1B
    egg_location: int  # MON_DATA_EGG_LOCATION (src/pokemon.c:1159-1166)

    def to_spec(self, tables, trainer=None) -> PokemonSpec:
        """The engine's :class:`PokemonSpec`. Raises :class:`RecordError` for an egg, an
        Arceus form (its plate type; Multitype is unsupported), a form beyond the species'
        form count, an ability the form cannot have, PP above the move's maximum, or
        (party) a stored stat outside the band the current EVs allow. Base stats and
        abilities are the form's personal row (``nuzline.data.forms.form_row``).

        ``trainer`` (the save's ``SaveTrainer``) resolves ``exp_boost`` (``exp_boost``);
        without it ``exp_boost`` is 0, unresolved, which coverage refuses for a Pokemon
        that can gain exp. ``exp`` is the stored experience, ``pokerus`` 1 for a nonzero
        Pokerus byte (``Pokemon_HasPokerus``, src/pokemon.c:4184-4199) and
        ``friend_bonus`` 1 for a Luxury Ball (src/pokemon.c:2664-2666). ``egg_location`` and
        ``pp_ups`` are copied for coverage (``PokemonSpec``)."""
        label = location_label(self.location)
        if self.is_egg:
            raise RecordError(f"{label}: is an egg")
        if self.form != 0 and self.species in _REFUSED_FORM_SPECIES:
            raise RecordError(
                f"{label}: Arceus form {self.form} is unsupported (Multitype, plan Ruling 8)"
            )
        row = self._row(tables, label)
        abilities = [int(a) for a in _as_np(tables.species_abilities_decomp)[row]]
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
            self._check_stats(tables, label, row)
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
            exp=self.experience,
            exp_boost=0 if trainer is None else self.exp_boost(trainer),
            pokerus=int(self.pokerus != 0),
            friend_bonus=int(self.pokeball == ITEM_LUXURY_BALL),
            egg_location=self.egg_location,
            pp_ups=self.pp_ups,
        )

    def is_ot(self, trainer) -> bool:
        """``BattleSystem_PokemonIsOT`` (src/battle/battle_lib.c:6286-6303): the full u32
        trainer id, the gender and the name all match. The name test is
        ``CharCode_CompareNumChars(trName, monOTName, TRAINER_NAME_LEN)`` on raw codes,
        where ``monOTName`` is ``MON_DATA_OT_NAME``: the first 7 codes of ``otName`` and
        an ``EOS`` (src/pokemon.c:1121-1128)."""
        mon_name = (*self.ot_name_codes[:OT_NAME_LEN], CHAR_EOS)
        return (
            trainer.id == self.ot_id
            and trainer.gender == self.ot_gender
            and _compare_num_chars(trainer.name_codes, mon_name, OT_NAME_LEN)
        )

    def exp_boost(self, trainer) -> int:
        """The exp multiplier in percent (``BattleScript_GetExpTask``,
        src/battle/battle_script.c:9757-9763): 100 for the OT (``is_ot``); else 170 when
        ``MON_DATA_LANGUAGE`` differs from the ROM's ``gGameLanguage``
        (``trainer.rom_language``, never the save's trainer-language byte), else 150."""
        if self.is_ot(trainer):
            return 100
        return 170 if self.language != trainer.rom_language else 150

    def _row(self, tables, label: str) -> int:
        """The personal row of this species' form; ``RecordError`` naming the slot when the
        species has no such form (a corrupt record)."""
        try:
            return form_row(_as_np(tables.species_form_row), self.species, self.form)
        except ValueError as e:
            raise RecordError(f"{label}: {e}") from None

    def _check_stats(self, tables, label: str, row: int) -> None:
        """Each stored stat lies between the stats with 0 EVs and with the current EVs
        (same IVs, nature and level): the game recomputes stats at level-up from an
        EV vector no larger than today's."""
        base = _as_np(tables.species_base_stats)[row]
        mods = _as_np(tables.nature_mods)[self.nature]
        low = calc_stats(base, self.ivs, (0,) * 6, self.level, mods, self.species)
        high = calc_stats(base, self.ivs, self.evs, self.level, mods, self.species)
        for stat in (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE):
            stored = self.stored_stats[stat]
            if self.species == SPECIES_SHEDINJA and stat == Stat.HP:
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
    # species_exp_rate has a row per personal row (forms too); the species ids are the
    # rows of the form grid. Indexed by species, not form row, as the game does
    # (Pokemon_GetSpeciesLevelAt, src/pokemon.c:2368, loads the species), and every form
    # has its species' rate.
    if not 0 <= species < len(_as_np(tables.species_form_row)):
        raise RecordError(f"species {species} out of range")
    row = _as_np(tables.exp_table)[int(rates[species])]
    return max(1, int(np.searchsorted(row[1:], exp, side="right")))


# Species whose non-zero forms the save parser refuses (plan K2a Ruling 8), as decomp ids
# (line N of generated/species.txt is id N-1): Arceus 493, whose form is its plate type
# and whose Multitype the engine does not model. Deoxys, Wormadam, Rotom, Giratina and
# Shaymin forms convert through their personal row (coverage refuses Giratina-Origin
# without the Griseous Orb and Shaymin-Sky); Burmy, Shellos, Gastrodon, Unown and Cherrim
# forms are cosmetic (their row is the species') and pass.
_REFUSED_FORM_SPECIES = frozenset({493})


def location_label(location: tuple[str, int, int]) -> str:
    """``party slot S`` or ``box B slot S``: ``location``'s box index is 0-based, the label's
    box number 1-based like ``save show`` and ``--exclude box:B:S``."""
    kind, box, slot = location
    return f"party slot {slot}" if kind == "party" else f"box {box + 1} slot {slot}"


def _compare_num_chars(a, b, num: int) -> bool:
    """``CharCode_CompareNumChars(a, b, num) == 0`` (src/charcode_util.c:87-104), line for
    line: equal when the codes agree up to a shared ``EOS``, or through ``num + 1``
    codes."""
    i = 0
    while a[i] == b[i]:
        if num == 0:
            return True
        if a[i] == CHAR_EOS and b[i] == CHAR_EOS:
            return True
        num -= 1
        i += 1
    return False


def _egg_location(rec: bytes) -> int:
    """``MON_DATA_EGG_LOCATION`` (src/pokemon.c:1159-1166): the PtHGSS field (block B
    0x1C) when the DP field (block D 0x16) is the fateful-encounter location and the
    PtHGSS field is nonzero, else the DP field."""
    dp, pt = _u16(rec, _D + 0x16), _u16(rec, _B + 0x1C)
    return pt if dp == _FATEFUL_ENCOUNTER_LOCATION and pt else dp


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
    label = location_label(location)
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
        language=rec[_A + 0x0F],
        ot_name_codes=tuple(_u16(rec, _D + 2 * i) for i in range(OT_NAME_LEN + 1)),
        ot_gender=rec[_D + 0x1C] >> 7,
        pokerus=rec[_D + 0x1A],
        pokeball=rec[_D + 0x1B],
        egg_location=_egg_location(rec),
    )
