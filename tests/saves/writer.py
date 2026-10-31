"""Test-only writer for synthetic Pokemon records (the inverse of ``decode_mon``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nuzline.data.tables import load_tables
from nuzline.saves.crypto import encrypt_record, record_checksum

_TABLES_DIR = Path(__file__).resolve().parents[2] / "data" / "gen4_platinum"
# ids.Stat order -> stored order (HP, Atk, Def, Spe, SpA, SpD)
_STAT_TO_STORED = (0, 1, 2, 5, 3, 4)


@lru_cache(maxsize=1)
def _tables():
    return load_tables(_TABLES_DIR)


def personality_for(nature: int, shuffle: int) -> int:
    """A personality with ``% 25 == nature`` and block-order index ``shuffle`` (0-31)."""
    for k in range(25):
        p = (k << 18) | (shuffle << 13)
        if p % 25 == nature:
            return p
    raise ValueError(f"no personality for nature {nature}, shuffle {shuffle}")


def mon_record(
    *,
    species,
    personality=0,
    ot_id=0,
    exp=None,
    level=5,
    nature=None,
    ability=0,
    ivs=(0,) * 6,
    evs=(0,) * 6,
    moves=(1, 0, 0, 0),
    pp=(35, 0, 0, 0),
    pp_ups=(0,) * 4,
    item=0,
    friendship=70,
    gender=0,
    form=0,
    is_egg=False,
    nickname_codes=(0xFFFF,),
    party=None,
) -> bytes:
    """An encrypted 136-byte record, or 236 bytes when ``party`` is a dict with
    ``status``, ``level``, ``hp`` and ``stats`` (ids.Stat order, max HP first).

    ``exp=None`` means the experience of ``level``. ``nature`` is not applied: the
    caller picks ``personality`` (see :func:`personality_for`).
    """
    if exp is None:
        t = _tables()
        exp = int(t.exp_table[int(t.species_exp_rate[species]), level])
    rec = bytearray(236 if party else 136)
    rec[0:4] = personality.to_bytes(4, "little")
    a, b, c = 8, 40, 72
    rec[a : a + 2] = species.to_bytes(2, "little")
    rec[a + 2 : a + 4] = item.to_bytes(2, "little")
    rec[a + 4 : a + 8] = ot_id.to_bytes(4, "little")
    rec[a + 8 : a + 12] = exp.to_bytes(4, "little")
    rec[a + 12] = friendship
    rec[a + 13] = ability
    for i, j in enumerate(_STAT_TO_STORED):
        rec[a + 0x10 + i] = evs[j]
    for i, m in enumerate(moves):
        rec[b + 2 * i : b + 2 * i + 2] = m.to_bytes(2, "little")
    rec[b + 8 : b + 12] = bytes(pp)
    rec[b + 12 : b + 16] = bytes(pp_ups)
    word = sum(ivs[j] << (5 * i) for i, j in enumerate(_STAT_TO_STORED))
    word |= (int(is_egg) << 30) | (int(tuple(nickname_codes) != (0xFFFF,)) << 31)
    rec[b + 0x10 : b + 0x14] = word.to_bytes(4, "little")
    rec[b + 0x18] = (gender << 1) | (form << 3)
    codes = list(nickname_codes) + [0xFFFF] * (11 - len(nickname_codes))
    for i, code in enumerate(codes[:11]):
        rec[c + 2 * i : c + 2 * i + 2] = code.to_bytes(2, "little")
    if party:
        rec[136:140] = party["status"].to_bytes(4, "little")
        rec[140] = party["level"]
        rec[142:144] = party["hp"].to_bytes(2, "little")
        for i, j in enumerate(_STAT_TO_STORED):
            rec[144 + 2 * i : 146 + 2 * i] = party["stats"][j].to_bytes(2, "little")
    rec[6:8] = record_checksum(bytes(rec[8:136])).to_bytes(2, "little")
    return encrypt_record(bytes(rec))
