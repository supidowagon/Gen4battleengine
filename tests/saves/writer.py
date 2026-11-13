"""Test-only writer for synthetic Pokemon records and whole save images (the inverse of
``decode_mon`` and ``load_save``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nuzline.data.tables import load_tables
from nuzline.saves import layout as L
from nuzline.saves.crypto import crc16_ccitt, encrypt_record, record_checksum

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
    language=2,
    ot_name_codes=(0xFFFF,),
    ot_gender=0,
    pokerus=0,
    pokeball=4,
    egg_location_dp=0,
    egg_location_pt=0,
) -> bytes:
    """An encrypted 136-byte record, or 236 bytes when ``party`` is a dict with
    ``status``, ``level``, ``hp`` and ``stats`` (ids.Stat order, max HP first).

    ``exp=None`` means the experience of ``level``. ``nature`` is not applied: the
    caller picks ``personality`` (see :func:`personality_for`). ``ot_name_codes`` is
    padded with ``EOS`` to 8 codes; with the defaults (``ot_id`` 0, ``ot_gender`` 0, an
    empty name, English) the record's OT is ``build_image``'s default trainer.
    """
    if exp is None:
        t = _tables()
        exp = int(t.exp_table[int(t.species_exp_rate[species]), level])
    rec = bytearray(236 if party is not None else 136)
    rec[0:4] = personality.to_bytes(4, "little")
    a, b, c, d = 8, 40, 72, 104
    rec[a : a + 2] = species.to_bytes(2, "little")
    rec[a + 2 : a + 4] = item.to_bytes(2, "little")
    rec[a + 4 : a + 8] = ot_id.to_bytes(4, "little")
    rec[a + 8 : a + 12] = exp.to_bytes(4, "little")
    rec[a + 12] = friendship
    rec[a + 13] = ability
    rec[a + 0x0F] = language
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
    rec[b + 0x1C : b + 0x1E] = egg_location_pt.to_bytes(2, "little")
    ot = list(ot_name_codes) + [0xFFFF] * (8 - len(ot_name_codes))
    for i, code in enumerate(ot[:8]):
        rec[d + 2 * i : d + 2 * i + 2] = code.to_bytes(2, "little")
    rec[d + 0x16 : d + 0x18] = egg_location_dp.to_bytes(2, "little")
    rec[d + 0x1A] = pokerus
    rec[d + 0x1B] = pokeball
    rec[d + 0x1C] = ot_gender << 7
    if party is not None:
        rec[136:140] = party["status"].to_bytes(4, "little")
        rec[140] = party["level"]
        rec[142:144] = party["hp"].to_bytes(2, "little")
        for i, j in enumerate(_STAT_TO_STORED):
            rec[144 + 2 * i : 146 + 2 * i] = party["stats"][j].to_bytes(2, "little")
    rec[6:8] = record_checksum(bytes(rec[8:136])).to_bytes(2, "little")
    return encrypt_record(bytes(rec))


_BLOCKS = {"normal": (0, L.NORMAL_SIZE), "boxes": (L.BOXES_OFFSET, L.BOXES_SIZE)}


def block_offset(copy: int, block: str) -> int:
    """Image offset of ``block`` ("normal"/"boxes") in copy 0 (primary) or 1 (backup)."""
    return L.COPY_OFFSETS[copy] + _BLOCKS[block][0]


def footer_offset(copy: int, block: str) -> int:
    return block_offset(copy, block) + _BLOCKS[block][1] - L.FOOTER_SIZE


def _footer(body: bytes, block: str, save_counter: int, block_counter: int) -> bytes:
    f = bytearray(L.FOOTER_SIZE)
    f[0:4] = save_counter.to_bytes(4, "little")
    f[4:8] = block_counter.to_bytes(4, "little")
    f[8:12] = _BLOCKS[block][1].to_bytes(4, "little")
    f[12:16] = L.SIGNATURE.to_bytes(4, "little")
    f[16] = L.BLOCK_IDS[block]
    f[18:20] = crc16_ccitt(body).to_bytes(2, "little")
    return bytes(f)


def build_image(
    *,
    party=(),
    boxes=None,
    bag=None,
    trainer_name_codes=(),
    tid=0,
    sid=0,
    trainer_gender=0,
    trainer_language=2,
    badges=0,
    counters=((1, 1, 1), None),
    party_count=None,
    vars=None,
) -> bytes:
    """A 0x80000 Platinum image with valid footers.

    ``party``: encrypted 236-byte records; ``boxes``: ``{(box, slot): 136-byte record}``;
    ``bag``: ``{pocket name: [(item, quantity), ...]}``. ``counters[i]`` is
    ``(save counter, normal block counter, boxes block counter)`` for copy ``i``, or
    ``None`` for an erased (0xFF) copy. ``party_count`` overrides ``currentCount``.
    ``vars``: ``{index into VarsFlags.vars: u16 value}``.
    """
    normal = bytearray(L.NORMAL_SIZE - L.FOOTER_SIZE)
    codes = list(trainer_name_codes) + [0xFFFF] * (L.TRAINER_NAME_LEN - len(trainer_name_codes))
    for i, code in enumerate(codes[: L.TRAINER_NAME_LEN]):
        normal[L.TRAINER_NAME + 2 * i : L.TRAINER_NAME + 2 * i + 2] = code.to_bytes(2, "little")
    normal[L.TRAINER_ID : L.TRAINER_ID + 4] = (tid | (sid << 16)).to_bytes(4, "little")
    normal[L.TRAINER_GENDER] = trainer_gender
    normal[L.TRAINER_GENDER + 1] = trainer_language  # TrainerInfo.language
    normal[L.BADGES] = badges
    normal[L.PARTY_COUNT - 4 : L.PARTY_COUNT] = L.PARTY_MAX.to_bytes(4, "little")
    count = len(party) if party_count is None else party_count
    normal[L.PARTY_COUNT : L.PARTY_COUNT + 4] = count.to_bytes(4, "little")
    for i, rec in enumerate(party):
        o = L.PARTY_MONS + i * L.PARTY_MON_SIZE
        normal[o : o + L.PARTY_MON_SIZE] = rec
    o = L.BAG
    for name, size in L.POCKETS:
        for j, (item, qty) in enumerate((bag or {}).get(name, ())):
            assert j < size
            normal[o + 4 * j : o + 4 * j + 2] = item.to_bytes(2, "little")
            normal[o + 4 * j + 2 : o + 4 * j + 4] = qty.to_bytes(2, "little")
        o += 4 * size
    for index, value in (vars or {}).items():
        normal[L.VARS + 2 * index : L.VARS + 2 * index + 2] = value.to_bytes(2, "little")

    box_body = bytearray(L.BOXES_SIZE - L.FOOTER_SIZE)
    for (b, s), rec in (boxes or {}).items():
        o = L.BOX_MONS + (b * L.BOX_SLOTS + s) * L.BOX_MON_SIZE
        box_body[o : o + L.BOX_MON_SIZE] = rec

    image = bytearray([0xFF]) * L.SAVE_SIZE
    for copy, ctr in enumerate(counters):
        if ctr is None:
            continue
        save_counter, normal_counter, boxes_counter = ctr
        for block, body, bc in (
            ("normal", normal, normal_counter),
            ("boxes", box_body, boxes_counter),
        ):
            o = block_offset(copy, block)
            image[o : o + len(body)] = body
            image[o + len(body) : o + len(body) + L.FOOTER_SIZE] = _footer(
                bytes(body), block, save_counter, bc
            )
    return bytes(image)


def corrupt(image: bytes, copy: int, block: str) -> bytes:
    """``image`` with one body byte of ``block`` in ``copy`` flipped (checksum now wrong)."""
    out = bytearray(image)
    out[block_offset(copy, block)] ^= 0x01
    return bytes(out)


def set_save_counter(image: bytes, copy: int, block: str, counter: int) -> bytes:
    """Rewrite one footer's save counter (outside the checksummed body: stays valid)."""
    out = bytearray(image)
    o = footer_offset(copy, block)
    out[o : o + 4] = counter.to_bytes(4, "little")
    return bytes(out)
