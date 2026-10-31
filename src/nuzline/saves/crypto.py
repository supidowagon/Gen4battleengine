"""Gen-4 save crypto: CRC-16, the LCRNG XOR stream and the Pokemon block shuffle."""

from __future__ import annotations

import numpy as np

__all__ = [
    "BLOCK_ORDERS",
    "crc16_ccitt",
    "decrypt_record",
    "encrypt_record",
    "lcrng_xor",
    "record_checksum",
]

# For shuffle value v = (personality & 0x3E000) >> 13: the stored position of
# blocks A, B, C, D. Transcribed from BoxPokemon_GetDataBlock
# (external/pokeplatinum/src/pokemon.c:4861); cases 24-31 repeat cases 0-7.
BLOCK_ORDERS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (0, 2, 3, 1),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (2, 0, 1, 3),
    (3, 0, 1, 2),
    (2, 0, 3, 1),
    (3, 0, 2, 1),
    (1, 2, 0, 3),
    (1, 3, 0, 2),
    (2, 1, 0, 3),
    (3, 1, 0, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (1, 2, 3, 0),
    (1, 3, 2, 0),
    (2, 1, 3, 0),
    (3, 1, 2, 0),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
) + (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (0, 2, 3, 1),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
)

_HEADER = 8
_BLOCK = 32
_BLOCKS_END = _HEADER + 4 * _BLOCK  # 136
_PARTY_END = 236


def _crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        table.append(c)
    return tuple(table)


_CRC_TABLE = _crc_table()


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no xorout."""
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ b]
    return crc


def lcrng_xor(data: bytes, seed: int) -> bytes:
    """XOR each little-endian u16 of ``data`` with the next LCRNG high half."""
    if len(data) % 2:
        raise ValueError(f"length must be even, got {len(data)}")
    n = len(data) // 2
    stream = np.empty(n, dtype="<u2")
    for i in range(n):
        seed = (seed * 1103515245 + 24691) & 0xFFFFFFFF
        stream[i] = seed >> 16
    words = np.frombuffer(data, dtype="<u2") ^ stream
    return words.astype("<u2").tobytes()


def record_checksum(blocks: bytes) -> int:
    """u16 sum of the 64 u16s of 128 decrypted block bytes (order-independent)."""
    if len(blocks) != 4 * _BLOCK:
        raise ValueError(f"expected {4 * _BLOCK} block bytes, got {len(blocks)}")
    return int(np.frombuffer(blocks, dtype="<u2").sum(dtype=np.uint64)) & 0xFFFF


def _check_len(rec: bytes) -> None:
    if len(rec) not in (_BLOCKS_END, _PARTY_END):
        raise ValueError(f"record must be 136 or 236 bytes, got {len(rec)}")


def _order(rec: bytes) -> tuple[int, int, int, int]:
    personality = int.from_bytes(rec[0:4], "little")
    return BLOCK_ORDERS[(personality & 0x3E000) >> 13]


def decrypt_record(raw: bytes) -> bytes:
    """Decrypt a 136/236-byte record and reorder its blocks to A, B, C, D.

    The header is unchanged; the checksum is *not* verified here.
    """
    _check_len(raw)
    order = _order(raw)
    checksum = int.from_bytes(raw[6:8], "little")
    stored = lcrng_xor(raw[_HEADER:_BLOCKS_END], checksum)
    blocks = b"".join(stored[p * _BLOCK : (p + 1) * _BLOCK] for p in order)
    out = raw[:_HEADER] + blocks
    if len(raw) == _PARTY_END:
        personality = int.from_bytes(raw[0:4], "little")
        out += lcrng_xor(raw[_BLOCKS_END:], personality)
    return out


def encrypt_record(plain: bytes) -> bytes:
    """Exact inverse of :func:`decrypt_record` (uses the checksum in bytes 6-7)."""
    _check_len(plain)
    order = _order(plain)
    checksum = int.from_bytes(plain[6:8], "little")
    stored = [b""] * 4
    for block, pos in enumerate(order):
        stored[pos] = plain[_HEADER + block * _BLOCK : _HEADER + (block + 1) * _BLOCK]
    out = plain[:_HEADER] + lcrng_xor(b"".join(stored), checksum)
    if len(plain) == _PARTY_END:
        personality = int.from_bytes(plain[0:4], "little")
        out += lcrng_xor(plain[_BLOCKS_END:], personality)
    return out
