"""Message banks (``msgdata/pl_msg.narc``), decrypted as ``MessageBank_Get`` does."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants, _index
from tools.rom.readers import ReaderError

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles

MSG_NARC = "msgdata/pl_msg.narc"
KEY_START = 596947  # src/message.c:8
KEY_INC = 18749  # src/message.c:9
TERMINATOR = 0xFFFF
COMPRESSED = 0xF100  # tools/msgenc/MessagesDecoder.cpp:138


def decode_bank(data: bytes, charmap: dict[int, str]) -> list[str]:
    """Decode a message bank: header ``u16 count, u16 seed`` and ``count`` ``(u32 offset,
    u32 length)`` entries (``include/message.h:9-19``), offsets from the bank start,
    lengths in u16 characters.

    The entry is XORed with ``seed' | seed' << 16`` where ``seed' = (seed * 765 *
    (i + 1)) & 0xFFFF`` (``src/message.c:16-23``); character ``k`` is XORed with
    ``((i + 1) * 596947 + k * 18749) & 0xFFFF`` (``src/message.c:25-34``). Decoding stops at
    ``0xFFFF``. A code outside the charmap or the ``0xF100`` compressed marker raises.
    """
    if len(data) < 4:
        raise ReaderError(f"message bank: {len(data)} bytes, shorter than its header")
    count, seed = struct.unpack_from("<HH", data, 0)
    if len(data) < 4 + 8 * count:
        raise ReaderError(f"message bank: {len(data)} bytes cannot hold {count} entries")
    out = []
    for i in range(count):
        off, length = struct.unpack_from("<II", data, 4 + 8 * i)
        k = (seed * 765 * (i + 1)) & 0xFFFF
        k |= k << 16
        off ^= k
        length ^= k
        if off + 2 * length > len(data):
            raise ReaderError(f"message bank entry {i}: offset {off} length {length} exceeds {len(data)} bytes")
        key = (i + 1) * KEY_START
        chars = []
        for j in range(length):
            code = struct.unpack_from("<H", data, off + 2 * j)[0] ^ (key & 0xFFFF)
            key = (key + KEY_INC) & 0xFFFF
            if code == TERMINATOR:
                break
            if code == COMPRESSED:
                raise ReaderError(f"message bank entry {i}: compressed string marker {code:#06x}")
            if code not in charmap:
                raise ReaderError(f"message bank entry {i}: code {code:#x} is not in the charmap")
            chars.append(charmap[code])
        out.append("".join(chars))
    return out


def message_bank(rom: RomFiles, decomp: Path, bank: str, charmap: dict[int, str]) -> list[str]:
    """Bank ``bank`` (a ``TEXT_BANK_*`` constant) of ``pl_msg.narc``; the member index is
    the constant's line in ``generated/text_banks.txt`` (``res/text/meson.build:4-8``)."""
    index = _index(_constants(decomp, "text_banks", "TEXT_BANK_"), "text bank")(bank)
    members = rom.narc(MSG_NARC)
    if index >= len(members):
        raise ReaderError(f"{MSG_NARC}: {bank} is member {index} but the archive has {len(members)}")
    try:
        return decode_bank(members[index], charmap)
    except ReaderError as e:
        raise ReaderError(f"{MSG_NARC} member {index} ({bank}): {e}") from e
