"""The trainer -> map join's readers for shared script files (tools/rom/readers/maps.py):
``map_script_ids`` (bg, object and coord event scripts that run the map's own file) and
``init_script_ids`` (the init-scripts member, src/script_manager.c:738-806). Synthetic
bytes; no ROM needed."""

import struct

import pytest

from tools.rom.readers import ReaderError
from tools.rom.readers.maps import EVENT_RECORD_SIZES, init_script_ids, map_script_ids


def events(bg=(), objects=(), warps=0, coords=()):
    """A zone_event member: bg (script at +0), object (script at +10), warp, coord (+0)."""
    out = b""
    for kind, scripts in enumerate((bg, objects, [None] * warps, coords)):
        size = EVENT_RECORD_SIZES[kind]
        out += struct.pack("<I", len(scripts))
        for s in scripts:
            rec = bytearray(size)
            if s is not None:
                struct.pack_into("<H", rec, 10 if kind == 1 else 0, s)
            out += bytes(rec)
    return out


def test_map_script_ids_keeps_local_entries_of_every_event_kind():
    member = events(bg=[25, 2500], objects=[3001, 16, 0], warps=2, coords=[9, 2000])
    # 2500 (bg range), 3001 (single battle), 0 (no script), 2000 (common) are not local
    assert map_script_ids(member, 0) == [25, 16, 9]


def test_init_script_ids_fixed_and_frame_table():
    # ON_TRANSITION (2) id 7; ON_FRAME_TABLE (1) with a u32 offset counted from the end of
    # that field (byte 10); ON_LOAD (4) id 3; terminator. The table's rows (var, compare,
    # script) start 5 padding bytes after the 16-byte head, so the offset is 11.
    head = (
        bytes([2]) + struct.pack("<H", 7) + b"\0\0"
        + bytes([1]) + struct.pack("<I", 11)
        + bytes([4]) + struct.pack("<H", 3) + b"\0\0"
        + b"\0"
    )  # fmt: skip
    table = struct.pack("<3H", 0x4001, 0x4002, 11) + struct.pack("<3H", 0x4003, 1, 12)
    assert init_script_ids(head + b"\0" * 5 + table + b"\0\0", 0) == [7, 11, 12, 3]
    # A row whose first byte is 0 ends the table (GET8 == 0, script_manager.c:783):
    stop = struct.pack("<3H", 0x4100, 0x4002, 13)
    assert init_script_ids(head + b"\0" * 5 + stop + table, 0) == [7, 3]


def test_init_script_ids_empty_and_truncated():
    assert init_script_ids(b"\0", 0) == []
    assert init_script_ids(b"", 0) == []
    with pytest.raises(ReaderError, match="truncated"):
        init_script_ids(bytes([2, 7]), 0)
