"""The trainer -> map join's readers for shared script files (tools/rom/readers/maps.py):
``map_script_ids`` (bg, object and coord event scripts that run the map's own file) and
``init_script_ids`` (the init-scripts member, src/script_manager.c:738-806). Synthetic
bytes; no ROM needed."""

import struct

import pytest

from tools.rom.readers import ReaderError
from tools.rom.readers.maps import (
    EVENT_RECORD_SIZES,
    command_table,
    header_battles,
    init_script_ids,
    join_trainer_maps,
    map_script_ids,
)


def events_member(bg=(), objects=(), warps=0, coords=()):
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
    member = events_member(bg=[25, 2500], objects=[3001, 16, 0], warps=2, coords=[9, 2000])
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


# --- the join itself (join_trainer_maps, header_battles), on synthetic script files ---------


@pytest.fixture(scope="module")
def table(decomp):
    return command_table(decomp)


def _cmd(table, name, *operands):
    op = next(op for op, (n, _) in table.items() if n == name)
    widths = table[op][1]
    assert len(widths) == len(operands), name
    fmt = {1: "B", 2: "H", 4: "i"}
    return struct.pack("<H", op) + b"".join(
        struct.pack("<" + fmt[w], v) for w, v in zip(widths, operands, strict=True)
    )


def script_file(table, *bodies):
    """A script file whose entry i (1-based) runs ``bodies[i - 1]`` (bytes) then End."""
    end = _cmd(table, "End")
    size = 4 * len(bodies) + 2
    code, starts = b"", []
    for body in bodies:
        starts.append(size + len(code))
        code += body + end
    head = b"".join(struct.pack("<i", s - (4 * i + 4)) for i, s in enumerate(starts))
    return head + struct.pack("<H", 0xFD13) + code


def battle(table, trainer):
    return _cmd(table, "StartTrainerBattle", trainer, 0)


def test_join_shared_file_per_entry_and_fallbacks(table):
    nothing = b""
    shared = script_file(
        table,
        battle(table, 10),  # entry 1
        *[nothing] * 7,  # entries 2-8: the C-started ids, empty here
        battle(table, 20),  # entry 9
        battle(table, 30),  # entry 10
        _cmd(table, "ScrCmd_Unused_013", 1),  # entry 11: starts a script by id
    )
    alone = script_file(table, battle(table, 80), battle(table, 90))
    empty_init = b"\0"
    scripts = [shared, alone, empty_init]
    events = [
        events_member(coords=[9]),  # 0: runs entry 9 only
        events_member(),  # 1: runs nothing
        events_member(coords=[12]),  # 2: past the table (11 entries)
        events_member(objects=[0xFFFF]),  # 3: a clone object
        events_member(bg=[11]),  # 4: reaches ScrCmd_Unused_013
        events_member(),  # 5: the only user of its file
    ]
    rows = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (1, 5, 0)]
    where = join_trainer_maps(rows, [2] * 6, scripts, events, table, 100)
    assert where[20] == {0, 2, 3, 4}
    assert where[10] == where[30] == {2, 3, 4}
    assert where[80] == where[90] == {5}  # single user: the whole-file walk
    assert 1 not in set().union(*where.values())


def test_join_runs_c_started_and_init_script_entries(table):
    shared = script_file(table, battle(table, 50), battle(table, 60), battle(table, 70))
    init = bytes([2]) + struct.pack("<H", 1) + b"\0\0" + b"\0"  # ON_TRANSITION -> entry 1
    scripts = [shared, b"\0", init]
    events = [events_member(), events_member()]
    rows = [(0, 0, 0), (0, 1, 0)]
    where = join_trainer_maps(rows, [1, 2], scripts, events, table, 100)
    # entries 2 and 3 are C-started ids every header on the file runs; entry 1 only by init
    assert where[60] == where[70] == {0, 1}
    assert where[50] == {1}


def test_header_battles_rules():
    per = [{1}, {2}, {3}, set(), set(), set(), set(), set(), {9}]
    plain = [False] * 9
    whole = {1, 2, 3, 9}
    assert header_battles([9], per, plain, whole, False) == {2, 3, 9}  # + C ids 2, 3
    assert header_battles([10], per, plain, whole, False) == whole  # past the table
    assert header_battles([], per, plain, whole, True) == whole  # clone object
    by_id = [False] * 8 + [True]
    assert header_battles([9], per, by_id, whole, False) == whole  # ScrCmd_Unused_013
    assert header_battles([1], [{1}], [False], {1, 5}, False) == {1}  # no C id in a 1-entry file
