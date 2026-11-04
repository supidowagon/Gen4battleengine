"""Trainer start conditions from map headers, zone events and field scripts."""

import collections
import json
import struct

import pytest

from tools.extract_decomp import extract_trainers
from tools.rom.readers.maps import (
    ScriptDecodeError,
    command_table,
    script_battles,
    trainer_start_conditions,
)


@pytest.fixture(scope="module")
def table(decomp):
    return command_table(decomp)


def _op(table, name):
    return next(op for op, (n, _) in table.items() if n == name)


def _cmd(table, name, *operands):
    op = _op(table, name)
    widths = table[op][1]
    assert len(widths) == len(operands), name
    fmt = {1: "B", 2: "H", 4: "i"}
    return struct.pack("<H", op) + b"".join(
        struct.pack("<" + fmt[w], v) for w, v in zip(widths, operands, strict=True)
    )


def _script(*entries: int) -> bytes:
    """A header with one entry per absolute offset in ``entries`` (relative to the end of
    the header, which is 4 bytes per entry plus the 2-byte terminator)."""
    size = 4 * len(entries) + 2
    head = b"".join(struct.pack("<i", size + e - (4 * i + 4)) for i, e in enumerate(entries))
    return head + struct.pack("<H", 0xFD13)


def test_command_table(decomp):
    t = command_table(decomp)
    by_name = {v[0]: (k, v[1]) for k, v in t.items()}
    assert by_name["StartTrainerBattle"][1] == (2, 2) and by_name["GoToIf"][1] == (1, 4)
    # Opcodes as the Platinum script engine numbers them.
    assert by_name["End"][0] == 0x2 and by_name["GoToIf"][0] == 0x1C
    assert by_name["StartTrainerBattle"][0] == 0xE5
    assert by_name["StartTagBattle"][1] == (2, 2, 2) and by_name["StartFirstBattle"][1] == (2,)
    assert by_name["ApplyMovement"][1] == (2, 4) and by_name["SetVarFromValue"][1] == (2, 2)
    assert len(t) == 840  # every ScriptCommand line of scrcmd.h has a macro
    fixed = [k for k in t if k not in t.variable]
    assert len(fixed) == 834  # 833 `.short SCRCMD_*` macros + GetRandomBattlegroundTrainers
    assert sorted(t[k][0] for k in t.variable) == sorted(
        [
            "CallTVBroadcast",
            "DoDefogFunc",
            "DoFlashFunc",
            "DoGroupConnectionAction",
            "DoStrengthFunc",
            "MysteryGiftGive",
        ]
    )


def test_script_walk_synthetic(table):
    # entry -> CompareVarToValue; GoToIf -> battle; End; <dead bytes>; battle: Start 246; End
    head = _script(0)
    body = _cmd(table, "CompareVarToValue", 0x4000, 1)
    goto = _cmd(table, "GoToIf", 1, 0)
    end = _cmd(table, "End")
    dead = b"\xff\xff" + _cmd(table, "StartTrainerBattle", 7, 0)
    battle = _cmd(table, "StartTrainerBattle", 246, 0) + _cmd(table, "End")
    goto = _cmd(table, "GoToIf", 1, len(end) + len(dead))
    script = head + body + goto + end + dead + battle
    assert script_battles(script, table) == {246}


def test_script_walk_follows_calls_not_movement(table):
    head = _script(0)
    # ApplyMovement's offset points at bytes that would decode as a battle; it is data.
    movement_target = _cmd(table, "StartTrainerBattle", 9, 0)
    apply = _cmd(table, "ApplyMovement", 0, 0)
    call = _cmd(table, "Call", 0)
    end = _cmd(table, "End")
    sub = _cmd(table, "StartTagBattle", 1, 2, 3) + _cmd(table, "Return")
    apply = _cmd(table, "ApplyMovement", 0, len(call) + len(end) + len(sub))
    call = _cmd(table, "Call", len(end))
    script = head + apply + call + end + sub + b"\xff\xff" + movement_target
    assert script_battles(script, table) == {1, 2, 3}


def test_setvar_resolution(table):
    head = _script(0)
    body = (
        _cmd(table, "SetVarFromValue", 0x8000, 5)
        + _cmd(table, "StartTrainerBattle", 0x8000, 0)
        + _cmd(table, "End")
    )
    assert script_battles(head + body, table) == {5}


def test_unknown_opcode_raises(table):
    script = _script(0, 2) + _cmd(table, "End") + struct.pack("<H", len(table))
    with pytest.raises(ScriptDecodeError, match=r"offset 0xc") as e:
        script_battles(script, table)
    assert e.value.offset == 0xC and e.value.opcode == len(table)


def test_truncated_command_raises(table):
    script = _script(0) + _cmd(table, "StartTrainerBattle", 1, 0)[:4]
    with pytest.raises(ScriptDecodeError, match=r"offset 0x6"):
        script_battles(script, table)


def test_vanilla_start_conditions_match_decomp(vanilla_files, decomp, symbols, tmp_path):
    extract_trainers(decomp, tmp_path)
    want = [
        t["start_condition"] for t in json.loads((tmp_path / "trainers.json").read_text("utf-8"))
    ]
    got = trainer_start_conditions(vanilla_files, decomp, symbols)
    assert got == want
    assert collections.Counter(got) == {
        "none": 810,
        "rain": 41,
        "fog": 36,
        "hail": 29,
        "sandstorm": 12,
    }


def test_setvar_resolution_follows_paths(table):
    # entry: GoToIf -> battle path; fall-through path sets the var to an item id and
    # ends without reaching the battle. Only the trainer id reaches StartTagBattle.
    head = _script(0)
    item_path = _cmd(table, "SetVarFromValue", 0x8004, 390) + _cmd(table, "End")
    battle_path = (
        _cmd(table, "SetVarFromValue", 0x8004, 12)
        + _cmd(table, "StartTagBattle", 0x8004, 20, 21)
        + _cmd(table, "End")
    )
    goto = _cmd(table, "GoToIf", 1, len(item_path))
    assert script_battles(head + goto + item_path + battle_path, table) == {12, 20, 21}


def test_setvar_resolution_through_call(table):
    # The value is set in a subroutine; the battle after the Call sees it, and a later
    # overwrite after the battle does not leak back.
    head = _script(0)
    call = _cmd(table, "Call", 0)
    battle = _cmd(table, "StartTrainerBattle", 0x8000, 0)
    later = _cmd(table, "SetVarFromValue", 0x8000, 99) + _cmd(table, "End")
    sub = _cmd(table, "SetVarFromValue", 0x8000, 7) + _cmd(table, "Return")
    call = _cmd(table, "Call", len(battle) + len(later))
    assert script_battles(head + call + battle + later + sub, table) == {7}


def test_partner_trainer_var_places_trainer(decomp, table):
    from tools.rom.readers.maps import var_ids

    partner = var_ids(decomp)["VAR_PARTNER_TRAINER_ID"]
    assert partner == table.partner_var
    head = _script(0)
    body = (
        _cmd(table, "SetVarFromValue", partner, 610)
        + _cmd(table, "SetVarFromValue", 0x4001, 611)  # another variable: not a placement
        + _cmd(table, "End")
    )
    assert script_battles(head + body, table) == {610}
