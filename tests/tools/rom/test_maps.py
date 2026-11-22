"""Trainer start conditions from map headers, zone events and field scripts."""

import collections
import json
import struct

import pytest

from tools.battle_scripts import Op
from tools.extract_decomp import extract_trainers
from tools.rom.readers.maps import (
    ScriptDecodeError,
    command_table,
    script_battles,
    script_ops,
    trainer_start_conditions,
)
from tools.rom.readers.trainers import read_trainers


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


def test_start_conditions_overrides_none_means_none(vanilla_files, decomp, symbols):
    # overrides=None reads no file (fix round 1): vanilla's date-dependent trainers raise.
    from tools.extract_decomp import ExtractError

    with pytest.raises(ExtractError, match="can start with any of.*list it in"):
        trainer_start_conditions(vanilla_files, decomp, symbols, overrides=None)


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


def test_unresolved_operand_reported(table):
    # Path 1 sets the var, path 2 reaches the same battle with it unset (another command
    # wrote it); the battle is reported with the one constant the other path gives.
    head = _script(0)
    battle = _cmd(table, "StartTrainerBattle", 0x8005, 0) + _cmd(table, "End")
    set_path = _cmd(table, "SetVarFromValue", 0x8005, 9)
    goto = _cmd(table, "GoToIf", 1, len(set_path))
    script = head + goto + set_path + battle
    sites = []
    assert script_battles(script, table, unresolved=sites) == {9}
    at = len(head) + len(goto) + len(set_path)
    assert sites == [(at, 0x8005, 1)]
    sites = []
    assert script_battles(_script(0) + battle, table, unresolved=sites) == set()
    assert sites == [(6, 0x8005, 0)]
    sites = []
    script_battles(
        _script(0) + _cmd(table, "SetVarFromValue", 0x8005, 9) + battle, table, unresolved=sites
    )
    assert sites == []


def test_vanilla_unresolved_operands_pinned(vanilla_files, decomp, symbols):
    # scr_seq members are scripts.order lines: 454 scripts_pokemon_mansion.s:183
    # (StartTrainerBattle VAR_0x8007) and 1101 scripts_battleground.s:242
    # (VAR_MAP_LOCAL_0x05). scripts_pokemon_center_daily_trainers.s:272 (member 400) is no
    # map header's script file, so the walk never reaches it.
    from tools.rom.readers.maps import UnresolvedOperand

    found = []
    trainer_start_conditions(vanilla_files, decomp, symbols, unresolved=found)
    assert found == [
        UnresolvedOperand(454, 0x321, 0x8007, 2),
        UnresolvedOperand(1101, 0x341, 0x4005, 13),
    ]


def test_kaizo_unresolved_operands(kaizo_files, table):
    # The walk over the Kaizo scripts alone (test_kaizo_rom.py walks them from the map
    # headers): member 136 (Veilstone City) adds a fully unresolved site to vanilla's two.
    from tools.rom.readers.maps import SCRIPTS_NARC

    scripts = kaizo_files.narc(SCRIPTS_NARC)
    got = {}
    for m in (136, 454, 1101):
        sites = []
        script_battles(scripts[m], table, unresolved=sites)
        got[m] = sites
    assert got == {136: [(909, 0x8004, 0)], 454: [(1403, 0x8007, 2)], 1101: [(833, 0x4005, 13)]}


def test_rematch_base_out_of_range_raises(vanilla_files, decomp, symbols, monkeypatch):
    from tools.rom.readers import ReaderError, maps

    real = maps.symbol_bytes

    def fake(rom, syms, name):
        data = real(rom, syms, name)
        return (5000).to_bytes(2, "little") + data[2:] if name == maps.REMATCH_SYMBOL else data

    monkeypatch.setattr(maps, "symbol_bytes", fake)
    with pytest.raises(ReaderError, match="offset 0: unknown base trainer 5000"):
        maps.trainer_maps(vanilla_files, decomp, symbols, 928)


# scrcmd.inc wrapper macros that pick a command by operand (scrcmd.inc:178-184,309-315); the
# helper takes constants, so the value form is the one assembled.
_WRAPPERS = {"CompareVar": "CompareVarToValue", "SetVar": "SetVarFromValue"}


def assemble(table, *cmds) -> bytes:
    """A one-entry header plus ``cmds`` (macro name, operands...); a jump operand is the
    offset from the end of its command, as ``scrcmd.inc`` encodes it."""
    return _script(0) + b"".join(_cmd(table, _WRAPPERS.get(n, n), *ops) for n, *ops in cmds)


def test_script_ops_fuse_compare_and_branch(decomp):  # hand-assembled with K1's opcodes
    table = command_table(decomp)
    script = assemble(table, ("CompareVar", 0x8004, 3), ("GoToIf", 1, 2), ("End",), ("Return",))
    ops = script_ops(script, table)  # offset 2 skips the 2-byte End -> the Return
    assert [o.name for o in ops] == ["label", "GoToIfEq", "End", "label", "Return"]
    assert ops[1].args == (0x8004, 3, ops[3].args[0])
    assert (ops[0].args, ops[3].args) == (("@6",), ("@21",))


def test_script_ops_other_conditions_are_not_fused(table):
    # GoToIf 5 is GoToIfNe (scrcmd.inc); a CompareVar without its GoToIf is not fused.
    script = assemble(
        table, ("CompareVar", 0x8004, 3), ("GoToIf", 5, 0), ("CompareVar", 0x8004, 3), ("End",)
    )
    assert [o.name for o in script_ops(script, table)] == [
        "label",
        "other",
        "other",
        "label",
        "other",
        "End",
    ]


def test_script_ops_kept_commands(table):
    script = assemble(
        table,
        ("SetVar", 0x403F, 609),
        ("SetHasPartner",),
        ("GetPlayerGender", 0x800C),
        ("GetPlayerStarterSpecies", 0x800C),
        ("Call", 16),
        ("StartTagBattle", 0x8004, 3, 4),
        ("StartTrainerBattle", 5, 6),
        ("End",),
        ("SetVar", 0x8004, 7),
        ("GoTo", 0),
        ("Return",),
    )
    ops = script_ops(script, table)
    assert ops == [
        Op("label", ("@6",)),
        Op("SetVar", (0x403F, 609)),
        Op("SetHasPartner", ()),
        Op("GetPlayerGender", (0x800C,)),
        Op("GetPlayerStarterSpecies", (0x800C,)),
        Op("Call", ("@44",)),
        Op("StartTagBattle", (0x8004, 3, 4)),
        Op("StartTrainerBattle", (5, 6)),
        Op("End", ()),
        Op("label", ("@44",)),
        Op("SetVar", (0x8004, 7)),
        Op("GoTo", ("@56",)),
        Op("label", ("@56",)),
        Op("Return", ()),
    ]


def test_script_ops_unknown_opcode_raises(table):
    script = _script(0) + struct.pack("<H", len(table))
    with pytest.raises(ScriptDecodeError, match="offset 0x6"):
        script_ops(script, table)


def test_rom_battle_structures_match(vanilla_files, decomp, symbols, charmap, decomp_inter):
    want = json.loads((decomp_inter / "trainers.json").read_text("utf-8"))
    # Only the battle structures are compared below: no fog headers needed, one empty list each.
    got = read_trainers(
        vanilla_files,
        decomp,
        symbols,
        charmap,
        [t["start_condition"] for t in want],
        [[] for _ in want],
    )

    def key(t):
        return (t["format"], t["opponent_2"], t["partner"])

    assert [key(t) for t in got] == [key(t) for t in want]
    assert sum(t["format"] == "double_two_trainer" for t in got) == 20
