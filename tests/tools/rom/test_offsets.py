"""tools.rom.offsets: the analysis on hand-assembled Thumb code, and trace_battle.lua's
battle-RAM offsets checked against both ROMs (skips without them or the linker map)."""

import struct

import pytest

from tools.rom.offsets import ACCESSES, analyse, check_offsets, lua_tables

BASE = 0x02000000


def test_analyse_follows_a_battlemon_access_through_a_literal_pool():
    pytest.importorskip("capstone")
    # movs r2,#0xC0; muls r2,r1; adds r2,r0,r2; ldr r3,[pc,#4]; ldr r0,[r2,r3]; bx lr;
    # then the literal 0x2D8C (battleMons[r1].curHP).
    code = struct.pack("<6HI", 0x22C0, 0x434A, 0x1882, 0x4B01, 0x58D0, 0x4770, 0x2D8C)
    a = analyse(code, BASE, BASE, len(code))
    assert [(m.base, m.off, m.width, m.store) for m in a.accesses] == [("arg0", 0x2D8C, 4, False)]
    assert a.multipliers == (0xC0,)
    assert a.mnemonics == ("movs", "muls", "adds", "ldr", "ldr", "bx")


def test_analyse_keeps_the_pointer_across_a_conditional_return():
    pytest.importorskip("capstone")
    # cmp r1,#0; beq +2 (to the ldr); bx lr; ldr r0,[r0,#0x30]; bx lr
    code = struct.pack("<5H", 0x2900, 0xD000, 0x4770, 0x6B00, 0x4770)
    a = analyse(code, BASE, BASE, len(code))
    assert [(m.base, m.off) for m in a.accesses] == [("arg0", 0x30)]


def test_every_lua_field_has_an_access():
    assert set(lua_tables()) == set(ACCESSES)


@pytest.mark.parametrize("game", ["platinum_us", "platinum_kaizo_nightmare_final"])
def test_offsets_vanilla_and_kaizo(kaizo_inputs, symbols, game):
    pytest.importorskip("capstone")
    checks = check_offsets(game)
    assert len(checks) == len(ACCESSES)
    bad = [c for c in checks if not c.ok]
    assert bad == [], bad  # a failure names the field to fix in trace_battle.lua


# Each mutation gives a field its neighbour's offset; the pinned instructions must refuse it.
SWAPS = [
    {("MON", "curHP"): 0x50, ("MON", "maxHP"): 0x4C},
    {("MON", "maxHP"): 0x4C},
    {("MON", "curHP"): 0x50},
    {("CTX", "commandNext"): 0x08},
    {("CTX", "criticalMul"): 0x216C},
    {("CTX", "hpCalcTemp"): 0x2D8C},
    {("SYS", "parties"): 0x2C},
]


@pytest.mark.parametrize("game", ["platinum_us", "platinum_kaizo_nightmare_final"])
@pytest.mark.parametrize("swap", SWAPS, ids=lambda s: "+".join(f"{t}.{n}" for t, n in s))
def test_a_neighbours_offset_fails(kaizo_inputs, symbols, game, swap):
    pytest.importorskip("capstone")
    tables = {**lua_tables(), **swap}
    bad = {tuple(c.field.split(".")) for c in check_offsets(game, tables) if not c.ok}
    assert bad == set(swap)


@pytest.mark.parametrize("game", ["platinum_us", "platinum_kaizo_nightmare_final"])
def test_no_field_accepts_another_fields_offset(kaizo_inputs, symbols, game):
    pytest.importorskip("capstone")
    tables = lua_tables()
    accepted = []
    for key, offset in tables.items():
        for other, wrong in tables.items():
            if other[0] == key[0] and wrong != offset:
                checks = check_offsets(game, {**tables, key: wrong})
                if next(c for c in checks if c.field == ".".join(key)).ok:
                    accepted.append((key, other))
    assert accepted == []
