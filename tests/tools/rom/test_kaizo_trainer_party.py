"""The patched Kaizo ``TrainerData_BuildParty`` decode in ``docs/kaizo/rom_diff.md``
("TrainerData_BuildParty, decoded", K2b Task 13 fix round 1), pinned as instruction
patterns on the patched ROM. Each pattern is a list of Thumb halfwords (named by their
instruction) and ``("bl", symbol or address)`` calls, like ``tools.rom.readers.maps``'s
``_KAIZO_BOUNDS``. Skips without the ROM inputs or the linker map."""

import struct

import pytest

pytest.importorskip("ndspy")

BUILD_PARTY = 0x020793B8
FRIENDSHIP_HELPER = 0x02079570
OVERRIDE_HELPER = 0x020795A0

MON_DATA_MOVE1, MON_DATA_FRIENDSHIP = 54, 9  # generated/pokemon_data_params.txt
SPECIES_DATA_GENDER_RATIO = 18  # generated/species_data_params.txt
MOVE_FRUSTRATION = 218

# 0x02079570 (mon): friendship = 0 if a move is Frustration, else 255.
FRIENDSHIP = (
    0xB5F8,  # push {r3-r7, lr}
    0x1C05,  # r5 = mon
    0x2400,  # r4 = 0 (move slot)
    0x27FF,  # r7 = 255 (friendship)
    0x2300 | MON_DATA_MOVE1,  # r3 = MON_DATA_MOVE1
    0x1C28,  # r0 = mon
    0x1C21,  # r1 = slot
    0x18C9,  # r1 += MON_DATA_MOVE1
    0x2200,  # r2 = 0
    ("bl", "Pokemon_GetValue"),
    0x2800 | MOVE_FRUSTRATION,  # cmp r0, #218
    0xD100,  # bne +0
    0x2700,  # r7 = 0
    0x1C64,  # slot += 1
    0x2C04,  # cmp slot, #4
    0xDBF3,  # blt loop
    0x1C28,  # r0 = mon
    0x2100 | MON_DATA_FRIENDSHIP,  # r1 = MON_DATA_FRIENDSHIP
    0x466A,  # r2 = sp
    0x7017,  # strb r7, [r2]
    ("bl", "Pokemon_SetValue"),
    0xBDF8,  # pop {r3-r7, pc}
)

# 0x020795A0 (species, form, override, &genderMod): the gender nibble sets the modifier to
# the form's gender ratio +2 / -2; the ability nibble then clears / sets bit 0 of r0, which
# is still the species argument when the gender nibble is 0 (the ROM bug).
OVERRIDE = (
    0xB5F0,  # push {r4-r7, lr}
    0x1C1D,  # r5 = &genderMod
    0x230F,  # r3 = 0xF
    0x4013,  # r3 &= override
    0x1C1E,  # r6 = gender nibble
    0x0914,  # r4 = override >> 4 (ability nibble)
    0x2A00,  # cmp override, #0
    0xD015,  # beq return
    0x2E00,  # cmp gender, #0
    0xD008,  # beq ability (r0 still = species)
    0x2200 | SPECIES_DATA_GENDER_RATIO,  # r2 = SPECIES_DATA_GENDER_RATIO
    ("bl", "SpeciesData_GetFormValue"),
    0x2E01,  # cmp gender, #1
    0xD101,  # bne female
    0x1C80,  # r0 += 2
    0xE000,  # b store
    0x1E80,  # r0 -= 2
    0x6028,  # *genderMod = r0
    0x2C01,  # cmp ability, #1
    0xD103,  # bne slot 2
    0x2101,  # r1 = 1
    0x4388,  # r0 &= ~1
    0x6028,  # *genderMod = r0
    0xBDF0,  # pop
    0x2C02,  # cmp ability, #2
    0xD102,  # bne return
    0x2101,  # r1 = 1
    0x4308,  # r0 |= 1
    0x6028,  # *genderMod = r0
    0xBDF0,  # pop
)


@pytest.fixture(scope="module")
def code(kaizo_files, symbols):
    """(start, bytes) of the patched routine's extent, at the vanilla linker address."""
    sym = symbols.find("TrainerData_BuildParty")
    start = sym.address & ~1
    assert start == BUILD_PARTY
    for ram, data in kaizo_files.arm9_sections():
        if ram <= start and start + sym.size <= ram + len(data):
            return start, data[start - ram : start - ram + sym.size]
    pytest.fail("TrainerData_BuildParty is outside the ARM9 sections")


def _bl(code, at):
    start, data = code
    off = at - start
    hi, lo = struct.unpack_from("<2H", data, off)
    assert hi >> 11 == 0x1E and lo >> 11 == 0x1F, f"no BL at {at:#x}"
    high = (hi & 0x7FF) << 12
    if high & 0x400000:
        high -= 0x800000
    return at + 4 + high + ((lo & 0x7FF) << 1)


def _match(code, symbols, at, pattern):
    start, data = code
    for item in pattern:
        if isinstance(item, int):
            got = struct.unpack_from("<H", data, at - start)[0]
            assert got == item, f"{at:#x}: {got:#06x}, expected {item:#06x}"
            at += 2
        else:
            want = symbols.find(item[1]).address & ~1
            assert _bl(code, at) == want, f"{at:#x}: expected bl {item[1]}"
            at += 4


def test_friendship_helper(code, symbols):
    _match(code, symbols, FRIENDSHIP_HELPER, FRIENDSHIP)


def test_override_helper(code, symbols):
    _match(code, symbols, OVERRIDE_HELPER, OVERRIDE)


def test_build_party_calls(code, symbols):
    assert _bl(code, 0x02079468) == OVERRIDE_HELPER  # before the seed
    assert _bl(code, 0x02079512) == FRIENDSHIP_HELPER  # after the moves
    assert _bl(code, 0x02079534) == symbols.find("Pokemon_CalcStats").address & ~1
    assert _bl(code, 0x0207947E) == symbols.find("LCRNG_SetSeed").address & ~1


def test_gender_modifier_slot_is_set_once_per_trainer(code):
    start, data = code

    def half(at):
        return struct.unpack_from("<H", data, at - start)[0]

    assert half(0x0207940E) == 0x9012  # str r0, [sp, #72]: the modifier slot, before the loop
    assert half(0x02079466) == 0xAB12  # add r3, sp, #72: its address, the helper's 4th arg
    assert half(0x020794A8) == 0x9812  # ldr r0, [sp, #72]: read for the PID
    # The loop's back edge (b at 0x02079554) returns to 0x0207943C, after the store.
    b = half(0x02079554)
    assert b >> 11 == 0x1C
    off = b & 0x7FF
    assert 0x02079554 + 4 + ((off - 0x800 if off & 0x400 else off) << 1) == 0x0207943C
