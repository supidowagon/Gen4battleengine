"""Unit tests for tools.rom.diff on synthetic ROMs (``diff_helpers.FakeRom``)."""

import struct

from tests.tools.rom.diff_helpers import (
    BASE,
    CODE,
    RELOC_PAT,
    RELOC_SYMS,
    RELOC_VAN,
    SYMS,
    fake,
    poke,
    reloc_pat,
    reloc_van,
    thumb_bl,
)
from tools.rom.diff import (
    HEADER_FIELDS,
    CodeChange,
    HeaderChange,
    RomDiff,
    diff_roms,
    member_label,
)


def test_identical_roms_have_no_diff():
    r = fake()
    assert diff_roms(r, r, None) == RomDiff((), (), (), {}, {})


def test_added_removed_changed_files_and_narc_members():
    v = fake(files={"a.bin": b"1", "gone.bin": b"2", "n.narc": [b"m"] * 5})
    p = fake(files={"a.bin": b"9", "new/x.bin": b"3",
                    "n.narc": [b"m", b"m", b"M", b"m", b"m"]})
    by = {f.path: f for f in diff_roms(v, p, None).files}
    assert [by[k].status for k in ("a.bin", "gone.bin", "new/x.bin")] == [
        "changed", "removed", "added"]
    assert by["n.narc"].members_changed == (2,) and by["n.narc"].member_counts == (5, 5)
    assert (by["gone.bin"].vanilla_size, by["gone.bin"].patched_size) == (1, None)
    assert (by["new/x.bin"].vanilla_size, by["new/x.bin"].patched_size) == (None, 1)


def test_narc_member_count_change_lists_extra_members():
    v = fake(files={"n.narc": [b"a", b"b"]})
    p = fake(files={"n.narc": [b"a", b"B", b"c"]})
    (f,) = diff_roms(v, p, None).files
    assert f.members_changed == (1, 2) and f.member_counts == (2, 3)


def test_narcs_are_found_by_magic_not_extension():
    # wazaeffect/we.arc is a NARC; poketool/personal/pms.narc is not one
    v = fake(files={"we.arc": [b"a", b"b"], "pms.narc": b"plain"})
    p = fake(files={"we.arc": [b"a", b"B"], "pms.narc": b"PLAIN"})
    by = {f.path: f for f in diff_roms(v, p, None).files}
    assert by["we.arc"].members_changed == (1,) and by["we.arc"].member_counts == (2, 2)
    assert by["pms.narc"].members_changed == () and by["pms.narc"].member_counts is None


def test_in_place_code_change_named_by_symbol():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x44, b"\xff\xff")), SYMS)
    assert d.code == (CodeChange("overlay3", BASE + 0x44, BASE + 0x46, BASE + 0x44,
                                 BASE + 0x46, "Foo"),)
    assert d.modules == {"overlay3": (0x100, 0x100)}


def test_change_spanning_two_symbols_is_split():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x5E, b"\xff" * 4)), SYMS)
    assert [(c.start - BASE, c.end - BASE, c.symbol) for c in d.code] == [
        (0x5E, 0x60, "Foo"), (0x60, 0x62, "Bar")]


def test_without_symbols_nothing_is_split():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x5E, b"\xff" * 4)), None)
    assert [(c.start - BASE, c.end - BASE, c.symbol) for c in d.code] == [(0x5E, 0x62, None)]


def test_change_outside_every_symbol_has_no_name():
    d = diff_roms(fake(), fake(ov3=poke(CODE, 0x10, b"\xff")), SYMS)
    assert [(c.start - BASE, c.symbol) for c in d.code] == [(0x10, None)]


def test_nearby_changes_merge():  # 3 equal bytes between -> one range; 5 -> two
    near = poke(poke(CODE, 0x44, b"\xff"), 0x48, b"\xff")
    far = poke(poke(CODE, 0x44, b"\xff"), 0x4A, b"\xff")
    edge = poke(poke(CODE, 0x44, b"\xff"), 0x49, b"\xff")  # exactly 4 equal bytes
    assert len(diff_roms(fake(), fake(ov3=near), SYMS).code) == 1
    assert len(diff_roms(fake(), fake(ov3=far), SYMS).code) == 2
    assert len(diff_roms(fake(), fake(ov3=edge), SYMS).code) == 2


def test_insertion_aligned():
    d = diff_roms(fake(), fake(ov3=CODE[:0x80] + b"\xee" * 8 + CODE[0x80:]), None)
    assert [(c.start - BASE, c.end - BASE, c.patched_start - BASE, c.patched_end - BASE)
            for c in d.code] == [(0x80, 0x80, 0x80, 0x88)]
    assert d.modules["overlay3"] == (0x100, 0x108)


def test_deletion_aligned():
    d = diff_roms(fake(), fake(ov3=CODE[:0x80] + CODE[0x88:]), None)
    assert [(c.start - BASE, c.end - BASE, c.patched_start - BASE, c.patched_end - BASE)
            for c in d.code] == [(0x80, 0x88, 0x80, 0x80)]


def test_change_after_insertion_has_patched_addresses():
    pat = CODE[:0x40] + b"\xee" * 4 + poke(CODE, 0xA0, b"\xff")[0x40:]
    d = diff_roms(fake(), fake(ov3=pat), None)
    assert [(c.start - BASE, c.end - BASE, c.patched_start - BASE, c.patched_end - BASE)
            for c in d.code] == [(0x40, 0x40, 0x40, 0x44), (0xA0, 0xA1, 0xA4, 0xA5)]


def test_relocated_thumb_bl_counted_not_listed():
    # an 8-byte insertion inside "Grown"; a Thumb BL in "Caller", before the insertion,
    # targets "Callee" past it and is re-encoded 8 bytes further
    d = diff_roms(RELOC_VAN, RELOC_PAT, RELOC_SYMS)
    assert [c.symbol for c in d.code] == ["Grown"] and d.relocations[("overlay3", "Caller")] == 1
    (grown,) = d.code
    assert (grown.start - BASE, grown.end - BASE, grown.patched_end - BASE) == (0x60, 0x60, 0x68)


def test_relocated_literal_pointer_counted():
    # same layout; "Holder" has a 32-bit word = Callee's absolute (Thumb) address
    d = diff_roms(RELOC_VAN, RELOC_PAT, RELOC_SYMS)
    assert [c.symbol for c in d.code] == ["Grown"] and d.relocations[("overlay3", "Holder")] == 1
    assert sum(d.relocations.values()) == 2


def test_relocated_arm_bl_counted():
    def arm_bl(pc, target):
        return struct.pack("<I", 0xEB000000 | (((target - (pc + 8)) >> 2) & 0xFFFFFF))

    van = poke(reloc_van(), 0x20, arm_bl(BASE + 0x20, BASE + 0xC0))
    pat = poke(reloc_pat(), 0x20, arm_bl(BASE + 0x20, BASE + 0xC8))
    d = diff_roms(fake(ov3=van), fake(ov3=pat), RELOC_SYMS)
    assert [c.symbol for c in d.code] == ["Grown"]
    assert d.relocations[("overlay3", None)] == 1  # 0x20 is outside every symbol


def test_retargeted_bl_is_a_change_not_a_relocation():
    # the BL now calls Callee + 4: a real change, not explained by the insertion
    pat = poke(reloc_pat(), 0x10, thumb_bl(BASE + 0x10, BASE + 0xCC))
    d = diff_roms(RELOC_VAN, fake(ov3=pat), RELOC_SYMS)
    assert [c.symbol for c in d.code] == ["Caller", "Grown"]
    assert ("overlay3", "Caller") not in d.relocations


def test_moved_call_out_of_the_module_is_a_relocation():
    # a Thumb BL in Holder, after the insertion, calls ARM9 code: the caller moved by 8,
    # the target did not, so the pair is re-encoded
    arm9_fn = 0x02001000
    van = poke(reloc_van(), 0x84, thumb_bl(BASE + 0x84, arm9_fn))
    pat = poke(reloc_pat(), 0x8C, thumb_bl(BASE + 0x8C, arm9_fn))
    d = diff_roms(fake(ov3=van), fake(ov3=pat), RELOC_SYMS)
    assert [c.symbol for c in d.code] == ["Grown"]
    assert d.relocations[("overlay3", "Holder")] == 2  # the BL and the literal word


def test_itcm_section_addresses():
    itcm = bytes(64)
    v = fake(arm9=[(0x02000000, CODE), (0x01FF8000, itcm)])
    p = fake(arm9=[(0x02000000, CODE), (0x01FF8000, poke(itcm, 0x10, b"\x01"))])
    assert diff_roms(v, p, None).code == (
        CodeChange("arm9", 0x01FF8010, 0x01FF8011, 0x01FF8010, 0x01FF8011, None),)


def test_header_only_named_fields():
    h = bytearray(fake().header())
    h[0xC0] ^= 0xFF  # logo byte: not a named field
    assert diff_roms(fake(), fake(header=bytes(h)), None).header == ()
    h[0x0C:0x10] = b"JAK7"
    assert diff_roms(fake(), fake(header=bytes(h)), None).header == (
        HeaderChange("game_code", "CPUE", "JAK7"),)


def test_header_int_fields_are_hex():
    h = bytearray(fake().header())
    h[0x80:0x84] = (0x08000000).to_bytes(4, "little")
    assert diff_roms(fake(), fake(header=bytes(h)), None).header == (
        HeaderChange("total_rom_size", "0x0", "0x8000000"),)


def test_header_fields_table():
    assert len(HEADER_FIELDS) == 29
    assert HEADER_FIELDS["game_code"] == (0x00C, 4, "ascii")
    assert HEADER_FIELDS["header_crc"] == (0x15E, 2, "int")
    assert all(kind in ("ascii", "int") for _, _, kind in HEADER_FIELDS.values())


def test_member_label(decomp):
    assert member_label("msgdata/pl_msg.narc", 0, decomp).startswith("TEXT_BANK_")
    assert member_label("poketool/personal/pl_personal.narc", 1, decomp) == "SPECIES_BULBASAUR"
    assert member_label("poketool/personal/evo.narc", 1, decomp) == "SPECIES_BULBASAUR"
    assert member_label("poketool/personal/wotbl.narc", 1, decomp) == "SPECIES_BULBASAUR"
    assert member_label("poketool/waza/pl_waza_tbl.narc", 1, decomp) == "MOVE_POUND"
    assert member_label("poketool/trainer/trdata.narc", 0, decomp).startswith("TRAINER_")
    assert member_label("poketool/trainer/trpoke.narc", 0, decomp).startswith("TRAINER_")
    assert member_label("poketool/personal/pl_personal.narc", 100000, decomp) is None
    assert member_label("fielddata/x.narc", 0, decomp) is None
