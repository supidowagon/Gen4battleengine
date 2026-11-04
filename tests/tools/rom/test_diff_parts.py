"""tools.rom.diff: the parts outside the file system and calls into injected code."""

import struct

import pytest

from tests.tools.rom.diff_helpers import BASE, CODE, SYMS, fake, poke, thumb_bl
from tools.rom.diff import FarCall, PartChange, diff_roms, write_report

PROV = {"vanilla_sha1": "v", "patch_sha256": "p", "patched_sha1": "k", "decomp_commit": "c"}


def _entry(i, ram, size, bss=0, sinit=(0, 0), file_id=None, compressed=0):
    return struct.pack("<8I", i, ram, size, bss, *sinit, i if file_id is None else file_id,
                       compressed)


def test_parts_report_only_differing_fields():
    van_table = _entry(0, 0x02200000, 0x100) + _entry(1, 0x02300000, 0x200, 0x20)
    pat_table = _entry(0, 0x02200000, 0x100) + _entry(1, 0x02300000, 0x220, 0x20,
                                                      compressed=0x01000220)
    v = fake(parts={"arm9_overlay_table": van_table, "arm9_footer": struct.pack("<3I", 0xDEC00621, 0xBA0, 0),
                    "arm7": b"a" * 8, "banner": b"b" * 4})
    p = fake(parts={"arm9_overlay_table": pat_table, "arm9_footer": struct.pack("<3I", 0xDEC00621, 0xAD8, 0),
                    "arm7": b"a" * 9, "banner": b"b" * 4,
                    "arm7_overlay_table": _entry(0, 0x037F8000, 0x10)})
    d = diff_roms(v, p, None)
    fields = {(c.part, c.field): (c.vanilla, c.patched) for c in d.parts}
    assert fields[("arm9_overlay_table", "entry 1 ram_size")] == ("0x200", "0x220")
    assert fields[("arm9_overlay_table", "entry 1 compressed_size")] == ("0x0", "0x220")
    assert fields[("arm9_overlay_table", "entry 1 compressed_flags")] == ("0x0", "0x1")
    assert fields[("arm9_footer", "module_params_offset")] == ("0xBA0", "0xAD8")
    assert fields[("arm7", "size")] == ("0x8", "0x9") and ("arm7", "sha1") in fields
    assert fields[("arm7_overlay_table", "entries")] == ("0", "1")
    assert not any(c.part == "banner" or "entry 0" in c.field for c in d.parts)
    assert len(d.parts) == 7


def test_overlay_table_of_partial_entries_raises():
    with pytest.raises(ValueError, match="arm9_overlay_table.*not whole entries"):
        diff_roms(fake(), fake(parts={"arm9_overlay_table": b"\0" * 33}), None)


def test_parts_section_in_report(tmp_path):
    d = diff_roms(fake(), fake(parts={"banner": b"x"}), None)
    assert d.parts == (PartChange("banner", "size", "0x0", "0x1"),
                       PartChange("banner", "sha1", d.parts[1].vanilla, d.parts[1].patched))
    write_report(d, {}, tmp_path / "r.md", PROV)
    t = (tmp_path / "r.md").read_text()
    assert "## Outside the file system" in t and "| banner | size | 0x0 | 0x1 |" in t


def _calls_out(*targets):
    """Overlay 3 with Thumb BLs at 0x40 (in Foo) and 0x60 (in Bar) to ``targets``."""
    code = CODE
    for off, t in zip((0x40, 0x60), targets, strict=False):
        code = poke(code, off, thumb_bl(BASE + off, t))
    return code


def test_injected_and_unexplained_far_calls(tmp_path):
    injected = ("data/x.narc", 1, 0x02400000)
    v = fake(files={"data/x.narc": [b"a", b"short"]})
    p = fake(files={"data/x.narc": [b"a", b"\0" * 0x100]},
             ov3=_calls_out(0x02400010, 0x02500000))
    d = diff_roms(v, p, SYMS, [injected])
    (r,) = d.injected
    assert (r.path, r.member, r.load_address, r.vanilla_size, r.patched_size) == (
        "data/x.narc", 1, 0x02400000, 5, 0x100)
    assert r.calls == (FarCall("overlay3", "Foo", BASE + 0x40, 0x02400010),)
    assert d.far_calls == (FarCall("overlay3", "Bar", BASE + 0x60, 0x02500000),)
    write_report(d, {}, tmp_path / "r.md", PROV)
    t = (tmp_path / "r.md").read_text()
    assert "### data/x.narc member 1" in t and "5 -> 256 bytes, loaded at 0x02400000-0x02400100" in t
    assert "### Unexplained far calls (1)" in t and "| overlay3 | Bar | 0x02200060 | 0x02500000 |" in t


def test_calls_inside_modules_are_not_far():
    d = diff_roms(fake(), fake(ov3=_calls_out(BASE + 0xC0)), SYMS)
    assert d.code and d.far_calls == () and d.injected == ()


def test_declared_member_missing_raises():
    with pytest.raises(ValueError, match="data/x.narc member 3"):
        diff_roms(fake(), fake(files={"data/x.narc": [b"a"]}), None, [("data/x.narc", 3, 0)])
