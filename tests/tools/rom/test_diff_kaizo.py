"""tools.rom.diff on the real vanilla and Kaizo ROMs; skips without them or the map."""

from collections import Counter

from tools.rom.diff import HEADER_FIELDS, HeaderChange, diff_roms


def test_kaizo_diff_shape(vanilla_files, kaizo_files, symbols):
    d = diff_roms(vanilla_files, kaizo_files, symbols)
    assert d.modules["overlay16"] == (0x35A80, 0x35AA0)
    assert {c.module for c in d.code} >= {"arm9", "overlay5", "overlay13", "overlay14",
                                          "overlay16"}
    assert len([f for f in d.files if f.status == "changed"]) == 38
    assert len([f for f in d.files if f.status == "added"]) == 1751
    assert HeaderChange("game_code", "CPUE", "JAK7") in d.header
    assert all(h.field in HEADER_FIELDS and len(h.vanilla) < 32 and len(h.patched) < 32
               for h in d.header)
    unnamed = [c for c in d.code if c.symbol is None]
    print(f"\n{len(d.code)} code changes, {len(unnamed)} outside every symbol:",
          dict(Counter(c.module for c in unnamed)))
    for c in unnamed:
        print(f"  {c.module} {c.start:#010x}-{c.end:#010x} -> "
              f"{c.patched_start:#010x}-{c.patched_end:#010x}")
