import pytest

from tests.tools.rom.vcdiff_writer import Window, encode
from tools.rom.vcdiff import VcdiffError, apply_vcdiff

SRC = bytes(range(256)) * 4
W = Window((0, 1024), False, [("add", b"hi"), ("copy", 10, 20, 0), ("run", 7, 5)])
WANT = b"hi" + SRC[10:30] + b"\x07" * 5


def test_add_run_copy_from_source():
    assert apply_vcdiff(SRC, encode([W], source=SRC)) == WANT


def test_overlapping_self_copy():
    p = encode([Window(None, False, [("add", b"ab"), ("copy", 0, 10, 0)])])
    assert apply_vcdiff(b"", p) == b"ab" * 6


def test_every_address_mode():
    # modes 0 self, 1 here, 2-5 near, 6-8 same; each address reachable given the caches so far
    plan = [(0, 10), (1, 40), (2, 12), (3, 44), (4, 600), (5, 300), (6, 10), (7, 300), (8, 600)]
    ops = [("add", b"x")] + [("copy", a, 4, m) for m, a in plan]
    p = encode([Window((0, 1024), False, ops)], source=SRC)
    assert apply_vcdiff(SRC, p) == b"x" + b"".join(SRC[a : a + 4] for _, a in plan)


def test_two_windows_and_target_window():
    p = encode(
        [Window(None, False, [("add", b"hello")]), Window((0, 5), True, [("copy", 0, 5, 0)])]
    )
    assert apply_vcdiff(b"", p) == b"hellohello"


def test_app_header_and_secondary_lzma():  # one window: the writer finishes each xz stream
    p = encode([W], app_header=b"t.nds//s.nds/", secondary=True, source=SRC)
    assert p[4] == 0x05 and apply_vcdiff(SRC, p) == WANT


def test_bad_adler_raises():
    with pytest.raises(VcdiffError, match="window 0.*Adler"):
        apply_vcdiff(SRC, encode([W], adler=0x12345678, source=SRC))


def test_truncated_patch_raises():
    with pytest.raises(VcdiffError, match="window 0"):
        apply_vcdiff(SRC, encode([W], source=SRC)[:-3])


def test_custom_code_table_refused():
    p = encode([W], source=SRC)
    p = p[:4] + bytes([p[4] | 0x02]) + p[5:]
    with pytest.raises(VcdiffError, match="code table"):
        apply_vcdiff(SRC, p)


def test_bad_magic_raises():
    with pytest.raises(VcdiffError, match="offset 0.*magic"):
        apply_vcdiff(SRC, b"\xd6\xc3\xc4\x01\x00")


def test_trailing_data_raises():
    with pytest.raises(VcdiffError, match="window 1 "):
        apply_vcdiff(SRC, encode([W], source=SRC) + b"\x00")


def test_copy_beyond_written_target_raises():
    # The writer cannot reach an address at or past "here" in HERE mode, so use SELF.
    p = encode([Window(None, False, [("add", b"ab"), ("copy", 2, 1, 0)])])
    with pytest.raises(VcdiffError, match="window 0 .*COPY"):
        apply_vcdiff(b"", p)


def test_source_segment_outside_source_raises():
    with pytest.raises(VcdiffError, match="window 0 .*source segment"):
        apply_vcdiff(SRC[:100], encode([W], source=SRC))


def test_unsupported_compressor_raises():
    p = encode([W], secondary=True, source=SRC)
    assert p[5] == 2  # compressor id follows the header indicator (no app header here)
    p = p[:5] + b"\x01" + p[6:]
    with pytest.raises(VcdiffError, match="compressor"):
        apply_vcdiff(SRC, p)


def test_writer_refuses_unreachable_mode():
    with pytest.raises(ValueError):
        encode([Window(None, False, [("add", b"ab"), ("copy", 1, 1, 6)])])
