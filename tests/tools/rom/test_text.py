"""Message-bank decryption: synthetic round trips and real-ROM checks."""

import struct

import pytest

from tools.extract_decomp import ExtractError, _text_bank, extract_species
from tools.rom.readers import ReaderError, fixed_members, symbol_bytes
from tools.rom.readers.text import decode_bank, message_bank


def encode_bank(entries, seed, charmap):
    """Inverse of ``decode_bank``; a ``list[int]`` entry is written as raw codes."""
    rev = {v: k for k, v in charmap.items()}
    coded = []
    for e in entries:
        codes = list(e) if isinstance(e, list) else [rev[c] for c in e]
        coded.append(codes + [0xFFFF])
    head = 4 + 8 * len(coded)
    body = b""
    table = []
    for i, codes in enumerate(coded):
        key = (i + 1) * 596947
        enc = []
        for c in codes:
            enc.append((c ^ key) & 0xFFFF)
            key = (key + 18749) & 0xFFFF
        k = (seed * 765 * (i + 1)) & 0xFFFF
        k |= k << 16
        table.append(((head + len(body)) ^ k, len(codes) ^ k))
        body += struct.pack(f"<{len(enc)}H", *enc)
    out = struct.pack("<HH", len(coded), seed)
    for off, ln in table:
        out += struct.pack("<II", off, ln)
    return out + body


def test_round_trip_synthetic(charmap):
    assert decode_bank(encode_bank(["Abra", "Mr. Mime", ""], 0x1234, charmap), charmap) == [
        "Abra",
        "Mr. Mime",
        "",
    ]


def test_unknown_code_raises(charmap):
    bad = encode_bank([[0x0FFE]], 0x1234, charmap)
    assert 0x0FFE not in charmap
    with pytest.raises(ReaderError, match="entry 0.*0xffe"):
        decode_bank(bad, charmap)


def test_compressed_marker_raises(charmap):
    bad = encode_bank(["Ab", [0xF100, 5]], 0x1234, charmap)
    with pytest.raises(ReaderError, match="entry 1.*0xf100"):
        decode_bank(bad, charmap)


def test_premature_terminator_raises(charmap):
    bad = encode_bank([[0x0001, 0xFFFF, 0x0001]], 0x1234, charmap)
    # encode_bank appends its own terminator, so entry 0 is 4 codes with one at index 1
    with pytest.raises(
        ReaderError, match=r"entry 0 \(offset 0x[0-9a-f]+\).*terminator at character 1"
    ):
        decode_bank(bad, charmap)


def test_missing_terminator_raises(charmap):
    good = encode_bank([[0x0001]], 0x1234, charmap)
    # drop the terminator: shrink the entry length by one character
    import struct

    seed = 0x1234
    k = (seed * 765) & 0xFFFF
    k |= k << 16
    _off, length = struct.unpack_from("<II", good, 4)
    bad = good[:8] + struct.pack("<I", ((length ^ k) - 1) ^ k) + good[12:]
    with pytest.raises(
        ReaderError, match=r"entry 0 \(offset 0x[0-9a-f]+\).*not the 0xffff terminator"
    ):
        decode_bank(bad, charmap)


def test_offset_inside_table_raises(charmap):
    good = encode_bank(["A"], 0x1234, charmap)
    seed = 0x1234
    k = (seed * 765) & 0xFFFF
    k |= k << 16
    import struct

    bad = good[:4] + struct.pack("<I", 4 ^ k) + good[8:]
    with pytest.raises(ReaderError, match="entry 0.*inside the entry table"):
        decode_bank(bad, charmap)


def test_truncated_bank_raises(charmap):
    data = encode_bank(["Abra"], 0x1234, charmap)
    with pytest.raises(ReaderError, match="entry 0"):
        decode_bank(data[:-4], charmap)
    with pytest.raises(ReaderError):
        decode_bank(data[:2], charmap)


def test_fixed_members_reports_member():
    assert fixed_members([b"ab", b"cd"], 2, "x.narc") == [b"ab", b"cd"]
    with pytest.raises(ReaderError, match=r"x\.narc.*member 1.*expected 2.*got 3"):
        fixed_members([b"ab", b"cde"], 2, "x.narc")


def test_symbol_bytes(vanilla_files, symbols):
    sym = symbols.find("sNatureStatAffinities")
    data = symbol_bytes(vanilla_files, symbols, "sNatureStatAffinities")
    assert len(data) == sym.size == 25 * 5
    # src/pokemon.c:2415: s8 [nature][stat - 1], stats in the order Attack, Defense, Speed,
    # Sp. Atk, Sp. Def; nature n raises stat n // 5 and lowers stat n % 5 (neutral if equal).
    want = bytes(
        (int(k == n // 5) - int(k == n % 5)) & 0xFF for n in range(25) for k in range(5)
    )
    assert data == want
    ram, arm9 = vanilla_files.arm9_sections()[0]
    assert data == arm9[sym.address - ram : sym.address - ram + sym.size]


def test_symbol_bytes_refuses_resized_overlay(kaizo_files, symbols):
    # Kaizo's overlay 16 is 0x20 bytes longer and its type chart 28 bytes further on.
    with pytest.raises(ReaderError, match=r"sTypeMatchupMultipliers.*overlay 16.*0x35aa0.*0x35a80"):
        symbol_bytes(kaizo_files, symbols, "sTypeMatchupMultipliers")


def test_symbol_bytes_without_extent_raises(vanilla_files, symbols):
    from tools.rom.symbols import SymbolMap

    bare = SymbolMap([symbols.find("sNatureStatAffinities")])
    with pytest.raises(ReaderError, match="no extent"):
        symbol_bytes(vanilla_files, bare, "sNatureStatAffinities")


def test_symbol_bytes_outside_module_raises(vanilla_files, symbols):
    bss = next(s for s in symbols.symbols("arm9") if s.section == ".bss" and s.size)
    with pytest.raises(ReaderError, match=bss.name):
        symbol_bytes(vanilla_files, symbols, bss.name)


def test_vanilla_banks_match_decomp(vanilla_files, decomp, charmap):
    names = message_bank(vanilla_files, decomp, "TEXT_BANK_SPECIES_NAME", charmap)
    species = extract_species(decomp)
    # speciesproc.c:846 writes "-----" for SPECIES_NONE; the pokedex data names it "----------"
    assert names[0] == "-----"
    assert names[1 : len(species)] == [s["name"] for s in species[1:]]
    assert message_bank(vanilla_files, decomp, "TEXT_BANK_ABILITY_NAMES", charmap) == _text_bank(
        decomp, "ability_names"
    )


def test_unknown_bank_raises(vanilla_files, decomp, charmap):
    with pytest.raises(ExtractError, match="TEXT_BANK_NOPE"):
        message_bank(vanilla_files, decomp, "TEXT_BANK_NOPE", charmap)


def test_kaizo_renamed_move(kaizo_files, vanilla_files, decomp, charmap):
    """Vanilla Comet Punch (move 4) is "Water Ball" in Kaizo (spec 5.1)."""
    moves = message_bank(kaizo_files, decomp, "TEXT_BANK_MOVE_NAMES", charmap)
    assert moves[4] == "Water Ball"
    van = message_bank(vanilla_files, decomp, "TEXT_BANK_MOVE_NAMES", charmap)
    assert van[4] != moves[4]


def pack_name(codes):
    """9-bit codes packed 15 bits to a u16, low bits first (MessagesDecoder.cpp:69-88)."""
    bits = sum(c << (9 * i) for i, c in enumerate(codes))
    n = (9 * len(codes) + 14) // 15
    return [(bits >> (15 * i)) & 0x7FFF for i in range(n)]


def test_compressed_trainer_name_round_trip(charmap):
    rev = {v: k for k, v in charmap.items()}
    codes = [rev[c] for c in "Roark"]
    assert all(c < 0x1FF for c in codes)
    bank = encode_bank([[0xF100, *pack_name([*codes, 0x1FF])], "Plain"], 0x1234, charmap)
    assert decode_bank(bank, charmap, trainer_names=True) == ["Roark", "Plain"]


def test_malformed_compressed_name_raises_reader_error(charmap):
    from tools.rom.readers.text import _trainer_name

    rev = {v: k for k, v in charmap.items()}
    # Words that end in the middle of a code, and words with no 0x1ff code at all.
    with pytest.raises(ReaderError, match="ends inside a 9-bit code"):
        _trainer_name(pack_name([rev["A"], rev["B"]]), "entry 0", charmap)
    with pytest.raises(ReaderError, match="no 0x1ff terminator"):
        _trainer_name(pack_name([rev["A"]] * 5), "entry 0", charmap)
    missing = next(c for c in range(0x1FF) if c not in charmap)
    bank = encode_bank([[0xF100, *pack_name([missing, 0x1FF])]], 0x1234, charmap)
    with pytest.raises(ReaderError, match=f"entry 0.*{missing:#x} is not in the charmap"):
        decode_bank(bank, charmap, trainer_names=True)
