"""Message-bank decryption: synthetic round trips and real-ROM checks."""

import struct

import pytest

from tools.extract_decomp import _text_bank, extract_species
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
    assert len(data) == sym.size > 0


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
    with pytest.raises(Exception, match="TEXT_BANK_NOPE"):
        message_bank(vanilla_files, decomp, "TEXT_BANK_NOPE", charmap)


def test_kaizo_renamed_move(kaizo_files, vanilla_files, decomp, charmap):
    """Vanilla Comet Punch (move 4) is "Water Ball" in Kaizo (spec 5.1)."""
    moves = message_bank(kaizo_files, decomp, "TEXT_BANK_MOVE_NAMES", charmap)
    assert moves[4] == "Water Ball"
    van = message_bank(vanilla_files, decomp, "TEXT_BANK_MOVE_NAMES", charmap)
    assert van[4] != moves[4]
