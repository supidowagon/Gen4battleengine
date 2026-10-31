from hypothesis import given
from hypothesis import strategies as st

from nuzline.saves.crypto import (
    BLOCK_ORDERS,
    crc16_ccitt,
    decrypt_record,
    encrypt_record,
    lcrng_xor,
    record_checksum,
)


def test_crc_check_value():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_lcrng_first_words():
    out = lcrng_xor(b"\0" * 4, 0)
    assert out == (24691 >> 16).to_bytes(2, "little") + (
        ((24691 * 1103515245 + 24691) & 0xFFFFFFFF) >> 16
    ).to_bytes(2, "little")


def test_block_orders_are_permutations_and_repeat():
    assert all(sorted(o) == [0, 1, 2, 3] for o in BLOCK_ORDERS)
    assert len(set(BLOCK_ORDERS[:24])) == 24 and BLOCK_ORDERS[24:] == BLOCK_ORDERS[:8]
    assert BLOCK_ORDERS[0] == (0, 1, 2, 3) and BLOCK_ORDERS[6] == (1, 0, 2, 3)


@given(st.binary(min_size=236, max_size=236), st.booleans())
def test_encrypt_decrypt_round_trip(plain, party):
    plain = plain if party else plain[:136]
    assert decrypt_record(encrypt_record(plain)) == plain
    assert encrypt_record(decrypt_record(plain)) == plain


def test_decrypt_reorders_and_decrypts_known_record():
    # shuffle value 6 -> stored order (1,0,2,3): A sits at position 1.
    personality = 6 << 13
    plain = bytearray(236)
    plain[0:4] = personality.to_bytes(4, "little")
    for i in range(128):
        plain[8 + i] = i // 32 + 1  # blocks A..D filled with 1..4
    plain[6:8] = record_checksum(bytes(plain[8:136])).to_bytes(2, "little")
    raw = encrypt_record(bytes(plain))
    assert raw != bytes(plain)
    assert decrypt_record(raw) == bytes(plain)
    # undo only the stream: stored block 1 must hold A (all 1s).
    stored = lcrng_xor(raw[8:136], int.from_bytes(raw[6:8], "little"))
    assert stored[32:64] == b"\x01" * 32 and stored[0:32] == b"\x02" * 32
    # party extension is keyed by personality
    assert lcrng_xor(raw[136:], personality) == bytes(plain[136:])


def test_record_checksum_ignores_block_order():
    blocks = bytes(range(128))
    swapped = blocks[32:64] + blocks[:32] + blocks[64:]
    assert record_checksum(blocks) == record_checksum(swapped)
