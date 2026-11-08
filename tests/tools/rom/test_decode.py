"""tools.rom.decode on the real vanilla and Kaizo ROMs; skips without them or the map."""

import pytest

pytest.importorskip("capstone")

from tools.rom.decode import decode_code, decode_member, disassemble_at, main


@pytest.fixture(autouse=True)
def _needs_inputs(kaizo_inputs, symbols):
    """Skip without the ROMs, the Kaizo patch or the linker map."""


def test_decode_trainer_ai_init():
    (r,) = decode_code("overlay14", "TrainerAI_Init")  # 1 range, 1 byte (rom_diff.md)
    assert r.vanilla != r.patched and len(r.vanilla) == len(r.patched)
    assert any(line.startswith(">") for line in r.patched)


def test_decode_member_be_seq_182():
    van, pat = decode_member("battle/skill/be_seq.narc", 182)
    assert van != pat


def test_decode_member_new_sub_seq_has_no_vanilla_side():
    van, pat = decode_member("battle/skill/sub_seq.narc", 297)
    assert van == [] and pat


def test_disassemble_at_returns_requested_count():
    assert len(disassemble_at(0x02000800, 3)) == 3


def test_no_bytes_written(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["code", "overlay14", "TrainerAI_Init"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert "vanilla" in capsys.readouterr().out
