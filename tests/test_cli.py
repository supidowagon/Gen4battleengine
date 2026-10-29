"""The ``nuzline`` command line: coverage, rank and trainers (plan Task 8)."""

from pathlib import Path

import pytest

from nuzline.cli import main, resolve_trainer
from nuzline.data.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
PARTY = str(ROOT / "tests" / "fixtures" / "sample_party.toml")
UNSUPPORTED = str(ROOT / "tests" / "fixtures" / "sample_party_unsupported.toml")


@pytest.fixture(autouse=True)
def _data(tables):
    """Every command needs the built tables (the ``tables`` fixture skips without them)
    and the extracted trainers."""
    if not (ROOT / "data" / "intermediate" / "trainers.json").exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")


def test_coverage_ok(capsys):
    assert main(["coverage", "--party", PARTY, "--trainer", "roark"]) == 0
    assert "missing: none" in capsys.readouterr().out.lower()


def test_coverage_failure_exit_code(capsys):
    assert main(["coverage", "--party", UNSUPPORTED, "--trainer", "roark"]) == 2
    assert "Metronome" in capsys.readouterr().out


def test_rank_prints_nine_rows(capsys):
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--n", "50"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 9 and "Switch" in out


def test_rank_output_is_plain_ascii(capsys):
    # Controller ruling: printable on a default Windows console (cp1252).
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--n", "20"]) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert out.isascii() and "Switch -> " in out


def test_rank_lead_and_opponent_active(capsys):
    # --lead 2 sends out Grotle (slot 2): its moves are ranked and Monferno (slot 0)
    # becomes a switch target; --opp-active 1 puts Onix out.
    args = ["rank", "--party", PARTY, "--trainer", "roark", "--n", "20"]
    assert main([*args, "--lead", "2", "--opp-active", "1"]) == 0
    out = capsys.readouterr().out.lower()
    assert "razor leaf" in out and "switch -> monferno" in out and "grotle vs onix" in out


def test_rank_rejects_a_fainted_or_missing_lead(capsys):
    fainted = str(ROOT / "tests" / "fixtures" / "sample_party_fainted.toml")
    assert main(["rank", "--party", fainted, "--trainer", "roark", "--lead", "5"]) == 1
    assert "has fainted" in capsys.readouterr().err
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--opp-active", "3"]) == 1
    assert "side 1: lead slot 3" in capsys.readouterr().err


def test_rank_refuses_unsupported_fight(capsys):
    assert main(["rank", "--party", UNSUPPORTED, "--trainer", "roark", "--n", "20"]) == 2
    assert "Metronome" in capsys.readouterr().out


def test_trainers_lists_names_and_ids(capsys):
    assert main(["trainers"]) == 0
    out = capsys.readouterr().out
    assert any(line.split()[:2] == ["246", "Roark"] for line in out.splitlines())


def _t(id: int, name: str, cls: str) -> Trainer:
    return Trainer(id, name, cls, 0, 0, 7, (), (), "none")


def test_trainer_name_prefers_a_leader_then_the_first_match():
    # Plan ruling: case-insensitive name; the first whose class starts with "leader",
    # else the first match.
    trainers = [_t(0, "Roark", "Hiker"), _t(1, "Roark", "Leader"), _t(2, "Roark", "Leader")]
    assert resolve_trainer(trainers, name="ROARK").id == 1
    assert resolve_trainer(trainers[:1], name="roark").id == 0
    with pytest.raises(ValueError, match="no trainer named"):
        resolve_trainer(trainers, name="Gardenia")


def test_trainer_id_selects_exactly():
    trainers = [_t(0, "Roark", "Leader"), _t(1, "Roark", "Leader")]
    assert resolve_trainer(trainers, trainer_id=1).id == 1
    with pytest.raises(ValueError, match="no trainer with id 5"):
        resolve_trainer(trainers, trainer_id=5)


def test_real_roark_is_trainer_246(capsys):
    assert main(["coverage", "--party", PARTY, "--trainer", "roark"]) == 0
    assert "Roark (Leader, id 246)" in capsys.readouterr().out
    # Roark's rematch (id 858) is a different fight: Full Restores, untranscribed moves.
    assert main(["coverage", "--party", PARTY, "--trainer-id", "858"]) == 2
    out = capsys.readouterr().out
    assert "Roark (Leader, id 858)" in out and "Full Restore" in out


def test_trainer_and_trainer_id_are_exclusive():
    with pytest.raises(SystemExit):
        main(["coverage", "--party", PARTY, "--trainer", "roark", "--trainer-id", "246"])
