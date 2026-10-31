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


@pytest.fixture(autouse=True)
def _restore_xla_flags(monkeypatch):
    """``main(["rank", ...])`` appends to ``XLA_FLAGS``; undo that after each test.

    ``setenv`` first so monkeypatch records the original state (set or absent) and
    restores it at teardown; ``delenv`` on an absent variable alone records nothing."""
    monkeypatch.setenv("XLA_FLAGS", "")
    monkeypatch.delenv("XLA_FLAGS")


def test_rank_devices_option_sets_the_flag(monkeypatch):
    import os

    monkeypatch.setattr("nuzline.cli._rank", lambda args: 0)
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--devices", "3"]) == 0
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=3"


def test_coverage_ok(capsys):
    assert main(["coverage", "--party", PARTY, "--trainer", "roark"]) == 0
    assert "missing: none" in capsys.readouterr().out.lower()


def test_coverage_failure_exit_code(capsys):
    assert main(["coverage", "--party", UNSUPPORTED, "--trainer", "roark"]) == 2
    assert "Metronome" in capsys.readouterr().out


def test_coverage_checks_the_state_rank_would(capsys, monkeypatch):
    from nuzline.engine import coverage

    seen = []
    check = coverage.check_coverage

    def spy(*args, **kwargs):
        seen.append(kwargs.get("leads", args[6] if len(args) > 6 else None))
        return check(*args, **kwargs)

    monkeypatch.setattr(coverage, "check_coverage", spy)
    args = ["coverage", "--party", PARTY, "--trainer", "roark"]
    assert main(args) == 0
    assert main([*args, "--lead", "2", "--opp-active", "1"]) == 0
    assert seen == [(0, 0), (2, 1)]
    capsys.readouterr()
    # The same validation as rank: a missing opponent slot and a fainted lead.
    assert main([*args, "--opp-active", "3"]) == 1
    assert "side 1: lead slot 3" in capsys.readouterr().err
    fainted = str(ROOT / "tests" / "fixtures" / "sample_party_fainted.toml")
    assert main(["coverage", "--party", fainted, "--trainer", "roark", "--lead", "5"]) == 1
    assert "has fainted" in capsys.readouterr().err


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


def _ranked_rows(capsys, monkeypatch, rows) -> list[str]:
    from nuzline.search import stage_a

    monkeypatch.setattr(stage_a, "rank_actions", lambda *a, **k: stage_a.Ranking.of(rows))
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--n", "100"]) == 0
    return capsys.readouterr().out.splitlines()


def test_rank_marks_only_the_rows_tied_with_the_leader(capsys, monkeypatch):
    from nuzline.search.stage_a import ActionResult

    legend = "* the interval overlaps the leader's: a statistical tie, not reordered"
    tied = [
        ActionResult(0, "Scratch", 0.50, 0.40, 0.60, 0.7, 0.3, 100),
        ActionResult(1, "Leer", 0.45, 0.35, 0.55, 0.6, 0.4, 100),
        ActionResult(2, "Ember", 0.10, 0.05, 0.18, 0.1, 0.9, 100),
    ]
    out = _ranked_rows(capsys, monkeypatch, tied)
    rows = [line for line in out if line[4:6].strip().isdigit()]
    assert [line[2] for line in rows] == [" ", "*", " "]  # the leader is not marked
    assert legend in out
    clear = [tied[0], tied[2]]
    out = _ranked_rows(capsys, monkeypatch, clear)
    assert legend not in out


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


def test_configure_cpu_devices_appends_flag():
    from nuzline.cli import configure_cpu_devices

    env = {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}
    configure_cpu_devices(4, env)
    assert env["XLA_FLAGS"] == (
        "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=4"
    )
    empty: dict[str, str] = {}
    configure_cpu_devices(None, empty)
    assert empty["XLA_FLAGS"] == (
        f"--xla_force_host_platform_device_count={__import__('os').cpu_count()}"
    )


def test_configure_cpu_devices_keeps_user_flag():
    from nuzline.cli import configure_cpu_devices

    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=3"}
    configure_cpu_devices(8, env)
    assert env["XLA_FLAGS"] == "--xla_force_host_platform_device_count=3"


def test_cli_module_does_not_import_jax():
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = "import nuzline.cli, sys; assert 'jax' not in sys.modules"
    done = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_cli_solve_small(tmp_path, monkeypatch, capsys):
    import json

    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    tree = tmp_path / "t.json"
    args = ["solve", "--party", PARTY, "--trainer", "roark", "--max-depth", "1"]
    args += ["--playouts", "2000", "--tree", str(tree), "--devices", "1"]
    assert main(args) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert out.isascii() and "P(win, 0 faints)" in out
    assert any(line.lstrip().startswith("if ") for line in out.splitlines())
    saved = json.loads(tree.read_text())
    assert saved["depth"] == 1 and saved["plan"]["children"]
    (record,) = (tmp_path / "runs").glob("solve_*.json")
    run = json.loads(record.read_text())
    import hashlib

    assert run["party_sha256"] == hashlib.sha256(Path(PARTY).read_bytes()).hexdigest()
    assert run["trainer_id"] == 246 and run["seed"] == 0
    assert run["settings"]["budget_playouts"] == 2000 and run["result"]["depth"] == 1


def test_cli_solve_refused(capsys, tmp_path, monkeypatch):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    args = ["solve", "--party", UNSUPPORTED, "--trainer", "roark", "--devices", "1"]
    assert main(args) == 2
    out = capsys.readouterr().out
    assert "refused:" in out and "Metronome" in out
    assert not (tmp_path / "runs").exists()
