"""The ``nuzline`` command line: coverage, rank and trainers (plan Task 8)."""

import json
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
    from nuzline.data.tables import DEFAULT
    from nuzline.jaxcache import table_version

    assert run["result"]["table_version"] == saved["table_version"] == table_version(DEFAULT)
    assert run["result"]["leaf_budget"] == 2000


def test_cli_solve_refused(capsys, tmp_path, monkeypatch):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    args = ["solve", "--party", UNSUPPORTED, "--trainer", "roark", "--devices", "1"]
    assert main(args) == 2
    out = capsys.readouterr().out
    assert "refused:" in out and "Metronome" in out
    assert not (tmp_path / "runs").exists()


def test_cli_solve_all_dropped_is_a_refusal(capsys, tmp_path, monkeypatch):
    """Every root action sampling-only: no ranking, so ``refused:`` + dropped reasons, exit 2
    and no run record."""
    from types import SimpleNamespace

    from nuzline import cli
    from nuzline.search import stage_b

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    fake = SimpleNamespace(ranking=[], dropped={0: "Spike Cannon is multi-hit (sampling only)"})
    monkeypatch.setattr(stage_b, "solve", lambda *a, **k: fake)
    args = ["solve", "--party", PARTY, "--trainer", "roark", "--devices", "1"]
    assert main(args) == 2
    out = capsys.readouterr().out
    assert "refused:" in out and "Spike Cannon" in out
    assert not (tmp_path / "runs").exists()


def test_cli_solve_unsupported_ai_is_a_refusal(capsys, tmp_path, monkeypatch):
    """``solve`` raising ``UnsupportedAi`` (the AI refused at the root): ``refused:`` with
    the reason, exit 2 and no run record."""
    from nuzline import cli
    from nuzline.ai.policy import UnsupportedAi
    from nuzline.search import stage_b

    def refuse(*args, **kwargs):
        raise UnsupportedAi("Spike Cannon: AI script not transcribed")

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(stage_b, "solve", refuse)
    args = ["solve", "--party", PARTY, "--trainer", "roark", "--devices", "1"]
    assert main(args) == 2
    out = capsys.readouterr().out
    assert "refused: Spike Cannon: AI script not transcribed" in out
    assert not (tmp_path / "runs").exists()


def test_cli_solve_rejects_a_negative_budget(capsys):
    args = ["solve", "--party", PARTY, "--trainer", "roark", "--budget", "-1", "--devices", "1"]
    assert main(args) == 1
    assert "--budget must not be negative" in capsys.readouterr().err


def test_run_records_never_overwrite(tmp_path, monkeypatch):
    """Two records in the same microsecond get two files (open mode ``x``)."""
    import json
    from datetime import datetime

    from nuzline import cli

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 31, 12, 0, 0, 123456)

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cli, "datetime", Frozen)
    first = cli._write_run_record({"n": 1})
    second = cli._write_run_record({"n": 2})
    assert first.name == "solve_20261031_120000_123456.json"
    assert second.name == "solve_20261031_120000_123456_1.json"
    assert json.loads(first.read_text()) == {"n": 1}
    assert json.loads(second.read_text()) == {"n": 2}


def _solve_settings(monkeypatch, tmp_path, extra: list[str]):
    """The ``SolveSettings`` ``nuzline solve`` passes for ``extra`` flags (``solve`` is a
    stub returning no ranking, so the command stops with exit 2 after it)."""
    from types import SimpleNamespace

    from nuzline import cli
    from nuzline.search import stage_b

    seen = []

    def fake(tables, names, state, key, settings, *a, **k):
        seen.append(settings)
        return SimpleNamespace(ranking=[], dropped={0: "stub"})

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(stage_b, "solve", fake)
    args = ["solve", "--party", PARTY, "--trainer", "roark", "--devices", "1", *extra]
    assert main(args) == 2
    (settings,) = seen
    return settings


def test_cli_solve_precision_is_the_default(monkeypatch, tmp_path):
    settings = _solve_settings(monkeypatch, tmp_path, [])
    assert settings.precision == pytest.approx(0.015) and settings.budget_playouts is None
    settings = _solve_settings(monkeypatch, tmp_path, ["--precision", "0.5"])
    assert settings.precision == pytest.approx(0.005) and settings.budget_playouts is None


def test_cli_solve_playouts_turn_precision_off(monkeypatch, tmp_path):
    settings = _solve_settings(monkeypatch, tmp_path, ["--playouts", "5000"])
    assert settings.precision is None and settings.budget_playouts == 5000


def test_cli_solve_rejects_bad_precision(capsys):
    base = ["solve", "--party", PARTY, "--trainer", "roark", "--devices", "1"]
    for bad in ("0", "-1"):
        assert main([*base, "--precision", bad]) == 1
        assert "--precision must be positive" in capsys.readouterr().err
    assert main([*base, "--precision", "1", "--playouts", "100"]) == 1
    assert "--precision and --playouts" in capsys.readouterr().err


# --- saves ---------------------------------------------------------------------------


@pytest.fixture
def save_path(tmp_path, tables, names):
    from tests.saves.synthetic import write_save

    return write_save(tmp_path, tables, names)


def test_save_show(save_path, capsys):
    assert main(["save", "show", str(save_path)]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "copy: primary" in out and "Trainer:" in out
    assert "MONFERNO Lv14" in out and "Ember" in out and "Party (3)" in out


def test_save_show_json(save_path, capsys):
    assert main(["save", "show", str(save_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["party"][0]["species_name"] == "MONFERNO"
    assert data["load_result"] == "ok"


def test_save_export_round_trips(save_path, tmp_path, tables, names, capsys):
    from nuzline.data.party import load_party
    from nuzline.saves import load_save

    out = tmp_path / "party.toml"
    assert main(["save", "export", str(save_path), "-o", str(out)]) == 0
    save = load_save(save_path, tables=tables, names=names)
    assert load_party(out, names) == [m.to_spec(tables) for m in save.party]
    with pytest.raises(SystemExit) as e:  # -o is required
        main(["save", "export", str(save_path)])
    assert e.value.code == 2


def test_rank_with_save(save_path, capsys):
    args = ["rank", "--save", str(save_path), "--trainer", "roark", "--n", "20", "--devices", "1"]
    assert main(args) == 0
    assert "Switch" in capsys.readouterr().out


def test_save_egg_first_is_dropped(tmp_path, tables, names, capsys):
    from tests.saves.synthetic import write_save

    path = write_save(tmp_path, tables, names, egg_first=True)
    args = ["coverage", "--save", str(path), "--trainer", "roark"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "party slot 0: egg left out" in captured.err
    # --lead indexes the egg-free party: 2 is Grotle, 3 does not exist.
    assert main([*args, "--lead", "2"]) == 0
    assert main([*args, "--lead", "3"]) == 1


def test_party_and_save_exclusive(save_path):
    with pytest.raises(SystemExit) as e:
        main(["coverage", "--party", PARTY, "--save", str(save_path), "--trainer", "roark"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["coverage", "--trainer", "roark"])
    assert e.value.code == 2


def test_corrupt_save_exit_1(tmp_path, tables, names, capsys):
    from tests.saves.synthetic import save_image
    from tests.saves.writer import set_save_counter

    image = save_image(tables, names, counters=((5, 5, 5), (6, 6, 4)))
    image = set_save_counter(image, 1, "boxes", 5)
    path = tmp_path / "bad.sav"
    path.write_bytes(image)
    assert main(["save", "show", str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("nuzline: error:") and len(err.strip().splitlines()) == 1
    assert main(["coverage", "--save", str(path), "--trainer", "roark"]) == 1
    capsys.readouterr()
    assert main(["save", "show", str(path), "--allow-corrupt"]) == 0
    assert "load result corrupt" in capsys.readouterr().out
    short = tmp_path / "short.sav"
    short.write_bytes(b"\0" * 10)
    assert main(["save", "show", str(short)]) == 1
    assert "not a Platinum save" in capsys.readouterr().err


def test_run_record_hashes_the_save(save_path):
    import argparse
    import hashlib

    from nuzline import cli

    got = cli._source_hash(argparse.Namespace(save=save_path, party=None))
    assert got == {"save_sha256": hashlib.sha256(save_path.read_bytes()).hexdigest()}
    got = cli._source_hash(argparse.Namespace(save=None, party=Path(PARTY)))
    assert list(got) == ["party_sha256"]


def test_save_export_unwritable_output_exit_1(save_path, tmp_path, capsys):
    out = tmp_path / "no_such_dir" / "party.toml"
    assert main(["save", "export", str(save_path), "-o", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("nuzline: error:") and len(err.strip().splitlines()) == 1


def _pool_save(tmp_path, tables, names, box_moves=("Growl", "Bide")):
    from tests.saves.synthetic import box_record, party_record
    from tests.saves.writer import build_image

    path = tmp_path / "pool.sav"
    path.write_bytes(
        build_image(
            party=[
                party_record(tables, names, "Furret", 16, ("Scratch",), ability_slot=1),
                party_record(tables, names, "Geodude", 16, ("Tackle",)),
            ],
            boxes={
                (0, 0): box_record(tables, names, "Kricketot", 5, box_moves),
                (0, 1): box_record(tables, names, "Bidoof", 4, ("Tackle",)),
            },
            bag={"tms_hms": [(names.item("TM10"), 1)]},
        )
    )
    return path


def test_coverage_save_reports_pool(tmp_path, tables, names, capsys):
    path = _pool_save(tmp_path, tables, names)
    assert main(["coverage", "--save", str(path), "--trainer", "roark"]) == 0
    out = capsys.readouterr().out
    assert "Pool at level cap 16 (4 Pokemon):" in out
    assert out.count("Support: EXACT") == 2


def test_coverage_pool_does_not_change_exit_code(tmp_path, tables, names, capsys):
    path = _pool_save(tmp_path, tables, names, box_moves=("Metronome",))
    assert main(["coverage", "--save", str(path), "--trainer", "roark"]) == 0
    out = capsys.readouterr().out
    assert "box 1 slot 0 KRICKETOT: Metronome (current, move effect UNSUPPORTED)" in out
    assert "Support: NONE" in out


def test_cap_needs_save(capsys):
    assert main(["coverage", "--party", PARTY, "--trainer", "roark", "--cap", "16"]) == 1
    assert "--cap needs --save" in capsys.readouterr().err
