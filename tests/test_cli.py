"""The ``nuzline`` command line: coverage, rank and trainers (plan Task 8)."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nuzline.cli import main, resolve_trainer
from nuzline.data import games
from nuzline.data.games import Game
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


def test_coverage_failure_exit_code(capsys, metronome_refused):
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


@pytest.mark.slow
def test_rank_prints_nine_rows(capsys):
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--n", "50"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 9 and "Switch" in out


@pytest.mark.slow
def test_rank_output_is_plain_ascii(capsys):
    # Controller ruling: printable on a default Windows console (cp1252).
    assert main(["rank", "--party", PARTY, "--trainer", "roark", "--n", "20"]) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert out.isascii() and "Switch -> " in out


def _labels_only_ranking(monkeypatch):
    """``stage_a.rank_actions`` replaced by one that ranks the state's legal actions by
    their labels (``stage_a.action_label``) without playouts: the CLI's own state building,
    validation and printing run as in a real ``rank``; the playouts are the slow tests'."""
    import numpy as np

    from nuzline.engine.actions import decision_actions, usable_moves
    from nuzline.search import stage_a

    def labels_only(tables, names, state, n, key, max_turns=100):
        legal = np.flatnonzero(np.asarray(decision_actions(tables, state)))
        struggle = not bool(np.any(np.asarray(usable_moves(tables, state, 0))))
        rows = [
            stage_a.ActionResult(
                int(a),
                stage_a.action_label(names, state, int(a), struggle),
                0.5,
                0.4,
                0.6,
                0.5,
                0.5,
                n,
            )
            for a in legal
        ]
        return stage_a.Ranking.of(rows)

    monkeypatch.setattr(stage_a, "rank_actions", labels_only)


def test_rank_lead_and_opponent_active(capsys, monkeypatch):
    # --lead 2 sends out Grotle (slot 2): its moves are ranked and Monferno (slot 0)
    # becomes a switch target; --opp-active 1 puts Onix out. The fast tier ranks without
    # playouts (_labels_only_ranking); the real ranking is test_rank_lead_and_opponent_active_played.
    _labels_only_ranking(monkeypatch)
    args = ["rank", "--party", PARTY, "--trainer", "roark", "--n", "20"]
    assert main([*args, "--lead", "2", "--opp-active", "1"]) == 0
    out = capsys.readouterr().out.lower()
    assert "razor leaf" in out and "switch -> monferno" in out and "grotle vs onix" in out


@pytest.mark.slow
def test_rank_lead_and_opponent_active_played(capsys):
    # As test_rank_lead_and_opponent_active, with the real playouts.
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


def test_rank_refuses_unsupported_fight(capsys, metronome_refused):
    assert main(["rank", "--party", UNSUPPORTED, "--trainer", "roark", "--n", "20"]) == 2
    assert "Metronome" in capsys.readouterr().out


def test_trainers_lists_names_and_ids(capsys):
    assert main(["trainers"]) == 0
    out = capsys.readouterr().out
    assert any(line.split()[:2] == ["246", "Roark"] for line in out.splitlines())


def _t(id: int, name: str, cls: str) -> Trainer:
    return Trainer(id, name, cls, 0, 0, 7, (), (), "none", "single")


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
    # Roark's rematch (id 858) is a different fight: untranscribed moves (its Full Restores
    # act by their party params, E1 ruling G-43, so they are not refused).
    assert main(["coverage", "--party", PARTY, "--trainer-id", "858"]) == 2
    out = capsys.readouterr().out
    assert "Roark (Leader, id 858)" in out and "Full Restore" not in out


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


@pytest.mark.slow
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


def test_cli_solve_refused(capsys, tmp_path, monkeypatch, metronome_refused):
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
    assert load_party(out, names) == [m.to_spec(tables, save.trainer) for m in save.party]
    with pytest.raises(SystemExit) as e:  # -o is required
        main(["save", "export", str(save_path)])
    assert e.value.code == 2


def test_rank_with_save(save_path, capsys, monkeypatch):
    # The fight from a save, ranked without playouts (_labels_only_ranking); the real
    # ranking is test_rank_with_save_played.
    _labels_only_ranking(monkeypatch)
    args = ["rank", "--save", str(save_path), "--trainer", "roark", "--n", "20", "--devices", "1"]
    assert main(args) == 0
    assert "Switch" in capsys.readouterr().out


@pytest.mark.slow
def test_rank_with_save_played(save_path, capsys):
    args = ["rank", "--save", str(save_path), "--trainer", "roark", "--n", "20", "--devices", "1"]
    assert main(args) == 0
    assert "Switch" in capsys.readouterr().out


def test_a_fight_from_a_save_resolves_the_exp_boost_with_its_trainer(
    tmp_path, tables, names, monkeypatch
):
    # K3d-exp T3: the fight's player specs come from SaveMon.to_spec with the save's
    # trainer (cli._fight), so exp_boost is resolved (100 for the save's own Pokemon).
    from nuzline.saves.pkm import SaveMon
    from tests.saves.synthetic import write_save

    specs = []
    to_spec = SaveMon.to_spec

    def spy(self, tables, trainer=None):
        specs.append(to_spec(self, tables, trainer))
        return specs[-1]

    monkeypatch.setattr(SaveMon, "to_spec", spy)
    path = write_save(tmp_path, tables, names)
    assert main(["coverage", "--save", str(path), "--trainer", "roark"]) == 0
    assert specs and {s.exp_boost for s in specs} == {100}


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


def test_coverage_pool_does_not_change_exit_code(
    tmp_path, tables, names, capsys, metronome_refused
):
    path = _pool_save(tmp_path, tables, names, box_moves=("Metronome",))
    assert main(["coverage", "--save", str(path), "--trainer", "roark"]) == 0
    out = capsys.readouterr().out
    assert "box 1 slot 0 KRICKETOT: Metronome (current, move effect UNSUPPORTED)" in out
    assert "Support: NONE" in out


def test_cap_needs_save(capsys):
    assert main(["coverage", "--party", PARTY, "--trainer", "roark", "--cap", "16"]) == 1
    assert "--cap needs --save" in capsys.readouterr().err


# ---- --deaths and --protect (plan 4b Task 10) ----

SAMPLE = PARTY
FAINTED = str(ROOT / "tests" / "fixtures" / "sample_party_fainted.toml")


def test_protected_slots(tables, names):
    from nuzline.cli import protected_slots
    from nuzline.data.party import load_party

    party = load_party(Path(PARTY), names)  # Monferno Prinplup Grotle Starly Shinx Budew
    # Digits are 0-based party indices, like --lead (plan 4b ruling).
    assert protected_slots(party, names, ["1"]) == (1,)
    assert protected_slots(party, names, ["prinplup", "1"]) == (1,)
    assert protected_slots(party, names, ["0", "Starly"]) == (0, 3)
    assert protected_slots(party, names, ["STARLY", "starly"]) == (3,)
    assert protected_slots(party, names, []) == ()
    for bad in ("6", "9", "-1", "Pikachu"):
        with pytest.raises(ValueError, match=f"--protect {bad}"):
            protected_slots(party, names, [bad])


def test_all_zero_footer_names_the_budget():
    from nuzline.cli import ALL_ZERO_FOOTER, all_zero_footer

    assert all_zero_footer(0) == ALL_ZERO_FOOTER
    assert all_zero_footer(2) == "all lines 0 %: no line within 2 deaths found at this depth"


# Slow: rank compiles or loads the whole-engine playout program (8-10 GB resident, minutes
# cold), past the fast tier's 60 s and 4 GB; so does the test below.
@pytest.mark.slow
def test_cli_rank_deaths_and_protect(capsys):
    args = ["rank", "--party", SAMPLE, "--trainer", "roark", "--deaths", "1"]
    assert main([*args, "--protect", "Prinplup", "--n", "50", "--devices", "1"]) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert out.isascii()
    assert "P(win, <=1 deaths)" in out and "E[deaths]" in out
    assert "<=0" in out and "<=3" in out and "(screen)" in out
    assert "protected: prinplup" in out.lower()
    assert "deaths under the leader:" in out


@pytest.mark.slow
def test_cli_rank_default_budget_keeps_the_header(capsys):
    # The playout program of the test above (same party, --n and --devices; the budget is a
    # state value): a second rank program on top of it passed 9 GB in the fast tier.
    assert (
        main(["rank", "--party", SAMPLE, "--trainer", "roark", "--n", "50", "--devices", "1"]) == 0
    )
    out = capsys.readouterr().out
    assert "P(win, 0 faints) [95% CI]" in out and "protected:" not in out


@pytest.mark.slow
def test_cli_solve_deaths(tmp_path, monkeypatch, capsys):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    args = ["solve", "--party", SAMPLE, "--trainer", "roark", "--max-depth", "1"]
    assert main([*args, "--playouts", "2000", "--deaths", "1", "--devices", "1"]) == 0
    out = capsys.readouterr().out
    assert "P(win, <=1 deaths)" in out and "deaths under the leader:" in out
    (record,) = (tmp_path / "runs").glob("solve_*.json")
    run = json.loads(record.read_text())
    assert run["deaths"] == 1 and run["protected"] == []
    assert run["result"]["deaths_allowed"] == 1


def _two_starly(tmp_path) -> str:
    text = Path(PARTY).read_text()
    start = text.index('[[pokemon]]\nspecies = "Starly"')
    block = text[start : text.index("[[pokemon]]", start + 1)]
    path = tmp_path / "two_starly.toml"
    last = text.rindex("[[pokemon]]")  # six is the party limit: Budew makes room
    path.write_text(text[:last] + block)
    return str(path)


@pytest.mark.parametrize(
    ("flags", "party", "named"),
    [
        (["--deaths", "4"], PARTY, ("--deaths", "4")),
        (["--deaths", "-1"], PARTY, ("--deaths", "-1")),
        (["--protect", "Pikachu"], PARTY, ("--protect", "Pikachu")),
        (["--protect", "9"], PARTY, ("--protect", "9")),
        (["--protect", "6"], PARTY, ("--protect", "6")),
        (["--protect", "Budew"], FAINTED, ("--protect", "Budew")),
        (["--protect", "Starly"], None, ("--protect", "Starly")),
    ],
)
@pytest.mark.parametrize("command", ["rank", "solve"])
def test_cli_bad_budget_flags(capsys, tmp_path, command, flags, party, named):
    party = party or _two_starly(tmp_path)
    assert main([command, "--party", party, "--trainer", "roark", *flags, "--devices", "1"]) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert named[0] in err and named[1] in err


@pytest.mark.slow  # the whole-engine rank program alone is ~9.5 GB (E1 memory job, 17 Nov)
def test_cli_user_save_rank(capsys):
    save = os.environ.get("NUZLINE_USER_SAVE")
    if not save:
        pytest.skip("NUZLINE_USER_SAVE not set")
    args = ["rank", "--save", save, "--trainer", "roark", "--deaths", "1", "--protect", "Furret"]
    assert main([*args, "--n", "50", "--devices", "1"]) == 0
    assert "<=0" in capsys.readouterr().out


# ---- nuzline prep (plan 4c Task 10) ----


def _prep_save(tmp_path, tables, names):
    """A synthetic save: Monferno, Prinplup and Grotle (L14), a Starly (L10) in box 1 slot 0,
    a Heart Scale and TM10 in the bag."""
    from tests.saves.synthetic import SAMPLE, box_record, party_record
    from tests.saves.writer import build_image

    path = tmp_path / "prep.sav"
    path.write_bytes(
        build_image(
            party=[party_record(tables, names, *row) for row in SAMPLE],
            boxes={(0, 0): box_record(tables, names, "Starly", 10, ("Tackle", "Growl"))},
            bag={
                "items": [(names.item("Heart Scale"), 1)],
                "tms_hms": [(names.item("TM10"), 1)],
            },
        )
    )
    return str(path)


@pytest.fixture
def prep_runs(tmp_path, monkeypatch):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    return tmp_path / "runs"


SMALL = ["--max-depth", "1", "--finalists", "1", "--screen-n", "4", "--devices", "1"]


@pytest.fixture(scope="module")
def prep_run(tables, names, tmp_path_factory):
    """ONE small ``nuzline prep`` run (the slow part) shared by the tests below:
    (exit code, stdout, stderr, runs dir, save path), default checkpoint path."""
    import contextlib
    import io

    from nuzline import cli

    root = tmp_path_factory.mktemp("prep_run")
    save = _prep_save(root, tables, names)
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "RUNS", root / "runs")
        mp.setenv("XLA_FLAGS", "")
        mp.delenv("XLA_FLAGS")
        argv = ["prep", "--save", save, "--trainer", "roark", "--cap", "14", *SMALL]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        os.environ.pop("XLA_FLAGS", None)
    return code, out.getvalue(), err.getvalue(), root / "runs", save


@pytest.mark.slow
def test_cli_prep_small(prep_run):
    code, out, err, runs, save = prep_run
    assert code == 0
    out.encode("ascii")
    for section in ("Setup:", "Prep steps:", "Cost:", "Odds ladder (rung 0 planned):", "Plan:"):
        assert section in out
    assert "[prep] teams round 0:" in err
    assert err.index("[prep] checkpoint: ") < err.index("[prep] teams round 0:")
    (record,) = runs.glob("prep_*[!t].json")
    run = json.loads(record.read_text())
    import hashlib

    assert run["save_sha256"] == hashlib.sha256(Path(save).read_bytes()).hexdigest()
    assert run["deaths"] == 0 and run["cap"] == 14
    assert f"run record: {record}" in out
    (ckpt,) = runs.glob("prep_*.ckpt.json")
    assert f"checkpoint: {ckpt}" in out


def test_cli_prep_needs_a_cap(prep_runs, tmp_path, tables, names, capsys):
    save = _prep_save(tmp_path, tables, names)
    # Anthony (34) is a supported fight with no configured cap.
    assert main(["prep", "--save", save, "--trainer-id", "34", *SMALL]) == 1
    assert "config/level_caps.toml" in capsys.readouterr().err
    # A refused fight exits 2 before the cap is needed (final review M8).
    assert main(["prep", "--save", save, "--trainer-id", "858", *SMALL]) == 2
    assert "Support: NONE" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "shown"),
    [
        (["--deaths", "4"], ["4"]),
        (["--protect", "Starly", "--exclude", "Starly"], ["starly"]),
        (
            ["--exclude", "party:0", "--exclude", "party:1", "--exclude", "party:2"]
            + ["--exclude", "box:1:0"],
            ["no usable Pokemon"],
        ),
        (["--reserve", "Potionx"], ["Potionx"]),
        (["--prestatus", "sleep"], ["sleep"]),
        (["--weight", "tm=x"], ["tm=x"]),
        (["--weight", "bogus=1"], ["bogus"]),
        (["--protect", "1"], ["party:1", "PRINPLUP"]),
        (["--protect", "Monferno"], ["party:0", "box:1:0", "box:B:S"]),
        (["--max-setups", "5"], ["32", "--exclude", "--max-team"]),
        (["--max-team", "7"], ["--max-team 7"]),
    ],
)
def test_cli_prep_bad_flags(prep_runs, tmp_path, tables, names, capsys, flags, shown):
    from tests.saves.synthetic import SAMPLE

    save = _prep_save(tmp_path, tables, names)
    if "Monferno" in flags:  # a second Monferno in the box instead of the Starly
        from tests.saves.synthetic import box_record, party_record
        from tests.saves.writer import build_image

        Path(save).write_bytes(
            build_image(
                party=[party_record(tables, names, *row) for row in SAMPLE],
                boxes={(0, 0): box_record(tables, names, "Monferno", 14, SAMPLE[0][2])},
            )
        )
    args = ["prep", "--save", save, "--trainer", "roark", "--cap", "14", *SMALL, *flags]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and err.startswith("nuzline: error:")
    for text in shown:
        assert text.lower() in err.lower()


@pytest.mark.slow
def test_cli_prep_resume(prep_run, capsys):
    _, first, _, runs, save = prep_run
    (ckpt,) = runs.glob("prep_*.ckpt.json")
    before = ckpt.read_bytes()
    base = ["prep", "--save", save, "--trainer", "roark", "--cap", "14", *SMALL]
    assert main([*base, "--resume", str(ckpt)]) == 0
    second = capsys.readouterr().out

    def block(out):
        return out[out.index("Setup:") : out.index("Prep steps:")]

    assert block(first) == block(second)
    before = ckpt.read_bytes()
    assert main([*base, "--resume", str(ckpt), "--deaths", "1"]) == 1
    err = capsys.readouterr().err
    errors = [line for line in err.splitlines() if not line.startswith("[prep] checkpoint: ")]
    assert len(errors) == 1 and "settings" in err
    assert main([*base, "--resume", str(ckpt), "--checkpoint", str(ckpt)]) == 1
    assert "--resume" in capsys.readouterr().err
    assert ckpt.read_bytes() == before  # a refused resume leaves the file alone


@pytest.mark.fullset
def test_cli_prep_unsupported_opponent(prep_runs, tmp_path, tables, names, capsys):
    save = _prep_save(tmp_path, tables, names)
    args = ["prep", "--save", save, "--trainer-id", "858", "--cap", "60", *SMALL]
    assert main(args) == 2
    assert "Missing (" in capsys.readouterr().out


def test_coverage_save_uses_the_config_cap(tmp_path, tables, names, capsys):
    save = _prep_save(tmp_path, tables, names)
    assert main(["coverage", "--save", save, "--trainer", "roark"]) == 0
    assert "Pool at level cap 16" in capsys.readouterr().out


# ---- --game vanilla|kaizo, rivals and optional trainers (plan K2b Task 18) ----

_KAIZO = "platinum_kaizo_nightmare_final"


def kt(label, *, id=0, starter=None, optional=False, gauntlet=None, fmt="single", cls="Trainer"):
    """A Kaizo-style trainer known by ``label`` (its ROM name is the label's last word)."""
    return replace(
        Trainer(id, label.split()[-1], cls, 0, 0, 7, (), (), "none", fmt),
        label=label,
        player_starter=starter,
        optional=optional,
        gauntlet=gauntlet,
    )


def write_trainers(dir: Path, trainers) -> None:
    """``trainers`` as a ``trainers.json`` that ``load_trainers`` reads back."""
    rows = [
        {
            "id": t.id,
            "name": t.name,
            "class": t.trainer_class,
            "class_id": t.class_id,
            "class_gender": t.class_gender,
            "ai_flags": t.ai_flags,
            "items": list(t.items),
            "party": [],
            "start_condition": t.start_condition,
            "format": t.format,
            "opponent_2": t.opponent_2,
            "partner": [],
            "terrain": t.terrain,
            "terrains": list(t.terrains),
            "label": t.label,
            "player_starter": t.player_starter,
            "optional": t.optional,
            "gauntlet": None if t.gauntlet is None else list(t.gauntlet),
        }
        for t in trainers
    ]
    (dir / "trainers.json").write_text(json.dumps(rows), encoding="utf-8")


def run_solve_with_record(tmp_path) -> Path:
    """The solve-record test's command, made small; returns the run record it wrote."""
    from nuzline import cli

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "RUNS", tmp_path / "runs")
        args = ["solve", "--party", SAMPLE, "--trainer", "roark", "--max-depth", "0"]
        assert main([*args, "--playouts", "50", "--devices", "1"]) == 0
    (record,) = (tmp_path / "runs").glob("solve_*.json")
    return record


def test_default_game_is_vanilla(capsys):
    rc = main(["coverage", "--party", SAMPLE, "--trainer", "roark"])
    plain = capsys.readouterr().out
    assert main(["coverage", "--game", "vanilla", "--party", SAMPLE, "--trainer", "roark"]) == rc
    assert capsys.readouterr().out == plain


def test_kaizo_without_tables_exit_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(games.GAMES, _KAIZO, Game(_KAIZO, tmp_path, tmp_path))
    assert main(["coverage", "--game", "kaizo", "--party", SAMPLE, "--trainer", "roark"]) == 1
    assert "tools.build_tables --game kaizo" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["rank", "solve", "prep", "trainers"])
def test_kaizo_without_tables_every_command(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setitem(games.GAMES, _KAIZO, Game(_KAIZO, tmp_path, tmp_path))
    args = {
        "rank": ["--party", SAMPLE, "--trainer", "roark", "--devices", "1"],
        "solve": ["--party", SAMPLE, "--trainer", "roark", "--devices", "1"],
        "prep": ["--save", SAMPLE, "--trainer", "roark", "--devices", "1"],
        "trainers": [],
    }[command]
    assert main([command, "--game", "kaizo", *args]) == 1
    err = capsys.readouterr().err
    # trainers needs only the trainer list: a missing one names the Kaizo build too.
    assert "tools.build_tables --game kaizo" in err and len(err.strip().splitlines()) == 1


def test_kaizo_without_tables_loads_no_jax(tmp_path):  # Review Focus 1, in a subprocess
    code = (
        "import sys; from pathlib import Path; from nuzline.data import games; "
        "from nuzline.cli import main;"
        f"games.GAMES['{_KAIZO}'] = games.Game('{_KAIZO}', Path(r'{tmp_path}'), "
        f"Path(r'{tmp_path}'));"
        f"rc = main(['coverage', '--game', 'kaizo', '--party', r'{SAMPLE}', '--trainer', 'roark']);"
        "assert rc == 1 and 'jax' not in sys.modules"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.slow
def test_solve_record_has_game(tmp_path, capsys):
    record = json.loads(run_solve_with_record(tmp_path).read_text("utf-8"))
    assert record["game"] == "platinum_us" and "table_version" in record
    assert record["result"]["game"] == "platinum_us"


def test_rival_variants_need_save():
    ts = [
        kt("Pok\xe9mon Trainer Barry #3 [Empoleon]", id=10, starter=390),
        kt("Pok\xe9mon Trainer Barry #3 [Torterra]", id=11, starter=393),
        kt("Pok\xe9mon Trainer Barry #3 [Infernape]", id=12, starter=387),
    ]
    with pytest.raises(ValueError, match=r"10, 11, 12.*--save or --trainer-id"):
        resolve_trainer(ts, name="barry #3")
    assert resolve_trainer(ts, name="barry #3", player_starter=390).id == 10


def test_resolve_trainer_label_rules():
    # Exact name or label first; a label substring only when nothing matches exactly.
    ts = [kt("Leader Roark", id=1, cls="Leader"), kt("Hiker Roark", id=2), kt("Lass B", id=3)]
    assert resolve_trainer(ts, name="hiker roark").id == 2
    assert resolve_trainer(ts, name="ROARK").id == 1  # the ROM name, a Leader preferred
    assert resolve_trainer(ts, name="lass").id == 3
    with pytest.raises(ValueError, match="no trainer named"):
        resolve_trainer(ts, name="barry")
    # Different fights sharing a substring are not starter variants: the first is taken.
    assert resolve_trainer(ts, name="r roark").id == 1


def test_trainers_marks_optional(tmp_path, monkeypatch, capsys):  # synthetic Kaizo dir
    write_trainers(
        tmp_path,
        [
            kt("Lass B", optional=True),
            kt("Worker C", id=1, gauntlet=("Oreburgh Gym", 2)),
            kt("Twins D & E", id=2, fmt="double"),
        ],
    )
    monkeypatch.setitem(games.GAMES, _KAIZO, Game(_KAIZO, tmp_path, tmp_path))
    assert main(["trainers", "--game", "kaizo"]) == 0
    out = capsys.readouterr().out
    assert "Lass B [optional]" in out and "Worker C [gauntlet Oreburgh Gym #2]" in out
    assert "Twins D & E [double]" in out


def test_trainers_with_save_shows_only_its_rival(tmp_path, monkeypatch, tables, names, capsys):
    from nuzline.saves.layout import VAR_PLAYER_STARTER_INDEX
    from tests.saves.synthetic import party_record
    from tests.saves.writer import build_image

    vanilla = games.GAMES[games.VANILLA]
    inter = tmp_path / "inter"
    inter.mkdir()
    (inter / "items.json").write_bytes((vanilla.intermediate_dir / "items.json").read_bytes())
    write_trainers(
        inter,
        [
            kt("Pok\xe9mon Trainer Barry #1 [Turtwig]", id=1, starter=393),
            kt("Pok\xe9mon Trainer Barry #1 [Piplup]", id=2, starter=390),
            kt("Lass B", id=3),
        ],
    )
    monkeypatch.setitem(games.GAMES, games.VANILLA, Game(games.VANILLA, vanilla.tables_dir, inter))
    save = tmp_path / "s.sav"
    party = [party_record(tables, names, "Prinplup", 14, ("Pound", "Bubble"))]
    save.write_bytes(build_image(party=party, vars={VAR_PLAYER_STARTER_INDEX: 390}))
    assert main(["trainers"]) == 0
    assert "[Turtwig]" in capsys.readouterr().out
    assert main(["trainers", "--save", str(save)]) == 0
    out = capsys.readouterr().out
    assert "[Piplup]" in out and "[Turtwig]" not in out and "Lass B" in out


def _kaizo_party(tmp_path, names) -> str:
    """SAMPLE for Kaizo (plan K2b preflight #3): every move Kaizo renamed (``renamed.move``
    in its manifest) under its new name, and each Pokemon's ability its first Kaizo one."""
    import tomllib

    manifest = json.loads(
        (games.GAMES[_KAIZO].tables_dir / "manifest.json").read_text(encoding="utf-8")
    )
    renamed = manifest["renamed"]["move"]
    lines = []
    for mon in tomllib.loads(Path(SAMPLE).read_text(encoding="utf-8"))["pokemon"]:
        mon["moves"] = [renamed.get(m.lower(), m) for m in mon["moves"]]
        first = names.species_abilities(names.species(mon["species"]), 0)[0]
        mon["ability"] = names.ability_names[first]
        lines.append("[[pokemon]]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in mon.items()]
    path = tmp_path / "kaizo_party.toml"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return str(path)


def test_kaizo_roark_covered(tmp_path, capsys, kaizo_tables, kaizo_names):  # exit 0
    party = _kaizo_party(tmp_path, kaizo_names)
    assert main(["coverage", "--game", "kaizo", "--party", party, "--trainer", "roark"]) == 0
    out = capsys.readouterr().out
    # K3b1: sandstorm is modelled, so Roark's battle-start Sandstorm is no longer refused.
    # K3c Task 2: Kaizo has an AI policy and its one AI script change is modelled. K3c Task 6
    # registers the RISKY flag for the Kaizo policy. Wild Charge (RECOIL_STATUS, B10-1) and the
    # sample party's Leaf Guard (E1 Gardenia batch, G-14) are modelled: the fight is covered.
    assert "Support: EXACT" in out and "Missing: none" in out
    assert "Sandstorm" not in out
    assert "no transcribed AI policy" not in out and "code changes unclassified" not in out
    assert "AI_FLAG" not in out and "AI flag set" not in out
    assert "Leaf Guard" not in out
    # K3b2 Task 4: Gible's Thunder Fang (effect 275) is modelled now.
    assert "Thunder Fang (move effect UNSUPPORTED)" not in out
    # Final review I3: Kaizo Roark's move 205 is Accelerock; Rollout is no Kaizo move. K3c
    # Task 3 keys the Kaizo AI's script paths by battle effect, so Accelerock (HIT) is
    # transcribed now. K3c Task 5 transcribes the effects of all of Kaizo Roark's moves, so no
    # "script paths" line is left, and K3c Task 6 their Expert labels, so no "Expert_" line
    # is either. The callers' names are pinned by tests/ai/test_policy.py
    # test_refusals_name_moves_and_items_in_the_callers_game.
    assert "are not transcribed" not in out
    assert "Expert_" not in out and "ai scoring" not in out
    assert "Rollout" not in out


@pytest.mark.fullset
def test_kaizo_rival_barry_3_is_refused(tmp_path, capsys, kaizo_tables, kaizo_names):
    # The refusal test_kaizo_rival_refusal_prints_on_a_cp1252_console prints, in process.
    party = _kaizo_party(tmp_path, kaizo_names)
    assert main(["coverage", "--game", "kaizo", "--party", party, "--trainer-id", "472"]) == 2
    out = capsys.readouterr().out
    assert "Support: NONE" in out and "Missing (" in out
    assert "AI_FLAG_CHECK_HP" not in out


def test_cli_still_imports_without_jax():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = "import sys, nuzline.cli; assert 'jax' not in sys.modules"
    done = subprocess.run([sys.executable, "-c", code], check=False, env=env, cwd=ROOT)
    assert done.returncode == 0


def _patched_roark(monkeypatch, **changes) -> list[int]:
    """Every load of the trainers gives Roark (246) ``changes``; returns the ids
    ``build_trainer_party`` is called with."""
    from nuzline.data import trainer as trainer_module

    load, build = trainer_module.load_trainers, trainer_module.build_trainer_party
    built: list[int] = []

    def patched(dir):
        return [replace(t, **changes) if t.id == 246 else t for t in load(dir)]

    def spy(trainer, inter):
        built.append(trainer.id)
        return build(trainer, inter)

    monkeypatch.setattr(trainer_module, "load_trainers", patched)
    monkeypatch.setattr(trainer_module, "build_trainer_party", spy)
    return built


UNKNOWN_LINE = "Battle start: condition unknown (user call) (start condition UNSUPPORTED)"
NO_SET_LINE = "Roark slot 0: nature/gender/ability unknown (no PKCalc set)"


@pytest.mark.parametrize("command", ["coverage", "rank", "solve"])
def test_unknown_start_condition_is_refused(command, monkeypatch, tmp_path, capsys):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    _patched_roark(monkeypatch, start_condition="unknown")
    extra = [] if command == "coverage" else ["--devices", "1"]
    assert main([command, "--party", SAMPLE, "--trainer", "roark", *extra]) == 2
    captured = capsys.readouterr()
    assert UNKNOWN_LINE in captured.out and "Roark (Leader, id 246)" in captured.out
    assert captured.err == "" and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("command", ["coverage", "rank", "solve"])
def test_missing_set_data_is_refused_without_building(command, monkeypatch, tmp_path, capsys):
    from nuzline import cli

    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    built = _patched_roark(monkeypatch, pid_routine=None)
    extra = [] if command == "coverage" else ["--devices", "1"]
    assert main([command, "--party", SAMPLE, "--trainer", "roark", *extra]) == 2
    captured = capsys.readouterr()
    assert NO_SET_LINE in captured.out and captured.err == "" and built == []


@pytest.mark.parametrize(
    ("changes", "line"),
    [({"start_condition": "unknown"}, UNKNOWN_LINE), ({"pid_routine": None}, NO_SET_LINE)],
)
def test_prep_refuses_unknown_start_and_missing_sets(
    changes, line, prep_runs, monkeypatch, tmp_path, tables, names, capsys
):
    built = _patched_roark(monkeypatch, **changes)
    save = _prep_save(tmp_path, tables, names)
    assert main(["prep", "--save", save, "--trainer", "roark", "--cap", "14", *SMALL]) == 2
    out = capsys.readouterr().out
    assert line in out and "Roark (Leader, id 246)" in out
    assert built == ([] if "pid_routine" in changes else [246])
    assert not prep_runs.exists()


def test_impossible_save_contents_warn(tmp_path, tables, names, capsys):
    # Spec section 8: the game is never inferred from a save; impossible contents warn
    # loudly on stderr and the command goes on.
    from tests.saves.synthetic import party_record
    from tests.saves.writer import build_image

    path = tmp_path / "odd.sav"
    path.write_bytes(
        build_image(
            party=[
                party_record(tables, names, "Monferno", 14, ("Scratch", "Ember")),
                party_record(tables, names, "Starly", 10, ("Tackle", "Hydro Pump")),
            ]
        )
    )
    for argv in (
        ["save", "show", str(path)],
        ["save", "export", str(path), "-o", str(tmp_path / "p.toml")],
        ["coverage", "--save", str(path), "--trainer", "roark"],
        ["trainers", "--save", str(path)],
    ):
        assert main(argv) == 0, argv
        err = capsys.readouterr().err
        warnings = [line for line in err.splitlines() if "Hydro Pump" in line]
        assert len(warnings) == 1, (argv, err)
        assert warnings[0].startswith("warning: ") and "STARLY" in warnings[0].upper()


# ---- Task 18 fix round 1: ASCII output; a rival name needs the starter ----

BARRYS = [
    kt("Pok\xe9mon Trainer Barry #2 [Turtwig]", id=247, starter=393),
    kt("Pok\xe9mon Trainer Barry #2 [Piplup]", id=249, starter=390),
    kt("Pok\xe9mon Trainer Barry #3 [Torterra]", id=470, starter=393),
    kt("Pok\xe9mon Trainer Barry #3 [Empoleon]", id=472, starter=390),
]


def test_bare_rival_name_needs_a_save():
    # One fight's variants without a save: never guessed (fix round 1, I2).
    with pytest.raises(ValueError, match=r"470, 472.*--save or --trainer-id"):
        resolve_trainer(BARRYS, name="barry #3")
    assert resolve_trainer(BARRYS, name="barry #3", player_starter=390).id == 472
    assert resolve_trainer(BARRYS, trainer_id=470).id == 470


@pytest.mark.parametrize("starter", [None, 390])
def test_rival_name_of_several_fights_must_name_the_fight(starter):
    # Fix round 2: "barry" is Barry #2 and Barry #3, with or without a save.
    fights = "Pok\xe9mon Trainer Barry #2; Pok\xe9mon Trainer Barry #3"
    with pytest.raises(ValueError, match=rf"2 rival fights: {fights}.*--trainer-id") as e:
        resolve_trainer(BARRYS, name="barry", player_starter=starter)
    assert "--trainer 'Pok\xe9mon Trainer Barry #3'" in str(e.value)
    # A non-rival name still takes the first match (ruling #39).
    grunts = [kt("Galactic Grunt Ann", id=1), kt("Galactic Grunt Bob", id=2)]
    assert resolve_trainer(grunts, name="grunt", player_starter=starter).id == 1


def test_rival_fights_with_a_save_on_the_command_line(tmp_path, monkeypatch, tables, names, capsys):
    from nuzline.saves.layout import VAR_PLAYER_STARTER_INDEX
    from tests.saves.synthetic import party_record
    from tests.saves.writer import build_image

    vanilla = games.GAMES[games.VANILLA]
    inter = tmp_path / "inter"
    inter.mkdir()
    (inter / "items.json").write_bytes((vanilla.intermediate_dir / "items.json").read_bytes())
    write_trainers(inter, BARRYS)
    monkeypatch.setitem(games.GAMES, games.VANILLA, Game(games.VANILLA, vanilla.tables_dir, inter))
    save = tmp_path / "s.sav"
    party = [party_record(tables, names, "Prinplup", 14, ("Pound", "Bubble"))]
    save.write_bytes(build_image(party=party, vars={VAR_PLAYER_STARTER_INDEX: 390}))
    args = ["coverage", "--save", str(save), "--trainer"]
    assert main([*args, "barry"]) == 1
    err = capsys.readouterr().err
    assert "2 rival fights: Pok\\xe9mon Trainer Barry #2; Pok\\xe9mon Trainer Barry #3" in err
    assert err.isascii()
    # "barry #3" with the save: the save's starter picks Empoleon's team (472). The spy
    # stops the command there (the synthetic trainers have no party to fight).
    from nuzline import cli

    seen = []
    resolve = cli.resolve_trainer

    def spy(*a, **k):
        seen.append(resolve(*a, **k))
        raise ValueError("stop after resolution")

    monkeypatch.setattr(cli, "resolve_trainer", spy)
    assert main([*args, "barry #3"]) == 1
    assert "stop after resolution" in capsys.readouterr().err
    assert [t.id for t in seen] == [472]


def test_rival_without_the_saves_starter_names_it():
    with pytest.raises(ValueError, match=r"starter \(species 387\).*470, 472"):
        resolve_trainer(BARRYS, name="barry #3", player_starter=387)


def test_trainers_output_is_ascii(tmp_path, monkeypatch, capsys):
    write_trainers(
        tmp_path,
        [kt("Pok\xe9mon Trainer Barry #1 [Piplup]", starter=390, cls="Swimmer♂")],
    )
    monkeypatch.setitem(games.GAMES, _KAIZO, Game(_KAIZO, tmp_path, tmp_path))
    assert main(["trainers", "--game", "kaizo"]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "Pok\\xe9mon Trainer Barry #1 [Piplup]" in out
    assert "(Swimmer\\u2642)" in out


def _cp1252_cli(*argv: str) -> subprocess.CompletedProcess:
    """``python -m nuzline.cli ARGV`` with stdout and stderr encoded as cp1252, the default
    Windows console; output decoded as cp1252."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "cp1252"}
    return subprocess.run(
        [sys.executable, "-m", "nuzline.cli", *argv],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        encoding="cp1252",
    )


def test_trainers_prints_on_a_cp1252_console():
    done = _cp1252_cli("trainers")
    assert done.returncode == 0, done.stderr
    assert done.stdout.isascii() and "246  Roark  (Leader)" in done.stdout


@pytest.mark.slow
def test_kaizo_rival_refusal_prints_on_a_cp1252_console(tmp_path, kaizo_tables, kaizo_names):
    party = _kaizo_party(tmp_path, kaizo_names)
    done = _cp1252_cli("coverage", "--game", "kaizo", "--party", party, "--trainer-id", "472")
    assert done.returncode == 2, done.stderr
    out = done.stdout
    assert out.isascii() and "Pok\\xe9mon Trainer Barry #3 [Empoleon] (" in out
    assert "Support: NONE" in out and "Missing (" in out
    # Barry's CHECK_HP flag (ai_flags 279) is transcribed (layout.KAIZO_FLAG_BITS).
    assert "AI_FLAG_CHECK_HP" not in out


@pytest.mark.parametrize("which", ["kaizo_save_pre_roark_path", "kaizo_save_path"])
def test_kaizo_save_export_checks_kaizo_tables(which, tmp_path, request, kaizo_tables, kaizo_names):
    # Final review I2: the party is checked against the Kaizo tables (Sandshrew's Kaizo
    # Rough Skin is no vanilla Sandshrew ability), and round-trips through load_party.
    from nuzline.data.party import load_party
    from nuzline.saves import load_save

    path = request.getfixturevalue(which)  # read in place, never copied
    out = tmp_path / "party.toml"
    assert main(["save", "export", "--game", "kaizo", str(path), "-o", str(out)]) == 0
    save = load_save(path, tables=kaizo_tables, names=kaizo_names, game=_KAIZO)
    want = [m.to_spec(kaizo_tables, save.trainer) for m in save.party if not m.is_egg]
    assert load_party(out, kaizo_names) == want


def test_renamed_slot_lookup_is_one_error_line(capsys, kaizo_tables, kaizo_names):  # M7
    # A vanilla party file under --game kaizo: Water Sport's slot is Scald in Kaizo.
    party = str(ROOT / "tests" / "fixtures" / "golden_party_roark_01.toml")
    assert main(["coverage", "--game", "kaizo", "--party", party, "--trainer", "roark"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("nuzline: error: unknown move")
    assert "in this game that slot is" in err[0] and "Traceback" not in err[0]


def test_kaizo_prep_refusal_exits_2_before_the_cap(
    tmp_path, monkeypatch, capsys, kaizo_tables, kaizo_names, kaizo_save_pre_roark_path
):  # M8: Gambler Evan (104) is placed on a map but has no split, so default_cap would raise "pass --cap" (exit 1)
    # A refused fight exits 2 before the cap is asked for. The refusal is synthetic (singles
    # marked unsupported), so the test does not depend on what coverage models for Evan.
    from nuzline import cli
    from nuzline.engine import coverage
    from nuzline.engine.coverage import Support

    monkeypatch.setitem(coverage.IMPLEMENTED_FORMATS, "single", Support.NONE)
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    path = str(kaizo_save_pre_roark_path)
    args = ["prep", "--game", "kaizo", "--save", path, "--trainer-id", "104", "--devices", "1"]
    assert main(args) == 2
    out = capsys.readouterr().out
    assert "Support: NONE" in out
    assert not (tmp_path / "runs").exists()


def test_kaizo_pool_cap_is_never_defaulted(
    monkeypatch, capsys, kaizo_tables, kaizo_names, kaizo_save_pre_roark_path
):
    # M9: Gambler Evan (104) is placed on a map (terrains 1, 2, 7) but has no split, so there is
    # no cap; the pool is not shown at the party max. (Martha 51 has a split since E3 batch 5.)
    # The refusal (exit 2) is synthetic, singles marked unsupported, so the test does not
    # depend on what coverage models for Evan.
    from nuzline.engine import coverage
    from nuzline.engine.coverage import Support

    monkeypatch.setitem(coverage.IMPLEMENTED_FORMATS, "single", Support.NONE)
    path = str(kaizo_save_pre_roark_path)
    assert main(["coverage", "--game", "kaizo", "--save", path, "--trainer-id", "104"]) == 2
    captured = capsys.readouterr()
    assert "Pool at level cap" not in captured.out
    assert "nuzline: error: pool left out: no level cap for" in captured.err
    assert "--cap" in captured.err
    assert (
        main(["coverage", "--game", "kaizo", "--save", path, "--trainer-id", "104", "--cap", "12"])
        == 2
    )
    assert "Pool at level cap 12" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.parametrize("save", ["kaizo_save_pre_roark_path", "kaizo_save_probe_path"])
def test_kaizo_roark_coverage_on_the_saves_is_exact(request, save, capsys, kaizo_tables):
    # K3c final review I-1: the milestone goal (plan K3c lines 5-7, Task 8 item 1) as a
    # regression test. The save is read in place, never copied.
    path = str(request.getfixturevalue(save))
    args = ["coverage", "--game", "kaizo", "--save", path, "--trainer", "roark", "--cap", "16"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Pool at level cap 16 (12 Pokemon):" in out
    assert out.count("Support: EXACT") == 2, out
    assert out.count("Missing: none") == 2, out
    assert out.count("Sampling only: none") == 2, out


def test_pkmn_glyphs_print_as_pkmn():  # final review M13
    from nuzline.cli import _listing_line, _trainer_line

    barry = kt("Pok\xe9mon Trainer Barry #2 [Turtwig]", id=247, cls="\u20a7\u20a6 Trainer")
    assert _trainer_line(barry) == "Pok\\xe9mon Trainer Barry #2 [Turtwig] (PKMN Trainer, id 247)"
    assert _listing_line(barry).endswith("(PKMN Trainer)") and _listing_line(barry).isascii()
