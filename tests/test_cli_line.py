"""``nuzline line odds`` and ``nuzline line find`` (L1 spec §6, plan Task 5)."""

import os
from pathlib import Path

import pytest

from nuzline.cli import main

ROOT = Path(__file__).resolve().parents[1]
PARTY = str(ROOT / "tests" / "fixtures" / "sample_party.toml")
BASE = ["--party", PARTY, "--trainer", "roark", "--devices", "1"]

# Vanilla Roark leads Geodude, then Onix and Cranidos; the sample party is Monferno,
# Prinplup, Grotle, Starly, Shinx, Budew.
RULES = """
[[rule]]
against = "Geodude"
use = "Grotle"
move = "Razor Leaf"

[[rule]]
against = "Onix"
use = "Prinplup"
move = "Bubble"
via = "Grotle"
"""


@pytest.fixture(autouse=True)
def _data(tables):
    """Every command needs the built tables (the ``tables`` fixture skips without them)
    and the extracted trainers."""
    if not (ROOT / "data" / "intermediate" / "trainers.json").exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")


@pytest.fixture(autouse=True)
def _restore_xla_flags(monkeypatch):
    """``main(["line", ...])`` appends to ``XLA_FLAGS``; undo that after each test (as
    tests/test_cli.py)."""
    monkeypatch.setenv("XLA_FLAGS", "")
    monkeypatch.delenv("XLA_FLAGS")


def _file(tmp_path, text: str = RULES, name: str = "line.toml") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


LADDER = ("win, 0 deaths", "win, <=1 deaths", "win, <=2 deaths", "win, <=3 deaths", "win, any")


@pytest.mark.slow
def test_line_odds_prints_the_ladder(tmp_path, capsys):
    args = ["line", "odds", *BASE, "--line", _file(tmp_path), "--n", "32"]
    assert main(args) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert out.isascii()
    rows = [line.strip() for line in out.splitlines()]
    for label in LADDER:
        assert sum(r.startswith(label) for r in rows) == 1, label
    assert "Roark (Leader, id 246)" in out
    low = out.lower()  # vanilla species names are upper case
    assert "vs geodude: grotle, razor leaf" in low
    assert "vs onix: prinplup, bubble, via grotle" in low
    assert "vs cranidos: (no rule: default player)" in low
    assert "32 playouts, seed 0, budget <=0 deaths" in out
    assert "mean turns" in out and "default player decided" in out


def _no_playouts(monkeypatch):
    from nuzline.search import line_search, player_eval

    def fail(*a, **k):
        raise AssertionError("nothing may be played")

    monkeypatch.setattr(player_eval, "evaluate_lines", fail)
    monkeypatch.setattr(line_search, "find_line", fail)


def test_line_odds_bad_rule_exit_1(tmp_path, capsys, monkeypatch):
    _no_playouts(monkeypatch)
    bad = RULES.replace('against = "Geodude"', 'against = "Gible"')
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path, bad)]) == 1
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith("nuzline: error:") and "Gible" in err[0]


def test_line_odds_team_table_needs_save(tmp_path, capsys, monkeypatch):
    _no_playouts(monkeypatch)
    text = '[[team]]\npokemon = "Monferno"\n' + RULES
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path, text)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("nuzline: error:") and "[[team]]" in err


def test_line_odds_save_needs_a_team_table(tmp_path, tables, names, capsys, monkeypatch):
    from tests.saves.synthetic import write_save

    _no_playouts(monkeypatch)
    save = write_save(tmp_path, tables, names)
    args = ["line", "odds", "--save", str(save), "--trainer", "roark", "--devices", "1"]
    assert main([*args, "--line", _file(tmp_path)]) == 1
    assert "[[team]]" in capsys.readouterr().err


def _found(tables, names):
    """A fixed ``LineSearchResult``: Grotle's Razor Leaf against Geodude."""
    import numpy as np

    from nuzline.search import deaths
    from nuzline.search.line_search import LineSearchResult
    from nuzline.search.line_text import Rule, rules_to_line
    from nuzline.search.player_eval import PlayerOdds

    counts = np.zeros(len(deaths.BUCKETS))
    counts[0], counts[deaths.BUCKETS.index("whiteout")] = 30, 70
    odds = PlayerOdds(
        n=100,
        wins=(30, 30, 30, 30),
        wins_any=30,
        deaths_total=210,
        distribution=deaths.DeathDistribution.from_counts(counts),
        mean_turns=12.5,
        fallback_share=0.25,
    )
    line = rules_to_line([Rule(against=0, use=2, move=3)])
    return LineSearchResult(line=line, odds=odds, sweeps=2, evaluations=999, stopped="converged")


def test_line_find_writes_out(tmp_path, tables, names, capsys, monkeypatch):
    from nuzline.data.party import load_party
    from nuzline.search import line_search
    from nuzline.search.line_text import Rule, read_line_file, resolve_rules

    seen = []

    def fake(tables_, names_, state, key, settings, start=None, progress=None, **k):
        seen.append((settings, start))
        progress("sweep 1: stub")
        return _found(tables, names)

    monkeypatch.setattr(line_search, "find_line", fake)
    out_path = tmp_path / "found.toml"
    args = ["line", "find", *BASE, "--line", _file(tmp_path), "--out", str(out_path)]
    assert main([*args, "--restarts", "0", "--final-n", "100", "--sweeps", "2"]) == 0
    captured = capsys.readouterr()
    assert "[line] sweep 1: stub" in captured.err
    out = captured.out
    assert out.isascii()
    assert "found by search (a heuristic: a good line, not a proven best)" in out
    assert "100 playouts, seed 0" in out and "win, any" in out
    assert '[[rule]]\nagainst = "geodude"' in out.lower()  # vanilla names are upper case
    ((settings, start),) = seen
    assert (settings.restarts, settings.final_n, settings.max_sweeps) == (0, 100, 2)
    assert start is not None and int(start.use[0]) == 2 and int(start.use[1]) == 1
    file = read_line_file(out_path)
    assert file.team == ()
    from nuzline.cli import _load_trainers, resolve_trainer
    from nuzline.data.games import get_game
    from nuzline.data.trainer import IntermediateData, build_trainer_party

    game = get_game("vanilla")
    roark = resolve_trainer(_load_trainers(game), name="roark")
    opp = build_trainer_party(roark, IntermediateData(game.intermediate_dir))
    player = load_party(PARTY, names)
    assert resolve_rules(file, names, player, opp) == [Rule(against=0, use=2, move=3)]


def test_line_find_empty_file_starts_empty(tmp_path, tables, names, monkeypatch, capsys):
    from nuzline.search import line_search

    seen = []

    def fake(*a, start=None, **k):
        seen.append(start)
        return _found(tables, names)

    monkeypatch.setattr(line_search, "find_line", fake)
    assert main(["line", "find", *BASE, "--line", _file(tmp_path, "")]) == 0
    assert seen == [None]


def test_line_find_refused_exit_2(tmp_path, capsys, monkeypatch):
    from nuzline.ai.policy import UnsupportedAi
    from nuzline.search import line_search

    def refuse(*a, **k):
        raise UnsupportedAi("3 of 128 line playouts refused")

    monkeypatch.setattr(line_search, "find_line", refuse)
    out_path = tmp_path / "found.toml"
    args = ["line", "find", *BASE, "--line", _file(tmp_path), "--out", str(out_path)]
    assert main(args) == 2
    assert "refused: 3 of 128 line playouts refused" in capsys.readouterr().out
    assert not out_path.exists()


def test_line_odds_refused_fight_exit_2(tmp_path, capsys, monkeypatch):
    _no_playouts(monkeypatch)
    unsupported = str(ROOT / "tests" / "fixtures" / "sample_party_unsupported.toml")
    args = ["line", "odds", "--party", unsupported, "--trainer", "roark", "--devices", "1"]
    assert main([*args, "--line", _file(tmp_path, "")]) == 2
    out = capsys.readouterr().out
    assert "Roark (Leader, id 246)" in out and "Metronome" in out


@pytest.mark.parametrize(
    ("flags", "shown"),
    [
        (["--deaths", "4"], "--deaths 4"),
        (["--protect", "Piplup"], "--protect Piplup"),
        (["--n", "0"], "--n must be at least 1"),
    ],
)
def test_line_odds_bad_flags_exit_1(tmp_path, capsys, monkeypatch, flags, shown):
    _no_playouts(monkeypatch)
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path), *flags]) == 1
    assert shown in capsys.readouterr().err


def test_line_commands_set_devices(tmp_path, monkeypatch):
    monkeypatch.setattr("nuzline.cli._line_odds", lambda args: 0)
    monkeypatch.setattr("nuzline.cli._line_find", lambda args: 0)
    line = _file(tmp_path)
    args = ["--party", PARTY, "--trainer", "roark", "--line", line, "--devices", "3"]
    assert main(["line", "odds", *args]) == 0
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=3"
    monkeypatch.delenv("XLA_FLAGS")
    assert main(["line", "find", *args]) == 0
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=3"
