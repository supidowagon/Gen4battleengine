"""``nuzline line odds`` and ``nuzline line find`` (L1 spec Â§6, plan Task 5)."""

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


def _found(tables, names, wins=(10, 20, 25, 30), wins_any=40):
    """A fixed ``LineSearchResult`` over 100 playouts: Grotle's Razor Leaf against
    Geodude, ``wins[k]`` wins with at most k deaths and ``wins_any`` wins."""
    import numpy as np

    from nuzline.search import deaths
    from nuzline.search.line_search import LineSearchResult
    from nuzline.search.line_text import Rule, rules_to_line
    from nuzline.search.player_eval import PlayerOdds

    counts = np.zeros(len(deaths.BUCKETS))
    counts[:4] = [wins[0], *np.diff(wins)]
    counts[4] = wins_any - wins[3]
    counts[deaths.BUCKETS.index("whiteout")] = 100 - wins_any
    odds = PlayerOdds(
        n=100,
        wins=wins,
        wins_any=wins_any,
        deaths_total=210,
        distribution=deaths.DeathDistribution.from_counts(counts),
        mean_turns=12.5,
        fallback_share=0.25,
    )
    line = rules_to_line([Rule(against=0, use=2, move=3)])
    return LineSearchResult(line=line, odds=odds, sweeps=2, evaluations=999, stopped="converged")


def _ladder(out: str) -> dict[str, str]:
    """Each ladder row's label and percentage; a row is ``  <label padded> <pct> % [..]``
    with at least one space between label and percentage."""
    rows = {}
    for line in out.splitlines():
        for label in LADDER:
            if line.startswith(f"  {label} "):
                rows[label] = line[len(label) + 2 :].split()[0]
    return rows


def test_line_find_full_ladder_keeps_the_columns(tmp_path, tables, names, capsys, monkeypatch):
    from nuzline.search import line_search

    full = _found(tables, names, wins=(100, 100, 100, 100), wins_any=100)
    monkeypatch.setattr(line_search, "find_line", lambda *a, **k: full)
    assert main(["line", "find", *BASE, "--line", _file(tmp_path)]) == 0
    assert _ladder(capsys.readouterr().out) == dict.fromkeys(LADDER, "100.0")


def test_line_find_out_needs_its_directory(tmp_path, capsys, monkeypatch):
    _no_playouts(monkeypatch)
    out_path = tmp_path / "missing" / "found.toml"
    args = ["line", "find", *BASE, "--line", _file(tmp_path), "--out", str(out_path)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "--out" in captured.err and "no directory" in captured.err
    assert captured.out == ""


def test_line_find_out_must_not_be_the_line_file(tmp_path, capsys, monkeypatch):
    # The same file through another spelling of its path: refused before any work.
    _no_playouts(monkeypatch)
    line = _file(tmp_path)
    (tmp_path / "sub").mkdir()
    other = str(tmp_path / "sub" / ".." / "line.toml")
    assert main(["line", "find", *BASE, "--line", line, "--out", other]) == 1
    captured = capsys.readouterr()
    assert "--out" in captured.err and "is the --line file" in captured.err
    assert captured.out == ""
    assert (tmp_path / "line.toml").read_text(encoding="utf-8") == RULES


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
    assert (
        main([*args, "--restarts", "0", "--final-n", "100", "--sweeps", "2", "--pick-n", "500"])
        == 0
    )
    captured = capsys.readouterr()
    assert "[line] sweep 1: stub" in captured.err
    out = captured.out
    assert out.isascii()
    assert "found by search (a heuristic: a good line, not a proven best)" in out
    assert "100 playouts, seed 0" in out
    assert _ladder(out) == {
        "win, 0 deaths": "10.0",
        "win, <=1 deaths": "20.0",
        "win, <=2 deaths": "25.0",
        "win, <=3 deaths": "30.0",
        "win, any": "40.0",
    }
    assert '[[rule]]\nagainst = "geodude"' in out.lower()  # vanilla names are upper case
    assert any(r.startswith("deaths: 0: ") for r in out.splitlines())
    assert "deaths under the leader" not in out
    ((settings, start),) = seen
    assert (settings.restarts, settings.final_n, settings.max_sweeps) == (0, 100, 2)
    assert settings.pick_n == 500
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


LEFT_OUT = "playouts stopped at a mechanic the engine does not model and are left out: "
REASON = "replacement: the stale aiSwitchedPartySlot (stub)"


def _refusal_stub(monkeypatch, answer=REASON):
    """``player.refusal_reason`` replaced: records its player, line, key and turn cap,
    returns ``answer``."""
    from nuzline.search import player

    asked = []

    def reason(tables, state, player_, params, key, max_turns=100, layout=None):
        asked.append((player_.name, params, key, max_turns))
        return answer

    monkeypatch.setattr(player, "refusal_reason", reason)
    return asked


def _refused_odds(tables, names, monkeypatch, answer=REASON):
    """``line odds`` stubbed: ``evaluate_lines`` returns ``_found``'s odds with 3 refused
    playouts and records ``(refused, max_turns)``; ``refusal_reason`` returns ``answer``.
    ``(modes, asked, refused_key)``."""
    import dataclasses

    import numpy as np

    from nuzline.search import player_eval

    refused_key = np.array([5, 6], np.uint32)
    odds = dataclasses.replace(_found(tables, names).odds, refused=3, refused_key=refused_key)
    modes = []

    def evaluate(tables_, state, lines, n, key, max_turns=100, refused="raise"):
        modes.append((refused, max_turns))
        return [odds]

    monkeypatch.setattr(player_eval, "evaluate_lines", evaluate)
    return modes, _refusal_stub(monkeypatch, answer), refused_key


def test_line_odds_names_refused_playouts(tmp_path, tables, names, capsys, monkeypatch):
    import numpy as np

    modes, asked, refused_key = _refused_odds(tables, names, monkeypatch)
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path), "--n", "103"]) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    rows = out.splitlines()
    left = rows.index(f"3 of 103 {LEFT_OUT}{REASON}")
    assert left > max(i for i, r in enumerate(rows) if r.startswith("  win, any"))
    ((mode, max_turns),) = modes
    assert mode == "exclude"
    ((player_name, params, key, replay_turns),) = asked
    assert player_name == "line" and np.array_equal(key, refused_key)
    assert int(params.use[0]) == 2  # the file's line: Grotle against Geodude
    assert replay_turns == max_turns  # the replay plays under the same turn cap


def test_line_odds_unnamed_refusal(tmp_path, tables, names, capsys, monkeypatch):
    # A replay that is not refused (it should not happen) never prints "None".
    _refused_odds(tables, names, monkeypatch, answer=None)
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"3 of 103 {LEFT_OUT}unnamed (the replay was not refused)" in out.splitlines()
    assert "None" not in out


def test_line_odds_every_playout_refused_exit_2(tmp_path, capsys, monkeypatch):
    # ``line odds`` evaluates with "exclude", under which a line refused in every playout
    # raises (``"skip"``, the search's mode, would not).
    from nuzline.ai.policy import UnsupportedAi
    from nuzline.search import player_eval

    def evaluate(tables_, state, lines, n, key, max_turns=100, refused="raise"):
        assert refused == "exclude"
        raise UnsupportedAi(f"every playout of entry 0 ({n}) refused")

    monkeypatch.setattr(player_eval, "evaluate_lines", evaluate)
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path), "--n", "8"]) == 2
    assert "refused: every playout of entry 0 (8) refused" in capsys.readouterr().out


def test_line_odds_without_refused_playouts_says_nothing(
    tmp_path, tables, names, capsys, monkeypatch
):
    from nuzline.search import player_eval

    monkeypatch.setattr(player_eval, "evaluate_lines", lambda *a, **k: [_found(tables, names).odds])
    asked = _refusal_stub(monkeypatch)
    assert main(["line", "odds", *BASE, "--line", _file(tmp_path)]) == 0
    assert LEFT_OUT not in capsys.readouterr().out
    assert asked == []


def test_line_find_names_refused_playouts_and_skipped_rules(
    tmp_path, tables, names, capsys, monkeypatch
):
    import dataclasses

    import numpy as np

    from nuzline.search import line_search

    found = _found(tables, names)
    odds = dataclasses.replace(found.odds, refused=2, refused_key=np.array([1, 2], np.uint32))
    result = dataclasses.replace(found, odds=odds, refused_candidates=4)
    monkeypatch.setattr(line_search, "find_line", lambda *a, **k: result)
    asked = _refusal_stub(monkeypatch)
    assert main(["line", "find", *BASE, "--line", _file(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"2 of 102 {LEFT_OUT}{REASON}" in out.splitlines()
    skipped = "skipped 4 candidate lines that met unmodelled mechanics in more than 1 % of playouts"
    assert skipped in out.splitlines()
    ((_, params, _, max_turns),) = asked
    assert np.array_equal(params.use, result.line.use)  # the found line is replayed
    assert max_turns == line_search.LineSearchSettings().max_turns


def test_line_odds_refused_fight_exit_2(tmp_path, capsys, monkeypatch):
    _no_playouts(monkeypatch)
    unsupported = str(ROOT / "tests" / "fixtures" / "sample_party_unsupported.toml")
    args = ["line", "odds", "--party", unsupported, "--trainer", "roark", "--devices", "1"]
    assert main([*args, "--line", _file(tmp_path, "")]) == 2
    out = capsys.readouterr().out
    assert "Roark (Leader, id 246)" in out and "Metronome" in out


def test_line_setup_unbuildable_fight_raises_fight_refused(tmp_path, monkeypatch):
    """``line_setup``'s callers (benchmarks) catch the public ``FightRefused``."""
    from nuzline import cli

    def refuse(tables, names, trainer, inter):
        raise cli._Refused(trainer, None)

    monkeypatch.setattr(cli, "_check_buildable", refuse)
    with pytest.raises(cli.FightRefused) as e:
        cli.line_setup("vanilla", 246, Path(_file(tmp_path)), party_path=Path(PARTY))
    assert e.value.trainer.id == 246


@pytest.mark.parametrize(
    ("flags", "shown"),
    [
        (["--deaths", "4"], "--deaths 4"),
        (["--protect", "Piplup"], "--protect Piplup: no such Pokemon in the team"),
        (["--protect", "9"], "--protect 9: no team index 9 (the team has 6)"),
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
