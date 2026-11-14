"""``benchmarks.line_roark`` imports and finds the ``cli`` names it calls (fast: no jax
compile, no save). The benchmark itself is run by hand (benchmarks/README.md)."""

import pytest

from nuzline import cli

USED = ("line_setup", "FightRefused", "trainer_line", "team_lines", "print_report")
USED += ("line_report_lines", "left_out_lines", "configure_cpu_devices", "MAX_DEATHS")
ALIASES = {
    "_trainer_line": "trainer_line",
    "_print_report": "print_report",
    "_team_lines": "team_lines",
    "_line_block": "line_report_lines",
    "_left_out_lines": "left_out_lines",
}


def test_benchmark_imports_and_cli_names_exist():
    from benchmarks import line_roark

    assert line_roark.USER_LINE.exists()
    for name in USED:
        assert hasattr(cli, name), name


def test_private_cli_names_still_work():
    for old, new in ALIASES.items():
        assert getattr(cli, old) is getattr(cli, new)


@pytest.mark.parametrize(
    "argv", [["--n", "0"], ["--deaths", "4"], ["--budget", "-1"], ["--sweeps", "0"]]
)
def test_bad_flags_exit_before_any_work(argv):
    from benchmarks import line_roark

    with pytest.raises(SystemExit) as e:
        line_roark.main(argv)
    assert e.value.code == 2


def test_defaults_run_without_a_time_limit(monkeypatch):
    # The recorded run stops on sweeps, not on the clock: --budget defaults to none.
    from benchmarks import line_roark

    seen = []
    monkeypatch.setattr(cli, "configure_cpu_devices", lambda n: None)
    monkeypatch.setattr(line_roark, "run", lambda args: seen.append(args) or 0)
    assert line_roark.main([]) == 0
    assert line_roark.main(["--sweeps", "3", "--budget", "60"]) == 0
    assert (seen[0].budget, seen[0].sweeps) == (None, 8)
    assert (seen[1].budget, seen[1].sweeps) == (60, 3)


def test_command_is_the_argv_main_was_given(monkeypatch, tmp_path):
    from benchmarks import line_roark, players

    seen = []
    monkeypatch.setattr(cli, "configure_cpu_devices", lambda n: None)
    monkeypatch.setattr(line_roark, "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr("sys.argv", ["pytest", "-q", "unrelated"])
    assert line_roark.main(["--seed", "3"]) == 0
    command = line_roark._command(seen[0].argv)
    assert command == "python -m benchmarks.line_roark --seed 3"
    # The found file's header carries the dirty flag; the scoreboard still reads its deaths.
    header = line_roark._header(command, 3, 1, "abc", True)
    assert header.endswith("; seed 3, deaths 1, commit abc, dirty True")
    found = tmp_path / "found.toml"
    found.write_text(f"# {header}\n", encoding="utf-8")
    assert players.found_deaths(found) == 1


def test_line_default_and_found_file_name(monkeypatch):
    # The six-member file is the main fight; the found file carries the line file's stem, so
    # the earlier five-member kaizo_roark_found_k1_s0.toml is never overwritten.
    from benchmarks import line_roark

    seen = []
    monkeypatch.setattr(cli, "configure_cpu_devices", lambda n: None)
    monkeypatch.setattr(line_roark, "run", lambda args: seen.append(args) or 0)
    assert line_roark.main([]) == 0
    assert line_roark.main(["--line", "benchmarks/lines/kaizo_roark_user.toml"]) == 0
    assert seen[0].line == line_roark.LINES / "kaizo_roark_user6.toml"
    six = line_roark.found_path(seen[0].line, 1, 0)
    five = line_roark.found_path(seen[1].line, 1, 0)
    assert six.name == "kaizo_roark_found_kaizo_roark_user6_k1_s0.toml"
    assert five.name == "kaizo_roark_found_kaizo_roark_user_k1_s0.toml"
    assert six.parent == five.parent == line_roark.LINES
    assert line_roark._display(seen[0].line) == "benchmarks/lines/kaizo_roark_user6.toml"


def test_six_member_line_file_parses():
    from benchmarks import line_roark
    from nuzline.search.line_text import read_line_file

    file = read_line_file(line_roark.USER_LINE)
    assert len(file.team) == 6 and len(file.rules) == 6
    assert file.team[5]["pokemon"] == "Luvdisc"
    assert sum(t.get("item") == "Oran Berry" for t in file.team) == 3
