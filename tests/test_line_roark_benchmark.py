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
