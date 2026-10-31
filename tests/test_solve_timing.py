"""``benchmarks/solve_timing.py``: the command line it runs and the facts it reads back."""

import pytest

from benchmarks.solve_timing import parse_output, solve_command

OUT = """Roark (Leader, id 246)
Monferno vs Geodude: seed 0
depth reached 2 of 2; 412345 playouts, 251.3 s
     #  action                    P(win, 0 faints) [95% CI]      lower  unexplored  ...
    1  Mach Punch                 94.1 % [92.8,  95.4]   90.0 %      0.1 %
  * 2  Ember                      93.0 % [91.0,  95.0]   89.0 %      0.1 %
the plan is shown under the best action only
precision target 1.5 pp: met (leader's half-A half-width 1.38 pp) after 3 top-up rounds
leaf budget 18000 playouts per pass (floor 1 per half); leaf estimates ...
"""


def test_parse_output_reads_the_precision_line():
    got = parse_output(OUT)
    assert got["completed_depth"] == 2 and got["playouts"] == 412345
    assert got["leader_ci"] == [92.8, 95.4]
    assert got["leader_half_width_pp"] == pytest.approx(1.3)
    assert got["rounds"] == 3 and got["precision_met"] is True
    assert got["half_width_a_pp"] == pytest.approx(1.38)


def test_parse_output_fixed_budget_and_missed_target():
    fixed = parse_output(OUT.replace(OUT.splitlines()[-2] + "\n", ""))
    assert fixed["rounds"] == 0 and fixed["precision_met"] is None
    assert fixed["half_width_a_pp"] is None
    missed = parse_output(OUT.replace(": met (", ": not met ("))
    assert missed["precision_met"] is False
    with pytest.raises(RuntimeError, match="no ranking"):
        parse_output("nothing here")


def test_solve_command_precision_or_playouts():
    cmd = solve_command(2, None, 1.5, None)
    assert cmd[cmd.index("--precision") + 1] == "1.5" and "--playouts" not in cmd
    assert cmd[cmd.index("--max-depth") + 1] == "2"
    cmd = solve_command(1, 80000, 1.5, 4)
    assert cmd[cmd.index("--playouts") + 1] == "80000" and "--precision" not in cmd
    assert cmd[cmd.index("--devices") + 1] == "4"
