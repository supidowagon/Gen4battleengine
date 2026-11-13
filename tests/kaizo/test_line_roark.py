"""The user's Kaizo Roark line reproduces its measured odds (L1 spec
docs/superpowers/specs/2026-11-12-l1-line-player-design.md section 7).

``line odds``' internals (``cli.line_setup``, then ``evaluate_lines`` with refused
playouts left out) on ``benchmarks/lines/kaizo_roark_user.toml`` from the pre-Roark save,
``--deaths 3``, 400 playouts, seed 0. The user measured 72 % wins and 42.5 % with at most
one death at 200 playouts; prototypes measured 68.2 % / 41.7 % (400) and 67.8 % / 41.5 %
(1000). The bands are those figures with sampling error. The empty line (the default
player on every turn) on the same keys wins far less. Skips without the save or the Kaizo
tables; the save's path comes from ``tools.rom.inputs`` and the save is never copied.
"""

from pathlib import Path

import pytest

USER_LINE = Path(__file__).resolve().parents[2] / "benchmarks" / "lines" / "kaizo_roark_user.toml"
ROARK_ID = 246  # Kaizo Leader Roark
N, SEED, DEATHS = 400, 0, 3


@pytest.mark.slow
def test_user_line_reproduces_roark_odds(kaizo_save_pre_roark_path, kaizo_tables):
    del kaizo_tables  # only skips until the Kaizo tables are built; line_setup loads them
    import jax

    from nuzline.cli import LINE_ODDS_MAX_TURNS, line_setup
    from nuzline.search.line import empty_line
    from nuzline.search.player_eval import evaluate_lines

    setup = line_setup(
        "kaizo", ROARK_ID, USER_LINE, save_path=kaizo_save_pre_roark_path, deaths=DEATHS
    )
    assert len(setup.rules) == 6  # one rule per opposing Pokemon
    user, empty = evaluate_lines(
        setup.tables,
        setup.state,
        [setup.line, empty_line()],
        N,
        jax.random.PRNGKey(SEED),
        LINE_ODDS_MAX_TURNS,
        refused="exclude",
    )
    any_win, at_most_one = user.wins_any / user.n, user.rung(1)
    print(f"user line: {any_win:.3f} any win, {at_most_one:.3f} <=1 death, {user.n} counted")
    print(f"empty line: {empty.wins_any / empty.n:.3f} any win")
    assert 0.60 <= any_win <= 0.80
    assert 0.32 <= at_most_one <= 0.52
    assert empty.wins_any / empty.n < 0.25
