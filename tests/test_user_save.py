"""Full Roark coverage on the user's own save (plan 4a Task 11).

Skipped unless ``NUZLINE_USER_SAVE`` names the user's Platinum save (see README).
"""

from __future__ import annotations

import os

import pytest

from nuzline.cli import main
from nuzline.data.learnsets import load_learnsets
from nuzline.prep.pool import check_pool, pool_candidates
from nuzline.saves import load_save

_SAVE = os.environ.get("NUZLINE_USER_SAVE")

pytestmark = pytest.mark.skipif(not _SAVE, reason="NUZLINE_USER_SAVE not set")


def test_roark_coverage_is_full(capsys):
    assert main(["coverage", "--save", _SAVE, "--trainer", "roark"]) == 0
    out = capsys.readouterr().out
    assert out.count("Support: EXACT") == 2
    assert "Pool at level cap 16 (11 Pokemon):" in out
    assert out.count("Missing: none") == 2
    assert "UNSUPPORTED" not in out


def test_user_pool(tables, names):
    save = load_save(_SAVE, tables=tables, names=names)
    candidates, _ = pool_candidates(save, tables, names, load_learnsets(), 16)
    assert len(candidates) == 11
    assert {c.spec.level for c in candidates} == {16}
    assert sorted(c.candies for c in candidates) == sorted([0] * 6 + [12, 6, 6, 11, 10])

    def moves(ids):
        return {names.move_names[m] for m in ids if m}

    current = moves(m for c in candidates for m in c.spec.moves)
    assert {"Foresight", "Night Shade", "Disable", "Bide", "Mud Sport"} <= current
    options = moves(
        [m for c in candidates for m in c.relearner] + [m for c in candidates for _, m in c.tms]
    )
    assert {"Confusion", "Rollout", "Magnitude", "Bug Bite", "Hidden Power"} <= options

    tm10 = names.item("TM10")
    lacking = [c for c in candidates if tm10 not in {item for item, _ in c.tms}]
    assert [c.spec.species for c in lacking] == [names.species("Kricketot")]

    assert check_pool(tables, names, candidates, ()).missing == []


def test_prep_on_the_user_save(capsys):
    args = ["prep", "--save", _SAVE, "--trainer", "roark", "--max-depth", "1"]
    args += ["--finalists", "1", "--budget", "300", "--devices", "1"]
    assert main(args) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "Setup:" in out
    assert "Odds ladder (rung 0 planned):" in out
    assert "left out" not in out.lower()
    assert "left out" not in captured.err.lower()
