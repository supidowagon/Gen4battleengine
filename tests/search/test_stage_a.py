"""Stage A ranking: Wilson intervals, common random numbers and the ranking order
(docs/design.md §4.3)."""

import jax
import numpy as np
import pytest

from nuzline.ai.policy import UnsupportedAi
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.search.deaths import DeathDistribution
from nuzline.search.stage_a import ActionResult, Ranking, rank_actions, wilson
from tests.engine.helpers import roark_real_state, roark_state
from tests.search.fixtures import at_replacement, dead_monferno

pytestmark = pytest.mark.slow


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


@pytest.mark.slow
def test_ranking_covers_legal_actions_and_is_sorted(tables, names):
    s = roark_state(tables, names)
    r = rank_actions(tables, names, s, n=200, key=jax.random.PRNGKey(0))
    assert len(r) == 9 and all(a.ci_low <= a.p_win0 <= a.ci_high for a in r)
    assert [a.p_win0 for a in r] == sorted([a.p_win0 for a in r], reverse=True)


@pytest.mark.slow
def test_common_random_numbers_are_deterministic(tables, names):
    s = roark_state(tables, names)
    a = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    b = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    assert [(x.action, x.p_win0) for x in a] == [(x.action, x.p_win0) for x in b]


@pytest.mark.slow
def test_all_zero_flag(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    s = s._replace(trapped=s.trapped.at[0].set(3))
    r = rank_actions(tables, names, s, n=50, key=jax.random.PRNGKey(1))
    assert r.all_zero is True


def test_tie_rule_prefers_fewer_expected_deaths():
    # Plan ruling 5: among rows overlapping the top score's interval the leader has the
    # fewest expected deaths (then the higher score, then the lower action); the others
    # follow by (-score, deaths, action); tied_with_best is relative to the leader.
    rows = [
        ActionResult(3, "c", 0.50, 0.40, 0.60, 0.7, 0.3, 100),
        ActionResult(1, "a", 0.50, 0.40, 0.60, 0.8, 0.2, 100),
        ActionResult(0, "z", 0.10, 0.05, 0.18, 0.1, 0.9, 100),
        ActionResult(2, "b", 0.52, 0.42, 0.62, 0.6, 0.4, 100),
        ActionResult(4, "d", 0.50, 0.40, 0.60, 0.8, 0.2, 100),
    ]
    r = Ranking.of(rows)
    assert [a.action for a in r] == [1, 2, 4, 3, 0]
    assert [a.tied_with_best for a in r] == [False, True, True, True, False]
    assert r.all_zero is False


def test_tie_rule_never_trades_a_whiteout_for_fewer_deaths():
    """Spec section 2: A (50 % deathless, 45 % one death, 5 % whiteout; 0.75 expected deaths)
    and B (50 % deathless, 50 % two deaths, no whiteout; 1.0) tie on rung 0: B leads."""
    a = ActionResult(0, "a", 0.5, 0.45, 0.55, 0.95, 0.75, 1000,
                     distribution=DeathDistribution((0.5, 0.45, 0, 0, 0, 0, 0.05, 0), 1000))  # fmt: skip
    b = ActionResult(1, "b", 0.5, 0.45, 0.55, 1.0, 1.0, 1000,
                     distribution=DeathDistribution((0.5, 0, 0.5, 0, 0, 0, 0, 0), 1000))  # fmt: skip
    assert [r.action for r in Ranking.of([a, b])] == [1, 0]


def test_rank_reports_ladder_and_distribution(tables, names):
    s = roark_real_state(tables, names).with_budget(k=2)
    r = rank_actions(tables, names, s, n=200, key=jax.random.PRNGKey(0))
    assert r
    for row in r:
        assert list(row.ladder) == sorted(row.ladder) and len(row.ladder) == 4
        assert row.ladder[0] == row.p_win0
        assert row.p_success == row.ladder[2] == row.score
        assert abs(sum(row.distribution.probs) - 1) < 1e-12
        assert row.distribution.n == 200


@pytest.mark.slow
def test_protected_death_fails_every_rung(tables, names):
    s = dead_monferno(tables, names).with_budget(3)
    monferno = int(np.flatnonzero(np.asarray(s.deaths))[0])
    s = s._replace(protected=s.protected.at[monferno].set(1))
    r = rank_actions(tables, names, s, n=64, key=jax.random.PRNGKey(0))
    assert r.all_zero
    for row in r:
        assert row.ladder == (0.0,) * 4 and row.p_success == 0.0
        assert row.distribution.probs[5] == row.p_win
    assert any(row.p_win > 0 for row in r)


@pytest.mark.slow
def test_rank_at_a_pending_root(tables, names):
    s = at_replacement(roark_state(tables, names))
    r = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(0))
    living = [i for i in range(1, 6) if int(s.hp[0, i]) > 0]
    assert sorted(a.action for a in r) == [MOVE_ACTIONS + i for i in living]
    for a in r:
        assert (
            a.label
            == f"Switch -> {names.species_names[int(s.species[0, a.action - MOVE_ACTIONS])]}"
        )


@pytest.mark.slow
def test_budget_is_traced(tables, names, monkeypatch):
    from nuzline.search import stage_a

    seen = []
    real = stage_a.cached_export

    def spy(fn, args, *, name, static, version):
        leaves = jax.tree.leaves(args)
        seen.append((name, static, [(x.shape, str(x.dtype)) for x in leaves]))
        return real(fn, args, name=name, static=static, version=version)

    monkeypatch.setattr(stage_a, "cached_export", spy)
    s = roark_state(tables, names, player_count=2)
    for k in (0, 2):
        rank_actions(tables, names, s.with_budget(k), n=4, key=jax.random.PRNGKey(0), max_turns=2)
    assert len(seen) == 2 and seen[0] == seen[1]


def test_a_clear_leader_has_no_ties():
    rows = [
        ActionResult(0, "a", 0.90, 0.85, 0.94, 0.9, 0.1, 100),
        ActionResult(1, "b", 0.20, 0.13, 0.29, 0.3, 0.8, 100),
    ]
    assert [a.tied_with_best for a in Ranking.of(rows)] == [False, False]


def test_labels_name_the_move_or_the_switch_target(tables, names):
    s = roark_state(tables, names, player_count=2)
    r = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(2))
    labels = {a.action: a.label for a in r}
    moves = [names.move_names[int(m)] for m in s.moves[0, 0]]
    target = names.species_names[int(s.species[0, 1])]
    assert labels == {0: moves[0], 1: moves[1], 2: moves[2], 3: moves[3], 5: f"Switch -> {target}"}


def test_struggle_is_labelled(tables, names):
    s = roark_state(tables, names, player_count=1).with_pp(0, 0, [0, 0, 0, 0])
    r = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(3))
    assert [(a.action, a.label) for a in r] == [(0, "Struggle")]


def test_stats_are_counts_over_the_shared_playouts(tables, names):
    # One Pokemon at 1 HP with no PP: Struggle's recoil faints it on turn 1 in every
    # playout, a certain loss.
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    (only,) = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(4))
    assert (only.p_win0, only.p_win, only.e_faints, only.n) == (0.0, 0.0, 1.0, 8)
    assert (only.ci_low, only.ci_high) == wilson(0, 8)


@pytest.mark.slow
def test_refused_playouts_are_refused_loudly(tables, names):
    # Levitate on the player's lead is refused by the AI: the ranking must not count
    # those playouts as losses.
    s = roark_state(tables, names, player_species="Gastly", player_ability="Levitate")
    with pytest.raises(UnsupportedAi, match=r"(?s)refused.*Levitate"):
        rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(5))


_MULTI_DEVICE_SCRIPT = """
import json, jax
from dataclasses import astuple
from nuzline.data.tables import load_names, load_tables
from nuzline.search.stage_a import rank_actions
from tests.engine.helpers import roark_real_state, roark_state
from tests.search.fixtures import at_replacement, dead_monferno

assert jax.device_count() == DEVICES, jax.device_count()
tables, names = load_tables(), load_names()
state = roark_state(tables, names)
ranking = rank_actions(tables, names, state, n=6, key=jax.random.PRNGKey(0), max_turns=3)
print(json.dumps([astuple(r) for r in ranking]))
"""


def _rank_on_devices(devices: int, script: str | None = None) -> list:
    """The ranking printed by ``script`` (default ``_MULTI_DEVICE_SCRIPT``) in a subprocess
    with ``devices`` host devices (whatever ``XLA_FLAGS`` this process has)."""
    script = _MULTI_DEVICE_SCRIPT if script is None else script
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
    }
    done = subprocess.run(
        [sys.executable, "-c", f"DEVICES = {devices}\n" + script],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_rank_actions_same_on_multiple_devices(tables):
    # Both runs in subprocesses, so the baseline has one device whatever XLA_FLAGS the
    # developer exported for this process.
    assert _rank_on_devices(4) == _rank_on_devices(1)


_KAIZO_MULTI_DEVICE_SCRIPT = """
import json, jax
from dataclasses import astuple
from nuzline.data.games import GAMES, KAIZO
from nuzline.ai.layout import fight_tables
from nuzline.data.tables import load_names, load_tables
from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
from nuzline.engine.coverage import start_weather
from nuzline.engine.state import make_state
from nuzline.search.stage_a import rank_actions

assert jax.device_count() == DEVICES, jax.device_count()
d = GAMES[KAIZO]
tables, names = load_tables(d.tables_dir), load_names(d.tables_dir)
roark = next(x for x in load_trainers(d.intermediate_dir) if x.name == "Roark")
party = list(build_trainer_party(roark, IntermediateData(d.intermediate_dir)))
tables = fight_tables(tables, roark, party)  # his AI layout (E1 Rulings R2, R6)
player = [party[2], party[5]]  # Roark's own Gible and Cranidos, as the coverage tests use
state = make_state(
    tables, player, party, start_weather(roark), roark.items, terrain=roark.terrain
)
ranking = rank_actions(tables, names, state, n=6, key=jax.random.PRNGKey(0), max_turns=3)
print(json.dumps([astuple(r) for r in ranking]))
"""


@pytest.mark.slow
def test_kaizo_rank_actions_same_on_multiple_devices(kaizo_tables):
    """K3c final review: the device-count claim of ``rank_actions`` for the Kaizo policy,
    whose playouts ``vmap`` the per-slot ``ai_action_distribution`` (a 1-ulp change there
    could change ``_choose``'s exact tie). Kaizo Roark at turn 0, 1 and 3 host devices
    (3 divides neither 6 nor a bucket, so the padding differs)."""
    one = _rank_on_devices(1, _KAIZO_MULTI_DEVICE_SCRIPT)
    assert one and _rank_on_devices(3, _KAIZO_MULTI_DEVICE_SCRIPT) == one


def test_single_pokemon_ladder_is_flat(tables, names):
    # No replacement exists: whatever k allows, the ladder cannot rise past rung 0.
    s = roark_state(tables, names, player_count=1).with_budget(k=2)
    rows = rank_actions(tables, names, s, n=200, key=jax.random.PRNGKey(0))
    assert rows
    for r in rows:
        assert len(set(r.ladder)) == 1
