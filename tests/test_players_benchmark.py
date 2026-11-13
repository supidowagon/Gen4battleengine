"""The player scoreboard (benchmarks/players.py, L1 spec section 7a)."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import players
from nuzline.ai.policy import UnsupportedAi
from nuzline.search import deaths, player, player_eval
from nuzline.search.player import GREEDY
from nuzline.search.player_eval import PlayerOdds
from tools.rom.inputs import RomInputs

ROARK = "kaizo-roark-user-five-k1"


def test_fights_file_parses():
    fights = players.read_fights()
    assert fights, "benchmarks/fights.toml has no [[fight]]"
    assert len({f.name for f in fights}) == len(fights)
    known = {f.name for f in dataclasses.fields(RomInputs)}
    for f in fights:
        assert (players.BENCH / f.team).is_file(), f.team
        if f.found is not None:
            # The found-line file may not exist yet (the entry is then left out), but it
            # must name a line file under benchmarks/.
            assert f.found.endswith(".toml") and not f.found.startswith(("/", "\\")), f.found
            assert (players.BENCH / f.found).parent.is_dir(), f.found
        assert f.save in known, f.save
        assert "/" not in f.save and "\\" not in f.save and "." not in f.save, f.save
        assert 0 <= f.deaths <= 3
    assert ROARK in {f.name for f in fights}


def _odds(n: int) -> PlayerOdds:
    return PlayerOdds(
        n=n,
        wins=(1, 2, 3, 4),
        wins_any=5,
        deaths_total=6,
        distribution=deaths.DeathDistribution((0.0,) * len(deaths.BUCKETS), n),
        mean_turns=7.0,
        fallback_share=0.5,
        refused=2,
    )


def test_scoreboard_rows(monkeypatch):
    calls = []

    def recorder(tables, state, who, params, n, key, max_turns=100, refused="raise"):
        if params == [("refused",)]:
            raise UnsupportedAi("every playout of entry 0 refused")
        calls.append(
            {
                "player": who,
                "params": params,
                "n": n,
                "key": np.asarray(key),
                "refused": refused,
            }
        )
        return [_odds(n) for _ in params]

    monkeypatch.setattr(player_eval, "evaluate_players", recorder)
    monkeypatch.setattr(player, "refusal_reason", lambda *a, **k: "some mechanic")
    ctx = SimpleNamespace(
        fight=SimpleNamespace(name="fake"), setup=SimpleNamespace(tables="T", state="S")
    )
    entries = [
        players.Entry("a", GREEDY, lambda c: ()),
        players.Entry("skipped", GREEDY, lambda c: None),
        players.Entry("all refused", GREEDY, lambda c: ("refused",)),
        players.Entry("b", GREEDY, lambda c: ("p",)),
    ]
    said = []
    rows = players.scoreboard([ctx], entries, n=16, seed=3, report=said.append)
    assert said == ["all refused: all playouts refused: some mechanic"]
    assert [(r["fight"], r["label"]) for r in rows] == [("fake", "a"), ("fake", "b")]
    assert len(calls) == 2
    assert [c["params"] for c in calls] == [[()], [("p",)]]
    assert all(c["n"] == 16 and c["refused"] == "exclude" for c in calls)
    assert np.array_equal(calls[0]["key"], calls[1]["key"])
    import jax

    assert np.array_equal(calls[0]["key"], np.asarray(jax.random.PRNGKey(3)))
    row = rows[0]
    assert row["n"] == 16 and row["wins"] == (1, 2, 3, 4) and row["refused"] == 2
    assert row["fallback_share"] == 0.5 and "refused_key" not in row


@pytest.mark.slow
def test_scoreboard_on_roark(kaizo_save_pre_roark_path):
    fight = next(f for f in players.read_fights() if f.name == ROARK)
    ctx = players.fight_context(fight, kaizo_save_pre_roark_path)
    entries = [e for e in players.entries() if e.label in ("greedy", "user line")]
    rows = {r["label"]: r for r in players.scoreboard([ctx], entries, n=200, seed=0)}
    greedy, user = rows["greedy"], rows["user line"]
    assert greedy["wins_any"] / greedy["n"] < user["wins_any"] / user["n"]
    assert 0.58 <= user["wins_any"] / user["n"] <= 0.82


def test_found_deaths(tmp_path):
    path = tmp_path / "found.toml"
    header = "# python -m benchmarks.line_roark; seed 0, deaths 1, commit abc"
    path.write_text(header + "\n\n[[team]]\n", encoding="utf-8")
    assert players.found_deaths(path) == 1
    path.write_text("[[team]]\npokemon = 'X'\n", encoding="utf-8")
    assert players.found_deaths(path) is None


def test_entries_alias():
    assert [e.label for e in players.ENTRIES] == [e.label for e in players.entries()]
