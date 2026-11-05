"""The prep checkpoint (plan 4c Task 8): a resumed run equals the uninterrupted one, a
checkpoint of another run is refused without being touched, and a write is atomic.

The funnel runs on Task 7's fakes (a fake ``screen`` and ``solve`` that read a setup's identity
back from its turn-0 state), so a run costs seconds; the fake Stage B value depends on the setup
only, since a resumed run makes fewer calls than the uninterrupted one.
"""

import dataclasses
import json
import os
import time

import jax
import pytest

from nuzline.prep import checkpoint, search
from nuzline.prep.checkpoint import (
    CHECKPOINT_VERSION,
    fingerprint,
    read_checkpoint,
    setup_from_json,
    setup_to_json,
    stats_from_json,
    stats_to_json,
    write_checkpoint,
)
from nuzline.prep.problem import Resources
from nuzline.prep.screen import ScreenStats
from nuzline.prep.search import run_prep
from nuzline.prep.setups import neighbours, team_setups
from tests.engine.helpers import sample_party
from tests.prep.fixtures import candidate, problem
from tests.prep.test_search import KEY, SETTINGS, Fakes, fake_solve, falling, make_problem

VERSIONS = {"table_version": "tables-1", "code_version": "code-1"}


def setup_run(monkeypatch, tables, names, **problem_kw):
    prob = make_problem(tables, names, **problem_kw)
    fakes = Fakes(monkeypatch, falling)
    monkeypatch.setattr(
        search,
        "solve",
        fake_solve(fakes, [], lambda s, i: 0.5 + 0.04 * (fakes.order.index(s) % 10)),
    )
    return prob, fakes


def interrupting(after: int):
    """A ``progress`` raising KeyboardInterrupt on the ``after``-th unit."""
    seen = []

    def progress(p):
        seen.append(p)
        if len(seen) == after:
            raise KeyboardInterrupt

    return progress, seen


def test_resume_equals_uninterrupted(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    units = []
    a = run_prep(tables, names, prob, SETTINGS, KEY, progress=units.append, **VERSIONS)
    total = len(units)
    assert a.stats.stopped is None and a.plan is not None and a.stats.variant_rounds == 1
    phases = {u.phase for u in units}
    assert {"teams", "variants 1", "finalists"} <= phases
    # a stop in the team phase, at its end, in a variant round, in the finalists and at the end
    for n, stop in enumerate([1, 3, 4, total - 2, total - 1, total]):
        path = tmp_path / f"run{n}.ckpt.json"
        progress, _ = interrupting(stop)
        b = run_prep(
            tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS
        )
        assert b.stats.stopped == "interrupted"
        if stop == 4:  # inside the expansion: more rounds to go, its parents saved
            pos = json.loads(path.read_text())["funnel"]["pos"]
            assert pos["phase"] == "variants" and pos["r"] == 1
            assert len(pos["pools"][prob.deaths]) > SETTINGS.variant_keep
            assert pos["parents"][prob.deaths]
            assert not set(pos["parents"][prob.deaths]) & set(pos["pools"][prob.deaths])
        c = run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS)
        assert c == a, f"stopped after unit {stop} of {total}"
        assert c.stats.stopped is None
        # the resumed run keeps writing to the same file: a second resume has nothing to do
        d = run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS)
        assert d == a


def test_interrupt_in_the_first_unit_of_a_resumed_run_keeps_the_answer(
    tables, names, monkeypatch, tmp_path
):
    prob, _ = setup_run(monkeypatch, tables, names)
    units = []
    run_prep(tables, names, prob, SETTINGS, KEY, progress=units.append, **VERSIONS)
    path = tmp_path / "first.ckpt.json"
    progress, _ = interrupting(len(units) - 1)  # stops inside the finalists
    run_prep(tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS)

    def solve_interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "solve", solve_interrupted)
    r = run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS)
    assert r.stats.stopped == "interrupted"
    assert r.chosen is not None


def test_resume_equals_uninterrupted_with_a_deep_leader(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    settings = dataclasses.replace(SETTINGS, max_depth=2)
    units = []
    a = run_prep(tables, names, prob, settings, KEY, progress=units.append, **VERSIONS)
    path = tmp_path / "deep.ckpt.json"
    progress, _ = interrupting(len(units) - 1)  # after the second finalist, before the deepening
    run_prep(tables, names, prob, settings, KEY, checkpoint=path, progress=progress, **VERSIONS)
    assert run_prep(tables, names, prob, settings, KEY, resume=path, **VERSIONS) == a


def test_resume_inside_the_merged_stage(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    probe = tmp_path / "probe.ckpt.json"
    seen = []
    a = run_prep(
        tables, names, prob, SETTINGS, KEY, checkpoint=probe,
        progress=lambda p: seen.append(json.loads(probe.read_text())["funnel"]["pos"]),
        **VERSIONS,
    )  # fmt: skip
    merged = [n for n, pos in enumerate(seen, 1) if pos.get("stage") == "merged"]
    assert merged, "the run never reached a merged round"
    assert "parents" not in seen[merged[0] - 1]
    path = tmp_path / "merged.ckpt.json"
    progress, _ = interrupting(merged[0])
    b = run_prep(tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS)
    assert b.stats.stopped == "interrupted"
    assert run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS) == a


def test_resume_reuses_the_saved_finalist_picks(tables, names, monkeypatch, tmp_path):
    prob, fakes = setup_run(monkeypatch, tables, names)
    settings = dataclasses.replace(SETTINGS, variant_keep=3)  # a pool larger than the picks
    units = []
    run_prep(tables, names, prob, settings, KEY, progress=units.append, **VERSIONS)
    first_final = next(n for n, u in enumerate(units, 1) if u.phase == "finalists")
    path = tmp_path / "picks.ckpt.json"
    progress, _ = interrupting(first_final)
    run_prep(tables, names, prob, settings, KEY, checkpoint=path, progress=progress, **VERSIONS)
    doc = json.loads(path.read_text())
    pos = doc["funnel"]["pos"]
    assert pos["phase"] == "finalists" and len(pos["picks"]) == settings.finalists
    other = next(i for i in pos["pool"] if i not in pos["picks"])
    pos["picks"][1] = other  # a resume must solve what the position says, not re-rank the pool
    path.write_text(json.dumps(doc))
    record = []
    monkeypatch.setattr(search, "solve", fake_solve(fakes, record, lambda s, i: 0.5))
    run_prep(tables, names, prob, settings, KEY, resume=path, **VERSIONS)
    setups = [setup_from_json(s) for s in doc["funnel"]["setups"]]
    assert record[0][0] == setups[other]


def test_resume_after_the_race_reuses_its_picks(tables, names, monkeypatch, tmp_path):
    prob, fakes = setup_run(monkeypatch, tables, names)
    settings = dataclasses.replace(SETTINGS, variant_keep=3)  # a pool larger than the picks
    units = []
    a = run_prep(tables, names, prob, settings, KEY, progress=units.append, **VERSIONS)
    race = [n for n, u in enumerate(units, 1) if u.phase == "race"]
    assert len(race) == 1 and units[race[0]].phase == "finalists"
    path = tmp_path / "race.ckpt.json"
    progress, _ = interrupting(race[0])
    b = run_prep(tables, names, prob, settings, KEY, checkpoint=path, progress=progress, **VERSIONS)
    assert b.stats.stopped == "interrupted"
    doc = json.loads(path.read_text())
    pos = doc["funnel"]["pos"]
    assert pos["phase"] == "finalists" and pos["next"] == 0 and not pos["deep"]
    assert len(pos["picks"]) == settings.finalists < len(pos["pool"])
    calls = len(fakes.calls)
    record = []
    monkeypatch.setattr(
        search,
        "solve",
        fake_solve(fakes, record, lambda s, i: 0.5 + 0.04 * (fakes.order.index(s) % 10)),
    )
    c = run_prep(tables, names, prob, settings, KEY, resume=path, **VERSIONS)
    assert len(fakes.calls) == calls  # never re-raced
    setups = [setup_from_json(s) for s in doc["funnel"]["setups"]]
    assert [s for s, _ in record] == [setups[i] for i in pos["picks"]]
    assert c == a


def test_version_2_checkpoint_is_refused(tables, names, saved):
    _, path = saved
    doc = json.loads(path.read_text())
    before = json.dumps({**doc, "version": 2})
    path.write_text(before)
    with pytest.raises(ValueError, match="version 2"):
        read_checkpoint(path, doc["fingerprint"])
    assert path.read_text() == before  # nothing written


def test_elapsed_time_adds_up_across_sessions(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    now = [0.0]

    def clock():
        now[0] += 1.0
        return now[0]

    path = tmp_path / "t.ckpt.json"
    progress, _ = interrupting(3)
    b = run_prep(
        tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, clock=clock,
        **VERSIONS,
    )  # fmt: skip
    saved = json.loads(path.read_text())["funnel"]["elapsed"]
    assert saved > 0 and b.stats.seconds >= saved
    c = run_prep(tables, names, prob, SETTINGS, KEY, resume=path, clock=clock, **VERSIONS)
    assert c.stats.seconds > saved


def refusal_cases(tables, names):
    other_cands = make_problem(tables, names)
    bite = names.move("Bite")
    party = sample_party(names)
    changed = [
        candidate(names, party[i], i, relearner=(bite if i else names.move("Ember"),))
        for i in range(3)
    ]
    moves_changed = problem(
        tables, names, changed, resources=Resources(1, (), (), ()), levelled=[()] * 3
    )
    assert moves_changed.candidates != other_cands.candidates
    return [
        ("seed", {"key": jax.random.PRNGKey(8)}, "another seed"),
        ("deaths", {"problem": make_problem(tables, names, deaths=1)}, "another save or pool"),
        ("cap", {"problem": make_problem(tables, names, cap=17)}, "another save or pool"),
        ("moves", {"problem": moves_changed}, "another save or pool"),
        ("settings", {"settings": dataclasses.replace(SETTINGS, screen_n=6)}, "other settings"),
        ("tables", {"table_version": "tables-2"}, "other tables"),
        ("code", {"code_version": "code-2"}, "another code version"),
    ]


@pytest.fixture
def saved(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    path = tmp_path / "saved.ckpt.json"
    progress, _ = interrupting(3)
    run_prep(
        tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS
    )  # fmt: skip
    return prob, path


@pytest.mark.parametrize("which", range(7))
def test_resume_refuses_mismatch(tables, names, saved, which):
    prob, path = saved
    name, change, message = refusal_cases(tables, names)[which]
    before = path.read_bytes()
    args = {
        "problem": prob,
        "settings": SETTINGS,
        "key": KEY,
        **VERSIONS,
    }
    args.update(change)
    with pytest.raises(ValueError, match=message):
        run_prep(
            tables, names, args.pop("problem"), args.pop("settings"), args.pop("key"),
            resume=path, **args,
        )  # fmt: skip
    assert path.read_bytes() == before, name


def test_resume_refuses_a_damaged_or_future_file(tables, names, saved):
    prob, path = saved
    good = path.read_bytes()
    doc = json.loads(good)
    cases = {
        "cut in half": (good[: len(good) // 2], "damaged"),
        "not json": (b"hello", "damaged"),
        "no funnel": (json.dumps({"version": 1, "fingerprint": {}}).encode(), "damaged"),
        "other version": (
            json.dumps({**doc, "version": CHECKPOINT_VERSION + 1}).encode(),
            f"version {CHECKPOINT_VERSION + 1}",
        ),
    }
    for name, (blob, message) in cases.items():
        path.write_bytes(blob)
        with pytest.raises(ValueError, match=message):
            run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS)
        assert path.read_bytes() == blob, name
    with pytest.raises(ValueError, match="damaged"):
        run_prep(tables, names, prob, SETTINGS, KEY, resume=path.with_name("missing"), **VERSIONS)
    assert not path.with_name("missing").exists()


def test_fingerprint_ignores_the_budget_only(tables, names):
    prob = make_problem(tables, names)
    base = fingerprint(prob, SETTINGS, 7, "t", "c")
    assert set(base) == {"problem", "settings", "seed", "tables", "code"}
    budgeted = dataclasses.replace(SETTINGS, budget_seconds=5.0)
    assert fingerprint(prob, budgeted, 7, "t", "c") == base
    assert fingerprint(prob, dataclasses.replace(SETTINGS, finalists=3), 7, "t", "c") != base
    assert fingerprint(prob, SETTINGS, 8, "t", "c")["seed"] == "8"
    protected = make_problem(tables, names, protected=[1])
    assert fingerprint(protected, SETTINGS, 7, "t", "c")["problem"] != base["problem"]


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "atomic.ckpt.json"
    first = {"version": CHECKPOINT_VERSION, "fingerprint": {}, "funnel": {"n": 1}}
    write_checkpoint(path, first)
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append(src)
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint.os, "replace", replace)
    with pytest.raises(OSError):
        write_checkpoint(path, {**first, "funnel": {"n": 2}})
    assert calls and json.loads(path.read_text()) == first
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp file is left behind
    monkeypatch.setattr(checkpoint.os, "replace", real)
    write_checkpoint(path, {**first, "funnel": {"n": 3}})
    assert json.loads(path.read_text())["funnel"] == {"n": 3}


def test_read_checkpoint_never_writes(tmp_path):
    path = tmp_path / "r.ckpt.json"
    print_ = {"problem": "p", "settings": "s", "seed": "1", "tables": "t", "code": "c"}
    write_checkpoint(
        path, {"version": CHECKPOINT_VERSION, "fingerprint": print_, "funnel": {"x": 1}}
    )
    before = path.read_bytes()
    assert read_checkpoint(path, print_)["funnel"] == {"x": 1}
    for key in print_:  # the first differing key is the one named
        other = {**print_, key: "other"}
        with pytest.raises(ValueError, match="checkpoint is for"):
            read_checkpoint(path, other)
    first = {**print_, "problem": "x", "code": "y"}
    with pytest.raises(ValueError, match="save or pool"):
        read_checkpoint(path, first)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_setup_and_stats_json_round_trip(tables, names):
    prob = make_problem(tables, names)
    parent = team_setups(prob, 2)[8]
    setups = [parent, *neighbours(tables, prob, parent)]
    assert any(m.moves or m.item or m.hp is not None for s in setups for m in s.members)
    for s in setups:
        assert setup_from_json(json.loads(json.dumps(setup_to_json(s)))) == s
    stats = ScreenStats((3, 1, 0, 0, 0, 0, 2, 0), 5, 1.25, 0.5, True)
    assert stats_from_json(json.loads(json.dumps(stats_to_json(stats)))) == stats


def test_restore_rebuilds_the_same_tie_ranks(tables, names):
    prob = make_problem(tables, names)
    teams = team_setups(prob, 2)

    def funnel(key=KEY):
        return search._Funnel(tables, names, prob, SETTINGS, key, None, time.perf_counter)

    f = funnel()
    f.register(teams)
    g = funnel()
    g.restore(f.snapshot()["funnel"])
    assert g.tie == f.tie and len(set(f.tie)) == len(f.tie)
    h = funnel()
    h.register(list(reversed(teams)))
    assert sorted(h.tie) == sorted(f.tie)  # registration order does not change any rank
    other = funnel(jax.random.PRNGKey(8))
    other.register(teams)
    assert other.tie != f.tie
