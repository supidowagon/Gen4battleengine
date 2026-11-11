"""Plan tree output (``search/plan_tree.py``; Stage B spec §5): ASCII labels that tell
sibling observations apart, a JSON tree that keeps every node and conserves probability,
the rendered layout and a root with a single action."""

from __future__ import annotations

import dataclasses
import json

import jax
import numpy as np
import pytest

from nuzline.search import stage_b
from nuzline.search.deaths import BUCKETS
from nuzline.search.leaves import LeafValue
from nuzline.search.plan_tree import label, render, to_json
from nuzline.search.stage_a import ActionResult, Ranking
from nuzline.search.stage_b import PlanNode, SolveSettings, solve
from tests.engine.helpers import roark_state
from tests.search.fixtures import certain_zero_leaf, sacrifice_endgame

pytestmark = pytest.mark.slow

KEY = jax.random.PRNGKey(0)
SETTINGS = SolveSettings(max_depth=1, budget_playouts=100, n_depth0=10)


def zero_leaf(states, weights, pairs=None):
    return [LeafValue(0, 1, 0, 1, False) for _ in range(len(states.hp))]


@pytest.fixture(scope="module")
def roark(tables, names):
    state = roark_state(tables, names)
    return state, solve(tables, names, state, KEY, SETTINGS, leaf_fn=zero_leaf)


def walk(node):
    yield node
    for kids in node.children.values():
        for kid in kids:
            yield from walk(kid)


@pytest.mark.slow
def test_labels_ascii_and_distinct(roark, names):
    state, result = roark
    nodes = list(walk(result.plan))
    assert len(nodes) > 10
    tree = to_json(result, names, root=state)
    stack = [tree["plan"]]
    while stack:
        node = stack.pop()
        for kids in node["children"].values():
            labels = [k["label"] for k in kids]
            assert all(lab.isascii() for lab in labels)
            assert len(set(labels)) == len(labels)
            # No sibling is told apart by a bare "#n": the differing field is spelled out.
            assert not any(" #" in lab for lab in labels), labels
            stack.extend(kids)
    text = render(result, names, 2, root=state)
    assert all(line.isascii() for line in text)
    for node in nodes:
        if node.obs is not None:
            assert label(node.obs, names).isascii()


def test_label_reads_the_observation(roark, names):
    state, result = roark
    hits = [n for n in walk(result.plan) if n.obs is not None]
    labels = {label(n.obs, names, max_hp=np.asarray(state.max_hp)[0].tolist()) for n in hits}
    assert any("used" in lab and "you" in lab and "HP" in lab and "foe" in lab for lab in labels)
    assert any("missed" in lab for lab in labels) or any("crit" in lab for lab in labels)
    assert any("/" in lab for lab in labels)  # "you 12/58 HP" with the maximum known


def test_json_roundtrip(roark, names):
    state, result = roark
    tree = to_json(result, names, root=state, table_version="0123456789abcdef")
    again = json.loads(json.dumps(tree))
    assert again == tree
    assert again["table_version"] == "0123456789abcdef"  # the solve's tables, as given
    assert to_json(result, names, root=state)["table_version"] is None
    assert again["leaf_budget"] == result.leaf_budget == 100

    def count(node):
        return 1 + sum(count(k) for kids in node["children"].values() for k in kids)

    assert count(again["plan"]) == len(list(walk(result.plan)))
    assert again["depth"] == 1 and again["playouts"] == result.playouts
    assert len(again["ranking"]) == len(result.ranking) and again["dropped"] == []
    assert again["table_version"] and again["code_version"]
    assert again["settings"]["max_depth"] == 1
    assert again["plan"]["depth"] == 0
    stack = [again["plan"]]
    while stack:
        node = stack.pop()
        for kids in node["children"].values():
            assert all(k["depth"] == node["depth"] + 1 for k in kids)
            assert sum(k["prob"] for k in kids) == pytest.approx(
                node["prob"], abs=1e-6
            )  # AI rows are float32
            stack.extend(kids)
    root_values = again["plan"]["values"]
    assert set(root_values) == {r["label"] for r in again["ranking"]}
    assert again["plan"]["children"].keys() == root_values.keys()


def test_render_layout(roark, names):
    state, result = roark
    lines = render(result, names, 1, root=state)
    assert "P(win, 0 faints)" in lines[1]
    best = result.ranking[0].label
    assert best in lines[2]
    assert any(line.lstrip().startswith("if ") and "->" in line for line in lines)
    assert "see the full state" in lines[-1] and "leaf budget 100" in lines[-1]
    assert not any(line.lstrip().startswith("if ") for line in render(result, names, 0, root=state))


def test_single_action(tables, names):
    state = roark_state(tables, names, player_count=1, opp_count=1, player_moves=["Scratch"])
    state = state._replace(moves=state.moves.at[0, 0, 1:].set(0))
    result = solve(tables, names, state, KEY, SETTINGS, leaf_fn=zero_leaf)
    assert len(result.ranking) == 1
    lines = render(result, names, 1, root=state)
    rows = [line for line in lines if line.strip()[:1] == "1" or " 1  " in line[:8]]
    assert len(rows) == 1 and "Scratch" in rows[0]
    tree = to_json(result, names, root=state)
    assert list(tree["plan"]["children"]) == ["Scratch"]


def test_siblings_differing_only_in_hidden_label_fields_are_named(roark, names):
    from types import SimpleNamespace

    from nuzline.engine.observe import OBS_FIELDS
    from nuzline.search.plan_tree import _Walk, decode

    state, result = roark
    base = next(n for n in walk(result.plan) if n.obs is not None).obs
    sl = dict(OBS_FIELDS)

    def variant(edits=None):
        vec = np.frombuffer(base, np.int32).copy()
        for (name, i), v in (edits or {}).items():
            vec[sl[name].start + i] = v
        return SimpleNamespace(obs=vec.tobytes(), kind="leaf", children={})

    kids = [variant(), variant({("stages", 8 + 5): -1}), variant({("confusion", 1): 1})]
    node = SimpleNamespace(children={0: kids})
    labels = [lab for _, lab, _ in _Walk(result, names, state, "Roark").kids(node, None, 0)]
    assert len(set(labels)) == 3 and not any(" #" in lab for lab in labels), labels
    assert any("foe Spe -1" in lab for lab in labels)
    assert any("foe confused" in lab for lab in labels)
    assert decode(kids[0].obs)["stages"].sum() == decode(base)["stages"].sum()


def test_unexpanded_siblings_name_the_opponent_action(roark, names):
    """Nodes with no observation (pruned opponent actions, a sampled pair, an AI refused
    below the root) are labelled by the opponent action they stand for, with the kind
    printed once per line, and stay distinct."""
    state, result = roark
    best = result.ranking[0].action

    def forced(o: int, text: str, kind: str) -> PlanNode:
        return PlanNode(None, 0.01, None, {}, {}, kind, opp_action=o, opp_label=text)

    kids = [
        forced(0, "Geodude uses Tackle", "pruned"),
        forced(1, "Geodude uses Rock Throw", "pruned"),
        forced(2, "Geodude uses Rock Throw", "sampled"),
        PlanNode(None, 0.01, None, {}, {}, "refused"),
    ]
    plan = dataclasses.replace(result.plan, children={best: kids})
    fake = dataclasses.replace(result, plan=plan)
    lines = [line.strip() for line in render(fake, names, 1, root=state) if "  if " in line]
    assert len(lines) == len(kids) and len(set(lines)) == len(lines), lines
    assert lines[0].startswith("if Geodude uses Tackle  [")
    assert lines[1].startswith("if Geodude uses Rock Throw  [")
    assert lines[3].startswith("if the foe's AI is not modelled here  [")
    for line, text in zip(lines, ("unexplored", "unexplored", "sampled", "refused"), strict=True):
        assert line.count(text) == 1, line
        assert "#" not in line, line
    tree = to_json(fake, names, root=state)
    (json_kids,) = tree["plan"]["children"].values()
    labels = [k["label"] for k in json_kids]
    assert len(set(labels)) == len(labels), labels
    assert labels[0] == "Geodude uses Tackle" and "[" not in "".join(labels), labels
    assert [k["opp_action"] for k in json_kids] == [0, 1, 2, None]


def test_render_shows_the_sampled_node_count(roark, names):
    state, result = roark
    rows = [dataclasses.replace(r, sampled=0.0123, sampled_nodes=3) for r in result.ranking]
    lines = render(dataclasses.replace(result, ranking=rows), names, 0, root=state)
    assert "sampled (nodes)" in lines[1]
    assert "1.2 % (3)" in lines[2]


def test_depth0_prints_no_plan_and_labels_its_leaves(tables, names, monkeypatch):
    """At depth 0 (Stage A) ``render`` shows no plan under the best action, and the JSON
    leaves say what they are: every foe reply, valued by Stage A's AI-driven playouts."""
    canned = Ranking.of(
        [
            ActionResult(0, "Ember", 0.625, 0.55, 0.69, 0.75, 0.4, 200),
            ActionResult(2, "Scratch", 0.5, 0.43, 0.57, 0.6, 0.9, 200),
        ]
    )
    monkeypatch.setattr(stage_b, "rank_actions", lambda *args: canned)
    settings = SolveSettings(max_depth=0, n_depth0=200)
    result = solve(None, names, roark_state(tables, names), KEY, settings)
    lines = render(result, names, 1)
    assert not any(line.lstrip().startswith("if ") for line in lines), lines
    assert "Stage A (depth 0)" in lines[-1]
    tree = to_json(result, names)
    for kids in tree["plan"]["children"].values():
        assert [(k["kind"], k["label"]) for k in kids] == [
            ("leaf", "any foe reply (Stage A playouts)")
        ]
    text = json.dumps(tree)
    assert "not modelled" not in text


def test_unlabelled_unobserved_node_is_an_error(roark, names):
    """An obs-None node with no opponent action that is neither a Stage A leaf nor a
    refusal has no truthful label: rendering it raises rather than guess."""
    state, result = roark
    best = result.ranking[0].action
    plan = dataclasses.replace(
        result.plan, children={best: [PlanNode(None, 1.0, None, {}, {}, "pruned")]}
    )
    with pytest.raises(ValueError, match="no opponent action"):
        render(dataclasses.replace(result, plan=plan), names, 1, root=state)


def test_precision_footer_and_json(roark, names):
    state, result = roark
    # Fixed budget: no precision line, the JSON says no target.
    assert not any("precision target" in line for line in render(result, names, 0, root=state))
    plain = to_json(result, names, root=state)
    assert plain["rounds"] == 0 and plain["half_width_a"] is None
    assert plain["precision_met"] is None
    settings = dataclasses.replace(SETTINGS, precision=0.015)
    met = dataclasses.replace(
        result, settings=settings, rounds=3, half_width_a=0.0123, precision_met=True
    )
    lines = render(met, names, 0, root=state)
    assert "see the full state" in lines[-1] and "leaf budget 100" in lines[-1]
    (line,) = [x for x in lines if "precision target" in x]
    assert "1.5 pp" in line and "met" in line and "not met" not in line
    assert "1.23 pp" in line and "3 top-up rounds" in line
    tree = json.loads(json.dumps(to_json(met, names, root=state)))
    assert tree["rounds"] == 3 and tree["half_width_a"] == 0.0123
    assert tree["precision_met"] is True and tree["settings"]["precision"] == 0.015
    missed = dataclasses.replace(met, rounds=8, half_width_a=0.02, precision_met=False)
    (line,) = [x for x in render(missed, names, 0, root=state) if "precision target" in x]
    assert "not met" in line and "2.00 pp" in line and "8 top-up rounds" in line
    # The target prints as given, never rounded to one decimal (0.25 pp, not 0.2 pp).
    for target, shown in ((0.0025, "0.25 pp"), (0.0125, "1.25 pp"), (0.0004, "0.04 pp")):
        fine = dataclasses.replace(met, settings=dataclasses.replace(settings, precision=target))
        (line,) = [x for x in render(fine, names, 0, root=state) if "precision target" in x]
        assert f"precision target {shown}:" in line, line


# ---- the faint budget: death labels, replacement and failed lines, distribution (plan 4b) ----


DEPTH2 = SolveSettings(max_depth=2, epsilon=0.0, n_depth0=8)


@pytest.fixture(scope="module")
def sacrifice(tables, names):
    """The Task 9 ``sacrifice_endgame`` solved at ``k = 0`` and ``k = 1`` (zero leaves)."""
    base = sacrifice_endgame(tables, names)
    out = {}
    for k in (0, 1):
        state = base.with_budget(k)
        out[k] = (state, solve(tables, names, state, KEY, DEPTH2, leaf_fn=certain_zero_leaf))
    return out


@pytest.mark.slow
def test_label_names_the_faint_and_the_budget(sacrifice, names):
    state, result = sacrifice[1]
    leer = names.move_slot(state, 0, "Leer")
    fainted = [
        n
        for kid in result.plan.children[leer]
        for n in walk(kid)
        if n.obs is not None and "monferno fainted" in label(n.obs, names, budget=(1, 0)).lower()
    ]
    assert fainted
    obs = fainted[0].obs
    # The names table is upper case (MONFERNO); compare case-insensitively.

    def low(**kw):
        return label(obs, names, **kw).lower()

    assert "monferno fainted (death 1 of 1 allowed)" in low(budget=(1, 0))
    assert "monferno fainted (death 1, over the 0 allowed)" in low(budget=(0, 0))
    assert "fainted (death" not in low()  # no budget: the faint only
    assert "monferno fainted" in low()


def test_render_replacement_and_failed_lines(sacrifice, names):
    state, result = sacrifice[1]
    text = render(result, names, 2, root=state)
    assert all(line.isascii() for line in text)
    assert result.ranking[0].label == "Leer"
    assert any(
        "monferno fainted" in ln.lower() and ln.lower().endswith("-> send in prinplup")
        for ln in text
    )
    assert any("P(win, <=1 deaths) [95% CI]" in ln and "over budget" in ln for ln in text)
    assert any(ln.startswith("deaths under the leader: 0: ") for ln in text)
    state0, result0 = sacrifice[0]
    text0 = render(result0, names, 2, root=state0)
    assert result0.ranking[0].label.lower() == "switch -> prinplup"
    assert any(
        ln.endswith("(over the death budget: play on for the win, not planned)") for ln in text0
    )
    assert any("P(win, 0 faints) [95% CI]" in ln for ln in text0)
    assert not any(ln.startswith("protected:") for ln in text0)


def test_render_names_the_protected_slots(tables, names, sacrifice):
    state, _ = sacrifice[1]
    state = state.with_budget(1, [1])
    result = solve(tables, names, state, KEY, DEPTH2, leaf_fn=certain_zero_leaf)
    text = render(result, names, 1, root=state)
    assert "protected: prinplup (a line where one faints fails)" in [ln.lower() for ln in text]
    assert to_json(result, names, root=state)["protected"] == [1]


def test_json_has_replacement_and_distribution(sacrifice, names):
    state, result = sacrifice[1]
    tree = json.loads(json.dumps(to_json(result, names, root=state)))
    assert tree["deaths_allowed"] == 1 and tree["protected"] == []
    for row in tree["ranking"]:
        assert row["distribution"] is not None
        assert tuple(row["distribution"]) == (*BUCKETS, "n")
        assert "e_deaths" in row and "failed" in row
    stack, replacements = [tree["plan"]], []
    while stack:
        node = stack.pop()
        assert isinstance(node["replacement"], bool)
        replacements.append(node["replacement"])
        for kids in node["children"].values():
            stack.extend(kids)
    assert any(replacements)
