"""Plan tree output (``search/plan_tree.py``; Stage B spec §5): ASCII labels that tell
sibling observations apart, a JSON tree that keeps every node and conserves probability,
the rendered layout and a root with a single action."""

from __future__ import annotations

import dataclasses
import json

import jax
import numpy as np
import pytest

from nuzline.search.leaves import LeafValue
from nuzline.search.plan_tree import label, render, to_json
from nuzline.search.stage_b import PlanNode, SolveSettings, solve
from tests.engine.helpers import roark_state

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
