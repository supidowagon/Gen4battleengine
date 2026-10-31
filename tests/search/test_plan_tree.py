"""Plan tree output (``search/plan_tree.py``; Stage B spec §5): ASCII labels that tell
sibling observations apart, a JSON tree that keeps every node and conserves probability,
the rendered layout and a root with a single action."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from nuzline.search.leaves import LeafValue
from nuzline.search.plan_tree import label, render, to_json
from nuzline.search.stage_b import SolveSettings, solve
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
    tree = to_json(result, names, root=state)
    again = json.loads(json.dumps(tree))
    assert again == tree

    def count(node):
        return 1 + sum(count(k) for kids in node["children"].values() for k in kids)

    assert count(again["plan"]) == len(list(walk(result.plan)))
    assert again["depth"] == 1 and again["playouts"] == result.playouts
    assert len(again["ranking"]) == len(result.ranking) and again["dropped"] == []
    assert again["table_version"] and again["code_version"]
    assert again["settings"]["max_depth"] == 1
    stack = [again["plan"]]
    while stack:
        node = stack.pop()
        for kids in node["children"].values():
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
