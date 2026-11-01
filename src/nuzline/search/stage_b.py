"""Stage B: depth-limited expectimax over information sets with Stage A leaves (Stage B
spec §3-4; plan rulings 4-10).

The tree alternates **belief nodes** (a weighted set of states at a player decision
point; the root holds the one given state) and chance layers: for a player action ``a``,
each state ``s`` of the node (absolute reach ``w``) meets every opponent action ``o``
with the AI's exact probability ``q_s(o)`` (``expand.ai_rows``), and every merged battle
outcome of ``(a, o)`` (``expand.expand``: stepped, replaced and observed, merged on
state and observation bytes). The children of ``a`` over all the node's states are
grouped by observation (and the replacement-refused flag) into child belief nodes: the
player picks one action per observation, never per hidden state (spec *Anti-goals*).
A node's value for ``a`` is therefore ``sum_s w_s sum_o q_s(o) sum_children p * value
(group)``; every value is carried as an absolute (reach-weighted) sum and divided by the
node's reach only for display (``PlanNode.values``).

Absolute reach (``PlanNode.prob``, the leaf weights, the epsilon test; ruling 10) is the
product of chance probabilities on the path: belief weight x opponent-action probability
x outcome probability; player choices are excluded. Child kinds:

* ``terminal``: the battle is over (the observation carries ``over`` and ``winner``, so a
  group is all over or none); value 1 iff the player won (state.py ``winner``), never
  pruned, no playouts.
* ``refused`` (ruling 4): the AI's row is refused (NaN) for a state below the root, the
  opponent's replacement was refused (``send_replacements``), or a playout of the node's
  evaluation met a refused mechanic. Valued 0 in the estimate and the bound; its reach is
  reported. A refused AI at the root refuses the solve (``UnsupportedAi``).
* ``pruned``: an opponent action of one state, or a child group, whose reach is below
  ``epsilon``: not expanded, valued by ``2 * leaf_floor`` playouts in round 0 (``evaluate_forced``
  for the opponent action, ``evaluate_leaves`` for each state of the group), 0 in the
  lower bound.
* ``sampled``: a pair below the root that ``enumerate_outcomes`` cannot enumerate
  (``SamplingOnly``, the width cap), valued like a pruned opponent action. At the root
  such a pair drops the player action from the ranking, named with the reason.
* ``leaf``: a group at the pass depth, each state valued by Stage A playouts
  (``evaluate_leaves``; the playout policy reads the full state, which the plan never
  does; spec §3).
* ``decision``: a group above the pass depth, expanded on the next level.

Each level is expanded in one batch (``ai_rows`` and ``expand`` over every frontier
state and pair, through an expansion cache keyed by state and pair that later passes
reuse); each round of a pass's evaluations runs as one ``evaluate_leaves`` and one
``evaluate_forced`` call. States shared by several leaves (transpositions) are evaluated
once.

Leaf budgets (rulings 8 and 9 as amended 2026-10-31, the precision target). **Round 0**
follows ruling 8: ``B_d`` (``budget_playouts``, else ``n_depth0`` per legal root action)
split by ``leaves.allocate`` in proportion to a state's summed leaf reach, a floor of
``leaf_floor`` (default 1) per half; pruned and sampled evaluations get exactly the floor.
With ``precision`` set, the pass then checks the half-A leader (the root action half A
ranks first, ties to the lowest action): its half-A half-width ``Z sqrt(sum_j W_j^2
var_a_j)`` (``LeafValue.var_a``). Above ``precision``, a **top-up round** ``r >= 1`` adds
as many playouts as the pass has run so far (the total doubles), split over every
evaluation -- leaves, pruned states and forced pairs -- in proportion to its summed
reach with no floor (a share rounding to 0 gets nothing; a refused evaluation gets
none), on the fresh keys ``fold_in(fold_in(fold_in(key, 2), r), 0 | 1)`` (round 0 uses
``fold_in(key, 0 | 1)``); wins and playouts add into each evaluation's halves, ``refused``
is or-ed. It stops when the target is met, after ``MAX_ROUNDS`` top-ups, or once
``budget_seconds`` has passed since the solve started; the pass always completes, and
the backup below runs once on the merged values. Half B never decides a stop or a
share. Without ``precision`` a pass is round 0 alone.

Max bias (ruling 5): each evaluation has two halves with independent keys. Half A picks
the action at every inner belief node; half B gives every reported value, bound and
interval, under half A's choices. Interval (ruling 6): for a root action, ``W_j`` is the
total reach of the tree leaves (and pruned or sampled nodes) evaluated by evaluation
``j`` under half A's policy, transpositions summed before squaring; ``var = sum_j W_j^2
var_j`` (``LeafValue.var_b``), the interval ``estimate +- 1.96 sqrt(var)`` clipped to
[0, 1]; terminal and refused children add no variance. Iterative deepening (ruling 9)
runs passes ``d = 1 .. max_depth`` from the root, each complete; ``budget_seconds``
decides whether the next pass, or the next top-up round, starts. Depth 0 is Stage A
(ruling 7).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np

import nuzline.search.stage_a  # noqa: F401 - registers Tables/BattleState for jax.export
from nuzline.ai.policy import UnsupportedAi, ai_refusals
from nuzline.data.tables import Names, Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION, MOVE_ACTIONS, legal_actions, usable_moves
from nuzline.engine.enumerate import SamplingOnly
from nuzline.engine.state import BattleState
from nuzline.engine.step import WINNER_PLAYER
from nuzline.search import deaths
from nuzline.search.expand import ai_rows, expand
from nuzline.search.leaves import LeafValue, allocate, evaluate_forced, evaluate_leaves
from nuzline.search.rows import concat, row_bytes, take
from nuzline.search.stage_a import action_label, rank_actions, tie_ordered

__all__ = ["ActionValue", "HiddenLegality", "PlanNode", "SolveResult", "SolveSettings", "solve"]

Z = 1.96  # 95 % normal interval (ruling 6)
MAX_ROUNDS = 8  # top-up rounds per pass at most (precision target)

LeafFn = Callable[..., list[LeafValue]]


@dataclass(frozen=True)
class SolveSettings:
    """``max_depth``: deepest pass (0 = Stage A). ``budget_playouts``: the round-0 leaf
    budget ``B_d`` of every pass (``None``: ``n_depth0`` per legal root action).
    ``budget_seconds``: no pass starts, and no top-up round, after this much time (depth
    1 always runs, its round 0 included). ``epsilon``: the pruning threshold on absolute
    reach. ``leaf_floor``: playouts per half at least in round 0. ``max_turns``: playout
    length from a leaf. ``n_depth0``: Stage A's playouts per action. ``precision``: the
    target half-width (a fraction: 0.015 is 1.5 percentage points) of the half-A leader's
    95 % interval; a pass tops up its evaluations until it is met (module docstring;
    ``None``: one round of ``B_d``)."""

    max_depth: int = 2
    budget_playouts: int | None = None
    budget_seconds: float | None = None
    epsilon: float = 1e-4
    leaf_floor: int = 1
    max_turns: int = 100
    n_depth0: int = 2000
    precision: float | None = None


@dataclass(frozen=True)
class ActionValue:
    """One ranked root action. ``estimate``: P(win with zero player faints) under the
    plan, half-B values; ``ci_low``/``ci_high`` its 95 % interval (ruling 6);
    ``lower_bound``: the estimate with pruned and sampled nodes counted 0; ``unexplored``,
    ``sampled``, ``refused``: the summed reach of those nodes in the action's plan;
    ``sampled_nodes``: how many sampled nodes that is (spec section 1)."""

    action: int
    label: str
    estimate: float
    ci_low: float
    ci_high: float
    lower_bound: float
    unexplored: float
    sampled: float
    refused: float
    tied_with_best: bool
    sampled_nodes: int = 0


@dataclass
class PlanNode:
    """A node of the plan. ``obs``: the observation bytes that lead to it (``None`` at
    the root and for nodes not expanded into observations: a pruned opponent action, a
    sampled pair, a state whose AI is refused). ``prob``: absolute reach (ruling 10).
    ``action``: the recommended action (decision nodes); ``values``: every legal action's
    half-B value given the node is reached. ``children``: per player action, the child
    nodes (at the root every ranked action; at an inner decision node only the chosen
    one; other kinds none). ``kind``: ``decision``, ``leaf``, ``terminal``, ``pruned``,
    ``refused`` or ``sampled`` (module docstring). ``opp_action``/``opp_label``: for a
    node of one opponent action not expanded into observations (a pruned opponent
    action, a sampled pair, or either refused by its evaluation), that action and its
    text, e.g. ``Geodude uses Rock Throw``; ``None`` otherwise."""

    obs: bytes | None
    prob: float
    action: int | None
    values: dict[int, float]
    children: dict[int, list[PlanNode]]
    kind: str
    opp_action: int | None = None
    opp_label: str | None = None


@dataclass
class SolveResult:
    """``ranking``: the root actions, best first; ``dropped``: root actions left out, with
    the reason; ``depth``: the deepest completed pass; ``playouts``: every playout run,
    over all passes; ``seconds``: wall time (not compared: equal solves differ in it);
    ``leaf_budget``: the round-0 leaf budget ``B_d`` every pass used (``None`` at depth 0,
    where Stage A runs ``n_depth0`` playouts per action instead; ruling 7). Of the final
    pass: ``rounds``, its top-up rounds; ``half_width_a``, the half-A leader's final half-A
    half-width (``None`` without a precision target, or with no root action);
    ``precision_met``, whether that is at most the target (``None`` likewise)."""

    ranking: list[ActionValue]
    dropped: dict[int, str]
    depth: int
    plan: PlanNode
    playouts: int
    seconds: float = field(compare=False)
    settings: SolveSettings
    leaf_budget: int | None = None
    rounds: int = 0
    half_width_a: float | None = None
    precision_met: bool | None = None


class HiddenLegality(Exception):
    """The states of one belief node differ in the player's legal actions: the player
    could tell them apart by the move menu, so the observation misses something visible
    (spec section 3 assumes legality is observable; a hidden lock such as Encore,
    Disable or a choice item would do this). The plan cannot be one action per
    observation, so the solve stops."""


@dataclass
class _Node:
    """A node under construction. Belief-like nodes hold ``states`` (state id ->
    absolute reach); a pruned opponent action or sampled pair holds ``forced`` (state
    id, player action, opponent action)."""

    kind: str
    obs: bytes | None
    prob: float
    states: dict[int, float] = field(default_factory=dict)
    forced: tuple[int, int, int] | None = None
    legal: list[int] = field(default_factory=list)
    branches: dict[int, list[_Node]] = field(default_factory=dict)


class _Store:
    """The distinct states of a solve, one id per state bytes."""

    def __init__(self):
        self.ids: dict[bytes, int] = {}
        self.parts: list[BattleState] = []
        self.over: list[bool] = []
        self.won: list[bool] = []
        self._all: BattleState | None = None

    def add(self, batch: BattleState) -> list[int]:
        n = len(batch.hp)
        keys = row_bytes(batch, n)
        out, new = [], []
        for i in range(n):
            k = keys[i].tobytes()
            sid = self.ids.get(k)
            if sid is None:
                sid = self.ids[k] = len(self.ids)
                new.append(i)
            out.append(sid)
        if new:
            part = take(batch, np.asarray(new))
            over = np.asarray(part.over) != 0
            self.parts.append(part)
            self.over.extend(over.tolist())
            won = over & (np.asarray(part.winner) == WINNER_PLAYER)
            ok = deaths.success(
                won, part.player_faints, part.deaths, part.faint_budget, part.protected
            )
            self.won.extend(ok.tolist())
            self._all = None
        return out

    def take(self, ids) -> BattleState:
        if self._all is None:
            self._all = concat(self.parts)
            self.parts = [self._all]
        return take(self._all, np.asarray(ids, np.int64))


_legal_jit = jax.jit(lambda tables, state: legal_actions(tables, state, 0))


def _opp_label(names: Names, state: BattleState, o: int) -> str:
    if o < MOVE_ACTIONS:
        mon = int(state.active[1])
        return names.move_names[int(state.moves[1, mon, o])]
    if o < FIRST_ITEM_ACTION:
        return f"switch to {names.species_names[int(state.species[1, o - MOVE_ACTIONS])]}"
    return names.item_names[int(state.trainer_items[1, o - FIRST_ITEM_ACTION])]


def _opp_phrase(names: Names, state: BattleState, o: int) -> str:
    """Opponent action ``o`` at ``state`` as a plan label: ``Geodude uses Rock Throw``,
    ``the foe switches to Onix``, ``the foe uses Potion``."""
    if o < MOVE_ACTIONS:
        mon = names.species_names[int(state.species[1, int(state.active[1])])]
        return f"{mon} uses {_opp_label(names, state, o)}"
    if o < FIRST_ITEM_ACTION:
        return f"the foe {_opp_label(names, state, o).replace('switch', 'switches', 1)}"
    return f"the foe uses {_opp_label(names, state, o)}"


def _mass(nodes: list[PlanNode], kind: str) -> float:
    """Summed ``prob`` of the ``kind`` nodes under ``nodes``, following the children the
    plan keeps (the chosen action at inner decision nodes)."""
    total = 0.0
    for node in nodes:
        if node.kind == kind:
            total += node.prob
        for kids in node.children.values():
            total += _mass(kids, kind)
    return total


def _count(nodes: list[PlanNode], kind: str) -> int:
    """How many ``kind`` nodes ``_mass`` sums over."""
    return sum(
        (node.kind == kind) + sum(_count(kids, kind) for kids in node.children.values())
        for node in nodes
    )


class _Solver:
    def __init__(
        self, tables: Tables, names: Names, root, settings: SolveSettings, leaf_fn, start=None
    ):
        self.tables, self.names, self.settings, self.leaf_fn = tables, names, settings, leaf_fn
        self.start = time.perf_counter() if start is None else start
        self.store = _Store()
        self.store.add(root)  # id 0
        self.root = _first(root)
        self.struggle = not bool(np.any(np.asarray(usable_moves(tables, self.root, 0))))
        self.ai: dict[int, np.ndarray | None] = {}  # float64 row, None when refused
        self.legal: dict[int, list[int]] = {}
        self.expansions: dict[tuple[int, int, int], list | SamplingOnly] = {}
        self.playouts = 0

    def _label(self, a: int) -> str:
        return action_label(self.names, self.root, a, self.struggle)

    def _forced_plan(self, forced: tuple[int, int, int], prob: float, kind: str) -> PlanNode:
        """The plan node of a forced (state, player, opponent) evaluation, naming the
        opponent action."""
        s, _, o = forced
        text = _opp_phrase(self.names, _first(self.store.take([s])), o)
        return PlanNode(None, prob, None, {}, {}, kind, opp_action=o, opp_label=text)

    # -- expansion ------------------------------------------------------------------------

    def _legal(self, sid: int) -> list[int]:
        if sid not in self.legal:
            state = jax.tree.map(lambda f: f[0], self.store.take([sid]))
            mask = np.asarray(_legal_jit(self.tables, state))
            self.legal[sid] = np.flatnonzero(mask).tolist()
        return self.legal[sid]

    def _node_legal(self, node: _Node) -> list[int]:
        """The player's legal actions at belief node ``node``: equal over its states
        (spec section 3), else ``HiddenLegality``."""
        sids = iter(node.states)
        first = next(sids)
        legal = self._legal(first)
        for sid in sids:
            if self._legal(sid) != legal:
                raise HiddenLegality(
                    f"states {first} and {sid} of one belief node have the legal actions "
                    f"{legal} and {self._legal(sid)}: the observation hides the move menu"
                )
        return legal

    def _ai_rows(self, sids: list[int]) -> None:
        need = [s for s in dict.fromkeys(sids) if s not in self.ai]
        if not need:
            return
        rows = ai_rows(self.tables, self.store.take(need)).astype(np.float64)
        for s, row in zip(need, rows, strict=True):
            self.ai[s] = None if np.isnan(row).any() else row

    def _expand(self, requests: list[tuple[int, int, int]]) -> None:
        sids = list(dict.fromkeys(r[0] for r in requests))
        index = {s: i for i, s in enumerate(sids)}
        pairs = [(index[s], a, o) for s, a, o in requests]
        local, out = expand(self.tables, self.store.take(sids), pairs)
        ids = self.store.add(local) if len(local.hp) else []
        for r, request in enumerate(requests):
            kids = out[r]
            if not isinstance(kids, SamplingOnly):
                kids = [(ids[c.state_row], c.obs, c.prob, c.refused) for c in kids]
            self.expansions[request] = kids

    def _level(self, frontier: list[_Node], k: int, depth: int, dropped: dict[int, str]) -> None:
        """Expand every node of ``frontier`` (``k`` turns played) one turn: children per
        legal action, grouped and classified (module docstring)."""
        eps = self.settings.epsilon
        self._ai_rows([s for node in frontier for s in node.states])
        if k == 0 and self.ai[0] is None:
            reasons = ai_refusals(self.tables, self.root)
            raise UnsupportedAi("; ".join(reasons) or "the AI's action distribution is refused")
        requests = {}
        for node in frontier:
            node.legal = self._node_legal(node)
            for a in node.legal:
                for s, w in node.states.items():
                    q = self.ai[s]
                    for o in [] if q is None else np.flatnonzero(q > 0).tolist():
                        # At the root every pair with mass is expanded: a sampling-only one
                        # drops the player action even when its reach is below epsilon.
                        if (k == 0 or w * q[o] >= eps) and (s, a, o) not in self.expansions:
                            requests[(s, a, o)] = None
        if requests:
            self._expand(list(requests))
        for node in frontier:
            if k == 0:
                for a in node.legal:
                    for o in np.flatnonzero(self.ai[0] > 0).tolist():
                        kids = self.expansions[(0, a, o)]
                        if isinstance(kids, SamplingOnly) and a not in dropped:
                            dropped[a] = (
                                f"{self._label(a)} is sampling-only against "
                                f"{_opp_label(self.names, self.root, o)}: {kids.reason}"
                            )
            for a in node.legal:
                if k == 0 and a in dropped:
                    continue
                node.branches[a] = self._children(node, a, k, depth)

    def _children(self, node: _Node, a: int, k: int, depth: int) -> list[_Node]:
        eps = self.settings.epsilon
        kids: list[_Node] = []
        groups: dict[tuple[bytes, bool], dict[int, float]] = {}
        for s, w in node.states.items():
            q = self.ai[s]
            if q is None:
                kids.append(_Node("refused", None, w))
                continue
            for o in np.flatnonzero(q > 0).tolist():
                reach = w * q[o]
                if reach < eps:
                    kids.append(_Node("pruned", None, reach, forced=(s, a, o)))
                    continue
                expansion = self.expansions[(s, a, o)]
                if isinstance(expansion, SamplingOnly):
                    kids.append(_Node("sampled", None, reach, forced=(s, a, o)))
                    continue
                for child, obs, p, refused in expansion:
                    members = groups.setdefault((obs, refused), {})
                    members[child] = members.get(child, 0.0) + reach * p
        for (obs, refused), members in groups.items():
            prob = math.fsum(members.values())
            if refused:
                kind = "refused"
            elif self.store.over[next(iter(members))]:
                kind = "terminal"
            elif prob < eps:
                kind = "pruned"
            elif k + 1 == depth:
                kind = "leaf"
            else:
                kind = "decision"
            kids.append(_Node(kind, obs, prob, states=members))
        return kids

    # -- evaluation -----------------------------------------------------------------------

    def _evaluate(self, sids, weights, pairs, counts, key) -> list[LeafValue]:
        """``counts[i]`` playouts of ``sids[i]`` (the default evaluators; a ``leaf_fn``
        gets ``weights`` instead and picks its own counts)."""
        if not sids:
            return []
        states = self.store.take(sids)
        weights = np.asarray(weights, np.float64)
        if self.leaf_fn is not None:
            values = list(self.leaf_fn(states, weights, pairs))
        else:
            turns = self.settings.max_turns
            if pairs is None:
                values = evaluate_leaves(self.tables, states, counts, key, turns)
            else:
                values = evaluate_forced(self.tables, states, pairs, counts, key, turns)
        if len(values) != len(sids):
            raise ValueError(f"leaf evaluator returned {len(values)} values for {len(sids)}")
        self.playouts += sum(v.n_a + v.n_b for v in values)
        return values

    # -- backup ---------------------------------------------------------------------------

    def _backup(self, node: _Node, leaf, forced, with_plan: bool = True):
        """``(A, B, bound, W, plan nodes)`` of ``node``: absolute half-A and half-B values,
        the half-B lower bound, ``{evaluation: reach}`` (ruling 6) and its plan nodes (none
        unless ``with_plan``: the precision check needs only the values)."""
        kind, prob = node.kind, node.prob
        if kind == "refused":
            return 0.0, 0.0, 0.0, {}, [PlanNode(node.obs, prob, None, {}, {}, "refused")]
        if kind == "terminal":
            won = math.fsum(w for s, w in node.states.items() if self.store.won[s])
            return won, won, won, {}, [PlanNode(node.obs, prob, None, {}, {}, "terminal")]
        if node.forced is not None:
            i, value = forced[node.forced]
            if value.refused:
                plan = [self._forced_plan(node.forced, prob, "refused")] if with_plan else []
                return 0.0, 0.0, 0.0, {}, plan
            plan = [self._forced_plan(node.forced, prob, kind)] if with_plan else []
            return prob * value.p_a, prob * value.p_b, 0.0, {("F", i): prob}, plan
        if kind in ("leaf", "pruned"):
            a_sum, b_sum, weights, kept, lost = 0.0, 0.0, {}, [], []
            for s, w in node.states.items():
                i, value = leaf[s]
                if value.refused:
                    lost.append(w)
                    continue
                kept.append(w)
                a_sum += w * value.p_a
                b_sum += w * value.p_b
                weights[("L", i)] = weights.get(("L", i), 0.0) + w
            plan = []
            if kept:
                plan.append(PlanNode(node.obs, math.fsum(kept), None, {}, {}, kind))
            if lost:
                plan.append(PlanNode(node.obs, math.fsum(lost), None, {}, {}, "refused"))
            return a_sum, b_sum, b_sum if kind == "leaf" else 0.0, weights, plan
        per_action = {
            a: self._branch(kids, leaf, forced, with_plan) for a, kids in node.branches.items()
        }
        chosen = _leader(per_action)
        a_val, b_val, bound, weights, plan = per_action[chosen]
        values = {a: v[1] / prob for a, v in per_action.items()}
        node_plan = PlanNode(node.obs, prob, chosen, values, {chosen: plan}, "decision")
        return a_val, b_val, bound, weights, [node_plan]

    def _branch(self, kids: list[_Node], leaf, forced, with_plan: bool = True):
        a_sum, b_sum, bound, weights, plan = 0.0, 0.0, 0.0, {}, []
        for kid in kids:
            a, b, lb, w, nodes = self._backup(kid, leaf, forced, with_plan)
            a_sum += a
            b_sum += b
            bound += lb
            for j, x in w.items():
                weights[j] = weights.get(j, 0.0) + x
            plan.extend(nodes)
        return a_sum, b_sum, bound, weights, plan

    # -- one pass -------------------------------------------------------------------------

    def _leader_half_width(self, root: _Node, leaf, forced, values) -> float | None:
        """The half-A half-width ``Z sqrt(sum_j W_j^2 var_a_j)`` of the root action half A
        picks (``_leader``); ``None`` when no root action is ranked. Half B is never read
        (ruling 5: it gives the reported values only). ``values``: ``{"L": leaf values,
        "F": forced values}``, indexed as the backup's evaluation ids."""
        if not root.branches:
            return None
        per_action = {
            a: self._branch(kids, leaf, forced, with_plan=False)
            for a, kids in root.branches.items()
        }
        _, _, _, weights, _ = per_action[_leader(per_action)]
        var = math.fsum(w * w * values[kind][i].var_a for (kind, i), w in weights.items())
        return Z * math.sqrt(var)

    def _top_up(self, key, sids, pairs, reach, counts, values) -> int:
        """One evaluator's part of a top-up round: ``counts[i]`` more playouts of each
        state given any (``reach``: the weights a ``leaf_fn`` sees), merged into
        ``values`` in place. Returns the playouts added."""
        pick = np.flatnonzero(counts >= 2)
        if not len(pick):
            return 0
        sub_pairs = None if pairs is None else pairs[pick]
        new = self._evaluate([sids[i] for i in pick], reach[pick], sub_pairs, counts[pick], key)
        added = 0
        for i, v in zip(pick, new, strict=True):
            old = values[i]
            values[i] = old.merge(v)
            added += v.n_a + v.n_b
        return added

    def run(self, depth: int, key, budget: int):
        """One pass of depth ``depth``: ``(ranking, dropped, plan, rounds, half-width,
        precision met)`` (module docstring; the last two ``None`` without a target)."""
        dropped: dict[int, str] = {}
        root = _Node("decision", None, 1.0, states={0: 1.0})
        frontier = [root]
        for k in range(depth):
            self._level(frontier, k, depth, dropped)
            frontier = [
                kid
                for node in frontier
                for kids in node.branches.values()
                for kid in kids
                if kid.kind == "decision"
            ]
        leaf_ids: dict[int, int] = {}
        leaf_w: list[float] = []  # ruling 8 weights: leaf nodes only
        leaf_reach: list[float] = []  # top-up weights: leaf and pruned nodes
        forced_ids: dict[tuple[int, int, int], int] = {}
        forced_reach: list[float] = []
        stack = [kid for a in root.branches for kid in root.branches[a]][::-1]
        while stack:
            node = stack.pop()
            if node.forced is not None:
                i = forced_ids.setdefault(node.forced, len(forced_ids))
                if i == len(forced_reach):
                    forced_reach.append(0.0)
                forced_reach[i] += node.prob
            elif node.kind in ("leaf", "pruned"):
                for s, w in node.states.items():
                    i = leaf_ids.setdefault(s, len(leaf_ids))
                    if i == len(leaf_w):
                        leaf_w.append(0.0)
                        leaf_reach.append(0.0)
                    if node.kind == "leaf":
                        leaf_w[i] += w
                    leaf_reach[i] += w
            elif node.kind == "decision":
                stack.extend(kid for a in node.branches for kid in node.branches[a][::-1])
        floor = self.settings.leaf_floor
        leaf_sids, forced_sids = list(leaf_ids), [f[0] for f in forced_ids]
        k_leaf, k_forced = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1)
        leaf_values = self._evaluate(
            leaf_sids, leaf_w, None, allocate(np.asarray(leaf_w), budget, floor), k_leaf
        )
        pairs = np.asarray([f[1:] for f in forced_ids], np.int32).reshape(-1, 2)
        zeros = np.zeros(len(forced_ids))
        forced_values = self._evaluate(
            forced_sids, zeros, pairs, allocate(zeros, budget, floor), k_forced
        )

        def lookup():
            return (
                {s: (i, leaf_values[i]) for s, i in leaf_ids.items()},
                {f: (i, forced_values[i]) for f, i in forced_ids.items()},
            )

        leaf, forced = lookup()
        rounds, half_a, met = 0, None, None
        precision = self.settings.precision
        if precision is not None:
            spent = sum(v.n_a + v.n_b for v in leaf_values + forced_values)
            seconds = self.settings.budget_seconds
            k_top = jax.random.fold_in(key, 2)  # round 0 uses 0 and 1: no stream reused
            while True:
                both = {"L": leaf_values, "F": forced_values}
                half_a = self._leader_half_width(root, leaf, forced, both)
                met = None if half_a is None else half_a <= precision
                late = seconds is not None and time.perf_counter() - self.start > seconds
                if met is not False or rounds == MAX_ROUNDS or late:
                    break
                # ``spent`` more playouts over every evaluation by its summed reach, no
                # floor (a share rounding to 0 gets nothing); refused ones get none.
                reach = np.asarray(leaf_reach + forced_reach, np.float64)
                reach[[v.refused for v in leaf_values + forced_values]] = 0.0
                counts = allocate(reach, spent, 0) if len(reach) else np.zeros(0, np.int64)
                n_leaf = len(leaf_values)
                k_round = jax.random.fold_in(k_top, rounds + 1)
                added = self._top_up(
                    jax.random.fold_in(k_round, 0),
                    leaf_sids,
                    None,
                    reach[:n_leaf],
                    counts[:n_leaf],
                    leaf_values,
                )
                added += self._top_up(
                    jax.random.fold_in(k_round, 1),
                    forced_sids,
                    pairs,
                    reach[n_leaf:],
                    counts[n_leaf:],
                    forced_values,
                )
                if not added:
                    break
                rounds += 1
                spent += added
                leaf, forced = lookup()
        variance = {("L", i): v.var_b for i, v in enumerate(leaf_values)}
        variance.update({("F", i): v.var_b for i, v in enumerate(forced_values)})
        rows = []
        plans = {}
        for a, kids in root.branches.items():
            _, b_val, bound, weights, plan = self._branch(kids, leaf, forced)
            var = math.fsum(w * w * variance[j] for j, w in weights.items())
            half = Z * math.sqrt(var)
            estimate = min(1.0, max(0.0, b_val))
            plans[a] = plan
            rows.append(
                ActionValue(
                    action=a,
                    label=self._label(a),
                    estimate=estimate,
                    ci_low=max(0.0, estimate - half),
                    ci_high=min(1.0, estimate + half),
                    lower_bound=min(1.0, max(0.0, bound)),
                    unexplored=_mass(plan, "pruned"),
                    sampled=_mass(plan, "sampled"),
                    refused=_mass(plan, "refused"),
                    tied_with_best=False,
                    sampled_nodes=_count(plan, "sampled"),
                )
            )
        ranking = _rank(rows)
        plan = PlanNode(
            None,
            1.0,
            ranking[0].action if ranking else None,
            {r.action: r.estimate for r in ranking},
            {r.action: plans[r.action] for r in ranking},
            "decision",
        )
        return ranking, dropped, plan, rounds, half_a, met


def _first(batch: BattleState) -> BattleState:
    return jax.tree.map(lambda f: f[0], batch)


def _leader(per_action: dict) -> int:
    """The action with the highest half-A value (``per_action[a][0]``), the lowest action
    among ties: the choice at every inner belief node and the precision check's leader."""
    return max(per_action, key=lambda a: (per_action[a][0], -a))


def _rank(rows: list[ActionValue]) -> list[ActionValue]:
    """``stage_a.tie_ordered`` by ``estimate``; the expected deaths are not valued here
    (Task 9), so every row has the same and the tie rule leaves the best estimate first."""
    return tie_ordered(rows, lambda r: r.estimate, lambda r: 0.0)


def _stage_a(tables, names, state, key, settings: SolveSettings, start: float) -> SolveResult:
    """Depth 0 (ruling 7): ``rank_actions``' rows, valued by their ``score``."""
    ranking = rank_actions(tables, names, state, settings.n_depth0, key, settings.max_turns)
    rows = [
        ActionValue(
            action=r.action,
            label=r.label,
            estimate=r.score,
            ci_low=r.ci_low,
            ci_high=r.ci_high,
            lower_bound=r.score,
            unexplored=0.0,
            sampled=0.0,
            refused=0.0,
            tied_with_best=r.tied_with_best,
        )
        for r in ranking
    ]
    leaf = [PlanNode(None, 1.0, None, {}, {}, "leaf")]
    plan = PlanNode(
        None,
        1.0,
        rows[0].action if rows else None,
        {r.action: r.estimate for r in rows},
        {r.action: list(leaf) for r in rows},
        "decision",
    )
    playouts = settings.n_depth0 * len(rows)
    return SolveResult(rows, {}, 0, plan, playouts, time.perf_counter() - start, settings)


def solve(
    tables: Tables,
    names: Names,
    state: BattleState,
    key,
    settings: SolveSettings,
    leaf_fn: LeafFn | None = None,
) -> SolveResult:
    """Solve the player's decision at ``state`` (module docstring). ``leaf_fn(states,
    weights, pairs=None) -> list[LeafValue]`` replaces both evaluators: ``states`` are the
    evaluated states (numpy, one row each), ``weights`` their ruling-8 chance weights (0
    for pruned and sampled evaluations), ``pairs`` ``None`` for post-turn leaves and
    ``int32[N, 2]`` for forced (player, opponent) evaluations. With ``precision`` set it is
    called again for every top-up round, with only the states that round tops up and
    ``weights`` their summed reach (forced evaluations included); the values it returns
    are added to the earlier ones. Raises ``UnsupportedAi``
    when the AI is refused at the root and ``HiddenLegality`` when the states of a belief
    node differ in the player's legal actions."""
    start = time.perf_counter()
    if (
        settings.max_depth < 0
        or settings.leaf_floor < 1
        or (settings.precision is not None and not settings.precision > 0)
        or settings.epsilon < 0
        or (settings.budget_playouts is not None and settings.budget_playouts < 1)
        or (settings.budget_seconds is not None and settings.budget_seconds < 0)
    ):
        raise ValueError(f"invalid settings {settings}")
    if settings.max_depth == 0:
        return _stage_a(tables, names, state, key, settings, start)
    root = jax.tree.map(lambda f: np.asarray(f)[None], state)
    if int(root.over[0]) != 0:
        raise ValueError("solve: the battle is already over")
    solver = _Solver(tables, names, root, settings, leaf_fn, start)
    n_root = len(solver._legal(0))
    budget = settings.budget_playouts
    if budget is None:
        budget = settings.n_depth0 * n_root
    result = None
    for depth in range(1, settings.max_depth + 1):
        elapsed = time.perf_counter() - start
        if depth > 1 and settings.budget_seconds is not None and elapsed > settings.budget_seconds:
            break
        result = (depth, *solver.run(depth, jax.random.fold_in(key, depth), budget))
    depth, ranking, dropped, plan, rounds, half, met = result
    seconds = time.perf_counter() - start
    return SolveResult(
        ranking, dropped, depth, plan, solver.playouts, seconds, settings, budget, rounds, half, met
    )
