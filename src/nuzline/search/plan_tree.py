"""The plan tree as text and JSON (Stage B spec §5; plan Task 6).

``label`` turns one observation (``engine.observe``'s ``int32[OBS_SIZE]`` bytes) into an
ASCII sentence: what each side visibly did this turn (the events ``step`` records:
``act_kind``/``act_id`` and the effect-gated flags), the player's exact HP (the player's
healthbox), the opponent's HP as its bar percentage (``hp_pixels`` / 48: the enemy
healthbox never shows numbers, healthbox.c:633-640), the actives' status, a fainted
opponent and the replacement it sends in. ``to_json`` and ``render`` walk
``stage_b.SolveResult.plan``: a decision node holds one recommended action and its
children are the observations that follow it.

What a label cannot know from the observation alone is passed in, all optional: the
player's maximum HP per slot (``max_hp``; without it HP prints as ``12 HP``), the opponent
active slot before the turn (``prev_opp``; without it a replacement after a faint is not
named), the trainer's name and, for action names below the root, the root state
(``root``; without it a move prints as ``move 2``). The plan never depends on any of them.

A node not expanded into observations (``obs`` is ``None``: a pruned opponent action, a
sampled pair, an AI refused below the root) is labelled by the opponent action it stands
for (``PlanNode.opp_label``, e.g. ``Geodude uses Rock Throw``); its kind shows once, as
the line's outcome in ``render`` and as ``kind`` in ``to_json``. Without an opponent action:
a depth-0 (Stage A) leaf stands for every foe reply, valued by Stage A's playouts (whose
foe is the AI), and a refused node for a state whose AI is not modelled. ``render`` shows
no plan at depth 0 (its footer says so); ``to_json`` keeps the leaves.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from nuzline.data.tables import Names
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.ids import Status
from nuzline.engine.observe import (
    ACT_DISABLED,
    ACT_ITEM,
    ACT_MOVE,
    ACT_SWITCH,
    ACT_TAUNTED,
    HP_BAR_PIXELS,
    OBS_FIELDS,
    OBS_SIZE,
)
from nuzline.engine.state import SLOTS
from nuzline.engine.step import WINNER_PLAYER
from nuzline.jaxcache import code_version
from nuzline.search.stage_b import PlanNode, SolveResult

__all__ = ["distribution_line", "label", "render", "to_json"]

_FIELDS = dict(OBS_FIELDS)
# (flag, text) in the order a message sequence reads; ``hits`` and ``crit`` are special.
_FLAG_TEXT = (
    ("missed", "missed"),
    ("no_target", "no target"),
    ("full_para", "fully paralyzed"),
    ("flinched", "flinched"),
    ("asleep", "is fast asleep"),
    ("woke", "woke up"),
    ("frozen", "is frozen solid"),
    ("thawed", "thawed out"),
    ("thawed_by_hit", "thawed by the hit"),
    ("confused", "is confused"),
    ("snapped_out", "snapped out of confusion"),
    ("self_hit", "hurt itself in confusion"),
    ("contact_ability", "ability triggered"),
)
_STATUS_TEXT = {
    Status.SLEEP: "SLP",
    Status.POISON: "PSN",
    Status.BURN: "BRN",
    Status.FREEZE: "FRZ",
    Status.PARALYSIS: "PAR",
    Status.TOXIC: "TOX",
}
_KIND_TEXT = {
    "pruned": "unexplored (below epsilon)",
    "sampled": "sampled (too many outcomes to enumerate)",
    "refused": "refused",
    "failed": "over the death budget: play on for the win, not planned",
    "terminal": "battle over",
    "leaf": "leaf",
    "decision": "decision",
}


def decode(obs: bytes) -> dict[str, np.ndarray]:
    """The observation's fields by name (flat ``int32`` arrays; ``OBS_FIELDS`` order)."""
    vec = np.frombuffer(obs, np.int32)
    if vec.shape != (OBS_SIZE,):
        raise ValueError(f"observation has {vec.size} entries, want {OBS_SIZE}")
    return {name: vec[sl] for name, sl in OBS_FIELDS}


def _species(names: Names, f: dict, side: int, slot: int) -> str:
    return names.species_names[int(f["species"][side * SLOTS + slot])]


def _status(f: dict, side: int) -> str:
    code = int(f["status"][side * SLOTS + int(f["active"][side])])
    return _STATUS_TEXT.get(code, "") if code else ""


def _side_text(f: dict, names: Names, side: int, who: str, trainer: str) -> str:
    """What ``side`` visibly did, e.g. ``Geodude used Rock Throw, crit``; empty when its
    action never showed."""
    kind, act = int(f["act_kind"][side]), int(f["act_id"][side])
    details = []
    for flag, text in _FLAG_TEXT:
        if int(f[flag][side]):
            details.append(text)
    hits = int(f["hits"][side])
    if hits > 1:
        details.append(f"hit {hits} times")
    if np.any(f["crit"].reshape(2, -1)[side]):
        details.append("crit")
    if kind == ACT_MOVE:
        head = f"{who} used {names.move_names[act]}"
    elif kind == ACT_TAUNTED:
        head = f"{who} can't use {names.move_names[act]} after the taunt"
    elif kind == ACT_DISABLED:
        head = f"{who}'s {names.move_names[act]} is disabled"
    elif kind == ACT_SWITCH:
        slot_name = _species(names, f, side, act)
        head = f"{trainer} sent out {slot_name}" if side else f"you switched to {slot_name}"
    elif kind == ACT_ITEM:
        head = f"{trainer if side else 'you'} used {names.item_names[act]}"
    elif details:
        head = who
    else:
        return ""
    return ", ".join([head, *details])


def label(
    obs: bytes,
    names: Names,
    *,
    prev_opp: int | None = None,
    prev_mine: int | None = None,
    trainer: str = "Opponent",
    max_hp: list[int] | None = None,
    budget: tuple[int, int] | None = None,
) -> str:
    """ASCII sentence for one observation (module docstring), e.g. ``Geodude used Rock
    Throw, crit, you 12/58 HP, foe 71 %``, ``Roark sent out Onix`` after a faint. The
    sides are listed in the order they acted (``order``; the opponent first when unknown).

    A player active at 0 HP reads ``Monferno fainted``; with ``budget = (k, n)`` (the
    deaths allowed, and the fainted slots the root already had that are not deaths of this
    battle) it adds ``(death d of k allowed)`` or, past the budget, ``(death d, over the
    k allowed)``, ``d`` counting the fainted slots beyond ``n``. ``prev_mine`` (the
    player's active slot before the turn) names ``you sent in X`` when the active changed
    without a switch action: a replacement."""
    f = decode(obs)
    opp_slot, my_slot = int(f["active"][1]), int(f["active"][0])
    opp = _species(names, f, 1, opp_slot)
    sides = [(1, opp), (0, "you")]
    if int(f["order"][0]) == 0:
        sides.reverse()
    parts = [t for s, who in sides if (t := _side_text(f, names, s, who, trainer))]
    if prev_opp is not None and prev_opp != opp_slot:
        if int(f["opp_pixels"][prev_opp]) == 0:
            parts.append(f"{_species(names, f, 1, prev_opp)} fainted")
        if int(f["act_kind"][1]) != ACT_SWITCH:
            parts.append(f"{trainer} sent out {opp}")
    hp = int(f["player_hp"][my_slot])
    if hp == 0:
        text = f"{_species(names, f, 0, my_slot)} fainted"
        if budget is not None:
            k, base = budget
            gone = sum(
                1
                for slot in range(SLOTS)
                if int(f["species"][slot]) and int(f["player_hp"][slot]) == 0
            )
            d = max(gone - base, 1)
            text += (
                f" (death {d} of {k} allowed)" if d <= k else f" (death {d}, over the {k} allowed)"
            )
        parts.append(text)
    elif prev_mine is not None and prev_mine != my_slot and int(f["act_kind"][0]) != ACT_SWITCH:
        parts.append(f"you sent in {_species(names, f, 0, my_slot)}")
    total = f"/{max_hp[my_slot]}" if max_hp is not None else ""
    mine = f"you {hp}{total} HP"
    if s := _status(f, 0):
        mine += f" {s}"
    pct = int(f["opp_pixels"][opp_slot]) * 100 // HP_BAR_PIXELS
    foe = f"foe {pct} %"
    if s := _status(f, 1):
        foe += f" {s}"
    parts += [mine, foe]
    if int(f["over"][0]):
        parts.append("you win" if int(f["winner"][0]) == WINNER_PLAYER else "you lose")
    return ", ".join(parts)


_STAGE_NAMES = ("HP", "Atk", "Def", "SpA", "SpD", "Spe", "Acc", "Eva")
_WEATHER_TEXT = {1: "sun", 2: "rain", 3: "sandstorm", 4: "hail", 5: "fog"}
# Per-side conditions that show as present or absent (``engine.observe._NONZERO_SIDE``).
_COND_TEXT = {
    "confusion": "confused",
    "taunt": "taunted",
    "trapped": "trapped",
    "leech_seed": "seeded",
    "substitute_hp": "behind a substitute",
    "water_sport": "Water Sport",
    "stealth_rock": "Stealth Rock",
    "reflect": "Reflect",
    "light_screen": "Light Screen",
    "safeguard": "Safeguard",
    "mist": "Mist",
    "lucky_chant": "Lucky Chant",
    "tailwind": "Tailwind",
    "foresight": "identified",
    "curled": "curled up",
    "bide_turns": "storing energy",
    "mud_sport": "Mud Sport",
    "aqua_ring": "Aqua Ring",
    "spikes": "Spikes",
    "toxic_spikes": "Toxic Spikes",
}


def _differences(f: dict, others: list[dict], names: Names) -> list[str]:
    """Readable text for every observation field of ``f`` that some sibling in ``others``
    shows differently (the stat stages, conditions, items, other slots' HP, ...)."""
    who = ("you", "foe")
    out = []

    def differs(name: str, i: int) -> bool:
        return any(int(o[name].reshape(-1)[i]) != int(f[name].reshape(-1)[i]) for o in others)

    for side in (0, 1):
        for st in range(8):
            if differs("stages", side * 8 + st):
                out.append(f"{who[side]} {_STAGE_NAMES[st]} {int(f['stages'][side * 8 + st]):+d}")
        for name, text in _COND_TEXT.items():
            if differs(name, side):
                on = int(f[name][side]) != 0
                out.append(f"{who[side]} {text}" if on else f"{who[side]} no {text}")
        for name, text in (("disabled_move", "{} disabled"), ("locked_move", "locked into {}")):
            if differs(name, side):
                move = int(f[name][side])
                if move:
                    out.append(f"{who[side]} {text.format(names.move_names[move])}")
                else:
                    out.append(f"{who[side]} no {name.replace('_', ' ')}")
    for slot in range(SLOTS):
        for side in (0, 1):
            i = side * SLOTS + slot
            mon = f"{who[side]} {_species(names, f, side, slot)}"
            if differs("item_consumed", i) and int(f["item_consumed"][i]):
                out.append(f"{mon} item used")
            if differs("item", i):
                out.append(f"{mon} item {names.item_names[int(f['item'][i])]}")
            if differs("status", i) and slot != int(f["active"][side]):
                out.append(f"{mon} {_STATUS_TEXT.get(int(f['status'][i]), 'healthy')}")
        if slot != int(f["active"][0]) and differs("player_hp", slot):
            out.append(f"{_species(names, f, 0, slot)} {int(f['player_hp'][slot])} HP")
        if slot != int(f["active"][1]) and differs("opp_pixels", slot):
            pct = int(f["opp_pixels"][slot]) * 100 // HP_BAR_PIXELS
            out.append(f"{_species(names, f, 1, slot)} {pct} %")
    for name, text in (
        ("trick_room", "Trick Room"),
        ("gravity", "Gravity"),
    ):
        if differs(name, 0):
            out.append(text if int(f[name][0]) else f"no {text}")
    if differs("weather", 0):
        out.append(_WEATHER_TEXT.get(int(f["weather"][0]), "clear weather"))
    if differs("pursuit", 0):
        out.append("Pursuit hit first" if int(f["pursuit"][0]) else "no Pursuit")
    if differs("player_pp", 0) or any(differs("player_pp", i) for i in range(1, SLOTS * 4)):
        out.append("PP " + "/".join(str(int(v)) for v in f["player_pp"]))
    return out


def _action_label(f: dict | None, names: Names, a: int, root, labels: dict[int, str]) -> str:
    """The label of player action ``a`` at a node: from the ranking at the root
    (``labels``), else from the node's observation (switch targets, Struggle from the PP
    it shows) and the root state's moves."""
    if f is None:
        return labels.get(a, f"action {a}")
    mon = int(f["active"][0])
    if a >= MOVE_ACTIONS:
        return f"Switch -> {_species(names, f, 0, a - MOVE_ACTIONS)}"
    if root is not None:
        moves = np.asarray(root.moves)[0, mon]
        pp = f["player_pp"][mon * 4 : mon * 4 + 4]
        if not np.any((moves > 0) & (pp > 0)):
            return "Struggle"
        return names.move_names[int(moves[a])]
    return f"move {a + 1}"


def _unique(labels: list[str], kinds: list[str], observed: list[bool], marks: bool) -> list[str]:
    """Sibling labels made distinct. With ``marks`` (JSON, where a label stands alone) a
    kind marker goes on the unexplored/sampled/refused observations (a refused group can
    share its observation with a kept one), then ``#n`` on any label that still
    collides. Without (``render``, whose line prints the kind as its outcome) ``#n`` goes
    only on a (label, kind) that collides."""
    out = [
        f"{lab} [{_KIND_TEXT[kind]}]"
        if marks and seen and kind not in ("decision", "leaf", "terminal")
        else lab
        for lab, kind, seen in zip(labels, kinds, observed, strict=True)
    ]
    keys = out if marks else list(zip(out, kinds, strict=True))
    count: dict = {}
    for i, key in enumerate(keys):
        count[key] = count.get(key, 0) + 1
        if count[key] > 1:
            out[i] = f"{out[i]} #{count[key]}"
    return out


class _Walk:
    """Shared context of one walk over the plan."""

    def __init__(self, result: SolveResult, names: Names, root, trainer: str):
        self.names, self.root, self.trainer = names, root, trainer
        self.labels = {r.action: r.label for r in result.ranking}
        self.max_hp = None if root is None else np.asarray(root.max_hp)[0].tolist()
        self.root_opp = None if root is None else int(np.asarray(root.active)[1])
        self.root_mine = None if root is None else int(np.asarray(root.active)[0])
        self.budget = None
        if root is not None:  # fainted slots that are not deaths of this battle
            down = (np.asarray(root.species)[0] != 0) & (np.asarray(root.hp)[0] == 0)
            self.budget = (result.deaths_allowed, int(down.sum()) - int(root.player_faints))

    def node_label(self, node: PlanNode, prev_opp: int | None, prev_mine: int | None) -> str:
        if node.obs is None:
            if node.opp_label:
                return node.opp_label
            if node.kind == "leaf":
                return "any foe reply (Stage A playouts)"
            if node.kind == "refused":
                return "the foe's AI is not modelled here"
            raise ValueError(f"a {node.kind} node with no observation and no opponent action")
        return label(
            node.obs,
            self.names,
            prev_opp=prev_opp,
            prev_mine=prev_mine,
            trainer=self.trainer,
            max_hp=self.max_hp,
            budget=self.budget,
        )

    def kids(
        self,
        node: PlanNode,
        prev_opp: int | None,
        a: int,
        marks: bool = True,
        prev_mine: int | None = None,
    ):
        """``[(child, label, opp slot after)]`` of ``node``'s children under ``a``, with
        distinct labels (``_unique``); ``prev_opp``/``prev_mine``: the active slots before."""
        kids = node.children[a]
        base = [self.node_label(k, prev_opp, prev_mine) for k in kids]
        fields = [None if k.obs is None else decode(k.obs) for k in kids]
        for lab in set(base):
            group = [i for i, b in enumerate(base) if b == lab]
            if len(group) < 2 or any(fields[i] is None for i in group):
                continue
            for i in group:
                more = _differences(fields[i], [fields[j] for j in group if j != i], self.names)
                if more:
                    base[i] = ", ".join([base[i], *more])
        labs = _unique(base, [k.kind for k in kids], [k.obs is not None for k in kids], marks)
        return [
            (k, lab, None if f is None else int(f["active"][1]))
            for k, lab, f in zip(kids, labs, fields, strict=True)
        ]

    def action_label(self, node: PlanNode, a: int) -> str:
        f = None if node.obs is None else decode(node.obs)
        return _action_label(f, self.names, a, self.root, self.labels)


def _plain(value):
    """``value`` (a ``dataclasses.asdict`` row) with numpy scalars as Python ones (``json``
    refuses ``np.bool_``/``np.int64``) and tuples as lists (what a JSON round trip gives)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _active(node: PlanNode) -> int | None:
    """The player's active slot in ``node``'s observation (``None`` without one)."""
    return None if node.obs is None else int(decode(node.obs)["active"][0])


def _node_json(
    walk: _Walk,
    node: PlanNode,
    lab: str,
    prev_opp: int | None,
    prev_mine: int | None,
    depth: int,
) -> dict:
    """``node`` as JSON; ``depth``: the turns played to reach it (the root is 0)."""
    out = {
        "kind": node.kind,
        "depth": depth,
        "prob": float(node.prob),
        "label": lab,
        "observation": None
        if node.obs is None
        else {
            name: (int(v[0]) if v.size == 1 else v.tolist()) for name, v in decode(node.obs).items()
        },
        "opp_action": node.opp_action,
        "replacement": node.replacement,
        "action": None if node.action is None else int(node.action),
        "action_label": None if node.action is None else walk.action_label(node, node.action),
        "values": {walk.action_label(node, a): float(v) for a, v in node.values.items()},
        "children": {},
    }
    for a in node.children:
        out["children"][walk.action_label(node, a)] = [
            _node_json(walk, k, klab, opp, _active(k), depth + 1)
            for k, klab, opp in walk.kids(node, prev_opp, a, prev_mine=prev_mine)
        ]
    return out


def to_json(
    result: SolveResult,
    names: Names,
    *,
    root=None,
    trainer: str = "Opponent",
    table_version: str | None = None,
) -> dict:
    """JSON-ready dict of the solve: settings, depth, leaf budget, ranking, dropped
    actions, playouts, seconds, table and code versions and the plan tree (module
    docstring; ``root`` is the solved state, optional). ``table_version``: the
    ``jaxcache.table_version`` of the tables the solve used (``None``: not known)."""
    walk = _Walk(result, names, root, trainer)
    plan = _node_json(walk, result.plan, "root", walk.root_opp, walk.root_mine, 0)
    ranking = []
    for r in result.ranking:
        row = _plain(dataclasses.asdict(r))
        row["distribution"] = None if r.distribution is None else r.distribution.as_dict()
        ranking.append(row)
    return {
        "settings": dataclasses.asdict(result.settings),
        "depth": result.depth,
        "leaf_budget": result.leaf_budget,
        "rounds": int(result.rounds),
        "half_width_a": None if result.half_width_a is None else float(result.half_width_a),
        "precision_met": result.precision_met,
        "deaths_allowed": result.deaths_allowed,
        "protected": list(result.protected),
        "ranking": ranking,
        "dropped": [
            {"action": int(a), "label": walk.labels.get(a, f"action {a}"), "reason": why}
            for a, why in result.dropped.items()
        ],
        "playouts": int(result.playouts),
        "seconds": float(result.seconds),
        "table_version": table_version,
        "code_version": code_version(),
        "plan": plan,
    }


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def distribution_line(dist) -> str:
    """``deaths under the leader: 0: a %  1: b % ...`` from a ``DeathDistribution`` (3 and
    4+ deaths folded into ``3+``)."""
    p = dist.probs
    cells = (
        ("0", p[0]),
        ("1", p[1]),
        ("2", p[2]),
        ("3+", p[3] + p[4]),
        ("whiteout", p[6]),
        ("unfinished", p[7]),
        ("protected lost", p[5]),
    )
    return "deaths under the leader: " + "  ".join(f"{n}: {100 * v:.1f} %" for n, v in cells)


def _precision_line(result: SolveResult) -> str:
    """The precision target of the final pass: met or not, the half-A leader's half-width
    and the top-up rounds run."""
    head = f"precision target {100 * result.settings.precision:g} pp"
    rounds = f"after {result.rounds} top-up round{'' if result.rounds == 1 else 's'}"
    if result.half_width_a is None:
        return f"{head}: no ranked action to measure, {rounds}"
    verdict = "met" if result.precision_met else "not met"
    width = f"{100 * result.half_width_a:.2f} pp"
    return f"{head}: {verdict} (leader's half-A half-width {width}) {rounds}"


def _outcome_text(node: PlanNode, walk: _Walk) -> str:
    if node.kind == "decision":
        if node.action is None:
            return "?"
        text = walk.action_label(node, node.action)
        return text.replace("Switch -> ", "send in ", 1) if node.replacement else text
    return f"({_KIND_TEXT[node.kind]})"


def _lines(
    walk: _Walk,
    node: PlanNode,
    a: int,
    prev_opp: int | None,
    prev_mine: int | None,
    depth: int,
    indent: int,
):
    pad = " " * indent
    for kid, lab, opp in walk.kids(node, prev_opp, a, marks=False, prev_mine=prev_mine):
        yield f"{pad}if {lab}  [{_pct(kid.prob).strip()}] -> {_outcome_text(kid, walk)}"
        if depth > 1 and kid.kind == "decision" and kid.action in kid.children:
            yield from _lines(walk, kid, kid.action, opp, _active(kid), depth - 1, indent + 3)


def render(
    result: SolveResult, names: Names, show_depth: int, *, root=None, trainer: str = "Opponent"
) -> list[str]:
    """The spec §5 layout: the ranking, then the plan under the best action to
    ``show_depth`` levels of observations, then the budget footer."""
    walk = _Walk(result, names, root, trainer)
    s = result.settings
    k = result.deaths_allowed
    win = "P(win, 0 faints)" if k == 0 else f"P(win, <={k} deaths)"
    out = [
        (
            f"depth reached {result.depth} of {s.max_depth}; {result.playouts} playouts, "
            f"{result.seconds:.1f} s"
        ),
        (
            f"     #  {'action':<24}  {win + ' [95% CI]':<27}  {'lower':>7}  "
            f"{'unexplored':>10}  {'sampled (nodes)':>15}  {'refused':>8}  {'over budget':>11}"
        ),
    ]
    for i, r in enumerate(result.ranking, 1):
        mark = "*" if r.tied_with_best else " "
        odds = f"{_pct(r.estimate)} [{100 * r.ci_low:4.1f}, {100 * r.ci_high:5.1f}]"
        sampled = f"{_pct(r.sampled)} ({r.sampled_nodes})"
        out.append(
            f"  {mark} {i:2d}  {r.label:<24}  {odds:<27}  {_pct(r.lower_bound):>7}  "
            f"{_pct(r.unexplored):>10}  {sampled:>15}  {_pct(r.refused):>8}  {_pct(r.failed):>11}"
        )
        staged = result.leaf_budget is not None  # depth 0 (Stage A) has no plan to show
        if i == 1 and staged and show_depth > 0 and r.action in result.plan.children:
            out.extend(
                _lines(walk, result.plan, r.action, walk.root_opp, walk.root_mine, show_depth, 6)
            )
    if result.ranking and result.ranking[0].distribution is not None:
        out.append(distribution_line(result.ranking[0].distribution))
    if result.protected:
        who = ", ".join(
            f"slot {slot}"
            if root is None
            else names.species_names[int(np.asarray(root.species)[0, slot])]
            for slot in result.protected
        )
        out.append(f"protected: {who} (a line where one faints fails)")
    if any(r.tied_with_best for r in result.ranking):
        out.append("* the interval overlaps the leader's: a statistical tie, not reordered")
    for why in result.dropped.values():
        out.append(f"dropped: {why}")
    if result.leaf_budget is None:
        out.append(
            f"Stage A (depth 0): {s.n_depth0} playouts per action, no plan tree; the "
            "estimates come from playouts that see the full state"
        )
        return out
    out.append("the plan is shown under the best action only")
    if s.precision is not None:
        out.append(_precision_line(result))
    out.append(
        f"leaf budget {result.leaf_budget} playouts per pass (floor {s.leaf_floor} per half); "
        "leaf estimates come from playouts that see the full state (the plan never does)"
    )
    return out
