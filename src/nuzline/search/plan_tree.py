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
"""

from __future__ import annotations

import dataclasses

import numpy as np

from nuzline.data import tables as tables_module
from nuzline.data.tables import Names
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.ids import Status
from nuzline.engine.observe import (
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
from nuzline.jaxcache import code_version, table_version
from nuzline.search.stage_b import PlanNode, SolveResult

__all__ = ["label", "render", "to_json"]

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
    trainer: str = "Opponent",
    max_hp: list[int] | None = None,
) -> str:
    """ASCII sentence for one observation (module docstring), e.g. ``Geodude used Rock
    Throw, crit, you 12/58 HP, foe 71 %``, ``Roark sent out Onix`` after a faint. The
    sides are listed in the order they acted (``order``; the opponent first when unknown)."""
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
    "spikes": "Spikes",
    "toxic_spikes": "Toxic Spikes",
}


def _differences(f: dict, others: list[dict], names: Names, trainer: str) -> list[str]:
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
        ("mud_sport", "Mud Sport"),
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


def _unique(labels: list[str], kinds: list[str]) -> list[str]:
    """Sibling labels made distinct: a kind marker on the unexplored/sampled/refused
    ones, then ``#n`` on any that still collide."""
    out = [
        lab if kind in ("decision", "leaf", "terminal") else f"{lab} [{_KIND_TEXT[kind]}]"
        for lab, kind in zip(labels, kinds, strict=True)
    ]
    seen: dict[str, int] = {}
    for i, lab in enumerate(out):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            out[i] = f"{lab} #{seen[lab]}"
    return out


class _Walk:
    """Shared context of one walk over the plan."""

    def __init__(self, result: SolveResult, names: Names, root, trainer: str):
        self.names, self.root, self.trainer = names, root, trainer
        self.labels = {r.action: r.label for r in result.ranking}
        self.max_hp = None if root is None else np.asarray(root.max_hp)[0].tolist()
        self.root_opp = None if root is None else int(np.asarray(root.active)[1])

    def node_label(self, node: PlanNode, prev_opp: int | None) -> str:
        if node.obs is None:
            return _KIND_TEXT[node.kind]
        return label(
            node.obs, self.names, prev_opp=prev_opp, trainer=self.trainer, max_hp=self.max_hp
        )

    def kids(self, node: PlanNode, prev_opp: int | None, a: int):
        """``[(child, label, opp slot after)]`` of ``node``'s children under ``a``, with
        distinct labels."""
        kids = node.children[a]
        base = [self.node_label(k, prev_opp) for k in kids]
        fields = [None if k.obs is None else decode(k.obs) for k in kids]
        for lab in set(base):
            group = [i for i, b in enumerate(base) if b == lab]
            if len(group) < 2 or any(fields[i] is None for i in group):
                continue
            for i in group:
                more = _differences(
                    fields[i], [fields[j] for j in group if j != i], self.names, self.trainer
                )
                if more:
                    base[i] = ", ".join([base[i], *more])
        labs = _unique(base, [k.kind for k in kids])
        return [
            (k, lab, None if k.obs is None else int(decode(k.obs)["active"][1]))
            for k, lab in zip(kids, labs, strict=True)
        ]

    def action_label(self, node: PlanNode, a: int) -> str:
        f = None if node.obs is None else decode(node.obs)
        return _action_label(f, self.names, a, self.root, self.labels)


def _plain(row: dict) -> dict:
    """``row`` with numpy scalars as Python ones (``json`` refuses ``np.bool_``/``np.int64``)."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}


def _node_json(walk: _Walk, node: PlanNode, lab: str, prev_opp: int | None) -> dict:
    out = {
        "kind": node.kind,
        "prob": float(node.prob),
        "label": lab,
        "observation": None
        if node.obs is None
        else {
            name: (int(v[0]) if v.size == 1 else v.tolist()) for name, v in decode(node.obs).items()
        },
        "action": None if node.action is None else int(node.action),
        "action_label": None if node.action is None else walk.action_label(node, node.action),
        "values": {walk.action_label(node, a): float(v) for a, v in node.values.items()},
        "children": {},
    }
    for a in node.children:
        out["children"][walk.action_label(node, a)] = [
            _node_json(walk, k, klab, opp) for k, klab, opp in walk.kids(node, prev_opp, a)
        ]
    return out


def to_json(result: SolveResult, names: Names, *, root=None, trainer: str = "Opponent") -> dict:
    """JSON-ready dict of the solve: settings, depth, ranking, dropped actions, playouts,
    seconds, table and code versions and the plan tree (module docstring; ``root`` is the
    solved state, optional)."""
    walk = _Walk(result, names, root, trainer)
    plan = _node_json(walk, result.plan, "root", walk.root_opp)
    return {
        "settings": dataclasses.asdict(result.settings),
        "depth": result.depth,
        "ranking": [_plain(dataclasses.asdict(r)) for r in result.ranking],
        "dropped": [
            {"action": int(a), "label": walk.labels.get(a, f"action {a}"), "reason": why}
            for a, why in result.dropped.items()
        ],
        "playouts": int(result.playouts),
        "seconds": float(result.seconds),
        "table_version": table_version(tables_module.DEFAULT),
        "code_version": code_version(),
        "plan": plan,
    }


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _outcome_text(node: PlanNode, walk: _Walk) -> str:
    if node.kind == "decision":
        return walk.action_label(node, node.action) if node.action is not None else "?"
    return f"({_KIND_TEXT[node.kind]})"


def _lines(walk: _Walk, node: PlanNode, a: int, prev_opp: int | None, depth: int, indent: int):
    pad = " " * indent
    for kid, lab, opp in walk.kids(node, prev_opp, a):
        yield f"{pad}if {lab}  [{_pct(kid.prob).strip()}] -> {_outcome_text(kid, walk)}"
        if depth > 1 and kid.kind == "decision" and kid.action in kid.children:
            yield from _lines(walk, kid, kid.action, opp, depth - 1, indent + 3)


def render(
    result: SolveResult, names: Names, show_depth: int, *, root=None, trainer: str = "Opponent"
) -> list[str]:
    """The spec §5 layout: the ranking, then the plan under the best action to
    ``show_depth`` levels of observations, then the budget footer."""
    walk = _Walk(result, names, root, trainer)
    s = result.settings
    out = [
        (
            f"depth reached {result.depth} of {s.max_depth}; {result.playouts} playouts, "
            f"{result.seconds:.1f} s"
        ),
        (
            f"     #  {'action':<24}  {'P(win, 0 faints) [95% CI]':<27}  {'lower':>7}  "
            f"{'unexplored':>10}  {'sampled':>8}  {'refused':>8}"
        ),
    ]
    for i, r in enumerate(result.ranking, 1):
        mark = "*" if r.tied_with_best else " "
        odds = f"{_pct(r.estimate)} [{100 * r.ci_low:4.1f}, {100 * r.ci_high:5.1f}]"
        out.append(
            f"  {mark} {i:2d}  {r.label:<24}  {odds:<27}  {_pct(r.lower_bound):>7}  "
            f"{_pct(r.unexplored):>10}  {_pct(r.sampled):>8}  {_pct(r.refused):>8}"
        )
        if i == 1 and show_depth > 0 and r.action in result.plan.children:
            out.extend(_lines(walk, result.plan, r.action, walk.root_opp, show_depth, 6))
    if any(r.tied_with_best for r in result.ranking):
        out.append("* the interval overlaps the leader's: a statistical tie, not reordered")
    for why in result.dropped.values():
        out.append(f"dropped: {why}")
    budget = s.budget_playouts
    if budget is None:
        budget = s.n_depth0 * (len(result.ranking) + len(result.dropped))
    out.append("the plan is shown under the best action only")
    out.append(
        f"leaf budget {budget} playouts per pass (floor {s.leaf_floor} per half); leaf "
        "estimates come from playouts that see the full state (the plan never does)"
    )
    return out
