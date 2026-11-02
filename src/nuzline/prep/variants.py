"""Member variants for the prep search: the changes one Pokemon can take into the fight
(moves, held item, entry HP, declared prestatus), what they cost, the HP regions that matter
and the conservative dominance prune (plan 4c rulings 4-8; spec §6)."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from nuzline.ai.commands import SIM_ROLL_MAX, ai_damage, is_damaging
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables, load_names, load_tables
from nuzline.engine.ids import AbilityId, EffectId, MoveFlag, Status, Type, Weather
from nuzline.engine.state import make_state
from nuzline.engine.stats import calc_stats
from nuzline.prep.problem import PrepProblem, Resources

__all__ = [
    "AI_DEFENDER_HP_CHECKS",
    "AI_HP_READERS",
    "DEFAULT_WEIGHTS",
    "LEVEL_UP",
    "RELEARNER",
    "STATUS_BLOCKING_ABILITIES",
    "Cost",
    "Member",
    "MoveChange",
    "ai_kill_damage",
    "base_member",
    "designated_change",
    "hp_cuts",
    "hp_regions",
    "max_hp_of",
    "member_cost",
    "member_spec",
    "move_changes",
    "prestatus_options",
    "prune",
    "single_changes",
]

RELEARNER = 0  # MoveChange.source: the Move Relearner (1 Heart Scale)
LEVEL_UP = -1  # MoveChange.source: learned by levelling (Rare Candies, already paid)

DEFAULT_WEIGHTS: dict[str, float] = {
    "candy": 0.0,
    "heart_scale": 3.0,
    "tm": 10.0,
    "hm": 5.0,
    "held": 1.0,
}

# The AI's checks of the player's HP percent, as (kind, percent): ``ai/policy.py`` calls
# ``if_hp_percent_greater_than(ctx, _D, 70)`` (the defender is the player's active).
AI_DEFENDER_HP_CHECKS: tuple[tuple[str, int], ...] = (("greater", 70),)

# Every function in ``src/nuzline/ai/*.py`` that indexes an ``hp`` array at the Task 4 commit,
# and how ``hp_cuts`` covers it: ``percent`` by ``AI_DEFENDER_HP_CHECKS``, ``kill`` by the
# kill damages, ``alive`` (HP > 0 only) and ``opponent`` (the trainer's side) need no cut.
AI_HP_READERS: dict[str, str] = {
    "_hp_percent": "percent",
    "if_current_move_kills": "kill",
    "count_alive_party_battlers": "alive",
    "_use_item": "opponent",
    "_post_ko": "opponent",
    "_layout": "opponent",
}

STATUS_BLOCKING_ABILITIES = {
    "Limber": Status.PARALYSIS,
    "Immunity": Status.POISON,
    "Water Veil": Status.BURN,
}
_STATUS_BLOCKING_TYPES = {
    Status.POISON: (Type.POISON, Type.STEEL),
    Status.BURN: (Type.FIRE,),
    Status.PARALYSIS: (),
}
_PINCH_ABILITIES = (AbilityId.BLAZE, AbilityId.TORRENT, AbilityId.OVERGROW)


@dataclass(frozen=True)
class MoveChange:
    """Move ``move`` goes into slot ``slot``; ``source`` is ``RELEARNER``, ``LEVEL_UP`` or the
    TM/HM item id."""

    slot: int
    move: int
    source: int


@dataclass(frozen=True)
class Member:
    """One Pokemon as brought: ``candidate`` (index into ``problem.candidates``), the move
    changes on top of its saved moves, ``item`` (0 keeps its own, -1 takes it off, else a bag
    item), entry ``hp`` (None = full) and prestatus."""

    candidate: int
    moves: tuple[MoveChange, ...] = ()
    item: int = 0
    hp: int | None = None
    status: int = 0


@dataclass(frozen=True)
class Cost:
    """What a member spends. ``tms`` and ``held`` are sorted item ids, one per use."""

    candies: int = 0
    heart_scales: int = 0
    tms: tuple[int, ...] = ()
    hm_moves: int = 0
    held: tuple[int, ...] = ()

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.candies + other.candies,
            self.heart_scales + other.heart_scales,
            tuple(sorted(self.tms + other.tms)),
            self.hm_moves + other.hm_moves,
            tuple(sorted(self.held + other.held)),
        )

    def fits(self, resources: Resources) -> bool:
        """Whether the bag covers the Heart Scales, each TM and each held item (HM moves are
        reusable and are only offered for HMs in the bag)."""
        if self.heart_scales > resources.heart_scales:
            return False
        for need, have in ((self.tms, resources.tms), (self.held, resources.held)):
            stock = dict(have)
            if any(n > stock.get(item, 0) for item, n in Counter(need).items()):
                return False
        return True

    def weighted(self, weights: Mapping[str, float]) -> float:
        """The user-weighted cost over the keys ``candy``, ``heart_scale``, ``tm``, ``hm``,
        ``held``."""
        return (
            weights["candy"] * self.candies
            + weights["heart_scale"] * self.heart_scales
            + weights["tm"] * len(self.tms)
            + weights["hm"] * self.hm_moves
            + weights["held"] * len(self.held)
        )


def _no_larger(a: Cost, b: Cost) -> bool:
    """Whether ``a`` is at most ``b`` in every component (a multiset test for items)."""
    return (
        a.candies <= b.candies
        and a.heart_scales <= b.heart_scales
        and a.hm_moves <= b.hm_moves
        and not Counter(a.tms) - Counter(b.tms)
        and not Counter(a.held) - Counter(b.held)
    )


def _pad(moves: Sequence[int]) -> tuple[int, int, int, int]:
    out = list(moves)[:4]
    return (*out, *([0] * (4 - len(out))))  # type: ignore[return-value]


def base_member(problem: PrepProblem, candidate: int) -> Member:
    """The candidate with the moves levelling teaches (``problem.levelled``) filling its
    empty slots in learn-level order as ``LEVEL_UP`` changes (ruling 4)."""
    spec = problem.candidates[candidate].spec
    empty = [i for i, m in enumerate(spec.moves) if m == 0]
    learned = problem.levelled[candidate]
    changes = tuple(
        MoveChange(slot, move, LEVEL_UP) for slot, (_, move) in zip(empty, learned, strict=False)
    )
    return Member(candidate, changes)


@lru_cache(maxsize=1)
def _default_tables() -> Tables:
    return load_tables()


def _move_pp(tables: Tables | None, move: int) -> int:
    return int(np.asarray((tables or _default_tables()).move_pp)[move]) if move else 0


def member_spec(problem: PrepProblem, member: Member, tables: Tables | None = None) -> PokemonSpec:
    """The candidate's spec with each change's move in its slot at that move's table PP, the
    member's item (0 keeps its own, -1 takes it off), entry HP and status (counter 0).
    ``tables`` defaults to the shipped tables (only ``move_pp`` is read)."""
    spec = problem.candidates[member.candidate].spec
    moves = list(spec.moves)
    pp = list(spec.pp) if spec.pp is not None else [_move_pp(tables, m) for m in moves]
    for change in member.moves:
        moves[change.slot] = change.move
        pp[change.slot] = _move_pp(tables, change.move)
    item = spec.item if member.item == 0 else max(member.item, 0)
    return dataclasses.replace(
        spec,
        moves=_pad(moves),
        pp=_pad(pp),
        item=item,
        current_hp=member.hp,
        status=member.status,
        status_counter=0,
    )


def _change_cost(problem: PrepProblem, change: MoveChange) -> Cost:
    if change.source == LEVEL_UP:
        return Cost()
    if change.source == RELEARNER:
        return Cost(heart_scales=1)
    if change.source in problem.resources.hms:
        return Cost(hm_moves=1)
    return Cost(tms=(change.source,))


def member_cost(problem: PrepProblem, member: Member) -> Cost:
    """What the member spends: the candidate's Rare Candies, each change's source (levelled 0,
    Move Relearner 1 Heart Scale, a TM its item, an HM one reusable use) and a bag item
    (taking the item off is free)."""
    cost = Cost(candies=problem.candidates[member.candidate].candies)
    for change in member.moves:
        cost = cost + _change_cost(problem, change)
    if member.item > 0:
        cost = cost + Cost(held=(member.item,))
    return cost


# ---------------------------------------------------------------- entry HP and prestatus


def max_hp_of(tables: Tables, spec: PokemonSpec) -> int:
    """The spec's maximum HP: its stored stats, else ``calc_stats`` from the tables."""
    if spec.stats is not None:
        return int(spec.stats[0])
    stats = calc_stats(
        np.asarray(tables.species_base_stats)[spec.species],
        np.asarray(spec.ivs),
        np.asarray(spec.evs),
        spec.level,
        np.asarray(tables.nature_mods)[spec.nature],
        species=spec.species,
    )
    return int(stats[0])


def ai_kill_damage(
    tables: Tables,
    spec: PokemonSpec,
    opp: Sequence[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
) -> tuple[int, ...]:
    """The AI's kill-check damages against ``spec`` (ruling 5 b): for each opponent slot ``j``
    and each damaging move of it, ``ai_damage`` at ``SIM_ROLL_MAX`` on the turn-0 matchup
    with that opponent leading; distinct, ascending, zeros dropped."""
    out: set[int] = set()
    for j, mon in enumerate(opp):
        state = make_state(tables, [spec], list(opp), start_condition, opp_items, leads=(0, j))
        for move in mon.moves:
            if move and bool(is_damaging(tables, move)):
                out.add(int(ai_damage(tables, state, move, SIM_ROLL_MAX)))
    return tuple(sorted(d for d in out if d > 0))


def hp_cuts(
    tables: Tables, spec: PokemonSpec, max_hp: int, kill_damage: Sequence[int] = ()
) -> tuple[int, ...]:
    """The region tops below ``max_hp`` (ruling 5): the AI's percent checks
    (``AI_DEFENDER_HP_CHECKS``), each kill damage in ``1..max_hp - 1`` and the pinch cut
    ``max_hp // 3`` for Blaze, Torrent and Overgrow. Unsorted, may repeat."""
    cuts = []
    for kind, percent in AI_DEFENDER_HP_CHECKS:
        if kind == "greater":
            cuts.append(((percent + 1) * max_hp - 1) // 100)
        else:
            cuts.append((percent * max_hp - 1) // 100)
    cuts.extend(d for d in kill_damage if 1 <= d < max_hp)
    ability = int(np.asarray(tables.species_abilities)[spec.species, spec.ability_slot])
    if ability in _PINCH_ABILITIES:
        cuts.append(max_hp // 3)
    return tuple(c for c in cuts if 1 <= c < max_hp)


def hp_regions(
    tables: Tables, spec: PokemonSpec, max_hp: int, kill_damage: Sequence[int] = ()
) -> tuple[int, ...]:
    """``max_hp`` then the cuts, descending, distinct, all at least 1 (each region is
    represented by its top HP)."""
    tops = {max_hp, *hp_cuts(tables, spec, max_hp, kill_damage)}
    return tuple(sorted((t for t in tops if t >= 1), reverse=True))


@lru_cache(maxsize=1)
def _blocking_ability_ids() -> dict[int, Status]:
    names = load_names()
    return {names.ability(name): status for name, status in STATUS_BLOCKING_ABILITIES.items()}


def prestatus_options(
    tables: Tables, spec: PokemonSpec, declared: Sequence[int]
) -> tuple[int, ...]:
    """``declared`` minus the statuses ``spec`` cannot have, by type (Poison and Steel: poison;
    Fire: burn) or ability (``STATUS_BLOCKING_ABILITIES``), ruling 6."""
    types = {int(t) for t in np.asarray(tables.species_types)[spec.species]}
    ability = int(np.asarray(tables.species_abilities_decomp)[spec.species][spec.ability_slot])
    by_ability = _blocking_ability_ids().get(ability)
    return tuple(
        s
        for s in declared
        if not types & {int(t) for t in _STATUS_BLOCKING_TYPES.get(Status(s), ())}
        and by_ability != s
    )


# ---------------------------------------------------------------- move changes


def move_changes(spec: PokemonSpec, move: int, source: int) -> tuple[MoveChange, ...]:
    """Where ``move`` can go (ruling 4): the first empty slot if there is one, else any of the
    four; none when ``spec`` already knows it."""
    if move in spec.moves:
        return ()
    if 0 in spec.moves:
        return (MoveChange(spec.moves.index(0), move, source),)
    return tuple(MoveChange(slot, move, source) for slot in range(4))


def _relearner_options(problem: PrepProblem, candidate: int) -> tuple[int, ...]:
    """The candidate's Move Relearner moves that levelling does not teach (ruling)."""
    levelled = {move for _, move in problem.levelled[candidate]}
    return tuple(m for m in problem.candidates[candidate].relearner if m not in levelled)


def _free_changes(
    spec: PokemonSpec, base: Member, move: int, source: int
) -> tuple[MoveChange, ...]:
    """``move_changes`` without the slots a change of ``base`` already took."""
    taken = {c.slot for c in base.moves}
    return tuple(c for c in move_changes(spec, move, source) if c.slot not in taken)


def designated_change(tables: Tables, problem: PrepProblem, candidate: int) -> MoveChange | None:
    """The candidate's designated Move Relearner change (ruling 14): the option with the
    highest table power (ties: learned later), in an empty slot of ``base_member``'s moves,
    else the slot holding the lowest-power move (status moves count 0; ties the lower slot)."""
    base = base_member(problem, candidate)
    spec = member_spec(problem, base, tables)
    options = tuple(m for m in _relearner_options(problem, candidate) if m not in spec.moves)
    if not options:
        return None
    power = np.asarray(tables.move_power)
    category = np.asarray(tables.move_category)

    def strength(move: int) -> int:
        return 0 if int(category[move]) == 2 else int(power[move])

    best = max(range(len(options)), key=lambda i: (strength(options[i]), i))
    move = options[best]
    if 0 in spec.moves:
        return MoveChange(spec.moves.index(0), move, RELEARNER)
    taken = {c.slot for c in base.moves}
    slots = [s for s in range(4) if s not in taken]
    if not slots:
        return None
    slot = min(slots, key=lambda s: (strength(spec.moves[s]), s))
    return MoveChange(slot, move, RELEARNER)


def single_changes(
    tables: Tables,
    problem: PrepProblem,
    candidate: int,
    *,
    relearner: str = "all",
    tms: bool = True,
    items: bool = True,
    entry_hp: bool = True,
) -> list[Member]:
    """Every one-change variant of ``base_member`` that fits ``problem.resources``: a move
    from levelling that met a full set, a Move Relearner move (``relearner``: ``"all"``,
    ``"designated"`` or ``"none"``), a TM/HM move (``tms``), a bag item or taking the item off
    (``items``), an entry-HP region (``entry_hp``) and a declared prestatus."""
    cand = problem.candidates[candidate]
    base = base_member(problem, candidate)
    spec = member_spec(problem, base, tables)
    out: list[Member] = []

    def add(member: Member) -> None:
        if member_cost(problem, member).fits(problem.resources):
            out.append(member)

    def add_move(change: MoveChange) -> None:
        add(dataclasses.replace(base, moves=(*base.moves, change)))

    for _, move in problem.levelled[candidate]:
        for change in _free_changes(spec, base, move, LEVEL_UP):
            add_move(change)
    if relearner == "designated":
        if (change := designated_change(tables, problem, candidate)) is not None:
            add_move(change)
    elif relearner == "all":
        for move in _relearner_options(problem, candidate):
            for change in _free_changes(spec, base, move, RELEARNER):
                add_move(change)
    elif relearner != "none":
        raise ValueError(f"relearner must be 'all', 'designated' or 'none', not {relearner!r}")
    if tms:
        stock = dict(problem.resources.tms)
        for item, move in cand.tms:
            if item in problem.resources.hms or item in stock:
                for change in _free_changes(spec, base, move, item):
                    add_move(change)
    if items:
        if spec.item != 0:
            add(dataclasses.replace(base, item=-1))
        for item, _ in problem.resources.held:
            if item != spec.item:
                add(dataclasses.replace(base, item=item))
    if entry_hp:
        kill = ai_kill_damage(
            tables, spec, problem.opp, problem.start_condition, problem.trainer.items
        )
        for top in hp_regions(tables, spec, max_hp_of(tables, spec), kill)[1:]:
            add(dataclasses.replace(base, hp=top))
    for status in prestatus_options(tables, spec, problem.prestatus):
        add(dataclasses.replace(base, status=status))
    return out


# ---------------------------------------------------------------- prune


def _covers(tables: Tables, w: int, v: int) -> bool:
    """Whether move ``w`` equals ``v`` or covers it: both ``EffectId.PLAIN`` with one type,
    category and priority, power and PP no lower, accuracy no lower (0 never misses) and
    contact only if ``v`` makes contact."""
    if w == v:
        return True
    if w == 0 or v == 0:
        return False
    t = tables
    effect = np.asarray(t.move_effect)
    if int(effect[w]) != EffectId.PLAIN or int(effect[v]) != EffectId.PLAIN:
        return False
    for field in (t.move_type, t.move_category, t.move_priority):
        arr = np.asarray(field)
        if int(arr[w]) != int(arr[v]):
            return False
    accuracy = np.asarray(t.move_accuracy)
    flags = np.asarray(t.move_flags)
    contact_w = bool(int(flags[w]) & MoveFlag.CONTACT)
    contact_v = bool(int(flags[v]) & MoveFlag.CONTACT)
    power, pp = np.asarray(t.move_power), np.asarray(t.move_pp)
    return (
        int(power[w]) >= int(power[v])
        and (int(accuracy[w]) or 101) >= (int(accuracy[v]) or 101)
        and int(pp[w]) >= int(pp[v])
        and (not contact_w or contact_v)
    )


def prune(
    tables: Tables, problem: PrepProblem, members: Sequence[Member]
) -> tuple[list[Member], list[tuple[Member, Member]]]:
    """The conservative prune (ruling 8): ``(kept, [(dropped, dominated_by)])``. A member is
    dropped when another of the same candidate has the same item, HP and status, a cost no
    larger in every component and, slot by slot, moves that equal or cover its own. Of two
    members that cover each other the later one is dropped; a candidate's ``base_member`` is
    never dropped."""
    specs = [member_spec(problem, m, tables) for m in members]
    costs = [member_cost(problem, m) for m in members]

    def dominates(w: int, v: int) -> bool:
        a, b = members[w], members[v]
        return (
            a.candidate == b.candidate
            and (a.item, a.hp, a.status) == (b.item, b.hp, b.status)
            and _no_larger(costs[w], costs[v])
            and all(
                _covers(tables, x, y) for x, y in zip(specs[w].moves, specs[v].moves, strict=True)
            )
        )

    bases = {m.candidate: base_member(problem, m.candidate) for m in members}
    kept: list[Member] = []
    dropped: list[tuple[Member, Member]] = []
    for v, member in enumerate(members):
        if member == bases[member.candidate]:
            kept.append(member)
            continue
        by = next(
            (
                w
                for w in range(len(members))
                if w != v and dominates(w, v) and (not dominates(v, w) or w < v)
            ),
            None,
        )
        if by is None:
            kept.append(member)
        else:
            dropped.append((member, members[by]))
    return kept, dropped
