"""Gauntlet transitions of the prep side (design v5.3 sections 3 and 5).

Built so far: the level-cap sit-out (ledger E3-B3-5.b) and the Carry type with its views
(GT-4a, E3-B3-4a). The other transitions of the option registry (``data.gauntlet.OPTIONS``,
each option's ``transition`` name) land with the groups that need the Carry."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.gauntlet import OPTIONS, Settings
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.state import spec_exp
from nuzline.engine.stats import SPECIES_SHEDINJA
from nuzline.prep.growth import ITEM_POKE_BALL, MAX_PARTY_SIZE, _max_pp, evolve
from nuzline.saves.pkm import ITEM_LUXURY_BALL

__all__ = [
    "CapSitOut",
    "Carry",
    "cap_sit_out",
    "carry_batch",
    "carry_from_party",
    "carry_row",
    "carry_stats",
    "evolve_carry",
    "member_spec",
    "set_member",
    "stack_carries",
]


class CapSitOut(NamedTuple):
    """The level-cap sit-out at one gap. ``over``: the party slots newly over the next cap;
    ``deposited``: the slots that leave the party at this gap (``cap_sit_out_in_party =
    forbidden``); ``barred``: the slots that stay in the party and may not fight, the earlier
    sit-outs included (``allowed``); ``lost``: a ``forbidden`` sit-out that no deposit at this
    gap can take, so the playout row counts as lost."""

    over: tuple[int, ...]
    deposited: tuple[int, ...]
    barred: tuple[int, ...]
    lost: bool


def _checked(name: str, value: str) -> str:
    allowed = OPTIONS[name].values
    if value not in allowed:
        raise ValueError(f"{name} {value!r} is not one of {', '.join(allowed)}")
    return value


def cap_sit_out(
    levels: Sequence[int | None],
    next_cap: int | None,
    *,
    cap_overflow: str,
    cap_sit_out_in_party: str,
    deposit_room: int | None,
    sat_out: Collection[int] = (),
) -> CapSitOut:
    """The sit-out after a won fight, from the party's levels after it (``levels`` by party
    slot, ``None`` for an empty slot) and the next fight's level cap (``next_cap``, from
    ``prep.caps.next_fight_cap``, or ``default_cap`` of the next fight along the resolved
    order; ``None`` after the last fight).

    Under ``cap_overflow = sit_out_after_fight`` a member whose level exceeds ``next_cap`` takes
    no part in any later fight of the chain; under ``fight`` nothing is removed (design section
    3). ``cap_sit_out_in_party = forbidden``: every such member must be deposited at this gap, in
    the PC (a ``pc_gaps`` or walk-out gap) or by a swap; ``deposit_room`` is how many members
    the gap can take (``None``: any number, a PC or a walk-out; ``0``: none, a gap without a PC
    or a swap), and more over-cap members than that is ``lost``. ``allowed``: they stay in the
    party, barred. ``sat_out`` holds the slots that already sit out: they stay barred. These
    are user settings (house rules), not game rules; the game lets the member fight."""
    _checked("cap_overflow", cap_overflow)
    _checked("cap_sit_out_in_party", cap_sit_out_in_party)
    if deposit_room is not None and deposit_room < 0:
        raise ValueError(f"deposit_room {deposit_room} is negative")
    before = tuple(sorted(set(sat_out)))
    if cap_overflow == "fight" or next_cap is None:
        return CapSitOut((), (), before, False)
    over = tuple(
        slot
        for slot, level in enumerate(levels)
        if level is not None and level > next_cap and slot not in before
    )
    if cap_sit_out_in_party == "allowed":
        return CapSitOut(over, (), tuple(sorted({*before, *over})), False)
    if deposit_room is not None and len(over) > deposit_room:
        return CapSitOut(over, (), before, True)
    return CapSitOut(over, over, before, False)


# --- the Carry (design v5.4 sections 5 and 7; ledger E3-B3-4a) --------------------------------


class Carry(NamedTuple):
    """The party between two fights of a chain: design section 7 per save-party slot, then the
    chain fields of section 5. Every field is a numpy array with an optional leading batch axis
    ``R`` (one row per playout row); the trailing shape of each is in the comments, and
    :func:`carry_batch` tells whether the axis is there.

    Per slot (``[6]``, ``[6, 4]`` for the move fields, ``[6, 6]`` in ``ids.Stat`` order for
    ``evs``, ``ivs`` and ``stats``). ``present`` marks the slots that hold a Pokemon (a party
    has one to six); every other per-slot field of an empty slot is 0.

    - what the battle does not write, constant from the save: ``species``, ``form``,
      ``personality``, ``ivs``, ``nature``, ``gender``, ``beauty`` (``MON_DATA_BEAUTY``, read by
      the beauty evolution), ``ball``, ``egg_location``, ``pokerus``, and ``exp_boost``
      (``PokemonSpec.exp_boost``, the OT and language test the battle's EXP payout reads);
    - ``ability_slot``: the slot of the species' abilities the Pokemon has (the party's
      ability; a battle-time change reverts, ``src/battle/battle_io_command.c:601-604``);
    - what persists: ``level``, ``exp``, ``evs``, ``hp``, ``status`` (an ``ids.Status``),
      ``status_counter``, ``moves``, ``pp``, ``pp_ups``, ``item``, ``friendship``;
    - ``stats``: all six stats, max HP first, recomputed by :func:`carry_stats`
      (``Pokemon_CalcStats``, ``src/pokemon.c:633-645``). Max HP is ``stats[..., 0]``; max PP is
      derived from move and PP Ups (``MoveTable_CalcMaxPP``, ``src/pokemon.c:961-965``).

    Chain fields: ``bag`` (``[n_items]``, the count of each item id); ``dead`` and ``sat_out``
    (``[6]``, by carry slot); ``deaths_used`` (a count); ``swap_left`` and ``bag_uses_left``
    (the amount left of each Counted option of ``data.gauntlet.OPTIONS``, ``swap`` by its
    ``budget`` and ``bag_uses``; -1 = unlimited); ``bench`` (``[n_box]``, the box members
    chosen as the bench that have not been swapped in); ``beaten`` (``[n_fights]``, the group's
    fights beaten, by fight index); ``partner_set`` (whether the partner has joined).

    Not here: the swap plan (GT-7) and the growth plan (GT-15) add their fields with their
    groups. ``levelled`` and the learn work are ``carry_out``'s (GT-4b)."""

    present: np.ndarray  # bool[6]
    species: np.ndarray  # int32[6]
    form: np.ndarray  # int32[6]
    ability_slot: np.ndarray  # int32[6]
    personality: np.ndarray  # int64[6]
    nature: np.ndarray  # int32[6]
    gender: np.ndarray  # int32[6], 0 male, 1 female, 2 none
    beauty: np.ndarray  # int32[6]
    ball: np.ndarray  # int32[6], the ball's item id
    egg_location: np.ndarray  # int32[6]
    pokerus: np.ndarray  # int32[6], 1 when the Pokerus byte is nonzero
    exp_boost: np.ndarray  # int32[6], percent
    ivs: np.ndarray  # int32[6, 6]
    level: np.ndarray  # int32[6]
    exp: np.ndarray  # int32[6]
    evs: np.ndarray  # int32[6, 6]
    stats: np.ndarray  # int32[6, 6]
    hp: np.ndarray  # int32[6]
    status: np.ndarray  # int32[6]
    status_counter: np.ndarray  # int32[6]
    moves: np.ndarray  # int32[6, 4]
    pp: np.ndarray  # int32[6, 4]
    pp_ups: np.ndarray  # int32[6, 4]
    item: np.ndarray  # int32[6]
    friendship: np.ndarray  # int32[6]
    bag: np.ndarray  # int32[n_items]
    dead: np.ndarray  # bool[6]
    sat_out: np.ndarray  # bool[6]
    deaths_used: np.ndarray  # int32[]
    swap_left: np.ndarray  # int32[]
    bag_uses_left: np.ndarray  # int32[]
    bench: np.ndarray  # bool[n_box]
    beaten: np.ndarray  # bool[n_fights]
    partner_set: np.ndarray  # bool[]


# The rank of each field after the optional batch axis; every field not listed has one.
_TRAILING = {
    "ivs": 2,
    "evs": 2,
    "stats": 2,
    "moves": 2,
    "pp": 2,
    "pp_ups": 2,
    "deaths_used": 0,
    "swap_left": 0,
    "bag_uses_left": 0,
    "partner_set": 0,
}
# Fields set from a member's PokemonSpec one slot at a time, besides ``present`` and the stats.
_SPEC_SCALARS = (
    "species",
    "form",
    "ability_slot",
    "nature",
    "gender",
    "egg_location",
    "pokerus",
    "exp_boost",
    "level",
    "status",
    "status_counter",
    "item",
    "friendship",
)
_SPEC_VECTORS = ("ivs", "evs", "moves", "pp_ups")


def carry_batch(carry: Carry) -> int | None:
    """``R`` when the Carry has the leading batch axis, ``None`` for a single Carry
    (``ValueError``: the fields disagree)."""
    sizes = set()
    for name, value in zip(Carry._fields, carry, strict=True):
        extra = np.ndim(value) - _TRAILING.get(name, 1)
        if extra not in (0, 1):
            raise ValueError(f"Carry.{name} has {np.ndim(value)} dimensions")
        sizes.add(None if extra == 0 else np.shape(value)[0])
    if len(sizes) != 1:
        raise ValueError("Carry fields disagree on the batch axis")
    return sizes.pop()


def stack_carries(rows: Sequence[Carry]) -> Carry:
    """``R`` single Carries as one with the leading batch axis (``ValueError``: an empty
    list, a batched row, or rows whose chain-field widths differ)."""
    if not rows:
        raise ValueError("no Carry rows to stack")
    for row in rows:
        if carry_batch(row) is not None:
            raise ValueError("stack_carries takes single Carries")
    return Carry(*(np.stack([getattr(r, f) for r in rows]) for f in Carry._fields))


def carry_row(carry: Carry, i: int) -> Carry:
    """Row ``i`` of a batched Carry, as a single Carry (copies)."""
    if carry_batch(carry) is None:
        raise ValueError("carry_row takes a batched Carry")
    return Carry(*(np.array(v[i]) for v in carry))


def _personal_rows(tables: Tables, species: np.ndarray, form: np.ndarray) -> np.ndarray:
    """The personal row of each ``(species, form)`` (``ValueError`` for a form the species
    lacks), looked up once per distinct pair."""
    pairs = np.stack([species, form], axis=-1).reshape(-1, 2)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    rows = np.array([personal_row(tables, int(s), int(f)) for s, f in unique])
    return rows[inverse.reshape(species.shape)]


def carry_stats(tables: Tables, carry: Carry) -> np.ndarray:
    """The six stats of every slot, ``int32[..., 6, 6]`` with max HP first, as
    ``Pokemon_CalcStats`` makes them (``src/pokemon.c:564-648``; the formula of
    ``engine.stats.calc_stats``): ``(2 * base + IV + EV / 4) * level / 100 + 5`` then the
    nature's 110/100 or 90/100, and max HP ``... + level + 10``, 1 for Shedinja. Read from the
    species' form row, level, EVs, IVs and nature of the Carry, never from a battler's stats
    (Transform holds the target's). An empty slot is 0. Vectorised over the batch axis; the
    tests pin it against ``spec_stats``."""
    rows = _personal_rows(tables, carry.species, carry.form)
    base = np.asarray(tables.species_base_stats)[rows].astype(np.int32)
    mods = np.asarray(tables.nature_mods)[carry.nature].astype(np.int32)
    level = carry.level.astype(np.int32)[..., None]
    core = (2 * base + carry.ivs + carry.evs // 4) * level // 100
    stats = core + 5
    stats = np.where(mods > 0, stats * 110 // 100, stats)
    stats = np.where(mods < 0, stats * 90 // 100, stats)
    stats[..., 0] = np.where(
        carry.species == SPECIES_SHEDINJA, 1, core[..., 0] + level[..., 0] + 10
    )
    return np.where(carry.present[..., None], stats, 0).astype(np.int32)


def _mask(values: Collection[int], size: int, what: str) -> np.ndarray:
    out = np.zeros(size, dtype=bool)
    for v in values:
        if not 0 <= v < size:
            raise ValueError(f"{what} {v} is outside 0..{size - 1}")
        out[v] = True
    return out


def _spec_pp(tables: Tables, mon: PokemonSpec) -> tuple[int, ...]:
    """The member's PP: its own, else each move's maximum at its PP Ups (an empty slot 0)."""
    if mon.pp is not None:
        return tuple(mon.pp)
    return tuple(
        _max_pp(tables, m, u) if m != 0 else 0 for m, u in zip(mon.moves, mon.pp_ups, strict=True)
    )


def _write_member(
    fields: dict[str, np.ndarray],
    tables: Tables,
    slot: int,
    mon: PokemonSpec,
    personality: int,
    beauty: int,
    ball: int,
) -> None:
    """Write ``mon`` into slot ``slot`` of ``fields`` (single, per-slot arrays); ``hp`` is
    left to the caller, which knows the recomputed maximum."""
    fields["present"][slot] = True
    for name in _SPEC_SCALARS:
        fields[name][slot] = getattr(mon, name)
    for name in _SPEC_VECTORS:
        fields[name][slot] = getattr(mon, name)
    fields["personality"][slot] = personality
    fields["beauty"][slot] = beauty
    fields["ball"][slot] = ball
    fields["exp"][slot] = spec_exp(tables, mon)
    fields["pp"][slot] = _spec_pp(tables, mon)


def _empty_fields() -> dict[str, np.ndarray]:
    """The per-slot fields of a Carry with no member, by name."""
    out: dict[str, np.ndarray] = {}
    for name in Carry._fields:
        rank = _TRAILING.get(name, 1)
        if name in ("moves", "pp", "pp_ups"):
            out[name] = np.zeros((6, 4), dtype=np.int32)
        elif name in ("ivs", "evs", "stats"):
            out[name] = np.zeros((6, 6), dtype=np.int32)
        elif name in ("present", "dead", "sat_out"):
            out[name] = np.zeros(6, dtype=bool)
        elif name == "personality":
            out[name] = np.zeros(6, dtype=np.int64)
        elif rank == 1 and name not in ("bag", "bench", "beaten"):
            out[name] = np.zeros(6, dtype=np.int32)
    return out


def carry_from_party(
    tables: Tables,
    party: Sequence[PokemonSpec],
    bag: Mapping[int, int],
    settings: Settings,
    *,
    personality: Sequence[int],
    beauty: Sequence[int] | None = None,
    ball: Sequence[int] | None = None,
    dead: Collection[int] = (),
    sat_out: Collection[int] = (),
    deaths_used: int = 0,
    box_size: int = 0,
    bench: Collection[int] = (),
    n_fights: int = 0,
    beaten: Collection[int] = (),
    partner_set: bool = False,
) -> Carry:
    """The first fight's Carry (single, no batch axis) from the save party, in its order.

    - ``party``: one to six :class:`~nuzline.data.spec.PokemonSpec`. The stats are computed
      (:func:`carry_stats`): a ``stats`` stored in the spec is not read, as the carried stats
      never are (``src/pokemon.c:633-645``); ``current_hp`` ``None`` is full HP, ``pp`` ``None``
      each move's maximum PP at its PP Ups (``MoveTable_CalcMaxPP``), ``exp`` ``None`` the base
      of the level (``spec_exp``). ``ability_form`` and ``stats_form`` are trainer-Pokemon
      quirks a save party does not have: the Carry reads the Pokemon's own form row.
    - ``personality``: per member, a constant of the Pokemon (``MON_DATA_PERSONALITY``; not in
      ``PokemonSpec``). ``beauty`` per member, ``None`` = 0 (a hand-typed party has none).
      ``ball`` per member, ``None`` = a Luxury Ball where ``friend_bonus`` is 1, else a Poke Ball.
    - ``bag``: item id -> count. ``settings`` gives the Counted amounts left at the start:
      ``swap`` by its ``budget`` and ``bag_uses``.
    - ``dead``: party slots dead at the root, besides every member at HP 0 (design section 5,
      "Nuzlocke state of the root"); ``sat_out``: slots sitting out; ``deaths_used`` the deaths
      used so far. ``bench`` box ids in ``0..box_size - 1``; ``beaten`` fight indices in
      ``0..n_fights - 1`` of the group; ``partner_set`` whether the partner has joined."""
    n = len(party)
    if not 1 <= n <= MAX_PARTY_SIZE:
        raise ValueError(f"a party holds 1-{MAX_PARTY_SIZE} Pokemon, got {n}")
    for name, per_member in (("personality", personality), ("beauty", beauty), ("ball", ball)):
        if per_member is not None and len(per_member) != n:
            raise ValueError(f"{name} needs {n} values, got {len(per_member)}")
    n_items = int(np.shape(tables.item_hold_effect)[0])  # one row per item id
    counts = np.zeros(n_items, dtype=np.int32)
    for item, count in bag.items():
        if not 0 <= item < n_items:
            raise ValueError(f"bag item {item} is outside 0..{n_items - 1}")
        if count < 0:
            raise ValueError(f"bag item {item} has a negative count {count}")
        counts[item] = count
    dead_slots = {i for i, mon in enumerate(party) if mon.current_hp == 0} | set(dead)
    for name, slots in (("dead", dead_slots), ("sat_out", set(sat_out))):
        for s in slots:
            if not 0 <= s < n:
                raise ValueError(f"{name} slot {s} is outside the party of {n}")
    fields = _empty_fields()
    for i, mon in enumerate(party):
        default_ball = ITEM_LUXURY_BALL if mon.friend_bonus else ITEM_POKE_BALL
        _write_member(
            fields,
            tables,
            i,
            mon,
            personality[i],
            0 if beauty is None else beauty[i],
            default_ball if ball is None else ball[i],
        )
    fields["dead"] = _mask(dead_slots, 6, "dead slot")
    fields["sat_out"] = _mask(sat_out, 6, "sat_out slot")
    carry = Carry(
        **fields,
        bag=counts,
        deaths_used=np.array(deaths_used, dtype=np.int32),
        swap_left=np.array(settings["swap"].value.budget, dtype=np.int32),
        bag_uses_left=np.array(settings["bag_uses"].value, dtype=np.int32),
        bench=_mask(bench, box_size, "bench box id"),
        beaten=_mask(beaten, n_fights, "fight index"),
        partner_set=np.array(partner_set, dtype=bool),
    )
    return _with_stats(tables, carry, party)


def _with_stats(tables: Tables, carry: Carry, party: Sequence[PokemonSpec]) -> Carry:
    """``carry`` with ``stats`` computed and ``hp`` set from each spec's ``current_hp``
    (``None`` = the computed maximum HP)."""
    stats = carry_stats(tables, carry)
    hp = np.zeros(6, dtype=np.int32)
    for i, mon in enumerate(party):
        hp[i] = stats[i, 0] if mon.current_hp is None else mon.current_hp
    return carry._replace(stats=stats, hp=hp)


def _single(carry: Carry) -> None:
    if carry_batch(carry) is not None:
        raise ValueError("a member view takes a single Carry: take a row with carry_row")


def member_spec(carry: Carry, slot: int) -> PokemonSpec:
    """The :class:`~nuzline.data.spec.PokemonSpec` of carry slot ``slot`` (a single Carry), the
    view GT-13's ``evolve`` and the prep code read: current HP as stored, ``pp`` and ``exp`` as
    carried, ``stats`` ``None`` (computed from the Pokemon's own form, ``stats_form`` and
    ``ability_form`` ``None``), ``friend_bonus`` 1 for a Luxury Ball. ``personality``, ``beauty``
    and ``ball`` have no ``PokemonSpec`` field; read them from the Carry (:func:`set_member`
    keeps them). ``ValueError`` for an empty slot."""
    _single(carry)
    if not carry.present[slot]:
        raise ValueError(f"carry slot {slot} is empty")
    return PokemonSpec(
        species=int(carry.species[slot]),
        level=int(carry.level[slot]),
        nature=int(carry.nature[slot]),
        ivs=tuple(int(v) for v in carry.ivs[slot]),
        evs=tuple(int(v) for v in carry.evs[slot]),
        moves=tuple(int(v) for v in carry.moves[slot]),
        item=int(carry.item[slot]),
        ability_slot=int(carry.ability_slot[slot]),
        gender=int(carry.gender[slot]),
        friendship=int(carry.friendship[slot]),
        form=int(carry.form[slot]),
        current_hp=int(carry.hp[slot]),
        status=int(carry.status[slot]),
        status_counter=int(carry.status_counter[slot]),
        pp=tuple(int(v) for v in carry.pp[slot]),
        exp=int(carry.exp[slot]),
        exp_boost=int(carry.exp_boost[slot]),
        pokerus=int(carry.pokerus[slot]),
        friend_bonus=int(carry.ball[slot] == ITEM_LUXURY_BALL),
        egg_location=int(carry.egg_location[slot]),
        pp_ups=tuple(int(v) for v in carry.pp_ups[slot]),
    )


def set_member(
    tables: Tables,
    carry: Carry,
    slot: int,
    mon: PokemonSpec,
    *,
    personality: int | None = None,
    beauty: int | None = None,
    ball: int | None = None,
) -> Carry:
    """``carry`` (single) with ``mon`` in slot ``slot``: the other direction of
    :func:`member_spec`. The slot's stats are recomputed (:func:`carry_stats`; ``mon.stats`` is
    not read) and ``current_hp`` ``None`` is full HP. ``personality``, ``beauty`` and ``ball``
    keep the slot's values when it was occupied and ``None``; a new member needs
    ``personality`` (``ValueError``), has ``beauty`` 0 and the ball of ``friend_bonus`` by
    default. ``dead`` and ``sat_out`` of the slot are not touched (the caller's), except that a
    slot that was empty is neither."""
    _single(carry)
    if not 0 <= slot < MAX_PARTY_SIZE:
        raise ValueError(f"carry slot {slot} is outside 0..{MAX_PARTY_SIZE - 1}")
    occupied = bool(carry.present[slot])
    if personality is None:
        if not occupied:
            raise ValueError(f"carry slot {slot} is empty: a new member needs its personality")
        personality = int(carry.personality[slot])
    if beauty is None:
        beauty = int(carry.beauty[slot]) if occupied else 0
    if ball is None:
        if occupied:
            ball = int(carry.ball[slot])
        else:
            ball = ITEM_LUXURY_BALL if mon.friend_bonus else ITEM_POKE_BALL
    fields = {name: np.array(getattr(carry, name)) for name in Carry._fields}
    _write_member(fields, tables, slot, mon, personality, beauty, ball)
    if not occupied:
        fields["dead"][slot] = False
        fields["sat_out"][slot] = False
    written = Carry(**fields)
    stats = carry_stats(tables, written)
    hp = fields["hp"]
    hp[slot] = stats[slot, 0] if mon.current_hp is None else mon.current_hp
    return written._replace(stats=stats, hp=hp)


def evolve_carry(
    tables: Tables,
    learnsets: Learnsets,
    carry: Carry,
    slot: int,
    target: int,
    method: int,
    settings: Settings,
    *,
    learn_answers: Mapping[int, int | None] | None = None,
) -> Carry:
    """GT-13's :func:`~nuzline.prep.growth.evolve` on carry slot ``slot`` (a single Carry; the
    evolution of one row is its own decision, so a batch is evolved row by row). The member,
    the party (the present slots, in carry order) and the bag count are read from the Carry, the
    evolved member goes back into its slot with its personality, beauty and ball
    (``MON_DATA_PERSONALITY`` is the ``personality`` ``evolve`` takes), and the bag keeps the
    result. A Shedinja goes into the first free carry slot (``src/evolution.c:773`` puts it at
    the end of the party; the slots of a party with no empty slot between members are the same
    place) with the evolved member's personality and beauty, a Poke Ball
    (``src/evolution.c:736-737``), not dead and not sitting out."""
    _single(carry)
    member = member_spec(carry, slot)
    party = [member_spec(carry, int(i)) for i in np.flatnonzero(carry.present)]
    bag = {int(i): int(carry.bag[i]) for i in np.flatnonzero(carry.bag)}
    personality = int(carry.personality[slot])
    evolved, extra, new_bag = evolve(
        tables,
        learnsets,
        member,
        target,
        method,
        bag,
        party,
        settings,
        personality=personality,
        learn_answers=learn_answers,
    )
    out = set_member(tables, carry, slot, evolved)
    if extra is not None:
        free = np.flatnonzero(~out.present)
        out = set_member(
            tables,
            out,
            int(free[0]),
            extra,
            personality=personality,
            beauty=int(carry.beauty[slot]),
            ball=ITEM_POKE_BALL,
        )
    counts = np.zeros_like(out.bag)
    for item, count in new_bag.items():
        counts[item] = count
    return out._replace(bag=counts)
