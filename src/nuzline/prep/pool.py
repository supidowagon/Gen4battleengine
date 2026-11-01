"""The candidate pool: every party and box Pokemon at the level cap, with the moves it could
bring (current moves, Move Relearner moves, the bag's TMs), and whether the engine and the
trainer AI support them (spec §5.2, §5.3; plan 4a rulings 1, 2 and 7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.engine.coverage import (
    IMPLEMENTED_ABILITIES,
    IMPLEMENTED_EFFECTS,
    IMPLEMENTED_ITEMS,
    CoverageReport,
    Support,
)
from nuzline.engine.ids import AbilityId, EffectId, ItemId
from nuzline.prep.leveling import candies_to_cap, level_to_cap
from nuzline.saves.pkm import RecordError
from nuzline.saves.save import SaveFile


@dataclass(frozen=True)
class Candidate:
    """One Pokemon the user could bring. ``location``: ``("party", 0, slot)`` or
    ``("box", box, slot)`` (0-based); ``spec`` is at the cap; ``tms``: ``(item, move)``."""

    location: tuple[str, int, int]
    label: str
    spec: PokemonSpec
    candies: int
    relearner: tuple[int, ...]
    tms: tuple[tuple[int, int], ...]


def _label(location: tuple[str, int, int], species: str) -> str:
    kind, box, slot = location
    where = f"party slot {slot}" if kind == "party" else f"box {box + 1} slot {slot}"
    return f"{where} {species}"


def pool_candidates(
    save: SaveFile, tables: Tables, names: Names, learnsets: Learnsets, cap: int
) -> tuple[list[Candidate], list[str]]:
    """The save's party, then its boxes in order, each at ``cap``. Eggs and records
    ``SaveMon.to_spec`` refuses are left out, one line each in the second list."""
    mons = list(save.party) + [m for box in save.boxes for m in box if m is not None]
    out: list[Candidate] = []
    left_out: list[str] = []
    for mon in mons:
        label = _label(mon.location, names.species_names[mon.species])
        if mon.is_egg:
            left_out.append(f"{label}: egg left out")
            continue
        try:
            spec = level_to_cap(mon, tables, cap)
        except RecordError as e:
            left_out.append(f"{label}: left out ({e})")
            continue
        known = set(mon.moves)
        relearner = tuple(m for m in learnsets.relearner_moves(mon.species, cap) if m not in known)
        tms = tuple(
            (item, learnsets.tm_move(item))
            for item, _ in save.bag.get("tms_hms", ())
            if learnsets.can_learn_tm(mon.species, item) and learnsets.tm_move(item) not in known
        )
        out.append(
            Candidate(mon.location, label, spec, candies_to_cap(mon.level, cap), relearner, tms)
        )
    return out, left_out


def check_pool(tables: Tables, names: Names, candidates: Sequence[Candidate]) -> CoverageReport:
    """The support level of the pool: per candidate its ability and held item, and every move
    it could know (current, Move Relearner, TM), by engine effect and the AI's player-side
    refusals (``ai.policy.player_move_refusal``/``player_ability_refusal``)."""
    from nuzline.ai.policy import player_ability_refusal, player_move_refusal

    report = CoverageReport()

    def missing(line: str) -> None:
        report.level = Support.NONE
        report.missing.append(line)

    effect = np.asarray(tables.move_effect)
    abilities = np.asarray(tables.species_abilities)
    decomp_abilities = np.asarray(tables.species_abilities_decomp)
    item_effect = np.asarray(tables.item_effect)

    def check_move(label: str, move: int, kind: str) -> None:
        name = names.move_names[move]
        if IMPLEMENTED_EFFECTS.get(EffectId(int(effect[move])), Support.NONE) == Support.NONE:
            missing(f"{label}: {name} ({kind}, move effect UNSUPPORTED)")
        elif (reason := player_move_refusal(move)) is not None:
            missing(f"{label}: {name} ({kind}, {reason})")

    for cand in candidates:
        spec = cand.spec
        slot = spec.ability_slot
        ability = AbilityId(int(abilities[spec.species, slot]))
        ability_name = names.ability_names[names.species_abilities(spec.species)[slot]]
        if IMPLEMENTED_ABILITIES.get(ability, Support.NONE) == Support.NONE:
            missing(f"{cand.label}: {ability_name} (ability UNSUPPORTED)")
        elif reason := player_ability_refusal(int(decomp_abilities[spec.species][slot])):
            missing(f"{cand.label}: {ability_name} (ability, {reason})")
        item = ItemId(int(item_effect[spec.item]))
        if IMPLEMENTED_ITEMS.get(item, Support.NONE) == Support.NONE:
            missing(f"{cand.label}: {names.item_names[spec.item]} (held item UNSUPPORTED)")
        for move in spec.moves:
            if move:
                check_move(cand.label, move, "current")
        for move in cand.relearner:
            check_move(cand.label, move, "relearner")
        for item_id, move in cand.tms:
            check_move(cand.label, move, names.item_names[item_id])
    return report
