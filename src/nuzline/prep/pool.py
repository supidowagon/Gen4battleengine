"""The candidate pool: every party and box Pokemon at the level cap, with the moves it could
bring (current moves, Move Relearner moves, the bag's TMs), and whether the engine and the
trainer AI support them (spec §5.2, §5.3; plan 4a rulings 1, 2 and 7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.games import KAIZO
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import Trainer
from nuzline.engine.coverage import (
    IMPLEMENTED_ABILITIES,
    IMPLEMENTED_ITEMS,
    CoverageReport,
    Support,
    _Collector,
    halved_by_line,
    halved_defense_line,
    halving_moves,
    low_defense,
    move_effect_line,
    rom_change_line,
    rom_change_refusals,
    terrain_refusal,
)
from nuzline.engine.ids import AbilityId, ItemId
from nuzline.prep.leveling import candies_to_cap, level_to_cap
from nuzline.saves.pkm import RecordError
from nuzline.saves.save import SaveFile


@dataclass(frozen=True)
class Candidate:
    """One Pokemon the user could bring. ``location``: ``("party", 0, slot)`` or
    ``("box", box, slot)`` (0-based); ``spec`` is at the cap; ``tms``: ``(item, move)``;
    ``nickname``: the save's (it equals the species name when the Pokemon has none)."""

    location: tuple[str, int, int]
    label: str
    spec: PokemonSpec
    candies: int
    relearner: tuple[int, ...]
    tms: tuple[tuple[int, int], ...]
    nickname: str = ""

    @property
    def exp(self) -> int | None:
        """The Pokemon's exp at the cap (``level_to_cap``)."""
        return self.spec.exp


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
            spec = level_to_cap(mon, tables, cap, save.trainer)
        except RecordError as e:
            left_out.append(f"{label}: left out ({e})")
            continue
        known = set(mon.moves)
        # Learnsets are keyed by the form's personal row (plan K2a Ruling 7); level_to_cap
        # has refused a form the species does not have.
        relearner = tuple(
            m for m in learnsets.relearner_moves(mon.species, cap, mon.form) if m not in known
        )
        tms = tuple(
            (item, learnsets.tm_move(item))
            for item, _ in save.bag.get("tms_hms", ())
            if learnsets.can_learn_tm(mon.species, item, mon.form)
            and learnsets.tm_move(item) not in known
        )
        out.append(
            Candidate(
                mon.location,
                label,
                spec,
                candies_to_cap(mon.level, cap),
                relearner,
                tms,
                mon.nickname,
            )
        )
    return out, left_out


def check_pool(
    tables: Tables,
    names: Names,
    candidates: Sequence[Candidate],
    opp: Sequence[PokemonSpec],
    trainer: Trainer | None = None,
    bag: Sequence[tuple[int, int]] = (),
) -> CoverageReport:
    """The support level of the pool: per candidate its ability and held item, and every move
    it could know (current, Move Relearner, TM), by engine effect and the AI's player-side
    refusals (``ai.policy.player_move_refusal``/``player_ability_refusal``). With ``trainer``
    (Kaizo), a candidate a ``not_reached`` ROM-change condition reaches through its own spec
    or a move it could learn is refused; the condition terms about the fight alone are
    already on the fight's report. ``bag`` is the save's ``(item, quantity)`` across the held
    pockets (``problem.HELD_POCKETS``): each distinct item with a decomp hold effect is a
    ``bag: <item>`` row, graded like a held item. Against the trainer's party ``opp``, a
    Selfdestruct-family move is refused against a foe whose Defense the halving and the -6
    stage would bring to 0 (``coverage.halved_defense_line``), and a candidate whose own
    Defense is that low is refused when a foe knows such a move (``coverage.halved_by_line``)."""
    from nuzline.ai.policy import player_ability_refusal, player_move_refusal

    c = _Collector()

    def refuse(line: str) -> None:
        c.report.level = Support.NONE
        c.report.missing.append(line)

    abilities = np.asarray(tables.species_abilities)
    decomp_abilities = np.asarray(tables.species_abilities_decomp)
    item_effect = np.asarray(tables.item_effect)

    foe_defense = low_defense(tables, opp)
    foe_halving = halving_moves(tables, opp)

    def check_move(label: str, move: int, kind: str, row: int) -> None:
        support, name = move_effect_line(tables, names, move, row)
        c.add(support, f"{label}: {name}", f"{kind}, move effect")
        if (low := halved_defense_line(tables, move, foe_defense)) is not None:
            c.add(Support.NONE, f"{label}: {name}{low}", f"{kind}, move effect")
        if support != Support.NONE and (reason := player_move_refusal(move)) is not None:
            refuse(f"{label}: {name} ({kind}, {reason})")

    for cand in candidates:
        spec = cand.spec
        slot = spec.ability_slot
        row = personal_row(tables, spec.species, spec.form)
        ability = AbilityId(int(abilities[row, slot]))
        ability_name = names.ability_names[names.species_abilities(spec.species, spec.form)[slot]]
        support = IMPLEMENTED_ABILITIES.get(ability, Support.NONE)
        c.add(support, f"{cand.label}: {ability_name}", "ability")
        if (low := halved_by_line(tables, names, spec, foe_halving)) is not None:
            c.add(Support.NONE, f"{cand.label}: {low}", "opponent move effect")
        if support != Support.NONE and (
            reason := player_ability_refusal(int(decomp_abilities[row][slot]))
        ):
            refuse(f"{cand.label}: {ability_name} (ability, {reason})")
        item = ItemId(int(item_effect[spec.item]))
        c.add(
            IMPLEMENTED_ITEMS.get(item, Support.NONE),
            f"{cand.label}: {names.item_names[spec.item]}",
            "held item",
        )
        if trainer is not None and names.game == KAIZO:
            could_learn = [*cand.relearner, *(move for _, move in cand.tms)]
            for cond in rom_change_refusals(
                tables, names, trainer, [spec], could_learn, spec_terms_only=True
            ):
                c.add(Support.NONE, f"{cand.label}: {rom_change_line(cond)}", "ROM change")
        if trainer is not None:
            could_know = [*spec.moves, *cand.relearner, *(move for _, move in cand.tms)]
            if (line := terrain_refusal(tables, names, trainer, could_know)) is not None:
                c.add(Support.NONE, f"{cand.label}: {line}", "battle terrain")
        for move in spec.moves:
            if move:
                check_move(cand.label, move, "current", row)
        for move in cand.relearner:
            check_move(cand.label, move, "relearner", row)
        for item_id, move in cand.tms:
            check_move(cand.label, move, names.item_names[item_id], row)
    hold_effect = np.asarray(tables.item_hold_effect)
    for item_id in sorted({item for item, _ in bag}):
        if int(hold_effect[item_id]) != 0:
            c.add(
                IMPLEMENTED_ITEMS.get(ItemId(int(item_effect[item_id])), Support.NONE),
                f"bag: {names.item_names[item_id]}",
                "held item",
            )
    return c.report
