"""Contents a save cannot have in the chosen game (spec §8).

Nothing infers the game from a save: a Kaizo save looks like a vanilla one. Instead a
save read with the wrong game's data shows up here, as a move its Pokemon cannot learn
or an item with no name, and the caller warns loudly.
"""

from __future__ import annotations

import numpy as np

from nuzline.saves.pkm import SaveMon, location_label
from nuzline.saves.save import SaveFile

__all__ = ["SPECIES_SMEARGLE", "impossible_contents", "special_moves"]

SPECIES_SMEARGLE = 235  # Sketch learns any move: never checked

# --- moves the game teaches outside the learnset tables ----------------------------------
# Each was checked against the Kaizo patch (data/kaizo/rom_diff.toml, docs/kaizo/rom_diff.md):
# the two field scripts are among scr_seq.narc's changed members, so their patched bytes were
# disassembled with tools.rom.readers.maps (members 450 and 493, scripts.order lines 451 and
# 494); the code routines are not among the changed symbols.

SPECIES_PICHU = 172
SPECIES_ROTOM = 479
TYPE_DRAGON = 16  # the decomp type id the Route 210 script compares (TYPE_DRAGON)
MOVE_DRACO_METEOR = 434
MOVE_VOLT_TACKLE = 344

# Pokemon_SetRotomForm, src/pokemon.c:4518-4535: changing Rotom's form replaces the old
# form's move with the new form's (form 1 Heat .. 5 Mow; form 0 has none). Unchanged in
# Kaizo: the routine is not a changed arm9 symbol. Moves: -, Overheat, Hydro Pump, Blizzard,
# Air Slash, Leaf Storm.
ROTOM_FORM_MOVES = (0, 315, 56, 59, 403, 437)

# Route 228 north house, scripts_route_228_north_house.s:55,69 (CheckFinalStageStarter
# :85-110): the final-stage starters of Gens 1-4 learn their type's ultimate move. Kaizo's
# patched member 493 compares the same twelve species with the same type split and moves; it
# only adds a CheckItem cost before teaching.
ULTIMATE_MOVES = {
    **dict.fromkeys((6, 157, 257, 392), 307),  # Charizard .. Infernape: Blast Burn
    **dict.fromkeys((9, 160, 260, 395), 308),  # Blastoise .. Empoleon: Hydro Cannon
    **dict.fromkeys((3, 154, 254, 389), 338),  # Venusaur .. Torterra: Frenzy Plant
}

# Route 210 grandma Wilma, scripts_route_210_grandma_wilma_house.s:117 (the
# CheckSelectedDragonType call; :129-138 compares both of the party mon's types with
# TYPE_DRAGON): any Dragon-type learns Draco Meteor. Kaizo's patched member 450 makes the
# same type check and teaches the same move (it reorders the party scan only); the type is
# read from the game's own tables, so Kaizo's type changes apply.

# Egg_TryGiveVoltTackle, src/overlay005/daycare.c:602-615, called for a Pichu egg (:771): a
# Light Ball parent gives Volt Tackle, so the Pichu line can carry it. Unchanged in Kaizo
# (overlay5's only changed symbols are the tutor tables); the line is read from the game's
# own evolutions, so it is the Kaizo line in Kaizo.


def special_moves(species: int, form: int, types, chain) -> set[int]:
    """The special-route moves ``species``' ``form`` can know: ``types`` are its two type
    ids, ``chain`` the species and its pre-evolutions."""
    moves = set()
    if species == SPECIES_ROTOM and 0 < form < len(ROTOM_FORM_MOVES):
        moves.add(ROTOM_FORM_MOVES[form])
    if species in ULTIMATE_MOVES:
        moves.add(ULTIMATE_MOVES[species])
    if TYPE_DRAGON in types:
        moves.add(MOVE_DRACO_METEOR)
    if SPECIES_PICHU in chain:
        moves.add(MOVE_VOLT_TACKLE)
    return moves


def _pre_evolutions(learnsets, n_species: int) -> dict[int, set[int]]:
    """species -> the species that evolve into it (direct parents only)."""
    parents: dict[int, set[int]] = {}
    for s in range(1, n_species):
        for _method, _param, target in learnsets.evolutions(s):
            if target:
                parents.setdefault(int(target), set()).add(s)
    return parents


def _chain(species: int, parents: dict[int, set[int]]) -> list[int]:
    """``species`` and every species before it in its evolution line."""
    out, todo = [], [species]
    while todo:
        s = todo.pop()
        if s not in out:
            out.append(s)
            todo.extend(sorted(parents.get(s, ())))
    return out


class _Learnable:
    """The union of level-up, TM/HM, tutor and egg moves over a species' line, plus the
    special-route moves (:func:`special_moves`), cached."""

    def __init__(self, tables, learnsets, names):
        self._ls = learnsets
        self._names = names
        self._types = np.asarray(tables.species_types)
        self._parents = _pre_evolutions(learnsets, len(names.species_names))
        self._tm_items = [i for i in range(len(names.item_names)) if learnsets.tm_move(i)]
        self._tutor = sorted({t.move for t in learnsets.tutor_moves()})
        self._cache: dict[tuple[int, int], frozenset[int]] = {}

    def _own(self, species: int, form: int) -> set[int]:
        ls = self._ls
        moves = {move for _lv, move in ls.level_up(species, form)}
        moves.update(ls.tm_move(i) for i in self._tm_items if ls.can_learn_tm(species, i, form))
        moves.update(m for m in self._tutor if ls.can_learn_tutor(species, m, form))
        moves.update(ls.egg_moves(species))
        return moves

    def moves(self, species: int, form: int) -> frozenset[int]:
        key = (species, form)
        if key not in self._cache:
            moves = self._own(species, form)
            chain = _chain(species, self._parents)
            # Pre-evolutions at form 0: the stored form is the evolved mon's. Safe in Gen 4:
            # Burmy's forms share one learnset (cosmetic rows), and Wormadam's own form is
            # resolved through form_row in _own above.
            for s in chain[1:]:
                moves |= self._own(s, 0)
            types = [int(t) for t in self._types[self._names.form_row(species, form)]]
            moves |= special_moves(species, form, types, chain)
            self._cache[key] = frozenset(moves)
        return self._cache[key]


def _item_named(names, item: int) -> bool:
    return 0 <= item < len(names.item_names) and bool(names.item_names[item])


def _mon_lines(mon: SaveMon, names, learnable: _Learnable) -> list[str]:
    where = f"{location_label(mon.location)} {names.species_names[mon.species]}"
    lines = []
    if mon.species != SPECIES_SMEARGLE:
        allowed = learnable.moves(mon.species, mon.form)
        for move in mon.moves:
            if move and move not in allowed:
                lines.append(f"{where}: {names.move_names[move]} is not learnable")
    if mon.item and not _item_named(names, mon.item):
        lines.append(f"{where}: held item {mon.item} has no name")
    return lines


def impossible_contents(save: SaveFile, tables, names, learnsets) -> list[str]:
    """One line per impossible fact, each starting with its location: a move outside the
    union of the species' and its pre-evolutions' level-up, TM/HM, tutor and egg moves and
    the special-route moves (Smeargle skipped), and a held or bag item whose name is empty
    or out of range. ``tables`` gives each form's types (Draco Meteor)."""
    learnable = _Learnable(tables, learnsets, names)
    lines = []
    for mon in save.party:
        lines += _mon_lines(mon, names, learnable)
    for box in save.boxes:
        for mon in box:
            if mon is not None:
                lines += _mon_lines(mon, names, learnable)
    for pocket, entries in save.bag.items():
        for slot, (item, _qty) in enumerate(entries):
            if not _item_named(names, item):
                lines.append(f"bag {pocket} slot {slot}: item {item} has no name")
    return lines
