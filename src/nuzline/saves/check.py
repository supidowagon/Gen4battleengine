"""Contents a save cannot have in the chosen game (spec §8).

Nothing infers the game from a save: a Kaizo save looks like a vanilla one. Instead a
save read with the wrong game's data shows up here, as a move its Pokemon cannot learn
or an item with no name, and the caller warns loudly.
"""

from __future__ import annotations

from nuzline.saves.pkm import SaveMon, _label
from nuzline.saves.save import SaveFile

__all__ = ["SPECIES_SMEARGLE", "impossible_contents"]

SPECIES_SMEARGLE = 235  # Sketch learns any move: never checked


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
    """The union of level-up, TM/HM, tutor and egg moves over a species' line, cached."""

    def __init__(self, learnsets, names):
        self._ls = learnsets
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
            for s in _chain(species, self._parents)[1:]:
                moves |= self._own(s, 0)  # a pre-evolution's form is not stored
            self._cache[key] = frozenset(moves)
        return self._cache[key]


def _item_named(names, item: int) -> bool:
    return 0 <= item < len(names.item_names) and bool(names.item_names[item])


def _mon_lines(mon: SaveMon, names, learnable: _Learnable) -> list[str]:
    where = f"{_label(mon.location)} {names.species_names[mon.species]}"
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
    union of the species' and its pre-evolutions' level-up, TM/HM, tutor and egg moves
    (Smeargle skipped), and a held or bag item whose name is empty or out of range.
    ``tables`` is accepted for symmetry with the other save checks and is not read."""
    del tables
    learnable = _Learnable(learnsets, names)
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
