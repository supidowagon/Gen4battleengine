"""Level-up learnsets and TM/HM compatibility (host-only lookups from ``manifest.json``).

Level-up learnsets are ``res/pokemon/<species>/data.json`` ``learnset.by_level``. TM/HM
compatibility is ``learnset.by_tm``, numbered as tools/dataproc/src/speciesproc.c
ProcTMLearnset sets the mask bits (TM01-TM92 = 0-91, HM01-HM08 = 92-99); the move an
item teaches is ``teachesMove`` in res/items/data/tm10.json etc. (src/item.c).

Both are keyed by personal row (species, then data forms; ``Pokemon_LoadLevelUpMovesOf``,
``src/pokemon.c:4585``), resolved from ``(species, form)`` by ``nuzline.data.forms.form_row``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nuzline.data import tables
from nuzline.data.forms import form_row

HM_FIRST_INDEX = 92  # TM01-TM92 = 0-91, HM01-HM08 = 92-99


@dataclass(frozen=True)
class TutorMove:
    """One ``TeachableMove`` (``include/tutor_movesets.h:7-14``): the move, its shard costs and
    the tutor's ``generated/tutor_locations.txt`` stem."""

    move: int
    red: int
    blue: int
    yellow: int
    green: int
    location: str


class Learnsets:
    def __init__(self, manifest: dict):
        self._level = {
            int(s): tuple((int(lv), int(mv)) for lv, mv in pairs)
            for s, pairs in manifest["level_learnsets"].items()
        }
        self._tm = {int(s): frozenset(ix) for s, ix in manifest["tm_learnsets"].items()}
        self._tm_moves = tuple(int(m) for m in manifest["tm_moves"])
        self._first_tm_item = int(manifest["first_tm_item"])
        self._form_rows = manifest["species_form_row"]
        self._evolutions = {
            int(s): tuple((int(m), int(p), int(t)) for m, p, t in rows)
            for s, rows in manifest["evolutions"].items()
        }
        self._egg_moves = {
            int(s): tuple(int(m) for m in ms) for s, ms in manifest["egg_moves"].items()
        }
        self._tutor_moves = tuple(
            TutorMove(
                int(t["move"]),
                *(int(t["costs"][k]) for k in ("red", "blue", "yellow", "green")),
                str(t["location"]),
            )
            for t in manifest["tutor_moves"]
        )
        self._tutor_sets = {
            int(r): frozenset(int(i) for i in ix) for r, ix in manifest["tutor_learnsets"].items()
        }

    def level_up(self, species: int, form: int = 0) -> tuple[tuple[int, int], ...]:
        """The form's ``(level, move)`` pairs in learn order."""
        return self._level[form_row(self._form_rows, species, form)]

    def relearner_moves(self, species: int, level: int, form: int = 0) -> tuple[int, ...]:
        """Distinct moves learned at a level <= ``level``, first-occurrence order."""
        out: dict[int, None] = {}
        for lv, move in self.level_up(species, form):
            if lv <= level:
                out.setdefault(move)
        return tuple(out)

    def _tm_index(self, item: int) -> int | None:
        i = item - self._first_tm_item
        return i if 0 <= i < len(self._tm_moves) else None

    def tm_move(self, item: int) -> int:
        """The move a TM/HM item teaches; 0 for any other item."""
        i = self._tm_index(item)
        return 0 if i is None else self._tm_moves[i]

    def is_hm(self, item: int) -> bool:
        """Whether ``item`` is an HM (TM index >= 92); False for any other item."""
        i = self._tm_index(item)
        return i is not None and i >= HM_FIRST_INDEX

    def can_learn_tm(self, species: int, item: int, form: int = 0) -> bool:
        """Whether ``species``' ``form`` can learn the TM/HM ``item``; False for any other
        item."""
        row = form_row(self._form_rows, species, form)
        i = self._tm_index(item)
        return i is not None and i in self._tm[row]

    def evolutions(self, species: int) -> tuple[tuple[int, int, int], ...]:
        """``(method, param, target species)`` per evolution; ``method`` indexes the
        manifest's ``evolution_methods`` and ``param`` is the level, item, move or species
        the method needs, else 0. Keyed by species id (``LoadSpeciesEvolutions``,
        ``src/pokemon.c:4812``), so a form has the species' evolutions. A lookup only: prep
        does not use it (Ruling 12). ``ValueError`` for a species id out of range."""
        return self._evolutions.get(self._species_id(species), ())

    def egg_moves(self, species: int) -> tuple[int, ...]:
        """The species' egg moves in table order, empty for a species with none (keyed by
        species id; ``LoadSpeciesEggMoves``, ``src/overlay005/daycare.c:445``). A lookup only.
        ``ValueError`` for a species id out of range."""
        return self._egg_moves.get(self._species_id(species), ())

    def _species_id(self, species: int) -> int:
        """``species`` as an int; ``ValueError`` outside the species ids, as ``form_row``
        and ``Names`` raise."""
        species = int(species)
        if not 0 <= species < len(self._form_rows):
            raise ValueError(f"species {species} out of range")
        return species

    def tutor_moves(self) -> tuple[TutorMove, ...]:
        """The move tutors' table in ``sTeachableMoves`` order. A lookup only."""
        return self._tutor_moves

    def can_learn_tutor(self, species: int, move: int, form: int = 0) -> bool:
        """Whether a move tutor can teach ``move`` to the form. Keyed by personal row
        (``Pokemon_ReadMovesetMaskByte``, ``src/overlay005/scrcmd_move_tutor.c:211-259``). A
        lookup only: prep does not use it (Ruling 12)."""
        row = form_row(self._form_rows, species, form)
        return any(
            t.move == move and i in self._tutor_sets[row] for i, t in enumerate(self._tutor_moves)
        )


def load_learnsets(dir: Path = tables.DEFAULT) -> Learnsets:
    return Learnsets(tables._read_manifest(dir))
