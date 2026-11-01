"""Level-up learnsets and TM/HM compatibility (host-only lookups from ``manifest.json``).

Level-up learnsets are ``res/pokemon/<species>/data.json`` ``learnset.by_level``. TM/HM
compatibility is ``learnset.by_tm``, numbered as tools/dataproc/src/speciesproc.c
ProcTMLearnset sets the mask bits (TM01-TM92 = 0-91, HM01-HM08 = 92-99); the move an
item teaches is ``teachesMove`` in res/items/data/tm10.json etc. (src/item.c).
"""

from __future__ import annotations

import json
from pathlib import Path

from nuzline.data import tables


class Learnsets:
    def __init__(self, manifest: dict):
        self._level = {
            int(s): tuple((int(lv), int(mv)) for lv, mv in pairs)
            for s, pairs in manifest["level_learnsets"].items()
        }
        self._tm = {int(s): frozenset(ix) for s, ix in manifest["tm_learnsets"].items()}
        self._tm_moves = tuple(int(m) for m in manifest["tm_moves"])
        self._first_tm_item = int(manifest["first_tm_item"])

    def level_up(self, species: int) -> tuple[tuple[int, int], ...]:
        """The species' ``(level, move)`` pairs in learn order."""
        return self._level[species]

    def relearner_moves(self, species: int, level: int) -> tuple[int, ...]:
        """Distinct moves learned at a level <= ``level``, first-occurrence order."""
        out: dict[int, None] = {}
        for lv, move in self._level[species]:
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

    def can_learn_tm(self, species: int, item: int) -> bool:
        """Whether ``species`` (form 0's data) can learn the TM/HM ``item``; False for
        any other item."""
        i = self._tm_index(item)
        return i is not None and i in self._tm[species]


def load_learnsets(dir: Path = tables.DEFAULT) -> Learnsets:
    return Learnsets(json.loads((Path(dir) / "manifest.json").read_text(encoding="utf-8")))
