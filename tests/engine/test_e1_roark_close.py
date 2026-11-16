"""E1 Roark close batch (rulings G-35 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

Family 1: the battle terrain of the Roark split's fights whose moves read it (G-35, G-36).
Family 2: AI_FLAG_HARRASSMENT (G-37, G-38).
"""

from __future__ import annotations

import pytest

from nuzline.data.trainer import TERRAIN_NAMES
from nuzline.engine.coverage import TERRAIN_EFFECTS, terrain_refusal

# --- Family 1: battle terrain (G-35, G-36) -------------------------------------------------

# Kaizo trainer ids -> the terrain tools/terrain.py narrows their battle to (G-35): Barry #2
# (247-249) starts in the Route 203 coord event's rectangle (Route203_CoordEvent_Rival), Lass
# Kaitlin (323) on her sight and talk tiles, Camper Zackary (377, Gardenia split) on his.
NARROWED = {
    247: "TERRAIN_PLAIN",
    248: "TERRAIN_PLAIN",
    249: "TERRAIN_PLAIN",
    323: "TERRAIN_GRASS",
    377: "TERRAIN_MOUNTAIN",
}


def _moves(trainer) -> list[int]:
    return [m for mon in trainer.party for m in (mon.moves or ()) if m]


@pytest.mark.parametrize("tid, name", sorted(NARROWED.items()))
def test_roark_split_terrain_readers_have_their_narrowed_terrain(
    kaizo_tables, kaizo_names, kaizo_trainers, tid, name
):
    t = next(x for x in kaizo_trainers if x.id == tid)
    assert TERRAIN_NAMES[t.terrain] == name and t.terrains == (TERRAIN_NAMES.index(name),)
    moves = _moves(t)
    effects = {int(kaizo_tables.move_ai_effect[m]) for m in moves}
    assert effects & set(TERRAIN_EFFECTS)  # Secret Power reads it (coverage.TERRAIN_EFFECTS)
    assert terrain_refusal(kaizo_tables, kaizo_names, t, moves) is None
