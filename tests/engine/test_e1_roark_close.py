"""E1 Roark close batch (rulings G-35 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

Family 1: the battle terrain of the Roark split's fights whose moves read it (G-35, G-36).
Family 2: AI_FLAG_HARRASSMENT (G-37).
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


# --- Family 2: AI_FLAG_HARRASSMENT (G-37) ----------------------------------------------------

LEER, THUNDER_WAVE = 43, 86  # Kaizo ids: DEF_DOWN (19), STATUS_PARALYZE (67)


def test_harrassment_effects_are_the_decomps():
    """``commands.HARRASSMENT_EFFECTS`` is Harrassment_Effects (script.s:8036-8071) entry by
    entry, through generated/move_battle_effects.txt (line n + 1 is id n)."""
    from nuzline.ai import commands as c
    from tests.ai.test_kaizo_policy import DECOMP

    script = DECOMP / "src/battle/trainer_ai/script.s"
    ids = DECOMP / "generated/move_battle_effects.txt"
    if not script.exists() or not ids.exists():
        pytest.skip("decomp not present")
    names = ids.read_text(encoding="utf-8").split()
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[8035] == "Harrassment_Effects:"
    assert lines[8029].split() == ["IfRandomLessThan", "128,", "Harrassment_Terminate"]
    flag_table = lines.index("FlagTable:")
    assert "Harrassment_Main" in lines[flag_table + 1 + 10]  # AI_FLAG_HARRASSMENT, bit 10
    entries = []
    for line in lines[8036:]:
        name = line.split()[1]
        if name == "TABLE_END":
            break
        entries.append(names.index(name))
    assert tuple(entries) == c.HARRASSMENT_EFFECTS


@pytest.mark.parametrize("jump", [0, 1])
def test_harrassment(kaizo_tables, kaizo_roark_party, jump):
    """Harrassment_Main (script.s:8021-8034): Leer (DEF_DOWN, :8039) and Thunder Wave
    (STATUS_PARALYZE, :8048) get +2 unless the 128 roll (:8030) jumps; Stealth Rock and Rock
    Polish (effects outside the table) neither roll nor score. Against BASIC, EVAL_ATTACK and
    EXPERT (set 0x7 -> 0x407); exact equals loose under the new set."""
    from tests.ai.test_kaizo_outcome import _assert_exact_equals_loose, layout_for
    from tests.ai.test_kaizo_policy import (
        BONSLY,
        ROCK_POLISH,
        STEALTH_ROCK,
        _delta,
        _path_lines,
        _state,
    )

    t = kaizo_tables
    effect = [int(t.move_ai_effect[m]) for m in (LEER, THUNDER_WAVE)]
    assert effect == [19, 67]
    s = _state(t, kaizo_roark_party, BONSLY, (STEALTH_ROCK, LEER, THUNDER_WAVE, ROCK_POLISH))
    jumps = ((1, 8030), (2, 8030)) if jump else ()
    plus = 0 if jump else 2
    assert _delta(t, s, 0x7, 0x407, jumps) == [0, plus, plus, 0]
    rolls = [8030 in lines for lines in _path_lines(t, s, layout_for(t, 0x407, s))]
    assert rolls == [False, True, True, False]
    _assert_exact_equals_loose(t, s, layout_for(t, 0x407, s))
