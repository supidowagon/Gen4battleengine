"""Cross-check the hand-maintained effect map against the decomp's own sources."""

import json
import re
from pathlib import Path

import pytest

from nuzline.engine.ids import EffectId, Stat
from tools.effect_map import (
    CERTAIN_SECONDARY_EFFECTS,
    DECOMP_EFFECT_NAMES,
    DECOMP_EFFECT_TO_ID,
    PUNCH_MOVES,
    SOUND_MOVES,
)

DECOMP = Path("external/pokeplatinum")
pytestmark = pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")

SUBSCRIPT_STAT = {
    "ATTACK": Stat.ATK,
    "DEFENSE": Stat.DEF,
    "SPEED": Stat.SPE,
    "SP_ATTACK": Stat.SPA,
    "SP_DEFENSE": Stat.SPD,
    "ACCURACY": Stat.ACC,
    "EVASION": Stat.EVA,
}
# EffectId -> (side-effect flag kind, recipient) the decomp script must set.
STAT_EFFECTS = {
    EffectId.STAT_SELF: ("DIRECT", "TO_ATTACKER"),
    EffectId.STAT_FOE: ("DIRECT", "TO_DEFENDER"),
    EffectId.SECONDARY_STAT_SELF: ("INDIRECT", "TO_ATTACKER"),
    EffectId.SECONDARY_STAT_FOE: ("INDIRECT", "TO_DEFENDER"),
}
SECONDARY = (EffectId.SECONDARY_STAT_SELF, EffectId.SECONDARY_STAT_FOE, EffectId.SECONDARY_STATUS)


def _script(effect: int) -> str:
    path = DECOMP / "res/battle/scripts/effects" / f"effect_script_{effect:04d}.s"
    return path.read_text(encoding="utf-8")


def test_effect_names_match_decomp_ids():
    lines = (DECOMP / "generated/move_battle_effects.txt").read_text().split()
    assert set(DECOMP_EFFECT_NAMES) == set(DECOMP_EFFECT_TO_ID)
    for effect, name in DECOMP_EFFECT_NAMES.items():
        assert lines[effect] == name


def test_stat_effects_match_decomp_scripts():
    pattern = re.compile(
        r"BTLVAR_SIDE_EFFECT_FLAGS_(DIRECT|INDIRECT),.*MOVE_SIDE_EFFECT_(TO_ATTACKER|TO_DEFENDER)"
        r"\|MOVE_SUBSCRIPT_PTR_([A-Z_]+?)_(UP|DOWN)_([12])_STAGES?\b"
    )
    for effect, (eid, (stat, stages, _)) in DECOMP_EFFECT_TO_ID.items():
        if eid not in STAT_EFFECTS:
            continue
        name = DECOMP_EFFECT_NAMES[effect]
        m = pattern.search(_script(effect))
        assert m, name
        kind, who, sub_stat, direction, n = m.groups()
        assert (kind, who) == STAT_EFFECTS[eid], name
        assert SUBSCRIPT_STAT[sub_stat] == stat, name
        assert stages == int(n) * (1 if direction == "UP" else -1), name


def test_certain_secondaries_are_on_hit():
    """A secondary effect is certain (chance column 0) exactly when its decomp
    script sets MOVE_SIDE_EFFECT_ON_HIT."""
    for effect, (eid, _) in DECOMP_EFFECT_TO_ID.items():
        if eid in SECONDARY:
            on_hit = "MOVE_SIDE_EFFECT_ON_HIT" in _script(effect)
            assert on_hit == (effect in CERTAIN_SECONDARY_EFFECTS), DECOMP_EFFECT_NAMES[effect]


def _c_move_names(array: str) -> set[str]:
    """English names of the MOVE_* constants listed in a battle_lib.c array."""
    src = (DECOMP / "src/battle/battle_lib.c").read_text(encoding="utf-8")
    body = re.search(array + r"\[\]\s*=\s*\{(.*?)\};", src, re.DOTALL).group(1)
    names = set()
    for const in re.findall(r"MOVE_([A-Z0-9_]+)", body):
        data = DECOMP / "res/moves" / const.lower() / "data.json"
        names.add(json.loads(data.read_text(encoding="utf-8"))["name"])
    return names


def test_sound_and_punch_lists_match_decomp():
    assert SOUND_MOVES == _c_move_names("sSoundMoves")
    assert PUNCH_MOVES == _c_move_names("sPunchingMoves")
