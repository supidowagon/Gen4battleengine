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
    TYPE_CHART_STATUS_MOVES,
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


def _move_names(text: str) -> set[str]:
    """English names of the MOVE_* constants in a piece of decomp C source."""
    names = set()
    for const in re.findall(r"\bMOVE_([A-Z0-9_]+)", text):
        data = DECOMP / "res/moves" / const.lower() / "data.json"
        names.add(json.loads(data.read_text(encoding="utf-8"))["name"])
    return names


def test_type_checked_status_moves_are_flagged():
    """Status moves skip the type chart (power 0) except the moves the decomp names
    in BattleControllerPlayer_CheckTypeChart; exactly those are type-checked."""
    src = (DECOMP / "src/battle/battle_controller_player.c").read_text(encoding="utf-8")
    body = re.search(
        r"static int BattleControllerPlayer_CheckTypeChart\([^;{]*\)\s*\{(.*?)\n\}", src, re.DOTALL
    ).group(1)
    special = _move_names(body.split("BattleSystem_ApplyTypeChart")[0])
    assert special == TYPE_CHART_STATUS_MOVES == {"Thunder Wave"}


def test_status_subscript_type_immunities():
    """The only target-type checks in the status subscripts are the ones
    engine/effects.py applies; sleep and paralysis have none."""
    expected = {
        "subscript_fall_asleep": set(),
        "subscript_paralyze": set(),
        "subscript_poison": {"POISON", "STEEL"},
        "subscript_burn": {"FIRE"},
        "subscript_freeze": {"ICE"},
    }
    for name, types in expected.items():
        text = (DECOMP / "res/battle/scripts/subscripts" / f"{name}.s").read_text(encoding="utf-8")
        assert set(re.findall(r"BATTLEMON_TYPE_[12], TYPE_([A-Z]+)", text)) == types, name


def test_thaw_flag_matches_the_self_thaw_check():
    """The effects the decomp lets a frozen user move with and thaws
    (CHECK_STATUS_STATE_SELF_THAW in BattleControllerPlayer_CheckStatusDisruption) are
    exactly the ones flagged move_param[2] = 1, or unsupported."""
    src = (DECOMP / "src/battle/battle_controller_player.c").read_text(encoding="utf-8")
    case = re.search(r"case CHECK_STATUS_STATE_SELF_THAW:(.*?)break;", src, re.DOTALL).group(1)
    thawing = set(re.findall(r"\bBATTLE_EFFECT_[A-Z_]+", case))
    assert thawing == {"BATTLE_EFFECT_THAW_AND_BURN_HIT", "BATTLE_EFFECT_RECOIL_BURN_HIT"}
    flagged = {
        DECOMP_EFFECT_NAMES[e]
        for e, (eid, params) in DECOMP_EFFECT_TO_ID.items()
        if eid == EffectId.SECONDARY_STATUS and params[2] == 1
    }
    assert flagged <= thawing
    assert thawing - flagged <= thawing - set(DECOMP_EFFECT_NAMES.values())
