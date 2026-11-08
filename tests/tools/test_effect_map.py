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


# move -> (EffectId, params) in every game that has the move with that decomp effect.
_TOXIC = (EffectId.STATUS_FOE, (6, 0, 0))  # Status.TOXIC; Toxic is not type-checked
_PLAIN = (EffectId.PLAIN, (0, 0, 0))
_VANILLA_MAPPED = {
    "Toxic": _TOXIC,
    "Swift": _PLAIN,
    "Shock Wave": _PLAIN,
    "Pay Day": _PLAIN,
    "Gust": _PLAIN,
}
_KAIZO_MAPPED = {
    **_VANILLA_MAPPED,
    "Sand Tomb": (EffectId.TRAP, (0, 0, 0)),  # Kaizo effect 261 (Whirlpool's script)
    "Aurora Beam": _PLAIN,  # Kaizo effect 17; vanilla's Aurora Beam lowers Attack
}


def _check_mapped(tables, names, vanilla_names, expected):
    from dataclasses import replace

    from nuzline.engine.coverage import Support
    from nuzline.prep.pool import Candidate, check_pool
    from tests.engine.helpers import sample_party

    for move, (eid, params) in expected.items():
        m = names.move(move)
        assert int(tables.move_effect[m]) == eid, move
        assert tuple(int(x) for x in tables.move_param[m]) == params, move
    moves = list(expected)
    for i in range(0, len(moves), 4):
        chunk = moves[i : i + 4]
        ids = [names.move(m) for m in chunk]
        spec = replace(sample_party(vanilla_names, {})[0], moves=tuple(ids + [0] * (4 - len(ids))))
        cand = Candidate(("party", 0, 0), "P", spec, 0, (), ())
        report = check_pool(tables, names, [cand])
        assert [ln for ln in report.missing if "move effect" in ln] == [], chunk


def test_mapping_only_moves(tables, names, kaizo_tables, kaizo_names):
    """Toxic, Swift, Shock Wave, Pay Day, Gust (and Kaizo's Sand Tomb and Aurora Beam) map to
    an effect the engine has, and the pool grades their move rows EXACT."""
    _check_mapped(tables, names, names, _VANILLA_MAPPED)
    _check_mapped(kaizo_tables, kaizo_names, names, _KAIZO_MAPPED)


def test_mapping_only_scripts_do_what_the_effect_does():
    """The decomp scripts behind the mapping-only effects carry nothing the target effect lacks."""
    s17, s34, s149 = _script(17), _script(34), _script(149)
    assert re.findall(r"^\s+([A-Z]\w+)", s17, re.M) == ["CalcCrit", "CalcDamage", "End"]
    # Pay Day: a damage hit plus the money subscript only
    assert "MOVE_SUBSCRIPT_PTR_PAY_DAY" in s34 and "CalcDamage" in s34
    assert _script(33).count("UpdateVar") == 1 and "MOVE_SUBSCRIPT_PTR_BADLY_POISON" in _script(33)
    # Sand Tomb: Whirlpool's script is Bind's (effect 42) plus the dive double
    s42, s261 = _script(42), _script(261)
    assert "MOVE_SUBSCRIPT_PTR_BIND_TARGET" in s42 and "MOVE_SUBSCRIPT_PTR_BIND_TARGET" in s261
    assert "SYSCTL_HIT_DURING_DIVE" in s261 and "MOVE_EFFECT_UNDERWATER" in s261
    # Gust: x2 only against an airborne defender
    assert "MOVE_EFFECT_AIRBORNE" in s149 and "BTLVAR_POWER_MULTI, 20" in s149


def test_gust_note_holds(tables, names, kaizo_tables, kaizo_names):
    """Gust's x2 against Fly/Bounce (effect 149) and Whirlpool's against Dive (effect 261) are
    unreachable only while the moves that set MOVE_EFFECT_AIRBORNE / MOVE_EFFECT_UNDERWATER stay
    unsupported: Fly (155), Dive (255) and Bounce (263). Kaizo rewrites Dive to a plain hit
    and Bounce to a Paralyze-hit (no semi-invulnerable turn). When one becomes supported, the
    double is reachable, and this test fails."""
    assert not {155, 255, 263} & set(DECOMP_EFFECT_TO_ID)
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Fly")]) == EffectId.UNSUPPORTED
    assert int(tables.move_effect[names.move("Dive")]) == EffectId.UNSUPPORTED
    assert int(tables.move_effect[names.move("Bounce")]) == EffectId.UNSUPPORTED
    for move in ("Dive", "Bounce"):
        scripts = {_script(e) for e in (0, 6)}  # Kaizo's Dive and Bounce effects
        assert all("MOVE_EFFECT_AIRBORNE" not in s and "UNDERWATER" not in s for s in scripts)
        assert int(kaizo_tables.move_effect[kaizo_names.move(move)]) != EffectId.UNSUPPORTED


def test_whirlpool_bind_matches_bind():
    """Sand Tomb runs the same bind subscript as Bind: duration Random 3, 3 (3..5), 6 with
    Grip Claw, and no dive state is reachable (Dive is unsupported)."""
    text = (DECOMP / "res/battle/scripts/subscripts/subscript_bind_start.s").read_text("utf-8")
    assert "Random 3, 3" in text and "0x00000006" in text
