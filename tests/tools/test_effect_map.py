"""Cross-check the hand-maintained effect map against the decomp's own sources."""

import json
import re
from pathlib import Path

import pytest

from nuzline.engine.ids import EffectId, MoveFlag, Stat
from tools.effect_map import (
    CERTAIN_SECONDARY_EFFECTS,
    CHARGE_FLAG_EFFECTS,
    DECOMP_EFFECT_NAMES,
    DECOMP_EFFECT_TO_ID,
    KAIZO_EFFECTS,
    KAIZO_FLAG_EFFECTS,
    KAIZO_THUNDER_EFFECTS,
    SCRIPT_FLAG_EFFECTS,
    SEMI_HIT_EFFECTS,
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
SECONDARY = (
    EffectId.SECONDARY_STAT_SELF,
    EffectId.SECONDARY_STAT_FOE,
    EffectId.SECONDARY_STATUS,
    EffectId.RAISE_ALL_STATS_HIT,
)


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
    for effect, (eid, (stat, stages, p2)) in DECOMP_EFFECT_TO_ID.items():
        if eid not in STAT_EFFECTS:
            continue
        name = DECOMP_EFFECT_NAMES[effect]
        if name == "BATTLE_EFFECT_EVA_UP_2_MINIMIZE":
            # B9-10: the direct TO_ATTACKER subscript_minimize.s sets the flag, then Evasion +1.
            assert "TO_ATTACKER|MOVE_SUBSCRIPT_PTR_MINIMIZE" in _script(effect)
            sub = DECOMP / "res/battle/scripts/subscripts/subscript_minimize.s"
            assert "MOVE_SUBSCRIPT_PTR_EVASION_UP_1_STAGE" in sub.read_text(encoding="utf-8")
            assert (stat, stages, p2) == (Stat.EVA, 1, 2)
            continue
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
    exactly the ones flagged move_param[2] = 1 (SECONDARY_STATUS, and RECOIL_STATUS since B10-1),
    or unsupported."""
    src = (DECOMP / "src/battle/battle_controller_player.c").read_text(encoding="utf-8")
    case = re.search(r"case CHECK_STATUS_STATE_SELF_THAW:(.*?)break;", src, re.DOTALL).group(1)
    thawing = set(re.findall(r"\bBATTLE_EFFECT_[A-Z_]+", case))
    assert thawing == {"BATTLE_EFFECT_THAW_AND_BURN_HIT", "BATTLE_EFFECT_RECOIL_BURN_HIT"}
    flagged = {
        DECOMP_EFFECT_NAMES[e]
        for e, (eid, params) in DECOMP_EFFECT_TO_ID.items()
        if eid in (EffectId.SECONDARY_STATUS, EffectId.RECOIL_STATUS) and params[2] == 1
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
        report = check_pool(tables, names, [cand], ())
        assert [ln for ln in report.missing if "move effect" in ln] == [], chunk


def test_mapping_only_moves(tables, names, kaizo_tables, kaizo_names):
    """Toxic, Swift, Shock Wave, Pay Day, Gust (and Kaizo's Sand Tomb and Aurora Beam) map to
    an effect the engine has, and the pool grades their move rows EXACT."""
    _check_mapped(tables, names, names, _VANILLA_MAPPED)
    _check_mapped(kaizo_tables, kaizo_names, names, _KAIZO_MAPPED)


def test_mapping_only_scripts_do_what_the_effect_does():
    """The decomp scripts behind the mapping-only effects carry nothing the target effect lacks."""
    s17, s34, s149 = _script(17), _script(34), _script(149)
    assert re.findall(r"^\s+([A-Z]\w+)", s17, re.MULTILINE) == ["CalcCrit", "CalcDamage", "End"]
    # Pay Day: a damage hit plus the money subscript only
    assert "MOVE_SUBSCRIPT_PTR_PAY_DAY" in s34 and "CalcDamage" in s34
    assert _script(33).count("UpdateVar") == 1 and "MOVE_SUBSCRIPT_PTR_BADLY_POISON" in _script(33)
    # Sand Tomb: Whirlpool's script is Bind's (effect 42) plus the dive double
    s42, s261 = _script(42), _script(261)
    assert "MOVE_SUBSCRIPT_PTR_BIND_TARGET" in s42 and "MOVE_SUBSCRIPT_PTR_BIND_TARGET" in s261
    assert "SYSCTL_HIT_DURING_DIVE" in s261 and "MOVE_EFFECT_UNDERWATER" in s261
    # Gust: x2 only against an airborne defender
    assert "MOVE_EFFECT_AIRBORNE" in s149 and "BTLVAR_POWER_MULTI, 20" in s149


def test_semi_invulnerable_flags(tables, names, kaizo_tables, kaizo_names):
    """B10-12: the charge moves that vanish (Fly 155, Bounce 263: AIRBORNE; Dig 256; Dive 255;
    Shadow Force 272) carry their SEMI_* bit, and the moves whose scripts set
    SYSCTL_HIT_DURING_* carry the HIT_* bits (with DOUBLE_SEMI for the x2 scripts), in both
    games by their ROM effect; each table entry is checked against its script's text."""
    for e in (155, 263, 256, 255, 272):
        assert "MOVE_EFFECT_" in _script(e) and "VANISH_CHARGE_TURN" in _script(e)
    for e, bits in SEMI_HIT_EFFECTS.items():
        text = _script(e)
        assert "SYSCTL_HIT_DURING_" in text
        assert ("BTLVAR_POWER_MULTI, 20" in text) == bool(bits & MoveFlag.DOUBLE_SEMI)
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        fly = int(t.move_flags[n.move("Fly")])
        assert fly & MoveFlag.CHARGE and fly & MoveFlag.SEMI_AIR
        gust = int(t.move_flags[n.move("Gust")])
        assert gust & MoveFlag.HIT_AIR and gust & MoveFlag.DOUBLE_SEMI
    assert int(tables.move_flags[names.move("Dive")]) & MoveFlag.SEMI_DIVE
    # Kaizo rewrites Dive and Bounce (effects 0 and 6): no charge.
    for move in ("Dive", "Bounce"):
        assert not int(kaizo_tables.move_flags[kaizo_names.move(move)]) & MoveFlag.CHARGE


@pytest.mark.fullset
def test_script_flag_effects_match_every_script():
    """B13-1, B13-2: every effect script that adds BTLVAR_CRITICAL_BOOSTS is HIGH_CRIT or carries
    MoveFlag.CRIT_UP (CHARGE_FLAG_EFFECTS, SCRIPT_FLAG_EFFECTS), and every SetMultiHit 2 is
    MULTI_HIT with 2 hits or MoveFlag.HIT_TWICE; each SCRIPT_FLAG_EFFECTS bit is in its script."""
    for e, bits in SCRIPT_FLAG_EFFECTS.items():
        text = _script(e)
        assert ("OPCODE_ADD, BTLVAR_CRITICAL_BOOSTS, 1" in text) == bool(bits & MoveFlag.CRIT_UP)
        assert ("SetMultiHit 2," in text) == bool(bits & MoveFlag.HIT_TWICE)
        assert e in DECOMP_EFFECT_TO_ID
    for path in sorted((DECOMP / "res/battle/scripts/effects").glob("effect_script_*.s")):
        e, text = int(path.stem[-4:]), path.read_text(encoding="utf-8")
        eid, params = DECOMP_EFFECT_TO_ID.get(e, (EffectId.UNSUPPORTED, (0, 0, 0)))
        flags = SCRIPT_FLAG_EFFECTS.get(e, 0) | CHARGE_FLAG_EFFECTS.get(e, 0)
        if "BTLVAR_CRITICAL_BOOSTS" in text and eid != EffectId.UNSUPPORTED:
            assert eid == EffectId.HIGH_CRIT or flags & MoveFlag.CRIT_UP, e
        if "SetMultiHit 2," in text and eid != EffectId.UNSUPPORTED:
            assert (eid == EffectId.MULTI_HIT and params[0] == 2) or flags & MoveFlag.HIT_TWICE, e


def test_triple_kick_and_beat_up_scripts():
    """B13-5, B13-9: every SetMultiHit 3 script is TRIPLE_KICK with its per-hit power step
    (effect_script_0104.s:5-7: + 10); Beat Up's script sets SYSCTL_IGNORE_IMMUNITIES and runs
    BtlCmd_BeatUp with no CalcDamage (effect_script_0154.s:5-7). Kaizo's 104 and 88 are
    TRIPLE_KICK with step 0 (the move's power, rom_diff.toml)."""
    for path in sorted((DECOMP / "res/battle/scripts/effects").glob("effect_script_*.s")):
        e, text = int(path.stem[-4:]), path.read_text(encoding="utf-8")
        if "SetMultiHit 3," in text:
            assert DECOMP_EFFECT_TO_ID[e] == (EffectId.TRIPLE_KICK, (10, 0, 0)), e
            assert "SYSCTL_TRIPLE_KICK" in text and "BTLVAR_MOVE_POWER, 10" in text, e
    text = _script(154)
    assert "SYSCTL_IGNORE_IMMUNITIES" in text and "BeatUp" in text and "CalcDamage" not in text
    assert DECOMP_EFFECT_TO_ID[154] == (EffectId.BEAT_UP, (0, 0, 0))
    for e in (88, 104):
        assert KAIZO_EFFECTS[e][1:] == (EffectId.TRIPLE_KICK, (0, 0, 0)), e
        assert e not in KAIZO_FLAG_EFFECTS


def test_kaizo_hurricane_flags_follow_its_notes():
    """B13-3, B13-4: Kaizo effect 141 is effect 76's script (SECONDARY_CONFUSE, as vanilla 76) plus
    SYSCTL_HIT_DURING_FLY; its two Thunder compares (MoveFlag.THUNDER) are E1's
    KAIZO_THUNDER_EFFECTS (G-50); the vanilla 141 is the unused plain script (UNSUPPORTED in
    vanilla)."""
    assert KAIZO_EFFECTS[141][1:] == DECOMP_EFFECT_TO_ID[76]
    assert KAIZO_FLAG_EFFECTS[141] == MoveFlag.HIT_AIR and 141 in KAIZO_THUNDER_EFFECTS
    assert set(KAIZO_FLAG_EFFECTS) <= set(KAIZO_EFFECTS)
    assert 141 not in DECOMP_EFFECT_TO_ID and "SYSCTL_HIT_DURING_FLY" in _script(152)


def test_whirlpool_bind_matches_bind():
    """Sand Tomb runs the same bind subscript as Bind: duration Random 3, 3 (3..5), 6 with
    Grip Claw, and no dive state is reachable (Dive is unsupported)."""
    text = (DECOMP / "res/battle/scripts/subscripts/subscript_bind_start.s").read_text("utf-8")
    assert "Random 3, 3" in text and "0x00000006" in text


_SMALL = {
    "Recover": (EffectId.RECOVER, (0, 0, 0)),
    "Fake Out": (EffectId.FAKE_OUT, (0, 0, 0)),
    "AncientPower": (EffectId.RAISE_ALL_STATS_HIT, (0, 0, 0)),
    "Curse": (EffectId.CURSE, (0, 0, 0)),
    "Double Kick": (EffectId.MULTI_HIT, (2, 0, 0)),
    "DoubleSlap": (EffectId.MULTI_HIT, (0, 0, 0)),  # vanilla: effect 29, the 2-5 law
}


def test_small_moves_mapped(tables, names, kaizo_tables, kaizo_names):
    """Task K3b1-7: Recover, Fake Out, AncientPower, Curse and the fixed two-hit family map in
    both games (Kaizo's DoubleSlap is effect 44, two hits), and the pool grades them EXACT."""
    _check_mapped(tables, names, names, _SMALL)
    kaizo = {**_SMALL, "DoubleSlap": (EffectId.MULTI_HIT, (2, 0, 0))}
    _check_mapped(kaizo_tables, kaizo_names, names, kaizo)


def _subscript(name: str) -> str:
    return (DECOMP / "res/battle/scripts/subscripts" / f"{name}.s").read_text("utf-8")


def _stat_steps(text: str) -> list[tuple[str, int]]:
    """The ``MOVE_SUBSCRIPT_PTR_<STAT>_<UP|DOWN>_1_STAGE`` params a subscript sets, in order."""
    return [
        (stat, 1 if d == "UP" else -1)
        for stat, d in re.findall(r"MOVE_SUBSCRIPT_PTR_([A-Z_]+?)_(UP|DOWN)_1_STAGE", text)
    ]


def test_small_move_scripts_do_what_the_engine_does():
    """The decomp scripts behind effects 32, 158, 140, 109 and 44 (engine/effects.py)."""
    from nuzline.engine.effects import ALL_STATS_RAISED, CURSE_STAGES

    # Recover: an ON_HIT direct effect; maxHP / 2; "HP is full" when cur == max.
    assert "FLAGS_DIRECT, MOVE_SIDE_EFFECT_ON_HIT|MOVE_SUBSCRIPT_PTR_RECOVER_HALF_MAX_HP" in (
        _script(32)
    )
    half = _subscript("subscript_recover_half_max_hp")
    assert "BATTLEMON_MAX_HP" in half and "DivideVarByValue BTLVAR_HP_CALC_TEMP, 2" in half
    assert "CompareMonDataToVar OPCODE_EQU, BTLSCR_MSG_TEMP, BATTLEMON_CUR_HP" in (
        _subscript("subscript_recover_hp")
    )
    # Fake Out: fails unless fakeOutTurnNumber == totalTurns, else an indirect flinch.
    s158 = _script(158)
    assert "OPCODE_NEQ, BTLSCR_ATTACKER, BATTLEMON_FAKE_OUT_TURN_NUMBER, BTLVAR_TOTAL_TURNS" in s158
    assert "FLAGS_INDIRECT, MOVE_SIDE_EFFECT_TO_DEFENDER|MOVE_SUBSCRIPT_PTR_FLINCH" in s158
    assert "MOVE_STATUS_FAILED" in s158
    # The flinch subscript names Inner Focus for Fake Out only (hooks.FLINCH_BLOCKED).
    flinch = _subscript("subscript_flinch_mon")
    assert "BTLVAR_CURRENT_MOVE, MOVE_FAKE_OUT" in flinch and "ABILITY_INNER_FOCUS" in flinch
    # AncientPower: indirect TO_ATTACKER, five +1 steps.
    assert "FLAGS_INDIRECT, MOVE_SIDE_EFFECT_TO_ATTACKER|MOVE_SUBSCRIPT_PTR_BOOST_ALL_STATS" in (
        _script(140)
    )
    boost = [
        (SUBSCRIPT_STAT[s], d) for s, d in _stat_steps(_subscript("subscript_boost_all_stats"))
    ]
    assert boost == [(stat, 1) for stat in ALL_STATS_RAISED]
    # Curse: a Ghost-type check on both types, else the direct TO_ATTACKER normal branch.
    s109 = _script(109)
    assert s109.count("TYPE_GHOST") == 2
    assert "FLAGS_DIRECT, MOVE_SIDE_EFFECT_TO_ATTACKER|MOVE_SUBSCRIPT_PTR_CURSE_NORMAL" in s109
    curse = [(SUBSCRIPT_STAT[s], d) for s, d in _stat_steps(_subscript("subscript_curse_normal"))]
    assert curse == list(CURSE_STAGES)
    # The two multi-hit scripts: SetMultiHit 0 (2-5) and SetMultiHit 2, accuracy once.
    assert "SetMultiHit 0, SYSCTL_MULTI_HIT_MOVE" in _script(29)
    assert "SetMultiHit 2, SYSCTL_MULTI_HIT_MOVE" in _script(44)
    assert DECOMP_EFFECT_TO_ID[29] == (EffectId.MULTI_HIT, (0, 0, 0))
    assert DECOMP_EFFECT_TO_ID[44] == (EffectId.MULTI_HIT, (2, 0, 0))


def test_self_destruct_script_does_what_the_engine_does(tables, names, kaizo_tables, kaizo_names):
    """Effect 7 (Task K3b1-8): the Damp check, then the user's HP set to 0, then the hit;
    the damage formula halves defenseStat for it; Selfdestruct and Explosion map in both
    games (Kaizo's Memento too)."""
    s7 = _script(7).splitlines()
    steps = [line.split()[0] for line in s7 if line.startswith("    ")]
    assert steps.index("CheckIgnorableAbility") < steps.index("UpdateMonData")
    assert steps.index("UpdateMonData") < steps.index("CalcCrit") < steps.index("CalcDamage")
    assert any("ABILITY_DAMP" in line for line in s7)
    assert any("UpdateMonData OPCODE_SET, BTLSCR_ATTACKER, BATTLEMON_CUR_HP, 0" in ln for ln in s7)
    lib = (DECOMP / "src/battle/battle_lib.c").read_text("utf-8").splitlines()
    assert "BATTLE_EFFECT_HALVE_DEFENSE" in lib[6942] and "defenseStat / 2" in lib[6943]
    assert DECOMP_EFFECT_TO_ID[7] == (EffectId.SELF_DESTRUCT, (0, 0, 0))
    for t, n, moves in (
        (tables, names, ("Selfdestruct", "Explosion")),
        (kaizo_tables, kaizo_names, ("Selfdestruct", "Explosion", "Memento")),
    ):
        for move in moves:
            assert int(t.move_effect[n.move(move)]) == EffectId.SELF_DESTRUCT, move


def test_flail_script_does_what_the_engine_does(tables, names, kaizo_tables, kaizo_names):
    """Effect 99 (Task K3b1-9): CalcFlailPower, then CalcCrit and CalcDamage; Flail and
    Reversal share it in both games."""
    steps = [line.split()[0] for line in _script(99).splitlines() if line.startswith("    ")]
    assert steps[:3] == ["CalcFlailPower", "CalcCrit", "CalcDamage"]
    assert DECOMP_EFFECT_TO_ID[99] == (EffectId.FLAIL, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for move in ("Flail", "Reversal"):
            assert int(t.move_effect[n.move(move)]) == EffectId.FLAIL, move


def test_resist_berry_hold_effects_match_decomp():
    """Names against generated/item_hold_effects.txt (line N+1 is id N) and each type
    against subscript_type_resist_berry.s: the jump for the hold effect lands on a compare
    of calcTemp with that type (34, Steel, compares NEQ and falls through to the halving)."""
    from nuzline.engine.ids import RESIST_ANY_NORMAL, Type
    from tools.effect_map import RESIST_BERRY_HOLD_EFFECTS

    lines = (DECOMP / "generated/item_hold_effects.txt").read_text().split()
    script = (DECOMP / "res/battle/scripts/subscripts/subscript_type_resist_berry.s").read_text(
        encoding="utf-8"
    )
    jumps = {int(e): label for e, label in re.findall(r"BTLVAR_SCRIPT_TEMP, (\d+), (_\d+)", script)}
    blocks = dict(
        re.findall(
            r"(_\d+):\n\s+CompareVarToValue OPCODE_(?:EQU|NEQ), BTLVAR_CALC_TEMP, (0x[0-9A-Fa-f]+)",
            script,
        )
    )
    assert sorted(RESIST_BERRY_HOLD_EFFECTS) == sorted(jumps) == list(range(19, 36))
    for effect, (name, resisted) in RESIST_BERRY_HOLD_EFFECTS.items():
        assert lines[effect] == name
        want = Type.NORMAL if resisted == RESIST_ANY_NORMAL else resisted
        assert int(blocks[jumps[effect]], 16) == want, name
    # Only Chilan (35) jumps before the super-effective test.
    assert script.index(", 35, ") < script.index("MOVE_STATUS_SUPER_EFFECTIVE")
    assert script.index(", 19, ") > script.index("MOVE_STATUS_SUPER_EFFECTIVE")


def test_ignore_type_checks_effects_match_decomp():
    """effects.IGNORES_TYPE_CHECKS (subscript_type_resist_berry.s:5 skips them) is every
    mapped effect whose script sets SYSCTL_IGNORE_TYPE_CHECKS or SYSCTL_IGNORE_IMMUNITIES.
    Present sets it on its heal path only (effect_script_0122.s _004), which deals no damage,
    so no resist-berry check runs there; its damage path keeps the berry (test_present)."""
    from nuzline.engine.effects import IGNORES_TYPE_CHECKS

    heal_path_only = {EffectId.PRESENT}

    flag = re.compile(
        r"FLAG_ON, BTLVAR_BATTLE_CTX_STATUS, .*SYSCTL_IGNORE_(TYPE_CHECKS|IMMUNITIES)"
    )
    for effect, (eid, _) in DECOMP_EFFECT_TO_ID.items():
        sets = bool(flag.search(_script(effect)))
        ignores = eid in IGNORES_TYPE_CHECKS or eid in heal_path_only
        assert sets == ignores, DECOMP_EFFECT_NAMES[effect]
        assert not (eid in heal_path_only and eid in IGNORES_TYPE_CHECKS)


def test_type_boost_hold_effects_match_decomp():
    """TYPE_BOOST_HOLD_EFFECTS is sTypeBoostingItems (battle_lib.c:6514-6549), hold effects by
    generated/item_hold_effects.txt and types by generated/pokemon_types.txt."""
    from tools.effect_map import TYPE_BOOST_HOLD_EFFECTS

    holds = (DECOMP / "generated/item_hold_effects.txt").read_text().split()
    types = (DECOMP / "generated/pokemon_types.txt").read_text().split()
    lib = (DECOMP / "src/battle/battle_lib.c").read_text(encoding="utf-8")
    table = lib[
        lib.index("sTypeBoostingItems[] = {") : lib.index("};", lib.index("sTypeBoostingItems"))
    ]
    pairs = re.findall(r"\{ (HOLD_EFFECT_\w+), (TYPE_\w+) \}", table)
    assert len(pairs) == 33
    want = {holds.index(h): (h, types.index(t)) for h, t in pairs}
    assert {h: (n, int(t)) for h, (n, t) in TYPE_BOOST_HOLD_EFFECTS.items()} == want


def test_hold_effect_ids_name_their_hold_effects():
    from tools.effect_map import HOLD_EFFECT_TO_ID

    holds = (DECOMP / "generated/item_hold_effects.txt").read_text().split()
    for effect, (name, _) in HOLD_EFFECT_TO_ID.items():
        assert holds[effect] == name


def test_held_item_ids_follow_the_data():
    """Every resist berry DECOMP_ITEM_TO_ID names has a resist hold effect, and every resist
    hold effect has its berry; the berries with no hold effect (NO_EFFECT_BERRY) are exactly
    the items with berryData (sBerryItemIDs, tools/dataproc/src/itemproc.c:272) and no hold
    effect."""
    from nuzline.engine.ids import ItemId
    from tools.effect_map import DECOMP_ITEM_TO_ID, RESIST_BERRY_HOLD_EFFECTS, held_item_ids

    items = json.loads(Path("data/intermediate/items.json").read_text(encoding="utf-8"))
    by_name = {i["name"]: i for i in items}
    resist = {n for n, e in DECOMP_ITEM_TO_ID.items() if e == ItemId.RESIST_BERRY}
    assert sorted(by_name[n]["hold_effect"] for n in resist) == sorted(RESIST_BERRY_HOLD_EFFECTS)
    data = [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted((DECOMP / "res/items/data").glob("*.json"))
    ]
    plain = {d["name"] for d in data if "berryData" in d and d["holdEffect"] == "HOLD_EFFECT_NONE"}
    ids = held_item_ids(items)
    assert {i["name"] for i, e in zip(items, ids) if e == ItemId.NO_EFFECT_BERRY} == plain


def test_batch2_berries_follow_their_item_data():
    """The berries E2 batch 2 names (DECOMP_ITEM_TO_ID: Pluck reads an item's pluck effect, not
    its hold effect) carry the hold and pluck effects the engine gives them
    (res/items/data/*.json), and PINCH_STAT_HOLD_EFFECTS names its hold effects
    (generated/item_hold_effects.txt) with the stat BattleSystem_TriggerHeldItem raises
    (battle_lib.c:4773-4836: msgTemp = BATTLE_STAT_*)."""
    from nuzline.engine.ids import ItemId, Stat
    from tools.effect_map import DECOMP_ITEM_TO_ID, PINCH_STAT_HOLD_EFFECTS

    want = {
        ItemId.CHERI_BERRY: ("HOLD_EFFECT_PRZ_RESTORE", "PLUCK_EFFECT_PRZ_RESTORE"),
        ItemId.RAWST_BERRY: ("HOLD_EFFECT_BRN_RESTORE", "PLUCK_EFFECT_BRN_RESTORE"),
        ItemId.ASPEAR_BERRY: ("HOLD_EFFECT_FRZ_RESTORE", "PLUCK_EFFECT_FRZ_RESTORE"),
        ItemId.PERSIM_BERRY: ("HOLD_EFFECT_CONFUSE_RESTORE", "PLUCK_EFFECT_CNF_RESTORE"),
        ItemId.CUSTAP_BERRY: ("HOLD_EFFECT_PINCH_PRIORITY", "PLUCK_EFFECT_NONE"),
    }
    pluck_up = {
        Stat.ATK: "ATK",
        Stat.DEF: "DEF",
        Stat.SPE: "SPEED",
        Stat.SPA: "SPATK",
        Stat.SPD: "SPDEF",
    }
    holds = (DECOMP / "generated/item_hold_effects.txt").read_text().split()
    data = {
        d["name"]: d
        for d in (
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted((DECOMP / "res/items/data").glob("*.json"))
        )
    }
    pinch = 0
    for name, eid in DECOMP_ITEM_TO_ID.items():
        if eid == ItemId.PINCH_BERRY:
            pinch += 1
            hold = holds.index(data[name]["holdEffect"])
            stat = PINCH_STAT_HOLD_EFFECTS[hold][1]
            assert data[name]["pluckEffect"] == f"PLUCK_EFFECT_{pluck_up[stat]}_UP", name
            assert "berryData" in data[name], name
        elif eid in want:
            assert (data[name]["holdEffect"], data[name]["pluckEffect"]) == want[eid], name
            assert "berryData" in data[name], name
    assert pinch == len(PINCH_STAT_HOLD_EFFECTS) == 5
    lib = (DECOMP / "src/battle/battle_lib.c").read_text(encoding="utf-8")
    trigger = lib[lib.index("BOOL BattleSystem_TriggerHeldItem(") :]
    stat_names = {Stat.ATK: "ATTACK", Stat.DEF: "DEFENSE", Stat.SPE: "SPEED"}
    stat_names |= {Stat.SPA: "SP_ATTACK", Stat.SPD: "SP_DEFENSE"}
    for hold, (name, stat) in PINCH_STAT_HOLD_EFFECTS.items():
        assert holds[hold] == name
        case = trigger[trigger.index(f"case {name}:") :]
        assert f"msgTemp = BATTLE_STAT_{stat_names[stat]};" in case[: case.index("break;")]


def test_pinch_berries_take_any_param(tables, names, kaizo_tables, kaizo_names):
    """A stat pinch berry or Custap Berry with a param below 2 divides maxHP by zero
    (battle_lib.c:4778, :1294; Gluttony halves the param), which the ROM's ``_u32_div_f``
    answers with the dividend (``order.pinch_limit``, G-53): Kaizo's Ganlon and Apicot (param 0)
    stay PINCH_BERRY."""
    from nuzline.engine.ids import ItemId

    for berry in ("Liechi Berry", "Ganlon Berry", "Salac Berry", "Petaya Berry", "Apicot Berry"):
        assert int(tables.item_effect[names.item(berry)]) == ItemId.PINCH_BERRY, berry
    assert int(tables.item_effect[names.item("Custap Berry")]) == ItemId.CUSTAP_BERRY
    for berry in ("Ganlon Berry", "Apicot Berry"):
        i = kaizo_names.item(berry)
        assert int(kaizo_tables.item_hold_param[i]) == 0
        assert int(kaizo_tables.item_effect[i]) == ItemId.PINCH_BERRY, berry


def test_divisor_items_need_a_param_to_divide_by():
    """Sticky Barb (``Divide(maxHP, param)``, battle_lib.c:5330-5336) and Shell Bell
    (``Divide(dealt, param)``, bcp.c:4736-4752) with param 0 are UNSUPPORTED
    (build_tables._pinch_divisor_ok, effect_map.DIVISOR_PARAM_MIN); a pinch berry is not (G-53)."""
    from nuzline.engine.ids import ItemId
    from tools.build_tables import _pinch_divisor_ok

    ids = [ItemId.STICKY_BARB, ItemId.SHELL_BELL, ItemId.STICKY_BARB, ItemId.PINCH_BERRY]
    items = [{"hold_param": p} for p in (0, 0, 8, 0)]
    assert _pinch_divisor_ok(items, [int(i) for i in ids]) == [
        ItemId.UNSUPPORTED,
        ItemId.UNSUPPORTED,
        ItemId.STICKY_BARB,
        ItemId.PINCH_BERRY,
    ]


def _function(path: str, name: str) -> str:
    """The body of the C function whose definition starts with ``name`` (not a declaration)."""
    src = (DECOMP / path).read_text(encoding="utf-8")
    start = re.search(re.escape(name) + r"[^;{]*\n\{", src).end()
    return src[start : src.index("\n}\n", start)]


def test_invoker_id_lists_match_the_decomp():
    """E2 batch 5: the move ids BtlCmd_TrySleepTalk, Move_IsInvoker, Move_IsMultiTurn and
    sCannotMetronomeMoves compare against (``tools.effect_map``)."""
    from tools.effect_map import (
        INVOKER_MOVES,
        METRONOME_EXCLUDED_MOVES,
        MULTI_TURN_EFFECTS,
        SLEEP_TALK_EXCLUDED_MOVES,
    )

    moves = (DECOMP / "generated/moves.txt").read_text().split()
    effects = (DECOMP / "generated/move_battle_effects.txt").read_text().split()
    invoker = _function("src/battle/battle_lib.c", "BOOL Move_IsInvoker(")
    assert sorted(moves.index(m) for m in re.findall(r"move == (MOVE_\w+)", invoker)) == sorted(
        INVOKER_MOVES
    )
    talk = _function("src/battle/battle_script.c", "static BOOL BtlCmd_TrySleepTalk(")
    listed = re.findall(r"ATTACKING_MON\.moves\[i\] == (MOVE_\w+)", talk)
    assert sorted(moves.index(m) for m in listed) == sorted(SLEEP_TALK_EXCLUDED_MOVES)
    multi = _function("src/battle/battle_lib.c", "BOOL Move_IsMultiTurn(")
    cases = re.findall(r"case (BATTLE_EFFECT_\w+):", multi)
    assert sorted(effects.index(e) for e in cases) == sorted(MULTI_TURN_EFFECTS)
    lib = (DECOMP / "src/battle/battle_lib.c").read_text(encoding="utf-8")
    table = lib[lib.index("sCannotMetronomeMoves[] = {") :]
    table = table[: table.index("FORBIDDEN_BY_METRONOME_DELIM,")]
    listed = [m for m in re.findall(r"(MOVE_\w+),", table)]
    assert [moves.index(m) for m in listed] == list(METRONOME_EXCLUDED_MOVES)
