"""Every hard-coded game id is classified and holds in each game's data (spec §7.2)."""

import ast
import importlib
import importlib.util
import json
import re
import shutil
from collections import defaultdict
from enum import IntEnum, IntFlag
from pathlib import Path

import pytest

from nuzline.data.game_constants import (
    CONSTANTS,
    EXEMPT,
    GAME_DATA,
    KINDS,
    POLICY,
    SCOPES,
    check_constants,
    message_path,
)
from nuzline.data.games import GAMES, KAIZO, VANILLA, Game, GameDataError, load_game
from nuzline.data.tables import Names
from tests.conftest import TABLES_DIR, _require_tables

# The plan's classification table (Task 4), plus the pre-flight additions (#4, #22) and
# the coverage.py ids classified alongside the Ruling 14 refusal.
LISTED = [
    # engine, asserted by name
    "nuzline.ai.commands.ABILITY_IDS",
    "nuzline.ai.commands.TYPE_IDS",
    "nuzline.ai.switching.ABILITY_IDS",
    "nuzline.ai.switching._TYPE_FLYING",
    "nuzline.ai.switching._TYPE_STEEL",
    "nuzline.engine.ids.Type",
    "nuzline.engine.ids.MOVE_STRUGGLE",
    "nuzline.engine.ids.WEATHER_ABILITY_IDS",
    "nuzline.engine.ids.ITEM_ABILITY_IDS",
    "nuzline.engine.ids.YAWN_ABILITY_IDS",
    "nuzline.engine.ids.STAT_MOVE_ABILITY_IDS",
    "nuzline.engine.ids.MOVE_HEAL_BELL",
    "nuzline.engine.ids.ABILITY_MULTITYPE",
    "nuzline.engine.ids.ABILITY_DECOMP_IDS",
    "nuzline.engine.ids.UNTRACEABLE_ABILITIES",
    "nuzline.ai.commands.CHANGE_USER_ABILITY_DESIRABLE",
    "nuzline.engine.ids.ABILITY_FORECAST",
    "nuzline.engine.ids.ITEM_GRISEOUS_ORB",
    "nuzline.engine.ids.TERRAIN_MOVES",
    "nuzline.engine.stats.SPECIES_SHEDINJA",
    "nuzline.ai.switching.MOVE_STRUGGLE",
    "nuzline.prep.problem.HEART_SCALE",
    "nuzline.engine.coverage.SPECIES_GIRATINA",
    "nuzline.engine.coverage.SPECIES_SHAYMIN",
    "nuzline.engine.coverage.ITEM_GRISEOUS_ORB",
    # engine, asserted by name and hold effect / hold param
    "nuzline.ai.commands.ITEM_EXPERT_BELT",
    "nuzline.ai.commands.NATURAL_GIFT_BERRIES",
    "nuzline.ai.commands.RECYCLE_DESIRABLE_ITEMS",
    "nuzline.ai.commands.ASSURANCE_RECOIL_BERRIES",
    "nuzline.engine.ids.ITEM_FULL_RESTORE",
    "nuzline.engine.ids.QUICK_CLAW_PARAM",
    # engine: saves/check.py's special routes (final review M12)
    "nuzline.saves.check.SPECIES_SMEARGLE",
    "nuzline.saves.check.SPECIES_PICHU",
    "nuzline.saves.check.SPECIES_ROTOM",
    "nuzline.saves.check.TYPE_DRAGON",
    "nuzline.saves.check.MOVE_DRACO_METEOR",
    "nuzline.saves.check.MOVE_VOLT_TACKLE",
    "nuzline.saves.check.ROTOM_FORM_MOVES",
    "nuzline.saves.check.ULTIMATE_MOVES",
    # vanilla_policy
    "nuzline.ai.commands.MOVE_EFFECTS",
    "nuzline.ai.commands.MOVE_IDS",
    "nuzline.ai.commands.SOUND_MOVES",
    "nuzline.ai.switching.MOVE_IDS",
    "nuzline.ai.switching.PLAYER_ABILITY_REFUSALS",
    "nuzline.ai.outcome._SITES",
    "nuzline.ai.commands.CALC_DAMAGE_MOVES",
    "nuzline.ai.commands.MOVE_TYPE_MOVES",
    "nuzline.ai.commands.COPYCAT_ENCOURAGED_MOVES",
    # vanilla_policy / kaizo_policy: script.s:78's compare (the Dry Skin word)
    "nuzline.ai.layout.VANILLA",
    "nuzline.ai.layout.KAIZO",
    # EXEMPT: engine encodings
    "nuzline.engine.ids.MoveTarget",
    "nuzline.engine.ids.EffectId",
    "nuzline.engine.ids.Status",
    "nuzline.engine.ids.Stat",
    "nuzline.engine.ids.MoveFlag",
    "nuzline.engine.ids.Weather",
    "nuzline.engine.ids.ItemId",
    "nuzline.engine.ids.ItemParam",
    "nuzline.engine.ids.ItemAiCategory",
    "nuzline.engine.ids.AbilityId",
    "nuzline.engine.ids.FlingEffect",
    "nuzline.engine.ids.GRAVITY_MOVES",  # EXEMPT: code move ids (_CODE_MOVE_IDS)
    "nuzline.engine.ids.MIMIC_FORBIDDEN",  # EXEMPT: code move ids (_CODE_MOVE_IDS)
    "nuzline.ai.layout.COPIER_EFFECTS",  # EXEMPT: keyed by decomp effect id
    "nuzline.engine.ids.CHATTER_CHANCES",  # EXEMPT: Chatter's chance per cry parameter
    # EXEMPT: decomp AI-script encodings
    "nuzline.ai.commands.Battler",
    "nuzline.ai.commands.LoadType",
    "nuzline.ai.commands.SideCondition",
    "nuzline.ai.commands.MonCondition",
    "nuzline.ai.commands.FieldCondition",
    "nuzline.ai.commands.AiWeather",
    "nuzline.ai.commands.VolatileCondition",
    "nuzline.ai.commands.MoveEffectMask",
    "nuzline.ai.commands.Gender",
    "nuzline.ai.commands.TypeMulti",
    "nuzline.ai.commands.BattleEffect",
    "nuzline.ai.commands.NO_DAMAGE_CALC_EFFECTS",
    "nuzline.ai.commands.ALT_POWER_EFFECTS",
    "nuzline.ai.commands.RISKY_EFFECTS",
    "nuzline.ai.commands.SETUP_FIRST_TURN_EFFECTS",
    "nuzline.ai.commands.HARRASSMENT_EFFECTS",
    "nuzline.ai.commands.FLING_ITEMS_POISON",
    "nuzline.ai.commands.FLING_ITEMS_BURN",
    "nuzline.ai.commands.FLING_ITEMS_PARALYZE",
    "nuzline.ai.commands.FLING_DESIRABLE_EFFECTS",
    "nuzline.ai.commands.FEINT_GRADUAL_RECOVERY_EFFECTS",
    "nuzline.ai.commands.CHECK_HP_TABLES",
    "nuzline.ai.outcome.KAIZO_TRANSCRIBED",
    "nuzline.ai.outcome.KAIZO_EXPERT_LABELS",
    "nuzline.ai.outcome.KAIZO_EXPERT_TRANSCRIBED",
    "nuzline.ai.commands.ReadKind",
    "nuzline.ai.policy._LOOSE_SHARE",
    "nuzline.ai.outcome.SITE_N",
    "nuzline.ai.switching.SwitchSite",
    "nuzline.ai.switching._CANDIDATE_TARGETS",  # engine layout (found by the array scan)
    # game data (Task 5)
    "tools.effect_map.SOUND_MOVES",
    "tools.effect_map.PUNCH_MOVES",
    # engine, asserted by name
    "tools.effect_map.TYPE_CHART_STATUS_MOVES",
    # keyed by name
    "tools.effect_map.DECOMP_ITEM_TO_ID",
    # keyed by effect id / ability name
    "tools.effect_map.DECOMP_EFFECT_TO_ID",
    "tools.effect_map._EFFECTS",
    "tools.effect_map.DECOMP_EFFECT_NAMES",
    "tools.effect_map.CERTAIN_SECONDARY_EFFECTS",
    "tools.effect_map.DECOMP_ABILITY_TO_ID",
    "tools.effect_map.UNSUPPORTED_MOVES",
]
MODULES = [
    "nuzline.ai.commands",
    "nuzline.ai.switching",
    "nuzline.ai.policy",
    "nuzline.ai.layout",
    "nuzline.engine.ids",
    "tools.effect_map",
    "nuzline.saves.check",
]
_UPPER = re.compile(r"_?[A-Z][A-Z0-9_]*")
_CONTAINER_CALLS = {"frozenset", "tuple", "set", "dict"}


def _holds_int(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and type(n.value) is int for n in ast.walk(node))


def _is_id_table(value: ast.AST) -> bool:
    if isinstance(value, ast.Dict | ast.Set | ast.Tuple | ast.List):
        return _holds_int(value)
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    container = (isinstance(func, ast.Name) and func.id in _CONTAINER_CALLS) or (
        isinstance(func, ast.Attribute) and func.attr == "array"  # jnp.array([...]), np.array
    )
    return container and _holds_int(value)


def scan_id_tables(modules: list[str]) -> set[str]:
    """Every module-level IntEnum/IntFlag class, and every UPPER/_UPPER name assigned a
    dict/set/tuple/list literal (or frozenset()/tuple()/set()/dict() call, or an ``x.array``
    call such as ``jnp.array``) holding an int."""
    found = set()
    for module in modules:
        path = Path(importlib.util.find_spec(module).origin)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                bases = {
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases
                }
                if bases & {"IntEnum", "IntFlag"}:
                    found.add(f"{module}.{node.name}")
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name) and _UPPER.fullmatch(target.id) and _is_id_table(value):
                found.add(f"{module}.{target.id}")
    return found


@pytest.mark.fullset
def test_every_id_constant_is_classified():
    assert set(LISTED) <= {c.where for c in CONSTANTS} | set(GAME_DATA) | set(EXEMPT)


@pytest.mark.fullset
def test_no_unlisted_id_tables():
    found = scan_id_tables(MODULES)
    assert found - set(LISTED) - set(GAME_DATA) - set(EXEMPT) == set()
    assert "nuzline.engine.ids.MoveTarget" in found  # the scan sees enums
    assert "nuzline.ai.commands.ABILITY_IDS" in found  # and int tables


def test_classes_are_disjoint_and_well_formed():
    where = {c.where for c in CONSTANTS}
    assert not where & set(GAME_DATA) and not where & set(EXEMPT)
    assert not set(GAME_DATA) & set(EXEMPT)
    for c in CONSTANTS:
        assert c.kind in KINDS and c.scope in SCOPES, c
        assert c.kind == "item" or (c.hold_effect is None and c.hold_param is None), c
    assert POLICY == {VANILLA: "vanilla", KAIZO: "kaizo"}
    assert set(POLICY) == set(GAMES)


def _resolve(where: str):
    module, attr = where.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def _code_entries(where: str) -> set:
    """The table at ``where`` as ``{(name or None, id or None)}``, to compare with its
    ``CONSTANTS`` entries."""
    from nuzline.ai.layout import AiLayout

    obj = _resolve(where)
    if isinstance(obj, AiLayout):  # its script.s:78 compare
        return {(None, obj.water2_ability)}
    if where == "nuzline.engine.ids.QUICK_CLAW_PARAM":
        (claw,) = [c for c in CONSTANTS if c.where == where]
        assert claw.hold_param == obj
        return {("Quick Claw", claw.id)}
    if where == "nuzline.engine.ids.FOCUS_BAND_PARAM":
        (band,) = [c for c in CONSTANTS if c.where == where]
        assert band.hold_param == obj
        return {("Focus Band", band.id)}
    if where == "nuzline.saves.check.ULTIMATE_MOVES":  # species keys and move values
        return {(None, k) for k in obj} | {(None, v) for v in set(obj.values())}
    if where == "nuzline.engine.ids.ABILITY_DECOMP_IDS":  # AbilityId keys, decomp id values
        return {(None, v) for v in obj.values()}
    if isinstance(obj, type) and issubclass(obj, IntEnum | IntFlag):
        return {(None, int(m)) for m in obj}
    if isinstance(obj, int):
        return {(None, obj)}
    if isinstance(obj, str):
        return {(obj, None)}
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return set(obj.items())
    if isinstance(obj, dict):
        return {(None, k) for k in obj}
    if all(isinstance(v, str) for v in obj):
        return {(v, None) for v in obj}
    return {(None, int(v)) for v in obj}


def test_constants_match_the_code():
    # A table both policies read is listed once per scope; each listing is the whole table.
    groups = defaultdict(list)
    for c in CONSTANTS:
        groups[c.where, c.scope].append(c)
    for (where, scope), cs in groups.items():
        code = _code_entries(where)
        names = {n for n, _ in code}
        ids = {i for _, i in code}
        mine = {
            (c.name if None not in names else None, c.id if None not in ids else None) for c in cs
        }
        assert mine == code, (where, scope)
        assert len(cs) == len(code), (where, scope)


def test_message_path():
    assert message_path("nuzline.ai.switching.MOVE_STRUGGLE") == "ai/switching.py MOVE_STRUGGLE"
    assert message_path("tools.effect_map.SOUND_MOVES") == "tools/effect_map.py SOUND_MOVES"


@pytest.mark.parametrize("game", [VANILLA, KAIZO])
def test_constants_hold(game, request):  # Kaizo skips without the Kaizo tables
    names = request.getfixturevalue("names" if game == VANILLA else "kaizo_names")
    path = GAMES[game].intermediate_dir / "items.json"
    if not path.exists():
        pytest.skip(f"{path} not extracted")
    items = json.loads(path.read_text("utf-8"))
    assert check_constants(game, names, items) == []


def test_kaizo_style_mismatch_reported(vanilla_manifest, vanilla_items):
    names = list(vanilla_manifest["move_names"])
    names[165] = "Struggle Bug"
    claw = [dict(i, hold_param=30) if i["name"] == "Quick Claw" else i for i in vanilla_items]
    lines = check_constants(VANILLA, Names({**vanilla_manifest, "move_names": names}), claw)
    assert any("MOVE_STRUGGLE" in s for s in lines) and any("hold param 30" in s for s in lines)
    assert (
        "ai/switching.py MOVE_STRUGGLE: id 165 is 'Struggle Bug' in this game, "
        "expected 'Struggle'" in lines
    )


def test_hold_effect_mismatch_reported(vanilla_manifest, vanilla_items):
    belt = [dict(i, hold_effect=0) if i["name"] == "Expert Belt" else i for i in vanilla_items]
    lines = check_constants(VANILLA, Names(vanilla_manifest), belt)
    assert lines == [
        "ai/commands.py ITEM_EXPERT_BELT: Expert Belt has hold effect 0 in this game, expected 96"
    ]


def test_quick_claw_param_mismatch_reported(vanilla_manifest, vanilla_items):
    # BattleState.qc_roll stores speedRand % (100 / param) == 0 for param 20 only.
    claw = [dict(i, hold_param=10) if i["name"] == "Quick Claw" else i for i in vanilla_items]
    lines = check_constants(VANILLA, Names(vanilla_manifest), claw)
    assert lines == [
        "engine/ids.py QUICK_CLAW_PARAM: Quick Claw has hold param 10 in this game, expected 20"
    ]


def test_policy_constants_only_for_their_policy(vanilla_manifest, vanilla_items):
    names = list(vanilla_manifest["move_names"])
    names[446] = "Pebble Trap"  # Stealth Rock: MOVE_EFFECTS is vanilla_policy only
    renamed = Names({**vanilla_manifest, "move_names": names})
    assert any("MOVE_EFFECTS" in s for s in check_constants(VANILLA, renamed, vanilla_items))
    assert not any("MOVE_EFFECTS" in s for s in check_constants(KAIZO, renamed, vanilla_items))
    abilities = list(vanilla_manifest["ability_names"])
    abilities[87] = "Wet Skin"  # Dry Skin: script.s:78's compare in the Kaizo policy only
    renamed = Names({**vanilla_manifest, "ability_names": abilities})
    # The engine's table (E2 batch 3, ids.ABILITY_DECOMP_IDS) is checked in both games.
    engine = (
        "engine/ids.py ABILITY_DECOMP_IDS: id 87 is 'Wet Skin' in this game, expected 'Dry Skin'"
    )
    assert check_constants(VANILLA, renamed, vanilla_items) == [engine]
    assert sorted(check_constants(KAIZO, renamed, vanilla_items)) == sorted(
        [engine, "ai/layout.py KAIZO: id 87 is 'Wet Skin' in this game, expected 'Dry Skin'"]
    )


def test_kaizo_policy_ids_are_checked(vanilla_manifest, vanilla_items):
    """The script's and trainer_ai.c's ids are the same numbers in both ROMs, so a slot the
    Kaizo paths read by id must keep its name: a renamed one is reported (149 is Triple
    Axel in Kaizo, so an id list keyed on Psywave by name would be wrong there)."""
    names = list(vanilla_manifest["move_names"])
    for move, new in ((90, "Triple Axel"), (360, "Spin Ball"), (393, "Float")):
        names[move] = new
    renamed = Names({**vanilla_manifest, "move_names": names})
    lines = check_constants(KAIZO, renamed, vanilla_items)
    assert (
        "ai/commands.py MOVE_IDS: id 90 is 'Triple Axel' in this game, expected 'Fissure'" in lines
    )
    assert any("MOVE_IDS: id 360" in s and "Gyro Ball" in s for s in lines)
    assert any("switching.py MOVE_IDS: id 393" in s for s in lines)


def test_id_cases_are_decomp_ids_in_every_policy():
    """The TrainerAI_CalcDamage and TrainerAI_MoveType id cases are compared by id in both
    policies (the C code is unchanged in Kaizo), so Kaizo's Triple Axel (149) hits
    MOVE_PSYWAVE: they are vanilla_policy constants, never name-checked against Kaizo."""
    scopes = {
        c.scope
        for c in CONSTANTS
        if c.where
        in ("nuzline.ai.commands.CALC_DAMAGE_MOVES", "nuzline.ai.commands.MOVE_TYPE_MOVES")
    }
    assert scopes == {"vanilla_policy"}


def _tmp_game(tmp_path: Path, items: list[dict] | None) -> Game:
    _require_tables()
    tables, inter = tmp_path / "tables", tmp_path / "inter"
    tables.mkdir()
    inter.mkdir()
    for f in ("tables.npz", "manifest.json"):
        shutil.copy(TABLES_DIR / f, tables / f)
    if items is not None:
        (inter / "items.json").write_text(json.dumps(items), encoding="utf-8")
    return Game(VANILLA, tables, inter)


def test_load_game_refuses_a_contradicted_constant(tmp_path, vanilla_items):
    claw = [dict(i, hold_param=30) if i["name"] == "Quick Claw" else i for i in vanilla_items]
    with pytest.raises(GameDataError, match="QUICK_CLAW_PARAM"):
        load_game(_tmp_game(tmp_path, claw))


def test_load_game_without_items_names_the_extract_command(tmp_path):
    with pytest.raises(GameDataError, match=r"tools\.extract_decomp"):
        load_game(_tmp_game(tmp_path, None))


def test_kaizo_load_game_without_items_names_build_tables(tmp_path):
    game = _tmp_game(tmp_path, None)
    manifest = json.loads((game.tables_dir / "manifest.json").read_text(encoding="utf-8"))
    (game.tables_dir / "manifest.json").write_text(
        json.dumps({**manifest, "game": KAIZO}), encoding="utf-8"
    )
    with pytest.raises(GameDataError, match=r"tools\.build_tables --game kaizo"):
        load_game(Game(KAIZO, game.tables_dir, game.intermediate_dir))
