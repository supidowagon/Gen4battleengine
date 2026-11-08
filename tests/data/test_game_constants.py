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
    "nuzline.engine.stats.SPECIES_SHEDINJA",
    "nuzline.ai.switching.MOVE_STRUGGLE",
    "nuzline.prep.problem.HEART_SCALE",
    "nuzline.engine.coverage.SPECIES_GIRATINA",
    "nuzline.engine.coverage.SPECIES_SHAYMIN",
    "nuzline.engine.coverage.ITEM_GRISEOUS_ORB",
    # engine, asserted by name and hold effect / hp_restored
    "nuzline.ai.commands.ITEM_EXPERT_BELT",
    "nuzline.ai.switching.ITEM_IRON_BALL",
    "nuzline.engine.ids.POTION_HP_RESTORED",
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
    "nuzline.ai.commands.BattleEffect",
    "nuzline.ai.commands.MOVE_IDS",
    "nuzline.ai.commands.SOUND_MOVES",
    "nuzline.ai.switching.MOVE_IDS",
    "nuzline.ai.switching._MOVES",
    "nuzline.ai.switching.PLAYER_ABILITY_REFUSALS",
    "nuzline.ai.switching.PLAYER_MOVE_REFUSALS",
    "nuzline.ai.policy._STATE_MOVE_IDS",
    "nuzline.ai.policy._basic",  # the Dry Skin quirk
    "nuzline.ai.outcome._SITES",
    # EXEMPT: engine encodings
    "nuzline.engine.ids.MoveTarget",
    "nuzline.engine.ids.EffectId",
    "nuzline.engine.ids.Status",
    "nuzline.engine.ids.Stat",
    "nuzline.engine.ids.MoveFlag",
    "nuzline.engine.ids.Weather",
    "nuzline.engine.ids.ItemId",
    "nuzline.engine.ids.TrainerItemId",
    "nuzline.engine.ids.AbilityId",
    # EXEMPT: decomp AI-script encodings
    "nuzline.ai.commands.Battler",
    "nuzline.ai.commands.LoadType",
    "nuzline.ai.commands.SideCondition",
    "nuzline.ai.commands.TypeMulti",
    "nuzline.ai.switching.SwitchSite",
    "nuzline.ai.switching._CANDIDATE_TARGETS",  # engine layout (found by the array scan)
    # game data (Task 5)
    "tools.effect_map.SOUND_MOVES",
    "tools.effect_map.PUNCH_MOVES",
    # engine, asserted by name
    "tools.effect_map.TYPE_CHART_STATUS_MOVES",
    # keyed by name
    "tools.effect_map.DECOMP_ITEM_TO_ID",
    "tools.effect_map.DECOMP_TRAINER_ITEM_TO_ID",
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


def test_every_id_constant_is_classified():
    assert set(LISTED) <= {c.where for c in CONSTANTS} | set(GAME_DATA) | set(EXEMPT)


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
        assert c.kind == "item" or (c.hold_effect is None and c.hp_restored is None), c
    assert POLICY == {VANILLA: "vanilla", KAIZO: None}
    assert set(POLICY) == set(GAMES)


def _resolve(where: str):
    module, attr = where.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def _code_entries(where: str) -> set:
    """The table at ``where`` as ``{(name or None, id or None)}``, to compare with its
    ``CONSTANTS`` entries."""
    from nuzline.ai import commands, switching

    obj = _resolve(where)
    if where == "nuzline.ai.policy._basic":
        src = Path(importlib.util.find_spec("nuzline.ai.policy").origin).read_text("utf-8")
        assert "(quirk) :78 should test Dry Skin" in src
        return {("Levitate", commands.ABILITY_IDS["Levitate"])}
    if where == "nuzline.engine.ids.POTION_HP_RESTORED":
        (potion,) = [c for c in CONSTANTS if c.where == where]
        assert potion.hp_restored == obj
        return {("Potion", potion.id)}
    if where == "nuzline.ai.commands.BattleEffect":
        assert set(commands.MOVE_EFFECTS.values()) == set(obj)
        return {(None, m) for m in commands.MOVE_EFFECTS}
    if where == "nuzline.saves.check.ULTIMATE_MOVES":  # species keys and move values
        return {(None, k) for k in obj} | {(None, v) for v in set(obj.values())}
    if where == "nuzline.ai.switching._MOVES":
        return {(n, switching.MOVE_IDS[n]) for n in obj}
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
    groups = defaultdict(list)
    for c in CONSTANTS:
        groups[c.where].append(c)
    for where, cs in groups.items():
        code = _code_entries(where)
        names = {n for n, _ in code}
        ids = {i for _, i in code}
        mine = {
            (c.name if None not in names else None, c.id if None not in ids else None) for c in cs
        }
        assert mine == code, where
        assert len(cs) == len(code), where


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
    potion = [dict(i, hp_restored=30) if i["name"] == "Potion" else i for i in vanilla_items]
    lines = check_constants(VANILLA, Names({**vanilla_manifest, "move_names": names}), potion)
    assert any("MOVE_STRUGGLE" in s for s in lines) and any("restores 30" in s for s in lines)
    assert (
        "ai/switching.py MOVE_STRUGGLE: id 165 is 'Struggle Bug' in this game, "
        "expected 'Struggle'" in lines
    )
    assert "engine/ids.py POTION_HP_RESTORED: Potion restores 30 in this game, expected 20" in lines


def test_hold_effect_mismatch_reported(vanilla_manifest, vanilla_items):
    belt = [dict(i, hold_effect=0) if i["name"] == "Expert Belt" else i for i in vanilla_items]
    lines = check_constants(VANILLA, Names(vanilla_manifest), belt)
    assert lines == [
        "ai/commands.py ITEM_EXPERT_BELT: Expert Belt has hold effect 0 in this game, expected 96"
    ]


def test_policy_constants_only_for_vanilla_policy(vanilla_manifest, vanilla_items):
    names = list(vanilla_manifest["move_names"])
    names[446] = "Pebble Trap"  # Stealth Rock: vanilla_policy only
    renamed = Names({**vanilla_manifest, "move_names": names})
    assert any("MOVE_EFFECTS" in s for s in check_constants(VANILLA, renamed, vanilla_items))
    assert check_constants(KAIZO, renamed, vanilla_items) == []


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
    potion = [dict(i, hp_restored=30) if i["name"] == "Potion" else i for i in vanilla_items]
    with pytest.raises(GameDataError, match="POTION_HP_RESTORED"):
        load_game(_tmp_game(tmp_path, potion))


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
