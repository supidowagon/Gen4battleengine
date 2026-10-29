import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from nuzline.data.party import load_party
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.state import BattleState, make_state

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STATE_MD = ROOT / "src" / "nuzline" / "engine" / "STATE.md"
DECOMP = ROOT / "external" / "pokeplatinum"


@pytest.fixture
def teams(names):
    p = load_party(FIXTURES / "sample_party.toml", names)
    return p, p[:3]


@pytest.fixture
def state(tables, teams):
    return make_state(tables, *teams)


def test_shapes_dtypes_and_no_64bit(state):
    s = state
    assert s.hp.shape == (2, 6) and s.hp.dtype == jnp.int16
    assert s.stats.shape == (2, 6, 5) and s.moves.shape == (2, 6, 4)
    assert s.ivs.shape == (2, 6, 6) and s.ivs.dtype == jnp.int8
    assert s.stages.shape == (2, 8) and s.trainer_items.shape == (2, 4)
    assert s.ai_seen_moves.shape == (6, 4) and s.weather.shape == ()
    assert all(x.dtype.itemsize <= 4 for x in s)


def test_empty_slots_are_zero_and_full_hp_default(tables, teams):
    p, o = teams
    s = make_state(tables, p, o)
    assert int(s.species[1, 5]) == 0 and int(s.hp[1, 5]) == 0
    assert int(s.hp[0, 0]) == int(s.max_hp[0, 0]) > 0


def test_fainted_member_and_start_weather(tables, names):
    p = load_party(FIXTURES / "sample_party_fainted.toml", names)
    s = make_state(tables, p, p[:2], start_condition=Weather.HAIL)
    assert int(s.hp[0, 5]) == 0 and int(s.max_hp[0, 5]) > 0
    assert int(s.weather) == Weather.HAIL and int(s.weather_timer) == 0


def test_state_is_a_valid_pytree_for_vmap(state):
    batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), state)
    assert batched.hp.shape == (3, 2, 6)


def test_slot_contents_come_from_spec_and_tables(state, names, tables):
    s = state
    # Monferno L14, base 64/78/52/78/52/81, IVs 20, EVs 0, Hardy (Pokemon_CalcStats).
    assert int(s.species[0, 0]) == names.species("Monferno") and int(s.level[0, 0]) == 14
    assert int(s.max_hp[0, 0]) == (2 * 64 + 20) * 14 // 100 + 14 + 10
    assert s.stats[0, 0].tolist() == [(2 * b + 20) * 14 // 100 + 5 for b in (78, 52, 78, 52, 81)]
    assert s.moves[0, 0].tolist() == [
        names.move(m) for m in ("Scratch", "Leer", "Ember", "Mach Punch")
    ]
    assert s.pp[0, 0].tolist() == [int(tables.move_pp[int(m)]) for m in s.moves[0, 0]]
    assert s.pp[0, 4, 3] == 0 and s.moves[0, 4, 3] == 0  # Shinx has three moves
    assert int(s.item[0, 2]) == names.item("Sitrus Berry") and int(s.item_consumed[0, 2]) == 0
    assert s.types[0, 0].tolist() == tables.species_types[names.species("Monferno")].tolist()
    assert int(s.ability[0, 0]) == int(tables.species_abilities[names.species("Monferno"), 0])
    assert s.ivs[0, 0].tolist() == [20] * 6
    assert s.active.tolist() == [0, 0] and s.stages.tolist() == [[0] * 8] * 2
    assert int(s.over) == 0 and int(s.winner) == 0 and int(s.turn) == 0


def test_current_hp_out_of_range_is_rejected(tables, teams):
    p, o = teams
    with pytest.raises(ValueError):
        make_state(tables, [dataclasses.replace(p[0], current_hp=999)], o)


def test_replace_hp(state):
    s = state.replace_hp(1, 2, 7)
    assert int(s.hp[1, 2]) == 7 and s.hp.dtype == jnp.int16
    assert int(state.hp[1, 2]) != 7  # the original is untouched


def test_with_stages(state):
    s = state.with_stages(1, def_=-6, spe=2, acc=1)
    assert int(s.stages[1, Stat.DEF]) == -6 and int(s.stages[1, Stat.SPE]) == 2
    assert int(s.stages[1, Stat.ACC]) == 1 and int(s.stages[0].sum()) == 0
    assert s.stages.dtype == jnp.int8
    with pytest.raises(TypeError):
        state.with_stages(0, hp=1)
    with pytest.raises(ValueError):
        state.with_stages(0, atk=7)


def test_with_pp(state):
    s = state.with_pp(0, 0, [0, 1, 2, 3])
    assert s.pp[0, 0].tolist() == [0, 1, 2, 3] and s.pp.dtype == jnp.int8


def test_with_status(state):
    s = state.with_status(1, 0, Status.SLEEP, counter=2)
    assert int(s.status[1, 0]) == Status.SLEEP and int(s.status_counter[1, 0]) == 2
    s = s.with_status(1, 0, Status.BURN)
    assert int(s.status[1, 0]) == Status.BURN and int(s.status_counter[1, 0]) == 0


# --- STATE.md: the decomp struct checklist -------------------------------------------

STATUS_RE = re.compile(r"^(mapped|later \([^)]+\)|not needed because .+)$")


def _state_md_rows() -> list[tuple[str, str, str]]:
    rows = []
    for line in STATE_MD.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 3 or cells[0] in ("Struct.field", ""):
            continue
        if set(cells[0]) <= set("-: "):
            continue
        rows.append(tuple(c.strip("`") for c in cells))
    return rows


def _decomp_structs() -> list[tuple[str, Path]]:
    """(struct name, header) for the three battle-state struct rows of decomp_paths.md."""
    out = []
    for line in (ROOT / "docs" / "decomp_paths.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| Battle-state struct"):
            cells = line.strip().strip("|").split("|")
            path = re.search(r"`([^`]+)`", cells[1]).group(1)
            name = re.search(r"`([^`]+)`", cells[2]).group(1)
            out.append((name, DECOMP / path))
    return out


def _struct_fields(header: Path, name: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", header.read_text(encoding="utf-8"), flags=re.DOTALL)
    m = re.search(r"struct\s+" + name + r"\s*\{(.*?)\n\}", text, flags=re.DOTALL)
    assert m, f"struct {name} not found in {header}"
    fields = []
    for decl in m.group(1).split(";"):
        decl = re.sub(r"\[[^\]]*\]|:\s*\d+", "", decl).strip()
        if decl:
            fields.append(re.findall(r"\w+", decl)[-1])
    return fields


def test_state_md_accounts_for_every_struct_field():
    rows = _state_md_rows()
    assert len(rows) > 100
    for field, pytree, status in rows:
        assert STATUS_RE.match(status), f"{field}: bad status {status!r}"
        if status == "mapped":
            for name in re.findall(r"\w+", pytree):
                assert name in BattleState._fields, f"{field}: {name} is not a BattleState field"
            assert re.findall(r"\w+", pytree), f"{field}: mapped row names no field"
    if not DECOMP.exists():
        pytest.skip("external/pokeplatinum not cloned")
    structs = _decomp_structs()
    assert [n for n, _ in structs] == ["BattleMon", "SideConditions", "BattleContext"]
    listed = {f for f, _, _ in rows}
    for name, header in structs:
        fields = _struct_fields(header, name)
        assert len(fields) > 10
        missing = [f for f in fields if f"{name}.{f}" not in listed]
        assert not missing, f"STATE.md lacks {name} fields: {missing}"


# The Task 6 adversarial audit (ledger) found these "not needed" judgements wrong: the
# switch-in ability checks re-run after every move gated by the announced flags
# (battle_lib.c:3649-4168, battle_controller_player.c:3673/3973), Quick Claw, Protect,
# the speed values and Sketch are read later in the turn, and ``moveLockedInto`` is a
# move id that no int8 field can hold. They are needed for M6, not M1.
AUDIT_LATER_ROWS = (
    "BattleMon.intimidateAnnounced",
    "BattleMon.weatherAbilityAnnounced",
    "BattleMon.traceAnnounced",
    "BattleMon.downloadAnnounced",
    "MoveEffectsData.quickClaw",
    "BattleContext.moveProtect",
    "BattleContext.monSpeedValues",
    "BattleContext.moveSketched",
    "BattleContext.moveLockedInto",
)


def test_state_md_audit_rows_are_later_m6():
    status = {field: status for field, _, status in _state_md_rows()}
    for field in AUDIT_LATER_ROWS:
        assert status[field] == "later (M6)", f"{field}: {status[field]!r}"
