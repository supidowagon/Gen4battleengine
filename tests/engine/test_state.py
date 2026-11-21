import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.party import load_party
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import LAYOUT, BattleState, make_state
from nuzline.reference.convert import from_ref, to_ref
from tests.engine.helpers import A, jit_step, roark_state, weather_of, weather_timer_of

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
    assert s.ai_seen_moves.shape == (2, 4) and s.field_mask.shape == ()
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
    assert int(weather_of(s)) == Weather.HAIL and int(weather_timer_of(s)) == 0


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


@pytest.mark.fullset
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
# the speed values and Sketch are read later in the turn. They are needed for M6, not M1
# (``moveLockedInto`` moved to ``locked_move`` in milestone 4a; ``moveProtect`` folded into
# ``protect_count`` in K3b2 Task 6, test_move_protect_is_folded_into_protect_count; the four
# announced flags mapped with Skill Swap, E3-2-5, ``hooks.move_end_switch_in``).
AUDIT_LATER_ROWS = ("BattleContext.moveSketched",)


def test_state_md_audit_rows_are_later_m6():
    status = {field: status for field, _, status in _state_md_rows()}
    for field in AUDIT_LATER_ROWS:
        assert status[field] == "later (M6)", f"{field}: {status[field]!r}"


def test_mon_speed_values_map_to_speed_rec():
    # E3-B2-15: monSpeedValues, the speeds of the last CompareBattlerSpeed, read by Gyro Ball's
    # script command and by TrainerAI_CalcDamage's Gyro Ball case: the per-battler `speed_rec`.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    assert rows["BattleContext.monSpeedValues"] == ("speed_rec", "mapped")
    assert "speed_rec" in BattleState._fields and LAYOUT["speed_rec"][0] == (2,)


def test_move_protect_is_folded_into_protect_count():
    # docs/kaizo/k3b2-rules.md §12: the moveProtect reset is folded into protect_count.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    assert rows["BattleContext.moveProtect"] == ("protect_count", "mapped")
    assert rows["MoveEffectsData.protectSuccessTurns"] == ("protect_count", "mapped")


def test_conversion_2_record_maps_to_conv2_fields():
    # E3-B2-3: UpdateFlagsWhenHit's conversion2Move / Battler / Type, read by Conversion 2.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    for part, name in (
        ("Move", "conv2_move"),
        ("Battler", "conv2_battler"),
        ("Type", "conv2_type"),
    ):
        assert rows[f"BattleContext.conversion2{part}"] == (name, "mapped")


def test_magic_coat_flag_maps_to_magic_coat():
    # E3-B2-6: turnFlags.magicCoat is the per-battler `magic_coat` (1 + the user's move slot);
    # magicCoatMon is the reflect phase's actor, per-action scratch.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    pytree, status = rows["BattleContext.turnFlags"]
    assert status == "mapped" and "magic_coat" in re.findall(r"\w+", pytree)
    assert "magic_coat" in BattleState._fields
    assert rows["BattleContext.magicCoatMon"][1].startswith("not needed because")


def test_attract_bits_map_to_infatuated():
    # docs/kaizo/k3b2-rules.md §12: statusVolatile's ATTRACT bits are `infatuated`.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    pytree, status = rows["BattleMon.statusVolatile"]
    assert status == "mapped" and "infatuated" in re.findall(r"\w+", pytree)
    later = re.search(r"other bits \(([^)]*)\)", STATE_MD.read_text(encoding="utf-8"))
    assert later and "Attract" not in later.group(1)


def test_speed_rand_maps_to_qc_roll():
    # docs/kaizo/k3b2-rules.md §12: speedRand is read only through Quick Claw's test.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    assert rows["BattleContext.speedRand"] == ("qc_roll", "mapped")
    # E2 B10-6: Knock Off reads moveEffectsData.quickClaw (subscript_knock_off.s:8): the turn
    # scratch ts_claw (order.claw_flags). SF2 (k): the flags persist across the turn end in
    # claw_flag / custap_flag, so each row names two fields (read as the audit reads a row).
    for field, names in (
        ("MoveEffectsData.quickClaw", {"claw_flag", "ts_claw"}),
        ("MoveEffectsData.custapBerry", {"custap_flag", "ts_custap"}),
    ):
        pytree, status = rows[field]
        assert status == "mapped" and set(re.findall(r"\w+", pytree)) == names


def test_leads_choose_the_active_slots_before_the_switch_in_effects(tables, names):
    # The leads are the active slots and the switch-in hooks run for them: an
    # Intimidate Shinx leading from opponent slot 1 lowers the player lead's Attack.
    p = load_party(FIXTURES / "sample_party.toml", names)
    shinx = dataclasses.replace(p[4], ability_slot=1)  # Rivalry / Intimidate
    assert names.ability_names[names.species_abilities(shinx.species)[1]] == "Intimidate"
    opp = [p[0], shinx]
    assert int(make_state(tables, p, opp).stages[0, Stat.ATK]) == 0
    s = make_state(tables, p, opp, leads=(2, 1))
    assert [int(x) for x in s.active] == [2, 1]
    assert int(s.stages[0, Stat.ATK]) == -1
    with pytest.raises(ValueError, match="side 0: lead slot 6"):
        make_state(tables, p, opp, leads=(6, 0))
    with pytest.raises(ValueError, match="side 1: lead slot 2"):
        make_state(tables, p, opp, leads=(0, 2))


@pytest.fixture
def sample_spec(names):
    return load_party(FIXTURES / "sample_party.toml", names)[0]


def test_make_state_uses_spec_pp_counter_and_stats(tables, sample_spec):
    spec = dataclasses.replace(
        sample_spec,
        pp=(3, 0, 1, 2),
        status=Status.SLEEP,
        status_counter=2,
        stats=(50, 30, 31, 32, 33, 34),
        current_hp=20,
    )
    s = make_state(tables, [spec], [sample_spec])
    assert s.pp[0, 0].tolist() == [3, 0, 1, 2] and int(s.status_counter[0, 0]) == 2
    assert int(s.max_hp[0, 0]) == 50 and s.stats[0, 0].tolist() == [30, 31, 32, 33, 34]
    assert int(s.hp[0, 0]) == 20


def test_make_state_defaults_unchanged(tables, sample_spec):
    s = make_state(tables, [sample_spec], [sample_spec])
    move_pp = [int(tables.move_pp[m]) if m else 0 for m in sample_spec.moves]
    assert s.pp[0, 0].tolist() == move_pp and int(s.status_counter[0, 0]) == 0
    assert int(s.max_hp[0, 0]) > 0 and all(v > 0 for v in s.stats[0, 0].tolist())


def test_current_hp_checked_against_override(tables, sample_spec):
    spec = dataclasses.replace(sample_spec, stats=(50, 30, 31, 32, 33, 34), current_hp=51)
    with pytest.raises(ValueError, match="current_hp"):
        make_state(tables, [spec], [sample_spec])


# The per-battler fields of the active battler Task 3 adds (all reset like ``_VOLATILE``).
NEW_VOLATILES = (
    "foresight",
    "disabled_move",
    "disable_turns",
    "locked_move",
    "bide_turns",
    "bide_damage",
    "curled",
    "mud_sport",
    "aqua_ring",
)


def _with_new_volatiles(s, side):
    # The disabled move is one the active knows (so the end-of-turn check keeps it), not its
    # first, which the tests below use.
    known = int([m for m in s.moves[side, int(s.active[side])].tolist() if m][-1])
    values = {"foresight": 1, "disabled_move": known}
    return s._replace(
        **{name: getattr(s, name).at[side].set(values.get(name, 3)) for name in NEW_VOLATILES}
    )


def _ticked(values):
    """``_new_volatiles`` of a side that acted in a turn: the Bide counter drops when the
    battler moves (``CHECK_STATUS_STATE_BIDE``) and the Disable counter at the end of it."""
    ticked = list(values)
    ticked[NEW_VOLATILES.index("disable_turns")] -= 1
    ticked[NEW_VOLATILES.index("bide_turns")] -= 1
    return ticked


def _new_volatiles(s, side):
    return [int(getattr(s, name)[side]) for name in NEW_VOLATILES]


@pytest.mark.slow
def test_new_volatiles_reset_on_switch_and_faint(tables, names):
    s = roark_state(tables, names)
    # A switch-out clears side 0's battler (BattleSystem_UpdateAfterSwitch zeroes
    # statusVolatile, moveEffectsMask and moveEffectsData); side 1 is untouched.
    both = _with_new_volatiles(_with_new_volatiles(s, 0), 1)
    out = jit_step(tables, both, A(4 + 1, "Tackle", both, names), empty_outcome())
    assert _new_volatiles(out, 0) == [0] * len(NEW_VOLATILES)
    assert _new_volatiles(out, 1) == _ticked(_new_volatiles(both, 1)) != [0] * len(NEW_VOLATILES)
    # A KO'd active (CleanupFaintedMon) clears its own fields, not the attacker's.
    ko = _with_new_volatiles(_with_new_volatiles(s.replace_hp(1, 0, 1), 0), 1)
    out = jit_step(tables, ko, A("Scratch", "Leer", ko, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0
    assert _new_volatiles(out, 1) == [0] * len(NEW_VOLATILES)
    assert _new_volatiles(out, 0) == _ticked(_new_volatiles(ko, 0))


def test_ref_roundtrip_new_fields(tables, names):
    s = _with_new_volatiles(_with_new_volatiles(roark_state(tables, names), 0), 1)
    s = s._replace(
        disabled_move=s.disabled_move.at[:].set(jnp.asarray([33, 1000], jnp.int16)),
        bide_damage=s.bide_damage.at[:].set(jnp.asarray([321, 7], jnp.int16)),
    )
    back = from_ref(to_ref(s))
    for name, x, y in zip(s._fields, s, back, strict=True):
        assert x.dtype == y.dtype and bool((x == y).all()), name
    assert [b.disabled_move for b in to_ref(s).battlers] == [33, 1000]


def test_rotom_wash_state_uses_its_row(tables, sample_spec):
    import numpy as np

    from nuzline.engine.stats import calc_stats

    wash = dataclasses.replace(sample_spec, species=479, form=2, ability_slot=0)
    s = make_state(tables, [wash], [sample_spec])
    assert s.types[0, 0].tolist() == [
        int(tables.species_types[504, 0]),
        int(tables.species_types[504, 1]),
    ]
    assert int(s.ability[0, 0]) == int(tables.species_abilities[504, 0])
    sp = sample_spec
    want = calc_stats(
        np.asarray(tables.species_base_stats[504]),
        np.asarray(sp.ivs),
        np.asarray(sp.evs),
        sp.level,
        np.asarray(tables.nature_mods[sp.nature]),
        species=479,
    )
    assert int(s.max_hp[0, 0]) == int(want[0])
    assert s.stats[0, 0].tolist() == [int(v) for v in want[1:]]
    assert int(s.form[0, 0]) == 2


def test_bad_form_raises(tables, sample_spec):
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        make_state(tables, [dataclasses.replace(sample_spec, species=479, form=6)], [sample_spec])


def test_battle_system_terrain_maps_to_terrain():
    # docs/kaizo/k3b2-rules.md §12: BattleSystem.terrain is a constant field scalar.
    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    assert rows["BattleSystem.terrain"] == ("terrain", "mapped")
    assert LAYOUT["terrain"] == ((), np.int8)
