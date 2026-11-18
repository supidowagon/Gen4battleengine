import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.forms import form_row
from nuzline.data.learnsets import load_learnsets
from nuzline.data.tables import MANIFEST_VERSION, Tables, load_names, load_tables
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, MoveTarget, Stat, Status

TABLES = Path(__file__).resolve().parents[2] / "data" / "gen4_platinum"

# Every move the M1a sample team, Roark's real party, the roark_state fixture
# moveset and the Task 7-9 tests use. None may be UNSUPPORTED.
M1A_MOVES = [
    # sample team
    "Scratch", "Leer", "Ember", "Mach Punch", "Pound", "Growl", "Bubble", "Water Sport",
    "Tackle", "Withdraw", "Absorb", "Razor Leaf", "Quick Attack", "Wing Attack", "Spark",
    "Growth", "Stun Spore",
    # Roark's real party (Geodude, Onix, Cranidos) and the fixture moveset
    "Stealth Rock", "Rock Throw", "Screech", "Headbutt", "Pursuit",
    # Task 7-9 tests
    "Hammer Arm", "Rock Polish", "Bind", "Taunt", "Leech Seed", "Supersonic", "Fury Swipes",
    "Defense Curl", "Bite",
]  # fmt: skip


def _effect(t, n, move):
    m = n.move(move)
    return int(t.move_effect[m]), tuple(int(x) for x in t.move_param[m])


def test_dtypes_and_shapes(tables, names):
    t, n = tables, names
    assert t.species_base_stats.dtype == jnp.int16 and t.species_base_stats.shape[1] == 6
    assert t.move_param.shape[1] == 3 and t.move_param.dtype == jnp.int16
    # Ruling: decomp type ids are kept, including the ??? (Mystery) type, so the
    # chart is 18x18 rather than the plan's 17x17.
    assert "???" in n.type_names
    assert t.type_chart.shape == (len(n.type_names), len(n.type_names))
    assert t.type_chart_order.shape == t.type_chart.shape
    assert t.type_chart_order.dtype == jnp.int16
    assert t.nature_mods.shape == (25, 6)
    # Deviation from the brief's int8: decomp base friendship reaches 140
    # (Clefairy), which int8 cannot hold. int16 keeps to the storage rule.
    assert t.species_friendship.dtype == jnp.int16
    assert all(x.dtype.itemsize <= 4 for x in t)


def test_effect_mapping(tables, names):
    t, n = tables, names
    assert int(t.move_effect[n.move("Tackle")]) == EffectId.PLAIN
    assert int(t.move_effect[n.move("Stealth Rock")]) == EffectId.STEALTH_ROCK
    assert int(t.move_effect[n.move("Leer")]) == EffectId.STAT_FOE
    assert tuple(int(x) for x in t.move_param[n.move("Leer")]) == (2, -1, 0)
    # E2 batch 5: Metronome has an engine effect; coverage refuses it while a move it can call
    # is unmodelled (ruling B5-2).
    assert int(t.move_effect[n.move("Metronome")]) == EffectId.METRONOME


def test_only_thunder_wave_is_a_type_checked_status_move(tables, names):
    """Thunder Wave is the one status move the decomp runs through the type chart
    (Ground immunity, battle_controller_player.c CheckTypeChart): move_param[1] = 1
    marks it. Glare and Stun Spore skip the type chart in Gen 4."""
    t, n = tables, names
    assert _effect(t, n, "Thunder Wave") == (EffectId.STATUS_FOE, (Status.PARALYSIS, 1, 0))
    for move in ("Glare", "Stun Spore"):
        assert _effect(t, n, move) == (EffectId.STATUS_FOE, (Status.PARALYSIS, 0, 0))


def test_thawing_moves_carry_the_thaw_flag(tables, names):
    """BATTLE_EFFECT_THAW_AND_BURN_HIT is a burn hit whose frozen user moves and thaws
    (move_param[2] = 1); a plain burn hit has no flag; Flare Blitz's recoil family carries it."""
    t, n = tables, names
    for move in ("Flame Wheel", "Sacred Fire"):
        assert _effect(t, n, move) == (EffectId.SECONDARY_STATUS, (Status.BURN, 0, 1))
    assert _effect(t, n, "Ember") == (EffectId.SECONDARY_STATUS, (Status.BURN, 0, 0))
    # Flare Blitz (BATTLE_EFFECT_RECOIL_BURN_HIT) thaws its user too (battle_controller_player.c
    # :2481-2482, :2730), modelled since B10-1 as RECOIL_STATUS (status, recoil divisor, thaw);
    # Volt Tackle, the same family without the thaw, carries no flag.
    assert _effect(t, n, "Flare Blitz") == (EffectId.RECOIL_STATUS, (Status.BURN, 3, 1))
    assert _effect(t, n, "Volt Tackle") == (EffectId.RECOIL_STATUS, (Status.PARALYSIS, 3, 0))


def test_move_target_ids_match_the_manifest(tables, names):
    """``MoveTarget`` (used by the engine for RANGE_USER moves) is the order the
    tables were built with; the self-targeting status moves are RANGE_USER."""
    manifest = json.loads((TABLES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_names"] == [t.name for t in MoveTarget]
    t, n = tables, names
    for move in ("Growth", "Withdraw", "Rock Polish", "Defense Curl"):
        assert int(t.move_target[n.move(move)]) == MoveTarget.RANGE_USER, move
    assert int(t.move_target[n.move("Leer")]) != MoveTarget.RANGE_USER


def test_every_m1a_move_is_supported(tables, names):
    t, n = tables, names
    unsupported = [m for m in M1A_MOVES if _effect(t, n, m)[0] == EffectId.UNSUPPORTED]
    assert unsupported == []


def test_m1a_effect_params(tables, names):
    t, n = tables, names
    E = EffectId
    assert _effect(t, n, "Growth") == (E.STAT_SELF, (Stat.SPA, 1, 0))
    assert _effect(t, n, "Defense Curl") == (E.STAT_SELF, (Stat.DEF, 1, 1))
    assert _effect(t, n, "Withdraw") == (E.STAT_SELF, (Stat.DEF, 1, 0))
    assert _effect(t, n, "Rock Polish") == (E.STAT_SELF, (Stat.SPE, 2, 0))
    assert _effect(t, n, "Growl") == (E.STAT_FOE, (Stat.ATK, -1, 0))
    assert _effect(t, n, "Screech") == (E.STAT_FOE, (Stat.DEF, -2, 0))
    assert _effect(t, n, "Bubble") == (E.SECONDARY_STAT_FOE, (Stat.SPE, -1, 0))
    assert _effect(t, n, "Hammer Arm") == (E.SECONDARY_STAT_SELF, (Stat.SPE, -1, 0))
    assert _effect(t, n, "Ember") == (E.SECONDARY_STATUS, (Status.BURN, 0, 0))
    assert _effect(t, n, "Spark") == (E.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0))
    assert _effect(t, n, "Stun Spore") == (E.STATUS_FOE, (Status.PARALYSIS, 0, 0))
    assert _effect(t, n, "Absorb") == (E.DRAIN, (50, 0, 0))
    assert _effect(t, n, "Take Down") == (E.RECOIL, (4, 0, 0))
    assert _effect(t, n, "Headbutt")[0] == E.FLINCH
    assert _effect(t, n, "Bite")[0] == E.FLINCH
    assert _effect(t, n, "Razor Leaf")[0] == E.HIGH_CRIT
    assert _effect(t, n, "Bind")[0] == E.TRAP
    assert _effect(t, n, "Pursuit")[0] == E.PURSUIT
    assert _effect(t, n, "Taunt")[0] == E.TAUNT
    assert _effect(t, n, "Supersonic")[0] == E.CONFUSE_FOE
    assert _effect(t, n, "Leech Seed")[0] == E.LEECH_SEED
    assert _effect(t, n, "Fury Swipes")[0] == E.MULTI_HIT
    assert _effect(t, n, "Water Sport")[0] == E.WATER_SPORT
    assert _effect(t, n, "Disable")[0] == E.DISABLE
    assert _effect(t, n, "Bide")[0] == E.BIDE
    for move in ("Rollout", "Ice Ball"):
        assert _effect(t, n, move)[0] == E.ROLLOUT
    for move in ("Foresight", "Odor Sleuth"):
        assert _effect(t, n, move)[0] == E.FORESIGHT
    # The rows after the 0xFE separator: exactly Normal and Fighting on Ghost.
    foresight = np.argwhere(np.asarray(t.type_chart_foresight))
    pairs = {(n.type_names[a], n.type_names[d]) for a, d in foresight}
    assert pairs == {("NORMAL", "GHOST"), ("FIGHTING", "GHOST")}
    for move in ("Quick Attack", "Mach Punch"):
        assert _effect(t, n, move)[0] == E.PLAIN
        assert int(t.move_priority[n.move(move)]) == 1
    # Effect chance is the move's own column. Hammer Arm's self drop is certain
    # (decomp MOVE_SIDE_EFFECT_ON_HIT) and its chance column is 0.
    assert int(t.move_effect_chance[n.move("Ember")]) == 10
    assert int(t.move_effect_chance[n.move("Hammer Arm")]) == 0


def test_move_flags(tables, names):
    t, n = tables, names

    def flags(m):
        return int(t.move_flags[n.move(m)])

    assert flags("Tackle") == MoveFlag.CONTACT | MoveFlag.PROTECT | MoveFlag.KINGS_ROCK
    assert not flags("Growl") & MoveFlag.KINGS_ROCK  # MOVE_FLAG_TRIGGERS_KINGS_ROCK
    assert flags("Mach Punch") & MoveFlag.PUNCH and flags("Hammer Arm") & MoveFlag.PUNCH
    assert flags("Growl") & MoveFlag.SOUND and flags("Supersonic") & MoveFlag.SOUND
    assert not flags("Ember") & MoveFlag.CONTACT
    assert not flags("Stealth Rock") & MoveFlag.PROTECT


def test_ability_and_item_mapping(tables, names):
    t, n = tables, names
    assert int(t.species_abilities[n.species("Geodude"), 0]) == AbilityId.ROCK_HEAD
    assert int(t.species_abilities[n.species("Geodude"), 1]) == AbilityId.STURDY
    assert int(t.species_abilities[n.species("Cranidos"), 1]) == AbilityId.NONE
    assert int(t.item_effect[n.item("Oran Berry")]) == ItemId.ORAN_BERRY
    assert int(t.item_effect[n.item("Sitrus Berry")]) == ItemId.SITRUS_BERRY
    assert int(t.item_effect[n.item("Scope Lens")]) == ItemId.CRIT_ITEM  # E2 batch 4 (B4-10)
    assert int(t.item_effect[n.item("Iron Ball")]) == ItemId.IRON_BALL  # E1 Batch D (G-56)
    assert int(t.item_effect[n.item("")]) == ItemId.NONE
    assert tuple(int(x) for x in t.item_param[n.item("Oran Berry")]) == (10, 0)
    assert tuple(int(x) for x in t.item_param[n.item("Sitrus Berry")]) == (25, 0)


def test_species_and_nature_values(tables, names):
    t, n = tables, names
    g = n.species("geodude")
    assert [int(x) for x in t.species_base_stats[g]] == [40, 80, 100, 30, 30, 20]
    rock, ground = n.type_names.index("ROCK"), n.type_names.index("GROUND")
    assert [int(x) for x in t.species_types[g]] == [rock, ground]
    assert int(t.type_chart[n.type_names.index("WATER"), rock]) == 20  # tenths
    assert int(t.species_weight[n.species("Onix")]) == 2100
    # Clefairy's base friendship 140 does not fit int8, so the column is int16.
    assert int(t.species_friendship[n.species("Clefairy")]) == 140
    adamant = n.nature_names.index("Adamant")
    assert [int(x) for x in t.nature_mods[adamant]] == [0, 1, 0, -1, 0, 0]
    assert [int(x) for x in t.nature_mods[n.nature_names.index("Hardy")]] == [0] * 6


def test_names_lookup(names):
    n = names
    assert n.move("") == 0 and n.item("") == 0
    assert n.move("tackle") == n.move("TACKLE") == 33
    assert n.ability("rock head") == n.ability_names.index("Rock Head")
    with pytest.raises(KeyError):
        n.move("Not A Move")


def test_names_species_abilities_and_friendship(names):
    n = names
    # Decomp ability ids per species slot; slot 1 is ABILITY_NONE (0) when absent.
    assert n.species_abilities(n.species("Shinx")) == (
        n.ability("Rivalry"),
        n.ability("Intimidate"),
    )
    assert n.species_abilities(n.species("Monferno")) == (n.ability("Blaze"), 0)
    assert n.species_base_friendship(n.species("Monferno")) == 70


def test_growth_tables(tables, names):
    assert int(tables.species_exp_rate[names.species("Abra")]) == 3
    assert int(tables.exp_table[0, 100]) == 1_000_000


def test_ai_effect_and_exp_yield_arrays(tables, names):
    assert int(tables.species_exp_yield[names.species("Bulbasaur")]) == 64
    assert int(tables.move_ai_effect[names.move("Stealth Rock")]) == 266


def test_charmap_loaded(names):
    assert names.charmap[0x0145] == "a"


def test_form_tables(tables, names):
    origin = form_row(tables.species_form_row, names.species("Giratina"), 1)
    assert origin == 501 and int(tables.species_abilities_decomp[origin, 0]) == names.ability(
        "Levitate"
    )
    assert tables.species_base_stats.shape[0] == 508 and len(names.species_names) == 496
    assert tables.species_form_row.shape == (496, 28) and tables.species_form_row.dtype == jnp.int16
    for f in Tables._fields:
        if f.startswith("species_") and f != "species_form_row":
            assert getattr(tables, f).shape[0] == 508, f
    assert names.form(479, "wash") == 2 and names.form_names(479)[1] == "Heat"
    assert names.form_names(479)[0] == names.species_names[479] and names.form(479, "rotom") == 0
    assert names.form_names(1) == (names.species_names[1],)
    assert names.form(479, 0) == 0
    assert names.species_abilities(487, 1) == (names.ability("Levitate"), 0)
    assert names.species_base_friendship(479, 2) == 70
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        names.species_abilities(479, 6)
    with pytest.raises(ValueError, match="Heat"):
        names.form(479, "Plasma")


def test_form_row_lookup():
    grid = [[0, -1], [1, 5]]
    assert form_row(grid, 1, 1) == 5 and form_row(np.asarray(grid), 1, 0) == 1
    for species, form in ((0, 1), (1, -1), (1, 28)):
        with pytest.raises(ValueError, match=f"species {species} has no form {form}"):
            form_row(grid, species, form)


def test_import_forms_does_not_import_jax():
    code = "import sys, nuzline.data.forms; assert 'jax' not in sys.modules, 'jax imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_names_form_checks_the_species_range(names):
    # Task 5 review: a negative id wrapped round species_names, a large one raised IndexError.
    for species in (-1, 496):
        with pytest.raises(ValueError, match=f"species {species} out of range"):
            names.form_names(species)
        with pytest.raises(ValueError, match=f"species {species} out of range"):
            names.form(species, "Wash")
    with pytest.raises(ValueError, match=r"ROTOM has no form 6 \(forms: ROTOM, Heat"):
        names.form(479, 6)


def test_committed_manifest_is_the_code_version(
    names,
):  # the committed manifest; `names` skips when tables are absent
    assert json.loads((TABLES / "manifest.json").read_text("utf-8"))["version"] == MANIFEST_VERSION


@pytest.mark.parametrize("manifest", ['{"version": 7}', "{}"])
def test_old_tables_are_refused(tmp_path, manifest):  # no npz needed: check_manifest runs first
    (tmp_path / "manifest.json").write_text(manifest)
    for load in (load_tables, load_names, load_learnsets):
        with pytest.raises(ValueError, match="re-run tools/extract_decomp.py"):
            load(tmp_path)


STALE_NPZ = (
    r"tables\.npz is version \w+; .*re-run tools/extract_decomp\.py and tools/build_tables\.py"
)


def _stale_npz(dir: Path, version: int | None) -> None:
    """A current manifest beside a ``tables.npz`` built by an older nuzline: the tracked
    manifest is pulled while the git-ignored npz stays (final review, stale tables)."""
    (dir / "manifest.json").write_text(json.dumps({"version": MANIFEST_VERSION}))
    arrays = {"species_base_stats": np.zeros((496, 6), np.int16)}
    if version is not None:
        arrays["version"] = np.int16(version)
    np.savez_compressed(dir / "tables.npz", **arrays)


@pytest.mark.parametrize("version", [None, 7])
def test_stale_npz_beside_current_manifest_is_refused(tmp_path, version):
    _stale_npz(tmp_path, version)
    with pytest.raises(ValueError, match=STALE_NPZ):
        load_tables(tmp_path)


def test_built_npz_records_its_version(tables):
    with np.load(TABLES / "tables.npz") as z:
        assert "version" in z.files and int(z["version"]) == MANIFEST_VERSION
    assert "version" not in Tables._fields


def test_item_hold_effect_matches_decomp(tables, names):
    # survey 1.5: the raw decomp hold effect, whether or not the engine models it.
    for item, effect in (
        ("Focus Sash", 103),
        ("Quick Claw", 52),
        ("Chilan Berry", 35),
        ("Honey", 0),
    ):
        assert int(tables.item_hold_effect[names.item(item)]) == effect, item


def test_item_hold_param_matches_decomp(tables, names):
    # K3d-exp T5: ItemData.effectParam as stored (res/items/data/*.json "effectParam"), whether
    # or not the engine models the item; BattleScript_CalcEffortValues reads a Power item's
    # (src/battle/battle_script.c:10228).
    for item, param in (
        ("Power Bracer", 4),
        ("Power Weight", 4),
        ("Exp. Share", 0),
        ("Quick Claw", 20),
        ("Oran Berry", 10),
    ):
        assert int(tables.item_hold_param[names.item(item)]) == param, item


def test_item_fling_and_natural_gift_match_decomp(tables, names, kaizo_tables, kaizo_names):
    # E2 batch 13: ItemData.flingPower / flingEffect / naturalGiftPower / naturalGiftType as
    # stored (res/items/data/*.json; include/constants/items.h:86-116 FLING_EFFECT_*; a null
    # Natural Gift type is 0x1F, tools/dataproc/src/itemproc.c:242).
    for item, power, effect, gift, gift_type in (
        ("Oran Berry", 10, 7, 60, 3),  # FLING_EFFECT_HP_RESTORE, TYPE_POISON
        ("King’s Rock", 30, 26, 0, 0x1F),  # FLING_EFFECT_FLINCH
        ("White Herb", 10, 24, 0, 0x1F),  # FLING_EFFECT_STATDOWN_RESTORE
        ("Iron Ball", 130, 0, 0, 0x1F),
        ("Potion", 30, 0, 0, 0x1F),
    ):
        i = names.item(item)
        got = (
            int(tables.item_fling_power[i]),
            int(tables.item_fling_effect[i]),
            int(tables.item_gift_power[i]),
            int(tables.item_gift_type[i]),
        )
        assert got == (power, effect, gift, gift_type), item
    # The Kaizo ROM's own item data (data/intermediate_kaizo/rom/items.json): Town Map flings
    # for 120 with FLING_EFFECT_FLINCH.
    i = kaizo_names.item("Town Map")
    assert (int(kaizo_tables.item_fling_power[i]), int(kaizo_tables.item_fling_effect[i])) == (
        120,
        26,
    )


def test_kaizo_power_items_are_macho_braces(kaizo_tables, kaizo_names):
    # data/intermediate_kaizo/items.json: Kaizo gives the six Power items Macho Brace's hold
    # effect (HOLD_EFFECT_EVS_UP_SPEED_DOWN, generated/item_hold_effects.txt:51) and param 0,
    # so the EV gain must read each game's own item data, never a hard-coded 4.
    for item in ("Power Bracer", "Power Belt", "Power Lens", "Power Band", "Power Anklet"):
        i = kaizo_names.item(item)
        assert int(kaizo_tables.item_hold_effect[i]) == 50, item
        assert int(kaizo_tables.item_hold_param[i]) == 0, item


# --- K3d-exp T2: EV yields, learnsets, HM flags ---------------------------------------------

DECOMP = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"
STAT_KEYS = ("hp", "attack", "defense", "special_attack", "special_defense", "speed")  # ids.Stat


def _ev(hp=0, atk=0, df=0, spa=0, spd=0, spe=0):
    return [hp, atk, df, spa, spd, spe]


def test_ev_yield_spot_rows(tables, names):
    # res/pokemon/<species>[/forms/<form>]/data.json ev_yields, by personal row (per form).
    wormadam = names.species("Wormadam")
    for row, want in (
        (names.species("Bonsly"), _ev(df=1)),
        (names.species("Geodude"), _ev(df=1)),
        (names.form_row(wormadam, 0), _ev(spd=2)),  # Plant: the species row
        (names.form_row(wormadam, names.form(wormadam, "Sandy")), _ev(df=2)),
        (names.form_row(wormadam, names.form(wormadam, "Trash")), _ev(df=1, spd=1)),
    ):
        assert np.asarray(tables.species_ev_yield[row]).tolist() == want, row
    assert tables.species_ev_yield.dtype == jnp.int8 and tables.species_ev_yield.shape == (508, 6)


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_ev_yield_equals_the_decomp_for_every_row(tables):
    from tools.extract_decomp import _personal_rows

    got = np.asarray(tables.species_ev_yield).tolist()
    rows = 0
    for row, label, d in _personal_rows(DECOMP):
        assert got[row] == [d["ev_yields"][k] for k in STAT_KEYS], label
        rows += 1
    assert rows == len(got) == 508


def _packed_learnset(t, row):
    lv = np.asarray(t.learnset_level[row]).tolist()
    mv = np.asarray(t.learnset_move[row]).tolist()
    n = lv.index(0) if 0 in lv else len(lv)
    assert lv[n:] == [0] * (len(lv) - n) and mv[n:] == [0] * (len(mv) - n), row
    return tuple(zip(lv[:n], mv[:n], strict=True))


def _check_learnsets(t, learnsets, form_rows):
    assert t.learnset_move.shape == t.learnset_level.shape == (508, 21)
    assert t.learnset_move.dtype == jnp.int16 and t.learnset_level.dtype == jnp.int8
    seen = set()
    for species, row_forms in enumerate(form_rows):
        for form, row in enumerate(row_forms):
            if row < 0 or row in seen:
                continue
            seen.add(row)
            # Source order, never sorted: Pokemon_LevelUpMove scans the whole list for each
            # level (src/pokemon.c:3945-3956).
            assert _packed_learnset(t, row) == learnsets.level_up(species, form), row
    assert seen == set(range(508))


def test_learnsets_packed_in_source_order(tables, learnsets, vanilla_manifest):
    _check_learnsets(tables, learnsets, vanilla_manifest["species_form_row"])
    # Vanilla row 354 is unsorted (level 8 at index 4 before level 5 at index 5) and stays so.
    levels = [lv for lv, _ in _packed_learnset(tables, 354)]
    assert levels[4:6] == [8, 5] and levels != sorted(levels)
    assert max(len(_packed_learnset(tables, r)) for r in range(508)) == 20


def test_kaizo_learnsets_packed_in_source_order(
    kaizo_tables, kaizo_learnsets, kaizo_manifest, kaizo_names
):
    _check_learnsets(kaizo_tables, kaizo_learnsets, kaizo_manifest["species_form_row"])
    assert len(_packed_learnset(kaizo_tables, 359)) == 21  # the one full row
    slash = kaizo_names.move("Slash")
    assert slash == 163
    assert (17, slash) in _packed_learnset(kaizo_tables, kaizo_names.species("Sandshrew"))


HMS = ("Cut", "Fly", "Surf", "Strength", "Defog", "Rock Smash", "Waterfall", "Rock Climb")


def test_move_is_hm(tables, names, vanilla_manifest):
    # Item_IsHMMove (src/item.c:306-315) tests sTMHMMoves[NUM_TMS + i] for i < NUM_HMS.
    hm = np.asarray(tables.move_is_hm)
    assert hm.dtype == np.int8 and hm.shape == (len(names.move_names),)
    assert set(np.flatnonzero(hm).tolist()) == {names.move(m) for m in HMS}
    assert set(np.flatnonzero(hm).tolist()) == set(vanilla_manifest["tm_moves"][92:100])


def test_kaizo_move_is_hm(kaizo_tables, kaizo_manifest, tables):
    # Kaizo's sTMHMMoves keeps the HMs (rom_diff.md: only TMs changed), so the flags are
    # vanilla's.
    hm = np.asarray(kaizo_tables.move_is_hm)
    assert set(np.flatnonzero(hm).tolist()) == set(kaizo_manifest["tm_moves"][92:100])
    assert np.array_equal(hm, np.asarray(tables.move_is_hm))
