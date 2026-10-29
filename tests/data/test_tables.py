from pathlib import Path

import jax.numpy as jnp
import pytest

from nuzline.data.tables import load_names, load_tables
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, Stat, Status

TABLES = Path("data/gen4_platinum")
pytestmark = pytest.mark.skipif(not (TABLES / "tables.npz").exists(), reason="tables not built")

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


def test_dtypes_and_shapes():
    t, n = load_tables(TABLES), load_names(TABLES)
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


def test_effect_mapping():
    t, n = load_tables(TABLES), load_names(TABLES)
    assert int(t.move_effect[n.move("Tackle")]) == EffectId.PLAIN
    assert int(t.move_effect[n.move("Stealth Rock")]) == EffectId.STEALTH_ROCK
    assert int(t.move_effect[n.move("Leer")]) == EffectId.STAT_FOE
    assert tuple(int(x) for x in t.move_param[n.move("Leer")]) == (2, -1, 0)
    assert int(t.move_effect[n.move("Metronome")]) == EffectId.UNSUPPORTED


def test_status_moves_needing_unmodelled_immunity_are_unsupported():
    """Thunder Wave is the one status move the decomp runs through the type chart
    (Ground immunity, battle_controller_player.c CheckTypeChart); that immunity is
    not in M1, so the move must not be simulated as a plain paralysis move. Glare
    and Stun Spore skip the type chart in Gen 4 and stay supported."""
    t, n = load_tables(TABLES), load_names(TABLES)
    assert _effect(t, n, "Thunder Wave")[0] == EffectId.UNSUPPORTED
    for move in ("Glare", "Stun Spore"):
        assert _effect(t, n, move) == (EffectId.STATUS_FOE, (Status.PARALYSIS, 0, 0))


def test_every_m1a_move_is_supported():
    t, n = load_tables(TABLES), load_names(TABLES)
    unsupported = [m for m in M1A_MOVES if _effect(t, n, m)[0] == EffectId.UNSUPPORTED]
    assert unsupported == []


def test_m1a_effect_params():
    t, n = load_tables(TABLES), load_names(TABLES)
    E = EffectId
    assert _effect(t, n, "Growth") == (E.STAT_SELF, (Stat.SPA, 1, 0))
    assert _effect(t, n, "Defense Curl") == (E.STAT_SELF, (Stat.DEF, 1, 0))
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
    assert _effect(t, n, "Take Down") == (E.RECOIL, (25, 0, 0))
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
    for move in ("Quick Attack", "Mach Punch"):
        assert _effect(t, n, move)[0] == E.PLAIN
        assert int(t.move_priority[n.move(move)]) == 1
    # Effect chance is the move's own column. Hammer Arm's self drop is certain
    # (decomp MOVE_SIDE_EFFECT_ON_HIT) and its chance column is 0.
    assert int(t.move_effect_chance[n.move("Ember")]) == 10
    assert int(t.move_effect_chance[n.move("Hammer Arm")]) == 0


def test_move_flags():
    t, n = load_tables(TABLES), load_names(TABLES)

    def flags(m):
        return int(t.move_flags[n.move(m)])

    assert flags("Tackle") == MoveFlag.CONTACT | MoveFlag.PROTECT
    assert flags("Mach Punch") & MoveFlag.PUNCH and flags("Hammer Arm") & MoveFlag.PUNCH
    assert flags("Growl") & MoveFlag.SOUND and flags("Supersonic") & MoveFlag.SOUND
    assert not flags("Ember") & MoveFlag.CONTACT
    assert not flags("Stealth Rock") & MoveFlag.PROTECT


def test_ability_and_item_mapping():
    t, n = load_tables(TABLES), load_names(TABLES)
    assert int(t.species_abilities[n.species("Geodude"), 0]) == AbilityId.ROCK_HEAD
    assert int(t.species_abilities[n.species("Geodude"), 1]) == AbilityId.STURDY
    assert int(t.species_abilities[n.species("Cranidos"), 1]) == AbilityId.NONE
    assert int(t.item_effect[n.item("Oran Berry")]) == ItemId.ORAN_BERRY
    assert int(t.item_effect[n.item("Sitrus Berry")]) == ItemId.SITRUS_BERRY
    assert int(t.item_effect[n.item("Leftovers")]) == ItemId.UNSUPPORTED
    assert int(t.item_effect[n.item("")]) == ItemId.NONE
    assert tuple(int(x) for x in t.item_param[n.item("Oran Berry")]) == (10, 0)
    assert tuple(int(x) for x in t.item_param[n.item("Sitrus Berry")]) == (25, 0)


def test_species_and_nature_values():
    t, n = load_tables(TABLES), load_names(TABLES)
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


def test_names_lookup():
    n = load_names(TABLES)
    assert n.move("") == 0 and n.item("") == 0
    assert n.move("tackle") == n.move("TACKLE") == 33
    assert n.ability("rock head") == n.ability_names.index("Rock Head")
    with pytest.raises(KeyError):
        n.move("Not A Move")


def test_names_species_abilities_and_friendship():
    n = load_names(TABLES)
    # Decomp ability ids per species slot; slot 1 is ABILITY_NONE (0) when absent.
    assert n.species_abilities(n.species("Shinx")) == (
        n.ability("Rivalry"),
        n.ability("Intimidate"),
    )
    assert n.species_abilities(n.species("Monferno")) == (n.ability("Blaze"), 0)
    assert n.species_base_friendship(n.species("Monferno")) == 70
