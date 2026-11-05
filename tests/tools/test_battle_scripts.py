"""The battle-structure evaluator shared by the decomp and ROM extractors (synthetic ops)."""

from pathlib import Path

import pytest

from nuzline.data.trainer import PartnerVariant
from tools.battle_scripts import (
    PARTNER_VAR,
    STARTERS,
    BattleStructure,
    Op,
    battle_structures,
    partner_variants,
    var_ids,
)

DECOMP = Path("external/pokeplatinum")
R, V = 0x800C, 0x8004  # VAR_RESULT, VAR_0x8004


def _team(label, sets, ret):
    """A starter branch: SetVar V, t; GoToIfEq R, starter, ret ... ; the last SetVar falls
    through to Return."""
    ops = [Op("label", (label,)), Op("GetPlayerStarterSpecies", (R,))]
    for i, (t, s) in enumerate(sets):
        ops.append(Op("SetVar", (V, t)))
        if i < len(sets) - 1:
            ops.append(Op("GoToIfEq", (R, s, ret)))
    return [*ops, Op("Return", ())]


# scripts_jubilife_city.s:901-926: a gender branch, then a starter branch per counterpart.
JUBILIFE_OPS = [
    Op("label", ("SetTeam",)),
    Op("GetPlayerGender", (R,)),
    Op("GoToIfEq", (R, 0, "Dawn")),
    Op("GoToIfEq", (R, 1, "Lucas")),
    Op("End", ()),
    *_team("Dawn", ((11, 390), (12, 393), (13, 387)), "Ret"),
    *_team("Lucas", ((21, 390), (22, 393), (23, 387)), "Ret"),
    Op("label", ("Ret",)),
    Op("Return", ()),
]

# scripts_spear_pillar.s:295-303: one starter branch whose last SetVar falls into the label.
SPEAR_PILLAR_OPS = [
    Op("label", ("SetTeam",)),
    Op("GetPlayerStarterSpecies", (R,)),
    Op("SetVar", (V, 31)),
    Op("GoToIfEq", (R, 390, "Ret")),
    Op("SetVar", (V, 32)),
    Op("GoToIfEq", (R, 393, "Ret")),
    Op("SetVar", (V, 33)),
    Op("label", ("Ret",)),
    Op("Return", ()),
]


def test_partner_variants_by_gender_and_starter():
    # Jubilife shape: gender branch, then a starter branch per counterpart
    assert set(partner_variants(JUBILIFE_OPS, "SetTeam", 0x8004)) == {
        PartnerVariant(t, s, g)
        for g, ts in ((0, (11, 12, 13)), (1, (21, 22, 23)))
        for t, s in zip(ts, (390, 393, 387), strict=True)
    }


def test_partner_variants_collapse_gender():
    # Spear Pillar shape: the same team for both genders
    assert set(partner_variants(SPEAR_PILLAR_OPS, "SetTeam", 0x8004)) == {
        PartnerVariant(t, s, None) for t, s in ((31, 390), (32, 393), (33, 387))
    }


def test_partner_variants_collapse_both():
    ops = [Op("label", ("A",)), Op("SetVar", (V, 7)), Op("Return", ())]
    assert partner_variants(ops, "A", V) == (PartnerVariant(7, None, None),)


def test_partner_variants_order_is_stable():
    got = partner_variants(JUBILIFE_OPS, "SetTeam", V)
    assert [(p.player_gender, p.starter) for p in got] == [
        (g, s) for g in (0, 1) for s in sorted(STARTERS)
    ]


def test_partner_routine_with_unknown_op_raises():
    ops = [Op("label", ("A",)), Op("other", ("Foo",)), Op("SetVar", (V, 7)), Op("Return", ())]
    with pytest.raises(ValueError, match="Foo"):
        partner_variants(ops, "A", V)


def test_partner_routine_loop_raises():
    ops = [Op("label", ("A",)), Op("SetVar", (V, 7)), Op("GoTo", ("A",))]
    with pytest.raises(ValueError, match="loop"):
        partner_variants(ops, "A", V)


def test_partner_routine_without_the_var_raises():
    ops = [Op("label", ("A",)), Op("SetVar", (R, 7)), Op("Return", ())]
    with pytest.raises(ValueError, match="without setting"):
        partner_variants(ops, "A", V)


def test_partner_routine_running_off_the_end_raises():
    ops = [Op("label", ("A",)), Op("SetVar", (V, 7))]
    with pytest.raises(ValueError, match="end"):
        partner_variants(ops, "A", V)


def test_partner_routine_follows_a_call():
    ops = [
        Op("label", ("A",)),
        Op("Call", ("B",)),
        Op("Return", ()),
        Op("label", ("B",)),
        Op("SetVar", (V, 9)),
        Op("Return", ()),
    ]
    assert partner_variants(ops, "A", V) == (PartnerVariant(9, None, None),)


def test_fall_through_into_label():
    ops = [Op("label", ("@0",)), Op("SetVar", (0x8004, 5)), Op("label", ("@9",)), Op("Return", ())]
    assert partner_variants(ops, "@0", 0x8004) == (PartnerVariant(5, None, None),)


def test_trainer_in_two_structures_raises():
    scripts = {
        "A": [Op("StartTrainerBattle", (5, 6))],
        "B": [Op("StartTrainerBattle", (5, 7))],
    }
    with pytest.raises(ValueError, match="trainer 5"):
        battle_structures(scripts, {})


def test_two_trainer_pair():
    scripts = {"A": [Op("StartTrainerBattle", (5, 6)), Op("StartTrainerBattle", (8, 0))]}
    got = battle_structures(scripts, {})
    assert got == {
        5: BattleStructure("double_two_trainer", 6, ()),
        6: BattleStructure("double_two_trainer", 5, ()),
    }


def test_the_same_pair_twice_is_one_structure():
    # scripts_restaurant.s fights each pair from two scripts, and headers share files.
    pair = [Op("StartTrainerBattle", (5, 6))]
    got = battle_structures({"A": pair + pair, "B": pair}, {})
    assert got[5] == BattleStructure("double_two_trainer", 6, ())


def test_variable_pair_is_not_recorded():
    # scripts_battles.s:173: two trainers spotting the player (Ruling 4).
    scripts = {"A": [Op("StartTrainerBattle", (V, 0x8005))]}
    assert battle_structures(scripts, {}) == {}


def test_tag_battle_partner_from_the_call_before_it():
    ops = [
        Op("label", ("Battle",)),
        Op("Call", ("SetTeam",)),
        Op("StartTagBattle", (V, 40, 41)),
        Op("End", ()),
        *SPEAR_PILLAR_OPS,
    ]
    got = battle_structures({"A": ops}, {})
    team = partner_variants(SPEAR_PILLAR_OPS, "SetTeam", V)
    assert got == {40: BattleStructure("multi", 41, team), 41: BattleStructure("multi", 40, team)}


def test_tag_battle_with_a_constant_partner():
    got = battle_structures({"A": [Op("StartTagBattle", (3, 40, 41))]}, {})
    assert got[41] == BattleStructure("multi", 40, (PartnerVariant(3, None, None),))


def test_tag_battle_partner_var_without_a_call_raises():
    ops = [Op("SetVar", (V, 3)), Op("StartTagBattle", (V, 40, 41))]
    with pytest.raises(ValueError, match="Call"):
        battle_structures({"A": ops}, {})


def test_walking_partner_map():
    # scripts_iron_island_b2f_left_room.s: Riley walks with the player, and a pair started
    # on his map is a multi battle with him (Ruling 2).
    ops = [
        Op("SetVar", (PARTNER_VAR, 9)),
        Op("SetHasPartner", ()),
        Op("StartTrainerBattle", (5, 6)),
    ]
    got = battle_structures({"A": ops}, {"A": {5, 6, 50}})
    riley = (PartnerVariant(9, None, None),)
    assert got == {
        5: BattleStructure("multi", 6, riley),
        6: BattleStructure("multi", 5, riley),
        50: BattleStructure("multi", None, riley),
    }
    assert 9 not in got


def test_placed_enemies_off_a_partner_map_are_single():
    assert battle_structures({"A": [Op("End", ())]}, {"A": {50}}) == {}


def test_partner_map_without_a_partner_id_raises():
    ops = [Op("SetHasPartner", ())]
    assert battle_structures({"A": ops}, {"A": set()}) == {}  # Amity Square: no enemies
    with pytest.raises(ValueError, match="A.*SetHasPartner"):
        battle_structures({"A": ops}, {"A": {50}})


def test_tag_enemy_on_two_partner_maps_raises():
    walk = [Op("SetVar", (PARTNER_VAR, 9)), Op("SetHasPartner", ())]
    with pytest.raises(ValueError, match="trainer 40"):
        battle_structures({"A": [Op("StartTagBattle", (3, 40, 41))], "B": walk}, {"B": {40}})


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_var_ids():
    ids = var_ids(DECOMP)
    assert ids["VAR_0x8004"] == 0x8004 and ids["VAR_RESULT"] == 0x800C
    assert ids["VAR_PARTNER_TRAINER_ID"] == PARTNER_VAR
