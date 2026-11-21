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
    lone_walk_formats,
    partner_variants,
    script_enemies,
    structure_fields,
    var_ids,
)
from tools.extract_decomp import _script_file_ops

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


def test_self_pair_is_a_one_trainer_double():
    # StartTrainerBattle t, t: Encounter_NewVsTrainer's BATTLE_TYPE_TRAINER_DOUBLES
    # (src/encounter.c:736-737), Kaizo's Volkner and Cyrus.
    got = battle_structures({"A": [Op("StartTrainerBattle", (5, 5))]}, {})
    assert got == {5: BattleStructure("double", None, ())}


def test_self_pair_beside_another_pair_raises():
    scripts = {"A": [Op("StartTrainerBattle", (5, 5)), Op("StartTrainerBattle", (5, 6))]}
    with pytest.raises(ValueError, match="trainer 5"):
        battle_structures(scripts, {})


def test_the_same_pair_twice_is_one_structure():
    # scripts_restaurant.s fights each pair from two scripts, and headers share files.
    pair = [Op("StartTrainerBattle", (5, 6))]
    got = battle_structures({"A": pair + pair, "B": pair}, {})
    assert got[5] == BattleStructure("double_two_trainer", 6, ())


def test_variable_pair_resolves_through_setvar():
    ops = [
        Op("SetVar", (V, 5)),
        Op("SetVar", (0x8005, 6)),
        Op("StartTrainerBattle", (V, 0x8005)),
    ]
    assert battle_structures({"A": ops}, {}) == {
        5: BattleStructure("double_two_trainer", 6, ()),
        6: BattleStructure("double_two_trainer", 5, ()),
    }


def test_unresolved_variable_pair_raises():
    # Like scripts_battles.s:173 (Ruling 4), but on a map script: never skipped silently.
    scripts = {"A": [Op("StartTrainerBattle", (V, 0x8005))]}
    with pytest.raises(ValueError, match="map A"):
        battle_structures(scripts, {})


# Final review: _resolve scanned back past ops that may rewrite the var (an `other`
# such as decomp `SetVar V, VAR_X` or ROM SetVarFromVar, a Call whose callee writes it)
# and across GoTo/End/Return, so a stale SetVar was read. Each now refuses.
SUB = [Op("label", ("Sub",)), Op("other", ("SetVar",)), Op("Return", ())]


@pytest.mark.parametrize(
    "between, extra",
    [
        (Op("other", ("SetVar",)), []),
        (Op("other", ("SCRCMD_SETVARFROMVAR",)), []),
        (Op("Call", ("Sub",)), SUB),
        (Op("GoTo", ("Sub",)), SUB),
        (Op("End", ()), []),
        (Op("Return", ()), []),
        (Op("GetPlayerGender", (V,)), []),
        (Op("GetPlayerStarterSpecies", (V,)), []),
        (Op("GoToIfEq", (R, 1, "Nowhere")), []),
        (Op("StartTrainerBattle", (9, 0)), []),
    ],
    ids=lambda x: getattr(x, "name", None) and f"{x.name}{x.args}",
)
def test_stale_setvar_is_refused(between, extra):
    ops = [
        Op("label", ("Battle",)),
        Op("SetVar", (V, 5)),
        between,
        Op("StartTrainerBattle", (V, 7)),
        Op("End", ()),
        *extra,
    ]
    with pytest.raises(ValueError, match=r"map A: .*StartTrainerBattle"):
        battle_structures({"A": ops}, {})
    single = [*ops[:3], Op("StartTrainerBattle", (V, 0)), *ops[4:]]
    with pytest.raises(ValueError, match=r"map A: .*StartTrainerBattle"):
        script_enemies(single, "map A")


def test_resolve_passes_ops_that_cannot_write_the_var():
    # Another var's SetVar or GetPlayerGender, SetHasPartner, and a GoToIfEq to a label
    # outside the block: on the fall-through path the SetVar still holds.
    ops = [
        Op("label", ("Battle",)),
        Op("SetVar", (V, 5)),
        Op("SetVar", (0x8005, 6)),
        Op("GetPlayerGender", (R,)),
        Op("GoToIfEq", (R, 1, "Other")),
        Op("SetHasPartner", ()),
        Op("StartTrainerBattle", (V, 0)),
        Op("End", ()),
        Op("label", ("Other",)),
        Op("End", ()),
    ]
    assert script_enemies(ops, "map A") == {5}


def test_single_variable_battle_is_not_a_pair():
    # scripts_pokemon_mansion.s:183: StartTrainerBattle VAR_0x8007 (enemyTrainer2 = 0).
    assert battle_structures({"A": [Op("StartTrainerBattle", (0x8007, 0))]}, {}) == {}


def test_jump_to_unknown_label_raises_value_error():
    ops = [Op("label", ("A",)), Op("GoTo", ("Elsewhere",))]
    with pytest.raises(ValueError, match="unknown label 'Elsewhere'"):
        partner_variants(ops, "A", V)
    ops = [Op("label", ("A",)), Op("Call", ("Elsewhere",)), Op("Return", ())]
    with pytest.raises(ValueError, match="unknown label 'Elsewhere'"):
        partner_variants(ops, "A", V)
    ops = [
        Op("label", ("A",)),
        Op("GetPlayerGender", (R,)),
        Op("GoToIfEq", (R, 0, "Elsewhere")),
        Op("SetVar", (V, 1)),
        Op("Return", ()),
    ]
    with pytest.raises(ValueError, match="unknown label 'Elsewhere'"):
        partner_variants(ops, "A", V)


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


def test_tag_battle_partner_var_without_a_call_unknown_when_allowed():
    # Kaizo's Veilstone City tag battle reaches its partner SetVar through GoTos (K2b
    # Task 8): with unknown_partner the pair is kept and the partner is None (not known).
    ops = [Op("SetVar", (V, 3)), Op("StartTagBattle", (V, 40, 41))]
    got = battle_structures({"A": ops}, {}, unknown_partner=True)
    assert got == {40: BattleStructure("multi", 41, None), 41: BattleStructure("multi", 40, None)}
    assert structure_fields(got[40], "single") == {
        "format": "multi",
        "opponent_2": 41,
        "partner": None,
        "walking_partner": [],
    }


def test_conflicting_partner_claims_unknown_when_allowed():
    # Kaizo Lake Verity (map 312): one tag battle per partner on a walking-partner map.
    ops = [
        Op("SetHasPartner"),
        Op("SetVar", (PARTNER_VAR, 800)),
        Op("SetVar", (PARTNER_VAR, 801)),
        Op("StartTagBattle", (800, 40, 41)),
        Op("StartTagBattle", (801, 40, 41)),
    ]
    with pytest.raises(ValueError, match=r"partner ids \[800, 801\], not one"):
        battle_structures({"A": ops}, {"A": {40, 41}})
    got = battle_structures({"A": ops}, {"A": {40, 41}}, unknown_partner=True)
    both = (800, 801)  # the tags are in the walk map's own script: fought under the flag
    assert got == {
        40: BattleStructure("multi", 41, None, both),
        41: BattleStructure("multi", 40, None, both),
    }
    pairs = [Op("StartTrainerBattle", (40, 41)), Op("StartTrainerBattle", (40, 42))]
    with pytest.raises(ValueError, match="trainer 40 is claimed"):
        battle_structures({"A": pairs}, {}, unknown_partner=True)


def test_pair_and_tag_claims_raise_even_with_unknown_partner():
    # Kaizo Spear Pillar (scr_seq 236): a two-trainer battle and a tag battle on the same
    # enemies disagree on the format, not on the partner (fix round 1, I1).
    ops = [Op("StartTrainerBattle", (40, 41)), Op("StartTagBattle", (3, 40, 41))]
    with pytest.raises(ValueError, match="trainer 40 is claimed by 2"):
        battle_structures({"A": ops}, {}, unknown_partner=True)
    got = battle_structures({"A": ops}, {}, unknown_partner=True, refused={40: "x", 41: "x"})
    assert got == {
        40: BattleStructure("unknown", 41, None),
        41: BattleStructure("unknown", 40, None),
    }
    with pytest.raises(ValueError, match="without conflicting claims.*42"):
        battle_structures({"A": ops}, {}, unknown_partner=True, refused={40: "x", 41: "x", 42: "x"})


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
    # A lone enemy (50) is BATTLE_TYPE_TRAINER ORed with the partner's header type
    # (encounter.c:738-739, trainer_data.c:31-50), not a multi; the pair stays multi.
    assert got == {
        5: BattleStructure("multi", 6, riley, (9,)),
        6: BattleStructure("multi", 5, riley, (9,)),
        50: BattleStructure("walk_alone", None, (), (9,)),
    }
    assert 9 not in got


def _walk_rows(partner_formats):
    """Row 50 (a double-header lone enemy on a walk map) beside its partner rows."""
    s50 = BattleStructure("walk_alone", None, (), tuple(partner_formats))
    rows = [{"id": 50, **structure_fields(s50, "double")}]
    for p, fmt in partner_formats.items():
        rows.append({"id": p, **structure_fields(None, fmt)})
    return rows


def test_lone_walk_enemy_takes_the_partners_header_type():
    # trainer_data.c:31-50: the last trainer loaded decides the header type, and with e2 = 0
    # and a walking partner that is the partner (encounter.c:751-753); the enemy's own
    # double header is not used.
    rows = _walk_rows({9: "single"})
    lone_walk_formats(rows)
    assert rows[0]["format"] == "single"
    assert rows[0]["opponent_2"] is None and rows[0]["partner"] == []
    assert rows[0]["walking_partner"] == [9]
    rows = _walk_rows({9: "double"})
    lone_walk_formats(rows)
    assert rows[0]["format"] == "double"
    rows = _walk_rows({9: "single", 10: "double"})
    with pytest.raises(ValueError, match="trainer 50"):
        lone_walk_formats(rows)


def test_walk_map_tag_keeps_multi_and_the_walk_ids():
    # The walk map runs the tag itself: fought under the flag (scrcmd_trainer.c:133-135).
    walk = [
        Op("SetVar", (PARTNER_VAR, 9)),
        Op("SetHasPartner", ()),
        Op("SetVar", (V, 3)),
        Op("StartTagBattle", (V, 40, 41)),
    ]
    got = battle_structures({"A": walk}, {"A": {40, 41}}, unknown_partner=True)
    assert got[40] == BattleStructure("multi", 41, None, (9,))
    # The same tag in another map's script, the enemy only placed on the walk map: not under it.
    tag = [Op("SetVar", (V, 3)), Op("StartTagBattle", (V, 40, 41))]
    walk = [Op("SetVar", (PARTNER_VAR, 9)), Op("SetHasPartner", ())]
    got = battle_structures({"A": walk, "B": tag}, {"A": {40}}, unknown_partner=True)
    assert got[40] == BattleStructure("multi", 41, None, ())


def test_scripted_pair_on_walk_map_is_multi():
    ops = [
        Op("SetVar", (PARTNER_VAR, 9)),
        Op("SetHasPartner", ()),
        Op("StartTrainerBattle", (5, 6)),
    ]
    got = battle_structures({"A": ops}, {"A": {5, 6}})
    riley = (PartnerVariant(9, None, None),)
    assert got[5] == BattleStructure("multi", 6, riley, (9,))
    assert got[6] == BattleStructure("multi", 5, riley, (9,))


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


def test_decomp_script_line_splits_on_any_whitespace(tmp_path):
    # Final review: `StartTrainerBattle\tTRAINER_X` was split on " " only and became
    # `other`, losing its battle. (_script_file_ops needs no decomp checkout.)
    path = tmp_path / "s.s"
    path.write_text(
        "Battle:\n\tStartTrainerBattle\tTRAINER_A,  TRAINER_B\n"
        "    SetVar\t VAR_0x8004, TRAINER_A\n\tEnd\n",
        "utf-8",
    )
    symbols = {"TRAINER_A": 5, "TRAINER_B": 6, "VAR_0x8004": V}
    assert _script_file_ops(path, symbols) == [
        Op("label", ("Battle",)),
        Op("StartTrainerBattle", (5, 6)),
        Op("SetVar", (V, 5)),
        Op("End", ()),
    ]
