"""E1 Batch A (rulings G-40 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

Family 1: trainer Pokemon in a data form (G-41).
Family 3: trainer bag items by their party params (G-43).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.trainer import IntermediateData, build_trainer_party, default_moves
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.bag import bag_item_category, refused_bag_params
from nuzline.engine.format import SINGLES
from nuzline.engine.ids import AbilityId, ItemAiCategory, ItemParam, Stat, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import initial_arrays
from nuzline.engine.stats import calc_stats
from tests.conftest import KAIZO_INTERMEDIATE
from tests.engine.helpers import INTERMEDIATE, _inter, assert_matches_reference, roark_state

# --- Family 1: trainer Pokemon in a data form (G-41) ----------------------------------------


def _slot(tables, opp, slot):
    """The opponent's party slot ``slot`` as ``initial_arrays`` builds it."""
    a = initial_arrays(SINGLES, tables, [list(opp), opp])
    return {k: np.asarray(v)[1, slot] for k, v in a.items() if k in _MON}


_MON = ("stats", "max_hp", "types", "ability", "ability_form", "form")


def _calc(tables, spec, row):
    stats = calc_stats(
        np.asarray(tables.species_base_stats[row]),
        np.asarray(spec.ivs),
        np.asarray(spec.evs),
        spec.level,
        np.asarray(tables.nature_mods[spec.nature]),
        species=spec.species,
    )
    return int(stats[Stat.HP]), list(stats[Stat.ATK : Stat.SPE + 1])


def test_kaizo_data_forms_take_the_form_stats_and_the_base_ability(kaizo_tables, kaizo_trainers):
    # Kaizo's builder calls Pokemon_CalcStats after MON_DATA_FORM (rom_diff.md, step 11):
    # Cyrus 913's Deoxys-Speed (form 3, row 498) has row 498's stats. The ability is set at
    # init from the base species (pokemon.c:473-485): Cyrus 780's Giratina-Origin (row 501,
    # Levitate) has Altered's Pressure, with row 501's stats and types.
    inter = IntermediateData(KAIZO_INTERMEDIATE)
    t = next(x for x in kaizo_trainers if x.id == 913)
    opp = build_trainer_party(t, inter)
    k = next(i for i, m in enumerate(opp) if m.species == 386)
    got = _slot(kaizo_tables, opp, k)
    hp, stats = _calc(kaizo_tables, opp[k], 498)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert (int(got["max_hp"]), list(got["stats"])) != _calc(kaizo_tables, opp[k], 386)

    t = next(x for x in kaizo_trainers if x.id == 780)
    opp = build_trainer_party(t, inter)
    k = next(i for i, m in enumerate(opp) if m.species == 487)
    got = _slot(kaizo_tables, opp, k)
    assert (int(got["form"]), int(got["ability_form"])) == (1, 0)
    assert int(got["ability"]) == AbilityId.PRESSURE
    assert np.asarray(kaizo_tables.species_abilities)[501, 0] == AbilityId.LEVITATE
    hp, stats = _calc(kaizo_tables, opp[k], 501)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert list(got["types"]) == list(np.asarray(kaizo_tables.species_types)[501])


def test_vanilla_data_form_keeps_the_base_stats_and_moves(tables):
    # Vanilla's builder never recomputes the stats after the form (src/trainer_data.c:216-218):
    # Devon's (491) Wormadam-Sandy (row 499) has Plant Cloak's stats and level-up default
    # moves, and Sandy's Bug/Ground types (pokemon.c:1219-1225).
    from nuzline.data.trainer import load_trainers

    devon = next(t for t in load_trainers(INTERMEDIATE) if t.id == 491)
    inter = _inter()
    opp = build_trainer_party(devon, inter)
    k = next(i for i, m in enumerate(opp) if m.form == 1)
    got = _slot(tables, opp, k)
    hp, stats = _calc(tables, opp[k], 413)
    assert (int(got["max_hp"]), list(got["stats"])) == (hp, stats)
    assert (int(got["max_hp"]), list(got["stats"])) != _calc(tables, opp[k], 499)
    assert list(got["types"]) == list(np.asarray(tables.species_types)[499])
    assert opp[k].moves == default_moves(inter.learnsets[413], opp[k].level)


# --- Family 3: trainer bag items by their party params (G-43) --------------------------------

SWITCH_TO_1 = 4 + 1


def _bag(s, names, item):
    return s._replace(
        trainer_items=s.trainer_items.at[1].set(jnp.asarray([names.item(item), 0, 0, 0])),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )


def test_bag_item_params_read_dummy_or_party_params(tables, kaizo_tables, names, kaizo_names):
    # Item_Get (item.c:186-191): a Poke Ball (partyUse FALSE) reads its dummy, 3, for every
    # party param in both games; a Potion its hpRestored 20. A ball's friendship amount can
    # pass 255, which is refused by name; a Full Restore's params are modelled. Dire Hit (crit
    # stages only) has no TrainerAI_ShouldUseItem branch, so it is never used and not refused.
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        params = np.asarray(t.item_party_param)
        ball = params[n.item("Poké Ball")]
        assert ball[: ItemParam.HOLD_FRIENDSHIP_UP].tolist() == [3] * ItemParam.HOLD_FRIENDSHIP_UP
        assert int(params[n.item("Potion")][ItemParam.HP_RESTORED]) == 20
        reasons = refused_bag_params(ball, n.item("Poké Ball"))
        assert any("friendship" in r for r in reasons)
        for item in ("Full Restore", "Dire Hit"):
            assert refused_bag_params(params[n.item(item)], n.item(item)) == []


def test_ai_item_choice_follows_the_param_branches(tables, names):
    # TrainerAI_ShouldUseItem (trainer_ai.c:4099-4189): only the first true branch is tested.
    s = roark_state(tables, names)
    top = int(s.max_hp[1, 0])
    assert top - 20 >= top // 4  # Geodude: the Potion's "whole amount fits" test decides

    def cat(state, item):
        return int(bag_item_category(tables, state, 1, names.item(item)))

    assert cat(s.replace_hp(1, 0, top - 21), "Potion") == ItemAiCategory.RECOVER_HP
    assert cat(s.replace_hp(1, 0, top - 20), "Potion") == ItemAiCategory.NONE
    # Full Heal's first param is the sleep cure: a poisoned battler gets no Full Heal.
    poisoned = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    asleep = s._replace(status=s.status.at[1, 0].set(Status.SLEEP))
    assert cat(poisoned, "Full Heal") == ItemAiCategory.NONE
    assert cat(asleep, "Full Heal") == ItemAiCategory.RECOVER_STATUS
    # Stat items only while fakeOutTurnNumber - totalTurns >= 0 (:4156).
    assert cat(s, "X Attack") == ItemAiCategory.STAT_BOOSTER
    assert cat(s._replace(turn=s.turn + 1), "X Attack") == ItemAiCategory.NONE


def test_bag_items_match_the_reference(tables, names):
    # One item turn per kind, the player switching (no damage): BattleSystem_UseBagItem and
    # its subscript (battle_system.c:435-758; battle_controller_player.c:1905-1944).
    s = roark_state(tables, names)
    top = int(s.max_hp[1, 0])
    o = empty_outcome()
    acts = [SWITCH_TO_1, FIRST_ITEM_ACTION]
    # Full Restore: status and confusion cured first, then all HP.
    hurt = s.replace_hp(1, 0, top // 5)._replace(
        status=s.status.at[1, 0].set(Status.POISON), confusion=s.confusion.at[1].set(3)
    )
    out = assert_matches_reference(tables, _bag(hurt, names, "Full Restore"), acts, o)
    assert (int(out.hp[1, 0]), int(out.status[1, 0]), int(out.confusion[1])) == (top, 0, 0)
    # X Accuracy: accuracy +1, and friendship 150 takes the medium band's +1.
    fond = s._replace(friendship=s.friendship.at[1, 0].set(150))
    out = assert_matches_reference(tables, _bag(fond, names, "X Accuracy"), acts, o)
    assert int(out.stages[1, Stat.ACC]) == 1 and int(out.friendship[1, 0]) == 151
    # Elixir (PP_RESTORE_ALL, 10): every slot below its max, capped.
    tired = s._replace(pp=s.pp.at[1, 0].set(jnp.asarray([0, 1, 2, 3])))
    out = assert_matches_reference(tables, _bag(tired, names, "Elixir"), acts, o)
    full = [int(tables.move_pp[m]) for m in np.asarray(s.moves[1, 0])]
    assert out.pp[1, 0].tolist() == [min(p + 10, f) for p, f in zip([0, 1, 2, 3], full)]
    # Persim Berry (confusion): the status subscript's bit-0 branch.
    dizzy = s._replace(confusion=s.confusion.at[1].set(2))
    out = assert_matches_reference(tables, _bag(dizzy, names, "Persim Berry"), acts, o)
    assert int(out.confusion[1]) == 0
