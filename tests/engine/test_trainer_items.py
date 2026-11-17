"""Trainer item use: opponent-only actions 10-13 use bag slot ``trainer_items[1, k]``.

Decomp: ``TrainerAI_ShouldUseItem`` (trainer_ai.c:4056) picks the item and empties its
slot; ``BattleControllerPlayer_CalcTurnOrder`` (battle_controller_player.c:707) runs items
and switches before moves, in battler order; ``subscript_use_potion.s`` heals through
``BattleSystem_UseBagItem`` (battle_system.c:435) and ``BATTLE_SUBSCRIPT_UPDATE_HP``.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    NUM_ACTIONS,
    NUM_OPP_ACTIONS,
    legal_actions,
    legal_opponent_actions,
)
from nuzline.engine.ids import ItemParam
from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.helpers import roark_state

DECOMP = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"
POTION = 17  # decomp ITEM_POTION
USE_SLOT_0 = FIRST_ITEM_ACTION  # opponent action 10
SWITCH_TO_1 = 4 + 1


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _ref(tables, s, acts, o):
    return from_ref(ref_step(to_numpy(tables), to_ref(s), tuple(acts), outcome_to_py(o)))


def test_potion_params_are_the_decomps(tables, names):
    # E1 ruling G-43: res/items/data/potion.json's itemUseParams, packed by itemproc.c.
    potion = DECOMP / "res" / "items" / "data" / "potion.json"
    if not potion.exists():
        pytest.skip("decomp not cloned (run tools/clone_decomp.sh)")
    data = json.loads(potion.read_text(encoding="utf-8"))
    p = tables.item_party_param[names.item("Potion")].tolist()
    assert data["itemUseParams"]["hpRestored"] == p[ItemParam.HP_RESTORED] == 20
    assert p[ItemParam.HP_RESTORE] == 1 and sum(p) == 21


def test_make_state_loads_roarks_bag(tables, names):
    s = roark_state(tables, names)
    assert s.trainer_items.tolist() == [[0, 0, 0, 0], [POTION, POTION, 0, 0]]


@pytest.mark.slow
def test_potion_heals_20_and_is_consumed(tables, names):
    s = roark_state(tables, names).replace_hp(1, 0, 5)  # Geodude at 5 HP
    out = step(tables, s, jnp.array([SWITCH_TO_1, USE_SLOT_0]), empty_outcome())
    assert int(out.hp[1, 0]) == min(5 + 20, int(s.max_hp[1, 0]))
    assert out.trainer_items[1].tolist() == [0, POTION, 0, 0]


def test_potion_is_capped_at_max_hp(tables, names):
    s = roark_state(tables, names)
    max_hp = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, max_hp - 3)
    out = step(tables, s, jnp.array([SWITCH_TO_1, USE_SLOT_0]), empty_outcome())
    assert int(out.hp[1, 0]) == max_hp and out.trainer_items[1].tolist() == [0, POTION, 0, 0]


@pytest.mark.slow
def test_item_goes_before_a_faster_move(tables, names):
    # Monferno (Spe 30) outspeeds Geodude (Spe 10), yet the Potion lands first: Geodude 3
    # HP short heals to full and then takes the Scratch. Healed after the Scratch, it
    # would end at full HP.
    base = roark_state(tables, names)
    max_hp = int(base.max_hp[1, 0])
    scratch = names.move_slot(base, 0, "Scratch")
    tackle = names.move_slot(base, 1, "Tackle")
    o = empty_outcome()
    healed = step(tables, base.replace_hp(1, 0, max_hp - 3), jnp.array([scratch, USE_SLOT_0]), o)
    # The same Scratch on a full-HP Geodude that does not heal.
    plain = step(tables, base, jnp.array([scratch, tackle]), o)
    assert 0 < int(plain.hp[1, 0]) < max_hp
    assert int(healed.hp[1, 0]) == int(plain.hp[1, 0])
    assert action_order(tables, base, jnp.array([scratch, USE_SLOT_0]), o).tolist() == [1, 0]


def test_player_switch_goes_before_an_item_in_battler_order(tables, names):
    # Items and switches are different commands: no speed comparison, battler order.
    s = roark_state(tables, names, opp_faster=True)
    order = action_order(tables, s, jnp.array([SWITCH_TO_1, USE_SLOT_0]), empty_outcome())
    assert order.tolist() == [0, 1]


def test_item_in_a_pursuit_slot_does_not_pursue(tables, names):
    # Action 10 must not read as move slot 3 (here Pursuit) when the player switches.
    s = roark_state(
        tables, names, opp_moves=["Tackle", "Rock Throw", "Stealth Rock", "Pursuit"]
    ).replace_hp(1, 0, 5)
    out = step(tables, s, jnp.array([SWITCH_TO_1, USE_SLOT_0]), empty_outcome())
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert out.pp[1, 0].tolist() == s.pp[1, 0].tolist()


def test_item_slot_illegal_when_empty(tables, names):
    s = roark_state(tables, names)
    bag = [POTION, 0, names.item("Super Potion"), POTION]
    s = s._replace(trainer_items=s.trainer_items.at[1].set(jnp.asarray(bag, jnp.int16)))
    assert legal_opponent_actions(tables, s).tolist()[NUM_ACTIONS:] == [True, False, True, True]
    fainted = s.replace_hp(1, 0, 0)
    assert legal_opponent_actions(tables, fainted).tolist()[NUM_ACTIONS:] == [False] * 4


def test_player_mask_is_still_10_wide(tables, names):
    s = roark_state(tables, names)
    assert legal_actions(tables, s, 0).shape == (NUM_ACTIONS,) == (10,)
    opp = legal_opponent_actions(tables, s)
    assert opp.shape == (NUM_OPP_ACTIONS,) == (14,)
    assert opp[:NUM_ACTIONS].tolist() == legal_actions(tables, s, 1).tolist()


@pytest.mark.slow
@pytest.mark.parametrize("player", ["Scratch", SWITCH_TO_1])
def test_reference_matches_on_item_use(tables, names, player):
    s = roark_state(tables, names).replace_hp(1, 0, 5)
    p = names.move_slot(s, 0, player) if isinstance(player, str) else player
    acts = [p, USE_SLOT_0 + 1]  # slot 1: the reference must use the slot it is given
    o = empty_outcome()
    out = step(tables, s, jnp.array(acts), o)
    _same(out, _ref(tables, s, acts, o))
    assert out.trainer_items[1].tolist() == [POTION, 0, 0, 0]
