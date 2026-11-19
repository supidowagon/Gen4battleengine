import jax.numpy as jnp
import pytest

from nuzline.engine.actions import NUM_ACTIONS, legal_actions, legal_replacements
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import A, jit_step, roark_state


def test_full_party_lead(tables, names):
    s = roark_state(tables, names)
    m = legal_actions(tables, s, 0).tolist()
    assert len(m) == NUM_ACTIONS and m[:4] == [True] * 4 and m[4] is False and all(m[5:10])


def test_fainted_and_empty_slots_illegal(tables, names):
    s = roark_state(tables, names, opp_count=2).replace_hp(1, 1, 0)
    m = legal_actions(tables, s, 1).tolist()
    assert m[4:10] == [False] * 6
    assert legal_replacements(s, 1).tolist() == [False] * 6


def test_zero_pp_and_struggle(tables, names):
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    m = legal_actions(tables, s, 0).tolist()
    assert m[0] is True and m[1:4] == [False] * 3


def test_trapped_cannot_switch_and_taunt_blocks_status(tables, names):
    s = roark_state(tables, names)
    s = s._replace(trapped=s.trapped.at[0].set(2), taunt=s.taunt.at[0].set(2))
    m = legal_actions(tables, s, 0).tolist()
    assert not any(m[4:10])
    leer = names.move_slot(s, 0, "Leer")
    assert m[leer] is False


@pytest.mark.slow
def test_taunted_with_only_status_moves_struggles(tables, names):
    # CHECK_INVALID_ALL includes CHECK_INVALID_TAUNTED (battle_lib.c:2282): with every
    # PP-holding move at power 0 the decomp sets STRUGGLING_ALL
    # (battle_controller_player.c:433), so action 0 means Struggle even though slot 0
    # (Leer) still has PP.
    s = roark_state(tables, names, player_moves=["Leer"]).with_pp(0, 0, [10, 0, 0, 0])
    s = s._replace(taunt=s.taunt.at[0].set(2))
    assert legal_actions(tables, s, 0).tolist()[:4] == [True, False, False, False]
    out = jit_step(tables, s, A(0, "Leer", s, names), empty_outcome())
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # Struggle hit
    max_hp = int(s.max_hp[0, 0])
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) - max(max_hp // 4, 1)  # Struggle recoil
    assert int(out.stages[1, 2]) == 0  # not Leer
    assert out.pp[0, 0].tolist() == [10, 0, 0, 0]  # Struggle spends no PP


def test_taunt_after_choosing_a_status_move_fails_it_without_struggling(tables, names):
    # Struggling is decided when the action is chosen (battle_controller_player.c:433):
    # a Pokemon taunted mid-turn after choosing Leer has Leer fail (:2557), it does
    # not switch to Struggle.
    s = roark_state(tables, names, opp_moves=["Taunt"], opp_faster=True)
    s = s.with_pp(0, 0, [0, 10, 0, 0])
    o = empty_outcome()._replace(duration=jnp.array([0, 3], jnp.int8))
    out = jit_step(tables, s, A("Leer", "Taunt", s, names), o)
    assert int(out.taunt[0]) > 0
    assert int(out.stages[1, 2]) == 0  # Leer failed
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])  # no Struggle hit
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])  # and no recoil
    assert out.pp[0, 0].tolist() == [0, 10, 0, 0]
