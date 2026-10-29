from nuzline.engine.actions import NUM_ACTIONS, legal_actions, legal_replacements
from tests.engine.helpers import roark_state


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
