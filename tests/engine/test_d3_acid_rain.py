"""D3-9 x FCM step 4: Acid Rain at B = 4 (subscript_pursuit.s:100, :115; battle_script.c:3282-3287,
ledger G-12). Each Pursuit pass that KOs the switcher subtracts ``faintedMon`` from
``fieldConditionsMask`` after its payout (``field.acid_rain``, ``reference._acid_rain``); the
pursuers are taken in battler index order (battle_script.c:6899, :6934). Ported from the B = 4
cases of the removed Acid Rain guard test: the operand is now the mask itself.

``four`` states: level-50 Bulbasaurs; ``play`` runs the action phase on both engines and asserts
every field equal. 0x80 is HAILING_PERM (condition.h:117), the turn end not played."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId, FieldCondition
from tests.engine.d2_states import four, with_ability, with_moves
from tests.engine.test_d3_action_loop import play

SW4 = MOVE_ACTIONS + 2
HAIL = int(FieldCondition.HAILING_PERM)


def _masked(tables, names, chasers, mask=HAIL):
    """DOUBLES with ``fieldConditionsMask`` ``mask``: battlers in ``chasers`` know Pursuit, the
    rest Swords Dance."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Pursuit" if b in chasers else "Swords Dance"])
    return s._replace(field_mask=jnp.asarray(mask, s.field_mask.dtype))


def _hurt(s, party, slot, hp):
    return s._replace(hp=s.hp.at[party, slot].set(hp))


def _mask(tables, s, acts, order):
    return int(play(tables, s, acts, order)[0].field_mask)


def test_four_ko_by_battler_0_of_the_switching_battler_3(tables, names):
    """faintedMon 3: 0x80 - 3; with no condition up the subtraction clamps at 0."""
    s = _hurt(_masked(tables, names, (0,)), 1, 1, 1)  # battler 3: party 1 slot 1
    assert _mask(tables, s, [0, 0, 0, SW4], (3, 0, 1, 2)) == HAIL - 3
    calm = s._replace(field_mask=jnp.zeros_like(s.field_mask))
    assert _mask(tables, calm, [0, 0, 0, SW4], (3, 0, 1, 2)) == 0


def test_four_spurious_uproar_bit_alone_battler_1_ko_by_battler_2(tables, names):
    """Battler 0's Uproar bit (0x100) alone; battler 2's Pursuit KOs the switching battler 1:
    0x100 - 1 = 0xFF; battler 1's own bit (0x200) is clear (UpdateAfterSwitch,
    battle_lib.c:2135)."""
    s = _hurt(_masked(tables, names, (2,), 0x100), 1, 0, 1)
    assert _mask(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3)) == 0xFF


def test_four_ko_of_battler_0_leaves_the_mask(tables, names):
    """faintedMon 0 (battler 0, the switcher): the mask unchanged."""
    s = _hurt(_masked(tables, names, (1,)), 0, 0, 1)
    out = play(tables, s, [SW4, 0, 0, 0], (0, 1, 2, 3))[0]
    assert int(out.hp[0, 0]) == 0 and int(out.field_mask) == HAIL


def test_four_rough_skin_fells_battler_3(tables, names):
    """Battler 0 (Rough Skin, 1 HP) switches; battler 3 (1 HP) KOs it and falls to Rough Skin
    (subscript_pursuit.s:86-87): UPDATE_HP's TryFaintMon makes it faintedMon (FCM 1.3)."""
    s = with_ability(_masked(tables, names, (3,)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(_hurt(s, 0, 0, 1), 1, 1, 1)
    assert _mask(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2)) == HAIL - 3


def test_four_life_orb_recoil_fells_battler_2(tables, names):
    """Battler 2 (Life Orb, 1 HP) KOs the switching battler 1 and falls to the recoil
    (subscript_pursuit.s:94-95, battle_lib.c:7357-7365): faintedMon 2."""
    s = _masked(tables, names, (2,))
    s = s._replace(item=s.item.at[0, 1].set(names.item("Life Orb")))  # battler 2: party 0 slot 1
    s = _hurt(_hurt(s, 1, 0, 1), 0, 1, 1)
    out = play(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3))[0]
    assert int(out.hp[0, 1]) == 0 and int(out.field_mask) == HAIL - 2


def test_four_aftermath_fells_battler_1(tables, names):
    """Battler 0 (Aftermath, 1 HP) switches; battler 1 (1 HP) KOs it and falls to Aftermath
    (battle_lib.c:4358-4371): faintedMon 1."""
    s = with_ability(_masked(tables, names, (1,)), 0, AbilityId.AFTERMATH)
    s = _hurt(_hurt(s, 0, 0, 1), 1, 0, 1)
    out = play(tables, s, [SW4, 0, 0, 0], (0, 1, 2, 3))[0]
    assert int(out.hp[1, 0]) == 0 and int(out.field_mask) == HAIL - 1


def test_four_destiny_bond_keeps_the_switcher_as_operand(tables, names):
    """Battler 0 (Destiny Bond, Rough Skin, 1 HP) switches; battler 3's Pursuit KOs it. Destiny
    Bond (subscript_pursuit.s:85) takes battler 3 down before Rough Skin and restores faintedMon
    to the switcher (subscript_faint_check_destiny_bond.s:25): battler 0, the mask unchanged."""
    s = with_ability(_masked(tables, names, (3,)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(s, 0, 0, 1)
    s = s._replace(destiny_bond=s.destiny_bond.at[0].set(1))
    out = play(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2))[0]
    assert int(out.hp[1, 1]) == 0 and int(out.field_mask) == HAIL


def test_four_destiny_bond_clears_the_pursuers_uproar_bit(tables, names):
    """Battler 2's Uproar bit (0x400) alone; battler 1 (Destiny Bond) is KO'd by battler 2: the
    bond's faint clears bit 10 (battle_lib.c:2213) before :115, 0 - 1 clamps to 0."""
    s = _masked(tables, names, (2,), 0x400)
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(1))
    s = _hurt(s, 1, 0, 1)
    assert _mask(tables, s, [0, SW4, 0, 0], (1, 2, 0, 3)) == 0


def test_four_the_lower_battler_hits_first_whatever_the_speeds(tables, names):
    """Battler 0 (Rough Skin, 1 HP) switches, battlers 1 (1 HP) and 3 both ready, battler 3
    faster: battler 1 hits first (index order, battle_script.c:6899) and falls to Rough Skin;
    battler 3 never pursues (the switcher is down): faintedMon 1."""
    s = with_ability(_masked(tables, names, (1, 3)), 0, AbilityId.ROUGH_SKIN)
    s = _hurt(_hurt(s, 0, 0, 1), 1, 0, 1)
    assert _mask(tables, s, [SW4, 0, 0, 0], (0, 3, 1, 2)) == HAIL - 1
