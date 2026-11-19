"""E1 batch F (rulings F-n, docs/kaizo/e1-mechanics-ledger.md): Heatproof, Flash Fire, Multitype
and Intimidate's substitute skip, each case checked against the reference engine."""

from __future__ import annotations

import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table
from nuzline.engine.ids import AbilityId, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_protect import _state

O = empty_outcome()


def _ability(s, side, ability):
    """``ability`` on ``side``'s active slot."""
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _both(tables, s, move):
    """The 16 damage rolls of ``move`` from side 0, JAX and reference (asserted equal)."""
    got = damage_table(tables, s, 0, move, False).tolist()
    assert got == ref_damage_table(to_numpy(tables), to_ref(s), 0, move, False)
    return got


# --- Heatproof (F-1) -----------------------------------------------------------------------


def test_heatproof_is_covered():
    assert IMPLEMENTED_ABILITIES[AbilityId.HEATPROOF] == Support.EXACT


def test_heatproof_halves_fire_power_like_thick_fat_and_mold_breaker_ignores_it(tables, names):
    """battle_lib.c:6841-6844 halves a Fire move's power for a Heatproof defender
    (Battler_IgnorableAbility), the same cut as Thick Fat's Fire half (:6785-6788); a Water move
    is untouched; a Mold Breaker attacker ignores it."""
    s = roark_state(tables, names, player_moves=["Ember", "Water Gun"])
    ember, water = names.move("Ember"), names.move("Water Gun")
    bare = _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)
    heat = _ability(bare, 1, AbilityId.HEATPROOF)
    thick = _ability(bare, 1, AbilityId.THICK_FAT)
    assert _both(tables, heat, ember) == _both(tables, thick, ember)
    assert max(_both(tables, heat, ember)) < max(_both(tables, bare, ember))
    assert _both(tables, heat, water) == _both(tables, bare, water)
    breaker = _ability(heat, 0, AbilityId.MOLD_BREAKER)
    assert _both(tables, breaker, ember) == _both(
        tables, _ability(bare, 0, AbilityId.MOLD_BREAKER), ember
    )


@pytest.mark.parametrize(
    ("max_hp", "loss"),
    [(40, 2), (9, 1)],  # Divide(40, 8) = 5, then Divide(5, 2) = 2; Divide(9, 8) = 1, then 1 (floor)
)
def test_heatproof_halves_burn_damage(tables, names, max_hp, loss):
    """subscript_burn_damage.s:6-9: Divide(maxHP, 8), then Divide(loss, 2) for Heatproof
    (BattleSystem_Divide floors at 1, battle_lib.c:3599-3618); other holders lose the eighth."""
    s = _state(tables, names, mine=["Protect"])
    s = s._replace(hp=s.hp.at[1, 0].set(max_hp), max_hp=s.max_hp.at[1, 0].set(max_hp))
    s = s.with_status(1, 0, Status.BURN)
    acts = A("Protect", "Tackle", s, names)
    plain = assert_matches_reference(tables, _ability(s, 1, AbilityId.NONE), acts, O)
    assert int(plain.hp[1, 0]) == max_hp - max(max_hp // 8, 1)
    heat = assert_matches_reference(tables, _ability(s, 1, AbilityId.HEATPROOF), acts, O)
    assert int(heat.hp[1, 0]) == max_hp - loss
