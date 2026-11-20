"""D3-15 x F-O3 (batch G): a Starf Berry eaten at another battler's action is a read of the turn's
draw (``enumerate._starf_reads``, index 0).

The faster foe's Swords Dance is self-targeted: AfterMoveEffects' ATTACKER_ITEM and DEFENDER_ITEM
check only the foe (bcp.c:3684-3696), and HELD_ITEM_STATUS walks every battler in monSpeedOrder
(bcp.c:3730-3756), as MoveEnd's TriggerEffectOnSwitch pass does (bcp.c:3973, battle_lib.c:4151-4165):
the pinched holder (curHP <= maxHP / 4, battle_lib.c:4850-4855) eats its Starf Berry there, with the
queue's entry 0 (battle_lib.c:5286-5288), before its own move. Its own ATTACKER_ITEM then finds no
berry. Every run is checked against the reference engine (``_consumed_starf``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.enumerate import outcome_laws
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_e1_batch_g import _acts, _boosts, _consumed_starf, _draws, _starf_fight


@pytest.mark.slow
def test_a_starf_eaten_at_the_foes_action_is_read_by_the_law(tables, names):
    s = _starf_fight(names, tables, ["Swords Dance"], ["Swords Dance"])
    acts = jnp.asarray(_acts(names, s, 0, "Swords Dance", "Swords Dance"))
    out = assert_matches_reference(tables, s, acts, _draws(3))
    # Rank 3 of the five stats below +6 (Sp. Atk, lib.c:5288), then the Abra's own Swords Dance.
    assert int(out.item[0, 0]) == 0 and _boosts(out) == {"ATK": 2, "SPA": 2}
    laws = jax.tree.map(np.asarray, outcome_laws(tables, s, acts))
    relevant = laws.relevant.starf[0].tolist()
    consumed = _consumed_starf(tables, s, acts)
    assert consumed == [True, False, False, False]
    assert all(r for r, c in zip(relevant, consumed, strict=True) if c)
