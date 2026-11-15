"""E2 batch 5 review follow-ups (rulings B5-13.., docs/kaizo/e2-mechanics-ledger.md).

* B5-14: the Future Sight hit's Zoom Lens reads ``Battler_MovedThisTurn`` (battle_lib.c:
  5365-5368: ``battlerActions[d][PICK_COMMAND] == BATTLE_CONTROL_MOVE_END``) at the turn's end,
  when every battler slot has it: each action's MoveEnd sets it for the actor (bcp.c:3989), a
  switch-in by choice included, and a battler dragged in by Roar / Whirlwind gets it from
  UpdateAfterSwitch (battle_lib.c:2053). So the boost holds for a defender that switched in or
  was dragged in this turn, in both engines.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference


def _mon(names, species: str, moves, level: int, item: str = "") -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=names.item(item) if item else 0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


@pytest.mark.parametrize("how", ["switched", "dragged"])
@pytest.mark.parametrize("zoom", [True, False])
def test_future_sight_zoom_lens_on_a_newcomer(tables, names, how, zoom):
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Slowbro", ["Whirlwind", "Tackle"], 50, "Zoom Lens" if zoom else "")],
        [_mon(n, "Geodude", ["Tackle"], 20), _mon(n, "Onix", ["Tackle"], 20)],
    )
    move = n.move("Future Sight")
    s = s._replace(
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(10),
        future_move=s.future_move.at[1].set(move),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
    )
    base = int(t.move_accuracy[move])
    # A roll above the plain accuracy and within Zoom Lens's x1.2 (bcp.c:2963-2965).
    o = empty_outcome()._replace(future_hit=jnp.asarray([1, base + 1], jnp.int8))
    actions = (1, MOVE_ACTIONS + 1) if how == "switched" else (0, 0)
    out = assert_matches_reference(t, s, actions, o)
    assert int(out.active[1]) == 1
    # The same turn with the hit missed (class 0): the hit's 10 HP is the difference.
    missed = assert_matches_reference(
        t, s, actions, o._replace(future_hit=jnp.asarray([1, 0], jnp.int8))
    )
    assert int(out.hp[1, 1]) == int(missed.hp[1, 1]) - (10 if zoom else 0)
