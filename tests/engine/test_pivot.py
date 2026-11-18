"""E2 batch 6: the mid-turn switch decision (U-turn, Baton Pass; rulings B6-n,
docs/kaizo/e2-mechanics-ledger.md). Hand cases the random differential draws rarely reach, each
half of the turn checked against the reference engine (``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS, decision_actions
from nuzline.engine.ids import Stat
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import TURN_SCRATCH, make_state
from tests.engine.helpers import assert_matches_reference


def _mon(names, species: str, moves, level=30, item="") -> PokemonSpec:
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


O = empty_outcome()


def _resume(tables, paused, slot, outcome=O):
    side = int(paused.pivot_battler) - 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[side] = MOVE_ACTIONS + slot
    return assert_matches_reference(tables, paused, tuple(acts), outcome)


def test_player_u_turn_pauses_then_the_foe_meets_the_newcomer(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 35)],
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    assert int(paused.active[0]) == 0
    assert int(paused.hp[1, 0]) < int(s.hp[1, 0])  # the U-turn hit
    assert int(paused.turn) == int(s.turn)  # the turn is not over
    # Only the player's replacements are offered.
    legal = np.flatnonzero(np.asarray(decision_actions(tables, paused))).tolist()
    assert legal == [MOVE_ACTIONS + 1]
    after = _resume(tables, paused, 1)
    assert int(after.pivot_battler) == 0
    assert int(after.active[0]) == 1
    assert int(after.hp[0, 1]) < int(s.hp[0, 1])  # the foe's Tackle hit the newcomer
    assert int(after.last_move[0]) == n.move("U-turn")  # written onto the newcomer
    assert int(after.turn) == int(s.turn) + 1
    for name in TURN_SCRATCH:
        assert not np.any(np.asarray(getattr(after, name))), name


def test_opponent_u_turn_and_a_faster_baton_pass(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35)],
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 2
    after = _resume(tables, paused, 1)
    assert int(after.active[1]) == 1 and int(after.pivot_battler) == 0
    # Baton Pass keeps the stages and confusion (BattleSystem_UpdateAfterSwitch, lib.c:2078).
    s = make_state(
        tables,
        [_mon(n, "Ninjask", ["Baton Pass"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 20)],
    ).with_stages(0, atk=2, spe=1)
    s = s._replace(confusion=s.confusion.at[0].set(3))
    paused = assert_matches_reference(
        tables, s, (0, 0), O._replace(confusion_hit=jnp.zeros(2, jnp.int8))
    )
    assert int(paused.pivot_battler) == 1
    after = _resume(tables, paused, 1)
    assert int(after.stages[0, Stat.ATK]) == 2 and int(after.stages[0, Stat.SPE]) == 1
    assert int(after.confusion[0]) == 2


def test_u_turn_meets_pursuit_and_items_around_the_pivot(tables, names):
    n = names
    # The slower foe's Pursuit hits the U-turn user before the list (subscript :54); its
    # action is over and powerMul 20 stands at the list.
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 25, "Life Orb"), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 35)],
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1
    assert int(paused.ts_pick_mul) == 20 and int(paused.ts_pursued) == 2
    # Life Orb recoil at the pivot item (:11) and the Pursuit hit, before the list.
    assert int(paused.hp[0, 0]) < int(s.hp[0, 0])
    after = _resume(tables, paused, 1)
    assert int(after.hp[0, 1]) == int(s.hp[0, 1])  # the pursuer does not act again
    # No replacement: a plain hit, no pause.
    s = make_state(
        tables, [_mon(n, "Scyther", ["U-turn"], 40)], [_mon(n, "Bulbasaur", ["Tackle"], 20)]
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.pivot_battler) == 0 and int(out.turn) == int(s.turn) + 1


def test_the_newcomers_kings_rock_flinches(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30, "King’s Rock")],
        [_mon(n, "Bulbasaur", ["Tackle"], 40)],
    )
    o = O._replace(flinch_item=jnp.ones((2, MAX_HITS), jnp.int8))
    paused = assert_matches_reference(tables, s, (0, 0), o)
    after = _resume(tables, paused, 1, o)
    assert int(after.hp[0, 1]) == int(s.hp[0, 1])  # the foe flinched
