"""D45-4a: an Encored or recharging AI battler's FIGHT command never opens the move menu
(battle_controller_player.c:442-454; ``Battler_CanPickCommand``, battle_lib.c:6224-6236), so
``TrainerAI_RecordLastMove`` (trainer_ai.c:2705-2717) records nothing. JAX
``_record_last_move`` against the reference ``_record_last_moves`` on the same input (direct
calls: no ``step`` compile, fast tier)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import format_of
from nuzline.engine.step import _record_last_move
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.helpers import roark_state


def _record(tables, s, acts):
    """Both engines' record on ``s`` with ``acts``; asserted equal; returns the JAX rows."""
    out = _record_last_move(tables, s, jnp.asarray(acts, jnp.int32))
    r = to_ref(s)
    t = to_numpy(tables)
    choices = [ref._choose(t, r, b, acts[b]) for b in range(format_of(s).B)]
    ref._record_last_moves(t, r, tuple(acts), choices)
    assert np.asarray(out.ai_seen_moves).tolist() == r.ai_seen_moves
    return np.asarray(out.ai_seen_moves)


def _singles(tables, names):
    """Roark's fixture with the player's last move Tackle and nothing recorded yet."""
    s = roark_state(tables, names)
    return s._replace(last_move=s.last_move.at[0].set(names.move("Tackle")))


def test_encored_enemy_records_nothing(tables, names):
    """D45-4a, bcp.c:442-454: battler 1 is Encored into a move it knows; the same state without
    the Encore records Tackle in row 0."""
    s = _singles(tables, names)
    tackle = names.move("Tackle")
    assert _record(tables, s, [0, 0])[0].tolist() == [tackle, 0, 0, 0]
    encored = s._replace(
        encore_move=s.encore_move.at[1].set(tackle),
        encore_slot=s.encore_slot.at[1].set(0),
        encore_turns=s.encore_turns.at[1].set(3),
    )
    assert _record(tables, encored, [0, 0])[0].tolist() == [0, 0, 0, 0]


def test_recharging_enemy_records_nothing(tables, names):
    """D45-4a, battle_lib.c:6228: a recharging enemy picks no command."""
    s = _singles(tables, names)
    s = s._replace(recharge=s.recharge.at[1].set(1))
    assert _record(tables, s, [0, 0])[0].tolist() == [0, 0, 0, 0]
