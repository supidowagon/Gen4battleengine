"""The Starf Berry count bound (``enumerate._starf_gain``): every raise before the eat counts.

The board (singles): side 1's Blissey holds a Starf Berry, has Speed Boost, is poisoned, sits at
Speed +3 (its other stats at 0) and one HP above its pinch plus a poison tick. Its trainer uses an
X Speed on it (the item action comes first, battle_controller_player.c:753-761; +1,
battle_system.c:603-608); side 0 Flings a Salac Berry at it (FLING_EFFECT_SPEED_UP,
battle_lib.c:6089-6094: +1; the hit leaves it above its pinch); at the turn end Speed Boost
(bcp.c:1354, battle_lib.c:3555-3562: +1) takes Speed to +6 before USE_ITEM (bcp.c:1362, still
above its pinch), the poison tick (bcp.c:1393) lands it at its pinch, and ITEM_CONDITION
(bcp.c:1699, battle_lib.c:5273-5292) eats the berry with four stats below +6: the draw is read
modulo 4, not 5. The old law (highest 3 + slack 2 = 5) gave the five-value law."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.enumerate import enumerate_outcomes, outcome_laws
from nuzline.engine.ids import STARF_DRAWS, AbilityId, Stat, Status
from nuzline.engine.outcome import Outcome, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_enumerate import rows, step_bytes
from tests.engine.test_pivot import _mon


def _board(tables, names):
    s = make_state(
        tables,
        [_mon(names, "Bidoof", ["Fling"], 5, "Salac Berry")],
        [_mon(names, "Blissey", ["Tackle"], 100, "Starf Berry")],
    )
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4 + top // 8 + 1)
    s = s._replace(
        ability=s.ability.at[1, 0].set(AbilityId.SPEED_BOOST),
        status=s.status.at[1, 0].set(Status.POISON),
        stages=s.stages.at[1, Stat.SPE].set(3),
        # Not its first turn out: Speed Boost acts (fakeOutTurnNumber != totalTurns + 1).
        fake_out_turn=s.fake_out_turn.at[1].set(s.turn),
        trainer_items=s.trainer_items.at[1, 0].set(names.item("X Speed")),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )
    acts = jnp.asarray([names.move_slot(s, 0, "Fling"), FIRST_ITEM_ACTION], jnp.int32)
    return s, acts


def test_the_fling_x_speed_speed_boost_board_eats_at_four_stats(tables, names):
    """Both engines: the berry is eaten at Speed +6, so draw 4 is rank 0 of the four (Attack) and
    draw 3 rank 3 (Sp. Def), where a count of 5 would give Sp. Def and Sp. Atk."""
    s, acts = _board(tables, names)
    top = int(s.max_hp[1, 0])
    for draw, stat in ((4, Stat.ATK), (3, Stat.SPD)):
        o = empty_outcome()
        o = o._replace(starf=o.starf.at[1, 0].set(draw))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.item[1, 0]) == 0 and int(out.hp[1, 0]) <= top // 4
        stages = np.asarray(out.stages[1])
        assert int(stages[Stat.SPE]) == 6 and int(stages[stat]) == 2


def test_the_law_is_loose_and_the_rows_match_the_full_product(tables, names):
    """The law reads all ``STARF_DRAWS`` values of side 1's first eat, and the mask is sound on
    the board: for one enumerated group (every draw but ``starf[1, 0]`` fixed), the children of
    the law's rows equal those of the full product, that cell at each of the 60 values."""
    s, acts = _board(tables, names)
    laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])
    outcomes, probs = enumerate_outcomes(laws)
    starf = np.asarray(outcomes.starf).copy()
    starf[:, 1, 0] = 0
    base = outcomes._replace(starf=starf)
    keys = [b"".join(np.asarray(f)[k].tobytes() for f in base) for k in range(len(probs))]
    group = np.flatnonzero([key == keys[0] for key in keys])
    total = float(np.sum(probs[group]))
    got: dict[bytes, float] = {}
    for key, p in zip(
        step_bytes(tables, s, acts, rows(outcomes, group)), probs[group], strict=True
    ):
        got[key] = got.get(key, 0.0) + float(p) / total
    full = rows(base, np.full(STARF_DRAWS, group[0]))
    cells = np.asarray(full.starf).copy()
    cells[:, 1, 0] = np.arange(STARF_DRAWS)
    want: dict[bytes, float] = {}
    for key in step_bytes(tables, s, acts, Outcome(*full)._replace(starf=cells)):
        want[key] = want.get(key, 0.0) + 1 / STARF_DRAWS
    assert set(got) == set(want)
    for key, p in want.items():
        assert abs(got[key] - p) < 1e-12
