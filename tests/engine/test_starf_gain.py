"""The Starf Berry count bound (``enumerate._starf_gain``): every raise before the eat counts.

The board (singles): side 1's Blissey holds a Starf Berry, has Speed Boost, is poisoned, sits at
Speed +3 (its other stats at 0) and one HP above its pinch plus a poison tick. Its trainer uses an
X Speed on it (the item action comes first, battle_controller_player.c:753-761; +1,
battle_system.c:603-608); side 0 Flings a Salac Berry at it (FLING_EFFECT_SPEED_UP,
battle_lib.c:6089-6094: +1; the hit leaves it above its pinch); at the turn end Speed Boost
(bcp.c:1354, battle_lib.c:3555-3562: +1) takes Speed to +6 before USE_ITEM (bcp.c:1362, still
above its pinch), the poison tick (bcp.c:1393) lands it at its pinch, and ITEM_CONDITION
(bcp.c:1699, battle_lib.c:5273-5292) eats the berry with four stats below +6: the draw is read
modulo 4, not 5. The old law (highest 3 + slack 2 = 5) gave the five-value law.

The Trace boards (singles): side 1's Blissey has Trace and has not traced (``trace_announced``
0: side 0's Castform has Forecast, which ChooseTraceTarget skips, battle_lib.c:7750-7781), Sp.
Atk +5, poisoned, one poison tick above its pinch; side 0's bench is one Download holder with
Defense above Sp. Def. When it enters, SWITCH_IN_CHECK_STATE_TRACE copies it for the armed
Blissey (battle_lib.c:3711-3731) and SWITCH_IN_CHECK_STATE_DOWNLOAD (:3814-3851) raises its Sp.
Atk to +6; the poison tick and ITEM_CONDITION then eat the berry with four stats below +6. On the
switch board side 0 switches to it (``outcome.pivot_view`` already shows the copy, G-25); on the
Roar board Castform uses Growl and Blissey's Roar drags the Download holder in (the drag's pick is
random, so only ``_starf_gain``'s armed-Trace term sees it: 47483525 gave the five-value law).

The law tests only jit ``outcome_laws`` (fast); the Roar board's step check and the product
checks (they enumerate and step every row) are slow. The control board is the Fling board
without the X Speed: highest 3 + Fling 1 + Speed Boost 1 = 5, tight, and the true count at the
eat is 5."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import FIRST_ITEM_ACTION, MOVE_ACTIONS
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


def _control(tables, names):
    """The Fling board with side 1 using Tackle instead of the X Speed: Speed reaches +5 only."""
    s, _ = _board(tables, names)
    acts = jnp.asarray([names.move_slot(s, 0, "Fling"), names.move_slot(s, 1, "Tackle")], jnp.int32)
    return s, acts


def _trace_board(tables, names, roar):
    """The Trace board: side 0 switches to its Download holder, or (``roar``) Blissey drags it."""
    s = make_state(
        tables,
        [_mon(names, "Castform", ["Growl"], 50), _mon(names, "Geodude", ["Tackle"], 50)],
        [_mon(names, "Blissey", ["Roar"], 100, "Starf Berry")],
    )
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4 + top // 8)
    s = s._replace(
        ability=s.ability.at[0, 0]
        .set(AbilityId.FORECAST)
        .at[0, 1]
        .set(AbilityId.DOWNLOAD)
        .at[1, 0]
        .set(AbilityId.TRACE),
        trace_announced=s.trace_announced.at[1].set(0),
        status=s.status.at[1, 0].set(Status.POISON),
        stages=s.stages.at[1, Stat.SPA].set(5),
    )
    assert int(s.stats[0, 1, Stat.DEF - 1]) >= int(s.stats[0, 1, Stat.SPD - 1])
    first = names.move_slot(s, 0, "Growl") if roar else MOVE_ACTIONS + 1
    acts = jnp.asarray([first, names.move_slot(s, 1, "Roar")], jnp.int32)
    return s, acts


def _laws(tables, s, acts):
    return jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))


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


def test_the_fling_board_law_reads_all_draws(tables, names):
    """The law alone (fast): side 1's first eat reads all ``STARF_DRAWS`` values."""
    laws = _laws(tables, *_board(tables, names))
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])


@pytest.mark.slow  # the switch-board twin took about 60 s, peak 10.8 GB locally: over the limits
def test_the_roar_trace_board_eats_at_four_stats(tables, names):
    """Both engines: the armed Trace copies the dragged newcomer's Download, Sp. Atk goes to +6,
    and the berry is eaten with four stats below +6 (draw 3: Sp. Def, where five would give Sp.
    Atk; Growl's Attack -1 makes draw 4's Attack +1)."""
    s, acts = _trace_board(tables, names, roar=True)
    top = int(s.max_hp[1, 0])
    for draw, stat, stage in ((4, Stat.ATK, 1), (3, Stat.SPD, 2)):
        o = empty_outcome()
        o = o._replace(starf=o.starf.at[1, 0].set(draw))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.active[0]) == 1 and int(out.ability[1, 0]) == AbilityId.DOWNLOAD
        assert int(out.item[1, 0]) == 0 and int(out.hp[1, 0]) <= top // 4
        stages = np.asarray(out.stages[1])
        assert int(stages[Stat.SPA]) == 6 and int(stages[stat]) == stage


@pytest.mark.parametrize("roar", [False, True], ids=["switch", "roar"])
def test_the_trace_board_law_reads_all_draws(tables, names, roar):
    """The law alone (fast): the copied Download makes side 1's first eat loose (the switch: the
    view's copy and ``_holds``; the drag: the armed-Trace term)."""
    laws = _laws(tables, *_trace_board(tables, names, roar))
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])


def _product_check(tables, s, acts, laws):
    """For one enumerated group (every draw but ``starf[1, 0]`` fixed), the children of the law's
    rows equal those of the full product, that cell at each of the 60 values."""
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


@pytest.mark.slow  # peaks near 8.6 GB, over the fast-tier memory limit
def test_the_law_is_loose_and_the_rows_match_the_full_product(tables, names):
    """The loose law on the Fling board and its children equal the full product. Once loose the
    law enumerates all 60 values, so this half guards only the enumeration plumbing; the tight
    control below is the check with teeth."""
    s, acts = _board(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])
    _product_check(tables, s, acts, laws)


@pytest.mark.slow  # as above
def test_the_tight_control_law_has_five_values_and_matches_the_full_product(tables, names):
    """Control: highest 3 + Fling 1 + Speed Boost 1 = 5 stays below +6, so the law is tight (five
    values), and the berry is eaten with all five stats below +6: the five-value rows' children
    equal the 60-value product's."""
    s, acts = _control(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 0]) and not bool(laws.starf_loose[1, 0])
    _product_check(tables, s, acts, laws)
