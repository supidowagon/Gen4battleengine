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
with Blissey's Growl in place of the X Speed (no damage, no raise: Bidoof still Flings):
highest 3 + Fling 1 + Speed Boost 1 = 5, tight, and the true count at the eat is 5 (the product
check asserts five distinct children; four on the Fling board).

The partner chain (DOUBLES, law alone): see ``_partner_chain``. The Future Sight board (singles):
see ``_foreseen``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import FIRST_ITEM_ACTION, MOVE_ACTIONS
from nuzline.engine.enumerate import enumerate_outcomes, outcome_laws
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import STARF_DRAWS, AbilityId, EffectId, Stat, Status
from nuzline.engine.outcome import Outcome, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.d2_states import four, with_ability, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_enumerate import rows, step_bytes
from tests.engine.test_pivot import _mon


def _board(tables, names):
    s = make_state(
        tables,
        [_mon(names, "Bidoof", ["Fling"], 5, "Salac Berry")],
        [_mon(names, "Blissey", ["Growl"], 100, "Starf Berry")],
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
    """The Fling board with side 1 using Growl instead of the X Speed: Growl deals no damage and
    raises nothing (its negative param is clipped), so Bidoof still Flings, Speed reaches +5
    only and the turn end eats the berry with all five stats below +6."""
    s, _ = _board(tables, names)
    acts = jnp.asarray([names.move_slot(s, 0, "Fling"), names.move_slot(s, 1, "Growl")], jnp.int32)
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


@pytest.mark.slow  # peaks near 10 GB, over the fast-tier memory limit
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


def _partner_chain(tables, names):
    """DOUBLES (battler b's party is b & 1): battler 0 holds Trace with ``trace_announced`` 0, a
    Starf Berry, Sp. Atk +5, poisoned, one poison tick above its pinch. Its foes are battler 1
    (Forecast) and battler 3 (Trace, armed: its foes are 0, Trace, and 2, Forecast). Battler 2
    switches to party 0's slot 2, a Download holder. Every move is Tackle. In
    ``switch_in._trace_state`` (battle_lib.c:3711-3737, restarting after each fire) battler 3
    copies Download from 2 (ChooseTraceTarget, :7750-7781), then battler 0 copies it from 3, and
    its Download fires (:3814-3851): Sp. Atk +6 before the turn end's eat. No foe of battler 0
    enters, so the old term (armed only where a foe enters) added nothing: 5 + 0 = 5, tight."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Tackle"])
    s = with_ability(s, 0, AbilityId.TRACE)
    s = with_ability(s, 1, AbilityId.FORECAST)
    s = with_ability(s, 2, AbilityId.FORECAST)
    s = with_ability(s, 3, AbilityId.TRACE)
    top = int(s.max_hp[0, 0])
    s = with_hp(s, 0, top // 4 + top // 8)
    s = s._replace(
        ability=s.ability.at[0, 2].set(AbilityId.DOWNLOAD),
        item=s.item.at[0, 0].set(names.item("Starf Berry")),
        trace_announced=s.trace_announced.at[0].set(0).at[3].set(0),
        status=s.status.at[0, 0].set(Status.POISON),
        stages=s.stages.at[0, Stat.SPA].set(5),
    )
    acts = jnp.asarray([0, 0, MOVE_ACTIONS + 2, 0], jnp.int32)
    return s, acts


def test_the_partner_switch_trace_chain_law_reads_all_draws(tables, names):
    """The law alone (fast, B = 4): the chain's copied Download makes battler 0's first eat loose
    (``_starf_gain``'s B = 4 armed-Trace term; 008ff3e5 gave the five-value law)."""
    laws = _laws(tables, *_partner_chain(tables, names))
    assert bool(laws.relevant.starf[0, 0]) and bool(laws.starf_loose[0, 0])


def _product_check(tables, s, acts, laws, count):
    """For one enumerated group (every draw but ``starf[1, 0]`` fixed), the children of the law's
    rows equal those of the full product, that cell at each of the 60 values. ``count``: the
    distinct children of the 60 values (the eligible stats at the eat), so the cell is read."""
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
    assert len(want) == count
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
    _product_check(tables, s, acts, laws, 4)


@pytest.mark.slow  # as above
def test_the_tight_control_law_has_five_values_and_matches_the_full_product(tables, names):
    """Control: highest 3 + Fling 1 + Speed Boost 1 = 5 stays below +6, so the law is tight (five
    values), and the berry is eaten with all five stats below +6 (five distinct children): the
    five-value rows' children equal the 60-value product's."""
    s, acts = _control(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 0]) and not bool(laws.starf_loose[1, 0])
    _product_check(tables, s, acts, laws, 5)


def _foreseen(tables, names):
    """Singles: side 1's Blissey holds a Starf Berry two HP above its pinch, and a Future Sight
    (2 damage) lands on it this turn (count 1). Side 0's lead, a poisoned Bidoof at 1 HP, faints
    at the poison tick; its bench is a Geodude. Both use Growl. Blissey's ITEM_CONDITION
    (bcp.c:1699) finds it above its pinch; the Future Sight hit (SIDE_COND_CHECK_STATE_FUTURE_SIGHT,
    bcp.c:1758-1783) lands it there, and subscript_future_sight_damage.s checks no item. TurnEnd's
    ReplaceFainted (bcp.c:1848) sends the Geodude in, and the next ShowBattleMon's held-item check
    over every battler (bcp.c:256, battle_lib.c:4151-4165; the engine's ``replace``) eats the
    berry with its own RandNext() % 5 (battle_lib.c:5273-5294): the replacement's draw,
    ``Outcome.starf[1, 1]`` (``step.starf_kept_active``)."""
    s = make_state(
        tables,
        [_mon(names, "Bidoof", ["Growl"], 5), _mon(names, "Geodude", ["Growl"], 5)],
        [_mon(names, "Blissey", ["Growl"], 100, "Starf Berry")],
    )
    move = names.move("Future Sight")
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4 + 2).replace_hp(0, 0, 1)
    s = s._replace(
        status=s.status.at[0, 0].set(Status.POISON),
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(2),
        future_move=s.future_move.at[1].set(move),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
    )
    acts = jnp.asarray([names.move_slot(s, 0, "Growl"), names.move_slot(s, 1, "Growl")], jnp.int32)
    o = empty_outcome()
    return s, acts, o._replace(future_hit=jnp.asarray([0, 1], o.future_hit.dtype))


def test_the_future_sight_holder_law_reads_the_replacement_draw(tables, names):
    """The law alone (fast): side 1's replacement draw is read for its own active holder, five
    values (0 + 0 + 1 + 2 below +6). 008ff3e5 read it only for a bench holder."""
    s, acts, _ = _foreseen(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 1]) and not bool(laws.starf_loose[1, 1])


@pytest.mark.slow  # a both-engine step plus replace, as the other step checks here
def test_the_future_sight_holder_eats_at_the_replacement_with_its_draw(tables, names):
    """Both engines: the turn leaves Blissey at its pinch with its berry and side 0 to replace;
    ``replace`` eats it with ``starf[1, 1]``: draw 3 is Sp. Atk (rank 3 of five), draw 0 Attack
    (Growl's -1 makes it +1). 008ff3e5 read 0 for both."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import replace
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace

    s, acts, base = _foreseen(tables, names)
    top = int(s.max_hp[1, 0])
    for draw, stat, stage in ((3, Stat.SPA, 2), (0, Stat.ATK, 1)):
        o = base._replace(starf=base.starf.at[1, 1].set(draw))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == top // 4
        assert int(out.item[1, 0]) != 0  # not eaten yet: the hit came after ITEM_CONDITION
        sent = replace(tables, out, 0, 1)
        ref = from_ref(ref_replace(to_numpy(tables), to_ref(out), 0, 1))
        for name, x, y in zip(sent._fields, sent, ref, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
        assert int(sent.item[1, 0]) == 0
        assert int(np.asarray(sent.stages[1])[stat]) == stage


_SPREAD = (
    "RANGE_ADJACENT_OPPONENTS",
    "RANGE_ALL_ADJACENT",
    "RANGE_USER_SIDE",
    "RANGE_FIELD",
    "RANGE_OPPONENT_SIDE",
)


@pytest.mark.parametrize("game", ["vanilla", "kaizo"])
def test_no_spread_move_raises_a_stat_at_its_target(game, request):
    """``_starf_gain`` counts one raise per other battler's move at ``d`` (no per-target count at
    B = 4): that holds while no move of a multi-battler range raises a stat at a battler it hits
    (a positive STAT_FOE / SECONDARY_STAT_FOE stage param, or Swagger), in the built tables."""
    import json

    from tests.conftest import KAIZO_TABLES_DIR, TABLES_DIR

    t = request.getfixturevalue("tables" if game == "vanilla" else "kaizo_tables")
    folder = TABLES_DIR if game == "vanilla" else KAIZO_TABLES_DIR
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["target_names"]
    eff, tgt, prm = (np.asarray(x) for x in (t.move_effect, t.move_target, t.move_param))
    spread = np.isin(tgt, [i for i, n in enumerate(names) if n in _SPREAD])
    foe = np.isin(eff, [int(EffectId.STAT_FOE), int(EffectId.SECONDARY_STAT_FOE)])
    assert np.any(spread & foe)  # the spread stat moves exist (Growl, String Shot, ...)
    raisers = spread & ((foe & (prm[:, 1] > 0)) | (eff == int(EffectId.SWAGGER)))
    assert np.flatnonzero(raisers).tolist() == []
