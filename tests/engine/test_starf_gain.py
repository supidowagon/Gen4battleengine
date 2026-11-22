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
from nuzline.engine.ids import STARF_DRAWS, STARF_STATS, AbilityId, EffectId, Stat, Status
from nuzline.engine.outcome import Outcome, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
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
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])


def _product_check(tables, s, acts, laws, count, cell=(1, 0), first=0):
    """For one enumerated group (every draw but ``starf[cell]`` fixed: the group of row
    ``first``), the children of the law's rows equal those of the full product, that cell at each
    of the 60 values. ``count``: the distinct children of the 60 values (the eligible stats at the
    eat, or the stored draws), so the cell is read."""
    d, k = cell
    outcomes, probs = enumerate_outcomes(laws)
    starf = np.asarray(outcomes.starf).copy()
    starf[:, d, k] = 0
    base = outcomes._replace(starf=starf)
    keys = [b"".join(np.asarray(f)[i].tobytes() for f in base) for i in range(len(probs))]
    group = np.flatnonzero([key == keys[first] for key in keys])
    total = float(np.sum(probs[group]))
    got: dict[bytes, float] = {}
    for key, p in zip(
        step_bytes(tables, s, acts, rows(outcomes, group)), probs[group], strict=True
    ):
        got[key] = got.get(key, 0.0) + float(p) / total
    full = rows(base, np.full(STARF_DRAWS, group[0]))
    cells = np.asarray(full.starf).copy()
    cells[:, d, k] = np.arange(STARF_DRAWS)
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


def _foreseen(tables, names, spa=0):
    """Singles: side 1's Blissey holds a Starf Berry two HP above its pinch, and a Future Sight
    (2 damage) lands on it this turn (count 1). Side 0's lead, a poisoned Bidoof at 1 HP, faints
    at the poison tick; its bench is a Geodude. Both use Growl. Blissey's ITEM_CONDITION
    (bcp.c:1699) finds it above its pinch; the Future Sight hit (SIDE_COND_CHECK_STATE_FUTURE_SIGHT,
    bcp.c:1758-1783) lands it there, and subscript_future_sight_damage.s checks no item. TurnEnd's
    ReplaceFainted (bcp.c:1848) sends the Geodude in, and the next ShowBattleMon's held-item check
    over every battler (bcp.c:256, battle_lib.c:4151-4165; the engine's ``replace``) eats the
    berry with its own RandNext() % 5 (battle_lib.c:5273-5294): the replacement's draw,
    ``Outcome.starf[1, 1]`` (``step.starf_kept_active``). ``spa``: Blissey's Sp. Atk stage."""
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
        stages=s.stages.at[1, Stat.SPA].set(spa),
    )
    acts = jnp.asarray([names.move_slot(s, 0, "Growl"), names.move_slot(s, 1, "Growl")], jnp.int32)
    o = empty_outcome()
    return s, acts, o._replace(future_hit=jnp.asarray([0, 1], o.future_hit.dtype))


def test_the_future_sight_holder_law_reads_the_replacement_draw(tables, names):
    """The law alone (fast): side 1's replacement draw is read for its own active holder, five
    values: highest 0 + gain 0 + 1 + 2 * own = 3, below +6 (``own`` is 1: Blissey holds the berry
    on the field and the Future Sight can take HP, ``held_eat`` through ``hp_drop``). 008ff3e5
    read it only for a bench holder."""
    s, acts, _ = _foreseen(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 1]) and not bool(laws.starf_loose[1, 1])


@pytest.mark.slow  # a both-engine step plus replace, as the other step checks here
@pytest.mark.parametrize(
    ("spa", "draw", "stored", "stat", "stage"),
    [(0, 8, 3, Stat.SPA, 2), (0, 0, 0, Stat.ATK, 1), (5, 7, 7, Stat.SPE, 2)],
    ids=["mod5-spa", "mod5-atk", "whole-spe"],
)
def test_the_future_sight_holder_eats_at_the_replacement_with_its_draw(
    tables, names, spa, draw, stored, stat, stage
):
    """Both engines: the turn leaves Blissey at its pinch with its berry and side 0 to replace;
    ``replace`` eats it with ``starf[1, 1]``, kept in ``starf_roll[1, 0]``: modulo 5 while its
    five stats are all at +4 or below (draw 8 is kept as 3: Sp. Atk, rank 3 of five; draw 0
    Attack, Growl's -1 makes it +1), whole with Sp. Atk at +5 (the law's 60 values: draw 7 is
    kept as 7; the eat meets five stats below +6, rank 2, Speed). 008ff3e5 read 0 for every
    draw; f840758b kept draw 8 whole."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import replace
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace

    s, acts, base = _foreseen(tables, names, spa)
    top = int(s.max_hp[1, 0])
    o = base._replace(starf=base.starf.at[1, 1].set(draw))
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == top // 4
    assert int(out.item[1, 0]) != 0  # not eaten yet: the hit came after ITEM_CONDITION
    assert np.asarray(out.starf_roll[1]).tolist() == [stored, 0, 0]
    sent = replace(tables, out, 0, 1)
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(out), 0, 1))
    for name, x, y in zip(sent._fields, sent, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert int(sent.item[1, 0]) == 0
    assert int(np.asarray(sent.stages[1])[stat]) == stage


@pytest.mark.slow  # enumerates and steps every row, as the other product checks here
def test_the_future_sight_holder_rows_match_the_full_product(tables, names):
    """Review 3, must 1: the tight index-1 law (five values) on ``_foreseen``: for the group whose
    Future Sight hits, the five rows' children equal the 60-value product's over ``starf[1, 1]``.
    Five distinct children: the kept draw is stored modulo 5 (f840758b stored it whole: 60)."""
    s, acts, _ = _foreseen(tables, names)
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 1]) and not bool(laws.starf_loose[1, 1])
    outcomes, _ = enumerate_outcomes(laws)
    hit = int(np.argmax(np.asarray(outcomes.future_hit)[:, 1]))
    _product_check(tables, s, acts, laws, 5, cell=(1, 1), first=hit)


def _waiting(tables, names, variant="eats"):
    """Singles, the state after a ``_foreseen`` turn: side 0's lead fainted, a Geodude to send in;
    side 1's Blissey at its pinch with its Starf Berry. ``variant``: ``eats`` (it can eat),
    ``klutz`` / ``embargo`` (``Battler_HeldItem`` gives ITEM_NONE, lib.c:5352-5363), ``maxed``
    (all five stats at +6, lib.c:5281-5286), ``gastro-gluttony`` (a Gluttony holder under Gastro
    Acid at a third of its HP: ``Battler_Ability`` is none, lib.c:3088-3093, so the param is not
    halved, lib.c:5273-5278, and it is above its pinch)."""
    s = make_state(
        tables,
        [_mon(names, "Bidoof", ["Growl"], 5), _mon(names, "Geodude", ["Growl"], 5)],
        [_mon(names, "Blissey", ["Growl"], 100, "Starf Berry")],
    )
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4).replace_hp(0, 0, 0)
    if variant == "klutz":
        s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.KLUTZ))
    elif variant == "embargo":
        s = s._replace(embargo=s.embargo.at[1].set(2))
    elif variant == "maxed":
        stages = s.stages
        for x in STARF_STATS:
            stages = stages.at[1, int(x)].set(6)
        s = s._replace(stages=stages)
    elif variant == "gastro-gluttony":
        s = s.replace_hp(1, 0, top // 3)
        s = s._replace(
            ability=s.ability.at[1, 0].set(AbilityId.GLUTTONY),
            ability_suppressed=s.ability_suppressed.at[1].set(1),
        )
    return s


@pytest.mark.parametrize("variant", ["eats", "klutz", "embargo", "maxed", "gastro-gluttony"])
def test_only_a_holder_that_can_eat_keeps_its_draw(tables, names, variant):
    """Review 3, must 2 (fast, both engines): ``starf_kept_active`` needs the effective item
    (Klutz, Embargo: lib.c:5352-5363, read by TriggerHeldItemOnStatus at :4970) and a stat below
    +6 (lib.c:5281-5286), as the law's ``standing``. f840758b kept it for the first four.
    Review 4, must 1: ``gastro-gluttony`` needs the effective ability's pinch test
    (lib.c:5273-5278, ``order.pinch_divisor``); 20137b9e kept it in step (raw Gluttony)."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import starf_kept, starf_kept_active
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _starf_kept, _starf_kept_active

    s = _waiting(tables, names, variant)
    want = [False, variant == "eats"]
    assert np.asarray(starf_kept_active(tables, s)).tolist() == want
    assert np.asarray(starf_kept(tables, s)).tolist() == want
    np_tables, rs = to_numpy(tables), to_ref(s)
    assert _starf_kept_active(np_tables, rs) == want
    assert _starf_kept(np_tables, rs) == want


@pytest.mark.parametrize(("spa", "roll", "stat"), [(0, 3, Stat.SPA), (6, 7, Stat.SPD)])
def test_the_b2_replace_eats_the_active_holder_with_entry_0(tables, names, spa, roll, stat):
    """Review 3, not verified (B = 2): ``replace``'s ``hooks.switch_in_effects`` runs the foe's
    ``after_damage_items`` (bcp.c:256, battle_lib.c:4151-4165) and Blissey's eat reads
    ``starf_roll[1, 0]`` (``starf_taken`` 0): roll 3 of five is Sp. Atk; with Sp. Atk at +6 roll 7
    is rank 3 of four, Sp. Def. Both engines."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import replace
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace

    s = _waiting(tables, names)
    s = s._replace(
        stages=s.stages.at[1, Stat.SPA].set(spa),
        starf_roll=s.starf_roll.at[1].set(jnp.asarray([roll, 0, 0], s.starf_roll.dtype)),
    )
    sent = replace(tables, s, 0, 1)
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(s), 0, 1))
    for name, x, y in zip(sent._fields, sent, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert int(sent.item[1, 0]) == 0 and int(np.asarray(sent.stages[1])[stat]) == 2


def _own_salac(tables, names):
    """Review 4, must 2; review 5, should 1 (singles): side 1's Blissey (level 100) holds a Salac
    Berry at Speed +5 (its other stats at 0), one HP above its pinch, and uses Double-Edge; side
    0's Shuckle (level 50) Flings its Starf Berry at it. Blissey moves first (Speed 130 at +5 to
    Shuckle's 17) and Double-Edge leaves Shuckle standing (Attack 40 into Defense 242, Rock
    resists: 27 at most, a crit with the top roll, of its 87 HP). The recoil
    (subscript_recoil_1_3.s:10-14, BattleSystem_Divide, at least 1; UPDATE_HP with no item check)
    puts Blissey at its pinch (lib.c:4799-4809) and AFTER_MOVE_EFFECT_ATTACKER_ITEM
    (bcp.c:3684-3689) eats the Salac: Speed +6. Shuckle's TryFling (effect_script_0233.s:7,
    lib.c:6110-6127) then reads the draw with four stats below +6. The old bound: highest 5 +
    gain 0 = 5, tight."""
    s = make_state(
        tables,
        [_mon(names, "Shuckle", ["Fling"], 50, "Starf Berry")],
        [_mon(names, "Blissey", ["Double-Edge"], 100, "Salac Berry")],
    )
    assert int(s.stats[1, 0, Stat.SPE - 1]) > int(s.stats[0, 0, Stat.SPE - 1])
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4 + 1)
    s = s._replace(stages=s.stages.at[1, Stat.SPE].set(5))
    acts = jnp.asarray(
        [names.move_slot(s, 0, "Fling"), names.move_slot(s, 1, "Double-Edge")], jnp.int32
    )
    return s, acts


def test_a_flung_starf_after_the_own_pinch_berry_reads_all_draws(tables, names):
    """Review 4, must 2 (fast, law alone): ``_starf_gain`` plus d's own held stat pinch berry,
    +1 on the flung path: side 1's flung-Starf draw is loose (20137b9e: tight, five values)."""
    laws = _laws(tables, *_own_salac(tables, names))
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])


def test_a_plucked_starf_after_the_own_pinch_berry_reads_all_draws(tables, names):
    """Review 5, note (fast, law alone): the plucked path. Side 0's Jolteon (level 50, Speed 142)
    holds a Starf Berry and Tackles side 1's Blissey (level 100, Speed 130), which holds a Liechi
    Berry at Attack +5, one HP above its pinch: the hit (31 at most, 62 with a crit, of 159) puts
    it at its pinch and AFTER_MOVE_EFFECT_DEFENDER_ITEM (bcp.c:3691-3696) eats the Liechi:
    Attack +6. Blissey's Bug Bite then eats the Starf Berry with four stats below +6: side 1's
    Starf draw is loose (the EAT_BERRY +1 and the own-berry +1 both count it)."""
    s = make_state(
        tables,
        [_mon(names, "Jolteon", ["Tackle"], 50, "Starf Berry")],
        [_mon(names, "Blissey", ["Bug Bite"], 100, "Liechi Berry")],
    )
    assert int(s.stats[0, 0, Stat.SPE - 1]) > int(s.stats[1, 0, Stat.SPE - 1])
    top = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, top // 4 + 1)
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(5))
    acts = jnp.asarray(
        [names.move_slot(s, 0, "Tackle"), names.move_slot(s, 1, "Bug Bite")], jnp.int32
    )
    laws = _laws(tables, s, acts)
    assert bool(laws.relevant.starf[1, 0]) and bool(laws.starf_loose[1, 0])


def _doubles_waiting(tables, names, holder, fainted):
    """DOUBLES (battler b's party is b & 1, every Pokemon a Bulbasaur): battler ``holder``'s
    active Pokemon holds a Starf Berry at its pinch, battler ``fainted``'s active Pokemon is at
    0 HP with a bench to send in; turn 1, no newcomer waiting."""
    s = four(tables, names, DOUBLES)
    s = s._replace(entry_pending=jnp.zeros_like(s.entry_pending), turn=jnp.asarray(1, s.turn.dtype))
    p, slot = party_of(s, holder), int(s.active[holder])
    top = int(s.max_hp[p, slot])
    s = with_hp(s, holder, top // 4)
    s = s._replace(item=s.item.at[p, slot].set(names.item("Starf Berry")))
    return with_hp(s, fainted, 0)


def test_the_partner_active_holder_is_not_the_replacement_bench(tables, names):
    """Review 3, should 3 (fast, B = 4, shared party): battler 0's pinched active holder is not on
    battler 2's replacement bench (``actions.replacement_bench``, bcp.c:4096-4107): battler 2's
    ``starf_kept`` is False (f840758b: True, ``slots != active[2]`` counted slot 0); battler 0
    keeps its own draw (``starf_kept_active``). Both engines."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import starf_kept
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _starf_kept

    s = _doubles_waiting(tables, names, 0, 2)
    want = [True, False, False, False]
    assert np.asarray(starf_kept(tables, s)).tolist() == want
    assert _starf_kept(to_numpy(tables), to_ref(s)) == want


@pytest.mark.parametrize(
    ("holder", "fainted", "pick"), [(1, 0, 2), (0, 2, 2)], ids=["foe-holder", "partner-holder"]
)
def test_the_b4_active_holder_eats_at_the_next_show_battle_mon(
    tables, names, holder, fainted, pick
):
    """Review 3, not verified (B = 4, G-32 / ``entry_pending``): ``replace_round`` sends the
    newcomer in, its states wait (``entry_pending``, D3-13), the active holder's kept draw stays
    in ``starf_roll`` entry 0 (``starf_kept_active``), the other battlers' are 0, and
    ``show_battle_mon4``'s SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS (battle_lib.c:4151-4165, every
    battler) eats it with that draw: roll 3 is Sp. Atk. Both engines."""
    from nuzline.data.tables import to_numpy
    from nuzline.engine.step import replace_round
    from nuzline.engine.switch_in import show_battle_mon4
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace_round, ref_show_battle_mon4

    s = _doubles_waiting(tables, names, holder, fainted)
    s = s._replace(starf_roll=s.starf_roll.at[holder, 0].set(3))
    picks = np.full(4, -1, np.int8)
    picks[fainted] = pick
    np_tables = to_numpy(tables)

    def same(x_state, ref_state):
        ref = from_ref(ref_state)
        for name, x, y in zip(x_state._fields, x_state, ref, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)

    waiting = replace_round(tables, s, picks)
    same(waiting, ref_replace_round(np_tables, to_ref(s), list(picks)))
    assert int(waiting.entry_pending[fainted]) == 1
    rolls = np.asarray(waiting.starf_roll)
    assert rolls[holder].tolist() == [3, 0, 0]
    assert not np.any(np.delete(rolls, holder, axis=0))
    ties = (0,) * 6
    shown = show_battle_mon4(tables, waiting, np.asarray(ties, np.int32))
    same(shown, ref_show_battle_mon4(np_tables, to_ref(waiting), list(ties)))
    p, slot = party_of(shown, holder), int(shown.active[holder])
    assert int(shown.item[p, slot]) == 0
    assert int(np.asarray(shown.stages[holder])[Stat.SPA]) == 2


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
