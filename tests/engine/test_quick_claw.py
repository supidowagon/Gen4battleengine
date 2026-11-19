"""Quick Claw (K3b2 Task 8, ``ItemId.QUICK_CLAW``, ``BattleState.qc_roll``,
``Outcome.quick_claw``; docs/kaizo/k3b2-rules.md §0.2a, §5).

Decomp (external/pokeplatinum):

* ``BattleSystem_CompareBattlerSpeed`` (src/battle/battle_lib.c:1188-1445): after the 0-HP
  shortcut (:1204-1211) each battler's ``HOLD_EFFECT_SOMETIMES_PRIORITY`` activates when
  ``speedRand[b] % (100 / param) == 0`` (:1278-1286, :1344-1352), whatever
  ``ignoreQuickClaw`` (only the ``moveEffectsData.quickClaw`` write needs it FALSE). Within
  equal priority both active compare plain speeds (:1402-1407), one active goes first
  (:1408-1411), and only then Lagging Tail (:1412-1421), Stall and Trick Room.
* ``speedRand`` is one ``RandNext()`` per battler slot (``MAX_BATTLERS``, holder or not):
  at battle start (``BattleContext_InitCounters``, battle_lib.c:2019-2025, from
  ``BattleContext_New``, battle_controller_player.c:185) and in the turn's pre-move step
  ``PRE_MOVE_ACTION_STATE_SPEED_RNG`` (bcp.c:853-857), which runs after
  ``BattleControllerPlayer_CalcTurnOrder`` (bcp.c:707-790 sets ``CHECK_PRE_MOVE_ACTIONS``).
  So a turn's order reads the previous turn's draw; the same turn's draw is read by every
  ``SortMonSpeedOrder`` of the turn (bcp.c:888: the end-of-turn order), the next turn's
  start-of-turn sorts (bcp.c:251, :263), the AI's ``IfSpeedCompare*`` (trainer_ai.c:
  1102-1124) and the next ``CalcTurnOrder``.
* ``subscript_check_quick_claw.s`` (first step of ``BeforeMove``, bcp.c:3140-3143): the
  held-item animation, no text, when ``moveEffectsData.quickClaw`` is set (only by
  ``CalcTurnOrder`` between two FIGHT commands) and the battler is not the last to act.
* Quick Claw (res/items/data/quick_claw.json): ``HOLD_EFFECT_SOMETIMES_PRIORITY`` (52),
  ``effectParam`` 20, the only item with that hold effect.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.coverage import check_coverage
from nuzline.engine.enumerate import (
    _LAWS,
    INPUT_NEUTRAL_ITEMS,
    enumerate_drawn,
)
from nuzline.engine.ids import QUICK_CLAW_MODULUS, QUICK_CLAW_PARAM, ItemId, Stat, Status
from nuzline.engine.observe import observe
from nuzline.engine.order import MOVE_ACTIONS, action_order, player_second
from nuzline.engine.outcome import Outcome, _can_go_first, empty_outcome, hit_thresholds
from nuzline.engine.state import make_state
from nuzline.engine.step import (
    WINNER_OPPONENT,
    WINNER_PLAYER,
    end_of_turn_order,
    needs_replacement,
    qc_kept,
    replace,
    root_belief,
    sample_root,
    step,
    step_events,
)
from nuzline.prep.problem import _item_refusal
from nuzline.search.plan_tree import _FLAG_TEXT, _QUICK_TEXT
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    jit_step,
    roark_state,
    sample_party,
    sample_rows,
)
from tests.engine.test_enumerate import (
    _kaizo_spec,
    assert_equals_bruteforce,
    check_frequencies,
    children,
    laws_of,
    sampled,
)

DECOMP = Path("external/pokeplatinum")
_SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe
SWITCH = MOVE_ACTIONS + 1  # to party slot 1
_step = jax.jit(step)


def _claw(s, side, names, slot=0, item="Quick Claw"):
    return s._replace(item=s.item.at[side, slot].set(names.item(item)))


def _state(tables, names, *, order="opp", qc=(0, 0), holders=(0,), mine=None, opp=None, **kw):
    """Monferno against Geodude, Geodude faster by one (``order``: "opp", "player" or
    "tie"), the ``holders`` sides' leads holding a Quick Claw, ``qc_roll = qc``."""
    s = roark_state(
        tables,
        names,
        player_moves=list(mine or ["Scratch", "Leer", "Quick Attack", "Ember"]),
        opp_moves=list(opp or ["Tackle"]),
        **kw,
    )
    for side in holders:
        s = _claw(s, side, names)
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    return s._replace(qc_roll=jnp.asarray(qc, s.qc_roll.dtype))


def _o(quick_claw=(0, 0), **fields) -> Outcome:
    o = empty_outcome()._replace(quick_claw=jnp.asarray(quick_claw, jnp.int8))
    return o._replace(**{k: jnp.asarray(v, jnp.int8) for k, v in fields.items()})


def _first(tables, s, acts, o=None) -> int:
    return int(action_order(tables, s, acts, empty_outcome() if o is None else o)[0])


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


# --- mapping, data and decomp -------------------------------------------------------------


def test_quick_claw_maps_in_both_games(tables, names, kaizo_tables, kaizo_names):
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        item = n.item("Quick Claw")
        assert item == 217
        assert int(t.item_effect[item]) == ItemId.QUICK_CLAW
        assert int(t.item_param[item, 0]) == QUICK_CLAW_PARAM == 20
        assert int(np.sum(np.asarray(t.item_effect) == ItemId.QUICK_CLAW)) == 1
    assert QUICK_CLAW_MODULUS == 100 // QUICK_CLAW_PARAM == 5


def _decomp(path: str) -> str:
    p = DECOMP / path
    if not p.exists():
        pytest.skip("external/pokeplatinum not cloned")
    return p.read_text(encoding="utf-8")


def test_item_data_is_the_only_sometimes_priority_item():
    claw = json.loads(_decomp("res/items/data/quick_claw.json"))
    assert claw["holdEffect"] == "HOLD_EFFECT_SOMETIMES_PRIORITY"
    assert claw["effectParam"] == QUICK_CLAW_PARAM
    others = [
        p.name
        for p in (DECOMP / "res/items/data").glob("*.json")
        if '"HOLD_EFFECT_SOMETIMES_PRIORITY"' in p.read_text(encoding="utf-8")
    ]
    assert others == ["quick_claw.json"]


def test_compare_speed_reads_speed_rand_and_beats_lagging_tail():
    lib = _decomp("src/battle/battle_lib.c")
    body = lib[lib.index("u8 BattleSystem_CompareBattlerSpeed(") :]
    body = body[: body.index("\n}\n")]
    for b in ("battler1", "battler2"):
        test = (
            f"if (battleCtx->speedRand[{b}] % (100 / {b}ItemParam) == 0) {{\n"
            f"            {b}QuickClaw = 1;\n\n"
            "            if (ignoreQuickClaw == FALSE) {"
        )
        assert test in body  # the flag whatever ignoreQuickClaw; only the write needs it
    # Within equal priority: Quick Claw (both, then one) before Lagging Tail and Trick Room.
    tail = body.index("if (battler1Priority == battler2Priority)")
    order = [
        body.index(s, tail)
        for s in (
            "if (battler1QuickClaw && battler2QuickClaw)",
            "battler1QuickClaw == FALSE && battler2QuickClaw",
            "battler1LaggingTail && battler2LaggingTail",
            "FIELD_CONDITION_TRICK_ROOM",
        )
    ]
    assert order == sorted(order)
    both = body[order[0] : order[1]]
    assert "TRICK_ROOM" not in both and "battler1Speed < battler2Speed" in both


def test_speed_rand_is_drawn_after_the_turn_order_for_every_battler():
    bcp = _decomp("src/battle/battle_controller_player.c")
    sig = "(BattleSystem *battleSys, BattleContext *battleCtx)\n{"
    calc = bcp.index("static void BattleControllerPlayer_CalcTurnOrder" + sig)
    calc_end = bcp.index("battleCtx->command = BATTLE_CONTROL_CHECK_PRE_MOVE_ACTIONS;", calc)
    pre = bcp.index("static void BattleControllerPlayer_CheckPreMoveActions" + sig)
    draw = bcp.index("case PRE_MOVE_ACTION_STATE_SPEED_RNG:", pre)
    assert calc < calc_end < pre < draw
    block = bcp[draw : bcp.index("break;", draw)]
    # Every battler slot draws (holder or not).
    assert "for (battler = 0; battler < MAX_BATTLERS; battler++)" in block
    assert "battleCtx->speedRand[battler] = BattleSystem_RandNext(battleSys);" in block
    # The action order's per-action sort reads that draw (bcp.c:888, BranchActions).
    branch = bcp.index("static void BattleControllerPlayer_BranchActions" + sig)
    assert "BattleSystem_SortMonSpeedOrder(battleSys, battleCtx);" in bcp[branch : branch + 900]
    # The battle start draws it too.
    lib = _decomp("src/battle/battle_lib.c")
    init = lib[lib.index("void BattleContext_InitCounters(") :]
    assert "battleCtx->speedRand[i] = BattleSystem_RandNext(battleSys);" in init[:400]
    assert re.search(r"BattleContext_InitCounters\(battleSys, battleContext\);", bcp)


def test_animation_script_needs_the_flag_and_not_acting_last():
    script = _decomp("res/battle/scripts/subscripts/subscript_check_quick_claw.s")
    quick = script.index("BATTLEMON_QUICK_CLAW, 0, _063")
    waiting = script.index("BTLVAR_WAITING_BATTLERS, 1, _039", quick)
    anim = script.index("PlayBattleAnimation BTLSCR_ATTACKER, BATTLE_ANIMATION_HELD_ITEM", waiting)
    assert quick < waiting < anim
    # No message on the Quick Claw path: the "let it move first" text is Custap's (_022).
    assert script.index("PrintMessage") > script.index("_022:")


# --- the law -------------------------------------------------------------------------------


def test_law_is_one_in_five(tables, names):
    assert _LAWS["quick_claw"](None, 0) == ([0, 1], [4, 1], 5)
    s = _state(tables, names)
    keys = jax.random.split(jax.random.key(3), 20000)

    draws = sample_rows(keys, tables, s, A("Scratch", "Tackle", s, names))
    freq = np.asarray(draws.quick_claw).mean(axis=0)
    assert np.all(np.abs(freq - 0.2) < 5 * math.sqrt(0.16 / 20000))


# --- the turn order -----------------------------------------------------------------------


def test_the_stored_draw_moves_the_slower_holder_first(tables, names):
    s = _state(tables, names)  # Geodude faster, Monferno holds the claw
    acts = A("Scratch", "Tackle", s, names)
    assert _first(tables, s, acts) == 1
    on = s._replace(qc_roll=s.qc_roll.at[0].set(1))
    assert _first(tables, on, acts) == 0
    # The turn's own draw never orders the turn (it is drawn after CalcTurnOrder).
    assert _first(tables, s, acts, _o((1, 0))) == 1
    assert _first(tables, on, acts, _o((0, 0))) == 0
    # A non-holder's bit does nothing.
    foe_bit = s._replace(qc_roll=s.qc_roll.at[1].set(1))
    assert _first(tables, foe_bit._replace(item=foe_bit.item.at[0, 0].set(0)), acts) == 1


def test_priority_comes_before_quick_claw(tables, names):
    s = _state(tables, names, order="player", holders=(1,), qc=(0, 1))
    assert _first(tables, s, A("Quick Attack", "Tackle", s, names)) == 0
    assert _first(tables, s, A("Scratch", "Tackle", s, names)) == 1


def test_both_active_compare_plain_speed(tables, names):
    s = _state(tables, names, holders=(0, 1), qc=(1, 1))
    acts = A("Scratch", "Tackle", s, names)
    assert _first(tables, s, acts) == 1  # Geodude is faster
    # Trick Room is ignored when both are active (lib.c:1402-1407) ...
    tr = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    assert _first(tables, tr, acts) == 1
    # ... and a lone active holder goes first even under Trick Room.
    lone = tr._replace(qc_roll=tr.qc_roll.at[1].set(0))
    assert _first(tables, lone, acts) == 0
    # A speed tie between two active claws: speed_tie decides.
    tie = _state(tables, names, order="tie", holders=(0, 1), qc=(1, 1))
    assert [_first(tables, tie, acts, _o(speed_tie=t)) for t in (0, 1)] == [0, 1]


def test_quick_claw_beats_lagging_tail(tables, names):
    # The faster foe holds a Lagging Tail, the slower player a Quick Claw.
    s = _claw(_state(tables, names, order="player"), 1, names, item="Lagging Tail")
    s = _claw(s, 0, names)._replace(stats=s.stats.at[1, 0, _SPE].set(int(s.stats[0, 0, _SPE]) + 5))
    acts = A("Scratch", "Tackle", s, names)
    assert _first(tables, s, acts) == 0  # the Lagging Tail holder goes second anyway
    # The Quick Claw holder facing a Lagging Tail holder that is faster: claw first.
    s2 = _claw(_state(tables, names), 1, names, item="Lagging Tail")
    for qc in (0, 1):
        t = s2._replace(qc_roll=s2.qc_roll.at[0].set(qc))
        assert _first(tables, t, acts) == 0
    # The claw holder is the faster one; the Lagging Tail foe is slower: the claw goes
    # first whatever, and two claws ignore the Lagging Tail rule.
    assert bool(player_second(tables, s2, 0, jnp.asarray([0, 0]))) is False


def test_two_switches_are_ordered_by_the_outgoing_holder(tables, names):
    s = _state(tables, names, qc=(1, 0))
    acts = jnp.asarray([SWITCH, SWITCH], jnp.int32)
    assert _first(tables, s, acts) == 0
    assert _first(tables, s._replace(qc_roll=s.qc_roll.at[0].set(0)), acts) == 1
    # A switch against a move: the switch first, Quick Claw or not.
    assert _first(tables, s, jnp.asarray([0, SWITCH], jnp.int32)) == 1


def test_can_go_first_and_accuracy_thresholds_see_quick_claw(tables, names):
    # The slower player's Sand-Attack lowers Geodude's Tackle only when it can go first.
    s = _state(tables, names, mine=["Sand-Attack"])
    acts = A("Sand-Attack", "Tackle", s, names)
    assert not bool(_can_go_first(tables, s, acts)[0])
    on = s._replace(qc_roll=s.qc_roll.at[0].set(1))
    assert bool(_can_go_first(tables, on, acts)[0])
    ts_off, ok_off = (np.asarray(x)[1] for x in hit_thresholds(tables, s, acts))
    ts_on, ok_on = (np.asarray(x)[1] for x in hit_thresholds(tables, on, acts))
    assert len(set(ts_on[ok_on])) > len(set(ts_off[ok_off]))


# --- the end-of-turn order ----------------------------------------------------------------------


@pytest.mark.slow
def test_this_turns_draw_orders_the_end_of_turn(tables, names):
    # Both last Pokemon poisoned at 1 HP: the battler checked first faints first and its side
    # loses (bcp.c:1289). The faster Geodude goes first unless Monferno's Quick Claw draw of
    # this turn activates (the stored qc_roll is the turn order's, not this one's).
    s = _state(tables, names, player_count=1, opp_count=1, mine=["Leer"], opp=["Leer"])
    s = s.with_status(0, 0, Status.POISON).with_status(1, 0, Status.POISON)
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    acts = A("Leer", "Leer", s, names)
    assert end_of_turn_order(tables, s, _o((0, 0))).tolist() == [1, 0]
    assert end_of_turn_order(tables, s, _o((1, 0))).tolist() == [0, 1]
    for qc_roll in (0, 1):
        t = s._replace(qc_roll=s.qc_roll.at[0].set(qc_roll))
        out = assert_matches_reference(tables, t, acts, _o((0, 0)))
        assert int(out.winner) == WINNER_PLAYER
        out = assert_matches_reference(tables, t, acts, _o((1, 0)))
        assert int(out.winner) == WINNER_OPPONENT


# --- storage (the tight rule) ---------------------------------------------------------------------


def test_the_draw_is_stored_for_an_active_holder(tables, names):
    s = _state(tables, names, holders=(0,))
    acts = A("Leer", "Leer", s, names)
    for bit in (0, 1):
        out = assert_matches_reference(tables, s, acts, _o((bit, bit)))
        assert [int(x) for x in out.qc_roll] == [bit, 0]  # the foe holds none: 0
    # And it orders the next turn.
    out = _step(tables, s, acts, _o((1, 0)))
    assert _first(tables, out, A("Scratch", "Tackle", out, names)) == 0


def test_the_draw_follows_the_battler_on_the_field(tables, names):
    # Monferno (holder) switches out: nothing reads the slot's draw before the redraw.
    s = _state(tables, names, holders=(0,), qc=(1, 0))
    out = assert_matches_reference(tables, s, jnp.asarray([SWITCH, 0], jnp.int32), _o((1, 0)))
    assert int(out.qc_roll[0]) == 0
    # A holder switching in keeps the turn's draw (the slot's speedRand).
    s = _claw(_state(tables, names, holders=()), 0, names, slot=1)
    out = assert_matches_reference(tables, s, jnp.asarray([SWITCH, 0], jnp.int32), _o((1, 0)))
    assert int(out.qc_roll[0]) == 1


def test_a_pending_replacement_keeps_the_draw_for_a_bench_holder(tables, names):
    # Monferno at 1 HP faints to Tackle; slot 1 holds a Quick Claw, slot 2 does not.
    s = _claw(_state(tables, names, holders=()), 0, names, slot=1).replace_hp(0, 0, 1)
    acts = A("Leer", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o((1, 0)))
    assert int(needs_replacement(out)[0]) == 1 and int(out.qc_roll[0]) == 1
    assert bool(qc_kept(tables, out)[0])
    assert int(replace(tables, out, 0, 1).qc_roll[0]) == 1  # the holder comes in
    assert int(replace(tables, out, 0, 2).qc_roll[0]) == 0  # a non-holder: unread
    # No living holder on the bench: not stored.
    t = _claw(s, 0, names, slot=1, item="Oran Berry")
    assert int(_step(tables, t, acts, _o((1, 0))).qc_roll[0]) == 0
    dead = _claw(s, 0, names, slot=1).replace_hp(0, 1, 0)
    assert int(_step(tables, dead, acts, _o((1, 0))).qc_roll[0]) == 0


def test_a_fainted_holder_with_no_holder_left_is_not_stored(tables, names):
    s = _state(tables, names, holders=(0,)).replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", "Tackle", s, names), _o((1, 1)))
    assert int(out.hp[0, 0]) == 0 and int(out.qc_roll[0]) == 0


def test_a_finished_battle_stores_nothing(tables, names):
    s = _state(tables, names, holders=(0, 1), player_count=1, opp_count=1).replace_hp(1, 0, 1)
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(1))  # Monferno first: Scratch KOs
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o((1, 1)))
    assert int(out.over) == 1 and [int(x) for x in out.qc_roll] == [0, 0]


@pytest.mark.slow
def test_reference_agreement_over_draws(tables, names):
    """Both engines agree over the stored bit, the turn's draw and the speed ties, for
    holders on either or both sides, Lagging Tail, Trick Room, switches and faints."""
    for holders, order, tail, trick, acts_kind, low in itertools.product(
        ((0,), (1,), (0, 1)),
        ("player", "opp", "tie"),
        (False, True),
        (False, True),
        ("moves", "switch", "priority"),
        (False, True),
    ):
        s = _state(tables, names, order=order, holders=holders)
        if tail:
            free = 1 if holders == (0,) else 0
            s = _claw(s, free, names, item="Lagging Tail")
        if trick:
            s = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
        if low:
            s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
        acts = {
            "moves": A("Scratch", "Tackle", s, names),
            "switch": jnp.asarray([SWITCH, SWITCH], jnp.int32),
            "priority": A("Quick Attack", "Tackle", s, names),
        }[acts_kind]
        for start in itertools.product((0, 1), (0, 1)):
            t = s._replace(qc_roll=jnp.asarray(start, s.qc_roll.dtype))
            for qc, tie in itertools.product(itertools.product((0, 1), (0, 1)), (0, 1)):
                assert_matches_reference(tables, t, acts, _o(qc, speed_tie=tie, end_speed_tie=tie))


def test_replace_matches_reference(tables, names):
    from nuzline.data.tables import to_numpy
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_replace

    s = _claw(_state(tables, names, holders=()), 0, names, slot=1).replace_hp(0, 0, 1)
    out = _step(tables, s, A("Leer", "Tackle", s, names), _o((1, 0)))
    for slot in (1, 2):
        jax_out = replace(tables, out, 0, slot)
        ref = from_ref(ref_replace(to_numpy(tables), to_ref(out), 0, slot))
        for name, x, y in zip(jax_out._fields, jax_out, ref, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


# --- observation --------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_animation_is_seen_when_the_claw_moves_it_first(tables, names):
    s = _state(tables, names, qc=(1, 0))
    acts = A("Scratch", "Tackle", s, names)
    ev = _events(tables, s, acts, _o())
    assert ev["quick_claw"].tolist() == [1, 0] and int(ev["order"][0]) == 0
    # The flag is set whatever the priorities, but the animation needs acting first.
    s2 = _state(tables, names, holders=(1,), qc=(0, 1), order="player")
    assert _events(tables, s2, A("Quick Attack", "Tackle", s2, names), _o())[
        "quick_claw"
    ].tolist() == [0, 0]
    # A faster holder whose claw activates still shows it.
    s3 = _state(tables, names, holders=(1,), qc=(0, 1))
    assert _events(tables, s3, A("Scratch", "Tackle", s3, names), _o())["quick_claw"].tolist() == [
        0,
        1,
    ]
    # Not between a switch and a move, nor two switches (no FIGHT compare / no move).
    for a in ([SWITCH, 0], [0, SWITCH], [SWITCH, SWITCH]):
        assert _events(tables, s, jnp.asarray(a, jnp.int32), _o())["quick_claw"].tolist() == [0, 0]
    # The turn's own draw is not seen: observations equal over it.
    obs = [
        np.asarray(observe(tables, s, acts, _o((k, 0)), jit_step(tables, s, acts, _o((k, 0)))))
        for k in (0, 1)
    ]
    assert (obs[0] == obs[1]).all()
    # The render reads the two bits: the animation (Quick Claw or Custap) and Custap's message.
    assert "quick_claw" not in dict(_FLAG_TEXT)
    assert [t for bit, t in _QUICK_TEXT if 3 & bit] == [
        "Quick Claw or Custap Berry activated",
        "Custap Berry let it move first",
    ]
    assert [t for bit, t in _QUICK_TEXT if 1 & bit] == ["Quick Claw or Custap Berry activated"]


# --- relevance and enumeration -------------------------------------------------------------------


def test_relevance_of_the_turns_draw(tables, names):
    def rel(s, acts):
        return np.asarray(laws_of(tables, s, acts).relevant.quick_claw).tolist()

    s = _state(tables, names, holders=())
    acts = A("Scratch", "Tackle", s, names)
    assert rel(s, acts) == [False, False]
    assert rel(_claw(s, 0, names), acts) == [True, False]
    assert rel(_claw(s, 1, names), acts) == [False, True]
    bench = _claw(s, 0, names, slot=3)
    assert rel(bench, acts) == [True, False]  # it may replace a fainted active
    assert rel(bench.replace_hp(0, 3, 0), acts) == [False, False]
    over = _claw(s, 0, names)._replace(over=jnp.asarray(1, s.over.dtype))
    assert rel(over, acts) == [False, False]


def test_damage_classes_classify_quick_claw():
    assert ItemId.QUICK_CLAW in INPUT_NEUTRAL_ITEMS


def test_end_of_turn_probability(tables, names):
    # The poisoned 1-HP endgame: the battler checked first faints first, so the player wins
    # exactly when Monferno's draw does not activate (Geodude is faster): 4/5.
    s = _state(tables, names, player_count=1, opp_count=1, mine=["Leer"], opp=["Leer"])
    s = s.with_status(0, 0, Status.POISON).with_status(1, 0, Status.POISON)
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    acts = A("Leer", "Leer", s, names)
    outcomes, probs, _drawn = enumerate_drawn(laws_of(tables, s, acts))
    wins = 0.0
    for i in range(len(probs)):
        row = Outcome(*(jnp.asarray(np.asarray(f)[i]) for f in outcomes))
        wins += float(probs[i]) * (int(_step(tables, s, acts, row).winner) == WINNER_PLAYER)
    assert wins == pytest.approx(0.8, abs=1e-12)


@pytest.mark.slow
def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _state(tables, names, holders=(0, 1), order="tie", qc=(1, 1))
    acts = A("Scratch", "Tackle", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


BRUTE_CASES = (
    # The holder on either side or both, every order and both stored bits.
    [
        (h, order, qc, None)
        for h in ((0,), (1,), (0, 1))
        for order in ("player", "opp", "tie")
        for qc in ((0, 0), (1, 1))
    ]
    # Against a Lagging Tail (either side).
    + [((0,), "opp", qc, "tail") for qc in ((0, 0), (1, 0))]
    + [((1,), "player", qc, "tail") for qc in ((0, 0), (0, 1))]
    # Order-conditioned reads: a flinch move by a slower holder (a sure flinch only when the
    # claw moves it first), Thunder Wave's full-para read, Protect's sure success.
    + [((0,), "opp", qc, "headbutt") for qc in ((0, 0), (1, 0))]
    + [((0,), "opp", qc, "thunder_wave") for qc in ((0, 0), (1, 0))]
    + [((0,), "opp", qc, "protect") for qc in ((0, 0), (1, 0))]
    # A bench holder (a replacement may send it in) and a 1-HP active that can faint.
    + [((), "opp", (0, 0), "bench"), ((), "opp", (0, 0), "bench_low")]
    # A poisoned 1-HP endgame (the end-of-turn order decides the winner).
    + [((0, 1), order, (0, 0), "endgame") for order in ("player", "tie")]
)


@pytest.mark.slow
@pytest.mark.parametrize("holders,order,qc,setup", BRUTE_CASES)
def test_gated_equals_bruteforce(tables, names, holders, order, qc, setup):
    mine, opp = ["Scratch", "Leer"], ["Tackle", "Leer"]
    if setup == "headbutt":
        mine = ["Headbutt"]
    if setup == "thunder_wave":
        mine = ["Thunder Wave"]
    if setup == "protect":
        mine = ["Protect"]
    count = {"endgame": 1}.get(setup, 6)
    s = _state(
        tables, names, order=order, qc=qc, holders=holders, mine=mine, opp=opp,
        player_count=count, opp_count=min(count, 3),
    )  # fmt: skip
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))  # no crits: a smaller brute force
    if setup == "tail":
        free = 1 if holders == (0,) else 0
        s = _claw(s, free, names, item="Lagging Tail")
    if setup == "protect":
        s = s._replace(protect_count=s.protect_count.at[0].set(1))
    if setup in ("bench", "bench_low"):
        s = _claw(s, 0, names, slot=2)
    if setup == "bench_low":
        s = s.replace_hp(0, 0, 1)
    if setup == "endgame":
        s = s.with_status(0, 0, Status.POISON).with_status(1, 0, Status.POISON)
        s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    acts = A(mine[0], opp[0], s, names)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width <= brute


@pytest.mark.parametrize("qc", [0, 1])
def test_kaizo_prinplup_bruteforce(kaizo_tables, kaizo_names, qc):
    # Rules §10.2: Prinplup (Quick Claw) BubbleBeam against Gible's Thunder Fang.
    t, n = kaizo_tables, kaizo_names
    from dataclasses import replace as replace_spec

    prinplup = replace_spec(_kaizo_spec(n, "Prinplup", "BubbleBeam", 20), item=n.item("Quick Claw"))
    s = make_state(t, [prinplup], [_kaizo_spec(n, "Gible", "Thunder Fang", 16)])
    s = s._replace(qc_roll=s.qc_roll.at[0].set(qc), lucky_chant=s.lucky_chant.at[:].set(5))
    width, brute = assert_equals_bruteforce(t, s, jnp.zeros(2, jnp.int32))
    assert width <= brute


# --- the root chance node --------------------------------------------------------------------------


def test_root_belief_splits_kept_sides(tables, names):
    s = _state(tables, names, holders=(), qc=(1, 1))
    ((only, p),) = root_belief(tables, s)
    assert p == 1.0 and [int(x) for x in only.qc_roll] == [0, 0]
    one = _state(tables, names, holders=(0,), qc=(1, 1))
    belief = root_belief(tables, one)
    assert [([int(x) for x in b.qc_roll], p) for b, p in belief] == [([0, 0], 0.8), ([1, 0], 0.2)]
    both = root_belief(tables, _state(tables, names, holders=(0, 1)))
    assert [[int(x) for x in b.qc_roll] for b, _ in both] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert [p for _, p in both] == pytest.approx([0.64, 0.16, 0.16, 0.04])
    assert math.fsum(p for _, p in both) == pytest.approx(1.0)


def test_sample_root_draws_one_in_five_for_kept_sides(tables, names):
    s = _state(tables, names, holders=(0,), qc=(1, 1))
    keys = jax.random.split(jax.random.key(7), 20000)
    roots = jax.jit(jax.vmap(sample_root, in_axes=(0, None, None)))(keys, tables, s)
    bits = np.asarray(roots.qc_roll)
    assert (bits[:, 1] == 0).all()
    assert abs(bits[:, 0].mean() - 0.2) < 5 * math.sqrt(0.16 / 20000)


def test_qc_kept_is_the_tight_rule(tables, names):
    s = _state(tables, names, holders=(0,))
    assert np.asarray(qc_kept(tables, s)).tolist() == [True, False]
    bench = _claw(_state(tables, names, holders=()), 1, names, slot=2)
    assert np.asarray(qc_kept(tables, bench)).tolist() == [False, False]  # no replacement due
    down = bench.replace_hp(1, 0, 0)
    assert np.asarray(qc_kept(tables, down)).tolist() == [False, True]


# --- coverage and prep ------------------------------------------------------------------------------


def test_quick_claw_is_supported(tables, names):
    from dataclasses import replace as replace_spec

    from tests.engine.helpers import roark_party, roark_trainer

    base = sample_party(names)
    mon = replace_spec(base[0], item=names.item("Quick Claw"))
    report = check_coverage(tables, names, [mon, *base[1:]], roark_trainer(), roark_party(names), 0)
    assert not [m for m in report.missing if "Quick Claw" in m]
    assert _item_refusal(tables, mon.item) is None
