"""Protect and Detect (K3b2 Task 6, ``EffectId.PROTECT``; docs/kaizo/k3b2-rules.md §2).

Decomp (external/pokeplatinum):

* ``res/battle/scripts/effects/effect_script_0111.s``: a DIRECT side effect
  ``CHECK_HP | TO_ATTACKER | PROTECT`` (Protect and Detect share effect 111).
* ``res/battle/scripts/subscripts/subscript_protect.s``: ``TryProtection _006``; on success
  the attack message and "X protected itself!", on failure ``MOVE_STATUS_FAILED``
  (``subscript_missed.s`` prints the attack message, then "But it failed!").
* ``BtlCmd_TryProtection`` (src/battle/battle_script.c:5149-5190):
  ``protectSuccessTurns = 0`` unless ``moveProtect`` is Protect, Detect or Endure
  (:5154-5158); ``moreBattlersThisTurn = waitingBattlers != 1`` (:5160-5165); success iff
  ``sProtectSuccessRate[n] >= BattleSystem_RandNext()`` and ``moreBattlersThisTurn``
  (:5167-5168; the rates :5128-5133 are 0xFFFF, 0x7FFF, 0x3FFF, 0x1FFF against a u16, so
  1, 1/2, 1/4, 1/8 exactly); success sets ``turnFlags.protecting`` and counts up to index 3
  (:5182-5184), failure sets the count to 0 (:5186).
* ``waitingBattlers`` counts the battlers whose action is not ``MOVE_END`` before each
  action (battle_controller_player.c:881-886): the battler acting last fails.
* ``UpdateMoveBuffers`` (battle_controller_player.c:3936-3942): ``moveProtect = moveCur``
  when the attack message was shown, else ``MOVE_NONE``. Every status-check disruption
  goes there without the message (bcp.c:2459-2711). A switch-in and a faint clear it
  (battle_lib.c:2120, :2198) with the whole ``moveEffectsData`` (:2099-2102, :2179-2182).
  The engine folds ``moveProtect`` into ``protect_count`` (rules §2(a), §12).
* ``CheckMoveHitOverrides`` (battle_controller_player.c:3001-3014), after the accuracy roll
  (:3253-3268): the defender protecting, ``MOVE_FLAG_CAN_PROTECT``, not a non-Ghost Curse,
  not the non-final turn of a multi-turn move (Bide's unleash is its final turn):
  ``Battler_UnlockMoveChoice`` and ``MOVE_STATUS_PROTECTED``.
* ``subscript_missed.s:18-63``: ``PROTECTED`` is tested before ``MISSED``; for Fake Out a set
  ``FAILED`` prints "But it failed!" instead (:20-27, :58-59).
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.game_constants import CONSTANTS
from nuzline.engine.enumerate import (
    _LAWS,
    INPUT_NEUTRAL_EFFECTS,
    NON_TABLE_DAMAGE_EFFECTS,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import (
    MOVE_DETECT,
    MOVE_ENDURE,
    MOVE_PROTECT,
    PROTECT_CHAIN_MOVES,
    AbilityId,
    EffectId,
    MoveFlag,
    Stat,
    Status,
)
from nuzline.engine.observe import observe
from nuzline.engine.outcome import PROTECT_RATES, Outcome, empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import replace, step, step_events
from nuzline.search.plan_tree import _FLAG_TEXT
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_enumerate import (
    _kaizo_spec,
    assert_equals_bruteforce,
    check_frequencies,
    children,
    laws_of,
    sampled,
)
from tools.effect_map import DECOMP_EFFECT_TO_ID

DECOMP = Path("external/pokeplatinum")
_SPE = 4  # stats[..., 0..4] are Atk..Spe
PLAYER_MOVES = ("Protect", "Detect", "Scratch", "Swords Dance")


def _state(tables, names, *, opp="Tackle", order="player", count=0, opp_count=0, mine=None):
    """Monferno (Protect, Detect, Scratch, Swords Dance, or ``mine``) against Geodude
    (``opp`` first, then the fixture moves); abilities none. ``order``: who is faster
    ("player", "opp") or "tie"; ``count`` / ``opp_count``: each side's ``protect_count``."""
    s = roark_state(
        tables,
        names,
        player_moves=list(mine or PLAYER_MOVES),
        opp_moves=[opp] if isinstance(opp, str) else list(opp),
    )
    s = _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    return s._replace(protect_count=jnp.asarray([count, opp_count], s.protect_count.dtype))


def _ability(s, side, ability):
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _o(protect=(1, 1), **fields) -> Outcome:
    """``empty_outcome`` with ``protect`` and any other field (whole arrays) replaced."""
    o = empty_outcome()._replace(protect=jnp.asarray(protect, jnp.int8))
    return o._replace(**{k: jnp.asarray(v, jnp.int8) for k, v in fields.items()})


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


def _hp(s, side):
    return int(s.hp[side, int(s.active[side])])


# --- mapping, data and decomp scripts -------------------------------------------------------


def test_effect_111_maps_protect_and_detect(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[111] == (EffectId.PROTECT, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for move in ("Protect", "Detect"):
            m = n.move(move)
            assert int(t.move_effect[m]) == EffectId.PROTECT, move
            # +3 in both games (rules §2 header), accuracy 0, the user's own range.
            assert int(t.move_priority[m]) == 3, move
            assert int(t.move_accuracy[m]) == 0 and int(t.move_power[m]) == 0
            assert not int(t.move_flags[m]) & MoveFlag.PROTECT
        # Endure (effect 116, BATTLE_EFFECT_SURVIVE_WITH_1_HP) stays unsupported.
        assert int(t.move_effect[n.move("Endure")]) == EffectId.UNSUPPORTED
    # Kaizo's Fake Out is +3 as well: against Protect, speed decides (rules §2(a)).
    assert int(kaizo_tables.move_priority[kaizo_names.move("Fake Out")]) == 3
    assert int(tables.move_priority[names.move("Fake Out")]) == 1


def test_chain_move_ids_are_game_constants(names):
    assert (MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE) == PROTECT_CHAIN_MOVES == (182, 197, 203)
    for move, name in zip(PROTECT_CHAIN_MOVES, ("Protect", "Detect", "Endure"), strict=True):
        assert names.move(name) == move
    listed = {(c.where, c.id, c.name) for c in CONSTANTS}
    for attr, move, name in (
        ("MOVE_PROTECT", 182, "Protect"),
        ("MOVE_DETECT", 197, "Detect"),
        ("MOVE_ENDURE", 203, "Endure"),
    ):
        assert (f"nuzline.engine.ids.{attr}", move, name) in listed


def _function(source: str, name: str) -> str:
    """The body of C function ``name`` (its definition, not a prototype)."""
    return re.search(rf"\b{name}\([^;{{]*\)\n\{{.*?\n\}}", source, re.DOTALL).group(0)


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_and_try_protection_match_the_engine():
    effects = DECOMP / "res/battle/scripts/effects"
    subs = DECOMP / "res/battle/scripts/subscripts"
    s111 = (effects / "effect_script_0111.s").read_text("utf-8")
    assert "MOVE_SIDE_EFFECT_CHECK_HP|MOVE_SIDE_EFFECT_TO_ATTACKER|MOVE_SUBSCRIPT_PTR_PROTECT" in (
        s111
    )
    sub = (subs / "subscript_protect.s").read_text("utf-8")
    assert "TryProtection _006" in sub and "MOVE_STATUS_FAILED" in sub
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8")
    table = re.search(r"sProtectSuccessRate\[\] = \{(.*?)\};", script, re.DOTALL).group(1)
    assert [int(x, 16) for x in re.findall(r"0x[0-9A-F]+", table)] == list(PROTECT_RATES)
    body = _function(script, "BtlCmd_TryProtection")
    # A u16 threshold comparison (exact, no %): success iff rate >= RandNext() and the user
    # is not the last battler to act; the draw happens first (C's left-to-right &&).
    assert (
        "sProtectSuccessRate[ATTACKING_MON.moveEffectsData.protectSuccessTurns] >= "
        "BattleSystem_RandNext(battleSys)\n        && moreBattlersThisTurn"
    ) in body
    assert "%" not in body
    assert "battleCtx->waitingBattlers == 1" in body
    for move in ("MOVE_PROTECT", "MOVE_DETECT", "MOVE_ENDURE"):
        assert f"moveProtect[battleCtx->attacker] != {move}" in body
    assert "protectSuccessTurns < NELEMS(sProtectSuccessRate) - 1" in body
    # protectSuccessTurns has no other writer than TryProtection outside the generic
    # BattleMon_Set/AddData, which no script calls for it.
    scripts = "".join(p.read_text("utf-8") for p in (DECOMP / "res/battle/scripts").rglob("*.s"))
    assert "SUCCESSFUL_PROTECT_TURNS" not in scripts
    # The blocking check reads the defender's turn flag and the move's flag.
    bcp = (DECOMP / "src/battle/battle_controller_player.c").read_text("utf-8")
    overrides = _function(bcp, "BattleControllerPlayer_CheckMoveHitOverrides")
    assert "turnFlags[defender].protecting" in overrides
    assert "MOVE_FLAG_CAN_PROTECT" in overrides
    assert "Battler_UnlockMoveChoice" in overrides
    buffers = _function(bcp, "BattleControllerPlayer_UpdateMoveBuffers")
    assert "moveProtect[battleCtx->attacker] = battleCtx->moveCur" in buffers
    assert "moveProtect[battleCtx->attacker] = MOVE_NONE" in buffers


def test_law_is_two_to_the_minus_n(tables, names):
    # u16 r <= rate: (rate + 1) / 65536 = 1, 1/2, 1/4, 1/8; the index is capped at 3.
    assert [(r + 1) / 65536 for r in PROTECT_RATES] == [1, 0.5, 0.25, 0.125]
    s = _state(tables, names)
    acts = A("Protect", "Tackle", s, names)
    for count in range(4):
        laws = laws_of(tables, s._replace(protect_count=s.protect_count.at[0].set(count)), acts)
        values, weights, den = _LAWS["protect"](laws, 0)
        assert dict(zip(values, (w / den for w in weights), strict=True)).get(1) == 2.0**-count
    # Sampled: fold_in(key, 18), the success frequency of each count.
    n = 4000
    keys = jax.random.split(jax.random.key(3), n)
    for count in range(4):
        st = s._replace(protect_count=s.protect_count.at[0].set(count))
        draws = jax.vmap(lambda k, st=st: sample_outcome(k, tables, st, acts).protect)(keys)
        p = 2.0**-count
        assert (
            abs(float(np.mean(np.asarray(draws)[:, 0])) - p) < 5 * math.sqrt(p * (1 - p) / n) + 1e-9
        )


# --- the turn: success, failure, the chain ------------------------------------------------


def test_protect_first_blocks_the_foe(tables, names):
    s = _state(tables, names)
    acts = A("Protect", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    assert _hp(out, 0) == _hp(s, 0)  # Tackle blocked
    assert int(out.protect_count[0]) == 1
    ev = _events(tables, s, acts, _o())
    assert ev["protect_ok"].tolist() == [1, 0] and ev["protect_failed"].tolist() == [0, 0]
    assert ev["protected"].tolist() == [0, 1] and ev["missed"].tolist() == [0, 0]
    assert ev["hits"][1] == 0
    # The foe's move still counts as used (attack message): moveHit and movePrevByBattler.
    assert int(out.last_move[1]) == names.move("Tackle")


def test_protect_moving_last_fails_whatever_the_roll(tables, names):
    # Vanilla: only another +3 move goes before Protect: the foe's own, when faster.
    s = _state(tables, names, opp="Protect", order="opp", count=1)
    acts = A("Protect", "Protect", s, names)
    for roll in (0, 1):
        out = assert_matches_reference(tables, s, acts, _o((roll, 1)))
        assert int(out.protect_count[0]) == 0 and int(out.protect_count[1]) == 1
        ev = _events(tables, s, acts, _o((roll, 1)))
        assert ev["protect_failed"].tolist() == [1, 0] and ev["protect_ok"].tolist() == [0, 1]
    # Behind a switch or a bag item (both go first) it is last too.
    s = _state(tables, names)
    switch = A("Protect", 4 + 1, s, names)
    out = assert_matches_reference(tables, s, switch, _o())
    assert int(out.protect_count[0]) == 0
    assert _events(tables, s, switch, _o())["protect_failed"][0] == 1
    item = A("Protect", 10, s, names)
    assert _events(tables, s, item, _o())["protect_failed"][0] == 1


def test_chain_counts_up_to_three_and_drops_on_a_failed_roll(tables, names):
    for count in range(4):
        s = _state(tables, names, count=count)
        acts = A("Detect", "Tackle", s, names)
        ok = assert_matches_reference(tables, s, acts, _o((1, 1)))
        assert int(ok.protect_count[0]) == min(count + 1, 3)
        assert _hp(ok, 0) == _hp(s, 0)
        failed = assert_matches_reference(tables, s, acts, _o((0, 1)))
        if count == 0:  # rate 0xFFFF: no u16 exceeds it, the roll cannot fail
            assert int(failed.protect_count[0]) == 1 and _hp(failed, 0) == _hp(s, 0)
        else:
            assert int(failed.protect_count[0]) == 0 and _hp(failed, 0) < _hp(s, 0)


def test_chain_resets_after_another_move_or_a_disruption(tables, names):
    s = _state(tables, names, count=2)
    # Another move with its attack message: moveProtect = that move.
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o())
    assert int(out.protect_count[0]) == 0
    # Protect and Detect keep it (both are in the family).
    out = assert_matches_reference(tables, s, A("Detect", "Tackle", s, names), _o())
    assert int(out.protect_count[0]) == 3
    # Full paralysis, a flinch, sleep: no attack message, moveProtect = MOVE_NONE.
    para = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    out = assert_matches_reference(
        tables, para, A("Protect", "Tackle", para, names), _o(full_para=[1, 0])
    )
    assert int(out.protect_count[0]) == 0
    asleep = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
    )
    out = assert_matches_reference(tables, asleep, A("Protect", "Tackle", asleep, names), _o())
    assert int(out.protect_count[0]) == 0
    # A taunted Protect (power 0) fails at the Taunt check: no message.
    taunted = s._replace(taunt=s.taunt.at[0].set(2))
    out = assert_matches_reference(tables, taunted, A(0, "Tackle", taunted, names), _o())
    assert int(out.protect_count[0]) == 0
    # A switch clears moveEffectsData; Roark's bag item leaves the user's chain alone.
    out = assert_matches_reference(tables, s, A(4 + 1, "Tackle", s, names), _o())
    assert int(out.protect_count[0]) == 0
    opp = _state(tables, names, opp_count=2)
    out = assert_matches_reference(tables, opp, A("Scratch", 10, opp, names), _o())
    assert int(out.protect_count[1]) == 2


def test_pursuit_leaves_the_pursuers_chain(tables, names):
    # BtlCmd_TryPursuit (battle_script.c:6884-6935) ends the pursuer's action without
    # UpdateMoveBuffers: its moveProtect, and so the folded chain, is untouched.
    s = _state(tables, names, opp="Pursuit", opp_count=1)
    out = assert_matches_reference(tables, s, A(4 + 1, "Pursuit", s, names), _o())
    assert int(out.protect_count[1]) == 1


def test_failed_protect_then_protect_is_a_fresh_chain(tables, names):
    """UNCERTAIN item 2 (rules §11): a failed Protect shows its attack message
    (subscript_missed.s:5), so ``moveProtect`` stays Protect and ``last_move`` agrees, but
    TryProtection's failure branch already wrote ``protectSuccessTurns = 0``
    (battle_script.c:5186). The next Protect moving first is certain (0xFFFF)."""
    s = _state(tables, names, opp=("Tackle", "Detect"))
    acts = A("Protect", "Tackle", s, names)
    t1 = assert_matches_reference(tables, s, acts, _o((1, 1)))  # success: count 1
    assert int(t1.protect_count[0]) == 1
    # Turn 2: the faster foe's Detect goes first, so Monferno's Protect moves last: fails.
    slow = t1._replace(stats=t1.stats.at[1, 0, _SPE].set(int(t1.stats[0, 0, _SPE]) + 1))
    detect = A("Protect", "Detect", slow, names)
    t2 = assert_matches_reference(tables, slow, detect, _o((1, 1)))
    assert int(t2.protect_count[0]) == 0
    assert int(t2.last_move[0]) == names.move("Protect")  # movePrevByBattler = Protect
    assert _events(tables, slow, detect, _o())["protect_failed"][0] == 1
    fast = t2._replace(stats=t2.stats.at[1, 0, _SPE].set(int(t2.stats[0, 0, _SPE]) - 1))
    before = _hp(fast, 0)
    t3 = assert_matches_reference(tables, fast, acts, _o((0, 1)))  # a "failed" roll
    assert int(t3.protect_count[0]) == 1 and _hp(t3, 0) == before
    # The same after a failed roll (count 1, roll 0) instead of moving last.
    r2 = assert_matches_reference(tables, t1, acts, _o((0, 1)))
    assert int(r2.protect_count[0]) == 0
    r3 = assert_matches_reference(tables, r2, acts, _o((0, 1)))
    assert int(r3.protect_count[0]) == 1


# --- what Protect blocks -------------------------------------------------------------------


@pytest.mark.parametrize(
    "move",
    ["Tackle", "Fury Swipes", "Ember", "Thunder Wave", "Take Down", "Absorb", "Leer", "Bite"],
)
def test_flagged_moves_are_blocked(tables, names, move):
    """A blocked move does nothing: no damage, no hits, no secondary, recoil or drain,
    no stage or status, whatever its rolls."""
    s = _state(tables, names, opp=move)
    s = s.replace_hp(1, 0, max(int(s.max_hp[1, 0]) // 2, 1))  # Absorb could heal
    assert int(tables.move_flags[names.move(move)]) & MoveFlag.PROTECT
    acts = A("Protect", move, s, names)
    o = _o(secondary=np.ones((2, 5)), crit=np.ones((2, 5)), n_hits=[5, 5])
    out = assert_matches_reference(tables, s, acts, o)
    for name in ("hp", "status", "stages", "flinched"):
        assert np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(s, name))), name
    ev = _events(tables, s, acts, o)
    assert ev["protected"][1] == 1 and ev["hits"][1] == 0 and not ev["crit"].any()


def test_unflagged_and_self_moves_go_through(tables, names):
    # Swords Dance (RANGE_USER, no CAN_PROTECT): the user is its own defender.
    s = _state(tables, names, opp="Swords Dance")
    out = assert_matches_reference(tables, s, A("Protect", "Swords Dance", s, names), _o())
    assert int(out.stages[1, Stat.ATK]) == 2
    # Stealth Rock (no CAN_PROTECT) lands on the protecting side.
    s = _state(tables, names, opp="Stealth Rock")
    out = assert_matches_reference(tables, s, A("Protect", "Stealth Rock", s, names), _o())
    assert int(out.stealth_rock[0]) != 0
    assert _events(tables, s, A("Protect", "Stealth Rock", s, names), _o())["protected"][1] == 0


def test_a_miss_into_protect_shows_protected(tables, names):
    # PROTECTED is tested before MISSED in subscript_missed.s (:18 vs :157).
    s = _state(tables, names)
    acts = A("Protect", "Tackle", s, names)
    ev = _events(tables, s, acts, _o(hit=np.zeros((2, 5))))
    assert ev["protected"][1] == 1 and ev["missed"][1] == 0


def test_selfdestruct_user_faints_into_protect(tables, names):
    # effect_script_0007.s sets the user's HP to 0 before TryMove: it faints, the target
    # is untouched.
    s = _state(tables, names, opp="Selfdestruct")
    out = assert_matches_reference(tables, s, A("Protect", "Selfdestruct", s, names), _o())
    assert _hp(out, 1) == 0 and _hp(out, 0) == _hp(s, 0)


def test_contact_ability_does_not_trigger(tables, names):
    s = _ability(_state(tables, names), 0, AbilityId.STATIC)
    o = _o(contact_proc=np.ones((2, 5)))
    out = assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), o)
    assert int(out.status[1, 0]) == Status.NONE


def test_a_blocked_rollout_unlocks(tables, names):
    # Battler_UnlockMoveChoice in CheckMoveHitOverrides: the Rollout lock ends.
    s = _state(tables, names, opp="Rollout")
    rolled = assert_matches_reference(tables, s, A("Scratch", "Rollout", s, names), _o())
    assert int(rolled.locked_move[1]) == names.move("Rollout")
    out = assert_matches_reference(tables, rolled, A("Protect", "Rollout", rolled, names), _o())
    assert int(out.locked_move[1]) == 0 and int(out.rollout_count[1]) == 0


def test_bide_unleash_is_blocked(tables, names):
    # Bide is multi-turn: blocked only on its unleash (SYSCTL_LAST_OF_MULTI_TURN, set by
    # subscript_bide_end.s), where its defender is the last hitter.
    s = _state(tables, names, opp="Bide")
    t1 = assert_matches_reference(tables, s, A("Scratch", "Bide", s, names), _o())
    assert int(t1.bide_turns[1]) == 2
    t2 = assert_matches_reference(tables, t1, A("Protect", "Bide", t1, names), _o())
    assert int(t2.bide_turns[1]) == 1  # a storing turn is not blocked (its own defender)
    t3 = assert_matches_reference(tables, t2, A("Scratch", "Bide", t2, names), _o())
    assert _hp(t3, 0) < _hp(t2, 0)  # unleashed
    fresh = t2._replace(protect_count=t2.protect_count.at[0].set(0))
    t3b = assert_matches_reference(tables, fresh, A("Protect", "Bide", fresh, names), _o())
    assert _hp(t3b, 0) == _hp(t2, 0) and int(t3b.bide_turns[1]) == 0
    assert _events(tables, fresh, A("Protect", "Bide", fresh, names), _o())["protected"][1] == 1


def test_present_heal_is_blocked(tables, names):
    s = _state(tables, names, opp="Present").replace_hp(0, 0, 5)
    out = assert_matches_reference(tables, s, A("Protect", "Present", s, names), _o(present=[0, 3]))
    assert _hp(out, 0) == 5


def test_fake_out_against_protect_kaizo(kaizo_tables, kaizo_names):
    """Both +3 in Kaizo: speed decides (rules §2(a))."""
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t,
        [_kaizo_spec(n, "Wormadam", "Protect", 20)],
        [_kaizo_spec(n, "Skitty", "Fake Out", 20)],
    )
    s = s._replace(protect_count=s.protect_count.at[0].set(1))
    spe = int(s.stats[0, 0, _SPE])
    acts = jnp.zeros(2, jnp.int32)
    # Protect first (count 1, success roll): Fake Out blocked on its first turn.
    fast = s._replace(stats=s.stats.at[1, 0, _SPE].set(spe - 1))
    out = assert_matches_reference(t, fast, acts, _o((1, 1)))
    assert _hp(out, 0) == _hp(s, 0) and int(out.protect_count[0]) == 2
    ev = _events(t, fast, acts, _o((1, 1)))
    assert ev["protected"][1] == 1 and ev["flinched"][0] == 0
    # Not Skitty's first turn: Fake Out failed, "But it failed!" (no "protected").
    late = fast._replace(turn=fast.turn + 3)
    ev = _events(t, late, acts, _o((1, 1)))
    assert ev["protected"][1] == 0 and ev["protect_ok"][0] == 1
    # Fake Out first: Wormadam flinches, never shows Protect, and its chain resets.
    slow = s._replace(stats=s.stats.at[1, 0, _SPE].set(spe + 1))
    out = assert_matches_reference(t, slow, acts, _o((1, 1), secondary=np.ones((2, 5))))
    assert _hp(out, 0) < _hp(s, 0) and int(out.protect_count[0]) == 0
    ev = _events(t, slow, acts, _o((1, 1), secondary=np.ones((2, 5))))
    assert ev["flinched"][0] == 1 and ev["protect_ok"][0] == 0 and ev["protect_failed"][0] == 0
    # The tie: either order (speed_tie).
    tie = s._replace(stats=s.stats.at[1, 0, _SPE].set(spe))
    for tie_roll, protected in ((0, 1), (1, 0)):
        ev = _events(t, tie, acts, _o((1, 1), speed_tie=tie_roll, secondary=np.ones((2, 5))))
        assert ev["protected"][1] == protected


def test_vanilla_protect_outspeeds_fake_out(tables, names):
    # Vanilla: Protect +3, Fake Out +1. A slower Protect user still goes first.
    s = _state(tables, names, opp="Fake Out", order="opp")
    acts = A("Protect", "Fake Out", s, names)
    out = assert_matches_reference(tables, s, acts, _o(secondary=np.ones((2, 5))))
    assert _hp(out, 0) == _hp(s, 0) and int(out.protect_count[0]) == 1


def test_faint_and_replacement_clear_the_chain(tables, names):
    # A faint clears moveEffectsData (battle_lib.c:2179-2182) and moveProtect (:2198).
    s = _state(tables, names, order="opp", count=2).replace_hp(0, 0, 1)
    acts = A("Scratch", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    assert _hp(out, 0) == 0 and int(out.protect_count[0]) == 0
    assert int(replace(tables, out, 0, 1).protect_count[0]) == 0


# --- observation and plan tree -------------------------------------------------------------


def test_observation_and_labels(tables, names):
    s = _state(tables, names, count=1)
    acts = A("Protect", "Leer", s, names)
    obs = []
    for roll in (0, 1):
        o = _o((roll, 1))
        after = step(tables, s, acts, o)
        obs.append(np.asarray(observe(tables, s, acts, o, after)))
    # Leer's stage drop would differ anyway; the flags differ too.
    assert (obs[0] != obs[1]).any()
    texts = dict(_FLAG_TEXT)
    assert texts["protect_ok"] == "protected itself"
    assert texts["protect_failed"] == "but it failed"
    assert texts["protected"] == "the foe protected itself"


def test_reference_agreement_over_draws(tables, names):
    """Both engines agree over protect x hit x crit x secondary x speed_tie, Protect or
    Detect against several foe moves, in every order and chain count."""
    for opp, order, count, mine in itertools.product(
        ("Tackle", "Fury Swipes", "Ember", "Protect", "Swords Dance"),
        ("player", "opp", "tie"),
        (0, 1, 3),
        ("Protect", "Detect"),
    ):
        s = _state(tables, names, opp=opp, order=order, count=count, opp_count=count)
        acts = A(mine, opp, s, names)
        for p0, p1, hit, tie in itertools.product((0, 1), (0, 1), (0, 1), (0, 1)):
            o = _o((p0, p1), hit=np.full((2, 5), hit), speed_tie=tie)
            assert_matches_reference(tables, s, acts, o)


# --- laws, relevance, enumeration ----------------------------------------------------------


def test_relevance(tables, names):
    # Both use Protect (+3): speed decides; the foe's chain is 0 (never relevant).
    acts_of = lambda s: A("Protect", "Protect", s, names)
    for count, order, want in (
        (0, "player", False),  # rate 1: a single value
        (1, "player", True),
        (1, "opp", False),  # moving last: it fails whatever the roll
        (1, "tie", True),  # the branch where it goes first
    ):
        s = _state(tables, names, opp="Protect", count=count, order=order)
        laws = laws_of(tables, s, acts_of(s))
        assert np.asarray(laws.relevant.protect).tolist() == [want, False], (count, order)
        loose = laws_of(tables, s, acts_of(s), loose=True)
        assert np.asarray(loose.relevant.protect).tolist() == [count > 0, False]
    s = _state(tables, names, opp="Protect", count=1, order="tie")
    branch = np.asarray(laws_of(tables, s, acts_of(s)).branch_relevant.protect)
    assert branch[:, 0].tolist() == [True, False]  # speed_tie 0: player first


def test_a_sure_protect_reads_none_of_the_foes_move(tables, names):
    # Count 0, moving first: the Protect is certain, so Tackle's hit, crit and roll and
    # the protecting side's contact rolls are never read.
    s = _ability(_state(tables, names), 0, AbilityId.STATIC)
    laws = laws_of(tables, s, A("Protect", "Tackle", s, names))
    _, probs, drawn = enumerate_drawn(laws)
    assert len(probs) == (2 if bool(laws.relevant.end_speed_tie) else 1)
    for name in ("hit", "crit", "roll", "contact_proc"):
        assert not np.asarray(getattr(drawn, name)).any(), name


def test_a_chained_protect_reads_the_foes_move_only_when_it_fails(tables, names):
    s = _state(tables, names, count=2)
    laws = laws_of(tables, s, A("Protect", "Tackle", s, names))
    outcomes, probs, drawn = enumerate_drawn(laws)
    protect = np.asarray(outcomes.protect)[:, 0]
    assert np.asarray(drawn.protect)[:, 0].all()
    for name in ("hit", "crit", "roll"):
        read = np.asarray(getattr(drawn, name))[:, 1, 0]
        assert not read[protect == 1].any() and read[protect == 0].any(), name
    assert (np.asarray(drawn.hit)[:, 1, 0] == (protect == 0)).all()
    assert math.fsum(probs[protect == 1]) == pytest.approx(0.25, abs=1e-15)


def test_the_foes_status_checks_are_still_read(tables, names):
    # A confused / paralyzed foe rolls before its move is blocked: those reads stay.
    s = _state(tables, names, count=0)
    s = s._replace(
        confusion=s.confusion.at[1].set(3),
        status=s.status.at[1, 0].set(Status.PARALYSIS),
    )
    laws = laws_of(tables, s, A("Protect", "Tackle", s, names))
    _, _, drawn = enumerate_drawn(laws)
    assert np.asarray(drawn.confusion_hit)[:, 1].any()
    assert np.asarray(drawn.full_para)[:, 1].any()
    assert np.asarray(drawn.roll)[:, 1, 0].any()  # the confusion self-hit's roll


def test_damage_classes_classify_protect():
    assert EffectId.PROTECT in INPUT_NEUTRAL_EFFECTS
    assert EffectId.PROTECT in NON_TABLE_DAMAGE_EFFECTS


def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _state(tables, names, count=1)
    acts = A("Protect", "Tackle", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


BRUTE_CASES = (
    [
        (mine, opp, order, count, extra)
        for mine in ("Protect", "Detect")
        for opp in ("Tackle", "Double Kick", "Ember", "Bite", "Fake Out", "Selfdestruct", "Protect")
        for order in ("player", "opp", "tie")
        for count, extra in ((0, None), (1, None), (3, None))
        if not (mine == "Detect" and count == 3)
    ]
    + [
        ("Protect", "Tackle", order, 1, extra)
        for order in ("player", "tie")
        for extra in ("confused", "paralyzed", "static", "sash")
    ]
    + [("Protect", "Present", order, 1, None) for order in ("player", "tie")]
    + [("Protect", "Thunder Wave", order, 2, None) for order in ("player", "opp", "tie")]
    + [
        ("Protect", "Swords Dance", "tie", 1, None),
    ]
)


@pytest.mark.parametrize("mine,opp,order,count,extra", BRUTE_CASES)
def test_gated_equals_bruteforce(tables, names, mine, opp, order, count, extra):
    s = _state(tables, names, opp=opp, order=order, count=count, opp_count=count)
    if extra == "confused":
        s = s._replace(confusion=s.confusion.at[1].set(3))
    elif extra == "paralyzed":
        s = s._replace(status=s.status.at[1, 0].set(Status.PARALYSIS))
    elif extra == "static":
        s = _ability(s, 0, AbilityId.STATIC)
    elif extra == "sash":
        s = s._replace(item=s.item.at[0, 0].set(names.item("Focus Sash")))
    width, brute = assert_equals_bruteforce(tables, s, A(mine, opp, s, names))
    # A damaging foe move behind a Protect that goes first (+3) skips reads.
    if int(tables.move_power[names.move(opp)]) > 0:
        assert width < brute, (width, brute)


@pytest.mark.parametrize("order", ["protect", "fake_out", "tie"])
def test_kaizo_fake_out_vs_protect_bruteforce(kaizo_tables, kaizo_names, order):
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t,
        [_kaizo_spec(n, "Wormadam", "Protect", 20)],
        [_kaizo_spec(n, "Skitty", "Fake Out", 20)],
    )
    s = s._replace(protect_count=s.protect_count.at[0].set(1))
    spe = int(s.stats[0, 0, _SPE])
    foe = {"protect": spe - 1, "fake_out": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    width, brute = assert_equals_bruteforce(t, s, jnp.zeros(2, jnp.int32))
    assert width <= brute
    assert math.isclose(math.fsum(enumerate_outcomes(laws_of(t, s, jnp.zeros(2, jnp.int32)))[1]), 1)
