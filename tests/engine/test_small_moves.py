"""Recover, Fake Out, AncientPower, Curse and the fixed two-hit family (survey section 6).

* Recover (``effect_script_0032.s``): ``subscript_recover_half_max_hp.s`` heals
  ``DivideVarByValue(maxHP, 2)`` (``BattleSystem_Divide``); ``subscript_recover_hp.s:6``
  stops at full HP ("HP is full!").
* Fake Out (``effect_script_0158.s``): fails unless ``fakeOutTurnNumber == totalTurns``; else
  an indirect TO_DEFENDER flinch at the move's chance through ``subscript_flinch_mon.s``
  (target not moved yet, Inner Focus, whose message names the ability for Fake Out).
* AncientPower (``effect_script_0140.s``): an indirect TO_ATTACKER effect at the move's chance,
  ``subscript_boost_all_stats.s``: Atk, Def, Spe, SpA, SpD +1, each capped on its own; the
  recipient is the user, so a fainted target does not stop it (battle_lib.c:1570-1586).
* Curse, non-Ghost user (``effect_script_0109.s``, ``subscript_curse_normal.s``): Spe -1, Atk +1,
  Def +1, each through ``BtlCmd_ChangeStatStage`` on its own (a capped one only prints "won't
  go higher/lower"). A Ghost-type user is refused by coverage.
* Double Kick (vanilla) / DoubleSlap (Kaizo), ``effect_script_0044.s``: ``SetMultiHit 2,
  SYSCTL_MULTI_HIT_MOVE``: exactly two hits, one accuracy check.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.coverage import check_coverage
from nuzline.engine.damage import damage_table
from nuzline.engine.effects import apply_move_effect
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.ids import EffectId, Stat
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.step import step, step_events
from nuzline.prep.pool import Candidate, check_pool
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    roark_party,
    roark_state,
    roark_trainer,
    sample_party,
)

PROC = empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8))
MISS = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
_step = jax.jit(step)
_events = jax.jit(step_events)

_RAISED = (Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD)


def _turn(s, turn, fake_out_turn):
    return s._replace(
        turn=jnp.asarray(turn, s.turn.dtype),
        fake_out_turn=jnp.full_like(s.fake_out_turn, fake_out_turn),
    )


def _stages(s, side=0):
    return [int(x) for x in np.asarray(s.stages[side])]


# --- Recover ------------------------------------------------------------------------


@pytest.mark.slow
def test_recover_half_and_fails_full(tables, names):
    s = roark_state(tables, names, player_moves=["Recover"])
    max_hp = int(s.max_hp[0, 0])
    acts = A("Recover", "Leer", s, names)
    low = s.replace_hp(0, 0, 3)
    out = assert_matches_reference(tables, low, acts, empty_outcome())
    assert int(out.hp[0, 0]) == min(3 + max(max_hp // 2, 1), max_hp)
    # Near full: the heal is capped at max HP.
    near = s.replace_hp(0, 0, max_hp - 1)
    assert int(assert_matches_reference(tables, near, acts, empty_outcome()).hp[0, 0]) == max_hp
    # Full HP: "HP is full!" (subscript_recover_hp.s:_036), nothing changes but the PP.
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.hp[0, 0]) == max_hp
    slot = names.move_slot(s, 0, "Recover")
    assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot]) - 1


# --- Fake Out -----------------------------------------------------------------------


@pytest.mark.slow
def test_fake_out_only_first_turn(tables, names):
    s = roark_state(tables, names, player_moves=["Fake Out"])
    acts = A("Fake Out", "Tackle", s, names)
    # The lead's first turn (fakeOutTurnNumber 0 == totalTurns 0): a hit, and the foe flinches.
    out = assert_matches_reference(tables, s, acts, PROC)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) == int(s.hp[0, 0])
    # A later turn: "But it failed!": no damage, no flinch, the foe's Tackle lands; the move
    # still counts as used (movePrevByBattler, PP spent).
    later = _turn(s, 3, 0)
    out = assert_matches_reference(tables, later, acts, PROC)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])
    assert int(out.last_move[0]) == names.move("Fake Out")
    slot = names.move_slot(s, 0, "Fake Out")
    assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot]) - 1
    # Switched in last turn (fakeOutTurnNumber = totalTurns + 1 then): it works again.
    out = assert_matches_reference(tables, _turn(s, 3, 3), acts, PROC)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) == int(s.hp[0, 0])
    # The failure shows "But it failed!", not a miss, even when the accuracy roll missed.
    _, events = _events(tables, later, acts, MISS)
    assert int(events["missed"][0]) == 0 and int(events["hits"][0]) == 0
    _, events = _events(tables, s, acts, MISS)
    assert int(events["missed"][0]) == 1


def test_fake_out_flinch_follows_the_flinch_rules(tables, names):
    # Inner Focus (Zubat) blocks it and, against Fake Out, names the ability
    # (subscript_flinch_mon.s:18-23 -> BattleAI_SetAbility).
    s = roark_state(tables, names, player_species="Zubat", player_moves=["Leer"],
                    opp_moves=["Fake Out"])  # fmt: skip
    s = s._replace(ai_ability_known=jnp.zeros_like(s.ai_ability_known))
    out = assert_matches_reference(tables, s, A("Leer", "Fake Out", s, names), PROC)
    assert int(out.stages[1, Stat.DEF]) == -1  # Zubat moved
    assert int(out.ai_ability_known[0]) != 0
    # A Headbutt flinch blocked by Inner Focus names nothing (only Fake Out does).
    h = roark_state(tables, names, player_species="Zubat", player_moves=["Leer"],
                    opp_moves=["Headbutt"], opp_faster=True)  # fmt: skip
    h = h._replace(ai_ability_known=jnp.zeros_like(h.ai_ability_known))
    out = assert_matches_reference(tables, h, A("Leer", "Headbutt", h, names), PROC)
    assert int(out.ai_ability_known[0]) == 0
    # Without Inner Focus the target flinches on the roll (chance 100: the law's only value).
    m = roark_state(tables, names, player_moves=["Leer"], opp_moves=["Fake Out"])
    acts = A("Leer", "Fake Out", m, names)
    assert int(assert_matches_reference(tables, m, acts, PROC).stages[1, Stat.DEF]) == 0
    out = assert_matches_reference(tables, m, acts, empty_outcome())
    assert int(out.stages[1, Stat.DEF]) == -1
    # A target that already moved does not flinch (IfMovedThisTurn).
    fake_out = names.move("Fake Out")
    for moved, flinched in ((False, 1), (True, 0)):
        hit = apply_move_effect(tables, m, 1, fake_out, PROC, 0, connected=True,
                                target_moved=moved)  # fmt: skip
        assert int(hit.flinched[0]) == flinched


# --- AncientPower ---------------------------------------------------------------------


def test_ancientpower_all_stats_on_roll(tables, names):
    s = roark_state(tables, names, player_moves=["AncientPower"])
    acts = A("AncientPower", "Stealth Rock", s, names)  # nothing on the user's stages
    out = assert_matches_reference(tables, s, acts, PROC)
    for stat in _RAISED:
        assert int(out.stages[0, stat]) == 1, stat
    assert int(out.stages[0, Stat.ACC]) == 0 and int(out.stages[0, Stat.EVA]) == 0
    # No roll: no boost.
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert all(int(out.stages[0, stat]) == 0 for stat in _RAISED)
    # A capped stat stays at +6; the others still rise.
    capped = s.with_stages(0, atk=6, spd=6)
    out = assert_matches_reference(tables, capped, acts, PROC)
    assert [int(out.stages[0, x]) for x in _RAISED] == [6, 1, 1, 1, 6]
    # The recipient is the user: a KO on the target does not stop it.
    ko = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, ko, acts, PROC)
    assert int(out.hp[1, 0]) == 0 and int(out.stages[0, Stat.SPA]) == 1
    # A miss raises nothing.
    out = assert_matches_reference(tables, s, acts, PROC._replace(hit=MISS.hit))
    assert all(int(out.stages[0, stat]) == 0 for stat in _RAISED)


# --- Curse ----------------------------------------------------------------------------


def test_curse_non_ghost_order_and_caps(tables, names):
    s = roark_state(tables, names, player_moves=["Curse"])
    acts = A("Curse", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    st = _stages(out)
    assert (st[Stat.SPE], st[Stat.ATK], st[Stat.DEF]) == (-1, 1, 1 - 1)  # Leer: Def -1
    # Each stage is changed on its own: a capped one is skipped, the rest still move.
    spe_min = s.with_stages(0, spe=-6)
    st = _stages(assert_matches_reference(tables, spe_min, acts, empty_outcome()))
    assert (st[Stat.SPE], st[Stat.ATK]) == (-6, 1)
    all_capped = s.with_stages(0, spe=-6, atk=6, def_=6)
    out = assert_matches_reference(tables, all_capped, A("Curse", "Tackle", s, names),
                                   empty_outcome())  # fmt: skip
    st = _stages(out)
    assert (st[Stat.SPE], st[Stat.ATK], st[Stat.DEF]) == (-6, 6, 6)
    # A non-Ghost Curse targets its user (MustSelectTarget, bcp.c:4344-4346): no moveHit
    # on the foe.
    assert int(out.move_hit[1]) == 0


def test_curse_ghost_user_supported(tables, names):
    # E2 batch 8 (B8-5) modelled a Ghost user's Curse: coverage no longer refuses it.
    curse = names.move("Curse")
    gastly = names.species("Gastly")
    base = sample_party(names)
    ghost = replace(base[0], species=gastly, form=0, ability_slot=0, moves=(curse, 0, 0, 0))
    report = check_coverage(tables, names, [ghost, *base[1:]], roark_trainer(),
                            roark_party(names), 0)  # fmt: skip
    assert not [ln for ln in report.missing if "Curse" in ln]
    plain = replace(base[0], moves=(curse, 0, 0, 0))
    report = check_coverage(tables, names, [plain, *base[1:]], roark_trainer(),
                            roark_party(names), 0)  # fmt: skip
    assert not [ln for ln in report.missing if "Curse" in ln]
    # The pool: a Ghost candidate that knows Curse or could relearn it.
    knows = Candidate(("party", 0, 0), "P", ghost, 0, (), ())
    could = Candidate(("party", 0, 1), "Q", replace(ghost, moves=(0, 0, 0, 0)), 0, (curse,), ())
    missing = check_pool(tables, names, [knows, could], ()).missing
    assert not [ln for ln in missing if "Curse" in ln]
    assert not [ln for ln in check_pool(tables, names, [Candidate(("party", 0, 0), "R", plain,
                                                                  0, (), ())], ()).missing
                if "Curse" in ln]  # fmt: skip


# --- the fixed two-hit family ----------------------------------------------------------


def test_kaizo_doubleslap_is_the_fixed_two_hit(kaizo_tables, kaizo_names):
    slap = kaizo_names.move("DoubleSlap")  # Kaizo: effect 44 HIT_TWICE, 30 power, 90 accuracy
    assert int(kaizo_tables.move_effect[slap]) == EffectId.MULTI_HIT
    assert [int(x) for x in kaizo_tables.move_param[slap]] == [2, 0, 0]


@pytest.mark.slow
def test_doubleslap_two_hits_one_accuracy(tables, names):
    # Vanilla's effect 44 users include Double Kick (vanilla DoubleSlap is the 2-5
    # MULTI_HIT); Kaizo's DoubleSlap is effect 44 (test_kaizo_doubleslap_is_the_fixed_two_hit).
    kick = names.move("Double Kick")
    assert int(tables.move_effect[kick]) == EffectId.MULTI_HIT
    assert [int(x) for x in tables.move_param[kick]] == [2, 0, 0]
    assert [int(x) for x in tables.move_param[names.move("Fury Swipes")]] == [0, 0, 0]

    s = roark_state(tables, names, player_moves=["Double Kick"])
    acts = A("Double Kick", "Leer", s, names)
    one = int(damage_table(tables, s, 0, kick, False)[15])
    for n in (2, MAX_HITS):  # the drawn 2-5 count is not read
        o = empty_outcome()._replace(n_hits=jnp.array([n, 2], jnp.int8))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == 2 * one
        _, events = _events(tables, s, acts, o)
        assert int(events["hits"][0]) == 2
    # One accuracy check: only hit[0] is read.
    first_only = empty_outcome()._replace(
        hit=jnp.array([[1] + [0] * (MAX_HITS - 1), [1] * MAX_HITS], jnp.int8)
    )
    out = assert_matches_reference(tables, s, acts, first_only)
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == 2 * one
    out = assert_matches_reference(tables, s, acts, MISS)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    # Enumeration: no hit-count draw, crit and roll for the two hits only.
    rel = outcome_laws(tables, s, acts).relevant
    assert not bool(rel.n_hits[0])
    # MAX_HITS is 6 (Beat Up, B13-9): every slot after the second is unread.
    assert np.asarray(rel.crit[0]).tolist() == [True, True] + [False] * (MAX_HITS - 2)
    fury = roark_state(tables, names, player_moves=["Fury Swipes"])
    rel = outcome_laws(tables, fury, A("Fury Swipes", "Leer", fury, names)).relevant
    # SetMultiHit 0 (effect_script_0029.s:5) draws 2-5 hits (battle_script.c:2641-2650): five
    # crit slots read; the sixth is Beat Up's only (BtlCmd_BeatUp, battle_script.c:6173-6247).
    assert bool(rel.n_hits[0])
    assert np.asarray(rel.crit[0]).tolist() == [True] * 5 + [False] * (MAX_HITS - 5)


# --- both engines ----------------------------------------------------------------------


def test_small_moves_match_reference(tables, names):
    moves = ["Recover", "Fake Out", "AncientPower", "Curse"]
    s = roark_state(tables, names, player_moves=moves, opp_moves=["Fake Out", "Recover",
                                                                  "Curse", "Double Kick"])  # fmt: skip
    hurt = s.replace_hp(0, 0, 5).replace_hp(1, 0, 7)
    states = [s, hurt, _turn(hurt, 2, 0), _turn(hurt, 2, 2),
              hurt.with_stages(0, spe=-6, atk=6, def_=6, spa=5)]  # fmt: skip
    outcomes = [empty_outcome(), PROC, MISS,
                PROC._replace(n_hits=jnp.array([5, 4], jnp.int8))]  # fmt: skip
    for state in states:
        for p in moves:
            for o in ("Fake Out", "Recover", "Curse", "Double Kick"):
                for outcome in outcomes:
                    assert_matches_reference(tables, state, A(p, o, state, names), outcome)
