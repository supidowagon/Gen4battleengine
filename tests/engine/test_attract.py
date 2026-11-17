"""Attract and Cute Charm (K3b2 Task 7, ``EffectId.ATTRACT``, ``AbilityId.CUTE_CHARM``;
docs/kaizo/k3b2-rules.md §4).

Decomp (external/pokeplatinum):

* ``res/battle/scripts/effects/effect_script_0120.s``: a DIRECT side effect
  ``TO_DEFENDER | INFATUATE`` (no ON_HIT), ``msgBattlerTemp = attacker``.
  ``BattleSystem_TriggerPrimaryEffect`` (src/battle/battle_lib.c:1481-1510) runs it only on a
  living defender and without ``MOVE_STATUS_NO_EFFECTS`` (a miss, Protect).
* ``res/battle/scripts/subscripts/subscript_infatuate.s``: Oblivious (refused: unsupported),
  ``MISSED|SEMI_INVULNERABLE``, then ``TryAttract`` (fail: ``MOVE_STATUS_FAILED`` for a move,
  silent for an ability, :60-65).
* ``BtlCmd_TryAttract`` (src/battle/battle_script.c:5727-5742): fails on equal genders, a
  target with any ATTRACT bit, or either side genderless; else the target gets
  ``FlagIndex(source) << VOLATILE_CONDITION_ATTRACT_SHIFT``: tied to the source battler.
* ``CHECK_STATUS_STATE_ATTRACT`` (src/battle/battle_controller_player.c:2669-2692), after
  CONFUSION (:2614) and PARALYSIS (:2652), before BIDE (:2694): ``RandNext() & 1`` nonzero
  shows "in love" and goes on; zero runs ``subscript_immobilized_by_love.s`` (UnlockMoveChoice)
  and ``commandNext = UPDATE_MOVE_BUFFERS``: no attack message.
* Cute Charm (``BattleSystem_TriggerAbilityOnHit``, battle_lib.c:4339-4355): a living attacker
  without ATTRACT, no NO_EFFECTS, damage taken, contact, a living defender, then
  ``RandNext() % 10 < 3`` (the last ``&&`` operand) -> subscript_infatuate as an ability side
  effect.
* Clearing: ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2088-2092) and
  ``BattleSystem_CleanupFaintedMon`` (:2168-2177) clear every ATTRACT bit sourced from the
  leaving battler; its own ``statusVolatile`` is reset (:2070, :2165).
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import replace as replace_spec
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.coverage import check_coverage
from nuzline.engine.enumerate import (
    _LAWS,
    INPUT_NEUTRAL_ABILITIES,
    INPUT_NEUTRAL_EFFECTS,
    NON_TABLE_DAMAGE_EFFECTS,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, Status
from nuzline.engine.observe import observe
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import MAX_HITS, Outcome, empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import _source_leaves, replace, step, step_events
from nuzline.prep.pool import Candidate, check_pool
from nuzline.prep.problem import _ability_refusal, _item_refusal
from nuzline.search.plan_tree import _FLAG_TEXT
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    roark_party,
    roark_state,
    roark_trainer,
    sample_party,
)
from tests.engine.test_enumerate import (
    _kaizo_spec,
    assert_equals_bruteforce,
    check_frequencies,
    children,
    laws_of,
    sampled,
)
from tools.effect_map import DECOMP_ABILITY_TO_ID, DECOMP_EFFECT_NAMES, DECOMP_EFFECT_TO_ID

DECOMP = Path("external/pokeplatinum")
_SPE = 4  # stats[..., 0..4] are Atk..Spe
MALE, FEMALE, NONE = 0, 1, 2
SWITCH = MOVE_ACTIONS + 1  # to party slot 1
PLAYER_MOVES = ("Attract", "Scratch", "Ember", "Double Kick")


def _state(
    tables,
    names,
    *,
    opp="Tackle",
    order="player",
    genders=(MALE, FEMALE),
    mine=None,
    infatuated=(0, 0),
    abilities=(AbilityId.NONE, AbilityId.NONE),
):
    """Monferno (Attract, Scratch, Ember, Double Kick, or ``mine``) against Geodude (``opp``
    first, then the fixture moves). ``order``: who is faster ("player", "opp", "tie");
    ``genders`` of the two leads; ``infatuated`` at the start; ``abilities`` of the leads."""
    s = roark_state(
        tables,
        names,
        player_moves=list(mine or PLAYER_MOVES),
        opp_moves=[opp]
        if isinstance(opp, str)
        else ["Tackle"]
        if isinstance(opp, int)
        else list(opp),
    )
    s = s._replace(
        ability=s.ability.at[0, 0].set(abilities[0]).at[1, 0].set(abilities[1]),
        gender=s.gender.at[0, 0].set(genders[0]).at[1, 0].set(genders[1]),
        infatuated=jnp.asarray(infatuated, s.infatuated.dtype),
    )
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    return s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))


def _o(infatuation=(0, 0), **fields) -> Outcome:
    o = empty_outcome()._replace(infatuation=jnp.asarray(infatuation, jnp.int8))
    return o._replace(**{k: jnp.asarray(v, jnp.int8) for k, v in fields.items()})


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


def _hp(s, side):
    return int(s.hp[side, int(s.active[side])])


def _inf(s):
    return [int(x) for x in s.infatuated]


# --- mapping, data and decomp ----------------------------------------------------------------


def test_effect_120_maps_attract(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[120] == (EffectId.ATTRACT, (0, 0, 0))
    assert DECOMP_EFFECT_NAMES[120] == "BATTLE_EFFECT_INFATUATE"
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        m = n.move("Attract")
        assert int(t.move_effect[m]) == EffectId.ATTRACT
        assert int(t.move_power[m]) == 0 and int(t.move_accuracy[m]) == 100
        flags = int(t.move_flags[m])
        assert flags & MoveFlag.PROTECT and not flags & MoveFlag.CONTACT


def test_cute_charm_maps_and_oblivious_stays_unsupported(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_ABILITY_TO_ID["Cute Charm"] == AbilityId.CUTE_CHARM
    assert "Oblivious" not in DECOMP_ABILITY_TO_ID
    for t, n, charmers in (
        (tables, names, ("Skitty", "Clefairy")),
        (kaizo_tables, kaizo_names, ("Skitty",)),  # Kaizo's Clefairy has other abilities
    ):
        for species in charmers:
            slots = n.species_abilities(n.species(species))
            k = [n.ability_names[a] for a in slots].index("Cute Charm")
            assert int(t.species_abilities[n.species(species), k]) == AbilityId.CUTE_CHARM
        # Every Oblivious holder (Slowpoke, Wailmer, ...) maps to UNSUPPORTED.
        table = np.asarray(t.species_abilities_decomp)
        oblivious = n.ability_names.index("Oblivious")
        rows, slots = np.nonzero(table == oblivious)
        assert rows.size > 0 or t is kaizo_tables
        assert (np.asarray(t.species_abilities)[rows, slots] == AbilityId.UNSUPPORTED).all()
        # Mental Herb (HOLD_EFFECT_HEAL_INFATUATION) and Destiny Knot (RECIPROCATE_INFAT)
        # would touch infatuation: unsupported, so coverage refuses them.
        for item in ("Mental Herb", "Destiny Knot"):
            assert int(t.item_effect[n.item(item)]) == ItemId.UNSUPPORTED


def _function(source: str, name: str) -> str:
    return re.search(rf"\b{name}\([^;{{]*\)\n\{{.*?\n\}}", source, re.DOTALL).group(0)


def _case(source: str, label: str) -> str:
    return re.search(rf"case {label}:(.*?)\n\s*break;", source, re.DOTALL).group(1)


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_match_the_engine():
    effects = DECOMP / "res/battle/scripts/effects"
    subs = DECOMP / "res/battle/scripts/subscripts"
    s120 = (effects / "effect_script_0120.s").read_text("utf-8")
    assert "MOVE_SIDE_EFFECT_TO_DEFENDER|MOVE_SUBSCRIPT_PTR_INFATUATE" in s120
    assert "ON_HIT" not in s120
    sub = (subs / "subscript_infatuate.s").read_text("utf-8")
    assert "ABILITY_OBLIVIOUS" in sub and "TryAttract _108" in sub
    immobilized = (subs / "subscript_immobilized_by_love.s").read_text("utf-8")
    assert "UnlockMoveChoice BTLSCR_ATTACKER" in immobilized
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8")
    body = _function(script, "BtlCmd_TryAttract")
    for cond in (
        (
            "battleMons[battleCtx->msgBattlerTemp].gender == "
            "battleCtx->battleMons[battleCtx->sideEffectMon].gender"
        ),
        "battleMons[battleCtx->sideEffectMon].statusVolatile & VOLATILE_CONDITION_ATTRACT",
        "battleMons[battleCtx->msgBattlerTemp].gender == GENDER_NONE",
        "battleMons[battleCtx->sideEffectMon].gender == GENDER_NONE",
        "FlagIndex(battleCtx->msgBattlerTemp) << VOLATILE_CONDITION_ATTRACT_SHIFT",
    ):
        assert cond in body, cond
    bcp = (DECOMP / "src/battle/battle_controller_player.c").read_text("utf-8")
    states = re.search(r"enum CheckStatusState \{(.*?)\};", bcp, re.DOTALL).group(1)
    order = re.findall(r"CHECK_STATUS_STATE_(\w+)", states)
    assert order.index("CONFUSION") < order.index("PARALYSIS") < order.index("ATTRACT")
    assert order.index("ATTRACT") + 1 == order.index("BIDE")
    assert order.index("FLINCH") < order.index("DISABLE") < order.index("TAUNT")
    assert order.index("TAUNT") < order.index("CONFUSION")
    attract = _case(bcp, "CHECK_STATUS_STATE_ATTRACT")
    # The law: RandNext() & 1, exact (no %); 0 is the immobilization.
    assert "BattleSystem_RandNext(battleSys) & 1" in attract and "%" not in attract
    assert "subscript_immobilized_by_love" in attract
    assert "BATTLE_CONTROL_UPDATE_MOVE_BUFFERS" in attract
    lib = (DECOMP / "src/battle/battle_lib.c").read_text("utf-8")
    cute = _case(lib, "ABILITY_CUTE_CHARM")
    terms = [
        "ATTACKING_MON.curHP",
        "(ATTACKING_MON.statusVolatile & VOLATILE_CONDITION_ATTRACT) == FALSE",
        "MOVE_STATUS_NO_EFFECTS",
        "physicalDamageTaken",
        "MOVE_FLAG_MAKES_CONTACT",
        "DEFENDING_MON.curHP",
        "BattleSystem_RandNext(battleSys) % 10 < 3",
    ]
    where = [cute.index(term) for term in terms]
    assert where == sorted(where)  # the roll is the last operand
    assert "subscript_infatuate" in cute
    for fn in ("BattleSystem_UpdateAfterSwitch", "BattleSystem_CleanupFaintedMon"):
        assert "FlagIndex(battler) << VOLATILE_CONDITION_ATTRACT_SHIFT" in _function(lib, fn)


@pytest.mark.slow
def test_law_is_a_fair_coin(tables, names):
    s = _state(tables, names, infatuated=(1, 0))
    acts = A("Scratch", "Tackle", s, names)
    laws = laws_of(tables, s, acts)
    values, weights, den = _LAWS["infatuation"](laws, 0)
    assert dict(zip(values, (w / den for w in weights), strict=True)) == {0: 0.5, 1: 0.5}
    n = 4000
    keys = jax.random.split(jax.random.key(7), n)
    draws = jax.vmap(lambda k: sample_outcome(k, tables, s, acts).infatuation)(keys)
    mean = float(np.mean(np.asarray(draws)[:, 0]))
    assert abs(mean - 0.5) < 5 * math.sqrt(0.25 / n)


# --- Attract -----------------------------------------------------------------------------------


@pytest.mark.slow
def test_attract_infatuates_an_opposite_gender_foe(tables, names):
    for genders in ((MALE, FEMALE), (FEMALE, MALE)):
        s = _state(tables, names, genders=genders)
        out = assert_matches_reference(tables, s, A("Attract", "Tackle", s, names), _o())
        assert _inf(out) == [0, 1]
        assert int(out.last_move[0]) == names.move("Attract")


@pytest.mark.parametrize(
    "case", ["same_gender", "genderless_user", "genderless_target", "already", "miss", "protect"]
)
def test_attract_fails(tables, names, case):
    genders = {
        "same_gender": (MALE, MALE),
        "genderless_user": (NONE, FEMALE),
        "genderless_target": (MALE, NONE),
    }.get(case, (MALE, FEMALE))
    opp = "Protect" if case == "protect" else "Tackle"
    s = _state(tables, names, genders=genders, opp=opp, infatuated=(0, int(case == "already")))
    o = _o(hit=np.zeros((2, MAX_HITS)) if case == "miss" else np.ones((2, MAX_HITS)))
    out = assert_matches_reference(tables, s, A("Attract", opp, s, names), o)
    assert _inf(out) == [0, int(case == "already")]


def test_attract_into_a_fainted_foe_has_no_target(tables, names):
    # The foe faints first (its own Selfdestruct): the no-target check stops Attract.
    s = _state(tables, names, opp="Selfdestruct", order="opp")
    out = assert_matches_reference(tables, s, A("Attract", "Selfdestruct", s, names), _o())
    assert _inf(out) == [0, 0]


# --- the status check ------------------------------------------------------------------------


@pytest.mark.slow
def test_immobilized_by_love(tables, names):
    s = _state(tables, names, infatuated=(1, 0))
    s = s._replace(protect_count=s.protect_count.at[0].set(2))
    acts = A("Scratch", "Tackle", s, names)
    stuck = assert_matches_reference(tables, s, acts, _o((1, 0)))
    assert _hp(stuck, 1) == _hp(s, 1)  # no Scratch
    assert int(stuck.pp[0, 0, 1]) == int(s.pp[0, 0, 1])  # no PP spent
    assert int(stuck.last_move[0]) == 0 and int(stuck.protect_count[0]) == 0
    assert _inf(stuck) == [1, 0]  # it stays in love
    ev = _events(tables, s, acts, _o((1, 0)))
    assert ev["in_love"].tolist() == [1, 0] and ev["immobilized"].tolist() == [1, 0]
    assert ev["act_kind"][0] == 0  # ACT_NONE: no attack message
    moves = assert_matches_reference(tables, s, acts, _o((0, 0)))
    assert _hp(moves, 1) < _hp(s, 1)
    ev = _events(tables, s, acts, _o((0, 0)))
    assert ev["in_love"].tolist() == [1, 0] and ev["immobilized"].tolist() == [0, 0]


def test_immobilized_unlocks_rollout_and_bide(tables, names):
    for move in ("Rollout", "Bide"):
        s = _state(tables, names, infatuated=(1, 0), mine=(move, "Scratch"))
        acts = A(move, "Tackle", s, names)
        t1 = assert_matches_reference(tables, s, acts, _o((0, 0)))  # locks in
        assert int(t1.locked_move[0]) == names.move(move)
        t2 = assert_matches_reference(tables, t1, acts, _o((1, 0)))
        assert int(t2.locked_move[0]) == 0
        assert int(t2.rollout_count[0]) == 0 and int(t2.bide_turns[0]) == 0


@pytest.mark.parametrize("before", ["confusion_hit", "full_para", "flinch", "sleep"])
def test_earlier_checks_come_first(tables, names, before):
    # Confusion (a self-hit), paralysis, a flinch and sleep stop the move before the
    # attract check: its roll changes nothing, and "in love" is not shown.
    opp = "Headbutt" if before == "flinch" else "Tackle"
    order = "opp" if before == "flinch" else "player"
    s = _state(tables, names, infatuated=(1, 0), opp=opp, order=order)
    if before == "confusion_hit":
        s = s._replace(confusion=s.confusion.at[0].set(3))
    elif before == "full_para":
        s = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    elif before == "sleep":
        s = s._replace(
            status=s.status.at[0, 0].set(Status.SLEEP),
            status_counter=s.status_counter.at[0, 0].set(3),
        )
    extra = {
        "confusion_hit": {"confusion_hit": [1, 0]},
        "full_para": {"full_para": [1, 0]},
        "flinch": {"secondary": np.ones((2, MAX_HITS))},
        "sleep": {},
    }[before]
    acts = A("Scratch", opp, s, names)
    outs = [assert_matches_reference(tables, s, acts, _o((k, 0), **extra)) for k in (0, 1)]
    for x, y in zip(outs[0], outs[1], strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    ev = _events(tables, s, acts, _o((1, 0), **extra))
    assert ev["in_love"][0] == 0 and ev["immobilized"][0] == 0


def test_confused_but_not_hurt_then_in_love(tables, names):
    s = _state(tables, names, infatuated=(1, 0))
    s = s._replace(
        confusion=s.confusion.at[0].set(3), status=s.status.at[0, 0].set(Status.PARALYSIS)
    )
    acts = A("Scratch", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o((1, 0)))
    assert _hp(out, 1) == _hp(s, 1)
    ev = _events(tables, s, acts, _o((1, 0)))
    assert ev["confused"][0] == 1 and ev["immobilized"][0] == 1


# --- clearing ----------------------------------------------------------------------------------


def test_the_source_switching_out_clears_it(tables, names):
    s = _state(tables, names, infatuated=(1, 0))
    out = assert_matches_reference(tables, s, A("Scratch", SWITCH, s, names), _o((1, 0)))
    assert _inf(out) == [0, 0]
    assert _hp(out, 1) < int(out.max_hp[1, 1])  # the switch went first: Scratch hit


def test_the_source_fainting_clears_it(tables, names):
    s = _state(tables, names, infatuated=(1, 0)).replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o((0, 0)))
    assert _hp(out, 1) == 0 and _inf(out) == [0, 0]
    assert _inf(replace(tables, out, 1, 1)) == [0, 0]


def test_the_holder_leaving_clears_it(tables, names):
    s = _state(tables, names, infatuated=(1, 0))
    out = assert_matches_reference(tables, s, A(SWITCH, "Tackle", s, names), _o())
    assert _inf(out) == [0, 0]
    s = _state(tables, names, infatuated=(1, 0), order="opp").replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o())
    assert _hp(out, 0) == 0 and _inf(out) == [0, 0]


def test_source_leaves_helper(tables, names):
    # The one place both sides of "the source leaves" are cleared (Roar reuses it).
    s = _state(tables, names, infatuated=(1, 1))
    s = s._replace(trapped=s.trapped.at[0].set(3), trap_move=s.trap_move.at[0].set(20))
    out = _source_leaves(s, 1, jnp.bool_(True))
    assert _inf(out) == [0, 1] and int(out.trapped[0]) == 0 and int(out.trap_move[0]) == 0
    assert _inf(_source_leaves(s, 1, jnp.bool_(False))) == [1, 1]


# --- Cute Charm --------------------------------------------------------------------------------


def _charm(tables, names, **kw):
    kw.setdefault("abilities", (AbilityId.NONE, AbilityId.CUTE_CHARM))
    return _state(tables, names, **kw)


def test_cute_charm_infatuates_a_contact_attacker(tables, names):
    s = _charm(tables, names)
    acts = A("Scratch", "Tackle", s, names)
    hit = assert_matches_reference(
        tables, s, acts, _o(contact_proc=[[0] * MAX_HITS, [1] + [0] * (MAX_HITS - 1)])
    )
    assert _inf(hit) == [1, 0]
    ev = _events(tables, s, acts, _o(contact_proc=[[0] * MAX_HITS, [1] + [0] * (MAX_HITS - 1)]))
    assert ev["contact_ability"].tolist() == [0, 1]
    miss = assert_matches_reference(tables, s, acts, _o())
    assert _inf(miss) == [0, 0]


@pytest.mark.parametrize(
    "case", ["same_gender", "genderless", "non_contact", "defender_faints", "already"]
)
def test_cute_charm_does_not_trigger(tables, names, case):
    genders = {"same_gender": (FEMALE, FEMALE), "genderless": (NONE, FEMALE)}.get(
        case, (MALE, FEMALE)
    )
    s = _charm(tables, names, genders=genders, infatuated=(int(case == "already"), 0))
    if case == "defender_faints":
        s = s.replace_hp(1, 0, 1)
    move = "Ember" if case == "non_contact" else "Scratch"
    o = _o(contact_proc=np.ones((2, MAX_HITS)))
    out = assert_matches_reference(tables, s, A(move, "Tackle", s, names), o)
    assert _inf(out)[0] == int(case == "already")
    if case != "already":
        assert _events(tables, s, A(move, "Tackle", s, names), o)["contact_ability"][1] == 0


def test_cute_charm_on_a_multi_hit_move(tables, names):
    s = _charm(tables, names)
    acts = A("Double Kick", "Tackle", s, names)
    for procs in ([1, 0], [0, 1], [1, 1]):
        o = _o(contact_proc=[[0] * MAX_HITS, procs + [0] * (MAX_HITS - 2)])
        out = assert_matches_reference(tables, s, acts, o)
        assert _inf(out) == [1, 0]


def test_cute_charm_holder_on_the_player_is_revealed(tables, names):
    s = _charm(tables, names, abilities=(AbilityId.CUTE_CHARM, AbilityId.NONE), order="opp")
    s = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(0))
    o = _o(contact_proc=[[1] + [0] * (MAX_HITS - 1), [0] * MAX_HITS])
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), o)
    assert _inf(out) == [0, 1] and int(out.ai_ability_known[0]) == 1
    none = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o())
    assert int(none.ai_ability_known[0]) == 0


def test_pursuit_into_a_switching_cute_charm_holder(tables, names):
    # Pursuit hits the outgoing holder (contact): the pursuer falls in love, then the holder
    # leaves and UpdateAfterSwitch clears the bit it was the source of.
    s = _charm(tables, names, opp="Pursuit", abilities=(AbilityId.CUTE_CHARM, AbilityId.NONE))
    o = _o(contact_proc=[[1] + [0] * (MAX_HITS - 1), [0] * MAX_HITS])
    out = assert_matches_reference(tables, s, A(SWITCH, "Pursuit", s, names), o)
    assert _inf(out) == [0, 0] and int(out.active[0]) == 1


# --- observation and plan tree ------------------------------------------------------------------


@pytest.mark.slow
def test_observation_and_labels(tables, names):
    # The infatuated player's Attract on a foe already in love changes no state: the
    # observation still tells moving from immobilized.
    s = _state(tables, names, infatuated=(1, 1))
    acts = A("Attract", "Tackle", s, names)
    obs = []
    for k in (0, 1):
        o = _o((k, 0))
        obs.append(np.asarray(observe(tables, s, acts, o, step(tables, s, acts, o))))
    assert (obs[0] != obs[1]).any()
    # The infatuation itself is seen ("fell in love"): Attract against a same-gender foe
    # (fails) and an opposite-gender one differ only in it.
    seen = []
    for genders in ((MALE, MALE), (MALE, FEMALE)):
        t = _state(tables, names, genders=genders, opp="Leer")
        a = A("Attract", "Leer", t, names)
        seen.append(np.asarray(observe(tables, t, a, _o(), step(tables, t, a, _o()))))
    assert (seen[0] != seen[1]).any()
    texts = dict(_FLAG_TEXT)
    assert texts["in_love"] == "is in love"
    assert texts["immobilized"] == "is immobilized by love"


@pytest.mark.slow
def test_reference_agreement_over_draws(tables, names):
    """Both engines agree over infatuation x contact_proc x hit x speed_tie, for Attract,
    contact and non-contact moves, every order, genders, Cute Charm and a start
    infatuation."""
    for mine, opp, order, genders, charm, start in itertools.product(
        ("Attract", "Scratch", "Ember"),
        ("Attract", "Tackle"),
        ("player", "opp", "tie"),
        ((MALE, FEMALE), (MALE, MALE)),
        (False, True),
        ((0, 0), (1, 1)),
    ):
        abilities = (AbilityId.CUTE_CHARM, AbilityId.CUTE_CHARM) if charm else None
        s = _state(
            tables,
            names,
            opp=opp,
            order=order,
            genders=genders,
            infatuated=start,
            abilities=abilities or (AbilityId.NONE, AbilityId.NONE),
        )
        acts = A(mine, opp, s, names)
        for i0, i1, proc, tie in itertools.product((0, 1), (0, 1), (0, 1), (0, 1)):
            o = _o((i0, i1), contact_proc=np.full((2, MAX_HITS), proc), speed_tie=tie)
            assert_matches_reference(tables, s, acts, o)


# --- relevance, enumeration -----------------------------------------------------------------------


def test_relevance_of_the_infatuation_roll(tables, names):
    def rel(s, acts, loose=False):
        return np.asarray(laws_of(tables, s, acts, loose=loose).relevant.infatuation).tolist()

    s = _state(tables, names, infatuated=(1, 0))
    assert rel(s, A("Scratch", "Tackle", s, names)) == [True, False]
    # The source switches first: the bit is gone before the check.
    assert rel(s, A("Scratch", SWITCH, s, names)) == [False, False]
    assert rel(s, A("Scratch", SWITCH, s, names), loose=True) == [True, False]
    # The foe's Attract first, opposite genders: read; moving after it or the same gender: not.
    s = _state(tables, names, opp="Attract", order="opp")
    assert rel(s, A("Scratch", "Attract", s, names)) == [True, False]
    s = _state(tables, names, opp="Attract", order="player")
    assert rel(s, A("Scratch", "Attract", s, names)) == [False, False]
    assert rel(s, A("Scratch", "Attract", s, names), loose=True) == [True, False]
    s = _state(tables, names, opp="Attract", order="opp", genders=(MALE, MALE))
    assert rel(s, A("Scratch", "Attract", s, names)) == [False, False]
    assert rel(s, A("Scratch", "Attract", s, names), loose=True) == [True, False]
    # The tie: read only in the branch where the foe goes first (speed_tie 1).
    s = _state(tables, names, opp="Attract", order="tie")
    laws = laws_of(tables, s, A("Scratch", "Attract", s, names))
    assert np.asarray(laws.branch_relevant.infatuation)[:, 0].tolist() == [False, True]


def test_relevance_of_cute_charm(tables, names):
    def rel(s, move, loose=False):
        laws = laws_of(tables, s, A(move, "Tackle", s, names), loose=loose)
        return bool(np.asarray(laws.relevant.contact_proc)[1, 0])

    assert rel(_charm(tables, names), "Scratch")
    assert not rel(_charm(tables, names), "Ember")
    for kw in ({"genders": (MALE, MALE)}, {"genders": (NONE, FEMALE)}, {"infatuated": (1, 0)}):
        s = _charm(tables, names, **kw)
        assert not rel(s, "Scratch") and rel(s, "Scratch", loose=True), kw


def _switch_to_charmer(tables, names):
    """The player is in love with Geodude, which switches to a female Cute Charm Onix: the
    switch clears the love first, so the player's Scratch can proc the newcomer's charm."""
    s = _state(tables, names, infatuated=(1, 0))
    return s._replace(
        ability=s.ability.at[1, 1].set(AbilityId.CUTE_CHARM),
        gender=s.gender.at[1, 1].set(FEMALE),
    )


@pytest.mark.slow
def test_cute_charm_of_a_switch_in_after_the_source_left(tables, names):
    s = _switch_to_charmer(tables, names)
    acts = A("Scratch", SWITCH, s, names)
    laws = laws_of(tables, s, acts)
    assert bool(np.asarray(laws.relevant.contact_proc)[1, 0])
    o = _o((1, 0), contact_proc=[[0] * MAX_HITS, [1] + [0] * (MAX_HITS - 1)])
    out = assert_matches_reference(tables, s, acts, o)
    assert _inf(out) == [1, 0] and int(out.active[1]) == 1
    assert_equals_bruteforce(tables, s, acts)


def test_damage_classes_classify_attract_and_cute_charm():
    assert EffectId.ATTRACT in INPUT_NEUTRAL_EFFECTS
    assert EffectId.ATTRACT in NON_TABLE_DAMAGE_EFFECTS
    assert AbilityId.CUTE_CHARM in INPUT_NEUTRAL_ABILITIES


def test_immobilization_probability(tables, names):
    s = _state(tables, names, opp="Attract", order="opp")
    outcomes, probs, drawn = enumerate_drawn(laws_of(tables, s, A("Scratch", "Attract", s, names)))
    read = np.asarray(drawn.infatuation)[:, 0]
    stuck = read & (np.asarray(outcomes.infatuation)[:, 0] == 1)
    assert read.all()  # Attract cannot miss here (accuracy 100)
    assert math.fsum(probs[stuck]) == pytest.approx(0.5, abs=1e-15)


@pytest.mark.slow
def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _charm(tables, names, opp="Attract", order="opp")
    acts = A("Scratch", "Attract", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


BRUTE_CASES = (
    # Attract by the faster side, then the infatuated slower mover; and the tie.
    [("Scratch", "Attract", order, None) for order in ("player", "opp", "tie")]
    + [("Attract", "Tackle", order, None) for order in ("player", "opp", "tie")]
    + [("Attract", "Attract", "tie", None)]
    # Same gender / genderless: the foe's Attract cannot work.
    + [("Scratch", "Attract", "opp", g) for g in ("same_gender", "genderless")]
    # A confused, paralysed and infatuated mover.
    + [("Scratch", "Tackle", order, "confused_paralysed") for order in ("player", "opp", "tie")]
    # A switch clearing infatuation: the source's, or the holder's own.
    + [("Scratch", SWITCH, "player", "infatuated"), (SWITCH, "Tackle", "opp", "infatuated")]
    + [("Scratch", "Tackle", order, "infatuated") for order in ("player", "opp", "tie")]
    # Cute Charm on contact (both directions), non-contact, Attract into Cute Charm.
    + [("Scratch", "Tackle", order, "charm") for order in ("player", "opp", "tie")]
    + [("Ember", "Tackle", "tie", "charm"), ("Double Kick", "Tackle", "player", "charm")]
    + [("Attract", "Tackle", "tie", "charm")]
    # Protect blocks the foe's Attract; a flinch stops the infatuated mover first.
    + [("Scratch", "Protect", "tie", None), ("Scratch", "Headbutt", "opp", "infatuated")]
    # An infatuated first mover's flinch or Protect is not sure: it may be immobilized.
    + [("Scratch", "Headbutt", "opp", "foe_infatuated")]
    + [("Scratch", "Protect", "player", "foe_infatuated")]
)


@pytest.mark.slow
@pytest.mark.parametrize("mine,opp,order,setup", BRUTE_CASES)
def test_gated_equals_bruteforce(tables, names, mine, opp, order, setup):
    genders = {"same_gender": (MALE, MALE), "genderless": (MALE, NONE)}.get(setup, (MALE, FEMALE))
    s = _state(tables, names, opp=opp, order=order, genders=genders)
    if setup in ("infatuated", "confused_paralysed"):
        s = s._replace(infatuated=s.infatuated.at[0].set(1))
    if setup == "foe_infatuated":
        s = s._replace(infatuated=s.infatuated.at[1].set(1))
    if setup == "confused_paralysed":
        s = s._replace(
            confusion=s.confusion.at[0].set(3), status=s.status.at[0, 0].set(Status.PARALYSIS)
        )
    if setup == "charm":
        s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.CUTE_CHARM))
    if mine == "Double Kick":  # no crits: keeps the brute force under the width cap
        s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    width, brute = assert_equals_bruteforce(tables, s, A(mine, opp, s, names))
    assert width <= brute


@pytest.mark.slow
def test_cute_charm_then_the_infatuated_attacker_bruteforce(tables, names):
    # Turn 1: the faster Scratch into the foe's Cute Charm infatuates the attacker (its own
    # check came before: the roll is read next turn). Turn 2: the attacker's check reads it.
    for order, side in (("player", 0), ("opp", 1)):
        s = _charm(tables, names, order=order)
        if side == 1:
            s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.CUTE_CHARM))
            s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.NONE))
        acts = A("Scratch", "Tackle", s, names)
        assert_equals_bruteforce(tables, s, acts)
        laws = laws_of(tables, s, acts)
        assert not np.asarray(laws.relevant.infatuation).any()
        proc = [[0] * MAX_HITS, [0] * MAX_HITS]
        proc[1 - side][0] = 1
        t2 = step(tables, s, acts, _o(contact_proc=proc))
        assert _inf(t2)[side] == 1
        assert np.asarray(laws_of(tables, t2, acts).relevant.infatuation)[side]
        assert_equals_bruteforce(tables, t2, acts)


@pytest.mark.parametrize("order", ["skitty", "foe", "tie"])
def test_kaizo_skitty_bruteforce(kaizo_tables, kaizo_names, order):
    # Rules §4(a): Skitty (F, Cute Charm) against a male Roark Pokemon.
    t, n = kaizo_tables, kaizo_names
    skitty = replace_spec(_kaizo_spec(n, "Skitty", "Attract", 20), gender=FEMALE)
    s = make_state(t, [skitty], [_kaizo_spec(n, "Cranidos", "Headbutt", 14)])
    assert int(s.ability[0, 0]) == AbilityId.CUTE_CHARM
    spe = int(s.stats[0, 0, _SPE])
    foe = {"skitty": spe - 1, "foe": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    width, brute = assert_equals_bruteforce(t, s, jnp.zeros(2, jnp.int32))
    assert width <= brute
    assert math.isclose(math.fsum(enumerate_outcomes(laws_of(t, s, jnp.zeros(2, jnp.int32)))[1]), 1)


# --- coverage: Oblivious --------------------------------------------------------------------------


def test_oblivious_is_refused(tables, names):
    base = sample_party(names)
    slow = names.species("Slowpoke")
    k = [names.ability_names[a] for a in names.species_abilities(slow)].index("Oblivious")
    mon = replace_spec(base[0], species=slow, form=0, ability_slot=k)
    report = check_coverage(tables, names, [mon, *base[1:]], roark_trainer(), roark_party(names), 0)
    assert "Player SLOWPOKE (slot 0): Oblivious (ability UNSUPPORTED)" in report.missing
    cand = Candidate(("party", 0, 0), "P", mon, 0, (), ())
    assert "P: Oblivious (ability UNSUPPORTED)" in check_pool(tables, names, [cand], ()).missing
    assert _ability_refusal(tables, mon) == "ability not implemented"
    # The opponent's side is refused the same way.
    opp = [mon, *roark_party(names)[1:]]
    report = check_coverage(tables, names, base, roark_trainer(), opp, 0)
    assert any(m.endswith("Oblivious (ability UNSUPPORTED)") for m in report.missing)


@pytest.mark.parametrize("item", ["Mental Herb", "Destiny Knot"])
def test_infatuation_items_are_refused(tables, names, item):
    # HOLD_EFFECT_HEAL_INFATUATION (battle_lib.c:4890, :5103) and HOLD_EFFECT_RECIPROCATE_INFAT
    # (:4545-4559) would change infatuation; neither is modelled, so the item is refused on
    # both sides, in the pool and in prep.
    base = sample_party(names)
    mon = replace_spec(base[0], item=names.item(item))
    line = f"MONFERNO (slot 0): {item} (held item UNSUPPORTED)"
    report = check_coverage(tables, names, [mon, *base[1:]], roark_trainer(), roark_party(names), 0)
    assert f"Player {line}" in report.missing
    opp = [mon, *roark_party(names)[1:]]
    assert f"Roark {line}" in check_coverage(tables, names, base, roark_trainer(), opp, 0).missing
    cand = Candidate(("party", 0, 0), "P", mon, 0, (), ())
    assert f"P: {item} (held item UNSUPPORTED)" in check_pool(tables, names, [cand], ()).missing
    assert _item_refusal(tables, mon.item) == "held item not implemented"
