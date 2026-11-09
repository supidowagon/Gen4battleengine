"""Thunder Fang (K3b2 Task 4, ``EffectId.FANG_PARALYZE_FLINCH``; docs/kaizo/k3b2-rules.md §1).

Decomp (external/pokeplatinum):

* ``res/battle/scripts/effects/effect_script_0275.s:5-7``: an indirect
  ``ON_HIT | TO_DEFENDER | PARALYZE_OR_FLINCH`` side effect, then ``CalcCrit``,
  ``CalcDamage``: a plain hit plus one subscript.
* ``BattleSystem_TriggerSecondaryEffect`` (src/battle/battle_lib.c:1524-1529): the ``ON_HIT``
  branch loads the subscript whenever the move had effects (no outer chance roll);
  battle_controller_player.c:3489-3490 also needs the move not to have missed.
* ``subscript_paralyze_or_flinch.s:5-10``: ``CheckEffectActivation`` then
  ``Call BATTLE_SUBSCRIPT_PARALYZE``, then ``CheckEffectActivation`` then
  ``Call BATTLE_SUBSCRIPT_FLINCH_MON``: two rolls, paralysis first, both always made.
* ``BtlCmd_CheckEffectActivation`` (src/battle/battle_script.c:8363-8383):
  ``RandNext() % 100 < effectChance`` (Serene Grace doubles it; unsupported) and the
  defender has HP, the roll first.
* ``subscript_paralyze.s``: Limber (ignorable, :5), Leaf Guard in sun (:6-8), Shield Dust
  (:11-12), substitute (:20-21), already paralyzed or any status (:24-25), then Safeguard
  (:28); every failure is silent for an indirect effect (:58-60, :65-67, :78-80). No type
  check: Electric types can be paralyzed in Gen 4.
* ``subscript_flinch_mon.s``: ``IfMovedThisTurn`` (:5; ``Battler_MovedThisTurn``,
  battle_lib.c:5365-5368) skips a target that already acted or switched; substitute (:6);
  Inner Focus (ignorable, :7), whose message is for Fake Out or a direct flinch only
  (:17-27); Shield Dust (:9).
* The contact ability (Static, Rough Skin) runs after the secondary
  (``ONE_HIT_TRIGGER_ABILITY``, battle_controller_player.c:3513-3523).
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

from nuzline.engine.enumerate import (
    INPUT_CLAUSE_EFFECTS,
    TABLE_DAMAGE_EFFECTS,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import AbilityId, EffectId, Status, Type
from nuzline.engine.outcome import Outcome, empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step_events
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
FANG = "Thunder Fang"
SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot


def _state(tables, names, *, player=FANG, opp="Tackle", **kw):
    """Monferno (faster) with ``player`` against Cranidos with ``opp``."""
    return roark_state(
        tables,
        names,
        player_moves=[player],
        opp_species="Cranidos",
        opp_moves=[opp],
        **kw,
    )


def _ability(s, side, ability):
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _o(para=0, flinch=0, side=0, contact=0) -> Outcome:
    o = empty_outcome()
    return o._replace(
        secondary=o.secondary.at[side, 0].set(para),
        fang_flinch=o.fang_flinch.at[side].set(flinch),
        contact_proc=o.contact_proc.at[1 - side, 0].set(contact),
    )


def _flinched(tables, s, acts, o) -> list[int]:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return np.asarray(events["flinched"]).tolist()


# --- mapping and decomp scripts -------------------------------------------------------------


def test_effect_275_maps_in_both_games(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[275] == (EffectId.FANG_PARALYZE_FLINCH, (Status.PARALYSIS, 0, 0))
    # Fire Fang (273) and Ice Fang (274) stay unmapped until their scripts are read
    # (rules §11 item 7); 276 is Charge Beam's RAISE_SP_ATK_HIT.
    assert 273 not in DECOMP_EFFECT_TO_ID and 274 not in DECOMP_EFFECT_TO_ID
    for t, n, power, accuracy in ((tables, names, 65, 95), (kaizo_tables, kaizo_names, 90, 100)):
        move = n.move(FANG)
        assert int(t.move_effect[move]) == EffectId.FANG_PARALYZE_FLINCH
        assert np.asarray(t.move_param[move]).tolist() == [Status.PARALYSIS, 0, 0]
        assert int(t.move_effect_chance[move]) == 10
        assert (int(t.move_power[move]), int(t.move_accuracy[move])) == (power, accuracy)
        for other in ("Fire Fang", "Ice Fang"):
            assert int(t.move_effect[n.move(other)]) == EffectId.UNSUPPORTED, other


def test_fang_with_chance_zero_is_a_build_error():
    # BtlCmd_CheckEffectActivation asserts effectChance != 0 (battle_script.c:8375).
    from tools.build_tables import BuildError, _move_effect

    with pytest.raises(BuildError):
        _move_effect({"name": "X", "effect": 275, "effect_chance": 0})
    assert _move_effect({"name": "X", "effect": 275, "effect_chance": 10}) == (
        int(EffectId.FANG_PARALYZE_FLINCH),
        (Status.PARALYSIS, 0, 0),
    )


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_do_what_the_engine_does():
    effects = DECOMP / "res/battle/scripts/effects"
    subs = DECOMP / "res/battle/scripts/subscripts"
    s275 = (effects / "effect_script_0275.s").read_text("utf-8")
    assert (
        "BTLVAR_SIDE_EFFECT_FLAGS_INDIRECT, MOVE_SIDE_EFFECT_ON_HIT|MOVE_SIDE_EFFECT_TO_DEFENDER"
        "|MOVE_SUBSCRIPT_PTR_PARALYZE_OR_FLINCH" in s275
    )
    steps = [ln.split()[0] for ln in s275.splitlines() if ln.startswith("    ")]
    assert steps == ["UpdateVar", "CalcCrit", "CalcDamage", "End"]
    both = (subs / "subscript_paralyze_or_flinch.s").read_text("utf-8")
    steps = [ln.strip() for ln in both.splitlines() if ln.startswith("    ")]
    assert steps == [
        "CheckEffectActivation _003",
        "Call BATTLE_SUBSCRIPT_PARALYZE",
        "CheckEffectActivation _007",
        "Call BATTLE_SUBSCRIPT_FLINCH_MON",
        "End",
    ]
    para = (subs / "subscript_paralyze.s").read_text("utf-8")
    # No type check (only SIDE_EFFECT_TYPE_*): no Electric-type immunity in Gen 4.
    assert not re.search(r"(?<!SIDE_EFFECT_)TYPE_", para)
    assert "BATTLEMON_TYPE" not in para
    assert "CheckIgnorableAbility CHECK_HAVE, BTLSCR_SIDE_EFFECT_MON, ABILITY_LIMBER" in para
    flinch = (subs / "subscript_flinch_mon.s").read_text("utf-8").splitlines()
    assert "IfMovedThisTurn BTLSCR_SIDE_EFFECT_MON" in flinch[4]
    assert "ABILITY_INNER_FOCUS" in flinch[6]
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8").splitlines()
    assert "BattleSystem_RandNext(battleSys) % 100 < effectChance" in script[8376]
    assert "battleMons[battleCtx->sideEffectMon].curHP" in script[8377]


# --- the two rolls ---------------------------------------------------------------------------


def test_both_rolls_independent(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    for para, flinch in itertools.product((0, 1), (0, 1)):
        out = assert_matches_reference(tables, s, acts, _o(para, flinch))
        assert int(out.status[1, 0]) == (Status.PARALYSIS if para else Status.NONE)
        assert _flinched(tables, s, acts, _o(para, flinch)) == [0, flinch]
        # A flinched Cranidos does not Tackle.
        assert (int(out.hp[0, 0]) == int(s.hp[0, 0])) == bool(flinch)


def test_no_flinch_once_the_target_moved_or_switched(tables, names):
    slow = _state(tables, names, opp_faster=True)
    acts = A(FANG, "Tackle", slow, names)
    a = assert_matches_reference(tables, slow, acts, _o(0, 0))
    b = assert_matches_reference(tables, slow, acts, _o(0, 1))
    for x, y in zip(a, b, strict=True):
        assert (np.asarray(x) == np.asarray(y)).all()
    # The paralysis still lands.
    assert int(assert_matches_reference(tables, slow, acts, _o(1, 1)).status[1, 0]) == (
        Status.PARALYSIS
    )
    # A switch goes first: the incoming Pokemon has moved this turn.
    s = _state(tables, names)
    acts = A(FANG, SWITCH_TO_1, s, names)
    a = assert_matches_reference(tables, s, acts, _o(0, 0))
    b = assert_matches_reference(tables, s, acts, _o(0, 1))
    for x, y in zip(a, b, strict=True):
        assert (np.asarray(x) == np.asarray(y)).all()


def test_inner_focus_blocks_the_flinch_silently(tables, names):
    # Cranidos (Sturdy, so no Mold Breaker) bites into an Inner Focus Monferno: no flinch,
    # and no ability message (subscript_flinch_mon.s:17-27: Fake Out or direct only).
    s = _state(tables, names, player="Scratch", opp=FANG, opp_faster=True)
    s = _ability(_ability(s, 1, AbilityId.STURDY), 0, AbilityId.INNER_FOCUS)
    acts = A("Scratch", FANG, s, names)
    out = assert_matches_reference(tables, s, acts, _o(0, 1, side=1))
    assert _flinched(tables, s, acts, _o(0, 1, side=1)) == [0, 0]
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # Monferno's Scratch still ran
    assert int(out.ai_ability_known[0]) == int(s.ai_ability_known[0])
    # Cranidos's own Mold Breaker pierces it.
    pierced = _ability(s, 1, AbilityId.MOLD_BREAKER)
    assert_matches_reference(tables, pierced, acts, _o(0, 1, side=1))
    assert _flinched(tables, pierced, acts, _o(0, 1, side=1)) == [1, 0]


# --- the paralysis -----------------------------------------------------------------------


def test_paralysis_needs_a_target_that_can_take_it(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    burned = s.with_status(1, 0, Status.BURN)
    assert int(assert_matches_reference(tables, burned, acts, _o(1)).status[1, 0]) == Status.BURN
    # Gen 4 has no Electric-type immunity to paralysis (subscript_paralyze.s).
    electric = s._replace(types=s.types.at[1, 0].set(Type.ELECTRIC))
    out = assert_matches_reference(tables, electric, acts, _o(1))
    assert int(out.status[1, 0]) == Status.PARALYSIS
    # A Ground type is immune to the hit, so neither roll applies.
    ground = s._replace(types=s.types.at[1, 0].set(Type.GROUND))
    out = assert_matches_reference(tables, ground, acts, _o(1, 1))
    assert int(out.status[1, 0]) == Status.NONE and int(out.hp[1, 0]) == int(s.hp[1, 0])
    assert _flinched(tables, ground, acts, _o(1, 1)) == [0, 0]
    # A target the hit faints takes neither (CheckEffectActivation needs its HP).
    low = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, low, acts, _o(1, 1))
    assert int(out.hp[1, 0]) == 0 and int(out.status[1, 0]) == Status.NONE


def test_limber_blocks_the_paralysis_silently(tables, names):
    # Cranidos (Sturdy) bites a Limber Monferno: nothing, and no ability message (indirect).
    s = _state(tables, names, player="Scratch", opp=FANG, opp_faster=True)
    s = _ability(_ability(s, 1, AbilityId.STURDY), 0, AbilityId.LIMBER)
    acts = A("Scratch", FANG, s, names)
    out = assert_matches_reference(tables, s, acts, _o(1, 0, side=1))
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == int(s.ai_ability_known[0])
    # Mold Breaker pierces it; Limber then cures it at the move end, with its message
    # (hooks.recover_status_by_ability, battle_lib.c:4397).
    pierced = _ability(s, 1, AbilityId.MOLD_BREAKER)
    out = assert_matches_reference(tables, pierced, acts, _o(1, 0, side=1))
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 1


# --- contact abilities ---------------------------------------------------------------------


def test_static_and_rough_skin(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    static = _ability(s, 1, AbilityId.STATIC)
    out = assert_matches_reference(tables, static, acts, _o(1, 1, contact=1))
    # Both paralyzed: the fang's on Cranidos, Static's on Monferno (after the secondary).
    assert int(out.status[1, 0]) == Status.PARALYSIS
    assert int(out.status[0, 0]) == Status.PARALYSIS
    out = assert_matches_reference(tables, static, acts, _o(0, 0, contact=0))
    assert int(out.status[0, 0]) == Status.NONE
    rough = _ability(s, 1, AbilityId.ROUGH_SKIN)
    out = assert_matches_reference(tables, rough, acts, _o(1, 1))
    mx = int(s.max_hp[0, 0])
    assert int(out.hp[0, 0]) == mx - max(mx // 8, 1)  # flinched: no Tackle
    assert int(out.status[1, 0]) == Status.PARALYSIS


def test_matches_reference_over_every_roll(tables, names):
    """Both engines agree over every combination of the action's rolls, in both orders,
    into Static, Inner Focus and Limber targets, a burned one and a Steadfast one."""
    base = _state(tables, names)
    setups = {
        "plain": base,
        "slow": _state(tables, names, opp_faster=True),
        "static": _ability(base, 1, AbilityId.STATIC),
        "inner_focus": _ability(base, 1, AbilityId.INNER_FOCUS),
        "limber": _ability(base, 1, AbilityId.LIMBER),
        "steadfast": _ability(base, 1, AbilityId.STEADFAST),
        "burned": base.with_status(1, 0, Status.BURN),
    }
    for s in setups.values():
        acts = A(FANG, "Tackle", s, names)
        for para, flinch, contact, crit, hit in itertools.product((0, 1), repeat=5):
            o = _o(para, flinch, contact=contact)
            o = o._replace(crit=o.crit.at[0, 0].set(crit), hit=o.hit.at[0, 0].set(hit))
            assert_matches_reference(tables, s, acts, o)


# --- laws and enumeration ------------------------------------------------------------------


def test_sampled_rolls_are_independent_tenths(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    n = 40000
    keys = jax.random.split(jax.random.key(1), n)
    o = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))(keys, tables, s, acts)
    para = np.asarray(o.secondary)[:, 0, 0] != 0
    flinch = np.asarray(o.fang_flinch)[:, 0] != 0
    for got, p in ((para, 0.1), (flinch, 0.1), (para & flinch, 0.01)):
        assert abs(got.mean() - p) <= 5 * math.sqrt(p * (1 - p) / n)
    # The fang_flinch draw is fold_in(key, 16): every earlier field keeps its samples.
    one = sample_outcome(jax.random.key(0), tables, s, acts)
    want = jax.random.randint(jax.random.fold_in(jax.random.key(0), 16), (2,), 0, 100) < 10
    assert np.asarray(one.fang_flinch).tolist() == np.asarray(want).astype(int).tolist()


def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _ability(_state(tables, names), 1, AbilityId.STATIC)
    acts = A(FANG, "Tackle", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


def test_fang_flinch_read_only_by_the_first_mover(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    laws = laws_of(tables, s, acts)
    assert np.asarray(laws.relevant.fang_flinch).tolist() == [True, False]
    assert bool(laws.relevant.secondary[0, 0])
    slow = _state(tables, names, opp_faster=True)
    laws = laws_of(tables, slow, A(FANG, "Tackle", slow, names))
    assert not np.asarray(laws.relevant.fang_flinch).any()
    assert bool(laws.relevant.secondary[0, 0])  # the paralysis can still land
    # An Inner Focus target, unless the user has Mold Breaker.
    focus = _ability(s, 1, AbilityId.INNER_FOCUS)
    assert not np.asarray(laws_of(tables, focus, acts).relevant.fang_flinch).any()
    pierced = _ability(focus, 0, AbilityId.MOLD_BREAKER)
    assert bool(laws_of(tables, pierced, acts).relevant.fang_flinch[0])
    # The loose brute force keeps it.
    assert bool(laws_of(tables, slow, acts, loose=True).relevant.fang_flinch[0])


def test_fang_paralysis_read_only_on_a_target_that_can_take_it(tables, names):
    s = _state(tables, names)
    acts = A(FANG, "Tackle", s, names)
    for blocked in (s.with_status(1, 0, Status.BURN), _ability(s, 1, AbilityId.LIMBER)):
        laws = laws_of(tables, blocked, acts)
        assert not bool(laws.relevant.secondary[0, 0])
        assert bool(laws.relevant.fang_flinch[0])
    pierced = _ability(_ability(s, 1, AbilityId.LIMBER), 0, AbilityId.MOLD_BREAKER)
    assert bool(laws_of(tables, pierced, acts).relevant.secondary[0, 0])
    # Moving second, the target's status can change first: still read.
    slow = _state(tables, names, opp_faster=True).with_status(1, 0, Status.BURN)
    assert bool(laws_of(tables, slow, A(FANG, "Tackle", slow, names)).relevant.secondary[0, 0])


def test_sure_fang_flinch_skips_the_foe_reads(tables, names):
    # Thunder Fang first (Lucky Chant keeps the brute force small): a sure hit that
    # flinches skips Cranidos's whole move, so fewer rows than the brute force.
    s = _state(tables, names)
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    acts = A(FANG, "Tackle", s, names)
    laws = laws_of(tables, s, acts)
    assert bool(laws.reads.flinch_sure[0])
    outcomes, _, drawn = enumerate_drawn(laws)
    flinched = np.asarray(drawn.fang_flinch)[:, 0] & (np.asarray(outcomes.fang_flinch)[:, 0] == 1)
    flinched &= np.asarray(outcomes.hit)[:, 0, 0] != 0
    assert flinched.any()
    for name in ("hit", "crit", "roll"):
        assert not np.asarray(getattr(drawn, name))[flinched, 1].any(), name
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width < brute


SETUPS = (
    None,
    "slow",
    "tie",
    "inner_focus",
    "mold_breaker_vs_inner_focus",
    "limber",
    "burned",
    "paralyzed",
    "static",
    "rough_skin",
    "steadfast_tie",
    "guts_foe",
    "focus_sash",
    "low_hp",
)


@pytest.mark.parametrize("opp", ["Tackle", FANG, "Zen Headbutt", "Thunder Wave"])
@pytest.mark.parametrize("setup", SETUPS)
def test_fang_equals_bruteforce(tables, names, setup, opp):
    """The gated enumeration (order branches, the order-gated fang flinch, the status
    eligibility of the paralysis, the sure flinch, damage classes) equals the loose brute
    force merged by state and observation."""
    s = _state(tables, names, opp=opp, opp_faster=setup == "slow")
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))  # no crits: small brute force
    s = _ability(s, 1, AbilityId.STURDY)  # Cranidos without Mold Breaker unless set
    if setup in ("tie", "steadfast_tie"):
        spe = 4  # stats[..., 4] is Spe
        s = s._replace(stats=s.stats.at[1, 0, spe].set(s.stats[0, 0, spe]))
    if setup == "inner_focus":
        s = _ability(s, 1, AbilityId.INNER_FOCUS)
    elif setup == "mold_breaker_vs_inner_focus":
        s = _ability(_ability(s, 1, AbilityId.INNER_FOCUS), 0, AbilityId.MOLD_BREAKER)
    elif setup == "limber":
        s = _ability(s, 1, AbilityId.LIMBER)
    elif setup == "burned":
        s = s.with_status(1, 0, Status.BURN)
    elif setup == "paralyzed":
        s = s.with_status(1, 0, Status.PARALYSIS)
    elif setup == "static":
        s = _ability(s, 1, AbilityId.STATIC)
    elif setup == "rough_skin":
        s = _ability(s, 1, AbilityId.ROUGH_SKIN)
    elif setup == "steadfast_tie":
        s = _ability(s, 1, AbilityId.STEADFAST)
    elif setup == "guts_foe":
        # Cranidos's Tackle after a paralysis: Guts raises its damage (a damage input).
        s = _ability(s, 1, AbilityId.GUTS)
    elif setup == "focus_sash":
        s = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
    elif setup == "low_hp":
        s = s.replace_hp(1, 0, 3)
    if opp == FANG and setup in ("tie", "steadfast_tie"):
        pytest.skip("over the cap at 95 %: test_kaizo_fang_tie_equals_bruteforce")
    acts = A(FANG, opp, s, names)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width <= brute


@pytest.mark.parametrize("steadfast", [False, True])
def test_kaizo_fang_tie_equals_bruteforce(kaizo_tables, kaizo_names, steadfast):
    # Two Thunder Fangs (Kaizo: 100 %, no accuracy roll) at a speed tie: each flinch roll
    # is read only in the branch where its user goes first; a Steadfast target's flinch
    # changes its end-of-turn speed.
    t, n = kaizo_tables, kaizo_names
    s = make_state(t, [_kaizo_spec(n, "Luvdisc", FANG, 16)], [_kaizo_spec(n, "Cranidos", FANG, 14)])
    s = s._replace(
        stats=s.stats.at[1, 0, 4].set(s.stats[0, 0, 4]),
        lucky_chant=s.lucky_chant.at[:].set(5),
        ability=s.ability.at[:, 0].set(AbilityId.STURDY),
    )
    if steadfast:
        s = _ability(s, 1, AbilityId.STEADFAST)
    acts = jnp.zeros(2, jnp.int32)
    laws = laws_of(t, s, acts)
    assert bool(laws.relevant.speed_tie)
    assert np.asarray(laws.branch_relevant.fang_flinch).tolist() == [[True, False], [False, True]]
    assert np.asarray(laws.reads.flinch_sure).tolist() == [True, True]
    width, brute = assert_equals_bruteforce(t, s, acts)
    assert width < brute


def test_guts_target_hit_is_not_compressed_after_a_fang(tables, names):
    # Thunder Fang can paralyze Cranidos before its Tackle: under Guts that is a damage
    # input, so Cranidos's hit is not drawn as damage classes in that order.
    assert EffectId.FANG_PARALYZE_FLINCH in INPUT_CLAUSE_EFFECTS
    assert EffectId.FANG_PARALYZE_FLINCH in TABLE_DAMAGE_EFFECTS
    s = _ability(_state(tables, names), 1, AbilityId.GUTS)
    laws = laws_of(tables, s, A(FANG, "Tackle", s, names))
    assert not bool(np.asarray(laws.compress)[0, 1, 0])
    assert bool(np.asarray(laws.compress)[0, 0, 0])  # the fang's own hit is
    plain = _ability(_state(tables, names), 1, AbilityId.STURDY)
    laws = laws_of(tables, plain, A(FANG, "Tackle", plain, names))
    assert bool(np.asarray(laws.compress)[0, 1, 0])


def test_kaizo_gible_thunder_fang_width(kaizo_tables, kaizo_names):
    # Roark's Gible's Thunder Fang (90 BP, 100 %) against a faster Pokemon: the flinch is
    # never read, the paralysis is.
    t, n = kaizo_tables, kaizo_names
    luvdisc = _kaizo_spec(n, "Luvdisc", "Tackle", 16)
    gible = _kaizo_spec(n, "Gible", FANG, 13)
    s = make_state(t, [luvdisc], [gible])
    s = s._replace(stats=s.stats.at[0, 0, 4].set(int(s.stats[1, 0, 4]) + 5))
    acts = jnp.zeros(2, jnp.int32)
    laws = laws_of(t, s, acts)
    assert np.asarray(laws.relevant.fang_flinch).tolist() == [False, False]
    assert bool(laws.relevant.secondary[1, 0])
    _, probs = enumerate_outcomes(laws)
    assert abs(math.fsum(probs) - 1) < 1e-12
