"""Present (K3b2 Task 5, ``EffectId.PRESENT``; docs/kaizo/k3b2-rules.md §3).

Decomp (external/pokeplatinum):

* ``res/battle/scripts/effects/effect_script_0122.s``: ``Present _004`` then ``CalcCrit``,
  ``CalcDamage`` (the damage path); ``_004`` (the heal path) sets the direct effect
  ``ON_HIT | PRESENT_HEAL`` and ``SYSCTL_IGNORE_IMMUNITIES`` and runs no ``CalcCrit`` or
  ``CalcDamage``.
* ``BtlCmd_Present`` (src/battle/battle_script.c:5789-5806): ``rnd = RandNext() & 0xFF``
  (:5793); ``rnd < 255 * 40 / 100`` (102) -> power 40, ``< 255 * 70 / 100`` (178) -> 80,
  ``< 255 * 80 / 100`` (204) -> 120, else ``hpCalcTemp = BattleSystem_Divide(DEFENDING_MON
  .maxHP, 4)`` and the jump (:5802-5803). The law is (102, 76, 26, 52) / 256, exact.
* The script runs before ``TryMove`` (battle_controller_player.c:3209-3216), so the draw
  comes before the accuracy check; under the project's i.i.d. convention that only says
  when the draw exists.
* ``BattleControllerPlayer_CheckTypeChart`` (battle_controller_player.c:2350-2358) skips the
  chart under ``SYSCTL_IGNORE_IMMUNITIES``: the heal reaches a Ghost type (and Wonder Guard,
  Levitate: their checks live in ``BattleSystem_ApplyTypeChart``, battle_lib.c:2600, :2649).
  Protect (``CheckMoveHitOverrides``, :3007-3012) does not read that flag.
* ``BattleSystem_TriggerPrimaryEffect`` (battle_lib.c:1485-1490): the ON_HIT direct effect
  runs only without ``MOVE_STATUS_NO_EFFECTS`` (a miss gives no heal).
* ``subscript_present_heal.s``: the attack message, a Heal Block check (unsupported move),
  then ``subscript_recover_hp.s``: at full HP "X's HP is full!" (:6, :_036), else the HP
  update (capped at max HP) and "X regained health!".
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.commands import ai_damage, is_damaging
from nuzline.engine.damage import damage_for_roll
from nuzline.engine.enumerate import (
    _LAWS,
    INPUT_NEUTRAL_EFFECTS,
    NON_TABLE_DAMAGE_EFFECTS,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import AbilityId, EffectId, Type
from nuzline.engine.observe import observe
from nuzline.engine.outcome import PRESENT_HEAL, Outcome, empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import step, step_events
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
PRESENT = "Present"
_SPE = 4  # stats[..., 0..4] are Atk..Spe
POWERS = (40, 80, 120)


def _state(tables, names, *, opp="Tackle", order="player", lucky=False):
    """Delibird (``Present``) against Cranidos (``opp``, Sturdy: no Mold Breaker); ``order``
    is who is faster ("player", "opp") or "tie"; ``lucky`` sets Lucky Chant on both sides
    (no crits: a smaller brute force)."""
    s = roark_state(
        tables,
        names,
        player_species="Delibird",
        player_moves=[PRESENT],
        opp_species="Cranidos",
        opp_moves=[opp],
    )
    # Delibird's own abilities (Vital Spirit, Hustle) are unsupported: none.
    s = _ability(_ability(s, 1, AbilityId.STURDY), 0, AbilityId.NONE)
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))
    if lucky:
        s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    return s


def _ability(s, side, ability):
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _o(present=0, crit=0, roll=15, hit=1, side=0) -> Outcome:
    o = empty_outcome()
    return o._replace(
        present=o.present.at[side].set(present),
        crit=o.crit.at[side, 0].set(crit),
        roll=o.roll.at[side, 0].set(roll),
        hit=o.hit.at[side, 0].set(hit),
    )


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


def _hurt(s, low=3):
    """Cranidos at ``max_hp // low`` HP."""
    return s.replace_hp(1, 0, max(int(s.max_hp[1, 0]) // low, 1))


# --- mapping and decomp scripts -------------------------------------------------------------


def test_effect_122_maps_in_both_games(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[122] == (EffectId.PRESENT, (0, 0, 0))
    for t, n, accuracy in ((tables, names, 90), (kaizo_tables, kaizo_names, 100)):
        move = n.move(PRESENT)
        assert int(t.move_effect[move]) == EffectId.PRESENT
        assert np.asarray(t.move_param[move]).tolist() == [0, 0, 0]
        # The table power is 1 in both games: the battle draws 40/80/120, the AI skips it.
        assert (int(t.move_power[move]), int(t.move_accuracy[move])) == (1, accuracy)
        assert int(t.move_type[move]) == Type.NORMAL


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_do_what_the_engine_does():
    effects = DECOMP / "res/battle/scripts/effects"
    subs = DECOMP / "res/battle/scripts/subscripts"
    s122 = (effects / "effect_script_0122.s").read_text("utf-8")
    steps = [ln.strip() for ln in s122.splitlines() if ln.startswith("    ")]
    assert steps[:4] == ["Present _004", "CalcCrit", "CalcDamage", "End"]
    assert "MOVE_SIDE_EFFECT_ON_HIT|MOVE_SUBSCRIPT_PTR_PRESENT_HEAL" in steps[5]
    assert "BTLVAR_SIDE_EFFECT_FLAGS_DIRECT" in steps[5]
    assert "SYSCTL_IGNORE_IMMUNITIES" in steps[6]
    assert not any("Calc" in step_ for step_ in steps[4:])  # no crit or damage on the heal
    heal = (subs / "subscript_present_heal.s").read_text("utf-8")
    assert "MOVE_STATUS_DID_NOT_HIT" in heal and "Call BATTLE_SUBSCRIPT_RECOVER_HP" in heal
    recover = (subs / "subscript_recover_hp.s").read_text("utf-8")
    assert "CompareMonDataToVar OPCODE_EQU, BTLSCR_MSG_TEMP, BATTLEMON_CUR_HP" in recover
    assert "PokemonsHPIsFull" in recover
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8").splitlines()
    assert "int rnd = BattleSystem_RandNext(battleSys) & 0xFF;" in script[5792]
    assert "rnd < (255 * 40 / 100)" in script[5794]
    assert "rnd < (255 * 70 / 100)" in script[5796]
    assert "rnd < (255 * 80 / 100)" in script[5798]
    assert "BattleSystem_Divide(DEFENDING_MON.maxHP, 4)" in script[5801]
    bcp = (DECOMP / "src/battle/battle_controller_player.c").read_text("utf-8").splitlines()
    assert "SYSCTL_IGNORE_IMMUNITIES) == FALSE" in bcp[2354]


# --- the law -----------------------------------------------------------------------------


def test_law_is_exact_over_the_low_byte():
    from nuzline.engine.outcome import present_kind

    kinds = np.asarray(present_kind(jnp.arange(256)))
    assert np.bincount(kinds, minlength=4).tolist() == [102, 76, 26, 52]
    assert (255 * 40 // 100, 255 * 70 // 100, 255 * 80 // 100) == (102, 178, 204)
    assert PRESENT_HEAL == 3
    values, weights, den = _LAWS["present"](None, 0)
    assert (values, weights, den) == ([0, 1, 2, 3], [102, 76, 26, 52], 256)


def test_sampled_present_follows_the_law(tables, names):
    s = _state(tables, names)
    acts = A(PRESENT, "Tackle", s, names)
    n = 40000
    keys = jax.random.split(jax.random.key(3), n)
    o = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))(keys, tables, s, acts)
    got = np.bincount(np.asarray(o.present)[:, 0], minlength=4) / n
    for k, w in enumerate((102, 76, 26, 52)):
        p = w / 256
        assert abs(got[k] - p) <= 5 * math.sqrt(p * (1 - p) / n), k
    # fold_in(key, 17): every earlier field keeps its samples.
    from nuzline.engine.outcome import present_kind

    one = sample_outcome(jax.random.key(0), tables, s, acts)
    rand = jax.random.randint(jax.random.fold_in(jax.random.key(0), 17), (2,), 0, 256)
    assert np.asarray(one.present).tolist() == np.asarray(present_kind(rand)).tolist()


# --- the damage path ---------------------------------------------------------------------


def test_damage_path_is_a_normal_hit_at_the_drawn_power(tables, names):
    """For every crit and roll, Present's damage at present k equals a PLAIN Normal
    physical move of table power 40/80/120 (the same move with its table changed)."""
    s = _state(tables, names)
    move = names.move(PRESENT)
    for k, power in enumerate(POWERS):
        plain = tables._replace(
            move_effect=tables.move_effect.at[move].set(EffectId.PLAIN),
            move_power=tables.move_power.at[move].set(power),
        )
        for crit, roll in itertools.product((False, True), (0, 7, 15)):
            want = int(damage_for_roll(plain, s, 0, move, crit, roll))
            got = int(damage_for_roll(tables, s, 0, move, crit, roll, present=k))
            assert got == want > 0, (k, crit, roll)
    # Without a drawn value (the turn-start table, CalcMoveDamage's inPower 0): table power.
    one = tables._replace(move_effect=tables.move_effect.at[move].set(EffectId.PLAIN))
    assert int(damage_for_roll(tables, s, 0, move, False, 15)) == int(
        damage_for_roll(one, s, 0, move, False, 15)
    )


def test_damage_path_in_step(tables, names):
    s = _state(tables, names)
    acts = A(PRESENT, "Tackle", s, names)
    hp = int(s.hp[1, 0])
    move = names.move(PRESENT)
    for k in range(3):
        for crit in (0, 1):
            out = assert_matches_reference(tables, s, acts, _o(k, crit, roll=4))
            want = int(damage_for_roll(tables, s, 0, move, bool(crit), 4, present=k))
            assert int(out.hp[1, 0]) == max(hp - want, 0)
            ev = _events(tables, s, acts, _o(k, crit, roll=4))
            assert ev["crit"][0, 0] == crit and ev["present_heal"][0] == 0
    # A Ghost type is immune to the damage path (the chart runs: power 1 != 0).
    ghost = s._replace(types=s.types.at[1, 0].set(Type.GHOST))
    for k in range(3):
        out = assert_matches_reference(tables, ghost, acts, _o(k, 1))
        assert int(out.hp[1, 0]) == hp


def test_damage_path_keeps_the_resist_berry(tables, names):
    # Only the heal path sets SYSCTL_IGNORE_IMMUNITIES (effect_script_0122.s _004), so a
    # Chilan Berry (any Normal hit, subscript_type_resist_berry.s) halves the damage path
    # and is eaten; the heal path deals no damage and leaves it.
    s = _state(tables, names)
    chilan = names.item("Chilan Berry")
    s = s._replace(item=s.item.at[1, 0].set(chilan))
    acts = A(PRESENT, "Leer", s, names)
    hp = int(s.hp[1, 0])
    want = int(damage_for_roll(tables, s, 0, names.move(PRESENT), False, 15, present=2))
    out = assert_matches_reference(tables, s, acts, _o(2))
    assert int(out.hp[1, 0]) == hp - max(want // 2, 1) and int(out.item[1, 0]) == 0
    out = assert_matches_reference(tables, _hurt(s), acts, _o(PRESENT_HEAL))
    assert int(out.item[1, 0]) == chilan


# --- the heal path -----------------------------------------------------------------------


def test_heal_path_heals_a_quarter_of_the_targets_max_hp(tables, names):
    s = _hurt(_state(tables, names))
    acts = A(PRESENT, "Leer", s, names)
    mx, hp = int(s.max_hp[1, 0]), int(s.hp[1, 0])
    out = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL))
    assert int(out.hp[1, 0]) == min(hp + max(mx // 4, 1), mx)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])  # the user is untouched
    assert _events(tables, s, acts, _o(PRESENT_HEAL))["present_heal"].tolist() == [1, 0]
    # Capped at max HP.
    near = s.replace_hp(1, 0, mx - 1)
    assert int(assert_matches_reference(tables, near, acts, _o(PRESENT_HEAL)).hp[1, 0]) == mx
    # Divide(maxHP, 4) is at least 1.
    tiny = s._replace(max_hp=s.max_hp.at[1, 0].set(3)).replace_hp(1, 0, 1)
    assert int(assert_matches_reference(tables, tiny, acts, _o(PRESENT_HEAL)).hp[1, 0]) == 2


def test_heal_reads_no_crit_or_roll(tables, names):
    s = _hurt(_state(tables, names))
    acts = A(PRESENT, "Leer", s, names)
    base = step(tables, s, acts, _o(PRESENT_HEAL, 0, 15))
    for crit, roll in itertools.product((0, 1), (0, 9)):
        out = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL, crit, roll))
        for x, y in zip(base, out, strict=True):
            assert (np.asarray(x) == np.asarray(y)).all()
        assert _events(tables, s, acts, _o(PRESENT_HEAL, crit, roll))["crit"][0].sum() == 0


def test_heal_at_full_hp_changes_nothing_and_says_so(tables, names):
    s = _state(tables, names)
    acts = A(PRESENT, "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL))
    assert int(out.hp[1, 0]) == int(s.max_hp[1, 0])
    assert _events(tables, s, acts, _o(PRESENT_HEAL))["present_heal"].tolist() == [2, 0]
    # A full-HP Focus Sash holder keeps its sash: no damage, so UpdateHP never runs.
    sash = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
    out = assert_matches_reference(tables, sash, acts, _o(PRESENT_HEAL))
    assert int(out.item[1, 0]) == names.item("Focus Sash")


def _sitrus_delibird(tables, names, opp):
    """Delibird (faster) at 3 HP holding a Sitrus Berry against a full-HP Cranidos."""
    s = _state(tables, names, opp=opp)
    s = s._replace(item=s.item.at[0, 0].set(names.item("Sitrus Berry")))
    return s.replace_hp(0, 0, 3)


def test_full_hp_heal_runs_no_item_check(tables, names):
    """subscript_recover_hp.s _036 sets MOVE_STATUS_NO_MORE_WORK; CheckMoveFailure
    (bcp.c:3313-3315) then skips AfterMoveEffects (bcp.c:3638-3769), so Delibird's Sitrus
    (ATTACKER_ITEM) is not checked after a heal on a full-HP foe, and Cranidos's Tackle KOs
    it. After a damage path it is checked and eaten first."""
    s = _sitrus_delibird(tables, names, "Tackle")
    acts = A(PRESENT, "Tackle", s, names)
    sitrus = names.item("Sitrus Berry")
    out = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL))
    assert int(out.hp[0, 0]) == 0 and int(out.item[0, 0]) == sitrus
    out = assert_matches_reference(tables, s, acts, _o(0))
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == sitrus
    # A heal on a hurt foe ("regained health") reaches AfterMoveEffects: eaten too.
    hurt = _hurt(s)
    assert int(assert_matches_reference(tables, hurt, acts, _o(PRESENT_HEAL)).item[0, 0]) == 0


@pytest.mark.parametrize("opp", ["Leer", "Rock Polish"])
def test_skipped_sitrus_is_eaten_later_in_the_turn(tables, names, opp):
    # The berry is not lost: Cranidos's Leer runs its own AfterMoveEffects with Delibird as
    # the defender (DEFENDER_ITEM, bcp.c:3691); after a Rock Polish (its defender is its
    # user) the end-of-turn MON_COND_CHECK_STATE_USE_ITEM (bcp.c:1362) eats it. Either way
    # it is gone by the end of the turn and Delibird has 3 + Sitrus's heal.
    s = _sitrus_delibird(tables, names, opp)
    out = assert_matches_reference(tables, s, A(PRESENT, opp, s, names), _o(PRESENT_HEAL))
    mx = int(s.max_hp[0, 0])
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == names.item("Sitrus Berry")
    assert int(out.hp[0, 0]) == 3 + max(mx // 4, 1)


def test_full_hp_recover_runs_no_item_check(tables, names):
    """Recover (effect_script_0032.s -> subscript_recover_half_max_hp.s:16 ->
    subscript_recover_hp.s) at full HP: the same NO_MORE_WORK path, so no item check after
    it. A confused Onix at exactly its self-hit's damage, holding a Sitrus Berry, then
    faints to the self-hit; the Sitrus is not eaten after the Recover."""
    from nuzline.engine.damage import confusion_damage

    s = roark_state(
        tables,
        names,
        player_species="Delibird",
        player_moves=["Recover"],
        opp_species="Onix",
        opp_moves=["Tackle"],
    )
    s = _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.STURDY)
    s = s._replace(
        stats=s.stats.at[0, 0, _SPE].set(int(s.stats[1, 0, _SPE]) + 1),
        item=s.item.at[1, 0].set(names.item("Sitrus Berry")),
        confusion=s.confusion.at[1].set(3),
    )
    hit = int(confusion_damage(tables, s, 1, 15))
    assert 0 < hit and hit * 2 <= int(s.max_hp[1, 0])  # the Sitrus would trigger
    s = s.replace_hp(1, 0, hit)
    acts = A("Recover", "Tackle", s, names)
    o = empty_outcome()._replace(confusion_hit=jnp.asarray([0, 1], jnp.int8))
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.hp[1, 0]) == 0 and int(out.item[1, 0]) == names.item("Sitrus Berry")
    # Below full HP the Recover reaches AfterMoveEffects (DEFENDER_ITEM is the user itself
    # in the decomp; the engine also checks the foe's item there): Onix eats its Sitrus
    # first and survives the self-hit.
    hurt = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2)
    out = assert_matches_reference(tables, hurt, acts, o)
    assert int(out.item[1, 0]) == 0 and int(out.hp[1, 0]) > 0


def test_heal_ignores_immunities(tables, names):
    # SYSCTL_IGNORE_IMMUNITIES: the type chart is skipped, so a Ghost type is healed.
    s = _hurt(_state(tables, names))
    ghost = s._replace(types=s.types.at[1, 0].set(Type.GHOST))
    acts = A(PRESENT, "Leer", ghost, names)
    out = assert_matches_reference(tables, ghost, acts, _o(PRESENT_HEAL))
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) + max(int(s.max_hp[1, 0]) // 4, 1)


def test_no_heal_on_a_miss(tables, names):
    s = _hurt(_state(tables, names))
    acts = A(PRESENT, "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL, hit=0))
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    ev = _events(tables, s, acts, _o(PRESENT_HEAL, hit=0))
    assert ev["missed"][0] == 1 and ev["present_heal"][0] == 0


def test_heal_before_the_target_acts(tables, names):
    # Delibird first heals a low-HP Cranidos, whose Flail then has less power.
    s = _hurt(_state(tables, names, opp="Flail"), low=8)
    acts = A(PRESENT, "Flail", s, names)
    healed = assert_matches_reference(tables, s, acts, _o(PRESENT_HEAL))
    hurt = assert_matches_reference(tables, s, acts, _o(0, hit=0))  # Present missed
    assert int(healed.hp[1, 0]) > int(hurt.hp[1, 0])
    assert int(healed.hp[0, 0]) > int(hurt.hp[0, 0])  # a weaker Flail


def test_observation_tells_the_paths_apart(tables, names):
    # Into a Ghost at full HP the state is the same on every path; the messages differ.
    s = _state(tables, names)
    ghost = s._replace(types=s.types.at[1, 0].set(Type.GHOST))
    acts = A(PRESENT, "Leer", ghost, names)
    obs = []
    for k in (0, PRESENT_HEAL):
        after = step(tables, ghost, acts, _o(k))
        obs.append(np.asarray(observe(tables, ghost, acts, _o(k), after)))
    assert (obs[0] != obs[1]).any()


def test_matches_reference_over_every_draw(tables, names):
    """Both engines agree over present x crit x roll x hit, in both orders, into a hurt,
    full-HP, Ghost, Focus Sash and low-HP target."""
    base = _state(tables, names)
    setups = {
        "plain": base,
        "slow": _state(tables, names, order="opp"),
        "hurt": _hurt(base),
        "ghost": _hurt(base._replace(types=base.types.at[1, 0].set(Type.GHOST))),
        "sash": base._replace(item=base.item.at[1, 0].set(names.item("Focus Sash"))),
        "one_hp": base.replace_hp(1, 0, 1),
    }
    for s in setups.values():
        acts = A(PRESENT, "Tackle", s, names)
        for k, crit, roll, hit in itertools.product(range(4), (0, 1), (0, 15), (0, 1)):
            assert_matches_reference(tables, s, acts, _o(k, crit, roll, hit))


# --- the trainer AI ------------------------------------------------------------------------


def test_ai_skips_present_damage(tables, names):
    # FlagMoveDamageScore and TrainerAI_CalcAllDamage compute damage only for power > 1
    # (trainer_ai.c:1035-1036, :2827-2828); Present is in neither sAltPowerMoveEffects nor
    # has power > 1, so the AI never scores its damage.
    move = names.move(PRESENT)
    assert not bool(is_damaging(tables, move))
    s = _state(tables, names)
    for roll in (0, 15):
        assert int(ai_damage(tables, s, move, roll)) == 0


# --- laws and enumeration ------------------------------------------------------------------


def test_present_relevance(tables, names):
    s = _state(tables, names)
    acts = A(PRESENT, "Tackle", s, names)
    laws = laws_of(tables, s, acts)
    assert np.asarray(laws.relevant.present).tolist() == [True, False]
    assert bool(laws.relevant.crit[0, 0]) and bool(laws.relevant.roll[0, 0])
    assert np.asarray(laws.reads.present).tolist() == [True, False]
    # Not for a switch, nor in loose mode's reads.
    assert not np.asarray(laws_of(tables, s, A(4 + 1, "Tackle", s, names)).relevant.present).any()
    assert not np.asarray(laws_of(tables, s, acts, loose=True).reads.present).any()


def test_crit_and_roll_read_only_on_the_damage_path(tables, names):
    """(present, crit, roll) is one joint factor of 3 * 32 + 1 values per hit class."""
    s = _state(tables, names, opp="Leer")
    acts = A(PRESENT, "Leer", s, names)
    laws = laws_of(tables, s, acts)
    outcomes, probs, drawn = enumerate_drawn(laws)
    present = np.asarray(outcomes.present)[:, 0]
    hit = np.asarray(outcomes.hit)[:, 0, 0]
    read_p = np.asarray(drawn.present)[:, 0]
    # The draw is behind the accuracy roll (90 %), and crit / roll behind present != heal.
    assert (read_p == (hit != 0)).all()
    for name in ("crit", "roll"):
        read = np.asarray(getattr(drawn, name))[:, 0, 0]
        assert (read == (read_p & (present != PRESENT_HEAL))).all(), name
    assert len(probs) == 1 + 3 * 32 + 1  # a miss, the damage paths, the heal
    heal = read_p & (present == PRESENT_HEAL)
    assert math.fsum(probs[heal]) == pytest.approx(0.9 * 52 / 256, abs=1e-15)
    for k, w in enumerate((102, 76, 26)):
        assert math.fsum(probs[read_p & (present == k)]) == pytest.approx(0.9 * w / 256)


def test_present_hits_are_never_compressed(tables, names, kaizo_tables, kaizo_names):
    # Present's damage is not its turn-start table (power 1): the hit stays (crit, roll)
    # although its joint factor gates crit / roll on present (only the allowlist decides).
    assert EffectId.PRESENT in NON_TABLE_DAMAGE_EFFECTS
    for order in ("player", "opp", "tie"):
        s = _state(tables, names, opp="Leer", order=order)
        laws = laws_of(tables, s, A(PRESENT, "Leer", s, names))
        assert not np.asarray(laws.compress)[:, 0].any(), order
        outcomes, _, drawn = enumerate_drawn(laws)
        # Every damage path draws all 32 (crit, roll) pairs, in each order branch (a tie
        # also reads end_speed_tie, so count distinct values, not rows).
        read = np.asarray(drawn.roll)[:, 0, 0]
        columns = (
            np.asarray(outcomes.speed_tie)[read],
            np.asarray(outcomes.present)[read, 0],
            np.asarray(outcomes.crit)[read, 0, 0],
            np.asarray(outcomes.roll)[read, 0, 0],
        )
        seen = {tuple(int(x) for x in row) for row in zip(*columns, strict=True)}
        assert len(seen) == 3 * 32 * (2 if order == "tie" else 1), order
    # Kaizo (100 %): the same.
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t, [_kaizo_spec(n, "Delibird", PRESENT, 16)], [_kaizo_spec(n, "Gible", "Leer", 13)]
    )
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(int(s.stats[0, 0, _SPE]) - 1))
    laws = laws_of(t, s, jnp.zeros(2, jnp.int32))
    assert not np.asarray(laws.compress)[:, 0].any()
    assert len(enumerate_outcomes(laws)[1]) == 3 * 32 + 1


def test_foe_hit_after_a_present(tables, names):
    # Present heals or hurts the foe: only its HP changes, an input of Flail and the pinch
    # abilities only (the HP clause). A plain Tackle after it stays compressed.
    assert EffectId.PRESENT in INPUT_NEUTRAL_EFFECTS
    s = _state(tables, names)
    laws = laws_of(tables, s, A(PRESENT, "Tackle", s, names))
    assert bool(np.asarray(laws.compress)[0, 1, 0])
    flail = _hurt(_state(tables, names, opp="Flail"), low=8)
    laws = laws_of(tables, flail, A(PRESENT, "Flail", flail, names))
    assert not bool(np.asarray(laws.compress)[0, 1, 0])
    torrent = _ability(_hurt(_state(tables, names, opp="Water Gun")), 1, AbilityId.TORRENT)
    laws = laws_of(tables, torrent, A(PRESENT, "Water Gun", torrent, names))
    assert not bool(np.asarray(laws.compress)[0, 1, 0])
    # Cranidos first: its Tackle comes before the Present.
    slow = _state(tables, names, order="opp")
    laws = laws_of(tables, slow, A(PRESENT, "Tackle", slow, names))
    assert bool(np.asarray(laws.compress)[0, 1, 0])


def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _hurt(_state(tables, names, opp="Leer"))
    acts = A(PRESENT, "Leer", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


SETUPS = (
    None,
    "hurt",  # the heal changes the target's HP
    "flail",  # a low-HP Flail after the heal: a weaker Flail
    "torrent",  # a pinch Water Gun after the heal: no longer in the pinch
    "ghost",
    "sash",
    "confused",  # the Present user's roll[0, 0] is also the self-hit's
)
ORDERS = ("player", "opp", "tie")
OPP_MOVES = {
    None: ("Tackle", "Headbutt", "Thunder Wave", PRESENT),
    "hurt": ("Tackle", PRESENT),
    "flail": ("Flail",),
    "torrent": ("Water Gun",),
    "ghost": ("Tackle",),
    "sash": ("Tackle",),
    "confused": ("Tackle",),
}
CASES = [(setup, opp, order) for setup in SETUPS for opp in OPP_MOVES[setup] for order in ORDERS]


@pytest.mark.parametrize(("setup", "opp", "order"), CASES)
def test_present_equals_bruteforce(tables, names, setup, opp, order):
    """The gated enumeration (the joint (present, crit, roll) factor, the hit gate, the
    sure flinch, damage classes of the foe's hit) equals the loose brute force merged by
    state and observation, Present first, second and at a speed tie."""
    s = _state(tables, names, opp=opp, order=order, lucky=True)
    if setup == "hurt":
        s = _hurt(s)
    elif setup == "flail":
        s = _hurt(s, low=8)
    elif setup == "torrent":
        s = _ability(_hurt(s), 1, AbilityId.TORRENT)
    elif setup == "ghost":
        s = _hurt(s._replace(types=s.types.at[1, 0].set(Type.GHOST)))
    elif setup == "sash":
        s = s._replace(item=s.item.at[1, 0].set(names.item("Focus Sash")))
    elif setup == "confused":
        s = _hurt(s._replace(confusion=s.confusion.at[0].set(3)))
    acts = A(PRESENT, opp, s, names)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width < brute


@pytest.mark.parametrize("order", ORDERS)
def test_present_crits_equal_bruteforce(tables, names, order):
    # With crits on the Present user's side (Lucky Chant only on Delibird's side blocks
    # Cranidos's crits): the crit gate on the heal path.
    s = _hurt(_state(tables, names, opp="Flail", order=order), low=8)
    s = s._replace(lucky_chant=s.lucky_chant.at[0].set(5))
    width, brute = assert_equals_bruteforce(tables, s, A(PRESENT, "Flail", s, names))
    assert width < brute


@pytest.mark.parametrize("delibird_first", [True, False])
def test_kaizo_delibird_vs_gible_thunder_fang(kaizo_tables, kaizo_names, delibird_first):
    # Rules §10.2: Delibird's Present against Gible's Thunder Fang, Kaizo tables (Present
    # 100 %, Thunder Fang 90 BP). Lucky Chant on Delibird's side keeps the brute force small.
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t, [_kaizo_spec(n, "Delibird", PRESENT, 16)], [_kaizo_spec(n, "Gible", "Thunder Fang", 13)]
    )
    spe = int(s.stats[0, 0, _SPE])
    s = s._replace(
        stats=s.stats.at[1, 0, _SPE].set(spe - 1 if delibird_first else spe + 1),
        lucky_chant=s.lucky_chant.at[0].set(5),
        ability=s.ability.at[:, 0].set(AbilityId.STURDY),
    )
    s = s.replace_hp(1, 0, max(int(s.max_hp[1, 0]) // 3, 1))
    width, brute = assert_equals_bruteforce(t, s, jnp.zeros(2, jnp.int32))
    assert width < brute
