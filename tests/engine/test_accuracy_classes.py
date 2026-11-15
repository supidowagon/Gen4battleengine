"""Mid-turn accuracy (K3b2 Task 1, docs/kaizo/k3b2-rules.md §0.2b-c): the decomp computes a
move's hit rate at its own check (``BattleControllerPlayer_CheckMoveHitAccuracy``,
battle_controller_player.c:2865-2979), with the stages of that moment, so a faster
Sand-Attack or Double Team changes a later move's hit chance in the same turn.
``outcome.hit[side, 0]`` is the roll's interval class over every threshold the move can
have this turn (``outcome.hit_thresholds``), and ``step`` compares it with the live one."""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.engine.enumerate import (
    SamplingOnly,
    enumerate_outcomes,
    hit_class_law,
    outcome_laws,
)
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat
from nuzline.engine.outcome import (
    Outcome,
    empty_outcome,
    hit_class,
    hit_threshold,
    sample_outcome,
)
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.strategies import legal_action_pair, outcome_law_state
from tools.effect_map import DECOMP_EFFECT_TO_ID

CHUNK = 1024
HYPOTHESIS = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
}
_laws = jax.jit(outcome_laws)
_step = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_sample = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))


def laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, actions))


def hit_law(tables, state, actions, side) -> dict[int, int]:
    """``value -> weight`` (out of 100) of ``hit[side, 0]`` in the enumeration."""
    laws = laws_of(tables, state, actions)
    values, weights, den = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
    assert den == 100
    return dict(zip(values, weights, strict=True))


def stepped(tables, state, actions, outcomes: Outcome):
    """``step`` of every row of ``outcomes`` (numpy fields with a leading axis), in fixed
    ``CHUNK`` batches; the result's fields are numpy with the same leading axis."""
    n = len(outcomes.hit)
    parts = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = Outcome(*(np.asarray(f)[pad] for f in outcomes))
        out = _step(tables, state, actions, rows)
        parts.append([np.asarray(f)[: len(idx)] for f in out])
    return [np.concatenate(fs) for fs in zip(*parts)]


def state_keys(fields) -> list[bytes]:
    n = len(fields[0])
    flat = np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in fields], axis=1
    )
    return [flat[i].tobytes() for i in range(n)]


def merged(keys, probs) -> dict[bytes, Fraction | float]:
    out: dict = {}
    for k, p in zip(keys, probs):
        out[k] = out.get(k, 0) + p
    return out


def sand_attack_first(tables, names):
    """Roark's Geodude, faster, uses Sand-Attack; the player's Monferno then Scratches
    (accuracy 100)."""
    s = roark_state(tables, names, opp_moves=["Sand-Attack"], opp_faster=True)
    return s, A("Scratch", "Sand-Attack", s, names)


def hit_probability(tables, state, actions, side) -> float:
    """The exact probability that ``side``'s move connects (the foe loses HP)."""
    outcomes, probs = enumerate_outcomes(laws_of(tables, state, actions))
    out = stepped(tables, state, actions, outcomes)
    hp = out[state._fields.index("hp")]
    foe = 1 - side
    target = int(state.active[foe])
    landed = hp[:, foe, target] < int(state.hp[foe, target])
    return math.fsum(probs[landed])


# --- the bug ---------------------------------------------------------------------


@pytest.mark.slow
def test_sand_attack_lowers_a_later_move_this_turn(tables, names):
    s, acts = sand_attack_first(tables, names)
    # HitRateByStage[5] = 75/100 (include/data/hit_rate_stages.h) at accStages -1
    # (bcp.c:2873); Scratch's own rate was 100 at the start of the turn.
    assert int(hit_threshold(tables, s, acts)[0]) == 100
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12
    # The stage drop is there after the turn (the move hit: Sand-Attack has accuracy 100).
    out = step(tables, s, acts, empty_outcome())
    assert int(out.stages[0, Stat.ACC]) == -1

    # A slower Sand-Attack comes after Scratch: the turn-start rate stands.
    slow = roark_state(tables, names, opp_moves=["Sand-Attack"])
    assert hit_law(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) == {1: 100}
    assert (
        abs(hit_probability(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) - 1) < 1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize("move", ["Sand-Attack", "Double Team"])
def test_accuracy_change_equals_bruteforce(tables, names, move):
    # K3b2 final review I1: the brute force cuts the hit law at every threshold variant
    # (outcome.hit_thresholds), not only the possible ones, so a variant the gated laws
    # wrongly drop shows as a difference here.
    from tests.engine.test_enumerate import assert_equals_bruteforce

    s = roark_state(tables, names, opp_moves=[move], opp_faster=True)
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))  # no crits: a smaller product
    width, brute = assert_equals_bruteforce(tables, s, A("Scratch", move, s, names))
    assert width <= brute


def test_double_team_raises_evasion_for_a_later_move(tables, names):
    s = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    acts = A("Scratch", "Double Team", s, names)
    # evaStages = 6 - statBoosts = -1 (bcp.c:2874): HitRateByStage[5] again.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12


@pytest.mark.slow
def test_nature_power_call_is_seen_by_hit_thresholds(tables, names):
    # On GREAT_MARSH a faster foe's Nature Power calls Mud Bomb (Mud-Slap's family:
    # SECONDARY_STAT_FOE, accuracy -1, effect_script_0073.s): the later Scratch can meet
    # stage -1 (hit_thresholds reads the called move, order.called_move).
    marsh = 10  # TERRAIN_GREAT_MARSH (nuzline.data.trainer.TERRAIN_NAMES)
    s = roark_state(tables, names, opp_moves=["Nature Power"], opp_faster=True)
    s = s._replace(terrain=jnp.asarray(marsh, s.terrain.dtype))
    acts = A("Scratch", "Nature Power", s, names)
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    o = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, s, acts, o._replace(hit=o.hit.at[0, 0].set(76)))
    assert int(out.stages[0, Stat.ACC]) == -1 and int(out.hp[1, 0]) == int(s.hp[1, 0])


@pytest.mark.slow
def test_class_values_against_live_threshold(tables, names):
    s, acts = sand_attack_first(tables, names)
    for value, connects in ((1, True), (75, True), (76, False), (0, False)):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        out = step(tables, s, acts, o)
        assert (int(out.hp[1, 0]) < int(s.hp[1, 0])) == connects, value
    # Without the Sand-Attack, 76 is a hit at 100.
    plain = roark_state(tables, names, opp_faster=True)
    o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(76))
    out = step(tables, plain, A("Scratch", "Leer", plain, names), o)
    assert int(out.hp[1, 0]) < int(plain.hp[1, 0])


# --- the law ---------------------------------------------------------------------


def test_hit_class_law_examples():
    assert hit_class_law([95]) == ([0, 1], [5, 95], 100)
    assert hit_class_law([100]) == ([1], [100], 100)
    assert hit_class_law([166, 100]) == ([1], [100], 100)
    assert hit_class_law([100, 75]) == ([1, 76], [75, 25], 100)
    assert hit_class_law([95, 71]) == ([0, 1, 72], [5, 71, 24], 100)
    assert hit_class_law([0]) == ([0], [100], 100)


@pytest.mark.slow
@settings(max_examples=40, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_class_law_sums_to_one(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    for side in (0, 1):
        possible = laws.hit_possible[side]
        assert possible[0], "the turn-start threshold is always possible"
        assert int(laws.hit_thresholds[side, 0]) == int(hit_threshold(tables, s, acts)[side])
        values, weights, den = hit_class_law(laws.hit_thresholds[side][possible])
        assert sum(weights) == den == 100 and min(weights) > 0
        assert len(set(values)) == len(values)
    try:
        _, probs = enumerate_outcomes(laws)
    except SamplingOnly:
        return
    assert abs(math.fsum(probs) - 1) < 1e-12


def test_no_change_keeps_two_classes(tables, names):
    s = roark_state(tables, names)
    acts = A("Scratch", "Tackle", s, names)  # Tackle 95, Scratch 100, Leer-free turn
    assert hit_law(tables, s, acts, 1) == {0: 5, 1: 95}
    assert hit_law(tables, s, acts, 0) == {1: 100}
    # A faster Sand-Attack on the other side does not split a turn that cannot change it.
    slow = roark_state(tables, names, player_moves=["Sand-Attack"], opp_faster=True)
    acts = A("Sand-Attack", "Tackle", slow, names)
    assert hit_law(tables, slow, acts, 1) == {0: 5, 1: 95}


def test_white_herb_restores_the_stage_before_the_check(tables, names):
    s, acts = sand_attack_first(tables, names)
    mon = int(s.active[0])
    herb = int(np.flatnonzero(np.asarray(tables.item_effect) == ItemId.WHITE_HERB)[0])
    s = s._replace(item=s.item.at[0, mon].set(herb))
    # Over-approximated: the herb's reset to 0 is a threshold the move may have.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    for value in (1, 76):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        assert_matches_reference(tables, s, acts, o)


# --- both engines ----------------------------------------------------------------


def test_accuracy_classes_match_reference(tables, names):
    cases = [sand_attack_first(tables, names)]
    dt = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    cases.append((dt, A("Scratch", "Double Team", dt, names)))
    # The player's Sand-Attack first, then Geodude's Tackle (95): three classes.
    sa = roark_state(tables, names, player_moves=["Sand-Attack"])
    cases.append((sa, A("Sand-Attack", "Tackle", sa, names)))
    for s, acts in cases:
        laws = laws_of(tables, s, acts)
        for side in (0, 1):
            values, _, _ = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
            for value in [*values, 0, 1, 100]:
                o = empty_outcome()._replace(hit=empty_outcome().hit.at[side, 0].set(value))
                assert_matches_reference(tables, s, acts, o)
    assert hit_law(tables, sa, cases[-1][1], 1) == {0: 5, 1: 71, 72: 24}


# --- sampling --------------------------------------------------------------------


def test_sampled_hit_frequencies_match_law(tables, names):
    s = roark_state(tables, names, player_moves=["Sand-Attack"])
    acts = A("Sand-Attack", "Tackle", s, names)
    law = hit_law(tables, s, acts, 1)
    assert law == {0: 5, 1: 71, 72: 24}
    n = 20_000
    keys = jax.random.split(jax.random.key(3), n)
    drawn = np.asarray(_sample(keys, tables, s, acts).hit)[:, 1, 0]
    assert set(np.unique(drawn).tolist()) <= set(law)
    for value, weight in law.items():
        p = weight / 100
        assert abs(np.mean(drawn == value) - p) < 5 * math.sqrt(p * (1 - p) / n), value


def test_hit_class_maps_every_roll():
    thresholds = jnp.array([95, 71, 100, 30], jnp.int32)
    possible = jnp.array([True, True, True, False])
    values, weights, _ = hit_class_law([95, 71, 100])
    got = [int(hit_class(u, thresholds, possible)) for u in range(1, 101)]
    assert [got.count(v) for v in values] == weights
    # Each class value decides every possible threshold exactly as its rolls do.
    for u, v in zip(range(1, 101), got, strict=True):
        for t in (95, 71, 100):
            assert (u <= t) == (0 < v <= t)


# --- soundness: classes equal the raw rolls ----------------------------------------


@pytest.mark.slow
@settings(max_examples=30, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_classes_equal_raw_rolls(tables, names, data):
    """With every other field fixed, the class enumeration of both sides' rolls gives the
    same merged children as all 100 x 100 raw rolls ``u`` (stored as is: ``u <= t`` is
    ``0 < u <= t``). A threshold missing from ``hit_thresholds`` would split a class's
    rolls between different states and fail this."""
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    assume(laws.relevant.hit[:, 0].any())
    base = jax.tree.map(
        np.asarray, sample_outcome(jax.random.key(data.draw(st.integers(0, 99))), tables, s, acts)
    )

    def rows(values0, values1):
        grid = [(a, b) for a in values0 for b in values1]
        hit = np.repeat(base.hit[None], len(grid), axis=0)
        hit[:, 0, 0] = [a for a, _ in grid]
        hit[:, 1, 0] = [b for _, b in grid]
        fields = {
            name: np.repeat(np.asarray(f)[None], len(grid), axis=0)
            for name, f in zip(Outcome._fields, base)
        }
        fields["hit"] = hit
        return Outcome(**fields), grid

    laws_side = [
        dict(zip(*hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])[:2]))
        for side in (0, 1)
    ]
    classes, grid = rows(list(laws_side[0]), list(laws_side[1]))
    probs = [Fraction(laws_side[0][a] * laws_side[1][b], 10_000) for a, b in grid]
    want = merged(state_keys(stepped(tables, s, acts, classes)), probs)
    raw, grid = rows(list(range(1, 101)), list(range(1, 101)))
    got = merged(state_keys(stepped(tables, s, acts, raw)), [Fraction(1, 10_000)] * len(grid))
    assert got == want


# (move the faster opponent uses, which side's stage, its value, White Herb, Simple)
FIXED_CASES = [
    *(
        ("Sand-Attack", 0, stage, herb, simple)
        for stage in (-6, -2, 0, 3)
        for herb, simple in ((False, False), (True, False), (True, True))
    ),
    *(("Double Team", 1, stage, True, False) for stage in (-3, 0, 6)),
]


@pytest.mark.parametrize(("move", "holder", "stage", "herb", "simple"), FIXED_CASES)
def test_accuracy_classes_equal_raw_rolls_fixed(tables, names, move, holder, stage, herb, simple):
    """Hand-written cases of ``test_accuracy_classes_equal_raw_rolls``: the faster
    opponent's Sand-Attack (on the player's accuracy stage) or Double Team (on its own
    evasion stage), from several starting stages, with a White Herb and Simple on the
    Pokemon whose stage changes; the player's Scratch then checks accuracy."""
    s = roark_state(tables, names, opp_moves=[move], opp_faster=True)
    s = s.with_stages(0, acc=stage) if holder == 0 else s.with_stages(1, eva=stage)
    mon = int(s.active[holder])
    if herb:
        white_herb = int(np.flatnonzero(np.asarray(tables.item_effect) == ItemId.WHITE_HERB)[0])
        s = s._replace(item=s.item.at[holder, mon].set(white_herb))
    if simple:
        s = s._replace(ability=s.ability.at[holder, mon].set(AbilityId.SIMPLE))
    acts = A("Scratch", move, s, names)
    law = hit_law(tables, s, acts, 0)

    def rows(values):
        base = empty_outcome()
        return Outcome(
            *(np.repeat(np.asarray(f)[None], len(values), axis=0) for f in base)
        )._replace(hit=np.repeat(np.asarray(base.hit)[None], len(values), axis=0).copy())

    classes, raw = rows(list(law)), rows(list(range(1, 101)))
    classes.hit[:, 0, 0] = list(law)
    raw.hit[:, 0, 0] = range(1, 101)
    want = merged(
        state_keys(stepped(tables, s, acts, classes)), [Fraction(w, 100) for w in law.values()]
    )
    got = merged(state_keys(stepped(tables, s, acts, raw)), [Fraction(1, 100)] * 100)
    assert got == want
    for value in [*law, 0]:
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        assert_matches_reference(tables, s, acts, o)


# Every EffectId when hit_thresholds was written, classified. A new one fails
# test_every_accuracy_or_evasion_writer_is_seen_by_hit_thresholds until it is added to
# one of these (and, if it writes Stat.ACC / Stat.EVA, to outcome.hit_thresholds).
_STAT_KINDS = frozenset(
    {
        EffectId.STAT_SELF,
        EffectId.STAT_FOE,
        EffectId.SECONDARY_STAT_SELF,
        EffectId.SECONDARY_STAT_FOE,
    }
)
# Effects that write no accuracy or evasion stage (effects.py; hooks.py's White Herb and
# Intimidate are an item and an ability): RAISE_ALL_STATS_HIT raises Atk, Def, Spe, SpA,
# SpD (subscript_boost_all_stats.s), CURSE changes Spe, Atk, Def (subscript_curse_normal.s),
# the rest change no stage. FANG_PARALYZE_FLINCH (K3b2 Task 4) paralyzes and flinches
# (subscript_paralyze_or_flinch.s): no stage. PRESENT (Task 5) damages or heals the target
# (effect_script_0122.s, subscript_recover_hp.s): no stage. PROTECT (Task 6) sets the
# user's protecting flag and chain (BtlCmd_TryProtection): no stage. ATTRACT (Task 7) sets
# the target's infatuated (BtlCmd_TryAttract): no stage. CAMOUFLAGE (Task 10) writes the
# user's types (BtlCmd_TryCamouflage) and TRI_ATTACK a burn, freeze or paralysis
# (effect_script_0036.s): no stage. NATURE_POWER writes none itself; the move it calls can
# (Mud Bomb's accuracy drop at GREAT_MARSH), and hit_thresholds reads that called move's
# effect (order.called_move): test_nature_power_call_is_seen_by_hit_thresholds. ROAR (Task 11)
# drags the target out (BtlCmd_TryWhirlwind): the newcomer's stages are fresh, and it runs no
# accuracy check this turn (UpdateAfterSwitch ends its action), nor does anyone after Roar.
_NO_ACC_EVA_WRITE = (frozenset(EffectId(i) for i in range(40)) - _STAT_KINDS) | {
    EffectId.FANG_PARALYZE_FLINCH,
    EffectId.PRESENT,
    EffectId.PROTECT,
    EffectId.ATTRACT,
    EffectId.CAMOUFLAGE,
    EffectId.NATURE_POWER,
    EffectId.TRI_ATTACK,
    EffectId.ROAR,
    # E2 batch 3: writes the weather, not a stage; hit_thresholds follows it (Sand Veil, Snow
    # Cloak) as its weather variant.
    EffectId.WEATHER,
    # E2 batch 4: no accuracy or evasion stage. Secret Power's terrain effect can lower accuracy
    # (or evasion), but hit_thresholds reads it resolved (order.move_effects): as
    # SECONDARY_STAT_FOE, a _SEEN pair.
    EffectId.DOUBLE_POWER,
    EffectId.HP_POWER,
    EffectId.WEIGHT_POWER,
    EffectId.HALVE_HP,
    EffectId.OHKO,
    EffectId.FALSE_SWIPE,
    EffectId.SWAGGER,
    EffectId.DRAGON_DANCE,
    EffectId.ROOST,
    EffectId.TAILWIND,
    EffectId.SPIKES,
    EffectId.TOXIC_SPIKES,
    EffectId.DESTINY_BOND,
    EffectId.SECRET_POWER,
    EffectId.COUNTER,
    # E2 batch 5: the invokers write none themselves; hit_thresholds reads the move they call
    # (outcome_laws' and sample_outcome's ``moves``, order.used_moves).
    EffectId.SLEEP_TALK,
    EffectId.METRONOME,
    EffectId.COPYCAT,
    # Uproar (a lock and a field bit) and Future Sight (a stored hit) write no stage.
    EffectId.UPROAR,
    EffectId.FUTURE_SIGHT,
    # E2 batch 6: U-turn and Baton Pass write no stage on the field mid-law: a Baton Pass
    # newcomer's passed stages arrive at the pause, whose resumed half draws its own laws from the
    # paused state (B6-6, B6-10).
    EffectId.U_TURN,
    EffectId.BATON_PASS,
    # E2 batches 7-8: traps, seals, a lock, counts; Lock-On sets no stage (it makes the locker's
    # next move skip the roll, outcome._rate), and its user has used its move this turn.
    EffectId.INGRAIN,
    EffectId.MEAN_LOOK,
    EffectId.IMPRISON,
    EffectId.LOCK_ON,
    EffectId.PERISH_SONG,
    EffectId.YAWN,
    # E2 batch 9: Follow Me and Helping Hand change nothing in singles; Belly Drum writes Attack
    # only (subscript_belly_drum.s). Bulk Up is DRAGON_DANCE (Attack and Defense); Minimize is
    # STAT_SELF on Stat.EVA, a _SEEN pair.
    EffectId.FOLLOW_ME,
    EffectId.HELPING_HAND,
    EffectId.BELLY_DRUM,
    # E2 batch 10: hits whose recoil, status or crash write no stage (B10-1, B10-2); Substitute
    # writes the user's HP and substitute only (B10-4).
    EffectId.RECOIL_STATUS,
    EffectId.CRASH,
    EffectId.SUBSTITUTE,
    # Endeavor and Pain Split write HP only (B10-5).
    EffectId.ENDEAVOR,
    EffectId.PAIN_SPLIT,
    # Knock Off takes an item only (B10-6).
    EffectId.KNOCK_OFF,
}
# (effect, stat) pairs hit_thresholds follows: they change a later foe move's threshold.
_SEEN = {
    (EffectId.STAT_FOE, Stat.ACC),
    (EffectId.SECONDARY_STAT_FOE, Stat.ACC),
    (EffectId.STAT_SELF, Stat.EVA),
    (EffectId.SECONDARY_STAT_SELF, Stat.EVA),
}
# Pairs that cannot: the user's own accuracy and the foe's evasion only feed the user's
# own move, which has already run (Sweet Scent lowers the foe's evasion).
_IRRELEVANT = {
    (EffectId.STAT_SELF, Stat.ACC),
    (EffectId.SECONDARY_STAT_SELF, Stat.ACC),
    (EffectId.STAT_FOE, Stat.EVA),
    (EffectId.SECONDARY_STAT_FOE, Stat.EVA),
}


def test_every_accuracy_or_evasion_writer_is_seen_by_hit_thresholds(tables):
    """``outcome.hit_thresholds`` follows the stat effects on ``Stat.ACC`` / ``Stat.EVA``
    (``move_param[m, 0]``). An effect added later that writes them (Minimize, Haze, Psych
    Up, ...) fails here until it is classified and, if it can, followed there."""
    assert set(EffectId) == _NO_ACC_EVA_WRITE | _STAT_KINDS, (
        "a new EffectId: say whether it writes Stat.ACC / Stat.EVA (outcome.hit_thresholds)"
    )
    pairs = {
        (effect, params[0])
        for effect, params in DECOMP_EFFECT_TO_ID.values()
        if effect in _STAT_KINDS and params[0] in (Stat.ACC, Stat.EVA)
    }
    effect, param = np.asarray(tables.move_effect), np.asarray(tables.move_param)
    for move in range(len(effect)):
        e, stat = EffectId(int(effect[move])), int(param[move, 0])
        if e in _STAT_KINDS and stat in (Stat.ACC, Stat.EVA):
            pairs.add((e, stat))
    assert pairs, "no accuracy / evasion stat effect found"
    assert pairs <= _SEEN | _IRRELEVANT, pairs - _SEEN - _IRRELEVANT
    assert (EffectId.STAT_FOE, Stat.ACC) in pairs and (EffectId.STAT_SELF, Stat.EVA) in pairs
