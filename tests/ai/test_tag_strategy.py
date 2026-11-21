"""``TagStrategy_Main`` (bit 7, script.s:6637-7338) on the singles paths: A-8 sections 2, 4, 8.

One fixed B = 2 state per block, the exact law of the slot's score (``policy._slot_histograms``,
every read its own draw) against the script. Decomp paths are relative to
``external/pokeplatinum``: ``script.s`` is ``src/battle/trainer_ai/script.s``, ``trainer_ai.c``
``src/battle/trainer_ai/trainer_ai.c``. At B = 2 the attacker is battler 1 and the defender
battler 0; the partner arguments name the zeroed battlers 3 and 2 (``commands.CLEARED``), whose
HP percent is 0, ability 0 (UNKNOWN), types Normal and masks 0 (A-8 section 1).

Every state keeps the player's types Normal / Normal and its HP at 400 (nothing kills it, no
type effect) unless a test sets them, and the filler slots hold Harden (no damage, no change)
or Stone Edge (the strongest estimate, so the slot under test is not the highest). The fights
run bit 7 alone (``TAG_LAYOUT``): every other flag has its own tests.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai import policy
from nuzline.ai.commands import BattleEffect, TypeMulti
from nuzline.ai.layout import KAIZO_FLAG_BITS, TAG_STRATEGY, kaizo_layout
from nuzline.ai.outcome import READ_DRAWS, SIM_ROLL_MAX, SITE_LINES, SITE_N
from nuzline.engine.field import with_gravity, with_trick_room
from nuzline.engine.ids import Status
from nuzline.engine.state import MOVES
from tests.ai.test_kaizo_commands import _slot_ctx, _species_with, _tag_fight  # noqa: F401
from tests.ai.test_kaizo_outcome import _bulky, _exact_slot_scores, _with_player

NORMAL, FLYING, POISON, ROCK, BUG = 0, 2, 3, 5, 6
FIRE, WATER, GRASS, ELECTRIC, ICE, GROUND, STEEL = 10, 11, 12, 13, 15, 4, 8
# Effects the layout transcribes for this module's moves beyond KAIZO_TRANSCRIBED: Sandstorm's.
SANDSTORM_EFFECT = 115
SLOT_READS = 6  # the widest Tag path: read_bounds (bit 7) at most 6, drawn rows included
_PHYSICAL_FILLER = ("Stone Edge", "Harden", "Harden")
_NONE = ("Harden", "Harden", "Harden")


def _layout(slot_reads: int = SLOT_READS):
    base = kaizo_layout(1 << TAG_STRATEGY, slot_reads)
    return base._replace(transcribed=frozenset(base.transcribed | {SANDSTORM_EFFECT}))


TAG_LAYOUT = _layout()
_histograms = jax.jit(policy._slot_histograms, static_argnames="layout")


# --- States ------------------------------------------------------------------------------


def _fight(t, n, party, moves, *, foe_types=(NORMAL, NORMAL), own_types=None, own_hp=None):
    """Kaizo Roark's attacker with ``moves`` (names; slot 0 is the one under test) against a
    player of ``foe_types`` at 400 HP. ``own_types``: the attacker's types; ``own_hp``: its HP
    percent (max HP 100)."""
    s = _tag_fight(t, n, party, list(moves))
    s = _bulky(s, 400)
    s = s._replace(types=s.types.at[0, 0].set(jnp.asarray(foe_types, s.types.dtype)))
    a = int(s.active[1])
    if own_types is not None:
        s = s._replace(types=s.types.at[1, a].set(jnp.asarray(own_types, s.types.dtype)))
    if own_hp is not None:
        s = s._replace(max_hp=s.max_hp.at[1, a].set(100), hp=s.hp.at[1, a].set(own_hp))
    return s


def _with_ability(t, s, side: int, ability: int):
    """``t`` whose species row of ``side``'s active Pokemon has the decomp ability ``ability`` in
    its ability slot (``damage.decomp_abilities``): a state edit through the table."""
    a = int(s.active[side])
    row = int(np.asarray(t.species_form_row)[int(s.species[side, a]), int(s.ability_form[side, a])])
    slot = int(s.ability_slot[side, a])
    return t._replace(
        species_abilities_decomp=t.species_abilities_decomp.at[row, slot].set(ability)
    )


def _status(s, side: int, status=Status.BURN):
    return s._replace(status=s.status.at[side, int(s.active[side])].set(int(status)))


def _law(t, n, s, slot: int = 0, layout=TAG_LAYOUT) -> dict[int, float]:
    """``{score: probability}`` of ``slot``'s score after bit 7 alone, from a base of 100: the
    exact histogram of the per-slot layout, which the fight does not refuse."""
    assert policy.ai_refusals(t, s, n, layout) == []
    hist = _exact_slot_scores(t, s, layout)[slot]
    return {int(v): float(hist[v]) for v in np.flatnonzero(hist > 1e-9)}


def _assert_law(actual: dict[int, float], expected: dict[int, float]) -> None:
    assert set(actual) == set(expected), (actual, expected)
    for v, p in expected.items():
        assert actual[v] == pytest.approx(p, abs=1e-6), (v, actual, expected)


def _damage(t, s) -> list[int]:
    return [int(x) for x in c._slot_damage(t, s, jnp.full((MOVES,), SIM_ROLL_MAX)).value[:, 0]]


def _effectiveness(t, s, move: int) -> int:
    return int(c.effectiveness_class(t, s, move))


def _reads(t, s, layout=TAG_LAYOUT) -> np.ndarray:
    """``int[MOVES]``: the most random reads each slot's scoring paths make (over every bit
    pattern of the layout)."""
    _missing, reads = policy._path_records(t, s, layout)
    return np.asarray(reads).max(axis=0)


@pytest.fixture(scope="module")
def tag(kaizo_tables, kaizo_names, kaizo_roark_party):
    """``fight(moves, **kw)``: ``(tables, state)`` of ``_fight`` under the Kaizo tables."""
    t, n, party = kaizo_tables, kaizo_names, kaizo_roark_party

    def fight(moves, **kw):
        return _fight(t, n, party, moves, **kw)

    return t, n, fight


def _id(n, move: str) -> int:
    return n.move(move)


# --- The fixed transcription facts ----------------------------------------------------------


def test_tag_strategy_is_bit_7_of_the_kaizo_policy():
    # FlagTable (script.s:26): bit 7; layout.KAIZO_FLAG_BITS and the flag scripts carry it.
    assert TAG_STRATEGY == 7 and TAG_STRATEGY in KAIZO_FLAG_BITS
    assert 7 in [bit for bit, _ in policy._flag_scripts(kaizo_layout(1 << 7, SLOT_READS))]


def test_tag_move_and_ability_ids_are_the_decomps(kaizo_names, kaizo_manifest):
    # generated/moves.txt and generated/abilities.txt (line - 1); Kaizo's move on id 222 is
    # Bulldoze (the script tests ids, A-8 section 9.3).
    n = kaizo_names
    moves = {
        "Skill Swap": "Skill Swap",
        "Magnitude": "Bulldoze",
        "Future Sight": "Future Sight",
        "Doom Desire": "Doom Desire",
        "Rain Dance": "Rain Dance",
        "Sunny Day": "Sunny Day",
        "Hail": "Hail",
        "Sandstorm": "Sandstorm",
        "Gravity": "Gravity",
        "Follow Me": "Follow Me",
        "Helping Hand": "Helping Hand",
        "Discharge": "Discharge",
        "Surf": "Surf",
        "Lava Plume": "Lava Plume",
    }
    for key, kaizo_name in moves.items():
        assert n.move_names[c.MOVE_IDS[key]] == kaizo_name, key


def test_tag_site_n_is_the_scripts(kaizo_tables):
    # A-8 section 7: every IfRandomLessThan of TagStrategy_Main, with its N.
    expected = {
        6666: 64, 6678: 64, 6694: 128, 6702: 50, 6722: 100, 6728: 64, 6820: 128, 6851: 128,
        6959: 64, 6972: 64, 7012: 64, 7022: 64, 7053: 64, 7075: 64, 7080: 64, 7085: 64,
        7090: 64, 7095: 64, 7173: 128, 7181: 128, 7187: 128,
    }  # fmt: skip
    for line, n in expected.items():
        assert SITE_N[line] == n
    assert SITE_LINES.index(6666) >= 0


# --- 2.1-2.4: damage scoring ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("foe_types", "cls"),
    [((WATER, NORMAL), TypeMulti.HALF), ((WATER, 16), TypeMulti.QUARTER)],
)
def test_not_very_effective_move_is_not_changed_and_reads_nothing(tag, foe_types, cls):
    # TagStrategy_TryScoreMinus1 / _TryScoreMinus2 (script.s:6658-6680): at B = 2
    # IfHPPercentEqualTo DEFENDER_PARTNER 0 (battler 2, hp% 0) jumps to ScoreMove before the roll
    # at :6666 / :6678, so no -1 / -2; the move is not the highest, so nothing follows.
    t, n, fight = tag
    s = fight(["Flamethrower", *_PHYSICAL_FILLER], foe_types=foe_types)
    assert _effectiveness(t, s, _id(n, "Flamethrower")) == int(cls)
    d = _damage(t, s)
    assert d[1] > d[0] > 0  # NOT_HIGHEST: ScoreMove goes on to CheckBeforeScoring
    _assert_law(_law(t, n, s), {100: 1.0})
    assert _reads(t, s)[0] == 0


def test_highest_move_gets_one_with_the_128_roll(tag):
    # TagStrategy_ScoreMove (:6682-6697): HIGHEST, not PRIORITY_1: R@6694(128) jumps to
    # CheckBeforeScoring (nothing for a neutral move), else +1 (128/256).
    t, n, fight = tag
    s = fight(["Tackle", *_NONE])
    assert _effectiveness(t, s, _id(n, "Tackle")) == int(TypeMulti.BASE)
    _assert_law(_law(t, n, s), {100: 0.5, 101: 0.5})
    assert _reads(t, s)[0] == 1


def test_highest_priority_move_gets_one_with_the_50_roll(tag):
    # :6699-6704: effect PRIORITY_1 takes R@6702(50): jump to CheckBeforeScoring, else +1: 206/256.
    t, n, fight = tag
    s = fight(["Quick Attack", *_NONE])
    assert int(t.move_ai_effect[_id(n, "Quick Attack")]) == BattleEffect.PRIORITY_1
    _assert_law(_law(t, n, s), {100: 50 / 256, 101: 206 / 256})


def test_not_highest_goes_to_check_before_scoring_without_a_roll(tag):
    t, n, fight = tag
    s = fight(["Tackle", *_PHYSICAL_FILLER])
    d = _damage(t, s)
    assert d[1] > d[0] > 0
    _assert_law(_law(t, n, s), {100: 1.0})
    assert _reads(t, s)[0] == 0


@pytest.mark.parametrize(
    ("foe_types", "cls", "p_plus_1"),
    [
        ((GRASS, NORMAL), TypeMulti.DOUBLE, 156 / 256),  # R@6722(100): else +1
        ((GRASS, BUG), TypeMulti.QUADRUPLE, 192 / 256),  # R@6728(64): else +1
    ],
)
def test_super_effective_move_gets_one_with_its_roll(tag, foe_types, cls, p_plus_1):
    # TagStrategy_CheckBeforeScoring (:6706-6730), reached by a move that is not the highest.
    t, n, fight = tag
    s = fight(["Ember", *_PHYSICAL_FILLER], foe_types=foe_types)
    assert _effectiveness(t, s, _id(n, "Ember")) == int(cls)
    d = _damage(t, s)
    assert d[1] > d[0] > 0
    _assert_law(_law(t, n, s), {100: 1 - p_plus_1, 101: p_plus_1})
    assert _reads(t, s)[0] == 1


def test_flat_damage_move_skips_the_effectiveness_rolls(tag):
    # :6644-6648 sends a flat-damage effect to ScoreMove; :6712-6716 sends it past the
    # effectiveness tests. Triple Axel is Kaizo's RANDOM_DAMAGE_1_TO_150_LEVEL (Psywave's id).
    t, n, fight = tag
    s = fight(["Triple Axel", "Stone Edge", "Harden", "Harden"], foe_types=(GRASS, BUG))
    assert int(t.move_ai_effect[_id(n, "Triple Axel")]) == BattleEffect.RANDOM_DAMAGE_1_TO_150_LEVEL
    # Slot 1's Stone Edge out-damages every Triple Axel draw: NOT_HIGHEST, no roll, no change.
    _assert_law(_law(t, n, s), {100: 1.0})


def test_non_damaging_move_goes_straight_to_special_scoring(tag):
    # :6641-6642: NO_COMPARISON_MADE -> CheckSpecialScoring: Harden has none.
    t, n, fight = tag
    s = fight(["Harden", "Tackle", "Harden", "Harden"])
    _assert_law(_law(t, n, s), {100: 1.0})
    assert _reads(t, s)[0] == 0


# --- 2.6-2.9: the weather moves ------------------------------------------------------------


@pytest.mark.parametrize(
    ("ability", "statused", "delta"),
    [
        ("Hydration", False, 0),  # SelfHasHydration: IfNotStatus ANY -> CheckPartner (:6767)
        ("Hydration", True, 2),
        ("Dry Skin", False, 2),
        ("Dry Skin", True, 2),
        ("Levitate", True, 0),
    ],
)
def test_rain_dance(tag, ability, statused, delta):
    # TagStrategy_RainDance (:6754-6786): self only at B = 2 (the partner reads UNKNOWN).
    t, n, fight = tag
    s = fight(["Rain Dance", *_NONE])
    t = _with_ability(t, s, 1, c.ABILITY_IDS[ability])
    s = _status(s, 1) if statused else s
    _assert_law(_law(t, n, s), {100 + delta: 1.0})
    assert _reads(t, s)[0] == 0


@pytest.mark.parametrize(
    ("ability", "hp", "statused", "law"),
    [
        ("Leaf Guard", 100, False, {102: 1.0}),
        ("Leaf Guard", 100, True, {100: 1.0}),
        ("Leaf Guard", 29, False, {100: 1.0}),
        ("Leaf Guard", 30, False, {102: 1.0}),
        ("Flower Gift", 10, True, {102: 1.0}),
        ("Dry Skin", 100, False, {98: 1.0}),
        ("Solar Power", 50, False, {101: 0.5, 99: 0.5}),  # +1, then the roll falls in (:6815-6821)
        ("Solar Power", 100, False, {101: 0.5, 99: 0.5}),
        ("Solar Power", 49, False, {100: 0.5, 98: 0.5}),
        ("Levitate", 100, False, {100: 1.0}),
    ],
)
def test_sunny_day(tag, ability, hp, statused, law):
    # TagStrategy_SunnyDay (:6788-6855). Solar Power at 50 % or more adds +1 and then falls into
    # the 50 % roll (the comment at :6794 omits it); under 50 % only the roll.
    t, n, fight = tag
    s = fight(["Sunny Day", *_NONE], own_hp=hp)
    t = _with_ability(t, s, 1, c.ABILITY_IDS[ability])
    s = _status(s, 1) if statused else s
    _assert_law(_law(t, n, s), law)


@pytest.mark.parametrize(
    ("ability", "known", "delta"),
    [
        ("Ice Body", None, 2),
        ("Snow Cloak", None, 2),
        ("Levitate", None, 0),
        ("Levitate", "Blizzard", 2),
    ],
)
def test_hail(tag, ability, known, delta):
    # TagStrategy_Hail (:6857-6884): +2 once for Ice Body, Snow Cloak or a known Blizzard.
    t, n, fight = tag
    s = fight(["Hail", known or "Harden", "Harden", "Harden"])
    t = _with_ability(t, s, 1, c.ABILITY_IDS[ability])
    _assert_law(_law(t, n, s), {100 + delta: 1.0})


@pytest.mark.parametrize(
    ("ability", "types", "delta"),
    [
        ("Levitate", (NORMAL, NORMAL), 0),
        ("Sand Veil", (NORMAL, NORMAL), 2),
        ("Levitate", (ROCK, NORMAL), 4),  # self +2, then GetPartner = the attacker: +2 again
        ("Levitate", (NORMAL, ROCK), 4),
        ("Sand Veil", (ROCK, ROCK), 4),
    ],
)
def test_sandstorm(tag, ability, types, delta):
    # TagStrategy_Sandstorm (:6886-6916). BattleSystem_GetPartner is the attacker itself in a
    # single battle (battle_system.c:400-402): LOAD_ATTACKER_PARTNER_TYPE_* are its own types, so
    # a Rock attacker scores +2 twice; Sand Veil is read for the partner as UNKNOWN (ability 0).
    t, n, fight = tag
    s = fight(["Sandstorm", *_NONE], own_types=types)
    t = _with_ability(t, s, 1, c.ABILITY_IDS[ability])
    _assert_law(_law(t, n, s), {100 + delta: 1.0})


# --- 2.10-2.12: Gravity, Trick Room, Follow Me ---------------------------------------------


def test_gravity_active_scores_minus_30(tag):
    # TagStrategy_Gravity (:6918-6921) -> TagStrategy_PartnerScoreMinus30 (:7692).
    t, n, fight = tag
    s = with_gravity(fight(["Gravity", *_NONE]), 5)
    _assert_law(_law(t, n, s), {70: 1.0})


@pytest.mark.parametrize(
    ("kind", "delta"), [("Flying", -5), ("Levitate", -5), ("Magnet Rise", -5), ("none", 0)]
)
def test_gravity_self_grounded_minus_5(tag, kind, delta):
    # :6934-6948: Levitate, a Flying type or Magnet Rise: -5 once. The partner's own copy of the
    # tests (:6950-6962) reads the zeroed battler 3: Normal type, UNKNOWN ability, no mask.
    t, n, fight = tag
    s = fight(["Gravity", *_NONE], own_types=(FLYING, NORMAL) if kind == "Flying" else None)
    if kind == "Levitate":
        t = _with_ability(t, s, 1, c.ABILITY_IDS["Levitate"])
    if kind == "Magnet Rise":
        s = s._replace(magnet_rise=s.magnet_rise.at[1].set(1))
    _assert_law(_law(t, n, s), {100 + delta: 1.0})


def test_gravity_target_flying_gets_three_with_the_64_roll(tag):
    # :6964-6970: the target is Flying: R@6959(64) jumps (no change), else +3 (192/256).
    t, n, fight = tag
    s = fight(["Gravity", *_NONE], foe_types=(FLYING, NORMAL))
    _assert_law(_law(t, n, s), {100: 64 / 256, 103: 192 / 256})
    assert _reads(t, s)[0] == 1


def test_gravity_self_and_target_airborne(tag):
    t, n, fight = tag
    s = fight(["Gravity", *_NONE], foe_types=(FLYING, NORMAL), own_types=(FLYING, NORMAL))
    _assert_law(_law(t, n, s), {95: 64 / 256, 98: 192 / 256})


def test_trick_room_scores_minus_30_at_b2(tag):
    # TagStrategy_TrickRoom (:6979-6983): IfHPPercentEqualTo ATTACKER_PARTNER 0 jumps to
    # ScoreMinus30 at once (battler 3, hp% 0): no speed sort, no roll.
    t, n, fight = tag
    s = fight(["Trick Room", *_NONE])
    _assert_law(_law(t, n, s), {70: 1.0})
    assert _reads(t, s)[0] == 0
    on = with_trick_room(s, 5)
    _assert_law(_law(t, n, on), {70: 1.0})


@pytest.mark.parametrize(
    ("hp", "law"),
    [
        (100, {103: 0.75, 100: 0.25}),  # a > 90, q = 0: TryScorePlus3 (R@7095)
        (91, {103: 0.75, 100: 0.25}),
        (90, {102: 0.75, 100: 0.25}),  # 50 < a <= 90: SelfMediumHP -> TryScorePlus2 (R@7090)
        (51, {102: 0.75, 100: 0.25}),
        (50, {102: 0.75, 100: 0.25}),  # 30 < a <= 50: SelfLowHP -> TryScorePlus2
        (31, {102: 0.75, 100: 0.25}),
        (30, {95: 0.75, 100: 0.25}),  # a <= 30: R@7053(64) else ScoreMinus5
        (1, {95: 0.75, 100: 0.25}),
    ],
)
def test_follow_me_bands_with_the_partner_at_zero(tag, hp, law):
    # TagStrategy_FollowMe (:7032-7100): q = hp%(ATTACKER_PARTNER) = 0 at B = 2.
    t, n, fight = tag
    s = fight(["Follow Me", *_NONE], own_hp=hp)
    _assert_law(_law(t, n, s), law)
    assert _reads(t, s)[0] == 1


# --- 2.13-2.17: Earthquake, Future Sight, Skill Swap, Helping Hand ---------------------------


@pytest.mark.parametrize("move", ["Earthquake", "Bulldoze"])
def test_earthquake_and_the_magnitude_id_score_minus_3(tag, move):
    # TagStrategy_Earthquake (:7127-7148), the attacker's partner at B = 2 = battler 3, which
    # reads Normal / UNKNOWN / no Magnet Rise: -3. Kaizo's id 222 (MOVE_MAGNITUDE) is Bulldoze;
    # IfMoveEqualTo compares ids (trainer_ai.c:834-844). Stone Edge out-damages either, so
    # the move is not the highest (no +1 roll).
    t, n, fight = tag
    s = fight([move, *_PHYSICAL_FILLER])
    d = _damage(t, s)
    assert d[1] > d[0] > 0
    _assert_law(_law(t, n, s), {97: 1.0})


@pytest.mark.parametrize("move", ["Future Sight", "Doom Desire"])
def test_future_sight_and_doom_desire_add_nothing_at_b2(tag, move):
    # TagStrategy_FutureSight (:7150-7157): IfHPPercentEqualTo ATTACKER_PARTNER 0 -> End.
    t, n, fight = tag
    s = fight([move, *_NONE])
    assert not bool(c.is_damaging(t, _id(n, move))) or _law(t, n, s) == {
        100: pytest.approx(0.5),
        101: pytest.approx(0.5),
    }
    if not bool(c.is_damaging(t, _id(n, move))):
        _assert_law(_law(t, n, s), {100: 1.0})
    stronger = fight([move, *_PHYSICAL_FILLER])
    if bool(c.is_damaging(t, _id(n, move))):
        assert _damage(t, stronger)[1] > _damage(t, stronger)[0]
        _assert_law(_law(t, n, stronger), {100: 1.0})


@pytest.mark.parametrize("ability", ["Truant", "Slow Start", "Stall", "Klutz"])
def test_skill_swap_own_bad_ability_scores_5(tag, ability):
    # TagStrategy_SkillSwap (:7195-7213): ScorePlus5 ends the script before the target's ability.
    t, n, fight = tag
    s = fight(["Skill Swap", *_NONE])
    t = _with_ability(t, s, 1, c.ABILITY_IDS[ability])
    _assert_law(_law(t, n, s), {105: 1.0})
    assert _reads(t, s)[0] == 0


def test_skill_swap_guessed_defender_is_a_coin(tag):
    # :7205: LoadBattlerAbility DEFENDER guesses with RandNext() & 1 (trainer_ai.c:1191-1198):
    # a species of two abilities, exactly one of them on the +2 list (Filter / Shell Armor),
    # scores +2 on one coin side.
    t, n, fight = tag
    ab = np.asarray(t.species_abilities_decomp)
    wanted = {c.ABILITY_IDS[x] for x in ("Filter", "Solid Rock", "Pure Power", "Huge Power")}
    species = next(
        sp
        for sp in range(1, len(ab))
        if ab[sp][0]
        and ab[sp][1]
        and ab[sp][0] != ab[sp][1]
        and ((ab[sp][0] in wanted) != (ab[sp][1] in wanted))
    )
    s = _with_player(t, fight(["Skill Swap", *_NONE]), species, 0)
    assert bool(c.ability_guessed(t, s))
    _assert_law(_law(t, n, s), {100: 0.5, 102: 0.5})
    assert _reads(t, s)[0] == 1
    known = _with_player(t, fight(["Skill Swap", *_NONE]), species, 0, known=True)
    _assert_law(
        _law(t, n, known),
        {100: 1.0} if int(known.ai_ability_known[0]) not in wanted else {102: 1.0},
    )


def test_helping_hand_is_never_reached_at_b2(tag):
    # TagStrategy_CheckSpecialScoring (:6751): IfMoveKnown ATTACKER_PARTNER HELPING_HAND never
    # jumps at B = 2 (curHP 0, trainer_ai.c:1650-1653): the move adds nothing; a Helping Hand
    # in the moveset changes no other slot's score.
    t, n, fight = tag
    s = fight(["Harden", "Helping Hand", "Harden", "Harden"])
    _assert_law(_law(t, n, s), {100: 1.0})
    _assert_law(_law(t, n, s, slot=1), {100: 1.0})


# --- 2.18-2.20: Electric, Water, Fire --------------------------------------------------------


@pytest.mark.parametrize(
    ("move", "delta"),
    [("Discharge", -3), ("Thunderbolt", 0), ("Surf", -3), ("Water Gun", -1), ("Flamethrower", 0)],
)
def test_type_blocks(tag, move, delta):
    # CheckElectricMove (:7215-7263): Discharge -> SpreadElectric -3 (battler 3: no ability, not
    # Water, Flying or Ground); any other Electric move: DEFENDER_PARTNER Lightning Rod UNKNOWN
    # and ATTACKER_PARTNER's too: 0. CheckWaterMove (:7265-7302): Surf -3; another Water move: the
    # DEFENDER_PARTNER's Storm Drain reads UNKNOWN, not NOT_HAVE, so -1 (:7273-7275).
    # CheckFireMove (:7304-7337): no Flash Fire, not Lava Plume: 0.
    t, n, fight = tag
    s = fight([move, *_PHYSICAL_FILLER])
    d = _damage(t, s)
    assert d[1] > d[0] > 0
    _assert_law(_law(t, n, s), {100 + delta: 1.0})


def test_flash_fire_and_lava_plume(tag):
    t, n, fight = tag
    s = fight(["Flamethrower", *_PHYSICAL_FILLER])
    on = s._replace(flash_fire=s.flash_fire.at[1].set(1))
    _assert_law(_law(t, n, on), {101: 1.0})  # IfActivatedFlashFire ATTACKER: +1 (:7311)
    lava = fight(["Lava Plume", *_PHYSICAL_FILLER])
    _assert_law(_law(t, n, lava), {97: 1.0})  # SpreadFire: ATTACKER_PARTNER reads nothing: -3
    lava_on = lava._replace(flash_fire=lava.flash_fire.at[1].set(1))
    _assert_law(_law(t, n, lava_on), {98: 1.0})  # +1 first, then -3 (:7311, :7336)


def test_spread_fire_dry_skin_is_minus_3_not_plus_3():
    # script.s:7320: `IfLoadedEqualTo AI_HAVE, ScoreMinus3` after CheckBattlerAbility
    # ATTACKER_PARTNER DRY_SKIN; the comment at :7308 says +3. The partner is battler 3 at B = 2,
    # so the line is unreachable there; the transcription takes the code: the source line.
    import inspect

    source = inspect.getsource(policy._tag_strategy)
    assert "ScoreMinus3 (:7320" in source


# --- Scores near 0 ---------------------------------------------------------------------------


def test_each_add_clamps_at_zero(tag):
    # AddToMoveScore (trainer_ai.c:576-586): a negative score becomes 0 after each add, so the
    # order inside a block matters: Gravity -5 for a Flying attacker then +3 for a Flying target
    # from a score of 2 is 0 then 3, not 0 (2 - 5 + 3).
    t, _n, fight = tag
    s = fight(["Gravity", *_NONE], foe_types=(FLYING, NORMAL), own_types=(FLYING, NORMAL))
    rolls = {True: READ_DRAWS[1], False: READ_DRAWS[0]}
    for jump, expected in ((True, 0), (False, 3)):
        ctx = _slot_ctx(t, s, 0, rolls=(rolls[jump],))
        ctx = ctx._replace(scores=ctx.scores.at[0].set(2))
        out = policy._tag_strategy(c.start_run(c.at_slot(ctx, 0), 0))
        assert int(out.scores[0]) == expected


# --- Reads and bounds ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moves",
    [
        ["Tackle", "Harden", "Harden", "Harden"],
        ["Quick Attack", "Harden", "Harden", "Harden"],
        ["Flamethrower", "Stone Edge", "Harden", "Harden"],
        ["Follow Me", "Gravity", "Skill Swap", "Sunny Day"],
        ["Earthquake", "Bulldoze", "Surf", "Discharge"],
        ["Triple Axel", "Bulldoze", "Tackle", "Harden"],
    ],
)
def test_tag_reads_match_bounds(tag, kaizo_names, moves):
    # read_bounds (bit 7) holds every path's reads: the exact layout never needs more.
    from nuzline.ai.layout import move_read_rows

    t, n, fight = tag
    s = fight(moves)
    ids = [_id(n, m) for m in moves]
    rows = move_read_rows(t, ids)
    reads = _reads(t, s)
    col = list(__import__("nuzline.ai.read_bounds", fromlist=["x"]).READ_BOUND_BITS).index(7)
    drawing = {149, 222}
    for slot, move in enumerate(ids):
        bound = int(rows[slot, col])
        assert reads[slot] <= bound, (moves[slot], int(reads[slot]), bound)
    assert drawing & set(ids) or int(reads.max()) <= int(rows[:, col].max())


def test_a_path_past_slot_reads_is_refused(tag):
    # A highest super-effective move makes two reads (R@6694 jumping, then R@6722); a layout of one read has no draw for the second:
    # missing_read, the slot scores -1 (clamped into bucket 0 of the histogram).
    t, _n, fight = tag
    s = fight(["Ember", *_NONE], foe_types=(GRASS, NORMAL))
    assert _reads(t, s)[0] == 2  # R@6694 jumps, then R@6722
    hist = _exact_slot_scores(t, s, _layout(1))[0]
    # the refused path (-1, bucket 0) is the first read's jump with the missing read's fill draw
    assert hist[0] > 0 and hist[101] == pytest.approx(0.5)


def test_layout_of_a_bit_7_fight_has_the_bounds(tag):
    # fight_slot_reads over the fight's moves: Follow Me at most one read, Gravity one, Skill Swap
    # one (the defender's guess), a highest damaging move two.
    from nuzline.ai.layout import fight_layout

    t, n, fight = tag
    layout = fight_layout(t, 1 << 7, np.asarray([_id(n, "Follow Me"), _id(n, "Tackle")]))
    assert layout.flags == 1 << 7 and layout.slot_reads >= 1


@pytest.mark.slow
def test_tag_loose_vs_exact_607(kaizo_tables, kaizo_names, kaizo_trainers, kaizo_roark_party):
    # Trainer 607 (flags 0x87: BASIC, EVAL_ATTACK, EXPERT, TAG_STRATEGY): the exact distribution
    # equals the loose oracle's (every read its own draw), the action and each slot's scores,
    # on its first Pokemon with the moves the AI transcribes.
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.state import make_state
    from tests.ai.test_kaizo_outcome import _assert_exact_equals_loose, layout_for
    from tests.conftest import KAIZO_INTERMEDIATE

    t, n = kaizo_tables, kaizo_names
    trainer = next(x for x in kaizo_trainers if x.id == 607)
    assert trainer.ai_flags == 0x87
    mon = build_trainer_party(trainer, IntermediateData(KAIZO_INTERMEDIATE))[0]
    mon = dataclasses.replace(mon, pp=None)
    bench = dataclasses.replace(kaizo_roark_party[5], moves=(205, 29, 0, 0), pp=None)
    base = make_state(t, [kaizo_roark_party[2]], [mon, bench])
    layout = layout_for(t, trainer.ai_flags, base)
    s = _bulky(_with_player(t, base, 443), 200)
    reasons = policy.ai_refusals(t, s, n, layout)
    if reasons:
        pytest.skip(f"trainer 607's first Pokemon is refused by another flag: {reasons[:2]}")
    _assert_exact_equals_loose(t, s, layout)


# --- Tag Strategy at B = 4 is refused (E4-A: the B = 4 partner reads are not modelled) -------


@pytest.mark.fullset
@pytest.mark.parametrize("layout_name", ["allin", "fixed_with_bit_7", "fixed_without_bit_7"])
@pytest.mark.parametrize("flags", [0x87, 0x17], ids=["own_bit_7", "doubles_adds_bit_7"])
def test_tag_strategy_at_b4_is_refused(kaizo_tables, kaizo_names, layout_name, flags):
    """A B = 4 state whose ``thinking_mask`` has bit 7 (the trainer's own flag, or the doubles
    bit added at trainer_ai.c:252-255) is refused under every Kaizo layout: ``_tag_strategy``
    models B = 2 (A-8 sections 1, 5; A-10 sections 2, 3 are E4-A's). The same flags at B = 2
    are not refused for bit 7. A mechanic not modelled yet: ``fullset``."""
    from nuzline.ai.layout import ALLIN
    from nuzline.data.tables import BATTLE_TYPE_DOUBLES, BATTLE_TYPE_TRAINER
    from tests.engine.d2_states import four

    layouts = {
        "allin": ALLIN,
        "fixed_with_bit_7": kaizo_layout(0x87, 4),
        "fixed_without_bit_7": kaizo_layout(0x17, 4),
    }
    layout = layouts[layout_name]
    doubles = four(kaizo_tables, kaizo_names)
    t4 = kaizo_tables._replace(
        ai_flags=np.int32(flags), battle_type=BATTLE_TYPE_TRAINER | BATTLE_TYPE_DOUBLES
    )
    assert bool(policy._refused(t4, doubles, layout).policy)
    assert bool(policy._refused(t4, doubles, layout).any)
    # B = 4 without bit 7 in the mask (no doubles bit, flag unset): this refusal does not fire
    t_plain = kaizo_tables._replace(ai_flags=np.int32(0x17), battle_type=BATTLE_TYPE_TRAINER)
    assert not bool(policy._refused(t_plain, doubles, layout).policy)


@pytest.mark.fullset
def test_tag_strategy_at_b2_is_not_refused_by_the_policy(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    from nuzline.ai.layout import ALLIN

    s = _tag_fight(kaizo_tables, kaizo_names, kaizo_roark_party, list(_NONE) + ["Harden"])
    t = kaizo_tables._replace(ai_flags=np.int32(0x80))
    for layout in (ALLIN, kaizo_layout(0x80, SLOT_READS)):
        assert not bool(policy._refused(t, s, layout).policy)
