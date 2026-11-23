"""``TagStrategy_Main`` (bit 7, script.s:6637-7338) at B = 4: the partner reads (A-8 sections 1, 2, 5).

One fixed four-battler state per block, the exact law of the attacker's slot 0 score after the Tag
Strategy script alone (base 100), against the script. The expected laws are written by hand from
``script.s`` and ``trainer_ai.c`` (paths relative to ``external/pokeplatinum``): exact fractions of 256
for the ``IfRandomLessThan`` rolls (``IfRandomLessThan N`` jumps with probability N / 256, so the change
after it is made with probability 1 - N / 256) and of 64 for the speed ranks (A-10 section 2).

The scoring is called directly: ``_run`` builds an ``AiContext`` for the attacker ``a`` against the enemy
target ``d`` (``commands.new_context``), starts the run of flag bit 7 as ``policy._run_flags`` does and runs
``policy._tag_strategy``. The reads of the per-slot layout are the enumeration patterns' draws
(``READ_DRAWS``): ``_law`` walks every pattern that is canonical (the draws past the reads of its path
repeat ``READ_DRAWS[0]``) and sums the path weights by final score. Battlers: the attacker is ``a``, its
partner ``a ^ 2`` (``ATTACKER_PARTNER``), the target ``d``, the target's partner ``d ^ 2``
(``DEFENDER_PARTNER``; ``AIScript_Battler``, trainer_ai.c:2765-2782).

``CheckIfHighestDamageWithPartner`` (script.s:6684) is pinned to NOT_HIGHEST by ``_pinned_sweep``: its
B = 4 sweep is A-7's (``tests/ai/test_doubles_damage.py``), and every damaging move below only needs the
script to go on through ``TagStrategy_CheckBeforeScoring`` to ``TagStrategy_CheckSpecialScoring``. A
move that is not damaging reaches ``CheckSpecialScoring`` at script.s:6642 without it. The target is
Normal / Normal, so every move below is neutral against it (no TryScoreMinus, no CheckBeforeScoring roll).
The ally-target path (``IfTargetIsPartner``, ``TagStrategy_Partner``) is A-9's and not run here.

Every state is Bulbasaur everywhere (``d2_states.four``): an ability is written to the battler's
``BattleState.ability`` (a script write, ``damage.decomp_abilities``), a species ability pair of the tables is
edited only where a test needs the AI to be unsure which of two abilities a defender partner has.
"""

from __future__ import annotations

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai import policy
from nuzline.ai.layout import TAG_STRATEGY
from nuzline.ai.outcome import READ_DRAWS, no_calc_draws
from nuzline.data.tables import BATTLE_TYPES
from nuzline.engine.field import with_gravity
from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS, format_of
from nuzline.engine.ids import ABILITY_DECOMP_IDS, Stat, Status, Type
from nuzline.engine.state import ai_record_clear
from tests.ai.fixtures import outcome
from tests.engine.d2_states import four, with_hp, with_moves

NORMAL, FLYING, POISON, GROUND, ROCK = (int(Type.NORMAL), int(Type.FLYING), int(Type.POISON),
                                        int(Type.GROUND), int(Type.ROCK))  # fmt: skip
BUG, STEEL, FIRE, WATER = int(Type.BUG), int(Type.STEEL), int(Type.FIRE), int(Type.WATER)
GRASS, ELECTRIC, ICE = int(Type.GRASS), int(Type.ELECTRIC), int(Type.ICE)

SLOT_READS = (
    8  # the draws of one slot: the widest path here makes 7 (Future Sight, two sorts and a roll)
)
ENGINE_ABILITY = {decomp: engine for engine, decomp in ABILITY_DECOMP_IDS.items()}
# every ``IfRandomLessThan`` of the Tag script that these tests reach: N, the change is made at 1 - N / 256
ROLL_64 = 64 / 256  # P(jump) of R@6666, 6678, 6959, 6972, 7012, 7022, 7053, 7075-7095
ROLL_128 = 128 / 256  # R@6851, 7173, 7181, 7187


# --- The runner ----------------------------------------------------------------------------


def _pinned_sweep(ctx, use_damage_roll):
    """Stand-in for ``CheckIfHighestDamageWithPartner`` (trainer_ai.c:2369-2436): ``NO_COMPARISON``
    for a move that is not damaging (its gate, :2393-2394), else NOT_HIGHEST, no read."""
    damaging = c.is_damaging(ctx.tables, ctx.move)
    value = jnp.where(damaging, c.AI_NOT_HIGHEST_DAMAGE, c.AI_NO_COMPARISON_MADE)
    return ctx._replace(calc_temp=value.astype(jnp.int32))


@functools.partial(jax.jit, static_argnames=("a",))
def _run(t, s, d, rolls, a):
    """``(score, reads, weight, missing_read)`` of slot 0 for attacker ``a`` (static) against target
    ``d`` (traced) after ``TagStrategy_Main``, from 100, the slot's reads taking ``rolls`` in order."""
    ai_outcome = outcome()._replace(script_rolls=rolls, calc_draws=no_calc_draws((), SLOT_READS))
    ctx = c.new_context(t, s, ai_outcome, attacker=a, defender=d)
    ctx = c.start_run(c.at_slot(ctx, 0), TAG_STRATEGY)
    with mock.patch.object(c, "check_if_highest_damage_with_partner", _pinned_sweep):
        out = policy._tag_strategy(ctx)
    return out.scores[0], out.reads[0], out.weight[0], out.missing_read


def _law(t, s, a=1, d=0):
    """``({score: probability}, most reads on a path)``: every canonical pattern of draws, each
    weighed by the product of its reads' probabilities (``_slot_read``)."""
    paths = []

    def walk(prefix):
        rolls = np.full((4, SLOT_READS), READ_DRAWS[0], np.uint8)
        rolls[0, : len(prefix)] = [READ_DRAWS[bit] for bit in prefix]
        score, reads, weight, missing = _run(
            t, s, jnp.asarray(d, jnp.int32), jnp.asarray(rolls), a=a
        )
        assert int(missing) == 0, "the slot's reads ran past SLOT_READS"
        if int(reads) <= len(prefix):
            paths.append((int(score), int(reads), float(weight)))
        else:  # a read past the prefix: its two draws
            walk((*prefix, 0))
            walk((*prefix, 1))

    walk(())
    law: dict[int, float] = {}
    for score, _, weight in paths:
        law[score] = law.get(score, 0.0) + weight
    assert sum(law.values()) == pytest.approx(1.0, abs=1e-6), law
    return law, max(reads for _, reads, _ in paths)


def _check(t, s, want, *, a=1, d=0, reads=None, case=""):
    """The law of slot 0 equals ``want`` (a certain score, or ``{score: probability}``); ``reads``: the
    most reads a path makes."""
    want = {want: 1.0} if isinstance(want, int) else want
    law, most = _law(t, s, a, d)
    assert set(law) == set(want), (case, law, want)
    for score, p in want.items():
        assert law[score] == pytest.approx(p, abs=1e-6), (case, score, law, want)
    if reads is not None:
        assert most == reads, (case, most, reads)


def _roll(change: int, n: int) -> dict[int, float]:
    """The law of one ``IfRandomLessThan n`` that skips ``change`` when it jumps: from 100."""
    return {100: n / 256, 100 + change: 1 - n / 256}


# --- The states ----------------------------------------------------------------------------


def _tables4(tables, fmt):
    """``tables`` with the fight's ``battleType`` (data.tables.BATTLE_TYPES, encounter.c:724-740)."""
    kind = {DOUBLES: "double", MULTI: "multi", TWO_TRAINERS: "double_two_trainer"}[fmt]
    return tables._replace(battle_type=BATTLE_TYPES[kind])


def _at(s, b):
    """``(party row, active slot)`` of battler ``b``."""
    return format_of(s).party_of[b], int(s.active[b])


def _types(s, b, first, second):
    p, slot = _at(s, b)
    return s._replace(types=s.types.at[p, slot].set(jnp.asarray([first, second], s.types.dtype)))


def _hp(s, b, hp, max_hp=100):
    """Battler ``b`` at ``hp`` of ``max_hp``: hp% = ``hp * 100 / max_hp``."""
    p, slot = _at(s, b)
    return s._replace(max_hp=s.max_hp.at[p, slot].set(max_hp), hp=s.hp.at[p, slot].set(hp))


def _status(s, b, status=Status.BURN):
    p, slot = _at(s, b)
    return s._replace(status=s.status.at[p, slot].set(int(status)))


def _speeds(s, speeds):
    """The Speed stat of battlers 0..3 (all else equal, no item, no stage, no weather)."""
    for b, speed in enumerate(speeds):
        p, slot = _at(s, b)
        s = s._replace(stats=s.stats.at[p, slot, Stat.SPE - 1].set(speed))
    return s


def _flag(s, field, b, value=1):
    """``BattleState.<field>[b] = value`` for a per-battler field (magnet_rise, flash_fire)."""
    return s._replace(**{field: getattr(s, field).at[b].set(value)})


def _record(s, b, ability: str):
    """The AI's record of battler ``b``'s ability (``battlerAbilities[b]``), a decomp id."""
    return s._replace(ai_ability_known=s.ai_ability_known.at[b].set(c.ABILITY_IDS[ability]))


def _faint(s, b):
    """Battler ``b`` fainted and not replaced as ``BattleSystem_CleanupFaintedMon`` leaves it
    (battle_lib.c:2156-2226): 0 HP, species, types, ability, moves and status kept, its AI record
    cleared (``ai_record_clear``, :2223-2225), moveEffects fields already 0 in these states."""
    return ai_record_clear(with_hp(s, b, 0), b, True)


def _ability(t, s, b, name: str):
    """``(t, s)`` with battler ``b``'s true ability ``name``: the engine ability written to its
    ``BattleState.ability`` where the engine names it, else (Motor Drive) the tables' ability row of
    its species edited, which every battler of that species shares."""
    decomp = c.ABILITY_IDS[name]
    p, slot = _at(s, b)
    if decomp in ENGINE_ABILITY:
        return t, s._replace(ability=s.ability.at[p, slot].set(int(ENGINE_ABILITY[decomp])))
    row = int(t.species_form_row[int(s.species[p, slot]), int(s.ability_form[p, slot])])
    table = t.species_abilities_decomp.at[row, int(s.ability_slot[p, slot])].set(decomp)
    return t._replace(species_abilities_decomp=table), s


def _two_abilities(t, s, b, first: str, second: str):
    """``t`` whose species of battler ``b`` has the two abilities ``first`` and ``second``: the AI,
    which knows neither battler's, reads ``CheckBattlerAbility`` of an unrecorded ``b`` as UNKNOWN when
    one of them is the tested one (trainer_ai.c:1238-1246). Every battler of the species also has
    ``first`` as its true ability unless a test writes another one."""
    p, slot = _at(s, b)
    species = int(s.species[p, slot])
    row = int(t.species_form_row[species, int(s.ability_form[p, slot])])
    pair = jnp.asarray(
        [c.ABILITY_IDS[first], c.ABILITY_IDS[second]], t.species_abilities_decomp.dtype
    )
    table = t.species_abilities_decomp
    for index in {species, row}:
        table = table.at[index].set(pair)
    return t._replace(species_abilities_decomp=table)


class _World:
    """A four-battler state of ``fmt`` for the AI battler ``a``: every other battler Normal / Normal,
    the attacker Bulbasaur's own Grass / Poison, full HP, the default move set, and the tables."""

    def __init__(self, tables, names, fmt, a):
        self.n, self.fmt, self.a = names, fmt, a
        self.t = _tables4(tables, fmt)
        s = four(self.t, names, fmt)
        for b in range(4):
            if b != a:
                s = _types(s, b, NORMAL, NORMAL)
        self.s = s

    def attack(self, s, *moves):
        """The attacker's move set: ``moves`` (the first is slot 0, the one under test)."""
        return with_moves(s, self.n, self.a, moves)

    def moves(self, s, b, *moves):
        return with_moves(s, self.n, b, moves)


@pytest.fixture(scope="module")
def w(tables, names):
    """Attacker 1 against the player's battlers, DOUBLES: partner 3, target 0 and its partner 2."""
    return _World(tables, names, DOUBLES, 1)


@pytest.fixture(scope="module")
def w_multi(tables, names):
    return _World(tables, names, MULTI, 2)  # the AI partner: its partner is the player's Pokemon


@pytest.fixture(scope="module")
def w_two(tables, names):
    return _World(tables, names, TWO_TRAINERS, 1)  # its partner: the other trainer's Pokemon


def test_move_ids_are_the_decomps(names):
    # generated/moves.txt (line - 1); the script tests ids (trainer_ai.c:834-844).
    for key in ("Rain Dance", "Sunny Day", "Hail", "Sandstorm", "Gravity", "Trick Room", "Follow Me",
                "Helping Hand", "Future Sight", "Doom Desire", "Earthquake", "Discharge", "Surf",
                "Lava Plume"):  # fmt: skip
        assert names.move(key) == c.MOVE_IDS[key], key
    assert names.move("Magnitude") == c.MOVE_MAGNITUDE


# --- 2.2 TryScoreMinus1 / TryScoreMinus2 ----------------------------------------------------


@pytest.mark.parametrize("d", [0, 2])
@pytest.mark.parametrize(("types", "minus"), [((ROCK, NORMAL), 1), ((ROCK, STEEL), 2)])
def test_try_minus_defender_partner_alive(w, d, types, minus):
    # TagStrategy_TryScoreMinus1 / _2 (script.s:6658-6680): a HALF / QUARTER move that does not kill
    # (step 1) reaches IfHPPercentEqualTo DEFENDER_PARTNER 0 (step 2): a living partner (hp% != 0)
    # goes on to R@6666 / R@6678(64): jump, no change; else -1 / -2 (P 192/256). Tackle (Normal) is
    # not STAB for the Grass / Poison attacker, and resisted by Rock (and Steel).
    s = w.attack(w.s, "Tackle")
    s = _hp(_types(s, d, *types), d, 500, 500)  # 500 HP: the maximum roll does not kill
    dp = d ^ 2
    _check(w.t, s, {100: ROLL_64, 100 - minus: 1 - ROLL_64}, d=d, reads=1, case="alive")
    # hp% 0: fainted, or alive with curHP * 100 < maxHP (5 * 100 / 1000 = 0): step 2 jumps, no roll
    _check(w.t, _faint(s, dp), 100, d=d, reads=0, case="fainted")
    _check(w.t, _hp(s, dp, 5, 1000), 100, d=d, reads=0, case="hp% 0")
    # hp% 1: reads it
    _check(w.t, _hp(s, dp, 10, 1000), {100: ROLL_64, 100 - minus: 1 - ROLL_64}, d=d, reads=1)
    # step 1: the maximum roll kills: ScoreMove before the partner test
    _check(w.t, _hp(s, d, 1, 500), 100, d=d, reads=0, case="kills")


# --- 2.6-2.9 weather blocks ------------------------------------------------------------------


def test_weather_partner_abilities(w):
    t, base = w.t, w.s
    ap = 3
    # TagStrategy_RainDance (script.s:6754-6786): the partner's Hydration (statused) or Dry Skin +2
    rain = w.attack(base, "Rain Dance")
    t1, s1 = _ability(t, rain, ap, "Hydration")
    _check(t1, _status(s1, ap), 102, case="partner Hydration burned")
    _check(t1, s1, 100, case="partner Hydration healthy (:6781)")
    t1, s1 = _ability(t, rain, ap, "Dry Skin")
    _check(t1, s1, 102, case="partner Dry Skin")
    t1, s1 = _ability(t, rain, ap, "Levitate")
    _check(t1, s1, 100, case="partner Levitate")
    t2, s2 = _ability(t, rain, 1, "Dry Skin")
    t2, s2 = _ability(t2, s2, ap, "Dry Skin")
    _check(t2, s2, 104, case="own and partner Dry Skin")
    t2, s2 = _ability(t, rain, 1, "Hydration")
    t2, s2 = _ability(t2, _status(s2, ap), ap, "Hydration")
    _check(t2, s2, 102, case="own Hydration healthy (:6767: no +2), partner Hydration burned")
    # TagStrategy_SunnyDay (:6788-6855): the first of the partner's Leaf Guard (no status, hp% >= 30)
    # / Flower Gift +2, Dry Skin -2, Solar Power (hp% >= 50: +1, then the roll R@6851(128) -2)
    sun = w.attack(base, "Sunny Day")
    t1, s1 = _ability(t, sun, ap, "Leaf Guard")
    _check(t1, s1, 102, case="Leaf Guard healthy")
    _check(t1, _status(s1, ap), 100, case="Leaf Guard statused")
    _check(t1, _hp(s1, ap, 29), 100, case="Leaf Guard hp% 29")
    _check(t1, _hp(s1, ap, 30), 102, case="Leaf Guard hp% 30")
    t1, s1 = _ability(t, sun, ap, "Flower Gift")
    _check(t1, _status(_hp(s1, ap, 10), ap), 102, case="Flower Gift")
    t1, s1 = _ability(t, sun, ap, "Dry Skin")
    _check(t1, s1, 98, case="Dry Skin")
    t1, s1 = _ability(t, sun, ap, "Solar Power")
    _check(t1, _hp(s1, ap, 50), {101: ROLL_128, 99: 1 - ROLL_128}, reads=1, case="Solar Power 50")
    _check(t1, s1, {101: ROLL_128, 99: 1 - ROLL_128}, reads=1, case="Solar Power 100")
    _check(t1, _hp(s1, ap, 49), {100: ROLL_128, 98: 1 - ROLL_128}, reads=1, case="Solar Power 49")
    t2, s2 = _ability(t, sun, 1, "Flower Gift")
    t2, s2 = _ability(t2, s2, ap, "Leaf Guard")
    _check(t2, s2, 104, case="own Flower Gift, partner Leaf Guard")
    # TagStrategy_Hail (:6857-6884): the partner's Ice Body, Snow Cloak or a known Blizzard +2
    hail = w.attack(base, "Hail", "Blizzard")
    for name in ("Ice Body", "Snow Cloak"):
        t1, s1 = _ability(t, hail, ap, name)
        _check(t1, s1, 102, case=f"partner {name}")
    _check(t, w.moves(hail, ap, "Blizzard", "Tackle"), 102, case="partner knows Blizzard (alive)")
    t1, s1 = _ability(t, hail, ap, "Levitate")
    _check(t1, s1, 100, case="partner Levitate")
    t1, s1 = _ability(t, hail, 1, "Ice Body")
    t1, s1 = _ability(t1, s1, ap, "Snow Cloak")
    _check(t1, s1, 104, case="own Ice Body, partner Snow Cloak")
    _check(t, w.moves(hail, ap, "Blizzard"), 104, case="both know Blizzard")


@pytest.mark.parametrize(
    ("attacker", "partner", "ability", "want"),
    [
        (
            None,
            (ROCK, NORMAL),
            None,
            102,
        ),  # the partner's own type 1 (LoadTypeFrom ..._PARTNER_TYPE_1)
        (None, (NORMAL, ROCK), None, 102),  # type 2
        (None, (ROCK, ROCK), None, 102),  # one +2: the first hit jumps to PartnerScorePlus2
        (None, (NORMAL, NORMAL), "Sand Veil", 102),
        (None, (ROCK, NORMAL), "Sand Veil", 102),  # Sand Veil first, then End: still one +2
        (None, (NORMAL, NORMAL), None, 100),
        (
            (ROCK, NORMAL),
            (NORMAL, NORMAL),
            None,
            102,
        ),  # own Rock +2 only: B = 2's +4 is the attacker twice
        ((ROCK, NORMAL), (NORMAL, ROCK), None, 104),
    ],
)
def test_sandstorm_partner_rock(w, attacker, partner, ability, want):
    # TagStrategy_Sandstorm (script.s:6886-6916): the attacker's own Sand Veil / Rock types +2, then the
    # partner's: CheckBattlerAbility ATTACKER_PARTNER SAND_VEIL, LoadTypeFrom ATTACKER_PARTNER_TYPE_1 /
    # _2 (BattleSystem_GetPartner = attacker ^ 2, trainer_ai.c:965-978), the first hit +2.
    t, s = w.t, w.attack(w.s, "Sandstorm")
    if attacker:
        s = _types(s, 1, *attacker)
    s = _types(s, 3, *partner)
    if ability:
        t, s = _ability(t, s, 3, ability)
    _check(t, s, want)


# --- 2.10 Gravity -----------------------------------------------------------------------------


def test_gravity_partners(w):
    # TagStrategy_Gravity (script.s:6918-6977): Gravity on: -30. Attacker and partner: Levitate (HAVE),
    # Flying or Magnet Rise: -5 each. Target and its partner: the same tests, R@6959 / R@6972(64):
    # jump, no change; else +3 (192/256). The defender partner's Levitate is read through the AI's
    # record (CheckBattlerAbility DEFENDER_PARTNER, :1223-1253).
    t, base = w.t, w.attack(w.s, "Gravity")
    _check(t, with_gravity(base, 5), 70, case="Gravity on")
    # the partner (battler 3)
    _check(t, _types(base, 3, FLYING, NORMAL), 95, case="partner Flying")
    t1, s1 = _ability(t, base, 3, "Levitate")
    _check(t1, s1, 95, case="partner Levitate")
    _check(t, _flag(base, "magnet_rise", 3), 95, case="partner Magnet Rise")
    _check(
        t1,
        _types(_flag(s1, "magnet_rise", 3), 3, FLYING, NORMAL),
        95,
        case="partner all three: once",
    )
    _check(
        t, _types(_types(base, 3, FLYING, NORMAL), 1, FLYING, NORMAL), 90, case="own and partner"
    )
    # the target (battler 0)
    up = {100: ROLL_64, 103: 1 - ROLL_64}
    _check(t, _types(base, 0, NORMAL, FLYING), up, reads=1, case="target Flying")
    _check(t, _flag(base, "magnet_rise", 0), up, reads=1, case="target Magnet Rise")
    _check(t, _record(base, 0, "Levitate"), up, reads=1, case="target Levitate recorded")
    # the target's partner (battler 2): R@6972
    _check(t, _types(base, 2, FLYING, NORMAL), up, reads=1, case="defender partner Flying")
    _check(t, _flag(base, "magnet_rise", 2), up, reads=1, case="defender partner Magnet Rise")
    _check(t, _record(base, 2, "Levitate"), up, reads=1, case="defender partner Levitate recorded")
    t1, s1 = _ability(t, base, 2, "Levitate")  # true but not recorded: the AI reads Overgrow
    _check(t1, s1, 100, reads=0, case="defender partner Levitate unrecorded")
    both = _types(_types(base, 0, FLYING, NORMAL), 2, FLYING, NORMAL)
    _check(t, both, {100: ROLL_64**2, 103: 2 * ROLL_64 * (1 - ROLL_64), 106: (1 - ROLL_64) ** 2},
           reads=2, case="target and its partner")  # fmt: skip
    everyone = _types(_types(both, 1, FLYING, NORMAL), 3, FLYING, NORMAL)
    _check(t, everyone, {90: ROLL_64**2, 93: 2 * ROLL_64 * (1 - ROLL_64), 96: (1 - ROLL_64) ** 2},
           reads=2, case="all four")  # fmt: skip
    # target 2: its partner is battler 0
    swapped = _types(_types(base, 2, FLYING, NORMAL), 0, FLYING, NORMAL)
    _check(t, swapped, {100: ROLL_64**2, 103: 2 * ROLL_64 * (1 - ROLL_64), 106: (1 - ROLL_64) ** 2},
           d=2, reads=2, case="d = 2")  # fmt: skip


# --- 2.11 Trick Room ---------------------------------------------------------------------------

# battlers 0 (target), 1 (attacker), 2 (target partner), 3 (attacker partner): their Speed; the
# attacker is rank r, its partner rank q (descending Speed, distinct: no tie, no read)
TRICK_ROOM_RANKS = [
    ((50, 100, 40, 90), 70, "r0 q1"),
    ((90, 100, 70, 80), 95, "r0 q2"),
    ((90, 100, 80, 10), 95, "r0 q3"),
    ((50, 90, 40, 100), 70, "r1 q0"),
    ((100, 90, 40, 80), 95, "r1 q2"),
    ((100, 90, 80, 10), 95, "r1 q3"),
    ((90, 80, 70, 100), 95, "r2 q0"),
    ((100, 80, 70, 90), 95, "r2 q1"),
    ((100, 70, 90, 80), {95: ROLL_64, 105: 1 - ROLL_64}, "r3 q2: R@7022"),
    ((100, 80, 90, 70), {95: ROLL_64, 105: 1 - ROLL_64}, "r2 q3: R@7012"),
    ((90, 10, 80, 100), 95, "r3 q0"),
    ((100, 10, 80, 90), 95, "r3 q1"),
]


def test_trick_room_ranks(w):
    # TagStrategy_TrickRoom (script.s:6979-7030). The attacker's rank r, then its partner's rank q: a
    # new sort per call.
    #   r0: q 1 or 0: -30, else -5.   r1: q 0: -30, else -5.
    #   r2: q != 3: -5, else R@7012(64): jump -5, else +5.   r3: q != 2: -5, else R@7022(64).
    t, base = w.t, w.attack(w.s, "Trick Room")
    for speeds, want, case in TRICK_ROOM_RANKS:
        reads = 1 if isinstance(want, dict) else 0
        _check(t, _speeds(base, speeds), want, reads=reads, case=case)
    # a fainted attacker's partner, target partner or target: -30 before any sort
    for b in (3, 2, 0):
        _check(
            t, _faint(_speeds(base, (90, 100, 70, 80)), b), 70, reads=0, case=f"battler {b} down"
        )
    # ties: the attacker and its partner share the top two ranks (Speed 100, the others slower). Each
    # LoadBattlerSpeedRank is its own sort with its own tie draw: r is 0 or 1 (1/2 each), then q
    # is 0 or 1 again (1/2 each), independent of r. r0: -30 whatever q; r1: q 0 -30, q 1 -5.
    _check(
        t, _speeds(base, (50, 100, 40, 100)), {70: 3 / 4, 95: 1 / 4}, reads=2, case="tie at the top"
    )
    # the two share the bottom two ranks (D 100, DP 90 above): r2 (1/2): q 2 -5, q 3 R@7012 +5 w.p. 3/4;
    # r3 (1/2): q 3 -5, q 2 R@7022 +5 w.p. 3/4: +5 with 2 * 1/2 * 1/2 * 3/4 = 3/8
    _check(
        t,
        _speeds(base, (100, 50, 90, 50)),
        {95: 5 / 8, 105: 3 / 8},
        reads=3,
        case="tie at the bottom",
    )


# --- 2.12 Follow Me ----------------------------------------------------------------------------

# own hp% a, partner hp% q: the change, made with P 192/256 (R@7075 -1, R@7080 -2, R@7085 +1, R@7090 +2,
# R@7095 +3); None: the own-hp% <= 30 case (R@7053, -5)
FOLLOW_ME = [
    (100, 100, -1), (100, 70, 1), (100, 40, 2), (100, 20, 3),
    (70, 100, -2), (70, 70, -1), (70, 40, 1), (70, 20, 2),
    (40, 100, -2), (40, 70, -2), (40, 40, 1), (40, 20, 2),
    (20, 100, -5), (20, 20, -5),
    # the strict bounds: a > 90 / 50 / 30, q > 90 / 50 / 30
    (90, 90, -1), (91, 91, -1), (51, 51, -1), (50, 50, 1), (31, 31, 1), (30, 30, -5),
    (100, 30, 3), (100, 31, 2), (100, 50, 2), (100, 51, 1), (100, 90, 1),
    (70, 30, 2), (70, 50, 1), (70, 90, -1), (40, 30, 2), (40, 50, 1), (40, 90, -2),
]  # fmt: skip


@pytest.mark.parametrize(("a_hp", "q_hp", "change"), FOLLOW_ME)
def test_follow_me_bands(w, a_hp, q_hp, change):
    # TagStrategy_FollowMe (script.s:7032-7100): a = hp%(attacker), q = hp%(attacker partner), strict >.
    #   a > 90: q > 90 -1; q > 50 +1; q > 30 +2; else +3.   a > 50: -2, -1, +1, +2.
    #   a > 30: -2, -2, +1, +2.   a <= 30: R@7053(64), -5 (the partner is not read).
    s = _hp(_hp(w.attack(w.s, "Follow Me"), 1, a_hp), 3, q_hp)
    _check(w.t, s, _roll(change, 64), reads=1)


# --- 2.13 Earthquake / Magnitude id -------------------------------------------------------------


@pytest.mark.parametrize("move", ["Earthquake", "Magnitude"])
def test_earthquake_partner_types(w, move):
    # TagStrategy_Earthquake (script.s:7127-7148), the attacker's partner, first match: Magnet Rise,
    # Levitate (HAVE), Flying +2; Fire, Electric, Poison, Rock -10; else -3. No HP test, no roll. The
    # block is reached by the move ids (Magnitude: id 222, trainer_ai.c:834-844).
    t, base = w.t, w.attack(w.s, move)
    ap = 3
    _check(
        t, _types(_flag(base, "magnet_rise", ap), ap, FIRE, NORMAL), 102, case="Magnet Rise, Fire"
    )
    t1, s1 = _ability(t, _types(base, ap, FIRE, NORMAL), ap, "Levitate")
    _check(t1, s1, 102, case="Levitate, Fire")
    _check(t, _types(base, ap, FIRE, FLYING), 102, case="Fire / Flying")
    _check(t, _types(base, ap, FLYING, NORMAL), 102, case="Flying")
    for first, second in ((FIRE, NORMAL), (NORMAL, ELECTRIC), (POISON, NORMAL), (NORMAL, ROCK),
                          (FIRE, ROCK), (ELECTRIC, POISON)):  # fmt: skip
        _check(t, _types(base, ap, first, second), 90, case=f"{first} / {second}")
    for first, second in ((NORMAL, NORMAL), (GRASS, NORMAL), (GROUND, NORMAL), (WATER, NORMAL),
                          (ICE, BUG), (STEEL, NORMAL)):  # fmt: skip
        _check(t, _types(base, ap, first, second), 97, case=f"{first} / {second}")
    # no HP test: a partner at 1 HP is read the same
    _check(t, _hp(_types(base, ap, FIRE, NORMAL), ap, 1), 90, case="partner at 1 HP")


# --- 2.5.5 + 2.14 Helping Hand ----------------------------------------------------------------


def test_partner_helping_hand_plus1(w):
    # TagStrategy_CheckSpecialScoring step 5 (script.s:6732-6752): IfMoveKnown ATTACKER_PARTNER
    # HELPING_HAND (true moves; no jump while the partner's curHP is 0, trainer_ai.c:1650-1664) ->
    # PartnerKnowsHelpingHand (:7102-7114): a flat-damage effect ends; FlagMoveDamageScore: a damaging move
    # +1, any other move nothing. An Electric / Fire / Water move leaves CheckSpecialScoring earlier.
    t = w.t
    knows = w.moves(w.s, 3, "Tackle", "Swords Dance", "Earthquake", "Helping Hand")  # any slot
    _check(t, w.attack(knows, "Tackle"), 101, case="damaging move")
    _check(t, w.attack(knows, "Harden"), 100, case="status move")
    _check(t, w.attack(knows, "Dragon Rage"), 100, case="flat damage (40)")
    _check(t, w.attack(knows, "Thunderbolt"), 100, case="Electric: its own block, not this one")
    _check(t, w.attack(_faint(knows, 3), "Tackle"), 100, case="fainted partner knows it")
    _check(t, w.attack(w.s, "Tackle"), 100, case="partner does not know it")
    _check(t, w.attack(_hp(knows, 3, 1), "Tackle"), 101, case="partner at 1 HP")


# --- 2.16 Future Sight / Doom Desire ----------------------------------------------------------

# Speed of battlers 0 (target), 1 (attacker), 2 (target partner), 3 (attacker partner). r = rank of
# the attacker, q of its partner. -3 is 97.
FUTURE_SIGHT = [
    ((100, 10, 90, 80), 97, 0, "r3"),
    ((90, 80, 70, 100), 97, 0, "r2 q0"),
    ((100, 80, 70, 90), 97, 0, "r2 q1"),
    ((100, 80, 90, 70), 100, 1, "r2 q3: R@7173, then q' is 3, not 2"),
    ((50, 90, 40, 100), 97, 0, "r1 q0"),
    ((100, 90, 40, 80), 100, 1, "r1 q2: R@7181, then q' is 2, not 1"),
    ((50, 100, 40, 90), 100, 1, "r0: R@7187, then q' is 1, not 0"),
    # ties: a new sort per call (A-10 section 2.4). Attacker and partner share ranks 2 and 3: r3 (1/2) -3;
    # r2 (1/2): q is 2 or 3, R@7173 does not jump (1/2), q' == 2 (1/2): -3 with 1/2 + 1/8
    ((100, 50, 90, 50), {97: 5 / 8, 100: 3 / 8}, 4, "tie at the bottom"),
    # share ranks 0 and 1: r0 (1/2): R@7187 (1/2), q' == 0 (1/2): 1/4; r1 (1/2): q 0 (1/2) -3, else
    # R@7181 (1/2), q' == 1 (1/2): 5/8: 1/2 * 1/4 + 1/2 * 5/8 = 7/16
    ((50, 100, 40, 100), {97: 7 / 16, 100: 9 / 16}, 4, "tie at the top"),
    # the target's partner is down: it ranks last whatever its Speed (a dead battler, no draw), the
    # sort goes on (no HP test of it before s:7162): D, AP, A, DP: r2 q1
    ((100, 80, 999, 90), 97, 0, "target partner fainted"),
]


@pytest.mark.parametrize(("speeds", "want", "reads", "case"), FUTURE_SIGHT)
@pytest.mark.parametrize(
    ("move", "known"), [("Future Sight", "Doom Desire"), ("Doom Desire", "Future Sight")]
)
def test_future_sight_ranks(w, speeds, want, reads, case, move, known):
    # TagStrategy_FutureSight (script.s:7150-7193): the partner alive and knowing Future Sight or
    # Doom Desire; r = rank(attacker), then each q a new LoadBattlerSpeedRank of the partner:
    #   r3 -3.  r2: q 0 or 1 -3; R@7173(128) jump End; q' == 2 -3.
    #   r1: q 0 -3; R@7181(128); q' == 1 -3.   r0: R@7187(128); q' == 0 -3.
    s = w.moves(w.attack(_speeds(w.s, speeds), move), 3, known, "Tackle")
    if case == "target partner fainted":
        s = _faint(s, 2)
    _check(w.t, s, want, reads=reads, case=case)


def test_future_sight_no_partner_effect(w):
    # IfHPPercentEqualTo ATTACKER_PARTNER 0 -> End (s:7157); IfMoveKnown ATTACKER_PARTNER Future Sight /
    # Doom Desire else End (:7158-7160): r3 would score -3.
    t, slow = w.t, _speeds(w.s, (100, 10, 90, 80))
    _check(t, w.moves(w.attack(slow, "Future Sight"), 3, "Future Sight"), 97, case="control")
    _check(
        t,
        _faint(w.moves(w.attack(slow, "Future Sight"), 3, "Future Sight"), 3),
        100,
        case="fainted",
    )
    _check(t, w.moves(w.attack(slow, "Future Sight"), 3, "Tackle"), 100, case="does not know it")
    _check(
        t, w.moves(w.attack(slow, "Future Sight"), 3, "Hail", "Psych Up"), 100, case="other moves"
    )


# --- 2.18-2.20 spread tables --------------------------------------------------------------------


def _spread_cases(name, cases, w, move, base_setup=None):
    t0, base = w.t, w.attack(w.s, move)
    for case in cases:
        t, s = t0, base
        want, ap_types, ap_ability = case[:3]
        if ap_types:
            s = _types(s, 3, *ap_types)
        if ap_ability:
            t, s = _ability(t, s, 3, ap_ability)
        _check(t, s, want, case=f"{name} {case}")


def test_spread_partner_tables(w):
    # TagStrategy_SpreadElectricMove (script.s:7240-7261) / SpreadWaterMove (:7287-7300) /
    # SpreadFireMove (:7321-7337), read for the attacker's partner (the id of the move, not its type).
    # Electric: Motor Drive, Volt Absorb +3; Water, Flying -10; Ground +3 (after them: the decomp's bug,
    # a Water / Ground or Flying / Ground partner is -10); else -3.
    _spread_cases("Discharge", [
        (103, None, "Motor Drive"), (103, None, "Volt Absorb"), (90, (WATER, NORMAL), None),
        (90, (NORMAL, FLYING), None), (103, (GROUND, NORMAL), None), (90, (WATER, GROUND), None),
        (90, (FLYING, GROUND), None), (97, (NORMAL, NORMAL), None), (97, (GRASS, NORMAL), None),
        (103, (WATER, NORMAL), "Motor Drive"),
    ], w, "Discharge")  # fmt: skip
    # Water: Dry Skin, Water Absorb +3; Ground, Fire -10 (Rock is the decomp's bug: not tested); else -3
    _spread_cases("Surf", [
        (103, None, "Dry Skin"), (103, None, "Water Absorb"), (90, (GROUND, NORMAL), None),
        (90, (NORMAL, FIRE), None), (97, (ROCK, NORMAL), None), (97, (NORMAL, NORMAL), None),
        (97, (WATER, NORMAL), None), (103, (GROUND, NORMAL), "Dry Skin"),
    ], w, "Surf")  # fmt: skip
    # Fire: Dry Skin -3 (ScoreMinus3, s:7322-7323; the comment at :7308 says +3), Flash Fire +3;
    # Grass, Steel, Ice, Bug -10; else -3
    _spread_cases("Lava Plume", [
        (97, None, "Dry Skin"), (103, None, "Flash Fire"), (90, (GRASS, NORMAL), None),
        (90, (NORMAL, STEEL), None), (90, (ICE, NORMAL), None), (90, (NORMAL, BUG), None),
        (97, (NORMAL, NORMAL), None), (97, (GROUND, NORMAL), None), (97, (GRASS, NORMAL), "Dry Skin"),
        (103, (GRASS, NORMAL), "Flash Fire"),
    ], w, "Lava Plume")  # fmt: skip
    # the Flash Fire +1 of the attacker comes first (TagStrategy_CheckFireMove, :7304-7312)
    lava = w.attack(w.s, "Lava Plume")
    _check(w.t, _flag(lava, "flash_fire", 1), 98, case="Flash Fire on, partner Normal: +1 - 3")
    t1, s1 = _ability(w.t, _flag(lava, "flash_fire", 1), 3, "Flash Fire")
    _check(t1, s1, 104, case="Flash Fire on, partner Flash Fire: +1 + 3")
    flame = w.attack(w.s, "Flamethrower")
    _check(w.t, flame, 100, case="Fire move, not Lava Plume")
    _check(w.t, _flag(flame, "flash_fire", 1), 101, case="Flash Fire on: +1")
    _check(w.t, _types(flame, 3, GRASS, NORMAL), 100, case="no spread table for another Fire move")


def test_lightning_rod_storm_drain_of_the_defender_partner(w):
    t, base = w.t, w.s
    dp, ap = 2, 3
    # TagStrategy_CheckElectricMove (script.s:7215-7263), a move that is not Discharge: the target's
    # partner HAVE Lightning Rod (the AI's record): -1, then -8 more unless FlagBattlerIsType
    # DEFENDER_PARTNER GROUND is NOT_HAVE (:7224-7226). The attacker's partner HAVE: -10.
    bolt = w.attack(base, "Thunderbolt")
    _check(t, bolt, 100, case="nothing")
    rod = _record(bolt, dp, "Lightningrod")
    _check(t, rod, 99, case="defender partner Lightning Rod")
    _check(t, _types(rod, dp, GROUND, NORMAL), 91, case="... and a Ground type: -1 - 8")
    t1, s1 = _ability(t, bolt, ap, "Lightningrod")
    _check(t1, s1, 90, case="attacker partner Lightning Rod")
    t1, s1 = _ability(t, rod, ap, "Lightningrod")
    _check(t1, s1, 89, case="both")
    t1, s1 = _ability(t, bolt, dp, "Lightningrod")
    _check(t1, s1, 100, case="true but unrecorded: the AI reads the species' Overgrow")
    # two abilities, one of them Lightning Rod, unrecorded: UNKNOWN, not HAVE: no -1
    t2 = _two_abilities(t, bolt, dp, "Lightningrod", "Overgrow")
    t2, s2 = _ability(t2, bolt, ap, "Levitate")
    _check(t2, s2, 100, case="defender partner UNKNOWN")
    # Discharge does not run these tests (:7219-7220)
    _check(t, _record(w.attack(base, "Discharge"), dp, "Lightningrod"), 97, case="Discharge")
    # TagStrategy_CheckWaterMove (:7265-7302), not Surf: the target's partner NOT_HAVE Storm Drain 0,
    # HAVE or UNKNOWN -1 (:7280-7283); the attacker's partner HAVE -10
    gun = w.attack(base, "Water Gun")
    _check(t, gun, 100, case="nothing")
    drain = _record(gun, dp, "Storm Drain")
    _check(t, drain, 99, case="defender partner Storm Drain")
    _check(t, _record(gun, dp, "Levitate"), 100, case="defender partner records another ability")
    t1, s1 = _ability(t, gun, ap, "Storm Drain")
    _check(t1, s1, 90, case="attacker partner Storm Drain")
    t1, s1 = _ability(t, drain, ap, "Storm Drain")
    _check(t1, s1, 89, case="both")
    t1, s1 = _ability(t, gun, dp, "Storm Drain")
    _check(t1, s1, 100, case="true but unrecorded: NOT_HAVE")
    t2 = _two_abilities(t, gun, dp, "Storm Drain", "Overgrow")
    t2, s2 = _ability(t2, gun, ap, "Levitate")
    _check(t2, s2, 99, case="defender partner UNKNOWN: -1")
    # Surf does not run them (:7269-7270)
    _check(t, _record(w.attack(base, "Surf"), dp, "Storm Drain"), 97, case="Surf")


# --- A-8 section 5: a fainted partner ---------------------------------------------------------


def test_fainted_partner_reads(w):
    # A-8 section 5 (BattleSystem_CleanupFaintedMon, battle_lib.c:2156-2226): a fainted battler keeps its
    # species, types, ability, moves, item and status; its HP reads 0, its Magnet Rise / Flash Fire and
    # its AI record are gone. HP-tested blocks read it as hp% 0; ability and type blocks still score it.
    t, base = w.t, w.s
    ap, dp = 3, 2
    # the attacker's partner fainted
    t1, s1 = _ability(t, w.attack(base, "Rain Dance"), ap, "Dry Skin")
    _check(t1, _faint(s1, ap), 102, case="Rain: Dry Skin still scores")
    t1, s1 = _ability(t, w.attack(base, "Rain Dance"), ap, "Hydration")
    _check(t1, _faint(_status(s1, ap), ap), 102, case="Rain: Hydration with the kept status")
    t1, s1 = _ability(t, w.attack(base, "Hail"), ap, "Ice Body")
    _check(t1, _faint(s1, ap), 102, case="Hail: Ice Body still scores")
    _check(
        t,
        _faint(w.moves(w.attack(base, "Hail"), ap, "Blizzard"), ap),
        100,
        case="Hail: no move known",
    )
    _check(
        t,
        _faint(_types(w.attack(base, "Sandstorm"), ap, ROCK, NORMAL), ap),
        102,
        case="Sandstorm: Rock",
    )
    gravity = w.attack(base, "Gravity")
    _check(t, _faint(_types(gravity, ap, FLYING, NORMAL), ap), 95, case="Gravity: Flying")
    t1, s1 = _ability(t, gravity, ap, "Levitate")
    _check(t1, _faint(s1, ap), 95, case="Gravity: Levitate")
    _check(t, _faint(gravity, ap), 100, case="Gravity: no Magnet Rise, nothing")
    follow = w.attack(base, "Follow Me")  # hp% 0 of the partner: the q = 0 bands, as at B = 2
    for a_hp, change in ((100, 3), (90, 2), (51, 2), (50, 2), (31, 2), (30, -5), (20, -5)):
        down = _faint(_hp(follow, 1, a_hp), ap)
        _check(t, down, _roll(change, 64), reads=1, case=f"Follow Me a = {a_hp}")
    _check(t, _faint(w.attack(base, "Trick Room"), ap), 70, reads=0, case="Trick Room")
    knows = w.moves(w.attack(base, "Future Sight"), ap, "Future Sight")
    _check(t, _faint(_speeds(knows, (100, 10, 90, 80)), ap), 100, reads=0, case="Future Sight")
    helping = w.moves(w.attack(base, "Tackle"), ap, "Helping Hand")
    _check(t, _faint(helping, ap), 100, case="Helping Hand")
    quake = w.attack(base, "Earthquake")
    _check(t, _faint(_types(quake, ap, FIRE, NORMAL), ap), 90, case="Earthquake: Fire")
    _check(t, _faint(quake, ap), 97, case="Earthquake: Normal")
    _check(
        t, _faint(_types(w.attack(base, "Discharge"), ap, WATER, NORMAL), ap), 90, case="Discharge"
    )
    _check(t, _faint(_types(w.attack(base, "Surf"), ap, GROUND, NORMAL), ap), 90, case="Surf")
    _check(t, _faint(w.attack(base, "Lava Plume"), ap), 97, case="Lava Plume")
    t1, s1 = _ability(t, w.attack(base, "Thunderbolt"), ap, "Lightningrod")
    _check(t1, _faint(s1, ap), 90, case="Lightning Rod of the attacker's partner")
    # the target's partner fainted
    tackle = _hp(_types(w.attack(base, "Tackle"), 0, ROCK, NORMAL), 0, 500, 500)
    _check(t, _faint(tackle, dp), 100, reads=0, case="TryScoreMinus: skip")
    _check(
        t,
        _faint(_types(gravity, dp, FLYING, NORMAL), dp),
        _roll(3, 64),
        reads=1,
        case="Gravity: Flying",
    )
    _check(
        t, _faint(_flag(gravity, "magnet_rise", dp, 0), dp), 100, case="Gravity: Magnet Rise gone"
    )
    _check(
        t,
        _faint(_speeds(w.attack(base, "Trick Room"), (100, 80, 90, 70)), dp),
        70,
        reads=0,
        case="TR",
    )
    # its record is cleared: a recorded Storm Drain no longer reads HAVE
    gun = _record(w.attack(base, "Water Gun"), dp, "Storm Drain")
    _check(t, gun, 99, case="alive: recorded Storm Drain")
    _check(t, _faint(gun, dp), 100, case="fainted: record cleared")


# --- MULTI, TWO_TRAINERS ----------------------------------------------------------------------


def test_multi_partner_attacker_true_reads(w_multi, w_two):
    # MULTI battler 2 (the AI partner, A-16): its ATTACKER_PARTNER is battler 0, the player's Pokemon,
    # and the target 1's partner is battler 3. Same-side reads are true data, never the AI's record
    # (trainer_ai.c:1255-1256, :1650-1664, :1925-1926). TWO_TRAINERS: the attacker's partner is the
    # other trainer's Pokemon (party 2, battler 3).
    for world, a, ap, d in ((w_multi, 2, 0, 1), (w_two, 1, 3, 0)):
        t, base = world.t, world.s
        quake = world.attack(base, "Earthquake")
        _check(
            t, _types(quake, ap, FIRE, NORMAL), 90, a=a, d=d, case=f"{world.fmt} Earthquake Fire"
        )
        _check(t, quake, 97, a=a, d=d, case=f"{world.fmt} Earthquake Normal")
        t1, s1 = _ability(t, world.attack(base, "Rain Dance"), ap, "Dry Skin")
        _check(t1, s1, 102, a=a, d=d, case=f"{world.fmt} true ability (no record)")
        hail = world.moves(world.attack(base, "Hail"), ap, "Blizzard")
        _check(t, hail, 102, a=a, d=d, case=f"{world.fmt} true moves")
        _check(t, _types(world.attack(base, "Gravity"), ap, FLYING, NORMAL), 95, a=a, d=d,
               case=f"{world.fmt} Gravity")  # fmt: skip
        s1 = _hp(_hp(world.attack(base, "Follow Me"), a, 100), ap, 40)
        _check(t, s1, _roll(2, 64), a=a, d=d, reads=1, case=f"{world.fmt} Follow Me")
    # the target's partner by target: attacker 2 against 1 (partner 3) and against 3 (partner 1)
    t, base = w_multi.t, w_multi.s
    bolt = w_multi.attack(base, "Thunderbolt")
    _check(t, _record(bolt, 3, "Lightningrod"), 99, a=2, d=1, case="d = 1: partner 3 records it")
    _check(
        t,
        _record(bolt, 1, "Lightningrod"),
        100,
        a=2,
        d=1,
        case="d = 1: battler 1 is not its partner",
    )
    _check(t, _record(bolt, 1, "Lightningrod"), 99, a=2, d=3, case="d = 3: partner 1")
