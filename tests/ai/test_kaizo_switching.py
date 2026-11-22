"""Voluntary switching and post-KO replacement with Kaizo Roark's six (K3c Task 7;
docs/ai/kaizo_roark.md 11).

Decomp paths are relative to ``external/pokeplatinum``: ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``, ``battle_lib.c`` is ``src/battle/battle_lib.c``.

Kaizo runs ``TrainerAI_ShouldSwitch`` (trainer_ai.c:3894-3988) unchanged. Its draws that can
change the result are the keep draws (``RandNext() % 10 != 0`` per super-effective move of
the active, :3588) and the party passes (``RandNext() % rand == 0`` per bench Pokemon and
super-effective move, :3796, passes :3975 and :3981), and against a player's Wonder Guard
``AI_CannotDamageWonderGuard``'s first-success draws (:3338), only when the active has no
super-effective move (no keep draw), and ``AI_HasAbsorbAbilityInParty``'s bench draws
(:3681-3708). With five bench Pokemon that is at most ``20 + 4 + 5 + 2 * 5 * 4 = 69`` sites, each
part at its bound, ``KAIZO.switch_sites`` (rulings G-33 and 19 Nov, batch I-1a); vanilla keeps 9.

The exact switch part merges each stop into one categorical (``outcome.switch_stop_probs``)
and folds ``AI_HasAbsorbAbilityInParty``'s ``RandNext() % 3`` gate (:3651-3653) into it, which
changes the result only through the bench absorb draws behind it (:3655-3713; G-36, G-37).
``_loose_switch`` checks both: it walks the decomp's control flow with every ``RandNext`` read
its own independent draw, the :3651 read included (a failed gate skips the bench draws), and
enumerates every combination of the reads. It reads no switch site layout.
"""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, event, given, settings, target
from hypothesis import strategies as st

from nuzline.ai import policy
from nuzline.ai.layout import VANILLA, layout_of
from nuzline.ai.outcome import (
    SWITCH_NONE,
    SWITCH_SITES,
    sample_ai_outcome,
    switch_stop_probs,
)
from nuzline.ai.replacement import post_ko_switch_in
from nuzline.ai.switching import (
    ABILITY_IDS,
    INEFFECTIVE,
    NO_SWITCH,
    NOT_VERY_EFFECTIVE,
    SUPER_EFFECTIVE,
    SwitchSite,
    calc_effectiveness_flags,
    switch_action,
    switch_sites,
    switching_refusals,
    type_chart_flags,
)
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import move_type_of
from nuzline.engine.field import with_gravity
from nuzline.engine.ids import ABILITY_MULTITYPE
from nuzline.engine.state import MOVES, SLOTS, BattleState, make_state
from tests.ai.test_kaizo_outcome import KAIZO  # Roark's fight layout (E1 Ruling R7)
from tests.ai.test_kaizo_policy import _species_with
from tests.engine.helpers import as_species, gravity_of

K = SwitchSite
# Kaizo move ids (data/gen4_platinum_kaizo names; types and powers asserted in the tests).
ROCK_THROW, ACCELEROCK, ANCIENT_POWER, HEAD_SMASH, EARTHQUAKE = 88, 205, 246, 457, 89
ROCK4 = (ROCK_THROW, ACCELEROCK, ANCIENT_POWER, HEAD_SMASH)
# The four cited parts: 5 x 4 Wonder Guard draws (trainer_ai.c:3315-3344), 4 keep draws
# (:3576-3592), 5 absorb draws (:3681-3708), 2 x 5 x 4 pass draws (:3756-3804): 69.
KAIZO_SWITCH_SITES = 5 * 4 + 4 + 5 + 2 * 5 * 4
SPARE = KAIZO_SWITCH_SITES - (MOVES + 2 * (SLOTS - 1) * MOVES)  # 21 unused by the six-bench case
TYPE_STEEL = 8  # generated/pokemon_types.txt; checked in test_type_steel_id
# AI_HasAbsorbAbilityInParty's checkAbility by moveHit's table type (trainer_ai.c:3665-3674);
# type ids from generated/pokemon_types.txt, checked in test_type_steel_id.
TYPE_FIRE, TYPE_WATER, TYPE_ELECTRIC = 10, 11, 13
_CHECK_ABILITY = {TYPE_FIRE: "Flash Fire", TYPE_WATER: "Water Absorb", TYPE_ELECTRIC: "Volt Absorb"}

# The u16 residue counts of each read (BattleSystem_RandNext is a u16), counted here rather
# than taken from nuzline.ai.outcome.
_U16 = np.arange(65536)
P_KEEP = np.count_nonzero(_U16 % 10 != 0) / 65536  # :3588, keep (stop: no switch)
P_PASS_1 = np.count_nonzero(_U16 % 2 == 0) / 65536  # :3796 with rand 2 (:3975)
P_PASS_2 = np.count_nonzero(_U16 % 3 == 0) / 65536  # :3796 with rand 3 (:3981)
P_ABSORB = np.count_nonzero(_U16 % 3 != 0) / 65536  # :3651, the gate returns FALSE
P_ABSORB_SWITCH = np.count_nonzero(_U16 & 1) / 65536  # :3706
P_GUARD = np.count_nonzero(_U16 % 3 < 2) / 65536  # :3338, AI_CannotDamageWonderGuard (G-23)
# MoveIsOnDamagingTurn's listed battle effects (battle_lib.c:7588-7604); none is on its damaging
# turn at the AI's reads (SYSCTL_LAST_OF_MULTI_TURN cleared by BattleContext_Init).
_MULTI_TURN = (26, 39, 75, 145, 151, 155, 255, 256, 263, 273)

_SETTINGS = settings(
    deadline=None,
    max_examples=80,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _types(n, a: str, b: str):
    return jnp.asarray([n.type(a), n.type(b)], jnp.int8)


def _six_bench(t, n, party, moves=ROCK4) -> BattleState:
    """Kaizo Roark's six, Bonsly active, every Pokemon knowing ``moves``; the player's Gible
    made Fire/Flying and hit by Earthquake; the five bench Pokemon made Flying/Bug."""
    opp = [dataclasses.replace(p, moves=tuple(moves), pp=None) for p in party]
    s = make_state(t, [party[2]], opp)
    return s._replace(
        types=s.types.at[0, 0]
        .set(_types(n, "Fire", "Flying"))
        .at[1, 1:]
        .set(_types(n, "Flying", "Bug")),
        move_hit=s.move_hit.at[1].set(EARTHQUAKE),
    )


def _actions(t, s, sites: int) -> list[int]:
    stops = jnp.arange(sites + 1, dtype=jnp.int8)
    return np.asarray(jax.vmap(lambda st: switch_action(t, s, st, sites))(stops)).tolist()


# --- The six-bench layout -------------------------------------------------------------------


def test_type_steel_id(kaizo_names):
    assert kaizo_names.type("Steel") == TYPE_STEEL
    assert kaizo_names.type("Fire") == TYPE_FIRE and kaizo_names.type("Water") == TYPE_WATER
    assert kaizo_names.type("Electric") == TYPE_ELECTRIC


def test_kaizo_switch_sites_are_the_six_party_maximum():
    """4 keep draws (one per move slot, :3576-3592), 5 x 4 Wonder Guard draws (:3315-3344), 5
    absorb draws (one per bench Pokemon, :3681-3708) and, for each of the two passes, one draw
    per (bench Pokemon, move slot) (:3756-3804): five bench Pokemon in a party of six. The cap
    is the sum of the four parts, each at its bound (ruling 19 Nov); no count refusal."""
    assert KAIZO.switch_sites == KAIZO_SWITCH_SITES == 69
    assert VANILLA.switch_sites == SWITCH_SITES == 9


def test_six_bench_layout_fits(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Every bench member resists ``move_hit`` and is immune to it (both bits, so both passes
    reach it) and has four super-effective moves; so has the active. 44 sites, no refusal,
    and the exact stop probabilities."""
    t, n = kaizo_tables, kaizo_names
    s = _six_bench(t, n, kaizo_roark_party)
    for m in ROCK4:
        assert int(t.move_type[m]) == n.type("Rock") and int(t.move_power[m]) > 0, m
    assert int(t.move_type[EARTHQUAKE]) == n.type("Ground") and int(t.move_power[EARTHQUAKE]) > 0
    # UpateMoveStatusForTypeMul (battle_lib.c:2781) in chart-row order: Ground on Flying
    # sets INEFFECTIVE, then Ground on Bug sets NOT_VERY_EFFECTIVE on top of it.
    hit = int(calc_effectiveness_flags(t, EARTHQUAKE, _types(n, "Flying", "Bug"), 0))
    assert hit & INEFFECTIVE and hit & NOT_VERY_EFFECTIVE
    for m in ROCK4:  # super effective against Fire/Flying, for the keep and the passes
        assert int(type_chart_flags(t, m, s.types[0, 0], 0)) & SUPER_EFFECTIVE
        assert int(calc_effectiveness_flags(t, m, s.types[0, 0], 0)) & SUPER_EFFECTIVE

    assert layout_of(t) == KAIZO
    assert switching_refusals(t, s, KAIZO.switch_sites) == []
    sites = switch_sites(t, s, KAIZO.switch_sites).tolist()
    assert sites == (
        [K.KEEP] * 4
        + [K.PASS_INEFFECTIVE] * 20
        + [K.PASS_NOT_VERY_EFFECTIVE] * 20
        + [K.NONE] * SPARE
    )
    # Bench slots 1-5 in party order, four moves each, in each pass.
    bench = [MOVE_ACTIONS + j for j in range(1, SLOTS) for _ in range(MOVES)]
    assert _actions(t, s, KAIZO.switch_sites) == (
        [NO_SWITCH] * 4 + bench * 2 + [NO_SWITCH] * (SPARE + 1)
    )

    p = np.array([P_KEEP] * 4 + [P_PASS_1] * 20 + [P_PASS_2] * 20)
    reach = np.cumprod(np.concatenate([[1.0], 1 - p]))
    expected = np.concatenate([reach[:-1] * p, np.zeros(SPARE), reach[-1:]])
    got = np.asarray(switch_stop_probs(switch_sites(t, s, KAIZO.switch_sites)), np.float64)
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=0)
    assert abs(got.sum() - 1) < 1e-6

    # The whole AI: the switch part of the exact distribution is the same, and the state is
    # accepted.
    assert policy.ai_refusals(t, s, n) == []
    dist = np.asarray(policy.ai_action_distribution(t, s), np.float64)
    for j in range(1, SLOTS):
        mine = [i for i, a in enumerate(bench * 2) if a == MOVE_ACTIONS + j]
        want = sum(expected[4 + i] for i in mine)
        np.testing.assert_allclose(dist[MOVE_ACTIONS + j], want, rtol=2e-5, atol=0)
    assert policy.ai_width(t, s) <= MOVES * 2**KAIZO.slot_reads + KAIZO.switch_sites + 1


def test_kaizo_switch_sites_reach_sixty(kaizo_tables, kaizo_names, kaizo_roark_party):
    """G-33: the worst case of ``KAIZO_SWITCH_SITES``. The player's Fire/Flying active has
    Wonder Guard (``battleMons[].ability``, trainer_ai.c:3298: a Shedinja made Fire/Flying) and the AI's active knows no
    super-effective move (Tackle: neutral, so ``AI_OnlyIneffectiveMoves`` does not apply), so
    step 4 draws for each bench Pokemon's four
    super-effective Rock moves (:3338, 20 sites, no keep draw); each Flying/Bug bench Pokemon
    is INEFFECTIVE and NOT_VERY_EFFECTIVE against Earthquake (battle_lib.c:2762-2781), so both
    passes draw for it too (40 sites): 60, no refusal."""
    t, n = kaizo_tables, kaizo_names
    from nuzline.engine.ids import AbilityId

    s = _six_bench(t, n, kaizo_roark_party)
    s = s._replace(
        moves=s.moves.at[1, 0].set(jnp.asarray([n.move("Tackle")] * MOVES, s.moves.dtype)),
        # Battler_Ability reads the species' ability (damage.decomp_abilities): a Shedinja,
        # its types still made Fire/Flying.
        species=s.species.at[0, 0].set(n.species("Shedinja")),
        ability_slot=s.ability_slot.at[0, 0].set(0),
        ability=s.ability.at[0, 0].set(AbilityId.WONDER_GUARD),
    )
    assert switching_refusals(t, s, KAIZO.switch_sites) == []
    sites = switch_sites(t, s, KAIZO.switch_sites).tolist()
    assert sites == (
        [K.WONDER_GUARD] * 20
        + [K.PASS_INEFFECTIVE] * 20
        + [K.PASS_NOT_VERY_EFFECTIVE] * 20
        + [K.NONE] * (KAIZO_SWITCH_SITES - 60)
    )


def test_kaizo_switch_sites_reach_sixty_five(kaizo_tables, kaizo_names, kaizo_roark_party):
    """I-1a: the 65 sites that need no Weather Ball and that the count refusal used to stop (20 + 5 +
    40). As ``test_kaizo_switch_sites_reach_sixty`` (a Wonder Guard player, the AI's active with
    no super-effective move), plus ``AI_HasAbsorbAbilityInParty`` (trainer_ai.c:3640-3713): the
    player's ``moveHit`` is ThunderShock (table type Electric, :3665-3674) and each bench Pokemon
    has Volt Absorb (:3703), so 5 absorb draws (:3706-3708; no gate: the active has no
    super-effective move, :3651). Each bench Pokemon is Ground / Dragon, which ThunderShock
    ``INEFFECTIVE`` (Ground, battle_lib.c:2419) and ``NOT_VERY_EFFECTIVE`` (Dragon, :2421, after
    the immune row), so both passes draw for it (40)."""
    t, n = kaizo_tables, kaizo_names
    from nuzline.engine.ids import AbilityId

    s = _six_bench(t, n, kaizo_roark_party)
    s = s._replace(
        moves=s.moves.at[1, 0].set(jnp.asarray([n.move("Tackle")] * MOVES, s.moves.dtype)),
        species=s.species.at[0, 0].set(n.species("Shedinja")),
        ability_slot=s.ability_slot.at[0, 0].set(0),
        ability=s.ability.at[0, 0].set(AbilityId.WONDER_GUARD),
        move_hit=s.move_hit.at[1].set(n.move("ThunderShock")),
    )
    for j in range(1, SLOTS):
        s = as_species(t, s, 1, j, _species_with(t, ABILITY_IDS["Volt Absorb"]))
    s = s._replace(types=s.types.at[1, 1:].set(_types(n, "Ground", "Dragon")))
    assert switching_refusals(t, s, KAIZO.switch_sites) == []
    sites = switch_sites(t, s, KAIZO.switch_sites).tolist()
    assert sites == (
        [K.WONDER_GUARD] * 20
        + [K.ABSORB] * 5
        + [K.PASS_INEFFECTIVE] * 20
        + [K.PASS_NOT_VERY_EFFECTIVE] * 20
        + [K.NONE] * (KAIZO_SWITCH_SITES - 65)
    )


def test_six_bench_sampled_stops(kaizo_tables, kaizo_names, kaizo_roark_party):
    """``sample_ai_outcome`` under the Kaizo layout draws one u16 per site, so its stops
    range over 0..69 (``KAIZO.switch_sites`` meaning none)."""
    t = kaizo_tables
    s = _six_bench(t, kaizo_names, kaizo_roark_party)
    keys = jax.random.split(jax.random.PRNGKey(3), 4000)
    stops = np.asarray(jax.vmap(lambda k: sample_ai_outcome(k, t, s, KAIZO).switch_stop)(keys))
    assert stops.min() >= 0 and stops.max() <= KAIZO.switch_sites
    # Keep site 0 stops 90% of the time; the four keep sites leave 1e-4.
    assert abs(np.mean(stops == 0) - P_KEEP) < 0.02


def test_six_bench_policy_switches(kaizo_tables, kaizo_names, kaizo_roark_party):
    """``ai_policy`` under the Kaizo layout maps each stop as ``switch_action`` does: a pass
    stop of bench slot j is the switch ``4 + j``, the none stop (69) the move pick."""
    t = kaizo_tables
    s = _six_bench(t, kaizo_names, kaizo_roark_party)
    o = sample_ai_outcome(jax.random.PRNGKey(0), t, s, KAIZO)
    for stop, action in ((4, MOVE_ACTIONS + 1), (43, MOVE_ACTIONS + 5), (24, MOVE_ACTIONS + 1)):
        assert int(policy.ai_policy(t, s, o._replace(switch_stop=jnp.int8(stop)))) == action
    none = int(policy.ai_policy(t, s, o._replace(switch_stop=jnp.int8(KAIZO.switch_sites))))
    assert 0 <= none < MOVES


# --- Vanilla is unchanged -------------------------------------------------------------------


def test_vanilla_switch_sites_unchanged(tables, names):
    """The vanilla layout keeps ``SWITCH_SITES`` (9) and ``SWITCH_NONE`` (9), and the default
    of every switching function is the vanilla count: roark.md 13, trace F."""
    from tests.ai.test_switching import _trace_f

    assert (SWITCH_SITES, SWITCH_NONE, VANILLA.switch_sites) == (9, 9, 9)
    s = _trace_f(tables, names)
    expected = [K.PASS_NOT_VERY_EFFECTIVE] * 4 + [K.NONE] * 5
    assert switch_sites(tables, s).tolist() == expected
    assert switch_sites(tables, s, VANILLA.switch_sites).tolist() == expected
    assert _actions(tables, s, 9) == [4, 4, 5, 5] + [NO_SWITCH] * 6
    o = sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    assert 0 <= int(o.switch_stop) <= SWITCH_NONE
    q = 43690 / 65536
    dist = np.asarray(policy.ai_action_distribution(tables, s), np.float64)
    assert abs(dist[4] - (1 - q**2)) < 1e-6 and abs(dist[5] - q**2 * (1 - q**2)) < 1e-6


# --- The loose oracle -----------------------------------------------------------------------


class _Skip(int):
    """A read's ``action`` that, on probability ``p``, skips the next ``int`` reads (unread)."""


def _loose_reads(t, s: BattleState) -> list[tuple[float, int]]:
    """The ``RandNext`` reads ``TrainerAI_ShouldSwitch`` (trainer_ai.c:3894) makes on the
    longest path, in order, each ``(p, action)``: with probability ``p`` (its u16 residue
    count) the read stops the check with ``action``, or, for ``action = None``, takes the
    branch that still goes on, as its other branch does; for a ``_Skip(k)`` action it goes on
    past the next k reads, which are not made. Written from the decomp's control flow, step
    by step, on concrete numpy values; it reads no switch site layout."""
    a, d = int(s.active[1]), int(s.active[0])
    species = np.asarray(s.species)
    hp = np.asarray(s.hp)
    row = np.asarray(t.species_form_row)[species, np.asarray(s.form)]
    ability = np.asarray(t.species_abilities_decomp)[row, np.asarray(s.ability_slot)]
    d_types = s.types[0, d]
    gravity = int(gravity_of(s))
    reads: list[tuple[float, int | None]] = []  # (P(the read's branch), action or None)

    # :3906-3913: trapped (Bind, Mean Look), ingrained, Shadow Tag / Arena Trap on the player's
    # side, Magnet Pull on Steel. ``CountAbility`` compares ``Battler_Ability`` (battle_lib.c:
    # 2896-2903, :2933-2940, :3088-3106): ABILITY_NONE under Gastro Acid unless Multitype.
    trap_ability = (
        0
        if int(s.ability_suppressed[0]) and int(ability[0, d]) != ABILITY_MULTITYPE
        else int(ability[0, d])
    )
    if (
        int(s.trapped[1]) > 0
        or int(s.mean_look[1]) != 0
        or int(s.ingrain[1]) != 0
        or trap_ability in (ABILITY_IDS["Shadow Tag"], ABILITY_IDS["Arena Trap"])
        or (
            trap_ability == ABILITY_IDS["Magnet Pull"]
            and TYPE_STEEL in np.asarray(s.types[1, a]).tolist()
        )
    ):
        return reads
    # :3924-3938: living party members other than the active (aiSwitchedPartySlot is 6:
    # battle_lib.c:2015, BattleContext_Init at battle_controller_player.c:1855 and :3998).
    bench = [j for j in range(SLOTS) if j != a and species[1, j] > 0 and hp[1, j] > 0]
    if not bench:
        return reads
    # :3941-3959. AI_PerishSongKO (step 3: these states never set Perish Song; its switch is
    # tested in test_switching.py, G-64), AI_OnlyIneffectiveMoves and Natural Cure are refused
    # (the caller assumes no refusal), so these return FALSE (Wonder Guard's check and
    # AI_HasAbsorbAbilityInParty are below).
    a_moves = [int(m) for m in np.asarray(s.moves[1, a])]
    identified = int(s.foresight[0]) != 0

    def keep_se(m):  # BattleSystem_ApplyTypeChart flags (:3582), TrainerAI_MoveType (:3578)
        return (
            m != 0 and int(type_chart_flags(t, m, d_types, gravity, identified)) & SUPER_EFFECTIVE
        )

    # :3943, AI_CannotDamageWonderGuard (:3286-3348): the player's battleMons[].ability (:3298)
    # is Wonder Guard and no move of the active is super effective (ApplyTypeChart, :3306); then
    # each bench Pokemon's super-effective moves (CalcEffectiveness, :3328) in party order draw
    # RandNext() % 3 < 2 (:3338), a success switching to it.
    if int(ability[0, d]) == ABILITY_IDS["Wonder Guard"] and not any(keep_se(m) for m in a_moves):
        for j in bench:
            for m in (int(x) for x in np.asarray(s.moves[1, j])):
                if m and int(calc_effectiveness_flags(t, m, d_types, gravity)) & SUPER_EFFECTIVE:
                    reads.append((P_GUARD, MOVE_ACTIONS + j))
    # :3947, AI_HasAbsorbAbilityInParty (:3640-3713): a moveHit with power (:3656-3663) of table
    # type Fire / Water / Electric names checkAbility (:3665-3674); the active's own one holds
    # the switch (:3677-3679); each bench holder of it, in party order, draws RandNext() & 1
    # (:3703-3708), a success switching to it.
    hit = int(s.move_hit[1])
    absorb: list[tuple[float, int]] = []
    hit_type = int(t.move_type[hit])
    if hit != 0 and int(t.move_power[hit]) != 0 and hit_type in _CHECK_ABILITY:
        check = ABILITY_IDS[_CHECK_ABILITY[hit_type]]
        if int(ability[1, a]) != check:
            absorb = [(P_ABSORB_SWITCH, MOVE_ACTIONS + j) for j in bench if ability[1, j] == check]
    if any(keep_se(m) for m in a_moves):
        # :3651: with a super-effective move (flag TRUE: no draw inside, :3586-3587) % 3 != 0
        # returns FALSE (:3652) before the bench draws; otherwise they follow.
        reads.append((P_ABSORB, _Skip(len(absorb)) if absorb else None))
    reads.extend(absorb)
    # :3964, AI_HasSuperEffectiveMove(..., FALSE) (:3560-3592).
    for m in a_moves:
        if keep_se(m):
            reads.append((P_KEEP, NO_SWITCH))
    # :3969, AI_IsHeavilyStatBoosted (:3872-3884).
    if int(np.maximum(np.asarray(s.stages[1]).astype(int), 0).sum()) >= 4:
        return reads
    # :3975 and :3981, AI_HasPartyMemberWithSuperEffectiveMove (:3727-3807).
    hit = int(s.move_hit[1])
    if hit == 0 or int(t.move_power[hit]) == 0:  # :3737-3744
        return reads
    hit_type = move_type_of(t, s, 0, hit)  # :3767, the move as its user used it
    for mask, p in ((INEFFECTIVE, P_PASS_1), (NOT_VERY_EFFECTIVE, P_PASS_2)):
        for j in bench:
            eff = int(calc_effectiveness_flags(t, hit, s.types[1, j], gravity, hit_type))
            # CalcEffectiveness (battle_lib.c:2734-2739): the bench Pokemon's Wonder Guard
            # (MON_DATA_ABILITY) against a hitter (moveHitBattler) without Mold Breaker.
            if (
                int(ability[1, j]) == ABILITY_IDS["Wonder Guard"]
                and int(ability[0, d]) != ABILITY_IDS["Mold Breaker"]
                and int(t.move_ai_effect[hit]) not in _MULTI_TURN
                and not eff & SUPER_EFFECTIVE
            ):
                eff |= INEFFECTIVE
            if not eff & mask:
                continue
            for m in (int(x) for x in np.asarray(s.moves[1, j])):
                if m and int(calc_effectiveness_flags(t, m, d_types, gravity)) & SUPER_EFFECTIVE:
                    reads.append((p, MOVE_ACTIONS + j))
    return reads


def _loose_switch(t, s: BattleState) -> np.ndarray:
    """``float64[MOVE_ACTIONS + SLOTS]``: P(switch to slot j) at ``MOVE_ACTIONS + j``, by
    enumerating every combination of every read's outcome (each read its own draw), walking
    the reads in order to the first that stops the check."""
    reads = _loose_reads(t, s)
    out = np.zeros(MOVE_ACTIONS + SLOTS)
    for bits in itertools.product((0, 1), repeat=len(reads)):
        weight = 1.0
        for (p, _), b in zip(reads, bits, strict=True):
            weight *= p if b else 1 - p
        if weight == 0:
            continue
        skip = 0
        for (_, action), b in zip(reads, bits, strict=True):
            if skip:  # not read: its two outcomes sum out
                skip -= 1
            elif b and isinstance(action, _Skip):
                skip = int(action)
            elif b and action is not None:
                if action != NO_SWITCH:
                    out[action] += weight
                break
    return out


@st.composite
def _switch_states(draw, t, n, party):
    """Kaizo Roark's six in a random order with a random active slot (so the bench sits on
    both sides of it, as after a switch or a replacement), fainted bench members, player
    types, ``move_hit`` and gates; bench types drawn so that the passes are reached."""
    order = draw(st.permutations(range(SLOTS)))
    s = make_state(t, [party[2]], [party[i] for i in order])
    active = draw(st.integers(0, SLOTS - 1))
    bench = [j for j in range(SLOTS) if j != active]
    event(f"active slot {active}")
    s = s._replace(active=s.active.at[1].set(active))
    type_names = [
        "Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost",
        "Steel", "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
    ]  # fmt: skip
    pair = st.tuples(st.sampled_from(type_names), st.sampled_from(type_names))
    d1, d2 = draw(pair)
    s = s._replace(types=s.types.at[0, 0].set(_types(n, d1, d2)))
    alive = draw(st.lists(st.booleans(), min_size=SLOTS - 1, max_size=SLOTS - 1))
    hp = s.hp
    for j, keep in zip(bench, alive, strict=True):
        if not keep:
            hp = hp.at[1, j].set(0)
        if draw(st.booleans()):
            b1, b2 = draw(pair)
            s = s._replace(types=s.types.at[1, j].set(_types(n, b1, b2)))
    hits = ("Earthquake", "Tackle", "Ember", "Water Gun", "Wing Attack", "Karate Chop",
            "ThunderShock", "Psybeam", "Bite", "Shadow Ball", "Leer", "Fire Fang",
            "Weather Ball")  # fmt: skip
    # G-23: Shedinja (Wonder Guard) as the player's active or on the AI's bench (its types stay
    # the drawn ones: only its ability matters here).
    shedinja = n.species("Shedinja")
    if draw(st.integers(0, 3)) == 0:
        s = as_species(t, s, 0, 0, shedinja)
    if draw(st.integers(0, 3)) == 0:
        j = draw(st.sampled_from(bench))
        s = as_species(t, s, 1, j, shedinja)
    # G-36: Flash Fire / Water Absorb / Volt Absorb holders on the AI's side, the active too.
    for _ in range(draw(st.integers(0, 2))):
        name = draw(st.sampled_from(sorted(set(_CHECK_ABILITY.values()))))
        j = draw(st.sampled_from(range(SLOTS)))
        s = as_species(t, s, 1, j, _species_with(t, ABILITY_IDS[name]))
    hit = n.move(draw(st.sampled_from(hits)))
    gravity = draw(st.sampled_from((0, 0, 0, 5)))
    s = s._replace(
        hp=hp,
        move_hit=s.move_hit.at[1].set(hit),
        foresight=s.foresight.at[0].set(draw(st.sampled_from((0, 0, 1)))),
        trapped=s.trapped.at[1].set(draw(st.sampled_from((0, 0, 0, 0, 2)))),
        mean_look=s.mean_look.at[1].set(draw(st.sampled_from((0, 0, 0, 0, 1)))),
        ingrain=s.ingrain.at[1].set(draw(st.sampled_from((0, 0, 0, 0, 1)))),
        stages=s.stages.at[1, 1].set(draw(st.sampled_from((0, 0, 0, 2, 4)))),
    )
    return with_gravity(s, gravity)


def _assert_switch_equals_loose(t, s: BattleState) -> None:
    exact = np.asarray(policy._distribution(t, s, KAIZO), np.float64)
    loose = _loose_switch(t, s)
    np.testing.assert_allclose(
        exact[MOVE_ACTIONS : MOVE_ACTIONS + SLOTS],
        loose[MOVE_ACTIONS : MOVE_ACTIONS + SLOTS],
        atol=2e-6,
    )


@pytest.mark.slow
@_SETTINGS
@given(data=st.data())
def test_roark_switch_distribution_matches_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, data
):
    """The switch part of the exact Kaizo distribution (``policy._distribution``: the
    layout's stop categorical and ``switch_action``) equals the loose enumeration, on states
    with at most 12 reads (the loose side enumerates 2^reads combinations)."""
    t, n = kaizo_tables, kaizo_names
    s = data.draw(_switch_states(t, n, kaizo_roark_party))
    assume(not switching_refusals(t, s, KAIZO.switch_sites))
        # I-2: Gastro Acid on the player's active (Battler_Ability reads none, lib.c:3088-3106).
        ability_suppressed=s.ability_suppressed.at[0].set(draw(st.sampled_from((0, 0, 1)))),
    assume(not policy.ai_refusals(t, s, n))
    reads = _loose_reads(t, s)
    assume(len(reads) <= 12)
    passes = sum(
        action not in (None, NO_SWITCH) and not isinstance(action, _Skip) for _, action in reads
    )
    event(f"party-pass reads: {min(passes, 6)}{'+' if passes >= 6 else ''}")
    target(float(passes))
    _assert_switch_equals_loose(t, s)


def _small_six_bench(t, n, party, alive: int, moves=ROCK4) -> BattleState:
    """``_six_bench`` with the active knowing Tackle only (no keep draw) and only the first
    ``alive`` bench Pokemon alive."""
    s = _six_bench(t, n, party, moves)
    s = s._replace(moves=s.moves.at[1, 0].set(jnp.asarray([33, 0, 0, 0], s.moves.dtype)))
    return s._replace(hp=s.hp.at[1, 1 + alive :].set(0))


@pytest.mark.parametrize(
    ("alive", "moves", "reads"),
    [
        (1, ROCK4, 8),  # one bench Pokemon, 4 moves, both passes
        (2, (ROCK_THROW, ACCELEROCK, 0, 0), 8),
        (5, (ROCK_THROW, 0, 0, 0), 10),  # every bench slot, one move each
        (5, (0, 0, 0, HEAD_SMASH), 10),  # the last move slot only, up to the last site
    ],
)
def test_small_six_bench_layouts_match_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, alive, moves, reads
):
    """Constructed small layouts of the six-bench state, every bench slot and move slot
    reachable: exact against loose."""
    t, n = kaizo_tables, kaizo_names
    s = _small_six_bench(t, n, kaizo_roark_party, alive, moves)
    assert len(_loose_reads(t, s)) == reads
    assert policy.ai_refusals(t, s, n) == []
    _assert_switch_equals_loose(t, s)


@pytest.mark.parametrize("alive", [1, 2])
def test_wonder_guard_layout_matches_loose(kaizo_tables, kaizo_names, kaizo_roark_party, alive):
    """G-23: the player's active is Shedinja (Wonder Guard) and the AI's active knows only
    Tackle: AI_CannotDamageWonderGuard draws once per super-effective bench move (four Rock
    moves each), before the passes; exact against loose."""
    t, n = kaizo_tables, kaizo_names
    s = _small_six_bench(t, n, kaizo_roark_party, alive)
    s = as_species(t, s, 0, 0, n.species("Shedinja"))
    reads = _loose_reads(t, s)
    assert [p for p, _ in reads[: 4 * alive]] == [P_GUARD] * (4 * alive)
    sites = switch_sites(t, s, KAIZO.switch_sites).tolist()
    assert sites[: 4 * alive] == [K.WONDER_GUARD] * (4 * alive)
    assert policy.ai_refusals(t, s, n) == []
    _assert_switch_equals_loose(t, s)


def test_loose_counts_the_absorb_read(kaizo_tables, kaizo_names, kaizo_roark_party):
    """``AI_HasAbsorbAbilityInParty``'s ``RandNext() % 3`` (:3651) is a loose read whenever
    the active has a super-effective move, and the exact layout has no site for it; the two
    still agree (the read cannot change the result)."""
    t, n = kaizo_tables, kaizo_names
    s = _six_bench(t, n, kaizo_roark_party, (ROCK_THROW, 0, 0, 0))
    s = s._replace(hp=s.hp.at[1, 2:].set(0))
    reads = _loose_reads(t, s)
    assert reads[0] == (P_ABSORB, None) and len(reads) == 1 + 1 + 2
    assert switch_sites(t, s, KAIZO.switch_sites).tolist()[:4] == [
        K.KEEP,
        K.PASS_INEFFECTIVE,
        K.PASS_NOT_VERY_EFFECTIVE,
        K.NONE,
    ]
    _assert_switch_equals_loose(t, s)


@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize(
    ("hit", "holder"), [("Water Gun", "Water Absorb"), ("Ember", "Flash Fire")]
)
def test_absorb_bench_matches_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, keep, hit, holder
):
    """G-36, G-37: two bench holders of the absorb ability ``hit`` names; with ``keep`` the
    active's Rock Throw is super effective, so AI_HasAbsorbAbilityInParty's gate (:3651) is read
    first and a failure skips both bench draws (:3706). Exact against loose."""
    t, n = kaizo_tables, kaizo_names
    s = _six_bench(t, n, kaizo_roark_party, (ROCK_THROW, 0, 0, 0))
    s = s._replace(hp=s.hp.at[1, 3:].set(0), move_hit=s.move_hit.at[1].set(n.move(hit)))
    for j in (1, 2):
        s = as_species(t, s, 1, j, _species_with(t, ABILITY_IDS[holder]))
    if not keep:
        s = s._replace(moves=s.moves.at[1, 0].set(jnp.asarray([33, 0, 0, 0], s.moves.dtype)))
    reads = _loose_reads(t, s)
    gate = [(P_ABSORB, _Skip(2))] if keep else []
    assert reads[: len(gate) + 2] == gate + [(P_ABSORB_SWITCH, MOVE_ACTIONS + j) for j in (1, 2)]
    kind = K.ABSORB_GATED if keep else K.ABSORB
    assert switch_sites(t, s, KAIZO.switch_sites).tolist()[:2] == [kind] * 2
    assert switching_refusals(t, s, KAIZO.switch_sites) == []
    assert policy.ai_refusals(t, s, n) == []
    _assert_switch_equals_loose(t, s)


# --- Replacement ----------------------------------------------------------------------------


def test_roark_moves_never_reach_the_stale_stage_2_compare(kaizo_tables, kaizo_roark_party):
    """The refusal of ``post_ko_switch_in`` needs the first candidate's move slot 0 empty or
    of power 1 (``if (move && MOVE_DATA(move).power != 1)``, battle_lib.c:8053, with the
    compare after it at :8080). Every one of Kaizo Roark's six has a move in slot 0 whose
    power is not 1."""
    t = kaizo_tables
    for spec in kaizo_roark_party:
        assert spec.moves[0] != 0 and int(t.move_power[spec.moves[0]]) != 1, spec.species


@pytest.mark.parametrize("player", ["Gible", "Fighting/Psychic", "Ghost/Dark", "Normal/Normal"])
@pytest.mark.parametrize("power_mul", [10, 20])
def test_roark_replacement_each_faint_order(
    kaizo_tables, kaizo_names, kaizo_roark_party, player, power_mul
):
    """Every state a faint order of Kaizo Roark's six reaches: the fainted active, and any
    nonempty set of the others still alive. ``BattleAI_PostKOSwitchIn`` (battle_lib.c:7923)
    never meets its refused stale compare, and picks a living non-active slot.

    Against the Ghost/Dark and Fighting/Psychic players no one of the six has a move super
    effective against the player (checked), so every state reaches stage 2 (:8034-8088);
    against Normal/Normal only Bonsly's Brick Break is, so stage 2 runs once Bonsly is gone."""
    t, n = kaizo_tables, kaizo_names
@pytest.mark.parametrize("suppressed", [False, True])
def test_gastro_acid_trap_gate_matches_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, suppressed
):
    """I-2: a Wobbuffet (Shadow Tag) player active gates the switch (trainer_ai.c:3908), unless
    Gastro Acid hid its ability from ``CountAbility`` (``Battler_Ability``, battle_lib.c:2896-2903,
    :3088-3106). The loose reference applies ``Battler_Ability`` too."""
    t, n = kaizo_tables, kaizo_names
    s = _six_bench(t, n, kaizo_roark_party)
    slot = n.species_abilities(n.species("Wobbuffet")).index(n.ability("Shadow Tag"))
    types = s.types[0, 0]
    s = as_species(t, s, 0, 0, n.species("Wobbuffet"), slot)
    s = s._replace(types=s.types.at[0, 0].set(types))
    if suppressed:
        s = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    gated = not suppressed
    assert (_loose_reads(t, s) == []) == gated
    assert (set(switch_sites(t, s, KAIZO.switch_sites).tolist()) == {K.NONE}) == gated
    _assert_switch_equals_loose(t, s)


    party = kaizo_roark_party
    s = make_state(t, [party[2]], list(party))
    if player != "Gible":
        a, b = player.split("/")
        s = s._replace(types=s.types.at[0, 0].set(_types(n, a, b)))
    d_types = s.types[0, 0]
    se = [
        any(
            m and int(calc_effectiveness_flags(t, m, d_types, 0)) & SUPER_EFFECTIVE
            for m in spec.moves
        )
        for spec in party
    ]
    if player in ("Fighting/Psychic", "Ghost/Dark"):
        assert not any(se)
    if player == "Normal/Normal":
        assert se == [True] + [False] * 5
    for active in range(SLOTS):
        others = [j for j in range(SLOTS) if j != active]
        for mask in range(1, 2 ** len(others)):
            alive = {j for k, j in enumerate(others) if mask >> k & 1}
            hp = s.hp.at[1, active].set(0)
            for j in others:
                if j not in alive:
                    hp = hp.at[1, j].set(0)
            state = s._replace(active=s.active.at[1].set(active), hp=hp)
            pick = int(post_ko_switch_in(t, state, power_mul))  # raises on the refusal
            assert pick in alive, (active, sorted(alive), pick)


# (player, fainted active, alive others) -> the pick, hand-traced from battle_lib.c:7923-8089.
# Stage 1 (:7959-8029) takes the first highest TypeMatchupMultiplier sum that has a move
# super effective against the player, disregarding the others; stage 2 (:8038-8088) the
# first highest damage of a candidate's move used by the fainted battler (its own stats).
_PICKS = {
    # Player Gible (Dragon/Ground): Gible (Dragon/Ground, 120) is the only one with a
    # super-effective move (DragonBreath), so it is picked whenever it is alive.
    ("Gible", 0, (1, 2, 5)): 2,
    ("Gible", 5, (0, 1, 2, 3, 4)): 2,
    # Gible fainted: nobody else has a super-effective move, so stage 2 with Gible's stats
    # (task-7-review [vanilla-replacement]: Selfdestruct 89, Sand Tomb 31, Crunch 22,
    # Water Pulse 9).
    ("Gible", 2, (3, 4)): 4,
    ("Gible", 2, (0, 1, 3)): 0,
    ("Gible", 2, (1, 5)): 5,
    ("Gible", 2, (3,)): 3,
    # Player Normal/Normal: only Bonsly's Brick Break is super effective, so stage 1
    # disregards everyone else, in party order or not, until Bonsly (slot 0).
    ("Normal/Normal", 5, (0, 1, 2, 3, 4)): 0,
    ("Normal/Normal", 3, (4, 0)): 0,
    # A stage-2 tie between slots 4 and 5 (slot 5 given Shuckle's moves, so both score the
    # same: the moves are used by the fainted Gible): the strict `maxScore < score` (:8080)
    # keeps the first, slot 4.
    ("Gible", 2, (4, 5), "tie"): 4,
}


@pytest.mark.parametrize("case", sorted(_PICKS), ids=str)
def test_roark_replacement_picks(kaizo_tables, kaizo_names, kaizo_roark_party, case):
    """K3c final review: the six-Pokemon picks of ``post_ko_switch_in`` pinned, not only
    "a living slot" (``test_roark_replacement_each_faint_order``)."""
    t, n = kaizo_tables, kaizo_names
    player, active, alive = case[:3]
    s = make_state(t, [kaizo_roark_party[2]], list(kaizo_roark_party))
    if case[3:] == ("tie",):
        s = s._replace(moves=s.moves.at[1, 5].set(s.moves[1, 4]))
    if player != "Gible":
        a, b = player.split("/")
        s = s._replace(types=s.types.at[0, 0].set(_types(n, a, b)))
    hp = s.hp.at[1, active].set(0)
    for j in range(SLOTS):
        if j != active and j not in alive:
            hp = hp.at[1, j].set(0)
    state = s._replace(active=s.active.at[1].set(active), hp=hp)
    assert int(post_ko_switch_in(t, state, 10)) == _PICKS[case]
