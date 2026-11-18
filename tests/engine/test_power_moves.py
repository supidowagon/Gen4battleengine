"""Night Shade / Seismic Toss, Hidden Power and Magnitude: power and type from the battler.

* ``effect_script_0087.s`` (``BATTLE_EFFECT_LEVEL_DAMAGE_FLAT``): damage = the attacker's
  level with ``SYSCTL_IGNORE_TYPE_CHECKS``; the chart's immunity bit is still set
  (``ApplyTypeMultiplier``, battle_lib.c:7544-7556).
* ``effect_script_0135.s`` + ``BtlCmd_CalcHiddenPowerParams`` (battle_script.c:6007-6033).
* ``effect_script_0126.s`` + ``BtlCmd_CalcMagnitudePower`` (battle_script.c:5820-5852) and
  ``subscript_magnitude.s``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.damage import (
    MAGNITUDE_POWER,
    ai_damage_for_roll,
    damage_for_roll,
    damage_table,
    fixed_damage,
    flail_power,
    hidden_power,
    move_type_of,
)
from nuzline.engine.ids import EffectId, Type
from nuzline.engine.observe import OBS_FIELDS, observation
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.step import step, step_events
from nuzline.prep.variants import flail_cuts, hp_cuts, hp_regions, max_hp_of
from tests.engine.helpers import A, assert_matches_reference, roark_state, sample_party

FIELDS = dict(OBS_FIELDS)
_step = jax.jit(step)
_events = jax.jit(step_events)


def decomp_hidden_power(ivs):
    """``CalcHiddenPowerParams`` transcribed (battle_script.c:6007-6033), ``ivs`` in the
    decomp's order (HP, Atk, Def, Spe, SpA, SpD)."""
    hp, atk, dfn, spe, spa, spd = ivs
    power = (
        ((hp & 2) >> 1) | (atk & 2) | ((dfn & 2) << 1) | ((spe & 2) << 2)
        | ((spa & 2) << 3) | ((spd & 2) << 4)
    )  # fmt: skip
    kind = (
        (hp & 1) | ((atk & 1) << 1) | ((dfn & 1) << 2) | ((spe & 1) << 3)
        | ((spa & 1) << 4) | ((spd & 1) << 5)
    )  # fmt: skip
    power = power * 40 // 63 + 30
    kind = kind * 15 // 63 + 1
    if kind >= Type.MYSTERY:
        kind += 1
    return kind, power


def stat_order(ivs):
    """Decomp order (HP, Atk, Def, Spe, SpA, SpD) -> ``ids.Stat`` order (HP, Atk, Def, SpA,
    SpD, Spe)."""
    hp, atk, dfn, spe, spa, spd = ivs
    return (hp, atk, dfn, spa, spd, spe)


def decomp_order(ivs):
    """``ids.Stat`` order -> the decomp's (HP, Atk, Def, Spe, SpA, SpD)."""
    hp, atk, dfn, spa, spd, spe = ivs
    return (hp, atk, dfn, spe, spa, spd)


def with_ivs(s, side, ivs):
    return s._replace(ivs=s.ivs.at[side, 0].set(jnp.asarray(ivs, s.ivs.dtype)))


def with_level(s, side, level):
    return s._replace(level=s.level.at[side, 0].set(level))


def outcome(*, crit=0, roll=15, magnitude=4):
    return empty_outcome()._replace(
        hit=jnp.ones((2, MAX_HITS), jnp.int8),
        crit=jnp.full((2, MAX_HITS), crit, jnp.int8),
        roll=jnp.full((2, MAX_HITS), roll, jnp.int8),
        magnitude=jnp.full((2,), magnitude, jnp.int8),
    )


def foe_hp_drop(tables, s, acts, out):
    after = _step(tables, s, acts, out)
    return int(s.hp[1, 0]) - int(after.hp[1, 0])


def as_plain(tables, names, move, *, kind=None, power=None):
    """``tables`` with ``move`` an ordinary hit of the given type and power."""
    m = names.move(move)
    tables = tables._replace(move_effect=tables.move_effect.at[m].set(EffectId.PLAIN))
    if kind is not None:
        tables = tables._replace(move_type=tables.move_type.at[m].set(kind))
    if power is not None:
        tables = tables._replace(move_power=tables.move_power.at[m].set(power))
    return tables


# ---------------------------------------------------------------------------- Hidden Power


def test_hidden_power_params():
    def params(ivs):
        kind, power = hidden_power(jnp.asarray(ivs, jnp.int32))
        return int(kind), int(power)

    assert params((31,) * 6) == (Type.DARK, 70)
    assert params((30,) * 6) == (Type.FIGHTING, 70)
    assert params((0,) * 6) == (Type.FIGHTING, 30)
    assert params((30, 30, 30, 30, 30, 31))[0] == Type.FLYING  # Spe odd: bit 3 -> 8
    assert params((30, 30, 30, 31, 30, 30))[0] == Type.GROUND  # SpA odd: bit 4 -> 16
    assert params((30, 31, 31, 30, 31, 30))[0] == Type.WATER
    # Every parity pattern of the low bit (type) crossed with every one of bit 1 (power).
    low = np.arange(64)[:, None] >> np.arange(6) & 1
    high = np.arange(64)[:, None] >> np.arange(6) & 1
    ivs = (high[:, None, :] * 2 + low[None, :, :]).reshape(-1, 6)  # decomp order
    kinds, powers = jax.vmap(hidden_power)(jnp.asarray([stat_order(v) for v in ivs], jnp.int32))
    expected = [decomp_hidden_power(tuple(int(x) for x in v)) for v in ivs]
    assert list(zip(kinds.tolist(), powers.tolist(), strict=True)) == expected
    assert {k for k, _ in expected}.isdisjoint({Type.NORMAL, Type.MYSTERY})
    assert min(p for _, p in expected) == 30 and max(p for _, p in expected) == 70


def test_move_type_of_reads_the_users_ivs(tables, names):
    s = roark_state(tables, names, player_species="Monferno", player_moves=("Hidden Power",))
    hp, tackle = names.move("Hidden Power"), names.move("Tackle")
    s = with_ivs(s, 0, (31,) * 6)
    assert int(move_type_of(tables, s, 0, hp)) == Type.DARK
    assert int(move_type_of(tables, s, 0, tackle)) == int(tables.move_type[tackle])
    # The user's own IVs: the foe's do not matter.
    t = with_ivs(s, 1, (0,) * 6)
    assert int(move_type_of(tables, t, 0, hp)) == Type.DARK
    assert int(move_type_of(tables, t, 1, hp)) == Type.FIGHTING


def test_hidden_power_damage(tables, names):
    s = roark_state(tables, names, player_species="Monferno", player_moves=("Hidden Power",))
    s = with_ivs(s, 0, (31,) * 6)
    hp = names.move("Hidden Power")
    plain = as_plain(tables, names, "Hidden Power", kind=Type.DARK, power=70)
    for crit in (False, True):
        real = jax.vmap(lambda r, c=crit: damage_for_roll(tables, s, 0, hp, c, r))(jnp.arange(16))
        ref = jax.vmap(lambda r, c=crit: damage_for_roll(plain, s, 0, hp, c, r))(jnp.arange(16))
        assert real.tolist() == ref.tolist()
        assert real.tolist() == damage_table(tables, s, 0, hp, crit).tolist()
        assert int(real[15]) > 0
    # Monferno is not Dark: no STAB; a Fire/Fighting user gets STAB only for its own types.
    water_ivs = (30, 31, 31, 30, 31, 30)  # ids.Stat order
    water = with_ivs(s, 0, water_ivs)
    assert int(move_type_of(tables, water, 0, hp)) == Type.WATER
    water_plain = as_plain(
        tables, names, "Hidden Power", kind=Type.WATER, power=decomp_hidden_power(decomp_order(water_ivs))[1]
    )  # fmt: skip
    assert (
        damage_table(tables, water, 0, hp, False).tolist()
        == damage_table(water_plain, water, 0, hp, False).tolist()
    )


def test_hidden_power_type_drives_stab_and_the_chart(tables, names):
    """A Fire-type Hidden Power gets Monferno's (Fire) STAB and the Fire chart row against
    Geodude, not Normal's."""
    s = roark_state(tables, names, player_species="Monferno", player_moves=("Hidden Power",))
    hp = names.move("Hidden Power")
    patterns = [tuple(v >> i & 1 for i in range(6)) for v in range(64)]
    bits = next(
        b for b in patterns if decomp_hidden_power(tuple(30 + x for x in b))[0] == Type.FIRE
    )
    ivs = tuple(30 + x for x in bits)
    s = with_ivs(s, 0, stat_order(ivs))
    assert int(move_type_of(tables, s, 0, hp)) == Type.FIRE
    kind, power = decomp_hidden_power(ivs)
    plain = as_plain(tables, names, "Hidden Power", kind=kind, power=power)
    stab = damage_table(tables, s, 0, hp, False).tolist()
    assert stab == damage_table(plain, s, 0, hp, False).tolist()
    # The same power as a Normal move (no STAB, Rock resists): strictly different.
    normal = as_plain(tables, names, "Hidden Power", kind=Type.NORMAL, power=power)
    assert stab != damage_table(normal, s, 0, hp, False).tolist()


# ------------------------------------------------------------------------------- Night Shade


def night_shade_state(tables, names, opp="Geodude", move="Night Shade", foresight=False):
    s = roark_state(
        tables,
        names,
        player_species="Spinarak",
        player_moves=(move,),
        opp_species=opp,
        opp_moves=("Leer",),
    )
    s = with_level(s, 0, 16)
    return s._replace(foresight=s.foresight.at[1].set(int(foresight)))


@pytest.mark.slow
def test_night_shade_deals_level(tables, names):
    s = night_shade_state(tables, names)
    acts = A("Night Shade", "Leer", s, names)
    for crit in (0, 1):
        for roll in range(16):
            assert foe_hp_drop(tables, s, acts, outcome(crit=crit, roll=roll)) == 16, (crit, roll)
    ns = names.move("Night Shade")
    assert damage_table(tables, s, 0, ns, False).tolist() == [16] * 16
    assert damage_table(tables, s, 0, ns, True).tolist() == [16] * 16
    fixed, amount = fixed_damage(tables, s, 0, ns)
    assert bool(fixed) and int(amount) == 16
    assert not bool(fixed_damage(tables, s, 0, names.move("Tackle"))[0])
    # The attacker's level, whatever it is (below the target's HP).
    big = with_level(s, 0, 25)
    assert foe_hp_drop(tables, big, acts, outcome()) == 25


def test_night_shade_type_immunity(tables, names):
    """Ghost into Normal: immune (the immunity bit is still set), 0."""
    s = night_shade_state(tables, names, opp="Bidoof")
    acts = A("Night Shade", "Leer", s, names)
    assert foe_hp_drop(tables, s, acts, outcome()) == 0
    assert damage_table(tables, s, 0, names.move("Night Shade"), False).tolist() == [0] * 16


def test_seismic_toss_foresight(tables, names):
    s = night_shade_state(tables, names, opp="Sableye", move="Seismic Toss")
    acts = A("Seismic Toss", "Leer", s, names)
    assert foe_hp_drop(tables, s, acts, outcome()) == 0  # Fighting into Ghost
    ident = night_shade_state(tables, names, opp="Sableye", move="Seismic Toss", foresight=True)
    # Foresight lifts the Ghost row; Sableye is Dark (Fighting x2) but damage is still the level.
    assert foe_hp_drop(tables, ident, acts, outcome()) == 16
    toss = names.move("Seismic Toss")
    assert damage_table(tables, ident, 0, toss, False).tolist() == [16] * 16


@pytest.mark.slow
def test_night_shade_crit_is_not_seen(tables, names):
    """No ``CalcCrit`` runs for it (effect_script_0087.s): the crit draw is ignored."""
    s = night_shade_state(tables, names)
    acts = A("Night Shade", "Leer", s, names)
    _, events = _events(tables, s, acts, outcome(crit=1))
    assert np.asarray(events["crit"][0]).tolist() == [0] * MAX_HITS
    assert int(events["hits"][0]) == 1
    # A normal damaging move shows it.
    t = night_shade_state(tables, names, move="Poison Sting")
    _, events = _events(tables, t, A("Poison Sting", "Leer", t, names), outcome(crit=1))
    assert int(events["crit"][0][0]) == 1


# --------------------------------------------------------------------------------- Magnitude


def magnitude_state(tables, names, opp="Bidoof"):
    return roark_state(
        tables,
        names,
        player_species="Geodude",
        player_moves=("Magnitude",),
        opp_species=opp,
        opp_moves=("Leer",),
    )


def test_magnitude_power_and_message(tables, names):
    assert MAGNITUDE_POWER.tolist() == [10, 30, 50, 70, 90, 110, 150]
    s = magnitude_state(tables, names)
    acts = A("Magnitude", "Leer", s, names)
    for level in range(4, 11):
        power = int(MAGNITUDE_POWER[level - 4])
        plain = as_plain(tables, names, "Magnitude", power=power)
        for roll in (0, 15):
            real = foe_hp_drop(tables, s, acts, outcome(magnitude=level, roll=roll))
            ref = foe_hp_drop(plain, s, acts, outcome(magnitude=level, roll=roll))
            assert real == ref and real > 0, (level, roll)
        after, events = _events(tables, s, acts, outcome(magnitude=level))
        obs = np.asarray(observation(events, after))
        assert obs[FIELDS["magnitude"]].tolist() == [level, 0]
    # The weakest and strongest hits differ a lot.
    low = foe_hp_drop(tables, s, acts, outcome(magnitude=4))
    high = foe_hp_drop(tables, s, acts, outcome(magnitude=10))
    assert high > low


def test_magnitude_into_a_flying_type(tables, names):
    """Ground into Flying: immune, no damage and (the message is a primary effect an
    ineffective hit does not run) no "Magnitude N!"."""
    s = magnitude_state(tables, names, opp="Staravia")
    acts = A("Magnitude", "Leer", s, names)
    assert foe_hp_drop(tables, s, acts, outcome(magnitude=10)) == 0
    after, events = _events(tables, s, acts, outcome(magnitude=10))
    assert np.asarray(observation(events, after))[FIELDS["magnitude"]].tolist() == [0, 0]


def test_magnitude_event_needs_the_move(tables, names):
    s = magnitude_state(tables, names)
    other = int(s.moves[0, 0, 1])
    assert other != names.move("Magnitude")
    acts = A(1, "Leer", s, names)  # the lead's second move
    after, events = _events(tables, s, acts, outcome(magnitude=9))
    assert int(events["act_id"][0]) == other
    assert np.asarray(observation(events, after))[FIELDS["magnitude"]].tolist() == [0, 0]
    missed = outcome(magnitude=9)._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    acts = A("Magnitude", "Leer", s, names)
    after, events = _events(tables, s, acts, missed)
    assert int(events["missed"][0]) == 1
    assert np.asarray(observation(events, after))[FIELDS["magnitude"]].tolist() == [0, 0]


# ----------------------------------------------------------------------------- the reference


def test_power_moves_match_reference(tables, names):
    # Hidden Power, several IV types, crit and roll variations.
    hp = roark_state(tables, names, player_species="Monferno", player_moves=("Hidden Power",))
    for ivs in ((31,) * 6, (30,) * 6, (30, 31, 31, 30, 31, 30), (31, 30, 31, 30, 30, 31)):
        s = with_ivs(hp, 0, ivs)
        for crit, roll in ((0, 0), (1, 15), (0, 7)):
            assert_matches_reference(
                tables, s, A("Hidden Power", "Tackle", s, names), outcome(crit=crit, roll=roll)
            )
    for ivs in range(0, 64, 5):
        v = stat_order(tuple(30 + (ivs >> i & 1) for i in range(6)))
        s = with_ivs(hp, 0, v)
        assert_matches_reference(tables, s, A("Hidden Power", "Tackle", s, names), outcome())
    # Hidden Power thawing a frozen foe depends on its IV type.
    for ivs in ((31,) * 6, stat_order((30, 30, 30, 30, 30, 30))):
        s = with_ivs(hp, 0, ivs)
        s = s._replace(status=s.status.at[1, 0].set(4))
        assert_matches_reference(tables, s, A("Hidden Power", "Tackle", s, names), outcome())
    # Night Shade / Seismic Toss: levels, immunities, Foresight.
    for opp, move, foresight in (
        ("Geodude", "Night Shade", False),
        ("Bidoof", "Night Shade", False),
        ("Sableye", "Seismic Toss", False),
        ("Sableye", "Seismic Toss", True),
    ):
        s = night_shade_state(tables, names, opp=opp, move=move, foresight=foresight)
        for crit, roll in ((0, 0), (1, 15)):
            assert_matches_reference(
                tables, s, A(move, "Leer", s, names), outcome(crit=crit, roll=roll)
            )
    # Magnitude at every level, into a neutral and an immune target.
    for opp in ("Bidoof", "Staravia", "Geodude"):
        s = magnitude_state(tables, names, opp=opp)
        for level in range(4, 11):
            assert_matches_reference(
                tables, s, A("Magnitude", "Tackle", s, names), outcome(magnitude=level, roll=3)
            )


# ------------------------------------------------------------------------------------ Flail
# effect_script_0099.s: CalcFlailPower, CalcCrit, CalcDamage. BtlCmd_CalcFlailPower
# (battle_script.c:4958-4970) reads App_PixelCount(curHP, maxHP, 64) (src/unk_0208C098.c:41-49:
# `cur * 64 / max`, at least 1 while cur > 0) and takes the first row of sHPPixelsToFlailPower
# (:4941-4948) whose bound satisfies `pixels <= bound` (not `<`).


def decomp_flail_power(cur, max_hp):
    """``BtlCmd_CalcFlailPower`` transcribed."""
    pixels = cur * 64 // max_hp
    if pixels == 0 and cur > 0:
        pixels = 1
    table = [(1, 200), (5, 150), (12, 100), (21, 80), (42, 40), (64, 20)]
    for bound, power in table:
        if pixels <= bound:
            return power
    return table[-1][1]


def flail_state(tables, names, move="Flail", hp=None, opp="Bidoof"):
    s = roark_state(
        tables,
        names,
        player_species="Luvdisc",
        player_moves=(move,),
        opp_species=opp,
        opp_moves=("Leer",),
    )
    return s if hp is None else s.replace_hp(0, 0, hp)


def test_flail_bands():
    # maxHP 42: pixels = hp * 64 // 42; the first hp of each band, and the one before it.
    expected = {1: 200, 2: 150, 3: 150, 4: 100, 8: 100, 9: 80, 14: 80, 15: 40, 28: 40, 29: 20}
    expected[42] = 20
    for hp, power in expected.items():
        assert int(flail_power(hp, 42)) == power == decomp_flail_power(hp, 42), hp
    assert flail_cuts(42) == (1, 3, 8, 14, 28)
    # The "at least 1" rule: 1 HP of 200 is 0 pixels before it, still the 200 band.
    assert 1 * 64 // 200 == 0 and int(flail_power(1, 200)) == 200
    # Every (cur, max) agrees with the transcription, and the bands are monotone in HP.
    for max_hp in (1, 2, 7, 33, 42, 64, 65, 100, 101, 255, 341, 714):
        powers = [int(flail_power(hp, max_hp)) for hp in range(1, max_hp + 1)]
        assert powers == [decomp_flail_power(hp, max_hp) for hp in range(1, max_hp + 1)]
        assert powers == sorted(powers, reverse=True)
        for cut in flail_cuts(max_hp):  # the cut is the last HP of its band
            if 1 <= cut < max_hp:
                assert powers[cut - 1] > powers[cut], (max_hp, cut)


def test_flail_damage_uses_the_band_power(tables, names):
    max_hp = int(flail_state(tables, names).max_hp[0, 0])
    for name in ("Flail", "Reversal"):
        s = flail_state(tables, names, name)
        acts = A(name, "Leer", s, names)
        for hp in sorted({1, 2, max_hp // 4, max_hp // 2, max_hp - 1, max_hp} - {0}):
            hurt = s.replace_hp(0, 0, hp)
            plain = as_plain(tables, names, name, power=decomp_flail_power(hp, max_hp))
            for roll in (0, 15):
                real = foe_hp_drop(tables, hurt, acts, outcome(roll=roll))
                assert real == foe_hp_drop(plain, hurt, acts, outcome(roll=roll)) > 0, (name, hp)
    # A 200-power hit really outdamages a 20-power one.
    s = flail_state(tables, names)
    acts = A("Flail", "Leer", s, names)
    low = foe_hp_drop(tables, s.replace_hp(0, 0, 1), acts, outcome())
    assert low > foe_hp_drop(tables, s, acts, outcome())


def test_flail_ai_damage_uses_power_one(tables, names):
    """``TrainerAI_CalcDamage`` has no Flail case (trainer_ai.c:2868-3109): the AI scores
    Flail at the move's table power (1), whatever the user's HP. Nor has it a Rollout case,
    so a curled user's doubled Rollout is not seen either."""
    s = flail_state(tables, names, "Flail", hp=1)
    flail = names.move("Flail")
    assert int(tables.move_power[flail]) == 1
    plain1 = as_plain(tables, names, "Flail", power=1)
    for roll in (0, 7, 15):
        ai = int(ai_damage_for_roll(tables, s, 0, flail, roll))
        assert ai == int(ai_damage_for_roll(plain1, s, 0, flail, roll))
        battle = int(damage_for_roll(tables, s, 0, flail, False, roll))
        assert battle > ai  # the real hit is the 200 band
    rollout = names.move("Rollout")
    curled = s._replace(curled=s.curled.at[0].set(1))
    for roll in (0, 15):
        assert int(ai_damage_for_roll(tables, curled, 0, rollout, roll)) == int(
            ai_damage_for_roll(tables, s, 0, rollout, roll)
        )
        assert int(damage_for_roll(tables, curled, 0, rollout, False, roll)) > int(
            damage_for_roll(tables, s, 0, rollout, False, roll)
        )


def test_mid_rollout_ai_damage_uses_table_power(tables, names):
    """``TrainerAI_CalcDamage`` passes power 0 for Rollout (its ``default`` case,
    trainer_ai.c:3074-3078), so ``BattleSystem_CalcMoveDamage`` reads the table power
    (battle_lib.c:6671-6675); the doubling exists only in ``BtlCmd_CalcRolloutPower``
    (battle_script.c:5661-5688). Mid-Rollout (locked, ``rollout_count`` < 5), curled or not,
    the AI still scores the table power (docs/discrepancies.md 8)."""
    rollout = names.move("Rollout")
    fresh = flail_state(tables, names, "Rollout")
    plain = as_plain(tables, names, "Rollout", power=int(tables.move_power[rollout]))
    for count in (4, 2, 1):
        rolling = fresh._replace(
            locked_move=fresh.locked_move.at[0].set(rollout),
            rollout_count=fresh.rollout_count.at[0].set(count),
        )
        for state in (rolling, rolling._replace(curled=rolling.curled.at[0].set(1))):
            for roll in (0, 15):
                ai = int(ai_damage_for_roll(tables, state, 0, rollout, roll))
                assert ai == int(ai_damage_for_roll(plain, state, 0, rollout, roll)), count
                assert int(damage_for_roll(tables, state, 0, rollout, False, roll)) > ai, count


def test_flail_hp_cuts(tables, names):
    flail, tackle = names.move("Flail"), names.move("Tackle")
    base = sample_party(names)[0]
    max_hp = max_hp_of(tables, base)
    plain = replace(base, moves=(tackle, 0, 0, 0))
    user = replace(base, moves=(tackle, flail, 0, 0))
    gained = set(hp_cuts(tables, user, max_hp)) - set(hp_cuts(tables, plain, max_hp))
    want = {c for c in flail_cuts(max_hp) if 1 <= c < max_hp}
    assert want and want <= set(hp_cuts(tables, user, max_hp))
    assert gained <= want
    # HP inside one region shares one Flail power.
    tops = hp_regions(tables, user, max_hp)
    for hi, lo in zip(tops, (*tops[1:], 0)):
        assert len({decomp_flail_power(h, max_hp) for h in range(lo + 1, hi + 1)}) == 1
    # Reversal shares the effect.
    reversal = replace(base, moves=(names.move("Reversal"), 0, 0, 0))
    assert want <= set(hp_cuts(tables, reversal, max_hp))


def test_flail_matches_reference(tables, names):
    for name in ("Flail", "Reversal"):
        s0 = flail_state(tables, names, name)
        max_hp = int(s0.max_hp[0, 0])
        hps = {1, 2, 3, 4, 8, 9, 14, 15, 28, 29, max_hp // 2, max_hp}
        for hp in sorted(h for h in hps if 1 <= h <= max_hp):
            s = s0.replace_hp(0, 0, hp)
            for crit, roll in ((0, 0), (1, 15), (0, 9)):
                assert_matches_reference(
                    tables, s, A(name, "Leer", s, names), outcome(crit=crit, roll=roll)
                )


def test_flail_and_reversal_map_in_both_games(tables, names, kaizo_tables, kaizo_names):
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for move in ("Flail", "Reversal"):
            assert int(t.move_effect[n.move(move)]) == EffectId.FLAIL, move
