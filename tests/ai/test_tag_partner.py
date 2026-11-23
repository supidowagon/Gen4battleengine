"""``TagStrategy_Partner`` (script.s:7339-7694), the Tag Strategy ally part at B = 4 (A-9).

The attacker is battler 1, the target its ally, battler 3 (``d = 3``): ``IfTargetIsPartner`` jumps
(trainer_ai.c:2630-2639) and the ally is ``ATTACKER_PARTNER`` and ``DEFENDER`` at once. The runner,
the fixed world and the law walker are ``test_tag_doubles``'s (the exact law of slot 0's score from 100
over the enumeration patterns' draws). Laws are written by hand from ``script.s``: ``IfRandomLessThan N``
jumps with probability N / 256. The ally's hp% is ``hp * 100 / max_hp`` (max 100 here).
"""

from __future__ import annotations

import pytest

from nuzline.ai import outcome as ai_outcome
from nuzline.ai.layout import CHECK_HP, TAG_STRATEGY
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import Stat, Status
from tests.ai.test_tag_doubles import (
    BUG,
    ELECTRIC,
    FIRE,
    GROUND,
    NORMAL,
    POISON,
    _ability,
    _at,
    _check,
    _flag,
    _hp,
    _record,
    _speeds,
    _status,
    _two_abilities,
    _types,
    _World,
)

A, ALLY = 1, 3
ITEM = {"Toxic Orb": 272, "Flame Orb": 273, "Persim Berry": 156, "Lum Berry": 157}


@pytest.fixture(scope="module")
def w(tables, names):
    return _World(tables, names, DOUBLES, A)


def _item(s, name, b=ALLY):
    p, slot = _at(s, b)
    return s._replace(item=s.item.at[p, slot].set(ITEM[name]))


def _stage(s, b, stat, engine_stage):
    return s._replace(stages=s.stages.at[b, stat].set(engine_stage))


def ally(world, t, s, want, **kw):
    kw.setdefault("a", A)
    _check(t, s, want, d=ALLY, **kw)


def _plus3(n):
    """The ladder's law: +3 when the roll jumps (P n / 256)."""
    p = n / 256
    return {103: p, 100: 1 - p}


def test_partner_site_n_is_the_scripts():
    expected = {7383: 160, 7395: 64, 7399: 128, 7403: 192, 7430: 64, 7434: 128, 7438: 192,
                7596: 64, 7679: 128, 7682: 80}  # fmt: skip
    for line, n in expected.items():
        assert ai_outcome.SITE_N[line] == n


def test_partner_ids_are_the_decomps(names):
    from nuzline.ai import commands as c

    for key in ("Will-O-Wisp", "Swagger", "Gastro Acid", "Acupressure", "Fling", "Fire Blast", "Thunder",
                "Cross Chop", "Hydro Pump", "DynamicPunch", "Zap Cannon", "Megahorn", "Focus Blast",
                "Gunk Shot", "Magma Storm", "Power Whip", "Seed Flare", "Head Smash"):  # fmt: skip
        assert names.move(key) == c.MOVE_IDS[key], key
    assert names.move("Trick") == c.MOVE_TRICK and names.move("Switcheroo") == c.MOVE_SWITCHEROO
    assert names.ability("Compoundeyes") == c.ABILITY_IDS["Compoundeyes"]
    for key, item in ITEM.items():
        assert names.item(key) == item, key


# --- 1.1-1.4 entry and the absorb ladders -------------------------------------------------------


def test_damaging_move_at_ally(w):
    # TagStrategy_Partner (:7339-7350): a damaging move that no absorb block takes falls into
    # ScoreMinus30; reads nothing (FlagMoveDamageScore's draws are no read).
    for move in ("Tackle", "Earthquake", "Razor Leaf"):
        ally(w, w.t, w.attack(w.s, move), 70, reads=0, case=move)


def test_fling_minus30(w):
    # Fling: vanilla power 1, effect FLING (not in the alt-power list): no damage comparison, so it is
    # scored by PartnerStatusMove, which has no case for it: -30 (:7455-7456). Trick / Switcheroo: End.
    ally(w, w.t, w.attack(w.s, "Fling"), 70, reads=0)
    ally(w, w.t, w.attack(w.s, "Trick"), 100, reads=0, case="Trick")
    ally(w, w.t, w.attack(w.s, "Switcheroo"), 100, reads=0, case="Switcheroo")
    ally(w, w.t, w.attack(w.s, "Growl"), 70, reads=0, case="any other status move")


def test_fire_flash_fire_ally(w):
    # TagStrategy_CheckPartnerFireAbsorption (:7352-7362): Flash Fire HAVE (true ability) and not yet
    # activated +3; activated, or another ability, -30. Will-O-Wisp goes there with Flash Fire (:7513-7516).
    for move in ("Flamethrower", "Will-O-Wisp"):
        s = w.attack(w.s, move)
        t1, s1 = _ability(w.t, s, ALLY, "Flash Fire")
        ally(w, t1, s1, 103, reads=0, case=f"{move} Flash Fire")
        ally(w, t1, _flag(s1, "flash_fire", ALLY), 70, reads=0, case=f"{move} activated")
        t2, s2 = _ability(w.t, s, ALLY, "Levitate")
        ally(w, t2, s2, 70, reads=0, case=f"{move} other ability")


def test_motor_drive_ally(w):
    # TagStrategy_CheckPartnerMotorDrive (:7380-7386): R@7383(160) jump: no change; else the ally at +6
    # Speed -30, else +3. Thunder Wave goes there (:7552-7554); a Ground ally is -30 first (:7546-7549).
    for move in ("Thunderbolt", "Thunder Wave"):
        t1, s1 = _ability(w.t, w.attack(w.s, move), ALLY, "Motor Drive")
        ally(w, t1, s1, {100: 160 / 256, 103: 96 / 256}, reads=1, case=move)
        up = _stage(s1, ALLY, Stat.SPE, 6)
        ally(w, t1, up, {100: 160 / 256, 70: 96 / 256}, reads=1, case=f"{move} +6")
        ally(w, t1, _stage(s1, ALLY, Stat.SPE, 5), {100: 160 / 256, 103: 96 / 256}, reads=1)
    t1, s1 = _ability(w.t, w.attack(w.s, "Thunder Wave"), ALLY, "Motor Drive")
    ally(w, t1, _types(s1, ALLY, GROUND, NORMAL), 70, reads=0, case="Ground type")
    ally(w, t1, _types(s1, ALLY, NORMAL, GROUND), 70, reads=0, case="Ground type 2")
    ally(w, w.t, w.attack(w.s, "Thunder Wave"), 70, reads=0, case="no absorbing ability")
    ally(w, w.t, w.attack(w.s, "Thunderbolt"), 70, reads=0, case="Electric, no absorbing ability")


@pytest.mark.parametrize(
    ("hp", "want", "reads"),
    [
        (100, 90, 0),  # ScoreMinus10
        (91, 100, 0),  # > 90: End
        (90, _plus3(64), 1),  # > 75
        (76, _plus3(64), 1),
        (75, _plus3(128), 1),  # > 50
        (51, _plus3(128), 1),
        (50, _plus3(192), 1),  # the rest
        (1, _plus3(192), 1),
    ],
)
def test_volt_absorb_ladder(w, hp, want, reads):
    # TagStrategy_CheckPartnerVoltAbsorb (:7388-7406): 100 % -10; > 90 % nothing; the three bands roll
    # R@7395 / 7399 / 7403 (64 / 128 / 192): the jump is the +3.
    t1, s1 = _ability(w.t, w.attack(w.s, "Thunderbolt"), ALLY, "Volt Absorb")
    ally(w, t1, _hp(s1, ALLY, hp), want, reads=reads, case=f"hp {hp}")
    t2, s2 = _ability(w.t, w.attack(w.s, "Thunder Wave"), ALLY, "Volt Absorb")
    ally(w, t2, _hp(s2, ALLY, hp), want, reads=reads, case=f"Thunder Wave hp {hp}")


@pytest.mark.parametrize(
    ("hp", "want", "reads"),
    [(100, 90, 0), (95, 100, 0), (80, _plus3(64), 1), (60, _plus3(128), 1), (40, _plus3(192), 1)],
)
@pytest.mark.parametrize("ability", ["Water Absorb", "Dry Skin"])
def test_water_absorb_ladder(w, ability, hp, want, reads):
    # TagStrategy_CheckPartnerWaterAbsorption (:7409-7442): Water Absorb or Dry Skin, the same ladder with
    # R@7430 / 7434 / 7438; any other ability -30.
    t1, s1 = _ability(w.t, w.attack(w.s, "Water Gun"), ALLY, ability)
    ally(w, t1, _hp(s1, ALLY, hp), want, reads=reads)
    ally(w, w.t, w.attack(w.s, "Water Gun"), 70, reads=0, case="no absorbing ability")


# --- 1.6 Skill Swap -------------------------------------------------------------------------------


def test_skill_swap_levitate(w):
    # TagStrategy_PartnerSkillSwap (:7458-7511). The ally's ability is the AI's knowledge (record, else
    # the species' ability: Bulbasaur has one, no coin).
    swap = w.attack(w.s, "Skill Swap")
    t1, s1 = _ability(w.t, swap, A, "Levitate")
    ally(w, t1, s1, 70, reads=0, case="Levitate attacker, ally not Electric: -30")
    ally(w, t1, _types(s1, ALLY, ELECTRIC, NORMAL), 71, reads=0, case="type 1 Electric: +1, -30")
    ally(w, t1, _types(s1, ALLY, ELECTRIC, ELECTRIC), 102, reads=0, case="mono-Electric: +2")
    ally(w, t1, _record(_types(s1, ALLY, ELECTRIC, ELECTRIC), ALLY, "Levitate"), 70, reads=0,
         case="ally recorded as Levitate")  # fmt: skip
    ally(w, t1, _record(s1, ALLY, "Truant"), 110, reads=0, case="Truant +10")
    ally(w, t1, _record(s1, ALLY, "Slow Start"), 110, reads=0, case="Slow Start +10")
    # a Compound Eyes / No Guard attacker: +3 with an inaccurate move known by the ally, else -30
    for name in ("Compoundeyes", "No Guard"):
        t2, s2 = _ability(w.t, swap, A, name)
        ally(w, t2, s2, 70, reads=0, case=f"{name}, no inaccurate move")
        for move in ("Fire Blast", "Thunder", "Blizzard", "Head Smash", "DynamicPunch"):
            ally(w, t2, w.moves(s2, ALLY, "Tackle", move), 103, reads=0, case=f"{name} {move}")
    # the ally's ability guessed: a species with two abilities, unrecorded: one coin per read
    t3 = _two_abilities(w.t, swap, ALLY, "Truant", "Overgrow")
    t3, s3 = _ability(t3, swap, A, "Levitate")
    ally(w, t3, s3, {110: 1 / 2, 70: 1 / 2}, reads=2, case="coin: Truant or Overgrow")
    t4 = _two_abilities(w.t, swap, ALLY, "Levitate", "Overgrow")
    t4, s4 = _ability(t4, swap, A, "Levitate")
    mono = _types(s4, ALLY, ELECTRIC, ELECTRIC)
    ally(w, t4, mono, {70: 1 / 2, 102: 1 / 2}, reads=2, case="second coin: Levitate or not")


# --- 1.7-1.9 ----------------------------------------------------------------------------------------


def test_will_o_wisp_guts(w):
    # TagStrategy_PartnerWillOWisp (:7513-7542): +5 for a Guts ally with no status, no Fire type, no Flame /
    # Toxic Orb and hp% >= 81; any other case -30.
    t1, s1 = _ability(w.t, w.attack(w.s, "Will-O-Wisp"), ALLY, "Guts")
    ally(w, t1, s1, 105, reads=0, case="all conditions")
    ally(w, w.t, w.attack(w.s, "Will-O-Wisp"), 70, case="no Guts")
    ally(w, t1, _status(s1, ALLY, Status.BURN), 70, case="statused")
    ally(w, t1, _status(s1, ALLY, Status.PARALYSIS), 70, case="paralysed")
    ally(w, t1, _types(s1, ALLY, FIRE, NORMAL), 70, case="Fire 1")
    ally(w, t1, _types(s1, ALLY, NORMAL, FIRE), 70, case="Fire 2")
    ally(w, t1, _item(s1, "Flame Orb"), 70, case="Flame Orb")
    ally(w, t1, _item(s1, "Toxic Orb"), 70, case="Toxic Orb")
    ally(w, t1, _hp(s1, ALLY, 80), 70, case="hp 80")
    ally(w, t1, _hp(s1, ALLY, 81), 105, case="hp 81")


def test_poison_heal_91_92(w):
    # TagStrategy_PartnerPoisonStatus (:7561-7580): +5 for a Poison Heal ally, `IfStatus DEFENDER` (the ally
    # again) clear, no Toxic Orb, hp% <= 91 (the comment says 81); else -30.
    for move in ("Toxic", "PoisonPowder"):
        t1, s1 = _ability(w.t, w.attack(w.s, move), ALLY, "Poison Heal")
        ally(w, t1, _hp(s1, ALLY, 91), 105, reads=0, case=f"{move} 91")
        ally(w, t1, _hp(s1, ALLY, 92), 70, reads=0, case=f"{move} 92")
        ally(w, t1, _hp(s1, ALLY, 30), 105, case=f"{move} 30")
        # each other condition at hp% 91, where it alone decides
        at91 = _hp(s1, ALLY, 91)
        ally(w, t1, _status(at91, ALLY, Status.PARALYSIS), 70, case=f"{move} statused")
        ally(w, t1, _item(at91, "Toxic Orb"), 70, case=f"{move} Toxic Orb")
        ally(w, w.t, _hp(w.attack(w.s, move), ALLY, 91), 70, case=f"{move} no Poison Heal")
        # the script's BUG: no Poison / Steel type test (a Poison-type ally scores the same)
        ally(w, t1, _types(at91, ALLY, BUG, NORMAL), 105, case=f"{move} Bug")
        ally(w, t1, _types(at91, ALLY, POISON, NORMAL), 105, case=f"{move} Poison")


# --- 1.10 Helping Hand ------------------------------------------------------------------------------

HH_LAW = {99: 64 / 256, 102: 192 / 256}  # R@7596(64): the jump is -1, else +2


def test_helping_hand_rank(w):
    # TagStrategy_PartnerUsingHelpingHand (:7582-7600): hp% 0 -30; hp% > 50 on to TryScorePlus2; else the
    # ally's speed rank (one call): rank 0 on to it, any other rank nothing.
    hh = w.attack(w.s, "Helping Hand")
    ally(w, w.t, _hp(hh, ALLY, 5, 1000), 70, reads=0, case="hp% 0")
    ally(w, w.t, hh, HH_LAW, reads=1, case="hp 100")
    ally(w, w.t, _hp(hh, ALLY, 51), HH_LAW, reads=1, case="hp 51")
    # speeds of battlers 0..3: the ally (3) fastest, then the attacker
    fast = _speeds(_hp(hh, ALLY, 40), (50, 60, 40, 100))
    ally(w, w.t, fast, HH_LAW, reads=1, case="hp 40 rank 0")
    ally(w, w.t, _hp(_speeds(fast, (50, 60, 40, 55)), ALLY, 50), 100, reads=0, case="rank 1")
    ally(w, w.t, _speeds(_hp(hh, ALLY, 40), (50, 60, 100, 55)), 100, reads=0, case="rank 2")
    # a tie for rank 0 with the attacker: the rank is 0 with probability 1/2, then the roll
    tied = _speeds(_hp(hh, ALLY, 40), (50, 100, 40, 100))
    ally(w, w.t, tied, {100: 1 / 2, 99: 1 / 8, 102: 3 / 8}, reads=2, case="tie")


# --- 1.11-1.14 ----------------------------------------------------------------------------------------


def test_swagger_stage(w):
    # TagStrategy_PartnerSwagger (:7602-7619): a Persim or Lum Berry holder (the ally's true item) at
    # Attack below +2 (decomp stage <= 7) +3, at +2 or more nothing; no such berry -30.
    swag = w.attack(w.s, "Swagger")
    for berry in ("Persim Berry", "Lum Berry"):
        s = _item(swag, berry)
        ally(w, w.t, s, 103, reads=0, case=berry)
        ally(w, w.t, _stage(s, ALLY, Stat.ATK, 1), 103, case=f"{berry} +1")
        ally(w, w.t, _stage(s, ALLY, Stat.ATK, 2), 100, case=f"{berry} +2")
        ally(w, w.t, _stage(s, ALLY, Stat.ATK, 6), 100, case=f"{berry} +6")
    ally(w, w.t, swag, 70, case="no item")
    ally(w, w.t, _item(swag, "Toxic Orb"), 70, case="another item")


def test_gastro_acid_truant(w):
    # TagStrategy_PartnerGastroAcid (:7624-7644): suppressed -30; Truant or Slow Start +5; else nothing.
    gastro = w.attack(w.s, "Gastro Acid")
    ally(w, w.t, gastro, 100, reads=0, case="neither")
    ally(w, w.t, _flag(gastro, "ability_suppressed", ALLY), 70, case="suppressed")
    for name in ("Truant", "Slow Start"):
        t1, s1 = _ability(w.t, gastro, ALLY, name)
        ally(w, t1, s1, 105, reads=0, case=name)
        ally(w, t1, _flag(s1, "ability_suppressed", ALLY), 70, case=f"{name} suppressed")


def test_acupressure_laws(w):
    # TagStrategy_PartnerAcupressure (:7646-7690). Not Simple: any stat at +6 -30. Simple: a stat at +3 or
    # more -10. Then hp% < 51 -1; hp% > 90 R@7682(80): jump nothing, else +2 (176/256); else R@7679(128)
    # jump nothing, then R@7682: +2 with 128/256 * 176/256 = 88/256.
    acu = w.attack(w.s, "Acupressure")
    ally(w, w.t, acu, {100: 80 / 256, 102: 176 / 256}, reads=1, case="hp 100")
    ally(w, w.t, _hp(acu, ALLY, 91), {100: 80 / 256, 102: 176 / 256}, reads=1, case="hp 91")
    ally(w, w.t, _hp(acu, ALLY, 90), {100: 168 / 256, 102: 88 / 256}, reads=2, case="hp 90")
    ally(w, w.t, _hp(acu, ALLY, 51), {100: 168 / 256, 102: 88 / 256}, reads=2, case="hp 51")
    ally(w, w.t, _hp(acu, ALLY, 50), 99, reads=0, case="hp 50")
    for stat in (Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD, Stat.EVA, Stat.ACC):
        ally(w, w.t, _stage(acu, ALLY, stat, 6), 70, reads=0, case=f"{stat.name} +6")
        ally(w, w.t, _stage(acu, ALLY, stat, 5), {100: 80 / 256, 102: 176 / 256}, case="+5")
    t1, s1 = _ability(w.t, acu, ALLY, "Simple")
    ally(w, t1, _stage(s1, ALLY, Stat.ATK, 3), 90, reads=0, case="Simple +3")
    ally(w, t1, _stage(s1, ALLY, Stat.ACC, 6), 90, reads=0, case="Simple +6: -10, not -30")
    ally(w, t1, _stage(s1, ALLY, Stat.SPE, 2), {100: 80 / 256, 102: 176 / 256}, case="Simple +2")


# --- drivers -------------------------------------------------------------------------------------------


def test_check_hp_ally_alone(w):
    # CheckHP_Main (:7696-7697) sends the ally target to the Tag Strategy ally part: bit 8 alone scores it.
    ally(w, w.t, w.attack(w.s, "Tackle"), 70, reads=0, flags=(CHECK_HP,), case="bit 8 alone")


@pytest.mark.slow  # one compile of the Tag Strategy and CheckHP scripts in a row: over 60 s
def test_check_hp_runs_twice(w):
    # CheckHP_Main (:7696-7697) sends the ally target to the same part: bit 7 then bit 8, both score
    # changes and both draw sets (A-9 0.4).
    both = (TAG_STRATEGY, CHECK_HP)
    s = w.attack(w.s, "Tackle")
    ally(w, w.t, s, 40, reads=0, flags=both, case="-30 twice")
    t1, s1 = _ability(w.t, w.attack(w.s, "Thunderbolt"), ALLY, "Volt Absorb")
    p = 64 / 256
    ally(w, t1, _hp(s1, ALLY, 80), {106: p * p, 103: 2 * p * (1 - p), 100: (1 - p) ** 2},
         reads=2, flags=both, case="two independent rolls")  # fmt: skip
    ally(w, t1, _hp(s1, ALLY, 100), 80, reads=0, flags=both, case="-10 twice")
