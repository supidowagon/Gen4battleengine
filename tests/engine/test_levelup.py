"""K3d-exp T5: the exp payout and level-up in both engines (design §1.1-1.4, §3.1).

Every expected number below is computed by hand from the decomp formulas and the vanilla
tables' data (``species_exp_yield``, ``exp_table``, ``species_base_stats``,
``species_ev_yield``), each step written out next to it. Decomp (external/pokeplatinum):

* ``BtlCmd_CalcExpGain`` (src/battle/battle_script.c:2187-2245): ``E = yield * level / 7``;
  ``g = max(1, E / n)`` (or ``(E / 2) / n`` and ``h = max(1, (E / 2) / s)`` with an Exp. Share
  holder), ``n`` / ``s`` counting party Pokemon with HP.
* ``BattleScript_GetExpTask`` (battle_script.c:9700-9836): ``t = g (+ h)``, ``* 150 / 100``
  Lucky Egg, ``* 150 / 100`` trainer, ``* 150|170 / 100`` not OT; EVs (:9776-9779) before the
  level check, which runs only when ``t != 0`` (:9782-9791).
* ``Pokemon_ShouldLevelUp`` (src/pokemon.c:3501-3526), ``Pokemon_UpdateFriendship``
  (:2635-2689), ``Pokemon_CalcStats`` (:564-648: ``hp += newMax - oldMax``).
* ``BattleScript_CalcEffortValues`` (battle_script.c:10214-10306): loop order HP, Atk, Def,
  Spe, SpA, SpD; stop at a sum of 510; clip the sum to 510, then the stat to 255.

Payout tests run the JAX ``pay_exp`` and the reference ``ref_grant_exp`` (the decomp's
per-level loop) and assert every field equal; step tests run ``step`` against ``ref_step``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine import levelup, stats
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.levelup import pay_exp
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import EXP_REFUSED_WINDOW_C, replace, step
from nuzline.reference import engine as ref_engine
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_grant_exp, ref_replace
from tests.engine.helpers import A, assert_matches_reference

_pay = jax.jit(pay_exp)
_SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe
MISS = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))


def _mon(names, species: str, level: int, *, moves=("Tackle",), **kw) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    base = {
        "species": names.species(species),
        "level": level,
        "nature": 0,  # Hardy: neutral
        "ivs": (0,) * 6,
        "evs": (0,) * 6,
        "moves": tuple(ids + [0] * (4 - len(ids))),
        "item": 0,
        "ability_slot": 0,
        "gender": 0,
        "friendship": 70,
    }
    base.update(kw)
    return PokemonSpec(**base)


def _base(tables, names, species: str, level: int) -> int:
    rate = int(tables.species_exp_rate[names.species(species)])
    return int(tables.exp_table[rate, level])


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _ko(tables, s, slot: int = 0, *, pending: bool = False, selected: int = 0):
    """The payout for the opponent's party ``slot`` fainting, in both engines (asserted
    equal); the JAX state."""
    after = s.replace_hp(1, slot, 0)
    out = _pay(tables, s, after, jnp.asarray(pending), jnp.asarray(selected, jnp.int32))
    ref = ref_grant_exp(to_numpy(tables), to_ref(after), slot, pending, selected)
    _same(out, from_ref(ref))
    return out


def _with(s, **fields):
    """``s`` with player-side per-slot values set: ``exp={slot: v}``, ``hp=...`` and so on."""
    out = {}
    for name, values in fields.items():
        arr = getattr(s, name)
        for slot, v in values.items():
            arr = arr.at[slot].set(v) if arr.ndim == 1 else arr.at[0, slot].set(v)
        out[name] = arr
    return s._replace(**out)


def _mask(s, *slots):
    return s._replace(exp_mask=jnp.asarray(sum(1 << i for i in slots), jnp.int8))


# --- E, the split and the multipliers ---------------------------------------------------------


def _prinplup_state(tables, names, *, opp=None, start=Weather.NONE):
    # Prinplup (base HP 64, medium slow) with HP IV 3: max HP (128 + 3) * 16 / 100 + 26 = 46
    # at L16, (128 + 3) * 17 / 100 + 27 = 49 at L17; exp one short of L17 (3120 - 1).
    prinplup = _mon(
        names,
        "Prinplup",
        16,
        moves=("Bubble", "Roar", "Tackle", "Water Gun"),
        ivs=(3, 3, 3, 3, 3, 3),
        current_hp=44,
        exp=_base(tables, names, "Prinplup", 17) - 1,
    )
    opp = opp or [_mon(names, "Bonsly", 15), _mon(names, "Geodude", 12)]
    return make_state(tables, [prinplup, _mon(names, "Machop", 20)], opp, start_condition=start)


def test_bonsly_pays_prinplup_217_and_it_reaches_l17(tables, names):
    # E = 68 * 15 / 7 = 145 (1020 / 7); n = 1, no Exp. Share: g = 145;
    # t = 145 * 150 / 100 = 217 (trainer). exp = 3119 + 217 = 3336: L17 (3120 <= 3336 < 3798).
    s = _prinplup_state(tables, names)
    assert int(s.max_hp[0, 0]) == 46 and int(s.exp[0]) == 3119
    out = _ko(tables, s)
    assert int(out.exp[0]) == 3119 + 217
    assert int(out.level[0, 0]) == 17
    # CalcStats at L17 with Bonsly's Def EV (1): HP 49 and hp 44 + 49 - 46 = 47; Atk
    # (132+3)*17/100+5 = 27, Def (136+3+0)*17/100+5 = 28, SpA (162+3)*17/100+5 = 33, SpD
    # (152+3)*17/100+5 = 31, Spe (100+3)*17/100+5 = 22.
    assert int(out.max_hp[0, 0]) == 49 and int(out.hp[0, 0]) == 47
    assert np.asarray(out.stats[0, 0]).tolist() == [27, 28, 33, 31, 22]
    assert np.asarray(out.evs[0, 0]).tolist() == [0, 0, 1, 0, 0, 0]
    assert int(out.friendship[0, 0]) == 75  # 70 < 100: +5
    assert int(out.exp_mask) == 0  # SEQ_GET_EXP_CHECK_DONE clears every visited bit
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (17, 0)
    # the bench Machop is no participant: untouched.
    for name in ("exp", "learn_level"):
        assert int(getattr(out, name)[1]) == int(getattr(s, name)[1])
    assert int(out.level[0, 1]) == 20


def test_lileep_is_split_91_and_pays_136_each(tables, names):
    # E = 99 * 13 / 7 = 183 (1287 / 7); n = 2: g = 183 / 2 = 91; t = 91 * 150 / 100 = 136.
    # Multiplying before the split would give 183 * 150 / 100 / 2 = 137.
    party = [_mon(names, "Machop", 20), _mon(names, "Machop", 20), _mon(names, "Machop", 20)]
    s = _mask(make_state(tables, party, [_mon(names, "Lileep", 13)]), 0, 1)
    out = _ko(tables, s)
    gained = (np.asarray(out.exp) - np.asarray(s.exp)).tolist()
    assert gained[:3] == [136, 136, 0]


def test_exp_share_takes_half_split_among_its_holders(tables, names):
    # Bonsly L15: E = 145. Slot 0 in the mask, slot 1 in the mask with an Exp. Share, slot 2
    # on the bench with an Exp. Share, slot 3 fainted with an Exp. Share (counted nowhere).
    # n = 2, s = 2: g = (145 / 2) / 2 = 36, h = 72 / 2 = 36.
    # slot 0: 36 * 1.5 = 54; slot 1: (36 + 36) * 1.5 = 108; slot 2: 36 * 1.5 = 54.
    share = names.item("Exp. Share")
    party = [
        _mon(names, "Machop", 20),
        _mon(names, "Machop", 20, item=share),
        _mon(names, "Machop", 20, item=share),
        _mon(names, "Machop", 20, item=share, current_hp=0),
    ]
    s = _mask(make_state(tables, party, [_mon(names, "Bonsly", 15)]), 0, 1)
    out = _ko(tables, s)
    gained = (np.asarray(out.exp) - np.asarray(s.exp)).tolist()
    assert gained[:4] == [54, 108, 54, 0]


def test_lucky_egg_then_trainer_truncate_in_turn(tables, names):
    # Magikarp L1: E = 20 / 7 = 2, n = 2: g = 1. Lucky Egg 1 * 150 / 100 = 1, trainer 1:
    # t = 1 (a single * 225 / 100 would give 2). Magikarp L2: E = 40 / 7 = 5, one participant:
    # Lucky Egg 5 -> 7, trainer 7 -> 10; without the egg 5 -> 7.
    egg = names.item("Lucky Egg")
    party = [_mon(names, "Machop", 20, item=egg), _mon(names, "Machop", 20)]
    s = _mask(make_state(tables, party, [_mon(names, "Magikarp", 1)]), 0, 1)
    out = _ko(tables, s)
    assert (np.asarray(out.exp) - np.asarray(s.exp)).tolist()[:2] == [1, 1]
    s = _mask(make_state(tables, party, [_mon(names, "Magikarp", 2)]), 0)
    assert (int(_ko(tables, s).exp[0]) - int(s.exp[0])) == 10
    s = _mask(make_state(tables, party, [_mon(names, "Magikarp", 2)]), 1)
    assert (int(_ko(tables, s).exp[1]) - int(s.exp[1])) == 7


@pytest.mark.parametrize("boost,gain", [(100, 217), (150, 325), (170, 368)])
def test_a_traded_pokemon_gains_the_boost(tables, names, boost, gain):
    # g = 145 (Bonsly L15), trainer 217; not OT: 217 * 150 / 100 = 325 (same language),
    # 217 * 170 / 100 = 368 (another language).
    party = [_mon(names, "Machop", 30, exp_boost=boost)]
    s = make_state(tables, party, [_mon(names, "Bonsly", 15)])
    assert int(_ko(tables, s).exp[0]) - int(s.exp[0]) == gain


def test_a_fainted_participant_counts_nowhere(tables, names):
    # Mask {0, 1}, slot 1 fainted: n = 1, so slot 0 gets the whole 145 -> 217, slot 1 nothing.
    party = [_mon(names, "Machop", 20), _mon(names, "Machop", 20, current_hp=0)]
    s = _mask(make_state(tables, party, [_mon(names, "Bonsly", 15)]), 0, 1)
    out = _ko(tables, s)
    assert (np.asarray(out.exp) - np.asarray(s.exp)).tolist()[:2] == [217, 0]
    assert int(out.exp_mask) == 0


def test_no_participant_left_pays_no_g(tables, names):
    # Ruling D4: the only mask slot (1) has fainted, so n = 0 (safe divide) and nobody is paid
    # g; slot 2 holds an Exp. Share: s = 1, h = (145 / 2) / 1 = 72 -> 108.
    share = names.item("Exp. Share")
    party = [
        _mon(names, "Machop", 20),
        _mon(names, "Machop", 20, current_hp=0),
        _mon(names, "Machop", 20, item=share),
    ]
    s = _mask(make_state(tables, party, [_mon(names, "Bonsly", 15)]), 1)
    out = _ko(tables, s)
    assert (np.asarray(out.exp) - np.asarray(s.exp)).tolist()[:3] == [0, 0, 108]


# --- EVs ------------------------------------------------------------------------------------


def test_evs_stop_at_510_in_decomp_order(tables, names):
    # Prinplup yields SpA 2; a Power Anklet adds its param 4 to Spe (HOLD_EFFECT_LVLUP_SPEED_
    # EV_UP). EVs sum 506: the decomp loop visits Spe before SpA, so Spe takes 4 (sum 510) and
    # SpA gets nothing (an ids.Stat-order loop would give SpA 2, Spe 2).
    anklet = names.item("Power Anklet")
    party = [_mon(names, "Machop", 30, item=anklet, evs=(255, 251, 0, 0, 0, 0))]
    out = _ko(tables, make_state(tables, party, [_mon(names, "Prinplup", 16)]))
    assert np.asarray(out.evs[0, 0]).tolist() == [255, 251, 0, 0, 0, 4]


def test_evs_clip_the_sum_then_the_stat(tables, names):
    # Pikachu yields Spe 2, doubled by Pokerus: 4. Sum 508: clipped to 2 (510 - 508).
    party = [_mon(names, "Machop", 30, pokerus=1, evs=(255, 253, 0, 0, 0, 0))]
    out = _ko(tables, make_state(tables, party, [_mon(names, "Pikachu", 16)]))
    assert np.asarray(out.evs[0, 0]).tolist() == [255, 253, 0, 0, 0, 2]
    # Spe 253, sum 253: 4 clipped to 255 - 253 = 2.
    party = [_mon(names, "Machop", 30, pokerus=1, evs=(0, 0, 0, 0, 0, 253))]
    out = _ko(tables, make_state(tables, party, [_mon(names, "Pikachu", 16)]))
    assert np.asarray(out.evs[0, 0]).tolist() == [0, 0, 0, 0, 0, 255]


def test_macho_brace_and_pokerus_both_double(tables, names):
    # Bonsly Def 1, Pokerus * 2, Macho Brace * 2: 4.
    brace = names.item("Macho Brace")
    party = [_mon(names, "Machop", 30, item=brace, pokerus=1)]
    out = _ko(tables, make_state(tables, party, [_mon(names, "Bonsly", 15)]))
    assert np.asarray(out.evs[0, 0]).tolist() == [0, 0, 4, 0, 0, 0]


def test_ev_yields_are_read_per_form(tables, names):
    # Wormadam Sandy (form 1, row 499) yields Def 2; Plant (form 0) yields SpD 2.
    party = [_mon(names, "Machop", 30)]
    sandy = _mon(names, "Wormadam", 20, form=1)
    out = _ko(tables, make_state(tables, party, [sandy]))
    assert np.asarray(out.evs[0, 0]).tolist() == [0, 0, 2, 0, 0, 0]


def test_kaizo_pays_no_evs(kaizo_tables, kaizo_names):
    # Every Kaizo EV yield is 0 (T2) and Kaizo's Power items are Macho Braces with param 0,
    # so Pokerus and the item double nothing: a Kaizo payout never changes EVs.
    bracer = kaizo_names.item("Power Bracer")
    party = [_mon(kaizo_names, "Machop", 30, item=bracer, pokerus=1, evs=(1, 2, 3, 4, 5, 6))]
    s = make_state(kaizo_tables, party, [_mon(kaizo_names, "Bonsly", 15)])
    out = _ko(kaizo_tables, s)
    assert np.asarray(out.evs[0, 0]).tolist() == [1, 2, 3, 4, 5, 6]
    assert int(out.exp[0]) > int(s.exp[0])


def test_evs_are_added_before_the_level_up(tables, names):
    # Machop L99 (medium slow) one short of L100, Def EV 3. Bonsly: t = 217 > 1, so exp is
    # clamped to the level-100 base 1059860 and the level becomes 100. Bonsly's Def EV makes
    # 4, so CalcStats at L100 sees EV / 4 = 1: Def (100 + 0 + 1) * 100 / 100 + 5 = 106
    # (105 if the EVs came after the recompute).
    top = _base(tables, names, "Machop", 100)
    party = [_mon(names, "Machop", 99, exp=top - 1, evs=(0, 0, 3, 0, 0, 0))]
    out = _ko(tables, make_state(tables, party, [_mon(names, "Bonsly", 15)]))
    assert int(out.exp[0]) == top == 1059860
    assert int(out.level[0, 0]) == 100
    assert int(out.stats[0, 0, Stat.DEF - 1]) == 106


# --- levels -------------------------------------------------------------------------------


def test_a_level_100_pokemon_gains_nothing(tables, names):
    # script.c:9740: no exp, no EVs at level 100; it still counts in n (Lileep's 183 / 2).
    top = _base(tables, names, "Machop", 100)
    party = [_mon(names, "Machop", 100, exp=top), _mon(names, "Machop", 20)]
    s = _mask(make_state(tables, party, [_mon(names, "Lileep", 13)]), 0, 1)
    out = _ko(tables, s)
    assert (np.asarray(out.exp) - np.asarray(s.exp)).tolist()[:2] == [0, 136]
    assert np.asarray(out.evs[0, 0]).tolist() == [0] * 6


def test_an_unpaid_slot_never_levels(tables, names):
    # Slot 1 (no participant, no Exp. Share) holds exp past its next base: ShouldLevelUp runs
    # only for a paid slot (bs.c:9782-9791), so it keeps its level and exp.
    party = [_mon(names, "Machop", 20), _mon(names, "Machop", 20)]
    s = make_state(tables, party, [_mon(names, "Bonsly", 15)])
    s = _with(s, exp={1: _base(tables, names, "Machop", 22) + 5})
    out = _ko(tables, s)
    assert int(out.level[0, 1]) == 20 and int(out.exp[1]) == int(s.exp[1])
    assert int(out.learn_level[1]) == 0


def test_a_multi_level_gain_telescopes_hp_and_friendship(tables, names):
    # Starly L5 (medium slow, base HP 40) beats Chansey L20: E = 255 * 20 / 7 = 728,
    # t = 728 * 1.5 = 1092; exp 135 + 1092 = 1227: L12 (973 <= 1227 < 1261), seven levels.
    # Max HP: L5 80 * 5 / 100 + 15 = 19; at L12 with Chansey's HP EV 2 (EV / 4 = 0):
    # 80 * 12 / 100 + 22 = 31; hp 10 + 31 - 19 = 22.
    # Friendship 90, Luxury Ball and Soothe Bell: tier 0 (5 + 1) * 1.5 = 9, tier 1
    # (3 + 1) * 1.5 = 6: 90 -> 99 -> 108 (tier 0 twice), then 114, 120, 126, 132, 138.
    bell = names.item("Soothe Bell")
    starly = _mon(names, "Starly", 5, item=bell, friendship=90, friend_bonus=1, current_hp=10)
    s = make_state(tables, [starly], [_mon(names, "Chansey", 20)])
    assert int(s.exp[0]) == 135 and int(s.max_hp[0, 0]) == 19
    out = _ko(tables, s)
    assert int(out.exp[0]) == 1227 and int(out.level[0, 0]) == 12
    assert int(out.max_hp[0, 0]) == 31 and int(out.hp[0, 0]) == 22
    assert int(out.friendship[0, 0]) == 138
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (6, 0)


def test_friendship_is_clamped_at_255(tables, names):
    # Starly L5 (exp 135) beats Bonsly: t = 217, exp 352: L8 (medium slow L8 = 314, L9 = 419),
    # three levels. Friendship 250 (tier 2, +2 per level): 252, 254, then 256 -> 255.
    starly = _mon(names, "Starly", 5, friendship=250)
    out = _ko(tables, make_state(tables, [starly], [_mon(names, "Bonsly", 15)]))
    assert int(out.exp[0]) == 352 and int(out.level[0, 0]) == 8
    assert int(out.friendship[0, 0]) == 255


def test_a_pending_learn_marker_is_kept(tables, names):
    # An earlier level's scan is still pending (learn_level 16, cursor 3): it stays, since
    # the resolve scans every level up to the current one (design §3.1 step 6).
    s = _prinplup_state(tables, names)
    s = s._replace(
        learn_level=s.learn_level.at[0].set(16), learn_cursor=s.learn_cursor.at[0].set(3)
    )
    out = _ko(tables, s)
    assert int(out.level[0, 0]) == 17
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (16, 3)


def test_shedinja_stays_at_1_hp(tables, names):
    # Pokemon_CalcStats: Shedinja's max HP and current HP are 1 (src/pokemon.c:597-601, 636).
    shedinja = _mon(names, "Shedinja", 20, exp=_base(tables, names, "Shedinja", 21) - 1)
    out = _ko(tables, make_state(tables, [shedinja], [_mon(names, "Bonsly", 15)]))
    assert int(out.level[0, 0]) == 21
    assert int(out.max_hp[0, 0]) == 1 and int(out.hp[0, 0]) == 1


def test_bench_rhyhorn_levels_23_55_to_26_58(tables, names):
    # Rhyhorn (base HP 80, slow) on the bench, a participant, HP IV 25: max HP
    # (160 + 25) * 16 / 100 + 26 = 55 at L16, 185 * 17 / 100 + 27 = 58 at L17. n = 2 with
    # the active: g = 145 / 2 = 72, t = 108; exp 6140 + 108 >= 6141: L17, hp 23 + 3 = 26.
    rhyhorn = _mon(
        names,
        "Rhyhorn",
        16,
        ability_slot=1,
        ivs=(25, 0, 0, 0, 0, 0),
        current_hp=23,
        exp=_base(tables, names, "Rhyhorn", 17) - 1,
    )
    party = [_mon(names, "Machop", 20), rhyhorn]
    s = _mask(make_state(tables, party, [_mon(names, "Bonsly", 15)]), 0, 1)
    assert int(s.max_hp[0, 1]) == 55
    out = _ko(tables, s)
    assert int(out.level[0, 1]) == 17
    assert (int(out.hp[0, 1]), int(out.max_hp[0, 1])) == (26, 58)


def test_growth_rate_and_yield_are_read_by_species_not_form(tables, names):
    # SpeciesData_GetSpeciesValue reads the species (src/pokemon.c:3506; bs.c:2216). The real
    # form rows repeat their species' rate and yield, so give Wormadam Sandy's row (499) a
    # different rate and yield and the payout must not see them.
    sandy_row = int(tables.species_form_row[names.species("Wormadam"), 1])
    t = tables._replace(
        species_exp_rate=tables.species_exp_rate.at[sandy_row].set(5),
        species_exp_yield=tables.species_exp_yield.at[sandy_row].set(1),
    )
    # Receiver: Wormadam Sandy L20 (rate 0 by species) one short of L21 (9261).
    me = _mon(names, "Wormadam", 20, form=1, exp=_base(tables, names, "Wormadam", 21) - 1)
    foe = _mon(names, "Wormadam", 20, form=1)
    out = _ko(t, make_state(t, [me], [foe]))
    # E = 159 * 20 / 7 = 454 (by species), t = 681; the form's yield 1 would give E = 2, t = 3.
    assert int(out.exp[0]) == _base(tables, names, "Wormadam", 21) - 1 + 681
    assert int(out.level[0, 0]) == 21  # rate 5 (slow) would need 9261 -> 11576 at L21


# --- through step: the hook points ---------------------------------------------------------


def test_turn_1_sandstorm_chips_from_the_new_max(tables, names):
    # Prinplup KOs Bonsly (1 HP) with Bubble at MoveEnd: L17, 47/49. The sandstorm chip is
    # maxHP / 16 of the battler after the reload: 49 / 16 = 3, so 44/49 (the old max would
    # give 46 / 16 = 2). Bonsly is fainted and Rock-type: no chip.
    s = _prinplup_state(tables, names, start=Weather.SANDSTORM)
    s = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.level[0, 0]) == 17
    assert (int(out.hp[0, 0]), int(out.max_hp[0, 0])) == (44, 49)
    assert int(out.over) == 0


def test_the_end_of_turn_payout_comes_after_the_fainted_battlers_checks(tables, names):
    # CheckMonConditions pays at its loop head (bcp.c:1295-1299), before the next battler's
    # checks. Geodude (1 HP, burned) and Prinplup (poisoned, 44/46, one short of L17) both
    # miss. Geodude first in the end-of-turn order: it faints to its burn, the payout levels
    # Prinplup (44 + 3 = 47/49), then the poison chip is 49 / 8 = 6 (bcp.c:1393-1402):
    # 41/49. Prinplup first: the chip is 46 / 8 = 5 (39/46), then the payout: 39 + 3 = 42/49.
    # Paying once after both battlers would give 42/49 in the first case too.
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = _prinplup_state(tables, names, opp=opp).replace_hp(1, 0, 1)
    status = s.status.at[1, 0].set(int(Status.BURN)).at[0, 0].set(int(Status.POISON))
    s = s._replace(status=status)
    fast = int(s.stats[0, 0, _SPE])
    for opp_speed, hp in ((fast + 5, 41), (1, 42)):
        state = s._replace(stats=s.stats.at[1, 0, _SPE].set(opp_speed))
        out = assert_matches_reference(tables, state, A("Tackle", "Tackle", state, names), MISS)
        assert int(out.hp[1, 0]) == 0 and int(out.level[0, 0]) == 17
        assert (int(out.hp[0, 0]), int(out.max_hp[0, 0])) == (hp, 49)


def test_the_final_ko_pays(tables, names):
    # The payout precedes CheckBattleOver (bcp.c:3965-3969): the last opponent's faint pays.
    s = _prinplup_state(tables, names, opp=[_mon(names, "Bonsly", 15)])
    s = s.replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.level[0, 0]) == 17 and int(out.exp[0]) == 3119 + 217


def test_roar_into_stealth_rock_pays_from_the_newcomer(tables, names):
    # Roar (priority -6) drags Bonsly (1 HP) in; Stealth Rock faints it within Roar's action.
    # The opponent's send-out reset the mask to the player's active, so slot 0 alone is paid
    # Bonsly's 145 -> 217, though the battler that was active when the action began (Geodude)
    # is alive. The bench participant (slot 1) gets nothing.
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15, current_hp=1)]
    s = _mask(_prinplup_state(tables, names, opp=opp), 0, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    out = assert_matches_reference(tables, s, A("Roar", "Tackle", s, names), empty_outcome())
    assert int(out.active[1]) == 1 and int(out.hp[1, 1]) == 0
    assert int(out.exp[0]) == 3119 + 217 and int(out.level[0, 0]) == 17
    assert int(out.exp[1]) == int(s.exp[1])


def _selfdestruct_state(tables, names):
    """Geodude (faster) Selfdestructs, which faints it on a miss too; the player chose a
    move. Prinplup is one short of L17."""
    opp = [_mon(names, "Geodude", 12, moves=("Selfdestruct",)), _mon(names, "Bonsly", 15)]
    s = _prinplup_state(tables, names, opp=opp)
    return s._replace(stats=s.stats.at[1, 0, _SPE].set(int(s.stats[0, 0, _SPE]) + 1))


def test_learn_pin_marks_the_pending_move_of_a_leveled_active(tables, names):
    s = _selfdestruct_state(tables, names)
    # Geodude L12: E = 73 * 12 / 7 = 125, t = 187; exp 3119 + 187: L17 before Water Gun runs.
    out = assert_matches_reference(tables, s, A("Water Gun", "Selfdestruct", s, names), MISS)
    assert int(out.hp[1, 0]) == 0 and int(out.level[0, 0]) == 17
    assert int(out.learn_pin) == 1 + names.move_slot(s, 0, "Water Gun") == 4


def test_learn_pin_needs_a_pending_move(tables, names):
    s = _selfdestruct_state(tables, names)
    # The player switches to Machop (switches go first): no move is pending when Geodude
    # faints. Mask {0, 1}: g = 125 / 2 = 62, t = 93; Prinplup 3119 + 93 levels on the bench.
    out = assert_matches_reference(tables, s, A(MOVE_ACTIONS + 1, "Selfdestruct", s, names), MISS)
    assert int(out.learn_pin) == 0 and int(out.level[0, 0]) == 17  # Prinplup was paid
    # The player moved first (it is faster) and KO'd the foe: its move has run.
    s = _prinplup_state(tables, names).replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    assert int(out.level[0, 0]) == 17 and int(out.learn_pin) == 0


def test_learn_pin_is_cleared_every_turn(tables, names):
    s = _prinplup_state(tables, names)
    s = s._replace(learn_pin=jnp.asarray(3, jnp.int8))
    out = assert_matches_reference(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert int(out.learn_pin) == 0


def _replace_both(tables, s, side, slot):
    out = replace(tables, s, side, slot)
    _same(out, from_ref(ref_replace(to_numpy(tables), to_ref(s), side, slot)))
    return out


@pytest.mark.fullset
def test_a_hazard_ko_inside_the_replacement_loop_refuses_learn_work(tables, names):
    # The opponent sends Bonsly (1 HP) into Stealth Rock: TurnEnd's payout (bcp.c:1846) levels
    # Prinplup, and the next replacement pick would read the moves before the learn
    # resolve: window (c), exp_refused = 2.
    opp = [_mon(names, "Geodude", 12, current_hp=0), _mon(names, "Bonsly", 15, current_hp=1)]
    opp.append(_mon(names, "Geodude", 12))
    s = _prinplup_state(tables, names, opp=opp)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    out = _replace_both(tables, s, 1, 1)
    assert int(out.hp[1, 1]) == 0 and int(out.level[0, 0]) == 17
    assert int(out.exp_refused) == EXP_REFUSED_WINDOW_C == 2
    # No level-up, no learn work: no refusal.
    low = _with(s, exp={0: _base(tables, names, "Prinplup", 16)})
    out = _replace_both(tables, low, 1, 1)
    assert int(out.level[0, 0]) == 16 and int(out.exp[0]) > int(low.exp[0])
    assert int(out.exp_refused) == 0
    # The battle is over: nothing is left to pick, no refusal (the learn stays, ruling D8).
    last = s.replace_hp(1, 2, 0)
    out = _replace_both(tables, last, 1, 1)
    assert int(out.over) == 1 and int(out.level[0, 0]) == 17
    assert int(out.exp_refused) == 0


# --- the closed form, the constants, the program size -------------------------------------------


def test_calc_stats_jnp_matches_calc_stats(tables, names):
    rng = np.random.default_rng(5)
    base = np.asarray(tables.species_base_stats)
    for _ in range(50):
        species = int(rng.integers(1, 494))
        level = int(rng.integers(1, 101))
        ivs, evs = rng.integers(0, 32, 6), rng.integers(0, 256, 6)
        mods = np.asarray(tables.nature_mods[int(rng.integers(0, 25))])
        want = stats.calc_stats(base[species], ivs, evs, level, mods, species=species)
        got = stats.calc_stats_jnp(base[species], ivs, evs, level, mods, species)
        assert np.asarray(got).tolist() == want.tolist()
    shedinja = stats.SPECIES_SHEDINJA
    got = stats.calc_stats_jnp(base[shedinja], [31] * 6, [0] * 6, 50, [0] * 6, shedinja)
    assert int(got[Stat.HP]) == 1


@pytest.mark.parametrize("game", ["tables", "kaizo_tables"])
def test_the_nature_product_fits_u16(game, request):
    # Pokemon_GetNatureStatValue stores stat * 110 in a u16 (src/pokemon.c:2601-2604): a stat
    # up to 595 fits (595 * 110 = 65450). The largest non-HP stat is
    # (2B + 31 + 252 / 4) * 100 / 100 + 5, which stays within 595 while B <= 248.
    t = request.getfixturevalue(game)
    assert int(np.asarray(t.species_base_stats)[:, 1:].max()) <= 248


def test_the_reference_constants_match_the_engine():
    assert ref_engine.SPECIES_SHEDINJA == stats.SPECIES_SHEDINJA
    for name in (
        "HOLD_EFFECT_EVS_UP_SPEED_DOWN",
        "HOLD_EFFECT_EXP_SHARE",
        "HOLD_EFFECT_FRIENDSHIP_UP",
        "HOLD_EFFECT_EXP_UP",
    ):
        assert getattr(ref_engine, name) == getattr(levelup, name), name
    for stat, effect in ref_engine.POWER_ITEM_HOLD_EFFECTS.items():
        assert levelup.POWER_ITEM_HOLD_EFFECTS[stat] == effect
    assert levelup.DECOMP_STAT_ORDER == ref_engine.DECOMP_STATS


def test_hold_effect_ids_match_the_items(tables, names):
    # generated/item_hold_effects.txt, line N is id N - 1.
    for item, effect in (
        ("Macho Brace", levelup.HOLD_EFFECT_EVS_UP_SPEED_DOWN),
        ("Exp. Share", levelup.HOLD_EFFECT_EXP_SHARE),
        ("Soothe Bell", levelup.HOLD_EFFECT_FRIENDSHIP_UP),
        ("Lucky Egg", levelup.HOLD_EFFECT_EXP_UP),
        ("Power Weight", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.HP]),
        ("Power Bracer", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.ATK]),
        ("Power Belt", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.DEF]),
        ("Power Lens", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.SPA]),
        ("Power Band", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.SPD]),
        ("Power Anklet", levelup.POWER_ITEM_HOLD_EFFECTS[Stat.SPE]),
    ):
        assert int(tables.item_hold_effect[names.item(item)]) == effect, item


def _equations(jaxpr) -> int:
    """Equations of ``jaxpr`` and of every sub-jaxpr (loops, conds, pjit calls)."""
    total = 0
    for eqn in jaxpr.eqns:
        total += 1
        for param in eqn.params.values():
            for sub in param if isinstance(param, list | tuple) else (param,):
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    total += _equations(getattr(inner, "jaxpr", inner))
    return total


# Fixed at implementation (T5): pay_exp measured 762 equations. step inlines it 6 times
# (Pursuit, the act body, the field pass, once per battler of the end of turn, replace):
# 46913 -> 50734 step equations, compile 14.7 s -> 16.9 s (task-5-report.md). One copy per
# end-of-turn check (22) made step take over an hour to compile. Raised 800 -> 880 at the E1 trial's
# Transform (G-69, c7ccdf8): a Transformed payer's party identity (``levelup.party_identity``) and
# the shadow stats it reloads took pay_exp 777 -> 834; linear growth from a modelled mechanic, about
# 340 step equations over the 6 copies. The ceiling still fails a duplicated stats calculation (about
# 74 equations) inside pay_exp; it does not count how often step inlines pay_exp.
PAY_EXP_CEILING = 880


def test_pay_exp_stays_small(tables, names):
    s = _prinplup_state(tables, names)
    jaxpr = jax.make_jaxpr(pay_exp)(tables, s, s, jnp.asarray(False), jnp.asarray(0))
    assert _equations(jaxpr.jaxpr) <= PAY_EXP_CEILING
    assert "while" not in str(jaxpr)  # no loop in the unrolled end of turn


def test_a_paid_state_keeps_its_dtypes(tables, names):
    s = _prinplup_state(tables, names)
    out = _ko(tables, s)
    for name, a, b in zip(s._fields, s, out, strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype, name


def test_step_without_a_faint_changes_no_exp(tables, names):
    s = _prinplup_state(tables, names)
    out = jax.jit(step)(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert np.asarray(out.exp).tolist() == np.asarray(s.exp).tolist()
    assert int(out.exp_mask) == int(s.exp_mask)


def test_spec_helper_matches_the_state(tables, names):
    # guard for the hand numbers above: _base reads the same table make_state uses.
    mon = _mon(names, "Prinplup", 16)
    s = make_state(tables, [mon], [mon])
    assert int(s.exp[0]) == _base(tables, names, "Prinplup", 16) == 2535
    assert dataclasses.replace(mon, exp=3119).exp == 3119
