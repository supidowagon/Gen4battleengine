"""Rindo and Chilan Berry (K3b1 task 13): ``subscript_type_resist_berry.s``.

``BattleControllerPlayer_UpdateHP`` (bcp.c:3346-3443) caps the damage for an endure
(bcp.c:3396-3404), stores it for Bide (:3407) and hands it to ``subscript_update_hp.s``
as ``hpCalcTemp``; that script calls the resist-berry subscript (:9) while a battler still
has an action and the sprite blinks (move hits, Pursuit, the confusion self-hit; not recoil,
Rough Skin or the end of turn). The berry is skipped for ``SYSCTL_IGNORE_TYPE_CHECKS``
(Night Shade, Seismic Toss) and OHKO moves; Chilan (``HOLD_EFFECT_WEAKEN_NORMAL``) halves
any Normal-type hit, Rindo (``HOLD_EFFECT_WEAKEN_SE_GRASS``) a super-effective Grass one
(``MOVE_STATUS_SUPER_EFFECTIVE``). ``DivideVarByValue`` (``BattleSystem_Divide``, at least
1) halves ``hpCalcTemp`` before ``BtlCmd_UpdateHealthBarValue`` caps it at the HP, and the
berry is removed. Drain and recoil read ``hitDamage`` (the halved loss); Bide keeps the
unhalved ``battleCtx->damage``. Pluck / Bug Bite eat either berry with no effect
(battle_lib.c:5833-5848).
"""

import jax
import jax.numpy as jnp
import pytest

from nuzline.engine.damage import confusion_damage, damage_table
from nuzline.engine.ids import RESIST_ANY_NORMAL, ItemId, Type
from nuzline.engine.observe import OBS_FIELDS, observe
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import SLOTS
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state

HP = 500
_jit_step = jax.jit(step)


def _item(s, side, item, names):
    return s._replace(item=s.item.at[side, 0].set(names.item(item)))


def _holder(tables, names, item, hp=HP, **kw):
    """Roark's lead (or ``opp_species``) holding ``item`` at ``hp`` of ``hp``; the player's
    lead holds nothing (Grotle's Sitrus Berry would heal it after a hit)."""
    s = _item(roark_state(tables, names, **kw), 0, "", names)
    s = _item(s, 1, item, names)
    return s._replace(hp=s.hp.at[1, 0].set(hp), max_hp=s.max_hp.at[1, 0].set(hp))


def _roll15(tables, s, side, move, names):
    """The hit's damage under ``empty_outcome`` (roll 15, no crit)."""
    return int(damage_table(tables, s, side, names.move(move), False)[15])


def _consumed(out, side, item, names):
    return int(out.item[side, 0]) == 0 and int(out.item_consumed[side, 0]) == names.item(item)


def _kept(out, side, item, names):
    return int(out.item[side, 0]) == names.item(item) and int(out.item_consumed[side, 0]) == 0


def test_tables_map_rindo_and_chilan_in_both_games(tables, names, kaizo_tables, kaizo_names):
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for item, resisted in (("Rindo Berry", Type.GRASS), ("Chilan Berry", RESIST_ANY_NORMAL)):
            assert int(t.item_effect[n.item(item)]) == ItemId.RESIST_BERRY, item
            assert int(t.item_param[n.item(item), 1]) == resisted, item
        # Only the berries whose code is implemented are supported.
        assert int(t.item_effect[n.item("Occa Berry")]) == ItemId.UNSUPPORTED


def test_rindo_halves_se_grass_and_is_consumed(tables, names):
    s = _holder(tables, names, "Rindo Berry", player_species="Grotle")  # vs Geodude: x4
    d = _roll15(tables, s, 0, "Razor Leaf", names)
    acts = A("Razor Leaf", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert HP - int(out.hp[1, 0]) == d // 2 and _consumed(out, 1, "Rindo Berry", names)
    # The halving comes before the HP cap: a hit that would KO leaves HP - d // 2.
    saved = _holder(tables, names, "Rindo Berry", hp=d - 1, player_species="Grotle")
    out = assert_matches_reference(tables, saved, acts, empty_outcome())
    assert int(out.hp[1, 0]) == d - 1 - d // 2 and _consumed(out, 1, "Rindo Berry", names)
    # Not enough: it still faints, and the berry was used.
    low = _holder(tables, names, "Rindo Berry", hp=d // 2, player_species="Grotle")
    out = assert_matches_reference(tables, low, acts, empty_outcome())
    assert int(out.hp[1, 0]) == 0 and _consumed(out, 1, "Rindo Berry", names)


def test_rindo_ignores_neutral_grass(tables, names):
    # Grass into Shinx (Electric) is neutral; into Wingull (Water/Flying) the x2 and the x0.5
    # cancel (ApplyTypeMultiplier clears the flag, battle_lib.c:7556-7573): no
    # MOVE_STATUS_SUPER_EFFECTIVE either way.
    for foe in ("Shinx", "Wingull"):
        s = _holder(tables, names, "Rindo Berry", player_species="Grotle", opp_species=foe)
        d = _roll15(tables, s, 0, "Razor Leaf", names)
        out = assert_matches_reference(
            tables, s, A("Razor Leaf", "Leer", s, names), empty_outcome()
        )
        assert HP - int(out.hp[1, 0]) == d and _kept(out, 1, "Rindo Berry", names), foe
    # A super-effective hit of another type (Water into Geodude) leaves it too.
    s = _holder(tables, names, "Rindo Berry", player_species="Prinplup")
    d = _roll15(tables, s, 0, "Bubble", names)
    out = assert_matches_reference(tables, s, A("Bubble", "Leer", s, names), empty_outcome())
    assert HP - int(out.hp[1, 0]) == d and _kept(out, 1, "Rindo Berry", names)


def test_chilan_any_normal(tables, names):
    # Tackle into Geodude is not very effective; Chilan needs no super-effective hit.
    s = _holder(tables, names, "Chilan Berry", player_moves=["Tackle"])
    d = _roll15(tables, s, 0, "Tackle", names)
    out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), empty_outcome())
    assert HP - int(out.hp[1, 0]) == max(d // 2, 1) and _consumed(out, 1, "Chilan Berry", names)
    # Neutral Normal (Scratch into Shinx) too.
    s = _holder(tables, names, "Chilan Berry", opp_species="Shinx")
    d = _roll15(tables, s, 0, "Scratch", names)
    out = assert_matches_reference(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert HP - int(out.hp[1, 0]) == d // 2 and _consumed(out, 1, "Chilan Berry", names)
    # A non-Normal hit (Mach Punch) leaves it.
    s = _holder(tables, names, "Chilan Berry")
    d = _roll15(tables, s, 0, "Mach Punch", names)
    out = assert_matches_reference(tables, s, A("Mach Punch", "Leer", s, names), empty_outcome())
    assert HP - int(out.hp[1, 0]) == d and _kept(out, 1, "Chilan Berry", names)


def test_resist_berry_first_hit_of_a_multi_hit_move_only(tables, names):
    s = _holder(tables, names, "Chilan Berry", player_moves=["DoubleSlap"], opp_species="Shinx")
    d = _roll15(tables, s, 0, "DoubleSlap", names)
    three = empty_outcome()._replace(n_hits=jnp.array([3, 2], jnp.int8))
    out = assert_matches_reference(tables, s, A("DoubleSlap", "Leer", s, names), three)
    assert HP - int(out.hp[1, 0]) == d // 2 + 2 * d and _consumed(out, 1, "Chilan Berry", names)


def test_resist_berry_skips_ignore_type_checks(tables, names):
    """effect_script_0087.s sets SYSCTL_IGNORE_TYPE_CHECKS: subscript_type_resist_berry.s:5
    leaves the berry even for a Normal-typed Seismic Toss (retyped here; vanilla's is
    Fighting)."""
    toss = names.move("Seismic Toss")
    normal = tables._replace(move_type=tables.move_type.at[toss].set(Type.NORMAL))
    s = _holder(tables, names, "Chilan Berry", player_moves=["Seismic Toss"])
    out = assert_matches_reference(normal, s, A("Seismic Toss", "Leer", s, names), empty_outcome())
    level = int(s.level[0, 0])
    assert HP - int(out.hp[1, 0]) == level and _kept(out, 1, "Chilan Berry", names)


def test_chilan_confusion_self_hit(tables, names):
    """subscript_hurt_self_in_confusion.s -> UPDATE_HP: calcTemp is the chosen move's table
    type (moveType is Normal since BattleContext_Init) and the hit is never super-effective,
    so Chilan halves the self-hit of a battler that chose a Normal move (Hidden Power's
    table type is Normal too), and Rindo never triggers."""
    hurt = empty_outcome()._replace(confusion_hit=jnp.array([1, 0], jnp.int8))

    def confused(item, moves):
        s = roark_state(tables, names, player_moves=moves)
        s = _item(s, 0, item, names)
        s = s._replace(hp=s.hp.at[0, 0].set(HP), max_hp=s.max_hp.at[0, 0].set(HP))
        return s._replace(confusion=s.confusion.at[0].set(3))

    raw = int(confusion_damage(tables, confused("", ["Scratch"]), 0, 15))
    for move, item, halved in (
        ("Scratch", "Chilan Berry", True),
        ("Hidden Power", "Chilan Berry", True),
        ("Ember", "Chilan Berry", False),
        ("Razor Leaf", "Rindo Berry", False),
    ):
        s = confused(item, [move])
        out = assert_matches_reference(tables, s, A(move, "Leer", s, names), hurt)
        lost = HP - int(out.hp[0, 0])
        if halved:
            assert lost == raw // 2 and _consumed(out, 0, item, names), move
        else:
            assert lost == raw and _kept(out, 0, item, names), move


def test_bide_stores_the_unhalved_damage(tables, names):
    """bcp.c:3407 adds battleCtx->damage to storedDamage before subscript_update_hp.s halves
    hpCalcTemp."""
    s = roark_state(tables, names, player_moves=["Bide"])
    s = _item(s, 0, "Chilan Berry", names)
    s = s._replace(hp=s.hp.at[0, 0].set(HP), max_hp=s.max_hp.at[0, 0].set(HP))
    d = _roll15(tables, s, 1, "Tackle", names)
    out = assert_matches_reference(tables, s, A("Bide", "Tackle", s, names), empty_outcome())
    assert HP - int(out.hp[0, 0]) == d // 2 and _consumed(out, 0, "Chilan Berry", names)
    assert int(out.bide_damage[0]) == d


def test_drain_and_recoil_use_the_halved_damage(tables, names):
    """subscript_drain_half_damage_dealt.s and subscript_recoil_1_4.s read hitDamage, the
    HP the target lost (battle_script.c:1699-1704)."""
    s = _holder(tables, names, "Rindo Berry", player_species="Grotle", player_moves=["Absorb"])
    s = s.replace_hp(0, 0, 10)
    d = _roll15(tables, s, 0, "Absorb", names)
    out = assert_matches_reference(tables, s, A("Absorb", "Leer", s, names), empty_outcome())
    lost = HP - int(out.hp[1, 0])
    assert lost == max(d // 2, 1) and int(out.hp[0, 0]) == 10 + max(lost // 2, 1)
    s = _holder(tables, names, "Chilan Berry", player_moves=["Take Down"], opp_species="Shinx")
    d = _roll15(tables, s, 0, "Take Down", names)
    out = assert_matches_reference(tables, s, A("Take Down", "Leer", s, names), empty_outcome())
    lost = HP - int(out.hp[1, 0])
    assert lost == d // 2
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(lost // 4, 1)


def test_struggle_into_chilan(tables, names):
    # Struggle is Normal-typed (no chart: battle_lib.c:2572), so Chilan halves it.
    s = _holder(tables, names, "Chilan Berry", opp_species="Shinx")
    s = s._replace(pp=s.pp.at[0, 0].set(0))
    out = assert_matches_reference(tables, s, A(0, "Leer", s, names), empty_outcome())
    assert _consumed(out, 1, "Chilan Berry", names)


@pytest.mark.parametrize("move", ["Pluck", "Bug Bite"])
def test_pluck_eats_a_resist_berry(tables, names, move):
    for item in ("Rindo Berry", "Chilan Berry"):
        s = _holder(tables, names, item, player_moves=[move])
        d = _roll15(tables, s, 0, move, names)
        out = assert_matches_reference(tables, s, A(move, "Leer", s, names), empty_outcome())
        assert HP - int(out.hp[1, 0]) == d and _consumed(out, 1, item, names), item
        assert int(out.hp[0, 0]) == int(s.hp[0, 0])


def test_the_berry_is_seen(tables, names):
    """The berry's message ("The {item} weakened {move}'s power!") always comes with its
    RemoveItem, which the observation shows in ``item`` / ``item_consumed``."""
    s = _holder(tables, names, "Rindo Berry", player_species="Grotle")
    acts = A("Razor Leaf", "Leer", s, names)
    used = observe(tables, s, acts, empty_outcome(), _jit_step(tables, s, acts, empty_outcome()))
    kept = _holder(tables, names, "Rindo Berry", player_species="Grotle", opp_species="Shinx")
    acts = A("Razor Leaf", "Leer", kept, names)
    other = observe(
        tables, kept, acts, empty_outcome(), _jit_step(tables, kept, acts, empty_outcome())
    )
    field = dict(OBS_FIELDS)["item_consumed"]
    assert int(used[field][SLOTS]) == names.item("Rindo Berry")  # [side 1, slot 0]
    assert int(other[field][SLOTS]) == 0


@pytest.mark.parametrize("crit", [0, 1])
@pytest.mark.parametrize("item", ["Rindo Berry", "Chilan Berry"])
@pytest.mark.parametrize(
    ("player", "move"),
    [
        ("Grotle", "Razor Leaf"),
        ("Grotle", "Absorb"),
        ("Grotle", "Tackle"),
        ("Monferno", "Scratch"),
        ("Prinplup", "Bubble"),
        ("Grotle", "Hidden Power"),
    ],
)
def test_resist_berries_match_reference(tables, names, crit, item, player, move):
    s = _holder(tables, names, item, hp=30, player_species=player, player_moves=[move])
    o = empty_outcome()._replace(crit=jnp.full((2, 5), crit, jnp.int8))
    # The foe hits back with Tackle: the player's lead holds the same berry.
    s = _item(s, 0, item, names)
    assert_matches_reference(tables, s, A(move, "Tackle", s, names), o)
