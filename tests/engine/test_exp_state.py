"""K3d-exp T4: the experience block of ``BattleState`` and the participation mask.

Decomp (external/pokeplatinum):

* ``BattleSystem_ClearSideExpGain`` / ``BattleSystem_FlagBattlerExpGain``
  (src/battle/battle_lib.c:1456-1479): clear sets ``sideGetExpMask`` to 0; flag ORs in the bit
  of ``selectedPartySlot[PLAYER_1]`` when neither battler is in ``battlersSwitchingMask`` and
  the enemy battler's ``curHP`` is nonzero. In singles ``PLAYER_2`` / ``ENEMY_2`` are in that
  mask from the start (battle_lib.c:2031-2033), so only ``PLAYER_1`` against ``ENEMY_1``
  counts.
* ``BtlCmd_PokemonSendOut`` (src/battle/battle_script.c:543-635): a player send-out flags both
  enemies (:571-572, :592-593, :607-608, :622-623); an enemy send-out clears, then flags
  (:580-581, :595-596, :610-611, :625-626). The battle-start ``PokemonSlideIn`` (:437-529) does
  the same (:465-466, :474-475), for both leads (subscript_start_encounter.s:84, :91).
* Every send-out runs ``PokemonSendOut`` before ``BATTLE_SUBSCRIPT_HAZARDS_CHECK``
  (subscript_switch_pokemon.s:39/:43, subscript_show_party_list.s:18/:22,
  subscript_force_target_to_switch_or_flee.s:22/:31): the flag reads the HP before the hazards.
* A double KO: ``GoToIfAnySwitches`` (battle_script.c:2545-2551) picks the lowest switching
  battler first, so the player (battler 0) is replaced before the enemy (battler 1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.party import load_party
from nuzline.data.tables import to_numpy
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import LAYOUT, BattleState, make_state, spec_exp
from nuzline.engine.step import replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace
from tests.engine.helpers import SAMPLE_PARTY, A, assert_matches_reference, roark_state

ROOT = Path(__file__).resolve().parents[2]
DECOMP = ROOT / "external" / "pokeplatinum"

# design.md §2.1: one contiguous block after ai_ability_known, before deaths.
EXP_BLOCK = {
    "exp": ((6,), np.int32),
    "evs": ((2, 6, 6), np.int16),
    "nature": ((2, 6), np.int8),
    "exp_mask": ((), np.int8),
    "exp_boost": ((6,), np.int16),
    "pokerus": ((6,), np.int8),
    "friend_bonus": ((6,), np.int8),
    "learn_level": ((6,), np.int8),
    "learn_cursor": ((6,), np.int8),
    "learn_pin": ((), np.int8),
    "exp_refused": ((), np.int8),
}


def _base(tables, species: int, level: int) -> int:
    """``Pokemon_GetSpeciesBaseExpAt`` (src/pokemon.c:2327-2330): the rate by species."""
    return int(tables.exp_table[int(tables.species_exp_rate[species]), level])


def _mask(s) -> int:
    return int(s.exp_mask)


def _with_mask(s, mask: int):
    return s._replace(exp_mask=jnp.asarray(mask, jnp.int8))


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _replace_both(tables, s, side: int, slot: int):
    """``replace`` in both engines, asserted equal; the JAX state."""
    out = replace(tables, s, side, slot)
    _same(out, from_ref(ref_replace(to_numpy(tables), to_ref(s), side, slot)))
    return out


# --- the block ------------------------------------------------------------------------------


def test_the_exp_block_sits_between_ai_ability_known_and_deaths():
    fields = BattleState._fields
    start = fields.index("ai_ability_known") + 1
    assert fields[start : start + len(EXP_BLOCK)] == tuple(EXP_BLOCK)
    assert fields[start + len(EXP_BLOCK)] == "deaths"
    for name, layout in EXP_BLOCK.items():
        assert LAYOUT[name] == layout, name


def test_make_state_fills_the_block_from_the_specs(tables, names):
    player = load_party(SAMPLE_PARTY, names)
    opp = player[3:5]
    p1 = player[1]
    exp = _base(tables, p1.species, p1.level) + 7
    player[1] = dataclasses.replace(
        p1, exp=exp, exp_boost=150, pokerus=1, friend_bonus=1, evs=(1, 2, 3, 4, 5, 6), nature=3
    )
    s = make_state(tables, player, opp)
    want_exp = [_base(tables, m.species, m.level) for m in player]
    want_exp[1] = exp
    assert np.asarray(s.exp).tolist() == want_exp
    assert np.asarray(s.evs[0, 1]).tolist() == [1, 2, 3, 4, 5, 6]
    assert np.asarray(s.evs[1, : len(opp)]).tolist() == [list(m.evs) for m in opp]
    assert np.asarray(s.nature[0]).tolist() == [m.nature for m in player]
    assert np.asarray(s.nature[1, : len(opp)]).tolist() == [m.nature for m in opp]
    assert np.asarray(s.exp_boost).tolist() == [100, 150, 100, 100, 100, 100]
    assert np.asarray(s.pokerus).tolist() == [0, 1, 0, 0, 0, 0]
    assert np.asarray(s.friend_bonus).tolist() == [0, 1, 0, 0, 0, 0]
    for name in ("learn_level", "learn_cursor", "learn_pin", "exp_refused"):
        assert not np.any(np.asarray(getattr(s, name))), name


def test_default_exp_is_the_base_of_the_level(tables, names):
    # spec exp None: exp_table[rate(species)][level], so no level-up a real save would not
    # show (real exp >= base exp).
    player = load_party(SAMPLE_PARTY, names)
    s = make_state(tables, player, player[:1])
    for slot, mon in enumerate(player):
        assert int(s.exp[slot]) == _base(tables, mon.species, mon.level)
        assert int(s.exp[slot]) < _base(tables, mon.species, mon.level + 1)
    assert np.asarray(s.exp[len(player) :]).tolist() == []


def test_empty_player_slots_have_no_exp(tables, names):
    player = load_party(SAMPLE_PARTY, names)[:2]
    s = make_state(tables, player, player)
    assert np.asarray(s.exp[2:]).tolist() == [0] * 4
    assert np.asarray(s.exp_boost[2:]).tolist() == [0] * 4


@pytest.mark.parametrize("leads,mask", [((0, 0), 1), ((2, 1), 4), ((5, 0), 32)])
def test_the_mask_starts_with_the_player_lead(tables, names, leads, mask):
    # Battle start: PokemonSlideIn BTLSCR_ENEMY (clear, flag) and BTLSCR_PLAYER (flag),
    # subscript_start_encounter.s:84, :91 -> 1 << the player lead.
    player = load_party(SAMPLE_PARTY, names)
    assert _mask(make_state(tables, player, player[:3], leads=leads)) == mask


def test_spec_exp_must_match_the_level(tables, names):
    # SpeciesData_GetLevelAt (src/pokemon.c:2379-2392): the level of exp x is the last
    # L in 1..100 with table[L] <= x.
    mon = load_party(SAMPLE_PARTY, names)[0]
    lo, hi = _base(tables, mon.species, mon.level), _base(tables, mon.species, mon.level + 1)
    assert spec_exp(tables, mon) == lo
    for good in (lo, hi - 1):
        assert spec_exp(tables, dataclasses.replace(mon, exp=good)) == good
    for bad in (lo - 1, hi, -1):
        with pytest.raises(ValueError, match="exp"):
            spec_exp(tables, dataclasses.replace(mon, exp=bad))
        with pytest.raises(ValueError, match="exp"):
            make_state(tables, [dataclasses.replace(mon, exp=bad)], [mon])
    top = dataclasses.replace(mon, level=100, exp=_base(tables, mon.species, 100))
    assert spec_exp(tables, top) == _base(tables, mon.species, 100)


def test_the_party_loader_refuses_exp_inconsistent_with_the_level(tmp_path, tables, names):
    mon = load_party(SAMPLE_PARTY, names)[0]  # Monferno L14
    lo, hi = _base(tables, mon.species, 14), _base(tables, mon.species, 15)
    text = SAMPLE_PARTY.read_text(encoding="utf-8").split("[[pokemon]]")
    for exp, ok in ((lo, True), (hi - 1, True), (hi, False), (lo - 1, False)):
        f = tmp_path / "p.toml"
        f.write_text("[[pokemon]]" + text[1] + f"exp = {exp}\n", encoding="utf-8")
        assert load_party(f, names)[0].exp == exp  # without tables: no check
        if ok:
            assert load_party(f, names, tables)[0].exp == exp
        else:
            with pytest.raises(ValueError, match="Monferno.*exp"):
                load_party(f, names, tables)


# --- the participation mask (both engines) -----------------------------------------------------


def test_a_player_switch_adds_the_newcomer_while_the_foe_has_hp(tables, names):
    s = roark_state(tables, names)
    assert _mask(s) == 1
    out = assert_matches_reference(
        tables, s, A(MOVE_ACTIONS + 2, "Tackle", s, names), empty_outcome()
    )
    assert int(out.active[0]) == 2 and _mask(out) == 0b101


def test_an_opponent_switch_resets_the_mask_to_the_player_active(tables, names):
    s = _with_mask(roark_state(tables, names), 0b100110)
    out = assert_matches_reference(
        tables, s, A("Scratch", MOVE_ACTIONS + 1, s, names), empty_outcome()
    )
    assert int(out.active[1]) == 1 and _mask(out) == 1


def test_an_opponent_replacement_resets_the_mask(tables, names):
    s = _with_mask(roark_state(tables, names).replace_hp(1, 0, 0), 0b1011)
    out = _replace_both(tables, s, 1, 2)
    assert _mask(out) == 1


def test_the_flag_reads_the_newcomer_before_the_hazards(tables, names):
    # PokemonSendOut (show_party_list.s:18) precedes the hazards check (:22): a newcomer that
    # Stealth Rock faints was still flagged with HP, so the mask was {player active} when
    # TurnEnd's payout (bcp.c:1846, K3d-exp T5) paid for it: slot 0 alone gains exp (not the
    # old participants 1 and 2), and the payout leaves the mask 0 (battle_script.c:10190-10193).
    s = roark_state(tables, names).replace_hp(1, 0, 0).replace_hp(1, 1, 1)
    s = _with_mask(s._replace(stealth_rock=s.stealth_rock.at[1].set(1)), 0b110)
    out = _replace_both(tables, s, 1, 1)
    assert int(out.hp[1, 1]) == 0 and _mask(out) == 0
    gained = (np.asarray(out.exp) - np.asarray(s.exp)).tolist()
    assert gained[0] > 0 and gained[1:3] == [0, 0]


def test_a_player_replacement_adds_the_newcomer(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 0)
    out = _replace_both(tables, s, 0, 3)
    assert _mask(out) == 0b1001


def test_a_player_entry_facing_a_fainted_foe_adds_nothing(tables, names):
    # The flag needs the enemy's curHP (battle_lib.c:1469).
    s = roark_state(tables, names).replace_hp(0, 0, 0).replace_hp(1, 0, 0)
    out = _replace_both(tables, s, 0, 3)
    assert _mask(out) == 1


def test_a_double_ko_ends_with_the_new_player_active(tables, names):
    # GoToIfAnySwitches (battle_script.c:2545-2551): the player (battler 0) enters first and
    # adds nothing against the fainted enemy; the enemy's entry then clears and flags the
    # player's new active.
    s = roark_state(tables, names).replace_hp(0, 0, 0).replace_hp(1, 0, 0)
    s = _replace_both(tables, s, 0, 4)
    assert _mask(s) == 1
    s = _replace_both(tables, s, 1, 1)
    assert _mask(s) == 1 << 4


def _roar_outcome(target=(0, 0)):
    return empty_outcome()._replace(
        roar_level=jnp.ones((2,), jnp.int8), roar_target=jnp.asarray(target, jnp.int8)
    )


def test_a_player_roar_resets_the_mask_to_the_player_active(tables, names):
    # BTLSCR_DEFENDER on an enemy (battle_script.c:610-611): clear, then flag.
    s = roark_state(tables, names, player_moves=["Roar"])
    s = _with_mask(s, 0b111)
    out = assert_matches_reference(tables, s, A("Roar", "Tackle", s, names), _roar_outcome())
    assert int(out.active[1]) == 1 and _mask(out) == 1


def test_an_opponent_roar_adds_the_dragged_player_pokemon(tables, names):
    # BTLSCR_DEFENDER on the player (battle_script.c:607-608): flag only.
    s = roark_state(tables, names, opp_moves=["Roar"])
    out = assert_matches_reference(tables, s, A("Scratch", "Roar", s, names), _roar_outcome())
    assert int(out.active[0]) == 1 and _mask(out) == 0b11


# --- decomp text ------------------------------------------------------------------------------


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_send_out_precedes_the_hazards_in_every_send_out_script():
    subs = DECOMP / "res" / "battle" / "scripts" / "subscripts"
    for name, battler in (
        ("subscript_switch_pokemon.s", "BTLSCR_SWITCHED_MON"),
        ("subscript_show_party_list.s", "BTLSCR_SWITCHED_MON"),
        ("subscript_replace_fainted.s", "BTLSCR_SWITCHED_MON"),
        ("subscript_force_target_to_switch_or_flee.s", "BTLSCR_DEFENDER"),
    ):
        text = (subs / name).read_text(encoding="utf-8")
        assert text.index(f"PokemonSendOut {battler}") < text.index(
            "BATTLE_SUBSCRIPT_HAZARDS_CHECK"
        ), name
    lib = (DECOMP / "src" / "battle" / "battle_lib.c").read_text(encoding="utf-8")
    assert "battleCtx->sideGetExpMask[(battler >> 1) & 1] = 0;" in lib
    assert "&& battleCtx->battleMons[battler].curHP) {" in lib


def test_state_md_maps_the_exp_fields():
    from tests.engine.test_state import _state_md_rows

    rows = {field: (pytree, status) for field, pytree, status in _state_md_rows()}
    assert rows["BattleMon.exp"] == ("exp", "mapped")
    assert rows["BattleContext.sideGetExpMask"] == ("exp_mask", "mapped")
    for field in (
        "BattleContext.gainedExp",
        "BattleContext.sharedExp",
        "BattleContext.levelUpMons",
    ):
        assert "do not change the fight" not in rows[field][1], field
