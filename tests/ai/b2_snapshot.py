# ruff: noqa: B023  (each lambda runs inside its loop iteration)
"""The B = 2 outputs of every battler-taking AI command on fixed states (E4-A-W1 step 1).

``snapshot`` runs each command of ``nuzline.ai.commands`` that resolves a battler, and every
command that read the player's or the AI's battler by index, on fixed Roark / sample-party states
at B = 2 (attacker 1, defender 0, the script arguments DEFENDER, ATTACKER, DEFENDER_PARTNER and
ATTACKER_PARTNER) and returns one JSON-able row per call. ``b2_snapshot.json`` holds the rows the
engine gave before the context took its battlers (tip 85dd4cd0); ``test_commands_four`` compares
them: the battler rework must leave every B = 2 output byte-identical. A call that raises (an
argument the command refuses) has no row: a refusal is coverage, not an output to pin.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.ai import commands as c
from nuzline.ai.commands import Battler, LoadType, MonCondition, SideCondition, TypeMulti
from nuzline.ai.outcome import READ_DRAWS, no_calc_draws
from nuzline.engine.ids import Stat
from tests.ai.fixtures import CRANIDOS, GEODUDE, ONIX, outcome, versus

_ARGS = (Battler.DEFENDER, Battler.ATTACKER, Battler.DEFENDER_PARTNER, Battler.ATTACKER_PARTNER)
_EFFECTS = (1, 4, 17, 31, 43, 111, 176)  # battle effect ids: a few, hit or miss
_MOVES = (0, 33, 89, 105)


def _filled(s, field: str, shape_axis: int, values):
    arr = getattr(s, field)
    for k, v in enumerate(values[:shape_axis]):
        arr = arr.at[k].set(jnp.asarray(v, arr.dtype))
    return s._replace(**{field: arr})


def states(tables, names):
    """The fixed B = 2 states: plain, and two with the per-battler fields set to a pattern."""
    plain = versus(tables, names, GEODUDE)
    busy = versus(tables, names, ONIX, "Starly", opp_hp=20)
    per_battler = (
        "taunt",
        "torment",
        "last_move",
        "protect_count",
        "fake_out_turn",
        "foresight",
        "disable_turns",
        "encore_turns",
        "aqua_ring",
        "infatuated",
        "miracle_eye",
        "last_resort_count",
        "curse",
        "nightmare",
        "perish_song",
        "yawn",
        "lock_on",
        "ingrain",
        "mean_look",
        "imprison",
        "imprisoned",
        "recycle_item",
        "focus_energy",
        "power_trick",
        "magnet_rise",
        "camouflage",
        "embargo",
        "flash_fire",
        "ability_suppressed",
        "confusion",
        "leech_seed",
        "substitute_hp",
    )
    for k, name in enumerate(per_battler):
        busy = _filled(busy, name, 2, [1 + (k % 3), 1 + ((k + 1) % 3)])
    for k, name in enumerate(
        ("reflect", "light_screen", "spikes", "toxic_spikes", "tailwind", "mist", "safeguard")
    ):
        busy = _filled(busy, name, 2, [1 + (k % 2), 2 - (k % 2)])
    busy = busy._replace(
        stealth_rock=busy.stealth_rock.at[0].set(1),
        stages=busy.stages.at[0, 1:]
        .set(jnp.arange(7, dtype=busy.stages.dtype) - 3)
        .at[1, 2]
        .set(2),
        item=busy.item.at[1, ONIX].set(c.ITEM_POWER_HERB).at[0, 0].set(111),
        status=busy.status.at[0, 0].set(2),
        ai_item_known=busy.ai_item_known.at[0].set(c.ITEM_POWER_HERB),
        ai_ability_known=busy.ai_ability_known.at[0].set(26),
        ai_seen_moves=busy.ai_seen_moves.at[0, 0].set(33),
    )
    third = versus(tables, names, CRANIDOS, "Starly", opp_hp=40, player_hp=11, turn=3)
    third = third._replace(last_move=third.last_move.at[0].set(89).at[1].set(33))
    return {"plain": plain, "busy": busy, "cranidos": third}


def _row(ctx):
    return [int(ctx.calc_temp), int(ctx.jump)]


_REFUSED = object()


def _run(fn):
    try:
        return _row(fn())
    except (AssertionError, NotImplementedError):  # a refusal: no row
        return _REFUSED


def _slot_ctx(tables, s, slot: int):
    reads = 8
    script = np.full((4, reads), READ_DRAWS[0], np.uint8)
    o = outcome()._replace(script_rolls=jnp.asarray(script), calc_draws=no_calc_draws((), reads))
    return c.at_slot(c.new_context(tables, s, o), slot)


def snapshot(tables, names) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, s in states(tables, names).items():
        for slot in (0, 1):
            ctx = c.at_slot(c.new_context(tables, s, outcome()), slot)
            sctx = _slot_ctx(tables, s, slot)
            key = f"{label}/{slot}"
            out[f"{key}/context"] = [
                np.asarray(ctx.scores).tolist(),
                np.asarray(ctx.seen_moves).tolist(),
                int(ctx.player_ability),
                np.asarray(ctx.guess_abilities).tolist(),
                bool(ctx.guessing),
                np.asarray(ctx.speeds).tolist(),
                np.asarray(ctx.damage.value).tolist(),
                np.asarray(ctx.damage.prob).tolist(),
                np.asarray(ctx.damage_w.value).tolist(),
                int(ctx.move),
            ]
            calls: dict[str, object] = {}
            for b in _ARGS:
                n = b.name
                one = {
                    f"hp_lt_{n}": lambda b=b: c.if_hp_percent_less_than(ctx, b, 50),
                    f"hp_gt_{n}": lambda b=b: c.if_hp_percent_greater_than(ctx, b, 50),
                    f"hp_eq_{n}": lambda b=b: c.if_hp_percent_equal_to(ctx, b, 100),
                    f"hp_ne_{n}": lambda b=b: c.if_hp_percent_not_equal_to(ctx, b, 100),
                    f"status_{n}": lambda b=b: c.if_status(ctx, b, MonCondition.ANY),
                    f"notstatus_{n}": lambda b=b: c.if_not_status(ctx, b, MonCondition.BURN),
                    f"party_status_{n}": lambda b=b: c.if_party_member_status(
                        ctx, b, MonCondition.POISON
                    ),
                    f"is_type_{n}": lambda b=b: c.flag_battler_is_type(ctx, b, 0),
                    f"is_type_rock_{n}": lambda b=b: c.flag_battler_is_type(ctx, b, 5),
                    f"prev_move_{n}": lambda b=b: c.load_battler_previous_move(ctx, b),
                    f"alive_{n}": lambda b=b: c.count_alive_party_battlers(ctx, b),
                    f"ability_{n}": lambda b=b: c.load_battler_ability(sctx, b),
                    f"ability_flat_{n}": lambda b=b: c.load_battler_ability(ctx, b),
                    f"check_ability_{n}": lambda b=b: c.check_battler_ability(ctx, b, 26),
                    f"check_ability0_{n}": lambda b=b: c.check_battler_ability(ctx, b, 5),
                    f"held_item_{n}": lambda b=b: c.load_held_item(ctx, b),
                    f"held_effect_{n}": lambda b=b: c.load_held_item_effect(ctx, b),
                    f"held_eq_{n}": lambda b=b: c.if_held_item_equal_to(ctx, b, c.ITEM_POWER_HERB),
                    f"held_eq0_{n}": lambda b=b: c.if_held_item_equal_to(ctx, b, 0),
                    f"fling_{n}": lambda b=b: c.load_fling_power(ctx, b),
                    f"recycle_{n}": lambda b=b: c.load_recycle_item(ctx, b),
                    f"sum_stages_{n}": lambda b=b: c.sum_positive_stat_stages(ctx, b),
                    f"turn_count_{n}": lambda b=b: c.load_battler_turn_count(ctx, b),
                    f"first_turn_{n}": lambda b=b: c.load_is_first_turn_in_battle(ctx, b),
                    f"protect_{n}": lambda b=b: c.load_protect_chain(ctx, b),
                    f"last_resort_{n}": lambda b=b: c.if_can_use_last_resort(ctx, b),
                    f"gender_{n}": lambda b=b: c.load_gender(ctx, b),
                    f"flash_{n}": lambda b=b: c.if_activated_flash_fire(ctx, b),
                    f"fainted_{n}": lambda b=b: c.if_battler_fainted(ctx, b),
                    f"under_disable_{n}": lambda b=b: c.if_battler_under_effect(
                        ctx, b, c.CHECK_DISABLE
                    ),
                    f"under_encore_{n}": lambda b=b: c.if_battler_under_effect(
                        ctx, b, c.CHECK_ENCORE
                    ),
                    f"speed_rank_{n}": lambda b=b: c.load_battler_speed_rank(sctx, b),
                    f"deals_more_{n}": lambda b=b: c.if_battler_deals_more_damage(
                        sctx, b, c.USE_MAX_DAMAGE
                    ),
                }
                for stat in (Stat.ATK, Stat.SPE, Stat.ACC):
                    sn = stat.name
                    one[f"stage_eq_{n}_{sn}"] = lambda b=b, st=stat: c.if_stat_stage_equal_to(
                        ctx, b, st, 6
                    )
                    one[f"stage_ne_{n}_{sn}"] = lambda b=b, st=stat: c.if_stat_stage_not_equal_to(
                        ctx, b, st, 6
                    )
                    one[f"stage_lt_{n}_{sn}"] = lambda b=b, st=stat: c.if_stat_stage_less_than(
                        ctx, b, st, 6
                    )
                    one[f"stage_gt_{n}_{sn}"] = lambda b=b, st=stat: c.if_stat_stage_greater_than(
                        ctx, b, st, 6
                    )
                    one[f"stage_diff_{n}_{sn}"] = lambda b=b, st=stat: c.diff_stat_stages(
                        ctx, b, st
                    )
                for move in _MOVES:
                    one[f"known_{n}_{move}"] = lambda b=b, m=move: c.if_move_known(ctx, b, m)
                    one[f"notknown_{n}_{move}"] = lambda b=b, m=move: c.if_move_not_known(ctx, b, m)
                for eff in _EFFECTS:
                    one[f"eff_known_{n}_{eff}"] = lambda b=b, e=eff: c.if_move_effect_known(
                        ctx, b, e
                    )
                    one[f"eff_notknown_{n}_{eff}"] = lambda b=b, e=eff: c.if_move_effect_not_known(
                        ctx, b, e
                    )
                for mask in c.VolatileCondition:
                    one[f"volatile_{n}_{mask.name}"] = lambda b=b, m=mask: c.if_volatile_status(
                        ctx, b, m
                    )
                    one[f"notvolatile_{n}_{mask.name}"] = lambda b=b, m=mask: (
                        c.if_not_volatile_status(ctx, b, m)
                    )
                for mask in c.MoveEffectMask:
                    one[f"effect_{n}_{mask.name}"] = lambda b=b, m=mask: c.if_move_effect(ctx, b, m)
                    one[f"noteffect_{n}_{mask.name}"] = lambda b=b, m=mask: c.if_not_move_effect(
                        ctx, b, m
                    )
                for cond in SideCondition:
                    one[f"side_{n}_{cond.name}"] = lambda b=b, m=cond: c.if_side_condition(
                        ctx, b, m
                    )
                for cond in (SideCondition.SPIKES, SideCondition.TOXIC_SPIKES):
                    one[f"layers_{n}_{cond.name}"] = lambda b=b, m=cond: c.load_spikes_layers(
                        ctx, b, m
                    )
                calls.update(one)
            for t in LoadType:
                calls[f"load_type_{t.name}"] = lambda t=t: c.load_type_from(ctx, t)
            for cls in TypeMulti:
                calls[f"effectiveness_{cls.name}"] = lambda k=cls: c.if_move_effectiveness_equals(
                    ctx, k
                )
            for op in (0, 1, 2):
                calls[f"level_{op}"] = lambda op=op: c.if_level(ctx, op)
            calls.update(
                {
                    "not_taunted": lambda: c.if_target_is_not_taunted(ctx),
                    "target_is_partner": lambda: c.if_target_is_partner(ctx),
                    "move_pp": lambda: c.load_current_move_pp(ctx),
                    "kills_max": lambda: c.if_current_move_kills(ctx, c.USE_MAX_DAMAGE),
                    "super_effective": lambda: c.if_has_super_effective_move(ctx),
                    "party_more": lambda: c.if_party_member_deals_more_damage(
                        sctx, c.USE_MAX_DAMAGE
                    ),
                    "flag_damage": lambda: c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE),
                    "with_partner": lambda: c.check_if_highest_damage_with_partner(
                        ctx, c.USE_MAX_DAMAGE
                    ),
                    "defender_move_class": lambda: c.load_defender_last_used_move_class(ctx),
                    "no_damaging": lambda: c.if_attacker_has_no_damaging_moves(ctx),
                    "slower": lambda: c.if_speed_compare_equal_to(ctx, 1),
                    "effectiveness_class": lambda: ctx._replace(
                        calc_temp=c.effectiveness_class(tables, s, ctx.move)
                    ),
                }
            )
            for name, fn in calls.items():
                row = _run(fn)
                if row is not _REFUSED:
                    out[f"{key}/{name}"] = row
            out[f"{key}/after_speed_compare"] = [
                bool(c.if_speed_compare_equal_to(ctx, 1).written.any()),
                int(c.if_speed_compare_equal_to(ctx, 1).first_cmp[slot]),
            ]
    return out
