"""Convert an emulator battle trace into engine states, actions and outcomes.

``tools/trace_battle.lua`` records a raw trace (JSON lines) in DeSmuME. This module
bundles it with the player's party and the trainer id into one JSON file
(``tests/golden/*.json``: extracted battle state only, never ROM bytes), and
``trace_to_states`` turns a bundle into ``(state, actions, outcome)`` per turn for
``tests/test_golden.py``. The formats, the RAM layout and the inference rules are in
``docs/tracing.md``.

Per turn, the before-state is the engine's replay of the previous turn with every field
the trace records (``COMPARED_FIELDS``) written over it. The fields the trace lacks are
carried forward from the engine. The actions come from PP drops and active-slot
changes. Hits, crits and damage rolls come from the ``hit``/``move_end`` records,
matched against ``damage_for_roll``. The remaining chance fields are found by
replaying every candidate through ``step`` and keeping the one that best matches the
next record.

CLI: ``uv run python -m tools.trace_to_state --trace <raw.jsonl> --out
tests/golden/roark_01.json (--party tests/fixtures/golden_party_roark_01.toml | --save
<save.dsv>) [--game vanilla|kaizo] [--trainer roark | --trainer-id N]``. ``--save`` takes
the player's party from a save (``SaveMon.to_spec``), so no party file is typed in.
``--game`` picks the data set (tables, trainers) the bundle is built and converted with;
the bundle records it as ``"game"`` (a bundle without it is vanilla).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.games import VANILLA, get_game
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import Trainer
from nuzline.engine.actions import FIRST_ITEM_ACTION, legal_actions, usable_trainer_items
from nuzline.engine.damage import (
    PRESENT_HEAL,
    PRESENT_TABLE,
    confusion_damage,
    damage_for_roll,
)
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status, Weather
from nuzline.engine.order import MOVE_ACTIONS, effective_speed, selected_move
from nuzline.engine.outcome import MAX_HITS, Outcome, empty_outcome
from nuzline.engine.state import LAYOUT, SLOTS, BattleState, make_state
from nuzline.engine.step import needs_replacement, replace, step
from nuzline.saves.save import SaveFile
from nuzline.saves.status import decode_status

INTERMEDIATE = Path(__file__).resolve().parents[1] / "data" / "intermediate"
BUNDLE_FORMAT = "nuzline-golden-trace/1"

# The BattleState fields a ``turn`` record fills; tests/test_golden.py compares them.
COMPARED_FIELDS: tuple[str, ...] = (
    "species",
    "level",
    "hp",
    "max_hp",
    "status",
    "status_counter",
    "item",
    "moves",
    "pp",
    "active",
    "stages",
    "weather",
)
_START_FIELDS = ("species", "level", "max_hp", "moves", "item")

# BattleMon.statBoosts order (generated/battle_stats.txt): HP, Atk, Def, Spe, SpA, SpD,
# Acc, Eva; 6 is neutral.
STAGE_ORDER = (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD, Stat.ACC, Stat.EVA)
STAGE_NEUTRAL = 6
MOVE_STATUS_MISSED = 1 << 0  # include/constants/battle/moves.h
MOVE_STATUS_FAILED = 1 << 6  # include/constants/battle/moves.h
MOVE_STATUS_PROTECTED = 1 << 15  # include/constants/battle/moves.h
COMMAND_RESULT = 41  # BATTLE_CONTROL_RESULT
NO_ACTIONS = (-1, -1)  # the last turn: the trace ends before any choice

# FIELD_CONDITION_* (TEMP | PERM pairs) and FIELD_CONDITION_DEEP_FOG
_WEATHER_BITS = (
    (0x3, Weather.RAIN),
    (0xC, Weather.SANDSTORM),
    (0x30, Weather.SUN),
    (0xC0, Weather.HAIL),
    (1 << 15, Weather.FOG),
)

_MAX_CANDIDATES = 4096
_CHUNK = 16  # candidates replayed per jitted call (one compile)
_I8, _I32 = jnp.int8, jnp.int32


class TraceError(ValueError):
    """A trace that cannot be converted."""


class TraceMismatch(TraceError):
    """The trace contradicts the party file, the trainer data or the engine."""


class AmbiguousTrace(TraceError):
    """More than one realized outcome fits the trace; ``candidates`` lists them."""

    def __init__(self, message: str, candidates: list):
        super().__init__(f"{message}; candidates {candidates}")
        self.candidates = candidates


@dataclasses.dataclass(frozen=True)
class Turn:
    """One converted turn. ``roll_candidates[(side, hit)]`` lists every ``(crit, roll)``
    that fits that hit (the lowest is used); ``mismatched`` names the
    ``COMPARED_FIELDS`` on which the best replay still differs from the next record."""

    state: BattleState
    actions: tuple[int, int]
    outcome: Outcome
    roll_candidates: dict[tuple[int, int], list[tuple[int, int]]]
    mismatched: tuple[str, ...]


def decode_weather(mask: int) -> Weather:
    """``ids.Weather`` from ``fieldConditionsMask``."""
    for bits, weather in _WEATHER_BITS:
        if mask & bits:
            return weather
    return Weather.NONE


def resolve_roll(candidates: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The ``(crit, roll)`` used for a hit that ``candidates`` all fit. They all give
    the same damage, so the lowest is used. A crit is an observable game event,
    however, so candidates that disagree on it raise ``AmbiguousTrace``."""
    candidates = list(candidates)
    if len({crit for crit, _ in candidates}) > 1:
        raise AmbiguousTrace("both a crit and a non-crit roll fit", candidates)
    return min(candidates)


def initial_state(
    tables: Tables,
    player: list[PokemonSpec],
    trainer: Trainer,
    leads: tuple[int, int],
    game: str = VANILLA,
) -> BattleState:
    """The fight's turn-0 state: ``player`` against the trainer's built party (from
    ``game``'s intermediates), bag and start weather, with the given active slots."""
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import start_weather

    opp = build_trainer_party(trainer, IntermediateData(get_game(game).intermediate_dir))
    return make_state(
        tables, player, opp, start_weather(trainer), opp_items=trainer.items, leads=leads
    )


def _fields(record: dict) -> dict[str, np.ndarray]:
    """The ``COMPARED_FIELDS`` of a ``turn`` record, shaped and typed as in ``LAYOUT``."""
    a = {name: np.zeros(*LAYOUT[name]) for name in COMPARED_FIELDS}
    for side, data in enumerate(record["sides"]):
        if len(data["party"]) > SLOTS:
            raise TraceError(f"side {side}: {len(data['party'])} Pokemon in one party")
        for slot, mon in enumerate(data["party"]):
            for name in ("species", "level", "hp", "max_hp", "item"):
                a[name][side, slot] = mon[name]
            a["moves"][side, slot] = mon["moves"]
            a["pp"][side, slot] = mon["pp"]
            a["status"][side, slot], a["status_counter"][side, slot] = decode_status(mon["status"])
        a["active"][side] = data["active"]
        a["stages"][side] = _stage_row(data["stages"])
    a["weather"][...] = decode_weather(record["field"])
    return a


def _stage_row(raw: Sequence[int]) -> np.ndarray:
    row = np.zeros(LAYOUT["stages"][0][1], LAYOUT["stages"][1])
    for i, stat in enumerate(STAGE_ORDER):
        if stat != Stat.HP:
            row[stat] = raw[i] - STAGE_NEUTRAL
    return row


def _overlay(state: BattleState, fields: dict[str, np.ndarray]) -> BattleState:
    return state._replace(**{name: jnp.asarray(value) for name, value in fields.items()})


def _check_start(state: BattleState, fields: dict[str, np.ndarray]) -> None:
    for name in _START_FIELDS:
        built, traced = np.asarray(getattr(state, name)), fields[name]
        diff = np.argwhere(built != traced)
        if diff.size:
            side, slot = (int(i) for i in diff[0][:2])
            who = "the party file" if side == 0 else "the trainer data"
            raise TraceMismatch(
                f"turn 0: {name} of side {side} slot {slot} is {traced[side, slot].tolist()} "
                f"in the trace but {built[side, slot].tolist()} in {who}"
            )


def _hit_state(state: BattleState, record: dict) -> BattleState:
    """The turn's start state with a ``hit`` record's active slots, HP, status and
    stages written over it."""
    active = np.asarray(record["active"], LAYOUT["active"][1])
    hp, status = np.asarray(state.hp).copy(), np.asarray(state.status).copy()
    counter, stages = np.asarray(state.status_counter).copy(), np.asarray(state.stages).copy()
    for b in (0, 1):
        hp[b, active[b]] = record["hp"][b]
        status[b, active[b]], counter[b, active[b]] = decode_status(record["status"][b])
        stages[b] = _stage_row(record["stages"][b])
    return state._replace(
        active=jnp.asarray(active),
        hp=jnp.asarray(hp),
        status=jnp.asarray(status),
        status_counter=jnp.asarray(counter),
        stages=jnp.asarray(stages),
    )


@jax.jit
def _damage_rows(tables, state, side, move, crit, power_mul, present=PRESENT_TABLE):
    """The 16 rolls' damage; ``present`` is Present's drawn power (``Outcome.present``)."""
    rolls = jnp.arange(16, dtype=_I32)
    return jax.vmap(
        lambda r: damage_for_roll(
            tables, state, side, move, crit, r, power_mul=power_mul, present=present
        )
    )(rolls)


@jax.jit
def _confusion_rows(tables, state, side):
    return jax.vmap(lambda r: confusion_damage(tables, state, side, r))(jnp.arange(16))


@jax.jit
def _replay(tables, state, actions, outcomes, slot):
    """``step`` for each candidate, then the opponent's replacement into ``slot``."""

    def one(a, o):
        s = step(tables, state, a, o)
        r = replace(tables, s, 1, slot)
        need = needs_replacement(s)[1] != 0
        return jax.tree.map(lambda x, y: jnp.where(need, x, y), r, s)

    return jax.vmap(one)(actions, outcomes)


class _Turn:
    """Inference for one turn: ``state`` before it, ``after`` the next record's fields,
    ``events`` the records in between."""

    def __init__(self, tables, state, after, events, index):
        self.tables, self.state, self.after, self.index = tables, state, after, index
        self.events = events
        self.candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def fail(self, message: str) -> TraceMismatch:
        return TraceMismatch(f"turn {self.index}: {message}")

    def action_choices(self, side: int) -> list[int]:
        state, after = self.state, self.after
        mon = int(state.active[side])
        drop = np.asarray(state.pp[side, mon], np.int32) - after["pp"][side, mon]
        used = np.flatnonzero(drop > 0)
        if used.size > 1:
            raise self.fail(f"side {side}: PP dropped in move slots {used.tolist()}")
        if used.size == 1:
            return [int(used[0])]
        if int(after["active"][side]) != mon and int(after["hp"][side, mon]) > 0:
            return [MOVE_ACTIONS + int(after["active"][side])]
        # It did not get to use a move: every legal move, and the trainer's bag.
        legal = np.asarray(legal_actions(self.tables, state, side))[:MOVE_ACTIONS]
        choices = [int(a) for a in np.flatnonzero(legal)]
        if side == 1:
            items = np.asarray(usable_trainer_items(self.tables, state, 1))
            choices += [FIRST_ITEM_ACTION + int(k) for k in np.flatnonzero(items)]
        return choices

    def fit(self, side: int, hit: int, record: dict, table: dict[int, list[int]]):
        """The ``(crit, roll)`` of a hit whose damage by roll is ``table[crit]``."""
        cands = [
            (crit, r)
            for crit, damages in table.items()
            for r, damage in enumerate(damages)
            if damage == record["damage"]
        ]
        if not cands:
            raise self.fail(
                f"side {side} hit {hit} with move {record['move']} did damage "
                f"{record['damage']}, which fits no roll of the engine's table {table}"
            )
        self.candidates[(side, hit)] = cands
        try:
            return resolve_roll(cands)
        except AmbiguousTrace as err:
            raise AmbiguousTrace(
                f"turn {self.index}: side {side} hit {hit}: both a crit and a non-crit roll fit",
                err.candidates,
            ) from None

    def _protect_roll(self, o: dict, side: int, action: int) -> None:
        """``o["protect"][side]`` for a Protect / Detect action: BtlCmd_TryProtection
        (battle_script.c:5167-5186) sets MOVE_STATUS_FAILED on a failure, which the move's
        ``move_end`` record keeps; a success does not (1 replays it; a chain of 0 or a user
        acting last ignores the roll). tools/trace_battle.lua writes a ``move_end`` at every
        MOVE_END with the context's ``attacker`` and ``moveCur``, which a switch or a bag item
        leaves stale, so only records naming this side and this move count. A side that never
        reached the move (flinched, fully paralyzed) has none: the roll stays unread. Several
        that disagree on the flag raise."""
        move = int(selected_move(self.tables, self.state, side, action)[0])
        if int(self.tables.move_effect[move]) != EffectId.PROTECT:
            return
        ends = [
            e
            for e in self.events
            if e["kind"] == "move_end" and e["attacker"] == side and e.get("move") == move
        ]
        failed = {bool(e["flags"] & MOVE_STATUS_FAILED) for e in ends}
        if len(failed) > 1:
            raise self.fail(
                f"side {side}'s Protect matches {len(ends)} move_end records that disagree on "
                "MOVE_STATUS_FAILED"
            )
        if failed:
            o["protect"][side] = 0 if failed.pop() else 1

    def damage_outcome(self, actions: list[list[int]]) -> dict[str, np.ndarray]:
        """``empty_outcome`` with the hits, crits and rolls of the trace's records."""
        o = {name: np.asarray(v).copy() for name, v in empty_outcome()._asdict().items()}
        for side in (0, 1):
            hits = [
                e
                for e in self.events
                if e["kind"] == "hit" and e["attacker"] == side and e["defender"] == 1 - side
            ]
            self_hits = [
                e
                for e in self.events
                if e["kind"] == "hit" and e["attacker"] == side and e["defender"] == side
            ]
            missed = any(
                e["kind"] == "move_end"
                and e["attacker"] == side
                and e["flags"] & MOVE_STATUS_MISSED
                for e in self.events
            )
            known_move = len(actions[side]) == 1 and actions[side][0] < MOVE_ACTIONS
            if (hits or missed) and not known_move:
                raise self.fail(f"side {side} has move records but used no PP")
            if missed:
                o["hit"][side] = 0
            if known_move:
                self._protect_roll(o, side, actions[side][0])
            if hits:
                move = int(selected_move(self.tables, self.state, side, actions[side][0])[0])
                o["n_hits"][side] = min(max(len(hits), 2), MAX_HITS)
            for i, e in enumerate(hits[:MAX_HITS]):
                if e["move"] != move:
                    raise self.fail(f"side {side} hit {i} is move {e['move']}, chose {move}")
                hs = _hit_state(self.state, e)
                other = actions[1 - side]
                pursuit = (
                    int(self.tables.move_effect[move]) == EffectId.PURSUIT
                    and len(other) == 1
                    and MOVE_ACTIONS <= other[0] < FIRST_ITEM_ACTION
                    and e["active"][1 - side] == int(self.state.active[1 - side])
                )
                power_mul = 20 if pursuit else 10
                crits = (0, 1) if e["crit"] is None else (int(e["crit"]),)
                # Present: a hit record is its damage path; its power 40/80/120 is drawn
                # (Outcome.present 0-2), so the lowest power with a fitting roll is used
                # (equal damage gives the same state). Its heal leaves no hit record
                # (``variants``).
                present = int(self.tables.move_effect[move]) == EffectId.PRESENT
                for k in range(PRESENT_HEAL) if present else (PRESENT_TABLE,):
                    table = {
                        crit: np.asarray(
                            _damage_rows(self.tables, hs, side, move, crit != 0, power_mul, k)
                        ).tolist()
                        for crit in crits
                    }
                    fits = any(e["damage"] in rows for rows in table.values())
                    if fits or k in (PRESENT_TABLE, PRESENT_HEAL - 1):
                        break
                if present:
                    o["present"][side] = k
                o["crit"][side, i], o["roll"][side, i] = self.fit(side, i, e, table)
            if self_hits:
                hs = _hit_state(self.state, self_hits[0])
                o["confusion_hit"][side] = 1
                table = {0: np.asarray(_confusion_rows(self.tables, hs, side)).tolist()}
                _, o["roll"][side, 0] = self.fit(side, 0, self_hits[0], table)
        return o

    def variants(self, actions: tuple[int, int], base: dict) -> list[dict]:
        """Every value of the chance fields the trace does not record that can matter
        in this state with these actions (docs/tracing.md)."""
        tables, state = self.tables, self.state
        options: list[tuple[str, tuple, list]] = []  # (field, index, values)
        if _tied(tables, state):
            options.append(("speed_tie", (), [0, 1]))
        if any(_tied(tables, s) for s in self._end_of_turn_states(actions)):
            options.append(("end_speed_tie", (), [0, 1]))
        connected = [bool(base["hit"][s, 0]) and self._acted(s) for s in (0, 1)]
        fixed: list[tuple[str, tuple, int]] = []
        for side in (0, 1):
            statuses = self._statuses(side)
            if Status.PARALYSIS in statuses:
                options.append(("full_para", (side,), [0, 1]))
            if Status.FREEZE in statuses:
                options.append(("thaw", (side,), [0, 1]))
            if self._may_be_in_love(side, actions):
                options.append(("infatuation", (side,), [0, 1]))
            if actions[side] >= MOVE_ACTIONS or not connected[side]:
                continue
            move = int(selected_move(tables, state, side, actions[side])[0])
            effect = int(tables.move_effect[move])
            chance = int(tables.move_effect_chance[move])
            per_hit = (side, slice(None))
            if chance == 0:
                fixed.append(("secondary", per_hit, 1))  # applies on every hit
            else:
                options.append(("secondary", per_hit, [0, 1]))
            hit_record = any(
                e["kind"] == "hit" and e["attacker"] == side and e["defender"] == 1 - side
                for e in self.events
            )
            if effect == EffectId.PRESENT and not hit_record:
                # Present connected with no hit record: its heal (no UPDATE_HP damage), or a
                # damage path into an immunity; the replay matches the target's HP gain.
                options.append(("present", (side,), [0, PRESENT_HEAL]))
            if effect == EffectId.FANG_PARALYZE_FLINCH:
                # Thunder Fang's second roll (Outcome.fang_flinch): a flinch shows only as the
                # target's missing move.
                options.append(("fang_flinch", (side,), [0, 1]))
            # The contact abilities with a roll (effects._contact_ability) of the defender:
            # the target's start-of-turn active (Pursuit hits the outgoing one) or the
            # Pokemon it switches in.
            target = 1 - side
            defenders = [int(state.active[target])]
            if MOVE_ACTIONS <= actions[target] < FIRST_ITEM_ACTION:
                defenders.append(actions[target] - MOVE_ACTIONS)
            rolled = (
                AbilityId.STATIC,
                AbilityId.POISON_POINT,
                AbilityId.FLAME_BODY,
                AbilityId.CUTE_CHARM,
            )
            if any(int(state.ability[target, mon]) in rolled for mon in defenders):
                options.append(("contact_proc", (target, slice(None)), [0, 1]))
            sleep = (
                effect in (EffectId.STATUS_FOE, EffectId.SECONDARY_STATUS)
                and int(tables.move_param[move, 0]) == Status.SLEEP
            )
            if effect == EffectId.TAUNT:
                options.append(("duration", (side,), [3, 4, 5]))
            elif effect == EffectId.TRAP:
                options.append(("duration", (side,), [3, 4, 5, 6]))
            elif sleep or effect == EffectId.CONFUSE_FOE:
                options.append(("duration", (side,), [2, 3, 4, 5]))
        out = []
        for values in itertools.product(*(vals for _, _, vals in options)):
            o = {k: v.copy() for k, v in base.items()}
            for name, index, value in fixed:
                o[name][index] = value
            for (name, index, _), value in zip(options, values):
                o[name][index] = value
            out.append(o)
        return out

    def _statuses(self, side: int) -> set[int]:
        """The statuses the side's start-of-turn active mon has at any point the trace
        shows: at the start, in a ``hit`` record and in the next record. A status
        inflicted before the mon moves (Stun Spore, Static) needs its ``full_para`` or
        ``thaw`` draw on the same turn."""
        mon = int(self.state.active[side])
        statuses = {int(self.state.status[side, mon]), int(self.after["status"][side, mon])}
        for e in self.events:
            if e["kind"] == "hit" and e["active"][side] == mon:
                statuses.add(int(decode_status(e["status"][side])[0]))
        return statuses

    def _may_be_in_love(self, side: int, actions: tuple[int, int]) -> bool:
        """The side uses a move and its attract check can run (CHECK_STATUS_STATE_ATTRACT,
        battle_controller_player.c:2669-2692): it is in love at the start (``infatuated``,
        carried by the engine: the trace has no volatile bits) or its foe uses Attract."""
        if actions[side] >= MOVE_ACTIONS:
            return False
        if int(self.state.infatuated[side]) != 0:
            return True
        foe = 1 - side
        if actions[foe] >= MOVE_ACTIONS:
            return False
        move = int(selected_move(self.tables, self.state, foe, actions[foe])[0])
        return int(self.tables.move_effect[move]) == EffectId.ATTRACT

    def _end_of_turn_states(self, actions: tuple[int, int]) -> list[BattleState]:
        """States whose speeds the end-of-turn order may compare: the start state, with
        the switches made, and with the next record's status and stages over that."""
        active = np.asarray(self.state.active).copy()
        for side in (0, 1):
            if MOVE_ACTIONS <= actions[side] < FIRST_ITEM_ACTION:
                active[side] = actions[side] - MOVE_ACTIONS
        switched = self.state._replace(active=jnp.asarray(active))
        shown = {name: self.after[name] for name in ("status", "stages")}
        return [self.state, switched, _overlay(switched, shown)]

    def _acted(self, side: int) -> bool:
        return any(e["kind"] == "move_end" and e["attacker"] == side for e in self.events) or any(
            e["kind"] == "hit" and e["attacker"] == side for e in self.events
        )

    def infer(self) -> tuple[tuple[int, int], dict, BattleState, tuple[str, ...]]:
        choices = [self.action_choices(0), self.action_choices(1)]
        base = self.damage_outcome(choices)
        candidates = [
            (pair, o) for pair in itertools.product(*choices) for o in self.variants(pair, base)
        ]
        if len(candidates) > _MAX_CANDIDATES:
            raise TraceError(f"turn {self.index}: {len(candidates)} candidates to replay")
        best = None
        for start in range(0, len(candidates), _CHUNK):
            chunk = candidates[start : start + _CHUNK]
            padded = chunk + [chunk[0]] * (_CHUNK - len(chunk))
            actions = jnp.asarray([pair for pair, _ in padded], _I32)
            outcomes = Outcome(
                *(
                    jnp.asarray(np.stack([o[name] for _, o in padded]), _I8)
                    for name in Outcome._fields
                )
            )
            slot = jnp.asarray(int(self.after["active"][1]), _I32)
            result = _replay(self.tables, self.state, actions, outcomes, slot)
            wrong = np.zeros((_CHUNK, len(COMPARED_FIELDS)), bool)
            for f, name in enumerate(COMPARED_FIELDS):
                got = np.asarray(getattr(result, name))
                axes = tuple(range(1, got.ndim))
                wrong[:, f] = ~np.all(got == self.after[name][None], axis=axes)
            score = wrong.sum(axis=1)[: len(chunk)]
            k = int(np.argmin(score))
            if best is None or score[k] < best[0]:
                state = jax.tree.map(lambda x, k=k: x[k], result)
                names = tuple(n for n, w in zip(COMPARED_FIELDS, wrong[k]) if w)
                best = (int(score[k]), chunk[k], state, names)
            if best[0] == 0:
                break
        _, (pair, outcome), state, mismatched = best
        return (int(pair[0]), int(pair[1])), outcome, state, mismatched


def _tied(tables: Tables, state: BattleState) -> bool:
    return int(effective_speed(tables, state, 0)) == int(effective_speed(tables, state, 1))


def _records(bundle: dict) -> tuple[list[dict], list[list[dict]]]:
    turns, events = [], []
    for record in bundle["records"]:
        if record["kind"] == "turn":
            turns.append(record)
            events.append([])
        elif not turns:
            raise TraceError(f"a {record['kind']!r} record comes before the first turn record")
        elif record["kind"] in ("hit", "move_end"):
            events[-1].append(record)
        else:
            raise TraceError(f"unknown record kind {record['kind']!r}")
    if not turns:
        raise TraceError("the trace has no turn record")
    return turns, events


def _spec(d: dict) -> PokemonSpec:
    return PokemonSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _trainer(trainer_id: int, game: str = VANILLA) -> Trainer:
    from nuzline.cli import resolve_trainer
    from nuzline.data.trainer import load_trainers

    trainers = load_trainers(get_game(game).intermediate_dir)
    return resolve_trainer(trainers, trainer_id=trainer_id)


def convert_bundle(bundle: dict, tables: Tables, names: Names) -> list[Turn]:
    """Every turn of a bundle (module docstring). The last ``Turn`` is the final record,
    with actions ``NO_ACTIONS``. The trace stops at the first player faint, which the
    engine treats as terminal."""
    del names  # ids throughout; kept for the plan's signature
    if bundle.get("format") != BUNDLE_FORMAT:
        raise TraceError(f"not a {BUNDLE_FORMAT} bundle")
    game = bundle.get("game", VANILLA)
    trainer = _trainer(bundle["trainer_id"], game)
    player = [_spec(d) for d in bundle["party"]]
    records, events = _records(bundle)
    first = _fields(records[0])
    leads = (int(first["active"][0]), int(first["active"][1]))
    state = initial_state(tables, player, trainer, leads, game)
    _check_start(state, first)
    state = _overlay(state, first)
    turns: list[Turn] = []
    for i in range(len(records) - 1):
        after = _fields(records[i + 1])
        inference = _Turn(tables, state, after, events[i], i)
        actions, outcome, replayed, mismatched = inference.infer()
        outcome = Outcome(*(jnp.asarray(outcome[name], _I8) for name in Outcome._fields))
        turns.append(Turn(state, actions, outcome, inference.candidates, mismatched))
        fainted = (np.asarray(turns[-1].state.hp[0]) > 0) & (after["hp"][0] == 0)
        state = _overlay(replayed, after)
        if int(state.over) or np.any(fainted):
            break
    turns.append(Turn(state, NO_ACTIONS, empty_outcome(), {}, ()))
    return turns


def convert(path: Path, tables: Tables, names: Names) -> list[Turn]:
    """``convert_bundle`` of the bundle at ``path``."""
    return convert_bundle(json.loads(Path(path).read_text(encoding="utf-8")), tables, names)


def trace_to_states(
    path: Path, tables: Tables, names: Names
) -> list[tuple[BattleState, tuple[int, int], Outcome]]:
    """``(state, actions, outcome)`` per turn of the bundle at ``path``; the last entry
    is the final state with ``NO_ACTIONS`` and an empty outcome."""
    return [(t.state, t.actions, t.outcome) for t in convert(path, tables, names)]


def bundle_trace(
    raw: Path, party: Path | SaveFile, names: Names, trainer_id: int, game: str = VANILLA
) -> dict:
    """A raw trace plus the player's party, the trainer id and the game. ``party`` is a
    party TOML or a loaded save, whose party becomes the specs (``SaveMon.to_spec`` with
    ``game``'s tables); an egg there is left out with a note on stderr, as
    ``nuzline fight --save`` does (an egg never enters a battle)."""
    from nuzline.data.party import load_party

    game = get_game(game).id
    if isinstance(party, SaveFile):
        from nuzline.data.games import load_game

        tables, _ = load_game(get_game(game))
        specs = []
        for slot, mon in enumerate(party.party):
            if mon.is_egg:
                print(f"party slot {slot}: egg left out", file=sys.stderr)
            else:
                specs.append(mon.to_spec(tables))
    else:
        specs = load_party(party, names)
    lines = Path(raw).read_text(encoding="utf-8").splitlines()
    return {
        "format": BUNDLE_FORMAT,
        "game": game,
        "trainer_id": trainer_id,
        "party": [dataclasses.asdict(m) for m in specs],
        "records": [json.loads(line) for line in lines if line.strip()],
    }


def main(argv: list[str] | None = None) -> int:
    from nuzline.cli import resolve_trainer
    from nuzline.data.games import load_game
    from nuzline.data.trainer import load_trainers
    from nuzline.saves.save import load_save

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True, type=Path, help="raw trace (JSON lines)")
    parser.add_argument("--out", required=True, type=Path, help="bundle to write")
    mine = parser.add_mutually_exclusive_group(required=True)
    mine.add_argument("--party", type=Path, help="the player's party TOML")
    mine.add_argument("--save", type=Path, help="a save (.sav/.dsv) whose party is used")
    parser.add_argument("--game", default="vanilla", help="vanilla (default) or kaizo")
    who = parser.add_mutually_exclusive_group()
    who.add_argument("--trainer", default="roark", help="trainer name (a Leader is preferred)")
    who.add_argument("--trainer-id", type=int, help="exact trainer id")
    args = parser.parse_args(argv)

    game = get_game(args.game)
    trainer = resolve_trainer(
        load_trainers(game.intermediate_dir), name=args.trainer, trainer_id=args.trainer_id
    )
    tables, names = load_game(game)
    party = args.party if args.save is None else load_save(args.save, game=game.id)
    bundle = bundle_trace(args.trace, party, names, trainer.id, game.id)
    try:
        turns = convert_bundle(bundle, tables, names)
    except TraceError as err:
        print(f"trace_to_state: error: {err}", file=sys.stderr)
        return 1
    for i, t in enumerate(turns[:-1]):
        rolls = {f"{s}.{h}": c for (s, h), c in t.roll_candidates.items()}
        print(f"turn {i}: actions {t.actions} rolls {rolls} mismatched {list(t.mismatched)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(turns)} turn records, trainer {trainer.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
