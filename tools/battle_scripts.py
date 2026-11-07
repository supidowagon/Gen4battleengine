"""Battle structures defined by field scripts, shared by the decomp and ROM extractors.

Both extractors turn a map's script file into the same normalized ``Op`` list
(``extract_decomp._script_ops`` from the ``.s`` text, ``tools.rom.readers.maps.script_ops``
from ``scr_seq.narc`` bytes), and this module derives from it which trainers are fought
in two-trainer doubles, tag battles and beside a walking partner (plan Rulings 2-5):

* ``multi``: an enemy of a ``StartTagBattle``, or an enemy placed on a map whose script
  sets ``VAR_PARTNER_TRAINER_ID`` and runs ``SetHasPartner``: ``ScrCmd_StartTrainerBattle``
  adds the partner whenever ``SystemFlag_CheckHasPartner`` is set
  (``src/scrcmd_trainer.c:133-134``).
* ``double_two_trainer``: an operand of a ``StartTrainerBattle t1, t2`` with two trainer
  constants. Two trainers who spot the player in the same step
  (``res/field/scripts/scripts_battles.s:173``, ``StartTrainerBattle VAR_0x8004,
  VAR_0x8005``) depend on runtime geometry and are not recorded (Ruling 4).
* ``double``: ``StartTrainerBattle t, t``, one trainer in a double
  (``BATTLE_TYPE_TRAINER_DOUBLES``, ``src/encounter.c:736-737``); Kaizo uses it.
* Otherwise the header's format ("double" or "single") stands; that is the caller's.

Ops (``Op.name``): ``label``(name), ``StartTrainerBattle``(t1, t2),
``StartTagBattle``(partner, e1, e2), ``SetVar``(var, value), ``GoToIfEq``(var, value,
label), ``GoTo``(label), ``Call``(label), ``Return``, ``End``, ``GetPlayerGender``(var),
``GetPlayerStarterSpecies``(var), ``SetHasPartner``, and ``other`` for every other
command. Numbers are ids (trainers, species, genders, vars); labels are strings.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

from nuzline.data.trainer import PartnerVariant

# Turtwig, Chimchar, Piplup (generated/species.txt).
STARTERS = (387, 390, 393)
GENDERS = (0, 1)  # GENDER_MALE, GENDER_FEMALE (generated/genders.txt)
PARTNER_VAR = 0x403F  # VAR_PARTNER_TRAINER_ID (generated/vars_flags.txt, checked by var_ids)
# The operand ranges the SetVar / CompareVar macros treat as variables
# (asm/macros/scrcmd.inc:179,310): VARS_START..VARS_END and
# SCRIPT_LOCAL_VARS_START..SCRIPT_LOCAL_VARS_END, inclusive (checked by var_ids).
VAR_RANGES = ((0x4000, 0x4120), (0x8000, 0x800D))
# The format of a trainer whose battle structure the scripts leave to game state
# (battle_structures' ``refused``): not modelled, so coverage refuses it.
UNKNOWN_FORMAT = "unknown"
# A partner routine's evaluation stops after this many ops (none in vanilla needs 20).
_MAX_STEPS = 10_000


@dataclass(frozen=True)
class Op:
    name: str
    args: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class BattleStructure:
    format: str
    opponent_2: int | None
    partner: tuple[PartnerVariant, ...] | None  # None: not evaluated (``unknown_partner``)


def is_var(value: int | str) -> bool:
    """Whether ``value`` is a variable operand (``asm/macros/scrcmd.inc:310``)."""
    return isinstance(value, int) and any(lo <= value <= hi for lo, hi in VAR_RANGES)


def var_ids(decomp: Path) -> dict[str, int]:
    """Var/flag constant -> id from ``generated/vars_flags.txt`` (a metang enum,
    ``generated/meson.build:112``): a bare name is one more than the previous entry, and
    ``NAME = X`` sets it to the integer or earlier name ``X``, as a C enum counts.

    The ids this module hard-codes (``VAR_0x8004``, ``VAR_RESULT``, ``PARTNER_VAR`` and
    ``VAR_RANGES``) are checked against the file."""
    path = Path(decomp) / "generated" / "vars_flags.txt"
    out: dict[str, int] = {}
    value = -1
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z_]\w*)(?:\s*=\s*(\w+))?", line)
        if not m:
            raise ValueError(f"{path}:{lineno}: unparsed line {raw!r}")
        name, expr = m.groups()
        if expr is None:
            value += 1
        elif expr.isdecimal():
            value = int(expr)
        elif expr in out:
            value = out[expr]
        else:
            raise ValueError(f"{path}:{lineno}: {expr!r} is not defined earlier")
        out[name] = value
    want = {
        "VAR_0x8004": 0x8004,
        "VAR_RESULT": 0x800C,
        "VAR_PARTNER_TRAINER_ID": PARTNER_VAR,
        "VARS_START": VAR_RANGES[0][0],
        "VARS_END": VAR_RANGES[0][1],
        "SCRIPT_LOCAL_VARS_START": VAR_RANGES[1][0],
        "SCRIPT_LOCAL_VARS_END": VAR_RANGES[1][1],
    }
    for name, v in want.items():
        if out.get(name) != v:
            raise ValueError(f"{path}: {name} is {out.get(name)}, expected {v:#x}")
    return out


def _labels(ops: list[Op]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, op in enumerate(ops):
        if op.name == "label":
            if op.args[0] in out:
                raise ValueError(f"label {op.args[0]!r} is defined twice")
            out[str(op.args[0])] = i
    return out


def _run(ops: list[Op], labels: dict[str, int], label: str, var: int, gender: int, starter: int):
    """``var`` at the ``Return`` that ends the routine at ``label``, for one player."""
    if label not in labels:
        raise ValueError(f"partner routine {label!r}: no such label")
    env: dict[int, int] = {}
    pc, stack = labels[label], []
    seen = set()

    def jump(op: Op, target) -> int:
        if target not in labels:
            raise ValueError(f"partner routine {label!r}: {op} jumps to unknown label {target!r}")
        return labels[target]

    for _ in range(_MAX_STEPS):
        if pc >= len(ops):
            raise ValueError(f"partner routine {label!r} runs off the end of the script")
        state = (pc, tuple(stack), tuple(sorted(env.items())))
        if state in seen:
            raise ValueError(f"partner routine {label!r} has a loop at op {pc} ({ops[pc]})")
        seen.add(state)
        op = ops[pc]
        pc += 1
        if op.name == "label":
            continue
        if op.name == "SetVar":
            env[op.args[0]] = op.args[1]
        elif op.name == "GetPlayerGender":
            env[op.args[0]] = gender
        elif op.name == "GetPlayerStarterSpecies":
            env[op.args[0]] = starter
        elif op.name == "GoToIfEq":
            v, value, target = op.args
            if v not in env:
                raise ValueError(f"partner routine {label!r}: {op} reads an unset variable")
            if env[v] == value:
                pc = jump(op, target)
        elif op.name == "GoTo":
            pc = jump(op, op.args[0])
        elif op.name == "Call":
            stack.append(pc)
            pc = jump(op, op.args[0])
        elif op.name == "Return":
            if stack:
                pc = stack.pop()
                continue
            if var not in env:
                raise ValueError(f"partner routine {label!r} returns without setting {var:#x}")
            return env[var]
        else:
            raise ValueError(f"partner routine {label!r}: cannot evaluate {op}")
    raise ValueError(f"partner routine {label!r} has a loop (over {_MAX_STEPS} steps)")


def partner_variants(ops: list[Op], label: str, var: int) -> tuple[PartnerVariant, ...]:
    """The partner's teams that the routine at ``label`` leaves in ``var`` (Ruling 5).

    The routine is run for player gender 0 and 1 and starters ``STARTERS``, recording
    ``var`` at the ``Return`` that ends it. A dimension that never changes the result is
    recorded as None. An ``other`` op on an evaluated path, a loop, an ``End``, or a run
    that ends without setting ``var`` raises ``ValueError``. Variants are ordered by
    player gender, then starter.
    """
    labels = _labels(ops)
    got = {(g, s): _run(ops, labels, label, var, g, s) for g in GENDERS for s in STARTERS}
    by_gender = any(got[0, s] != got[1, s] for s in STARTERS)
    by_starter = any(got[g, s] != got[g, STARTERS[0]] for g in GENDERS for s in STARTERS)
    out: dict[PartnerVariant, None] = {}
    for g in GENDERS if by_gender else GENDERS[:1]:
        for s in sorted(STARTERS) if by_starter else STARTERS[:1]:
            out[PartnerVariant(got[g, s], s if by_starter else None, g if by_gender else None)] = (
                None
            )
    return tuple(out)


def _resolve(ops: list[Op], i: int, value: int, where: str) -> int:
    """A battle operand: a constant, or a var through the ``SetVar`` of it before op ``i``
    in the same label block (Ruling 3).

    Only straight-line ops that cannot write the var may sit between that ``SetVar`` and
    op ``i``: a ``SetVar``, ``GetPlayerGender`` or ``GetPlayerStarterSpecies`` of another
    var, ``SetHasPartner``, and a ``GoToIfEq`` to a known label (every branch target is a
    label, so its fall-through still holds the ``SetVar``). Anything else met first raises
    ``ValueError`` naming ``where`` (refuse, never approximate): an ``other`` (decomp
    ``SetVar V, VAR_X`` and ROM ``SetVarFromVar``/``CopyVar`` are ``other``), a ``Call``
    (its callee may write the var), a ``GoTo``, ``End`` or ``Return`` (the scan left the
    block), a write of the var, another battle, or a ``GoToIfEq`` to an unknown label. A
    ``label`` ends the scan, as does the start of the script."""
    if not is_var(value):
        return value
    labels = _labels(ops)
    for j in range(i - 1, -1, -1):
        op = ops[j]
        if op.name == "label":
            break
        if op.name == "SetVar":
            if op.args[0] == value:
                return int(op.args[1])
            continue
        if op.name in ("GetPlayerGender", "GetPlayerStarterSpecies") and op.args[0] != value:
            continue
        if op.name == "SetHasPartner":
            continue
        if op.name == "GoToIfEq" and op.args[2] in labels:
            continue
        raise ValueError(
            f"{where}: {ops[i]} has variable operand {value:#x}, and {op} before it may "
            f"change it or leave the block; refused rather than read a stale SetVar"
        )
    raise ValueError(f"{where}: {ops[i]} has variable operand {value:#x} with no SetVar before it")


def script_enemies(ops: list[Op], where: str = "script") -> set[int]:
    """The enemy operands of a script's ``StartTrainerBattle`` (both) and ``StartTagBattle``
    (operands 2-3), a variable resolved through the ``SetVar`` before it (Ruling 3)."""
    out = set()
    for i, op in enumerate(ops):
        if op.name == "StartTrainerBattle":
            out |= {_resolve(ops, i, t, where) for t in op.args}
        elif op.name == "StartTagBattle":
            out |= {_resolve(ops, i, t, where) for t in op.args[1:]}
    out.discard(0)
    return out


def has_partner(ops: list[Op]) -> bool:
    return any(op.name == "SetHasPartner" for op in ops)


def partner_ids(ops: list[Op]) -> set[int]:
    """The constants a script assigns to ``VAR_PARTNER_TRAINER_ID``."""
    return {int(op.args[1]) for op in ops if op.name == "SetVar" and op.args[0] == PARTNER_VAR}


def placed_enemies(
    ops: list[Op], objects: set[int], rematches: Mapping[int, list[int]], where: str = "script"
) -> set[int]:
    """A walking-partner map's placed enemies (Ruling 3): its trainer object events
    ``objects`` and its script's enemy operands (``script_enemies``), plus the Vs. Seeker
    rematches of those (``rematches``: base trainer -> rematch ids), less the partner ids
    the script sets."""
    base = set(objects) | script_enemies(ops, where)
    out = set(base)
    for t in base:
        out.update(rematches.get(t, ()))
    return out - partner_ids(ops)


def battle_structures(
    scripts: Mapping[Hashable, list[Op]],
    placed: Mapping[Hashable, set[int]],
    *,
    unknown_partner: bool = False,
    refused: Mapping[int, str] | None = None,
) -> dict[int, BattleStructure]:
    """Trainer id -> its battle structure, for the trainers fought in a two-trainer or
    multi battle (Rulings 2-3). ``scripts`` maps each map header to its script's ops in
    file order, ``placed`` each header whose script runs ``SetHasPartner`` to its placed
    enemies (``placed_enemies``; other headers' entries are not read).

    * A ``StartTrainerBattle t1, t2`` with two constant trainers pairs them.
    * A ``StartTagBattle partner, e1, e2`` pairs ``e1`` and ``e2`` beside the partner. A
      variable partner's teams come from the ``Call`` immediately before the battle
      (``partner_variants``); any other op there raises, unless ``unknown_partner`` (the
      Kaizo ROM reader, K2b Task 8), which keeps the pair with partner ``None``: present
      but not evaluated, never guessed.
    * On a header whose script runs ``SetHasPartner``, every placed enemy is fought beside
      the one trainer the script assigns to ``VAR_PARTNER_TRAINER_ID``. Placed enemies
      with no such trainer, or with several, raise (with several and ``unknown_partner``,
      the partner is ``None``).

    The same structure seen twice (a pair fought from two scripts, a script shared by
    several headers) is one structure. A trainer claimed by two different pairs, two
    different tag battles, two different partners, or a tag battle and anything else
    raises ``ValueError`` naming it; with ``unknown_partner``, tag or walking claims that
    agree on the other enemy but not on the partner (Kaizo's Lake Verity: a tag battle per
    player gender on a walking-partner map) give "multi" with partner ``None``. A pair claim
    beside a tag claim is a disagreement about the format, not the partner, and still
    raises. A trainer in ``refused`` (trainer id -> reason; the Kaizo ROM reader's
    ``data/kaizo/battle_formats.json``) whose claims conflict gets format ``"unknown"``,
    which coverage refuses, with its other enemy if all claims agree on one; an entry for a
    trainer whose claims do not conflict is stale and raises. A pair on a walking-partner
    map is "multi" with both the pair's other enemy and the partner.
    """
    refused = dict(refused or {})
    claims: dict[int, dict[tuple, set]] = {}

    def claim(t: int, kind: tuple, header) -> None:
        claims.setdefault(t, {}).setdefault(kind, set()).add(header)

    for header, ops in scripts.items():
        for i, op in enumerate(ops):
            if op.name == "StartTrainerBattle":
                if op.args[1] == 0:
                    continue  # one enemy
                # A variable operand resolves through the SetVar before it, else raises.
                # Ruling 4's spotting pair (scripts_battles.s:173) is a common script that
                # no map header runs, so it never reaches here.
                t1, t2 = (_resolve(ops, i, t, f"map {header}") for t in op.args)
                if t1 == t2:
                    # Encounter_NewVsTrainer: the same trainer twice is one trainer in a
                    # double, BATTLE_TYPE_TRAINER_DOUBLES (src/encounter.c:736-737).
                    claim(t1, ("self",), header)
                    continue
                claim(t1, ("pair", t2), header)
                claim(t2, ("pair", t1), header)
            elif op.name == "StartTagBattle":
                p, e1, e2 = op.args
                if is_var(e1) or is_var(e2):
                    raise ValueError(f"map {header}: {op} has a variable enemy")
                if is_var(p):
                    prev = ops[i - 1] if i else None
                    if prev is not None and prev.name == "Call":
                        team = partner_variants(ops, str(prev.args[0]), int(p))
                    elif unknown_partner:
                        team = None
                    else:
                        raise ValueError(
                            f"map {header}: {op} has a variable partner and no Call before it"
                        )
                else:
                    team = (PartnerVariant(int(p), None, None),)
                claim(e1, ("tag", e2, team), header)
                claim(e2, ("tag", e1, team), header)
        if not has_partner(ops):
            continue
        enemies = placed.get(header, set())
        if not enemies:
            continue
        partners = partner_ids(ops)
        if len(partners) == 1:
            team = (PartnerVariant(partners.pop(), None, None),)
        elif unknown_partner and partners:
            team = None
        else:
            raise ValueError(
                f"map {header}: runs SetHasPartner and places enemy trainers "
                f"{sorted(enemies)} but sets partner ids {sorted(partners)}, not one"
            )
        for t in enemies:
            claim(t, ("walk", team), header)

    out = {}
    for t, cs in sorted(claims.items()):
        pairs = [c for c in cs if c[0] == "pair"]
        tags = [c for c in cs if c[0] == "tag"]
        walks = [c for c in cs if c[0] == "walk"]
        selfs = [c for c in cs if c[0] == "self"]
        if selfs and len(cs) == 1:
            out[t] = BattleStructure("double", None, ())
            continue
        if (
            len(pairs) > 1
            or len(tags) > 1
            or len(walks) > 1
            or (tags and (pairs or walks))
            or selfs
        ):
            listed = "; ".join(f"{c} on maps {sorted(map(str, h))}" for c, h in cs.items())
            others = {c[1] for c in pairs + tags}
            if t in refused:
                refused.pop(t)
                out[t] = BattleStructure(
                    UNKNOWN_FORMAT, next(iter(others), None) if len(others) == 1 else None, None
                )
                continue
            if (
                unknown_partner
                and (tags or walks)
                and not (pairs and tags)
                and not selfs
                and len(others) <= 1
            ):
                # One enemy pair, several partner claims: multi, partner not evaluated.
                out[t] = BattleStructure("multi", next(iter(others), None), None)
                continue
            raise ValueError(f"trainer {t} is claimed by {len(cs)} battle structures: {listed}")
        if tags:
            out[t] = BattleStructure("multi", tags[0][1], tags[0][2])
        elif walks:
            out[t] = BattleStructure("multi", pairs[0][1] if pairs else None, walks[0][1])
        else:
            out[t] = BattleStructure("double_two_trainer", pairs[0][1], ())
    if refused:
        raise ValueError(f"refused formats for trainers without conflicting claims: {refused}")
    return out


def structure_fields(structure: BattleStructure | None, header_format: str) -> dict:
    """The ``trainers.json`` fields ``format``, ``opponent_2`` and ``partner`` of a trainer
    whose header format is ``header_format`` (Ruling 2's precedence). ``partner`` is null
    for a partner present but not evaluated (``battle_structures``' ``unknown_partner``)."""
    if structure is None:
        return {"format": header_format, "opponent_2": None, "partner": []}
    return {
        "format": structure.format,
        "opponent_2": structure.opponent_2,
        "partner": None
        if structure.partner is None
        else [
            {"trainer": p.trainer, "starter": p.starter, "player_gender": p.player_gender}
            for p in structure.partner
        ],
    }
