"""Map PKCalc trainer labels (and spreadsheet split headers) to Kaizo ROM trainer ids.

Neither source carries a trainer id, so a trainer is matched by its party: the
``(species, form, level)`` of every member in party order (``(species, level)`` for the
spreadsheet, which gives no forms). A label that no rule settles stays ``unmapped``;
nothing is guessed.

PKCalc labels (:func:`map_trainers`), in order:

1. ``override``: ``overrides[label]`` is a ROM trainer id (applied first).
2. ``exact``: exactly one ROM trainer has the party signature.
3. ``pid``: several do; keep those whose vanilla party PID (``TrainerData_BuildParty``:
   difficulty = the low byte, mons with a non-zero ``ability_override`` high byte left
   out) reproduces the most PKCalc natures and genders; one left.
4. ``name``: still several; the one whose ROM name occurs in the label as a whole word.

When the strict signature has no ROM trainer, the same steps run again on
``(species, level)`` alone; a trainer settled that way has ``how="form"`` and
``form_diffs`` names every slot where PKCalc's form differs from the ROM's (a PKCalc-vs-ROM
field conflict for Task 15 to report).

Spreadsheet headers (:func:`map_sheet_headers`) use steps 1, 2 and 4 only, never the PID
tie-break (the sheet has no natures), and are labelled by the header's ``cite``. A label
that loses an id to an override is ``unmapped`` with ``lost_to`` set to that id.
:func:`check_override_keys` rejects override keys that are neither label nor cite.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from nuzline.data.trainer import GENDER_FEMALE, GENDER_MALE, GENDER_NONE, gender_of, trainer_pid

# Nature names in index order (Pokemon_GetNatureOf: PID % 25).
NATURES = (
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty", "Bold", "Docile", "Relaxed", "Impish",
    "Lax", "Timid", "Hasty", "Serious", "Jolly", "Naive", "Modest", "Mild", "Quiet", "Bashful",
    "Rash", "Calm", "Gentle", "Sassy", "Careful", "Quirky",
)  # fmt: skip
_NATURE_INDEX = {n: i for i, n in enumerate(NATURES)}
_GENDER_NAMES = {"Male": GENDER_MALE, "Female": GENDER_FEMALE}


class MappingError(ValueError):
    """Two labels claim one ROM trainer id and no override settles it."""


@dataclass(frozen=True)
class Mapping:
    """``how``: ``override``, ``exact``, ``pid``, ``name``, ``form`` or ``unmapped``.
    ``candidates`` are the ROM ids sharing the label's party signature, before any
    tie-break. ``form_diffs`` (``how="form"``) is ``(slot, pkcalc_form, rom_form)`` per
    differing slot. ``lost_to`` is the id an override took from this label."""

    label: str
    id: int | None
    how: str
    candidates: tuple[int, ...]
    form_diffs: tuple[tuple[int, int, int], ...] = ()
    lost_to: int | None = None


def _pid_score(label_party: list[dict], trainer: dict, species: list[dict]) -> int:
    """How many PKCalc natures and genders the ROM trainer's vanilla PIDs reproduce."""
    female_class = trainer["class_gender"] == GENDER_FEMALE
    score = 0
    for want, mon in zip(label_party, trainer["party"], strict=True):
        if mon["ability_override"] != 0:
            continue
        pid = trainer_pid(
            mon["difficulty"],
            mon["level"],
            mon["species"],
            trainer["id"],
            trainer["class_id"],
            female_class=female_class,
        )
        score += _NATURE_INDEX.get(want.get("nature")) == pid % 25
        got = gender_of(species[mon["species"]]["gender_ratio"], pid)
        score += _GENDER_NAMES.get(want.get("gender"), GENDER_NONE) == got
    return score


def _settle(
    label: str,
    text: str,
    candidates: list[dict],
    party: list[dict] | None,
    species: list[dict],
) -> Mapping:
    """Steps 2-4 for one label; ``party`` None skips the PID step."""
    ids = tuple(t["id"] for t in candidates)
    if not candidates:
        return Mapping(label, None, "unmapped", ())
    if len(candidates) == 1:
        return Mapping(label, ids[0], "exact", ids)
    if party is not None:
        scores = [_pid_score(party, t, species) for t in candidates]
        best = max(scores)
        kept = [t for t, s in zip(candidates, scores, strict=True) if s == best]
        if len(kept) == 1:
            return Mapping(label, kept[0]["id"], "pid", ids)
        candidates = kept
    named = [
        t
        for t in candidates
        if t["name"]
        and re.search("(?<![A-Za-z0-9_])" + re.escape(t["name"]) + "(?![A-Za-z0-9_])", text)
    ]
    if len(named) == 1:
        return Mapping(label, named[0]["id"], "name", ids)
    return Mapping(label, None, "unmapped", ids)


def _resolve_claims(
    maps: list[Mapping], same_trainer: dict[str, object] | None = None
) -> list[Mapping]:
    """Two labels on one id raise ``MappingError``. An override on that id wins and the
    others become unmapped. Labels with an equal ``same_trainer`` key are one trainer
    listed twice, not a conflict."""
    claims: dict[int, list[Mapping]] = defaultdict(list)
    for m in maps:
        if m.id is not None:
            claims[m.id].append(m)
    demote: dict[str, int] = {}
    for tid, group in claims.items():
        if len(group) < 2:
            continue
        if same_trainer is not None and len({same_trainer[m.label] for m in group}) == 1:
            continue
        winners = [m for m in group if m.how == "override"]
        if len(winners) == 1:
            demote.update({m.label: tid for m in group if m is not winners[0]})
            continue
        labels = ", ".join(repr(m.label) for m in group)
        raise MappingError(f"trainer id {tid} is claimed by {len(group)} labels: {labels}")
    return [
        Mapping(m.label, None, "unmapped", m.candidates, lost_to=demote[m.label])
        if m.label in demote
        else m
        for m in maps
    ]


def _check_overrides(overrides: dict[str, int], labels: list[str], known: set[int]) -> None:
    for label in labels:
        if label in overrides and overrides[label] not in known:
            raise MappingError(f"override {label!r} names {overrides[label]}, not a ROM trainer id")


def check_override_keys(
    overrides: dict[str, int], labels: Iterable[str], cites: Iterable[str]
) -> None:
    """Raise ``MappingError`` naming every override key that is neither a PKCalc label
    nor a spreadsheet header cite (a typo would otherwise be ignored silently)."""
    known = set(labels) | set(cites)
    bad = sorted(k for k in overrides if k not in known)
    if bad:
        raise MappingError(f"override keys match no PKCalc label or sheet cite: {bad}")


def map_trainers(
    pkcalc_trainers: dict[str, dict],
    rom_trainers: list[dict],
    species: list[dict],
    overrides: dict[str, int],
) -> list[Mapping]:
    """One ``Mapping`` per non-template PKCalc label, in label order. A label without a
    party is unmapped."""
    known = {t["id"] for t in rom_trainers}
    by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for t in rom_trainers:
        if t["party"]:
            by_sig[tuple((m["species"], m["form"], m["level"]) for m in t["party"])].append(t)
    by_blind: dict[tuple, list[dict]] = defaultdict(list)
    for t in rom_trainers:
        if t["party"]:
            by_blind[tuple((m["species"], m["level"]) for m in t["party"])].append(t)
    labels = [k for k, v in pkcalc_trainers.items() if not v.get("template")]
    _check_overrides(overrides, labels, known)
    maps = []
    for label in labels:
        if label in overrides:
            maps.append(Mapping(label, overrides[label], "override", (overrides[label],)))
            continue
        party = pkcalc_trainers[label].get("party") or []
        if not party:
            maps.append(Mapping(label, None, "unmapped", ()))
            continue
        sig = tuple((m["species"], m["form"], m["level"]) for m in party)
        if sig in by_sig:
            maps.append(_settle(label, label, by_sig[sig], party, species))
            continue
        blind = by_blind.get(tuple((s, lv) for s, _f, lv in sig), [])
        m = _settle(label, label, blind, party, species)
        if m.id is not None:
            rom = next(t for t in blind if t["id"] == m.id)
            diffs = tuple(
                (i, want["form"], have["form"])
                for i, (want, have) in enumerate(zip(party, rom["party"], strict=True))
                if want["form"] != have["form"]
            )
            m = Mapping(label, m.id, "form", m.candidates, diffs)
        maps.append(m)
    return _resolve_claims(maps)


def _norm(name: str) -> str:
    """Species names compare case-, punctuation- and accent-blind; the Nidoran signs
    become ``f`` / ``m`` (``NIDORAN♀`` and ``Nidoran-F`` agree)."""
    return re.sub(r"[^a-z0-9]", "", name.replace("♀", "f").replace("♂", "m").casefold())


def map_sheet_headers(
    docs: dict,
    rom_trainers: list[dict],
    species: list[dict],
    overrides: dict[str, int],
) -> list[Mapping]:
    """One ``Mapping`` per spreadsheet split trainer header, labelled by its ``cite``
    (``"Roark Split:272"``): header text such as ``Barry #2`` repeats across splits.

    The signature is ``(species, level)`` in party order. Several ROM ids: the one whose
    name occurs in the header text (``name``), else unmapped. Overrides are keyed by cite.
    A form-suffixed species (``Rotom-Mow``) resolves by its base name when the full name is
    no ROM species; a name matching neither leaves the header unmapped. The same trainer
    (equal normalised name and party, e.g. Galactic Rosalind) listed in several places is
    no conflict; different headers on one id raise ``MappingError``."""
    known = {t["id"] for t in rom_trainers}
    ids_by_name: dict[str, set[int]] = defaultdict(set)
    for i, s in enumerate(species):
        ids_by_name[_norm(s["name"])].add(i)
    by_len: dict[int, list[dict]] = defaultdict(list)
    for t in rom_trainers:
        if t["party"]:
            by_len[len(t["party"])].append(t)

    def ids_of(raw: str) -> set[int]:
        """ROM species ids for a sheet species: the full name, else (a form suffix such
        as ``Rotom-Mow`` or ``Shellos -West``) the base before the first ``-``, space or
        ``/`` when that is a ROM species."""
        full = ids_by_name.get(_norm(raw))
        if full:
            return full
        return ids_by_name.get(_norm(re.split(r"[-\s/]", raw.strip(), maxsplit=1)[0]), set())

    headers = [
        h for split in docs["splits"].values() for sec in split["sections"] for h in sec["trainers"]
    ]
    _check_overrides(overrides, [h["cite"] for h in headers], known)

    maps: list[Mapping] = []
    same: dict[str, object] = {}
    for h in headers:
        label = h["cite"]
        sheet = [(frozenset(ids_of(m["species"])), m["level"]) for m in h["party"]]
        same[label] = (
            _norm(h["name"]),
            tuple((_norm(m["species"]), m["level"]) for m in h["party"]),
        )
        if label in overrides:
            maps.append(Mapping(label, overrides[label], "override", (overrides[label],)))
            continue
        candidates = [
            t
            for t in by_len.get(len(sheet), [])
            if all(
                m["species"] in ids and m["level"] == level
                for m, (ids, level) in zip(t["party"], sheet, strict=True)
            )
        ]
        maps.append(_settle(label, h["text"], candidates, None, species))
    return _resolve_claims(maps, same)
