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
4. ``name``: still several; the one whose ROM name occurs in the label.

Spreadsheet headers (:func:`map_sheet_headers`) use steps 1, 2 and 4 only, never the PID
tie-break (the sheet has no natures), and are labelled by the header's ``cite``.
"""

from __future__ import annotations

import re
from collections import defaultdict
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
    """``how``: ``override``, ``exact``, ``pid``, ``name`` or ``unmapped``. ``candidates``
    are the ROM ids sharing the label's party signature, before any tie-break."""

    label: str
    id: int | None
    how: str
    candidates: tuple[int, ...]


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
    named = [t for t in candidates if t["name"] and t["name"] in text]
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
    demote: set[str] = set()
    for tid, group in claims.items():
        if len(group) < 2:
            continue
        if same_trainer is not None and len({same_trainer[m.label] for m in group}) == 1:
            continue
        winners = [m for m in group if m.how == "override"]
        if len(winners) == 1:
            demote.update(m.label for m in group if m is not winners[0])
            continue
        labels = ", ".join(repr(m.label) for m in group)
        raise MappingError(f"trainer id {tid} is claimed by {len(group)} labels: {labels}")
    return [
        Mapping(m.label, None, "unmapped", m.candidates) if m.label in demote else m for m in maps
    ]


def _check_overrides(overrides: dict[str, int], labels: list[str], known: set[int]) -> None:
    for label in labels:
        if label in overrides and overrides[label] not in known:
            raise MappingError(f"override {label!r} names {overrides[label]}, not a ROM trainer id")


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
        maps.append(_settle(label, label, by_sig.get(sig, []), party, species))
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
    A header whose species name matches no ROM species is unmapped. The same trainer
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

    headers = [
        h for split in docs["splits"].values() for sec in split["sections"] for h in sec["trainers"]
    ]
    _check_overrides(overrides, [h["cite"] for h in headers], known)

    maps: list[Mapping] = []
    same: dict[str, object] = {}
    for h in headers:
        label = h["cite"]
        sheet = [(_norm(m["species"]), m["level"]) for m in h["party"]]
        same[label] = (_norm(h["name"]), tuple(sheet))
        if label in overrides:
            maps.append(Mapping(label, overrides[label], "override", (overrides[label],)))
            continue
        candidates = [
            t
            for t in by_len.get(len(sheet), [])
            if all(
                m["species"] in ids_by_name.get(name, ()) and m["level"] == level
                for m, (name, level) in zip(t["party"], sheet, strict=True)
            )
        ]
        maps.append(_settle(label, h["text"], candidates, None, species))
    return _resolve_claims(maps, same)
