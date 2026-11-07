"""Cross-check rows for trainers, AI flags, battle formats and areas, and the vanilla PID
report (plan Task 13; spec 5.4, 5.8, 6.5; Rulings 6, 9-11, 19), fed to
``tools.kaizo.crosscheck.compare``.

Sources: the Kaizo ROM's ``trainers.json`` (K1 reader with K2a's ``format``,
``opponent_2`` and ``partner``; Task 8's ``difficulty`` / ``ability_override`` split),
the PKCalc label and sheet header mappings (Tasks 9-10, the ``Mappings`` bundle),
``pkcalc.json`` (Task 6), ``docs.json`` (Task 7: the split headers' tags and AI cells, and
the ``RAW TRAINER`` rows keyed by trainer id) and ``overrides.toml`` (Task 10). ``species``
is the ROM bundle of ``rows_tables.load_rom`` (``species``, ``learnsets``, ``items``).

Rows are table ``trainers``, one set per PKCalc label mapped to a ROM id: trainer fields
keyed ``"<id> <label>"``, party fields keyed ``"<id> <label>/<slot>"`` (slot 0-based, as
``Mapping.form_diffs``). Every row is executed. ``None`` = that source has no value.

- ``party_size``, and per mon ``species``, ``form``, ``level``, ``item`` (no item is 0 in
  every source), ``moves`` (the non-zero moves in slot order; a ROM trainer without moves
  gets ``default_moves`` from the base species' Kaizo level-up list, as
  ``TrainerData_BuildParty`` creates the mon with the base species), ``ivs`` (six
  ``difficulty_to_ivs`` values vs PKCalc's six); sheet = the ``RAW TRAINER`` row.
- ``difficulty`` and ``ability_override``: ROM vs ``RAW TRAINER`` (``ability_number``);
  PKCalc has neither, so a difference is ``rom_vs_sheet`` (Ruling 6).
- ``ability_slot`` where the override byte's high nibble forces a slot (1 = slot 1, 2 =
  slot 2): the ability id that slot gives (slot 2 of a species whose ability 2 is 0 gives
  ability 1) vs PKCalc's ability; sheet ``None``. Read from the patched
  ``TrainerData_BuildParty`` (arm9 0x020793B8; helper at 0x020795A0): the high nibble
  clears (1) or sets (2) bit 0 of the PID's gender modifier, and the unmodified
  ``BoxPokemon_InitWith`` (``src/pokemon.c:473-483``) picks the ability from that bit.
- ``ai_flags``: bits 0-10 (``AI_MASK``) of ROM and PKCalc vs the split's AI cells (the AI
  key's first eleven names in bit order; ``Left Side AI`` is no bit). A header without AI
  cells (``ai_known`` false) gives ``None``, never 0. Bits above 10 go to ``area_section``.
- ``format``, ``opponent_2``, ``partner_present`` (Ruling 10): ROM K2a fields vs PKCalc
  ``formats`` (``true_double`` -> ``double``; a ``double`` pair -> ``double_two_trainer``
  with the other label's id; a ``tag`` triple -> ``multi`` with the other label's id and a
  partner) vs the sheet: ``DOUBLE`` -> ``double`` without a partner; ``TAG BATTLE with X``
  (``tag_partner`` + ``multi_with``) -> ``multi`` with a partner; ``MULTI BATTLE with X``
  names the second opponent only (the sheet writes it for PKCalc's two-trainer doubles
  too, e.g. Cyclist Axel), so it sets ``opponent_2`` and no format. X is the header in the
  same section whose name ends with X's last word; no single such header -> ``"name: X"``.
  A ROM ``partner`` of ``null`` (a partner the rules cannot evaluate) counts as present.
  A label PKCalc lists only as a tag partner gets no PKCalc format values.
- ``player_starter`` (Ruling 11, preflight #28): the ``rom`` slot holds the derived value:
  the overrides ``fields`` entry, else the ROM partner variant's starter (K2a
  ``PartnerVariant``), else the sheet tag, else the PKCalc bracket. PKCalc's bracket names
  the rival's own starter: Barry's line maps per Ruling 11; Dawn's or Lucas's line maps to
  the third starter (Barry takes the one strong against the player's), which the ROM's
  partner variants confirm. ``area_section`` cites each derived value's source.
- ``area_weather``: ROM ``start_condition`` vs the sheet tag (the header's, else its
  section's) vs PKCalc's weather lists after dropping a weather a lead's, second lead's,
  other opponent's lead's or tag partner's lead's weather ability explains (spec 5.8).
- ``mapped``: an unmapped PKCalc label (ROM ``None``, PKCalc ``True``) and an unmapped sheet
  header (sheet ``True``), keyed by label or cite.

ROM ``start_condition`` / ``format`` ``unknown`` (controller ruling) make no rows; they are
listed in ``area_section`` under ``refused: unknown (user call)`` with every source.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from nuzline.data.trainer import (
    GENDER_FEMALE,
    GENDER_MALE,
    GENDER_NONE,
    default_moves,
    difficulty_to_ivs,
    gender_of,
    trainer_pid,
)
from tools.extract_kaizo_docs import AI_KEY, LEFT_SIDE
from tools.kaizo.crosscheck import Row
from tools.kaizo.overrides import GRAVITY_LABEL, Mappings, Overrides
from tools.kaizo.rows_tables import (
    _get,
    _name_index,
    _ref,
    _sheet_item_ids,
    _sheet_move_index,
)
from tools.kaizo.trainer_map import NATURES

TABLE = "trainers"
AI_BITS = 11  # AI_FLAG_BASIC .. AI_FLAG_HARASSMENT
AI_MASK = (1 << AI_BITS) - 1
SHEET_AI_BITS = {name: bit for bit, name in enumerate(AI_KEY[:AI_BITS])}
if LEFT_SIDE in SHEET_AI_BITS or len(SHEET_AI_BITS) != AI_BITS:
    raise ValueError("the sheet's AI key no longer lists the eleven AI flag bits first")

UNKNOWN = "unknown"  # ROM start_condition / format the extraction could not settle
WEATHER_ABILITIES = {"Drizzle": "rain", "Drought": "sun", "Sand Stream": "sandstorm",
                     "Snow Warning": "hail"}  # fmt: skip
PKCALC_WEATHERS = {"": None, "clear": None, "sun": "sun", "rain": "rain", "sand": "sandstorm",
                   "hail": "hail", "fog": "fog"}  # fmt: skip
SHEET_WEATHERS = ("rain", "sun", "sandstorm", "hail")

# Ruling 11: the bracket names Barry's starter; Barry takes the one strong against the
# player's, so his Chimchar line means the player has Turtwig (387).
TURTWIG, CHIMCHAR, PIPLUP = 387, 390, 393
_LINES = {
    TURTWIG: ("Turtwig", "Grotle", "Torterra"),
    CHIMCHAR: ("Chimchar", "Monferno", "Infernape"),
    PIPLUP: ("Piplup", "Prinplup", "Empoleon"),
}
_BARRY_HAS = {CHIMCHAR: TURTWIG, PIPLUP: CHIMCHAR, TURTWIG: PIPLUP}  # Barry's line -> player's
_RIVAL2_HAS = {PIPLUP: TURTWIG, TURTWIG: CHIMCHAR, CHIMCHAR: PIPLUP}  # Dawn/Lucas's -> player's
_RIVAL = re.compile(r"\b(Barry|Dawn|Lucas)\b.*\[([^\]{}]+)\]$")

# Headline trainers of the PID section (spec 5.4: "Kaizo Roark 6/6").
PID_HEADLINE = ("Leader Roark",)

_NATURE_INDEX = {n: i for i, n in enumerate(NATURES)}
_GENDERS = {"Male": GENDER_MALE, "Female": GENDER_FEMALE, None: GENDER_NONE}
_GENDER_NAMES = {GENDER_MALE: "Male", GENDER_FEMALE: "Female", GENDER_NONE: "none"}


# --- joins ------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def _joined(values: Iterable[Any]) -> Any:
    """The one value of ``values`` (``None`` ignored); several -> ``"a+b"``; none -> None."""
    got = sorted({v for v in values if v is not None}, key=str)
    if not got:
        return None
    if len(got) == 1:
        return got[0]
    return "+".join(map(str, got))


def _sheet_rows(docs: dict) -> dict[str, tuple[dict, dict]]:
    """cite -> (section, header) for every split trainer header."""
    return {
        h["cite"]: (sec, h)
        for split in docs["splits"].values()
        for sec in split["sections"]
        for h in sec["trainers"]
    }


def _sheet_by_id(docs: dict, mappings: Mappings) -> dict[int, list[tuple[dict, dict]]]:
    rows = _sheet_rows(docs)
    out: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for m in mappings.sheet:
        if m.id is not None:
            out[m.id].append(rows[m.label])
    return out


def _pk_formats(pkcalc: dict) -> tuple[dict[str, tuple[str, str | None, str | None]], set[str]]:
    """label -> (format, other opponent label, partner label); and the labels PKCalc lists
    only as tag partners. A label in two format lists raises."""
    formats = pkcalc.get("formats", {})
    out: dict[str, tuple[str, str | None, str | None]] = {}

    def put(label: str, value: tuple[str, str | None, str | None]) -> None:
        if out.setdefault(label, value) != value:
            raise ValueError(f"PKCalc formats list {label!r} twice: {out[label]} and {value}")

    for label in formats.get("true_double", []):
        put(label, ("double", None, None))
    for a, b in formats.get("double", []):
        put(a, ("double_two_trainer", b, None))
        put(b, ("double_two_trainer", a, None))
    for a, b, partner in formats.get("tag", []):
        put(a, ("multi", b, partner))
        put(b, ("multi", a, partner))
    partners = {t[2] for t in formats.get("tag", [])} - set(out)
    return out, partners


def _pk_starter(label: str) -> int | None:
    m = _RIVAL.search(label)
    if m is None:
        return None
    line = next((s for s, names in _LINES.items() if m[2] in names), None)
    if line is None:
        raise ValueError(f"PKCalc label {label!r}: bracket {m[2]!r} is no starter line")
    return (_BARRY_HAS if m[1] == "Barry" else _RIVAL2_HAS)[line]


def _partner_starters(rom: Sequence[dict]) -> dict[int, set[int]]:
    """ROM partner id -> the player starters its K2a partner variants are fought with."""
    out: dict[int, set[int]] = defaultdict(set)
    for t in rom:
        for p in t.get("partner") or []:
            if p.get("starter") is not None:
                out[p["trainer"]].add(p["starter"])
    return out


def _sheet_weather(sec: dict, h: dict) -> Any:
    def claims(tags: dict) -> set[str]:
        out = set()
        if tags.get("permanent") in SHEET_WEATHERS:
            out.add(tags["permanent"])
        if tags.get("weather"):
            out.add(tags["weather"])
        if tags.get("fog"):
            out.add("fog")
        return out

    return _joined(claims(h["tags"]) or claims(sec["tags"]))


def _sheet_ai(h: dict) -> int | None:
    if not h.get("ai_known"):
        return None
    return sum(1 << SHEET_AI_BITS[f] for f in h["ai_flags"] if f in SHEET_AI_BITS)


def _sheet_opponent(sec: dict, h: dict, sheet_ids: dict[str, int | None]) -> Any:
    x = h["tags"].get("multi_with")
    if x is None:
        return None
    last = _norm(x.split()[-1])
    hits = [o for o in sec["trainers"] if o is not h and last and _norm(o["name"]).endswith(last)]
    if len(hits) != 1 or sheet_ids.get(hits[0]["cite"]) is None:
        return f"name: {x}"
    return sheet_ids[hits[0]["cite"]]


def _sheet_format(h: dict) -> tuple[Any, Any]:
    """(format, partner_present) the header's own tags claim."""
    tags = h["tags"]
    if tags.get("format") == "double":
        return "double", False
    if tags.get("tag_partner") and tags.get("multi_with"):
        return "multi", True
    return None, None


# --- trainer rows -----------------------------------------------------------------------


def _moves(moves: Iterable[int | None] | None) -> list[int] | None:
    return None if moves is None else [m for m in moves if m]


def _forced_ability(override: int, abilities: Sequence[int]) -> int | None:
    """The ability the override byte's high nibble forces, or None when it forces none."""
    nibble = override >> 4
    if nibble == 1:
        return abilities[0]
    if nibble == 2:
        return abilities[1] or abilities[0]
    return None


def _explained(
    label: str,
    pk_trainers: dict,
    others: Iterable[str | None],
    double: bool,
    names: dict[int, str],
) -> set[str]:
    """Weathers the leads' abilities set: the label's lead (and second mon in a true
    double), the other opponent's lead and the tag partner's lead."""
    leads: list[dict] = []
    party = pk_trainers.get(label, {}).get("party") or []
    leads += party[: 2 if double else 1]
    for other in others:
        p = pk_trainers.get(other, {}).get("party") if other else None
        if p:
            leads.append(p[0])
    return {
        WEATHER_ABILITIES[names[m["ability"]]]
        for m in leads
        if names.get(m.get("ability")) in WEATHER_ABILITIES
    }


def _pk_weathers(pkcalc: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for key, labels in pkcalc.get("weather", {}).items():
        if key not in PKCALC_WEATHERS:
            raise ValueError(f"PKCalc weather list {key!r} is not one of {sorted(PKCALC_WEATHERS)}")
        if PKCALC_WEATHERS[key] is not None:
            for label in labels:
                out[label].add(PKCALC_WEATHERS[key])
    return out


def trainer_rows(
    rom: Sequence[dict],
    mappings: Mappings,
    pkcalc: dict,
    docs: dict,
    species: dict[str, Any],
    overrides: Overrides,
) -> list[Row]:
    """Every Task 13 row (module docstring), in PKCalc label order, then the unmapped
    sheet headers."""
    by_id = {t["id"]: t for t in rom}
    pk_trainers = pkcalc.get("trainers", {})
    label_ids = {m.label: m.id for m in mappings.pkcalc}
    sheet_ids = {m.label: m.id for m in mappings.sheet}
    sheet = _sheet_by_id(docs, mappings)
    raw = docs.get("raw_trainers", {})
    formats, partner_only = _pk_formats(pkcalc)
    weathers = _pk_weathers(pkcalc)
    ability_names = {int(k): a["name"] for k, a in pkcalc.get("abilities", {}).items()}
    partner_starts = _partner_starters(rom)
    personal = species["species"]
    sheet_moves = _sheet_move_index(docs)
    sheet_items = _name_index(
        ((i, n) for n, i in _sheet_item_ids(docs, species["items"]).items()), "sheet item"
    )

    rows: list[Row] = []
    for m in mappings.pkcalc:
        if m.id is None:
            rows.append(Row(TABLE, m.label, "mapped", None, True, None, True))
            continue
        t, label = by_id[m.id], m.label
        key = f"{m.id} {label}"
        pk_party = pk_trainers[label].get("party") or []
        raw_t = raw.get(str(m.id))
        raw_party = raw_t["party"] if raw_t else None

        def row(field: str, rv: Any, pv: Any, sv: Any, key: str = key) -> None:
            rows.append(Row(TABLE, key, field, rv, pv, sv, True))

        row("party_size", len(t["party"]), len(pk_party), None if raw_t is None else len(raw_party))
        for slot in range(max(len(t["party"]), len(pk_party))):
            rm = t["party"][slot] if slot < len(t["party"]) else None
            pm = pk_party[slot] if slot < len(pk_party) else None
            sm = raw_party[slot] if raw_party is not None and slot < len(raw_party) else None
            mkey = f"{key}/{slot}"

            def mon(field: str, rv: Any, pv: Any, sv: Any, mkey: str = mkey) -> None:
                row(field, rv, pv, sv, key=mkey)

            def get(d: dict | None, name: str) -> Any:
                return None if d is None else d.get(name)

            mon("species", get(rm, "species"), get(pm, "species"), get(sm, "species"))
            mon("form", get(rm, "form"), get(pm, "form"), get(sm, "form"))
            mon("level", get(rm, "level"), get(pm, "level"), get(sm, "level"))
            sheet_item = None
            if sm is not None and raw_t["has_items"]:
                sheet_item = 0 if sm["item"] is None else _ref(sheet_items, sm["item"])
            mon(
                "item",
                get(rm, "item"),
                None if pm is None else (pm.get("item") or 0),
                sheet_item,
            )
            rom_moves = None
            if rm is not None:
                rom_moves = rm["moves"]
                if rom_moves is None:
                    rom_moves = default_moves(
                        _get(species["learnsets"], rm["species"]), rm["level"]
                    )
            sheet_mv = None
            if sm is not None and raw_t["has_moves"]:
                sheet_mv = [_ref(sheet_moves, n) for n in sm["moves"] if n is not None]
            mon("moves", _moves(rom_moves), _moves(get(pm, "moves")), sheet_mv)
            mon(
                "ivs",
                None if rm is None else [difficulty_to_ivs(rm["difficulty"])] * 6,
                None if pm is None else list(pm["ivs"]),
                None,
            )
            mon("difficulty", get(rm, "difficulty"), None, get(sm, "difficulty"))
            mon("ability_override", get(rm, "ability_override"), None, get(sm, "ability_number"))
            if rm is not None:
                forced = _forced_ability(
                    rm["ability_override"], personal[rm["species"]]["abilities"]
                )
                if forced is not None:
                    mon("ability_slot", forced, get(pm, "ability"), None)

        headers = sheet.get(m.id, [])
        pk_ai = pkcalc.get("ai_flags", {}).get(label)
        row(
            "ai_flags",
            t["ai_flags"] & AI_MASK,
            None if pk_ai is None else pk_ai & AI_MASK,
            _joined(_sheet_ai(h) for _, h in headers),
        )

        if label in partner_only:
            pk_fmt = pk_opp = pk_partner = None
            other = partner = None
        else:
            pk_fmt, other, partner = formats.get(label, ("single", None, None))
            pk_opp = None
            if other is not None:
                pk_opp = label_ids.get(other)
                if pk_opp is None:
                    pk_opp = f"label: {other}"
            pk_partner = pk_fmt == "multi"
        sheet_fmts = [_sheet_format(h) for _, h in headers]
        if t["format"] != UNKNOWN:
            row("format", t["format"], pk_fmt, _joined(f for f, _ in sheet_fmts))
            row(
                "opponent_2",
                t["opponent_2"],
                pk_opp,
                _joined(_sheet_opponent(sec, h, sheet_ids) for sec, h in headers),
            )
            row(
                "partner_present",
                t["partner"] is None or len(t["partner"]) > 0,
                pk_partner,
                _joined(p for _, p in sheet_fmts),
            )

        pk_start = _pk_starter(label)
        sheet_start = _joined(h["tags"].get("player_starter") for _, h in headers)
        derived, _source = _starter(m.id, overrides, partner_starts, sheet_start, pk_start)
        if derived is not None:
            row("player_starter", derived, pk_start, sheet_start)

        if t["start_condition"] != UNKNOWN:
            pk_w = set(weathers.get(label, set()))
            leads = (other, partner)
            pk_w -= _explained(label, pk_trainers, leads, pk_fmt == "double", ability_names)
            row(
                "area_weather",
                t["start_condition"],
                _joined(pk_w),
                _joined(_sheet_weather(sec, h) for sec, h in headers),
            )

    for m in mappings.sheet:
        if m.id is None:
            rows.append(Row(TABLE, m.label, "mapped", None, None, True, True))
    return rows


def _starter(
    tid: int,
    overrides: Overrides,
    partner_starts: dict[int, set[int]],
    sheet: Any,
    pkcalc: int | None,
) -> tuple[Any, str]:
    """The derived player starter and its source (preflight #28 order, the ROM's partner
    variant before the sheet: ROM > sheet)."""
    fixed = overrides.fields.get(tid, {}).get("player_starter")
    if fixed is not None:
        return fixed, f"overrides fields.{tid} ({overrides.sources.get(f'fields.{tid}', '?')})"
    variants = partner_starts.get(tid, set())
    if len(variants) > 1:
        raise ValueError(f"ROM partner {tid} is fought with several player starters {variants}")
    if len(variants) == 1:
        return next(iter(variants)), "rom: partner variant"
    if sheet is not None:
        return sheet, "sheet tag"
    if pkcalc is not None:
        return pkcalc, "pkcalc bracket"
    return None, ""


# --- sections ---------------------------------------------------------------------------


def _ids(ids: Iterable[int]) -> str:
    return "[" + ", ".join(map(str, ids)) + "]"


def mapping_section(mappings: Mappings) -> list[str]:
    """Every non-``exact`` mapping and every unmapped label or header with its candidates;
    ids several sheet headers share."""
    names = {i: t["name"] for i, t in mappings.rom_trainers.items()}
    lines: list[str] = []
    for what, maps in (("PKCalc label", mappings.pkcalc), ("sheet header", mappings.sheet)):
        hows = Counter(m.how for m in maps)
        lines.append(f"{what}s: {len(maps)}, " + ", ".join(f"{h} {hows[h]}" for h in sorted(hows)))
        for m in maps:
            if m.how == "exact":
                continue
            text = f"{m.how}: {what} {m.label!r}"
            if m.id is not None:
                text += f" -> {m.id} {names.get(m.id, '?')}"
            text += f" (candidates {_ids(m.candidates)})"
            if m.form_diffs:
                diffs = ", ".join(
                    f"slot {s} PKCalc form {p} ROM form {r}" for s, p, r in m.form_diffs
                )
                text += f"; forms differ: {diffs}"
            if m.lost_to is not None:
                text += f"; lost id {m.lost_to} to an override"
            lines.append(text)
    shared: dict[int, list[str]] = defaultdict(list)
    for m in mappings.sheet:
        if m.id is not None:
            shared[m.id].append(m.label)
    for tid, cites in sorted(shared.items()):
        if len(cites) > 1:
            lines.append(
                f"sheet headers sharing id {tid} {names.get(tid, '?')}: {', '.join(cites)}"
            )
    return lines


def pid_section(
    rom: Sequence[dict], mappings: Mappings, pkcalc: dict, species: dict[str, Any]
) -> list[str]:
    """The vanilla ``TrainerData_BuildParty`` PID (``nuzline.data.trainer``) on the ROM
    inputs of every override-0 mon of a mapped label vs PKCalc's nature, gender and
    ability (Ruling 19: recomputed, recorded, not asserted). A trainer reproduces completely
    when all its override-0 mons do (it needs one). A miss after an override mon in the same
    party says so: the patched routine keeps that mon's PID adjustment for the rest of the
    party (arm9 0x020795A0 writes the trainer's gender modifier in place)."""
    by_id = {t["id"]: t for t in rom}
    personal = species["species"]
    pk_trainers = pkcalc.get("trainers", {})
    n_mons = ng_ok = ab_ok = 0
    n_trainers = complete = 0
    headline: list[str] = []
    misses: list[str] = []
    unchecked: list[str] = []
    for m in mappings.pkcalc:
        if m.id is None:
            continue
        t = by_id[m.id]
        party = pk_trainers[m.label].get("party") or []
        checked = good = 0
        prior: tuple[int, int] | None = None
        for slot, (rm, pm) in enumerate(zip(t["party"], party, strict=False)):
            key = f"{m.id} {m.label}/{slot}"
            if rm["ability_override"] != 0:
                unchecked.append(f"not checked: override byte {rm['ability_override']}: {key}")
                prior = prior or (slot, rm["ability_override"])
                continue
            pid = trainer_pid(
                rm["difficulty"],
                rm["level"],
                rm["species"],
                t["id"],
                t["class_id"],
                female_class=t["class_gender"] == GENDER_FEMALE,
            )
            sp = personal[rm["species"]]
            a1, a2 = sp["abilities"]
            got = (
                NATURES[pid % 25],
                gender_of(sp["gender_ratio"], pid),
                a2 if a2 and pid & 1 else a1,
            )
            if pm["nature"] not in _NATURE_INDEX or pm.get("gender") not in _GENDERS:
                raise ValueError(
                    f"PKCalc {key}: nature {pm['nature']!r} gender {pm.get('gender')!r}"
                )
            want = (pm["nature"], _GENDERS[pm.get("gender")], pm["ability"])
            checked += 1
            ng = got[:2] == want[:2]
            ab = got[2] == want[2]
            ng_ok += ng
            ab_ok += ab
            good += ng and ab
            if not (ng and ab):
                diffs = []
                if got[0] != want[0]:
                    diffs.append(f"nature {got[0]} (PKCalc {want[0]})")
                if got[1] != want[1]:
                    diffs.append(
                        f"gender {_GENDER_NAMES[got[1]]} (PKCalc {_GENDER_NAMES[want[1]]})"
                    )
                if got[2] != want[2]:
                    diffs.append(f"ability {got[2]} (PKCalc {want[2]})")
                text = f"miss: {key} species {rm['species']}: " + ", ".join(diffs)
                if prior is not None:
                    text += f" (after override byte {prior[1]} at slot {prior[0]})"
                misses.append(text)
        n_mons += checked
        if checked:
            n_trainers += 1
            complete += good == checked
        if m.label in PID_HEADLINE:
            headline.append(f"{m.label} ({m.id}): {good}/{checked}")
    return [
        f"nature and gender reproduced for {ng_ok} of {n_mons} override-0 Pokemon",
        f"ability reproduced for {ab_ok} of {n_mons}",
        f"trainers reproduced completely: {complete} of {n_trainers}",
        *headline,
        *misses,
        *unchecked,
    ]


def _v(value: Any) -> str:
    return "-" if value is None else str(value)


def area_section(
    rom: Sequence[dict],
    mappings: Mappings,
    pkcalc: dict,
    docs: dict,
    overrides: Overrides,
) -> list[str]:
    """Battle setup claims by source: refused ``unknown`` start conditions and formats (with
    every source), Trick Room and Gravity (ROM map weather, overrides, sheet, PKCalc),
    non-single formats, derived player starters, AI flag bits above 10, and the flags.js
    labels without a party."""
    by_id = {t["id"]: t for t in rom}
    labels: dict[int, list[str]] = defaultdict(list)
    for m in mappings.pkcalc:
        if m.id is not None:
            labels[m.id].append(m.label)
    sheet = _sheet_by_id(docs, mappings)
    sheet_ids = {m.label: m.id for m in mappings.sheet}
    formats, partner_only = _pk_formats(pkcalc)
    weathers = _pk_weathers(pkcalc)

    def name(tid: int) -> str:
        return f"{tid} {by_id[tid]['name'] if tid in by_id else '?'}"

    def pk_weather(tid: int) -> str:
        return ", ".join(f"{lb}: {_v(_joined(weathers.get(lb, ())))}" for lb in labels[tid]) or "-"

    def pk_format(tid: int) -> str:
        return (
            ", ".join(
                f"{lb}: {'partner only' if lb in partner_only else formats.get(lb, ('single',))[0]}"
                for lb in labels[tid]
            )
            or "-"
        )

    lines = ["refused: unknown (user call)"]
    for t in rom:
        tid = t["id"]
        if t["start_condition"] == UNKNOWN:
            a = overrides.area.get(tid)
            exc = overrides.area_exceptions.get(tid)
            ov = "-" if a is None else f"weather {_v(a.weather)} ({a.source})"
            if exc is not None:
                ov = f"area_exception ({exc.reason}; {exc.source})"
            sheet_w = _joined(_sheet_weather(s, h) for s, h in sheet.get(tid, []))
            cites = ", ".join(h["cite"] for _, h in sheet.get(tid, [])) or "no sheet header"
            lines.append(
                f"- {name(tid)} start_condition: ROM unknown | PKCalc {pk_weather(tid)} | "
                f"sheet {_v(sheet_w)} ({cites}) | overrides {ov}"
            )
        if t["format"] == UNKNOWN:
            sheet_f = _joined(_sheet_format(h)[0] for _, h in sheet.get(tid, []))
            sheet_o = _joined(_sheet_opponent(s, h, sheet_ids) for s, h in sheet.get(tid, []))
            lines.append(
                f"- {name(tid)} format: ROM unknown (opponent_2 {_v(t['opponent_2'])}, "
                f"partner {_v(t['partner'])}) | PKCalc {pk_format(tid)} | "
                f"sheet {_v(sheet_f)} (opponent_2 {_v(sheet_o)})"
            )

    lines.append("")
    lines.append("Trick Room and Gravity claims (ROM map weather 18, overrides, sheet, PKCalc):")
    rom_tr = {r["id"]: r for r in mappings.field_effects if r["effect"] == "trick_room"}
    sheet_claims: dict[str, dict[int | str, list[str]]] = {
        "trick_room": defaultdict(list),
        "gravity": defaultdict(list),
    }
    for cite, (sec, h) in _sheet_rows(docs).items():
        for claim in ("trick_room", "gravity"):
            says = any(
                tags.get("permanent") == claim or (claim == "trick_room" and tags.get("trick_room"))
                for tags in (sec["tags"], h["tags"])
            )
            if says:
                tid = sheet_ids.get(cite)
                sheet_claims[claim][tid if tid is not None else f"unmapped {cite}"].append(cite)
    gravity_pk = {m.id for m in mappings.pkcalc if m.label == GRAVITY_LABEL and m.id is not None}
    for claim in ("trick_room", "gravity"):
        ids: set[int | str] = set(sheet_claims[claim])
        ids |= {i for i, a in overrides.area.items() if getattr(a, claim)}
        if claim == "trick_room":
            ids |= set(rom_tr) | set(overrides.area_exceptions)
        else:
            ids |= gravity_pk
        for tid in sorted(ids, key=lambda i: (isinstance(i, str), str(i).zfill(6))):
            if isinstance(tid, str):
                lines.append(f"{claim}: {tid}: sheet only, no ROM id")
                continue
            a = overrides.area.get(tid)
            says_ov = a is not None and getattr(a, claim)
            ov = f"yes ({a.source})" if says_ov else "-"
            if claim == "trick_room" and tid in overrides.area_exceptions:
                ov = f"area_exception ({overrides.area_exceptions[tid].reason})"
            says_sheet = bool(sheet_claims[claim].get(tid))
            sh = ", ".join(sheet_claims[claim].get(tid, [])) or "-"
            if claim == "trick_room":
                says_src = tid in rom_tr
                src = f"ROM map headers {rom_tr[tid]['map_headers']}" if says_src else "ROM -"
            else:
                says_src = tid in gravity_pk
                src = f"PKCalc {GRAVITY_LABEL}" if says_src else "PKCalc -"
            claimed = {says_src, says_sheet}
            if not (claim == "trick_room" and tid in overrides.area_exceptions):
                claimed.add(says_ov)
            mark = " (differs)" if len(claimed) > 1 else ""
            lines.append(f"{claim}: {name(tid)}: {src} | overrides {ov} | sheet {sh}{mark}")

    lines.append("")
    lines.append("Formats other than single (ROM | PKCalc | sheet):")
    for t in rom:
        tid = t["id"]
        heads = sheet.get(tid, [])
        sheet_f = _joined(_sheet_format(h)[0] for _, h in heads)
        sheet_o = _joined(_sheet_opponent(s, h, sheet_ids) for s, h in heads)
        pk_f = pk_format(tid)
        if (
            t["format"] == "single"
            and sheet_f is None
            and sheet_o is None
            and (pk_f == "-" or all(p.endswith(": single") for p in pk_f.split(", ")))
        ):
            continue
        partner = (
            "null (present)"
            if t["partner"] is None
            else (", ".join(str(p["trainer"]) for p in t["partner"]) or "none")
        )
        lines.append(
            f"format: {name(tid)}: ROM {t['format']} opponent_2 {_v(t['opponent_2'])} "
            f"partner {partner} | PKCalc {pk_f} | sheet {_v(sheet_f)} opponent_2 {_v(sheet_o)}"
        )

    lines.append("")
    lines.append("player_starter derived values (overrides, ROM partner variant, sheet, PKCalc):")
    partner_starts = _partner_starters(rom)
    for m in mappings.pkcalc:
        if m.id is None:
            continue
        pk_start = _pk_starter(m.label)
        sheet_start = _joined(h["tags"].get("player_starter") for _, h in sheet.get(m.id, []))
        derived, source = _starter(m.id, overrides, partner_starts, sheet_start, pk_start)
        if derived is not None:
            lines.append(f"player_starter: {m.id} {m.label}: {derived} from {source}")

    lines.append("")
    high = [
        f"ai flag bits above {AI_BITS - 1}: {name(t['id'])} ROM {t['ai_flags']:#x}"
        for t in rom
        if t["ai_flags"] & ~AI_MASK and t["id"] in labels
    ]
    high += [
        f"ai flag bits above {AI_BITS - 1}: PKCalc {lb!r} {v:#x}"
        for lb, v in pkcalc.get("ai_flags", {}).items()
        if v & ~AI_MASK
    ]
    lines += high or [f"ai flag bits above {AI_BITS - 1}: none"]

    lines.append("")
    for label in pkcalc.get("flag_labels_without_party", []):
        where = [k for k in ("hidden",) if label in pkcalc.get(k, [])]
        where += [f"weather {w}" for w, ls in pkcalc.get("weather", {}).items() if label in ls]
        where += (
            [f"ai_flags {pkcalc['ai_flags'][label]}"] if label in pkcalc.get("ai_flags", {}) else []
        )
        lines.append(
            f"flags.js label without a PKCalc party (no id, not compared): {label} "
            f"[{'; '.join(where) or 'no flag'}]"
        )
    return lines
