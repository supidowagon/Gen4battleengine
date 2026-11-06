"""Write the intermediate JSON files from the ROM (spec §6.3, §9.1).

``species, moves, items, learnsets, tm_learnsets, exp_tables`` and ``trainers`` come from the ROM
readers, and so does ``forms`` except its names; those and ``form_rows`` come from the
decomp's form registry (Ruling 6: the ROM holds no form names, and the row order is
``Pokemon_GetFormNarcIndex``'s). ``types, natures, charmap, abilities`` are reused from a
fresh decomp extraction (Ruling 14: the charmap is the font encoding and has no table in the ROM; the others are
reused for a non-vanilla game only when the ROM diff shows them unchanged). Outputs are
formatted by ``extract_decomp._write``.

``--game kaizo`` reads the patched ROM (``tools.rom.patch.kaizo_rom``) with every reader,
after the ROM diff's reuse check: trainers with the Kaizo party layout (K2b Ruling 6), map
headers from ``d_test.narc`` (Ruling 8) and start conditions settled by
``data/kaizo/start_conditions.json``. It also writes ``changed_scripts.json`` (the changed
members of the battle script NARCs, Ruling 13); the CLI appends an explicit ``unknown``
``[[member]]`` entry to ``data/kaizo/rom_diff.toml`` for each changed member it lacks.

Battle command operands that are variables no constant reaches on some walked script path
(``tools.rom.readers.maps.UnresolvedOperand``) place no trainer; each is printed to stderr
and listed in ``rom_unresolved.json`` beside the outputs, so a trainer they hide is visible
(K1 final review).

Writes go to a temporary directory beside ``out`` and are moved in file by file with
``os.replace``; files already in ``out`` that this run does not write are left alone, and a
failure before the files are moved in leaves ``out`` untouched (Ruling 13). A failure of
``os.replace`` midway through the move leaves the files moved so far.

CLI: ``python -m tools.extract_rom --game vanilla|kaizo [--out DIR] [--decomp DIR]
[--symbols MAP] [--config data/local.toml]``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tools.extract_decomp import (
    _write,
    extract_charmap,
    extract_move_lists,
    extract_static,
)
from tools.rom.inputs import (
    REPO_ROOT,
    VANILLA_SHA1,
    RomInputError,
    RomInputs,
    load_inputs,
    read_checked,
)
from tools.rom.symbols import DEFAULT_MAP

ROM_FILES = (
    "species",
    "moves",
    "move_lists",
    "items",
    "learnsets",
    "tm_learnsets",
    "evolutions",
    "egg_moves",
    "tutor_moves",
    "tutor_learnsets",
    "exp_tables",
    "trainers",
)
REUSED_FILES = ("types", "natures", "charmap", "abilities")
DECOMP_FILES = ("form_rows", "evolution_methods")  # taken from the decomp for every game (Ruling 6)
REUSE_SYMBOLS = ("sTypeMatchupMultipliers", "sNatureStatAffinities")
REUSE_BANKS = (
    "TEXT_BANK_ABILITY_NAMES",
    "TEXT_BANK_POKEMON_TYPE_NAMES",
    "TEXT_BANK_NATURE_NAMES",
)
UNRESOLVED_FILE = "rom_unresolved.json"
FIELD_EFFECTS_FILE = "rom_field_effects.json"
CHANGED_SCRIPTS_FILE = "changed_scripts.json"
ROM_DIFF_TOML = REPO_ROOT / "data" / "kaizo" / "rom_diff.toml"
# Raw ROM output; data/intermediate_kaizo/*.json are the resolved files (K2b Ruling 2).
DEFAULT_OUT = {
    "vanilla": REPO_ROOT / "data" / "intermediate",
    "kaizo": REPO_ROOT / "data" / "intermediate_kaizo" / "rom",
}


def check_reuse_names(symbols, decomp: Path) -> None:
    """Raise unless every ``REUSE_SYMBOLS`` name is in the linker map and every
    ``REUSE_BANKS`` name in ``generated/text_banks.txt``: a misspelt name would match no
    diff entry and pass the reuse check silently."""
    from tools.extract_decomp import _constants

    banks = set(_constants(Path(decomp), "text_banks", "TEXT_BANK_"))
    missing = [b for b in REUSE_BANKS if b not in banks]
    for name in REUSE_SYMBOLS:
        try:
            symbols.find(name)
        except KeyError as e:
            missing.append(f"{name} ({e.args[0]})")
    if missing:
        raise RomInputError(f"reuse check names not found: {', '.join(missing)}")


def reuse_violations(diff, decomp: Path) -> list[str]:
    """What the diff changed among the sources of the reused tables (Ruling 14)."""
    from tools.rom.diff import member_label

    bad = [f"symbol {c.symbol} ({c.module})" for c in diff.code if c.symbol in REUSE_SYMBOLS]
    for f in diff.files:
        if not f.path.endswith("pl_msg.narc"):
            continue
        for i in f.members_changed:
            label = member_label(f.path, i, decomp)
            if label in REUSE_BANKS:
                bad.append(f"{f.path} member {i} ({label})")
    return bad


def _kaizo_checks(inputs: RomInputs, decomp: Path, symbols_path: Path, vanilla_files):
    """The patched Kaizo ROM and its diff against vanilla, once the reuse check passes."""
    from tools.rom.diff import diff_roms
    from tools.rom.nds import RomFiles
    from tools.rom.patch import kaizo_rom
    from tools.rom.symbols import SymbolMap

    symbols = SymbolMap.from_map(symbols_path)
    check_reuse_names(symbols, decomp)
    path, _ = kaizo_rom(inputs)
    kaizo = RomFiles.open(path.read_bytes())
    diff = diff_roms(vanilla_files, kaizo, symbols)
    bad = reuse_violations(diff, decomp)
    if bad:
        raise RomInputError(
            "the Kaizo ROM changes the sources of the reused tables: " + "; ".join(bad)
        )
    return kaizo, diff


def _member_entries(path: Path, changed: dict[str, list[int]]) -> set[tuple[str, int]]:
    """The ``[[member]]`` keys ``path`` lists, once ``load_member_classification`` accepts
    it against ``changed`` (an unknown status or a stale entry raises ``RomInputError``)."""
    import tomllib

    from tools.rom.diff import load_member_classification

    try:
        load_member_classification(path, changed)
    except ValueError as e:
        raise RomInputError(str(e)) from e
    with Path(path).open("rb") as f:
        return {(m["narc"], m["index"]) for m in tomllib.load(f).get("member", [])}


def append_unknown_members(path: Path, changed: dict[str, list[int]]) -> list[tuple[str, int]]:
    """Append an explicit ``unknown`` ``[[member]]`` entry to ``path`` for each changed
    member it does not list (K2b Ruling 13); return those (narc, index), sorted."""
    listed = _member_entries(path, changed)
    missing = sorted({(n, i) for n, ids in changed.items() for i in ids} - listed)
    if missing:
        text = Path(path).read_text(encoding="utf-8")
        blocks = [
            f'[[member]]\nnarc = "{n}"\nindex = {i}\nstatus = "unknown"\nnote = ""\n'
            for n, i in missing
        ]
        head = text.rstrip("\n") + "\n\n" if text.strip() else ""
        Path(path).write_text(head + "\n".join(blocks), encoding="utf-8", newline="\n")
    return missing


def _report_kaizo(trainers: list[dict], effects: dict[int, dict[str, list[int]]]) -> list[dict]:
    """Print what the Kaizo trainers leave open to stderr: each ``unknown`` start
    condition (``data/kaizo/start_conditions.json``), each partner not evaluated
    (``partner`` null) and the field effects map weathers start (Trick Room); return the
    ``FIELD_EFFECTS_FILE`` rows ``{id, name, effect, map_headers}`` (the trainer's map
    headers whose weather starts ``effect``; it may be on other maps too)."""
    from tools.extract_decomp import UNKNOWN_CONDITION

    unknown = [t["id"] for t in trainers if t["start_condition"] == UNKNOWN_CONDITION]
    if unknown:
        print(f"warning: start condition unknown for trainers {unknown}", file=sys.stderr)
    partners = [t["id"] for t in trainers if t["partner"] is None]
    if partners:
        print(f"warning: partner not evaluated for trainers {partners}", file=sys.stderr)
    rows = []
    for tid, by_effect in sorted(effects.items()):
        for effect, headers in sorted(by_effect.items()):
            row = {"id": tid, "name": trainers[tid]["name"], "effect": effect}
            rows.append({**row, "map_headers": headers})
    if rows:
        print(
            f"note: {len(rows)} trainers fought on a map that starts a field effect "
            f"(listed in {FIELD_EFFECTS_FILE})",
            file=sys.stderr,
        )
    return rows


def _unresolved_rows(unresolved, decomp: Path) -> list[dict]:
    from tools.rom.readers.maps import SCRIPTS_NARC, var_ids

    names = {v: k for k, v in var_ids(decomp).items()}
    return [
        {
            "narc": SCRIPTS_NARC,
            "member": u.member,
            "offset": u.offset,
            "variable": u.variable,
            "variable_name": names.get(u.variable),
            "constants_on_other_paths": u.constants,
        }
        for u in unresolved
    ]


def extract(game: str, out: Path, decomp: Path, symbols: Path, inputs: RomInputs) -> dict[str, str]:
    """Write the intermediate files for ``game`` into ``out``; return ``rom_sources.json``.

    Unresolved battle operands are printed to stderr and written to ``UNRESOLVED_FILE``.
    For Kaizo every reader reads the patched ROM, and ``CHANGED_SCRIPTS_FILE`` is written
    from the ROM diff; ``ROM_DIFF_TOML``'s ``[[member]]`` entries are checked (a stale one
    raises) but never written here (``main`` appends the missing ones)."""
    if game not in DEFAULT_OUT:
        raise ValueError(f"unknown game {game!r}")
    out, decomp, symbols = Path(out), Path(decomp), Path(symbols)

    from tools.rom.nds import RomFiles

    vanilla_files = RomFiles.open(read_checked(inputs.vanilla_rom, VANILLA_SHA1, "sha1"))
    rom, changed = vanilla_files, None
    if game == "kaizo":
        from tools.rom.diff import changed_scripts

        rom, diff = _kaizo_checks(inputs, decomp, symbols, vanilla_files)
        changed = changed_scripts(diff)
        _member_entries(ROM_DIFF_TOML, changed)

    from tools.rom.readers.code_tables import egg_moves, tutor_learnsets, tutor_moves
    from tools.rom.readers.evolutions import read_evolutions
    from tools.rom.readers.items import read_items
    from tools.rom.readers.learnsets import read_learnsets
    from tools.rom.readers.maps import trainer_start_conditions
    from tools.rom.readers.moves import read_move_lists, read_moves
    from tools.rom.readers.personal import (
        read_exp_tables,
        read_forms,
        read_species,
        read_tm_learnsets,
    )
    from tools.rom.readers.trainers import read_trainers
    from tools.rom.symbols import SymbolMap

    smap = SymbolMap.from_map(symbols)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        charmap = extract_charmap(decomp)
        unresolved = []
        effects: dict[int, dict[str, list[int]]] = {}
        starts = trainer_start_conditions(
            rom, decomp, smap, unresolved=unresolved, field_effects=effects
        )
        rows = _unresolved_rows(unresolved, decomp)
        for r in rows:
            print(
                f"warning: {r['narc']} member {r['member']} offset {r['offset']:#x}: battle "
                f"operand {r['variable_name'] or hex(r['variable'])} has no constant on some "
                f"path ({r['constants_on_other_paths']} on others); its trainer is placed "
                "on no map through it",
                file=sys.stderr,
            )
        tutors = tutor_moves(rom, smap, decomp)
        vanilla_rom = vanilla_files if rom is not vanilla_files else None
        tables = {
            "species": read_species(rom, decomp, charmap),
            "forms": read_forms(rom, decomp, charmap),
            "moves": read_moves(rom, decomp, charmap),
            "move_lists": read_move_lists(rom, extract_move_lists(decomp), vanilla_rom=vanilla_rom),
            "items": read_items(rom, decomp, smap, charmap),
            "learnsets": read_learnsets(rom, decomp),
            "tm_learnsets": read_tm_learnsets(rom),
            "evolutions": read_evolutions(rom),
            "egg_moves": {str(k): v for k, v in egg_moves(rom, smap, decomp).items()},
            "tutor_moves": tutors,
            "tutor_learnsets": tutor_learnsets(rom, smap, len(tutors)),
            "exp_tables": read_exp_tables(rom),
            "trainers": read_trainers(rom, decomp, smap, charmap, starts, layout=game),
        }
        for name, data in tables.items():
            _write(stage / f"{name}.json", data)
        if changed is not None:
            _write(stage / CHANGED_SCRIPTS_FILE, changed)
            _write(stage / FIELD_EFFECTS_FILE, _report_kaizo(tables["trainers"], effects))
        fresh = stage / "decomp"
        extract_static(decomp, fresh)
        for name in (*REUSED_FILES, *DECOMP_FILES):
            os.replace(fresh / f"{name}.json", stage / f"{name}.json")
        shutil.rmtree(fresh)
        sources = {f"{n}.json": "rom" for n in ROM_FILES}
        sources["forms.json"] = "rom; names: decomp"
        sources["form_rows.json"] = "decomp: form registry"
        sources["evolution_methods.json"] = "decomp: generated/evolution_methods.txt"
        reused = (
            "decomp: vanilla ROM (decomp is the source)"
            if changed is None
            else "decomp: the Kaizo ROM leaves its sources unchanged (reuse check)"
        )
        sources.update({f"{n}.json": reused for n in REUSED_FILES})
        if changed is not None:
            sources[CHANGED_SCRIPTS_FILE] = "rom diff"
            sources[FIELD_EFFECTS_FILE] = "rom: map weather through the battle weather switch"
        _write(stage / "rom_sources.json", sources)
        _write(stage / UNRESOLVED_FILE, rows)
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(stage.iterdir()):
            os.replace(p, out / p.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m tools.extract_rom", description=__doc__.split("\n")[0]
    )
    p.add_argument("--game", required=True, choices=sorted(DEFAULT_OUT))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--decomp", type=Path, default=REPO_ROOT / "external" / "pokeplatinum")
    p.add_argument("--symbols", type=Path, default=DEFAULT_MAP)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "data" / "local.toml")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = DEFAULT_OUT[args.game]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = load_inputs(args.config)
        sources = extract(args.game, args.out, args.decomp, args.symbols, inputs)
        added = []
        if args.game == "kaizo":
            changed = json.loads((args.out / CHANGED_SCRIPTS_FILE).read_text(encoding="utf-8"))
            added = append_unknown_members(ROM_DIFF_TOML, changed)
    except RomInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if added:
        print(
            f"{ROM_DIFF_TOML}: appended {len(added)} unknown [[member]] entries: "
            + ", ".join(f"{n}#{i}" for n, i in added)
        )
    for name, src in sorted(sources.items()):
        print(f"{name}: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
