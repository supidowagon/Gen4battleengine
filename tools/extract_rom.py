"""Write the intermediate JSON files from the ROM (spec §6.3, §9.1).

``species, moves, items, learnsets, exp_tables`` and ``trainers`` come from the ROM
readers; ``types, natures, charmap, abilities`` are reused from a fresh decomp extraction
(Ruling 14: the charmap is the font encoding and has no table in the ROM; the others are
reused for a non-vanilla game only when the ROM diff shows them unchanged). Outputs are
formatted by ``extract_decomp._write``.

Writes go to a temporary directory beside ``out`` and are moved in file by file with
``os.replace``; files already in ``out`` that this run does not write are left alone, and
on any failure ``out`` is neither created nor changed (Ruling 13).

CLI: ``python -m tools.extract_rom --game vanilla|kaizo [--out DIR] [--decomp DIR]
[--symbols MAP] [--config data/local.toml]``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tools.extract_decomp import (
    _write,
    extract_charmap,
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

ROM_FILES = ("species", "moves", "items", "learnsets", "exp_tables", "trainers")
REUSED_FILES = ("types", "natures", "charmap", "abilities")
REUSE_SYMBOLS = ("sTypeMatchupMultipliers", "sNatureStatAffinities")
REUSE_BANKS = (
    "TEXT_BANK_ABILITY_NAMES",
    "TEXT_BANK_POKEMON_TYPE_NAMES",
    "TEXT_BANK_NATURE_NAMES",
)
KAIZO_REFUSAL = (
    "trainers: the Kaizo trainer layout is not confirmed (spec §5.4); Kaizo extraction lands in K2b"
)
DEFAULT_OUT = {
    "vanilla": REPO_ROOT / "data" / "intermediate",
    "kaizo": REPO_ROOT / "data" / "intermediate_kaizo",
}


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


def _kaizo_checks(inputs: RomInputs, decomp: Path, symbols_path: Path, vanilla_files) -> None:
    """Diff, reuse check and reader layout checks on the Kaizo ROM, then the refusal."""
    from tools.rom.diff import diff_roms
    from tools.rom.nds import RomFiles
    from tools.rom.patch import kaizo_rom
    from tools.rom.readers.items import read_items
    from tools.rom.readers.learnsets import read_learnsets
    from tools.rom.readers.moves import read_moves
    from tools.rom.readers.personal import read_exp_tables, read_species
    from tools.rom.symbols import SymbolMap

    symbols = SymbolMap.from_map(symbols_path)
    path, _ = kaizo_rom(inputs)
    kaizo = RomFiles.open(path.read_bytes())
    diff = diff_roms(vanilla_files, kaizo, symbols)
    bad = reuse_violations(diff, decomp)
    if bad:
        raise RomInputError(
            "the Kaizo ROM changes the sources of the reused tables: " + "; ".join(bad)
        )
    charmap = extract_charmap(decomp)
    read_species(kaizo, decomp, charmap)
    read_exp_tables(kaizo)
    read_learnsets(kaizo, decomp)
    read_moves(kaizo, decomp, charmap)
    read_items(kaizo, decomp, symbols, charmap)
    raise RomInputError(KAIZO_REFUSAL)


def extract(game: str, out: Path, decomp: Path, symbols: Path, inputs: RomInputs) -> dict[str, str]:
    """Write the intermediate files for ``game`` into ``out``; return ``rom_sources.json``."""
    if game not in DEFAULT_OUT:
        raise ValueError(f"unknown game {game!r}")
    out, decomp, symbols = Path(out), Path(decomp), Path(symbols)

    from tools.rom.nds import RomFiles

    vanilla_files = RomFiles.open(read_checked(inputs.vanilla_rom, VANILLA_SHA1, "sha1"))
    if game == "kaizo":
        _kaizo_checks(inputs, decomp, symbols, vanilla_files)

    from tools.rom.readers.items import read_items
    from tools.rom.readers.learnsets import read_learnsets
    from tools.rom.readers.maps import trainer_start_conditions
    from tools.rom.readers.moves import read_moves
    from tools.rom.readers.personal import read_exp_tables, read_species
    from tools.rom.readers.trainers import read_trainers
    from tools.rom.symbols import SymbolMap

    smap = SymbolMap.from_map(symbols)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        charmap = extract_charmap(decomp)
        starts = trainer_start_conditions(vanilla_files, decomp, smap)
        tables = {
            "species": read_species(vanilla_files, decomp, charmap),
            "moves": read_moves(vanilla_files, decomp, charmap),
            "items": read_items(vanilla_files, decomp, smap, charmap),
            "learnsets": read_learnsets(vanilla_files, decomp),
            "exp_tables": read_exp_tables(vanilla_files),
            "trainers": read_trainers(vanilla_files, decomp, smap, charmap, starts),
        }
        for name, data in tables.items():
            _write(stage / f"{name}.json", data)
        fresh = stage / "decomp"
        extract_static(decomp, fresh)
        for name in REUSED_FILES:
            os.replace(fresh / f"{name}.json", stage / f"{name}.json")
        shutil.rmtree(fresh)
        sources = {f"{n}.json": "rom" for n in ROM_FILES}
        sources.update(
            {f"{n}.json": "decomp: vanilla ROM (decomp is the source)" for n in REUSED_FILES}
        )
        _write(stage / "rom_sources.json", sources)
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
    except RomInputError as e:
        print(f"error: {e}", file=__import__("sys").stderr)
        return 1
    for name, src in sorted(sources.items()):
        print(f"{name}: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
