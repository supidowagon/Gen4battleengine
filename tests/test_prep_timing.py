"""The prep timing benchmark's parsing and command line (plan 4c Task 13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.prep_timing import TARGET_SECONDS, parse_record, prep_command


def test_parse_record(tmp_path):
    record = tmp_path / "prep_x.json"
    record.write_text(
        json.dumps(
            {
                "stats": {
                    "setups_screened": 120,
                    "playouts": 5000,
                    "seconds": 321.5,
                    "stopped": None,
                    "phase_seconds": [["teams", 100.0], ["finalists", 200.5]],
                },
                "setup": [{"label": "Starly"}],
                "ladder": [{"k": 0, "p": 0.75}, {"k": 1, "p": 0.9}],
            }
        )
    )
    got = parse_record(f"Setup:\nrun record: {record}\ncheckpoint: elsewhere\n")
    assert got == {
        "record": str(record),
        "seconds": 321.5,
        "phase_seconds": {"teams": 100.0, "finalists": 200.5},
        "setups_screened": 120,
        "playouts": 5000,
        "stopped": None,
        "setup": [{"label": "Starly"}],
        "rungs": {"0": 0.75, "1": 0.9},
    }
    with pytest.raises(ValueError, match="run record"):
        parse_record("Setup:\nnothing else\n")


def test_prep_command():
    cmd = prep_command(Path("a.sav"), 2)
    assert cmd[-6:] == ["--save", "a.sav", "--trainer", "roark", "--devices", "2"]
    assert "--devices" not in prep_command(Path("a.sav"), None)
    assert TARGET_SECONDS == 900.0
