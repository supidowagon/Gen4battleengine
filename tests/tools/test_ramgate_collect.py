"""tools/ops/ramgate.py is_rom_only on a real collection (pytest's own fixture defs, not fakes).
Pod-only (tools/ops/day/pod_only.txt): collecting the engine test modules imports the engine."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = '''
import os
import sys


def pytest_collection_finish(session):
    rg = sys.modules["nuzline_ramgate"]  # loaded by the root conftest.py
    with open(os.environ["RG_PROBE_OUT"], "w", encoding="utf-8") as out:
        for it in session.items:
            out.write(f"{rg.is_rom_only(it)} {it.nodeid}\\n")
'''

# Species parametrized directly: pytest gives `species` its own pseudo fixture def, so the ROM
# fixture `species` of tests/conftest.py is not in force and these engine tests are not ROM-only.
ENGINE_IDS = (
    "tests/engine/test_coverage.py::test_in_battle_form_changes_are_refused",
    "tests/engine/test_e1_batch_h.py::test_a_secondary_status_is_blocked_silently",
    "tests/engine/test_field_pass.py::test_sand_ability_immunities",
)
ROM_ID = "tests/tools/rom/test_nds.py::test_vanilla_layout"  # vanilla_files -> rom_inputs


def test_real_collection_classifies_direct_species_params_and_a_rom_test(tmp_path):
    (tmp_path / "rg_probe.py").write_text(PROBE, encoding="utf-8")
    out = tmp_path / "probe.txt"
    sep = ";" if sys.platform == "win32" else ":"
    env = {
        **os.environ,
        "RG_PROBE_OUT": str(out),
        "PYTHONPATH": sep.join(p for p in (str(tmp_path), os.environ.get("PYTHONPATH", "")) if p),
    }
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    run = subprocess.run(
        [*cmd, "-p", "rg_probe", *ENGINE_IDS, ROM_ID],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    rows = [ln.split(" ", 1) for ln in out.read_text(encoding="utf-8").splitlines()]
    for ident in ENGINE_IDS:
        flags = [flag for flag, node in rows if node.split("[")[0] == ident]
        assert flags and set(flags) == {"False"}, (ident, flags)
    assert [flag for flag, node in rows if node == ROM_ID] == ["True"]
