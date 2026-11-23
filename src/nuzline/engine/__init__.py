import os
import sys

# The local test gate (root conftest.py, tools/ops/ramgate.py; docs/process.md section 5).
# NUZLINE_ENGINE_BLOCKED: set by the gate for a local engine-free test run, and so for every child
# process of it ("" and "0" count as unset). The pod never sets it.
if os.environ.get("NUZLINE_ENGINE_BLOCKED", "") not in ("", "0"):
    raise ImportError(
        "NUZLINE_ENGINE_BLOCKED is set (local test gate, tools/ops/ramgate.py): no engine tests on "
        "the PC; send at most 5 ids to the quick lane, run name q-STREAM-SHA7 (docs/process.md "
        "section 5). Outside a local pytest run, unset it."
    )
# A pytest on the PC that never loaded the gate (--noconftest, --confcutdir, -p no:<root conftest>,
# also through PYTEST_ADDOPTS): the root conftest registers the gate as `nuzline_ramgate`. The pods
# are Linux, so NUZLINE_ON_POD means nothing here; the product never imports _pytest.
if "_pytest" in sys.modules and sys.platform == "win32" and "nuzline_ramgate" not in sys.modules:
    raise ImportError(
        "pytest runs without the local test gate (the root conftest.py, tools/ops/ramgate.py, was "
        "not loaded: --noconftest, --confcutdir, -p no:<conftest> or PYTEST_ADDOPTS): no engine "
        "tests on the PC; send at most 5 ids to the quick lane, run name q-STREAM-SHA7 "
        "(docs/process.md section 5)."
    )
