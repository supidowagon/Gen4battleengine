import os

# Set by the root conftest.py (tools/ops/ramgate.py) for a local engine-free test run, and so by
# every child process of it. The pod never sets it.
if os.environ.get("NUZLINE_ENGINE_BLOCKED"):
    raise ImportError(
        "no engine on the PC: send at most 5 ids to the dispatcher's quick lane, "
        "run name q-<stream>-<sha7> (docs/process.md section 5)"
    )
