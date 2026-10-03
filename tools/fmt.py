#!/usr/bin/env python3
"""Every formatter this repository has, in place.

There is exactly one: prettier, for the visualizer. `tools/` is stdlib-only and
its formatting is hand-maintained -- black would be a dependency for a
repositories whose selling point is having none, and the files are read far more
often than they are edited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Run as `python3 -m tools.fmt`; see tools/_entrypoint.py. This must precede any
# `import subprocess`/`selectors`: a script run makes `tools/selectors.py`
# shadow the standard library, so the failure happens *at* that import.
_spec = importlib.util.spec_from_file_location(
    "_tools_entrypoint", Path(__file__).resolve().parent / "_entrypoint.py"
)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
_guard.require_module_invocation("fmt", __package__)

import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
VIS = REPO_ROOT / "visualizer"


def main() -> int:
    if not (VIS / "node_modules").is_dir():
        print("visualizer/node_modules is absent; run `make -C visualizer install` first", file=sys.stderr)
        return 1
    result = subprocess.run(
        ["npx", "prettier", "--write", "."], cwd=str(VIS), text=True
    )
    if result.returncode != 0:
        return result.returncode
    print("formatted the visualizer")
    print("tools/ is hand-formatted on purpose; see the module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
