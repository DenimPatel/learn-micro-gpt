"""Turn a direct-script run of a `tools/` module into an actionable error.

## The bug this exists for

Every tool here is meant to be run as a module:

    python3 -m tools.trace --all

Run as a plain script instead, `python3 tools/trace.py`, and two things break at
once:

1. **`tools/selectors.py` shadows the standard library.** A script run puts the
   script's own directory -- `tools/` -- at `sys.path[0]`. So when `subprocess`
   is imported it does `import selectors` and gets `tools/selectors.py`, which
   is this repository's code-anchor selector, not a selector for I/O readiness.
   The import dies inside `subprocess.py` with an `AttributeError` on
   `SelectSelector` that points nowhere near the real cause.
2. **`from tools import ...` cannot resolve.** `tools/` is on the path, but the
   repository root is not, so `tools` is not importable as a package at all.

Both are fixed by putting the *repository root* on the path and *not* `tools/`,
which is exactly what `python3 -m tools.X` does. Running the module form is also
what every `Makefile` target and every document already uses; only a learner
typing the path by hand hits this.

## What this does

`require_module_invocation` is called at the very top of each affected tool,
before the import that would fail, and aborts a script run with a message that
names the fix. Each tool loads this file *by path* rather than with
`import tools._entrypoint`, because the whole point is to run before `tools` is
importable at all.

Nothing here imports `subprocess` or `selectors`, so the guard itself cannot be
shadowed.
"""

from __future__ import annotations

import sys

#: The message every direct-script run prints. One string, one place, so the
#: tools cannot drift apart in how they explain the same problem.
_SCRIPT_RUN_MESSAGE = (
    "error: run this tool as a module, not as a script:\n"
    "  python3 -m tools.{tool}\n"
    "\n"
    "`python3 tools/{tool}.py` leaves tools/ on sys.path, where tools/selectors.py\n"
    "shadows the standard-library `selectors` module that subprocess imports, so the\n"
    "tool fails with an error that names neither this file nor this cause. The module\n"
    "form puts the repository root on sys.path instead. See CONTRIBUTING.md.\n"
)


def require_module_invocation(tool: str, package: str | None) -> None:
    """Abort if `tool` is being run as a script rather than as a module.

    `package` is the caller's `__package__`. Under `python3 -m tools.tool` it is
    `"tools"`; under `python3 tools/tool.py` it is `""` (or `None` when the
    loader is unusual). Anything falsy means a script run.
    """
    if package:
        return
    sys.stderr.write(_SCRIPT_RUN_MESSAGE.format(tool=tool))
    raise SystemExit(2)
