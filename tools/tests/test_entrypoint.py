"""A direct-script run of a tool must fail with the fix, not with the shadow bug.

## The bug these are for

`python3 tools/trace.py` puts `tools/` at `sys.path[0]`. There, `import
selectors` inside the standard library's `subprocess` resolves to
`tools/selectors.py` -- this repository's code-anchor selector -- and the tool
dies inside `subprocess.py` with `AttributeError: module 'selectors' has no
attribute 'SelectSelector'`. The message names neither `tools/trace.py` nor the
collision. Separately, `from tools import ...` cannot resolve at all, because
the repository root is not on the path in a script run. Both are avoided by
`python3 -m tools.trace`, which is what every `Makefile` target already uses.

## What these tests assert

Three properties, each a way the fix could rot:

1. Running a guarded tool as a script exits non-zero and its stderr names the
   module form (`python3 -m tools.<name>`) and the shadowing file
   (`tools/selectors.py`). Message-only, so it holds however the guard is coded.
2. The guard actually runs *before* the import that would break: the script run
   must not print the `selectors` traceback. A guard placed one line too late
   passes (1) only by accident of a later failure; this pins the ordering.
3. Every tool that imports `subprocess` or `from tools import ...` carries the
   guard, so the next such tool cannot be added without it.

`validate_concepts`, `check_provenance` and `concepts` are checked here too even
though they do not import `subprocess`: they fail on `from tools import` and get
the same message. Four tools (`gen_anchors`, `gen_provenance_doc`, `provenance`,
`render_content`) run under a plain script today and are deliberately **not**
guarded -- aborting them would be a regression -- so they are outside `GUARDED`.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOLS_DIR.parent

#: The tools that must refuse a direct-script run. Kept literal rather than
#: derived: the whole point is that adding one is a deliberate act.
GUARDED = (
    "autoresearch",
    "bench",
    "fmt",
    "lint",
    "parity",
    "trace",
    "validate_concepts",
    "check_provenance",
    "concepts",
)


def _tool_modules() -> list[str]:
    """Every `tools/*.py` that is a runnable tool, not a support module or test."""
    names: list[str] = []
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.stem.startswith("_") or path.stem.startswith("test_"):
            continue
        names.append(path.stem)
    return names


def _imports_module(path: Path, module: str) -> bool:
    """True if `path` has a top-level `import <module>`.

    Parsed, not grepped: the module docstrings in this tree discuss `subprocess`
    at length, and a text search would match the prose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module for alias in node.names):
                return True
    return False


class ScriptRunIsRefused(unittest.TestCase):
    """Property 1 and 2, exercised by really running each tool."""

    def _run_as_script(self, tool: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOLS_DIR / f"{tool}.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_each_guarded_tool_aborts_and_names_the_fix(self) -> None:
        for tool in GUARDED:
            with self.subTest(tool=tool):
                result = self._run_as_script(tool)
                self.assertNotEqual(result.returncode, 0, msg=result.stdout)
                self.assertIn(f"python3 -m tools.{tool}", result.stderr)
                self.assertIn("tools/selectors.py", result.stderr)

    def test_the_shadow_traceback_never_reaches_the_user(self) -> None:
        """The guard is useless if it fires after `import subprocess` already ran."""
        for tool in GUARDED:
            with self.subTest(tool=tool):
                result = self._run_as_script(tool)
                self.assertNotIn("SelectSelector", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_the_module_form_is_accepted(self) -> None:
        """The guard must not fire when the tool is run the blessed way.

        Importing the tool as `tools.<name>` runs its top-level guard with
        `__package__ == "tools"`, which is the case the guard must let through.
        It is *not* run as a script here on purpose: several tools (`bench`,
        `fmt`, `lint`) have no argument parsing, so invoking them -- even with
        `--help` -- executes `main()` and rewrites files (`benchmarks/
        results.json`, the visualizer tree). The module *import* exercises the
        guard's decision without those side effects.
        """
        for tool in GUARDED:
            with self.subTest(tool=tool):
                result = subprocess.run(
                    [sys.executable, "-c", f"import tools.{tool}"],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertNotIn("run this tool as a module", result.stderr)
                self.assertNotIn("SelectSelector", result.stderr)


class EveryRiskyToolCarriesTheGuard(unittest.TestCase):
    """Property 3, so the next tool with the same imports cannot skip it.

    The automated rule is narrow on purpose: **any tool that imports
    `subprocess` must be guarded**, because that import is where `tools/selectors.py`
    does its damage. `validate_concepts`, `check_provenance` and `concepts` do not
    import `subprocess` -- they fail on `from tools import` -- and are held to the
    contract by the explicit `GUARDED` list and the script-run tests above
    instead of by a rule that would have to model `sys.path` ordering.
    """

    def test_every_tool_importing_subprocess_is_guarded(self) -> None:
        imports_subprocess = {
            name
            for name in _tool_modules()
            if _imports_module(TOOLS_DIR / f"{name}.py", "subprocess")
        }
        self.assertTrue(imports_subprocess, "expected some tool to import subprocess")
        self.assertTrue(
            imports_subprocess <= set(GUARDED),
            msg=f"unguarded but imports subprocess: "
            f"{sorted(imports_subprocess - set(GUARDED))}",
        )

    def test_guarded_tools_reference_the_entrypoint(self) -> None:
        for tool in GUARDED:
            with self.subTest(tool=tool):
                source = (TOOLS_DIR / f"{tool}.py").read_text(encoding="utf-8")
                self.assertIn("require_module_invocation", source)
                self.assertIn("_entrypoint.py", source)


class GuardHelperItself(unittest.TestCase):
    """The helper's contract, tested without a subprocess."""

    def test_importable_by_path_without_the_tools_package(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_tools_entrypoint_probe", TOOLS_DIR / "_entrypoint.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_module_invocation("trace", "tools")  # returns cleanly

    def test_aborts_when_package_is_empty_or_none(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_tools_entrypoint_probe2", TOOLS_DIR / "_entrypoint.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for package in (None, ""):
            with self.subTest(package=package):
                with mock.patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        module.require_module_invocation("bench", package)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
