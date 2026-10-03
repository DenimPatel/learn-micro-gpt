"""`tools/lint.py` and `tools/bench.py` must never call a subprocess unbounded.

## The bug these are for

`lint` is the first step of the improvement loop. Its `step()` called
`subprocess.run` with no `timeout=`, so a hung `npx`, `go vet` or `cargo clippy`
-- a network wait, a lock file, a prompt on stdin -- blocked the loop *before*
it produced a single line of output, and every subsequent step waited on a
process that was never going to finish. `compile_if_needed` in `bench.py` had
the same hole. `measure()` three functions below it already passed
`timeout=1800`, so the pattern was in the file and simply not applied where the
build happens.

## What these tests assert

That a timeout is *forwarded*, not what its value is: the value is a judgement
about slow machines, the presence is the correctness property. `subprocess.run`
is monkeypatched, so nothing here compiles or installs anything.
"""

from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from tools import bench, lint


def _unbounded_run_calls(path: Path) -> list[int]:
    """Line numbers of every `subprocess.run(...)` with no `timeout=` keyword.

    Parsed rather than grepped: the call sites here are multi-line and the
    interesting thing is a keyword in the call, which a text search gets wrong
    in both directions (comments mention `timeout`; a nested call's timeout
    would be attributed to the outer one).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    unbounded: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "attr", None) != "run" or getattr(func.value, "id", None) != "subprocess":
            continue
        if "timeout" not in {kw.arg for kw in node.keywords}:
            unbounded.append(node.lineno)
    return unbounded


class FakeCompletedProcess:
    """The two attributes `step()` reads after a run."""

    returncode = 0
    stdout = ""
    stderr = ""


class TestLintSubprocessesAreBounded(unittest.TestCase):
    def test_step_forwards_a_timeout(self) -> None:
        calls: list[dict] = []

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            calls.append({"argv": argv, **kwargs})
            return FakeCompletedProcess()

        with mock.patch.object(subprocess, "run", fake_run):
            lint.step("demo", ["true"])

        self.assertEqual(len(calls), 1)
        timeout = calls[0].get("timeout")
        self.assertIsInstance(timeout, (int, float))
        self.assertGreater(timeout, 0)

    def test_step_records_a_failure_when_the_timeout_fires(self) -> None:
        """A timed-out step has to be a lint failure, not a silent pass. The
        whole point of the loop reading `lint ok` is that it is true."""
        lint.failures.clear()
        self.addCleanup(lint.failures.clear)

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

        with mock.patch.object(subprocess, "run", fake_run):
            lint.step("hung tool", ["sleep", "forever"])

        self.assertIn("hung tool", lint.failures)

    def test_have_probes_with_a_timeout(self) -> None:
        calls: list[dict] = []

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return mock.Mock(returncode=0)

        with mock.patch.object(subprocess, "run", fake_run):
            self.assertTrue(lint._have("go"))

        self.assertIn("timeout", calls[0])

    def test_clippy_probe_forwards_a_timeout(self) -> None:
        calls: list[dict] = []

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return mock.Mock(returncode=0, stderr="")

        with mock.patch.object(subprocess, "run", fake_run):
            self.assertTrue(lint._clippy_available("cargo"))

        self.assertIn("timeout", calls[0])

    def test_every_subprocess_call_in_lint_passes_a_timeout(self) -> None:
        """A new unbounded call added later is the same bug again, so the check
        is over the module's source rather than over today's call sites."""
        self.assertEqual(_unbounded_run_calls(lint.REPO_ROOT / "tools" / "lint.py"), [])


class TestBenchBuildsAreBounded(unittest.TestCase):
    def test_compile_if_needed_forwards_a_timeout(self) -> None:
        calls: list[dict] = []

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            calls.append({"argv": argv, **kwargs})
            return mock.Mock(returncode=0, stderr="")

        measurement = bench.Measurement(
            lang="c",
            config="micro",
            command=[],
            cwd=bench.REPO_ROOT,
            build_flags="",
            build_mode="",
            build_command=["make", "microgpt"],
        )
        with mock.patch.object(subprocess, "run", fake_run):
            self.assertTrue(bench.compile_if_needed(measurement))

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].get("timeout"), (int, float))

    def test_a_measurement_with_no_build_command_runs_nothing(self) -> None:
        measurement = bench.Measurement(
            lang="python",
            config="micro",
            command=[],
            cwd=bench.REPO_ROOT,
            build_flags="",
            build_mode="",
        )
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("ran")):
            self.assertTrue(bench.compile_if_needed(measurement))

    def test_every_subprocess_call_in_bench_passes_a_timeout(self) -> None:
        self.assertEqual(_unbounded_run_calls(bench.REPO_ROOT / "tools" / "bench.py"), [])


if __name__ == "__main__":
    unittest.main()
