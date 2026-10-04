"""Tests for tools/check_annotated_sync.py.

The module makes one claim: `reference/microgpt-annotated.py` carries the
reference's *code*, modulo comments, whitespace, line wrapping, and the one
documented split of the four optimizer hyperparameters. A check like that fails
in two directions, and both need a test:

  1. It must not fire on a clean tree. A checker that always fails is worthless,
     so the first test runs it against the real files and demands zero drifts.
     This is the loosest of the tests -- it only proves the tree is currently
     consistent -- but it is the one that would have caught the allowance bugs
     during development, so it stays.

  2. It must fire on real drift. The interesting part is *which* edits count.
     Comment-only edits, re-indentation and PEP 8 re-wrapping are not code and
     must pass; a changed constant, a swapped operator, a dropped statement and
     an edit inside a function must all fail. The fixtures below perturb the real
     annotated file by exact string replacement, so they stay honest as the file
     changes: if a replacement stops matching, the test fails rather than passing
     vacuously.

`compare()` is pure and takes two strings, so no sandbox or temporary tree is
needed. That is deliberate -- it is the same design that makes the checker
testable without copying the reference around.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from tools import check_annotated_sync

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _reference() -> str:
    return (REPO_ROOT / check_annotated_sync.REFERENCE_PATH).read_text(encoding="utf-8")


def _annotated() -> str:
    return (REPO_ROOT / check_annotated_sync.ANNOTATED_PATH).read_text(encoding="utf-8")


def _replace(source: str, old: str, new: str) -> str:
    """Replace `old` once, asserting it was there.

    A fixture built from a stale string literal would otherwise compare the
    *unmodified* file and pass, which is the failure mode these tests exist to
    avoid.
    """
    if old not in source:
        raise AssertionError(f"fixture string is not in the annotated file: {old!r}")
    return source.replace(old, new, 1)


def _where(drifts: list[check_annotated_sync.Drift]) -> list[str]:
    return [drift.where for drift in drifts]


class TestTheRealTreeIsConsistent(unittest.TestCase):
    def test_the_committed_annotated_file_matches_the_reference(self) -> None:
        drifts = check_annotated_sync.compare(_reference(), _annotated())
        self.assertEqual(
            drifts,
            [],
            "the committed tree has drift; the checker should be green on main",
        )

    def test_the_checker_walks_the_definitions_rather_than_nothing(self) -> None:
        # A checker that compared nothing would also report no drift. Pin the
        # count so a regression that silently stops walking the tree fails.
        reference = check_annotated_sync._normalise(ast.parse(_reference()))
        self.assertEqual(len(check_annotated_sync._nested_definitions(reference)), 21)


class TestFormattingIsNotDrift(unittest.TestCase):
    """Non-code differences must pass, or the checker becomes a style linter."""

    def test_a_trailing_comment_does_not_matter(self) -> None:
        annotated = _replace(_annotated(), "eps_adam = 1e-8", "eps_adam = 1e-8  # a new note")
        self.assertEqual(check_annotated_sync.compare(_reference(), annotated), [])

    def test_rewrapping_a_def_over_several_lines_does_not_matter(self) -> None:
        annotated = _replace(
            _annotated(), "def linear(x, w):", "def linear(\n    x,\n    w,\n):"
        )
        self.assertEqual(check_annotated_sync.compare(_reference(), annotated), [])

    def test_extra_blank_lines_do_not_matter(self) -> None:
        annotated = _replace(_annotated(), "def rmsnorm(x):", "\n\n\ndef rmsnorm(x):")
        self.assertEqual(check_annotated_sync.compare(_reference(), annotated), [])


class TestRealDriftFails(unittest.TestCase):
    """The other direction: perturb something the compiled program would see."""

    def test_a_changed_constant_in_a_function_fails(self) -> None:
        annotated = _replace(
            _annotated(),
            "return [e / total for e in exps]",
            "return [e / (total + 1e-9) for e in exps]",
        )
        drifts = check_annotated_sync.compare(_reference(), annotated)
        self.assertTrue(drifts, "a changed constant inside softmax must be drift")
        self.assertIn("FunctionDef softmax", _where(drifts))

    def test_a_swapped_operator_fails(self) -> None:
        annotated = _replace(_annotated(), "p.data -= lr_t", "p.data += lr_t")
        self.assertTrue(check_annotated_sync.compare(_reference(), annotated))

    def test_a_dropped_statement_fails(self) -> None:
        annotated = _replace(
            _annotated(), "import math     # math.log, math.exp\n", ""
        )
        self.assertTrue(check_annotated_sync.compare(_reference(), annotated))

    def test_a_changed_hyperparameter_names_itself(self) -> None:
        # The split-unpack allowance must not swallow this. Before the allowance
        # was located by name, one changed constant disabled the whole match and
        # the report only said "added 4x", which is why the reason is asserted.
        annotated = _replace(_annotated(), "eps_adam = 1e-8", "eps_adam = 1e-9")
        drifts = check_annotated_sync.compare(_reference(), annotated)
        self.assertTrue(drifts)
        joined = " ".join(f"{d.where} {d.detail}" for d in drifts)
        self.assertIn("eps_adam", joined)
        self.assertIn("1e-09", joined)


class TestTheKnownLimitation(unittest.TestCase):
    """A top-level *reorder* passes this comparison, and that is a real hole.

    Tested rather than only documented, because it is the one place where a
    green run of `compare()` does not mean "the code is the same program".
    Swapping the two statements below moves `random.shuffle(docs)` above the
    `docs = [...]` it consumes -- a genuine `NameError`. `compare()` reports no
    drift, and the module docstring says so plainly.

    The hole is closed in practice by the digest phase of `make
    verify-provenance`, which is unchanged when the two files are edited
    together. If this test ever starts failing, the hole was closed and the
    docstring needs updating.
    """

    def test_reordering_two_top_level_statements_is_invisible(self) -> None:
        before = 'docs = [w for w in tokens]\nrandom.shuffle(docs)\n'
        after = 'random.shuffle(docs)\ndocs = [w for w in tokens]\n'
        self.assertNotIn(before, _reference(), "fixture must be shaped like microgpt's tail")
        self.assertEqual(check_annotated_sync.compare(before, after), [])


class TestTheAllowanceCannotBeAbused(unittest.TestCase):
    """The one documented difference is matched structurally, not by string."""
    def test_a_comment_on_a_split_line_still_excuses_it(self) -> None:
        # The whole reason the allowance is located by name: the split lines
        # already carry comments, and an allowance keyed on raw text would stop
        # applying the moment someone edited one.
        annotated = _replace(
            _annotated(), "beta1 = 0.85", "beta1 = 0.85  # now with a longer comment"
        )
        self.assertEqual(check_annotated_sync.compare(_reference(), annotated), [])

    def test_a_reordered_split_is_still_excused_as_a_multiset(self) -> None:
        # Top-level statements are compared unordered, so moving the four
        # assignments around is formatting. The names and values are unchanged.
        annotated = _replace(
            _annotated(),
            "learning_rate = 0.01\nbeta1 = 0.85",
            "beta1 = 0.85\nlearning_rate = 0.01",
        )
        self.assertEqual(check_annotated_sync.compare(_reference(), annotated), [])


if __name__ == "__main__":
    unittest.main()
