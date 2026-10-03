"""Tests for tools/trace.py and tools/parity.py.

Two things are being defended here.

**Trace instrumentation must not change the run.** The reference is byte-pinned,
and a trace is only evidence if it describes the run the unpinned file would
have produced. So one test asserts that the instrumented source's loss curve is
identical to the committed one, digit for digit.

**The parity gate must actually reject things.** An earlier draft of the
specification -- "loss strictly decreasing, within 10% of the reference at
steps 50 and 200" -- would have failed *the reference itself*. The gate that
replaced it is only meaningful if it fails for the right reasons, so most of
this file is about constructing bad runs and watching them get caught.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import parity, trace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACE_DIR = trace.TRACES_DIR / "python" / "micro"


def load_losses() -> list[float]:
    return [
        json.loads(line)["loss"]
        for line in (TRACE_DIR / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_rows(name: str) -> list[dict]:
    path = TRACE_DIR / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestTraceArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = json.loads((TRACE_DIR / "meta.json").read_text(encoding="utf-8"))

    def test_all_documented_files_exist(self) -> None:
        for name in ("meta.json", "steps.jsonl", "attn.jsonl", "probs.jsonl", "samples.jsonl"):
            self.assertTrue((TRACE_DIR / name).is_file(), name)

    def test_meta_records_the_digest_of_what_was_actually_run(self) -> None:
        # If this drifts from the reference's real digest, every number in the
        # site is describing code that no longer exists.
        import hashlib

        actual = hashlib.sha256((REPO_ROOT / "reference" / "microgpt.py").read_bytes()).hexdigest()
        self.assertEqual(self.meta["reference"]["sha256"], actual)
        self.assertEqual(
            self.meta["reference"]["sha256"],
            json.loads((REPO_ROOT / "content" / "index.json").read_text())["reference"]["sha256"],
        )

    def test_meta_records_the_dataset_digest(self) -> None:
        import hashlib

        actual = hashlib.sha256((REPO_ROOT / "data" / "input.txt").read_bytes()).hexdigest()
        self.assertEqual(self.meta["dataset"]["sha256"], actual)

    def test_hyperparameters_were_parsed_not_asserted(self) -> None:
        self.assertEqual(self.meta["hyperparameters"]["n_embd"], 16)
        self.assertEqual(self.meta["hyperparameters"]["n_head"], 4)
        self.assertEqual(self.meta["hyperparameters"]["n_layer"], 1)
        self.assertEqual(self.meta["hyperparameters"]["block_size"], 16)
        # `head_dim = n_embd // n_head` is a definition, not a literal; it has to
        # be evaluated or every shape annotation in the docs is unverified.
        self.assertEqual(self.meta["hyperparameters"]["head_dim"], 4)

    def test_trace_is_within_the_size_budget(self) -> None:
        total = sum(
            (TRACE_DIR / name).stat().st_size
            for name in ("meta.json", "steps.jsonl", "attn.jsonl", "probs.jsonl", "samples.jsonl")
        )
        self.assertLess(total, 2 * 1024 * 1024, f"trace is {total} bytes")

    def test_loss_curve_is_complete(self) -> None:
        rows = load_rows("steps.jsonl")
        self.assertEqual(len(rows), self.meta["steps"])
        self.assertEqual([r["step"] for r in rows], list(range(len(rows))))

    def test_every_loss_is_finite_and_positive(self) -> None:
        for value in load_losses():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1e6)

    def test_attention_rows_are_distributions(self) -> None:
        rows = load_rows("attn.jsonl")
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(step=row["step"], pos=row["pos"], head=row["head"]):
                self.assertAlmostEqual(sum(row["weights"]), 1.0, places=5)
                self.assertTrue(all(w >= 0 for w in row["weights"]))

    def test_attention_is_causal_by_construction(self) -> None:
        # The trace must never contain a row longer than the causal window. If it
        # does, the "mask" is not structural and the whole claim in
        # multi-head-attention.md is wrong.
        for row in load_rows("attn.jsonl"):
            self.assertEqual(len(row["weights"]), row["pos"] + 1)

    def test_position_zero_is_trivially_certain(self) -> None:
        # With one position visible there is nothing to choose between, so the
        # softmax of a single logit is exactly 1. If this ever fails, something is
        # wrong upstream of anything interesting.
        for row in load_rows("attn.jsonl"):
            if row["pos"] == 0:
                self.assertEqual(row["weights"], [1.0])

    def test_probability_rows_sum_to_one_and_are_ranked(self) -> None:
        for row in load_rows("probs.jsonl"):
            with self.subTest(step=row["step"], pos=row["pos"]):
                self.assertTrue(row["top"])
                values = [entry["p"] for entry in row["top"]]
                self.assertEqual(values, sorted(values, reverse=True))
                self.assertLessEqual(sum(values), 1.0 + 1e-6)

    def test_samples_are_non_empty_strings(self) -> None:
        samples = load_rows("samples.jsonl")
        self.assertEqual(len(samples), 20)
        for row in samples:
            self.assertTrue(row["text"])
            self.assertNotIn("BOS", row["text"])


class TestInstrumentationIsNonIntrusive(unittest.TestCase):
    def test_injection_is_minimal(self) -> None:
        source = trace.build_instrumented_source(
            (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8")
        )
        calls = [line.strip() for line in source.split("\n") if line.strip().startswith("_emit(")]
        # One for attn_weights, two for probs (training and inference).
        self.assertEqual(len(calls), 3)
        for call in calls:
            # A single argument each: the counters come from the calling frame, so
            # the rewrite cannot accidentally shadow a local.
            self.assertEqual(call.count(","), 1, call)

    def test_instrumented_source_still_parses(self) -> None:
        import ast

        source = trace.build_instrumented_source(
            (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8")
        )
        ast.parse(source)

    def test_instrumentation_does_not_perturb_the_loss_curve(self) -> None:
        """The committed trace must describe the run the real file would produce.

        A re-run costs ~105 s, so this is opt-in via MICROGPT_SLOW_TESTS=1. It is
        the one test that would catch instrumentation changing behaviour -- for
        instance by consuming randomness before the model is initialised.
        """
        import os

        if not os.environ.get("MICROGPT_SLOW_TESTS"):
            self.skipTest("set MICROGPT_SLOW_TESTS=1 to re-run the reference")
        import shutil
        import tempfile

        work = Path(tempfile.mkdtemp(prefix="microgpt-test-"))
        try:
            shutil.copy(REPO_ROOT / "data" / "input.txt", work / "input.txt")
            stdout, _payload, _elapsed = trace.run_reference(1000, 42, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        steps, samples, _total = trace.parse_stdout(stdout)
        self.assertEqual([row["loss"] for row in steps], load_losses())
        self.assertEqual([row["text"] for row in samples], [r["text"] for r in load_rows("samples.jsonl")])


class TestSmoothing(unittest.TestCase):
    def test_ema_preserves_length_and_stays_in_range(self) -> None:
        values = [3.0, 1.0, 2.0, 5.0]
        smoothed = trace.ema(values, 0.05)
        self.assertEqual(len(smoothed), len(values))
        self.assertTrue(all(min(values) - 1 <= s <= max(values) + 1 for s in smoothed))

    def test_ema_removes_noise_without_removing_the_trend(self) -> None:
        import random

        rng = random.Random(0)
        noisy = [3.0 - 0.001 * i + rng.gauss(0, 0.4) for i in range(1000)]
        smooth = trace.ema(noisy, 0.05)

        def sd(xs: list[float]) -> float:
            mean = sum(xs) / len(xs)
            return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5

        def step_to_step(xs: list[float]) -> list[float]:
            return [b - a for a, b in zip(xs, xs[1:])]

        # The noise we want gone is the step-to-step jitter. The overall spread of
        # the smoothed series also contains the real trend, so asserting on that
        # would be asserting the wrong thing.
        self.assertLess(sd(step_to_step(smooth)), sd(step_to_step(noisy)) / 3)
        # And the trend has to survive: the whole point of smoothing is to keep
        # the signal and drop the variance, not the other way round.
        self.assertLess(smooth[-1], smooth[0] - 0.9)

    def test_window_mean(self) -> None:
        self.assertAlmostEqual(trace.window_mean([1.0, 2.0, 3.0, 4.0], 0, 2), 1.5)
        self.assertAlmostEqual(trace.window_mean([1.0, 2.0, 3.0, 4.0], 2, 99), 3.5)

    def test_loss_statistics_reports_the_noise(self) -> None:
        stats = trace.loss_statistics(load_losses())
        # These are measured facts about this run, and the concept documents
        # quote them. If instrumentation ever changes the run, this is where it
        # shows up first.
        self.assertEqual(stats["steps"], 1000)
        self.assertAlmostEqual(stats["mean"], 2.4517, places=3)
        self.assertAlmostEqual(stats["stdev"], 0.3920, places=3)
        self.assertEqual(stats["upward_moves"], 500)


class TestParityGateRejectsBadRuns(unittest.TestCase):
    """A gate that has never been seen to fail is not a gate."""

    def setUp(self) -> None:
        self.reference = parity.load_committed_trace("python", "micro")
        assert self.reference is not None

    def check(self, losses: list[float]) -> parity.ParityResult:
        return parity.compare(self.reference, parity.TrackRun("test", "bad", losses))

    def test_a_healthy_run_passes(self) -> None:
        result = self.check(load_losses())
        self.assertTrue(result.ok(), result.failures)

    def test_a_flat_run_fails_the_trend_check(self) -> None:
        result = self.check([2.5] * 1000)
        self.assertFalse(result.ok())
        self.assertTrue(any("windowed loss" in f for f in result.failures), result.failures)

    def test_a_diverging_run_is_caught_as_non_finite(self) -> None:
        losses = list(load_losses())
        losses[500] = float("inf")
        result = self.check(losses)
        self.assertFalse(result.ok())
        self.assertTrue(any("non-finite" in f for f in result.failures), result.failures)

    def test_a_nan_run_is_caught(self) -> None:
        losses = list(load_losses())
        losses[10] = float("nan")
        result = self.check(losses)
        self.assertFalse(result.ok())
        self.assertTrue(any("non-finite" in f for f in result.failures), result.failures)

    def test_a_run_that_learns_the_wrong_amount_fails_the_band(self) -> None:
        # Train correctly but end up systematically 30% higher: a plausible bug
        # (a learning rate that is too small, a gradient that is scaled wrong).
        # It improves, so only the band check can catch it.
        losses = [value * 1.30 for value in load_losses()]
        result = self.check(losses)
        self.assertFalse(result.ok())
        self.assertTrue(any("from the reference" in f for f in result.failures), result.failures)

    def test_a_constant_offset_within_the_band_passes(self) -> None:
        # +5% everywhere: different PRNG, float32 vs float64, and still inside
        # the band. This is what the band is for -- it admits real numerical
        # difference and rejects a different learning outcome.
        losses = [value * 1.05 for value in load_losses()]
        self.assertTrue(self.check(losses).ok())

    def test_a_shuffled_document_order_fails(self) -> None:
        # The same loss distribution attached to the wrong documents. Raw
        # per-step values are within noise, so only the smoothed comparison has
        # a chance -- and the smoothing is seeded by the first value, so this is
        # exactly the case that makes the point.
        import random

        shuffled = list(load_losses())
        random.Random(7).shuffle(shuffled)
        result = self.check(shuffled)
        self.assertFalse(result.ok())

    def test_band_and_trend_constants_are_the_documented_ones(self) -> None:
        self.assertEqual(parity.BAND, 0.10)
        self.assertEqual(parity.MIN_RELATIVE_IMPROVEMENT, 0.10)
        self.assertEqual(parity.EMA_ALPHA, 0.05)

    def test_the_reference_itself_would_fail_the_original_specification(self) -> None:
        """Documented, so nobody re-derives it and reintroduces the mistake.

        The original gate demanded a strictly decreasing loss. The reference
        rises on exactly half its steps, so that gate was unsatisfiable.
        """
        losses = load_losses()
        upward = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        self.assertEqual(upward, 500)
        self.assertGreater(upward, len(losses) // 4)

    def test_a_raw_single_step_band_would_be_inside_the_noise(self) -> None:
        """The other half of the argument, also measured rather than asserted."""
        import statistics

        losses = load_losses()
        spread = (max(losses) - min(losses)) / statistics.mean(losses)
        self.assertGreater(spread, parity.BAND, "a 10% band on a raw step is inside the noise")


class TestTraceDirectivesPointAtRealRows(unittest.TestCase):
    """`make validate` must reject a `:::trace` the trace cannot answer.

    A known kind and in-range integers are not enough. The app looks the
    request up in a JSONL file, and a miss silently renders the widget's empty
    state inside a concept that promised a picture -- a failure nothing else
    in the pipeline notices.
    """

    def setUp(self) -> None:
        from tools import concepts

        self.concepts = concepts.load_all(REPO_ROOT)
        self.index = json.loads(
            (REPO_ROOT / "content" / "index.json").read_text(encoding="utf-8")
        )

    def test_current_directives_all_resolve_to_recorded_rows(self) -> None:
        from tools import validate_concepts

        report = validate_concepts.Report()
        validate_concepts.check_bodies(self.concepts, self.index, report)
        self.assertEqual(report.errors, [], "\n".join(report.errors))

    def test_an_unrecorded_position_is_rejected(self) -> None:
        """`block_size` is 16 and position 15 is never traced; asking for it is
        exactly the kind of request that used to pass silently."""
        from tools import concepts, validate_concepts

        report = validate_concepts.Report()
        block = concepts.Block(
            kind="directive",
            name="trace",
            args={"kind": "attention", "step": "500", "pos": "15", "head": "0"},
            line_no=1,
        )
        defaults = {"step": 500, "pos": 15, "head": 0}
        validate_concepts.check_trace_directive(
            block, "multi-head-attention", defaults, report
        )
        self.assertEqual(len(report.errors), 1, report.errors)
        message = report.errors[0]
        self.assertIn("multi-head-attention", message)
        self.assertIn("attn.jsonl", message)
        self.assertIn("pos=15", message)

    def test_a_recorded_kind_at_an_unrecorded_step_is_rejected(self) -> None:
        """`softmax` has no heads; asking for `head=0` on it can never match."""
        from tools import concepts, validate_concepts

        report = validate_concepts.Report()
        block = concepts.Block(
            kind="directive",
            name="trace",
            args={"kind": "samples", "step": "200"},
            line_no=1,
        )
        validate_concepts.check_trace_directive(block, "inference-sampling", {"step": 200}, report)
        self.assertEqual(len(report.errors), 1, report.errors)
        self.assertIn("samples.jsonl", report.errors[0])
        self.assertIn("step=200", report.errors[0])

    def test_defaults_are_applied_before_the_lookup(self) -> None:
        """A directive with no args still asks for a row: the app's defaults."""
        from tools import concepts, validate_concepts

        index = validate_concepts._trace_index()
        defaults = validate_concepts.TRACE_DEFAULTS["attention"]
        self.assertTrue(index.has("attention", dict(defaults)))
        self.assertFalse(index.has("attention", {"step": 0, "pos": 15, "head": 0}))

    def test_kinds_without_recorded_data_are_not_checked(self) -> None:
        """`tokens` reads no file and `shapes` renders nothing; neither can fail."""
        from tools import concepts, validate_concepts

        report = validate_concepts.Report()
        for kind in ("tokens", "shapes"):
            block = concepts.Block(
                kind="directive", name="trace", args={"kind": kind}, line_no=1
            )
            validate_concepts.check_trace_directive(block, "tokenization", {}, report)
        self.assertEqual(report.errors, [])

    def test_a_loss_curve_ignores_step_and_cannot_be_wrong(self) -> None:
        """`LossChart` takes no props: it draws every row, so a stray `step=` on a
        loss curve is ignored rather than rendering an empty state. Failing it
        would be a false positive on a directive the app renders perfectly."""
        from tools import concepts, validate_concepts

        report = validate_concepts.Report()
        block = concepts.Block(
            kind="directive",
            name="trace",
            args={"kind": "loss-curve", "step": "2000"},
            line_no=1,
        )
        validate_concepts.check_trace_directive(block, "cross-entropy-loss", {"step": 2000}, report)
        self.assertEqual(report.errors, [], report.errors)

    def test_the_wiring_is_exercised_through_check_bodies(self) -> None:
        """The check is only real if `check_bodies` actually calls it.

        Written against a concept *object* rather than by calling
        `check_trace_directive` directly, because a test that calls the helper
        itself still passes when the call site is deleted -- which is exactly
        the regression this guards against.
        """
        from tools import concepts, validate_concepts

        block = concepts.Block(
            kind="directive",
            name="trace",
            args={"kind": "attention", "step": "0", "pos": "15", "head": "0"},
            line_no=1,
        )
        concept = concepts.Concept(
            id="multi-head-attention",
            path=Path("content/concepts/multi-head-attention.md"),
            frontmatter={"title": "Multi-head attention"},
            body=":::trace kind=attention step=0 pos=15 head=0",
            blocks=[block],
        )
        report = validate_concepts.Report()
        validate_concepts.check_bodies([concept], self.index, report)
        self.assertEqual(len(report.errors), 1, report.errors)
        self.assertIn("pos=15", report.errors[0])

    def test_a_missing_trace_file_is_reported_as_missing_not_as_bad_content(self) -> None:
        """A trace that was never recorded must not be misreported as a directive
        that asks for a row which does not exist -- those need different fixes."""
        from tools import concepts, trace, validate_concepts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "traces" / "python" / "micro").mkdir(parents=True)
            index = trace.load_trace_index(root=root)
        self.assertTrue(index.missing_files(), "an empty traces dir reports every file missing")
        self.assertIn("missing", index.missing_files()["attention"])

        report = validate_concepts.Report()
        block = concepts.Block(
            kind="directive", name="trace", args={"kind": "attention"}, line_no=1
        )
        with mock.patch.object(validate_concepts, "_TRACE_INDEX", index):
            validate_concepts.check_trace_directive(block, "multi-head-attention", {}, report)
        self.assertEqual(len(report.errors), 1, report.errors)
        self.assertIn("missing", report.errors[0])
        self.assertNotIn("does not contain", report.errors[0])


if __name__ == "__main__":
    unittest.main()
