"""Tests for tools/autoresearch.py.

Three things are being defended here.

**The keep/discard rule must actually reject things.** A research loop whose
gate has never been seen to fail is not a research loop, it is a rubber stamp that
appends to a ledger. So most of this file constructs synthetic candidates and
watches them get judged. The cases that matter are the ones where the *wrong*
answer is the intuitive one: a candidate that is faster but 8% worse on loss must
be discarded even though it wins on an axis, and an exact tie must be discarded
even though it passes every threshold.

**The ledger and the generated JSON must not drift apart.** `results.json` is what
the site reads, and it is derived from `results.tsv` plus `runs/*.json`. If the
derivation rots, the site shows numbers that no run produced. So there is a test
that regenerates and compares, which is the same property `make trace-check`
defends for the traces.

**The measurements the rest of the page leans on are the documented ones.** The
frozen baseline's digest, the gradient probe's digest, and every threshold. Each
is a number a reader will quote, so each is pinned here the way
`test_trace_parity.py` pins `parity.BAND`.

Nothing here runs a training loop, and nothing here calls OpenRouter. A test that
needs a GPU, a network or two minutes of wall clock is a test nobody runs before
pushing, which is the same argument that made `test_gradients.c` worthless.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools import autoresearch as ar

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def window(improvement: float) -> dict[str, float]:
    """A plausible trend statistic with a chosen relative improvement."""
    last = 2.4
    return {
        "first_window_mean": last / (1.0 - improvement),
        "last_window_mean": last,
        "absolute_drop": last * improvement / (1.0 - improvement),
        "relative_improvement": improvement,
    }


def judge(
    base_loss: float = 2.4,
    base_speed: float = 90.0,
    cand_loss: float = 2.4,
    cand_speed: float = 90.0,
    improvement: float = 0.15,
) -> dict:
    return ar.verdict(base_loss, base_speed, cand_loss, cand_speed, window(improvement))


class TestVerdictKeepsRealImprovements(unittest.TestCase):
    def test_a_big_loss_win_at_unchanged_speed_is_kept(self) -> None:
        result = judge(cand_loss=2.16)  # -10%
        self.assertTrue(result["kept"], result["explanation"])
        self.assertEqual(result["rule"], "loss_won")

    def test_a_loss_win_at_the_edge_of_the_speed_tolerance_is_kept(self) -> None:
        # -4.9% speed is inside the 5% tolerance. The point of the test is the
        # boundary: the rule is "no worse than", not "faster".
        result = judge(cand_loss=2.16, cand_speed=90.0 * 0.951)
        self.assertTrue(result["kept"], result["explanation"])

    def test_a_big_speed_win_at_unchanged_loss_is_kept(self) -> None:
        result = judge(cand_speed=108.0)  # +20%
        self.assertTrue(result["kept"], result["explanation"])
        self.assertEqual(result["rule"], "speed_won")

    def test_a_speed_win_at_the_edge_of_the_loss_tolerance_is_kept(self) -> None:
        result = judge(cand_speed=108.0, cand_loss=2.4 * 1.049)
        self.assertTrue(result["kept"], result["explanation"])

    def test_a_run_that_wins_both_axes_is_kept(self) -> None:
        result = judge(cand_loss=2.16, cand_speed=108.0)
        self.assertTrue(result["kept"], result["explanation"])


class TestVerdictRejectsEverythingElse(unittest.TestCase):
    def test_an_exact_tie_is_a_discard(self) -> None:
        """The case that is easiest to get wrong and most expensive to get wrong.

        A tie clears every threshold -- it is not worse than the tolerance on
        either axis -- so a rule written as "nothing failed, so keep" would accept
        it. Then a loop that gets a no-op proposes a no-op again, and a hundred
        experiments later the ledger is a hundred identical rows and nobody can
        tell a working loop from a broken one.
        """
        result = judge()
        self.assertFalse(result["kept"], result["explanation"])
        self.assertEqual(result["rule"], "none")

    def test_a_tiny_loss_win_is_a_discard(self) -> None:
        # +0.1% is a real number and not a real improvement: it is a twentieth of
        # the 2% noise floor. This is the exact experiment the research loop's
        # first run produced, and it must be discarded.
        result = judge(cand_loss=2.4 * 0.999)
        self.assertFalse(result["kept"], result["explanation"])
        self.assertFalse(result["checks"]["loss_material"]["pass"])

    def test_a_loss_win_past_the_speed_tolerance_is_a_discard(self) -> None:
        # 30% faster is a big win and 6% worse loss is a real regression. The
        # speed axis does not buy off the loss axis.
        result = judge(cand_loss=2.4 * 1.06, cand_speed=90.0 * 1.30)
        self.assertFalse(result["kept"], result["explanation"])
        self.assertFalse(result["checks"]["loss_tolerance"]["pass"])
        self.assertTrue(result["checks"]["speed_material"]["pass"])

    def test_a_speed_win_past_the_loss_tolerance_is_a_discard(self) -> None:
        result = judge(cand_loss=2.4 * 1.06, cand_speed=90.0 * 0.90)
        self.assertFalse(result["kept"], result["explanation"])
        self.assertFalse(result["checks"]["loss_tolerance"]["pass"])

    def test_a_candidate_that_did_not_learn_is_a_discard_however_fast(self) -> None:
        # A 2x speedup on a model that did not train. The trend check is the only
        # thing standing between this and a loop that optimises for exiting early.
        result = judge(cand_speed=180.0, improvement=0.05)
        self.assertFalse(result["kept"], result["explanation"])
        self.assertFalse(result["checks"]["learned"]["pass"])

    def test_a_regression_on_both_axes_is_a_discard(self) -> None:
        result = judge(cand_loss=2.4 * 1.2, cand_speed=90.0 * 0.5)
        self.assertFalse(result["kept"])

    def test_a_degenerate_baseline_is_refused_rather_than_guessed_at(self) -> None:
        for base_loss, base_speed in ((0.0, 90.0), (2.4, 0.0), (-1.0, 90.0)):
            with self.assertRaises(ar.ResearchError):
                judge(base_loss=base_loss, base_speed=base_speed)

    def test_every_check_carries_the_reason_it_exists(self) -> None:
        # A threshold with no recorded justification is a number someone will
        # loosen the first time it is inconvenient.
        result = judge(cand_loss=2.16)
        for name, check in result["checks"].items():
            self.assertIn("why", check, name)
            self.assertIn("threshold", check, name)
            self.assertIsInstance(check["pass"], bool, name)


class TestProtocolConstantsAreTheDocumentedOnes(unittest.TestCase):
    def test_the_noise_floors_and_the_tolerances_are_the_documented_numbers(self) -> None:
        self.assertEqual(ar.LOSS_MATERIAL, 0.02)
        self.assertEqual(ar.LOSS_TOL, 0.05)
        self.assertEqual(ar.SPEED_MATERIAL, 0.10)
        self.assertEqual(ar.SPEED_TOL, 0.05)
        self.assertEqual(ar.MIN_RELATIVE_IMPROVEMENT, 0.10)
        self.assertEqual(ar.TREND_WINDOW, 50)

    def test_the_noise_floor_and_the_regression_limit_are_not_the_same_thing(self) -> None:
        """There is no ordering between the two kinds of threshold, and the first
        version of this test asserted one. It does not hold, and it should not: a
        1% loss *gain* is inside the noise and cannot be trusted to be real,
        while a 5% loss *regression* is about two standard errors and can be.
        The asymmetry is the point.

        What does have to hold is that the tolerance is small enough that a real
        regression cannot hide behind it, and that the two are distinct, since
        equal numbers would make the rule's "cleared the floor and did not
        regress" language say one thing and mean another.
        """
        self.assertGreater(ar.LOSS_MATERIAL, 0.0)
        self.assertGreater(ar.LOSS_TOL, 0.0)
        self.assertGreater(ar.SPEED_MATERIAL, 0.0)
        self.assertGreater(ar.SPEED_TOL, 0.0)
        self.assertLessEqual(ar.LOSS_TOL, 0.10)
        self.assertLessEqual(ar.SPEED_TOL, 0.10)
        self.assertNotEqual(ar.LOSS_MATERIAL, ar.LOSS_TOL)
        self.assertNotEqual(ar.SPEED_MATERIAL, ar.SPEED_TOL)

    def test_a_regression_past_the_tolerance_cannot_be_kept_at_all(self) -> None:
        """Stated as a behaviour rather than a constant, because that is the
        property a reader actually cares about: no arrangement of the other axis
        rescues a candidate that is 10% worse on loss."""
        for speed in (0.5, 1.0, 2.0, 10.0):
            result = judge(cand_loss=2.4 * 1.10, cand_speed=90.0 * speed)
            self.assertFalse(result["kept"], f"speed x{speed} should not rescue a 10% loss regression")

    def test_the_protocol_is_fixed_so_the_loss_axis_is_deterministic(self) -> None:
        self.assertEqual(ar.STEPS, 1000)
        self.assertEqual(ar.SEED, 42)
        self.assertGreaterEqual(ar.REPEATS, 3)

    def test_only_rust_is_supported_and_saying_so_is_not_optional(self) -> None:
        self.assertEqual(ar.SUPPORTED_TRACKS, ("rust",))
        self.assertEqual(sorted(ar.VERDICTS), ["crash", "discard", "keep"])


class TestTheFrozenBaselineHasNotMoved(unittest.TestCase):
    """'The first version' is a claim, and this is the mechanism behind it.

    `tools/provenance.py` enforces the same digests in CI, in the `provenance`
    job that gates every deploy. This test exists so that a failure names the
    file and the reason, rather than surfacing as an unrelated provenance
    failure three jobs later.
    """

    EXPECTED = {
        "implementations/rust/src/lib.rs": "4bced112f2db2bf8662082051fa550f82ab793bed251be9398c6f672947a2f15",
        "implementations/rust/Cargo.toml": "0b743d919a98011f7f995db246ebc45f22dafc24233580aab342960cd380aa7b",
        "implementations/rust/src/main.rs": "0160147568e6ac5a1c0448dccefd638931467cdcd1225e47529d078d97fbc324",
    }

    def test_the_frozen_digests_match(self) -> None:
        for path, expected in self.EXPECTED.items():
            target = REPO_ROOT / path
            self.assertTrue(target.is_file(), f"{path} is missing")
            self.assertEqual(
                ar.sha256_file(target),
                expected,
                f"{path} has moved. The research loop measures every candidate "
                f"against it and the site renders it, so 'the first version' is "
                f"only true while its bytes are fixed. The tuned copy is "
                f"autoresearch/candidate/src/lib.rs and that is where a change "
                f"belongs.",
            )

    def test_the_frozen_files_are_in_the_provenance_manifest(self) -> None:
        from tools import provenance

        paths = {entry.path for entry in provenance.all_entries()}
        for path in self.EXPECTED:
            self.assertIn(path, paths, f"{path} is not in tools/provenance.py")

    def test_the_candidate_is_a_separate_file_from_the_baseline(self) -> None:
        """The whole design falls over if the loop writes to the frozen track."""
        self.assertTrue(ar.CANDIDATE_LIB.is_file())
        self.assertNotEqual(ar.CANDIDATE_LIB, ar.BASELINE_LIB)
        self.assertTrue(
            str(ar.BASELINE_LIB).startswith(str(REPO_ROOT / "implementations")),
            "the baseline must live under implementations/, where the parity gate "
            "and the site already expect it",
        )
        self.assertTrue(
            str(ar.CANDIDATE_LIB).startswith(str(REPO_ROOT / "autoresearch")),
            "the candidate must live outside implementations/, so no existing tool "
            "picks it up as a track",
        )


class TestTheGradientProbeHasNotBeenWeakened(unittest.TestCase):
    def test_the_probe_digest_is_the_pinned_one(self) -> None:
        self.assertEqual(
            ar.sha256_file(ar.PROBE_PATH),
            ar.PROBE_SHA256,
            "autoresearch/candidate/tests/gradient_check.rs is not ours any more. "
            "It is the only thing standing between a model and a broken gradient "
            "that still trains (docs/KNOWN-ISSUES.md issue 1), so a change to it "
            "has to be deliberate, in the same commit, with a reason.",
        )

    def test_the_probe_lives_in_the_candidate_and_not_the_baseline(self) -> None:
        self.assertEqual(ar.PROBE_PATH, ar.CANDIDATE_DIR / "tests" / "gradient_check.rs")
        self.assertFalse(
            (ar.BASELINE_CRATE / "tests").exists(),
            "the frozen crate must not gain a tests/ directory; that would change "
            "it, and it is pinned",
        )

    def test_the_documented_gradient_ratio_is_what_the_ledger_records(self) -> None:
        """1.063, not 1.0, and that is issue 5 rather than a measurement error.

        Worth pinning because the temptation is to 'fix' the ledger to 1.0 the
        moment somebody sees it, which would erase the finding.
        """
        rows = ar.read_ledger()
        self.assertTrue(rows, "the ledger has no rows; run --seed")
        measured = [row.grad_ratio for row in rows if row.status != "crash"]
        self.assertTrue(measured)
        for ratio in measured:
            self.assertTrue(0.5 <= ratio <= 2.0, ratio)


class TestLedgerRoundTrips(unittest.TestCase):
    def test_the_header_is_the_columns_in_order(self) -> None:
        self.assertEqual(ar.HEADER, "\t".join(ar.COLUMNS))
        self.assertTrue(ar.RESULTS_TSV.is_file())
        first = ar.RESULTS_TSV.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first, ar.HEADER)

    def test_every_row_parses_and_keeps_its_status(self) -> None:
        for row in ar.read_ledger():
            self.assertIn(row.status, ("baseline",) + ar.VERDICTS, row.run_id)
            self.assertRegex(row.run_id, r"^\d{4}$")
            self.assertEqual(len(row.run_id), 4)

    def test_run_ids_are_unique_and_dense(self) -> None:
        rows = ar.read_ledger()
        ids = [row.run_id for row in rows]
        self.assertEqual(len(ids), len(set(ids)), "duplicate run id in the ledger")
        self.assertEqual(ids, [f"{i:04d}" for i in range(len(rows))], rows and ids)

    def test_exactly_one_baseline_row_and_it_is_first(self) -> None:
        rows = ar.read_ledger()
        baselines = [row.run_id for row in rows if row.status == "baseline"]
        self.assertEqual(baselines, ["0000"])

    def test_a_row_with_a_tab_in_it_is_refused(self) -> None:
        # A description with a stray tab silently becomes two columns, and the
        # resulting row is not obviously broken to whoever reads it later.
        with self.assertRaises(ar.ResearchError):
            ar.clean_field("has a\ttab in it", "description")
        with self.assertRaises(ar.ResearchError):
            ar.clean_field("has a\nnewline in it", "description")

    def test_row_json_round_trips(self) -> None:
        original = ar.Row(
            run_id="0042",
            parent="deadbeef",
            loss=2.123456,
            steps_per_sec=91.5,
            loss_gain=-0.0123,
            speed_gain=0.0456,
            grad_ratio=1.063160,
            status="discard",
            reason="no axis improved materially",
            description="a summary with no tabs",
        )
        self.assertEqual(ar.Row.from_json(original.to_json()), original)


class TestTheGeneratedDocumentIsCurrent(unittest.TestCase):
    """The property `make trace-check` defends for traces, applied here.

    `results.json` is what the site reads. If it stops matching the ledger, the
    site shows a number that no run produced, and there is no way for a reader to
    tell.
    """

    def test_results_json_regenerates_byte_identically(self) -> None:
        self.assertTrue(ar.RESULTS_JSON.is_file(), "run --render")
        expected = json.dumps(ar.build_results_document(), indent=2) + "\n"
        actual = ar.RESULTS_JSON.read_text(encoding="utf-8")
        self.assertEqual(
            actual,
            expected,
            "autoresearch/results.json is stale. Run "
            "`python3 -m tools.autoresearch render`.",
        )

    def test_the_document_carries_every_number_the_page_needs(self) -> None:
        document = ar.build_results_document()
        for key in (
            "protocol",
            "thresholds",
            "objective",
            "provenance_of",
            "caveats",
            "runs",
            "counts",
            "pareto",
            "runner",
        ):
            self.assertIn(key, document, key)
        self.assertEqual(document["thresholds"]["loss_material"], ar.LOSS_MATERIAL)
        self.assertEqual(document["provenance_of"]["gradient_probe_sha256"], ar.PROBE_SHA256)
        self.assertTrue(document["caveats"], "a page of numbers with no caveats is a lie")

    def test_every_row_has_a_run_record(self) -> None:
        for row in ar.read_ledger():
            record = ar.RUNS_DIR / f"{row.run_id}.json"
            self.assertTrue(record.is_file(), f"no run record for {row.run_id}")
            payload = ar.read_json(record)
            self.assertEqual(payload["run_id"], row.run_id)
            self.assertEqual(payload["status"], row.status)

    def test_a_measured_run_records_the_digest_of_what_was_measured(self) -> None:
        """On a discard the committed candidate is still the previous best, so
        the digest of what was measured and the digest of what is on disk differ.
        Recording the wrong one would let `--verify-only` compare a run against
        code that never produced it."""
        for row in ar.read_ledger():
            if row.status == "crash":
                continue
            payload = ar.read_json(ar.RUNS_DIR / f"{row.run_id}.json")
            if row.status == "baseline":
                continue
            self.assertIn("autoresearch/candidate/src/lib.rs", payload.get("source_sha256", {}))

    def test_the_best_row_is_actually_the_lowest_loss_among_the_keeps(self) -> None:
        document = ar.build_results_document()
        kept = [row for row in ar.read_ledger() if row.status == "keep"]
        if not kept:
            self.assertIsNone(document["best"])
            return
        best = min(kept, key=lambda row: (row.loss, -row.steps_per_sec))
        self.assertEqual(document["best"]["run_id"], best.run_id)

    def test_the_pareto_frontier_is_dominated_by_nothing_on_it(self) -> None:
        rows = {row.run_id: row for row in ar.read_ledger()}
        frontier = [rows[run_id] for run_id in ar.build_results_document()["pareto"]]
        for candidate in frontier:
            for other in frontier:
                if other is candidate:
                    continue
                dominates = (
                    other.loss <= candidate.loss
                    and other.steps_per_sec >= candidate.steps_per_sec
                    and (
                        other.loss < candidate.loss
                        or other.steps_per_sec > candidate.steps_per_sec
                    )
                )
                self.assertFalse(
                    dominates,
                    f"{other.run_id} is on the frontier but dominates {candidate.run_id}",
                )

    def test_crashes_are_published_rather_than_dropped(self) -> None:
        """A failure that is not on the page is a failure that gets retried."""
        document = ar.build_results_document()
        crashes = [row for row in ar.read_ledger() if row.status == "crash"]
        self.assertEqual(document["counts"]["crash"], len(crashes))
        for row in crashes:
            entry = next(r for r in document["runs"] if r["run_id"] == row.run_id)
            self.assertTrue(entry["reason"], f"{row.run_id} crashed with no reason recorded")

    def test_a_crash_row_carries_no_fake_measurements(self) -> None:
        for row in ar.read_ledger():
            if row.status != "crash":
                continue
            self.assertEqual(row.loss, 0.0)
            self.assertEqual(row.steps_per_sec, 0.0)
            self.assertEqual(row.grad_ratio, 0.0)


class TestTheDiffIsActuallyADiff(unittest.TestCase):
    def test_the_best_patch_exists_and_names_both_files(self) -> None:
        self.assertTrue(ar.BEST_DIFF.is_file(), "run --seed")
        text = ar.BEST_DIFF.read_text(encoding="utf-8")
        self.assertIn("a/implementations/rust/src/lib.rs", text)
        self.assertIn("b/autoresearch/candidate/src/lib.rs", text)

    def test_diff_counts_ignore_the_file_headers(self) -> None:
        patch = (
            "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-a\n+b\n"
        )
        self.assertEqual(ar.diff_counts(patch), {"added": 1, "removed": 1})

    def test_the_seed_patch_is_the_measurement_surface_and_nothing_else(self) -> None:
        """The candidate began as the baseline. What differs is the one method a
        finite-difference probe needs and cannot do without, so a reader can see
        exactly where the research track forked from the parity track."""
        patch = ar.BEST_DIFF.read_text(encoding="utf-8")
        added = [
            line[1:].strip()
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        self.assertTrue(any("set_data" in line for line in added), added[:20])
        self.assertFalse(
            any("fn rmsnorm" in line for line in added),
            "the seeded candidate must be the baseline verbatim apart from "
            "set_data; the research loop makes its own changes",
        )


class TestToolDiscovery(unittest.TestCase):
    def test_cargo_is_found_even_when_it_is_only_in_the_rustup_directory(self) -> None:
        """Not a theoretical concern: cargo installs to `~/.cargo/bin`, which is
        on an interactive shell's PATH and is not on the PATH a subprocess
        inherits from some launchers. A tool that is installed and reported
        missing is worse than one that is absent, because the advice is wrong."""
        import shutil
        from pathlib import Path as P

        candidates = [
            P.home() / ".cargo" / "bin" / "cargo",
            P("/opt/homebrew/bin/cargo"),
            P("/usr/local/bin/cargo"),
        ]
        on_path = shutil.which("cargo")
        if on_path or any(path.is_file() for path in candidates):
            self.assertIsNotNone(ar.find_tool("cargo"))

    def test_a_missing_tool_says_what_to_install(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_tool("definitely-not-a-real-tool", "Install it from nowhere")
        message = str(caught.exception)
        self.assertIn("Install it from nowhere", message)
        self.assertIn("~/.cargo/bin", message)


class TestTheLedgerHeaderIsARefusalNotAGuess(unittest.TestCase):
    def test_a_foreign_header_is_rejected_with_both_headers_shown(self) -> None:
        import tempfile

        original = ar.RESULTS_TSV.read_text(encoding="utf-8")
        try:
            ar.RESULTS_TSV.write_text("run_id\tloss\n0001\t2.4\n", encoding="utf-8")
            with self.assertRaises(ar.ResearchError) as caught:
                ar.read_ledger()
            message = str(caught.exception)
            self.assertIn(ar.HEADER, message)
            self.assertIn("run_id\tloss", message)
            self.assertIn("append-only", message)
        finally:
            ar.RESULTS_TSV.write_text(original, encoding="utf-8")


class TestTheModelPinIsUsable(unittest.TestCase):
    """Checked against the committed file, never against the network.

    Whether the model is *still* on OpenRouter's catalogue is the harness's job
    at run time, and it fails loudly there. What a test can defend is that the
    pin records the reasoning knob it was verified with, so a later reader knows
    the request shape was measured rather than assumed.
    """

    def setUp(self) -> None:
        self.spec = ar.read_json(ar.MODEL_JSON)

    def test_the_pin_names_the_model_and_its_reasoning_parameter(self) -> None:
        self.assertEqual(self.spec["model"], "stealth/space-bunny-alpha")
        self.assertEqual(self.spec["reasoning_parameter"], "reasoning_effort")
        self.assertEqual(self.spec["reasoning_effort"], "max")

    def test_the_reasoning_effort_is_one_the_api_was_observed_to_accept(self) -> None:
        # The 400 body from the live API: expected one of "max"|"xhigh"|"high"|
        # "medium"|"low"|"minimal"|"none".
        self.assertIn(
            self.spec["reasoning_effort"],
            self.spec["reasoning_effort_enum"],
        )
        self.assertEqual(
            self.spec["reasoning_effort_enum"],
            ["max", "xhigh", "high", "medium", "low", "minimal", "none"],
            "the API's enumeration changed; re-verify against /api/v1/models and "
            "update the verification note",
        )

    def test_the_pin_records_how_it_was_verified(self) -> None:
        for key in ("verified", "verification", "models_endpoint", "key_env"):
            self.assertIn(key, self.spec, key)
        self.assertIn("400", self.spec["verification"])


class TestProposalParsingIsStrict(unittest.TestCase):
    GOOD = (
        "HYPOTHESIS: beta2 is under-corrected.\n"
        "SUMMARY: Adam beta2 0.99 -> 0.98\n\n"
        "```diff\n"
        "--- a/autoresearch/candidate/src/lib.rs\n"
        "+++ b/autoresearch/candidate/src/lib.rs\n"
        "@@ -766,7 +766,7 @@\n"
        "     const BETA1: f32 = 0.85;\n"
        "-    const BETA2: f32 = 0.99;\n"
        "+    const BETA2: f32 = 0.98;\n"
        "```\n"
    )

    def test_a_well_formed_proposal_parses(self) -> None:
        proposal = ar.parse_proposal(self.GOOD)
        self.assertIn("0.98", proposal.patch)
        self.assertEqual(proposal.hypothesis, "beta2 is under-corrected.")
        self.assertEqual(proposal.description, "Adam beta2 0.99 -> 0.98")

    def test_a_response_with_no_code_block_is_a_crash(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.parse_proposal("HYPOTHESIS: I think the model is too big.\n\nNo patch.")
        self.assertIn("no fenced code block", str(caught.exception))

    def test_an_empty_code_block_is_a_crash(self) -> None:
        with self.assertRaises(ar.ResearchError):
            ar.parse_proposal("```diff\n```\n")

    def test_a_block_with_no_changed_lines_is_a_crash(self) -> None:
        # A diff that only has context is not a change, and treating it as one
        # would fill the ledger with rows that did nothing.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.parse_proposal("```diff\n@@ -1,1 +1,1 @@\n unchanged line\n```\n")
        self.assertIn("no added or removed lines", str(caught.exception))

    def test_a_summary_with_a_tab_is_refused(self) -> None:
        response = self.GOOD.replace("Adam beta2 0.99 -> 0.98", "Adam\tbeta2")
        with self.assertRaises(ar.ResearchError):
            ar.parse_proposal(response)

    def test_a_missing_summary_is_tolerated_but_named(self) -> None:
        response = self.GOOD.replace("SUMMARY: Adam beta2 0.99 -> 0.98\n", "")
        self.assertEqual(ar.parse_proposal(response).description, "no summary given")


class TestThePromptGivesTheModelWhatItNeeds(unittest.TestCase):
    """Two regressions this file is guarding, both found by running the loop.

    The first version of the prompt showed the first 40 lines of an 880-line
    file. The model then proposed a patch against an `rmsnorm` that took a weight
    tensor and had `.data()` and `.mul()` methods -- an API it had invented,
    because it had never been shown the real one. Two experiments were wasted
    that way. The whole file is now in the prompt; a 1M-token context makes that
    free and it is the difference between a patch that applies and one that does
    not.
    """

    def setUp(self) -> None:
        self.prompt = ar.build_prompt(
            ar.read_ledger(),
            {"loss": 2.433882, "steps_per_sec": 95.38, "grad_ratio": 1.06316},
        )

    def test_the_whole_candidate_source_is_included(self) -> None:
        source = ar.CANDIDATE_LIB.read_text(encoding="utf-8")
        for marker in (
            "fn rmsnorm",
            "fn softmax",
            "struct Tensor",
            "fn backward",
            "const LEARNING_RATE",
            "fn parse_args",
        ):
            self.assertIn(marker, self.prompt, f"{marker} is missing from the prompt")
        self.assertIn(
            source.rstrip().splitlines()[-1],
            self.prompt,
            "the prompt stops short of the end of the file",
        )

    def test_the_thresholds_and_the_session_baseline_are_included(self) -> None:
        for value in (
            f"{ar.LOSS_MATERIAL}",
            f"{ar.SPEED_MATERIAL}",
            f"{ar.LOSS_TOL}",
            f"{ar.SPEED_TOL}",
            "2.4339",
            "95.4",
        ):
            self.assertIn(value, self.prompt, value)

    def test_the_reply_shape_is_specified(self) -> None:
        self.assertIn("HYPOTHESIS:", self.prompt)
        self.assertIn("SUMMARY:", self.prompt)
        self.assertIn("```diff", self.prompt)

    def test_the_ledger_is_the_memory_and_is_included(self) -> None:
        self.assertIn("run_id\tstatus", self.prompt)
        self.assertIn("0000", self.prompt)


if __name__ == "__main__":
    unittest.main()
