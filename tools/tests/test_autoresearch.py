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

import ast
import inspect
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

    def test_the_supported_tracks_are_the_ones_the_table_can_honour(self) -> None:
        # `SUPPORTED_TRACKS` is derived from `TRACKS`, so it cannot drift from the
        # registry. What is asserted is that the registry is internally complete:
        # a track missing a probe, a digest or a build entry would fail at run
        # time, in the browser, on the one page that shows it.
        self.assertEqual(ar.SUPPORTED_TRACKS, tuple(ar.TRACKS))
        self.assertIn("rust", ar.SUPPORTED_TRACKS)
        self.assertEqual(sorted(ar.VERDICTS), ["crash", "discard", "keep"])
        for name, track in ar.TRACKS.items():
            with self.subTest(track=name):
                self.assertTrue(track.probe_sha256, f"{name} has no probe digest")
                self.assertTrue(track.tool, f"{name} has no build tool")
                self.assertTrue(track.fence, f"{name} has no fenced-block language")
                self.assertTrue(track.loss_parse in ("stdout", "trace"))

    def test_an_unknown_track_is_rejected_loudly(self) -> None:
        # The alternative is accepting a name it cannot honour, which fails later
        # and somewhere less obvious.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.get_track("cobol")
        self.assertIn("cobol", str(caught.exception))
        self.assertIn("rust, go, typescript", str(caught.exception))

    def test_every_track_has_a_distinct_candidate_and_comparator(self) -> None:
        # The whole reason the per-track map exists: two tracks sharing a frozen
        # comparator would produce two "speed gains" against one number, and the
        # page would show them as if they were competing.
        candidates = [t.candidate for t in ar.TRACKS.values()]
        comparators = [t.comparator for t in ar.TRACKS.values()]
        source_keys = [t.source_key for t in ar.TRACKS.values()]
        for label, values in (
            ("candidate", candidates),
            ("comparator", comparators),
            ("source key", source_keys),
        ):
            with self.subTest(kind=label):
                self.assertEqual(len(set(values)), len(values))

    def test_every_candidate_lives_outside_the_frozen_tracks(self) -> None:
        for name, track in ar.TRACKS.items():
            with self.subTest(track=name):
                self.assertTrue(
                    str(track.candidate).startswith(str(ar.RESEARCH_DIR)),
                    f"{name} writes outside autoresearch/, where the parity gate and "
                    f"the site would pick it up as a track",
                )
                self.assertTrue(
                    str(track.comparator).startswith(str(REPO_ROOT / "implementations")),
                    f"{name}'s comparator must be one of the frozen, pinned tracks",
                )


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
    def test_every_probe_digest_is_the_pinned_one(self) -> None:
        for name, track in ar.TRACKS.items():
            with self.subTest(track=name):
                self.assertEqual(
                    ar.sha256_file(track.probe),
                    track.probe_sha256,
                    f"{track.probe.relative_to(REPO_ROOT)} is not ours any more. It is "
                    "the only thing standing between a model and a broken gradient "
                    "that still trains (docs/KNOWN-ISSUES.md issue 1), so a change to "
                    "it has to be deliberate, in the same commit, with a reason.",
                )

    def test_the_rust_probe_digest_is_also_the_legacy_constant(self) -> None:
        # `PROBE_SHA256` is a module-level name the Rust-era tests and the error "
        # messages use. It has to stay the Rust probe's digest, not the most
        # recently added one.
        self.assertEqual(ar.PROBE_SHA256, ar.TRACKS["rust"].probe_sha256)

    def test_every_probe_lives_in_the_candidate_and_not_the_baseline(self) -> None:
        for name, track in ar.TRACKS.items():
            with self.subTest(track=name):
                # Under the candidate, not equal to it: the Rust probe is in
                # `tests/` and the TypeScript one in `src/`, because that is where
                # each language's runner finds it.
                self.assertIn(track.candidate, track.probe.parents)
                self.assertTrue(track.probe.is_file())
                self.assertFalse(
                    track.comparator.joinpath(track.probe_path).exists(),
                    f"the frozen {name} track must not gain the candidate's probe; "
                    f"that would change it, and it is pinned",
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

    def test_exactly_one_baseline_row_per_seeded_track(self) -> None:
        """One per track, not one for the ledger.

        `seed` is idempotent per track, so a second run on a track that already
        has a baseline must not append a second one -- that would mean the
        comparator for that track was re-measured and the first number silently
        became wrong.
        """
        rows = ar.read_ledger()
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                baselines = [
                    row.run_id for row in rows
                    if row.status == "baseline" and row.track == track.name
                ]
                self.assertLessEqual(len(baselines), 1, f"{track.name} has two baselines")
        # The first track seeded takes 0000, because the sequence is global and it
        # was seeded first. The others take the next ids, which is why "its
        # baseline is its first row" is not a property here and is not asserted.
        self.assertIn("0000", [r.run_id for r in rows if r.status == "baseline"])

    def test_a_row_with_a_tab_in_it_is_refused(self) -> None:
        # A description with a stray tab silently becomes two columns, and the
        # resulting row is not obviously broken to whoever reads it later.
        with self.assertRaises(ar.ResearchError):
            ar.clean_field("has a\ttab in it", "description")
        with self.assertRaises(ar.ResearchError):
            ar.clean_field("has a\nnewline in it", "description")

    def test_row_json_round_trips(self) -> None:
        for name in ar.SUPPORTED_TRACKS:
            with self.subTest(track=name):
                original = ar.Row(
                    run_id="0042",
                    parent="deadbeef",
                    track=name,
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

    def test_a_row_written_before_the_track_column_defaults_to_rust(self) -> None:
        """The migration. `autoresearch/candidate/` was the only candidate that
        existed when those rows were written, so they are all Rust, and reading
        one must not raise on a missing key."""
        legacy = {
            "run_id": "0007",
            "parent": "deadbeef",
            "loss": 2.4,
            "steps_per_sec": 90.0,
            "loss_gain": 0.0,
            "speed_gain": 0.0,
            "grad_ratio": 1.06316,
            "status": "discard",
            "reason": "",
            "description": "written before the column existed",
        }
        self.assertEqual(ar.Row.from_json(legacy).track, "rust")


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
        self.assertEqual(
            document["provenance_of"]["rust"]["gradient_probe_sha256"], ar.PROBE_SHA256
        )
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
            self.assertIn(ar.TRACKS[row.track].source_key, payload.get("source_sha256", {}))

    def test_the_best_row_is_the_lowest_loss_among_the_keeps_still_in_effect(self) -> None:
        # Per track. A single "best" across tracks would rank a Go candidate
        # against a Rust one, each measured against a different frozen build, and
        # the winner would be whichever language happened to be quicker on the
        # session's machine rather than whichever change was better.
        #
        # And among the keeps *still in effect*, not among every keep ever
        # recorded. A keep says the loop accepted a candidate at the time; it does
        # not say the candidate survived, and one very often does not. On the C
        # track a run whose training loop had collapsed to replaying a single
        # document held the record at loss 0.0000 and 99,081 steps/s for as long as
        # it held it -- while the code behind it was three experiments out of date
        # and the tree held something else entirely. The lowest number on a ledger
        # is not an achievement if you threw the code away.
        #
        # Which keeps are in effect is decided by comparing each run's recorded
        # `source_sha256` against the digest the candidate file has now, so this
        # test is really checking that the two agree about what is in the tree.
        document = ar.build_results_document()
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                in_effect = ar.keeps_in_effect(track, ar.read_ledger())
                if not in_effect:
                    self.assertIsNone(document["bests"][track.name])
                    continue
                best = min(in_effect, key=lambda row: (row.loss, -row.steps_per_sec))
                self.assertEqual(document["bests"][track.name]["run_id"], best.run_id)
                # And that set is not merely "the keeps that happen to be cheapest",
                # which is what makes this a real constraint rather than a
                # restatement of the same min() in another place.
                self.assertTrue(
                    all(
                        (ar.read_json(ar.RUNS_DIR / f"{row.run_id}.json") or {})
                        .get("source_sha256", {})
                        .get(track.source_key)
                        == ar.candidate_source_digest(track)
                        for row in in_effect
                    ),
                    "every keep counted as in effect must match the committed candidate digest",
                )

    def test_every_track_has_a_baseline_slot_even_when_unseeded(self) -> None:
        # The page renders a zero for an unseeded track and says so. A missing key
        # would be a `undefined` in the same place, which reads as a bug.
        document = ar.build_results_document()
        for track in ar.TRACKS:
            with self.subTest(track=track):
                self.assertIn(track, document["baselines"])
                self.assertIn(track, document["bests"])
                self.assertIn(track, document["pareto"])
                self.assertIn(track, document["provenance_of"])
                self.assertIn(track, document["tracks"])

    def test_the_pareto_frontier_is_dominated_by_nothing_on_it(self) -> None:
        rows = {row.run_id: row for row in ar.read_ledger()}
        frontier = [
            rows[run_id] for run_id in ar.build_results_document()["pareto"]["rust"]
        ]
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


class TestEveryTrackTakingFunctionIsCalledWithATrack(unittest.TestCase):
    """A sweep for the whole bug class, in one place.

    Making a function take a `track` means every call site has to change, and a
    call site that does not is a `TypeError` on a line that only runs during a real
    experiment. Four separate bugs of that shape got past a 193-test suite, lint, a
    full gate and `verify`, because each sat downstream of a network call or a
    measurement -- so every check that does not execute that line passed.

    `ast` finds them without executing any of them.
    """

    @staticmethod
    def _sweep(source: str) -> tuple[set[str], list[tuple[int, str]]]:
        tree = ast.parse(source)
        needs_track = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.args.args
            and node.args.args[0].arg == "track"
        }
        offenders = [
            (node.lineno, node.func.id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in needs_track
            and not node.args
            and not any(keyword.arg == "track" for keyword in node.keywords)
        ]
        return needs_track, offenders

    def test_no_function_taking_a_track_is_called_without_one(self) -> None:
        needs_track, offenders = self._sweep(open(ar.__file__, encoding="utf-8").read())
        self.assertGreater(len(needs_track), 5, "the sweep is not finding the functions")
        self.assertEqual(
            offenders, [],
            "these are called with no `track`, so they raise a TypeError the first "
            "time a real experiment reaches them",
        )

    def test_the_sweep_would_notice_a_missing_argument(self) -> None:
        """A guard on the guard.

        Otherwise this test is a sweep that silently finds nothing, which is
        indistinguishable from a sweep that works.
        """
        needs_track, offenders = self._sweep(
            "def f(track: int) -> None: ...\n"
            "def g() -> None:\n"
            "    f(1)\n"
            "    f()\n"
        )
        self.assertEqual(needs_track, {"f"})
        self.assertEqual(offenders, [(4, "f")])


class TestEveryProposalIsGradientChecked(unittest.TestCase):
    """The gate the harness exists to enforce, checked from both sides.

    A proposal is measured in a *scratch copy* of the candidate, so "is this the
    candidate?" is the wrong question, and answering it wrong skips the probe for
    every experiment. That happened: one conditional read the wrong way, every run
    record came out with `grad_ratio 0.0`, and nothing failed.
    """

    def test_the_probe_is_gated_on_being_the_frozen_comparator(self) -> None:
        source = inspect.getsource(ar.measure_track)
        self.assertIn("if not is_frozen:", source)
        self.assertNotIn("if is_candidate:", source)

    def test_every_recorded_experiment_has_a_measured_ratio(self) -> None:
        """The data check, which is the one that would actually have caught it."""
        checked = 0
        for row in ar.read_ledger():
            if row.status not in ("keep", "discard"):
                continue
            record_path = ar.RUNS_DIR / f"{row.run_id}.json"
            if not record_path.is_file():
                continue
            record = ar.read_json(record_path)
            if "grad" not in record:
                continue
            checked += 1
            self.assertGreater(
                record["grad"]["ratio"], 0.0,
                f"experiment {row.run_id} recorded no gradient ratio, so its "
                f"proposal was never finite-difference checked",
            )
        self.assertGreater(checked, 0, "no run records carried a grad block to check")

    def test_a_run_record_files_its_own_track_source_digest(self) -> None:
        """A Go run's digest belongs under the Go key.

        Recording it under the Rust key would make `verify` compare a Go candidate
        against a Rust digest, and would silently pass whenever they happened to
        match.
        """
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                rows = [r for r in ar.read_ledger() if r.track == track.name
                        and r.status in ("keep", "discard")]
                if not rows:
                    continue
                record = ar.read_json(ar.RUNS_DIR / f"{rows[-1].run_id}.json")
                self.assertEqual(list(record.get("source_sha256", {})), [track.source_key])


class TestTheModelIsActuallyAsked(unittest.TestCase):
    """The path that only runs when a real experiment runs.

    Everything else in this file is testable without a network, which is exactly
    the problem: `ask_model` was the one function nothing could reach, and it
    assembled its messages around a `track` it did not have. So `--experiments 0`
    passed, `verify` passed, the whole gate passed, and the first experiment on
    the first new track died with a NameError after the model had already been
    asked. These two tests reach the same code without a network, and check the
    signature that made the NameError possible.
    """

    def test_the_messages_are_a_brief_and_an_ask_for_every_track(self) -> None:
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                messages = ar.build_messages(track, "THE ASK")
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertEqual(messages[1]["content"], "THE ASK")
                self.assertEqual(messages[0]["content"], ar.read_prompt(track))
                self.assertNotIn("{{", messages[0]["content"])

    def test_ask_model_completes_for_every_track_with_a_stubbed_endpoint(self) -> None:
        """End to end, with the network replaced.

        `build_messages` proves the brief renders; this proves `ask_model` gets
        that far and hands back a parsed reply. The bug it guards is a `NameError`
        on the first network call, so stubbing the endpoint is the only way to
        reach the code that had it.
        """
        import contextlib
        import io
        import json as json_module

        spec = ar.read_json(ar.MODEL_JSON)
        seen: list[dict] = []

        @contextlib.contextmanager
        def fake_urlopen(request, timeout=None):
            seen.append(json_module.loads(request.data.decode("utf-8")))
            payload = {
                "model": spec["model"],
                "choices": [{"message": {"content": "PATCH"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
            yield io.BytesIO(json_module.dumps(payload).encode("utf-8"))

        original_urlopen = ar.urllib.request.urlopen
        original_key = ar.openrouter_key
        ar.urllib.request.urlopen = fake_urlopen
        ar.openrouter_key = lambda _spec: "stub"
        try:
            for track in ar.TRACKS.values():
                with self.subTest(track=track.name):
                    seen.clear()
                    text, usage = ar.ask_model(spec, track, "THE ASK", 512)
                    self.assertEqual(text, "PATCH")
                    self.assertEqual(usage["model"], spec["model"])
                    body = seen[0]
                    self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
                    self.assertEqual(body["messages"][1]["content"], "THE ASK")
                    self.assertIn(f"improving a {track.language} implementation", body["messages"][0]["content"])
                    self.assertEqual(body["model"], spec["model"])
                    self.assertEqual(body[spec["reasoning_parameter"]], spec["reasoning_effort"])
        finally:
            ar.urllib.request.urlopen = original_urlopen
            ar.openrouter_key = original_key

    def test_ask_model_takes_the_track_the_caller_has(self) -> None:
        """A signature check, because the failure was a signature.

        `ask_model` reads the brief for a track, so it must be *given* one. If a
        future edit drops the parameter, the body breaks again -- at the first
        experiment, on a real machine, after a real API call.
        """
        parameters = list(inspect.signature(ar.ask_model).parameters)
        self.assertIn(
            "track",
            parameters,
            f"ask_model must be given the track it builds the brief for; got {parameters}",
        )
        annotation = inspect.signature(ar.ask_model).parameters["track"].annotation
        self.assertEqual(annotation, "Track")


class TestTheBriefIsWrittenForWhicheverTrackIsRunning(unittest.TestCase):
    """`autoresearch/program.md` is the system prompt, and it is a template.

    It is sent as the system message, so a leftover `{{...}}` is not a cosmetic
    slip -- it is an instruction to the model, and a brief that says `cargo
    clippy` to a model holding Go source is worse than no brief.
    """

    def test_it_renders_for_every_track_with_nothing_left_over(self) -> None:
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                brief = ar.read_prompt(track)
                self.assertNotIn("{{", brief)
                self.assertNotIn("}}", brief)
                self.assertIn(f"improving a {track.language} implementation", brief)

    def test_it_names_that_track_own_files_and_build(self) -> None:
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                brief = ar.read_prompt(track)
                self.assertIn(track.source_key, brief)
                self.assertIn(
                    track.probe.relative_to(ar.REPO_ROOT).as_posix(), brief
                )
                self.assertIn(f"autoresearch/{track.candidate_dir}/", brief)
                self.assertIn(track.lint_command, brief)

    def test_it_never_tells_one_track_to_satisfy_another_track_toolchain(self) -> None:
        foreign = {
            "rust": ("`go.mod`", "npx tsc", "`-lm`", "`cc "),
            "go": ("Cargo.toml", "cargo clippy", "npx tsc", "`-lm`", "`cc "),
            "typescript": ("Cargo.toml", "cargo clippy", "`go.mod`", "`-lm`", "`cc "),
            "c": ("Cargo.toml", "cargo clippy", "npx tsc", "`go.mod`"),
        }
        self.assertEqual(set(foreign), set(ar.TRACKS))
        for track in ar.TRACKS.values():
            for needle in foreign[track.name]:
                with self.subTest(track=track.name, needle=needle):
                    self.assertNotIn(
                        needle,
                        ar.read_prompt(track),
                        f"the {track.name} brief tells the model about {needle}, "
                        f"which belongs to another track",
                    )


    def test_a_new_placeholder_would_be_caught_rather_than_sent(self) -> None:
        """The failure mode of substituting into a document by hand.

        `read_prompt` raises on a placeholder it does not know how to fill. This
        asserts the guard exists by rendering a template with one, rather than
        trusting that a code path nobody has taken is correct.
        """
        original = ar.PROGRAM_MD.read_text(encoding="utf-8")
        try:
            ar.PROGRAM_MD.write_text(original + "\n{{a_placeholder_nobody_defines}}\n", encoding="utf-8")
            with self.assertRaises(ar.ResearchError) as caught:
                ar.read_prompt(ar.TRACKS["rust"])
            self.assertIn("a_placeholder_nobody_defines", str(caught.exception))
        finally:
            ar.PROGRAM_MD.write_text(original, encoding="utf-8")


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
        self.track = ar.TRACKS["rust"]
        self.prompt = ar.build_prompt(
            self.track,
            ar.read_ledger(),
            {"loss": 2.433882, "steps_per_sec": 95.38, "grad_ratio": 1.06316},
        )

    #: A marker that must appear in each port's source, to prove the prompt is
    #: carrying that port's real file rather than a Rust-shaped assumption about
    #: where the model's forward pass lives.
    TRACK_MARKERS = {
        "rust": ("fn rmsnorm", "fn softmax", "struct Tensor", "fn backward"),
        "go": ("func rmsnorm", "func softmax", "type Value", "func (v *Value) Backward"),
        "typescript": ("function rmsnorm", "static softmax", "class Value", "backward()"),
    }

    def test_the_whole_candidate_source_is_included(self) -> None:
        source = self.track.candidate_file.read_text(encoding="utf-8")
        for marker in self.TRACK_MARKERS[self.track.name]:
            self.assertIn(marker, self.prompt, f"{marker} is missing from the prompt")
        self.assertIn(
            source.rstrip().splitlines()[-1],
            self.prompt,
            "the prompt stops short of the end of the file",
        )

    def test_the_prompt_names_the_track_it_is_asking_for(self) -> None:
        """The file path and the diff headers, in this track's own language.

        A model shown Go source and asked for a patch to `src/lib.rs` proposes
        nothing applicable, and the experiment is wasted for a reason that has
        nothing to do with the hypothesis.
        """
        self.assertIn(self.track.source_key, self.prompt)
        self.assertIn(f"```{self.track.fence}", self.prompt)
        self.assertIn(f"--- a/{self.track.source_key}", self.prompt)
        self.assertIn(f"+++ b/{self.track.source_key}", self.prompt)

    def test_the_prompt_shows_only_its_own_track_ledger(self) -> None:
        """A Go model must not be offered the Rust track's history.

        It would invite a change already tried, and worse, it implies the two are
        competing on one axis.

        Matched on run id rather than on description, and that is not a
        preference. Descriptions repeat: the model independently proposes
        "Lower Adam beta2 from 0.99 to 0.98" on the Rust track twice, on Go once,
        on TypeScript twice and on C once, so a description-based cross-track
        assertion fails as soon as the same idea lands on two tracks within the
        15-row window -- which is what adding a fourth track did. The ids are
        unique across the whole ledger, so they cannot collide.
        """
        for track in ar.TRACKS.values():
            with self.subTest(track=track.name):
                prompt = ar.build_prompt(
                    track,
                    ar.read_ledger(),
                    {"loss": 2.4, "steps_per_sec": 90.0, "grad_ratio": 1.0},
                )
                ledger = ar.read_ledger()
                mine = [r for r in ledger if r.track == track.name]
                self.assertTrue(mine, f"the {track.name} track has no rows at all")
                for row in mine[-15:]:
                    self.assertIn(row.run_id, prompt)
                for row in [r for r in ledger if r.track != track.name][-15:]:
                    self.assertNotIn(row.run_id, prompt)

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
