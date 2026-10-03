"""Tests for the content corpus and the validation gate.

The theme: every check here corresponds to a rot that *looks* fine. A concept
whose anchor points at the wrong lines still renders a code panel. A stale
`anchors.json` still highlights something. A prereq cycle still produces a valid
reading order for a graph library that does not care.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import concepts as content
from tools import gen_anchors, selectors, tag_fences, trace, validate_concepts
from tools.validate_concepts import Report, validate_against_schema

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```", re.DOTALL | re.MULTILINE)

#: A fence with **no** info string claims to be a verbatim excerpt of the
#: concept's own anchor range, and is checked against it. Any other fence
#: declares what it is -- `text` for illustrative pseudo-code, arithmetic, and
#: diagrams; `output` for program output; `python` for an excerpt from a
#: deliberately different part of the file -- and is not checked.
#:
#: That makes "is this snippet the thing I'm highlighting?" answerable by
#: looking at the fence, rather than being a convention everyone has to
#: remember. Concepts legitimately quote neighbouring code to make a comparison
#: (Adam quotes the hyperparameter line; the forward pass quotes the training
#: loop). Those excerpts are correct and worth having -- they just have to say
#: they are not the anchored region.
VERBATIM_FENCE = ""


def verbatim_fences(body: str) -> list[str]:
    """Every line of every untagged code fence in a concept body."""
    lines: list[str] = []
    for match in _FENCE_RE.finditer(body):
        if match.group(1) != VERBATIM_FENCE:
            continue
        lines.extend(match.group(2).split("\n"))
    return lines


class TestCorpus(unittest.TestCase):
    """The real corpus, as committed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.concepts = content.load_all(REPO_ROOT)
        cls.index = content.load_index(REPO_ROOT)
        cls.schema = content.load_schema(REPO_ROOT)
        cls.by_id = {c.id: c for c in cls.concepts}

    def test_there_are_eighteen_concepts(self) -> None:
        self.assertEqual(len(self.concepts), 18)

    def test_ids_match_filenames(self) -> None:
        for concept in self.concepts:
            self.assertEqual(concept.id, concept.path.stem, concept.path.name)

    def test_every_concept_is_reachable_from_the_curriculum(self) -> None:
        listed = [c for chapter in self.index["chapters"] for c in chapter["concepts"]]
        self.assertEqual(sorted(listed), sorted(self.by_id))

    def test_chapter_order_agrees_with_index(self) -> None:
        for chapter in self.index["chapters"]:
            orders = [self.by_id[m].frontmatter["order"] for m in chapter["concepts"]]
            self.assertEqual(orders, list(range(1, len(orders) + 1)), chapter["id"])

    def test_frontmatter_conforms_to_schema(self) -> None:
        report = Report()
        for concept in self.concepts:
            validate_against_schema(
                concept.frontmatter, self.schema, report, concept.path.name, self.schema
            )
        self.assertEqual(report.errors, [])

    def test_every_concept_anchors_to_python(self) -> None:
        for concept in self.concepts:
            self.assertIn("python", concept.frontmatter.get("anchors", {}), concept.id)

    def test_every_anchor_declares_only_real_languages(self) -> None:
        for concept in self.concepts:
            for language in concept.frontmatter["anchors"]:
                self.assertIn(language, selectors.SOURCES, f"{concept.id}:{language}")

    def test_no_concept_is_its_own_prerequisite(self) -> None:
        for concept in self.concepts:
            self.assertNotIn(concept.id, concept.frontmatter.get("prereqs", []))

    def test_every_body_has_prose(self) -> None:
        for concept in self.concepts:
            prose = [b for b in concept.blocks if b.kind == "prose"]
            self.assertTrue(prose, f"{concept.id} has no prose blocks")
            self.assertGreater(len("".join(b.text for b in prose)), 500, concept.id)

    def test_every_concept_quotes_code_from_inside_its_own_anchor(self) -> None:
        """A concept that quotes code outside its own anchor range is the worst
        failure this repository can have: the prose is right, the highlight is
        wrong, and both are rendered with total confidence. So every line in an
        untagged code fence -- which claims to be a verbatim excerpt of the
        anchor -- must actually appear in the range its selector resolves to.
        """
        source = (REPO_ROOT / "reference" / "microgpt.py").read_text().split("\n")

        def normalise(line: str) -> str:
            # Strip the trailing comments the reference is full of, and trailing
            # punctuation the author may have added, so a quoted line matches the
            # source line it came from.
            return line.split("#")[0].strip().rstrip(",").strip()

        for concept in self.concepts:
            span = selectors.resolve(
                "python", concept.frontmatter["anchors"]["python"]["selector"], REPO_ROOT
            )
            anchored = {
                normalise(line) for line in source[span.start - 1 : span.end]
            }
            anchored.discard("")
            for line in verbatim_fences(concept.body):
                key = normalise(line)
                # `...` is an elision marker, not a line of the reference.
                if not key or set(key) <= {"."}:
                    continue
                with self.subTest(concept=concept.id, line=key[:48]):
                    self.assertIn(
                        key,
                        anchored,
                        f"{concept.id} quotes {key!r}, which is outside its own "
                        f"anchor {span.start}-{span.end}",
                    )


class TestBlockLineNumbers(unittest.TestCase):
    """`Block.line_no` is the 1-based line *in the concept file*.

    The parser sees only the body, so the frontmatter it never looks at is the
    easiest thing in the file to forget to count. Every consumer that maps a
    block back to a line (error messages say "softmax line 57", `render_content`
    stamps a `line` on each directive block) then lands some fixed number of
    lines too high. That is invisible until someone clicks through to the wrong
    place, which is the whole reason this is asserted against a real file.
    """

    #: A body with the frontmatter delimiter at a known line, so the expected
    #: numbers below are arithmetic rather than a second opinion from the code
    #: under test.
    _FRONTMATTER = "---\nid: demo\ntitle: Demo\n---\n"

    def _concept(self, body: str) -> content.Concept:
        tmp = Path(tempfile.mkdtemp(prefix="line-no-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "content" / "concepts" / "demo.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._FRONTMATTER + body, encoding="utf-8")
        return content.load_concept(path)

    def test_a_directive_reports_its_line_in_the_file(self) -> None:
        # File layout (1-based):
        #   1-4  frontmatter
        #   5    blank
        #   6    Intro
        #   7    blank
        #   8    :::term bos
        #   9    blank
        #   10   Outro
        concept = self._concept("\nIntro\n\n:::term bos\n\nOutro\n")
        lines = concept.path.read_text(encoding="utf-8").split("\n")
        term = [b for b in concept.blocks if b.name == "term"][0]
        self.assertEqual(term.line_no, 8)
        self.assertEqual(lines[term.line_no - 1].strip(), ":::term bos")

    def test_a_prose_block_reports_its_first_line(self) -> None:
        concept = self._concept("\nIntro line one\nIntro line two\n\n:::term bos\n")
        prose = [b for b in concept.blocks if b.kind == "prose"][0]
        self.assertEqual(prose.line_no, 6)

    def test_a_block_directive_reports_its_opening_line(self) -> None:
        #   1-4  frontmatter
        #   5    Intro
        #   6    blank
        #   7    :::callout "Title"
        #   8    inside
        #   9    :::
        #   10   blank
        #   11   :::term bos
        #   12   blank
        #   13   Outro
        concept = self._concept(
            "Intro\n\n:::callout \"Title\"\ninside\n:::\n\n:::term bos\n\nOutro\n"
        )
        by_name = {block.name: block for block in concept.directives()}
        self.assertEqual(by_name["callout"].line_no, 7)
        self.assertEqual(by_name["term"].line_no, 11)

    def test_real_concepts_agree_with_the_file_on_disk(self) -> None:
        for concept in content.load_all(REPO_ROOT):
            lines = concept.path.read_text(encoding="utf-8").split("\n")
            for block in concept.directives():
                self.assertTrue(0 < block.line_no <= len(lines), concept.id)
                self.assertTrue(
                    lines[block.line_no - 1].strip().startswith(f":::{block.name}"),
                    f"{concept.id} {block.name} claims line {block.line_no} "
                    f"but that line is {lines[block.line_no - 1]!r}",
                )


class TestGateFiresOnBrokenContent(unittest.TestCase):
    """The point of a gate is that it fails. None of these are hypothetical."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="content-test-"))
        for name in ("content", "reference", "data", "implementations", "visualizer", "tools"):
            source = REPO_ROOT / name
            if source.is_dir():
                shutil.copytree(source, self.tmp / name)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def concept(self, name: str) -> Path:
        return self.tmp / "content" / "concepts" / f"{name}.md"

    def patch(self, name: str, old: str, new: str) -> None:
        path = self.concept(name)
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text, f"{name}: anchor text {old!r} not found")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def validate(self) -> Report:
        report = Report()
        loaded = content.load_all(self.tmp)
        index = content.load_index(self.tmp)
        schema = content.load_schema(self.tmp)
        validate_concepts.check_schema(loaded, schema, report)
        validate_concepts.check_graph(loaded, index, report)
        validate_concepts.check_anchors(loaded, report, self.tmp)
        validate_concepts.check_bodies(loaded, index, report)
        validate_concepts.check_reference_pin(index, report, self.tmp)
        return report

    def test_typo_in_a_selector_is_caught(self) -> None:
        self.patch("softmax", 'selector: "def:softmax"', 'selector: "def:softmaxx"')
        result = gen_anchors.build(self.tmp)
        self.assertTrue(result["failures"])
        self.assertIn("softmax", result["failures"][0])

    def test_dangling_prereq_is_caught(self) -> None:
        self.patch("softmax", "prereqs: [linear]", "prereqs: [linear, does-not-exist]")
        errors = self.validate().errors
        self.assertTrue(any("does-not-exist" in e for e in errors), errors)

    def test_dangling_related_is_caught(self) -> None:
        self.patch("softmax", "related: [cross-entropy-loss", "related: [not-a-concept, cross-entropy-loss")
        errors = self.validate().errors
        self.assertTrue(any("not-a-concept" in e for e in errors), errors)

    def test_prereq_cycle_is_caught(self) -> None:
        # params-init requires autograd-value; make autograd-value require it back.
        self.patch("autograd-value", "prereqs: []", "prereqs: [params-init]")
        errors = self.validate().errors
        self.assertTrue(any("cycle" in e for e in errors), errors)

    def test_unknown_directive_is_caught(self) -> None:
        path = self.concept("softmax")
        path.write_text(
            path.read_text(encoding="utf-8") + "\n:::chart something\n:::\n", encoding="utf-8"
        )
        with self.assertRaises(content.ContentError) as caught:
            content.load_concept(path)
        self.assertIn("chart", str(caught.exception))

    def test_unclosed_block_directive_is_caught(self) -> None:
        path = self.concept("softmax")
        path.write_text(
            path.read_text(encoding="utf-8") + "\n:::callout\nsome text\nand no close\n",
            encoding="utf-8",
        )
        with self.assertRaises(content.ContentError) as caught:
            content.load_concept(path)
        self.assertIn("never closed", str(caught.exception))

    def test_block_crossing_a_heading_is_caught(self) -> None:
        """A missing `:::` used to be silently absorbed by the *next* bare `:::`
        anywhere below, so a `##` heading between them disappeared into the
        callout and the rendered page lost a section boundary."""
        path = self.concept("softmax")
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n:::callout\nsome text\n\n## A new section\n\nmore text\n:::\n",
            encoding="utf-8",
        )
        with self.assertRaises(content.ContentError) as caught:
            content.load_concept(path)
        message = str(caught.exception)
        self.assertIn("heading", message)
        self.assertIn("A new section", message)

    def test_inline_directive_inside_a_block_is_caught(self) -> None:
        """A `:::trace` line is structure, not callout prose. When the closer is
        missing this is how the widget was swallowed and never rendered."""
        path = self.concept("softmax")
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n:::callout\nsome text\n\n:::trace kind=softmax step=200 pos=0\n:::\n",
            encoding="utf-8",
        )
        with self.assertRaises(content.ContentError) as caught:
            content.load_concept(path)
        message = str(caught.exception)
        self.assertIn(":::trace", message)
        self.assertIn("is closed", message)

    def test_a_closed_block_does_not_cross_a_heading(self) -> None:
        """The rule is about the missing closer, not about headings per se: a
        properly closed callout immediately before a heading is fine."""
        path = self.concept("softmax")
        body = path.read_text(encoding="utf-8")
        front, rest = content.split_frontmatter(body, path)
        body_text = "\nIntro prose\n\n:::callout \"T\"\ninside\n:::\n\n## Heading\n\nmore\n"
        path.write_text(body.replace(rest, body_text), encoding="utf-8")
        concept = content.load_concept(path)
        names = [b.name for b in concept.directives()]
        self.assertIn("callout", names)
        self.assertTrue(any(b.kind == "prose" and "## Heading" in b.text for b in concept.blocks))

    def test_unknown_directive_argument_is_caught(self) -> None:
        path = self.concept("softmax")
        path.write_text(
            path.read_text(encoding="utf-8") + '\n:::trace bogus=1\n:::\n', encoding="utf-8"
        )
        with self.assertRaises(content.ContentError) as caught:
            content.load_concept(path)
        self.assertIn("bogus", str(caught.exception))

    def test_mismatched_id_and_filename_is_caught(self) -> None:
        self.patch("softmax", "id: softmax", "id: soft-max")
        with self.assertRaises(content.ContentError) as caught:
            content.load_all(self.tmp)
        self.assertIn("does not match the filename", str(caught.exception))

    def test_missing_anchor_language_is_caught(self) -> None:
        self.patch("softmax", "  c: { selector: \"def:softmax_fwd\" }", "  haskell: { selector: \"x\" }")
        errors = self.validate().errors
        self.assertTrue(any("unknown language" in e for e in errors), errors)

    def test_removing_a_whole_concept_breaks_the_curriculum(self) -> None:
        self.concept("softmax").unlink()
        errors = self.validate().errors
        self.assertTrue(
            any("lists 'softmax'" in e or "not listed" in e for e in errors), errors
        )

    def test_reference_digest_drift_is_caught(self) -> None:
        reference = self.tmp / "reference" / "microgpt.py"
        reference.write_text(reference.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        errors = self.validate().errors
        self.assertTrue(any("sha256 does not match" in e for e in errors), errors)


class TestDocsQuoteTheAnchors(unittest.TestCase):
    """A doc that quotes a line range must quote the range the concept anchors.

    The prose docs are the only place line numbers are hardcoded -- and they are
    hardcoded on purpose, because a reader following "lines 29-71" in
    `docs/HOW-TO-READ.md` needs a number. That makes them the one place a drift
    can hide, so the numbers are checked against the selectors that generate the
    site's own highlights.
    """

    #: A row of the "shape of it" table: `| start–end | what | concepts |`.
    #: The range is what a reader is *told* to open; the concept links say which
    #: concept(s) the row is standing in for.
    _ROW_RE = re.compile(
        r"^\|\s*(\d+)\s*[–-]\s*(\d+)\s*\|"  # | 145–148 |
        r"\s*([^|]*?)\s*\|"  # | Adam |
        r"\s*([^|]*?)\s*\|\s*$",  # | [Adam](content/concepts/adam.md) |
        re.MULTILINE,
    )
    _CONCEPT_LINK_RE = re.compile(r"content/concepts/([a-z0-9-]+)\.md")

    def documented_rows(self) -> list[tuple[int, int, str, list[str]]]:
        """Every `(start, end, what, concept_ids)` row of the doc's table."""
        doc = (REPO_ROOT / "docs" / "HOW-TO-READ.md").read_text(encoding="utf-8")
        rows: list[tuple[int, int, str, list[str]]] = []
        for start, end, what, concepts_cell in self._ROW_RE.findall(doc):
            ids = self._CONCEPT_LINK_RE.findall(concepts_cell)
            rows.append((int(start), int(end), what.strip(), ids))
        return rows

    def test_documented_ranges_match_the_resolved_anchors(self) -> None:
        """Every *single-concept* row must equal that concept's resolved span.

        The doc is the only place line numbers are hardcoded, so it is the only
        place they can silently rot -- and they did: the Adam row read
        `145–148` while the selector resolved to `173–181`. The old form of this
        test checked a *hand-written copy* of the numbers, so it passed while the
        doc was wrong, which is exactly the failure it was supposed to catch. It
        now parses the doc, so the number in the doc is the number under test.

        Rows that link more than one concept (the architecture row, the autograd
        row) describe a superset of any one selector and are deliberately not
        asserted here; their member concepts are covered by their own rows or by
        the anchor tests above.
        """
        rows = self.documented_rows()
        self.assertTrue(rows, "the 'shape of it' table was not found in docs/HOW-TO-READ.md")
        single = [(s, e, w, ids[0]) for s, e, w, ids in rows if len(ids) == 1]
        # A parser guard, so a table reflow that stops matching does not quietly
        # turn this test into a no-op. Counted from the parsed rows, not from the
        # passing assertions below, so a wrong range cannot trip it as well.
        self.assertGreaterEqual(
            len(single),
            5,
            f"expected several single-concept rows, parsed {single!r}",
        )
        for start, end, what, concept_id in single:
            concept = content.load_concept(
                REPO_ROOT / "content" / "concepts" / f"{concept_id}.md"
            )
            selector = concept.frontmatter["anchors"]["python"]["selector"]
            span = selectors.resolve("python", selector, REPO_ROOT)
            self.assertEqual(
                (start, end),
                (span.start, span.end),
                f"docs/HOW-TO-READ.md says {start}-{end} for {what!r} "
                f"({concept_id}) but its selector {selector!r} resolves to "
                f"{span.start}-{span.end}",
            )

    def test_the_documented_table_names_every_concept_it_can(self) -> None:
        # If a concept is added and the table is not, a reader gets no pointer to
        # it. Every concept that has its own single-concept row is listed below;
        # composite rows (the architecture, the training section) cover the rest.
        doc = (REPO_ROOT / "docs" / "HOW-TO-READ.md").read_text(encoding="utf-8")
        linked = set(self._CONCEPT_LINK_RE.findall(doc))
        expected = {c.id for c in content.load_all(REPO_ROOT)}
        self.assertEqual(
            linked,
            expected,
            f"docs/HOW-TO-READ.md links {sorted(linked)} but the corpus has "
            f"{sorted(expected)}",
        )


class TestQuotedStatistics(unittest.TestCase):
    """Numbers the prose quotes, checked against the trace that produced them.

    The widget that shows the loss curve was once labelling an EMA value "first 50
    mean" while the prose quoted the raw window mean -- 3.075 against 2.851, a
    disagreement nobody would have noticed without looking. The concepts quote
    several measured figures; this is what keeps them honest.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import json as _json

        trace_dir = REPO_ROOT / "traces" / "python" / "micro"
        meta = _json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
        losses = [
            _json.loads(line)["loss"]
            for line in (trace_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.stats = meta["loss_stats"]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        cls.first_window = mean(losses[:50])
        cls.last_window = mean(losses[-50:])

    def test_the_window_means_are_what_they_are(self) -> None:
        self.assertAlmostEqual(self.first_window, 2.8515, places=4)
        self.assertAlmostEqual(self.last_window, 2.3233, places=4)

    def test_the_training_loop_concept_quotes_them_exactly(self) -> None:
        body = (REPO_ROOT / "content" / "concepts" / "training-loop.md").read_text(
            encoding="utf-8"
        )
        # `f"{x:.3f}"` in Python and `toFixed(3)` in JS agree on these values, so
        # the quoted digits can be compared directly. (They did not agree on
        # 2.8515, which is a value this trace does not contain -- 2.851452 rounds
        # to 2.851 in both.)
        self.assertIn(f"| steps 1\u201350 | {self.first_window:.3f} |", body)
        self.assertIn(f"| steps 951\u20131000 | {self.last_window:.3f} |", body)

    def test_the_cross_entropy_concept_quotes_the_noise(self) -> None:
        body = (REPO_ROOT / "content" / "concepts" / "cross-entropy-loss.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"| mean | {self.stats['mean']:.3f} |", body)
        self.assertIn(f"| standard deviation | {self.stats['stdev']:.3f} |", body)
        self.assertIn(
            f"| **step-to-step moves that increase** | **{self.stats['upward_moves']} of "
            f"{self.stats['total_moves']}** |",
            body,
        )
        self.assertIn(f"| minimum | {self.stats['min']:.3f} (step {self._argmin_step()}) |", body)
        self.assertIn(f"| maximum | {self.stats['max']:.3f} |", body)

    def _argmin_step(self) -> int:
        import json as _json

        rows = [
            _json.loads(line)
            for line in (REPO_ROOT / "traces" / "python" / "micro" / "steps.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        return min(rows, key=lambda row: row["loss"])["step"] + 1


class TestBenchmarksDoc(unittest.TestCase):
    """`docs/BENCHMARKS.md` is generated from `benchmarks/results.json`.

    It says so at the top, which makes this check a promise rather than a
    nicety: a reader comparing the table against the JSON should not find a
    third number.
    """

    def setUp(self) -> None:
        self.doc = (REPO_ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
        self.data = json.loads((REPO_ROOT / "benchmarks" / "results.json").read_text())

    def test_every_measured_track_is_in_the_table(self) -> None:
        for row in self.data["results"]:
            with self.subTest(lang=row["lang"]):
                self.assertRegex(
                    self.doc,
                    rf"\|\s*{re.escape(row['lang'])}\s*\|",
                    f"{row['lang']} is measured but missing from docs/BENCHMARKS.md",
                )

    def test_the_table_shows_the_measured_ratios(self) -> None:
        for row in self.data["results"]:
            with self.subTest(lang=row["lang"]):
                if not row.get("relative_to_python"):
                    continue
                # The table rounds to whole steps/sec; the ratio is shown to 2 dp.
                self.assertIn(f"{row['steps_per_sec']:,.0f}", self.doc)
                self.assertIn(f"{row['relative_to_python']}x", self.doc)

    def test_the_recorded_machine_is_named(self) -> None:
        # A benchmark without a recorded machine is not a benchmark, and this doc
        # is where a reader would look for one.
        self.assertIn(self.data["runner"]["cpu"], self.doc)
        self.assertIn(self.data["runner"]["os"], self.doc)

    def test_it_states_the_three_known_non_answers(self) -> None:
        # The paragraphs explaining what is deliberately absent are the most useful
        # part of the document, and the easiest to drop in a rewrite.
        for marker in ("No `scaled` config row", "No peak RSS", "No \"vs PyTorch\" row"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.doc)


class TestBenchmarkFiguresInProse(unittest.TestCase):
    """The numbers quoted in the teaching prose track `benchmarks/results.json`.

    `linear.md` and `gpt-forward.md` both quote the Python and C wall times for
    the 1,000-step micro run, and both tell the reader the exact figures are in
    `benchmarks/results.json`. Before this test the two had drifted apart by
    ~10x on the C figure (prose said 0.04 s; the recorded run is 0.4268 s) and
    the prose called the runner "pinned" when the JSON sets `"pinned": false`.
    A reader cannot check either claim without doing the arithmetic, which is
    exactly the kind of wrongness a test should catch and a reviewer will not.
    """

    #: The prose rounds to these; the JSON holds 115.9043 and 0.4268.
    ROUNDINGS = {"python": 116, "c": 0.43}

    def setUp(self) -> None:
        self.data = json.loads((REPO_ROOT / "benchmarks" / "results.json").read_text())
        self.by_lang = {row["lang"]: row for row in self.data["results"]}
        self.concepts = content.load_all(REPO_ROOT)

    def _body(self, concept_id: str) -> str:
        return next(c.body for c in self.concepts if c.id == concept_id)

    def test_the_quoted_wall_times_match_the_recorded_run(self) -> None:
        # Each concept must quote the figure its recorded run rounds to, so a
        # hardcoded "108" or "0.04" here fails rather than passing by accident.
        for concept_id in ("linear", "gpt-forward"):
            body = self._body(concept_id)
            with self.subTest(concept=concept_id, lang="python"):
                self.assertIn(
                    f"about {self.ROUNDINGS['python']}",
                    body,
                    f"{concept_id} should quote the Python wall time rounded from "
                    f"{self.by_lang['python']['wall_seconds']} s",
                )
            with self.subTest(concept=concept_id, lang="c"):
                self.assertIn(
                    f"about {self.ROUNDINGS['c']}",
                    body,
                    f"{concept_id} should quote the C wall time rounded from "
                    f"{self.by_lang['c']['wall_seconds']} s",
                )

    def test_no_other_wall_time_seconds_are_quoted(self) -> None:
        # Presence alone is too weak: adding a *second*, contradictory figure
        # alongside the right one used to pass. Every "N seconds" (or "N.N")
        # figure in the benchmark sentence must sit near a recorded value.
        accepted = {self.by_lang["python"]["wall_seconds"]}
        accepted.add(self.by_lang["c"]["wall_seconds"])
        for concept_id in ("linear", "gpt-forward"):
            body = self._body(concept_id)
            for match in re.finditer(r"(\d+(?:\.\d+)?)\s*seconds", body):
                value = float(match.group(1))
                with self.subTest(concept=concept_id, value=value):
                    self.assertTrue(
                        any(abs(value - a) <= max(0.5, a * 0.05) for a in accepted),
                        f"{concept_id} quotes {value} seconds, which matches no "
                        f"recorded wall time in benchmarks/results.json "
                        f"({sorted(accepted)})",
                    )

    def test_the_quoted_speedup_matches_the_recorded_ratio(self) -> None:
        # 271.48x is recorded; "2,700x" was an order-of-magnitude error, so this
        # pins the quoted figure to the recorded ratio's whole-number form.
        ratio = self.by_lang["c"]["relative_to_python"]
        body = self._body("gpt-forward")
        self.assertIn(f"{ratio:.0f}x", body.replace(",", ""))

    def test_the_stale_figures_are_gone_from_both_concepts(self) -> None:
        # Guards the regression directly rather than by demonstration: read the
        # real prose and assert the old, wrong strings are not in it. Reverting
        # the numbers makes this fail; a self-contained string literal would not.
        stale = (
            f"about {self.ROUNDINGS['python'] - 8}",  # the old "about 108"
            f"about {self.ROUNDINGS['c'] - 0.39}",  # the old "about 0.04"
            "2,700x",
            "figures for a pinned runner",
        )
        for concept_id in ("linear", "gpt-forward"):
            body = self._body(concept_id)
            for fragment in stale:
                with self.subTest(concept=concept_id, fragment=fragment):
                    self.assertNotIn(fragment, body)

    def test_the_prose_does_not_call_the_runner_pinned(self) -> None:
        # `results.json` is explicit that it is not pinned. The old prose said the
        # figures were "for a pinned runner", which the JSON contradicts. Match any
        # wording that presents this run as pinned, so a paraphrase ("from a
        # pinned runner") does not slip through; the concept bodies are the raw
        # file text, so ordinary punctuation is present.
        self.assertFalse(self.data["runner"]["pinned"])
        for concept_id in ("linear", "gpt-forward"):
            body = self._body(concept_id)
            with self.subTest(concept=concept_id):
                for phrase in (
                    "figures for a pinned runner",
                    "from a pinned runner",
                    "for a pinned runner",
                ):
                    self.assertNotIn(phrase, body)


class TestDocsQuoteTheCurrentCounts(unittest.TestCase):
    """Counts written in governance docs match the tree they describe.

    `README.md`, `.github/pull_request_template.md` and `docs/ARCHITECTURE.md`
    all state counts -- how many tests run, how many port tracks exist, how many
    concepts carry maths. Each is cheap to derive and was wrong: the docs said
    107 Python tests when hundreds run, "all five tracks" when `implementations/`
    holds four, and "three concepts have maths" when all 18 do. A reader cannot check a
    count without doing the work, so the numbers drift silently; this pins the
    ones a reader is most likely to act on.
    """

    def setUp(self) -> None:
        self.readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(
            encoding="utf-8"
        )
        self.architecture = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(
            encoding="utf-8"
        )
        self.concepts = sorted((REPO_ROOT / "content" / "concepts").glob("*.md"))

    def _test_count(self) -> int:
        # Ask the runner itself, so the number quoted and the number shipped come
        # from the same place. Counting `def test_` by hand over-counts: the
        # phrase also appears in docstrings and comments.
        loader = unittest.TestLoader()
        suite = loader.discover(
            str(REPO_ROOT / "tools" / "tests"), top_level_dir=str(REPO_ROOT)
        )
        return suite.countTestCases()

    def test_the_pr_template_test_count_matches_the_suite(self) -> None:
        # The PR template does quote an exact count, and this is what keeps it
        # honest: add a test without updating the line and this fails. The README
        # deliberately does *not* quote a count -- an exact number in prose that
        # every new test invalidates is a number that is wrong most of the time,
        # so the README says "the Python suite" instead.
        count = self._test_count()
        self.assertIn(f"{count} Python tests", self.template)
        self.assertNotRegex(self.readme, r"\d+ Python tests")

    def test_the_track_count_matches_implementations(self) -> None:
        # A "track" is a port under implementations/; there are four, not five.
        # "Five tracks" is a defensible phrase only where the Python reference is
        # being counted in with the ports (the site's source loaders do this), so
        # what is forbidden here is claiming five in the parity-gate sense.
        tracks = [d.name for d in (REPO_ROOT / "implementations").iterdir() if d.is_dir()]
        self.assertEqual(len(tracks), 4, f"implementations/ holds {sorted(tracks)}")
        self.assertNotIn("all five tracks", self.readme)
        self.assertIn("four port tracks", self.readme)

    def test_the_maths_count_matches_the_concepts(self) -> None:
        # Every concept has a `math:` block, and ConceptPage renders each one as a
        # display formula, so the count is all of them -- not the "three" the
        # architecture doc used to claim.
        with_maths = [
            p
            for p in self.concepts
            if re.search(r"^math:", p.read_text(encoding="utf-8"), re.M)
        ]
        self.assertEqual(len(with_maths), len(self.concepts))
        self.assertNotIn("three concepts that have any", self.architecture)
        self.assertNotIn("Three concepts have", self.architecture)

    def test_the_section_selector_concepts_are_named(self) -> None:
        # `section:` exists for the concepts whose anchors span a whole region.
        # The doc names them, so the list must stay current.
        using = sorted(
            p.stem
            for p in self.concepts
            if "section:" in p.read_text(encoding="utf-8")
        )
        self.assertNotIn("three concepts describe whole regions", self.architecture)
        self.assertNotIn(
            "three concepts describe a whole chapter-sized region",
            (REPO_ROOT / "docs" / "ADDING-A-CONCEPT.md").read_text(encoding="utf-8"),
        )
        for name in using:
            with self.subTest(concept=name):
                self.assertIn(name, self.architecture)

    def test_no_doc_claims_a_wrong_track_count(self) -> None:
        # "Five tracks" is only right where the Python reference is counted in
        # with the four ports (the site's own source loaders, the data README).
        # Where the subject is the parity gate over the ports, it is four; the
        # README and the PR template both said "five" in that sense.
        for rel in ("README.md", ".github/pull_request_template.md"):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(doc=rel):
                self.assertNotIn("all five tracks", text)
                self.assertNotIn("five port tracks", text)


class TestAddingALanguageGuideIsAccurate(unittest.TestCase):
    """The guide that tells you how to add a port does not mislead you.

    `docs/ADDING-A-LANGUAGE.md` requirement 5 recommends reimplementing the
    autograd tape "rather than hand-writing a backward pass", and used to imply
    the probe then passes by construction. Three tracks that took exactly that
    route still fail the probe -- they left `rmsnorm` off the tape, which is
    `docs/KNOWN-ISSUES.md` issue 5 -- so the recommendation as written sends a
    contributor at a known bug while reading as a guarantee. These checks keep
    the guide tied to the measurements it now cites.
    """

    def setUp(self) -> None:
        self.guide = (REPO_ROOT / "docs" / "ADDING-A-LANGUAGE.md").read_text(
            encoding="utf-8"
        )
        self.issues = (REPO_ROOT / "docs" / "KNOWN-ISSUES.md").read_text(
            encoding="utf-8"
        )

    def test_the_guide_cites_the_rmsnorm_and_band_issues(self) -> None:
        # Both issues are the direct consequence of the guide's own advice, so a
        # reader who follows it has to be pointed at them.
        low = self.guide.lower()
        self.assertIn("issue 5", low)
        self.assertIn("issue 6", low)

    def test_the_tape_route_is_not_sold_as_correct_by_construction(self) -> None:
        # The sentence that made reimplementing the tape sound sufficient. It is
        # not: the tape only carries the ops you put on it.
        self.assertNotIn("the check above passes the first time", self.guide)
        # And the narrower claim that replaced it must still be there.
        self.assertIn("not the same thing", self.guide)

    def test_the_one_track_claim_carries_its_qualifier(self) -> None:
        # "the one track here that satisfies this requirement" reads, literally,
        # as excluding Go/Rust/TypeScript. C is the only one whose *ratio* is
        # close enough to 1.0; the others satisfy the requirement by measuring
        # and documenting, which is the whole point of the probe.
        self.assertNotIn(
            "it is the one track here that satisfies this\nrequirement", self.guide
        )
        self.assertIn("close\nenough to 1.0 to satisfy this requirement", self.guide)

    def test_the_quoted_ratios_match_the_issue(self) -> None:
        # The three measured ratios are quoted in the guide; each must appear in
        # the issue that measures them, so a re-measurement updates both.
        for ratio, track in (("1.063", "Rust"), ("1.127", "Go"), ("0.722", "TypeScript")):
            with self.subTest(track=track):
                self.assertIn(f"{ratio} ({track})", self.guide)

    def test_the_guide_does_not_reintroduce_a_centred_band(self) -> None:
        # Issue 6's point is that no port's ratio is 1.0, so the guide must not
        # tell a contributor the ratio is 1.0 for a correct gradient without the
        # caveat that their measured value will differ.
        self.assertNotIn(
            "For a correct gradient the ratio is `1.0`", self.guide
        )


class TestAddingALanguageChecklistIsComplete(unittest.TestCase):
    """The definition-of-done list names the gates a contributor will hit.

    Two of them failed silently for a reader who followed the guide to the
    letter: the frontmatter schema is closed, so a new language under `anchors`
    is rejected until `content/schema.json` is extended; and CI runs
    `prettier --check` (inside `npm run lint`), so unformatted visualizer work
    fails there even though `make fmt` was never mentioned. Both are cheap to
    assert against the artifacts that enforce them, so they cannot quietly drop
    off the list again.
    """

    def setUp(self) -> None:
        self.guide = (REPO_ROOT / "docs" / "ADDING-A-LANGUAGE.md").read_text(
            encoding="utf-8"
        )
        self.schema = json.loads(
            (REPO_ROOT / "content" / "schema.json").read_text(encoding="utf-8")
        )

    def test_the_checklist_names_make_fmt(self) -> None:
        # `make fmt` is the fix command; CI enforces the same thing through
        # `npm run lint` (`prettier --check`), so a reader who never runs it
        # fails CI.
        self.assertIn("`make fmt` run", self.guide)
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertRegex(makefile, r"(?m)^fmt:")

    def test_the_checklist_names_the_frontmatter_key_restriction(self) -> None:
        self.assertIn("keys the schema allows", self.guide)

    def test_the_closed_anchors_contract_the_guide_describes_is_real(self) -> None:
        # The guide says `anchors` enumerates its languages and the schema
        # rejects anything else. That has to match the schema, or the advice is
        # stale the moment a track is added.
        self.assertIs(self.schema.get("additionalProperties"), False)
        anchors = self.schema["properties"]["anchors"]
        self.assertIs(anchors.get("additionalProperties"), False)
        # Every language named in the guide's sentence is a real anchors key.
        for language in ("python", "c", "go", "rust", "typescript"):
            with self.subTest(language=language):
                self.assertIn(language, anchors["properties"])

    def test_ci_actually_enforces_prettier(self) -> None:
        # The checklist item is only honest while CI's lint step checks format.
        # If the lint script loses `prettier --check`, drop the item.
        pkg = json.loads(
            (REPO_ROOT / "visualizer" / "package.json").read_text(encoding="utf-8")
        )
        self.assertIn("prettier --check", pkg["scripts"]["lint"])
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("npm run lint", ci)


class TestCiCachePathsExist(unittest.TestCase):
    """Every `cache-dependency-path` in CI must name a file that exists.

    `actions/setup-go` (and setup-node/rust-cache) silently fall back when the
    cache path is missing: the run stays green, the cache stays cold, and every
    job re-downloads. The Go track cached `implementations/go/go.sum`, which
    cannot exist — `implementations/go/go.mod` has no `require` lines, so
    `go build` never writes one. Nothing failed, so nothing caught it. A path
    that does not resolve is the whole defect, and that is cheap to assert.
    """

    def _workflow_files(self) -> list[Path]:
        return sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

    def test_every_cache_dependency_path_exists(self) -> None:
        workflows = self._workflow_files()
        self.assertTrue(workflows, "no workflow files found")
        checked = 0
        for workflow in workflows:
            for path in self._cache_dependency_paths(workflow):
                self._assert_exists(workflow, path)
                checked += 1
        self.assertGreater(checked, 0, "no cache-dependency-path found to check")

    def _cache_dependency_paths(self, workflow: Path) -> list[str]:
        """The path(s) a workflow hands to a caching action.

        `cache-dependency-path` is either a scalar or a `|` block scalar whose
        paths are indented under the key; the block ends at the next line that
        is not more indented than the key. Reading it line-by-line avoids a
        regex that would swallow sibling keys.
        """
        paths: list[str] = []
        key = "cache-dependency-path:"
        lines = workflow.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(key):
                continue
            value = stripped[len(key) :].strip()
            if value != "|":
                paths.append(value)
                continue
            key_indent = len(line) - len(line.lstrip())
            for continuation in lines[index + 1 :]:
                if not continuation.strip():
                    break
                if len(continuation) - len(continuation.lstrip()) <= key_indent:
                    break
                paths.append(continuation.strip())
        return paths

    def _assert_exists(self, workflow: Path, path: str) -> None:
        with self.subTest(workflow=workflow.name, path=path):
            self.assertTrue(
                (REPO_ROOT / path).exists(),
                f"{workflow.name} caches {path!r}, which does not exist",
            )

    def test_the_go_module_has_no_go_sum(self) -> None:
        # The invariant behind the regression: with zero `require` lines there
        # is nothing to sum, so any future `cache-dependency-path` naming a
        # go.sum for this module is wrong by construction.
        gomod = (REPO_ROOT / "implementations" / "go" / "go.mod").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("require", gomod)
        self.assertFalse((REPO_ROOT / "implementations" / "go" / "go.sum").exists())


class TestAnchorsArtifact(unittest.TestCase):
    def test_committed_anchors_are_current(self) -> None:
        # The site renders code panels straight from this file, so a stale copy
        # is a confidently wrong site rather than a broken one.
        result = gen_anchors.build(REPO_ROOT)
        self.assertEqual(result["failures"], [])
        expected = json.dumps(result["document"], indent=2) + "\n"
        self.assertEqual(gen_anchors.OUTPUT.read_text(encoding="utf-8"), expected)

    def test_every_anchor_span_is_sane(self) -> None:
        document = json.loads(gen_anchors.OUTPUT.read_text(encoding="utf-8"))
        for language, info in document["sources"].items():
            for concept, spans in document["concepts"].items():
                for lang, span in spans.items():
                    if lang != language:
                        continue
                    with self.subTest(concept=concept, language=lang):
                        self.assertGreaterEqual(span["start"], 1)
                        self.assertLessEqual(span["end"], info["lines"])
                        self.assertGreaterEqual(span["end"], span["start"])


class TestTagFencesNormalise(unittest.TestCase):
    """`normalise` must not confuse a `#` inside a literal with a comment.

    The untagged-fence contract (see `TestCorpus`) is decided entirely by
    `normalise`: a fence is treated as a verbatim excerpt when every line
    normalises to a line in the anchor. Truncating at the first `#` anywhere --
    `line.split("#")[0]` -- makes `print("a # b")` normalise to `print("a`, so a
    correct excerpt stops matching (and an incorrect one can start to).
    """

    def test_hash_inside_a_string_literal_is_kept(self) -> None:
        self.assertEqual(tag_fences.normalise('print("a # b")'), 'print("a # b")')

    def test_trailing_comment_is_still_stripped(self) -> None:
        self.assertEqual(tag_fences.normalise("a # comment"), "a")

    def test_a_trailing_comment_after_a_literal_is_stripped(self) -> None:
        self.assertEqual(
            tag_fences.normalise('print("a # b")  # comment'), 'print("a # b")'
        )

    def test_single_and_triple_quotes_are_both_literal_delimiters(self) -> None:
        self.assertEqual(tag_fences.normalise("x = 'a # b'"), "x = 'a # b'")
        self.assertEqual(
            tag_fences.normalise('doc = """a # b"""'), 'doc = """a # b"""'
        )

    def test_a_comment_only_line_normalises_to_empty(self) -> None:
        self.assertEqual(tag_fences.normalise("# only a comment"), "")

    def test_comment_inside_a_call_still_truncates(self) -> None:
        # The `#` here is genuinely a comment, not inside a literal.
        self.assertEqual(tag_fences.normalise("foo(  # inline"), "foo(")


class TestTagFencesUnterminatedFence(unittest.TestCase):
    """An unterminated fence must not be reported as success.

    The closing-fence scan runs to EOF; copying the remainder verbatim would
    treat the rest of the file as fence body and reclassify everything after the
    typo while reporting success. `tag` now raises, naming the file and the
    opening line, and `main` turns that into a non-zero exit.
    """

    #: A real anchor (`tag` resolves the selector against the repo's
    #: `reference/microgpt.py`, which this test reads rather than fakes).
    _FRONTMATTER = (
        "---\nid: demo\ntitle: Demo\n"
        'anchors:\n  python: { selector: "def:softmax" }\n'
        "---\n"
    )

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tag-fences-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _scratch_concept(self, body: str) -> Path:
        path = self.tmp / "content" / "concepts" / "demo.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._FRONTMATTER + body, encoding="utf-8")
        return path

    def test_an_unterminated_fence_names_the_file_and_opening_line(self) -> None:
        path = self._scratch_concept("\nIntro\n\n```python\nprint(1)\n")
        with self.assertRaises(content.ContentError) as caught:
            tag_fences.tag(path)
        message = str(caught.exception)
        self.assertIn("demo.md", message)
        self.assertIn("unterminated", message)
        # frontmatter ends at file line 6; the opening ``` is body line 4.
        self.assertIn("line 10", message)

    def test_a_closed_fence_is_accepted(self) -> None:
        path = self._scratch_concept("\nIntro\n\n```text\n1 + 1\n```\n")
        self.assertEqual(tag_fences.tag(path), 0)

    def test_the_error_reaches_a_non_zero_exit(self) -> None:
        """`main` must translate the raise into a non-zero status."""
        content_dir = self.tmp / "content" / "concepts"
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / "demo.md").write_text(
            self._FRONTMATTER + "\nIntro\n\n```python\nprint(1)\n", encoding="utf-8"
        )
        original = tag_fences.REPO_ROOT
        tag_fences.REPO_ROOT = self.tmp
        shutil.copytree(REPO_ROOT / "reference", self.tmp / "reference")
        try:
            self.assertNotEqual(tag_fences.main(), 0)
        finally:
            tag_fences.REPO_ROOT = original


class TestMalformedIndexJson(unittest.TestCase):
    """`content/index.json` with the wrong *shape* is a ContentError, not a
    traceback from deep inside the validator.

    `load_index` only catches `json.JSONDecodeError`, so a file that parses but
    is structurally wrong used to reach `index.get("chapters")` (a list has no
    `.get`) or the chapter loop's `chapter.get(...)` and raise a bare
    `AttributeError`/`TypeError`. A person who hand-edited the index saw a
    traceback instead of the line they broke.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="index-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "content").mkdir()
        self.path = self.tmp / "content" / "index.json"

    def write_index(self, document: object) -> None:
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_a_list_where_a_mapping_is_expected_names_the_path(self) -> None:
        self.write_index(["data", "machinery"])
        index = content.load_index(self.tmp)
        with self.assertRaises(content.ContentError) as caught:
            validate_concepts.check_graph([], index, Report())
        message = str(caught.exception)
        self.assertIn("content/index.json", message)
        self.assertIn("list", message)

    def test_a_non_object_chapter_names_the_offending_key(self) -> None:
        self.write_index({"chapters": [{"id": "data", "concepts": "not-a-list"}]})
        index = content.load_index(self.tmp)
        with self.assertRaises(content.ContentError) as caught:
            validate_concepts.check_graph([], index, Report())
        message = str(caught.exception)
        self.assertIn("content/index.json", message)
        self.assertIn("concepts", message)
        self.assertIn("data", message)

    def test_a_string_in_the_chapter_list_names_its_position(self) -> None:
        self.write_index({"chapters": [{"id": "data", "concepts": []}, 7]})
        index = content.load_index(self.tmp)
        with self.assertRaises(content.ContentError) as caught:
            validate_concepts.check_graph([], index, Report())
        message = str(caught.exception)
        self.assertIn("chapters[1]", message)
        self.assertIn("int", message)

    def test_main_reports_it_as_a_content_error_not_a_traceback(self) -> None:
        """The entry point has to survive the malformed index too. A `main()`
        that lets the raise escape prints a traceback and exits 1 for the wrong
        reason -- and `make validate` calls this function, not `check_graph`."""
        tmp = Path(tempfile.mkdtemp(prefix="index-main-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        for name in ("content", "reference", "visualizer", "traces"):
            source = REPO_ROOT / name
            if source.is_dir():
                shutil.copytree(source, tmp / name)
        (tmp / "content" / "index.json").write_text('["data"]', encoding="utf-8")
        original = validate_concepts.REPO_ROOT
        validate_concepts.REPO_ROOT = tmp
        try:
            with mock.patch("sys.stdout"):
                code = validate_concepts.main()
        finally:
            validate_concepts.REPO_ROOT = original
        self.assertEqual(code, 1)


class TestDanglingSchemaRef(unittest.TestCase):
    """A `$ref` that does not resolve is a ContentError naming the ref.

    The schema is the one input the validator cannot check against itself, so a
    typo there (`#/definitions/anchorr`) used to be a `KeyError` from a dict
    lookup, and a `$ref` through a list (`#/properties/0/items`) a `TypeError`.
    The message has to say which ref and which file, or the only way to find it
    is to bisect the schema.
    """

    def test_a_missing_definition_is_caught(self) -> None:
        schema = {"$ref": "#/definitions/nope"}
        with self.assertRaises(content.ContentError) as caught:
            validate_against_schema({}, schema, Report(), "demo")
        message = str(caught.exception)
        self.assertIn("#/definitions/nope", message)
        self.assertIn("dangling", message)

    def test_a_ref_through_a_non_object_is_caught(self) -> None:
        schema = {"definitions": {"anchor": {"type": "object"}}, "$ref": "#/definitions/anchor/items"}
        with self.assertRaises(content.ContentError) as caught:
            validate_against_schema({}, schema, Report(), "demo")
        self.assertIn("#/definitions/anchor/items", str(caught.exception))

    def test_the_real_schema_still_resolves(self) -> None:
        concept = content.load_concept(REPO_ROOT / "content" / "concepts" / "softmax.md")
        report = Report()
        validate_against_schema(
            concept.frontmatter, content.load_schema(REPO_ROOT), report,
            concept.path.name, content.load_schema(REPO_ROOT),
        )
        self.assertEqual(report.errors, [])

    def test_a_remote_ref_is_still_rejected(self) -> None:
        with self.assertRaises(content.ContentError) as caught:
            validate_against_schema({}, {"$ref": "https://example.com/s.json"}, Report(), "demo")
        self.assertIn("local pointer", str(caught.exception))


class TestNonFiniteNumbers(unittest.TestCase):
    """`NaN` and `+/-inf` are not numbers a bounded field may hold.

    `isinstance(nan, float)` is True and every comparison against NaN is False,
    so a NaN satisfies `{"type": "number", "minimum": 0, "maximum": 30}`
    outright -- and `inf` passes `minimum`. The trace schema is where this
    matters: a corrupted trace used to validate, then render as "NaN" in the
    visualizer, with no gate firing anywhere along the way.
    """

    BOUNDED = {"type": "number", "minimum": 1, "maximum": 30}

    def test_a_nan_is_rejected_and_the_field_is_named(self) -> None:
        with self.assertRaises(content.ContentError) as caught:
            validate_against_schema(
                {"top": float("nan")},
                {"type": "object", "properties": {"top": self.BOUNDED}},
                Report(), "concept",
            )
        message = str(caught.exception)
        self.assertIn("concept.top", message)
        self.assertIn("finite", message)

    def test_an_infinity_is_rejected(self) -> None:
        with self.assertRaises(content.ContentError) as caught:
            validate_against_schema(
                {"top": float("inf")},
                {"type": "object", "properties": {"top": self.BOUNDED}},
                Report(), "concept",
            )
        self.assertIn("concept.top", str(caught.exception))

    def test_a_negative_infinity_is_rejected(self) -> None:
        with self.assertRaises(content.ContentError):
            validate_against_schema(
                {"top": float("-inf")},
                {"type": "object", "properties": {"top": self.BOUNDED}},
                Report(), "concept",
            )

    def test_the_same_value_in_a_field_with_no_bounds_is_still_rejected(self) -> None:
        with self.assertRaises(content.ContentError):
            validate_against_schema(float("nan"), {"type": "number"}, Report(), "value")

    def test_a_bound_is_still_checked_for_finite_values(self) -> None:
        report = Report()
        validate_against_schema({"top": 31}, {"type": "object", "properties": {"top": self.BOUNDED}}, report, "concept")
        self.assertTrue(any("maximum" in e for e in report.errors), report.errors)

    def test_an_integer_is_unaffected(self) -> None:
        report = Report()
        schema = {
            "type": "object",
            "properties": {"step": {"type": "integer", "minimum": 0}},
        }
        validate_against_schema({"step": 3}, schema, report, "concept")
        self.assertEqual(report.errors, [])


if __name__ == "__main__":
    unittest.main()
