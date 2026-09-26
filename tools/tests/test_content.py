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

from tools import concepts as content
from tools import gen_anchors, selectors, trace, validate_concepts
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

    #: concept id -> the range `docs/HOW-TO-READ.md` claims.
    HOW_TO_READ = {
        "data-loading": (13, 20),
        "tokenizer": (22, 26),
        "autograd-value": (29, 71),
        "autograd-backward": (58, 71),
        "params-init": (73, 89),
        "linear": (93, 94),
        "softmax": (96, 100),
        "rmsnorm": (102, 105),
        "embedding": (108, 111),
        "multi-head-attention": (122, 131),
        "mlp-block": (134, 140),
        "lm-head": (142, 143),
        "adam": (173, 181),
        "cross-entropy-loss": (162, 168),
        "training-loop": (150, 183),
        "inference-sampling": (185, 199),
    }

    def test_documented_ranges_match_the_resolved_anchors(self) -> None:
        for concept_id, claimed in self.HOW_TO_READ.items():
            with self.subTest(concept=concept_id):
                concept = content.load_concept(REPO_ROOT / "content" / "concepts" / f"{concept_id}.md")
                selector = concept.frontmatter["anchors"]["python"]["selector"]
                span = selectors.resolve("python", selector, REPO_ROOT)
                self.assertEqual(
                    (span.start, span.end),
                    claimed,
                    f"docs/HOW-TO-READ.md says {claimed[0]}-{claimed[1]} for {concept_id} "
                    f"but its selector {selector!r} resolves to {span.start}-{span.end}",
                )

    def test_the_documented_table_names_every_concept_it_can(self) -> None:
        # If a concept is added and the table is not, a reader gets no pointer to
        # it. The converse -- a table row for a concept that no longer exists --
        # is caught by the previous test.
        doc = (REPO_ROOT / "docs" / "HOW-TO-READ.md").read_text(encoding="utf-8")
        for concept_id in self.HOW_TO_READ:
            self.assertIn(
                f"content/concepts/{concept_id}.md",
                doc,
                f"docs/HOW-TO-READ.md does not link {concept_id}",
            )


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


if __name__ == "__main__":
    unittest.main()
