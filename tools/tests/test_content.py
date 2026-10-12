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
from tools import gen_anchors, render_content, selectors, tag_fences, trace, validate_concepts
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
    site's own highlights. The README's copy-paste selector snippet is checked
    the same way: its printed `# -> lines N-M` note is resolved, not trusted.
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
    #: The doc lives in `docs/`, so it links concepts the way GitHub resolves
    #: them -- relative to the file, `../content/concepts/adam.md`. The optional
    #: prefix keeps this matching if a link is ever written without it.
    _CONCEPT_LINK_RE = re.compile(r"(?:\.\./)?content/concepts/([a-z0-9-]+)\.md")

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

    #: A README exemplar row: a YAML `lang: { selector: "..." }` line with an
    #: inline `# -> lines N-M` note. The README teaches the selector syntax with
    #: this snippet, so a reader's first copy-paste runs it verbatim.
    _README_EXAMPLE_RE = re.compile(
        r'^\s*(?P<lang>[a-z-]+):\s*\{\s*selector:\s*"(?P<selector>[^"]+)"\s*\}'
        r".*?#\s*->\s*lines?\s*(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*$",
        re.MULTILINE,
    )

    def test_the_readme_example_resolves_to_the_range_it_quotes(self) -> None:
        """The README's selector snippet must resolve to the range it prints.

        The README is where a newcomer copies a selector from, so its example is
        the one place a stale selector teaches a wrong thing with total
        confidence. It *was* stale -- the `c` row read
        `comment:Multi-head attention  # -> lines 446-465`, and that comment
        resolves to a single line (550), while the concept it stands for anchors
        `def:forward_pos offset:[28, -39]` at 541-569. Deriving the range from
        the quoted selector, rather than trusting the printed number, is what
        makes the example self-checking.
        """
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        rows = [
            (m.group("lang"), m.group("selector"), int(m.group("start")), int(m.group("end")))
            for m in self._README_EXAMPLE_RE.finditer(readme)
        ]
        # A parser guard: if the snippet is reflowed so this stops matching, the
        # test must fail loudly rather than pass having checked nothing.
        self.assertGreaterEqual(
            len(rows),
            2,
            f"expected the README's selector snippet (python + c rows), parsed {rows!r}",
        )
        for lang, selector, start, end in rows:
            span = selectors.resolve(lang, selector, REPO_ROOT)
            self.assertEqual(
                (start, end),
                (span.start, span.end),
                f"README.md quotes lines {start}-{end} for {lang} selector "
                f"{selector!r}, but it resolves to {span.start}-{span.end}",
            )


class TestDocsLinksResolve(unittest.TestCase):
    """Every link in `docs/*.md` must point at something that exists.

    A doc is the one artifact a reader trusts without executing it, so a link
    is the one claim nothing else in the build verifies. `docs/HOW-TO-READ.md`
    linked twenty-one concepts as `content/concepts/adam.md`, which resolves
    against `docs/` and lands on a directory that does not exist -- every one of
    them a 404 on GitHub, and no test noticed because no test read the docs for
    links at all.

    Three kinds of link are checked, and the third is the reason this is not a
    one-line `os.path.exists`:

    | link | what must exist |
    | --- | --- |
    | `foo.md`, `../tools/yamlite.py` | the path, resolved against the *file* that links it |
    | `#some-heading` | a heading in the same file, under GitHub's slug rules |
    | `https://…` (or `<https://…>`) | nothing -- skipped, so tests need no network |

    Fenced code blocks are stripped first. `docs/ADDING-A-CONCEPT.md` teaches the
    documentation format with worked examples -- a template row linking
    `content/concepts/head-dim.md`, a concept that deliberately does not exist --
    and a link checker that read it as a real link would demand the example be
    true, which is the opposite of what an example is for.
    """

    _LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    _AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
    _FENCE_RE = re.compile(r"^```.*$", re.MULTILINE)
    _DOCS = REPO_ROOT / "docs"

    def doc_paths(self) -> list[Path]:
        return sorted(self._DOCS.glob("*.md"))

    @staticmethod
    def slug(heading: str) -> str:
        """GitHub's heading-to-anchor slug for a markdown heading line.

        Lowercase, drop punctuation (keeping `-` and spaces), spaces to hyphens.
        """
        text = heading.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"\s+", "-", text)

    @classmethod
    def visible_text(cls, text: str) -> str:
        """`text` with fenced code blocks removed, fence markers and all.

        A lone ``` toggles: an even-run of fence lines is open-then-close, so
        everything between a pair is dropped. `docs/ADDING-A-CONCEPT.md` nests a
        ```` ````-fence around a ```-fence for its "add a concept" example; the
        outer markers close the inner block, which is exactly how a reader skims
        it, so the simple toggle matches the intent.
        """
        out: list[str] = []
        in_fence = False
        for line in text.splitlines():
            if cls._FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if not in_fence:
                out.append(line)
        return "\n".join(out)

    def links_in(self, path: Path) -> list[str]:
        text = self.visible_text(path.read_text(encoding="utf-8"))
        return self._LINK_RE.findall(text) + self._AUTOLINK_RE.findall(text)

    def test_the_scan_finds_the_links_it_is_supposed_to(self) -> None:
        # A guard, so a regex that stops matching cannot silently make this
        # whole class a no-op. The corpus has dozens of relative links and at
        # least one anchor; if that ever reads as zero, the check is broken.
        relative = [
            link
            for path in self.doc_paths()
            for link in self.links_in(path)
            if not link.startswith(("http://", "https://", "mailto:", "#"))
        ]
        self.assertGreater(
            len(relative), 10, f"expected many relative links, found {relative!r}"
        )
        anchors = [
            link
            for path in self.doc_paths()
            for link in self.links_in(path)
            if link.startswith("#")
        ]
        self.assertTrue(anchors, "expected at least one in-page anchor link")

    def test_every_relative_link_resolves(self) -> None:
        for path in self.doc_paths():
            for link in self.links_in(path):
                if link.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = link.split("#", 1)[0]
                if not target:
                    continue
                resolved = (path.parent / target).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"{path.relative_to(REPO_ROOT)} links to {link!r}, which "
                    f"resolves to {resolved.relative_to(REPO_ROOT.resolve())} -- "
                    f"and nothing is there",
                )

    def test_every_in_page_anchor_names_a_heading(self) -> None:
        for path in self.doc_paths():
            text = self.visible_text(path.read_text(encoding="utf-8"))
            headings = {
                self.slug(line.lstrip("#").strip())
                for line in text.splitlines()
                if line.startswith("#")
            }
            for link in self.links_in(path):
                if not link.startswith("#"):
                    continue
                anchor = link[1:]
                self.assertIn(
                    anchor,
                    headings,
                    f"{path.relative_to(REPO_ROOT)} links to {link!r}, but no "
                    f"heading slugs to that. Headings are {sorted(headings)}",
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


class TestConceptProseCountsTheCallSites(unittest.TestCase):
    """`softmax.md` states how many places call `softmax`; the reference decides.

    The concept said "exactly two places" and named training and attention, but
    the reference calls `softmax` three times -- the third is the inference
    sampler, which `inference-sampling.md` already describes. A prose count like
    this is cheap to write and easy to leave behind: adding a call site is a
    one-line change, and nothing about the page looks wrong when the sentence
    goes stale. This is the check that ties the sentence to the file.
    """

    def _softmax_call_sites(self) -> list[int]:
        # The definition line is `def softmax(`, not a call. Everything else
        # matching `softmax(` is a call site -- counted, not hard-coded, so a
        # fourth call makes this fail rather than silently staying green.
        lines = (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8").splitlines()
        return [
            n
            for n, line in enumerate(lines, start=1)
            if "softmax(" in line and not line.lstrip().startswith("def ")
        ]

    def test_softmax_prose_count_matches_the_reference(self) -> None:
        call_sites = self._softmax_call_sites()
        self.assertEqual(
            len(call_sites),
            3,
            f"softmax.md says 'three places'; the reference calls it at lines {call_sites}",
        )
        body = (REPO_ROOT / "content" / "concepts" / "softmax.md").read_text(encoding="utf-8")
        self.assertIn("called in exactly three places", body)
        # And it must not still carry the old count in the heading, which said
        # "Two uses" and read as a second, contradicting statement.
        self.assertNotIn("## Two uses", body)


class TestConceptProseDerivesTheLossRatio(unittest.TestCase):
    """`cross-entropy-loss.md` quotes a loss-x ratio; the vocab size decides it.

    The page used to say a loss of 2.0 was "about 7x more likely than uniform".
    That is the `e^2 = 7.4` factor, which is how much more likely the model's
    pick is than an *event of probability 1* -- not than a uniform guess over the
    vocabulary. The real ratio is `e^(uniform - loss)`, and it moves with the
    vocab size: get the vocab wrong and the quoted ratio is wrong too.

    So this test derives the vocab size from the dataset with the reference's own
    formula, computes the ratio from it, and insists the prose carries *that*
    number. Hard-coding 3.65x here would just move the rot.
    """

    def _vocab_size(self) -> int:
        # Exactly the reference's derivation (`reference/microgpt.py`): unique
        # characters across the documents, plus one for BOS.
        text = (REPO_ROOT / "data" / "input.txt").read_text(encoding="utf-8")
        docs = [line.strip() for line in text.strip().split("\n") if line.strip()]
        uchars = sorted(set("".join(docs)))
        return len(uchars) + 1

    def test_the_quoted_ratio_matches_the_vocab_size(self) -> None:
        import math

        vocab = self._vocab_size()
        # A uniform guess over `vocab` characters costs this many nats.
        uniform = -math.log(1 / vocab)
        # A loss of 2.0 sits `uniform - 2.0` nats below it; that gap is a log-ratio.
        ratio = math.exp(uniform - 2.0)

        # 1/27 ~= 0.037, uniform ~= 3.30, ratio ~= 3.65 -- all read from `vocab`,
        # not typed in.
        self.assertAlmostEqual(uniform, 3.30, places=2)
        self.assertAlmostEqual(ratio, 3.65, places=2)

        body = (REPO_ROOT / "content" / "concepts" / "cross-entropy-loss.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"1/{vocab}", body)
        self.assertIn(f"over {vocab} characters", body)
        self.assertIn(f"{ratio:.2f}x more likely", body)

    def test_the_e_squared_factor_is_not_sold_as_the_ratio(self) -> None:
        # `e^2 = 7.4` is the baseline-to-one-event factor, not the model's edge
        # over uniform. If it reappears as the quoted ratio, this fails.
        body = (REPO_ROOT / "content" / "concepts" / "cross-entropy-loss.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("7x more likely", body)
        self.assertNotIn("7.4x more likely", body)
        # The trap is allowed to be *named* -- but only while it is being called
        # out as the tempting mistake, which is how the page frames it now.
        if "7.4" in body:
            self.assertIn("belongs to the uniform *baseline*", body)


class TestConceptProseCountsTheAttentionLoop(unittest.TestCase):
    """`multi-head-attention.md` states the loop's line count; the reference decides.

    The page opened with "Twelve lines of Python" while the attention loop it
    quotes is ten lines -- the same block, verbatim, in `reference/microgpt.py`.
    A prose count is cheap to write and easy to leave behind: the sentence is not
    the highlighted region, so nothing about the rendered page looks wrong when
    it drifts. This check derives the count from the reference by content (the
    `x_attn = []` line through the `x_attn.extend(head_out)` line), so a
    re-shaped loop updates the expected number instead of silently passing.
    """

    def _attention_loop_lines(self) -> int:
        lines = (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8").splitlines()
        start = next(n for n, line in enumerate(lines) if line.strip() == "x_attn = []")
        end = next(n for n, line in enumerate(lines) if line.strip() == "x_attn.extend(head_out)")
        # Inclusive of both endpoints: the loop body from the accumulator's
        # creation to the last head output being appended.
        return end - start + 1

    def test_the_loop_count_matches_the_reference(self) -> None:
        count = self._attention_loop_lines()
        self.assertEqual(count, 10, "the reference attention loop grew; update the concept's count")
        word = {10: "Ten"}[count]
        body = (REPO_ROOT / "content" / "concepts" / "multi-head-attention.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"{word} lines of Python", body)
        # And it must not still carry the old, wrong count.
        self.assertNotIn("Twelve lines", body)


class TestConceptProseCountsTheConfigNumbers(unittest.TestCase):
    """`params-init.md`'s "five numbers" heading and its prose must agree.

    The heading said "## The five numbers" while the paragraph below listed four
    and closed with "follows from those four" -- a miscount a reader has no way
    to resolve. This check derives the count of assigned config names from the
    concept's own code block, so the number in both places is the block's, not a
    second number typed into the test.
    """

    _NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}

    def _assigned_config_names(self) -> list[str]:
        # The config numbers are the scalars: `name = <int>` (or an expression of
        # them, as with `head_dim = n_embd // n_head`). The matrices, the lambda
        # and the flattened list are not "numbers", so they are excluded -- the
        # derivation stays about the block, not about a hand-count here.
        body = (REPO_ROOT / "content" / "concepts" / "params-init.md").read_text(encoding="utf-8")
        block = _FENCE_RE.search(body)
        assert block is not None, "params-init.md has no fenced code block"
        return re.findall(r"^([a-z_]+)\s*=\s*(?:\d|n_)", block.group(2), re.MULTILINE)

    def test_the_heading_and_the_prose_agree_on_the_count(self) -> None:
        names = self._assigned_config_names()
        self.assertEqual(
            names,
            ["n_embd", "n_head", "n_layer", "block_size", "head_dim"],
            f"the config block assigns {names}; the test should be updated alongside it",
        )
        word = self._NUMBER_WORDS[len(names)]
        body = (REPO_ROOT / "content" / "concepts" / "params-init.md").read_text(encoding="utf-8")
        self.assertIn(f"## The {word} numbers", body)
        self.assertIn(f"follows from those {word}", body)
        # The old contradiction -- heading says five, closing sentence says four.
        self.assertNotIn("follows from those four", body)


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

    def test_the_candidate_list_matches_the_autoresearch_tree(self) -> None:
        # The architecture doc enumerates the research candidates twice: once in
        # prose ("one per port, currently ...") and once as a backticked list of
        # directories. Both said three while `autoresearch/` held four --
        # `candidate-c/` was added and neither sentence moved. Deriving the list
        # from the tree means a fifth candidate fails here until both are updated.
        candidates = sorted(
            d.name
            for d in (REPO_ROOT / "autoresearch").iterdir()
            if d.is_dir() and d.name.startswith("candidate")
        )
        self.assertEqual(
            len(candidates), 4, f"autoresearch/ holds {candidates}"
        )
        # The directory list: every candidate dir (minus the `autoresearch/`
        # prefix, which the sentence omits) must appear backticked.
        sentence = re.search(
            r"Each candidate has its own directory \((.*?)\)",
            self.architecture,
            re.DOTALL,
        )
        self.assertIsNotNone(sentence, "the candidate directory list is gone")
        listed = set(re.findall(r"`([^`]+)`", sentence.group(1)))
        self.assertEqual(
            listed,
            {name + "/" for name in candidates},
            f"docs/ARCHITECTURE.md lists {sorted(listed)}",
        )
        # The prose enumeration: the languages named in the "one per port,
        # currently ..." clause. Ordering is the doc's own; only membership is
        # pinned, so wording can change without churn.
        prose = re.search(
            r"one per port, currently ([^&]+) &mdash;", self.architecture
        )
        self.assertIsNotNone(prose, "the 'one per port, currently' clause is gone")
        for language in ("Rust", "Go", "TypeScript", "C"):
            with self.subTest(language=language):
                self.assertIn(language, prose.group(1))

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

    def test_a_lone_triple_quote_delimiter_is_not_treated_as_a_comment(self) -> None:
        # A `"""` line opens a docstring whose run closes on a later line. The
        # scanner matches the three quotes as one token, finds no second `"""` on
        # the same line, and stops instead of treating the unclosed run as text
        # it has passed. This is the *routine* path for docstring delimiters, not
        # a malformed-input edge case: two such lines sit in the anchored
        # reference source `tag` normalises. The line comes back unchanged.
        self.assertEqual(tag_fences.normalise('"""'), '"""')
        # Anything after a complete literal, then an unclosed quote, is still
        # scanned for a comment from the safe position -- so a real trailing
        # comment is stripped even when an apostrophe follows it.
        self.assertEqual(tag_fences.normalise("x = 'a'  # don't"), "x = 'a'")

    def test_comment_inside_a_call_still_truncates(self) -> None:
        # The `#` here is genuinely a comment, not inside a literal.
        self.assertEqual(tag_fences.normalise("foo(  # inline"), "foo(")

    def test_escaped_quote_does_not_close_the_literal(self) -> None:
        # IMP-110: a `\"` is data, not the end of the string. The scanner used
        # to close the literal at the backslash-quote and read the following `#`
        # as a comment, truncating mid-literal to `print("a \" `.
        self.assertEqual(
            tag_fences.normalise('print("a \\" # not a comment")'),
            'print("a \\" # not a comment")',
        )
        # A real comment *after* the escaped quote and the closing quote still
        # strips -- the escape only protects the quote, not everything after it.
        self.assertEqual(
            tag_fences.normalise('x = "a \\" b"  # trailing'),
            'x = "a \\" b"',
        )

    def test_even_backslashes_do_not_escape_the_closing_quote(self) -> None:
        # A literal ending in a backslash is written `"...\\\\"`; the pair of
        # backslashes is even, so the following quote *does* close it and a
        # trailing comment is found. Counting the run, rather than looking at
        # one preceding character, is what makes this line come out right.
        self.assertEqual(
            tag_fences.normalise('x = "a\\\\"  # comment'),
            'x = "a\\\\"',
        )

    def test_unterminated_literal_is_not_treated_as_a_comment(self) -> None:
        # IMP-110: `x = "abc # def` never closes its string, so the `#` is data
        # and the line must come back unchanged -- truncating to `x = "abc ` was
        # the bug. Only an unterminated run that *begins* the line is data; an
        # unclosed quote after a balanced literal is read as trailing prose (the
        # apostrophe case above), because one line cannot distinguish the two.
        self.assertEqual(tag_fences.normalise('x = "abc # def'), 'x = "abc # def')
        self.assertEqual(
            tag_fences.normalise("x = 'abc # def"), "x = 'abc # def"
        )


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


class TestMalformedGlossary(unittest.TestCase):
    """A `glossary` of the wrong shape is a ContentError naming the key.

    The outer shape (`glossary` is not an object) is the easy half: it used to
    reach `.get` and raise a bare `AttributeError`. The inner shape is the one
    that actually bit, because it fails *silently and plausibly* instead of
    loudly: `check_bodies` builds its term set with
    `set(index.get("glossary", {}).get("terms", {}))`, so `terms: ["a", "b"]` is
    a perfectly valid set containing the wrong members -- `:::term` validation
    then compares against nonsense and passes -- and `terms: "abc"` becomes
    `{"a", "b", "c"}`, validating terms against individual characters.
    `build_index_payload` shipped both straight into the visualizer's glossary.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="glossary-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "content").mkdir()
        self.path = self.tmp / "content" / "index.json"

    def write_index(self, document: object) -> None:
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_a_non_mapping_glossary_names_the_key(self) -> None:
        for bad in (["attention", "logits"], "attention", 42):
            with self.subTest(glossary=bad):
                self.write_index({"glossary": bad})
                index = content.load_index(self.tmp)
                with self.assertRaises(content.ContentError) as caught:
                    validate_concepts.check_graph([], index, Report())
                message = str(caught.exception)
                self.assertIn("glossary", message)
                self.assertIn(type(bad).__name__, message)

    def test_a_list_of_terms_names_the_inner_key(self) -> None:
        """The silent-wrong-answer case: this used to validate clean."""
        self.write_index({"glossary": {"terms": ["attention", "logits"]}})
        index = content.load_index(self.tmp)
        with self.assertRaises(content.ContentError) as caught:
            validate_concepts.check_graph([], index, Report())
        message = str(caught.exception)
        self.assertIn("glossary.terms", message)
        self.assertIn("list", message)

    def test_a_string_of_terms_names_the_inner_key(self) -> None:
        """`set("abc")` is `{"a","b","c"}` -- terms checked against characters."""
        self.write_index({"glossary": {"terms": "attention"}})
        index = content.load_index(self.tmp)
        with self.assertRaises(content.ContentError) as caught:
            validate_concepts.check_graph([], index, Report())
        self.assertIn("glossary.terms", str(caught.exception))

    def test_an_absent_or_null_glossary_is_still_allowed(self) -> None:
        """The guard must not be strict about absence. `None` is how JSON spells
        "not set" and a document without the key is legal -- rejecting either
        would break content that is fine."""
        for document in (
            {},
            {"glossary": None},
            {"glossary": {}},
            {"glossary": {"terms": None}},
        ):
            with self.subTest(document=document):
                self.write_index(document)
                index = content.load_index(self.tmp)
                validate_concepts.check_graph([], index, Report())  # must not raise

    def test_the_render_payload_refuses_the_same_shapes(self) -> None:
        """`render_content` is reachable without the validator, so it repeats the
        check rather than trusting it. Both must reject the same input, or a
        second entry point reintroduces exactly the silent payload this class of
        bug is about."""
        for bad in (["attention"], "attention", 42, {"terms": ["attention"]},
                    {"terms": "attention"}):
            with self.subTest(glossary=bad):
                with self.assertRaises(content.ContentError):
                    render_content.build_index_payload(
                        [],
                        {"chapters": [], "reference": {}, "hyperparameters": {},
                         "glossary": bad},
                    )

    def test_the_two_guards_agree_on_every_shape(self) -> None:
        """The validator and the renderer must make the same call, shape for shape.

        They are separate code paths and were written separately, so the failure
        mode is a divergence: one rejects what the other tolerates. That is how a
        second entry point reintroduces a bad shape the first one catches -- the
        renderer's original `or {}` silently accepted `[]`, `''`, `0` and `False`
        while `check_graph` rejected all four. Asserting agreement rather than
        listing the rejected shapes means a future edit to either guard is caught
        here instead of in a payload nobody validates.
        """
        shapes = [
            [], "", 0, False, None, {},
            ["attention"], "attention", 42,
            {"terms": []}, {"terms": ""}, {"terms": 0}, {"terms": None},
            {"terms": {}}, {"terms": {"logit": "a score"}}, {"terms": ["a"]},
        ]
        for glossary in shapes:
            with self.subTest(glossary=glossary):
                index = {"chapters": [], "reference": {}, "hyperparameters": {},
                         "glossary": glossary}
                try:
                    validate_concepts.check_graph([], dict(index), Report())
                    validator_ok = True
                except content.ContentError:
                    validator_ok = False
                try:
                    render_content.build_index_payload([], dict(index))
                    renderer_ok = True
                except content.ContentError:
                    renderer_ok = False
                self.assertEqual(
                    validator_ok,
                    renderer_ok,
                    f"guards disagree on glossary={glossary!r}: "
                    f"validator accepts={validator_ok}, renderer accepts={renderer_ok}",
                )

    def test_the_render_payload_accepts_the_real_index(self) -> None:
        """Guarding must not change valid output. The real 12-term glossary has
        to come through unchanged, or this fix is a regression the tests miss."""
        index = content.load_index(REPO_ROOT)
        payload = render_content.build_index_payload([], index)
        self.assertEqual(len(payload["glossary"]), 12)
        self.assertIn("logit", payload["glossary"])


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


class TestCitationGate(unittest.TestCase):
    """A prose citation that points outside its own anchor fails the build.

    This is the machine half of "citations resolve from the anchors, not from a
    line number in the markup". The number in the sentence and the window the
    panel crops to have to agree, and nothing at render time can check that: by
    then the prose is text and the anchor is a different file's line range. So
    the check lives here, against the same committed `anchors.json` the browser
    reads.

    The grammar is mirrored in `visualizer/src/content/cite.ts`, and the examples
    below are the same ones its vitest suite uses -- a form one side accepts and
    the other rejects is a citation that either fails the build for no reason or
    renders as a dead control.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="citation-test-"))
        for name in ("content", "reference", "data", "implementations", "visualizer", "tools"):
            source = REPO_ROOT / name
            if source.is_dir():
                shutil.copytree(source, self.tmp / name)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # `check_citations` reads the committed file, so the tmp copy has to be
        # regenerated once the tree exists -- the copy carried the *real*
        # repository's resolved ranges, which no longer line up with the tmp
        # reference file's line numbers if any test patched a concept above.
        self._regen()

    def _regen(self) -> None:
        result = gen_anchors.build(self.tmp)
        self.assertFalse(result["failures"], result["failures"])
        path = self.tmp / "visualizer" / "src" / "data" / "generated" / "anchors.json"
        path.write_text(json.dumps(result["document"], indent=2) + "\n", encoding="utf-8")

    def patch(self, name: str, old: str, new: str) -> None:
        path = self.tmp / "content" / "concepts" / f"{name}.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text, f"{name}: {old!r} not found")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        self._regen()

    def citations(self) -> list[str]:
        report = Report()
        loaded = content.load_all(self.tmp)
        validate_concepts.check_citations(loaded, report, self.tmp)
        return report.errors

    def test_the_real_corpus_has_a_resolving_citation(self) -> None:
        # The feature is only "done" if it is used; an empty check would pass on
        # any corpus, including one where nobody wrote a citation at all.
        self.assertEqual(self.citations(), [])
        text = (self.tmp / "content" / "concepts" / "data-loading.md").read_text(encoding="utf-8")
        self.assertIn("](cite:19)", text)

    def test_a_range_inside_the_anchor_passes(self) -> None:
        # data-loading's Python anchor is the section at lines 13-20.
        self.patch("data-loading", "](cite:19)", "](cite:14-16)")
        self.assertEqual(self.citations(), [])

    def test_a_range_outside_the_anchor_is_caught(self) -> None:
        # The failure the gate exists for: the prose cites a line the panel crops
        # away, so the reader clicks and nothing visibly happens.
        self.patch("data-loading", "](cite:19)", "](cite:200)")
        errors = self.citations()
        self.assertTrue(any("outside this concept's" in e and "anchored range" in e for e in errors), errors)

    def test_a_range_straddling_the_anchor_edge_is_caught(self) -> None:
        # Only the end overhangs. A check that compared only the start, or only
        # the end, would let this through.
        self.patch("data-loading", "](cite:19)", "](cite:20-21)")
        errors = self.citations()
        self.assertTrue(any("outside this concept's" in e and "anchored range" in e for e in errors), errors)

    def test_a_range_inside_a_non_primary_window_is_still_caught(self) -> None:
        # The disagreement this gate had to be *taught* not to have. data-loading
        # also anchors C at 397-484, and the two windows do not overlap. A
        # citation valid in the C window is a dead control on the page, whose
        # visible panel is Python -- so checking "any declared language" would
        # pass it and check the wrong thing. The gate has to name the primary
        # language, the same one `canCite` in `ConceptPage.tsx` checks.
        self.patch("data-loading", "](cite:19)", "](cite:400)")
        errors = self.citations()
        self.assertTrue(
            any("outside this concept's python anchored range" in e for e in errors), errors
        )

    def test_an_inverted_range_is_caught(self) -> None:
        self.patch("data-loading", "](cite:19)", "](cite:20-14)")
        errors = self.citations()
        self.assertTrue(any("is not a line range" in e for e in errors), errors)

    def test_a_non_numeric_range_is_caught(self) -> None:
        self.patch("data-loading", "](cite:19)", "](cite:the-shuffle)")
        errors = self.citations()
        self.assertTrue(any("is not a line range" in e for e in errors), errors)

    def test_a_typoed_scheme_is_not_a_citation(self) -> None:
        # `Cite:19` is a different href. It renders as an ordinary link (or a
        # broken one), and the gate correctly has nothing to say about it --
        # which is the boundary between "citation" and "link" holding.
        self.patch("data-loading", "](cite:19)", "](Cite:19)")
        self.assertEqual(self.citations(), [])

    def test_the_grammar_agrees_with_the_typescript_copy(self) -> None:
        # The examples `visualizer/tests/cite.test.ts` asserts on, run through
        # the Python parser. Inverting or loosening either side has to fail here.
        for raw, expected in (
            ("11", (11, 11)),
            ("11-12", (11, 12)),
            ("1", (1, 1)),
            ("7-7", (7, 7)),
            ("20-11", None),
            ("0", None),
            ("0-3", None),
            ("eleven", None),
            ("11-", None),
            ("-11", None),
            ("1.5", None),
            ("1-2-3", None),
        ):
            self.assertEqual(validate_concepts._parse_cite(raw), expected, raw)


if __name__ == "__main__":
    unittest.main()
