"""Tests for the pure half of `tools/render_content.py`.

The rot this catches: the render step used to be welded to the writer. `main()`
assembled the `content.json` payload, the index payload, `mkdir`, the write loop
and the summary print in one body, so "does the app get the right JSON?" could
only be answered by executing the whole thing against the real `content/` tree
and diffing the generated files. A bug in the payload assembly and a stale file
on disk were indistinguishable, and any test would have had to write to the
repository.

`render_all` is pure -- it returns the two payloads and does nothing else -- so
these tests build a fixture in memory, call it, and assert on the returned
values. Nothing here reads or writes `visualizer/.../generated/`.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from tools import render_content
from tools.concepts import Block, Concept


def _block(**kwargs) -> SimpleNamespace:
    """A duck-typed stand-in for a parsed body block.

    `render_block` only reads attributes (`.kind`, `.name`, `.line_no`, `.args`,
    `.quoted`, `.text`), so the fixture does not have to be a real `Block`.
    """
    defaults = {"kind": "prose", "name": "", "line_no": 0, "args": {}, "quoted": "", "text": ""}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _concept(concept_id: str, *, frontmatter: dict | None = None, blocks: list | None = None) -> Concept:
    front = {"id": concept_id, "title": concept_id.upper()}
    front.update(frontmatter or {})
    return Concept(id=concept_id, path=None, frontmatter=front, body="", blocks=blocks or [])


def _index() -> dict:
    return {"chapters": [], "reference": {}, "hyperparameters": {}}


class RenderAllPureTest(unittest.TestCase):
    def test_returns_both_payloads_without_touching_disk(self) -> None:
        concepts = [_concept("b"), _concept("a")]
        index = _index()
        with mock.patch.object(render_content.Path, "write_text") as write, mock.patch.object(
            render_content.Path, "mkdir"
        ) as mkdir:
            content_payload, index_payload = render_content.render_all(concepts, index)
        self.assertFalse(write.called, "render_all must not write files")
        self.assertFalse(mkdir.called, "render_all must not create directories")
        self.assertEqual(set(content_payload), {"$comment", "concepts"})
        self.assertEqual(content_payload["$comment"], render_content.GENERATED_HEADER)
        self.assertEqual(set(index_payload), {
            "$comment", "reference", "hyperparameters", "chapters", "edges", "glossary",
        })

    def test_concepts_are_sorted_by_id(self) -> None:
        # Titles deliberately reverse the id order. `_concept` defaults
        # `title` to `id.upper()`, so a fixture without titles could not tell a
        # sort by `id` from a sort by `title` -- this one can.
        concepts = [
            _concept("gamma", frontmatter={"title": "AAA"}),
            _concept("alpha", frontmatter={"title": "ZZZ"}),
            _concept("beta", frontmatter={"title": "MMM"}),
        ]
        content_payload, _ = render_content.render_all(concepts, _index())
        self.assertEqual([c["id"] for c in content_payload["concepts"]], ["alpha", "beta", "gamma"])
        self.assertEqual(
            [c["title"] for c in content_payload["concepts"]], ["ZZZ", "MMM", "AAA"]
        )

    def test_blocks_are_rendered_through_render_block(self) -> None:
        concepts = [
            _concept(
                "alpha",
                blocks=[
                    _block(kind="prose", text="some prose"),
                    _block(kind="directive", name="callout", quoted="A title", text="body", line_no=7),
                    _block(kind="directive", name="term", text="rmsnorm", line_no=9),
                ],
            )
        ]
        content_payload, _ = render_content.render_all(concepts, _index())
        blocks = content_payload["concepts"][0]["blocks"]
        self.assertEqual(blocks[0], {"kind": "prose", "markdown": "some prose"})
        self.assertEqual(blocks[1], {"kind": "callout", "line": 7, "title": "A title", "markdown": "body"})
        self.assertEqual(blocks[2]["term"], "rmsnorm")
        self.assertEqual(blocks[2]["line"], 9)

    def test_frontmatter_keys_are_copied_and_anchors_default(self) -> None:
        concepts = [_concept("alpha", frontmatter={"chapter": "ch1", "summary": "s"})]
        content_payload, _ = render_content.render_all(concepts, _index())
        entry = content_payload["concepts"][0]
        self.assertEqual(entry["chapter"], "ch1")
        self.assertEqual(entry["summary"], "s")
        self.assertEqual(entry["anchors"], {})


class RenderAllIndexTest(unittest.TestCase):
    def _index_with_chapter(self, members: list[str]) -> dict:
        index = _index()
        index["chapters"] = [
            {"id": "ch1", "order": 1, "title": "Chapter One", "blurb": "b", "concepts": members}
        ]
        index["glossary"] = {"terms": {"rmsnorm": {"definition": "root mean square norm"}}}
        return index

    def test_chapters_list_present_members_with_metadata(self) -> None:
        concepts = [
            _concept(
                "alpha",
                frontmatter={"order": 3, "summary": "sum", "difficulty": 4},
            )
        ]
        _, index_payload = render_content.render_all(concepts, self._index_with_chapter(["alpha", "ghost"]))
        chapter = index_payload["chapters"][0]
        self.assertEqual(chapter["id"], "ch1")
        self.assertEqual(chapter["blurb"], "b")
        self.assertEqual([c["id"] for c in chapter["concepts"]], ["alpha"])
        self.assertEqual(chapter["concepts"][0]["order"], 3)
        self.assertEqual(chapter["concepts"][0]["difficulty"], 4)
        self.assertEqual(chapter["concepts"][0]["summary"], "sum")

    def test_prereq_edges_are_derived_from_frontmatter(self) -> None:
        concepts = [_concept("alpha", frontmatter={"prereqs": ["beta"]})]
        _, index_payload = render_content.render_all(concepts, self._index_with_chapter([]))
        self.assertEqual(index_payload["edges"], [{"from": "beta", "to": "alpha"}])

    def test_related_edges_carry_kind(self) -> None:
        concepts = [_concept("alpha", frontmatter={"related": ["beta"]})]
        _, index_payload = render_content.render_all(concepts, self._index_with_chapter([]))
        self.assertEqual(index_payload["edges"], [{"from": "alpha", "to": "beta", "kind": "related"}])

    def test_glossary_is_passed_through(self) -> None:
        _, index_payload = render_content.render_all([], self._index_with_chapter([]))
        self.assertEqual(set(index_payload["glossary"]), {"rmsnorm"})


class RealBlockTest(unittest.TestCase):
    """The duck-typed fixture still has to agree with the real `Block`."""

    def test_real_block_renders_identically(self) -> None:
        real = Block(kind="directive", name="callout", quoted="T", text="body", line_no=5)
        standin = _block(kind="directive", name="callout", quoted="T", text="body", line_no=5)
        self.assertEqual(render_content.render_block(real), render_content.render_block(standin))


if __name__ == "__main__":
    unittest.main()
