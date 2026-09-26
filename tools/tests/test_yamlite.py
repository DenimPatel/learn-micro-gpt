"""Tests for tools/yamlite.py -- the stdlib-only YAML subset.

The parser's contract is mostly about *refusing*. A permissive frontmatter parser
is how a content repository ends up with a concept whose `shapes` silently
became a string, rendering a table of one row forever. Every "rejects" test here
corresponds to a way that could happen quietly.
"""

from __future__ import annotations

import unittest

from tools import yamlite


class TestScalars(unittest.TestCase):
    def test_plain_scalar(self) -> None:
        self.assertEqual(yamlite.parse("title: Multi-head attention")["title"], "Multi-head attention")

    def test_int_and_float(self) -> None:
        parsed = yamlite.parse("order: 7\ndifficulty: 2")
        self.assertEqual(parsed["order"], 7)
        self.assertIsInstance(parsed["order"], int)

    def test_bool(self) -> None:
        self.assertIs(yamlite.parse("tracing: true")["tracing"], True)
        self.assertIs(yamlite.parse("tracing: false")["tracing"], False)

    def test_quoted_string_keeps_colons_and_hashes(self) -> None:
        parsed = yamlite.parse('summary: "loss is noisy: really # yes"')
        self.assertEqual(parsed["summary"], "loss is noisy: really # yes")

    def test_trailing_comment_is_stripped_only_outside_quotes(self) -> None:
        self.assertEqual(yamlite.parse("a: 1  # note")["a"], 1)
        self.assertEqual(yamlite.parse('a: "1  # not a comment"')["a"], "1  # not a comment")

    def test_hash_inside_a_word_is_not_a_comment(self) -> None:
        self.assertEqual(yamlite.parse("url: http://x/#frag")["url"], "http://x/#frag")


class TestCollections(unittest.TestCase):
    def test_flow_sequence(self) -> None:
        self.assertEqual(yamlite.parse("related: [a, b, c]")["related"], ["a", "b", "c"])

    def test_flow_sequence_is_empty_not_missing(self) -> None:
        # `prereqs: []` must mean "no prerequisites", not "field absent". A
        # concept with an explicit empty list and a concept with no key are
        # different things and only one of them can be validated.
        self.assertEqual(yamlite.parse("prereqs: []")["prereqs"], [])

    def test_flow_map(self) -> None:
        parsed = yamlite.parse('shapes:\n  - { name: q_h, shape: "[T, 4]" }')
        self.assertEqual(parsed["shapes"], [{"name": "q_h", "shape": "[T, 4]"}])

    def test_block_sequence_of_scalars(self) -> None:
        self.assertEqual(yamlite.parse("x:\n  - one\n  - two")["x"], ["one", "two"])

    def test_nested_block_mapping(self) -> None:
        # The shape every anchor in this repository uses.
        parsed = yamlite.parse(
            'anchors:\n  python: { selector: "def:gpt" }\n  c: { selector: "def:main" }'
        )
        self.assertEqual(
            parsed["anchors"],
            {"python": {"selector": "def:gpt"}, "c": {"selector": "def:main"}},
        )


class TestBlockScalar(unittest.TestCase):
    def test_literal_block_preserves_backslashes(self) -> None:
        # LaTeX is full of backslashes. A parser that treats them as escapes
        # would silently mangle every formula in the site.
        text = yamlite.parse("math: |\n  \\text{attn} = \\frac{QK^\\top}{\\sqrt{d}}")["math"]
        self.assertEqual(text, "\\text{attn} = \\frac{QK^\\top}{\\sqrt{d}}")

    def test_block_scalar_does_not_swallow_the_next_key(self) -> None:
        # The bug this guards: a nested mapping iterated past the end of its own
        # block and treated an indented block-scalar line as a bad indent.
        parsed = yamlite.parse("math: |\n  x = 1\n  y = 2\nanchors:\n  python: { selector: a }")
        self.assertEqual(parsed["math"], "x = 1\ny = 2")
        self.assertEqual(parsed["anchors"], {"python": {"selector": "a"}})

    def test_trailing_blank_lines_are_trimmed(self) -> None:
        self.assertEqual(yamlite.parse("math: |\n  x\n\n")["math"], "x")


class TestRefusals(unittest.TestCase):
    def test_indentation_with_no_key_is_rejected(self) -> None:
        with self.assertRaises(yamlite.YamliteError):
            yamlite.parse("  stray: 1")

    def test_line_without_a_colon_is_rejected_with_a_line_number(self) -> None:
        with self.assertRaises(yamlite.YamliteError) as caught:
            yamlite.parse("a: 1\nthis is not a mapping entry")
        self.assertIn("line 2", str(caught.exception))

    def test_anchors_and_aliases_are_rejected(self) -> None:
        with self.assertRaises(yamlite.YamliteError):
            yamlite.parse_scalar("&anchor")

    def test_nesting_past_the_limit_is_rejected(self) -> None:
        deep = "a:\n  b:\n    c:\n      d:\n        e: 1"
        with self.assertRaises(yamlite.YamliteError):
            yamlite.parse(deep)

    def test_unterminated_quote_is_rejected(self) -> None:
        with self.assertRaises(yamlite.YamliteError):
            yamlite.parse_scalar('{ name: "unterminated }')

    def test_unbalanced_flow_collection_is_rejected(self) -> None:
        with self.assertRaises(yamlite.YamliteError):
            yamlite.parse_scalar("[a, b")


class TestRoundTrip(unittest.TestCase):
    SAMPLE = {
        "id": "attn-heads",
        "title": "Multi-head attention",
        "summary": "Q, K, V are sliced into heads; each head attends independently",
        "order": 7,
        "difficulty": 2,
        "related": ["softmax", "rmsnorm"],
        "prereqs": [],
        "shapes": [{"name": "q_h", "shape": "[T, 4]"}, {"name": "attn_weights", "shape": "[4, T]"}],
        "anchors": {
            "python": {"selector": "def:gpt", "offset": [15, -12]},
            "c": {"selector": "def:forward_pos"},
        },
        "math": "\\text{attn}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) V",
    }

    def test_dump_then_parse_is_identity(self) -> None:
        dumped = "\n".join(yamlite.dump_frontmatter(self.SAMPLE))
        self.assertEqual(yamlite.parse(dumped), self.SAMPLE)

    def test_empty_list_survives_the_round_trip(self) -> None:
        # `prereqs: []` must not come back as `prereqs:` with a null value.
        dumped = "\n".join(yamlite.dump_frontmatter({"prereqs": []}))
        self.assertIn("prereqs: []", dumped)
        self.assertEqual(yamlite.parse(dumped)["prereqs"], [])


if __name__ == "__main__":
    unittest.main()
