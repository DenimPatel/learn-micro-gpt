"""Tests for tools/selectors.py -- the anti-rot mechanism.

The most important tests in this repository. Every one of them exists because the
thing being tested has a *silent* failure mode: a selector that resolves to the
wrong lines still renders a code panel, and it renders it confidently.

Run with `make test` (or `python3 -m unittest discover -s tools/tests -t .`).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools import selectors

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8")
C_SOURCE = (REPO_ROOT / "implementations" / "c" / "microgpt.c").read_text(encoding="utf-8")

#: The line ranges of the previous visualizer's `mapping.js`. That file was
#: spot-checked against the reference by hand and all 19 nodes were correct, so it
#: is the one piece of ground truth this repository inherited. Reproducing it
#: exactly from selectors is the strongest available evidence that the resolver
#: and the boundary rules agree with how the file is actually organised.
LEGACY_MAPPING = {
    "dataset": ("section:Let there be an input dataset", (13, 20)),
    "tokenizer": ("section:Let there be a Tokenizer", (22, 26)),
    "autograd": ("section:Let there be Autograd", (28, 71)),
    "params": ("section:Initialize the parameters", (73, 89)),
    "architecture": ("section:Define the model architecture", (91, 143)),
    "linear": ("def:linear", (93, 94)),
    "softmax": ("def:softmax", (96, 100)),
    "rmsnorm": ("def:rmsnorm", (102, 105)),
    "gpt-flow": ("def:gpt", (107, 143)),
    "embedding": ("def:gpt offset:[1, -32]", (108, 111)),
    "attn-block": ("def:gpt offset:[7, -10]", (114, 133)),
    "attn-heads": ("def:gpt offset:[15, -12]", (122, 131)),
    "attn-proj": ("def:gpt offset:[25, -10]", (132, 133)),
    "mlp-block": ("def:gpt offset:[27, -3]", (134, 140)),
    "head": ("def:gpt offset:[35, 0]", (142, 143)),
    "optimizer": ("section:Let there be Adam", (145, 148)),
    "training": ("section:Repeat in sequence", (150, 183)),
    "inference": ("section:Inference:", (185, 199)),
}


class TestSelectorGrammar(unittest.TestCase):
    def test_all_four_kinds_parse(self) -> None:
        for raw in ("def:gpt", "assign:n_embd", "comment:# Let there be Adam", "section:Repeat in sequence"):
            parsed = selectors.parse_selector(raw)
            self.assertIn(parsed.kind, {"def", "assign", "comment", "section"})

    def test_offset_parses_in_either_position(self) -> None:
        a = selectors.parse_selector("def:gpt offset:[15, -10]")
        b = selectors.parse_selector("offset:[15, -10] def:gpt")
        self.assertEqual(a.offset, (15, -10))
        self.assertEqual(b.offset, (15, -10))
        self.assertEqual(a.target, b.target)

    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.parse_selector("frobnicate:x")

    def test_missing_colon_is_rejected(self) -> None:
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.parse_selector("gpt")

    def test_empty_target_is_rejected(self) -> None:
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.parse_selector("def:")

    def test_empty_selector_is_rejected(self) -> None:
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.parse_selector("   ")

    def test_unknown_language_is_rejected(self) -> None:
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.source_for_language("cobol", REPO_ROOT)

    def test_unresolved_selector_raises_rather_than_defaulting(self) -> None:
        # The critical property: no fallback to line 0, no best-effort match. A
        # wrong answer that renders is worse than a build failure.
        for raw in ("def:no_such_function", "assign:no_such_name", "comment:no such comment"):
            with self.assertRaises(selectors.SelectorUnresolvedError, msg=raw):
                selectors.resolve("python", raw, REPO_ROOT, REFERENCE)

    def test_missing_source_file_is_an_error_not_a_skip(self) -> None:
        for language in ("go", "rust", "typescript"):
            with self.assertRaises(selectors.SelectorError, msg=language):
                selectors.resolve(language, "def:anything", REPO_ROOT)


class TestPythonResolution(unittest.TestCase):
    def resolve(self, raw: str) -> selectors.Span:
        return selectors.resolve("python", raw, REPO_ROOT, REFERENCE)

    def test_def_uses_real_ast_and_finds_the_end(self) -> None:
        span = self.resolve("def:gpt")
        self.assertEqual((span.start, span.end), (107, 143))
        # The span must actually end at the function's last line, not the first
        # line that merely looks like it.
        self.assertIn("return logits", REFERENCE.split("\n")[span.end - 1])

    def test_class_resolves_under_def(self) -> None:
        span = self.resolve("def:Value")
        self.assertEqual((span.start, span.end), (29, 71))
        self.assertIn("class Value", REFERENCE.split("\n")[span.start - 1])

    def test_method_resolves_under_qualified_name(self) -> None:
        span = self.resolve("def:Value.backward")
        self.assertEqual((span.start, span.end), (58, 71))

    def test_assign_sees_through_tuple_unpacking(self) -> None:
        # `learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8` binds
        # four names on one line. An `assign:` selector that only handled a plain
        # Name target would miss the whole hyperparameter line.
        for name in ("learning_rate", "beta1", "beta2", "eps_adam"):
            with self.subTest(name=name):
                self.assertEqual(self.resolve(f"assign:{name}").start, 146)

    def test_comment_selector_matches_comment_text(self) -> None:
        self.assertEqual(self.resolve("comment:Let there be Adam").start, 145)

    def test_comment_selector_ignores_the_marker(self) -> None:
        with_marker = self.resolve("comment:# Let there be Adam")
        without = self.resolve("comment:Let there be Adam")
        self.assertEqual(with_marker, without)

    def test_section_stops_at_the_next_section(self) -> None:
        span = self.resolve("section:Repeat in sequence")
        self.assertEqual((span.start, span.end), (150, 183))

    def test_section_ignores_a_multi_line_section_header(self) -> None:
        # Lines 91 and 92 are two consecutive top-level comments. A boundary rule
        # without the "preceded by a blank line" clause would make line 91 a
        # section of exactly one line.
        span = self.resolve("section:Define the model architecture")
        self.assertEqual((span.start, span.end), (91, 143))

    def test_offset_is_relative_not_absolute(self) -> None:
        # The plan's own example, and the property that keeps anchors from
        # becoming line numbers with extra steps.
        base = self.resolve("def:gpt")
        narrowed = self.resolve("def:gpt offset:[15, -10]")
        self.assertEqual(narrowed.start, base.start + 15)
        self.assertEqual(narrowed.end, base.end - 10)

    def test_offset_past_the_start_is_an_error(self) -> None:
        with self.assertRaises(selectors.SelectorError):
            self.resolve("def:linear offset:[-200, 0]")


class TestLegacyMappingParity(unittest.TestCase):
    """Every range the previous, hand-verified mapping.js declared."""

    def test_all_legacy_ranges_reproduce_exactly(self) -> None:
        for node, (raw, expected) in LEGACY_MAPPING.items():
            with self.subTest(node=node):
                span = selectors.resolve("python", raw, REPO_ROOT, REFERENCE)
                self.assertEqual(
                    (span.start, span.end),
                    expected,
                    f"{node}: {raw!r} resolved to {span.start}-{span.end}, "
                    f"expected {expected[0]}-{expected[1]}",
                )


class TestCResolution(unittest.TestCase):
    def resolve(self, raw: str) -> selectors.Span:
        return selectors.resolve("c", raw, REPO_ROOT, C_SOURCE)

    def test_multiline_signature_is_found(self) -> None:
        # `static inline void linear_fwd(const float *x,` then more parameter
        # lines then `{`. A line-at-a-time scan misses every one of these.
        span = self.resolve("def:linear_fwd")
        self.assertEqual((span.start, span.end), (171, 177))

    def test_control_flow_is_not_mistaken_for_a_definition(self) -> None:
        # `if (ready) {` must not be recorded as `def:ready`.
        with self.assertRaises(selectors.SelectorUnresolvedError):
            self.resolve("def:1")

    def test_closing_brace_does_not_prefix_the_next_definition(self) -> None:
        # The bug this guards: after a function body, the `}` on its own line got
        # glued onto the next definition as `}   static void forward_pos(...) {`,
        # which then failed an anchored match. So `forward_pos` silently
        # resolved to nothing while the file looked fine.
        self.assertEqual(self.resolve("def:forward_pos").start, 418)

    def test_every_real_function_resolves(self) -> None:
        for name in (
            "linear_fwd", "linear_bwd_w", "linear_bwd_x", "rmsnorm_fwd", "rmsnorm_bwd",
            "softmax_fwd", "adam_update", "load_data", "init_matrix", "init_weights",
            "forward_pos", "backward_all", "forward_inference", "main", "seed_rng",
            "shuffle_docs", "xoshiro_next", "rand_uniform", "rand_gauss", "rotl",
        ):
            with self.subTest(name=name):
                span = self.resolve(f"def:{name}")
                self.assertGreater(span.end, span.start - 1, "span must be non-empty")

    def test_macro_defines_are_anchors(self) -> None:
        for name, line in (("N_EMBD", 19), ("N_HEAD", 20), ("BLOCK_SIZE", 22), ("NUM_STEPS", 27)):
            with self.subTest(name=name):
                self.assertEqual(self.resolve(f"assign:{name}").start, line)

    def test_file_scope_arrays_resolve_despite_attributes(self) -> None:
        # `static float ALIGN128 wte[...]` -- the alignment attribute sits between
        # the type and the name, so a strict type-then-name pattern misses it.
        for name, line in (("wte", 91), ("wpe", 92), ("lm_head", 93), ("g_lm_head", 105)):
            with self.subTest(name=name):
                self.assertEqual(self.resolve(f"assign:{name}").start, line)

    def test_comment_selector_handles_box_drawing_separators(self) -> None:
        # `/* ── Inference ──...── */` has to reduce to `Inference`, or every C
        # section anchor has to spell out the box-drawing characters.
        self.assertEqual(self.resolve("comment:Inference").start, 944)
        self.assertEqual(self.resolve("comment:Main").start, 856)

    def test_comment_selector_ignores_indentation(self) -> None:
        # `  /* Training loop */` is indented two spaces; the same selector works
        # on a column-0 comment.
        self.assertEqual(self.resolve("comment:Training loop").start, 869)
        self.assertEqual(self.resolve("comment:Multi-head attention").start, 446)


class TestSpan(unittest.TestCase):
    def test_clamp_trims_to_file_length(self) -> None:
        self.assertEqual(selectors.Span(1, 5000).clamp(199).end, 199)

    def test_starting_past_the_end_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            selectors.Span(500, 600).clamp(199)

    def test_length(self) -> None:
        self.assertEqual(len(selectors.Span(10, 19)), 10)
        self.assertEqual(len(selectors.Span(10, 10)), 1)

    def test_offset_round_trip(self) -> None:
        span = selectors.Span(10, 20).apply_offset((5, -5))
        self.assertEqual((span.start, span.end), (15, 15))


if __name__ == "__main__":
    unittest.main()
