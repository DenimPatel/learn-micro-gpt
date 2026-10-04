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


class TestDuplicateNames(unittest.TestCase):
    """A *definition* name must mean one thing. Two of them is a hard error.

    The failure this guards: `dict.setdefault` kept the first definition and
    discarded the second with no diagnostic, so a concept that typo'd one
    definition's name for the other's pointed at the wrong code and rendered it
    confidently. That is the same silent-wrong-answer class as a selector that
    matches nothing.

    Only definitions are strict. `assign:` of a mutable local is a value
    binding, not a declaration -- `x = rmsnorm(x)` twice in one function, or
    `logits` in two different functions, is ordinary code -- so `assign:` keeps
    the documented first-wins behaviour. Making those fatal would reject the
    reference implementation itself.
    """

    def test_python_duplicate_function_name_raises_and_names_both_origins(self) -> None:
        source = (
            "def thing():\n"
            "    return 1\n"
            "\n"
            "def other():\n"
            "    return 2\n"
            "\n"
            "def thing():\n"
            "    return 3\n"
        )
        with self.assertRaises(selectors.SelectorDuplicateError) as caught:
            selectors.resolve("python", "def:thing", REPO_ROOT, source)
        message = str(caught.exception)
        self.assertIn("def:thing", message)
        self.assertIn("1..2", message)  # first origin
        self.assertIn("7..8", message)  # second origin
        # It is still a SelectorError, so the documented catch contract holds.
        self.assertIsInstance(caught.exception, selectors.SelectorError)

    def test_c_duplicate_function_name_raises(self) -> None:
        source = (
            "static void f(void) {\n"
            "  int x = 1;\n"
            "}\n"
            "\n"
            "static void f(void) {\n"
            "  int x = 2;\n"
            "}\n"
        )
        with self.assertRaises(selectors.SelectorDuplicateError) as caught:
            selectors.resolve("c", "def:f", REPO_ROOT, source)
        self.assertIn("def:f", str(caught.exception))

    def test_c_duplicate_macro_name_raises(self) -> None:
        # A `#define` is a declaration, so a same-named macro twice is a
        # collision even though it is registered under `assign:`.
        source = "#define N_STEPS 10\n\nvoid f(void) { (void)0; }\n\n#define N_STEPS 20\n"
        with self.assertRaises(selectors.SelectorDuplicateError) as caught:
            selectors.resolve("c", "assign:N_STEPS", REPO_ROOT, source)
        self.assertIn("assign:N_STEPS", str(caught.exception))

    def test_local_rebinding_is_not_a_duplicate(self) -> None:
        # The regression this whole rule has to survive: the reference's own
        # `gpt` reassigns `x` ten times and one function's `logits` is another
        # function's `logits`. Ordinary reuse must resolve, not raise.
        source = (
            "def thing():\n"
            "    x = 1\n"
            "    x = 2\n"
            "    return x\n"
            "\n"
            "def other():\n"
            "    x = 3\n"
            "    return x\n"
        )
        self.assertEqual(
            selectors.resolve("python", "assign:x", REPO_ROOT, source).as_dict(),
            {"start": 2, "end": 2},  # first binding wins
        )

    def test_same_span_recorded_twice_is_not_a_duplicate(self) -> None:
        # `ast.walk` can reach one node by more than one route, and Go's export
        # spelling registers `gpt` and `GPT` from one definition. Neither is a
        # collision: the name still means exactly one span.
        source = "def thing():\n    return 1\n"
        self.assertEqual(
            selectors.resolve("python", "def:thing", REPO_ROOT, source).as_dict(),
            {"start": 1, "end": 2},
        )
        go_source = "// pkg header\nfunc add(a int) int {\n\treturn a\n}\n"
        span = selectors.resolve("go", "def:add", REPO_ROOT, go_source)
        self.assertEqual(span.as_dict(), {"start": 2, "end": 4})

    def test_distinct_names_still_register(self) -> None:
        # The ordinary case must be untouched: two different names, two spans,
        # no error, each resolving to its own definition.
        source = (
            "def thing():\n"
            "    alpha = 1\n"
            "    return alpha\n"
            "\n"
            "def other():\n"
            "    beta = 2\n"
            "    return beta\n"
        )
        self.assertEqual(
            selectors.resolve("python", "assign:alpha", REPO_ROOT, source).as_dict(),
            {"start": 2, "end": 2},
        )
        self.assertEqual(
            selectors.resolve("python", "assign:beta", REPO_ROOT, source).as_dict(),
            {"start": 6, "end": 6},
        )

    def test_real_sources_are_definition_duplicate_free(self) -> None:
        # The registry is only strict if the tracked content is clean. If this
        # fails, a real definition collision landed in a source file and every
        # concept anchored into that file is now ambiguous. These all resolve
        # against the *real* tracked files (no injected source), so the check is
        # on the content itself.
        selectors.resolve("python", "def:gpt", REPO_ROOT)
        selectors.resolve("c", "def:forward_pos", REPO_ROOT)
        selectors.resolve("go", "def:Add", REPO_ROOT)
        selectors.resolve("rust", "def:run", REPO_ROOT)
        selectors.resolve("typescript", "def:sum", REPO_ROOT)


class TestCResolution(unittest.TestCase):
    """
    These assert *semantic* properties -- that a selector resolves to a range
    containing the right thing -- rather than hardcoded line numbers.

    An earlier version of this file asserted exact line numbers for the C
    source, and every one of them broke the moment microgpt.c got a longer header
    comment. That is precisely the failure mode the whole selector design exists
    to prevent, and this test file was the thing committing it: a test that
    encodes line numbers is a test that will break on a harmless edit and get
    "fixed" by updating the number rather than by noticing what changed.
    """

    def resolve(self, raw: str) -> selectors.Span:
        return selectors.resolve("c", raw, REPO_ROOT, C_SOURCE)

    def lines(self, span: selectors.Span) -> str:
        return "\n".join(C_SOURCE.split("\n")[span.start - 1 : span.end])

    def test_multiline_signature_is_found(self) -> None:
        # `static inline void linear_fwd(const float *x,` then more parameter
        # lines then `{`. A line-at-a-time scan misses every one of these.
        span = self.resolve("def:linear_fwd")
        text = self.lines(span)
        self.assertIn("static inline void linear_fwd", text)
        self.assertIn("cblas_sgemv", text)
        # The span must end at the closing brace, not the opening one.
        self.assertEqual(text.rstrip().split("\n")[-1].strip(), "}")

    def test_control_flow_is_not_mistaken_for_a_definition(self) -> None:
        # `if (ready) {` must not be recorded as `def:ready`.
        with self.assertRaises(selectors.SelectorUnresolvedError):
            self.resolve("def:1")

    def test_closing_brace_does_not_prefix_the_next_definition(self) -> None:
        # The bug this guards: after a function body, the `}` on its own line got
        # glued onto the next definition as `}   static void forward_pos(...) {`,
        # which then failed an anchored match. So `forward_pos` silently
        # resolved to nothing while the file looked fine.
        first = self.lines(self.resolve("def:forward_pos"))
        self.assertTrue(first.startswith("static void forward_pos"), first[:60])
        self.assertIn("Multi-head attention", first)

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
                # And the resolved text must actually declare that function.
                self.assertIn(name, self.lines(span))

    def test_macro_defines_are_anchors(self) -> None:
        for name, value in (
            ("N_EMBD", "16"),
            ("N_HEAD", "4"),
            ("N_LAYER", "1"),
            ("BLOCK_SIZE", "16"),
            ("NUM_STEPS", "1000"),
            ("MAX_DOC_LEN", "20"),
        ):
            with self.subTest(name=name):
                text = self.lines(self.resolve(f"assign:{name}"))
                self.assertRegex(text, rf"#\s*define\s+{name}\b")
                self.assertIn(value, text)

    def test_file_scope_arrays_resolve_despite_attributes(self) -> None:
        # `static float ALIGN128 wte[...]` -- the alignment attribute sits between
        # the type and the name, so a strict type-then-name pattern misses it.
        for name in ("wte", "wpe", "lm_head", "g_lm_head", "attn_wq", "mlp_fc1"):
            with self.subTest(name=name):
                text = self.lines(self.resolve(f"assign:{name}"))
                self.assertRegex(text, rf"\b{name}\b\s*\[")
                self.assertNotIn("static void", text.split(name)[0][-20:] if name in text else "")

    def test_comment_selector_handles_box_drawing_separators(self) -> None:
        # `/* ── Inference ──...── */` has to reduce to `Inference`, or every C
        # section anchor has to spell out the box-drawing characters.
        for prefix in ("Inference", "Main", "Data loading", "Weight initialization"):
            with self.subTest(prefix=prefix):
                text = self.lines(self.resolve(f"comment:{prefix}"))
                self.assertIn(prefix, text)

    def test_comment_selector_ignores_indentation(self) -> None:
        # `  /* Training loop */` is indented two spaces; the same selector works
        # on a column-0 comment.
        for prefix in ("Training loop", "Multi-head attention", "MLP block"):
            with self.subTest(prefix=prefix):
                self.assertIn(prefix, self.lines(self.resolve(f"comment:{prefix}")))

    def test_section_selectors_work_on_c_too(self) -> None:
        span = self.resolve("section:Data loading")
        self.assertIn("load_data", self.lines(span))
        # A section must not run past the next one.
        self.assertNotIn("Weight initialization", self.lines(span))


class TestSpan(unittest.TestCase):
    def test_clamp_trims_to_file_length(self) -> None:
        self.assertEqual(selectors.Span(1, 5000).clamp(199).end, 199)

    def test_a_span_cannot_start_before_line_one(self) -> None:
        # The sibling of the clamp() case below, and it was a bare ValueError.
        # `__post_init__` is the choke point: it runs at construction, so a span
        # with a start below 1 cannot exist for `clamp()` to be called on and
        # `clamp()` guards only the top end. `apply_offset` does guard the bottom
        # itself, with a better message naming the offsets, and that is
        # deliberate -- but it is the backstop here that keeps the class's catch
        # contract whole for every other construction path.
        with self.assertRaises(selectors.SelectorUnresolvedError) as ctx:
            selectors.Span(-3, 5)
        self.assertIsInstance(ctx.exception, selectors.SelectorError)
        message = str(ctx.exception)
        # Named line number, so a caller can point at the offending anchor.
        self.assertIn("start=-3", message)
        self.assertIn("1-based", message)

    def test_a_non_integer_line_number_is_a_selector_error(self) -> None:
        # The annotation says `int` and nothing enforced it. Two distinct failures
        # hid behind that, split by position. A bad *start* leaked a bare
        # `TypeError` from the `self.start < 1` comparison at construction -- the
        # same escape the bare `ValueError` was before IMP-DNEW-1. A bad *end*
        # constructed fine and only leaked later, out of `len()`'s arithmetic or
        # `clamp()`'s comparison, which is arguably worse: the span exists, is
        # passed around, and fails somewhere that no longer knows which anchor
        # caused it. Neither is a `SelectorError`, so a caller doing the documented
        # `except SelectorError` got a traceback instead of a diagnostic naming
        # the concept and selector.
        for value in ("1", None, "abc", [], {}, b"1"):
            for start, end in ((value, 5), (1, value)):
                with self.subTest(start=start, end=end):
                    with self.assertRaises(selectors.SelectorError) as ctx:
                        selectors.Span(start, end)
                    message = str(ctx.exception)
                    self.assertIn("integer", message)
                    self.assertIn(type(value).__name__, message)

    def test_a_float_line_number_is_not_silently_accepted(self) -> None:
        # `Span(1.5, 5)` is not a line number, and it is not a TypeError either --
        # it *constructs*, and only falls over later inside `len()` or arithmetic,
        # far from the anchor that caused it. This is the item's real finding.
        for start, end in ((1.5, 5), (1, 1.5), (2.0, 3.0)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(selectors.SelectorSyntaxError):
                    selectors.Span(start, end)

    def test_a_boolean_line_number_is_not_silently_accepted(self) -> None:
        # The trap the item called out: `bool` is a subclass of `int`, so
        # `isinstance(True, int)` is true and a naive check accepts it. It is worse
        # than a type error because `True == 1`, so `Span(True, 5)` was a
        # *silently wrong* span at line 1 -- a plausible span, rendered and never
        # questioned. A false `end` is the same shape: `Span(1, False)` used to
        # construct, and its `len()` then depended on `max(0, ...)`.
        for start, end in ((True, 5), (1, True), (False, 5), (1, False)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(selectors.SelectorSyntaxError):
                    selectors.Span(start, end)

    def test_type_errors_say_which_end_is_wrong(self) -> None:
        # `start` and `end` fail for different reasons and a caller fixing an anchor
        # needs to know which one to edit. The loop over both fields could easily
        # report the wrong name, so pin it.
        with self.assertRaises(selectors.SelectorSyntaxError) as ctx:
            selectors.Span(1, None)
        self.assertIn("end=None", str(ctx.exception))
        with self.assertRaises(selectors.SelectorSyntaxError) as ctx:
            selectors.Span(None, 5)
        self.assertIn("start=None", str(ctx.exception))

    def test_a_non_integer_span_is_a_syntax_error_not_an_unresolved_one(self) -> None:
        # The distinction is load-bearing and easy to get backwards. Unresolved
        # means "matched, but not a region that exists" -- a stale anchor, which is
        # a content problem. A `Span("1", 5)` never matched anything and no source
        # text could make it valid, so it is a malformed *request*. Both are
        # `SelectorError`, so callers are unaffected, but a caller that reports the
        # two differently must not be told to go look at the source.
        with self.assertRaises(selectors.SelectorSyntaxError):
            selectors.Span("1", 5)
        with self.assertRaises(selectors.SelectorUnresolvedError):
            selectors.Span(0, 5)

    def test_an_offset_that_goes_off_the_top_is_still_a_selector_error(self) -> None:
        # The route that does reach it today raises earlier, in `apply_offset`.
        # Pinned so the two stay consistent: whichever guard fires, the caller
        # must see a SelectorError naming the line, never a ValueError.
        with self.assertRaises(selectors.SelectorUnresolvedError) as ctx:
            selectors.Span(1, 5).apply_offset((-1, 0))
        self.assertIsInstance(ctx.exception, selectors.SelectorError)

    def test_starting_past_the_end_is_an_error(self) -> None:
        # A SelectorError, not a bare ValueError: `tools/validate_concepts.py` and
        # `tools/gen_anchors.py` catch SelectorError to turn this into a clean
        # diagnostic, so a ValueError here escapes as a traceback instead.
        with self.assertRaises(selectors.SelectorUnresolvedError) as ctx:
            selectors.Span(500, 600).clamp(199)
        # Still a ValueError-compatible failure? No -- the module's contract is
        # SelectorError; assert the hierarchy rather than the old type.
        self.assertIsInstance(ctx.exception, selectors.SelectorError)
        message = str(ctx.exception)
        self.assertIn("500", message)
        self.assertIn("199", message)

    def test_length(self) -> None:
        self.assertEqual(len(selectors.Span(10, 19)), 10)
        self.assertEqual(len(selectors.Span(10, 10)), 1)

    def test_offset_round_trip(self) -> None:
        span = selectors.Span(10, 20).apply_offset((5, -5))
        self.assertEqual((span.start, span.end), (15, 15))

    def test_offset_that_inverts_the_span_is_an_error(self) -> None:
        # `offset:[0, -5000]` on a one-line span used to produce
        # `Span(start=29, end=-4929)`: `len()` said 0 and `clamp()` accepted it,
        # so the selector rendered nothing and no build step complained.
        with self.assertRaises(selectors.SelectorUnresolvedError) as ctx:
            selectors.Span(29, 29).apply_offset((0, -5000))
        message = str(ctx.exception)
        self.assertIn("29", message)
        self.assertIn("-4971", message)
        self.assertIn("0", message)
        self.assertIn("-5000", message)

    def test_offset_that_empties_the_span_is_an_error(self) -> None:
        # `Span(28, 29)` names one line; shifting the end back past the start is
        # an inverted span even though it is only one line deep.
        with self.assertRaises(selectors.SelectorUnresolvedError):
            selectors.Span(28, 29).apply_offset((0, -2))
        self.assertEqual((selectors.Span(28, 29).apply_offset((0, -1))).end, 28)


if __name__ == "__main__":
    unittest.main()
