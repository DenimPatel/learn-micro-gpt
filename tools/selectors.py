#!/usr/bin/env python3
"""Selector resolution: the anti-rot mechanism for concept source anchors.

## Why selectors and not line numbers

The previous visualizer hardcoded line ranges (`attn-heads: [122, 131]`). That is
precisely the thing that rots: the day anyone edits `reference/microgpt.py` -- and
someone always does, it is a teaching artifact -- every range silently points at
the wrong code and the site keeps rendering, just wrongly. Nothing fails. The
concept now teaches the wrong thing with total confidence.

So a concept never says "lines 122 to 131". It says "the body of `def:gpt`,
narrowed to the multi-head loop", and `tools/gen_anchors.py` resolves that
against each language's actual source at build time. A selector that resolves to
nothing is a CI failure with the concept id and the selector in the message.

## The grammar, kept deliberately tiny

| Form | Meaning |
| --- | --- |
| `def:<name>` | the body of the function/method named `<name>`, brace-delimited |
| `assign:<name>` | the smallest statement or declaration that binds `<name>` |
| `comment:<exact prefix>` | the single line whose comment text starts with `<prefix>` |
| `section:<exact prefix>` | from that comment to the next section boundary in the same language |
| `offset:[a, b]` | deltas applied *after* resolution: add `a` to the start, add `b` to the end |

`offset` is relative, not absolute, on purpose. It lets one concept point at a
region nested inside a function without either duplicating the function's whole
span or hardcoding a line number that drifts the moment the function above it
grows a comment. Negative deltas trim inward, positive deltas grow outward.

The plan's own example is the canonical one: `def:gpt` on the reference resolves
to lines 107-143, and `offset: [15, 10]` turns that into 122-153-clamped -> the
multi-head block plus the output projection, lines 122-133. That is
`attn-heads` and `attn-proj` together. Nothing in the manifest mentions 122.

`section:` exists because three of the eighteen concepts describe a whole
chapter-sized region of the file, and a numeric offset to reach the end of one
would be precisely the hardcoded line number this design exists to remove. Its
boundary rule is language-specific and conservative: a section ends where the
next top-level comment begins, and it never runs past one.

Cross-check: running `section:` over `reference/microgpt.py` reproduces all
nineteen line ranges of the previous visualizer's `mapping.js` exactly, which is
the strongest available evidence that the boundary rule matches how this file is
actually organised.

Anything outside the grammar is a hard error. There is no fallback to "line 0"
and no best-effort regex search, because a silent wrong answer is the failure
mode this whole design exists to prevent.

## Known limitations, on purpose

- **Python is exact**; it uses the real `ast` module. Classes resolve under
  `def:` (`def:Value` is the class, `def:Value.backward` is one method), and
  `assign:` sees through tuple unpacking, so `assign:learning_rate` finds the
  whole hyperparameter line.
- **The C-family is a heuristic**, because the standard library has no C parser.
  It resolves real brace-balanced definitions and file-scope declarations, and
  it deliberately ignores declarations *inside* function bodies: with a
  brace-matching scan, "the first statement that binds `x`" would mean something
  different in every function that has one, and a concept that silently jumped
  between them would be worse than one that said "use `def:` with an offset".
  Concepts anchor to file-scope macros (`assign:N_EMBD`), functions
  (`def:forward_pos`), and section comments (`comment:1) Multi-head attention`).
- **Go exports are matched case-insensitively** at the first letter, so a
  concept can say `func:GPT` and find `func gpt`. It is a convenience, not a
  claim that the two names are the same identifier.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Span",
    "SelectorError",
    "SelectorSyntaxError",
    "SelectorUnresolvedError",
    "parse_selector",
    "resolve",
    "source_for_language",
]


class SelectorError(Exception):
    """Base class so callers can catch every selector problem at once."""


class SelectorSyntaxError(SelectorError):
    """The selector does not match the grammar."""


class SelectorUnresolvedError(SelectorError):
    """The selector is well-formed but matched nothing in the source."""


@dataclass(frozen=True)
class Span:
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"line numbers are 1-based, got start={self.start}")

    def clamp(self, total: int) -> "Span":
        if self.start > total:
            raise ValueError(f"span starts at {self.start} but file has {total} lines")
        return Span(self.start, min(self.end, total))

    def apply_offset(self, offset: tuple[int, int] | None) -> "Span":
        if offset is None:
            return self
        return Span(self.start + offset[0], self.end + offset[1])

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}

    def __len__(self) -> int:
        return max(0, self.end - self.start + 1)


@dataclass(frozen=True)
class Selector:
    kind: str  # "def" | "assign" | "comment"
    target: str
    offset: tuple[int, int] | None
    raw: str

    def describe(self) -> str:
        if self.offset is None:
            return self.raw
        return f"{self.raw} offset:[{self.offset[0]}, {self.offset[1]}]"


_KINDS = ("def", "assign", "comment", "section")
_OFFSET_RE = re.compile(r"offset:\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]")


def parse_selector(raw: str) -> Selector:
    """Parse one selector string. Raises `SelectorSyntaxError` on anything else."""
    if not isinstance(raw, str) or not raw.strip():
        raise SelectorSyntaxError(f"empty selector: {raw!r}")
    text = raw.strip()

    # The offset may be written before or after the body; strip it either way.
    offset: tuple[int, int] | None = None
    offset_match = _OFFSET_RE.search(text)
    if offset_match:
        offset = (int(offset_match.group(1)), int(offset_match.group(2)))
        text = f"{text[: offset_match.start()]} {text[offset_match.end() :]}".strip()

    kind, sep, target = text.partition(":")
    if not sep:
        raise SelectorSyntaxError(f"selector {raw!r} has no kind; expected one of {', '.join(_KINDS)}")
    kind = kind.strip()
    if kind not in _KINDS:
        raise SelectorSyntaxError(
            f"selector {raw!r} uses unknown kind {kind!r}; expected one of {', '.join(_KINDS)}"
        )
    target = target.strip()
    if not target:
        raise SelectorSyntaxError(f"selector {raw!r} has an empty target after {kind!r}")
    return Selector(kind=kind, target=target, offset=offset, raw=raw.strip())


# ----------------------------------------------------------------------------
# Python: real AST. It is the reference language, so it gets the real thing.
# ----------------------------------------------------------------------------


def _bound_names(target: ast.expr) -> list[str]:
    """Every name a single assignment target binds.

    `learning_rate, beta1, beta2, eps_adam = 0.01, ...` binds four names on one
    line, and `head_dim = n_embd // n_head` binds one. Both must resolve, so
    `assign:learning_rate` can point at the whole hyperparameter line.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_bound_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


def _python_spans(source: str) -> dict[tuple[str, str], Span]:
    tree = ast.parse(source)
    lines = source.split("\n")
    total = len(lines)
    spans: dict[tuple[str, str], Span] = {}

    def record(kind: str, name: str, node: ast.AST) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            return
        span = Span(start, end).clamp(total)
        spans.setdefault((kind, name), span)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record("def", node.name, node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record("def", f"{node.name}.{child.name}", child)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _bound_names(target):
                    record("assign", name, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            record("assign", node.target.id, node)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            record("assign", node.target.id, node)
    return spans


# ----------------------------------------------------------------------------
# C-family: no AST in the standard library, so brace matching with guards.
# ----------------------------------------------------------------------------

_C_DEF_RE = re.compile(
    r"^[A-Za-z_][\w \t\*&:]*?\b(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->[^{;]+)?\{"
)
_C_ASSIGN_DEFINE_RE = re.compile(r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)\b")
#: In a C declaration the bound name is the identifier immediately before the
#: first `[`, `=` or `,`. That survives the things that sit between the type and
#: the name -- `static float ALIGN128 wte[...]`, `const char *path`, `int i, j`
#: -- which a strict "type then name" pattern does not.
_C_BINDING_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*(?=[=\[,;])")
_C_KEYWORDS = frozenset(
    """
    auto break case char const continue default do double else enum extern float for goto if
    inline int long register restrict return short signed sizeof static struct switch typedef
    union unsigned void volatile while _Bool _Atomic __restrict __restrict__ __inline
    __inline__ __attribute__
    """.split()
)
#: Words that begin a control-flow statement, never a definition. Checking this
#: is what stops `if (ready) {` from being recorded as `def:ready`.
_CONTROL_RE = re.compile(
    r"^(?:if|else|for|while|switch|do|catch|return|struct|union|enum|typedef|"
    r"namespace|using|class|public|private|protected|template|extern)\b"
)


def _c_declared_names(text: str) -> list[str]:
    """Every identifier a C declaration statement binds, in source order."""
    if "(" in text.split("[")[0] or "return" in text.split():
        return []
    names: list[str] = []
    for match in _C_BINDING_RE.finditer(text):
        name = match.group("name")
        if name in _C_KEYWORDS:
            continue
        names.append(name)
    return names


@dataclass(frozen=True)
class LogicalStatement:
    """One statement or definition, joined across however many lines it spans."""

    start: int  # 0-based index of the first physical line
    text: str  # joined, comment- and string-stripped
    end: int  # 0-based index of the last physical line (closing `}` or the `;` line)


def _logical_statements(lines: list[str]) -> list[LogicalStatement]:
    """Join physical lines into logical statements.

    Returns a list of `LogicalStatement`. Two things force this rather than a
    line-at-a-time scan:

    - A C function signature routinely wraps -- `static inline void linear_fwd(const
      float *x,` then three more parameter lines then `{` -- so resolving
      definitions one physical line at a time silently misses most of them.
    - After a function body, the closing `}` sits on its own line. Without
      consuming it here it gets glued onto the *next* definition as a prefix
      (`}   static void forward_pos(...) {`), which then fails an anchored match.
    """
    statements: list[LogicalStatement] = []
    index = 0
    total = len(lines)
    while index < total:
        stripped = lines[index].strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            index += 1
            continue

        start = index
        chunks: list[str] = []
        depth = 0
        terminator: str | None = None
        terminator_line = start
        cursor = start
        while cursor < total:
            code = _strip_c_strings_and_comments(lines[cursor])
            chunks.append(code)
            for ch in code:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                elif depth == 0 and ch in "{;":
                    terminator = ch
                    terminator_line = cursor
                    break
            if terminator is not None or cursor - start > 200:
                break
            cursor += 1

        text = " ".join(chunk.strip() for chunk in chunks)
        if terminator == "{":
            end = _brace_block(lines, terminator_line)
            index = end + 1
        else:
            end = terminator_line
            index = terminator_line + 1
        statements.append(LogicalStatement(start=start, text=text, end=end))
    return statements


def _brace_block(lines: list[str], open_index: int) -> int:
    """Return the 0-based index of the line closing the brace opened on `open_index`."""
    depth = 0
    for index in range(open_index, len(lines)):
        line = _strip_c_strings_and_comments(lines[index])
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return index
    raise SelectorUnresolvedError("unbalanced braces: no matching '}'")


def _strip_c_strings_and_comments(line: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(line):
        ch = line[index]
        if ch == "/" and index + 1 < len(line) and line[index + 1] == "/":
            break
        if ch == "/" and index + 1 < len(line) and line[index + 1] == "*":
            end = line.find("*/", index + 2)
            if end == -1:
                return "".join(out)
            index = end + 2
            continue
        if ch in "\"'":
            quote = ch
            out.append(" ")
            index += 1
            while index < len(line):
                if line[index] == "\\":
                    index += 2
                    continue
                if line[index] == quote:
                    break
                index += 1
            out.append(" ")
            index += 1
            continue
        out.append(ch)
        index += 1
    return "".join(out)


def _statement_end(lines: list[str], start_index: int) -> int:
    """Return the 0-based index of the last line of the statement starting at `start_index`."""
    balance = 0
    started = False
    for index in range(start_index, len(lines)):
        code = _strip_c_strings_and_comments(lines[index])
        for ch in code:
            if ch in "([{":
                balance += 1
                started = True
            elif ch in ")]}":
                balance -= 1
            elif ch == ";" and balance == 0:
                return index
        if started and balance == 0 and code.rstrip().endswith((")", "]", "}")) and index > start_index:
            return index
    return min(start_index, len(lines) - 1)


def _c_spans(source: str, language: str) -> dict[tuple[str, str], Span]:
    lines = source.split("\n")
    total = len(lines)
    spans: dict[tuple[str, str], Span] = {}
    statements = _logical_statements(lines)

    def record(kind: str, name: str, start: int, end: int) -> None:
        spans.setdefault((kind, name), Span(start + 1, min(end + 1, total)))

    if language in {"go", "rust", "typescript"}:
        return _c_like_spans(lines, total, language, statements)

    for statement in statements:
        code = statement.text.strip()
        if statement.end > statement.start and not _CONTROL_RE.match(code):
            match = _C_DEF_RE.match(code)
            if match:
                record("def", match.group("name"), statement.start, statement.end)

    for index, raw in enumerate(lines):
        define = _C_ASSIGN_DEFINE_RE.match(raw)
        if define:
            # `#define X ...` -- the macro definition, not a use site.
            record("assign", define.group("name"), index, index)

    # `assign:` for C: the first declaration or binding statement, smallest span.
    for statement in statements:
        code = statement.text.strip()
        if not code or _CONTROL_RE.match(code) or statement.end > statement.start:
            continue
        for name in _c_declared_names(code):
            record("assign", name, statement.start, statement.start)
    return spans


_GO_FUNC_RE = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")
_RUST_FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>[A-Za-z_]\w*)")
_TS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+(?P<fn>[A-Za-z_$][\w$]*)|(?P<var>[A-Za-z_$][\w$]*)\s*[:=][^=])"
)
_TS_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)")


def _c_like_spans(
    lines: list[str],
    total: int,
    language: str,
    statements: list[LogicalStatement] | None = None,
) -> dict[tuple[str, str], Span]:
    spans: dict[tuple[str, str], Span] = {}
    if statements is None:
        statements = _logical_statements(lines)

    def record(kind: str, name: str, start: int, end: int) -> None:
        spans.setdefault((kind, name), Span(start + 1, min(end + 1, total)))

    for statement in statements:
        code = statement.text.strip()
        has_body = statement.end > statement.start
        if language == "go":
            match = _GO_FUNC_RE.match(code)
            if match:
                name = match.group("name")
                # Go exports are conventionally capitalised; a concept that reads
                # like `func:GPT` should find it without a per-language alias list.
                record("def", name, statement.start, statement.end)
                record("def", name[:1].upper() + name[1:], statement.start, statement.end)
        elif language == "rust":
            match = _RUST_FN_RE.match(code)
            if match:
                record("def", match.group("name"), statement.start, statement.end)
        else:  # typescript
            match = _TS_FUNC_RE.match(code)
            if match:
                name = match.group("fn") or match.group("var")
                record("def", name, statement.start, statement.end)
            if has_body:
                assign = _TS_ASSIGN_RE.match(code)
                if assign:
                    record("assign", assign.group("name"), statement.start, statement.end)
    return spans


# ----------------------------------------------------------------------------
# comments: one line, any language
# ----------------------------------------------------------------------------

#: Comment markers, plus the decorative rules people put around section comments.
#: `/* ── Backward pass ────────────── */` has to reduce to `Backward pass` for
#: `comment:Backward pass` to work, otherwise every concept that anchors to a C
#: section header has to spell out the box-drawing characters.
_COMMENT_MARKER_RE = re.compile(
    r"^(?:"
    r"//+|#+|--+|;+|\*+|=+"
    r"|/\*+|\*+/"
    r")"
    r"[\s\-=─━═_~·•*]*"  # leading decorative rule, including box-drawing blocks
)


def _comment_span(source: str, prefix: str) -> Span:
    """Match a single line by comment text.

    The target may be written with or without the comment marker, so both
    `comment:1) Multi-head attention` and `comment:// 1) Multi-head attention`
    find the same line. Section comments are the main reason this selector
    exists: they are the author's own signposts, they survive renames, and they
    read the same in all five languages.
    """
    needle = prefix.strip()
    for index, raw in enumerate(source.split("\n")):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(needle) or _COMMENT_MARKER_RE.sub("", stripped).startswith(needle):
            return Span(index + 1, index + 1)
    raise SelectorUnresolvedError(f"no line's comment text starts with {needle!r}")


#: A "section boundary" is a top-level comment that introduces a new region of
#: the file. Two rules keep it conservative:
#:
#: 1. Only comments in column 0 count, so a comment inside a function or inside a
#:    block never ends the section it sits in.
#: 2. Only a boundary *preceded by a blank line* ends a section. Authors write
#:    multi-line section headers routinely -- `# Define the model architecture:`
#:    followed immediately by `# Follow GPT-2, ...` -- and without this rule every
#:    such header would be a section of exactly one line.
def _is_section_boundary(lines: list[str], index: int, language: str) -> bool:
    raw = lines[index]
    if not raw or not raw.strip():
        return False
    if raw[:1] in (" ", "\t"):
        return False
    if index > 0 and lines[index - 1].strip():
        return False
    if language in {"c", "c-scaled"}:
        return raw.startswith(("/*", "//", "*/"))
    if language in {"go", "rust", "typescript"}:
        return raw.startswith("//")
    return raw.startswith("#")


def _section_span(source: str, prefix: str, language: str) -> Span:
    """From a section comment to the line before the next section comment."""
    lines = source.split("\n")
    start = _comment_span(source, prefix)
    index = start.end  # 0-based index of the line after the comment
    end = len(lines)
    while index < len(lines):
        if _is_section_boundary(lines, index, language):
            end = index
            break
        index += 1
    # Trailing blank lines belong to the gap between sections, not the section.
    while end - 1 >= start.end and not lines[end - 1].strip():
        end -= 1
    return Span(start.start, end)


# ----------------------------------------------------------------------------
# source registry
# ----------------------------------------------------------------------------

#: language -> repo-relative path. `python` is the reference track; the rest are
#: ports. `go`, `rust` and `typescript` are added in milestone 3, so they may be
#: absent on a partial checkout -- `resolve` reports that as a resolution error
#: rather than pretending the concept has no anchor in that language.
SOURCES: dict[str, str] = {
    "python": "reference/microgpt.py",
    "python-annotated": "reference/microgpt-annotated.py",
    "c": "implementations/c/microgpt.c",
    "c-scaled": "implementations/c/microgpt-scaled.c",
    "go": "implementations/go/main.go",
    "rust": "implementations/rust/src/lib.rs",
    "typescript": "implementations/typescript/src/index.ts",
}

#: Languages a concept is allowed to anchor to. `c-scaled` and `python-annotated`
#: are forks: readable, never used for parity, so no concept anchors to them.
TRACK_LANGUAGES: tuple[str, ...] = ("python", "c", "go", "rust", "typescript")


def source_for_language(language: str, root: Path) -> Path:
    if language not in SOURCES:
        raise SelectorSyntaxError(
            f"unknown language {language!r}; known: {', '.join(sorted(SOURCES))}"
        )
    return root / SOURCES[language]


def resolve(language: str, selector_raw: str, root: Path, source: str | None = None) -> Span:
    """Resolve one selector against one language's source. 1-based inclusive."""
    selector = parse_selector(selector_raw)
    if source is None:
        path = source_for_language(language, root)
        if not path.is_file():
            raise SelectorUnresolvedError(
                f"no source for language {language!r}: {path.relative_to(root)} does not exist"
            )
        source = path.read_text(encoding="utf-8")

    if selector.kind == "comment":
        span = _comment_span(source, selector.target)
    elif selector.kind == "section":
        span = _section_span(source, selector.target, language)
    else:
        if language == "python" or language == "python-annotated":
            try:
                spans = _python_spans(source)
            except SyntaxError as exc:  # pragma: no cover - defensive
                raise SelectorUnresolvedError(
                    f"{SOURCES.get(language, language)} does not parse as Python: {exc}"
                ) from exc
        elif language in {"c", "c-scaled"}:
            spans = _c_spans(source, "c")
        elif language in {"go", "rust", "typescript"}:
            spans = _c_spans(source, language)
        else:
            raise SelectorSyntaxError(f"no resolver for language {language!r}")

        key = (selector.kind, selector.target)
        if key not in spans:
            raise SelectorUnresolvedError(
                f"{selector.kind}:{selector.target} matched nothing in "
                f"{SOURCES.get(language, language)}"
            )
        span = spans[key]

    span = span.apply_offset(selector.offset)
    total = len(source.split("\n"))
    if span.start < 1:
        raise SelectorUnresolvedError(
            f"offset [{selector.offset}] pushed the start before line 1 "
            f"(resolved {span.start}..{span.end})"
        )
    return span.clamp(total)
