#!/usr/bin/env python3
"""Check that `reference/microgpt-annotated.py` still has the reference's code.

## Why this exists

`reference/microgpt-annotated.py` is a heavily commented fork of the pinned
`reference/microgpt.py`. It is the file learners actually read, and every claim
it makes -- "this is the same algorithm, with commentary" -- is only true while
its *code* matches the reference. Nothing checked that. `make
verify-provenance` hashes the annotated file (`tools/provenance.py`), but its
entry is `derived=True`, which means the digest is *recorded*, not enforced: the
file is expected to change whenever we change it, and `make
verify-provenance-refresh` will happily rewrite the digest. So a stray edit to
the annotated file -- an operator, a constant, a dropped statement -- passed
every gate in this repository and quietly turned the reading aid into a
different program.

## Why the comparison is done on the AST, not the text

The two files are not line-parallel and a text comparison is mostly noise. On
the committed tree, `difflib` over the raw lines reports 409 changed lines: 359
added and 50 removed, of which 43 and 32 respectively survive comment stripping.
Those 32 "removed code" lines are almost all one-line `def`s that the annotated
file reformatted over several lines, plus one genuine merge of four assignments
into one. Verified empirically:

    reference/microgpt.py:           learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    reference/microgpt-annotated.py: four separate assignments, one per name

Parsing both files and comparing `ast.dump` with the formatting-only attributes
(`lineno`, `col_offset`, `end_*`, `type_comment`) blanked out removes all of
that noise: comments do not exist in the AST at all, indentation and line breaks
are irrelevant, and PEP 8 wrapping of one statement into several lines is the
*same node*. Measured: the whole-module normalised dump differs by 19 characters
out of ~37,000, and all 19 are that one merge. What survives is a comparison of
the executable structure.

## What it proves, and what it does not

Everything nested inside a `def` or `class` must be identical, node for node,
in every shared definition. That is the invariant the acceptance criteria ask
for: perturb any statement inside any function and this fails, and it fails even
when the perturbation is a new name the reference does not have (new code
changes both counts).

It is *not* a claim that the two files are the same program. Two real
differences exist at module level and both are deliberate, so both are handled
explicitly rather than papered over:

  1. Top-level statements are compared as an *unordered multiset* of normalised
     node dumps, not as a sequence. The annotated fork's commentary is attached
     as whole-line comments between top-level statements, so the two files'
     top-level statements are not in the same order, and this is exactly the
     difference the annotation pass is entitled to make. Dropping a top-level
     statement, changing one, or adding one still changes the multiset and still
     fails. The merge itself is the one allowance this module cannot infer from
     the tree.

     The price is real and worth naming: a *reorder* is invisible here, and a
     reorder is not always harmless. Swapping two independent constant
     assignments changes nothing a reader would notice, but hoisting
     `random.shuffle(docs)` above the `docs = [...]` it consumes is a genuine
     `NameError` that this comparison would pass. That edit is caught by the
     digest phase of `make verify-provenance` (the file is pinned, and a change
     moves its sha256) but *not* by this module alone when it is handed two
     strings -- a reader evaluating an annotated-file diff should read the
     top-level block, not trust the green line.

  2. Everything nested inside a `def` or `class` is compared *in order*, node
     for node, so a swapped pair of statements inside a function fails.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import io
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Run as `python3 -m tools.check_annotated_sync`; see tools/_entrypoint.py. This
# must precede `from tools import ...`: a script run leaves `tools/` on sys.path
# but not the repository root, so the package is not importable at all.
_spec = importlib.util.spec_from_file_location(
    "_tools_entrypoint", Path(__file__).resolve().parent / "_entrypoint.py"
)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
_guard.require_module_invocation("check_annotated_sync", __package__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The pinned reference and the annotated fork of it.
REFERENCE_PATH = "reference/microgpt.py"
ANNOTATED_PATH = "reference/microgpt-annotated.py"

#: Node attributes that encode where a node sits in the file, or how it was
#: typed by a comment. Two files that differ only in line numbers, indentation,
#: or PEP 8 line wrapping must compare equal, and these are the only attributes
#: that carry that information. Everything else in an `ast.dump` is structure.
_FORMAT_ONLY_ATTRIBUTES = (
    "lineno",
    "col_offset",
    "end_lineno",
    "end_col_offset",
    "type_comment",
)

#: Top-level statements the annotated fork is allowed to contain more than once.
#: Each entry is the statement's source text, matched after comment stripping,
#: and an entry that matches nothing in the file is itself a failure -- so the
#: list cannot rot into a rubber stamp. Empty today: the fork differs only in
#: comments, in splitting one assignment into four, and in line wrapping.
#:
#: Empty and unused, deliberately kept. Nothing needs it, but the tolerant path
#: it guards is the one place someone would reach for when a legitimate
#: repeated statement appears, and an accidental fifth copy of a statement
#: would otherwise be reported as an opaque "added 1x". A `main()` guard shared
#: by two tracks is the shape that would want it.
ALLOWED_TOP_LEVEL_REPEATS: tuple[str, ...] = ()

#: The one deliberate module-level difference between the two files, and the
#: reason the top-level comparison has an allowance at all. `reference/microgpt.py`
#: writes the four optimizer hyperparameters as one tuple-unpacking assignment;
#: the annotated fork writes four assignments so each name can carry a comment.
#: Both bind the same names to the same values, in the same order, and nothing
#: between the statements could observe the difference -- which is exactly why
#: this is allowed to be a formatting choice rather than drift.
#:
#: The allowance is located by name, not by source text: `_SPLIT_UNPACK_NAMES`
#: is the four names, and `_find_grouped_unpack` / `_find_simple_assignment`
#: find the two shapes. A comment or a rename of an *unrelated* variable cannot
#: make the allowance stop applying; a changed constant is reported as the
#: changed constant it is.
REFERENCE_GROUPED_UNPACK = "learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8"
_SPLIT_UNPACK_NAMES = ("learning_rate", "beta1", "beta2", "eps_adam")
ANNOTATED_SPLIT_UNPACK = (
    "learning_rate = 0.01",
    "beta1 = 0.85",
    "beta2 = 0.99",
    "eps_adam = 1e-8",
)

#: `ast.dump` uses the operator class name (`Mult`), the source uses the symbol.
#: Blanking an operator node's attributes destroys its identity for the
#: unparser as well, so symbols are re-stamped by hand. The diff of a re-merged
#: statement is far easier to read as source than as `dump` text.
_OPERATOR_SYMBOLS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.USub: "-",
    ast.UAdd: "+",
    ast.Invert: "~",
    ast.Not: "not",
    ast.And: "and",
    ast.Or: "or",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}


@dataclass(frozen=True)
class Drift:
    """One reason the annotated file is not the reference's code.

    `where` is the definition the drift sits in (`const module` for top level),
    so the message can point a reader at a place to look rather than at a
    whole-file dump. `diff` is a rendered unified diff of the mismatched node,
    when there is a single node to show; it is empty when the mismatch is a
    count (something added or missing).
    """

    where: str
    detail: str
    diff: str = field(default="")


def _normalise(tree: ast.AST) -> ast.AST:
    """Blank the formatting-only attributes so a dump carries only structure."""
    for node in ast.walk(tree):
        for attribute in _FORMAT_ONLY_ATTRIBUTES:
            if attribute in getattr(node, "_attributes", ()):
                setattr(node, attribute, None)
    return tree


def _operator_symbol(node: ast.AST) -> str:
    return _OPERATOR_SYMBOLS.get(type(node), type(node).__name__)


def unparse(node: ast.AST) -> str:
    """Source text for a node whose formatting attributes have been blanked.

    `ast.unparse` reads operator *symbols* off the operator node's attributes,
    and `_normalise` blanks attributes, so the symbol is copied onto a private
    `_symbol` field first. Without that, `1 - step / num_steps` unparses as
    `1 - step()`, which is unreadable in a failure message.
    """
    for child in ast.walk(node):
        if isinstance(child, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            child.__dict__["_symbol"] = _operator_symbol(child)
    return ast.unparse(node)


def _unified_diff(where: str, expected: str, actual: str) -> str:
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"{REFERENCE_PATH}:{where}",
            tofile=f"{ANNOTATED_PATH}:{where}",
            lineterm="",
            n=1,
        )
    )
    # A reader needs the shape of the change, not 200 lines of it. A truncated
    # diff still fails the build, which is the property that matters.
    if len(lines) > 40:
        lines = lines[:40] + ["... (diff truncated)"]
    return "\n".join(f"      {line}" for line in lines)


def _unpack_pair(
    source: str,
) -> tuple[tuple[str, ...], tuple[ast.expr, ...]] | None:
    """(target names, values) for an assignment that unpacks a tuple, else None.

    Scoped to "one assignment statement" on purpose: this is only used to
    recognise the documented split-unpack allowance, and a helper that tried to
    resolve names across statements would be a much bigger claim than the check
    is entitled to make.
    """
    try:
        node = ast.parse(source)
    except SyntaxError:  # pragma: no cover - the constant is always parseable
        return None
    if len(node.body) != 1:
        return None
    statement = node.body[0]
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        # A plain assignment is a one-name "unpack": `beta1 = 0.85` binds one
        # name to one value, which is what the split half of the allowance is.
        return (target.id,), (statement.value,)
    if not isinstance(target, ast.Tuple) or not isinstance(statement.value, ast.Tuple):
        return None
    if len(target.elts) != len(statement.value.elts):
        return None
    if not all(isinstance(element, ast.Name) for element in target.elts):
        return None
    return tuple(e.id for e in target.elts), tuple(statement.value.elts)


def _split_unpack_allowance(
    reference_source: str, annotated_source: str
) -> tuple[set[str], set[str], str]:
    """Handle the documented split-unpack difference; say what was not excused.

    Returns the two sets of *dumps that are only different because of it*: the
    grouped assignment from the reference and the split assignments from the
    annotated file. The pair is recognised only when the annotated module really
    contains four assignments binding those four names to those four values --
    the values are compared from the AST, where comments do not exist, so a
    trailing `# ...` on the split lines cannot make this fail -- and the two
    sides really do bind the same names to the same values. So the allowance
    cannot hide a changed constant; it can only excuse the regrouping of these
    exact four values.

    An absence returns two empty sets plus a one-line reason, which means "not
    excused": the difference then fails loudly as ordinary drift. The reason
    exists because a disabled allowance turns one changed constant into five
    confusing drifts ("added 4x ... missing 1x"); naming the cause in the report
    is what makes that readable.
    """
    # Both halves are located by the *names they bind*, not by their source
    # text. Matching on text would mean the allowance silently stops applying
    # the moment someone changes a constant -- and then the report would blame
    # the regrouping instead of the actual edit. Each side is looked up by name
    # and the values are compared node by node, so a changed constant is
    # reported as a changed constant.
    unpacked = _unpack_pair(REFERENCE_GROUPED_UNPACK)
    if unpacked is None:  # pragma: no cover - the constant is well-formed
        return set(), set(), ""
    names, values = unpacked

    excused_reference_node = _find_grouped_unpack(reference_source)
    if excused_reference_node is None:
        return set(), set(), ""
    excused_annotated_nodes: list[ast.AST] = []
    for index, split in enumerate(ANNOTATED_SPLIT_UNPACK):
        name = names[index]
        node = _find_simple_assignment(annotated_source, name)
        if node is None:
            return set(), set(), ""
        if ast.dump(node.value) != ast.dump(values[index]):
            return (
                set(),
                set(),
                f"the split of the four optimizer hyperparameters binds `{name}` "
                f"to `{unparse(node.value)}`, but the reference binds it to "
                f"`{unparse(values[index])}`",
            )
        excused_annotated_nodes.append(node)

    excused_reference = {ast.dump(excused_reference_node, include_attributes=False)}
    excused_annotated = {
        ast.dump(node, include_attributes=False) for node in excused_annotated_nodes
    }
    return excused_reference, excused_annotated, ""

def _top_level_key(source: str) -> str:
    """The normalised dump of the one-statement source `source`."""
    return ast.dump(_normalise(ast.parse(source)).body[0], include_attributes=False)


def _find_grouped_unpack(source: str) -> ast.Assign | None:
    """The top-level tuple-unpacking assignment binding the four hyperparameters.

    Found by structure, not by text: a single assignment whose target is a tuple
    of the four names. Answers None when the reference no longer has that shape,
    which disables the allowance and lets the difference surface as ordinary
    drift.
    """
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple)
        ):
            bound = [elt.id for elt in node.targets[0].elts if isinstance(elt, ast.Name)]
            if len(bound) == len(node.targets[0].elts) and tuple(bound) == _SPLIT_UNPACK_NAMES:
                return node
    return None


def _find_simple_assignment(source: str, name: str) -> ast.Assign | None:
    """The top-level `name = <expr>` assignment, if there is exactly one."""
    found = None
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            if found is not None:
                return None
            found = node
    return found


def _nested_definitions(tree: ast.Module) -> dict[tuple[str, str], ast.AST]:
    """Every `def` and `class` in the module, keyed by (kind, name).

    Nested functions are included. Two same-named definitions in different
    scopes would collide; microgpt has none, and the collision would show up as
    a wrong comparison rather than a crash, so the key is deliberately
    qualified by the node kind to make that unlikely.
    """
    definitions: dict[tuple[str, str], ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.setdefault((type(node).__name__, node.name), node)
    return definitions


def _source_segments(source: str) -> list[str]:
    """The source text of each top-level statement, in order.

    Read from the AST rather than by scanning lines, so a multi-line statement
    comes back whole.
    """
    lines = source.splitlines()
    segments: list[str] = []
    for node in ast.parse(source).body:
        start = (node.lineno or 1) - 1
        end = node.end_lineno or start + 1
        segments.append("\n".join(lines[start:end]).strip())
    return segments


def _code_segments(source: str) -> list[str]:
    """Like `_source_segments`, but with comments stripped from each statement.

    The annotated fork puts `# ...` on the split hyperparameter assignments
    (`beta1 = 0.85 # Decay rate for first moment (momentum)`), and an allowance
    whose existence test compared raw text would be a hidden "you may not
    comment these lines" rule. Comments are not code, this module compares code,
    and so the test has to ignore them too. Python's own tokenizer does the
    stripping, so a `#` inside a string literal is not mistaken for a comment.
    """
    stripped: list[str] = []
    for segment in _source_segments(source):
        pieces: list[str] = []
        for token in tokenize.generate_tokens(io.StringIO(segment).readline):
            if token.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue
            pieces.append(token.string)
        stripped.append(" ".join(pieces).strip())
    return stripped


def compare(reference_source: str, annotated_source: str) -> list[Drift]:
    """Return every way `annotated_source` differs from `reference_source`.

    Pure and total: it never touches the filesystem, so a test can call it on
    two strings. An empty list means the annotated file carries the reference's
    code modulo comments, whitespace, line wrapping, and the reordering of
    top-level statements.
    """
    reference_tree = _normalise(ast.parse(reference_source))
    annotated_tree = _normalise(ast.parse(annotated_source))

    drifts: list[Drift] = []

    reference_defs = _nested_definitions(reference_tree)
    annotated_defs = _nested_definitions(annotated_tree)
    for key in sorted(reference_defs):
        where = f"{key[0]} {key[1]}"
        if key not in annotated_defs:
            drifts.append(Drift(where, "missing from the annotated file entirely"))
            continue
        expected = ast.dump(reference_defs[key], include_attributes=False)
        actual = ast.dump(annotated_defs[key], include_attributes=False)
        if expected != actual:
            drifts.append(
                Drift(
                    where,
                    "the code inside it does not match the reference",
                    _unified_diff(
                        where,
                        unparse(reference_defs[key]),
                        unparse(annotated_defs[key]),
                    ),
                )
            )
    for key in sorted(set(annotated_defs) - set(reference_defs)):
        drifts.append(
            Drift(
                f"{key[0]} {key[1]}",
                "defined here but not in the reference: the fork has grown a definition",
                _unified_diff(f"{key[0]} {key[1]}", "", unparse(annotated_defs[key])),
            )
        )

    # Top-level statements, as an unordered multiset of normalised dumps. See
    # the module docstring for why the order is deliberately not compared.
    reference_top = Counter(ast.dump(n, include_attributes=False) for n in reference_tree.body)
    annotated_top = Counter(ast.dump(n, include_attributes=False) for n in annotated_tree.body)

    # A repeated statement is tolerated only if its *source text* is listed.
    # Allowances are consumed before the comparison, and an unconsumed (stale)
    # allowance is itself a drift, so the list cannot become a rubber stamp.
    segments = Counter(_code_segments(annotated_source))
    for allowed in ALLOWED_TOP_LEVEL_REPEATS:
        if not segments[allowed]:
            drifts.append(
                Drift(
                    "const module",
                    "ALLOWED_TOP_LEVEL_REPEATS names a statement the annotated "
                    f"file does not contain, so the allowance is stale:\n{allowed}",
                )
            )
            continue
        key = _top_level_key(allowed)
        annotated_top[key] -= 1
        if annotated_top[key] <= 0:
            del annotated_top[key]

    # Take the documented split-unpack difference off both multisets before
    # comparing. Everything else at module level is compared strictly.
    excused_reference, excused_annotated, allowance_reason = _split_unpack_allowance(
        reference_source, annotated_source
    )
    if allowance_reason:
        drifts.append(
            Drift(
                "const module",
                "the documented split-unpack allowance does not apply, so the "
                f"regrouped hyperparameters count as ordinary drift: {allowance_reason}",
            )
        )
    for key in excused_reference:
        reference_top[key] -= 1
        if reference_top[key] <= 0:
            del reference_top[key]
    for key in excused_annotated:
        annotated_top[key] -= 1
        if annotated_top[key] <= 0:
            del annotated_top[key]

    added = annotated_top - reference_top
    missing = reference_top - annotated_top

    for key, count in sorted(added.items()):
        node = _find_top_level(annotated_source, key)
        text = unparse(node) if node is not None else key
        drift_diff = ""
        if node is not None:
            drift_diff = _unified_diff(
                "const module", _find_matching_top_level(reference_source, node), text
            )
        drifts.append(
            Drift(
                "const module",
                f"added {count}x at top level; the reference has no such statement",
                drift_diff,
            )
        )
    for key, count in sorted(missing.items()):
        node = _find_top_level(reference_source, key)
        text = unparse(node) if node is not None else key
        drifts.append(
            Drift(
                "const module",
                f"missing {count}x at top level, present in the reference",
                _unified_diff("const module", text, ""),
            )
        )

    return drifts


def _find_top_level(source: str, key: str) -> ast.AST | None:
    """The first top-level statement in `source` whose normalised dump is `key`."""
    for node in _normalise(ast.parse(source)).body:
        if ast.dump(node, include_attributes=False) == key:
            return node
    return None


def _find_matching_top_level(source: str, node: ast.AST) -> str:
    """The source of the top-level statement in `source` that `node` was split from.

    Only used to make the diff readable, so it is best-effort: it pairs the
    added assignment with the reference statement containing the same name, and
    returns "" when nothing matches.
    """
    names = {t.id for t in ast.walk(node) if isinstance(t, ast.Name)}
    best = ""
    for candidate in ast.parse(source).body:
        candidate_names = {t.id for t in ast.walk(candidate) if isinstance(t, ast.Name)}
        if names & candidate_names:
            text = unparse(_normalise(candidate))
            if not best or len(text) < len(best):
                best = text
    return best


def format_report(drifts: list[Drift]) -> str:
    """Render the drifts as one loud, actionable block for stderr."""
    lines = [
        f"annotated-sync check FAILED: {ANNOTATED_PATH} no longer carries the code "
        f"of {REFERENCE_PATH}.",
        "",
    ]
    for drift in drifts:
        lines.append(f"  ! {drift.where}: {drift.detail}")
        if drift.diff:
            lines.append(drift.diff)
    lines += [
        "",
        f"{ANNOTATED_PATH} is a reading aid for the pinned {REFERENCE_PATH}: the same",
        "algorithm with our commentary. When the reference moves, move this file with",
        "it in the same commit. If a difference is deliberate, a repeated top-level",
        "statement can be named in ALLOWED_TOP_LEVEL_REPEATS in this module -- with a",
        "sentence saying why. Inside a function there is no allowance at all, because",
        "that is where the annotated file's claim about the algorithm is on the line.",
    ]
    return "\n".join(lines)


def run() -> int:
    reference_file = REPO_ROOT / REFERENCE_PATH
    annotated_file = REPO_ROOT / ANNOTATED_PATH

    for path in (reference_file, annotated_file):
        if not path.is_file():
            print(f"annotated-sync check FAILED: {path} is missing", file=sys.stderr)
            return 1

    reference_source = reference_file.read_text(encoding="utf-8")
    annotated_source = annotated_file.read_text(encoding="utf-8")

    for label, source in ((REFERENCE_PATH, reference_source), (ANNOTATED_PATH, annotated_source)):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            print(
                f"annotated-sync check FAILED: {label} does not parse: {exc}",
                file=sys.stderr,
            )
            return 1

    drifts = compare(reference_source, annotated_source)
    if drifts:
        print(format_report(drifts), file=sys.stderr)
        return 1

    definitions = _nested_definitions(_normalise(ast.parse(reference_source)))
    print(f"ok  {ANNOTATED_PATH}")
    print(f"    {len(definitions)} definitions match {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
