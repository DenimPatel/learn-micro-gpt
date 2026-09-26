#!/usr/bin/env python3
"""Validate the concept corpus. This is the gate that keeps the Atlas honest.

Run via `make validate`, which also runs `tools/gen_anchors.py`. Every check
here corresponds to a way a teaching repository rots in a way that *looks* fine:

| check | the rot it prevents |
| --- | --- |
| schema conformance | a concept that renders without its shape table |
| `id` == filename, ids unique | a concept nothing links to, or two that collide |
| `index.json` order == frontmatter order | the curriculum and the graph disagreeing |
| prereqs/related resolve | a "see also" link to a concept that does not exist |
| no prereq cycles | a reading order that cannot be walked |
| `anchors.python` required | a concept with no code to show, on a site about code |
| anchor languages have sources | a tab that renders empty forever |
| selectors resolve (in gen_anchors) | a code panel highlighting the wrong lines |
| `:::` directives are known and closed | a body that silently loses half its content |
| `:::term` is in the glossary | a link to a definition that is not there |
| `:::trace` args are in range | a widget that renders "undefined" forever |
| reference sha256 pinned | every recorded number silently becoming a lie |
| trace size budget | a repo that is 400 MB because traces grew |

Exit code 0 means the Atlas can be trusted to say what it says.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import selectors  # noqa: E402
from tools.concepts import (  # noqa: E402
    CONCEPTS_DIR,
    DIRECTIVE_SPEC,
    ContentError,
    Concept,
    load_all,
    load_index,
    load_schema,
)

#: Per-track trace budget. At micro hyperparams (block_size 16, n_head 4) a
#: trace is kilobytes; anything approaching this means something started
#: recording activations instead of summaries.
TRACE_BUDGET_BYTES = 2 * 1024 * 1024


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.error(message)
        return condition


# ----------------------------------------------------------------------------
# a minimal JSON Schema check, covering exactly the keywords schema.json uses
# ----------------------------------------------------------------------------

_SUPPORTED = {
    "$schema", "$id", "$comment", "title", "description", "type", "required",
    "properties", "additionalProperties", "items", "enum", "minLength",
    "maxLength", "minimum", "maximum", "pattern", "uniqueItems", "minItems",
    "maxItems", "definitions", "$ref",
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _resolve_ref(ref: str, schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContentError(f"schema $ref {ref!r} is not a local pointer")
    node: Any = schema
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    report: Report,
    path: str,
    root: dict[str, Any] | None = None,
) -> None:
    """Validate against the keyword subset this repository's schema uses.

    Unknown keywords in the schema are a hard error, not a silent pass: a schema
    that grows `patternProperties` and a validator that ignores it is a validator
    that has quietly stopped validating.
    """
    for keyword in schema:
        if keyword not in _SUPPORTED:
            report.error(f"{path}: schema uses unsupported keyword {keyword!r}")

    if "$ref" in schema:
        # A local pointer is always resolved against the document root, not
        # against whichever subschema happens to contain the `$ref`.
        base = root if root is not None else schema
        validate_against_schema(
            value, _resolve_ref(schema["$ref"], base), report, path, base
        )
        return

    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        report.error(
            f"{path}: expected {expected}, got {type(value).__name__}"
        )
        return

    if "enum" in schema and value not in schema["enum"]:
        report.error(f"{path}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            report.error(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            report.error(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema:
            import re

            if not re.search(schema["pattern"], value):
                report.error(f"{path}: {value!r} does not match /{schema['pattern']}/")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            report.error(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            report.error(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                report.error(f"{path}: missing required key {name!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    report.error(f"{path}: unexpected key {name!r}")
        for name, sub in properties.items():
            if name in value:
                validate_against_schema(
                    value[name], sub, report, f"{path}.{name}", root or schema
                )

    if isinstance(value, list):
        if schema.get("uniqueItems") and len(value) != len({json.dumps(v, sort_keys=True) for v in value}):
            report.error(f"{path}: items are not unique")
        if "minItems" in schema and len(value) < schema["minItems"]:
            report.error(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            report.error(f"{path}: more than maxItems {schema['maxItems']}")
        if "items" in schema:
            for position, item in enumerate(value):
                validate_against_schema(
                    item, schema["items"], report, f"{path}[{position}]", root or schema
                )


# ----------------------------------------------------------------------------
# individual checks
# ----------------------------------------------------------------------------


def check_schema(concepts: Iterable[Concept], schema: dict[str, Any], report: Report) -> None:
    for concept in concepts:
        validate_against_schema(
            concept.frontmatter, schema, report, f"{concept.path.name}:frontmatter", schema
        )


def check_graph(
    concepts: list[Concept], index: dict[str, Any], report: Report
) -> None:
    by_id = {c.id: c for c in concepts}

    if len(by_id) != len(concepts):
        seen: set[str] = set()
        for concept in concepts:
            if concept.id in seen:
                report.error(f"duplicate concept id {concept.id!r}")
            seen.add(concept.id)

    chapters = index.get("chapters", [])
    if not chapters:
        report.error("index.json declares no chapters")
        return

    chapter_ids = [c.get("id") for c in chapters]
    if len(chapter_ids) != len(set(chapter_ids)):
        report.error("index.json has duplicate chapter ids")
    orders = [c.get("order") for c in chapters]
    if orders != sorted(orders):
        report.error(f"index.json chapters are not in ascending order: {orders}")

    listed: list[str] = []
    for chapter in chapters:
        chapter_id = chapter.get("id")
        members = chapter.get("concepts", [])
        if not members:
            report.error(f"chapter {chapter_id!r} has no concepts")
        expected_order = list(range(1, len(members) + 1))
        actual_order = [
            by_id[m].frontmatter.get("order") for m in members if m in by_id
        ]
        if actual_order != expected_order:
            report.error(
                f"chapter {chapter_id!r}: index.json lists {members} but the "
                f"concepts' `order` fields are {actual_order}; they must agree "
                f"and be 1..{len(members)}"
            )
        for member in members:
            if member not in by_id:
                report.error(
                    f"chapter {chapter_id!r} lists {member!r}, which has no file in "
                    f"{CONCEPTS_DIR.name}/"
                )
                continue
            listed.append(member)
            concept = by_id[member]
            if concept.frontmatter.get("chapter") != chapter_id:
                report.error(
                    f"{member}: frontmatter says chapter "
                    f"{concept.frontmatter.get('chapter')!r} but index.json lists it "
                    f"under {chapter_id!r}"
                )

    orphans = sorted(set(by_id) - set(listed))
    if orphans:
        report.error(
            f"concepts not listed in any chapter of index.json: {', '.join(orphans)}. "
            f"Every concept must be reachable from the curriculum."
        )

    for concept in concepts:
        for key in ("prereqs", "related"):
            for target in concept.frontmatter.get(key, []) or []:
                if target not in by_id:
                    report.error(
                        f"{concept.id}: {key} references {target!r}, which does not exist"
                    )
                if key == "related" and target == concept.id:
                    report.error(f"{concept.id}: related includes itself")

    _check_cycles(concepts, by_id, report)


def _check_cycles(
    concepts: list[Concept], by_id: dict[str, Concept], report: Report
) -> None:
    """A prereq cycle means the guided order cannot be walked. Hard error."""
    state: dict[str, int] = {}  # 0 unvisited, 1 on stack, 2 done
    stack: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = stack[stack.index(node) :] + [node]
            report.error(
                "prereq cycle: " + " -> ".join(cycle)
                + ". A concept cannot be a prerequisite of itself, directly or "
                "through other concepts."
            )
            return
        state[node] = 1
        stack.append(node)
        for parent in by_id[node].frontmatter.get("prereqs", []) or []:
            if parent in by_id:
                visit(parent)
        stack.pop()
        state[node] = 2

    for concept in concepts:
        visit(concept.id)


def check_anchors(concepts: list[Concept], report: Report, root: Path = REPO_ROOT) -> None:
    for concept in concepts:
        anchors = concept.frontmatter.get("anchors") or {}
        if "python" not in anchors:
            report.error(
                f"{concept.id}: `anchors` must at least declare `python`. This is "
                f"a site about one specific file; a concept with no anchor to it "
                f"has no code to show."
            )
        for language, anchor in anchors.items():
            if language not in selectors.SOURCES:
                report.error(
                    f"{concept.id}: anchor names unknown language {language!r}; "
                    f"known: {', '.join(sorted(selectors.SOURCES))}"
                )
                continue
            path = root / selectors.SOURCES[language]
            if not path.is_file():
                report.error(
                    f"{concept.id}: anchors {language!r} but "
                    f"{selectors.SOURCES[language]} does not exist. Either the "
                    f"language is not implemented yet -- drop the anchor -- or the "
                    f"file is missing."
                )
            selector = str(anchor.get("selector", ""))
            try:
                parsed = selectors.parse_selector(selector)
            except selectors.SelectorError as exc:
                report.error(f"{concept.id} [{language}]: {exc}")
                continue
            if parsed.kind not in {"def", "assign", "comment", "section"}:
                report.error(f"{concept.id} [{language}]: unknown selector kind {parsed.kind!r}")


def check_bodies(concepts: list[Concept], index: dict[str, Any], report: Report) -> None:
    glossary = set((index.get("glossary", {}) or {}).get("terms", {}) or {})
    for concept in concepts:
        if not concept.body.strip():
            report.error(f"{concept.id}: body is empty")
        for block in concept.blocks:
            if block.kind != "directive":
                continue
            if block.name == "term":
                term = (block.text or block.quoted or "").strip()
                if not term:
                    report.error(f"{concept.id} line {block.line_no}: `:::term` needs a term")
                elif term not in glossary:
                    report.error(
                        f"{concept.id} line {block.line_no}: `:::term {term}` is not "
                        f"defined in index.json's glossary. Add it there or use a "
                        f"different word."
                    )
            if block.name == "trace":
                kind = block.args.get("kind")
                TRACE_KINDS = {
                    "loss-curve", "attention", "softmax", "shapes", "tokens", "samples",
                }
                if kind is None:
                    report.error(
                        f"{concept.id} line {block.line_no}: `:::trace` needs a kind "
                        f"(one of {', '.join(sorted(TRACE_KINDS))})"
                    )
                elif kind not in TRACE_KINDS:
                    # The app's `BlockView` switches on this set and renders
                    # nothing for anything else, so a typo here produces a concept
                    # with a silently missing widget.
                    report.error(
                        f"{concept.id} line {block.line_no}: unknown trace kind "
                        f"{kind!r}. Known: {', '.join(sorted(TRACE_KINDS))}. A kind "
                        f"outside this set renders nothing at all."
                    )
                for key in ("step", "pos", "head", "top"):
                    raw = block.args.get(key)
                    if raw is None:
                        continue
                    try:
                        value = int(raw)
                    except ValueError:
                        report.error(
                            f"{concept.id} line {block.line_no}: `:::trace {key}={raw}` "
                            f"is not an integer"
                        )
                        continue
                    if value < 0:
                        report.error(
                            f"{concept.id} line {block.line_no}: `:::trace {key}={raw}` "
                            f"is negative"
                        )
            if block.name == "callout" and not block.text.strip():
                report.warn(
                    f"{concept.id} line {block.line_no}: empty `:::callout` block"
                )


def check_reference_pin(index: dict[str, Any], report: Report, root: Path = REPO_ROOT) -> None:
    reference = index.get("reference", {})
    path = root / reference.get("path", "reference/microgpt.py")
    if not path.is_file():
        report.error(f"index.json reference path does not exist: {path}")
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = reference.get("sha256")
    if not expected:
        report.error(
            f"index.json reference.sha256 is missing. Every pinned file needs a "
            f"digest, or nothing notices when it moves."
        )
    elif digest != expected:
        report.error(
            f"index.json reference.sha256 does not match {path.name}.\n"
            f"      expected {expected}\n"
            f"      actual   {digest}\n"
            f"    reference/microgpt.py is byte-pinned. If the change is deliberate,\n"
            f"    update docs/PROVENANCE.md, regenerate traces/ and benchmarks/, and\n"
            f"    say so in the pull request -- every recorded number describes the\n"
            f"    old bytes."
        )
    # splitlines(), not split("\n"): the reference deliberately has no trailing
    # newline, so the two disagree by one on a file that is otherwise perfect.
    lines = len(path.read_text(encoding="utf-8").splitlines())
    if reference.get("lines") and lines != reference["lines"]:
        report.error(
            f"index.json reference.lines says {reference['lines']}, "
            f"{path.name} has {lines} lines"
        )


def check_anchors_current(concepts: list[Concept], report: Report, root: Path = REPO_ROOT) -> None:
    """The committed anchors.json must match what the selectors resolve to now.

    Without this, the order `gen_anchors` then `validate` matters: someone who
    runs only the validator after editing a concept gets a clean pass and ships
    a site whose code panels highlight the wrong lines. The visualizer reads a
    committed file, so staleness there is a silent wrong answer, which is the
    one thing this repository is built to avoid.
    """
    from tools import gen_anchors

    path = root / "visualizer" / "src" / "data" / "generated" / "anchors.json"
    if not path.is_file():
        report.error(
            f"{path.relative_to(root)} does not exist. The visualizer reads "
            f"it; run `make validate` (or `python3 -m tools.gen_anchors`)."
        )
        return

    result = gen_anchors.build(root)
    if result["failures"]:
        for failure in result["failures"]:
            report.error(f"anchor does not resolve: {failure}")
        return

    expected = json.dumps(result["document"], indent=2) + "\n"
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        report.error(
            f"{path.relative_to(root)} is stale. Re-run `make validate` to "
            f"regenerate it.\n"
            f"    The site renders code panels straight from this file, so a stale "
            f"copy highlights the wrong lines and says so with total confidence."
        )
    else:
        report.note(
            f"anchors current: {len(result['document']['concepts'])} concepts, "
            f"{len(result['document']['sources'])} sources"
        )


def check_traces(report: Report, repo_root: Path = REPO_ROOT) -> None:
    root = repo_root / "traces"
    if not root.is_dir():
        return
    total = 0
    for path in sorted(root.rglob("*.jsonl")):
        size = path.stat().st_size
        total += size
        if size > TRACE_BUDGET_BYTES:
            report.error(
                f"{path.relative_to(repo_root)} is {size} bytes, over the "
                f"{TRACE_BUDGET_BYTES} byte per-track budget. At micro "
                f"hyperparameters a trace is kilobytes; something is recording "
                f"raw activations."
            )
    if total:
        report.note(f"traces: {total / 1024:.1f} KiB total")



# -----------------------------------------------------------------------------


def main() -> int:
    report = Report()
    try:
        concepts = load_all(REPO_ROOT)
        index = load_index(REPO_ROOT)
        schema = load_schema(REPO_ROOT)
    except ContentError as exc:
        print(f"content error: {exc}", file=sys.stderr)
        return 1

    print(f"validating {len(concepts)} concepts against content/schema.json")
    check_schema(concepts, schema, report)
    check_graph(concepts, index, report)
    check_anchors(concepts, report, REPO_ROOT)
    check_bodies(concepts, index, report)
    check_reference_pin(index, report, REPO_ROOT)
    check_anchors_current(concepts, report, REPO_ROOT)
    check_traces(report, REPO_ROOT)

    for note in report.notes:
        print(f"  - {note}")
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    if report.errors:
        print(f"\n{len(report.errors)} problem(s):\n", file=sys.stderr)
        for error in report.errors:
            print(f"  ! {error}", file=sys.stderr)
        return 1

    chapters = index.get("chapters", [])
    print(
        f"content ok: {len(concepts)} concepts, {len(chapters)} chapters, "
        f"{len(index.get('glossary', {}).get('terms', {}))} glossary terms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
