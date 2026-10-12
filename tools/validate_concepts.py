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
| `:::trace` asks for a recorded row | a widget stuck on its empty state |
| reference sha256 pinned | every recorded number silently becoming a lie |
| trace size budget | a repo that is 400 MB because traces grew |

Exit code 0 means the Atlas can be trusted to say what it says.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Run as `python3 -m tools.validate_concepts`; see tools/_entrypoint.py. This
# must precede `from tools import ...`: a script run leaves `tools/` on sys.path
# but not the repository root, so the package is not importable at all.
_spec = importlib.util.spec_from_file_location(
    "_tools_entrypoint", Path(__file__).resolve().parent / "_entrypoint.py"
)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
_guard.require_module_invocation("validate_concepts", __package__)

import hashlib
import json
import math
import re
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import selectors  # noqa: E402
from tools import trace  # noqa: E402
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


class NonFiniteNumber(Exception):
    """A `number` field held `NaN`/`inf`. Carries the value.

    A private signal, not part of the project's error hierarchy: `_type_ok` is a
    pure predicate with no field path to name, so it raises this and the caller
    -- which knows the path -- re-raises the public `ContentError`.
    """

    def __init__(self, value: float) -> None:
        super().__init__(f"number {value!r} is not finite")
        self.value = value


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
        if isinstance(value, bool):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            # `NaN`/`inf` satisfy `isinstance(x, float)` and compare False against
            # every bound, so they slip through `minimum`/`maximum` untouched.
            # A trace with a corrupted value used to render as "NaN" forever.
            # Rejected here too, so a field with no bounds is still checked; the
            # caller upgrades this to a path-naming message before it escapes.
            raise NonFiniteNumber(value)
        return isinstance(value, (int, float))
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
        if not isinstance(node, dict) or part not in node:
            raise ContentError(
                f"schema $ref {ref!r} does not resolve: no {part!r} in "
                f"{type(node).__name__}; the reference is dangling"
            )
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
    if expected:
        try:
            if not _type_ok(value, expected):
                report.error(
                    f"{path}: expected {expected}, got {type(value).__name__}"
                )
                return
        except NonFiniteNumber as exc:
            # `minimum`/`maximum` cannot reject inf or NaN -- every comparison
            # against NaN is False -- so the type check is the only place to
            # catch it. Named with the field, because "a number somewhere was
            # NaN" is not something anyone can act on.
            raise ContentError(
                f"{path}: number {exc.value!r} is not finite; NaN and +/-inf "
                f"bypass minimum/maximum and can carry a corrupted value into "
                f"the visualizer"
            ) from exc

    if "enum" in schema and value not in schema["enum"]:
        report.error(f"{path}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            report.error(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            report.error(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema:
            if not re.search(schema["pattern"], value):
                report.error(f"{path}: {value!r} does not match /{schema['pattern']}/")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            # `minimum`/`maximum` cannot reject inf or NaN (every comparison
            # against NaN is False), so the only place to catch this is here.
            raise ContentError(
                f"{path}: number {value!r} is not finite; NaN and +/-inf bypass "
                f"minimum/maximum and can carry a corrupted value into the visualizer"
            )
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

    if not isinstance(index, dict):
        raise ContentError(
            f"content/index.json: the document must be an object, got "
            f"{type(index).__name__}. Every `chapters`/`glossary`/`reference` "
            f"lookup here assumes a mapping."
        )

    # `glossary` is checked here, next to `chapters`, because the two have the
    # same failure mode and the same fix: every consumer reaches into the
    # mapping with `.get` and no shape check, so a wrong shape is not rejected,
    # it is *dereferenced*. `check_bodies` builds its term set with
    # `set(index.get("glossary", {}).get("terms", {}))` and `build_index_payload`
    # emits `index.get("glossary", {}).get("terms", {})` verbatim into the
    # visualizer's payload.
    #
    # The inner `terms` shape is the one that actually bites. A list there is
    # silently *wrong* rather than fatal on both sides: `set(["a","b"])` is a
    # perfectly good set that happens to contain the wrong members, so `:::term`
    # validation compares against nonsense and passes, and the payload ships a
    # list where the visualizer expects a mapping. A bare string is worse still --
    # `set("abc")` is `{"a","b","c"}`, so the check starts validating terms
    # against individual characters. Both are the silent-wrong-answer class this
    # file exists to catch, which is why they are errors and not warnings.
    glossary = index.get("glossary", {})
    if glossary is None:
        glossary = {}
    if not isinstance(glossary, dict):
        raise ContentError(
            f"content/index.json: 'glossary' must be an object, got "
            f"{type(glossary).__name__}"
        )
    terms = glossary.get("terms", {})
    if terms is None:
        terms = {}
    if not isinstance(terms, dict):
        raise ContentError(
            f"content/index.json: 'glossary.terms' must be an object mapping each "
            f"term to its definition, got {type(terms).__name__}. Every `:::term` "
            f"check and the visualizer's glossary payload read it as a mapping."
        )

    chapters = index.get("chapters", [])
    if not chapters:
        report.error("index.json declares no chapters")
        return

    if not isinstance(chapters, list):
        raise ContentError(
            f"content/index.json: 'chapters' must be a list, got "
            f"{type(chapters).__name__}"
        )
    for position, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            raise ContentError(
                f"content/index.json: chapters[{position}] must be an object, got "
                f"{type(chapter).__name__}"
            )
        if not isinstance(chapter.get("concepts", []), list):
            raise ContentError(
                f"content/index.json: 'concepts' in chapter "
                f"{chapter.get('id', position)!r} must be a list, got "
                f"{type(chapter['concepts']).__name__}"
            )

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


#: The argument defaults the app itself applies before looking a row up, read
#: off the switch in `BlockView` (`visualizer/src/components/ConceptPage.tsx`)
#: and the components' own default parameters
#: (`visualizer/src/components/widgets.tsx`):
#:
#:   attention  -> <AttentionHeatmap step={num('step', 0)} pos={num('pos', 3)} head={num('head', 0)} />
#:   softmax    -> <SoftmaxBars step={num('step', 0)} pos={num('pos', 0)} />
#:   samples    -> <SamplesList /> with `step = args.step ?? trace.samples[0].step`
#:
#: A directive that writes none of these still asks for a specific row, so the
#: check has to run against the *effective* request -- otherwise
#: `:::trace kind=samples` would slip through even though it resolves to a real
#: step. Keeping this table equal to the app's defaults is what makes the check
#: describe the rendered page.
#:
#: `samples` has no literal default: the widget uses the first recorded sample's
#: step, so the validator reads it from the trace rather than hardcoding 1000.
#: That keeps the two in step if the trace is ever recorded with another step.
TRACE_DEFAULTS: dict[str, dict[str, int]] = {
    "attention": {"step": 0, "pos": 3, "head": 0},
    "softmax": {"step": 0, "pos": 0},
}


def _trace_defaults(kind: str, index: trace.TraceIndex) -> dict[str, int]:
    """The app's defaults for one kind, with `samples` read from the trace."""
    if kind == "samples":
        steps = index.recorded("samples", "step")
        return {"step": steps[0] if steps else 1000}
    return TRACE_DEFAULTS.get(kind, {})


def _trace_index() -> trace.TraceIndex:
    """The committed trace's index, loaded once per process.

    Cached because `check_bodies` calls this for every directive and the index
    is a full read of ~1100 JSONL rows; seven directives should not mean seven
    passes over the same files.
    """
    global _TRACE_INDEX
    if _TRACE_INDEX is None:
        _TRACE_INDEX = trace.load_trace_index()
    return _TRACE_INDEX


_TRACE_INDEX: trace.TraceIndex | None = None


def check_trace_directive(
    block: Any, concept_id: str, values: dict[str, int], report: Report
) -> None:
    """Fail when a `:::trace` asks for data the committed trace does not hold.

    `kind` being known and the integers parsing is not enough: the app looks the
    request up in a JSONL file, and a miss makes the widget render "No attention
    weights were recorded for step 500, position 15, head 0" inside a concept
    that promised a picture. Nothing else in the pipeline notices, so this is
    where it has to be noticed.
    """
    kind = block.args.get("kind")
    fields = trace.KIND_TRACE_FILES.get(str(kind), (None, ()))[1]
    if not fields:
        return  # not data-backed: `tokens` reads no file, `shapes` renders nothing

    try:
        index = _trace_index()
    except (OSError, ValueError) as exc:
        # A missing or malformed trace is a real problem, but not one this
        # check should disguise as a content error. Say what happened.
        report.error(
            f"{concept_id} line {block.line_no}: `:::trace kind={kind}` cannot be "
            f"checked against {trace.TRACES_DIR.name}/python/micro because the trace "
            f"could not be read ({exc}). Regenerate it with `make trace`."
        )
        return

    missing = index.missing_files().get(str(kind))
    if missing is not None and not index.entries.get(str(kind)):
        # The file this kind is backed by was never read, so "no such row" would
        # be a lie -- the rows are unknown, not absent.
        report.error(
            f"{concept_id} line {block.line_no}: `:::trace kind={kind}` cannot be "
            f"checked because {missing}. Regenerate the trace with `make trace`."
        )
        return

    request = {
        field: values.get(field, _trace_defaults(str(kind), index).get(field, 0))
        for field in fields
    }

    if index.has(str(kind), request):
        return

    asked = ", ".join(f"{field}={request[field]}" for field in fields)
    defaults = _trace_defaults(str(kind), index)
    alternatives = []
    for field in fields:
        recorded = index.recorded(str(kind), field)
        if field in defaults and request[field] == defaults[field]:
            continue  # the app's default; naming it as "what exists" is noise
        if recorded:
            alternatives.append(f"{field} {_summarise(recorded)}")
    detail = "; ".join(alternatives) if alternatives else "the trace contains no rows at all"
    report.error(
        f"{concept_id} line {block.line_no}: `:::trace kind={kind}` asks for "
        f"{asked}, which {index.files[str(kind)]} does not contain. The widget "
        f"would render its empty state instead of a picture. Recorded: {detail}."
    )


def _summarise(values: list[int], limit: int = 12) -> str:
    """A readable `0, 1, 3, 7, 15` -- truncated rather than dumped."""
    if len(values) <= limit:
        return ", ".join(str(value) for value in values)
    # The loss curve is every step; write the endpoints rather than 1000 numbers.
    return ", ".join(str(value) for value in values[:limit]) + ", ..."


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
                values: dict[str, int] = {}
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
                        continue
                    values[key] = value
                # A directive whose kind and integers are all fine can still ask
                # for a data point the committed trace never recorded -- at which
                # point the widget renders its empty state and the reader is told
                # nothing. That is the one failure of this gate that used to be
                # completely silent, so the request is resolved through the app's
                # own defaults and looked up in the rows.
                check_trace_directive(block, concept.id, values, report)
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


#: A markdown link whose href is `cite:<start>` or `cite:<start>-<end>`.
#:
#: This grammar is deliberately tiny and is mirrored, character for character, in
#: `visualizer/src/content/cite.ts`. The two runtimes cannot share code -- one is
#: Python at build time, one is TypeScript in the browser -- so the grammar has to
#: be small enough to keep in step by eye. The test suites on both sides use the
#: same examples for the same reason: a form one accepts and the other rejects is
#: a citation that either fails the build for no reason or renders as a dead
#: control, and both are worse than not having the feature.
CITE_RE = re.compile(r"\[[^\]]*\]\(cite:([^)\s]+)\)")


def _parse_cite(raw: str) -> tuple[int, int] | None:
    """`11` or `11-12` -> `(start, end)`; anything else -> `None`.

    Mirrors `parseCite` in `visualizer/src/content/cite.ts`, including the
    rejections: 1-based, inclusive, `start <= end`. An inverted range is refused
    rather than swapped, because the author's number and the highlighted number
    have to be the same number -- that is the whole point of the feature.
    """
    single = re.fullmatch(r"(\d+)", raw)
    if single:
        line = int(single.group(1))
        return (line, line) if line >= 1 else None
    span = re.fullmatch(r"(\d+)-(\d+)", raw)
    if span:
        start, end = int(span.group(1)), int(span.group(2))
        if start < 1 or end < start:
            return None
        return (start, end)
    return None


def check_citations(concepts: list[Concept], report: Report, root: Path = REPO_ROOT) -> None:
    """Every prose citation must resolve against the concept's primary anchor.

    A citation is a line range written into a sentence: "the seed on line 11".
    The reader clicks it and the code panel scrolls there. That only means
    anything if the number is *right*, and "right" here has a precise meaning --
    inside the window the panel crops to, in the language whose panel the prose
    is attached to.

    A concept can anchor several languages, but the prose sits beside one panel:
    the primary language (`selectors.primary_language`, mirroring `TRACK_ORDER`
    in the visualiser). A citation resolves against *that* window and no other.
    Checking every declared language instead would be the seductive mistake -- it
    passes a C-window citation on a page whose visible panel is Python, and the
    reader gets a click that goes nowhere and a sentence citing a line they
    cannot see. That is exactly the dead control this gate exists to prevent, so
    the gate and the browser have to ask the same question, of the same language.

    This is the one failure the browser cannot catch on its own, which is why it
    is a build gate: by the time the page renders, the prose and the anchor have
    parted ways, and the reader has no way to tell that the number is stale.

    Read from the committed `anchors.json` rather than re-resolving the selectors,
    because that file is what the browser will actually use -- `check_anchors_current`
    has already failed the run if the two disagree, so this check and the rendered
    page are looking at the same numbers by the time it runs.
    """
    path = root / "visualizer" / "src" / "data" / "generated" / "anchors.json"
    if not path.is_file():
        # `check_anchors_current` reports this; do not report it twice.
        return
    resolved = json.loads(path.read_text(encoding="utf-8")).get("concepts", {})

    for concept in concepts:
        declared = (concept.frontmatter.get("anchors") or {}).keys()
        resolved_spans = resolved.get(concept.id) or {}
        # The panel the prose sits next to: the first declared language in track
        # order. A citation is only clickable in *this* panel, so this is the one
        # window a range has to fall inside.
        primary = selectors.primary_language(declared)
        window = None
        if primary is not None and primary in resolved_spans:
            span = resolved_spans[primary]
            window = (span["start"], span["end"])

        for block in concept.blocks:
            if not block.text:
                continue
            for match in CITE_RE.finditer(block.text):
                parsed = _parse_cite(match.group(1))
                if parsed is None:
                    report.error(
                        f"{concept.id} line {block.line_no}: `cite:{match.group(1)}` "
                        f"is not a line range. Write `cite:11` or `cite:11-12` "
                        f"(1-based, inclusive). See `visualizer/src/content/cite.ts`."
                    )
                    continue
                start, end = parsed
                if window is None:
                    report.error(
                        f"{concept.id} line {block.line_no}: `cite:{match.group(1)}` "
                        f"has no anchor to resolve against -- this concept declares "
                        f"no primary-language anchor that resolved. A citation needs "
                        f"a code panel to point at."
                    )
                    continue
                low, high = window
                if start < low or end > high:
                    report.error(
                        f"{concept.id} line {block.line_no}: `cite:{match.group(1)}` "
                        f"(lines {start}\u2013{end}) is outside this concept's "
                        f"{primary} anchored range ({low}\u2013{high}). The code panel "
                        f"crops to the primary language's anchor, so this citation "
                        f"would scroll to a line the reader cannot see. Either point "
                        f"it at a line inside the range, or widen the anchor."
                    )
                else:
                    report.note(
                        f"{concept.id}: citation cite:{match.group(1)} resolves"
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
    try:
        check_schema(concepts, schema, report)
        check_graph(concepts, index, report)
        check_anchors(concepts, report, REPO_ROOT)
        check_bodies(concepts, index, report)
        check_reference_pin(index, report, REPO_ROOT)
        check_anchors_current(concepts, report, REPO_ROOT)
        check_citations(concepts, report, REPO_ROOT)
        check_traces(report, REPO_ROOT)
    except ContentError as exc:
        print(f"content error: {exc}", file=sys.stderr)
        return 1

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
