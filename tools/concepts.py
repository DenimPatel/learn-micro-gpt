#!/usr/bin/env python3
"""Loading and validating concept files.

One module owns "what a concept is", so `gen_anchors.py`,
`validate_concepts.py`, `render_content.py`, and the tests all agree. It also
owns the frontmatter split and the `:::` directive body syntax, because those
two things are the format the content authors actually write.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tools import yamlite

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO_ROOT / "content"
CONCEPTS_DIR = CONTENT_DIR / "concepts"
INDEX_PATH = CONTENT_DIR / "index.json"
SCHEMA_PATH = CONTENT_DIR / "schema.json"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)

#: `:::trace step=0 pos=3`  /  `:::callout "Why divide by sqrt(d)?"`
DIRECTIVE_RE = re.compile(
    r"^:::(?P<name>[a-z][a-z-]*)(?P<args>.*)$",
)

#: A line that starts with `:::` but does not match `DIRECTIVE_RE`. `::::trace`
#: arrived from a stray keystroke in an edit, and because it simply failed the
#: regex it was rendered as a paragraph of literal colons -- silently, with no
#: build failure. A near-miss on a directive is a typo, and typos should be loud.
_MALFORMED_DIRECTIVE_RE = re.compile(r"^:{2,}\S*\S")

#: Known directives, and the argument names each one accepts.
#:
#: `block` means "owns the lines below it until a closing `:::`" -- i.e. the
#: directive has a body. Getting this wrong is not a cosmetic error: `trace` was
#: originally marked as a block, and a self-closing `:::trace kind=attn` therefore
#: swallowed every following line up to the next `:::`, which silently deleted
#: most of the multi-head-attention concept. A directive with no body is not a
#: block.
DIRECTIVE_SPEC: dict[str, dict[str, Any]] = {
    "trace": {
        "block": False,
        "args": {"kind", "step", "pos", "head", "top", "lang", "config"},
        "note": "Inline a widget backed by a committed trace. One line. See docs/TRACE-FORMAT.md.",
    },
    "lang": {
        "block": True,
        "args": {"config"},
        "note": "Start a language tab range; closed by the next :::lang or the end of the body.",
    },
    "shape": {
        "block": False,
        "args": {"name"},
        "note": "Inline one shape table entry, by name.",
    },
    "term": {
        "block": False,
        "args": {},
        "note": "Link a glossary term. Validation fails if the term is not defined.",
    },
    "callout": {
        "block": True,
        "args": {},
        "note": "A boxed aside, optionally titled. Body until a closing :::.",
    },
    "note": {
        "block": True,
        "args": {},
        "note": "A quieter boxed aside.",
    },
}


class ContentError(Exception):
    """A concept file is malformed. Carries the file so CI can name it."""


@dataclass
class Block:
    """One piece of a concept body: prose, or a directive."""

    kind: str  # "prose" | "directive"
    text: str = ""
    name: str = ""
    args: dict[str, str] = field(default_factory=dict)
    quoted: str = ""  # a first double-quoted argument, e.g. a callout title
    line_no: int = 0


@dataclass
class Concept:
    id: str
    path: Path
    frontmatter: dict[str, Any]
    body: str
    blocks: list[Block]

    @property
    def title(self) -> str:
        return str(self.frontmatter.get("title", self.id))

    def directives(self) -> Iterator[Block]:
        for block in self.blocks:
            if block.kind == "directive":
                yield block


def split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ContentError(f"{path.name}: no YAML frontmatter (expected a leading `---` block)")
    try:
        front = yamlite.parse(match.group(1))
    except yamlite.YamliteError as exc:
        raise ContentError(f"{path.name}: frontmatter: {exc}") from exc
    return front, match.group(2)


def parse_args(raw: str, line_no: int) -> tuple[dict[str, str], str]:
    """Parse a directive's tail: `step=0 pos=3 head=0` plus an optional title.

    A leading double-quoted string is a title, which is the only quoted argument
    any directive takes, so this does not need to be a real expression parser.
    """
    quoted = ""
    text = raw.strip()
    if text.startswith('"'):
        close = text.find('"', 1)
        if close == -1:
            raise ContentError(f"unterminated quoted string in directive: {raw!r}")
        quoted = text[1:close]
        text = text[close + 1 :].strip()

    args: dict[str, str] = {}
    for position, token in enumerate(text.split()):
        if not token:
            continue
        if "=" not in token:
            # One leading bare token is the directive's subject -- the term being
            # linked, the shape being shown. `:::term rmsnorm` reads better than
            # `:::term name=rmsnorm`, and a second bare token is a typo rather
            # than a shorthand worth supporting.
            if position == 0 and not quoted:
                quoted = token.strip('"')
                continue
            raise ContentError(
                f"directive argument {token!r} is not key=value (line {line_no}). "
                f"A directive may have one bare leading argument and then only key=value."
            )
        key, _, value = token.partition("=")
        args[key] = value.strip('"')
    return args, quoted


def parse_body(body: str) -> list[Block]:
    """Split a concept body into prose and directives.

    A block directive (`:::callout`) owns the lines up to the closing `:::`. An
    inline directive (`:::term bos`) is exactly one line. A bare `:::` closes the
    most recent block, or does nothing at all if none is open -- which is how
    `:::lang` ranges are terminated.
    """
    blocks: list[Block] = []
    lines = body.split("\n")
    open_block: Block | None = None
    buffer: list[str] = []
    prose_start = 1

    def flush_prose(end_line: int) -> None:
        chunk = "\n".join(buffer).strip("\n")
        if chunk.strip():
            blocks.append(Block(kind="prose", text=chunk, line_no=prose_start))
        buffer.clear()

    for index, raw in enumerate(lines):
        line_no = index + 2  # +1 for zero-based, +1 for the frontmatter block
        stripped = raw.strip()

        if open_block is not None:
            if stripped == ":::":
                open_block.text = "\n".join(buffer).strip("\n")
                blocks.append(open_block)
                open_block = None
                buffer.clear()
            else:
                buffer.append(raw)
            continue

        match = DIRECTIVE_RE.match(stripped)
        if not match:
            if _MALFORMED_DIRECTIVE_RE.match(stripped):
                raise ContentError(
                    f"line {line_no}: {stripped!r} starts with ':::' but is not a valid "
                    f"directive. Expected `:::name` or `:::name key=value`. A near-miss "
                    f"here renders as a paragraph of colons rather than failing."
                )
            buffer.append(raw)
            continue

        name = match.group("name")
        if name not in DIRECTIVE_SPEC:
            raise ContentError(
                f"line {line_no}: unknown directive `:::{name}`. "
                f"Known: {', '.join(sorted(DIRECTIVE_SPEC))}"
            )
        flush_prose(line_no)
        spec = DIRECTIVE_SPEC[name]
        args, quoted = parse_args(match.group("args"), line_no)
        unknown = set(args) - set(spec["args"])
        if unknown:
            raise ContentError(
                f"line {line_no}: `:::{name}` does not accept "
                f"{', '.join(sorted(unknown))}. Accepted: {', '.join(sorted(spec['args'])) or '(none)'}"
            )
        block = Block(kind="directive", name=name, args=args, quoted=quoted, line_no=line_no)
        if spec["block"]:
            open_block = block
            buffer.clear()
        else:
            blocks.append(block)
        prose_start = line_no + 1

    if open_block is not None:
        raise ContentError(
            f"line {open_block.line_no}: `:::{open_block.name}` block is never closed with `:::`"
        )
    flush_prose(len(lines) + 1)
    return blocks


def load_concept(path: Path) -> Concept:
    text = path.read_text(encoding="utf-8")
    front, body = split_frontmatter(text, path)
    if "id" not in front:
        raise ContentError(f"{path.name}: frontmatter has no `id`")
    concept_id = str(front["id"])
    if concept_id != path.stem:
        raise ContentError(
            f"{path.name}: frontmatter id {concept_id!r} does not match the filename. "
            f"Rename one of them; the filename is what index.json and the routes use."
        )
    return Concept(
        id=concept_id,
        path=path,
        frontmatter=front,
        body=body,
        blocks=parse_body(body),
    )


def load_all(root: Path = REPO_ROOT) -> list[Concept]:
    directory = root / "content" / "concepts"
    if not directory.is_dir():
        raise ContentError(f"{directory} does not exist")
    concepts = [load_concept(path) for path in sorted(directory.glob("*.md"))]
    if not concepts:
        raise ContentError(f"no concept files in {directory}")
    return concepts


def load_index(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "content" / "index.json"
    if not path.is_file():
        raise ContentError(f"{path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContentError(f"{path}: {exc}") from exc


def load_schema(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "content" / "schema.json"
    return json.loads(path.read_text(encoding="utf-8"))
