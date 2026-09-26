#!/usr/bin/env python3
"""A deliberately tiny YAML subset, so `tools/` can stay stdlib-only.

The repository's ethos is that you should be able to audit and run the tooling
without installing anything. That rules out PyYAML, but the concept frontmatter
still wants to be pleasant to read. So: a strict subset, ~200 lines, with a
test suite, that supports exactly what `content/concepts/*.md` uses.

## Supported

A mapping, optionally containing nested mappings and block sequences, up to
`MAX_DEPTH` levels deep. Scalars, flow sequences, flow maps, literal block
scalars, and comments are supported at any depth.

```yaml
id: attn-heads                       # plain scalar
title: Multi-head attention          # plain scalar
summary: "Q, K, V are sliced"        # quoted scalar
order: 7                             # int
tracing: true                        # bool
related: [softmax, rmsnorm]          # flow sequence
shapes:                              # block sequence of flow maps
  - { name: q_h, shape: "[T, 4]" }
anchors:                             # block mapping, one level down
  python: { selector: "def:gpt" }
  c: { selector: "def:gpt_forward" }
math: |                              # literal block scalar
  \\text{attn}(Q,K,V) = ...
```

## Deliberately unsupported

Anchors, aliases, tags, multi-document streams, and mappings nested more than
`MAX_DEPTH` levels. `parse()` raises `YamliteError` with a line number rather
than guessing. A content author who reaches for the unsupported half of YAML
gets a precise error instead of a silently misread manifest, which is the
failure mode that actually rots a teaching repository.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["YamliteError", "MAX_DEPTH", "parse", "parse_scalar", "dump_frontmatter"]

#: How deep a block mapping may nest. Two is what `anchors: {python: {selector}}`
#: needs and one more than most hand-written content will ever want.
MAX_DEPTH = 3


class YamliteError(ValueError):
    """Raised with a line number when the input leaves the supported subset."""

    def __init__(self, message: str, line_no: int | None = None) -> None:
        if line_no is not None:
            message = f"line {line_no}: {message}"
        super().__init__(message)
        self.line_no = line_no


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def _strip_comment(value: str) -> str:
    """Remove a trailing ``# comment`` that is not inside quotes."""
    out: list[str] = []
    quote: str | None = None
    prev = ""
    for ch in value:
        if quote:
            out.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (not out or out[-1] in " \t"):
            break
        else:
            out.append(ch)
        prev = ch
    return "".join(out).strip()


def _unquote(value: str, line_no: int) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        body = value[1:-1]
        if value[0] == '"':
            # Only the escapes we actually might write in prose.
            return (
                body.replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return body.replace("''", "'")
    return value


def _split_flow(body: str, line_no: int) -> list[str]:
    """Split a flow collection body on top-level commas."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    prev = ""
    for ch in body:
        if quote:
            current.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
            current.append(ch)
        elif ch in "[{":
            depth += 1
            current.append(ch)
        elif ch in "]}":
            depth -= 1
            if depth < 0:
                raise YamliteError("unbalanced flow collection", line_no)
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        prev = ch
    if quote:
        raise YamliteError("unterminated quoted string", line_no)
    if depth:
        raise YamliteError("unbalanced flow collection", line_no)
    tail = "".join(current)
    if tail.strip() or parts:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip() != ""]


def parse_scalar(token: str, line_no: int | None = None) -> Any:
    """Parse one flow-context scalar or collection."""
    token = token.strip()
    if not token:
        return ""
    if token[0] == "[" and token[-1] == "]":
        return [parse_scalar(p, line_no) for p in _split_flow(token[1:-1], line_no or 0)]
    if token[0] == "{" and token[-1] == "}":
        mapping: dict[str, Any] = {}
        for part in _split_flow(token[1:-1], line_no or 0):
            if ":" not in part:
                raise YamliteError(f"flow map entry without ':' -> {part!r}", line_no)
            key, _, value = part.partition(":")
            mapping[parse_scalar(key, line_no)] = parse_scalar(value, line_no)
        return mapping
    if token[0] == "&" or token[0] == "*" or token[0] == "!":
        raise YamliteError("anchors, aliases and tags are not supported", line_no)
    unquoted = _unquote(token, line_no or 0)
    if unquoted is not token:
        return unquoted
    lowered = token.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "~"}:
        return None
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token) and not _INT_RE.match(token):
        return float(token)
    return token


def _read_block_scalar(
    lines: list[str], start: int, indent: int, stop: int
) -> tuple[str, int]:
    """Collect an indented literal block (``|``). Returns (text, next_index)."""
    body: list[str] = []
    index = start
    base_indent: int | None = None
    while index < stop:
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            body.append("")
            index += 1
            continue
        line_indent = _indent_of(raw)
        if line_indent <= indent:
            break
        if base_indent is None:
            base_indent = line_indent
        body.append(raw[base_indent:].rstrip() if len(raw) >= base_indent else "")
        index += 1
    while body and not body[-1]:
        body.pop()
    return "\n".join(body), index


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def parse(text: str) -> dict[str, Any]:
    """Parse the supported YAML subset into a dict."""
    lines = text.split("\n")
    value, _ = _parse_mapping(lines, 0, len(lines), 0)
    return value


def _parse_mapping(
    lines: list[str], start: int, stop: int, depth: int, indent: int = 0
) -> tuple[dict[str, Any], int]:
    """Parse an aligned run of `key: value` lines into a dict.

    Iteration stops at the first non-blank line indented less than `indent`,
    which is what returns control to the parent block. Without that, a nested
    mapping swallows the rest of the document -- including the body of a
    neighbouring `key: |` block scalar, and then reporting a confusing error
    about indentation.
    """
    if depth > MAX_DEPTH:
        raise YamliteError(
            f"mappings nested more than {MAX_DEPTH} levels deep; flatten this", start + 1
        )
    result: dict[str, Any] = {}
    index = start
    while index < stop:
        raw = lines[index]
        line_no = index + 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        line_indent = _indent_of(raw)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamliteError("unexpected indentation", line_no)
        if raw.rstrip().endswith("---") and not raw.strip():
            index += 1
            continue
        if ":" not in raw:
            raise YamliteError(f"expected 'key: value', got {raw.strip()!r}", line_no)

        key, _, rest = raw.partition(":")
        key = key.strip()
        rest = _strip_comment(rest)
        if not key:
            raise YamliteError("empty key", line_no)

        if rest in {"|", "|-", "|+"}:
            body, index = _read_block_scalar(lines, index + 1, _indent_of(raw), stop)
            result[key] = body
            continue
        if rest:
            result[key] = parse_scalar(rest, line_no)
            index += 1
            continue

        # No inline value: a block sequence or a nested mapping follows, both
        # indented. Blank lines and comments between here and the block are
        # skipped, so `anchors:` may be followed by a comment.
        lookahead = index + 1
        while lookahead < stop and (
            not lines[lookahead].strip() or lines[lookahead].lstrip().startswith("#")
        ):
            lookahead += 1
        if lookahead >= stop or lines[lookahead][:1] not in (" ", "\t"):
            result[key] = None
            index = lookahead
            continue

        child_indent = _indent_of(lines[lookahead])
        if lines[lookahead].lstrip().startswith("- "):
            items, index = _read_block_sequence(lines, lookahead, stop, child_indent, depth)
            result[key] = items
        else:
            child, index = _parse_mapping(lines, lookahead, stop, depth + 1, child_indent)
            result[key] = child
    return result, index


def _read_block_sequence(
    lines: list[str], index: int, stop: int, indent: int, depth: int
) -> tuple[list[Any], int]:
    """Collect an aligned block sequence. Returns (items, next_index)."""
    if depth > MAX_DEPTH:
        raise YamliteError(
            f"sequences nested more than {MAX_DEPTH} levels deep; flatten this", index + 1
        )
    items: list[Any] = []
    while index < stop:
        candidate = lines[index]
        if not candidate.strip():
            index += 1
            continue
        if _indent_of(candidate) != indent or not candidate.lstrip().startswith("-"):
            break
        body = _strip_comment(candidate.lstrip()[1:].strip())
        if not body:
            raise YamliteError("empty sequence item", index + 1)
        if body in {"|", "|-", "|+"}:
            text_block, index = _read_block_scalar(lines, index + 1, indent, stop)
            items.append(text_block)
            continue
        if body[0] in "[{":
            # A flow collection may wrap across several aligned lines.
            buf = [body]
            while not _is_balanced("".join(buf)) and index + 1 < stop:
                index += 1
                buf.append(_strip_comment(lines[index].strip()))
            items.append(parse_scalar(" ".join(buf), index + 1))
        elif body[0] in "[{" or ":" in body and not body.startswith('"'):
            # A flow map, or a compact nested mapping on the dash line.
            if body.endswith("}") or "}" in body:
                items.append(parse_scalar(body, index + 1))
            else:
                child_lines = [" " * (indent + 2) + body] + lines[index + 1 : stop]
                child, consumed = _parse_mapping(child_lines, 0, len(child_lines), depth + 1)
                items.append(child)
                index += consumed - 1
        else:
            items.append(parse_scalar(body, index + 1))
        index += 1
    return items, index


def _is_balanced(text: str) -> bool:
    depth = 0
    quote: str | None = None
    prev = ""
    for ch in text:
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        prev = ch
    return depth == 0 and quote is None


_PLAIN_SAFE = re.compile(r"^[A-Za-z0-9_./+*-]+$")


def dump_frontmatter(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Render a dict back to the supported subset. Round-trips our own output."""
    pad = " " * indent
    out: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            out.append(f"{pad}{key}:")
            out.extend(dump_frontmatter(value, indent + 2))
        elif isinstance(value, str) and ("\n" in value or value != value.strip()):
            out.append(f"{pad}{key}: |")
            out.extend(f"{pad}  {line}" if line else "" for line in value.split("\n"))
        elif isinstance(value, (list, tuple)):
            rendered = [_dump_inline(v) for v in value]
            if all("\n" not in r and len(r) < 60 for r in rendered):
                out.append(f"{pad}{key}: [{', '.join(rendered)}]")
            else:
                out.append(f"{pad}{key}:")
                for item in value:
                    out.append(f"{pad}  - {_dump_inline(item)}")
        else:
            out.append(f"{pad}{key}: {_dump_inline(value)}")
    return out


def _dump_inline(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_dump_inline(v) for v in value) + "]"
    if isinstance(value, dict):
        body = ", ".join(f"{k}: {_dump_inline(v)}" for k, v in value.items())
        return "{" + body + "}"
    text = str(value)
    if "\n" in text:
        return _quote(text)
    if _PLAIN_SAFE.match(text) and text.lower() not in {"true", "false", "null", "yes", "no"}:
        return text
    return _quote(text)


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'
