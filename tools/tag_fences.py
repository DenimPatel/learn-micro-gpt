#!/usr/bin/env python3
"""One-off: tag code fences that are not verbatim excerpts of their own anchor.

An untagged fence claims to be a verbatim excerpt of the concept's own anchor
range -- that is the convention `tools/tests/test_content.py` enforces, and it
is what makes "is this snippet the thing I'm highlighting?" answerable by
looking at the fence. Fences that show something else (illustrative arithmetic,
a diagram, program output, or an excerpt from a deliberately different part of
the file) must say so by declaring an info string.

Idempotent: fences that already have a tag are left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import concepts as content  # noqa: E402
from tools import selectors  # noqa: E402

SOURCE = (REPO_ROOT / "reference" / "microgpt.py").read_text(encoding="utf-8").split("\n")

#: Lines that look like prose or diagrams rather than source.
_KEYWORDS = (
    "def ", "class ", "for ", "if ", "return", "x = ", "probs", "logits", "attn_",
    "loss", "num_", "m[", "v[", "p.", "doc =", "token", "sample.", "learning",
    "temperature", "n =", "keys,", "uchars", "BOS", "vocab_size", "docs =",
    "print(", "self.", "import ", "random.", "print", "matrix", "state_dict",
    "params", "head_dim", "n_embd", "n_head", "n_layer", "block_size",
)


def normalise(line: str) -> str:
    return line.split("#")[0].strip().rstrip(",").strip()


def classify(lines: list[str], anchored: set[str]) -> str | None:
    """Return the info string this fence needs, or None if it is already fine."""
    keys = [normalise(line) for line in lines]
    keys = [key for key in keys if key and not set(key) <= {"."}]
    if not keys:
        return None
    if all(key in anchored for key in keys):
        return None  # verbatim excerpt of the anchor; leave untagged
    joined = "\n".join(lines)
    if "sample" in joined and "---" in joined:
        return "output"
    if any(key.startswith(_KEYWORDS) for key in keys):
        return "python"  # real code, but from a different part of the file
    return "text"  # arithmetic, a diagram, or a table


def tag(path: Path) -> int:
    concept = content.load_concept(path)
    _, body = content.split_frontmatter(path.read_text(encoding="utf-8"), path)
    span = selectors.resolve("python", concept.frontmatter["anchors"]["python"]["selector"], REPO_ROOT)
    anchored = {normalise(line) for line in SOURCE[span.start - 1 : span.end]}
    anchored.discard("")

    out: list[str] = []
    index = 0
    changed = 0
    lines = body.split("\n")
    while index < len(lines):
        line = lines[index]
        if not line.startswith("```"):
            out.append(line)
            index += 1
            continue
        info = line[3:].strip()
        # find the closing fence
        close = index + 1
        while close < len(lines) and not lines[close].startswith("```"):
            close += 1
        if info:
            out.extend(lines[index : close + 1])
        else:
            needed = classify(lines[index + 1 : close], anchored)
            if needed:
                out.append(f"```{needed}")
                changed += 1
            else:
                out.append(line)
            out.extend(lines[index + 1 : close + 1])
        index = close + 1

    if changed:
        head, _, _ = path.read_text(encoding="utf-8").partition("\n---\n")
        path.write_text(f"{head}\n---\n" + "\n".join(out), encoding="utf-8")
    return changed


def main() -> int:
    total = 0
    for path in sorted((REPO_ROOT / "content" / "concepts").glob("*.md")):
        count = tag(path)
        total += count
        if count:
            print(f"  {path.name}: tagged {count} fence(s)")
    print(f"tagged {total} fence(s) in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
