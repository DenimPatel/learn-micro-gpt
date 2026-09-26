#!/usr/bin/env python3
"""Regenerate `docs/PROVENANCE.md` from `tools/provenance.py`.

Run via `make provenance`. The doc is generated rather than hand-written so it
cannot drift from the pins it describes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.provenance import all_entries  # noqa: E402

HEADER = """<!-- GENERATED FILE -- do not edit by hand. Run `make provenance`. -->
# Provenance

Every file in this repository that came from somewhere else, with its origin,
its license, and a sha256 digest. This document is generated from
`tools/provenance.py`, which is also what CI's provenance job checks against.

## Why this exists

A teaching repository that quietly ships other people's code without saying
whose it is is worse than no repository. So:

- **Pinned** files are byte-compared on every CI run. If the bytes move, the
  build fails. Changing a pin is a deliberate, reviewable act.
- **Derived** files are files we adapted. Their digest is recorded so you can
  see what we shipped, but they are expected to change. Their upstream and
  license are still mandatory, and a license file is vendored alongside them.
- Anything that is *ours* is MIT, in `LICENSE`, with no digest needed.

## Pinned vs derived

`tools/provenance.py` marks each entry `derived`. A pinned entry means
"upstream bytes, do not touch". A derived entry means "upstream bytes plus our
work, tracked but not frozen".

## Summary

| File | Status | License |
| --- | --- | --- |
"""

FOOTER = """
## Licenses reproduced in-tree

- `THIRD_PARTY_LICENSES/microgpt-c-LICENSE` -- MIT, Copyright (c) 2026 Vishal
  (TheVixhal), for the C port lineage.
- `LICENSE` -- MIT for this repository's own code.

## Changing a pin

1. Change the file deliberately, with a commit message that says why.
2. Re-run `make provenance` to refresh this document.
3. If you changed `reference/microgpt.py`, bump `PIN_EPOCH` in
   `tools/check_provenance.py` and re-record every number in
   `benchmarks/results.json` and `traces/`. The old numbers describe the old
   bytes.
"""


def main() -> int:
    entries = all_entries()
    rows = []
    details = []
    for entry in entries:
        status = "derived" if entry.derived else "**pinned**"
        rows.append(f"| `{entry.path}` | {status} | {entry.license} |")
        digest = entry.sha256
        details.append(
            f"### `{entry.path}`\n\n"
            f"- **Status:** {'derived (adapted by us)' if entry.derived else 'pinned (upstream bytes)'}\n"
            f"- **sha256:** `{digest}`\n"
            f"- **Origin:** {entry.origin}\n"
            f"- **Author / copyright:** {entry.author}\n"
            f"- **License:** {entry.license}\n"
            f"- **Upstream:** <{entry.url}>\n"
            + (f"- **License text vendored at:** `{entry.license_file}`\n" if entry.license_file else "")
            + f"\n{entry.note}\n"
        )

    out = (
        HEADER
        + "\n".join(rows)
        + "\n\n## File detail\n\n"
        + "\n".join(details)
        + FOOTER
    )

    target = REPO_ROOT / "docs" / "PROVENANCE.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.read_text() if target.exists() else None
    if previous == out:
        print(f"up to date: {target.relative_to(REPO_ROOT)}")
        return 0
    target.write_text(out)
    print(f"wrote {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
