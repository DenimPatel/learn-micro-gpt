#!/usr/bin/env python3
"""Verify pinned vendored files still match their recorded sha256.

The pin itself lives in `tools/provenance.py`. This module recomputes the
digests and reports drift. A failure here means one of two things:

  1. Somebody edited a vendored file without saying so. That is a bug.
  2. Upstream changed and somebody re-fetched. That needs a deliberate,
     reviewable edit to the pin in `tools/provenance.py` -- and a note in the
     pull request explaining why the reference changed.

Either way it should stop the build. `reference/microgpt.py` is the reference
track: the browser runs it, the parity gate measures against it, and the
concepts' anchors are resolved against it. If its bytes move, every recorded
number in this repository silently becomes a lie.
"""

from __future__ import annotations

import sys

from tools.provenance import all_entries

#: Bumped when a pin is changed on purpose. CI records it, so a diff that
#: moves the reference without moving this shows up in review.
PIN_EPOCH = 1


def run() -> int:
    failures: list[str] = []

    for entry in all_entries():
        actual = entry.actual_sha256()
        if actual is None:
            failures.append(f"{entry.path}: file is missing")
            continue
        if actual != entry.sha256:
            kind = "derived digest is stale" if entry.derived else "PINNED FILE MODIFIED"
            failures.append(
                f"{entry.path}: {kind}\n"
                f"      expected {entry.sha256}\n"
                f"      actual   {actual}"
            )
            continue
        if entry.derived:
            print(f"ok  {entry.path}  {actual}  (derived, recorded)")
        else:
            print(f"ok  {entry.path}  {actual}  (pinned)")

    if failures:
        print("\nprovenance check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  ! {failure}", file=sys.stderr)
        print(
            "\nIf the change was deliberate: re-run `make provenance` to refresh\n"
            "the recorded digests, and bump PIN_EPOCH if reference/microgpt.py moved\n"
            "(which also invalidates traces/ and benchmarks/results.json).",
            file=sys.stderr,
        )
        return 1

    print(f"\nprovenance ok (pin epoch {PIN_EPOCH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
