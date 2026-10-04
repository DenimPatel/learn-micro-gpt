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

A digest is only a claim about *bytes*. `reference/microgpt-annotated.py` is a
heavily commented fork of the reference, and its provenance entry is
`derived=True`: the digest is recorded, not enforced, so any edit passes. A
stray edit to a constant or an operator would therefore satisfy every gate here
while quietly turning the reading aid into a different program. The second phase
below runs `tools.check_annotated_sync`, which compares the two files' *code*
(AST, modulo comments and wrapping) and fails on exactly that edit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Run as `python3 -m tools.check_provenance`; see tools/_entrypoint.py. This must
# precede `from tools import ...`: a script run leaves `tools/` on sys.path but
# not the repository root, so the package is not importable at all.
_spec = importlib.util.spec_from_file_location(
    "_tools_entrypoint", Path(__file__).resolve().parent / "_entrypoint.py"
)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)
_guard.require_module_invocation("check_provenance", __package__)

from tools.provenance import all_entries
from tools import check_annotated_sync

#: Bumped when a pin is changed on purpose. CI records it, so a diff that
#: moves the reference without moving this shows up in review.
PIN_EPOCH = 1


def _check_annotated_sync() -> list[str]:
    """The second phase: does the annotated fork still carry the reference code?

    Kept separate from the digest loop because it answers a different question.
    A digest failure is "these bytes moved"; this is "these bytes are not the
    same program any more". Both stop the build, but only this one catches an
    edit to a `derived=True` file, which is the hole this phase closes.
    """
    root = Path(__file__).resolve().parent.parent
    reference = (root / check_annotated_sync.REFERENCE_PATH).read_text(encoding="utf-8")
    annotated = (root / check_annotated_sync.ANNOTATED_PATH).read_text(encoding="utf-8")
    drifts = check_annotated_sync.compare(reference, annotated)
    if not drifts:
        print(f"ok  {check_annotated_sync.ANNOTATED_PATH}  (code matches the reference)")
        return []
    return [
        f"{check_annotated_sync.ANNOTATED_PATH}: {drift.where}: {drift.detail}"
        for drift in drifts
    ]


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

    failures.extend(_check_annotated_sync())

    if failures:
        print("\nprovenance check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  ! {failure}", file=sys.stderr)
        print(
            "\nIf the change was deliberate: re-run `make provenance` to refresh\n"
            "the recorded digests, and bump PIN_EPOCH if reference/microgpt.py moved\n"
            "(which also invalidates traces/ and benchmarks/results.json).\n"
            "An annotated-sync failure is different: the annotated fork drifted from\n"
            "the reference's *code*. Move the two files together, or see the\n"
            "reference's `make verify-provenance` output for the exact statement.",
            file=sys.stderr,
        )
        return 1

    print(f"\nprovenance ok (pin epoch {PIN_EPOCH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
