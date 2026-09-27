"""Tests for tools/provenance.py's `--refresh-derived`.

A command that rewrites a source file needs a sharper test than the one that
produced it. Two properties matter and they pull in opposite directions:

  1. A `derived=True` entry whose file legitimately changed gets a fresh digest,
     so `--check` stops failing. This is the whole point: the two C ports were
     failing `--check` on a clean checkout of `main` because nothing refreshed
     them, and the docstring claimed `make provenance` did it when no code did.

  2. A `derived=False` entry is **never** rewritten. Its digest is the upstream
     byte sequence, and a refresh that recomputed it from the local file would
     convert the tamper alarm into a rubber stamp. If a vendored file is edited,
     the command must say so and fail.

The second is the one worth a test. The first is exercised by the C ports on
every run, and a test that re-hashes the real C files would be a test that
passes only while nobody touches them.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tools import provenance


def build_sandbox() -> tuple[Path, Path]:
    """A throwaway tree with one derived and one pinned file, plus a copy of the
    manifest to rewrite.

    The pinned file is given its *correct* digest, so the fixture starts in a
    healthy state and a single test can tamper it deliberately. An earlier
    version seeded it with a digest that never matched, which meant the guard
    fired on every test and the derived path was never reached at all.

    The real repository is never touched: `--refresh-derived` writes to
    `Path(__file__)`, so both that and `REPO_ROOT` are repointed at the sandbox
    for the duration of the test and restored afterwards.
    """
    import hashlib

    root = Path(tempfile.mkdtemp(prefix="provenance-sandbox-"))
    (root / "vendored").mkdir()
    (root / "derived").mkdir()

    vendored = root / "vendored" / "upstream.py"
    vendored.write_text("original upstream bytes\n")
    vendored_sha = hashlib.sha256(vendored.read_bytes()).hexdigest()
    ours = root / "derived" / "ours.c"
    ours.write_text("the current revision\n")
    ours_sha = hashlib.sha256(ours.read_bytes()).hexdigest()

    # A minimal manifest with the same declaration shape the real one uses, so
    # the regex that rewrites the sha256 line is exercised for real.
    manifest = root / "provenance.py"
    manifest.write_text(
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "\n"
        "REPO_ROOT = Path(__file__).resolve().parent\n"
        "\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ProvenanceEntry:\n"
        "    path: str\n"
        "    sha256: str\n"
        "    derived: bool = False\n"
        "\n"
        "    def actual_sha256(self):\n"
        "        target = REPO_ROOT / self.path\n"
        "        if not target.is_file():\n"
        "            return None\n"
        "        import hashlib\n"
        "        return hashlib.sha256(target.read_bytes()).hexdigest()\n"
        "\n"
        "\n"
        "PROVENANCE = (\n"
        '    ProvenanceEntry(\n'
        '        path="vendored/upstream.py",\n'
        f'        sha256="{vendored_sha}",\n'
        "    ),\n"
        '    ProvenanceEntry(\n'
        '        path="derived/ours.c",\n'
        f'        sha256="{ours_sha}",\n'
        "        derived=True,\n"
        "    ),\n"
        ")\n"
    )
    return root, manifest


class SandboxCase(unittest.TestCase):
    def setUp(self) -> None:
        import hashlib

        self.root, self.manifest = build_sandbox()
        self.saved_file = provenance.__file__
        self.saved_root = provenance.REPO_ROOT
        self.saved_provenance = provenance.PROVENANCE
        provenance.__file__ = str(self.manifest)
        provenance.REPO_ROOT = self.root
        self.ours_sha = hashlib.sha256(
            (self.root / "derived" / "ours.c").read_bytes()
        ).hexdigest()
        self.vendored_sha = hashlib.sha256(
            (self.root / "vendored" / "upstream.py").read_bytes()
        ).hexdigest()

        def entry(path: str, sha: str, derived: bool) -> provenance.ProvenanceEntry:
            return provenance.ProvenanceEntry(
                path=path,
                sha256=sha,
                origin="",
                url="",
                license="",
                author="",
                note="",
                derived=derived,
            )

        self.entries = (
            entry("vendored/upstream.py", self.vendored_sha, False),
            # Deliberately stale: this is the case the command exists for.
            entry("derived/ours.c", "1" * 64, True),
        )
        provenance.PROVENANCE = self.entries

    def tearDown(self) -> None:
        provenance.__file__ = self.saved_file
        provenance.REPO_ROOT = self.saved_root
        provenance.PROVENANCE = self.saved_provenance
        shutil.rmtree(self.root, ignore_errors=True)


class TestRefreshDerived(SandboxCase):
    def test_a_changed_derived_entry_gets_a_fresh_digest(self) -> None:
        code = provenance.main(["--refresh-derived"])
        self.assertEqual(code, 0)
        text = self.manifest.read_text()
        self.assertIn(self.ours_sha, text)
        self.assertNotIn("1" * 64, text)

    def test_a_pinned_digest_is_never_rewritten(self) -> None:
        provenance.main(["--refresh-derived"])
        self.assertIn(
            self.vendored_sha,
            self.manifest.read_text(),
            "a pinned digest was rewritten from a tampered local file",
        )

    def test_a_pinned_move_fails_loudly(self) -> None:
        (self.root / "vendored" / "upstream.py").write_text("TAMPERED\n")
        code = provenance.main(["--refresh-derived"])
        self.assertEqual(code, 1)
        self.assertIn(self.vendored_sha, self.manifest.read_text())

    def test_the_derived_refresh_still_happens_alongside_a_pinned_failure(self) -> None:
        # The derived entry is fixed and the pinned one is reported, rather than
        # the run aborting before it does anything useful.
        (self.root / "vendored" / "upstream.py").write_text("TAMPERED\n")
        provenance.main(["--refresh-derived"])
        self.assertIn(self.ours_sha, self.manifest.read_text())

    def test_it_is_idempotent(self) -> None:
        provenance.main(["--refresh-derived"])
        once = self.manifest.read_text()
        provenance.main(["--refresh-derived"])
        self.assertEqual(self.manifest.read_text(), once)


class TestTheRealManifestIsInASaneState(unittest.TestCase):
    """On the real repository, not the sandbox."""

    def test_no_derived_entry_is_stale(self) -> None:
        # This is the assertion that was failing on a clean checkout of `main`
        # before `--refresh-derived` existed.
        from tools import check_provenance

        result = check_provenance.run()
        self.assertEqual(result, 0, "run `python3 -m tools.provenance --refresh-derived`")

    def test_every_pinned_entry_is_actually_present_on_disk(self) -> None:
        for entry in provenance.all_entries():
            if entry.derived:
                continue
            self.assertIsNotNone(
                entry.actual_sha256(),
                f"{entry.path} is pinned but missing; the check would pass vacuously",
            )

    def test_the_manifest_docstring_names_the_real_command(self) -> None:
        # The docstring used to claim `make provenance` refreshed derived
        # digests. No code did, and the two C ports drifted as a result.
        text = Path(provenance.__file__).read_text(encoding="utf-8")
        self.assertIn("--refresh-derived", text)
        self.assertNotIn("refreshed with\n    #: `make provenance`", text)


if __name__ == "__main__":
    unittest.main()
