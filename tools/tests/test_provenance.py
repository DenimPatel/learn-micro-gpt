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

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tools import check_provenance, provenance

REPO_ROOT = Path(__file__).resolve().parents[2]


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


class TestTheCheckFlagActuallyChecks(unittest.TestCase):
    """`--check` must delegate to `check_provenance`, and report what it finds.

    `--check` is the flag `make verify-provenance` drives, and until this test
    existed nothing exercised the delegation. Renaming `check_provenance.run` to
    `main` meant editing the call site in `provenance.py`, and breaking that call
    site left the whole suite green -- a renamed dead branch would have been
    caught by CI on the next drift and by nothing else.

    The fixture moves one global: `tools.provenance.REPO_ROOT`. It has to be that
    global and not a copy of `provenance.py`: `check_provenance` does `from
    tools.provenance import all_entries`, so the check always calls the function
    in the real module and always reads that module's `REPO_ROOT`. A fixture that
    copied `provenance.py` into a temp package looked right and asserted nothing
    -- the first version of this test tampered with a pinned file, saw exit 0,
    and the check turned out to be reading the real repository the whole time.

    Tampering in a sandbox rather than asserting on the live tree also keeps the
    test from depending on whether anyone happens to have edited the repository.
    """

    def _sandbox(self) -> Path:
        """Copy every pinned file into a temp tree and point the check at it.

        `all_entries` resolves `REPO_ROOT` from its own module globals at call
        time, so that entry in `tools.provenance.__dict__` is what decides which
        files are hashed. `check_provenance` holds no copy of it.
        """
        root = Path(tempfile.mkdtemp(prefix="provenance-check-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for entry in provenance.all_entries():
            destination = root / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / entry.path, destination)

        globals_of_all_entries = provenance.all_entries.__globals__
        original = globals_of_all_entries["REPO_ROOT"]
        self.addCleanup(globals_of_all_entries.__setitem__, "REPO_ROOT", original)
        globals_of_all_entries["REPO_ROOT"] = root
        return root

    def test_the_check_flag_passes_on_a_healthy_tree(self) -> None:
        self._sandbox()
        self.assertEqual(provenance.main(["--check"]), 0)

    def test_the_check_flag_fails_on_a_tampered_pin(self) -> None:
        root = self._sandbox()
        pinned = next(entry for entry in provenance.all_entries() if not entry.derived)
        (root / pinned.path).write_text("TAMPERED\n")
        self.assertEqual(provenance.main(["--check"]), 1)

    def test_the_check_flag_fails_on_a_missing_pin(self) -> None:
        # A deleted file is the other half of the same report, and the branch is
        # separate: `actual_sha256` returns None rather than a wrong digest.
        root = self._sandbox()
        pinned = next(entry for entry in provenance.all_entries() if not entry.derived)
        (root / pinned.path).unlink()
        self.assertEqual(provenance.main(["--check"]), 1)

    def test_a_listing_without_the_flag_is_not_a_check(self) -> None:
        """The default branch prints digests and must not fail.

        Guards the other direction: if `--check` were wired to the default
        branch, or the reverse, the tampered case above would still pass as a
        pair -- so this pins the default path's exit code on a broken tree, which
        is what tells the two branches apart.
        """
        root = self._sandbox()
        pinned = next(entry for entry in provenance.all_entries() if not entry.derived)
        (root / pinned.path).write_text("TAMPERED\n")
        self.assertEqual(provenance.main([]), 0)


class TestTheRealManifestIsInASaneState(unittest.TestCase):
    """On the real repository, not the sandbox."""

    def test_no_derived_entry_is_stale(self) -> None:
        # This is the assertion that was failing on a clean checkout of `main`
        # before `--refresh-derived` existed.
        from tools import check_provenance

        result = check_provenance.main()
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


def _without_comments(text: str) -> str:
    """Strip C comments, so a mention inside one does not count as a use."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


class TestTheCHeadersItActuallyNeeds(unittest.TestCase):
    """A portability check that does not need the platform it is checking.

    `microgpt-scaled.c` used `uint32_t` without including `<stdint.h>`. Apple's
    headers pull stdint.h in transitively through stdio.h and glibc's do not, so
    the file compiled on every Mac and failed on the Linux CI runner with
    `'uint32_t' undeclared` -- in a build that had never run, because the workflow
    containing it has been failing at 0s since it was added.

    The general form is a file using a type whose header it does not include,
    relying on someone else's to include it, where the two platform C libraries
    disagree about which. So the check is static: for each C file, does it use a
    type from a header it does not include?

    Not a substitute for building on the other platform -- a type from
    `<stdbool.h>` or a POSIX header would slip past it. It catches the class that
    actually bit us, on a machine that cannot build for the platform that noticed.
    """

    #: Type name -> the header required to declare it. Deliberately short: these
    #: have a well-known single owning header, and every entry is a thing that has
    #: been found missing rather than one imagined.
    REQUIRED_HEADER = {
        "uint8_t": "stdint.h",
        "uint16_t": "stdint.h",
        "uint32_t": "stdint.h",
        "uint64_t": "stdint.h",
        "int8_t": "stdint.h",
        "int16_t": "stdint.h",
        "int32_t": "stdint.h",
        "int64_t": "stdint.h",
        "intptr_t": "stdint.h",
        "uintptr_t": "stdint.h",
    }

    SOURCES = (
        "implementations/c/microgpt.c",
        "implementations/c/microgpt-scaled.c",
        "implementations/c/test_gradients.c",
        "implementations/c/microgpt_simd.h",
    )

    def test_no_c_file_uses_a_type_whose_header_it_omits(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for relative in self.SOURCES:
            with self.subTest(source=relative):
                code = _without_comments((root / relative).read_text(encoding="utf-8"))
                includes = set(re.findall(r"#\s*include\s*<([^>]+)>", code))
                for type_name, header in self.REQUIRED_HEADER.items():
                    if not re.search(rf"\b{type_name}\b", code):
                        continue
                    self.assertIn(
                        header,
                        includes,
                        f"{relative} uses {type_name} but does not include <{header}>. "
                        f"It may compile here because another header pulls it in, and "
                        f"fail on a runner whose libc does not.",
                    )

    def test_the_check_would_notice_a_removed_include(self) -> None:
        """A guard on the guard.

        Otherwise this test is a check that silently finds nothing, which is
        indistinguishable from a check that works -- and the whole reason this
        check exists is a bug nothing noticed.
        """
        offenders = []
        for type_name, header in self.REQUIRED_HEADER.items():
            code = "uint32_t x;\n#include <stdio.h>\n"
            if re.search(rf"\b{type_name}\b", code) and header not in re.findall(
                r"#\s*include\s*<([^>]+)>", code
            ):
                offenders.append(type_name)
        self.assertIn("uint32_t", offenders)
