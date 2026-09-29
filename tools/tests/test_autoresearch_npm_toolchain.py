"""Tests for the npm scratch-tree protocol in tools/autoresearch.py.

## The bug these are for

`apply_to_scratch` excluded `node_modules` from the copy, on the stated belief that
"`npx tsx` will rebuild or reuse what it needs". For `target/` that is true:
`cargo build` puts it back in the tree it was asked to build. For `node_modules` it
is false, and not expensively false -- npm has no build step, nothing in this file
installs anything, and `npx` resolves a binary it cannot find against the *registry*
rather than against the project. It fetched the npm package named `tsc`, which is a
deprecated stub that prints "This is not the tsc command you are looking for" and
exits 1.

`build_track` then raised `tsc --noEmit failed` for every proposal, correct or not,
and every one of them became a `crash` row in `autoresearch/results.tsv` with that
reason. Runs 0171-0231 on the TypeScript track: sixty-one rows, sixty-one of them
this, and not one of them the model's fault.

The second half of the fix is the one that matters longer. A harness that cannot
tell "the toolchain is missing" from "the candidate does not typecheck" will keep
manufacturing evidence about candidates it never measured, and the ledger will
keep looking like a record of a model's failures. So `require_local_tsc` exists to
make the two messages different, and the tests below hold them apart.

## What these tests may assume

Nothing about the machine they run on. The first version of this file reached for
the repository's own `autoresearch/candidate-ts/node_modules` and
`implementations/typescript/node_modules`, which is a thing a developer has and a
checkout does not: `.gitignore` keeps `node_modules/` out, so every one of those
tests passed at home and errored on CI -- four `ResearchError`s from the `npm ci`
fallback and two failures from a subtest that wanted a compiler it assumed was
there. A test whose result depends on whether somebody ran `npm ci` is not a test
of this code, it is a test of the machine, and the machine is the one thing a test
cannot control.

So every tree here is built by `FixtureTrackTestCase` in a throwaway directory, and
the `node_modules` in it is two stub files. Nothing in this file runs a training
loop, nothing reads the repository's real `node_modules`, and the fixture makes
the `npm ci` fallback unreachable rather than merely unused -- see the note on it
in `setUp`. What the real committed trees have installed is not this file's
business: `make autoresearch-verify` runs this same guard against them, on every
CI run, against the actual toolchain.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import autoresearch as ar

TYPESCRIPT = ar.get_track("typescript")
RUST = ar.get_track("rust")

#: The message a *candidate* failure produces. The whole point of the preflight is
#: that a missing toolchain never produces this one, so the tests assert against
#: this string rather than against the current wording of the guard.
CANDIDATE_FAILURE = "tsc --noEmit failed"

#: Each fixture tree's `node_modules` carries its own marker, so a test that claims
#: the link points at one particular tree can prove which by reading the file
#: through the link, instead of by trusting that only one of them exists.
CANDIDATE_MARKER = "the candidate's tree"
FROZEN_MARKER = "the frozen comparator's tree"

#: The fixture candidate's `src/index.ts`. Not a copy of the real one: the patch
#: below is generated against this file, so a test about how a patch is applied
#: does not start failing because somebody edited `candidate-ts/src/index.ts` for
#: a reason of their own.
FIXTURE_SOURCE = """\
/** A scalar in the computation graph. Karpathy's `class Value`, in TypeScript. */
export class Value {
  constructor(public data: number) {}
}

export function loss(v: Value): number {
  return v.data ** 2
}
"""

#: The line the patch is spliced in front of. A doc comment opening the file, which
#: is where the real candidate's source opens too, so the shape of the patch a test
#: builds is the shape the loop asks a model for.
FIXTURE_INSERTION_POINT = "export class Value {"


def dependency_tree(project: Path, marker: str) -> Path:
    """The `node_modules` a committed npm project has, minus 37 MB of install.

    Two files, and nothing that runs them. `require_local_tsc` stats
    `node_modules/.bin/tsc` and `link_dependencies` links the directory, so a
    `stat` and a symlink are the whole protocol here and node is never asked to
    confirm anything -- which is what lets the same fixture serve a checkout with
    the toolchain installed and one without it.
    """
    binaries = project / "node_modules" / ".bin"
    binaries.mkdir(parents=True)
    for name in ("tsc", "tsx"):
        (binaries / name).write_text(
            f"#!/bin/sh\n# {marker}\nexit 0\n", encoding="utf-8"
        )
    return project / "node_modules"


class FixtureTrackTestCase(unittest.TestCase):
    """A throwaway committed tree, and a `Track` that resolves to it.

    Both directories are removed whether the test passes or fails, and neither of
    them is the repository's own tree, so what these tests measure is the code and
    not whether somebody remembered to run `npm ci`.
    """

    def setUp(self) -> None:
        # Under `autoresearch/` rather than in the system temp directory, and the
        # reason is in the comment on `self.track` below: `candidate_dir` becomes
        # a path in a patch, and `git apply` will not accept one containing `..`.
        # The prefix is so that a run killed before its cleanup leaves something
        # recognisable in `git status` rather than a mystery directory.
        self.root = Path(
            tempfile.mkdtemp(dir=ar.RESEARCH_DIR, prefix="test-npm-fixture-")
        )
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        # The empty directory a scratch copy would have been measured in: the
        # target `link_dependencies` is asked to put a dependency tree into.
        self.scratch = Path(tempfile.mkdtemp(prefix="test-npm-scratch-"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)

        # The shape `apply_to_scratch` copies: a `candidate-ts`-like directory
        # under an `autoresearch` one, with a real source file for a patch to
        # target and a dependency tree to be linked in afterwards.
        self.candidate = self.root / "autoresearch" / "candidate-ts"
        (self.candidate / "src").mkdir(parents=True)
        (self.candidate / "package.json").write_text(
            '{"name": "candidate-ts"}\n', encoding="utf-8"
        )
        (self.candidate / "src" / "index.ts").write_text(
            FIXTURE_SOURCE, encoding="utf-8"
        )
        dependency_tree(self.candidate, CANDIDATE_MARKER)

        # The comparator, carrying a dependency tree of its own as
        # `implementations/typescript/` does on a machine that has run `npm ci`.
        self.comparator = self.root / "implementations" / "typescript"
        (self.comparator / "src").mkdir(parents=True)
        (self.comparator / "src" / "index.ts").write_text(
            FIXTURE_SOURCE, encoding="utf-8"
        )
        dependency_tree(self.comparator, FROZEN_MARKER)

        # A third tree: committed-shaped, and never installed. The guard needs a
        # pair to say anything -- one tree with a dependency tree and one without
        # is the property -- and a single directory can only ever show one half.
        self.uninstalled = self.root / "uninstalled"
        (self.uninstalled / "src").mkdir(parents=True)
        (self.uninstalled / "package.json").write_text(
            '{"name": "uninstalled"}\n', encoding="utf-8"
        )
        (self.uninstalled / "src" / "index.ts").write_text(
            FIXTURE_SOURCE, encoding="utf-8"
        )

        # `Track` resolves both names against the repository layout --
        # `RESEARCH_DIR / candidate_dir` and `REPO_ROOT / "implementations" /
        # comparator_dir` -- so a fixture is addressed by a path relative to those,
        # and `os.path.relpath` is what makes one.
        #
        # The candidate's relpath must not contain `..`, and that is a constraint
        # rather than tidiness. `apply_to_scratch` copies the tree to
        # `<scratch>/repo/autoresearch/<candidate_dir>` and applies a patch whose
        # paths are `a/<source_key>`, and `git apply` refuses a path containing
        # `..` outright: "error: invalid path ...". A fixture in the system temp
        # directory needs `..` to be reachable from `autoresearch/`, and it also
        # sends the copy to `<scratch>/repo/autoresearch/../../...` -- a path that
        # depends on how deep the checkout and the scratch directory happen to be,
        # and on this laptop resolved to `/var/folders/var/...`, which is not
        # writable by anyone. Keeping the fixture under `autoresearch/` makes
        # `candidate_dir` a plain relative name, which also keeps
        # `track.candidate.relative_to(ar.REPO_ROOT)` -- called by the error
        # messages in `require_local_tsc` and `link_dependencies`, and a
        # `ValueError` on a path outside the repository -- inside it.
        #
        # The comparator's relpath *does* contain `..`, and that is fine: only
        # `candidate_dir` reaches a patch, and the comparator is only ever a
        # directory to point a guard at.
        self.track = dataclasses.replace(
            TYPESCRIPT,
            candidate_dir=os.path.relpath(self.candidate, ar.RESEARCH_DIR),
            comparator_dir=os.path.relpath(
                self.comparator, ar.REPO_ROOT / "implementations"
            ),
        )

        # `link_dependencies`'s last resort is `npm ci`, which wants the network
        # and a `package-lock.json` to install from. Every test here either
        # supplies a tree or asks about a directory that already has one, so the
        # fallback is never the right branch -- and refusing `npm` outright makes
        # that a fact rather than a promise. It is also the branch that broke CI:
        # an empty scratch directory, a committed tree with no `node_modules` in
        # it, and `npm ci` exiting non-zero inside a test suite that is supposed
        # to need neither a network nor a `node_modules`.
        real = ar.run_command

        def no_installs(argv, **kwargs):
            if [str(part) for part in argv][:1] == ["npm"]:
                raise AssertionError(
                    f"a test reached the `npm ci` fallback: {argv}. The fixture has "
                    "a dependency tree, so the link above this fallback should "
                    "always have happened."
                )
            return real(argv, **kwargs)

        patcher = mock.patch.object(ar, "run_command", no_installs)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestTheScratchTreeGetsADependencyTree(FixtureTrackTestCase):
    def test_a_npm_track_gets_its_node_modules_back(self) -> None:
        """The regression itself, at the level it happened.

        A scratch tree with no `node_modules` cannot compile, probe or train. The
        check is that the directory is *reachable* -- a symlink resolves exactly as
        a copy does for every consumer here, `node_modules/.bin/tsc` and
        `node_modules/.bin/tsx` are relative shims into the committed tree -- so
        `Path.exists()` is the same assertion as "npx will find a real compiler",
        and it does not need node to say so.
        """
        ar.link_dependencies(self.track, self.scratch)
        link = self.scratch / "node_modules"
        self.assertTrue(link.is_symlink(), "node_modules should be linked, not absent")
        self.assertEqual(
            Path(link).resolve(), (self.candidate / "node_modules").resolve()
        )
        self.assertTrue((link / ".bin" / "tsc").is_file())
        self.assertTrue((link / ".bin" / "tsx").is_file())

    def test_it_links_rather_than_copying_seventy_megabytes(self) -> None:
        # 37 MB per experiment, for a tree that is deleted when the row is written.
        # A copy would also be a *second* thing to go stale: the committed one is
        # the tree `npm ci` last produced, and a copy of a copy is harder to argue
        # about than a link to it.
        ar.link_dependencies(self.track, self.scratch)
        self.assertTrue((self.scratch / "node_modules").is_symlink())

    def test_a_track_whose_toolchain_rebuilds_its_own_output_is_left_alone(self) -> None:
        # `target/` really is rebuilt by `cargo build`, so linking it would be
        # wrong in the other direction: it would pin a scratch tree to the
        # committed build cache and measure against a stale one.
        ar.link_dependencies(RUST, self.scratch)
        self.assertFalse((self.scratch / "node_modules").exists())

    def test_a_tree_that_already_has_one_is_not_touched(self) -> None:
        # `link_dependencies` runs after the copy, so it must not clobber a
        # directory that is already there -- and it must not raise when there is one.
        existing = self.scratch / "node_modules"
        existing.mkdir()
        ar.link_dependencies(self.track, self.scratch)
        self.assertFalse(existing.is_symlink())

    def test_the_missing_tree_fallback_says_where_to_install_and_stops(self) -> None:
        """The order of the three things `link_dependencies` does.

        A symlink, an `npm ci` fallback, and a raise. They have to be in that
        order, and the assertion is on line numbers rather than on a substring
        because the docstring also says "npm ci" -- which is how the first version
        of this test passed while asserting nothing.
        """
        function = ast.parse(inspect.getsource(ar.link_dependencies)).body[0]
        docstring = function.body[0]
        symlink = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "symlink_to"
        )
        install = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.List)
            and [element.value for element in node.args[0].elts][:2] == ["npm", "ci"]
        )
        stop = next(
            node for node in ast.walk(function) if isinstance(node, ast.Raise)
        )
        self.assertLess(docstring.end_lineno, symlink.lineno)
        self.assertLess(
            symlink.lineno, install.lineno,
            "npm ci must be the fallback, so a checkout with a committed "
            "node_modules never pays for an install",
        )
        self.assertLess(
            install.lineno, stop.lineno,
            "a failed install must end the run, not fall through to a tree that "
            "cannot typecheck",
        )


class TestAPhaseOfAMissingToolchainCannotLookLikeACandidateFailure(FixtureTrackTestCase):
    def test_a_bare_npm_tree_raises_rather_than_compiling(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(self.track, self.scratch)
        message = str(caught.exception)
        self.assertNotEqual(message.splitlines()[0], CANDIDATE_FAILURE)
        self.assertNotIn(f"{CANDIDATE_FAILURE}\n", message)
        self.assertNotIn(CANDIDATE_FAILURE, message.splitlines()[0])

    def test_the_message_names_the_directory_the_dependency_is_missing_from(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(self.track, self.scratch)
        message = str(caught.exception)
        self.assertIn(str(self.scratch), message)
        self.assertIn("node_modules", message)
        self.assertIn(".bin/tsc", message)

    def test_the_message_says_what_to_do_about_it(self) -> None:
        # A guard that only says "something is wrong" is a slower version of the
        # bug: an operator who cannot act on the message has to go and find out
        # what changed, which is the manual step this was supposed to remove.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(self.track, self.scratch)
        message = str(caught.exception)
        self.assertIn("npm ci", message)
        self.assertIn(
            self.track.candidate.relative_to(ar.REPO_ROOT).as_posix(), message
        )

    def test_it_says_which_of_the_two_messages_is_which(self) -> None:
        # The regression's cost was 61 rows in a published ledger that all read
        # `tsc --noEmit failed` and none of which meant it. The new message has to
        # name the old one so that a reader of either can tell what happened.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(self.track, self.scratch)
        self.assertIn(CANDIDATE_FAILURE, str(caught.exception))

    def test_both_ways_into_the_compiler_are_guarded(self) -> None:
        """`build_track` and `lint_track` are two independent paths to `npx tsc`.

        `measure_track` happens to call the first before the second today. A
        check that is only correct because of that ordering is one refactor away
        from being wrong, and the cost of being wrong is the whole loop.
        """
        for name in ("build_track", "lint_track"):
            with self.subTest(entry_point=name):
                source = inspect.getsource(getattr(ar, name))
                self.assertIn(
                    "require_local_tsc(track, project_dir)", source,
                    f"{name} reaches the compiler without checking that the project "
                    f"has one",
                )

    def test_the_guard_is_satisfied_by_a_linked_dependency_tree(self) -> None:
        ar.link_dependencies(self.track, self.scratch)
        ar.require_local_tsc(self.track, self.scratch)  # must not raise

    def test_the_guard_does_nothing_for_a_track_that_has_no_npm_step(self) -> None:
        # Rust has a real `tsc --noEmit` failure mode of its own -- none at all.
        # A guard that ran for every track would invent a dependency the track does
        # not have, and eventually fail on a perfectly good checkout.
        ar.require_local_tsc(RUST, self.scratch)

    def test_an_installed_tree_passes_the_guard_and_an_uninstalled_one_does_not(
        self,
    ) -> None:
        """The property the comparator test was reaching for, hermetically.

        `measure_session_baseline` builds the comparator and then the candidate,
        in that order, and both are committed trees. So the guard has to answer the
        same way for either: a tree that still has its dependency tree compiles,
        and a tree that has lost it says so in the operator's voice -- not in the
        candidate's.

        It is a pair of trees rather than one directory because the difference is
        the property. The first version of this test asserted that the
        *repository's* two trees happened to be installed, which is a statement
        about a developer's machine, and `.gitignore` keeps `node_modules/` out of
        a checkout, so it failed on CI for exactly the state it was asserting. The
        real committed trees are not left undefended by dropping it:
        `make autoresearch-verify` runs this same guard against them.
        """
        ar.require_local_tsc(self.track, self.candidate)  # must not raise

        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(self.track, self.uninstalled)
        message = str(caught.exception)
        self.assertNotIn(CANDIDATE_FAILURE, message.splitlines()[0])
        self.assertIn(str(self.uninstalled), message)
        self.assertIn("npm ci", message)


class TestApplyToScratchProducesAMeasurableTree(FixtureTrackTestCase):
    def test_the_copy_is_followed_by_the_link(self) -> None:
        """`apply_to_scratch` end to end, minus the measurement.

        The two halves of the fix are one line apart and a copy without the link
        is exactly the tree that caused 61 crashes, so they are checked together:
        `node_modules` is absent from the copy and present in the result. The
        result's is a *symlink*, which is the part that shows the copy did not
        carry a tree of its own -- a copy would be a real directory, and
        `link_dependencies` returns early for one.

        The patch is generated from the fixture's own file rather than written out
        by hand or read from the repository's, so this test cannot start failing
        because somebody edited `candidate-ts/src/index.ts` -- and a test that
        fails for a reason that has nothing to do with what it is testing is a test
        that gets deleted.
        """
        source = self.track.candidate_file.read_text(encoding="utf-8")
        self.assertIn(
            FIXTURE_INSERTION_POINT, source, "the fixture lost its insertion point"
        )
        cut = source.index(FIXTURE_INSERTION_POINT)
        patched = (
            source[:cut]
            + "/** Exported, so `noUnusedLocals` is satisfied; read by nothing. */\n"
            + "export const TEST_NOOP = 0\n\n"
            + source[cut:]
        )
        patch = "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{self.track.source_key}",
                tofile=f"b/{self.track.source_key}",
            )
        )
        applied, project, error = ar.apply_to_scratch(self.track, patch)
        self.addCleanup(ar.cleanup_scratch)
        self.assertTrue(applied, error)

        link = project / "node_modules"
        self.assertTrue(link.is_symlink(), "the copy should have been linked to")
        self.assertTrue((link / ".bin" / "tsc").is_file())
        self.assertTrue((link / ".bin" / "tsx").is_file())
        self.assertTrue((project / self.track.candidate_source).is_file())
        self.assertIn(
            "TEST_NOOP",
            (project / self.track.candidate_source).read_text(encoding="utf-8"),
        )
        # The committed candidate is untouched: a patch that fails must not be able
        # to damage the thing the next experiment starts from.
        self.assertNotIn("TEST_NOOP", source)
        self.assertNotIn(
            "TEST_NOOP", self.track.candidate_file.read_text(encoding="utf-8")
        )

    def test_the_comparator_is_not_what_gets_linked(self) -> None:
        # The link points at the candidate's tree, not the frozen one's. The
        # comparator is pinned and its `node_modules` is a different install; a
        # scratch candidate measured against it would be measuring a toolchain
        # nobody proposed. The fixture's comparator has a dependency tree too, with
        # a marker of its own in it, so this is a difference the test can see
        # rather than an absence it has to assume.
        ar.link_dependencies(self.track, self.scratch)
        self.assertEqual(
            Path(self.scratch / "node_modules").resolve().parent,
            self.candidate.resolve(),
        )
        self.assertIn(
            CANDIDATE_MARKER,
            (self.scratch / "node_modules" / ".bin" / "tsc").read_text(
                encoding="utf-8"
            ),
        )


if __name__ == "__main__":
    unittest.main()
