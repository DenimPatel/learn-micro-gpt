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

Nothing here runs a training loop and nothing calls the network, for the reason
`test_autoresearch.py` gives: `apply_to_scratch` and the preflight are the code
under test, and neither needs a measurement to be exercised.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import shutil
import tempfile
import unittest
from pathlib import Path

from tools import autoresearch as ar

TYPESCRIPT = ar.get_track("typescript")
RUST = ar.get_track("rust")

#: The message a *candidate* failure produces. The whole point of the preflight is
#: that a missing toolchain never produces this one, so the tests assert against
#: this string rather than against the current wording of the guard.
CANDIDATE_FAILURE = "tsc --noEmit failed"


class ScratchTreeTestCase(unittest.TestCase):
    """Each test gets its own throwaway directory, cleaned up either way."""

    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp(prefix="test-npm-scratch-"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)


class TestTheScratchTreeGetsADependencyTree(ScratchTreeTestCase):
    def test_a_npm_track_gets_its_node_modules_back(self) -> None:
        """The regression itself, at the level it happened.

        A scratch tree with no `node_modules` cannot compile, probe or train. The
        check is that the directory is *reachable* -- a symlink resolves exactly as
        a copy does for every consumer here, `node_modules/.bin/tsc` and
        `node_modules/.bin/tsx` are relative shims into the committed tree -- so
        `Path.exists()` is the same assertion as "npx will find a real compiler",
        and it does not need node to say so.
        """
        ar.link_dependencies(TYPESCRIPT, self.scratch)
        link = self.scratch / "node_modules"
        self.assertTrue(link.is_symlink(), "node_modules should be linked, not absent")
        self.assertEqual(
            Path(link).resolve(), (TYPESCRIPT.candidate / "node_modules").resolve()
        )
        self.assertTrue((link / ".bin" / "tsc").is_file())
        self.assertTrue((link / ".bin" / "tsx").is_file())

    def test_it_links_rather_than_copying_seventy_megabytes(self) -> None:
        # 37 MB per experiment, for a tree that is deleted when the row is written.
        # A copy would also be a *second* thing to go stale: the committed one is
        # the tree `npm ci` last produced, and a copy of a copy is harder to argue
        # about than a link to it.
        ar.link_dependencies(TYPESCRIPT, self.scratch)
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
        ar.link_dependencies(TYPESCRIPT, self.scratch)
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


class TestAPhaseOfAMissingToolchainCannotLookLikeACandidateFailure(ScratchTreeTestCase):
    def test_a_bare_npm_tree_raises_rather_than_compiling(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(TYPESCRIPT, self.scratch)
        message = str(caught.exception)
        self.assertNotEqual(message.splitlines()[0], CANDIDATE_FAILURE)
        self.assertNotIn(f"{CANDIDATE_FAILURE}\n", message)
        self.assertNotIn(CANDIDATE_FAILURE, message.splitlines()[0])

    def test_the_message_names_the_directory_the_dependency_is_missing_from(self) -> None:
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(TYPESCRIPT, self.scratch)
        message = str(caught.exception)
        self.assertIn(str(self.scratch), message)
        self.assertIn("node_modules", message)
        self.assertIn(".bin/tsc", message)

    def test_the_message_says_what_to_do_about_it(self) -> None:
        # A guard that only says "something is wrong" is a slower version of the
        # bug: an operator who cannot act on the message has to go and find out
        # what changed, which is the manual step this was supposed to remove.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(TYPESCRIPT, self.scratch)
        message = str(caught.exception)
        self.assertIn("npm ci", message)
        self.assertIn(
            TYPESCRIPT.candidate.relative_to(ar.REPO_ROOT).as_posix(), message
        )

    def test_it_says_which_of_the_two_messages_is_which(self) -> None:
        # The regression's cost was 61 rows in a published ledger that all read
        # `tsc --noEmit failed` and none of which meant it. The new message has to
        # name the old one so that a reader of either can tell what happened.
        with self.assertRaises(ar.ResearchError) as caught:
            ar.require_local_tsc(TYPESCRIPT, self.scratch)
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

    def test_the_guard_is_satisfied_by_a_real_dependency_tree(self) -> None:
        ar.link_dependencies(TYPESCRIPT, self.scratch)
        ar.require_local_tsc(TYPESCRIPT, self.scratch)  # must not raise

    def test_the_guard_does_nothing_for_a_track_that_has_no_npm_step(self) -> None:
        # Rust has a real `tsc --noEmit` failure mode of its own -- none at all.
        # A guard that ran for every track would invent a dependency the track does
        # not have, and eventually fail on a perfectly good checkout.
        ar.require_local_tsc(RUST, self.scratch)

    def test_the_comparator_and_the_candidate_both_pass_the_guard(self) -> None:
        """Both are measured in the same session and both are committed trees.

        The comparator is `implementations/typescript/`, which carries its own
        `node_modules`, and `measure_session_baseline` builds it before it builds
        the candidate. If either had lost its tree the loop would stop at the
        baseline -- correctly, with this message rather than `tsc --noEmit failed`.
        """
        for directory in (TYPESCRIPT.candidate, TYPESCRIPT.comparator):
            with self.subTest(directory=directory):
                self.assertTrue(
                    (directory / "node_modules" / ".bin" / "tsc").is_file(),
                    f"{directory} has no local compiler; run `npm ci` in it",
                )
                ar.require_local_tsc(TYPESCRIPT, directory)


class TestApplyToScratchProducesAMeasurableTree(ScratchTreeTestCase):
    def test_the_copy_is_followed_by_the_link(self) -> None:
        """`apply_to_scratch` end to end, minus the measurement.

        The two halves of the fix are one line apart and a copy without the link
        is exactly the tree that caused 61 crashes, so they are checked together:
        `node_modules` is absent from the copy and present in the result.

        The patch is generated from the file on disk rather than written out by
        hand, so this test cannot start failing because somebody edited
        `index.ts` -- and a test that fails for a reason that has nothing to do
        with what it is testing is a test that gets deleted.
        """
        source = TYPESCRIPT.candidate_file.read_text(encoding="utf-8")
        marker = "/** A scalar in the computation graph. Karpathy's `class Value`"
        self.assertIn(marker, source, "the candidate's source no longer opens this way")
        cut = source.index(marker)
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
                fromfile=f"a/{TYPESCRIPT.source_key}",
                tofile=f"b/{TYPESCRIPT.source_key}",
            )
        )
        applied, project, error = ar.apply_to_scratch(TYPESCRIPT, patch)
        self.addCleanup(ar.cleanup_scratch)
        self.assertTrue(applied, error)

        self.assertTrue((project / "node_modules" / ".bin" / "tsc").is_file())
        self.assertTrue((project / "node_modules" / ".bin" / "tsx").is_file())
        self.assertTrue((project / TYPESCRIPT.candidate_source).is_file())
        self.assertIn("TEST_NOOP", (project / TYPESCRIPT.candidate_source).read_text(encoding="utf-8"))
        # The committed candidate is untouched: a patch that fails must not be able
        # to damage the thing the next experiment starts from.
        self.assertNotIn("TEST_NOOP", source)

    def test_the_comparator_is_not_what_gets_linked(self) -> None:
        # The link points at the candidate's tree, not the frozen one's. The
        # comparator is pinned and its `node_modules` is a different install; a
        # scratch candidate measured against it would be measuring a toolchain
        # nobody proposed.
        ar.link_dependencies(TYPESCRIPT, self.scratch)
        self.assertEqual(
            Path(self.scratch / "node_modules").resolve().parent,
            TYPESCRIPT.candidate.resolve(),
        )


if __name__ == "__main__":
    unittest.main()
