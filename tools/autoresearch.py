#!/usr/bin/env python3
"""An autoresearch loop for this repository's micro-gpt tracks, on a laptop.

## What this is

Descends from [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
(MIT, (c) Karpathy): give a model a small but real training setup, let it edit one
file, measure the result, keep the edit if it helped and throw it away if it did
not, and repeat. That project's whole point is that the *protocol* is fixed and
only the code moves, so two runs are comparable and the log is a real record of
what was tried.

Three things are different here, all of them forced by having no GPU and by this
repository already being a thing that asserts its own correctness.

**It runs on three tracks, and a track is a comparator pair.** `--track` picks
Rust, Go or TypeScript. Each has its own candidate directory, its own frozen
comparator, its own finite-difference probe, and its own session baseline, and
the two axes are never compared across them: a Go steps-per-second is a ratio
against the frozen Go build and says nothing about the Rust one. `results.tsv`
carries a `track` column and `results.json` keys its bests, baselines, frontiers
and digests by track, because "the best candidate" without a track is a
comparison between two numbers about different builds.

Adding a track was four questions, not a configuration change, and the answers
are the fields of the `Track` dataclass: the comparator, the candidate, the
gradient probe, and where the loss curve comes from. That last one is the one
that is easy to get wrong. Rust and Go print every step, but TypeScript prints the
first five and then every hundredth -- sparse on purpose, so a human watching a
laptop is not shown 1,000 lines -- so its loss axis is read from the JSONL trace it
writes with `--trace`, exactly as `tools/parity.py` already does. Reading 15 points
where the statistic wants a 50-step window would have been a number about the
wrong steps, and it would have looked like a plausible one.

**The budget is a fixed 1,000 steps, not a wall-clock window.** karpathy's budget
is 5 minutes of training, which is what makes his runs comparable on his hardware
and incomparable on yours. A step count has neither problem: the loss axis becomes
*deterministic*, because a fixed seed and a fixed PRNG mean a fixed sequence of
documents through a fixed tape, so a given candidate either produces a given loss
curve or it does not. Only the speed axis stays noisy, and only the speed axis is
measured more than once.

**The baseline is re-measured every session, and the committed benchmark table is
not used as one.** `benchmarks/results.json` was taken on an Apple M1; CI runs
`ubuntu-24.04`. A speed ratio against a number from another machine is a number
about the other machine. So the first thing a session does is build the frozen
track and run it `REPEATS` times, and everything after that is compared to *that*.

**The objective is two-dimensional.** "Lower loss" is not the only axis, and
optimising it alone is how a repository ends up with a slower model it is proud
of. A candidate is kept if the loss drops materially with speed not materially
worse, or the speed rises materially with loss not materially worse. That is a
Pareto rule, and `verdict()` is where it lives.

## The part that is not negotiable: the gradient check

`docs/KNOWN-ISSUES.md` issue 1 is the reason this repository is careful, and it is
the reason the loop cannot just optimise loss. The C port's hand-written backward
pass was wrong by a factor of -8, and its loss curve *still* tracked the reference
to within 7%, and it *still* trained; it measures 1.03 now, but the argument is
the one that number's history makes. A loss number cannot tell a correct gradient
from a wrong one. Only a finite difference can.

So every candidate is put through `autoresearch/candidate/tests/gradient_check.rs`
before it is allowed to compete on loss. That probe is written here, not by the
model, and its sha256 is checked before every experiment: a probe the candidate
can weaken is not a probe. The check found something on its first run -- this
crate's own tape omits the `rmsnorm` path, so its gradient is 6.32% high. See
`docs/KNOWN-ISSUES.md` issue 5, and the module docs of the probe itself.

## What the model is and is not allowed to do

The model proposes. It does not measure, it does not decide, and its own reported
numbers are never recorded as measurements. `verdict()` is a pure function of
measured medians and named constants, and it is unit-tested on synthetic
candidates. An LLM told "keep it if the number went down" drifts toward keeping
its own bad ideas within about ten experiments, and then the log stops being
evidence of anything.

## Where the results go

Every experiment -- keep, discard or crash -- appends a row to
`autoresearch/results.tsv`, writes `autoresearch/runs/<id>.json`, and commits.
`autoresearch/results.json` is regenerated from those and is the only file the
site reads. A failure that is not published is a failure that gets retried.

Run `make autoresearch-rust`, optionally with `TRACK=go` or `TRACK=typescript`.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import parity  # noqa: E402

# ─── the protocol ────────────────────────────────────────────────────────────
#
# Every one of these is a decision, and the reason it is that decision is written
# next to it. `tools/tests/test_autoresearch.py` pins the numbers so that changing
# one is a deliberate, reviewable edit rather than a quiet loosening.

#: Training steps. Fixed, so the loss axis is deterministic and two candidates
#: are comparable. `tools/parity.py` uses the same 1,000 for the same reason.
STEPS = 1000

#: The seed. Fixed, and that is what makes the loss axis deterministic: the
#: track's PRNG is xoshiro256++ with SplitMix64 seeding, so this one number fixes
#: the parameter init, the document shuffle and the sampling stream.
SEED = 42

#: Runs per measurement, and the median is reported rather than the mean. Wall
#: clock on a machine that is also running a browser is heavy-tailed, and one
#: background `cargo` should move the result by less than the effect being
#: measured. A mean would let it.
REPEATS = 3

#: A run that takes longer than this is killed and recorded as a crash, the same
#: judgement karpathy's `program.md` makes at 10 minutes. Generous, because this
#: is 1,000 steps of a 4,192-parameter model and should take well under a minute.
RUN_TIMEOUT_SECONDS = 600

#: How much the windowed loss must fall for the loss axis to count as improved.
#:
#: The reference's per-step loss rises on 500 of its 999 steps, and its per-step
#: standard deviation is 0.392 against a mean of 2.4517 -- it is 16% noise
#: (`tools/parity.py` docstring, measured). Over a 50-step window that shrinks to
#: a standard error near 0.055, about 2.2% of the mean. So 2% is roughly one
#: standard error, and that is the honest floor for "this went somewhere".
LOSS_MATERIAL = 0.02

#: How much the windowed loss may rise before it counts as a regression. Two
#: standard errors, which is where a difference stops being noise and starts
#: being the thing you are trying not to do.
LOSS_TOL = 0.05

#: How much faster the candidate must be for the speed axis to count as improved.
#: Much larger than the loss thresholds, and deliberately so: wall-clock varies
#: with what else the machine is doing, and pretending otherwise would make every
#: run a coin flip. A 10% speedup is unambiguous; a 2% one is not.
SPEED_MATERIAL = 0.10

#: How much slower the candidate may be before it counts as a regression.
SPEED_TOL = 0.05

#: Mirrors `tools/parity.py::MIN_RELATIVE_IMPROVEMENT`, and for the same reason:
#: a candidate that does not learn at all must not be able to win on speed. The
#: C port's gradient used to be wrong by a factor of -8 and still achieved 17.6%
#: windowed improvement, so this catches divergence but is nowhere near
#: sufficient on its own. The gradient check is what catches a broken tape.
MIN_RELATIVE_IMPROVEMENT = parity.MIN_RELATIVE_IMPROVEMENT

#: The window the loss axis is read over. `tools/parity.py`'s trend gate uses 50
#: and the reference's own improvement is 18.5%, so this is a number the rest of
#: the repository already reasons about.
TREND_WINDOW = parity.TREND_WINDOW

#: `--verify-only` re-runs the loss axis on whatever machine CI is on. Bit-exact
#: agreement is not claimed and could not be: `f32` contraction under a different
#: LLVM and a different target CPU puts the two runs in the same place to about
#: the third significant figure, and `docs/BENCHMARKS.md` already says so. This
#: is the band inside which a recorded loss is considered reproduced, and it is
#: `LOSS_TOL` because a re-run should satisfy the same bar as a candidate.
VERIFY_LOSS_TOLERANCE = LOSS_TOL

#: The speed axis is deliberately not verified. It was measured on the machine
#: that ran the loop, and CI is not that machine. The constant exists so that
#: `--verify-only` can say that out loud instead of quietly omitting a check.
SPEED_AXIS_VERIFIABLE = False

VERDICTS = ("keep", "discard", "crash")

# ─── layout ──────────────────────────────────────────────────────────────────

RESEARCH_DIR = REPO_ROOT / "autoresearch"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
RESULTS_JSON = RESEARCH_DIR / "results.json"
BASELINE_JSON = RESEARCH_DIR / "baseline.json"
MODEL_JSON = RESEARCH_DIR / "model.json"
PROGRAM_MD = RESEARCH_DIR / "program.md"
RUNS_DIR = RESEARCH_DIR / "runs"
DIFFS_DIR = RESEARCH_DIR / "diffs"
LOGS_DIR = RESEARCH_DIR / "logs"
DATASET = REPO_ROOT / "data" / "input.txt"

#: The two Rust binary names. Not one constant because they are genuinely two
#: programs: the frozen track's `Cargo.toml` declares `[[bin]] name = "microgpt-rs"`,
#: and the candidate crate was renamed `microgpt` with a `microgpt-tuned` bin when
#: the workspace was seeded. `build_track` picks by which directory it was given.
CANDIDATE_BIN = "microgpt-tuned"
FROZEN_BIN = "microgpt-rs"

# ─── tracks ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Track:
    """Everything that is wired to one language by name.

    The four things `--track` has to answer for, from the module docstring: the
    comparator, the candidate, the gradient probe, and the stdout contract. They
    are fields here rather than module constants so that adding a language is a
    new entry in `TRACKS` and not an edit that reaches through the whole file.

    `run` is a command *list*, not a path, because only Rust produces a binary
    you can exec directly. Go and TypeScript are `go run` and `tsx` invocations,
    so their wall clock includes an interpreter start. That is not a problem for
    the comparison, and the reason is the argument `train_once` already makes: the
    baseline is measured the same way, in the same session, on the same machine,
    so the overhead cancels. The absolute number is not a pure training time for
    any track, and the run record says so rather than implying otherwise.
    """

    name: str
    language: str
    #: Under `autoresearch/`, and what the model is told it may edit.
    candidate_dir: str
    candidate_source: str
    #: The frozen track under `implementations/`, used as the comparator.
    comparator_dir: str
    comparator_source: str
    #: The finite-difference probe, relative to `candidate_dir`, and its digest.
    probe_path: str
    probe_sha256: str
    #: The build tool, and how to find it.
    tool: str
    tool_hint: str
    #: Fenced-block language for the source shown to the model, and the comment
    #: syntax its diff hunks use, which the model has to copy correctly.
    fence: str
    build_hint: str
    #: Where the loss curve comes from: the track's own stdout, or the JSONL trace
    #: it writes with `--trace`.
    #:
    #: This is not a preference. The loss axis is the mean over the last 50 steps,
    #: and TypeScript prints only the first 5 steps and then every 100th -- sparse
    #: on purpose, because a human watching a laptop does not want 1000 lines. So
    #: for that track the stdout curve has 15 points where the statistic wants 50,
    #: and a window taken from it would be a number about the wrong steps. The
    #: trace has all 1000 at full precision, which is why `tools/parity.py` reads
    #: TypeScript's curve from there too, and why the two must agree: one
    #: definition of the loss axis, used by both.
    loss_parse: str
    #: What `autoresearch/program.md` says about this port's build and its
    #: dependency rule. Per track because the sentence is not language-neutral: a
    #: Go module with no `require` block and a Go package.json with no
    #: dependencies are the same commitment written two ways, and a brief that says
    #: "no Cargo.toml dependencies" to a model holding Go source is a brief about a
    #: file that does not exist.
    dependency_note: str
    lint_command: str
    #: The bullet describing what the harness and the probe read out of this track's
    #: source, and the anecdote about a patch that did not apply.
    #:
    #: Both exist because they are the two things a model gets wrong most often,
    #: and both were originally written once, for Rust, in `program.md` -- so Go
    #: and TypeScript were being told to preserve a `pub fn run()` and to look for
    #: `fn rmsnorm(x: &[TensorHandle]) -> Vec<TensorHandle>`, neither of which
    #: exists in their source. `TensorHandle` was not in the `foreign` table
    #: because that table holds build vocabulary, not type names, so the test that
    #: was supposed to catch this passed for two extra tracks. Per track, because
    #: the honest sentence for C -- that its probe `#include`s the file and reads
    #: its statics -- is a different sentence from the honest one for Rust.
    probe_api: str
    patch_caution: str

    @property
    def candidate(self) -> Path:
        return RESEARCH_DIR / self.candidate_dir

    @property
    def candidate_file(self) -> Path:
        return self.candidate / self.candidate_source

    @property
    def probe(self) -> Path:
        return self.candidate / self.probe_path

    @property
    def comparator(self) -> Path:
        return REPO_ROOT / "implementations" / self.comparator_dir

    @property
    def comparator_file(self) -> Path:
        return self.comparator / self.comparator_source

    @property
    def source_key(self) -> str:
        """The key a run record files this track's digest under."""
        return f"autoresearch/{self.candidate_dir}/{self.candidate_source}"

    @property
    def best_diff(self) -> Path:
        """Where the baseline-to-candidate patch for this track is written.

        Rust keeps the name it has always had: the file is committed, the site
        renders it, and four run records on disk refer to it. The others take a
        suffix rather than a subdirectory, so all of them sit together in
        `autoresearch/diffs/` where a reader expects to find them.
        """
        if self.name == "rust":
            return BEST_DIFF
        return DIFFS_DIR / f"best-vs-baseline-{self.name}.patch"

    @property
    def baseline_json(self) -> Path:
        """Where this track's session baseline is recorded.

        One file per track for the same reason `best_diff` is: a baseline is a
        claim about one comparator measured on one machine, and averaging three
        of them into one number would be a number about nothing.
        """
        if self.name == "rust":
            return BASELINE_JSON
        return RESEARCH_DIR / f"baseline-{self.name}.json"


#: The Rust track is the original and stays the default. Its candidate directory
#: keeps the name it has always had rather than moving under a `candidates/`
#: directory, because four committed run records already file their source
#: digest under `autoresearch/candidate/src/lib.rs` and `verify` compares against
#: that key: moving the tree would invalidate every experiment on record to
#: accommodate a tidier layout.
TRACKS: dict[str, Track] = {
    "rust": Track(
        name="rust",
        language="Rust",
        candidate_dir="candidate",
        candidate_source="src/lib.rs",
        comparator_dir="rust",
        comparator_source="src/lib.rs",
        probe_path="tests/gradient_check.rs",
        probe_sha256="b5beae0d11579987c954867846127d5119d4cb07ed2ea47078cf1c2e29807c6b",
        dependency_note=(
            "`Cargo.toml` has none and must keep none. The reference's whole point "
            "is that the algorithm fits in one file with nothing but a standard "
            "library, and a port that reached for a crate would be measuring the crate."
        ),
        lint_command="`cargo clippy --release -- -D warnings`",
        probe_api=(
            "* `pub fn run()`, and the `[[bin]]` that calls it;\n"
            "* the public items the probe uses: `Config` and its fields, `Model::{new,\n"
            "  forward, params}`, `Rng::new`, `TensorHandle`, `Tensor::{leaf, data, grad,\n"
            "  backward, set_data, add, mul, neg, log, exp, div, sub_scalar, div_scalar}`,\n"
            "  and the `pub` visibility of the crate root. In particular `Tensor::set_data`\n"
            "  is the only thing separating the probe from a private arena, and a\n"
            "  rewrite that drops it fails the probe."
        ),
        patch_caution=(
            "Two of the first attempts at this task failed for the same reason: a patch\n"
            "  written against an `rmsnorm` that took a weight tensor and had `.data()`\n"
            "  and `.mul()` methods. Neither exists.\n"
            "  `fn rmsnorm(x: &[TensorHandle]) -> Vec<TensorHandle>` does, and you can read it"
        ),
        tool="cargo",
        tool_hint="Install Rust from https://rustup.rs",
        fence="rust",
        build_hint="`cargo build --release` and `cargo clippy --release -- -D warnings`",
        loss_parse="stdout",
    ),
    "go": Track(
        name="go",
        language="Go",
        candidate_dir="candidate-go",
        candidate_source="main.go",
        comparator_dir="go",
        comparator_source="main.go",
        probe_path="probe_test.go",
        probe_sha256="d64b85b9006cbc36d67c57a29ae165969fd08456d5bb1e4f14ff4b5e3580dcc8",
        dependency_note=(
            "`go.mod` has no `require` block and must keep none. The reference's "
            "whole point is that the algorithm fits in one file with nothing but a "
            "standard library, and a port that reached for a package would be "
            "measuring the package."
        ),
        lint_command="`go vet ./...`",
        probe_api=(
            "* a `main` package whose `Train`, `Config` and `Rng` are the ones the "
            "probe calls — the probe is an in-package test file, so it reaches the "
            "unexported names directly and they must keep those names"
        ),
        patch_caution=(
            "The first attempt at this task failed because it patched a function "
            "that does not exist under that name. Read the file before writing the "
            "hunk"
        ),
        tool="go",
        tool_hint="Install Go from https://go.dev/dl/",
        fence="go",
        build_hint="`go build` and `go vet ./...`",
        loss_parse="stdout",
    ),
    "typescript": Track(
        name="typescript",
        language="TypeScript",
        candidate_dir="candidate-ts",
        candidate_source="src/index.ts",
        comparator_dir="typescript",
        comparator_source="src/index.ts",
        probe_path="src/probe.ts",
        probe_sha256="5c16a817ee3e391d29d0aefc1e1f67b008ad028be70a2521db582197e1084441",
        dependency_note=(
            "`package.json` has only `typescript`, `tsx` and `@types/node`, all "
            "`devDependencies`, and the source must not reach for a "
            "`dependencies` entry. The reference's whole point is that the algorithm "
            "fits in one file with nothing but a standard library."
        ),
        lint_command="`npx tsc --noEmit`",
        probe_api=(
            "* the `export`s the probe imports: `Config`, `Tensor`, `TensorHandle` and "
            "`Model`. `setData` is the only thing separating the probe from a private "
            "arena, and a rewrite that drops it fails the probe"
        ),
        patch_caution=(
            "The first attempt at this task failed because it patched an `rmsnorm` "
            "that took a weight tensor. The one in this file takes a `TensorHandle`"
        ),
        tool="npm",
        tool_hint="Install Node from https://nodejs.org/",
        fence="typescript",
        build_hint="`npx tsc --noEmit`",
        loss_parse="trace",
    ),
    "c": Track(
        name="c",
        language="C",
        candidate_dir="candidate-c",
        candidate_source="microgpt.c",
        comparator_dir="c",
        comparator_source="microgpt.c",
        probe_path="probe.c",
        probe_sha256="4c6e6572f83913e8a2f4d824087228253dab6122df3e504b1158c05271df59bc",
        dependency_note=(
            "The link line is `-lm` and nothing else. `microgpt_simd.h` supplies "
            "the 19 NEON intrinsics and 3 BLAS calls as real ones on aarch64 and "
            "as small portable fallbacks everywhere else, which is what lets this "
            "track build in CI on Linux. A candidate must not add a BLAS binding: "
            "the reference's whole point is that the algorithm fits in one file, "
            "and a port that reached for a library would be measuring the library."
        ),
        lint_command="`cc -fsyntax-only -Wall -Wextra`",
        probe_api=(
            "* a `main(int argc, char **argv)` that accepts `--input`, `--steps` and "
            "`--seed`;\n"
            "* the names the probe reads. The probe `#include`s this file, so it uses "
            "the statics directly: `wte`, `wpe`, `lm_head`, `attn_wq`, `attn_wk`, "
            "`attn_wv`, `attn_wo`, `mlp_fc1`, `mlp_fc2`, their `g_`-prefixed "
            "gradient twins, `forward_pos`, `backward_all`, `zero_gradients`, "
            "`seed_rng`, `load_data`, `init_weights`, `char_to_idx`, `BOS_TOKEN`, "
            "`vocab_size`, `saved_probs`, and the "
            "`N_EMBD`/`N_HEAD`/`N_LAYER`/`BLOCK_SIZE`/`MLP_DIM` macros.\n"
            "* the *order* of the gradient arrays, because the probe finds them by a "
            "positional walk: `wte`, `wpe`, `lm_head`, then per layer `wq`, `wk`, "
            "`wv`, `wo`, `fc1`, `fc2`. Moving a declaration is safe. Reordering those "
            "blocks is not — the probe will read a gradient out of the wrong array, "
            "and the resulting ratio will be plausible and wrong."
        ),
        patch_caution=(
            "Read the whole file before writing the hunk. This port has no autograd "
            "tape, so `backward_all` is a hand-written chain rule over the saved "
            "activations, and the natural place to add a term is very often also a "
            "place that has to be added to `zero_gradients`"
        ),
        tool="cc",
        tool_hint="Install a C compiler — on macOS the Xcode command line tools (`xcode-select --install`), on Debian `build-essential`",
        fence="c",
        build_hint="`cc -O3 -o microgpt microgpt.c -lm`",
        loss_parse="stdout",
    ),
}

#: `--track` takes one of these and rejects anything else loudly, rather than
#: accepting a name it cannot honour.
SUPPORTED_TRACKS = tuple(TRACKS)


def get_track(name: str) -> Track:
    try:
        return TRACKS[name]
    except KeyError:
        raise ResearchError(
            f"error: --track {name!r} is not supported. This tool knows about "
            f"{', '.join(SUPPORTED_TRACKS)}.\n"
            f"  Adding a track is not a configuration change: the comparator, the "
            f"candidate, the gradient probe and the stdout contract are each wired "
            f"to one language, and all four have to be answered for another. See "
            f"the `Track` dataclass in tools/autoresearch.py."
        ) from None


#: The Rust track's digest, asserted by `tools/tests/test_autoresearch.py` so it
#: cannot drift from the file quietly. Kept as a module name because the test
#: refers to it.
PROBE_SHA256 = TRACKS["rust"].probe_sha256
CANDIDATE_DIR = TRACKS["rust"].candidate
CANDIDATE_LIB = TRACKS["rust"].candidate_file
PROBE_PATH = TRACKS["rust"].probe
BASELINE_CRATE = TRACKS["rust"].comparator
BASELINE_LIB = TRACKS["rust"].comparator_file
BEST_DIFF = DIFFS_DIR / "best-vs-baseline.patch"

#: The ledger, as a TSV. Tab-separated because karpathy's is, and because
#: commas appear in these descriptions ("beta1 0.85 -> 0.9, no bias correction").
#: `PARSE` order is the column order and both readers derive from it.
COLUMNS = (
    "run_id",
    "parent",
    "track",
    "loss",
    "steps_per_sec",
    "loss_gain",
    "speed_gain",
    "grad_ratio",
    "status",
    "reason",
    "description",
)
HEADER = "\t".join(COLUMNS)


# ─── small helpers ───────────────────────────────────────────────────────────


class ResearchError(RuntimeError):
    """Anything the operator needs to read and fix. Never swallowed."""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Deterministic on purpose.

    `results.json` is regenerated in CI and diffed, so a key order that varied
    between runs would report drift every single time -- and a check that always
    fails is one everyone learns to ignore. The same argument as
    `tools/parity.py` gives for comparing traces with a comparator rather than
    `git diff`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def find_tool(name: str) -> str | None:
    """Locate a tool, including the ones a login shell would have but this
    process does not.

    `cargo` is the case that matters. It installs to `~/.cargo/bin`, which is on
    an interactive shell's PATH via `~/.zshenv` or `~/.profile` and is *not* on
    the PATH a subprocess inherits from some launchers, and from GitHub Actions
    it is not either until `actions-rust-lang/setup-rust-toolchain` runs. A tool
    that is installed and reported missing is worse than one that is absent, so
    the well-known install locations are checked too.
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in (
        Path.home() / ".cargo" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ):
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def require_tool(name: str, install: str) -> str:
    found = find_tool(name)
    if not found:
        raise ResearchError(
            f"error: '{name}' not found. {install}\n"
            f"  Looked on PATH and in ~/.cargo/bin, /opt/homebrew/bin and "
            f"/usr/local/bin."
        )
    return found


def run_command(
    argv: list[str],
    cwd: Path,
    timeout: int = 900,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


# ─── the ledger ──────────────────────────────────────────────────────────────


@dataclass
class Row:
    run_id: str
    parent: str
    track: str
    loss: float
    steps_per_sec: float
    loss_gain: float
    speed_gain: float
    grad_ratio: float
    status: str
    reason: str
    description: str

    def fields(self) -> list[str]:
        def number(value: float) -> str:
            return f"{value:.6f}"

        return [
            self.run_id,
            self.parent,
            self.track,
            number(self.loss),
            number(self.steps_per_sec),
            f"{self.loss_gain:+.6f}",
            f"{self.speed_gain:+.6f}",
            number(self.grad_ratio),
            self.status,
            self.reason,
            self.description,
        ]

    def to_json(self) -> dict[str, Any]:
        payload = dict(zip(COLUMNS, self.fields()))
        payload["loss"] = round(self.loss, 6)
        payload["steps_per_sec"] = round(self.steps_per_sec, 6)
        payload["loss_gain"] = round(self.loss_gain, 6)
        payload["speed_gain"] = round(self.speed_gain, 6)
        payload["grad_ratio"] = round(self.grad_ratio, 6)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Row":
        return cls(
            run_id=str(payload["run_id"]),
            parent=str(payload["parent"]),
            # Rows written before the loop learned about Go and TypeScript have no
            # `track` cell, and they are all Rust: `autoresearch/candidate/` was
            # the only candidate that existed. Defaulting is the migration.
            track=str(payload.get("track") or "rust"),
            loss=float(payload["loss"]),
            steps_per_sec=float(payload["steps_per_sec"]),
            loss_gain=float(payload["loss_gain"]),
            speed_gain=float(payload["speed_gain"]),
            grad_ratio=float(payload["grad_ratio"]),
            status=str(payload["status"]),
            reason=str(payload["reason"]),
            description=str(payload["description"]),
        )


def clean_field(value: str, name: str) -> str:
    """A TSV cell must not contain a tab or a newline, or the ledger is two
    tables wearing a trench coat. Refused here rather than escaped, because a
    description with a stray tab should be a loud failure, not a corrupted row
    that a later reader cannot see is corrupted."""
    if "\t" in value or "\n" in value or "\r" in value:
        raise ResearchError(
            f"error: the {name!r} field contains a tab or a newline: {value!r}. "
            f"The ledger is tab-separated; write it as a sentence."
        )
    return value


def read_ledger() -> list[Row]:
    if not RESULTS_TSV.is_file():
        return []
    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        raise ResearchError(
            f"error: {RESULTS_TSV.relative_to(REPO_ROOT)} has a header this tool "
            f"does not recognise.\n  expected: {HEADER}\n  found:    "
            f"{lines[0] if lines else '(empty file)'}\n"
            f"  The ledger is append-only. Do not rewrite its columns; regenerate "
            f"it with `python3 -m tools.autoresearch --render` if you think the "
            f"schema should change, and expect a test to have an opinion."
        )
    rows: list[Row] = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) != len(COLUMNS):
            raise ResearchError(
                f"error: {RESULTS_TSV.relative_to(REPO_ROOT)} line {number} has "
                f"{len(cells)} tab-separated cells, expected {len(COLUMNS)}: {line!r}"
            )
        record = dict(zip(COLUMNS, cells))
        try:
            rows.append(
                Row(
                    run_id=record["run_id"],
                    parent=record["parent"],
                    track=record["track"],
                    loss=float(record["loss"]),
                    steps_per_sec=float(record["steps_per_sec"]),
                    loss_gain=float(record["loss_gain"]),
                    speed_gain=float(record["speed_gain"]),
                    grad_ratio=float(record["grad_ratio"]),
                    status=record["status"],
                    reason=record["reason"],
                    description=record["description"],
                )
            )
        except ValueError as exc:
            raise ResearchError(
                f"error: {RESULTS_TSV.relative_to(REPO_ROOT)} line {number} has a "
                f"number that is not a number: {exc}"
            ) from exc
    return rows


def append_ledger(row: Row) -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.is_file():
        RESULTS_TSV.write_text(HEADER + "\n", encoding="utf-8")
    fields = [clean_field(cell, name) for cell, name in zip(row.fields(), COLUMNS)]
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")


def next_run_id(rows: list[Row]) -> str:
    return f"{len(rows):04d}"


# ─── measurement ─────────────────────────────────────────────────────────────


GRADCHECK_RE = re.compile(
    r"GRADCHECK\s+analytic=(?P<analytic>[-0-9.eE+]+)\s+params=(?P<params>\d+)\s+ratios=(?P<ratios>.+)"
)
RATIO_RE = re.compile(r"h=(?P<h>[-0-9.eE+]+)\s+ratio=(?P<ratio>[-0-9.eE+]+)")
REPORTED_SPEED_RE = re.compile(
    r"Total time:\s*(?P<ms>[-0-9.eE+]+)\s*ms\s*\((?P<sps>[-0-9.eE+]+)\s*steps/sec\)"
)


@dataclass
class Measurement:
    """One candidate's numbers. Everything the verdict is allowed to see."""

    losses: list[float]
    window: dict[str, float]
    steps_per_sec: float
    wall_seconds: list[float]
    reported_steps_per_sec: float | None
    grad_ratio: float
    grad_ratios: dict[str, float]
    grad_analytic: float
    grad_params: int
    log: str


def windowed(losses: list[float]) -> dict[str, float]:
    """The loss axis: the mean over the last `TREND_WINDOW` steps, and how far it
    fell from the first window. `tools/parity.py` already defines this and its
    `TrackRun.trend()` is the reference implementation; reimplementing it would
    be a second definition of the same statistic, which is how two numbers stop
    agreeing and nobody can say which is right."""
    track = parity.TrackRun("candidate", "micro", losses)
    return track.trend()


def parse_losses(stdout: str, limit: int) -> list[float]:
    losses = parity.parse_losses(stdout, limit)
    if not losses:
        return []
    return losses


def train_once(
    track: Track, command: list[str], steps: int, seed: int, cwd: Path
) -> tuple[list[float], float, float | None]:
    """One timed training run.

    `command` is the track's entry point (`Track.run`), not a path: Go and
    TypeScript are run through their toolchain rather than exec'd from disk. The
    wall clock is the whole process, which for those two includes an interpreter
    start. That is deliberate and it is safe: the baseline is measured the same
    way, in the same session, on the same machine, so the comparison is fair even
    though the absolute number is not a pure training time. It is recorded raw in
    the run record so a reader can check the overhead rather than take it on
    trust.

    The trace file, when a track needs one, is written inside a fresh temporary
    directory per run rather than a fixed name: the speed axis runs the same
    binary three times, and a shared path would have the second run measure the
    first one's file.
    """
    command = list(command)
    trace_dir: Path | None = None
    if track.loss_parse == "trace":
        trace_dir = Path(tempfile.mkdtemp(prefix="autoresearch-trace-"))
        _SCRATCH_ROOTS.append(trace_dir)
        command += ["--trace", str(trace_dir / "track.jsonl")]

    started = time.monotonic()
    completed = run_command(
        command + ["--input", str(DATASET), "--steps", str(steps), "--seed", str(seed)],
        cwd=cwd,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-2500:]
        raise ResearchError(
            f"training run failed (exit {completed.returncode})\n{tail}"
        )
    if track.loss_parse == "trace":
        trace_path = trace_dir / "track.jsonl"
        if not trace_path.is_file():
            raise ResearchError(
                f"the {track.name} run wrote no trace at {trace_path}. The loss axis "
                f"is a mean over the last {TREND_WINDOW} steps and this track's "
                f"stdout is deliberately sparse -- see `Track.loss_parse` -- so the "
                f"trace is the only complete record of the curve."
            )
        losses = [
            float(row["loss"])
            for row in parity.read_trace(trace_path)
            if row.get("type") == "step" and "loss" in row
        ]
        if len(losses) < TREND_WINDOW:
            raise ResearchError(
                f"the {track.name} trace has {len(losses)} step records, fewer than "
                f"the {TREND_WINDOW}-step window the loss axis is defined over. A "
                f"window taken from fewer points would be a number about the wrong "
                f"steps."
            )
    else:
        losses = parse_losses(completed.stdout, steps)
        if not losses:
            raise ResearchError(
                "training run produced no `step N / M | loss X` lines. The harness "
                "reads that line, and so does tools/parity.py, which is why all five "
                "tracks print it. Whatever replaced it, put it back:\n"
                f"{(completed.stdout or '')[-1500:]}"
            )
        # The mirror of the trace branch's count check, and it is here because of
        # what it catches.
        #
        # The TypeScript track's `loss_parse` was once "stdout" when it should have
        # been "trace". Nothing threw. That track prints the first 5 steps and then
        # every hundredth -- sparse on purpose, so a human watching a laptop does not
        # sit through 1,000 lines -- so the parse succeeded, produced 15 points, and
        # the loss axis was quietly computed as the mean of those 15. It recorded
        # 2.8047 where the real last-50 window is 2.3091, a 21% error, and it did so
        # in a way that still passed every check: the ledger was internally
        # consistent, the baseline and the candidate agreed with each other, and
        # `verify` re-measured the same wrong number and found it unchanged.
        #
        # A declared parse mode is a claim about a program this harness does not
        # control, and a claim that silently degrades is worse than a wrong one that
        # throws. Fewer points than the window needs is not a measurement.
        if len(losses) < TREND_WINDOW:
            raise ResearchError(
                f"the {track.name} track declares loss_parse='stdout' and its stdout "
                f"yielded {len(losses)} `step N / M | loss X` lines, fewer than the "
                f"{TREND_WINDOW}-step window the loss axis is defined over. Either the "
                f"track has started printing sparsely -- in which case its "
                f"`loss_parse` should be 'trace' and it should write one -- or it has "
                f"stopped printing the per-step line altogether. A window taken from "
                f"fewer points than the window is wide is a number about the wrong "
                f"steps, and it is worse than no number because it looks like one."
            )
    reported = REPORTED_SPEED_RE.search(completed.stdout)
    reported_sps = float(reported.group("sps")) if reported else None
    return losses, elapsed, reported_sps


def require_local_tsc(track: Track, project_dir: Path) -> None:
    """Preflight on the npm path: can this project actually run its own compiler?

    `npx` does not ask the project whether it has a compiler before it runs one.
    With no `node_modules` to resolve `tsc` from, it falls back to the registry, and
    the package published there under the name `tsc` is a deprecated stub whose whole
    output is "This is not the tsc command you are looking for" followed by exit 1.
    `build_track` then raised `tsc --noEmit failed` -- the same message, for the same
    reason, on every candidate, correct or not. Sixty-one consecutive TypeScript
    experiments (runs 0171-0231) were recorded that way and not one of them was the
    model's fault.

    So the compiler is resolved *here*, before anything is compiled, and a harness
    that is not set up says so. The two failures have to be told apart: `tsc --noEmit
    failed` is the model's problem and this is the operator's, and a reader of
    `results.tsv` has one `reason` column and no other way to know which one a row
    was.

    A `stat`, not a subprocess. `npm ls typescript` would spend a node startup on the
    one path that has to be cheap, and it would answer a broader question than the
    one being asked -- whether *this* tree has a compiler it can run -- by a
    round-trip that has its own failure modes.
    """
    if track.tool != "npm":
        return
    tsc = project_dir / "node_modules" / ".bin" / "tsc"
    if tsc.exists():
        return
    raise ResearchError(
        f"error: no local TypeScript compiler in {project_dir}: {tsc} is missing."
        f" Run `npm ci` in {track.candidate.relative_to(REPO_ROOT)} and commit the"
        f" result, or run `npm ci` in {project_dir} itself.\n"
        "  `npx tsc --noEmit` does not check whether this project has a compiler. With"
        " nothing local to resolve `tsc` from it fetches the npm package of that name,"
        " which is a deprecated stub that prints \"This is not the tsc command you are"
        " looking for\" and exits 1. That is this message. `tsc --noEmit failed` is a"
        " different message and means the candidate's own types are wrong; a run that"
        " reports this one has measured nothing, and must not be recorded as evidence"
        " about the candidate."
    )


def build_track(tool: str, track: Track, project_dir: Path) -> list[str]:
    """Compile the candidate or the comparator, and say how to run it.

    Returns the command rather than leaving it to a constant, because the entry
    point is not a property of the *language*: the frozen Rust track's `[[bin]]`
    is `microgpt-rs` and the candidate crate's is `microgpt-tuned`, and the
    comparator is a read-only, pinned directory that a build must not write a
    binary into.
    """
    if track.tool == "cargo":
        result = run_command(
            [tool, "build", "--release", "--quiet", "--manifest-path", str(project_dir / "Cargo.toml")],
            cwd=project_dir,
        )
        if result.returncode != 0:
            raise ResearchError(f"cargo build failed\n{(result.stderr or '')[-3000:]}")
        name = CANDIDATE_BIN if project_dir == track.candidate else FROZEN_BIN
        binary = project_dir / "target" / "release" / name
        if not binary.is_file():
            raise ResearchError(f"error: {binary} was not produced by the build")
        return [str(binary)]

    if track.tool == "go":
        # Into a temporary directory, never into the project. The comparator is
        # the frozen track: dropping a 2.5 MB binary into a pinned directory is a
        # change to it, and `go build ./...` in a module whose directory is `go`
        # writes a file called `go`.
        out = Path(tempfile.mkdtemp(prefix="autoresearch-go-"))
        _SCRATCH_ROOTS.append(out)
        result = run_command(
            [tool, "build", "-o", str(out / "microgpt-go"), "."], cwd=project_dir
        )
        if result.returncode != 0:
            raise ResearchError(f"go build failed\n{(result.stderr or '')[-3000:]}")
        return [str(out / "microgpt-go")]

    if track.tool == "cc":
        # Also into a temporary directory, for the same reason as Go: the
        # comparator is a pinned directory and a binary dropped into it is a
        # change to the frozen track. `-O3` and `-lm` are the whole link line.
        out = Path(tempfile.mkdtemp(prefix="autoresearch-c-"))
        _SCRATCH_ROOTS.append(out)
        binary = out / "microgpt-c"
        result = run_command(
            [tool, "-O3", "-o", str(binary), track.candidate_source, "-lm"],
            cwd=project_dir,
        )
        if result.returncode != 0:
            raise ResearchError(f"cc failed\n{(result.stderr or '')[-3000:]}")
        return [str(binary)]

    require_local_tsc(track, project_dir)
    result = run_command(["npx", "tsc", "--noEmit"], cwd=project_dir)
    if result.returncode != 0:
        raise ResearchError(f"tsc --noEmit failed\n{(result.stderr or '')[-3000:]}")
    return ["npx", "tsx", "src/cli.ts"]


def lint_track(tool: str, track: Track, project_dir: Path) -> None:
    """The bar a candidate has to clear before it is allowed to compete.

    Deliberately the same bar as the frozen track: a model rewrites this without
    supervision, so "it compiles and it is lint-clean" is the only thing between
    a good patch and a published one.
    """
    if track.tool == "cargo":
        result = run_command(
            [tool, "clippy", "--release", "--quiet", "--manifest-path", str(project_dir / "Cargo.toml"),
             "--", "-D", "warnings"],
            cwd=project_dir,
        )
        failed = "cargo clippy failed with -D warnings"
    elif track.tool == "go":
        result = run_command([tool, "vet", "./..."], cwd=project_dir)
        failed = "go vet failed"
    elif track.tool == "cc":
        # Warnings are not errors here, and deliberately: `-Werror` on a
        # candidate would reject a patch for an unused variable that the compiler
        # can see is harmless, and the other three tracks all lint at a
        # warning-is-error bar while C cannot without becoming a different
        # language. What this does buy is the one thing that matters for a
        # hand-written port: a syntax error is caught before the timing run, by a
        # path that compiles nothing and writes no binary.
        result = run_command(
            [tool, "-fsyntax-only", "-Wall", "-Wextra", "-Wno-unused-parameter",
             "-Wno-unused-function", track.candidate_source],
            cwd=project_dir,
        )
        failed = "cc -fsyntax-only failed"
    else:
        # Lint is a second, independent path to the same compiler, so the preflight
        # runs again here rather than relying on `build_track` having been called
        # first. The two are called one after the other in `measure_track` today;
        # "today" is not a reason for a check to be conditional on it.
        require_local_tsc(track, project_dir)
        result = run_command(["npx", "tsc", "--noEmit"], cwd=project_dir)
        failed = "tsc --noEmit failed"
    if result.returncode != 0:
        raise ResearchError(
            f"{failed}\n{(result.stderr or '')[-3000:]}"
        )


def run_gradient_probe(tool: str, track: Track, project_dir: Path) -> dict[str, Any]:
    """The finite-difference gate. See the module docstring for why it exists and
    `autoresearch/candidate/tests/gradient_check.rs` for what it measures. The
    Go and TypeScript probes measure the same thing and print the same line.

    The digest is checked against the *committed* probe, not the one in
    `project_dir`. A candidate could otherwise ship its own weakened probe
    alongside its own weakened gradient, and the two would agree with each other.
    """
    actual = sha256_file(track.probe)
    if actual != track.probe_sha256:
        raise ResearchError(
            f"error: the gradient probe has been modified.\n"
            f"  expected sha256 {track.probe_sha256}\n"
            f"  found    sha256 {actual}\n"
            f"  {track.probe.relative_to(REPO_ROOT)} is written by this repository, not "
            f"by the research loop, and it is the only thing standing between a "
            f"model and a broken gradient that still trains "
            f"(docs/KNOWN-ISSUES.md issue 1). If you changed it on purpose, update "
            f"the {track.probe_path} entry in the `Track` table in "
            f"tools/autoresearch.py and say why in the commit."
        )
    if track.tool == "cargo":
        probe_cmd = [tool, "test", "--release", "--test", "gradient_check", "--", "--nocapture"]
    elif track.tool == "go":
        probe_cmd = [tool, "test", "-run", "TestGradcheck", "-v", "."]
    elif track.tool == "cc":
        # The C probe `#include`s the candidate's own source, so compiling it
        # measures the *candidate's* gradient -- which is the point, since the
        # candidate is what a model rewrites. It is built into a temporary
        # directory for the same reason the track's own binary is: the
        # comparator is pinned.
        out = Path(tempfile.mkdtemp(prefix="autoresearch-c-probe-"))
        _SCRATCH_ROOTS.append(out)
        binary = out / "probe-c"
        build = run_command(
            [tool, "-O2", "-Wno-unused-function", "-o", str(binary),
             track.probe_path, "-lm"],
            cwd=project_dir,
        )
        if build.returncode != 0:
            raise ResearchError(
                f"the gradient probe did not build\n{(build.stderr or '')[-3000:]}"
            )
        # `--input` rather than a relative "input.txt": the C port was the one
        # track that used to resolve its dataset from the current working
        # directory, and docs/ADDING-A-LANGUAGE.md requirement 4 exists because
        # of it. The probe is held to the same rule.
        probe_cmd = [str(binary), "--input", str(DATASET)]
    else:
        probe_cmd = ["npx", "tsx", track.probe_path]
    result = run_command(probe_cmd, cwd=project_dir)
    output = f"{result.stdout}\n{result.stderr}"
    match = GRADCHECK_RE.search(output)
    if result.returncode != 0 or not match:
        raise ResearchError(
            f"the gradient check did not pass\n{(result.stderr or result.stdout)[-3000:]}"
        )
    ratios = {
        item.group("h"): float(item.group("ratio")) for item in RATIO_RE.finditer(match.group("ratios"))
    }
    if not ratios:
        raise ResearchError(f"the gradient check printed no ratios: {match.group(0)!r}")
    # The harness reads the ratio at the coarsest step size, which is the one
    # least exposed to f32 roundoff in the difference quotient. The probe has
    # already asserted that the ratios agree; picking the coarse one here keeps
    # the number it records the stable one.
    ratio = ratios[max(ratios, key=lambda h: float(h))]
    return {
        "ratio": ratio,
        "ratios": ratios,
        "analytic": float(match.group("analytic")),
        "params": int(match.group("params")),
    }


def measure_track(
    track: Track,
    project_dir: Path,
    steps: int = STEPS,
    seed: int = SEED,
    repeats: int = REPEATS,
) -> Measurement:
    """Build, lint, gradient-check and time one project. Everything that has to be
    true before a loss number is allowed to mean anything.

    The same function measures the committed candidate and a scratch copy of it
    with a patch applied. That is the point: the checks a candidate has to pass
    to be considered at all are the checks it is judged by, so there is no way
    for a proposal to be scored on a weaker protocol than the baseline was
    measured on.
    """
    tool = require_tool(track.tool, track.tool_hint)
    # Whether this is the *frozen* comparator, not whether it is the candidate.
    # The distinction is load-bearing: a proposal is measured in a scratch copy of
    # the candidate, which is neither the candidate nor the comparator, and the
    # first version of this asked the wrong question. Answering "is this the
    # candidate?" skipped the gradient probe for every scratch measurement, so
    # every experiment was recorded with `grad_ratio 0.0` and, worse, no proposal
    # was finite-difference checked at all. The gate the whole harness exists to
    # enforce was silently not running, and 0.0 is the documented "not measured"
    # value, so a skipped probe was indistinguishable from a probe that had not run.
    is_frozen = project_dir == track.comparator
    source_name = track.comparator_source if is_frozen else track.candidate_source
    for required in (DATASET, project_dir / source_name, track.probe):
        if not required.is_file():
            raise ResearchError(
                f"error: {required} is missing. Run `python3 -m tools.autoresearch "
                f"--seed` to create the workspace."
            )
    command = build_track(tool, track, project_dir)
    lint_track(tool, track, project_dir)
    if not is_frozen:
        grad = run_gradient_probe(tool, track, project_dir)
    else:
        # The frozen track has no probe -- the probe is part of the candidate, not
        # the baseline. The seeded candidate is the baseline's source plus at most
        # the accessors the probe needs, so its ratio is the baseline tape's
        # ratio, and measuring the candidate supplies it. Until it does, 0.0 means
        # "not measured" and the page says so.
        grad = {"ratio": 0.0, "ratios": {}, "analytic": 0.0, "params": 0}

    # One run for the loss axis -- it is deterministic at a fixed seed -- and
    # `repeats` for the speed axis, which is not.
    losses, first_elapsed, reported = train_once(track, command, steps, seed, project_dir)
    wall = [first_elapsed]
    for _ in range(max(0, repeats - 1)):
        _, elapsed, _ = train_once(track, command, steps, seed, project_dir)
        wall.append(elapsed)

    return Measurement(
        losses=losses,
        window=windowed(losses),
        steps_per_sec=steps / statistics.median(wall),
        wall_seconds=[round(value, 4) for value in wall],
        reported_steps_per_sec=reported,
        grad_ratio=grad["ratio"],
        grad_ratios=grad["ratios"],
        grad_analytic=grad["analytic"],
        grad_params=grad["params"],
        log="",
    )


def measure_candidate(track: Track, steps: int = STEPS, seed: int = SEED, repeats: int = REPEATS) -> Measurement:
    return measure_track(track, track.candidate, steps, seed, repeats)


def measure_baseline(track: Track, steps: int = STEPS, seed: int = SEED, repeats: int = REPEATS) -> Measurement:
    """The frozen track, measured the same way, in the same session.

    The speed axis of `benchmarks/results.json` is not usable here and the reason
    is worth stating: it was measured on an Apple M1, and a ratio against a
    number from another machine is a number about that machine. So the comparator
    is rebuilt from source every session and thrown away with it.
    """
    return measure_track(track, track.comparator, steps, seed, repeats)


# ─── the verdict ─────────────────────────────────────────────────────────────


def verdict(
    base_loss: float,
    base_speed: float,
    cand_loss: float,
    cand_speed: float,
    window: dict[str, float],
) -> dict[str, Any]:
    """Decide keep or discard from measured numbers. A pure function, and the
    single place the research policy lives.

    The rule is Pareto, not scalar, and the reason is that "lower loss" alone is
    how a repository ends up quietly shipping a slower model. A candidate is kept
    when it wins one axis materially and loses the other by no more than the
    tolerance:

    * loss down by at least `LOSS_MATERIAL`, and speed no more than `SPEED_TOL`
      worse -- or
    * speed up by at least `SPEED_MATERIAL`, and loss no more than `LOSS_TOL`
      worse.

    The two thresholds in each pair are deliberately different numbers. Material
    is "this went somewhere", tolerance is "this is not a regression"; the first
    has to clear the noise floor, the second only has to be generous.

    Three things this deliberately does not do. It does not accept a tie: an
    exact tie is a no-op, and a no-op that advances the branch is how a log fills
    with 200 identical rows and no information. It does not look at the gradient
    ratio, because that is a gate and not a score -- a broken tape is a crash,
    not a bad candidate. And it does not consult anything the model said.
    """
    if base_loss <= 0 or base_speed <= 0:
        raise ResearchError("the baseline is degenerate; refusing to compare against it")

    loss_gain = (base_loss - cand_loss) / base_loss
    speed_gain = (cand_speed - base_speed) / base_speed

    learned = window.get("relative_improvement", 0.0) >= MIN_RELATIVE_IMPROVEMENT
    loss_won = loss_gain >= LOSS_MATERIAL and speed_gain >= -SPEED_TOL
    speed_won = speed_gain >= SPEED_MATERIAL and loss_gain >= -LOSS_TOL

    checks = {
        "learned": {
            "value": window.get("relative_improvement", 0.0),
            "threshold": MIN_RELATIVE_IMPROVEMENT,
            "pass": learned,
            "why": "a candidate that does not learn cannot win, and cannot be allowed "
            "to win on speed alone",
        },
        "loss_material": {
            "value": loss_gain,
            "threshold": LOSS_MATERIAL,
            "pass": loss_gain >= LOSS_MATERIAL,
            "why": "the loss axis only counts once it clears the noise floor, which "
            "is about one standard error of a 50-step window",
        },
        "loss_tolerance": {
            "value": loss_gain,
            "threshold": -LOSS_TOL,
            "pass": loss_gain >= -LOSS_TOL,
            "why": "winning on speed is not a licence to lose real loss",
        },
        "speed_material": {
            "value": speed_gain,
            "threshold": SPEED_MATERIAL,
            "pass": speed_gain >= SPEED_MATERIAL,
            "why": "wall clock is heavy-tailed; only an unambiguous speedup counts",
        },
        "speed_tolerance": {
            "value": speed_gain,
            "threshold": -SPEED_TOL,
            "pass": speed_gain >= -SPEED_TOL,
            "why": "winning on loss is not a licence to be meaningfully slower",
        },
    }

    kept = learned and (loss_won or speed_won)
    if kept:
        rule = "loss_won" if loss_won else "speed_won"
        explanation = (
            f"kept: loss {loss_gain * 100:+.1f}% and speed {speed_gain * 100:+.1f}% "
            f"against the session baseline, via {rule}"
        )
    else:
        rule = "none"
        failing = [name for name, check in checks.items() if not check["pass"]]
        explanation = (
            f"discarded: loss {loss_gain * 100:+.1f}%, speed {speed_gain * 100:+.1f}%; "
            f"failing {', '.join(failing)}"
        )
    return {
        "kept": kept,
        "rule": rule,
        "explanation": explanation,
        "loss_gain": loss_gain,
        "speed_gain": speed_gain,
        "checks": checks,
    }


# ─── diffs ───────────────────────────────────────────────────────────────────


def unified(before: str, after: str, before_name: str, after_name: str) -> str:
    """A unified diff, from the stdlib. `difflib` is the only diff this
    repository has ever needed and adding a dependency for one would be worse
    than the 300 lines it saves."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=before_name,
        tofile=after_name,
        n=3,
    )
    text = "".join(diff)
    return text if text.endswith("\n") or not text else text + "\n"


def diff_counts(patch: str) -> dict[str, int]:
    added = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {"added": added, "removed": removed}


def regenerate_best_diff(track: Track) -> dict[str, int]:
    """The patch from the frozen baseline to whatever the candidate currently is.
    This is the "admire how the code changed" artefact, and it is regenerated on
    every keep so it is always the current best against the original."""
    if not track.comparator_file.is_file() or not track.candidate_file.is_file():
        return {"added": 0, "removed": 0}
    patch = unified(
        track.comparator_file.read_text(encoding="utf-8"),
        track.candidate_file.read_text(encoding="utf-8"),
        f"a/implementations/{track.comparator_dir}/{track.comparator_source}",
        f"b/{track.source_key}",
    )
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    track.best_diff.write_text(patch, encoding="utf-8")
    return diff_counts(patch)


# ─── rendering results.json ──────────────────────────────────────────────────

CAVEATS = [
    "The speed axis is only comparable within one session. It was measured on the "
    "machine named under `runner`, and a ratio against a number from another "
    "machine is a number about that machine. `benchmarks/results.json` is not used "
    "as the baseline for exactly this reason.",
    "The loss axis is deterministic at a fixed seed: the track's PRNG is "
    "xoshiro256++ with SplitMix64 seeding, so a given candidate produces a given "
    "loss curve or it does not. The speed axis is not, and is the median of "
    f"{REPEATS} runs.",
    "Wall clock here is the whole process, including startup and the twenty-sample "
    "inference pass. Both are small and both are constant, and the baseline is "
    "measured the same way in the same session, so the comparison is fair. The "
    "absolute figure is not a pure training time.",
    "A loss number cannot tell a correct gradient from a wrong one -- the C port's "
    "was wrong by a factor of -8, its loss curve still tracked the reference to "
    "within 7%, and it still trained "
    "(docs/KNOWN-ISSUES.md issue 1; fixed since, at 1.03). Every candidate is "
    "finite-difference checked before it competes; see `grad_ratio` on each row.",
]


def pareto_frontier(rows: list[Row]) -> list[str]:
    """Run ids on the Pareto frontier: no other measured run is both faster and
    lower-loss. Precomputed here rather than in the browser so the chart and the
    test agree about which points are on it."""
    measured = [r for r in rows if r.status in ("keep", "baseline") and r.loss > 0]
    frontier: list[str] = []
    for candidate in measured:
        dominated = any(
            other is not candidate
            and other.loss <= candidate.loss
            and other.steps_per_sec >= candidate.steps_per_sec
            and (other.loss < candidate.loss or other.steps_per_sec > candidate.steps_per_sec)
            for other in measured
        )
        if not dominated:
            frontier.append(candidate.run_id)
    return frontier


def build_results_document() -> dict[str, Any]:
    rows = read_ledger()
    runs: list[dict[str, Any]] = []
    for row in rows:
        record = row.to_json()
        run_file = RUNS_DIR / f"{row.run_id}.json"
        record["measured"] = (
            read_json(run_file).get("measured", "") if run_file.is_file() else ""
        )
        record["has_record"] = run_file.is_file()
        runs.append(record)

    # Per track, because "the best" is only meaningful against its own
    # comparator: the Go candidate's loss is not comparable to the Rust one's
    # speed, and a single global best would be a number about nothing.
    baselines: dict[str, Any] = {}
    bests: dict[str, Any] = {}
    provenances: dict[str, Any] = {}
    for track in TRACKS.values():
        track_rows = [r for r in rows if r.track == track.name]
        track_kept = [r for r in track_rows if r.status == "keep"]
        baselines[track.name] = next(
            (r.to_json() for r in track_rows if r.status == "baseline"), None
        )
        bests[track.name] = (
            min(track_kept, key=lambda r: (r.loss, -r.steps_per_sec)).to_json()
            if track_kept
            else None
        )
        provenances[track.name] = {
            "baseline_source": (
                str(track.comparator_file.relative_to(REPO_ROOT))
                if track.comparator_file.is_file()
                else f"implementations/{track.comparator_dir}/{track.comparator_source}"
            ),
            "baseline_sha256": (
                sha256_file(track.comparator_file) if track.comparator_file.is_file() else ""
            ),
            "candidate_source": f"autoresearch/{track.candidate_dir}/{track.candidate_source}",
            "candidate_sha256": (
                sha256_file(track.candidate_file) if track.candidate_file.is_file() else ""
            ),
            "gradient_probe": f"autoresearch/{track.candidate_dir}/{track.probe_path}",
            "gradient_probe_sha256": (
                sha256_file(track.probe) if track.probe.is_file() else ""
            ),
        }

    kept = [r for r in rows if r.status == "keep"]
    baseline_doc = read_json(BASELINE_JSON) if BASELINE_JSON.is_file() else {}

    return {
        "$comment": (
            "GENERATED by tools/autoresearch.py --render -- do not edit. The "
            "source of truth is autoresearch/results.tsv plus autoresearch/runs/*.json. "
            "CI regenerates this and fails on any diff, so a number on the site "
            "cannot stop matching the run that produced it."
        ),
        "generator": "tools/autoresearch.py --render",
        "track": "multi",
        "tracks": {
            name: {
                "language": track.language,
                "candidate_dir": track.candidate_dir,
                "source": track.candidate_source,
                "probe": track.probe_path,
                "build": track.build_hint,
            }
            for name, track in TRACKS.items()
        },
        "provenance_of": provenances,
        "protocol": {
            "steps": STEPS,
            "seed": SEED,
            "repeats": REPEATS,
            "trend_window": TREND_WINDOW,
            "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
            "loss_axis": (
                "mean loss over the last "
                f"{TREND_WINDOW} of {STEPS} steps, at seed {SEED}"
            ),
            "speed_axis": f"steps per second, median of {REPEATS} runs of the whole process",
        },
        "thresholds": {
            "loss_material": LOSS_MATERIAL,
            "loss_tolerance": LOSS_TOL,
            "speed_material": SPEED_MATERIAL,
            "speed_tolerance": SPEED_TOL,
            "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "rule": (
                "keep if (loss down at least loss_material and speed no worse than "
                "speed_tolerance) or (speed up at least speed_material and loss no "
                "worse than loss_tolerance), and the run learned"
            ),
            "note": (
                "loss_material and speed_material are noise floors, tolerances are "
                "regression limits; they are different numbers on purpose. The "
                "justification for each is in the comment above its definition in "
                "tools/autoresearch.py."
            ),
        },
        "objective": {
            "kind": "pareto",
            "axes": ["windowed loss (lower is better)", "steps per second (higher is better)"],
            "decided_by": "tools/autoresearch.py, from measured medians. The model "
            "proposes an edit and never states a verdict.",
        },
        "model": model_summary(),
        "runner": baseline_doc.get("runner", {}),
        "dataset_sha256": sha256_file(DATASET) if DATASET.is_file() else "",
        "baselines": baselines,
        "bests": bests,
        "counts": {
            "experiments": len([r for r in rows if r.status != "baseline"]),
            "keep": len(kept),
            "discard": len([r for r in rows if r.status == "discard"]),
            "crash": len([r for r in rows if r.status == "crash"]),
        },
        # Per track, for the same reason `bests` is: a frontier is a claim about
        # one comparator, and the Go candidate's steps-per-second says nothing
        # about the Rust candidate's.
        "pareto": {
            name: pareto_frontier([r for r in rows if r.track == name])
            for name in TRACKS
        },
        "runs": runs,
        "caveats": CAVEATS,
    }


def model_summary() -> dict[str, Any]:
    """The pinned model, as the page needs it.

    A trimmed copy rather than the whole `model.json`, because this file is
    imported eagerly by the site and the enum list and the pricing note are for a
    reader auditing the pin, not for a chart. The full file is committed and the
    page links to it.
    """
    if not MODEL_JSON.is_file():
        return {}
    spec = read_json(MODEL_JSON)
    return {
        key: spec[key]
        for key in (
            "model",
            "model_display_name",
            "reasoning_parameter",
            "reasoning_effort",
            "verified",
            "verification",
            "pricing_note",
        )
        if key in spec
    }


def render(published: bool = False) -> bool:
    """Regenerate `results.json`. Returns True if the file changed."""
    payload = json.dumps(build_results_document(), indent=2) + "\n"
    previous = RESULTS_JSON.read_text(encoding="utf-8") if RESULTS_JSON.is_file() else None
    if previous == payload:
        return False
    if published and previous is not None:
        print(
            f"  wrote {RESULTS_JSON.relative_to(REPO_ROOT)} (changed)",
            file=sys.stderr,
        )
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(payload, encoding="utf-8")
    return True


# ─── the model ───────────────────────────────────────────────────────────────


@dataclass
class Proposal:
    patch: str
    hypothesis: str
    description: str


FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
HYPOTHESIS_RE = re.compile(r"^HYPOTHESIS:\s*(.+)$", re.MULTILINE)
SUMMARY_RE = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE)
ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def openrouter_key(spec: dict[str, Any]) -> str:
    name = spec.get("key_env", "OPENROUTER_API_KEY")
    key = os.environ.get(name, "")
    if key:
        return key
    # The common case is that the key is set by an interactive rc file and this
    # process did not inherit it, because a non-interactive shell reads a
    # different set of files than `zsh` does in your terminal. Worth one cheap
    # probe, because the alternative is a user staring at "not set" while their
    # key sits three files away.
    for shell, flag in (("zsh", "-lic"), ("bash", "-lic")):
        try:
            probe = subprocess.run(
                [shell, flag, f"printenv {name}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        found = probe.stdout.strip()
        if probe.returncode == 0 and found:
            return found
    raise ResearchError(
        f"error: ${name} is not set in this environment, nor in a login shell.\n"
        f"  The research loop talks to {spec.get('endpoint')} directly rather than "
        f"going through a coding agent, so it needs the key in the environment it "
        f"runs in.\n"
        f"  `printenv {name}` in your terminal should print something. If it does "
        f"and this still says it is unset, the key is in an rc file a non-interactive "
        f"shell does not read -- export it in the shell you run `make` from, or put "
        f"it in a .env and source that first."
    )


def verify_model_spec(spec: dict[str, Any]) -> None:
    """Fail loudly if the pinned model is gone or the pinned reasoning knob has
    moved. The alternative is worse than an error: a silently-dropped
    `reasoning_effort` looks exactly like a model that is thinking and is not,
    and the run records would claim otherwise."""
    endpoint = str(spec.get("models_endpoint", "")).strip()
    model = str(spec.get("model", "")).strip()
    parameter = str(spec.get("reasoning_parameter", "reasoning_effort")).strip()
    if not endpoint or not model:
        raise ResearchError(
            "error: autoresearch/model.json must set `model` and `models_endpoint`."
        )
    try:
        request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - reported, not raised as-is
        raise ResearchError(
            f"error: could not read the model catalogue at {endpoint}: {exc}\n"
            f"  Pinning the model is the point of autoresearch/model.json. If this "
            f"is a network problem, retry; if the model has been retired, update the "
            f"file and note it in the commit."
        ) from exc
    entry = next((m for m in payload.get("data", []) if m.get("id") == model), None)
    if entry is None:
        raise ResearchError(
            f"error: the pinned model {model!r} is not in the catalogue at {endpoint}.\n"
            f"  Update `model` in autoresearch/model.json to a model that exists, and "
            f"update `verified` to the date you checked."
        )
    supported = entry.get("supported_parameters") or []
    if supported and parameter not in supported:
        raise ResearchError(
            f"error: {model} no longer advertises {parameter!r}.\n"
            f"  supported_parameters: {', '.join(supported)}\n"
            f"  If the reasoning knob has been renamed, update `reasoning_parameter` "
            f"and `reasoning_effort` in autoresearch/model.json. The harness will not "
            f"quietly continue without it: a dropped reasoning parameter produces "
            f"output that looks identical and is not."
        )


def build_messages(track: Track, prompt: str) -> list[dict[str, str]]:
    """The two messages the model gets: the brief, then the ask.

    Split out from `ask_model` so that it can be checked without a network or an
    API key. That is not a stylistic preference: `ask_model`'s previous version
    assembled the messages inline, referenced a `track` it did not have, and
    nothing in the test suite called it -- so `make lint`, 193 Python tests, the
    web suite, the full gate and `autoresearch verify` all passed, and the very
    first experiment on a new track died with a `NameError` about 95 seconds in.
    A function whose only caller needs a network is a function nothing tests.
    """
    return [
        {"role": "system", "content": read_prompt(track)},
        {"role": "user", "content": prompt},
    ]


def ask_model(
    spec: dict[str, Any], track: Track, prompt: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": spec["model"],
        "messages": build_messages(track, prompt),
        "max_tokens": max_tokens,
        "temperature": spec.get("temperature", 1.0),
    }
    # The top-level `reasoning_effort` is used rather than the `reasoning` object
    # because OpenRouter enumerates its values in the 400 it returns for a bad
    # one, and because the two cannot be combined with `reasoning.max_tokens`.
    # Verified against the live API; see autoresearch/model.json.
    body[str(spec.get("reasoning_parameter", "reasoning_effort"))] = spec.get("reasoning_effort")
    headers = {
        "Authorization": f"Bearer {openrouter_key(spec)}",
        "Content-Type": "application/json",
        "HTTP-Referer": spec.get("http_referer", "https://github.com/denimpatel/learn-micro-gpt"),
        "X-Title": spec.get("title", "learn-micro-gpt autoresearch"),
    }
    request = urllib.request.Request(
        str(spec["endpoint"]),
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        parameter = spec.get("reasoning_parameter", "reasoning_effort")
        hint = ""
        if parameter in detail:
            hint = (
                f"\n  The API is talking about `{parameter}`, which is the knob this "
                f"harness sets. Update autoresearch/model.json rather than retrying: "
                f"the value it wants to use no longer exists."
            )
        raise ResearchError(f"the model request failed (HTTP {exc.code})\n{detail}{hint}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ResearchError(f"the model request failed: {exc}") from exc

    message = (payload.get("choices") or [{}])[0].get("message") or {}
    usage = payload.get("usage") or {}
    return str(message.get("content") or ""), {
        "model": payload.get("model", spec["model"]),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": usage.get("cost"),
    }


def read_prompt(track: Track) -> str:
    """The agent brief, with this track's names substituted in.

    `program.md` is a template, and a template rather than three files because it
    is "the one part of this system a human is meant to edit": three copies of the
    same argument would drift, and a brief that describes `cargo clippy` to a model
    holding Go source is worse than no brief at all. Every placeholder is filled
    from the `Track` table, so a new language gets a correct brief by adding a row
    rather than by writing prose.
    """
    if not PROGRAM_MD.is_file():
        raise ResearchError(
            f"error: {PROGRAM_MD.relative_to(REPO_ROOT)} is missing. It is the agent "
            f"brief and the one part of this system a human is meant to edit."
        )
    values = {
        "language": track.language,
        "candidate_dir": f"autoresearch/{track.candidate_dir}",
        "source_key": track.source_key,
        "probe_path": track.probe.relative_to(REPO_ROOT).as_posix(),
        "dependency_note": track.dependency_note,
        "lint_command": track.lint_command,
        "probe_api": track.probe_api,
        "patch_caution": track.patch_caution,
    }
    # Substituted with a regex rather than `str.format`, because the brief contains
    # literal braces: the example patch adds a `Tensor { new, forward, params }`
    # literal, and `format` would read that as a field name and raise.
    PROGRAM_TEXT = PROGRAM_MD.read_text(encoding="utf-8")
    filled = re.sub(
        r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), PROGRAM_TEXT
    )
    leftover = re.findall(r"\{\{(\w+)\}\}", filled)
    if leftover:
        raise ResearchError(
            f"error: {PROGRAM_MD.relative_to(REPO_ROOT)} still has unfilled "
            f"placeholders after rendering for the {track.name} track: "
            f"{', '.join(sorted(set(leftover)))}. Add them to the format call in "
            f"`read_prompt` and to the `Track` table -- a brief with braces in it is "
            f"a brief the model will read as instructions."
        )
    return filled


def parse_proposal(text: str) -> Proposal:
    """Strictly. An unparseable response is a crash, not a skip, and it is never
    retried with the same prompt: a model that could not produce a parseable
    patch this round will not produce one next round, and retrying just spends
    tokens to learn the same thing twice."""
    fenced = FENCE_RE.search(text)
    if not fenced:
        fenced = ANY_FENCE_RE.search(text)
    if not fenced:
        raise ResearchError(
            "the response contained no fenced code block, so there is no patch to "
            "apply.\n"
            f"--- response ---\n{text[:1500]}"
        )
    patch = fenced.group(1).strip()
    if not patch:
        raise ResearchError("the fenced block in the response was empty")
    if not any(line.startswith(("+", "-")) for line in patch.splitlines()):
        raise ResearchError(
            f"the fenced block contains no added or removed lines, so it is not a "
            f"patch:\n{patch[:800]}"
        )
    hypothesis = HYPOTHESIS_RE.search(text)
    summary = SUMMARY_RE.search(text)
    return Proposal(
        patch=patch + "\n",
        hypothesis=hypothesis.group(1).strip() if hypothesis else "",
        description=clean_field(summary.group(1).strip() if summary else "no summary given", "description"),
    )


def build_prompt(
    track: Track,
    rows: list[Row],
    baseline: dict[str, Any],
    history: int = 15,
) -> str:
    """Everything the model gets, for one track.

    The ledger it is shown is that track's rows only. Showing a Go candidate the
    Rust track's history would invite it to propose a change already tried, and
    worse, would imply the two are competing on the same axis.
    """
    """Everything the model gets. No diff of the previous attempt, no hidden
    state: the ledger is the memory, and it is a committed file."""
    rows = [r for r in rows if r.track == track.name]
    recent = rows[-history:]
    ledger = "\n".join(
        f"{r.run_id}\t{r.status}\tloss={r.loss:.4f}\tsps={r.steps_per_sec:.1f}\t"
        f"dloss={r.loss_gain * 100:+.1f}%\tdspeed={r.speed_gain * 100:+.1f}%\t"
        f"grad={r.grad_ratio:.3f}\t{r.description}"
        for r in recent
    )
    source_text = track.candidate_file.read_text(encoding="utf-8")
    source_lines = source_text.splitlines()
    thresholds = (
        f"loss_material={LOSS_MATERIAL}  loss_tolerance={LOSS_TOL}  "
        f"speed_material={SPEED_MATERIAL}  speed_tolerance={SPEED_TOL}  "
        f"min_relative_improvement={MIN_RELATIVE_IMPROVEMENT}"
    )
    return (
        f"# Session baseline (measured today, on this machine, {REPEATS}x)\n\n"
        f"  windowed loss (last {TREND_WINDOW} of {STEPS} steps): {baseline['loss']:.4f}\n"
        f"  steps per second:                            {baseline['steps_per_sec']:.1f}\n"
        f"  gradient directional-derivative ratio:       {baseline['grad_ratio']:.4f}\n"
        f"  grad ratio band: [0.5, 2.0] and stable across step sizes\n\n"
        f"# Thresholds a candidate must clear\n\n"
        f"  {thresholds}\n\n"
        f"A candidate is KEPT if it lowers the windowed loss by at least "
        f"{LOSS_MATERIAL * 100:.0f}% while staying within {SPEED_TOL * 100:.0f}% of the "
        f"baseline speed, OR raises the speed by at least "
        f"{SPEED_MATERIAL * 100:.0f}% while staying within {LOSS_TOL * 100:.0f}% of the "
        f"baseline loss. A tie is a discard: only a real improvement advances the "
        f"candidate.\n\n"
        f"# The ledger, most recent {len(recent)} rows\n\n"
        f"```\nrun_id\tstatus\tloss\tsps\tdloss\tdspeed\tgrad\tdescription\n{ledger}\n```\n\n"
        f"# The file you are editing\n\n"
        f"`{track.source_key}`, reproduced here in full. It is "
        f"{len(source_lines)} lines, and this is all of it. Read it. Do not invent a "
        f"signature, a method name or a helper that is not in this text: a patch "
        f"that references an API you imagined does not apply, and the experiment is "
        f"wasted.\n\n"
        f"```{track.fence}\n{source_text}\n```\n\n"
        f"Reply with exactly this shape and nothing else:\n\n"
        f"HYPOTHESIS: one sentence -- why you think this will help, stated so it "
        f"could be wrong\n"
        f"SUMMARY: one line, no commas needed, describing the change for the results "
        f"table\n\n"
        f"```diff\n"
        f"--- a/{track.source_key}\n"
        f"+++ b/{track.source_key}\n"
        f"@@ -766,7 +766,7 @@\n"
        f" unchanged context line\n"
        f"-a line you are removing\n"
        f"+a line you are adding\n"
        f"```\n\n"
        f"The diff is applied with `git apply --recount`, which ignores the line "
        f"counts in your hunk headers -- so get them approximately right and do not "
        f"waste effort counting -- but the **context lines must match the file "
        f"exactly**, including indentation. Copy them out of the text above. Change "
        f"only what your hypothesis needs."
    )


# ─── git ─────────────────────────────────────────────────────────────────────


def git(*args: str, check: bool = True) -> str:
    result = run_command(["git", *args], cwd=REPO_ROOT, timeout=120)
    if check and result.returncode != 0:
        raise ResearchError(
            f"git {' '.join(args)} failed (exit {result.returncode})\n"
            f"{(result.stderr or '')[-2000:]}"
        )
    return result.stdout


def outside_research_dirty() -> list[str]:
    """Tracked modifications outside `autoresearch/`.

    The loop stages one path, so it cannot sweep up the user's work by accident.
    But a *stale* tracked modification elsewhere means the working tree is not
    what the loop thinks it is, and it is better to say so before spending an
    hour of compute than after.
    """
    out = git("status", "--porcelain", "--untracked-files=no")
    dirty: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if ' -> ' in path:
            path = path.split(' -> ')[-1]
        if not path.startswith("autoresearch/"):
            dirty.append(path)
    return dirty


def apply_patch(patch: str, worktree: Path) -> tuple[bool, str]:
    """`git apply` into a scratch checkout, so a patch that does not apply cannot
    touch the committed candidate. Returns (applied, error).

    The patch goes through a temporary file rather than stdin: this tool does not
    shell out to a pipe, and `git apply -` would need one.

    `--recount` because a model's line counts in a hunk header are its arithmetic,
    not its reading of the file, and they are the single most common reason an
    otherwise correct patch is rejected. It makes git infer the counts from the
    body instead. It does *not* forgive wrong context lines, which is right:
    context that does not match means the patch was written against a file the
    model was imagining, and the experiment is not worth running.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as handle:
        handle.write(patch)
        patch_file = Path(handle.name)
    try:
        result = run_command(
            ["git", "apply", "--verbose", "--recount", str(patch_file)],
            cwd=worktree,
            timeout=60,
        )
        return result.returncode == 0, (result.stderr or result.stdout or "").strip()
    finally:
        patch_file.unlink(missing_ok=True)


def commit_and_push(paths: list[str], message: str, branch: str, push: bool) -> str:
    for path in paths:
        git("add", "--", path)
    if not git("diff", "--cached", "--name-only").strip():
        raise ResearchError("nothing staged; refusing to make an empty commit")
    git("commit", "-m", message)
    sha = git("rev-parse", "HEAD").strip()
    if push:
        git("push", "origin", branch)
    return sha


def commit_message(
    run_id: str,
    status: str,
    row: Row,
    baseline: dict[str, Any],
    measurement: Measurement | None,
    judged: dict[str, Any] | None,
    hypothesis: str,
    model_label: str,
    failure: str,
) -> str:
    """Informative on purpose.

    The user reads these, and so does anyone who finds the repository later. The
    body carries the hypothesis, the measured numbers, the arithmetic of the
    verdict, and where the full record is -- so the commit alone is enough to
    reconstruct what was tried and why, without opening the JSON. A commit
    message that only says "keep" is a worse record than the results.tsv row it
    duplicates.
    """
    lines = [f"research({row.track}): {status} {run_id} -- {row.description}", ""]
    if status == "crash":
        lines += [
            f"Experiment {run_id} · crash · {failure}",
            "",
            "The candidate did not produce a measurement, so the committed candidate "
            "is unchanged. This row exists so the attempt is on the record rather "
            "than silently retried; the raw model response, if there was one, is in "
            "autoresearch/logs/ and is gitignored.",
        ]
    else:
        assert measurement is not None and judged is not None
        lines += [
            f"Experiment {run_id} · {status} · loss {row.loss:.4f} "
            f"({row.loss_gain * 100:+.1f}% vs baseline {baseline['loss']:.4f})",
            f"                   · speed {row.steps_per_sec:.1f} steps/s "
            f"({row.speed_gain * 100:+.1f}%, median of {REPEATS})",
        ]
        if hypothesis:
            lines += ["", f"Hypothesis: {hypothesis}"]
        lines += [
            "",
            f"Measured: last-{TREND_WINDOW}-step windowed mean {measurement.window['last_window_mean']:.4f} "
            f"(first window {measurement.window['first_window_mean']:.4f}, "
            f"{measurement.window['relative_improvement'] * 100:+.1f}%); "
            f"steps/sec {measurement.steps_per_sec:.1f} over {len(measurement.wall_seconds)} runs.",
            f"Gradient directional-derivative ratio {row.grad_ratio:.4f} "
            f"(band 0.5-2.0, checked at two step sizes).",
            "",
            f"Verdict: {judged['explanation']}.",
            f"Thresholds: loss_material={LOSS_MATERIAL} loss_tol={LOSS_TOL} "
            f"speed_material={SPEED_MATERIAL} speed_tol={SPEED_TOL} "
            f"min_improvement={MIN_RELATIVE_IMPROVEMENT}.",
        ]
    if failure and status != "crash":
        lines += ["", f"Failure: {failure}"]
    lines += [
        "",
        f"Record: autoresearch/runs/{run_id}.json",
        f"Diff:   autoresearch/diffs/{run_id}.patch",
        f"Model:  {model_label}",
    ]
    return "\n".join(lines) + "\n"


# ─── the session ─────────────────────────────────────────────────────────────


def runner_identity() -> dict[str, Any]:
    cargo = find_tool("cargo")
    rustc = None
    if cargo:
        version = run_command([cargo, "--version"], cwd=REPO_ROOT, timeout=60)
        rustc = (version.stdout or version.stderr or "").strip() or None
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "rustc": rustc,
        "measured": now_iso(),
        "note": (
            "The speed axis is only comparable within a session. The baseline is "
            "rebuilt from source and re-measured every session for this reason; "
            "benchmarks/results.json was taken on a different machine and is not used."
        ),
    }


def measure_session_baseline(
    track: Track, steps: int, seed: int, repeats: int
) -> dict[str, Any]:
    """The comparator for this session, and its identity.

    Re-measured every session on purpose. The obvious optimisation -- reuse the
    committed benchmark number -- would be wrong: that number was taken on an
    Apple M1, and a speed ratio against a number from another machine is a number
    about that machine, not about this one.
    """
    frozen = measure_baseline(track, steps, seed, repeats)
    candidate = measure_candidate(track, steps, seed, repeats)
    payload = {
        "$comment": (
            f"The session baseline for the {track.name} track: the frozen "
            "implementation, rebuilt and re-measured on the machine running the "
            "loop. Regenerated by every session and committed so the site can say "
            "what the numbers were relative to. See docs/AUTORESEARCH.md."
        ),
        "track": track.name,
        "measured": now_iso(),
        "runner": runner_identity(),
        "protocol": {"steps": steps, "seed": seed, "repeats": repeats},
        "loss": round(frozen.window["last_window_mean"], 6),
        "window": {key: round(value, 6) for key, value in frozen.window.items()},
        "steps_per_sec": round(frozen.steps_per_sec, 6),
        "wall_seconds": frozen.wall_seconds,
        "loss_curve_sha256": sha256_text(",".join(f"{value:.6f}" for value in frozen.losses)),
        "grad_ratio": round(candidate.grad_ratio, 6),
        "grad_note": (
            f"Measured on the seeded {track.name} candidate, which is the frozen "
            "track's source plus whatever accessors its probe needs. The frozen "
            "track carries no probe of its own, so this is the *frozen* tape's ratio, "
            f"not the candidate's: {candidate.grad_ratio:.4f} for this port is "
            "docs/KNOWN-ISSUES.md issue 5 -- an rmsnorm path that sits outside the "
            "tape -- and not an error in the measurement. The candidate on disk can "
            "and does differ, because the loop is allowed to fix it: the Rust "
            "candidate measures 1.0003, having put rmsnorm on the tape."
        ),
        "source_sha256": {
            str(track.comparator_file.relative_to(REPO_ROOT)): sha256_file(track.comparator_file),
        },
    }
    write_json(track.baseline_json, payload)
    return {
        "loss": payload["loss"],
        "steps_per_sec": payload["steps_per_sec"],
        "grad_ratio": payload["grad_ratio"],
        "window": payload["window"],
    }


# ─── subcommands ─────────────────────────────────────────────────────────────


def cmd_seed(args: argparse.Namespace) -> int:
    """Create the workspace and record the baseline row. Idempotent per track.

    Idempotent *per track*, which is the change multi-track forces: seeding Go
    against a ledger that already holds 104 Rust rows has to add the Go baseline
    rather than decide the ledger is full.
    """
    track = get_track(args.track)
    for path in (track.candidate, RUNS_DIR, DIFFS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.is_file():
        RESULTS_TSV.write_text(HEADER + "\n", encoding="utf-8")
    rows = read_ledger()
    mine = [r for r in rows if r.track == track.name]
    if mine:
        print(
            f"{RESULTS_TSV.relative_to(REPO_ROOT)} already has {len(mine)} "
            f"{track.name} row(s); nothing to seed"
        )
        regenerate_best_diff(track)
        render()
        return 0

    print(
        f"measuring the session baseline for {track.name}: rebuilding the frozen "
        f"track and timing it"
    )
    baseline = measure_session_baseline(track, args.steps, args.seed, args.repeats)
    print(
        f"  windowed loss {baseline['loss']:.4f}   "
        f"{baseline['steps_per_sec']:.1f} steps/s   "
        f"grad ratio {baseline['grad_ratio']:.4f}"
    )

    parent = git("rev-parse", "HEAD").strip()
    row = Row(
        run_id=next_run_id(rows),
        parent=parent,
        track=track.name,
        loss=baseline["loss"],
        steps_per_sec=baseline["steps_per_sec"],
        loss_gain=0.0,
        speed_gain=0.0,
        grad_ratio=baseline["grad_ratio"],
        status="baseline",
        reason=f"frozen implementations/{track.comparator_dir}, re-measured this session",
        description=(
            f"the frozen {track.language} track, unmodified, as this session's comparator"
        ),
    )
    record = {
        "$comment": "One experiment. GENERATED by tools/autoresearch.py -- do not edit.",
        "run_id": row.run_id,
        "track": row.track,
        "measured": now_iso(),
        "status": row.status,
        "kind": "baseline",
        "reason": row.reason,
        "description": row.description,
        "parent_commit": parent,
        "protocol": {"steps": args.steps, "seed": args.seed, "repeats": args.repeats},
        "loss": row.loss,
        "window": baseline["window"],
        "speed": {
            "steps_per_sec": row.steps_per_sec,
            "note": (
                "median of the session repeats; see "
                f"{track.baseline_json.relative_to(REPO_ROOT)} for the raw samples"
            ),
        },
        "grad": {
            "ratio": row.grad_ratio,
            "note": (
                f"From the seeded {track.name} candidate, which is the frozen "
                f"track's source plus whatever its probe needs. "
                + (
                    "It is 1.0, and that is the point: this track differentiates "
                    "through its normalisation, so its backward pass is a correct "
                    "gradient and there is no known-issue discount to read. See "
                    "KNOWN-ISSUES.md issue 1 for what the alternative looked like."
                    if 0.95 <= row.grad_ratio <= 1.05
                    else "See KNOWN-ISSUES.md issue 5 for why it is not 1.0."
                )
            ),
        },
        "source_sha256": {
            str(track.comparator_file.relative_to(REPO_ROOT)): sha256_file(
                track.comparator_file
            ),
        },
    }
    write_json(RUNS_DIR / f"{row.run_id}.json", record)
    append_ledger(row)
    regenerate_best_diff(track)
    render()
    print(
        f"seeded {RESULTS_TSV.relative_to(REPO_ROOT)} with the {track.name} "
        f"baseline row {row.run_id}"
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    changed = render()
    print(
        f"  {RESULTS_JSON.relative_to(REPO_ROOT)} "
        f"{'updated' if changed else 'up to date'}"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """What CI runs. No API key, no loop, no writes.

    Four checks, and the one that matters most is the second: the recorded
    results must still describe the *committed* code, which is a digest
    comparison and is therefore exact on any machine. The loss re-run after it
    is a weaker check than that -- bit-exact agreement is not claimed across
    toolchains, for the reason `docs/BENCHMARKS.md` gives -- so it is checked
    against the same tolerance a candidate is held to, and the delta is printed
    rather than hidden. The speed axis is not checked at all, and says so.
    """
    rows = read_ledger()
    if not rows:
        print("no experiments recorded; nothing to verify", file=sys.stderr)
        return 0

    # Every track that has a kept run, each against its own comparator. A track
    # with no kept candidate is reported as unverified rather than skipped in
    # silence, for the reason the speed axis below is reported rather than skipped.
    targets: list[tuple[Track, Row]] = []
    for track in TRACKS.values():
        kept = [r for r in rows if r.track == track.name and r.status == "keep"]
        if not kept:
            if any(r.track == track.name for r in rows):
                print(f"--  {track.name}: no kept candidate yet; nothing to verify")
            continue
        # Two different runs, and conflating them is a bug this file used to have.
        #
        # The loop promotes the candidate on *every* keep, so the committed file is
        # the LAST keep -- a run that traded a little loss for a lot of speed. The
        # lowest-loss keep is what `results.json` calls the track's best, and the
        # two are not the same run: after a hundred experiments, 0102 was the
        # promoted candidate and 0066 was the lowest loss, and comparing the file
        # against 0066 failed the check with a digest that had never been wrong.
        #
        # A digest check has one question to ask -- does the committed code still
        # produce the number on record -- and the run to ask it of is the run that
        # produced the committed code.
        targets.append((track, kept[-1], min(kept, key=lambda r: (r.loss, -r.steps_per_sec))))
    if not targets:
        print("no kept candidate recorded; nothing to verify")
        return 0

    for track, promoted, best in targets:
        record_path = RUNS_DIR / f"{promoted.run_id}.json"
        record = read_json(record_path)

        print(
            f"\nverifying the committed {track.name} candidate: promoted by "
            f"experiment {promoted.run_id}"
            + (
                f" (lowest loss on this track is {best.run_id}, a different run)"
                if best.run_id != promoted.run_id
                else ""
            )
        )

        # 1. The recorded digest of the candidate must still be the committed file.
        recorded_digest = (record.get("source_sha256") or {}).get(track.source_key)
        actual_digest = (
            sha256_file(track.candidate_file) if track.candidate_file.is_file() else ""
        )
        if recorded_digest and recorded_digest != actual_digest:
            print(
                f"  ! the committed candidate no longer matches experiment "
                f"{promoted.run_id}\n"
                f"    recorded sha256 {recorded_digest}\n"
                f"    on disk        sha256 {actual_digest}\n"
                f"    results.json claims a number this source no longer produces.",
                file=sys.stderr,
            )
            return 1
        print(
            f"  ok  candidate source matches experiment {promoted.run_id} "
            f"(sha256 {actual_digest[:12]})"
        )

        # 2. Build, lint, gradient check.
        measurement = measure_candidate(track, args.steps, args.seed, args.repeats)
        print(
            f"  ok  builds, lint-clean, gradient ratio {measurement.grad_ratio:.4f}"
        )

        # 3. The loss axis, within the tolerance a candidate is held to.
        drift = (
            abs(measurement.window["last_window_mean"] - promoted.loss) / promoted.loss
        )
        status = "ok" if drift <= VERIFY_LOSS_TOLERANCE else "FAIL"
        print(
            f"  {status}  windowed loss {measurement.window['last_window_mean']:.4f} vs recorded "
            f"{promoted.loss:.4f} ({drift * 100:+.2f}%, tolerance "
            f"{VERIFY_LOSS_TOLERANCE * 100:.0f}%)"
        )
        if drift > VERIFY_LOSS_TOLERANCE:
            print(
                "    This is a different loss curve from the one on record. Either the "
                "toolchain rounds differently here, which is expected and bounded, or "
                "the code changed, which the digest above would have caught.",
                file=sys.stderr,
            )
            return 1

        # 4. The speed axis, explicitly not checked, and said out loud rather than
        # silently skipped -- a check that is quietly absent is indistinguishable
        # from one that quietly passed.
        print(
            f"  --  speed axis NOT verified. This runner measured "
            f"{measurement.steps_per_sec:.1f} steps/s against the recorded "
            f"{promoted.steps_per_sec:.1f}, and those are different machines: "
            f"SPEED_AXIS_VERIFIABLE is False for exactly that reason. The recorded "
            f"figure is the honest one for the session that produced it."
        )

    # 5. results.json must be current.
    if render():
        print(
            f"  ! {RESULTS_JSON.relative_to(REPO_ROOT)} was stale and has been "
            f"regenerated; commit it.",
            file=sys.stderr,
        )
        return 1
    print(f"  ok  {RESULTS_JSON.relative_to(REPO_ROOT)} is current")
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    track = get_track(args.track)
    if not RESULTS_TSV.is_file() or not read_ledger():
        raise ResearchError(
            f"error: no {track.name} baseline recorded. Run `python3 -m tools.autoresearch "
            f"seed --track {track.name}` first, or use `make autoresearch-rust TRACK="
            f"{track.name}` which does both."
        )
    dirty = outside_research_dirty()
    if dirty:
        raise ResearchError(
            "error: there are uncommitted changes to tracked files outside "
            f"autoresearch/:\n  {chr(10).join('  ' + p for p in dirty)}\n"
            "  The loop only ever stages autoresearch/, so it would not commit "
            "these -- but it would be measuring a tree that is not the one you "
            "think it is. Commit or stash them first."
        )
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if args.branch and branch != args.branch:
        raise ResearchError(
            f"error: on branch {branch!r}, expected {args.branch!r}. The loop pushes "
            f"to the branch it is on, so it refuses to guess."
        )

    spec = read_json(MODEL_JSON) if MODEL_JSON.is_file() else {}
    if not spec:
        raise ResearchError(
            f"error: {MODEL_JSON.relative_to(REPO_ROOT)} is missing or empty."
        )
    if not args.skip_model_check:
        verify_model_spec(spec)

    baseline = measure_session_baseline(track, args.steps, args.seed, args.repeats)
    print(
        f"[{track.name}] session baseline: loss {baseline['loss']:.4f}  "
        f"{baseline['steps_per_sec']:.1f} steps/s  grad {baseline['grad_ratio']:.4f}"
    )
    if baseline["window"].get("relative_improvement", 0.0) < MIN_RELATIVE_IMPROVEMENT:
        raise ResearchError(
            f"error: the baseline itself only improved "
            f"{baseline['window'].get('relative_improvement', 0.0) * 100:.1f}%, below "
            f"the {MIN_RELATIVE_IMPROVEMENT * 100:.0f}% floor. A session that starts "
            f"from a track that does not learn cannot tell an improvement from a "
            f"regression, so the loop stops here rather than reporting noise."
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    model_label = f"{spec.get('model', '?')} ({spec.get('reasoning_effort', '?')})"
    started = time.monotonic()
    completed_experiments = 0

    for index in range(args.experiments):
        if args.budget_seconds and time.monotonic() - started > args.budget_seconds:
            print(f"\nbudget of {args.budget_seconds}s spent; stopping after {completed_experiments}")
            break
        rows = read_ledger()
        run_id = next_run_id(rows)
        print(
            f"\n[{run_id}] asking the model for a {track.name} experiment "
            f"({completed_experiments + 1})"
        )

        failure = ""
        proposal: Proposal | None = None
        usage: dict[str, Any] = {}
        try:
            text, usage = ask_model(
                spec,
                track,
                build_prompt(track, rows, baseline),
                int(spec.get("max_tokens", 32768)),
            )
            # The raw response is kept, unedited, next to the run record. When a
            # session produces six crashes in a row this is the difference
            # between "the model is not cooperating" and "the model keeps
            # returning the same malformed hunk header".
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            (LOGS_DIR / f"{run_id}.response.md").write_text(text, encoding="utf-8")
            proposal = parse_proposal(text)
        except ResearchError as exc:
            failure = str(exc).splitlines()[0]
            print(f"  crash: {failure}")

        measurement: Measurement | None = None
        judged: dict[str, Any] | None = None
        crate: Path | None = None
        if proposal is not None:
            applied, crate, error = apply_to_scratch(track, proposal.patch)
            if not applied:
                failure = "the patch did not apply"
                detail = error.splitlines()[-1] if error else "no output from git apply"
                print(f"  crash: {failure}: {detail}")
            else:
                try:
                    measurement = measure_track(track, crate, args.steps, args.seed, args.repeats)
                    judged = verdict(
                        baseline["loss"],
                        baseline["steps_per_sec"],
                        measurement.window["last_window_mean"],
                        measurement.steps_per_sec,
                        measurement.window,
                    )
                    print(
                        f"  {'keep' if judged['kept'] else 'discard'}: "
                        f"loss {measurement.window['last_window_mean']:.4f} "
                        f"({judged['loss_gain'] * 100:+.1f}%)  "
                        f"{measurement.steps_per_sec:.1f} steps/s "
                        f"({judged['speed_gain'] * 100:+.1f}%)  "
                        f"grad {measurement.grad_ratio:.4f}"
                    )
                except ResearchError as exc:
                    failure = str(exc).splitlines()[0]
                    print(f"  crash: {failure}")

        if failure or measurement is None or judged is None:
            row = Row(
                run_id=run_id,
                parent=git("rev-parse", "HEAD").strip(),
                track=track.name,
                loss=0.0,
                steps_per_sec=0.0,
                loss_gain=0.0,
                speed_gain=0.0,
                grad_ratio=0.0,
                status="crash",
                reason=clean_field(failure or "no measurement", "reason"),
                description=(proposal.description if proposal else "the model returned nothing usable"),
            )
            record = {
                "$comment": "One experiment. GENERATED by tools/autoresearch.py -- do not edit.",
                "run_id": run_id,
                "measured": now_iso(),
                "status": "crash",
                "kind": "candidate",
                "reason": row.reason,
                "hypothesis": proposal.hypothesis if proposal else "",
                "description": row.description,
                "parent_commit": row.parent,
                "protocol": {"steps": args.steps, "seed": args.seed, "repeats": args.repeats},
                "failure": failure,
                "model": {"id": spec.get("model"), "reasoning_effort": spec.get("reasoning_effort")},
                "usage": usage,
            }
        else:
            if judged["kept"] and crate is not None:
                # Promote only after the measurement exists, and only on a keep.
                # A discard leaves the candidate byte-identical, which is what
                # makes "the next experiment starts from the last thing that
                # worked" true rather than approximately true.
                commit_candidate(track, crate)
            row = Row(
                run_id=run_id,
                parent=git("rev-parse", "HEAD").strip(),
                track=track.name,
                loss=measurement.window["last_window_mean"],
                steps_per_sec=measurement.steps_per_sec,
                loss_gain=judged["loss_gain"],
                speed_gain=judged["speed_gain"],
                grad_ratio=measurement.grad_ratio,
                status="keep" if judged["kept"] else "discard",
                reason=(
                    ""
                    if judged["kept"]
                    else clean_field(
                        "no axis improved materially: " + judged["explanation"], "reason"
                    )
                ),
                description=proposal.description if proposal else "",
            )
            patch_text = proposal.patch if proposal else ""
            DIFFS_DIR.mkdir(parents=True, exist_ok=True)
            (DIFFS_DIR / f"{run_id}.patch").write_text(patch_text, encoding="utf-8")
            record = {
                "$comment": "One experiment. GENERATED by tools/autoresearch.py -- do not edit.",
                "run_id": run_id,
                "measured": now_iso(),
                "status": row.status,
                "kind": "candidate",
                "hypothesis": proposal.hypothesis if proposal else "",
                "description": row.description,
                "parent_commit": row.parent,
                "protocol": {"steps": args.steps, "seed": args.seed, "repeats": args.repeats},
                "loss": round(measurement.window["last_window_mean"], 6),
                "window": {key: round(value, 6) for key, value in measurement.window.items()},
                "loss_curve": [round(value, 6) for value in measurement.losses],
                "speed": {
                    "steps_per_sec": round(measurement.steps_per_sec, 6),
                    "wall_seconds": measurement.wall_seconds,
                    "reported_steps_per_sec": measurement.reported_steps_per_sec,
                    "note": (
                        "wall_seconds is the whole process, including startup and the "
                        "twenty-sample inference pass. `reported_steps_per_sec` is what "
                        "the candidate claimed about itself; it is recorded so a reader "
                        "can see whether the two agree, and the harness's own figure is "
                        "the one that counts."
                    ),
                },
                "grad": {
                    "ratio": round(measurement.grad_ratio, 6),
                    "ratios": measurement.grad_ratios,
                    "analytic": measurement.grad_analytic,
                    "params": measurement.grad_params,
                    "band": [0.5, 2.0],
                    "note": (
                        "Directional derivative over all parameters against a central "
                        "difference, at two step sizes. See the probe's module docs and "
                        "docs/KNOWN-ISSUES.md issues 1 and 5."
                    ),
                },
                "gains": {
                    "loss": round(judged["loss_gain"], 6),
                    "speed": round(judged["speed_gain"], 6),
                },
                "verdict": judged,
                "model": {"id": spec.get("model"), "reasoning_effort": spec.get("reasoning_effort")},
                "usage": usage,
                "source_sha256": {
                    # The digest of what was *measured*, not of what is on disk.
                    # On a discard those differ -- the committed candidate is
                    # still the previous best -- and recording the wrong one
                    # would make `--verify-only` compare a run against code that
                    # never produced it.
                    track.source_key: (
                        sha256_file(crate / track.candidate_source)
                        if crate is not None
                        else ""
                    ),
                },
                "kept_as_candidate": bool(judged["kept"]),
            }

        write_json(RUNS_DIR / f"{run_id}.json", record)
        append_ledger(row)
        regenerate_best_diff(track)
        render()

        message = commit_message(
            run_id, row.status, row, baseline, measurement, judged,
            proposal.hypothesis if proposal else "", model_label, failure,
        )
        if args.push:
            sha = commit_and_push(
                ["autoresearch"], message, branch, True
            )
            print(f"  committed and pushed {sha[:12]}")
            if args.push_delay:
                time.sleep(args.push_delay)
        else:
            sha = commit_and_push(["autoresearch"], message, branch, False)
            print(f"  committed {sha[:12]} (not pushed; --no-push was implied)")
        completed_experiments += 1

    print(
        f"\n{completed_experiments} experiment(s), "
        f"{time.monotonic() - started:.0f}s elapsed"
    )
    return 0


# Scratch-apply helpers.
#
# The patch is applied to a throwaway copy of the crate and measured there. The
# committed candidate is replaced only on a keep, and only after the measurement
# has already been taken -- so a patch that turns out to be bad, slow or broken
# cannot damage the thing the next experiment starts from.

_SCRATCH_ROOTS: list[Path] = []


def link_dependencies(track: Track, project: Path) -> None:
    """Give a scratch copy the dependency tree a toolchain will not rebuild for it.

    `target/` and `dist/` are left out of the copy because the toolchains that
    produce them put them back on demand in the tree they were asked to build:
    `cargo build` and `tsc -p` do that as their first act. npm has no such step.
    Nothing in `build_track`, `lint_track`, `run_gradient_probe` or `train_once`
    installs anything, and `npx` resolves a binary it cannot find against the
    registry rather than against the project -- so a scratch tree with no
    `node_modules` is not a tree that takes a moment to get ready, it is a tree
    that cannot run its own compiler.

    So the directory is excluded and then linked back in. A symlink to the
    committed `autoresearch/candidate-ts/node_modules`, not a copy: it is 37 MB per
    experiment, the scratch tree is throwaway and has exactly two jobs -- typecheck
    and train -- and nothing is ever written into it, so a link costs nothing and
    copies nothing. Node resolves through a symlinked `node_modules` the way npm's
    own workspace links rely on: `node_modules/.bin/tsc` and `node_modules/.bin/tsx`
    are relative shims, so they land in the committed tree and find their packages
    there, and every `import` resolves out of the same place.

    The fallback is `npm ci` into the scratch tree itself -- slow, and it wants the
    network, so it is only for a checkout that has never installed -- and if it
    fails the run stops rather than continuing into a tree that cannot typecheck.
    Carrying on would put the harness back where it started: a compiler that is not
    there, reached by a path that reports itself as a candidate that does not
    compile.
    """
    if track.tool != "npm":
        return
    link = project / "node_modules"
    if link.is_symlink() or link.exists():
        return
    committed = track.candidate / "node_modules"
    if committed.is_dir():
        link.symlink_to(committed, target_is_directory=True)
        return
    result = run_command(
        ["npm", "ci", "--no-audit", "--no-fund"], cwd=project, timeout=600
    )
    if result.returncode != 0:
        raise ResearchError(
            f"error: there is no dependency tree for {track.name} and `npm ci` could"
            f" not build one in {project}\n{(result.stderr or result.stdout or '')[-2000:]}\n"
            f"  Run `npm ci` in {track.candidate.relative_to(REPO_ROOT)} and commit the"
            f" result, so every experiment can link the same tree it was written"
            f" against. Continuing without one would have `npx` fetch the npm package"
            f" named `tsc` -- a stub that is not the compiler -- and report its failure"
            f" as `tsc --noEmit failed`, which is a claim about the candidate and"
            f" would be a lie."
        )


def apply_to_scratch(track: Track, patch: str) -> tuple[bool, Path, str]:
    """Copy the candidate aside, apply the patch, return the project to measure.

    The scratch tree *mirrors the repository layout* -- `<work>/repo/autoresearch/
    candidate/` -- rather than putting the project at the top. `git apply` resolves
    the paths in a patch relative to its working directory, and the prompt asks for
    the canonical `a/autoresearch/candidate/src/lib.rs` form, so a flat scratch
    directory would make every patch fail to apply for a reason that has nothing to
    do with the patch. Mirroring means the usual `-p1` works and the model can write
    the paths it was asked to write.

    `target`, `dist` and `node_modules` are excluded, and the three are excluded for
    two different reasons, which is the rule worth stating: a directory the toolchain
    rebuilds on demand may be left out and forgotten, and a directory it does not
    rebuild must be put back. `cargo build` and `tsc -p` rebuild the first kind
    themselves, which is why `target/` is 700 MB of committed directory in
    `implementations/rust/` and costs this copy nothing. npm rebuilds nothing, so
    `node_modules` is excluded and then re-created by `link_dependencies` -- the
    first version of this function dropped it and said "`npx tsx` will rebuild or
    reuse what it needs", which is false: `npx` fetched the npm package named `tsc`,
    a stub that is not the compiler, and every TypeScript candidate was recorded as
    a crash with the reason `tsc --noEmit failed`.
    """
    work = Path(tempfile.mkdtemp(prefix="autoresearch-"))
    _SCRATCH_ROOTS.append(work)
    repo = work / "repo"
    project = repo / "autoresearch" / track.candidate_dir
    project.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        track.candidate,
        project,
        ignore=shutil.ignore_patterns("target", "node_modules", "dist"),
    )
    link_dependencies(track, project)
    applied, error = apply_patch(patch, repo)
    return applied, project, error


def commit_candidate(track: Track, source: Path) -> None:
    """Promote a measured scratch project to the committed candidate.

    Only the files that make up the candidate move. The probe is deliberately not
    among them: it is ours, and if a proposal somehow changed it, the digest check
    would already have failed the experiment.
    """
    if track.tool == "cargo":
        movable = ("Cargo.toml", "Cargo.lock", "src/lib.rs", "src/main.rs")
    elif track.tool == "go":
        movable = ("go.mod", "go.sum", track.candidate_source)
    elif track.tool == "cc":
        # `microgpt_simd.h` is here defensively rather than because the prompt
        # offers it: the model is shown one file and asked for one diff, but a
        # patch that does reach the header has already been built and measured,
        # so dropping it on the way back would leave the committed candidate
        # different from the thing that won.
        movable = ("microgpt_simd.h", track.candidate_source)
    else:
        movable = ("package.json", "package-lock.json", "tsconfig.json", track.candidate_source)
    for relative in movable:
        origin = source / relative
        if not origin.is_file():
            continue
        target = track.candidate / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)


def cleanup_scratch() -> None:
    for work in _SCRATCH_ROOTS:
        shutil.rmtree(work, ignore_errors=True)
    _SCRATCH_ROOTS.clear()


# ─── entry point ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autoresearch loop for the Rust, Go and TypeScript micro-gpt tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="loop",
        choices=("loop", "seed", "render", "verify"),
        help="loop (default): run experiments; seed: create the workspace and "
        "measure the baseline; render: regenerate results.json; verify: what CI runs",
    )
    parser.add_argument(
        "--track",
        default="rust",
        help="which implementation to optimise: "
        + ", ".join(SUPPORTED_TRACKS)
        + " (default rust)",
    )
    parser.add_argument("--experiments", type=int, default=1, help="how many experiments to run")
    parser.add_argument("--steps", type=int, default=STEPS, help=f"training steps (default {STEPS})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"PRNG seed (default {SEED})")
    parser.add_argument(
        "--repeats", type=int, default=REPEATS,
        help=f"timed runs for the speed axis (default {REPEATS}); the loss axis runs once "
        f"because it is deterministic at a fixed seed",
    )
    parser.add_argument(
        "--budget-seconds", type=float, default=0.0,
        help="stop after this long, checked between experiments (0 = no limit)",
    )
    parser.add_argument(
        "--branch", default="main",
        help="the branch the loop must be on, and the one it pushes to (default main)",
    )
    parser.add_argument(
        "--no-push", dest="push", action="store_false",
        help="commit locally but do not push; the site will not update",
    )
    parser.add_argument(
        "--push-delay", type=float, default=0.0,
        help="seconds to wait after each push. CI cancels in-progress runs on the same "
        "ref (concurrency: cancel-in-progress), so a burst of pushes leaves only the "
        "last one validated and deployed. 90 is a reasonable first run.",
    )
    parser.add_argument(
        "--skip-model-check", action="store_true",
        help="do not verify the pinned model against the OpenRouter catalogue",
    )
    parser.set_defaults(push=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "seed":
            return cmd_seed(args)
        if args.command == "render":
            return cmd_render(args)
        if args.command == "verify":
            return cmd_verify(args)
        return cmd_loop(args)
    except ResearchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\ninterrupted. Whatever was committed is committed; the candidate is only "
            "replaced on a keep, so stopping here is safe.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
