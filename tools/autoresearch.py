#!/usr/bin/env python3
"""An autoresearch loop for the Rust micro-gpt track, on a laptop, in this repository.

## What this is

Descends from [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
(MIT, (c) Karpathy): give a model a small but real training setup, let it edit one
file, measure the result, keep the edit if it helped and throw it away if it did
not, and repeat. That project's whole point is that the *protocol* is fixed and
only the code moves, so two runs are comparable and the log is a real record of
what was tried.

Three things are different here, all of them forced by having no GPU and by this
repository already being a thing that asserts its own correctness.

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
pass is wrong by a factor of -8, and its loss curve *still* tracks the reference
to within 7%, and it *still* trains. A loss number cannot tell a correct gradient
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

Run `make autoresearch-rust`.
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
#: C port's broken gradient still achieves 17.6% windowed improvement, so this
#: catches divergence but is nowhere near sufficient on its own. The gradient
#: check is what catches a broken tape.
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

#: The digest of the gradient probe, checked before every experiment. See the
#: module docstring: a probe the candidate can weaken is not a probe.
#: `tools/tests/test_autoresearch.py` asserts this constant matches the file, so
#: the two cannot drift apart quietly.
PROBE_SHA256 = "2345e04de35305879243f69b72a992f93ad7fcf6175174d49c8adf14372aa287"

#: The only track. `--track` takes this and rejects anything else loudly, rather
#: than accepting a name it cannot honour.
SUPPORTED_TRACKS = ("rust",)

VERDICTS = ("keep", "discard", "crash")

# ─── layout ──────────────────────────────────────────────────────────────────

RESEARCH_DIR = REPO_ROOT / "autoresearch"
CANDIDATE_DIR = RESEARCH_DIR / "candidate"
CANDIDATE_LIB = CANDIDATE_DIR / "src" / "lib.rs"
PROBE_PATH = CANDIDATE_DIR / "tests" / "gradient_check.rs"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
RESULTS_JSON = RESEARCH_DIR / "results.json"
BASELINE_JSON = RESEARCH_DIR / "baseline.json"
MODEL_JSON = RESEARCH_DIR / "model.json"
PROGRAM_MD = RESEARCH_DIR / "program.md"
RUNS_DIR = RESEARCH_DIR / "runs"
DIFFS_DIR = RESEARCH_DIR / "diffs"
LOGS_DIR = RESEARCH_DIR / "logs"
BEST_DIFF = DIFFS_DIR / "best-vs-baseline.patch"

BASELINE_CRATE = REPO_ROOT / "implementations" / "rust"
BASELINE_LIB = BASELINE_CRATE / "src" / "lib.rs"
BASELINE_BIN = BASELINE_CRATE / "target" / "release" / "microgpt-rs"
DATASET = REPO_ROOT / "data" / "input.txt"

#: The ledger, as a TSV. Tab-separated because karpathy's is, and because
#: commas appear in these descriptions ("beta1 0.85 -> 0.9, no bias correction").
#: `PARSE` order is the column order and both readers derive from it.
COLUMNS = (
    "run_id",
    "parent",
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


def train_once(binary: Path, steps: int, seed: int, cwd: Path) -> tuple[list[float], float, float | None]:
    """One timed training run.

    The wall clock is the whole process, which includes interpreter-free startup
    and the twenty-sample inference pass at the end. That is deliberate and it is
    safe: both are small and both are constant, and -- the part that actually
    matters -- the baseline is measured the same way, in the same session, on the
    same machine. The comparison is therefore fair even though the absolute
    number is not a pure training time. It is recorded raw in the run record so a
    reader can check the overhead rather than take it on trust.
    """
    started = time.monotonic()
    completed = run_command(
        [str(binary), "--input", str(DATASET), "--steps", str(steps), "--seed", str(seed)],
        cwd=cwd,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-2500:]
        raise ResearchError(
            f"training run failed (exit {completed.returncode})\n{tail}"
        )
    losses = parse_losses(completed.stdout, steps)
    if not losses:
        raise ResearchError(
            "training run produced no `step N / M | loss X` lines. The harness "
            "reads that line, and so does tools/parity.py, which is why all five "
            "tracks print it. Whatever replaced it, put it back:\n"
            f"{(completed.stdout or '')[-1500:]}"
        )
    reported = REPORTED_SPEED_RE.search(completed.stdout)
    reported_sps = float(reported.group("sps")) if reported else None
    return losses, elapsed, reported_sps


def build_crate(cargo: str, crate_dir: Path) -> None:
    result = run_command(
        [cargo, "build", "--release", "--quiet", "--manifest-path", str(crate_dir / "Cargo.toml")],
        cwd=crate_dir,
    )
    if result.returncode != 0:
        raise ResearchError(f"cargo build failed\n{(result.stderr or '')[-3000:]}")


def lint_crate(cargo: str, crate_dir: Path) -> None:
    result = run_command(
        [cargo, "clippy", "--release", "--quiet", "--manifest-path", str(crate_dir / "Cargo.toml"),
         "--", "-D", "warnings"],
        cwd=crate_dir,
    )
    if result.returncode != 0:
        raise ResearchError(
            f"cargo clippy failed with -D warnings\n{(result.stderr or '')[-3000:]}"
        )


def run_gradient_probe(cargo: str, crate_dir: Path) -> dict[str, Any]:
    """The finite-difference gate. See the module docstring for why it exists and
    `autoresearch/candidate/tests/gradient_check.rs` for what it measures.

    The digest is checked against the *committed* probe, not the one in
    `crate_dir`. A candidate could otherwise ship its own weakened probe alongside
    its own weakened gradient, and the two would agree with each other.
    """
    actual = sha256_file(PROBE_PATH)
    if actual != PROBE_SHA256:
        raise ResearchError(
            f"error: the gradient probe has been modified.\n"
            f"  expected sha256 {PROBE_SHA256}\n"
            f"  found    sha256 {actual}\n"
            f"  {PROBE_PATH.relative_to(REPO_ROOT)} is written by this repository, not "
            f"by the research loop, and it is the only thing standing between a "
            f"model and a broken gradient that still trains "
            f"(docs/KNOWN-ISSUES.md issue 1). If you changed it on purpose, update "
            f"PROBE_SHA256 in tools/autoresearch.py and say why in the commit."
        )
    result = run_command(
        [cargo, "test", "--release", "--test", "gradient_check", "--", "--nocapture"],
        cwd=crate_dir,
    )
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


def measure_crate(crate: Path, steps: int = STEPS, seed: int = SEED, repeats: int = REPEATS) -> Measurement:
    """Build, lint, gradient-check and time one crate. Everything that has to be
    true before a loss number is allowed to mean anything.

    The same function measures the committed candidate and a scratch copy of it
    with a patch applied. That is the point: the checks a candidate has to pass
    to be considered at all are the checks it is judged by, so there is no way
    for a proposal to be scored on a weaker protocol than the baseline was
    measured on.
    """
    cargo = require_tool("cargo", "Install Rust from https://rustup.rs")
    for required in (DATASET, crate / "src" / "lib.rs", PROBE_PATH):
        if not required.is_file():
            raise ResearchError(
                f"error: {required} is missing. Run `python3 -m tools.autoresearch "
                f"--seed` to create the workspace."
            )
    build_crate(cargo, crate)
    lint_crate(cargo, crate)
    grad = run_gradient_probe(cargo, crate)

    binary = crate / "target" / "release" / "microgpt-tuned"
    if not binary.is_file():
        raise ResearchError(f"error: {binary} was not produced by the build")

    # One run for the loss axis -- it is deterministic at a fixed seed -- and
    # `repeats` for the speed axis, which is not.
    losses, first_elapsed, reported = train_once(binary, steps, seed, crate)
    wall = [first_elapsed]
    for _ in range(max(0, repeats - 1)):
        _, elapsed, _ = train_once(binary, steps, seed, crate)
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


def measure_candidate(steps: int = STEPS, seed: int = SEED, repeats: int = REPEATS) -> Measurement:
    return measure_crate(CANDIDATE_DIR, steps, seed, repeats)


def measure_baseline(steps: int = STEPS, seed: int = SEED, repeats: int = REPEATS) -> Measurement:
    """The frozen track, measured the same way, in the same session.

    The speed axis of `benchmarks/results.json` is not usable here and the reason
    is worth stating: it was measured on an Apple M1, and a ratio against a
    number from another machine is a number about that machine. So the comparator
    is rebuilt from source every session and thrown away with it.
    """
    cargo = require_tool("cargo", "Install Rust from https://rustup.rs")
    build_crate(cargo, BASELINE_CRATE)
    lint_crate(cargo, BASELINE_CRATE)
    if not BASELINE_BIN.is_file():
        raise ResearchError(f"error: {BASELINE_BIN} was not produced by the build")

    losses, first_elapsed, reported = train_once(BASELINE_BIN, steps, seed, BASELINE_CRATE)
    wall = [first_elapsed]
    for _ in range(max(0, repeats - 1)):
        _, elapsed, _ = train_once(BASELINE_BIN, steps, seed, BASELINE_CRATE)
        wall.append(elapsed)

    return Measurement(
        losses=losses,
        window=windowed(losses),
        steps_per_sec=steps / statistics.median(wall),
        wall_seconds=[round(value, 4) for value in wall],
        reported_steps_per_sec=reported,
        # The frozen crate has no probe -- the probe is part of the candidate,
        # not the baseline. The seeded candidate is the baseline's `lib.rs` plus
        # one method, so its ratio is the baseline tape's ratio, and
        # `measure_candidate` supplies it. Until it does, 0.0 means "not
        # measured" and the page says so.
        grad_ratio=0.0,
        grad_ratios={},
        grad_analytic=0.0,
        grad_params=0,
        log="",
    )


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


def regenerate_best_diff() -> dict[str, int]:
    """The patch from the frozen baseline to whatever the candidate currently is.
    This is the "admire how the code changed" artefact, and it is regenerated on
    every keep so it is always the current best against the original."""
    if not BASELINE_LIB.is_file() or not CANDIDATE_LIB.is_file():
        return {"added": 0, "removed": 0}
    patch = unified(
        BASELINE_LIB.read_text(encoding="utf-8"),
        CANDIDATE_LIB.read_text(encoding="utf-8"),
        "a/implementations/rust/src/lib.rs",
        "b/autoresearch/candidate/src/lib.rs",
    )
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIFF.write_text(patch, encoding="utf-8")
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
    "is wrong by a factor of -8 and its loss curve still tracks the reference to "
    "within 7% (docs/KNOWN-ISSUES.md issue 1). Every candidate is finite-difference "
    "checked before it competes; see `grad_ratio` on each row.",
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

    kept = [r for r in rows if r.status == "keep"]
    best = min(kept, key=lambda r: (r.loss, -r.steps_per_sec)) if kept else None

    baseline_row = next((r for r in rows if r.status == "baseline"), None)
    baseline_doc = read_json(BASELINE_JSON) if BASELINE_JSON.is_file() else {}

    return {
        "$comment": (
            "GENERATED by tools/autoresearch.py --render -- do not edit. The "
            "source of truth is autoresearch/results.tsv plus autoresearch/runs/*.json. "
            "CI regenerates this and fails on any diff, so a number on the site "
            "cannot stop matching the run that produced it."
        ),
        "generator": "tools/autoresearch.py --render",
        "track": "rust",
        "provenance_of": {
            "baseline_source": str(BASELINE_LIB.relative_to(REPO_ROOT)),
            "baseline_sha256": sha256_file(BASELINE_LIB) if BASELINE_LIB.is_file() else "",
            "candidate_source": str(CANDIDATE_LIB.relative_to(REPO_ROOT)),
            "candidate_sha256": sha256_file(CANDIDATE_LIB) if CANDIDATE_LIB.is_file() else "",
            "gradient_probe": str(PROBE_PATH.relative_to(REPO_ROOT)),
            "gradient_probe_sha256": sha256_file(PROBE_PATH) if PROBE_PATH.is_file() else "",
            "dataset_sha256": sha256_file(DATASET) if DATASET.is_file() else "",
        },
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
        "baseline": baseline_row.to_json() if baseline_row else None,
        "best": best.to_json() if best else None,
        "counts": {
            "experiments": len([r for r in rows if r.status != "baseline"]),
            "keep": len(kept),
            "discard": len([r for r in rows if r.status == "discard"]),
            "crash": len([r for r in rows if r.status == "crash"]),
        },
        "pareto": pareto_frontier(rows),
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


def ask_model(spec: dict[str, Any], prompt: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": spec["model"],
        "messages": [
            {"role": "system", "content": read_prompt()},
            {"role": "user", "content": prompt},
        ],
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


def read_prompt() -> str:
    if not PROGRAM_MD.is_file():
        raise ResearchError(
            f"error: {PROGRAM_MD.relative_to(REPO_ROOT)} is missing. It is the agent "
            f"brief and the one part of this system a human is meant to edit."
        )
    return PROGRAM_MD.read_text(encoding="utf-8")


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
    rows: list[Row],
    baseline: dict[str, Any],
    history: int = 15,
) -> str:
    """Everything the model gets. No diff of the previous attempt, no hidden
    state: the ledger is the memory, and it is a committed file."""
    recent = rows[-history:]
    ledger = "\n".join(
        f"{r.run_id}\t{r.status}\tloss={r.loss:.4f}\tsps={r.steps_per_sec:.1f}\t"
        f"dloss={r.loss_gain * 100:+.1f}%\tdspeed={r.speed_gain * 100:+.1f}%\t"
        f"grad={r.grad_ratio:.3f}\t{r.description}"
        for r in recent
    )
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
        f"`autoresearch/candidate/src/lib.rs`, reproduced here in full. It is "
        f"{len(CANDIDATE_LIB.read_text(encoding='utf-8').splitlines())} lines, and "
        f"this is all of it. Read it. Do not invent a signature, a method name or a "
        f"helper that is not in this text: a patch that references an API you "
        f"imagined does not apply, and the experiment is wasted.\n\n"
        f"```rust\n{CANDIDATE_LIB.read_text(encoding='utf-8')}\n```\n\n"
        f"Reply with exactly this shape and nothing else:\n\n"
        f"HYPOTHESIS: one sentence -- why you think this will help, stated so it "
        f"could be wrong\n"
        f"SUMMARY: one line, no commas needed, describing the change for the results "
        f"table\n\n"
        f"```diff\n"
        f"--- a/autoresearch/candidate/src/lib.rs\n"
        f"+++ b/autoresearch/candidate/src/lib.rs\n"
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
    lines = [f"research(rust): {status} {run_id} -- {row.description}", ""]
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


def measure_session_baseline(steps: int, seed: int, repeats: int) -> dict[str, Any]:
    """The comparator for this session, and its identity.

    Re-measured every session on purpose. The obvious optimisation -- reuse the
    committed benchmark number -- would be wrong: that number was taken on an
    Apple M1, and a speed ratio against a number from another machine is a number
    about that machine, not about this one.
    """
    frozen = measure_baseline(steps, seed, repeats)
    candidate = measure_candidate(steps, seed, repeats)
    payload = {
        "$comment": (
            "The session baseline: the frozen track, rebuilt and re-measured on the "
            "machine running the loop. Regenerated by every session and committed so "
            "the site can say what the numbers were relative to. See docs/AUTORESEARCH.md."
        ),
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
            "Measured on the seeded candidate, which is the frozen crate's lib.rs "
            "plus Tensor::set_data. The frozen crate carries no probe of its own; the "
            "tape is the same, so the ratio is the baseline's. 1.063 is expected and "
            "is docs/KNOWN-ISSUES.md issue 5, not an error in the measurement."
        ),
        "source_sha256": {
            "implementations/rust/src/lib.rs": sha256_file(BASELINE_LIB),
            "implementations/rust/Cargo.toml": sha256_file(BASELINE_CRATE / "Cargo.toml"),
        },
    }
    write_json(BASELINE_JSON, payload)
    return {
        "loss": payload["loss"],
        "steps_per_sec": payload["steps_per_sec"],
        "grad_ratio": payload["grad_ratio"],
        "window": payload["window"],
    }


# ─── subcommands ─────────────────────────────────────────────────────────────


def cmd_seed(args: argparse.Namespace) -> int:
    """Create the workspace and record the baseline row. Idempotent."""
    for path in (CANDIDATE_DIR, RUNS_DIR, DIFFS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.is_file():
        RESULTS_TSV.write_text(HEADER + "\n", encoding="utf-8")
    rows = read_ledger()
    if rows:
        print(f"{RESULTS_TSV.relative_to(REPO_ROOT)} already has {len(rows)} row(s); nothing to seed")
        regenerate_best_diff()
        render()
        return 0

    print("measuring the session baseline: rebuilding the frozen track and timing it")
    baseline = measure_session_baseline(args.steps, args.seed, args.repeats)
    print(
        f"  windowed loss {baseline['loss']:.4f}   "
        f"{baseline['steps_per_sec']:.1f} steps/s   "
        f"grad ratio {baseline['grad_ratio']:.4f}"
    )

    parent = git("rev-parse", "HEAD").strip()
    row = Row(
        run_id="0000",
        parent=parent,
        loss=baseline["loss"],
        steps_per_sec=baseline["steps_per_sec"],
        loss_gain=0.0,
        speed_gain=0.0,
        grad_ratio=baseline["grad_ratio"],
        status="baseline",
        reason="frozen implementations/rust, re-measured this session",
        description="the frozen track, unmodified, as this session's comparator",
    )
    record = {
        "$comment": "One experiment. GENERATED by tools/autoresearch.py -- do not edit.",
        "run_id": row.run_id,
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
            "note": "median of the session repeats; see baseline.json for the raw samples",
        },
        "grad": {
            "ratio": row.grad_ratio,
            "note": (
                "From the seeded candidate, which is this crate's lib.rs plus one "
                "method. See KNOWN-ISSUES.md issue 5 for why it is 1.063 and not 1.0."
            ),
        },
        "source_sha256": {"implementations/rust/src/lib.rs": sha256_file(BASELINE_LIB)},
    }
    write_json(RUNS_DIR / f"{row.run_id}.json", record)
    append_ledger(row)
    regenerate_best_diff()
    render()
    print(f"seeded {RESULTS_TSV.relative_to(REPO_ROOT)} with the baseline row")
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
    kept = [r for r in rows if r.status == "keep"]
    if not kept:
        print("no kept candidate recorded; nothing to verify")
        return 0
    best = min(kept, key=lambda r: (r.loss, -r.steps_per_sec))
    record_path = RUNS_DIR / f"{best.run_id}.json"
    record = read_json(record_path)

    print(f"verifying the current best candidate: experiment {best.run_id}")

    # 1. The recorded digest of the candidate must still be the committed file.
    recorded_digest = (record.get("source_sha256") or {}).get(
        "autoresearch/candidate/src/lib.rs"
    )
    actual_digest = sha256_file(CANDIDATE_LIB)
    if recorded_digest and recorded_digest != actual_digest:
        print(
            f"  ! the committed candidate no longer matches experiment {best.run_id}\n"
            f"    recorded sha256 {recorded_digest}\n"
            f"    on disk        sha256 {actual_digest}\n"
            f"    results.json claims a number this source no longer produces.",
            file=sys.stderr,
        )
        return 1
    print(f"  ok  candidate source matches experiment {best.run_id} (sha256 {actual_digest[:12]})")

    # 2. Build, lint, gradient check.
    measurement = measure_candidate(args.steps, args.seed, args.repeats)
    print(f"  ok  builds, clippy-clean, gradient ratio {measurement.grad_ratio:.4f}")

    # 3. The loss axis, within the tolerance a candidate is held to.
    drift = abs(measurement.window["last_window_mean"] - best.loss) / best.loss
    status = "ok" if drift <= VERIFY_LOSS_TOLERANCE else "FAIL"
    print(
        f"  {status}  windowed loss {measurement.window['last_window_mean']:.4f} vs recorded "
        f"{best.loss:.4f} ({drift * 100:+.2f}%, tolerance "
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
        f"{best.steps_per_sec:.1f}, and those are different machines: "
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
    if args.track not in SUPPORTED_TRACKS:
        raise ResearchError(
            f"error: --track {args.track!r} is not supported. This tool knows "
            f"about {', '.join(SUPPORTED_TRACKS)}.\n"
            f"  Adding a track is not a configuration change. Four things are wired to "
            f"the Rust track by name and each has to be answered for another "
            f"language:\n"
            f"    - the comparator, via BASELINE_CRATE / BASELINE_LIB / BASELINE_BIN, "
            f"and the binary's name;\n"
            f"    - the candidate, via CANDIDATE_DIR / CANDIDATE_LIB, and the binary "
            f"measure_crate() looks for;\n"
            f"    - the gradient probe, which needs a finite-difference entry point the "
            f"language can express. Several languages cannot check a gradient at all "
            f"without writing a reference implementation of the model's forward pass, "
            f"which is a much bigger job than adding a runner -- so \"no probe\" would "
            f"have to be a supported answer, and then the loss-only objection in the "
            f"module docstring applies with nothing to catch it;\n"
            f"    - the stdout contract, `step N / M | loss X`, which the harness parses "
            f"and which the parity gate already depends on.\n"
            f"  The parity gate in tools/parity.py would also need the track registered, "
            f"or the new track is unmeasured by the repository's own definition of "
            f"measured."
        )
    if not RESULTS_TSV.is_file() or not read_ledger():
        raise ResearchError(
            f"error: no baseline recorded. Run `python3 -m tools.autoresearch --seed` "
            f"first, or use `make autoresearch-rust` which does both."
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

    baseline = measure_session_baseline(args.steps, args.seed, args.repeats)
    print(
        f"session baseline: loss {baseline['loss']:.4f}  "
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
        print(f"\n[{run_id}] asking the model for experiment {completed_experiments + 1}")

        failure = ""
        proposal: Proposal | None = None
        usage: dict[str, Any] = {}
        try:
            text, usage = ask_model(
                spec, build_prompt(rows, baseline), int(spec.get("max_tokens", 32768))
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
            applied, crate, error = apply_to_scratch(proposal.patch)
            if not applied:
                failure = "the patch did not apply"
                detail = error.splitlines()[-1] if error else "no output from git apply"
                print(f"  crash: {failure}: {detail}")
            else:
                try:
                    measurement = measure_crate(crate, args.steps, args.seed, args.repeats)
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
                commit_candidate(crate)
            row = Row(
                run_id=run_id,
                parent=git("rev-parse", "HEAD").strip(),
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
                    "autoresearch/candidate/src/lib.rs": (
                        sha256_file(crate / "src" / "lib.rs")
                        if crate is not None
                        else ""
                    ),
                },
                "kept_as_candidate": bool(judged["kept"]),
            }

        write_json(RUNS_DIR / f"{run_id}.json", record)
        append_ledger(row)
        regenerate_best_diff()
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


def apply_to_scratch(patch: str) -> tuple[bool, Path, str]:
    """Copy the candidate aside, apply the patch, return the crate to measure.

    The scratch tree *mirrors the repository layout* -- `<work>/repo/autoresearch/
    candidate/` -- rather than putting the crate at the top. `git apply` resolves
    the paths in a patch relative to its working directory, and `program.md` asks
    for the canonical `a/autoresearch/candidate/src/lib.rs` form, so a flat
    scratch directory would make every patch fail to apply for a reason that has
    nothing to do with the patch. Mirroring means the usual `-p1` works and the
    model can write the paths it was asked to write.
    """
    work = Path(tempfile.mkdtemp(prefix="autoresearch-"))
    _SCRATCH_ROOTS.append(work)
    repo = work / "repo"
    crate = repo / "autoresearch" / "candidate"
    crate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CANDIDATE_DIR, crate, ignore=shutil.ignore_patterns("target"))
    applied, error = apply_patch(patch, repo)
    return applied, crate, error


def commit_candidate(source: Path) -> None:
    """Promote a measured scratch crate to the committed candidate.

    Only the three files that make up the crate move. The probe is deliberately
    not among them: it is ours, and if a proposal somehow changed it, the digest
    check would already have failed the experiment.
    """
    for relative in ("Cargo.toml", "src/lib.rs", "src/main.rs"):
        target = CANDIDATE_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def cleanup_scratch() -> None:
    for work in _SCRATCH_ROOTS:
        shutil.rmtree(work, ignore_errors=True)
    _SCRATCH_ROOTS.clear()


# ─── entry point ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autoresearch loop for the Rust micro-gpt track.",
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
    parser.add_argument("--track", default="rust", help="only 'rust' is supported")
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
