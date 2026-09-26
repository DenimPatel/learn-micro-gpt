#!/usr/bin/env python3
"""Statistical parity across tracks. This is the gate the plan asked for, and it
is not the gate the plan described.

## What the plan specified, and why it cannot work

The original specification was: *loss strictly decreasing, and within +/-10% of
the Python reference at single steps 50 and 200.*

Both halves are unsatisfiable **by the reference itself**:

- The training loop reads `doc = docs[step % len(docs)]` -- one document per step,
  no batching. So the printed loss is the loss on that one name, and the printed
  value is a single sample, not an estimate of anything. Measured over the
  reference's own 1,000 steps:

  | statistic | value |
  | --- | --- |
  | mean | 2.4517 |
  | standard deviation | 0.3920 |
  | min (step 537) | 1.5782 |
  | max | 3.9066 |
  | **step-to-step moves that increase** | **500 of 999** |

  Half of all steps go up. "Strictly decreasing" is not a property this program
  has. Asking a port to reproduce it would be asking it to reproduce noise.

- The per-step standard deviation is 16% of the mean, and the spread between the
  best and worst step is 95% of it. A +/-10% band on a **raw single-step** value
  is inside the noise floor of the reference's own run, so it would fail about
  half the time even for a bit-identical implementation.

Bit-exact parity is impossible anyway: the reference seeds Python's Mersenne
Twister with `random.gauss` while every other track uses xoshiro or PCG, and
Python uses float64 where the C port uses float32. So the question is not "do
the numbers match" but "do the two runs learn the same thing".

## What this checks instead

1. **Trend.** The mean loss over the last 50 steps must be meaningfully below
   the mean over the first 50. The reference drops 0.528 nats (18.5%); the C
   port drops 0.522 (17.6%). The gate requires at least `MIN_RELATIVE_IMPROVEMENT`
   of that, so a port that trains but learns nothing useful still fails.

2. **Smoothed agreement.** An exponential moving average (alpha 0.05, a ~20-step
   effective window) removes the document-to-document variance while keeping the
   trend. At the checkpoints, each track must sit within `BAND` of the
   reference's smoothed value. Measured Python vs C:

   | step | python EMA | C EMA | delta | relative |
   | --- | --- | --- | --- | --- |
   | 50 | 2.7778 | 2.8808 | +0.1030 | +3.7% |
   | 100 | 2.6960 | 2.6633 | -0.0328 | -1.2% |
   | 200 | 2.4924 | 2.5991 | +0.1067 | +4.3% |
   | 500 | 2.4525 | 2.5073 | +0.0548 | +2.2% |
   | 1000 | 2.3026 | 2.4628 | +0.1602 | +7.0% |

   A +/-10% band on the *smoothed* value has real headroom, which is what makes
   it a meaningful gate rather than a coin flip. The 10% figure is kept from the
   plan; what changed is what it is applied to.

3. **Structural invariants.** For every step, loss must be finite and positive.
   An implementation that diverges produces NaN or infinity, which no band check
   would catch cleanly.

Run via `make parity`. Exit code 0 means every track learns like the reference.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.trace import TRACES_DIR, ema, loss_statistics, window_mean  # noqa: E402

#: Relative band on the smoothed loss at each checkpoint. Measured worst case
#: between the reference and the C port is 7.0% at step 1000, so 10% is a real
#: bound with headroom rather than a number chosen to pass.
BAND = 0.10

#: A track must improve its windowed mean loss by at least this fraction. The
#: reference achieves 18.5% and the C port 17.6%, so this leaves room for a
#: differently-seeded implementation while still failing a broken one.
MIN_RELATIVE_IMPROVEMENT = 0.10

#: Checkpoints, as 0-based step indices. Deliberately including step 0: a track
#: that starts in a different place is telling you something real.
CHECKPOINTS = (0, 49, 99, 199, 499, 999)

TREND_WINDOW = 50
EMA_ALPHA = 0.05


# ----------------------------------------------------------------------------
# running a track
# ----------------------------------------------------------------------------


class TrackRun:
    """A track's loss curve, plus whatever the runner wanted to say about it."""

    def __init__(self, lang: str, config: str, losses: list[float], meta: dict[str, Any] | None = None):
        self.lang = lang
        self.config = config
        self.losses = losses
        self.meta = meta or {}
        self.smoothed = ema(losses, EMA_ALPHA)

    @property
    def name(self) -> str:
        return f"{self.lang}/{self.config}"

    def at(self, step: int) -> float:
        index = min(step, len(self.smoothed) - 1)
        return self.smoothed[index]

    def trend(self) -> dict[str, float]:
        first = window_mean(self.losses, 0, TREND_WINDOW)
        last = window_mean(self.losses, max(0, len(self.losses) - TREND_WINDOW), len(self.losses))
        improvement = (first - last) / first if first else 0.0
        return {
            "first_window_mean": round(first, 6),
            "last_window_mean": round(last, 6),
            "absolute_drop": round(first - last, 6),
            "relative_improvement": round(improvement, 6),
        }


def load_committed_trace(lang: str, config: str) -> TrackRun | None:
    """Read a track's loss curve from its committed trace, if there is one."""
    steps_path = TRACES_DIR / lang / config / "steps.jsonl"
    if not steps_path.is_file():
        return None
    losses = []
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "step":
            losses.append(float(row["loss"]))
    if not losses:
        return None
    meta_path = steps_path.with_name("meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return TrackRun(lang, config, losses, meta)


def run_c(config: str, source: Path, steps: int) -> TrackRun | None:
    """Build and run a C track, returning its parsed loss curve."""
    import os
    import re
    import shutil
    import subprocess
    import tempfile

    binary = REPO_ROOT / "implementations" / "c" / f"microgpt-{config}"
    flags = ["-O3", "-Wall", "-Wno-unused-function"]
    link: list[str] = []
    if sys.platform == "darwin":
        flags.append("-mcpu=apple-m1")
        link = ["-framework", "Accelerate"]
    build = subprocess.run(
        [*_cc(), *flags, "-o", str(binary), str(source), "-lm", *link],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(build.stderr[-3000:], file=sys.stderr)
        return None

    work = Path(tempfile.mkdtemp(prefix="microgpt-parity-"))
    try:
        shutil.copy(REPO_ROOT / "data" / "input.txt", work / "input.txt")
        run = subprocess.run(
            [str(binary)], cwd=work, capture_output=True, text=True, timeout=900
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if run.returncode != 0:
        print(run.stderr[-3000:], file=sys.stderr)
        return None

    losses = []
    for line in run.stdout.split("\n"):
        match = re.match(r"^step\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\s+([0-9.]+)\s*$", line.strip())
        if match:
            losses.append(float(match.group(3)))
    return TrackRun("c", config, losses[:steps] or losses)


def _cc() -> list[str]:
    import os
    import shutil as _shutil

    return [_shutil.which("cc") or _shutil.which("gcc") or "cc"]


# ----------------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------------


class ParityResult:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def ok(self) -> bool:
        return not self.failures


def check_finite(run: TrackRun, result: ParityResult) -> bool:
    bad = [i for i, v in enumerate(run.losses) if not math.isfinite(v) or v <= 0]
    if bad:
        result.failures.append(
            f"{run.name}: {len(bad)} step(s) have a non-finite or non-positive loss "
            f"(first at step {bad[0]}). The run diverged; every other check below "
            f"would be meaningless."
        )
        return False
    return True


def check_trend(run: TrackRun, result: ParityResult) -> None:
    trend = run.trend()
    if trend["relative_improvement"] < MIN_RELATIVE_IMPROVEMENT:
        result.failures.append(
            f"{run.name}: windowed loss only improved "
            f"{trend['relative_improvement'] * 100:.1f}% "
            f"(first {TREND_WINDOW} steps mean {trend['first_window_mean']:.4f}, "
            f"last {TREND_WINDOW} steps mean {trend['last_window_mean']:.4f}). "
            f"Required at least {MIN_RELATIVE_IMPROVEMENT * 100:.0f}%. "
            f"Note the per-step curve is noisy by design -- the reference itself "
            f"moves upwards on half of all steps -- so this compares window means, "
            f"not individual steps."
        )


def check_band(reference: TrackRun, run: TrackRun, result: ParityResult) -> None:
    for step in CHECKPOINTS:
        if step >= len(reference.losses) or step >= len(run.losses):
            continue
        expected = reference.at(step)
        actual = run.at(step)
        if expected == 0:
            continue
        delta = (actual - expected) / abs(expected)
        row = {
            "lang": run.lang,
            "config": run.config,
            "step": step,
            "reference_ema": round(expected, 6),
            "track_ema": round(actual, 6),
            "relative_delta": round(delta, 6),
            "within_band": abs(delta) <= BAND,
        }
        result.rows.append(row)
        if abs(delta) > BAND:
            result.failures.append(
                f"{run.name} at step {step}: smoothed loss {actual:.4f} is "
                f"{delta * 100:+.1f}% from the reference's {expected:.4f} "
                f"(band is +/-{BAND * 100:.0f}%)."
            )


def compare(reference: TrackRun, run: TrackRun) -> ParityResult:
    result = ParityResult()
    if not check_finite(run, result):
        return result
    check_trend(run, result)
    check_band(reference, run, result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Statistical parity across tracks.")
    parser.add_argument(
        "--steps", type=int, default=1000, help="checkpoint horizon for fresh runs"
    )
    parser.add_argument(
        "--committed-only",
        action="store_true",
        help="only check tracks that have a committed trace; do not build or run C",
    )
    parser.add_argument("--json", type=Path, help="write the full result as JSON")
    args = parser.parse_args(argv)

    reference = load_committed_trace("python", "micro")
    if reference is None:
        print(
            "error: no committed trace for python/micro. Run `make trace` first -- "
            "parity is measured against the reference's recorded run, not against "
            "a re-derivation of it.",
            file=sys.stderr,
        )
        return 1

    print("reference: python/micro")
    stats = loss_statistics(reference.losses)
    trend = reference.trend()
    print(
        f"  {stats['steps']} steps  mean {stats['mean']:.4f}  sd {stats['stdev']:.4f}  "
        f"upward moves {stats['upward_moves']}/{stats['total_moves']}"
    )
    print(
        f"  trend: first {TREND_WINDOW} mean {trend['first_window_mean']:.4f} -> "
        f"last {TREND_WINDOW} mean {trend['last_window_mean']:.4f}  "
        f"({trend['relative_improvement'] * 100:+.1f}%)"
    )
    print(
        f"  note: the per-step curve rises on {stats['upward_moves']} of "
        f"{stats['total_moves']} steps. This is one document per step with no "
        f"batching, so it is a sample, not an estimate."
    )
    print()

    tracks: list[TrackRun] = []
    if not args.committed_only:
        c_run = run_c("micro", REPO_ROOT / "implementations" / "c" / "microgpt.c", args.steps)
        if c_run is None:
            print("error: could not build or run the C parity track", file=sys.stderr)
            return 1
        tracks.append(c_run)
    for config_dir in sorted((TRACES_DIR).glob("*/*")):
        if config_dir.name == "micro" or config_dir.parent.name == "python":
            continue
        committed = load_committed_trace(config_dir.parent.name, config_dir.name)
        if committed:
            tracks.append(committed)

    if not tracks:
        print(
            "no other tracks to compare. The C parity track is built and run by "
            "this command unless --committed-only is passed.",
        )
        return 0

    all_failures: list[str] = []
    all_rows: list[dict[str, Any]] = []
    for run in tracks:
        print(f"checking {run.name}  ({len(run.losses)} steps)")
        result = compare(reference, run)
        trend = run.trend()
        print(
            f"  trend {trend['relative_improvement'] * 100:+.1f}%  "
            f"({'ok' if not any('windowed loss' in f for f in result.failures) else 'FAIL'})"
        )
        for row in result.rows:
            mark = "ok  " if row["within_band"] else "FAIL"
            print(
                f"    {mark} step {row['step']:>4}: {row['track_ema']:.4f} vs "
                f"{row['reference_ema']:.4f}  ({row['relative_delta'] * 100:+.1f}%)"
            )
        all_failures.extend(result.failures)
        all_rows.extend(result.rows)
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "band": BAND,
                    "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
                    "ema_alpha": EMA_ALPHA,
                    "checkpoints": list(CHECKPOINTS),
                    "reference": {
                        "name": "python/micro",
                        "stats": loss_statistics(reference.losses),
                        "trend": reference.trend(),
                    },
                    "comparisons": all_rows,
                    "failures": all_failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    if all_failures:
        print(f"{len(all_failures)} parity failure(s):\n", file=sys.stderr)
        for failure in all_failures:
            print(f"  ! {failure}", file=sys.stderr)
        return 1

    print(
        f"parity ok: {len(tracks)} track(s) within +/-{BAND * 100:.0f}% of the "
        f"reference's smoothed loss and improving by at least "
        f"{MIN_RELATIVE_IMPROVEMENT * 100:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
