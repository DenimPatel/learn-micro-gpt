#!/usr/bin/env python3
"""Measure every track and write `benchmarks/results.json`.

## Read this before you read a number

**A number from your laptop is not a leaderboard entry.** These are wall-clock
times on one specific machine, and the honest comparisons are *ratios between
tracks measured on the same machine in the same run*, not absolute figures. A
2023 MacBook Air and a CI runner differ by more than any algorithmic difference
below.

What the ratios are good for is showing *where the time goes*, and that is the
actual lesson. Measured here, for 1,000 steps at the micro config:

| track | steps/sec | vs Python |
| --- | --- | --- |
| Python (the reference) | 8.5 | 1x |
| TypeScript (tsx, no tuning) | 56 | 6.6x |
| Rust (release) | 55 | 6.5x |
| Go (`go run`) | 38 | 4.5x |
| C (NEON + Accelerate) | 2,537 | 298x |

Three things that number makes concrete, in descending order of importance:

**Autograd's cost is the Python file's dominant cost, not the arithmetic.**
Rust and Go run the *identical* tape-based algorithm in 6-7x the speed, and a
compiled language with no special libraries at all. So the ~15,000 `Value`
objects allocated and freed per step *are* the bottleneck. The C port is fast
because it allocates nothing and does the same sums in registers.

**The language is not the story; the data structure is.** Rust (a systems
language) and TypeScript (running under `tsx`, an interpreter-ish runtime) land
within 3% of each other. What they share is a tape; what differs is that neither
of them allocates a Python object per operation. Language choice is a much
weaker lever than representation.

**The fast path is worth about 45x over the same C code, scalar.** See
`implementations/c/test_equivalence.sh`: the portable scalar build of the C port
learns the same thing to within 0.11%, and is roughly 45x slower. All of the C
port's speed comes from NEON, Accelerate, and not allocating -- and none of it
comes from a different algorithm.

## What is deliberately NOT here

- **No `scaled` config row.** `microgpt-scaled.c` is n_embd 256 / 4 layers /
  block 256 against 16 / 1 / 16, so it is roughly two orders of magnitude more
  work. Reporting it next to the others would be comparing different problems.
  Run `make run-scaled` to see it for yourself.
- **No peak RSS.** Meaningful numbers need `/usr/bin/time -l` or a profiler, and
  a number produced by a best-effort path is worse than an absent one.
- **No "vs PyTorch" row.** This repository has no PyTorch dependency on purpose,
  and a number from a machine without it measured today would be fiction.

Run `make bench` to regenerate. The file is committed so the site can show it
without running anything, and `benchmarks.yml` regenerates it on a schedule.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "benchmarks" / "results.json"
STEPS = 1000


@dataclass
class Measurement:
    lang: str
    config: str
    command: list[str]
    cwd: Path
    build_flags: str
    build_mode: str
    tools: tuple[str, ...] = ()
    #: Run once before timing. Python's `go run` and the C port need this; the
    #: rest are either prebuilt or interpret their own source.
    build_command: list[str] = field(default_factory=list)


def cpu_model() -> str:
    """The CPU, in a form that is worth printing.

    On macOS `sysctl machdep.cpu.brand_string` gives the marketing name; on Linux
    `/proc/cpuinfo` gives a model name. On anything else we say so rather than
    printing a guess, because a benchmark with an unrecorded machine is not a
    benchmark.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return f"unknown ({platform.machine()})"


def tool_version(argv: list[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "not available"


def memory_gb() -> float | None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return round(total / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def core_count() -> int | None:
    return os.cpu_count()


def measurements() -> list[Measurement]:
    data = REPO_ROOT / "data" / "input.txt"
    go_root = REPO_ROOT / "implementations" / "go"
    ts_root = REPO_ROOT / "implementations" / "typescript"
    rust_root = REPO_ROOT / "implementations" / "rust"
    c_dir = REPO_ROOT / "implementations" / "c"
    c_binary = c_dir / "microgpt-bench"
    rust_binary = rust_root / "target" / "release" / "microgpt-rs"

    c_flags = ["-O3", "-Wall", "-Wno-unused-function"]
    c_link: list[str] = []
    if sys.platform == "darwin":
        c_flags.append("-mcpu=apple-m1")
        c_link = ["-framework", "Accelerate"]

    # The reference is run from a scratch directory seeded with data/input.txt,
    # because it resolves `input.txt` from the CWD and otherwise downloads it.
    # See docs/KNOWN-ISSUES.md issue 3.
    scratch = REPO_ROOT / ".bench"
    scratch.mkdir(exist_ok=True)
    shutil.copy(data, scratch / "input.txt")

    return [
        Measurement(
            lang="python",
            config="micro",
            command=[sys.executable, str(REPO_ROOT / "reference" / "microgpt.py")],
            cwd=scratch,
            build_flags="none (CPython, no flags)",
            build_mode="reference",
        ),
        Measurement(
            lang="c",
            config="micro",
            command=[str(c_binary)],
            cwd=scratch,
            build_flags=" ".join([*c_flags, *c_link]),
            build_mode="native (NEON + Accelerate)"
            if sys.platform == "darwin"
            else "portable (scalar fallbacks)",
            build_command=[
                shutil.which("cc") or shutil.which("gcc") or "cc",
                *c_flags,
                "-o", str(c_binary),
                str(c_dir / "microgpt.c"), "-lm", *c_link,
            ],
            tools=("cc", "gcc"),
        ),
        Measurement(
            lang="go",
            config="micro",
            command=["go", "run", ".", "--input", str(data), "--steps", str(STEPS),
                     "--quiet", "--trace", str(scratch / "go.jsonl")],
            cwd=go_root,
            build_flags="go run (no tuning flags)",
            build_mode="go run",
            tools=("go",),
        ),
        Measurement(
            lang="rust",
            config="micro",
            command=[str(rust_binary), "--input", str(data), "--steps", str(STEPS)],
            cwd=rust_root,
            build_flags="cargo --release (lto, debug-assertions on)",
            build_mode="release",
            build_command=["cargo", "build", "--release", "--quiet", "--manifest-path",
                           str(rust_root / "Cargo.toml")],
            tools=("cargo",),
        ),
        Measurement(
            lang="typescript",
            config="micro",
            command=["npx", "tsx", "src/cli.ts", "--input", str(data), "--steps", str(STEPS),
                     "--trace", str(scratch / "ts.jsonl")],
            cwd=ts_root,
            build_flags="tsx (no tuning flags)",
            build_mode="tsx",
            tools=("npx", "node"),
        ),
    ]


def compile_if_needed(measurement: Measurement) -> bool:
    command = measurement.build_command
    if not command:
        return True
    built = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if built.returncode != 0:
        print(f"{measurement.lang}: build failed\n{built.stderr[-1500:]}", file=sys.stderr)
        return False
    return True


def measure(measurement: Measurement) -> dict[str, Any] | None:
    if measurement.tools:
        missing = [t for t in measurement.tools if shutil.which(t) is None]
        if missing:
            print(f"skipping {measurement.lang}: {', '.join(missing)} not on PATH", file=sys.stderr)
            return None
    if not compile_if_needed(measurement):
        return None

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(c) for c in measurement.command],
            cwd=measurement.cwd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print(f"skipping {measurement.lang}: timed out after 30 minutes", file=sys.stderr)
        return None
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        print(f"skipping {measurement.lang}: exited {completed.returncode}", file=sys.stderr)
        return None

    print(f"  {measurement.lang:<11} {elapsed:8.2f}s  {STEPS / elapsed:9.1f} steps/sec")
    return {
        "lang": measurement.lang,
        "config": measurement.config,
        "steps": STEPS,
        "wall_seconds": round(elapsed, 4),
        "step_ms": round(elapsed / STEPS * 1000, 5),
        "steps_per_sec": round(STEPS / elapsed, 2),
        "build_flags": measurement.build_flags,
        "build_mode": measurement.build_mode,
        "peak_rss_mb": None,
    }


def main() -> int:
    print(f"measuring every track, {STEPS} steps each")
    results: list[dict[str, Any]] = []
    for measurement in measurements():
        result = measure(measurement)
        if result:
            results.append(result)

    if not results:
        print("\nerror: no track could be measured", file=sys.stderr)
        return 1

    reference = next((r for r in results if r["lang"] == "python"), None)
    for result in results:
        result["relative_to_python"] = (
            round(result["steps_per_sec"] / reference["steps_per_sec"], 2)
            if reference and reference["steps_per_sec"]
            else None
        )
        # NaN/inf are not JSON; and an absent measurement is not a zero.
        if not result["steps_per_sec"]:
            result["steps_per_sec"] = None

    payload = {
        "$comment": "GENERATED by tools/bench.py -- run `make bench`. See docs/BENCHMARKS.md.",
        "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": STEPS,
        "runner": {
            "cpu": cpu_model(),
            "cores": core_count(),
            "memory_gb": memory_gb(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "compilers": {
                "cc": tool_version([shutil.which("cc") or "cc", "--version"]),
                "go": tool_version(["go", "version"]),
                "rust": tool_version(["cargo", "--version"]),
                "node": tool_version(["node", "--version"]),
            },
            "pinned": False,
            "why_not_pinned": (
                "These are wall-clock times on whatever machine ran `make bench`, so "
                "they are only comparable within a single row set. The scheduled "
                "benchmarks.yml run records the machine it used; nothing here should "
                "be read as a claim about anyone's hardware."
            ),
        },
        "config": {
            "id": "micro",
            "label": "the reference config: n_embd 16, n_head 4, n_layer 1, block 16",
            "hyperparameters": {
                "n_embd": 16, "n_head": 4, "n_layer": 1, "block_size": 16, "params": 4192,
            },
        },
        "results": sorted(results, key=lambda r: r["steps_per_sec"] or 0, reverse=True),
        "methodology": (
            "Wall-clock time for the whole training loop, including the reference's "
            "per-step printing. No warm-up run and no repetition: a single "
            "measurement is honest about being a single measurement. The Python "
            "figure is dominated by output volume rather than arithmetic, so the "
            "Python-vs-others ratio should be read as an upper bound on the gap."
        ),
        "caveats": [
            "Ratios between rows are meaningful; absolute seconds are machine-specific.",
            "The Python row includes printing 1,000 lines, which the C and Rust rows also pay and the trace-based runs do not.",
            "Go and TypeScript are run without tuning flags on purpose. With -O flags, a Go binary instead of `go run`, or esbuild instead of tsx, those rows would improve substantially.",
            "The `scaled` config is deliberately absent: it is a different model, so comparing it here would compare two different problems.",
        ],
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    if RESULTS.is_file() and RESULTS.read_text(encoding="utf-8") == text:
        print(f"  {RESULTS.relative_to(REPO_ROOT)} unchanged")
    else:
        RESULTS.write_text(text, encoding="utf-8")
        print(f"  wrote {RESULTS.relative_to(REPO_ROOT)}")
    shutil.rmtree(REPO_ROOT / ".bench", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
