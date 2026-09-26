#!/usr/bin/env python3
"""Run a track, record a trace, emit JSONL.

## Why the reference is run hermetically

`reference/microgpt.py` resolves `input.txt` from the *current working
directory* and downloads it from a mutable URL if it is missing. So this tool
copies `data/input.txt` into a throwaway directory and execs the script there.
The reference file stays byte-identical -- CI pins its sha256 -- and the run
stays reproducible. Nothing here edits it; the point is to run the real file,
exactly as a learner would.

## How it records without editing

Two sources, because they need different machinery:

- **The loss curve and the samples come from stdout.** The reference already
  prints one line per step and one per sample. Parsing that is exact, and it
  needs no cleverness.
- **Attention weights and the top-k probabilities need instrumentation.** The
  only way to see a local variable in a script you are not allowed to edit is to
  rewrite its AST. `build_instrumented_source` injects a call to `__emit`
  immediately after any assignment to a watched name, and nothing else changes.

The useful trick: in this script `step`, `pos_id`, and `h` are all *module-level*
globals, because the whole thing runs at module level. So `__emit` can read them
out of `globals()` and know the step, the position, and the head without any of
them being threaded through the injected call. That is why the injection is four
lines rather than a framework.

## Output

`traces/<lang>/<config>/{meta.json,steps.jsonl,attn.jsonl,probs.jsonl,samples.jsonl}`
-- see `docs/TRACE-FORMAT.md`. Regenerate with `make trace`; CI runs
`make trace-check`, which regenerates and fails on any diff. That is what stops
a committed trace from quietly ceasing to describe the code it came from.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRACES_DIR = REPO_ROOT / "traces"
DATASET = REPO_ROOT / "data" / "input.txt"
REFERENCE = REPO_ROOT / "reference" / "microgpt.py"

TRACE_VERSION = 1
TOOL_VERSION = "tools/trace.py@1"

#: Positions and heads recorded in detail. At block_size=16 and n_head=4 these
#: are kilobytes. Recording every position and head would be megabytes of noise
#: nobody reads. This set is chosen to make the causal structure visible: the
#: diagonal is the self-match, and 0/1/3/7/15 show the triangle filling in.
RECORD_POSITIONS = (0, 1, 3, 7, 15)
RECORD_HEADS = (0, 1)
RECORD_STEPS = (0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
TOP_K = 8

STEP_RE = re.compile(r"^step\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\s+([0-9.]+)\s*$")
SAMPLE_RE = re.compile(r"^sample\s+(\d+):\s*(.*)$")

#: Assignments worth instrumenting. `loss` is parsed from stdout instead, so it
#: is not in this list -- the printed value and `loss.data` are the same number
#: and parsing the printed one is simpler.
WATCH_NAMES = ("attn_weights", "probs")

TRACE_FILES = ("meta.json", "steps.jsonl", "attn.jsonl", "probs.jsonl", "samples.jsonl")

#: Written for debugging a drift mismatch, gitignored: it is exactly what
#: steps.jsonl and samples.jsonl were parsed from, so committing both would give
#: the drift check two things to disagree about.
DEBUG_ONLY_FILES = ("stdout.txt",)


# ----------------------------------------------------------------------------
# instrumentation
# ----------------------------------------------------------------------------

PREAMBLE = f'''
# Injected in memory by {TOOL_VERSION}. Not part of reference/microgpt.py, which
# stays byte-identical to upstream and is pinned by sha256 in CI.
import sys as _sys

_WATCH = {WATCH_NAMES!r}
_POSITIONS = {list(RECORD_POSITIONS)!r}
_HEADS = {list(RECORD_HEADS)!r}
_STEPS = {list(RECORD_STEPS)!r}
_TOP_K = {TOP_K}
_TRACE = {{"attn": [], "probs": []}}


def _num(v):
    """Value -> plain float/list, unwrapping the autograd Value type."""
    if hasattr(v, "data"):
        return float(v.data)
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _emit(name, value):
    """Record one watched assignment, with the loop counters around it.

    The counters are read out of the *calling frame* rather than passed in. That
    matters because they are not all module globals: `step` and `pos_id` are,
    since the training loop runs at module level, but `h` is a local of `gpt()`.
    Reading the caller's locals gets all three, at any nesting depth, without the
    AST rewrite having to thread a single extra argument -- and it keeps working
    if somebody indents the code differently.

    The injected call is a single statement: `_emit('attn_weights', attn_weights)`.
    """
    frame = _sys._getframe(1)
    scope = frame.f_locals
    scope.update(frame.f_globals)
    st = scope.get("step", -1)
    pos = scope.get("pos_id", -1)
    head = scope.get("h", -1)
    if st not in _STEPS:
        return

    if name == "attn_weights" and pos in _POSITIONS and head in _HEADS:
        # `attn_weights` is a flat list of T scalars -- one weight per position
        # seen so far -- not a list of rows. The reference's softmax returns
        # `list[Value]`, so each element is a single probability. Recording a 2D
        # array here would be inventing a dimension the model does not have: the
        # query is one position, and the only axis is position.
        flat = [_num(w) for w in value]
        if any(p is None for p in flat):
            return
        _TRACE["attn"].append(
            {{
                "type": "attn",
                "step": st,
                "pos": pos,
                "head": head,
                "weights": [round(float(p), 6) for p in flat],
                "sum": round(float(sum(flat)), 6),
            }}
        )
    elif name == "probs" and pos in _POSITIONS:
        flat = [_num(p) for p in value]
        if not flat or any(p is None for p in flat):
            return
        uchars = scope.get("uchars", [])
        order = sorted(range(len(flat)), key=lambda i: -flat[i])[:_TOP_K]
        _TRACE["probs"].append(
            {{
                "type": "probs",
                "step": st,
                "pos": pos,
                "top": [
                    {{
                        "token": (uchars[i] if i < len(uchars) else "?"),
                        "p": round(float(flat[i]), 6),
                    }}
                    for i in order
                ],
            }}
        )
'''


def build_instrumented_source(source: str) -> str:
    """Return `source` with `__emit` calls injected after watched assignments.

    Uses `ast.unparse`, so formatting and comments in the original are not
    preserved in the instrumented copy. That does not matter: this text is never
    written to disk, never displayed, and never hashed. The file on disk is
    untouched.
    """
    import ast

    tree = ast.parse(source)

    class Injector(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if name not in _WATCH_SET:
                    continue
                call = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="_emit", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=name),
                            ast.Name(id=name, ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                )
                ast.copy_location(call, node)
                node.__dict__["_emit_call"] = call
            return node

    # Mark first, then splice. Running these in the other order finds nothing to
    # splice, because the marker is only attached by the Injector.
    Injector().visit(tree)

    for node in ast.walk(tree):
        for field_name in ("body", "orelse", "finalbody"):
            block = getattr(node, field_name, None)
            if not isinstance(block, list):
                continue
            expanded: list[Any] = []
            for child in block:
                expanded.append(child)
                call = child.__dict__.pop("_emit_call", None)
                if call is not None:
                    expanded.append(call)
            setattr(node, field_name, expanded)

    ast.fix_missing_locations(tree)
    return PREAMBLE + "\n" + ast.unparse(tree)


_WATCH_SET = set(WATCH_NAMES)


# ----------------------------------------------------------------------------
# running the reference
# ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_hyperparameters(path: Path = REFERENCE) -> dict[str, Any]:
    """Read the reference's hyperparameters from the file rather than asserting them.

    Hardcoding them here would be a claim nobody checks. Parsing the assignments
    means a change to `n_embd` shows up as a diff in the trace metadata instead of
    as silently wrong shape annotations in the documentation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {
        "n_embd", "n_head", "n_layer", "block_size", "head_dim",
        "learning_rate", "beta1", "beta2", "eps_adam", "temperature", "num_steps",
    }
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            names: list[str] = []
            if isinstance(target, ast.Name):
                names = [target.id]
            elif isinstance(target, (ast.Tuple, ast.List)):
                names = [e.id for e in target.elts if isinstance(e, ast.Name)]
            hit = [n for n in names if n in wanted]
            if not hit:
                continue
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                value = ast.unparse(node.value)
            if len(hit) == 1:
                found[hit[0]] = value
            elif isinstance(value, (list, tuple)) and len(value) == len(hit):
                found.update(zip(hit, value))
    _derive(found)
    return found


def _derive(found: dict[str, Any]) -> None:
    """Resolve hyperparameters the reference defines in terms of other ones.

    `head_dim = n_embd // n_head` is a real number the docs quote in every shape
    annotation. Leaving it as the source text `'n_embd // n_head'` would be
    honest but useless in a machine-readable trace, so it is evaluated -- over
    literals only, with no names other than ones already resolved, which is the
    whole reason this is not just `eval`.
    """
    import ast  # noqa: F811 - local so the module surface stays small

    for _ in range(4):
        changed = False
        for key, value in list(found.items()):
            if not isinstance(value, str) or "/" not in value and "*" not in value:
                continue
            try:
                resolved = eval(  # noqa: S307 - literals only, see docstring
                    compile(ast.Expression(ast.parse(value, mode="eval").body), "<hp>", "eval"),
                    {"__builtins__": {}},
                    {k: v for k, v in found.items() if isinstance(v, (int, float))},
                )
            except Exception:
                continue
            if isinstance(resolved, (int, float)):
                found[key] = resolved
                changed = True
        if not changed:
            return


DRIVER = '''
import json, sys, time

reference, instrumented_path, payload_path = sys.argv[1], sys.argv[2], sys.argv[3]

started = time.perf_counter()
code = compile(open(instrumented_path, encoding="utf-8").read(), reference, "exec")
namespace = {"__name__": "__main__", "__file__": reference}
exec(code, namespace)
elapsed = time.perf_counter() - started

with open(payload_path, "w", encoding="utf-8") as handle:
    json.dump({"elapsed": elapsed, "trace": namespace["_TRACE"]}, handle)
'''


def run_reference(steps: int, seed: int, work: Path) -> tuple[str, dict[str, Any], float]:
    """Run the real reference file, instrumented, in `work`. Returns (stdout, trace, seconds)."""
    source = REFERENCE.read_text(encoding="utf-8")
    instrumented = work / "_instrumented_microgpt.py"
    instrumented.write_text(build_instrumented_source(source), encoding="utf-8")
    driver = work / "_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    payload = work / "_trace.json"

    completed = subprocess.run(
        [sys.executable, str(driver), str(REFERENCE), str(instrumented), str(payload)],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-6000:])
        raise SystemExit(f"trace run failed (exit {completed.returncode})")
    data = json.loads(payload.read_text(encoding="utf-8"))
    return completed.stdout, data["trace"], float(data["elapsed"])


# ----------------------------------------------------------------------------
# parsing stdout
# ----------------------------------------------------------------------------


def parse_stdout(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Pull the loss curve and the samples out of the reference's own output."""
    steps: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    total = 0
    for line in text.split("\n"):
        match = STEP_RE.match(line.strip())
        if match:
            index, total, loss = int(match.group(1)), int(match.group(2)), float(match.group(3))
            steps.append({"type": "step", "step": index - 1, "loss": loss})
            continue
        match = SAMPLE_RE.match(line.strip())
        if match:
            samples.append(
                {
                    "type": "sample",
                    "index": int(match.group(1)) - 1,
                    "text": match.group(2).strip(),
                }
            )
    return steps, samples, total


# ----------------------------------------------------------------------------
# analysis helpers, shared with tools/parity.py
# ----------------------------------------------------------------------------


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Exponential moving average, seeded with the first value.

    The per-step loss is the loss on *one document*, so its standard deviation is
    about 16% of its mean and it moves upwards on half of all steps. Comparing
    raw per-step values between two implementations compares noise. Smoothing
    with a ~20-step effective window keeps the trend and removes the
    document-to-document variance. `docs/BENCHMARKS.md` has the arithmetic.
    """
    if not values:
        return []
    out: list[float] = []
    current = values[0]
    for value in values:
        current = alpha * value + (1 - alpha) * current
        out.append(current)
    return out


def window_mean(values: list[float], lo: int, hi: int) -> float:
    if not values:
        return float("nan")
    hi = min(hi, len(values))
    lo = min(lo, hi)
    window = values[lo:hi]
    if not window:
        return float("nan")
    return sum(window) / len(window)


def loss_statistics(losses: list[float]) -> dict[str, float]:
    """The summary a loss curve needs before anyone compares it to anything."""
    if not losses:
        return {}
    mean = sum(losses) / len(losses)
    variance = sum((v - mean) ** 2 for v in losses) / len(losses)
    upward = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
    smoothed = ema(losses)
    return {
        "steps": len(losses),
        "mean": round(mean, 6),
        "stdev": round(variance**0.5, 6),
        "min": round(min(losses), 6),
        "max": round(max(losses), 6),
        "final": round(losses[-1], 6),
        "first": round(losses[0], 6),
        "upward_moves": upward,
        "total_moves": max(1, len(losses) - 1),
        "ema_alpha": 0.05,
        "ema_first50": round(window_mean(smoothed, 0, 50), 6),
        "ema_step200": round(smoothed[min(199, len(smoothed) - 1)], 6),
        "ema_final50": round(window_mean(smoothed, len(smoothed) - 50, len(smoothed)), 6),
    }


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def record(config: str, steps: int, seed: int, quiet: bool = False) -> Path:
    """Record one Python trace. Returns the output directory."""
    for required in (REFERENCE, DATASET):
        if not required.is_file():
            raise SystemExit(f"{required} is missing")

    hyper = read_hyperparameters()
    declared_steps = hyper.get("num_steps")
    if declared_steps is not None and declared_steps != steps:
        # We do not patch num_steps -- the file is byte-pinned -- so we record
        # what it actually says rather than quietly running a different number
        # of steps than the trace claims.
        if not quiet:
            print(
                f"  note: reference/microgpt.py declares num_steps={declared_steps}, "
                f"not {steps}; recording all of them",
                file=sys.stderr,
            )
        steps = int(declared_steps)

    out_dir = TRACES_DIR / "python" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="microgpt-trace-"))
    try:
        shutil.copy(DATASET, work / "input.txt")
        if not quiet:
            print(f"recording python/{config}: {steps} steps, seed {seed}")
            print(f"  in {work}")
        stdout, trace, elapsed = run_reference(steps, seed, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    loss_rows, sample_rows, reported_total = parse_stdout(stdout)
    losses = [row["loss"] for row in loss_rows]

    if not losses:
        raise SystemExit("no loss lines in the reference output; the format changed?")

    for row in sample_rows:
        row["step"] = reported_total
        row.pop("index", None)

    stats = loss_statistics(losses)
    meta = {
        "trace_version": TRACE_VERSION,
        "tool_version": TOOL_VERSION,
        "lang": "python",
        "config": config,
        "steps": len(losses),
        "seed": seed,
        "dataset": {
            "path": "data/input.txt",
            "sha256": sha256_file(DATASET),
            "docs": sum(1 for line in DATASET.read_text(encoding="utf-8").split("\n") if line.strip()),
        },
        "reference": {
            "path": "reference/microgpt.py",
            "sha256": sha256_file(REFERENCE),
            "note": "byte-identical to upstream; executed unmodified in a scratch CWD",
        },
        "hyperparameters": hyper,
        "hyperparameter_source": "parsed from reference/microgpt.py at record time",
        "selection": {
            "positions": list(RECORD_POSITIONS),
            "heads": list(RECORD_HEADS),
            "steps": list(RECORD_STEPS),
            "top_k": TOP_K,
        },
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "timing": {
            "wall_seconds": round(elapsed, 3),
            "honesty": (
                "CPython on the host, not a benchmark. The recorded throughput "
                "in benchmarks/results.json is the only number this repository "
                "treats as a speed measurement; see docs/BENCHMARKS.md."
            ),
        },
        "loss_stats": stats,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "steps.jsonl", loss_rows)
    write_jsonl(out_dir / "samples.jsonl", sample_rows)
    write_jsonl(out_dir / "attn.jsonl", trace.get("attn", []))
    write_jsonl(out_dir / "probs.jsonl", trace.get("probs", []))
    (out_dir / "stdout.txt").write_text(stdout, encoding="utf-8")

    total_bytes = sum((out_dir / name).stat().st_size for name in TRACE_FILES)
    if not quiet:
        print(
            f"  {len(loss_rows)} steps, {len(sample_rows)} samples, "
            f"{len(trace.get('attn', []))} attn, {len(trace.get('probs', []))} probs "
            f"({total_bytes / 1024:.1f} KiB total)"
        )
        print(
            f"  loss: mean {stats['mean']:.4f}  sd {stats['stdev']:.4f}  "
            f"ema(0.05) {stats['ema_first50']:.4f} -> {stats['ema_final50']:.4f}  "
            f"upward moves {stats['upward_moves']}/{stats['total_moves']}"
        )
        print(f"  wrote {out_dir.relative_to(REPO_ROOT)}/")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a trace for a track.")
    parser.add_argument("--lang", default="python")
    parser.add_argument("--config", default="micro")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all", action="store_true", help="record every configured track"
    )
    args = parser.parse_args(argv)

    if args.lang != "python":
        print(
            f"error: no runner for track {args.lang!r}. Every track must satisfy the "
            f"contract in docs/ADDING-A-LANGUAGE.md.",
            file=sys.stderr,
        )
        return 1

    record(args.config, args.steps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
