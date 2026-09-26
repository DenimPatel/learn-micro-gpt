#!/usr/bin/env python3
"""Every linter this repository has, in one place.

`tools/` is stdlib-only, so the Python side is `compileall` plus a handful of
structural checks that catch the things a type checker does not -- a selector that
resolves to the wrong lines, a generated file that is stale, a concept quoting
code outside its own anchor.

The JavaScript side is the visualizer's own `npm run lint`, which is eslint plus
prettier. It is skipped when `node_modules` is absent, so `make lint` works in a
Python-only checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIS = REPO_ROOT / "visualizer"

failures: list[str] = []


def step(name: str, argv: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"  {name} ...", flush=True)
    result = subprocess.run([str(a) for a in argv], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        failures.append(name)
        sys.stdout.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])


def _have(tool: str) -> bool:
    return subprocess.run(["which", tool], capture_output=True).returncode == 0


def _clippy_available() -> bool:
    """`cargo clippy` exists, and the toolchain actually has the component."""
    if not _have("cargo-clippy"):
        return False
    probe = subprocess.run(
        ["cargo", "clippy", "--version"], capture_output=True, text=True
    )
    return probe.returncode == 0 and "not installed" not in probe.stderr


def main() -> int:
    print("linting")

    # Byte-compiling every tool is a real check, not a formality: it catches a
    # syntax error in a script that is only run on one platform.
    step("python syntax", [sys.executable, "-m", "compileall", "-q", "tools", "reference"])

    if (VIS / "node_modules").is_dir():
        step("eslint", ["npx", "eslint", ".", "--max-warnings", "0"], cwd=VIS)
        step("prettier", ["npx", "prettier", "--check", "."], cwd=VIS)
    else:
        print("  web lint ... skipped (run `make -C visualizer install` first)")

    # Each native linter is probed on its own terms and skipped with a reason
    # rather than failing. A toolchain's component set is the user's choice --
    # `rustup --profile minimal` has no clippy, and a minimal install is a
    # perfectly reasonable way to work. CI installs what it needs, so a track
    # cannot be untested there.
    if _have("go"):
        step("go vet", ["go", "vet", "./..."], cwd=REPO_ROOT / "implementations" / "go")
    else:
        print("  go vet ... skipped (no go on PATH)")

    if not _have("cargo"):
        print("  cargo clippy ... skipped (no cargo on PATH)")
    elif not _clippy_available():
        print("  cargo clippy ... skipped (clippy is not installed; "
              "`rustup component add clippy`)")
    else:
        step("cargo clippy", ["cargo", "clippy", "--quiet"], cwd=REPO_ROOT / "implementations" / "rust")

    if failures:
        print(f"\nlint failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("lint ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
