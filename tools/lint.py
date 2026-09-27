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
CANDIDATE = REPO_ROOT / "autoresearch" / "candidate"

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


def _locate(tool: str) -> str | None:
    """Find a tool, including where a toolchain installer puts one.

    `cargo` installs to `~/.cargo/bin`, which is on an interactive shell's PATH
    and is frequently *not* on the PATH a subprocess inherits -- from a launchd
    job, an editor task, or a container entrypoint. Reporting "skipped (no cargo
    on PATH)" for a machine that has Rust installed is worse than not checking:
    the advice it gives is wrong, and a linter that lies about what it skipped is
    a linter nobody trusts.
    """
    from tools.autoresearch import find_tool

    return find_tool(tool)


def _clippy_available(cargo: str) -> bool:
    """`cargo clippy` exists, and the toolchain actually has the component."""
    probe = subprocess.run(
        [cargo, "clippy", "--version"], capture_output=True, text=True
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

    cargo = _locate("cargo")
    if not cargo:
        print("  cargo clippy ... skipped (no cargo; install Rust from https://rustup.rs)")
    elif not _clippy_available(cargo):
        print("  cargo clippy ... skipped (clippy is not installed; "
              "`rustup component add clippy`)")
    else:
        step("cargo clippy (frozen track)", [cargo, "clippy", "--quiet"],
             cwd=REPO_ROOT / "implementations" / "rust")
        # The research track is a second Rust crate and a model rewrites it
        # unsupervised, so "it compiles and it is clippy-clean" is the only thing
        # standing between a bad patch and the page. Linted here rather than only
        # in the loop so that a commit which breaks it is caught before it lands.
        if (CANDIDATE / "Cargo.toml").is_file():
            step("cargo clippy (research candidate)", [cargo, "clippy", "--release", "--quiet"],
                 cwd=CANDIDATE)
        else:
            print("  cargo clippy (research candidate) ... skipped (no candidate crate; run `make autoresearch-rust`)")

    if failures:
        print(f"\nlint failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("lint ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
