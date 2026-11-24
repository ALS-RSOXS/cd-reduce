#!/usr/bin/env python3
"""Decide whether the version on this checkout still needs a release.

Reads ``version`` from ``pyproject.toml`` and ``__version__`` from
``src/cd_reduce/__init__.py`` and exits non-zero when they differ. Reports
``release=true`` when the git tag ``vX.Y.Z`` does not exist on ``origin``.
With ``--github-output FILE`` the ``version`` and ``release`` keys are appended
in GitHub Actions output format.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "cd_reduce" / "__init__.py"

TOML_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
INIT_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def read_version(path: Path, pattern: re.Pattern[str]) -> str:
    """First version string in ``path`` matching ``pattern``."""
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"missing version in {path}")
    return match.group(1)


def tag_on_origin(tag: str) -> bool:
    """True when ``refs/tags/<tag>`` exists on the ``origin`` remote."""
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 2):  # 2 = no matching ref
        raise SystemExit(f"git ls-remote failed: {result.stderr.strip()}")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args()

    py_version = read_version(PYPROJECT, TOML_RE)
    init_version = read_version(INIT, INIT_RE)
    if py_version != init_version:
        raise SystemExit(
            f"version mismatch: pyproject.toml {py_version} vs "
            f"src/cd_reduce/__init__.py {init_version}"
        )

    release = not tag_on_origin(f"v{py_version}")
    print(f"version={py_version} release={str(release).lower()}")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write(f"version={py_version}\nrelease={str(release).lower()}\n")


if __name__ == "__main__":
    main()
