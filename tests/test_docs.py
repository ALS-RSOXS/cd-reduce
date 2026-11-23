"""Docs reference only names that exist in the package."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import cd_reduce

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = (
    "theory/roughness_psd.md",
    "theory/born_and_roughness.md",
    "guides/mapping/roughness_psd.md",
    "guides/mapping/slicing_and_1d.md",
    "guides/mapping/index.md",
    "guides/simulation/rough_gratings_3d.md",
)
_DOTTED = re.compile(r"\bcd_reduce((?:\.\w+)+)")
_ALIAS = re.compile(r"\bcdr\.(\w+)")
_CALL = re.compile(r"`(\w+)\(")
# Owners of bare `name(` references that are methods rather than functions.
_OWNERS = (
    cd_reduce,
    cd_reduce.CdRsoxsLoader,
    cd_reduce.PsdRecord,
    cd_reduce.RoughEnsembleResult,
    cd_reduce.BornSimulationResult3D,
)


def _resolve(dotted: str) -> bool:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for name in parts[i:]:
            if not hasattr(obj, name):
                return False
            obj = getattr(obj, name)
        return True
    return False


@pytest.mark.parametrize("page", PAGES)
def test_docs_reference_existing_api(page: str) -> None:
    text = (DOCS / page).read_text()
    missing = sorted(
        {
            "cd_reduce" + m
            for m in _DOTTED.findall(text)
            if not _resolve("cd_reduce" + m)
        }
        | {m for m in _ALIAS.findall(text) if not hasattr(cd_reduce, m)}
        | {m for m in _CALL.findall(text) if not any(hasattr(o, m) for o in _OWNERS)}
    )
    assert not missing, f"{page} references missing names: {missing}"
