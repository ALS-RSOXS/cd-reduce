"""Tests for `cd_reduce` CLI and package exports."""

from __future__ import annotations

from pathlib import Path

import cd_reduce
from cd_reduce.cli import main


def test_version_and_cli_help() -> None:
    assert hasattr(cd_reduce, "__version__")
    assert main([]) == 0


def test_cli_simulate(tmp_path: Path) -> None:
    out_path = tmp_path / "sim_map.npz"
    rc = main(
        [
            "simulate",
            "--method",
            "born_2d",
            "--pitch-nm",
            "100.0",
            "--cd-nm",
            "45.0",
            "--thickness-nm",
            "60.0",
            "-o",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
