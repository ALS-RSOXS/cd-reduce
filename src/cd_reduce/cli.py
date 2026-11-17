"""Command-line interface for `cd-reduce`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from cd_reduce.loader import CdRsoxsLoader
from cd_reduce.simulation import (
    build_grating_2d,
    build_grating_3d,
    simulate_born_2d,
    simulate_born_3d,
    simulate_multislice_2d,
    simulate_multislice_3d,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the `cd-reduce` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cd-reduce",
        description=(
            "CD-RSoXS / CD-SAXS data reduction, Q-space mapping, "
            "and Born/Multislice simulation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # 1. Calibrate subcommand
    cal = sub.add_parser(
        "calibrate", help="Extract diffraction spots and calibrate geometry."
    )
    cal.add_argument("data_dir", type=Path, help="Directory containing FITS frames.")
    cal.add_argument(
        "--pitch-nm", type=float, default=100.0, help="Grating pitch in nm."
    )
    cal.add_argument(
        "--fit-walkoff", action="store_true", help="Jointly fit beam walk-off."
    )
    cal.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("geometry_calibration.json"),
        help="Output JSON path for calibrated geometry.",
    )

    # 2. Reduce / map subcommand
    red = sub.add_parser(
        "reduce", help="Reduce FITS frames to a 2D Q-map and/or 1D trace."
    )
    red.add_argument("data_dir", type=Path, help="Directory containing FITS frames.")
    red.add_argument(
        "-g", "--geometry", type=Path, required=True, help="Path to geometry JSON."
    )
    red.add_argument(
        "--pitch-nm", type=float, default=100.0, help="Grating pitch in nm."
    )
    red.add_argument(
        "--x-axis",
        type=str,
        default="Qx",
        help="Horizontal Q domain (Qx, Qy, Qz, Qr, signed_Qr, Q, omega, two_theta).",
    )
    red.add_argument(
        "--y-axis",
        type=str,
        default="Qz",
        help="Vertical Q domain (Qx, Qy, Qz, Qr, signed_Qr, Q, omega, two_theta).",
    )
    red.add_argument(
        "--mode",
        type=str,
        choices=["correct", "small_angle", "angle_linear"],
        default="correct",
        help="Reciprocal-space reduction mode.",
    )
    red.add_argument(
        "--background",
        type=str,
        choices=[
            "none",
            "corner",
            "row_col",
            "offline_band",
            "strip_fit",
            "photon_events",
        ],
        default="corner",
        help="Background subtraction method.",
    )
    red.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("qmap_2d.npz"),
        help="Output .npz path for the reduced 2D Q-map.",
    )

    # 3. Simulate subcommand
    sim = sub.add_parser(
        "simulate", help="Run 2D or 3D Born / Multislice grating simulation."
    )
    sim.add_argument(
        "--method",
        type=str,
        choices=["born_2d", "multislice_2d", "born_3d", "multislice_3d"],
        default="multislice_2d",
        help="Simulation method.",
    )
    sim.add_argument(
        "--pitch-nm", type=float, default=100.0, help="Grating pitch in nm."
    )
    sim.add_argument(
        "--cd-nm", type=float, default=45.0, help="Line critical dimension in nm."
    )
    sim.add_argument(
        "--thickness-nm", type=float, default=80.0, help="Grating thickness in nm."
    )
    sim.add_argument(
        "--energy-ev", type=float, default=285.0, help="Photon energy in eV."
    )
    sim.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("simulation_qmap.npz"),
        help="Output .npz path for the simulated (Qx, Qz) map.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `cd-reduce` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "calibrate":
        loader = CdRsoxsLoader(args.data_dir, pitch_nm=args.pitch_nm)
        res = loader.calibrate_geometry(fit_walkoff=args.fit_walkoff)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(res.geometry.to_dict(), indent=2), encoding="utf-8"
        )
        return 0

    if args.command == "reduce":
        loader = CdRsoxsLoader(
            args.data_dir,
            geometry=args.geometry,
            pitch_nm=args.pitch_nm,
            approximation=args.mode,
            bg_mode=args.background,
        )
        qmap = loader.to_qmap_2d(x_domain=args.x_axis, y_domain=args.y_axis)
        qmap.save_npz(args.output)
        return 0

    if args.command == "simulate":
        if args.method in ("born_2d", "multislice_2d"):
            struct2d = build_grating_2d(
                pitch_nm=args.pitch_nm,
                cd_nm=args.cd_nm,
                thickness_nm=args.thickness_nm,
                energy_ev=args.energy_ev,
            )
            res = (
                simulate_born_2d(struct2d)
                if args.method == "born_2d"
                else simulate_multislice_2d(struct2d)
            )
        else:
            struct3d = build_grating_3d(
                pitch_nm=args.pitch_nm,
                cd_nm=args.cd_nm,
                thickness_nm=args.thickness_nm,
                energy_ev=args.energy_ev,
            )
            res = (
                simulate_born_3d(struct3d)
                if args.method == "born_3d"
                else simulate_multislice_3d(struct3d)
            )
        res.qmap_qx_qz.save_npz(args.output)
        return 0

    return 0


__all__ = ["build_parser", "main"]
