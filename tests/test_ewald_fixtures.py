"""Pin the docs' interactive Ewald figure to the package geometry.

``docs/assets/ewald/fixtures.json`` holds rod-sphere intersections that the
figure (docs/javascripts/ewald-figure.js) evaluates with its own closed form.
This test checks that closed form against ``compute_pixel_q`` (the pixel on the
spot line where Qx = m G) and that the
stored fixtures are current. Regenerate with::

    uv run python tests/test_ewald_fixtures.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from cd_reduce.reduction.geometry import Geometry, compute_pixel_q

HC = 1239.84198
FIXTURES = Path(__file__).resolve().parents[1] / "docs/assets/ewald/fixtures.json"
CASES = [
    (250.0, 80.0, 0.0, 1),
    (250.0, 80.0, 15.0, -2),
    (285.0, 80.0, 15.0, 1),
    (285.0, 80.0, 15.0, 3),
    (285.0, 80.0, -30.0, 2),
    (285.0, 80.0, 35.0, -4),
    (285.0, 120.0, 0.0, -3),
    (500.0, 60.0, -30.0, 4),
    (500.0, 60.0, 35.0, 1),
    (500.0, 200.0, 15.0, -1),
    (285.0, 25.0, 35.0, 12),  # unreachable: m G > k (1 + sin w)
    (150.0, 20.0, 0.0, -10),  # unreachable
]
REGENERATE = "uv run python tests/test_ewald_fixtures.py --write"


def rod_qz(energy_ev: float, pitch_nm: float, omega_deg: float, m: int) -> float | None:
    """Closed form used by the docs figure (reduction convention)."""
    k = 2 * np.pi * energy_ev / HC
    g = 2 * np.pi / pitch_nm
    s, c = np.sin(np.radians(omega_deg)), np.cos(np.radians(omega_deg))
    disc = k * k - (m * g - k * s) ** 2
    return None if disc <= 0 else float(-k * c + np.sqrt(disc))


def build() -> dict:
    return {
        "hc": HC,
        "convention": "Qz = -k cos(w) + sqrt(k^2 - (m G - k sin w)^2), reduction frame",
        "cases": [
            {"E": e, "pitch": p, "omega": w, "m": m, "qz": rod_qz(e, p, w, m)}
            for e, p, w, m in CASES
        ],
    }


def package_qz(
    energy_ev: float, pitch_nm: float, omega_deg: float, m: int
) -> float | None:
    """Qz from the package: find the spot-line pixel where Qx = m G."""
    geo = Geometry(
        u0_px=511.5, v0_px=511.5, Lb_px=400.0, line_rot_deg=90.0, Lambda_nm=pitch_nm
    )
    target = m * 2 * np.pi / pitch_nm

    def q(s: float):
        return compute_pixel_q(
            geo, np.array([511.5]), np.array([511.5 + s]), energy_ev, omega_deg
        )

    def f(s: float) -> float:
        return float(q(s).Qx[0]) - target

    lo, hi = -5000.0, 5000.0
    if f(lo) * f(hi) > 0:
        return None
    s0 = brentq(f, lo, hi, xtol=1e-12)
    return float(q(s0).Qz[0])


@pytest.mark.parametrize(("energy", "pitch", "omega", "m"), CASES)
def test_closed_form_matches_package(energy, pitch, omega, m) -> None:
    qz = rod_qz(energy, pitch, omega, m)
    if qz is None:
        return
    pkg = package_qz(energy, pitch, omega, m)
    if pkg is None:
        pytest.skip("rod not reachable on the extended spot line")
    assert pkg == pytest.approx(qz, abs=1e-9)


def test_stored_fixtures_are_current() -> None:
    assert FIXTURES.exists(), f"missing {FIXTURES}; run: {REGENERATE}"
    stored = json.loads(FIXTURES.read_text())
    fresh = build()
    assert len(stored["cases"]) == len(fresh["cases"]), f"stale; run: {REGENERATE}"
    for a, b in zip(stored["cases"], fresh["cases"], strict=True):
        assert (a["E"], a["pitch"], a["omega"], a["m"]) == (
            b["E"],
            b["pitch"],
            b["omega"],
            b["m"],
        )
        if b["qz"] is None:
            assert a["qz"] is None, f"stale; run: {REGENERATE}"
        else:
            assert a["qz"] == pytest.approx(b["qz"], abs=1e-9), (
                f"stale; run: {REGENERATE}"
            )


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {FIXTURES}")
