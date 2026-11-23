"""Model error with the multislice engine, and the multislice/Born ratio report.

The ratio and sigma_win criteria of `test_model_error` are rerun for both
engines on the two-mode cases (same realisations per seed, thresholds
unchanged). Multislice frames carry the film absorption themselves
(``A_eng = 1`` in `render_ensemble_frame`), so they test the HOWTO absorption
factor ``A`` as well as the first-order model.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from cd_reduce.reduction.roughness import absorption_factor

from . import reference as ref
from .forward.ensemble_frames import frame_omega_deg
from .test_model_error import RESULTS, _check_ratio, _check_sigma_window

pytestmark = pytest.mark.validation

ENGINES = ("born", "multislice")
MS_CASES = ("both", "both_developed")
# Modes whose Born-epsilon diffuse is below this fraction of its maximum (per
# omega) are left out of the MS/Born median (near nulls the ratio is noisy).
REPORT_LEVEL = 0.01


@pytest.fixture(params=ENGINES)
def engine(request):
    return request.param


@pytest.mark.parametrize("check", ["ratio", "sigma_window"])
@pytest.mark.parametrize("case", MS_CASES)
def test_model_error_multislice(ensembles, engine, case, check):
    if check == "ratio":
        _check_ratio(ensembles, case, engine)
    else:
        _check_sigma_window(ensembles, case, engine)


def ms_over_born(ensembles, case: str) -> dict[str, dict[str, float]]:
    """Per-omega median of multislice / (Born x A) over the diffuse modes.

    Two references on the same realisations: the epsilon-scaled multislice
    (``born_eps``) and the Born engine. Modes are pooled over the K seeds and
    restricted to the first-order cut ``|Qx| sigma_e <= 0.3`` (R19), ``qy != 0``
    and ``born_eps >= REPORT_LEVEL`` x its maximum. Keys are detector omegas.
    """
    xs = ref.CASES[case]["xs"](285.0)
    geo = ensembles.cfg["geometry"]
    qx_cut = ensembles.qx_max(case, pooled=True)
    out: dict[str, dict[str, float]] = {"epsilon": {}, "born_engine": {}}
    per = {"epsilon": [], "born_engine": []}
    for s in ensembles.seeds(True):
        ms = ensembles.result(case, s, "multislice")
        born = ensembles.result(case, s, "born")
        W = np.deg2rad(ms.omega_deg)[:, None, None]
        cos_out = np.cos(W) / ms.flux_factor
        A = absorption_factor(xs.mu_h, W, cos_out)
        eps = ms.metadata["born_eps"]
        sel = (
            ms.valid
            & (np.abs(ms.qx)[None, :, None] <= qx_cut)
            & (ms.qy != 0)[None, None, :]
            & np.isfinite(eps)
            & (eps >= REPORT_LEVEL * np.nanmax(eps, axis=(1, 2), keepdims=True))
        )
        per["epsilon"].append(np.where(sel, ms.diffuse / (eps * A), np.nan))
        per["born_engine"].append(
            np.where(sel, ms.diffuse / (born.diffuse * A), np.nan)
        )
    for ref_name, arrs in per.items():
        r = np.concatenate(arrs, axis=2)
        for i, om in enumerate(frame_omega_deg(ms.omega_deg, geo)):
            out[ref_name][str(float(om))] = round(float(np.nanmedian(r[i])), 4)
    return out


def test_ms_over_born_report(ensembles):
    """Report only: median MS/(Born x A) per omega (285 eV); warns outside 5%."""
    report = {"energy_ev": 285.0, "cases": {}}
    for case in MS_CASES:
        res = ms_over_born(ensembles, case)
        report["cases"][case] = res
        for ref_name, by_om in res.items():
            for om, v in by_om.items():
                assert np.isfinite(v), f"{case} {ref_name} omega {om}: {v}"
                if not 0.95 <= v <= 1.05:
                    warnings.warn(
                        f"MS/(Born x A) {case} ({ref_name}) omega {om}: {v}",
                        stacklevel=1,
                    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ms_over_born.json").write_text(json.dumps(report, indent=1))
