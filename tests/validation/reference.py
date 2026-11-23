"""Reference grating, energies, PSD truths and cases for the validation suite."""

from __future__ import annotations

from functools import partial

import numpy as np

from cd_reduce.reduction.geometry import Geometry
from cd_reduce.reduction.roughness import (
    DevelopedLine,
    LatentFilm,
    psd_palasantzas,
    sigma_from_psd,
)

GEOMETRY = Geometry(
    u0_px=512, v0_px=512, Lb_px=1970, beta_deg=1.0, psi_deg=0.0, Lambda_nm=80
)
SHAPE = (1024, 1024)
ENERGIES = (280.0, 285.0, 288.0)
OMEGAS = np.arange(-20, 21, 2.0)
ADU_PER_PHOTON = 27.0
SCALE = 5e13

PSD_LPR = partial(psd_palasantzas, sigma_nm=1.0, xi_nm=25.0, hurst=0.7)
PSD_LWR = partial(psd_palasantzas, sigma_nm=1.5, xi_nm=12.0, hurst=0.5)
_PSD_E = partial(psd_palasantzas, sigma_nm=1.2, xi_nm=20.0, hurst=0.6)


def _psd_e_lpr(f):
    return 0.5 * _PSD_E(f)


def _psd_e_lwr(f):
    return 2.0 * _PSD_E(f)


_DN_A = {280.0: -1e-3 + 2e-4j, 285.0: -2e-3 + 1.5e-3j, 288.0: -1.2e-3 + 2.5e-3j}
_R = {280.0: 0.7 + 0.05j, 285.0: 1.15 + 0.05j, 288.0: 0.9 + 0.1j}


def latent(E: float) -> LatentFilm:
    """Latent-image film (relief plus index contrast) at energy ``E``."""
    return LatentFilm(80.0, 45.0, 2.5, 40.0, 4.0, 25.0, _R[E], _DN_A[E], energy_ev=E)


def developed(E: float) -> DevelopedLine:
    """Developed line (relief only), height 25 nm."""
    return DevelopedLine(80.0, 45.0, 2.5, 25.0, _DN_A[E], energy_ev=E)


def tapered(E: float):
    """Tapered developed line: 85 deg sidewall, h 25 nm, mean width 45 nm."""
    from .forward.tapered import tapered_from_sidewall

    return tapered_from_sidewall(80.0, 45.0, 2.5, 25.0, _DN_A[E], energy_ev=E)


def sigma_ler_full(psd_lpr, psd_lwr) -> float:
    """Debye-Waller edge rms: sqrt(sigma_LPR^2 + sigma_LWR^2 / 4)."""
    s2 = 0.0
    for psd, c in ((psd_lpr, 1.0), (psd_lwr, 0.25)):
        if psd is not None:
            s2 += c * sigma_from_psd(psd) ** 2
    return float(np.sqrt(s2))


CASES: dict[str, dict] = {
    "smooth": {"xs": latent, "psd_lpr": None, "psd_lwr": None},
    "lpr": {"xs": latent, "psd_lpr": PSD_LPR, "psd_lwr": None},
    "lwr": {"xs": latent, "psd_lpr": None, "psd_lwr": PSD_LWR},
    "both": {"xs": latent, "psd_lpr": PSD_LPR, "psd_lwr": PSD_LWR},
    "independent": {"xs": latent, "psd_lpr": _psd_e_lpr, "psd_lwr": _psd_e_lwr},
    "lpr_developed": {"xs": developed, "psd_lpr": PSD_LPR, "psd_lwr": None},
    "lwr_developed": {"xs": developed, "psd_lpr": None, "psd_lwr": PSD_LWR},
    "both_developed": {"xs": developed, "psd_lpr": PSD_LPR, "psd_lwr": PSD_LWR},
    "tapered": {"xs": tapered, "psd_lpr": PSD_LPR, "psd_lwr": PSD_LWR},
}


def small() -> dict:
    """Reduced config for the default validation run (same Q coverage)."""
    return {
        "omegas": np.array([-10.0, 0.0, 10.0]),
        "energies": (285.0,),
        "n_lines": 4,
        "n_real": 2,
        "shape": (512, 512),
        "geometry": Geometry(
            u0_px=256, v0_px=256, Lb_px=985, beta_deg=1.0, psi_deg=0.0, Lambda_nm=80
        ),
        # Born ensemble of explicit rough edges (Task 8)
        "ensemble_omegas": np.linspace(-10.0, 10.0, 5),
        "ly_nm": 1024.0,
        "dx_nm": 1.0,
        "dy_nm": 4.0,
        "dz_nm": 1.0,
        "ensemble_n_real": 2,
    }
