"""Two-sided PSD models in spatial frequency ``f = Qy / 2 pi`` [nm^-1].

All PSDs are in nm^3 and satisfy ``sigma^2 = int S df`` over both signs of ``f``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import gammaln

_TWO_PI = 2.0 * np.pi


def psd_palasantzas(
    f: ArrayLike, sigma_nm: float, xi_nm: float, hurst: float
) -> NDArray[np.float64]:
    """Palasantzas PSD normalised so that ``int S df = sigma_nm**2``."""
    p0 = (
        sigma_nm**2
        * _TWO_PI
        * xi_nm
        * np.exp(gammaln(hurst + 0.5) - gammaln(hurst))
        / np.sqrt(np.pi)
    )
    return psd_mack(f, p0, xi_nm, hurst)


def psd_mack(
    f: ArrayLike, p0: float, xi_nm: float, hurst: float, floor: float = 0.0
) -> NDArray[np.float64]:
    """Mack PSD ``S = p0 / (1 + (2 pi f xi)^2)^(H + 1/2) + floor``."""
    q = _TWO_PI * np.asarray(f, dtype=np.float64)
    return p0 / (1.0 + (q * xi_nm) ** 2) ** (hurst + 0.5) + floor


def psd_power_law(
    f: ArrayLike, amplitude: float, exponent: float
) -> NDArray[np.float64]:
    """Pure power law ``S = A |f|^(-exponent)``."""
    a = np.abs(np.asarray(f, dtype=np.float64))
    return amplitude * np.power(np.clip(a, 1e-12, None), -exponent)


def psd_spectral_peak(
    f: ArrayLike,
    p0: float,
    xi_nm: float,
    hurst: float,
    peak_amplitude: float,
    peak_q: float,
    peak_width: float,
) -> NDArray[np.float64]:
    """Mack PSD plus a Gaussian peak; ``peak_q`` and ``peak_width`` are in nm^-1 (f)."""
    a = np.asarray(f, dtype=np.float64)
    gauss = peak_amplitude * np.exp(-0.5 * ((np.abs(a) - peak_q) / peak_width) ** 2)
    return psd_mack(a, p0, xi_nm, hurst) + gauss


PSD_MODELS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "palasantzas": psd_palasantzas,
    "mack": psd_mack,
    "mack_floor": psd_mack,
    "power_law": psd_power_law,
    "spectral_peak": psd_spectral_peak,
}

_PARAMS: dict[str, tuple[str, ...]] = {
    "palasantzas": ("sigma_nm", "xi_nm", "hurst"),
    "mack": ("p0", "xi_nm", "hurst"),
    "mack_floor": ("p0", "xi_nm", "hurst", "floor"),
    "power_law": ("amplitude", "exponent"),
    "spectral_peak": (
        "p0",
        "xi_nm",
        "hurst",
        "peak_amplitude",
        "peak_q",
        "peak_width",
    ),
}


def sigma_from_psd(
    psd: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    f_lo: float = 0.0,
    f_hi: float = np.inf,
) -> float:
    """RMS ``sqrt(2 int_{f_lo}^{f_hi} S df)`` of a two-sided PSD callable [nm]."""
    val, _ = quad(
        lambda f: float(psd(np.asarray(f, dtype=np.float64))), f_lo, f_hi, limit=400
    )
    return float(np.sqrt(2.0 * val)) if val > 0 else float("nan")


@dataclass
class PsdFit:
    """Result of :func:`fit_psd_model`."""

    model: str
    params: dict[str, float]
    errors: dict[str, float]
    cost: float
    f_range: tuple[float, float]

    def __call__(self, f: ArrayLike) -> NDArray[np.float64]:
        return PSD_MODELS[self.model](f, *(self.params[n] for n in _PARAMS[self.model]))

    def sigma_nm(self, f_min: float = 0.0, f_max: float = np.inf) -> float:
        """RMS roughness from integrating the model (``floor`` excluded)."""
        params = {k: v for k, v in self.params.items() if k != "floor"}
        names = tuple(n for n in _PARAMS[self.model] if n != "floor")
        fn = PSD_MODELS[self.model]
        return sigma_from_psd(
            lambda f: fn(f, *(params[n] for n in names)), f_min, f_max
        )


def fit_psd_model(
    f: ArrayLike,
    S: ArrayLike,
    sigma_S: ArrayLike | None = None,
    model: str = "mack",
    *,
    p_init: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> PsdFit:
    """Fit a PSD model in log space with error weighting.

    Residuals are ``(log S_model - log S) * S / sigma_S`` so relative errors weight
    the fit. ``sigma_S`` defaults to 10 % of ``S``.
    """
    if model not in PSD_MODELS:
        raise ValueError(
            f"Unknown PSD model {model!r}; choose from {sorted(PSD_MODELS)}"
        )
    fn = PSD_MODELS[model]
    names = _PARAMS[model]
    q = np.asarray(f, dtype=np.float64)
    s = np.asarray(S, dtype=np.float64)
    e = (
        np.asarray(sigma_S, dtype=np.float64)
        if sigma_S is not None
        else 0.1 * np.abs(s)
    )
    ok = np.isfinite(q) & np.isfinite(s) & (s > 0) & np.isfinite(e) & (e > 0)
    q, s, e = q[ok], s[ok], e[ok]
    if q.size < len(names):
        raise ValueError("Not enough finite PSD points to fit the model.")

    fmed = float(np.median(q))
    var0 = abs(float(np.trapezoid(s, q))) * 2.0 if q.size > 1 else float(s[0])
    defaults: dict[str, float] = {
        "sigma_nm": float(np.sqrt(max(var0, 1e-12))),
        "p0": float(s[0]) * 1.2,
        "xi_nm": float(1.0 / (_TWO_PI * fmed)),
        "hurst": 0.5,
        "floor": float(np.min(s)) * 0.1,
        "amplitude": float(s[0] * q[0]),
        "exponent": 1.0,
        "peak_amplitude": float(np.max(s)) * 0.1,
        "peak_q": fmed,
        "peak_width": fmed * 0.2,
    }
    default_bounds: dict[str, tuple[float, float]] = {
        "sigma_nm": (1e-9, np.inf),
        "p0": (1e-12, np.inf),
        "xi_nm": (0.1, 1e5),
        "hurst": (0.0, 1.0),
        "floor": (0.0, np.inf),
        "amplitude": (1e-12, np.inf),
        "exponent": (0.0, 6.0),
        "peak_amplitude": (0.0, np.inf),
        "peak_q": (float(q.min()), float(q.max())),
        "peak_width": (1e-5, float(q.max())),
    }
    init = {**defaults, **(p_init or {})}
    bnd = {**default_bounds, **(bounds or {})}
    log_par = {"sigma_nm", "p0", "xi_nm", "amplitude", "peak_amplitude"}

    def _to(v: float, n: str) -> float:
        return float(np.log(v)) if n in log_par else float(v)

    def _from(v: float, n: str) -> float:
        return float(np.exp(v)) if n in log_par else float(v)

    x0 = np.array(
        [
            _to(
                np.clip(init[n], *bnd[n])
                if np.isfinite(bnd[n][1])
                else max(init[n], bnd[n][0]),
                n,
            )
            for n in names
        ]
    )
    lo = np.array(
        [_to(max(bnd[n][0], 1e-300), n) if n in log_par else bnd[n][0] for n in names]
    )
    hi = np.array(
        [
            _to(bnd[n][1], n)
            if (n in log_par and np.isfinite(bnd[n][1]))
            else bnd[n][1]
            for n in names
        ]
    )
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    def _res(x: NDArray[np.float64]) -> NDArray[np.float64]:
        pars = [_from(float(v), n) for v, n in zip(x, names, strict=True)]
        m = np.clip(fn(q, *pars), 1e-300, None)
        return (np.log(m) - np.log(s)) * s / e

    res = least_squares(_res, x0, bounds=(lo, hi))
    params = {n: _from(v, n) for v, n in zip(res.x, names, strict=True)}
    errors: dict[str, float] = {}
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac)
        cov *= 2.0 * res.cost / max(q.size - len(names), 1)
        for i, n in enumerate(names):
            sd = float(np.sqrt(max(cov[i, i], 0.0)))
            errors[n] = params[n] * sd if n in log_par else sd
    except np.linalg.LinAlgError:
        errors = dict.fromkeys(names, np.nan)
    return PsdFit(
        model=model,
        params=params,
        errors=errors,
        cost=float(res.cost),
        f_range=(float(q.min()), float(q.max())),
    )
