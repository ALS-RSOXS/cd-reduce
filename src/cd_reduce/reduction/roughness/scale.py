"""Absolute scale and Debye-Waller sigma from Bragg order efficiencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from cd_reduce.reduction.roughness.cross_section import CrossSection
from cd_reduce.reduction.roughness.reduce import absorption_factor

_HC_EV_NM = 1239.84198
_DEFAULT_ORDERS = (-4, -3, -2, -1, 1, 2, 3, 4)


def _debye_waller(m: ArrayLike, L: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """``exp(-m^2 G^2 sigma^2)`` with ``G = 2 pi / L``."""
    return np.exp(
        -(np.asarray(m, dtype=float) ** 2)
        * (2 * np.pi / np.asarray(L, dtype=float)) ** 2
        * np.asarray(sigma) ** 2
    )


def order_efficiency(
    xs: CrossSection,
    m: int,
    omega_deg: ArrayLike,
    energy_ev: float,
    Qz: ArrayLike,
    *,
    sigma_ler: float,
    omega0_deg: float = 0.0,
    window_T: float = 1.0,
) -> NDArray[np.float64]:
    """HOWTO order efficiency (photons per incident photon) of order ``m``.

    ``eta_m = k^2 |dn_A|^2 |f_m(Qz)|^2 / (cos W cos theta_out)
    * exp(-m^2 G^2 sigma^2) * A * T_m`` with ``W = omega + omega0``,
    ``cos theta_out = cos W + Qz/k`` and ``T_m = window_T**(1/cos W)``.
    """
    k = 2 * np.pi * energy_ev / _HC_EV_NM
    W = np.deg2rad(np.asarray(omega_deg, dtype=float) + omega0_deg)
    qz = np.asarray(Qz, dtype=float)
    cos_w = np.cos(W)
    cos_out = cos_w + qz / k
    fm2 = np.abs(xs.f_m(m, qz)) ** 2
    dw = _debye_waller(m, xs.L, sigma_ler)
    a = absorption_factor(xs.mu_h, W, cos_out)
    t_m = window_T ** (1.0 / cos_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return k**2 * abs(xs.dn_A) ** 2 * fm2 / (cos_w * cos_out) * dw * a * t_m


@dataclass
class OrderScaleFit:
    scale_by_energy: dict[float, float]
    scale_err: dict[float, float]
    sigma_ler: float
    sigma_ler_err: float
    chi2_red: float


def _xs_for(
    xs_by_energy: Mapping[float, CrossSection], e: float
) -> tuple[float, CrossSection]:
    key = min(xs_by_energy, key=lambda x: abs(x - e))
    if not np.isclose(key, e, rtol=0, atol=1e-6):
        raise KeyError(f"no cross section for energy {e}")
    return key, xs_by_energy[key]


def order_scale(
    order_table: pd.DataFrame,
    xs_by_energy: Mapping[float, CrossSection],
    *,
    adu_per_photon: float,
    orders: Sequence[int] = _DEFAULT_ORDERS,
    omega0_deg: float = 0.0,
    window_T: float = 1.0,
) -> OrderScaleFit:
    """Fit c(E) (photons/s) per energy and sigma_LER from order intensities.

    Model ``I_cps / adu_per_photon = c(E) eta_m``; weighted least squares in
    log I (parameters: log c per energy, sigma >= 0). Rows with
    ``I_cps <= 0`` or non-finite values are dropped. Errors come from the
    Jacobian covariance scaled by ``max(1, chi2_red)``.
    """
    df = order_table
    om_col = "omega_true_deg" if "omega_true_deg" in df.columns else "omega_deg"
    d = df[df["order"].isin(list(orders))]
    cols = ["order", "energy_ev", "Qz", "I_cps", "sigma_cps", om_col]
    d = d[cols].astype(float)
    d = d[np.isfinite(d).all(axis=1) & (d["I_cps"] > 0) & (d["sigma_cps"] > 0)]
    if d.empty:
        raise ValueError("no usable rows in order table")
    keys, xs_row = [], []
    for e in d["energy_ev"].to_numpy():
        key, xs = _xs_for(xs_by_energy, e)
        keys.append(key)
        xs_row.append(xs)
    keys_a = np.array(keys)
    energies = sorted(set(keys))
    eidx = np.array([energies.index(k) for k in keys])
    ms = d["order"].to_numpy().astype(int)
    qz = d["Qz"].to_numpy()
    om = d[om_col].to_numpy()
    i_ph = d["I_cps"].to_numpy() / adu_per_photon
    rel = d["sigma_cps"].to_numpy() / d["I_cps"].to_numpy()
    log_i = np.log(i_ph)

    def eta(sigma: float) -> NDArray[np.float64]:
        out = np.empty(len(d))
        for j in range(len(d)):
            out[j] = order_efficiency(
                xs_row[j], int(ms[j]), om[j], keys_a[j], qz[j],
                sigma_ler=sigma, omega0_deg=omega0_deg, window_T=window_T,
            )  # fmt: skip
        return out

    # Precompute sigma-independent part once; DW factor is analytic in sigma.
    base = eta(0.0)
    ok = np.isfinite(base) & (base > 0)
    if ok.sum() < len(energies) + 1:
        raise ValueError("too few usable rows for the order fit")
    lens = np.array([x.L for x in xs_row])
    ne = len(energies)
    for i, e in enumerate(energies):
        if not np.any(ok & (eidx == i)):
            msg = (
                f"no usable order rows at {e} eV (order efficiency not finite "
                "and positive for every row); drop the energy or check Qz/omega"
            )
            raise ValueError(msg)

    def resid(p: NDArray[np.float64]) -> NDArray[np.float64]:
        model = p[eidx] + np.log(base * _debye_waller(ms, lens, p[ne]))
        return ((log_i - model) / rel)[ok]

    c0 = np.array(
        [
            np.median(
                (log_i - np.log(base * _debye_waller(ms, lens, 1.0)))[ok & (eidx == i)]
            )
            for i in range(ne)
        ]
    )
    best = None
    for s0 in (0.5, 1.5, 3.0):
        r = least_squares(
            resid, np.append(c0, s0), bounds=(np.r_[[-np.inf] * ne, 0.0], np.inf)
        )
        if best is None or r.cost < best.cost:
            best = r
    dof = max(int(ok.sum()) - (ne + 1), 1)
    chi2 = 2 * best.cost / dof
    cov = np.linalg.pinv(best.jac.T @ best.jac) * max(1.0, chi2)
    err = np.sqrt(np.diag(cov))
    c = np.exp(best.x[:ne])
    return OrderScaleFit(
        scale_by_energy={e: float(c[i]) for i, e in enumerate(energies)},
        scale_err={e: float(c[i] * err[i]) for i, e in enumerate(energies)},
        sigma_ler=float(best.x[ne]),
        sigma_ler_err=float(err[ne]),
        chi2_red=float(chi2),
    )


def debye_waller_sigma(
    order_table: pd.DataFrame,
    xs_by_energy: Mapping[float, CrossSection],
    **kwargs,
) -> tuple[float, float]:
    """``(sigma_ler, sigma_ler_err)`` from :func:`order_scale`."""
    fit = order_scale(order_table, xs_by_energy, **kwargs)
    return fit.sigma_ler, fit.sigma_ler_err
