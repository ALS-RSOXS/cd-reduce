"""Line-edge, line-width and line-placement roughness PSDs from diffuse scattering.

Physics (first-order Born, transmission, vertical conformal edges)
-----------------------------------------------------------------
A line of width ``w`` centred at ``x = 0`` has the in-plane form factor
``F(qx) = 2 sin(qx w / 2) / qx``. Displacing the left and right edges by
``dL(y)`` and ``dR(y)`` (positive = towards ``+x``) changes it by

``dF = exp(-i qx w/2) dR - exp(+i qx w/2) dL = -2 i sin(qx w/2) P + cos(qx w/2) W``

with the line placement ``P = (dL + dR) / 2`` and width ``W = dR - dL``. For lines
whose roughness is uncorrelated from line to line, the diffuse intensity per
unit reciprocal area normalised by the integrated Bragg order ``I_ref`` is

``R_m(Qy) = (dI/d^2Q)(m G, Qy) / I_ref``
``        = p / ((2 pi)^2 F_ref^2) * [4 sin^2(mGw/2) S_LPR + cos^2(mGw/2) S_LWR]``

with two-sided PSDs ``S`` [nm^3] (``sigma^2 = int S dQy / 2 pi``). ``F_ref`` is
``F_m`` (normalising by the order's own intensity, ``normalize="order"``) or
``F_1`` (normalising by the first orders, ``normalize="first_order"``; valid when
the orders share ``Qz``, e.g. the ``omega = 0`` frame). Contrast, line height and
the ``Qz`` form factor cancel in the ratio.

Each order weights ``S_LPR`` and ``S_LWR`` differently, so a per-``Qy`` linear
least-squares fit across orders separates them. With identical left/right edge
statistics, ``S_LER = S_LPR + S_LWR / 4`` and the left/right edge correlation is
``rho = (S_LPR - S_LWR/4) / (S_LPR + S_LWR/4)``. A cross term proportional to
``Im C_LR`` (asymmetric lagged left/right correlation) is neglected.

Practical limits
----------------
- At 50 % duty cycle the odd orders are blind to ``S_LWR`` (``cos(m pi/2) = 0``)
  and the even orders are extinct, so the split is ill-conditioned; the
  ``condition`` column reports it per bin.
- The relation assumes roughness uncorrelated between lines (diffuse continuous
  in ``Qx``). The ``flank`` columns measure the inter-order level; a rod-localised
  excess (``wing - flank``) signals line-to-line correlated roughness.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, nnls

from cd_reduce.reduction.corrections import DEFAULT_ADU_PER_PHOTON
from cd_reduce.reduction.frames import Frame, load_frames
from cd_reduce.reduction.geometry import (
    ApproximationMode,
    Geometry,
    OmegaRefinement,
)
from cd_reduce.reduction.mapping import (
    BackgroundMode,
    pixel_q_area,
    prepare_frame,
)
from cd_reduce.reduction.roughness import psd_models as _pm

_TWO_PI = 2.0 * np.pi

QyMode = Literal["line_cut", "qz_trace"]
NormalizeMode = Literal["order", "first_order"]

# ---------------------------------------------------------------------------
# PSD models (two-sided, Qy in 1/nm, S in nm^3)
# ---------------------------------------------------------------------------


def psd_mack(
    qy: ArrayLike, p0: float, xi_nm: float, hurst: float, floor: float = 0.0
) -> NDArray[np.float64]:
    """Mack PSD of the wavenumber ``qy`` (delegates to :mod:`psd_models`)."""
    return _pm.psd_mack(
        np.asarray(qy, dtype=np.float64) / _TWO_PI, p0, xi_nm, hurst, floor
    )


def psd_power_law(
    qy: ArrayLike, amplitude: float, exponent: float
) -> NDArray[np.float64]:
    """Pure power law ``S = A * Qy^(-exponent)`` (``Qy`` in 1/nm)."""
    q = np.abs(np.asarray(qy, dtype=np.float64))
    return amplitude * np.power(np.clip(q, 1e-12, None), -exponent)


def psd_spectral_peak(
    qy: ArrayLike,
    p0: float,
    xi_nm: float,
    hurst: float,
    peak_amplitude: float,
    peak_q: float,
    peak_width: float,
) -> NDArray[np.float64]:
    """Mack PSD plus a Gaussian peak at wavenumber ``peak_q`` (delegates)."""
    return _pm.psd_spectral_peak(
        np.asarray(qy, dtype=np.float64) / _TWO_PI,
        p0,
        xi_nm,
        hurst,
        peak_amplitude,
        peak_q / _TWO_PI,
        peak_width / _TWO_PI,
    )


PSD_MODELS: dict[str, tuple[Callable[..., NDArray[np.float64]], tuple[str, ...]]] = {
    "mack": (psd_mack, ("p0", "xi_nm", "hurst")),
    "mack_floor": (psd_mack, ("p0", "xi_nm", "hurst", "floor")),
    "power_law": (psd_power_law, ("amplitude", "exponent")),
    "spectral_peak": (
        psd_spectral_peak,
        ("p0", "xi_nm", "hurst", "peak_amplitude", "peak_q", "peak_width"),
    ),
}


def sigma_from_psd(
    qy_edges: ArrayLike, psd: ArrayLike, psd_err: ArrayLike | None = None
) -> tuple[float, float]:
    """Band-limited RMS ``sigma`` [nm] from a binned two-sided PSD over ``Qy > 0``.

    ``sigma^2 = (1/pi) * sum S_i dQy_i`` over the finite bins. Returns
    ``(sigma, sigma_err)``.
    """
    e = np.asarray(qy_edges, dtype=np.float64)
    s = np.asarray(psd, dtype=np.float64)
    dq = np.diff(e)
    ok = np.isfinite(s)
    var = float(np.sum(s[ok] * dq[ok]) / np.pi)
    if psd_err is None:
        err = 0.0
    else:
        se = np.asarray(psd_err, dtype=np.float64)
        err_var = float(np.sqrt(np.nansum((se[ok] * dq[ok]) ** 2)) / np.pi)
        err = 0.5 * err_var / np.sqrt(var) if var > 0 else np.nan
    return (float(np.sqrt(var)) if var > 0 else np.nan), err


@dataclass
class PsdFit:
    """Result of :func:`fit_psd_model`."""

    model: str
    params: dict[str, float]
    errors: dict[str, float]
    cost: float
    qy_range: tuple[float, float]

    def __call__(self, qy: ArrayLike) -> NDArray[np.float64]:
        fn, names = PSD_MODELS[self.model]
        return fn(qy, *(self.params[n] for n in names))

    def sigma_nm(self, qy_min: float = 0.0, qy_max: float = np.inf) -> float:
        """RMS roughness from integrating the model (``floor`` excluded)."""
        from scipy.integrate import quad

        params = dict(self.params)
        params.pop("floor", None)
        fn, names = PSD_MODELS[self.model]
        names = tuple(n for n in names if n != "floor")
        val, _ = quad(
            lambda q: float(fn(q, *(params[n] for n in names))),
            qy_min,
            qy_max,
            limit=400,
        )
        return float(np.sqrt(val / np.pi)) if val > 0 else np.nan


def fit_psd_model(
    qy: ArrayLike,
    psd: ArrayLike,
    psd_err: ArrayLike | None = None,
    *,
    model: str = "mack",
    p_init: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> PsdFit:
    """Fit a PSD model in log space with error weighting.

    Residuals are ``(log S_model - log S) * S / dS`` so that relative errors weight
    the fit.
    """
    if model not in PSD_MODELS:
        raise ValueError(
            f"Unknown PSD model {model!r}; choose from {sorted(PSD_MODELS)}"
        )
    fn, names = PSD_MODELS[model]
    q = np.asarray(qy, dtype=np.float64)
    s = np.asarray(psd, dtype=np.float64)
    e = (
        np.asarray(psd_err, dtype=np.float64)
        if psd_err is not None
        else 0.1 * np.abs(s)
    )
    ok = np.isfinite(q) & np.isfinite(s) & (s > 0) & np.isfinite(e) & (e > 0)
    q, s, e = q[ok], s[ok], e[ok]
    if q.size < len(names):
        raise ValueError("Not enough finite PSD points to fit the model.")

    defaults: dict[str, float] = {
        "p0": float(s[0]) * 1.2,
        "xi_nm": float(1.0 / np.median(q)),
        "hurst": 0.5,
        "floor": float(np.min(s)) * 0.1,
        "amplitude": float(s[0] * q[0]),
        "exponent": 1.0,
        "peak_amplitude": float(np.max(s)) * 0.1,
        "peak_q": float(np.median(q)),
        "peak_width": float(np.median(q)) * 0.2,
    }
    default_bounds: dict[str, tuple[float, float]] = {
        "p0": (1e-12, np.inf),
        "xi_nm": (0.1, 1e5),
        "hurst": (0.0, 1.0),
        "floor": (0.0, np.inf),
        "amplitude": (1e-12, np.inf),
        "exponent": (0.0, 6.0),
        "peak_amplitude": (0.0, np.inf),
        "peak_q": (float(q.min()), float(q.max())),
        "peak_width": (1e-4, float(q.max())),
    }
    init = {**defaults, **(p_init or {})}
    bnd = {**default_bounds, **(bounds or {})}
    # Positive-scale parameters are fitted in log space
    log_par = {"p0", "xi_nm", "amplitude", "peak_amplitude"}

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
        pars = [_from(v, n) for v, n in zip(x, names, strict=True)]
        m = np.clip(fn(q, *pars), 1e-300, None)
        return (np.log(m) - np.log(s)) * s / e

    res = least_squares(_res, x0, bounds=(lo, hi))
    params = {n: _from(v, n) for v, n in zip(res.x, names, strict=True)}
    errors: dict[str, float] = {}
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac)
        dof = max(q.size - len(names), 1)
        cov *= 2.0 * res.cost / dof
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
        qy_range=(float(q.min()), float(q.max())),
    )


# ---------------------------------------------------------------------------
# Born kernel
# ---------------------------------------------------------------------------


def form_factor_sq(
    order: int | ArrayLike, g_inv_nm: float, line_width_nm: float
) -> NDArray[np.float64]:
    """``F_m^2 = (2 sin(m G w / 2) / (m G))^2`` [nm^2]."""
    m = np.asarray(order, dtype=np.float64)
    qx = m * g_inv_nm
    return (2.0 * np.sin(qx * line_width_nm / 2.0) / qx) ** 2


def edge_psd_kernel(
    order: int | ArrayLike,
    pitch_nm: float,
    line_width_nm: float,
    *,
    normalize: NormalizeMode = "order",
    qx_half_width_g: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kernel ``(k_LPR, k_LWR)`` with ``R_m = k_LPR S_LPR + k_LWR S_LWR``.

    ``R_m`` is the diffuse ``dI/d^2Q`` around ``Qx = m G`` divided by the
    integrated reference order (``normalize="order"``: the order itself;
    ``"first_order"``: the mean of orders ``+-1``). ``qx_half_width_g`` averages
    the ``sin^2`` / ``cos^2`` weights uniformly over ``|Qx/G - m| <`` that
    half width, matching the wing band actually read off the detector.
    """
    G = 2.0 * np.pi / pitch_nm
    m = np.asarray(order, dtype=np.float64)
    a = float(qx_half_width_g) * G
    # <cos(Qx w)> over Qx in [mG - a, mG + a]
    damp = np.sinc(a * line_width_nm / np.pi)
    mean_cos = np.cos(m * G * line_width_nm) * damp
    sin2 = 0.5 * (1.0 - mean_cos)
    cos2 = 0.5 * (1.0 + mean_cos)
    f_ref = (
        form_factor_sq(m, G, line_width_nm)
        if normalize == "order"
        else form_factor_sq(1, G, line_width_nm) * np.ones_like(m)
    )
    pref = pitch_nm / ((2.0 * np.pi) ** 2 * f_ref)
    return pref * 4.0 * sin2, pref * cos2


def estimate_line_width(
    order_intensity: Mapping[int, float],
    pitch_nm: float,
    *,
    w_bounds: tuple[float, float] | None = None,
) -> float:
    """Estimate the line width ``w`` from ``Qz ~ 0`` order ratios ``I_m / I_1``.

    Fits ``|sin(m G w/2) / m|^2`` (box profile) in log space. Only meaningful for
    orders measured at a common ``Qz`` (``omega = 0`` frame). The box model has
    the ``w <-> p - w`` ambiguity; ``w_bounds`` defaults to ``(0.05 p, 0.5 p)``.
    """
    G = 2.0 * np.pi / pitch_nm
    ms = np.array([m for m in order_intensity if m != 0 and order_intensity[m] > 0])
    vals = np.array([order_intensity[int(m)] for m in ms], dtype=np.float64)
    if ms.size < 2:
        raise ValueError("Need at least two positive order intensities.")
    lo, hi = w_bounds or (0.05 * pitch_nm, 0.5 * pitch_nm)
    grid = np.linspace(lo, hi, 2001)
    best, best_cost = lo, np.inf
    for w in grid:
        model = (np.sin(ms * G * w / 2.0) / ms) ** 2
        if np.any(model <= 0):
            continue
        r = np.log(vals) - np.log(model)
        cost = float(np.var(r))
        if cost < best_cost:
            best, best_cost = float(w), cost
    return best


# ---------------------------------------------------------------------------
# Measuring the order-normalised diffuse ratio R_m(Qy)
# ---------------------------------------------------------------------------


@dataclass
class DiffuseRatio:
    """Order-normalised diffuse ``R_m(|Qy|)`` per order (tidy table in ``table``).

    Columns of ``table``: ``order, qy_lo, qy_hi, qy_center, ratio, ratio_err,
    wing_photons, wing_area, flank_ratio, order_photons, qz_mean, n_frames``.
    ``ratio`` is in nm^2 (``dI/d^2Q / I_ref``); ``flank_ratio`` is the same
    quantity measured between orders (``|Qx/G - m|`` in ``flank_g``).
    """

    table: pd.DataFrame
    qy_edges: NDArray[np.float64]
    mode: str
    normalize: str
    pitch_nm: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_order(self, order: int) -> pd.DataFrame:
        return self.table[self.table["order"] == order].reset_index(drop=True)

    @property
    def orders(self) -> list[int]:
        return sorted(int(m) for m in self.table["order"].unique())


def _collect(
    d_ph: NDArray[np.float64],
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    qz: NDArray[np.float64],
    area: NDArray[np.float64],
    G: float,
    order: int,
    qy_edges: NDArray[np.float64],
    *,
    qx_half_g: float,
    core_qx_g: float,
    core_qy: float,
    flank_g: tuple[float, float],
    qz_range: tuple[float, float] | None,
) -> dict[str, Any]:
    """Core photons of order ``m`` and the wing / flank pixels of one frame."""
    fin = np.isfinite(d_ph)
    dx = qx / G - order
    near = fin & (np.abs(dx) < core_qx_g)
    core = near & (np.abs(qy) < core_qy)
    # Diffuse / residual background under the Bragg core, from the ring
    # core_qy < |Qy| < 2 core_qy at the same Qx; matters near extinction.
    ring = near & (np.abs(qy) >= core_qy) & (np.abs(qy) < 2.0 * core_qy)
    ring_area = float(np.sum(area[ring]))
    ring_dens = float(np.sum(d_ph[ring])) / ring_area if ring_area > 0 else 0.0
    n_core = float(np.sum(d_ph[core])) - ring_dens * float(np.sum(area[core]))
    out: dict[str, Any] = {
        "core": n_core,
        "qz_core": float(np.mean(qz[core])) if np.any(core) else np.nan,
    }
    sel_qz = fin.copy()
    if qz_range is not None:
        sel_qz &= (qz >= qz_range[0]) & (qz <= qz_range[1])
    nb = len(qy_edges) - 1
    for name, sel in (
        ("wing", sel_qz & (np.abs(dx) < qx_half_g)),
        ("flank", sel_qz & (np.abs(dx) > flank_g[0]) & (np.abs(dx) < flank_g[1])),
    ):
        ib = np.digitize(np.abs(qy[sel]), qy_edges) - 1
        ok = (ib >= 0) & (ib < nb)
        out[name] = (ib[ok], d_ph[sel][ok], area[sel][ok], qz[sel][ok])
    return out


def _rod_interp(
    qz: Sequence[float], val: Sequence[float]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Log-linear interpolation of an order rod ``I_m(Qz)``; NaN outside."""
    q = np.asarray(qz, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    ok = np.isfinite(q) & (v > 1.0)
    q, v = q[ok], v[ok]
    order = np.argsort(q)
    q, lv = q[order], np.log(v[order])

    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        if q.size == 0:
            return np.full_like(x, np.nan)
        if q.size == 1:
            return np.where(np.abs(x - q[0]) < 1e-12, np.exp(lv[0]), np.nan)
        y = np.exp(np.interp(x, q, lv))
        return np.where((x >= q[0]) & (x <= q[-1]), y, np.nan)

    return f


def extract_diffuse_ratio(
    source: Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    *,
    orders: Sequence[int] = (-4, -3, -2, -1, 1, 2, 3, 4),
    qy_edges: ArrayLike | None = None,
    mode: QyMode = "line_cut",
    omega: float | None = 0.0,
    omega_tol_deg: float = 0.4,
    qz_range: tuple[float, float] | None = None,
    normalize: NormalizeMode = "order",
    qx_half_g: float = 0.25,
    core_qx_g: float = 0.25,
    core_qy_inv_nm: float = 0.015,
    flank_g: tuple[float, float] = (0.35, 0.5),
    bg_mode: BackgroundMode = "photon_events",
    approximation: ApproximationMode = "correct",
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    beamstop_mask: NDArray[np.bool_] | None = None,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
    energy_ev: float | None = None,
    ewald_correction: bool = True,
) -> DiffuseRatio:
    """Measure ``R_m(|Qy|) = (dI/d^2Q)/I_ref`` for each order from detector frames.

    Two ways to choose where ``I(Qy)`` is read:

    - ``mode="line_cut"``: the detector line cut through order ``m`` in the
      frame(s) at ``omega`` (``+- omega_tol_deg``); ``Qz`` follows that frame's
      Ewald trace (``Qz ~ m G tan(omega)``). ``omega=None`` pools all frames.
    - ``mode="qz_trace"``: follow a constant-``Qz`` trace through the tilt series:
      every frame contributes the pixels with ``Qz`` in ``qz_range``, and only
      frames whose order core lies in ``qz_range`` are used, so the diffuse and
      its reference order share ``Qz``. Gives the in-plane PSD at a chosen
      height frequency (``Qz = 0`` for the height-averaged edge).

    Ewald curvature puts wing pixels at large ``|Qy|`` at a lower ``Qz`` than
    their Bragg core (``dQz ~ -Qy^2 / 2k``); with a finite line height the
    ``Qz`` form factor then no longer cancels. With ``ewald_correction`` (and a
    tilt series in ``source``) the order rods ``I_m(Qz)`` measured from all
    frames' cores (per second of exposure) correct each wing pixel to its own
    ``Qz``. With a single frame no correction is possible.

    The integrated core has the local diffuse level (ring
    ``core_qy < |Qy| < 2 core_qy``) subtracted, which matters for weak orders.

    Pooling over frames is a ratio of sums, ``sum(wing photons) /
    sum(wing area * I_ref)``, so frames with more photons weigh more.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    G = geo.g_inv_nm
    if qy_edges is None:
        qy_edges = np.geomspace(max(core_qy_inv_nm, 0.01), 0.2, 14)
    qe = np.asarray(qy_edges, dtype=np.float64)
    if np.any(qe < 0):
        raise ValueError("qy_edges are |Qy| edges and must be non-negative.")
    if mode == "qz_trace" and qz_range is None:
        qz_range = (-0.01, 0.01)
    if mode == "line_cut":
        qz_range = None

    frames = load_frames(source, deduplicate_omega=False, energy_ev=energy_ev)
    if mode == "line_cut" and omega is not None:
        if not any(abs(f.omega_deg - omega) <= omega_tol_deg for f in frames):
            raise ValueError(f"No frame within {omega_tol_deg} deg of omega={omega}.")
        if not ewald_correction:
            frames = [f for f in frames if abs(f.omega_deg - omega) <= omega_tol_deg]
    if not frames:
        raise ValueError("No frames match the requested omega / energy selection.")

    want = sorted(
        {int(m) for m in orders} | ({-1, 1} if normalize == "first_order" else set())
    )
    nb = len(qe) - 1
    cores: dict[int, list[tuple[float, float]]] = {m: [] for m in want}
    rate: dict[int, list[tuple[float, float]]] = {m: [] for m in want}
    exposures: list[float] = []
    use_wing: list[bool] = []
    pix: dict[int, list[dict[str, Any]]] = {m: [] for m in want}

    for fr in frames:
        cleaned, qm, _, _ = prepare_frame(
            fr,
            geo,
            bg_mode=bg_mode,
            approximation=approximation,
            omega_refinement=omega_refinement,
            beamstop_mask=beamstop_mask,
            exposure_normalization=False,
            adu_per_photon=adu_per_photon,
        )
        d_ph = cleaned / adu_per_photon
        area = pixel_q_area(qm.Qx, qm.Qy)
        per = {
            m: _collect(
                d_ph,
                qm.Qx,
                qm.Qy,
                qm.Qz,
                area,
                G,
                m,
                qe,
                qx_half_g=qx_half_g,
                core_qx_g=core_qx_g,
                core_qy=core_qy_inv_nm,
                flank_g=flank_g,
                qz_range=qz_range,
            )
            for m in want
        }
        exp_s = max(float(fr.exposure_s), 1e-12)
        for m in want:
            cores[m].append((per[m]["qz_core"], per[m]["core"]))
            rate[m].append((per[m]["qz_core"], per[m]["core"] / exp_s))
            pix[m].append(per[m])
        exposures.append(exp_s)
        use_wing.append(
            mode != "line_cut"
            or omega is None
            or abs(fr.omega_deg - omega) <= omega_tol_deg
        )

    # Order rods I_m(Qz) [photons / s] from the cores of every frame
    rods = {m: _rod_interp(*zip(*rate[m], strict=True)) for m in want}

    def _shape(m: int, qz_pix: NDArray[np.float64], qz0: float) -> NDArray[np.float64]:
        """Rod shape correction I_m(Qz_pix) / I_m(Qz_core); 1 where unknown."""
        if not ewald_correction:
            return np.ones_like(qz_pix)
        r0 = rods[m](np.array([qz0]))[0]
        r = rods[m](qz_pix) / r0
        return np.where(np.isfinite(r) & (r > 0), r, 1.0)

    def _ref(
        m: int, frame_idx: int, qz_pix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if mode == "line_cut":
            if normalize == "first_order":
                v1, v2 = cores[1][frame_idx][1], cores[-1][frame_idx][1]
                val = 0.5 * (
                    v1 * _shape(1, qz_pix, cores[1][frame_idx][0])
                    + v2 * _shape(-1, qz_pix, cores[-1][frame_idx][0])
                )
                return np.where(v1 + v2 > 2.0, val, np.nan)
            val = cores[m][frame_idx][1]
            if not val > 1.0:
                return np.full_like(qz_pix, np.nan)
            return val * _shape(m, qz_pix, cores[m][frame_idx][0])
        exp_s = exposures[frame_idx]
        if normalize == "first_order":
            return 0.5 * exp_s * (rods[1](qz_pix) + rods[-1](qz_pix))
        return exp_s * rods[m](qz_pix)

    rows = []
    qc = np.sqrt(qe[1:] * np.maximum(qe[:-1], 1e-12))
    for m in sorted(int(x) for x in orders):
        sums = {k: np.zeros(nb) for k in ("wph", "wden", "warea", "fph", "fden")}
        used, qz_used = 0, []
        for i, rec in enumerate(pix[m]):
            if not use_wing[i]:
                continue
            any_used = False
            for name, kph, kden in (("wing", "wph", "wden"), ("flank", "fph", "fden")):
                ib, ph, ar, qz = rec[name]
                ref = _ref(m, i, qz)
                ok = np.isfinite(ref)
                if not np.any(ok):
                    continue
                sums[kph] += np.bincount(ib[ok], ph[ok], minlength=nb)
                sums[kden] += np.bincount(ib[ok], ar[ok] * ref[ok], minlength=nb)
                if name == "wing":
                    sums["warea"] += np.bincount(ib[ok], ar[ok], minlength=nb)
                    any_used = True
            if any_used:
                used += 1
                qz_used.append(cores[m][i][0])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sums["wden"] > 0, sums["wph"] / sums["wden"], np.nan)
            err = np.where(
                sums["wden"] > 0,
                np.sqrt(np.clip(sums["wph"], 1.0, None)) / sums["wden"],
                np.nan,
            )
            flank = np.where(sums["fden"] > 0, sums["fph"] / sums["fden"], np.nan)
        core_total = float(sum(c for _, c in cores[m]))
        qz_mean = float(np.nanmean(qz_used)) if qz_used else np.nan
        for i in range(nb):
            rows.append(
                {
                    "order": m,
                    "qy_lo": qe[i],
                    "qy_hi": qe[i + 1],
                    "qy_center": qc[i],
                    "ratio": ratio[i],
                    "ratio_err": err[i],
                    "wing_photons": sums["wph"][i],
                    "wing_den": sums["wden"][i],
                    "flank_photons": sums["fph"][i],
                    "flank_den": sums["fden"][i],
                    "wing_area": sums["warea"][i],
                    "flank_ratio": flank[i],
                    "order_photons": core_total,
                    "qz_mean": qz_mean,
                    "n_frames": used,
                }
            )
    return DiffuseRatio(
        table=pd.DataFrame(rows),
        qy_edges=qe,
        mode=mode,
        normalize=normalize,
        pitch_nm=geo.Lambda_nm,
        metadata={
            "omega": omega,
            "qz_range": qz_range,
            "bg_mode": bg_mode,
            "approximation": approximation,
            "energy_ev": float(frames[0].energy_ev),
            "qx_half_g": float(qx_half_g),
            "ewald_correction": bool(ewald_correction) and len(frames) > 1,
        },
    )


def combine_diffuse_ratios(
    ratios: Sequence[DiffuseRatio], *, clip_sigma: float | None = 5.0
) -> DiffuseRatio:
    """Pool several :class:`DiffuseRatio` (mounts, energies, runs) as a ratio of sums.

    All inputs must share ``qy_edges``, ``normalize`` and the pitch (within
    1 %). Photons and ``area * I_ref`` denominators are summed per order and
    bin, so better-exposed inputs weigh more; errors are Poisson on the pooled
    photons.

    ``clip_sigma`` rejects, per order and bin, any input whose photon count
    exceeds the expectation from the others (their pooled ratio times its own
    denominator) by more than ``clip_sigma`` Poisson sigmas (leave-one-out;
    catches cosmic rays, hot pixels and bleed residue in single inputs). The
    number of rejected inputs is in the ``n_clipped`` column.
    """
    if not ratios:
        raise ValueError("No DiffuseRatio to combine.")
    ref = ratios[0]
    for r in ratios[1:]:
        if not np.allclose(r.qy_edges, ref.qy_edges):
            raise ValueError("qy_edges differ between inputs.")
        if r.normalize != ref.normalize:
            raise ValueError("normalize differs between inputs.")
        if abs(r.pitch_nm / ref.pitch_nm - 1.0) > 0.01:
            raise ValueError("pitch differs by more than 1 % between inputs.")
    t = pd.concat(
        [r.table.assign(_part=i) for i, r in enumerate(ratios)], ignore_index=True
    )
    keys = ["order", "qy_lo", "qy_hi", "qy_center"]
    t["_clipped"] = False
    if clip_sigma is not None and len(ratios) > 2:
        g = t.groupby(keys)
        tot_ph = g["wing_photons"].transform("sum")
        tot_den = g["wing_den"].transform("sum")
        rest_ph = tot_ph - t["wing_photons"]
        rest_den = tot_den - t["wing_den"]
        with np.errstate(divide="ignore", invalid="ignore"):
            expect = np.where(rest_den > 0, rest_ph / rest_den, np.nan) * t["wing_den"]
        excess = (t["wing_photons"] - expect) / np.sqrt(np.clip(expect, 1.0, None))
        t["_clipped"] = np.nan_to_num(excess, nan=0.0) > clip_sigma
        for col in ("wing_photons", "wing_den", "flank_photons", "flank_den"):
            t.loc[t["_clipped"], col] = 0.0
    agg = t.groupby(keys, as_index=False).agg(
        wing_photons=("wing_photons", "sum"),
        wing_den=("wing_den", "sum"),
        flank_photons=("flank_photons", "sum"),
        flank_den=("flank_den", "sum"),
        wing_area=("wing_area", "sum"),
        order_photons=("order_photons", "sum"),
        qz_mean=("qz_mean", "mean"),
        n_frames=("n_frames", "sum"),
        n_clipped=("_clipped", "sum"),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        den = agg["wing_den"].to_numpy()
        ph = agg["wing_photons"].to_numpy()
        agg["ratio"] = np.where(den > 0, ph / den, np.nan)
        agg["ratio_err"] = np.where(
            den > 0, np.sqrt(np.clip(ph, 1.0, None)) / den, np.nan
        )
        fden = agg["flank_den"].to_numpy()
        agg["flank_ratio"] = np.where(fden > 0, agg["flank_photons"] / fden, np.nan)
    meta = dict(ref.metadata)
    meta["n_inputs"] = len(ratios)
    meta["energy_ev"] = sorted({r.metadata.get("energy_ev") for r in ratios})
    meta["pitch_nm_inputs"] = [r.pitch_nm for r in ratios]
    return DiffuseRatio(
        table=agg.sort_values(["order", "qy_lo"]).reset_index(drop=True),
        qy_edges=ref.qy_edges,
        mode=ref.mode,
        normalize=ref.normalize,
        pitch_nm=float(np.mean([r.pitch_nm for r in ratios])),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Decomposition into LER / LWR / LPR
# ---------------------------------------------------------------------------


@dataclass
class EdgePsdResult:
    """LER / LWR / LPR PSDs per ``|Qy|`` bin (two-sided, nm^3).

    ``table`` columns: ``qy_lo, qy_hi, qy_center, f_inv_nm, S_LPR, S_LPR_err,
    S_LWR, S_LWR_err, S_LER, S_LER_err, rho_LR, n_orders, condition,
    decomposed``. Where only one usable order (or an ill-conditioned set) is
    available, ``decomposed`` is False and ``S_LER`` is the single-edge
    read-off assuming uncorrelated edges (``rho = 0``).
    """

    table: pd.DataFrame
    pitch_nm: float
    line_width_nm: float
    orders: list[int]
    qy_edges: NDArray[np.float64]

    def sigma(self, which: Literal["LER", "LWR", "LPR"] = "LER") -> tuple[float, float]:
        """Band-limited RMS (nm) of the chosen PSD over the measured bins."""
        return sigma_from_psd(
            self.qy_edges,
            self.table[f"S_{which}"].to_numpy(),
            self.table[f"S_{which}_err"].to_numpy(),
        )

    def fit(
        self,
        which: Literal["LER", "LWR", "LPR"] = "LER",
        model: str = "mack",
        **kw: Any,
    ) -> PsdFit:
        t = self.table
        return fit_psd_model(
            t["qy_center"], t[f"S_{which}"], t[f"S_{which}_err"], model=model, **kw
        )


def decompose_edge_psd(
    ratio: DiffuseRatio,
    line_width_nm: float,
    *,
    orders: Sequence[int] | None = None,
    use: Literal["wing", "wing_minus_flank"] = "wing",
    max_condition: float = 1e3,
    min_photons: float = 10.0,
) -> EdgePsdResult:
    """Split ``R_m(Qy)`` into ``S_LPR`` and ``S_LWR`` by a multi-order fit.

    Per ``Qy`` bin, solves the non-negative weighted least squares
    ``R_m = k_LPR(m) S_LPR + k_LWR(m) S_LWR`` over the usable orders (``+-m``
    are both used; they share the kernel). Uncertainties come from the
    unconstrained weighted normal equations.

    ``use="wing_minus_flank"`` subtracts the inter-order level first, keeping
    only the rod-localised (line-to-line correlated) part.
    """
    p = ratio.pitch_nm
    t = ratio.table
    sel_orders = sorted(set(orders) if orders is not None else set(ratio.orders))
    rows = []
    for qlo, sub in t.groupby("qy_lo", sort=True):
        sub = sub[sub["order"].isin(sel_orders)]
        y = sub["ratio"].to_numpy(dtype=np.float64)
        if use == "wing_minus_flank":
            y = y - np.nan_to_num(sub["flank_ratio"].to_numpy(dtype=np.float64))
        e = sub["ratio_err"].to_numpy(dtype=np.float64)
        m = sub["order"].to_numpy(dtype=np.float64)
        ok = (
            np.isfinite(y)
            & np.isfinite(e)
            & (e > 0)
            & (sub["wing_photons"].to_numpy() >= min_photons)
        )
        kP, kW = edge_psd_kernel(
            m,
            p,
            line_width_nm,
            normalize=ratio.normalize,  # type: ignore[arg-type]
            qx_half_width_g=float(ratio.metadata.get("qx_half_g", 0.0)),
        )
        ok &= np.isfinite(kP) & np.isfinite(kW)
        row: dict[str, Any] = {
            "qy_lo": float(qlo),
            "qy_hi": float(sub["qy_hi"].iloc[0]),
            "qy_center": float(sub["qy_center"].iloc[0]),
            "n_orders": len(np.unique(np.abs(m[ok]))),
        }
        A = np.column_stack([kP[ok], kW[ok]]) / e[ok, None]
        b = y[ok] / e[ok]
        cond = float(np.linalg.cond(A)) if A.shape[0] >= 2 else np.inf
        row["condition"] = cond
        if row["n_orders"] >= 2 and cond < max_condition:
            x, _ = nnls(A, b)
            cov = np.linalg.pinv(A.T @ A)
            sP, sW = float(x[0]), float(x[1])
            eP, eW = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
            sL = sP + 0.25 * sW
            eL = float(np.sqrt(cov[0, 0] + 0.0625 * cov[1, 1] + 0.5 * cov[0, 1]))
            rho = (sP - 0.25 * sW) / sL if sL > 0 else np.nan
            row.update(
                S_LPR=sP,
                S_LPR_err=eP,
                S_LWR=sW,
                S_LWR_err=eW,
                S_LER=sL,
                S_LER_err=eL,
                rho_LR=rho,
                decomposed=True,
            )
        elif np.any(ok):
            # Uncorrelated edges: S_LPR = S/2, S_LWR = 2 S -> R = S (k_P/2 + 2 k_W)
            kk = 0.5 * kP[ok] + 2.0 * kW[ok]
            w = 1.0 / (e[ok] / kk) ** 2
            sL = float(np.sum(w * y[ok] / kk) / np.sum(w))
            eL = float(1.0 / np.sqrt(np.sum(w)))
            row.update(
                S_LPR=0.5 * sL,
                S_LPR_err=0.5 * eL,
                S_LWR=2.0 * sL,
                S_LWR_err=2.0 * eL,
                S_LER=sL,
                S_LER_err=eL,
                rho_LR=0.0,
                decomposed=False,
            )
        else:
            row.update(
                S_LPR=np.nan,
                S_LPR_err=np.nan,
                S_LWR=np.nan,
                S_LWR_err=np.nan,
                S_LER=np.nan,
                S_LER_err=np.nan,
                rho_LR=np.nan,
                decomposed=False,
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table["f_inv_nm"] = table["qy_center"] / (2.0 * np.pi)
    return EdgePsdResult(
        table=table,
        pitch_nm=p,
        line_width_nm=float(line_width_nm),
        orders=sel_orders,
        qy_edges=ratio.qy_edges,
    )


__all__ = [
    "PSD_MODELS",
    "DiffuseRatio",
    "EdgePsdResult",
    "PsdFit",
    "combine_diffuse_ratios",
    "decompose_edge_psd",
    "edge_psd_kernel",
    "estimate_line_width",
    "extract_diffuse_ratio",
    "fit_psd_model",
    "form_factor_sq",
    "psd_mack",
    "psd_power_law",
    "psd_spectral_peak",
    "sigma_from_psd",
]
