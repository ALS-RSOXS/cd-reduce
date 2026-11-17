"""Background subtraction, detector offset removal, and physical/geometric corrections.

Includes:
1. Corner dark offset and read-noise estimation.
2. Per-row photon-clipped mean + cubic column ramp offset removal.
3. Joint strip model fitting (Poisson deviance ML with analytic Jacobian for flat floor
   + order pseudo-Voigts + beamstop bleed Gaussians) and per-frame origin-shift
   measurement.
4. Off-line ``Qy`` band non-Bragg background profile subtraction per ``Qx`` bin.
5. Read-noise gating and connected-cluster photon event thresholding.
6. Beamstop diamond + arm mask construction.
7. Si3N4 membrane window transmission correction ``T = exp(-mu(E) * t / cos(omega))``.
8. Beam walk-off illumination factor ``f(omega) = Phi(a - b * tan(omega))`` and
   fitting.
9. Parasitic scatter masking and solid-angle / exposure / polarization corrections.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.stats import norm

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import (
    Geometry,
    PixelQMap,
    compute_pixel_q,
    frame_pixel_grid,
)
from cd_reduce.utils.image import detector_corner_offset

#: Tabulated Si3N4 linear attenuation coefficient energies [eV] and values [1/nm]
#: (2.8 g/cm^3 Si3N4, T = 0.515958, 0.528600, 0.536654 at 100 nm).
_SI3N4_E_EV: NDArray[np.float64] = np.array([250.0, 280.0, 285.2, 288.6, 320.0])
_SI3N4_MU_INV_NM: NDArray[np.float64] = np.array(
    [
        0.00850,
        -np.log(0.515958) / 100.0,
        -np.log(0.528600) / 100.0,
        -np.log(0.536654) / 100.0,
        0.00520,
    ],
    dtype=np.float64,
)

#: Default strip-fit constants.
DEFAULT_ADU_PER_PHOTON: float = 27.0
DEFAULT_READ_NOISE_ADU: float = 3.0
DEFAULT_STRIP_HALF_WIDTH_PX: int = 18
_NB_BLEED: int = 7
_REACH_R1: float = 4.6
_BLEED_SA_MAX_R1: float = 0.6


def mu_si3n4(energy_ev: ArrayLike) -> NDArray[np.float64] | float:
    """Linear attenuation coefficient ``mu(E)`` of Si3N4 (2.8 g/cm^3) in 1/nm.

    Interpolates across the carbon K-edge (280 to 288.6 eV) and extrapolates
    smoothly over soft X-ray energies (250 to 320 eV).

    Parameters
    ----------
    energy_ev : float or array-like
        Photon energy in eV.

    Returns
    -------
    float or ndarray
        Attenuation coefficient ``mu`` in 1/nm.
    """
    e_arr = np.asarray(energy_ev, dtype=np.float64)
    res = np.interp(e_arr, _SI3N4_E_EV[1:4], _SI3N4_MU_INV_NM[1:4])
    outside = (e_arr < 279.5) | (e_arr > 289.1)
    if np.any(outside):
        res = np.where(outside, np.interp(e_arr, _SI3N4_E_EV, _SI3N4_MU_INV_NM), res)
    if np.ndim(energy_ev) == 0:
        return float(res)
    return res


def membrane_transmission(
    omega_deg: ArrayLike,
    energy_ev: ArrayLike,
    thickness_nm: float = 50.0,
) -> NDArray[np.float64] | float:
    """Compute Si3N4 window transmission ``T = exp(-mu(E) * t / cos(omega))``.

    Parameters
    ----------
    omega_deg : float or array-like
        Refined sample incidence angle in degrees.
    energy_ev : float or array-like
        Photon energy in eV.
    thickness_nm : float, optional
        Si3N4 membrane thickness in nm. Default is ``50.0``.

    Returns
    -------
    float or ndarray
        Window transmission factor ``T`` in ``(0, 1]``.
    """
    om_rad = np.radians(np.asarray(omega_deg, dtype=np.float64))
    mu_val = np.asarray(mu_si3n4(energy_ev), dtype=np.float64)
    cos_om = np.clip(np.cos(om_rad), 1e-3, None)
    trans = np.exp(-mu_val * thickness_nm / cos_om)
    if np.ndim(omega_deg) == 0 and np.ndim(energy_ev) == 0:
        return float(trans)
    return trans


def walkoff_factor(
    omega_deg: ArrayLike,
    a: float,
    b: float,
) -> NDArray[np.float64] | float:
    """Compute beam walk-off illumination factor ``f(omega) = Phi(a - b * tan(omega))``.

    Models a Gaussian beam of width ``sigma_b`` whose footprint translates by
    ``d * tan(omega)`` across a sample pad edge at distance ``e``, where
    ``a = e / sigma_b`` and ``b = d / sigma_b``.

    Parameters
    ----------
    omega_deg : float or array-like
        Sample rotation angle in degrees.
    a, b : float
        Dimensionless pad-edge distance and rotation-axis offset parameters.

    Returns
    -------
    float or ndarray
        Illumination fraction ``f(omega)`` in ``[0, 1]``.
    """
    if np.isinf(a):
        if np.ndim(omega_deg) == 0:
            return 1.0
        return np.ones_like(np.asarray(omega_deg, dtype=np.float64))
    om_rad = np.radians(np.asarray(omega_deg, dtype=np.float64))
    f_val = np.asarray(norm.cdf(a - b * np.tan(om_rad)), dtype=np.float64)
    if np.ndim(omega_deg) == 0:
        return float(f_val)
    return f_val


def fit_walkoff_parameters(
    omega_deg: ArrayLike,
    order_flux_sum: ArrayLike,
) -> tuple[float, float, float]:
    """Fit walk-off parameters ``(a, b, scale)`` to frame sums of strong Bragg orders.

    Parameters
    ----------
    omega_deg : array-like
        Incidence angles in degrees.
    order_flux_sum : array-like
        Summed intensity of low-index orders (e.g. ``|m| in {1, 2, 3}``) per frame.

    Returns
    -------
    a, b, scale : float
        Best-fit walk-off parameters ``a``, ``b`` and normal-incidence scale.
    """
    om = np.asarray(omega_deg, dtype=np.float64)
    fl = np.asarray(order_flux_sum, dtype=np.float64)
    valid = np.isfinite(om) & np.isfinite(fl) & (fl > 0)
    om, fl = om[valid], fl[valid]
    if om.size < 3:
        return float("inf"), 0.0, float(np.nanmedian(fl) if fl.size else 1.0)

    s0 = float(np.percentile(fl, 90))
    p0 = np.array([1.5, -4.0, np.log(max(s0, 1e-6))], dtype=np.float64)

    def _res(p: NDArray[np.float64]) -> NDArray[np.float64]:
        pred = np.exp(p[2]) * np.clip(walkoff_factor(om, p[0], p[1]), 1e-6, 1.0)
        return np.log(fl) - np.log(pred)

    sol = least_squares(
        _res,
        p0,
        bounds=([-5.0, -30.0, -30.0], [10.0, 30.0, 50.0]),
        loss="soft_l1",
    )
    return float(sol.x[0]), float(sol.x[1]), float(np.exp(sol.x[2]))


def parasitic_mask(
    omega_deg: float,
    two_theta_deg: ArrayLike,
    *,
    omega_min: float = 29.5,
    two_theta_max: float = -8.0,
) -> NDArray[np.bool_]:
    """Return boolean mask of clean (non-parasitic) pixels.

    In high-angle rotation series, parasitic scatter from the sample holder/window
    can appear at ``omega >= 29.5 deg`` for ``2*theta <= -8.0 deg``.

    Parameters
    ----------
    omega_deg : float
        Frame incidence angle in degrees.
    two_theta_deg : array-like
        Detector scattering angle ``2*theta`` in degrees.
    omega_min : float, optional
        Angle threshold above which parasitic masking applies. Default is ``29.5``.
    two_theta_max : float, optional
        Upper bound of parasitic ``2*theta`` region. Default is ``-8.0``.

    Returns
    -------
    ndarray of bool
        True where pixels are clean, False where contaminated by parasitic scatter.
    """
    tth = np.asarray(two_theta_deg, dtype=np.float64)
    if omega_deg >= omega_min:
        return tth > two_theta_max
    return np.ones(tth.shape, dtype=bool)


def subtract_corner_offset(
    image: ArrayLike,
    *,
    corner_px: int | None = None,
    use_clipped_mean: bool = False,
) -> tuple[NDArray[np.float64], float, float]:
    """Subtract global dark offset estimated from the four detector corner blocks.

    Returns
    -------
    cleaned : ndarray of float64
        Offset-subtracted image.
    offset_adu : float
        Subtracted dark level in ADU.
    read_sigma_adu : float
        Robust per-pixel read noise in ADU.
    """
    img = np.asarray(image, dtype=np.float64)
    off, sig = detector_corner_offset(
        img, corner_px=corner_px, use_clipped_mean=use_clipped_mean
    )
    return img - off, off, sig


def subtract_row_col_offset(
    image: ArrayLike,
    geometry: Geometry,
    qy_map: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    qy_min_g: float = 0.35,
    r_min_px: float = 150.0,
    clip_adu: float = 10.0,
) -> NDArray[np.float64]:
    """Subtract CCD row pattern and cubic column ramp from off-line pixels.

    Computes the per-row photon-clipped mean of off-line pixels (``|Qy| > 0.35 * G``,
    ``r > 150 px``) plus a degree-3 polynomial along columns.

    Parameters
    ----------
    image : array-like
        Raw 2D detector image (ADU).
    geometry : Geometry
        Frame geometry.
    qy_map : array-like
        2D ``Qy`` map (1/nm) of the same shape as ``image``.
    mask : array-like of bool, optional
        Beamstop or bad-pixel mask (True = masked/excluded).
    qy_min_g : float, optional
        Minimum ``|Qy| / G`` for off-line background pixels. Default is ``0.35``.
    r_min_px : float, optional
        Minimum radial distance from beam center in pixels. Default is ``150.0``.
    clip_adu : float, optional
        Clipping window around the median in ADU. Default is ``10.0``.

    Returns
    -------
    ndarray of float64
        Row- and column-offset subtracted image (ADU).
    """
    im = np.asarray(image, dtype=np.float64)
    qy = np.asarray(qy_map, dtype=np.float64)
    nrow, ncol = im.shape
    yy, xx = np.mgrid[:nrow, :ncol].astype(np.float64)
    mask_arr = (
        np.asarray(mask, dtype=bool)
        if mask is not None
        else np.zeros(im.shape, dtype=bool)
    )

    G = geometry.g_inv_nm
    r = np.hypot(xx - geometry.u0_px, yy - geometry.v0_px)
    r_cut = min(r_min_px, 0.25 * min(nrow, ncol))
    m = (np.abs(qy) > qy_min_g * G) & (~mask_arr) & (r > r_cut)
    if not np.any(m):
        m = ~mask_arr
    med_val = float(np.median(im[m])) if np.any(m) else float(np.median(im))
    m = m & (np.abs(im - med_val) < clip_adu)
    if not np.any(m):
        return im - med_val

    yi, xi = yy[m].astype(int), xx[m].astype(int)
    row_cnt = np.bincount(yi, None, nrow)
    row_sum = np.bincount(yi, im[m], nrow)
    rows = np.where(row_cnt > 0, row_sum / np.maximum(row_cnt, 1), med_val)
    res = im - rows[:, None]

    col_cnt = np.bincount(xi, None, ncol)
    col_mean = np.bincount(xi, res[m], ncol) / np.maximum(col_cnt, 1)
    ok = col_cnt > max(min(50, nrow // 20), 3)
    if ok.sum() >= 4:
        x_norm = np.arange(ncol, dtype=np.float64) / float(ncol)
        poly = np.polyval(np.polyfit(x_norm[ok], col_mean[ok], 3), x_norm)
        return res - poly[None, :]
    return res


def gate_read_noise(
    image: ArrayLike,
    threshold_adu: float = 15.0,
) -> NDArray[np.float64]:
    """Keep pixels at or 8-connected to an event above ``threshold_adu``, zero
    elsewhere.
    """
    arr = np.asarray(image, dtype=np.float64)
    dilated = ndimage.binary_dilation(
        np.nan_to_num(arr, nan=0.0) > threshold_adu, np.ones((3, 3), dtype=bool)
    )
    return np.where(dilated, arr, 0.0)


def threshold_photon_events(
    image: ArrayLike,
    energy_ev: float = 250.0,
    *,
    read_sigma_adu: float = 7.4,
    ev_low_sigma: float = 2.5,
    ev_seed_sigma: float = 7.0,
    ev_min_photons: float = 0.5,
    adu_per_photon_250ev: float = 110.0,
    adu_per_photon: float | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Filter a bias-subtracted image by connected single-photon charge clusters.

    Keeps connected clusters above ``ev_low_sigma * read_sigma_adu`` that contain
    at least one pixel above ``ev_seed_sigma * read_sigma_adu`` and total cluster
    charge above ``ev_min_photons * adu_per_photon(E)``. Preserves analog ADU inside
    kept clusters so bright Bragg cores do not suffer from pile-up.

    Parameters
    ----------
    image : array-like
        Bias-subtracted 2D image (ADU).
    energy_ev : float, optional
        Photon energy in eV. Default is ``250.0``.
    read_sigma_adu : float, optional
        Per-pixel read noise in ADU. Default is ``7.4``.
    ev_low_sigma, ev_seed_sigma : float, optional
        Cluster envelope and seed peak thresholds in units of ``read_sigma_adu``.
    ev_min_photons : float, optional
        Minimum integrated photon count per cluster. Default is ``0.5``.
    adu_per_photon_250ev : float, optional
        Single-photon cluster ADU at 250 eV, scaled linearly with energy.
        Default is ``110.0``. Ignored when ``adu_per_photon`` is given.
    adu_per_photon : float, optional
        Calibrated ADU per photon of the detector at ``energy_ev`` (e.g. 27 for
        the 11.0.1.2 CCD). Takes precedence over ``adu_per_photon_250ev``.

    Returns
    -------
    gated_image : ndarray of float64
        Event-thresholded image (ADU).
    n_events : int
        Number of kept photon clusters.
    """
    d = np.asarray(image, dtype=np.float64)
    labl, nlab = ndimage.label(d > ev_low_sigma * read_sigma_adu)
    if nlab == 0:
        return np.zeros_like(d), 0
    idx = np.arange(1, nlab + 1)
    mx = ndimage.maximum(d, labl, idx)
    sm = ndimage.sum(d, labl, idx)
    a_ph = (
        float(adu_per_photon)
        if adu_per_photon is not None
        else adu_per_photon_250ev * (energy_ev / 250.0)
    )
    keep = np.r_[
        False,
        (mx > ev_seed_sigma * read_sigma_adu) & (sm > ev_min_photons * a_ph),
    ]
    out = np.where(keep[labl], d, 0.0)
    return out, int(keep.sum())


def build_beamstop_mask(
    frames: Iterable[Frame | NDArray[np.floating]],
    geometry: Geometry,
    *,
    event_threshold_adu: float = 15.0,
    dilate_iterations: int = 6,
    max_radius_px: float = 450.0,
) -> NDArray[np.bool_]:
    """Detect the beamstop shadow and support arm from summed photon events across
    frames.

    Parameters
    ----------
    frames : iterable of Frame or 2D ndarray
        Frames at a given energy used to accumulate photon hits.
    geometry : Geometry
        Mount geometry providing beam center ``(u0_px, v0_px)``.
    event_threshold_adu : float, optional
        Threshold above row background for counting a photon event. Default is ``15.0``.
    dilate_iterations : int, optional
        Binary dilation iterations around the detected shadow. Default is 6.
    max_radius_px : float, optional
        Maximum radius from ``(u0_px, v0_px)`` to search for the beamstop arm.

    Returns
    -------
    ndarray of bool
        Boolean mask (True = inside beamstop shadow).
    """
    accum: NDArray[np.float64] | None = None
    for item in frames:
        im = (
            item.image
            if isinstance(item, Frame)
            else np.asarray(item, dtype=np.float64)
        )
        if accum is None:
            accum = np.zeros(im.shape, dtype=np.float64)
        col_slice = min(300, max(im.shape[1] // 4, 10))
        row_bg = np.median(im[:, :col_slice], axis=1)[:, None]
        accum += (im - row_bg) > event_threshold_adu

    if accum is None:
        raise ValueError("build_beamstop_mask requires at least one frame.")

    nrow, ncol = accum.shape
    yy, xx = np.mgrid[:nrow, :ncol].astype(np.float64)
    u = float(np.clip(geometry.u0_px, 0, ncol - 1))
    v = float(np.clip(geometry.v0_px, 0, nrow - 1))

    c_smooth = ndimage.gaussian_filter(accum, 2.0)
    r = np.hypot(xx - u, yy - v)
    ring = (r > 60.0) & (r < min(200.0, 0.45 * min(nrow, ncol)))
    ref_level = (
        float(np.median(c_smooth[ring])) if np.any(ring) else float(np.median(c_smooth))
    )
    if ref_level <= 0:
        return np.zeros(accum.shape, dtype=bool)

    low = c_smooth < 0.15 * ref_level
    lab, _ = ndimage.label(low & (r < max_radius_px))
    center_label = lab[round(v), round(u)]
    if center_label == 0:
        return np.zeros(accum.shape, dtype=bool)
    mask = ndimage.binary_fill_holes(lab == center_label)
    mask = ndimage.binary_dilation(mask, iterations=dilate_iterations)
    if mask.sum() >= 0.25 * mask.size:
        # Guard against under-exposed frame stacks where the entire center is dark
        return np.zeros(accum.shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _pv(
    da: NDArray[np.float64],
    db: NDArray[np.float64],
    wa: float,
    wb: float,
    eta: float,
) -> NDArray[np.float64]:
    """2D elliptical pseudo-Voigt profile."""
    r2 = (da / wa) ** 2 + (db / wb) ** 2
    return eta / (1.0 + r2) + (1.0 - eta) * np.exp(-np.log(2.0) * r2)


def _bleed(
    q: NDArray[np.float64] | list[float],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Beamstop bleed component: pseudo-Voigt along line times Gaussian across line."""
    B, ac, bc, sa, sb, t_tilt, eta = q
    u = ((a - ac) / sa) ** 2
    fa = (1.0 - eta) * np.exp(-0.5 * u) + eta / (1.0 + 0.5 * u)
    fb = np.exp(-0.5 * ((b - bc - t_tilt * (a - ac)) / sb) ** 2)
    return B * fa * fb


def _mu_strip(
    p: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    M: int,
    K: int,
) -> NDArray[np.float64]:
    """Evaluate full strip model ``C + sum_m PV_m + sum_k bleed_k``."""
    C, wa, wb, eta = p[:4]
    out = np.full(a.shape, C, dtype=np.float64)
    for i in range(M):
        A, am, bm = p[4 + 3 * i : 7 + 3 * i]
        out += A * _pv(a - am, b - bm, wa, wb, eta)
    for k in range(K):
        out += _bleed(
            p[4 + 3 * M + _NB_BLEED * k : 4 + 3 * M + _NB_BLEED * (k + 1)], a, b
        )
    return out


def _mu_strip_jac(
    p: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    M: int,
    K: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate strip model and its exact analytic Jacobian ``d(mu)/d(p)``."""
    C, wa, wb, eta = p[:4]
    mu = np.full(a.shape, C, dtype=np.float64)
    J = np.zeros((a.size, len(p)), dtype=np.float64)
    J[:, 0] = 1.0
    for i in range(M):
        A, am, bm = p[4 + 3 * i : 7 + 3 * i]
        da, db = a - am, b - bm
        r2 = (da / wa) ** 2 + (db / wb) ** 2
        G = np.exp(-np.log(2.0) * r2)
        L = 1.0 / (1.0 + r2)
        pv = eta * L + (1.0 - eta) * G
        dpv = -eta * (L**2) - (1.0 - eta) * np.log(2.0) * G
        mu += A * pv
        j = 4 + 3 * i
        J[:, j] = pv
        J[:, j + 1] = A * dpv * (-2.0 * da / (wa**2))
        J[:, j + 2] = A * dpv * (-2.0 * db / (wb**2))
        J[:, 1] += A * dpv * (-2.0 * (da**2) / (wa**3))
        J[:, 2] += A * dpv * (-2.0 * (db**2) / (wb**3))
        J[:, 3] += A * (L - G)
    for k in range(K):
        j = 4 + 3 * M + _NB_BLEED * k
        B, ac, bc, sa, sb, t_tilt, e = p[j : j + _NB_BLEED]
        x = a - ac
        u = (x / sa) ** 2
        Eg = np.exp(-0.5 * u)
        Lr = 1.0 / (1.0 + 0.5 * u)
        Fa = (1.0 - e) * Eg + e * Lr
        dFa = -0.5 * (1.0 - e) * Eg - 0.5 * e * (Lr**2)
        v = (b - bc - t_tilt * x) / sb
        Fb = np.exp(-0.5 * v * v)
        dFb = -v * Fb
        mu += B * Fa * Fb
        J[:, j] = Fa * Fb
        J[:, j + 1] = B * (dFa * (-2.0 * x / (sa**2)) * Fb + Fa * dFb * (t_tilt / sb))
        J[:, j + 2] = B * Fa * dFb * (-1.0 / sb)
        J[:, j + 3] = B * dFa * (-2.0 * u / sa) * Fb
        J[:, j + 4] = B * Fa * dFb * (-v / sb)
        J[:, j + 5] = B * Fa * dFb * (-x / sb)
        J[:, j + 6] = B * (Lr - Eg) * Fb
    return mu, J


def _poisson_deviance_residuals(
    n: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Signed Poisson deviance residuals ``sign(n - mu) * sqrt(2 * (mu - n +
    n*ln(n/mu)))``.
    """
    mu_c = np.clip(mu, 1e-6, None)
    nn = np.clip(n, 0.0, None)
    n_safe = np.where(nn > 0, nn, 1.0)
    t_val = mu_c - nn + np.where(nn > 0, nn * np.log(n_safe / mu_c), 0.0)
    return np.sign(nn - mu_c) * np.sqrt(2.0 * np.clip(t_val, 0.0, None))


def _poisson_deviance_jac(
    n: NDArray[np.float64],
    W: NDArray[np.float64],
    p: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    M: int,
    K: int,
) -> NDArray[np.float64]:
    """Analytic Jacobian of the signed Poisson deviance residuals."""
    mu, J = _mu_strip_jac(p, a, b, M, K)
    m = np.clip(W * mu, 1e-6, None)
    r = _poisson_deviance_residuals(n, m)
    nn = np.clip(n, 0.0, None)
    r_safe = np.where(np.abs(r) > 1e-6, r, 1.0)
    drdm = np.where(np.abs(r) > 1e-6, (1.0 - nn / m) / r_safe, -1.0 / np.sqrt(m))
    return (drdm * W)[:, None] * J


def order_seeds(
    geometry: Geometry,
    qx_map: NDArray[np.float64],
    qy_map: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    orders: tuple[int, ...] = (-4, -3, -2, -1, 1, 2, 3, 4),
) -> list[tuple[int, float, float]]:
    """Find ``(m, a_m, b_m)`` seeds for diffraction orders falling on the detector."""
    G = geometry.g_inv_nm
    nrow, ncol = qx_map.shape
    yy, xx = np.mgrid[:nrow, :ncol].astype(np.float64)
    margin = 0.48 * min(nrow, ncol)
    cx, cy = 0.5 * ncol, 0.5 * nrow
    out: list[tuple[int, float, float]] = []
    for m in orders:
        s = (np.abs(qy_map) < 0.03 * G) & (np.abs(qx_map - m * G) < 0.03 * G)
        if s.sum() >= 3 and np.all(
            np.abs([xx[s].mean() - cx, yy[s].mean() - cy]) < margin
        ):
            out.append((m, float(a[s].mean()), float(b[s].mean())))
    return out


def origin_shift_from_orders(
    orders: list[dict[str, float]],
    *,
    min_peak_photons: float = 1.0,
) -> tuple[float, int]:
    """Compute diffraction origin shift [px] along the spot line from ``+m / -m`` pairs.

    Returns
    -------
    delta_px : float
        Median pair shift ``0.5 * (da_{+m} + da_{-m})`` in pixels.
    n_pairs : int
        Number of symmetric ``+m / -m`` pairs above ``min_peak_photons``.
    """
    sh = {int(o["m"]): o for o in orders if o["A"] > min_peak_photons}
    pairs = [
        0.5 * (sh[m]["da"] + sh[-m]["da"]) for m in (1, 2, 3, 4) if m in sh and -m in sh
    ]
    d = float(np.median(pairs)) if pairs else 0.0
    return d, len(pairs)


def fit_strip_background(
    geometry: Geometry,
    d0: ArrayLike,
    qx_map: ArrayLike,
    qy_map: ArrayLike,
    mask: ArrayLike | None,
    energy_ev: float,
    exposure_s: float = 1.0,
    *,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
    strip_half_width_px: int = DEFAULT_STRIP_HALF_WIDTH_PX,
) -> tuple[NDArray[np.float64], dict[str, Any], list[dict[str, float]]]:
    """Fit the joint detector-strip model (flat floor + order pseudo-Voigts + beamstop
    bleed).

    Fits by maximum-likelihood Poisson deviance on 2px (along) x 1px (across)
    bins over the strip ``|b - b_line| < strip_half_width_px``, selecting the
    number of beamstop bleed components ``K in {1, 2, 3}`` via BIC, and returns
    only the ``C + bleed`` background model on the full 2D frame so that Bragg
    orders and diffuse tails are preserved when subtracting.

    Parameters
    ----------
    geometry : Geometry
        Frame geometry.
    d0 : array-like
        Offset-subtracted 2D detector image (ADU).
    qx_map, qy_map : array-like
        2D ``Qx`` and ``Qy`` maps (1/nm).
    mask : array-like of bool or None
        Beamstop mask (True = masked).
    energy_ev : float
        Photon energy in eV.
    exposure_s : float, optional
        Exposure time in seconds. Default is ``1.0``.
    adu_per_photon : float, optional
        ADU per photon conversion factor. Default is ``27.0``.
    strip_half_width_px : int, optional
        Half-width of strip band around ``Qy = 0`` in pixels. Default is 18.

    Returns
    -------
    bg_model_adu : ndarray of float64
        2D background image ``(C + sum_k bleed_k) * adu_per_photon`` in ADU.
    params : dict
        Fitted model parameters and BIC / deviance diagnostics.
    orders : list of dict
        Fitted order peaks ``{"m": m, "A": A, "da": da, "db": db}``.
    """
    d_arr = np.asarray(d0, dtype=np.float64)
    qx = np.asarray(qx_map, dtype=np.float64)
    qy = np.asarray(qy_map, dtype=np.float64)
    mask_arr = (
        np.asarray(mask, dtype=bool)
        if mask is not None
        else np.zeros(d_arr.shape, dtype=bool)
    )
    nrow, ncol = d_arr.shape
    yy, xx = np.mgrid[:nrow, :ncol].astype(np.float64)
    a, b = geometry.streak_coords(xx, yy)
    R1 = geometry.r1_px(energy_ev)

    seeds = order_seeds(geometry, qx, qy, a, b)
    M = len(seeds)
    if M < 2:
        return np.zeros_like(d_arr), {"K": 0, "M": M}, []

    sa_s = np.array([s[1] for s in seeds], dtype=np.float64)
    sb_s = np.array([s[2] for s in seeds], dtype=np.float64)
    bl = np.polyfit(sa_s, sb_s, 1)
    band = (
        (~mask_arr)
        & np.isfinite(d_arr)
        & (np.abs(b - np.polyval(bl, a)) < strip_half_width_px)
        & (np.abs(a) < _REACH_R1 * R1)
    )
    if band.sum() < 20:
        return np.zeros_like(d_arr), {"K": 0, "M": M}, []

    ia = np.floor(a[band] / 2.0).astype(int)
    ib = np.floor(b[band] - np.polyval(bl, a[band])).astype(int)
    key = (ia - ia.min()) * 64 + (ib + 32)
    npx = np.bincount(key)
    kk = np.nonzero(npx)[0]
    W = npx[kk].astype(np.float64)
    A_ = np.bincount(key, a[band])[kk] / W
    B_ = np.bincount(key, b[band])[kk] / W
    n = np.bincount(key, gate_read_noise(d_arr)[band] / adu_per_photon)[kk]

    mask_on_line = mask_arr & (np.abs(b - np.polyval(bl, a)) < 3.0)
    edge = float(np.abs(a[mask_on_line]).max()) if np.any(mask_on_line) else 0.3 * R1
    C0 = max(float(np.median(n / W)), 0.0)

    def _near(am: float, bm: float, r: float = 4.0) -> NDArray[np.float64]:
        return (n / W)[(np.abs(A_ - am) < r) & (np.abs(B_ - bm) < r)]

    p0: list[float] = [C0, 2.0, 2.0, 0.3]
    lo: list[float] = [-0.05, 0.7, 0.7, 0.0]
    hi: list[float] = [5.0, 15.0, 15.0, 1.0]
    for _, am, bm in seeds:
        v_near = _near(am, bm)
        p0 += [max(float(v_near.max()) if v_near.size else 1.0, 0.5), am, bm]
        lo += [0.0, am - 12.0, bm - 5.0]
        hi += [np.inf, am + 12.0, bm + 5.0]

    side_I = [
        float(
            np.sum(
                (n / W)[
                    (sgn * A_ > edge)
                    & (sgn * A_ < edge + 0.3 * R1)
                    & (np.abs(B_ - np.polyval(bl, A_)) < 4.0)
                ]
            )
        )
        for sgn in (1, -1)
    ]
    sb_sgn = 1 if side_I[0] >= side_I[1] else -1
    v_edge = (n / W)[
        (sb_sgn * A_ > edge)
        & (sb_sgn * A_ < edge + 10.0)
        & (np.abs(B_ - np.polyval(bl, A_)) < 4.0)
    ]
    B0 = max(float(np.mean(v_edge) * 3.0) if v_edge.size else 0.5, 0.1)

    def _bpeak(sgn: int, x0: float, x1: float) -> float:
        s = (
            (sgn * A_ > edge + x0 * R1)
            & (sgn * A_ < edge + x1 * R1)
            & (np.abs(B_ - np.polyval(bl, A_)) < 10.0)
        )
        if not np.any(s):
            return float(np.polyval(bl, sgn * edge))
        kb = np.rint(B_[s]).astype(int)
        h = np.bincount(kb - kb.min(), n[s])
        return float(np.argmax(h) + kb.min())

    comps = [
        (B0, sb_sgn * 0.8 * edge, _bpeak(sb_sgn, 0.15, 0.5), 0.3 * R1, 1.5),
        (B0, sb_sgn * 0.95 * edge, _bpeak(sb_sgn, 0.0, 0.1), 0.08 * R1, 5.0),
        (0.3 * B0, -sb_sgn * 0.8 * edge, _bpeak(-sb_sgn, 0.15, 0.5), 0.2 * R1, 1.5),
    ]
    best: tuple[float, int, NDArray[np.float64], float] | None = None
    pk, lk, hk = list(p0), list(lo), list(hi)
    for K, (Bi, aci, bci, sai, sbi) in enumerate(comps, start=1):
        pk += [Bi, aci, bci, sai, sbi, 0.0, 0.2]
        lk += [0.0, -edge, -12.0, 3.0, 1.2, -0.15, 0.0]
        hk += [np.inf, edge, 12.0, max(_BLEED_SA_MAX_R1 * R1, 3.5), 12.0, 0.15, 1.0]
        pk = list(np.clip(pk, np.asarray(lk) + 1e-9, np.asarray(hk) - 1e-9))
        fr = least_squares(
            lambda p, k_cur=K: _poisson_deviance_residuals(
                n, W * _mu_strip(p, A_, B_, M, k_cur)
            ),
            pk,
            jac=lambda p, k_cur=K: _poisson_deviance_jac(n, W, p, A_, B_, M, k_cur),
            bounds=(lk, hk),
            x_scale="jac",
            max_nfev=100,
        )
        chi = float(np.sum(fr.fun**2))
        bic = chi + len(pk) * float(np.log(max(n.size, 1)))
        if best is None or bic < best[0]:
            best = (bic, K, fr.x.copy(), chi)
        pk = list(fr.x)

    assert best is not None
    _, K_best, p_best, chi_best = best
    model = np.full(d_arr.shape, p_best[0], dtype=np.float64)
    for k in range(K_best):
        model += _bleed(
            p_best[4 + 3 * M + _NB_BLEED * k : 4 + 3 * M + _NB_BLEED * (k + 1)], a, b
        )
    orders_out = [
        {
            "m": float(s[0]),
            "A": float(p_best[4 + 3 * i]),
            "da": float(p_best[5 + 3 * i] - s[1]),
            "db": float(p_best[6 + 3 * i] - s[2]),
        }
        for i, s in enumerate(seeds)
    ]
    par_out: dict[str, Any] = {
        "K": K_best,
        "M": M,
        "C_cps": float(p_best[0] * adu_per_photon / exposure_s),
        "wa_px": float(p_best[1]),
        "wb_px": float(p_best[2]),
        "eta": float(p_best[3]),
        "dev_nu": float(chi_best / max(n.size - len(p_best), 1)),
        "n_bins": int(n.size),
        "bleed": [
            {
                "B_cps": float(q[0] * adu_per_photon / exposure_s),
                "ac_px": float(q[1]),
                "bc_px": float(q[2]),
                "sa_px": float(q[3]),
                "sb_px": float(q[4]),
                "tilt": float(q[5]),
                "eta": float(q[6]),
            }
            for q in [
                p_best[4 + 3 * M + _NB_BLEED * k : 4 + 3 * M + _NB_BLEED * (k + 1)]
                for k in range(K_best)
            ]
        ],
    }
    return model * adu_per_photon, par_out, orders_out


@dataclass(slots=True)
class ProcessedFrameStrip:
    """Output of the 5-step per-frame detector-strip background subtraction and Q
    mapping.
    """

    cleaned_adu: NDArray[np.float64]
    before_adu: NDArray[np.float64]
    bg_model_adu: NDArray[np.float64]
    params: dict[str, Any]
    orders: list[dict[str, float]]
    origin_shift_px: float
    qmap: PixelQMap


def process_frame_strip(
    geometry: Geometry,
    image: ArrayLike,
    energy_ev: float,
    omega_deg: float,
    mask: ArrayLike | None = None,
    exposure_s: float = 1.0,
    *,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
) -> ProcessedFrameStrip:
    """Run the 5-step detector-strip background subtraction and origin-shifted Q
    mapping for one frame.

    Steps:
    1. Row/column detector offset removal (``subtract_row_col_offset``).
    2. Joint strip model fit (``fit_strip_background``).
    3. Subtract ``C + bleed`` and gate read noise at 15 ADU.
    4. Measure diffraction origin shift from ``+m / -m`` pairs and recompute Q maps
       if ``|delta_px| > 0.3``.
    5. Mask beamstop pixels with ``NaN``.
    """
    im = np.asarray(image, dtype=np.float64)
    mask_arr = (
        np.asarray(mask, dtype=bool)
        if mask is not None
        else np.zeros(im.shape, dtype=bool)
    )
    xx, yy = frame_pixel_grid(im.shape, bin_px=1)
    qmap = compute_pixel_q(geometry, xx, yy, energy_ev, omega_deg)
    d0 = subtract_row_col_offset(im, geometry, qmap.Qy, mask_arr)
    bg, par, orders = fit_strip_background(
        geometry,
        d0,
        qmap.Qx,
        qmap.Qy,
        mask_arr,
        energy_ev,
        exposure_s,
        adu_per_photon=adu_per_photon,
    )
    delta, _ = origin_shift_from_orders(orders)
    if abs(delta) > 0.3:
        qmap = compute_pixel_q(
            geometry.with_origin_shift(delta), xx, yy, energy_ev, omega_deg
        )
    out = gate_read_noise(d0 - bg)
    out[mask_arr] = np.nan
    before = np.where(mask_arr, np.nan, gate_read_noise(d0))
    bg_masked = np.where(mask_arr, np.nan, bg)
    return ProcessedFrameStrip(
        cleaned_adu=out,
        before_adu=before,
        bg_model_adu=bg_masked,
        params=par,
        orders=orders,
        origin_shift_px=delta,
        qmap=qmap,
    )


def subtract_offline_band_background(
    density: ArrayLike,
    qx_map: ArrayLike,
    qy_map: ArrayLike,
    g_inv_nm: float,
    qx_edges: ArrayLike,
    *,
    qy_off_g: tuple[float, float] = (0.15, 0.40),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract non-Bragg background profile estimated from an off-line ``|Qy|`` band.

    Within each ``Qx`` bin defined by ``qx_edges``, estimates the mean background
    density in ``qy_off_g[0] * G < |Qy| < qy_off_g[1] * G`` and subtracts it from
    ``density``.

    Parameters
    ----------
    density : array-like
        Per-pixel intensity density (e.g. ``ADU / sr / s``).
    qx_map, qy_map : array-like
        ``Qx`` and ``Qy`` coordinates of each pixel (1/nm).
    g_inv_nm : float
        Reciprocal lattice constant ``G = 2 * pi / Lambda_nm`` (1/nm).
    qx_edges : array-like
        Bin edges along ``Qx`` (1/nm).
    qy_off_g : tuple of (float, float), optional
        Lower and upper ``|Qy| / G`` bounds for the off-line background band.
        Default is ``(0.15, 0.40)``.

    Returns
    -------
    density_bgsub : ndarray of float64
        Background-subtracted density array of the same shape as ``density``.
    bg_profile : ndarray of float64
        1D background profile of length ``len(qx_edges) - 1``.
    """
    dens = np.asarray(density, dtype=np.float64)
    qx = np.asarray(qx_map, dtype=np.float64)
    qy = np.asarray(qy_map, dtype=np.float64)
    edges = np.asarray(qx_edges, dtype=np.float64)
    nbins = len(edges) - 1

    off_band = (
        (np.abs(qy) > qy_off_g[0] * g_inv_nm)
        & (np.abs(qy) < qy_off_g[1] * g_inv_nm)
        & np.isfinite(dens)
    )
    if not np.any(off_band):
        return dens.copy(), np.zeros(nbins, dtype=np.float64)

    ix_off = np.clip(np.digitize(qx[off_band], edges) - 1, 0, nbins - 1)
    bsum = np.bincount(ix_off, weights=dens[off_band], minlength=nbins)
    bcnt = np.bincount(ix_off, minlength=nbins)
    bprof = np.where(bcnt > 0, bsum / np.maximum(bcnt, 1), 0.0)

    ix_all = np.clip(np.digitize(qx, edges) - 1, 0, nbins - 1)
    return dens - bprof[ix_all], bprof


def apply_geometric_corrections(
    image: ArrayLike,
    qmap: PixelQMap,
    *,
    exposure_s: float = 1.0,
    i0: float = 1.0,
    solid_angle: bool = True,
    polarization: bool = False,
    walkoff_f: float = 1.0,
    membrane_t: float = 1.0,
) -> NDArray[np.float64]:
    """Convert a 2D detector image (ADU) to corrected scattering density.

    Divides pixel counts by exposure time, incident monitor ``i0``, pixel solid
    angle ``Omega_pix`` (if ``solid_angle=True``), dipole polarization factor
    (if ``polarization=True``), beam walk-off factor ``walkoff_f``, and window
    transmission ``membrane_t``.

    Returns
    -------
    ndarray of float64
        Corrected intensity density (e.g. ``ADU / s / sr``).
    """
    arr = np.asarray(image, dtype=np.float64).copy()
    denom = float(exposure_s * i0 * walkoff_f * membrane_t)
    if denom <= 0:
        raise ValueError(f"Combined normalization factor must be positive, got {denom}")
    out = arr / denom
    if solid_angle:
        out = out / np.maximum(qmap.solid_angle_sr, 1e-18)
    if polarization:
        out = out / np.maximum(qmap.polarization_factor, 1e-6)
    return out
