"""Detector frames rendered from a Born ensemble of explicit rough gratings.

The ensemble lives on the supercell mode grid ``(2 pi/Lx) Z x (2 pi/Ly) Z``; the
diffuse intensity per mode becomes a density per unit ``(Qx, Qy)`` area by the
factor ``Lx Ly / 4 pi^2`` and is integrated over each pixel's ``(Qx, Qy)`` area.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import Geometry, compute_pixel_q, frame_pixel_grid
from cd_reduce.reduction.mapping import pixel_q_area
from cd_reduce.reduction.roughness import (
    CrossSection,
    absorption_factor,
    cos_theta_out,
)
from cd_reduce.simulation.rough_scattering import RoughEnsembleResult

_HC_EV_NM = 1239.84198


def _interp(qx, qy, values, Qx, Qy) -> np.ndarray:
    """Linear interpolation from the mode grid to pixels; NaN outside or invalid."""
    f = RegularGridInterpolator(
        (qx, qy), values, method="linear", bounds_error=False, fill_value=np.nan
    )
    return f(np.stack([Qx, Qy], axis=-1))


def render_ensemble_frame(
    result: RoughEnsembleResult,
    i_omega: int,
    xs: CrossSection,
    geometry: Geometry,
    *,
    scale: float,
    shape: tuple[int, int],
    exposure_s: float = 1.0,
    adu_per_photon: float,
    window_T: float = 1.0,
    sigma_Q: float = 0.0015,
    beamstop_radius_px: float = 30,
    rng: np.random.Generator,
    poisson: bool = True,
) -> Frame:
    """Render one frame (ADU) of ensemble ``result`` at tilt index ``i_omega``.

    The frame's ``omega_deg`` is ``-result.omega_deg[i_omega] - omega0_deg``
    (opposite tilt sense of the simulation, checked by the psi guard).

    Pixels outside the stored mode grid (or on invalid modes) are NaN, as is the
    beamstop. Orders ``m != 0`` on the stored grid are Gaussians of rms ``sigma_Q``
    carrying the flux efficiency ``|<A>(mG, 0)|^2 cos W / cos theta_out`` photons
    per incident photon (the Debye-Waller factor is already in ``<A>``). The
    diffuse uses ``diffuse_flux`` (same flux factor).
    """
    E = result.energy_ev
    # The simulation tilts the beam by +omega_sim in the sample frame (Qz ~
    # -qx sin omega_sim); the detector geometry's W = omega + omega0 has the
    # opposite sense (Qz ~ +Qx sin W). The frame therefore has W = -omega_sim.
    om = -float(result.omega_deg[i_omega]) - geometry.omega0_deg
    k = 2 * np.pi * E / _HC_EV_NM
    uu, vv = frame_pixel_grid(shape)
    qm = compute_pixel_q(geometry, uu, vv, E, om, approximation="correct")
    Qx, Qy, Qz = (np.asarray(a, dtype=float) for a in (qm.Qx, qm.Qy, qm.Qz))
    W = np.radians(om + geometry.omega0_deg)

    diffuse = _interp(result.qx, result.qy, result.diffuse_flux[i_omega], Qx, Qy)
    qz_mode = _interp(result.qx, result.qy, result.qz[i_omega], Qx, Qy)

    # psi guard: the pixel and the interpolated mode must sit on the same Ewald
    # surface to well within one mode-grid Qz step.
    qz_grid = result.qz[i_omega]
    step = np.nanmax(np.abs(np.diff(qz_grid, axis=0)))
    ok = np.isfinite(qz_mode) & np.isfinite(Qz)
    resid = float(np.max(np.abs(Qz[ok] - qz_mode[ok]))) if ok.any() else 0.0
    assert resid < step, f"pixel/mode Qz residual {resid:.3g} >= grid step {step:.3g}"

    if result.engine == "born":
        a_eng = absorption_factor(xs.mu_h, W, cos_theta_out(qm, k, W))
    else:
        a_eng = np.ones_like(Qx)
    Tm = window_T ** (1.0 / np.cos(W))
    lx = result.n_lines * result.pitch_nm
    density = diffuse * (lx * result.ly_nm / (4 * np.pi**2))
    area = pixel_q_area(Qx, Qy)
    lam = scale * exposure_s * density * area * Tm * a_eng

    G = 2 * np.pi / result.pitch_nm
    m_max = int(np.floor(result.qx.max() / G + 1e-9))
    for m in [j for j in range(-m_max, m_max + 1) if j]:
        try:
            eff = result.order_flux_efficiency(m)[i_omega]
            qz_m = result.order_qz(m)[i_omega]
        except KeyError:
            continue
        if not np.isfinite(eff):
            continue
        r2 = (Qx - m * G) ** 2 + Qy**2
        near = r2 < (8 * sigma_Q) ** 2
        if not near.any():
            continue
        if result.engine == "born":
            cos_out = np.cos(W) + qz_m / k
            a_m = float(absorption_factor(xs.mu_h, W, np.asarray(cos_out)))
        else:
            a_m = 1.0
        dens = np.exp(-r2[near] / (2 * sigma_Q**2)) / (2 * np.pi * sigma_Q**2)
        lam[near] += scale * exposure_s * eff * Tm * a_m * dens * area[near]

    outside = ~np.isfinite(lam)
    lam = np.where(outside, 0.0, np.maximum(lam, 0.0))
    img = (rng.poisson(lam).astype(float) if poisson else lam) * adu_per_photon
    img[outside] = np.nan
    r2_bs = (uu - geometry.u0_px) ** 2 + (vv - geometry.v0_px) ** 2
    img[r2_bs <= beamstop_radius_px**2] = np.nan
    return Frame.from_array(img, energy_ev=E, omega_deg=om, exposure_s=exposure_s)


def ensemble_vs_model_ratio(
    frame_ens: Frame,
    frame_model: Frame,
    geometry: Geometry,
    *,
    pitch_nm: float,
    level: float = 0.02,
    disc_sigma: float = 8.0,
    sigma_Q: float = 0.0015,
    block_px: tuple[int, int] = (1, 1),
    qx_max: float | None = None,
    null_level: float | None = None,
    qy_bin: float = 0.005,
) -> np.ndarray:
    """Ensemble/model ratio over diffuse pixels above ``level`` x the diffuse maximum.

    Pixels within ``disc_sigma * sigma_Q`` of an order (m != 0), on the beamstop,
    or NaN in either frame are excluded. With ``block_px = (bu, bv)`` both frames
    are first summed over non-overlapping blocks of excluded-free pixels, so that
    each ratio averages several independent ensemble modes (the per-mode diffuse
    of a small ensemble is exponentially distributed). ``qx_max`` (nm^-1) keeps
    only pixels with ``|Qx| <= qx_max`` (after the level threshold).
    ``null_level`` drops pixels whose model value is below ``null_level`` x the
    maximum model value (same exclusions) in their ``|Qy|`` bin of width
    ``qy_bin``: near the nulls of the first-order edge factor the model is ~0 and
    the ratio is ill-defined (R22). Blocks touching a dropped pixel are dropped.
    """
    a = np.asarray(frame_ens.image, dtype=float)
    b = np.asarray(frame_model.image, dtype=float)
    uu, vv = frame_pixel_grid(a.shape)
    qm = compute_pixel_q(
        geometry, uu, vv, frame_model.energy_ev, frame_model.omega_deg,
        approximation="correct",
    )  # fmt: skip
    Qx, Qy = np.asarray(qm.Qx, dtype=float), np.asarray(qm.Qy, dtype=float)
    G = 2 * np.pi / pitch_nm
    keep = np.isfinite(a) & np.isfinite(b)
    mmax = int(np.ceil(np.nanmax(np.abs(Qx)) / G)) + 1
    for m in range(-mmax, mmax + 1):
        if m:
            keep &= (Qx - m * G) ** 2 + Qy**2 >= (disc_sigma * sigma_Q) ** 2
    if null_level is not None:
        ib = np.floor(np.abs(Qy) / qy_bin).astype(int)
        peak = np.zeros(ib.max() + 1)
        np.maximum.at(peak, ib[keep], b[keep])
        null = keep & (b < null_level * peak[ib])
    keep &= b >= level * np.max(b[keep])
    if null_level is not None:
        keep &= ~null
    if qx_max is not None:
        keep &= np.abs(Qx) <= qx_max
    bu, bv = block_px
    nu, nv = a.shape[0] // bu, a.shape[1] // bv
    sl = (slice(0, nu * bu), slice(0, nv * bv))
    k4 = keep[sl].reshape(nu, bu, nv, bv)
    full = k4.all(axis=(1, 3))
    sa = np.where(keep, a, 0.0)[sl].reshape(nu, bu, nv, bv).sum(axis=(1, 3))
    sb = np.where(keep, b, 0.0)[sl].reshape(nu, bu, nv, bv).sum(axis=(1, 3))
    return sa[full] / sb[full]
