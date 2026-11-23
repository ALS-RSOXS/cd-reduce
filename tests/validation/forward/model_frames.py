"""Analytic HOWTO forward model: PSDs in, Poisson detector frame out.

Inverts the definitions in ``cd_reduce.reduction.roughness.reduce``.
"""

from __future__ import annotations

import numpy as np

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import Geometry, compute_pixel_q, frame_pixel_grid
from cd_reduce.reduction.mapping import pixel_q_area
from cd_reduce.reduction.roughness import (
    CrossSection,
    absorption_factor,
    cos_theta_out,
    howto_prefactor,
    order_efficiency,
)

_HC_EV_NM = 1239.84198


def render_model_frame(
    xs: CrossSection,
    geometry: Geometry,
    omega_deg: float,
    *,
    psd_lpr,
    psd_lwr,
    scale: float,
    shape: tuple[int, int],
    exposure_s: float = 1.0,
    adu_per_photon: float,
    window_T: float = 1.0,
    sigma_Q: float = 0.0015,
    sigma_ler_full: float,
    max_order: int = 5,
    beamstop_radius_px: float = 30,
    rng: np.random.Generator,
    poisson: bool = True,
) -> Frame:
    """Render one frame (ADU) for the given cross section and PSDs.

    With ``poisson=False`` the expected counts are returned without noise.
    """
    E = xs.energy_ev
    k = 2 * np.pi * E / _HC_EV_NM
    uu, vv = frame_pixel_grid(shape)
    qm = compute_pixel_q(geometry, uu, vv, E, omega_deg, approximation="correct")
    W = np.radians(omega_deg + geometry.omega0_deg)
    Qx = np.asarray(qm.Qx, dtype=float)
    Qy = np.asarray(qm.Qy, dtype=float)
    Qz = np.asarray(qm.Qz, dtype=float)
    f = Qy / (2 * np.pi)

    cos_out = cos_theta_out(qm, k, W)
    A = absorption_factor(xs.mu_h, W, cos_out)
    Tm = window_T ** (1.0 / np.cos(W))
    norm = howto_prefactor(k, xs, W) * np.asarray(qm.solid_angle_sr) * A * Tm
    contrast = xs.gamma2(Qz) * np.exp(-(Qx**2) * xs.s**2)

    M = np.zeros_like(Qx)
    if psd_lpr is not None:
        M += 4 * np.sin(Qx * xs.w / 2) ** 2 * psd_lpr(f)
    if psd_lwr is not None:
        M += np.cos(Qx * xs.w / 2) ** 2 * psd_lwr(f)
    lam = scale * exposure_s * M * contrast * norm

    area = pixel_q_area(Qx, Qy)
    G = 2 * np.pi / xs.L
    for m in [j for j in range(-max_order, max_order + 1) if j]:
        r2 = (Qx - m * G) ** 2 + Qy**2
        near = r2 < (8 * sigma_Q) ** 2
        if not near.any():
            continue
        eta = order_efficiency(
            xs, m, omega_deg, E, Qz[near],
            sigma_ler=sigma_ler_full,
            omega0_deg=geometry.omega0_deg,
            window_T=window_T,
        )  # fmt: skip
        dens = np.exp(-r2[near] / (2 * sigma_Q**2)) / (2 * np.pi * sigma_Q**2)
        lam[near] += scale * exposure_s * eta * dens * area[near]

    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    img = (rng.poisson(lam).astype(float) if poisson else lam) * adu_per_photon
    r2_bs = (uu - geometry.u0_px) ** 2 + (vv - geometry.v0_px) ** 2
    bs = r2_bs <= beamstop_radius_px**2
    img[bs] = np.nan
    return Frame.from_array(
        img, energy_ev=E, omega_deg=omega_deg, exposure_s=exposure_s
    )
