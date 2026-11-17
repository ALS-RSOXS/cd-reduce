"""Synthetic detector frames of a rough line grating (Bragg spots + Born diffuse).

Used to validate the roughness extraction end to end on real detector geometry
(pixel Q mapping, pixel areas, Ewald curvature, Poisson noise), and to estimate
the photon budget needed for a given PSD level.

Model (transmission, box lines of width ``w`` and height ``h``, roughness
uncorrelated between lines, conformal in ``z``):

- Bragg order ``m``: integrated photons ``N_m = flux * F_m^2 * |F_z(Qz)|^2``,
  ``F_m = 2 sin(m G w/2) / (m G)``, ``F_z = h sinc(Qz h / 2)``, spread as a
  Gaussian of width ``spot_sigma_inv_nm`` in ``(Qx, Qy)``.
- Diffuse density (photons per unit ``Qx Qy`` area)
  ``flux |F_z|^2 p / (2 pi)^2 [4 sin^2(Qx w/2) S_LPR + cos^2(Qx w/2) S_LWR]``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from cd_reduce.reduction.corrections import DEFAULT_ADU_PER_PHOTON
from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import (
    ApproximationMode,
    Geometry,
    compute_pixel_q,
    frame_pixel_grid,
)
from cd_reduce.reduction.mapping import pixel_q_area

PsdFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def synthetic_rough_grating_frame(
    geometry: Geometry,
    omega_deg: float,
    *,
    line_width_nm: float,
    height_nm: float = 60.0,
    psd_lpr: PsdFn | None = None,
    psd_lwr: PsdFn | None = None,
    energy_ev: float = 285.2,
    shape: tuple[int, int] = (512, 512),
    flux: float = 1e3,
    max_order: int = 5,
    spot_sigma_inv_nm: float = 2.5e-3,
    beamstop_qx_g: float = 0.5,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
    poisson: bool = True,
    approximation: ApproximationMode = "correct",
    rng: np.random.Generator | int | None = None,
) -> Frame:
    """Render one synthetic frame (ADU) of a rough grating at incidence ``omega``.

    ``flux`` scales photons: order ``m`` holds ``flux * F_m^2 |F_z|^2`` photons
    (``F`` in nm). Pixels with ``|Qx| < beamstop_qx_g * G`` are set to ``NaN``.
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    xx, yy = frame_pixel_grid(shape, 1)
    qm = compute_pixel_q(
        geometry, xx, yy, energy_ev, omega_deg, approximation=approximation
    )
    G = geometry.g_inv_nm
    p = geometry.Lambda_nm
    w = float(line_width_nm)
    qx, qy, qz = qm.Qx, qm.Qy, qm.Qz
    area = pixel_q_area(qx, qy)
    fz2 = (height_nm * np.sinc(qz * height_nm / (2.0 * np.pi))) ** 2

    zero = np.zeros_like(qy)
    s_p = psd_lpr(np.abs(qy)) if psd_lpr is not None else zero
    s_w = psd_lwr(np.abs(qy)) if psd_lwr is not None else zero
    dens = (
        flux
        * fz2
        * p
        / (2.0 * np.pi) ** 2
        * (4.0 * np.sin(qx * w / 2.0) ** 2 * s_p + np.cos(qx * w / 2.0) ** 2 * s_w)
    )
    photons = dens * area

    s2 = spot_sigma_inv_nm**2
    for m in range(-max_order, max_order + 1):
        if m == 0:
            continue
        fm2 = (2.0 * np.sin(m * G * w / 2.0) / (m * G)) ** 2
        g = np.exp(-0.5 * ((qx - m * G) ** 2 + qy**2) / s2) / (2.0 * np.pi * s2)
        photons = photons + flux * fm2 * fz2 * g * area

    if poisson:
        photons = gen.poisson(np.clip(photons, 0.0, None)).astype(np.float64)
    img = photons * adu_per_photon
    img = np.where(np.abs(qx) < beamstop_qx_g * G, np.nan, img)
    return Frame.from_array(
        img, omega_deg=float(omega_deg), energy_ev=energy_ev, exposure_s=1.0
    )


__all__ = ["synthetic_rough_grating_frame"]
