"""Matched-realisation first-order HOWTO prediction for a Born ensemble (R25).

For each realisation and line j the first-order edge amplitude is
``2i sin(Qx w/2) P~_j(Qy) - cos(Qx w/2) W~_j(Qy)`` times ``exp(-i Qx x_j)``
(package convention u_L = P - W/2, u_R = P + W/2), summed coherently over the
lines. Its realisation variance (the engine's ``ddof = 1`` estimator, so the
coherent mean is removed like in ``diffuse``), periodogram-normalised per line,
replaces ``4 sin^2 P + cos^2 W`` in `render_model_frame`; contrast, flux,
absorption and order terms are therefore identical to the averaged-spectrum
model. It keeps the cross-line and cross-realisation terms that the
line-averaged realised spectra miss for a small ensemble.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cd_reduce.reduction.geometry import compute_pixel_q, frame_pixel_grid

from .ensemble_frames import frame_omega_deg
from .model_frames import render_model_frame


def mode_grid_M(result, edges, pitch_nm: float, w_nm: float, ly_nm: float):
    """Matched ``M(Qx, Qy)`` (nm^3) on the ensemble mode grid, shape (nqx, nqy)."""
    nl, ny = edges[0][0].shape
    dy = ly_nm / ny
    iy = result.metadata["iy_full"]
    qx = result.qx[:, None]
    phase = np.exp(-1j * np.outer(result.qx, np.arange(nl) * pitch_nm))
    amps = []
    for p, w in edges:
        sp = phase @ np.fft.fft(p, axis=-1)[:, iy]
        sw = phase @ np.fft.fft(w, axis=-1)[:, iy]
        amps.append(2j * np.sin(qx * w_nm / 2) * sp - np.cos(qx * w_nm / 2) * sw)
    var = np.var(np.array(amps), axis=0, ddof=1)
    return var * dy**2 / ly_nm / nl


def matched_model_frames(result, edges, xs, geometry, *, scale, shape, adu_per_photon,
                         sigma_ler_full, ly_nm):  # fmt: skip
    """Noiseless matched-model frames, one per tilt of ``result``."""
    Mg = mode_grid_M(result, edges, xs.L, xs.w, ly_nm)
    interp = RegularGridInterpolator(
        (result.qx, result.qy), Mg, method="linear", bounds_error=False,
        fill_value=np.nan,
    )  # fmt: skip
    uu, vv = frame_pixel_grid(shape)
    frames = []
    for om in frame_omega_deg(result.omega_deg, geometry):
        qm = compute_pixel_q(
            geometry, uu, vv, xs.energy_ev, om, approximation="correct"
        )
        M = interp(np.stack([np.asarray(qm.Qx), np.asarray(qm.Qy)], axis=-1))
        frames.append(render_model_frame(
            xs, geometry, float(om), psd_lpr=None, psd_lwr=None, scale=scale,
            shape=shape, adu_per_photon=adu_per_photon,
            sigma_ler_full=sigma_ler_full, rng=np.random.default_rng(0),
            poisson=False, M=np.nan_to_num(M),
        ))  # fmt: skip
    return frames
