"""Integrate Bragg-order photons from frames (shared by validation tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import Geometry, compute_pixel_q, frame_pixel_grid


def order_rows(
    frame: Frame,
    geometry: Geometry,
    *,
    adu_per_photon: float,
    orders=(-3, -2, -1, 1, 2, 3),
    sigma_Q: float = 0.0015,
) -> list[dict]:
    """One row per order: disc (5 sigma_Q) minus local diffuse from a 6-8 sigma_Q ring.

    ``I_cps`` is ADU/s; ``Qz`` is the photon-weighted mean over the disc; the
    uncertainty is Poisson from the disc and (scaled) ring photons.
    """
    E, om, t = frame.energy_ev, frame.omega_deg, frame.exposure_s
    uu, vv = frame_pixel_grid(frame.image.shape)
    qm = compute_pixel_q(geometry, uu, vv, E, om, approximation="correct")
    Qx, Qy, Qz = (np.asarray(a, dtype=float) for a in (qm.Qx, qm.Qy, qm.Qz))
    ph = np.asarray(frame.image, dtype=float) / adu_per_photon
    G = geometry.g_inv_nm
    rows = []
    for m in orders:
        r = np.hypot(Qx - m * G, Qy)
        disc = r <= 5 * sigma_Q
        ring = (r >= 6 * sigma_Q) & (r <= 8 * sigma_Q)
        if disc.sum() < 5 or ring.sum() < 5:
            continue
        if np.isnan(ph[disc | ring]).any():
            continue
        n_d, n_r = int(disc.sum()), int(ring.sum())
        p_d, p_r = ph[disc], ph[ring]
        net = p_d.sum() - n_d * p_r.mean()
        var = p_d.sum() + (n_d / n_r) ** 2 * p_r.sum()
        w = np.maximum(p_d, 0.0)
        qz = float(np.sum(w * Qz[disc]) / w.sum()) if w.sum() > 0 else np.nan
        rows.append(
            {
                "order": m,
                "energy_ev": E,
                "Qz": qz,
                "I_cps": net * adu_per_photon / t,
                "sigma_cps": np.sqrt(var) * adu_per_photon / t,
                "omega_deg": om,
            }
        )
    return rows


def order_table(frames, geometry: Geometry, **kw) -> pd.DataFrame:
    """Stack :func:`order_rows` for several frames into an order table."""
    rows = [r for fr in frames for r in order_rows(fr, geometry, **kw)]
    return pd.DataFrame(rows)
