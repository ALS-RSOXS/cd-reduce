"""Per-pixel reduction of detector frames to the roughness quantity M.

``M = 4 sin^2(Qx w/2) S_LPR + cos^2(Qx w/2) S_LWR`` (nm^3) is recovered pixel
by pixel by dividing out the kinematic prefactor, solid angle, absorption,
window transmission and the cross-section contrast ``|Gamma(Qz)|^2 exp(-Qx^2 s^2)``.
No PSD model is used here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import Geometry, OmegaRefinement, PixelQMap
from cd_reduce.reduction.mapping import BackgroundMode, prepare_frame
from cd_reduce.reduction.roughness.cross_section import CrossSection

_HC_EV_NM = 1239.84198
_ARRAYS = (
    "Qx", "Qy", "Qz", "f", "D", "M", "var_M", "energy_ev", "omega_deg", "frame_id"
)  # fmt: skip


@dataclass(frozen=True)
class ReducedPixels:
    """Valid reduced pixels (1D arrays of equal length)."""

    Qx: NDArray[np.float64]
    Qy: NDArray[np.float64]
    Qz: NDArray[np.float64]
    f: NDArray[np.float64]
    D: NDArray[np.float64]
    M: NDArray[np.float64]
    var_M: NDArray[np.float64]
    energy_ev: NDArray[np.float64]
    omega_deg: NDArray[np.float64]
    frame_id: NDArray[np.int64]
    m_per_photon: NDArray[np.float64] | None = None
    d_per_photon: NDArray[np.float64] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def select(self, mask: NDArray[np.bool_]) -> ReducedPixels:
        """Return the pixels where ``mask`` is true (``meta`` is kept)."""
        m = np.asarray(mask, dtype=bool)
        kw = {n: getattr(self, n)[m] for n in _ARRAYS}
        if self.m_per_photon is not None:
            kw["m_per_photon"] = self.m_per_photon[m]
        if self.d_per_photon is not None:
            kw["d_per_photon"] = self.d_per_photon[m]
        return replace(self, **kw)

    @classmethod
    def concat(cls, items: Sequence[ReducedPixels]) -> ReducedPixels:
        """Concatenate items, renumbering ``frame_id`` to keep ids unique."""
        if not items:
            raise ValueError("concat requires at least one item")
        rb = items[0].meta.get("R_B")
        if any(it.meta.get("R_B") != rb for it in items):
            raise ValueError("cannot concat ReducedPixels with different R_B")
        if any(it.meta.get("w") != items[0].meta.get("w") for it in items):
            raise ValueError("cannot concat ReducedPixels with different w")
        rn = items[0].meta.get("read_noise_photons", 0.0)
        if any(it.meta.get("read_noise_photons", 0.0) != rn for it in items):
            raise ValueError(
                "cannot concat ReducedPixels with different read_noise_photons"
            )
        ids, offset = [], 0
        for it in items:
            fid = np.asarray(it.frame_id, dtype=np.int64)
            if fid.size:
                ids.append(fid - fid.min() + offset)
                offset += int(fid.max() - fid.min()) + 1
            else:
                ids.append(fid)
        kw = {n: np.concatenate([getattr(it, n) for it in items]) for n in _ARRAYS}
        kw["frame_id"] = np.concatenate(ids)
        if any(it.m_per_photon is not None for it in items):
            kw["m_per_photon"] = np.concatenate(
                [
                    it.m_per_photon
                    if it.m_per_photon is not None
                    else np.full(it.M.shape, np.nan)
                    for it in items
                ]
            )
        if any(it.d_per_photon is not None for it in items):
            kw["d_per_photon"] = np.concatenate(
                [
                    it.d_per_photon
                    if it.d_per_photon is not None
                    else np.full(it.M.shape, np.nan)
                    for it in items
                ]
            )
        return cls(**kw, meta=dict(items[0].meta))


def cos_theta_out(qm: PixelQMap, k: float, W_rad: float) -> NDArray[np.float64]:
    """Cosine of the exit angle from the surface normal: ``cos W + Qz/k``."""
    return np.cos(W_rad) + np.asarray(qm.Qz, dtype=float) / k


def absorption_factor(mu_h, W_rad, cos_out) -> NDArray[np.float64]:
    """``exp[-(mu h/2)(1/cos W + 1/cos theta_out)]``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(-(mu_h / 2.0) * (1.0 / np.cos(W_rad) + 1.0 / cos_out))


def howto_prefactor(k: float, xs: CrossSection, W_rad: float) -> float:
    """``k^4 |dn_A|^2 / (4 pi^2 L cos W)``."""
    return float(k**4 * abs(xs.dn_A) ** 2 / (4 * np.pi**2 * xs.L * np.cos(W_rad)))


def reduce_frame(
    frame: Frame,
    geometry: Geometry,
    xs: CrossSection,
    *,
    scale: float,
    window_T: float = 1.0,
    R_B: float = 0.008,
    keep: float = 0.1,
    gamma_zero_halfwidth: float = 0.01,
    adu_per_photon: float,
    read_noise_photons: float = 0.0,
    bg_mode: BackgroundMode = "none",
    beamstop_mask=None,
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    frame_id: int = 0,
) -> ReducedPixels:
    """Reduce one frame to per-pixel ``M`` with variance, dropping masked pixels.

    ``scale`` is incident photons per second. The *contrast* used for the
    ``keep`` threshold is ``|Gamma(Qz)|^2 exp(-Qx^2 s^2)``, relative to its
    maximum over the otherwise valid pixels of this frame.

    ``omega_refinement`` is applied through :func:`prepare_frame`; the pixel Q,
    the incidence angle ``W`` (absorption, window transmission, prefactor) all
    come from the refined per-frame geometry.
    """
    if beamstop_mask is not None and np.shape(beamstop_mask) != frame.image.shape:
        raise ValueError("beamstop_mask shape must match the frame image shape")
    cleaned, qm, _, gf = prepare_frame(
        frame,
        geometry,
        bg_mode=bg_mode,
        approximation="correct",
        beamstop_mask=beamstop_mask,
        omega_refinement=omega_refinement,
        exposure_normalization=False,
        adu_per_photon=adu_per_photon,
    )
    E = frame.energy_ev
    om = frame.omega_deg
    t = frame.exposure_s
    k = 2 * np.pi * E / _HC_EV_NM
    W = np.radians(om + gf.omega0_deg)

    photons = np.asarray(cleaned, dtype=float) / adu_per_photon
    var_ph = np.maximum(photons, 0.0) + read_noise_photons**2
    inten = photons / (t * scale)
    var_inten = var_ph / (t * scale) ** 2

    Qx = np.asarray(qm.Qx, dtype=float)
    Qy = np.asarray(qm.Qy, dtype=float)
    Qz = np.asarray(qm.Qz, dtype=float)
    cos_out = cos_theta_out(qm, k, W)
    A = absorption_factor(xs.mu_h, W, cos_out)
    Tm = window_T ** (1.0 / np.cos(W))
    contrast = xs.gamma2(Qz) * np.exp(-(Qx**2) * xs.s**2)
    norm = howto_prefactor(k, xs, W) * np.asarray(qm.solid_angle_sr) * A * Tm

    with np.errstate(divide="ignore", invalid="ignore"):
        D = inten / norm
        var_D = var_inten / norm**2
        m_ph = 1.0 / (t * scale * norm * contrast)
        M = D / contrast
        var_M = var_D / contrast**2

    valid = (
        np.isfinite(inten) & np.isfinite(var_inten) & np.isfinite(Qx) & np.isfinite(Qy)
        & np.isfinite(Qz) & np.isfinite(norm) & (norm > 0)
        & np.isfinite(contrast) & (contrast > 0) & np.isfinite(M) & np.isfinite(var_M)
        & (cos_out > 0)
    )  # fmt: skip
    if beamstop_mask is not None:
        valid &= ~np.asarray(beamstop_mask, dtype=bool)

    G = 2 * np.pi / xs.L
    if valid.any():
        mmax = int(np.ceil(np.max(np.abs(Qx[valid])) / G)) + 1
        for m in range(-mmax, mmax + 1):
            if m:
                valid &= (Qx - m * G) ** 2 + Qy**2 >= R_B**2
        zeros = xs.zeros_qz(float(np.max(np.abs(Qz[valid]))) if valid.any() else 0.0)
        for z0 in np.asarray(zeros, dtype=float).ravel():
            valid &= np.abs(Qz - z0) >= gamma_zero_halfwidth
    if valid.any():
        valid &= contrast >= keep * np.max(contrast[valid])

    n = int(valid.sum())
    return ReducedPixels(
        Qx=Qx[valid],
        Qy=Qy[valid],
        Qz=Qz[valid],
        f=Qy[valid] / (2 * np.pi),
        D=D[valid],
        M=M[valid],
        var_M=var_M[valid],
        energy_ev=np.full(n, float(E)),
        omega_deg=np.full(n, float(om)),
        frame_id=np.full(n, frame_id, dtype=np.int64),
        m_per_photon=m_ph[valid],
        d_per_photon=(m_ph * contrast)[valid],
        meta={
            "read_noise_photons": read_noise_photons,
            "R_B": R_B,
            "w": xs.w,
            "scale": scale,
            "scale_source": "known",
        },
    )
