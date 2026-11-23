"""Tapered (sloped-sidewall) developed line: violates the separable-Gamma assumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from cd_reduce.reduction.roughness.cross_section import (
    CArr,
    FArr,
    _Base,
    _exposed,
)

_N_Z = 64


@dataclass(frozen=True)
class TaperedLine(_Base):
    """Developed line with width ``w(z)`` linear from ``w_bottom`` (z=0) to ``w_top``.

    Edges are displaced conformally (the same ``u`` at every depth). ``w`` is the
    mean width and ``gamma`` the separable depth-weighted ``Gamma_dev(h)``.
    """

    L: float
    w_bottom: float
    w_top: float
    s: float
    h: float
    dn_A: complex
    top_facing: Literal["upstream", "downstream"] = "upstream"
    energy_ev: float = field(kw_only=True)

    @property
    def w(self) -> float:  # type: ignore[override]
        return 0.5 * (self.w_bottom + self.w_top)

    @property
    def height(self) -> float:
        return self.h

    def _gamma(self, q: FArr) -> CArr:
        return -np.expm1(-1j * q * self.h) / (1j * q)

    def _g(self, x: FArr, q: FArr) -> CArr:
        dz = self.h / _N_Z
        z = (np.arange(_N_Z) + 0.5) * dz
        wz = self.w_bottom + (self.w_top - self.w_bottom) * z / self.h
        xe = np.asarray(x)[..., None]
        qe = np.asarray(q)[..., None]
        p = _exposed(xe, wz, self.s, 0.0, 0.0)
        return np.sum((1.0 - p) * np.exp(-1j * qe * z) * dz, axis=-1)

    def delta_n_line(
        self, x_rel: ArrayLike, u_left: ArrayLike, u_right: ArrayLike, z: ArrayLike
    ) -> CArr:
        z = np.asarray(z, dtype=np.float64)
        wz = self.w_bottom + (self.w_top - self.w_bottom) * z / self.h
        p = _exposed(x_rel, wz, self.s, u_left, u_right)
        inside = (z >= 0) & (z < self.h)
        return np.where(inside, self.dn_A * (1 - p), 0.0 + 0.0j)

    def surface_height(
        self, x_rel: ArrayLike, u_left: ArrayLike, u_right: ArrayLike
    ) -> FArr:
        shape = np.broadcast(
            np.asarray(x_rel), np.asarray(u_left), np.asarray(u_right)
        ).shape
        return np.full(shape, float(self.h))

    def zeros_qz(self, qz_max: float) -> FArr:
        n = int(np.floor(qz_max * self.h / (2 * np.pi)))
        pos = 2 * np.pi * np.arange(1, n + 1) / self.h
        return np.concatenate([-pos[::-1], pos])


def tapered_from_sidewall(
    L: float, w_mean: float, s: float, h: float, dn_A: complex, *, energy_ev: float,
    sidewall_deg: float = 85.0,
) -> TaperedLine:  # fmt: skip
    """Tapered line of mean width ``w_mean``: ``w_top = w_bottom - 2 h / tan(sw)``."""
    d = h / np.tan(np.radians(sidewall_deg))
    return TaperedLine(L, w_mean + d, w_mean - d, s, h, dn_A, energy_ev=energy_ev)
