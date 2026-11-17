"""Line cross sections for edge-contrast roughness scattering.

A line of width ``w`` and edge blur ``s`` (Gaussian) sits at ``x = 0`` in a period
``L``. Its left and right edges are displaced by ``u_L`` and ``u_R``. The exposed
fraction is ``p(x) = 1 - [Phi((x + w/2 - u_L)/s) - Phi((x - w/2 - u_R)/s)]``, which is
0 inside the line and 1 in the space. ``g(x, Qz)`` is the depth integral of
``delta_n_line(x, 0, 0, z) / dn_A * exp(-i Qz z)`` and ``Gamma`` is the difference
between the line (``p = 0``) and space (``p = 1``) columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

if TYPE_CHECKING:
    from cd_reduce.simulation.structures import RefractiveIndexMap2D

_HC_EV_NM = 1239.841984
_QZ_MIN = 1e-9
_N_X = 2048

CArr = NDArray[np.complex128]
FArr = NDArray[np.float64]


@runtime_checkable
class CrossSection(Protocol):
    """Interface the reduction uses for a line cross section."""

    L: float
    w: float
    s: float
    dn_A: complex
    energy_ev: float
    top_facing: Literal["upstream", "downstream"]

    @property
    def height(self) -> float: ...
    def gamma(self, Qz: ArrayLike) -> CArr: ...
    def gamma2(self, Qz: ArrayLike) -> FArr: ...
    def g(self, x: ArrayLike, Qz: ArrayLike) -> CArr: ...
    def f_m(self, m: int, Qz: ArrayLike) -> CArr: ...
    @property
    def mu_h(self) -> float: ...
    def delta_n_line(
        self, x_rel: ArrayLike, u_left: ArrayLike, u_right: ArrayLike, z: ArrayLike
    ) -> CArr: ...
    def zeros_qz(self, qz_max: float) -> FArr: ...


def _safe_qz(Qz: ArrayLike) -> FArr:
    q = np.asarray(Qz, dtype=np.float64)
    return np.where(np.abs(q) < _QZ_MIN, _QZ_MIN, q)


def _exposed(x: ArrayLike, w: float, s: float, uL: ArrayLike, uR: ArrayLike) -> FArr:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 - (
        ndtr((x + w / 2 - np.asarray(uL)) / s) - ndtr((x - w / 2 - np.asarray(uR)) / s)
    )


class _Base:
    """Shared Qz orientation, f_m, mu_h; subclasses give ``_g``/``_gamma``/Delta n."""

    L: float
    w: float
    s: float
    dn_A: complex
    energy_ev: float
    top_facing: str

    def _q(self, Qz: ArrayLike) -> FArr:
        q = _safe_qz(Qz)
        return -q if self.top_facing == "downstream" else q

    def gamma(self, Qz: ArrayLike) -> CArr:
        return self._gamma(self._q(Qz))

    def gamma2(self, Qz: ArrayLike) -> FArr:
        return np.abs(self.gamma(Qz)) ** 2

    def g(self, x: ArrayLike, Qz: ArrayLike) -> CArr:
        return self._g(np.asarray(x, dtype=np.float64), self._q(Qz))

    def f_m(self, m: int, Qz: ArrayLike) -> CArr:
        """Fourier coefficient ``(1/L) int g exp(-i m G x) dx`` over one period."""
        x = (np.arange(_N_X) + 0.5) * (self.L / _N_X) - self.L / 2
        q = np.asarray(Qz, dtype=np.float64)
        kern = np.exp(-1j * m * (2 * np.pi / self.L) * x)
        gx = self.g(x, q[..., None]) if q.ndim else self.g(x, q)
        return np.mean(gx * kern, axis=-1)

    @property
    def mu_h(self) -> float:
        k = 2 * np.pi * self.energy_ev / _HC_EV_NM
        f0 = self.f_m(0, 0.0)
        return float(2 * k * np.imag(self.dn_A * f0))

    def _gamma(self, q: FArr) -> CArr:  # pragma: no cover
        raise NotImplementedError

    def _g(self, x: FArr, q: FArr) -> CArr:  # pragma: no cover
        raise NotImplementedError

    def zeros_qz(self, qz_max: float) -> FArr:  # noqa: ARG002
        return np.empty(0)


@dataclass(frozen=True)
class LatentFilm(_Base):
    """Latent-image film: contrast ``r`` decaying with depth, plus shrinkage."""

    L: float
    w: float
    s: float
    h: float
    dh: float
    lam: float
    r: complex
    dn_A: complex
    top_facing: Literal["upstream", "downstream"] = "upstream"
    energy_ev: float = field(kw_only=True)

    @property
    def height(self) -> float:
        return self.h

    def _gamma(self, q: FArr) -> CArr:
        h, dh, lam = self.h, self.dh, self.lam
        b = 1.0 / lam - 1j * q
        geo = np.exp(-1j * q * h) * np.expm1(1j * q * dh) / (1j * q)
        return geo - (self.r - 1) * np.exp(-h / lam) * np.expm1(b * (h - dh)) / b

    def _g(self, x: FArr, q: FArr) -> CArr:
        p = _exposed(x, self.w, self.s, 0.0, 0.0)
        T = self.h - self.dh * p
        b = 1.0 / self.lam - 1j * q
        geo = -np.expm1(-1j * q * T) / (1j * q)
        return geo + (self.r - 1) * p * np.exp(-self.h / self.lam) * np.expm1(b * T) / b

    def delta_n_line(
        self, x_rel: ArrayLike, u_left: ArrayLike, u_right: ArrayLike, z: ArrayLike
    ) -> CArr:
        p = _exposed(x_rel, self.w, self.s, u_left, u_right)
        z = np.asarray(z, dtype=np.float64)
        dn = self.dn_A * (1 + (self.r - 1) * p * np.exp(-(self.h - z) / self.lam))
        inside = (z >= 0) & (z < self.h - self.dh * p)
        return np.where(inside, dn, 0.0 + 0.0j)


@dataclass(frozen=True)
class DevelopedLine(_Base):
    """Developed line: uniform material of height ``h_d`` with blurred edges."""

    L: float
    w: float
    s: float
    h_d: float
    dn_A: complex
    top_facing: Literal["upstream", "downstream"] = "upstream"
    energy_ev: float = field(kw_only=True)

    @property
    def height(self) -> float:
        return self.h_d

    def _gamma(self, q: FArr) -> CArr:
        return -np.expm1(-1j * q * self.h_d) / (1j * q)

    def _g(self, x: FArr, q: FArr) -> CArr:
        return (1.0 - _exposed(x, self.w, self.s, 0.0, 0.0)) * self._gamma(q)

    def delta_n_line(
        self, x_rel: ArrayLike, u_left: ArrayLike, u_right: ArrayLike, z: ArrayLike
    ) -> CArr:
        p = _exposed(x_rel, self.w, self.s, u_left, u_right)
        z = np.asarray(z, dtype=np.float64)
        inside = (z >= 0) & (z < self.h_d)
        return np.where(inside, self.dn_A * (1 - p), 0.0 + 0.0j)

    def zeros_qz(self, qz_max: float) -> FArr:
        n = int(np.floor(qz_max * self.h_d / (2 * np.pi)))
        pos = 2 * np.pi * np.arange(1, n + 1) / self.h_d
        return np.concatenate([-pos[::-1], pos])


def to_index_map_2d(
    xs: CrossSection, *, nx: int, nz: int, energy_ev: float
) -> RefractiveIndexMap2D:
    """Sample ``delta_n_line(x, 0, 0, z)`` on one period as a 2D index map.

    ``delta = -Re(dn)``, ``beta = Im(dn)``, vacuum reference. Cells are centred on
    ``x`` in ``[-L/2, L/2)`` and ``z`` in ``[0, height)``.
    """
    from cd_reduce.simulation.structures import RefractiveIndexMap2D

    height = float(xs.height)
    x = (np.arange(nx) + 0.5) * (xs.L / nx) - xs.L / 2
    z = (np.arange(nz) + 0.5) * (height / nz)
    dn = xs.delta_n_line(x[:, None], 0.0, 0.0, z[None, :])
    return RefractiveIndexMap2D(
        x_nm=x,
        z_nm=z,
        delta=-dn.real,
        beta=dn.imag,
        energy_ev=energy_ev,
        pitch_nm=float(xs.L),
        delta_ref=0.0,
        beta_ref=0.0,
    )
