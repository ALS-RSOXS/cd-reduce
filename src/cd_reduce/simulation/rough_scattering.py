"""Ensemble scattering of rough gratings on the supercell's FFT mode grid.

Each realisation of a rough 3D grating (a supercell of ``n_lines`` periods over a
length ``ly``) is a periodic structure whose far field lives on the discrete modes
``qx = 2*pi*fftfreq(nx, dx)``, ``qy = 2*pi*fftfreq(ny, dy)``. Every mode is evaluated
at its own exact Ewald ``Qz``. Averaging over realisations gives the coherent
amplitude ``<A>`` (Bragg orders at ``qx = m * 2*pi/pitch, qy = 0``) and the diffuse
intensity ``<|A - <A>|^2>`` everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cd_reduce.simulation.multislice import (
    _longitudinal_delta_kz_2d,
    _multislice_3d_mode_grid,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cd_reduce.simulation.structures import RefractiveIndexMap3D

_ENGINES = ("born", "multislice")


@dataclass
class RoughEnsembleResult:
    """Ensemble-averaged scattering of rough-grating realisations on the mode grid.

    Arrays indexed ``(omega, qx, qy)`` cover the cropped modes
    ``|qx| <= qx_max``, ``|qy| <= qy_max`` in ascending (fftshifted) order.

    Attributes
    ----------
    qz : ndarray
        Exact Ewald ``Qz`` of each mode; NaN where the mode is evanescent.
    valid : ndarray of bool
        False for evanescent modes, ``k0^2 - (k0 sin(omega) + qx)^2 - qy^2 <= 0``.
    coherent : ndarray of complex
        ``<A>`` over realisations. The (0, 0) mode carries the transmitted beam:
        ``1 + <A>`` for Born, ``<c(0, 0)>`` for multislice. NaN on invalid modes.
    diffuse : ndarray
        Unbiased ``<|A - <A>|^2>``: ``sum_r |A_r - Abar|^2 / (n - 1)``, i.e.
        ``(<|A|^2> - |<A>|^2) * n / (n - 1)``. NaN when ``n = 1`` (one
        realisation has no diffuse estimate), on invalid modes and on the
        (0, 0) mode (excluded from the diffuse).
    diffuse_var : ndarray
        Variance of the diffuse estimate: ``var_r(|A_r - Abar|^2, ddof=1) / n``.
        NaN when ``n = 1``, on invalid modes and on the (0, 0) mode.
    flux_factor : ndarray
        ``cos(omega) / cos(theta_out)`` per mode, with
        ``cos(theta_out) = k_z,out / k0 = (k0 cos(omega) + Qz) / k0`` (engine sign
        convention: ``k_z,in = k0 cos(omega)``, ``k_z,out = k_z,in + Qz``). NaN on
        invalid modes, exactly 1 on the (0, 0) mode.
    diffuse_flux, diffuse_flux_var : ndarray
        ``diffuse * flux_factor`` and ``diffuse_var * flux_factor**2``.

    Notes
    -----
    Two intensity conventions. The amplitude ``A`` uses the tilted-path prefactor
    ``-i k0 / cos(omega)``, the convention of `simulate_born_2d` and the multislice
    phase screen, so ``|A|^2 = k0^2 |F|^2 / cos^2(omega)`` (``coherent``,
    ``diffuse``, `order_efficiency`). The exact first-Born transmitted field of a
    mode is ``-i k0 F / cos(theta_out)``; the power it carries through a
    ``z = const`` plane, normalised to the incident power, is
    ``|field|^2 cos(theta_out) / cos(omega) = k0^2 |F|^2 / (cos(omega)
    cos(theta_out))`` = ``|A|^2 * flux_factor``. That flux form is what a detector
    counts and is the HOWTO model's form (``diffuse_flux``,
    `order_flux_efficiency`).

    The engine's ``omega`` sense is opposite to `cd_reduce.reduction.geometry`: a
    frame taken at detector-geometry angle ``W`` corresponds to engine
    ``omega = -W``.
    """

    engine: str
    omega_deg: np.ndarray
    energy_ev: float
    pitch_nm: float
    n_lines: int
    ly_nm: float
    n_realisations: int
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    valid: np.ndarray
    coherent: np.ndarray
    diffuse: np.ndarray
    diffuse_var: np.ndarray
    flux_factor: np.ndarray
    diffuse_flux: np.ndarray
    diffuse_flux_var: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def _order_index(self, m: int) -> tuple[int, int]:
        nx = int(self.metadata["nx"])
        target_x = (int(m) * int(self.metadata["n_periods_x"])) % nx
        hits_x = np.flatnonzero(self.metadata["ix_full"] == target_x)
        hits_y = np.flatnonzero(self.metadata["iy_full"] == 0)
        if hits_x.size != 1 or hits_y.size != 1:
            msg = f"Order {m} lies outside the stored (cropped) mode grid."
            raise KeyError(msg)
        return int(hits_x[0]), int(hits_y[0])

    def order_qz(self, m: int) -> np.ndarray:
        """Ewald ``Qz`` of diffraction order ``m`` (mode ``(m * n_periods_x, 0)``)."""
        ix, iy = self._order_index(m)
        return self.qz[:, ix, iy].copy()

    def order_efficiency(self, m: int) -> np.ndarray:
        """Coherent efficiency ``|<A>|^2`` of order ``m`` versus ``omega``."""
        ix, iy = self._order_index(m)
        return np.abs(self.coherent[:, ix, iy]) ** 2

    def order_flux_efficiency(self, m: int) -> np.ndarray:
        """Flux efficiency ``order_efficiency(m) * flux_factor`` of order ``m``."""
        ix, iy = self._order_index(m)
        return self.order_efficiency(m) * self.flux_factor[:, ix, iy]


def _crop_indices(n: int, d: float, q_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Full-FFT indices (ascending in q) and q values with ``|q| <= q_max``."""
    q_full = 2.0 * np.pi * np.fft.fftfreq(n, d=d)
    idx = np.fft.fftshift(np.arange(n))
    keep = np.abs(q_full[idx]) <= q_max * (1.0 + 1e-12)
    return idx[keep], q_full[idx[keep]]


def _born_mode_amplitudes(
    structure: RefractiveIndexMap3D,
    omega_arr: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    """Born amplitude ``A(omega, qx, qy)`` on the cropped modes.

    ``F_iz = fft2(dn[:, :, iz]) * exp(-i(qx x0 + qy y0)) * dx dy / (Lx Ly)`` and
    ``A = (-i k0 / cos(omega)) * sum_iz F_iz exp(-i Qz (z_iz - zbar)) dz``, the
    convention of `simulate_born_2d`. ``qz`` must be finite (evanescent modes are
    masked by the caller).
    """
    dn = structure.delta_n
    nx, ny = structure.nx, structure.ny
    dx, dy, dz = structure.dx_nm, structure.dy_nm, structure.dz_nm
    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)[ix]
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)[iy]
    x0, y0 = float(structure.x_nm[0]), float(structure.y_nm[0])
    phase_origin = np.exp(-1j * (qx[:, None] * x0 + qy[None, :] * y0))
    f_xyz = np.fft.fft2(dn, axes=(0, 1))[np.ix_(ix, iy)]  # (nqx, nqy, nz)
    f_xyz *= (phase_origin * (dx * dy / (structure.lx_nm * structure.ly_nm)))[
        :, :, None
    ]
    z_c = structure.z_nm - float(np.mean(structure.z_nm))
    k0 = structure.k0_nm_inv
    cos_om = np.maximum(np.abs(np.cos(np.deg2rad(omega_arr))), 1e-3)
    out = np.empty(qz.shape, dtype=np.complex128)
    for i_om in range(omega_arr.size):
        phase_z = np.exp(-1j * qz[i_om][:, :, None] * z_c[None, None, :])
        out[i_om] = (-1j * k0 / cos_om[i_om]) * np.sum(f_xyz * phase_z, axis=2) * dz
    return out


class _Moments:
    """Shifted per-mode moments of ``A_r`` (see `simulate_rough_ensemble` Notes)."""

    def __init__(self) -> None:
        self.n = 0

    def add(self, amp: np.ndarray) -> None:
        if self.n == 0:
            self.shift = amp.copy()
            self.s1 = np.zeros_like(amp)
            self.t2 = np.zeros_like(amp)
            self.u3 = np.zeros_like(amp)
            self.s2 = np.zeros(amp.shape)
            self.v4 = np.zeros(amp.shape)
        b = amp - self.shift
        b2 = np.abs(b) ** 2
        self.s1 += b
        self.s2 += b2
        self.t2 += b * b
        self.u3 += b2 * b
        self.v4 += b2 * b2
        self.n += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(coherent, diffuse, diffuse_var)`` (unbiased, ``ddof = 1``)."""
        n, s2 = self.n, self.s2
        c = self.s1 / n
        c2 = np.abs(c) ** 2
        sum_d = s2 - n * c2
        sum_d2 = (
            self.v4
            + 2.0 * (s2 * c2 + np.real(self.t2 * np.conj(c) ** 2))
            + n * c2**2
            - 4.0 * np.real(self.u3 * np.conj(c))
            + 2.0 * s2 * c2
            - 4.0 * n * c2**2
        )
        if n > 1:
            diffuse = np.maximum(sum_d, 0.0) / (n - 1)
            diffuse_var = np.maximum(sum_d2 - sum_d**2 / n, 0.0) / (n - 1) / n
        else:
            diffuse = np.full(c.shape, np.nan)
            diffuse_var = np.full(c.shape, np.nan)
        return self.shift + c, diffuse, diffuse_var


def _multislice_mode_amplitudes(
    structure: RefractiveIndexMap3D,
    omega_arr: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
) -> np.ndarray:
    """Multislice exit-wave modes ``c(omega, qx, qy)`` on the cropped modes."""
    qx = 2.0 * np.pi * np.fft.fftfreq(structure.nx, d=structure.dx_nm)
    qy = 2.0 * np.pi * np.fft.fftfreq(structure.ny, d=structure.dy_nm)
    qx_2d, qy_2d = np.meshgrid(qx, qy, indexing="ij")
    sub = np.ix_(ix, iy)
    return np.stack(
        [
            _multislice_3d_mode_grid(structure, float(om), qx_2d, qy_2d)[0][sub]
            for om in omega_arr
        ]
    )


def _scaled_contrast(structure: RefractiveIndexMap3D, eps: float):
    """Copy of ``structure`` with ``delta_n`` scaled by ``eps``."""
    return replace(
        structure,
        delta=structure.delta_ref + eps * (structure.delta - structure.delta_ref),
        beta=structure.beta_ref + eps * (structure.beta - structure.beta_ref),
    )


def simulate_rough_ensemble(
    make_structure: Callable[[np.random.Generator], RefractiveIndexMap3D],
    omega_deg: Sequence[float] | np.ndarray,
    *,
    engine: Literal["born", "multislice"] = "born",
    n_realisations: int = 8,
    seed: int = 0,
    qx_max: float = 0.4,
    qy_max: float = 0.21,
    born_reference: Literal[None, "epsilon"] = None,
    epsilon: float = 1e-3,
) -> RoughEnsembleResult:
    """Ensemble-average scattering of rough-grating realisations on the mode grid.

    Parameters
    ----------
    make_structure
        Called once per realisation with an independent child generator spawned
        from ``np.random.SeedSequence(seed)``; returns a `RefractiveIndexMap3D`.
        All realisations must share one grid.
    omega_deg
        Sample tilts in degrees.
    engine
        ``"born"`` (kinematic, this module) or ``"multislice"``.
    n_realisations
        Number of realisations averaged. ``1`` is allowed for coherent-only
        work (e.g. `order_efficiency` of a deterministic structure); ``diffuse``,
        ``diffuse_var``, their flux forms and ``metadata["born_eps"]`` are then
        NaN. Use ``n >= 2`` for any diffuse estimate.
    qx_max, qy_max
        Modes with ``|qx| <= qx_max`` and ``|qy| <= qy_max`` (nm^-1) are stored.
    born_reference, epsilon
        Weak-scattering reference for the multislice engine: with ``"epsilon"``
        each realisation is also propagated with ``epsilon * delta_n`` and
        ``metadata["born_eps"]`` holds that diffuse divided by ``epsilon**2``.
        The Born engine is already kinematic and ignores them.

    Notes
    -----
    Moments of ``B_r = A_r - A_1`` (shifted by the first realisation to limit
    cancellation) are accumulated per mode: ``sum B``, ``sum |B|^2``, ``sum B^2``,
    ``sum |B|^2 B`` and ``sum |B|^4``. From these ``sum_r d_r`` and ``sum_r d_r^2``
    with ``d_r = |A_r - Abar|^2`` follow exactly, so realisations are not stored.
    """
    if engine not in _ENGINES:
        msg = f"engine must be one of {_ENGINES}, got {engine!r}"
        raise ValueError(msg)
    if born_reference not in (None, "epsilon"):
        msg = f"born_reference must be None or 'epsilon', got {born_reference!r}"
        raise ValueError(msg)
    n = int(n_realisations)
    if n < 1:
        raise ValueError("n_realisations must be >= 1")

    omega_arr = np.atleast_1d(np.asarray(omega_deg, dtype=np.float64))
    children = np.random.SeedSequence(int(seed)).spawn(n)

    eps_ref = engine == "multislice" and born_reference == "epsilon"
    eps = float(epsilon)
    if eps_ref and not eps > 0:
        raise ValueError("epsilon must be > 0")
    first: RefractiveIndexMap3D | None = None
    mom, mom_eps = _Moments(), _Moments()
    for child in children:
        st = make_structure(np.random.default_rng(child))
        if first is None:
            first = st
            k0 = st.k0_nm_inv
            ix, qx = _crop_indices(st.nx, st.dx_nm, float(qx_max))
            iy, qy = _crop_indices(st.ny, st.dy_nm, float(qy_max))
            qx_3d = np.broadcast_to(
                qx[None, :, None], (omega_arr.size, qx.size, qy.size)
            )
            qy_3d = np.broadcast_to(qy[None, None, :], qx_3d.shape)
            om_rad = np.deg2rad(omega_arr)[:, None, None]
            arg = k0**2 - (k0 * np.sin(om_rad) + qx_3d) ** 2 - qy_3d**2
            valid = arg > 0.0
            qz = np.stack(
                [
                    _longitudinal_delta_kz_2d(qx_3d[i], qy_3d[i], float(om), k0)
                    for i, om in enumerate(omega_arr)
                ]
            )
            qz = np.where(valid, qz, np.nan)
            qz_eval = np.where(valid, qz, 0.0)
        elif (
            st.delta.shape != first.delta.shape
            or not np.allclose(st.x_nm, first.x_nm)
            or not np.allclose(st.y_nm, first.y_nm)
            or not np.allclose(st.z_nm, first.z_nm)
            or st.energy_ev != first.energy_ev
        ):
            raise ValueError("all realisations must share one grid and energy")

        if engine == "born":
            mom.add(_born_mode_amplitudes(st, omega_arr, ix, iy, qz_eval))
        else:
            mom.add(_multislice_mode_amplitudes(st, omega_arr, ix, iy))
            if eps_ref:
                st_eps = _scaled_contrast(st, eps)
                mom_eps.add(_multislice_mode_amplitudes(st_eps, omega_arr, ix, iy))

    assert first is not None
    coherent, diffuse, diffuse_var = mom.result()
    born_eps = mom_eps.result()[1] / eps**2 if eps_ref else None

    i0 = np.flatnonzero(ix == 0)
    j0 = np.flatnonzero(iy == 0)
    cos_in = np.cos(np.deg2rad(omega_arr))[:, None, None]
    cos_out = cos_in + qz_eval / k0
    flux_factor = np.where(valid, cos_in / np.where(valid, cos_out, 1.0), np.nan)
    if i0.size and j0.size:
        flux_factor[:, i0[0], j0[0]] = np.where(valid[:, i0[0], j0[0]], 1.0, np.nan)
        if engine == "born":
            coherent[:, i0[0], j0[0]] += 1.0
        diffuse[:, i0[0], j0[0]] = np.nan
        if born_eps is not None:
            born_eps[:, i0[0], j0[0]] = np.nan
        diffuse_var[:, i0[0], j0[0]] = np.nan
    coherent = np.where(valid, coherent, np.nan + 1j * np.nan)
    diffuse = np.where(valid, diffuse, np.nan)
    diffuse_var = np.where(valid, diffuse_var, np.nan)
    extra = {}
    if born_eps is not None:
        extra["born_eps"] = np.where(valid, born_eps, np.nan)

    pitch = first.effective_pitch_nm
    n_per = max(round(first.lx_nm / pitch), 1)
    return RoughEnsembleResult(
        engine=engine,
        omega_deg=omega_arr,
        energy_ev=float(first.energy_ev),
        pitch_nm=float(pitch),
        n_lines=int(first.metadata.get("n_lines", n_per)),
        ly_nm=float(first.ly_nm),
        n_realisations=n,
        qx=qx,
        qy=qy,
        qz=qz,
        valid=valid,
        coherent=coherent,
        diffuse=diffuse,
        diffuse_var=diffuse_var,
        flux_factor=flux_factor,
        diffuse_flux=diffuse * flux_factor,
        diffuse_flux_var=diffuse_var * flux_factor**2,
        metadata={
            "nx": first.nx,
            "ny": first.ny,
            "n_periods_x": n_per,
            "ix_full": ix,
            "iy_full": iy,
            "seed": int(seed),
            "qx_max": float(qx_max),
            "qy_max": float(qy_max),
            "born_reference": born_reference,
            "epsilon": float(epsilon),
            **extra,
        },
    )
