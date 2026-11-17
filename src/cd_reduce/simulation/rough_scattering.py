"""Ensemble scattering of rough gratings on the supercell's FFT mode grid.

Each realisation of a rough 3D grating (a supercell of ``n_lines`` periods over a
length ``ly``) is a periodic structure whose far field lives on the discrete modes
``qx = 2*pi*fftfreq(nx, dx)``, ``qy = 2*pi*fftfreq(ny, dy)``. Every mode is evaluated
at its own exact Ewald ``Qz``. Averaging over realisations gives the coherent
amplitude ``<A>`` (Bragg orders at ``qx = m * 2*pi/pitch, qy = 0``) and the diffuse
intensity ``<|A - <A>|^2>`` everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cd_reduce.simulation.multislice import _longitudinal_delta_kz_2d

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
        ``<A>`` over realisations. The (0, 0) mode carries the unit beam, ``1 + <A>``.
        NaN on invalid modes.
    diffuse : ndarray
        Unbiased ``<|A - <A>|^2>``: ``sum_r |A_r - Abar|^2 / (n - 1)``, i.e.
        ``(<|A|^2> - |<A>|^2) * n / (n - 1)``. Zero when ``n = 1``. NaN on invalid
        modes and on the (0, 0) mode (excluded from the diffuse).
    diffuse_var : ndarray
        Variance of the diffuse estimate: ``var_r(|A_r - Abar|^2, ddof=1) / n``.
        NaN when ``n = 1``, on invalid modes and on the (0, 0) mode.
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
        Number of realisations averaged.
    qx_max, qy_max
        Modes with ``|qx| <= qx_max`` and ``|qy| <= qy_max`` (nm^-1) are stored.
    born_reference, epsilon
        Weak-scattering reference for the multislice engine; the Born engine is
        already kinematic and ignores them.

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
    if engine == "multislice":
        raise NotImplementedError("the multislice engine is not implemented yet")
    n = int(n_realisations)
    if n < 1:
        raise ValueError("n_realisations must be >= 1")

    omega_arr = np.atleast_1d(np.asarray(omega_deg, dtype=np.float64))
    children = np.random.SeedSequence(int(seed)).spawn(n)

    first: RefractiveIndexMap3D | None = None
    shift = s1 = s2 = t2 = u3 = v4 = None
    for r, child in enumerate(children):
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

        amp = _born_mode_amplitudes(st, omega_arr, ix, iy, qz_eval)
        if r == 0:
            shift = amp.copy()
            zeros_c = np.zeros_like(amp)
            s1, t2, u3 = zeros_c, zeros_c.copy(), zeros_c.copy()
            s2, v4 = np.zeros(amp.shape), np.zeros(amp.shape)
        b = amp - shift
        b2 = np.abs(b) ** 2
        s1 += b
        s2 += b2
        t2 += b * b
        u3 += b2 * b
        v4 += b2 * b2

    assert first is not None
    c = s1 / n
    c2 = np.abs(c) ** 2
    sum_d = s2 - n * c2
    sum_d2 = (
        v4
        + 2.0 * (s2 * c2 + np.real(t2 * np.conj(c) ** 2))
        + n * c2**2
        - 4.0 * np.real(u3 * np.conj(c))
        + 2.0 * s2 * c2
        - 4.0 * n * c2**2
    )
    coherent = shift + c
    if n > 1:
        diffuse = np.maximum(sum_d, 0.0) / (n - 1)
        diffuse_var = np.maximum(sum_d2 - sum_d**2 / n, 0.0) / (n - 1) / n
    else:
        diffuse = np.zeros(c.shape)
        diffuse_var = np.full(c.shape, np.nan)

    i0 = np.flatnonzero(ix == 0)
    j0 = np.flatnonzero(iy == 0)
    if i0.size and j0.size:
        coherent[:, i0[0], j0[0]] += 1.0
        diffuse[:, i0[0], j0[0]] = np.nan
        diffuse_var[:, i0[0], j0[0]] = np.nan
    coherent = np.where(valid, coherent, np.nan + 1j * np.nan)
    diffuse = np.where(valid, diffuse, np.nan)
    diffuse_var = np.where(valid, diffuse_var, np.nan)

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
        },
    )
