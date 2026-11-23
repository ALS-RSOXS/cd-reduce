"""2D and 3D Multislice (Split-Step Fourier) simulators for CD-RSoXS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import ApproximationMode, Geometry, compute_pixel_q
from cd_reduce.reduction.mapping import QMap2D, Trace1D
from cd_reduce.simulation.born import _edges_from_centers
from cd_reduce.simulation.structures import RefractiveIndexMap2D, RefractiveIndexMap3D

if TYPE_CHECKING:
    from collections.abc import Sequence


def _longitudinal_delta_kz_1d(
    qx: np.ndarray,
    omega_deg: float,
    k0: float,
    mode: ApproximationMode = "correct",
) -> np.ndarray:
    """Compute longitudinal propagator wavenumber `Delta k_z(Qx, omega)` in 1D."""
    om_rad = np.deg2rad(float(omega_deg))
    if mode in ("small_angle", "sax"):
        return -qx * np.tan(om_rad)
    if mode in ("angle_linear", "lin"):
        return -qx * np.sin(om_rad)
    kx_in = k0 * np.sin(om_rad)
    kz_in = k0 * np.cos(om_rad)
    arg = np.maximum(k0 * k0 - (kx_in + qx) ** 2, 0.0)
    return np.sqrt(arg) - kz_in


def _longitudinal_delta_kz_2d(
    qx_2d: np.ndarray,
    qy_2d: np.ndarray,
    omega_deg: float,
    k0: float,
    mode: ApproximationMode = "correct",
) -> np.ndarray:
    """Compute longitudinal propagator wavenumber `Delta k_z(Qx, Qy, omega)` in 2D."""
    om_rad = np.deg2rad(float(omega_deg))
    if mode in ("small_angle", "sax"):
        return -qx_2d * np.tan(om_rad) - 0.5 * (qy_2d**2) / max(
            k0 * np.cos(om_rad), 1e-6
        )
    if mode in ("angle_linear", "lin"):
        return -qx_2d * np.sin(om_rad)
    kx_in = k0 * np.sin(om_rad)
    kz_in = k0 * np.cos(om_rad)
    arg = np.maximum(k0 * k0 - (kx_in + qx_2d) ** 2 - qy_2d**2, 0.0)
    return np.sqrt(arg) - kz_in


@dataclass
class MultisliceSimulationResult2D:
    """Result of a 2D Multislice beam-propagation simulation from a
    `RefractiveIndexMap2D`.
    """

    structure: RefractiveIndexMap2D
    omega_deg: np.ndarray
    orders: np.ndarray
    order_qx: np.ndarray
    order_qz: dict[int, np.ndarray]
    order_amplitudes: dict[int, np.ndarray]
    order_efficiencies: dict[int, np.ndarray]
    exit_waves: np.ndarray
    internal_wavefield: np.ndarray
    field_omega_deg: float
    qmap_qx_qz: QMap2D
    mode: ApproximationMode = "correct"
    metadata: dict[str, Any] = field(default_factory=dict)

    def bragg_rod(self, order: int, x_axis: str = "Qz") -> Trace1D:
        """Return the simulated 1D Bragg rod `Trace1D` for diffraction order `m`."""
        m = int(order)
        if m not in self.order_efficiencies:
            msg = f"Order {m} not in simulated orders {self.orders.tolist()}."
            raise KeyError(msg)
        intensity = self.order_efficiencies[m].copy()
        if x_axis == "omega":
            x = self.omega_deg.copy()
            axis_name = "omega"
        elif x_axis == "Qz":
            x = self.order_qz[m].copy()
            axis_name = "Qz"
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            intensity = intensity[sort_idx]
        else:
            msg = f"Unsupported x_axis={x_axis!r}; choose 'Qz' or 'omega'."
            raise ValueError(msg)
        return Trace1D(
            coords=x,
            edges=_edges_from_centers(x),
            intensity=intensity,
            sigma=np.zeros_like(intensity),
            counts=np.ones_like(intensity),
            domain=axis_name,
            metadata={"order": m, "simulator": "multislice_2d", "mode": self.mode},
        )

    def to_detector_frame(
        self,
        geometry: Geometry,
        omega_deg: float = 0.0,
        *,
        shape: tuple[int, int] = (256, 256),
        spot_sigma_q: float = 0.008,
        flux_scale: float = 1e6,
        background_adu: float = 0.0,
        poisson_noise: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> Frame:
        """Project the 2D Multislice diffraction orders onto a synthetic 2D CCD
        `Frame`.
        """
        ny_det, nx_det = int(shape[0]), int(shape[1])
        yy, xx = np.meshgrid(
            np.arange(ny_det, dtype=np.float64),
            np.arange(nx_det, dtype=np.float64),
            indexing="ij",
        )
        pq = compute_pixel_q(
            geometry,
            xx,
            yy,
            self.structure.energy_ev,
            float(omega_deg),
            approximation=self.mode,
        )
        image = np.full((ny_det, nx_det), float(background_adu), dtype=np.float64)
        sigma_q = max(float(spot_sigma_q), 1e-6)
        qy_env = np.exp(-0.5 * (pq.Qy / sigma_q) ** 2)

        for m, qx_m in zip(self.orders, self.order_qx, strict=True):
            if int(m) == 0:
                continue
            eff_m = float(
                np.interp(
                    float(omega_deg), self.omega_deg, self.order_efficiencies[int(m)]
                )
            )
            qx_env = np.exp(-0.5 * ((pq.Qx - float(qx_m)) / sigma_q) ** 2)
            image += float(flux_scale) * eff_m * qx_env * qy_env

        if poisson_noise:
            gen = (
                rng
                if isinstance(rng, np.random.Generator)
                else np.random.default_rng(rng)
            )
            image = gen.poisson(np.clip(image, 0.0, None)).astype(np.float64)

        return Frame.from_array(
            image=image,
            omega_deg=float(omega_deg),
            energy_ev=self.structure.energy_ev,
            exposure_s=1.0,
        )


def _propagate_multislice_2d_single_omega(
    structure: RefractiveIndexMap2D,
    omega_deg: float,
    qx_fft: np.ndarray,
    mode: ApproximationMode = "correct",
    record_internal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run Strang-split 2D multislice propagation at a single tilt angle `omega_deg`."""
    k0 = structure.k0_nm_inv
    dz = structure.dz_nm
    cos_om = max(abs(float(np.cos(np.deg2rad(omega_deg)))), 1e-3)
    delta_kz = _longitudinal_delta_kz_1d(qx_fft, omega_deg=omega_deg, k0=k0, mode=mode)
    half_prop = np.exp(1j * delta_kz * (0.5 * dz))

    dn = structure.delta_n  # (nx, nz)
    psi = np.ones(structure.nx, dtype=np.complex128)
    internal = (
        np.empty((structure.nx, structure.nz), dtype=np.complex128)
        if record_internal
        else None
    )

    for iz in range(structure.nz):
        psi_k = np.fft.fft(psi) * half_prop
        psi = np.fft.ifft(psi_k)
        psi = psi * np.exp(-1j * (k0 / cos_om) * dn[:, iz] * dz)
        psi_k = np.fft.fft(psi) * half_prop
        psi = np.fft.ifft(psi_k)
        if internal is not None:
            internal[:, iz] = psi

    return psi, delta_kz, internal


def simulate_multislice_2d(
    structure: RefractiveIndexMap2D,
    *,
    omega_deg: Sequence[float] | np.ndarray | None = None,
    max_order: int = 8,
    field_omega_deg: float = 0.0,
    qx_range: tuple[float, float] | None = None,
    qz_range: tuple[float, float] = (-0.6, 0.6),
    nqx: int = 241,
    nqz: int = 161,
    rod_sigma_qx: float | None = None,
    mode: ApproximationMode = "correct",
) -> MultisliceSimulationResult2D:
    """Simulate 2D CD-RSoXS scattering using the Multislice (Split-Step Fourier)
    method.
    """
    if omega_deg is None:
        omega_arr = np.linspace(-45.0, 45.0, 181, dtype=np.float64)
    else:
        omega_arr = np.asarray(omega_deg, dtype=np.float64)

    nx = structure.nx
    dx = structure.dx_nm
    lx = structure.lx_nm
    lz = structure.lz_nm
    pitch = structure.effective_pitch_nm
    q0 = 2.0 * np.pi / pitch
    n_periods = max(round(lx / pitch), 1)

    qx_fft = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    max_m_nyquist = int(nx // (2 * n_periods)) - 1
    max_m = min(int(max_order), max(max_m_nyquist, 1))
    orders = np.arange(-max_m, max_m + 1, dtype=int)
    order_qx = orders.astype(np.float64) * q0

    order_qz: dict[int, np.ndarray] = {
        int(m): np.empty(omega_arr.size, dtype=np.float64) for m in orders
    }
    order_amplitudes: dict[int, np.ndarray] = {
        int(m): np.empty(omega_arr.size, dtype=np.complex128) for m in orders
    }
    order_efficiencies: dict[int, np.ndarray] = {
        int(m): np.empty(omega_arr.size, dtype=np.float64) for m in orders
    }
    exit_waves = np.empty((omega_arr.size, nx), dtype=np.complex128)

    x0 = float(structure.x_nm[0])

    for i_om, om in enumerate(omega_arr):
        psi_exit, delta_kz, _ = _propagate_multislice_2d_single_omega(
            structure, float(om), qx_fft=qx_fft, mode=mode, record_internal=False
        )
        exit_waves[i_om] = psi_exit
        c_fft = (
            (np.fft.fft(psi_exit) / float(nx))
            * np.exp(-1j * qx_fft * x0)
            * np.exp(-1j * delta_kz * (0.5 * lz))
        )
        for m_val in orders:
            m = int(m_val)
            fft_idx = (m * n_periods) % nx
            amp = complex(c_fft[fft_idx])
            order_qz[m][i_om] = float(delta_kz[fft_idx])
            order_amplitudes[m][i_om] = amp
            order_efficiencies[m][i_om] = float(np.abs(amp) ** 2)

    _, _, internal_wavefield = _propagate_multislice_2d_single_omega(
        structure,
        float(field_omega_deg),
        qx_fft=qx_fft,
        mode=mode,
        record_internal=True,
    )
    assert internal_wavefield is not None

    if qx_range is None:
        qx_max = (max_m + 0.75) * q0
        qx_range = (-qx_max, qx_max)
    qx_centers = np.linspace(float(qx_range[0]), float(qx_range[1]), int(nqx))
    qz_centers = np.linspace(float(qz_range[0]), float(qz_range[1]), int(nqz))
    dqx = float(qx_centers[1] - qx_centers[0]) if nqx > 1 else 0.01
    qx_edges = _edges_from_centers(qx_centers)
    qz_edges = _edges_from_centers(qz_centers)

    sigma_rod = (
        float(rod_sigma_qx) if rod_sigma_qx is not None else max(1.5 * dqx, 0.06 * q0)
    )
    intensity_qx_qz = np.zeros((int(nqx), int(nqz)), dtype=np.float64)
    for m_val, qx_m in zip(orders, order_qx, strict=True):
        m = int(m_val)
        if m == 0:
            continue
        qz_m = order_qz[m]
        eff_m = order_efficiencies[m]
        sort_idx = np.argsort(qz_m)
        qz_sorted = qz_m[sort_idx]
        eff_sorted = eff_m[sort_idx]
        eff_interp = np.interp(qz_centers, qz_sorted, eff_sorted, left=0.0, right=0.0)
        qx_profile = np.exp(-0.5 * ((qx_centers - float(qx_m)) / sigma_rod) ** 2)
        intensity_qx_qz += np.outer(qx_profile, eff_interp)

    qmap = QMap2D(
        H=intensity_qx_qz,
        N=np.ones_like(intensity_qx_qz),
        x_edges=qx_edges,
        y_edges=qz_edges,
        x_domain="Qx",
        y_domain="Qz",
        approximation=mode,
        metadata={
            "simulator": "multislice_2d",
            "energy_ev": structure.energy_ev,
            "Lambda_nm": pitch,
        },
    )

    return MultisliceSimulationResult2D(
        structure=structure,
        omega_deg=omega_arr,
        orders=orders,
        order_qx=order_qx,
        order_qz=order_qz,
        order_amplitudes=order_amplitudes,
        order_efficiencies=order_efficiencies,
        exit_waves=exit_waves,
        internal_wavefield=internal_wavefield,
        field_omega_deg=float(field_omega_deg),
        qmap_qx_qz=qmap,
        mode=mode,
    )


@dataclass
class MultisliceSimulationResult3D:
    """Result of a 3D Multislice beam-propagation simulation from a
    `RefractiveIndexMap3D`.
    """

    structure: RefractiveIndexMap3D
    omega_deg: np.ndarray
    orders: np.ndarray
    order_qx: np.ndarray
    order_qz: dict[int, np.ndarray]
    order_efficiencies: dict[int, np.ndarray]
    qy_grid: np.ndarray
    order_diffuse_qy: dict[int, np.ndarray]
    exit_wave_xy: np.ndarray
    field_omega_deg: float
    qmap_qx_qz: QMap2D
    qmap_qx_qy: QMap2D
    qmap_qy_qz: QMap2D
    mode: ApproximationMode = "correct"
    metadata: dict[str, Any] = field(default_factory=dict)

    def bragg_rod(self, order: int, x_axis: str = "Qz") -> Trace1D:
        """Return the simulated 1D Bragg rod `Trace1D` for diffraction order `m`."""
        m = int(order)
        if m not in self.order_efficiencies:
            msg = f"Order {m} not in simulated orders {self.orders.tolist()}."
            raise KeyError(msg)
        intensity = self.order_efficiencies[m].copy()
        if x_axis == "omega":
            x = self.omega_deg.copy()
            axis_name = "omega"
        elif x_axis == "Qz":
            x = self.order_qz[m].copy()
            axis_name = "Qz"
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            intensity = intensity[sort_idx]
        else:
            msg = f"Unsupported x_axis={x_axis!r}; choose 'Qz' or 'omega'."
            raise ValueError(msg)
        return Trace1D(
            coords=x,
            edges=_edges_from_centers(x),
            intensity=intensity,
            sigma=np.zeros_like(intensity),
            counts=np.ones_like(intensity),
            domain=axis_name,
            metadata={"order": m, "simulator": "multislice_3d", "mode": self.mode},
        )

    def diffuse_qy_trace(self, order: int = 1) -> Trace1D:
        """Return the off-specular `I_m(Qy)` diffuse trace for order `m`."""
        m = int(order)
        if m not in self.order_diffuse_qy:
            msg = f"Order {m} not in simulated orders {self.orders.tolist()}."
            raise KeyError(msg)
        intensity = self.order_diffuse_qy[m].copy()
        return Trace1D(
            coords=self.qy_grid.copy(),
            edges=_edges_from_centers(self.qy_grid),
            intensity=intensity,
            sigma=np.zeros_like(intensity),
            counts=np.ones_like(intensity),
            domain="Qy",
            metadata={"order": m, "simulator": "multislice_3d"},
        )


def _propagate_multislice_3d_single_omega(
    structure: RefractiveIndexMap3D,
    omega_deg: float,
    qx_2d: np.ndarray,
    qy_2d: np.ndarray,
    mode: ApproximationMode = "correct",
) -> tuple[np.ndarray, np.ndarray]:
    """Run Strang-split 3D multislice propagation at a single tilt angle `omega_deg`."""
    k0 = structure.k0_nm_inv
    dz = structure.dz_nm
    cos_om = max(abs(float(np.cos(np.deg2rad(omega_deg)))), 1e-3)
    delta_kz = _longitudinal_delta_kz_2d(
        qx_2d, qy_2d, omega_deg=omega_deg, k0=k0, mode=mode
    )
    half_prop = np.exp(1j * delta_kz * (0.5 * dz))

    dn = structure.delta_n  # (nx, ny, nz)
    psi = np.ones((structure.nx, structure.ny), dtype=np.complex128)
    for iz in range(structure.nz):
        psi_k = np.fft.fft2(psi) * half_prop
        psi = np.fft.ifft2(psi_k)
        psi = psi * np.exp(-1j * (k0 / cos_om) * dn[:, :, iz] * dz)
        psi_k = np.fft.fft2(psi) * half_prop
        psi = np.fft.ifft2(psi_k)
    return psi, delta_kz


def _multislice_3d_mode_grid(
    structure: RefractiveIndexMap3D,
    omega_deg: float,
    qx_2d: np.ndarray,
    qy_2d: np.ndarray,
    mode: ApproximationMode = "correct",
) -> tuple[np.ndarray, np.ndarray]:
    """Exit-wave mode amplitudes ``c(qx, qy)`` on the full FFT grid at one tilt.

    ``c = fft2(psi_exit) / (nx ny) * exp(-i(qx x0 + qy y0)) * exp(-i dkz lz / 2)``
    (the (0, 0) mode carries the transmitted beam). Returns ``(c, delta_kz)``.
    """
    psi_exit, delta_kz = _propagate_multislice_3d_single_omega(
        structure, float(omega_deg), qx_2d=qx_2d, qy_2d=qy_2d, mode=mode
    )
    x0 = float(structure.x_nm[0])
    y0 = float(structure.y_nm[0])
    c = (
        (np.fft.fft2(psi_exit) / float(structure.nx * structure.ny))
        * np.exp(-1j * (qx_2d * x0 + qy_2d * y0))
        * np.exp(-1j * delta_kz * (0.5 * structure.lz_nm))
    )
    return c, delta_kz


def simulate_multislice_3d(
    structure: RefractiveIndexMap3D,
    *,
    omega_deg: Sequence[float] | np.ndarray | None = None,
    max_order: int = 6,
    field_omega_deg: float = 0.0,
    diffuse_order: int = 1,
    qz_range: tuple[float, float] = (-0.6, 0.6),
    nqx: int = 181,
    nqz: int = 121,
    mode: ApproximationMode = "correct",
) -> MultisliceSimulationResult3D:
    """Simulate 3D CD-RSoXS scattering using the 3D Multislice (Split-Step Fourier)
    method.
    """
    if omega_deg is None:
        omega_arr = np.linspace(-40.0, 40.0, 41, dtype=np.float64)
    else:
        omega_arr = np.asarray(omega_deg, dtype=np.float64)

    nx, ny = structure.nx, structure.ny
    dx, dy = structure.dx_nm, structure.dy_nm
    lx, lz = structure.lx_nm, structure.lz_nm
    pitch = structure.effective_pitch_nm
    q0 = 2.0 * np.pi / pitch
    n_periods = max(round(lx / pitch), 1)

    qx_fft = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    qy_fft = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    qx_2d, qy_2d = np.meshgrid(qx_fft, qy_fft, indexing="ij")

    max_m_nyquist = int(nx // (2 * n_periods)) - 1
    max_m = min(int(max_order), max(max_m_nyquist, 1))
    orders = np.arange(-max_m, max_m + 1, dtype=int)
    order_qx = orders.astype(np.float64) * q0

    order_qz: dict[int, np.ndarray] = {
        int(m): np.empty(omega_arr.size, dtype=np.float64) for m in orders
    }
    order_efficiencies: dict[int, np.ndarray] = {
        int(m): np.empty(omega_arr.size, dtype=np.float64) for m in orders
    }

    qy_shifted = np.fft.fftshift(qy_fft)
    diffuse_qy_vs_omega = np.empty((omega_arr.size, ny), dtype=np.float64)
    diff_fft_idx = (int(diffuse_order) * n_periods) % nx

    x0 = float(structure.x_nm[0])
    y0 = float(structure.y_nm[0])
    phase_origin = np.exp(-1j * (qx_2d * x0 + qy_2d * y0))

    for i_om, om in enumerate(omega_arr):
        c_fft2, delta_kz = _multislice_3d_mode_grid(
            structure, float(om), qx_2d=qx_2d, qy_2d=qy_2d, mode=mode
        )
        for m_val in orders:
            m = int(m_val)
            ix = (m * n_periods) % nx
            order_qz[m][i_om] = float(delta_kz[ix, 0])
            order_efficiencies[m][i_om] = float(np.abs(c_fft2[ix, 0]) ** 2)
        diffuse_qy_vs_omega[i_om] = np.fft.fftshift(
            np.abs(c_fft2[diff_fft_idx, :]) ** 2
        )

    exit_wave_xy, delta_kz_field = _propagate_multislice_3d_single_omega(
        structure, float(field_omega_deg), qx_2d=qx_2d, qy_2d=qy_2d, mode=mode
    )
    psi_scattered = exit_wave_xy - np.mean(exit_wave_xy)
    c_field = (
        (np.fft.fft2(psi_scattered) / float(nx * ny))
        * phase_origin
        * np.exp(-1j * delta_kz_field * (0.5 * lz))
    )
    order_diffuse_qy: dict[int, np.ndarray] = {}
    for m_val in orders:
        m = int(m_val)
        ix = (m * n_periods) % nx
        order_diffuse_qy[m] = np.fft.fftshift(np.abs(c_field[ix, :]) ** 2)

    qx_shifted = np.fft.fftshift(qx_fft)
    int_qx_qy_full = np.fft.fftshift(np.abs(c_field) ** 2)  # (nx, ny)
    qx_keep = np.abs(qx_shifted) <= (max_m + 0.75) * q0
    qx_sub = qx_shifted[qx_keep]
    int_qx_qy = int_qx_qy_full[qx_keep, :]  # (nqx_sub, ny)

    qmap_qx_qy = QMap2D(
        H=int_qx_qy,
        N=np.ones_like(int_qx_qy),
        x_edges=_edges_from_centers(qx_sub),
        y_edges=_edges_from_centers(qy_shifted),
        x_domain="Qx",
        y_domain="Qy",
        cuts={"omega": (float(field_omega_deg), float(field_omega_deg))},
        approximation=mode,
        metadata={
            "simulator": "multislice_3d",
            "omega_deg": float(field_omega_deg),
            "Lambda_nm": pitch,
        },
    )

    qx_centers = np.linspace(-(max_m + 0.75) * q0, (max_m + 0.75) * q0, int(nqx))
    qz_centers = np.linspace(float(qz_range[0]), float(qz_range[1]), int(nqz))
    sigma_rod = max(1.5 * float(qx_centers[1] - qx_centers[0]), 0.06 * q0)
    int_qx_qz = np.zeros((int(nqx), int(nqz)), dtype=np.float64)
    for m_val, qx_m in zip(orders, order_qx, strict=True):
        m = int(m_val)
        if m == 0:
            continue
        qz_m = order_qz[m]
        eff_m = order_efficiencies[m]
        sort_idx = np.argsort(qz_m)
        eff_interp = np.interp(
            qz_centers, qz_m[sort_idx], eff_m[sort_idx], left=0.0, right=0.0
        )
        qx_prof = np.exp(-0.5 * ((qx_centers - float(qx_m)) / sigma_rod) ** 2)
        int_qx_qz += np.outer(qx_prof, eff_interp)

    qmap_qx_qz = QMap2D(
        H=int_qx_qz,
        N=np.ones_like(int_qx_qz),
        x_edges=_edges_from_centers(qx_centers),
        y_edges=_edges_from_centers(qz_centers),
        x_domain="Qx",
        y_domain="Qz",
        cuts={"Qy": (0.0, 0.0)},
        approximation=mode,
        metadata={"simulator": "multislice_3d", "Lambda_nm": pitch},
    )

    qz_diff = order_qz[int(diffuse_order)]
    sort_idx = np.argsort(qz_diff)
    qz_sorted = qz_diff[sort_idx]
    diff_sorted = diffuse_qy_vs_omega[sort_idx, :]  # (n_omega, ny)
    int_qy_qz = np.zeros((ny, int(nqz)), dtype=np.float64)
    for iy in range(ny):
        int_qy_qz[iy, :] = np.interp(
            qz_centers, qz_sorted, diff_sorted[:, iy], left=0.0, right=0.0
        )

    qmap_qy_qz = QMap2D(
        H=int_qy_qz,
        N=np.ones_like(int_qy_qz),
        x_edges=_edges_from_centers(qy_shifted),
        y_edges=_edges_from_centers(qz_centers),
        x_domain="Qy",
        y_domain="Qz",
        cuts={"Qx": (float(diffuse_order) * q0, float(diffuse_order) * q0)},
        approximation=mode,
        metadata={"simulator": "multislice_3d", "diffuse_order": int(diffuse_order)},
    )

    return MultisliceSimulationResult3D(
        structure=structure,
        omega_deg=omega_arr,
        orders=orders,
        order_qx=order_qx,
        order_qz=order_qz,
        order_efficiencies=order_efficiencies,
        qy_grid=qy_shifted,
        order_diffuse_qy=order_diffuse_qy,
        exit_wave_xy=exit_wave_xy,
        field_omega_deg=float(field_omega_deg),
        qmap_qx_qz=qmap_qx_qz,
        qmap_qx_qy=qmap_qx_qy,
        qmap_qy_qz=qmap_qy_qz,
        mode=mode,
    )


__all__ = [
    "MultisliceSimulationResult2D",
    "MultisliceSimulationResult3D",
    "simulate_multislice_2d",
    "simulate_multislice_3d",
]
