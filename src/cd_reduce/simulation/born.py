"""2D and 3D Born-approximation (kinematic) scattering simulators for CD-RSoXS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import ApproximationMode, Geometry, compute_pixel_q
from cd_reduce.reduction.mapping import QMap2D, Trace1D
from cd_reduce.simulation.structures import RefractiveIndexMap2D, RefractiveIndexMap3D

if TYPE_CHECKING:
    from collections.abc import Sequence


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.size < 2:
        val = float(c[0]) if c.size == 1 else 0.0
        return np.array([val - 0.005, val + 0.005], dtype=np.float64)
    diffs = 0.5 * np.diff(c)
    return np.concatenate([[c[0] - diffs[0]], c[:-1] + diffs, [c[-1] + diffs[-1]]])


def _order_ewald_qz(
    qx: np.ndarray | float,
    omega_deg: np.ndarray | float,
    k0: float,
    mode: ApproximationMode = "correct",
) -> np.ndarray:
    """Compute sample-frame `Qz` for given `Qx` and tilt `omega_deg`."""
    qx_arr = np.asarray(qx, dtype=np.float64)
    om_rad = np.deg2rad(np.asarray(omega_deg, dtype=np.float64))
    if mode in ("small_angle", "sax"):
        return -qx_arr * np.tan(om_rad)
    if mode in ("angle_linear", "lin"):
        return -qx_arr * np.sin(om_rad)
    kx_in = k0 * np.sin(om_rad)
    kz_in = k0 * np.cos(om_rad)
    arg = np.maximum(k0 * k0 - (kx_in + qx_arr) ** 2, 0.0)
    qz_ewald = np.sqrt(arg) - kz_in
    return np.asarray(qz_ewald, dtype=np.float64)


@dataclass
class BornSimulationResult2D:
    """Result of a 2D Born-approximation simulation from a `RefractiveIndexMap2D`."""

    structure: RefractiveIndexMap2D
    qmap_qx_qz: QMap2D
    amplitude_qx_qz: np.ndarray
    omega_deg: np.ndarray
    orders: np.ndarray
    order_qx: np.ndarray
    order_qz: dict[int, np.ndarray]
    order_amplitudes: dict[int, np.ndarray]
    order_efficiencies: dict[int, np.ndarray]
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
            metadata={"order": m, "simulator": "born_2d", "mode": self.mode},
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
        """Project the 2D Born scattering onto a synthetic 2D CCD `Frame`."""
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
        k0 = self.structure.k0_nm_inv
        cos_om = max(abs(float(np.cos(np.deg2rad(omega_deg)))), 1e-3)

        z_centered = self.structure.z_nm - float(np.mean(self.structure.z_nm))
        dx = self.structure.dx_nm
        dz = self.structure.dz_nm
        lx = self.structure.lx_nm
        dn = self.structure.delta_n  # (nx, nz)

        image = np.full((ny_det, nx_det), float(background_adu), dtype=np.float64)
        sigma_q = max(float(spot_sigma_q), 1e-6)
        qy_env = np.exp(-0.5 * (pq.Qy / sigma_q) ** 2)

        for m, qx_m in zip(self.orders, self.order_qx, strict=True):
            if int(m) == 0:
                continue
            qx_env = np.exp(-0.5 * ((pq.Qx - float(qx_m)) / sigma_q) ** 2)
            mask = (qx_env * qy_env) > 1e-5
            if not np.any(mask):
                continue
            phase_x = np.exp(-1j * float(qx_m) * self.structure.x_nm)
            c_m_z = (phase_x @ dn) * (dx / lx)
            qz_pts = pq.Qz[mask]
            phase_z = np.exp(-1j * qz_pts[:, None] * z_centered[None, :])
            amp_pts = (-1j * k0 / cos_om) * (phase_z @ c_m_z) * dz
            eff_pts = np.abs(amp_pts) ** 2
            image[mask] += float(flux_scale) * eff_pts * qx_env[mask] * qy_env[mask]

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


def simulate_born_2d(
    structure: RefractiveIndexMap2D,
    *,
    omega_deg: Sequence[float] | np.ndarray | None = None,
    max_order: int = 8,
    qx_range: tuple[float, float] | None = None,
    qz_range: tuple[float, float] = (-0.6, 0.6),
    nqx: int = 241,
    nqz: int = 161,
    mode: ApproximationMode = "correct",
    include_zero_order: bool = False,
) -> BornSimulationResult2D:
    """Simulate 2D CD-RSoXS scattering in the Born (kinematic) approximation."""
    if omega_deg is None:
        omega_arr = np.linspace(-45.0, 45.0, 181, dtype=np.float64)
    else:
        omega_arr = np.asarray(omega_deg, dtype=np.float64)

    pitch = structure.effective_pitch_nm
    q0 = 2.0 * np.pi / pitch
    if qx_range is None:
        qx_max = (int(max_order) + 0.75) * q0
        qx_range = (-qx_max, qx_max)

    qx_centers = np.linspace(float(qx_range[0]), float(qx_range[1]), int(nqx))
    qz_centers = np.linspace(float(qz_range[0]), float(qz_range[1]), int(nqz))
    qx_edges = _edges_from_centers(qx_centers)
    qz_edges = _edges_from_centers(qz_centers)

    k0 = structure.k0_nm_inv
    dx = structure.dx_nm
    dz = structure.dz_nm
    lx = structure.lx_nm
    z_centered = structure.z_nm - float(np.mean(structure.z_nm))

    dn = structure.delta_n.copy()
    dn_for_map = dn if include_zero_order else (dn - np.mean(dn, axis=0, keepdims=True))

    exp_x = np.exp(-1j * np.outer(qx_centers, structure.x_nm)) * (dx / lx)  # (nqx, nx)
    exp_z = np.exp(-1j * np.outer(qz_centers, z_centered)) * dz  # (nqz, nz)
    amp_qx_qz = -1j * k0 * (exp_x @ dn_for_map @ exp_z.T)
    intensity_qx_qz = np.abs(amp_qx_qz) ** 2

    qmap = QMap2D(
        H=intensity_qx_qz,
        N=np.ones_like(intensity_qx_qz),
        x_edges=qx_edges,
        y_edges=qz_edges,
        x_domain="Qx",
        y_domain="Qz",
        approximation=mode,
        metadata={
            "simulator": "born_2d",
            "energy_ev": structure.energy_ev,
            "Lambda_nm": pitch,
        },
    )

    orders = np.arange(-int(max_order), int(max_order) + 1, dtype=int)
    order_qx = orders.astype(np.float64) * q0
    cos_om = np.maximum(np.abs(np.cos(np.deg2rad(omega_arr))), 1e-3)

    order_qz: dict[int, np.ndarray] = {}
    order_amplitudes: dict[int, np.ndarray] = {}
    order_efficiencies: dict[int, np.ndarray] = {}

    for m_val, qx_m in zip(orders, order_qx, strict=True):
        m = int(m_val)
        qz_m = _order_ewald_qz(qx_m, omega_arr, k0=k0, mode=mode)
        phase_x = np.exp(-1j * float(qx_m) * structure.x_nm) * (dx / lx)
        c_m_z = phase_x @ dn
        phase_z = np.exp(-1j * np.outer(qz_m, z_centered)) * dz
        amp_m = (-1j * k0 / cos_om) * (phase_z @ c_m_z)
        if m == 0:
            amp_m = 1.0 + amp_m
        order_qz[m] = qz_m
        order_amplitudes[m] = amp_m
        order_efficiencies[m] = np.abs(amp_m) ** 2

    return BornSimulationResult2D(
        structure=structure,
        qmap_qx_qz=qmap,
        amplitude_qx_qz=amp_qx_qz,
        omega_deg=omega_arr,
        orders=orders,
        order_qx=order_qx,
        order_qz=order_qz,
        order_amplitudes=order_amplitudes,
        order_efficiencies=order_efficiencies,
        mode=mode,
    )


@dataclass
class BornSimulationResult3D:
    """Result of a 3D Born-approximation simulation from a `RefractiveIndexMap3D`."""

    structure: RefractiveIndexMap3D
    qmap_qx_qz: QMap2D
    qmap_qx_qy: QMap2D
    qmap_qy_qz: QMap2D
    omega_deg: np.ndarray
    orders: np.ndarray
    order_qx: np.ndarray
    order_qz: dict[int, np.ndarray]
    order_efficiencies: dict[int, np.ndarray]
    qy_grid: np.ndarray
    order_diffuse_qy: dict[int, np.ndarray]
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
            metadata={"order": m, "simulator": "born_3d", "mode": self.mode},
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
            metadata={"order": m, "simulator": "born_3d"},
        )

    def to_detector_frame(
        self,
        geometry: Geometry,
        omega_deg: float = 0.0,
        *,
        shape: tuple[int, int] = (256, 256),
        flux_scale: float = 1e6,
        background_adu: float = 0.0,
        poisson_noise: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> Frame:
        """Project the 3D Born scattering onto a synthetic 2D CCD `Frame`."""
        from scipy.interpolate import RegularGridInterpolator

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
        interp_qx_qy = RegularGridInterpolator(
            (self.qmap_qx_qy.x_centers, self.qmap_qx_qy.y_centers),
            self.qmap_qx_qy.mean,
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.column_stack([pq.Qx.ravel(), pq.Qy.ravel()])
        img_flat = float(background_adu) + float(flux_scale) * interp_qx_qy(pts)
        image = img_flat.reshape((ny_det, nx_det))
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


def simulate_born_3d(
    structure: RefractiveIndexMap3D,
    *,
    omega_deg: Sequence[float] | np.ndarray | None = None,
    max_order: int = 6,
    qx_range: tuple[float, float] | None = None,
    qy_range: tuple[float, float] | None = None,
    qz_range: tuple[float, float] = (-0.6, 0.6),
    nqx: int = 181,
    nqy: int = 121,
    nqz: int = 121,
    fixed_qz: float = 0.0,
    diffuse_order: int = 1,
    mode: ApproximationMode = "correct",
    include_zero_order: bool = False,
) -> BornSimulationResult3D:
    """Simulate 3D CD-RSoXS scattering in the Born (kinematic) approximation."""
    if omega_deg is None:
        omega_arr = np.linspace(-45.0, 45.0, 181, dtype=np.float64)
    else:
        omega_arr = np.asarray(omega_deg, dtype=np.float64)

    pitch = structure.effective_pitch_nm
    q0 = 2.0 * np.pi / pitch
    if qx_range is None:
        qx_max = (int(max_order) + 0.75) * q0
        qx_range = (-qx_max, qx_max)
    if qy_range is None:
        qy_max = min(0.5 * np.pi / max(structure.dy_nm, 1.0), 0.25)
        qy_range = (-qy_max, qy_max)

    qx_centers = np.linspace(float(qx_range[0]), float(qx_range[1]), int(nqx))
    qy_centers = np.linspace(float(qy_range[0]), float(qy_range[1]), int(nqy))
    qz_centers = np.linspace(float(qz_range[0]), float(qz_range[1]), int(nqz))

    qx_edges = _edges_from_centers(qx_centers)
    qy_edges = _edges_from_centers(qy_centers)
    qz_edges = _edges_from_centers(qz_centers)

    k0 = structure.k0_nm_inv
    dx = structure.dx_nm
    dy = structure.dy_nm
    dz = structure.dz_nm
    lx = structure.lx_nm
    ly = structure.ly_nm
    z_centered = structure.z_nm - float(np.mean(structure.z_nm))

    dn = structure.delta_n  # (nx, ny, nz)
    dn_map = (
        dn if include_zero_order else (dn - np.mean(dn, axis=(0, 1), keepdims=True))
    )

    # 1. (Qx, Qz) map at Qy = 0
    dn_y_avg = np.mean(dn_map, axis=1)  # (nx, nz)
    exp_x = np.exp(-1j * np.outer(qx_centers, structure.x_nm)) * (dx / lx)  # (nqx, nx)
    exp_z = np.exp(-1j * np.outer(qz_centers, z_centered)) * dz  # (nqz, nz)
    amp_qx_qz = -1j * k0 * (exp_x @ dn_y_avg @ exp_z.T)  # (nqx, nqz)
    int_qx_qz = np.abs(amp_qx_qz) ** 2
    qmap_qx_qz = QMap2D(
        H=int_qx_qz,
        N=np.ones_like(int_qx_qz),
        x_edges=qx_edges,
        y_edges=qz_edges,
        x_domain="Qx",
        y_domain="Qz",
        cuts={"Qy": (0.0, 0.0)},
        approximation=mode,
        metadata={"simulator": "born_3d", "Lambda_nm": pitch},
    )

    # 2. (Qx, Qy) map at Qz = fixed_qz
    phase_z_fixed = np.exp(-1j * float(fixed_qz) * z_centered) * dz  # (nz,)
    dn_xy_proj = np.tensordot(dn_map, phase_z_fixed, axes=([2], [0]))  # (nx, ny)
    exp_y = np.exp(-1j * np.outer(qy_centers, structure.y_nm)) * (dy / ly)  # (nqy, ny)
    amp_qx_qy = -1j * k0 * (exp_x @ dn_xy_proj @ exp_y.T)  # (nqx, nqy)
    int_qx_qy = np.abs(amp_qx_qy) ** 2
    qmap_qx_qy = QMap2D(
        H=int_qx_qy,
        N=np.ones_like(int_qx_qy),
        x_edges=qx_edges,
        y_edges=qy_edges,
        x_domain="Qx",
        y_domain="Qy",
        cuts={"Qz": (float(fixed_qz), float(fixed_qz))},
        approximation=mode,
        metadata={
            "simulator": "born_3d",
            "fixed_qz": float(fixed_qz),
            "Lambda_nm": pitch,
        },
    )

    # 3. (Qy, Qz) map at Qx = diffuse_order * q0
    qx_diff = float(diffuse_order) * q0
    phase_x_diff = np.exp(-1j * qx_diff * structure.x_nm) * (dx / lx)  # (nx,)
    dn_yz_proj = np.tensordot(phase_x_diff, dn_map, axes=([0], [0]))  # (ny, nz)
    amp_qy_qz = -1j * k0 * (exp_y @ dn_yz_proj @ exp_z.T)  # (nqy, nqz)
    int_qy_qz = np.abs(amp_qy_qz) ** 2
    qmap_qy_qz = QMap2D(
        H=int_qy_qz,
        N=np.ones_like(int_qy_qz),
        x_edges=qy_edges,
        y_edges=qz_edges,
        x_domain="Qy",
        y_domain="Qz",
        cuts={"Qx": (qx_diff, qx_diff)},
        approximation=mode,
        metadata={
            "simulator": "born_3d",
            "diffuse_order": int(diffuse_order),
            "qx": qx_diff,
        },
    )

    # 4. Bragg rod efficiencies and diffuse Qy profiles for each order m
    orders = np.arange(-int(max_order), int(max_order) + 1, dtype=int)
    order_qx = orders.astype(np.float64) * q0
    cos_om = np.maximum(np.abs(np.cos(np.deg2rad(omega_arr))), 1e-3)

    order_qz: dict[int, np.ndarray] = {}
    order_efficiencies: dict[int, np.ndarray] = {}
    order_diffuse_qy: dict[int, np.ndarray] = {}

    dn_y_mean_full = np.mean(dn, axis=1)  # (nx, nz)
    for m_val, qx_m in zip(orders, order_qx, strict=True):
        m = int(m_val)
        qz_m = _order_ewald_qz(qx_m, omega_arr, k0=k0, mode=mode)
        phase_x_m = np.exp(-1j * float(qx_m) * structure.x_nm) * (dx / lx)
        c_m_z = phase_x_m @ dn_y_mean_full
        phase_z_m = np.exp(-1j * np.outer(qz_m, z_centered)) * dz
        amp_m = (-1j * k0 / cos_om) * (phase_z_m @ c_m_z)
        if m == 0:
            amp_m = 1.0 + amp_m
        order_qz[m] = qz_m
        order_efficiencies[m] = np.abs(amp_m) ** 2

        c_m_y = phase_x_m @ dn_xy_proj
        amp_qy_m = -1j * k0 * (exp_y @ c_m_y)
        order_diffuse_qy[m] = np.abs(amp_qy_m) ** 2

    return BornSimulationResult3D(
        structure=structure,
        qmap_qx_qz=qmap_qx_qz,
        qmap_qx_qy=qmap_qx_qy,
        qmap_qy_qz=qmap_qy_qz,
        omega_deg=omega_arr,
        orders=orders,
        order_qx=order_qx,
        order_qz=order_qz,
        order_efficiencies=order_efficiencies,
        qy_grid=qy_centers,
        order_diffuse_qy=order_diffuse_qy,
        mode=mode,
    )


__all__ = [
    "BornSimulationResult2D",
    "BornSimulationResult3D",
    "simulate_born_2d",
    "simulate_born_3d",
]
