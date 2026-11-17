"""General 2D and 3D complex refractive index maps and grating builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cd_reduce.utils.units import energy_to_wavelength_nm, wavelength_nm_to_k


@dataclass
class RefractiveIndexMap2D:
    """General 2D complex refractive index structure in `(x, z)`.

    Follows the soft X-ray convention `n(x, z) = 1 - delta(x, z) + i * beta(x, z)`
    where `x` is the in-plane lateral coordinate (shape `nx`) and `z` is the
    through-thickness / propagation coordinate (shape `nz`).

    Attributes
    ----------
    x_nm : np.ndarray
        1D array of lateral `x` coordinates in nm, shape `(nx,)`.
    z_nm : np.ndarray
        1D array of depth `z` coordinates in nm, shape `(nz,)`.
    delta : np.ndarray
        2D array of real refractive index decrement `delta(x, z)`, shape `(nx, nz)`.
    beta : np.ndarray
        2D array of absorption index `beta(x, z)`, shape `(nx, nz)`.
    energy_ev : float
        Photon energy in eV.
    pitch_nm : float | None
        Fundamental grating pitch in nm (defaults to the lateral domain size `lx_nm`).
    delta_ref : float
        Reference ambient/background `delta` (default 0.0 for vacuum).
    beta_ref : float
        Reference ambient/background `beta` (default 0.0 for vacuum).
    metadata : dict[str, Any]
        Optional structure metadata.
    """

    x_nm: np.ndarray
    z_nm: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    energy_ev: float = 285.0
    pitch_nm: float | None = None
    delta_ref: float = 0.0
    beta_ref: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_nm = np.asarray(self.x_nm, dtype=np.float64)
        self.z_nm = np.asarray(self.z_nm, dtype=np.float64)
        self.delta = np.asarray(self.delta, dtype=np.float64)
        self.beta = np.asarray(self.beta, dtype=np.float64)
        expected_shape = (self.x_nm.size, self.z_nm.size)
        if self.delta.shape != expected_shape or self.beta.shape != expected_shape:
            msg = (
                f"delta {self.delta.shape} and beta {self.beta.shape} must match "
                f"(nx, nz) = {expected_shape}."
            )
            raise ValueError(msg)

    @classmethod
    def from_complex_n(
        cls,
        n: np.ndarray,
        x_nm: np.ndarray | None = None,
        z_nm: np.ndarray | None = None,
        *,
        dx_nm: float = 1.0,
        dz_nm: float = 1.0,
        energy_ev: float = 285.0,
        pitch_nm: float | None = None,
        n_ref: complex = 1.0 + 0.0j,
        metadata: dict[str, Any] | None = None,
    ) -> RefractiveIndexMap2D:
        """Construct `RefractiveIndexMap2D` from complex `n = 1 - delta + i*beta`."""
        n_arr = np.asarray(n, dtype=np.complex128)
        if n_arr.ndim != 2:
            msg = f"Expected a 2D array for n(x, z), got shape {n_arr.shape}."
            raise ValueError(msg)
        nx, nz = n_arr.shape
        if x_nm is None:
            x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * float(dx_nm)
        if z_nm is None:
            z_nm = np.arange(nz, dtype=np.float64) * float(dz_nm)
        delta = 1.0 - np.real(n_arr)
        beta = np.imag(n_arr)
        delta_ref = float(1.0 - np.real(n_ref))
        beta_ref = float(np.imag(n_ref))
        return cls(
            x_nm=x_nm,
            z_nm=z_nm,
            delta=delta,
            beta=beta,
            energy_ev=energy_ev,
            pitch_nm=pitch_nm,
            delta_ref=delta_ref,
            beta_ref=beta_ref,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_delta_beta(
        cls,
        delta: np.ndarray,
        beta: np.ndarray | None = None,
        x_nm: np.ndarray | None = None,
        z_nm: np.ndarray | None = None,
        *,
        dx_nm: float = 1.0,
        dz_nm: float = 1.0,
        energy_ev: float = 285.0,
        pitch_nm: float | None = None,
        delta_ref: float = 0.0,
        beta_ref: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> RefractiveIndexMap2D:
        """Construct a `RefractiveIndexMap2D` from `delta(x, z)` and `beta(x, z)`
        arrays.
        """
        delta_arr = np.asarray(delta, dtype=np.float64)
        if delta_arr.ndim != 2:
            msg = f"Expected a 2D array for delta(x, z), got shape {delta_arr.shape}."
            raise ValueError(msg)
        beta_arr = (
            np.zeros_like(delta_arr)
            if beta is None
            else np.asarray(beta, dtype=np.float64)
        )
        nx, nz = delta_arr.shape
        if x_nm is None:
            x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * float(dx_nm)
        if z_nm is None:
            z_nm = np.arange(nz, dtype=np.float64) * float(dz_nm)
        return cls(
            x_nm=x_nm,
            z_nm=z_nm,
            delta=delta_arr,
            beta=beta_arr,
            energy_ev=energy_ev,
            pitch_nm=pitch_nm,
            delta_ref=delta_ref,
            beta_ref=beta_ref,
            metadata=dict(metadata or {}),
        )

    @property
    def nx(self) -> int:
        """Number of grid points along `x`."""
        return int(self.x_nm.size)

    @property
    def nz(self) -> int:
        """Number of grid points along `z`."""
        return int(self.z_nm.size)

    @property
    def dx_nm(self) -> float:
        """Grid spacing along `x` in nm."""
        if self.nx < 2:
            return 1.0
        return float(np.mean(np.diff(self.x_nm)))

    @property
    def dz_nm(self) -> float:
        """Grid spacing along `z` in nm."""
        if self.nz < 2:
            return 1.0
        return float(np.mean(np.diff(self.z_nm)))

    @property
    def lx_nm(self) -> float:
        """Total lateral domain width in nm."""
        return float(self.nx * self.dx_nm)

    @property
    def lz_nm(self) -> float:
        """Total thickness in nm."""
        return float(self.nz * self.dz_nm)

    @property
    def effective_pitch_nm(self) -> float:
        """Fundamental pitch in nm (or `lx_nm` if `pitch_nm` is not set)."""
        return float(self.pitch_nm) if self.pitch_nm is not None else self.lx_nm

    @property
    def wavelength_nm(self) -> float:
        """Vacuum photon wavelength in nm."""
        return float(energy_to_wavelength_nm(self.energy_ev))

    @property
    def k0_nm_inv(self) -> float:
        """Vacuum wavenumber `2*pi / lambda` in nm^-1."""
        return float(wavelength_nm_to_k(self.wavelength_nm))

    @property
    def n(self) -> np.ndarray:
        """Complex refractive index `n(x, z) = 1 - delta(x, z) + i * beta(x, z)`."""
        return (1.0 - self.delta) + 1j * self.beta

    @property
    def delta_n(self) -> np.ndarray:
        """Complex refractive index decrement `(delta - delta_ref) - i * (beta -
        beta_ref)`.
        """
        return (self.delta - self.delta_ref) - 1j * (self.beta - self.beta_ref)

    def to_3d(self, ny: int = 32, ly_nm: float = 500.0) -> RefractiveIndexMap3D:
        """Extrude this 2D `(x, z)` structure uniformly along `y` into a 3D map."""
        dy_nm = float(ly_nm) / max(int(ny), 1)
        y_nm = (np.arange(ny, dtype=np.float64) - 0.5 * ny) * dy_nm
        delta_3d = np.broadcast_to(
            self.delta[:, None, :], (self.nx, int(ny), self.nz)
        ).copy()
        beta_3d = np.broadcast_to(
            self.beta[:, None, :], (self.nx, int(ny), self.nz)
        ).copy()
        return RefractiveIndexMap3D(
            x_nm=self.x_nm.copy(),
            y_nm=y_nm,
            z_nm=self.z_nm.copy(),
            delta=delta_3d,
            beta=beta_3d,
            energy_ev=self.energy_ev,
            pitch_nm=self.pitch_nm,
            delta_ref=self.delta_ref,
            beta_ref=self.beta_ref,
            metadata=dict(self.metadata),
        )


@dataclass
class RefractiveIndexMap3D:
    """General 3D complex refractive index structure in `(x, y, z)`.

    Follows the soft X-ray convention `n(x, y, z) = 1 - delta(x, y, z) + i * beta(x, y,
    z)`
    where `x` is across the grating lines (`nx`), `y` is along the grating lines (`ny`),
    and `z` is the through-thickness / propagation axis (`nz`).
    """

    x_nm: np.ndarray
    y_nm: np.ndarray
    z_nm: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    energy_ev: float = 285.0
    pitch_nm: float | None = None
    delta_ref: float = 0.0
    beta_ref: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_nm = np.asarray(self.x_nm, dtype=np.float64)
        self.y_nm = np.asarray(self.y_nm, dtype=np.float64)
        self.z_nm = np.asarray(self.z_nm, dtype=np.float64)
        self.delta = np.asarray(self.delta, dtype=np.float64)
        self.beta = np.asarray(self.beta, dtype=np.float64)
        expected_shape = (self.x_nm.size, self.y_nm.size, self.z_nm.size)
        if self.delta.shape != expected_shape or self.beta.shape != expected_shape:
            msg = (
                f"delta {self.delta.shape} and beta {self.beta.shape} must match "
                f"(nx, ny, nz) = {expected_shape}."
            )
            raise ValueError(msg)

    @classmethod
    def from_complex_n(
        cls,
        n: np.ndarray,
        x_nm: np.ndarray | None = None,
        y_nm: np.ndarray | None = None,
        z_nm: np.ndarray | None = None,
        *,
        dx_nm: float = 1.0,
        dy_nm: float = 5.0,
        dz_nm: float = 1.0,
        energy_ev: float = 285.0,
        pitch_nm: float | None = None,
        n_ref: complex = 1.0 + 0.0j,
        metadata: dict[str, Any] | None = None,
    ) -> RefractiveIndexMap3D:
        """Construct `RefractiveIndexMap3D` from complex `n = 1 - delta + i*beta`."""
        n_arr = np.asarray(n, dtype=np.complex128)
        if n_arr.ndim != 3:
            msg = f"Expected a 3D array for n(x, y, z), got shape {n_arr.shape}."
            raise ValueError(msg)
        nx, ny, nz = n_arr.shape
        if x_nm is None:
            x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * float(dx_nm)
        if y_nm is None:
            y_nm = (np.arange(ny, dtype=np.float64) - 0.5 * ny) * float(dy_nm)
        if z_nm is None:
            z_nm = np.arange(nz, dtype=np.float64) * float(dz_nm)
        delta = 1.0 - np.real(n_arr)
        beta = np.imag(n_arr)
        return cls(
            x_nm=x_nm,
            y_nm=y_nm,
            z_nm=z_nm,
            delta=delta,
            beta=beta,
            energy_ev=energy_ev,
            pitch_nm=pitch_nm,
            delta_ref=float(1.0 - np.real(n_ref)),
            beta_ref=float(np.imag(n_ref)),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_delta_beta(
        cls,
        delta: np.ndarray,
        beta: np.ndarray | None = None,
        x_nm: np.ndarray | None = None,
        y_nm: np.ndarray | None = None,
        z_nm: np.ndarray | None = None,
        *,
        dx_nm: float = 1.0,
        dy_nm: float = 5.0,
        dz_nm: float = 1.0,
        energy_ev: float = 285.0,
        pitch_nm: float | None = None,
        delta_ref: float = 0.0,
        beta_ref: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> RefractiveIndexMap3D:
        """Construct a `RefractiveIndexMap3D` from `delta(x, y, z)` and `beta(x, y,
        z)`.
        """
        delta_arr = np.asarray(delta, dtype=np.float64)
        if delta_arr.ndim != 3:
            msg = (
                f"Expected a 3D array for delta(x, y, z), got shape {delta_arr.shape}."
            )
            raise ValueError(msg)
        beta_arr = (
            np.zeros_like(delta_arr)
            if beta is None
            else np.asarray(beta, dtype=np.float64)
        )
        nx, ny, nz = delta_arr.shape
        if x_nm is None:
            x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * float(dx_nm)
        if y_nm is None:
            y_nm = (np.arange(ny, dtype=np.float64) - 0.5 * ny) * float(dy_nm)
        if z_nm is None:
            z_nm = np.arange(nz, dtype=np.float64) * float(dz_nm)
        return cls(
            x_nm=x_nm,
            y_nm=y_nm,
            z_nm=z_nm,
            delta=delta_arr,
            beta=beta_arr,
            energy_ev=energy_ev,
            pitch_nm=pitch_nm,
            delta_ref=delta_ref,
            beta_ref=beta_ref,
            metadata=dict(metadata or {}),
        )

    @property
    def nx(self) -> int:
        """Number of grid points along `x`."""
        return int(self.x_nm.size)

    @property
    def ny(self) -> int:
        """Number of grid points along `y`."""
        return int(self.y_nm.size)

    @property
    def nz(self) -> int:
        """Number of grid points along `z`."""
        return int(self.z_nm.size)

    @property
    def dx_nm(self) -> float:
        """Grid spacing along `x` in nm."""
        if self.nx < 2:
            return 1.0
        return float(np.mean(np.diff(self.x_nm)))

    @property
    def dy_nm(self) -> float:
        """Grid spacing along `y` in nm."""
        if self.ny < 2:
            return 1.0
        return float(np.mean(np.diff(self.y_nm)))

    @property
    def dz_nm(self) -> float:
        """Grid spacing along `z` in nm."""
        if self.nz < 2:
            return 1.0
        return float(np.mean(np.diff(self.z_nm)))

    @property
    def lx_nm(self) -> float:
        """Total domain width along `x` in nm."""
        return float(self.nx * self.dx_nm)

    @property
    def ly_nm(self) -> float:
        """Total domain length along `y` in nm."""
        return float(self.ny * self.dy_nm)

    @property
    def lz_nm(self) -> float:
        """Total thickness along `z` in nm."""
        return float(self.nz * self.dz_nm)

    @property
    def effective_pitch_nm(self) -> float:
        """Fundamental pitch in nm (or `lx_nm` if `pitch_nm` is not set)."""
        return float(self.pitch_nm) if self.pitch_nm is not None else self.lx_nm

    @property
    def wavelength_nm(self) -> float:
        """Vacuum photon wavelength in nm."""
        return float(energy_to_wavelength_nm(self.energy_ev))

    @property
    def k0_nm_inv(self) -> float:
        """Vacuum wavenumber `2*pi / lambda` in nm^-1."""
        return float(wavelength_nm_to_k(self.wavelength_nm))

    @property
    def n(self) -> np.ndarray:
        """Complex refractive index `n(x, y, z) = 1 - delta + i * beta`."""
        return (1.0 - self.delta) + 1j * self.beta

    @property
    def delta_n(self) -> np.ndarray:
        """Complex refractive index decrement `(delta - delta_ref) - i * (beta -
        beta_ref)`.
        """
        return (self.delta - self.delta_ref) - 1j * (self.beta - self.beta_ref)

    def slice_2d_xz(self, y_index: int | None = None) -> RefractiveIndexMap2D:
        """Return a 2D `(x, z)` cross-section at `y_index` (or averaged over `y` if
        None).
        """
        if y_index is None:
            delta_2d = np.mean(self.delta, axis=1)
            beta_2d = np.mean(self.beta, axis=1)
        else:
            delta_2d = self.delta[:, int(y_index), :].copy()
            beta_2d = self.beta[:, int(y_index), :].copy()
        return RefractiveIndexMap2D(
            x_nm=self.x_nm.copy(),
            z_nm=self.z_nm.copy(),
            delta=delta_2d,
            beta=beta_2d,
            energy_ev=self.energy_ev,
            pitch_nm=self.pitch_nm,
            delta_ref=self.delta_ref,
            beta_ref=self.beta_ref,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class GratingLayerSpec:
    """Specification of a single layer in a 2D or 3D line-space grating stack.

    Attributes
    ----------
    thickness_nm : float
        Layer thickness along `z` in nm.
    cd_bottom_nm : float
        Line critical dimension (width) at the bottom of the layer in nm.
    cd_top_nm : float | None
        Line critical dimension at the top of the layer in nm. If None, computed
        from `sidewall_angle_deg` (default 90 deg -> `cd_top_nm = cd_bottom_nm`).
    sidewall_angle_deg : float
        Sidewall angle in degrees (90 deg is vertical; <90 deg tapers inward at the
        top).
    delta_line : float
        Real refractive index decrement `delta` of the line material.
    beta_line : float
        Absorption index `beta` of the line material.
    delta_space : float
        Real refractive index decrement `delta` of the space/trench material.
    beta_space : float
        Absorption index `beta` of the space/trench material.
    x_center_nm : float
        Lateral center offset of the line in nm.
    top_rounding_nm : float
        Vertical extent of parabolic top-corner rounding in nm.
    """

    thickness_nm: float
    cd_bottom_nm: float
    cd_top_nm: float | None = None
    sidewall_angle_deg: float = 90.0
    delta_line: float = 2.5e-3
    beta_line: float = 4.0e-4
    delta_space: float = 0.0
    beta_space: float = 0.0
    x_center_nm: float = 0.0
    top_rounding_nm: float = 0.0

    def width_at_z(self, z_rel_nm: np.ndarray | float) -> np.ndarray:
        """Return the line width `w(z_rel)` for `0 <= z_rel <= thickness_nm`."""
        z_arr = np.asarray(z_rel_nm, dtype=np.float64)
        h = max(float(self.thickness_nm), 1e-12)
        u = np.clip(z_arr / h, 0.0, 1.0)
        if self.cd_top_nm is not None:
            w_top = float(self.cd_top_nm)
        else:
            swa_rad = np.deg2rad(float(self.sidewall_angle_deg))
            cot_swa = 0.0 if np.isclose(swa_rad, 0.5 * np.pi) else 1.0 / np.tan(swa_rad)
            w_top = max(0.0, float(self.cd_bottom_nm) - 2.0 * h * cot_swa)
        w = (1.0 - u) * float(self.cd_bottom_nm) + u * w_top
        if self.top_rounding_nm > 0.0:
            r = min(float(self.top_rounding_nm), 0.5 * h)
            in_round = z_arr > (h - r)
            if np.any(in_round):
                t = np.clip((z_arr - (h - r)) / max(r, 1e-12), 0.0, 1.0)
                w = np.where(
                    in_round, w * np.sqrt(np.maximum(1.0 - 0.35 * t * t, 0.0)), w
                )
        return np.maximum(w, 0.0)


def _smooth_box_indicator(
    x: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    pitch_nm: float,
    smoothing_nm: float,
) -> np.ndarray:
    """Evaluate a periodic smooth indicator function in `[0, 1]` for `[left_edge,
    right_edge]`.
    """
    center = 0.5 * (left_edge + right_edge)
    half_width = np.maximum(0.5 * (right_edge - left_edge), 0.0)
    # Periodic signed distance from line center within [-pitch/2, +pitch/2]
    dx = (x - center + 0.5 * pitch_nm) % pitch_nm - 0.5 * pitch_nm
    dist_from_edge = half_width - np.abs(dx)
    if smoothing_nm <= 0.0:
        return (dist_from_edge >= 0.0).astype(np.float64)
    return 0.5 * (1.0 + np.tanh(dist_from_edge / float(smoothing_nm)))


def build_grating_2d(
    pitch_nm: float = 100.0,
    cd_nm: float = 45.0,
    thickness_nm: float = 80.0,
    *,
    sidewall_angle_deg: float = 90.0,
    cd_top_nm: float | None = None,
    top_rounding_nm: float = 0.0,
    delta_line: float = 2.5e-3,
    beta_line: float = 4.0e-4,
    delta_space: float = 0.0,
    beta_space: float = 0.0,
    layers: list[GratingLayerSpec] | None = None,
    n_periods: int = 1,
    nx: int = 256,
    nz: int = 80,
    energy_ev: float = 285.0,
    edge_smoothing_nm: float = 0.4,
) -> RefractiveIndexMap2D:
    """Build a 2D line-space grating `RefractiveIndexMap2D` in `(x, z)`.

    Supports single-layer trapezoidal/rounded line profiles or arbitrary
    multi-layer stacks via `layers`.
    """
    if layers is None:
        layer_list = [
            GratingLayerSpec(
                thickness_nm=thickness_nm,
                cd_bottom_nm=cd_nm,
                cd_top_nm=cd_top_nm,
                sidewall_angle_deg=sidewall_angle_deg,
                delta_line=delta_line,
                beta_line=beta_line,
                delta_space=delta_space,
                beta_space=beta_space,
                top_rounding_nm=top_rounding_nm,
            )
        ]
    else:
        layer_list = list(layers)

    total_z = sum(float(layer.thickness_nm) for layer in layer_list)
    lx_nm = float(pitch_nm) * max(int(n_periods), 1)
    dx_nm = lx_nm / int(nx)
    dz_nm = total_z / int(nz)
    x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * dx_nm
    z_nm = (np.arange(nz, dtype=np.float64) + 0.5) * dz_nm

    delta = np.full((nx, nz), float(delta_space), dtype=np.float64)
    beta = np.full((nx, nz), float(beta_space), dtype=np.float64)

    z_cursor = 0.0
    for layer in layer_list:
        z_start = z_cursor
        z_end = z_cursor + float(layer.thickness_nm)
        z_cursor = z_end
        in_layer = (z_nm >= z_start - 1e-12) & (z_nm <= z_end + 1e-12)
        if not np.any(in_layer):
            continue
        z_rel = z_nm[in_layer] - z_start
        widths = layer.width_at_z(z_rel)  # (nz_layer,)
        left_edge = float(layer.x_center_nm) - 0.5 * widths[None, :]
        right_edge = float(layer.x_center_nm) + 0.5 * widths[None, :]
        frac = _smooth_box_indicator(
            x_nm[:, None],
            left_edge,
            right_edge,
            pitch_nm=float(pitch_nm),
            smoothing_nm=edge_smoothing_nm,
        )
        delta[:, in_layer] = float(layer.delta_space) + frac * (
            float(layer.delta_line) - float(layer.delta_space)
        )
        beta[:, in_layer] = float(layer.beta_space) + frac * (
            float(layer.beta_line) - float(layer.beta_space)
        )

    return RefractiveIndexMap2D(
        x_nm=x_nm,
        z_nm=z_nm,
        delta=delta,
        beta=beta,
        energy_ev=energy_ev,
        pitch_nm=float(pitch_nm),
        delta_ref=float(delta_space),
        beta_ref=float(beta_space),
        metadata={
            "cd_nm": float(cd_nm),
            "thickness_nm": float(total_z),
            "sidewall_angle_deg": float(sidewall_angle_deg),
            "n_periods": int(n_periods),
        },
    )


def generate_rough_edge_1d(
    ny: int,
    ly_nm: float,
    sigma_nm: float,
    correlation_length_nm: float = 30.0,
    hurst_exponent: float = 0.5,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 1D zero-mean periodic rough edge profile `dx(y)` from a Palasantzas
    PSD.

    The one-dimensional power spectral density is:
    `PSD(q_y) = C / (1 + (q_y * xi)^2)^(0.5 + H)`
    normalized so that `Var[dx(y)] = sigma_nm^2`.

    Returns
    -------
    dx_nm : np.ndarray
        Real-space edge displacement profile in nm, shape `(ny,)`.
    qy_nm_inv : np.ndarray
        Angular wavenumber grid `q_y` in nm^-1, shape `(ny,)`.
    psd_nm3 : np.ndarray
        Target 1D power spectral density `PSD(q_y)` in nm^3, shape `(ny,)`.
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    dy_nm = float(ly_nm) / int(ny)
    qy = 2.0 * np.pi * np.fft.fftfreq(int(ny), d=dy_nm)
    if sigma_nm <= 0.0 or int(ny) < 2:
        return (
            np.zeros(int(ny), dtype=np.float64),
            qy,
            np.zeros(int(ny), dtype=np.float64),
        )

    xi = max(float(correlation_length_nm), 1e-6)
    h = float(hurst_exponent)
    shape_unnorm = 1.0 / np.power(1.0 + (qy * xi) ** 2, 0.5 + h)
    shape_unnorm[0] = 0.0  # zero mean
    dq = 2.0 * np.pi / float(ly_nm)
    integral = float(np.sum(shape_unnorm) * dq / (2.0 * np.pi))
    psd = shape_unnorm * (float(sigma_nm) ** 2 / max(integral, 1e-30))

    # Construct Hermitian-symmetric Fourier coefficients with random phase
    phases = gen.uniform(0.0, 2.0 * np.pi, size=int(ny))
    amps = np.sqrt(psd * float(ly_nm)) * np.exp(1j * phases)
    amps[0] = 0.0
    dx_raw = np.fft.ifft(amps).real * (int(ny) / float(ly_nm))
    std_raw = float(np.std(dx_raw))
    if std_raw > 0.0:
        dx_nm = dx_raw * (float(sigma_nm) / std_raw)
    else:
        dx_nm = np.zeros(int(ny), dtype=np.float64)
    return dx_nm, qy, psd


def build_grating_3d(
    pitch_nm: float = 100.0,
    cd_nm: float = 45.0,
    thickness_nm: float = 80.0,
    *,
    sidewall_angle_deg: float = 90.0,
    cd_top_nm: float | None = None,
    delta_line: float = 2.5e-3,
    beta_line: float = 4.0e-4,
    delta_space: float = 0.0,
    beta_space: float = 0.0,
    ler_sigma_nm: float = 1.5,
    correlation_length_y_nm: float = 30.0,
    hurst_exponent: float = 0.5,
    roughness_correlation: float = 0.0,
    n_periods: int = 1,
    nx: int = 128,
    ny: int = 64,
    nz: int = 40,
    ly_nm: float = 640.0,
    energy_ev: float = 285.0,
    edge_smoothing_nm: float = 0.5,
    seed: int | None = 42,
) -> RefractiveIndexMap3D:
    """Build a 3D line-space grating `RefractiveIndexMap3D` with 3D LER / LWR / LPR.

    Parameters
    ----------
    pitch_nm, cd_nm, thickness_nm : float
        Nominal grating pitch, bottom critical dimension, and thickness in nm.
    ler_sigma_nm : float
        1-sigma line-edge roughness amplitude in nm on each sidewall.
    correlation_length_y_nm : float
        Along-line (`y`) roughness correlation length in nm.
    hurst_exponent : float
        Roughness Hurst exponent `H` (0.5 gives exponential covariance).
    roughness_correlation : float
        Cross-correlation coefficient `c in [-1, 1]` between the left and right
        sidewall displacements of each line:
        - `c = +1.0`: pure line-placement roughness (LPR / pitch walking / wiggle)
        - `c =  0.0`: uncorrelated left/right line-edge roughness (standard LER)
        - `c = -1.0`: pure line-width roughness (LWR / symmetric width breathing)
    n_periods : int
        Number of grating periods included along `x`.
    """
    n_per = max(int(n_periods), 1)
    lx_nm = float(pitch_nm) * n_per
    dx_nm = lx_nm / int(nx)
    dy_nm = float(ly_nm) / int(ny)
    dz_nm = float(thickness_nm) / int(nz)

    x_nm = (np.arange(nx, dtype=np.float64) - 0.5 * nx) * dx_nm
    y_nm = (np.arange(ny, dtype=np.float64) - 0.5 * ny) * dy_nm
    z_nm = (np.arange(nz, dtype=np.float64) + 0.5) * dz_nm

    layer = GratingLayerSpec(
        thickness_nm=thickness_nm,
        cd_bottom_nm=cd_nm,
        cd_top_nm=cd_top_nm,
        sidewall_angle_deg=sidewall_angle_deg,
        delta_line=delta_line,
        beta_line=beta_line,
        delta_space=delta_space,
        beta_space=beta_space,
    )
    widths_z = layer.width_at_z(z_nm)  # (nz,)

    rng = np.random.default_rng(seed)
    corr = float(np.clip(roughness_correlation, -1.0, 1.0))

    frac_total = np.zeros((nx, ny, nz), dtype=np.float64)
    left_edges_all = []
    right_edges_all = []

    for period_idx in range(n_per):
        x_center = (period_idx - 0.5 * (n_per - 1)) * float(pitch_nm)
        u1, qy, psd_1d = generate_rough_edge_1d(
            ny=ny,
            ly_nm=ly_nm,
            sigma_nm=ler_sigma_nm,
            correlation_length_nm=correlation_length_y_nm,
            hurst_exponent=hurst_exponent,
            rng=rng,
        )
        u2, _, _ = generate_rough_edge_1d(
            ny=ny,
            ly_nm=ly_nm,
            sigma_nm=ler_sigma_nm,
            correlation_length_nm=correlation_length_y_nm,
            hurst_exponent=hurst_exponent,
            rng=rng,
        )
        dx_left = u1
        dx_right = corr * u1 + np.sqrt(max(0.0, 1.0 - corr * corr)) * u2
        left_edges_all.append(dx_left)
        right_edges_all.append(dx_right)

        left_yz = x_center - 0.5 * widths_z[None, :] + dx_left[:, None]  # (ny, nz)
        right_yz = x_center + 0.5 * widths_z[None, :] + dx_right[:, None]  # (ny, nz)

        if n_per == 1:
            frac = _smooth_box_indicator(
                x_nm[:, None, None],
                left_yz[None, :, :],
                right_yz[None, :, :],
                pitch_nm=float(pitch_nm),
                smoothing_nm=edge_smoothing_nm,
            )
        else:
            center_yz = 0.5 * (left_yz + right_yz)
            half_w_yz = np.maximum(0.5 * (right_yz - left_yz), 0.0)
            dx = (
                x_nm[:, None, None] - center_yz[None, :, :] + 0.5 * lx_nm
            ) % lx_nm - 0.5 * lx_nm
            dist = half_w_yz[None, :, :] - np.abs(dx)
            if edge_smoothing_nm <= 0.0:
                frac = (dist >= 0.0).astype(np.float64)
            else:
                frac = 0.5 * (1.0 + np.tanh(dist / float(edge_smoothing_nm)))
        frac_total = np.clip(frac_total + frac, 0.0, 1.0)

    delta = float(delta_space) + frac_total * (float(delta_line) - float(delta_space))
    beta = float(beta_space) + frac_total * (float(beta_line) - float(beta_space))

    return RefractiveIndexMap3D(
        x_nm=x_nm,
        y_nm=y_nm,
        z_nm=z_nm,
        delta=delta,
        beta=beta,
        energy_ev=energy_ev,
        pitch_nm=float(pitch_nm),
        delta_ref=float(delta_space),
        beta_ref=float(beta_space),
        metadata={
            "cd_nm": float(cd_nm),
            "thickness_nm": float(thickness_nm),
            "sidewall_angle_deg": float(sidewall_angle_deg),
            "ler_sigma_nm": float(ler_sigma_nm),
            "correlation_length_y_nm": float(correlation_length_y_nm),
            "hurst_exponent": float(hurst_exponent),
            "roughness_correlation": corr,
            "n_periods": n_per,
            "left_edge_dx_nm": np.asarray(left_edges_all),
            "right_edge_dx_nm": np.asarray(right_edges_all),
            "qy_nm_inv": qy,
            "input_psd_nm3": psd_1d,
        },
    )


__all__ = [
    "GratingLayerSpec",
    "RefractiveIndexMap2D",
    "RefractiveIndexMap3D",
    "build_grating_2d",
    "build_grating_3d",
    "generate_rough_edge_1d",
]
