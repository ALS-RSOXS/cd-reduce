"""Unified detector and sample geometry for transmission CD-RSoXS / CD-SAXS.

Unifies the streak-coordinate and Laue-invariant coordinate conventions into a
single canonical representation.

Conventions
-----------
- Lab frame: incident beam along ``+z``, sample rotation axis along ``+y``
  (vertical), nominal horizontal scattering plane ``(x, z)``.
- Detector pixel ``(u, v)`` (column ``x``, row ``y``) with beam-hit position
  ``(u0_px, v0_px)`` and in-plane spot-line angle ``line_rot_deg`` (``rot``):
  ``X = (u - u0)*cos(rot) + (v - v0)*sin(rot)``,
  ``Y = -(u - u0)*sin(rot) + (v - v0)*cos(rot)``.
- Tilted flat detector rotated by ``beta_deg`` about the vertical axis:
  ``P = (X*cos(beta), Y, Lb_px - X*sin(beta))``, ``r_hat = P / |P|``.
- Sample frame at true rotation ``W = omega + omega0`` and grating azimuth ``psi``
  (plus optional out-of-plane tilt ``chi``):
  - ``Qx = q . g_hat`` (perpendicular to grating lines, in sample plane)
  - ``Qy = q . l_hat`` (parallel to grating lines, in sample plane)
  - ``Qz = q . n_hat`` (along sample surface normal)
  - ``Qr = sqrt(Qx^2 + Qy^2)`` (in-plane radial momentum transfer)
  - ``signed_Qr = sign(Qx) * sqrt(Qx^2 + Qy^2)``
  - ``Q = sqrt(Qx^2 + Qy^2 + Qz^2)`` (total momentum transfer magnitude)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cd_reduce.utils.units import HC_EV_NM, energy_to_wavelength_nm

ApproximationMode = Literal[
    "correct",
    "ewald",
    "flat",
    "small_angle",
    "sax",
    "angle_linear",
    "lin",
]


@dataclass(frozen=True, slots=True)
class OmegaRefinement:
    """Rod-verticality refinement parameters across an ``omega`` rotation series.

    Models a sample whose illuminated grating patch sits off the rotation axis
    by ``t_px`` along the sample normal and ``c0_px`` along the spot line, with
    a beamtime-shared detector distance scale ``Lb_scale``.

    Attributes
    ----------
    Lb_scale : float
        Multiplicative scale factor on ``Lb_px``.
    c0_px : float
        Constant origin offset along the spot line in pixels.
    t_px : float
        Normal offset of the illuminated grating patch from the rotation axis in pixels.
    d_omega0_deg : float
        Additional correction to ``omega0_deg`` in degrees.
    Lambda_nm : float or None
        Refined grating period in nm (if None, keeps base geometry's ``Lambda_nm``).
    """

    Lb_scale: float = 1.0
    c0_px: float = 0.0
    t_px: float = 0.0
    d_omega0_deg: float = 0.0
    Lambda_nm: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OmegaRefinement:
        """Construct ``OmegaRefinement`` from a calibration dictionary."""
        return cls(
            Lb_scale=float(data.get("Lb_scale", 1.0)),
            c0_px=float(data.get("c0_px", 0.0)),
            t_px=float(data.get("t_px", 0.0)),
            d_omega0_deg=float(data.get("d_omega0_deg", 0.0)),
            Lambda_nm=(
                float(data["Lambda_nm"])
                if "Lambda_nm" in data and data["Lambda_nm"] is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class Geometry:
    """Unified detector, sample, and grating geometry.

    All angles are stored in degrees, detector coordinates and distances in
    detector pixels, and ``Lambda_nm`` in nanometres.

    Attributes
    ----------
    u0_px : float
        Beam-hit column coordinate (x) in detector pixels.
    v0_px : float
        Beam-hit row coordinate (y) in detector pixels.
    Lb_px : float
        Sample-to-beam-hit distance in detector pixels.
    beta_deg : float
        Detector tilt about the axis perpendicular to the spot line (degrees).
    line_rot_deg : float
        In-plane angle of the diffraction spot line on the detector (degrees).
    omega0_deg : float
        Sample rotation zero offset in degrees (``W = omega + omega0_deg``).
    chi_deg : float
        Sample normal out-of-plane tilt about lab x-axis (degrees).
    psi_deg : float
        In-plane azimuthal rotation of the grating lines (degrees).
    Lambda_nm : float
        Grating pitch / period in nanometres (``G = 2 * pi / Lambda_nm``).
    pixel_um : float
        Detector pixel size in micrometres (used only when converting between
        physical SDD in cm/mm and ``Lb_px``). Default is ``27.0``.
    s_u : float
        Horizontal pixel coordinate handedness (``+1.0`` or ``-1.0``).
    walkoff_offsets_px : dict of str to float
        Optional per-frame walk-off shifts along the spot line keyed by filename.
    """

    u0_px: float = 512.0
    v0_px: float = 512.0
    Lb_px: float = 1969.0
    beta_deg: float = 0.0
    line_rot_deg: float = 0.0
    omega0_deg: float = 0.0
    chi_deg: float = 0.0
    psi_deg: float = 0.0
    Lambda_nm: float = 80.0
    pixel_um: float = 27.0
    s_u: float = 1.0
    walkoff_offsets_px: dict[str, float] | None = None

    @property
    def g_inv_nm(self) -> float:
        """Reciprocal lattice vector magnitude ``G = 2 * pi / Lambda_nm`` in 1/nm."""
        return float(2.0 * np.pi / self.Lambda_nm)

    @property
    def G_inv_nm(self) -> float:
        """Alias for ``g_inv_nm`` (``2 * pi / Lambda_nm`` in 1/nm)."""
        return self.g_inv_nm

    @property
    def sdd_cm(self) -> float:
        """Sample-to-detector distance in centimetres."""
        return float(self.Lb_px * self.pixel_um * 1e-4)

    @classmethod
    def from_sdd(
        cls,
        *,
        u0_px: float,
        v0_px: float,
        sdd_cm: float,
        pixel_um: float = 27.0,
        beta_deg: float = 0.0,
        line_rot_deg: float = 0.0,
        omega0_deg: float = 0.0,
        chi_deg: float = 0.0,
        psi_deg: float = 0.0,
        Lambda_nm: float = 80.0,
    ) -> Geometry:
        """Construct ``Geometry`` from physical sample-to-detector distance in cm."""
        if sdd_cm <= 0 or pixel_um <= 0:
            raise ValueError("sdd_cm and pixel_um must be positive")
        lb_px = sdd_cm * 1e4 / pixel_um
        return cls(
            u0_px=float(u0_px),
            v0_px=float(v0_px),
            Lb_px=float(lb_px),
            beta_deg=float(beta_deg),
            line_rot_deg=float(line_rot_deg),
            omega0_deg=float(omega0_deg),
            chi_deg=float(chi_deg),
            psi_deg=float(psi_deg),
            Lambda_nm=float(Lambda_nm),
            pixel_um=float(pixel_um),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Geometry:
        """Construct ``Geometry`` from a calibration JSON dictionary.

        Supports both ``u0_px``/``Lb_px``/``beta_deg``/``line_rot_deg`` (degree
        convention) and ``u0``/``dist_px``/``Lb``/``rot``/``om0``/``logLam``
        (radian/log convention).
        """
        u0 = float(data.get("u0_px", data.get("u0", 512.0)))
        v0 = float(data.get("v0_px", data.get("v0", 512.0)))
        lb = float(data.get("Lb_px", data.get("dist_px", data.get("Lb", 1969.0))))

        if "beta_deg" in data:
            beta_deg = float(data["beta_deg"])
        elif "beta" in data:
            beta_deg = float(np.degrees(data["beta"]))
        elif "rot2" in data:
            beta_deg = float(np.degrees(data["rot2"]))
        else:
            beta_deg = 0.0

        if "line_rot_deg" in data:
            line_rot_deg = float(data["line_rot_deg"])
        elif "rot" in data:
            line_rot_deg = float(np.degrees(data["rot"]))
        elif "rot3" in data:
            line_rot_deg = float(np.degrees(data["rot3"]))
        else:
            line_rot_deg = 0.0

        if "omega0_deg" in data:
            omega0_deg = float(data["omega0_deg"])
        elif "om0" in data:
            omega0_deg = float(np.degrees(data["om0"]))
        elif "omega0" in data:
            omega0_deg = float(np.degrees(data["omega0"]))
        else:
            omega0_deg = 0.0

        if "chi_deg" in data:
            chi_deg = float(data["chi_deg"])
        elif "chi" in data:
            chi_deg = float(np.degrees(data["chi"]))
        else:
            chi_deg = 0.0

        if "psi_deg" in data:
            psi_deg = float(data["psi_deg"])
        elif "psi" in data:
            psi_deg = float(np.degrees(data["psi"]))
        else:
            psi_deg = 0.0

        if "Lambda_nm" in data:
            lam_nm = float(data["Lambda_nm"])
        elif "logLam" in data:
            lam_nm = float(np.exp(data["logLam"]))
        elif "g_inv_nm" in data:
            lam_nm = float(2.0 * np.pi / float(data["g_inv_nm"]))
        else:
            lam_nm = 80.0

        walkoff = data.get("walkoff_offsets_px")
        walkoff_dict = (
            {str(k): float(v) for k, v in walkoff.items()}
            if isinstance(walkoff, Mapping)
            else None
        )
        return cls(
            u0_px=u0,
            v0_px=v0,
            Lb_px=lb,
            beta_deg=beta_deg,
            line_rot_deg=line_rot_deg,
            omega0_deg=omega0_deg,
            chi_deg=chi_deg,
            psi_deg=psi_deg,
            Lambda_nm=lam_nm,
            pixel_um=float(data.get("pixel_um", 27.0)),
            s_u=float(data.get("s_u", 1.0)),
            walkoff_offsets_px=walkoff_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize geometry to a JSON-compatible dictionary."""
        out: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                out[f.name] = val
        return out

    def with_origin_shift(
        self,
        delta_px: float,
        *,
        convention: str = "standard",
    ) -> Geometry:
        """Return a copy with the beam origin shifted by ``delta_px`` along the spot
        line.

        Shifting spot coordinates by ``+delta_px * (cos(rot), sin(rot))`` is
        equivalent to shifting ``(u0, v0)`` by ``-delta_px * (cos(rot), sin(rot))``
        (i.e. along ``phi = line_rot_deg + 180``).

        Parameters
        ----------
        delta_px : float
            Shift along the spot line in pixels.
        convention : str, optional
            Sign convention identifier (retained for compatibility).

        Returns
        -------
        Geometry
            Shifted geometry.
        """
        del convention
        phi = np.radians(self.line_rot_deg + 180.0)
        return replace(
            self,
            u0_px=float(self.u0_px + np.cos(phi) * delta_px),
            v0_px=float(self.v0_px + np.sin(phi) * delta_px),
        )

    def with_omega_refinement(
        self,
        refinement: OmegaRefinement | Mapping[str, Any] | None,
        omega_deg: float,
    ) -> Geometry:
        """Return per-frame geometry with rod-verticality refinement applied.

        Applies the sample-offset rotation kinematics:
        - ``W = radians(omega_deg + omega0_deg + d_omega0_deg)``
        - Origin shifts along the line by ``c0_px + t_px * sin(W)``
        - ``Lb_px = Lb_px * Lb_scale - t_px * (cos(W) - 1)``

        Parameters
        ----------
        refinement : OmegaRefinement or dict or None
            Calibration parameters. If None, returns ``self`` unchanged.
        omega_deg : float
            Nominal incidence angle of the frame in degrees.

        Returns
        -------
        Geometry
            Frame-specific refined geometry.
        """
        if refinement is None:
            return self
        cal = (
            refinement
            if isinstance(refinement, OmegaRefinement)
            else OmegaRefinement.from_dict(refinement)
        )
        w = np.radians(omega_deg + self.omega0_deg + cal.d_omega0_deg)
        shifted = self.with_origin_shift(cal.c0_px + cal.t_px * float(np.sin(w)))
        new_lb = self.Lb_px * cal.Lb_scale - cal.t_px * float(np.cos(w) - 1.0)
        return replace(
            shifted,
            Lb_px=float(new_lb),
            omega0_deg=float(self.omega0_deg + cal.d_omega0_deg),
            Lambda_nm=float(
                cal.Lambda_nm if cal.Lambda_nm is not None else self.Lambda_nm
            ),
        )

    def r1_px(self, energy_ev: float) -> float:
        """Detector radius in pixels of the first diffraction order ``|m| = 1``."""
        k = 2.0 * np.pi * energy_ev / HC_EV_NM
        return float(self.Lb_px * self.g_inv_nm / k)

    def streak_coords(
        self,
        x_px: ArrayLike,
        y_px: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Signed distance along (``a``) and across (``b``) the spot line [px]."""
        phi = np.radians(self.line_rot_deg + 180.0)
        dx = np.asarray(x_px, dtype=np.float64) - self.u0_px
        dy = np.asarray(y_px, dtype=np.float64) - self.v0_px
        a = np.cos(phi) * dx + np.sin(phi) * dy
        b = -np.sin(phi) * dx + np.cos(phi) * dy
        return a, b

    def sample_axes(
        self,
        omega_deg: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute sample-frame unit vectors ``(g_hat, l_hat, n_hat)`` in the lab frame.

        Parameters
        ----------
        omega_deg : float or array-like
            Incidence angle(s) in degrees (``0`` at nominal normal incidence).

        Returns
        -------
        g_hat, l_hat, n_hat : ndarray of shape (..., 3)
            Unit vectors along the grating periodicity direction, line axis,
            and sample surface normal.
        """
        w = np.radians(np.asarray(omega_deg, dtype=np.float64) + self.omega0_deg)
        psi = np.radians(self.psi_deg)
        chi = np.radians(self.chi_deg)

        cw, sw = np.cos(w), np.sin(w)
        cp, sp = np.cos(psi), np.sin(psi)
        cc, sc = np.cos(chi), np.sin(chi)

        # Base vectors before R_y(W) R_x(chi):
        # g0 = (cp, sp, 0), l0 = (-sp, cp, 0), n0 = (0, 0, 1)
        # After R_x(chi):
        # g1 = (cp, cc*sp, sc*sp)
        # l1 = (-sp, cc*cp, sc*cp)
        # n1 = (0, -sc, cc)
        # After R_y(W): (x' = cw*x + sw*z, y' = y, z' = -sw*x + cw*z)
        g_hat = np.stack(
            [
                cw * cp + sw * sc * sp,
                cc * sp + 0.0 * w,
                -sw * cp + cw * sc * sp,
            ],
            axis=-1,
        )
        l_hat = np.stack(
            [
                -cw * sp + sw * sc * cp,
                cc * cp + 0.0 * w,
                sw * sp + cw * sc * cp,
            ],
            axis=-1,
        )
        n_hat = np.stack(
            [
                sw * cc,
                -sc + 0.0 * w,
                cw * cc,
            ],
            axis=-1,
        )
        return g_hat, l_hat, n_hat


@dataclass(slots=True)
class PixelQMap:
    """Reciprocal-space coordinates and geometric correction maps for a frame.

    Attributes
    ----------
    Qx : ndarray of float64
        Sample-frame momentum transfer along grating vector (1/nm).
    Qy : ndarray of float64
        Sample-frame momentum transfer along grating lines (1/nm).
    Qz : ndarray of float64
        Sample-frame momentum transfer along sample normal (1/nm).
    Qr : ndarray of float64
        In-plane radial momentum transfer ``sqrt(Qx^2 + Qy^2)`` (1/nm).
    signed_Qr : ndarray of float64
        Signed radial momentum transfer ``sign(Qx) * sqrt(Qx^2 + Qy^2)`` (1/nm).
    Q : ndarray of float64
        Total momentum transfer magnitude ``sqrt(Qx^2 + Qy^2 + Qz^2)`` (1/nm).
    omega_deg : ndarray of float64
        Sample incidence angle ``omega`` at each pixel (degrees).
    two_theta_deg : ndarray of float64
        Scattering angle ``2*theta`` in the grating plane (degrees).
    solid_angle_sr : ndarray of float64
        Solid angle subtended by each (possibly binned) pixel (steradians).
    polarization_factor : ndarray of float64
        Dipole polarization factor ``1 - (r_hat . e_pol)^2`` at each pixel.
    px_x : ndarray of float64
        Detector column coordinate ``u`` (pixels).
    px_y : ndarray of float64
        Detector row coordinate ``v`` (pixels).
    """

    Qx: NDArray[np.float64]
    Qy: NDArray[np.float64]
    Qz: NDArray[np.float64]
    Qr: NDArray[np.float64]
    signed_Qr: NDArray[np.float64]
    Q: NDArray[np.float64]
    omega_deg: NDArray[np.float64]
    two_theta_deg: NDArray[np.float64]
    solid_angle_sr: NDArray[np.float64]
    polarization_factor: NDArray[np.float64]
    px_x: NDArray[np.float64]
    px_y: NDArray[np.float64]

    def get_coord(self, domain: str) -> NDArray[np.float64]:
        """Retrieve coordinate array by domain name.

        Supported domain names (case-insensitive aliases supported):
        ``"Qx"``, ``"Qy"``, ``"Qz"``, ``"Qr"``, ``"signed_Qr"``, ``"Q"``,
        ``"omega"``, ``"two_theta"``, ``"px_x"``, ``"px_y"``.
        """
        key = normalize_domain_name(domain)
        mapping: dict[str, NDArray[np.float64]] = {
            "Qx": self.Qx,
            "Qy": self.Qy,
            "Qz": self.Qz,
            "Qr": self.Qr,
            "signed_Qr": self.signed_Qr,
            "Q": self.Q,
            "omega": self.omega_deg,
            "two_theta": self.two_theta_deg,
            "px_x": self.px_x,
            "px_y": self.px_y,
        }
        return mapping[key]


def normalize_domain_name(domain: str) -> str:
    """Normalize a coordinate axis / domain identifier to its canonical name.

    Parameters
    ----------
    domain : str
        Domain name such as ``"Qx"``, ``"qx"``, ``"Qr"``, ``"signed_qr"``,
        ``"omega"``, ``"two_theta"``, ``"2theta"``, ``"Q"``, ``"px_x"``.

    Returns
    -------
    str
        Canonical domain string.
    """
    clean = domain.strip().lower().replace("-", "_")
    aliases: dict[str, str] = {
        "qx": "Qx",
        "q_x": "Qx",
        "qy": "Qy",
        "q_y": "Qy",
        "qz": "Qz",
        "q_z": "Qz",
        "qr": "Qr",
        "q_r": "Qr",
        "q_xy": "Qr",
        "qxy": "Qr",
        "signed_qr": "signed_Qr",
        "sqr": "signed_Qr",
        "q": "Q",
        "q_tot": "Q",
        "qtot": "Q",
        "q_mag": "Q",
        "omega": "omega",
        "omega_deg": "omega",
        "w": "omega",
        "aoi": "omega",
        "two_theta": "two_theta",
        "twotheta": "two_theta",
        "2theta": "two_theta",
        "tth": "two_theta",
        "px_x": "px_x",
        "x": "px_x",
        "u": "px_x",
        "px_y": "px_y",
        "y": "px_y",
        "v": "px_y",
    }
    if clean not in aliases:
        raise ValueError(
            f"Unknown coordinate domain {domain!r}. "
            f"Valid domains: {sorted(set(aliases.values()))}"
        )
    return aliases[clean]


def compute_pixel_q(
    geometry: Geometry | Mapping[str, Any],
    x_px: ArrayLike,
    y_px: ArrayLike,
    energy_ev: ArrayLike,
    omega_deg: ArrayLike,
    *,
    bin_px: int = 1,
    offset_px: float = 0.0,
    approximation: ApproximationMode = "correct",
    polarization_deg: float = 0.0,
) -> PixelQMap:
    """Map detector pixel coordinates ``(x_px, y_px)`` to sample-frame Q domains.

    Supports both full Ewald-sphere curvature on a tilted detector
    (``approximation="correct"``), the small-angle approximation
    (``approximation="small_angle"``), and angle-linear mapping
    (``approximation="angle_linear"``).

    Parameters
    ----------
    geometry : Geometry or dict
        Calibrated detector and sample geometry.
    x_px, y_px : array-like
        Detector column (``u``) and row (``v``) coordinates in full-resolution
        pixels.
    energy_ev : float or array-like
        Photon energy in eV.
    omega_deg : float or array-like
        Motor incidence angle ``omega = sample_theta + 90`` in degrees.
    bin_px : int, optional
        Pixel binning factor used to scale the per-pixel solid angle ``b^2``.
        Default is 1.
    offset_px : float, optional
        Per-frame walk-off shift along the spot line in pixels (shifts spot
        coordinates by ``+offset_px * (cos(rot), sin(rot))``).
        Default is ``0.0``.
    approximation : ApproximationMode, optional
        Coordinate reduction mode. Default is ``"correct"``.
    polarization_deg : float, optional
        Linear electric-field polarization angle in degrees (``0`` = horizontal
        lab ``+x``, ``90`` = vertical lab ``+y``). Default is ``0.0``.

    Returns
    -------
    PixelQMap
        Complete reciprocal-space coordinate and correction arrays with the
        broadcast shape of ``(x_px, y_px, energy_ev, omega_deg)``.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    u = np.asarray(x_px, dtype=np.float64)
    v = np.asarray(y_px, dtype=np.float64)
    E = np.asarray(energy_ev, dtype=np.float64)
    om = np.asarray(omega_deg, dtype=np.float64)

    u_b, v_b, E_b, om_b = np.broadcast_arrays(u, v, E, om)
    rot = np.radians(geo.line_rot_deg)

    u_eff = u_b + offset_px * np.cos(rot)
    v_eff = v_b + offset_px * np.sin(rot)
    du = geo.s_u * (u_eff - geo.u0_px)
    dv = v_eff - geo.v0_px

    X = np.cos(rot) * du + np.sin(rot) * dv
    Y = -np.sin(rot) * du + np.cos(rot) * dv

    beta = np.radians(geo.beta_deg)
    lb = geo.Lb_px
    k = 2.0 * np.pi / np.asarray(energy_to_wavelength_nm(E_b), dtype=np.float64)

    mode = approximation.lower().strip()
    if mode in {"correct", "ewald", "flat"}:
        px_lab = X * np.cos(beta)
        py_lab = Y
        pz_lab = lb - X * np.sin(beta)
        P = np.stack([px_lab, py_lab, pz_lab], axis=-1)
        R = np.linalg.norm(P, axis=-1)
        r_hat = P / R[..., None]
        q_lab = k[..., None] * (r_hat - np.array([0.0, 0.0, 1.0]))
        nd = np.array([np.sin(beta), 0.0, np.cos(beta)])
        solid_angle = (bin_px**2) * np.abs(np.sum(r_hat * nd, axis=-1)) / (R**2)
        two_theta_deg = np.degrees(np.arctan2(px_lab, pz_lab))
    elif mode in {"small_angle", "sax", "paraxial"}:
        # Small-angle flat-detector approximation:
        # Neglects Ewald curvature (q_z,lab = 0) and detector tilt foreshortening.
        tx = X / lb
        ty = Y / lb
        q_lab = k[..., None] * np.stack([tx, ty, np.zeros_like(tx)], axis=-1)
        r_norm = np.sqrt(1.0 + tx**2 + ty**2)
        r_hat = np.stack([tx / r_norm, ty / r_norm, 1.0 / r_norm], axis=-1)
        solid_angle = np.full_like(X, (bin_px / lb) ** 2, dtype=np.float64)
        two_theta_deg = np.degrees(tx)
    elif mode in {"angle_linear", "lin", "spherical"}:
        rho = np.hypot(X, Y)
        alpha = np.arctan2(Y, X)
        phi = rho / lb
        r_hat = np.stack(
            [np.sin(phi) * np.cos(alpha), np.sin(phi) * np.sin(alpha), np.cos(phi)],
            axis=-1,
        )
        q_lab = k[..., None] * (r_hat - np.array([0.0, 0.0, 1.0]))
        solid_angle = np.full_like(X, (bin_px / lb) ** 2, dtype=np.float64)
        two_theta_deg = np.degrees(np.arctan2(r_hat[..., 0], r_hat[..., 2]))
    else:
        raise ValueError(
            f"Unknown approximation mode {approximation!r}. "
            "Use 'correct', 'small_angle', or 'angle_linear'."
        )

    g_hat, l_hat, n_hat = geo.sample_axes(om_b)
    Qx = np.sum(q_lab * g_hat, axis=-1)
    Qy = np.sum(q_lab * l_hat, axis=-1)
    Qz = np.sum(q_lab * n_hat, axis=-1)

    Qr = np.hypot(Qx, Qy)
    sgn = np.where(Qx >= 0.0, 1.0, -1.0)
    signed_Qr = sgn * Qr
    Q_tot = np.sqrt(Qx**2 + Qy**2 + Qz**2)

    pol_rad = np.radians(polarization_deg)
    e_pol = np.array([np.cos(pol_rad), np.sin(pol_rad), 0.0])
    pol_factor = np.clip(1.0 - np.sum(r_hat * e_pol, axis=-1) ** 2, 1e-6, 1.0)

    return PixelQMap(
        Qx=Qx,
        Qy=Qy,
        Qz=Qz,
        Qr=Qr,
        signed_Qr=signed_Qr,
        Q=Q_tot,
        omega_deg=om_b.astype(np.float64),
        two_theta_deg=two_theta_deg,
        solid_angle_sr=solid_angle,
        polarization_factor=pol_factor,
        px_x=u_b.astype(np.float64),
        px_y=v_b.astype(np.float64),
    )


def pixel_q(
    geometry: Geometry | Mapping[str, Any],
    x_px: ArrayLike,
    y_px: ArrayLike,
    energy_ev: ArrayLike,
    omega_deg: ArrayLike,
    *,
    approximation: ApproximationMode = "correct",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return ``(Qx, Qy, Qz, solid_angle_sr)`` for detector pixels ``(x_px, y_px)``."""
    qmap = compute_pixel_q(
        geometry,
        x_px,
        y_px,
        energy_ev,
        omega_deg,
        approximation=approximation,
    )
    return qmap.Qx, qmap.Qy, qmap.Qz, qmap.solid_angle_sr


def frame_pixel_grid(
    shape: tuple[int, int],
    bin_px: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build 2D detector pixel-centre grids ``(xx, yy)`` for a frame of ``shape``.

    When ``bin_px > 1``, returns the full-resolution pixel coordinates of the
    centres of the ``bin_px x bin_px`` blocks: ``(u + 0.5) * bin_px - 0.5``.

    Parameters
    ----------
    shape : tuple of (int, int)
        Full-resolution image shape ``(n_row, n_col)``.
    bin_px : int, optional
        Block binning factor. Default is 1.

    Returns
    -------
    xx, yy : ndarray of float64
        Column and row coordinate grids of shape ``(n_row // bin_px, n_col // bin_px)``.
    """
    n0, n1 = shape[0] // bin_px, shape[1] // bin_px
    vv, uu = np.mgrid[0:n0, 0:n1].astype(np.float64)
    if bin_px > 1:
        uu = (uu + 0.5) * bin_px - 0.5
        vv = (vv + 0.5) * bin_px - 0.5
    return uu, vv


def ewald_trace_qxqz(
    geometry: Geometry | Mapping[str, Any],
    omega_deg: float,
    energy_ev: float,
    *,
    shape: tuple[int, int] = (1024, 1024),
    approximation: ApproximationMode = "correct",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the ``Qy = 0`` Ewald-sphere arc in ``(Qx, Qz)`` across the detector.

    Parameters
    ----------
    geometry : Geometry or dict
        Detector and sample geometry.
    omega_deg : float
        Incidence angle in degrees.
    energy_ev : float
        Photon energy in eV.
    shape : tuple of (int, int), optional
        Detector dimensions ``(n_row, n_col)``. Default is ``(1024, 1024)``.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.

    Returns
    -------
    qx_trace, qz_trace : 1D ndarray of float64
        ``(Qx, Qz)`` coordinates along the ``Qy = 0`` locus on the detector.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    n_pts = 512
    diag = float(np.hypot(shape[0], shape[1])) * 0.5
    s = np.linspace(-diag, diag, n_pts)
    rot = np.radians(geo.line_rot_deg)
    u = geo.u0_px + s * np.cos(rot)
    v = geo.v0_px + s * np.sin(rot)
    inside = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    qmap = compute_pixel_q(
        geo,
        u[inside],
        v[inside],
        energy_ev,
        omega_deg,
        approximation=approximation,
    )
    order_idx = np.argsort(qmap.Qx)
    return qmap.Qx[order_idx], qmap.Qz[order_idx]
