"""2D reciprocal-space mapping and 1D specular/off-specular reduction for CD-RSoXS.

Supports:
1. Converting a series of CCD frames into 2D maps ``I(x_domain, y_domain)`` for
   any combination of coordinate domains:
   - ``Qx``, ``Qy``, ``Qz`` (Cartesian sample-frame momentum transfer)
   - ``Qr = sqrt(Qx^2 + Qy^2)`` and ``signed_Qr = sign(Qx) * Qr`` (radial in-plane Q)
   - ``Q = sqrt(Qx^2 + Qy^2 + Qz^2)`` (total momentum transfer magnitude)
   - ``omega`` (sample rotation / incidence angle in degrees)
   - ``two_theta`` (detector scattering angle in degrees)
   - ``px_x``, ``px_y`` (detector pixel coordinates)
2. Choosing fixed-value slices or cuts along any orthogonal coordinate:
   - Fixed ``Qz`` when building a ``(Qx, Qy)`` or ``(Qr, Qy)`` map
   - Fixed ``Qx`` (e.g. ``Qx = m * G``) when building a ``(Qy, Qz)`` map
   - Fixed ``omega`` rather than fixed ``Qz`` (single-frame or narrow-omega slice)
   - Arbitrary ``qy_cut``, ``qx_cut``, ``qz_cut``, ``qr_cut``, or ``fixed_domain``
3. Geometric corrections (solid angle, exposure, beam walk-off, Si3N4 membrane
   window transmission, polarization) and small-angle vs exact Ewald reduction
   (``approximation="correct" | "small_angle" | "angle_linear"``).
4. Background subtraction modes (``"none"``, ``"corner"``, ``"row_col"``,
   ``"offline_band"``, ``"strip_fit"``, ``"photon_events"``).
5. Specular and off-specular 1D extraction:
   - ``extract_order_rods_qz``: ``I_m(Qz)`` for every diffraction order ``m``
   - ``extract_qx_by_omega``: ``I(Qx)`` for every measured tilt angle ``omega``
   - ``extract_diffuse_cut``: off-specular diffuse scattering along ``Qy``, ``Qz``,
     or ``Qx``
   - ``extract_offspec_rod_map``: normalized 2D off-specular sheet
     ``(dI / d^2Q) / I_m(Qy, Qz)``
   - ``readoff_edge_psd_born`` and ``extract_rod_localized_diffuse_qy`` for Born
     line-edge roughness PSD extraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from cd_reduce.reduction.corrections import (
    DEFAULT_ADU_PER_PHOTON,
    apply_geometric_corrections,
    build_beamstop_mask,
    membrane_transmission,
    parasitic_mask,
    process_frame_strip,
    subtract_corner_offset,
    subtract_offline_band_background,
    subtract_row_col_offset,
    threshold_photon_events,
    walkoff_factor,
)
from cd_reduce.reduction.frames import Frame, load_frames
from cd_reduce.reduction.geometry import (
    ApproximationMode,
    Geometry,
    OmegaRefinement,
    PixelQMap,
    compute_pixel_q,
    ewald_trace_qxqz,
    frame_pixel_grid,
    normalize_domain_name,
)
from cd_reduce.utils.image import bin_image_2d

BackgroundMode = Literal[
    "none",
    "corner",
    "row_col",
    "offline_band",
    "strip_fit",
    "photon_events",
]

ORDER_PALETTE: dict[int, str] = {
    1: "#1f77b4",
    2: "#d62728",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#ff7f0e",
    6: "#17becf",
}


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.03,
    color: str = "#18181b",
    fontsize: float = 11.0,
) -> None:
    """Place a bold lower-case reference label outside the top-left axes spine."""
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=fontsize,
        color=color,
        ha="right",
        va="bottom",
        clip_on=False,
    )


def domain_axis_label(domain: str) -> str:
    """Return a formatted matplotlib axis label for a coordinate domain."""
    key = normalize_domain_name(domain)
    labels: dict[str, str] = {
        "Qx": r"$Q_x$ (nm$^{-1}$)",
        "Qy": r"$Q_y$ (nm$^{-1}$)",
        "Qz": r"$Q_z$ (nm$^{-1}$)",
        "Qr": r"$Q_r$ (nm$^{-1}$)",
        "signed_Qr": r"$\mathrm{sign}(Q_x)\,Q_r$ (nm$^{-1}$)",
        "Q": r"$|Q|$ (nm$^{-1}$)",
        "omega": r"$\omega$ (deg)",
        "two_theta": r"$2\theta$ (deg)",
        "px_x": r"$u$ (px)",
        "px_y": r"$v$ (px)",
    }
    return labels.get(key, key)


@dataclass(slots=True)
class Trace1D:
    """1D reduced scattering intensity trace ``I`` vs a chosen coordinate domain.

    Attributes
    ----------
    coords : ndarray of float64
        Bin centers (or discrete order/angle coordinates) along ``domain``.
    edges : ndarray of float64
        Bin edges of length ``len(coords) + 1``.
    intensity : ndarray of float64
        Mean (or integrated) intensity at each point (``NaN`` where unsampled).
    sigma : ndarray of float64
        Estimated standard error of ``intensity``.
    counts : ndarray of float64
        Number of contributing detector pixels (or frames) per point.
    domain : str
        Canonical domain name (e.g. ``"Qx"``, ``"Qy"``, ``"Qz"``, ``"Qr"``,
        ``"signed_Qr"``, ``"Q"``, ``"omega"``, ``"two_theta"``).
    cuts : dict of str to tuple of (float, float)
        Orthogonal coordinate cuts applied during reduction.
    metadata : dict of str to Any
        Additional provenance metadata.
    """

    coords: NDArray[np.float64]
    edges: NDArray[np.float64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64]
    counts: NDArray[np.float64]
    domain: str
    cuts: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the 1D trace as a tidy ``pandas.DataFrame``."""
        return pd.DataFrame(
            {
                self.domain: self.coords,
                "intensity": self.intensity,
                "sigma": self.sigma,
                "counts": self.counts,
            }
        )

    def save_csv(self, path: Path | str) -> Path:
        """Save the 1D trace to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(p, index=False)
        return p

    def plot(
        self,
        ax: Axes | None = None,
        *,
        log_y: bool = True,
        show_errors: bool = False,
        label: str | None = None,
        color: str | None = None,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the 1D intensity trace."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
        else:
            fig = cast(Figure, ax.get_figure())
        valid = np.isfinite(self.intensity)
        if log_y:
            valid = valid & (self.intensity > 0)
        x = self.coords[valid]
        y = self.intensity[valid]
        e = self.sigma[valid]
        if show_errors and np.any(np.isfinite(e)):
            ax.errorbar(
                x, y, yerr=e, fmt="o-", ms=3.5, lw=1.2, label=label, color=color
            )
        else:
            ax.plot(x, y, "-", lw=1.4, label=label, color=color)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(domain_axis_label(self.domain))
        ax.set_ylabel("Intensity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title:
            ax.set_title(title)
        if label:
            ax.legend(frameon=False)
        return fig, ax


@dataclass(slots=True)
class QMap2D:
    """2D accumulated reciprocal-space histogram map ``I(x_domain, y_domain)``.

    Stores both the weighted intensity sum ``H`` and pixel count ``N`` per bin so
    that normalized Gaussian smoothing and Ewald-arc dilation can be applied without
    distorting raw bins.

    Attributes
    ----------
    H : ndarray of float64
        2D histogram of summed intensity density, shape ``(nx, ny)``.
    N : ndarray of float64
        2D histogram of contributing pixel counts, shape ``(nx, ny)``.
    x_edges : ndarray of float64
        Bin edges along ``x_domain``, length ``nx + 1``.
    y_edges : ndarray of float64
        Bin edges along ``y_domain``, length ``ny + 1``.
    x_domain : str
        Canonical horizontal axis domain (e.g. ``"Qx"``, ``"Qr"``, ``"signed_Qr"``).
    y_domain : str
        Canonical vertical axis domain (e.g. ``"Qz"``, ``"Qy"``, ``"omega"``).
    cuts : dict of str to tuple of (float, float)
        Orthogonal slice bounds applied when accumulating pixels.
    approximation : str
        Coordinate reduction mode used (``"correct"``, ``"small_angle"``,
        or ``"angle_linear"``).
    bg_mode : str
        Background subtraction mode applied.
    metadata : dict of str to Any
        Provenance metadata (e.g. ``Lambda_nm``, ``energy_ev``, ``omega_deg``).
    """

    H: NDArray[np.float64]
    N: NDArray[np.float64]
    x_edges: NDArray[np.float64]
    y_edges: NDArray[np.float64]
    x_domain: str
    y_domain: str
    cuts: dict[str, tuple[float, float]] = field(default_factory=dict)
    approximation: str = "correct"
    bg_mode: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x_centers(self) -> NDArray[np.float64]:
        """Bin centers along ``x_domain``."""
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self) -> NDArray[np.float64]:
        """Bin centers along ``y_domain``."""
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    @property
    def mean(self) -> NDArray[np.float64]:
        """Unsmoothed mean intensity ``H / N`` per bin (``NaN`` where ``N == 0``)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.N > 0, self.H / np.maximum(self.N, 1.0), np.nan)

    def smoothed(
        self,
        sigma_bins: tuple[float, float] | None = (1.2, 1.8),
        *,
        dilate_bins: tuple[int, int] | None = None,
    ) -> NDArray[np.float64]:
        """Return display-ready 2D intensity map with normalized Gaussian smoothing or
        dilation.

        - When ``sigma_bins`` is given, convolves ``H`` and ``N`` separately with a
          Gaussian filter before dividing, masking bins far from sampled data.
        - When ``dilate_bins`` is given, widens single-bin Ewald arcs with a maximum
          filter of size ``dilate_bins``.
        """
        if dilate_bins is not None:
            m = np.where(self.N > 0, self.H / np.maximum(self.N, 1.0), 0.0)
            md = ndimage.maximum_filter(m, size=dilate_bins)
            sampled = (
                ndimage.maximum_filter((self.N > 0).astype(np.uint8), size=dilate_bins)
                > 0
            )
            return np.where(sampled & (md > 0), md, np.nan)

        if sigma_bins is None or (sigma_bins[0] <= 0 and sigma_bins[1] <= 0):
            return self.mean

        Hs = ndimage.gaussian_filter(self.H.astype(np.float64), sigma_bins)
        Ns = ndimage.gaussian_filter(self.N.astype(np.float64), sigma_bins)
        pos_n = self.N[self.N > 0]
        if pos_n.size == 0:
            return np.full_like(self.H, np.nan, dtype=np.float64)
        med_n = float(np.median(pos_n))
        norm_env = ndimage.gaussian_filter(
            np.ones_like(self.N, dtype=np.float64), sigma_bins
        )
        valid = Ns > 0.05 * norm_env * med_n
        return np.where(valid, Hs / np.maximum(Ns, 1e-12), np.nan)

    def rod_tilts_mrad(
        self,
        g_inv_nm: float | None = None,
        *,
        orders: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
        sigma_bins: tuple[float, float] = (1.5, 1.5),
    ) -> dict[int, float]:
        """Measure per-order Bragg rod slope ``dQx / dQz`` [mrad] from row centroids."""
        G = g_inv_nm or float(2.0 * np.pi / float(self.metadata.get("Lambda_nm", 80.0)))
        M = self.smoothed(sigma_bins=sigma_bins)
        qx = self.x_centers
        qz = self.y_centers
        out: dict[int, float] = {}
        for m in orders:
            sel = np.abs(qx - m * G) < 0.2 * G
            if not np.any(sel):
                continue
            P, x_sub = M[sel], qx[sel]
            zs: list[float] = []
            cs: list[float] = []
            ws: list[float] = []
            for j in range(P.shape[1]):
                p = P[:, j]
                if np.isfinite(p).sum() < 0.5 * len(p):
                    continue
                p_clean = np.nan_to_num(p, nan=0.0) - float(np.nanmedian(p))
                p_clean[p_clean < 0] = 0.0
                s_val = float(p_clean.sum())
                if s_val > 0:
                    zs.append(float(qz[j]))
                    cs.append(float((p_clean * x_sub).sum() / s_val))
                    ws.append(s_val)
            if len(zs) >= 3 and np.ptp(zs) > 0:
                slope = float(np.polyfit(zs, cs, 1, w=np.sqrt(ws))[0])
                out[int(m)] = 1e3 * slope
        return out

    def slice_1d(
        self,
        domain: str | None = None,
        *,
        cut_range: tuple[float, float] | None = None,
        fixed_value: float | None = None,
        fixed_tol: float | None = None,
    ) -> Trace1D:
        """Project this 2D map onto a 1D trace along either ``x_domain`` or
        ``y_domain``.
        """
        target = normalize_domain_name(domain) if domain is not None else self.x_domain
        if fixed_value is not None and cut_range is None:
            tol = (
                fixed_tol
                if fixed_tol is not None
                else 0.05
                * float(
                    np.ptp(self.y_edges if target == self.x_domain else self.x_edges)
                )
            )
            cut_range = (fixed_value - tol, fixed_value + tol)

        if target == self.x_domain:
            ortho_name = self.y_domain
            ortho_c = self.y_centers
            sel = (
                (ortho_c >= cut_range[0]) & (ortho_c <= cut_range[1])
                if cut_range is not None
                else np.ones(len(ortho_c), dtype=bool)
            )
            h_1d = self.H[:, sel].sum(axis=1)
            n_1d = self.N[:, sel].sum(axis=1)
            coords = self.x_centers
            edges = self.x_edges
        elif target == self.y_domain:
            ortho_name = self.x_domain
            ortho_c = self.x_centers
            sel = (
                (ortho_c >= cut_range[0]) & (ortho_c <= cut_range[1])
                if cut_range is not None
                else np.ones(len(ortho_c), dtype=bool)
            )
            h_1d = self.H[sel, :].sum(axis=0)
            n_1d = self.N[sel, :].sum(axis=0)
            coords = self.y_centers
            edges = self.y_edges
        else:
            raise ValueError(
                f"domain {target!r} does not match map axes "
                f"({self.x_domain!r}, {self.y_domain!r})"
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_1d = np.where(n_1d > 0, h_1d / np.maximum(n_1d, 1.0), np.nan)
            sigma_1d = np.where(
                n_1d > 0, np.abs(mean_1d) / np.sqrt(np.maximum(n_1d, 1.0)), np.nan
            )
        cuts = dict(self.cuts)
        if cut_range is not None:
            cuts[ortho_name] = cut_range
        return Trace1D(
            coords=coords,
            edges=edges,
            intensity=mean_1d,
            sigma=sigma_1d,
            counts=n_1d,
            domain=target,
            cuts=cuts,
            metadata=dict(self.metadata),
        )

    def save_npz(self, path: Path | str) -> Path:
        """Save the 2D map to a compressed ``.npz`` archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "H": self.H.astype(np.float32),
            "N": self.N.astype(np.float32),
            "sum_bgsub": self.H.astype(np.float32),
            "npix": self.N.astype(np.float32),
            "x_edges": self.x_edges,
            "y_edges": self.y_edges,
            "x_domain": self.x_domain,
            "y_domain": self.y_domain,
            "approximation": self.approximation,
            "bg_mode": self.bg_mode,
        }
        if self.x_domain == "Qx":
            payload["qx_edges"] = self.x_edges
        if self.y_domain == "Qz":
            payload["qz_edges"] = self.y_edges
        for k, v in self.metadata.items():
            if isinstance(v, (int, float, str, np.ndarray, list, tuple)):
                payload[k] = np.asarray(v) if isinstance(v, (list, tuple)) else v
        np.savez_compressed(p, **payload)
        return p

    @classmethod
    def load_npz(cls, path: Path | str) -> QMap2D:
        """Load a ``QMap2D`` from a saved ``.npz`` file."""
        z = np.load(Path(path), allow_pickle=False)
        H = np.asarray(z["H"] if "H" in z else z["sum_bgsub"], dtype=np.float64)
        N = np.asarray(z["N"] if "N" in z else z["npix"], dtype=np.float64)
        x_edges = np.asarray(
            z["x_edges"] if "x_edges" in z else z["qx_edges"], dtype=np.float64
        )
        y_edges = np.asarray(
            z["y_edges"] if "y_edges" in z else z["qz_edges"], dtype=np.float64
        )
        x_dom = str(z["x_domain"]) if "x_domain" in z else "Qx"
        y_dom = str(z["y_domain"]) if "y_domain" in z else "Qz"
        approx = str(z["approximation"]) if "approximation" in z else "correct"
        bg = str(z["bg_mode"]) if "bg_mode" in z else "none"
        meta = {
            k: z[k].item() if z[k].ndim == 0 else z[k]
            for k in z.files
            if k
            not in {
                "H",
                "N",
                "sum_bgsub",
                "npix",
                "x_edges",
                "y_edges",
                "qx_edges",
                "qz_edges",
                "x_domain",
                "y_domain",
                "approximation",
                "bg_mode",
            }
        }
        return cls(
            H=H,
            N=N,
            x_edges=x_edges,
            y_edges=y_edges,
            x_domain=x_dom,
            y_domain=y_dom,
            approximation=approx,
            bg_mode=bg,
            metadata=meta,
        )

    def plot(
        self,
        ax: Axes | None = None,
        *,
        sigma_bins: tuple[float, float] | None = (1.2, 1.8),
        dilate_bins: tuple[int, int] | None = None,
        dynamic_range_decades: float = 4.0,
        cmap: str = "turbo",
        show_order_ticks: bool = True,
        title: str | None = None,
        dark_background: bool = True,
        colorbar: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plot the 2D reciprocal-space map on a logarithmic spectral color scale."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
        else:
            fig = cast(Figure, ax.get_figure())

        m = self.smoothed(sigma_bins=sigma_bins, dilate_bins=dilate_bins)
        pos = m[np.isfinite(m) & (m > 0)]
        if pos.size > 0:
            vmax = float(np.percentile(pos, 99.9))
            vmin = max(vmax * (10.0 ** (-dynamic_range_decades)), float(pos.min()))
            if vmin >= vmax:
                vmin = 0.1 * vmax
        else:
            vmin, vmax = 1e-3, 1.0

        cm = plt.get_cmap(cmap)
        if dark_background:
            cm = cm.with_extremes(bad="black", under="black")
            ax.set_facecolor("black")

        display_m = np.where(np.isfinite(m) & (m > 0), m, np.nan)
        pc = ax.pcolormesh(
            self.x_edges,
            self.y_edges,
            display_m.T,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cm,
            shading="flat",
            rasterized=True,
        )
        if show_order_ticks and self.x_domain in {"Qx", "signed_Qr"}:
            lam = self.metadata.get("Lambda_nm")
            if lam is not None and float(lam) > 0:
                G = 2.0 * np.pi / float(lam)
                n_max = int(
                    np.floor(max(abs(self.x_edges[0]), abs(self.x_edges[-1])) / G)
                )
                for m_ord in range(-n_max, n_max + 1):
                    if m_ord != 0:
                        ax.axvline(m_ord * G, color="white", lw=0.4, ls=":", alpha=0.4)
        ax.set_xlim(float(self.x_edges[0]), float(self.x_edges[-1]))
        ax.set_ylim(float(self.y_edges[0]), float(self.y_edges[-1]))
        ax.set_xlabel(domain_axis_label(self.x_domain))
        ax.set_ylabel(domain_axis_label(self.y_domain))
        if title:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(pc, ax=ax, pad=0.015, fraction=0.03, label="Intensity")
        return fig, ax


def _resolve_cuts(
    *,
    g_inv_nm: float,
    fixed_domain: str | None = None,
    fixed_value: float | None = None,
    fixed_tol: float | None = None,
    fixed_range: tuple[float, float] | None = None,
    qx_cut: tuple[float, float] | None = None,
    qy_cut: tuple[float, float] | float | None = None,
    qz_cut: tuple[float, float] | None = None,
    qr_cut: tuple[float, float] | None = None,
    omega_cut: tuple[float, float] | float | None = None,
    qy_cut_over_g: float | None = None,
    cuts: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Consolidate user-specified orthogonal slice constraints into a canonical dict."""
    resolved: dict[str, tuple[float, float]] = {}
    if cuts is not None:
        for k, v in cuts.items():
            resolved[normalize_domain_name(k)] = (float(v[0]), float(v[1]))

    if fixed_domain is not None:
        dom = normalize_domain_name(fixed_domain)
        if fixed_range is not None:
            resolved[dom] = (float(fixed_range[0]), float(fixed_range[1]))
        elif fixed_value is not None:
            tol = (
                float(fixed_tol)
                if fixed_tol is not None
                else (0.5 if dom == "omega" else 0.01)
            )
            resolved[dom] = (float(fixed_value - tol), float(fixed_value + tol))

    if qx_cut is not None:
        resolved["Qx"] = (float(qx_cut[0]), float(qx_cut[1]))
    if qy_cut is not None:
        if isinstance(qy_cut, (int, float)):
            resolved["Qy"] = (-abs(float(qy_cut)), abs(float(qy_cut)))
        else:
            resolved["Qy"] = (float(qy_cut[0]), float(qy_cut[1]))
    elif qy_cut_over_g is not None:
        half = abs(float(qy_cut_over_g)) * g_inv_nm
        resolved["Qy"] = (-half, half)
    if qz_cut is not None:
        resolved["Qz"] = (float(qz_cut[0]), float(qz_cut[1]))
    if qr_cut is not None:
        resolved["Qr"] = (float(qr_cut[0]), float(qr_cut[1]))
    if omega_cut is not None:
        if isinstance(omega_cut, (int, float)):
            resolved["omega"] = (float(omega_cut) - 0.25, float(omega_cut) + 0.25)
        else:
            resolved["omega"] = (float(omega_cut[0]), float(omega_cut[1]))
    return resolved


def prepare_frame(
    frame: Frame,
    geometry: Geometry,
    *,
    bg_mode: BackgroundMode = "corner",
    bin_px: int = 1,
    approximation: ApproximationMode = "correct",
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    beamstop_mask: NDArray[np.bool_] | None = None,
    exposure_normalization: bool = True,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
) -> tuple[NDArray[np.float64], PixelQMap, float, Geometry]:
    """Background-subtract one frame and map its pixels to Q.

    This is the single per-frame cleaning path shared by :func:`map_2d`,
    :func:`reduce_1d` and the roughness extraction in
    :mod:`cd_reduce.reduction.roughness`.

    Returns
    -------
    cleaned_adu : ndarray
        Background-subtracted counts (ADU, not yet exposure / solid-angle
        normalized; ``NaN`` where masked).
    qmap : PixelQMap
        Pixel Q coordinates for ``cleaned_adu`` (same shape).
    origin_shift_px : float
        Origin shift (strip fit) or walk-off offset applied, in pixels.
    frame_geometry : Geometry
        Geometry after the per-omega refinement.
    """
    geo = geometry
    om = frame.omega_deg
    E_val = frame.energy_ev
    exp_s = frame.exposure_s if exposure_normalization else 1.0
    gf = geo.with_omega_refinement(omega_refinement, om)
    walk_off_px = (gf.walkoff_offsets_px or {}).get(frame.name, 0.0)
    bs_mask = beamstop_mask

    if bg_mode == "strip_fit" and bin_px == 1:
        strip_res = process_frame_strip(
            gf,
            frame.image,
            E_val,
            om,
            mask=bs_mask,
            exposure_s=exp_s,
            adu_per_photon=adu_per_photon,
        )
        qm = (
            strip_res.qmap
            if approximation in {"correct", "ewald", "flat"}
            else compute_pixel_q(
                gf.with_origin_shift(strip_res.origin_shift_px),
                *frame_pixel_grid(frame.image.shape, 1),
                E_val,
                om,
                approximation=approximation,
                polarization_deg=frame.polarization_deg,
            )
        )
        return strip_res.cleaned_adu, qm, float(strip_res.origin_shift_px), gf

    raw_im = frame.image.copy()
    if bs_mask is not None and bs_mask.shape == raw_im.shape:
        raw_im = np.where(bs_mask, np.nan, raw_im)

    if bg_mode in {"corner", "offline_band"}:
        raw_im, _, _ = subtract_corner_offset(raw_im)
    elif bg_mode == "photon_events":
        raw_im, _, r_sig = subtract_corner_offset(raw_im, use_clipped_mean=True)
        raw_im, _ = threshold_photon_events(
            raw_im,
            E_val,
            read_sigma_adu=max(r_sig, 1.0),
            adu_per_photon=adu_per_photon,
        )
    elif bg_mode == "row_col":
        xx_full, yy_full = frame_pixel_grid(raw_im.shape, 1)
        qm_full = compute_pixel_q(
            gf,
            xx_full,
            yy_full,
            E_val,
            om,
            offset_px=walk_off_px,
            approximation=approximation,
        )
        raw_im = subtract_row_col_offset(raw_im, gf, qm_full.Qy, bs_mask)

    img_clean = (
        bin_image_2d(raw_im, bin_px=bin_px, mode="sum") if bin_px > 1 else raw_im
    )
    xx, yy = frame_pixel_grid(frame.image.shape, bin_px=bin_px)
    qm = compute_pixel_q(
        gf,
        xx,
        yy,
        E_val,
        om,
        bin_px=bin_px,
        offset_px=walk_off_px,
        approximation=approximation,
        polarization_deg=frame.polarization_deg,
    )
    return img_clean, qm, float(walk_off_px), gf


def map_2d(
    source: Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    *,
    x_domain: str = "Qx",
    y_domain: str = "Qz",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    bins: tuple[int, int] = (300, 200),
    bin_px: int = 1,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "corner",
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    solid_angle_correction: bool = True,
    exposure_normalization: bool = True,
    polarization_correction: bool = False,
    apply_membrane: bool = False,
    membrane_thickness_nm: float = 50.0,
    walkoff_params: tuple[float, float] | None = None,
    walkoff_f_min: float = 0.5,
    mask_parasitic: bool = False,
    beamstop_mask: NDArray[np.bool_] | None = None,
    auto_beamstop_mask: bool = False,
    deduplicate_omega: bool = True,
    energy_ev: float | None = None,
    fixed_domain: str | None = None,
    fixed_value: float | None = None,
    fixed_tol: float | None = None,
    fixed_range: tuple[float, float] | None = None,
    qx_cut: tuple[float, float] | None = None,
    qy_cut: tuple[float, float] | float | None = None,
    qz_cut: tuple[float, float] | None = None,
    qr_cut: tuple[float, float] | None = None,
    omega_cut: tuple[float, float] | float | None = None,
    qy_cut_over_g: float | None = None,
    cuts: Mapping[str, tuple[float, float]] | None = None,
    qy_off_g: tuple[float, float] = (0.15, 0.40),
) -> QMap2D:
    """Convert a series of CCD frames into a 2D map ``I(x_domain, y_domain)``."""
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    x_dom = normalize_domain_name(x_domain)
    y_dom = normalize_domain_name(y_domain)
    G = geo.g_inv_nm

    resolved_cuts = _resolve_cuts(
        g_inv_nm=G,
        fixed_domain=fixed_domain,
        fixed_value=fixed_value,
        fixed_tol=fixed_tol,
        fixed_range=fixed_range,
        qx_cut=qx_cut,
        qy_cut=qy_cut,
        qz_cut=qz_cut,
        qr_cut=qr_cut,
        omega_cut=omega_cut,
        qy_cut_over_g=qy_cut_over_g,
        cuts=cuts,
    )
    if (
        "Qy" not in resolved_cuts
        and x_dom in {"Qx", "Qr", "signed_Qr"}
        and y_dom in {"Qz", "omega"}
    ):
        resolved_cuts["Qy"] = (-0.05 * G, 0.05 * G)

    frames_list = load_frames(
        source, deduplicate_omega=deduplicate_omega, energy_ev=energy_ev
    )
    if not frames_list:
        raise ValueError("No frames matched the selection criteria for map_2d.")

    bs_mask = beamstop_mask
    if bs_mask is None and auto_beamstop_mask:
        bs_mask = build_beamstop_mask(frames_list, geo)

    prepared: list[tuple[NDArray[np.float64], Any, float, float]] = []
    used_omegas: list[float] = []
    origin_shifts: list[float] = []

    for fr in frames_list:
        om = fr.omega_deg
        if "omega" in resolved_cuts:
            lo_om, hi_om = resolved_cuts["omega"]
            if not (lo_om <= om <= hi_om):
                continue

        f_walk = 1.0
        if walkoff_params is not None:
            f_walk = float(walkoff_factor(om, walkoff_params[0], walkoff_params[1]))
            if f_walk < walkoff_f_min:
                continue

        img_clean, qm, shift_px, gf = prepare_frame(
            fr,
            geo,
            bg_mode=bg_mode,
            bin_px=bin_px,
            approximation=approximation,
            omega_refinement=omega_refinement,
            beamstop_mask=bs_mask,
            exposure_normalization=exposure_normalization,
        )
        origin_shifts.append(shift_px)
        E_val = fr.energy_ev
        exp_s = fr.exposure_s if exposure_normalization else 1.0

        t_mem = (
            float(
                membrane_transmission(
                    om + gf.omega0_deg, E_val, thickness_nm=membrane_thickness_nm
                )
            )
            if apply_membrane
            else 1.0
        )
        dens = apply_geometric_corrections(
            img_clean,
            qm,
            exposure_s=exp_s,
            solid_angle=solid_angle_correction,
            polarization=polarization_correction,
            walkoff_f=f_walk,
            membrane_t=t_mem,
        )
        prepared.append((dens, qm, om, f_walk))
        used_omegas.append(om)

    if not prepared:
        raise ValueError(
            "No frames remained after applying omega cuts and walk-off threshold."
        )

    if x_range is None or y_range is None:
        x_vals_all: list[NDArray[np.float64]] = []
        y_vals_all: list[NDArray[np.float64]] = []
        for dens, qm, _, _ in prepared:
            sel = np.isfinite(dens)
            for c_dom, (lo_c, hi_c) in resolved_cuts.items():
                if c_dom in {x_dom, y_dom, "omega"}:
                    continue
                c_arr = qm.get_coord(c_dom)
                sel = sel & (c_arr >= lo_c) & (c_arr <= hi_c)
            if np.any(sel):
                x_vals_all.append(qm.get_coord(x_dom)[sel])
                y_vals_all.append(qm.get_coord(y_dom)[sel])
        if x_range is None:
            if x_vals_all:
                xv = np.concatenate(x_vals_all)
                x_range = (float(np.min(xv)), float(np.max(xv)))
                if x_range[0] == x_range[1]:
                    x_range = (x_range[0] - 0.1, x_range[1] + 0.1)
            else:
                x_range = (-4.2 * G, 4.2 * G)
        if y_range is None:
            if y_vals_all:
                yv = np.concatenate(y_vals_all)
                y_range = (float(np.min(yv)), float(np.max(yv)))
                if y_range[0] == y_range[1]:
                    y_range = (y_range[0] - 0.1, y_range[1] + 0.1)
            else:
                y_range = (-0.25, 0.25)

    nx, ny = int(bins[0]), int(bins[1])
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1, dtype=np.float64)
    y_edges = np.linspace(y_range[0], y_range[1], ny + 1, dtype=np.float64)
    H = np.zeros((nx, ny), dtype=np.float64)
    N = np.zeros((nx, ny), dtype=np.float64)

    for dens, qm, om, _ in prepared:
        dens_use = dens
        if bg_mode == "offline_band":
            qx_edges_bg = (
                x_edges if x_dom == "Qx" else np.linspace(-4.5 * G, 4.5 * G, nx + 1)
            )
            dens_use, _ = subtract_offline_band_background(
                dens, qm.Qx, qm.Qy, G, qx_edges_bg, qy_off_g=qy_off_g
            )

        sel = np.isfinite(dens_use)
        if mask_parasitic:
            sel = sel & parasitic_mask(om, qm.two_theta_deg)
        for c_dom, (lo_c, hi_c) in resolved_cuts.items():
            if c_dom == "omega":
                continue
            c_arr = qm.get_coord(c_dom)
            sel = sel & (c_arr >= lo_c) & (c_arr <= hi_c)

        if not np.any(sel):
            continue
        x_c = qm.get_coord(x_dom)[sel]
        y_c = qm.get_coord(y_dom)[sel]
        w_c = dens_use[sel]

        h_frame, _, _ = np.histogram2d(x_c, y_c, bins=[x_edges, y_edges], weights=w_c)
        n_frame, _, _ = np.histogram2d(x_c, y_c, bins=[x_edges, y_edges])
        H += h_frame
        N += n_frame

    meta: dict[str, Any] = {
        "Lambda_nm": geo.Lambda_nm,
        "G_inv_nm": G,
        "energy_ev": float(frames_list[0].energy_ev),
        "omega_deg": np.array(used_omegas, dtype=np.float64),
        "origin_shift_px": np.array(origin_shifts, dtype=np.float64),
        "n_frames": len(prepared),
    }
    return QMap2D(
        H=H,
        N=N,
        x_edges=x_edges,
        y_edges=y_edges,
        x_domain=x_dom,
        y_domain=y_dom,
        cuts=resolved_cuts,
        approximation=approximation,
        bg_mode=bg_mode,
        metadata=meta,
    )


def reduce_1d(
    source: QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any] | None = None,
    *,
    domain: str = "Qx",
    domain_range: tuple[float, float] | None = None,
    bins: int = 300,
    bin_px: int = 1,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "corner",
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    solid_angle_correction: bool = True,
    exposure_normalization: bool = True,
    polarization_correction: bool = False,
    apply_membrane: bool = False,
    membrane_thickness_nm: float = 50.0,
    walkoff_params: tuple[float, float] | None = None,
    beamstop_mask: NDArray[np.bool_] | None = None,
    deduplicate_omega: bool = True,
    energy_ev: float | None = None,
    fixed_domain: str | None = None,
    fixed_value: float | None = None,
    fixed_tol: float | None = None,
    fixed_range: tuple[float, float] | None = None,
    qx_cut: tuple[float, float] | None = None,
    qy_cut: tuple[float, float] | float | None = None,
    qz_cut: tuple[float, float] | None = None,
    qr_cut: tuple[float, float] | None = None,
    omega_cut: tuple[float, float] | float | None = None,
    qy_cut_over_g: float | None = None,
    cuts: Mapping[str, tuple[float, float]] | None = None,
) -> Trace1D:
    """Reduce CCD frames or an existing ``QMap2D`` to a 1D trace ``I(domain)``."""
    target = normalize_domain_name(domain)
    if isinstance(source, QMap2D):
        return source.slice_1d(
            target,
            cut_range=fixed_range,
            fixed_value=fixed_value,
            fixed_tol=fixed_tol,
        )

    if geometry is None:
        raise ValueError("geometry must be provided when reducing from frames.")
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    G = geo.g_inv_nm
    resolved_cuts = _resolve_cuts(
        g_inv_nm=G,
        fixed_domain=fixed_domain,
        fixed_value=fixed_value,
        fixed_tol=fixed_tol,
        fixed_range=fixed_range,
        qx_cut=qx_cut,
        qy_cut=qy_cut,
        qz_cut=qz_cut,
        qr_cut=qr_cut,
        omega_cut=omega_cut,
        qy_cut_over_g=qy_cut_over_g,
        cuts=cuts,
    )
    if "Qy" not in resolved_cuts and target in {"Qx", "Qr", "signed_Qr", "Qz"}:
        resolved_cuts["Qy"] = (-0.05 * G, 0.05 * G)

    frames_list = load_frames(
        source, deduplicate_omega=deduplicate_omega, energy_ev=energy_ev
    )
    if not frames_list:
        raise ValueError("No frames matched selection criteria for reduce_1d.")

    coord_chunks: list[NDArray[np.float64]] = []
    weight_chunks: list[NDArray[np.float64]] = []
    weight2_chunks: list[NDArray[np.float64]] = []

    for fr in frames_list:
        om = fr.omega_deg
        if "omega" in resolved_cuts and target != "omega":
            lo_om, hi_om = resolved_cuts["omega"]
            if not (lo_om <= om <= hi_om):
                continue

        f_walk = (
            float(walkoff_factor(om, walkoff_params[0], walkoff_params[1]))
            if walkoff_params is not None
            else 1.0
        )
        if f_walk < 0.05:
            continue

        img_clean, qm, _, gf = prepare_frame(
            fr,
            geo,
            bg_mode=bg_mode,
            bin_px=bin_px,
            approximation=approximation,
            omega_refinement=omega_refinement,
            beamstop_mask=beamstop_mask,
            exposure_normalization=exposure_normalization,
        )
        E_val = fr.energy_ev
        exp_s = fr.exposure_s if exposure_normalization else 1.0

        t_mem = (
            float(
                membrane_transmission(
                    om + gf.omega0_deg, E_val, thickness_nm=membrane_thickness_nm
                )
            )
            if apply_membrane
            else 1.0
        )
        dens = apply_geometric_corrections(
            img_clean,
            qm,
            exposure_s=exp_s,
            solid_angle=solid_angle_correction,
            polarization=polarization_correction,
            walkoff_f=f_walk,
            membrane_t=t_mem,
        )
        if bg_mode == "offline_band":
            qx_edges_bg = np.linspace(-4.5 * G, 4.5 * G, max(bins, 200) + 1)
            dens, _ = subtract_offline_band_background(
                dens, qm.Qx, qm.Qy, G, qx_edges_bg
            )

        sel = np.isfinite(dens)
        for c_dom, (lo_c, hi_c) in resolved_cuts.items():
            if c_dom == target:
                continue
            c_arr = qm.get_coord(c_dom)
            sel = sel & (c_arr >= lo_c) & (c_arr <= hi_c)

        if np.any(sel):
            c_vals = qm.get_coord(target)[sel]
            w_vals = dens[sel]
            coord_chunks.append(c_vals)
            weight_chunks.append(w_vals)
            weight2_chunks.append(w_vals**2)

    if not coord_chunks:
        raise ValueError("No pixels matched the orthogonal cuts in reduce_1d.")

    all_c = np.concatenate(coord_chunks)
    all_w = np.concatenate(weight_chunks)
    all_w2 = np.concatenate(weight2_chunks)

    if domain_range is None:
        domain_range = (float(np.min(all_c)), float(np.max(all_c)))
        if domain_range[0] == domain_range[1]:
            domain_range = (domain_range[0] - 0.1, domain_range[1] + 0.1)

    edges = np.linspace(
        domain_range[0], domain_range[1], int(bins) + 1, dtype=np.float64
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_sum, _ = np.histogram(all_c, bins=edges, weights=all_w)
    h_sum2, _ = np.histogram(all_c, bins=edges, weights=all_w2)
    n_cnt, _ = np.histogram(all_c, bins=edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_val = np.where(n_cnt > 0, h_sum / np.maximum(n_cnt, 1), np.nan)
        var_val = np.where(
            n_cnt > 1,
            np.clip(h_sum2 / n_cnt - mean_val**2, 0.0, None) / (n_cnt - 1),
            np.where(n_cnt == 1, np.abs(mean_val), np.nan),
        )
        sigma_val = np.sqrt(var_val)

    return Trace1D(
        coords=centers,
        edges=edges,
        intensity=mean_val,
        sigma=sigma_val,
        counts=n_cnt.astype(np.float64),
        domain=target,
        cuts=resolved_cuts,
        metadata={
            "Lambda_nm": geo.Lambda_nm,
            "G_inv_nm": G,
            "approximation": approximation,
            "bg_mode": bg_mode,
        },
    )


def _infer_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    if centers.size == 0:
        return np.array([-0.01, 0.01], dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.005, centers[0] + 0.005], dtype=np.float64)
    diffs = 0.5 * np.diff(centers)
    return np.concatenate(
        [[centers[0] - diffs[0]], centers[:-1] + diffs, [centers[-1] + diffs[-1]]]
    )


def extract_bragg_rod(
    source: QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    order: int,
    *,
    domain: Literal["Qz", "omega"] = "Qz",
    qx_half_width_g: float = 0.08,
    qy_cut_over_g: float = 0.05,
    bins: int = 120,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "offline_band",
) -> Trace1D:
    """Extract the 1D Bragg diffraction rod profile ``I_m(Qz)`` or ``I_m(omega)`` for
    order ``m``.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    G = geo.g_inv_nm
    q_center = float(order) * G
    dq = abs(qx_half_width_g) * G
    if isinstance(source, QMap2D):
        return source.slice_1d(domain, cut_range=(q_center - dq, q_center + dq))
    return reduce_1d(
        source,
        geo,
        domain=domain,
        bins=bins,
        approximation=approximation,
        bg_mode=bg_mode,
        qx_cut=(q_center - dq, q_center + dq),
        qy_cut_over_g=qy_cut_over_g,
    )


def extract_order_rods_qz(
    source: pd.DataFrame | QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any] | None = None,
    *,
    orders: Sequence[int] = (-4, -3, -2, -1, 1, 2, 3, 4),
    normalize_first_order: bool = False,
    intensity_col: str | None = None,
    sigma_col: str | None = None,
    qx_half_width_g: float = 0.08,
    qy_cut_over_g: float = 0.05,
    bins: int = 100,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "offline_band",
) -> dict[int, Trace1D]:
    """Extract ``I_m(Qz)`` Bragg truncation rods for every diffraction order ``m``.

    Accepts an indexed order table ``DataFrame`` (from ``build_order_table``), a
    projected ``QMap2D(x_domain="Qx", y_domain="Qz")``, or a ``Frame`` sequence.
    When ``normalize_first_order=True`` on an order table, divides each frame's
    order intensity by ``I_{+1} + I_{-1}`` so incident flux and beam walk-off
    cancel within each frame.

    Parameters
    ----------
    source : DataFrame, QMap2D, or frame sequence
        Projected dataset or indexed Bragg order table.
    geometry : Geometry or dict, optional
        Calibrated geometry (required when ``source`` is frames, optional if
        ``Lambda_nm`` is stored in ``QMap2D.metadata`` or ``DataFrame``).
    orders : sequence of int, optional
        Diffraction orders to extract. Default is ``(-4, -3, -2, -1, 1, 2, 3, 4)``.
    normalize_first_order : bool, optional
        Normalize per-frame intensities by ``I_{+1} + I_{-1}``. Default is False.
    intensity_col, sigma_col : str, optional
        Explicit intensity and uncertainty column names when ``source`` is a
        ``DataFrame`` (defaults to ``I_cps_mc`` if present, else ``I_cps``).
    qx_half_width_g : float, optional
        Fractional ``Qx / G`` integration half-width when slicing ``QMap2D`` or
        frames. Default is ``0.08``.
    qy_cut_over_g : float, optional
        Specular ``|Qy| / G`` half-width when reducing from frames. Default is ``0.05``.
    bins : int, optional
        Number of ``Qz`` bins when reducing from frames. Default is 100.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.
    bg_mode : BackgroundMode, optional
        Background subtraction mode when reducing from frames.

    Returns
    -------
    dict of int to Trace1D
        Mapping ``{m: Trace1D(domain="Qz", ...)}`` for each available order ``m``.
    """
    out: dict[int, Trace1D] = {}
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        if df.empty:
            return out
        qz_col = "Qz_inv_nm" if "Qz_inv_nm" in df.columns else "Qz"
        i_col = intensity_col or ("I_cps_mc" if "I_cps_mc" in df.columns else "I_cps")
        s_col = sigma_col or (
            "sigma_cps_mc" if "sigma_cps_mc" in df.columns else "sigma_cps"
        )
        if normalize_first_order and "file" in df.columns:
            p1 = df[df["order"] == 1].set_index("file")[i_col]
            m1 = df[df["order"] == -1].set_index("file")[i_col]
            denom_map = (p1 + m1).dropna()
            df = df[df["file"].isin(denom_map.index)].copy()
            denom = df["file"].map(denom_map).to_numpy(dtype=np.float64)
            df["_val"] = df[i_col].to_numpy(dtype=np.float64) / np.maximum(denom, 1e-30)
            df["_sig"] = (
                df[s_col].to_numpy(dtype=np.float64) / np.maximum(denom, 1e-30)
                if s_col in df.columns
                else np.zeros(len(df), dtype=np.float64)
            )
            val_col, err_col = "_val", "_sig"
        else:
            val_col = i_col
            err_col = s_col if s_col in df.columns else ""

        om_key = "omega_deg" if "omega_deg" in df.columns else qz_col
        df["_om_round"] = np.round(df[om_key].to_numpy(dtype=np.float64), 3)
        for m in orders:
            sub = df[df["order"] == int(m)]
            if sub.empty:
                continue
            grp = (
                sub.groupby("_om_round", as_index=False)
                .agg(
                    qz=(qz_col, "mean"),
                    intensity=(val_col, "mean"),
                    sigma=(err_col, "mean") if err_col else (val_col, lambda _s: 0.0),
                    counts=(val_col, "count"),
                )
                .sort_values("qz")
            )
            qz_arr = grp["qz"].to_numpy(dtype=np.float64)
            out[int(m)] = Trace1D(
                coords=qz_arr,
                edges=_infer_edges(qz_arr),
                intensity=grp["intensity"].to_numpy(dtype=np.float64),
                sigma=grp["sigma"].to_numpy(dtype=np.float64),
                counts=grp["counts"].to_numpy(dtype=np.float64),
                domain="Qz",
                metadata={
                    "order": int(m),
                    "normalized_first_order": normalize_first_order,
                },
            )
        return out

    if isinstance(source, QMap2D):
        lam = (
            geometry.Lambda_nm
            if isinstance(geometry, Geometry)
            else (
                Geometry.from_dict(geometry).Lambda_nm
                if geometry is not None
                else float(source.metadata.get("Lambda_nm", 80.0))
            )
        )
        geo_obj = Geometry(Lambda_nm=float(lam))
        for m in orders:
            tr = extract_bragg_rod(
                source, geo_obj, int(m), domain="Qz", qx_half_width_g=qx_half_width_g
            )
            if np.any(np.isfinite(tr.intensity)):
                tr.metadata["order"] = int(m)
                out[int(m)] = tr
        return out

    if geometry is None:
        raise ValueError("geometry is required when extracting order rods from frames.")
    for m in orders:
        try:
            tr = extract_bragg_rod(
                source,
                geometry,
                int(m),
                domain="Qz",
                qx_half_width_g=qx_half_width_g,
                qy_cut_over_g=qy_cut_over_g,
                bins=bins,
                approximation=approximation,
                bg_mode=bg_mode,
            )
        except ValueError:
            continue
        if np.any(np.isfinite(tr.intensity)):
            tr.metadata["order"] = int(m)
            out[int(m)] = tr
    return out


def extract_qx_by_omega(
    source: pd.DataFrame | QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any] | None = None,
    *,
    omegas: Sequence[float] | None = None,
    omega_tol_deg: float = 0.4,
    intensity_col: str | None = None,
    sigma_col: str | None = None,
    qy_cut_over_g: float = 0.05,
    domain: str = "Qx",
    bins: int = 300,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "corner",
) -> dict[float, Trace1D]:
    """Extract ``I(Qx)`` profiles for each measured sample tilt angle ``omega``.

    Supports:
    - An indexed order ``DataFrame`` (returning discrete integrated Bragg order
      intensities ``I(Qx = m * G)`` per measured ``omega``).
    - A projected ``QMap2D(x_domain="Qx", y_domain="omega")``.
    - A ``Frame`` sequence (returning continuous detector-streak ``I(Qx)`` profiles
      at each measured ``omega``).

    ``domain`` selects the abscissa: ``"Qx"`` (default), ``"Qr"`` or
    ``"signed_Qr"`` (``sign(Qx) * sqrt(Qx^2 + Qy^2)``).

    Returns
    -------
    dict of float to Trace1D
        Mapping ``{omega_deg: Trace1D(domain="Qx", ...)}`` sorted by ascending
        ``omega_deg``.
    """
    out: dict[float, Trace1D] = {}
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        if df.empty:
            return out
        om_col = (
            "omega_motor_deg"
            if "omega_motor_deg" in df.columns
            else ("omega_deg" if "omega_deg" in df.columns else "omega_true_deg")
        )
        qx_col = "Qx_inv_nm" if "Qx_inv_nm" in df.columns else "Qx"
        dom = normalize_domain_name(domain)
        if dom in {"Qr", "signed_Qr"}:
            qy_col = "Qy_inv_nm" if "Qy_inv_nm" in df.columns else "Qy"
            qy_v = (
                df[qy_col].to_numpy(dtype=np.float64) if qy_col in df.columns else 0.0
            )
            qr_v = np.hypot(df[qx_col].to_numpy(dtype=np.float64), qy_v)
            df["_q_dom"] = (
                np.sign(df[qx_col].to_numpy(dtype=np.float64)) * qr_v
                if dom == "signed_Qr"
                else qr_v
            )
            qx_col = "_q_dom"
        elif dom != "Qx":
            raise ValueError(f"domain must be Qx, Qr or signed_Qr, got {domain!r}")
        i_col = intensity_col or ("I_cps_mc" if "I_cps_mc" in df.columns else "I_cps")
        s_col = sigma_col or (
            "sigma_cps_mc" if "sigma_cps_mc" in df.columns else "sigma_cps"
        )
        df["_om_key"] = np.round(df[om_col].to_numpy(dtype=np.float64), 2)
        target_omegas = (
            sorted(float(o) for o in omegas)
            if omegas is not None
            else sorted(float(o) for o in df["_om_key"].unique())
        )
        for om_val in target_omegas:
            sub = df[np.abs(df["_om_key"] - om_val) <= omega_tol_deg]
            if sub.empty:
                continue
            grp = (
                sub.groupby("order", as_index=False)
                .agg(
                    qx=(qx_col, "mean"),
                    intensity=(i_col, "mean"),
                    sigma=(s_col, "mean")
                    if s_col in sub.columns
                    else (i_col, lambda _s: 0.0),
                    counts=(i_col, "count"),
                )
                .sort_values("qx")
            )
            qx_arr = grp["qx"].to_numpy(dtype=np.float64)
            out[float(om_val)] = Trace1D(
                coords=qx_arr,
                edges=_infer_edges(qx_arr),
                intensity=grp["intensity"].to_numpy(dtype=np.float64),
                sigma=grp["sigma"].to_numpy(dtype=np.float64),
                counts=grp["counts"].to_numpy(dtype=np.float64),
                domain=dom,
                cuts={"omega": (om_val - omega_tol_deg, om_val + omega_tol_deg)},
                metadata={
                    "omega_deg": float(om_val),
                    "orders": grp["order"].to_numpy(dtype=int),
                },
            )
        return out

    if isinstance(source, QMap2D):
        if source.y_domain != "omega":
            raise ValueError(
                "QMap2D must have y_domain='omega' for extract_qx_by_omega; "
                f"got ({source.x_domain!r}, {source.y_domain!r})."
            )
        om_list = (
            [float(o) for o in omegas]
            if omegas is not None
            else [float(o) for o in source.y_centers]
        )
        for om_val in sorted(om_list):
            tr = source.slice_1d(
                source.x_domain, fixed_value=om_val, fixed_tol=omega_tol_deg
            )
            if np.any(np.isfinite(tr.intensity)):
                tr.metadata["omega_deg"] = float(om_val)
                out[float(om_val)] = tr
        return out

    if geometry is None:
        raise ValueError(
            "geometry is required when extracting I(Qx; omega) from frames."
        )
    frames_list = load_frames(source, deduplicate_omega=True)
    avail_om = sorted({round(fr.omega_deg, 2) for fr in frames_list})
    target_om = sorted(float(o) for o in omegas) if omegas is not None else avail_om
    for om_val in target_om:
        matched = [
            fr for fr in frames_list if abs(fr.omega_deg - om_val) <= omega_tol_deg
        ]
        if not matched:
            continue
        tr = reduce_1d(
            matched,
            geometry,
            domain=domain,
            bins=bins,
            approximation=approximation,
            bg_mode=bg_mode,
            qy_cut_over_g=qy_cut_over_g,
        )
        tr.metadata["omega_deg"] = float(om_val)
        out[float(om_val)] = tr
    return out


def extract_inplane_linecut(
    source: QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    *,
    domain: Literal["Qx", "Qr", "signed_Qr"] = "Qx",
    fixed_qz: float | None = None,
    qz_tol: float = 0.015,
    fixed_omega: float | None = None,
    omega_tol: float = 0.5,
    qy_cut_over_g: float = 0.05,
    bins: int = 300,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "corner",
) -> Trace1D:
    """Extract an in-plane 1D linecut ``I(Qx)``, ``I(Qr)``, or ``I(signed_Qr)`` at fixed
    ``Qz`` or ``omega``.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    if isinstance(source, QMap2D):
        if fixed_qz is not None:
            return source.slice_1d(domain, fixed_value=fixed_qz, fixed_tol=qz_tol)
        if fixed_omega is not None:
            return source.slice_1d(domain, fixed_value=fixed_omega, fixed_tol=omega_tol)
        return source.slice_1d(domain)

    qz_cut = (fixed_qz - qz_tol, fixed_qz + qz_tol) if fixed_qz is not None else None
    om_cut = (
        (fixed_omega - omega_tol, fixed_omega + omega_tol)
        if fixed_omega is not None
        else None
    )
    return reduce_1d(
        source,
        geo,
        domain=domain,
        bins=bins,
        approximation=approximation,
        bg_mode=bg_mode,
        qz_cut=qz_cut,
        omega_cut=om_cut,
        qy_cut_over_g=qy_cut_over_g,
    )


def extract_diffuse_cut(
    source: QMap2D | Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    *,
    direction: Literal["Qy", "Qz", "Qx"] = "Qy",
    order: int = 1,
    qx_half_width_g: float = 0.12,
    qy_band_inv_nm: tuple[float, float] = (0.018, 0.085),
    qz_cut: tuple[float, float] | None = None,
    omega_cut: tuple[float, float] | float | None = None,
    bins: int = 120,
    approximation: ApproximationMode = "correct",
    bg_mode: BackgroundMode = "corner",
) -> Trace1D:
    """Extract 1D off-specular diffuse scattering along ``Qy``, ``Qz``, or ``Qx``.

    - ``direction="Qy"``: off-specular streak ``I(Qy)`` through Bragg order ``m``
      (``|Qx - m*G| <= qx_half_width_g * G``) at optional ``qz_cut`` or ``omega_cut``.
    - ``direction="Qz"``: off-specular rod wing ``I_diff(Qz)`` integrated over
      ``qy_band_inv_nm[0] <= |Qy| <= qy_band_inv_nm[1]`` around ``Qx = m*G``.
    - ``direction="Qx"``: in-plane off-specular profile ``I_diff(Qx)`` integrated
      over ``qy_band_inv_nm[0] <= |Qy| <= qy_band_inv_nm[1]`` (outside the ``Qy = 0``
      specular core, revealing rod-localized LER diffuse ridges at ``Qx = m*G``).
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    G = geo.g_inv_nm
    q_center = float(order) * G
    dq = abs(qx_half_width_g) * G

    if isinstance(source, QMap2D):
        if direction == "Qy":
            if source.x_domain == "Qx" and source.y_domain == "Qy":
                return source.slice_1d("Qy", cut_range=(q_center - dq, q_center + dq))
            if source.x_domain == "Qy" and source.y_domain == "Qz":
                return source.slice_1d("Qy", cut_range=qz_cut)
        elif direction == "Qz" and source.x_domain == "Qy" and source.y_domain == "Qz":
            lo_y, hi_y = qy_band_inv_nm
            y_c = source.x_centers
            sel = (np.abs(y_c) >= lo_y) & (np.abs(y_c) <= hi_y)
            h_1d = source.H[sel, :].sum(axis=0)
            n_1d = source.N[sel, :].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_1d = np.where(n_1d > 0, h_1d / np.maximum(n_1d, 1.0), np.nan)
                sig_1d = np.where(
                    n_1d > 0, np.abs(mean_1d) / np.sqrt(np.maximum(n_1d, 1.0)), np.nan
                )
            return Trace1D(
                coords=source.y_centers,
                edges=source.y_edges,
                intensity=mean_1d,
                sigma=sig_1d,
                counts=n_1d,
                domain="Qz",
                cuts={"|Qy|": qy_band_inv_nm},
                metadata=dict(source.metadata),
            )
        elif direction == "Qx" and source.x_domain == "Qx" and source.y_domain == "Qy":
            lo_y, hi_y = qy_band_inv_nm
            y_c = source.y_centers
            sel = (np.abs(y_c) >= lo_y) & (np.abs(y_c) <= hi_y)
            h_1d = source.H[:, sel].sum(axis=1)
            n_1d = source.N[:, sel].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_1d = np.where(n_1d > 0, h_1d / np.maximum(n_1d, 1.0), np.nan)
                sig_1d = np.where(
                    n_1d > 0, np.abs(mean_1d) / np.sqrt(np.maximum(n_1d, 1.0)), np.nan
                )
            return Trace1D(
                coords=source.x_centers,
                edges=source.x_edges,
                intensity=mean_1d,
                sigma=sig_1d,
                counts=n_1d,
                domain="Qx",
                cuts={"|Qy|": qy_band_inv_nm},
                metadata=dict(source.metadata),
            )

    if direction == "Qy":
        return reduce_1d(
            source,
            geo,
            domain="Qy",
            bins=bins,
            approximation=approximation,
            bg_mode=bg_mode,
            qx_cut=(q_center - dq, q_center + dq),
            qz_cut=qz_cut,
            omega_cut=omega_cut,
        )

    # For direction in {"Qz", "Qx"}, accumulate both positive and negative |Qy| bands
    lo_y, hi_y = float(qy_band_inv_nm[0]), float(qy_band_inv_nm[1])
    qx_window = (q_center - dq, q_center + dq) if direction == "Qz" else None
    pos_tr = reduce_1d(
        source,
        geo,
        domain=direction,
        bins=bins,
        approximation=approximation,
        bg_mode=bg_mode,
        qx_cut=qx_window,
        qy_cut=(lo_y, hi_y),
        qz_cut=qz_cut,
        omega_cut=omega_cut,
    )
    neg_tr = reduce_1d(
        source,
        geo,
        domain=direction,
        bins=bins,
        domain_range=(float(pos_tr.edges[0]), float(pos_tr.edges[-1])),
        approximation=approximation,
        bg_mode=bg_mode,
        qx_cut=qx_window,
        qy_cut=(-hi_y, -lo_y),
        qz_cut=qz_cut,
        omega_cut=omega_cut,
    )
    n_tot = pos_tr.counts + neg_tr.counts
    h_tot = np.nan_to_num(pos_tr.intensity * pos_tr.counts) + np.nan_to_num(
        neg_tr.intensity * neg_tr.counts
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_comb = np.where(n_tot > 0, h_tot / np.maximum(n_tot, 1.0), np.nan)
        sig_comb = np.where(
            n_tot > 0, np.abs(mean_comb) / np.sqrt(np.maximum(n_tot, 1.0)), np.nan
        )
    return Trace1D(
        coords=pos_tr.coords,
        edges=pos_tr.edges,
        intensity=mean_comb,
        sigma=sig_comb,
        counts=n_tot,
        domain=direction,
        cuts={"|Qy|": (lo_y, hi_y)},
        metadata={"order": int(order), "Lambda_nm": geo.Lambda_nm},
    )


def pixel_q_area(
    qx_map: ArrayLike,
    qy_map: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the area of each detector pixel in the ``(Qx, Qy)`` plane [1/nm^2] via
    Jacobian.
    """
    qx = np.asarray(qx_map, dtype=np.float64)
    qy = np.asarray(qy_map, dtype=np.float64)
    ax, ay = np.gradient(qx)
    bx, by = np.gradient(qy)
    return np.abs(ax * by - ay * bx)


def extract_offspec_rod_map(
    source: Iterable[str | Path | Frame] | Path | str,
    geometry: Geometry | Mapping[str, Any],
    order: int = 1,
    *,
    omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
    qy_range: tuple[float, float] = (-0.10, 0.10),
    n_qy: int = 80,
    qx_half_g: float = 0.25,
    core_qy_inv_nm: float = 0.015,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
    bg_mode: Literal["strip_fit", "corner"] = "corner",
) -> QMap2D:
    """Build a 2D off-specular rod sheet ``(dI / d^2Q) / I_m(Qy, Qz)`` [nm^2] across a
    rotation series.

    At each frame (incident angle ``omega``), integrates pixels within
    ``|Qx - m*G| < qx_half_g * G`` into ``Qy`` bins, divides by the Jacobian pixel
    area ``d^2Q`` in ``(Qx, Qy)`` and by the specular Bragg order photons ``I_m``
    in the same frame, and places that row at the order's ``Qz(m, omega)``.
    """
    geo = geometry if isinstance(geometry, Geometry) else Geometry.from_dict(geometry)
    G = geo.g_inv_nm
    frames_list = load_frames(source, deduplicate_omega=True)
    if not frames_list:
        raise ValueError("No frames found for extract_offspec_rod_map.")

    qy_edges = np.linspace(qy_range[0], qy_range[1], int(n_qy) + 1, dtype=np.float64)
    rows_val: list[NDArray[np.float64]] = []
    rows_cnt: list[NDArray[np.float64]] = []
    qz_list: list[float] = []

    for fr in frames_list:
        om = fr.omega_deg
        gf = geo.with_omega_refinement(omega_refinement, om)
        if bg_mode == "strip_fit":
            res = process_frame_strip(
                gf,
                fr.image,
                fr.energy_ev,
                om,
                exposure_s=fr.exposure_s,
                adu_per_photon=adu_per_photon,
            )
            d = res.cleaned_adu
            qm = res.qmap
        else:
            d, _, _ = subtract_corner_offset(fr.image)
            xx, yy = frame_pixel_grid(fr.image.shape, 1)
            qm = compute_pixel_q(gf, xx, yy, fr.energy_ev, om)

        in_core = (
            (np.abs(qm.Qx - order * G) < qx_half_g * G)
            & (np.abs(qm.Qy) < core_qy_inv_nm)
            & np.isfinite(d)
        )
        n_order = float(np.nansum(d[in_core])) / adu_per_photon
        if n_order <= 1.0 or in_core.sum() == 0:
            continue

        qz_m = float(np.nanmean(qm.Qz[in_core]))
        area = pixel_q_area(qm.Qx, qm.Qy)
        band = (np.abs(qm.Qx - order * G) < qx_half_g * G) & np.isfinite(d)
        ib = np.digitize(qm.Qy[band], qy_edges) - 1
        ok = (ib >= 0) & (ib < n_qy)
        ph = np.bincount(ib[ok], d[band][ok] / adu_per_photon, minlength=n_qy)
        ba = np.bincount(ib[ok], area[band][ok], minlength=n_qy)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_row = np.where(ba > 0, (ph / ba) / n_order, 0.0)
        rows_val.append(norm_row)
        rows_cnt.append((ba > 0).astype(np.float64))
        qz_list.append(qz_m)

    if not rows_val:
        raise ValueError(f"Order m={order} was not observed in the provided frames.")

    sort_idx = np.argsort(qz_list)
    qz_sorted = np.asarray(qz_list, dtype=np.float64)[sort_idx]
    qz_edges = _infer_edges(qz_sorted)
    H = np.column_stack([rows_val[i] for i in sort_idx])
    N = np.column_stack([rows_cnt[i] for i in sort_idx])
    return QMap2D(
        H=H,
        N=N,
        x_edges=qy_edges,
        y_edges=qz_edges,
        x_domain="Qy",
        y_domain="Qz",
        cuts={"Qx": ((order - qx_half_g) * G, (order + qx_half_g) * G)},
        metadata={
            "order": int(order),
            "Lambda_nm": geo.Lambda_nm,
            "units": "nm^2",
        },
    )


def readoff_edge_psd_born(
    cleaned_adu: ArrayLike,
    qx_map: ArrayLike,
    qy_map: ArrayLike,
    order: int,
    pitch_nm: float,
    line_width_nm: float,
    qy_edges: ArrayLike,
    *,
    adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
    qx_half_g: float = 0.25,
    qy_min_inv_nm: float = 0.015,
    box_qx_g: float = 0.25,
    box_qy_inv_nm: float = 0.015,
) -> pd.DataFrame:
    """Read off the two-sided line-edge roughness PSD ``S(Qy)`` [nm^3] from off-specular
    diffuse wings.

    For a grating of pitch ``p`` and line width ``w`` whose edges have independent,
    uncorrelated roughness with two-sided PSD ``S(Qy)``, first-order Born
    approximation gives at ``Qx = m*G``:

    ``(dI / d^2Q) / I_m = 2 * p * S(Qy) / (F_m^2 * (2 * pi)^2)``

    where ``F_m^2 = (2 * sin(m * G * w / 2) / (m * G))^2``. Contrast, line height,
    ``Qz`` form factor, and absorption cancel in the ratio to ``I_m``.
    """
    d = np.asarray(cleaned_adu, dtype=np.float64)
    qx = np.asarray(qx_map, dtype=np.float64)
    qy = np.asarray(qy_map, dtype=np.float64)
    qyb = np.asarray(qy_edges, dtype=np.float64)
    G = 2.0 * np.pi / pitch_nm

    area = pixel_q_area(qx, qy)
    in_core = (
        (np.abs(qx - order * G) < box_qx_g * G)
        & (np.abs(qy) < box_qy_inv_nm)
        & np.isfinite(d)
    )
    n_order_ph = float(np.nansum(d[in_core])) / adu_per_photon

    band = (
        (np.abs(qx - order * G) < qx_half_g * G)
        & np.isfinite(d)
        & (np.abs(qy) > qy_min_inv_nm)
    )
    ib = np.digitize(np.abs(qy[band]), qyb) - 1
    nb = len(qyb) - 1
    ok = (ib >= 0) & (ib < nb)
    cnt = np.bincount(ib[ok], d[band][ok] / adu_per_photon, minlength=nb)
    bin_area = np.bincount(ib[ok], area[band][ok], minlength=nb)

    f2_val = (2.0 * np.sin(order * G * line_width_nm / 2.0) / (order * G)) ** 2
    scale = f2_val * ((2.0 * np.pi) ** 2) / (2.0 * pitch_nm) / max(n_order_ph, 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(bin_area > 0, scale * cnt / bin_area, np.nan)
        dS = np.where(
            bin_area > 0,
            scale * np.sqrt(np.clip(cnt, 1.0, None)) / bin_area,
            np.nan,
        )
    qy_c = np.sqrt(qyb[1:] * qyb[:-1])
    return pd.DataFrame(
        {
            "qy_center_inv_nm": qy_c,
            "f_inv_nm": qy_c / (2.0 * np.pi),
            "psd_nm3": S,
            "psd_err_nm3": dS,
            "photons": cnt,
            "order_photons": n_order_ph,
        }
    )


def extract_rod_localized_diffuse_qy(
    image: ArrayLike,
    geometry: Geometry,
    energy_ev: float,
    omega_deg: float,
    order: int,
    *,
    qy_edges: ArrayLike | None = None,
    on_g: float = 0.06,
    off_g: tuple[float, float] = (0.12, 0.30),
    bin_px: int = 1,
) -> pd.DataFrame:
    """Extract rod-localized ``Qy`` streak profile around ``Qx = m*G`` with flanking
    subtraction.

    Sums intensity over ``|Qx / G - m| < on_g`` minus the same number of pixels
    at the mean level of the flanking bands ``off_g[0] < |Qx / G - m| < off_g[1]``.
    """
    im, _, _ = subtract_corner_offset(image, use_clipped_mean=True)
    im_binned = bin_image_2d(im, bin_px=bin_px, mode="sum") if bin_px > 1 else im
    xx, yy = frame_pixel_grid(np.asarray(image).shape, bin_px=bin_px)
    qm = compute_pixel_q(geometry, xx, yy, energy_ev, omega_deg, bin_px=bin_px)

    if qy_edges is None:
        qy_mag = np.geomspace(2.5e-4, 0.30, 32)
        edges = np.r_[-qy_mag[::-1], qy_mag]
    else:
        edges = np.asarray(qy_edges, dtype=np.float64)

    G = geometry.g_inv_nm
    x_frac = qm.Qx / G
    mm = np.rint(x_frac).astype(int)
    dx = x_frac - mm

    nq = len(edges) - 1
    iq = np.digitize(qm.Qy, edges) - 1
    in_q = (iq >= 0) & (iq < nq) & np.isfinite(im_binned)
    sel = in_q & (mm == order)
    on_mask = sel & (np.abs(dx) < on_g)
    off_mask = sel & (np.abs(dx) > off_g[0]) & (np.abs(dx) < off_g[1])

    on_sum = np.bincount(iq[on_mask], weights=im_binned[on_mask], minlength=nq)
    on_n = np.bincount(iq[on_mask], minlength=nq)
    off_sum = np.bincount(iq[off_mask], weights=im_binned[off_mask], minlength=nq)
    off_n = np.bincount(iq[off_mask], minlength=nq)

    off_mean = np.where(off_n > 0, off_sum / np.maximum(off_n, 1), 0.0)
    streak = np.where((on_n > 0) & (off_n > 0), on_sum - on_n * off_mean, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dqy = np.diff(edges)
    return pd.DataFrame(
        {
            "Qy_inv_nm": centers,
            "dQy_inv_nm": dqy,
            "streak_adu": streak,
            "on_sum_adu": on_sum,
            "on_npix": on_n,
            "off_sum_adu": off_sum,
            "off_npix": off_n,
        }
    )


def extract_line_strip(
    geometry: Geometry,
    image: ArrayLike,
    qx_map: ArrayLike,
    qy_map: ArrayLike,
    *,
    half_width_px: int = 28,
) -> NDArray[np.float64]:
    """Extract ``+-half_width_px`` detector strip about the ``Qy = 0`` diffraction line,
    oriented with ``Qx`` increasing to the right.
    """
    arr = np.asarray(image, dtype=np.float64)
    qx = np.asarray(qx_map, dtype=np.float64)
    _ = np.asarray(qy_map, dtype=np.float64)
    nrow, ncol = arr.shape
    rot = np.radians(geometry.line_rot_deg)
    # Orient along the spot line (works for both horizontal and vertical line_rot_deg)
    diag = int(min(nrow, ncol) * 0.48)
    s = np.arange(-diag, diag + 1, dtype=np.float64)
    t = np.arange(-half_width_px, half_width_px + 1, dtype=np.float64)
    ss, tt = np.meshgrid(s, t, indexing="xy")
    uu = np.clip(
        np.rint(geometry.u0_px + ss * np.cos(rot) - tt * np.sin(rot)).astype(int),
        0,
        ncol - 1,
    )
    vv = np.clip(
        np.rint(geometry.v0_px + ss * np.sin(rot) + tt * np.cos(rot)).astype(int),
        0,
        nrow - 1,
    )
    st = arr[vv, uu]
    qx_line = qx[vv[half_width_px], uu[half_width_px]]
    if qx_line[0] > qx_line[-1]:
        return st[:, ::-1]
    return st


def plot_detector_strips(
    strips_by_omega: Mapping[float, NDArray[np.float64]],
    *,
    omega_highlights: Sequence[tuple[float, str]] = (
        (0.0, "white"),
        (10.0, "#ff3fd0"),
        (20.0, "#8cff00"),
    ),
    cmap: str = "turbo",
) -> Figure:
    """Plot thin ``Qy = 0`` CCD detector strips stacked by sample tilt ``omega``."""
    n_strips = len(omega_highlights)
    fig = plt.figure(figsize=(10.5, 0.62 * n_strips + 0.25), facecolor="black")
    gs = fig.add_gridspec(
        n_strips,
        1,
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.06,
        hspace=0.14,
    )
    cm = plt.get_cmap(cmap).with_extremes(bad="black", under="black")

    for idx, (om_val, col) in enumerate(omega_highlights):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("black")
        sp = strips_by_omega.get(float(om_val))
        if sp is not None and np.any(np.isfinite(sp)):
            v_sp = np.log10(np.clip(sp, 1.0, None))
            pos = v_sp[np.isfinite(v_sp) & (v_sp > 0)]
            vmax_s = float(np.percentile(pos, 99.95)) if pos.size else 3.0
            ax.imshow(
                v_sp,
                cmap=cm,
                aspect="auto",
                vmin=0.4,
                vmax=max(vmax_s, 1.0),
                interpolation="nearest",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(col)
            spine.set_linewidth(1.8)
        ax.text(
            -0.012,
            0.5,
            f"{om_val:g}°",
            transform=ax.transAxes,
            color=col,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    fig.text(
        0.022,
        0.5,
        r"$\omega$",
        color="white",
        ha="center",
        va="center",
        fontsize=13,
    )
    return fig


def plot_omega_butterfly(
    qmap: QMap2D,
    geometry: Geometry,
    strips_by_omega: Mapping[float, NDArray[np.float64]] | None = None,
    *,
    energy_ev: float = 285.2,
    omega_highlights: tuple[tuple[float, str], ...] = (
        (0.0, "white"),
        (10.0, "#ff3fd0"),
        (20.0, "#8cff00"),
    ),
    label: str | None = None,
) -> Figure:
    """Render stacked thin CCD detector strips above the ``(Qx, Qz)`` butterfly map
    with matching ``Qy = 0`` Ewald-sphere arc overlays.
    """
    n_strips = len(omega_highlights) if strips_by_omega else 0
    fig = plt.figure(figsize=(9.5, 4.8 + 0.65 * n_strips), facecolor="white")
    ratios = [0.22] * n_strips + [1.65]
    gs = fig.add_gridspec(n_strips + 1, 1, height_ratios=ratios, hspace=0.14)
    cm = plt.get_cmap("turbo").with_extremes(bad="black", under="black")

    if strips_by_omega:
        for idx, (om_val, col) in enumerate(omega_highlights):
            ax_s = fig.add_subplot(gs[idx])
            ax_s.set_facecolor("black")
            sp = strips_by_omega.get(float(om_val))
            if sp is not None and np.any(np.isfinite(sp)):
                v_sp = np.log10(np.clip(sp, 1.0, None))
                vmax_s = (
                    float(np.nanpercentile(v_sp, 99.9))
                    if np.any(np.isfinite(v_sp))
                    else 3.0
                )
                ax_s.imshow(
                    v_sp,
                    cmap=cm,
                    aspect="auto",
                    vmin=0.5,
                    vmax=max(vmax_s, 1.0),
                    interpolation="nearest",
                )
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            border_col = "#18181b" if col == "white" else col
            for spine in ax_s.spines.values():
                spine.set_color(border_col)
                spine.set_linewidth(1.8)
            ax_s.text(
                -0.015,
                0.5,
                f"{om_val:g}°",
                transform=ax_s.transAxes,
                color=border_col,
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
            if idx == 0:
                add_panel_label(ax_s, "a", x=-0.075, y=1.02)

    ax_m = fig.add_subplot(gs[-1])
    ax_m.set_facecolor("black")
    m_sm = qmap.smoothed(sigma_bins=(1.2, 1.8))
    pos = m_sm[np.isfinite(m_sm) & (m_sm > 0)]
    vmax = float(np.percentile(pos, 99.9)) if pos.size else 1.0
    vmin = max(vmax * 1e-4, float(pos.min()) if pos.size else 1e-4)
    disp = np.where(np.isfinite(m_sm) & (m_sm > 0), m_sm, np.nan)
    ax_m.pcolormesh(
        qmap.x_edges,
        qmap.y_edges,
        disp.T,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap=cm,
        shading="flat",
        rasterized=True,
    )
    for om_val, col in omega_highlights:
        qx_tr, qz_tr = ewald_trace_qxqz(geometry, om_val, energy_ev)
        ax_m.plot(qx_tr, qz_tr, "--", color=col, lw=1.1)

    ax_m.set_xlim(float(qmap.x_edges[0]), float(qmap.x_edges[-1]))
    ax_m.set_ylim(float(qmap.y_edges[0]), float(qmap.y_edges[-1]))
    if label:
        ax_m.text(
            0.02,
            0.94,
            label,
            transform=ax_m.transAxes,
            color="white",
            fontsize=10,
            va="top",
            bbox={"fc": "black", "ec": "none", "alpha": 0.7, "pad": 1.5},
        )
    if n_strips > 0:
        add_panel_label(ax_m, "b", x=-0.075, y=1.01)
    ax_m.set_xlabel(domain_axis_label(qmap.x_domain))
    ax_m.set_ylabel(domain_axis_label(qmap.y_domain))
    return fig


def plot_specular_1d_panels(
    qx_by_omega: Mapping[float, Trace1D],
    rods_by_order: Mapping[int, Trace1D],
    *,
    ridge_step_decades: float = 0.25,
    order_offset_decades: float = 0.75,
    stack_orders: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the two canonical 1D specular summaries side-by-side:
    (a) ``I vs Qx`` ridge waterfall per measured ``omega``, and
    (b) ``I vs Qz`` per diffraction order ``m``.
    """
    fig, (ax_qx, ax_qz) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    cmap_om = plt.get_cmap("RdBu_r")
    om_sorted = sorted(qx_by_omega.keys())
    om_min = om_sorted[0] if om_sorted else -20.0
    om_max = om_sorted[-1] if om_sorted else 20.0
    span = max(om_max - om_min, 1e-6)

    for idx, om_val in enumerate(om_sorted):
        tr = qx_by_omega[om_val]
        ok = np.isfinite(tr.intensity) & (tr.intensity > 0)
        if not np.any(ok):
            continue
        col = cmap_om((om_val - om_min) / span)
        y = np.log10(tr.intensity[ok]) + ridge_step_decades * idx
        marker = "o-" if ok.sum() <= 20 else "-"
        ax_qx.plot(
            tr.coords[ok],
            y,
            marker,
            ms=3.5 if ok.sum() <= 20 else 0.0,
            lw=1.1,
            color=col,
        )
    ax_qx.set_yticks([])
    ax_qx.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
    ax_qx.set_ylabel(r"$\log_{10} I$ (offset per $\omega$)")
    add_panel_label(ax_qx, "a")

    orders_sorted = sorted(
        (m for m in rods_by_order if m != 0),
        key=lambda m: -m if m > 0 else abs(m) + 100,
    )
    # Symmetrically order from +M..+1, -1..-M when stacking
    pos_m = sorted([m for m in rods_by_order if m > 0], reverse=True)
    neg_m = sorted([m for m in rods_by_order if m < 0], reverse=True)
    sym_orders = pos_m + neg_m if stack_orders else orders_sorted

    for k, m in enumerate(sym_orders):
        tr = rods_by_order[m]
        ok = np.isfinite(tr.intensity) & (tr.intensity > 0)
        if not np.any(ok):
            continue
        col = ORDER_PALETTE.get(abs(m), "#3f3f46")
        if stack_orders:
            shift = (0.5 * (len(sym_orders) - 1) - k) * order_offset_decades
            y = np.log10(tr.intensity[ok]) + shift
            ax_qz.plot(
                tr.coords[ok],
                y,
                "o-",
                ms=3.2,
                lw=1.0,
                color=col,
                mfc=col if m > 0 else "white",
            )
            ax_qz.text(
                float(tr.coords[ok][-1]) + 0.008,
                float(y[-1]),
                f"{m:+d}",
                color=col,
                fontsize=8.5,
                va="center",
            )
        else:
            mk = "o-" if m > 0 else "s--"
            ax_qz.plot(
                tr.coords[ok],
                tr.intensity[ok],
                mk,
                ms=3.4,
                lw=1.0,
                color=col,
                mfc=col if m > 0 else "white",
                label=f"m = {m:+d}",
            )

    if stack_orders:
        ax_qz.set_yticks([])
        ax_qz.set_ylabel(r"$\log_{10} I_m$ (offset per order)")
    else:
        ax_qz.set_yscale("log")
        ax_qz.set_ylabel(r"$I_m(Q_z)$")
        ax_qz.legend(ncol=2, fontsize=8, frameon=False)
    ax_qz.set_xlabel(r"$Q_z$ (nm$^{-1}$)")
    add_panel_label(ax_qz, "b")
    fig.tight_layout()
    return fig, (ax_qx, ax_qz)
