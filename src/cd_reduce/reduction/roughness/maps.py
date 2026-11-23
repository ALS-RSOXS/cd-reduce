"""Variance-weighted reduced-intensity maps with explicit slab bookkeeping."""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from cd_reduce.reduction.geometry import Geometry, compute_pixel_q
from cd_reduce.reduction.mapping import QMap2D
from cd_reduce.reduction.roughness.reduce import ReducedPixels

DEFAULT_ROUGHNESS_QY_SLAB_G = 0.05
"""Default Qy slab half-width, in units of the grating vector G = 2*pi/L."""


def detector_q_pitch(geometry: Geometry, energy_ev: float) -> float:
    """|dQx/du| at the beam centre in nm^-1 per pixel (central difference)."""
    u0, v0 = float(geometry.u0_px), float(geometry.v0_px)
    q = compute_pixel_q(
        geometry,
        np.array([u0 - 0.5, u0 + 0.5]),
        np.array([v0, v0]),
        energy_ev,
        0.0,
        approximation="correct",
    )
    return float(abs(q.Qx[1] - q.Qx[0]))


def map_reduced(
    pixels: ReducedPixels,
    plane: Literal["qx-qz", "qy-qz"],
    *,
    slab_center: float = 0.0,
    slab_halfwidth: float,
    bin_q: tuple[float, float],
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> QMap2D:
    """Bin reduced pixels into an inverse-variance-weighted ``D`` map.

    Plane ``"qx-qz"`` keeps ``|Qy - slab_center| <= slab_halfwidth``; plane
    ``"qy-qz"`` slabs on Qx. Per bin ``H = sum(D / var_D)`` and
    ``N = sum(1 / var_D)``, so ``QMap2D.mean`` is the weighted mean of ``D``.
    Empty bins are NaN.

    ``var_D = var_M * (D / M)**2`` (``D / M`` is the contrast factor).
    Pixels with ``M == 0`` or non-finite / non-positive ``var_D`` carry no
    usable variance and are dropped.
    """
    if plane == "qx-qz":
        xs, ys, slab = pixels.Qx, pixels.Qz, pixels.Qy
        xd, yd, cut_name = "Qx", "Qz", "Qy"
    elif plane == "qy-qz":
        xs, ys, slab = pixels.Qy, pixels.Qz, pixels.Qx
        xd, yd, cut_name = "Qy", "Qz", "Qx"
    else:
        raise ValueError(f"plane must be 'qx-qz' or 'qy-qz', got {plane!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        var_d = pixels.var_M * (pixels.D / pixels.M) ** 2
    ok = (
        (np.abs(slab - slab_center) <= slab_halfwidth)
        & (pixels.M != 0)
        & np.isfinite(var_d)
        & (var_d > 0)
        & np.isfinite(pixels.D)
    )
    xs, ys, d, v = xs[ok], ys[ok], pixels.D[ok], var_d[ok]
    if x_range is None:
        x_range = (float(xs.min()), float(xs.max())) if xs.size else (-1.0, 1.0)
    if y_range is None:
        y_range = (float(ys.min()), float(ys.max())) if ys.size else (-1.0, 1.0)
    nx = max(1, int(np.ceil((x_range[1] - x_range[0]) / bin_q[0] - 1e-9)))
    ny = max(1, int(np.ceil((y_range[1] - y_range[0]) / bin_q[1] - 1e-9)))
    xe = np.linspace(x_range[0], x_range[1], nx + 1)
    ye = np.linspace(y_range[0], y_range[1], ny + 1)
    H, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=d / v)
    N, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=1.0 / v)
    return QMap2D(
        H=H,
        N=N,
        x_edges=xe,
        y_edges=ye,
        x_domain=xd,
        y_domain=yd,
        cuts={cut_name: (slab_center - slab_halfwidth, slab_center + slab_halfwidth)},
        approximation="correct",
        metadata={
            "slab_center": slab_center,
            "slab_halfwidth": slab_halfwidth,
            "bin_q": tuple(bin_q),
            "quantity": "D",
        },
    )


def plot_reduced_map(
    qmap: QMap2D, *, w: float | None = None, ax: Axes | None = None
) -> Axes:
    """Plot a reduced map (log colour); slab in the title, Qx = +-k*pi/w lines."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.5))
    img = qmap.mean
    pos = img[np.isfinite(img) & (img > 0)]
    kw = {}
    if pos.size:
        vmax = float(np.percentile(pos, 99.9))
        kw["norm"] = LogNorm(vmin=max(vmax * 1e-4, float(pos.min())), vmax=vmax)
    ax.pcolormesh(
        qmap.x_edges,
        qmap.y_edges,
        np.where(img > 0, img, np.nan).T,
        shading="flat",
        rasterized=True,
        **kw,
    )
    md = qmap.metadata
    cut = "Qy" if qmap.x_domain == "Qx" else "Qx"
    ax.set_xlabel(f"{qmap.x_domain} (nm$^{{-1}}$)")
    ax.set_ylabel(f"{qmap.y_domain} (nm$^{{-1}}$)")
    ax.set_title(
        f"|{cut} - {md.get('slab_center', 0.0):g}| <= "
        f"{md.get('slab_halfwidth', float('nan')):g} nm$^{{-1}}$"
    )
    if w is not None and qmap.x_domain == "Qx":
        lo, hi = float(qmap.x_edges[0]), float(qmap.x_edges[-1])
        kmax = int(np.floor(max(abs(lo), abs(hi)) * w / np.pi))
        for k in range(1, kmax + 1):
            for sgn in (-1, 1):
                x = sgn * k * np.pi / w
                if lo <= x <= hi:
                    ax.axvline(x, color="white", lw=0.6, ls=":", alpha=0.7)
    return ax
