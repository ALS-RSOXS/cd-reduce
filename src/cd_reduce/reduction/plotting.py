"""Publication plots for reduced CD-RSoXS data.

- :func:`plot_traces_by_omega`: ``I`` vs ``Qx`` / ``Qr`` / ``signed_Qr`` per
  incidence angle, offset waterfall, diverging colour map with a colour bar and
  an ``omega`` label on every trace.
- :func:`plot_rods_by_order`: ``I_m`` vs ``Qz`` per order, offset and labelled
  ``+m`` / ``-m`` at the right.
- :func:`plot_qmap_grid`: grid of 2D maps (e.g. samples x {exact, small-angle})
  with the ``Qy = 0`` Ewald trace overlaid in each panel's own mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from cd_reduce.reduction.geometry import Geometry, ewald_trace_qxqz
from cd_reduce.reduction.mapping import (
    ORDER_PALETTE,
    QMap2D,
    Trace1D,
    domain_axis_label,
)


def _finite_pos(tr: Trace1D) -> np.ndarray:
    return np.isfinite(tr.intensity) & (tr.intensity > 0) & np.isfinite(tr.coords)


def plot_traces_by_omega(
    traces: Mapping[float, Trace1D],
    *,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    offset_decades: float = 0.3,
    normalize: bool = False,
    annotate: bool = True,
    colorbar: bool = True,
    marker: str | None = "o",
    omega_range: tuple[float, float] | None = None,
    max_labels: int = 9,
    min_count_frac: float = 0.3,
) -> tuple[Figure, Axes]:
    """Waterfall of ``log10 I`` vs ``Qx`` (or ``Qr``) per incidence angle ``omega``.

    Each trace is coloured by ``omega`` on a diverging map centred on 0, offset by
    ``offset_decades`` per angle (lowest omega at the bottom), and labelled with
    its ``omega`` at the right end. ``normalize=True`` divides each trace by its
    maximum so only the shape is compared.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
    else:
        fig = ax.figure  # type: ignore[assignment]
    oms = sorted(traces)
    if not oms:
        return fig, ax  # type: ignore[return-value]
    lo, hi = omega_range or (min(oms), max(oms))
    lim = max(abs(lo), abs(hi), 1e-6)
    norm = Normalize(-lim, lim) if lo < 0 < hi else Normalize(lo, hi)
    cm = plt.get_cmap(cmap)
    dense = max(int(_finite_pos(traces[o]).sum()) for o in oms) > 40
    for k, om in enumerate(oms):
        tr = traces[om]
        ok = _finite_pos(tr)
        if dense and np.any(tr.counts[ok] > 0):
            # drop partially filled bins at the detector edge
            ok &= tr.counts >= min_count_frac * np.median(tr.counts[ok])
        if not ok.any():
            continue
        y = np.log10(tr.intensity[ok])
        if normalize:
            y = y - np.max(y)
        y = y + k * offset_decades
        col = cm(norm(om))
        ax.plot(
            tr.coords[ok],
            y,
            "-" if dense or marker is None else f"{marker}-",
            color=col,
            lw=1.0,
            ms=4.0,
            mec=col,
        )
        step = max(1, int(np.ceil(len(oms) / max(max_labels, 1))))
        if annotate and (k % step == 0 or k == len(oms) - 1):
            lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
            ax.annotate(
                f"{om:+g}°",
                (float(tr.coords[ok][-1]), float(y[-1])),
                xytext=(4, 0),
                textcoords="offset points",
                color=col if lum < 0.7 else "0.35",
                fontsize=7,
                va="center",
            )
    dom = next(iter(traces.values())).domain
    ax.set_xlabel(domain_axis_label(dom))
    ax.set_ylabel(r"$\log_{10} I$ (offset)")
    ax.set_yticks([])
    if colorbar:
        cb = fig.colorbar(
            ScalarMappable(norm, cm),
            ax=ax,
            pad=0.12 if annotate else 0.03,
            fraction=0.05,
        )
        cb.set_label(r"$\omega$ (°)")
    return fig, ax  # type: ignore[return-value]


def plot_rods_by_order(
    rods: Mapping[int, Trace1D],
    *,
    ax: Axes | None = None,
    offset_decades: float = 0.75,
    palette: Mapping[int, str] | None = None,
    annotate: bool = True,
    normalize: bool = True,
) -> tuple[Figure, Axes]:
    """Stacked ``log10 I_m`` vs ``Qz`` per order (``+M..+1`` above ``-1..-M``).

    Positive and negative orders share a colour per ``|m|``; labels sit at the
    right edge of the axes. ``normalize`` scales each rod to its maximum so the
    offsets separate shapes rather than absolute order strengths.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.6))
    else:
        fig = ax.figure  # type: ignore[assignment]
    pal = dict(ORDER_PALETTE) | dict(palette or {})
    seq = sorted((m for m in rods if m > 0), reverse=True) + sorted(
        (m for m in rods if m < 0), reverse=True
    )
    n = len(seq)
    for k, m in enumerate(seq):
        tr = rods[m]
        ok = _finite_pos(tr)
        if not ok.any():
            continue
        col = pal.get(abs(m), "#3f3f46")
        y = np.log10(tr.intensity[ok])
        if normalize:
            y = y - np.max(y)
        y = y + (0.5 * (n - 1) - k) * offset_decades
        ax.plot(tr.coords[ok], y, "o-", color=col, ms=4, lw=1.0, alpha=0.9)
        if annotate:
            ax.annotate(
                f"{m:+d}",
                (1.01, float(np.median(y))),
                xycoords=("axes fraction", "data"),
                color=col,
                fontsize=9,
                va="center",
                annotation_clip=False,
            )
    ax.set_xlabel(r"$Q_z$ (nm$^{-1}$)")
    ax.set_ylabel(r"$\log_{10} I_m$ (offset)")
    ax.set_yticks([])
    return fig, ax  # type: ignore[return-value]


def plot_qmap_grid(
    qmaps: Sequence[Sequence[QMap2D]],
    *,
    labels: Sequence[Sequence[str]] | None = None,
    geometries: Sequence[Geometry] | None = None,
    ewald_omega_deg: float | None = 0.0,
    energy_ev: float | None = None,
    cmap: str = "turbo",
    sigma_q: tuple[float, float] | None = None,
    dynamic_range: float = 1e4,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    panel_size: tuple[float, float] = (3.6, 3.6),
    detector_shape: tuple[int, int] = (1024, 1024),
    share_axes: bool = True,
) -> Figure:
    """Grid of 2D maps sharing axes, e.g. rows = samples, cols = exact / small-angle.

    If ``geometries`` (one per row) is given, the ``Qy = 0`` locus at
    ``ewald_omega_deg`` is drawn dashed in each panel using that panel's own
    ``approximation`` (curved Ewald arc for ``"correct"``, straight for
    ``"small_angle"``). Only meaningful for ``(Qx, Qz)``-type maps.
    """
    nr = len(qmaps)
    nc = max(len(r) for r in qmaps)
    fig, axes = plt.subplots(
        nr,
        nc,
        figsize=(panel_size[0] * nc, panel_size[1] * nr),
        sharex=share_axes,
        sharey=share_axes,
        squeeze=False,
    )
    cm = plt.get_cmap(cmap).with_extremes(bad="black", under="black")
    for i, row in enumerate(qmaps):
        for j, qm in enumerate(row):
            ax = axes[i, j]
            ax.set_facecolor("black")
            img = qm.smoothed(sigma_q=sigma_q)
            pos = img[np.isfinite(img) & (img > 0)]
            if pos.size:
                vmax = float(np.percentile(pos, 99.9))
                ax.pcolormesh(
                    qm.x_edges,
                    qm.y_edges,
                    np.where(img > 0, img, np.nan).T,
                    norm=LogNorm(vmax / dynamic_range, vmax),
                    cmap=cm,
                    shading="flat",
                    rasterized=True,
                )
            if geometries is not None and ewald_omega_deg is not None:
                e_ev = (
                    energy_ev
                    if energy_ev is not None
                    else float(qm.metadata.get("energy_ev", 285.0))
                )
                qx_t, qz_t = ewald_trace_qxqz(
                    geometries[i],
                    ewald_omega_deg,
                    e_ev,
                    shape=detector_shape,
                    approximation=qm.approximation,  # type: ignore[arg-type]
                )
                ax.plot(qx_t, qz_t, "--", color="white", lw=1.0)
            if labels is not None:
                ax.text(
                    0.04,
                    0.95,
                    labels[i][j],
                    transform=ax.transAxes,
                    color="white",
                    va="top",
                    fontsize=9,
                )
            if i == nr - 1 or not share_axes:
                ax.set_xlabel(domain_axis_label(qm.x_domain))
            if j == 0 or not share_axes:
                ax.set_ylabel(domain_axis_label(qm.y_domain))
            ax.set_xlim(*(xlim or (qm.x_edges[0], qm.x_edges[-1])))
            ax.set_ylim(*(ylim or (qm.y_edges[0], qm.y_edges[-1])))
        for j in range(len(row), nc):
            axes[i, j].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_qmap_grid",
    "plot_rods_by_order",
    "plot_traces_by_omega",
]
