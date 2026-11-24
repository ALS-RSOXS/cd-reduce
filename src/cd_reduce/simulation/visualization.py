"""Visualization helpers for 2D/3D refractive index maps and Born/Multislice
simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from cd_reduce.reduction.mapping import ORDER_PALETTE, add_panel_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D

    from cd_reduce.simulation.born import BornSimulationResult2D, BornSimulationResult3D
    from cd_reduce.simulation.multislice import (
        MultisliceSimulationResult2D,
        MultisliceSimulationResult3D,
    )
    from cd_reduce.simulation.structures import (
        RefractiveIndexMap2D,
        RefractiveIndexMap3D,
    )


def _cell_edge_extent(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float, float]:
    """``imshow`` extent spanning cell edges, so cells sit at their true centres."""

    def edges(c: np.ndarray) -> tuple[float, float]:
        c = np.asarray(c, dtype=np.float64)
        h = 0.5 * float(c[1] - c[0]) if c.size > 1 else 0.5
        return float(c[0]) - h, float(c[-1]) + h

    return (*edges(a), *edges(b))


def _style_clean_axes(ax: Any, *, line_width: float = 0.75) -> None:
    """Apply clean publication-grade spine and tick styling matching `mkdocs-rsoxs`."""
    for spine in ax.spines.values():
        spine.set_linewidth(line_width)
        spine.set_color("#27272a")
    ax.tick_params(colors="#27272a", width=line_width, labelsize=8.5)


def plot_structure_2d(
    structure: RefractiveIndexMap2D,
    component: str = "both",
    ax: Any = None,
    cmap_delta: str = "cividis",
    cmap_beta: str = "magma",
    *,
    show_titles: bool = False,
    panel_labels: Sequence[str] | None = ("a", "b"),
) -> Any:
    """Plot a 2D refractive index map `n(x, z) = 1 - delta(x, z) + i * beta(x, z)`."""
    import matplotlib.pyplot as plt

    extent = _cell_edge_extent(structure.x_nm, structure.z_nm)
    if component == "both":
        if ax is None:
            _, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
        else:
            axes = ax
        im0 = axes[0].imshow(
            structure.delta.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_delta,
        )
        axes[0].set_xlabel(r"$x\ (\mathrm{nm})$")
        axes[0].set_ylabel(r"$z\ (\mathrm{nm})$")
        _style_clean_axes(axes[0])
        if show_titles:
            axes[0].set_title(r"$\delta(x, z)$")
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb0.set_label(r"$\delta(x, z)$")

        im1 = axes[1].imshow(
            structure.beta.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_beta,
        )
        axes[1].set_xlabel(r"$x\ (\mathrm{nm})$")
        axes[1].set_ylabel(r"$z\ (\mathrm{nm})$")
        _style_clean_axes(axes[1])
        if show_titles:
            axes[1].set_title(r"$\beta(x, z)$")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.set_label(r"$\beta(x, z)$")

        if panel_labels is not None and len(panel_labels) >= 2:
            add_panel_label(axes[0], panel_labels[0])
            add_panel_label(axes[1], panel_labels[1])
        return axes

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    arr = structure.delta.T if component == "delta" else structure.beta.T
    cmap = cmap_delta if component == "delta" else cmap_beta
    label = r"$\delta(x, z)$" if component == "delta" else r"$\beta(x, z)$"
    im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set_xlabel(r"$x\ (\mathrm{nm})$")
    ax.set_ylabel(r"$z\ (\mathrm{nm})$")
    _style_clean_axes(ax)
    if show_titles:
        ax.set_title(label)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    if panel_labels is not None and len(panel_labels) >= 1:
        add_panel_label(ax, panel_labels[0])
    return ax


def plot_structure_3d(
    structure: RefractiveIndexMap3D,
    component: str = "delta",
    z_index: int | None = None,
    y_index: int | None = None,
    axes: Sequence[Any] | None = None,
    cmap: str = "cividis",
    *,
    show_titles: bool = False,
    panel_labels: Sequence[str] | None = ("a", "b"),
) -> Any:
    """Plot orthogonal slices `(x, y)` and `(x, z)` of a 3D refractive index map."""
    import matplotlib.pyplot as plt

    arr = structure.delta if component == "delta" else structure.beta
    iz = structure.nz // 2 if z_index is None else int(z_index)
    iy = structure.ny // 2 if y_index is None else int(y_index)
    comp_tex = r"$\delta$" if component == "delta" else r"$\beta$"

    if axes is None:
        _, axes_arr = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True)
    else:
        axes_arr = axes

    extent_xy = _cell_edge_extent(structure.x_nm, structure.y_nm)
    im0 = axes_arr[0].imshow(
        arr[:, :, iz].T,
        origin="lower",
        aspect="auto",
        extent=extent_xy,
        cmap=cmap,
    )
    axes_arr[0].set_xlabel(r"$x\ (\mathrm{nm})$")
    axes_arr[0].set_ylabel(r"$y\ (\mathrm{nm})$")
    _style_clean_axes(axes_arr[0])
    if show_titles:
        axes_arr[0].set_title(f"{component}(x, y) at z = {structure.z_nm[iz]:.1f} nm")
    cb0 = plt.colorbar(im0, ax=axes_arr[0], fraction=0.046, pad=0.04)
    cb0.set_label(f"{comp_tex}(x, y)")

    extent_xz = _cell_edge_extent(structure.x_nm, structure.z_nm)
    im1 = axes_arr[1].imshow(
        arr[:, iy, :].T,
        origin="lower",
        aspect="auto",
        extent=extent_xz,
        cmap=cmap,
    )
    axes_arr[1].set_xlabel(r"$x\ (\mathrm{nm})$")
    axes_arr[1].set_ylabel(r"$z\ (\mathrm{nm})$")
    _style_clean_axes(axes_arr[1])
    if show_titles:
        axes_arr[1].set_title(f"{component}(x, z) at y = {structure.y_nm[iy]:.1f} nm")
    cb1 = plt.colorbar(im1, ax=axes_arr[1], fraction=0.046, pad=0.04)
    cb1.set_label(f"{comp_tex}(x, z)")

    if panel_labels is not None and len(panel_labels) >= 2:
        add_panel_label(axes_arr[0], panel_labels[0])
        add_panel_label(axes_arr[1], panel_labels[1])
    return axes_arr


def plot_structure_3d_surface(
    structure: RefractiveIndexMap3D,
    *,
    component: str = "delta",
    ax: Any = None,
    cmap: Any = None,
    color: str | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    y_cut: float | None = None,
    relief_scale: float = 1.0,
    stride: int = 1,
    threshold: float = 0.05,
    substrate_color: str = "#3a3f47",
    dark: bool = True,
    elev: float = 32.0,
    azim: float = -62.0,
) -> Any:
    """Render a 3D refractive-index map as a shaded perspective view.

    The film's top surface ``T(x, y)`` is drawn as a relief, coloured by
    ``component`` (``"delta"`` or ``"beta"``) in the top filled cell. The front
    face at ``y = y_cut`` (default: the lowest ``y`` shown) shows the profile
    into depth. Uncovered regions are drawn as substrate.

    Parameters
    ----------
    structure : RefractiveIndexMap3D
        Map to render, e.g. from :func:`build_rough_grating_3d`.
    component : {"delta", "beta"}
        Optical constant used for the colours.
    color : str, optional
        One solid colour for all material (e.g. a uniform developed line)
        instead of colouring by ``component``.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Axes created with ``projection="3d"``; a new figure is made if omitted.
    x_range, y_range : tuple of float, optional
        Region to draw in nm (defaults: the whole map).
    y_cut : float, optional
        ``y`` of the front cut face in nm.
    relief_scale : float
        Exaggeration of the surface relief above the lowest film top (heights
        below it are drawn to scale; 1 = true scale everywhere).
    stride : int
        Draw every ``stride``-th cell in ``x`` and ``y`` (speed).
    threshold : float
        Cells with ``|delta| + |beta|`` above ``threshold`` times the maximum
        count as filled.
    dark : bool
        Dark background styling.

    Returns
    -------
    Axes3D
        The axes; ``ax.surface_norm`` holds the colour normalisation.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(structure.delta if component == "delta" else structure.beta)
    mag = np.abs(structure.delta) + np.abs(structure.beta)
    x, y, z = structure.x_nm, structure.y_nm, structure.z_nm
    dz = float(z[1] - z[0]) if z.size > 1 else 1.0
    xs = (x >= x_range[0]) & (x <= x_range[1]) if x_range else np.ones(x.size, bool)
    ys = (y >= y_range[0]) & (y <= y_range[1]) if y_range else np.ones(y.size, bool)
    ix = np.flatnonzero(xs)[:: max(int(stride), 1)]
    iy = np.flatnonzero(ys)[:: max(int(stride), 1)]
    a = arr[np.ix_(ix, iy)]
    m = mag[np.ix_(ix, iy)]
    filled = m > threshold * float(np.max(mag))
    any_fill = filled.any(axis=2)
    k_top = np.where(any_fill, z.size - 1 - np.argmax(filled[:, :, ::-1], axis=2), 0)
    ii, jj = np.indices(k_top.shape)
    below = np.clip(k_top - 1, 0, None)
    ref = np.maximum(m[ii, jj, below], m[ii, jj, k_top])
    frac = np.clip(m[ii, jj, k_top] / np.where(ref > 0, ref, 1.0), 0.0, 1.0)
    height = np.where(any_fill, (z[k_top] - 0.5 * dz) + frac * dz, 0.0)
    col = np.where(any_fill, a[ii, jj, below], np.nan)

    tops = height[any_fill]
    base = float(tops.min()) if tops.size else 0.0

    def lift(zz: np.ndarray) -> np.ndarray:
        return np.where(zz > base, base + (zz - base) * relief_scale, zz)

    vals = col[np.isfinite(col)]  # full cells below the surface (no partial fill)
    lo, hi = float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0))
    if hi - lo <= 1e-6 * max(abs(hi), 1e-30):  # uniform material: low end of the map
        lo, hi = lo, lo + max(abs(lo), 1e-30)
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    cm = (
        mcolors.LinearSegmentedColormap.from_list(
            "film", ["#4a6fb5", "#8a5bb0", "#e0706a", "#f39150"]
        )
        if cmap is None
        else plt.get_cmap(cmap)
    )
    sub = np.array(mcolors.to_rgba(substrate_color))

    def rgba(values: np.ndarray) -> np.ndarray:
        if color is not None:
            out = np.broadcast_to(mcolors.to_rgba(color), (*values.shape, 4)).copy()
        else:
            out = cm(norm(np.nan_to_num(values, nan=float(norm.vmin or 0.0))))
        out[np.isnan(values)] = sub
        return out

    ls = mcolors.LightSource(azdeg=315, altdeg=45)
    top = rgba(col)
    top[..., :3] = ls.shade_rgb(  # normals from a lightly smoothed relief
        top[..., :3], gaussian_filter(lift(height), 1.0), fraction=0.35
    )

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(x[ix], y[iy], indexing="ij")
    ax.plot_surface(
        X,
        Y,
        lift(height),
        facecolors=top,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    # front x-z cut face
    jc = 0 if y_cut is None else int(np.argmin(np.abs(y[iy] - y_cut)))
    zf = np.concatenate([[0.0], z + 0.5 * dz])
    Zf = np.minimum(zf[None, :], height[:, jc][:, None])
    Xf = np.repeat(x[ix][:, None], zf.size, axis=1)
    Yf = np.full_like(Xf, y[iy][jc])
    face_vals = np.where(filled[:, jc, :], a[:, jc, :], np.nan)
    face_vals = np.concatenate([face_vals[:, :1], face_vals], axis=1)
    ax.plot_surface(
        Xf,
        Yf,
        lift(Zf),
        facecolors=rgba(face_vals),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot(
        x[ix],
        np.full(ix.size, y[iy][jc]),
        lift(height[:, jc]),
        color="white" if dark else "black",
        lw=1.0,
    )

    zmax = float(np.max(lift(height)))
    ax.set_box_aspect((np.ptp(x[ix]), np.ptp(y[iy]), max(zmax, 1e-6)), zoom=1.1)
    ax.zaxis.set_ticks([0.0, round(zmax)])
    if relief_scale != 1:
        ax.zaxis.set_ticklabels(["0", f"{base + (zmax - base) / relief_scale:.0f}"])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")
    ax.set_zlabel(r"$z$ (nm)")
    if dark:
        ax.set_facecolor("#06121a")
        ax.figure.set_facecolor("#06121a")
        for axis in cast("tuple[Axis3D, ...]", (ax.xaxis, ax.yaxis, ax.zaxis)):
            axis.set_pane_color((0, 0, 0, 0))
            axis.label.set_color("#d4d4d8")
            axis.line.set_color("#52525b")
        ax.tick_params(colors="#a1a1aa", labelsize=7)
        ax.grid(False)
    setattr(ax, "surface_norm", norm)  # noqa: B010  (documented extra attribute)
    return ax


def plot_multislice_wavefield(
    result: MultisliceSimulationResult2D,
    axes: Sequence[Any] | None = None,
    overlay_structure: bool = True,
    *,
    show_titles: bool = False,
    panel_labels: Sequence[str] | None = ("a", "b"),
) -> Any:
    """Plot internal multislice wavefield `|psi(x, z)|^2` and phase `arg(psi)`."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes_arr = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True)
    else:
        axes_arr = axes

    s = result.structure
    extent = _cell_edge_extent(s.x_nm, s.z_nm)
    intensity = np.abs(result.internal_wavefield.T) ** 2
    phase = np.angle(result.internal_wavefield.T)

    im0 = axes_arr[0].imshow(
        intensity,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
    )
    axes_arr[0].set_xlabel(r"$x\ (\mathrm{nm})$")
    axes_arr[0].set_ylabel(r"$z\ (\mathrm{nm})$")
    _style_clean_axes(axes_arr[0])
    if show_titles:
        axes_arr[0].set_title(
            rf"$|\psi(x, z)|^2$ ($\omega = {result.field_omega_deg:.1f}^\circ$)"
        )
    cb0 = plt.colorbar(im0, ax=axes_arr[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$|\psi(x, z)|^2$")

    im1 = axes_arr[1].imshow(
        phase,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes_arr[1].set_xlabel(r"$x\ (\mathrm{nm})$")
    axes_arr[1].set_ylabel(r"$z\ (\mathrm{nm})$")
    _style_clean_axes(axes_arr[1])
    if show_titles:
        axes_arr[1].set_title(
            rf"$\arg\psi(x, z)$ ($\omega = {result.field_omega_deg:.1f}^\circ$)"
        )
    cb1 = plt.colorbar(im1, ax=axes_arr[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"$\arg\psi(x, z)\ (\mathrm{rad})$")

    if overlay_structure and np.ptp(s.delta) > 0:
        mid_level = 0.5 * (float(np.min(s.delta)) + float(np.max(s.delta)))
        for ax in axes_arr:
            ax.contour(
                s.x_nm,
                s.z_nm,
                s.delta.T,
                levels=[mid_level],
                colors="cyan",
                linewidths=0.9,
                alpha=0.8,
            )
    if panel_labels is not None and len(panel_labels) >= 2:
        add_panel_label(axes_arr[0], panel_labels[0])
        add_panel_label(axes_arr[1], panel_labels[1])
    return axes_arr


def plot_simulation_comparison(
    born_result: BornSimulationResult2D | BornSimulationResult3D,
    multislice_result: MultisliceSimulationResult2D | MultisliceSimulationResult3D,
    orders: Sequence[int] = (1, 2, 3),
    x_axis: str = "Qz",
    ax: Any = None,
    log_scale: bool = True,
    *,
    show_title: bool = False,
    panel_label: str | None = None,
) -> Any:
    """Plot Born vs. Multislice Bragg rod efficiencies for selected `orders`."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)

    for idx, m in enumerate(orders):
        color = ORDER_PALETTE.get(
            abs(int(m)),
            plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % 6],
        )
        rod_born = born_result.bragg_rod(int(m), x_axis=x_axis)
        rod_ms = multislice_result.bragg_rod(int(m), x_axis=x_axis)
        ax.plot(
            rod_born.coords,
            rod_born.intensity,
            linestyle="--",
            color=color,
            linewidth=1.4,
            label=f"Born m={int(m):+d}",
        )
        ax.plot(
            rod_ms.coords,
            rod_ms.intensity,
            linestyle="-",
            color=color,
            linewidth=1.8,
            label=f"Multislice m={int(m):+d}",
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(
        r"$Q_z\ (\mathrm{nm}^{-1})$" if x_axis == "Qz" else r"$\omega\ (\mathrm{deg})$"
    )
    ax.set_ylabel("Diffraction efficiency")
    if show_title:
        ax.set_title("Born vs. Multislice")
    _style_clean_axes(ax)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    if panel_label:
        add_panel_label(ax, panel_label)
    return ax


def plot_born_vs_multislice_grid(
    comparisons: Sequence[tuple[BornSimulationResult2D, MultisliceSimulationResult2D]],
    *,
    orders: Sequence[int] = (4, 3, 2, 1, -1, -2, -3, -4),
    omega_samples_deg: Sequence[float] | None = None,
    dn_ref_values: Sequence[complex] | None = None,
    row_tags: Sequence[str] | None = None,
    scale_born_for_residual: bool = True,
    residual_lim_pct: float = 5.0,
    panel_labels: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Render a multi-row 4-column Born vs. Multislice comparison figure.

    Columns across each row:
    1. Normalized 2D cross-section ``Re[Delta n(x, z) / Delta n_A]`` (`cividis` with
       black vacuum above the surface).
    2. ``I vs Q_x`` ridge waterfall per sample tilt ``omega`` (`RdBu_r`), comparing
       Multislice (filled circles + solid curves) against Born (open circles).
    3. ``I vs Q_z`` stacked truncation rods per diffraction order ``m`` (`+4..-4`),
       colored by ``|m|`` using ``ORDER_PALETTE``, comparing Multislice vs. Born.
    4. 2D Bragg-rod Born shape residuals in ``(Q_x, Q_z)`` (`PuOr_r`, percent).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrows = len(comparisons)
    if nrows == 0:
        msg = (
            "At least one (BornSimulationResult2D, MultisliceSimulationResult2D) "
            "pair is required."
        )
        raise ValueError(msg)

    if figsize is None:
        figsize = (12.2, 3.05 * nrows + 1.35)

    fig = plt.figure(figsize=figsize, facecolor="white")
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[float(nrows), 0.28],
        hspace=0.12,
    )
    gs = outer_gs[0].subgridspec(
        nrows,
        4,
        width_ratios=[0.88, 1.08, 1.08, 1.08],
        hspace=0.14,
        wspace=0.22,
    )
    cb_gs = outer_gs[1].subgridspec(
        1,
        4,
        width_ratios=[0.88, 1.08, 1.08, 1.08],
        wspace=0.22,
    )
    axes = np.empty((nrows, 4), dtype=object)

    cmap_struct = plt.get_cmap("cividis")
    cmap_om = plt.get_cmap("RdBu_r")
    cmap_res = plt.get_cmap("PuOr_r")

    first_born = comparisons[0][0]
    if omega_samples_deg is None:
        om_min = float(np.min(first_born.omega_deg))
        om_max = float(np.max(first_born.omega_deg))
        omega_samples = np.linspace(om_min, om_max, 11)
    else:
        omega_samples = np.asarray(omega_samples_deg, dtype=np.float64)

    om_norm = mcolors.Normalize(
        vmin=float(np.min(omega_samples)),
        vmax=float(np.max(omega_samples)),
    )
    struct_norm = mcolors.Normalize(vmin=0.0, vmax=1.4)
    res_norm = mcolors.Normalize(
        vmin=-float(residual_lim_pct), vmax=float(residual_lim_pct)
    )

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for r_idx, (born_res, ms_res) in enumerate(comparisons):
        s = ms_res.structure
        ax_xs = fig.add_subplot(gs[r_idx, 0])
        ax_qx = fig.add_subplot(gs[r_idx, 1])
        ax_qz = fig.add_subplot(gs[r_idx, 2])
        ax_df = fig.add_subplot(gs[r_idx, 3])
        axes[r_idx, :] = [ax_xs, ax_qx, ax_qz, ax_df]

        # 1. Cross-section Re[dn / dn_A] with black vacuum
        dn_map = s.delta_n  # (nx, nz), note delta - i*beta
        if dn_ref_values is not None:
            dn_a = complex(dn_ref_values[r_idx])
        else:
            core_ix = int(np.argmin(np.abs(s.x_nm)))
            nz_mid = max(min(s.nz // 3, s.nz - 1), 0)
            dn_a = complex(dn_map[core_ix, nz_mid])
            if abs(dn_a) < 1e-12:
                dn_a = complex(np.max(s.delta), -float(np.max(s.beta)))

        rel_re = np.real(dn_map / dn_a).T  # (nz, nx)
        vac_mask = (np.abs(s.delta) + np.abs(s.beta)).T < 1e-7
        rgba = cmap_struct(struct_norm(np.clip(rel_re, 0.0, 1.4)))
        rgba[vac_mask] = [0.04, 0.04, 0.04, 1.0]

        extent_xz = _cell_edge_extent(s.x_nm, s.z_nm)
        ax_xs.imshow(
            rgba,
            origin="lower",
            aspect="auto",
            extent=extent_xz,
            interpolation="bilinear",
        )
        tag = row_tags[r_idx] if row_tags is not None else f"{s.energy_ev:g} eV"
        ax_xs.text(
            0.06,
            0.92,
            tag,
            transform=ax_xs.transAxes,
            color="white",
            fontsize=9,
            va="top",
            ha="left",
        )
        ax_xs.set_ylabel(r"$z\ (\mathrm{nm})$")
        if r_idx == nrows - 1:
            ax_xs.set_xlabel(r"$x\ (\mathrm{nm})$")
        else:
            ax_xs.tick_params(labelbottom=False)
        _style_clean_axes(ax_xs)

        # Match sampled omega indices on simulation grid
        om_indices = [
            int(np.argmin(np.abs(ms_res.omega_deg - om_val)))
            for om_val in omega_samples
        ]
        nz_orders = [
            int(m)
            for m in orders
            if int(m) != 0 and int(m) in ms_res.order_efficiencies
        ]
        nz_orders_sorted_qx = sorted(nz_orders)
        q0 = 2.0 * np.pi / s.effective_pitch_nm
        qx_pts = np.array([m * q0 for m in nz_orders_sorted_qx], dtype=np.float64)

        # 2. I vs Q_x, per omega (log10 waterfall)
        step_qx = 0.42
        for k_om, i_om in enumerate(om_indices):
            om_val = float(ms_res.omega_deg[i_om])
            color = cmap_om(om_norm(om_val))
            y_ms = (
                np.array(
                    [
                        np.log10(max(float(ms_res.order_efficiencies[m][i_om]), 1e-18))
                        for m in nz_orders_sorted_qx
                    ]
                )
                + k_om * step_qx
            )
            y_bn = (
                np.array(
                    [
                        np.log10(
                            max(float(born_res.order_efficiencies[m][i_om]), 1e-18)
                        )
                        for m in nz_orders_sorted_qx
                    ]
                )
                + k_om * step_qx
            )
            ax_qx.plot(qx_pts, y_ms, "-", color=color, lw=0.95, zorder=2)
            ax_qx.plot(
                qx_pts,
                y_ms,
                "o",
                color=color,
                ms=3.6,
                mec="none",
                zorder=3,
            )
            ax_qx.plot(
                qx_pts,
                y_bn,
                "o",
                mfc="none",
                mec=color,
                mew=0.85,
                ms=3.8,
                zorder=4,
            )

        ax_qx.set_yticks([])
        ax_qx.set_ylabel("I (offset)")
        qx_pad = 0.9 * q0
        ax_qx.set_xlim(float(np.min(qx_pts)) - qx_pad, float(np.max(qx_pts)) + qx_pad)
        if r_idx == nrows - 1:
            ax_qx.set_xlabel(r"$Q_x\ (\mathrm{nm}^{-1})$")
        else:
            ax_qx.tick_params(labelbottom=False)
        _style_clean_axes(ax_qx)

        # 3. I vs Q_z, per order (+4, +3, +2, +1, -1, -2, -3, -4)
        ordered_stack = list(orders)[::-1]  # bottom to top: -4 .. +4
        qz_all_min = min(
            float(np.min(ms_res.order_qz[m][om_indices])) for m in nz_orders
        )
        qz_all_max = max(
            float(np.max(ms_res.order_qz[m][om_indices])) for m in nz_orders
        )
        base_cursor = 0.0
        gap_decade = 0.55

        for m in ordered_stack:
            if m not in ms_res.order_efficiencies or m == 0:
                continue
            c_m = ORDER_PALETTE.get(abs(int(m)), "#3f3f46")
            qz_dense = ms_res.order_qz[m]
            s_idx = np.argsort(qz_dense)
            ms_dense_log = np.log10(
                np.maximum(ms_res.order_efficiencies[m][s_idx], 1e-18)
            )

            qz_samp = ms_res.order_qz[m][om_indices]
            s_samp = np.argsort(qz_samp)
            qz_samp_s = qz_samp[s_samp]
            ms_samp_log = np.log10(
                np.maximum(ms_res.order_efficiencies[m][om_indices][s_samp], 1e-18)
            )
            bn_samp_log = np.log10(
                np.maximum(born_res.order_efficiencies[m][om_indices][s_samp], 1e-18)
            )

            band_min = min(float(np.min(ms_samp_log)), float(np.min(bn_samp_log)))
            shift = base_cursor - band_min
            in_span = (qz_dense[s_idx] >= float(np.min(qz_samp_s)) - 1e-6) & (
                qz_dense[s_idx] <= float(np.max(qz_samp_s)) + 1e-6
            )
            ax_qz.plot(
                qz_dense[s_idx][in_span],
                ms_dense_log[in_span] + shift,
                "-",
                color=c_m,
                lw=0.95,
                alpha=0.85,
                zorder=2,
            )
            ax_qz.plot(
                qz_samp_s,
                ms_samp_log + shift,
                "o",
                color=c_m,
                ms=3.4,
                mec="none",
                zorder=3,
            )
            ax_qz.plot(
                qz_samp_s,
                bn_samp_log + shift,
                "o",
                mfc="none",
                mec=c_m,
                mew=0.85,
                ms=3.6,
                zorder=4,
            )
            ax_qz.text(
                qz_all_max + 0.025 * max(qz_all_max - qz_all_min, 0.2),
                float(np.mean(ms_samp_log + shift)),
                f"{int(m):+d}",
                color=c_m,
                fontsize=7.5,
                va="center",
                ha="left",
            )
            band_max = max(float(np.max(ms_samp_log)), float(np.max(bn_samp_log)))
            base_cursor = band_max + shift + gap_decade

        ax_qz.set_yticks([])
        qz_span = max(qz_all_max - qz_all_min, 0.2)
        ax_qz.set_xlim(qz_all_min - 0.08 * qz_span, qz_all_max + 0.20 * qz_span)
        if r_idx == nrows - 1:
            ax_qz.set_xlabel(r"$Q_z\ (\mathrm{nm}^{-1})$")
        else:
            ax_qz.tick_params(labelbottom=False)
        _style_clean_axes(ax_qz)

        # 4. 2D Born residual strips in (Q_x, Q_z)
        ms_vals = np.concatenate(
            [ms_res.order_efficiencies[m][om_indices] for m in nz_orders]
        )
        bn_vals = np.concatenate(
            [born_res.order_efficiencies[m][om_indices] for m in nz_orders]
        )
        if scale_born_for_residual:
            valid = (ms_vals > 0) & (bn_vals > 0)
            scale_k = (
                float(np.median(ms_vals[valid] / bn_vals[valid]))
                if np.any(valid)
                else 1.0
            )
        else:
            scale_k = 1.0

        max_res_pct = 0.0
        bar_half_w = 0.22 * q0
        qz_grid = np.linspace(qz_all_min, qz_all_max, 120)

        for m in nz_orders:
            qx_m = float(m) * q0
            qz_m = ms_res.order_qz[m][om_indices]
            s_i = np.argsort(qz_m)
            qz_s = qz_m[s_i]
            ms_s = ms_res.order_efficiencies[m][om_indices][s_i]
            bn_s = scale_k * born_res.order_efficiencies[m][om_indices][s_i]
            res_pct = 100.0 * (bn_s - ms_s) / np.maximum(ms_s, 1e-20)
            max_res_pct = max(max_res_pct, float(np.max(np.abs(res_pct))))

            in_rod = (qz_grid >= float(qz_s[0])) & (qz_grid <= float(qz_s[-1]))
            if np.count_nonzero(in_rod) < 2:
                continue
            qz_sub = qz_grid[in_rod]
            res_interp = np.interp(qz_sub, qz_s, res_pct)
            strip_2d = res_interp[:, None]
            ax_df.imshow(
                strip_2d,
                origin="lower",
                aspect="auto",
                extent=(
                    qx_m - bar_half_w,
                    qx_m + bar_half_w,
                    float(qz_sub[0]),
                    float(qz_sub[-1]),
                ),
                cmap=cmap_res,
                norm=res_norm,
                interpolation="bilinear",
            )

        ax_df.set_xlim(float(np.min(qx_pts)) - qx_pad, float(np.max(qx_pts)) + qx_pad)
        ax_df.set_ylim(qz_all_min - 0.12 * qz_span, qz_all_max + 0.24 * qz_span)
        ax_df.yaxis.tick_right()
        ax_df.yaxis.set_label_position("right")
        ax_df.set_ylabel(r"$Q_z\ (\mathrm{nm}^{-1})$")
        ax_df.text(
            0.5,
            0.93,
            f"max residual {max_res_pct:.1f}%",
            transform=ax_df.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#18181b",
        )
        if r_idx == nrows - 1:
            ax_df.set_xlabel(r"$Q_x\ (\mathrm{nm}^{-1})$")
        else:
            ax_df.tick_params(labelbottom=False)
        _style_clean_axes(ax_df)

        if panel_labels:
            for c_idx, ax_cell in enumerate((ax_xs, ax_qx, ax_qz, ax_df)):
                p_idx = r_idx * 4 + c_idx
                lbl = (
                    alphabet[p_idx]
                    if p_idx < len(alphabet)
                    else f"{alphabet[p_idx // 26 - 1]}{alphabet[p_idx % 26]}"
                )
                add_panel_label(ax_cell, lbl, x=-0.12 if c_idx > 0 else -0.18, y=1.02)

    # Bottom colorbar & legend strip
    cax0 = fig.add_subplot(cb_gs[0, 0])
    cax0.set_axis_off()
    cb_ax0 = cax0.inset_axes((0.08, 0.20, 0.84, 0.22))
    cb0 = fig.colorbar(
        plt.cm.ScalarMappable(norm=struct_norm, cmap=cmap_struct),
        cax=cb_ax0,
        orientation="horizontal",
    )
    cb0.set_label(r"$\mathrm{Re}[\Delta n / \Delta n_A]$", fontsize=8.5)
    cb0.ax.tick_params(labelsize=7.5)

    cax1 = fig.add_subplot(cb_gs[0, 1])
    cax1.set_axis_off()
    cb_ax1 = cax1.inset_axes((0.10, 0.20, 0.80, 0.22))
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=om_norm, cmap=cmap_om),
        cax=cb_ax1,
        orientation="horizontal",
    )
    cb1.set_label(r"$\omega\ (\mathrm{deg})$", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)

    cax2 = fig.add_subplot(cb_gs[0, 2])
    cax2.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markersize=5,
            label="multislice",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5,
            label="Born",
        ),
    ]
    cax2.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
    )

    cax3 = fig.add_subplot(cb_gs[0, 3])
    cax3.set_axis_off()
    cb_ax3 = cax3.inset_axes((0.08, 0.20, 0.84, 0.22))
    cb3 = fig.colorbar(
        plt.cm.ScalarMappable(norm=res_norm, cmap=cmap_res),
        cax=cb_ax3,
        orientation="horizontal",
        extend="both",
    )
    cb3.set_label("Born residual (%)", fontsize=8.5)
    cb3.ax.tick_params(labelsize=7.5)

    return fig, axes


__all__ = [
    "plot_born_vs_multislice_grid",
    "plot_multislice_wavefield",
    "plot_simulation_comparison",
    "plot_structure_2d",
    "plot_structure_3d",
]
