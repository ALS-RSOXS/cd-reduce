"""Static Ewald-sphere figures for the docs, computed with the package geometry.

Writes:

- ``docs/assets/images/ewald_sphere_rods.png``: the Ewald sphere near the origin
  (a detector frame mapped to reciprocal space with ``compute_pixel_q``), the
  grating rods and their intersections, plus the (Qx, Qz) butterfly of an omega
  sweep. It is the fallback for the interactive figure and the home-page figure.
- ``docs/assets/images/ewald_exact_vs_small_angle.png``: Qz(omega) on each rod,
  exact Ewald vs flat-detector (small-angle), and their difference.

Run: ``uv run python examples/ewald_figures.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cd_reduce.reduction.geometry import Geometry, compute_pixel_q, ewald_trace_qxqz

IMAGES = Path(__file__).resolve().parents[1] / "docs/assets/images"
ENERGY, PITCH, OMEGA, MMAX = 285.0, 80.0, 15.0, 4
ORDER_COLORS = ["#0ea5e9", "#f59e0b", "#10b981", "#f43f5e"]
SPHERE, ARC, MUTED = "#7cc4e4", "#0b4f71", "#52606d"
G = 2 * np.pi / PITCH
X = (MMAX + 0.7) * G
GEO = Geometry(
    u0_px=511.5, v0_px=511.5, Lb_px=400.0, line_rot_deg=90.0, Lambda_nm=PITCH
)


def rod_qz(m: int, omega: float, approximation: str = "correct") -> float:
    """Qz where rod m meets the frame at ``omega``, from the package trace."""
    qx, qz = ewald_trace_qxqz(GEO, omega, ENERGY, approximation=approximation)
    return float(np.interp(m * G, qx, qz))


def sphere_patch(omega: float):
    """Reciprocal-space sheet seen by one frame: compute_pixel_q on a pixel grid."""
    u, v = np.meshgrid(np.linspace(457, 566, 25), np.linspace(-150, 1173, 81))
    q = compute_pixel_q(GEO, u, v, ENERGY, omega)
    return q.Qx, q.Qy, q.Qz


def sphere_and_rods() -> None:
    fig = plt.figure(figsize=(11.5, 4.6), facecolor="white")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    qx, qy, qz = sphere_patch(OMEGA)
    keep = (np.abs(qx) <= X) & (np.abs(qy) <= 0.55 * X)
    qx, qy, qz = (np.where(keep, a, np.nan) for a in (qx, qy, qz))
    ax.plot_surface(qx, qy, qz, color=SPHERE, alpha=0.28, linewidth=0, shade=False)
    ax.plot_wireframe(qx, qy, qz, rstride=5, cstride=4, color=SPHERE, lw=0.6, alpha=0.9)
    zr = 0.32
    for m in range(-MMAX, MMAX + 1):
        col = MUTED if m == 0 else ORDER_COLORS[abs(m) - 1]
        ax.plot([m * G] * 2, [0, 0], [-zr, zr], color=col, lw=1.4, alpha=0.8)
        if m:
            ax.text(m * G, 0, -zr - 0.04, f"{m:+d}", color=col, ha="center", fontsize=8)
            ax.scatter(
                [m * G],
                [0],
                [rod_qz(m, OMEGA)],
                color=col,
                s=34,
                edgecolor="white",
                linewidth=1.2,
                depthshade=False,
                zorder=5,
            )
    tq, tz = ewald_trace_qxqz(GEO, OMEGA, ENERGY)
    sel = np.abs(tq) <= X
    ax.plot(tq[sel], np.zeros(sel.sum()), tz[sel], color=ARC, lw=2.2)
    ax.set_box_aspect((2 * X, 1.1 * X, 2 * zr))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
    ax.set_ylabel(r"$Q_y$")
    ax.set_zlabel(r"$Q_z$")
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
    ax.set_title(
        rf"Ewald sphere near the origin, $\omega = {OMEGA:g}^\circ$", fontsize=10
    )

    ax2 = fig.add_subplot(1, 2, 2)
    for w in np.arange(-40, 41, 1.0):
        a, b = ewald_trace_qxqz(GEO, w, ENERGY)
        s = np.abs(a) <= X
        ax2.plot(a[s], b[s], color=SPHERE, lw=0.8, alpha=0.35)
    for m in range(-MMAX, MMAX + 1):
        col = MUTED if m == 0 else ORDER_COLORS[abs(m) - 1]
        ax2.axvline(m * G, color=col, lw=1.1, alpha=0.6)
        if m:
            ax2.plot(
                m * G, rod_qz(m, OMEGA), "o", color=col, ms=6, mec="white", mew=1.2
            )
    ax2.plot(tq[sel], tz[sel], color=ARC, lw=2.2, label="this frame (exact)")
    xx = np.array([-X, X])
    ax2.plot(
        xx,
        xx * np.tan(np.radians(OMEGA)),
        "--",
        color="#1f2933",
        lw=1,
        label="small-angle",
    )
    ax2.set_xlim(-X, X)
    ax2.set_ylim(-0.37, 0.37)
    ax2.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
    ax2.set_ylabel(r"$Q_z$ (nm$^{-1}$)")
    ax2.set_title(
        r"$(Q_x, Q_z)$: one frame and an $\omega$ sweep of $\pm 40^\circ$", fontsize=10
    )
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(0.01, 0.02, f"{ENERGY:g} eV, pitch {PITCH:g} nm", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(IMAGES / "ewald_sphere_rods.png", dpi=150)
    plt.close(fig)


def exact_vs_small_angle() -> None:
    omegas = np.linspace(-40, 40, 161)
    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    for m in range(1, MMAX + 1):
        col = ORDER_COLORS[m - 1]
        ex = np.array([rod_qz(m, w) for w in omegas])
        sa = np.array([rod_qz(m, w, "small_angle") for w in omegas])
        ax.plot(omegas, ex, color=col, lw=1.8, label=f"m = {m}")
        ax.plot(omegas, sa, "--", color=col, lw=1.1)
        axd.plot(omegas, ex - sa, color=col, lw=1.6)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(r"$Q_z$ on rod $m$ (nm$^{-1}$)")
    ax.set_title(
        "Exact Ewald (solid) vs small-angle flat detector (dashed)", fontsize=10
    )
    ax.legend(fontsize=8, frameon=False, ncol=4)
    axd.axhline(0, color=MUTED, lw=0.6)
    axd.set_xlabel(r"sample tilt $\omega$ (deg)")
    axd.set_ylabel(r"exact $-$ small-angle")
    fig.savefig(IMAGES / "ewald_exact_vs_small_angle.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    sphere_and_rods()
    exact_vs_small_angle()
    print("wrote ewald_sphere_rods.png and ewald_exact_vs_small_angle.png")


if __name__ == "__main__":
    main()
