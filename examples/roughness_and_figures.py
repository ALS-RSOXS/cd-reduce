"""Reduction figures and LER / LWR / LPR PSD extraction on a synthetic tilt series.

Builds a synthetic rough-grating tilt series with known line-placement (LPR)
and line-width (LWR) PSDs, then runs the standard outputs:

1. Thin detector strips about the ``Qy = 0`` line per omega.
2. ``(Qx, Qz)`` butterfly maps, exact Ewald vs small-angle, with ``Qy = 0`` trace.
3. 2D maps along every plane: ``(Qx, Qz)``, ``(Qx, Qy)``, ``(Qy, Qz)``, ``(Qr, Qz)``.
4. ``I`` vs ``Qx`` per omega (colour bar + omega labels), ``I`` vs ``Qr``.
5. ``I_m`` vs ``Qz`` per order, labelled.
6. ``I(Qy)`` diffuse per order, both as a detector line cut at omega = 0 and
   along a constant-``Qz`` trace through the series.
7. Multi-order Born decomposition into LER / LWR / LPR PSDs with Mack fits.

    uv run python examples/roughness_and_figures.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cd_reduce as cdr
from cd_reduce.simulation.diffuse import synthetic_rough_grating_frame

PITCH, WIDTH, HEIGHT = 100.0, 32.0, 60.0


def S_LPR(q):
    return cdr.psd_mack(q, 20.0, 25.0, 0.5)


def S_LWR(q):
    return cdr.psd_mack(q, 60.0, 15.0, 0.5)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    geo = cdr.Geometry(u0_px=256, v0_px=256, Lb_px=1200, Lambda_nm=PITCH)
    omegas = np.arange(-20.0, 20.1, 2.0)
    frames = [
        synthetic_rough_grating_frame(
            geo,
            om,
            line_width_nm=WIDTH,
            height_nm=HEIGHT,
            psd_lpr=S_LPR,
            psd_lwr=S_LWR,
            flux=3e5,
            rng=k,
        )
        for k, om in enumerate(omegas)
    ]
    G = geo.g_inv_nm
    kw = {"bg_mode": "none"}

    # 1. Detector strips
    strips = {}
    for fr in frames:
        if fr.omega_deg in (0.0, 10.0, 20.0):
            qm = cdr.compute_pixel_q(
                geo, *cdr.frame_pixel_grid(fr.image.shape), fr.energy_ev, fr.omega_deg
            )
            strips[fr.omega_deg] = cdr.extract_line_strip(
                geo, fr.image, qm.Qx, qm.Qy, half_width_px=24
            )
    cdr.plot_detector_strips(strips).savefig(out / "01_detector_strips.png", dpi=150)

    # 2. Butterfly: exact vs small-angle
    grid = [
        [
            cdr.map_2d(
                frames,
                geo,
                x_domain="Qx",
                y_domain="Qz",
                bins=(260, 220),
                x_range=(-4.4 * G, 4.4 * G),
                y_range=(-0.3, 0.3),
                approximation=a,
                qy_cut_over_g=0.05,
                **kw,
            )
            for a in ("correct", "small_angle")
        ]
    ]
    fig = cdr.plot_qmap_grid(
        grid,
        labels=[["synthetic  exact", "synthetic  small-angle"]],
        geometries=[geo],
        ewald_omega_deg=0.0,
        detector_shape=(512, 512),
    )
    fig.savefig(out / "02_butterfly_exact_vs_small_angle.png", dpi=150)

    # 3. Maps along every plane
    planes = [
        cdr.map_2d(
            frames,
            geo,
            x_domain="Qx",
            y_domain="Qz",
            bins=(220, 180),
            qy_cut_over_g=0.05,
            **kw,
        ),
        cdr.map_2d(
            frames,
            geo,
            x_domain="Qx",
            y_domain="Qy",
            bins=(220, 180),
            qz_cut=(-0.01, 0.01),
            **kw,
        ),
        cdr.map_2d(
            frames,
            geo,
            x_domain="Qy",
            y_domain="Qz",
            bins=(180, 180),
            qx_cut=(0.85 * G, 1.15 * G),
            **kw,
        ),
        cdr.map_2d(
            frames,
            geo,
            x_domain="signed_Qr",
            y_domain="Qz",
            bins=(220, 180),
            qy_cut_over_g=0.05,
            **kw,
        ),
    ]
    fig = cdr.plot_qmap_grid(
        [planes[:2], planes[2:]],
        labels=[["Qx-Qz", "Qx-Qy  |Qz|<0.01"], ["Qy-Qz  m=+1", "signed Qr-Qz"]],
    )
    fig.savefig(out / "03_planes.png", dpi=150)

    # 4/5. 1D specular traces: I vs Qx per omega, I vs Qr, I_m vs Qz
    by_om = cdr.extract_qx_by_omega(frames, geo, domain="Qx", bins=400, **kw)
    by_om_r = cdr.extract_qx_by_omega(frames, geo, domain="Qr", bins=300, **kw)
    rods = cdr.extract_order_rods_qz(
        frames, geo, orders=(-4, -3, -2, -1, 1, 2, 3, 4), bins=24, **kw
    )
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
    cdr.plot_traces_by_omega(by_om, ax=axs[0], offset_decades=0.35)
    cdr.plot_traces_by_omega(by_om_r, ax=axs[1], offset_decades=0.35)
    cdr.plot_rods_by_order(rods, ax=axs[2])
    fig.tight_layout()
    fig.savefig(out / "04_specular_1d.png", dpi=150)

    # 6. I(Qy): line cut at omega = 0 vs constant-Qz trace
    qe = np.geomspace(0.015, 0.18, 10)
    orders = (-4, -3, -2, -1, 1, 2, 3, 4)
    r_cut = cdr.extract_diffuse_ratio(
        frames, geo, orders=orders, qy_edges=qe, mode="line_cut", omega=0.0, **kw
    )
    r_qz = cdr.extract_diffuse_ratio(
        frames,
        geo,
        orders=orders,
        qy_edges=qe,
        mode="qz_trace",
        qz_range=(0.03, 0.06),
        **kw,
    )
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    cdr.plot_diffuse_ratio(r_cut, ax=axs[0])
    cdr.plot_diffuse_ratio(r_qz, ax=axs[1])
    fig.tight_layout()
    fig.savefig(out / "05_diffuse_qy.png", dpi=150)

    # 7. LER / LWR / LPR
    res = cdr.decompose_edge_psd(r_cut, WIDTH)
    fits = {k: res.fit(k, model="mack") for k in ("LER", "LWR", "LPR")}
    qq = np.geomspace(0.01, 0.25, 100)
    ref = {"true LPR": (qq, S_LPR(qq)), "true LWR": (qq, S_LWR(qq))}
    cdr.plot_edge_psds(res, fits=fits, reference=ref).savefig(
        out / "06_edge_psds.png", dpi=150
    )
    for k, f in fits.items():
        s_band, ds = res.sigma(k)  # type: ignore[arg-type]
        print(
            f"{k}: xi={f.params['xi_nm']:.1f} nm  H={f.params['hurst']:.2f}  "
            f"3sigma(band)={3 * s_band:.2f}+-{3 * ds:.2f} nm  3sigma(model)={3 * f.sigma_nm():.2f} nm"
        )
    print(
        res.table[
            ["qy_center", "S_LPR", "S_LWR", "S_LER", "rho_LR", "condition"]
        ].round(3)
    )
    print(f"figures in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures_roughness"))
