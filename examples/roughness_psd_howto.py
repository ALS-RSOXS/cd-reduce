"""Roughness PSDs from diffuse scattering: the HOWTO reduction on synthetic frames.

Renders a rotation series of detector frames from the first-order two-edge model
(known LPR and LWR PSDs, a latent-film cross section, Bragg orders, Poisson noise),
then runs the reduction exactly as for measured data:

frames -> ``reduce_frame`` -> ``fit_psds`` -> ``purity_bands`` / ``isolation_row_psds``
-> ``PsdRecord`` (via ``CdRsoxsLoader.extract_psds``) -> ``map_reduced``.

Figures are written to ``docs/assets/images/roughness/``. Public API only.
"""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")

import cd_reduce as cdr

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "images" / "roughness"
HC_EV_NM = 1239.84198
E = 285.0
SCALE = 5e13  # incident photons per second (known here; fit with order_scale on data)
ADU = 27.0  # ADU per detected photon
GEOMETRY = cdr.Geometry(u0_px=256, v0_px=256, Lb_px=985, beta_deg=1.0, Lambda_nm=80)
SHAPE = (512, 512)
OMEGAS = np.arange(-20.0, 21.0, 2.0)
XS = cdr.LatentFilm(
    L=80.0, w=45.0, s=2.5, h=40.0, dh=4.0, lam=25.0,
    r=1.15 + 0.05j, dn_A=-2e-3 + 1.5e-3j, energy_ev=E,
)  # fmt: skip
PSD_LPR = partial(cdr.psd_palasantzas, sigma_nm=1.0, xi_nm=25.0, hurst=0.7)
PSD_LWR = partial(cdr.psd_palasantzas, sigma_nm=1.5, xi_nm=12.0, hurst=0.5)


def render_frame(omega_deg: float, rng: np.random.Generator) -> cdr.Frame:
    """First-order two-edge diffuse plus Gaussian Bragg spots, Poisson counts in ADU."""
    k = 2 * np.pi * E / HC_EV_NM
    uu, vv = cdr.frame_pixel_grid(SHAPE)
    qm = cdr.compute_pixel_q(GEOMETRY, uu, vv, E, omega_deg, approximation="correct")
    Qx, Qy, Qz = (np.asarray(a, dtype=float) for a in (qm.Qx, qm.Qy, qm.Qz))
    W = np.radians(omega_deg + GEOMETRY.omega0_deg)
    A = cdr.absorption_factor(XS.mu_h, W, cdr.cos_theta_out(qm, k, W))
    norm = cdr.howto_prefactor(k, XS, W) * np.asarray(qm.solid_angle_sr) * A
    phi, f = Qx * XS.w / 2, Qy / (2 * np.pi)
    M = 4 * np.sin(phi) ** 2 * PSD_LPR(f) + np.cos(phi) ** 2 * PSD_LWR(f)
    lam = SCALE * M * XS.gamma2(Qz) * np.exp(-((Qx * XS.s) ** 2)) * norm
    # Bragg orders: integrated efficiency spread over a Gaussian spot of rms sigma_Q.
    sigma_q, area = 0.0015, cdr.pixel_q_area(Qx, Qy)
    sigma_ler = np.hypot(1.0, 1.5 / 2)  # full sigma_LER of the two input PSDs, nm
    for m in (-4, -3, -2, -1, 1, 2, 3, 4):
        r2 = (Qx - m * 2 * np.pi / XS.L) ** 2 + Qy**2
        near = r2 < (8 * sigma_q) ** 2
        if near.any():
            eta = cdr.order_efficiency(
                XS, m, omega_deg, E, Qz[near], sigma_ler=sigma_ler
            )
            spot = np.exp(-r2[near] / (2 * sigma_q**2)) / (2 * np.pi * sigma_q**2)
            lam[near] += SCALE * eta * spot * area[near]
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    img = rng.poisson(lam).astype(float) * ADU
    img[np.hypot(uu - GEOMETRY.u0_px, vv - GEOMETRY.v0_px) <= 30] = np.nan  # beamstop
    return cdr.Frame.from_array(img, energy_ev=E, omega_deg=float(omega_deg))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frames = [render_frame(om, rng) for om in OMEGAS]

    # 1. Reduce every frame to per-pixel M (nm^3) and fit both PSDs per |f| bin.
    pixels = cdr.ReducedPixels.concat(
        [
            cdr.reduce_frame(
                fr, GEOMETRY, XS, scale=SCALE, adu_per_photon=ADU, frame_id=i
            )
            for i, fr in enumerate(frames)
        ]
    )
    qx_max = 0.3 / np.hypot(1.0, 1.5 / 2)  # first-order limit |Qx| sigma_LER <= 0.3
    table = cdr.fit_psds(pixels, w=XS.w, qx_max=qx_max)
    fin = np.isfinite(table.S_lpr) & np.isfinite(table.S_lwr)
    fc = table.f_center[fin]

    # 2. Isolation-row check with 90 % purity bands from a prior (here the inputs).
    bands = cdr.purity_bands(XS.w, PSD_LPR, PSD_LWR, (fc[0], fc[-1]), 0.9, qx_max)
    rows = cdr.isolation_row_psds(pixels, XS.w, bands, table.f_edges, qx_max=qx_max)

    # 3. The same reduction through the loader, returning a self-describing record.
    loader = cdr.CdRsoxsLoader(frames=frames, geometry=GEOMETRY, bg_mode="none")
    record = loader.extract_psds(
        {E: XS}, scale=SCALE, adu_per_photon=ADU, qx_max=qx_max
    )
    with tempfile.TemporaryDirectory() as tmp:
        record.save(Path(tmp) / "psds.csv")  # CSV table with a JSON metadata header
        record = cdr.PsdRecord.load(Path(tmp) / "psds.csv")
    for mode, truth in (("lpr", PSD_LPR), ("lwr", PSD_LWR)):
        sig, err = record.sigma_win[mode]
        exact = cdr.sigma_from_psd(truth, fc[0], fc[-1])
        print(f"sigma_win {mode}: {sig:.3f} +- {err:.4f} nm (input {exact:.3f} nm)")

    # Figure 1: one frame and its reduced M on the detector.
    i0 = int(np.argmin(np.abs(OMEGAS)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(
        np.log10(np.maximum(frames[i0].image, ADU)), origin="lower", cmap="magma"
    )
    ax[0].set_title(r"synthetic frame, $\omega$ = 0°, log$_{10}$ ADU")
    one = pixels.select(pixels.frame_id == i0)
    sc = ax[1].scatter(one.Qx, one.Qy, c=np.log10(np.clip(one.M, 1e-1, None)), s=0.3)
    ax[1].set(xlabel=r"$Q_x$ (nm$^{-1}$)", ylabel=r"$Q_y$ (nm$^{-1}$)",
              title=r"reduced $M$ (log$_{10}$ nm$^3$), masks applied")  # fmt: skip
    fig.colorbar(sc, ax=ax[1])
    fig.tight_layout()
    fig.savefig(OUT / "howto_frame_and_M.png", dpi=120)
    plt.close(fig)

    # Figure 2: M against Qx in one f bin, with the two-edge fit and isolation rows.
    j = int(np.flatnonzero(fin)[len(fc) // 3])
    sel = (pixels.f >= table.f_edges[j]) & (pixels.f < table.f_edges[j + 1])
    qx = np.linspace(0, np.abs(pixels.Qx[sel]).max(), 400)
    model = (
        4 * np.sin(qx * XS.w / 2) ** 2 * table.S_lpr[j]
        + np.cos(qx * XS.w / 2) ** 2 * table.S_lwr[j]
    )
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(np.abs(pixels.Qx[sel]), pixels.M[sel], ".", ms=1, alpha=0.3, label="pixels")
    ax.plot(qx, model, "k-", label="global fit")
    for kk in range(1, 6):
        ax.axvline(kk * np.pi / XS.w, color="C1" if kk % 2 else "C2", ls="--", lw=0.8)
    ax.axvline(qx_max, color="0.5", lw=1, label=r"$|Q_x|\sigma_{LER}$ = 0.3")
    title = f"f = {table.f_center[j]:.3f} nm$^{{-1}}$; rows at $k\\pi/w$"
    ax.set(xlabel=r"$|Q_x|$ (nm$^{-1}$)", ylabel=r"$M$ (nm$^3$)", yscale="log")
    ax.set_title(title + " (odd: LPR, even: LWR)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "howto_M_vs_qx.png", dpi=120)
    plt.close(fig)

    # Figure 3: recovered PSDs against the inputs.
    fig, ax = plt.subplots(figsize=(6, 4))
    truths = {
        "lpr": PSD_LPR,
        "lwr": PSD_LWR,
        "ler": lambda f: PSD_LPR(f) + PSD_LWR(f) / 4,
    }
    for c, (mode, truth) in enumerate(truths.items()):
        S, var = getattr(table, f"S_{mode}")[fin], getattr(table, f"var_{mode}")[fin]
        ax.errorbar(
            fc,
            S,
            yerr=np.sqrt(var),
            fmt="o",
            ms=3,
            color=f"C{c}",
            label=f"{mode.upper()} fit",
        )
        ax.plot(fc, truth(fc), "-", color=f"C{c}", lw=1)
        for kk, r in rows.get(mode, {}).items():
            ax.plot(table.f_center, r["S"], "x", ms=4, color=f"C{c}", alpha=0.6,
                    label=f"{mode.upper()} row k={kk}")  # fmt: skip
    ax.set(xscale="log", yscale="log", xlabel=r"$f = Q_y/2\pi$ (nm$^{-1}$)")
    ax.set(ylabel=r"two-sided PSD (nm$^3$)", title="recovered (markers), input (lines)")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "howto_psds.png", dpi=120)
    plt.close(fig)

    # Figure 4: reduced Qx-Qz map over the rotation series.
    qmap = cdr.map_reduced(
        pixels, "qx-qz", slab_center=0.05, slab_halfwidth=0.02, bin_q=(0.004, 0.004)
    )
    ax = cdr.plot_reduced_map(qmap, w=XS.w)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "howto_qxqz_map.png", dpi=120)
    plt.close(ax.figure)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
