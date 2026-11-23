"""3D line-grating simulations: a clean developed line, a rough developed line,
and a rough latent image, with their Born-ensemble scattering in (Qx, Qy).

Writes ``docs/assets/images/sim_3d_gratings.png`` (used by the 3D guide).

Run: ``uv run python examples/simulation_3d_gratings.py``
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import cd_reduce as cdr

OUT = Path(__file__).resolve().parents[1] / "docs/assets/images/sim_3d_gratings.png"

# Cross sections (same objects the reduction uses).
developed = cdr.DevelopedLine(
    L=80.0, w=45.0, s=1.0, h_d=30.0, dn_A=-2.0e-3 + 1.5e-3j, energy_ev=285.0
)
latent = cdr.LatentFilm(
    L=80.0,
    w=45.0,
    s=2.5,
    h=40.0,
    dh=4.0,
    lam=25.0,
    r=1.15 + 0.05j,
    dn_A=-2.0e-3 + 1.5e-3j,
    energy_ev=285.0,
)

# Two-sided edge PSDs of f = Qy / 2pi (nm^3 vs nm^-1).
psd_lpr = partial(cdr.psd_palasantzas, sigma_nm=1.5, xi_nm=25.0, hurst=0.7)
psd_lwr = partial(cdr.psd_palasantzas, sigma_nm=2.0, xi_nm=12.0, hurst=0.5)

GRID = {"n_lines": 8, "ly_nm": 1024.0, "dx_nm": 1.0, "dy_nm": 4.0, "dz_nm": 1.0}


def maker(xs, rough: bool):
    def make(rng):
        return cdr.build_rough_grating_3d(
            xs,
            psd_lpr=psd_lpr if rough else None,
            psd_lwr=psd_lwr if rough else None,
            rng=rng,
            **GRID,
        )

    return make


CASES = [
    ("Clean developed line", maker(developed, rough=False)),
    ("Rough developed line", maker(developed, rough=True)),
    ("Rough latent image", maker(latent, rough=True)),
]


def total_intensity(res):
    """Flux-correct coherent (Bragg) + diffuse intensity per mode at the first omega."""
    coh = np.abs(res.coherent[0]) ** 2 * res.flux_factor[0]
    dif = np.nan_to_num(res.diffuse_flux[0])
    tot = coh + dif
    tot[np.argmin(np.abs(res.qx)), np.argmin(np.abs(res.qy))] = np.nan  # zero order
    return tot


def edges(c):
    c = np.asarray(c, dtype=float)
    h = 0.5 * (c[1] - c[0])
    return c[0] - h, c[-1] + h


def main() -> None:
    fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.5), constrained_layout=True)
    letters = iter("abcdefghi")

    for row, (title, make) in enumerate(CASES):
        struct = make(np.random.default_rng(7))
        dn = struct.delta * 1e3  # delta x 10^3
        iy = struct.ny // 2
        x0 = edges(struct.x_nm)[0]
        x_lim = (x0, x0 + 3 * struct.pitch_nm)
        if row < 2:
            lo, hi = 0.0, 2.6
        else:  # latent image: stretch the colour scale over the chemical contrast
            lo, hi = np.percentile(dn[dn > 0.5 * dn.max()], [0.5, 99.5])

        # (x, z) cross-section at one y
        a = ax[row, 0]
        im = a.imshow(
            dn[:, iy, :].T,
            origin="lower",
            aspect="auto",
            cmap="cividis",
            extent=(*edges(struct.x_nm), *edges(struct.z_nm)),
            vmin=lo,
            vmax=hi,
        )
        a.set_xlim(*x_lim)
        a.set_xlabel(r"$x$ (nm)")
        a.set_ylabel(r"$z$ (nm)")
        a.set_title(f"{title}: cross-section", fontsize=10)
        fig.colorbar(im, ax=a, label=r"$\delta \times 10^3$")

        # (x, y) top view at mid-height
        b = ax[row, 1]
        iz = struct.nz // 2
        im = b.imshow(
            dn[:, :, iz].T,
            origin="lower",
            aspect="auto",
            cmap="cividis",
            extent=(*edges(struct.x_nm), *edges(struct.y_nm)),
            vmin=lo,
            vmax=hi,
        )
        b.set_xlim(*x_lim)
        b.set_xlabel(r"$x$ (nm)")
        b.set_ylabel(r"$y$ (nm)")
        b.set_title(f"top view at $z$ = {struct.z_nm[iz]:.0f} nm", fontsize=10)
        fig.colorbar(im, ax=b, label=r"$\delta \times 10^3$")

        # Born ensemble scattering at omega = 0
        res = cdr.simulate_rough_ensemble(
            make,
            [0.0],
            engine="born",
            n_realisations=8,
            seed=11,
            qx_max=0.4,
            qy_max=0.2,
        )
        c = ax[row, 2]
        tot = total_intensity(res)
        im = c.imshow(
            tot.T,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            extent=(*edges(res.qx), *edges(res.qy)),
            norm=LogNorm(vmin=np.nanmax(tot) * 1e-9, vmax=np.nanmax(tot)),
        )
        c.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
        c.set_ylabel(r"$Q_y$ (nm$^{-1}$)")
        c.set_title(r"Born ensemble, $\omega = 0$ (8 realisations)", fontsize=10)
        fig.colorbar(im, ax=c, label="intensity per mode")

        for a_ in ax[row]:
            a_.text(
                -0.12,
                1.04,
                next(letters),
                transform=a_.transAxes,
                fontsize=12,
                fontweight="bold",
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
