"""3D line-grating simulations: a clean developed line, a rough developed line,
and a rough latent image.

Writes two figures used by the 3D guide:

- ``docs/assets/images/sim_3d_structures.png``: 3D renders of the three gratings
  (roughness and latent relief exaggerated for display only);
- ``docs/assets/images/sim_3d_scattering.png``: their Born-ensemble intensity in
  (Qx, Qy) at omega = 0, simulated with the true (unexaggerated) roughness.

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

IMAGES = Path(__file__).resolve().parents[1] / "docs/assets/images"

# Cross sections (the same objects the roughness reduction uses).
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
SIGMA_LPR, SIGMA_LWR = 1.5, 2.0


def psds(scale: float = 1.0):
    lpr = partial(
        cdr.psd_palasantzas, sigma_nm=SIGMA_LPR * scale, xi_nm=25.0, hurst=0.7
    )
    lwr = partial(
        cdr.psd_palasantzas, sigma_nm=SIGMA_LWR * scale, xi_nm=12.0, hurst=0.5
    )
    return lpr, lwr


def maker(xs, rough: bool, *, scale: float = 1.0, **grid):
    lpr, lwr = psds(scale)
    grid = {
        "n_lines": 8,
        "ly_nm": 1024.0,
        "dx_nm": 1.0,
        "dy_nm": 4.0,
        "dz_nm": 1.0,
    } | grid

    def make(rng):
        return cdr.build_rough_grating_3d(
            xs,
            psd_lpr=lpr if rough else None,
            psd_lwr=lwr if rough else None,
            rng=rng,
            **grid,
        )

    return make


CASES = [
    ("Clean developed line", developed, False, 1.0),
    ("Rough developed line", developed, True, 1.0),
    ("Rough latent image", latent, True, 3.0),  # last value: relief exaggeration
]
DRAW_ROUGHNESS = 2.3  # roughness exaggeration in the renders only
RESIST = "#4a6fb5"  # unexposed resist, the low end of the latent colour map


def render_structures() -> None:
    fig = plt.figure(figsize=(16.5, 5.6), facecolor="#06121a")
    for i, (title, xs, rough, relief) in enumerate(CASES):
        make = maker(
            xs,
            rough,
            scale=DRAW_ROUGHNESS,
            n_lines=5,
            ly_nm=512.0,
            dx_nm=0.5,
            dy_nm=4.0,
        )
        struct = make(np.random.default_rng(3))
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        cdr.plot_structure_3d_surface(
            struct,
            ax=ax,
            relief_scale=relief,
            color=RESIST if xs is developed else None,
        )
        note = "" if not rough else f"roughness drawn {DRAW_ROUGHNESS:g}x"
        if relief != 1:
            note += f", relief {relief:g}x"
        ax.set_title(f"{title}\n{note}", color="#e4e4e7", fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04, wspace=0.08)
    fig.savefig(
        IMAGES / "sim_3d_structures.png", dpi=150, facecolor=fig.get_facecolor()
    )


def total_intensity(res):
    """Flux-correct coherent (Bragg) + diffuse intensity per mode at the first omega."""
    coh = np.abs(res.coherent[0]) ** 2 * res.flux_factor[0]
    tot = coh + np.nan_to_num(res.diffuse_flux[0])
    tot[np.argmin(np.abs(res.qx)), np.argmin(np.abs(res.qy))] = np.nan  # zero order
    return tot


def edges(c):
    c = np.asarray(c, dtype=float)
    h = 0.5 * (c[1] - c[0])
    return c[0] - h, c[-1] + h


def render_scattering() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    for ax, (title, xs, rough, _) in zip(axes, CASES, strict=True):
        res = cdr.simulate_rough_ensemble(
            maker(xs, rough),
            [0.0],
            engine="born",
            n_realisations=8,
            seed=11,
            qx_max=0.4,
            qy_max=0.2,
        )
        tot = total_intensity(res)
        im = ax.imshow(
            tot.T,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            extent=(*edges(res.qx), *edges(res.qy)),
            norm=LogNorm(vmin=np.nanmax(tot) * 1e-9, vmax=np.nanmax(tot)),
        )
        ax.set_xlabel(r"$Q_x$ (nm$^{-1}$)")
        ax.set_ylabel(r"$Q_y$ (nm$^{-1}$)")
        ax.set_title(f"{title}: Born ensemble, $\\omega = 0$", fontsize=10)
        fig.colorbar(im, ax=ax, label="intensity per mode")
    fig.savefig(IMAGES / "sim_3d_scattering.png", dpi=150, facecolor="white")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    render_structures()
    render_scattering()
    print(f"wrote {IMAGES / 'sim_3d_structures.png'} and sim_3d_scattering.png")


if __name__ == "__main__":
    main()
