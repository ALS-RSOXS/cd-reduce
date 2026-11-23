"""Full-size validation study (not collected by pytest).

Builds Born ensembles of explicit rough edges for every reference case at every
energy over the full omega range (16 lines, Ly 2048 nm, 8 realisations), renders
Poisson detector frames, reduces them with the package and compares the
recovered PSDs with the realised spectra of the generated edges. The ``both``
case is then repeated with the multislice engine (with the epsilon Born
reference). Finally it runs the large-ensemble isolation report.

Writes to ``tests/validation/results/``: ``summary.csv``, ``isolation_systematic.json``
and figures (recovered vs input PSDs per case, M vs Qx at fixed f, MS/Born ratio).

Usage::

    uv run python tests/validation/run_study.py           # full size (long)
    uv run python tests/validation/run_study.py --quick   # small config, smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.validation import reference as ref  # noqa: E402
from tests.validation.forward.ensemble_frames import (  # noqa: E402
    frame_omega_deg,
    render_ensemble_frame,
)
from tests.validation.test_model_error import isolation_systematic  # noqa: E402

from cd_reduce import (  # noqa: E402
    ReducedPixels,
    absorption_factor,
    build_rough_grating_3d,
    fit_psds,
    reduce_frame,
    sigma_window,
    simulate_rough_ensemble,
)

RESULTS = Path(__file__).parent / "results"
MODES = ("lpr", "lwr", "ler")


def full_config() -> dict:
    """Reference size: 16 lines, Ly 2048 nm, 8 realisations, all energies and omega."""
    return {
        "cases": tuple(c for c in ref.CASES if c != "smooth"),
        "energies": ref.ENERGIES,
        "omegas": ref.OMEGAS,
        "n_lines": 16,
        "ly_nm": 2048.0,
        "n_real": 8,
        "shape": ref.SHAPE,
        "geometry": ref.GEOMETRY,
        "dx_nm": 1.0,
        "dy_nm": 4.0,
        "dz_nm": 1.0,
        "isolation": {"n_real": 32, "k_seeds": 4},
    }


def quick_config() -> dict:
    """Smoke-test size: the ``small()`` detector and ensemble, one case."""
    s = ref.small()
    return {
        "cases": ("both",),
        "energies": (285.0,),
        "omegas": np.array([-10.0, 0.0, 10.0]),
        "n_lines": s["n_lines"],
        "ly_nm": s["ly_nm"],
        "n_real": s["ensemble_n_real"],
        "shape": s["shape"],
        "geometry": s["geometry"],
        "dx_nm": s["dx_nm"],
        "dy_nm": s["dy_nm"],
        "dz_nm": s["dz_nm"],
        "isolation": {"n_real": 2, "k_seeds": 2},
    }


def run_ensemble(cfg: dict, case: str, energy: float, engine: str, seed: int = 0):
    """Ensemble result and the realised edge fields ``[(P, W), ...]``."""
    c, xs = ref.CASES[case], ref.CASES[case]["xs"](energy)
    edges = []

    def make(rng):
        st = build_rough_grating_3d(
            xs, n_lines=cfg["n_lines"], ly_nm=cfg["ly_nm"], dx_nm=cfg["dx_nm"],
            dy_nm=cfg["dy_nm"], dz_nm=cfg["dz_nm"], psd_lpr=c["psd_lpr"],
            psd_lwr=c["psd_lwr"], rng=rng,
        )  # fmt: skip
        edges.append((st.metadata["P_nm"], st.metadata["W_nm"]))
        return st

    # The engine's omega sense is opposite to the detector's (frame_omega_deg).
    res = simulate_rough_ensemble(
        make, -np.asarray(cfg["omegas"], dtype=float), engine=engine,
        n_realisations=cfg["n_real"], seed=seed,
        born_reference="epsilon" if engine == "multislice" else None,
    )  # fmt: skip
    return res, edges


def realised_truth(edges, ly_nm: float, f_edges) -> tuple[dict, float]:
    """Bin-averaged realised two-sided spectra of P, W (and LER), and sigma_e."""
    ny = edges[0][0].shape[-1]
    dy = ly_nm / ny
    f = np.fft.fftfreq(ny, d=dy)

    def pgram(fields):
        x = np.concatenate([np.atleast_2d(a) for a in fields], axis=0)
        return np.mean(np.abs(np.fft.fft(x, axis=-1)) ** 2, axis=0) * dy**2 / ly_nm

    pos = f > 0
    o = np.argsort(f[pos])
    fp = f[pos][o]
    sp = pgram([p for p, _ in edges])[pos][o]
    sw = pgram([w for _, w in edges])[pos][o]
    out = {"lpr": [], "lwr": []}
    for lo, hi in pairwise(f_edges):
        g = np.linspace(lo, hi, 65)
        out["lpr"].append(np.mean(np.interp(g, fp, sp)))
        out["lwr"].append(np.mean(np.interp(g, fp, sw)))
    truth = {k: np.asarray(v) for k, v in out.items()}
    truth["ler"] = truth["lpr"] + truth["lwr"] / 4
    p2 = np.mean([np.mean(p**2) for p, _ in edges])
    w2 = np.mean([np.mean(w**2) for _, w in edges])
    return truth, float(np.sqrt(p2 + w2 / 4))


def reduce_ensemble(cfg: dict, res, case: str, energy: float) -> ReducedPixels:
    xs = ref.CASES[case]["xs"](energy)
    items = []
    for i in range(res.omega_deg.size):
        fr = render_ensemble_frame(
            res, i, xs, cfg["geometry"], scale=ref.SCALE, shape=cfg["shape"],
            adu_per_photon=ref.ADU_PER_PHOTON, rng=np.random.default_rng(3000 + i),
        )  # fmt: skip
        items.append(
            reduce_frame(
                fr, cfg["geometry"], xs, scale=ref.SCALE,
                adu_per_photon=ref.ADU_PER_PHOTON, frame_id=i,
            )
        )  # fmt: skip
    return ReducedPixels.concat(items)


def analyse(cfg, case, energy, engine, res, edges) -> tuple[list[dict], dict]:
    """Global fit inside the first-order cut and sigma_win against realised truth."""
    xs = ref.CASES[case]["xs"](energy)
    pix = reduce_ensemble(cfg, res, case, energy)
    _, sigma_e = realised_truth(edges, cfg["ly_nm"], np.array([1e-3, 1e-2]))
    qx_max = 0.3 / sigma_e
    tab = fit_psds(pix, w=xs.w, qx_max=qx_max)
    truth, _ = realised_truth(edges, cfg["ly_nm"], tab.f_edges)
    rows = []
    c = ref.CASES[case]
    present = {"lpr": c["psd_lpr"] is not None, "lwr": c["psd_lwr"] is not None}
    for mode in MODES:
        if mode != "ler" and not present[mode]:
            continue
        S, var = getattr(tab, f"S_{mode}"), getattr(tab, f"var_{mode}")
        fin = np.isfinite(S)
        if fin.sum() < 2:
            continue
        sig, err = sigma_window(tab.f_center[fin], S[fin], var[fin])
        exact, _ = sigma_window(
            tab.f_center[fin], truth[mode][fin], np.zeros(fin.sum())
        )
        med = float(np.median(S[fin] / truth[mode][fin]))
        rows.append(
            {
                "case": case, "energy_ev": energy, "engine": engine, "mode": mode,
                "sigma_e_nm": round(sigma_e, 4), "qx_max": round(qx_max, 4),
                "sigma_win_nm": round(sig, 4), "sigma_win_err_nm": round(err, 4),
                "sigma_truth_nm": round(exact, 4), "sigma_ratio": round(sig / exact, 4),
                "median_S_ratio": round(med, 4),
            }
        )  # fmt: skip
    return rows, {"pix": pix, "tab": tab, "truth": truth, "qx_max": qx_max, "xs": xs}


def plot_psds(out: dict, case: str, energy: float, engine: str) -> None:
    tab, truth = out["tab"], out["truth"]
    c = ref.CASES[case]
    inputs = {"lpr": c["psd_lpr"], "lwr": c["psd_lwr"]}
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(MODES):
        S = getattr(tab, f"S_{mode}")
        fin = np.isfinite(S)
        if fin.sum() < 2 or (mode != "ler" and inputs[mode] is None):
            continue
        err = np.sqrt(getattr(tab, f"var_{mode}")[fin])
        fc = tab.f_center[fin]
        ax.errorbar(fc, S[fin], yerr=err, fmt="o", ms=3, color=f"C{i}", label=mode)
        ax.plot(fc, truth[mode][fin], "-", color=f"C{i}", lw=1)
        if mode != "ler":
            ax.plot(fc, inputs[mode](fc), ":", color=f"C{i}", lw=1)
    ax.set(xscale="log", yscale="log", xlabel=r"$f$ (nm$^{-1}$)", ylabel="PSD (nm$^3$)")
    ax.set_title(f"{case}, {energy:g} eV, {engine}: fit, realised (-), input (:)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"psd_{case}_{energy:g}eV_{engine}.png", dpi=110)
    plt.close(fig)


def plot_m_vs_qx(out: dict, case: str, energy: float) -> None:
    pix, tab, xs = out["pix"], out["tab"], out["xs"]
    fin = np.flatnonzero(np.isfinite(tab.S_lpr) & np.isfinite(tab.S_lwr))
    if fin.size == 0:
        return
    j = int(fin[fin.size // 3])
    sel = (pix.f >= tab.f_edges[j]) & (pix.f < tab.f_edges[j + 1])
    q = np.linspace(0, np.abs(pix.Qx[sel]).max(), 400)
    sp, sw = np.nan_to_num(tab.S_lpr[j]), np.nan_to_num(tab.S_lwr[j])
    fit = 4 * np.sin(q * xs.w / 2) ** 2 * sp + np.cos(q * xs.w / 2) ** 2 * sw
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(np.abs(pix.Qx[sel]), pix.M[sel], ".", ms=1, alpha=0.3)
    ax.plot(q, fit, "k-")
    ax.axvline(out["qx_max"], color="0.5")
    ax.set(xlabel=r"$|Q_x|$ (nm$^{-1}$)", ylabel=r"$M$ (nm$^3$)", yscale="log")
    ax.set_title(f"{case}, {energy:g} eV, f = {tab.f_center[j]:.3g} nm$^{{-1}}$")
    fig.tight_layout()
    fig.savefig(RESULTS / f"M_vs_qx_{case}_{energy:g}eV.png", dpi=110)
    plt.close(fig)


def ms_over_born(cfg, case, energy, ms, born, qx_max) -> dict[str, float]:
    """Median MS / (Born x A) per detector omega; plots the ratio against Qx."""
    xs = ref.CASES[case]["xs"](energy)
    W = np.deg2rad(ms.omega_deg)[:, None, None]
    A = absorption_factor(xs.mu_h, W, np.cos(W) / ms.flux_factor)
    sel = (
        ms.valid
        & (np.abs(ms.qx)[None, :, None] <= qx_max)
        & (ms.qy != 0)[None, None, :]
    )
    sel &= np.isfinite(born.diffuse) & (born.diffuse > 0.01 * np.nanmax(born.diffuse))
    r = np.where(sel, ms.diffuse / (born.diffuse * A), np.nan)
    oms = frame_omega_deg(ms.omega_deg, cfg["geometry"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    out = {}
    for i, om in enumerate(oms):
        out[f"{float(om):g}"] = round(float(np.nanmedian(r[i])), 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmedian(r[i], axis=1)
        ax.plot(ms.qx, prof, "-", lw=0.8, label=f"{float(om):g}°")
    ax.set(
        xlabel=r"$Q_x$ (nm$^{-1}$)", ylabel=r"MS / (Born $\times$ A), median over $Q_y$"
    )
    ax.set_title(f"{case}, {energy:g} eV")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"ms_over_born_{case}_{energy:g}eV.png", dpi=110)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small smoke-test config")
    args = ap.parse_args(argv)
    cfg = quick_config() if args.quick else full_config()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    rows: list[dict] = []
    born_both: dict[float, tuple] = {}
    for case in cfg["cases"]:
        for energy in cfg["energies"]:
            res, edges = run_ensemble(cfg, case, energy, "born")
            r, out = analyse(cfg, case, energy, "born", res, edges)
            rows += r
            plot_psds(out, case, energy, "born")
            plot_m_vs_qx(out, case, energy)
            if case == "both":
                born_both[energy] = (res, out["qx_max"])
            print(
                f"[{time.perf_counter() - t0:7.0f} s] born {case} {energy:g} eV",
                flush=True,
            )
    ms_report = {}
    for energy, (born, qx_max) in born_both.items():
        ms, edges = run_ensemble(cfg, "both", energy, "multislice")
        r, out = analyse(cfg, "both", energy, "multislice", ms, edges)
        ms_med = ms_over_born(cfg, "both", energy, ms, born, qx_max)
        ms_report[f"{energy:g}"] = ms_med
        for row in r:
            row["ms_over_born_A_median"] = float(np.median(list(ms_med.values())))
        rows += r
        plot_psds(out, "both", energy, "multislice")
        print(
            f"[{time.perf_counter() - t0:7.0f} s] multislice both {energy:g} eV",
            flush=True,
        )
    (RESULTS / "ms_over_born_study.json").write_text(json.dumps(ms_report, indent=1))

    iso = isolation_systematic(**cfg["isolation"])
    (RESULTS / "isolation_systematic.json").write_text(json.dumps(iso, indent=1))
    print(
        f"[{time.perf_counter() - t0:7.0f} s] isolation systematic: {iso}", flush=True
    )

    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 99)  # fmt: skip
    with (RESULTS / "summary.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"done in {time.perf_counter() - t0:.0f} s; results in {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
