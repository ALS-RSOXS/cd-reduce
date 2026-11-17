"""Case study: resist line gratings through the full cd-reduce pipeline.

Runs measured CD-RSoXS tilt series of resist gratings (latent image, post-exposure
bake, developed) through background subtraction, Q mapping, specular reduction and
the LER / LWR / LPR diffuse-scattering analysis, and compares the roughness with
AFM where available.

The data are anonymised: the script only knows samples by alias ("A", "B", ...).
The mapping from alias to data lives in a private JSON config that is *not* part
of the repository::

    {
      "geometry_json": ".../geometry.json",       # per-mount calibrated geometry
      "afm_npz": ".../afm_psd.npz",               # optional AFM PSDs
      "afm_metrics_csv": ".../afm_metrics.csv",   # optional AFM 3 sigma table
      "afm_stage_names": {"latent": "...", ...},
      "samples": {
        "A": {"id": "...", "geometry_prefix": "...", "root": ".../frames",
              "dir_prefix": "...", "mounts": ["", "_L", "_R"]},
        ...
      }
    }

Usage::

    uv run python examples/case_study_resist_gratings.py CONFIG.json OUT_DIR [--cache DIR]

Nothing identifying (ids, paths, file names, dates) is written to OUT_DIR.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cd_reduce as cdr

STAGES = ("latent", "PEB", "dvlp")
STAGE_LABEL = {"latent": "latent", "PEB": "PEB", "dvlp": "developed"}
STAGE_COLOR = {"latent": "#7c3aed", "PEB": "#0891b2", "dvlp": "#ea580c"}
SAMPLE_COLOR = {"A": "#1f6fb4", "B": "#119c84", "C": "#e8772e", "D": "#8a5cb5"}
E_MAP = 285.2
ENERGIES = (280.0, 285.2, 288.6)
ORDERS = (-4, -3, -2, -1, 1, 2, 3, 4)
QY_EDGES = np.geomspace(0.014, 0.2, 10)
BG = "strip_fit"


# ---------------------------------------------------------------------------
# Data access (all identifying information stays inside this section)
# ---------------------------------------------------------------------------


class Study:
    def __init__(self, cfg_path: Path, cache: Path) -> None:
        self.cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self.geo_all = json.loads(Path(self.cfg["geometry_json"]).read_text())
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.afm = np.load(self.cfg["afm_npz"]) if self.cfg.get("afm_npz") else None
        m = self.cfg.get("afm_metrics_csv")
        self.afm_metrics = pd.read_csv(m) if m else None
        self._frames: dict[tuple[str, str, str], list[cdr.Frame]] = {}

    def samples(self) -> list[str]:
        return list(self.cfg["samples"])

    def mounts(self, s: str) -> list[str]:
        return list(self.cfg["samples"][s]["mounts"])

    def _key(self, s: str, stage: str, mount: str) -> str:
        return f"{self.cfg['samples'][s]['geometry_prefix']}_{stage}{mount}"

    def has(self, s: str, stage: str, mount: str = "") -> bool:
        sc = self.cfg["samples"][s]
        d = Path(sc["root"]) / f"{sc['dir_prefix']}_{stage}{mount}"
        return d.is_dir() and self._key(s, stage, mount) in self.geo_all

    def frames(self, s: str, stage: str, mount: str = "") -> list[cdr.Frame]:
        k = (s, stage, mount)
        if k not in self._frames:
            sc = self.cfg["samples"][s]
            d = Path(sc["root"]) / f"{sc['dir_prefix']}_{stage}{mount}"
            frs = cdr.load_frames(d, deduplicate_omega=False)
            for f in frs:  # strip provenance so nothing identifying propagates
                f.path = None
            self._frames[k] = frs
        return self._frames[k]

    def frames_at(self, s: str, stage: str, energy: float, mount: str = "") -> list:
        return [
            f for f in self.frames(s, stage, mount) if abs(f.energy_ev - energy) < 0.5
        ]

    def geometry(self, s: str, stage: str, mount: str = "") -> tuple[cdr.Geometry, Any]:
        g = self.geo_all[self._key(s, stage, mount)]
        return cdr.Geometry.from_dict(g["base"]), g.get("omega_refine")

    def afm_psd(self, s: str, stage: str, kind: str):
        if self.afm is None:
            return None
        st = self.cfg.get("afm_stage_names", {}).get(stage, stage)
        k = f"{self.cfg['samples'][s]['id']}_{st}_{kind}"
        if k + "_f" not in self.afm.files:
            return None
        return 2 * np.pi * self.afm[k + "_f"], 0.5 * self.afm[k + "_P1"]

    def afm_cd(self, s: str, stage: str) -> float | None:
        if self.afm is None:
            return None
        st = self.cfg.get("afm_stage_names", {}).get(stage, stage)
        k = f"{self.cfg['samples'][s]['id']}_{st}_cd"
        return float(self.afm[k]) if k in self.afm.files else None

    def afm_three_sigma(self, s: str, stage: str) -> dict[str, float] | None:
        if self.afm_metrics is None:
            return None
        st = self.cfg.get("afm_stage_names", {}).get(stage, stage)
        t = self.afm_metrics
        sel = t[
            (t.film == self.cfg["samples"][s]["id"])
            & (t.stage.str.lower() == st.lower())
        ]
        if sel.empty:
            return None
        return {
            k: float(sel[f"three_sigma_{k}_nm"].mean()) for k in ("LER", "LWR", "LPR")
        }

    def cached(self, name: str, fn):
        f = self.cache / f"{name}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        val = fn()
        f.write_bytes(pickle.dumps(val))
        return val


def label(s: str, stage: str | None = None) -> str:
    return f"Sample {s}" + (f"  {STAGE_LABEL[stage]}" if stage else "")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_butterfly(st: Study, out: Path, samples=("A", "B")) -> None:
    rows, labels, geos = [], [], []
    for s in samples:
        geo, cal = st.geometry(s, "dvlp")
        frs = st.frames_at(s, "dvlp", E_MAP)
        G = geo.g_inv_nm

        def mk(a, frs=frs, geo=geo, cal=cal, G=G, s=s):
            return st.cached(
                f"map_{s}_dvlp_{a}",
                lambda: cdr.map_2d(
                    frs,
                    geo,
                    x_domain="Qx",
                    y_domain="Qz",
                    bins=(300, 260),
                    x_range=(-4.3 * G, 4.3 * G),
                    y_range=(-0.3, 0.3),
                    approximation=a,
                    bg_mode=BG,
                    omega_refinement=cal,
                    qy_cut_over_g=0.08,
                ),
            )

        rows.append([mk("correct"), mk("small_angle")])
        labels.append([f"Sample {s}  exact", f"Sample {s}  small-angle"])
        geos.append(geo)
    fig = cdr.plot_qmap_grid(
        rows,
        labels=labels,
        geometries=geos,
        ewald_omega_deg=0.0,
        energy_ev=E_MAP,
        detector_shape=(1024, 1024),
        ylim=(-0.3, 0.3),
    )
    fig.savefig(out / "01_butterfly_exact_vs_small_angle.png", dpi=170)
    plt.close(fig)


def fig_strips(st: Study, out: Path, samples=("A", "B")) -> None:
    for s in samples:
        geo, cal = st.geometry(s, "dvlp")
        strips = {}
        for f in st.frames_at(s, "dvlp", E_MAP):
            if f.omega_deg in (0.0, 10.0, 20.0) and f.omega_deg not in strips:
                c, qm, _, _ = cdr.prepare_frame(
                    f, geo, bg_mode=BG, omega_refinement=cal
                )
                strips[f.omega_deg] = cdr.extract_line_strip(
                    geo, np.nan_to_num(c), qm.Qx, qm.Qy, half_width_px=40
                )
        fig = cdr.plot_detector_strips(strips)
        fig.savefig(out / f"02_detector_strips_sample{s}.png", dpi=170)
        plt.close(fig)


def fig_planes(st: Study, out: Path, s: str = "A") -> None:
    geo, cal = st.geometry(s, "dvlp")
    frs = st.frames_at(s, "dvlp", E_MAP)
    G = geo.g_inv_nm
    kw = {"bg_mode": BG, "omega_refinement": cal}
    specs = {
        "xz": dict(
            x_domain="Qx",
            y_domain="Qz",
            bins=(260, 200),
            qy_cut_over_g=0.08,
            x_range=(-4.3 * G, 4.3 * G),
            y_range=(-0.3, 0.3),
        ),
        "xy": dict(
            x_domain="Qx",
            y_domain="Qy",
            bins=(260, 200),
            qz_cut=(-0.01, 0.01),
            x_range=(-4.3 * G, 4.3 * G),
            y_range=(-0.2, 0.2),
        ),
        "yz": dict(
            x_domain="Qy",
            y_domain="Qz",
            bins=(160, 160),
            qx_cut=(0.8 * G, 1.2 * G),
            x_range=(-0.2, 0.2),
            y_range=(-0.15, 0.15),
        ),
        "rz": dict(
            x_domain="signed_Qr",
            y_domain="Qz",
            bins=(260, 200),
            qy_cut_over_g=0.08,
            x_range=(-4.3 * G, 4.3 * G),
            y_range=(-0.3, 0.3),
        ),
    }
    m = {
        k: st.cached(f"plane_{s}_{k}", lambda v=v: cdr.map_2d(frs, geo, **v, **kw))
        for k, v in specs.items()
    }
    fig = cdr.plot_qmap_grid(
        [[m["xz"], m["xy"]], [m["yz"], m["rz"]]],
        labels=[
            [r"$Q_x$-$Q_z$", r"$Q_x$-$Q_y$, $|Q_z|<0.01$"],
            [r"$Q_y$-$Q_z$, $m=+1$", r"signed $Q_r$-$Q_z$"],
        ],
        share_axes=False,
    )
    fig.savefig(out / f"03_planes_sample{s}.png", dpi=170)
    plt.close(fig)


def order_table(st: Study, s: str, stage: str, energy: float = E_MAP) -> pd.DataFrame:
    def build():
        geo, cal = st.geometry(s, stage)
        L = cdr.CdRsoxsLoader(
            frames=st.frames_at(s, stage, energy),
            geometry=geo,
            omega_refinement=cal,
            pitch_nm=geo.Lambda_nm,
        )
        t = L.build_order_table()
        return t.drop(columns=[c for c in ("file", "sample") if c in t.columns])

    return st.cached(f"orders_{s}_{stage}_{energy:g}", build)


def fig_specular(st: Study, out: Path, samples=("A", "B")) -> None:
    for s in samples:
        ot = order_table(st, s, "dvlp")
        geo, _ = st.geometry(s, "dvlp")
        by_om = cdr.extract_qx_by_omega(ot, geo)
        by_om_r = cdr.extract_qx_by_omega(ot, geo, domain="signed_Qr")
        rods = cdr.extract_order_rods_qz(ot, geo, orders=ORDERS)
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
        cdr.plot_traces_by_omega(by_om, ax=axs[0], offset_decades=0.45)
        cdr.plot_traces_by_omega(by_om_r, ax=axs[1], offset_decades=0.45)
        cdr.plot_rods_by_order(rods, ax=axs[2], offset_decades=1.0)
        for ax, t in zip(
            axs,
            ("I vs Qx per ω", "I vs signed Qr per ω", "I_m vs Qz per order"),
            strict=True,
        ):
            ax.set_title(f"Sample {s} developed — {t}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"04_specular_1d_sample{s}.png", dpi=170)
        plt.close(fig)


def fig_stage_rods(st: Study, out: Path, samples=("A", "B")) -> None:
    """First-order normalised rods I_m / (I_+1 + I_-1) per stage."""
    fig, axs = plt.subplots(
        len(samples), 3, figsize=(11, 3.3 * len(samples)), sharex=True, squeeze=False
    )
    for i, s in enumerate(samples):
        for j, m in enumerate((2, 3, 4)):
            ax = axs[i, j]
            for stage in STAGES:
                if not st.has(s, stage):
                    continue
                ot = order_table(st, s, stage)
                geo, _ = st.geometry(s, stage)
                rods = cdr.extract_order_rods_qz(
                    ot, geo, orders=(m, -m), normalize_first_order=True
                )
                for mm, mk in ((m, "o-"), (-m, "s--")):
                    tr = rods.get(mm)
                    if tr is None:
                        continue
                    ok = np.isfinite(tr.intensity) & (tr.intensity > 0)
                    ax.plot(
                        tr.coords[ok],
                        tr.intensity[ok],
                        mk,
                        color=STAGE_COLOR[stage],
                        ms=3.5,
                        lw=0.9,
                        mfc=STAGE_COLOR[stage] if mm > 0 else "white",
                        label=STAGE_LABEL[stage] if mm > 0 else None,
                    )
            ax.set_yscale("log")
            ax.set_title(f"Sample {s},  m = ±{m}", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"$I_m / (I_{+1}+I_{-1})$")
            if i == len(samples) - 1:
                ax.set_xlabel(r"$Q_z$ (nm$^{-1}$)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "05_stage_rods_first_order_normalised.png", dpi=170)
    plt.close(fig)


def line_width(st: Study, s: str, stage: str) -> float:
    """Box line width from the omega = 0 order ratios (w <-> p - w symmetric)."""
    ot = order_table(st, s, stage)
    geo, _ = st.geometry(s, stage)
    z = ot[np.abs(ot["omega_deg"]) < 0.5].groupby("order")["I_cps"].mean()
    vals = {int(m): 0.5 * (z.get(m, np.nan) + z.get(-m, np.nan)) for m in (1, 2, 3, 4)}
    vals = {m: v for m, v in vals.items() if np.isfinite(v) and v > 0}
    return cdr.estimate_line_width(vals, geo.Lambda_nm)


def pooled_ratio(
    st: Study, s: str, stage: str, mode: str = "line_cut", **kw
) -> cdr.DiffuseRatio:
    """Pool all mounts and energies (ratio of sums, 5 sigma leave-one-out clip)."""
    tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))

    def build():
        parts = []
        for mount in st.mounts(s):
            if not st.has(s, stage, mount):
                continue
            geo, cal = st.geometry(s, stage, mount)
            mask = None
            for E in ENERGIES:
                frs = st.frames_at(s, stage, E, mount)
                if len(frs) < 2:
                    continue
                r = cdr.extract_diffuse_ratio(
                    frs,
                    geo,
                    orders=ORDERS,
                    qy_edges=QY_EDGES,
                    mode=mode,
                    omega=None if mode == "line_cut" else 0.0,
                    bg_mode=BG,
                    omega_refinement=cal,
                    beamstop_mask=mask,
                    **kw,
                )
                r.metadata.pop("omega", None)
                parts.append(r)
        return parts

    parts = st.cached(f"ratioparts_{s}_{stage}_{mode}_{tag}", build)
    return cdr.combine_diffuse_ratios(parts)


def analyse_roughness(st: Study, s: str, stage: str) -> dict[str, Any] | None:
    if not st.has(s, stage):
        return None
    ratio = pooled_ratio(st, s, stage)
    w_afm = st.afm_cd(s, stage) if stage == "dvlp" else None
    w = w_afm if w_afm is not None else line_width(st, s, stage)
    res = cdr.decompose_edge_psd(ratio, w)
    fits = {}
    for k in ("LER", "LWR", "LPR"):
        try:
            fits[k] = res.fit(k, model="mack")
        except (ValueError, RuntimeError):
            pass
    out: dict[str, Any] = {
        "ratio": ratio,
        "res": res,
        "fits": fits,
        "w": w,
        "w_source": "AFM CD" if w_afm is not None else "order ratios",
    }
    for k in ("LER", "LWR", "LPR"):
        sg, ds = res.sigma(k)  # type: ignore[arg-type]
        out[f"3s_{k}"] = 3 * sg
        out[f"3s_{k}_err"] = 3 * ds
    # AFM integrated over the same Qy band, for a like-for-like comparison
    for k in ("LER", "LWR"):
        a = st.afm_psd(s, stage, k)
        if a is not None:
            q, S = a
            centers = np.sqrt(QY_EDGES[1:] * QY_EDGES[:-1])
            Sb = np.interp(centers, q, S)
            out[f"afm_band_3s_{k}"] = 3 * cdr.sigma_from_psd(QY_EDGES, Sb)[0]
    out["afm_full"] = st.afm_three_sigma(s, stage)
    return out


def fig_diffuse(st: Study, out: Path, s: str = "A") -> None:
    r_cut = pooled_ratio(st, s, "dvlp")
    r_qz = pooled_ratio(st, s, "dvlp", mode="qz_trace", qz_range=(-0.02, 0.02))
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    cdr.plot_diffuse_ratio(r_cut, ax=axs[0])
    cdr.plot_diffuse_ratio(r_qz, ax=axs[1])
    axs[0].set_title(f"Sample {s} developed — line cuts, all ω, E, mounts", fontsize=9)
    axs[1].set_title(f"Sample {s} developed — Qz trace |Qz| < 0.02", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"06_diffuse_ratio_sample{s}.png", dpi=170)
    plt.close(fig)


def fig_stage_diffuse(st: Study, out: Path, samples=("A", "B")) -> None:
    """Model-free: pooled R_m(|Qy|) per processing stage for orders +-1 and +-2."""
    fig, axs = plt.subplots(
        1, len(samples), figsize=(5.0 * len(samples), 4.2), sharey=True, squeeze=False
    )
    for ax, s in zip(axs[0], samples, strict=True):
        for stage in STAGES:
            if not st.has(s, stage):
                continue
            t = pooled_ratio(st, s, stage).table
            for mabs, mk in ((1, "o-"), (2, "s--")):
                sub = t[t.order.abs() == mabs].groupby("qy_center")
                ph = sub.wing_photons.sum()
                den = sub.wing_den.sum()
                ok = (den > 0) & (ph > 10)
                r = (ph / den)[ok]
                e = (np.sqrt(ph) / den)[ok]
                ax.errorbar(
                    r.index,
                    r,
                    e,
                    fmt=mk,
                    ms=4,
                    lw=0.9,
                    color=STAGE_COLOR[stage],
                    mfc=STAGE_COLOR[stage] if mabs == 1 else "white",
                    label=f"{STAGE_LABEL[stage]}, |m| = {mabs}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|Q_y|$ (nm$^{-1}$)")
        ax.set_title(f"Sample {s}: diffuse / Bragg order, by stage", fontsize=9)
    axs[0, 0].set_ylabel(r"$(dI/d^2Q)\,/\,I_m$ (nm$^2$)")
    axs[0, 0].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out / "09_diffuse_by_stage.png", dpi=170)
    plt.close(fig)


def fig_psds(st: Study, out: Path, results: dict, samples=("A", "B")) -> None:
    fig, axs = plt.subplots(
        1, len(samples), figsize=(5.0 * len(samples), 4.6), sharey=True, squeeze=False
    )
    for ax, s in zip(axs[0], samples, strict=True):
        r = results.get((s, "dvlp"))
        if r is None:
            continue
        t = r["res"].table
        for k in ("LER", "LWR", "LPR"):
            col = cdr.reduction.plotting.PSD_COLORS[k]
            ok = np.isfinite(t[f"S_{k}"]) & (t[f"S_{k}"] > 0)
            ax.errorbar(
                t.qy_center[ok],
                t[f"S_{k}"][ok],
                t[f"S_{k}_err"][ok],
                fmt="o",
                ms=4.5,
                color=col,
                lw=0.8,
                label=f"X-ray {k}",
                mec="k",
                mew=0.3,
            )
            if k in r["fits"]:
                f = r["fits"][k]
                qq = np.geomspace(*f.qy_range, 100)
                ax.plot(qq, f(qq), "-", color=col, lw=1.0)
            a = st.afm_psd(s, "dvlp", k)
            if a is not None:
                ax.plot(
                    a[0], a[1], ".", color=col, ms=2.5, alpha=0.35, label=f"AFM {k}"
                )
        ax.axvspan(QY_EDGES[0], QY_EDGES[-1], color="0.94", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5e-3, 1.0)
        ax.set_xlabel(r"$|Q_y| = 2\pi f$ (nm$^{-1}$)")
        ax.set_title(
            f"Sample {s} developed  (w = {r['w']:.0f} nm from order ratios)", fontsize=9
        )
    axs[0, 0].set_ylabel(r"two-sided PSD (nm$^3$)")
    axs[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "07_edge_psds_xray_vs_afm.png", dpi=170)
    plt.close(fig)


def fig_summary(out: Path, results: dict) -> pd.DataFrame:
    rows = []
    for (s, stage), r in results.items():
        if r is None or stage != "dvlp":
            continue
        row = {"sample": s, "stage": STAGE_LABEL[stage], "w_source": r["w_source"]}
        for k in ("LER", "LWR", "LPR"):
            row[f"xray_3sigma_{k}_band"] = r[f"3s_{k}"]
            row[f"afm_3sigma_{k}_band"] = r.get(f"afm_band_3s_{k}", np.nan)
            row[f"afm_3sigma_{k}_full"] = (r["afm_full"] or {}).get(k, np.nan)
        row["rho_LR_median"] = float(np.nanmedian(r["res"].table["rho_LR"]))
        row["bins_decomposed"] = int(r["res"].table["decomposed"].sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(out / "summary_roughness.csv", index=False)

    # LER is the robust quantity; the LWR / LPR split depends on the assumed
    # line width and is reported in the CSV and the PSD figure only.
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(len(df))
    cols = [SAMPLE_COLOR.get(s, "0.4") for s in df["sample"]]
    ax.bar(
        x,
        df["xray_3sigma_LER_band"],
        color=cols,
        width=0.6,
        label="X-ray, measured band",
    )
    ax.plot(x, df["afm_3sigma_LER_band"], "k_", ms=22, mew=1.8, label="AFM, same band")
    ax.plot(x, df["afm_3sigma_LER_full"], "kx", ms=7, label="AFM, full")
    for xi, v in zip(x, df["xray_3sigma_LER_band"], strict=True):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=8, color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Sample {s}" for s in df["sample"]])
    ax.set_ylabel(r"3$\sigma_{\rm LER}$ (nm), developed")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "08_roughness_summary.png", dpi=170)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--cache", type=Path, default=Path("/tmp/cd_reduce_case_study_cache")
    )
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = Study(a.config, a.cache)
    todo = set(
        a.only
        or ["butterfly", "strips", "planes", "specular", "stages", "diffuse", "psd"]
    )
    if "butterfly" in todo:
        fig_butterfly(st, a.out)
    if "strips" in todo:
        fig_strips(st, a.out)
    if "planes" in todo:
        fig_planes(st, a.out)
    if "specular" in todo:
        fig_specular(st, a.out)
    if "stages" in todo:
        fig_stage_rods(st, a.out)
    if "diffuse" in todo:
        fig_diffuse(st, a.out)
    if "psd" in todo:
        results = {
            (s, stage): analyse_roughness(st, s, stage)
            for s in st.samples()
            for stage in STAGES
        }
        fig_psds(st, a.out, results)
        fig_stage_diffuse(st, a.out)
        df = fig_summary(a.out, results)
        with pd.option_context("display.width", 200):
            print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
