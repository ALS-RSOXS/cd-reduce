"""Model error: Born-ensemble frames of explicit rough edges, HOWTO-reduced.

Frames are rendered from `simulate_rough_ensemble` (no HOWTO model inside) and
reduced with the package; the recovered PSDs, their omega invariance, the
isolation rows and the Debye-Waller sigma are checked against the known inputs.

Rulings applied: criteria use pixels with ``|Qx| sigma_e <= 0.3`` (R19, the
HOWTO's first-order limit; the deficit beyond is reported, not tested); PSD
truths are the realised spectra of the generated edge fields (R20a); chi^2
criteria take their errors from the scatter over K independent ensembles (R20b).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import compute_pixel_q, frame_pixel_grid
from cd_reduce.reduction.roughness import (
    debye_waller_sigma,
    fit_psds,
    isolation_row_psds,
    purity_bands,
    sigma_window,
)

from . import reference as ref
from .forward.ensemble_frames import ensemble_vs_model_ratio
from .forward.matched_model import matched_model_frames, mode_grid_M
from .forward.model_frames import render_model_frame
from .forward.orders import order_table

pytestmark = pytest.mark.validation

ROUGH = ("lpr", "lwr", "both", "independent")
# R24: single-mode separable-model tests run on the developed line (assumption 4).
MODEL_CASES = ("lpr_developed", "lwr_developed", "both", "independent")
# Sum both frames over ~3 x 6 ensemble modes before taking ratios: the per-mode
# diffuse of a 2-realisation ensemble is exponentially distributed (R20c).
BLOCK_PX = (40, 25)
# R20b: the ``ensembles`` fixture builds K = 8 seed ensembles (with K = 4 the
# per-bin seed variance of the M-invariance chi^2 has only (G - 1)(K - 1) = 12
# dof, so E[1/v] inflates chi^2/dof by ~1.2). R22: the ratio and sigma_win tests
# pool all K seeds; the chi^2 tests use their scatter.
NULL_LEVEL = 0.10
RESULTS = Path(__file__).parent / "results"


def _modes(case: str) -> tuple[str, ...]:
    """Non-zero modes of a case (the zero mode of single-mode cases is skipped)."""
    c = ref.CASES[case]
    return tuple(
        n for n, p in (("lpr", c["psd_lpr"]), ("lwr", c["psd_lwr"])) if p is not None
    )


def _realised_psds(ensembles, case: str, seed: int = 0):
    """PSD callables (two-sided, of f) interpolating the realised spectra (R20a)."""
    c = ref.CASES[case]
    f, sp, sw = ensembles.realised_spectra(case, seed)
    pos = f > 0
    f, sp, sw = f[pos], sp[pos], sw[pos]

    def interp(s):
        return lambda q: np.interp(np.abs(q), f, s)

    return (
        interp(sp) if c["psd_lpr"] is not None else None,
        interp(sw) if c["psd_lwr"] is not None else None,
    )


def _model_frames(ensembles, case: str, seed: int = 0):
    """Noiseless HOWTO frames for the edge spectra realised in ensemble ``seed``."""
    cfg = ensembles.cfg
    xs = ref.CASES[case]["xs"](285.0)
    psd_lpr, psd_lwr = _realised_psds(ensembles, case, seed)
    return [
        render_model_frame(
            xs, cfg["geometry"], fe.omega_deg,
            psd_lpr=psd_lpr, psd_lwr=psd_lwr, scale=ref.SCALE,
            shape=cfg["shape"], adu_per_photon=ref.ADU_PER_PHOTON,
            sigma_ler_full=ensembles.sigma_e(case, seed),
            rng=np.random.default_rng(0), poisson=False,
        )
        for fe in ensembles.frames(case, poisson=False, seed=seed)
    ]  # fmt: skip


def _matched_frames(ensembles, case: str, seed: int = 0):
    """Noiseless matched-realisation first-order frames of ensemble ``seed`` (R25)."""
    cfg = ensembles.cfg
    return matched_model_frames(
        ensembles.result(case, seed), ensembles.edges(case, seed),
        ref.CASES[case]["xs"](285.0), cfg["geometry"], scale=ref.SCALE,
        shape=cfg["shape"], adu_per_photon=ref.ADU_PER_PHOTON,
        sigma_ler_full=ensembles.sigma_e(case, seed), ly_nm=cfg["ly_nm"],
    )  # fmt: skip


def _pooled_frames(ensembles, case: str, matched: bool = True, engine: str = "born"):
    """Noiseless ensemble and model frames per omega, summed over the K seeds."""

    def total(frames):
        f0 = frames[0]
        img = np.sum([np.asarray(f.image, dtype=float) for f in frames], axis=0)
        return Frame.from_array(
            img, energy_ev=f0.energy_ev, omega_deg=f0.omega_deg,
            exposure_s=f0.exposure_s,
        )  # fmt: skip

    ens = [
        ensembles.frames(case, poisson=False, seed=s, engine=engine)
        for s in ensembles.seeds(True)
    ]
    model = _matched_frames if matched else _model_frames
    mod = [model(ensembles, case, s) for s in ensembles.seeds(True)]
    return (
        [total(fr) for fr in zip(*ens, strict=True)],
        [total(fr) for fr in zip(*mod, strict=True)],
    )


def ratio_medians(
    ensembles,
    case: str,
    matched: bool = True,
    min_blocks: int = 10,
    engine: str = "born",
) -> list[tuple[float, float]]:
    """``(omega, median ensemble/model block ratio)`` per omega, inside the cut.

    The model is the matched-realisation first-order prediction (R25), or with
    ``matched=False`` the HOWTO model on the realised spectra. Frames are pooled
    over the K seed ensembles (R22); blocks near the nulls of
    the first-order edge factor are excluded (``NULL_LEVEL``).
    """
    cfg = ensembles.cfg
    xs = ref.CASES[case]["xs"](285.0)
    meds = []
    pooled = _pooled_frames(ensembles, case, matched, engine)
    for fe, fm in zip(*pooled, strict=True):
        r = ensemble_vs_model_ratio(
            fe, fm, cfg["geometry"], pitch_nm=xs.L, block_px=BLOCK_PX,
            qx_max=ensembles.qx_max(case, pooled=True), null_level=NULL_LEVEL,
        )  # fmt: skip
        assert r.size >= min_blocks, (
            f"{case} omega {fe.omega_deg}: only {r.size} blocks"
        )
        meds.append((fe.omega_deg, round(float(np.median(r)), 4)))
    return meds


def _check_ratio(ensembles, case: str, engine: str = "born"):
    meds = ratio_medians(ensembles, case, engine=engine)
    bad = [(om, m) for om, m in meds if not 0.95 <= m <= 1.05]
    assert not bad, f"{case} {engine} ensemble/model median ratio per omega: {meds}"


def _global_fit(
    ensembles, case: str, seed: int | None = 0, f_edges=None, engine: str = "born"
):
    """Global fit of one seed ensemble, or of all seeds pooled (``seed=None``)."""
    pix = (
        ensembles.reduced_pooled(case, engine)
        if seed is None
        else ensembles.reduced(case, seed, engine)
    )
    w = ref.CASES[case]["xs"](285.0).w
    qx_max = ensembles.qx_max(case, pooled=seed is None)
    tab = fit_psds(pix, w=w, f_edges=f_edges, qx_max=qx_max)
    return pix, w, tab


def _realised_binned(ensembles, case: str, f_edges, seed: int | None = 0):
    """Realised P, W, LER spectra averaged over each |f| bin (R20a).

    The periodogram lives on the discrete field modes; the ensemble frames
    interpolate linearly between them, so each bin takes the mean of the linear
    interpolant over the bin.
    """
    if seed is None:
        f, sp, sw = ensembles.realised_spectra(case, pooled=True)
    else:
        f, sp, sw = ensembles.realised_spectra(case, seed)
    pos = f > 0
    f, sp, sw = f[pos], sp[pos], sw[pos]
    out = {"lpr": [], "lwr": []}
    for lo, hi in pairwise(f_edges):
        g = np.linspace(lo, hi, 65)
        out["lpr"].append(np.mean(np.interp(g, f, sp)))
        out["lwr"].append(np.mean(np.interp(g, f, sw)))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["ler"] = res["lpr"] + res["lwr"] / 4
    return res


def sigma_window_ratios(ensembles, case: str, engine: str = "born") -> dict[str, float]:
    """sigma_win / realised truth per mode (and ler), inside the cut, pooled (R22)."""
    _, _, tab = _global_fit(ensembles, case, seed=None, engine=engine)
    truth = _realised_binned(ensembles, case, tab.f_edges, seed=None)
    names = (*_modes(case), "ler")
    res = {}
    for name in names:
        S, var = getattr(tab, f"S_{name}"), getattr(tab, f"var_{name}")
        fin = np.isfinite(S)
        assert fin.sum() >= 5, f"{case} {name}: {fin.sum()} finite bins"
        fc = tab.f_center[fin]
        sig, _ = sigma_window(fc, S[fin], var[fin])
        exact, _ = sigma_window(fc, truth[name][fin], np.zeros(fin.sum()))
        res[name] = round(sig / exact, 4)
    return res


def _check_sigma_window(ensembles, case: str, engine: str = "born"):
    res = sigma_window_ratios(ensembles, case, engine)
    bad = {k: v for k, v in res.items() if not abs(v - 1) < 0.05}
    assert not bad, f"{case} {engine} sigma_win / realised truth: {res}"


@pytest.mark.parametrize("case", MODEL_CASES)
def test_ensemble_matches_model(ensembles, case):
    _check_ratio(ensembles, case)


@pytest.mark.parametrize("case", MODEL_CASES)
def test_ensemble_recovers_sigma_window(request, case):
    # Single-mode developed cases pool 32 realisations (n = 4 x K = 8): at 16
    # they sat at 0.952 / 0.953 (noise); at 32 they are ~0.975. The two-mode
    # cases are within 1-3% at 16 and keep the shared n = 2 ensembles (runtime).
    fixture = "ensembles_pooled32" if case.endswith("_developed") else "ensembles"
    _check_sigma_window(request.getfixturevalue(fixture), case)


def test_latent_nonseparable_report(ensembles):
    """Report only (R24): single-mode latent film violates HOWTO assumption 4.

    The latent film's shrinkage sits inside the depth integral, so the edge
    spectrum is not separable across the blurred edge and the first-order
    model is biased 4-7% for LPR-only / LWR-only (the bias cancels when both
    modes are present). See task-8-investigation.md. Asserts finiteness only.
    """
    out = {}
    for case in ("lpr", "lwr"):
        meds = ratio_medians(ensembles, case, min_blocks=1)
        howto = ratio_medians(ensembles, case, matched=False, min_blocks=1)
        sw = sigma_window_ratios(ensembles, case)
        assert all(np.isfinite(m) for _, m in meds + howto)
        assert all(np.isfinite(v) for v in sw.values())
        out[case] = {
            "matched_ratio_median_by_omega": {str(om): m for om, m in meds},
            "realised_spectra_ratio_median_by_omega": {str(om): m for om, m in howto},
            "sigma_win_ratio": sw,
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latent_nonseparable.json").write_text(json.dumps(out, indent=1))


def test_single_mode_sigma_window_report(ensembles, ensembles_pooled32):
    """Report only: sigma_win ratios of the developed single-mode cases at 16 and 32."""
    out = {}
    for case in ("lpr_developed", "lwr_developed"):
        r16 = sigma_window_ratios(ensembles, case)
        r32 = sigma_window_ratios(ensembles_pooled32, case)
        assert all(np.isfinite(v) for v in (*r16.values(), *r32.values()))
        out[case] = {"pooled_16": r16, "pooled_32": r32}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sigma_win_single_mode.json").write_text(json.dumps(out, indent=1))


def test_matched_model_normalisation_guard():
    """White-noise edges with known S_P, S_W: matched M averages to the HOWTO M."""
    xs = ref.developed(285.0)
    nl, ny, ly, n_real = 4, 256, 1024.0, 400
    dy = ly / ny
    sp_w, sw_w = 0.8, 1.7  # white PSDs, nm^3: variance = S / dy per sample
    rng = np.random.default_rng(7)
    edges = [
        (
            rng.normal(0, np.sqrt(sp_w / dy), (nl, ny)),
            rng.normal(0, np.sqrt(sw_w / dy), (nl, ny)),
        )
        for _ in range(n_real)
    ]
    qx = np.linspace(0.02, 0.3, 8)
    iy = np.arange(1, 40)
    res = SimpleNamespace(qx=qx, metadata={"iy_full": iy})
    Mg = mode_grid_M(res, edges, xs.L, xs.w, ly)
    expect = (
        4 * np.sin(qx * xs.w / 2) ** 2 * sp_w + np.cos(qx * xs.w / 2) ** 2 * sw_w
    )[:, None]
    ratio = Mg.mean(axis=1) / expect[:, 0]
    # sin^2/cos^2 weights vanish at nulls; compare away from them
    ok = expect[:, 0] > 0.2 * expect.max()
    assert np.all(np.abs(ratio[ok] - 1) < 0.02), ratio


def m_invariance_chi2(ensembles, case: str) -> tuple[float, int]:
    """Per-omega-group S around the across-group mean, errors from seed scatter."""
    _, _, tab0 = _global_fit(ensembles, case)
    edges = tab0.f_edges
    groups = np.asarray(ensembles.cfg["ensemble_omegas"], dtype=float)
    assert groups.size == 5
    qx_max = ensembles.qx_max(case)
    chi2, dof = 0.0, 0
    for name in _modes(case):
        S = np.full((ensembles.k_seeds, groups.size, len(edges) - 1), np.nan)
        for s in range(ensembles.k_seeds):
            pix = ensembles.reduced(case, s)
            for j, g in enumerate(groups):
                t = fit_psds(
                    pix.select(np.isclose(pix.omega_deg, g)),
                    w=ref.CASES[case]["xs"](285.0).w, f_edges=edges, qx_max=qx_max,
                )  # fmt: skip
                S[s, j] = getattr(t, f"S_{name}")
        d = S - S.mean(axis=1, keepdims=True)  # (K, G, nb)
        ok = np.isfinite(d).all(axis=(0, 1))
        d = d[:, :, ok]
        K, G = d.shape[:2]
        mean = d.mean(axis=0)
        v = np.sum((d - mean) ** 2, axis=(0, 1)) / ((G - 1) * (K - 1))
        chi2 += float(np.sum(K * mean**2 / v))
        dof += (G - 1) * int(ok.sum())
    return chi2 / dof, dof


@pytest.mark.parametrize("case", ROUGH)
def test_M_invariant_across_omega(ensembles, case):
    red, dof = m_invariance_chi2(ensembles, case)
    assert dof >= 10
    assert red < 2, f"{case} M-invariance chi2/dof {red:.3g} ({dof})"


def isolation_chi2(ensembles, case: str) -> tuple[float, int]:
    """Rows minus global per bin, errors from seed scatter (pooled per row)."""
    _, w, tab0 = _global_fit(ensembles, case)
    fin = np.isfinite(tab0.S_lpr) & np.isfinite(tab0.S_lwr)
    window = (float(tab0.f_center[fin][0]), float(tab0.f_center[fin][-1]))
    sig_e = ensembles.sigma_e(case)
    qx_max = ensembles.qx_max(case)
    bands = purity_bands(45, ref.PSD_LPR, ref.PSD_LWR, window, 0.9, 0.37)
    bands = {
        m: [b for b in items if b[0] <= 4 and b[0] * np.pi / w * sig_e <= 0.3]
        for m, items in bands.items()
        if m in _modes(case)
    }
    n_rows = sum(len(v) for v in bands.values())
    assert n_rows >= 1, f"{case}: no isolation rows inside the first-order cut"
    rel = {}
    for s in range(ensembles.k_seeds):
        pix, _, tab = _global_fit(ensembles, case, s, f_edges=tab0.f_edges)
        rows = isolation_row_psds(pix, w, bands, tab.f_edges, qx_max=qx_max)
        for mode, per_k in rows.items():
            glob = getattr(tab, f"S_{mode}")
            for k, r in per_k.items():
                rel.setdefault((mode, k), []).append((r["S"] - glob) / glob)
    chi2, dof = 0.0, 0
    for r in rel.values():
        r = np.asarray(r)  # (K, nb)
        ok = np.isfinite(r).all(axis=0)
        r = r[:, ok]
        K, nb = r.shape
        mean = r.mean(axis=0)
        v = np.sum((r - mean) ** 2) / (nb * (K - 1))
        chi2 += float(np.sum(K * mean**2 / v))
        dof += nb
    return chi2 / dof, dof


@pytest.mark.parametrize("case", ROUGH)
def test_isolation_rows_agree_with_global(ensembles, case):
    # Passes at the prescribed K = 8 x n = 2 (R23). With larger ensembles a
    # systematic row-vs-global difference appears (chi^2/dof ~ 3.4 for both and
    # independent at n = 32); see `isolation_systematic` (report only).
    red, dof = isolation_chi2(ensembles, case)
    assert dof >= 10
    assert red < 2, f"{case} isolation chi2/dof {red:.3g} ({dof})"


def dw_sigma(ensembles, case: str) -> tuple[float, float, float]:
    """``(sigma_DW, err, realised truth)`` from the integrated orders."""
    cfg = ensembles.cfg
    xs = ref.CASES[case]["xs"](285.0)
    tab = order_table(
        ensembles.frames(case), cfg["geometry"], adu_per_photon=ref.ADU_PER_PHOTON
    )
    sig, err = debye_waller_sigma(
        tab, {285.0: xs}, adu_per_photon=ref.ADU_PER_PHOTON,
        orders=(-3, -2, -1, 1, 2, 3),
    )  # fmt: skip
    return float(sig), float(err), ensembles.sigma_e(case)  # truth: R4


@pytest.mark.parametrize("case", ROUGH)
def test_debye_waller_sigma_from_ensemble_frames(ensembles, case):
    sig, err, truth = dw_sigma(ensembles, case)
    assert abs(sig / truth - 1) < 0.10, f"{case} DW sigma {sig} +- {err}, truth {truth}"


def test_smooth_case_has_no_diffuse(ensembles):
    res = ensembles.result("smooth")
    eta1 = float(np.nanmax(res.order_flux_efficiency(1)))
    assert eta1 > 0
    assert np.nanmax(res.diffuse_flux) < 1e-12 * eta1


def test_developed_line_case(ensembles):
    _check_ratio(ensembles, "both_developed")
    _check_sigma_window(ensembles, "both_developed")


def test_first_order_deficit_report(ensembles):
    """Report (no criterion): pooled ensemble/matched-model ratio vs |Qx| (R19)."""
    cfg = ensembles.cfg
    geo = cfg["geometry"]
    uu, vv = frame_pixel_grid(cfg["shape"])
    edges = np.arange(0.0, 0.375 + 1e-9, 0.025)
    report = {"qx_edges": edges.tolist(), "cases": {}}
    for case in (*ROUGH, "both_developed"):
        G = 2 * np.pi / ref.CASES[case]["xs"](285.0).L
        per_om = {}
        for fe, fm in zip(*_pooled_frames(ensembles, case), strict=True):
            qm = compute_pixel_q(
                geo, uu, vv, fe.energy_ev, fe.omega_deg, approximation="correct"
            )
            qx, qy = np.asarray(qm.Qx, float), np.asarray(qm.Qy, float)
            a, b = np.asarray(fe.image, float), np.asarray(fm.image, float)
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(qy) > 0.02)
            for m in range(-5, 6):
                if m:
                    ok &= np.hypot(qx - m * G, qy) > 0.012
            ratios = []
            for lo, hi in pairwise(edges):
                s = ok & (np.abs(qx) >= lo) & (np.abs(qx) < hi)
                ratios.append(float(a[s].sum() / b[s].sum()) if s.any() else None)
            per_om[str(fe.omega_deg)] = ratios
        report["cases"][case] = {
            "sigma_e_nm": ensembles.sigma_e(case),
            "qx_cut": ensembles.qx_max(case),
            "ratio_by_omega": per_om,
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "first_order_deficit.json").write_text(json.dumps(report, indent=1))


def test_band_limited_truth_matches_psd(ensembles):
    """The realised DW truth is close to the PSD band integral (sanity, R4)."""
    cfg = ensembles.cfg
    ny = round(cfg["ly_nm"] / cfg["dy_nm"])
    f = np.fft.fftfreq(ny, d=cfg["ly_nm"] / ny)[1:]
    band = np.sqrt(
        (np.sum(ref.PSD_LPR(np.abs(f))) + np.sum(ref.PSD_LWR(np.abs(f))) / 4)
        / cfg["ly_nm"]
    )
    assert band < ref.sigma_ler_full(ref.PSD_LPR, ref.PSD_LWR)
    assert abs(ensembles.sigma_e("both") / band - 1) < 0.15


def isolation_systematic(n_real: int = 32, k_seeds: int = 4) -> dict:
    """Report only (R23): isolation chi^2/dof with a large ensemble.

    Not collected by pytest (no ``test_`` prefix); called by the study script.
    Returns ``{case: {"chi2_dof": x, "dof": n}}``; asserts finiteness only.
    """
    from .conftest import _EnsembleCache

    ens = _EnsembleCache(n_real=n_real, k_seeds=k_seeds)
    out = {}
    for case in ROUGH:
        red, dof = isolation_chi2(ens, case)
        assert np.isfinite(red)
        out[case] = {"chi2_dof": float(red), "dof": int(dof)}
    return out
