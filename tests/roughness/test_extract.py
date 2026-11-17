import numpy as np
import pytest

from cd_reduce.reduction.roughness.extract import (
    default_f_edges,
    fit_psds,
    isolation_row_psds,
    purity_bands,
    quick_ler,
    sigma_window,
)
from cd_reduce.reduction.roughness.psd_models import psd_palasantzas, sigma_from_psd
from cd_reduce.reduction.roughness.reduce import ReducedPixels

W = 45.0
F_LO, F_HI = 0.0013, 0.03


def S_P(f):
    return psd_palasantzas(f, 1.0, 25.0, 0.7)


def S_W(f):
    return psd_palasantzas(f, 1.5, 12.0, 0.5)


def make(qx, f, M, var, w=W):
    z = np.zeros_like(qx)
    return ReducedPixels(
        Qx=qx,
        Qy=2 * np.pi * f,
        Qz=z,
        f=f,
        D=z,
        M=M,
        var_M=var,
        energy_ev=z,
        omega_deg=z,
        frame_id=np.zeros(len(qx), dtype=np.int64),
        meta={"R_B": 2 * np.pi * F_LO, "w": w, "scale": 1.0, "scale_source": "test"},
    )


def synth(n=60000, edges=None, snap=True, rel=0.05, sp=S_P, sw=S_W, qx=None, seed=0):
    rng = np.random.default_rng(seed)
    if qx is None:
        qx = rng.uniform(0.02, 0.37, n)
    n = len(qx)
    f = np.exp(rng.uniform(np.log(F_LO), np.log(F_HI), n))
    fe = f
    if snap and edges is not None:
        i = np.clip(np.searchsorted(edges, f) - 1, 0, len(edges) - 2)
        fe = np.sqrt(edges[i] * edges[i + 1])
    M = 4 * np.sin(qx * W / 2) ** 2 * sp(fe) + np.cos(qx * W / 2) ** 2 * sw(fe)
    return make(
        qx * rng.choice([-1, 1], n),
        f * rng.choice([-1, 1], n),
        M,
        (rel * M) ** 2 + 1e-300,
    )


def test_fit_psds_noiseless_exact():
    edges = np.geomspace(F_LO, F_HI, 101)
    px = synth(
        edges=edges, rel=1e-6
    )  # M evaluated at bin-centre f so truth is the centre value
    t = fit_psds(px, W, edges)
    fc = t.f_center
    assert np.all(np.isfinite(t.S_lpr))
    assert np.allclose(t.S_lpr / S_P(fc), 1, atol=1e-3)
    assert np.allclose(t.S_lwr / S_W(fc), 1, atol=1e-3)


def test_fit_psds_noise_inflates_errors():
    edges = default_f_edges(2 * np.pi * F_LO, F_HI)
    clean = synth(edges=edges, snap=False, seed=1)
    rng = np.random.default_rng(2)
    noisy = make(
        clean.Qx,
        clean.f,
        clean.M + 5 * np.sqrt(clean.var_M) * rng.standard_normal(len(clean.M)),
        clean.var_M,
    )
    t0, t1 = fit_psds(clean, W, edges), fit_psds(noisy, W, edges)
    assert np.nanmedian(t1.chi2_red) > 10
    assert np.nanmedian(np.sqrt(t1.var_lpr) / np.sqrt(t0.var_lpr)) > 3


def test_fit_psds_negative_not_clipped():
    edges = default_f_edges(2 * np.pi * F_LO, F_HI)
    px = synth(edges=edges, sp=lambda f: 0 * f, seed=3)
    sd = (
        0.2 * np.sqrt(np.mean(px.M**2)) * np.ones_like(px.M)
    )  # constant noise, S_P truth is 0
    rng = np.random.default_rng(4)
    px = make(px.Qx, px.f, px.M + sd * rng.standard_normal(len(px.M)), sd**2)
    t = fit_psds(px, W, edges)
    assert np.all(np.isfinite(t.S_lpr))
    assert np.any(t.S_lpr < 0)


def test_fit_psds_order_rows_only_is_flagged():
    L = 2 * W
    G = 2 * np.pi / L
    rng = np.random.default_rng(5)
    qx = np.concatenate([G * m + rng.uniform(-1e-3, 1e-3, 3000) for m in (1, 2)])
    # one frequency window per order so some bins hold a single order
    base = synth(qx=qx, seed=6)
    px = base
    only1 = px.select(np.arange(len(qx)) < 3000)  # bins without the other row
    t = fit_psds(only1, W, np.geomspace(F_LO, F_HI, 6))
    assert np.all(np.isinf(t.cond) | np.isnan(t.S_lpr))
    empty = fit_psds(px.select(np.zeros(len(qx), bool)), W, np.geomspace(F_LO, F_HI, 6))
    assert np.all(np.isnan(empty.S_lpr))
    fit_psds(px, W, np.geomspace(F_LO, F_HI, 6))  # must not raise


def test_ler_combination():
    edges = default_f_edges(2 * np.pi * F_LO, F_HI)
    t = fit_psds(synth(edges=edges, seed=7), W, edges)
    assert np.allclose(t.S_ler, t.S_lpr + t.S_lwr / 4, equal_nan=True)


def test_cross_term_and_default_edges():
    px = synth(seed=8)
    t = fit_psds(px, W, None, cross_term=True)
    assert t.S_cross is not None and t.var_cross is not None
    bad = make(px.Qx, px.f, px.M, px.var_M)
    bad.meta.pop("R_B")
    with pytest.raises(ValueError):
        fit_psds(bad, W)


def test_isolation_rows_match_truth():
    bands = purity_bands(45, S_P, S_W, (F_LO, F_HI), 0.9, 0.37)
    assert {k for k, *_ in bands["lpr"]} and all(k % 2 == 1 for k, *_ in bands["lpr"])
    assert all(k % 2 == 0 for k, *_ in bands["lwr"])
    for items in bands.values():
        for k, lo, hi in items:
            assert abs(0.5 * (lo + hi) - k * np.pi / W) < 1e-3
    edges = np.geomspace(F_LO, F_HI, 11)
    rng = np.random.default_rng(9)
    centres = [k * np.pi / W for k in range(1, 6)]
    qx = np.concatenate([c + rng.uniform(-2e-4, 2e-4, 4000) for c in centres])
    px = synth(qx=qx, edges=edges, rel=1e-6, seed=10)
    # pixels sit within 2e-4 of k*pi/w, where leakage from the other mode is <1e-4 of M
    res = isolation_row_psds(px, W, bands, edges)
    fc = np.sqrt(edges[:-1] * edges[1:])
    assert res["lpr"] and res["lwr"]
    for d in res["lpr"].values():
        ok = np.isfinite(d["S"])
        assert ok.any()
        assert np.allclose(d["S"][ok] / S_P(fc)[ok], 1, atol=1e-3)
    for d in res["lwr"].values():
        ok = np.isfinite(d["S"])
        assert np.allclose(d["S"][ok] / S_W(fc)[ok], 1, atol=1e-3)


def test_quick_ler():
    edges = np.geomspace(F_LO, F_HI, 11)
    qx = np.random.default_rng(12).uniform(
        0.0, 0.37, 400000
    )  # whole periods need coverage from Qx=0
    px = synth(qx=qx, edges=edges, rel=0.01, seed=11)
    S, V = quick_ler(px, W, edges)
    fc = np.sqrt(edges[:-1] * edges[1:])
    assert np.all(np.isfinite(V))
    assert np.allclose(S, S_P(fc) + S_W(fc) / 4, rtol=0.02)


def test_sigma_window_matches_integral():
    fe = np.geomspace(0.008, 0.2, 27) / (2 * np.pi)
    fc = np.sqrt(fe[:-1] * fe[1:])
    sig, _ = sigma_window(fc, S_P(fc), np.zeros_like(fc))
    ref = sigma_from_psd(S_P, fc[0], fc[-1])
    assert sig == pytest.approx(ref, rel=0.02)
    s2, e2 = sigma_window(fc, S_P(fc), (0.1 * S_P(fc)) ** 2)
    assert e2 > 0 and s2 == sig
    S = S_P(fc).copy()
    S[3] = np.nan
    keep = np.arange(len(fc)) != 3
    got = sigma_window(fc, S, np.zeros_like(fc))
    assert got == sigma_window(fc[keep], S[keep], np.zeros(keep.sum()))


def _bin_design(px, w, i, edges):
    f = np.abs(px.f)
    s = (f >= edges[i]) & (f < edges[i + 1])
    x = np.abs(px.Qx[s])
    cols = [4 * np.sin(x * w / 2) ** 2, np.cos(x * w / 2) ** 2]
    return np.stack(cols, 1), px.M[s], px.var_M[s]


def test_fit_psds_covariance_matches_bruteforce():
    edges = np.geomspace(F_LO, F_HI, 6)
    rng = np.random.default_rng(20)
    px = synth(n=5000, edges=edges, snap=False, seed=21)
    px = make(px.Qx, px.f, px.M * (1 + 0.15 * rng.standard_normal(len(px.M))), px.var_M)
    t = fit_psds(px, W, edges)
    i = 2
    A, m, v = _bin_design(px, W, i, edges)
    N = A.T @ (A / v[:, None])
    inv = np.linalg.inv(N)
    beta = inv @ (A.T @ (m / v))
    chi2 = np.sum((m - A @ beta) ** 2 / v) / (len(m) - 2)
    assert chi2 > 1
    cov = inv * max(1.0, chi2)
    assert t.chi2_red[i] == pytest.approx(chi2)
    assert t.var_lpr[i] == pytest.approx(cov[0, 0])
    assert t.var_lwr[i] == pytest.approx(cov[1, 1])
    assert t.cov_pw[i] == pytest.approx(cov[0, 1])
    assert t.var_ler[i] == pytest.approx(cov[0, 0] + cov[1, 1] / 16 + cov[0, 1] / 2)


def test_fit_psds_monte_carlo_std_matches_reported():
    edges = np.geomspace(F_LO, F_HI, 4)
    base = synth(n=3000, edges=edges, snap=True, rel=0.05, seed=22)
    rng = np.random.default_rng(23)
    est, rep = [], []
    for _ in range(200):
        M = base.M + np.sqrt(base.var_M) * rng.standard_normal(len(base.M))
        t = fit_psds(make(base.Qx, base.f, M, base.var_M), W, edges)
        est.append(t.S_lpr[1])
        rep.append(np.sqrt(t.var_lpr[1]))
    assert np.std(est) == pytest.approx(np.mean(rep), rel=0.15)


def test_fit_psds_max_cond_flags_ill_conditioned_bin():
    edges = np.geomspace(F_LO, F_HI, 4)
    px = synth(n=5000, edges=edges, seed=24)
    ok = fit_psds(px, W, edges)
    assert np.all(np.isfinite(ok.S_lpr)) and np.all(np.isfinite(ok.cond))
    cmin = float(np.min(ok.cond))
    bad = fit_psds(px, W, edges, max_cond=cmin / 2)
    assert np.all(np.isnan(bad.S_lpr)) and np.all(np.isnan(bad.var_lpr))
    assert np.all(np.isinf(bad.cond))


def test_cross_term_recovered():
    edges = np.geomspace(F_LO, F_HI, 6)
    rng = np.random.default_rng(25)
    n = 20000
    qx = rng.uniform(0.02, 0.37, n)
    f = np.exp(rng.uniform(np.log(F_LO), np.log(F_HI), n))
    i = np.clip(np.searchsorted(edges, f) - 1, 0, 4)
    fc = np.sqrt(edges[i] * edges[i + 1])

    def S_X(g):
        return 0.1 * S_P(g)

    def build(sx):
        M = (
            4 * np.sin(qx * W / 2) ** 2 * S_P(fc)
            + np.cos(qx * W / 2) ** 2 * S_W(fc)
            + 2 * np.sin(qx * W) * sx(fc)
        )
        return make(qx, f, M, (1e-6 * np.abs(M) + 1e-9) ** 2)

    t = fit_psds(build(S_X), W, edges, cross_term=True)
    c = np.sqrt(edges[:-1] * edges[1:])
    assert np.allclose(t.S_lpr / S_P(c), 1, atol=1e-3)
    assert np.allclose(t.S_lwr / S_W(c), 1, atol=1e-3)
    assert np.allclose(t.S_cross / S_X(c), 1, atol=1e-3)
    z = build(lambda g: 0 * g)
    t3 = fit_psds(z, W, edges, cross_term=True)
    t2 = fit_psds(z, W, edges)
    assert np.allclose(t3.S_lpr, t2.S_lpr, rtol=1e-3)
    assert np.allclose(t3.S_lwr, t2.S_lwr, rtol=1e-3)
    assert np.allclose(t3.S_cross, 0, atol=1e-3 * S_P(c).min())


def test_purity_bands_drops_zero_width_band():
    # S_LPR prior vanishes, so odd-k (LPR) bands have zero purity everywhere
    bands = purity_bands(W, lambda f: 0 * f, S_W, (F_LO, F_HI), 0.9, 0.37)
    assert bands["lpr"] == []
    assert [k for k, *_ in bands["lwr"]] == [2, 4]


def test_fit_psds_low_count_poisson_unbiased():
    """Model-based IRLS variance removes the low-photon observed-variance bias."""
    rng = np.random.default_rng(7)
    edges = default_f_edges(2 * np.pi * F_LO, F_HI)
    n = 400000
    qx = rng.uniform(0.02, 0.37, n)
    f = np.exp(rng.uniform(np.log(F_LO), np.log(F_HI), n))
    i = np.clip(np.searchsorted(edges, f) - 1, 0, len(edges) - 2)
    fe = np.sqrt(edges[i] * edges[i + 1])
    mt = 4 * np.sin(qx * W / 2) ** 2 * S_P(fe) + np.cos(qx * W / 2) ** 2 * S_W(fe)
    lam = rng.uniform(3.0, 7.0, n)
    m = mt / lam
    ph = rng.poisson(lam).astype(float)
    px = make(qx, f, m * ph, m * m * ph)
    px = ReducedPixels(
        **{k: getattr(px, k) for k in ("Qx", "Qy", "Qz", "f", "D", "M", "var_M")},
        energy_ev=px.energy_ev,
        omega_deg=px.omega_deg,
        frame_id=px.frame_id,
        m_per_photon=m,
        meta={**px.meta, "read_noise_photons": 0.0},
    )
    t = fit_psds(px, W, edges)
    ok = np.isfinite(t.S_lpr) & np.isfinite(t.S_lwr)
    assert ok.sum() >= 15
    assert 0.99 <= np.median(t.S_lpr[ok] / S_P(t.f_center[ok])) <= 1.01
    assert 0.99 <= np.median(t.S_lwr[ok] / S_W(t.f_center[ok])) <= 1.01
