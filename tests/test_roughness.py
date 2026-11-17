"""Validation of the LER / LWR / LPR Born kernel and multi-order decomposition."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from cd_reduce.reduction.roughness import (
    DiffuseRatio,
    decompose_edge_psd,
    edge_psd_kernel,
    estimate_line_width,
    fit_psd_model,
    psd_mack,
    sigma_from_psd,
)

PITCH = 100.0
WIDTH = 32.0
G = 2.0 * np.pi / PITCH


def _edges(n_lines, ny, ly, sigma_p, sigma_w, xi, rng):
    """Placement and width fluctuations with independent Mack PSDs."""
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    shape = psd_mack(qy, 1.0, xi, 0.5)
    shape[0] = 0.0
    norm = np.sum(shape) * (2.0 * np.pi / ly) / (2.0 * np.pi)

    def draw(sig):
        S = shape * sig**2 / norm
        z = rng.normal(size=(n_lines, ny)) + 1j * rng.normal(size=(n_lines, ny))
        amp = np.sqrt(S * ly / 2.0) * z
        # Real field with two-sided PSD S: <|FT|^2> = Ly S
        return np.fft.ifft(amp, axis=1).real * (ny / ly) * np.sqrt(2.0), S

    P, S_P = draw(sigma_p)
    W, S_W = draw(sigma_w)
    return qy, P, W, S_P, S_W


def _simulated_ratio(orders, sigma_p, sigma_w, *, n_real=40, seed=3):
    """Exact per-line Fourier transform of rough lines -> R_m(qy) table."""
    rng = np.random.default_rng(seed)
    n_lines, ny, ly = 16, 512, 4096.0
    dy = ly / ny
    acc = {m: np.zeros(ny) for m in orders}
    bragg = dict.fromkeys(orders, 0.0)
    S_P = S_W = qy = None
    for _ in range(n_real):
        qy, P, W, S_P, S_W = _edges(n_lines, ny, ly, sigma_p, sigma_w, 30.0, rng)
        x_c = np.arange(n_lines)[:, None] * PITCH + P
        xl = x_c - 0.5 * (WIDTH + W)
        xr = x_c + 0.5 * (WIDTH + W)
        for m in orders:
            q = m * G
            f = (np.exp(-1j * q * xl) - np.exp(-1j * q * xr)) / (1j * q)
            a = np.fft.fft(f.sum(axis=0)) * dy
            acc[m] += np.abs(a) ** 2
            # Bragg amplitude from the mean (rough-free) line
            f0 = (np.exp(-1j * q * (-WIDTH / 2)) - np.exp(-1j * q * (WIDTH / 2))) / (
                1j * q
            )
            bragg[m] += abs(f0 * n_lines * ly) ** 2
    dqx = 2.0 * np.pi / (n_lines * PITCH)
    dqy = 2.0 * np.pi / ly
    qe = np.geomspace(0.01, 0.3, 9)
    rows = []
    for m in orders:
        R = acc[m] / (bragg[m] * dqx * dqy)
        for lo, hi in itertools.pairwise(qe):
            sel = (np.abs(qy) >= lo) & (np.abs(qy) < hi)
            rows.append(
                {
                    "order": m,
                    "qy_lo": lo,
                    "qy_hi": hi,
                    "qy_center": np.sqrt(lo * hi),
                    "ratio": R[sel].mean(),
                    "ratio_err": R[sel].mean() * 0.03,
                    "wing_photons": 1e4,
                    "wing_area": 1.0,
                    "flank_ratio": 0.0,
                    "order_photons": 1e6,
                    "qz_mean": 0.0,
                    "n_frames": n_real,
                }
            )
    ratio = DiffuseRatio(
        table=pd.DataFrame(rows),
        qy_edges=qe,
        mode="line_cut",
        normalize="order",
        pitch_nm=PITCH,
    )
    # Expected PSDs bin-averaged on the same grid
    exp_P = [
        S_P[(np.abs(qy) >= lo) & (np.abs(qy) < hi)].mean()
        for lo, hi in itertools.pairwise(qe)
    ]
    exp_W = [
        S_W[(np.abs(qy) >= lo) & (np.abs(qy) < hi)].mean()
        for lo, hi in itertools.pairwise(qe)
    ]
    return ratio, np.array(exp_P), np.array(exp_W)


def test_kernel_reduces_to_single_edge_ler():
    # Uncorrelated edges: S_P = S/2, S_W = 2S -> R = 2 p S / ((2pi)^2 F_m^2)
    kP, kW = edge_psd_kernel(np.array([1, 2, 3]), PITCH, WIDTH)
    S = 1.0
    R = kP * 0.5 * S + kW * 2.0 * S
    F2 = (
        2.0 * np.sin(np.array([1, 2, 3]) * G * WIDTH / 2) / (np.array([1, 2, 3]) * G)
    ) ** 2
    np.testing.assert_allclose(R, 2.0 * PITCH * S / ((2 * np.pi) ** 2 * F2))


def test_decomposition_recovers_lpr_and_lwr():
    orders = (-3, -2, -1, 1, 2, 3)
    ratio, exp_P, exp_W = _simulated_ratio(orders, sigma_p=0.6, sigma_w=0.8)
    res = decompose_edge_psd(ratio, WIDTH)
    t = res.table
    assert t["decomposed"].all()
    np.testing.assert_allclose(t["S_LPR"], exp_P, rtol=0.25)
    np.testing.assert_allclose(t["S_LWR"], exp_W, rtol=0.25)
    np.testing.assert_allclose(t["S_LER"], exp_P + exp_W / 4, rtol=0.2)


def test_pure_placement_gives_rho_one():
    ratio, _, _ = _simulated_ratio(
        (-2, -1, 1, 2, 3), sigma_p=0.8, sigma_w=0.0, n_real=10
    )
    res = decompose_edge_psd(ratio, WIDTH)
    assert np.nanmedian(res.table["rho_LR"]) > 0.9


def test_mack_fit_and_sigma():
    q = np.geomspace(0.01, 0.5, 30)
    S = psd_mack(q, 40.0, 25.0, 0.6)
    fit = fit_psd_model(q, S, 0.05 * S, model="mack")
    assert fit.params["xi_nm"] == pytest.approx(25.0, rel=0.05)
    assert fit.params["hurst"] == pytest.approx(0.6, abs=0.05)
    edges = np.geomspace(1e-4, 50.0, 4000)
    centers = np.sqrt(edges[1:] * edges[:-1])
    s_band, _ = sigma_from_psd(edges, psd_mack(centers, 40.0, 25.0, 0.6))
    assert s_band == pytest.approx(fit.sigma_nm(), rel=0.05)


def test_estimate_line_width():
    m = np.array([1, 2, 3, 4])
    intens = {int(k): float((np.sin(k * G * WIDTH / 2) / k) ** 2) for k in m}
    assert estimate_line_width(intens, PITCH) == pytest.approx(WIDTH, abs=0.1)


# ---------------------------------------------------------------------------
# End to end on synthetic detector frames
# ---------------------------------------------------------------------------


def _series(flux, omegas, rng=0):
    from cd_reduce.reduction.geometry import Geometry
    from cd_reduce.simulation.diffuse import synthetic_rough_grating_frame

    geo = Geometry(u0_px=192, v0_px=192, Lb_px=900, Lambda_nm=PITCH)
    frames = [
        synthetic_rough_grating_frame(
            geo,
            om,
            line_width_nm=WIDTH,
            psd_lpr=_SP,
            psd_lwr=_SW,
            shape=(384, 384),
            flux=flux,
            max_order=4,
            rng=rng + k,
        )
        for k, om in enumerate(omegas)
    ]
    return geo, frames


def _SP(q):
    return psd_mack(q, 20.0, 25.0, 0.5)


def _SW(q):
    return psd_mack(q, 60.0, 15.0, 0.5)


@pytest.mark.parametrize(
    ("mode", "kw"),
    [("line_cut", {"omega": 0.0}), ("qz_trace", {"qz_range": (0.02, 0.05)})],
)
def test_end_to_end_frames_recover_psds(mode, kw):
    from cd_reduce.reduction.roughness import extract_diffuse_ratio

    geo, frames = _series(3e5, np.arange(-20.0, 21.0, 4.0))
    qe = np.geomspace(0.015, 0.15, 7)
    qc = np.sqrt(qe[1:] * qe[:-1])
    ratio = extract_diffuse_ratio(
        frames,
        geo,
        orders=(-4, -3, -2, -1, 1, 2, 3, 4),
        qy_edges=qe,
        mode=mode,
        bg_mode="none",
        **kw,
    )
    t = decompose_edge_psd(ratio, WIDTH).table
    np.testing.assert_allclose(t["S_LPR"], _SP(qc), rtol=0.12)
    np.testing.assert_allclose(t["S_LWR"], _SW(qc), rtol=0.12)


def test_single_frame_line_cut_runs_without_series():
    from cd_reduce.reduction.roughness import extract_diffuse_ratio

    geo, frames = _series(3e5, [0.0])
    ratio = extract_diffuse_ratio(frames, geo, orders=(1, 2, 3), bg_mode="none")
    assert ratio.metadata["ewald_correction"] is False
    assert np.isfinite(ratio.table["ratio"]).any()
