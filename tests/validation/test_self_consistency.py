"""Self-consistency: analytic HOWTO frames in, known PSDs and scale out."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cd_reduce.reduction.roughness import (
    fit_psds,
    order_scale,
    sigma_from_psd,
    sigma_window,
)

from . import reference as ref

pytestmark = pytest.mark.validation


def test_self_consistency_recovers_psds(small_reduced):
    cfg = ref.small()
    tab = fit_psds(small_reduced, w=ref.latent(285.0).w)
    fin = np.isfinite(tab.S_lpr) & np.isfinite(tab.S_lwr)
    assert fin.sum() >= 5
    fc = tab.f_center[fin]
    qx, w = small_reduced.Qx, ref.latent(285.0).w
    m_true = 4 * np.sin(qx * w / 2) ** 2 * ref.PSD_LPR(small_reduced.f) + np.cos(
        qx * w / 2
    ) ** 2 * ref.PSD_LWR(small_reduced.f)
    clean = fit_psds(
        replace(
            small_reduced,
            M=m_true,
            var_M=small_reduced.m_per_photon * m_true,
        ),
        w=w,
        f_edges=tab.f_edges,
    )
    for S, var, psd, name in (
        (tab.S_lpr, tab.var_lpr, ref.PSD_LPR, "lpr"),
        (tab.S_lwr, tab.var_lwr, ref.PSD_LWR, "lwr"),
    ):
        ratio = np.median(S[fin] / psd(fc))
        print(name, "median ratio", ratio)
        assert 0.97 <= ratio <= 1.03, name
        # Bin truth: the same estimator on noise-free M from the PSDs at the
        # actual pixel (Qx, f). Removes in-bin PSD curvature and weighting, which
        # shift S by 0.1-0.5%, comparable to the error bars here.
        truth = getattr(clean, "S_lpr" if name == "lpr" else "S_lwr")[fin]
        dev = np.abs(S[fin] - truth) / np.sqrt(var[fin])
        print(name, "worst deviation (sigma)", dev.max())
        assert np.all(dev < 3), name

    def ler(f):
        return ref.PSD_LPR(f) + ref.PSD_LWR(f) / 4.0

    lo, hi = fc[0], fc[-1]
    for S, var, psd, name in (
        (tab.S_lpr, tab.var_lpr, ref.PSD_LPR, "lpr"),
        (tab.S_lwr, tab.var_lwr, ref.PSD_LWR, "lwr"),
        (tab.S_ler, tab.var_ler, ler, "ler"),
    ):
        sig, _ = sigma_window(tab.f_center[fin], S[fin], var[fin])
        exact = sigma_from_psd(psd, lo, hi)
        print(name, "sigma", sig, "truth", exact)
        assert abs(sig / exact - 1) < 0.03, name
    assert cfg["n_real"] == 2


def test_scale_recovery(small_order_table):
    xs = ref.latent(285.0)
    fit = order_scale(
        small_order_table,
        {285.0: xs},
        adu_per_photon=ref.ADU_PER_PHOTON,
        orders=(-3, -2, -1, 1, 2, 3),
    )
    got = fit.scale_by_energy[285.0]
    print("scale", got, "sigma_ler", fit.sigma_ler)
    assert abs(got / ref.SCALE - 1) < 0.02
