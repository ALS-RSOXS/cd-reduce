import numpy as np
import pytest

from cd_reduce.reduction.roughness.psd_models import (
    fit_psd_model,
    psd_mack,
    psd_palasantzas,
    sigma_from_psd,
)


def test_palasantzas_integrates_to_sigma2():
    def S(f):
        return psd_palasantzas(f, sigma_nm=1.2, xi_nm=20.0, hurst=0.6)

    assert sigma_from_psd(S) == pytest.approx(1.2, rel=1e-3)


def test_sigma_from_psd_window_is_smaller():
    def S(f):
        return psd_palasantzas(f, sigma_nm=1.0, xi_nm=25.0, hurst=0.7)

    assert sigma_from_psd(S, 0.008 / (2 * np.pi), 0.2 / (2 * np.pi)) < 1.0


def test_fit_recovers_mack_in_f():
    f = np.geomspace(0.002, 0.08, 30)
    S = psd_mack(f, 40.0, 25.0, 0.6)
    fit = fit_psd_model(f, S, 0.05 * S, model="mack")
    assert fit.params["xi_nm"] == pytest.approx(25.0, rel=0.05)
    assert fit.params["hurst"] == pytest.approx(0.6, abs=0.05)
