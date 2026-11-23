import numpy as np
import pandas as pd

from cd_reduce.reduction.roughness.cross_section import DevelopedLine
from cd_reduce.reduction.roughness.scale import (
    OrderScaleFit,
    debye_waller_sigma,
    order_efficiency,
    order_scale,
)

HC = 1239.84198
KW = {"L": 80, "w": 45, "s": 2.5, "dn_A": -1e-3 + 2e-4j}
C_TRUE, SIG_TRUE, ADU = 3e11, 1.25, 2.0
XS = {E: DevelopedLine(h_d=25, energy_ev=E, **KW) for E in (270.0, 285.0)}


def _table(seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    rows = []
    for E, xs in XS.items():
        k = 2 * np.pi * E / HC
        for om in range(-20, 21, 5):
            for m in (-4, -3, -2, -1, 1, 2, 3, 4):
                W = np.deg2rad(om)
                s_out = np.sin(W) + m * 2 * np.pi / xs.L / k
                if abs(s_out) >= 1:
                    continue
                Qz = k * (np.cos(np.arcsin(s_out)) - np.cos(W))
                eta = order_efficiency(xs, m, om, E, Qz, sigma_ler=SIG_TRUE)
                i_ph = C_TRUE * float(np.squeeze(eta))
                i_cps = ADU * i_ph * (1 + noise * rng.standard_normal())
                rows.append(
                    {"order": m, "energy_ev": E, "Qz": Qz, "I_cps": i_cps,
                     "sigma_cps": noise * i_cps, "omega_deg": om}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_order_scale_recovers_known_scale():
    fit = order_scale(_table(), XS, adu_per_photon=ADU)
    assert isinstance(fit, OrderScaleFit)
    for E in XS:
        assert abs(fit.scale_by_energy[E] / C_TRUE - 1) < 0.02
        assert fit.scale_err[E] > 0
    assert abs(fit.sigma_ler / SIG_TRUE - 1) < 0.05


def test_debye_waller_sigma_matches():
    s, err = debye_waller_sigma(_table(), XS, adu_per_photon=ADU)
    assert abs(s / SIG_TRUE - 1) < 0.05
    assert err >= 0


def test_order_efficiency_hand_computed():
    xs = XS[285.0]
    m, om, om0, T, qz, sig, E = 2, 7.0, 1.5, 0.5, 0.12, 1.1, 285.0
    k = 2 * np.pi * E / HC
    W = np.deg2rad(om + om0)
    cos_out = np.cos(W) + qz / k
    G = 2 * np.pi / xs.L
    A = np.exp(-(xs.mu_h / 2) * (1 / np.cos(W) + 1 / cos_out))
    want = (
        k**2 * abs(xs.dn_A) ** 2 * abs(xs.f_m(m, qz)) ** 2 / (np.cos(W) * cos_out)
        * np.exp(-(m**2) * G**2 * sig**2) * A * T ** (1 / np.cos(W))
    )  # fmt: skip
    got = order_efficiency(xs, m, om, E, qz, sigma_ler=sig, omega0_deg=om0, window_T=T)
    assert np.isclose(got, want, rtol=1e-12, atol=0)


def test_order_scale_uses_omega_true_column():
    t = _table()
    t["omega_true_deg"] = t["omega_deg"]
    t["omega_deg"] = t["omega_deg"] + 7.0  # wrong values; must be ignored
    fit = order_scale(t, XS, adu_per_photon=ADU)
    assert abs(fit.sigma_ler / SIG_TRUE - 1) < 0.05
    assert abs(fit.scale_by_energy[285.0] / C_TRUE - 1) < 0.02


def test_order_scale_energy_without_usable_rows_raises():
    import pytest

    t = _table()
    # cos(theta_out) < 0 for every 285 eV row: rows present, none usable
    t.loc[t["energy_ev"] == 285.0, "Qz"] = -5.0
    with pytest.raises(ValueError, match="285"):
        order_scale(t, XS, adu_per_photon=ADU)
