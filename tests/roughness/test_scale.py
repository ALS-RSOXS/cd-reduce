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
