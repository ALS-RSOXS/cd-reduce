import dataclasses
from dataclasses import replace

import numpy as np

from cd_reduce.reduction.roughness.cross_section import DevelopedLine
from cd_reduce.reduction.roughness.extract import PsdTable
from cd_reduce.reduction.roughness.record import PsdRecord

KW = {"L": 80, "w": 45, "s": 2.5, "dn_A": -1e-3 + 2e-4j}
XS = DevelopedLine(h_d=25, energy_ev=285.0, **KW)


def _rec(xs=XS):
    n = 5
    e = np.geomspace(0.01, 0.5, n + 1)
    a = np.linspace(1.0, 2.0, n)
    nan = a.copy()
    nan[2] = np.nan
    t = PsdTable(
        f_edges=e, f_center=np.sqrt(e[:-1] * e[1:]), S_lpr=a, S_lwr=nan,
        S_ler=a * 2, var_lpr=a * 0.1, var_lwr=nan * 0.1, var_ler=a * 0.2,
        cov_pw=a * 0.01, cond=a * 3, chi2_red=nan, n=np.arange(n),
        S_cross=a * 0.5, var_cross=None,
    )  # fmt: skip
    return PsdRecord(
        table=t,
        cross_section={"class": "DevelopedLine", "params": dataclasses.asdict(xs)},
        scale=3e11, scale_err=1e9, scale_source="bragg_orders",
        source={"energies": [285.0], "qz_mean": 0.5, "rows": 12},
        window=(0.01, 0.5),
        sigma_win={"lpr": (1.0, 0.1), "lwr": (float("nan"),) * 2, "ler": (2.0, 0.2)},
        model_fit=None,
    )  # fmt: skip


def _eq(a, b):
    if a is None or b is None:
        return a is b
    return np.array_equal(a, b, equal_nan=True)


def test_record_roundtrip(tmp_path):
    r = _rec()
    p = tmp_path / "psd.csv"
    r.save(p)
    assert p.read_text().splitlines()[0] == "# two-sided PSD, nm^3, f = Qy/2pi in nm^-1"
    q = PsdRecord.load(p)
    for name in ("f_center", "S_lpr", "S_lwr", "S_ler", "var_lpr", "var_lwr",
                 "var_ler", "cov_pw", "cond", "chi2_red", "n", "S_cross",
                 "var_cross", "f_edges"):  # fmt: skip
        assert _eq(getattr(r.table, name), getattr(q.table, name)), name
    assert isinstance(q.cross_section["params"]["dn_A"], complex)
    assert q.cross_section == r.cross_section
    assert (q.scale, q.scale_err, q.scale_source) == (
        r.scale,
        r.scale_err,
        r.scale_source,
    )
    assert q.source == r.source and q.window == r.window and q.model_fit is None
    assert np.isnan(q.sigma_win["lwr"][0]) and q.sigma_win["ler"] == (2.0, 0.2)


def test_record_rescale_by_gamma_ratio():
    new = replace(XS, dn_A=2 * XS.dn_A)
    r = _rec()
    q = r.rescale(new)
    assert np.allclose(q.table.S_lpr, r.table.S_lpr / 4, equal_nan=True)
    assert np.allclose(q.table.var_lpr, r.table.var_lpr / 16, equal_nan=True)
    assert np.allclose(q.table.var_ler, r.table.var_ler / 16)
    assert np.allclose(q.table.cov_pw, r.table.cov_pw / 16)
    assert np.allclose(q.sigma_win["ler"], np.array(r.sigma_win["ler"]) / 2)
