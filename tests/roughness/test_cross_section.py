from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from cd_reduce.reduction.roughness.cross_section import (
    CrossSection,
    DevelopedLine,
    LatentFilm,
    to_index_map_2d,
)

KW = {"L": 80, "w": 45, "s": 2.5, "dn_A": -1e-3 + 2e-4j, "energy_ev": 285.0}
lat = LatentFilm(h=40, dh=4, lam=25, r=0.6 + 0.05j, **KW)
dev = DevelopedLine(h_d=25, **KW)


def test_protocol():
    assert isinstance(lat, CrossSection) and isinstance(dev, CrossSection)


def test_latent_gamma_finite_at_zero():
    assert np.isfinite(lat.gamma(np.array([0.0]))).all()
    assert lat.gamma(0) == pytest.approx(lat.gamma(1e-7), rel=1e-5)


@pytest.mark.parametrize("q", [0.05, 0.3])
def test_latent_relief_only_when_r_is_one(q):
    f = replace(lat, r=1.0 + 0j)
    re = quad(lambda z: np.cos(q * z), 36, 40)[0]
    im = quad(lambda z: -np.sin(q * z), 36, 40)[0]
    assert f.gamma(q) == pytest.approx(re + 1j * im, rel=1e-6)


def test_developed_zeros():
    assert abs(dev.gamma(2 * np.pi / 25)) < 1e-9
    z = np.array(dev.zeros_qz(0.6))
    e = 2 * np.pi / 25
    np.testing.assert_allclose(z, [-2 * e, -e, e, 2 * e], atol=1e-9)
    assert len(lat.zeros_qz(0.6)) == 0


def test_gamma2_is_abs2():
    q = np.linspace(-0.5, 0.5, 11)
    assert np.allclose(lat.gamma2(q), abs(lat.gamma(q)) ** 2)


def test_downstream_flips_qz():
    q = np.linspace(0.01, 0.5, 7)
    d = replace(lat, top_facing="downstream")
    assert np.allclose(d.gamma(q), lat.gamma(-q))


def test_exposed_fraction_inside_line_is_zero():
    assert lat.delta_n_line(0.0, 0, 0, 20.0) == pytest.approx(lat.dn_A, abs=1e-12)
    want = lat.dn_A * (1 + (lat.r - 1) * np.exp(-(40 - 20) / 25))
    assert lat.delta_n_line(40.0, 0, 0, 20.0) == pytest.approx(want, rel=1e-6)
    assert dev.delta_n_line(0.0, 0, 0, 10.0) == pytest.approx(dev.dn_A, abs=1e-12)
    assert dev.delta_n_line(40.0, 0, 0, 10.0) == pytest.approx(0, abs=1e-12)
    assert dev.delta_n_line(0.0, 0, 0, 30.0) == 0
    assert dev.delta_n_line(0.0, 0, 0, -1.0) == 0


def test_edges_displaced():
    # right edge moved +8: x=26 (outside nominal) is now inside the line
    assert abs(dev.delta_n_line(26.0, 0, 8.0, 10.0)) > 0.9 * abs(dev.dn_A)
    assert abs(dev.delta_n_line(26.0, 0, 0, 10.0)) < 0.15 * abs(dev.dn_A)


@pytest.mark.parametrize("obj", [lat, dev])
def test_f_m_matches_fft_of_g(obj):
    x = (np.arange(4096) + 0.5) / 4096 * obj.L - obj.L / 2
    G = 2 * np.pi / obj.L
    got = np.mean(obj.g(x, 0.1) * np.exp(-1j * G * x))
    assert obj.f_m(1, 0.1) == pytest.approx(got, rel=1e-4)


def _num_g(xi, q):
    def integrand(z, part):
        v = lat.delta_n_line(xi, 0, 0, z) / lat.dn_A * np.exp(-1j * q * z)
        return float(getattr(v, part))

    re = quad(integrand, 0, 40, args=("real",), limit=200)[0]
    im = quad(integrand, 0, 40, args=("imag",), limit=200)[0]
    return re + 1j * im


def test_g_matches_numeric_depth_integral():
    x = np.array([0.0, 20.0, 40.0])
    for q in (0.0, 0.2):
        num = [_num_g(xi, q) for xi in x]
        assert np.allclose(lat.g(x, q), num, rtol=1e-4, atol=1e-6)


def test_mu_h_positive_finite():
    assert np.isfinite(lat.mu_h) and lat.mu_h > 0


def test_to_index_map_2d_shape_and_values():
    m = to_index_map_2d(lat, nx=160, nz=40, energy_ev=285.0)
    assert m.x_nm.size == 160
    val = -m.delta + 1j * m.beta
    assert val[80, 20] == pytest.approx(lat.dn_A, rel=1e-3)


@pytest.mark.parametrize("q", [-0.3, -0.05, 0.05, 0.3])
def test_latent_gamma_equals_line_minus_space_column(q):
    assert lat.gamma(q) == pytest.approx(lat.g(0.0, q) - lat.g(lat.L / 2, q), rel=1e-6)


@pytest.mark.parametrize("obj", [lat, dev])
def test_mu_h_matches_numeric_mean_and_scales_with_energy(obj):
    x = (np.arange(20000) + 0.5) / 20000 * obj.L - obj.L / 2
    f0 = np.mean(obj.g(x, 0.0))
    k = 2 * np.pi * obj.energy_ev / 1239.841984
    assert obj.mu_h == pytest.approx(2 * k * np.imag(obj.dn_A * f0), rel=1e-4)
    doubled = replace(obj, energy_ev=2 * obj.energy_ev)
    assert doubled.mu_h == pytest.approx(2 * obj.mu_h, rel=1e-12)


def _born_orders(xs, dz, omegas=(0.0, 10.0)):
    from cd_reduce.simulation import simulate_born_2d

    m2 = to_index_map_2d(xs, nx=160, nz=round(xs.height / dz), energy_ev=285.0)
    res = simulate_born_2d(m2, omega_deg=np.asarray(omegas), max_order=4)
    return {m: np.asarray(res.order_efficiencies[m]) for m in (1, 2, 3, 4)}


def test_z_filling_converges_in_dz():
    xs = LatentFilm(
        80.0,
        45.0,
        2.5,
        40.0,
        4.0,
        25.0,
        1.15 + 0.05j,
        -2e-3 + 1.5e-3j,
        energy_ev=285.0,
    )
    ref = _born_orders(xs, 0.05)
    tol = {2.0: None, 1.0: 0.015, 0.5: 0.005}
    for dz, t in tol.items():
        got = _born_orders(xs, dz)
        err = max(float(np.max(np.abs(got[m] / ref[m] - 1))) for m in got)
        print(f"dz={dz}: max rel err over m=1..4, omega=0,10: {err:.4f}")
        for m in got:
            print(f"  m={m}: {np.abs(got[m] / ref[m] - 1)}")
        if t is not None:
            assert err <= t, (dz, err)
