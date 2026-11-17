"""Rough-grating ensemble scattering on the supercell mode grid (Born engine)."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from cd_reduce.reduction.roughness.cross_section import LatentFilm, to_index_map_2d
from cd_reduce.reduction.roughness.psd_models import psd_palasantzas
from cd_reduce.simulation import simulate_born_2d, simulate_rough_ensemble
from cd_reduce.simulation.structures import (
    RefractiveIndexMap3D,
    build_rough_grating_3d,
)

# Same values as the validation reference grating (redefined: no tests->tests import).
PSD_LPR = partial(psd_palasantzas, sigma_nm=1.0, xi_nm=25.0, hurst=0.7)
PSD_LWR = partial(psd_palasantzas, sigma_nm=1.5, xi_nm=12.0, hurst=0.5)


def _latent(dn_a: complex = -2e-3 + 1.5e-3j) -> LatentFilm:
    return LatentFilm(
        80.0, 45.0, 2.5, 40.0, 4.0, 25.0, 1.15 + 0.05j, dn_a, energy_ev=285.0
    )


def _random_map(rng, nx, ny, nz, dx, dy, dz, energy_ev=285.0) -> RefractiveIndexMap3D:
    return RefractiveIndexMap3D(
        x_nm=(np.arange(nx) + 0.5) * dx - 3.0,
        y_nm=(np.arange(ny) + 0.5) * dy + 1.7,
        z_nm=(np.arange(nz) + 0.5) * dz,
        delta=1e-3 * rng.standard_normal((nx, ny, nz)),
        beta=1e-3 * rng.standard_normal((nx, ny, nz)),
        energy_ev=energy_ev,
        pitch_nm=nx * dx,
    )


def test_born_matches_direct_sum():
    m3 = _random_map(np.random.default_rng(1), 8, 6, 4, 2.0, 3.0, 1.5)
    om = 7.0
    res = simulate_rough_ensemble(
        lambda _rng: m3, [om], n_realisations=1, qx_max=10.0, qy_max=10.0
    )
    assert res.coherent.shape == (1, 8, 6)
    k0 = m3.k0_nm_inv
    dn = m3.delta_n
    zc = m3.z_nm - m3.z_nm.mean()
    picks = [(5, 3), (3, 3), (4, 4), (5, 1), (2, 5)]  # (ix, iy) in shifted arrays
    for ix, iy in picks:
        qx, qy = res.qx[ix], res.qy[iy]
        assert res.valid[0, ix, iy]
        assert not (qx == 0 and qy == 0)
        qz = res.qz[0, ix, iy]
        ph = (
            np.exp(-1j * qx * m3.x_nm)[:, None, None]
            * np.exp(-1j * qy * m3.y_nm)[None, :, None]
            * np.exp(-1j * qz * zc)[None, None, :]
        )
        direct = (
            (-1j * k0 / np.cos(np.deg2rad(om)))
            * np.sum(dn * ph)
            * m3.dx_nm
            * m3.dy_nm
            * m3.dz_nm
            / (m3.lx_nm * m3.ly_nm)
        )
        kx_in = k0 * np.sin(np.deg2rad(om))
        qz_expect = np.sqrt(k0**2 - (kx_in + qx) ** 2 - qy**2) - k0 * np.cos(
            np.deg2rad(om)
        )
        assert qz == pytest.approx(qz_expect, rel=1e-12, abs=1e-14)
        assert abs(res.coherent[0, ix, iy] - direct) <= 1e-10 * abs(direct)


def test_born_y_uniform_matches_born_2d():
    m2 = to_index_map_2d(_latent(), nx=160, nz=40, energy_ev=285.0)
    m3 = m2.to_3d(ny=8, ly_nm=64.0)
    omegas = np.array([-12.0, -3.0, 0.0, 5.0, 20.0])
    res = simulate_rough_ensemble(lambda _rng: m3, omegas, n_realisations=1)
    b2 = simulate_born_2d(m2, omega_deg=omegas, max_order=4)
    worst = 0.0
    for m in (-4, -3, -2, -1, 1, 2, 3, 4):
        eff3 = res.order_efficiency(m)
        eff2 = b2.order_efficiencies[m]
        np.testing.assert_allclose(eff3, eff2, rtol=1e-6)
        worst = max(worst, float(np.max(np.abs(eff3 / eff2 - 1))))
        assert np.all(np.abs(res.order_qz(m) - b2.order_qz[m]) < 1e-12)
    np.testing.assert_allclose(
        res.order_efficiency(0), b2.order_efficiencies[0], rtol=1e-6
    )
    print(f"max rel err 3D-vs-2D order efficiency: {worst:.3e}")


def test_evanescent_modes_invalid():
    m3 = _random_map(np.random.default_rng(2), 64, 4, 4, 0.5, 4.0, 2.0)
    om = 40.0
    res = simulate_rough_ensemble(
        lambda _rng: m3, [om], n_realisations=3, qx_max=10.0, qy_max=10.0
    )
    k0 = m3.k0_nm_inv
    kx = k0 * np.sin(np.deg2rad(om)) + res.qx[:, None]
    arg = k0**2 - kx**2 - res.qy[None, :] ** 2
    expected_invalid = arg <= 0
    assert expected_invalid.any() and (~expected_invalid).any()
    np.testing.assert_array_equal(res.valid[0], ~expected_invalid)
    bad = ~res.valid[0]
    assert np.all(np.isnan(res.coherent[0][bad]))
    assert np.all(np.isnan(res.diffuse[0][bad]))
    assert np.all(np.isnan(res.qz[0][bad]))
    good = res.valid[0]
    assert np.all(np.isfinite(res.coherent[0][good]))
    assert np.all(np.isfinite(res.qz[0][good]))


def test_unknown_engine_raises():
    m3 = _random_map(np.random.default_rng(3), 8, 4, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="engine"):
        simulate_rough_ensemble(lambda _rng: m3, [0.0], engine="fdtd")
    with pytest.raises(NotImplementedError):
        simulate_rough_ensemble(lambda _rng: m3, [0.0], engine="multislice")


def test_diffuse_estimators_against_stored_realisations():
    """diffuse = sum|A-<A>|^2/(n-1); diffuse_var = var(|A-<A>|^2, ddof=1)/n."""
    base = np.random.default_rng(4)
    maps = [_random_map(base, 8, 4, 3, 1.0, 2.0, 1.0) for _ in range(5)]
    it = iter(maps)
    res = simulate_rough_ensemble(
        lambda _rng: next(it), [3.0], n_realisations=5, qx_max=10.0, qy_max=10.0
    )
    singles = [
        simulate_rough_ensemble(
            lambda _rng, m=m: m, [3.0], n_realisations=1, qx_max=10.0, qy_max=10.0
        ).coherent[0]
        for m in maps
    ]
    a = np.stack(singles)
    i0 = int(np.argmin(np.abs(res.qx)))
    j0 = int(np.argmin(np.abs(res.qy)))
    a[:, i0, j0] -= 1.0  # zero-order coherent carries the unit beam
    mean = a.mean(axis=0)
    d = np.abs(a - mean) ** 2
    good = res.valid[0].copy()
    good[i0, j0] = False
    np.testing.assert_allclose(res.diffuse[0][good], d.sum(0)[good] / 4, rtol=1e-8)
    np.testing.assert_allclose(
        res.diffuse_var[0][good], d.var(0, ddof=1)[good] / 5, rtol=1e-6
    )
    np.testing.assert_allclose(res.coherent[0][good], mean[good], rtol=1e-10)
    assert np.isnan(res.diffuse[0, i0, j0])


def test_seeded_realisations_reproducible():
    seen: list[list[float]] = [[], []]

    def make(store):
        def f(rng):
            store.append(float(rng.random()))
            return _random_map(np.random.default_rng(5), 8, 4, 2, 1.0, 1.0, 1.0)

        return f

    simulate_rough_ensemble(make(seen[0]), [0.0], n_realisations=3, seed=11)
    simulate_rough_ensemble(make(seen[1]), [0.0], n_realisations=3, seed=11)
    assert seen[0] == seen[1]
    assert len(set(seen[0])) == 3


def test_debye_waller_from_ensemble():
    # The latent film of tests/simulation/test_edge_fields.py with |dn_A| = 1e-5.
    dn_a = -2e-3 + 4e-4j
    lat_low = LatentFilm(
        L=100.0,
        w=40.0,
        s=3.0,
        h=40.0,
        dh=4.0,
        lam=25.0,
        r=0.6 + 0.05j,
        dn_A=1e-5 * dn_a / abs(dn_a),
        energy_ev=285.0,
    )
    kw = {"n_lines": 8, "ly_nm": 1024.0, "dx_nm": 1.0, "dy_nm": 4.0, "dz_nm": 2.0}
    s2_samples: list[float] = []

    def make_rough(rng):
        st = build_rough_grating_3d(
            lat_low, psd_lpr=PSD_LPR, psd_lwr=PSD_LWR, rng=rng, **kw
        )
        p, w = st.metadata["P_nm"], st.metadata["W_nm"]
        s2_samples.append(float(np.mean(p**2 + w**2 / 4)))
        return st

    smooth = build_rough_grating_3d(lat_low, rng=0, **kw)
    rough = simulate_rough_ensemble(make_rough, [0.0], n_realisations=16, seed=0)
    ref = simulate_rough_ensemble(lambda _rng: smooth, [0.0], n_realisations=1)
    s2 = float(np.mean(s2_samples))
    g = 2 * np.pi / lat_low.L
    for m in (1, 2, 3, 4):
        ratio = rough.order_efficiency(m)[0] / ref.order_efficiency(m)[0]
        expect = np.exp(-(m**2) * g**2 * s2)
        print(
            f"m={m} ratio={ratio:.5f} expect={expect:.5f} rel={ratio / expect - 1:+.4f}"
        )
        assert ratio == pytest.approx(expect, rel=0.03)
