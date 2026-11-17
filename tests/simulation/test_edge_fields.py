"""Edge fields and rough 3D structures."""

from __future__ import annotations

import numpy as np
import pytest

from cd_reduce.reduction.roughness.cross_section import LatentFilm, to_index_map_2d
from cd_reduce.reduction.roughness.psd_models import psd_palasantzas
from cd_reduce.simulation.structures import (
    build_grating_3d,
    build_rough_grating_3d,
    generate_edge_field,
)


def S(f):
    return psd_palasantzas(f, 1.0, 25.0, 0.7)


lat = LatentFilm(
    L=100.0,
    w=40.0,
    s=3.0,
    h=40.0,
    dh=4.0,
    lam=25.0,
    r=0.6 + 0.05j,
    dn_A=-2e-3 + 4e-4j,
    energy_ev=285.0,
)


def test_edge_field_variance_matches_band_integral():
    ny, ly = 512, 2048.0
    rng = np.random.default_rng(0)
    var = np.mean([np.var(generate_edge_field(S, ny, ly, rng)) for _ in range(400)])
    f = np.fft.fftfreq(ny, d=ly / ny)
    k = np.arange(1, ny // 2)
    expected = 2 * np.sum(S(f[k])) / ly + S(np.abs(f[ny // 2])) / ly
    assert var == pytest.approx(expected, rel=0.05)


def test_edge_field_periodogram_matches_psd():
    ny, ly = 512, 2048.0
    rng = np.random.default_rng(1)
    pg = (
        np.mean(
            [
                np.abs(np.fft.fft(generate_edge_field(S, ny, ly, rng))) ** 2
                for _ in range(400)
            ],
            axis=0,
        )
        * (ly / ny) ** 2
        / ly
    )
    f = np.fft.fftfreq(ny, d=ly / ny)
    k = np.arange(1, 51)
    assert 0.95 < np.median(pg[k] / S(f[k])) < 1.05


def test_edge_field_none_is_zeros():
    assert np.all(generate_edge_field(None, 16, 100.0, None) == 0)


def test_build_grating_3d_lpr_only_moves_edges_together():
    md = build_grating_3d(psd_lpr=S, psd_lwr=None, nx=32, ny=64, nz=8).metadata
    assert np.any(md["left_edge_dx_nm"] != 0)
    assert np.allclose(md["left_edge_dx_nm"], md["right_edge_dx_nm"])
    assert md["P_nm"].shape == (1, 64)


def test_build_grating_3d_lwr_only_is_antisymmetric():
    md = build_grating_3d(psd_lpr=None, psd_lwr=S, nx=32, ny=64, nz=8).metadata
    assert np.any(md["left_edge_dx_nm"] != 0)
    assert np.allclose(md["left_edge_dx_nm"], -md["right_edge_dx_nm"])
    assert md["W_nm"].shape == (1, 64)


def test_build_rough_grating_3d_smooth_equals_extruded_2d():
    g = build_rough_grating_3d(
        lat, n_lines=2, ly_nm=64, dx_nm=1.0, dy_nm=4.0, dz_nm=1.0
    )
    ref = to_index_map_2d(lat, nx=100, nz=40, energy_ev=285.0)
    assert g.delta.shape == (200, 16, 40)
    for iy in range(16):
        assert np.allclose(g.delta[:, iy, :], np.tile(ref.delta, (2, 1)), atol=1e-12)
        assert np.allclose(g.beta[:, iy, :], np.tile(ref.beta, (2, 1)), atol=1e-12)
    assert g.pitch_nm == 100.0 and g.energy_ev == 285.0
    assert g.metadata["cross_section"] is lat


def test_build_rough_grating_3d_rejects_overlap():
    with pytest.raises(ValueError):
        build_rough_grating_3d(
            lat,
            n_lines=2,
            ly_nm=256,
            dx_nm=1.0,
            dy_nm=4.0,
            dz_nm=1.0,
            psd_lpr=lambda f: psd_palasantzas(f, 30.0, 25.0, 0.7),
            rng=0,
        )


def test_build_rough_grating_3d_with_psds():
    small = lambda f: psd_palasantzas(f, 0.5, 25.0, 0.7)  # noqa: E731
    g = build_rough_grating_3d(
        lat,
        n_lines=3,
        ly_nm=256,
        dx_nm=1.0,
        dy_nm=4.0,
        dz_nm=1.0,
        psd_lpr=small,
        psd_lwr=small,
        rng=3,
    )
    assert g.metadata["P_nm"].shape == (3, 64)
    assert g.metadata["W_nm"].shape == (3, 64)
    assert g.delta.shape == (300, 64, 40)
    assert np.ptp(g.delta[:, :, 5], axis=1).max() > 0  # varies along y
