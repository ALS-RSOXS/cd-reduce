from dataclasses import replace

import numpy as np
import pytest

from cd_reduce.reduction.frames import Frame
from cd_reduce.reduction.geometry import Geometry, compute_pixel_q, frame_pixel_grid
from cd_reduce.reduction.roughness.cross_section import DevelopedLine, LatentFilm
from cd_reduce.reduction.roughness.reduce import (
    ReducedPixels,
    absorption_factor,
    cos_theta_out,
    howto_prefactor,
    reduce_frame,
)

HC = 1239.84198
APP = 27.0
KW = {"L": 80, "w": 45, "s": 2.5, "dn_A": -1e-3 + 2e-4j}
lat = LatentFilm(h=40, dh=4, lam=25, r=0.6 + 0.05j, energy_ev=285.0, **KW)
dev = DevelopedLine(h_d=25, energy_ev=285.0, **KW)
dev80 = DevelopedLine(h_d=80, energy_ev=285.0, **KW)
geo = Geometry(u0_px=256, v0_px=256, Lb_px=1970, Lambda_nm=80)
SHAPE = (512, 512)


def _qm(E, om, g=geo):
    xx, yy = frame_pixel_grid(SHAPE, 1)
    return compute_pixel_q(g, xx, yy, E, om, approximation="correct")


def _m_true(qx, w):
    return 4 * np.sin(qx * w / 2) ** 2 * 3.0 + np.cos(qx * w / 2) ** 2 * 5.0


def _frame(
    xs, E=285.0, om=10.0, scale=1e12, m_fn=_m_true, counts=None, g=geo, window_T=1.0
):
    qm = _qm(E, om, g)
    k = 2 * np.pi * E / HC
    W = np.radians(om + g.omega0_deg)
    pref = k**4 * abs(xs.dn_A) ** 2 / (4 * np.pi**2 * xs.L * np.cos(W))
    cos_out = np.cos(W) + qm.Qz / k
    A = np.exp(-(xs.mu_h / 2) * (1 / np.cos(W) + 1 / cos_out))
    Tm = window_T ** (1 / np.cos(W))
    gam = xs.gamma2(qm.Qz) * np.exp(-(qm.Qx**2) * xs.s**2)
    photons = scale * pref * qm.solid_angle_sr * A * Tm * gam * m_fn(qm.Qx, xs.w)
    if counts is not None:
        photons = np.full(SHAPE, counts, dtype=float)
    fr = Frame.from_array(photons * APP, omega_deg=om, energy_ev=E, exposure_s=1.0)
    return fr, qm


def _check(rp, xs, E, om):
    assert rp.M.size > 1000
    np.testing.assert_allclose(rp.M, _m_true(rp.Qx, xs.w), rtol=1e-6)


def test_prefactor_explicit():
    W = np.radians(10.0)
    k = 2 * np.pi * 285.0 / HC
    exp = k**4 * abs(lat.dn_A) ** 2 / (4 * np.pi**2 * lat.L * np.cos(W))
    assert howto_prefactor(k, lat, W) == pytest.approx(exp, rel=1e-12)


def test_cos_out_and_absorption():
    qm = _qm(285.0, 10.0)
    k = 2 * np.pi * 285.0 / HC
    W = np.radians(10.0)
    c = cos_theta_out(qm, k, W)
    np.testing.assert_allclose(c, np.cos(W) + qm.Qz / k)
    a = absorption_factor(0.01, W, c)
    np.testing.assert_allclose(a, np.exp(-0.005 * (1 / np.cos(W) + 1 / c)))


def test_reduce_frame_recovers_M():
    fr, _ = _frame(lat)
    rp = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP)
    _check(rp, lat, 285.0, 10.0)
    assert rp.meta["scale_source"] == "known"
    assert rp.meta["R_B"] == 0.008 and rp.meta["w"] == lat.w
    np.testing.assert_allclose(rp.f, rp.Qy / (2 * np.pi))
    assert np.all(rp.energy_ev == 285.0) and np.all(rp.omega_deg == 10.0)


def test_reduce_frame_masks_bragg_discs():
    fr, _ = _frame(lat)
    rp = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP)
    G = 2 * np.pi / lat.L
    for m in range(-6, 7):
        if m:
            assert not np.any((rp.Qx - m * G) ** 2 + rp.Qy**2 < 0.008**2)


def test_reduce_frame_drops_nan_pixels():
    fr, _ = _frame(lat)
    base = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP)
    img = fr.image.copy()
    img[100:120, 100:120] = np.nan
    bs = np.zeros(SHAPE, bool)
    bs[300:320, 300:320] = True
    fr2 = Frame.from_array(img, omega_deg=10, energy_ev=285.0, exposure_s=1.0)
    rp = reduce_frame(fr2, geo, lat, scale=1e12, adu_per_photon=APP, beamstop_mask=bs)
    assert np.isfinite(rp.M).all()
    assert base.M.size - rp.M.size >= 1  # at least some removed
    # both blocks fully removed from the pixel set (400 each), unless already masked
    qm = _qm(285.0, 10.0)
    for sl in (np.s_[100:120, 100:120], np.s_[300:320, 300:320]):
        for qx, qy in zip(qm.Qx[sl].ravel(), qm.Qy[sl].ravel(), strict=True):
            assert not np.any((rp.Qx == qx) & (rp.Qy == qy))


def test_reduce_frame_masks_gamma_zeros():
    fr, _ = _frame(dev80, om=30.0)
    rp = reduce_frame(fr, geo, dev80, scale=1e12, adu_per_photon=APP, keep=0.0)
    assert rp.M.size > 1000
    z0 = 2 * np.pi / 80
    assert not np.any(np.abs(rp.Qz - z0) < 0.01)
    all_z = dev80.zeros_qz(float(np.nanmax(np.abs(_qm(285.0, 30.0).Qz))))
    assert len(all_z) >= 2
    for z in all_z:
        assert not np.any(np.abs(rp.Qz - z) < 0.01)


def test_reduce_mixed_energies_concat():
    items = []
    for i, E in enumerate((280.0, 288.6)):
        xs = DevelopedLine(
            h_d=25, energy_ev=E, L=80, w=45, s=2.5, dn_A=(-1e-3 + 2e-4j) * (1 + i)
        )
        fr, _ = _frame(xs, E=E)
        items.append(
            reduce_frame(fr, geo, xs, scale=1e12, adu_per_photon=APP, frame_id=0)
        )
    rp = ReducedPixels.concat(items)
    assert rp.M.size == items[0].M.size + items[1].M.size
    np.testing.assert_allclose(rp.M, _m_true(rp.Qx, 45), rtol=1e-6)
    assert set(np.unique(rp.frame_id)) == {0, 1}
    assert set(np.unique(rp.energy_ev)) == {280.0, 288.6}
    sel = rp.select(rp.frame_id == 1)
    assert sel.M.size == items[1].M.size and sel.meta == rp.meta


def test_concat_rejects_mismatched_RB():
    fr, _ = _frame(lat)
    a = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP)
    b = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP, R_B=0.02)
    with pytest.raises(ValueError):
        ReducedPixels.concat([a, b])


def test_var_M_scales_with_counts():
    n = 400.0
    fr, _ = _frame(lat, counts=n)
    rp = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP, keep=0.0)
    assert rp.M.size > 100
    np.testing.assert_allclose(rp.var_M / rp.M**2, 1 / n, rtol=1e-9)


def test_window_T_exponent():
    fr, _ = _frame(lat, window_T=0.5)
    rp = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP, window_T=0.5)
    _check(rp, lat, 285.0, 10.0)


def test_omega0_used():
    g2 = replace(geo, omega0_deg=1.5)
    fr, _ = _frame(lat, g=g2, window_T=0.5)
    rp = reduce_frame(fr, g2, lat, scale=1e12, adu_per_photon=APP, window_T=0.5)
    _check(rp, lat, 285.0, 10.0)


def test_beamstop_shape_mismatch_raises():
    fr, _ = _frame(lat)
    with pytest.raises(ValueError):
        reduce_frame(
            fr,
            geo,
            lat,
            scale=1e12,
            adu_per_photon=APP,
            beamstop_mask=np.zeros((10, 10), bool),
        )


def test_concat_rejects_mismatched_w():
    fr, _ = _frame(lat)
    a = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP)
    b = replace(a, meta={**a.meta, "w": 50.0})
    with pytest.raises(ValueError):
        ReducedPixels.concat([a, b])


def test_masks_have_positive_controls():
    fr, _ = _frame(lat)
    rp = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP, R_B=0.0)
    G = 2 * np.pi / lat.L
    assert any(
        np.any((rp.Qx - m * G) ** 2 + rp.Qy**2 < 0.008**2) for m in range(-6, 7) if m
    )
    fr, _ = _frame(dev80, om=30.0)
    rp = reduce_frame(
        fr,
        geo,
        dev80,
        scale=1e12,
        adu_per_photon=APP,
        gamma_zero_halfwidth=0.0,
        keep=0.0,
    )
    assert np.any(np.abs(rp.Qz - 2 * np.pi / 80) < 0.01)


def test_reduce_frame_uses_omega_refinement():
    from cd_reduce.reduction.geometry import OmegaRefinement

    ref = OmegaRefinement(d_omega0_deg=1.5, c0_px=3.0)
    g_ref = geo.with_omega_refinement(ref, 10.0)
    fr, _ = _frame(lat, g=g_ref, window_T=0.5)
    rp = reduce_frame(
        fr, geo, lat, scale=1e12, adu_per_photon=APP, window_T=0.5,
        omega_refinement=ref,
    )  # fmt: skip
    _check(rp, lat, 285.0, 10.0)
    rp0 = reduce_frame(fr, geo, lat, scale=1e12, adu_per_photon=APP, window_T=0.5)
    assert not np.allclose(rp0.M, _m_true(rp0.Qx, lat.w), rtol=1e-3)


def test_loader_reduce_roughness_passes_omega_refinement():
    from cd_reduce import CdRsoxsLoader
    from cd_reduce.reduction.geometry import OmegaRefinement

    ref = OmegaRefinement(d_omega0_deg=1.5, c0_px=3.0)
    fr, _ = _frame(lat, g=geo.with_omega_refinement(ref, 10.0))
    loader = CdRsoxsLoader(
        frames=[fr], geometry=geo, omega_refinement=ref, bg_mode="none"
    )
    rp = loader.reduce_roughness({285.0: lat}, scale=1e12, adu_per_photon=APP)
    _check(rp, lat, 285.0, 10.0)


def test_loader_extract_psds_no_pixels_raises_clearly():
    from cd_reduce import CdRsoxsLoader

    fr, _ = _frame(lat)
    loader = CdRsoxsLoader(frames=[fr], geometry=geo, bg_mode="none")
    with pytest.raises(ValueError, match="no pixels survived the masks"):
        loader.extract_psds({285.0: lat}, scale=1e12, adu_per_photon=APP, keep=1.5)
