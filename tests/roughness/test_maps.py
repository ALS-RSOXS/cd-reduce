import numpy as np
import pytest

from cd_reduce.reduction.geometry import Geometry
from cd_reduce.reduction.mapping import QMap2D, map_2d
from cd_reduce.reduction.roughness import (
    ReducedPixels,
    detector_q_pitch,
    map_reduced,
    plot_reduced_map,
)


def _px(Qx, Qy, Qz, D, dpp):
    n = len(D)
    z = np.zeros(n)
    D = np.asarray(D, float)
    return ReducedPixels(
        Qx=np.asarray(Qx, float), Qy=np.asarray(Qy, float), Qz=np.asarray(Qz, float),
        f=z, D=D, M=D, var_M=z + 1.0, energy_ev=z, omega_deg=z,
        frame_id=np.zeros(n, dtype=np.int64), d_per_photon=np.asarray(dpp, float),
    )  # fmt: skip


def test_map_reduced_exposure_weighting_arithmetic():
    # photons 2 and 0 with d = 0.5 and 1.5: D = 1.0 and 0.0
    p = _px([0.1, 0.1], [0, 0], [0.2, 0.2], [1.0, 0.0], [0.5, 1.5])
    qm = map_reduced(
        p, "qx-qz", slab_halfwidth=0.1, bin_q=(0.2, 0.4),
        x_range=(0, 0.2), y_range=(0, 0.4),
    )  # fmt: skip
    assert qm.H[0, 0] == pytest.approx(2.0)
    assert qm.N[0, 0] == pytest.approx(1 / 0.5 + 1 / 1.5)
    assert qm.mean[0, 0] == pytest.approx(2.0 / (2 + 2 / 3))
    assert qm.metadata["quantity"] == "D"


def test_map_reduced_low_count_unbiased():
    # Uniform true D; d varies x3; ~2 photons/pixel. Inverse-variance weighting
    # (1/var_obs, zero-count pixels dropped) is biased here and fails this test.
    rng = np.random.default_rng(1)
    nb, per = 2000, 40
    D0 = 1.0
    dpp = rng.uniform(1.0, 3.0, nb * per) * (2.0 / 2.0)
    photons = rng.poisson(D0 / dpp)
    D = photons * dpp
    qx = np.repeat((np.arange(nb) + 0.5) * 1e-3, per)
    p = _px(qx, np.zeros_like(qx), np.full_like(qx, 0.1), D, dpp)
    qm = map_reduced(
        p, "qx-qz", slab_halfwidth=0.1, bin_q=(1e-3, 0.2),
        x_range=(0, nb * 1e-3), y_range=(0, 0.2),
    )  # fmt: skip
    assert qm.H.sum() / qm.N.sum() == pytest.approx(D0, rel=0.01)


def test_map_reduced_slab():
    p = _px([0.1, 0.1, 0.1], [0.0, 0.05, 0.3], [0.2] * 3, [1, 1, 100], [1, 1, 1])
    qm = map_reduced(
        p, "qx-qz", slab_center=0.0, slab_halfwidth=0.1, bin_q=(0.2, 0.4),
        x_range=(0, 0.2), y_range=(0, 0.4),
    )  # fmt: skip
    assert qm.metadata["slab_halfwidth"] == 0.1
    assert qm.metadata["slab_center"] == 0.0
    assert qm.mean[0, 0] == pytest.approx(1.0)
    qm2 = map_reduced(
        p, "qy-qz", slab_halfwidth=0.2, bin_q=(0.1, 0.4),
        x_range=(0, 0.4), y_range=(0, 0.4),
    )  # fmt: skip
    assert qm2.x_domain == "Qy"


def test_map_reduced_empty_bins_nan():
    p = _px([0.05], [0], [0.1], [1], [1])
    qm = map_reduced(
        p, "qx-qz", slab_halfwidth=0.1, bin_q=(0.1, 0.1),
        x_range=(0, 0.2), y_range=(0, 0.2),
    )  # fmt: skip
    m = qm.mean
    assert np.isfinite(m[0, 1]) and np.isnan(m[1, 1]) and np.isnan(m[0, 0])
    ax = plot_reduced_map(qm, w=45.0)
    assert ax.get_title()


def test_detector_q_pitch_small_angle():
    g = Geometry(u0_px=256, v0_px=256, Lb_px=985, Lambda_nm=80)
    k = 2 * np.pi * 285.0 / 1239.84198
    assert detector_q_pitch(g, 285.0) == pytest.approx(k / 985, rel=0.02)


def _frames():
    from cd_reduce.reduction.frames import Frame

    g = Geometry(u0_px=64, v0_px=64, Lb_px=985, Lambda_nm=80)
    fr = [
        Frame.from_array(
            np.ones((128, 128)), energy_ev=285.0, omega_deg=om, exposure_s=1.0
        )
        for om in (0.0, 5.0)
    ]  # fmt: skip
    return fr, g


def test_map_2d_bin_q_sets_bin_count():
    fr, g = _frames()
    qm = map_2d(
        fr, g, x_range=(-0.4, 0.4), y_range=(0.0, 0.2), bins=(10, 10),
        bin_q=(0.001, 0.002), bg_mode="none",
    )  # fmt: skip
    assert qm.H.shape == (800, 100)


def _point(nx, half):
    x = np.linspace(-1, 1, nx + 1)
    H = np.zeros((nx, nx))
    H[nx // 2, nx // 2] = 1.0
    N = np.ones_like(H)
    return QMap2D(H=H, N=N, x_edges=x, y_edges=x.copy(), x_domain="Qx", y_domain="Qz")


def test_smoothed_default_is_unsmoothed():
    qm = _point(40, 1)
    np.testing.assert_array_equal(qm.smoothed(), qm.mean)


def test_smoothed_sigma_q_in_inverse_nm():
    widths = []
    for nx in (100, 200):  # bin width halves
        qm = _point(nx, 1)
        img = qm.smoothed(sigma_q=(0.1, 0.1))
        prof = np.nan_to_num(img[:, nx // 2])
        xc = qm.x_centers
        prof = prof / prof.sum()
        widths.append(np.sqrt(np.sum(prof * xc**2) - np.sum(prof * xc) ** 2))
    assert widths[0] == pytest.approx(widths[1], rel=0.05)
    assert widths[0] == pytest.approx(0.1, rel=0.1)


def test_map_reduced_mean_no_floor_when_exposure_below_one():
    # d_per_photon >> 1 so N = sum(1/d) << 1; mean must still be H / N.
    p = _px([0.1, 0.1], [0, 0], [0.2, 0.2], [50.0, 150.0], [25.0, 25.0])
    qm = map_reduced(
        p, "qx-qz", slab_halfwidth=0.1, bin_q=(0.2, 0.4),
        x_range=(0, 0.2), y_range=(0, 0.4),
    )  # fmt: skip
    assert qm.N[0, 0] == pytest.approx(0.08)
    assert qm.mean[0, 0] == pytest.approx(qm.H[0, 0] / qm.N[0, 0])
    assert qm.mean[0, 0] == pytest.approx(100.0)
    assert qm.smoothed(dilate_bins=(1, 1))[0, 0] == pytest.approx(100.0)
    tr = qm.slice_1d("Qx")
    assert tr.intensity[0] == pytest.approx(100.0)
    # exposure-weighted sigma = sqrt(H) / N (H = detected photons = 8)
    assert tr.sigma[0] == pytest.approx(np.sqrt(8.0) / 0.08)


def test_count_map_mean_and_slice_unchanged():
    H = np.array([[4.0, 0.0], [9.0, 2.0]])
    N = np.array([[2.0, 0.0], [3.0, 1.0]])
    qm = QMap2D(
        H=H, N=N, x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0, 2.0]), x_domain="Qx", y_domain="Qz",
    )  # fmt: skip
    np.testing.assert_allclose(qm.mean, [[2.0, np.nan], [3.0, 2.0]])
    tr = qm.slice_1d("Qx")
    np.testing.assert_allclose(tr.intensity, [2.0, 11.0 / 4.0])
    np.testing.assert_allclose(tr.sigma, [2.0 / np.sqrt(2.0), 2.75 / 2.0])
