"""Unit and integration tests for `cd_reduce.simulation` (Born & Multislice)."""

from __future__ import annotations

import matplotlib
import numpy as np

from cd_reduce.reduction.roughness.psd_models import psd_palasantzas

matplotlib.use("Agg")

from cd_reduce import (
    Geometry,
    GratingLayerSpec,
    RefractiveIndexMap2D,
    RefractiveIndexMap3D,
    build_grating_2d,
    build_grating_3d,
    generate_edge_field,
    plot_born_vs_multislice_grid,
    plot_multislice_wavefield,
    plot_simulation_comparison,
    plot_structure_2d,
    plot_structure_3d,
    simulate_born_2d,
    simulate_born_3d,
    simulate_multislice_2d,
    simulate_multislice_3d,
)


def test_refractive_index_maps_and_builders() -> None:
    # 1. General 2D map from complex n and from delta/beta
    nx, nz = 64, 32
    n_complex = np.ones((nx, nz), dtype=np.complex128) - 1e-3 + 2e-4j
    m2d = RefractiveIndexMap2D.from_complex_n(
        n_complex, dx_nm=1.5, dz_nm=2.0, energy_ev=285.0
    )
    assert m2d.nx == nx
    assert m2d.nz == nz
    assert np.allclose(m2d.delta, 1e-3)
    assert np.allclose(m2d.beta, 2e-4)

    m2d_db = RefractiveIndexMap2D.from_delta_beta(
        m2d.delta, m2d.beta, dx_nm=1.5, dz_nm=2.0
    )
    assert np.allclose(m2d_db.n, m2d.n)

    # Extrude 2D -> 3D and slice 3D -> 2D
    m3d_ext = m2d.to_3d(ny=16, ly_nm=160.0)
    assert (m3d_ext.nx, m3d_ext.ny, m3d_ext.nz) == (nx, 16, nz)
    m2d_slice = m3d_ext.slice_2d_xz(y_index=4)
    assert np.allclose(m2d_slice.delta, m2d.delta)

    m3d_cn = RefractiveIndexMap3D.from_complex_n(
        m3d_ext.n, dx_nm=1.5, dy_nm=10.0, dz_nm=2.0
    )
    m3d_db = RefractiveIndexMap3D.from_delta_beta(
        m3d_ext.delta, m3d_ext.beta, dx_nm=1.5, dy_nm=10.0, dz_nm=2.0
    )
    assert np.allclose(m3d_cn.n, m3d_db.n)

    # 2. Multi-layer trapezoidal grating builder
    layers = [
        GratingLayerSpec(
            thickness_nm=50.0,
            cd_bottom_nm=50.0,
            sidewall_angle_deg=86.0,
            delta_line=2e-3,
            beta_line=3e-4,
        ),
        GratingLayerSpec(
            thickness_nm=20.0,
            cd_bottom_nm=43.0,
            top_rounding_nm=8.0,
            delta_line=1.5e-3,
            beta_line=1e-4,
        ),
    ]
    g2d = build_grating_2d(pitch_nm=100.0, layers=layers, nx=128, nz=70)
    assert g2d.nx == 128
    assert g2d.nz == 70
    assert np.isclose(g2d.lz_nm, 70.0)

    # 3. 1D edge field and 3D rough grating builder
    dx_y = generate_edge_field(
        lambda f: psd_palasantzas(f, 2.0, 30.0, 0.5), 64, 640.0, 7
    )
    assert dx_y.shape == (64,)

    g3d = build_grating_3d(
        pitch_nm=100.0,
        cd_nm=45.0,
        thickness_nm=60.0,
        psd_lpr=lambda f: psd_palasantzas(f, 1.8, 30.0, 0.5),
        nx=64,
        ny=32,
        nz=20,
    )
    assert (g3d.nx, g3d.ny, g3d.nz) == (64, 32, 20)


def test_born_2d_and_multislice_2d_kinematic_equivalence() -> None:
    """In the weak-scattering limit (small delta, beta), Multislice and Born must
    agree closely across all orders.
    """
    struct_weak = build_grating_2d(
        pitch_nm=100.0,
        cd_nm=45.0,
        thickness_nm=40.0,
        sidewall_angle_deg=88.0,
        delta_line=1.0e-5,
        beta_line=2.0e-6,
        nx=128,
        nz=40,
        energy_ev=285.0,
    )
    omegas = np.linspace(-30.0, 30.0, 31)
    born = simulate_born_2d(struct_weak, omega_deg=omegas, max_order=4, mode="correct")
    ms = simulate_multislice_2d(
        struct_weak, omega_deg=omegas, max_order=4, mode="correct"
    )

    for m in (1, 2, 3, -1, -2):
        eff_b = born.order_efficiencies[m]
        eff_m = ms.order_efficiencies[m]
        rel_err = np.linalg.norm(eff_b - eff_m) / max(np.linalg.norm(eff_b), 1e-30)
        assert rel_err < 0.005, (
            f"Order m={m} relative mismatch {rel_err:.4f} exceeds 0.5% in weak limit"
        )

        rod_b = born.bragg_rod(m, x_axis="Qz")
        rod_m = ms.bragg_rod(m, x_axis="Qz")
        assert rod_b.coords.size == omegas.size
        assert rod_m.coords.size == omegas.size


def test_multislice_2d_dynamical_and_detector_projection() -> None:
    """For a strong, absorbing, asymmetric grating, Multislice captures dynamical
    effects.
    """
    layers = [
        GratingLayerSpec(
            thickness_nm=60.0,
            cd_bottom_nm=55.0,
            cd_top_nm=35.0,
            delta_line=4e-3,
            beta_line=1.5e-3,
        ),
        GratingLayerSpec(
            thickness_nm=40.0,
            cd_bottom_nm=35.0,
            cd_top_nm=25.0,
            delta_line=2e-3,
            beta_line=5e-4,
            x_center_nm=6.0,
        ),
    ]
    struct_strong = build_grating_2d(pitch_nm=100.0, layers=layers, nx=128, nz=50)
    omegas = np.linspace(-30.0, 30.0, 25)
    born = simulate_born_2d(struct_strong, omega_deg=omegas, max_order=4)
    ms = simulate_multislice_2d(struct_strong, omega_deg=omegas, max_order=4)

    assert ms.internal_wavefield.shape == (128, 50)
    assert ms.qmap_qx_qz.H.shape == (241, 161)

    geom = Geometry(u0_px=64.0, v0_px=64.0, Lb_px=900.0)
    frame_ms = ms.to_detector_frame(
        geom, omega_deg=0.0, shape=(128, 128), poisson_noise=True, rng=9
    )
    assert frame_ms.image.shape == (128, 128)
    frame_born = born.to_detector_frame(
        geom, omega_deg=0.0, shape=(128, 128), poisson_noise=True, rng=9
    )
    assert frame_born.image.shape == (128, 128)


def test_born_3d_and_multislice_3d_rough_grating() -> None:
    struct3d = build_grating_3d(
        pitch_nm=100.0,
        cd_nm=45.0,
        thickness_nm=50.0,
        delta_line=2.0e-3,
        beta_line=3.0e-4,
        psd_lpr=lambda f: psd_palasantzas(f, 2.0, 25.0, 0.5),
        psd_lwr=lambda f: psd_palasantzas(f, 1.0, 25.0, 0.5),
        nx=64,
        ny=32,
        nz=20,
        ly_nm=320.0,
        seed=11,
    )
    omegas = np.linspace(-25.0, 25.0, 11)
    born3d = simulate_born_3d(
        struct3d,
        omega_deg=omegas,
        max_order=3,
        nqx=61,
        nqy=31,
        nqz=41,
    )
    ms_3d = simulate_multislice_3d(
        struct3d,
        omega_deg=omegas,
        max_order=3,
        nqx=61,
        nqz=41,
    )

    assert born3d.qmap_qx_qz.H.shape == (61, 41)
    assert born3d.qmap_qx_qy.H.shape == (61, 31)
    assert born3d.qmap_qy_qz.H.shape == (31, 41)

    assert ms_3d.qmap_qx_qz.H.shape == (61, 41)
    assert ms_3d.qmap_qy_qz.H.shape == (32, 41)
    assert ms_3d.exit_wave_xy.shape == (64, 32)

    diff_b = born3d.diffuse_qy_trace(order=1)
    diff_m = ms_3d.diffuse_qy_trace(order=1)
    assert diff_b.domain == "Qy"
    assert diff_m.domain == "Qy"

    geom = Geometry(u0_px=48.0, v0_px=48.0, Lb_px=600.0)
    f3d = born3d.to_detector_frame(geom, omega_deg=0.0, shape=(96, 96))
    assert f3d.image.shape == (96, 96)


def test_visualization_helpers() -> None:
    import matplotlib.pyplot as plt

    s2d = build_grating_2d(pitch_nm=80.0, cd_nm=40.0, thickness_nm=40.0, nx=64, nz=25)
    b2d = simulate_born_2d(s2d, omega_deg=np.linspace(-20, 20, 15), max_order=4)
    m2d = simulate_multislice_2d(s2d, omega_deg=np.linspace(-20, 20, 15), max_order=4)

    axes_s2 = plot_structure_2d(s2d, component="both")
    assert len(axes_s2) == 2
    ax_s2_single = plot_structure_2d(s2d, component="delta")
    assert ax_s2_single is not None

    s3d = s2d.to_3d(ny=16, ly_nm=160.0)
    axes_s3 = plot_structure_3d(s3d)
    assert len(axes_s3) == 2

    axes_wf = plot_multislice_wavefield(m2d)
    assert len(axes_wf) == 2

    ax_cmp = plot_simulation_comparison(b2d, m2d, orders=(1, 2))
    assert ax_cmp is not None

    fig_grid, axes_grid = plot_born_vs_multislice_grid([(b2d, m2d)])
    assert fig_grid is not None
    assert axes_grid.shape == (1, 4)

    fig_qmap, ax_qmap = b2d.qmap_qx_qz.plot()
    assert fig_qmap is not None
    assert ax_qmap is not None
    fig_tr, ax_tr = b2d.bragg_rod(1).plot()
    assert fig_tr is not None
    assert ax_tr is not None
    plt.close("all")


def test_plot_structure_3d_surface_renders_relief_and_face() -> None:
    import matplotlib.pyplot as plt

    from cd_reduce import LatentFilm, build_rough_grating_3d, plot_structure_3d_surface

    lat = LatentFilm(
        L=80.0,
        w=45.0,
        s=2.5,
        h=40.0,
        dh=4.0,
        lam=25.0,
        r=1.15 + 0.05j,
        dn_A=-2e-3 + 1.5e-3j,
        energy_ev=285.0,
    )
    st = build_rough_grating_3d(
        lat, n_lines=2, ly_nm=64.0, dx_nm=2.0, dy_nm=4.0, dz_nm=2.0
    )
    fig = plt.figure()
    ax = plot_structure_3d_surface(
        st, ax=fig.add_subplot(projection="3d"), relief_scale=3.0
    )
    assert len(ax.collections) == 2  # top relief + front cut face
    assert ax.surface_norm.vmax > ax.surface_norm.vmin
    plot_structure_3d_surface(st, ax=fig.add_subplot(projection="3d"), color="#4a6fb5")
    plt.close(fig)
