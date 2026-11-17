"""Unit and integration tests for `cd_reduce.reduction` and `CdRsoxsLoader`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")

from cd_reduce import (
    CdRsoxsLoader,
    Frame,
    Geometry,
    OmegaRefinement,
    apply_geometric_corrections,
    build_grating_2d,
    build_order_table,
    compute_pixel_q,
    ewald_trace_qxqz,
    extract_diffuse_cut,
    extract_frame_spots,
    extract_inplane_linecut,
    extract_line_strip,
    extract_offspec_rod_map,
    extract_order_rods_qz,
    extract_qx_by_omega,
    extract_rod_localized_diffuse_qy,
    fit_geometry_with_walkoff,
    fit_joint_geometry,
    fit_strip_background,
    initial_geometry,
    map_2d,
    membrane_transmission,
    mu_si3n4,
    pixel_q,
    plot_detector_strips,
    plot_specular_1d_panels,
    process_frame_strip,
    readoff_edge_psd_born,
    reduce_1d,
    refine_rod_verticality,
    simulate_born_2d,
    subtract_corner_offset,
    subtract_offline_band_background,
    subtract_row_col_offset,
    threshold_photon_events,
    walkoff_factor,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_geometry() -> Geometry:
    return Geometry(
        u0_px=128.0,
        v0_px=128.0,
        Lb_px=1800.0,
        line_rot_deg=-2.5,
        beta_deg=1.2,
        psi_deg=4.0,
        Lambda_nm=100.0,
    )


def test_geometry_serialization_and_shift(tmp_path: Path) -> None:
    geom = _sample_geometry()
    json_path = tmp_path / "geom.json"
    json_path.write_text(json.dumps(geom.to_dict()), encoding="utf-8")
    loaded = Geometry.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    assert loaded == geom

    shifted_det = geom.with_origin_shift(2.5, convention="detector_line")
    phi = np.deg2rad(geom.line_rot_deg + 180.0)
    assert np.isclose(shifted_det.u0_px, geom.u0_px + np.cos(phi) * 2.5)
    assert np.isclose(shifted_det.v0_px, geom.v0_px + np.sin(phi) * 2.5)


def test_compute_pixel_q_domains_and_modes() -> None:
    geom = _sample_geometry()
    px = np.array([[100.0, 128.0, 160.0], [100.0, 128.0, 160.0]])
    py = np.array([[120.0, 128.0, 136.0], [140.0, 140.0, 140.0]])

    pq_correct = compute_pixel_q(geom, px, py, 285.0, 15.0, approximation="correct")
    pq_small = compute_pixel_q(geom, px, py, 285.0, 15.0, approximation="small_angle")
    pq_linear = compute_pixel_q(geom, px, py, 285.0, 15.0, approximation="angle_linear")

    # Direct beam pixel (u0_px, v0_px) should have Q = 0 in all modes
    assert np.isclose(pq_correct.Q[0, 1], 0.0, atol=1e-12)
    assert np.isclose(pq_small.Q[0, 1], 0.0, atol=1e-12)
    assert np.isclose(pq_linear.Q[0, 1], 0.0, atol=1e-12)

    # Check all domain accessors
    for domain in (
        "Qx",
        "Qy",
        "Qz",
        "Qr",
        "signed_Qr",
        "Q",
        "omega",
        "two_theta",
        "px_x",
        "px_y",
    ):
        arr = pq_correct.get_coord(domain)
        assert arr.shape == px.shape

    assert np.allclose(pq_correct.Qr, np.hypot(pq_correct.Qx, pq_correct.Qy))
    assert np.allclose(
        pq_correct.Q, np.sqrt(pq_correct.Qx**2 + pq_correct.Qy**2 + pq_correct.Qz**2)
    )

    # Legacy tuple wrapper
    qx, qy, qz, sa = pixel_q(geom, px, py, 285.0, 15.0, approximation="correct")
    assert np.allclose(qx, pq_correct.Qx)
    assert np.allclose(qy, pq_correct.Qy)
    assert np.allclose(qz, pq_correct.Qz)
    assert np.allclose(sa, pq_correct.solid_angle_sr)

    # Ewald trace helper
    qx_tr, qz_tr = ewald_trace_qxqz(
        geom, 15.0, 285.0, shape=(256, 256), approximation="correct"
    )
    assert qx_tr.size > 10
    assert qz_tr.size == qx_tr.size


def test_spots_and_geometry_calibration_roundtrip() -> None:
    geom_true = Geometry(
        u0_px=128.0,
        v0_px=128.0,
        Lb_px=1500.0,
        line_rot_deg=0.0,
        beta_deg=0.0,
        psi_deg=0.0,
        Lambda_nm=100.0,
    )
    struct = build_grating_2d(
        pitch_nm=100.0, cd_nm=45.0, thickness_nm=60.0, nx=128, nz=30
    )
    sim = simulate_born_2d(struct, omega_deg=[-20.0, 0.0, 20.0], max_order=4)

    rng = np.random.default_rng(42)
    frames: list[Frame] = []
    for idx, om in enumerate([-20.0, 0.0, 20.0]):
        base_f = sim.to_detector_frame(
            geom_true,
            omega_deg=om,
            shape=(256, 256),
            spot_sigma_q=0.006,
            flux_scale=2e7,
            background_adu=20.0,
        )
        noisy_img = base_f.image + rng.normal(0.0, 1.5, size=base_f.image.shape)
        f = Frame.from_array(
            noisy_img,
            omega_deg=om,
            energy_ev=285.0,
            exposure_s=1.0,
            path=f"frame_{idx:03d}.fits",
        )
        frames.append(f)

    spots_0 = extract_frame_spots(frames[1])
    assert len(spots_0) >= 4

    loader = CdRsoxsLoader(frames=frames, pitch_nm=100.0)
    cal = loader.calibrate_geometry(
        start=geom_true,
        fix_lambda=100.0,
        fix_Lb=1500.0,
    )
    assert cal.n_bragg >= 6
    assert abs(cal.geometry.u0_px - geom_true.u0_px) < 3.0
    assert abs(cal.geometry.v0_px - geom_true.v0_px) < 3.0

    spots_df = loader.extract_spots()
    g_init = initial_geometry(spots_df, Lb_px=1500.0, Lambda_nm=100.0)
    assert abs(g_init.u0_px - 128.0) < 10.0

    cal_walk = fit_geometry_with_walkoff(spots_df, geom_true, fix_lambda=100.0)
    assert cal_walk.n_bragg >= 6

    joint = fit_joint_geometry({"mount_a": spots_df}, {"mount_a": geom_true})
    assert "mount_a" in joint.geometries

    order_table = build_order_table(spots_df, geom_true)
    assert "order" in order_table.columns
    ref = refine_rod_verticality({"mount_a": spots_df}, {"mount_a": geom_true})
    assert isinstance(ref["mount_a"], OmegaRefinement)


def test_corrections_and_backgrounds() -> None:
    geom = Geometry(
        u0_px=48.0, v0_px=48.0, Lb_px=600.0, line_rot_deg=0.0, Lambda_nm=100.0
    )
    ny, nx = 96, 96
    rng = np.random.default_rng(123)
    img = rng.normal(loc=50.0, scale=2.0, size=(ny, nx))

    img_sub, c_off, c_sig = subtract_corner_offset(img, corner_px=12)
    assert np.isclose(c_off, 50.0, atol=1.5)
    assert c_sig > 0.0
    assert abs(float(np.mean(img_sub))) < 1.5

    yy, xx = np.meshgrid(
        np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
    )
    pq = compute_pixel_q(geom, xx, yy, 285.0, 0.0)
    mask = np.ones((ny, nx), dtype=bool)

    sub_rc = subtract_row_col_offset(img_sub, geom, pq.Qy, mask)
    assert sub_rc.shape == img.shape

    qx_edges = np.linspace(-0.2, 0.2, 41)
    sub_band, bg_band = subtract_offline_band_background(
        img_sub, pq.Qx, pq.Qy, geom.g_inv_nm, qx_edges
    )
    assert sub_band.shape == img.shape
    assert bg_band.shape == (40,)

    photons, n_events = threshold_photon_events(img_sub, energy_ev=285.0)
    assert np.all(photons >= 0.0)
    assert isinstance(n_events, int)

    # Si3N4 membrane and walk-off corrections
    mu = float(mu_si3n4(285.0))
    assert mu > 0.0
    trans = np.asarray(
        membrane_transmission(
            np.array([0.0, 30.0]), energy_ev=285.0, thickness_nm=100.0
        )
    )
    assert trans[1] < trans[0] < 1.0

    wf = np.asarray(walkoff_factor(np.array([-20.0, 0.0, 20.0]), a=0.01, b=0.002))
    assert wf.shape == (3,)

    corr = apply_geometric_corrections(
        img_sub,
        pq,
        exposure_s=2.0,
        solid_angle=True,
        polarization=True,
        walkoff_f=0.98,
        membrane_t=0.85,
    )
    assert corr.shape == img.shape

    # Strip background & strip processor on a small synthetic frame
    synth = 5.0 + 120.0 * np.exp(-0.5 * ((yy - 48.0) / 3.0) ** 2) * np.exp(
        -0.5 * ((xx - 68.0) / 2.5) ** 2
    )
    bg_fit = fit_strip_background(
        geom, synth - 5.0, pq.Qx, pq.Qy, mask, energy_ev=285.0, exposure_s=1.0
    )
    assert len(bg_fit) == 3
    strip_res = process_frame_strip(
        geom, synth, energy_ev=285.0, omega_deg=0.0, mask=mask, exposure_s=1.0
    )
    assert strip_res.cleaned_adu.shape == synth.shape


def test_mapping_2d_all_combinations_and_1d_reduction() -> None:
    import matplotlib.pyplot as plt

    geom = Geometry(
        u0_px=64.0,
        v0_px=64.0,
        Lb_px=550.0,
        line_rot_deg=0.0,
        beta_deg=0.0,
        psi_deg=0.0,
        Lambda_nm=100.0,
    )
    struct = build_grating_2d(
        pitch_nm=100.0, cd_nm=45.0, thickness_nm=70.0, nx=128, nz=30
    )
    sim = simulate_born_2d(
        struct, omega_deg=[-25.0, -10.0, 0.0, 10.0, 25.0], max_order=3
    )
    frames = [
        sim.to_detector_frame(geom, omega_deg=om, shape=(128, 128), spot_sigma_q=0.012)
        for om in [-25.0, -10.0, 0.0, 10.0, 25.0]
    ]

    loader = CdRsoxsLoader(frames=frames, geometry=geom, pitch_nm=100.0, bg_mode="none")

    # 1. Standard (Qx, Qz) map in both 'correct' and 'small_angle' modes
    qmap_qx_qz = loader.to_qmap_2d("Qx", "Qz", bins=(60, 50))
    assert qmap_qx_qz.H.shape == (60, 50)

    loader.approximation = "small_angle"
    assert loader.process_vars["reprocess_vars"] is True
    qmap_small = loader.to_qmap_2d("Qx", "Qz", bins=(60, 50))
    assert qmap_small.approximation == "small_angle"
    loader.approximation = "correct"

    # 2. (Qr, Qz) and (signed_Qr, Qz) maps
    qmap_qr_qz = loader.to_qmap_2d("Qr", "Qz", bins=(50, 40))
    assert qmap_qr_qz.x_domain == "Qr"
    qmap_sqr_qz = loader.to_qmap_2d("signed_Qr", "Qz", bins=(50, 40))
    assert qmap_sqr_qz.x_domain == "signed_Qr"

    # 3. (Qx, Qy) map sliced by fixed Qz vs fixed omega
    qmap_qx_qy_qz = loader.to_qmap_2d("Qx", "Qy", bins=(50, 40), qz_cut=(-0.05, 0.05))
    assert "Qz" in qmap_qx_qy_qz.cuts
    qmap_qx_qy_om = loader.to_qmap_2d("Qx", "Qy", bins=(50, 40), omega_cut=(-2.0, 2.0))
    assert "omega" in qmap_qx_qy_om.cuts

    # 4. (Qy, Qz) map sliced by fixed Qx vs fixed Qr
    q0 = 2.0 * np.pi / 100.0
    qmap_qy_qz_qx = loader.to_qmap_2d(
        "Qy", "Qz", bins=(40, 40), qx_cut=(q0 - 0.02, q0 + 0.02)
    )
    assert "Qx" in qmap_qy_qz_qx.cuts
    qmap_qy_qz_qr = loader.to_qmap_2d(
        "Qy", "Qz", bins=(40, 40), qr_cut=(q0 - 0.02, q0 + 0.02)
    )
    assert "Qr" in qmap_qy_qz_qr.cuts

    # 5. 2D map -> slice_1d
    slice_x = qmap_qx_qz.slice_1d(domain="Qx")
    assert slice_x.domain == "Qx"
    slice_y = qmap_qx_qz.slice_1d(domain="Qz", cut_range=(q0 - 0.02, q0 + 0.02))
    assert slice_y.domain == "Qz"

    # 6. Specular 1D reductions: per-order rods I(Qz) & per-omega ridges I(Qx)
    rod_1 = loader.extract_bragg_rod(order=1, bins=30)
    assert rod_1.domain == "Qz"
    assert np.nanmax(rod_1.intensity) > 0.0

    rods_dict = loader.extract_order_rods_qz(orders=(-2, -1, 1, 2), bins=25)
    assert set(rods_dict.keys()) == {-2, -1, 1, 2}
    direct_rods = extract_order_rods_qz(frames, geom, orders=(1, -1), bins=20)
    assert set(direct_rods.keys()) == {1, -1}

    ridges_dict = loader.extract_qx_by_omega(omegas=[-25.0, 0.0, 25.0], bins=50)
    assert len(ridges_dict) == 3
    direct_ridges = extract_qx_by_omega(frames, geom, omegas=[0.0, 10.0], bins=40)
    assert len(direct_ridges) == 2

    linecut_qx = loader.extract_inplane_linecut(
        domain="Qx", fixed_omega=0.0, omega_tol=2.0, bins=60
    )
    assert linecut_qx.domain == "Qx"
    linecut_qr = extract_inplane_linecut(
        frames, geom, domain="Qr", fixed_qz=0.0, qz_tol=0.05, bins=50
    )
    assert linecut_qr.domain == "Qr"

    tr_2th = reduce_1d(frames, geom, domain="two_theta", bins=40)
    assert tr_2th.domain == "two_theta"

    # 7. Off-specular diffuse cuts (Qy, Qz, Qx), 2D rod map, and PSD read-off
    diff_qy = loader.extract_diffuse_cut(direction="Qy", order=1, bins=24)
    assert diff_qy.domain == "Qy"
    diff_qz = extract_diffuse_cut(
        frames, geom, direction="Qz", order=1, qy_band_inv_nm=(0.015, 0.06), bins=20
    )
    assert diff_qz.domain == "Qz"
    diff_qx = extract_diffuse_cut(
        frames, geom, direction="Qx", qy_band_inv_nm=(0.015, 0.06), bins=30
    )
    assert diff_qx.domain == "Qx"

    offspec_map = loader.extract_offspec_rod_map(order=1, n_qy=20)
    assert offspec_map.x_domain == "Qy"
    assert offspec_map.y_domain == "Qz"
    offspec_direct = extract_offspec_rod_map(frames, geom, order=1, n_qy=16)
    assert offspec_direct.H.shape[0] == 16

    diff_df = loader.extract_diffuse_qy(order=1, frame_index=2)
    assert "streak_adu" in diff_df.columns
    yy, xx = np.meshgrid(
        np.arange(128, dtype=float), np.arange(128, dtype=float), indexing="ij"
    )
    pq = compute_pixel_q(geom, xx, yy, 285.0, 0.0)
    psd_res = readoff_edge_psd_born(
        frames[2].image,
        pq.Qx,
        pq.Qy,
        order=1,
        pitch_nm=100.0,
        line_width_nm=45.0,
        qy_edges=np.linspace(0.005, 0.08, 12),
    )
    assert "psd_nm3" in psd_res
    rod_diff = extract_rod_localized_diffuse_qy(
        frames[2].image,
        geom,
        energy_ev=285.0,
        omega_deg=0.0,
        order=1,
        qy_edges=np.linspace(0.005, 0.08, 10),
    )
    assert len(rod_diff) == 9

    # 8. Plotting helpers (detector strips and specular 1D panels)
    strips_map = {
        0.0: extract_line_strip(geom, frames[2].image, pq.Qx, pq.Qy, half_width_px=12),
        10.0: extract_line_strip(geom, frames[3].image, pq.Qx, pq.Qy, half_width_px=12),
        25.0: extract_line_strip(geom, frames[4].image, pq.Qx, pq.Qy, half_width_px=12),
    }
    fig_s = plot_detector_strips(
        strips_map,
        omega_highlights=((0.0, "white"), (10.0, "#ff3fd0"), (25.0, "#8cff00")),
    )
    assert fig_s is not None

    fig_1d, axes_1d = plot_specular_1d_panels(ridges_dict, rods_dict)
    assert fig_1d is not None
    assert len(axes_1d) == 2
    plt.close("all")

    # Direct map_2d call with fixed_domain
    qmap_fixed = map_2d(
        frames,
        geom,
        x_domain="Qx",
        y_domain="Qz",
        bins=(30, 30),
        fixed_domain="Qy",
        fixed_value=0.0,
        fixed_tol=0.02,
    )
    assert "Qy" in qmap_fixed.cuts
