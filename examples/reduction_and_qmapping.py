"""Focused example of CD-RSoXS / CD-SAXS data reduction and reciprocal-space mapping.

Covers:
1. Single-mount Laue calibration (`fit_geometry`), joint walk-off fitting
   (`fit_geometry_with_walkoff`), multi-dataset joint beamtime calibration
   (`fit_joint_geometry`), and Bragg rod verticality refinement
   (`refine_rod_verticality`).
2. High-SNR reference-template extraction for low-dose mounts
   (`extract_template_amplitudes`).
3. Pixel-to-Q mapping modes (`"correct"`, `"small_angle"`, `"angle_linear"`).
4. All 2D QMap2D projections (`Qx`, `Qy`, `Qz`, `Qr`, `signed_Qr`, `Q`, `omega`,
   `two_theta`) and slice filters (`qz_cut`, `omega_cut`, `qx_cut`, `qr_cut`).
5. Specular and off-specular 1D extractions (`extract_order_rods_qz`,
   `extract_qx_by_omega`, `extract_diffuse_cut`, and `reduce_1d`).
"""

from __future__ import annotations

import numpy as np

import cd_reduce as cdr


def build_synthetic_frames(
    pitch_nm: float = 100.0,
    lb_px: float = 1500.0,
    line_rot_deg: float = 0.0,
    flux_scale: float = 2e7,
    seed: int = 11,
) -> tuple[list[cdr.Frame], cdr.Geometry]:
    """Build an in-memory list of Frame objects with known geometry."""
    struct = cdr.build_grating_2d(
        pitch_nm=pitch_nm,
        cd_nm=48.0,
        thickness_nm=65.0,
        nx=128,
        nz=40,
        energy_ev=285.0,
    )
    omega_deg = np.linspace(-25.0, 25.0, 11)
    born = cdr.simulate_born_2d(struct, omega_deg=omega_deg, max_order=4)
    geom = cdr.Geometry(
        u0_px=128.0,
        v0_px=128.0,
        Lb_px=lb_px,
        line_rot_deg=line_rot_deg,
        beta_deg=0.0,
        psi_deg=0.0,
        Lambda_nm=pitch_nm,
    )
    rng = np.random.default_rng(seed)
    frames: list[cdr.Frame] = []
    for idx, om in enumerate(omega_deg):
        base = born.to_detector_frame(
            geom,
            omega_deg=float(om),
            shape=(256, 256),
            spot_sigma_q=0.006,
            flux_scale=flux_scale,
            background_adu=15.0,
        )
        noisy = base.image + rng.normal(0.0, 1.5, size=base.image.shape)
        frames.append(
            cdr.Frame.from_array(
                noisy,
                omega_deg=float(om),
                energy_ev=285.0,
                exposure_s=1.0,
                path=f"frame_{idx:04d}.fits",
            )
        )
    return frames, geom


def main() -> None:
    """Run all calibration, background, 2D mapping, and 1D reduction modes."""
    pitch_nm = 100.0
    frames_dev, true_geom = build_synthetic_frames(pitch_nm=pitch_nm, seed=11)
    frames_latent, _ = build_synthetic_frames(
        pitch_nm=pitch_nm, flux_scale=3e6, seed=22
    )

    # 1. Spot extraction & single-mount + walk-off + joint beamtime calibration
    spots_dev = cdr.extract_frames_spots(frames_dev)
    spots_latent = cdr.extract_frames_spots(frames_latent)

    calib_walkoff = cdr.fit_geometry_with_walkoff(
        spots_dev,
        true_geom,
        fix_lambda=pitch_nm,
    )
    print(
        f"Single-mount + walk-off calibration: RMS={calib_walkoff.rms_px:.3f} px "
        f"across {calib_walkoff.n_bragg} Bragg spots"
    )

    joint_calibs = cdr.fit_joint_geometry(
        {"developed": spots_dev, "latent": spots_latent},
        {"developed": true_geom, "latent": true_geom},
    )
    print(
        f"Joint beamtime Lb_px fit across 2 mounts: "
        f"shared Lb_px={joint_calibs.shared_Lb_px:.2f} "
        f"+/- {joint_calibs.shared_Lb_err_px:.2f} px"
    )

    # 2. High-SNR reference-template extraction for low-contrast latent mounts
    template_df = cdr.extract_template_amplitudes(
        target_frames=frames_latent,
        reference_frames=frames_dev,
        target_spots=spots_latent,
        reference_spots=spots_dev,
        target_geometry=true_geom,
        reference_geometry=true_geom,
        half_box_px=20,
        windows=((8, 10), (10, 12), (12, 14)),
    )
    print(f"Template-matched low-dose order rows extracted: {len(template_df)}")

    # 3. Compare exact Ewald ("correct"), "small_angle", and "angle_linear" Q-mapping
    geom = calib_walkoff.geometry
    for approx_mode in ("correct", "small_angle", "angle_linear"):
        qmap = cdr.map_2d(
            frames_dev,
            geom,
            x_domain="Qx",
            y_domain="Qz",
            bins=(120, 80),
            approximation=approx_mode,  # type: ignore[arg-type]
            bg_mode="corner",
        )
        print(
            f"QMap2D (Qx, Qz) approximation={approx_mode:<13s}: "
            f"non-empty bins={int(np.sum(qmap.N > 0))}"
        )

    # 4. Slicing combinations: fixed Qz, fixed omega, fixed Qx, fixed Qr
    g1 = 2.0 * np.pi / pitch_nm
    slice_configs: list[tuple[str, str, dict[str, tuple[float, float]]]] = [
        ("Qx", "Qy", {"qz_cut": (-0.03, 0.03)}),
        ("Qx", "Qy", {"omega_cut": (-2.0, 2.0)}),
        ("Qy", "Qz", {"qx_cut": (g1 - 0.015, g1 + 0.015)}),
        ("Qy", "Qz", {"qr_cut": (g1 - 0.015, g1 + 0.015)}),
        ("Qr", "Qz", {}),
        ("signed_Qr", "Qz", {}),
        ("Qx", "omega", {}),
    ]
    for x_ax, y_ax, kwargs in slice_configs:
        qm = cdr.map_2d(
            frames_dev,
            geom,
            x_domain=x_ax,
            y_domain=y_ax,
            bins=(100, 80),
            **kwargs,  # type: ignore[arg-type]
        )
        print(
            f"Slice ({x_ax}, {y_ax}) with {kwargs or 'full stack'}: "
            f"non-empty bins={int(np.sum(qm.N > 0))}"
        )

    # 5. Specular and off-specular 1D extractions
    rods = cdr.extract_order_rods_qz(frames_dev, geom, orders=(-1, 1), bins=40)
    ridges = cdr.extract_qx_by_omega(
        frames_dev, geom, omegas=[-20.0, 0.0, 20.0], bins=100
    )
    diff_qy = cdr.extract_diffuse_cut(frames_dev, geom, direction="Qy", order=1)
    print(
        f"1D extractions: orders={sorted(rods.keys())}, "
        f"omega ridges={len(ridges)}, diffuse Qy bins={diff_qy.coords.size}"
    )


if __name__ == "__main__":
    main()
