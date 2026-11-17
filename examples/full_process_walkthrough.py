"""End-to-end demonstration of the full cd-reduce pipeline.

This script demonstrates the complete process in a single self-contained run:
1. Constructing 2D and 3D complex refractive index maps n = 1 - delta + i*beta.
2. Simulating scattering via 2D/3D Born approximation and 2D/3D Multislice propagation.
3. Projecting simulated scattering onto synthetic 2D CCD detector FITS frames across a
   tilt series omega and writing them to disk.
4. Model-free diffraction spot extraction and Laue-invariant geometry calibration
   (including walk-off fitting, multi-mount joint Lb_px calibration, order indexing,
   and Bragg rod verticality refinement).
5. Applying background subtraction ("corner", "row_col", "offline_band", "strip_fit",
   "photon_events") and physical/geometric corrections (exposure, solid angle,
   polarization, Si3N4 membrane transmission, walk-off).
6. Converting the CCD tilt series into 2D reciprocal-space maps across all domain
   combinations (Qx, Qz, Qr, signed_Qr, Qy, omega, two_theta) and slicing modes
   (fixed Qz, fixed omega, fixed Qx, fixed Qr; exact Ewald vs. small-angle).
7. Reducing to 1D specular & off-specular traces: I(Qx) per omega, I_m(Qz) per order,
   directional diffuse cuts (Qy, Qz, Qx), and Born LER PSD read-off.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

import cd_reduce as cdr


def write_synthetic_fits_series(
    out_dir: Path,
    frames: list[cdr.Frame],
) -> Path:
    """Write a synthetic tilt-series list of Frame objects to FITS files on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames, start=1):
        hdr = fits.Header()
        # sample_theta_deg + 90.0 = omega_deg in ALS BL 11.0.1.2 FITS convention
        hdr["SAM_TH"] = float(frame.omega_deg - 90.0)
        hdr["ENERGY"] = float(frame.energy_ev)
        hdr["EXPOSURE"] = float(frame.exposure_s)
        hdr["SHUTTER"] = 1
        fits.PrimaryHDU(
            data=frame.image.astype(np.float32),
            header=hdr,
        ).writeto(out_dir / f"scan_{idx:05d}.fits", overwrite=True)
    return out_dir


def main() -> None:
    """Run the full simulation + data reduction + mapping workflow."""
    print("=== Step 1: Build 2D & 3D Grating Optical Constant Maps ===")
    pitch_nm = 100.0
    energy_ev = 285.0  # Carbon K-edge RSoXS energy

    struct_2d = cdr.build_grating_2d(
        pitch_nm=pitch_nm,
        cd_nm=45.0,
        thickness_nm=70.0,
        sidewall_angle_deg=87.0,
        top_rounding_nm=3.0,
        delta_line=2.5e-3,
        beta_line=4.0e-4,
        nx=128,
        nz=50,
        energy_ev=energy_ev,
    )
    print(
        f"2D Structure: shape={struct_2d.delta.shape}, "
        f"pitch={struct_2d.effective_pitch_nm} nm, thickness={struct_2d.lz_nm} nm"
    )

    struct_3d = cdr.build_grating_3d(
        pitch_nm=pitch_nm,
        cd_nm=45.0,
        thickness_nm=70.0,
        sidewall_angle_deg=87.0,
        delta_line=2.5e-3,
        beta_line=4.0e-4,
        nx=64,
        ny=32,
        nz=24,
        ly_nm=320.0,
        energy_ev=energy_ev,
        seed=7,
    )
    print(
        f"3D Structure: shape={struct_3d.delta.shape}, "
        f"Lx={struct_3d.lx_nm} nm, Ly={struct_3d.ly_nm} nm, Lz={struct_3d.lz_nm} nm"
    )

    print("\n=== Step 2: 2D & 3D Born and Multislice Simulations ===")
    omega_sim = np.linspace(-30.0, 30.0, 13)
    born_2d = cdr.simulate_born_2d(struct_2d, omega_deg=omega_sim, max_order=5)
    ms_2d = cdr.simulate_multislice_2d(struct_2d, omega_deg=omega_sim, max_order=5)
    rod_born_m1 = born_2d.bragg_rod(order=1, x_axis="Qz")
    rod_ms_m1 = ms_2d.bragg_rod(order=1, x_axis="Qz")
    print(
        f"m=+1 peak efficiency: Born={rod_born_m1.intensity.max():.3e}, "
        f"Multislice={rod_ms_m1.intensity.max():.3e}"
    )

    born_3d = cdr.simulate_born_3d(
        struct_3d,
        omega_deg=np.linspace(-20.0, 20.0, 9),
        max_order=3,
        nqx=61,
        nqy=31,
        nqz=41,
    )
    ms_3d = cdr.simulate_multislice_3d(
        struct_3d,
        omega_deg=np.linspace(-20.0, 20.0, 5),
        max_order=3,
        nqx=61,
        nqz=41,
    )
    diffuse_cut_m1 = born_3d.diffuse_qy_trace(order=1)
    print(
        f"3D simulation complete: Born Qy-Qz map shape={born_3d.qmap_qy_qz.H.shape}, "
        f"Multislice 3D orders={ms_3d.orders.tolist()}, "
        f"diffuse cut points={diffuse_cut_m1.coords.size}"
    )

    print("\n=== Step 3: Synthetic Detector Tilt-Series Generation ===")
    true_geom = cdr.Geometry(
        u0_px=128.0,
        v0_px=128.0,
        Lb_px=1500.0,
        line_rot_deg=0.0,
        beta_deg=0.0,
        psi_deg=0.0,
        Lambda_nm=pitch_nm,
    )
    rng = np.random.default_rng(42)
    frames_sim: list[cdr.Frame] = []
    for idx, om in enumerate(omega_sim):
        base_f = born_2d.to_detector_frame(
            true_geom,
            omega_deg=float(om),
            shape=(256, 256),
            spot_sigma_q=0.006,
            flux_scale=2e7,
            background_adu=20.0,
        )
        noisy_img = base_f.image + rng.normal(0.0, 1.5, size=base_f.image.shape)
        frames_sim.append(
            cdr.Frame.from_array(
                noisy_img,
                omega_deg=float(om),
                energy_ev=energy_ev,
                exposure_s=1.0,
                path=f"scan_{idx + 1:05d}.fits",
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fits_dir = write_synthetic_fits_series(tmp_path / "raw_fits", frames_sim)

        print("\n=== Step 4: Spot Extraction & Geometry Calibration ===")
        loader = cdr.CdRsoxsLoader(fits_dir, pitch_nm=pitch_nm)
        spots_df = loader.extract_spots()
        calib = loader.calibrate_geometry(
            start=true_geom,
            fix_lambda=pitch_nm,
            fix_Lb=1500.0,
        )
        print(
            f"Calibrated geometry: RMS residual={calib.rms_px:.3f} px "
            f"across {calib.n_bragg} Bragg spots (u0={calib.geometry.u0_px:.2f}, "
            f"v0={calib.geometry.v0_px:.2f})"
        )

        # Refine Bragg rod verticality across sample tilts
        omega_ref = loader.refine_omega()
        geom_path = tmp_path / "geometry_calibrated.json"
        geom_path.write_text(
            json.dumps(loader.geometry.to_dict(), indent=2), encoding="utf-8"
        )
        print(
            f"Rod verticality refinement: d_omega0={omega_ref.d_omega0_deg:+.4f} deg, "
            f"c0={omega_ref.c0_px:+.3f} px, t={omega_ref.t_px:+.3f} px "
            f"(saved to {geom_path.name})"
        )

        # Build indexed order table
        order_table = loader.build_order_table()
        print(f"Indexed order table rows: {len(order_table)}")

        print("\n=== Step 5: Background Subtraction & Physical Corrections ===")
        mid_frame = loader.frames[len(loader.frames) // 2]
        ny, nx = mid_frame.image.shape
        yy, xx = np.meshgrid(
            np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
        )
        pq = cdr.compute_pixel_q(
            loader.geometry, xx, yy, mid_frame.energy_ev, mid_frame.omega_deg
        )
        mask = np.ones((ny, nx), dtype=bool)

        img_corner, off_adu, noise_adu = cdr.subtract_corner_offset(mid_frame.image)
        img_rc = cdr.subtract_row_col_offset(img_corner, loader.geometry, pq.Qy, mask)
        img_band, _ = cdr.subtract_offline_band_background(
            img_corner,
            pq.Qx,
            pq.Qy,
            loader.geometry.g_inv_nm,
            np.linspace(-0.25, 0.25, 51),
        )
        _img_phot, n_events = cdr.threshold_photon_events(
            img_corner, energy_ev=mid_frame.energy_ev
        )
        img_corr = cdr.apply_geometric_corrections(
            img_band,
            pq,
            exposure_s=mid_frame.exposure_s,
            solid_angle=True,
            polarization=True,
            walkoff_f=float(cdr.walkoff_factor(mid_frame.omega_deg, a=2.5, b=0.1)),
            membrane_t=float(
                cdr.membrane_transmission(
                    mid_frame.omega_deg, mid_frame.energy_ev, thickness_nm=100.0
                )
            ),
        )
        print(
            f"  Corner offset={off_adu:.2f} ADU (sigma={noise_adu:.2f} ADU), "
            f"row_col sum={img_rc.sum():.1f}, offline_band sum={img_band.sum():.1f}, "
            f"photon events={n_events}, corrected sum={img_corr.sum():.1e}"
        )
        assert spots_df is not None

        print("\n=== Step 6: 2D Reciprocal-Space Mapping Across Domains & Slices ===")
        qmap_qx_qz_exact = loader.to_qmap_2d(
            "Qx", "Qz", bins=(120, 80), approximation="correct", bg_mode="corner"
        )
        qmap_qx_qz_sa = loader.to_qmap_2d(
            "Qx", "Qz", bins=(120, 80), approximation="small_angle", bg_mode="corner"
        )
        qmap_qr_qz = loader.to_qmap_2d("Qr", "Qz", bins=(100, 80))
        qmap_sqr_qz = loader.to_qmap_2d("signed_Qr", "Qz", bins=(120, 80))
        qmap_qx_qy_qz0 = loader.to_qmap_2d(
            "Qx", "Qy", bins=(100, 80), qz_cut=(-0.03, 0.03)
        )
        qmap_qx_qy_om0 = loader.to_qmap_2d(
            "Qx", "Qy", bins=(100, 80), omega_cut=(-2.0, 2.0)
        )
        g1_nm_inv = 2.0 * np.pi / pitch_nm
        qmap_qy_qz_qx1 = loader.to_qmap_2d(
            "Qy", "Qz", bins=(80, 80), qx_cut=(g1_nm_inv - 0.015, g1_nm_inv + 0.015)
        )
        qmap_qy_qz_qr1 = loader.to_qmap_2d(
            "Qy", "Qz", bins=(80, 80), qr_cut=(g1_nm_inv - 0.015, g1_nm_inv + 0.015)
        )
        qmap_out = tmp_path / "qmap_qx_qz.npz"
        qmap_qx_qz_exact.save_npz(qmap_out)
        print(
            "Generated 2D Q-maps: "
            f"(Qx,Qz) exact={qmap_qx_qz_exact.H.shape}, "
            f"(Qx,Qz) small-angle={qmap_qx_qz_sa.H.shape}, "
            f"(Qr,Qz)={qmap_qr_qz.H.shape}, "
            f"(signed_Qr,Qz)={qmap_sqr_qz.H.shape}, "
            f"(Qx,Qy|Qz=0)={qmap_qx_qy_qz0.H.shape}, "
            f"(Qx,Qy|omega=0)={qmap_qx_qy_om0.H.shape}, "
            f"(Qy,Qz|Qx=G1)={qmap_qy_qz_qx1.H.shape}, "
            f"(Qy,Qz|Qr=G1)={qmap_qy_qz_qr1.H.shape}"
        )

        print("\n=== Step 7: 1D Specular & Off-Specular Reduction ===")
        rods_by_order = loader.extract_order_rods_qz(orders=(-1, 1), bins=40)
        ridges_by_omega = loader.extract_qx_by_omega(omegas=[-20.0, 0.0, 20.0], bins=80)
        diff_qy = loader.extract_diffuse_cut(direction="Qy", order=1, bins=30)
        psd_df = loader.readoff_edge_psd(
            img_corner,
            pq.Qx,
            pq.Qy,
            order=1,
            line_width_nm=45.0,
            qy_edges=np.linspace(0.005, 0.08, 12),
        )
        print(
            f"1D reductions complete: "
            f"orders extracted={sorted(rods_by_order.keys())}, "
            f"omega ridges={len(ridges_by_omega)}, "
            f"diffuse Qy bins={diff_qy.coords.size}, "
            f"LER PSD rows={len(psd_df)}"
        )


if __name__ == "__main__":
    main()
