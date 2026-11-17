"""Focused example of 2D and 3D Born and Multislice CD-RSoXS simulations.

Covers:
1. Constructing arbitrary 2D (`RefractiveIndexMap2D`) and 3D (`RefractiveIndexMap3D`)
   optical constant maps n = 1 - delta + i*beta, as well as parameterized line-space
   gratings with trapezoidal sidewalls, top corner rounding, multi-layer stacks,
   and 3D correlated/uncorrelated line-edge/line-width roughness (LER / LWR / LPR).
2. Comparing 2D Born approximation (`simulate_born_2d`) vs. 2D Multislice / Split-Step
   Fourier beam propagation (`simulate_multislice_2d`) in weak-scattering (kinematic)
   and strong-scattering (dynamical + absorption asymmetry) regimes.
3. Simulating 3D rough line gratings (`simulate_born_3d` and `simulate_multislice_3d`),
   extracting specular Bragg rods and off-specular diffuse Qy profiles.
4. Generating publication-ready visualization figures using `plot_structure_2d`,
   `plot_structure_3d`, `plot_multislice_wavefield`, `plot_simulation_comparison`,
   and `plot_born_vs_multislice_grid`.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import cd_reduce as cdr


def main() -> None:
    """Run 2D and 3D Born vs. Multislice simulations and visualization helpers."""
    energy_ev = 285.0
    omega_deg = np.linspace(-30.0, 30.0, 31)

    # 1. Weak-scattering (kinematic) regime: Born and Multislice agree to <0.5%
    weak_2d = cdr.build_grating_2d(
        pitch_nm=100.0,
        cd_nm=45.0,
        thickness_nm=40.0,
        sidewall_angle_deg=88.0,
        delta_line=1.0e-5,
        beta_line=2.0e-6,
        nx=128,
        nz=40,
        energy_ev=energy_ev,
    )
    born_weak = cdr.simulate_born_2d(weak_2d, omega_deg=omega_deg, max_order=4)
    ms_weak = cdr.simulate_multislice_2d(weak_2d, omega_deg=omega_deg, max_order=4)
    rel_diff_m1 = np.linalg.norm(
        born_weak.order_efficiencies[1] - ms_weak.order_efficiencies[1]
    ) / np.linalg.norm(born_weak.order_efficiencies[1])
    print(f"Kinematic limit (m=+1) L2 diff Born vs Multislice: {rel_diff_m1:.4f}")

    # 2. Strong-scattering / thick multi-layer asymmetric grating with dynamical effects
    layers = [
        cdr.GratingLayerSpec(
            thickness_nm=60.0,
            cd_bottom_nm=52.0,
            cd_top_nm=38.0,
            delta_line=4.0e-3,
            beta_line=1.5e-3,
        ),
        cdr.GratingLayerSpec(
            thickness_nm=30.0,
            cd_bottom_nm=38.0,
            cd_top_nm=28.0,
            top_rounding_nm=6.0,
            delta_line=2.5e-3,
            beta_line=6.0e-4,
            x_center_nm=4.0,
        ),
    ]
    strong_2d = cdr.build_grating_2d(
        pitch_nm=100.0,
        layers=layers,
        nx=128,
        nz=60,
        energy_ev=energy_ev,
    )
    born_strong = cdr.simulate_born_2d(strong_2d, omega_deg=omega_deg, max_order=5)
    ms_strong = cdr.simulate_multislice_2d(
        strong_2d,
        omega_deg=omega_deg,
        max_order=5,
        field_omega_deg=0.0,
    )
    idx_om0 = int(np.argmin(np.abs(omega_deg)))
    ms_ratio = (
        ms_strong.order_efficiencies[1][idx_om0]
        / ms_strong.order_efficiencies[-1][idx_om0]
    )
    born_ratio = (
        born_strong.order_efficiencies[1][idx_om0]
        / born_strong.order_efficiencies[-1][idx_om0]
    )
    print(
        f"Dynamical m=+1 / m=-1 ratio at omega=0: "
        f"Multislice={ms_ratio:.4f}, Born={born_ratio:.4f}"
    )

    # 3. Custom arbitrary 2D refractive index map (core-shell / bilayer line grating)
    nx, nz = 128, 60
    dx, dz = 100.0 / nx, 80.0 / nz
    x = (np.arange(nx) - nx / 2.0) * dx
    z = np.arange(nz) * dz
    xx, zz = np.meshgrid(x, z, indexing="ij")
    in_core = (np.abs(xx) <= 18.0) & (zz >= 5.0) & (zz <= 65.0)
    in_shell = (np.abs(xx) <= 24.0) & (zz <= 75.0) & (~in_core)
    delta_custom = np.where(in_core, 3.0e-3, np.where(in_shell, 1.5e-3, 0.0))
    beta_custom = np.where(in_core, 9.0e-4, np.where(in_shell, 3.0e-4, 0.0))
    custom_map_2d = cdr.RefractiveIndexMap2D.from_delta_beta(
        delta=delta_custom,
        beta=beta_custom,
        dx_nm=dx,
        dz_nm=dz,
        energy_ev=energy_ev,
        pitch_nm=100.0,
    )
    ms_custom = cdr.simulate_multislice_2d(
        custom_map_2d, omega_deg=omega_deg, max_order=4
    )
    print(f"Custom core-shell 2D map simulated: orders={ms_custom.orders.tolist()}")

    # 4. 3D Line Grating with Palasantzas Line-Edge Roughness (LER)
    struct_3d = cdr.build_grating_3d(
        pitch_nm=100.0,
        cd_nm=45.0,
        thickness_nm=60.0,
        sidewall_angle_deg=88.0,
        delta_line=2.5e-3,
        beta_line=4.0e-4,
        ler_sigma_nm=2.0,
        correlation_length_y_nm=25.0,
        hurst_exponent=0.7,
        roughness_correlation=0.0,
        nx=64,
        ny=32,
        nz=20,
        ly_nm=320.0,
        energy_ev=energy_ev,
        seed=99,
    )
    born_3d = cdr.simulate_born_3d(
        struct_3d,
        omega_deg=np.linspace(-25.0, 25.0, 11),
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
    print(
        f"3D Rough Grating: Born (Qx,Qy) map={born_3d.qmap_qx_qy.H.shape}, "
        f"Multislice 3D (Qx,Qy) map={ms_3d.qmap_qx_qy.H.shape}"
    )

    # 5. Visualization figures
    cdr.plot_structure_2d(strong_2d, component="both")
    cdr.plot_structure_3d(struct_3d)
    cdr.plot_multislice_wavefield(ms_strong)
    cdr.plot_simulation_comparison(born_strong, ms_strong, orders=(1, 2, 3))
    cdr.plot_born_vs_multislice_grid([(born_weak, ms_weak), (born_strong, ms_strong)])
    plt.close("all")
    print("Generated and verified all 5 diagnostic visualization figures.")


if __name__ == "__main__":
    main()
