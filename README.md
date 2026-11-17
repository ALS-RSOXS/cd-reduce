# cd-reduce

Standalone CD-RSoXS / CD-SAXS data reduction, reciprocal-space mapping, and 2D/3D Born & Multislice scattering simulation package following the architecture and ergonomics of [`pxr-reduce`](https://github.com/ALS-RSOXS/pxr-reduce).

## Core Architecture

`cd-reduce` is organized into two self-contained modules with zero hard-coded paths or external script dependencies:

### 1. Data Reduction & Reciprocal-Space Mapping (`cd_reduce.reduction` & `CdRsoxsLoader`)

- **Unified Pixel-to-$Q$ Geometry (`cd_reduce.reduction.geometry`)**:
  - Unifies transmission tilt-series geometry conventions into a single [`Geometry`](src/cd_reduce/reduction/geometry.py) dataclass and [`compute_pixel_q`](src/cd_reduce/reduction/geometry.py) engine.
  - Supports exact Ewald-sphere (`approximation="correct"`), small-angle (`approximation="small_angle"`), and linear-angle (`approximation="angle_linear"`) reductions.
  - Evaluates all reciprocal-space and detector domains: `Qx`, `Qy`, `Qz`, `Qr`, `signed_Qr`, `Q`, `omega`, `two_theta`, `px_x`, `px_y`.
- **Spot Extraction & Geometry Calibration (`cd_reduce.reduction.spots`, `cd_reduce.reduction.calibration`)**:
  - Model-free spot detection, spot-line angle regression, sub-pixel centroiding, aperture integration, and reference-template amplitude matching.
  - Laue-invariant geometry calibration (`fit_geometry`), joint beam walk-off fitting (`fit_geometry_with_walkoff`), multi-dataset joint sample-to-detector distance calibration (`fit_joint_geometry`), and Bragg rod verticality refinement (`refine_rod_verticality`).
- **Background Subtraction & Corrections (`cd_reduce.reduction.corrections`)**:
  - Background subtraction modes: `"corner"`, `"row_col"`, `"offline_band"`, `"strip_fit"` (`process_frame_strip` with Poisson deviance ML + analytic Jacobian + BIC bleed selection + origin shift), and `"photon_events"`.
  - Geometric & transmission corrections: exposure normalization, solid-angle $\cos^{-3}(2\theta)$, polarization, $\mathrm{Si_3N_4}$ window attenuation $T(\omega, E)$, and beam walk-off $f(\omega)$.
- **2D Mapping & 1D Specular / Off-Specular Extraction (`cd_reduce.reduction.mapping`)**:
  - [`map_2d`](src/cd_reduce/reduction/mapping.py) and [`CdRsoxsLoader.to_qmap_2d`](src/cd_reduce/loader.py): bin CCD frame series into 2D [`QMap2D`](src/cd_reduce/reduction/mapping.py) maps across any domain pair (`(Qx, Qz)`, `(Qr, Qz)`, `(signed_Qr, Qz)`, `(Qx, Qy)`, `(Qy, Qz)`, `(Qx, omega)`) with flexible orthogonal cuts (`qz_cut`, `qx_cut`, `qy_cut`, `qr_cut`, `omega_cut`, `fixed_domain`).
  - Specular 1D extraction: [`extract_qx_by_omega`](src/cd_reduce/reduction/mapping.py) ($I$ vs $Q_x$ per sample tilt $\omega$) and [`extract_order_rods_qz`](src/cd_reduce/reduction/mapping.py) ($I_m$ vs $Q_z$ per diffraction order $m$).
  - Off-specular & diffuse extraction: [`extract_diffuse_cut`](src/cd_reduce/reduction/mapping.py) (directional diffuse cuts along $Q_y$, $Q_z$, or $Q_x$), [`extract_offspec_rod_map`](src/cd_reduce/reduction/mapping.py), and [`readoff_edge_psd_born`](src/cd_reduce/reduction/mapping.py) (Born LER power spectral density).

- **Roughness PSDs (`cd_reduce.reduction.roughness`)**: [`extract_diffuse_ratio`](src/cd_reduce/reduction/roughness.py) reads the order-normalised diffuse $R_m(Q_y)$, either as a detector line cut (`mode="line_cut"`) or along a constant-$Q_z$ trace through the tilt series (`mode="qz_trace"`), with Ewald-$Q_z$ and core-background corrections. [`decompose_edge_psd`](src/cd_reduce/reduction/roughness.py) splits it by a multi-order Born fit into **LPR, LWR and LER** PSDs plus the edge correlation $\rho$. Mack, power-law and spectral-peak fits are included. See [the guide](docs/guides/mapping/roughness_psd.md).
- **Plots (`cd_reduce.reduction.plotting`)**: `plot_traces_by_omega` (I vs Qx/Qr per ω, colour bar and labels), `plot_rods_by_order`, `plot_qmap_grid` (e.g. exact vs small-angle butterflies), `plot_diffuse_ratio`, `plot_edge_psds`. Run them all with `uv run python examples/roughness_and_figures.py`.

### 2. 2D & 3D Scattering Simulation (`cd_reduce.simulation`)

- **General Optical Constant Maps (`cd_reduce.simulation.structures`)**:
  - [`RefractiveIndexMap2D`](src/cd_reduce/simulation/structures.py) ($n(x, z) = 1 - \delta(x, z) + i\beta(x, z)$) and [`RefractiveIndexMap3D`](src/cd_reduce/simulation/structures.py) ($n(x, y, z)$), constructed from arbitrary arrays or parameterized grating builders ([`build_grating_2d`](src/cd_reduce/simulation/structures.py), [`build_grating_3d`](src/cd_reduce/simulation/structures.py) with 3D Palasantzas LER / LWR / LPR edge roughness).
- **Born Approximation (`cd_reduce.simulation.born`)**:
  - [`simulate_born_2d`](src/cd_reduce/simulation/born.py) and [`simulate_born_3d`](src/cd_reduce/simulation/born.py) compute continuous 2D reciprocal-space maps (`(Qx, Qz)`, `(Qx, Qy)`, `(Qy, Qz)`), Bragg rod efficiencies $\eta_m(\omega, Q_z)$, off-specular diffuse $Q_y$ traces, and synthetic 2D detector frames (`to_detector_frame`).
- **Multislice / Beam Propagation (`cd_reduce.simulation.multislice`)**:
  - [`simulate_multislice_2d`](src/cd_reduce/simulation/multislice.py) and [`simulate_multislice_3d`](src/cd_reduce/simulation/multislice.py) propagate wavefields through arbitrary 2D/3D refractive index maps via Strang-split angular-spectrum propagation across tilt angles $\omega$, capturing soft X-ray attenuation, dynamical phase refraction, resonant Friedel asymmetry, and internal wavefields $\psi(x, z)$.
- **Visualization (`cd_reduce.simulation.visualization`)**:
  - [`plot_born_vs_multislice_grid`](src/cd_reduce/simulation/visualization.py), [`plot_structure_2d`](src/cd_reduce/simulation/visualization.py), [`plot_structure_3d`](src/cd_reduce/simulation/visualization.py), [`plot_multislice_wavefield`](src/cd_reduce/simulation/visualization.py), and [`plot_simulation_comparison`](src/cd_reduce/simulation/visualization.py).

---

## Examples

Runnable scripts covering the complete workflow live in [`examples/`](examples/):

- [`examples/full_process_walkthrough.py`](examples/full_process_walkthrough.py) — End-to-end pipeline from 2D/3D structure construction and Born/Multislice simulation to synthetic FITS generation, spot extraction, Laue geometry calibration, background subtraction, 2D $Q$-mapping, and 1D specular/diffuse reduction.
- [`examples/reduction_and_qmapping.py`](examples/reduction_and_qmapping.py) — Calibration modes (single-mount, walk-off, joint beamtime $L_b$, rod verticality), template matching, coordinate projections, and slicing filters.
- [`examples/simulation_2d_3d.py`](examples/simulation_2d_3d.py) — 2D/3D optical constant maps, kinematic vs. dynamical Multislice wave propagation, 3D rough gratings, and diagnostic plotting.

```bash
uv run python examples/full_process_walkthrough.py
uv run python examples/reduction_and_qmapping.py
uv run python examples/simulation_2d_3d.py
```

---

## Quick Start

### 1. Calibrate, Reduce & Extract 1D Specular / Diffuse Profiles

```python
import numpy as np
import cd_reduce as cdr

loader = cdr.CdRsoxsLoader("path/to/fits_dir", pitch_nm=100.0)

# Calibrate detector & sample tilt geometry and refine rod verticality
calib = loader.calibrate_geometry(Lb_px=1969.0, pitch_nm=100.0, fit_walkoff=True)
omega_ref = loader.refine_omega()

# Project tilt series into 2D reciprocal-space maps
qmap_qx_qz = loader.to_qmap_2d("Qx", "Qz", bins=(300, 250), approximation="correct")
qmap_qx_qy = loader.to_qmap_2d("Qx", "Qy", qz_cut=(-0.015, 0.015))
qmap_qy_qz = loader.to_qmap_2d("Qy", "Qz", qx_cut=(0.055, 0.071))

# Extract main specular 1D views: I vs Qx per omega, and I vs Qz per order
ridges_qx = loader.extract_qx_by_omega()
rods_qz = loader.extract_order_rods_qz(orders=(-4, -3, -2, -1, 1, 2, 3, 4))

# Extract directional off-specular diffuse scattering (Qy, Qz, or Qx)
diffuse_qy = loader.extract_diffuse_cut(direction="Qy", order=1)
diffuse_qz = loader.extract_diffuse_cut(direction="Qz", order=1)
```

### 2. Simulate 2D & 3D Structures (Born vs. Multislice)

```python
import numpy as np
import cd_reduce as cdr

omega_deg = np.linspace(-30.0, 30.0, 61)
struct_2d = cdr.build_grating_2d(
    pitch_nm=80.0,
    cd_nm=40.0,
    thickness_nm=40.0,
    sidewall_angle_deg=87.5,
    delta_line=2.2e-3,
    beta_line=2.8e-3,
    energy_ev=285.2,
)

born_2d = cdr.simulate_born_2d(struct_2d, omega_deg=omega_deg, max_order=4)
ms_2d = cdr.simulate_multislice_2d(struct_2d, omega_deg=omega_deg, max_order=4)
cdr.plot_born_vs_multislice_grid([(born_2d, ms_2d)])
```

### 3. Command-Line Interface

```bash
uv run cd-reduce calibrate path/to/fits_dir --pitch-nm 100.0 --fit-walkoff -o geometry.json
uv run cd-reduce reduce path/to/fits_dir -g geometry.json --x-axis Qx --y-axis Qz -o qmap.npz
uv run cd-reduce simulate --method multislice_2d --pitch-nm 80.0 --cd-nm 40.0 --energy-ev 285.2 -o sim.npz
```
