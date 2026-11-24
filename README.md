# cd-reduce

Standalone CD-RSoXS / CD-SAXS data reduction, reciprocal-space mapping, and 2D/3D Born & Multislice scattering simulation package following the architecture and ergonomics of [`pxr-reduce`](https://github.com/ALS-RSOXS/pxr-reduce).

## Core Architecture

`cd-reduce` is organized into two self-contained modules with zero hard-coded paths or external script dependencies:

### 1. Data Reduction & Reciprocal-Space Mapping (`cd_reduce.reduction` & `CdRsoxsLoader`)

- **Unified Pixel-to-$Q$ Geometry (`cd_reduce.reduction.geometry`)**:
  - Unifies transmission tilt-series geometry conventions into a single [`Geometry`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/geometry.py) dataclass and [`compute_pixel_q`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/geometry.py) engine.
  - Supports exact Ewald-sphere (`approximation="correct"`), small-angle (`approximation="small_angle"`), and linear-angle (`approximation="angle_linear"`) reductions.
  - Evaluates all reciprocal-space and detector domains: `Qx`, `Qy`, `Qz`, `Qr`, `signed_Qr`, `Q`, `omega`, `two_theta`, `px_x`, `px_y`.
- **Spot Extraction & Geometry Calibration (`cd_reduce.reduction.spots`, `cd_reduce.reduction.calibration`)**:
  - Model-free spot detection, spot-line angle regression, sub-pixel centroiding, aperture integration, and reference-template amplitude matching.
  - Laue-invariant geometry calibration (`fit_geometry`), joint beam walk-off fitting (`fit_geometry_with_walkoff`), multi-dataset joint sample-to-detector distance calibration (`fit_joint_geometry`), and Bragg rod verticality refinement (`refine_rod_verticality`).
- **Background Subtraction & Corrections (`cd_reduce.reduction.corrections`)**:
  - Background subtraction modes: `"corner"`, `"row_col"`, `"offline_band"`, `"strip_fit"` (`process_frame_strip` with Poisson deviance ML + analytic Jacobian + BIC bleed selection + origin shift), and `"photon_events"`.
  - Geometric & transmission corrections: exposure normalization, solid-angle $\cos^{-3}(2\theta)$, polarization, $\mathrm{Si_3N_4}$ window attenuation $T(\omega, E)$, and beam walk-off $f(\omega)$.
- **2D Mapping & 1D Specular / Off-Specular Extraction (`cd_reduce.reduction.mapping`)**:
  - [`map_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/mapping.py) and [`CdRsoxsLoader.to_qmap_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/loader.py): bin CCD frame series into 2D [`QMap2D`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/mapping.py) maps across any domain pair (`(Qx, Qz)`, `(Qr, Qz)`, `(signed_Qr, Qz)`, `(Qx, Qy)`, `(Qy, Qz)`, `(Qx, omega)`) with flexible orthogonal cuts (`qz_cut`, `qx_cut`, `qy_cut`, `qr_cut`, `omega_cut`, `fixed_domain`).
  - Specular 1D extraction: [`extract_qx_by_omega`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/mapping.py) ($I$ vs $Q_x$ per sample tilt $\omega$) and [`extract_order_rods_qz`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/mapping.py) ($I_m$ vs $Q_z$ per diffraction order $m$).

- **Roughness PSDs (`cd_reduce.reduction.roughness`)**: LPR / LWR / LER PSDs from the off-specular diffuse of a rotation series ([theory](https://github.com/ALS-RSOXS/cd-reduce/blob/main/docs/theory/roughness_psd.md), [how-to](https://github.com/ALS-RSOXS/cd-reduce/blob/main/docs/guides/mapping/roughness_psd.md)). [`reduce_frame`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/roughness/reduce.py) turns each frame into per-pixel reduced intensities $M$ through a cross section (`LatentFilm`, `DevelopedLine`); [`fit_psds`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/roughness/extract.py) fits two-sided LPR / LWR PSDs (nm$^3$, vs $f = Q_y/2\pi$) per $|f|$ bin with model-based (IRLS) variances, and `purity_bands` / `isolation_row_psds` check them on the isolation rows $Q_x = k\pi/w$; [`order_scale`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/reduction/roughness/scale.py) calibrates the incident photon rate from Bragg orders. `CdRsoxsLoader.reduce_roughness` and `CdRsoxsLoader.extract_psds` wrap these and return a `PsdRecord` (CSV with a JSON header, rescalable to a refitted cross section). `map_reduced` / `plot_reduced_map` build exposure-weighted reduced $Q_x$–$Q_z$ and $Q_y$–$Q_z$ maps. Validated against 3D Born and multislice ensembles of rough gratings (`uv run pytest -m validation`, see [`tests/validation/README.md`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/tests/validation/README.md)).
- **Plots (`cd_reduce.reduction.plotting`)**: `plot_traces_by_omega` (I vs Qx/Qr per ω, colour bar and labels), `plot_rods_by_order`, `plot_qmap_grid` (e.g. exact vs small-angle butterflies); `plot_reduced_map` for reduced roughness maps. Butterfly plots (`plot_omega_butterfly`) and `rod_tilts_mrad` are unsmoothed by default; bins and smoothing are given in nm$^{-1}$ (`bin_q`, `sigma_q`).

### 2. 2D & 3D Scattering Simulation (`cd_reduce.simulation`)

- **General Optical Constant Maps (`cd_reduce.simulation.structures`)**:
  - [`RefractiveIndexMap2D`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/structures.py) ($n(x, z) = 1 - \delta(x, z) + i\beta(x, z)$) and [`RefractiveIndexMap3D`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/structures.py) ($n(x, y, z)$), constructed from arbitrary arrays or parameterized grating builders ([`build_grating_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/structures.py), [`build_grating_3d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/structures.py) with LPR / LWR edge roughness from two PSDs, [`build_rough_grating_3d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/structures.py) from a reduction cross section).
- **Rough-grating ensembles (`cd_reduce.simulation.rough_scattering`)**: [`simulate_rough_ensemble`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/rough_scattering.py) averages coherent and diffuse scattering over realisations with `engine="born"` or `engine="multislice"` (optional ε Born reference).
- **Born Approximation (`cd_reduce.simulation.born`)**:
  - [`simulate_born_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/born.py) and [`simulate_born_3d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/born.py) compute continuous 2D reciprocal-space maps (`(Qx, Qz)`, `(Qx, Qy)`, `(Qy, Qz)`), Bragg rod efficiencies $\eta_m(\omega, Q_z)$, off-specular diffuse $Q_y$ traces, and synthetic 2D detector frames (`to_detector_frame`).
- **Multislice / Beam Propagation (`cd_reduce.simulation.multislice`)**:
  - [`simulate_multislice_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/multislice.py) and [`simulate_multislice_3d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/multislice.py) propagate wavefields through arbitrary 2D/3D refractive index maps via Strang-split angular-spectrum propagation across tilt angles $\omega$, capturing soft X-ray attenuation, dynamical phase refraction, resonant Friedel asymmetry, and internal wavefields $\psi(x, z)$.
- **Visualization (`cd_reduce.simulation.visualization`)**:
  - [`plot_born_vs_multislice_grid`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/visualization.py), [`plot_structure_2d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/visualization.py), [`plot_structure_3d`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/visualization.py), [`plot_multislice_wavefield`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/visualization.py), and [`plot_simulation_comparison`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/src/cd_reduce/simulation/visualization.py).

---

## Examples

Runnable scripts covering the complete workflow live in [`examples/`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/examples/):

- [`examples/full_process_walkthrough.py`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/examples/full_process_walkthrough.py) — End-to-end pipeline from 2D/3D structure construction and Born/Multislice simulation to synthetic FITS generation, spot extraction, Laue geometry calibration, background subtraction, 2D $Q$-mapping, and 1D specular reduction.
- [`examples/reduction_and_qmapping.py`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/examples/reduction_and_qmapping.py) — Calibration modes (single-mount, walk-off, joint beamtime $L_b$, rod verticality), template matching, coordinate projections, and slicing filters.
- [`examples/roughness_psd_howto.py`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/examples/roughness_psd_howto.py) — Roughness PSD reduction on synthetic frames: `reduce_frame`, `fit_psds`, isolation rows, `PsdRecord`, reduced maps.
- [`examples/simulation_2d_3d.py`](https://github.com/ALS-RSOXS/cd-reduce/blob/main/examples/simulation_2d_3d.py) — 2D/3D optical constant maps, kinematic vs. dynamical Multislice wave propagation, 3D rough gratings, and diagnostic plotting.

```bash
uv run python examples/full_process_walkthrough.py
uv run python examples/reduction_and_qmapping.py
uv run python examples/simulation_2d_3d.py
uv run python examples/roughness_psd_howto.py
```

Tests:

```bash
uv run pytest                  # unit tests
uv run pytest -m validation    # roughness validation suite (~3.5 min)
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
