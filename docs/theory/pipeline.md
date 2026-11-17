# Pipeline at a glance

`cd-reduce` decomposes into two core workflows: **Data Reduction & Reciprocal-Space Mapping** (`cd_reduce.reduction`) and **2D/3D Scattering Simulation** (`cd_reduce.simulation`).

## Part 1: Data reduction and reciprocal-space mapping

1. **Frame ingestion & metadata parsing** (`frames.py`): Reads 2D CCD FITS images and parses exposure time $t_{\mathrm{exp}}$, sample tilt angle $\omega$, photon energy $E$, and synchrotron ring current $I_0$.
2. **Diffraction spot extraction** (`spots.py`): Estimates global background via median and MAD, masks the beamstop region, detects connected-component Bragg peaks above a configurable SNR threshold, and computes sub-pixel centroids and local-annulus-subtracted intensities.
3. **Laue cone calibration & rod refinement** (`calibration.py`): Fits the 7 detector/sample Euler parameters $(u_0, v_0, L_b, \beta, \theta_{\mathrm{line}}, \omega_0, \psi)$ plus optional pitch $\Lambda$ and per-frame walk-off shifts $\Delta v_f$ by minimizing deviations from the 1D Laue cone $Q_y = 0, Q_x = m G_0$. A second refinement pass (`refine_rod_verticality`) enforces vertical Bragg truncation rods across all tilts.
4. **Background subtraction & beamstop masking** (`corrections.py`): Removes electronic dark offset (`corner`), 2D row median + cubic column ramp (`row_col`), or off-streak $Q_y$ band background $B(Q_x)$ (`offline_band`), and masks the beamstop shadow across the tilt series.
5. **Physical corrections** (`corrections.py`): Normalizes pixel intensities by exposure time, ring current, planar detector solid-angle Jacobian $\cos^3(2\theta)$, Thomson polarization factor, $\mathrm{Si_3N_4}$ membrane transmission $T(\omega, E)$, and Gaussian beam walk-off $f(\omega)$.
6. **2D & 1D reciprocal-space binning** (`mapping.py`): Projects every valid detector pixel into $(Q_x, Q_y, Q_z, Q_r, \mathrm{signed}\ Q_r, |\mathbf{Q}|, \omega, 2\theta)$, applies optional orthogonal coordinate slices, and bins onto 2D [`QMap2D`](../api/python/reduction.md#cd_reduce.reduction.mapping.QMap2D) grids or 1D [`Trace1D`](../api/python/reduction.md#cd_reduce.reduction.mapping.Trace1D) profiles.

## Part 2: 2D and 3D scattering simulation

1. **Optical constant map construction** (`structures.py`): Builds a 2D cross-section $\Delta n(x, z) = -\delta(x, z) + i\beta(x, z)$ or a 3D rough grating volume $\Delta n(x, y, z)$ with self-affine Palasantzas line-edge / line-width roughness.
2. **Born approximation solver** (`born.py`): Evaluates the kinematic Fourier form factor $F(Q_x, Q_z)$ (2D) or $F(Q_x, Q_y, Q_z)$ (3D) along exact or small-angle Ewald rocking trajectories $Q_z(m, \omega)$ and on full reciprocal-space grids.
3. **Split-Step Fourier Multislice solver** (`multislice.py`): Propagates the complex scalar wavefield $\psi$ slice-by-slice through the tilted refractive index distribution using alternating real-space phase-grating transmission and Fourier-space angular-spectrum propagation, then projects the exit wave onto diffraction orders.
