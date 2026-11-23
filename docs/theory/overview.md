# Theory overview

## What this page is

This is the landing page for the theoretical documentation of `cd-reduce`. It states the physical problem solved by the package — both inverse reciprocal-space mapping from transmission CD-RSoXS / CD-SAXS CCD tilt series and forward scattering simulation from 2D and 3D refractive index maps — fixes the coordinate conventions and physical assumptions, contrasts the kinematic Born approximation with the dynamical Split-Step Fourier Multislice propagator, and routes the reader to the companion derivation files.

## Table of contents

- [Pipeline at a glance](pipeline.md) — compact stage-by-stage walkthrough of the data reduction pipeline and the 2D/3D forward scattering solvers.
- [Ewald sphere and geometry](geometry_and_ewald.md) — 3D lab-to-sample Ewald sphere transformation, small-angle flat-detector limit, Laue cone calibration, rod-verticality refinement, solid-angle and polarization corrections, $\mathrm{Si_3N_4}$ attenuation, and beam walk-off.
- [Born approximation and roughness PSD](born_and_roughness.md) — kinematic 2D and 3D Born form-factor integrals, self-affine Palasantzas line-edge / line-width roughness power spectral density.
- [Roughness PSDs from diffuse scattering](roughness_psd.md) — the two-edge diffuse model, intensity normalisation, edge contrast $\Gamma(Q_z)$, isolation rows, the weighted extraction of LPR / LWR / LER PSDs and its error budget.
- [Split-step Fourier multislice](multislice.md) — paraxial scalar Helmholtz equation in the tilted sample frame, phase-grating slice transmission and angular-spectrum free-space propagation, internal wavefields, and dynamical Friedel asymmetry.

## Problem statement

A monochromatic soft X-ray beam of photon energy $E$ (wavelength $\lambda = hc / E$, vacuum wavenumber $k_0 = 2\pi / \lambda$) illuminates a periodic line-space nanostructure of pitch $\Lambda$ supported on a thin $\mathrm{Si_3N_4}$ membrane window in transmission geometry. The sample is rotated through a tilt series of angles $\omega$ about the grating line direction ($\hat{\mathbf{y}}$), and a 2D area detector at distance $L_b$ records the transmitted diffraction pattern $I(x_{\mathrm{px}}, y_{\mathrm{px}}; \omega, E)$.

In the sample frame $(\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}})$:

- $\hat{\mathbf{x}}$ lies in the substrate plane perpendicular to the grating lines (the periodicity direction, with fundamental reciprocal lattice vector $G_0 = 2\pi / \Lambda$),
- $\hat{\mathbf{y}}$ lies in the substrate plane parallel to the grating lines,
- $\hat{\mathbf{z}}$ points along the substrate surface normal.

As the sample rotates in $\omega$, the Ewald sphere sweeps through the reciprocal-space Bragg truncation rods at $Q_x = m G_0$, encoding the 2D cross-sectional profile $n(x, z) = 1 - \delta(x, z) + i\beta(x, z)$ along $Q_z$ and the 3D line-edge / line-width roughness along $Q_y$.

## Kinematic vs. dynamical regimes

In hard X-ray CD-SAXS ($E \gtrsim 10\ \mathrm{keV}$), $\delta \sim 10^{-6}$ and $\beta \sim 10^{-8}$, so the kinematic First Born Approximation holds to high accuracy. Near soft X-ray absorption edges in CD-RSoXS (such as the Carbon $K$-edge at $280\text{--}290\ \mathrm{eV}$), the optical constants $\delta, \beta \sim 10^{-3}\text{--}10^{-2}$ become two to three orders of magnitude larger. Over a grating thickness $h \sim 50\text{--}150\ \mathrm{nm}$ (and tilted path length $h / \cos\omega$), phase shifts $k_0 \delta h \sim 0.3\text{--}1\ \mathrm{rad}$ and linear absorption $k_0 \beta h$ produce:

1. **Dynamical depth attenuation** of the internal wavefield before reaching the bottom of the grating,
2. **Waveguiding and channelling** along high-index grating walls,
3. **Breakdown of Friedel's law** ($I_{+m}(\omega) \neq I_{-m}(\omega)$) for absorbing or depth-asymmetric structures.

`cd-reduce` provides both the fast kinematic **Born solver** and the dynamical **Split-Step Fourier Multislice solver** on identical 2D and 3D refractive index grids so users can directly quantify and model dynamical effects.

## References

1. R. L. Jones, T. Hu, E. K. Lin, W.-L. Wu, R. Kolb, D. M. Casa, P. J. Bolton, and G. G. Barclay, "Small angle x-ray scattering for sub-100 nm pattern characterization," *Appl. Phys. Lett.* **83**, 4059 (2003).
2. C. Wang, R. L. Jones, E. K. Lin, W.-L. Wu, B. J. Rice, K.-W. Choi, G. Thompson, and S. J. Weigand, "Characterization of correlated line edge roughness of nanoscale line gratings using small angle x-ray scattering," *J. Appl. Phys.* **102**, 024901 (2007).
3. G. Palasantzas, "Roughness spectrum and surface width of self-affine fractal surfaces via the K-correlation model," *Phys. Rev. B* **48**, 14472 (1993).
4. J. M. Cowley and A. F. Moodie, "The scattering of electrons by atoms and crystals. I. A new theoretical approach," *Acta Crystallogr.* **10**, 609 (1957).
5. B. L. Henke, E. M. Gullikson, and J. C. Davis, "X-ray interactions: photoabsorption, scattering, transmission, and reflection at $E = 50\text{--}30{,}000\ \mathrm{eV}$, $Z = 1\text{--}92$," *At. Data Nucl. Data Tables* **54**, 181 (1993).
