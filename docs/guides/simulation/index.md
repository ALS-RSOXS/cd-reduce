# Building and simulating structures

Guide to assembling general 2D and 3D refractive index maps $n(\mathbf{r}) = 1 - \delta(\mathbf{r}) + i\beta(\mathbf{r})$ in `cd-reduce` and simulating transmission CD-RSoXS / CD-SAXS diffraction using the **Born Approximation** ([`simulate_born_2d`](../../api/python/simulation.md#cd_reduce.simulation.born.simulate_born_2d), [`simulate_born_3d`](../../api/python/simulation.md#cd_reduce.simulation.born.simulate_born_3d)) and **Split-Step Fourier Multislice** ([`simulate_multislice_2d`](../../api/python/simulation.md#cd_reduce.simulation.multislice.simulate_multislice_2d), [`simulate_multislice_3d`](../../api/python/simulation.md#cd_reduce.simulation.multislice.simulate_multislice_3d)).

## Contents

| Page | What it covers |
| --- | --- |
| [2D structures and wavefields](structures_2d.md) | Parametric multi-layer profiles (`GratingLayerSpec`, `build_grating_2d`), arbitrary custom 2D $\delta(x, z) + i\beta(x, z)$ maps (`RefractiveIndexMap2D`), and internal multislice wavefields $|\psi(x, z)|^2$ & $\arg\psi(x, z)$ |
| [2D Born vs. Multislice](born_vs_multislice_2d.md) | Multi-energy comparison (`plot_born_vs_multislice_grid`) unravelling pre-edge kinematic agreement (`280 eV`), resonant internal refraction & Friedel asymmetry (`285.2 eV`), and continuum dynamical absorption (`288.6 eV`) |
| [3D rough line gratings](rough_gratings_3d.md) | 3D rough line gratings (`RefractiveIndexMap3D`, `build_grating_3d`) with self-affine Palasantzas LER vs. LWR, 3D Born & 3D Multislice solvers, and simulated off-specular diffuse sheets |
