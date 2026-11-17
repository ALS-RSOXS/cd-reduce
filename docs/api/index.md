# API reference

Python API pages are generated from source docstrings with `mkdocstrings`.

## Guides

- [Data reduction](../guides/reduction/index.md) — spot extraction, Laue geometry calibration, rod verticality refinement, background subtraction, and window/walk-off corrections
- [Reciprocal-space mapping](../guides/mapping/index.md) — 2D reciprocal-space domains, exact vs. small-angle Ewald projections, fixed-coordinate slices, and 1D traces
- [Building and simulating structures](../guides/simulation/index.md) — 2D & 3D optical constant maps, Born approximation, Split-Step Fourier Multislice, and rough line gratings

## Python

- [`cd_reduce`](python/cd_reduce.md) — top-level package exports
- [`cd_reduce.loader`](python/loader.md) — high-level [`CdRsoxsLoader`](python/loader.md#cd_reduce.loader.CdRsoxsLoader) interface
- [`cd_reduce.reduction`](python/reduction.md) — FITS I/O, spot extraction, Ewald geometry, calibration, background subtraction, physical corrections, and 2D/1D $Q$-space mapping
- [`cd_reduce.simulation`](python/simulation.md) — 2D & 3D refractive index builders, Born and Multislice solvers, and visualization utilities
