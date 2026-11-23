# Slicing and 1D traces

From a calibrated tilt series or projected [`QMap2D`](../../api/python/reduction.md#cd_reduce.reduction.mapping.QMap2D), `cd-reduce` separates **specular Bragg diffraction** ($Q_y \approx 0$) from **off-specular diffuse roughness scattering** ($Q_y \neq 0$) across every diffraction order $m$ and tilt angle $\omega$. This page covers the specular side; the diffuse is reduced to roughness PSDs in [LER / LWR / LPR PSDs](roughness_psd.md).

## Specular 1D Extraction: $I(Q_x;\,\omega)$ and $I_m(Q_z)$

The two primary specular 1D reductions extract order intensities either per tilt angle $\omega$ or per diffraction order $m$:

1. **$I$ vs. $Q_x$ for each tilt $\omega$** ([`extract_qx_by_omega`](../../api/python/reduction.md#cd_reduce.reduction.mapping.extract_qx_by_omega) / [`CdRsoxsLoader.extract_qx_by_omega`](../../api/python/loader.md#cd_reduce.loader.CdRsoxsLoader.extract_qx_by_omega)): returns `{omega_deg: Trace1D}` across all indexed Bragg orders, revealing Friedel tilts and order-envelope evolution with $\omega$ (panel **a**).
2. **$I$ vs. $Q_z$ for each order $m$** ([`extract_order_rods_qz`](../../api/python/reduction.md#cd_reduce.reduction.mapping.extract_order_rods_qz) / [`CdRsoxsLoader.extract_order_rods_qz`](../../api/python/loader.md#cd_reduce.loader.CdRsoxsLoader.extract_order_rods_qz)): returns `{order: Trace1D}` along each Bragg truncation rod $+4 \dots -4$, encoding the 2D cross-sectional line profile (panel **b**). Setting `normalize_first_order=True` computes $I_m(Q_z) / \bigl(I_{+1}(Q_z) + I_{-1}(Q_z)\bigr)$, canceling incident flux and contrast prefactors across `latent`, `PEB`, and `developed` stages (panel **c**).

![Specular 1D ridge waterfall I vs Qx per omega, stacked Bragg rods I vs Qz per order, and first-order normalized stage comparison](../../assets/images/specular_1d_rods_and_ridges.png)

```python
import cd_reduce as cdr

# 1. Main view A: I vs Qx for each sample tilt omega
ridges_by_omega = loader.extract_qx_by_omega()  # dict[float, Trace1D]

# 2. Main view B: I vs Qz for each diffraction order m
rods_by_order = loader.extract_order_rods_qz(
    orders=(-4, -3, -2, -1, 1, 2, 3, 4),
    normalize_first_order=False,
)

# First-order normalized ratio I_{+2}(Qz) / (I_{+1} + I_{-1})
rods_norm = loader.extract_order_rods_qz(orders=(2,), normalize_first_order=True)
```

Arbitrary single-domain cuts at any fixed $(Q_x, Q_y, Q_z, Q_r, \omega)$ window are also available via [`CdRsoxsLoader.to_trace_1d`](../../api/python/loader.md#cd_reduce.loader.CdRsoxsLoader.to_trace_1d) and [`QMap2D.slice_1d`](../../api/python/reduction.md#cd_reduce.reduction.mapping.QMap2D.slice_1d).

## Off-specular diffuse

Line-edge and line-width roughness scatter out of the $Q_y = 0$ plane into diffuse sheets that are continuous in $Q_x$. Raw diffuse counts are not comparable between frames: each frame has its own footprint, absorption path and solid angle per pixel. The package therefore works on the reduced intensity of every pixel (`reduce_frame`), and builds off-specular $Q_x$–$Q_z$ and $Q_y$–$Q_z$ maps from it with `map_reduced` and `plot_reduced_map`. The workflow is in [LER / LWR / LPR PSDs](roughness_psd.md), and the model behind it in [Roughness PSDs from diffuse scattering](../../theory/roughness_psd.md).
