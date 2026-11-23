# Roughness PSDs from diffuse scattering

This page describes how `cd_reduce.reduction.roughness` turns the off-specular diffuse of a CD-RSoXS rotation series into the two-sided power spectral densities of line placement $S_{\mathrm{LPR}}(f)$ and line width $S_{\mathrm{LWR}}(f)$, and from them $S_{\mathrm{LER}}(f)$. The how-to for measured data is [LER / LWR / LPR PSDs](../guides/mapping/roughness_psd.md).

## Scope and assumptions

The model rests on five assumptions. A reduction can fail silently at each of them.

1. **First-order Born scattering in the edge displacement.** This holds while $Q_x\sigma_{\mathrm{LER}}$ stays well below 1. Beyond that, the nulls of each mode fill (see [Error budget and pitfalls](#error-budget-and-pitfalls)).
2. **Roughness is uncorrelated from line to line.** The diffuse is then continuous in $Q_x$ and adds incoherently over lines. Only the smooth grating is coherent, and it gives the Bragg rods.
3. **Two edges per line.** Within one line, the edges are described by a placement $P(y)$ and a width deviation $W(y)$. Any correlation between the two edges is carried by the relative size of $S_{\mathrm{LPR}}$ and $S_{\mathrm{LWR}}$ and, at most, a $P$–$W$ cross spectrum.
4. **Separable edge contrast.** The edge contrast has a depth profile $\Gamma(Q_z)$ that is the same at every point along the edge. The cross section (pitch, width, edge blur, height, relief, exposure decay, optical constants) comes from a prior fit of the specular and order intensities.
5. **Kinematic attenuation only.** Each pixel is corrected for the mean film absorption on the path in and out, and for any upstream membrane. Multiple scattering and refraction in the film are neglected.

The detector covers a limited range of $|Q_y|$: from the edge of the Bragg mask ($R_B$, about $0.008\ \mathrm{nm^{-1}}$ by default) to the detector edge. The extracted PSDs, and any $\sigma$ integrated from them, refer to that window. Extrapolating beyond it needs a fitted PSD model.

## Notation and conventions

The sample frame has $Q_x$ across the lines (along the grating vector), $Q_y$ along the lines and $Q_z$ along the film normal. Lengths are in nm and $Q$ in $\mathrm{nm^{-1}}$, with $k = 2\pi E/hc$.

| Symbol | Meaning | Units | Package name |
| --- | --- | --- | --- |
| $L$, $G = 2\pi/L$ | pitch, grating vector | nm, $\mathrm{nm^{-1}}$ | `L` |
| $w$ | mean line width (edge to edge) | nm | `w` |
| $s$ | Gaussian edge blur (rms) | nm | `s` |
| $h$, $\Delta h$, $\lambda$ | film thickness, relief depth of the exposed top, exposure decay length | nm | `h`, `dh`, `lam` |
| $h_d$ | developed line height | nm | `h_d` |
| $\Delta n_A$ | index contrast of the unexposed resist against vacuum, $-\delta + i\beta$ | 1 | `dn_A` |
| $r$ | complex ratio of exposed to unexposed contrast at the photon energy | 1 | `r` |
| $P(y)$, $W(y)$ | line placement and width deviation along the line | nm | |
| $u_L$, $u_R$ | edge displacements, $u_L = P - W/2$, $u_R = P + W/2$ | nm | |
| $S_X(f)$ | two-sided PSD of $X$ at $f = Q_y/2\pi$ | $\mathrm{nm^3}$ | `S_lpr`, `S_lwr`, `S_ler` |
| $\omega$, $W = \omega + \omega_0$ | stage rotation about the line direction, and the true tilt | deg | `omega_deg`, `omega0_deg` |

PSDs are two-sided, so the variance is the integral over both signs of $f$:

$$
\sigma_X^2 = \int_{-\infty}^{\infty} S_X(f)\,\mathrm{d}f = \frac{1}{\pi}\int_{0}^{\infty} S_X\!\left(\tfrac{Q_y}{2\pi}\right)\mathrm{d}Q_y
$$

A one-sided PSD is twice as large over $f \ge 0$. Every PSD callable in the package takes $f$, not $Q_y$.

No PSD model is used in the extraction. $M$, the isolation rows and the global fit all work bin by bin in $f$. A model enters in only two places, and both are pluggable:

- the prior $S_{\mathrm{LPR}}/S_{\mathrm{LWR}}$ that sets the purity bands (`purity_bands` takes any callables);
- the optional fit used to extrapolate $\sigma$ outside the window (`fit_psd_model` with a model from `PSD_MODELS`).

The reference simulation uses the Palasantzas (K-correlation) shape, with correlation length $\xi$ and Hurst exponent $H$ (`psd_palasantzas`):

$$
S(f) = 2\sqrt{\pi}\,\xi\,\sigma^2\,\frac{\Gamma(H+\tfrac12)}{\Gamma(H)}\,\left[1 + (2\pi f \xi)^2\right]^{-(H+\frac12)}
$$

Results are stored as a `PsdRecord`: the `PsdTable` (bin edges and centres in $f$, $S$ and variance per mode, $\mathrm{cov}(S_P, S_W)$, condition number, $\chi^2_\mathrm{red}$ and pixel count per bin), the cross section used to divide out $|\Gamma|^2 e^{-Q_x^2 s^2}$, the intensity scale and its source, the data used (energies, $\omega$ range, $R_B$, $w$), the window and $\sigma_{\mathrm{win}}$ per mode. `PsdRecord.save` writes a CSV with a JSON header that states the convention. Because $S$ scales as $1/|\Delta n_A\Gamma|^2$, a later refit of the cross section rescales every stored PSD without re-reducing frames (`PsdRecord.rescale`).

## Geometry: detector pixel to sample frame

Every pixel of every frame is mapped to $(Q_x, Q_y, Q_z)$ in the sample frame by `compute_pixel_q` (`approximation="correct"`, see [Ewald sphere and geometry](geometry_and_ewald.md)). The extraction never works in detector coordinates. Besides $\mathbf Q$, each pixel carries its solid angle $\Omega_{\mathrm{pix}}$ and the exit cosine

$$
\cos\theta_{\mathrm{out}} = \hat{\mathbf r}\cdot\hat{\mathbf n} = \cos W + Q_z/k .
$$

**The inverse map.** Solving $|k\hat{\mathbf z} + \mathbf q| = k$ for $Q_z$ at a given $(Q_x, Q_y)$ and tilt gives, with $g_z$, $l_z$, $n_z$ the beam-direction components of the grating vector, line direction and film normal,

$$
C = Q_x^2 + Q_y^2 + 2k\,(Q_x g_z + Q_y l_z),\qquad Q_z = -k n_z + \operatorname{sgn}(n_z)\sqrt{k^2 n_z^2 - C} .
$$

There is no real root where the point is unreachable at that tilt. One frame therefore samples a curved sheet $Q_z(Q_x, Q_y; \omega)$. $|Q_z|$ grows roughly as $Q_x\tan W$, plus the curvature term $-(Q_x^2 + Q_y^2)/2k$. This is why every pixel carries its own $Q_z$, and why $\Gamma(Q_z)$ is evaluated per pixel, not per frame.

## Intensity normalisation

The measured diffuse per pixel, as a fraction of the incident flux, is

$$
I_{\mathrm{d}} = \frac{k^4|\Delta n_A|^2}{4\pi^2 L\cos W}\;|\Gamma(Q_z)|^2\,e^{-Q_x^2 s^2}\left[4\sin^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LPR}}(f) + \cos^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LWR}}(f)\right]\Omega_{\mathrm{pix}}\,A\,T_m .
$$

The factors, in the order they enter:

- $k^4|\Delta n_A|^2/4\pi^2$ is the Born cross section per unit volume contrast squared. $1/L$ converts the per-line result to per unit area of grating, and $1/\cos W$ is the footprint of the tilted sample. The combination is `howto_prefactor`.
- $e^{-Q_x^2 s^2}$ is the edge blur. A Gaussian blur of rms $s$ multiplies the edge amplitude by $e^{-Q_x^2 s^2/2}$.
- $A$ is the mean film absorption on a path that enters at $W$ and leaves at $\theta_{\mathrm{out}}$, taken from mid-depth (`absorption_factor`, with $\mu h$ from the cross section's `mu_h`):

$$
A = \exp\!\left[-\frac{\mu h}{2}\left(\frac{1}{\cos W} + \frac{1}{\cos\theta_{\mathrm{out}}}\right)\right],\qquad \mu h = 2k\,\operatorname{Im}\!\left(\Delta n_A\,\bar g_0\right),\qquad \bar g_0 = \frac{1}{L}\int_0^L g(x; Q_z{=}0)\,\mathrm dx
$$

- $T_m = T^{1/\cos W}$ is an upstream window of normal-incidence transmission $T$ (`window_T`).
- The polarisation factor $1 - (\hat{\mathbf e}\cdot\hat{\mathbf r})^2$ is omitted. Below $|2\theta| \approx 8°$ it departs from 1 by less than 2 %.

`reduce_frame` starts from a frame in ADU. It converts the frame to detected photons (`adu_per_photon`) and divides by the exposure and the incident photon rate `scale`, so that $I$ is per incident photon. Dividing out the known factors leaves the **reduced intensity** $\mathcal D$, which depends only on the edge contrast and the PSDs. Frames at different $\omega$ and energies can be compared through $\mathcal D$:

$$
\mathcal D(\mathbf Q) \equiv \frac{4\pi^2 L\cos W}{k^4|\Delta n_A|^2\,\Omega_{\mathrm{pix}}\,A\,T_m}\,I_{\mathrm d} = |\Gamma(Q_z)|^2 e^{-Q_x^2 s^2}\left[4\sin^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LPR}} + \cos^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LWR}}\right]
$$

$$
M(Q_x, Q_y) \equiv \frac{\mathcal D}{|\Gamma(Q_z)|^2\,e^{-Q_x^2 s^2}} = 4\sin^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LPR}}(f) + \cos^2\!\tfrac{Q_x w}{2}\,S_{\mathrm{LWR}}(f)
$$

The extraction works on $M$, in $\mathrm{nm^3}$. If the cross section is right, $M$ does not depend on $Q_z$ or $\omega$. Checking that across a rotation series is the first consistency test of a reduction. `ReducedPixels` holds $Q_x, Q_y, Q_z, f, \mathcal D, M$, the variance of $M$, and the factors $M$ and $\mathcal D$ per detected photon (`m_per_photon`, `d_per_photon`) for every valid pixel.

## Scattering model: two edges per line

The method follows from the interference of the two edges of one line. A line occupies $x$ between $-w/2 + u_L(y)$ and $w/2 + u_R(y)$. To first order, moving an edge by $u$ adds a sheet of contrast $\pm\Gamma u$ at that edge, with opposite signs at the rising and falling edge. The Fourier transform along $x$ and $y$ is

$$
A(\mathbf Q) \propto \Gamma(Q_z)\,e^{-Q_x^2 s^2/2}\left[\tilde u_L(Q_y)\,e^{\,iQ_x w/2} - \tilde u_R(Q_y)\,e^{-iQ_x w/2}\right] .
$$

With $u_L = P - W/2$, $u_R = P + W/2$ and $\varphi = Q_x w/2$, this becomes

$$
A \propto \Gamma\,e^{-Q_x^2 s^2/2}\left[\,2i\,\tilde P\sin\varphi - \tilde W\cos\varphi\,\right] .
$$

Averaging $|A|^2$ over uncorrelated lines and over the ensemble turns $|\tilde P|^2$, $|\tilde W|^2$ and $\tilde P\tilde W^*$ per unit line length into $S_{\mathrm{LPR}}$, $S_{\mathrm{LWR}}$ and the cross spectrum $S_{PW}$:

$$
|A|^2 \propto |\Gamma|^2 e^{-Q_x^2 s^2}\left[\,4\sin^2\varphi\,S_{\mathrm{LPR}} + \cos^2\varphi\,S_{\mathrm{LWR}} + 2\sin(Q_x w)\,\operatorname{Im}S_{PW}\right]
$$

The cross term vanishes for any real, even $P$–$W$ cross-correlation. It also vanishes at every isolation row $Q_x w = k\pi$, whatever the correlation. `fit_psds` drops it by default. With `cross_term=True`, it carries $\operatorname{Im}S_{PW}$ as a third unknown.

**LER, LPR and LWR.** The edge PSDs are $S_{u_L} = S_{\mathrm{LPR}} + S_{\mathrm{LWR}}/4 - \operatorname{Re}S_{PW}$ and $S_{u_R} = S_{\mathrm{LPR}} + S_{\mathrm{LWR}}/4 + \operatorname{Re}S_{PW}$. Their mean, the line-edge roughness, needs no assumption about the $P$–$W$ correlation:

$$
S_{\mathrm{LER}} = \tfrac12\left(S_{u_L} + S_{u_R}\right) = S_{\mathrm{LPR}} + \tfrac14 S_{\mathrm{LWR}},\qquad \sigma_{\mathrm{LER}}^2 = \sigma_{\mathrm{LPR}}^2 + \tfrac14\sigma_{\mathrm{LWR}}^2
$$

Consider two edges of equal $\sigma$ with correlation coefficient $\rho$. Then $\sigma^2_{\mathrm{LPR}} = \sigma^2(1 + \rho)/2$ and $\sigma^2_{\mathrm{LWR}} = 2\sigma^2(1 - \rho)$. Independent edges ($\rho = 0$) give $S_{\mathrm{LPR}} = S_E/2$ and $S_{\mathrm{LWR}} = 2S_E$, so $M = 2S_E$ at every $Q_x$. The diffuse is then flat across $Q_x$. Any modulation of the diffuse across $Q_x$ at fixed $Q_y$ is the signature of correlated edges.

The reported LWR is the width deviation $W$ itself, not a per-edge quantity. Some conventions quote LWR as $\sqrt2$ times an edge $\sigma$ for independent edges.

## Edge contrast Γ(Qz)

$\Gamma(Q_z)$ is the difference in depth-integrated scattering amplitude between the two sides of an edge, in units of $\Delta n_A$ and nm. An edge displacement modulates it. It multiplies both PSDs equally and cancels from every ratio between them. The cross sections implement the `CrossSection` protocol (`gamma`, `gamma2`, `g`, `f_m`, `mu_h`, `zeros_qz`).

**Latent film (`LatentFilm`).** Write the exposed fraction across the period as $p(x)$. It is 0 under the unexposed line, 1 between lines, and an erf step of rms $s$ at each edge. Exposure decays from the top with length $\lambda$, the exposed top sinks by $\Delta h\,p(x)$, and $r$ is the exposed-to-unexposed contrast ratio. The depth-integrated amplitude is

$$
g(x; Q_z) = \int_0^{T(x)} \left[\,1 + (r-1)\,p(x)\,e^{-(h - z)/\lambda}\right] e^{-iQ_z z}\,\mathrm dz,\qquad T(x) = h - \Delta h\,p(x) .
$$

The edge contrast is $g$ on the unexposed side minus $g$ on the exposed side, a relief wall minus a chemical wall:

$$
\Gamma(Q_z) = \underbrace{\int_{h-\Delta h}^{h} e^{-iQ_z z}\,\mathrm dz}_{\text{relief}} \;-\; \underbrace{(r-1)\int_0^{h-\Delta h} e^{-(h-z)/\lambda}\,e^{-iQ_z z}\,\mathrm dz}_{\text{chemical}}
$$

$$
\Gamma(Q_z) = \frac{e^{-iQ_z h}\left(e^{iQ_z\Delta h} - 1\right)}{iQ_z} \;-\; (r-1)\,e^{-h/\lambda}\,\frac{e^{\,b\,(h-\Delta h)} - 1}{b},\qquad b = \frac{1}{\lambda} - iQ_z
$$

The relief term vanishes when $\Delta h = 0$, but the chemical term does not: a flat latent film still scatters wherever $r \ne 1$. Both terms are complex, so they can partly cancel. Treat $r$ at each energy as a fitted optical quantity, not a constant.

**Developed line (`DevelopedLine`).** Resist of height $h_d$ against vacuum, with the sidewall blur in $s$:

$$
\Gamma_{\mathrm{dev}}(Q_z) = \int_0^{h_d} e^{-iQ_z z}\,\mathrm dz = \frac{1 - e^{-iQ_z h_d}}{iQ_z}
$$

$|\Gamma_{\mathrm{dev}}|^2$ has true zeros at $Q_z = 2\pi n/h_d$. Pixels near those $Q_z$ carry no diffuse. `reduce_frame` excludes them (`gamma_zero_halfwidth`) rather than dividing by them.

**Numerics and orientation.** $\Gamma$ at $Q_z = 0$ is evaluated by its limit and carried as complex. $|\Gamma|^2$ is what enters $M$. `top_facing="downstream"` flips the sign of $Q_z$ for a sample mounted with its top away from the beam.

## Bragg orders, Debye–Waller factor and masking

The smooth grating scatters into rods at $Q_x = mG$, $Q_y = 0$. The rods swamp the diffuse near $Q_y = 0$, and they also give an independent check on the roughness. The integrated efficiency of order $m$, as photons per incident photon, is (`order_efficiency`)

$$
\eta_m = \frac{k^2|\Delta n_A|^2\,|f_m(Q_z)|^2}{\cos W\,\cos\theta_{\mathrm{out}}}\;e^{-m^2G^2\sigma_{\mathrm{LER}}^2}\;A\,T_m,\qquad f_m(Q_z) = \frac1L\int_0^L g(x;Q_z)\,e^{-imGx}\,\mathrm dx
$$

Here $g(x; Q_z)$ is the depth-integrated amplitude of the cross section; for the developed line it is $(1 - p(x))\,\Gamma_{\mathrm{dev}}$. On the detector each order is a spot of the instrument resolution: a Gaussian of rms $\sigma_Q$ in $Q$, with peak $\eta_m/(2\pi\sigma_Q^2)$ per unit $Q$ area.

**Absolute scale.** `order_scale` fits the incident photon rate $c(E)$ per energy, and one $\sigma_{\mathrm{LER}}$, to the integrated order intensities of an order table: $I_{\mathrm{cps}}/\text{adu\_per\_photon} = c(E)\,\eta_m$. Its scale is what `reduce_frame` takes as `scale`. `CdRsoxsLoader.reduce_roughness(scale="bragg_orders")` does this automatically.

**Debye–Waller.** $e^{-m^2G^2\sigma^2_{\mathrm{LER}}}$ is the coherent loss to edge roughness. To first order the same power reappears in the diffuse. The ratio of orders at fixed $Q_z$, divided by $|f_m/f_1|^2$ from the cross section, yields $\sigma_{\mathrm{LER}}$ without any diffuse (`debye_waller_sigma`). Compare it with the $\sigma$ integrated from the extracted PSDs. The Debye–Waller factor sees the full $\sigma$, the diffuse only the window.

**Mask.** `reduce_frame` removes discs of radius $R_B$ around every $(mG, 0)$, $m \ne 0$, at any $Q_z$. $R_B \approx 5\sigma_Q$ puts the spot tail below $10^{-5}$ of its peak. The beamstop (NaN pixels or `beamstop_mask`) is masked the same way. Masked pixels are dropped, never interpolated, and $R_B$ is the lower edge of the usable $Q_y$ range.

## Isolation rows at Qx = kπ/w

At $Q_x w = k\pi$ one of the two weights in $M$ is exactly zero, so that row of the frame holds one PSD alone. The row positions depend only on the line width, not on the pitch.

| Row | $4\sin^2(Q_x w/2)$ | $\cos^2(Q_x w/2)$ | Row holds | $w = 45$ nm |
| --- | --- | --- | --- | --- |
| $k$ odd ($\pi/w$, $3\pi/w$, $5\pi/w$) | 4 | 0 | $4S_{\mathrm{LPR}}$ | 0.070, 0.209, 0.349 $\mathrm{nm^{-1}}$ |
| $k$ even ($2\pi/w$, $4\pi/w$) | 0 | 1 | $S_{\mathrm{LWR}}$ | 0.140, 0.279 $\mathrm{nm^{-1}}$ |
| $k = 0$ | 0 | 1 | $S_{\mathrm{LWR}}$ | under the beamstop; reachable on $Q_z$ maps only |

At $Q_x = k\pi/w$ the probing period $2\pi/Q_x = 2w/k$ fits the line width an integer number of half periods. For odd $k$ the two edges scatter in antiphase, so a width change cancels. For even $k$ they scatter in phase, so a rigid shift cancels.

**Purity.** The share of LPR in $M$ at one pixel is

$$
\eta_{\mathrm{LPR}}(Q_x, f) = \frac{4\sin^2(Q_x w/2)\,S_{\mathrm{LPR}}(f)}{4\sin^2(Q_x w/2)\,S_{\mathrm{LPR}}(f) + \cos^2(Q_x w/2)\,S_{\mathrm{LWR}}(f)} .
$$

It is independent of $\Gamma$, blur and geometry. A band of purity $p$ is the set of $Q_x$ where $\eta_{\mathrm{LPR}} \ge p$ (or $1 - \eta_{\mathrm{LPR}} \ge p$) for every $f$ in the window. `purity_bands(w, prior_lpr, prior_lwr, f_window, p, qx_max)` returns them as `(k, qx_lo, qx_hi)`. The prior can be AFM, a previous measurement or a first pass of the global fit. Recompute the bands when the prior changes.

**Leakage near a row.** With $\delta = Q_x w - k\pi$, the fraction of the other mode in the row is, to leading order,

$$
\epsilon_{\text{odd }k} \approx \frac{\delta^2}{16}\,\frac{S_{\mathrm{LWR}}}{S_{\mathrm{LPR}}},\qquad \epsilon_{\text{even }k} \approx \delta^2\,\frac{S_{\mathrm{LPR}}}{S_{\mathrm{LWR}}} .
$$

The band half-width in $Q_x$ is therefore $\delta_p/w$, with $\delta_p$ from $\epsilon = 1 - p$. An error $\Delta w$ in the assumed width shifts the chosen row by $\delta = k\pi\Delta w/w$ from the true null. The leakage grows as $(k\Delta w/w)^2$, so the higher rows need a better $w$. At the same $\delta$, LWR rows are about 16 times more sensitive than LPR rows.

## Extraction algorithm

The estimator is a weighted two-mode least squares over every usable pixel in each $f$ bin. The isolation rows are its special case and its check. Neither needs a PSD model.

1. **Per pixel, per frame (`reduce_frame`).** Map to $(Q_x, Q_y, Q_z)$, apply the Bragg and beamstop masks, and form $\mathcal D$ and $M$.
2. **Reject weak contrast.** Drop pixels where $|\Gamma(Q_z)|^2 e^{-Q_x^2 s^2}$ is below `keep` (default 0.1) times its maximum, and pixels near the zeros of $\Gamma$. Dividing by a small, model-dependent number turns cross-section error into PSD error.
3. **Fold and bin.** $M$ is even in $Q_x$ and $Q_y$, so fold to $|Q_x|$ and $|f|$. `default_f_edges` gives 26 log-spaced bins from $R_B/2\pi$ to the detector edge.
4. **Fit each $f$ bin (`fit_psds`).** With $a_i = 4\sin^2(Q_{x,i} w/2)$, $b_i = \cos^2(Q_{x,i} w/2)$ and weights $\omega_i = 1/\mathrm{var}(M_i)$ over the pixels $i$ in the bin:

    $$
    \min_{S_P, S_W}\;\sum_i \omega_i\left(M_i - a_i S_P - b_i S_W\right)^2 \;\Rightarrow\; \begin{pmatrix}\sum\omega a^2 & \sum\omega ab\\ \sum\omega ab & \sum\omega b^2\end{pmatrix}\begin{pmatrix}S_P\\ S_W\end{pmatrix} = \begin{pmatrix}\sum\omega a M\\ \sum\omega b M\end{pmatrix}
    $$

    The inverse of the normal matrix, scaled by $\max(1, \chi^2_\mathrm{red})$, is the covariance of $(S_P, S_W)$ in that bin. Its condition number (`cond`) shows whether the pixels used separate the modes. Pixels near both kinds of null give a well-conditioned fit; pixels only near the orders do not. Bins above `max_cond` or with fewer than `min_pixels` are NaN.

    **Variance.** The variance is model-based, not taken from the observed counts. With $m_i$ the value of $M$ per detected photon (`m_per_photon`) and $\hat M_i = a_i S_P + b_i S_W$ the current prediction, $\mathrm{var}(M_i) = m_i^2\,(\hat M_i/m_i + \sigma_{\mathrm{read}}^2)$. The fit iterates (IRLS) from the observed variance to convergence. Weights from the observed counts correlate with the noise. At low counts per pixel they bias the PSD low, by about 20 % in the package's unit test.

    `qx_max` keeps only pixels with $|Q_x| \le$ `qx_max`. Set it to about $0.3/\sigma_{\mathrm{LER}}$ to stay inside the first-order range.

5. **Isolation-row estimate, as a check (`isolation_row_psds`).** On the pixels of each purity band, $S_{\mathrm{LPR}} = M/(4\sin^2(Q_x w/2))$ for odd $k$ and $S_{\mathrm{LWR}} = M/\cos^2(Q_x w/2)$ for even $k$, per $f$ bin and per $k$. Agreement with step 4 within the errors confirms the two-term form. A systematic offset between $k$ values points to a wrong $w$, $s$ or $|\Gamma|$.
6. **Residuals.** Plot $M_i - a_i S_P - b_i S_W$ against $Q_x$ at fixed $f$, and against $Q_z$. Structure in $Q_x$ means the two-edge model or $w$ is wrong. Structure in $Q_z$ or between $\omega$ means $\Gamma(Q_z)$ is wrong.

**Quick estimate (`quick_ler`).** The mean of $M$ over whole periods of $Q_x w$ needs no fit. It equals $2S_{\mathrm{LPR}} + S_{\mathrm{LWR}}/2 = 2S_{\mathrm{LER}}$, because $\sin^2$ and $\cos^2$ both average to 1/2. It gives $S_{\mathrm{LER}}$ directly and is the right first number to report.

## LER and σ, in the window and in full

$S_{\mathrm{LER}}$ follows bin by bin from the two extracted spectra, with its error from their covariance in that bin:

$$
S_{\mathrm{LER}}(f) = S_{\mathrm{LPR}}(f) + \tfrac14 S_{\mathrm{LWR}}(f),\qquad \operatorname{var}S_{\mathrm{LER}} = \operatorname{var}S_P + \tfrac1{16}\operatorname{var}S_W + \tfrac12\operatorname{cov}(S_P,S_W)
$$

If LPR and LWR come from different rows (isolation-row estimate), combine them only at common $f$ bins.

**σ in the window (`sigma_window`).** Integrate the binned PSD over the measured range only, with the trapezoid rule on the finite bin centres:

$$
\sigma_{X,\mathrm{win}}^2 = \frac{1}{\pi}\int_{Q_{y,1}}^{Q_{y,2}} S_X\!\left(\tfrac{Q_y}{2\pi}\right)\mathrm dQ_y = 2\int_{f_1}^{f_2} S_X(f)\,\mathrm df
$$

This number is model-free, and it is the one to report from the reduction. `PsdRecord.sigma_win` holds it, with its error, for each mode.

**σ in full.** The missing variance lies below $f_1$ and above $f_2$. Below $f_1$ are long wavelengths along the line, limited by the beam's coherence length and the Bragg mask. Above $f_2$ are short wavelengths beyond the detector. Extrapolating needs a model fitted to the stored table (`fit_psd_model`). Report the model, its parameters and the fraction of $\sigma^2$ it supplies outside the window, so that the extrapolated part is never mistaken for a measurement. The Debye–Waller ratio of the orders gives an independent full $\sigma_{\mathrm{LER}}$ to compare against.

## Reciprocal-space maps from ω scans

$Q_x$–$Q_z$ and $Q_y$–$Q_z$ maps bin the reduced intensity $\mathcal D$ of every pixel of every frame, never raw counts. Each frame carries its own $1/\cos W$, absorption path, window transmission and solid angle. Only after that division can frames at different $\omega$ share a bin.

`map_reduced(pixels, plane, slab_center=..., slab_halfwidth=..., bin_q=...)` keeps a slab: $|Q_y - c| \le \Delta$ for `"qx-qz"` and $|Q_x - c| \le \Delta$ for `"qy-qz"`. Each bin is an exposure-weighted Poisson maximum-likelihood estimate. With $d_i$ the value of $\mathcal D$ per detected photon, the bin holds $H = \sum \mathcal D_i/d_i$ (detected photons) and $N = \sum 1/d_i$ (exposure), and the map is $H/N$. Zero-count pixels are kept. Weighting by the observed variance would drop them and bias low-count bins high. Empty bins stay empty: they are $Q$ the scan never reached, a bow-tie around $Q_x = 0$ for a symmetric $\omega$ range. `plot_reduced_map` draws the map with the slab in the title and the rows $Q_x = \pm k\pi/w$.

The slab is an average over the fixed coordinate. A smooth diffuse barely notices it, but the Bragg rods are only $\sigma_Q$ wide in $Q_y$, so a slab that does not contain $Q_y = 0$ shows no rods. The rod intensity in the slab is the mean of a Gaussian over the slab:

$$
\langle e^{-q^2/2\sigma_Q^2}\rangle_{c\pm\Delta} = \frac{\sigma_Q\sqrt{\pi/2}}{2\Delta}\left[\operatorname{erf}\!\left(\frac{c+\Delta}{\sqrt2\,\sigma_Q}\right) - \operatorname{erf}\!\left(\frac{c-\Delta}{\sqrt2\,\sigma_Q}\right)\right]
$$

Here $c$ is the slab centre measured from the rod. Report $\Delta$ and the slab centre with every map.

On the maps the isolation rows become columns at $Q_x = k\pi/w$ ($Q_x$–$Q_z$). The profile of $\mathcal D$ along $Q_z$ in such a column is $|\Gamma(Q_z)|^2$ times one PSD value. This is the cleanest direct test of the $\Gamma$ model and of the cross section, independent of the roughness.

## Error budget and pitfalls

The largest systematic is $|\Gamma(Q_z)|^2$, because the PSDs are measured only relative to it. Every other item below is smaller, or testable inside the data set.

| Source | Effect on the extracted PSD | Check inside the data |
| --- | --- | --- |
| $\lvert\Gamma(Q_z)\rvert^2$ ($h$, $\Delta h$, $\lambda$, $r$ from the cross-section fit) | $S$ scales as $1/\lvert\Gamma\rvert^2$. A $Q_z$-dependent error distorts $S$ and differs between $\omega$ | $M$ must agree between frames at different $\omega$ and between energies |
| $\Delta n_A$ | common scale on both PSDs; cancels in $S_{\mathrm{LPR}}/S_{\mathrm{LWR}}$ | none; propagate its fit error |
| line width $w$ | rows sit off the nulls; leakage $\approx (k\pi\Delta w/w)^2$ times the PSD ratio | isolation-row estimates at different $k$ must agree |
| edge blur $s$ | relative error $2Q_x^2 s\,\Delta s$ in $S$, growing with $Q_x$ | residuals of the global fit against $Q_x$ |
| first order in $u$ | beyond $\lvert Q_x\rvert\sigma_{\mathrm{LER}} \approx 0.3$ the first-order model underestimates the diffuse: the nulls fill and the peaks fall short. In the validation ensembles, the ratio of ensemble to model drops to about 0.9 at $\lvert Q_x\rvert\sigma \approx 0.45$. At large ensemble size, null filling at the isolation rows leaves a few-percent row-vs-global offset | fit with `qx_max` $\approx 0.3/\sigma_{\mathrm{LER}}$; compare low and high $k$ rows of the same type |
| separable $\Gamma$ (assumption 4) | violated by a latent film whose relief shrinkage $\Delta h\,p(x)$ sits inside the depth integral. Across the blurred edge, the edge spectrum is then not a single $\Gamma(Q_z)$. For single-mode roughness (LPR only or LWR only) power is redistributed between the $\sin^2$ and $\cos^2$ lobes. In the validation ensembles the separable model misses the single-mode latent diffuse by 5 to 14 % per pixel, while $\sigma_{\mathrm{win}}$ stays within about 1 to 5 %. With both modes present the effect cancels | compare isolation rows of both types; a developed line (no shrinkage) satisfies the assumption |
| flux obliquity | detector flux carries $1/(\cos W\cos\theta_{\mathrm{out}})$, not $1/\cos^2 W$. Amplitudes simulated along a tilted path need the factor $\cos W/\cos\theta_{\mathrm{out}}$ before comparison with frames (`RoughEnsembleResult.flux_factor`, `diffuse_flux`, `order_flux_efficiency`) | orders and diffuse at large $\omega$ against the model |
| low-count weighting | per-pixel weights from observed counts bias PSDs low (fit) and maps high (zero-count pixels dropped) | the package uses model-based IRLS variances in the fits and exposure-weighted maps |
| line-to-line correlation | diffuse concentrates near the orders instead of being continuous in $Q_x$ | modulation of $M$ with period $G$ across $Q_x$ |
| resolution and coherence | $Q_y$ smearing by $\sigma_Q$ flattens $S$ below $f$ of a few $\sigma_Q/2\pi$ | do not report bins below about $5\sigma_Q$ |
| speckle and counting | scatter per pixel of order 100 % before binning | binned scatter against propagated variance ($\chi^2_\mathrm{red}$ per bin) |
| background (membrane, fluorescence, dark) | additive floor that flattens the high-$f$ roll-off | measure an off-grating spot at each $\omega$ and energy |
| absorption path and window | frame-to-frame scale | same as for $\lvert\Gamma\rvert^2$ |
| polarisation | under 2 % below $2\theta \approx 8°$ | none needed in that range |
| dose | latent contrast and roughness change during the scan | repeat the first frame at the end |

## Validation

The forward model and the reduction are checked against simulated data in `tests/validation/` (run with `uv run pytest -m validation`). The checks recover known PSDs from frames simulated with the analytic model and from 3D Born and multislice ensembles of explicit rough edges. They also check that $M$ is the same at every $\omega$, that the isolation rows agree with the global fit, and that the Debye–Waller $\sigma$ matches the generated edges. `tests/validation/README.md` gives the pass criteria and the measured numbers for the reference grating used there.

## Beyond first-order Born: 3D multislice

`simulate_rough_ensemble(..., engine="multislice")` propagates each realisation of a rough grating (`build_rough_grating_3d`) through the film with the split-step method of [Split-step Fourier multislice](multislice.md). The field $\psi(x, y)$ on a periodic supercell is marched through slices of thickness $\Delta z$. The incident plane wave at $W$ has transverse wavevector $\mathbf k_t$. With the envelope $\phi = \psi\,e^{-i\mathbf k_t\cdot\mathbf r}$, each slice applies the local phase screen and then the angular-spectrum propagator:

$$
\phi \leftarrow \mathcal F^{-1}\!\left\{\mathcal F\!\left[\phi\,e^{\,ik\,\Delta n(x,y,z)\,\Delta z}\right] e^{\,i\left[k_z(\mathbf q) - k_z(0)\right]\Delta z}\right\},\qquad k_z(\mathbf q) = \sqrt{k^2 - |\mathbf k_t + \mathbf q|^2}
$$

The rough top surface is filled by volume fraction in the surface slice, and evanescent components decay. The exit amplitudes $c_{\mathbf q} = \mathcal F[\phi]/N$ leave at $Q_z = k_z(\mathbf q) - k_z(0)$, so each mode lands on the Ewald sphere exactly as a detector pixel does.

With `born_reference="epsilon"`, the same realisation is also propagated with $\Delta n$ scaled by $\epsilon$ (default $10^{-3}$). That gives first Born (exact $Q_z$, no absorption) as $|c_{\mathbf q}(\epsilon)|^2/\epsilon^2$, stored in `metadata["born_eps"]`. The ratio $|c_{\mathbf q}|^2\epsilon^2/|c_{\mathbf q}(\epsilon)|^2$ has no speckle. It isolates absorption, refraction and multiple scattering. Compare it with $A$ to test the kinematic attenuation of assumption 5.

For the reference grating in `tests/validation` at 285 eV, multislice / (Born × $A$) is 1.002 for the latent film and 1.007 for the developed line. The recovered PSDs pass the same criteria with either engine. Multislice is the right forward model where Born × $A$ is not enough: strongly absorbing energies, large $\omega$, and high-contrast developed lines. The size of the correction follows the fitted $\beta$, so check it with the cross section of the actual sample.
