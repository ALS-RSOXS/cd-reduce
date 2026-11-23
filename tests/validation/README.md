# Roughness reduction: validation suite

These tests check the roughness-PSD reduction (`cd_reduce.reduction.roughness`) and the rough-grating simulation (`simulate_rough_ensemble`) against data with known roughness. The method is described in `docs/theory/roughness_psd.md`. Nothing here is part of the package or its API docs.

## Running

```bash
uv run pytest -m validation                     # about 3.5 min, 41 tests
uv run python tests/validation/run_study.py     # full-size study (long)
uv run python tests/validation/run_study.py --quick   # smoke run of the study, seconds
```

The default `uv run pytest` deselects these tests. Report tests and the study write JSON, CSV and PNG files to `tests/validation/results/`, which is git-ignored.

## Reference grating and data

`reference.py` holds the reference grating. All numbers are generic and round.

- **Cross section.** Pitch 80 nm, width 45 nm, blur 2.5 nm.
  - Latent film: $h$ 40 nm, $\Delta h$ 4 nm, $\lambda$ 25 nm. At 285 eV, $r = 1.15 + 0.05i$, so the relief and chemical terms partly cancel ($|\Gamma(0)|$ 1.76 nm vs 4.0 nm relief only).
  - Developed line: $h_d$ 25 nm.
  - Tapered line: 85° sidewall, 25 nm high, mean width 45 nm.
- **Roughness.** Palasantzas PSDs:
  - LPR: $\sigma$ 1.0 nm, $\xi$ 25 nm, $H$ 0.7;
  - LWR: $\sigma$ 1.5 nm, $\xi$ 12 nm, $H$ 0.5;
  - "independent" edges: $S_E$ with $\sigma$ 1.2 nm, $\xi$ 20 nm, $H$ 0.6.
- **Cases.**
  - On the latent film: smooth, `lpr`, `lwr`, `both`, `independent`.
  - On the developed line: `lpr_developed`, `lwr_developed`, `both_developed`.
  - On the tapered line: `tapered` (LPR + LWR).
- **Sizes.**
  - `small()`, used by the tests: 512² detector, 285 eV, 4 lines, $L_y$ 1024 nm, 2 realisations per ensemble. Ensembles use 5 tilts from −10° to 10°, analytic frames 3.
  - `run_study.py`: 1024² detector, three energies, $\omega$ from −20° to 20° in 2° steps, 16 lines, $L_y$ 2048 nm, 8 realisations.

There are two kinds of frames:

- **Analytic frames** (`forward/model_frames.py`) are rendered from the HOWTO forward model with known PSDs, Bragg orders, Poisson noise and a beamstop.
- **Ensemble frames** (`forward/ensemble_frames.py`) are rendered from `simulate_rough_ensemble` results: explicit rough edges and no HOWTO model inside. They have the same flux normalisation, orders, noise and beamstop.

Both kinds of frame are reduced with the package (`reduce_frame`, `fit_psds`, ...) exactly as measured data would be.

## What each test proves

| Test | Proves |
| --- | --- |
| `test_self_consistency.py::test_self_consistency_recovers_psds` | the reduction inverts its own forward model: known PSDs come back from noisy analytic frames |
| `test_self_consistency.py::test_scale_recovery` | `order_scale` recovers the incident photon rate from the Bragg orders |
| `test_self_consistency.py::test_latent_contrast_reduces_gamma0` | the reference latent film has partial relief/chemical cancellation |
| `test_loader_smoke.py` | `CdRsoxsLoader.extract_psds` runs end to end and fills the `PsdRecord`; unknown energies raise |
| `test_model_error.py::test_ensemble_matches_model` | the first-order two-edge model reproduces explicit-edge Born ensembles (model error) |
| `test_model_error.py::test_ensemble_recovers_sigma_window` | PSDs reduced from ensemble frames give the right $\sigma$ in the window |
| `test_model_error.py::test_developed_line_case` | the same two checks on the developed line with both modes |
| `test_model_error.py::test_M_invariant_across_omega` | $M$ is independent of tilt (geometry, $\Gamma(Q_z)$, absorption) |
| `test_model_error.py::test_isolation_rows_agree_with_global` | isolation rows $Q_x = k\pi/w$ agree with the global fit ($w$, $s$, two-edge form) |
| `test_model_error.py::test_debye_waller_sigma_from_ensemble_frames` | `debye_waller_sigma` recovers the edge $\sigma$ from the orders |
| `test_model_error.py::test_smooth_case_has_no_diffuse` | a smooth grating gives orders and no diffuse |
| `test_model_error.py::test_matched_model_normalisation_guard` | the matched-realisation model has the HOWTO normalisation (white-noise edges) |
| `test_model_error.py::test_band_limited_truth_matches_psd` | the realised edge $\sigma$ agrees with the band integral of the input PSDs |
| `test_multislice.py::test_model_error_multislice` | ratio and $\sigma_{\mathrm{win}}$ criteria hold with the multislice engine (`both`, `both_developed`) |
| report: `test_ms_over_born_report` | multislice / (Born × $A$) per $\omega$ → `ms_over_born.json` |
| report: `test_latent_nonseparable_report` | single-mode latent cases, where assumption 4 fails → `latent_nonseparable.json` |
| report: `test_single_mode_sigma_window_report` | single-mode developed $\sigma_{\mathrm{win}}$ at 16 and 32 realisations → `sigma_win_single_mode.json` |
| report: `test_first_order_deficit_report` | ensemble / model against $\lvert Q_x\rvert$ beyond the first-order cut → `first_order_deficit.json` |
| report: `test_tapered.py` | bias of the separable model for a tapered line → `tapered_bias.json` |
| study only: `isolation_systematic()` | isolation-vs-global $\chi^2$ at 32 realisations → `isolation_systematic.json` |

Report tests assert only that their numbers are finite.

## Pass criteria (as implemented)

Some choices apply to all the ensemble criteria.

- **First-order cut.** Only pixels with $|Q_x|\sigma_e \le 0.3$ are used (`fit_psds(qx_max=...)`), where $\sigma_e$ is the realised LER $\sigma$ of the generated edges. Isolation rows must have $k\pi/w \cdot \sigma_e \le 0.3$. The first-order model is not expected to hold beyond the cut; the deficit there is reported, not tested.
- **Truths.**
  - $\sigma_{\mathrm{win}}$ is compared with the realised spectra of the generated $P$ and $W$ fields (periodograms averaged over lines and realisations, binned like the fit), not with the input PSDs.
  - The model for the ensemble/model ratio is the matched-realisation first-order prediction from the actual Fourier coefficients of the generated edges.
  - The Debye–Waller truth is the realised $\sigma_e$.
- **Errors.** The $\chi^2$ criteria take their errors from the scatter over $K = 8$ independent ensembles (seeds), not from Poisson statistics, because speckle dominates. The ratio and $\sigma_{\mathrm{win}}$ checks pool the 8 ensembles (16 realisations).
- **Ratio blocks.** The ratio is taken on blocks of 40 × 25 pixels (about 3 × 6 ensemble modes). Blocks where the model is below 10 % of its maximum at that $|Q_y|$ are excluded.

| Test | Data | Pass |
| --- | --- | --- |
| Self-consistency | analytic frames, `both` | median over $f$ bins of recovered / input in 0.97–1.03 for LPR and LWR; every bin within 3σ of the same estimator on noise-free $M$; $\sigma_{\mathrm{win}}$ (LPR, LWR, LER) within 3 % |
| Scale recovery | analytic frames | `order_scale` within 2 % of the known scale |
| Model error, ratio | Born ensembles: `both`, `independent`, `lpr_developed`, `lwr_developed`, `both_developed` | median ensemble / matched model per $\omega$ in 0.95–1.05 |
| Model error, $\sigma_{\mathrm{win}}$ | same cases | each mode and LER within 5 % of the realised truth. Single-mode developed cases pool 32 realisations (4 × 8) |
| Model error, multislice | multislice ensembles, `both` and `both_developed` (ε Born reference) | the same ratio and $\sigma_{\mathrm{win}}$ criteria |
| $M$ invariance | Born ensembles: `lpr`, `lwr`, `both`, `independent` | $\chi^2/\mathrm{dof} < 2$ (dof ≥ 10) for $S$ per $\omega$ group against the mean over groups, per $f$ bin |
| Isolation vs global | same cases, $k \le 4$ inside the cut, 90 % purity | $\chi^2/\mathrm{dof} < 2$ (dof ≥ 10) for (row − global)/global |
| Debye–Waller | same cases, orders ±1 to ±3 | `debye_waller_sigma` within 10 % of the realised $\sigma_e$ |
| Smooth | smooth latent film | diffuse < $10^{-12}$ of the first-order efficiency |
| Latent single mode | `lpr`, `lwr` | report only (assumption 4 does not hold; see Findings) |
| Tapered line | `tapered` | report only |
| MS / (Born × $A$) | `both`, `both_developed`, 285 eV | report; warns outside 0.95–1.05 |
| Isolation systematic | 32 realisations × 4 seeds | report only, `run_study.py` |

Single-mode cases are tested on the developed line, where the separable-$\Gamma$ assumption holds. The isolation and $M$-invariance criteria run at the small size (8 × 2 realisations), where they pass. At larger size, the isolation systematic below appears.

## Findings

The numbers come from the `small()` size unless stated otherwise.

- **Self-consistency.**
  - Median recovered / input: 1.000 (LPR) and 1.000 (LWR).
  - $\sigma_{\mathrm{win}}$ ratios: 1.001 to 1.002.
  - Worst bin: 2.2σ.
  - Scale recovered within 0.01 %.
- **Model error (Born).**
  - Ensemble / matched model medians per $\omega$: 0.988 to 0.999 for every tested case.
  - $\sigma_{\mathrm{win}}$ / realised truth: 0.97 to 1.00 for the two-mode cases (`both` 0.988 / 0.995 / 0.990 for LPR / LWR / LER).
  - Developed single mode at 32 realisations: 0.976 (LPR) and 0.974 (LWR). At 16 realisations these sat at 0.952 and 0.953, which is sampling noise.
- **Multislice.**
  - Ratios 0.990 to 1.000 and $\sigma_{\mathrm{win}}$ 0.989 to 0.996.
  - MS / (Born × $A$) at 285 eV: 1.0015 to 1.0017 (latent, `both`) and 1.0069 to 1.0070 (developed). The kinematic absorption factor accounts for all but 0.2 % and 0.7 %.
- **$M$ invariance.** $\chi^2/\mathrm{dof}$ 1.03, 1.02, 0.75, 0.75 (`lpr`, `lwr`, `both`, `independent`).
- **Isolation vs global.** $\chi^2/\mathrm{dof}$ 0.58, 0.70, 0.81, 0.77.
- **Debye–Waller.** $\sigma$ / realised truth 1.021, 1.038, 1.011, 1.012.
- **Separable $\Gamma$ fails for a latent film with shrinkage.** The relief shrinkage sits inside the depth integral, so across the blurred edge the edge spectrum is not a single $\Gamma(Q_z)$.
  - For single-mode roughness, power is redistributed between the $\sin^2$ and $\cos^2$ lobes. The separable model on the realised spectra gives ensemble / model medians of 0.86 to 0.89 (LPR only) and 0.93 to 0.95 (LWR only), and $\sigma_{\mathrm{win}}$ 0.948 and 0.949.
  - The matched first-order model gives 0.968 to 1.004, so the ensemble itself is first-order.
  - With both modes present the effect cancels.
  - In the full-size study (16 lines, 8 realisations) the latent single-mode $\sigma_{\mathrm{win}}$ ratios are 0.986 to 0.994. Much of the 5 % at the small size is sampling noise, and the separable bias shows mainly in the per-pixel ratio.
- **Tapered line.** The separable model biases $\sigma_{\mathrm{win}}$ low by about 1.4 % (0.986 / 0.985 / 0.986 for LPR / LWR / LER).
- **First-order deficit.** At 32 realisations, ensemble / model is 0.99 to 1.00 up to $|Q_x| \approx 0.16\ \mathrm{nm^{-1}}$ and 0.95 to 0.97 near the cut. It falls to 0.89 to 0.90 at $|Q_x| = 0.36\ \mathrm{nm^{-1}}$ ($|Q_x|\sigma_e \approx 0.44$), as the nulls fill and the peaks fall short. `first_order_deficit.json` holds the small-size curve, which is noisy.
- **Full-size study** (`run_study.py`, 1674 s, single process, all cases × 280 / 285 / 288 eV, $\omega$ −20° to 20°).
  - $\sigma_{\mathrm{win}}$ / realised truth is 0.986 to 0.996 for every case, mode and energy (tapered 0.988 to 0.990).
  - Median recovered / realised $S$ is 0.973 to 1.004.
  - Multislice (`both`) gives the same numbers within 0.5 %. MS / (Born × $A$) is 1.0005 at 280 eV, 1.0005 to 1.0012 at 285 eV and 1.0028 to 1.0035 at 288 eV over all $\omega$.
- **Isolation systematic.** At 32 realisations (4 seeds), isolation vs global $\chi^2/\mathrm{dof}$ is 3.4 for `both` and `independent` (1.4 for `lpr`, 1.0 for `lwr`). This is a few-percent row-vs-global offset from null filling at the isolation rows, hidden by the noise of small ensembles.
- **Flux obliquity.** Detector flux carries $1/(\cos W\cos\theta_{\mathrm{out}})$. `RoughEnsembleResult` applies the per-mode factor $\cos W/\cos\theta_{\mathrm{out}}$ (`flux_factor`, `diffuse_flux`, `order_flux_efficiency`). The 2D engines (`simulate_born_2d`, `simulate_multislice_2d`) do not apply it to their order efficiencies.
- **Low-count weighting.** At the default flux, diffuse pixels hold a few photons. Per-pixel weights from the observed counts biased the PSDs about 20 % low. The fits now use model-based IRLS variances, and the maps use exposure weighting.
- **Tilt sense.** The simulation's $\omega$ is opposite to the reduction geometry's: a frame rendered from engine tilt $\omega$ sits at $-\omega - \omega_0$ (`forward/ensemble_frames.frame_omega_deg`).
