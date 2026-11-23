"""High-level `CdRsoxsLoader` class for end-to-end CD-RSoXS / CD-SAXS reduction and
mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from cd_reduce.reduction.calibration import (
    CalibrationResult,
    build_order_table,
    fit_geometry,
    fit_geometry_with_walkoff,
    initial_geometry,
    refine_rod_verticality,
)
from cd_reduce.reduction.corrections import DEFAULT_ADU_PER_PHOTON, build_beamstop_mask
from cd_reduce.reduction.frames import Frame, load_frames
from cd_reduce.reduction.geometry import (
    ApproximationMode,
    Geometry,
    OmegaRefinement,
)
from cd_reduce.reduction.mapping import (
    BackgroundMode,
    QMap2D,
    Trace1D,
    extract_bragg_rod,
    extract_inplane_linecut,
    extract_order_rods_qz,
    extract_qx_by_omega,
    map_2d,
    reduce_1d,
)
from cd_reduce.reduction.roughness import (
    CrossSection,
    PsdRecord,
    ReducedPixels,
    fit_psds,
    order_scale,
    reduce_frame,
    sigma_window,
)
from cd_reduce.reduction.roughness.record import cross_section_dict
from cd_reduce.reduction.spots import extract_frames_spots
from cd_reduce.utils.attributes import process_vars_properties

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


@process_vars_properties
class CdRsoxsLoader:
    """End-to-end processor and Q-space mapper for CD-RSoXS / CD-SAXS frame series.

    Follows the property-driven design of `pxr-reduce`: modifying any processing
    attribute (`approximation`, `bg_mode`, `pitch_nm`, `membrane_thickness_nm`,
    `solid_angle_correction`, `exposure_normalization`, `polarization_correction`,
    `bin_px`) automatically updates `self.process_vars` and marks `reprocess_vars=True`.
    """

    if TYPE_CHECKING:
        pitch_nm: float
        approximation: ApproximationMode
        bg_mode: BackgroundMode
        bin_px: int
        exposure_normalization: bool
        solid_angle_correction: bool
        polarization_correction: bool
        apply_membrane: bool
        membrane_thickness_nm: float
        walkoff_params: tuple[float, float] | None
        mask_parasitic: bool
        auto_beamstop_mask: bool
        reprocess_vars: bool

    process_vars_defaults: ClassVar[dict[str, Any]] = {
        "pitch_nm": 100.0,
        "approximation": "correct",
        "bg_mode": "corner",
        "bin_px": 1,
        "exposure_normalization": True,
        "solid_angle_correction": True,
        "polarization_correction": False,
        "apply_membrane": False,
        "membrane_thickness_nm": 100.0,
        "walkoff_params": None,
        "mask_parasitic": False,
        "auto_beamstop_mask": False,
        "reprocess_vars": False,
    }

    def __init__(
        self,
        source: str | Path | Sequence[Frame] | None = None,
        *,
        frames: Sequence[Frame] | None = None,
        geometry: Geometry | Mapping[str, Any] | str | Path | None = None,
        omega_refinement: OmegaRefinement | Mapping[str, Any] | None = None,
        pitch_nm: float = 100.0,
        approximation: ApproximationMode = "correct",
        bg_mode: BackgroundMode = "corner",
        bin_px: int = 1,
        exposure_normalization: bool = True,
        solid_angle_correction: bool = True,
        polarization_correction: bool = False,
        apply_membrane: bool = False,
        membrane_thickness_nm: float = 100.0,
        walkoff_params: tuple[float, float] | None = None,
        mask_parasitic: bool = False,
        auto_beamstop_mask: bool = False,
        deduplicate_omega: bool = True,
        energy_ev: float | None = None,
    ) -> None:
        if frames is not None:
            self.frames: list[Frame] = list(frames)
        elif source is not None:
            self.frames = load_frames(
                source,
                deduplicate_omega=deduplicate_omega,
                energy_ev=energy_ev,
            )
        else:
            self.frames = []

        if isinstance(geometry, (str, Path)):
            data = json.loads(Path(geometry).read_text(encoding="utf-8"))
            self._geometry: Geometry | None = Geometry.from_dict(data)
        elif isinstance(geometry, Geometry):
            self._geometry = geometry
        elif geometry is not None:
            self._geometry = Geometry.from_dict(geometry)
        else:
            self._geometry = None

        if isinstance(omega_refinement, OmegaRefinement):
            self._omega_refinement: OmegaRefinement | None = omega_refinement
        elif omega_refinement is not None:
            self._omega_refinement = OmegaRefinement.from_dict(omega_refinement)
        else:
            self._omega_refinement = None
        self._spots_df: pd.DataFrame | None = None
        self._order_table_df: pd.DataFrame | None = None
        self._last_qmap: QMap2D | None = None

        self.process_vars: dict[str, Any] = {
            "pitch_nm": float(pitch_nm),
            "approximation": str(approximation),
            "bg_mode": str(bg_mode),
            "bin_px": int(bin_px),
            "exposure_normalization": bool(exposure_normalization),
            "solid_angle_correction": bool(solid_angle_correction),
            "polarization_correction": bool(polarization_correction),
            "apply_membrane": bool(apply_membrane),
            "membrane_thickness_nm": float(membrane_thickness_nm),
            "walkoff_params": walkoff_params,
            "mask_parasitic": bool(mask_parasitic),
            "auto_beamstop_mask": bool(auto_beamstop_mask),
            "reprocess_vars": False,
        }

    def _invalidate_cache(self) -> None:
        """Invalidate cached Q-maps when processing parameters change."""
        self._last_qmap = None
        self.process_vars["reprocess_vars"] = True

    @property
    def geometry(self) -> Geometry:
        """Active detector and sample `Geometry`."""
        if self._geometry is None:
            msg = (
                "Geometry is not set. Pass `geometry=...` on initialization or call "
                "`calibrate_geometry()` first."
            )
            raise RuntimeError(msg)
        return self._geometry

    @geometry.setter
    def geometry(self, value: Geometry | None) -> None:
        self._geometry = value
        self._invalidate_cache()

    @property
    def omega_refinement(self) -> OmegaRefinement | None:
        """Active `OmegaRefinement` for Bragg rod verticality."""
        return self._omega_refinement

    @omega_refinement.setter
    def omega_refinement(self, value: OmegaRefinement | None) -> None:
        self._omega_refinement = value
        self._invalidate_cache()

    def extract_spots(self, *, force: bool = False, **kwargs: Any) -> pd.DataFrame:
        """Detect and integrate diffraction spots across all loaded frames."""
        if self._spots_df is None or force:
            self._spots_df = extract_frames_spots(self.frames, **kwargs)
        return self._spots_df

    def calibrate_geometry(
        self,
        *,
        start: Geometry | None = None,
        Lb_px: float = 1969.0,
        pitch_nm: float | None = None,
        fit_walkoff: bool = False,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Fit detector and sample geometry from extracted diffraction spots."""
        if pitch_nm is not None:
            self.pitch_nm = float(pitch_nm)
        spots = self.extract_spots()
        if start is None:
            start = (
                self._geometry
                if self._geometry is not None
                else initial_geometry(spots, Lb_px=Lb_px, Lambda_nm=self.pitch_nm)
            )
        if fit_walkoff:
            result = fit_geometry_with_walkoff(spots, start, **kwargs)
        else:
            result = fit_geometry(spots, start, **kwargs)
        self._geometry = result.geometry
        self._order_table_df = result.spots
        self._invalidate_cache()
        return result

    def build_order_table(self, **kwargs: Any) -> pd.DataFrame:
        """Index extracted spots by integer diffraction order `m`."""
        spots = self.extract_spots()
        self._order_table_df = build_order_table(
            spots,
            self.geometry,
            omega_refinement=self._omega_refinement,
            apply_membrane=self.apply_membrane,
            membrane_thickness_nm=self.membrane_thickness_nm,
            **kwargs,
        )
        return self._order_table_df

    def refine_omega(self, mount_name: str = "default") -> OmegaRefinement:
        """Refine rod verticality (`OmegaRefinement`) for the current mount."""
        spots = self.extract_spots()
        ref_map = refine_rod_verticality(
            {mount_name: spots},
            {mount_name: self.geometry},
        )
        self._omega_refinement = ref_map[mount_name]
        self._invalidate_cache()
        return self._omega_refinement

    def to_qmap_2d(
        self,
        x_domain: str = "Qx",
        y_domain: str = "Qz",
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        bins: int | tuple[int, int] = (300, 300),
        approximation: ApproximationMode | None = None,
        bg_mode: BackgroundMode | None = None,
        fixed_domain: str | None = None,
        fixed_value: float | None = None,
        fixed_tol: float | None = None,
        fixed_range: tuple[float, float] | None = None,
        qx_cut: tuple[float, float] | None = None,
        qy_cut: tuple[float, float] | None = None,
        qz_cut: tuple[float, float] | None = None,
        qr_cut: tuple[float, float] | None = None,
        omega_cut: tuple[float, float] | None = None,
        qy_cut_over_g: float | None = None,
        cuts: Mapping[str, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> QMap2D:
        """Convert loaded frames into a 2D reciprocal-space map `QMap2D`."""
        approx: ApproximationMode = (
            approximation if approximation is not None else self.approximation
        )
        bg: BackgroundMode = bg_mode if bg_mode is not None else self.bg_mode
        bins_2d: tuple[int, int] = (bins, bins) if isinstance(bins, int) else bins
        qmap = map_2d(
            self.frames,
            self.geometry,
            x_domain=x_domain,
            y_domain=y_domain,
            x_range=x_range,
            y_range=y_range,
            bins=bins_2d,
            bin_px=self.bin_px,
            approximation=approx,
            bg_mode=bg,
            omega_refinement=self._omega_refinement,
            solid_angle_correction=self.solid_angle_correction,
            exposure_normalization=self.exposure_normalization,
            polarization_correction=self.polarization_correction,
            apply_membrane=self.apply_membrane,
            membrane_thickness_nm=self.membrane_thickness_nm,
            walkoff_params=self.walkoff_params,
            mask_parasitic=self.mask_parasitic,
            auto_beamstop_mask=self.auto_beamstop_mask,
            fixed_domain=fixed_domain,
            fixed_value=fixed_value,
            fixed_tol=fixed_tol,
            fixed_range=fixed_range,
            qx_cut=qx_cut,
            qy_cut=qy_cut,
            qz_cut=qz_cut,
            qr_cut=qr_cut,
            omega_cut=omega_cut,
            qy_cut_over_g=qy_cut_over_g,
            cuts=cuts,
            **kwargs,
        )
        self._last_qmap = qmap
        self.process_vars["reprocess_vars"] = False
        return qmap

    def to_trace_1d(
        self,
        domain: str = "Qx",
        *,
        domain_range: tuple[float, float] | None = None,
        bins: int = 250,
        approximation: ApproximationMode | None = None,
        bg_mode: BackgroundMode | None = None,
        fixed_domain: str | None = None,
        fixed_value: float | None = None,
        fixed_tol: float | None = None,
        fixed_range: tuple[float, float] | None = None,
        qx_cut: tuple[float, float] | None = None,
        qy_cut: tuple[float, float] | None = None,
        qz_cut: tuple[float, float] | None = None,
        qr_cut: tuple[float, float] | None = None,
        omega_cut: tuple[float, float] | None = None,
        qy_cut_over_g: float | None = None,
        cuts: Mapping[str, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> Trace1D:
        """Reduce loaded frames into a 1D trace `Trace1D` vs any Q/angle domain."""
        approx: ApproximationMode = (
            approximation if approximation is not None else self.approximation
        )
        bg: BackgroundMode = bg_mode if bg_mode is not None else self.bg_mode
        return reduce_1d(
            self.frames,
            self.geometry,
            domain=domain,
            domain_range=domain_range,
            bins=bins,
            bin_px=self.bin_px,
            approximation=approx,
            bg_mode=bg,
            omega_refinement=self._omega_refinement,
            solid_angle_correction=self.solid_angle_correction,
            exposure_normalization=self.exposure_normalization,
            polarization_correction=self.polarization_correction,
            apply_membrane=self.apply_membrane,
            membrane_thickness_nm=self.membrane_thickness_nm,
            walkoff_params=self.walkoff_params,
            fixed_domain=fixed_domain,
            fixed_value=fixed_value,
            fixed_tol=fixed_tol,
            fixed_range=fixed_range,
            qx_cut=qx_cut,
            qy_cut=qy_cut,
            qz_cut=qz_cut,
            qr_cut=qr_cut,
            omega_cut=omega_cut,
            qy_cut_over_g=qy_cut_over_g,
            cuts=cuts,
            **kwargs,
        )

    def extract_bragg_rod(
        self,
        order: int,
        *,
        domain: Literal["Qz", "omega"] = "Qz",
        qx_half_width_g: float = 0.25,
        qy_cut_over_g: float = 0.5,
        bins: int = 120,
        approximation: ApproximationMode | None = None,
        bg_mode: BackgroundMode | None = None,
    ) -> Trace1D:
        """Extract the 1D Bragg rod `I_m(Qz)` (or `I_m(omega)`) for diffraction order
        `m`.
        """
        approx: ApproximationMode = (
            approximation if approximation is not None else self.approximation
        )
        bg: BackgroundMode = bg_mode if bg_mode is not None else self.bg_mode
        return extract_bragg_rod(
            self.frames,
            self.geometry,
            order=order,
            domain=domain,
            qx_half_width_g=qx_half_width_g,
            qy_cut_over_g=qy_cut_over_g,
            bins=bins,
            approximation=approx,
            bg_mode=bg,
        )

    def extract_order_rods_qz(
        self,
        *,
        orders: Sequence[int] = (-4, -3, -2, -1, 1, 2, 3, 4),
        use_order_table: bool = False,
        normalize_first_order: bool = False,
        qx_half_width_g: float = 0.12,
        qy_cut_over_g: float = 0.10,
        bins: int = 100,
    ) -> dict[int, Trace1D]:
        """Extract `I_m(Qz)` Bragg truncation rods for all requested orders `m`."""
        if use_order_table:
            tbl = self._order_table_df
            if tbl is None or "I_cps" not in tbl.columns:
                tbl = self.build_order_table()
            return extract_order_rods_qz(
                tbl,
                self.geometry,
                orders=orders,
                normalize_first_order=normalize_first_order,
            )
        return extract_order_rods_qz(
            self.frames,
            self.geometry,
            orders=orders,
            qx_half_width_g=qx_half_width_g,
            qy_cut_over_g=qy_cut_over_g,
            bins=bins,
            approximation=self.approximation,
            bg_mode=self.bg_mode,
        )

    def extract_qx_by_omega(
        self,
        *,
        omegas: Sequence[float] | None = None,
        use_order_table: bool = False,
        omega_tol_deg: float = 0.4,
        qy_cut_over_g: float = 0.10,
        bins: int = 300,
    ) -> dict[float, Trace1D]:
        """Extract `I(Qx)` profiles for each measured sample tilt angle `omega`."""
        if use_order_table:
            tbl = self._order_table_df
            if tbl is None or "I_cps" not in tbl.columns:
                tbl = self.build_order_table()
            return extract_qx_by_omega(
                tbl,
                self.geometry,
                omegas=omegas,
                omega_tol_deg=omega_tol_deg,
            )
        return extract_qx_by_omega(
            self.frames,
            self.geometry,
            omegas=omegas,
            omega_tol_deg=omega_tol_deg,
            qy_cut_over_g=qy_cut_over_g,
            bins=bins,
            approximation=self.approximation,
            bg_mode=self.bg_mode,
        )

    def extract_inplane_linecut(
        self,
        *,
        domain: Literal["Qx", "Qr", "signed_Qr"] = "Qx",
        fixed_qz: float | None = None,
        qz_tol: float = 0.02,
        fixed_omega: float | None = None,
        omega_tol: float = 0.6,
        qy_cut_over_g: float = 0.5,
        bins: int = 300,
        approximation: ApproximationMode | None = None,
        bg_mode: BackgroundMode | None = None,
    ) -> Trace1D:
        """Extract an in-plane 1D linecut (`Qx` or `Qr`) at fixed `Qz` or fixed
        `omega`.
        """
        approx: ApproximationMode = (
            approximation if approximation is not None else self.approximation
        )
        bg: BackgroundMode = bg_mode if bg_mode is not None else self.bg_mode
        return extract_inplane_linecut(
            self.frames,
            self.geometry,
            domain=domain,
            fixed_qz=fixed_qz,
            qz_tol=qz_tol,
            fixed_omega=fixed_omega,
            omega_tol=omega_tol,
            qy_cut_over_g=qy_cut_over_g,
            bins=bins,
            approximation=approx,
            bg_mode=bg,
        )

    def extract_profiles_by_omega(
        self,
        domain: str = "Qx",
        *,
        use_orders: bool = True,
        **kwargs: Any,
    ) -> dict[float, Trace1D]:
        """``I`` vs ``Qx`` / ``Qr`` / ``signed_Qr`` per incidence angle.

        ``use_orders=True`` uses the indexed order table (discrete Bragg peaks);
        otherwise continuous detector-line profiles from the frames.
        """
        src: Any = self.build_order_table() if use_orders else self.frames
        return extract_qx_by_omega(
            src,
            self.geometry,
            domain=domain,
            approximation=self.approximation,
            bg_mode=self.bg_mode,
            **kwargs,
        )

    def reduce_roughness(
        self,
        xs_by_energy: Mapping[float, CrossSection],
        *,
        scale: float | Literal["bragg_orders"] = "bragg_orders",
        window_T: float = 1.0,
        R_B: float = 0.008,
        keep: float = 0.1,
        adu_per_photon: float = DEFAULT_ADU_PER_PHOTON,
        read_noise_photons: float = 0.0,
        gamma_zero_halfwidth: float = 0.01,
    ) -> ReducedPixels:
        """Reduce every frame to per-pixel ``M`` (HOWTO reduction).

        ``scale`` is the incident photon rate: ``"bragg_orders"`` fits it per energy
        from the indexed order table (:func:`order_scale`); a float is used for all
        frames. Cross sections are matched to frame energies within 1e-6 eV
        (``KeyError`` otherwise). ``meta["scale_source"]`` records the choice and
        ``meta["scale_by_energy"]`` / ``meta["scale_err_by_energy"]`` the values.
        """
        if scale == "bragg_orders":
            fit = order_scale(
                self.build_order_table(),
                xs_by_energy,
                adu_per_photon=adu_per_photon,
                window_T=window_T,
            )
            by_e, err_e = dict(fit.scale_by_energy), dict(fit.scale_err)
            source = "bragg_orders"
        else:
            by_e = err_e = None
            source = "known"
        mask = (
            build_beamstop_mask(self.frames, self.geometry)
            if self.auto_beamstop_mask
            else None
        )

        def match(table: Mapping[float, Any], e: float) -> Any:
            key = min(table, key=lambda k: abs(k - e))
            if abs(key - e) > 1e-6:
                msg = f"no entry for energy {e} eV"
                raise KeyError(msg)
            return table[key]

        items = []
        for i, fr in enumerate(self.frames):
            xs = match(xs_by_energy, fr.energy_ev)
            sc = float(scale) if by_e is None else float(match(by_e, fr.energy_ev))
            items.append(
                reduce_frame(
                    fr,
                    self.geometry,
                    xs,
                    scale=sc,
                    window_T=window_T,
                    R_B=R_B,
                    keep=keep,
                    gamma_zero_halfwidth=gamma_zero_halfwidth,
                    adu_per_photon=adu_per_photon,
                    read_noise_photons=read_noise_photons,
                    bg_mode=self.bg_mode,
                    beamstop_mask=mask,
                    omega_refinement=self._omega_refinement,
                    frame_id=i,
                )
            )
        if not items:
            msg = "no frames loaded"
            raise ValueError(msg)
        out = ReducedPixels.concat(items)
        out.meta["scale_source"] = source
        if by_e is not None and err_e is not None:
            out.meta["scale_by_energy"] = by_e
            out.meta["scale_err_by_energy"] = err_e
        else:
            out.meta["scale_by_energy"] = {
                fr.energy_ev: float(scale) for fr in self.frames
            }
            out.meta["scale_err_by_energy"] = {fr.energy_ev: 0.0 for fr in self.frames}
        return out

    def extract_psds(
        self,
        xs_by_energy: Mapping[float, CrossSection],
        *,
        w: float | None = None,
        f_edges: np.ndarray | None = None,
        cross_term: bool = False,
        min_pixels: int = 10,
        max_cond: float = 1e8,
        qx_max: float | None = None,
        **reduce_kwargs: Any,
    ) -> PsdRecord:
        """LER / LWR / LPR PSDs (two-sided, nm^3, vs ``f = Qy/2pi``) as a `PsdRecord`.

        ``w`` defaults to the first cross section's ``w``. ``reduce_kwargs`` go to
        :meth:`reduce_roughness`. With several energies, ``record.scale`` and
        ``scale_err`` hold the first energy's value; the per-energy dicts are in
        ``record.source["scale_by_energy"]`` / ``["scale_err_by_energy"]``.
        """
        first = next(iter(xs_by_energy.values()))
        width = float(first.w if w is None else w)
        px = self.reduce_roughness(xs_by_energy, **reduce_kwargs)
        table = fit_psds(
            px,
            width,
            f_edges,
            cross_term=cross_term,
            min_pixels=min_pixels,
            max_cond=max_cond,
            qx_max=qx_max,
        )
        sigma_win = {
            k: sigma_window(
                table.f_center, getattr(table, f"S_{k}"), getattr(table, f"var_{k}")
            )
            for k in ("lpr", "lwr", "ler")
        }
        fin = table.f_edges[np.isfinite(table.f_edges)]
        by_e = px.meta["scale_by_energy"]
        err_e = px.meta["scale_err_by_energy"]
        e0 = next(iter(by_e))
        source = {
            "energies_ev": sorted({float(e) for e in px.energy_ev}),
            "omega_range_deg": [float(px.omega_deg.min()), float(px.omega_deg.max())],
            "R_B": px.meta.get("R_B"),
            "qz_mean": float(np.mean(px.Qz)),
            "w": width,
            "scale_by_energy": {float(k): float(v) for k, v in by_e.items()},
            "scale_err_by_energy": {float(k): float(v) for k, v in err_e.items()},
        }
        return PsdRecord(
            table=table,
            cross_section=cross_section_dict(first),
            scale=float(by_e[e0]),
            scale_err=float(err_e[e0]),
            scale_source=px.meta["scale_source"],
            source=source,
            window=(float(fin[0]), float(fin[-1])),
            sigma_win=sigma_win,
        )


__all__ = ["CdRsoxsLoader"]
