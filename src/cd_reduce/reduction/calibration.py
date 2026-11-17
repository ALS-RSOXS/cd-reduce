"""Model-free Laue-invariant geometry calibration and order indexing.

Implements:
1. Single-mount geometry calibration from Laue invariants ``Qx = m*G, Qy = 0``.
2. Per-frame walk-off offset calibration.
3. Joint multi-mount calibration sharing detector distance ``Lb_px`` across a
   beamtime.
4. Rod-verticality refinement making every diffraction rod vertical in
   ``(Qx, Qz)`` across ``omega``.
5. Order indexing and intensity catalog construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from cd_reduce.reduction.geometry import (
    ApproximationMode,
    Geometry,
    OmegaRefinement,
    compute_pixel_q,
)
from cd_reduce.utils.units import energy_to_wavelength_nm

#: Default fractional order tolerance ``|Qx / G - m| < INDEX_TOL`` for Bragg spots.
INDEX_TOL: float = 0.2


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of a single-mount or walk-off geometry calibration.

    Attributes
    ----------
    geometry : Geometry
        Calibrated geometry (including ``walkoff_offsets_px`` if fitted).
    sigma : dict of str to float
        One-sigma uncertainties on calibrated parameters.
    spots : DataFrame
        Spot table with ``Qx``, ``Qy``, ``Qz``, ``order``, ``index_residual``,
        ``is_bragg``, ``res_x_px``, ``res_y_px`` attached.
    rms_px : float
        Combined RMS residual (in approximate detector pixels) over Bragg spots.
    rms_x_px : float
        In-plane (along grating vector) RMS residual in pixels.
    rms_y_px : float
        Out-of-plane (along grating lines) RMS residual in pixels.
    n_bragg : int
        Number of indexed Bragg spots used in the fit.
    """

    geometry: Geometry
    sigma: dict[str, float]
    spots: pd.DataFrame
    rms_px: float
    rms_x_px: float
    rms_y_px: float
    n_bragg: int


@dataclass(frozen=True, slots=True)
class JointCalibrationResult:
    """Outcome of a joint multi-mount geometry calibration sharing ``Lb_px``.

    Attributes
    ----------
    geometries : dict of str to Geometry
        Calibrated geometry for each mount/sample key.
    shared_Lb_px : float
        Shared sample-to-detector distance in pixels.
    shared_Lb_err_px : float
        Uncertainty on ``shared_Lb_px``.
    per_mount_results : dict of str to CalibrationResult
        Detailed per-mount calibration summary.
    """

    geometries: dict[str, Geometry]
    shared_Lb_px: float
    shared_Lb_err_px: float
    per_mount_results: dict[str, CalibrationResult]


def initial_geometry(
    spots_df: pd.DataFrame,
    *,
    Lb_px: float = 1969.0,
    Lambda_nm: float | None = None,
    pixel_um: float = 27.0,
) -> Geometry:
    """Estimate starting geometry from the brightest spot pair near normal incidence.

    Parameters
    ----------
    spots_df : DataFrame
        Spot table from ``extract_frames_spots``.
    Lb_px : float, optional
        Initial sample-to-detector distance in pixels. Default is ``1969.0``.
    Lambda_nm : float or None, optional
        Initial grating period in nm. If None, estimated from the brightest
        spot pair separation at ``Lb_px``.
    pixel_um : float, optional
        Detector pixel pitch in micrometres. Default is ``27.0``.

    Returns
    -------
    Geometry
        Initial geometry estimate.
    """
    if spots_df.empty:
        return Geometry(Lb_px=Lb_px, Lambda_nm=Lambda_nm or 80.0, pixel_um=pixel_um)

    omega = (
        spots_df["omega_deg"].to_numpy(dtype=np.float64)
        if "omega_deg" in spots_df.columns
        else spots_df["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
    )
    near_normal = np.abs(omega) <= 5.0
    sub = spots_df[near_normal] if np.any(near_normal) else spots_df

    flux_col = "I_adu" if "I_adu" in sub.columns else "score"
    mids: list[NDArray[np.float64]] = []
    g_ests: list[float] = []
    rots: list[float] = []

    for _, grp in sub.groupby("file"):
        top = grp.nlargest(2, flux_col)
        if len(top) < 2:
            continue
        p0 = top[["x", "y"]].iloc[0].to_numpy(dtype=np.float64)
        p1 = top[["x", "y"]].iloc[1].to_numpy(dtype=np.float64)
        mids.append(0.5 * (p0 + p1))
        diff = p1 - p0
        if diff[0] < 0:
            diff = -diff
        rots.append(float(np.degrees(np.arctan2(diff[1], diff[0]))))
        half_dist = 0.5 * float(np.hypot(diff[0], diff[1]))
        two_theta = np.arctan2(half_dist, Lb_px)
        lam = float(energy_to_wavelength_nm(float(top["energy_ev"].iloc[0])))
        g_val = 2.0 * np.pi * np.sin(two_theta) / lam
        if g_val > 0:
            g_ests.append(g_val)

    if mids:
        centre = np.median(np.asarray(mids), axis=0)
        u0, v0 = float(centre[0]), float(centre[1])
    else:
        u0 = float(spots_df["x"].median())
        v0 = float(spots_df["y"].median())

    if Lambda_nm is not None:
        lam_nm = float(Lambda_nm)
    elif g_ests:
        lam_nm = float(2.0 * np.pi / np.median(g_ests))
    else:
        lam_nm = 80.0

    rot_deg = float(np.median(rots)) if rots else 0.0
    return Geometry(
        u0_px=u0,
        v0_px=v0,
        Lb_px=float(Lb_px),
        line_rot_deg=rot_deg,
        Lambda_nm=lam_nm,
        pixel_um=pixel_um,
    )


def index_spots(
    geometry: Geometry,
    spots_df: pd.DataFrame,
    *,
    index_tol: float = INDEX_TOL,
    approximation: ApproximationMode = "correct",
    ref_Lb_px: float | None = None,
    ref_G_inv_nm: float | None = None,
) -> pd.DataFrame:
    """Assign diffraction order indices ``m = round(Qx / G)`` and attach residuals.

    Parameters
    ----------
    geometry : Geometry
        Detector and sample geometry.
    spots_df : DataFrame
        Spot table with ``x``, ``y``, ``energy_ev``, and ``omega_deg`` (or
        ``sample_theta_deg``).
    index_tol : float, optional
        Maximum ``|Qx / G - m|`` for a spot to be flagged ``is_bragg``.
        Default is ``0.2``.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.
    ref_Lb_px, ref_G_inv_nm : float, optional
        Reference distance and reciprocal lattice vector used to convert
        fractional index residuals into approximate detector pixels.

    Returns
    -------
    DataFrame
        Copy of ``spots_df`` with ``Qx``, ``Qy``, ``Qz``, ``Qr``, ``Q``,
        ``order``, ``index_residual``, ``res_x_px``, ``res_y_px``, ``is_bragg``.
    """
    if spots_df.empty:
        return spots_df.copy()

    omega = (
        spots_df["omega_deg"].to_numpy(dtype=np.float64)
        if "omega_deg" in spots_df.columns
        else spots_df["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
    )
    walkoff = geometry.walkoff_offsets_px or {}
    if "file" in spots_df.columns and walkoff:
        off = spots_df["file"].map(walkoff).fillna(0.0).to_numpy(dtype=np.float64)
    else:
        off = np.zeros(len(spots_df), dtype=np.float64)

    rot = np.radians(geometry.line_rot_deg)
    x_eff = spots_df["x"].to_numpy(dtype=np.float64) + off * np.cos(rot)
    y_eff = spots_df["y"].to_numpy(dtype=np.float64) + off * np.sin(rot)
    E = spots_df["energy_ev"].to_numpy(dtype=np.float64)

    qmap = compute_pixel_q(
        geometry, x_eff, y_eff, E, omega, approximation=approximation
    )
    G = geometry.g_inv_nm
    ratio = qmap.Qx / G
    order = np.rint(ratio).astype(int)
    frac = ratio - order

    k = 2.0 * np.pi / np.asarray(energy_to_wavelength_nm(E), dtype=np.float64)
    lb0 = ref_Lb_px if ref_Lb_px is not None else geometry.Lb_px
    g0 = ref_G_inv_nm if ref_G_inv_nm is not None else G
    sc = lb0 * g0 / k

    out = spots_df.copy()
    out["Qx"] = qmap.Qx
    out["Qy"] = qmap.Qy
    out["Qz"] = qmap.Qz
    out["Qr"] = qmap.Qr
    out["signed_Qr"] = qmap.signed_Qr
    out["Q"] = qmap.Q
    out["two_theta_deg"] = qmap.two_theta_deg
    out["omega_true_deg"] = omega + geometry.omega0_deg
    out["order"] = order
    out["index_residual"] = frac
    out["res_x_px"] = frac * sc
    out["res_y_px"] = (qmap.Qy / G) * sc
    out["is_bragg"] = (np.abs(frac) < index_tol) & (order != 0)
    return out


def _pack_geom_vec(g: Geometry) -> NDArray[np.float64]:
    """Pack geometry into the 8-element vector ``[u0, v0, Lb, beta, rot, om0, psi,
    logLam]``.
    """
    return np.array(
        [
            g.u0_px,
            g.v0_px,
            g.Lb_px,
            np.radians(g.beta_deg),
            np.radians(g.line_rot_deg),
            np.radians(g.omega0_deg),
            np.radians(g.psi_deg),
            np.log(g.Lambda_nm),
        ],
        dtype=np.float64,
    )


def _unpack_geom_vec(
    p: NDArray[np.float64],
    base: Geometry,
    walkoff: dict[str, float] | None = None,
) -> Geometry:
    """Unpack 8-element vector into a ``Geometry`` instance."""
    return replace(
        base,
        u0_px=float(p[0]),
        v0_px=float(p[1]),
        Lb_px=float(p[2]),
        beta_deg=float(np.degrees(p[3])),
        line_rot_deg=float(np.degrees(p[4])),
        omega0_deg=float(np.degrees(p[5])),
        psi_deg=float(np.degrees(p[6])),
        Lambda_nm=float(np.exp(p[7])),
        walkoff_offsets_px=walkoff if walkoff is not None else base.walkoff_offsets_px,
    )


def fit_geometry(
    spots_df: pd.DataFrame,
    start: Geometry | None = None,
    *,
    fix_lambda: float | None = None,
    fix_Lb: float | None = None,
    free_beta: bool = True,
    n_iter: int = 5,
    index_tol: float = INDEX_TOL,
    approximation: ApproximationMode = "correct",
    loss: str = "soft_l1",
    f_scale_px: float = 2.0,
) -> CalibrationResult:
    """Calibrate detector and sample geometry from Laue invariants ``Qx = m*G, Qy = 0``.

    Alternates order assignment ``m = round(Qx / G)`` and robust non-linear
    least squares on fractional-index residuals ``(Qx / G - m, Qy / G)`` scaled
    to approximate detector pixels.

    Parameters
    ----------
    spots_df : DataFrame
        Spot table from ``extract_frames_spots``.
    start : Geometry, optional
        Initial geometry guess. Inferred automatically via ``initial_geometry``
        if omitted.
    fix_lambda : float or None, optional
        If provided, fix ``Lambda_nm`` to this value during calibration.
    fix_Lb : float or None, optional
        If provided, fix ``Lb_px`` to this value during calibration.
    free_beta : bool, optional
        Whether detector tilt ``beta_deg`` is varied. Default is True.
    n_iter : int, optional
        Number of alternating index/refine cycles. Default is 5.
    index_tol : float, optional
        Bragg order indexing tolerance. Default is ``0.2``.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.
    loss : str, optional
        Robust loss function for ``scipy.optimize.least_squares``. Default is
        ``"soft_l1"``.
    f_scale_px : float, optional
        Soft-L1 scale in pixels. Default is ``2.0``.

    Returns
    -------
    CalibrationResult
        Best-fit geometry, parameter uncertainties, and indexed spots.
    """
    base = start if start is not None else initial_geometry(spots_df)
    if fix_lambda is not None:
        base = replace(base, Lambda_nm=float(fix_lambda))
    if fix_Lb is not None:
        base = replace(base, Lb_px=float(fix_Lb))

    p0 = _pack_geom_vec(base)
    free = [
        i
        for i in range(8)
        if not (i == 2 and fix_Lb is not None)
        and not (i == 3 and (not free_beta or approximation in {"angle_linear", "lin"}))
        and not (i == 7 and fix_lambda is not None)
    ]

    x = spots_df["x"].to_numpy(dtype=np.float64)
    y = spots_df["y"].to_numpy(dtype=np.float64)
    E = spots_df["energy_ev"].to_numpy(dtype=np.float64)
    omega = (
        spots_df["omega_deg"].to_numpy(dtype=np.float64)
        if "omega_deg" in spots_df.columns
        else spots_df["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
    )
    k_all = 2.0 * np.pi / np.asarray(energy_to_wavelength_nm(E), dtype=np.float64)
    lb0 = base.Lb_px
    g0 = base.g_inv_nm

    def _full(pf: NDArray[np.float64]) -> NDArray[np.float64]:
        p = p0.copy()
        p[free] = pf
        return p

    sol = None
    for _ in range(n_iter):
        g_cur = _unpack_geom_vec(p0, base)
        qmap = compute_pixel_q(g_cur, x, y, E, omega, approximation=approximation)
        G = g_cur.g_inv_nm
        m = np.rint(qmap.Qx / G)
        ok = (np.abs(qmap.Qx / G - m) < index_tol) & (m != 0)
        if ok.sum() < len(free):
            ok = m != 0

        def _res(
            pf: NDArray[np.float64],
            mask_ok: NDArray[np.bool_] = ok,
            m_idx: NDArray[np.float64] = m,
        ) -> NDArray[np.float64]:
            g_try = _unpack_geom_vec(_full(pf), base)
            qm = compute_pixel_q(
                g_try,
                x[mask_ok],
                y[mask_ok],
                E[mask_ok],
                omega[mask_ok],
                approximation=approximation,
            )
            G_try = g_try.g_inv_nm
            sc = lb0 * g0 / k_all[mask_ok]
            return np.concatenate(
                [(qm.Qx / G_try - m_idx[mask_ok]) * sc, (qm.Qy / G_try) * sc]
            )

        sol = least_squares(
            _res, p0[free], loss=loss, f_scale=f_scale_px, max_nfev=4000
        )
        p0 = _full(sol.x)

    assert sol is not None
    final_geom = _unpack_geom_vec(p0, base)
    indexed = index_spots(
        final_geom,
        spots_df,
        index_tol=index_tol,
        approximation=approximation,
        ref_Lb_px=lb0,
        ref_G_inv_nm=g0,
    )
    bragg = indexed["is_bragg"].to_numpy(dtype=bool)
    rx = indexed["res_x_px"].to_numpy(dtype=np.float64)[bragg]
    ry = indexed["res_y_px"].to_numpy(dtype=np.float64)[bragg]

    jac = sol.jac
    dof = max(len(sol.fun) - len(free), 1)
    s2 = float(np.sum(sol.fun**2) / dof)
    cov = np.linalg.pinv(jac.T @ jac) * s2
    sig_vec = np.zeros(8, dtype=np.float64)
    sig_vec[free] = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    sigma_dict = {
        "u0_px": float(sig_vec[0]),
        "v0_px": float(sig_vec[1]),
        "Lb_px": float(sig_vec[2]),
        "beta_deg": float(np.degrees(sig_vec[3])),
        "line_rot_deg": float(np.degrees(sig_vec[4])),
        "omega0_deg": float(np.degrees(sig_vec[5])),
        "psi_deg": float(np.degrees(sig_vec[6])),
        "Lambda_nm": float(final_geom.Lambda_nm * sig_vec[7]),
    }
    rms_x = float(np.sqrt(np.mean(rx**2))) if rx.size else 0.0
    rms_y = float(np.sqrt(np.mean(ry**2))) if ry.size else 0.0
    rms_tot = float(np.sqrt(0.5 * (rms_x**2 + rms_y**2))) if rx.size else 0.0
    return CalibrationResult(
        geometry=final_geom,
        sigma=sigma_dict,
        spots=indexed,
        rms_px=rms_tot,
        rms_x_px=rms_x,
        rms_y_px=rms_y,
        n_bragg=int(bragg.sum()),
    )


def fit_geometry_with_walkoff(
    spots_df: pd.DataFrame,
    start: Geometry | None = None,
    *,
    fix_lambda: float | None = None,
    n_iter: int = 4,
    index_tol: float = INDEX_TOL,
    approximation: ApproximationMode = "correct",
    loss: str = "soft_l1",
    f_scale_px: float = 2.0,
) -> CalibrationResult:
    """Calibrate geometry jointly with per-frame walk-off offsets ``o_f``.

    Each frame ``f`` with ``|omega| >= 0.5 deg`` carries a free translation
    ``o_f`` (in pixels) along the spot line; ``|omega| < 0.5 deg`` frames are
    anchored at ``o_f = 0``.

    Parameters
    ----------
    spots_df : DataFrame
        Spot table with ``file``, ``x``, ``y``, ``energy_ev``, ``omega_deg``.
    start : Geometry, optional
        Initial geometry estimate.
    fix_lambda : float or None, optional
        If provided, hold ``Lambda_nm`` fixed.
    n_iter : int, optional
        Number of alternating index/refine passes. Default is 4.
    index_tol : float, optional
        Bragg order indexing tolerance. Default is ``0.2``.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.
    loss : str, optional
        Robust loss name. Default is ``"soft_l1"``.
    f_scale_px : float, optional
        Loss scale in pixels. Default is ``2.0``.

    Returns
    -------
    CalibrationResult
        Calibrated geometry with ``walkoff_offsets_px`` populated.
    """
    base = (
        start
        if start is not None
        else fit_geometry(spots_df, fix_lambda=fix_lambda).geometry
    )
    if fix_lambda is not None:
        base = replace(base, Lambda_nm=float(fix_lambda))

    x0 = spots_df["x"].to_numpy(dtype=np.float64)
    y0 = spots_df["y"].to_numpy(dtype=np.float64)
    E = spots_df["energy_ev"].to_numpy(dtype=np.float64)
    omega = (
        spots_df["omega_deg"].to_numpy(dtype=np.float64)
        if "omega_deg" in spots_df.columns
        else spots_df["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
    )
    k_all = 2.0 * np.pi / np.asarray(energy_to_wavelength_nm(E), dtype=np.float64)

    u_files = sorted(set(spots_df["file"].astype(str)))
    fi = np.array([u_files.index(str(f)) for f in spots_df["file"]], dtype=int)
    ref_mask = np.array(
        [abs(omega[fi == i][0]) < 0.5 for i in range(len(u_files))], dtype=bool
    )
    free_f = np.where(~ref_mask)[0]

    g_free = [0, 1, 2, 3, 4, 5, 6] + ([] if fix_lambda is not None else [7])
    p_base = _pack_geom_vec(base)
    lb0 = base.Lb_px
    g0 = base.g_inv_nm

    def _unpack(
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pp = p_base.copy()
        pp[g_free] = v[: len(g_free)]
        off = np.zeros(len(u_files), dtype=np.float64)
        off[free_f] = v[len(g_free) :]
        return pp, off

    def _coords(
        pp: NDArray[np.float64], off: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rot = pp[4]
        return x0 + off[fi] * np.cos(rot), y0 + off[fi] * np.sin(rot)

    v = np.concatenate([p_base[g_free], np.zeros(len(free_f), dtype=np.float64)])
    sol = None
    for _ in range(n_iter):
        pp, off = _unpack(v)
        xs, ys = _coords(pp, off)
        g_cur = _unpack_geom_vec(pp, base)
        qm = compute_pixel_q(g_cur, xs, ys, E, omega, approximation=approximation)
        G = g_cur.g_inv_nm
        m = np.rint(qm.Qx / G)
        ok = (np.abs(qm.Qx / G - m) < index_tol) & (m != 0)

        def _res(
            vec: NDArray[np.float64],
            mask_ok: NDArray[np.bool_] = ok,
            m_idx: NDArray[np.float64] = m,
        ) -> NDArray[np.float64]:
            pp_try, off_try = _unpack(vec)
            xs_t, ys_t = _coords(pp_try, off_try)
            g_try = _unpack_geom_vec(pp_try, base)
            q_try = compute_pixel_q(
                g_try,
                xs_t[mask_ok],
                ys_t[mask_ok],
                E[mask_ok],
                omega[mask_ok],
                approximation=approximation,
            )
            G_t = g_try.g_inv_nm
            sc = lb0 * g0 / k_all[mask_ok]
            return np.concatenate(
                [(q_try.Qx / G_t - m_idx[mask_ok]) * sc, (q_try.Qy / G_t) * sc]
            )

        sol = least_squares(_res, v, loss=loss, f_scale=f_scale_px, max_nfev=3000)
        v = sol.x

    assert sol is not None
    pp_final, off_final = _unpack(v)
    walkoff_map = {fn: float(o) for fn, o in zip(u_files, off_final, strict=True)}
    final_geom = _unpack_geom_vec(pp_final, base, walkoff=walkoff_map)
    indexed = index_spots(
        final_geom,
        spots_df,
        index_tol=index_tol,
        approximation=approximation,
        ref_Lb_px=lb0,
        ref_G_inv_nm=g0,
    )
    bragg = indexed["is_bragg"].to_numpy(dtype=bool)
    rx = indexed["res_x_px"].to_numpy(dtype=np.float64)[bragg]
    ry = indexed["res_y_px"].to_numpy(dtype=np.float64)[bragg]

    jac = sol.jac
    dof = max(len(sol.fun) - len(v), 1)
    s2 = float(np.sum(sol.fun**2) / dof)
    cov = np.linalg.pinv(jac.T @ jac) * s2
    sig_g = np.zeros(8, dtype=np.float64)
    sig_g[g_free] = np.sqrt(np.clip(np.diag(cov)[: len(g_free)], 0.0, None))
    sigma_dict = {
        "u0_px": float(sig_g[0]),
        "v0_px": float(sig_g[1]),
        "Lb_px": float(sig_g[2]),
        "beta_deg": float(np.degrees(sig_g[3])),
        "line_rot_deg": float(np.degrees(sig_g[4])),
        "omega0_deg": float(np.degrees(sig_g[5])),
        "psi_deg": float(np.degrees(sig_g[6])),
        "Lambda_nm": float(final_geom.Lambda_nm * sig_g[7]),
    }
    rms_x = float(np.sqrt(np.mean(rx**2))) if rx.size else 0.0
    rms_y = float(np.sqrt(np.mean(ry**2))) if ry.size else 0.0
    rms_tot = float(np.sqrt(0.5 * (rms_x**2 + rms_y**2))) if rx.size else 0.0
    return CalibrationResult(
        geometry=final_geom,
        sigma=sigma_dict,
        spots=indexed,
        rms_px=rms_tot,
        rms_x_px=rms_x,
        rms_y_px=rms_y,
        n_bragg=int(bragg.sum()),
    )


def fit_joint_geometry(
    spots_by_mount: Mapping[str, pd.DataFrame],
    start_by_mount: Mapping[str, Geometry] | None = None,
    *,
    n_iter: int = 5,
    index_tol: float = INDEX_TOL,
    approximation: ApproximationMode = "correct",
    loss: str = "soft_l1",
    f_scale_px: float = 2.0,
) -> JointCalibrationResult:
    """Fit joint geometry across multiple mounts of a beamtime sharing ``Lb_px``.

    All mounts in one beamtime share the sample-to-detector distance ``Lb_px``
    while ``(u0, v0, beta, line_rot, omega0, psi, Lambda)`` are fitted per
    mount, breaking the ``Lambda / Lb`` small-angle degeneracy.

    Parameters
    ----------
    spots_by_mount : mapping of str to DataFrame
        Spot tables keyed by mount/sample identifier.
    start_by_mount : mapping of str to Geometry, optional
        Initial geometry per mount. If omitted, each mount is initialized with
        ``initial_geometry``.
    n_iter : int, optional
        Alternating index/refine passes. Default is 5.
    index_tol : float, optional
        Bragg order indexing tolerance. Default is ``0.2``.
    approximation : ApproximationMode, optional
        Coordinate mapping mode. Default is ``"correct"``.
    loss : str, optional
        Robust loss name. Default is ``"soft_l1"``.
    f_scale_px : float, optional
        Soft-L1 loss scale in pixels. Default is ``2.0``.

    Returns
    -------
    JointCalibrationResult
        Jointly calibrated geometries and uncertainties.
    """
    keys = list(spots_by_mount.keys())
    if not keys:
        raise ValueError("spots_by_mount must contain at least one mount.")
    n = len(keys)
    starts: dict[str, Geometry] = {}
    for k in keys:
        if start_by_mount and k in start_by_mount:
            starts[k] = start_by_mount[k]
        else:
            starts[k] = initial_geometry(spots_by_mount[k])

    lb_init = float(np.median([starts[k].Lb_px for k in keys]))
    per_init = []
    for k in keys:
        vec = _pack_geom_vec(starts[k])
        per_init.append(
            np.r_[vec[:2], vec[3:]]
        )  # 7 params: u0, v0, beta, rot, om0, psi, logLam

    v = np.r_[lb_init, np.concatenate(per_init)]
    g0_ref = 2.0 * np.pi / 80.0
    lb0_ref = lb_init

    def _unpack_joint(vec: NDArray[np.float64]) -> tuple[float, list[Geometry]]:
        lb = float(vec[0])
        per = vec[1:].reshape(n, 7)
        geoms = [
            _unpack_geom_vec(np.r_[q[0], q[1], lb, q[2:]], starts[k])
            for k, q in zip(keys, per, strict=True)
        ]
        return lb, geoms

    sol = None
    mlist: list[tuple[NDArray[np.float64], NDArray[np.bool_]]] = []
    for _ in range(n_iter):
        _, gs = _unpack_joint(v)
        mlist = []
        for k, g in zip(keys, gs, strict=True):
            d = spots_by_mount[k]
            om = (
                d["omega_deg"].to_numpy(dtype=np.float64)
                if "omega_deg" in d.columns
                else d["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
            )
            qm = compute_pixel_q(
                g,
                d["x"].to_numpy(dtype=np.float64),
                d["y"].to_numpy(dtype=np.float64),
                d["energy_ev"].to_numpy(dtype=np.float64),
                om,
                approximation=approximation,
            )
            G = g.g_inv_nm
            m = np.rint(qm.Qx / G)
            ok = (np.abs(qm.Qx / G - m) < index_tol) & (m != 0)
            mlist.append((m, ok))

        def _res(
            vec: NDArray[np.float64],
            _mlist: list[tuple[NDArray[np.float64], NDArray[np.bool_]]] = mlist,
        ) -> NDArray[np.float64]:
            _, gs_try = _unpack_joint(vec)
            r_parts: list[NDArray[np.float64]] = []
            for (k, g_try), (m_arr, ok_arr) in zip(
                zip(keys, gs_try, strict=True), _mlist, strict=True
            ):
                d = spots_by_mount[k][ok_arr]
                om = (
                    d["omega_deg"].to_numpy(dtype=np.float64)
                    if "omega_deg" in d.columns
                    else d["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
                )
                E = d["energy_ev"].to_numpy(dtype=np.float64)
                qm = compute_pixel_q(
                    g_try,
                    d["x"].to_numpy(dtype=np.float64),
                    d["y"].to_numpy(dtype=np.float64),
                    E,
                    om,
                    approximation=approximation,
                )
                G_try = g_try.g_inv_nm
                k_w = (
                    2.0
                    * np.pi
                    / np.asarray(energy_to_wavelength_nm(E), dtype=np.float64)
                )
                sc = lb0_ref * g0_ref / k_w
                r_parts.append((qm.Qx / G_try - m_arr[ok_arr]) * sc)
                r_parts.append((qm.Qy / G_try) * sc)
            return np.concatenate(r_parts)

        sol = least_squares(_res, v, loss=loss, f_scale=f_scale_px, max_nfev=4000)
        v = sol.x

    assert sol is not None
    jac = sol.jac
    dof = max(len(sol.fun) - len(v), 1)
    s2 = float(np.sum(sol.fun**2) / dof)
    cov = np.linalg.pinv(jac.T @ jac) * s2
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    lb_final, gs_final = _unpack_joint(v)
    geoms_out: dict[str, Geometry] = {}
    per_mount_out: dict[str, CalibrationResult] = {}

    for i, (k, g_fin) in enumerate(zip(keys, gs_final, strict=True)):
        geoms_out[k] = g_fin
        indexed = index_spots(
            g_fin,
            spots_by_mount[k],
            index_tol=index_tol,
            approximation=approximation,
            ref_Lb_px=lb0_ref,
            ref_G_inv_nm=g0_ref,
        )
        bragg = indexed["is_bragg"].to_numpy(dtype=bool)
        rx = indexed["res_x_px"].to_numpy(dtype=np.float64)[bragg]
        ry = indexed["res_y_px"].to_numpy(dtype=np.float64)[bragg]
        j = 1 + 7 * i
        sigma_dict = {
            "u0_px": float(sig[j + 0]),
            "v0_px": float(sig[j + 1]),
            "Lb_px": float(sig[0]),
            "beta_deg": float(np.degrees(sig[j + 2])),
            "line_rot_deg": float(np.degrees(sig[j + 3])),
            "omega0_deg": float(np.degrees(sig[j + 4])),
            "psi_deg": float(np.degrees(sig[j + 5])),
            "Lambda_nm": float(g_fin.Lambda_nm * sig[j + 6]),
        }
        rms_x = float(np.sqrt(np.mean(rx**2))) if rx.size else 0.0
        rms_y = float(np.sqrt(np.mean(ry**2))) if ry.size else 0.0
        rms_tot = float(np.sqrt(0.5 * (rms_x**2 + rms_y**2))) if rx.size else 0.0
        per_mount_out[k] = CalibrationResult(
            geometry=g_fin,
            sigma=sigma_dict,
            spots=indexed,
            rms_px=rms_tot,
            rms_x_px=rms_x,
            rms_y_px=rms_y,
            n_bragg=int(bragg.sum()),
        )

    return JointCalibrationResult(
        geometries=geoms_out,
        shared_Lb_px=lb_final,
        shared_Lb_err_px=float(sig[0]),
        per_mount_results=per_mount_out,
    )


def refine_rod_verticality(
    spots_by_mount: Mapping[str, pd.DataFrame],
    base_geometries: Mapping[str, Geometry],
) -> dict[str, OmegaRefinement]:
    """Refine shared ``Lb_scale`` and per-mount ``(c0_px, t_px, d_omega0_deg,
    Lambda_nm)`` for vertical rods.

    Models rod-verticality kinematics across ``omega``:
    - An ``omega0`` error ``d_omega0`` rotates reciprocal space, tilting all rods
      by the same slope ``dQx/dQz``.
    - A wrong detector distance ``Lb`` tilts rods with a slope proportional to
      ``m^2 * tan(omega)`` via Ewald curvature, breaking the ``Lb / Lambda``
      degeneracy.
    - An illuminated patch off the rotation axis by ``t_px`` along the sample
      normal shifts the scattering origin along the spot line by
      ``c0_px + t_px * sin(W)`` and along the beam by ``t_px * (cos(W) - 1)``.

    Parameters
    ----------
    spots_by_mount : mapping of str to DataFrame
        Indexed spot tables per mount with columns ``order``, ``x``, ``y``,
        ``energy_ev``, ``omega_deg`` (or ``sample_theta_deg``), and weight column
        ``I_adu`` (or ``A``).
    base_geometries : mapping of str to Geometry
        Base geometry for each mount.

    Returns
    -------
    dict of str to OmegaRefinement
        Refined rod-verticality calibration for each mount.
    """
    keys = list(spots_by_mount.keys())
    if not keys:
        return {}

    data: list[tuple[NDArray[np.float64], Geometry]] = []
    for k in keys:
        df = spots_by_mount[k]
        g0 = base_geometries[k]
        if "order" not in df.columns:
            df = index_spots(g0, df)
            df = df[df["is_bragg"]]
        om = (
            df["omega_deg"].to_numpy(dtype=np.float64)
            if "omega_deg" in df.columns
            else df["sample_theta_deg"].to_numpy(dtype=np.float64) + 90.0
        )
        amp = (
            df["A"].to_numpy(dtype=np.float64)
            if "A" in df.columns
            else np.clip(df["I_adu"].to_numpy(dtype=np.float64), 1.0, None)
        )
        r = np.column_stack(
            [
                df["order"].to_numpy(dtype=np.float64),
                amp,
                df["x"].to_numpy(dtype=np.float64),
                df["y"].to_numpy(dtype=np.float64),
                df["energy_ev"].to_numpy(dtype=np.float64),
                om,
            ]
        )
        data.append((r, g0))

    p0: list[float] = [1.0]
    lo: list[float] = [0.85]
    hi: list[float] = [1.20]
    for _, g0 in data:
        p0 += [0.0, 0.0, 0.0, g0.Lambda_nm]
        lo += [-20.0, -2000.0, -3.0, 0.85 * g0.Lambda_nm]
        hi += [20.0, 2000.0, 3.0, 1.20 * g0.Lambda_nm]

    def _cal_of(p: NDArray[np.float64], j: int) -> OmegaRefinement:
        c0, t_val, d_om, lam = p[1 + 4 * j : 5 + 4 * j]
        return OmegaRefinement(
            Lb_scale=float(p[0]),
            c0_px=float(c0),
            t_px=float(t_val),
            d_omega0_deg=float(d_om),
            Lambda_nm=float(lam),
        )

    def _qx_spots(
        r: NDArray[np.float64], g0: Geometry, cal: OmegaRefinement
    ) -> NDArray[np.float64]:
        qx_out = np.zeros(len(r), dtype=np.float64)
        for om_val in np.unique(r[:, 5]):
            sel = r[:, 5] == om_val
            gf = g0.with_omega_refinement(cal, float(om_val))
            qm = compute_pixel_q(gf, r[sel, 2], r[sel, 3], r[sel, 4], r[sel, 5])
            qx_out[sel] = qm.Qx
        return qx_out

    def _res(p: NDArray[np.float64]) -> NDArray[np.float64]:
        parts: list[NDArray[np.float64]] = []
        for j, (r, g0) in enumerate(data):
            cal = _cal_of(p, j)
            assert cal.Lambda_nm is not None
            G = 2.0 * np.pi / cal.Lambda_nm
            qx = _qx_spots(r, g0, cal)
            parts.append((qx - r[:, 0] * G) / G * np.sqrt(r[:, 1]))
        return np.concatenate(parts)

    x_scale = [0.01] + [1.0, 100.0, 0.1, 0.5] * len(keys)
    sol = least_squares(_res, p0, bounds=(lo, hi), x_scale=x_scale)
    return {k: _cal_of(sol.x, j) for j, k in enumerate(keys)}


def build_order_table(
    spots_df: pd.DataFrame,
    geometry: Geometry,
    *,
    omega_refinement: OmegaRefinement | None = None,
    edge_margin_min_px: float = 25.0,
    index_tol: float = INDEX_TOL,
    apply_membrane: bool = False,
    membrane_thickness_nm: float = 50.0,
) -> pd.DataFrame:
    """Index spots and build a standardized Bragg order intensity table.

    Computes ``I_cps``, Poisson + read-noise uncertainty ``sigma_cps``, sample-frame
    ``(Qx, Qy, Qz, Qr, Q)``, and optional Si3N4 membrane transmission correction.

    Parameters
    ----------
    spots_df : DataFrame
        Raw extracted spot table from ``extract_frames_spots``.
    geometry : Geometry
        Calibrated geometry for the mount.
    omega_refinement : OmegaRefinement, optional
        Optional per-frame rod-verticality refinement.
    edge_margin_min_px : float, optional
        Minimum detector edge distance to keep a spot. Default is ``25.0``.
    index_tol : float, optional
        Bragg order fractional tolerance ``|Qx / G - m|``. Default is ``0.2``.
    apply_membrane : bool, optional
        If True, attach ``T_membrane``, ``I_cps_mc``, and ``sigma_cps_mc``.
    membrane_thickness_nm : float, optional
        Si3N4 window thickness in nm. Default is ``50.0``.

    Returns
    -------
    DataFrame
        Indexed Bragg order table with one row per ``(file, order)``.
    """
    if spots_df.empty:
        return pd.DataFrame()

    df = spots_df.copy()
    if "omega_deg" not in df.columns:
        df["omega_deg"] = df["sample_theta_deg"] + 90.0

    # Compute per-frame Q if omega_refinement is provided
    if omega_refinement is not None:
        qx_all = np.zeros(len(df), dtype=np.float64)
        qy_all = np.zeros(len(df), dtype=np.float64)
        qz_all = np.zeros(len(df), dtype=np.float64)
        qr_all = np.zeros(len(df), dtype=np.float64)
        q_all = np.zeros(len(df), dtype=np.float64)
        om_vals = df["omega_deg"].to_numpy(dtype=np.float64)
        for o_val in np.unique(om_vals):
            sel = om_vals == o_val
            gf = geometry.with_omega_refinement(omega_refinement, float(o_val))
            qm = compute_pixel_q(
                gf,
                df["x"].to_numpy(dtype=np.float64)[sel],
                df["y"].to_numpy(dtype=np.float64)[sel],
                df["energy_ev"].to_numpy(dtype=np.float64)[sel],
                om_vals[sel],
            )
            qx_all[sel] = qm.Qx
            qy_all[sel] = qm.Qy
            qz_all[sel] = qm.Qz
            qr_all[sel] = qm.Qr
            q_all[sel] = qm.Q
        lam_nm = (
            omega_refinement.Lambda_nm
            if omega_refinement.Lambda_nm is not None
            else geometry.Lambda_nm
        )
        G = 2.0 * np.pi / lam_nm
        m = np.rint(qx_all / G).astype(int)
        df["Qx"] = qx_all
        df["Qy"] = qy_all
        df["Qz"] = qz_all
        df["Qr"] = qr_all
        df["Q"] = q_all
        df["order"] = m
        df["index_residual"] = qx_all / G - m
        df["omega_true_deg"] = (
            om_vals + geometry.omega0_deg + omega_refinement.d_omega0_deg
        )
        df["Lambda_nm"] = lam_nm
        df["q_refined"] = True
    else:
        df = index_spots(geometry, df, index_tol=index_tol)
        df["Lambda_nm"] = geometry.Lambda_nm
        df["q_refined"] = False

    exp_s = df["exposure_s"].to_numpy(dtype=np.float64)
    i_adu = df["I_adu"].to_numpy(dtype=np.float64)
    raw_adu = df["raw_adu"].to_numpy(dtype=np.float64)
    npix = df["npix"].to_numpy(dtype=np.float64)
    offset_adu = df["offset_adu"].to_numpy(dtype=np.float64)
    read_sig = df["read_sigma_adu"].to_numpy(dtype=np.float64)

    df["I_cps"] = i_adu / exp_s
    df["sigma_cps"] = (
        np.sqrt(np.clip(raw_adu - npix * offset_adu, 1.0, None) + npix * (read_sig**2))
        / exp_s
    )

    keep = (
        (np.abs(df["index_residual"]) < index_tol)
        & (df["order"] != 0)
        & (df["edge_margin_px"] > edge_margin_min_px)
    )
    df = df[keep].sort_values("I_adu").groupby(["file", "order"]).tail(1)
    df = df.sort_values(["energy_ev", "omega_deg", "file", "order"]).reset_index(
        drop=True
    )

    if apply_membrane and not df.empty:
        from cd_reduce.reduction.corrections import membrane_transmission

        t_mem = membrane_transmission(
            df["omega_true_deg"].to_numpy(dtype=np.float64),
            df["energy_ev"].to_numpy(dtype=np.float64),
            thickness_nm=membrane_thickness_nm,
        )
        df["T_membrane"] = t_mem
        df["I_cps_mc"] = df["I_cps"] / t_mem
        df["sigma_cps_mc"] = df["sigma_cps"] / t_mem

    return df
