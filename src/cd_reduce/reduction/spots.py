"""Model-free spot detection, centroid refinement, aperture-grid integration, and
template fitting.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from cd_reduce.reduction.frames import Frame, load_frames
from cd_reduce.reduction.geometry import Geometry, compute_pixel_q
from cd_reduce.utils.image import bin_image_2d, detector_corner_offset, robust_sigma

#: Default along-line half-widths (px) for the multi-aperture curve-of-growth grid.
APERTURE_ALONG_PX: tuple[int, ...] = (15, 25, 35, 50, 70, 95)
#: Default transverse half-widths (px) for the multi-aperture curve-of-growth grid.
APERTURE_TRANS_PX: tuple[int, ...] = (15, 25, 35, 50, 70, 100, 140)
#: Default single-aperture parameters (1024x1024 binned detector): ``(A, T, GAP, W)``.
DEFAULT_APERTURE: tuple[int, int, int, int] = (12, 18, 3, 10)


@dataclass(frozen=True, slots=True)
class SpotLine:
    """Straight line passing through the collinear diffraction orders of a frame.

    Attributes
    ----------
    origin_xy : ndarray of shape (2,)
        Point on the line (midpoint of brightest spot pair) as ``(x, y)`` in pixels.
    direction : ndarray of shape (2,)
        Unit vector along the line ``(dx, dy)`` in pixels, with ``dx >= 0``.
    """

    origin_xy: NDArray[np.float64]
    direction: NDArray[np.float64]

    @property
    def normal(self) -> NDArray[np.float64]:
        """Unit vector perpendicular to ``direction``."""
        return np.array([-self.direction[1], self.direction[0]], dtype=np.float64)

    @property
    def angle_deg(self) -> float:
        """Line orientation angle in degrees."""
        return float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))


def find_candidates(
    image: ArrayLike,
    *,
    bin_px: int = 8,
    threshold_sigma: float = 8.0,
    median_size: int = 21,
    smooth_sigma: float = 1.2,
    max_filter_size: int = 7,
) -> NDArray[np.float64]:
    """Detect candidate diffraction spots as local maxima of a block-binned image.

    Parameters
    ----------
    image : array-like
        2D detector counts (ADU).
    bin_px : int, optional
        Block-binning factor before peak finding (e.g. 8 for 4096px, 2 for 1024px).
        Default is 8.
    threshold_sigma : float, optional
        Detection threshold in units of robust noise of the smoothed excess.
        Default is 8.0.
    median_size : int, optional
        Median filter window size (in binned pixels) for local background.
        Default is 21.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in binned pixels). Default is 1.2.
    max_filter_size : int, optional
        Local maximum neighborhood size (in binned pixels). Default is 7.

    Returns
    -------
    ndarray of shape (n, 3)
        Detected candidates as ``(x, y, score)`` in full-resolution pixel
        coordinates, sorted by descending score.
    """
    img = np.asarray(image, dtype=np.float64)
    binned = bin_image_2d(img, bin_px=bin_px, mode="sum")
    local_bg = ndimage.median_filter(binned, size=median_size, mode="nearest")
    smooth = ndimage.gaussian_filter(binned - local_bg, smooth_sigma)
    sigma = robust_sigma(smooth)
    if sigma <= 0:
        return np.empty((0, 3), dtype=np.float64)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=max_filter_size)) & (
        smooth > threshold_sigma * sigma
    )
    rows, cols = np.nonzero(peaks)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    score = smooth[rows, cols] / sigma
    xy = np.column_stack(
        [(cols + 0.5) * bin_px - 0.5, (rows + 0.5) * bin_px - 0.5, score]
    )
    return xy[np.argsort(-xy[:, 2])]


def fit_spot_line(
    candidates: ArrayLike,
    *,
    tolerance_px: float = 40.0,
) -> tuple[SpotLine, NDArray[np.bool_]]:
    """Fit the diffraction spot line by score-weighted total least squares.

    Seeds the line with the two brightest candidates, keeps inliers within
    ``tolerance_px`` perpendicular distance, and refines the line direction
    via weighted orthogonal regression.

    Parameters
    ----------
    candidates : array-like of shape (n, 3)
        Candidate array ``(x, y, score)`` from ``find_candidates``.
    tolerance_px : float, optional
        Inlier distance threshold in pixels. Default is ``40.0``.

    Returns
    -------
    line : SpotLine
        Fitted diffraction line.
    inliers : ndarray of bool
        Boolean mask of candidates belonging to the line.
    """
    cand = np.asarray(candidates, dtype=np.float64)
    if len(cand) < 2:
        raise ValueError("At least two candidates are required to fit a spot line.")
    p0, p1 = cand[0, :2], cand[1, :2]
    diff = p1 - p0
    norm = float(np.linalg.norm(diff))
    if norm <= 0:
        raise ValueError("Top two candidates have identical coordinates.")
    direction = diff / norm
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    origin = 0.5 * (p0 + p1)
    inliers = np.zeros(len(cand), dtype=bool)
    for _ in range(3):
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        dist = (cand[:, :2] - origin) @ normal
        inliers = np.abs(dist) < tolerance_px
        if inliers.sum() < 2:
            inliers[:2] = True
        pts = cand[inliers, :2]
        w = np.clip(cand[inliers, 2], 1e-6, None)
        centre = np.average(pts, axis=0, weights=w)
        if len(pts) >= 2:
            cov = np.cov((pts - centre).T, aweights=w)
            evals, evecs = np.linalg.eigh(cov)
            direction = evecs[:, int(np.argmax(evals))]
            if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
                direction = -direction
        origin = origin + float((centre - origin) @ normal) * normal
    return SpotLine(origin_xy=origin, direction=direction), inliers


def _rotated_window(
    image: NDArray[np.float64],
    centre_xy: NDArray[np.float64],
    line: SpotLine,
    half_along: int,
    half_trans: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract raw pixel values in an axis-aligned bounding box and line coordinates."""
    r = int(np.ceil(np.hypot(half_along, half_trans))) + 2
    cx, cy = round(centre_xy[0]), round(centre_xy[1])
    y0, y1 = max(cy - r, 0), min(cy + r + 1, image.shape[0])
    x0, x1 = max(cx - r, 0), min(cx + r + 1, image.shape[1])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - centre_xy[0]
    dy = yy - centre_xy[1]
    along = dx * line.direction[0] + dy * line.direction[1]
    trans = dx * line.normal[0] + dy * line.normal[1]
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float64), along, trans


def refine_centroid(
    image: ArrayLike,
    guess_xy: ArrayLike,
    line: SpotLine,
    offset_adu: float,
    *,
    half_px: int = 40,
    n_iter: int = 4,
    smooth_sigma: float = 2.0,
) -> tuple[NDArray[np.float64], float]:
    """Refine a spot centroid by iterated center-of-mass over pixels above 25% of peak.

    Parameters
    ----------
    image : array-like
        2D detector image (ADU).
    guess_xy : array-like of shape (2,)
        Initial ``(x, y)`` guess in pixels.
    line : SpotLine
        Spot line defining the box orientation.
    offset_adu : float
        Detector dark offset subtracted before centroiding.
    half_px : int, optional
        Half-width of centroid box in pixels (e.g. 40 for 4096px, 10 for 1024px).
        Default is 40.
    n_iter : int, optional
        Maximum refinement iterations. Default is 4.
    smooth_sigma : float, optional
        Smoothing sigma in pixels for peak threshold determination. Default is 2.0.

    Returns
    -------
    xy : ndarray of shape (2,)
        Refined ``(x, y)`` centroid in pixels.
    peak_excess : float
        Maximum smoothed excess in the box (ADU/pixel).
    """
    img = np.asarray(image, dtype=np.float64)
    xy = np.asarray(guess_xy, dtype=np.float64).copy()
    peak = 0.0
    for _ in range(n_iter):
        vals, along, trans = _rotated_window(img, xy, line, half_px, half_px)
        if vals.size == 0:
            break
        excess = vals - offset_adu
        smooth = ndimage.gaussian_filter(excess, smooth_sigma)
        inside = (np.abs(along) <= half_px) & (np.abs(trans) <= half_px)
        if not np.any(inside):
            break
        peak = float(smooth[inside].max())
        if peak <= 0:
            break
        sel = inside & (smooth > 0.25 * peak)
        w = excess[sel]
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            break
        shift = np.array(
            [
                float(np.sum(w * along[sel]) / w_sum),
                float(np.sum(w * trans[sel]) / w_sum),
            ],
            dtype=np.float64,
        )
        xy = xy + shift[0] * line.direction + shift[1] * line.normal
        if np.hypot(shift[0], shift[1]) < 0.05:
            break
    return xy, peak


def integrate_single_aperture(
    image: ArrayLike,
    centre_xy: ArrayLike,
    line: SpotLine,
    *,
    along_px: int = 12,
    trans_px: int = 18,
    bg_gap_px: int = 3,
    bg_width_px: int = 10,
) -> dict[str, float]:
    """Integrate a spot in one rectangular aperture with transverse side-band
    background.

    Parameters
    ----------
    image : array-like
        2D detector image (ADU).
    centre_xy : array-like of shape (2,)
        Spot centroid ``(x, y)`` in pixels.
    line : SpotLine
        Spot line orientation.
    along_px, trans_px : int, optional
        Half-widths along and across the spot line (pixels).
    bg_gap_px, bg_width_px : int, optional
        Gap and width of the transverse side bands used for median background.

    Returns
    -------
    dict
        Keys ``I_adu``, ``raw_adu``, ``npix``, ``bg_adu``, ``bg_err_adu``.
    """
    img = np.asarray(image, dtype=np.float64)
    xy = np.asarray(centre_xy, dtype=np.float64)
    vals, al, tr = _rotated_window(
        img, xy, line, along_px, trans_px + bg_gap_px + bg_width_px
    )
    in_a = np.abs(al) <= along_px
    ap = in_a & (np.abs(tr) <= trans_px)
    band = (
        in_a
        & (np.abs(tr) > trans_px + bg_gap_px)
        & (np.abs(tr) <= trans_px + bg_gap_px + bg_width_px)
    )
    if not np.any(ap) or not np.any(band):
        return {
            "I_adu": 0.0,
            "raw_adu": 0.0,
            "npix": 0.0,
            "bg_adu": 0.0,
            "bg_err_adu": 0.0,
        }
    bvals = vals[band]
    bg = float(np.median(bvals))
    bg_err = 1.2533 * robust_sigma(bvals) / np.sqrt(max(bvals.size, 1))
    raw = float(vals[ap].sum())
    n = float(ap.sum())
    return {
        "I_adu": raw - n * bg,
        "raw_adu": raw,
        "npix": n,
        "bg_adu": bg,
        "bg_err_adu": float(bg_err),
    }


def integrate_grid(
    image: ArrayLike,
    centre_xy: ArrayLike,
    line: SpotLine,
    *,
    along_px: tuple[int, ...] = APERTURE_ALONG_PX,
    trans_px: tuple[int, ...] = APERTURE_TRANS_PX,
    bg_gap_px: int = 10,
    bg_width_px: int = 40,
) -> dict[str, NDArray[np.float64]]:
    """Integrate a spot over a 2D grid of rectangular apertures (curve of growth).

    Parameters
    ----------
    image : array-like
        2D detector image (ADU).
    centre_xy : array-like of shape (2,)
        Spot centroid ``(x, y)`` in pixels.
    line : SpotLine
        Spot line orientation.
    along_px, trans_px : tuple of int, optional
        Grid of aperture half-widths along and across the line.
    bg_gap_px, bg_width_px : int, optional
        Gap and width of transverse background side bands.

    Returns
    -------
    dict of ndarray
        Arrays ``sum``, ``npix``, ``bg``, ``bg_err``, ``raw`` each of shape
        ``(len(along_px), len(trans_px))``.
    """
    img = np.asarray(image, dtype=np.float64)
    xy = np.asarray(centre_xy, dtype=np.float64)
    a_max = max(along_px)
    t_max = max(trans_px) + bg_gap_px + bg_width_px
    vals, along, trans = _rotated_window(img, xy, line, a_max, t_max)
    shape = (len(along_px), len(trans_px))
    out = {k: np.full(shape, np.nan) for k in ("sum", "npix", "bg", "bg_err", "raw")}
    for i, a in enumerate(along_px):
        in_a = np.abs(along) <= a
        for j, t in enumerate(trans_px):
            ap = in_a & (np.abs(trans) <= t)
            band = (
                in_a
                & (np.abs(trans) > t + bg_gap_px)
                & (np.abs(trans) <= t + bg_gap_px + bg_width_px)
            )
            if band.sum() < 5 or ap.sum() == 0:
                continue
            bvals = vals[band]
            bg = float(np.median(bvals))
            bg_err = 1.2533 * robust_sigma(bvals) / np.sqrt(bvals.size)
            n = float(ap.sum())
            raw = float(vals[ap].sum())
            out["sum"][i, j] = raw - n * bg
            out["npix"][i, j] = n
            out["bg"][i, j] = bg
            out["bg_err"][i, j] = bg_err
            out["raw"][i, j] = raw
    return out


def extract_frame_spots(
    frame: Frame,
    *,
    preset: str = "auto",
    bin_px: int | None = None,
    threshold_sigma: float = 8.0,
    line_tolerance_px: float | None = None,
    centroid_half_px: int | None = None,
    merge_px: float | None = None,
    aperture_grid: bool = False,
    along_px: int | None = None,
    trans_px: int | None = None,
    bg_gap_px: int | None = None,
    bg_width_px: int | None = None,
) -> pd.DataFrame:
    """Detect, refine, and integrate all collinear diffraction spots in a single frame.

    Automatically scales detection and integration window sizes to the detector
    resolution (e.g. 1024x1024 binned vs 4096x4096 full-resolution) when
    ``preset="auto"``.

    Parameters
    ----------
    frame : Frame
        Input detector frame.
    preset : {"auto", "compact", "full_res"}, optional
        Preset scaling for detector resolution. Default is ``"auto"``.
    bin_px : int, optional
        Override candidate-detection binning factor.
    threshold_sigma : float, optional
        Detection threshold in robust sigmas. Default is 8.0.
    line_tolerance_px : float, optional
        Inlier tolerance for fitting the spot line.
    centroid_half_px : int, optional
        Half-width of centroid box in pixels.
    merge_px : float, optional
        Minimum distance (px) below which duplicate centroids are merged.
    aperture_grid : bool, optional
        If True, also compute the full curve-of-growth aperture grid ``sum_a{a}_t{t}``.
    along_px, trans_px, bg_gap_px, bg_width_px : int, optional
        Primary single-aperture half-widths and side-band parameters.

    Returns
    -------
    DataFrame
        One row per detected spot with position, line direction, and integrated counts.
    """
    min_dim = min(frame.image.shape)
    is_large = (preset == "full_res") or (preset == "auto" and min_dim >= 2048)

    b_px = bin_px if bin_px is not None else (8 if is_large else 2)
    tol_px = (
        line_tolerance_px
        if line_tolerance_px is not None
        else (40.0 if is_large else 12.0)
    )
    c_half = (
        centroid_half_px if centroid_half_px is not None else (40 if is_large else 10)
    )
    m_px = merge_px if merge_px is not None else (30.0 if is_large else 8.0)

    a_px = along_px if along_px is not None else (50 if is_large else 12)
    t_px = trans_px if trans_px is not None else (70 if is_large else 18)
    gap_px = bg_gap_px if bg_gap_px is not None else (10 if is_large else 3)
    w_px = bg_width_px if bg_width_px is not None else (40 if is_large else 10)

    off, sig = detector_corner_offset(frame.image)
    cand = find_candidates(frame.image, bin_px=b_px, threshold_sigma=threshold_sigma)
    if len(cand) < 2:
        return pd.DataFrame()

    line, inl = fit_spot_line(cand, tolerance_px=tol_px)
    refined: list[tuple[float, float, float, float]] = []
    for x, y, s in cand[inl]:
        xy, peak = refine_centroid(
            frame.image, np.array([x, y]), line, off, half_px=c_half
        )
        if all(np.hypot(xy[0] - r[0], xy[1] - r[1]) >= m_px for r in refined):
            refined.append((float(xy[0]), float(xy[1]), float(s), float(peak)))

    rows: list[dict[str, object]] = []
    h, w = frame.image.shape
    for x, y, s, peak in refined:
        ap_res = integrate_single_aperture(
            frame.image,
            np.array([x, y]),
            line,
            along_px=a_px,
            trans_px=t_px,
            bg_gap_px=gap_px,
            bg_width_px=w_px,
        )
        row: dict[str, Any] = {
            "file": frame.name,
            "energy_ev": frame.energy_ev,
            "sample_theta_deg": frame.sample_theta_deg,
            "omega_deg": frame.omega_deg,
            "exposure_s": frame.exposure_s,
            "x": x,
            "y": y,
            "score": s,
            "peak_excess_adu": peak,
            "I_adu": ap_res["I_adu"],
            "raw_adu": ap_res["raw_adu"],
            "npix": int(ap_res["npix"]),
            "bg_adu": ap_res["bg_adu"],
            "bg_err_adu": ap_res["bg_err_adu"],
            "offset_adu": off,
            "read_sigma_adu": sig,
            "max_adu": float(frame.image.max()),
            "edge_margin_px": float(min(x, y, w - 1 - x, h - 1 - y)),
            "line_x0": float(line.origin_xy[0]),
            "line_y0": float(line.origin_xy[1]),
            "line_dx": float(line.direction[0]),
            "line_dy": float(line.direction[1]),
        }
        if aperture_grid:
            grid = integrate_grid(
                frame.image,
                np.array([x, y]),
                line,
                bg_gap_px=gap_px,
                bg_width_px=w_px,
            )
            for key, arr in grid.items():
                for i_a, a_val in enumerate(APERTURE_ALONG_PX):
                    for j_t, t_val in enumerate(APERTURE_TRANS_PX):
                        row[f"{key}_a{a_val}_t{t_val}"] = float(arr[i_a, j_t])
        rows.append(row)
    return pd.DataFrame(rows)


def extract_frames_spots(
    source: Iterable[str | Path | Frame] | Path | str,
    *,
    sample: str = "sample",
    preset: str = "auto",
    aperture_grid: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Extract spots across all frames in ``source`` into a combined DataFrame.

    Parameters
    ----------
    source : Path, str, or iterable of Frame/Path
        Frames to process.
    sample : str, optional
        Sample label attached to the output table. Default is ``"sample"``.
    preset : {"auto", "compact", "full_res"}, optional
        Resolution preset. Default is ``"auto"``.
    aperture_grid : bool, optional
        Whether to compute the full aperture grid. Default is False.
    **kwargs : object
        Additional keyword arguments forwarded to ``extract_frame_spots``.

    Returns
    -------
    DataFrame
        Combined spot catalog with ``sample`` and ``frame_index`` columns.
    """
    frames_list = load_frames(source)
    tables: list[pd.DataFrame] = []
    for idx, fr in enumerate(frames_list):
        df = extract_frame_spots(
            fr,
            preset=preset,
            aperture_grid=aperture_grid,
            **kwargs,
        )
        if not df.empty:
            df.insert(0, "frame_index", idx)
            df.insert(0, "sample", sample)
            tables.append(df)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def extract_template_amplitudes(
    target_frames: Iterable[Frame],
    reference_frames: Iterable[Frame],
    target_spots: pd.DataFrame,
    reference_spots: pd.DataFrame,
    target_geometry: Geometry,
    reference_geometry: Geometry,
    *,
    half_box_px: int = 80,
    windows: tuple[tuple[int, int], ...] = ((25, 35), (35, 50), (45, 65)),
) -> pd.DataFrame:
    """Extract sharp Bragg order intensities by fitting a reference template + quadratic
    background.

    Each spot window ``P(x)`` in a latent/PEB frame is fitted as
    ``A * T(x) + c0 + c1*al + c2*tr + c3*al^2 + c4*tr^2 + c5*al*tr``,
    where ``T(x)`` is the corresponding diffraction order from a developed
    reference frame at the same energy and rotation angle, and the quadratic
    surface absorbs broad diffuse wings.

    Parameters
    ----------
    target_frames, reference_frames : iterable of Frame
        Target (e.g. latent/PEB) and reference (e.g. developed) frames.
    target_spots, reference_spots : DataFrame
        Spot tables from ``extract_frames_spots`` for target and reference.
    target_geometry, reference_geometry : Geometry
        Calibrated geometries for target and reference mounts.
    half_box_px : int, optional
        Half-width of subpixel-shifted cutout window in pixels. Default is 80.
    windows : tuple of (int, int), optional
        List of ``(along_px, trans_px)`` fit windows for window-spread uncertainty.

    Returns
    -------
    DataFrame
        Template-fitted order intensities ``I_cps`` and ``sigma_window_cps``.
    """
    t_map = {fr.name: fr for fr in target_frames}
    r_map = {fr.name: fr for fr in reference_frames}

    def _index_df(df: pd.DataFrame, geom: Geometry) -> pd.DataFrame:
        if df.empty:
            return df
        walkoff = geom.walkoff_offsets_px or {}
        off = df["file"].map(walkoff).fillna(0.0).to_numpy(dtype=np.float64)
        rot = np.radians(geom.line_rot_deg)
        qmap = compute_pixel_q(
            geom,
            df["x"].to_numpy() + off * np.cos(rot),
            df["y"].to_numpy() + off * np.sin(rot),
            df["energy_ev"].to_numpy(),
            df["omega_deg"].to_numpy(),
        )
        G = geom.g_inv_nm
        m = np.rint(qmap.Qx / G).astype(int)
        frac = qmap.Qx / G - m
        out = df.assign(
            order=m,
            frac=frac,
            Qx=qmap.Qx,
            Qy=qmap.Qy,
            Qz=qmap.Qz,
            omega_round=np.round(df["omega_deg"]).astype(int),
            E_round=df["energy_ev"].round(1),
        )
        out = out[(np.abs(out["frac"]) < 0.2) & (out["order"] != 0)]
        return out.sort_values("I_adu").groupby(["file", "order"]).tail(1)

    S = _index_df(target_spots, target_geometry)
    R = _index_df(reference_spots, reference_geometry)
    if S.empty or R.empty:
        return pd.DataFrame()

    ref_first = (
        R.sort_values("frame_index").groupby(["E_round", "omega_round"])["file"].first()
    )

    def _cutout(
        im: NDArray[np.float64], x: float, y: float
    ) -> NDArray[np.float64] | None:
        cx, cy = round(x), round(y)
        if (
            cy - half_box_px < 0
            or cx - half_box_px < 0
            or cy + half_box_px + 1 > im.shape[0]
            or cx + half_box_px + 1 > im.shape[1]
        ):
            return None
        w = im[
            cy - half_box_px : cy + half_box_px + 1,
            cx - half_box_px : cx + half_box_px + 1,
        ].astype(np.float64)
        return ndimage.shift(w, (-(y - cy), -(x - cx)), order=1)

    yy, xx = np.mgrid[-half_box_px : half_box_px + 1, -half_box_px : half_box_px + 1]

    rows: list[dict[str, object]] = []
    for fn, grp in S.groupby("file"):
        if fn not in t_map:
            continue
        E_r = grp["E_round"].iloc[0]
        om_r = grp["omega_round"].iloc[0]
        if (E_r, om_r) not in ref_first.index:
            continue
        rfn = str(ref_first[(E_r, om_r)])
        if rfn not in r_map:
            continue
        rg = R[R["file"] == rfn].set_index("order")
        im = t_map[str(fn)].image
        rim = r_map[rfn].image
        rot = float(np.arctan2(grp["line_dy"].iloc[0], grp["line_dx"].iloc[0]))
        al = xx * np.cos(rot) + yy * np.sin(rot)
        tr = -xx * np.sin(rot) + yy * np.cos(rot)

        for _, s in grp.iterrows():
            order_m = int(s["order"])
            if order_m not in rg.index:
                continue
            r_row = rg.loc[order_m]
            P = _cutout(im, float(s["x"]), float(s["y"]))
            T = _cutout(rim, float(r_row["x"]), float(r_row["y"]))
            if P is None or T is None:
                continue
            b_edge = max(min(15, half_box_px // 5), 2)
            T = T - float(np.median(np.r_[T[:b_edge].ravel(), T[-b_edge:].ravel()]))
            tsum = float(T.sum())
            exp_s = float(s["exposure_s"])

            row: dict[str, Any] = {
                "file": fn,
                "energy_ev": float(s["energy_ev"]),
                "omega_deg": float(s["omega_deg"]),
                "order": order_m,
                "Qx": float(s["Qx"]),
                "Qy": float(s["Qy"]),
                "Qz": float(s["Qz"]),
                "template_file": rfn,
                "exposure_s": exp_s,
            }
            win_vals: list[float] = []
            for a_w, t_w in windows:
                mask_w = (np.abs(al) <= a_w) & (np.abs(tr) <= t_w)
                design = np.stack(
                    [
                        T[mask_w],
                        np.ones(mask_w.sum()),
                        al[mask_w],
                        tr[mask_w],
                        al[mask_w] ** 2,
                        tr[mask_w] ** 2,
                        al[mask_w] * tr[mask_w],
                    ],
                    axis=1,
                )
                coef, *_ = np.linalg.lstsq(design, P[mask_w], rcond=None)
                dof = max(int(mask_w.sum()) - design.shape[1], 1)
                s2 = float(np.sum((P[mask_w] - design @ coef) ** 2) / dof)
                cov = np.linalg.pinv(design.T @ design) * s2
                amp = float(coef[0])
                amp_err = float(np.sqrt(max(cov[0, 0], 0.0)))
                i_val = amp * tsum / exp_s
                sig_val = amp_err * abs(tsum) / exp_s
                row[f"I_cps_w{a_w}_{t_w}"] = i_val
                row[f"sigma_cps_w{a_w}_{t_w}"] = sig_val
                win_vals.append(i_val)
            mid_idx = len(windows) // 2
            mid_a, mid_t = windows[mid_idx]
            row["I_cps"] = row[f"I_cps_w{mid_a}_{mid_t}"]
            row["sigma_fit_cps"] = row[f"sigma_cps_w{mid_a}_{mid_t}"]
            row["sigma_window_cps"] = 0.5 * (max(win_vals) - min(win_vals))
            row["sigma_cps"] = float(
                np.hypot(
                    float(row["sigma_fit_cps"]),
                    float(row["sigma_window_cps"]),
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)
