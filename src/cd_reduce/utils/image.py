"""Image pre-processing utilities (dezingering, binning, robust noise statistics)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter


def dezinger_image(
    image: ArrayLike,
    med_result: NDArray[np.floating] | None = None,
    threshold: float = 10.0,
    size: int = 3,
) -> NDArray[np.float64]:
    """Remove hot pixels (zingers) exceeding ``threshold`` times local median.

    Parameters
    ----------
    image : array-like
        Input 2D detector image.
    med_result : ndarray, optional
        Pre-computed median-filtered image to avoid recomputation.
    threshold : float, optional
        Ratio threshold above which a pixel is replaced by the local median.
    size : int, optional
        Median filter window size. Default is 3.

    Returns
    -------
    ndarray of float64
        Dezingered 2D image.
    """
    img = np.asarray(image, dtype=np.float64)
    if med_result is None:
        med = median_filter(img, size=size)
    else:
        med = np.asarray(med_result, dtype=np.float64)
    denom = np.maximum(np.abs(med), 1e-12)
    diff_image = img / denom
    keep = diff_image <= threshold
    return np.where(keep, img, med)


def bin_image_2d(
    image: ArrayLike,
    bin_px: int = 2,
    mode: str = "sum",
) -> NDArray[np.float64]:
    """Block-bin a 2D image by an integer factor ``bin_px``.

    Parameters
    ----------
    image : array-like
        2D array of shape ``(n_row, n_col)``.
    bin_px : int, optional
        Integer binning factor (>= 1). Default is 2.
    mode : {"sum", "mean"}, optional
        Whether to sum or average pixels within each block. Default is ``"sum"``.

    Returns
    -------
    ndarray of float64
        Binned image of shape ``(n_row // bin_px, n_col // bin_px)``.
    """
    if bin_px < 1:
        raise ValueError(f"bin_px must be >= 1, got {bin_px}")
    img = np.asarray(image, dtype=np.float64)
    if bin_px == 1:
        return img.copy()
    n0, n1 = img.shape[0] // bin_px, img.shape[1] // bin_px
    blocks = img[: n0 * bin_px, : n1 * bin_px].reshape(n0, bin_px, n1, bin_px)
    if mode == "sum":
        return blocks.sum(axis=(1, 3), dtype=np.float64)
    if mode == "mean":
        return blocks.mean(axis=(1, 3), dtype=np.float64)
    raise ValueError(f"mode must be 'sum' or 'mean', got {mode!r}")


def robust_sigma(values: ArrayLike) -> float:
    """Compute Gaussian-equivalent standard deviation from the median absolute
    deviation.

    Parameters
    ----------
    values : array-like
        Sample values.

    Returns
    -------
    float
        ``1.4826 * median(|x - median(x)|)``.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def clipped_mean(values: ArrayLike, n_sigma: float = 4.0) -> float:
    """Compute mean after clipping outliers beyond ``n_sigma`` robust standard
    deviations.

    Parameters
    ----------
    values : array-like
        Input values.
    n_sigma : float, optional
        Clipping threshold in robust sigmas. Default is 4.0.

    Returns
    -------
    float
        Clipped mean.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    sig = robust_sigma(v) + 1e-9
    sel = np.abs(v - med) < n_sigma * sig
    return float(v[sel].mean()) if np.any(sel) else med


def detector_corner_offset(
    image: ArrayLike,
    corner_px: int | None = None,
    use_clipped_mean: bool = False,
) -> tuple[float, float]:
    """Estimate dark offset and per-pixel read noise from the four corner blocks.

    Parameters
    ----------
    image : array-like
        2D detector image.
    corner_px : int, optional
        Side length of each corner block in pixels. Defaults to
        ``min(400, min(shape) // 8)``.
    use_clipped_mean : bool, optional
        If True, average the four outlier-clipped corner means; if False, take the
        median across all four corner blocks.

    Returns
    -------
    offset_adu : float
        Estimated dark offset in ADU.
    read_sigma_adu : float
        Robust per-pixel read noise in ADU.
    """
    img = np.asarray(image, dtype=np.float64)
    b = corner_px if corner_px is not None else max(min(400, min(img.shape) // 8), 8)
    c1 = img[:b, :b].ravel()
    c2 = img[:b, -b:].ravel()
    c3 = img[-b:, :b].ravel()
    c4 = img[-b:, -b:].ravel()
    corners = np.concatenate([c1, c2, c3, c4])
    sigma = robust_sigma(corners)
    if use_clipped_mean:
        offset = float(
            np.mean(
                [
                    clipped_mean(c1),
                    clipped_mean(c2),
                    clipped_mean(c3),
                    clipped_mean(c4),
                ]
            )
        )
    else:
        offset = float(np.median(corners))
    return offset, sigma
