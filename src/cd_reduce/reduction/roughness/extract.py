"""PSD extraction: M = 4 sin^2(Qx w/2) S_LPR + cos^2(Qx w/2) S_LWR per f bin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cd_reduce.reduction.roughness.reduce import ReducedPixels

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class PsdTable:
    """Per-f-bin PSD estimates (two-sided, nm^3); NaN where a bin cannot be fit."""

    f_edges: Arr
    f_center: Arr
    S_lpr: Arr
    S_lwr: Arr
    S_ler: Arr
    var_lpr: Arr
    var_lwr: Arr
    var_ler: Arr
    cov_pw: Arr
    cond: Arr
    chi2_red: Arr
    n: NDArray[np.int64]
    S_cross: Arr | None = None
    var_cross: Arr | None = None


def default_f_edges(R_B: float, f_max: float, n: int = 26) -> Arr:
    """Log-spaced bin edges (n points) from R_B/2pi to f_max."""
    return np.geomspace(R_B / (2 * np.pi), f_max, n)


def _bin_index(f_abs: Arr, edges: Arr) -> NDArray[np.int64]:
    """Bin index per pixel; -1 outside. The last edge is inclusive."""
    idx = np.searchsorted(edges, f_abs, side="right") - 1
    idx[f_abs == edges[-1]] = len(edges) - 2
    idx[(f_abs < edges[0]) | (f_abs > edges[-1])] = -1
    return idx


def _resolve_edges(pixels: ReducedPixels, f_edges: Arr | None) -> Arr:
    if f_edges is not None:
        return np.asarray(f_edges, dtype=float)
    if "R_B" not in pixels.meta:
        raise ValueError("f_edges not given and pixels.meta has no 'R_B'")
    f_max = float(np.max(np.abs(pixels.f))) if pixels.f.size else 1.0
    return default_f_edges(pixels.meta["R_B"], f_max)


def fit_psds(
    pixels: ReducedPixels,
    w: float,
    f_edges: Arr | None = None,
    *,
    cross_term: bool = False,
    min_pixels: int = 10,
    max_cond: float = 1e8,
) -> PsdTable:
    """Weighted two-mode (optionally three-unknown) least squares per |f| bin.

    Covariance is inv(N) * max(1, chi2_red); estimates are never clipped.
    """
    edges = _resolve_edges(pixels, f_edges)
    nb = len(edges) - 1
    qx = np.abs(pixels.Qx)
    ok = np.isfinite(pixels.M) & np.isfinite(pixels.var_M) & (pixels.var_M > 0)
    idx = _bin_index(np.abs(pixels.f), edges)
    idx[~ok] = -1
    p = 3 if cross_term else 2

    def nan():
        return np.full(nb, np.nan)

    S = np.full((nb, p), np.nan)
    var = np.full((nb, p), np.nan)
    cov_pw, chi2, cond = nan(), nan(), np.full(nb, np.inf)
    n_arr = np.zeros(nb, dtype=np.int64)
    for i in range(nb):
        sel = idx == i
        n = int(sel.sum())
        n_arr[i] = n
        if n < min_pixels:
            continue
        x, m, v = qx[sel], pixels.M[sel], pixels.var_M[sel]
        cols = [4 * np.sin(x * w / 2) ** 2, np.cos(x * w / 2) ** 2]
        if cross_term:
            cols.append(2 * np.sin(x * w))
        A = np.stack(cols, axis=1)
        wt = 1.0 / v
        N = A.T @ (A * wt[:, None])
        with np.errstate(all="ignore"):
            c = np.linalg.cond(N)
        if not np.isfinite(c) or c > max_cond:
            continue
        cond[i] = c
        inv = np.linalg.inv(N)
        beta = inv @ (A.T @ (wt * m))
        r = m - A @ beta
        dof = n - p
        c2 = float(np.sum(wt * r * r) / dof) if dof > 0 else np.nan
        chi2[i] = c2
        cov = inv * max(1.0, c2 if np.isfinite(c2) else 1.0)
        S[i] = beta
        var[i] = np.diag(cov)
        cov_pw[i] = cov[0, 1]
    S_ler = S[:, 0] + S[:, 1] / 4
    var_ler = var[:, 0] + var[:, 1] / 16 + cov_pw / 2
    return PsdTable(
        f_edges=edges,
        f_center=np.sqrt(edges[:-1] * edges[1:])
        if edges[0] > 0
        else 0.5 * (edges[:-1] + edges[1:]),
        S_lpr=S[:, 0],
        S_lwr=S[:, 1],
        S_ler=S_ler,
        var_lpr=var[:, 0],
        var_lwr=var[:, 1],
        var_ler=var_ler,
        cov_pw=cov_pw,
        cond=cond,
        chi2_red=chi2,
        n=n_arr,
        S_cross=S[:, 2] if cross_term else None,
        var_cross=var[:, 2] if cross_term else None,
    )


def purity_bands(
    w: float,
    prior_lpr: Callable,
    prior_lwr: Callable,
    f_window: tuple[float, float],
    p: float,
    qx_max: float,
) -> dict[str, list[tuple[int, float, float]]]:
    """Qx bands around k*pi/w where the targeted mode gives >= p of M over the window.

    Odd k isolate S_LPR, even k >= 2 isolate S_LWR. Returns (k, qx_lo, qx_hi) per band.
    """
    f = np.geomspace(max(f_window[0], 1e-12), f_window[1], 64)
    sp = np.asarray(prior_lpr(f), dtype=float)
    sw = np.asarray(prior_lwr(f), dtype=float)
    out: dict[str, list[tuple[int, float, float]]] = {"lpr": [], "lwr": []}
    k = 1
    while k * np.pi / w <= qx_max:
        q0 = k * np.pi / w
        d = np.linspace(0.0, np.pi / (2 * w), 4001)
        a = 4 * np.sin((q0 + d) * w / 2) ** 2
        b = np.cos((q0 + d) * w / 2) ** 2
        if k % 2:
            ta, to = a[:, None] * sp, b[:, None] * sw
        else:
            ta, to = b[:, None] * sw, a[:, None] * sp
        with np.errstate(all="ignore"):
            purity = np.min(ta / (ta + to), axis=1)
        good = purity >= p
        # largest contiguous half-width from the centre
        bad = np.where(~good)[0]
        half = (
            d[bad[0] - 1]
            if bad.size and bad[0] > 0
            else (d[-1] if not bad.size else 0.0)
        )
        half = min(half, q0)
        if half <= 0:
            continue
        lo, hi = q0 - half, q0 + half
        hi = min(hi, qx_max)
        out["lpr" if k % 2 else "lwr"].append((k, float(lo), float(hi)))
        k += 1
    return out


def _wmean_bins(vals: Arr, var: Arr, idx: NDArray, nb: int, min_pixels: int = 1):
    """Weighted mean (weights 1/var), variance inflated by max(1, chi2_red), per bin.

    Note: 1/var weighting biases bin means when var scales as M^2.
    """
    S, V = np.full(nb, np.nan), np.full(nb, np.nan)
    n = np.zeros(nb, dtype=np.int64)
    for i in range(nb):
        s = idx == i
        n[i] = int(s.sum())
        if n[i] < min_pixels:
            continue
        wt = 1.0 / var[s]
        sw = wt.sum()
        mu = float(np.sum(wt * vals[s]) / sw)
        chi2 = float(np.sum(wt * (vals[s] - mu) ** 2) / (n[i] - 1)) if n[i] > 1 else 1.0
        S[i], V[i] = mu, max(1.0, chi2) / sw
    return S, V, n


def isolation_row_psds(
    pixels: ReducedPixels,
    w: float,
    bands: dict[str, list[tuple[int, float, float]]],
    f_edges: Arr,
) -> dict[str, dict[int, dict[str, Arr]]]:
    """Single-mode PSD per band k: S_LPR=M/(4 sin^2) (odd k), S_LWR=M/cos^2 (even k)."""
    edges = np.asarray(f_edges, dtype=float)
    nb = len(edges) - 1
    qx = np.abs(pixels.Qx)
    base = _bin_index(np.abs(pixels.f), edges)
    base[~(np.isfinite(pixels.M) & np.isfinite(pixels.var_M) & (pixels.var_M > 0))] = -1
    out: dict[str, dict[int, dict[str, Arr]]] = {"lpr": {}, "lwr": {}}
    for mode, items in bands.items():
        for k, lo, hi in items:
            idx = np.where((qx >= lo) & (qx <= hi), base, -1)
            if mode == "lpr":
                g = 4 * np.sin(qx * w / 2) ** 2
            else:
                g = np.cos(qx * w / 2) ** 2
            sel = idx >= 0
            g = np.where(sel & (g > 0), g, 1.0)
            S, V, n = _wmean_bins(pixels.M / g, pixels.var_M / g**2, idx, nb)
            out[mode][k] = {"S": S, "var": V, "n": n}
    return out


def quick_ler(pixels: ReducedPixels, w: float, f_edges: Arr) -> tuple[Arr, Arr]:
    """S_LER ~ mean(M)/2 over whole Qx periods (|Qx| w in [0, 2 pi K]).

    Unweighted mean: weights 1/var_M correlate with M itself and would bias a
    phase-average. Variance propagates var_M assuming independent pixels.
    """
    edges = np.asarray(f_edges, dtype=float)
    nb = len(edges) - 1
    S, V = np.full(nb, np.nan), np.full(nb, np.nan)
    if pixels.Qx.size == 0:
        return S, V
    qx = np.abs(pixels.Qx)
    K = int(np.floor(qx.max() * w / (2 * np.pi)))
    if K < 1:
        return S, V
    idx = _bin_index(np.abs(pixels.f), edges)
    idx[(qx * w > 2 * np.pi * K) | ~np.isfinite(pixels.M)] = -1
    for i in range(nb):
        s = idx == i
        n = int(s.sum())
        if n == 0:
            continue
        S[i] = float(pixels.M[s].mean()) / 2
        V[i] = float(np.sum(pixels.var_M[s])) / (4 * n * n)
    return S, V


def sigma_window(f_center: Arr, S: Arr, var_S: Arr) -> tuple[float, float]:
    """sigma = sqrt(2 int S df) by trapezoid on finite bin centres."""
    f, s, v = (np.asarray(a, dtype=float) for a in (f_center, S, var_S))
    ok = np.isfinite(f) & np.isfinite(s)
    f, s, v = f[ok], s[ok], np.where(np.isfinite(v[ok]), v[ok], 0.0)
    if f.size < 2:
        return float("nan"), float("nan")
    df = np.diff(f)
    c = np.zeros_like(f)
    c[:-1] += df / 2
    c[1:] += df / 2
    s2 = 2.0 * float(np.sum(c * s))
    if s2 <= 0:
        return float("nan"), float("nan")
    sig = float(np.sqrt(s2))
    var_s2 = 4.0 * float(np.sum(c**2 * v))
    return sig, float(np.sqrt(var_s2) / (2 * sig))
