"""Session-scoped cached frames for the validation suite."""

from __future__ import annotations

import numpy as np
import pytest

from cd_reduce.reduction.roughness import ReducedPixels, reduce_frame
from cd_reduce.simulation.rough_scattering import simulate_rough_ensemble
from cd_reduce.simulation.structures import build_rough_grating_3d

from . import reference as ref
from .forward.ensemble_frames import render_ensemble_frame
from .forward.model_frames import render_model_frame
from .forward.orders import order_table


@pytest.fixture(scope="session")
def small_frames():
    cfg = ref.small()
    case = ref.CASES["both"]
    sle = ref.sigma_ler_full(case["psd_lpr"], case["psd_lwr"])
    frames = []
    for i, om in enumerate(np.tile(cfg["omegas"], cfg["n_real"])):
        rng = np.random.default_rng(1000 + i)
        xs = case["xs"](285.0)
        frames.append(
            render_model_frame(
                xs, cfg["geometry"], float(om),
                psd_lpr=case["psd_lpr"], psd_lwr=case["psd_lwr"],
                scale=ref.SCALE, shape=cfg["shape"],
                adu_per_photon=ref.ADU_PER_PHOTON,
                sigma_ler_full=sle, rng=rng,
            )
        )  # fmt: skip
    return frames


@pytest.fixture(scope="session")
def small_reduced(small_frames):
    cfg = ref.small()
    xs = ref.CASES["both"]["xs"](285.0)
    items = [
        reduce_frame(
            fr, cfg["geometry"], xs, scale=ref.SCALE,
            adu_per_photon=ref.ADU_PER_PHOTON, frame_id=i,
        )
        for i, fr in enumerate(small_frames)
    ]  # fmt: skip
    return ReducedPixels.concat(items)


@pytest.fixture(scope="session")
def small_order_table(small_frames):
    cfg = ref.small()
    return order_table(small_frames, cfg["geometry"], adu_per_photon=ref.ADU_PER_PHOTON)


class _EnsembleCache:
    """Lazily built ensembles, edge fields, frames and pixels (session cache).

    ``seed`` indexes independent ensembles of ``small()`` size; seed 0 is the
    primary ensemble, seeds 0..K-1 give the scatter for the chi^2 criteria (R20).
    ``engine`` selects Born or multislice; both engines see the same realisations
    for a given seed, so the edge fields are shared. Multislice ensembles carry
    the epsilon Born reference (``metadata["born_eps"]``).
    """

    def __init__(self, n_real: int | None = None, k_seeds: int = 8):
        self.cfg = ref.small()
        if n_real is not None:
            self.cfg["ensemble_n_real"] = int(n_real)
        self.k_seeds = int(k_seeds)
        self._res, self._edges, self._frames, self._red = {}, {}, {}, {}

    def result(self, case: str, seed: int = 0, engine: str = "born"):
        key = (case, seed, engine)
        if key not in self._res:
            cfg, c = self.cfg, ref.CASES[case]
            xs = c["xs"](285.0)
            edges = []

            def make(rng):
                st = build_rough_grating_3d(
                    xs, n_lines=cfg["n_lines"], ly_nm=cfg["ly_nm"],
                    dx_nm=cfg["dx_nm"], dy_nm=cfg["dy_nm"], dz_nm=cfg["dz_nm"],
                    psd_lpr=c["psd_lpr"], psd_lwr=c["psd_lwr"], rng=rng,
                )  # fmt: skip
                edges.append((st.metadata["P_nm"], st.metadata["W_nm"]))
                return st

            # The engine's omega sense is opposite to the detector's W (R21):
            # simulate at -W so the rendered frames sit at +W.
            ms = engine == "multislice"
            self._res[key] = simulate_rough_ensemble(
                make, -np.asarray(cfg["ensemble_omegas"]), engine=engine,
                n_realisations=cfg["ensemble_n_real"], seed=2000 + 101 * seed,
                born_reference="epsilon" if ms else None,
            )  # fmt: skip
            assert len(edges) == cfg["ensemble_n_real"], (
                f"{len(edges)} edge sets collected, expected {cfg['ensemble_n_real']}"
            )
            prev = self._edges.setdefault((case, seed), edges)
            assert all(
                np.array_equal(a, b)
                for pw, qw in zip(prev, edges, strict=True)
                for a, b in zip(pw, qw, strict=True)
            ), "engines must see identical realisations"
        return self._res[key]

    def edges(self, case: str, seed: int = 0):
        """``[(P, W), ...]`` per realisation, each of shape (n_lines, ny)."""
        if (case, seed) not in self._edges:
            self.result(case, seed)
        return self._edges[(case, seed)]

    def seeds(self, pooled: bool) -> range:
        return range(self.k_seeds) if pooled else range(1)

    def sigma_e(self, case: str, seed: int = 0, *, pooled: bool = False) -> float:
        """Realised LER sigma: sqrt(mean P^2 + mean W^2 / 4) over all fields (R4).

        ``pooled`` uses the fields of all ``k_seeds`` ensembles (equal sizes).
        """
        seeds = self.seeds(True) if pooled else [seed]
        e = [pw for s in seeds for pw in self.edges(case, s)]
        p2 = np.mean([np.mean(p**2) for p, _ in e])
        w2 = np.mean([np.mean(w**2) for _, w in e])
        return float(np.sqrt(p2 + w2 / 4))

    def qx_max(self, case: str, *, pooled: bool = False) -> float:
        """First-order cut |Qx| sigma_e <= 0.3 (R19).

        sigma_e is that of the primary ensemble, or of all seeds if ``pooled``.
        """
        return 0.3 / self.sigma_e(case, pooled=pooled)

    def realised_spectra(self, case: str, seed: int = 0, *, pooled: bool = False):
        """Two-sided periodograms of P, W (nm^3) averaged over lines/realisations.

        Returns ``(f, S_P, S_W)`` with ``f`` the field's FFT frequencies (f = 0
        excluded), sorted ascending. ``pooled`` averages over all seeds too.
        """
        seeds = self.seeds(True) if pooled else [seed]
        e = [pw for s in seeds for pw in self.edges(case, s)]
        ly = self.cfg["ly_nm"]
        ny = e[0][0].shape[-1]
        dy = ly / ny
        f = np.fft.fftfreq(ny, d=dy)

        def pgram(fields):
            x = np.concatenate([np.atleast_2d(a) for a in fields], axis=0)
            return np.mean(np.abs(np.fft.fft(x, axis=-1)) ** 2, axis=0) * dy**2 / ly

        sp = pgram([p for p, _ in e])
        sw = pgram([w for _, w in e])
        o = np.argsort(f)
        keep = f[o] != 0
        return f[o][keep], sp[o][keep], sw[o][keep]

    def frames(
        self, case: str, poisson: bool = True, seed: int = 0, engine: str = "born"
    ):
        key = (case, poisson, seed, engine)
        if key not in self._frames:
            res = self.result(case, seed, engine)
            xs = ref.CASES[case]["xs"](285.0)
            self._frames[key] = [
                render_ensemble_frame(
                    res, i, xs, self.cfg["geometry"], scale=ref.SCALE,
                    shape=self.cfg["shape"], adu_per_photon=ref.ADU_PER_PHOTON,
                    rng=np.random.default_rng(3000 + 37 * seed + i),
                    poisson=poisson,
                )
                for i in range(res.omega_deg.size)
            ]  # fmt: skip
        return self._frames[key]

    def reduced(self, case: str, seed: int = 0, engine: str = "born") -> ReducedPixels:
        key = (case, seed, engine)
        if key not in self._red:
            xs = ref.CASES[case]["xs"](285.0)
            items = [
                reduce_frame(
                    fr, self.cfg["geometry"], xs, scale=ref.SCALE,
                    adu_per_photon=ref.ADU_PER_PHOTON, frame_id=i,
                )
                for i, fr in enumerate(self.frames(case, seed=seed, engine=engine))
            ]  # fmt: skip
            self._red[key] = ReducedPixels.concat(items)
        return self._red[key]

    def reduced_pooled(self, case: str, engine: str = "born") -> ReducedPixels:
        """Reduced pixels of all ``k_seeds`` ensembles (R22)."""
        return ReducedPixels.concat(
            [self.reduced(case, s, engine) for s in self.seeds(True)]
        )


@pytest.fixture(scope="session")
def ensembles():
    return _EnsembleCache()


@pytest.fixture(scope="session")
def ensembles_pooled32():
    """n = 4 per seed (32 pooled realisations) for the sigma_win criteria."""
    return _EnsembleCache(n_real=4)
