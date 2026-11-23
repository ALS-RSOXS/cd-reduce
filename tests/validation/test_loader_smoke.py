"""Loader entry point over in-memory analytic frames."""

from __future__ import annotations

import numpy as np
import pytest

from cd_reduce import CdRsoxsLoader

from . import reference as ref

pytestmark = pytest.mark.validation


def test_loader_extract_psds_smoke(small_frames):
    cfg = ref.small()
    loader = CdRsoxsLoader(frames=small_frames[:3], geometry=cfg["geometry"])
    xs = ref.latent(285.0)
    rec = loader.extract_psds(
        {285.0: xs}, scale=ref.SCALE, adu_per_photon=ref.ADU_PER_PHOTON
    )
    assert rec.scale_source == "known"
    assert rec.scale == ref.SCALE
    assert rec.cross_section["class"] == "LatentFilm"
    assert rec.source["energies_ev"] == [285.0]
    assert all(np.isfinite(rec.sigma_win[k][0]) for k in ("lpr", "lwr", "ler"))
    with pytest.raises(KeyError):
        loader.reduce_roughness(
            {290.0: xs}, scale=ref.SCALE, adu_per_photon=ref.ADU_PER_PHOTON
        )
