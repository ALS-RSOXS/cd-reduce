"""Session-scoped cached frames for the validation suite."""

from __future__ import annotations

import numpy as np
import pytest

from cd_reduce.reduction.roughness import ReducedPixels, reduce_frame

from . import reference as ref
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
