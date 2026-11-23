"""Tapered-line bias report (report only): sloped sidewalls break separable Gamma."""

from __future__ import annotations

import json

import numpy as np
import pytest

from .test_model_error import RESULTS, ratio_medians, sigma_window_ratios

pytestmark = pytest.mark.validation


def test_tapered_bias_reported(ensembles):
    """sigma_win / realised truth for a tapered line reduced with the same model."""
    res = sigma_window_ratios(ensembles, "tapered")
    meds = ratio_medians(ensembles, "tapered", min_blocks=1)
    assert all(np.isfinite(v) for v in res.values())
    assert all(np.isfinite(m) for _, m in meds)
    out = {
        "lpr": res["lpr"],
        "lwr": res["lwr"],
        "ler": res["ler"],
        "matched_ratio_median_by_omega": {str(om): m for om, m in meds},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tapered_bias.json").write_text(json.dumps(out, indent=1))
