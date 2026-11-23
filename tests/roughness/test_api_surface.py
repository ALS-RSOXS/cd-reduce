"""Public API surface: removed legacy names are gone, new HOWTO names exported."""

import sys

import cd_reduce

REMOVED = [
    "extract_diffuse_ratio", "combine_diffuse_ratios", "decompose_edge_psd",
    "DiffuseRatio", "EdgePsdResult", "edge_psd_kernel", "form_factor_sq",
    "estimate_line_width", "readoff_edge_psd_born", "extract_rod_localized_diffuse_qy",
    "extract_offspec_rod_map", "extract_diffuse_cut", "synthetic_rough_grating_frame",
    "generate_rough_edge_1d",
]  # fmt: skip
ADDED = [
    "reduce_frame", "fit_psds", "PsdRecord", "order_scale", "LatentFilm",
    "DevelopedLine", "map_reduced", "simulate_rough_ensemble",
    "build_rough_grating_3d", "psd_palasantzas",
]  # fmt: skip


def test_removed_names_gone():
    for n in REMOVED:
        assert not hasattr(cd_reduce, n), n


def test_added_names_exported():
    for n in ADDED:
        assert hasattr(cd_reduce, n), n


def test_validation_not_importable_from_package():
    assert not any("validation" in m for m in sys.modules if m.startswith("cd_reduce"))
