"""Roughness: PSD models, line cross sections and the legacy Born-kernel extractor."""

from cd_reduce.reduction.roughness._legacy import (
    DiffuseRatio,
    EdgePsdResult,
    combine_diffuse_ratios,
    decompose_edge_psd,
    edge_psd_kernel,
    estimate_line_width,
    extract_diffuse_ratio,
    form_factor_sq,
)
from cd_reduce.reduction.roughness.cross_section import (
    CrossSection,
    DevelopedLine,
    LatentFilm,
    to_index_map_2d,
)
from cd_reduce.reduction.roughness.psd_models import (
    PSD_MODELS,
    PsdFit,
    fit_psd_model,
    psd_mack,
    psd_palasantzas,
    psd_power_law,
    psd_spectral_peak,
    sigma_from_psd,
)
from cd_reduce.reduction.roughness.reduce import (
    ReducedPixels,
    absorption_factor,
    cos_theta_out,
    howto_prefactor,
    reduce_frame,
)

__all__ = [
    "PSD_MODELS",
    "CrossSection",
    "DevelopedLine",
    "DiffuseRatio",
    "EdgePsdResult",
    "LatentFilm",
    "PsdFit",
    "ReducedPixels",
    "absorption_factor",
    "combine_diffuse_ratios",
    "cos_theta_out",
    "decompose_edge_psd",
    "edge_psd_kernel",
    "estimate_line_width",
    "extract_diffuse_ratio",
    "fit_psd_model",
    "form_factor_sq",
    "howto_prefactor",
    "psd_mack",
    "psd_palasantzas",
    "psd_power_law",
    "psd_spectral_peak",
    "reduce_frame",
    "sigma_from_psd",
    "to_index_map_2d",
]
