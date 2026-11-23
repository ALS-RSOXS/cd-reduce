"""Roughness: HOWTO pixel reduction, PSD extraction, models and line cross sections."""

from cd_reduce.reduction.roughness.cross_section import (
    CrossSection,
    DevelopedLine,
    LatentFilm,
    to_index_map_2d,
)
from cd_reduce.reduction.roughness.extract import (
    PsdTable,
    default_f_edges,
    fit_psds,
    isolation_row_psds,
    purity_bands,
    quick_ler,
    sigma_window,
)
from cd_reduce.reduction.roughness.maps import (
    DEFAULT_ROUGHNESS_QY_SLAB_G,
    detector_q_pitch,
    map_reduced,
    plot_reduced_map,
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
from cd_reduce.reduction.roughness.record import PsdRecord
from cd_reduce.reduction.roughness.reduce import (
    ReducedPixels,
    absorption_factor,
    cos_theta_out,
    howto_prefactor,
    reduce_frame,
)
from cd_reduce.reduction.roughness.scale import (
    OrderScaleFit,
    debye_waller_sigma,
    order_efficiency,
    order_scale,
)

__all__ = [
    "DEFAULT_ROUGHNESS_QY_SLAB_G",
    "PSD_MODELS",
    "CrossSection",
    "DevelopedLine",
    "LatentFilm",
    "OrderScaleFit",
    "PsdFit",
    "PsdRecord",
    "PsdTable",
    "ReducedPixels",
    "absorption_factor",
    "cos_theta_out",
    "debye_waller_sigma",
    "default_f_edges",
    "detector_q_pitch",
    "fit_psd_model",
    "fit_psds",
    "howto_prefactor",
    "isolation_row_psds",
    "map_reduced",
    "order_efficiency",
    "order_scale",
    "plot_reduced_map",
    "psd_mack",
    "psd_palasantzas",
    "psd_power_law",
    "psd_spectral_peak",
    "purity_bands",
    "quick_ler",
    "reduce_frame",
    "sigma_from_psd",
    "sigma_window",
    "to_index_map_2d",
]
