"""Utility functions and decorators for cd-reduce."""

from cd_reduce.utils.attributes import process_vars_properties
from cd_reduce.utils.image import (
    bin_image_2d,
    clipped_mean,
    detector_corner_offset,
    dezinger_image,
    robust_sigma,
)
from cd_reduce.utils.name import infer_index_regex
from cd_reduce.utils.units import (
    HC_EV_ANGSTROM,
    HC_EV_NM,
    energy_to_wavelength,
    energy_to_wavelength_nm,
    q_to_theta,
    sample_theta_to_omega_deg,
    theta_to_q,
    wavelength_nm_to_k,
)

__all__ = [
    "HC_EV_ANGSTROM",
    "HC_EV_NM",
    "bin_image_2d",
    "clipped_mean",
    "detector_corner_offset",
    "dezinger_image",
    "energy_to_wavelength",
    "energy_to_wavelength_nm",
    "infer_index_regex",
    "process_vars_properties",
    "q_to_theta",
    "robust_sigma",
    "sample_theta_to_omega_deg",
    "theta_to_q",
    "wavelength_nm_to_k",
]
