"""Unit conversion helpers for soft X-ray scattering and reflectivity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import constants

#: Planck constant times speed of light in eV * nm.
HC_EV_NM: float = float(constants.h * constants.c / constants.e * 1e9)
#: Planck constant times speed of light in eV * Angstrom.
HC_EV_ANGSTROM: float = HC_EV_NM * 10.0


def energy_to_wavelength_nm(
    energy_ev: ArrayLike,
) -> NDArray[np.float64] | float:
    """Convert photon energy in eV to wavelength in nm (or vice versa).

    Parameters
    ----------
    energy_ev : float or array-like
        Photon energy in eV (must be positive).

    Returns
    -------
    float or ndarray
        Wavelength in nm.
    """
    arr = np.asarray(energy_ev, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError(f"energy_ev must be positive, got {energy_ev}")
    res = HC_EV_NM / arr
    if np.ndim(energy_ev) == 0:
        return float(res)
    return res


def energy_to_wavelength(val: ArrayLike) -> NDArray[np.float64] | float:
    """Convert photon energy [eV] to wavelength [Angstrom] (pxr-reduce compatible).

    Parameters
    ----------
    val : float or array-like
        Photon energy in eV or wavelength in Angstroms.

    Returns
    -------
    float or ndarray
        Wavelength in Angstroms or energy in eV.
    """
    arr = np.asarray(val, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError(f"val must be positive, got {val}")
    res = HC_EV_ANGSTROM / arr
    if np.ndim(val) == 0:
        return float(res)
    return res


def wavelength_nm_to_k(
    wavelength_nm: ArrayLike,
) -> NDArray[np.float64] | float:
    """Convert wavelength in nm to wavenumber ``k = 2 * pi / lambda`` in 1/nm.

    Parameters
    ----------
    wavelength_nm : float or array-like
        Wavelength in nm.

    Returns
    -------
    float or ndarray
        Wavenumber in 1/nm.
    """
    arr = np.asarray(wavelength_nm, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError(f"wavelength_nm must be positive, got {wavelength_nm}")
    res = 2.0 * np.pi / arr
    if np.ndim(wavelength_nm) == 0:
        return float(res)
    return res


def sample_theta_to_omega_deg(
    sample_theta_deg: ArrayLike,
    offset_deg: float = 90.0,
) -> NDArray[np.float64] | float:
    """Convert ALS BL 11.0.1.2 ``Sample Theta`` motor angle to incidence angle
    ``omega``.

    In transmission CD-RSoXS/CD-SAXS at 11.0.1.2, normal incidence corresponds
    to ``Sample Theta = -90 deg``, so ``omega = sample_theta_deg + 90.0``.

    Parameters
    ----------
    sample_theta_deg : float or array-like
        Motor ``Sample Theta`` in degrees.
    offset_deg : float, optional
        Offset added to motor angle. Default is ``90.0``.

    Returns
    -------
    float or ndarray
        Incidence angle ``omega`` in degrees.
    """
    arr = np.asarray(sample_theta_deg, dtype=np.float64) + offset_deg
    if np.ndim(sample_theta_deg) == 0:
        return float(arr)
    return arr


def theta_to_q(
    theta_rad: ArrayLike,
    wavelength: ArrayLike,
) -> NDArray[np.float64] | float:
    """Convert half-scattering angle ``theta`` [rad] to ``q = 4*pi*sin(theta)/lambda``.

    Parameters
    ----------
    theta_rad : float or array-like
        Half-scattering (or reflection) angle in radians.
    wavelength : float or array-like
        Wavelength in the desired length unit (nm or Angstrom).

    Returns
    -------
    float or ndarray
        Momentum transfer in inverse length units (1/nm or 1/Angstrom).
    """
    th = np.asarray(theta_rad, dtype=np.float64)
    lam = np.asarray(wavelength, dtype=np.float64)
    res = 4.0 * np.pi * np.sin(th) / lam
    if np.ndim(theta_rad) == 0 and np.ndim(wavelength) == 0:
        return float(res)
    return res


def q_to_theta(
    q: ArrayLike,
    wavelength: ArrayLike,
) -> NDArray[np.float64] | float:
    """Convert momentum transfer ``q`` to reflection angle in degrees.

    Parameters
    ----------
    q : float or array-like
        Momentum transfer in inverse length units.
    wavelength : float or array-like
        Wavelength in matching length units.

    Returns
    -------
    float or ndarray
        Angle in degrees.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    lam = np.asarray(wavelength, dtype=np.float64)
    arg = np.clip(lam * q_arr / (4.0 * np.pi), -1.0, 1.0)
    res = np.rad2deg(np.arcsin(arg))
    if np.ndim(q) == 0 and np.ndim(wavelength) == 0:
        return float(res)
    return res
