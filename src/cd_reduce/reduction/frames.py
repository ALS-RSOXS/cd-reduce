"""FITS frame loading and metadata standardization for ALS BL 11.0.1.2 CD-RSoXS."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray

#: Standard mapping from canonical metadata keys to primary FITS header keys.
HEADER_KEYS: dict[str, tuple[str, ...]] = {
    "energy_ev": ("Beamline Energy", "ENERGY"),
    "sample_theta_deg": ("Sample Theta", "SAM_TH"),
    "exposure_s": ("EXPOSURE", "EXPTIME"),
    "beam_current": ("Beam Current", "RINGCRNT"),
    "ccd_theta_deg": ("CCD Theta", "DET_TH"),
    "sample_x_mm": ("Sample X", "SAM_X"),
    "sample_y_mm": ("Sample Y", "SAM_Y"),
    "sample_z_mm": ("Sample Z", "SAM_Z"),
    "sample_azimuth_deg": ("Sample Azimuthal Rotation", "SAM_AZ"),
    "polarization": ("EPU Polarization", "POLAR"),
    "hos": ("Higher Order Suppressor", "HOS"),
    "slits_vert": ("Upstream JJ Vert Aperture",),
    "slits_horz": ("Upstream JJ Horz Aperture",),
    "i0": ("AI 3 Izero", "IZERO"),
    "date": ("DATE", "DATE-OBS"),
}


@dataclass(slots=True)
class Frame:
    """One detector frame with its 2D image and motor/counter metadata.

    Attributes
    ----------
    image : ndarray of float64
        2D detector counts (ADU), shape ``(n_row, n_col)``.
    meta : dict of str to float or str
        Standardized header metadata.
    path : Path or None
        Source FITS filepath if loaded from disk.
    """

    image: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=np.float64)
        if self.image.ndim != 2:
            raise ValueError(f"Frame.image must be 2D, got shape {self.image.shape}")

    @classmethod
    def from_array(
        cls,
        image: ArrayLike,
        *,
        energy_ev: float = 285.2,
        omega_deg: float | None = None,
        sample_theta_deg: float | None = None,
        exposure_s: float = 1.0,
        polarization: float = 0.0,
        beam_current: float = 500.0,
        i0: float = 1.0,
        path: Path | str | None = None,
        **extra_meta: Any,
    ) -> Frame:
        """Create an in-memory ``Frame`` from a 2D array and explicit parameters.

        Parameters
        ----------
        image : array-like
            2D detector image (ADU).
        energy_ev : float, optional
            Photon energy in eV. Default is ``285.2``.
        omega_deg : float, optional
            Sample incidence angle in degrees (``0`` at normal incidence).
            If given, ``sample_theta_deg`` is set to ``omega_deg - 90.0``.
        sample_theta_deg : float, optional
            Motor ``Sample Theta`` in degrees. Default is ``-90.0`` if
            ``omega_deg`` is not specified.
        exposure_s : float, optional
            Frame exposure time in seconds. Default is ``1.0``.
        polarization : float, optional
            EPU polarization in degrees. Default is ``0.0`` (horizontal).
        beam_current : float, optional
            Storage ring current in mA. Default is ``500.0``.
        i0 : float, optional
            Incident intensity monitor. Default is ``1.0``.
        path : Path or str, optional
            Optional virtual or disk path.
        **extra_meta : Any
            Additional metadata fields.

        Returns
        -------
        Frame
            Constructed frame instance.
        """
        if omega_deg is not None:
            sam_th = float(omega_deg) - 90.0
        elif sample_theta_deg is not None:
            sam_th = float(sample_theta_deg)
        else:
            sam_th = -90.0

        meta: dict[str, Any] = {
            "energy_ev": float(energy_ev),
            "sample_theta_deg": sam_th,
            "exposure_s": float(exposure_s),
            "polarization": float(polarization),
            "beam_current": float(beam_current),
            "i0": float(i0),
            "ccd_theta_deg": 0.0,
            "sample_x_mm": 0.0,
            "sample_y_mm": 0.0,
            "sample_z_mm": 0.0,
            **extra_meta,
        }
        return cls(
            image=np.asarray(image, dtype=np.float64),
            meta=meta,
            path=Path(path) if path is not None else None,
        )

    @property
    def name(self) -> str:
        """Filename or identifier of the frame."""
        if self.path is not None:
            return self.path.name
        return str(self.meta.get("file", "memory_frame.fits"))

    @property
    def energy_ev(self) -> float:
        """Photon energy in eV."""
        return float(self.meta.get("energy_ev", 285.2))

    @property
    def sample_theta_deg(self) -> float:
        """Motor Sample Theta in degrees."""
        return float(self.meta.get("sample_theta_deg", -90.0))

    @property
    def omega_deg(self) -> float:
        """Sample incidence angle ``omega = sample_theta_deg + 90.0`` in degrees."""
        if "omega_deg" in self.meta:
            return float(self.meta["omega_deg"])
        return self.sample_theta_deg + 90.0

    @property
    def exposure_s(self) -> float:
        """Exposure time in seconds."""
        val = float(self.meta.get("exposure_s", 1.0))
        return val if val > 0 else 1.0

    @property
    def polarization_deg(self) -> float:
        """EPU polarization in degrees."""
        val = float(self.meta.get("polarization", 0.0))
        return 0.0 if np.isnan(val) else val


def read_meta(path: Path | str) -> dict[str, Any]:
    """Read standardized header metadata from a FITS file without loading the image.

    Parameters
    ----------
    path : Path or str
        Path to FITS file.

    Returns
    -------
    dict
        Standardized metadata dictionary.
    """
    p = Path(path)
    with fits.open(p, memmap=False) as hdul:
        header = hdul[0].header
        out: dict[str, Any] = {"file": p.name}
        for name, candidates in HEADER_KEYS.items():
            val: Any = np.nan
            for key in candidates:
                if key in header:
                    val = header[key]
                    break
            if name == "date":
                out[name] = str(val) if val is not np.nan else ""
            else:
                try:
                    out[name] = float(val)
                except (TypeError, ValueError):
                    out[name] = float("nan")
    return out


def load_frame(path: Path | str, *, max_retries: int = 3) -> Frame:
    """Load a FITS frame (image in HDU 2 or primary HDU) and its metadata.

    Includes a retry loop for cloud-synced filesystems that may hydrate files
    on first access.

    Parameters
    ----------
    path : Path or str
        Path to the FITS file.
    max_retries : int, optional
        Number of retries on partial read. Default is 3.

    Returns
    -------
    Frame
        Loaded frame with 2D float64 image and metadata.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not a valid file.")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with fits.open(p, memmap=False) as hdul:
                hdu_idx = 2 if len(hdul) > 2 and hdul[2].data is not None else 0
                raw_data = hdul[hdu_idx].data
                if raw_data is None:
                    raise ValueError(f"No image data found in {p}")
                image = np.asarray(raw_data, dtype=np.float64)
            meta = read_meta(p)
            return Frame(image=image, meta=meta, path=p)
        except ValueError as err:
            last_error = err
            if attempt + 1 < max_retries:
                time.sleep(0.5 * (attempt + 1))
    raise OSError(f"Could not read complete image from {p}") from last_error


def load_ai_metadata(ai_path: Path | str) -> pd.DataFrame:
    """Load companion ``-AI.txt`` metadata table written by BL 11.0.1.2 Scan Panel.

    Parameters
    ----------
    ai_path : Path or str
        Path to the ``-AI.txt`` file.

    Returns
    -------
    DataFrame
        Parsed metadata rows.
    """
    p = Path(ai_path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not a valid file.")
    header_line = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if "DATA" in line:
                header_line = i
                break
    return pd.read_csv(p, skiprows=header_line, header=1, delimiter="\t")


def load_frames(
    source: Iterable[str | Path | Frame] | Path | str,
    *,
    deduplicate_omega: bool = False,
    energy_ev: float | None = None,
    energy_tol_ev: float = 0.3,
) -> list[Frame]:
    """Load a collection of frames from a directory, path list, or ``Frame`` sequence.

    Parameters
    ----------
    source : Path, str, or iterable of Path/str/Frame
        Directory to search recursively for ``*.fits``, or an explicit iterable
        of filepaths or in-memory ``Frame`` objects.
    deduplicate_omega : bool, optional
        If True, keep only the first frame for each ``(round(energy_ev, 1), omega_deg)``
        pair.
    energy_ev : float or None, optional
        If provided, filter frames to within ``energy_tol_ev`` of ``energy_ev``.
    energy_tol_ev : float, optional
        Energy matching tolerance in eV. Default is ``0.3``.

    Returns
    -------
    list of Frame
        Loaded (and optionally filtered) frames.
    """
    raw_frames: list[Frame] = []
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            paths = sorted(p.rglob("*.fits"), key=lambda x: x.name)
            raw_frames = [load_frame(fp) for fp in paths]
        elif p.is_file():
            raw_frames = [load_frame(p)]
        else:
            raise FileNotFoundError(f"Path does not exist: {p}")
    elif isinstance(source, Iterable):
        for item in source:
            if isinstance(item, Frame):
                raw_frames.append(item)
            elif isinstance(item, (str, Path)):
                raw_frames.append(load_frame(item))
            else:
                raise TypeError(f"Unsupported frame source item: {type(item)}")
    else:
        raise TypeError(f"Invalid source input: {source}")

    filtered: list[Frame] = []
    seen: set[tuple[float, float]] = set()
    for fr in raw_frames:
        if energy_ev is not None and abs(fr.energy_ev - energy_ev) > energy_tol_ev:
            continue
        if deduplicate_omega:
            key = (round(fr.energy_ev, 1), round(fr.omega_deg, 4))
            if key in seen:
                continue
            seen.add(key)
        filtered.append(fr)
    return filtered
