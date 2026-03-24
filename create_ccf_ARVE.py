from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np


def compute_ccf_from_spectra(
    spectra: Sequence[dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]],
    mask_path: str | Path = "G2_mask.csv",
    *,
    time_val: np.ndarray | None = None,
    target: str = "Sun",
    stellar_parameters: dict[str, Any] | None = None,
    resolution: float = 100000,
    medium: str = "vac",
    mask_medium: str = "vac",
    vrad_grid: tuple[float, float, float] = (-20.0, 20.0, 1.0),
    exclude_tellurics: bool = False,
    ccf_err_scale: bool = True,
    weight_name: str = "weight",
    same_wave_grid: bool = True,
) -> dict[str, np.ndarray]:
    """Compute ARVE CCFs from spectra arrays.

    Each spectrum can be provided as:
    - ``{"wave": wave, "flux": flux}``
    - ``{"wave": wave, "flux": flux, "flux_err": flux_err}``
    - ``(wave, flux)``
    - ``(wave, flux, flux_err)``

    Args:
        spectra: Sequence of spectra.
        mask_path: Path to ARVE mask CSV (default: ``G2_mask.csv``).
        time_val: Optional observation times. If None, ``np.arange(N)`` is used.
        target: Target name stored in ARVE object.
        stellar_parameters: Optional ARVE stellar parameters dictionary.
        resolution: Spectral resolution passed to ARVE.
        medium: Spectrum wavelength medium (``"vac"`` or ``"air"``).
        mask_medium: Mask wavelength medium (``"vac"`` or ``"air"``).
        vrad_grid: RV grid ``(min, max, step)`` in km/s.
        exclude_tellurics: Forwarded to ``compute_vrad_ccf``.
        ccf_err_scale: Forwarded to ``compute_vrad_ccf``.
        weight_name: Mask weight column name for ARVE.
        same_wave_grid: Forwarded to ARVE ``add_data``.

    Returns:
        Dictionary with ``ccf_vrad``, ``ccf_val``, ``ccf_err``, ``time_val``,
        ``vrad_val`` and ``vrad_err``.
    """
    if len(spectra) == 0:
        raise ValueError("spectra must contain at least one spectrum.")

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    parsed = [_parse_spectrum(spec) for spec in spectra]
    nspec = len(parsed)

    if time_val is None:
        time_val = np.arange(nspec, dtype=float)
    else:
        time_val = np.asarray(time_val, dtype=float)
        if time_val.shape[0] != nspec:
            raise ValueError("time_val length must match number of spectra.")

    import arve

    arve_obj = arve.ARVE()
    arve_obj.star.target = target
    if stellar_parameters is not None:
        arve_obj.star.stellar_parameters = stellar_parameters

    with TemporaryDirectory(prefix="arve_ccf_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        files = []
        for i, (wave, flux, flux_err) in enumerate(parsed):
            file_path = tmpdir_path / f"spectrum_{i:04d}.csv"
            _write_arve_s1d_csv(file_path, wave, flux, flux_err)
            files.append(str(file_path))

        arve_obj.data.add_data(
            time_val=time_val,
            files=files,
            format="s1d",
            extension="csv",
            medium=medium,
            resolution=resolution,
            same_wave_grid=same_wave_grid,
        )
        arve_obj.data.get_aux_data(mask_path=str(mask_path), mask_medium=mask_medium)
        arve_obj.data.compute_vrad_ccf(
            exclude_tellurics=exclude_tellurics,
            vrad_grid=list(vrad_grid),
            ccf_err_scale=ccf_err_scale,
            weight_name=weight_name,
        )

    ccf_vrad = np.asarray(arve_obj.data.ccf["ccf_vrad"])
    ccf_val = np.asarray(arve_obj.data.ccf["ccf_val"][:, -1, :])
    ccf_err = np.asarray(arve_obj.data.ccf["ccf_err"][:, -1, :])

    return {
        "ccf_vrad": ccf_vrad,
        "ccf_val": ccf_val,
        "ccf_err": ccf_err,
        "time_val": np.asarray(arve_obj.data.time["time_val"]),
        "vrad_val": np.asarray(arve_obj.data.vrad["vrad_val"]),
        "vrad_err": np.asarray(arve_obj.data.vrad["vrad_err"]),
    }


def _parse_spectrum(
    spectrum: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(spectrum, dict):
        wave = np.asarray(spectrum["wave"], dtype=float)
        flux = np.asarray(spectrum["flux"], dtype=float)
        flux_err_raw = spectrum.get("flux_err", None)
    elif isinstance(spectrum, tuple):
        if len(spectrum) == 2:
            wave, flux = spectrum
            flux_err_raw = None
        elif len(spectrum) == 3:
            wave, flux, flux_err_raw = spectrum
        else:
            raise ValueError("Tuple spectrum must be (wave, flux) or (wave, flux, flux_err).")
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
    else:
        raise TypeError("Spectrum must be a dict or tuple.")

    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError("wave and flux must be 1D arrays.")
    if wave.shape[0] != flux.shape[0]:
        raise ValueError("wave and flux must have the same length.")

    if flux_err_raw is None:
        # If uncertainty is unknown, use unit errors so ARVE can ingest s1d files.
        flux_err = np.ones_like(flux, dtype=float)
    else:
        flux_err = np.asarray(flux_err_raw, dtype=float)
        if flux_err.ndim != 1 or flux_err.shape[0] != flux.shape[0]:
            raise ValueError("flux_err must be a 1D array with same length as flux.")

    return wave, flux, flux_err


def _write_arve_s1d_csv(path: Path, wave: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> None:
    data = np.column_stack((wave, flux, flux_err))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="wave_val,flux_val,flux_err",
        comments="",
    )


__all__ = ["compute_ccf_from_spectra"]
