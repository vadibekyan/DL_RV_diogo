from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STELLAR_PARAMETERS: dict[str, float | str] = {
    "sptype": "G2",
    "vrad_sys": 0.0,
    "berv_max": 1.0,
    "Teff": 5777,
    "logg": 4.44,
    "Fe_H": 0.0,
    "M": 1.0,
    "R": 1.0,
    "vsini": 1.63,
    "vmic": 0.85,
    "vmac": 3.98,
}


def create_arve_ccf(
    input_spectrum: str | Path,
    mask_path: str | Path,
    *,
    resolution: float = 100000,
    medium: str = "vac",
    mask_medium: str = "vac",
    vrad_grid: tuple[float, float, float] = (-20.0, 20.0, 0.1),
    target: str = "Sun",
    stellar_parameters: dict[str, Any] | None = None,
    exclude_tellurics: bool = False,
    ccf_err_scale: bool = True,
    same_wave_grid: bool = False,
) -> dict[str, Any]:
    """Compute one ARVE CCF from one input spectrum CSV and one mask CSV.

    The input spectrum file must be an ARVE-compatible ``s1d`` CSV with
    ``wave_val``, ``flux_val``, and ``flux_err`` columns.
    """
    input_spectrum = Path(input_spectrum)
    mask_path = Path(mask_path)

    if not input_spectrum.exists():
        raise FileNotFoundError(f"Input spectrum file not found: {input_spectrum}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    import arve

    example = arve.ARVE()
    example.star.target = target
    example.star.stellar_parameters = dict(
        DEFAULT_STELLAR_PARAMETERS if stellar_parameters is None else stellar_parameters
    )

    files = [str(input_spectrum)]
    time_val = np.arange(len(files), dtype=float)

    example.data.add_data(
        time_val=time_val,
        files=files,
        format="s1d",
        extension="csv",
        medium=medium,
        resolution=resolution,
        same_wave_grid=same_wave_grid,
    )
    example.data.get_aux_data(mask_path=str(mask_path), mask_medium=mask_medium)
    example.data.compute_vrad_ccf(
        exclude_tellurics=exclude_tellurics,
        vrad_grid=list(vrad_grid),
        ccf_err_scale=ccf_err_scale,
    )

    return {
        "example": example,
        "files": files,
        "ccf_vrad": np.asarray(example.data.ccf["ccf_vrad"]),
        "ccf_val": np.asarray(example.data.ccf["ccf_val"][:, -1, :]),
        "ccf_err": np.asarray(example.data.ccf["ccf_err"][:, -1, :]),
        "time_val": np.asarray(example.data.time["time_val"]),
        "vrad_val": np.asarray(example.data.vrad["vrad_val"]),
        "vrad_err": np.asarray(example.data.vrad["vrad_err"]),
    }


def compute_ccf_from_spectra(
    input_spectrum: str | Path,
    mask_path: str | Path = "G2_mask.csv",
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible wrapper around ``create_arve_ccf``."""
    return create_arve_ccf(input_spectrum=input_spectrum, mask_path=mask_path, **kwargs)


__all__ = ["DEFAULT_STELLAR_PARAMETERS", "create_arve_ccf", "compute_ccf_from_spectra"]
