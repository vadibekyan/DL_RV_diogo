from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


C_KMS = 299792.458


def apply_rv_shift_on_original_grid(
    wave,
    flux,
    rv,
    *,
    flux_err=None,
    fill_value=1.0,
):
    """Apply an RV Doppler shift and resample the shifted spectrum onto the original grid.

    Parameters
    ----------
    wave : array-like
        Original wavelength grid.
    flux : array-like
        Flux values on the original wavelength grid.
    rv : float
        Radial velocity in km/s. Positive values redshift the spectrum.
    flux_err : array-like or None, optional
        Flux uncertainties. If given, they are interpolated in the same way.
    fill_value : float, optional
        Value used outside the interpolation range.

    Returns
    -------
    tuple
        ``(wave_out, flux_out)`` or ``(wave_out, flux_out, flux_err_out)`` if
        ``flux_err`` is provided.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError("wave and flux must be 1D arrays.")
    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape.")

    beta = rv / C_KMS
    if abs(beta) >= 1.0:
        raise ValueError("rv must be smaller than the speed of light.")

    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    wave_shifted = wave * doppler_factor

    flux_out = np.interp(
        wave,
        wave_shifted,
        flux,
        left=fill_value,
        right=fill_value,
    )

    if flux_err is None:
        return wave.copy(), flux_out

    flux_err = np.asarray(flux_err, dtype=float)
    if flux_err.ndim != 1 or flux_err.shape != flux.shape:
        raise ValueError("flux_err must be a 1D array with the same shape as flux.")

    flux_err_out = np.interp(
        wave,
        wave_shifted,
        flux_err,
        left=fill_value,
        right=fill_value,
    )

    #return wave.copy(), flux_out, flux_err_out
    return wave_shifted, flux, flux_err


def shift_spectrum_csv(
    input_csv,
    output_csv,
    rv,
    *,
    fill_value=1.0,
    default_flux_err=0.001,
):
    """Read a CSV spectrum, apply an RV shift, and save a new CSV on the original grid.

    The input CSV must contain ``wave_val`` and ``flux_val`` columns. If
    ``flux_err`` is missing, a constant uncertainty is written using
    ``default_flux_err``.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    if "wave_val" not in df.columns or "flux_val" not in df.columns:
        raise ValueError("input_csv must contain 'wave_val' and 'flux_val' columns.")

    wave = df["wave_val"].to_numpy(dtype=float)
    flux = df["flux_val"].to_numpy(dtype=float)

    if "flux_err" in df.columns:
        flux_err = df["flux_err"].to_numpy(dtype=float)
        wave_out, flux_out, flux_err_out = apply_rv_shift_on_original_grid(
            wave,
            flux,
            rv,
            flux_err=flux_err,
            fill_value=fill_value,
        )
    else:
        wave_out, flux_out = apply_rv_shift_on_original_grid(
            wave,
            flux,
            rv,
            fill_value=fill_value,
        )
        flux_err_out = np.full_like(flux_out, default_flux_err, dtype=float)

    df_out = pd.DataFrame(
        {
            "wave_val": wave_out,
            "flux_val": flux_out,
            "flux_err": flux_err_out,
        }
    )

    df_out.to_csv(output_csv, index=False)
    return df_out


__all__ = [
    "C_KMS",
    "apply_rv_shift_on_original_grid",
    "shift_spectrum_csv",
]
