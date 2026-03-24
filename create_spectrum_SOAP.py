from __future__ import annotations

from typing import Literal

import numpy as np
import SOAP.classes as sc

WavelengthMode = Literal["air", "vacuum"]


def create_spectrum_soap(
    teff: float,
    logg: float,
    z: float,
    wave_min: float,
    wave_max: float,
    wavelength_mode: WavelengthMode = "air",
    r: int | None = None,
    cache: bool = False,
    normalize: bool = False,
    normalize_max: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a SOAP PHOENIX spectrum and return wavelength and flux arrays.

    Args:
        teff: Effective temperature (K).
        logg: Surface gravity (dex).
        z: Metallicity parameter Z used by SOAP/expecto.
        wave_min: Minimum wavelength for extraction.
        wave_max: Maximum wavelength for extraction.
        wavelength_mode: "air" (default SOAP behavior) or "vacuum".
        r: Instrumental resolving power for optional IP convolution.
        cache: Forwarded to ``SOAP.classes.PHOENIX`` cache option.
        normalize: Forwarded to ``SOAP.classes.PHOENIX`` normalize option.
        normalize_max: If True, apply ``flux = flux / np.max(flux)``.

    Returns:
        (wave, flux): 1D numpy arrays.
    """
    if wave_min >= wave_max:
        raise ValueError("wave_min must be smaller than wave_max.")
    if wavelength_mode not in ("air", "vacuum"):
        raise ValueError("wavelength_mode must be 'air' or 'vacuum'.")
    if r is not None and r <= 0:
        raise ValueError("r must be a positive integer when provided.")

    original_vacuum_to_air = sc.vacuum_to_air
    try:
        # SOAP converts PHOENIX vacuum wavelengths to air by default.
        # Override temporarily when vacuum output is requested.
        if wavelength_mode == "vacuum":
            sc.vacuum_to_air = lambda x: x

        spec = sc.PHOENIX(
            wave_range=(wave_min, wave_max),
            teff=teff,
            logg=logg,
            Z=z,
            cache=cache,
            normalize=normalize,
        )
    finally:
        sc.vacuum_to_air = original_vacuum_to_air

    wave = np.asarray(spec.wave, dtype=np.float64)
    flux = np.asarray(spec.flux, dtype=np.float64)

    if r is not None:
        wave, flux = sc.ip_convolution(wave, flux, R=int(r))

    if normalize_max:
        max_flux = np.max(flux)
        if max_flux == 0:
            raise ValueError("Cannot normalize by max flux because max(flux) == 0.")
        flux = flux / max_flux

    return spec, wave, flux 


__all__ = ["create_spectrum_soap"]
