from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import SOAP
import SOAP.classes as sc

WavelengthMode = Literal["air", "vacuum"]


@dataclass
class SOAPSpectrumResult:
    wave: np.ndarray
    flux: np.ndarray
    ccf_flux: np.ndarray
    rv_ccf: np.ndarray
    sim: SOAP.Simulation
    output: object
    input_spectrum: sc.Spectrum


def create_spectrum_phoenix(
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
    prot: float = 25.0,
    radius: float = 1.0,
    mass: float = 1.0,
    start_psi: float = 0.0,
    u1: float | None = None,
    u2: float | None = None,
    teff_sig: float | None = None,
    logg_sig: float | None = None,
    z_sig: float | None = None,
    psi: list[float] | tuple[float, ...] | np.ndarray = (0,),
    grid: int = 300,
    inst_reso: int = 115000,
    diffrotB: float = 0.0,
    diffrotC: float = 0.0,
    active_regions: list | None = None,
    ring=None,
    resample_spectra: int = 1,
    interp_strategy: str = "spot2quiet",
    verbose: bool = False,
    skip_bis: bool = True,
    skip_fwhm: bool = True,
    skip_rv: bool = False,
) -> SOAPSpectrumResult:
    """Create an integrated SOAP spectrum and return spectrum and CCF products.

    Args:
        teff: Effective temperature (K).
        logg: Surface gravity (dex).
        z: Metallicity parameter Z used by SOAP/expecto.
        wave_min: Minimum wavelength for extraction.
        wave_max: Maximum wavelength for extraction.
        wavelength_mode: "air" (default SOAP behavior) or "vacuum".
        r: Instrumental resolving power for optional IP convolution of the input spectrum.
        cache: Forwarded to ``SOAP.classes.PHOENIX`` cache option.
        normalize: Forwarded to ``SOAP.classes.PHOENIX`` normalize option.
        normalize_max: If True, normalize the input spectrum by its maximum.
        prot: Stellar rotation period in days.
        radius: Stellar radius in solar radii.
        mass: Stellar mass in solar masses.
        start_psi: Starting stellar phase.
        u1: Linear quadratic limb-darkening coefficient.
        u2: Quadratic quadratic limb-darkening coefficient.
        teff_sig: Uncertainty in ``teff`` for optional LD coefficient estimation with ``ldtk``.
        logg_sig: Uncertainty in ``logg`` for optional LD coefficient estimation with ``ldtk``.
        z_sig: Uncertainty in ``z`` for optional LD coefficient estimation with ``ldtk``.
        psi: Rotation phases passed to ``sim.calculate_signal``.
        grid: SOAP stellar grid resolution.
        inst_reso: Spectrograph resolution used by SOAP.
        diffrotB: Linear differential rotation coefficient.
        diffrotC: Quadratic differential rotation coefficient.
        active_regions: Optional SOAP active regions.
        ring: Optional planet ring configuration.
        resample_spectra: SOAP spectrum resampling factor.
        interp_strategy: SOAP interpolation strategy when spot and quiet spectra differ.
        verbose: Forwarded to ``SOAP.Simulation``.
        skip_bis: Forwarded to ``sim.calculate_signal``.
        skip_fwhm: Forwarded to ``sim.calculate_signal``.
        skip_rv: Forwarded to ``sim.calculate_signal``.

    Returns:
        ``SOAPSpectrumResult`` with:
        - ``wave`` from ``sim.pixel.wave``
        - ``flux`` from ``sim.integrated_spectra``
        - ``ccf_flux`` from ``sim.ccf``
        - ``rv_ccf`` from ``sim.rv``
        - plus ``sim``, ``output``, and ``input_spectrum``
    """
    spec, wave, flux = create_spectrum_phoenix(
        teff=teff,
        logg=logg,
        z=z,
        wave_min=wave_min,
        wave_max=wave_max,
        wavelength_mode=wavelength_mode,
        r=r,
        cache=cache,
        normalize=normalize,
        normalize_max=normalize_max,
    )

    if u1 is None or u2 is None:
        if teff_sig is not None and logg_sig is not None and z_sig is not None:
            from ldtk import BoxcarFilter, LDPSetCreator

            filters = [BoxcarFilter("filter", wave_min / 10.0, wave_max / 10.0)]
            ld_creator = LDPSetCreator(
                teff=(teff, teff_sig),
                logg=(logg, logg_sig),
                z=(z, z_sig),
                filters=filters,
            )
            profiles = ld_creator.create_profiles()
            ld_coeffs, _ = profiles.coeffs_qd(do_mc=True)
            u1 = float(ld_coeffs[0][0])
            u2 = float(ld_coeffs[0][1])
        else:
            # Fall back to SOAP defaults when limb-darkening is not supplied.
            u1 = 0.29 if u1 is None else u1
            u2 = 0.34 if u2 is None else u2

    input_spectrum = sc.Spectrum(
        wave=np.asarray(wave, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
    )
    input_spectrum.flux = input_spectrum.flux / np.max(input_spectrum.flux)
    active_regions = [] if active_regions is None else active_regions
    pixel_spot = input_spectrum if active_regions else None

    sim = SOAP.Simulation(
        pixel=input_spectrum,
        pixel_spot=pixel_spot,
        inst_reso=inst_reso,
        grid=grid,
        active_regions=active_regions,
        ring=ring,
        resample_spectra=resample_spectra,
        interp_strategy=interp_strategy,
        verbose=verbose,
    )
    sim.planet.Rp = 0
    sim.star.set(
        prot=prot,
        u1=u1,
        u2=u2,
        start_psi=start_psi,
        radius=radius,
        mass=mass,
        teff=teff,
        diffrotB=diffrotB,
        diffrotC=diffrotC,
    )

    output = sim.calculate_signal(
        psi=np.atleast_1d(psi),
        skip_bis=skip_bis,
        skip_fwhm=skip_fwhm,
        skip_rv=skip_rv,
    )

    return SOAPSpectrumResult(
        wave=np.asarray(sim.pixel.wave, dtype=np.float64),
        flux=np.asarray(sim.integrated_spectra),
        ccf_flux=np.asarray(sim.ccf),
        rv_ccf=np.asarray(sim.rv),
        sim=sim,
        output=output,
        input_spectrum=input_spectrum,
    )


__all__ = ["SOAPSpectrumResult", "create_spectrum_phoenix", "create_spectrum_soap"]
