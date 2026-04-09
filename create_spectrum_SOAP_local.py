from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import inspect

import numpy as np
import SOAP
import SOAP.classes as sc
import copy

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


def load_phoenix_from_npz(
    npz_path: str | Path,
    wave_key: str = "wave_phoenix",
    flux_key: str = "flux_phoenix",
) -> tuple[np.ndarray, np.ndarray]:
    """Load PHOENIX wavelength/flux arrays from a local NPZ file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"PHOENIX NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    if wave_key not in data.files or flux_key not in data.files:
        raise KeyError(
            f"Missing keys in {npz_path.name}. Required: '{wave_key}', '{flux_key}'. "
            f"Available: {list(data.files)}"
        )

    wave = np.asarray(data[wave_key], dtype=np.float64)
    flux = np.asarray(data[flux_key], dtype=np.float64)
    return wave, flux


def _resolve_u1_u2(
    wave_min: float,
    wave_max: float,
    teff: float,
    logg: float,
    z: float,
    u1: float | None,
    u2: float | None,
    teff_sig: float | None,
    logg_sig: float | None,
    z_sig: float | None,
) -> tuple[float, float]:
    if u1 is not None and u2 is not None:
        return float(u1), float(u2)

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
        coeffs, _ = profiles.coeffs_qd(do_mc=True)
        return float(coeffs[0][0]), float(coeffs[0][1])

    # SOAP fallback defaults
    return 0.29 if u1 is None else float(u1), 0.34 if u2 is None else float(u2)


def _ensure_soap_interpolate_compat() -> None:
    """Patch older/inconsistent SOAP installs where solarCCF lacks interpolate_to."""
    solar_ccf_cls = getattr(sc, "solarCCF", None)
    spectrum_cls = getattr(sc, "Spectrum", None)
    if solar_ccf_cls is None or spectrum_cls is None:
        return
    if hasattr(solar_ccf_cls, "interpolate_to"):
        return
    if not hasattr(spectrum_cls, "interpolate_to"):
        return

    def _interpolate_to(self, new_wave, inplace=True):
        # Variant A (newer/expected): spectrum-like object with wave/flux arrays.
        if hasattr(self, "wave") and hasattr(self, "flux"):
            return spectrum_cls.interpolate_to(self, new_wave, inplace=inplace)

        # Variant B (older/mixed installs): CCF-like object (e.g., rv/ccf only).
        # In this layout, forcing spectrum interpolation is invalid; keep object unchanged.
        if inplace:
            return self
        return copy.deepcopy(self)

    setattr(solar_ccf_cls, "interpolate_to", _interpolate_to)


def _build_simulation_compatible(
    *,
    pixel,
    pixel_spot,
    inst_reso: int,
    grid: int,
    active_regions,
    ring,
    resample_spectra: int,
    interp_strategy: str,
    verbose: bool,
):
    """Instantiate SOAP.Simulation with keyword compatibility across SOAP variants."""
    sig = inspect.signature(SOAP.Simulation)
    params = sig.parameters
    kwargs = {
        "pixel": pixel,
        "inst_reso": inst_reso,
        "grid": grid,
        "active_regions": active_regions,
        "ring": ring,
        "resample_spectra": resample_spectra,
        "interp_strategy": interp_strategy,
        "verbose": verbose,
    }
    # Different SOAP versions use one or the other.
    if "pixel_spot" in params:
        kwargs["pixel_spot"] = pixel_spot
    elif "pixel_ar" in params:
        kwargs["pixel_ar"] = pixel_spot
    return SOAP.Simulation(**kwargs)


def _set_star_compatible(
    star,
    *,
    prot: float,
    u1: float,
    u2: float,
    start_psi: float,
    radius: float,
    mass: float,
    teff: float,
    diffrotB: float,
    diffrotC: float,
) -> None:
    """Set star parameters across SOAP variants with differing Star.set signatures."""
    values = {
        "prot": float(prot),
        "u1": float(u1),
        "u2": float(u2),
        "start_psi": float(start_psi),
        "radius": float(radius),
        "mass": float(mass),
        "teff": float(teff),
        "diffrotB": float(diffrotB),
        "diffrotC": float(diffrotC),
    }
    alias = {
        "prot": ("prot", "Prot", "period", "rotation_period"),
        "u1": ("u1", "ld_u1", "limb_u1"),
        "u2": ("u2", "ld_u2", "limb_u2"),
        "start_psi": ("start_psi", "psi0"),
        "radius": ("radius", "R", "rstar"),
        "mass": ("mass", "M", "mstar"),
        "teff": ("teff", "Teff", "temperature"),
        "diffrotB": ("diffrotB",),
        "diffrotC": ("diffrotC",),
    }

    star_set = getattr(star, "set", None)
    if callable(star_set):
        sig = inspect.signature(star_set)
        params = sig.parameters
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {}
        for key, val in values.items():
            names = alias[key]
            if accepts_varkw:
                kwargs[names[0]] = val
            else:
                matched = next((n for n in names if n in params), None)
                if matched is not None:
                    kwargs[matched] = val
        if kwargs:
            star_set(**kwargs)

    # Ensure critical attributes exist with canonical names for SOAP internals.
    for key, val in values.items():
        try:
            setattr(star, key, val)
        except Exception:
            pass


def _apply_ld_calling_convention_compat(star) -> None:
    """Adapt star limb-darkening attrs for SOAP builds using ld(y,z,coeffs,law)."""
    fast_mod = getattr(SOAP, "fast_starspot", None)
    if fast_mod is None:
        return
    ld_obj = getattr(fast_mod, "ld", None)
    if ld_obj is None:
        return

    # Numba dispatchers expose the python implementation at .py_func.
    ld_func = getattr(ld_obj, "py_func", ld_obj)
    try:
        params = list(inspect.signature(ld_func).parameters.keys())
    except (TypeError, ValueError):
        return

    # Newer SOAP API: ld(y, z, coeffs, law)
    if len(params) >= 4 and params[2] == "coeffs" and params[3] == "law":
        u1_val = float(np.asarray(getattr(star, "u1", 0.29)).reshape(-1)[0])
        u2_val = float(np.asarray(getattr(star, "u2", 0.34)).reshape(-1)[0])
        setattr(star, "u1", np.array([u1_val, u2_val], dtype=np.float64))
        setattr(star, "u2", np.int64(2))


def create_spectrum_soap_from_arrays(
    wave: np.ndarray,
    flux: np.ndarray,
    *,
    teff: float = 5777.0,
    prot: float = 25.0,
    radius: float = 1.0,
    mass: float = 1.0,
    start_psi: float = 0.0,
    u1: float | None = None,
    u2: float | None = None,
    teff_sig: float | None = None,
    logg: float = 4.44,
    logg_sig: float | None = None,
    z: float = 0.0,
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
    normalize_max: bool = True,
) -> SOAPSpectrumResult:
    """Create integrated SOAP spectrum/CCF from in-memory PHOENIX wave+flux arrays."""
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError("wave and flux must be 1D arrays.")
    if len(wave) != len(flux):
        raise ValueError("wave and flux must have the same length.")

    wave_min = float(np.min(wave))
    wave_max = float(np.max(wave))
    u1v, u2v = _resolve_u1_u2(
        wave_min=wave_min,
        wave_max=wave_max,
        teff=teff,
        logg=logg,
        z=z,
        u1=u1,
        u2=u2,
        teff_sig=teff_sig,
        logg_sig=logg_sig,
        z_sig=z_sig,
    )

    input_spectrum = sc.Spectrum(wave=wave.copy(), flux=flux.copy())
    if normalize_max:
        fmax = np.max(input_spectrum.flux)
        if fmax == 0:
            raise ValueError("Cannot normalize by max flux because max(flux) == 0.")
        input_spectrum.flux = input_spectrum.flux / fmax

    _ensure_soap_interpolate_compat()
    active_regions = [] if active_regions is None else active_regions
    pixel_spot = input_spectrum if active_regions else None

    sim = _build_simulation_compatible(
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
    _set_star_compatible(
        sim.star,
        prot=prot,
        u1=u1v,
        u2=u2v,
        start_psi=start_psi,
        radius=radius,
        mass=mass,
        teff=teff,
        diffrotB=diffrotB,
        diffrotC=diffrotC,
    )
    _apply_ld_calling_convention_compat(sim.star)

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


def create_spectrum_soap_from_saved_phoenix(
    phoenix_npz_path: str | Path,
    *,
    wave_key: str = "wave_phoenix",
    flux_key: str = "flux_phoenix",
    **kwargs,
) -> SOAPSpectrumResult:
    """Load local PHOENIX arrays from NPZ and run SOAP (no online PHOENIX download)."""
    wave, flux = load_phoenix_from_npz(
        npz_path=phoenix_npz_path,
        wave_key=wave_key,
        flux_key=flux_key,
    )
    return create_spectrum_soap_from_arrays(wave=wave, flux=flux, **kwargs)


__all__ = [
    "SOAPSpectrumResult",
    "load_phoenix_from_npz",
    "create_spectrum_soap_from_arrays",
    "create_spectrum_soap_from_saved_phoenix",
]
