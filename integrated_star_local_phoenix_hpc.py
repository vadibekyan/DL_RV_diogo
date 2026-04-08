from __future__ import annotations

"""
HPC workflow for SOAP spectra generated from a saved PHOENIX spectrum.

Main modes:

1. Generate SOAP spectra only:
   python integrated_star_local_phoenix_hpc.py \
     --phoenix-npz /projects/F202418352CPCAA1/saved_spectra/phoenix_output.npz \
     --save-dir /projects/F202418352CPCAA1/saved_spectra \
     --grid

2. Generate SOAP spectra and immediately calculate iCCF CCFs + indicators:
   python integrated_star_local_phoenix_hpc.py \
     --phoenix-npz /projects/F202418352CPCAA1/saved_spectra/phoenix_output.npz \
     --save-dir /projects/F202418352CPCAA1/saved_spectra \
     --grid \
     --ccf-backend iccf

3. Calculate iCCF CCFs + indicators from already-generated spectrum CSVs:
   python integrated_star_local_phoenix_hpc.py \
     --save-dir /projects/F202418352CPCAA1/saved_spectra \
     --ccf-only

4. Generate a quiet SOAP spectrum with no active regions:
   python integrated_star_local_phoenix_hpc.py \
     --phoenix-npz /projects/F202418352CPCAA1/saved_spectra/phoenix_output.npz \
     --save-dir /projects/F202418352CPCAA1/saved_spectra \
     --quiet

Default CCF-only input:
   <save-dir>/csv_spectra/soap_spectrum_*.csv

Default outputs:
   SOAP NPZ:        <save-dir>/soap_output_<active_region_label>.npz
   SOAP spectrum:   <save-dir>/csv_spectra/soap_spectrum_<active_region_label>.csv
   iCCF CCF:        <save-dir>/ccf_spectra/soap_spectrum_<active_region_label>_iccf.csv
   iCCF indicators: <save-dir>/ccf_spectra/soap_spectrum_<active_region_label>_iccf_indicators.csv

Wavelength-medium defaults for iCCF:
   SOAP spectrum CSV: vacuum
   iCCF mask:         air
   CCF calculation:   vacuum
"""

import argparse
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - useful on stripped-down HPC envs.
    tqdm = lambda x, **_: x


DEFAULT_ACTIVE_REGION_CONFIGS = [
    {
        "lon": 45.0,
        "lat": 0.0,
        "size": 0.05,
        "active_region_type": "spot",
        "temp_diff": 600.0,
        "check": True,
    }
]


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def label_number(value: float, *, signed: bool = False) -> str:
    value = float(value)
    sign = ""
    if signed:
        sign = "m" if value < 0 else "p"
    text = f"{abs(value):g}".replace(".", "p")
    return f"{sign}{text}"


def triangular_active_region_size(
    t: float | np.ndarray,
    size_max: float,
    tmax: float,
    lifetime: float,
    growth_fraction: float = 0.1,
) -> float | np.ndarray:
    a = tmax - growth_fraction * lifetime
    b = tmax + (1.0 - growth_fraction) * lifetime
    c = tmax
    t = np.atleast_1d(t).astype(float)

    growing = (t >= a) & (t < c)
    decaying = (t >= c) & (t <= b)
    size = np.zeros_like(t, dtype=float)
    size[growing] = size_max * (t[growing] - a) / (c - a)
    size[decaying] = size_max * (b - t[decaying]) / (b - c)

    if size.size == 1:
        return float(size[0])
    return size


def resolve_active_region_configs(
    active_region_configs: list[dict[str, Any]],
    *,
    time: float | None = None,
) -> list[dict[str, Any]]:
    resolved_configs = []

    for config in active_region_configs:
        config = dict(config)
        size_evolution = config.pop("size_evolution", None)

        if size_evolution is not None:
            if time is None:
                raise ValueError("active_region_time is required when size_evolution is set.")
            if size_evolution.get("model", "triangular") != "triangular":
                raise ValueError("Only triangular size_evolution is currently supported.")
            config["size"] = triangular_active_region_size(
                time,
                size_max=size_evolution["size_max"],
                tmax=size_evolution["tmax"],
                lifetime=size_evolution["lifetime"],
                growth_fraction=size_evolution.get("growth_fraction", 0.1),
            )

        resolved_configs.append(config)

    return resolved_configs


def build_active_regions(
    active_region_configs: list[dict[str, Any]],
    *,
    time: float | None = None,
) -> list:
    import SOAP

    active_regions = []

    for config in resolve_active_region_configs(active_region_configs, time=time):
        active_regions.append(
            SOAP.ActiveRegion(
                lon=config.get("lon", config.get("longitude", 180.0)),
                lat=config.get("lat", config.get("latitude", 0.0)),
                size=config.get("size", 0.1),
                active_region_type=config.get("active_region_type", config.get("type", "spot")),
                temp_diff=config.get("temp_diff", 663.0),
                check=config.get("check", True),
            )
        )

    return active_regions


def active_region_label(active_region_configs: list[dict[str, Any]]) -> str:
    if not active_region_configs:
        return "quiet"

    parts = []
    for i, config in enumerate(active_region_configs, start=1):
        kind = config.get("active_region_type", config.get("type", "spot"))
        lon = label_number(config.get("lon", config.get("longitude", 180.0)), signed=True)
        lat = label_number(config.get("lat", config.get("latitude", 0.0)), signed=True)
        temp = label_number(config.get("temp_diff", 663.0))

        if "size" in config:
            size = label_number(config["size"])
        elif "size_evolution" in config:
            size = label_number(config["size_evolution"]["size_max"])
        else:
            size = label_number(0.1)

        parts.append(f"ar{i}_{kind}_lon{lon}_lat{lat}_size{size}_dT{temp}")

    return "__".join(parts)


def make_single_spot_grid_configs(
    sizes: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
    temp_diffs: tuple[float, ...] = (200.0, 400.0, 600.0, 800.0),
    lons: tuple[float, ...] = (-80.0, -40.0, -10.0, 10.0, 40.0, 80.0),
    lat: float = 0.0,
    active_region_type: str = "spot",
) -> list[dict[str, Any]]:
    grid_configs = []

    for size in sizes:
        for temp_diff in temp_diffs:
            for lon in lons:
                config = {
                    "lon": float(lon),
                    "lat": float(lat),
                    "size": float(size),
                    "active_region_type": active_region_type,
                    "temp_diff": float(temp_diff),
                    "check": True,
                }
                grid_configs.append(
                    {
                        "run_label": active_region_label([config]),
                        "active_region_configs": [config],
                    }
                )

    return grid_configs


def save_integrated_spectrum_csv(
    path: str | Path,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: float = 0.001,
) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum_df = pd.DataFrame(
        {
            "wave_val": np.asarray(wave, dtype=float),
            "flux_val": np.asarray(flux, dtype=float),
            "flux_err": np.full(len(wave), float(flux_err), dtype=float),
        }
    )
    spectrum_df.to_csv(path, index=False)
    return spectrum_df


def save_integrated_spectra_csvs(
    soap_result,
    csv_dir: str | Path,
    run_label: str,
    flux_err: float = 0.001,
) -> list[Path]:
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    wave = np.asarray(soap_result.wave, dtype=float)
    flux = np.asarray(soap_result.flux, dtype=float)
    if flux.ndim == 1:
        flux = flux.reshape(1, -1)

    csv_paths = []
    for i, flux_i in enumerate(flux):
        suffix = "" if len(flux) == 1 else f"_phase{i:03d}"
        csv_path = csv_dir / f"soap_spectrum_{run_label}{suffix}.csv"
        save_integrated_spectrum_csv(csv_path, wave, flux_i, flux_err=flux_err)
        csv_paths.append(csv_path)

    return csv_paths


def calculate_iccf_from_csv(
    spectrum_csv_path: str | Path,
    *,
    output_csv_path: str | Path,
    mask_name: str = "G2",
    mask_instrument: str = "ESPRESSO",
    spectrum_medium: str = "vacuum",
    mask_medium: str = "air",
    ccf_medium: str = "vacuum",
    rvarray: np.ndarray | None = None,
    mask_width: float = 0.5,
) -> dict[str, Any]:
    try:
        import iCCF
        from iCCF.meta import espdr_compute_CCF_fast
    except ImportError as exc:
        raise ImportError(
            "iCCF is required for --ccf-backend iccf. Install it in the HPC "
            "environment before using this option."
        ) from exc

    valid_media = {"air", "vacuum"}
    if spectrum_medium not in valid_media:
        raise ValueError("spectrum_medium must be 'air' or 'vacuum'.")
    if mask_medium not in valid_media:
        raise ValueError("mask_medium must be 'air' or 'vacuum'.")
    if ccf_medium not in valid_media:
        raise ValueError("ccf_medium must be 'air' or 'vacuum'.")

    def convert_wavelength_medium(wavelength, source_medium, target_medium):
        wavelength = np.asarray(wavelength, dtype=float)
        if source_medium == target_medium:
            return wavelength
        try:
            from airvacuumvald import air_to_vacuum, vacuum_to_air
        except ImportError as exc:
            raise ImportError(
                "airvacuumvald is required to convert between air and vacuum wavelengths."
            ) from exc

        if source_medium == "air" and target_medium == "vacuum":
            return air_to_vacuum(wavelength)
        if source_medium == "vacuum" and target_medium == "air":
            return vacuum_to_air(wavelength)
        raise ValueError(f"Unsupported wavelength conversion: {source_medium} -> {target_medium}")

    spec = pd.read_csv(spectrum_csv_path)
    wave = spec["wave_val"].to_numpy(float)
    flux = spec["flux_val"].to_numpy(float)
    err = spec["flux_err"].to_numpy(float)

    wave = convert_wavelength_medium(wave, spectrum_medium, ccf_medium)
    dll = np.diff(wave)
    dll = np.r_[dll, dll[-1]]
    blaze = np.ones_like(wave)
    quality = np.zeros_like(wave, dtype=int)
    if rvarray is None:
        rvarray = np.arange(-30.0, 30.0 + 0.1, 0.1)

    mask_obj = iCCF.Mask(mask_name, instrument=mask_instrument)
    mask_wavelength = convert_wavelength_medium(
        mask_obj.wavelength,
        mask_medium,
        ccf_medium,
    )

    mask = np.zeros(mask_obj.nlines, dtype=[("lambda", "f8"), ("contrast", "f8")])
    mask["lambda"] = mask_wavelength
    mask["contrast"] = mask_obj.contrast
    mask.sort(order="lambda")

    ccf, ccfe, ccfq = espdr_compute_CCF_fast(
        wave,
        dll,
        flux,
        err,
        blaze,
        quality,
        rvarray,
        mask,
        berv=0.0,
        bervmax=0.0,
        mask_width=mask_width,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ccf_vrad": rvarray,
            "ccf_val": ccf,
            "ccf_err": ccfe,
            "ccf_quality": ccfq,
        }
    ).to_csv(output_csv_path, index=False)

    I = iCCF.Indicators(rvarray, ccf, ccfe)
    indicators = {
        "RV [km/s]": I.RV * 1000,
        "FWHM [km/s]": I.FWHM,
        "BIS [km/s]": I.BIS * 1000,
        "Vspan [km/s]": I.Vspan * 1000,
        "Wspan [km/s]": I.Wspan * 1000,
        "contrast [%]": I.contrast,
    }
    indicators_csv_path = output_csv_path.with_name(
        f"{output_csv_path.stem}_indicators.csv"
    )
    pd.Series(indicators).to_csv(indicators_csv_path, header=["value"])

    return {
        "ccf_csv": output_csv_path,
        "indicators_csv": indicators_csv_path,
        "ccf_vrad": rvarray,
        "ccf_val": ccf,
        "ccf_err": ccfe,
        "ccf_quality": ccfq,
        "indicators": indicators,
    }


def run_local_phoenix_workflow(
    phoenix_npz_path: str | Path = "./saved_spectra/phoenix_output.npz",
    save_dir: str | Path = "./saved_spectra",
    save_csv: bool = True,
    active_region_configs: list[dict[str, Any]] | None = None,
    active_region_time: float | None = None,
    psi_values: tuple[float, ...] = (0.0,),
    run_label: str | None = None,
    ccf_backend: str = "none",
    iccf_mask_name: str = "G2",
    iccf_mask_instrument: str = "ESPRESSO",
    iccf_spectrum_medium: str = "vacuum",
    iccf_mask_medium: str = "air",
    iccf_ccf_medium: str = "vacuum",
    iccf_mask_width: float = 0.5,
    iccf_rv_start: float = -30.0,
    iccf_rv_stop: float = 30.0,
    iccf_rv_step: float = 0.1,
) -> dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from create_spectrum_SOAP_local import (
        load_phoenix_from_npz,
        create_spectrum_soap_from_saved_phoenix,
    )

    wave_phoenix, flux_phoenix = load_phoenix_from_npz(
        npz_path=phoenix_npz_path,
        wave_key="wave_phoenix",
        flux_key="flux_phoenix",
    )

    if active_region_configs is None:
        active_region_configs = DEFAULT_ACTIVE_REGION_CONFIGS
    resolved_active_region_configs = resolve_active_region_configs(
        active_region_configs,
        time=active_region_time,
    )
    active_regions = build_active_regions(
        active_region_configs,
        time=active_region_time,
    )
    if run_label is None:
        run_label = active_region_label(resolved_active_region_configs)

    soap_result = create_spectrum_soap_from_saved_phoenix(
        phoenix_npz_path=phoenix_npz_path,
        teff=5777.0,
        logg=4.44,
        z=0.0,
        psi=psi_values,
        grid=300,
        inst_reso=115000,
        active_regions=active_regions,
        skip_bis=True,
        skip_fwhm=True,
        skip_rv=False,
        verbose=False,
    )

    soap_npz_path = save_dir / f"soap_output_{run_label}.npz"
    np.savez(
        soap_npz_path,
        wave_soap=np.asarray(soap_result.wave),
        flux_soap=np.asarray(soap_result.flux),
        ccf_flux_soap=np.asarray(soap_result.ccf_flux),
        rv_ccf_soap=np.asarray(soap_result.rv_ccf),
        active_region_configs=np.asarray(resolved_active_region_configs, dtype=object),
        spectrum_soap_wave=np.asarray(soap_result.input_spectrum.wave),
        spectrum_soap_flux=np.asarray(soap_result.input_spectrum.flux),
    )

    soap_csv_paths = []
    if save_csv:
        soap_csv_paths = save_integrated_spectra_csvs(
            soap_result,
            save_dir / "csv_spectra",
            run_label,
        )

    ccf_results = []
    if ccf_backend == "iccf":
        rvarray = np.arange(iccf_rv_start, iccf_rv_stop + iccf_rv_step, iccf_rv_step)
        for spectrum_csv_path in soap_csv_paths:
            ccf_csv_path = save_dir / "ccf_spectra" / f"{spectrum_csv_path.stem}_iccf.csv"
            ccf_results.append(
                calculate_iccf_from_csv(
                    spectrum_csv_path,
                    output_csv_path=ccf_csv_path,
                    mask_name=iccf_mask_name,
                    mask_instrument=iccf_mask_instrument,
                    spectrum_medium=iccf_spectrum_medium,
                    mask_medium=iccf_mask_medium,
                    ccf_medium=iccf_ccf_medium,
                    mask_width=iccf_mask_width,
                    rvarray=rvarray,
                )
            )
    elif ccf_backend != "none":
        raise ValueError("--ccf-backend must be 'none' or 'iccf'.")

    return {
        "wave_phoenix": wave_phoenix,
        "flux_phoenix": flux_phoenix,
        "soap_result": soap_result,
        "soap_npz": soap_npz_path,
        "soap_csv": soap_csv_paths,
        "ccf_results": ccf_results,
        "run_label": run_label,
        "active_region_configs": resolved_active_region_configs,
    }


def run_single_spot_grid(
    *,
    phoenix_npz_path: str | Path,
    save_dir: str | Path,
    sizes: tuple[float, ...],
    temp_diffs: tuple[float, ...],
    lons: tuple[float, ...],
    lat: float,
    max_runs: int | None,
    **workflow_kwargs,
) -> list[dict[str, Any]]:
    grid_configs = make_single_spot_grid_configs(
        sizes=sizes,
        temp_diffs=temp_diffs,
        lons=lons,
        lat=lat,
    )
    if max_runs is not None:
        grid_configs = grid_configs[:max_runs]

    results = []
    for item in tqdm(grid_configs, desc="SOAP single-spot grid"):
        results.append(
            run_local_phoenix_workflow(
                phoenix_npz_path=phoenix_npz_path,
                save_dir=save_dir,
                active_region_configs=item["active_region_configs"],
                run_label=item["run_label"],
                **workflow_kwargs,
            )
        )

    return results


def run_iccf_for_existing_spectra(
    *,
    spectrum_csv_paths: list[Path],
    save_dir: str | Path,
    iccf_mask_name: str,
    iccf_mask_instrument: str,
    iccf_spectrum_medium: str,
    iccf_mask_medium: str,
    iccf_ccf_medium: str,
    iccf_mask_width: float,
    iccf_rv_start: float,
    iccf_rv_stop: float,
    iccf_rv_step: float,
) -> list[dict[str, Any]]:
    save_dir = Path(save_dir)
    rvarray = np.arange(iccf_rv_start, iccf_rv_stop + iccf_rv_step, iccf_rv_step)

    ccf_results = []
    for spectrum_csv_path in tqdm(spectrum_csv_paths, desc="iCCF existing spectra"):
        ccf_csv_path = save_dir / "ccf_spectra" / f"{spectrum_csv_path.stem}_iccf.csv"
        ccf_results.append(
            calculate_iccf_from_csv(
                spectrum_csv_path,
                output_csv_path=ccf_csv_path,
                mask_name=iccf_mask_name,
                mask_instrument=iccf_mask_instrument,
                spectrum_medium=iccf_spectrum_medium,
                mask_medium=iccf_mask_medium,
                ccf_medium=iccf_ccf_medium,
                mask_width=iccf_mask_width,
                rvarray=rvarray,
            )
        )

    return ccf_results


def inspect_local_soap_npz(soap_npz_path: str | Path) -> None:
    soap_npz_path = Path(soap_npz_path)
    if not soap_npz_path.exists():
        print(f"No local SOAP NPZ found: {soap_npz_path}")
        return

    data = np.load(soap_npz_path, allow_pickle=True)
    print("Available keys:", list(data.files))
    for key in data.files:
        arr = np.asarray(data[key])
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SOAP using a locally saved PHOENIX NPZ (no online PHOENIX download)."
    )
    parser.add_argument(
        "--phoenix-npz",
        default="./saved_spectra/phoenix_output.npz",
        help="Path to local PHOENIX NPZ file.",
    )
    parser.add_argument(
        "--save-dir",
        default="./saved_spectra",
        help="Directory to save SOAP NPZ, CSV spectra, and optional CCF output.",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="If set, do not write SOAP CSV spectrum files.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run the full single-spot grid instead of one default active-region case.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run one quiet-star SOAP simulation with no active regions.",
    )
    parser.add_argument(
        "--sizes",
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated active-region sizes for --grid.",
    )
    parser.add_argument(
        "--temp-diffs",
        default="200,400,600,800",
        help="Comma-separated active-region temperature differences [K] for --grid.",
    )
    parser.add_argument(
        "--lons",
        default="-80,-40,-10,10,40,80",
        help="Comma-separated active-region longitudes [deg] for --grid.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=0.0,
        help="Active-region latitude [deg] for the single-spot grid.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of grid runs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--ccf-backend",
        choices=("none", "iccf"),
        default="none",
        help="Optional CCF backend to run after saving CSV spectra.",
    )
    parser.add_argument(
        "--ccf-only",
        action="store_true",
        help="Skip SOAP and calculate iCCF CCFs from already-generated spectrum CSVs.",
    )
    parser.add_argument(
        "--ccf-input-glob",
        default=None,
        help=(
            "Glob for existing spectrum CSVs used with --ccf-only. "
            "Defaults to <save-dir>/csv_spectra/soap_spectrum_*.csv."
        ),
    )
    parser.add_argument("--iccf-mask-name", default="G2")
    parser.add_argument("--iccf-mask-instrument", default="ESPRESSO")
    parser.add_argument(
        "--iccf-spectrum-medium",
        choices=("vacuum", "air"),
        default="vacuum",
        help="Wavelength medium of the input SOAP spectrum CSV.",
    )
    parser.add_argument(
        "--iccf-mask-medium",
        choices=("vacuum", "air"),
        default="air",
        help="Wavelength medium of the iCCF mask.",
    )
    parser.add_argument(
        "--iccf-ccf-medium",
        choices=("vacuum", "air"),
        default="vacuum",
        help="Wavelength medium to use internally for iCCF calculation.",
    )
    parser.add_argument("--iccf-mask-width", type=float, default=0.5)
    parser.add_argument("--iccf-rv-start", type=float, default=-30.0)
    parser.add_argument("--iccf-rv-stop", type=float, default=30.0)
    parser.add_argument("--iccf-rv-step", type=float, default=0.1)
    parser.add_argument(
        "--inspect-only",
        type=str,
        default=None,
        metavar="NPZ",
        help="Inspect an existing SOAP NPZ and exit.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.inspect_only:
        inspect_local_soap_npz(args.inspect_only)
        return

    iccf_kwargs = {
        "iccf_mask_name": args.iccf_mask_name,
        "iccf_mask_instrument": args.iccf_mask_instrument,
        "iccf_spectrum_medium": args.iccf_spectrum_medium,
        "iccf_mask_medium": args.iccf_mask_medium,
        "iccf_ccf_medium": args.iccf_ccf_medium,
        "iccf_mask_width": args.iccf_mask_width,
        "iccf_rv_start": args.iccf_rv_start,
        "iccf_rv_stop": args.iccf_rv_stop,
        "iccf_rv_step": args.iccf_rv_step,
    }

    if args.ccf_only:
        pattern = args.ccf_input_glob
        if pattern is None:
            pattern = str(Path(args.save_dir) / "csv_spectra" / "soap_spectrum_*.csv")
        spectrum_csv_paths = [Path(p) for p in sorted(glob(pattern))]
        if not spectrum_csv_paths:
            raise FileNotFoundError(f"No spectrum CSV files matched: {pattern}")

        ccf_results = run_iccf_for_existing_spectra(
            spectrum_csv_paths=spectrum_csv_paths,
            save_dir=args.save_dir,
            **iccf_kwargs,
        )
        print(f"Finished {len(ccf_results)} iCCF calculations.")
        if ccf_results:
            print("First saved CCF:", ccf_results[0]["ccf_csv"])
            print("First saved indicators:", ccf_results[0]["indicators_csv"])
            print("Last saved CCF:", ccf_results[-1]["ccf_csv"])
            print("Last saved indicators:", ccf_results[-1]["indicators_csv"])
        return

    if args.quiet and args.grid:
        raise ValueError("Use either --quiet or --grid, not both.")

    common_kwargs = {
        "save_csv": not args.no_save_csv,
        "ccf_backend": args.ccf_backend,
        **iccf_kwargs,
    }

    if args.grid:
        results = run_single_spot_grid(
            phoenix_npz_path=args.phoenix_npz,
            save_dir=args.save_dir,
            sizes=parse_float_list(args.sizes),
            temp_diffs=parse_float_list(args.temp_diffs),
            lons=parse_float_list(args.lons),
            lat=args.lat,
            max_runs=args.max_runs,
            **common_kwargs,
        )
        print(f"Finished {len(results)} runs.")
        if results:
            print("First saved NPZ:", results[0]["soap_npz"])
            print("Last saved NPZ:", results[-1]["soap_npz"])
        return

    result = run_local_phoenix_workflow(
        phoenix_npz_path=args.phoenix_npz,
        save_dir=args.save_dir,
        active_region_configs=[] if args.quiet else None,
        **common_kwargs,
    )

    print("Saved:")
    print(" -", result["soap_npz"])
    for csv_path in result["soap_csv"]:
        print(" -", csv_path)
    for ccf_result in result["ccf_results"]:
        print(" -", ccf_result["ccf_csv"])
        print(" -", ccf_result["indicators_csv"])

    inspect_local_soap_npz(result["soap_npz"])


if __name__ == "__main__":
    main()
