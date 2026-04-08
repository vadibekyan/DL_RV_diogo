from __future__ import annotations

"""
Generate a random-RV iCCF dataset from one saved SOAP spectrum CSV.

Each output row contains:
   rv_true_mps, iCCF indicators, ccf_0000, ccf_0001, ...

The shifted spectra and individual CCF files are not saved.

Example:
   python generate_random_rv_iccf_dataset_parallel_hpc.py \
     --input-spectrum /projects/F202418352CPCAA1/saved_spectra/csv_spectra/soap_spectrum_ar1_spot_lonm80_latp0_size0p05_dT200.csv \
     --output-csv /projects/F202418352CPCAA1/saved_spectra/random_rv_iccf_dataset.csv \
     --n-samples 1000 \
     --rv-min-mps -100 \
     --rv-max-mps 100 \
     --n-workers 48

Batch example:
   python generate_random_rv_iccf_dataset_parallel_hpc.py \
     --input-glob "/projects/F202418352CPCAA1/saved_spectra/csv_spectra/soap_spectrum_*.csv" \
     --output-dir /projects/F202418352CPCAA1/saved_spectra/random_rv_iccf_datasets \
     --n-samples 1000 \
     --rv-min-mps -100 \
     --rv-max-mps 100 \
     --n-workers 48

Wavelength-medium defaults:
   SOAP spectrum CSV: vacuum
   iCCF mask:         air
   CCF calculation:   vacuum
"""

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from spectrum_rv_utils import apply_rv_shift_on_original_grid


def read_spectrum_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spectrum CSV not found: {path}")

    df = pd.read_csv(path)
    if "wave_val" not in df.columns or "flux_val" not in df.columns:
        raise ValueError("Spectrum CSV must contain 'wave_val' and 'flux_val' columns.")

    wave = df["wave_val"].to_numpy(dtype=float)
    flux = df["flux_val"].to_numpy(dtype=float)
    if "flux_err" in df.columns:
        flux_err = df["flux_err"].to_numpy(dtype=float)
    else:
        flux_err = np.full_like(flux, 0.001, dtype=float)

    return wave, flux, flux_err


def convert_wavelength_medium(wavelength, source_medium: str, target_medium: str):
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


def validate_medium(name: str, value: str) -> None:
    if value not in {"air", "vacuum"}:
        raise ValueError(f"{name} must be 'air' or 'vacuum'.")


def build_iccf_mask(
    *,
    mask_name: str,
    mask_instrument: str,
    mask_medium: str,
    ccf_medium: str,
):
    import iCCF

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
    return mask


def calculate_iccf_row(
    *,
    rv_true_mps: float,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    rvarray: np.ndarray,
    mask,
    spectrum_medium: str,
    ccf_medium: str,
    mask_width: float,
) -> dict[str, Any]:
    import iCCF
    from iCCF.meta import espdr_compute_CCF_fast

    wave_shifted, flux_shifted, err_shifted = apply_rv_shift_on_original_grid(
        wave,
        flux,
        rv_true_mps / 1000.0,
        flux_err=flux_err,
    )
    wave_shifted = convert_wavelength_medium(wave_shifted, spectrum_medium, ccf_medium)

    dll = np.diff(wave_shifted)
    dll = np.r_[dll, dll[-1]]
    blaze = np.ones_like(wave_shifted)
    quality = np.zeros_like(wave_shifted, dtype=int)

    ccf, ccfe, ccfq = espdr_compute_CCF_fast(
        wave_shifted,
        dll,
        flux_shifted,
        err_shifted,
        blaze,
        quality,
        rvarray,
        mask,
        berv=0.0,
        bervmax=0.0,
        mask_width=mask_width,
    )

    indicators = iCCF.Indicators(rvarray, ccf, ccfe)
    row = {
        "rv_true_mps": float(rv_true_mps),
        "rv_iccf_mps": float(indicators.RV * 1000.0),
        "fwhm_kms": float(indicators.FWHM),
        "bis_mps": float(indicators.BIS * 1000.0),
        "vspan_mps": float(indicators.Vspan * 1000.0),
        "wspan_mps": float(indicators.Wspan * 1000.0),
        "contrast_percent": float(indicators.contrast),
    }
    for i, value in enumerate(ccf):
        row[f"ccf_{i:04d}"] = float(value)
    return row


def worker_chunk(
    *,
    chunk_id: int,
    rv_chunk_mps: np.ndarray,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    chunk_dir: str,
    rvarray: np.ndarray,
    mask_name: str,
    mask_instrument: str,
    spectrum_medium: str,
    mask_medium: str,
    ccf_medium: str,
    mask_width: float,
) -> str:
    mask = build_iccf_mask(
        mask_name=mask_name,
        mask_instrument=mask_instrument,
        mask_medium=mask_medium,
        ccf_medium=ccf_medium,
    )

    rows = [
        calculate_iccf_row(
            rv_true_mps=float(rv_true_mps),
            wave=wave,
            flux=flux,
            flux_err=flux_err,
            rvarray=rvarray,
            mask=mask,
            spectrum_medium=spectrum_medium,
            ccf_medium=ccf_medium,
            mask_width=mask_width,
        )
        for rv_true_mps in rv_chunk_mps
    ]

    chunk_csv = Path(chunk_dir) / f"dataset_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(chunk_csv, index=False)
    return str(chunk_csv)


def generate_random_rv_iccf_dataset_parallel_hpc(
    input_spectrum_csv: str | Path,
    output_csv: str | Path,
    *,
    n_samples: int = 1000,
    rv_min_mps: float = -100.0,
    rv_max_mps: float = 100.0,
    seed: int | None = 42,
    n_workers: int | None = None,
    mask_name: str = "G2",
    mask_instrument: str = "ESPRESSO",
    spectrum_medium: str = "vacuum",
    mask_medium: str = "air",
    ccf_medium: str = "vacuum",
    mask_width: float = 0.5,
    vrad_min: float = -15.0,
    vrad_max: float = 15.0,
    vrad_step: float = 0.1,
) -> pd.DataFrame:
    validate_medium("spectrum_medium", spectrum_medium)
    validate_medium("mask_medium", mask_medium)
    validate_medium("ccf_medium", ccf_medium)

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if rv_min_mps >= rv_max_mps:
        raise ValueError("rv_min_mps must be smaller than rv_max_mps.")
    if vrad_step <= 0:
        raise ValueError("vrad_step must be > 0.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    n_workers = max(1, int(n_workers))

    wave, flux, flux_err = read_spectrum_csv(input_spectrum_csv)
    rvarray = np.arange(vrad_min, vrad_max + vrad_step, vrad_step, dtype=float)

    rng = np.random.default_rng(seed)
    rv_true_mps = rng.uniform(rv_min_mps, rv_max_mps, size=n_samples).astype(float)

    n_chunks = min(n_workers, n_samples)
    rv_chunks = np.array_split(rv_true_mps, n_chunks)

    chunk_csv_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="iccf_parallel_") as tmpdir:
        futures = []
        with ProcessPoolExecutor(max_workers=n_chunks) as executor:
            for chunk_id, rv_chunk in enumerate(rv_chunks):
                futures.append(
                    executor.submit(
                        worker_chunk,
                        chunk_id=chunk_id,
                        rv_chunk_mps=rv_chunk,
                        wave=wave,
                        flux=flux,
                        flux_err=flux_err,
                        chunk_dir=tmpdir,
                        rvarray=rvarray,
                        mask_name=mask_name,
                        mask_instrument=mask_instrument,
                        spectrum_medium=spectrum_medium,
                        mask_medium=mask_medium,
                        ccf_medium=ccf_medium,
                        mask_width=mask_width,
                    )
                )

            with tqdm(total=len(futures), desc="Random RV + iCCF CCF", unit="chunk") as pbar:
                for future in as_completed(futures):
                    chunk_csv_paths.append(future.result())
                    pbar.update(1)

        dfs = [pd.read_csv(path) for path in sorted(chunk_csv_paths)]
        df_out = pd.concat(dfs, axis=0, ignore_index=True)
        df_out.to_csv(output_csv, index=False)

    return df_out


def read_input_list(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input list not found: {path}")
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def batch_output_path(input_spectrum_csv: str | Path, output_dir: str | Path) -> Path:
    stem = Path(input_spectrum_csv).stem
    if stem.startswith("soap_spectrum_"):
        stem = stem[len("soap_spectrum_") :]
    return Path(output_dir) / f"random_rv_iccf_{stem}.csv"


def collect_batch_inputs(
    *,
    input_glob: str | None,
    input_list: str | Path | None,
) -> list[Path]:
    inputs: list[Path] = []
    if input_glob:
        inputs.extend(Path(path) for path in sorted(glob(input_glob)))
    if input_list:
        inputs.extend(read_input_list(input_list))

    unique_inputs = []
    seen = set()
    for path in inputs:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique_inputs.append(path)
    return unique_inputs


def generate_batch_random_rv_iccf_datasets(
    *,
    input_spectra: list[Path],
    output_dir: str | Path,
    skip_existing: bool,
    **kwargs,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for input_spectrum in tqdm(input_spectra, desc="Input spectra", unit="spectrum"):
        output_csv = batch_output_path(input_spectrum, output_dir)
        if skip_existing and output_csv.exists():
            output_paths.append(output_csv)
            continue
        generate_random_rv_iccf_dataset_parallel_hpc(
            input_spectrum_csv=input_spectrum,
            output_csv=output_csv,
            **kwargs,
        )
        output_paths.append(output_csv)

    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a row-wise random RV dataset from one saved spectrum CSV "
            "using iCCF CCFs and indicators."
        )
    )
    parser.add_argument("--input-spectrum", default=None, help="Input CSV with wave_val/flux_val(/flux_err).")
    parser.add_argument("--output-csv", default=None, help="Output row-wise dataset CSV path.")
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Batch mode: glob matching input spectrum CSVs. Quote this argument in shell scripts.",
    )
    parser.add_argument(
        "--input-list",
        default=None,
        help="Batch mode: text file with one input spectrum CSV path per line.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Batch mode: directory for one output dataset CSV per input spectrum.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip input spectra whose output dataset CSV already exists.",
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of random RV samples.")
    parser.add_argument("--rv-min-mps", type=float, default=-100.0, help="Minimum injected RV [m/s].")
    parser.add_argument("--rv-max-mps", type=float, default=100.0, help="Maximum injected RV [m/s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers.")
    parser.add_argument("--mask-name", default="G2", help="iCCF mask name.")
    parser.add_argument("--mask-instrument", default="ESPRESSO", help="iCCF mask instrument.")
    parser.add_argument(
        "--spectrum-medium",
        choices=("vacuum", "air"),
        default="vacuum",
        help="Wavelength medium of the input spectrum CSV.",
    )
    parser.add_argument(
        "--mask-medium",
        choices=("vacuum", "air"),
        default="air",
        help="Wavelength medium of the iCCF mask.",
    )
    parser.add_argument(
        "--ccf-medium",
        choices=("vacuum", "air"),
        default="vacuum",
        help="Wavelength medium used internally for the CCF calculation.",
    )
    parser.add_argument("--mask-width", type=float, default=0.5, help="iCCF mask width.")
    parser.add_argument("--vrad-min", type=float, default=-15.0, help="CCF RV grid min [km/s].")
    parser.add_argument("--vrad-max", type=float, default=15.0, help="CCF RV grid max [km/s].")
    parser.add_argument("--vrad-step", type=float, default=0.1, help="CCF RV grid step [km/s].")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    common_kwargs = {
        "n_samples": args.n_samples,
        "rv_min_mps": args.rv_min_mps,
        "rv_max_mps": args.rv_max_mps,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "mask_name": args.mask_name,
        "mask_instrument": args.mask_instrument,
        "spectrum_medium": args.spectrum_medium,
        "mask_medium": args.mask_medium,
        "ccf_medium": args.ccf_medium,
        "mask_width": args.mask_width,
        "vrad_min": args.vrad_min,
        "vrad_max": args.vrad_max,
        "vrad_step": args.vrad_step,
    }

    batch_inputs = collect_batch_inputs(
        input_glob=args.input_glob,
        input_list=args.input_list,
    )
    if batch_inputs:
        if args.output_dir is None:
            raise ValueError("--output-dir is required with --input-glob or --input-list.")
        output_paths = generate_batch_random_rv_iccf_datasets(
            input_spectra=batch_inputs,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            **common_kwargs,
        )
        print(f"Finished {len(output_paths)} input spectra.")
        if output_paths:
            print("First saved dataset:", output_paths[0])
            print("Last saved dataset:", output_paths[-1])
        return

    if args.input_spectrum is None or args.output_csv is None:
        raise ValueError(
            "Use either --input-spectrum with --output-csv, or batch mode with "
            "--input-glob/--input-list and --output-dir."
        )

    df = generate_random_rv_iccf_dataset_parallel_hpc(
        input_spectrum_csv=args.input_spectrum,
        output_csv=args.output_csv,
        **common_kwargs,
    )
    print(f"Saved dataset: {args.output_csv}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
