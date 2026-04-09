from __future__ import annotations

"""
Generate noisy realizations of input spectra by adding flux-dependent Gaussian noise.

Each input CSV must contain:
    wave_val, flux_val

If flux_err exists, it is replaced in the outputs by the generated per-pixel noise sigma.

Examples:
    python generate_noisy_spectra_hpc.py \
        --input-dir hpc_results/csv_spectra \
        --output-dir hpc_results/csv_spectra_noisy \
        --snr 100 \
        --n-realizations 100 \
        --n-workers 48

    python generate_noisy_spectra_hpc.py \
        --input-list spectra_to_process.txt \
        --output-dir hpc_results/csv_spectra_noisy \
        --snr 150 \
        --n-realizations 20
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def read_input_list(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input list not found: {path}")
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def collect_inputs(
    *,
    input_dir: str | Path | None,
    input_list: str | Path | None,
    pattern: str,
) -> list[Path]:
    inputs: list[Path] = []

    if input_dir is not None:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
        inputs.extend(sorted(input_dir.glob(pattern)))

    if input_list is not None:
        inputs.extend(read_input_list(input_list))

    deduped: list[Path] = []
    seen = set()
    for path in inputs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def sanitize_snr_label(snr: float) -> str:
    return str(snr).replace(".", "p")


def build_output_path(input_path: Path, output_dir: Path, snr: float, realization_idx: int) -> Path:
    snr_label = sanitize_snr_label(snr)
    return output_dir / f"{input_path.stem}_snr{snr_label}_real{realization_idx:04d}.csv"


def add_flux_dependent_noise(
    flux: np.ndarray,
    *,
    snr: float,
    rng: np.random.Generator,
    min_flux_for_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    if snr <= 0:
        raise ValueError(f"SNR must be > 0, got {snr}")

    safe_flux = np.clip(np.abs(flux), min_flux_for_noise, None)
    sigma = safe_flux / snr
    noise = rng.normal(loc=0.0, scale=sigma, size=flux.shape)
    return flux + noise, sigma


def generate_noisy_realizations_for_one_spectrum(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    snr: float,
    n_realizations: int,
    min_flux_for_noise: float,
    seed: int,
    overwrite: bool,
) -> list[str]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Spectrum CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"wave_val", "flux_val"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Spectrum CSV {input_path} is missing required columns: {sorted(missing)}"
        )

    wave = df["wave_val"].to_numpy(dtype=float)
    flux = df["flux_val"].to_numpy(dtype=float)

    written_paths: list[str] = []
    for realization_idx in range(n_realizations):
        realization_seed = seed + 10_000 * realization_idx
        rng = np.random.default_rng(realization_seed)
        noisy_flux, sigma = add_flux_dependent_noise(
            flux,
            snr=snr,
            rng=rng,
            min_flux_for_noise=min_flux_for_noise,
        )

        out_df = pd.DataFrame(
            {
                "wave_val": wave,
                "flux_val": noisy_flux,
                "flux_err": sigma,
            }
        )

        output_path = build_output_path(input_path, output_dir, snr, realization_idx)
        if output_path.exists() and not overwrite:
            written_paths.append(str(output_path))
            continue

        out_df.to_csv(output_path, index=False)
        written_paths.append(str(output_path))

    return written_paths


def worker_task(
    *,
    input_path: str,
    output_dir: str,
    snr: float,
    n_realizations: int,
    min_flux_for_noise: float,
    base_seed: int,
    overwrite: bool,
) -> tuple[str, int]:
    input_path_obj = Path(input_path)
    # Offset the base seed by input filename for reproducible but distinct realizations.
    per_file_seed = base_seed + sum(ord(ch) for ch in input_path_obj.stem)
    outputs = generate_noisy_realizations_for_one_spectrum(
        input_path=input_path_obj,
        output_dir=output_dir,
        snr=snr,
        n_realizations=n_realizations,
        min_flux_for_noise=min_flux_for_noise,
        seed=per_file_seed,
        overwrite=overwrite,
    )
    return str(input_path_obj), len(outputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noisy spectrum realizations with flux-dependent Gaussian noise."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing input spectrum CSV files.",
    )
    parser.add_argument(
        "--input-list",
        default=None,
        help="Text file with one input spectrum CSV path per line.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used with --input-dir. Default: *.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where noisy spectra CSV files will be written.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        required=True,
        help="Target SNR for the noise model. For normalized spectra, sigma_i = max(|flux_i|, floor) / SNR.",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=100,
        help="Number of noisy spectra to generate for each input spectrum.",
    )
    parser.add_argument(
        "--min-flux-for-noise",
        type=float,
        default=1e-3,
        help="Lower floor used in sigma_i = max(|flux_i|, floor) / SNR to avoid zero noise in deep lines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to SLURM_CPUS_PER_TASK or 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_dir is None and args.input_list is None:
        parser.error("Provide at least one of --input-dir or --input-list.")

    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be > 0")
    if args.snr <= 0:
        raise ValueError("--snr must be > 0")
    if args.min_flux_for_noise <= 0:
        raise ValueError("--min-flux-for-noise must be > 0")

    input_paths = collect_inputs(
        input_dir=args.input_dir,
        input_list=args.input_list,
        pattern=args.pattern,
    )
    if not input_paths:
        raise FileNotFoundError("No input spectra found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.n_workers
    if n_workers is None:
        n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
    n_workers = max(1, int(n_workers))

    print(f"Found {len(input_paths)} input spectra.")
    print(f"Writing noisy realizations to {output_dir}")
    print(f"SNR={args.snr}, n_realizations={args.n_realizations}, n_workers={n_workers}")

    futures = []
    total_written = 0
    with ProcessPoolExecutor(max_workers=min(n_workers, len(input_paths))) as executor:
        for input_path in input_paths:
            futures.append(
                executor.submit(
                    worker_task,
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    snr=args.snr,
                    n_realizations=args.n_realizations,
                    min_flux_for_noise=args.min_flux_for_noise,
                    base_seed=args.seed,
                    overwrite=args.overwrite,
                )
            )

        with tqdm(total=len(futures), desc="Noisy spectra", unit="spectrum") as pbar:
            for future in as_completed(futures):
                input_path, n_written = future.result()
                total_written += n_written
                print(f"Completed {input_path}: {n_written} files")
                pbar.update(1)

    print(f"Done. Generated or confirmed {total_written} noisy spectra files.")


if __name__ == "__main__":
    main()
