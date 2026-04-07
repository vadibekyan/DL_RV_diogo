from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import arve

from create_ccf_ARVE import DEFAULT_STELLAR_PARAMETERS
from spectrum_rv_utils import apply_rv_shift_on_original_grid


def _read_spectrum_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def generate_random_rv_ccf_dataset(
    input_spectrum_csv: str | Path,
    mask_path: str | Path,
    output_csv: str | Path,
    *,
    n_samples: int = 1000,
    rv_min: float = -20.0,
    rv_max: float = 20.0,
    seed: int | None = 42,
    resolution: float = 115000.0,
    medium: str = "vac",
    mask_medium: str = "vac",
    vrad_grid: tuple[float, float, float] = (-20.0, 20.0, 0.5),
    exclude_tellurics: bool = False,
    ccf_err_scale: bool = True,
    target: str = "Sun",
) -> pd.DataFrame:
    input_spectrum_csv = Path(input_spectrum_csv)
    mask_path = Path(mask_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask CSV not found: {mask_path}")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if rv_min >= rv_max:
        raise ValueError("rv_min must be smaller than rv_max.")

    wave, flux, flux_err = _read_spectrum_csv(input_spectrum_csv)

    rng = np.random.default_rng(seed)
    rv_true = rng.uniform(rv_min, rv_max, size=n_samples).astype(float)

    time_val = np.arange(n_samples, dtype=float)
    tmp_files: list[str] = []

    # One global progress bar for the full workflow.
    with tqdm(total=n_samples + 1, desc="Random RV + ARVE CCF", unit="step") as pbar:
        with tempfile.TemporaryDirectory(prefix="arve_shifted_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for i, rv in enumerate(rv_true):
                wave_shifted, flux_out, flux_err_out = apply_rv_shift_on_original_grid(
                    wave, flux, rv, flux_err=flux_err
                )
                shifted_path = tmpdir_path / f"spectrum_{i:06d}.csv"
                pd.DataFrame(
                    {
                        "wave_val": wave_shifted,
                        "flux_val": flux_out,
                        "flux_err": flux_err_out,
                    }
                ).to_csv(shifted_path, index=False)
                tmp_files.append(str(shifted_path))
                pbar.update(1)

            obj = arve.ARVE()
            obj.star.target = target
            obj.star.stellar_parameters = dict(DEFAULT_STELLAR_PARAMETERS)
            obj.data.add_data(
                time_val=time_val,
                files=tmp_files,
                format="s1d",
                extension="csv",
                medium=medium,
                resolution=resolution,
                same_wave_grid=False,
            )
            obj.data.get_aux_data(mask_path=str(mask_path), mask_medium=mask_medium)
            obj.data.compute_vrad_ccf(
                exclude_tellurics=exclude_tellurics,
                vrad_grid=[float(v) for v in vrad_grid],
                ccf_err_scale=ccf_err_scale,
            )
            pbar.update(1)

    ccf_vrad = np.asarray(obj.data.ccf["ccf_vrad"], dtype=float)
    ccf_val = np.asarray(obj.data.ccf["ccf_val"][:, -1, :], dtype=float)
    vrad_val = np.asarray(obj.data.vrad["vrad_val"], dtype=float)
    vrad_err = np.asarray(obj.data.vrad["vrad_err"], dtype=float)

    ccf_cols = [f"ccf_{i:04d}" for i in range(ccf_val.shape[1])]
    df_out = pd.DataFrame(ccf_val, columns=ccf_cols)
    df_out.insert(0, "rv_pred", vrad_val)
    df_out.insert(1, "rv_err", vrad_err)
    df_out.insert(2, "rv_true", rv_true)
    df_out.to_csv(output_csv, index=False)

    np.savez(
        output_csv.with_suffix(".npz"),
        rv_true=rv_true,
        rv_pred=vrad_val,
        rv_err=vrad_err,
        ccf_vrad=ccf_vrad,
        ccf_val=ccf_val,
    )
    return df_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random RV-shifted spectra from one input CSV and compute ARVE CCFs. "
            "Saves one row per sample."
        )
    )
    parser.add_argument("--input-spectrum", required=True, help="Input CSV with wave_val/flux_val(/flux_err).")
    parser.add_argument("--mask-path", required=True, help="ARVE mask CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output row-wise dataset CSV path.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of random RV samples.")
    parser.add_argument("--rv-min", type=float, default=-20.0, help="Minimum RV [km/s].")
    parser.add_argument("--rv-max", type=float, default=20.0, help="Maximum RV [km/s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resolution", type=float, default=115000.0, help="Instrument resolution.")
    parser.add_argument("--medium", default="vac", choices=["vac", "air"], help="Input spectrum medium.")
    parser.add_argument("--mask-medium", default="vac", choices=["vac", "air"], help="Mask medium.")
    parser.add_argument("--vrad-min", type=float, default=-20.0, help="ARVE CCF RV grid min [km/s].")
    parser.add_argument("--vrad-max", type=float, default=20.0, help="ARVE CCF RV grid max [km/s].")
    parser.add_argument("--vrad-step", type=float, default=0.5, help="ARVE CCF RV grid step [km/s].")
    parser.add_argument("--target", default="Sun", help="Target name.")
    parser.add_argument("--exclude-tellurics", action="store_true", help="Exclude tellurics in ARVE.")
    parser.add_argument("--no-ccf-err-scale", action="store_true", help="Disable ARVE ccf_err_scale.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_random_rv_ccf_dataset(
        input_spectrum_csv=args.input_spectrum,
        mask_path=args.mask_path,
        output_csv=args.output_csv,
        n_samples=args.n_samples,
        rv_min=args.rv_min,
        rv_max=args.rv_max,
        seed=args.seed,
        resolution=args.resolution,
        medium=args.medium,
        mask_medium=args.mask_medium,
        vrad_grid=(args.vrad_min, args.vrad_max, args.vrad_step),
        exclude_tellurics=args.exclude_tellurics,
        ccf_err_scale=not args.no_ccf_err_scale,
        target=args.target,
    )


if __name__ == "__main__":
    main()
