from __future__ import annotations

"""
Generate one combined iCCF dataset from all ESPRESSO S1D FITS files in an input
directory.

Each output row contains:
   rv_iccf_mps, fwhm_kms, bis_mps, vspan_mps, wspan_mps, contrast_percent,
   optional file metadata, ccf_0000, ccf_0001, ...

The CCF is calculated directly from the S1D FITS table using the same low-level
`espdr_compute_CCF_fast` route used in exploratory notebook work.

Example:
    python generate_real_s1d_iccf_dataset.py \
      --input-dir ./S1D_spectra \
      --output-csv ./real_s1d_iccf_dataset.csv \
      --mask-name G2 \
      --mask-instrument ESPRESSO \
      --vrad-min -32 \
      --vrad-max -2 \
      --vrad-step 0.1
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm


def read_s1d_spectrum(path: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"S1D FITS not found: {path}")

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        tab = hdul[1].data

        wave_air = np.asarray(tab["wavelength_air"], dtype=float)
        flux = np.asarray(tab["flux"], dtype=float)
        error = np.asarray(tab["error"], dtype=float)
        quality = np.asarray(tab["quality"], dtype=int)

        meta = {
            "source_file": path.name,
            "source_path": str(path),
            "object": hdr.get("OBJECT"),
            "date_obs": hdr.get("DATE-OBS"),
            "mjd_obs": hdr.get("MJD-OBS"),
            "berv_kms": float(hdr.get("HIERARCH ESO QC BERV", np.nan)),
            "bervmax_kms": float(hdr.get("HIERARCH ESO QC BERVMAX", np.nan)),
        }

    return meta, wave_air, flux, error, quality


def build_iccf_mask(mask_name: str, mask_instrument: str):
    import iCCF

    mask_obj = iCCF.Mask(mask_name, instrument=mask_instrument)
    mask = np.zeros(mask_obj.nlines, dtype=[("lambda", "f8"), ("contrast", "f8")])
    mask["lambda"] = mask_obj.wavelength
    mask["contrast"] = mask_obj.contrast
    mask.sort(order="lambda")
    return mask


def filter_spectrum_rows(
    wave_air: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dll = np.diff(wave_air)
    dll = np.r_[dll, dll[-1]]
    blaze = np.ones_like(flux)

    good = (
        np.isfinite(wave_air)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
        & (quality == 0)
    )

    return (
        wave_air[good],
        dll[good],
        flux[good],
        error[good],
        blaze[good],
        quality[good],
    )


def calculate_s1d_iccf_row(
    *,
    path: str | Path,
    rvarray: np.ndarray,
    mask,
    mask_width: float,
) -> dict[str, Any]:
    import iCCF
    from iCCF.meta import espdr_compute_CCF_fast

    meta, wave_air, flux, error, quality = read_s1d_spectrum(path)
    ll, dll, flux_use, error_use, blaze_use, quality_use = filter_spectrum_rows(
        wave_air, flux, error, quality
    )

    ccf, ccfe, ccfq = espdr_compute_CCF_fast(
        ll,
        dll,
        flux_use,
        error_use,
        blaze_use,
        quality_use,
        rvarray,
        mask,
        berv=meta["berv_kms"],
        bervmax=meta["bervmax_kms"],
        mask_width=mask_width,
    )

    indicators = iCCF.Indicators(rvarray, ccf, ccfe)
    row = {
        "rv_iccf_mps": float(indicators.RV * 1000.0),
        "fwhm_kms": float(indicators.FWHM),
        "bis_mps": float(indicators.BIS * 1000.0),
        "vspan_mps": float(indicators.Vspan * 1000.0),
        "wspan_mps": float(indicators.Wspan * 1000.0),
        "contrast_percent": float(indicators.contrast),
        "source_file": meta["source_file"],
        "source_path": meta["source_path"],
        "object": meta["object"],
        "date_obs": meta["date_obs"],
        "mjd_obs": meta["mjd_obs"],
        "berv_kms": meta["berv_kms"],
        "bervmax_kms": meta["bervmax_kms"],
        "n_used_pixels": int(len(ll)),
    }
    for i, value in enumerate(ccf):
        row[f"ccf_{i:04d}"] = float(value)
    return row


def collect_input_files(input_dir: str | Path, recursive: bool, pattern: str) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")
    return files


def build_rvarray(vrad_min: float = -32.0, vrad_max: float = -2.0, vrad_step: float = 0.1) -> np.ndarray:
    if vrad_step <= 0:
        raise ValueError("vrad_step must be > 0.")
    if vrad_min >= vrad_max:
        raise ValueError("vrad_min must be smaller than vrad_max.")
    return np.arange(vrad_min, vrad_max + vrad_step, vrad_step, dtype=float)


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    indicator_cols = [
        "rv_iccf_mps",
        "fwhm_kms",
        "bis_mps",
        "vspan_mps",
        "wspan_mps",
        "contrast_percent",
        "source_file",
        "source_path",
        "object",
        "date_obs",
        "mjd_obs",
        "berv_kms",
        "bervmax_kms",
        "n_used_pixels",
    ]
    ccf_cols = [c for c in df_out.columns if c.startswith("ccf_")]
    ordered_cols = [c for c in indicator_cols if c in df_out.columns] + ccf_cols
    return df_out[ordered_cols]


def calculate_real_s1d_iccf_rows(
    input_files: list[str | Path],
    *,
    mask_name: str = "G2",
    mask_instrument: str = "ESPRESSO",
    mask_width: float = 0.5,
    vrad_min: float = -32.0,
    vrad_max: float = -2.0,
    vrad_step: float = 0.1,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    input_files = [Path(path) for path in input_files]
    rvarray = build_rvarray(vrad_min=vrad_min, vrad_max=vrad_max, vrad_step=vrad_step)
    mask = build_iccf_mask(mask_name=mask_name, mask_instrument=mask_instrument)

    rows = []
    iterator = input_files
    if show_progress:
        iterator = tqdm(input_files, desc="S1D -> iCCF", unit="file")

    for path in iterator:
        rows.append(
            calculate_s1d_iccf_row(
                path=path,
                rvarray=rvarray,
                mask=mask,
                mask_width=mask_width,
            )
        )
    return rows


def generate_real_s1d_iccf_dataframe(
    input_dir: str | Path,
    *,
    recursive: bool = False,
    pattern: str = "*.fits",
    mask_name: str = "G2",
    mask_instrument: str = "ESPRESSO",
    mask_width: float = 0.5,
    vrad_min: float = -32.0,
    vrad_max: float = -2.0,
    vrad_step: float = 0.1,
    show_progress: bool = True,
) -> pd.DataFrame:
    input_files = collect_input_files(input_dir=input_dir, recursive=recursive, pattern=pattern)
    rows = calculate_real_s1d_iccf_rows(
        input_files,
        mask_name=mask_name,
        mask_instrument=mask_instrument,
        mask_width=mask_width,
        vrad_min=vrad_min,
        vrad_max=vrad_max,
        vrad_step=vrad_step,
        show_progress=show_progress,
    )
    return rows_to_dataframe(rows)


def save_real_s1d_iccf_dataset(df_out: pd.DataFrame, output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return output_csv


def generate_real_s1d_iccf_dataset(
    input_dir: str | Path,
    output_csv: str | Path,
    *,
    recursive: bool = False,
    pattern: str = "*.fits",
    mask_name: str = "G2",
    mask_instrument: str = "ESPRESSO",
    mask_width: float = 0.5,
    vrad_min: float = -32.0,
    vrad_max: float = -2.0,
    vrad_step: float = 0.1,
    show_progress: bool = True,
) -> pd.DataFrame:
    df_out = generate_real_s1d_iccf_dataframe(
        input_dir=input_dir,
        recursive=recursive,
        pattern=pattern,
        mask_name=mask_name,
        mask_instrument=mask_instrument,
        mask_width=mask_width,
        vrad_min=vrad_min,
        vrad_max=vrad_max,
        vrad_step=vrad_step,
        show_progress=show_progress,
    )
    save_real_s1d_iccf_dataset(df_out, output_csv)
    return df_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one combined iCCF dataset from all ESPRESSO S1D FITS files in a directory."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing S1D FITS files.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for FITS files.")
    parser.add_argument("--pattern", default="*.fits", help="Filename pattern to match.")
    parser.add_argument("--mask-name", default="G2", help="iCCF mask name.")
    parser.add_argument("--mask-instrument", default="ESPRESSO", help="iCCF mask instrument.")
    parser.add_argument("--mask-width", type=float, default=0.5, help="iCCF mask width.")
    parser.add_argument("--vrad-min", type=float, default=-32.0, help="RV grid min [km/s].")
    parser.add_argument("--vrad-max", type=float, default=-2.0, help="RV grid max [km/s].")
    parser.add_argument("--vrad-step", type=float, default=0.1, help="RV grid step [km/s].")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = generate_real_s1d_iccf_dataset(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        recursive=args.recursive,
        pattern=args.pattern,
        mask_name=args.mask_name,
        mask_instrument=args.mask_instrument,
        mask_width=args.mask_width,
        vrad_min=args.vrad_min,
        vrad_max=args.vrad_max,
        vrad_step=args.vrad_step,
    )
    print(f"Saved dataset: {args.output_csv}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
