#!/usr/bin/env python3
from __future__ import annotations

"""
Scan vrad ranges for real S1D spectra, predict RVs with a saved MLP model,
and save one final statistics table.

For each tested vrad_min, the script uses:
    vrad_max = vrad_min + window_size

Workflow per range:
1. Generate iCCF CCFs from S1D FITS files
2. Normalize each CCF row by its row maximum
3. Apply the saved MLP model
4. Compare predicted RVs against iCCF RVs
5. Save one statistics row to the final output table

Example:
    python scan_real_ccf_mlp_vrad_hpc.py \
        --input-dir ./S1D_spectra \
        --model-dir ./mlp_full_model_526_256_128_64_32 \
        --output-csv ./vrad_scan_stats.csv \
        --vrad-min-start -20 \
        --vrad-min-stop -10 \
        --vrad-min-step 1.0 \
        --window-size 30 \
        --vrad-step 0.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan vrad ranges for real CCF generation and MLP RV prediction."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing S1D FITS files.")
    parser.add_argument("--output-csv", required=True, help="Path to the final statistics CSV.")
    parser.add_argument("--model-dir", required=True, help="Directory containing the saved MLP model bundle.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for FITS files.")
    parser.add_argument("--pattern", default="*.fits", help="Filename pattern to match. Default: *.fits")
    parser.add_argument("--mask-name", default="G2", help="iCCF mask name.")
    parser.add_argument("--mask-instrument", default="ESPRESSO", help="iCCF mask instrument.")
    parser.add_argument("--mask-width", type=float, default=0.5, help="iCCF mask width.")
    parser.add_argument("--vrad-min-start", type=float, required=True, help="First vrad_min value [km/s].")
    parser.add_argument("--vrad-min-stop", type=float, required=True, help="Last vrad_min value [km/s], inclusive.")
    parser.add_argument("--vrad-min-step", type=float, required=True, help="Step for vrad_min scan [km/s].")
    parser.add_argument("--window-size", type=float, default=30.0, help="Use vrad_max = vrad_min + window_size.")
    parser.add_argument("--vrad-step", type=float, default=0.1, help="Step of the CCF RV grid [km/s].")
    parser.add_argument(
        "--no-center-series",
        action="store_true",
        help="Disable mean-centering before comparison. By default both RV series are centered.",
    )
    return parser


def build_scan_values(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--vrad-min-step must be > 0")
    if stop < start:
        raise ValueError("--vrad-min-stop must be >= --vrad-min-start")
    n_steps = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = start + step * np.arange(n_steps, dtype=float)
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return values


def load_model_bundle(model_dir: str | Path) -> dict:
    from tensorflow import keras

    model_dir = Path(model_dir)
    model_path = model_dir / "rv_model.keras"
    metadata_path = model_dir / "preprocessing_metadata.json"
    artifacts_path = model_dir / "preprocessing_artifacts.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

    metadata = json.loads(metadata_path.read_text())
    artifacts = np.load(artifacts_path, allow_pickle=False)

    return {
        "model": keras.models.load_model(model_path),
        "ccf_columns": metadata["ccf_columns"],
        "x_mean": artifacts["x_mean"].astype(np.float32),
        "x_scale": artifacts["x_scale"].astype(np.float32),
        "y_mean": artifacts["y_mean"].astype(np.float32),
        "y_scale": artifacts["y_scale"].astype(np.float32),
    }


def normalize_ccf_for_model(ccf_df: pd.DataFrame, ccf_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in ccf_columns if col not in ccf_df.columns]
    if missing:
        raise ValueError(
            f"Generated CCF dataframe is missing {len(missing)} expected columns. "
            f"First missing columns: {missing[:5]}"
        )

    normalized_df = ccf_df[ccf_columns].copy()
    row_max = normalized_df.max(axis=1)
    if (row_max == 0).any():
        raise ValueError("At least one CCF row has max=0, so normalization would fail.")

    normalized_df[ccf_columns] = normalized_df[ccf_columns].div(row_max, axis=0)
    normalized_df[ccf_columns] = normalized_df[ccf_columns].round(4)
    return normalized_df


def predict_rvs(normalized_df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X = normalized_df[bundle["ccf_columns"]].to_numpy(dtype=np.float32)
    X_scaled = ((X - bundle["x_mean"]) / bundle["x_scale"]).astype(np.float32)

    y_pred_scaled = bundle["model"].predict(X_scaled, verbose=0).reshape(-1)
    y_pred = y_pred_scaled * bundle["y_scale"][0] + bundle["y_mean"][0]
    return y_pred.astype(np.float64)


def safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compare_predicted_and_true(
    true_rv_mps: np.ndarray,
    predicted_rv_mps: np.ndarray,
    *,
    center_series: bool,
) -> dict[str, float]:
    true_used = true_rv_mps.astype(float).copy()
    pred_used = predicted_rv_mps.astype(float).copy()

    if center_series:
        true_used = true_used - np.mean(true_used)
        pred_used = pred_used - np.mean(pred_used)

    residual = pred_used - true_used

    return {
        "n_samples": int(len(true_used)),
        "bias_mps": float(np.mean(residual)),
        "std_mps": float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan,
        "mae_mps": float(np.mean(np.abs(residual))),
        "median_abs_err_mps": float(np.median(np.abs(residual))),
        "rmse_mps": float(np.sqrt(np.mean(residual ** 2))),
        "min_residual_mps": float(np.min(residual)),
        "max_residual_mps": float(np.max(residual)),
        "pearson_r": safe_pearson_r(true_used, pred_used),
    }


def run_one_scan_case(
    *,
    input_dir: str | Path,
    bundle: dict,
    vrad_min: float,
    vrad_max: float,
    vrad_step: float,
    recursive: bool,
    pattern: str,
    mask_name: str,
    mask_instrument: str,
    mask_width: float,
    center_series: bool,
) -> dict[str, float]:
    from generate_real_s1d_iccf_dataset import generate_real_s1d_iccf_dataframe

    ccf_df = generate_real_s1d_iccf_dataframe(
        input_dir=input_dir,
        recursive=recursive,
        pattern=pattern,
        mask_name=mask_name,
        mask_instrument=mask_instrument,
        mask_width=mask_width,
        vrad_min=vrad_min,
        vrad_max=vrad_max,
        vrad_step=vrad_step,
        show_progress=False,
    )

    normalized_df = normalize_ccf_for_model(ccf_df, bundle["ccf_columns"])
    predicted_rv_mps = predict_rvs(normalized_df, bundle)
    stats = compare_predicted_and_true(
        ccf_df["rv_iccf_mps"].to_numpy(dtype=float),
        predicted_rv_mps,
        center_series=center_series,
    )

    return {
        "vrad_min": float(vrad_min),
        "vrad_max": float(vrad_max),
        "vrad_step": float(vrad_step),
        **stats,
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if args.vrad_step <= 0:
        raise ValueError("--vrad-step must be > 0")

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    vrad_min_values = build_scan_values(
        start=args.vrad_min_start,
        stop=args.vrad_min_stop,
        step=args.vrad_min_step,
    )
    bundle = load_model_bundle(args.model_dir)

    stats_rows: list[dict[str, float]] = []
    for vrad_min in tqdm(vrad_min_values, desc="Scanning vrad ranges", unit="range"):
        vrad_max = float(vrad_min + args.window_size)
        stats = run_one_scan_case(
            input_dir=input_dir,
            bundle=bundle,
            vrad_min=float(vrad_min),
            vrad_max=vrad_max,
            vrad_step=args.vrad_step,
            recursive=args.recursive,
            pattern=args.pattern,
            mask_name=args.mask_name,
            mask_instrument=args.mask_instrument,
            mask_width=args.mask_width,
            center_series=not args.no_center_series,
        )
        stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows).sort_values("vrad_min").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)

    print(f"Saved statistics to {output_csv}")
    print(f"Shape: {stats_df.shape}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
