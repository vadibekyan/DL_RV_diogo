#!/usr/bin/env python3
"""
Predict RV values for real CCFs using a saved trained model bundle.

Supported inputs:
1. CSV/parquet files with columns matching the saved ccf column names.
2. NPY files containing shape (300,) or (n_samples, 300).

Example:
    python tested_models/predict_real_ccf.py \
        --model-dir tested_models/results_positional_multiscale_conv1d/final_model \
        --input-path real_ccfs.csv \
        --output-path predicted_rv.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict RVs for real CCFs using a saved trained model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing rv_model.keras and preprocessing files.")
    parser.add_argument("--input-path", required=True, help="Path to input CSV, parquet, or NPY file.")
    parser.add_argument("--output-path", default=None, help="Optional path to save predictions as CSV.")
    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional identifier column to preserve in CSV/parquet inputs.",
    )
    return parser.parse_args()


def load_inputs(input_path: Path, ccf_columns: list[str], id_col: str | None) -> tuple[np.ndarray, pd.DataFrame]:
    suffix = input_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(input_path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D array in {input_path}, got shape {arr.shape}.")
        if arr.shape[1] != len(ccf_columns):
            raise ValueError(
                f"Input has {arr.shape[1]} CCF bins, but model expects {len(ccf_columns)}."
            )
        info_df = pd.DataFrame({"sample_index": np.arange(arr.shape[0], dtype=int)})
        return arr, info_df

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported input format for {input_path}. Use .csv, .parquet, or .npy.")

    missing = [col for col in ccf_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing {len(missing)} expected CCF columns. "
            f"First missing columns: {missing[:5]}"
        )

    X = df[ccf_columns].to_numpy(dtype=np.float32)

    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(f"id column '{id_col}' not found in input file.")
        info_df = df[[id_col]].copy()
    else:
        info_df = pd.DataFrame({"sample_index": np.arange(len(df), dtype=int)})

    return X, info_df


def build_model_input(X: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray, use_vrad_channel: bool, vrad: np.ndarray) -> np.ndarray:
    X_scaled = ((X - x_mean) / x_scale).astype(np.float32)

    if use_vrad_channel:
        if len(vrad) != X_scaled.shape[1]:
            raise ValueError(
                f"Saved VRAD length ({len(vrad)}) does not match input width ({X_scaled.shape[1]})."
            )
        vrad_channel = np.broadcast_to(vrad, X_scaled.shape).astype(np.float32)
        return np.stack([X_scaled, vrad_channel], axis=-1).astype(np.float32)

    return X_scaled[..., np.newaxis]


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input_path)

    model_path = model_dir / "rv_model.keras"
    metadata_path = model_dir / "preprocessing_metadata.json"
    artifacts_path = model_dir / "preprocessing_artifacts.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    metadata = json.loads(metadata_path.read_text())
    artifacts = np.load(artifacts_path, allow_pickle=False)

    ccf_columns = metadata["ccf_columns"]
    use_vrad_channel = bool(metadata["use_vrad_channel"])

    x_mean = artifacts["x_mean"].astype(np.float32)
    x_scale = artifacts["x_scale"].astype(np.float32)
    y_mean = artifacts["y_mean"].astype(np.float32)
    y_scale = artifacts["y_scale"].astype(np.float32)
    vrad = artifacts["vrad"].astype(np.float32)

    X, info_df = load_inputs(input_path=input_path, ccf_columns=ccf_columns, id_col=args.id_col)
    X_in = build_model_input(
        X=X,
        x_mean=x_mean,
        x_scale=x_scale,
        use_vrad_channel=use_vrad_channel,
        vrad=vrad,
    )

    model = keras.models.load_model(model_path)
    y_pred_scaled = model.predict(X_in, verbose=0).reshape(-1)
    y_pred = y_pred_scaled * y_scale[0] + y_mean[0]

    results_df = info_df.copy()
    results_df["predicted_rv_mps"] = y_pred.astype(np.float64)

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
