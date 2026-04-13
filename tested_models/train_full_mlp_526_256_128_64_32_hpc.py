#!/usr/bin/env python3
"""
Train a full-data MLP RV model and save a deployable model bundle.

Architecture:
    Input(300) -> Dense(526) -> Dense(256) -> Dense(128) -> Dense(64) ->
    Dense(32) -> Dense(1)

Saved bundle:
    output_dir/
      rv_model.keras
      preprocessing_artifacts.npz
      preprocessing_metadata.json
      training_history.csv
      model_summary.txt
      training_config.json
      training_metrics.json

Example:
    python train_full_mlp_526_256_128_64_32_hpc.py \
        --input-path CCFs/full_iccf_dataset_normalized.parquet \
        --output-dir mlp_full_model_526_256_128_64_32
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TrainingConfig:
    input_path: str
    output_dir: str
    target_col: str
    ccf_prefix: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    lr_patience: int
    lr_factor: float
    min_lr: float
    validation_fraction: float
    hidden_units: list[int]
    dropout: float
    l2_reg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full-data MLP RV model and save deployable artifacts.")
    parser.add_argument("--input-path", required=True, help="Processed dataset path (.csv or .parquet).")
    parser.add_argument("--output-dir", required=True, help="Directory where model/artifacts will be written.")
    parser.add_argument("--target-col", default="rv_true_mps", help="Target column name.")
    parser.add_argument("--ccf-prefix", default="ccf_", help="Prefix used to identify CCF feature columns.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1000, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial Adam learning rate.")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping patience on val_loss.")
    parser.add_argument("--lr-patience", type=int, default=8, help="ReduceLROnPlateau patience on val_loss.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor on plateau.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Validation split used during final training.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs="+",
        default=[526, 256, 128, 64, 32],
        help="MLP hidden layer sizes.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate after each hidden layer.")
    parser.add_argument("--l2-reg", type=float, default=0.0, help="L2 regularization coefficient.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input format for {input_path}. Use .csv or .parquet.")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _regularizer(l2_reg: float):
    return keras.regularizers.l2(l2_reg) if l2_reg > 0 else None


def build_model(input_dim: int, hidden_units: list[int], dropout: float, learning_rate: float, l2_reg: float) -> keras.Model:
    reg = _regularizer(l2_reg)
    model = keras.Sequential(name="mlp_rv_regressor_526_256_128_64_32")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dense_{i}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i}"))

    model.add(layers.Dense(1, name="rv_output"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def save_preprocessing_artifacts(output_dir: Path, ccf_cols: list[str], x_scaler: StandardScaler, y_scaler: StandardScaler) -> None:
    np.savez(
        output_dir / "preprocessing_artifacts.npz",
        x_mean=x_scaler.mean_.astype(np.float32),
        x_scale=x_scaler.scale_.astype(np.float32),
        y_mean=y_scaler.mean_.astype(np.float32),
        y_scale=y_scaler.scale_.astype(np.float32),
    )

    metadata = {
        "ccf_columns": ccf_cols,
        "use_vrad_channel": False,
        "n_ccf_features": len(ccf_cols),
        "model_type": "mlp",
        "hidden_units": [int(x) for x in args.hidden_units] if False else None,
    }
    metadata.pop("hidden_units")
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    set_global_seed(args.seed)

    cfg = TrainingConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        ccf_prefix=args.ccf_prefix,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        validation_fraction=args.validation_fraction,
        hidden_units=list(args.hidden_units),
        dropout=args.dropout,
        l2_reg=args.l2_reg,
    )

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_path} ...")
    df = load_dataset(input_path)
    ccf_cols = [c for c in df.columns if c.startswith(args.ccf_prefix)]
    if not ccf_cols:
        raise ValueError(f"No feature columns found with prefix '{args.ccf_prefix}'")
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset")

    X = df[ccf_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of CCF features: {len(ccf_cols)}")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_s = x_scaler.fit_transform(X).astype(np.float32)
    y_s = y_scaler.fit_transform(y).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_s,
        y_s,
        test_size=args.validation_fraction,
        random_state=args.seed,
        shuffle=True,
    )

    model = build_model(
        input_dim=X_s.shape[1],
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    print("Training final MLP model on the full dataset ...")
    start = perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    fit_seconds = perf_counter() - start

    y_train_pred = y_scaler.inverse_transform(model.predict(X_train, verbose=0)).ravel()
    y_val_pred = y_scaler.inverse_transform(model.predict(X_val, verbose=0)).ravel()
    y_all_pred = y_scaler.inverse_transform(model.predict(X_s, verbose=0)).ravel()

    train_metrics = evaluate_predictions(y_scaler.inverse_transform(y_train), y_train_pred)
    val_metrics = evaluate_predictions(y_scaler.inverse_transform(y_val), y_val_pred)
    all_metrics = evaluate_predictions(y, y_all_pred)

    model.save(output_dir / "rv_model.keras")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    save_preprocessing_artifacts(output_dir=output_dir, ccf_cols=ccf_cols, x_scaler=x_scaler, y_scaler=y_scaler)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "model_summary.txt").write_text("\n".join(summary_lines) + "\n")
    (output_dir / "training_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    metrics = {
        "fit_seconds": float(fit_seconds),
        "epochs_ran": int(len(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "all_data_metrics": all_metrics,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved model bundle to {output_dir}")
    print(f"Validation MAE: {val_metrics['mae']:.6f} m/s")
    print(f"All-data MAE:   {all_metrics['mae']:.6f} m/s")


if __name__ == "__main__":
    main()
