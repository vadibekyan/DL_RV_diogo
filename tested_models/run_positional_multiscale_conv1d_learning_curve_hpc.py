#!/usr/bin/env python3
"""
Run a learning-curve experiment for RV prediction on a processed dataset using
a position-aware multi-scale 1D CNN.

Why this model:
1. It preserves absolute position information by avoiding pooling/GAP.
2. It can ingest the VRAD grid as an explicit second channel.
3. It uses parallel kernels to capture CCF structure at multiple widths.
4. It ends with Flatten + Dense layers so the regressor still knows where
   each learned feature occurred along the velocity axis.

Recommended first run:
    python tested_models/run_positional_multiscale_conv1d_learning_curve_hpc.py \
        --input-path CCFs/full_iccf_dataset_normalized.parquet \
        --output-dir tested_models/results_positional_multiscale_conv1d \
        --train-sizes 1000 5000 10000 20000 50000 100000 200000 300000 370000 \
        --repeats 5
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
class ExperimentConfig:
    input_path: str
    output_dir: str
    target_col: str
    ccf_prefix: str
    train_sizes: list[int]
    repeats: int
    train_fraction: float
    dev_fraction: float
    test_fraction: float
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    lr_patience: int
    lr_factor: float
    min_lr: float
    branch_filters: int
    kernel_sizes: list[int]
    fusion_filters: list[int]
    hidden_units: list[int]
    dropout: float
    spatial_dropout: float
    l2_reg: float
    use_vrad_channel: bool
    vrad_start: float
    vrad_stop: float
    vrad_step: float
    save_final_model: bool
    final_validation_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a learning-curve experiment for RV regression using a position-aware multi-scale 1D CNN."
    )
    parser.add_argument("--input-path", default=None, help="Processed dataset path (.csv or .parquet).")
    parser.add_argument("--input-parquet", default=None, help="Backward-compatible alias for --input-path.")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    parser.add_argument("--target-col", default="rv_true_mps", help="Target column name.")
    parser.add_argument("--ccf-prefix", default="ccf_", help="Prefix used to identify feature columns.")
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000, 20000, 40000, 70000],
        help="Training subset sizes to evaluate.",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of random subsets per train size.")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Fraction of rows used for training.")
    parser.add_argument("--dev-fraction", type=float, default=0.15, help="Fraction of rows used for dev.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of rows used for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1000, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial Adam learning rate.")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStopping patience on val_loss.")
    parser.add_argument("--lr-patience", type=int, default=8, help="ReduceLROnPlateau patience on val_loss.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor on plateau.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument(
        "--branch-filters",
        type=int,
        default=16,
        help="Filters per multiscale branch.",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=[3, 7, 15],
        help="Parallel Conv1D kernel sizes.",
    )
    parser.add_argument(
        "--fusion-filters",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Filters in sequential fusion Conv1D layers after concatenation.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Dense hidden-layer sizes after Flatten.",
    )
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate after dense layers.")
    parser.add_argument(
        "--spatial-dropout",
        type=float,
        default=0.05,
        help="SpatialDropout1D rate after convolutional fusion blocks.",
    )
    parser.add_argument("--l2-reg", type=float, default=1e-5, help="L2 regularization coefficient.")
    parser.add_argument(
        "--use-vrad-channel",
        dest="use_vrad_channel",
        action="store_true",
        help="Use the VRAD grid as an explicit second input channel.",
    )
    parser.add_argument(
        "--no-vrad-channel",
        dest="use_vrad_channel",
        action="store_false",
        help="Disable the VRAD channel and use only the normalized CCF.",
    )
    parser.set_defaults(use_vrad_channel=True)
    parser.add_argument("--vrad-start", type=float, default=-15.0, help="Start of VRAD grid.")
    parser.add_argument("--vrad-stop", type=float, default=15.0, help="Stop of VRAD grid.")
    parser.add_argument("--vrad-step", type=float, default=0.1, help="Step of VRAD grid.")
    parser.add_argument(
        "--save-final-model",
        dest="save_final_model",
        action="store_true",
        help="Train one final model on the full labeled dataset and save deployable artifacts.",
    )
    parser.add_argument(
        "--no-save-final-model",
        dest="save_final_model",
        action="store_false",
        help="Skip final full-dataset model training/export.",
    )
    parser.set_defaults(save_final_model=True)
    parser.add_argument(
        "--final-validation-fraction",
        type=float,
        default=0.10,
        help="Validation split used only for the final full-dataset model training.",
    )
    return parser.parse_args()


def validate_split_fractions(train_fraction: float, dev_fraction: float, test_fraction: float) -> None:
    total = train_fraction + dev_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_fraction + dev_fraction + test_fraction must sum to 1.0, "
            f"got {total:.6f}"
        )


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_dataset(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input format for {input_path}. Use .csv or .parquet.")


def build_vrad_grid(n_bins: int, vrad_start: float, vrad_stop: float, vrad_step: float) -> np.ndarray:
    vrad = np.arange(vrad_start, vrad_stop + 0.5 * vrad_step, vrad_step, dtype=np.float32)
    if len(vrad) != n_bins:
        raise ValueError(
            f"VRAD grid length ({len(vrad)}) does not match number of CCF bins ({n_bins})."
        )
    return vrad


def standardize_vrad(vrad: np.ndarray) -> np.ndarray:
    vrad_mean = np.mean(vrad, dtype=np.float32)
    vrad_std = np.std(vrad, dtype=np.float32)
    if vrad_std == 0:
        raise ValueError("VRAD grid standard deviation is zero.")
    return ((vrad - vrad_mean) / vrad_std).astype(np.float32)


def add_vrad_channel(X_2d: np.ndarray, vrad: np.ndarray) -> np.ndarray:
    vrad_channel = np.broadcast_to(vrad, X_2d.shape).astype(np.float32)
    return np.stack([X_2d, vrad_channel], axis=-1).astype(np.float32)


def _regularizer(l2_reg: float):
    return keras.regularizers.l2(l2_reg) if l2_reg > 0 else None


def build_model(
    input_length: int,
    input_channels: int,
    branch_filters: int,
    kernel_sizes: list[int],
    fusion_filters: list[int],
    hidden_units: list[int],
    dropout: float,
    spatial_dropout: float,
    learning_rate: float,
    l2_reg: float,
) -> keras.Model:
    reg = _regularizer(l2_reg)

    inputs = keras.Input(shape=(input_length, input_channels), name="ccf_input")
    branches = []

    for i, kernel_size in enumerate(kernel_sizes, start=1):
        branch = layers.Conv1D(
            filters=branch_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name=f"branch{i}_conv",
        )(inputs)
        branches.append(branch)

    if len(branches) == 1:
        x = branches[0]
    else:
        x = layers.Concatenate(name="multiscale_concat")(branches)

    x = layers.BatchNormalization(name="multiscale_bn")(x)

    for i, filters in enumerate(fusion_filters, start=1):
        if filters <= 0:
            continue
        x = layers.Conv1D(
            filters=filters,
            kernel_size=1 if i == 1 else 3,
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name=f"fusion_conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"fusion_bn_{i}")(x)
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(spatial_dropout, name=f"spatial_dropout_{i}")(x)

    x = layers.Flatten(name="flatten")(x)

    for i, units in enumerate(hidden_units, start=1):
        if units <= 0:
            continue
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=reg,
            name=f"dense_{i}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, name="rv_output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="positional_multiscale_conv1d_rv_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def save_preprocessing_artifacts(
    output_dir: Path,
    ccf_cols: list[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    use_vrad_channel: bool,
    vrad: np.ndarray | None,
) -> None:
    scaler_path = output_dir / "preprocessing_artifacts.npz"
    metadata_path = output_dir / "preprocessing_metadata.json"

    np.savez(
        scaler_path,
        x_mean=x_scaler.mean_.astype(np.float32),
        x_scale=x_scaler.scale_.astype(np.float32),
        y_mean=y_scaler.mean_.astype(np.float32),
        y_scale=y_scaler.scale_.astype(np.float32),
        vrad=(vrad.astype(np.float32) if vrad is not None else np.array([], dtype=np.float32)),
    )

    metadata = {
        "ccf_columns": ccf_cols,
        "use_vrad_channel": bool(use_vrad_channel),
        "n_ccf_features": len(ccf_cols),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def train_and_save_final_model(
    output_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    ccf_cols: list[str],
    args: argparse.Namespace,
    vrad: np.ndarray | None,
) -> None:
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_s = x_scaler.fit_transform(X).astype(np.float32)
    y_s = y_scaler.fit_transform(y).astype(np.float32)

    if args.use_vrad_channel:
        X_in = add_vrad_channel(X_s, vrad)
        input_channels = 2
    else:
        X_in = X_s[..., np.newaxis]
        input_channels = 1

    set_global_seed(args.seed + 999999)
    model = build_model(
        input_length=X_in.shape[1],
        input_channels=input_channels,
        branch_filters=args.branch_filters,
        kernel_sizes=args.kernel_sizes,
        fusion_filters=args.fusion_filters,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        spatial_dropout=args.spatial_dropout,
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

    print("\nTraining final model on the full labeled dataset ...")
    start = perf_counter()
    history = model.fit(
        X_in,
        y_s,
        validation_split=args.final_validation_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    fit_seconds = perf_counter() - start

    model_path = final_dir / "rv_model.keras"
    history_path = final_dir / "training_history.csv"
    final_cfg_path = final_dir / "final_model_config.json"
    summary_path = final_dir / "model_summary.txt"

    model.save(model_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    save_preprocessing_artifacts(
        output_dir=final_dir,
        ccf_cols=ccf_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        use_vrad_channel=args.use_vrad_channel,
        vrad=vrad,
    )

    final_cfg = {
        "fit_seconds": float(fit_seconds),
        "epochs_ran": int(len(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "final_validation_fraction": float(args.final_validation_fraction),
    }
    final_cfg_path.write_text(json.dumps(final_cfg, indent=2))

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved final model to {model_path}")
    print(f"Saved preprocessing artifacts to {final_dir}")
    print(f"Final model epochs: {final_cfg['epochs_ran']}, best_val_loss={final_cfg['best_val_loss']:.6f}")

    keras.backend.clear_session()


def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.dev_fraction, args.test_fraction)
    if not 0.0 < args.final_validation_fraction < 1.0:
        raise ValueError("--final-validation-fraction must be between 0 and 1.")
    set_global_seed(args.seed)

    resolved_input_path = args.input_path or args.input_parquet
    if resolved_input_path is None:
        raise ValueError("Provide --input-path (or legacy --input-parquet).")

    cfg = ExperimentConfig(
        input_path=resolved_input_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        ccf_prefix=args.ccf_prefix,
        train_sizes=list(args.train_sizes),
        repeats=args.repeats,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        branch_filters=args.branch_filters,
        kernel_sizes=list(args.kernel_sizes),
        fusion_filters=list(args.fusion_filters),
        hidden_units=list(args.hidden_units),
        dropout=args.dropout,
        spatial_dropout=args.spatial_dropout,
        l2_reg=args.l2_reg,
        use_vrad_channel=bool(args.use_vrad_channel),
        vrad_start=args.vrad_start,
        vrad_stop=args.vrad_stop,
        vrad_step=args.vrad_step,
        save_final_model=bool(args.save_final_model),
        final_validation_fraction=args.final_validation_fraction,
    )

    input_path = Path(resolved_input_path)
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

    vrad = None
    if args.use_vrad_channel:
        vrad = build_vrad_grid(
            n_bins=X.shape[1],
            vrad_start=args.vrad_start,
            vrad_stop=args.vrad_stop,
            vrad_step=args.vrad_step,
        )
        vrad = standardize_vrad(vrad)
        print(f"Using VRAD channel with standardized grid length {len(vrad)}")
    else:
        print("Using CCF-only single-channel input")

    test_size = args.dev_fraction + args.test_fraction
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=test_size, random_state=args.seed, shuffle=True
    )

    relative_test_size = args.test_fraction / (args.dev_fraction + args.test_fraction)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_holdout, y_holdout, test_size=relative_test_size, random_state=args.seed, shuffle=True
    )

    print(f"Train full: {X_train_full.shape}, {y_train_full.shape}")
    print(f"Dev:        {X_dev.shape}, {y_dev.shape}")
    print(f"Test:       {X_test.shape}, {y_test.shape}")

    train_sizes = []
    max_train = len(X_train_full)
    for train_size in args.train_sizes:
        if train_size <= 0:
            continue
        if train_size <= max_train:
            train_sizes.append(train_size)
        else:
            print(f"Skipping train_size={train_size}: larger than available train rows ({max_train})")

    if not train_sizes:
        raise ValueError("No valid train sizes remain after filtering against the training split.")

    results = []
    total_runs = len(train_sizes) * args.repeats
    run_counter = 0

    for train_size in train_sizes:
        for repeat_idx in range(args.repeats):
            run_counter += 1
            run_seed = args.seed + 1000 * repeat_idx + train_size
            rng = np.random.default_rng(run_seed)
            subset_idx = rng.choice(max_train, size=train_size, replace=False)

            X_train = X_train_full[subset_idx]
            y_train = y_train_full[subset_idx]

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
            X_dev_s = x_scaler.transform(X_dev).astype(np.float32)
            X_test_s = x_scaler.transform(X_test).astype(np.float32)

            y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
            y_dev_s = y_scaler.transform(y_dev).astype(np.float32)

            if args.use_vrad_channel:
                X_train_in = add_vrad_channel(X_train_s, vrad)
                X_dev_in = add_vrad_channel(X_dev_s, vrad)
                X_test_in = add_vrad_channel(X_test_s, vrad)
                input_channels = 2
            else:
                X_train_in = X_train_s[..., np.newaxis]
                X_dev_in = X_dev_s[..., np.newaxis]
                X_test_in = X_test_s[..., np.newaxis]
                input_channels = 1

            set_global_seed(run_seed)
            model = build_model(
                input_length=X_train_in.shape[1],
                input_channels=input_channels,
                branch_filters=args.branch_filters,
                kernel_sizes=args.kernel_sizes,
                fusion_filters=args.fusion_filters,
                hidden_units=args.hidden_units,
                dropout=args.dropout,
                spatial_dropout=args.spatial_dropout,
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

            print(
                f"[{run_counter}/{total_runs}] "
                f"train_size={train_size}, repeat={repeat_idx + 1}/{args.repeats}, seed={run_seed}"
            )
            start = perf_counter()
            history = model.fit(
                X_train_in,
                y_train_s,
                validation_data=(X_dev_in, y_dev_s),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=0,
            )
            fit_seconds = perf_counter() - start

            y_dev_pred = y_scaler.inverse_transform(model.predict(X_dev_in, verbose=0)).ravel()
            y_test_pred = y_scaler.inverse_transform(model.predict(X_test_in, verbose=0)).ravel()

            dev_metrics = evaluate_predictions(y_dev, y_dev_pred)
            test_metrics = evaluate_predictions(y_test, y_test_pred)

            result = {
                "train_size": train_size,
                "repeat": repeat_idx,
                "seed": run_seed,
                "n_train_available": max_train,
                "n_dev": int(len(X_dev)),
                "n_test": int(len(X_test)),
                "epochs_ran": int(len(history.history["loss"])),
                "best_val_loss": float(np.min(history.history["val_loss"])),
                "final_train_loss": float(history.history["loss"][-1]),
                "fit_seconds": float(fit_seconds),
                "dev_mae": dev_metrics["mae"],
                "dev_rmse": dev_metrics["rmse"],
                "dev_r2": dev_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
            results.append(result)

            print(
                f"  dev_mae={result['dev_mae']:.6f}, "
                f"test_mae={result['test_mae']:.6f}, "
                f"epochs={result['epochs_ran']}"
            )

            keras.backend.clear_session()

    results_df = pd.DataFrame(results).sort_values(["train_size", "repeat"]).reset_index(drop=True)
    summary_df = (
        results_df.groupby("train_size", as_index=False)
        .agg(
            runs=("train_size", "size"),
            dev_mae_mean=("dev_mae", "mean"),
            dev_mae_std=("dev_mae", "std"),
            dev_rmse_mean=("dev_rmse", "mean"),
            dev_rmse_std=("dev_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values("train_size")
        .reset_index(drop=True)
    )

    results_path = output_dir / "learning_curve_runs.csv"
    summary_path = output_dir / "learning_curve_summary.csv"
    config_path = output_dir / "learning_curve_config.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(asdict(cfg), indent=2))

    print(f"Saved run-level results to {results_path}")
    print(f"Saved summary results to {summary_path}")
    print(f"Saved config to {config_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    if args.save_final_model:
        train_and_save_final_model(
            output_dir=output_dir,
            X=X,
            y=y,
            ccf_cols=ccf_cols,
            args=args,
            vrad=vrad,
        )


if __name__ == "__main__":
    main()
