#!/usr/bin/env python3
"""
Run a learning-curve experiment for RV prediction on a processed dataset using
a shallow 1D CNN with optional VRAD positional channel.

Model:
    Conv1D -> Conv1D -> Flatten -> Dense(s) -> Output
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
    conv_filters: list[int]
    kernel_sizes: list[int]
    hidden_units: list[int]
    dropout: float
    l2_reg: float
    use_vrad_channel: bool
    vrad_start: float
    vrad_stop: float
    vrad_step: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a learning-curve experiment for RV regression using a shallow 1D CNN."
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
    parser.add_argument("--patience", type=int, default=15, help="EarlyStopping patience on val_loss.")
    parser.add_argument("--lr-patience", type=int, default=10, help="ReduceLROnPlateau patience on val_loss.")
    parser.add_argument("--lr-factor", type=float, default=0.7, help="LR reduction factor on plateau.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument(
        "--conv-filters",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Filters in the two Conv1D layers, e.g. --conv-filters 16 32",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=[9, 5],
        help="Kernel sizes in the two Conv1D layers, e.g. --kernel-sizes 9 5",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs="+",
        default=[64],
        help="Dense hidden-layer sizes after Flatten, e.g. --hidden-units 64 or 128 64",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate after dense hidden layers.")
    parser.add_argument("--l2-reg", type=float, default=0.0, help="L2 regularization coefficient.")
    parser.add_argument(
        "--use-vrad-channel",
        action="store_true",
        help="If set, stack a second constant channel containing the VRAD grid.",
    )
    parser.add_argument("--vrad-start", type=float, default=-15.0, help="Start of VRAD grid.")
    parser.add_argument("--vrad-stop", type=float, default=15.0, help="Stop of VRAD grid.")
    parser.add_argument("--vrad-step", type=float, default=0.1, help="Step of VRAD grid.")
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


def add_vrad_channel(X_2d: np.ndarray, vrad: np.ndarray) -> np.ndarray:
    vrad_channel = np.broadcast_to(vrad, X_2d.shape).astype(np.float32)
    return np.stack([X_2d, vrad_channel], axis=-1).astype(np.float32)


def _regularizer(l2_reg: float):
    return keras.regularizers.l2(l2_reg) if l2_reg > 0 else None


def build_model(
    input_length: int,
    input_channels: int,
    conv_filters: list[int],
    kernel_sizes: list[int],
    hidden_units: list[int],
    dropout: float,
    learning_rate: float,
    l2_reg: float,
) -> keras.Model:
    if len(conv_filters) != 2:
        raise ValueError("This model expects exactly two values in --conv-filters.")
    if len(kernel_sizes) != 2:
        raise ValueError("This model expects exactly two values in --kernel-sizes.")

    reg = _regularizer(l2_reg)

    model = keras.Sequential(name="shallow_conv1d_flatten_rv_regressor")
    model.add(layers.Input(shape=(input_length, input_channels)))
    model.add(
        layers.Conv1D(
            conv_filters[0],
            kernel_sizes[0],
            activation="relu",
            padding="same",
            kernel_regularizer=reg,
            name="conv1d_1",
        )
    )
    model.add(
        layers.Conv1D(
            conv_filters[1],
            kernel_sizes[1],
            activation="relu",
            padding="same",
            kernel_regularizer=reg,
            name="conv1d_2",
        )
    )
    model.add(layers.Flatten(name="flatten"))

    for i, units in enumerate(hidden_units, start=1):
        if units <= 0:
            continue
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=reg,
                name=f"dense_{i}",
            )
        )
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i}"))

    model.add(layers.Dense(1, name="rv_output"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_fraction, args.dev_fraction, args.test_fraction)
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
        conv_filters=list(args.conv_filters),
        kernel_sizes=list(args.kernel_sizes),
        hidden_units=list(args.hidden_units),
        dropout=args.dropout,
        l2_reg=args.l2_reg,
        use_vrad_channel=bool(args.use_vrad_channel),
        vrad_start=args.vrad_start,
        vrad_stop=args.vrad_stop,
        vrad_step=args.vrad_step,
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
        print(f"Using VRAD channel with grid length {len(vrad)}")
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
                conv_filters=args.conv_filters,
                kernel_sizes=args.kernel_sizes,
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


if __name__ == "__main__":
    main()
