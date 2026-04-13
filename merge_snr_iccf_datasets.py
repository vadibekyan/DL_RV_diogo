#!/usr/bin/env python3
"""
Merge SNR-specific ICCF CSV files, normalize them, save one parquet per SNR,
save one parquet with all SNR groups combined, and optionally append that
combined table to an existing full normalized parquet dataset.

Typical notebook usage:
    !python merge_snr_iccf_datasets.py
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


SNR_PATTERN = re.compile(r"_snr(?P<snr>[0-9]+p[0-9]+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SNR ICCF CSV files and append them to the full normalized parquet dataset."
    )
    parser.add_argument(
        "--snr-input-dir",
        default="hpc_results/random_rv_iccf_datasets_snr",
        help="Directory containing the SNR ICCF CSV files.",
    )
    parser.add_argument(
        "--ccfs-dir",
        default="CCFs",
        help="Directory where parquet outputs and summary CSV will be written.",
    )
    parser.add_argument(
        "--full-dataset-path",
        default="CCFs/full_iccf_dataset_normalized.parquet",
        help="Existing full normalized parquet to append the merged SNR data to.",
    )
    parser.add_argument(
        "--backup-path",
        default="CCFs/full_iccf_dataset_normalized_before_snr_merge.parquet",
        help="Backup path for the original full parquet before overwrite.",
    )
    parser.add_argument(
        "--skip-append-to-full",
        action="store_true",
        help="Only write the per-SNR and all-SNR parquet files, without modifying the full parquet.",
    )
    return parser.parse_args()


def extract_snr(path: Path) -> str:
    match = SNR_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract SNR from filename: {path.name}")
    return match.group("snr")


def sort_snr_key(snr: str) -> float:
    return float(snr.replace("p", "."))


def normalize_ccf_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rv_true_mps" in df.columns:
        df["rv_true_mps"] = df["rv_true_mps"].round(2)

    ccf_cols = [col for col in df.columns if col.startswith("ccf_")]
    if ccf_cols:
        max_ccf = df[ccf_cols].max(axis=1)
        df[ccf_cols] = df[ccf_cols].div(max_ccf, axis=0)
        df[ccf_cols] = df[ccf_cols].round(4)

    return df


def align_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = list(dict.fromkeys(list(left.columns) + list(right.columns)))

    left = left.copy()
    right = right.copy()

    for col in all_columns:
        if col not in left.columns:
            left[col] = pd.NA
        if col not in right.columns:
            right[col] = pd.NA

    return left[all_columns], right[all_columns]


def main() -> None:
    args = parse_args()

    snr_input_dir = Path(args.snr_input_dir)
    ccfs_dir = Path(args.ccfs_dir)
    full_dataset_path = Path(args.full_dataset_path)
    backup_path = Path(args.backup_path)

    if not snr_input_dir.exists():
        raise FileNotFoundError(f"SNR input directory not found: {snr_input_dir}")

    ccfs_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(snr_input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {snr_input_dir}")

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in csv_paths:
        grouped_paths[extract_snr(path)].append(path)

    print(f"Found {len(csv_paths)} CSV files across {len(grouped_paths)} SNR groups.")

    merged_snr_dfs = []
    summary_rows = []

    for snr in sorted(grouped_paths.keys(), key=sort_snr_key):
        paths = grouped_paths[snr]
        dfs = [pd.read_csv(path) for path in paths]
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = normalize_ccf_df(merged_df)
        merged_df["dataset_label"] = f"active_snr_{snr}"

        out_path = ccfs_dir / f"random_rv_iccf_datasets_snr_snr{snr}_merged_normalized.parquet"
        merged_df.to_parquet(out_path, index=False)

        merged_snr_dfs.append(merged_df)
        summary_rows.append(
            {
                "snr": snr,
                "n_files": len(paths),
                "n_rows": len(merged_df),
                "output_path": str(out_path),
            }
        )
        print(f"SNR {snr}: {len(paths)} files -> {len(merged_df)} rows")

    all_snr_df = pd.concat(merged_snr_dfs, axis=0, ignore_index=True)
    all_snr_path = ccfs_dir / "random_rv_iccf_datasets_snr_merged_normalized.parquet"
    all_snr_df.to_parquet(all_snr_path, index=False)
    print(f"Saved all-SNR parquet to {all_snr_path} with shape {all_snr_df.shape}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ccfs_dir / "random_rv_iccf_datasets_snr_merge_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved merge summary to {summary_path}")

    if args.skip_append_to_full:
        print("Skipping append to full dataset as requested.")
        return

    if not full_dataset_path.exists():
        raise FileNotFoundError(f"Full dataset parquet not found: {full_dataset_path}")

    full_df = pd.read_parquet(full_dataset_path)
    full_df, all_snr_df = align_columns(full_df, all_snr_df)
    combined_df = pd.concat([full_df, all_snr_df], axis=0, ignore_index=True)

    if not backup_path.exists():
        full_df.to_parquet(backup_path, index=False)
        print(f"Saved backup of original full dataset to {backup_path}")
    else:
        print(f"Backup already exists, leaving it unchanged: {backup_path}")

    combined_df.to_parquet(full_dataset_path, index=False)
    print(f"Overwrote {full_dataset_path} with combined dataset shape {combined_df.shape}")


if __name__ == "__main__":
    main()
