#!/usr/bin/env python3
"""
Process CCF datasets and merge multiple row-wise CSV files.

Usage:
    python process_ccf.py --input input.csv --output output.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def process_ccf_dataset(input_path, output_path):
    """
    Process a CCF dataset: round values, normalize CCFs, and save to Parquet.

    Args:
        input_path (str or Path): Path to input CSV file
        output_path (str or Path): Path to output Parquet file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    # Load the dataset
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Round rv_true_mps to 2 decimal places
    if 'rv_true_mps' in df.columns:
        df['rv_true_mps'] = df['rv_true_mps'].round(2)
        print("Rounded rv_true_mps to 2 decimals")

    # Identify CCF columns (assuming they start with 'ccf_')
    ccf_columns = [col for col in df.columns if col.startswith('ccf_')]
    if not ccf_columns:
        print("Warning: No CCF columns found (columns starting with 'ccf_')")

    # Normalize CCFs by dividing by the maximum value in each row
    if ccf_columns:
        max_ccf = df[ccf_columns].max(axis=1)
        df[ccf_columns] = df[ccf_columns].div(max_ccf, axis=0)
        print("Normalized CCFs")

        # Round CCF values to 4 decimal places
        df[ccf_columns] = df[ccf_columns].round(4)
        print("Rounded CCFs to 4 decimals")

    # Save to Parquet
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Done!")


def concat_ccf_csv_files(input_dir, output_path=None, pattern="*.csv", sort_paths=True):
    """
    Concatenate all matching CSV files in a directory into one DataFrame.

    Args:
        input_dir (str or Path): Directory containing CSV files to merge.
        output_path (str or Path, optional): If provided, write the merged table
            to this path. Supported suffixes are .csv and .parquet.
        pattern (str): Glob pattern used to select files inside input_dir.
        sort_paths (bool): Sort matched paths before concatenation.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path {input_dir} is not a directory")

    csv_paths = list(input_dir.glob(pattern))
    if sort_paths:
        csv_paths = sorted(csv_paths)

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files matching pattern '{pattern}' were found in {input_dir}"
        )

    print(f"Found {len(csv_paths)} CSV files in {input_dir}")
    dfs = []
    for csv_path in csv_paths:
        print(f"Loading {csv_path}...")
        dfs.append(pd.read_csv(csv_path))

    df_merged = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Concatenated shape: {df_merged.shape}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".csv":
            df_merged.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == ".parquet":
            df_merged.to_parquet(output_path, index=False)
        else:
            raise ValueError(
                f"Unsupported output format for {output_path}. "
                "Use .csv or .parquet."
            )

        print(f"Saved concatenated dataset to {output_path}")

    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Process CCF dataset")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    process_ccf_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
