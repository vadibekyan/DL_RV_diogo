#!/usr/bin/env python3
"""
Process CCF dataset: round values, normalize CCFs, and save to Parquet.

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

def main():
    parser = argparse.ArgumentParser(description="Process CCF dataset")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    process_ccf_dataset(args.input, args.output)

if __name__ == "__main__":
    main()