from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from create_spectrum_SOAP_local import (
    load_phoenix_from_npz,
    create_spectrum_soap_from_saved_phoenix,
)


def save_csv_spectrum(path: str | Path, wave: np.ndarray, flux: np.ndarray, flux_err: float = 0.001) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum_df = pd.DataFrame(
        {
            "wave_val": np.asarray(wave, dtype=float),
            "flux_val": np.asarray(flux, dtype=float),
            "flux_err": np.full(len(wave), float(flux_err), dtype=float),
        }
    )
    spectrum_df.to_csv(path, index=False)
    return spectrum_df


def run_local_phoenix_workflow(
    phoenix_npz_path: str | Path = "./saved_spectra/phoenix_output.npz",
    save_dir: str | Path = "./saved_spectra",
    save_csv: bool = True,
    soap_csv_name: str = "soap_local.csv",
) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wave_phoenix, flux_phoenix = load_phoenix_from_npz(
        npz_path=phoenix_npz_path,
        wave_key="wave_phoenix",
        flux_key="flux_phoenix",
    )

    soap_result = create_spectrum_soap_from_saved_phoenix(
        phoenix_npz_path=phoenix_npz_path,
        teff=5777.0,
        logg=4.44,
        z=0.0,
        psi=[0],
        grid=300,
        inst_reso=115000,
        skip_bis=True,
        skip_fwhm=True,
        skip_rv=False,
        verbose=False,
    )

    soap_npz_path = save_dir / "soap_output_local.npz"
    np.savez(
        soap_npz_path,
        wave_soap=np.asarray(soap_result.wave),
        flux_soap=np.asarray(soap_result.flux),
        ccf_flux_soap=np.asarray(soap_result.ccf_flux),
        rv_ccf_soap=np.asarray(soap_result.rv_ccf),
        spectrum_soap_wave=np.asarray(soap_result.input_spectrum.wave),
        spectrum_soap_flux=np.asarray(soap_result.input_spectrum.flux),
    )

    soap_csv_path = None
    if save_csv:
        soap_csv_path = save_dir / "csv_spectra" / soap_csv_name
        save_csv_spectrum(soap_csv_path, soap_result.wave, soap_result.flux[0])

    return {
        "wave_phoenix": wave_phoenix,
        "flux_phoenix": flux_phoenix,
        "soap_result": soap_result,
        "soap_npz": soap_npz_path,
        "soap_csv": soap_csv_path,
    }


def inspect_local_soap_npz(soap_npz_path: str | Path = "./saved_spectra/soap_output_local.npz") -> None:
    soap_npz_path = Path(soap_npz_path)
    if not soap_npz_path.exists():
        print(f"No local SOAP NPZ found: {soap_npz_path}")
        return

    data = np.load(soap_npz_path)
    print("Available keys:", list(data.files))
    for key in data.files:
        arr = np.asarray(data[key])
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SOAP using a locally saved PHOENIX NPZ (no online PHOENIX download)."
    )
    parser.add_argument(
        "--phoenix-npz",
        default="./saved_spectra/phoenix_output.npz",
        help="Path to local PHOENIX NPZ file.",
    )
    parser.add_argument(
        "--save-dir",
        default="./saved_spectra",
        help="Directory to save SOAP NPZ and optional CSV output.",
    )
    parser.add_argument(
        "--soap-csv-name",
        default="soap_local.csv",
        help="CSV filename to save inside <save-dir>/csv_spectra/.",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="If set, do not write the SOAP CSV spectrum file.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect existing soap_output_local.npz and exit.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.inspect_only:
        inspect_local_soap_npz(Path(args.save_dir) / "soap_output_local.npz")
        return

    result = run_local_phoenix_workflow(
        phoenix_npz_path=args.phoenix_npz,
        save_dir=args.save_dir,
        save_csv=not args.no_save_csv,
        soap_csv_name=args.soap_csv_name,
    )

    print("Saved:")
    print(" -", result["soap_npz"])
    print(" -", result["soap_csv"])

    inspect_local_soap_npz(result["soap_npz"])


if __name__ == "__main__":
    main()

