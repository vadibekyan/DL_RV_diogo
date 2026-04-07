from __future__ import annotations

from pathlib import Path

from generate_random_rv_ccf_dataset import generate_random_rv_ccf_dataset


# Edit these values directly in this file.
INPUT_SPECTRUM = Path("/projects/F202418352CPCAA1/saved_spectra/csv_spectra/soap_local.csv")
MASK_PATH = Path("/projects/F202418352CPCAA1/saved_spectra/mask.csv")
OUTPUT_CSV = Path("/projects/F202418352CPCAA1/saved_spectra/random_100ccf_with_rv_0p001step.csv")

N_SAMPLES = 100
RV_MIN = 0.0
RV_MAX = 100.0
SEED = 42
RESOLUTION = 140000.0
MEDIUM = "vac"
MASK_MEDIUM = "vac"
VRAD_GRID = (-15.0, 15.0, 0.01)
EXCLUDE_TELLURICS = False
CCF_ERR_SCALE = True
TARGET = "Sun"


def main() -> None:
    df_out = generate_random_rv_ccf_dataset(
        input_spectrum_csv=INPUT_SPECTRUM,
        mask_path=MASK_PATH,
        output_csv=OUTPUT_CSV,
        n_samples=N_SAMPLES,
        rv_min=RV_MIN,
        rv_max=RV_MAX,
        seed=SEED,
        resolution=RESOLUTION,
        medium=MEDIUM,
        mask_medium=MASK_MEDIUM,
        vrad_grid=VRAD_GRID,
        exclude_tellurics=EXCLUDE_TELLURICS,
        ccf_err_scale=CCF_ERR_SCALE,
        target=TARGET,
    )
    print(f"Saved dataset: {OUTPUT_CSV}")
    print(f"Shape: {df_out.shape}")


if __name__ == "__main__":
    main()
