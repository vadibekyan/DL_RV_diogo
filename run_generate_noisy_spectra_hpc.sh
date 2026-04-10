#!/bin/bash
#SBATCH --account=f202418352cpcaa1a
#SBATCH --job-name=generate_noisy_spectra
#SBATCH --mail-user=vadibekyan@astro.up.pt
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/F202418352CPCAA1/logs/generate_noisy_spectra_%j.out
#SBATCH --error=/projects/F202418352CPCAA1/logs/generate_noisy_spectra_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

set -euo pipefail

module purge
module load Python/3.11
source /projects/F202418352CPCAA1/soap_env/bin/activate

export PIP_CACHE_DIR=/projects/F202418352CPCAA1/pip_cache
export TMPDIR=/projects/F202418352CPCAA1/tmp
export MPLCONFIGDIR=/projects/F202418352CPCAA1/tmp/matplotlib
export XDG_CACHE_HOME=/projects/F202418352CPCAA1/cache

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /projects/F202418352CPCAA1/logs
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/astropy"

cd /projects/F202418352CPCAA1

/projects/F202418352CPCAA1/soap_env/bin/python generate_noisy_spectra_hpc.py \
  --input-dir /projects/F202418352CPCAA1/hpc_results/csv_spectra \
  --output-dir /projects/F202418352CPCAA1/hpc_results/csv_spectra_noisy_SNR100 \
  --snr 100 \
  --n-realizations 20 \
  --n-workers "$SLURM_CPUS_PER_TASK"
  
# for snr in 30 200 300; do
#  /projects/F202418352CPCAA1/soap_env/bin/python generate_noisy_spectra_hpc.py \
#    --input-dir /projects/F202418352CPCAA1/hpc_results/csv_spectra \
#    --output-dir /projects/F202418352CPCAA1/hpc_results/csv_spectra_noisy \
#    --snr "$snr" \
#    --n-realizations 100 \
#    --n-workers "$SLURM_CPUS_PER_TASK"
# done

  

# Alternative: process only a subset listed in a text file
# /projects/F202418352CPCAA1/soap_env/bin/python generate_noisy_spectra_hpc.py \
#   --input-list /projects/F202418352CPCAA1/spectra_to_process.txt \
#   --output-dir /projects/F202418352CPCAA1/hpc_results/csv_spectra_noisy \
#   --snr 100 \
#   --n-realizations 100 \
#   --n-workers "$SLURM_CPUS_PER_TASK"
