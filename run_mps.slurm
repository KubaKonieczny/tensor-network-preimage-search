#!/bin/bash -l
#SBATCH --job-name=md5_mps
#SBATCH --output=logs_%A_%a.out
#SBATCH --error=logs_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350GB
#SBATCH --account=plgkwantowy5-cpu-bigmem
#SBATCH --partition=plgrid-bigmem
#SBATCH --array=0-1

module load python/3.10

SCRATCH_DIR="${PLG_USER_SCRATCH:-${SCRATCH:-/net/scratch/users/plg${USER}}}"
VENV_DIR="${SCRATCH_DIR}/quantum_venv"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if ! python3 -c "import quimb" 2>/dev/null; then
    pip install --upgrade pip
    pip install "quimb>=1.11.1" "numpy>=1.25" scipy opt_einsum psutil }
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

N_VALUES=(1  2)
W_VALUES=(16 32)
MPS_BONDS_LIST=("8 16 32 64 128 256 512 1024" "64 128 256 512 1024 2048")
N_HASHES_LIST=(10 10)

N=${N_VALUES[${SLURM_ARRAY_TASK_ID}]}
W=${W_VALUES[${SLURM_ARRAY_TASK_ID}]}
MPS_BONDS=${MPS_BONDS_LIST[${SLURM_ARRAY_TASK_ID}]}
N_HASHES=${N_HASHES_LIST[${SLURM_ARRAY_TASK_ID}]}

cd "${SLURM_SUBMIT_DIR}"

python3 -u benchmark_mps.py \
    --n ${N} \
    --w ${W} \
    --mps-bonds ${MPS_BONDS} \
    --include-uncompressed \
    --n-hashes ${N_HASHES} \
    --output "results_mps_${SLURM_ARRAY_JOB_ID}_n${N}w${W}.json"
