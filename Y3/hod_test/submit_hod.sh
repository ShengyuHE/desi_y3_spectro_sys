#!/bin/bash
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -c 32
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name=fit_HOD
#SBATCH --time=31:00:00
#SBATCH --account desi
#SBATCH --output=./slurms/hod/mini-QSO-0-%A_%a.out
#SBATCH --array=0-1

module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate hod_env

# List of systematics models
SYSTEMATICS=(standard dv-obs)
# SYSTEMATICS=(standard dv-LRG dv-QSO-G dv-QSO-L dv-ELGcatas dv-slitless)
SYS=${SYSTEMATICS[$SLURM_ARRAY_TASK_ID]}

# List of systematics models
srun python hod_fit_dv.py --systematics "$SYS" --tracer QSO --zind 3 --fitnum 10