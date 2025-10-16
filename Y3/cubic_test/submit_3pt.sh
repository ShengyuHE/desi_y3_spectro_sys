#!/bin/bash
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 24:00:00
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH -A desi
#SBATCH --output=./slurm/bk000-%j.out

module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate rc_env

srun python compute_cubic_3pt.py --tracers BGS LRG ELG QSO --mode 202