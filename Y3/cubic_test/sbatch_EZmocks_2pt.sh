#!/bin/bash
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 22:00:00
#SBATCH -C gpu
#SBATCH --gpus=4
#SBATCH -q regular
#SBATCH -A desi_g

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun python compute_cubic_recon2pt.py