#!/bin/bash
# task: cat, 2pt, 3pt, bao, fs

# Function to activate environments
activate_environment() {
    case $1 in
        cat | 2pt | bao | fs)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            ;;
        3pt)
            module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
            conda activate rc_env
            ;;
    esac
}

# Function to run srun command
run_srun() {
    case $1 in
        cat)
            srun -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python create_spec_sys_catalogs.py 
            ;;
        2pt)
            # srun -n 1 -c 128 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi python compute_cubic_recon2pt.py
            srun -n 1 -c 128 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi python compute_cubic_2pt.py 
            ;;
        3pt)
            srun -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python compute_cubic_3pt.py 
            ;;
        bao)
            srun -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python fit_cubic_bao.py 
            ;;
        fs)
            srun -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python fit_cubic_full-shape.py
            ;;
    esac
}

# Check if a computation type was provided
if [ -z "$1" ]; then
    echo "Usage: ./srun_combined.sh [cat|2pt|3pt|bao|fs]"
    exit 1
fi

# Activate the appropriate environment
activate_environment $1

# Run the srun command
run_srun $1
