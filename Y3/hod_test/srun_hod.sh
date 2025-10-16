#!/bin/bash

# Function to activate environments

activate_environment() {
    case $1 in
        fit | plot)
            module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
            conda activate hod_env
            export NUMBA_DIAGNOSTICS=1
            ;;
    esac
}

# Function to run srun command
run_srun() {
    case $1 in
        fit)
            srun -N 4 -n 8 -c 64 -C cpu -t 04:00:00 --qos interactive --account desi python hod_fit_dv.py --tracer QSO --zind 0 --fitnum 1
            ;;
        plot)
            srun -N 1 -n 1 -c 64 -C cpu -t 04:00:00 --qos interactive --account desi python plot_hod.py --tracer QSO --zind 0 --fitind 0
            ;;
    esac
}

# Activate the appropriate environment
activate_environment $1

# Run the srun command
run_srun $1