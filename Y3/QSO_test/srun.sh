#!/bin/bash

# Function to activate environments
activate_environment() {
    case $1 in
        cp | add_dv)
            source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
            ;;
    esac
}

# Function to run srun command
run_srun() {
    case $1 in
        cp)
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python Y3_QSO_copy.py
            ;;
        add_dv)
            # implement the redshift error
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python Y3_QSO_error.py
            srun -N 2 -n 128 -c 4 -C cpu -t 04:00:00 --qos interactive --account desi python Y3_QSO_error_ran_MPI.py
            srun -N 1 -n 1 -c 128 -C cpu -t 04:00:00 --qos interactive --account desi python Y3_QSO_error_NX_FKP.py
            ;;
    esac
}

# Check if a computation type was provided
if [ -z "$1" ]; then
    echo "Usage: ./srun_combined.sh [cp|add_dv]"
    exit 1
fi

# Activate the appropriate environment
activate_environment $1

# Run the srun command
run_srun $1

