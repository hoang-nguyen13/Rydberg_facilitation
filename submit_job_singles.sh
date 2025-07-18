#!/bin/bash

OMEGA_VALUES=(5 10 15 20 30)
GAMMA_DEPHASING_VALUES=(0.1)

PAIRS=()
for gamma_dephasing in "${GAMMA_DEPHASING_VALUES[@]}"; do
    for omega in "${OMEGA_VALUES[@]}"; do
        PAIRS+=("$omega $gamma_dephasing")
    done
done

PAIRS_FILE="params/pairs.txt"
mkdir -p params
> "$PAIRS_FILE"
for pair in "${PAIRS[@]}"; do
    echo "$pair" >> "$PAIRS_FILE"
done

ARRAY_SIZE=$(( ${#PAIRS[@]} - 1 ))
sbatch --array=0-$ARRAY_SIZE run_job.slurm
