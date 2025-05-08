#!/bin/bash


declare -A OMEGA_RANGES
#OMEGA_RANGES[20]="$(seq 0 2 8; seq 8.5 0.15 13; seq 13.5 0.5 30 | xargs -n1 printf '%.2f\n')"
OMEGA_RANGES[0.1]="$(seq 0 4 24; seq 24.5 0.25 32; seq 32.5 0.5 60 | xargs -n1 printf '%.2f\n')"

GAMMA_DEPHASING_VALUES=(0.1)


PAIRS=()
for gamma_dephasing in "${GAMMA_DEPHASING_VALUES[@]}"; do
    readarray -t omega_values <<< "${OMEGA_RANGES[$gamma_dephasing]}"
    for omega in "${omega_values[@]}"; do
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

