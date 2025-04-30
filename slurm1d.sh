#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=16
#SBATCH -t 200:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu=7G

GAMMA=1
DELTA=2000
V=$DELTA
NATOMS=1500
TF=160
NT=400
NTRAJ=32
CASE=2

OMEGA_VALUES=($(seq 0 2 8; seq 8.5 0.25 13; seq 14 0.5 30 | xargs -n1 printf "%.2f"))
GAMMA_VALUES=(20)

PAIRS=()
for omega in "${OMEGA_VALUES[@]}"; do
    for gamma in "${GAMMA_VALUES[@]}"; do
        PAIRS+=("$omega $gamma")
    done
done

#SBATCH --array=0-$(( ${#PAIRS[@]} - 1 ))

id=${SLURM_ARRAY_TASK_ID}
PAIR=${PAIRS[$id]}
read OMEGA GAMMA <<< "$PAIR"
JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main1d.jl
$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 16 --check-bounds=yes $SCRIPT $OMEGA $GAMMA $GAMMA $DELTA $V $NATOMS $TF $NT $NTRAJ $CASE