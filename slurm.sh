#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 200:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem=0
#SBATCH --array=0-42
#SBATCH --tasks-per-node=16

id=${SLURM_ARRAY_TASK_ID}

JULIA=~/julia-1.11.2/bin/julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl

MAX_RETRIES=2

run_julia() {
    local attempt=$1
    echo "Attempt $attempt of $MAX_RETRIES for TASK_ID=$id at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
    $JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 16 --check-bounds=yes $SCRIPT $id
    return $?
}

trap 'echo "Job terminated (signal caught) for TASK_ID=$id at $(date), last exit code: $?" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err; exit 1' SIGTERM SIGINT SIGABRT SIGSEGV

# Main retry loop
ATTEMPT=1
while [ $ATTEMPT -le $MAX_RETRIES ]; do
    run_julia $ATTEMPT
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed TASK_ID=$id successfully on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
        break
    elif [ $EXIT_CODE -eq 2 ]; then
        echo "Julia signaled retry (exit code 2) for TASK_ID=$id on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        ATTEMPT=$((ATTEMPT + 1))
        if [ $ATTEMPT -le $MAX_RETRIES ]; then
            echo "Retrying after 5 seconds..." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
            sleep 5
        else
            echo "Max retries reached for TASK_ID=$id at $(date). Requesting SLURM requeue." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
            scontrol requeue $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID
            exit 1
        fi
    else
        echo "Fatal error (exit code $EXIT_CODE) for TASK_ID=$id on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        exit 1
    fi
done
