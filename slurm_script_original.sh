#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=16
#SBATCH -t 200:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem=0
#SBATCH --array=0-70

# Task ID from SLURM array
id=${SLURM_ARRAY_TASK_ID}

# Julia executable
JULIA=~/julia-1.11.2/bin/julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl

# Retry settings
MAX_RETRIES=2
ATTEMPT=1

while [ $ATTEMPT -le $MAX_RETRIES ]; do
    echo "Attempt $ATTEMPT of $MAX_RETRIES for TASK_ID=$id at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
    
    # Run Julia with 16 threads and project environment
    $JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 13 $SCRIPT $id
    
    # Check exit code
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed TASK_ID=$id successfully on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
        break
    else
        echo "Failed TASK_ID=$id with exit code $EXIT_CODE on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        ATTEMPT=$((ATTEMPT + 1))
        if [ $ATTEMPT -le $MAX_RETRIES ]; then
            echo "Retrying after 5 seconds..." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
            sleep 5
        else
            echo "Max retries reached for TASK_ID=$id. Check main.jl checkpoint for Ω=$id progress at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
            exit 1  # Indicate failure, but other Ω tasks continue
        fi
    fi
done
