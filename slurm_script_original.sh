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

# Julia executable and script
JULIA=~/julia-1.11.2/bin/julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl
CHECKPOINT_DIR=/home/quw51vuk/Rydberg_facilitation/checkpoints
CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint_Ω=${id}.jld2"

# Retry settings
MAX_RETRIES=2

# List of γ values (must match main.jl's γ_values)
GAMMA_VALUES="0.001 0.1 10 100"

# Function to get completed γ values from checkpoint
get_completed_gamma() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        COMPLETED_GAMMA=$($JULIA -e "using JLD2; print(join(load(\"$CHECKPOINT_FILE\", \"completed_γ\"), \" \"))" 2>/dev/null)
        echo "$COMPLETED_GAMMA"
    else
        echo ""
    fi
}

# Function to find the failed γ
find_failed_gamma() {
    local completed_gamma="$1"
    local last_gamma=$(grep "Starting TWA computation for γ" $SLURM_SUBMIT_DIR/out/%x_%A_%a.out | tail -1 | grep -o "γ = [0-9e.-]*" | awk '{print $3}')
    
    if [ -n "$last_gamma" ]; then
        if ! echo "$completed_gamma" | grep -q "$last_gamma"; then
            echo "$last_gamma"
            return
        fi
    fi
    
    for gamma in $GAMMA_VALUES; do
        if ! echo "$completed_gamma" | grep -q "$gamma"; then
            echo "$gamma"
            return
        fi
    done
    echo ""
}

# Function to run Julia and capture exit code
run_julia() {
    local attempt=$1
    local gamma=$2
    echo "Attempt $attempt of $MAX_RETRIES for TASK_ID=$id at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
    if [ -z "$gamma" ]; then
        $JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 13 $SCRIPT $id
    else
        echo "Retrying only γ = $gamma for TASK_ID=$id" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
        $JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 13 $SCRIPT $id $gamma
    fi
    return $?
}

# Ensure checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
fi

# Main retry loop
ATTEMPT=1
while [ $ATTEMPT -le $MAX_RETRIES ]; do
    if [ $ATTEMPT -eq 1 ]; then
        run_julia $ATTEMPT ""
    else
        run_julia $ATTEMPT "$LAST_GAMMA"
    fi
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Completed TASK_ID=$id successfully on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/out/%x_%A_%a.out
        break
    fi

    echo "Detected failure (exit code $EXIT_CODE) for TASK_ID=$id on attempt $ATTEMPT at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
    
    COMPLETED_GAMMA=$(get_completed_gamma)
    echo "Completed γ values for Ω=$id: $COMPLETED_GAMMA" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
    
    LAST_GAMMA=$(find_failed_gamma "$COMPLETED_GAMMA")
    if [ -n "$LAST_GAMMA" ]; then
        echo "Determined γ = $LAST_GAMMA failed with exit code $EXIT_CODE. Preparing to retry..." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
    else
        echo "Could not determine failed γ or all γ completed. No retry possible for TASK_ID=$id." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        LAST_GAMMA=""
    fi

    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -le $MAX_RETRIES ]; then
        echo "Retrying after 5 seconds..." >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        sleep 5
    else
        echo "Max retries reached for TASK_ID=$id. Check checkpoint for Ω=$id progress at $(date)" >> $SLURM_SUBMIT_DIR/err/%x_%A_%a.err
        exit 1
    fi
done
