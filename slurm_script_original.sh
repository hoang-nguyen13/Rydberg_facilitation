#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=16
#SBATCH -t 50:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem=0
#SBATCH --array=0-20

id=${SLURM_ARRAY_TASK_ID}

# Run Julia with omega range
~/julia-1.11.2/bin/julia -t 16 main.jl ${id}
