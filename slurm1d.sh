#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=16
#SBATCH -t 200:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu=7G
#SBATCH --array=0-97

id=${SLURM_ARRAY_TASK_ID}

JULIA=~/julia-1.11.2/bin/julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main1d.jl

$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 16 --check-bounds=yes $SCRIPT $id
