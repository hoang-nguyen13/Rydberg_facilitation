#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -cpus-per-task=1
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu 8G
#SBATCH --array=0-25  

id = ${SLURM_ARRAY_TASK_ID}

# Run Julia with omega range
~/julia-1.11.2/bin/julia main.jl ${id}
