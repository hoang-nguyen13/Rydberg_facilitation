#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH -t 200:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem=0

JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl

srun $JULIA --project=/home/quw51vuk/Rydberg_facilitation $SCRIPT

