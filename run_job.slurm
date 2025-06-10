#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=2
#SBATCH -t 400:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu=6G

GAMMA=1
DELTA=2000
V=$DELTA
NATOMS=81
TF=10000
NT=800
NTRAJ=50
CASE=2

PAIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params/pairs.txt)
read OMEGA GAMMA_DEPHASING <<< "$PAIR"

JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl
$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t 2 --check-bounds=yes $SCRIPT $OMEGA $GAMMA_DEPHASING $GAMMA $DELTA $V $NATOMS $TF $NT $NTRAJ $CASE
