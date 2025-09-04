#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=16
#SBATCH -t 50:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu=6G

GAMMA=1
DELTA=2000
V=$DELTA
NATOMS=15625
TF=1000
NT=400
NTRAJ=50
CASE=3

PAIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params/pairs.txt)
read OMEGA GAMMA_DEPHASING <<< "$PAIR"

JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl
$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t auto --check-bounds=yes $SCRIPT $OMEGA $GAMMA_DEPHASING $GAMMA $DELTA $V $NATOMS $TF $NT $NTRAJ $CASE
