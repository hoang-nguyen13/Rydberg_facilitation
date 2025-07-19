#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=32
#SBATCH -t 50:00:00
#SBATCH -p epyc-768
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem=0

GAMMA=1
DELTA=2000
V=$DELTA
NATOMS=2197
TF=1000
NT=1000
NTRAJ=32
CASE=3

PAIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params/pairs.txt)
read OMEGA GAMMA_DEPHASING <<< "$PAIR"

JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/dp_perc.jl
$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t auto --check-bounds=yes $SCRIPT $OMEGA $GAMMA_DEPHASING $GAMMA $DELTA $V $NATOMS $TF $NT $NTRAJ $CASE
