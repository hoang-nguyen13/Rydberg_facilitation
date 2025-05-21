#!/bin/bash

#SBATCH -J Rydberg
#SBATCH --cpus-per-task=1
#SBATCH -t 200:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem-per-cpu=5G

GAMMA=1
DELTA=2000
V=$DELTA
NATOMS=100
TF=1500
NT=1000
NTRAJ=1
CASE=2

PAIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params/pairs.txt)
read OMEGA GAMMA_DEPHASING <<< "$PAIR"

JULIA=julia
SCRIPT=/home/quw51vuk/Rydberg_facilitation/main.jl
$JULIA --project=/home/quw51vuk/Rydberg_facilitation -t auto --check-bounds=yes $SCRIPT $OMEGA $GAMMA_DEPHASING $GAMMA $DELTA $V $NATOMS $TF $NT $NTRAJ $CASE

