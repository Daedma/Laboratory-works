#!/bin/bash

PROGRAMS=("lab3mulmpi" "lab3nonmpi" "lab3mulmpiq" "lab3nonmpiq" "lab3omp" "lab3ompq")
THREADS=(12)

mkdir -p -v results

for PROGRAM in "${PROGRAMS[@]}"; do
    for THREAD in "${THREADS[@]}"; do
        sbatch run_program.sh $PROGRAM $THREAD
    done
done
