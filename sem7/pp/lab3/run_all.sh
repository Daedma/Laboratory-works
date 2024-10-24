#!/bin/bash

PROGRAMS=("lab3mulmpi" "lab3nonmpi" "lab3mulmpiq" "lab3nonmpiq" "lab3omp" "lab3ompq")
THREADS=(3 9 12)

for PROGRAM in "${PROGRAMS[@]}"; do
    for THREAD in "${THREADS[@]}"; do
        sbatch run_program.sh $PROGRAM $THREAD
    done
done
