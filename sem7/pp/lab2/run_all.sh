#!/bin/bash

PROGRAMS=("lab2mpi" "lab2mpiq" "lab2omp" "lab2ompq")
THREADS=(3 9 12)

mkdir -p -v results

for PROGRAM in "${PROGRAMS[@]}"; do
    for THREAD in "${THREADS[@]}"; do
        sbatch run_program.sh $PROGRAM $THREAD
    done
done
