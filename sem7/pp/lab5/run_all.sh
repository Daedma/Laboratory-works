#!/bin/bash

PROGRAM="lab5"
N=(300 600 900)
TYPE=TYPE_DOUBLE
TRANSPOSE_A=1
TRANSPOSE_B=1

mkdir -p -v results

for dim in "${N[@]}"; do
    ./run_program.sh ${PROGRAM} ${dim} ${TYPE} ${TRANSPOSE_A} ${TRANSPOSE_B}
done
