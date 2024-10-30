#!/bin/bash

PROGRAM="lab5"
N=(300 600 900)
for dim in "${N[@]}"; do
    ./run_program.sh ${PROGRAM} ${dim}
done
