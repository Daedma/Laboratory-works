#!/bin/bash

PROGRAM="lab4"
GRID_SIZES=(1024 512)
BLOCK_SIZES=(1024 512 256)
DIMDIVS=(1 3 9)

mkdir -p -v results

for GRID_SIZE in "${GRID_SIZES[@]}"; do
    for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
        for DIMDIV in "${DIMDIVS[@]}"; do
            ./run_program.sh $PROGRAM $GRID_SIZE $BLOCK_SIZE $DIMDIV
        done
    done
done
