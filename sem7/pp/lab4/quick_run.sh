#!/bin/bash

PROGRAM="test"
GRID_SIZE=1024
BLOCK_SIZE=512
DIMDIV=1
N=3000
K=3
TYPE=TYPE_DOUBLE
OUTPUT_FILE="./results/${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV}.out"

./run_program.sh ${PROGRAM} ${GRID_SIZE} ${BLOCK_SIZE} ${DIMDIV} ${K} ${N} ${TYPE}

sleep 10 # Ждем выполнение скрипта

cat ${OUTPUT_FILE}