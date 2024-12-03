#!/bin/bash

PROGRAM="lab4"
GRID_SIZES=(512)
BLOCK_SIZES=(1024 512 256)
DIMDIVS=(1 3 9)
K=3
N=6300000
TYPE=TYPE_DOUBLE
# TYPE=TYPE_FLOAT
# TYPE=TYPE_COMPLEX

# Запускаем по 6 задач
for DIMDIV in "${DIMDIVS[@]}"; do
	for GRID_SIZE in "${GRID_SIZES[@]}"; do
		for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
            ./run_program.sh $PROGRAM "$GRID_SIZE" "$BLOCK_SIZE" "$DIMDIV" ${K} ${N} ${TYPE}
        done
    done
	sleep 20 # Выжидаем 20 секунд, чтобы предыдущие задачи успели выполниться
done
