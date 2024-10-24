#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=1gb
#SBATCH --output=./results/$1-$2-results.out

PROGRAM=$1
THREADS=$2

if [[ $PROGRAM == "lab3nonmpi" || $PROGRAM == "lab3nonmpiq" || $PROGRAM == "lab3mulmpi" || $PROGRAM == "lab3mulmpiq" ]]; then
    mpirun -np $THREADS ./$PROGRAM
else
    export OMP_NUM_THREADS=$THREADS
    ./$PROGRAM
fi
