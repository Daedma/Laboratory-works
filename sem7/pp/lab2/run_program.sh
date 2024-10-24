#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=1gb
#SBATCH --output=./results/$1-$2-results.out

PROGRAM=$1
THREADS=$2

if [[ $PROGRAM == "lab2mpi" || $PROGRAM == "lab2mpiq" ]]; then
    export I_MPI_LIBRARY=/usr/lib64/slurm/mpi_pmi2.so
    srun --mpi=pmi2 -n $THREADS ./$PROGRAM
else
    export OMP_NUM_THREADS=$THREADS
    ./$PROGRAM
fi
