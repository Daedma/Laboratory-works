#!/bin/bash
#PBS -N matMulGPU
#PBS -l walltime=00:01:10
#PBS -l nodes=1:ppn=1:gpu
#PBS -j oe
#PBS -A tk
cd $PBS_O_WORKDIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/COMMON/cuda-6.5/lib64
./Add 20000000