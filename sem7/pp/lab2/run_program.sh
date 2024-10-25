#!/bin/bash

PROGRAM=$1
THREADS=$2
JOB_NAME="${PROGRAM}-${THREADS}"
OUTPUT_FILE="./results/${JOB_NAME}-results.out"

cat <<EOF | sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --mem=1gb
#SBATCH --output=${OUTPUT_FILE}

if [[ ${PROGRAM} == "lab2mpi" || ${PROGRAM} == "lab2mpiq" ]]; then
    export I_MPI_LIBRARY=/usr/lib64/slurm/mpi_pmi2.so
    srun --mpi=pmi2 -n ${THREADS} ./${PROGRAM}
else
    export OMP_NUM_THREADS=${THREADS}
    ./${PROGRAM}
fi
EOF
