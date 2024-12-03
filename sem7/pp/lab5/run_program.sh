#!/bin/bash

PROGRAM=$1
N=$2
TYPE=$3
TRANSPOSE_A=$4
TRANSPOSE_B=$5
JOB_NAME="${PROGRAM}_${N}"
OUTPUT_FILE="./results/${JOB_NAME}-results.out"

# Создаем временный файл для задачи SLURM
TEMP_FILE=$(mktemp)

cat <<EOF > $TEMP_FILE
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:03:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=1gb
#SBATCH --output=${OUTPUT_FILE}

module load cuda/8.0

make clean EXEC=${PROGRAM}
make ${PROGRAM}_${N} EXEC=${PROGRAM} TYPE=${TYPE} TRANSPOSE_A=${TRANSPOSE_A} TRANSPOSE_B=${TRANSPOSE_B}

./${PROGRAM}_${N}
EOF

# Отправляем задачу SLURM
sbatch $TEMP_FILE

# Удаляем временный файл
rm $TEMP_FILE