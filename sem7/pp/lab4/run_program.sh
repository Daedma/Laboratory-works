#!/bin/bash

PROGRAM=$1
GRID_SIZE=$2
BLOCK_SIZE=$3
DIMDIV=$4
JOB_NAME="${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV}"
OUTPUT_FILE="./results/${JOB_NAME}-results.out"

# Создаем временный файл для задачи PBS
TEMP_FILE=$(mktemp)

cat <<EOF > $TEMP_FILE
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=1gb
#SBATCH --output=${OUTPUT_FILE}

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/COMMON/cuda-6.5/lib64
./${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV}
EOF

# Отправляем задачу SLURM
sbatch $TEMP_FILE

# Удаляем временный файл
rm $TEMP_FILE
