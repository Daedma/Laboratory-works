#!/bin/bash

PROGRAM=$1
# Данные варианта
GRID_SIZE=$2
BLOCK_SIZE=$3
DIMDIV=$4
K=${5:-3}  # Используем значение по умолчанию, если K не задан
N=${6:-6300000}  # Используем значение по умолчанию, если N не задан
TYPE=${7:-"TYPE_DOUBLE"}

JOB_NAME="${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV}"
OUTPUT_FILE="./results/${JOB_NAME}.out"

# Создаем временный файл для задачи SLURM
TEMP_FILE=$(mktemp)

cat <<EOF > "$TEMP_FILE"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --output=${OUTPUT_FILE}

module load cuda/8.0

make clean EXEC=${PROGRAM}
make ${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV} K=${K} N=${N} EXEC=${PROGRAM} TYPE=${TYPE}

./${PROGRAM}_${GRID_SIZE}_${BLOCK_SIZE}_${DIMDIV}
EOF

mkdir -p -v results

# Отправляем задачу SLURM
sbatch "$TEMP_FILE"

# Удаляем временный файл
rm "$TEMP_FILE"
