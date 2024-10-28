# Makefile для компиляции программы с различными параметрами

# Параметры для компиляции
N = 6300000
K = 3
TYPE = TYPE_DOUBLE
GRID_SIZES = 1024 512
BLOCK_SIZES = 1024 512 256
DIMDIVS = 1 3 9

# Компилятор CUDA
NVCC = nvcc

# Флаги компилятора
CFLAGS = -g -G -O0 -lcublas -DN=$(N) -DK=$(K) -D$(TYPE)

# Исходные файлы
SRCS = main.c kernel.cu

# Имя исполняемого файла
EXEC = lab4

# Цель по умолчанию
all: $(foreach grid,$(GRID_SIZES),$(foreach block,$(BLOCK_SIZES),$(foreach dimdiv,$(DIMDIVS),$(EXEC)_$(grid)_$(block)_$(dimdiv))))

# Правило для компиляции с различными параметрами
$(EXEC)_%_%_%:
	$(NVCC) $(CFLAGS) -DGRID_SIZE=$(word 1,$(subst _,_ ,$@)) -DBLOCK_SIZE=$(word 2,$(subst _,_ ,$@)) -DDIMDIV=$(word 3,$(subst _,_ ,$@)) -o $@ $(SRCS)

# Правило для очистки
clean:
	rm -f $(EXEC)_*

# Правило для полной пересборки проекта
rebuild: clean all

.PHONY: all clean rebuild
