# Makefile для компиляции программы с различными параметрами

# Параметры для компиляции
N ?= 300 600 900
TYPE ?= TYPE_DOUBLE
TRANSPOSE_A ?= 1
TRANSPOSE_B ?= 1

# Компилятор CUDA
NVCC = nvcc

# Флаги компилятора
CFLAGS = -g -G -O0 -lcublas -lcurand -D$(TYPE) -DTRANSPOSE_A=$(TRANSPOSE_A) -DTRANSPOSE_B=$(TRANSPOSE_B)

# Имя исполняемого файла
EXEC = lab5

# Цель по умолчанию
all: $(foreach dim,$(N),$(EXEC)_$(dim))

# Правило для компиляции с различными параметрами
$(EXEC)_%: main.c
	$(NVCC) $(CFLAGS) -DN=$(strip $*) -o $@ $<

# Правило для очистки
clean:
	rm -f $(EXEC)_*

# Правило для полной пересборки проекта
rebuild: clean all

.PHONY: all clean rebuild
