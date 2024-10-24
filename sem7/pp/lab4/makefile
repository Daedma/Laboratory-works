# The size of shared memory block size
BLOCK_SIZE = 128
NVCC = nvcc 
CFLAGS = -g -G -O0 -DBLOCK_SIZE=$(BLOCK_SIZE) -lcublas
Add: kernel.cu main.c
	$(NVCC) $(CFLAGS) $< -o $@