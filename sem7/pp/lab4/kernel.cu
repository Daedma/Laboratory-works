#include "common.h"

__global__ void addKernel(TYPE* c, DECLARE_LIST_OF_ARGS(a), unsigned int size)
{
	int gridSize = blockDim.x * gridDim.x;
	int start = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start; i < size; i += gridSize)
	{
		c[i] = APPLY_BIN_OP(a, i, +);
	}
}

#define kernel addKernel
#include "main.c"
