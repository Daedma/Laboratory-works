__global__ void addKernel(int *c, int *a, int *b, unsigned int size) {
  int gridSize = blockDim.x * gridDim.x;
  int start = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = start; i < size; i += gridSize) {
    c[i] = a[i] + b[i];
  }
}

#define kernel addKernel
#include "main.c"
