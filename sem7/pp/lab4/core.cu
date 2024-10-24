__global__ void addKernel(int *c, int *a, int *b, unsigned int size) {
  // Код функции ядра
}

#define kernel addKernel
#include "mainGPU.h"
