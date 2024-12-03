#ifndef ADD_KERNEL_H
#include "common.h"

__global__
void addKernel(TYPE* c, DECLARE_LIST_OF_ARGS(a), unsigned int size);

#define kernel addKernel

#endif