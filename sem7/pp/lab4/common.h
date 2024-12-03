#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined(TYPE_DOUBLE)
#define TYPE double
#define TYPE_FORMAT "%f"
#elif defined(TYPE_FLOAT)
#define TYPE float
#define TYPE_FORMAT "%f"
#elif defined(TYPE_INT)
#define TYPE int
#define TYPE_FORMAT "%d"
#else
#define TYPE double
#define TYPE_FORMAT "%f"
#endif

#ifndef K
#define K 3
#endif

#ifndef DIMDIV
#define DIMDIV 1
#endif

#ifndef N
#define N 6300000/DIMDIV
#endif

#ifndef DIMDIV1
#define DIMDIV1 3
#endif

#ifndef DIMDIV2
#define DIMDIV2 9
#endif

#ifndef GRID_SIZE
#define GRID_SIZE 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define STR(x) STR_HELPER(x)
#define STR_HELPER(x) #x

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define VECTORS_1(x) TYPE *x##1;
#define VECTORS_2(x) VECTORS_1(x) TYPE *x##2;
#define VECTORS_3(x) VECTORS_2(x) TYPE *x##3;
#define VECTORS_4(x) VECTORS_3(x) TYPE *x##4;
#define VECTORS_5(x) VECTORS_4(x) TYPE *x##5;
#define VECTORS_6(x) VECTORS_5(x) TYPE *x##6;
#define VECTORS_7(x) VECTORS_6(x) TYPE *x##7;
#define VECTORS_8(x) VECTORS_7(x) TYPE *x##8;
#define VECTORS_9(x) VECTORS_8(x) TYPE *x##9;
#define VECTORS_10(x) VECTORS_9(x) TYPE *x##10;
#define DECLARE_VECTORS(x) CAT(VECTORS_, K)(x)

#define APPLY_BIN_OP_1(x, index, op) x##1 [index]
#define APPLY_BIN_OP_2(x, index, op)                                           \
  (APPLY_BIN_OP_1(x, index, op) op x##2 [index])
#define APPLY_BIN_OP_3(x, index, op)                                           \
  (APPLY_BIN_OP_2(x, index, op) op x##3 [index])
#define APPLY_BIN_OP_4(x, index, op)                                           \
  (APPLY_BIN_OP_3(x, index, op) op x##4 [index])
#define APPLY_BIN_OP_5(x, index, op)                                           \
  (APPLY_BIN_OP_4(x, index, op) op x##5 [index])
#define APPLY_BIN_OP_6(x, index, op)                                           \
  (APPLY_BIN_OP_5(x, index, op) op x##6 [index])
#define APPLY_BIN_OP_7(x, index, op)                                           \
  (APPLY_BIN_OP_6(x, index, op) op x##7 [index])
#define APPLY_BIN_OP_8(x, index, op)                                           \
  (APPLY_BIN_OP_7(x, index, op) op x##8 [index])
#define APPLY_BIN_OP_9(x, index, op)                                           \
  (APPLY_BIN_OP_8(x, index, op) op x##9 [index])
#define APPLY_BIN_OP_10(x, index, op)                                          \
  (APPLY_BIN_OP_9(x, index, op) op x##10 [index])
#define APPLY_BIN_OP(x, index, op) CAT(APPLY_BIN_OP_, K)(x, index, op)

#define APPLY_FUNCTION_1(x, func) func(x##1);
#define APPLY_FUNCTION_2(x, func) APPLY_FUNCTION_1(x, func) func(x##2);
#define APPLY_FUNCTION_3(x, func) APPLY_FUNCTION_2(x, func) func(x##3);
#define APPLY_FUNCTION_4(x, func) APPLY_FUNCTION_3(x, func) func(x##4);
#define APPLY_FUNCTION_5(x, func) APPLY_FUNCTION_4(x, func) func(x##5);
#define APPLY_FUNCTION_6(x, func) APPLY_FUNCTION_5(x, func) func(x##6);
#define APPLY_FUNCTION_7(x, func) APPLY_FUNCTION_6(x, func) func(x##7);
#define APPLY_FUNCTION_8(x, func) APPLY_FUNCTION_7(x, func) func(x##8);
#define APPLY_FUNCTION_9(x, func) APPLY_FUNCTION_8(x, func) func(x##9);
#define APPLY_FUNCTION_10(x, func) APPLY_FUNCTION_9(x, func) func(x##10);
#define APPLY_FUNCTION(x, func) CAT(APPLY_FUNCTION_, K)(x, func)

#define APPLY_FUNCTION2_1(x, y, func) func(x##1, y##1);
#define APPLY_FUNCTION2_2(x, y, func)                                          \
  APPLY_FUNCTION2_1(x, y, func) func(x##2, y##2);
#define APPLY_FUNCTION2_3(x, y, func)                                          \
  APPLY_FUNCTION2_2(x, y, func) func(x##3, y##3);
#define APPLY_FUNCTION2_4(x, y, func)                                          \
  APPLY_FUNCTION2_3(x, y, func) func(x##4, y##4);
#define APPLY_FUNCTION2_5(x, y, func)                                          \
  APPLY_FUNCTION2_4(x, y, func) func(x##5, y##5);
#define APPLY_FUNCTION2_6(x, y, func)                                          \
  APPLY_FUNCTION2_5(x, y, func) func(x##6, y##6);
#define APPLY_FUNCTION2_7(x, y, func)                                          \
  APPLY_FUNCTION2_6(x, y, func) func(x##7, y##7);
#define APPLY_FUNCTION2_8(x, y, func)                                          \
  APPLY_FUNCTION2_7(x, y, func) func(x##8, y##8);
#define APPLY_FUNCTION2_9(x, y, func)                                          \
  APPLY_FUNCTION2_8(x, y, func) func(x##9, y##9);
#define APPLY_FUNCTION2_10(x, y, func)                                         \
  APPLY_FUNCTION2_9(x, y, func) func(x##10, y##10);
#define APPLY_FUNCTION2(x, y, func) CAT(APPLY_FUNCTION2_, K)(x, y, func)

#define LIST_OF_ARGS_1(name) name##1
#define LIST_OF_ARGS_2(name) LIST_OF_ARGS_1(name), name##2
#define LIST_OF_ARGS_3(name) LIST_OF_ARGS_2(name), name##3
#define LIST_OF_ARGS_4(name) LIST_OF_ARGS_3(name), name##4
#define LIST_OF_ARGS_5(name) LIST_OF_ARGS_4(name), name##5
#define LIST_OF_ARGS_6(name) LIST_OF_ARGS_5(name), name##6
#define LIST_OF_ARGS_7(name) LIST_OF_ARGS_6(name), name##7
#define LIST_OF_ARGS_8(name) LIST_OF_ARGS_7(name), name##8
#define LIST_OF_ARGS_9(name) LIST_OF_ARGS_8(name), name##9
#define LIST_OF_ARGS_10(name) LIST_OF_ARGS_9(name), name##10
#define LIST_OF_ARGS(name) CAT(LIST_OF_ARGS_, K)(name)

#define EMPTY()

#define DECLARE_LIST_OF_ARGS(name) CAT(LIST_OF_ARGS_, K)(TYPE* name)

// Функция проверки ошибок CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static void fill_array(TYPE* a, int size)
{
	for (int i = 0; i < size; ++i)
	{
		a[i] = (TYPE)1;
	}
}

static void print_vector(const char* label, TYPE* vec, int size)
{
	assert(size >= 4);
	printf("%s : [" TYPE_FORMAT ", " TYPE_FORMAT "... " TYPE_FORMAT ", " TYPE_FORMAT "]\n",
		label, vec[0], vec[1], vec[size - 2], vec[size - 1]);
}

#endif