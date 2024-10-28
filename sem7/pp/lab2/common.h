#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(TYPE_DOUBLE)
#define TYPE double
#define TYPE_FORMAT "%f"
#define MPI_TYPE MPI_DOUBLE
#elif defined(TYPE_FLOAT)
#define TYPE float
#define TYPE_FORMAT "%f"
#define MPI_TYPE MPI_FLOAT
#elif defined(TYPE_INT)
#define TYPE int
#define TYPE_FORMAT "%d"
#define MPI_TYPE MPI_INT
#else
#define TYPE double
#define MPI_TYPE MPI_DOUBLE
#define TYPE_FORMAT "%f"
#endif

#ifndef K
#define K 3
#endif

#ifndef N
#define N 6300000
#endif

#ifndef Q
#define Q 26
#endif

#ifndef THREADS_COUNT
#define THREADS_COUNT "[3, 9, 12]"
#endif

#ifdef SLOWER
#define REPEAT_Q_TIMES for (int _rqt = 0; _rqt < Q; ++_rqt)
#else
#define REPEAT_Q_TIMES
#endif

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define VECTORS_1(var) TYPE *var##1;
#define VECTORS_2(var) VECTORS_1(var) TYPE *var##2;
#define VECTORS_3(var) VECTORS_2(var) TYPE *var##3;
#define VECTORS_4(var) VECTORS_3(var) TYPE *var##4;
#define VECTORS_5(var) VECTORS_4(var) TYPE *var##5;
#define VECTORS_6(var) VECTORS_5(var) TYPE *var##6;
#define VECTORS_7(var) VECTORS_6(var) TYPE *var##7;
#define VECTORS_8(var) VECTORS_7(var) TYPE *var##8;
#define VECTORS_9(var) VECTORS_8(var) TYPE *var##9;
#define VECTORS_10(var) VECTORS_9(var) TYPE *var##10;
#define DECLARE_VECTORS(var) CAT(VECTORS_, K)(var)

#define APPLY_BIN_OP_1(var, index, op) var##1 [index]
#define APPLY_BIN_OP_2(var, index, op)                                         \
  (APPLY_BIN_OP_1(var, index, op) op var##2 [index])
#define APPLY_BIN_OP_3(var, index, op)                                         \
  (APPLY_BIN_OP_2(var, index, op) op var##3 [index])
#define APPLY_BIN_OP_4(var, index, op)                                         \
  (APPLY_BIN_OP_3(var, index, op) op var##4 [index])
#define APPLY_BIN_OP_5(var, index, op)                                         \
  (APPLY_BIN_OP_4(var, index, op) op var##5 [index])
#define APPLY_BIN_OP_6(var, index, op)                                         \
  (APPLY_BIN_OP_5(var, index, op) op var##6 [index])
#define APPLY_BIN_OP_7(var, index, op)                                         \
  (APPLY_BIN_OP_6(var, index, op) op var##7 [index])
#define APPLY_BIN_OP_8(var, index, op)                                         \
  (APPLY_BIN_OP_7(var, index, op) op var##8 [index])
#define APPLY_BIN_OP_9(var, index, op)                                         \
  (APPLY_BIN_OP_8(var, index, op) op var##9 [index])
#define APPLY_BIN_OP_10(var, index, op)                                        \
  (APPLY_BIN_OP_9(var, index, op) op var##10 [index])
#define APPLY_BIN_OP(var, index, op) CAT(APPLY_BIN_OP_, K)(var, index, op)

#define APPLY_FUNCTION_1(var, func) (func(var##1));
#define APPLY_FUNCTION_2(var, func) APPLY_FUNCTION_1(var, func)(func(var##2));
#define APPLY_FUNCTION_3(var, func) APPLY_FUNCTION_2(var, func)(func(var##3));
#define APPLY_FUNCTION_4(var, func) APPLY_FUNCTION_3(var, func)(func(var##4));
#define APPLY_FUNCTION_5(var, func) APPLY_FUNCTION_4(var, func)(func(var##5));
#define APPLY_FUNCTION_6(var, func) APPLY_FUNCTION_5(var, func)(func(var##6));
#define APPLY_FUNCTION_7(var, func) APPLY_FUNCTION_6(var, func)(func(var##7));
#define APPLY_FUNCTION_8(var, func) APPLY_FUNCTION_7(var, func)(func(var##8));
#define APPLY_FUNCTION_9(var, func) APPLY_FUNCTION_8(var, func)(func(var##9));
#define APPLY_FUNCTION_10(var, func) APPLY_FUNCTION_9(var, func)(func(var##10));
#define APPLY_FUNCTION(var, func) CAT(APPLY_FUNCTION_, K)(var, func)

#define STR(x) STR_HELPER(x)
#define STR_HELPER(x) #x

static void fill_array(TYPE* a, int size)
{
	for (int i = 0; i < size; ++i)
	{
		a[i] = (TYPE)1;
	}
}

#endif