#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(TYPE_DOUBLE)
#define TYPE double
#elif defined(TYPE_FLOAT)
#define TYPE float
#elif defined(TYPE_INT)
#define TYPE int
#else
#define TYPE double
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

void fill_array(TYPE *a, int size) {
  for (int i = 0; i < size; ++i) {
    a[i] = (TYPE)1;
  }
}

int main(int argc, char *argv[]) {
  // if (argc < 2) {
  //   printf("Usage: %s <number_of_threads>", argv[0]);
  //   return -1;
  // }

  // int nthreads = atoi(argv[1]);
  // omp_set_num_threads(nthreads);

  // Вывод параметров индивидуального варианта
  printf("Type: %s\n", STR(TYPE));
  printf("K: %d\n", K);
  printf("N: %d\n", N);
  printf("Threads count: %s\n", THREADS_COUNT);
  printf("Q: %d\n", Q);

  // Создание массивов
  DECLARE_VECTORS(a);
#define MALLOC(var) var = (TYPE *)malloc(N * sizeof(TYPE))
  APPLY_FUNCTION(a, MALLOC);
#define FILL(var) fill_array(var, N)
  APPLY_FUNCTION(a, FILL);

  // Время
  double ts = 0; // последовательный алгоритм
  double tp = 0; // инициализация параллельной области
  double tc = 0, ta = 0, tr = 0; // параллельные алгоритмы

  for (int i = 0; i < 20; ++i) // внешний цикл для 20-ти повторений
  {
    TYPE sum;
    int i;
    double st_time, end_time;

    // Последовательный алгоритм
    sum = 0;
    st_time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
      REPEAT_Q_TIMES
      sum = sum + APPLY_BIN_OP(a, i, +);
    }
    end_time = omp_get_wtime();
    ts += end_time - st_time;

    // Инициализация параллельной области
    st_time = omp_get_wtime();
#pragma omp parallel
    {
      // Пустая параллельная область для замера времени инициализации
    }
    end_time = omp_get_wtime();
    tp += end_time - st_time;

    // critical
    sum = 0;
    st_time = omp_get_wtime();
#pragma omp parallel for
    for (i = 0; i < N; ++i) {
      REPEAT_Q_TIMES
#pragma omp critical
      {
        sum = sum + APPLY_BIN_OP(a, i, +);
      }
    }
    end_time = omp_get_wtime();
    tc += end_time - st_time;

    // atomic
    sum = 0;
    st_time = omp_get_wtime();
#pragma omp parallel for
    for (i = 0; i < N; ++i) {
      REPEAT_Q_TIMES
#pragma omp atomic
      sum = sum + APPLY_BIN_OP(a, i, +);
    }
    end_time = omp_get_wtime();
    ta += end_time - st_time;

    // reduction
    sum = 0;
    st_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < N; ++i) {
      REPEAT_Q_TIMES
      sum = sum + APPLY_BIN_OP(a, i, +);
    }
    end_time = omp_get_wtime();
    tr += end_time - st_time;
  }
  // Расчет ускорения
  ts /= 20;
  tp /= 20;
  tc /= 20;
  ta /= 20;
  tr /= 20;

  // Вычисление ускорения
  double acp = ts / (tp + tc);
  double aap = ts / (tp + ta);
  double arp = ts / (tp + tr);
  double ac = ts / tc;
  double aa = ts / ta;
  double ar = ts / tr;

  // Вывод полученных значений времени и ускорения
  printf("Sequential time: %f\n", ts);
  printf("Parallel initialization time: %f\n", tp);
  printf("Parallel time (critical): %f\n", tc);
  printf("Parallel time (atomic): %f\n", ta);
  printf("Parallel time (reduction): %f\n", tr);
  printf("Speedup (critical with init): %f\n", acp);
  printf("Speedup (atomic with init): %f\n", aap);
  printf("Speedup (reduction with init): %f\n", arp);
  printf("Speedup (critical without init): %f\n", ac);
  printf("Speedup (atomic without init): %f\n", aa);
  printf("Speedup (reduction without init): %f\n", ar);

  // Освобождение памяти
  APPLY_FUNCTION(a, free);

  return 0;
}
