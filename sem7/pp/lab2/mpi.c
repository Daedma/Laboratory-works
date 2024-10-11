#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define TYPE double
#define K 3
#define N 6300000
#define Q 26

void fill_array(TYPE *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (TYPE)rand() / RAND_MAX;
  }
}

TYPE sequential_sum(TYPE *a, int size) {
  TYPE sum = 0;
  for (int i = 0; i < size; i++) {
    sum += a[i];
  }
  return sum;
}

int main(int argc, char *argv[]) {
  // Инициализация MPI
  MPI_Init(&argc, &argv);
  int proc_rank, proc_num;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // Вывод параметров индивидуального варианта
  if (proc_rank == 0) {
    printf("Type: %s\n", "double");
    printf("K: %d\n", K);
    printf("N: %d\n", N);
    printf("Threads count: %d\n", proc_num);
    printf("Q: %d\n", Q);
  }

  // Создание массивов
  TYPE *a[K];
  for (int i = 0; i != K; ++i) {
    a[i] = (TYPE *)malloc(N * sizeof(TYPE));
    if (proc_rank == 0) {
      fill_array(a[i], N);
    }
  }

  // Замер времени последовательного алгоритма
  double ts = 0;
  if (proc_rank == 0) {
    for (int i = 0; i != 20; ++i) {
      double st_time = MPI_Wtime();
      for (int j = 0; j != K; ++j) {
        sequential_sum(a[j], N);
      }
      double end_time = MPI_Wtime();
      ts += end_time - st_time;
    }
    ts /= 20;
  }

  // Широковещательная рассылка и замер времени ее выполнения
  double tB = 0;
  for (int i = 0; i != 20; ++i) {
    double st_time = MPI_Wtime();
    for (int j = 0; j != K; ++j) {
      MPI_Bcast(a[j], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    double end_time = MPI_Wtime();
    tB += end_time - st_time;
  }
  tB /= 20;

  // Замер времени параллельных алгоритмов
  double tpp = 0, tR = 0;
  for (int i = 0; i != 20; ++i) {
    // Параллельный алгоритм с использованием операций "точка-точка"
    double st_time = MPI_Wtime();
    TYPE proc_sum = 0.0;
    int k = N / proc_num;
    int i1 = k * proc_rank;
    int i2 = k * (proc_rank + 1);
    if (proc_rank == proc_num - 1)
      i2 = N;
    for (int j = 0; j != K; ++j) {
      for (int l = i1; l != i2; ++l) {
        proc_sum += a[j][l];
      }
    }
    TYPE total_sum = 0.0;
    if (proc_rank == 0) {
      total_sum = proc_sum;
      for (int j = 1; j != proc_num; ++j) {
        MPI_Recv(&proc_sum, 1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        total_sum += proc_sum;
      }
    } else {
      MPI_Send(&proc_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    tpp += end_time - st_time;

    // Параллельный алгоритм с использованием коллективной операции
    st_time = MPI_Wtime();
    proc_sum = 0.0;
    for (int j = 0; j != K; ++j) {
      for (int l = i1; l != i2; ++l) {
        proc_sum += a[j][l];
      }
    }
    MPI_Reduce(&proc_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    tR += end_time - st_time;
  }
  tpp /= 20;
  tR /= 20;

  // Вычисление ускорения
  double aRB = ts / (tB + tpp);
  double appB = ts / (tB + tR);
  double aR = ts / tpp;
  double ap = ts / tR;

  // Вывод полученных значений времени и ускорения
  if (proc_rank == 0) {
    printf("Sequential time: %.17f\n", ts);
    printf("Broadcast time: %.17f\n", tB);
    printf("Parallel time (point-to-point): %.17f\n", tpp);
    printf("Parallel time (reduce): %.17f\n", tR);
    printf("Speedup (point-to-point with broadcast): %.17f\n", aRB);
    printf("Speedup (reduce with broadcast): %.17f\n", appB);
    printf("Speedup (point-to-point without broadcast): %.17f\n", aR);
    printf("Speedup (reduce without broadcast): %.17f\n", ap);
  }

  // Освобождение памяти
  for (int i = 0; i < K; i++) {
    free(a[i]);
  }

  // Завершение MPI
  MPI_Finalize();
  return 0;
}
