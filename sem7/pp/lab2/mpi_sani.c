#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define VEC_COUNT 3
#define VEC_SIZE 6300000
#define Q 26

int main(int argc, char *argv[]) {

  double **a = (double **)malloc(VEC_COUNT * sizeof(double *));
  for (int v = 0; v < VEC_COUNT; v++) {
    a[v] = (double *)malloc(VEC_SIZE * sizeof(double));
  }

  double TotalSum, ProcSum = 0;
  int ProcRank, ProcNum;
  MPI_Status Status;
  double ts = 0, tB = 0, tpp = 0, tR = 0;
  int i, j, v;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  if (ProcRank == 0) {
    printf("Number of vectors -> %d\n", VEC_COUNT);
    printf("Vector dimension -> %d\n", VEC_SIZE);
    printf("Number of repetitions -> %d\n", Q);
    printf("-------------------MPI-------------------\n");

    for (v = 0; v < VEC_COUNT; v++) {
      for (i = 0; i < VEC_SIZE; i++) {
        a[v][i] = sin(i + v);
      }
    }
  }

  for (j = 0; j < Q; j++) {

    // Broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    double st_time = MPI_Wtime();
    for (v = 0; v < VEC_COUNT; v++) {
      MPI_Bcast(a[v], VEC_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    tB += MPI_Wtime() - st_time;

    // Sequential
    if (ProcRank == 0) {
      double seq_sum = 0;
      st_time = MPI_Wtime();
      for (v = 0; v < VEC_COUNT; v++) {
        for (i = 0; i < VEC_SIZE; i++) {
          seq_sum += a[v][i];
        }
      }
      ts += MPI_Wtime() - st_time;
      if (j == Q - 1) {
        printf("\n-------------------Sum-------------------\n");
        printf("Sequential Sum = %f\n", seq_sum);
      }
    }

    // Point-to-point
    ProcSum = 0;
    int k = VEC_SIZE / ProcNum;
    int i1 = k * ProcRank;
    int i2 = k * (ProcRank + 1);
    if (ProcRank == ProcNum - 1)
      i2 = VEC_SIZE;

    st_time = MPI_Wtime();
    for (v = 0; v < VEC_COUNT; v++) {
      for (i = i1; i < i2; i++) {
        ProcSum += a[v][i];
      }
    }

    if (ProcRank == 0) {
      TotalSum = ProcSum;
      for (i = 1; i < ProcNum; i++) {
        MPI_Recv(&ProcSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
        TotalSum += ProcSum;
      }
      tpp += MPI_Wtime() - st_time;
      if (j == Q - 1)
        printf("Point-to-point Total Sum = %f\n", TotalSum);
    } else {
      MPI_Send(&ProcSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Collective
    ProcSum = 0;
    for (v = 0; v < VEC_COUNT; v++) {
      for (i = i1; i < i2; i++) {
        ProcSum += a[v][i];
      }
    }

    st_time = MPI_Wtime();
    MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    tR += MPI_Wtime() - st_time;

    if (ProcRank == 0) {
      if (j == Q - 1)
        printf("Collective Total Sum = %f\n", TotalSum);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (ProcRank == 0) {
    ts /= Q;
    tpp /= Q;
    tR /= Q;
    tB /= Q;

    double appB = ts / (tpp + tB);
    double aRB = ts / (tR + tB);
    double app = ts / tpp;
    double aR = ts / tR;

    printf("\n-------------------Time-------------------\n");
    printf("Broadcast Time -> %f\n", tB);
    printf("Sequential Time -> %f\n", ts);
    printf("Point-to-Point Time -> %f\n", tpp);
    printf("Collective Time -> %f\n", tR);

    printf("\n-------------------Speedup-------------------\n");
    printf("Point-to-Point Speedup (With Broadcast) -> %f\n", appB);
    printf("Collective Speedup (With Broadcast) -> %f\n", aRB);
    printf("Point-to-Point Speedup (without Broadcast) -> %f\n", app);
    printf("Collective Speedup (without Broadcast) -> %f\n", aR);
  }

  for (v = 0; v < VEC_COUNT; v++) {
    free(a[v]);
  }
  free(a);

  MPI_Finalize();
  return 0;
}