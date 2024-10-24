#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция случайной генерации матрицы на хосте
void CPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  for (int i = 0; i < nr_rows_A * nr_cols_A; ++i) {
    A[i] = rand() / (float)RAND_MAX;
  }
}

// Функция случайной генерации матрицы на GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
  curandDestroyGenerator(prng);
}

// Функция умножения матриц на GPU
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m,
                   const int k, const int n) {
  int lda = m, ldb = k, ldc = m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc);
  cublasDestroy(handle);
}

// Функция вывода матрицы
void print_matrix(float *A, int nr_rows_A, int nr_cols_A) {
  for (int i = 0; i < nr_rows_A; ++i) {
    for (int j = 0; j < nr_cols_A; ++j) {
      printf("%f ", A[i * nr_cols_A + j]);
    }
    printf("\n");
  }
}

int main() {
  srand(time(0));

  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
  nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;

  float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
  float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
  float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

  float tMh = 0, tMd = 0, ts = 0, ttr = 0, tcu = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 20; ++i) {
    // Создание матриц на хосте
    cudaEventRecord(start, 0);
    CPU_fill_rand(h_A, nr_rows_A, nr_cols_A);
    CPU_fill_rand(h_B, nr_rows_B, nr_cols_B);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float tMh_temp;
    cudaEventElapsedTime(&tMh_temp, start, stop);
    tMh += tMh_temp;

    // Создание матриц на девайсе
    cudaEventRecord(start, 0);
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float tMd_temp;
    cudaEventElapsedTime(&tMd_temp, start, stop);
    tMd += tMd_temp;

    // Последовательный алгоритм
    cudaEventRecord(start, 0);
    for (int i = 0; i < nr_rows_A; ++i) {
      for (int j = 0; j < nr_cols_B; ++j) {
        h_C[i * nr_cols_B + j] = 0;
        for (int k = 0; k < nr_cols_A; ++k) {
          h_C[i * nr_cols_B + j] +=
              h_A[i * nr_cols_A + k] * h_B[k * nr_cols_B + j];
        }
      }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ts_temp;
    cudaEventElapsedTime(&ts_temp, start, stop);
    ts += ts_temp;

    // Передача данных на видеокарту
    cudaEventRecord(start, 0);
    cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ttr_temp;
    cudaEventElapsedTime(&ttr_temp, start, stop);
    ttr += ttr_temp;

    // Умножение матриц на GPU
    cudaEventRecord(start, 0);
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float tcu_temp;
    cudaEventElapsedTime(&tcu_temp, start, stop);
    tcu += tcu_temp;
  }

  tMh /= 20;
  tMd /= 20;
  ts /= 20;
  ttr /= 20;
  tcu /= 20;

  float acu = ts / tcu;
  float aMhcu = (tMh + ttr + tcu) / ts;
  float aMdcu = (tMd + tcu) / ts;

  printf("Average times:\n");
  printf("tMh: %f ms\n", tMh);
  printf("tMd: %f ms\n", tMd);
  printf("ts: %f ms\n", ts);
  printf("ttr: %f ms\n", ttr);
  printf("tcu: %f ms\n", tcu);
  printf("Accelerations:\n");
  printf("acu: %f\n", acu);
  printf("aMhcu: %f\n", aMhcu);
  printf("aMdcu: %f\n", aMdcu);

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
