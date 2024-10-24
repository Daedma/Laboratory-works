#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(TYPE_DOUBLE)
#define TYPE double
#elif defined(TYPE_FLOAT)
#define TYPE float
#elif defined(TYPE_INT)
#define TYPE int
#else
#define TYPE double
#endif

#define STR(x) STR_HELPER(x)
#define STR_HELPER(x) #x

#ifndef GRID_SIZE
#define GRID_SIZE 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <number_of_elements>\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  printf("n = %d\n", n);

  int n2b = n * sizeof(int);
  int n2 = n;

  // Выделение памяти на хосте
  int *a = (int *)calloc(n2, sizeof(int));
  int *b = (int *)calloc(n2, sizeof(int));
  int *c = (int *)calloc(n2, sizeof(int));

  // Инициализация массивов
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i;
  }

  // Выделение памяти на устройстве
  int *adev = NULL;
  cudaError_t cuerr = cudaMalloc((void **)&adev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  int *bdev = NULL;
  cuerr = cudaMalloc((void **)&bdev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  int *cdev = NULL;
  cuerr = cudaMalloc((void **)&cdev, n2b);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  // Создание обработчиков событий
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;
  cuerr = cudaEventCreate(&start);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  cuerr = cudaEventCreate(&stop);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
    return 0;
  }

  double ts = 0.0, ttr = 0.0, tcu = 0.0;
  clock_t start_time, end_time;

  for (int i = 0; i < 20; ++i) {
    // Последовательный алгоритм и замер его времени выполнения ts
    start_time = clock();
    for (int i = 0; i < n; ++i) {
      c[i] = a[i] + b[i];
    }
    end_time = clock();
    ts += (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Замер времени передачи данных на видеокарту ttr
    start_time = clock();
    cuerr = cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
    cuerr = cudaMemcpy(bdev, b, n2b, cudaMemcpyHostToDevice);
    end_time = clock();
    ttr += (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
      fprintf(stderr, "Cannot record CUDA event: %s\n",
              cudaGetErrorString(cuerr));
      return 0;
    }

    // Запуск ядра
    kernel<<<GRID_SIZE, BLOCK_SIZE>>>(cdev, adev, bdev, n);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
      fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
              cudaGetErrorString(cuerr));
      return 0;
    }

    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
      fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
              cudaGetErrorString(cuerr));
      return 0;
    }

    // Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess) {
      fprintf(stderr, "Cannot copy c array from device to host: %s\n",
              cudaGetErrorString(cuerr));
      return 0;
    }

    // Расчет времени
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    tcu += gpuTime / 1000.0;

    // Копирование результата на хост
    cuerr = cudaMemcpy(c, cdev, n2b, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      fprintf(stderr, "Cannot copy c array from device to host: %s\n",
              cudaGetErrorString(cuerr));
      return 0;
    }
  }

  ts /= 20.0;
  ttr /= 20.0;
  tcu /= 20.0;

  double acu = ts / tcu;
  double acutr = ts / (ttr + tcu);

  printf("Average sequential time (ts): %.9f seconds\n", ts);
  printf("Average transfer time (ttr): %.9f seconds\n", ttr);
  printf("Average kernel execution time (tcu): %.9f seconds\n", tcu);
  printf("Speedup without transfer (acu): %.9f\n", acu);
  printf("Speedup with transfer (acutr): %.9f\n", acutr);

  // Очищение памяти
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(adev);
  cudaFree(bdev);
  cudaFree(cdev);
  free(a);
  free(b);
  free(c);

  return 0;
}
