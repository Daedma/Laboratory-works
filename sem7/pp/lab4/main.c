#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"

int main(int argc, char* argv[])
{
	// Вывод параметров индивидуального варианта
	printf("Type : %s\n", STR(TYPE));
	printf("K : %d\n", K);
	printf("N : %d\n", N);
	printf("(GridDim, BlockDim) : (%d, %d)\n", GRID_SIZE, BLOCK_SIZE);

	DECLARE_VECTORS(a);
	TYPE* res;

	// Выделение памяти на хосте
#define MALLOC(x) x = (TYPE *)malloc(N * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
	MALLOC(res);
#undef MALLOC

	// Инициализация массивов
#define FILL(x) fill_array(x, N)
	APPLY_FUNCTION(a, FILL);
#undef FILL

	DECLARE_VECTORS(adev);
	TYPE* resdev;

	cudaError_t cuerr;

	// Выделение памяти на устройстве
#define CUDA_MALLOC(x) cuerr = cudaMalloc((void**)&x, N * sizeof(TYPE));\
			if (cuerr != cudaSuccess)\
			{\
				fprintf(stderr, "Cannot allocate device array for %s: %s\n",\
					STR(x), cudaGetErrorString(cuerr));\
				return EXIT_FAILURE;\
			}
	APPLY_FUNCTION(adev, CUDA_MALLOC);
	CUDA_MALLOC(resdev);
#undef CUDA_MALLOC

	// Создание обработчиков событий
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cuerr = cudaEventCreate(&start);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA start event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaEventCreate(&stop);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create CUDA end event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	double ts = 0.0, ttr = 0.0, tcu = 0.0;
	clock_t start_time, end_time;

	for (int j = 0; j < 20; ++j)
	{
		// Последовательный алгоритм и замер его времени выполнения ts
		start_time = clock();
		for (int i = 0; i < N; ++i)
		{
			res[i] = APPLY_BIN_OP(a, i, +);
		}
		end_time = clock();
		ts += (double)(end_time - start_time) / CLOCKS_PER_SEC;
		if (j == 19)
		{
			print_vector("Sequentional sum", res, N);
		}

		// Замер времени передачи данных на видеокарту ttr
		start_time = clock();
#define CUDA_MEMCPY(dest, src) cuerr = cudaMemcpy(dest, src, N * sizeof(TYPE), cudaMemcpyHostToDevice)
		APPLY_FUNCTION2(adev, a, CUDA_MEMCPY);
#undef CUDA_MEMCPY
		end_time = clock();
		ttr += (double)(end_time - start_time) / CLOCKS_PER_SEC;

		// Установка точки старта
		cuerr = cudaEventRecord(start, 0);
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot record CUDA event: %s\n",
				cudaGetErrorString(cuerr));
			return EXIT_FAILURE;
		}

		// Запуск ядра
		kernel << <GRID_SIZE, BLOCK_SIZE >> > (res, LIST_OF_ARGS(adev), N);

		cuerr = cudaGetLastError();
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
				cudaGetErrorString(cuerr));
			return EXIT_FAILURE;
		}

		// Синхронизация устройств
		cuerr = cudaDeviceSynchronize();
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
				cudaGetErrorString(cuerr));
			return EXIT_FAILURE;
		}

		// Установка точки окончания
		cuerr = cudaEventRecord(stop, 0);
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot record CUDA event: %s\n",
				cudaGetErrorString(cuerr));
			return EXIT_FAILURE;
		}

		// Расчет времени
		cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
		tcu += gpuTime / 1000.0;

		// Копирование результата на хост
		cuerr = cudaMemcpy(res, resdev, N * sizeof(TYPE), cudaMemcpyDeviceToHost);
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy res array from device to host: %s\n",
				cudaGetErrorString(cuerr));
			return EXIT_FAILURE;
		}
		if (j == 19)
		{
			print_vector("Parallel sum", res, N);
		}
	}

	print_vector("Parallel sum", res, N);

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
	APPLY_FUNCTION(adev, cudaFree);
	cudaFree(resdev);
	APPLY_FUNCTION(a, free);
	free(res);

	return 0;
}
