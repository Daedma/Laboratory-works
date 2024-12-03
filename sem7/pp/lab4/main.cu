#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "addKernel.h"

int main(int argc, char* argv[])
{
	// Вывод параметров индивидуального варианта
	printf("Type : %s\n", STR(TYPE));
	printf("K : %d\n", K);
	printf("(N/DIMDIV) : %d\n", (N/DIMDIV));
	printf("(GridDim, BlockDim) : (%d, %d)\n", GRID_SIZE, BLOCK_SIZE);

	DECLARE_VECTORS(a);
	TYPE* res;

	// Выделение памяти на хосте
#define MALLOC(x) x = (TYPE *)malloc((N/DIMDIV) * sizeof(TYPE))
	APPLY_FUNCTION(a, MALLOC);
	MALLOC(res);
#undef MALLOC

	// Инициализация массивов
#define FILL(x) fill_array(x, (N/DIMDIV))
	APPLY_FUNCTION(a, FILL);
#undef FILL

	DECLARE_VECTORS(adev);
	TYPE* resdev;

	// Выделение памяти на устройстве
#define CUDA_MALLOC(x) CHECK_CUDA(cudaMalloc((void**)&x, (N/DIMDIV) * sizeof(TYPE)))
	APPLY_FUNCTION(adev, CUDA_MALLOC);
	CUDA_MALLOC(resdev);
#undef CUDA_MALLOC

	// Создание обработчиков событий
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	CHECK_CUDA(cudaEventCreate(&start));

	CHECK_CUDA(cudaEventCreate(&stop));

	double ts = 0.0, ttr = 0.0, tcu = 0.0;

	for (int j = 0; j < 20; ++j)
	{
		// Последовательный алгоритм и замер его времени выполнения ts
		CHECK_CUDA(cudaEventRecord(start));
		for (int i = 0; i < (N/DIMDIV); ++i)
		{
			res[i] = APPLY_BIN_OP(a, i, +);
		}
		CHECK_CUDA(cudaEventRecord(stop));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		ts += gpuTime / 1000.0;

		if (j == 19)
		{
			print_vector("Sequentional sum", res, (N/DIMDIV));
		}

		// Замер времени передачи данных на видеокарту ttr
		CHECK_CUDA(cudaEventRecord(start));
#define CUDA_MEMCPY(dest, src) CHECK_CUDA(cudaMemcpy(dest, src, (N/DIMDIV) * sizeof(TYPE), cudaMemcpyHostToDevice))
		APPLY_FUNCTION2(adev, a, CUDA_MEMCPY);
#undef CUDA_MEMCPY
		CHECK_CUDA(cudaEventRecord(stop));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		ttr += gpuTime / 1000.0;

		// Установка точки старта
		CHECK_CUDA(cudaEventRecord(start));

		// Запуск ядра
		kernel << <GRID_SIZE, BLOCK_SIZE >> > (resdev, LIST_OF_ARGS(adev), (N/DIMDIV));

		CHECK_CUDA(cudaGetLastError());

		// Установка точки конца
		CHECK_CUDA(cudaEventRecord(stop));

		CHECK_CUDA(cudaEventSynchronize(stop));

		// Синхронизация устройств
		CHECK_CUDA(cudaDeviceSynchronize());

		// Расчет времени
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		tcu += gpuTime / 1000.0;

		// Копирование результата на хост
		CHECK_CUDA(cudaMemcpy(res, resdev, (N/DIMDIV) * sizeof(TYPE), cudaMemcpyDeviceToHost));
		if (j == 19)
		{
			print_vector("Parallel sum", res, (N/DIMDIV));
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
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));
	APPLY_FUNCTION(adev, cudaFree);
#define CUDA_FREE(x) CHECK_CUDA(cudaFree(x))
	CUDA_FREE(resdev);
#undef CUDA_FREE
	APPLY_FUNCTION(a, free);
	free(res);

	return 0;
}
