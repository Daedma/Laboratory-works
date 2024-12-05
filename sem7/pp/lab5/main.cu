#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 300
#endif

#if defined(TYPE_DOUBLE)
#define TYPE double
#define GEMMFUNC cublasDgemm
#define GENERATE_UNIFORM_FUNC curandGenerateUniformDouble
#elif defined(TYPE_FLOAT)
#define TYPE float
#define GEMMFUNC cublasSgemm
#define GENERATE_UNIFORM_FUNC curandGenerateUniform
#elif defined(TYPE_COMPLEX)
#define TYPE cuComplex
#define SUBTYPE float
#define GEMMFUNC cublasCgemm
#define GENERATE_UNIFORM_FUNC curandGenerateUniform
#define NUM_OF_DIM 2
#define ALPHA make_cuComplex(1, 0)
#define BETA make_cuComplex(0, 0)
#else
#define TYPE double
#define GEMMFUNC cublasDgemm
#define GENERATE_UNIFORM_FUNC curandGenerateUniformDouble
#endif

#ifndef TYPE_COMPLEX
#define SUBTYPE TYPE
#define NUM_OF_DIM 1
#define ALPHA (TYPE)1
#define BETA (TYPE)0
#endif

#ifndef TRANSPOSE_A
#define TRANSPOSE_A 0
#endif

#ifndef TRANSPOSE_B
#define TRANSPOSE_B 0
#endif

#if (TRANSPOSE_A == 1)
#define A_OP CUBLAS_OP_T
#define TRANSPOSE_A_INFO "T"
#else
#define A_OP CUBLAS_OP_N
#define TRANSPOSE_A_INFO "non-T"
#endif

#if (TRANSPOSE_B == 1)
#define B_OP CUBLAS_OP_T
#define TRANSPOSE_B_INFO "T"
#else
#define B_OP CUBLAS_OP_N
#define TRANSPOSE_B_INFO "non-T"
#endif

#define TRANSPOSE_INFO "(" TRANSPOSE_A_INFO ", " TRANSPOSE_B_INFO ")"

#define STR(x) STR_HELPER(x)
#define STR_HELPER(x) #x

#define PRINTABLE_ELEMENT_COUNT 6

// Функция проверки ошибок CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Функция случайной генерации матрицы на хосте
void CPU_fill_rand(TYPE* A, int nr_rows_A, int nr_cols_A)
{
	for (int i = 0; i < nr_rows_A * nr_cols_A * NUM_OF_DIM; ++i)
	{
		((SUBTYPE*)A)[i] = rand() / (SUBTYPE)RAND_MAX;
	}
}

// Функция случайной генерации матрицы на GPU
void GPU_fill_rand(TYPE* A, int nr_rows_A, int nr_cols_A)
{
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	GENERATE_UNIFORM_FUNC(prng, (SUBTYPE*)A, nr_rows_A * nr_cols_A * NUM_OF_DIM);
	curandDestroyGenerator(prng);
}

// Функция умножения матриц на GPU
void gpu_blas_mmul(const TYPE* A, const TYPE* B, TYPE* C, const int m,
	const int k, const int n)
{
	static const TYPE alpha = ALPHA;
	static const TYPE beta = BETA;
	int lda = m, ldb = k, ldc = m;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GEMMFUNC(handle, A_OP, B_OP, m, n, k, &alpha, A, lda, B, ldb,
		&beta, C, ldc);
	cublasDestroy(handle);
}

// Функция вывода матрицы
void print_matrix(TYPE* A, int nr_rows_A, int nr_cols_A)
{
	int prowc = nr_rows_A < PRINTABLE_ELEMENT_COUNT ? nr_rows_A : PRINTABLE_ELEMENT_COUNT;
	int pcolc = nr_cols_A < PRINTABLE_ELEMENT_COUNT ? nr_cols_A : PRINTABLE_ELEMENT_COUNT;
	for (int i = 0; i < prowc; ++i)
	{
		for (int j = 0; j < pcolc; ++j)
		{
#ifndef TYPE_COMPLEX
			printf("%.2f ", A[i * nr_cols_A + j]);
#else
			printf("(%.2f; %.2f) ", A[i * nr_cols_A + j].x, A[i * nr_cols_A + j].y);
#endif
		}
		printf("\n");
	}
}

int main()
{
	// Вывод параметров индивидуального варианта
	printf("Type : %s\n", STR(TYPE));
	printf("N : %d\n", N);
	printf(TRANSPOSE_INFO "\n");

	srand(time(0));

	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = N;

	TYPE* h_A = (TYPE*)malloc(nr_rows_A * nr_cols_A * sizeof(TYPE));
	TYPE* h_B = (TYPE*)malloc(nr_rows_B * nr_cols_B * sizeof(TYPE));
	TYPE* h_C = (TYPE*)malloc(nr_rows_C * nr_cols_C * sizeof(TYPE));

	TYPE* d_A, * d_B, * d_C;
	CHECK_CUDA(cudaMalloc((void**)&d_A, nr_rows_A * nr_cols_A * sizeof(TYPE)));
	CHECK_CUDA(cudaMalloc((void**)&d_B, nr_rows_B * nr_cols_B * sizeof(TYPE)));
	CHECK_CUDA(cudaMalloc((void**)&d_C, nr_rows_C * nr_cols_C * sizeof(TYPE)));

	double tMh = 0, tMd = 0, ts = 0, ttr = 0, tcu = 0;
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	for (int i = 0; i < 20; ++i)
	{
		// Создание матриц на хосте
		CHECK_CUDA(cudaEventRecord(start, 0));
		CPU_fill_rand(h_A, nr_rows_A, nr_cols_A);
		CPU_fill_rand(h_B, nr_rows_B, nr_cols_B);
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		tMh += gpuTime / 1000.0;

		// Создание матриц на девайсе
		CHECK_CUDA(cudaEventRecord(start, 0));
		GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
		GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		tMd += gpuTime / 1000.;

		// Последовательный алгоритм
		CHECK_CUDA(cudaEventRecord(start, 0));
		for (int i = 0; i < nr_rows_A; ++i)
		{
			for (int j = 0; j < nr_cols_B; ++j)
			{
#ifndef TYPE_COMPLEX
				h_C[i * nr_cols_B + j] = 0;
#else
				h_C[i * nr_cols_B + j].x = 0;
				h_C[i * nr_cols_B + j].y = 0;
#endif
				for (int k = 0; k < nr_cols_A; ++k)
				{
#ifndef TYPE_COMPLEX
					h_C[i * nr_cols_B + j] +=
						h_A[i * nr_cols_A + k] * h_B[k * nr_cols_B + j];
#else
					h_C[i * nr_cols_B + j] = cuCfmaf(h_A[i * nr_cols_A + k], h_B[k * nr_cols_B + j], h_C[i * nr_cols_B + j]);
#endif
				}
			}
		}
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		ts += gpuTime / 1000.;

		if(i == 19)
		{
			printf("Sequantional multiplication:\n");
			print_matrix(h_C, N, N);
		}

		// Передача данных на видеокарту
		CHECK_CUDA(cudaEventRecord(start, 0));
		CHECK_CUDA(cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(TYPE),
			cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(TYPE),
			cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		ttr += gpuTime / 1000.;

		// Умножение матриц на GPU
		CHECK_CUDA(cudaEventRecord(start, 0));
		gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaEventElapsedTime(&gpuTime, start, stop));
		tcu += gpuTime / 1000.;

		CHECK_CUDA(cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(TYPE),
			cudaMemcpyDeviceToHost));

		if(i == 19)
		{
			printf("CuBLAS multiplication:\n");
			print_matrix(h_C, N, N);
		}
	}

	tMh /= 20;
	tMd /= 20;
	ts /= 20;
	ttr /= 20;
	tcu /= 20;

	float acu = ts / tcu;
	float aMhcu = (tMh + ttr + tcu) / ts;
	float aMdcu = (tMd + tcu) / ts;

	printf("Matrix creation on host time : %f\n", tMh);
	printf("Matrix creation on device time: %f\n", tMd);
	printf("Sequantional time: %f\n", ts);
	printf("Transfer time: %f\n", ttr);
	printf("CuBLAS time: %f\n", tcu);
	printf("Speedup (without transfer): %f\n", acu);
	printf("Speedup (with creation on host and transfer): %f\n", aMhcu);
	printf("Speedup (with creation on device): %f\n", aMdcu);

	// Free GPU memory
	CHECK_CUDA(cudaFree(d_A));
	CHECK_CUDA(cudaFree(d_B));
	CHECK_CUDA(cudaFree(d_C));

	// Destroy CUDA events
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
